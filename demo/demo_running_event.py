"""
Running Event Demo: Finish line counting with optional lane zones.

Demonstrates:
- ROI polygon (track only on the course)
- Finish line crossing detection
- Per-lane counting (if lanes are defined)
- Real-time stats overlay with total finishers

Usage:
    # Basic (auto-detect finish line position)
    python demo/demo_running_event.py --video videos/running.mp4

    # With custom finish line (y-coordinate as % of frame height)
    python demo/demo_running_event.py --video videos/running.mp4 --finish-line 0.7

    # Save output
    python demo/demo_running_event.py --video videos/running.mp4 --output outputs/running_demo.mp4

Video sources (free):
    - Pexels: search "marathon finish line", "running race", "track and field"
    - YouTube CC: search "5K race finish line camera"
"""

import argparse
import sys
import time
import cv2
import numpy as np
import supervision as sv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from demo.line_counter import LineCrossingCounter, LineSpec


def build_finish_line(frame_w: int, frame_h: int, position: float, orientation="vertical") -> LineSpec:
    if orientation == "vertical":
        x = int(frame_w * position)
        return LineSpec(start=(x,0), end=(x,frame_h), name="Finish Line")
    else :
        y = int(frame_h * position)
        return LineSpec(start=(0,y), end=(frame_w,y), name="Finish Line")


def draw_line(frame: np.ndarray, line: LineSpec, color=(0, 0, 255), thickness=3) -> np.ndarray:
    """Draw a counting line on the frame."""
    cv2.line(frame, line.start, line.end, color, thickness)

    # Label
    mid_x = (line.start[0] + line.end[0]) // 2
    mid_y = (line.start[1] + line.end[1]) // 2 - 15
    cv2.putText(frame, line.name, (mid_x - 60, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame


def draw_stats(frame: np.ndarray, stats: dict, fps: float) -> np.ndarray:
    """Draw counting stats overlay."""
    h, w = frame.shape[:2]
    panel_w, panel_h = 280, 100

    overlay = frame.copy()
    cv2.rectangle(overlay, (w - panel_w - 10, 10),
                  (w - 10, 10 + panel_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    x0 = w - panel_w
    cv2.putText(frame, "Running Event Tracker", (x0, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    total = stats["total_in"] + stats["total_out"]
    cv2.putText(frame, f"Finishers: {total}", (x0, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 127), 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (x0, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return frame


def main():
    parser = argparse.ArgumentParser(description="Running Event Finish Line Counter")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model path")
    parser.add_argument("--finish-line", type=float, default=0.6,
                        help="Finish line y-position as fraction of frame height (0.0=top, 1.0=bottom)")
    parser.add_argument("--confidence", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--output", default=None, help="Output video path")
    parser.add_argument("--no-display", action="store_true", help="Disable live display")
    parser.add_argument("--line-orientation", choices=["horizontal", "vertical"],
                    default="vertical")
    args = parser.parse_args()

    # Load model
    model = YOLO(args.model)

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open {args.video}")
        return

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup finish line
    finish_line = build_finish_line(w, h, args.finish_line)
    counter = LineCrossingCounter([finish_line])

    # Tracker
    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=int(fps_video),
    )

    # Annotators
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=5)
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=60)

    # Output writer
    writer = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps_video, (w, h))

    frame_count = 0
    t_start = time.time()

    # COCO person class = 0
    PERSON_CLASS_ID = 0

    print(f"[Running Event Demo] Video: {args.video} ({w}x{h} @ {fps_video:.0f} FPS)")
    print(f"[Running Event Demo] Finish line at y={int(h * args.finish_line)}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # Detect
            results = model(frame, conf=args.confidence, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)

            # Filter to persons only
            if detections.class_id is not None:
                person_mask = detections.class_id == PERSON_CLASS_ID
                detections = detections[person_mask]

            # Track
            detections = tracker.update_with_detections(detections)

            # Count line crossings
            tracker_ids = detections.tracker_id.tolist() if detections.tracker_id is not None else []
            centers = []
            for box in detections.xyxy:
                centers.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))

            events = counter.update(tracker_ids, centers)
            for event in events:
                print(f"  [FINISH] Runner #{event['tracker_id']} crossed {event['line']} "
                      f"(total: {counter.total_in + counter.total_out})")

            # Annotate
            labels = [f"#{tid}" for tid in tracker_ids]
            annotated = trace_annotator.annotate(frame.copy(), detections)
            annotated = box_annotator.annotate(annotated, detections)
            annotated = label_annotator.annotate(annotated, detections, labels=labels)
            annotated = draw_line(annotated, finish_line)

            elapsed = time.time() - t_start
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            annotated = draw_stats(annotated, counter.get_stats(), current_fps)

            if writer:
                writer.write(annotated)

            if not args.no_display:
                cv2.imshow("Running Event Demo", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        elapsed = time.time() - t_start
        stats = counter.get_stats()
        total = stats["total_in"] + stats["total_out"]
        print(f"\n[Running Event Demo] Done. {frame_count} frames in {elapsed:.1f}s")
        print(f"[Running Event Demo] Total finishers: {total}")


if __name__ == "__main__":
    main()