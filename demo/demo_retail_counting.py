"""
Retail People Counting Demo: Entry/exit counting with zone-based analytics.

Demonstrates:
- Polygonal ROI (store entrance area)
- Entry/exit line crossing (doorway)
- Zone occupancy tracking (how many people currently inside)
- Real-time stats overlay

Usage:
    # Basic
    python demo/demo_retail_counting.py --video videos/retail.mp4

    # With custom entry line position
    python demo/demo_retail_counting.py --video videos/retail.mp4 --entry-line 0.5

    # Horizontal line (default) or vertical line
    python demo/demo_retail_counting.py --video videos/retail.mp4 --line-orientation vertical

    # Save output
    python demo/demo_retail_counting.py --video videos/retail.mp4 --output outputs/retail_demo.mp4

Video sources (free):
    - Pexels: search "people walking entrance", "mall entrance", "store entrance overhead"
    - MOT benchmark: MOT17 pedestrian sequences
    - YouTube CC: search "pedestrian counting camera overhead"
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


def build_entry_line(
    frame_w: int,
    frame_h: int,
    position: float,
    orientation: str = "horizontal",
) -> LineSpec:
    """Create an entry/exit counting line."""
    if orientation == "horizontal":
        y = int(frame_h * position)
        return LineSpec(start=(0, y), end=(frame_w, y), name="Entrance")
    else:
        x = int(frame_w * position)
        return LineSpec(start=(x, 0), end=(x, frame_h), name="Entrance")


def draw_line(frame: np.ndarray, line: LineSpec, color=(0, 165, 255), thickness=3) -> np.ndarray:
    """Draw the entry/exit line on the frame."""
    cv2.line(frame, line.start, line.end, color, thickness)

    mid_x = (line.start[0] + line.end[0]) // 2
    mid_y = (line.start[1] + line.end[1]) // 2 - 15
    cv2.putText(frame, line.name, (mid_x - 40, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Direction arrows
    if line.start[1] == line.end[1]:  # horizontal
        y = line.start[1]
        cv2.putText(frame, "IN", (20, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "OUT", (20, y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:  # vertical
        x = line.start[0]
        cv2.putText(frame, "IN", (x + 15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "OUT", (x - 60, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame


def draw_stats(frame: np.ndarray, stats: dict, fps: float) -> np.ndarray:
    """Draw retail counting stats overlay."""
    h, w = frame.shape[:2]
    panel_w, panel_h = 260, 150

    overlay = frame.copy()
    cv2.rectangle(overlay, (w - panel_w - 10, 10),
                  (w - 10, 10 + panel_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    x0 = w - panel_w
    y = 35

    cv2.putText(frame, "Retail Counter", (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    y += 30
    entered = stats["total_in"]
    exited = stats["total_out"]
    inside = entered - exited

    cv2.putText(frame, f"Entered: {entered}", (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    y += 28
    cv2.putText(frame, f"Exited:  {exited}", (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    y += 28
    cv2.putText(frame, f"Inside:  {max(0, inside)}", (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    y += 28
    cv2.putText(frame, f"FPS: {fps:.1f}", (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return frame


def main():
    parser = argparse.ArgumentParser(description="Retail People Entry/Exit Counter")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model path")
    parser.add_argument("--entry-line", type=float, default=0.5,
                        help="Entry line position as fraction (0.0-1.0)")
    parser.add_argument("--line-orientation", choices=["horizontal", "vertical"],
                        default="horizontal", help="Line orientation")
    parser.add_argument("--confidence", type=float, default=0.3, help="Detection confidence")
    parser.add_argument("--output", default=None, help="Output video path")
    parser.add_argument("--no-display", action="store_true", help="Disable live display")
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

    # Setup entry line
    entry_line = build_entry_line(w, h, args.entry_line, args.line_orientation)
    counter = LineCrossingCounter([entry_line])

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
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=40)

    # Output writer
    writer = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps_video, (w, h))

    frame_count = 0
    t_start = time.time()

    PERSON_CLASS_ID = 0

    print(f"[Retail Demo] Video: {args.video} ({w}x{h} @ {fps_video:.0f} FPS)")
    print(f"[Retail Demo] Entry line: {args.line_orientation} at {args.entry_line}")

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
                direction = "ENTERED" if event["direction"] == "in" else "EXITED"
                print(f"  [{direction}] Person #{event['tracker_id']} "
                      f"(in: {counter.total_in}, out: {counter.total_out})")

            # Annotate
            labels = [f"#{tid}" for tid in tracker_ids]
            annotated = trace_annotator.annotate(frame.copy(), detections)
            annotated = box_annotator.annotate(annotated, detections)
            annotated = label_annotator.annotate(annotated, detections, labels=labels)
            annotated = draw_line(annotated, entry_line)

            elapsed = time.time() - t_start
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            annotated = draw_stats(annotated, counter.get_stats(), current_fps)

            if writer:
                writer.write(annotated)

            if not args.no_display:
                cv2.imshow("Retail People Counter", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        elapsed = time.time() - t_start
        print(f"\n[Retail Demo] Done. {frame_count} frames in {elapsed:.1f}s")
        print(f"[Retail Demo] Entered: {counter.total_in} | Exited: {counter.total_out} "
              f"| Inside: {max(0, counter.total_in - counter.total_out)}")


if __name__ == "__main__":
    main()