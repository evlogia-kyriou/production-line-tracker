"""
Comparison Runner: Runs multiple trackers on the same video and collects metrics.

Usage:
    # Run all 4 trackers
    python comparison/run_comparison.py \
        --video videos/running.mp4 \
        --trackers bytetrack boosttrack tracktrack fasttracker \
        --output outputs/comparisons/running/

    # Run only the two that work out of the box
    python comparison/run_comparison.py \
        --video videos/running.mp4 \
        --trackers bytetrack boosttrack \
        --output outputs/comparisons/running/

    # With a finish line for counting
    python comparison/run_comparison.py \
        --video videos/running.mp4 \
        --trackers bytetrack boosttrack \
        --output outputs/comparisons/running/ \
        --count-line 0.6

    # Vertical finish line at 90% of frame width
    python comparison/run_comparison.py \
        --video videos/running.mp4 \
        --trackers bytetrack boosttrack \
        --output outputs/comparisons/running/ \
        --count-line 0.9 --line-orientation vertical
"""

import argparse
import csv
import json
import sys
import time
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from comparison.adapters import TRACKER_REGISTRY, TrackedDetection
from demo.line_counter import LineCrossingCounter, LineSpec


def load_tracker(tracker_name: str, config_dir: str):
    """Load a tracker adapter from its config file."""
    config_path = Path(config_dir) / f"{tracker_name}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    adapter_cls = TRACKER_REGISTRY.get(tracker_name)
    if adapter_cls is None:
        raise ValueError(f"Unknown tracker: {tracker_name}. "
                         f"Available: {list(TRACKER_REGISTRY.keys())}")

    adapter = adapter_cls()
    adapter.load(config)
    return adapter


def run_single_tracker(
    adapter,
    video_path: str,
    output_video: str,
    count_line_pos: float | None = None,
    line_orientation: str = "horizontal",
) -> dict:
    """Run one tracker on a video and collect metrics.

    Returns:
        Dictionary with metrics: name, detector, total_frames, avg_fps,
        id_count, id_switches_estimate, line_count
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps_video, (w, h))

    # Optional line counting via LineCrossingCounter
    counter = None
    line_spec = None
    if count_line_pos is not None:
        if line_orientation == "vertical":
            x = int(w * count_line_pos)
            line_spec = LineSpec(start=(x, 0), end=(x, h), name="Count Line")
        else:
            y = int(h * count_line_pos)
            line_spec = LineSpec(start=(0, y), end=(w, y), name="Count Line")
        counter = LineCrossingCounter([line_spec])

    # Metrics tracking
    frame_count = 0
    total_time = 0.0
    all_ids_seen = set()
    id_per_frame = []

    print(f"  Processing with {adapter.name} ({adapter.detector_name})...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        annotated, detections, elapsed = adapter.benchmark_frame(frame)
        total_time += elapsed

        # Collect IDs
        frame_ids = set()
        tracker_ids = []
        centers = []
        for det in detections:
            all_ids_seen.add(det.tracker_id)
            frame_ids.add(det.tracker_id)
            tracker_ids.append(det.tracker_id)
            centers.append(det.center)

        # Line crossing count
        if counter is not None:
            events = counter.update(tracker_ids, centers)
            for event in events:
                print(f"    [CROSSED] #{event['tracker_id']} -> {event['direction']}")

        id_per_frame.append(frame_ids)

        # Draw tracker name + stats on frame
        cv2.putText(annotated, f"{adapter.name} ({adapter.detector_name})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        avg_fps = frame_count / total_time if total_time > 0 else 0
        cv2.putText(annotated, f"FPS: {avg_fps:.1f} | IDs: {len(all_ids_seen)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if counter is not None:
            line_total = counter.total_in + counter.total_out
            cv2.line(annotated, line_spec.start, line_spec.end, (0, 0, 255), 2)
            cv2.putText(annotated, f"Count: {line_total}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        writer.write(annotated)

    cap.release()
    writer.release()

    # Estimate ID switches (simplified: count IDs that disappear and reappear)
    id_switches_est = max(0, len(all_ids_seen) - estimate_unique_objects(id_per_frame))

    avg_fps = frame_count / total_time if total_time > 0 else 0
    line_count = (counter.total_in + counter.total_out) if counter else 0

    metrics = {
        "tracker": adapter.name,
        "detector": adapter.detector_name,
        "total_frames": frame_count,
        "avg_fps": round(avg_fps, 1),
        "unique_ids": len(all_ids_seen),
        "id_switches_estimate": id_switches_est,
        "line_count": line_count,
        "line_in": counter.total_in if counter else 0,
        "line_out": counter.total_out if counter else 0,
        "total_time_seconds": round(total_time, 2),
    }

    print(f"  Done: {frame_count} frames, {avg_fps:.1f} FPS, "
          f"{len(all_ids_seen)} unique IDs, count={line_count}")

    return metrics


def estimate_unique_objects(id_per_frame: list[set]) -> int:
    """Rough estimate of how many real unique objects existed.

    Counts max concurrent IDs across all frames as a proxy.
    """
    if not id_per_frame:
        return 0
    return max(len(ids) for ids in id_per_frame)


def save_metrics_csv(all_metrics: list[dict], output_path: str):
    """Save comparison metrics to CSV."""
    if not all_metrics:
        return

    keys = all_metrics[0].keys()
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nMetrics saved to {output_path}")


def print_comparison_table(all_metrics: list[dict]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 85)
    print(f"{'Tracker':<18} {'Detector':<12} {'FPS':>8} {'IDs':>8} {'ID Sw.':>8} {'Count':>8}")
    print("-" * 85)
    for m in all_metrics:
        print(f"{m['tracker']:<18} {m['detector']:<12} {m['avg_fps']:>8.1f} "
              f"{m['unique_ids']:>8} {m['id_switches_estimate']:>8} {m['line_count']:>8}")
    print("=" * 85)


def main():
    parser = argparse.ArgumentParser(description="Multi-Tracker Comparison Runner")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--trackers", nargs="+", required=True,
                        choices=list(TRACKER_REGISTRY.keys()),
                        help="Trackers to compare")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--config-dir", default="comparison/configs",
                        help="Directory containing tracker config JSONs")
    parser.add_argument("--count-line", type=float, default=None,
                        help="Position of counting line (fraction 0.0-1.0)")
    parser.add_argument("--line-orientation", choices=["horizontal", "vertical"],
                        default="horizontal",
                        help="Counting line orientation (default: horizontal)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    for i, tracker_name in enumerate(args.trackers):
        print(f"\n[{i + 1}/{len(args.trackers)}] Loading {tracker_name}...")

        try:
            adapter = load_tracker(tracker_name, args.config_dir)
        except Exception as e:
            print(f"  ERROR loading {tracker_name}: {e}")
            print(f"  Skipping.")
            continue

        output_video = str(output_dir / f"{tracker_name}.mp4")

        try:
            metrics = run_single_tracker(
                adapter=adapter,
                video_path=args.video,
                output_video=output_video,
                count_line_pos=args.count_line,
                line_orientation=args.line_orientation,
            )
            metrics["output_video"] = output_video
            all_metrics.append(metrics)
        except Exception as e:
            print(f"  ERROR running {tracker_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    if all_metrics:
        print_comparison_table(all_metrics)
        save_metrics_csv(all_metrics, str(output_dir / "metrics.csv"))

        # Save as JSON too
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2)

    print(f"\nIndividual videos saved to {output_dir}/")
    print(f"Run generate_report.py to create side-by-side video.")


if __name__ == "__main__":
    main()