"""
Report Generator: Creates a 2x2 side-by-side comparison video from individual tracker outputs.

Usage:
    python comparison/generate_report.py \
        --input outputs/comparisons/running/ \
        --output outputs/comparisons/running/side_by_side.mp4

    # Custom grid size (for 2 trackers: 1x2 layout)
    python comparison/generate_report.py \
        --input outputs/comparisons/running/ \
        --output outputs/comparisons/running/side_by_side.mp4 \
        --layout 1x2
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path


def get_tracker_videos(input_dir: Path) -> list[dict]:
    """Find all tracker output videos and their metrics."""
    metrics_path = input_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics.json found in {input_dir}. Run run_comparison.py first.")

    with open(metrics_path) as f:
        all_metrics = json.load(f)

    # Verify video files exist
    valid = []
    for m in all_metrics:
        video_path = Path(m.get("output_video", ""))
        if video_path.exists():
            valid.append(m)
        else:
            print(f"Warning: Video not found for {m['tracker']}: {video_path}")

    return valid


def parse_layout(layout_str: str, num_videos: int) -> tuple[int, int]:
    """Parse layout string like '2x2' or auto-detect from video count."""
    if layout_str:
        rows, cols = layout_str.split("x")
        return int(rows), int(cols)

    # Auto layout
    if num_videos <= 2:
        return 1, 2
    elif num_videos <= 4:
        return 2, 2
    elif num_videos <= 6:
        return 2, 3
    else:
        return 3, 3


def generate_side_by_side(
    tracker_data: list[dict],
    output_path: str,
    layout: tuple[int, int],
    cell_width: int = 640,
    cell_height: int = 360,
):
    """Generate a grid video with all trackers side by side."""
    rows, cols = layout
    grid_w = cols * cell_width
    grid_h = rows * cell_height

    # Open all video captures
    caps = []
    for data in tracker_data:
        cap = cv2.VideoCapture(data["output_video"])
        if not cap.isOpened():
            print(f"Error: Cannot open {data['output_video']}")
            return
        caps.append(cap)

    # Get FPS from first video
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 30

    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (grid_w, grid_h))

    frame_count = 0

    print(f"Generating {rows}x{cols} grid ({grid_w}x{grid_h}) at {fps:.0f} FPS...")

    while True:
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        all_done = True

        for idx, (cap, data) in enumerate(zip(caps, tracker_data)):
            row = idx // cols
            col = idx % cols

            ret, frame = cap.read()
            if ret:
                all_done = False
                # Resize to cell size
                cell = cv2.resize(frame, (cell_width, cell_height))
            else:
                # Show last frame or black
                cell = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
                cv2.putText(cell, f"{data['tracker']}: ended", (20, cell_height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

            # Add tracker label banner at top of cell
            banner_h = 32
            cv2.rectangle(cell, (0, 0), (cell_width, banner_h), (30, 30, 30), -1)
            label = f"{data['tracker']} ({data['detector']}) | FPS: {data['avg_fps']}"
            cv2.putText(cell, label, (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Place cell in grid
            y0 = row * cell_height
            x0 = col * cell_width
            grid[y0:y0 + cell_height, x0:x0 + cell_width] = cell

        if all_done:
            break

        # Draw grid lines
        for r in range(1, rows):
            y = r * cell_height
            cv2.line(grid, (0, y), (grid_w, y), (255, 255, 255), 1)
        for c in range(1, cols):
            x = c * cell_width
            cv2.line(grid, (x, 0), (x, grid_h), (255, 255, 255), 1)

        writer.write(grid)
        frame_count += 1

    # Release
    for cap in caps:
        cap.release()
    writer.release()

    print(f"Side-by-side video saved: {output_path} ({frame_count} frames)")


def print_summary_table(tracker_data: list[dict]):
    """Print final comparison table."""
    print("\n" + "=" * 85)
    print("TRACKER COMPARISON SUMMARY")
    print("=" * 85)
    print(f"{'Tracker':<18} {'Detector':<12} {'FPS':>8} {'IDs':>8} {'ID Sw.':>8} {'Count':>8}")
    print("-" * 85)
    for m in tracker_data:
        print(f"{m['tracker']:<18} {m['detector']:<12} {m['avg_fps']:>8.1f} "
              f"{m['unique_ids']:>8} {m['id_switches_estimate']:>8} {m['line_count']:>8}")
    print("=" * 85)


def main():
    parser = argparse.ArgumentParser(description="Generate Side-by-Side Comparison Video")
    parser.add_argument("--input", required=True, help="Directory with individual tracker videos + metrics.json")
    parser.add_argument("--output", required=True, help="Output side-by-side video path")
    parser.add_argument("--layout", default=None, help="Grid layout (e.g., 2x2, 1x2). Auto-detected if omitted.")
    parser.add_argument("--cell-width", type=int, default=640, help="Width of each cell in the grid")
    parser.add_argument("--cell-height", type=int, default=360, help="Height of each cell in the grid")
    args = parser.parse_args()

    input_dir = Path(args.input)
    tracker_data = get_tracker_videos(input_dir)

    if not tracker_data:
        print("No valid tracker videos found. Run run_comparison.py first.")
        return

    print(f"Found {len(tracker_data)} tracker outputs:")
    for d in tracker_data:
        print(f"  - {d['tracker']} ({d['detector']})")

    layout = parse_layout(args.layout, len(tracker_data))
    print(f"Layout: {layout[0]}x{layout[1]}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    generate_side_by_side(
        tracker_data=tracker_data,
        output_path=args.output,
        layout=layout,
        cell_width=args.cell_width,
        cell_height=args.cell_height,
    )

    print_summary_table(tracker_data)


if __name__ == "__main__":
    main()