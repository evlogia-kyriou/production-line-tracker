"""
Run the Production Line Throughput Tracker.

Usage:
    # Local YOLO model
    python tools/run.py --video videos/sample.mp4

    # Roboflow API
    python tools/run.py --video videos/sample.mp4 --roboflow --api-key YOUR_KEY --model-id your/model

    # RTSP stream
    python tools/run.py --video rtsp://camera-ip:554/stream --display

    # Save output video
    python tools/run.py --video videos/sample.mp4 --output outputs/result.mp4
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import ProductionLinePipeline


def main():
    parser = argparse.ArgumentParser(description="Production Line Throughput Tracker")
    parser.add_argument("--video", required=True, help="Path to video file or RTSP URL")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model path (local)")
    parser.add_argument("--zones-config", default="config/zones_config.json")
    parser.add_argument("--tracking-config", default="config/tracking_config.json")
    parser.add_argument("--output", default=None, help="Path to save annotated output video")
    parser.add_argument("--output-dir", default="outputs", help="Directory for CSV logs")
    parser.add_argument("--display", action="store_true", default=True, help="Show live display")
    parser.add_argument("--no-display", action="store_true", help="Disable live display")

    # Roboflow API options
    parser.add_argument("--roboflow", action="store_true", help="Use Roboflow hosted API")
    parser.add_argument("--api-key", default=None, help="Roboflow API key")
    parser.add_argument("--model-id", default=None, help="Roboflow model ID (e.g., project/version)")

    args = parser.parse_args()

    display = args.display and not args.no_display

    pipeline = ProductionLinePipeline(
        model_path=args.model,
        zones_config=args.zones_config,
        tracking_config=args.tracking_config,
        output_dir=args.output_dir,
        use_roboflow_api=args.roboflow,
        roboflow_api_key=args.api_key,
        roboflow_model_id=args.model_id,
    )

    pipeline.run(
        source=args.video,
        output_path=args.output,
        display=display,
    )


if __name__ == "__main__":
    main()