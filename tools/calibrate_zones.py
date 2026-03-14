"""
Interactive Worker Anchor Zone Calibration Tool.

Usage:
    python tools/calibrate_zones.py --video videos/sample.mp4 --workers 4

Controls:
    - Left click: Add polygon vertex for current worker zone
    - Right click: Remove last vertex
    - Enter: Confirm current zone, move to next worker
    - Esc: Cancel
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path


class ZoneCalibrator:
    def __init__(self, frame: np.ndarray, num_workers: int, config_path: str):
        self.frame = frame.copy()
        self.display = frame.copy()
        self.num_workers = num_workers
        self.config_path = Path(config_path)
        self.current_worker = 0
        self.current_points = []
        self.completed_zones = {}
        self.window_name = "Zone Calibration"

        # Load existing ROI if available
        self.roi_polygon = None
        if self.config_path.exists():
            with open(self.config_path) as f:
                config = json.load(f)
                roi = config.get("roi_polygon", [])
                if roi:
                    self.roi_polygon = np.array(roi, dtype=np.int32)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append([x, y])
            self._redraw()
        elif event == cv2.EVENT_RBUTTONDOWN and self.current_points:
            self.current_points.pop()
            self._redraw()

    def _redraw(self):
        self.display = self.frame.copy()

        # Draw existing ROI
        if self.roi_polygon is not None:
            cv2.polylines(self.display, [self.roi_polygon], True, (0, 255, 255), 2)

        # Draw completed zones
        colors = [
            (255, 165, 0), (0, 255, 127), (255, 105, 180),
            (0, 191, 255), (50, 205, 50), (238, 130, 238),
        ]
        for name, pts in self.completed_zones.items():
            idx = int(name.split("_")[1]) - 1
            color = colors[idx % len(colors)]
            poly = np.array(pts, dtype=np.int32)
            overlay = self.display.copy()
            cv2.fillPoly(overlay, [poly], color)
            self.display = cv2.addWeighted(overlay, 0.2, self.display, 0.8, 0)
            cv2.polylines(self.display, [poly], True, color, 2)
            cx = int(poly[:, 0].mean())
            cy = int(poly[:, 1].mean())
            cv2.putText(self.display, name, (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw current zone in progress
        if self.current_points:
            pts = np.array(self.current_points, dtype=np.int32)
            cv2.polylines(self.display, [pts], True, (255, 255, 255), 2)
            for pt in self.current_points:
                cv2.circle(self.display, tuple(pt), 5, (0, 0, 255), -1)

        worker_name = f"Worker_{self.current_worker + 1}"
        info = (f"Drawing zone for {worker_name} "
                f"({self.current_worker + 1}/{self.num_workers}) | "
                f"Vertices: {len(self.current_points)} | Enter=confirm, Esc=cancel")
        cv2.putText(self.display, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    def run(self) -> dict:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while self.current_worker < self.num_workers:
            self._redraw()
            cv2.imshow(self.window_name, self.display)
            key = cv2.waitKey(1) & 0xFF

            if key == 13 and len(self.current_points) >= 3:  # Enter
                name = f"Worker_{self.current_worker + 1}"
                self.completed_zones[name] = self.current_points.copy()
                print(f"  Zone saved for {name} ({len(self.current_points)} vertices)")
                self.current_worker += 1
                self.current_points = []
            elif key == 27:  # Esc
                print("Calibration cancelled.")
                cv2.destroyAllWindows()
                return {}

        cv2.destroyAllWindows()
        return self.completed_zones

    def save(self, zones: dict):
        config = {"roi_polygon": [], "workers": {}}
        if self.config_path.exists():
            with open(self.config_path) as f:
                config = json.load(f)

        config["workers"] = {
            name: {"zone": pts} for name, pts in zones.items()
        }
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Worker zones saved to {self.config_path}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate worker anchor zones")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--workers", type=int, required=True, help="Number of workers")
    parser.add_argument("--config", default="config/zones_config.json", help="Config path")
    parser.add_argument("--frame", type=int, default=30, help="Frame number to use")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame {args.frame} from {args.video}")
        return

    calibrator = ZoneCalibrator(frame, args.workers, args.config)
    zones = calibrator.run()

    if zones:
        calibrator.save(zones)
    else:
        print("No zones saved.")


if __name__ == "__main__":
    main()