"""
Interactive ROI Calibration Tool.

Usage:
    python tools/calibrate_roi.py --video videos/sample.mp4

Controls:
    - Left click: Add polygon vertex
    - Right click: Remove last vertex
    - Enter: Confirm and save
    - Esc: Cancel
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path


class ROICalibrator:
    def __init__(self, frame: np.ndarray, config_path: str):
        self.frame = frame.copy()
        self.display = frame.copy()
        self.points = []
        self.config_path = Path(config_path)
        self.window_name = "ROI Calibration - Click to add points, Enter to save, Esc to cancel"

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append([x, y])
            self._redraw()
        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            self.points.pop()
            self._redraw()

    def _redraw(self):
        self.display = self.frame.copy()
        if len(self.points) > 0:
            pts = np.array(self.points, dtype=np.int32)
            # Draw filled polygon with transparency
            if len(self.points) > 2:
                overlay = self.display.copy()
                cv2.fillPoly(overlay, [pts], (0, 255, 255))
                self.display = cv2.addWeighted(overlay, 0.2, self.display, 0.8, 0)
            cv2.polylines(self.display, [pts], True, (0, 255, 255), 2)
            for pt in self.points:
                cv2.circle(self.display, tuple(pt), 5, (0, 0, 255), -1)

        info = f"Vertices: {len(self.points)} | Left click=add, Right click=undo, Enter=save, Esc=cancel"
        cv2.putText(self.display, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def run(self) -> list:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self._redraw()

        while True:
            cv2.imshow(self.window_name, self.display)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter
                break
            elif key == 27:  # Esc
                self.points = []
                break

        cv2.destroyAllWindows()
        return self.points

    def save(self):
        # Load existing config or create new
        config = {"roi_polygon": [], "workers": {}}
        if self.config_path.exists():
            with open(self.config_path) as f:
                config = json.load(f)

        config["roi_polygon"] = self.points
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"ROI saved with {len(self.points)} vertices to {self.config_path}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate ROI polygon on work surface")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--config", default="config/zones_config.json", help="Config output path")
    parser.add_argument("--frame", type=int, default=30, help="Frame number to use for calibration")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame {args.frame} from {args.video}")
        return

    calibrator = ROICalibrator(frame, args.config)
    points = calibrator.run()

    if points:
        calibrator.save()
    else:
        print("Calibration cancelled.")


if __name__ == "__main__":
    main()