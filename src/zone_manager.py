"""
Zone Manager: Handles ROI polygon filtering and worker anchor zone assignment.
"""

import json
import numpy as np
import cv2
from pathlib import Path
from typing import Optional


class ZoneManager:
    """Manages the work surface ROI and per-worker anchor zones."""

    def __init__(self, config_path: str = "config/zones_config.json"):
        self.config_path = Path(config_path)
        self.roi_polygon: Optional[np.ndarray] = None
        self.worker_zones: dict[str, np.ndarray] = {}
        self._load_config()

    def _load_config(self):
        if not self.config_path.exists():
            print(f"[ZoneManager] No config found at {self.config_path}. Run calibration first.")
            return

        with open(self.config_path) as f:
            config = json.load(f)

        roi = config.get("roi_polygon", [])
        if roi:
            self.roi_polygon = np.array(roi, dtype=np.int32)

        for name, data in config.get("workers", {}).items():
            self.worker_zones[name] = np.array(data["zone"], dtype=np.int32)

        print(f"[ZoneManager] Loaded ROI with {len(roi)} vertices, {len(self.worker_zones)} worker zones.")

    @property
    def is_calibrated(self) -> bool:
        return self.roi_polygon is not None and len(self.worker_zones) > 0

    def is_inside_roi(self, cx: float, cy: float) -> bool:
        """Check if a point (object center) is inside the ROI polygon."""
        if self.roi_polygon is None:
            return True  # No ROI = accept all
        result = cv2.pointPolygonTest(self.roi_polygon, (float(cx), float(cy)), False)
        return result >= 0

    def filter_detections_by_roi(self, detections) -> "detections":
        """Filter a supervision Detections object to only include those inside the ROI.

        Args:
            detections: sv.Detections object with xyxy bounding boxes.

        Returns:
            Filtered sv.Detections with only in-ROI objects.
        """
        if self.roi_polygon is None or len(detections) == 0:
            return detections

        centers_x = (detections.xyxy[:, 0] + detections.xyxy[:, 2]) / 2
        centers_y = (detections.xyxy[:, 1] + detections.xyxy[:, 3]) / 2

        mask = np.array([
            self.is_inside_roi(cx, cy)
            for cx, cy in zip(centers_x, centers_y)
        ])

        return detections[mask]

    def get_worker_for_point(self, cx: float, cy: float) -> Optional[str]:
        """Determine which worker's anchor zone contains the given point.

        Returns:
            Worker name string, or None if the point is not in any zone.
        """
        for name, zone_polygon in self.worker_zones.items():
            result = cv2.pointPolygonTest(zone_polygon, (float(cx), float(cy)), False)
            if result >= 0:
                return name
        return None

    def save_config(self, roi_polygon: list, workers: dict):
        """Save ROI and worker zones to config file."""
        config = {
            "roi_polygon": roi_polygon,
            "workers": workers,
        }
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"[ZoneManager] Config saved to {self.config_path}")
        self._load_config()