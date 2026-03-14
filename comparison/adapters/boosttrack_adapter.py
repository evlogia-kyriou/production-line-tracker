"""
BoostTrack++ Adapter: Uses BoxMOT library with YOLOv11x.
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from .base_tracker import BaseTrackerAdapter, TrackedDetection


class BoostTrackAdapter(BaseTrackerAdapter):

    def __init__(self):
        self.model = None
        self.tracker = None
        self.confidence = 0.3
        self._detector_name = "YOLOv11x"

    @property
    def name(self) -> str:
        return "BoostTrack++"

    @property
    def detector_name(self) -> str:
        return self._detector_name

    def load(self, config: dict):
        from boxmot import BoostTrack

        model_path = config.get("detector_weights", "yolo11x.pt")
        reid_weights = config.get("reid_weights", "osnet_x0_25_msmt17.pt")
        self.confidence = config.get("confidence", 0.3)
        self._detector_name = config.get("detector_name", "YOLOv11x")
        device = config.get("device", "cuda:0")

        self.model = YOLO(model_path)
        self.tracker = BoostTrack(
            reid_weights=Path(reid_weights),
            device=device,
            half=False,
        )

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[TrackedDetection]]:
        results = self.model(frame, conf=self.confidence, verbose=False)[0]

        # Extract detections as numpy array: [x1, y1, x2, y2, conf, cls]
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return frame.copy(), []

        dets = np.hstack([
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy().reshape(-1, 1),
            boxes.cls.cpu().numpy().reshape(-1, 1),
        ])

        # Filter to persons (class 0)
        person_mask = dets[:, 5] == 0
        dets = dets[person_mask]

        if len(dets) == 0:
            return frame.copy(), []

        # Update tracker: expects (N, 6) array [x1,y1,x2,y2,conf,cls]
        # Returns (N, 7+) array [x1,y1,x2,y2,id,conf,cls,...]
        tracks = self.tracker.update(dets, frame)

        tracked = []
        annotated = frame.copy()

        if tracks.shape[0] > 0:
            for track in tracks:
                x1, y1, x2, y2 = track[0:4].astype(int)
                tid = int(track[4])
                conf = float(track[5])
                cls_id = int(track[6])
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                tracked.append(TrackedDetection(
                    tracker_id=tid,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls_id,
                    center=(cx, cy),
                ))

                # Draw on frame
                color = self._id_color(tid)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"#{tid}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return annotated, tracked

    def reset(self):
        # BoxMOT BoostTrack doesn't have a built-in reset;
        # re-initialize if needed
        if self.tracker:
            from boxmot import BoostTrack
            config = {
                "reid_weights": str(self.tracker.reid_weights)
                if hasattr(self.tracker, "reid_weights") else "osnet_x0_25_msmt17.pt",
                "device": "cuda:0",
            }
            self.tracker = BoostTrack(
                reid_weights=Path(config["reid_weights"]),
                device=config["device"],
                half=False,
            )

    @staticmethod
    def _id_color(track_id: int) -> tuple:
        """Generate a consistent color per track ID."""
        np.random.seed(track_id)
        return tuple(int(c) for c in np.random.randint(80, 255, 3))