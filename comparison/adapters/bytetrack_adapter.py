"""
ByteTrack Adapter: Baseline tracker using supervision + YOLOv11x.
"""

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from .base_tracker import BaseTrackerAdapter, TrackedDetection


class ByteTrackAdapter(BaseTrackerAdapter):

    def __init__(self):
        self.model = None
        self.tracker = None
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=5)
        self.trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=60)
        self.confidence = 0.3
        self._detector_name = "YOLOv11x"

    @property
    def name(self) -> str:
        return "ByteTrack"

    @property
    def detector_name(self) -> str:
        return self._detector_name

    def load(self, config: dict):
        model_path = config.get("detector_weights", "yolo11x.pt")
        self.confidence = config.get("confidence", 0.3)
        self._detector_name = config.get("detector_name", "YOLOv11x")

        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack(
            track_activation_threshold=config.get("track_activation_threshold", 0.25),
            lost_track_buffer=config.get("lost_track_buffer", 30),
            minimum_matching_threshold=config.get("minimum_matching_threshold", 0.8),
            frame_rate=config.get("frame_rate", 30),
        )

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[TrackedDetection]]:
        results = self.model(frame, conf=self.confidence, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Filter to persons (COCO class 0)
        if detections.class_id is not None and len(detections) > 0:
            mask = detections.class_id == 0
            detections = detections[mask]

        detections = self.tracker.update_with_detections(detections)

        # Build standardized output
        tracked = []
        labels = []
        if detections.tracker_id is not None:
            for i, tid in enumerate(detections.tracker_id):
                box = detections.xyxy[i]
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                tracked.append(TrackedDetection(
                    tracker_id=int(tid),
                    bbox=tuple(box),
                    confidence=float(detections.confidence[i]) if detections.confidence is not None else 0.0,
                    class_id=int(detections.class_id[i]) if detections.class_id is not None else 0,
                    center=(cx, cy),
                ))
                labels.append(f"#{tid}")

        # Annotate
        annotated = self.trace_annotator.annotate(frame.copy(), detections)
        annotated = self.box_annotator.annotate(annotated, detections)
        annotated = self.label_annotator.annotate(annotated, detections, labels=labels)

        return annotated, tracked

    def reset(self):
        if self.tracker:
            self.tracker = sv.ByteTrack(
                track_activation_threshold=0.25,
                lost_track_buffer=30,
                minimum_matching_threshold=0.8,
                frame_rate=30,
            )