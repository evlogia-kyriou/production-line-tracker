"""
Production Line Pipeline: Orchestrates detection, tracking, zone filtering,
worker assignment, transition counting, and visualization.
"""

import json
import time
import numpy as np
import cv2
import supervision as sv
from pathlib import Path
from ultralytics import YOLO

from .zone_manager import ZoneManager
from .transition_counter import TransitionCounter
from .csv_logger import CSVLogger
from .visualizer import Visualizer


class ProductionLinePipeline:
    """End-to-end pipeline for production line throughput tracking."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        zones_config: str = "config/zones_config.json",
        tracking_config: str = "config/tracking_config.json",
        output_dir: str = "outputs",
        use_roboflow_api: bool = False,
        roboflow_api_key: str | None = None,
        roboflow_model_id: str | None = None,
    ):
        # Load tracking config
        with open(tracking_config) as f:
            self.config = json.load(f)

        # Detection setup
        self.use_roboflow_api = use_roboflow_api
        if use_roboflow_api:
            from inference_sdk import InferenceHTTPClient
            self.rf_client = InferenceHTTPClient(
                api_url="https://detect.roboflow.com",
                api_key=roboflow_api_key,
            )
            self.rf_model_id = roboflow_model_id
            self.model = None
        else:
            self.model = YOLO(model_path)
            self.rf_client = None

        # Tracker (ByteTrack via supervision)
        self.tracker = sv.ByteTrack(
            track_activation_threshold=self.config["track_activation_threshold"],
            lost_track_buffer=self.config["lost_track_buffer"],
            minimum_matching_threshold=self.config["minimum_matching_threshold"],
            frame_rate=30,
        )

        # Components
        self.zone_manager = ZoneManager(zones_config)
        self.counter = TransitionCounter(
            raw_class_id=self.config["raw_class_id"],
            finished_class_id=self.config["finished_class_id"],
        )
        self.logger = CSVLogger(output_dir)
        self.visualizer = Visualizer()

        self.class_names = self.config.get("class_names", {})
        self.confidence_threshold = self.config["confidence_threshold"]
        self.frame_skip = self.config.get("frame_skip", 0)

    def _detect_local(self, frame: np.ndarray) -> sv.Detections:
        """Run detection with local YOLO model."""
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)[0]
        return sv.Detections.from_ultralytics(results)

    def _detect_roboflow(self, frame: np.ndarray) -> sv.Detections:
        """Run detection via Roboflow Hosted API."""
        result = self.rf_client.infer(frame, model_id=self.rf_model_id)

        xyxy, confidences, class_ids = [], [], []
        for pred in result.get("predictions", []):
            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            xyxy.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
            confidences.append(pred["confidence"])
            # Map class name to ID
            cls_name = pred["class"]
            cls_id = next(
                (int(k) for k, v in self.class_names.items() if v == cls_name),
                -1,
            )
            class_ids.append(cls_id)

        if not xyxy:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.array(xyxy, dtype=np.float32),
            confidence=np.array(confidences, dtype=np.float32),
            class_id=np.array(class_ids, dtype=int),
        )

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Run object detection (local or API)."""
        if self.use_roboflow_api:
            return self._detect_roboflow(frame)
        return self._detect_local(frame)

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """Process a single frame through the full pipeline.

        Returns:
            (annotated_frame, list_of_transition_events)
        """
        # Step 1: Detect
        detections = self.detect(frame)

        # Step 2: Filter by ROI
        detections = self.zone_manager.filter_detections_by_roi(detections)

        # Step 3: Track
        detections = self.tracker.update_with_detections(detections)

        # Step 4: Worker assignment + transition counting
        tracker_ids = detections.tracker_id.tolist() if detections.tracker_id is not None else []
        class_ids = detections.class_id.tolist() if detections.class_id is not None else []
        centers = []
        for box in detections.xyxy:
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            centers.append((cx, cy))

        events = self.counter.update(
            tracker_ids=tracker_ids,
            class_ids=class_ids,
            centers=centers,
            get_worker_fn=self.zone_manager.get_worker_for_point,
        )

        # Step 5: Log events
        for event in events:
            self.logger.log_event(event)
            print(f"  [TRANSITION] {event['worker']}: "
                  f"#{event['tracker_id']} completed in {event['duration_seconds']}s "
                  f"(total: {event['worker_total']})")

        worker_stats = self.counter.get_worker_stats()
        self.logger.log_summary_if_due(worker_stats)

        # Step 6: Visualize
        labels = []
        for tid, cid in zip(tracker_ids, class_ids):
            cls_name = self.class_names.get(str(cid), f"cls_{cid}")
            obj = self.counter.objects.get(tid)
            worker = obj.assigned_worker if obj else "?"
            labels.append(f"#{tid} {cls_name} [{worker}]")

        annotated = self.visualizer.draw_zones(
            frame, self.zone_manager.roi_polygon, self.zone_manager.worker_zones
        )
        annotated = self.visualizer.draw_detections(annotated, detections, labels)

        return annotated, events

    def run(self, source: str, output_path: str | None = None, display: bool = True):
        """Run the pipeline on a video file or RTSP stream.

        Args:
            source: Path to video file or RTSP URL.
            output_path: Optional path to save annotated video.
            display: Whether to show live display.
        """
        if not self.zone_manager.is_calibrated:
            print("[Pipeline] WARNING: Zones not calibrated. Run calibration tools first.")

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        frame_count = 0
        t_start = time.time()

        print(f"[Pipeline] Running on {source} ({w}x{h} @ {fps:.0f} FPS)")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Frame skip
                if self.frame_skip > 0 and frame_count % (self.frame_skip + 1) != 0:
                    continue

                annotated, events = self.process_frame(frame)

                # Stats overlay
                elapsed = time.time() - t_start
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                annotated = self.visualizer.draw_stats_overlay(
                    annotated,
                    self.counter.get_worker_stats(),
                    self.counter.total_completed,
                    current_fps,
                )

                if writer:
                    writer.write(annotated)

                if display:
                    cv2.imshow("Production Line Tracker", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            self.logger.close()

            elapsed = time.time() - t_start
            print(f"\n[Pipeline] Done. {frame_count} frames in {elapsed:.1f}s "
                  f"({frame_count / elapsed:.1f} FPS)")
            print(f"[Pipeline] Total completed: {self.counter.total_completed}")
            for worker, stats in self.counter.get_worker_stats().items():
                print(f"  {worker}: {stats['completed']} completed, "
                      f"{stats['in_progress']} in progress")