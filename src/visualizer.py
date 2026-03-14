"""
Visualizer: Draws ROI, worker zones, tracked objects, and stats overlay.
"""

import cv2
import numpy as np
import supervision as sv


# Color palette for workers (BGR)
WORKER_COLORS = [
    (255, 165, 0),   # orange
    (0, 255, 127),   # spring green
    (255, 105, 180), # hot pink
    (0, 191, 255),   # deep sky blue
    (50, 205, 50),   # lime green
    (238, 130, 238), # violet
]


class Visualizer:
    """Draws live annotations on frames."""

    def __init__(self):
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=5)
        self.trace_annotator = sv.TraceAnnotator(
            thickness=2, trace_length=60, position=sv.Position.CENTER
        )

    def draw_zones(
        self,
        frame: np.ndarray,
        roi_polygon: np.ndarray | None,
        worker_zones: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Draw ROI boundary and worker anchor zones."""
        annotated = frame.copy()

        # Draw ROI polygon
        if roi_polygon is not None and len(roi_polygon) > 0:
            cv2.polylines(annotated, [roi_polygon], True, (0, 255, 255), 2)

        # Draw worker zones
        for i, (name, zone) in enumerate(worker_zones.items()):
            color = WORKER_COLORS[i % len(WORKER_COLORS)]
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [zone], color)
            annotated = cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0)
            cv2.polylines(annotated, [zone], True, color, 2)

            # Zone label
            cx = int(zone[:, 0].mean())
            cy = int(zone[:, 1].min()) - 10
            cv2.putText(annotated, name, (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return annotated

    def draw_detections(
        self,
        frame: np.ndarray,
        detections,
        labels: list[str],
    ) -> np.ndarray:
        """Draw bounding boxes, labels, and movement traces."""
        annotated = self.trace_annotator.annotate(frame.copy(), detections)
        annotated = self.box_annotator.annotate(annotated, detections)
        annotated = self.label_annotator.annotate(annotated, detections, labels=labels)
        return annotated

    def draw_stats_overlay(
        self,
        frame: np.ndarray,
        worker_stats: dict,
        total_completed: int,
        fps: float,
    ) -> np.ndarray:
        """Draw per-worker stats panel on the frame."""
        h, w = frame.shape[:2]
        panel_w = 250
        panel_h = 40 + len(worker_stats) * 30 + 50

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - panel_w - 10, 10),
                      (w - 10, 10 + panel_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        x0 = w - panel_w
        y = 35

        cv2.putText(frame, "Worker Stats", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 10

        for i, (worker, stats) in enumerate(sorted(worker_stats.items())):
            y += 30
            color = WORKER_COLORS[i % len(WORKER_COLORS)]
            text = f"{worker}: {stats['completed']} done ({stats['in_progress']} active)"
            cv2.putText(frame, text, (x0, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        y += 35
        cv2.putText(frame, f"Total: {total_completed} | FPS: {fps:.1f}", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return frame