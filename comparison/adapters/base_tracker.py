"""
Base Tracker Adapter: Abstract interface that all tracker adapters must implement.

This allows the comparison runner to treat all trackers uniformly regardless
of their underlying implementation (BoxMOT, TrackTrack, FastTracker, supervision).
"""

import time
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TrackedDetection:
    """Standardized detection output across all trackers."""
    tracker_id: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    center: tuple[float, float]


class BaseTrackerAdapter(ABC):
    """Abstract base class for tracker adapters."""

    @abstractmethod
    def load(self, config: dict):
        """Load detector and tracker models.

        Args:
            config: Dictionary with model paths and parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[TrackedDetection]]:
        """Run detection + tracking on a single frame.

        Args:
            frame: BGR numpy array from cv2.

        Returns:
            (annotated_frame, list_of_tracked_detections)
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Reset tracker state (for running on a new video)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for display and reports."""
        raise NotImplementedError

    @property
    @abstractmethod
    def detector_name(self) -> str:
        """Name of the detector used."""
        raise NotImplementedError

    def benchmark_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[TrackedDetection], float]:
        """Process frame with timing.

        Returns:
            (annotated_frame, detections, elapsed_seconds)
        """
        t0 = time.perf_counter()
        annotated, detections = self.process_frame(frame)
        elapsed = time.perf_counter() - t0
        return annotated, detections, elapsed