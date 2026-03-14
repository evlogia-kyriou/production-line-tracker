from .base_tracker import BaseTrackerAdapter, TrackedDetection
from .bytetrack_adapter import ByteTrackAdapter
from .boosttrack_adapter import BoostTrackAdapter
from .tracktrack_adapter import TrackTrackAdapter
from .fasttracker_adapter import FastTrackerAdapter

TRACKER_REGISTRY = {
    "bytetrack": ByteTrackAdapter,
    "boosttrack": BoostTrackAdapter,
    "tracktrack": TrackTrackAdapter,
    "fasttracker": FastTrackerAdapter,
}