"""
Transition Counter: Detects class transitions for tracked objects
and maintains per-worker completion counts.
"""

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrackedObject:
    """State for a single tracked object."""
    tracker_id: int
    initial_class_id: int
    current_class_id: int
    assigned_worker: Optional[str] = None
    transition_counted: bool = False
    first_seen: float = field(default_factory=time.time)
    transition_time: Optional[float] = None


class TransitionCounter:
    """Tracks class transitions (raw -> finished) and counts per worker.

    Core logic:
    - When a new tracker_id appears, register it with its initial class.
    - When a tracked object first enters a worker's anchor zone, assign it permanently.
    - When the class changes from raw_class_id to finished_class_id, count it once.
    """

    def __init__(self, raw_class_id: int = 0, finished_class_id: int = 1):
        self.raw_class_id = raw_class_id
        self.finished_class_id = finished_class_id

        self.objects: dict[int, TrackedObject] = {}
        self.worker_counts: dict[str, int] = {}
        self.total_completed: int = 0

    def update(
        self,
        tracker_ids: list[int],
        class_ids: list[int],
        centers: list[tuple[float, float]],
        get_worker_fn,
    ) -> list[dict]:
        """Process a frame's tracked detections.

        Args:
            tracker_ids: List of persistent track IDs from ByteTrack.
            class_ids: List of class predictions for each tracked object.
            centers: List of (cx, cy) center points for each detection.
            get_worker_fn: Callable(cx, cy) -> Optional[str] that returns worker name.

        Returns:
            List of transition events that occurred this frame.
        """
        events = []
        active_ids = set(tracker_ids)

        for tid, cid, (cx, cy) in zip(tracker_ids, class_ids, centers):
            # Register new objects
            if tid not in self.objects:
                self.objects[tid] = TrackedObject(
                    tracker_id=tid,
                    initial_class_id=cid,
                    current_class_id=cid,
                )

            obj = self.objects[tid]
            obj.current_class_id = cid

            # Assign worker (permanent, first zone wins)
            if obj.assigned_worker is None:
                worker = get_worker_fn(cx, cy)
                if worker is not None:
                    obj.assigned_worker = worker
                    if worker not in self.worker_counts:
                        self.worker_counts[worker] = 0

            # Detect class transition: raw -> finished
            if (
                not obj.transition_counted
                and obj.initial_class_id == self.raw_class_id
                and cid == self.finished_class_id
                and obj.assigned_worker is not None
            ):
                obj.transition_counted = True
                obj.transition_time = time.time()
                self.worker_counts[obj.assigned_worker] += 1
                self.total_completed += 1

                duration = obj.transition_time - obj.first_seen
                event = {
                    "tracker_id": tid,
                    "worker": obj.assigned_worker,
                    "duration_seconds": round(duration, 2),
                    "timestamp": obj.transition_time,
                    "worker_total": self.worker_counts[obj.assigned_worker],
                }
                events.append(event)

        # Clean up lost tracks (keep recent ones for potential re-id)
        stale_ids = [
            tid for tid in self.objects
            if tid not in active_ids
            and (time.time() - self.objects[tid].first_seen) > 60
        ]
        for tid in stale_ids:
            del self.objects[tid]

        return events

    def get_worker_stats(self) -> dict[str, dict]:
        """Get current per-worker statistics."""
        stats = {}
        for worker, count in sorted(self.worker_counts.items()):
            active = sum(
                1 for obj in self.objects.values()
                if obj.assigned_worker == worker and not obj.transition_counted
            )
            stats[worker] = {
                "completed": count,
                "in_progress": active,
            }
        return stats

    def reset(self):
        """Reset all counts and tracked objects."""
        self.objects.clear()
        self.worker_counts.clear()
        self.total_completed = 0