"""
Line Counter: Counts objects crossing a virtual line.

Used in demo scenarios where class transition is not available
(e.g., running events, retail counting, vehicle tracking).
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class LineSpec:
    """Defines a counting line between two points."""
    start: tuple[int, int]
    end: tuple[int, int]
    name: str = "Line"

    @property
    def vector(self) -> np.ndarray:
        return np.array([self.end[0] - self.start[0], self.end[1] - self.start[1]])


class LineCrossingCounter:
    """Counts objects that cross one or more virtual lines.

    Uses the sign of the cross product between the line vector and
    the object's movement vector to determine crossing direction.
    Each tracked object is counted at most once per line.
    """

    def __init__(self, lines: list[LineSpec]):
        self.lines = lines
        # track_id -> last known center position
        self._prev_positions: dict[int, tuple[float, float]] = {}
        # track_id -> set of line names already crossed
        self._crossed: dict[int, set[str]] = {}
        # per-line counts (in positive direction)
        self.in_counts: dict[str, int] = {line.name: 0 for line in lines}
        # per-line counts (in negative direction)
        self.out_counts: dict[str, int] = {line.name: 0 for line in lines}
        self.total_in: int = 0
        self.total_out: int = 0

    def _cross_product_sign(self, line: LineSpec, point: tuple[float, float]) -> float:
        """Compute which side of the line a point is on.

        Returns positive if point is on the left side, negative if right.
        """
        dx = line.end[0] - line.start[0]
        dy = line.end[1] - line.start[1]
        px = point[0] - line.start[0]
        py = point[1] - line.start[1]
        return dx * py - dy * px

    def update(
        self,
        tracker_ids: list[int],
        centers: list[tuple[float, float]],
    ) -> list[dict]:
        """Process a frame's tracked detections.

        Args:
            tracker_ids: List of persistent track IDs.
            centers: List of (cx, cy) center points.

        Returns:
            List of crossing events that occurred this frame.
        """
        events = []

        for tid, center in zip(tracker_ids, centers):
            if tid not in self._crossed:
                self._crossed[tid] = set()

            prev = self._prev_positions.get(tid)
            self._prev_positions[tid] = center

            if prev is None:
                continue

            for line in self.lines:
                if line.name in self._crossed[tid]:
                    continue

                prev_sign = self._cross_product_sign(line, prev)
                curr_sign = self._cross_product_sign(line, center)

                # Crossing detected when sign changes
                if prev_sign * curr_sign < 0:
                    self._crossed[tid].add(line.name)

                    if curr_sign > 0:
                        self.in_counts[line.name] += 1
                        self.total_in += 1
                        direction = "in"
                    else:
                        self.out_counts[line.name] += 1
                        self.total_out += 1
                        direction = "out"

                    events.append({
                        "tracker_id": tid,
                        "line": line.name,
                        "direction": direction,
                        "in_count": self.in_counts[line.name],
                        "out_count": self.out_counts[line.name],
                    })

        # Clean up stale tracks
        active_ids = set(tracker_ids)
        stale = [tid for tid in self._prev_positions if tid not in active_ids]
        for tid in stale:
            del self._prev_positions[tid]
            if tid in self._crossed:
                del self._crossed[tid]

        return events

    def get_stats(self) -> dict:
        """Get current counting stats."""
        return {
            "total_in": self.total_in,
            "total_out": self.total_out,
            "per_line": {
                line.name: {
                    "in": self.in_counts[line.name],
                    "out": self.out_counts[line.name],
                }
                for line in self.lines
            },
        }

    def reset(self):
        self._prev_positions.clear()
        self._crossed.clear()
        for line in self.lines:
            self.in_counts[line.name] = 0
            self.out_counts[line.name] = 0
        self.total_in = 0
        self.total_out = 0