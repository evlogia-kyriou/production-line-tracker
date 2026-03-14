"""
CSV Logger: Logs per-worker throughput events and periodic summaries.
"""

import csv
import time
from datetime import datetime
from pathlib import Path


class CSVLogger:
    """Logs transition events and periodic summaries to CSV files."""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Event log: one row per completed transition
        self.events_path = self.output_dir / f"events_{timestamp}.csv"
        self._events_file = open(self.events_path, "w", newline="")
        self._events_writer = csv.writer(self._events_file)
        self._events_writer.writerow([
            "timestamp", "tracker_id", "worker", "duration_seconds", "worker_total"
        ])

        # Summary log: periodic snapshots of all worker counts
        self.summary_path = self.output_dir / f"summary_{timestamp}.csv"
        self._summary_file = open(self.summary_path, "w", newline="")
        self._summary_writer = csv.writer(self._summary_file)
        self._summary_header_written = False

        self._last_summary_time = time.time()
        self._summary_interval = 60  # seconds

        print(f"[CSVLogger] Events: {self.events_path}")
        print(f"[CSVLogger] Summary: {self.summary_path}")

    def log_event(self, event: dict):
        """Log a single transition event."""
        self._events_writer.writerow([
            datetime.fromtimestamp(event["timestamp"]).isoformat(),
            event["tracker_id"],
            event["worker"],
            event["duration_seconds"],
            event["worker_total"],
        ])
        self._events_file.flush()

    def log_summary_if_due(self, worker_stats: dict):
        """Write a summary row if the interval has elapsed."""
        now = time.time()
        if now - self._last_summary_time < self._summary_interval:
            return

        if not self._summary_header_written:
            workers = sorted(worker_stats.keys())
            self._summary_writer.writerow(["timestamp"] + workers)
            self._summary_header_written = True
            self._summary_workers = workers

        row = [datetime.now().isoformat()]
        for worker in self._summary_workers:
            stats = worker_stats.get(worker, {"completed": 0})
            row.append(stats["completed"])
        self._summary_writer.writerow(row)
        self._summary_file.flush()
        self._last_summary_time = now

    def close(self):
        self._events_file.close()
        self._summary_file.close()
        print(f"[CSVLogger] Logs saved.")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass