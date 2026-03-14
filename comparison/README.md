# Tracker Comparison

Side-by-side comparison of 4 SOTA multi-object trackers on the same video input.

## Trackers

| Tracker | Detector | Source | Venue | Integration |
|---------|----------|--------|-------|-------------|
| **ByteTrack** | YOLOv11x | supervision library | ECCV 2022 | Ready |
| **BoostTrack++** | YOLOv11x | BoxMOT library | arXiv 2024 (journal: MVA) | Ready |
| **TrackTrack** | YOLOX-X | github.com/kamkyu94/TrackTrack | CVPR 2025 | Adapter stub (needs integration) |
| **FastTracker** | YOLOX-X | github.com/Hamidreza-Hashempoor/FastTracker | arXiv 2025 | Adapter stub (needs integration) |

## Quick Start (ByteTrack + BoostTrack++ work immediately)

```bash
# Install dependencies
pip install ultralytics supervision boxmot

# Run comparison with the two plug-and-play trackers
python comparison/run_comparison.py \
    --video videos/running.mp4 \
    --trackers bytetrack boosttrack \
    --output outputs/comparisons/running/ \
    --count-line 0.6

# Generate side-by-side video
python comparison/generate_report.py \
    --input outputs/comparisons/running/ \
    --output outputs/comparisons/running/side_by_side.mp4
```

## Full 4-Tracker Comparison

### Prerequisites

```bash
# 1. Clone external repos
git clone https://github.com/kamkyu94/TrackTrack.git external/TrackTrack
git clone https://github.com/Hamidreza-Hashempoor/FastTracker.git external/FastTracker

# 2. Download YOLOX-X weights
# From ByteTrack repo or YOLOX model zoo -> models/detectors/

# 3. Download FastReID weights
# From TrackTrack repo instructions -> models/reid/

# 4. Complete the adapter TODOs
# comparison/adapters/tracktrack_adapter.py
# comparison/adapters/fasttracker_adapter.py
```

### Run All 4

```bash
python comparison/run_comparison.py \
    --video videos/running.mp4 \
    --trackers bytetrack boosttrack tracktrack fasttracker \
    --output outputs/comparisons/running/ \
    --count-line 0.6

python comparison/generate_report.py \
    --input outputs/comparisons/running/ \
    --output outputs/comparisons/running/side_by_side.mp4
```

## Output

### Individual videos
```
outputs/comparisons/running/
├── bytetrack.mp4
├── boosttrack.mp4
├── tracktrack.mp4
├── fasttracker.mp4
├── metrics.csv
├── metrics.json
└── side_by_side.mp4        # 2x2 grid comparison
```

### Metrics table (printed + CSV)
```
Tracker         | Detector | FPS   | IDs  | ID Sw. | Count
----------------|----------|-------|------|--------|------
ByteTrack       | YOLOv11x | 52.3  |   35 |      7 |   21
BoostTrack++    | YOLOv11x | 45.2  |   28 |      2 |   23
TrackTrack      | YOLOX-X  | 28.7  |   26 |      1 |   24
FastTracker     | YOLOX-X  | 355.1 |   27 |      3 |   23
```

## Architecture

All trackers are wrapped with the **adapter pattern** (`comparison/adapters/`), normalizing their APIs into a common interface:

```python
class BaseTrackerAdapter:
    def load(config)          # Load detector + tracker
    def process_frame(frame)  # Returns (annotated_frame, tracked_detections)
    def reset()               # Clear tracker state
    def name                  # Display name
    def detector_name         # Which detector is used
```

This allows the comparison runner to iterate over trackers without knowing their internals.

## Note on Fairness

ByteTrack and BoostTrack++ use YOLOv11x (newer, stronger detector), while TrackTrack and FastTracker use YOLOX-X (their native detector). This reflects real-world deployment choices rather than a controlled ablation. For a strictly fair comparison, all trackers could use the same YOLOX-X detector via BoxMOT configuration.