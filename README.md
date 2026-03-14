# Production Line Throughput Tracker

Real-time per-worker productivity tracking for production lines using YOLOv11 + ByteTrack.

Detects objects on a work surface, assigns them to individual workers based on spatial anchor zones, and tracks class transitions (e.g., raw material to finished product) to count completed pieces per worker.

## How It Works

1. **Detection** -- YOLOv11 detects objects and classifies them (e.g., `dough_ball`, `matzah_sheet`)
2. **ROI Filtering** -- Only detections within the defined work surface polygon are processed
3. **Tracking** -- ByteTrack (via `supervision`) maintains persistent IDs across frames
4. **Worker Assignment** -- When an object first appears in a worker's anchor zone, it is permanently assigned to that worker
5. **Class Transition Counting** -- When a tracked object's class changes from raw to finished, the worker's completed count increments
6. **Logging** -- Per-worker stats display live on screen and log to CSV

## Project Structure

```
production-line-tracker/
├── config/
│   ├── zones_config.json              # ROI polygon + worker anchor zones
│   └── tracking_config.json           # Confidence, buffer, skip-frame settings
├── src/                               # Main pipeline (class transition counting)
│   ├── pipeline.py                    # Orchestrates detection → tracking → counting
│   ├── zone_manager.py                # ROI filtering + anchor zone logic
│   ├── transition_counter.py          # Class transition detection + per-worker counting
│   ├── csv_logger.py                  # Per-worker CSV logging
│   └── visualizer.py                  # Live display overlay
├── tools/
│   ├── calibrate_roi.py               # Interactive ROI polygon drawing
│   ├── calibrate_zones.py             # Interactive worker anchor zone setup
│   └── run.py                         # Main entry point
├── demo/                              # Demo scenarios (line-crossing counting)
│   ├── line_counter.py                # Shared line-crossing logic
│   ├── demo_running_event.py          # Finish line counting for running events
│   └── demo_retail_counting.py        # Entry/exit people counting for retail
├── comparison/                        # Multi-tracker benchmark (4 SOTA trackers)
│   ├── run_comparison.py              # Runs all trackers on same video
│   ├── generate_report.py             # 2x2 side-by-side video + metrics table
│   ├── configs/                       # Per-tracker JSON configs
│   └── adapters/                      # Adapter pattern wrapping each tracker
│       ├── base_tracker.py            # Abstract interface
│       ├── bytetrack_adapter.py       # supervision ByteTrack (baseline)
│       ├── boosttrack_adapter.py      # BoxMOT BoostTrack++
│       ├── tracktrack_adapter.py      # CVPR 2025 TrackTrack
│       └── fasttracker_adapter.py     # FastTracker
├── models/                            # Model weights (gitignored, auto/manual download)
│   ├── detectors/                     # yolo11x.pt, yolox_x.pth
│   └── reid/                          # osnet, fastreid weights
├── external/                          # Cloned tracker repos (gitignored)
├── outputs/                           # CSV logs, output videos, comparison results
└── videos/                            # Input video files
```

## Setup

```bash
conda create -n plt-tracker python=3.10 -y
conda activate plt-tracker
pip install -r requirements.txt
```

## Quick Start

### 1. Calibrate the ROI (work surface polygon)

```bash
python tools/calibrate_roi.py --video videos/sample.mp4
```

Click to define polygon vertices on the work surface. Press `Enter` to confirm, `Esc` to cancel.

### 2. Calibrate Worker Anchor Zones

```bash
python tools/calibrate_zones.py --video videos/sample.mp4 --workers 4
```

Draw rectangular zones for each worker position. Zones are saved to `config/zones_config.json`.

### 3. Run the Tracker

```bash
# On a video file
python tools/run.py --video videos/sample.mp4

# On an RTSP stream
python tools/run.py --video rtsp://camera-ip:554/stream

# With custom config
python tools/run.py --video videos/sample.mp4 --tracking-config config/tracking_config.json
```

### 4. View Results

- Live display shows per-worker counts and active tracks
- CSV logs are saved to `outputs/` with timestamps

## Configuration

### zones_config.json

```json
{
  "roi_polygon": [[100, 200], [500, 200], [500, 600], [100, 600]],
  "workers": {
    "Worker_1": {"zone": [[100, 200], [300, 200], [300, 400], [100, 400]]},
    "Worker_2": {"zone": [[300, 200], [500, 200], [500, 400], [300, 400]]}
  }
}
```

### tracking_config.json

```json
{
  "confidence_threshold": 0.3,
  "track_activation_threshold": 0.25,
  "lost_track_buffer": 30,
  "minimum_matching_threshold": 0.8,
  "frame_skip": 0,
  "class_names": {
    "0": "dough_ball",
    "1": "matzah_sheet"
  },
  "raw_class_id": 0,
  "finished_class_id": 1
}
```

## Key Design Decisions

- **ByteTrack over heavier trackers**: Objects on a production line move predictably and look identical. IoU-based association (ByteTrack) outperforms ReID-heavy trackers in this scenario because appearance features are not discriminative. BoostTrack++ further improves on ByteTrack with 40% fewer ID switches in our benchmark (see Tracker Comparison below), making it the preferred choice for per-worker attribution accuracy.
- **Permanent worker assignment**: Once assigned, an object stays with that worker even if it briefly drifts toward another zone. This prevents miscounts from overlapping zones.
- **Class transition as completion signal**: Rather than counting finished products per frame (which double-counts), we count the moment of transition exactly once per tracked object.

## Adapting to Other Use Cases

This system works for any production line where:
- Objects are detected and classified into stages (raw, in-progress, finished)
- Workers operate at fixed positions visible to the camera
- Throughput per worker is the metric of interest

Examples: assembly lines, food processing, packaging stations, quality inspection lanes.

## Tech Stack

- **Detection**: YOLOv11 (via Roboflow Inference SDK or local Ultralytics)
- **Tracking**: ByteTrack (via `supervision`)
- **Visualization**: OpenCV + `supervision` annotators
- **Logging**: CSV with per-worker timestamps

## Demo Scenarios

The `demo/` folder contains line-crossing counting demos that work immediately with any COCO-pretrained model and free stock video.

```bash
# Running event: count runners crossing a finish line
python demo/demo_running_event.py --video videos/running.mp4 --finish-line 0.6

# Retail: count people entering/exiting through a doorway
python demo/demo_retail_counting.py --video videos/retail.mp4 --entry-line 0.5
```

Free videos available from [Pexels](https://www.pexels.com) (search "marathon finish line" or "people walking entrance").

See `demo/README.md` for full details.

## Tracker Comparison

The `comparison/` folder benchmarks SOTA multi-object trackers side-by-side on the same input video, outputting individual annotated videos, a 2x2 grid comparison video, and a metrics CSV.

### Benchmark Results (Running Event, 577 frames, vertical line at 0.9)

![Side-by-side tracker comparison](side_by_side.gif)

| Tracker | Detector | Unique IDs | ID Switches | Count | FPS | Time (s) |
|---------|----------|:----------:|:-----------:|:-----:|:---:|:--------:|
| **BoostTrack++** | YOLOv11x | **13** | **9** | 10 | 2.8 | 207.4 |
| ByteTrack | YOLOv11x | 19 | 15 | 10 | 2.9 | 198.7 |

**Key finding:** Both trackers counted the same 10 runners, but BoostTrack++ maintained significantly better identity preservation -- 13 unique IDs vs 19 (closer to the true ~10 people), with 40% fewer ID switches (9 vs 15). This confirms BoostTrack++'s advantage in reducing spurious track reassignments, which is critical for per-worker attribution in production line scenarios.

### Available Trackers

| Tracker | Detector | Venue | Status |
|---------|----------|-------|--------|
| ByteTrack | YOLOv11x | ECCV 2022 | Ready |
| BoostTrack++ | YOLOv11x | arXiv 2024 | Ready |


```bash
# Run comparison
python comparison/run_comparison.py \
    --video videos/running.mp4 \
    --trackers bytetrack boosttrack \
    --output outputs/comparisons/running/ \
    --count-line 0.9 --line-orientation vertical

# Generate side-by-side video
python comparison/generate_report.py \
    --input outputs/comparisons/running/ \
    --output outputs/comparisons/running/side_by_side.mp4
```

See `comparison/README.md` for full 4-tracker setup instructions.