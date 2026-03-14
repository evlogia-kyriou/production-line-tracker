# Demo Scenarios

These demos showcase the tracking and counting pipeline on publicly available video sources. They share the same core architecture as the main production line tracker but use **line-crossing counting** instead of class-transition counting.

## Demo 1: Running Event (Finish Line Counter)

Counts runners crossing a virtual finish line. Demonstrates ROI filtering, ByteTrack persistence through occlusion (runners in packs), and real-time counting.

```bash
# Download a free running video from Pexels, e.g.:
# https://www.pexels.com/search/videos/marathon%20finish%20line/

python demo/demo_running_event.py --video videos/running.mp4

# Custom finish line position (70% from top)
python demo/demo_running_event.py --video videos/running.mp4 --finish-line 0.7

# Save output
python demo/demo_running_event.py --video videos/running.mp4 --output outputs/running_demo.mp4
```

**What it shows:**
- Person detection (COCO pretrained YOLOv11)
- ByteTrack maintaining IDs through runner overlap
- Line-crossing counting with direction awareness
- Real-time stats overlay

## Demo 2: Retail People Counter (Entry/Exit)

Counts people entering and exiting through a doorway. Tracks current occupancy (entered minus exited). Demonstrates bidirectional counting and occupancy analytics.

```bash
# Download a free pedestrian video from Pexels, e.g.:
# https://www.pexels.com/search/videos/people%20walking%20entrance/

python demo/demo_retail_counting.py --video videos/retail.mp4

# Vertical entry line (for side-view camera)
python demo/demo_retail_counting.py --video videos/retail.mp4 --line-orientation vertical --entry-line 0.4

# Save output
python demo/demo_retail_counting.py --video videos/retail.mp4 --output outputs/retail_demo.mp4
```

**What it shows:**
- Bidirectional counting (in vs out)
- Real-time occupancy (people currently inside)
- Works with horizontal or vertical entry lines
- Adaptable to any doorway or passage

## Free Video Sources

| Source | URL | License |
|--------|-----|---------|
| Pexels | pexels.com/videos | Free for commercial use |
| Videezy | videezy.com | CC / Royalty-free |
| Pixabay | pixabay.com/videos | Free for commercial use |
| MOT17 benchmark | motchallenge.net | Research use |

## Architecture

Both demos use:
- `ultralytics` YOLO for detection
- `supervision` ByteTrack for tracking
- `demo/line_counter.py` for line-crossing logic (shared)
- OpenCV for visualization and video I/O

The main pipeline (`src/pipeline.py`) uses **class-transition counting** instead of line-crossing, which is required for scenarios where objects change form (e.g., dough_ball to matzah_sheet on a production line).