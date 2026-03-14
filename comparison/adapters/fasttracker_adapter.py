"""
FastTracker Adapter: Wraps the FastTracker pipeline (YOLOX + occlusion-aware tracking).

SETUP REQUIRED:
    1. Clone: git clone https://github.com/Hamidreza-Hashempoor/FastTracker.git external/FastTracker
    2. Download YOLOX-X weights (bytetrack_x_mot17.pth.tar) into models/detectors/
    3. Follow FastTracker repo setup instructions:
       cd external/FastTracker && python setup.py develop

This adapter imports FastTracker's modules and wraps them into the standard interface.
Adjust import paths as needed based on FastTracker's actual code structure.
"""

import cv2
import sys
import numpy as np
from pathlib import Path

from .base_tracker import BaseTrackerAdapter, TrackedDetection


class FastTrackerAdapter(BaseTrackerAdapter):

    def __init__(self):
        self.detector = None
        self.tracker = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "FastTracker"

    @property
    def detector_name(self) -> str:
        return "YOLOX-X"

    def load(self, config: dict):
        """Load FastTracker pipeline.

        Expected config keys:
            fasttracker_root: Path to cloned FastTracker repo
            detector_weights: Path to YOLOX-X weights (bytetrack_x_mot17.pth.tar)
            exp_file: Path to YOLOX experiment file
            confidence: Detection confidence threshold
            device: "cuda:0" or "cpu"
        """
        fasttracker_root = Path(config.get("fasttracker_root", "external/FastTracker"))

        if not fasttracker_root.exists():
            raise FileNotFoundError(
                f"FastTracker repo not found at {fasttracker_root}. "
                f"Clone it: git clone https://github.com/Hamidreza-Hashempoor/FastTracker.git {fasttracker_root}"
            )

        sys.path.insert(0, str(fasttracker_root))

        self.confidence = config.get("confidence", 0.3)
        self.device = config.get("device", "cuda:0")

        # ---------------------------------------------------------------
        # TODO: Import and initialize FastTracker components
        #
        # FastTracker is built on top of ByteTrack's codebase.
        # The tracking logic lives in their custom tracker files.
        #
        # Rough pattern:
        #
        # from yolox.exp import get_exp
        # exp = get_exp("exps/example/mot/yolox_x_mix_det.py", None)
        # self.detector = exp.get_model()
        # ... load weights from bytetrack_x_mot17.pth.tar ...
        #
        # from tools.fasttracker import FastTracker as FT
        # self.tracker = FT(args)
        #
        # Or with class-aware motion prediction:
        # from tools.fasttracker_cls import FastTracker as FT
        # ---------------------------------------------------------------

        self._loaded = True
        print(f"[FastTracker] Loaded from {fasttracker_root}")
        print(f"[FastTracker] NOTE: You need to complete the imports in fasttracker_adapter.py")

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[TrackedDetection]]:
        if not self._loaded:
            raise RuntimeError("FastTracker not loaded. Call load() first.")

        # ---------------------------------------------------------------
        # TODO: Implement the actual FastTracker inference pipeline
        #
        # Rough pattern:
        #
        # 1. Preprocess frame for YOLOX
        #    img, ratio = preprocess(frame, input_size)
        #
        # 2. Run YOLOX detection
        #    outputs = self.detector(img_tensor)
        #    dets = postprocess(outputs, num_classes=1, ...)
        #
        # 3. Run FastTracker association (with occlusion handling)
        #    tracks = self.tracker.update(dets, frame_size)
        #
        # 4. Convert to TrackedDetection list + annotate frame
        # ---------------------------------------------------------------

        tracked = []
        annotated = frame.copy()

        # Placeholder
        cv2.putText(annotated, "FastTracker: integration pending", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return annotated, tracked

    def reset(self):
        # TODO: Reset FastTracker's internal state
        pass