# MediaPipe Hand Landmarker wrapper / Tasks API
# Owns all interaction with MediaPipe. Nothing else in this codebase should
# import mediapipe directly, go through HandTracker
# so the wild garv or the ui guys never has to know how landmark extraction works internally

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"


def _ensure_model() -> Path:
    # Download the hand_landmarker.task model file if not already cached
    if not _MODEL_PATH.exists():
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading hand landmark model to {_MODEL_PATH} ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


@dataclass
class HandResult:
    # A single detected hand's 21 landmarks.
    #
    # landmarks: shape (21, 3) array of (x, y, z) in normalized image
    # coordinates (x, y in [0, 1] relative to image width/height; z is
    # roughly depth relative to the wrist, smaller = closer to camera).
    # handedness: "Left" or "Right" as reported by MediaPipe.
    # confidence: MediaPipe's handedness classification score, 0-1.
    
    landmarks: np.ndarray
    handedness: str
    confidence: float


class HandTracker:
    # Typed wrapper around mediapipe.tasks.vision.HandLandmarker
    #
    # mode="video" (default): for a live webcam loop -- call process() once
    # per frame in order, timestamps are tracked internally
    # mode="image": for batch-processing independent static images 
    # where there's no meaningful frame ordering.
    #
    # Usage:
    #     tracker = HandTracker(mode="video")
    #     results = tracker.process(frame_bgr)  # list[HandResult], len 0-2
    #     tracker.close()
    #
    # Or as a context manager:
    #     with HandTracker(mode="image") as tracker:
    #         results = tracker.process(image_bgr)
     
    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        mode: Literal["video", "image"] = "video",
    ) -> None:
        model_path = _ensure_model()
        self._mode = mode
        running_mode = vision.RunningMode.VIDEO if mode == "video" else vision.RunningMode.IMAGE
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=running_mode,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._frame_index = 0

    def process(self, frame_bgr: np.ndarray) -> list[HandResult]:
        # Run hand detection on a single BGR frame as read by cv2
        #
        # Returns an empty list if no hand is detected -- callers must handle
        # that case 
        
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])  # BGR -> RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self._mode == "video":
            timestamp_ms = int(self._frame_index * (1000 / 30))  # assume ~30fps
            self._frame_index += 1
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return []

        out: list[HandResult] = []
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float64)
            if i < len(result.handedness):
                cls = result.handedness[i][0]
                label, score = cls.category_name, cls.score
            else:
                label, score = "Unknown", 0.0
            out.append(HandResult(landmarks=coords, handedness=label, confidence=score))
        return out

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
