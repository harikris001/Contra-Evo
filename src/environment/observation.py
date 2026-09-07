"""Observation processing: pixels -> grayscale -> resize -> stack, and features mode."""
from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np

try:
    import cv2  # type: ignore

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert HWC RGB frame to grayscale. Input uint8."""
    if frame.ndim == 2:
        return frame
    if frame.shape[2] == 3:
        # Use luminosity if cv2 not available
        if HAS_CV2:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            # numpy weighted
            return np.dot(frame[..., :3], [0.2989, 0.587, 0.114]).astype(np.uint8)
    return frame[:, :, 0]


def resize_frame(frame: np.ndarray, size: int = 84) -> np.ndarray:
    if frame.shape[0] == size and frame.shape[1] == size:
        return frame
    if HAS_CV2:
        return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    else:
        # naive nearest - prefer cv2
        # fallback: use numpy resize via PIL if available
        try:
            from PIL import Image

            im = Image.fromarray(frame)
            im = im.resize((size, size), Image.BILINEAR)
            return np.array(im, dtype=np.uint8)
        except ImportError:
            # crude
            h, w = frame.shape[:2]
            return frame[:: h // size, :: w // size][:size, :size]


def process_frame(frame: np.ndarray, size: int = 84, grayscale: bool = True) -> np.ndarray:
    """Single frame processing -> 84x84 uint8."""
    if grayscale:
        frame = to_grayscale(frame)
    else:
        if frame.ndim == 3:
            frame = to_grayscale(frame)
    frame = resize_frame(frame, size)
    return frame.astype(np.uint8)


class FrameStack:
    """Keep last N frames, produce stacked observation uint8.

    Keeps memory small: uses deque of uint8 (84x84).
    Output shape: (84,84,4) or (4,84,84) depending on channel_first.
    We default to (4,84,84)? Spec says 84x84x4, we'll provide configurable.
    For CNN, we output (stack, H, W) as (4,84,84) uint8.

    flicker_pool > 1 takes an element-wise max over the most recent
    `flicker_pool` raw frames before pushing into the stack. NES games
    (Contra included) flicker sprites when too many are on screen; max
    pooling keeps the player/enemies visible instead of intermittently
    vanishing from the observation.
    """

    def __init__(
        self,
        stack: int = 4,
        size: int = 84,
        grayscale: bool = True,
        channel_first: bool = True,
        flicker_pool: int = 1,
    ):
        self.stack = stack
        self.size = size
        self.grayscale = grayscale
        self.channel_first = channel_first
        self.flicker_pool = max(1, int(flicker_pool))
        self.frames: Deque[np.ndarray] = deque(maxlen=stack)
        self._recent: Deque[np.ndarray] = deque(maxlen=self.flicker_pool)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        self._recent.clear()  # no flicker-pool leakage across episodes (cached envs)
        proc = self._pool_frame(frame)
        self.frames.clear()
        for _ in range(self.stack):
            self.frames.append(proc.copy())
        return self.get()

    def push(self, frame: np.ndarray) -> np.ndarray:
        proc = self._pool_frame(frame)
        self.frames.append(proc)
        return self.get()

    def _pool_frame(self, frame: np.ndarray) -> np.ndarray:
        proc = process_frame(frame, self.size, self.grayscale)
        if self.flicker_pool <= 1:
            return proc
        self._recent.append(proc)
        if len(self._recent) == 1:
            return proc
        return np.maximum.reduce(list(self._recent))

    def get(self) -> np.ndarray:
        assert len(self.frames) == self.stack
        stacked = np.stack(list(self.frames), axis=0)  # (4,84,84)
        if not self.channel_first:
            stacked = np.transpose(stacked, (1, 2, 0))  # (84,84,4)
        return stacked

    def get_float(self) -> np.ndarray:
        """Normalized float32 0-1 for network if needed."""
        return self.get().astype(np.float32) / 255.0


def extract_features(info: dict) -> np.ndarray:
    """Lightweight structured features for debugging.

    If retro info dict has keys, extract; else dummy zeros.
    Returns float32 vector normalized.
    """
    # Expected Contra RAM values: x_pos, y_pos, lives, etc - depends on ROM
    # We'll be defensive: if info empty, return zeros
    # Let's define fixed 10-dim feature vector
    feats = np.zeros(10, dtype=np.float32)
    # Try common keys
    mapping = ["x_pos", "y_pos", "lives", "score", "level_scroll", "level_screen", "stage", "game_routine", "x", "kills"]
    for i, k in enumerate(mapping):
        for key in info.keys():
            if k.lower() in key.lower():
                v = info[key]
                try:
                    feats[i] = float(v)
                except:
                    pass
                break
    # Normalize heuristic
    feats[0] /= 3000.0  # x progress
    feats[1] /= 240.0
    feats[2] /= 3.0
    feats[3] /= 10000.0
    feats[4] /= 3000.0
    feats[5] /= 10.0
    feats[6] /= 100.0
    feats[7] /= 10.0
    feats[8] /= 10.0
    feats[9] /= 300.0
    return np.clip(feats, -5, 5)
