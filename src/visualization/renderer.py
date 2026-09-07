"""Renderer: periodically grabs framebuffers without blocking training.

Design: emulator workers publish latest frame/state to shared queue/dict.
Renderer runs at 10-20 FPS independently.
"""
from __future__ import annotations

import time
from typing import Dict, Any, Callable
from collections import deque

import numpy as np


class FrameBuffer:
    """Thread-safe-ish latest frame store (single-process fallback uses dict).

    For multiprocessing, we'll use Manager.dict or Queue.
    Here we provide simple in-memory for single-process / testing.
    """

    def __init__(self, population_size: int = 8):
        self.population_size = population_size
        self.frames: Dict[int, np.ndarray | None] = {i: None for i in range(population_size)}
        self.states: Dict[int, Dict[str, Any]] = {i: {} for i in range(population_size)}
        self.timestamps: Dict[int, float] = {i: 0 for i in range(population_size)}

    def update(self, agent_id: int, frame: np.ndarray | None, state: Dict[str, Any]) -> None:
        self.frames[agent_id] = frame
        self.states[agent_id] = dict(state)
        self.timestamps[agent_id] = time.time()

    def get(self, agent_id: int) -> tuple[np.ndarray | None, Dict[str, Any]]:
        return self.frames.get(agent_id), self.states.get(agent_id, {})


class Renderer:
    """Polls FrameBuffer at fixed FPS and calls draw callback."""

    def __init__(self, buffer: FrameBuffer, fps: int = 15, draw_fn: Callable | None = None):
        self.buffer = buffer
        self.fps = fps
        self.draw_fn = draw_fn
        self._running = False
        self._frame_times: deque[float] = deque(maxlen=30)

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def tick(self) -> float:
        """Sleep to maintain fps, return actual fps."""
        now = time.time()
        self._frame_times.append(now)
        if len(self._frame_times) >= 2:
            dt = self._frame_times[-1] - self._frame_times[0]
            actual = (len(self._frame_times) - 1) / dt if dt > 0 else 0
        else:
            actual = 0
        time.sleep(max(0, 1.0 / self.fps - 0.005))
        return actual

    def render_once(self, global_state: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Collect snapshot for dashboard."""
        snapshot: Dict[str, Any] = {"agents": {}, "global": global_state or {}}
        for aid in range(self.buffer.population_size):
            frame, state = self.buffer.get(aid)
            snapshot["agents"][aid] = {"frame": frame, "state": state}
        if self.draw_fn:
            self.draw_fn(snapshot)
        return snapshot
