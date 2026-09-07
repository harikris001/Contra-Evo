"""Resource monitoring: RAM, CPU, throughput."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict

import psutil


@dataclass
class ResourceStats:
    ram_percent: float = 0.0
    ram_used_mb: float = 0.0
    ram_total_mb: float = 0.0
    cpu_percent: float = 0.0
    episodes_per_sec: float = 0.0
    emulator_fps: float = 0.0
    generation_duration: float = 0.0


class ResourceMonitor:
    """Poll psutil for system stats. Lightweight, non-blocking."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self._last_time = time.time()
        self._episode_count = 0
        self._step_count = 0
        self._last_episode_time = time.time()
        # prime cpu_percent
        psutil.cpu_percent(interval=None)

    def poll(self) -> ResourceStats:
        vm = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=None)
        now = time.time()
        dt = max(now - self._last_time, 1e-6)
        eps = self._episode_count / dt if dt > 1 else 0.0
        fps = self._step_count / dt if dt > 1 else 0.0
        return ResourceStats(
            ram_percent=vm.percent,
            ram_used_mb=vm.used / 1024 / 1024,
            ram_total_mb=vm.total / 1024 / 1024,
            cpu_percent=cpu,
            episodes_per_sec=eps,
            emulator_fps=fps,
        )

    def reset_counters(self) -> None:
        self._last_time = time.time()
        self._episode_count = 0
        self._step_count = 0

    def record_episode(self, steps: int = 1) -> None:
        self._episode_count += 1
        self._step_count += steps

    def record_steps(self, n: int) -> None:
        self._step_count += n

    def check_limits(self, max_ram_percent: float = 85.0) -> bool:
        """Return True if under limit, False if over."""
        vm = psutil.virtual_memory()
        return vm.percent < max_ram_percent


def estimate_memory_mb(frame_size: int = 84, frame_stack: int = 4, population: int = 8, workers: int = 4) -> Dict[str, float]:
    """Estimate memory footprint to avoid OOM on 16GB."""
    # uint8 frame: 84*84*4 ~ 28k per obs, float32 would be 4x
    obs_mb = (frame_size * frame_size * frame_stack) / 1024 / 1024 * population
    # overhead per env ~20MB (retro)
    env_mb = workers * 30
    total = obs_mb + env_mb + 200  # base
    return {"obs_mb": obs_mb, "env_mb": env_mb, "total_est_mb": total}
