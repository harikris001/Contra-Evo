"""Curriculum manager."""
from __future__ import annotations

from typing import Dict, Any, List

from environment.scenarios import get_scenario


class Curriculum:
    def __init__(
        self,
        stages: List[str],
        enabled: bool = False,
        advance_threshold: float = 80.0,
        advance_metric: str = "progress",
        advance_patience: int = 2,
    ):
        self.enabled = enabled
        self.stages = stages
        self.advance_threshold = advance_threshold
        self.advance_metric = advance_metric
        # consecutive generations the metric must exceed the threshold before
        # advancing (prevents a single lucky episode from racing the stages)
        self.advance_patience = max(1, int(advance_patience))
        self._above_count = 0
        self.current_idx = 0

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "Curriculum":
        return cls(
            stages=cfg.get("stages", ["movement", "full_level_1"]),
            enabled=cfg.get("enabled", False),
            advance_threshold=float(cfg.get("advance_threshold", 80.0)),
            advance_metric=str(cfg.get("advance_metric", "progress")),
            advance_patience=int(cfg.get("advance_patience", 2)),
        )

    def current(self) -> str:
        if not self.enabled or not self.stages:
            return "full_level_1"
        return self.stages[min(self.current_idx, len(self.stages) - 1)]

    def maybe_advance(
        self,
        best_fitness: float,
        best_progress: float | None = None,
        threshold: float | None = None,
        best_kills: float | None = None,
    ) -> bool:
        if not self.enabled:
            return False
        thr = threshold if threshold is not None else self.advance_threshold
        sc = get_scenario(self.current())
        if getattr(sc, "progress_threshold", None) is not None:
            thr = float(sc.progress_threshold)
        metric = best_progress if self.advance_metric == "progress" and best_progress is not None else best_fitness
        if metric is None or metric <= thr:
            self._above_count = 0
            return False
        kill_thr = float(getattr(sc, "kill_threshold", 0) or 0)
        if kill_thr > 0 and (best_kills is None or best_kills < kill_thr):
            self._above_count = 0
            return False
        self._above_count += 1
        if self._above_count < self.advance_patience:
            return False
        if self.current_idx < len(self.stages) - 1:
            self.current_idx += 1
            self._above_count = 0
            return True
        return False

    def get_scenario_config(self) -> Dict[str, Any]:
        name = self.current() if self.enabled else (self.stages[0] if self.stages else "full_level_1")
        if not self.enabled:
            # When disabled, honor whatever name the env config already uses via caller
            name = self.current()
        sc = get_scenario(name)
        return {
            "scenario": sc.name,
            "max_steps": sc.max_steps,
            "reward_overrides": sc.reward_overrides,
            "progress_threshold": getattr(sc, "progress_threshold", None),
        }

    def apply_to_env(self, env_config: Dict[str, Any], reward_config: Dict[str, Any] | None = None) -> None:
        """Mutate env/reward dicts for the current stage."""
        if not self.enabled:
            return
        sc = get_scenario(self.current())
        env_config["scenario"] = sc.name
        env_config["max_episode_steps"] = sc.max_steps
        if reward_config is not None:
            reward_config.update(sc.reward_overrides)
