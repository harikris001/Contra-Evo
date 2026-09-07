"""Baselines for comparison."""
from __future__ import annotations

import numpy as np


class RandomController:
    def __init__(self, seed: int = 0, num_actions: int = 16):
        self.rng = np.random.default_rng(seed)
        self.num_actions = num_actions

    def act(self, obs) -> int:
        return int(self.rng.integers(0, self.num_actions))


class AlwaysRightController:
    def act(self, obs) -> int:
        return 2  # RIGHT


class AlwaysShootController:
    def act(self, obs) -> int:
        return 6  # RIGHT+SHOOT


class ScriptedController:
    """Simple scripted: right+shoot, jump occasionally."""

    def __init__(self):
        self.steps = 0

    def act(self, obs) -> int:
        self.steps += 1
        if self.steps % 50 == 0:
            return 9  # RIGHT+JUMP+SHOOT
        return 6  # RIGHT+SHOOT


def evaluate_controller(controller, env, episodes: int = 5) -> dict:
    """Run controller for N episodes, return stats."""
    fitnesses = []
    for ep in range(episodes):
        obs, info = env.reset(seed=ep)
        done = truncated = False
        # handle both ContraEnv and MockContraEnv get_fitness
        while not (done or truncated):
            act = controller.act(obs)
            obs, rew, done, truncated, info = env.step(act)
        if hasattr(env, "get_fitness"):
            fit = env.get_fitness()
        else:
            fit = float(info.get("progress", 0))
        fitnesses.append(fit)
    return {
        "mean": float(np.mean(fitnesses)),
        "max": float(np.max(fitnesses)),
        "min": float(np.min(fitnesses)),
        "all": fitnesses,
    }
