"""Configuration loading and validation.

No hardcoded values - all from YAML with defaults.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULTS: Dict[str, Any] = {
    "env": {
        "rom_path": None,
        "scenario": "movement",
        "observation_mode": "pixels",
        "frame_stack": 4,
        "frame_size": 84,
        "grayscale": True,
        "max_episode_steps": 5000,
        "action_repeat": 4,
        "flicker_pool": 2,
    },
    "actions": {"space": "playable"},
    "reward": {
        "progress_coef": 6.0,
        "kill_coef": 20.0,
        "boss_damage_coef": 5.0,
        "survival_coef": 0.001,
        "death_penalty": -50.0,
        "idle_penalty": -4.0,
        "level_complete_bonus": 1000.0,
        "idle_threshold_steps": 24,
        "idle_timeout_steps": 320,
        "idle_min_progress": 2.0,
        "idle_combat_scale": 0.05,
        "idle_forfeit_scale": 1.0,
        "idle_progress_ref": 64.0,
        "terminate_on_death": True,
        "death_forfeit_progress": False,
        "death_progress_scale": 1.0,
        "score_coef": 0.0,
        "no_combat_progress_scale": 0.2,
        "combat_grace_progress": 64.0,
        "score_points_per_kill": 100.0,
        "max_score_delta_per_step": 800.0,
        "max_kills_from_score_per_step": 4,
        "fall_y_threshold": 6.0,
        "fall_no_water_penalty": -2.0,
        "down_no_water_penalty": -0.5,
        "climb_bonus": 0.5,
    },
    "agent": {
        "cnn_channels": [16, 32],
        "kernel_sizes": [8, 4],
        "strides": [4, 2],
        "dense_hidden": [64, 32],
        "activation": "relu",
        "param_limit": 100000,
    },
    "evolution": {
        "population_size": 16,
        "elite_count": 2,
        "tournament_size": 3,
        "mutation_rate": 0.04,
        "mutation_std": 0.08,
        "crossover_rate": 0.5,
        "crossover_type": "blend",
        "adaptive_mutation": True,
        "diversity_threshold": 0.05,
        "adaptive_boost": 1.5,
        "immigrant_frac": 0.1,
        "stall_generations": 5,
        "stall_immigrant_frac": 0.25,
        "mutation_std_min": 0.02,
        "mutation_std_max": 0.16,
        "reevaluate_elites": False,
        "stall_rel_epsilon": 0.001,
        "init_genome": None,
        "seed": 42,
    },
    "evaluation": {
        "workers": 8,
        "vectorized": False,
        "seed_policy": "fixed",
        "seeds_per_individual": 1,
    },
    "training": {
        "generations": 100,
        "mode": "training",
        "checkpoint_interval": 5,
        "save_best": True,
        "headless": True,
    },
    "curriculum": {
        "enabled": False,
        "advance_metric": "progress",
        "advance_threshold": 80.0,
        "advance_patience": 2,
    },
    "visualization": {"enabled": False, "fps": 15, "layout": "auto", "show_charts": True},
    "logging": {"log_interval": 1},
    "resources": {"monitor_interval": 1.0, "max_ram_percent": 85.0},
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: str | Path | None = None, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Load YAML config and merge with defaults.

    Args:
        path: path to yaml file
        overrides: dict overrides (e.g. CLI)
    """
    cfg = DEFAULTS
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        with open(p) as f:
            data = yaml.safe_load(f) or {}
        cfg = deep_merge(cfg, data)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    # env var expansion for rom_path
    rom = cfg.get("env", {}).get("rom_path")
    if isinstance(rom, str):
        cfg["env"]["rom_path"] = os.path.expanduser(os.path.expandvars(rom))
    _validate(cfg)
    return cfg


def _validate(cfg: Dict[str, Any]) -> None:
    ev = cfg["evolution"]
    assert 1 <= ev["population_size"] <= 1024, "population_size out of range"
    assert 0 <= ev["elite_count"] < ev["population_size"]
    assert 0 < ev["mutation_rate"] <= 1.0
    assert ev["tournament_size"] >= 2
    assert cfg["env"]["frame_stack"] >= 1
    assert cfg["env"]["frame_size"] in (42, 84, 96, 128)
    assert cfg["env"]["observation_mode"] in ("pixels", "features")
    assert cfg["training"]["mode"] in ("debug", "training", "live_training")
