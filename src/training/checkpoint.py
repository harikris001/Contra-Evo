"""Checkpoint save/load."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np
import yaml


def save_checkpoint(
    path: Path,
    generation: int,
    population: list,
    fitnesses: list[float],
    best_genome: np.ndarray,
    ga_config: Dict[str, Any],
    env_config: Dict[str, Any],
    net_config: Dict[str, Any],
    metrics: list[Dict[str, Any]],
    rng_state: Any | None = None,
    ga_state: Dict[str, Any] | None = None,
    curriculum_stage: int | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "generation": generation,
        "population": [ind.genome.vector for ind in population],
        "fitnesses": fitnesses,
        "best_genome": best_genome,
        "ga_config": ga_config,
        "env_config": env_config,
        "net_config": net_config,
        "metrics": metrics,
        "rng_state": rng_state,
        "ga_state": ga_state or {},
        "curriculum_stage": curriculum_stage,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    # also save best.npy
    np.save(path.with_suffix(".best.npy"), best_genome)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data
