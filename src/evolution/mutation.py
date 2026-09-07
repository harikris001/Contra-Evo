"""Mutation operator."""
from __future__ import annotations

import numpy as np

from agent.genome import Genome


def mutate(
    genome: Genome,
    mutation_rate: float = 0.05,
    mutation_std: float = 0.02,
    rng: np.random.Generator | None = None,
    scales: np.ndarray | None = None,
) -> Genome:
    """Gaussian mutation per-gene with probability mutation_rate.

    If `scales` (per-gene init-scale vector from parameter_scale_vector) is
    provided, `mutation_std` is a RELATIVE multiplier of each gene's layer
    scale, so every layer is perturbed proportionally to its weight magnitude.
    Otherwise `mutation_std` is an absolute sigma (legacy behavior).
    """
    rng = rng or np.random.default_rng()
    mask = rng.random(genome.vector.size) < mutation_rate
    if not np.any(mask):
        return genome
    noise = rng.standard_normal(genome.vector.size).astype(np.float32)
    if scales is not None:
        scales = np.asarray(scales, dtype=np.float32)
        if scales.size == genome.vector.size:
            noise *= mutation_std * scales
        else:
            noise *= np.float32(mutation_std)
    else:
        noise *= np.float32(mutation_std)
    new_vec = genome.vector.copy()
    new_vec[mask] += noise[mask]
    return Genome(vector=new_vec, generation=genome.generation, parent_ids=genome.parent_ids)


def adaptive_mutation_rate(
    base_rate: float,
    diversity: float,
    threshold: float = 0.05,
    boost: float = 1.5,
) -> float:
    """Increase mutation if diversity too low."""
    if diversity < threshold:
        return min(base_rate * boost, 0.5)
    return base_rate


def adaptive_mutation_std(
    base_std: float,
    stall_count: int,
    stall_generations: int = 5,
    std_min: float = 0.01,
    std_max: float | None = None,
    boost: float = 1.5,
) -> float:
    """Increase mutation step size when best fitness has stalled.

    Capped at `std_max` (defaults to 2x base) so stall responses stay a local
    search instead of destroying children with near-init-scale noise.
    """
    if std_max is None:
        std_max = base_std * 2.0
    if stall_generations <= 0 or stall_count < stall_generations:
        return float(np.clip(base_std, std_min, std_max))
    steps = stall_count // stall_generations
    return float(min(base_std * (boost**steps), std_max))
