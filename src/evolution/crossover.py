"""Crossover operators on flat genome vectors."""
from __future__ import annotations

import numpy as np

from agent.genome import Genome


def uniform_crossover(
    parent_a: Genome,
    parent_b: Genome,
    crossover_rate: float = 0.5,
    rng: np.random.Generator | None = None,
) -> Genome:
    """Uniform crossover: each gene from A with p=0.5 else B. If random > crossover_rate, return copy of A."""
    rng = rng or np.random.default_rng()
    if rng.random() > crossover_rate:
        return parent_a.copy()
    mask = rng.random(parent_a.vector.size) < 0.5
    child_vec = np.where(mask, parent_a.vector, parent_b.vector).astype(np.float32)
    return Genome(vector=child_vec, generation=max(parent_a.generation, parent_b.generation) + 1, parent_ids=(id(parent_a), id(parent_b)))


def single_point_crossover(
    parent_a: Genome,
    parent_b: Genome,
    rng: np.random.Generator | None = None,
) -> Genome:
    rng = rng or np.random.default_rng()
    point = rng.integers(1, parent_a.vector.size)
    child_vec = np.concatenate([parent_a.vector[:point], parent_b.vector[point:]]).astype(np.float32)
    return Genome(vector=child_vec, generation=max(parent_a.generation, parent_b.generation) + 1)


def blend_crossover(
    parent_a: Genome,
    parent_b: Genome,
    alpha: float | None = None,
    rng: np.random.Generator | None = None,
) -> Genome:
    """Blend: child = alpha*A + (1-alpha)*B.

    alpha=None samples a per-child BLX-style coefficient ~ U(-0.5, 1.5) so
    children can extrapolate beyond the parents. A fixed alpha=0.5 midpoint is
    contractive: it halves variance every generation and collapses diversity.
    """
    rng = rng or np.random.default_rng()
    if alpha is None:
        alpha = float(rng.uniform(-0.5, 1.5))
    child_vec = (alpha * parent_a.vector + (1 - alpha) * parent_b.vector).astype(np.float32)
    return Genome(vector=child_vec, generation=max(parent_a.generation, parent_b.generation) + 1)
