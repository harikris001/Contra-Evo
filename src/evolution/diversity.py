"""Population diversity measurement."""
from __future__ import annotations

import numpy as np

from agent.genome import Individual


def population_diversity(population: list[Individual]) -> float:
    """Measure diversity as mean pairwise std or mean distance to centroid.

    Returns normalized 0-1 approx: std of genomes.
    Low diversity ~ <0.05 indicates convergence.

    NOTE: raw per-gene std is dominated by large-scale layers (e.g. the big
    dense matrix). Prefer population_diversity_normalized when a scale vector
    is available.
    """
    if len(population) < 2:
        return 0.0
    # stack vectors
    mat = np.stack([ind.genome.vector for ind in population], axis=0)  # (pop, dim)
    # mean pairwise distance approximation via std per gene
    std_per_gene = np.std(mat, axis=0)
    diversity = float(np.mean(std_per_gene))
    return diversity


def population_diversity_normalized(population: list[Individual], scales: np.ndarray) -> float:
    """Mean per-gene std normalized by each gene's init scale.

    ~1.0 for freshly random populations, ->0 as the population converges.
    Threshold comparisons (diversity_threshold) are meaningful on this scale.
    Falls back to unnormalized std if scales don't match the genome.
    """
    if len(population) < 2:
        return 0.0
    mat = np.stack([ind.genome.vector for ind in population], axis=0)
    scales = np.asarray(scales, dtype=np.float32)
    if scales.size != mat.shape[1]:
        return population_diversity(population)
    std_per_gene = np.std(mat, axis=0)
    return float(np.mean(std_per_gene / np.maximum(scales, 1e-8)))


def euclidean_diversity(population: list[Individual]) -> float:
    if len(population) < 2:
        return 0.0
    mat = np.stack([ind.genome.vector for ind in population], axis=0)
    centroid = np.mean(mat, axis=0)
    dists = np.linalg.norm(mat - centroid, axis=1)
    return float(np.mean(dists))
