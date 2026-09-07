"""Selection operators."""
from __future__ import annotations

import numpy as np

from agent.genome import Individual


def tournament_selection(
    population: list[Individual],
    tournament_size: int = 3,
    rng: np.random.Generator | None = None,
) -> Individual:
    """Select one individual via tournament."""
    rng = rng or np.random.default_rng()
    # sample without replacement if possible
    k = min(tournament_size, len(population))
    contestants = rng.choice(population, size=k, replace=False)
    # highest fitness wins
    best = max(contestants, key=lambda ind: ind.fitness)
    return best


def elite_selection(population: list[Individual], elite_count: int = 2) -> list[Individual]:
    """Return top elite_count individuals sorted by fitness descending."""
    sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)
    return sorted_pop[:elite_count]


def roulette_selection(population: list[Individual], rng: np.random.Generator | None = None) -> Individual:
    """Fitness-proportionate. Handles negative fitness by shifting."""
    rng = rng or np.random.default_rng()
    fitnesses = np.array([ind.fitness for ind in population], dtype=np.float64)
    # shift to positive
    min_f = np.min(fitnesses)
    if min_f < 0:
        fitnesses = fitnesses - min_f + 1e-6
    total = np.sum(fitnesses)
    if total == 0:
        return rng.choice(population)
    probs = fitnesses / total
    return rng.choice(population, p=probs)
