"""Genome representation: flat weight vector + metadata."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any

import numpy as np

from .network import CompactCNN, NetworkConfig, count_parameters


@dataclass
class Genome:
    vector: np.ndarray  # float32 flat
    generation: int = 0
    parent_ids: tuple[int, ...] = field(default_factory=tuple)

    def copy(self) -> "Genome":
        return Genome(vector=self.vector.copy(), generation=self.generation, parent_ids=self.parent_ids)

    def size(self) -> int:
        return self.vector.size


@dataclass
class Individual:
    """Population member."""
    id: int
    genome: Genome
    fitness: float = -np.inf
    stats: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending | evaluating | evaluated | elite | dead

    def to_network(self, config: NetworkConfig) -> CompactCNN:
        net = CompactCNN(config)
        # Ensure genome size matches
        expected = count_parameters(config)
        assert self.genome.vector.size == expected, f"genome {self.genome.vector.size} vs expected {expected}"
        net.set_genome(self.genome.vector)
        return net


def random_genome(config: NetworkConfig, rng: np.random.Generator, generation: int = 0) -> Genome:
    net = CompactCNN(config, rng=rng)
    # net already randomly initialized
    vec = net.get_genome()
    return Genome(vector=vec, generation=generation)
