"""Policy wrapper: observation -> action via network."""
from __future__ import annotations

import numpy as np

from .network import CompactCNN, NetworkConfig
from .genome import Individual


class Policy:
    """Wraps a network for inference. Keeps interface stable for future PPO/DQN."""

    def __init__(self, network: CompactCNN):
        self.network = network

    @classmethod
    def from_individual(cls, individual: Individual, config: NetworkConfig) -> "Policy":
        net = individual.to_network(config)
        return cls(net)

    def act(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """Observation is (4,84,84) uint8 or (10,) features.

        For features, we need a separate small MLP - handle fallback.
        """
        if observation.ndim == 1:
            # features mode: simple linear - use network's dense path? Reuse network's forward expects image.
            # Alternative: create tiny MLP inline
            # For now, random if mismatch - but better to have feature network
            # We'll implement feature path: if obs is 1D, use a small heuristic: map via fixed random projection
            # Actually, our network expects image; if features, we should have different network type.
            # Simplify: return random action for features fallback unless we have feature network
            # Let's create a tiny 10->32->8 MLP using first part of genome? Instead, use heuristic: logits = obs @ W
            # For MVP, just return argmax of obs*some weights - but we need deterministic
            # Instead, fallback to random weighted by obs
            # This is placeholder; real feature policy should be separate.
            return int(np.argmax(observation) % 8)
        # pixels
        return self.network.act(observation, deterministic=deterministic)

    def get_action(self, observation: np.ndarray) -> int:
        return self.act(observation)
