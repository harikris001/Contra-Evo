"""Curriculum scenarios / starting states."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class Scenario:
    name: str
    description: str
    state_file: str | None  # retro .state
    max_steps: int
    reward_overrides: Dict[str, float]
    progress_threshold: float = 80.0
    kill_threshold: float = 0.0


SCENARIOS: Dict[str, Scenario] = {
    "movement": Scenario(
        name="movement",
        description="Rightward progress only, no shooting needed",
        state_file="level1_start.state",  # skips title screen (capture via scripts/capture_states.py)
        max_steps=1000,
        reward_overrides={},
        progress_threshold=80.0,
        kill_threshold=0,
    ),
    "movement_shooting": Scenario(
        name="movement_shooting",
        description="Movement + shooting",
        state_file="level1_start.state",
        max_steps=1500,
        reward_overrides={},
        progress_threshold=150.0,
        kill_threshold=2,
    ),
    "enemies": Scenario(
        name="enemies",
        description="First enemy wave",
        state_file="level1_enemies.state",
        max_steps=2000,
        reward_overrides={},
        progress_threshold=250.0,
        kill_threshold=4,
    ),
    "projectiles": Scenario(
        name="projectiles",
        description="Enemy projectiles / dodging",
        state_file="level1_projectiles.state",
        max_steps=2000,
        reward_overrides={},
        progress_threshold=400.0,
        kill_threshold=6,
    ),
    "extended_level1": Scenario(
        name="extended_level1",
        description="Extended Level 1 section (starts at screen 8, before the cliff gap)",
        state_file="level1_cliff.state",
        max_steps=3000,
        reward_overrides={},
        progress_threshold=600.0,
        kill_threshold=8,
    ),
    "full_level_1": Scenario(
        name="full_level_1",
        description="Full Level 1",
        state_file=None,
        max_steps=5000,
        reward_overrides={},
        progress_threshold=1200.0,
        kill_threshold=10,
    ),
    "boss": Scenario(
        name="boss",
        description="Boss fight",
        state_file="level1_boss.state",
        max_steps=4000,
        reward_overrides={"boss_damage_coef": 5.0},
        progress_threshold=50.0,
        kill_threshold=0,
    ),
    "full_game": Scenario(
        name="full_game",
        description="Full game",
        state_file=None,
        max_steps=15000,
        reward_overrides={},
        progress_threshold=5000.0,
        kill_threshold=12,
    ),
}


def get_scenario(name: str) -> Scenario:
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario {name}, choices {list(SCENARIOS.keys())}")
    return SCENARIOS[name]


def list_scenarios() -> List[str]:
    return list(SCENARIOS.keys())
