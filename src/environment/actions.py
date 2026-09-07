"""Action space mapping. Isolated as per spec.

We map discrete actions to retro button combinations.
Retro FCEUmm uses 12 buttons: B, A, SELECT, START, UP, DOWN, LEFT, RIGHT.

Playable 16-action set (default):
0 NOOP
1 LEFT
2 RIGHT
3 SHOOT (B)
4 JUMP (A)
5 LEFT+SHOOT
6 RIGHT+SHOOT
7 JUMP+SHOOT
8 RIGHT+JUMP
9 RIGHT+JUMP+SHOOT
10 DOWN
11 UP
12 UP+JUMP
13 UP+JUMP+SHOOT
14 UP+RIGHT+JUMP
15 UP+RIGHT+JUMP+SHOOT

Reduced 8-action set keeps 0-7 for backward compatibility.
"""
from __future__ import annotations

import numpy as np

ACTION_NAMES = [
    "NOOP",
    "LEFT",
    "RIGHT",
    "SHOOT",
    "JUMP",
    "LEFT+SHOOT",
    "RIGHT+SHOOT",
    "JUMP+SHOOT",
    "RIGHT+JUMP",
    "RIGHT+JUMP+SHOOT",
    "DOWN",
    "UP",
    "UP+JUMP",
    "UP+JUMP+SHOOT",
    "UP+RIGHT+JUMP",
    "UP+RIGHT+JUMP+SHOOT",
]

REDUCED_ACTION_NAMES = ACTION_NAMES[:8]

# Retro button order for NES via FCEUmm/stable-retro 9-button:
# Index: 0=B, 1=None, 2=SELECT, 3=START, 4=UP, 5=DOWN, 6=LEFT, 7=RIGHT, 8=A
NUM_RETRO_BUTTONS = 12

# Mapping each discrete action to button indices pressed (using 9-button layout)
_ACTION_TO_BUTTONS = {
    0: [],
    1: [6],  # LEFT
    2: [7],  # RIGHT
    3: [0],  # B (shoot)
    4: [8],  # A (jump)
    5: [6, 0],  # LEFT+SHOOT
    6: [7, 0],  # RIGHT+SHOOT
    7: [8, 0],  # JUMP+SHOOT
    8: [7, 8],  # RIGHT+JUMP
    9: [7, 8, 0],  # RIGHT+JUMP+SHOOT
    10: [5],  # DOWN
    11: [4],  # UP
    12: [4, 8],  # UP+JUMP
    13: [4, 8, 0],  # UP+JUMP+SHOOT
    14: [4, 7, 8],  # UP+RIGHT+JUMP
    15: [4, 7, 8, 0],  # UP+RIGHT+JUMP+SHOOT
}

# For 8-button compatibility, map A from 8 -> 1
_ACTION_TO_BUTTONS_8 = {
    0: [],
    1: [6],
    2: [7],
    3: [0],
    4: [1],
    5: [6, 0],
    6: [7, 0],
    7: [1, 0],
    8: [7, 1],
    9: [7, 1, 0],
    10: [5],
    11: [4],
    12: [4, 1],
    13: [4, 1, 0],
    14: [4, 7, 1],
    15: [4, 7, 1, 0],
}

ACTIONS = list(_ACTION_TO_BUTTONS.keys())
NUM_PLAYABLE_ACTIONS = 16
NUM_REDUCED_ACTIONS = 8


def discrete_to_retro(action: int, num_buttons: int | None = None) -> np.ndarray:
    """Convert discrete action to retro multi-binary vector.

    Args:
        action: 0-15 (playable) or 0-7 (reduced)
        num_buttons: if provided, size output to match env (8,9,12). Otherwise 12.
    """
    if action not in _ACTION_TO_BUTTONS:
        raise ValueError(f"Invalid action {action}, must be 0-{max(_ACTION_TO_BUTTONS)}")
    size = num_buttons if num_buttons is not None else NUM_RETRO_BUTTONS
    arr = np.zeros(size, dtype=np.int8)
    mapping = _ACTION_TO_BUTTONS if size == 9 or size == 12 else _ACTION_TO_BUTTONS_8
    for b in mapping[action]:
        if b < size:
            arr[b] = 1
        if size == 12:
            if b == 8:
                arr[1] = 1
            elif b == 1:
                arr[8] = 1
    return arr


def retro_to_discrete(vec: np.ndarray) -> int:
    """Inverse: find closest discrete action for a retro vector (for logging)."""
    best = 0
    best_match = -1
    for act, btns in _ACTION_TO_BUTTONS.items():
        expected = np.zeros_like(vec)
        for b in btns:
            if b < expected.size:
                expected[b] = 1
        score = int((vec == expected).sum())
        if score > best_match:
            best_match = score
            best = act
    return best


def get_action_space_size(space: str = "playable") -> int:
    if space in ("reduced", "legacy"):
        return NUM_REDUCED_ACTIONS
    if space in ("playable", "full"):
        return NUM_PLAYABLE_ACTIONS
    raise ValueError(f"Unknown space {space}")
