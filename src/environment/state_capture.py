"""Capture gameplay start states by replaying a trained genome.

Used by the trainer to auto-create scenario start states when the curriculum
reaches a stage whose state file is missing: the current best genome replays
from power-on and we snapshot state every N pixels of progress, committing a
snapshot only once the genome has advanced further (verified survivability).
This deliberately does NOT require the pilot to be grounded - jump-heavy
policies never stabilize - but it guarantees the state is not within a couple
of frames of the pilot's death.
"""
from __future__ import annotations

import gzip

import numpy as np

from .contra_env import ContraEnv
from utils.logging import get_logger

logger = get_logger(__name__)


def _serialize_gzip_state(env: ContraEnv) -> bytes:
    """em.get_state() -> gzip bytes (stable-retro loads gzipped states)."""
    data = env._retro_env.em.get_state()
    if isinstance(data, memoryview):
        data = bytes(data)
    if not data.startswith(b"\x1f\x8b"):
        data = gzip.compress(data)
    return data


def capture_progress_state(
    rom_path: str,
    genome_vector: np.ndarray,
    net_config,
    reach_screen: int = 7,
    margin_px: int = 32,
    snapshot_every_px: int = 32,
    max_frames: int = 30000,
    scenario: str = "full_game",
    action_repeat: int = 4,
    flicker_pool: int = 2,
    num_actions: int | None = None,
) -> tuple[bytes, str] | None:
    """Replay `genome_vector` from power-on; return (gzipped_state, label) for
    the furthest verified-survivable snapshot at/after `reach_screen` (or the
    furthest committed snapshot anywhere as fallback). None if nothing commits.

    Deterministic: the same genome + env yields the same state.
    """
    from agent.genome import Genome, Individual
    from agent.policy import Policy

    policy = Policy.from_individual(
        Individual(id=0, genome=Genome(vector=np.asarray(genome_vector, dtype=np.float32))), net_config
    )
    env = ContraEnv(
        rom_path=rom_path,
        observation_mode="pixels",
        frame_stack=4,
        frame_size=84,
        max_episode_steps=max_frames,
        action_repeat=action_repeat,  # must match the policy's training
        scenario=scenario,  # must be stateless so the start matches training
        use_mock_if_missing=False,
        num_actions=num_actions or net_config.num_actions,
        flicker_pool=flicker_pool,
    )
    committed: tuple[int, int, bytes, str] | None = None  # (prog, screen, data, label)
    pending: tuple[int, int, bytes] | None = None  # (prog, screen, data)
    last_snapshot_prog = -(10 ** 9)
    best_progress = -1
    stalled_episodes = 0
    try:
        obs, info = env.reset(seed=0)
        for step in range(max_frames):
            obs, _, term, trunc, info = env.step(int(policy.act(obs)))
            screen = int(info.get("level_screen", 0) or 0)
            scroll = int(info.get("level_scroll", 0) or 0)
            prog = screen * 256 + scroll
            if term or trunc:
                pending = None
                if prog > best_progress:
                    best_progress = prog
                    stalled_episodes = 0
                else:
                    stalled_episodes += 1
                    if stalled_episodes >= 8:
                        logger.warning(f"state capture: no progress for {stalled_episodes} episodes - stopping")
                        break
                obs, info = env.reset(seed=0)
                continue
            if pending is not None and prog >= pending[0] + margin_px:
                # survived margin_px beyond the snapshot: it is a safe start
                committed = (pending[0], pending[1], pending[2], f"screen {pending[1]} scroll {pending[0] % 256} step~{step}")
                pending = None
                if committed[1] >= reach_screen:
                    logger.info(f"state capture: committed screen {committed[1]} at {committed[0]}px (step {step})")
                    return committed[2], committed[3]
            if prog - last_snapshot_prog >= snapshot_every_px:
                pending = (prog, screen, _serialize_gzip_state(env))
                last_snapshot_prog = prog
        if committed is not None:
            logger.warning(
                f"state capture: screen {reach_screen} not reached - using furthest committed ({committed[3]})"
            )
            return committed[2], committed[3]
        logger.warning("state capture: nothing committed - no state captured")
        return None
    finally:
        env.close()
