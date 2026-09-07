#!/usr/bin/env python3
"""Capture retro save-states at gameplay milestones for curriculum start states.

Creates gzipped .state files (stable-retro loads them via retro.make(state=...)):

  level1_start.state                 - first frame inside gameplay (skips title screen)
  level1_screen{NN}.state            - first frame where level_screen == NN
  <name>.state for --capture-at N:file entries

Files are written to the custom integration game dir (.custom_retro/Contra-Nes/)
so retro.make(state=...) resolves them, with copies in roms/states/ for review.

Usage:
  uv run python scripts/capture_states.py --rom roms/Contra.nes
  uv run python scripts/capture_states.py --rom roms/Contra.nes \
      --capture-at 3:level1_enemies.state --capture-at 6:level1_boss.state
"""
from __future__ import annotations

import argparse
import gzip
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from environment.emulator_factory import make_retro_env, custom_retro_dir, resolve_rom_path, GAME_NAME
from environment.reward import is_gameplay
from utils.logging import setup_logging, get_logger

logger = get_logger(__name__)

# stable-retro NES (FCEUmm) 9-button layout
B, START, UP, DOWN, LEFT, RIGHT, A = 0, 3, 4, 5, 6, 7, 8


def make_buttons(env, indices):
    nb = getattr(env, "num_buttons", None) or 9
    arr = np.zeros(nb, dtype=np.uint8)
    for i in indices:
        if i < nb:
            arr[i] = 1
    return arr


def auto_start(env, max_frames: int = 600):
    """Tap START until gameplay RAM routines are detected."""
    env.reset()
    start_btn = make_buttons(env, [START])
    noop = make_buttons(env, [])
    info = {}
    for i in range(max_frames):
        act = start_btn if (i % 20) < 3 else noop
        _, _, _, _, info = env.step(act)
        if info and is_gameplay(info):
            return True, info
    return False, info


def save_state(env, out_path: Path) -> None:
    """Save a gzipped state stable-retro can load (em.get_state -> gzip)."""
    em = getattr(env, "em", None)
    if em is None or not hasattr(em, "get_state"):
        raise RuntimeError("retro env has no .em.get_state handle - cannot save state")
    data = em.get_state()
    if isinstance(data, memoryview):
        data = bytes(data)
    if not data.startswith(b"\x1f\x8b"):  # not gzip -> compress ourselves
        data = gzip.compress(data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)


def install_state(data_path: Path, name: str) -> Path:
    """Copy a state into the custom game dir + roms/states/, return game-dir path."""
    game_path = custom_retro_dir() / GAME_NAME / name
    game_path.parent.mkdir(parents=True, exist_ok=True)
    game_path.write_bytes(data_path.read_bytes())
    mirror = Path("roms") / "states" / name
    mirror.parent.mkdir(parents=True, exist_ok=True)
    mirror.write_bytes(data_path.read_bytes())
    return game_path


def _serialize_gzip_state(env) -> bytes:
    """em.get_state() -> gzip-compressed bytes (stable-retro loads gzipped states)."""
    data = env._retro_env.em.get_state()
    if isinstance(data, memoryview):
        data = bytes(data)
    if not data.startswith(b"\x1f\x8b"):
        data = _gzip.compress(data)
    return data


def capture_with_genome(args) -> int:
    """Play with a trained genome until it reaches --reach-screen, save the state."""
    import tempfile as _tempfile

    import numpy as _np

    from utils.config import load_config
    from agent.network import NetworkConfig
    from environment.actions import get_action_space_size
    from environment.state_capture import capture_progress_state

    cfg = load_config(args.config)
    agent = cfg["agent"]
    net_cfg = NetworkConfig(
        cnn_channels=agent["cnn_channels"],
        kernel_sizes=agent["kernel_sizes"],
        strides=agent["strides"],
        dense_hidden=agent["dense_hidden"],
        num_actions=get_action_space_size(cfg.get("actions", {}).get("space", "playable")),
    )
    path = Path(args.genome)
    if path.suffix == ".npy":
        vec = _np.load(path).astype(_np.float32)
    else:
        vec = _np.asarray(load_checkpoint(path)["best_genome"], dtype=_np.float32)

    result = capture_progress_state(
        rom_path=str(resolve_rom_path(args.rom)),
        genome_vector=vec,
        net_config=net_cfg,
        reach_screen=args.reach_screen,
        max_frames=args.max_frames,
        scenario=args.scenario,
        action_repeat=cfg["env"].get("action_repeat", 4),
        flicker_pool=cfg["env"].get("flicker_pool", 2),
        num_actions=net_cfg.num_actions,
    )
    if result is None:
        logger.error("Capture failed: genome produced no survivable snapshot")
        return 1
    data, label = result
    tmp = Path(_tempfile.gettempdir()) / args.name
    tmp.write_bytes(data)
    logger.info(f"Saved {args.name} ({label}) -> {install_state(tmp, args.name)}")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rom", type=str, default="roms/Contra.nes")
    p.add_argument("--max-frames", type=int, default=30000, help="emulator frame cap for the run")
    p.add_argument("--jump-every", type=int, default=45, help="tap jump every N frames")
    p.add_argument("--no-auto-screens", action="store_true", help="skip per-screen level1_screenNN states")
    p.add_argument(
        "--capture-at",
        type=str,
        action="append",
        default=[],
        help="capture when level_screen first reaches N: e.g. 3:level1_enemies.state (repeatable)",
    )
    p.add_argument("--genome", type=str, default=None, help="play with a trained genome (.npy/.pkl) instead of the macro")
    p.add_argument("--config", type=str, default="configs/default.yaml", help="config defining the network architecture")
    p.add_argument("--reach-screen", type=int, default=8, help="capture when level_screen first reaches N (genome mode)")
    p.add_argument("--name", type=str, default="level1_cliff.state", help="state filename (genome mode)")
    p.add_argument(
        "--scenario",
        type=str,
        default="extended_level1",
        help="scenario whose start the genome was trained on (genome mode; must match training)",
    )
    p.add_argument(
        "--stable-frames",
        type=int,
        default=10,
        help="grounded stable frames required before capturing (genome mode)",
    )
    args = p.parse_args()

    setup_logging()

    if args.genome:
        from training.checkpoint import load_checkpoint  # noqa: F401

        return capture_with_genome(args)

    rom = resolve_rom_path(args.rom)
    env = make_retro_env(rom, state=None, render_mode=None)
    try:
        ok, info = auto_start(env)
        if not ok:
            logger.error("Could not reach gameplay via START taps - aborting")
            return 1
        screen = int(info.get("level_screen", 0) or 0)
        start_path = Path(tempfile.gettempdir()) / "level1_start.state"
        save_state(env, start_path)
        logger.info(f"Saved level1_start.state (screen={screen}) -> {install_state(start_path, 'level1_start.state')}")

        capture_at = {}
        for spec in args.capture_at:
            n, _, name = spec.partition(":")
            capture_at[int(n)] = name or f"level1_screen{int(n):02d}.state"

        saved_screens = {screen}
        frame = 0
        while frame < args.max_frames:
            buttons = [RIGHT, B]
            if frame % max(1, args.jump_every) < 8:
                buttons.append(A)
            _, _, terminated, truncated, info = env.step(make_buttons(env, buttons))
            frame += 1
            if terminated or truncated:
                logger.warning(f"Episode ended at frame {frame} - auto-restart via START taps")
                ok, info = auto_start(env)
                if not ok:
                    break
            cur = int(info.get("level_screen", screen) or screen)
            if cur != screen and cur not in saved_screens and cur > screen:
                screen = cur
                saved_screens.add(cur)
                tmp = Path(tempfile.gettempdir()) / f"level1_screen{cur:02d}.state"
                save_state(env, tmp)
                logger.info(f"Saved level1_screen{cur:02d}.state -> {install_state(tmp, f'level1_screen{cur:02d}.state')}")
                if not args.no_auto_screens and cur in capture_at:
                    name = capture_at.pop(cur)
                    tmp2 = Path(tempfile.gettempdir()) / name
                    save_state(env, tmp2)
                    logger.info(f"Saved {name} -> {install_state(tmp2, name)}")
            elif cur in capture_at and cur not in saved_screens:
                pass
            for n in [n for n in capture_at if cur >= n]:
                name = capture_at.pop(n)
                tmp = Path(tempfile.gettempdir()) / name
                save_state(env, tmp)
                logger.info(f"Saved {name} (screen={cur}) -> {install_state(tmp, name)}")
            if not capture_at and args.no_auto_screens:
                break
        logger.info(f"Done. Captured screens: {sorted(saved_screens)}; pending capture targets: {capture_at}")
        logger.info("Assign states to curriculum stages via Scenario.state_file in src/environment/scenarios.py")
        return 0
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
