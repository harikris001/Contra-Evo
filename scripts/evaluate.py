#!/usr/bin/env python3
"""Watch a trained genome play Contra.

  uv run python scripts/evaluate.py --checkpoint checkpoints/gen_0100.pkl --render
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import load_config
from agent.network import NetworkConfig, count_parameters
from agent.genome import Genome, Individual
from agent.policy import Policy
from environment.contra_env import ContraEnv
from environment.reward import RewardConfig
from environment.actions import ACTION_NAMES, get_action_space_size
from training.checkpoint import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate / watch a trained Contra agent")
    p.add_argument("--checkpoint", type=str, default="checkpoints/gen_0100.pkl")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--render", action="store_true", default=True, help="open a pygame window (default)")
    p.add_argument("--headless", action="store_true", help="print stats only, no window")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--scale", type=int, default=3)
    p.add_argument(
        "--from-beginning",
        action="store_true",
        help="ignore the checkpoint's curriculum stage: play the full level from the true "
        "gameplay spawn (level1_start.state - skips title/intro frames; falls back to "
        "power-on if that state file is missing)",
    )
    return p.parse_args()


def _network_from_data(data: dict, cfg: dict, genome_size: int | None) -> NetworkConfig:
    raw = dict(data.get("net_config") or {})
    if "input_shape" in raw and isinstance(raw["input_shape"], list):
        raw["input_shape"] = tuple(raw["input_shape"])
    allowed = ("input_shape", "cnn_channels", "kernel_sizes", "strides", "dense_hidden", "num_actions", "activation")
    kwargs = {k: raw[k] for k in allowed if k in raw and raw[k] is not None}
    if not kwargs.get("cnn_channels"):
        agent = cfg["agent"]
        kwargs.update(
            cnn_channels=agent["cnn_channels"],
            kernel_sizes=agent["kernel_sizes"],
            strides=agent["strides"],
            dense_hidden=agent["dense_hidden"],
        )
    if "num_actions" not in kwargs:
        kwargs["num_actions"] = get_action_space_size(cfg.get("actions", {}).get("space", "playable"))
    net = NetworkConfig(**kwargs)
    if genome_size is not None:
        # If saved num_actions does not match the vector, try playable vs reduced.
        if count_parameters(net) != genome_size:
            for n in (16, 12, 8):
                trial = NetworkConfig(
                    input_shape=net.input_shape,
                    cnn_channels=net.cnn_channels,
                    kernel_sizes=net.kernel_sizes,
                    strides=net.strides,
                    dense_hidden=net.dense_hidden,
                    num_actions=n,
                    activation=net.activation,
                )
                if count_parameters(trial) == genome_size:
                    return trial
    return net


def _load_genome(path: Path, cfg: dict) -> tuple[np.ndarray, dict]:
    if path.suffix == ".npy":
        vec = np.load(path).astype(np.float32)
        return vec, {"generation": "npy", "fitnesses": [float("nan")], "env_config": cfg["env"], "net_config": {}}
    data = load_checkpoint(path)
    vec = np.asarray(data["best_genome"], dtype=np.float32)
    return vec, data


def main():
    args = parse_args()
    show = args.render and not args.headless
    cfg = load_config(args.config)
    ckpt_path = Path(args.checkpoint)
    vec, data = _load_genome(ckpt_path, cfg)
    fits = data.get("fitnesses") or []
    best_fit = max(fits) if fits and fits[0] == fits[0] else float("nan")
    print(f"Loaded {ckpt_path} gen {data.get('generation')} best fitness {best_fit:.1f} genome {vec.size}", flush=True)

    net_cfg = _network_from_data(data, cfg, vec.size)
    if count_parameters(net_cfg) != vec.size:
        raise SystemExit(
            f"Genome size {vec.size} does not match network {count_parameters(net_cfg)} "
            f"(num_actions={net_cfg.num_actions}). Pass the config used for training."
        )
    genome = Genome(vector=vec)
    ind = Individual(id=0, genome=genome)
    policy = Policy.from_individual(ind, net_cfg)

    env_cfg = dict(cfg["env"])
    env_cfg.update(data.get("env_config") or {})
    if args.from_beginning:
        # 'movement' resolves to level1_start.state = the first gameplay frame
        # (lives=2, x=0): the full level from the true spawn, no title/intro
        # frames. If that state file is missing ContraEnv falls back to
        # power-on automatically.
        env_cfg["scenario"] = "movement"
        env_cfg["max_episode_steps"] = max(int(env_cfg.get("max_episode_steps") or 0), 15000)
        print("[EVAL] --from-beginning: starting from the level-1 gameplay spawn (skips title/intro frames)")
    else:
        print(f"[EVAL] start: checkpoint stage scenario '{env_cfg.get('scenario')}'")
    reward_cfg = RewardConfig.from_dict(cfg.get("reward", {}))
    env = ContraEnv(
        rom_path=env_cfg.get("rom_path"),
        observation_mode=env_cfg.get("observation_mode", "pixels"),
        frame_stack=env_cfg.get("frame_stack", 4),
        frame_size=env_cfg.get("frame_size", 84),
        max_episode_steps=env_cfg.get("max_episode_steps", 5000),
        action_repeat=env_cfg.get("action_repeat", 4),
        reward_config=reward_cfg,
        scenario=env_cfg.get("scenario", "full_level_1"),
        render_mode="rgb_array" if show else None,
        use_mock_if_missing=env_cfg.get("rom_path") in (None, "", "null"),
        num_actions=net_cfg.num_actions,
    )

    window = None
    if show:
        from visualization.playback import PlaybackWindow

        window = PlaybackWindow(scale=args.scale, fps=args.fps)

    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=ep)
            done = truncated = False
            steps = 0
            act = 0
            while not (done or truncated):
                if window and not window.poll():
                    print("Window closed")
                    return
                if window and window.paused:
                    frame = env.render()
                    window.draw(frame, _hud(ep, steps, act, env, info))
                    continue
                act = policy.act(obs)
                obs, rew, done, truncated, info = env.step(act)
                steps += 1
                if window:
                    frame = env.render()
                    window.draw(frame, _hud(ep, steps, act, env, info))
            print(
                f"Episode {ep}: fitness {env.get_fitness():.1f} steps {steps} "
                f"progress {info.get('progress', 0)} kills {info.get('kills', 0)}",
                flush=True,
            )
    finally:
        if window:
            window.close()
        env.close()


def _hud(ep, steps, act, env, info):
    name = ACTION_NAMES[act] if 0 <= act < len(ACTION_NAMES) else str(act)
    return {
        "episode": ep,
        "steps": steps,
        "action_name": name,
        "fitness": env.get_fitness(),
        "progress": info.get("progress", 0),
        "kills": info.get("kills", 0),
        "lives": info.get("lives", "?"),
        "screen_type": info.get("screen_type", 0),
    }


if __name__ == "__main__":
    main()
