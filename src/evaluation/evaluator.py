"""Single-agent evaluator: runs one individual in Contra env."""
from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np

from environment.contra_env import ContraEnv
from environment.reward import RewardConfig
from agent.genome import Individual
from agent.network import NetworkConfig
from agent.policy import Policy


def _allow_mock(env_config: Dict[str, Any]) -> bool:
    if "use_mock_if_missing" in env_config and env_config["use_mock_if_missing"] is not None:
        return bool(env_config["use_mock_if_missing"])
    rom = env_config.get("rom_path")
    return rom in (None, "", "null")


class Evaluator:
    """Evaluates one individual, returns fitness and stats.

    The emulator is cached and reused across episodes (rebuilt only when the
    env/reward config changes), avoiding per-episode retro.make + ROM hashing.
    Keep observation memory small: uint8 stacked frames.
    """

    def __init__(
        self,
        env_config: Dict[str, Any],
        net_config: NetworkConfig,
        reward_config: RewardConfig,
    ):
        self.env_config = env_config
        self.net_config = net_config
        self.reward_config = reward_config
        self._env: ContraEnv | None = None
        self._env_sig: str | None = None

    def _build_env_kwargs(self, render_mode: str | None) -> Dict[str, Any]:
        return dict(
            rom_path=self.env_config.get("rom_path"),
            observation_mode=self.env_config.get("observation_mode", "pixels"),
            frame_stack=self.env_config.get("frame_stack", 4),
            frame_size=self.env_config.get("frame_size", 84),
            max_episode_steps=self.env_config.get("max_episode_steps", 5000),
            action_repeat=self.env_config.get("action_repeat", 4),
            reward_config=self.reward_config,
            scenario=self.env_config.get("scenario", "full_level_1"),
            render_mode=render_mode,
            use_mock_if_missing=_allow_mock(self.env_config),
            num_actions=self.env_config.get("num_actions", self.net_config.num_actions),
            flicker_pool=self.env_config.get("flicker_pool", 2),
        )

    def _get_env(self, render_mode: str | None) -> ContraEnv:
        """Return cached env, rebuilding only when the config signature changes."""
        sig = repr(
            (
                sorted(self.env_config.items()),
                sorted(self.reward_config.__dict__.items()),
                render_mode,
            )
        )
        if self._env is None or self._env_sig != sig:
            if self._env is not None:
                try:
                    self._env.close()
                except Exception:
                    pass
            self._env = ContraEnv(**self._build_env_kwargs(render_mode))
            self._env_sig = sig
        return self._env

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None
            self._env_sig = None

    def evaluate(self, individual: Individual, seed: int | None = None, publish_frame: Any | None = None) -> Tuple[float, Dict[str, Any], int]:
        """Run episode and return (fitness, stats, steps).

        publish_frame: optional callable(frame, info) or shared dict to publish live frames for visualization
        """
        # For live visualization, always use rgb_array so render() returns frame
        render_mode = "rgb_array" if publish_frame is not None else self.env_config.get("render_mode", None)
        env = self._get_env(render_mode)
        # Determine publish helper
        is_callable = callable(publish_frame)
        is_shared = publish_frame is not None and hasattr(publish_frame, "__setitem__") and not is_callable
        should_render = is_callable or is_shared
        policy = Policy.from_individual(individual, self.net_config)
        try:
            obs, info = env.reset(seed=seed)
            done = False
            truncated = False
            steps = 0
            last_frame = None
            if should_render:
                try:
                    last_frame = env.render()
                except Exception:
                    pass
            while not (done or truncated):
                action = policy.act(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                steps += 1
                # Publish live frame every 4 steps (to match action_repeat and keep 15 FPS)
                if should_render and steps % 4 == 0:
                    try:
                        frame = env.render()
                        if frame is not None:
                            last_frame = frame
                            if is_callable:
                                publish_frame(frame, info)
                            elif is_shared:
                                key = getattr(individual, "id", 0)
                                try:
                                    publish_frame[key] = (frame.copy() if hasattr(frame, "copy") else frame, dict(info))
                                except Exception:
                                    publish_frame[key] = (frame, dict(info))
                    except Exception:
                        pass
                if steps >= self.env_config.get("max_episode_steps", 5000):
                    truncated = True
                    break
            fitness = env.get_fitness()
            # Ensure final frame is published
            if should_render:
                try:
                    frame = env.render()
                    if frame is not None:
                        last_frame = frame
                        if is_callable:
                            publish_frame(frame, info)
                        elif is_shared:
                            key = getattr(individual, "id", 0)
                            try:
                                publish_frame[key] = (frame.copy() if hasattr(frame, "copy") else frame, dict(info))
                            except Exception:
                                pass
                except Exception:
                    pass
            # stats
            stats = {
                "fitness": fitness,
                "progress": float(info.get("progress", info.get("x", info.get("x_pos", 0)))),
                "kills": float(info.get("kills", 0)),
                "steps": steps,
                "lives": float(info.get("lives", 0)),
                "deaths": float(info.get("deaths", 0)),
                "idle_steps": float(info.get("idle_steps", 0)),
                "screen_type": float(info.get("screen_type", 0)),
                "status": str(info.get("status") or info.get("end_reason") or ("dead" if done else "truncated")),
                "end_reason": str(info.get("end_reason") or ""),
                "using_mock": float(1.0 if getattr(env, "_mock", None) is not None else 0.0),
            }
            # enrich with env reward stats
            for k in ("fitness", "progress", "kills", "steps", "deaths"):
                if k in info:
                    stats[k] = float(info[k])
            # Attach last frame for dashboard fallback (when not using shared dict)
            stats["_frame"] = last_frame
            return fitness, stats, steps
        except Exception:
            # A crashed/corrupted emulator must not poison the cached env
            self.close()
            raise

    def evaluate_with_frames(self, individual: Individual, seed: int | None = None, max_steps: int = 1000):
        """Generator yielding frames for visualization (not used in headless)."""
        env = ContraEnv(**self._build_env_kwargs(self.env_config.get("render_mode", None)))
        policy = Policy.from_individual(individual, self.net_config)
        obs, info = env.reset(seed=seed)
        done = False
        truncated = False
        while not (done or truncated):
            action = policy.act(obs)
            obs, rew, done, truncated, info = env.step(action)
            frame = env.render()
            yield frame, info, rew, done
        env.close()
