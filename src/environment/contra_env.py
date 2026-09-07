"""Gymnasium-style Contra environment.

Wraps stable-retro if available, else provides MockContraEnv for testing/CPU training without ROM.

Interface: reset() -> obs, info ; step(action) -> obs, reward, terminated, truncated, info ; render() ; close()

Designed to be extensible for PPO/DQN: observation_space, action_space follow Gymnasium spec.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .actions import discrete_to_retro, ACTION_NAMES
from .observation import FrameStack, extract_features, process_frame
from .reward import RewardCalculator, RewardConfig, is_gameplay
from .scenarios import get_scenario
from .emulator_factory import (
    check_rom,
    check_retro_install,
    make_retro_env,
    custom_retro_dir,
    project_root,
    GAME_NAME,
)
from utils.logging import get_logger

logger = get_logger(__name__)


class ContraEnv(gym.Env):
    """Gymnasium wrapper for NES Contra via stable_retro.

    Observation: either pixels (84x84x4 uint8) or features (10-dim float32)
    Action: Discrete(8) reduced set
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 60}

    def __init__(
        self,
        rom_path: str | None = None,
        observation_mode: str = "pixels",
        frame_stack: int = 4,
        frame_size: int = 84,
        max_episode_steps: int = 5000,
        action_repeat: int = 4,
        reward_config: RewardConfig | None = None,
        scenario: str = "full_level_1",
        render_mode: str | None = None,
        use_mock_if_missing: bool | None = None,
        num_actions: int | None = None,
        flicker_pool: int = 2,
    ):
        super().__init__()
        self.rom_path = rom_path
        self.observation_mode = observation_mode
        self.frame_stack = frame_stack
        self.frame_size = frame_size
        self.max_episode_steps = max_episode_steps
        self.action_repeat = action_repeat
        self.scenario_name = scenario
        self.render_mode = render_mode
        self.reward_config = reward_config or RewardConfig()
        self.reward_calc = RewardCalculator(self.reward_config)
        self.num_actions = num_actions if num_actions is not None else len(ACTION_NAMES)
        self.flicker_pool = max(1, int(flicker_pool))
        self.steps = 0
        self.prev_info: Dict[str, Any] = {}
        self._mock: Any = None
        self._retro_env: Any = None
        self._frame_stacker: FrameStack | None = None
        self._needs_mock = False
        if use_mock_if_missing is None:
            use_mock_if_missing = self.rom_path in (None, "", "null")

        # Action space
        self.action_space = spaces.Discrete(self.num_actions)
        if observation_mode == "pixels":
            # gymnasium expects (C,H,W) or (H,W,C)? Use (stack,84,84) uint8
            self.observation_space = spaces.Box(low=0, high=255, shape=(frame_stack, frame_size, frame_size), dtype=np.uint8)
            self._frame_stacker = FrameStack(
                stack=frame_stack,
                size=frame_size,
                grayscale=True,
                channel_first=True,
                flicker_pool=self.flicker_pool,
            )
        else:
            self.observation_space = spaces.Box(low=-5, high=5, shape=(10,), dtype=np.float32)

        # Try to create retro env
        self._init_emulator(use_mock_if_missing)

    def _resolve_state(self) -> str | None:
        """Resolve the retro save-state file for the current scenario, if any.

        Search order: custom integration game dir (so `retro.make(state=name)`
        can resolve it), roms/states/, and any direct path.
        Returns an absolute path string (gzipped .state) or None.
        """
        try:
            sc = get_scenario(self.scenario_name)
        except ValueError:
            return None
        sf = sc.state_file
        if not sf:
            return None
        game_dir = custom_retro_dir() / GAME_NAME
        candidates = [
            game_dir / sf,
            game_dir / "states" / sf,
            project_root() / "roms" / "states" / sf,
            project_root() / sf,
            Path(sf),
        ]
        for p in candidates:
            try:
                if p.exists():
                    return str(p.resolve())
            except OSError:
                continue
        logger.warning(
            f"Scenario '{self.scenario_name}' state file '{sf}' not found "
            f"(searched {game_dir}, roms/states/) - starting from power-on. "
            f"Run scripts/capture_states.py to create it."
        )
        return None

    def _init_emulator(self, use_mock_if_missing: bool) -> None:
        ok_retro, msg_retro = check_retro_install()
        ok_rom, msg_rom = check_rom(self.rom_path)
        logger.info(f"Retro check: {msg_retro}")
        logger.info(f"ROM check: {msg_rom}")
        if ok_retro and ok_rom:
            try:
                from .emulator_factory import make_retro_env

                state_path = self._resolve_state()
                if state_path:
                    logger.info(f"Loading scenario start state: {state_path}")
                self._retro_env = make_retro_env(self.rom_path, state=state_path, render_mode=self.render_mode)
                logger.info("Retro env created successfully")
                self._warm_up_retro_env()
            except Exception as e:
                logger.error(f"Failed to create retro env: {e}")
                if use_mock_if_missing:
                    logger.warning("Falling back to MockContraEnv (no ROM configured)")
                    self._needs_mock = True
                else:
                    raise FileNotFoundError(
                        f"Refusing MockContraEnv because a ROM was requested ({self.rom_path}). {e}"
                    ) from e
        else:
            if use_mock_if_missing:
                logger.warning("Using MockContraEnv (no ROM/retro). Set env.rom_path for real Contra.")
                self._needs_mock = True
            else:
                raise FileNotFoundError(f"Cannot create env: {msg_rom} | {msg_retro}")

        if self._needs_mock:
            self._mock = MockContraEnv(
                observation_mode=self.observation_mode,
                frame_stack=self.frame_stack,
                frame_size=self.frame_size,
                max_episode_steps=self.max_episode_steps,
                num_actions=self.num_actions,
            )
        # cache retro meta for action mapping
        self._retro_num_buttons = None
        self._retro_is_discrete = False
        if self._retro_env is not None:
            try:
                # Detect action space
                if hasattr(self._retro_env, "action_space"):
                    from gymnasium.spaces import Discrete

                    self._retro_is_discrete = isinstance(self._retro_env.action_space, Discrete)
                if hasattr(self._retro_env, "num_buttons"):
                    self._retro_num_buttons = int(self._retro_env.num_buttons)
                elif hasattr(self._retro_env, "buttons"):
                    self._retro_num_buttons = len(self._retro_env.buttons)  # type: ignore
            except Exception:
                pass
        self._menu_frames = 0

    def _warm_up_retro_env(self) -> None:
        """Establish a deterministic, settled episode-start point.

        1. If no scenario state is loaded, play through the title/menu once and
           freeze that gameplay frame as `initial_state` - every future reset
           restores it (no per-episode menu replay, no cross-episode leakage).
        2. The first savestate restore after a cold run behaves subtly
           differently from later restores (incomplete core savestate), so two
           throwaway resets settle the restore path. Every measured episode
           then starts identically, on every worker.
        """
        if self._retro_env is None:
            return
        if getattr(self._retro_env, "initial_state", None) is None:
            try:
                obs, info = self._retro_env.reset()
                try:
                    obs, _, _, _, info = self._retro_env.step(self._retro_noop_action())
                except Exception:
                    pass
                if not is_gameplay(info):
                    obs, info = self._enter_gameplay(obs, info)
                    self._retro_env.initial_state = bytes(self._retro_env.em.get_state())
                    logger.info("Captured deterministic gameplay start state")
            except Exception as e:
                logger.debug(f"start-state capture failed: {e}")
        try:
            self._retro_env.reset()
            self._retro_env.reset()
        except Exception as e:
            logger.debug(f"warm-up resets failed: {e}")

    def _retro_start_action(self):
        import numpy as np

        nb = self._retro_num_buttons or 9
        if self._retro_is_discrete:
            return 8
        a = np.zeros(nb, dtype=np.uint8)
        a[min(3, nb - 1)] = 1  # START
        return a

    def _retro_noop_action(self):
        import numpy as np

        if self._retro_is_discrete:
            return 0
        nb = self._retro_num_buttons or 9
        return np.zeros(nb, dtype=np.uint8)

    def _retro_raw_step(self, retro_act):
        return self._retro_env.step(retro_act)  # type: ignore

    def _enter_gameplay(self, obs, info, max_frames: int = 240):
        """Leave title/demo by tapping START until is_gameplay(info)."""
        if self._retro_env is None:
            return obs, info
        last_obs, last_info = obs, info or {}
        try:
            for i in range(max_frames):
                if last_info and is_gameplay(last_info):
                    break
                act = self._retro_start_action() if (i % 20) < 3 else self._retro_noop_action()
                try:
                    last_obs, _, _, _, last_info = self._retro_raw_step(act)
                except Exception:
                    break
        except Exception:
            pass
        return last_obs, last_info if last_info is not None else {}

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.steps = 0
        self._menu_frames = 0
        self.reward_calc.reset()
        self.prev_info = {}
        if self._mock is not None:
            obs, info = self._mock.reset(seed=seed)
            if self._frame_stacker is not None and self.observation_mode == "pixels":
                # mock already returns stacked; reset stacker
                assert isinstance(obs, np.ndarray)
                # initialize stacker
                dummy_frame = np.zeros((240, 256, 3), dtype=np.uint8)
                self._frame_stacker.reset(dummy_frame)
                # push actual frame? use obs[0]
                # simpler: just return mock obs
                pass
            return obs, info
        assert self._retro_env is not None
        obs, info = self._retro_env.reset()
        # retro reset() returns an empty info dict; step one frame (noop) to
        # read RAM. With a loaded/captured initial_state this restores the
        # frozen gameplay start deterministically.
        try:
            obs, _, _, _, info = self._retro_env.step(self._retro_noop_action())
        except Exception as e:
            logger.debug(f"post-reset noop step failed: {e}")
        if not is_gameplay(info):
            # Fallback (start-state capture failed): run the menu sequence
            try:
                obs, info = self._enter_gameplay(obs, info)
            except Exception as e:
                logger.debug(f"auto-start failed: {e}")
        if info:
            self.prev_info = dict(info)
        # process observation
        if self.observation_mode == "pixels":
            assert self._frame_stacker is not None
            # Ensure obs is RGB image; if obs is from get_screen, it is (H,W,3)
            # If auto-start already stepped, obs may be outdated; just use current obs
            # If obs is None or shape mismatch, try to get screen again
            if obs is None or not hasattr(obs, "shape"):
                try:
                    obs = self._retro_env.get_screen()  # type: ignore
                except Exception:
                    obs = np.zeros((240, 256, 3), dtype=np.uint8)
            obs_proc = self._frame_stacker.reset(obs)
        else:
            obs_proc = extract_features(info)
        return obs_proc, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid action {action}"
        if self._mock is not None:
            obs, rew, term, trunc, info = self._mock.step(action)
            # apply reward calc overlay? mock already has reward, but we recalc for consistency
            # Use our reward calculator for fitness tracking
            calc_rew = self.reward_calc.step(info, term or trunc, self.prev_info, action=int(action))
            self.prev_info = dict(info)
            info.update(self.reward_calc.get_stats())
            self.steps += 1
            if self.reward_calc.state.should_terminate:
                term = True
            if self.reward_calc.state.should_truncate:
                trunc = True
            if self.steps >= self.max_episode_steps:
                trunc = True
                if not self.reward_calc.state.end_reason:
                    self.reward_calc.state.end_reason = "time"
            info["end_reason"] = self.reward_calc.state.end_reason
            info["status"] = self.reward_calc.state.end_reason or ("running" if not (term or trunc) else "truncated")
            return obs, calc_rew, term, trunc, info

        assert self._retro_env is not None
        # action repeat
        total_rew = 0.0
        last_obs = None
        last_info: Dict[str, Any] = {}
        terminated = False
        truncated = False
        for _ in range(self.action_repeat):
            in_game = is_gameplay(self.prev_info) if self.prev_info else False
            if not in_game:
                retro_act = self._retro_start_action()
            elif self._retro_is_discrete:
                retro_act = int(action)
            else:
                nb = self._retro_num_buttons or 9
                retro_act = discrete_to_retro(action, num_buttons=nb)
            try:
                last_obs, rew, terminated, truncated, last_info = self._retro_env.step(retro_act)  # type: ignore
            except Exception:
                try:
                    last_obs, rew, terminated, truncated, last_info = self._retro_raw_step(
                        self._retro_start_action() if not in_game else self._retro_noop_action()
                    )
                except Exception:
                    last_obs, rew, terminated, truncated, last_info = self._retro_env.step(
                        self._retro_env.action_space.sample()
                    )  # type: ignore
            total_rew += float(rew)
            if terminated or truncated:
                break
        playing = is_gameplay(last_info) if last_info else False
        if playing:
            self.steps += 1
        else:
            self._menu_frames += 1
            if self._menu_frames > 400:
                truncated = True
        # process obs
        if self.observation_mode == "pixels":
            assert self._frame_stacker is not None and last_obs is not None
            obs_proc = self._frame_stacker.push(last_obs)
        else:
            obs_proc = extract_features(last_info)
        # compute fitness reward (override retro rew with our shaped reward)
        shaped = self.reward_calc.step(
            last_info, terminated or truncated, self.prev_info, action=int(action)
        )
        self.prev_info = dict(last_info)
        # combine? Use shaped as primary, but log retro rew
        last_info["retro_reward"] = total_rew
        last_info["shaped_reward"] = shaped
        # inject stats
        last_info.update(self.reward_calc.get_stats())
        if self.reward_calc.state.should_terminate:
            terminated = True
        if self.reward_calc.state.should_truncate:
            truncated = True
        if self.steps >= self.max_episode_steps:
            truncated = True
            if not self.reward_calc.state.end_reason:
                self.reward_calc.state.end_reason = "time"
        last_info["end_reason"] = self.reward_calc.state.end_reason
        last_info["status"] = self.reward_calc.state.end_reason or (
            "running" if not (terminated or truncated) else "truncated"
        )
        return obs_proc, shaped, terminated, truncated, last_info

    def render(self):
        if self._mock is not None:
            return self._mock.render()
        if self._retro_env is not None and hasattr(self._retro_env, "render"):
            return self._retro_env.render()
        return None

    def close(self):
        if self._retro_env is not None:
            try:
                self._retro_env.close()
            except:
                pass
        if self._mock is not None:
            self._mock.close()

    def get_fitness(self) -> float:
        return self.reward_calc.get_fitness()


class MockContraEnv(gym.Env):
    """Lightweight mock for testing without ROM. Simulates progress, kills, etc.

    Useful for CI, unit tests, and M4 debugging without emulator overhead.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        observation_mode: str = "pixels",
        frame_stack: int = 4,
        frame_size: int = 84,
        max_episode_steps: int = 5000,
        num_actions: int = 16,
    ):
        super().__init__()
        self.observation_mode = observation_mode
        self.frame_stack = frame_stack
        self.frame_size = frame_size
        self.max_episode_steps = max_episode_steps
        self.num_actions = num_actions
        self.action_space = spaces.Discrete(num_actions)
        if observation_mode == "pixels":
            self.observation_space = spaces.Box(low=0, high=255, shape=(frame_stack, frame_size, frame_size), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=-5, high=5, shape=(10,), dtype=np.float32)
        self.steps = 0
        self.x = 0.0
        self.y = 100.0
        self.kills = 0
        self.lives = 3
        self.water_state = 0
        self._frame_stacker = FrameStack(stack=frame_stack, size=frame_size, grayscale=True, channel_first=True) if observation_mode == "pixels" else None
        self.rng = np.random.default_rng(0)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.x = 0.0
        self.y = 100.0
        self.kills = 0
        self.lives = 3
        self.water_state = 0
        info = {
            "x": self.x,
            "x_pos": self.x,
            "y_pos": self.y,
            "kills": self.kills,
            "lives": self.lives,
            "scroll": self.x,
            "progress": self.x,
            "water_state": self.water_state,
            "in_gameplay": True,
            "game_routine": 4,
            "screen_type": 4,
        }
        if self.observation_mode == "pixels":
            dummy = self.rng.integers(0, 255, size=(240, 256, 3), dtype=np.uint8)
            assert self._frame_stacker is not None
            obs = self._frame_stacker.reset(dummy)
        else:
            obs = extract_features(info)
        return obs, info

    def step(self, action: int):
        self.steps += 1
        # Simulate behavior: RIGHT actions increase progress, SHOOT increases kills randomly
        if action in (2, 6, 8, 9, 14, 15):  # RIGHT and right+jump/shoot/up combos
            self.x += self.rng.uniform(2, 5)
        elif action in (1, 5):
            self.x += self.rng.uniform(-1, 1)
        # small random drift
        self.x += self.rng.uniform(0, 0.5)
        if action == 10:  # DOWN: drop on dry ground (NES Y grows downward)
            self.y += 12.0
        elif action in (4, 8, 9, 12, 13, 14, 15):  # JUMP / RIGHT+JUMP / UP+JUMP
            self.y = max(20.0, self.y - 8.0)
        if action in (3, 5, 6, 7, 9, 13, 15):  # shoot combos
            if self.rng.random() < 0.1:
                self.kills += 1
        # Deaths are handled by RAM/lives in the real env; keep mock lives stable
        # so unit tests are not randomly terminated by the large death penalty.
        terminated = False
        truncated = self.steps >= self.max_episode_steps
        info = {
            "x": self.x,
            "x_pos": self.x,
            "y_pos": self.y,
            "kills": self.kills,
            "lives": self.lives,
            "scroll": self.x,
            "progress": self.x,
            "score": self.kills * 100,
            "water_state": self.water_state,
            "in_gameplay": True,
            "game_routine": 4,
            "screen_type": 4,
        }
        if self.observation_mode == "pixels":
            dummy = self.rng.integers(0, 255, size=(240, 256, 3), dtype=np.uint8)
            # add horizontal line indicating progress
            dummy[int(self.x) % 240, :, 0] = 255
            assert self._frame_stacker is not None
            obs = self._frame_stacker.push(dummy)
        else:
            obs = extract_features(info)
        # mock reward: progress delta
        reward = 0.01
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.observation_mode == "pixels" and self._frame_stacker is not None:
            # return last frame as RGB
            f = self._frame_stacker.frames[-1] if self._frame_stacker.frames else np.zeros((84, 84), dtype=np.uint8)
            rgb = np.stack([f, f, f], axis=-1)
            return rgb
        return np.zeros((84, 84, 3), dtype=np.uint8)

    def get_fitness(self) -> float:
        # for baselines compatibility, fitness = progress + kills heuristic
        return float(self.x * 10 + self.kills * 5)

    def close(self):
        pass
