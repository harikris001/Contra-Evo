"""Configurable reward/fitness system.

Strongly reward forward progress, moderate kills/survival, penalize idle/death.
Do not allow stationary farming.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


# Contra title/menu routines are low indices; gameplay is typically >= 4
# (game_routine $18 / level routine $2C after START).
_GAMEPLAY_ROUTINE_MIN = 4
_REASONABLE_LIVES_MAX = 6


# Discrete DOWN in playable action set (actions.py)
_DOWN_ACTION = 10
# edge_fall bits 5–7: walking off / falling through a ledge
_EDGE_FALL_MASK = 0xE0
_PLAYER_STATE_DEAD = 2
_IDLE_END_PENALTY_CAP = -25.0


@dataclass
class RewardConfig:
    progress_coef: float = 10.0
    kill_coef: float = 5.0
    boss_damage_coef: float = 2.0
    survival_coef: float = 0.01
    death_penalty: float = 0.0
    idle_penalty: float = -4.0
    level_complete_bonus: float = 1000.0
    idle_threshold_steps: int = 24
    idle_timeout_steps: int = 320
    idle_min_progress: float = 2.0
    idle_combat_scale: float = 0.05
    idle_forfeit_scale: float = 1.0
    idle_progress_ref: float = 64.0
    terminate_on_death: bool = True
    death_forfeit_progress: bool = False
    death_progress_scale: float = 1.0
    score_coef: float = 8.0
    no_combat_progress_scale: float = 0.2
    combat_grace_progress: float = 64.0
    score_points_per_kill: float = 100.0
    max_score_delta_per_step: float = 800.0
    max_kills_from_score_per_step: int = 4
    fall_y_threshold: float = 6.0
    fall_no_water_penalty: float = -2.0
    down_no_water_penalty: float = -0.5
    climb_bonus: float = 0.5

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RewardConfig":
        return cls(
            progress_coef=d.get("progress_coef", 10.0),
            kill_coef=d.get("kill_coef", 5.0),
            boss_damage_coef=d.get("boss_damage_coef", 2.0),
            survival_coef=d.get("survival_coef", 0.01),
            death_penalty=d.get("death_penalty", 0.0),
            idle_penalty=d.get("idle_penalty", -4.0),
            level_complete_bonus=d.get("level_complete_bonus", 1000.0),
            idle_threshold_steps=d.get("idle_threshold_steps", 24),
            idle_timeout_steps=d.get("idle_timeout_steps", 320),
            idle_min_progress=d.get("idle_min_progress", 2.0),
            idle_combat_scale=d.get("idle_combat_scale", 0.05),
            idle_forfeit_scale=d.get("idle_forfeit_scale", 1.0),
            idle_progress_ref=d.get("idle_progress_ref", 64.0),
            terminate_on_death=d.get("terminate_on_death", True),
            death_forfeit_progress=d.get("death_forfeit_progress", False),
            death_progress_scale=d.get("death_progress_scale", 1.0),
            score_coef=d.get("score_coef", 8.0),
            no_combat_progress_scale=d.get("no_combat_progress_scale", 0.2),
            combat_grace_progress=d.get("combat_grace_progress", 64.0),
            score_points_per_kill=d.get("score_points_per_kill", 100.0),
            max_score_delta_per_step=d.get("max_score_delta_per_step", 800.0),
            max_kills_from_score_per_step=d.get("max_kills_from_score_per_step", 4),
            fall_y_threshold=d.get("fall_y_threshold", 6.0),
            fall_no_water_penalty=d.get("fall_no_water_penalty", -2.0),
            down_no_water_penalty=d.get("down_no_water_penalty", -0.5),
            climb_bonus=d.get("climb_bonus", 0.5),
        )


@dataclass
class RewardState:
    max_progress: float = 0.0
    progress_start: float | None = None  # progress at the spawn point (baseline)
    idle_steps: int = 0
    kills: int = 0
    fitness: float = 0.0
    total_reward: float = 0.0
    steps: int = 0
    deaths: int = 0
    last_raw_x: float | None = None
    wrap_count: int = 0
    last_score: float | None = None
    in_gameplay: bool = False
    should_truncate: bool = False
    should_terminate: bool = False
    screen_type: float = 0.0
    spawned: bool = False
    score_gained: float = 0.0
    last_y: float | None = None
    end_reason: str = ""


class RewardCalculator:
    """Calculates per-step reward and tracks fitness.

    Uses info dict from environment which may contain:
    - x, x_pos, scroll, x_scroll, level_screen, level_scroll, progress
    - score (kill/combat proxy), kills
    - lives, deaths, game_over / game_status
    - game_routine, screen_type
    """

    def __init__(self, config: RewardConfig):
        self.cfg = config
        self.state = RewardState()

    def reset(self) -> None:
        self.state = RewardState()

    def step(
        self,
        info: Dict[str, Any],
        done: bool,
        prev_info: Dict[str, Any] | None = None,
        action: int | None = None,
    ) -> float:
        """Compute step reward."""
        reward = 0.0
        in_game = is_gameplay(info)
        self.state.in_gameplay = in_game
        self.state.screen_type = float(_first_number(info, ("screen_type", "game_routine"), 0.0))

        if not in_game:
            self.state.steps += 1
            self.state.total_reward += reward
            self.state.fitness += reward
            return reward

        raw_x = _first_number(info, ("x_pos", "player_x", "x"), None)
        scroll = _first_number(info, ("level_scroll", "x_scroll"), None)
        screen = _first_number(info, ("level_screen",), None)
        progress = self._extract_progress(info)
        just_spawned = False
        if not self.state.spawned:
            if (raw_x is not None and 8 <= raw_x <= 248) or (screen is not None and screen > 0) or (
                scroll is not None and scroll > 0
            ):
                self.state.spawned = True
                just_spawned = True
                # Baseline progress at the spawn point: with scenario start
                # states the agent may begin mid-level, so only progress
                # BEYOND the start earns reward (no free credit for the
                # distance already covered before the start state).
                self.state.progress_start = progress
                self.state.max_progress = progress

        delta_progress = max(0.0, progress - self.state.max_progress)
        min_move = max(0.0, float(self.cfg.idle_min_progress))
        # Falling off a ledge must NOT count as movement: in vertically
        # scrolling sections the camera follows the fall and the scroll byte
        # advances, which otherwise REWARDS jumping into pits (+progress > the
        # small fall penalty). Falling counts as idling instead.
        edge_now = _first_number(info, ("edge_fall",), 0.0) or 0.0
        falling_now = (int(edge_now) & _EDGE_FALL_MASK) != 0
        moving = (delta_progress >= min_move) and not falling_now
        stalling = self.state.spawned and not moving
        idle_if_stall = self.state.idle_steps + 1 if stalling else 0
        combat_scale = 1.0
        if stalling and idle_if_stall >= self.cfg.idle_threshold_steps:
            combat_scale = float(self.cfg.idle_combat_scale)

        kills_now = self._extract_kills(info)
        delta_kills = max(0, kills_now - self.state.kills)
        self.state.kills = max(self.state.kills, kills_now)

        score = _first_number(info, ("score",), None)
        delta_score = 0.0
        if score is not None:
            score = float(score)
            if self.state.last_score is None:
                self.state.last_score = score
            else:
                raw_delta = score - self.state.last_score
                max_delta = max(0.0, float(self.cfg.max_score_delta_per_step))
                if raw_delta > max_delta:
                    # Leftover demo/high-score RAM or a 2-byte BCD glitch — resync, no credit.
                    self.state.last_score = score
                elif raw_delta > 0:
                    delta_score = raw_delta
                    if delta_kills == 0:
                        # No RAM kill counter: infer kills from score points
                        # (round so a 500pt enemy doesn't credit 5 kills).
                        per = max(self.cfg.score_points_per_kill, 1.0)
                        inferred = int(round(delta_score / per))
                        cap = max(0, int(self.cfg.max_kills_from_score_per_step))
                        inferred = min(inferred, cap)
                        if inferred > 0:
                            delta_kills += inferred
                            self.state.kills += inferred
                    self.state.last_score = score
                else:
                    self.state.last_score = score

        if delta_kills > 0:
            reward += delta_kills * self.cfg.kill_coef * combat_scale
        if delta_score > 0:
            reward += delta_score * self.cfg.score_coef * combat_scale
            self.state.score_gained += delta_score

        if moving:
            scale = 1.0
            rel = self._rel_progress()
            if (
                rel >= self.cfg.combat_grace_progress
                and self.state.score_gained <= 0
                and self.state.kills <= 0
            ):
                scale = self.cfg.no_combat_progress_scale
            reward += delta_progress * self.cfg.progress_coef * scale
            self.state.max_progress = max(self.state.max_progress, progress)
            self.state.idle_steps = 0
        else:
            if delta_progress > 0:
                self.state.max_progress = max(self.state.max_progress, progress)
            if self.state.spawned and not just_spawned:
                self.state.idle_steps += 1
                if self.state.idle_steps >= self.cfg.idle_threshold_steps:
                    rel = self._rel_progress()
                    timeout = max(float(self.cfg.idle_timeout_steps), 1.0)
                    grow = 1.0 + (self.state.idle_steps / timeout)
                    ref = max(float(self.cfg.idle_progress_ref), 1.0)
                    place = 1.0 + (rel / ref)
                    reward += self.cfg.idle_penalty * grow * place
                if self.state.idle_steps >= self.cfg.idle_timeout_steps:
                    self.state.should_truncate = True
                    self.state.end_reason = "idle"
                    reward -= (
                        self.cfg.idle_forfeit_scale
                        * self._rel_progress()
                        * self.cfg.progress_coef
                    )

        if prev_info is not None:
            boss_before = prev_info.get("boss_hp", prev_info.get("bossHp", None))
            boss_now = info.get("boss_hp", info.get("bossHp", None))
            if boss_before is not None and boss_now is not None:
                dmg = max(0, float(boss_before) - float(boss_now))
                reward += dmg * self.cfg.boss_damage_coef

        reward += self.cfg.survival_coef

        if self.state.spawned:
            reward += self._ledge_vs_water(info, delta_progress, action)

        died = _detect_death(info, prev_info)

        if died:
            reward += self.cfg.death_penalty
            if self.cfg.death_forfeit_progress:
                reward -= (
                    self.cfg.death_progress_scale
                    * self._rel_progress()
                    * self.cfg.progress_coef
                )
            self.state.deaths += 1
            if self.cfg.terminate_on_death:
                self.state.should_terminate = True
                self.state.end_reason = "dead"

        if info.get("level_complete", False) or info.get("stage_clear", False) or info.get("done_level", False):
            reward += self.cfg.level_complete_bonus

        if done and self.state.idle_steps >= self.cfg.idle_threshold_steps:
            reward += max(self.cfg.death_penalty * 0.5, _IDLE_END_PENALTY_CAP)

        self.state.total_reward += reward
        self.state.fitness += reward
        self.state.steps += 1
        return reward

    def get_fitness(self) -> float:
        return self.state.fitness

    def _rel_progress(self) -> float:
        """Progress beyond the spawn baseline (0 for power-on starts)."""
        if self.state.progress_start is None:
            return max(0.0, self.state.max_progress)
        return max(0.0, self.state.max_progress - self.state.progress_start)

    def get_stats(self) -> Dict[str, float]:
        return {
            "fitness": self.state.fitness,
            "progress": self._rel_progress(),
            "abs_progress": self.state.max_progress,
            "kills": float(self.state.kills),
            "steps": float(self.state.steps),
            "deaths": float(self.state.deaths),
            "idle_steps": float(self.state.idle_steps),
            "screen_type": self.state.screen_type,
            "in_gameplay": float(self.state.in_gameplay),
            "score_gained": float(self.state.score_gained),
        }

    def _extract_progress(self, info: Dict[str, Any]) -> float:
        screen = _first_number(info, ("level_screen",), None)
        scroll = _first_number(info, ("level_scroll", "x_scroll", "scroll_x", "scroll"), None)
        if screen is not None and scroll is not None:
            return float(screen) * 256.0 + float(scroll)

        raw = _first_number(
            info,
            ("x_pos", "player_x", "x", "progress", "distance"),
            None,
        )
        if raw is None:
            return float(self.state.max_progress)

        raw = float(raw)
        if self.state.last_raw_x is not None:
            delta = raw - self.state.last_raw_x
            if delta < -128:
                self.state.wrap_count += 1
            elif delta > 128:
                self.state.wrap_count = max(0, self.state.wrap_count - 1)
        self.state.last_raw_x = raw
        return self.state.wrap_count * 256.0 + raw

    def _ledge_vs_water(self, info: Dict[str, Any], delta_progress: float, action: int | None) -> float:
        extra = 0.0
        in_water = _in_water(info)
        y = _first_number(info, ("y_pos",), None)
        edge = _first_number(info, ("edge_fall",), 0.0) or 0.0
        falling_ledge = (int(edge) & _EDGE_FALL_MASK) != 0

        if y is not None:
            if self.state.last_y is not None and not in_water:
                dy = float(y) - self.state.last_y
                if dy > self.cfg.fall_y_threshold:
                    extra += self.cfg.fall_no_water_penalty
                if dy < 0 and delta_progress > 0:
                    extra += self.cfg.climb_bonus
            self.state.last_y = float(y)

        if not in_water:
            if falling_ledge:
                extra += self.cfg.fall_no_water_penalty
            if action is not None and int(action) == _DOWN_ACTION:
                extra += self.cfg.down_no_water_penalty
        return extra

    def _extract_kills(self, info: Dict[str, Any]) -> int:
        for k in ("kills", "enemies_defeated", "enemies_killed", "kill_count", "enemy_kills"):
            if k in info:
                try:
                    return int(info[k])
                except (TypeError, ValueError):
                    pass
            for ik in info:
                if ik.lower() == k:
                    try:
                        return int(info[ik])
                    except (TypeError, ValueError):
                        pass
        return self.state.kills


def is_gameplay(info: Dict[str, Any]) -> bool:
    """True once Contra has left title/demo and entered a level routine."""
    if info.get("in_gameplay") is True:
        return True
    demo = _first_number(info, ("demo_mode",), None)
    if demo is not None and int(demo) == 1:
        return False
    routine = _first_number(info, ("game_routine", "screen_type"), None)
    if routine is not None:
        return int(routine) >= _GAMEPLAY_ROUTINE_MIN
    lives = _first_number(info, ("lives", "life"), None)
    if lives is not None and 0 <= int(lives) <= _REASONABLE_LIVES_MAX:
        return True
    # Mock / tests that only pass x / x_pos
    if any(k in info for k in ("x", "x_pos", "progress", "level_screen")):
        return True
    return False


def _in_water(info: Dict[str, Any]) -> bool:
    water = _first_number(info, ("water_state",), None)
    if water is None:
        return False
    return (int(water) >> 2) & 1 == 1


def _detect_death(info: Dict[str, Any], prev_info: Dict[str, Any] | None) -> bool:
    lives_now = _first_number(info, ("lives", "life"), None)
    lives_prev = _first_number(prev_info or {}, ("lives", "life"), None)
    if lives_now is not None and lives_prev is not None and lives_now < lives_prev:
        if 0 <= lives_prev <= _REASONABLE_LIVES_MAX and 0 <= lives_now <= _REASONABLE_LIVES_MAX:
            return True
    game_over_now = _game_over(info)
    game_over_prev = _game_over(prev_info) if prev_info else False
    if game_over_now and not game_over_prev:
        return True
    if info.get("dead", False) or info.get("is_dead", False):
        return True
    flag_now = int(_first_number(info, ("death_flag",), 0.0) or 0) & 1
    flag_prev = int(_first_number(prev_info or {}, ("death_flag",), 0.0) or 0) & 1
    if flag_now and not flag_prev:
        return True
    state_now = _first_number(info, ("player_state",), None)
    state_prev = _first_number(prev_info or {}, ("player_state",), None)
    if state_now is not None and int(state_now) == _PLAYER_STATE_DEAD:
        if state_prev is None or int(state_prev) != _PLAYER_STATE_DEAD:
            return True
    return False


def _game_over(info: Dict[str, Any]) -> bool:
    # Only trust an explicit boolean flag. The raw "game_status" RAM byte's
    # semantics are not verified for Contra (treating value 1 as game-over
    # produced spurious deaths), so numeric status bytes are ignored here.
    v = info.get("game_over")
    return isinstance(v, bool) and v


def _first_number(info: Dict[str, Any], keys: tuple[str, ...], default):
    for k in keys:
        if k in info:
            try:
                return float(info[k])
            except (TypeError, ValueError):
                pass
        for ik in info:
            if ik.lower() == k.lower():
                try:
                    return float(info[ik])
                except (TypeError, ValueError):
                    pass
    return default
