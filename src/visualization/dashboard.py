"""Pygame dashboard: tiled population view + global stats + chart.

Runs in separate thread/process at 10-20 FPS, independent of training.
"""
from __future__ import annotations

import time
from typing import Dict, Any, List
from pathlib import Path

import numpy as np

from .population_grid import grid_dimensions, tile_rects
from .charts import render_chart_matplotlib


class Dashboard:
    """Main dashboard window.

    For headless testing, can run without pygame (returns snapshot).
    For live mode, call run() which opens pygame window.

    Compatibility: works with Mock env frames (uint8 RGB).
    """

    def __init__(
        self,
        population_size: int = 8,
        fps: int = 15,
        layout: str = "3x3",
        window_size: tuple[int, int] = (1200, 800),
        show_charts: bool = True,
    ):
        self.population_size = population_size
        self.fps = fps
        self.layout = layout
        self.window_size = window_size
        self.show_charts = show_charts
        self.history: List[Dict[str, Any]] = []
        self.global_state: Dict[str, Any] = {}
        # lazy pygame
        self._pygame = None
        self._screen = None
        self._font = None
        self._clock = None
        self._chart_surf = None
        self._last_chart_update = 0.0

    def _init_pygame(self) -> None:
        import pygame

        self._pygame = pygame
        pygame.init()
        self._screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption(f"Contra-Evo Population Dashboard - {self.population_size} agents")
        self._font = pygame.font.SysFont("monospace", 12)
        self._font_small = pygame.font.SysFont("monospace", 10)
        self._font_big = pygame.font.SysFont("monospace", 16, bold=True)
        self._clock = pygame.time.Clock()

    def update(self, frames: Dict[int, np.ndarray | None], states: Dict[int, Dict[str, Any]], global_state: Dict[str, Any]) -> None:
        """Update internal buffers from trainer."""
        self.global_state = dict(global_state)
        # history for chart
        if global_state:
            # avoid duplicates
            if not self.history or self.history[-1].get("generation") != global_state.get("generation"):
                self.history.append(dict(global_state))
        self._frames = frames
        self._states = states

    def draw(self) -> None:
        """Draw one frame. Requires pygame initialized."""
        if self._pygame is None:
            self._init_pygame()
        assert self._pygame and self._screen and self._font
        import pygame

        self._screen.fill((18, 18, 24))
        rows, cols = grid_dimensions(self.population_size, self.layout)
        n_tiles = rows * cols
        # Last tile is always the global stats tile; agents fill the rest.
        rects = tile_rects(self.window_size[0], self.window_size[1], rows, cols, padding=6)

        for idx, rect in enumerate(rects):
            x, y, w, h = rect
            if idx == n_tiles - 1:
                self._draw_global_tile(x, y, w, h)
            elif idx < self.population_size:
                self._draw_agent_tile(idx, x, y, w, h)
            else:
                # empty filler tile (population smaller than the grid)
                pygame.draw.rect(self._screen, (24, 24, 32), (x, y, w, h), border_radius=6)

        pygame.display.flip()
        self._clock.tick(self.fps)

    @staticmethod
    def _end_overlay(state: Dict[str, Any]) -> tuple[str, tuple[int, int, int]] | None:
        """Classify a published agent state into an end-of-episode overlay.

        Returns (label, rgb) or None while the episode is running. Elite tiles
        take precedence (their stats are carried, not live).
        """
        if state.get("elite_carry"):
            return None
        status = str(state.get("status") or "").strip().lower()
        reason = str(state.get("end_reason") or "").strip().lower()
        tag = reason or status
        if tag == "dead":
            return ("DEAD", (255, 80, 80))
        if tag == "idle":
            return ("IDLE-OUT", (255, 165, 40))
        if tag in ("time", "truncated", "done", "level"):
            return ("DONE", (90, 170, 255))
        return None

    def _draw_agent_tile(self, aid: int, x: int, y: int, w: int, h: int) -> None:
        import pygame

        # background
        col = (30, 30, 40) if aid % 2 == 0 else (36, 36, 48)
        pygame.draw.rect(self._screen, col, (x, y, w, h), border_radius=6)
        pygame.draw.rect(self._screen, (60, 60, 80), (x, y, w, h), 1, border_radius=6)

        frame = getattr(self, "_frames", {}).get(aid)
        state = getattr(self, "_states", {}).get(aid, {})
        is_elite = bool(state.get("elite_carry"))
        overlay = self._end_overlay(state)

        # frame area top 70%
        frame_h = int(h * 0.65)
        if frame is not None:
            # Convert RGB to pygame surface
            try:
                # Ensure RGB uint8
                if frame.ndim == 2:
                    frame = np.stack([frame] * 3, axis=-1)
                if frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                # resize to fit
                import cv2

                face = cv2.resize(frame, (w - 8, frame_h - 8), interpolation=cv2.INTER_NEAREST)
                surf = pygame.surfarray.make_surface(np.transpose(face, (1, 0, 2)))
                self._screen.blit(surf, (x + 4, y + 4))
            except Exception:
                # fallback: fill
                pygame.draw.rect(self._screen, (20, 50, 30), (x + 4, y + 4, w - 8, frame_h - 8))

            if is_elite:
                # Dim carried elite frames: a static picture, not a live agent
                dim = pygame.Surface((w - 8, frame_h - 8), pygame.SRCALPHA)
                dim.fill((15, 15, 25, 150))
                self._screen.blit(dim, (x + 4, y + 4))

        # end-of-episode overlay: the episode is over, don't mistake the
        # frozen last frame for an idling agent
        if overlay is not None:
            label, color = overlay
            dim = pygame.Surface((w - 8, frame_h - 8), pygame.SRCALPHA)
            dim.fill((10, 10, 15, 120))
            self._screen.blit(dim, (x + 4, y + 4))
            surf = self._font_big.render(label, True, color)
            self._screen.blit(
                surf,
                (x + 4 + (w - 8 - surf.get_width()) // 2, y + 4 + (frame_h - 8 - surf.get_height()) // 2),
            )

        # elite badge
        if is_elite:
            eval_gen = state.get("evaluated_at_gen", "?")
            badge = self._font_big.render(f"ELITE \u00b7 gen {eval_gen}", True, (255, 220, 100))
            bg = pygame.Surface((badge.get_width() + 6, badge.get_height() + 2), pygame.SRCALPHA)
            bg.fill((40, 35, 10, 200))
            self._screen.blit(bg, (x + 6, y + 6))
            self._screen.blit(badge, (x + 9, y + 7))

        # text overlay bottom 30%
        txt_y = y + frame_h + 6
        first = f"Agent {aid}  {'ELITE (carried)' if is_elite else state.get('status','')}"
        lines = [
            first,
            f"fit:{state.get('fitness',0):.1f} prog:{state.get('progress',0):.0f}",
            f"kills:{state.get('kills',0):.0f} idle:{state.get('idle_steps',0):.0f}",
        ]
        if is_elite:
            lines[2] = f"eval @ gen {state.get('evaluated_at_gen','?')}"
        for i, line in enumerate(lines):
            surf = self._font_small.render(line, True, (220, 220, 220))
            self._screen.blit(surf, (x + 6, txt_y + i * 11))
        # fitness bar (normalized to the current population's best fitness)
        fit = state.get("fitness", 0)
        states = getattr(self, "_states", {}) or {}
        max_fit = max(
            (float(s.get("fitness", 0)) for s in states.values() if isinstance(s, dict)),
            default=0.0,
        )
        denom = max_fit if max_fit > 0 else 1000.0
        bar_w = int(np.clip(fit / denom * (w - 12), 0, w - 12))
        pygame.draw.rect(self._screen, (0, 180, 120), (x + 6, y + h - 8, bar_w, 4), border_radius=2)

    def _draw_global_tile(self, x: int, y: int, w: int, h: int) -> None:
        import pygame

        pygame.draw.rect(self._screen, (40, 30, 50), (x, y, w, h), border_radius=6)
        pygame.draw.rect(self._screen, (100, 80, 120), (x, y, w, h), 1, border_radius=6)
        gs = self.global_state
        title = self._font_big.render(f"GEN {gs.get('generation',0)}", True, (255, 220, 100))
        self._screen.blit(title, (x + 8, y + 8))
        lines = [
            f"best: {gs.get('best_fitness',0):.1f} (#{gs.get('best_id',0)})",
            f"avg: {gs.get('mean_fitness',0):.1f} med:{gs.get('median_fitness',0):.1f}",
            f"worst:{gs.get('worst_fitness',0):.1f}",
            f"diversity:{gs.get('diversity',0):.3f}",
            f"mut_rate:{gs.get('mutation_rate',0):.3f}",
            f"eps/sec:{gs.get('episodes_per_sec',0):.1f} FPS:{gs.get('emulator_fps',0):.0f}",
            f"RAM:{gs.get('ram_percent',0):.1f}% CPU:{gs.get('cpu_percent',0):.0f}%",
            f"gen_time:{gs.get('generation_duration',0):.1f}s",
        ]
        for i, line in enumerate(lines):
            color = (200, 200, 220) if i < 5 else (160, 200, 160)
            surf = self._font_small.render(line, True, color)
            self._screen.blit(surf, (x + 8, y + 30 + i * 12))

        # chart thumbnail
        if self.show_charts and self.history and (time.time() - self._last_chart_update > 0.5):
            try:
                chart_img = render_chart_matplotlib(self.history, width=w - 16, height= int(h*0.35))
                import cv2

                ch = cv2.resize(chart_img, (w - 16, int(h * 0.35)))
                surf = self._pygame.surfarray.make_surface(np.transpose(ch, (1, 0, 2)))
                self._screen.blit(surf, (x + 8, y + h - int(h * 0.35) - 8))
                self._last_chart_update = time.time()
                self._chart_surf = surf
            except Exception as e:
                # fallback text
                surf = self._font_small.render(f"chart err {e}", True, (180, 120, 120))
                self._screen.blit(surf, (x + 8, y + h - 20))
        elif self._chart_surf is not None:
            self._screen.blit(self._chart_surf, (x + 8, y + h - int(h * 0.35) - 8))

    def handle_events(self) -> bool:
        """Return False if quit requested."""
        if self._pygame is None:
            return True
        import pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def run(self, data_fn, global_fn, max_frames: int | None = None) -> None:
        """Blocking run loop. data_fn returns (frames, states), global_fn returns global_state."""
        self._init_pygame()
        frame = 0
        running = True
        while running:
            running = self.handle_events()
            frames, states = data_fn()
            global_state = global_fn()
            self.update(frames, states, global_state)
            self.draw()
            frame += 1
            if max_frames and frame >= max_frames:
                break
        self.close()

    def close(self) -> None:
        if self._pygame:
            try:
                self._pygame.quit()
            except:
                pass


def headless_snapshot(
    population_size: int,
    frames: Dict[int, np.ndarray | None],
    states: Dict[int, Dict[str, Any]],
    global_state: Dict[str, Any],
) -> str:
    """Text snapshot for headless/log mode."""
    lines = [f"=== GEN {global_state.get('generation',0)} ==="]
    lines.append(
        f"best {global_state.get('best_fitness',0):.1f} avg {global_state.get('mean_fitness',0):.1f} div {global_state.get('diversity',0):.3f}"
    )
    for aid in range(population_size):
        s = states.get(aid, {})
        lines.append(f"  Agent {aid}: fit {s.get('fitness',0):.1f} prog {s.get('progress',0):.0f} kills {s.get('kills',0):.0f}")
    return "\n".join(lines)
