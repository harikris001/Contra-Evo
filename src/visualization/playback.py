"""Single-agent playback window (pygame)."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


class PlaybackWindow:
    """Scale Contra frames into a pygame window with a HUD overlay."""

    def __init__(self, scale: int = 3, fps: int = 30, caption: str = "Contra-Evo — trained agent"):
        import pygame

        self.fps = fps
        self.scale = max(1, int(scale))
        self._pygame = pygame
        pygame.init()
        self._base_w, self._base_h = 256, 240
        self._hud_h = 72
        w = self._base_w * self.scale
        h = self._base_h * self.scale + self._hud_h
        self._screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption(caption)
        self._font = pygame.font.SysFont("monospace", 16)
        self._font_small = pygame.font.SysFont("monospace", 13)
        self._clock = pygame.time.Clock()
        self.paused = False
        self.quit_requested = False

    def poll(self) -> bool:
        """Handle events. Returns False if the window should close."""
        pygame = self._pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_requested = True
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.quit_requested = True
                    return False
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
        return True

    def draw(self, frame: np.ndarray | None, hud: Dict[str, Any]) -> None:
        pygame = self._pygame
        rgb = _to_rgb(frame)
        h, w = rgb.shape[:2]
        surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        dest_w = self._base_w * self.scale
        dest_h = self._base_h * self.scale
        if (w, h) != (dest_w, dest_h):
            surf = pygame.transform.scale(surf, (dest_w, dest_h))
        self._screen.fill((12, 12, 18))
        self._screen.blit(surf, (0, 0))
        hud_y = dest_h
        pygame.draw.rect(self._screen, (24, 24, 32), (0, hud_y, dest_w, self._hud_h))
        lines = [
            f"ep {hud.get('episode', 0)}  step {hud.get('steps', 0)}  "
            f"fit {hud.get('fitness', 0):.1f}  prog {hud.get('progress', 0):.0f}  "
            f"kills {hud.get('kills', 0):.0f}  lives {hud.get('lives', '?')}",
            f"action {hud.get('action_name', '?')}  screen {hud.get('screen_type', 0):.0f}  "
            f"{'[PAUSED space]' if self.paused else 'space=pause  esc=quit'}",
        ]
        for i, line in enumerate(lines):
            text = self._font_small.render(line, True, (230, 230, 230))
            self._screen.blit(text, (12, hud_y + 10 + i * 24))
        pygame.display.flip()
        self._clock.tick(self.fps)

    def close(self) -> None:
        try:
            self._pygame.quit()
        except Exception:
            pass


def _to_rgb(frame: np.ndarray | None) -> np.ndarray:
    if frame is None:
        return np.zeros((240, 256, 3), dtype=np.uint8)
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        # channel-first (C,H,W)
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr
