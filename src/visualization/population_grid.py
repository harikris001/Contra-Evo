"""Population grid layout calculations."""
from __future__ import annotations

from typing import Tuple, List


def grid_dimensions(population_size: int, layout: str = "auto") -> Tuple[int, int]:
    """Return (rows, cols) for a grid holding population_size agents + 1 global tile.

    "auto" (and any layout string that can't fit population_size + 1 tiles)
    computes rows/cols automatically: cols = ceil(sqrt(pop + 1)).
    Explicit layouts ("3x3", "2x2", "4x3", "4x4", ...) are honored only when
    they actually fit; otherwise they fall back to auto.
    """
    tiles_needed = population_size + 1  # agents + global stats tile
    if layout != "auto":
        try:
            rows, cols = (int(x) for x in layout.lower().split("x"))
            if rows > 0 and cols > 0 and rows * cols >= tiles_needed:
                return rows, cols
        except ValueError:
            pass  # unrecognized layout string -> auto
    import math

    cols = math.ceil(math.sqrt(tiles_needed))
    rows = math.ceil(tiles_needed / cols)
    return rows, cols


def tile_rects(window_w: int, window_h: int, rows: int, cols: int, padding: int = 4) -> List[Tuple[int, int, int, int]]:
    """Return list of (x,y,w,h) for each tile."""
    tile_w = (window_w - padding * (cols + 1)) // cols
    tile_h = (window_h - padding * (rows + 1)) // rows
    rects = []
    for r in range(rows):
        for c in range(cols):
            x = padding + c * (tile_w + padding)
            y = padding + r * (tile_h + padding)
            rects.append((x, y, tile_w, tile_h))
    return rects
