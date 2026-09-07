"""Live fitness chart using matplotlib embedded in pygame."""

from __future__ import annotations

from typing import List

import numpy as np


def prepare_chart_data(history: List[dict], key: str = "best_fitness") -> tuple[np.ndarray, np.ndarray]:
    gens = np.array([h.get("generation", i) for i, h in enumerate(history)])
    vals = np.array([h.get(key, 0) for h in history])
    return gens, vals


def render_chart_matplotlib(history: List[dict], width: int = 400, height: int = 300) -> np.ndarray:
    """Render fitness vs generation to RGB array (for blitting). Uses Agg."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    if history:
        gens, best = prepare_chart_data(history, "best_fitness")
        _, avg = prepare_chart_data(history, "mean_fitness")
        _, median = prepare_chart_data(history, "median_fitness")
        ax.plot(gens, best, label="best", color="#00ff88")
        ax.plot(gens, avg, label="avg", color="#ff8800", linestyle="--")
        ax.plot(gens, median, label="median", color="#0088ff", linestyle=":")
        ax.legend(fontsize=6)
    ax.set_xlabel("Generation", fontsize=7)
    ax.set_ylabel("Fitness", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title("Fitness vs Generation", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(pad=0.5)
    # render to array
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    # matplotlib 3.8+ removed tostring_rgb, use buffer_rgba
    try:
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)  # type: ignore
        img = buf.reshape(h, w, 3)
    except AttributeError:
        # fallback to buffer_rgba
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # type: ignore
        img_rgba = buf.reshape(h, w, 4)
        img = img_rgba[:, :, :3].copy()
    plt.close(fig)
    return img
