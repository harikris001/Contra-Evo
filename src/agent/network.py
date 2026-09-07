"""Compact CNN for evolutionary search (<100k params).

CPU-first, numpy-only (no torch required for GA). MPS optional later.

Architecture:
84x84x4 -> Conv(16,8,4) -> ReLU -> Conv(32,4,2) -> ReLU -> Flatten -> Dense64 -> Dense32 -> 8 logits

Params estimate: ~20k-60k depending on channels.

All weights stored flat for GA genome operations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


@dataclass
class NetworkConfig:
    input_shape: Tuple[int, int, int] = (4, 84, 84)  # C,H,W
    cnn_channels: List[int] = None  # type: ignore
    kernel_sizes: List[int] = None  # type: ignore
    strides: List[int] = None  # type: ignore
    dense_hidden: List[int] = None  # type: ignore
    num_actions: int = 8
    activation: str = "relu"

    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [8, 16]
        if self.kernel_sizes is None:
            self.kernel_sizes = [8, 4]
        if self.strides is None:
            self.strides = [4, 2]
        if self.dense_hidden is None:
            self.dense_hidden = [64, 32]


def conv_output_size(h_w: Tuple[int, int], kernel: int, stride: int) -> Tuple[int, int]:
    h, w = h_w
    return (h - kernel) // stride + 1, (w - kernel) // stride + 1


def count_parameters(cfg: NetworkConfig) -> int:
    """Count params for given config without instantiating."""
    c, h, w = cfg.input_shape
    total = 0
    # conv layers
    for i, (ch, k, s) in enumerate(zip(cfg.cnn_channels, cfg.kernel_sizes, cfg.strides)):
        # weights: ch * c * k * k + biases ch
        total += ch * c * k * k + ch
        # update spatial
        h, w = conv_output_size((h, w), k, s)
        c = ch
    flat = c * h * w
    prev = flat
    for hd in cfg.dense_hidden:
        total += prev * hd + hd
        prev = hd
    total += prev * cfg.num_actions + cfg.num_actions
    return total


def _layer_entries(cfg: NetworkConfig) -> List[Tuple[str, float, int]]:
    """(name, init_scale, num_params) per parameter block, matching _build order."""
    c, h, w = cfg.input_shape
    entries: List[Tuple[str, float, int]] = []
    for i, (ch, k, s) in enumerate(zip(cfg.cnn_channels, cfg.kernel_sizes, cfg.strides)):
        fan_in = c * k * k
        fan_out = ch * k * k
        scale = math.sqrt(2.0 / (fan_in + fan_out))
        entries.append((f"conv{i}_w", scale, ch * c * k * k))
        # biases init at zero: scale relative to layer weight scale (small shifts)
        entries.append((f"conv{i}_b", scale * _BIAS_SCALE_REL, ch))
        h, w = conv_output_size((h, w), k, s)
        c = ch
    flat = c * h * w
    prev = flat
    for j, hd in enumerate(cfg.dense_hidden):
        scale = math.sqrt(2.0 / (prev + hd))
        entries.append((f"dense{j}_w", scale, prev * hd))
        entries.append((f"dense{j}_b", scale * _BIAS_SCALE_REL, hd))
        prev = hd
    scale = math.sqrt(2.0 / (prev + cfg.num_actions))
    entries.append(("out_w", scale, prev * cfg.num_actions))
    entries.append(("out_b", scale * _BIAS_SCALE_REL, cfg.num_actions))
    return entries


_BIAS_SCALE_REL = 0.1


def parameter_scale_vector(cfg: NetworkConfig) -> np.ndarray:
    """Per-gene Xavier init-scale vector matching get_genome() flattened order.

    Used by mutation/diversity so noise is proportional to each layer's weight
    scale (mutation_std becomes a relative multiplier) instead of one absolute
    sigma for all 89k genes.
    """
    entries = _layer_entries(cfg)
    entries.sort(key=lambda e: e[0])  # match get_genome(): sorted(self.params.keys())
    return np.concatenate([np.full(size, s, dtype=np.float32) for _, s, size in entries])


class CompactCNN:
    """Numpy CNN with manual conv2d (CPU optimized, small)."""

    def __init__(self, config: NetworkConfig, rng: np.random.Generator | None = None):
        self.cfg = config
        self.rng = rng or np.random.default_rng()
        # Build layers: store weights dict
        self.params: dict[str, np.ndarray] = {}
        self.shapes: dict[str, Tuple[int, ...]] = {}
        self.scale_values: dict[str, float] = {}
        self._build()
        # cache conv indices? Use simple im2col? For speed with small nets, do naive but vectorized.
        # We will implement fast conv via numpy tensordot + stride tricks if needed.
        # For now, implement straightforward vectorized conv.

    def _build(self) -> None:
        c, h, w = self.cfg.input_shape
        # init with Xavier
        for i, (ch, k, s) in enumerate(zip(self.cfg.cnn_channels, self.cfg.kernel_sizes, self.cfg.strides)):
            fan_in = c * k * k
            fan_out = ch * k * k
            scale = math.sqrt(2.0 / (fan_in + fan_out))
            w_shape = (ch, c, k, k)
            b_shape = (ch,)
            self.params[f"conv{i}_w"] = self.rng.normal(0, scale, size=w_shape).astype(np.float32)
            self.params[f"conv{i}_b"] = np.zeros(b_shape, dtype=np.float32)
            self.shapes[f"conv{i}_w"] = w_shape
            self.shapes[f"conv{i}_b"] = b_shape
            self.scale_values[f"conv{i}_w"] = scale
            self.scale_values[f"conv{i}_b"] = scale * 0.1
            h, w = conv_output_size((h, w), k, s)
            c = ch
        flat = c * h * w
        self._conv_out_shape = (c, h, w)
        self._flat_dim = flat
        prev = flat
        for j, hd in enumerate(self.cfg.dense_hidden):
            scale = math.sqrt(2.0 / (prev + hd))
            self.params[f"dense{j}_w"] = self.rng.normal(0, scale, size=(prev, hd)).astype(np.float32)
            self.params[f"dense{j}_b"] = np.zeros((hd,), dtype=np.float32)
            self.shapes[f"dense{j}_w"] = (prev, hd)
            self.shapes[f"dense{j}_b"] = (hd,)
            self.scale_values[f"dense{j}_w"] = scale
            self.scale_values[f"dense{j}_b"] = scale * 0.1
            prev = hd
        # output
        scale = math.sqrt(2.0 / (prev + self.cfg.num_actions))
        self.params["out_w"] = self.rng.normal(0, scale, size=(prev, self.cfg.num_actions)).astype(np.float32)
        self.params["out_b"] = np.zeros((self.cfg.num_actions,), dtype=np.float32)
        self.shapes["out_w"] = (prev, self.cfg.num_actions)
        self.shapes["out_b"] = (self.cfg.num_actions,)
        self.scale_values["out_w"] = scale
        self.scale_values["out_b"] = scale * 0.1

    def get_scales(self) -> np.ndarray:
        """Per-gene init-scale vector in the same order as get_genome()."""
        parts = []
        for k in sorted(self.scale_values.keys()):
            size = int(np.prod(self.shapes[k]))
            parts.append(np.full(size, self.scale_values[k], dtype=np.float32))
        return np.concatenate(parts)

    def get_genome(self) -> np.ndarray:
        """Flatten all params to 1D float32 vector."""
        parts = []
        for k in sorted(self.params.keys()):
            parts.append(self.params[k].ravel())
        return np.concatenate(parts).astype(np.float32)

    def set_genome(self, genome: np.ndarray) -> None:
        """Load flat vector into params."""
        offset = 0
        for k in sorted(self.params.keys()):
            shape = self.shapes[k]
            size = int(np.prod(shape))
            self.params[k] = genome[offset : offset + size].reshape(shape).astype(np.float32)
            offset += size
        assert offset == genome.size, f"genome size mismatch {offset} vs {genome.size}"

    def num_parameters(self) -> int:
        return sum(int(np.prod(s)) for s in self.shapes.values())

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.

        Args:
            x: (C,H,W) uint8 0-255 or float32 0-1, or (N,C,H,W)
        Returns:
            logits (num_actions,) or (N, num_actions)
        """
        # normalize
        if x.dtype == np.uint8:
            x = x.astype(np.float32) / 255.0
        # add batch dim if needed
        batched = x.ndim == 4
        if not batched:
            x = x[None, ...]  # (1,C,H,W)
        # conv layers
        for i in range(len(self.cfg.cnn_channels)):
            w = self.params[f"conv{i}_w"]  # (out_c, in_c, k, k)
            b = self.params[f"conv{i}_b"]  # (out_c,)
            x = self._conv2d(x, w, b, stride=self.cfg.strides[i])
            x = relu(x)
        # flatten
        n = x.shape[0]
        x = x.reshape(n, -1)
        for j in range(len(self.cfg.dense_hidden)):
            w = self.params[f"dense{j}_w"]
            b = self.params[f"dense{j}_b"]
            x = x @ w + b
            x = relu(x)
        logits = x @ self.params["out_w"] + self.params["out_b"]
        if not batched:
            return logits[0]
        return logits

    def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
        logits = self.forward(obs)
        if deterministic:
            return int(np.argmax(logits))
        probs = softmax(logits)
        return int(self.rng.choice(len(probs), p=probs))

    def _conv2d(self, x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int = 1) -> np.ndarray:
        """Naive but correct conv2d.

        x: (N, C_in, H, W)
        w: (C_out, C_in, K, K)
        b: (C_out,)
        stride: int
        Returns: (N, C_out, H_out, W_out)
        """
        N, C_in, H, W = x.shape
        C_out, _, K, _ = w.shape
        H_out = (H - K) // stride + 1
        W_out = (W - K) // stride + 1
        # Use im2col via stride tricks for efficiency? Keep simple loops over H_out*W_out but vectorized over N and C
        # Implement via tensordot: unfold x into patches
        # Create strided view
        # x_patches shape: (N, C_in, H_out, W_out, K, K) via as_strided? Instead do loop over K for clarity and small sizes
        out = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)
        # Vectorized: for each output position, compute dot
        # Optimize: reshape w to (C_out, C_in*K*K)
        w_col = w.reshape(C_out, -1)  # (C_out, C_in*K*K)
        # Unfold x
        # Build col matrix: (N, H_out*W_out, C_in*K*K)
        # Use loops over H_out,W_out but numpy for N
        # For small 84x84 with stride 4, H_out~20 => 400 positions, feasible
        col = np.zeros((N, H_out * W_out, C_in * K * K), dtype=np.float32)
        idx = 0
        for i in range(H_out):
            for j in range(W_out):
                patch = x[:, :, i * stride : i * stride + K, j * stride : j * stride + K]  # (N, C_in, K, K)
                col[:, idx, :] = patch.reshape(N, -1)
                idx += 1
        # Compute: (N, H_out*W_out, C_out) = col @ w_col.T
        out_col = col @ w_col.T  # (N, H_out*W_out, C_out)
        out_col = out_col + b  # broadcast
        # reshape to (N, C_out, H_out, W_out)
        out = out_col.transpose(0, 2, 1).reshape(N, C_out, H_out, W_out)
        return out
