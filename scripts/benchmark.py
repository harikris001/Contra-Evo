#!/usr/bin/env python3
"""Benchmark worker counts: 4,6,8,10,12 and record RAM/CPU/episodes/sec."""
import argparse
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import load_config
from agent.network import NetworkConfig
from agent.genome import random_genome
from agent.genome import Individual
from environment.reward import RewardConfig
from evaluation.parallel import ParallelEvaluator
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--workers", type=str, default="4,6,8,10,12")
    p.add_argument("--output", type=str, default="runs/benchmark.json")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    worker_counts = [int(x) for x in args.workers.split(",") if x.strip()]

    net_cfg = NetworkConfig(
        cnn_channels=cfg["agent"]["cnn_channels"],
        kernel_sizes=cfg["agent"]["kernel_sizes"],
        strides=cfg["agent"]["strides"],
        dense_hidden=cfg["agent"]["dense_hidden"],
    )
    reward_cfg = RewardConfig.from_dict(cfg["reward"])
    env_cfg = cfg["env"]

    # create small population
    rng = np.random.default_rng(42)
    pop = []
    for i in range(cfg["evolution"]["population_size"]):
        g = random_genome(net_cfg, rng=rng)
        pop.append(Individual(id=i, genome=g))

    evaluator = ParallelEvaluator(env_cfg, net_cfg, reward_cfg, workers=worker_counts[0])

    results = evaluator.benchmark(pop, worker_counts=worker_counts, base_seed=cfg["evolution"]["seed"])

    # Pretty print
    print("\nBenchmark results (M4 16GB):")
    print(f"{'workers':>7} | {'duration':>8} | {'eps/sec':>7} | {'RAM%':>5} | {'CPU%':>5} | {'mean_fit':>8}")
    print("-" * 60)
    for wc, r in results.items():
        print(f"{wc:7d} | {r['duration']:8.2f} | {r['episodes_per_sec']:7.2f} | {r['ram_used_percent']:5.0f} | {r['cpu_percent']:5.0f} | {r['mean_fitness']:8.1f}")

    # recommend optimal
    # Choose max eps/sec where RAM <85%
    candidates = {k: v for k, v in results.items() if v["ram_used_percent"] < 85}
    if candidates:
        best = max(candidates.items(), key=lambda x: x[1]["episodes_per_sec"])
        print(f"\nRecommended workers for M4 16GB: {best[0]} (eps/sec {best[1]['episodes_per_sec']:.2f}, RAM {best[1]['ram_used_percent']:.0f}%)")
    else:
        print("\nNo worker count under RAM limit; smallest is recommended")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
