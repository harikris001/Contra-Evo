"""Parallel evaluation across population.

- Persistent ProcessPoolExecutor (spawn context): workers are created once and
  reused across generations instead of re-importing retro/numpy/cv2 every gen.
- Each worker caches its Evaluator + emulator and only rebuilds when the env
  config changes (e.g. curriculum stage transition).
- Fair seeds: every individual in a generation is evaluated on the same seed
  set, so fitness ranking compares like with like.
- Elite carry-forward: individuals already marked "evaluated" (elites copied
  into the new population under a fixed seed policy) are skipped.
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed, Future, BrokenExecutor
from multiprocessing import get_context
from typing import List, Dict, Any, Tuple

import numpy as np

from agent.genome import Individual
from agent.network import NetworkConfig
from environment.reward import RewardConfig
from evaluation.evaluator import Evaluator


# ---------------------------------------------------------------------------
# Worker-side cached evaluator (module global, one per worker process)
# ---------------------------------------------------------------------------
_WORKER_STATE: Dict[str, Any] = {"evaluator": None, "sig": None}


def _config_sig(env_cfg: Dict[str, Any], net_cfg_dict: Dict[str, Any], reward_cfg_dict: Dict[str, Any]) -> str:
    return repr(
        (
            sorted(env_cfg.items()),
            sorted(net_cfg_dict.items()),
            sorted(reward_cfg_dict.items()),
        )
    )


def _get_worker_evaluator(
    env_cfg: Dict[str, Any],
    net_cfg_dict: Dict[str, Any],
    reward_cfg_dict: Dict[str, Any],
) -> Evaluator:
    sig = _config_sig(env_cfg, net_cfg_dict, reward_cfg_dict)
    if _WORKER_STATE["evaluator"] is None or _WORKER_STATE["sig"] != sig:
        old = _WORKER_STATE["evaluator"]
        if old is not None:
            try:
                old.close()
            except Exception:
                pass
        net_cfg = NetworkConfig(**net_cfg_dict)
        reward_cfg = RewardConfig.from_dict(reward_cfg_dict)
        _WORKER_STATE["evaluator"] = Evaluator(dict(env_cfg), net_cfg, reward_cfg)
        _WORKER_STATE["sig"] = sig
    return _WORKER_STATE["evaluator"]


def _run_episodes(
    evaluator: Evaluator,
    genome_bytes: bytes,
    ind_id: int,
    seeds: List[int],
    publish=None,
) -> Tuple[float, Dict[str, Any], int]:
    """Evaluate one genome on one or more seeds; aggregate to a single result."""
    vec = np.frombuffer(genome_bytes, dtype=np.float32).copy()
    from agent.genome import Genome, Individual

    fitnesses = []
    stats_list = []
    total_steps = 0
    for seed in seeds:
        genome = Genome(vector=vec, generation=0)
        ind = Individual(id=ind_id, genome=genome, fitness=-np.inf)
        fitness, stats, steps = evaluator.evaluate(ind, seed=seed, publish_frame=publish)
        fitnesses.append(fitness)
        stats_list.append(stats)
        total_steps += steps
    fitness = float(np.mean(fitnesses))
    stats = _merge_stats(stats_list)
    steps = int(round(total_steps / max(1, len(seeds))))
    _downscale_frame(stats)
    return fitness, stats, steps


def _merge_stats(stats_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(stats_list) == 1:
        return stats_list[0]
    merged: Dict[str, Any] = {}
    for key in stats_list[0].keys():
        values = [s.get(key) for s in stats_list]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            merged[key] = float(np.mean(values))
        else:
            merged[key] = values[-1]
    return merged


def _downscale_frame(stats: Dict[str, Any]) -> None:
    """Strip non-serializable frame for multiprocessing return (keep small preview)."""
    try:
        frame = stats.get("_frame")
        if frame is not None and frame.ndim >= 2 and (frame.shape[0] > 84 or frame.shape[1] > 84):
            try:
                import cv2

                stats["_frame"] = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
            except Exception:
                stats["_frame"] = None
    except Exception:
        pass


def _eval_worker(args: Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], bytes, int, List[int]]):
    """Worker entry: unpack and evaluate using the cached per-worker Evaluator."""
    env_cfg, net_cfg_dict, reward_cfg_dict, genome_bytes, ind_id, seeds = args
    evaluator = _get_worker_evaluator(env_cfg, net_cfg_dict, reward_cfg_dict)
    fitness, stats, steps = _run_episodes(evaluator, genome_bytes, ind_id, seeds)
    return ind_id, fitness, stats, steps


def _eval_worker_live(
    args: Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], bytes, int, List[int], Any]
):
    """Live worker that publishes frames to a shared dict."""
    env_cfg, net_cfg_dict, reward_cfg_dict, genome_bytes, ind_id, seeds, shared = args
    evaluator = _get_worker_evaluator(env_cfg, net_cfg_dict, reward_cfg_dict)

    def publish_fn(frame, info):
        try:
            try:
                import cv2

                if frame.shape[0] > 84 or frame.shape[1] > 84:
                    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
            except Exception:
                pass
            shared[ind_id] = (frame.copy() if hasattr(frame, "copy") else frame, dict(info))
        except Exception:
            pass

    fitness, stats, steps = _run_episodes(evaluator, genome_bytes, ind_id, seeds, publish=publish_fn)
    try:
        if stats.get("_frame") is not None:
            shared[ind_id] = (stats["_frame"], dict(stats))
    except Exception:
        pass
    return ind_id, fitness, stats, steps


class ParallelEvaluator:
    """Evaluates population concurrently with a persistent worker pool."""

    def __init__(
        self,
        env_config: Dict[str, Any],
        net_config: NetworkConfig,
        reward_config: RewardConfig,
        workers: int = 4,
        seed_policy: str = "fixed",
        seeds_per_individual: int = 1,
    ):
        self.env_config = env_config
        self.net_config = net_config
        self.reward_config = reward_config
        self.workers = workers
        self.seed_policy = seed_policy  # "fixed" | "per_generation"
        self.seeds_per_individual = max(1, int(seeds_per_individual))
        self._pool: ProcessPoolExecutor | None = None

    # -- pool lifecycle -----------------------------------------------------
    def _ensure_pool(self) -> ProcessPoolExecutor | None:
        if self.workers <= 1:
            return None
        if self._pool is None:
            ctx = get_context("spawn")
            self._pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=ctx)
        return self._pool

    def shutdown(self) -> None:
        if self._pool is not None:
            try:
                self._pool.shutdown(wait=False, cancel_futures=True)
            except TypeError:  # py<3.9
                self._pool.shutdown(wait=False)
            except Exception:
                pass
            self._pool = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    # -- seeds ---------------------------------------------------------------
    def _seeds_for_generation(self, base_seed: int, generation: int) -> List[int]:
        if self.seed_policy == "per_generation":
            base = (base_seed * 100000 + generation * 1000) & 0x7FFFFFFF
        else:  # fixed: identical fitness landscape every generation
            base = base_seed & 0x7FFFFFFF
        # Same seed set for EVERY individual -> fair ranking; k>1 averages noise
        return [(base + i * 7919) & 0x7FFFFFFF for i in range(self.seeds_per_individual)]

    # -- main entry ----------------------------------------------------------
    def evaluate_population(
        self,
        population: List[Individual],
        base_seed: int = 42,
        generation: int = 0,
        shared_frames: Any | None = None,
    ) -> Tuple[List[float], List[Dict[str, Any]], float]:
        """Evaluate all individuals in parallel.

        Returns (fitnesses, stats_list, eps_per_sec)
        shared_frames: optional Manager.dict() for live visualization (workers publish frames)
        """
        start = time.time()
        net_cfg_dict = {
            "input_shape": self.net_config.input_shape,
            "cnn_channels": self.net_config.cnn_channels,
            "kernel_sizes": self.net_config.kernel_sizes,
            "strides": self.net_config.strides,
            "dense_hidden": self.net_config.dense_hidden,
            "num_actions": self.net_config.num_actions,
            "activation": self.net_config.activation,
        }
        reward_cfg_dict = self.reward_config.__dict__.copy()

        seeds = self._seeds_for_generation(base_seed, generation)
        use_live = shared_frames is not None

        tasks = []
        results: Dict[int, Tuple[float, Dict[str, Any], int]] = {}
        n_evaluated = 0
        for ind in population:
            # Elite carry-forward: fitness already valid under the fixed seed policy
            if ind.status == "evaluated" and np.isfinite(ind.fitness):
                results[ind.id] = (ind.fitness, dict(ind.stats), int(ind.stats.get("steps", 0)))
                continue
            n_evaluated += 1
            if use_live:
                args = (self.env_config, net_cfg_dict, reward_cfg_dict, ind.genome.vector.tobytes(), ind.id, seeds, shared_frames)
            else:
                args = (self.env_config, net_cfg_dict, reward_cfg_dict, ind.genome.vector.tobytes(), ind.id, seeds)
            tasks.append(args)

        if tasks:
            if self.workers <= 1:
                worker_fn = _eval_worker_live if use_live else _eval_worker
                for args in tasks:
                    ind_id, fit, stats, steps = worker_fn(args)
                    results[ind_id] = (fit, stats, steps)
            else:
                self._run_pool(tasks, use_live, results)

        # Order by population order
        fitnesses = []
        stats_list = []
        total_steps = 0
        for ind in population:
            fit, stats, steps = results[ind.id]
            fitnesses.append(fit)
            stats_list.append(stats)
            total_steps += steps
        eps = n_evaluated / max(time.time() - start, 1e-6)
        return fitnesses, stats_list, eps

    def _run_pool(self, tasks: List[tuple], use_live: bool, results: Dict[int, Tuple[float, Dict[str, Any], int]]) -> None:
        worker_fn = _eval_worker_live if use_live else _eval_worker
        for attempt in range(2):
            pool = self._ensure_pool()
            assert pool is not None
            futures: dict[Future, int] = {}
            try:
                for args in tasks:
                    fut = pool.submit(worker_fn, args)
                    futures[fut] = args[4]  # ind_id
                for fut in as_completed(futures):
                    ind_id, fit, stats, steps = fut.result(timeout=1800)
                    results[ind_id] = (fit, stats, steps)
                return
            except (BrokenExecutor, OSError):
                # A worker crashed (emulator/RAM); rebuild pool and retry once.
                self.shutdown()
                if attempt == 1:
                    raise
                time.sleep(1.0)

    def benchmark(
        self,
        population: List[Individual],
        worker_counts: List[int] = [4, 6, 8, 10, 12],
        base_seed: int = 42,
    ) -> Dict[int, Dict[str, Any]]:
        """Benchmark different worker counts and record RAM/CPU/episodes_sec."""
        import psutil

        results: Dict[int, Dict[str, Any]] = {}
        original_workers = self.workers
        for wc in worker_counts:
            self.shutdown()  # pool must be rebuilt for a new worker count
            self.workers = wc
            vm_before = psutil.virtual_memory().percent
            cpu_before = psutil.cpu_percent(interval=None)
            start = time.time()
            fitnesses, stats, eps = self.evaluate_population(population, base_seed=base_seed, generation=0)
            dur = time.time() - start
            vm_after = psutil.virtual_memory().percent
            cpu_after = psutil.cpu_percent(interval=None)
            results[wc] = {
                "workers": wc,
                "duration": dur,
                "episodes_per_sec": eps,
                "ram_before": vm_before,
                "ram_after": vm_after,
                "ram_used_percent": max(vm_before, vm_after),
                "cpu_percent": cpu_after,
                "mean_fitness": float(np.mean(fitnesses)) if fitnesses else 0,
            }
            time.sleep(0.5)  # brief cool down
        self.workers = original_workers
        return results
