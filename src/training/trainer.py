"""Trainer: orchestrates GA + parallel evaluation + checkpoint + resources + visualization.

Supports modes: debug, training, live_training
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any, Callable, List

import numpy as np
import yaml

from utils.config import load_config
from utils.logging import get_logger, setup_logging
from utils.resources import ResourceMonitor, estimate_memory_mb
from utils.seeds import set_global_seed
from environment.reward import RewardConfig
from environment.actions import get_action_space_size
from environment.emulator_factory import custom_retro_dir, GAME_NAME
from environment.state_capture import capture_progress_state
from agent.network import NetworkConfig, count_parameters
from evolution.genetic_algorithm import GeneticAlgorithm, GAConfig
from evaluation.parallel import ParallelEvaluator
from evaluation.metrics import MetricsLogger
from .checkpoint import save_checkpoint, load_checkpoint
from .curriculum import Curriculum

logger = get_logger(__name__)


class Trainer:
    def __init__(self, config: Dict[str, Any], run_dir: Path | None = None):
        self.cfg = config
        self.run_dir = Path(run_dir) if run_dir else Path("runs") / time.strftime("%Y%m%d_%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # setup logging
        setup_logging(log_file=self.run_dir / "train.log")
        # save config
        with open(self.run_dir / "config.yaml", "w") as f:
            yaml.safe_dump(config, f)
        # components
        env_cfg = config["env"]
        ga_cfg_dict = config["evolution"]
        agent_cfg = config["agent"]
        reward_cfg_dict = dict(config["reward"])
        self.num_actions = get_action_space_size(config.get("actions", {}).get("space", "playable"))
        env_cfg["num_actions"] = self.num_actions

        self.curriculum = Curriculum.from_config(config.get("curriculum", {}))
        self.curriculum.apply_to_env(env_cfg, reward_cfg_dict)

        self.net_cfg = NetworkConfig(
            cnn_channels=agent_cfg["cnn_channels"],
            kernel_sizes=agent_cfg["kernel_sizes"],
            strides=agent_cfg["strides"],
            dense_hidden=agent_cfg["dense_hidden"],
            num_actions=self.num_actions,
        )
        param_count = count_parameters(self.net_cfg)
        logger.info(f"Network params: {param_count} (limit {agent_cfg.get('param_limit',100000)})")
        if param_count > agent_cfg.get("param_limit", 100000):
            logger.warning(f"Param count {param_count} exceeds limit")

        self.ga_cfg = GAConfig.from_dict(ga_cfg_dict)
        self.reward_cfg = RewardConfig.from_dict(reward_cfg_dict)
        # Elite fitness can only be carried forward when the evaluation seed is
        # identical across generations ("fixed"); otherwise re-evaluate.
        eval_cfg = config["evaluation"]
        seed_policy = str(eval_cfg.get("seed_policy", "fixed"))
        if seed_policy != "fixed":
            self.ga_cfg.reevaluate_elites = True
        self.ga = GeneticAlgorithm(self.ga_cfg, self.net_cfg)

        # resources
        res_cfg = config.get("resources", {})
        self.monitor = ResourceMonitor(interval=res_cfg.get("monitor_interval", 1.0))
        self.max_ram = res_cfg.get("max_ram_percent", 85.0)
        est = estimate_memory_mb(
            frame_size=env_cfg.get("frame_size", 84),
            frame_stack=env_cfg.get("frame_stack", 4),
            population=self.ga_cfg.population_size,
            workers=eval_cfg.get("workers", 4),
        )
        logger.info(f"Estimated memory: {est}")

        # evaluator
        self.evaluator = ParallelEvaluator(
            env_config=env_cfg,
            net_config=self.net_cfg,
            reward_config=self.reward_cfg,
            workers=eval_cfg.get("workers", 4),
            seed_policy=seed_policy,
            seeds_per_individual=eval_cfg.get("seeds_per_individual", 1),
        )
        self.metrics = MetricsLogger(self.run_dir)

        self.generation = 0
        self.best_fitness = -np.inf
        self.best_genome = None
        self._short_episode_streak = 0  # death-state guard (see _check_death_state)

        set_global_seed(self.ga_cfg.seed)

    @staticmethod
    def _is_death_state(ga_stats: Dict[str, Any]) -> bool:
        """True when the best agent's episodes are near-instant deaths.

        A stage start state that kills everyone within a few steps makes the
        stage unreachable and silently wastes every following generation.
        """
        try:
            return float(ga_stats.get("episode_length", 10 ** 9)) < 5.0
        except (TypeError, ValueError):
            return False

    def train(
        self,
        generations: int | None = None,
        resume_path: Path | None = None,
        render_callback: Callable | None = None,
    ) -> None:
        gens = generations or self.cfg["training"]["generations"]
        checkpoint_interval = self.cfg["training"].get("checkpoint_interval", 5)
        training_mode = self.cfg["training"].get("mode", "training")

        if resume_path:
            self._resume(resume_path)
        else:
            self.ga.initialize()

        logger.info(f"Starting training for {gens} generations mode={training_mode} workers={self.evaluator.workers}")

        for gen in range(self.generation, self.generation + gens):
            gen_start = time.time()
            # evaluate - check for live shared dict (for visualization)
            shared = getattr(self.evaluator, "shared_frames", None)
            if shared is not None:
                fitnesses, stats_list, eps = self.evaluator.evaluate_population(
                    self.ga.population, base_seed=self.ga_cfg.seed, generation=gen, shared_frames=shared
                )
            else:
                fitnesses, stats_list, eps = self.evaluator.evaluate_population(
                    self.ga.population, base_seed=self.ga_cfg.seed, generation=gen
                )
            self.ga.set_fitness(fitnesses, stats_list)
            # stats
            ga_stats = self.ga.get_stats()
            # resource
            res = self.monitor.poll()
            res.episodes_per_sec = eps
            res.generation_duration = time.time() - gen_start
            self.monitor.record_episode()
            # enrich ga_stats
            ga_stats["episodes_per_sec"] = eps
            ga_stats["ram_percent"] = res.ram_percent
            ga_stats["cpu_percent"] = res.cpu_percent
            ga_stats["generation_duration"] = res.generation_duration
            # best progress/kills
            best_idx = int(np.argmax(fitnesses))
            best_stats = stats_list[best_idx]
            ga_stats["best_progress"] = best_stats.get("progress", 0)
            self._last_best_progress = ga_stats["best_progress"]
            ga_stats["best_kills"] = best_stats.get("kills", 0)
            ga_stats["episode_length"] = best_stats.get("steps", 0)
            ga_stats["best_deaths"] = best_stats.get("deaths", 0)
            ga_stats["best_idle"] = best_stats.get("idle_steps", 0)
            ga_stats["screen_type"] = best_stats.get("screen_type", 0)

            # death-state guard: a lethal stage start state must never eat a long run
            if self.curriculum.enabled and self._is_death_state(ga_stats):
                self._short_episode_streak += 1
                if self._short_episode_streak >= 10:
                    self._retire_lethal_stage_state(gen)
                    self._short_episode_streak = 0
            else:
                self._short_episode_streak = 0

            # check RAM limit
            if not self.monitor.check_limits(self.max_ram):
                logger.warning(f"RAM {res.ram_percent:.1f}% exceeds limit {self.max_ram}%, consider reducing workers")

            # log
            self.metrics.log(gen, ga_stats, resource=res)
            logger.info(
                f"Gen {gen:03d} | best {ga_stats['best_fitness']:.1f} avg {ga_stats['mean_fitness']:.1f} "
                f"med {ga_stats['median_fitness']:.1f} worst {ga_stats['worst_fitness']:.1f} "
                f"div {ga_stats['diversity']:.3f} mut {ga_stats['mutation_rate']:.3f} "
                f"std {ga_stats.get('mutation_std', 0):.3f} prog {ga_stats['best_progress']:.1f} "
                f"eps/s {eps:.1f} RAM {res.ram_percent:.0f}%"
            )

            # render callback for live mode
            if render_callback:
                try:
                    render_callback(gen, ga_stats, self.ga.population, stats_list)
                except Exception as e:
                    logger.warning(f"render callback failed: {e}")

            # update best
            if ga_stats["best_fitness"] > self.best_fitness:
                self.best_fitness = ga_stats["best_fitness"]
                # find best genome
                best_ind = max(self.ga.population, key=lambda x: x.fitness)
                self.best_genome = best_ind.genome.vector.copy()
                # save best
                np.save(self.run_dir / "best_genome.npy", self.best_genome)

            # curriculum
            if self.curriculum.enabled:
                advanced = self.curriculum.maybe_advance(
                    ga_stats["best_fitness"],
                    best_progress=float(ga_stats.get("best_progress") or 0),
                    best_kills=float(ga_stats.get("best_kills") or 0),
                )
                if advanced:
                    logger.info(f"Curriculum advanced to {self.curriculum.current()}")
                    self._ensure_stage_state()
                    fitnesses, stats_list = self._apply_curriculum(gen)
                    self.ga.set_fitness(fitnesses, stats_list)
                    # refresh stats under the new stage for logging/checkpointing
                    ga_stats = self.ga.get_stats()
                    ga_stats["episodes_per_sec"] = eps
                    ga_stats["best_progress"] = max(
                        ga_stats.get("best_progress", 0), float(stats_list[int(np.argmax(fitnesses))].get("progress", 0))
                    )

            # checkpoint
            if (gen + 1) % checkpoint_interval == 0:
                ckpt_path = Path("checkpoints") / f"gen_{gen+1:04d}.pkl"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                self._save_checkpoint(ckpt_path, gen, fitnesses)
                # also copy to run_dir
                self._save_checkpoint(self.run_dir / "checkpoints" / f"gen_{gen+1:04d}.pkl", gen, fitnesses)
                logger.info(f"Checkpoint saved gen {gen+1} -> {ckpt_path}")

            # evolve
            if gen < self.generation + gens - 1:
                self.ga.next_generation()

            # visualization headless snapshot?
            if training_mode == "debug":
                time.sleep(0.1)

        self.evaluator.shutdown()
        self.metrics.close()
        # save final
        with open(self.run_dir / "best_fitness.txt", "w") as f:
            f.write(str(self.best_fitness))
        logger.info(f"Training done best {self.best_fitness:.1f} -> {self.run_dir}")

    def _save_checkpoint(self, path: Path, gen: int, fitnesses: List[float]) -> None:
        save_checkpoint(
            path,
            generation=gen + 1,
            population=self.ga.population,
            fitnesses=fitnesses,
            best_genome=self.best_genome if self.best_genome is not None else self.ga.population[0].genome.vector,
            ga_config=self.ga_cfg.__dict__,
            env_config=self.cfg["env"],
            net_config=self.net_cfg.__dict__,
            metrics=self.metrics.history,
            rng_state=self.ga.rng.bit_generator.state,
            ga_state=self.ga.ga_state(),
            curriculum_stage=self.curriculum.current_idx if self.curriculum.enabled else None,
        )

    def _resume(self, path: Path) -> None:
        logger.info(f"Resuming from {path}")
        data = load_checkpoint(path)
        self.generation = data["generation"]
        # reconstruct population
        self.ga.generation = self.generation
        # data["population"] is list of vectors
        self.ga.population = []
        for i, vec in enumerate(data["population"]):
            from agent.genome import Genome, Individual

            g = Genome(vector=vec, generation=self.generation)
            ind = Individual(id=i, genome=g, fitness=data["fitnesses"][i] if i < len(data["fitnesses"]) else -np.inf)
            self.ga.population.append(ind)
        self.best_genome = data["best_genome"]
        self.best_fitness = float(max(data["fitnesses"])) if data["fitnesses"] else -np.inf
        # restore rng
        if data.get("rng_state"):
            try:
                self.ga.rng.bit_generator.state = data["rng_state"]
            except Exception:
                pass
        # restore GA bookkeeping (stall detection, adaptive std, best individual)
        if data.get("ga_state"):
            try:
                self.ga.load_ga_state(data["ga_state"])
            except Exception as e:
                logger.warning(f"Failed to restore ga_state: {e}")
        # restore curriculum stage + env config (scenario/max_steps/reward overrides)
        stage = data.get("curriculum_stage")
        env_config = data.get("env_config")
        if self.curriculum.enabled and stage is not None:
            self.curriculum.current_idx = int(stage)
            logger.info(f"Restored curriculum stage {self.curriculum.current()}")
        if env_config:
            self.cfg["env"] = dict(env_config)
            reward_dict = dict(self.cfg["reward"])
            self.curriculum.apply_to_env(self.cfg["env"], reward_dict)
            self.reward_cfg = RewardConfig.from_dict(reward_dict)
            self.evaluator.env_config = self.cfg["env"]
            self.evaluator.reward_config = self.reward_cfg
        # metrics history
        if data.get("metrics"):
            self.metrics.history = data["metrics"]
        logger.info(f"Resumed at gen {self.generation} best {self.best_fitness:.1f}")

    def _retire_lethal_stage_state(self, gen: int) -> None:
        """CRITICAL fallback: the active stage start state kills every agent
        within a few steps (e.g. a snapshot taken mid-fall over a pit). Delete
        it, switch to a stateless scenario, and disable the curriculum for the
        rest of the run so training can actually proceed."""
        sc_name = self.curriculum.current()
        logger.critical(
            f"DEATH-STATE GUARD: best agent died within 5 steps for 10 consecutive generations "
            f"on stage '{sc_name}' (gen {gen}) - the stage start state is lethal. "
            f"Retiring it and disabling the curriculum for the rest of this run."
        )
        try:
            from environment.scenarios import get_scenario

            sc = get_scenario(sc_name)
            if sc.state_file:
                for p in (custom_retro_dir() / GAME_NAME / sc.state_file, Path("roms") / "states" / sc.state_file):
                    if p.exists():
                        p.unlink()
                        logger.critical(f"Deleted lethal state file: {p}")
        except Exception as e:
            logger.warning(f"Could not delete stage state: {e}")
        # switch to a stateless scenario so worker envs rebuild WITHOUT the lethal state
        self.cfg["env"]["scenario"] = "full_level_1"
        self.curriculum.enabled = False
        self.cfg["curriculum"]["enabled"] = False
        self.evaluator.env_config = self.cfg["env"]
        self.ga.invalidate_fitnesses()

    def _ensure_stage_state(self) -> None:
        """Auto-capture the new stage's start state if it is missing.

        The current best genome just proved it can reach this stage's area
        (that's why the stage advanced), so replay it and snapshot the furthest
        survivable point. CRITICAL: replay from the PREVIOUS stage's start (the
        scenario still in env config) - the pilot's trajectory is conditioned
        on that exact start; a different one (e.g. power-on) diverges within
        frames and commits useless screen-0 snapshots. Missing/failed capture
        falls back to the previous stage's behavior with a warning.
        """
        from environment.scenarios import get_scenario

        try:
            sc = get_scenario(self.curriculum.current())
        except ValueError:
            return
        if sc.state_file is None:
            return
        game_path = custom_retro_dir() / GAME_NAME / sc.state_file
        if game_path.exists():
            return
        best = self.ga.best_individual
        if best is None or not np.isfinite(best.fitness):
            logger.warning(f"Stage '{sc.name}' has no start state and no evaluated best genome - keeping previous start")
            return
        pilot_scenario = str(self.cfg["env"].get("scenario") or "full_game")
        logger.info(
            f"Auto-capturing start state '{sc.state_file}' from best genome "
            f"(fitness {best.fitness:.1f}, progress {getattr(self, '_last_best_progress', 0):.0f}px, "
            f"replaying from scenario '{pilot_scenario}')"
        )
        try:
            result = capture_progress_state(
                rom_path=self.cfg["env"].get("rom_path"),
                genome_vector=best.genome.vector,
                net_config=self.net_cfg,
                reach_screen=7,
                margin_px=32,
                snapshot_every_px=32,
                max_frames=30000,
                scenario=pilot_scenario,
                action_repeat=self.cfg["env"].get("action_repeat", 4),
                flicker_pool=self.cfg["env"].get("flicker_pool", 2),
                num_actions=self.net_cfg.num_actions,
            )
        except Exception as e:
            logger.warning(f"Auto-capture failed ({e}) - stage keeps the previous start")
            return
        if result is None:
            logger.warning("Auto-capture found no survivable moment - stage keeps the previous start")
            return
        data, label = result
        game_path.parent.mkdir(parents=True, exist_ok=True)
        game_path.write_bytes(data)
        mirror = Path("roms") / "states" / sc.state_file
        mirror.parent.mkdir(parents=True, exist_ok=True)
        mirror.write_bytes(data)
        # Divergence guard: the committed snapshot should be near the pilot's
        # known progress; a big gap means the replay diverged from training.
        try:
            committed_screen = int(label.split("screen ")[1].split()[0])
        except (IndexError, ValueError):
            committed_screen = -1
        expected_screen = int(getattr(self, "_last_best_progress", 0) // 256)
        if committed_screen < expected_screen - 1:
            logger.warning(
                f"Auto-captured state only reached screen {committed_screen} but the pilot's "
                f"progress was screen {expected_screen} - replay divergence, the state may be too early"
            )
        logger.info(f"Auto-captured '{sc.state_file}' ({label}) - stage will start there")

    def _apply_curriculum(self, generation: int):
        """Switch stage and re-score the population under the new config.

        Returns (fitnesses, stats_list). Re-evaluating BEFORE the next
        next_generation() is essential: without it, elite selection would run
        on an all-invalid fitness population and the best genome would be lost.
        """
        reward_dict = dict(self.cfg["reward"])
        self.curriculum.apply_to_env(self.cfg["env"], reward_dict)
        self.reward_cfg = RewardConfig.from_dict(reward_dict)
        self.evaluator.env_config = self.cfg["env"]
        self.evaluator.reward_config = self.reward_cfg
        # Fitness earned under the previous stage is not comparable: invalidate
        # (so carry-forward doesn't skip anyone) and re-evaluate immediately.
        self.ga.invalidate_fitnesses()
        fitnesses, stats_list, _ = self.evaluator.evaluate_population(
            self.ga.population, base_seed=self.ga_cfg.seed, generation=generation
        )
        return fitnesses, stats_list
