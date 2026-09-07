"""Genetic Algorithm main loop."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

from agent.genome import Individual, Genome, random_genome
from agent.network import NetworkConfig, count_parameters, parameter_scale_vector
from utils.logging import get_logger
from .selection import tournament_selection, elite_selection
from .crossover import uniform_crossover, blend_crossover
from .mutation import mutate, adaptive_mutation_rate, adaptive_mutation_std
from .diversity import population_diversity_normalized

logger = get_logger(__name__)


@dataclass
class GAConfig:
    population_size: int = 8
    elite_count: int = 2
    tournament_size: int = 3
    mutation_rate: float = 0.05
    mutation_std: float = 0.08  # RELATIVE multiplier of per-layer init scale
    crossover_rate: float = 0.5
    crossover_type: str = "blend"
    adaptive_mutation: bool = True
    diversity_threshold: float = 0.05
    adaptive_boost: float = 1.5
    immigrant_frac: float = 0.1
    stall_generations: int = 5
    stall_immigrant_frac: float = 0.25  # immigrant burst while stalled
    mutation_std_min: float = 0.02
    mutation_std_max: float = 0.16
    reevaluate_elites: bool = False  # False: elites carry fitness (fixed eval seed)
    stall_rel_epsilon: float = 0.001  # improvement threshold as fraction of best
    init_genome: str | None = None  # optional .npy/.pkl genome to seed the population from
    seed: int = 42

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GAConfig":
        pop = d.get("population_size", 8)
        imm = d.get("immigrant_frac", d.get("immigrants", 0.1))
        if isinstance(imm, int) and imm >= 1:
            imm_frac = float(imm) / max(int(pop), 1)
        else:
            imm_frac = float(imm)
        return cls(
            population_size=pop,
            elite_count=d.get("elite_count", 2),
            tournament_size=d.get("tournament_size", 3),
            mutation_rate=d.get("mutation_rate", 0.05),
            mutation_std=d.get("mutation_std", 0.08),
            crossover_rate=d.get("crossover_rate", 0.5),
            crossover_type=d.get("crossover_type", "blend"),
            adaptive_mutation=d.get("adaptive_mutation", True),
            diversity_threshold=d.get("diversity_threshold", 0.05),
            adaptive_boost=d.get("adaptive_boost", 1.5),
            immigrant_frac=imm_frac,
            stall_generations=d.get("stall_generations", 5),
            stall_immigrant_frac=d.get("stall_immigrant_frac", 0.25),
            mutation_std_min=d.get("mutation_std_min", 0.02),
            mutation_std_max=d.get("mutation_std_max", 0.16),
            reevaluate_elites=d.get("reevaluate_elites", False),
            stall_rel_epsilon=d.get("stall_rel_epsilon", 0.001),
            init_genome=d.get("init_genome"),
            seed=d.get("seed", 42),
        )



class GeneticAlgorithm:
    """Elitism + (mu+lambda) mutation, blend/uniform crossover via tournament
    selection over the full population, random immigrants (burst on stall)."""

    def __init__(self, ga_config: GAConfig, net_config: NetworkConfig):
        self.cfg = ga_config
        self.net_cfg = net_config
        self.rng = np.random.default_rng(ga_config.seed)
        self.generation: int = 0
        self.population: List[Individual] = []
        self.best_individual: Individual | None = None
        self.history: List[Dict[str, Any]] = []
        self.stall_count: int = 0
        self.prev_best_fitness: float = -np.inf
        self.current_mutation_std: float = ga_config.mutation_std
        # Per-gene init scales: mutation/diversity operate relative to these
        expected = count_parameters(net_config)
        self.gene_scales = parameter_scale_vector(net_config)
        assert self.gene_scales.size == expected

    def initialize(self) -> List[Individual]:
        self.population = []
        expected = count_parameters(self.net_cfg)
        for i in range(self.cfg.population_size):
            g = random_genome(self.net_cfg, rng=self.rng, generation=0)
            assert g.vector.size == expected
            ind = Individual(id=i, genome=g, fitness=-np.inf, status="pending")
            self.population.append(ind)
        if self.cfg.init_genome:
            self._seed_from_genome(self.cfg.init_genome, expected)
        self.generation = 0
        self.stall_count = 0
        self.prev_best_fitness = -np.inf
        self.current_mutation_std = self.cfg.mutation_std
        return self.population

    def _seed_from_genome(self, path: str, expected: int) -> None:
        """Replace the random population with mutated copies of a saved genome.

        Individual 0 is the genome verbatim; the rest are mutated copies
        (per-layer mutation, same operators as training) so the population
        explores around everything that genome already knew. The last
        `immigrant_frac` of the population stay random for diversity.
        Fail-soft: a missing/corrupt file leaves the random population intact.
        """
        try:
            p = Path(path)
            if p.suffix == ".npy":
                vec = np.load(p).astype(np.float32)
            else:
                import pickle

                with open(p, "rb") as f:
                    vec = np.asarray(pickle.load(f)["best_genome"], dtype=np.float32)
            if vec.size != expected:
                logger.warning(
                    f"init_genome size {vec.size} != network size {expected} - ignoring seed"
                )
                return
            base = Genome(vector=vec.copy(), generation=0)
            n_random = int(round(self.cfg.population_size * float(self.cfg.immigrant_frac)))
            n_seeded = self.cfg.population_size - n_random
            for i in range(self.cfg.population_size):
                if i < n_seeded:
                    g = base.copy() if i == 0 else mutate(
                        base, self.cfg.mutation_rate, self.cfg.mutation_std, rng=self.rng, scales=self.gene_scales
                    )
                else:
                    g = random_genome(self.net_cfg, rng=self.rng, generation=0)
                g.generation = 0
                ind = self.population[i]
                ind.genome = g
                ind.fitness = -np.inf
                ind.status = "pending"
            logger.info(
                f"Seeded population from {path}: {n_seeded} copies/mutants + {n_random} random"
            )
        except Exception as e:
            logger.warning(f"init_genome '{path}' could not be loaded ({e}) - using random population")

    def set_fitness(self, fitnesses: List[float], stats_list: List[Dict[str, Any]] | None = None) -> None:
        """Assign fitnesses from evaluator. Order must match population."""
        assert len(fitnesses) == len(self.population)
        for i, (ind, fit) in enumerate(zip(self.population, fitnesses)):
            new_fit = float(fit)
            # Skipped (carried) elites come back with their exact fitness and
            # carry stamps; anything else is a fresh evaluation.
            was_carried = (
                ind.status == "evaluated"
                and np.isfinite(ind.fitness)
                and ind.fitness == new_fit
                and isinstance(ind.stats, dict)
                and bool(ind.stats.get("elite_carry"))
            )
            ind.fitness = new_fit
            if stats_list:
                ind.stats = stats_list[i]
            if not isinstance(ind.stats, dict):
                ind.stats = {}
            if was_carried:
                ind.stats["elite_carry"] = 1.0
                ind.stats.setdefault("evaluated_at_gen", self.generation)
            else:
                ind.stats.pop("elite_carry", None)
                if new_fit != -np.inf:
                    ind.stats.setdefault("evaluated_at_gen", self.generation)
            if new_fit != -np.inf:
                ind.status = "evaluated"
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        if self.best_individual is None or sorted_pop[0].fitness > self.best_individual.fitness:
            self.best_individual = sorted_pop[0]
        best = float(sorted_pop[0].fitness)
        if self._improved(best):
            self.stall_count = 0
            self.current_mutation_std = self.cfg.mutation_std
            self.prev_best_fitness = best
        else:
            self.stall_count += 1
            if self.cfg.adaptive_mutation:
                self.current_mutation_std = adaptive_mutation_std(
                    self.cfg.mutation_std,
                    self.stall_count,
                    self.cfg.stall_generations,
                    self.cfg.mutation_std_min,
                    self.cfg.mutation_std_max,
                    self.cfg.adaptive_boost,
                )

    def _improved(self, best: float) -> bool:
        if self.prev_best_fitness == -np.inf:
            return np.isfinite(best)
        eps = max(1e-6, self.cfg.stall_rel_epsilon * abs(self.prev_best_fitness))
        return best > self.prev_best_fitness + eps

    def invalidate_fitnesses(self) -> None:
        """Mark all fitnesses stale (e.g. reward/eval config changed mid-run)."""
        for ind in self.population:
            ind.fitness = -np.inf
            ind.status = "pending"

    def ga_state(self) -> Dict[str, Any]:
        return {
            "stall_count": self.stall_count,
            "prev_best_fitness": self.prev_best_fitness,
            "current_mutation_std": self.current_mutation_std,
            "best_genome": self.best_individual.genome.vector.copy() if self.best_individual else None,
            "best_fitness": float(self.best_individual.fitness) if self.best_individual else -np.inf,
        }

    def load_ga_state(self, state: Dict[str, Any]) -> None:
        self.stall_count = int(state.get("stall_count", 0))
        self.prev_best_fitness = float(state.get("prev_best_fitness", -np.inf))
        self.current_mutation_std = float(state.get("current_mutation_std", self.cfg.mutation_std))
        bg = state.get("best_genome")
        if bg is not None:
            self.best_individual = Individual(
                id=-1,
                genome=Genome(vector=np.asarray(bg, dtype=np.float32)),
                fitness=float(state.get("best_fitness", -np.inf)),
                status="evaluated",
            )

    def next_generation(self) -> List[Individual]:
        """Evolve to next generation, return new population."""
        assert self.population, "call initialize first"
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        elites = elite_selection(sorted_pop, self.cfg.elite_count)
        if not elites:
            elites = [sorted_pop[0]]
        new_pop: List[Individual] = []
        for e in elites:
            g = e.genome.copy()
            g.generation = self.generation + 1
            carry = (not self.cfg.reevaluate_elites) and np.isfinite(e.fitness)
            if carry:
                # Stamp carried stats so the dashboard can label this tile as a
                # frozen elite (not an idling agent) and show how old its
                # evaluation is. `evaluated_at_gen` = the generation whose
                # evaluation produced this fitness.
                carried = dict(e.stats)
                carried["elite_carry"] = 1.0
                carried.setdefault("evaluated_at_gen", self.generation)
            else:
                carried = {}
            new_pop.append(
                Individual(
                    id=len(new_pop),
                    genome=g,
                    fitness=float(e.fitness) if carry else -np.inf,
                    stats=carried,
                    status="evaluated" if carry else "pending",
                )
            )

        div = population_diversity_normalized(self.population, self.gene_scales)
        mut_rate = self.cfg.mutation_rate
        mut_std = self.current_mutation_std
        if self.cfg.adaptive_mutation:
            mut_rate = adaptive_mutation_rate(mut_rate, div, self.cfg.diversity_threshold, self.cfg.adaptive_boost)
            mut_std = adaptive_mutation_std(
                self.cfg.mutation_std,
                self.stall_count,
                self.cfg.stall_generations,
                self.cfg.mutation_std_min,
                self.cfg.mutation_std_max,
                self.cfg.adaptive_boost,
            )
            self.current_mutation_std = mut_std

        stalled = self.stall_count >= self.cfg.stall_generations
        imm_frac = self.cfg.stall_immigrant_frac if stalled else self.cfg.immigrant_frac
        n_immigrants = int(round(self.cfg.population_size * float(imm_frac)))
        n_immigrants = min(max(0, n_immigrants), self.cfg.population_size - len(new_pop))

        while len(new_pop) < self.cfg.population_size - n_immigrants:
            # Tournament over the FULL population: the non-elite genomes carry
            # genetic material too and cost nothing extra to reuse.
            parent_a = tournament_selection(self.population, self.cfg.tournament_size, rng=self.rng)
            child_genome = parent_a.genome.copy()
            if self.cfg.crossover_rate > 0 and self.rng.random() < self.cfg.crossover_rate:
                parent_b = tournament_selection(self.population, self.cfg.tournament_size, rng=self.rng)
                if self.cfg.crossover_type == "uniform":
                    child_genome = uniform_crossover(parent_a.genome, parent_b.genome, 1.0, rng=self.rng)
                else:
                    child_genome = blend_crossover(parent_a.genome, parent_b.genome, rng=self.rng)
            child_genome = mutate(child_genome, mut_rate, mut_std, rng=self.rng, scales=self.gene_scales)
            child_genome.generation = self.generation + 1
            new_pop.append(Individual(id=len(new_pop), genome=child_genome, fitness=-np.inf, status="pending"))

        while len(new_pop) < self.cfg.population_size:
            g = random_genome(self.net_cfg, rng=self.rng, generation=self.generation + 1)
            new_pop.append(Individual(id=len(new_pop), genome=g, fitness=-np.inf, status="pending"))

        self.generation += 1
        self.population = new_pop
        return self.population

    def get_stats(self) -> Dict[str, Any]:
        if not self.population:
            return {}
        fitnesses = [ind.fitness for ind in self.population if ind.fitness != -np.inf]
        if not fitnesses:
            fitnesses = [0.0]
        div = population_diversity_normalized(self.population, self.gene_scales)
        best = max(fitnesses)
        worst = min(fitnesses)
        mean = float(np.mean(fitnesses))
        median = float(np.median(fitnesses))
        mut_rate = (
            adaptive_mutation_rate(self.cfg.mutation_rate, div, self.cfg.diversity_threshold, self.cfg.adaptive_boost)
            if self.cfg.adaptive_mutation
            else self.cfg.mutation_rate
        )
        return {
            "generation": self.generation,
            "best_fitness": best,
            "mean_fitness": mean,
            "median_fitness": median,
            "worst_fitness": worst,
            "diversity": div,
            "mutation_rate": mut_rate,
            "mutation_std": self.current_mutation_std,
            "stall_count": self.stall_count,
            "best_id": int(np.argmax(fitnesses)) if fitnesses else 0,
        }
