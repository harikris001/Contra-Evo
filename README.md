# Contra-Evo

Contra-Evo is a CPU-first neuroevolution project that learns to play the NES game Contra. It evolves the weights of a small NumPy convolutional neural network with a genetic algorithm instead of using gradient descent.

The project has two useful modes:

- **Real Contra mode** uses `stable-retro` and a legally obtained Contra ROM.
- **Mock mode** uses a synthetic Contra-like environment, so the software can be tested and demonstrated without a ROM.

The same Gymnasium-style environment API is used in both modes. That makes the project useful as an experiment in genetic algorithms today, while leaving room for future PPO, DQN, or hybrid agents.

## What The Project Does

Each generation follows this loop:

1. Create or restore a population of neural-network genomes.
2. Run every individual in the environment and collect fitness statistics.
3. Rank individuals by progress, kills, survival, and other configurable signals.
4. Keep the best individuals as elites.
5. Create the next population with tournament selection, crossover, mutation, and random immigrants.
6. Save metrics and periodic checkpoints.

Evaluation is deterministic by default: every individual in a generation is evaluated with the same seed policy and emulator start state. This makes comparisons fair and allows elite fitness to be carried into the next generation without wasting an evaluation.

## Quick Start Without A ROM

This is the fastest way to verify the installation. Mock mode is synthetic; success here proves that the training pipeline works, not that an agent can play the original game.

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"

uv run python scripts/train.py \
  --config configs/smoke_mock.yaml
```

For a live mock dashboard:

```bash
uv run python scripts/train.py \
  --config configs/live_mock.yaml \
  --render
```

## Requirements

- macOS, Linux, or another platform supported by the Python dependencies
- Python 3.10 or newer
- `uv` is recommended, although a normal `venv` and `pip` also work
- A graphical desktop is needed for pygame visualization
- A legally obtained Contra ROM is required for real-emulator training

The main dependencies are declared in [pyproject.toml](pyproject.toml):

- NumPy
- Gymnasium
- stable-retro
- OpenCV
- pygame
- matplotlib
- psutil
- PyYAML
- tqdm

Development dependencies add pytest, pytest-cov, Ruff, and mypy. The optional `benchmark` extra adds PyTorch for optional CPU/MPS comparisons; the evolutionary agent itself does not require PyTorch.

## Installation

Using `uv`:

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
```

Using standard Python tooling:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The runtime creates a custom stable-retro integration under `.custom_retro/Contra-Nes/`. You do not need to import the ROM into a pre-existing stable-retro game definition. The older stable-retro import command may still be useful for other tooling, but Contra-Evo's emulator factory installs and uses its own custom integration.

## Using A Real Contra ROM

Contra is copyrighted and is not included in this repository. Obtain a compatible ROM legally and place it at a path such as:

```text
roms/Contra.nes
```

The default configs already point to `./roms/Contra.nes`. You can also use an absolute path, `$ENV_VAR`, or `~` in `env.rom_path`.

Before training, verify that the file is present and large enough to be a valid ROM:

```bash
test -f roms/Contra.nes && ls -lh roms/Contra.nes
```

The first real-emulator run will:

1. Validate the ROM and stable-retro installation.
2. Copy the ROM into `.custom_retro/Contra-Nes/` when needed.
3. Write the custom RAM map used for progress, lives, score, screen, death, and gameplay-state detection.
4. Load a scenario save state when one exists.
5. Otherwise, start the game and freeze a deterministic gameplay start state for later resets.

If a ROM path is configured but the ROM cannot be opened, the environment fails instead of silently switching to mock mode. Mock fallback is enabled by setting `rom_path` to `null`, an empty string, or the string `"null"`.

## Training

### Common commands

Short headless run:

```bash
uv run python scripts/train.py \
  --config configs/debug.yaml \
  --generations 5
```

Default real-ROM training:

```bash
uv run python scripts/train.py \
  --config configs/default.yaml
```

Resume from a checkpoint:

```bash
uv run python scripts/train.py \
  --config configs/default.yaml \
  --resume checkpoints/gen_0020.pkl
```

Live population dashboard:

```bash
uv run python scripts/train.py \
  --config configs/live.yaml \
  --render
```

The training CLI accepts:

```text
--config PATH
--resume PATH
--render
--generations INTEGER
--workers INTEGER
--render-mode {population,best,comparison}
```

`--render` enables live training and a pygame dashboard. Closing the window or pressing `ESC` stops live training. The `--render-mode` option is accepted for compatibility, but the current implementation always constructs the population dashboard.

### Configuration presets

| File | Intended use | Important defaults |
| --- | --- | --- |
| `configs/default.yaml` | Main real-ROM training | 16 agents, 8 workers, 1000 generations, headless |
| `configs/debug.yaml` | Short development run | 4 agents, 2 workers, 10 generations, 200-step episodes |
| `configs/live.yaml` | Real-ROM visual demonstration | 16 agents, 8 workers, pygame dashboard |
| `configs/live_mock.yaml` | Visual demonstration without a ROM | 8 agents, 4 workers, 20 generations |
| `configs/smoke_mock.yaml` | CI and quick pipeline check | 6 agents, 2 workers, 5 generations, headless |

YAML files are deep-merged with defaults by `utils.config.load_config`. Missing keys inherit default values. The training script exposes only the CLI overrides listed above; other settings should be changed in a YAML file.

## Evaluation

Evaluate a saved pickle checkpoint with a playback window:

```bash
uv run python scripts/evaluate.py \
  --checkpoint checkpoints/gen_0100.pkl \
  --config configs/default.yaml \
  --episodes 3
```

For statistics only:

```bash
uv run python scripts/evaluate.py \
  --checkpoint checkpoints/gen_0100.pkl \
  --config configs/default.yaml \
  --episodes 3 \
  --headless
```

The evaluator accepts both `.pkl` checkpoints and `.npy` genome vectors. Useful options are:

```text
--checkpoint PATH       Checkpoint or best-genome .npy file
--config PATH           Config used to reconstruct the network/environment
--episodes INTEGER
--headless              Disable the playback window
--fps INTEGER            Playback rate, default 30
--scale INTEGER         Window scale, default 3
--from-beginning        Evaluate from the level-1 gameplay spawn
```

By default, evaluation restores the environment configuration saved in the checkpoint when available. `--from-beginning` overrides the scenario to start at the true level-1 gameplay spawn and skips title/intro frames when `level1_start.state` is available.

## Benchmarking Worker Counts

Real retro emulators are relatively heavy. Benchmark worker counts on the machine that will run training:

```bash
uv run python scripts/benchmark.py \
  --config configs/default.yaml \
  --workers 4,6,8,10,12 \
  --output runs/benchmark.json
```

The benchmark records duration, episodes per second, RAM percentage, CPU percentage, and mean fitness. It recommends the fastest worker count below the configured 85% RAM limit. Mock environments are much lighter and can behave differently, so benchmark with the same ROM and configuration used for training.

## Environment API

`ContraEnv` follows the standard Gymnasium shape:

```python
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
frame = env.render()
env.close()
```

The environment wraps stable-retro when a ROM is available and `MockContraEnv` when no ROM is configured. The mock environment produces synthetic frames, progress, and kills. It is intentionally useful for tests and pipeline development, but its scores do not measure real Contra performance.

### Actions

The configured default is the 16-action `playable` space. `reduced` and `legacy` select the first eight actions for compatibility.

| Index | Action | Index | Action |
| ---: | --- | ---: | --- |
| 0 | NOOP | 8 | RIGHT+JUMP |
| 1 | LEFT | 9 | RIGHT+JUMP+SHOOT |
| 2 | RIGHT | 10 | DOWN |
| 3 | SHOOT | 11 | UP |
| 4 | JUMP | 12 | UP+JUMP |
| 5 | LEFT+SHOOT | 13 | UP+JUMP+SHOOT |
| 6 | RIGHT+SHOOT | 14 | UP+RIGHT+JUMP |
| 7 | JUMP+SHOOT | 15 | UP+RIGHT+JUMP+SHOOT |

`src/environment/actions.py` converts these discrete actions to the multi-button vectors expected by the NES emulator.

### Observations

Pixel mode is the normal training mode:

- RGB frames are converted to grayscale.
- Frames are resized to 84 x 84 by default.
- Four frames are stacked in channel-first form: `(4, 84, 84)`.
- Values are `uint8` from 0 to 255.
- `flicker_pool: 2` max-pools recent frames to reduce NES sprite flicker.

Feature mode returns a 10-element `float32` vector containing heuristics such as position, lives, score, screen, stage, game routine, and kills. Values are normalized and clipped to `[-5, 5]`; this mode is mainly useful for debugging and experiments.

## Agent And Genome

The default policy is a compact NumPy CNN:

```text
(4, 84, 84)
  -> Conv 8 channels, kernel 8, stride 4 -> ReLU
  -> Conv 16 channels, kernel 4, stride 2 -> ReLU
  -> Dense 64 -> ReLU
  -> Dense 32 -> ReLU
  -> output logits for 16 actions
```

The network is CPU-oriented and has fewer than 100,000 parameters with the default settings. `CompactCNN` accepts `uint8` observations and normalizes them internally. It produces action logits; `Policy` selects the action used by the environment.

`Genome` stores all network parameters as one flat `float32` NumPy vector. This representation makes crossover, mutation, checkpointing, and comparison straightforward. The genome also stores generation and parent metadata. `Individual` adds an ID, fitness, evaluation statistics, and lifecycle status.

## Genetic Algorithm

The default evolutionary settings are:

```text
population_size: 16
elite_count: 2
tournament_size: 3
crossover_rate: 0.5
crossover_type: blend
mutation_rate: 0.04
mutation_std: 0.08
adaptive_mutation: true
immigrant_frac: 0.1
stall_generations: 5
```

Mutation standard deviation is a relative multiplier of each layer's Xavier-like initialization scale, rather than one absolute noise value for every gene. Diversity-aware mutation increases exploration when the population collapses. Stalls can also boost mutation and increase the random immigrant fraction to 0.25. Elites are copied without mutation.

The optional `evolution.init_genome` setting can seed a population from a `.npy` vector or checkpoint. The first seeded individual is copied exactly, other seeded individuals are mutated copies, and the configured immigrant fraction remains random.

## Fitness And Reward Shaping

The default reward is designed to value actual progress while making idle or score-farming behavior unattractive. Its major terms are:

```text
progress * 6
kills * 20
boss_damage * 5
survival_steps * 0.001
level_completion * 1000
death * -50
```

Additional rules include:

- Raw score is not rewarded by default (`score_coef: 0`); score deltas can still infer kills when a direct RAM kill counter is unavailable.
- Progress without combat after 64 pixels is scaled down to 20%.
- Idle behavior is penalized after 24 idle steps.
- 320 consecutive idle steps trigger an idle timeout and forfeit accumulated relative progress.
- Falling and dropping with `DOWN` on dry ground have penalties.
- Kills inferred from score are capped per step to prevent score spikes from dominating fitness.
- Death terminates an episode by default.

Fitness coefficients and thresholds are configuration values in the `reward` section of each YAML file.

## Curriculum Learning

The scenario registry contains these stages:

1. `movement`
2. `movement_shooting`
3. `enemies`
4. `projectiles`
5. `extended_level1`
6. `full_level_1`
7. `boss`
8. `full_game`

Stages can set their own starting save state, maximum episode length, progress threshold, and kill threshold. Advancement normally requires the threshold to be met for `advance_patience` consecutive generations. When the stage changes, the population is re-scored under the new environment before evolution continues.

Curriculum is disabled in the default and live configurations. Enable it explicitly with `curriculum.enabled: true` and choose the stages for the experiment. The default configured curriculum list contains seven stages and omits the initial `movement` entry; this is intentional for experiments that begin with movement plus shooting, but it is not the complete eight-stage registry.

### Capturing save states

The repository includes several states under `roms/states/`. If a required state is missing, capture states from a real ROM:

```bash
uv run python scripts/capture_states.py \
  --rom roms/Contra.nes
```

Capture named milestones:

```bash
uv run python scripts/capture_states.py \
  --rom roms/Contra.nes \
  --capture-at 3:level1_enemies.state \
  --capture-at 6:level1_boss.state
```

The script installs states into both `.custom_retro/Contra-Nes/` and `roms/states/`. It can also replay a trained `.npy` genome or `.pkl` checkpoint to capture a survivable progress state. The start state used for that replay must match the scenario used during training, or the deterministic trajectory can diverge.

The `boss` scenario refers to `level1_boss.state`; make sure that file exists before enabling that stage.

## Deterministic Parallel Evaluation

The evaluator uses persistent worker processes with the multiprocessing `spawn` context. Each worker creates and reuses its emulator rather than rebuilding it for every individual. Workers are rebuilt when an environment configuration changes, such as a curriculum transition.

With `evaluation.seed_policy: fixed`, all individuals use the same seed set in a generation and elites can carry their exact fitness forward. `per_generation` re-evaluates elites using generation-specific seeds. `seeds_per_individual` can be increased to average across multiple episodes at the cost of throughput.

The resource monitor records RAM, CPU, episodes per second, emulator FPS, and generation duration. A warning is emitted when RAM exceeds `resources.max_ram_percent`.

## Visualization

Population visualization uses a pygame window with an automatically sized tiled layout. It can show:

- One tile per population member with the latest frame.
- Fitness, progress, kills, and status for each tile.
- Elite and carried-fitness indicators.
- Generation, best/mean/median/worst fitness, diversity, mutation rate, throughput, emulator FPS, and resource usage.
- A live matplotlib chart of fitness history.

Single-agent evaluation uses a playback window. Space pauses playback; `ESC` or `q` closes it.

Visualization is a live surface, not a guaranteed artifact. Training always writes CSV metrics, but the current training flow does not reliably save plot image files or `metadata.json` even though the metrics logger has support for metadata.

## Checkpoints And Run Outputs

Training creates a timestamped run directory:

```text
runs/YYYYMMDD_HHMMSS/
  config.yaml
  train.log
  metrics.csv
  best_genome.npy
  best_fitness.txt
  checkpoints/
    gen_XXXX.pkl
    gen_XXXX.best.npy
```

It also writes root-level checkpoints at the configured interval:

```text
checkpoints/gen_0005.pkl
checkpoints/gen_0005.best.npy
```

Pickle checkpoints contain the population vectors, fitnesses, best genome, GA configuration, environment configuration, network configuration, metric history, NumPy RNG state, GA bookkeeping, and curriculum stage. This is enough for `--resume` to restore the evolutionary run rather than merely reload one policy.

`metrics.csv` contains per-generation fitness and resource fields such as best/mean/median/worst fitness, diversity, mutation settings, progress, kills, RAM, CPU, episodes per second, emulator FPS, and generation duration.

## Project Layout

```text
configs/                  YAML experiment configurations
roms/                     User-provided ROM location and save states
checkpoints/              Root-level saved populations and best genomes
runs/                     Timestamped logs, metrics, configs, and checkpoints
scripts/train.py          Genetic training CLI
scripts/evaluate.py       Checkpoint/genome evaluation CLI
scripts/benchmark.py      Worker-count benchmark CLI
scripts/capture_states.py Save-state capture CLI
src/agent/                CNN, genome, and policy implementations
src/baselines/            Random, scripted, always-right, and always-shoot controllers
src/environment/          Gymnasium wrapper, mock env, actions, observations, rewards, scenarios
src/evaluation/           Single-agent and parallel population evaluation plus metrics
src/evolution/            Selection, crossover, mutation, diversity, and GA loop
src/training/             Trainer, checkpoints, curriculum, and orchestration
src/utils/                Config loading, logging, seeds, resources, and paths
src/visualization/        Dashboard, playback, charts, and render helpers
tests/                    Unit and integration tests
docs/SETUP.md             Setup and hardware notes
```

## Tests And Development Checks

Run the full test suite:

```bash
uv run pytest -q
```

Useful focused checks:

```bash
uv run pytest -q tests/test_environment.py tests/test_reward.py
uv run pytest -q tests/test_ga_operators.py tests/test_ga_improvements.py
uv run ruff check .
uv run mypy src
```

The tests cover action mapping, configuration merging, the mock environment, rewards, network and genome round trips, GA operators, diversity and adaptive mutation, checkpoint serialization, parallel evaluation, dashboard layout, playback, baselines, and integration behavior.

The suite primarily exercises the mock environment. Passing tests does not prove that a particular Contra ROM, RAM map, save-state set, stable-retro build, or macOS pygame session works end to end. Validate those pieces with a short real-ROM debug run before committing to a long training job.

## Performance Notes

The implementation is designed for Apple Silicon laptops, including an M4 with 16 GB RAM:

- Pixel observations remain `uint8` until the network forward pass.
- A `(4, 84, 84)` observation uses about 28 KB before Python/container overhead.
- Real retro workers are heavier than mock workers; 4-8 workers is a reasonable starting range, but benchmarking is recommended.
- Headless mode avoids pygame overhead.
- The manual NumPy convolution implementation is intentionally small and CPU-oriented. Large populations, long episodes, and many workers increase runtime and memory use.

## Limitations And Caveats

- Mock mode is synthetic and cannot establish real gameplay ability.
- Real-ROM compatibility depends on the ROM variant and the custom RAM addresses in `emulator_factory.py`.
- The ROM is not included and must be supplied by the user.
- The `boss` scenario requires a `level1_boss.state` file, which may need to be captured locally.
- `--render-mode best` and `--render-mode comparison` are parsed but currently use the same population dashboard path.
- Live visualization requires a working graphical pygame environment and adds thread/process coordination.
- The training flow writes CSV and checkpoint artifacts, but plot image files and metadata JSON should not be assumed to exist.
- Configuration values describe experiments; they are not claims that the current default genome has completed the game.

## License

The project is intended to be released under the MIT License. Contra and any ROM or save-state data remain subject to their respective copyright and distribution restrictions. Do not redistribute copyrighted ROM files with this project.
