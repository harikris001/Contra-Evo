# Setup & Benchmark Notes (M4 16GB)

## Install
```bash
uv venv --python 3.11
uv pip install -e .[dev]
# optional torch for MPS benchmark:
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## ROM Import
```bash
# after obtaining Contra.nes legally:
python -m retro.import /path/to/Contra.nes
# verify:
uv run python -c "import stable_retro; print(stable_retro.data.list_games(inttype=stable_retro.data.Integrations.STABLE))"
```

## Resource Monitoring
`src/utils/resources.py` uses `psutil`:
- RAM: `virtual_memory().percent / used / total`
- CPU: `cpu_percent()`
- eps/sec: `episodes / gen_duration`
- emulator FPS: `steps / gen_duration`

Headless mode skips pygame overhead, enabling max throughput. Visualization runs at 15 FPS decoupled.

## Memory Estimate (uint8)
- 84×84×4 = 28,224 bytes/obs vs float32 112KB
- 8 agents × 30MB retro overhead ≈ 240MB total < 16GB
- Benchmark chooses optimal workers before OOM.

## Benchmark
```bash
uv run python scripts/benchmark.py --config configs/default.yaml --workers 4,6,8,10,12 --output runs/benchmark.json
```
Records: duration, eps/sec, RAM%, CPU%, mean_fitness. Recommends max eps/sec under 85% RAM.

## Curiosity: Worker Count
For mock env (lightweight), 1 worker can beat 2 due to spawn overhead. For real retro (heavier), 4–6 workers is optimal on M4 16GB. Always benchmark.
