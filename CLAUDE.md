# CLAUDE.md

NOTE FROM USER (integrate this note upon reading): you can browse the factorio
wiki to get canonical information. e.g. https://wiki.factorio.com/Splitter has
information about splitters, this page does a search:
https://wiki.factorio.com/index.php?search=some+search+term&title=Special%3ASearch&go=Go
and this page contains information about electronic circuits:
https://wiki.factorio.com/Electronic_circuit. You should use this to double
check throughputs and crafting speeds, this information is usually in a sidebar
`div.infobox`, where there's a table with a bunch of info. we're mostly
interested in `Belt speed`

## Project Overview

Factorion is a reinforcement learning project that trains agents to autonomously design and build factories inspired by Factorio. The agent places entities (transport belts, assembling machines, inserters) on a grid to transform input items into desired outputs, optimizing for maximum throughput. Uses curriculum learning to gradually increase complexity.

## Tech Stack

- **Language**: Python 3.11+
- **RL Framework**: Gymnasium
- **ML Framework**: PyTorch (`torch`)
- **Experiment Tracking**: Weights & Biases (`wandb`)
- **CLI Parsing**: tyro (dataclass-based)
- **Notebook**: marimo (for `factorion.py`)
- **Graphs**: networkx (throughput calculation)

## Project Structure

- `ppo.py` — Main PPO training script. Contains `Args` dataclass, `FactorioEnv` (Gymnasium env), and `AgentCNN` (PyTorch policy network).
- `factorion.py` — Marimo notebook with environment utilities: enums (`Channel`, `Direction`, `Entity`, `Item`, `Recipe`), blueprint encoding/decoding, factory generation, lesson creation, throughput calculation.
- `sweep.yaml` — Weights & Biases Bayesian hyperparameter sweep config.
- `b64-to-json.py` / `json-to-b64.py` — Blueprint encoding/decoding utilities.
- `factorio-data/` — Git submodule with Factorio game data.
- `factorio-icons/` — Entity icon PNGs.

## Python Environment

Use `uv run` to run all Python commands. The `.venv` is managed by `uv` and is **not** activated in the shell PATH. Never use bare `python`, `ruff`, `pytest`, etc. — always prefix with `uv run`.

## Setup

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
uv pip install maturin
uv run maturin develop --release --manifest-path factorion_rs/Cargo.toml
```

## Running

```bash
uv run python ppo.py \
  --seed 1 \
  --env-id factorion/FactorioEnv-v0 \
  --track \
  --wandb-project-name factorion \
  --total-timesteps 500000
```

Run a W&B sweep:

```bash
bash run_sweep.sh
```

### Sweep configuration

There is exactly one sweep config file: `sweep.yaml`. The CI sweep pipeline (`gpu-sweep` job) always reads from this file. To change what gets swept, edit `sweep.yaml` directly — do NOT create separate sweep files. The old version is always recoverable from git history.

## Linting

```bash
uv run ruff check .
```

**Note**: `ruff` is a Python linter. Never run it on non-Python files (YAML, TOML, etc.) — it will produce nonsensical parse errors.

## Testing

```bash
WANDB_MODE=disabled WANDB_DISABLED=true uv run python -m pytest tests/ -v
```

Rust unit tests:

```bash
cd factorion_rs && cargo test && cd ..
```

Benchmarks (Python vs Rust throughput):

```bash
WANDB_MODE=disabled WANDB_DISABLED=true uv run python tests/bench_throughput.py
```

## Pre-completion Checklist

Before claiming work is done, run all of the following:

1. **Rust format + lint**: `cd factorion_rs && cargo fmt && cargo clippy -- -D warnings && cd ..`
2. **Rust tests**: `cd factorion_rs && cargo test && cd ..`
3. **Build the Rust extension**: `cd factorion_rs && maturin develop --release && cd ..`
4. **Python tests**: `WANDB_MODE=disabled WANDB_DISABLED=true uv run python -m pytest tests/ -v`
5. **Python linter**: `uv run ruff check .`
6. **PPO smoke test**: `WANDB_MODE=disabled uv run python ppo.py --seed 1 --env-id factorion/FactorioEnv-v0 --total-timesteps 5000`

All six must pass before the work is considered complete.

## Code Conventions

- CLI arguments are defined via a `Args` dataclass parsed by `tyro`.
- The RL environment follows the Gymnasium API (`reset`, `step`, `render`).
- Factory state is represented as 3D tensors with semantic channels.
- Experiment runs are tracked with Weights & Biases.
- The project uses pinned dependencies in `requirements.txt`.

## Personal Preferences

- Always run smoke tests before claiming work is done.
- New code must be accompanied by end-to-end tests.
