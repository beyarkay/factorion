# CLAUDE.md

## Project Overview

Factorion is a reinforcement learning project that trains agents to autonomously design and build factories inspired by Factorio. The agent places entities (transport belts, assembling machines, inserters) on a grid to transform input items into desired outputs, optimizing for maximum throughput. Uses curriculum learning to gradually increase complexity.

## Tech Stack

- **Language**: Python 3
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

## Setup

### Prerequisites

- **Python 3.11+**
- **Rust toolchain** (rustc, cargo) — install via [rustup](https://rustup.rs/)
- **maturin** — needed to build the `factorion_rs` Rust extension; install with `pip install maturin`
- **uv** (optional) — fast pip alternative; install with `pip install uv`

### Full environment setup

```bash
# 1. Create and activate a virtualenv (maturin requires one)
python -m venv .venv
source .venv/bin/activate

# 2. Install Python dependencies
uv pip install -r requirements.txt   # or: pip install -r requirements.txt

# 3. Install dev/test tools not in requirements.txt
pip install maturin pytest

# 4. Build and install the Rust extension (must be inside the venv)
cd factorion_rs && maturin develop --release && cd ..
```

### Without a virtualenv (e.g. Docker / CI)

`maturin develop` requires a virtualenv. If you cannot create one, build a
wheel and install it manually:

```bash
pip install maturin pytest
cd factorion_rs
maturin build --release
pip install target/wheels/factorion_rs-*.whl --force-reinstall --no-deps
cd ..
```

### Common pitfalls

- **`maturin: command not found`** — run `pip install maturin` first.
- **`Couldn't find a virtualenv or conda environment`** — `maturin develop`
  requires an activated virtualenv (`source .venv/bin/activate`) or conda env.
  If you can't use one, use `maturin build` + `pip install` instead (see above).
- **`ModuleNotFoundError: No module named 'numpy'` (or matplotlib, pandas, etc.)** —
  run `pip install -r requirements.txt` to install all Python dependencies.
- **`No module named 'pytest'`** — `pytest` is not in `requirements.txt`;
  install it with `pip install pytest`.

## Running

```bash
python ppo.py \
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

## Linting

```bash
ruff check .
```

**Note**: `ruff` is a Python linter. Never run it on non-Python files (YAML, TOML, etc.) — it will produce nonsensical parse errors.

## Testing

```bash
WANDB_MODE=disabled WANDB_DISABLED=true python -m pytest tests/ -v
```

Rust unit tests:
```bash
cd factorion_rs && cargo test && cd ..
```

Benchmarks (Python vs Rust throughput):
```bash
WANDB_MODE=disabled WANDB_DISABLED=true python tests/bench_throughput.py
```

## Pre-completion Checklist

Before claiming work is done, run all of the following:

1. **Rust format + lint**: `cd factorion_rs && cargo fmt && cargo clippy -- -D warnings && cd ..`
2. **Rust tests**: `cd factorion_rs && cargo test && cd ..`
3. **Build the Rust extension**: `cd factorion_rs && maturin develop --release && cd ..` (or `maturin build --release && pip install target/wheels/factorion_rs-*.whl --force-reinstall --no-deps` if no venv)
4. **Python tests**: `WANDB_MODE=disabled WANDB_DISABLED=true python -m pytest tests/ -v`
5. **Python linter**: `ruff check .`
6. **PPO smoke test**: `WANDB_MODE=disabled python ppo.py --seed 1 --env-id factorion/FactorioEnv-v0 --total-timesteps 5000`

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
