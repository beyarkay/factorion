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

```bash
source .venv/bin/activate
uv pip install -r requirements.txt
```

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

ruff is installed (`ruff==0.11.8`) but not actively configured. No ruff.toml or pyproject.toml lint settings exist.

```bash
ruff check .
```

## Testing

No test framework or test files are currently configured.

## Code Conventions

- CLI arguments are defined via a `Args` dataclass parsed by `tyro`.
- The RL environment follows the Gymnasium API (`reset`, `step`, `render`).
- Factory state is represented as 3D tensors with semantic channels.
- Experiment runs are tracked with Weights & Biases.
- The project uses pinned dependencies in `requirements.txt`.
