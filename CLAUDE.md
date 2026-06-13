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

Factorion trains agents to autonomously design and build factories inspired by Factorio. The agent places entities (transport belts, assembling machines, inserters, splitters, …) on a grid to transform input items into desired outputs, optimizing for throughput.

The training pipeline is moving toward an **LLM-style two-stage split**:

1. **Data generation** — `build_factory()` in `factorion.py` constructs a known-correct factory (returning `Optional[Factory]`), then `blank_entities()` produces a *(partial-factory, correct-completion)* training pair by removing N entities from it. Each lesson type (`MOVE_ONE_ITEM`, `SPLITTER_SPLIT`, `SPLITTER_MERGE`, …) covers a different entity or layout pattern.
2. **SFT pretraining** — supervised training on those pairs (see `sft.py` when PR #47 lands) gives the policy a strong prior over entity placement.
3. **RL finetuning** — PPO (`ppo.py`) refines the pretrained policy to push beyond what imitation achieves. Load an SFT checkpoint with `--start_from` and skip easy curriculum levels via `--start_curriculum_level`.

Historically the project did RL-from-scratch with heavy scaffolding (curriculum on `num_missing_entities`, reward shaping, action masking) to handle the sparse-reward problem. Most of that scaffolding still exists but its role changes under the new pipeline: the curriculum axis becomes a data-sampling knob during SFT, and RL starts from a much better policy so sparse rewards matter less.

**Implication when adding features:** new lesson types expand pretraining coverage (more diverse entity/layout patterns the pretrained model understands). They are **not** rungs of a fixed difficulty ladder — diversity matters at every `num_missing_entities` level.

## Tech Stack

- **Language**: Python 3.11+
- **RL Framework**: Gymnasium
- **ML Framework**: PyTorch (`torch`)
- **Experiment Tracking**: Weights & Biases (`wandb`)
- **CLI Parsing**: tyro (dataclass-based)
- **Graphs**: networkx (throughput calculation)

## Project Structure

- `ppo.py` — Main PPO training script. Contains `Args` dataclass, `FactorioEnv` (Gymnasium env), and `AgentCNN` (PyTorch policy network).
- `factorion.py` — Environment utilities module: enums (`Channel`, `Direction`, `Entity`, `Item`, `Recipe`), blueprint encoding/decoding, factory generation, lesson creation, throughput calculation. Import symbols directly (`from factorion import build_factory, blank_entities, Channel, ...`).
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
uv run maturin develop --release --manifest-path factorion_rs/Cargo.toml
```

### Fresh worktree / venv setup

Each `.claude/worktrees/<name>/` is a brand-new checkout with an **empty
`.venv`**. The `pre-push` git hook runs the full pre-completion checklist
(`cargo fmt/clippy/test`, `maturin`, `ruff`, `pytest`) and will fail with
`command not found` until the venv is populated.

**Do not use `requirements.txt` in a worktree** — it does not resolve cleanly
under `uv` (fastapi/runpod/starlette pin conflicts). Instead mirror CI's
per-package install list (kept in sync with `.github/workflows/ci.yml`):

```bash
uv pip install ruff==0.11.8 pytest pytest-timeout
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install numpy gymnasium networkx tyro pydantic marimo maturin tqdm matplotlib pandas plotly wandb tensorboard runpod
uv run maturin develop --release --manifest-path factorion_rs/Cargo.toml
```

`runpod` is required at pytest **collection** time (`tests/test_runpod_cost.py`
imports `scripts/ci/runpod_destroy.py`), so don't omit it.

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

## CI & RunPod GPU jobs

CI lives in `.github/workflows/ci.yml`. The cheap jobs (`lint`, `python-tests`,
`rust-tests`) run on every PR. The **GPU jobs run on RunPod and are
label-gated** — they do **not** fire on a plain PR open. The `check-gpu-trigger`
job decides what runs based on PR **labels** or a manual `workflow_dispatch`.
Because `labeled` is in the trigger list, **adding a label to an existing PR
kicks the job off** (no new push needed).

Existing labels → what they run:

| Label | Job / script | What it does |
|-------|--------------|--------------|
| `gpu-test` | `gpu-smoke-test` → `scripts/ci/gpu_smoke_test.sh` | Quick PPO/env GPU smoke |
| `gpu-benchmark` | `gpu-benchmark` | Multi-seed PPO throughput comparison vs baseline |
| `sweep` | `sweep-setup` → `gpu-sweep` → `sweep-report` | PPO W&B Bayesian sweep (`sweep.yaml`) |
| `sft-smoketest` | `sft-smoke-test` → `scripts/ci/sft_smoke_test.sh` | **One** tracked run: `sft.py --size 11 --num-samples 300000 --epochs 30 --track`. Despite the name it's a full run (4h watchdog). |
| `sft-sweep` | `sft-sweep-setup` → `sft-sweep` → `sft-sweep-report` | SFT W&B Bayesian sweep (`sft_sweep.yaml`): every run pinned to size=11 / 300k samples / 30 epochs, searches LR schedule, regularisation, per-head loss weights, encoder channels. `run_cap: 60`, ~20 runs / 5 agents by default. Optimises `val/acc`. |

**Gotcha:** the trigger logic also checks for an `sft-benchmark` label
(`scripts/ci/sft_benchmark.sh`, a multi-seed SFT comparison), **but no such
GitHub label exists** in the repo — so the benchmark can only be launched via
`workflow_dispatch`, or by creating the label first.

RunPod pods self-terminate via a watchdog (4h smoketest / 2h benchmark) plus a
teardown action; result summaries (and sweep URLs) are posted back as PR
comments. All SFT runs log to the `factorion` W&B project; the north-star
metric for the current MOVE_ONE_ITEM overfit work is `val/MOVE_ONE_ITEM/throughput`.

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
7. **SFT smoke test**: `WANDB_MODE=disabled uv run python sft.py --seed 1 --size 5 --num-samples 200 --epochs 2 --batch-size 32 --chan1 16 --chan2 16 --chan3 16 --checkpoint-path /tmp/sft_smoke.pt --summary-path /tmp/sft_smoke.json`

All seven must pass before the work is considered complete.

## Code Conventions

- CLI arguments are defined via a `Args` dataclass parsed by `tyro`.
- The RL environment follows the Gymnasium API (`reset`, `step`, `render`).
- Factory state is represented as 3D tensors with semantic channels.
- Experiment runs are tracked with Weights & Biases.
- The project uses pinned dependencies in `requirements.txt`.

## Rust Type System

**Make invalid states unrepresentable.** Anything that can be encoded in
the Rust type system — non-empty collections, mutually exclusive states,
pre/post-conditions on construction — should be expressed as types
rather than checked at runtime. Compile-time guarantees are strictly
preferable to runtime asserts or unit tests that re-check static data.

Useful crates:
- [`nonempty`](https://docs.rs/nonempty/) — `NonEmpty<T>` for collections
  that must contain at least one element. Use this instead of
  `Vec<T>` + a runtime `is_empty()` check.

If a runtime test reads "this hardcoded data is non-empty / has positive
values / has the right shape", that test is a smell — express the
invariant in the type instead, and delete the test.

## Personal Preferences

- Always run smoke tests before claiming work is done.
- New code must be accompanied by end-to-end tests.
- **Small, incremental commits.** Make each commit as small as possible while still being a logical, self-contained unit of change (e.g., "expand entity head" is one commit, "sample all lesson kinds" is another, "update tests for new return shape" is another). Do not bundle unrelated changes. Each commit should pass the pre-completion checklist on its own.
