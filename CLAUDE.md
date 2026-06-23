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
2. **SFT pretraining** — `sft.py` runs supervised training on those pairs, giving the policy a strong prior over entity placement. It uploads the best checkpoint as a W&B artifact (type `model`, containing `sft_checkpoint.pt`) so RL can pull it by run id.
3. **RL finetuning** — PPO (`ppo.py`) refines the pretrained policy to push beyond what imitation achieves. Load an SFT checkpoint with `--start-from` (accepts **either** a local `.pt` path **or** a W&B run id like `j0s5y2mc` — the run's model artifact is downloaded automatically). Every episode builds from a **fully-blank** grid (no curriculum). The canonical SFT base is run **`j0s5y2mc`**; the bar PPO must clear is its `val/throughput_eot ≈ 0.11` (see "Throughput metrics" below).

Historically the project did RL-from-scratch with heavy scaffolding (curriculum on `num_missing_entities`, reward shaping, action masking) to handle the sparse-reward problem. Most of that scaffolding still exists but its role changes under the new pipeline: the curriculum axis becomes a data-sampling knob during SFT, and RL starts from a much better policy so sparse rewards matter less.

**Implication when adding features:** new lesson types expand pretraining coverage (more diverse entity/layout patterns the pretrained model understands). They are **not** rungs of a fixed difficulty ladder — diversity matters at every `num_missing_entities` level.

## Tech Stack

- **Language**: Python 3.11+
- **RL Framework**: Gymnasium
- **ML Framework**: PyTorch (`torch`)
- **Experiment Tracking**: Weights & Biases (`wandb`)
- **CLI Parsing**: tyro (dataclass-based)
- **Graphs**: networkx (flow-graph visualization only; graph construction and throughput live in the Rust engine)

## Project Structure

- `ppo.py` — Main PPO training script. Contains `Args` dataclass, `FactorioEnv` (Gymnasium env), and `AgentCNN` (PyTorch policy network).
- `sft.py` — Supervised pretraining. `SFTArgs` dataclass, `extract_expert_actions` (factory pair → training tuples), `run_rollout_eval` (greedy throughput eval), `train_sft` (training loop). Imports `AgentCNN`/`FactorioEnv` **from `ppo.py`**, so the SFT and PPO policies are literally the same network — a checkpoint from one loads into the other.
- `factorion.py` — Environment utilities module: enums (`Channel`, `Direction`, `Entity`, `Item`, `Recipe`, `LessonKind`), blueprint encoding/decoding, factory generation, lesson creation, factory-graph building (`world2graph`). Import symbols directly (`from factorion import build_factory, blank_entities, Channel, ...`).
- `factorion_rs/` — Rust extension (PyO3/maturin) for the throughput simulation (`simulate_throughput`). Built into the venv via `maturin develop`; stubs in `factorion_rs/__init__.pyi` (keep in sync or `ty` fails).
- `scripts/factory_builder.py` — Local HTTP UI to hand-build factories and query a checkpoint's predictions. Shares `_resolve_wandb_checkpoint` with `ppo.py`.
- `scripts/ci/` — CI/training automation: `ppo_train.sh` & `sft_train.sh` (in-pod RunPod training), `create_sweep.py`/`apply_sweep_best.py`/`apply_sft_sweep_best.py` (W&B sweeps → PR), `runpod_create.py`/`runpod_destroy.py`.
- `.github/workflows/` — `ppo-train.yml` & `sft-train.yml` (manual `workflow_dispatch` GPU runs on RunPod), `ci.yml`, `claude.yml`.
- `sweep.yaml` — W&B Bayesian sweep config: PPO RL-finetuning from the SFT base, maximising `eval/throughput_eot`. Pins the SFT checkpoint/architecture/timesteps in `command`; sweeps the RL levers (step_penalty, ent_coef, lr, throughput_reward_scale, target_kl, gamma, gae_lambda).
- `b64-to-json.py` / `json-to-b64.py` — Blueprint encoding/decoding utilities.
- `factorio-data/` — Git submodule with Factorio game data.
- `factorio-icons/` — Entity icon PNGs.

### Codebase map (grep these symbols)

- **State tensor** is `(C, W, H)` with `C = len(Channel) = 5` channels: `ENTITIES`, `DIRECTION`, `ITEMS` (recipe/filter), `MISC` (underground up/down), `FOOTPRINT` (1 = buildable).
- **Lessons** (`LessonKind` in `factorion.py`): `MOVE_ONE_ITEM`, `SPLITTER_SPLIT`, `SPLITTER_MERGE`, `ASSEMBLE_1IN_1OUT`, `MOVE_VIA_UG_BELT`, `ASSEMBLE_2IN_1OUT`, `FROM_BLUEPRINT`. Each `build_factory(size, kind, seed, ...)` returns `Optional[Factory]` (rejection sampling can fail → `None`); `blank_entities(factory, num_missing_entities, seed)` removes N entity *units* (multi-tile entities count as one).
- **`ppo.py`**: `Args` (all PPO hyperparams + `start_from`, `critic_warmup`, `eval_every`), `FactorioEnv` (`reset`/`step`; reward = `-step_penalty` per step `+ throughput_reward_scale * thput_normed` on termination; `eot` is a real action that ends the episode; `info['kind']` carries the lesson for per-lesson logging), `AgentCNN` (encoder + per-head outputs: tile/entity/direction/item/misc + `critic_head` + `eot_head`; stashes `_last_head_entropy`/`_last_eot_prob` for the policy/* metrics), `_resolve_start_from`/`_resolve_wandb_checkpoint` (checkpoint loading), `_run_signature` (run name), `_build_eval_set`/`_run_greedy_eval` (the eval/ section).
- **`sft.py`**: `run_rollout_eval` returns `RolloutEval` with `overall`/`overall_eot` and per-`LessonKind` breakdowns; checkpoint is selected on `val/throughput` (EOT ignored). PPO reuses this for its `eval/` metrics (lazy import to avoid the ppo↔sft cycle).

## W&B dashboards

Runs are named by a hyperparameter signature, not a timestamp (`ppo.py:_run_signature`, `sft.py:_artifact_name`), e.g. `ppo-s11-lr5e-05-ent0-cw10-fromj0s5y2mc-c93-69-96-seed1`. `global_step` (env steps) is the PPO x-axis. PPO logs once per iteration into these sections (see `define_metric` block in `ppo.py`):

- **`eval/`** — periodic greedy held-out throughput (`eval/throughput`, `eval/throughput_eot`, `eval/{LESSON}/*`), every `--eval-every` iters; directly overlay-able with the SFT baseline. **This is the headline progress signal**, and the sweep metric (`sweep.yaml`).
- **`rollout/`** — on-policy sampled episode stats (`throughput`, `thput_raw`, `reward`, `length`, `eot_rate`, `invalid_frac`, `num_entities`, `entity_efficiency`, `frac_reachable`) + per-lesson `rollout/{LESSON}/{throughput,reward,length}`.
- **`policy/`** — acting-policy distribution: `entropy`, per-head `entropy_{tile,entity,direction,item,misc,eot}`, `eot_prob`.
- **`losses/`** `policy,value,entropy,total,approx_kl,clipfrac,explained_variance`; **`optim/`** `lr,critic_lr,ent_coef,grad_norm,critic_warmup`; **`perf/`** `sps,rollout_seconds,update_seconds,eval_seconds`.

## Throughput metrics (`throughput` vs `throughput_eot`)

Both come from a greedy rollout that blanks the **whole** grid and rebuilds from empty (the honest "can it build from scratch" test). Per-factory throughput is `info['thput_normed']` ∈ [0, 1] — raw items/sec ÷ that factory's max, so a perfectly-rebuilt factory scores 1.0 regardless of belt speed. Defined in `sft.py:run_rollout_eval`:

- **`throughput`** (`overall`) — build skill *ignoring* the EOT head: step until the env finishes (throughput 1.0 or max_steps) and take the final throughput. "Can the model physically reconstruct the factory?"
- **`throughput_eot`** (`overall_eot`) — build skill *respecting* the EOT head: snapshot the throughput the first time the EOT prob crosses `rollout_eot_threshold` (0.5); if it never fires, fall back to the final throughput. "How good is the factory at the moment the model decides it's done?"

`throughput_eot ≤ throughput` whenever the model stops early. **The RL goal is for PPO's achieved throughput to exceed the SFT base's `val/throughput_eot` (≈0.11 for `j0s5y2mc`).** Per-lesson the SFT base is uneven (MOVE_ONE_ITEM ~0.38, assembler lessons ~0).

## SFT → PPO handoff

`ppo.py --start-from <ckpt>` loads the full SFT `AgentCNN` state dict (encoder + every policy/eot head). The **critic head is the only part SFT never trained**, so `--critic-warmup N` freezes the actor (encoder + policy heads) for N iterations and trains the value head alone against the fixed SFT features before joint PPO begins; LR anneal + entropy schedule restart at the unfreeze. For finetuning an SFT policy, set `--ent-coef-start/--ent-coef-end 0` and a small `--learning-rate` (e.g. 5e-5). Every episode builds from a full blank (the legacy `num_missing_entities` curriculum was removed).

## Python Environment

Use `uv run` to run all Python commands. The `.venv` is managed by `uv` and is **not** activated in the shell PATH. Never use bare `python`, `ruff`, `pytest`, etc. — always prefix with `uv run`.

## Setup

```bash
# Create .venv and install all deps (runtime + dev tooling) from pyproject.toml + uv.lock
uv sync
# Build the Rust extension into the venv
uv run maturin develop --release --manifest-path factorion_rs/Cargo.toml
```

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`. There is
no `requirements.txt` — `uv sync` (driven by `uv.lock`) is the only supported
way to set up the environment.

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

Benchmarks (Rust throughput):

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
6. **Type checker**: `uv run ty check .` (also enforced by the pre-push hook)
7. **PPO smoke test**: `WANDB_MODE=disabled uv run python ppo.py --seed 1 --env-id factorion/FactorioEnv-v0 --total-timesteps 5000`
8. **SFT smoke test**: `WANDB_MODE=disabled uv run python sft.py --seed 1 --size 5 --num-samples 200 --epochs 2 --batch-size 32 --layer1 16 --layer2 16 --layer3 16 --checkpoint-path /tmp/sft_smoke.pt --summary-path /tmp/sft_smoke.json`

All eight must pass before the work is considered complete.

## Code Conventions

- CLI arguments are defined via a `Args` dataclass parsed by `tyro`.
- The RL environment follows the Gymnasium API (`reset`, `step`, `render`).
- Factory state is represented as 3D tensors with semantic channels.
- Experiment runs are tracked with Weights & Biases.
- Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`; set up the env with `uv sync`.

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
