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

1. **Data generation** — `build_factory()` in `factorion.py` constructs a known-correct factory (returning `Optional[Factory]`), then `blank_entities()` produces a *(partial-factory, correct-completion)* training pair by removing N entities from it. Each lesson type (`MOVE_ONE_ITEM`, `SPLITTER_SPLIT`, `SPLITTER_MERGE_SIDELOADED`, …) covers a different entity or layout pattern.
2. **SFT pretraining** — `sft.py` runs supervised training on those pairs, giving the policy a strong prior over entity placement. It uploads the best checkpoint as a W&B artifact (type `model`, containing `sft_checkpoint.pt`) so RL can pull it by run id.
3. **RL finetuning** — PPO (`ppo.py`) refines the pretrained policy to push beyond what imitation achieves. Load an SFT checkpoint with `--start-from` (accepts **either** a local `.pt` path **or** a W&B run id like `j0s5y2mc` — the run's model artifact is downloaded automatically). Every episode builds from a **fully-blank** grid (no curriculum). The canonical SFT base is run **`j0s5y2mc`**; the bar PPO must clear is its `val/thput ≈ 0.11` (see "Throughput metric" below).

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

- `ppo.py` — Main PPO training script. Contains `FactorioEnv` (Gymnasium env) and `AgentCNN` (PyTorch policy network); imports the `PpoArgs` hyperparameter dataclass from `training_config.py`.
- `training_config.py` — Single source of truth for training hyperparameters. Defines `SharedArgs` (defaults common to PPO + SFT: grid `size`, CNN encoder shape, W&B project, seed, tags) and the `PpoArgs`/`SftArgs` dataclasses that inherit it. `ppo.py`/`sft.py` and the CI scripts all read defaults from here.
- `sft.py` — Supervised pretraining. `SftArgs` dataclass, `extract_expert_actions` (factory pair → training tuples), `run_rollout_eval` (greedy throughput eval), `train_sft` (training loop). Imports `AgentCNN`/`FactorioEnv` **from `ppo.py`**, so the SFT and PPO policies are literally the same network — a checkpoint from one loads into the other.
- `factorion.py` — Environment utilities module: enums (`Channel`, `Direction`, `Entity`, `Item`, `Recipe`, `LessonKind`), blueprint encoding/decoding, factory generation, lesson creation, factory-graph building (`world2graph`). Import symbols directly (`from factorion import build_factory, blank_entities, Channel, ...`).
- `factorion_rs/` — Rust extension (PyO3/maturin) for the throughput simulation (`simulate_throughput`) and the lesson generator (`build_factory` in `src/factory_gen.rs`, with a fast deterministic RNG in `src/rng.rs`; exposed to Python as `factorion.build_factory`, which is what training calls). Built into the venv via `maturin develop`; stubs in `factorion_rs/__init__.pyi` (keep in sync or `ty` fails).
- `scripts/factory_builder.py` — Local HTTP UI to hand-build factories and query a checkpoint's predictions. Shares `_resolve_wandb_checkpoint` with `ppo.py`.
- `factorion-mod/` — Factorio mod + Python tooling (read `factorion-mod/README.md`). Two jobs: (1) in-game model inference over RCON (`mod/control.lua` + `server/server.py`); (2) the **engine-parity harness** (#261): `server/parity.py` replays `build_factory` factories inside real Factorio via `mod/parity.lua` and compares measured per-sink items/s against `factorion_rs.py_sink_deliveries` — the tool for verifying `simulate_throughput` against the real game (`scripts/parity_launch.sh` for one-command headless runs; `--dry-run` needs no Factorio).
- `ci/` — **All CI/training automation** (read `ci/README.md`). GPU jobs run on fire-and-forget, self-terminating RunPod pods, triggered by `/ci ...` PR comments (`ci/gh_command.py`) or `uv run python -m ci ...`; results are posted back as PR comments. Job specs in `ci/config.py` are the complete CI override surface (commitish, SFT `num_samples`, PPO `start_from`/`total_timesteps`, compare seeds) — every other hyperparameter flows from `training_config.py`. Includes the every-metric compare report + `assert pr:metric > main:metric` gating (`ci/report.py`), the leaked-pod watchdog (`ci/watchdog.py`), and the sweep configs `ci/sweep_ppo.yaml`/`ci/sweep_sft.yaml` (metric `eval/thput` / `val/thput`; results are reported, not auto-applied — defaults in `training_config.py` are edited by hand).
- `.github/workflows/` — thin pointers into `ci/`: `ci.yml` (lint + tests, no GPU), `ci-command.yml` (`/ci` comments), `ci-reporter.yml` (posts run results to PRs), `pod-watchdog.yml` (6-hourly pod reaper), `launch.yml` (manual dispatch), `claude.yml`.
- `factorio-icons/` — Entity icon PNGs.

### Codebase map (grep these symbols)

- **State tensor** is `(C, W, H)` with `C = len(Channel) = 5` channels: `ENTITIES`, `DIRECTION`, `ITEMS` (recipe/filter), `MISC` (underground up/down), `FOOTPRINT` (1 = buildable).
- **Lessons** (`LessonKind` in `factorion.py`, built from `factory_gen.rs::LessonKind` via `factorion_rs.py_lesson_kinds()`): `MOVE_ONE_ITEM`, `SPLITTER_SPLIT`, `SPLITTER_MERGE_SIDELOADED`, `ASSEMBLE_1IN_1OUT`, `MOVE_VIA_UG_BELT`, `ASSEMBLE_2IN_1OUT`, `MEMORISE_1_INGREDIENT_RECIPES` … `MEMORISE_4_INGREDIENT_RECIPES`, `MOVE_ONE_ITEM_CHAOS`, `CROSS_UNDER_BELT`, `FACTORY_1_INGREDIENT`. `build_factory(size, kind, seed, ...)` (Rust-backed) returns `Optional[Factory]` (rejection sampling can fail → `None`); `blank_entities(factory, num_missing_entities, seed)` removes N entity *units* (multi-tile entities count as one). Each lesson has a `build_*` generator in `factory_gen.rs`; `SPLITTER_MERGE_SIDELOADED` (`build_splitter_merge_sideloaded`) replaces the old throughput-hackable `SPLITTER_MERGE`: two sources are each capped to 7.5 i/s by a **side-load gadget** (a protected empty "decoy" belt forces the source to side-load onto one lane instead of curving to a full 15), then a splitter merges the two 7.5 arms onto one 15 i/s belt to a single sink — so *both* arms are throughput-necessary (drop one and the sink falls to 7.5), unlike the old merge whose lone sink was saturated by either source alone. Its decoy belts are the one sanctioned exception to the no-orphan invariant (see below). The `MEMORISE_N_INGREDIENT_RECIPES` lessons (`build_memorise_recipes`, one per exact ingredient count 1–4) place a random recipe's assembler fed/drained by `source → belt → inserter → assembler → inserter → belt → sink` arms with exactly one belt per arm (one source per ingredient, so lesson `N` has exactly `N` input sources); `CROSS_UNDER_BELT` (`build_cross_under_belt`) is validated by its own throughput/orphan invariants; `FACTORY_1_INGREDIENT` (`build_factory_1_ingredient`) places a row of 1+ assemblers sharing an input and an output belt lane (1–3 input and 1–3 output inserters per assembler), with the source/sink at semi-arbitrary cells wired to the lane ends by the UG-aware router — throughput deliberately varies (input-inserter / recipe-speed / output-inserter limited) so the critic sees good and bad layouts.
- **Underground lessons**: `MOVE_VIA_UG_BELT` forces underground belts with an `UNAVAILABLE`-footprint wall; `CROSS_UNDER_BELT` gives a real, protected belt line (the "obstruction" — a winding edge-to-edge cut) the crossing must tunnel *under* via `factory_gen.rs::ug_aware_belt_path` (a Dijkstra that may dip under blocked cells; minimal tunnel span, no 180° reversals or tile reuse, source→UG-down / UG-up→sink shortcuts).
- **No-orphan invariant**: no lesson's solved factory may contain orphan tiles — throughput must report `unreachable == 0` (checked by `factory_gen.rs::tests::test_no_orphan_tiles_every_lesson` across every `LessonKind`). Unreachability is judged per ENTITY (an entity is an orphan iff none of its lane nodes lie on a source→sink path) since e.g. an inserter-fed belt legitimately has one forever-empty lane. The **one opt-out** is `LessonKind::allows_orphans()` (exposed to Python as `factorion_rs.py_lesson_allows_orphans()`): `SPLITTER_MERGE_SIDELOADED` returns `true` because its side-load decoy belts are intentional protected orphans; the test then requires every orphan to be a protected tile (`unreachable == protected_positions.len()`), so no *accidental* orphan slips through.
- **Jargon**: the orientation-relative sides of a directional entity are **left, right, fore, aft** (a north-facing belt's left side is west, its fore side north). Use these consistently in code comments and docs.
- **Dual-lane belts** (`wiki/two-lane-system.md`, #65): graph nodes are NOT 1:1 with entities — belt-ish tiles (belt, UG, each splitter tile) emit one node per `Lane` (`NodeId.lane`), each capped at half the tile's `Item::flow_rate()` (belt tile 15 → 7.5/lane; a splitter is two belt tiles, so `flow_rate(Splitter)` is 15 per tile, 4 lane pools × 7.5 = 30 total); inserter/assembler/source/sink stay single-node. Node references use ONE canonical format everywhere (engine labels, `py_build_graph`, and fixture `graph:` blocks): `<entity_char>@x,y` plus `:L`/`:R` for lane nodes (grid-registry chars: `b@1,0:L`, `i@0,1`, `d`/`u`, `S`, `K`); fixture graph blocks may elide lane markers per document (entity-level, lane-projected comparison). All connection logic funnels through `entities.rs::calc_lane_aware_edges`: straight/curve/tunnel/splitter flow is lane-preserving; a perpendicular feed onto a belt with any other belt-connectable input sideloads onto the near-side lane (`belt_feeders`/`is_curved_belt` — lone side feed = curve; UG mouths and splitters never curve; a side feed onto a UG tile connects only the feeder lane over its surface half — aft of an entrance, fore of an exit); inserters drop onto exactly one lane (far when perpendicular, belt's right when in-line) and pick up with lane priority (near when perpendicular, left when in-line/curved) via greedy fill in `throughput.rs::pickup_input`.
- **Rendering factories to ASCII** — when asked to "render"/"show"/"draw" factories, use the real renderer, never hand-roll one: `from factorion import render_factory; print(render_factory(factory))` (accepts a `Factory` or a `(C, W, H)` world tensor; Rust binding `factorion_rs.render_factory(world_WHC)` lives in `factorion_rs/src/render.rs`). It returns the two-char grid format the textual fixtures use — `b`=belt, `i`=inserter, `l`=long-handed inserter, `a`=assembler box, `Y`=splitter, `d`/`u`=underground down/up, `S`=source, `K`=sink, each plus a facing marker `^>v<`; `..` is an empty tile. To also show each factory's recipe, read the assembler's tagged item from the `ITEMS` channel (or the sink's carried item).
- **Placing entities in a generator** (`factory_gen.rs`) — mutate the `World` through the existing `place_*` helpers, never hand-write `world.set(Channel::Entities/Direction, …)` for a belt/marker/splitter (same rule as the renderer above: don't re-implement a helper that exists). `place_belts(world, &[(x, y, dir, Misc)])` lays plain belts **and** underground tunnel ends, one tuple per tile; `place_marker` for sources/sinks, `place_splitter` for the 2-tile splitter, `place_assembler` for the 3×3. Route belt runs with `find_belt_path`/`find_belt_paths` (the UG-aware Dijkstra) and feed the returned placements straight into `place_belts`. Raw `world.set` on the entity channel is only for post-hoc tweaks (e.g. neutralising a tile inside a validation check), not layout.
- **`ppo.py`**: `PpoArgs` (all PPO hyperparams + `start_from`, `critic_warmup`, `eval_every`; defined in `training_config.py`), `FactorioEnv` (`reset`/`step`; terminal reward = `thput_raw / (1 + entity_cost_scale * entity_cost)`, where entity cost is per-output recursively-expanded raw-item quantity + craft time; normalized throughput remains a reference-based evaluation diagnostic and is not used by the reward; `eot` is a real action that ends the episode; `info['kind']` carries the lesson for per-lesson logging), `AgentCNN` (encoder + per-head outputs: tile/entity/direction/item/misc + `critic_head` + `eot_head`; stashes `_last_head_entropy`/`_last_eot_prob` for the policy/* metrics), `_resolve_start_from`/`_resolve_wandb_checkpoint` (checkpoint loading), `_run_signature` (run name), `_build_eval_set`/`_run_greedy_eval` (the eval/ section).
- **`sft.py`**: `run_rollout_eval` returns `RolloutEval` with `overall`/`overall_eot` and per-`LessonKind` breakdowns; checkpoint is selected on the EOT-respecting `val/thput`. PPO reuses this for its `eval/` metrics (lazy import to avoid the ppo↔sft cycle).

## W&B dashboards

Runs are named by a hyperparameter signature, not a timestamp (`ppo.py:_run_signature`, `sft.py:_artifact_name`), e.g. `ppo-s11-lr5e-05-ent0-cw10-fromj0s5y2mc-c93-69-96-seed1`. `global_step` (env steps) is the PPO x-axis. PPO logs once per iteration into these sections (see `define_metric` block in `ppo.py`):

- **`eval/`** — periodic EOT-respecting greedy held-out throughput (`eval/thput`, `eval/{LESSON}/thput`), every `--eval-every` iters; directly overlay-able with the SFT baseline. **This is the headline progress signal**, and the sweep metric (`ci/sweep_ppo.yaml`).
- **`rollout/`** — on-policy sampled episode stats (`thput`, `thput_raw`, `reward`, `length`, `eot_rate`, `invalid_frac`, `num_entities`, `entity_efficiency`, `frac_reachable`, `entity_cost`, `cost_efficiency`) + per-lesson `rollout/{LESSON}/{thput,thput_raw,reward,length,entity_cost,cost_efficiency}` (raw items/s kept alongside the normalized throughput so lessons with very different ceilings stay comparable).
- **`policy/`** — acting-policy distribution: `entropy`, per-head `entropy_{tile,entity,direction,item,misc,eot}`, `eot_prob`.
- **`critic/`** — value-head health (is the critic predicting factory value?), global + per-lesson (`ppo.py:_critic_diagnostics`). Global: `explained_variance` (headline: 1=perfect, 0=predicts-the-mean, <0=worse), `value_rmse`/`value_bias` (error size / signed over-under-estimation, in reward units), `value_return_corr` (ordering skill, robust to scale/offset), `value_std` vs `return_std` (tells a collapsed-to-constant critic apart from a responsive one), `value_mean`/`return_mean`, `adv_abs_mean` (mean |GAE advantage|, a TD-error proxy). Per-lesson `critic/{LESSON}/{explained_variance,value_rmse,value_bias,value_return_corr}` — so a critic that nails belts but is clueless on assemblers shows as a split curve (variance-based fields are NaN when fewer than 2 transitions back that slice).
- **`losses/`** `policy,value,entropy,total,approx_kl,clipfrac,explained_variance` (the last kept for back-compat; `critic/` is the richer view); **`optim/`** `lr,critic_lr,ent_coef,grad_norm,critic_warmup`; **`perf/`** `sps,rollout_seconds,update_seconds,eval_seconds`.

## Throughput metric (`thput`)

The metric comes from a greedy rollout that blanks the **whole** grid and rebuilds from empty (the honest "can it build from scratch" test). Per-factory throughput is `info['thput_normed']` ∈ [0, 1] — raw items/sec ÷ that factory's max, so a perfectly-rebuilt factory scores 1.0 regardless of belt speed. Defined in `sft.py:run_rollout_eval`:

- **`thput`** (`overall_eot`) — build skill *respecting* the EOT head: snapshot the throughput the first time the EOT prob crosses `rollout_eot_threshold` (0.5); if it never fires, fall back to the final throughput. "How good is the factory at the moment the model decides it's done?"

**The RL goal is for PPO's achieved throughput to exceed the SFT base's `val/thput` (≈0.11 for `j0s5y2mc`).** Per-lesson the SFT base is uneven (MOVE_ONE_ITEM ~0.38, assembler lessons ~0).

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

Run a W&B sweep (creates the sweep from `ci/sweep_ppo.yaml`/`ci/sweep_sft.yaml`
at the given commit and drains it on self-terminating RunPod pods; or comment
`/ci sweep ppo` on a PR):

```bash
uv run python -m ci sweep ppo --ref main
```

## GPU training via CI (`/ci` PR comments)

**When asked to "do an SFT/PPO run" or "compare X vs main", use this — never
launch pods by hand.** GPU jobs run on fire-and-forget, self-terminating
RunPod pods, triggered by commenting on a PR (the comment runs the **PR's
head commit**, so the branch must contain the `ci/` directory — merge main
into stale branches first). Full details: `ci/README.md`; `/ci help` posts
the grammar.

The workflow: push a branch, open a PR, then comment one of

```
/ci sft --num-samples 200000              # SFT smoke (~minutes); omit flag for production-sized
/ci ppo --start-from <sft_run_id> --total-timesteps 100000
/ci compare ppo --start-from <sft_run_id> --seeds 3 --total-timesteps 250000
assert pr:rollout/thput > main:rollout/thput
/ci sweep sft --pods 2
/ci pods      /ci kill <pod_id>      /ci watchdog --dry-run
```

`assert` lines under a compare become a `factorion-ci/compare` commit
status: sides `pr:`/`main:`, ops `< > <= >=` plus `==`/`~=` (approx-equal,
`+- tol`). Compare runs land in W&B groups
`cmp-<sha7>-<algo>-<nonce>-{pr,main}`, one pod per side running its seeds
sequentially.

What comes back, and what to relay to the user (always give links):

- **👀 on the comment within ~30s** = event received. No 👀 → GitHub
  delayed/dropped the `issue_comment` event (observed live) — repost.
  **👍** = the command ran to completion.
- A **launch comment** with pod ids (→ RunPod console), pre-assigned W&B
  run URLs, the job spec, and for compares live ⏳/🚀/✅/❌/💀/🗑 statuses
  edited in place every ~2 min. Link the user to this comment plus the
  W&B run/group URLs (`https://wandb.ai/beyarkay/factorion/runs/<id>`).
- **Results as PR comments**: sft/ppo runs get a headline-metrics comment
  from the reporter cron (≤30 min after the run finishes); compares post
  the seed-paired every-metric report + commit status from the waiting
  workflow itself (~2 min after the last run finishes).

Pods cannot leak (EXIT trap, in-pod deadline timer, hourly watchdog reaping
by the deadline encoded in the pod name) — never babysit them; `/ci pods`
to check state. Only whitelisted knobs are overridable (SFT `num_samples`,
PPO `start_from`/`total_timesteps`, compare `seeds`); everything else comes
from `training_config.py` at the launched commit.

## Benchmarks

All speed benchmarks live in **`tests/benchmarks/`** — read
`tests/benchmarks/CLAUDE.md` for the playbook and
`tests/benchmarks/EXPERIMENT_LOG.md` for what's been tried (read it before
re-testing an idea). Two scripts cover everything: `bench_run.sh <kind>` (one
run) and `bench_measure.sh <kind>` (repeat/aggregate → CSV + correctness gate),
with `<kind>` ∈ `ppo-speed` (pure-speed fixed-iteration loop → `results.csv`),
`ppo-quality` (time-to-quality finetune → `quality_results.csv`), `sft` (SFT
training loop → `sft_bench_results.csv`).

Headlines so far: PPO time-to-quality **113 → 36 s (−68%)** — recipe (now the
`PpoArgs` defaults) + fused sampling + `torch.compile(reduce-overhead)` CUDA graphs
+ numpy world-writes. SFT training **88.4 → 22.6 s (−74%)** — GPU-resident data
(both loops) + lazy imports; the conv fwd/backward is then the floor.

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
7. **PPO smoke test**: `CI=1 WANDB_MODE=disabled uv run python ppo.py --seed 1 --env-id factorion/FactorioEnv-v0 --total-timesteps 5000`
8. **SFT smoke test**: `CI=1 WANDB_MODE=disabled uv run python sft.py --seed 1 --size 5 --num-samples 200 --epochs 2 --batch-size 32 --layer1 16 --layer2 16 --layer3 16 --checkpoint-path /tmp/sft_smoke.pt --summary-path /tmp/sft_smoke.json`

All eight must pass before the work is considered complete.

## Code Conventions

- CLI arguments are defined via the `PpoArgs`/`SftArgs` dataclasses (in `training_config.py`) parsed by `tyro`.
- The RL environment follows the Gymnasium API (`reset`, `step`, `render`).
- Factory state is represented as 3D tensors with semantic channels.
- Experiment runs are tracked with Weights & Biases.
- Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`; set up the env with `uv sync`.
- **Comments say *why*, never *what* or *when*.** Never leave a comment describing how the code used to be, or narrating a change you just made ("now does X", "removed Y", "was previously Z") — the diff and git history record that. Don't restate what the code plainly says either. A comment earns its place only by explaining *why* the code is the way it is (a non-obvious constraint, tradeoff, or gotcha), and it should be terse.

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
- **Easy-to-review diffs, minimal churn.** Keep the diff as small as the change requires and no larger. Touch only lines relevant to the task — no changes to unrelated code, no reformatting/renaming/reordering for taste, no "while I'm here" cleanups, and don't reflow whitespace or rewrite comments the change doesn't force. A one-line behaviour change should be a one-line diff, not a one-liner buried in twenty lines of churn (and don't answer a +1/-1 change with a paragraph of new comments). If you spot an unrelated improvement worth making, mention it separately rather than folding it in.
