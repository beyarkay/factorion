# CLAUDE.md

## Project Overview

Factorion trains agents to autonomously design and build factories inspired by the video game Factorio. The agent places entities (transport belts, assembling machines, inserters, splitters, electric poles, …) on a grid to transform input items into desired outputs, optimizing for throughput.

The training pipeline is a **LLM-style two-stage split**:

1. **Data generation** — `build_factory()` in `factorion.py` constructs a known-correct factory (returning `Optional[Factory]`), then `blank_entities()` produces a _(partial-factory, correct-completion)_ training pair by removing N entities from it. Each lesson type (`MOVE_ONE_ITEM`, `SPLITTER_SPLIT`, `SPLITTER_MERGE_SIDELOADED`, …) covers a different entity or layout pattern. The exception is **trial** kinds (`LESSON_IS_TRIAL`): scenarios with no known solution — the built world is only the source/sink markers, so they yield no training targets — the denoiser is only *evaluated* on them, and RL can train on them.
2. **SFT pretraining (denoising)** — `sft.py` trains the whole-factory UNet denoiser in `denoise.py`: each solved factory has a random fraction t ∈ [0, 1] of its free tiles resampled from noise, and the model always predicts the clean factory for every tile in one forward pass. Generation starts from pure noise around the markers, iterates until the prediction reproduces its own input, and keeps the best of 8 chains by simulated throughput. It uploads the best checkpoint as a W&B artifact (type `model`, containing `sft_checkpoint.pt`) so RL can pull it by run id.
3. **RL finetuning** — `rl.py --start-from <run>` refines the pretrained denoiser to push beyond what imitation achieves, currently by RAFT (online rejection-sampling fine-tuning). `--start-from` is a local `.pt` path or a W&B run id, whose model artifact is downloaded automatically. Every build starts from a **fully-blank** grid (only the markers). The base must have been trained on the current architecture (any architecture change makes every older artifact unloadable, so mint a new production SFT run after one lands), and the bar RL must clear is that base's own `val/thput@N` on the same lesson mix.

**Implication when adding features:** new lesson types expand pretraining coverage (more diverse entity/layout patterns the pretrained model understands). They are **not** rungs of a fixed difficulty ladder — diversity matters at every `num_missing_entities` level.

## Tech Stack

- **Language**: Python 3.11+
- **ML Framework**: PyTorch (`torch`)
- **Experiment Tracking**: Weights & Biases (`wandb`)
- **CLI Parsing**: tyro (dataclass-based)
- **Graphs**: networkx (flow-graph visualization only; graph construction and throughput live in the Rust engine)

## Project Structure

- `training_config.py` — Single source of truth for training hyperparameters. Defines `SharedArgs` (defaults common to RL + SFT: grid `size`, W&B project, seed, tags, `eval_best_of`) and the `RlArgs`/`SftArgs` dataclasses that inherit it. `rl.py`/`sft.py` and the CI scripts all read defaults from here.
- `denoise.py` — The whole-factory denoiser. Each tile is three tokens: a type (entity × facing × underground mode, `TYPES`), a recipe, and a part (the tile's offset inside its multi-tile entity). The module holds the noise (`corrupt`, `noise_tokens`) and `DenoiserUNet`, a fully-convolutional UNet with 2-D rotary attention at the bottleneck; it is never told the noise level and runs at any grid size. `legalize` turns a token grid into a valid world by majority vote over multi-tile footprints, `generate` does ancestral denoising plus greedy refinement until the chain reaches a fixed point, and `run_denoise_eval` is the best-of-N generation eval.
- `rl.py` — RL for the denoiser (`RlArgs`): online rejection-sampling fine-tuning (RAFT). Each iteration generates `group_size` builds of each of `groups` fresh factories (blanked to their markers) the way the eval does, at `temperature`, scores them with the reward (`denoise.score_world`), and takes one step of SFT's denoising loss toward each factory's best build, plus `sft_coef` times it toward the iteration's lesson references (without which entity cost climbs and greedy refinement decays). A policy gradient on the same builds is mostly noise (#484). A timestep is one ancestral step of one build. `eval/` is SFT's generation eval on SFT's held-out set, logged at iteration 0 too, so a run's first point is its base.
- `sft.py` — Denoising pretraining. `FactoryStream` (generated factories), `denoise_step` (corruption, an unrolled SUNDAE second step, and the losses: per-tile CE plus the `lw_cost` expected entity cost), `val_denoise_metrics` (accuracy at fixed noise levels), `generation_metrics`, `build_eval_factories` (the held-out set), `train_sft`, and the training helpers `rl.py` shares (`resolve_checkpoint`, `upload_replacing`, `assert_device_ok`). The checkpoint is `{"arch", "model"}`, loaded by `denoise.load_denoiser`.
- `factorion.py` — Environment utilities module: enums (`Channel`, `Direction`, `Entity`, `Item`, `Recipe`, `LessonKind`), blueprint encoding/decoding, factory generation, lesson creation, `apply_placement_action`, factory-graph building (`build_graph_nx`). Import symbols directly (`from factorion import build_factory, blank_entities, Channel, ...`).
- `factorion_rs/` — Rust extension (PyO3/maturin) for the throughput simulation (`simulate_throughput`) and the lesson generator (`build_factory` in `src/factory_gen.rs`, with a fast deterministic RNG in `src/rng.rs`; exposed to Python as `factorion.build_factory`, which is what training calls). Built into the venv via `maturin develop`; stubs in `factorion_rs/__init__.pyi` (keep in sync or `ty` fails).
- `factory_diff.py` — Qualitative diff of two checkpoints: best-of-N generate both over the same held-out factories (`denoise.run_denoise_eval` with `records=[...]`) and render every factory whose throughput disagrees side by side, largest gap first. Run it on any pair (SFT vs RL, RL@2M vs RL@10M) with `uv run python -m ci compare-renders <A> <B>`; `/ci compare` posts the same report as a second PR comment.
- `scripts/factory_builder.py` — Local HTTP UI to hand-build factories and watch a denoiser checkpoint build them: `a` runs one `denoise.generate` step from noise around the markers via the stateless `/denoise` endpoint (the page carries the chain's tokens between calls; holding `a` runs it to its fixed point), and the Scan tab rebuilds N lessons best-of-N. Hand edits go through `factorion.apply_placement_action`. Its **copy YAML** buttons (next to the Graph readout, and on every Scan-seeds card) serialise the on-screen factory via `factory_yaml` into a ready-to-paste `factorion_rs/tests/factories/*.yaml` fixture — the way to turn a factory you can see into a regression test.
- `factorion-mod/` + `factorion_server/` — the Factorio mod and its Python side (read `factorion-mod/README.md`). Two jobs: (1) in-game model inference over RCON (`mod/control.lua` + `factorion_server/server.py`, which generates best-of-N around the region's markers and streams the best build); (2) the **engine-parity harness** (#261): `factorion_server/parity.py` replays `build_factory` factories inside real Factorio via `mod/parity.lua` and compares measured per-sink items/s against `factorion_rs.py_sink_deliveries` — the tool for verifying `simulate_throughput` against the real game (`scripts/parity_launch.sh` for one-command headless runs; `--dry-run` needs no Factorio). To check anything against the real game, use the `factorio` skill: it installs the free headless server on demand. Players get both halves from a release (`RELEASING.md`): the mod from the mod portal and the server as the `factorion-server` wheel built from the root `pyproject.toml` (`uvx factorion-server@X.Y.Z`), which fetches the mod's default checkpoint from the GitHub release.
- `ci/` — **All CI/training automation** (read `ci/README.md`). GPU jobs run on fire-and-forget, self-terminating RunPod pods, triggered by `/ci ...` PR comments (`ci/gh_command.py`) or `uv run python -m ci ...`; results are posted back as PR comments. Job specs in `ci/config.py` are the complete CI override surface (commitish, SFT `num_samples`, RL `start_from`/`total_timesteps`, compare seeds) — every other hyperparameter flows from `training_config.py`. Includes the every-metric compare report + `assert pr:metric > main:metric` gating (`ci/report.py`), the leaked-pod watchdog (`ci/watchdog.py`), and the sweep configs `ci/sweep_rl.yaml`/`ci/sweep_sft.yaml` (metric `eval/thput@8` / `val/thput@8`; results are reported, not auto-applied — defaults in `training_config.py` are edited by hand).
- `.github/workflows/` — thin pointers into `ci/`: `ci.yml` (lint + tests, no GPU), `ci-command.yml` (`/ci` comments), `ci-reporter.yml` (posts run results to PRs), `pod-watchdog.yml` (6-hourly pod reaper), `launch.yml` (manual dispatch), `claude.yml`.
- `factorio-icons/` — Entity icon PNGs.
- `docs/EXPERIMENTS.md` — one line per PR (idea → result, linked) across every experiment tried. **Search it before proposing an experiment**; most obvious ideas have already been run.

### Codebase map (grep these symbols)

- **State tensor** is `(C, W, H)` with `C = len(Channel) = 6` channels: `ENTITIES`, `DIRECTION`, `ITEMS` (recipe/filter), `MISC` (underground up/down), `FOOTPRINT` (1 = buildable), `POWER` (1 = powered tile). `POWER` is derived from the poles purely as model input — the engine recomputes power and never reads it — so anything that moves a pole must `factorion.refresh_power(world)` (`apply_placement_action`, `blank_entities`, `blueprint2world`, `denoise.tokens_to_state`/`legalize` and the generator's `finish` already do).
- **Electric power** (`factorion_rs/src/power.rs`, `wiki/electric-network.md`, rules checked in headless Factorio 2.0): placeable `small_electric_pole` (1x1, 5x5 supply, wire reach 7.5), `medium_electric_pole` (1x1, 7x7, 9) and `substation` (2x2, 18x18, 18), all facing `Direction.NONE` (`factorion.POLE_ENT_IDS`; a pole with a facing is rejected as `pole_with_direction`). Every pair of poles whose centres are within the smaller of their two reaches is wired; power comes from wherever the factory may not build — a network is live iff one of its poles could wire to a 1x1 pole on a tile just outside the grid or on an unbuildable (`FOOTPRINT` 0) tile; a tile is powered iff it lies in a live pole's supply area. Inserters, long-handed inserters and assemblers (`Item::needs_power`) are powered iff any of their tiles is, and `build_graph` drops an unpowered one as if it weren't there; belts, undergrounds, splitters and the source/sink markers need no power, and poles carry no items (no graph node). There is no way to assume power: an unpowered factory doesn't work, everywhere — generators, fixtures, tests and parity. Each generator powers its layout with medium poles before validating it (`factory_gen.rs::place_poles`: greedily the free tile powering the most unpowered entities, relays where none is live; a layout with no room for them is rejected and the builder retries), so poles are ordinary blankable units in every non-trial lesson. Every textual fixture and hand-built test factory carries its own poles (Python tests build theirs with `helpers.power(world)`, a wrapper over the `factorion_rs.py_place_poles` binding). The parity harness reproduces the off-map rule in-game: a ring of `factorion-offmap-pole`s (supply area: their own tile) just outside each grid, so only a factory's own poles power it.
- **Lessons** (`LessonKind` in `factorion.py`, built from `factory_gen.rs::LessonKind` via `factorion_rs.py_lesson_kinds()`): `MOVE_ONE_ITEM`, `SPLITTER_SPLIT`, `SPLITTER_MERGE_SIDELOADED`, `MOVE_VIA_UG_BELT`, `MEMORISE_1_INGREDIENT_RECIPES` … `MEMORISE_4_INGREDIENT_RECIPES`, `MOVE_ONE_ITEM_CHAOS`, `CROSS_UNDER_BELT`, `ROUTE_BELTS`, `ROUTE_BELTS_BUS`, `OPPOSITE_SIDES_1IN`, `SAME_SIDE_1IN`, `SPLITTER_1IN`, `MIRRORED_1IN`, `MIRRORED_2IN`, `REACH_OVER_2IN`, `SHARED_BELT_2IN`, `UG_WEAVE_2IN`, `OPPOSITE_FEEDS_2IN`, `DIRECT_INSERT_2IN`, `INTERMEDIATE_BELT_2IN`, `REACH_OVER_3IN`, `REACH_OVER_4IN`, `TRIAL_RECIPE_TREE_DEPTH_1` … `TRIAL_RECIPE_TREE_DEPTH_3`. SFT and RL sample and evaluate only `factorion.TRAINING_KINDS` (currently `FULL_FACTORY_KINDS`: the belt-routing and full-factory lessons plus the trials; the test suite sets `FACTORION_TRAIN_ALL_KINDS=1` to train on every kind). `build_factory(size, kind, seed, ...)` (Rust-backed) returns `Optional[Factory]` (rejection sampling can fail → `None`); `blank_entities(factory, num_missing_entities, seed)` removes N entity _units_ (multi-tile entities count as one). Every `Factory` carries `max_throughput`, the items/s reference throughput is normalized by (the solved world's simulated rate, or an analytic ceiling for trials). Each lesson has a `build_*` generator in `factory_gen.rs`; `SPLITTER_MERGE_SIDELOADED` (`build_splitter_merge_sideloaded`): two sources are each capped to 7.5 i/s by a **side-load gadget** (a protected empty "decoy" belt forces the source to side-load onto one lane instead of curving to a full 15), then a splitter merges the two 7.5 arms onto one 15 i/s belt to a single sink — so _both_ arms are throughput-necessary (drop one and the sink falls to 7.5). Its decoy belts are a sanctioned exception to the no-orphan invariant (see below). The `MEMORISE_N_INGREDIENT_RECIPES` lessons (`build_memorise_recipes`, one per exact ingredient count 1–4) place a random recipe's assembler fed/drained by `source → belts → inserter → assembler → inserter → belts → sink` arms whose belt count is drawn per arm from `0..=MEMORISE_MAX_ARM_BELTS` (5) and laid along a shortest straight-or-single-L walk, so a zero-belt arm has its marker straight against the inserter and the motif can't be memorised as "always one belt" (one source per ingredient, so lesson `N` has exactly `N` input sources); `CROSS_UNDER_BELT` (`build_cross_under_belt`) is validated by its own throughput/orphan invariants; `ROUTE_BELTS`/`ROUTE_BELTS_BUS` (`build_route_belts`) are belts only — 1–8 sinks, each fed by a source of its item directly or as one output of a splitter splitting one source, plus, off the bus, sometimes sources whose item no sink takes (left unconnected), every route a sampled shortest path that tunnels only where blocked; the bus lines sources up side by side (sinks likewise, in the same order) so its belts run as a tidy bundle. Every `find_belt_paths` route (all lessons) breaks length ties toward fewer turns and hugging the `hug` cells its caller passes (`TURN_COST`/`HUG_DISCOUNT`); `OPPOSITE_SIDES_1IN` (`build_factory_1_ingredient`) packs a row with as many assemblers as the grid fits, sharing an input and an output belt lane (2–3 input and 2–3 output inserters per assembler), with the source/sink at semi-arbitrary cells wired to the lane ends by the UG-aware router — the full row is always throughput-necessary, so a single-assembler build can never match the reference (#410), while throughput still varies (input-inserter / recipe-speed / output-inserter limited) so good and bad layouts score apart; `SAME_SIDE_1IN` (the same generator's `same_side` flag) runs both lanes along one side of the row, the output lane beyond the input lane, reached by long-handed inserters from the machines; `SPLITTER_1IN` (`build_splitter_1in`) feeds the same row from its middle — the source feeds a splitter over the input lane, whose two outputs curve into halves running away from each other; `MIRRORED_1IN`/`MIRRORED_2IN` (`build_mirrored`, ≥ 13 tall) put two rows face to face across one shared input belt (two ingredients: one per lane), their outer output lanes merging through a collector column; `REACH_OVER_2IN`/`SHARED_BELT_2IN`/`UG_WEAVE_2IN` (`build_factory_2_ingredients`, one per `Feed`) are `OPPOSITE_SIDES_1IN`'s two-input siblings (a packed column fed by a near + long-handed-reached far belt, one belt with an ingredient per lane, or one column where the second ingredient tunnels under the first); `OPPOSITE_FEEDS_2IN` (`build_opposite_feeds_2in`) is a packed row fed one ingredient along each face, its long-handed inserters reaching over the second ingredient's lane to an output lane beyond; `DIRECT_INSERT_2IN` (`build_direct_insert_2in`) is the first two-stage lesson — pairs of machines where one crafts an intermediate from a raw source and inserts it straight into the next, which adds a second belt-fed ingredient; `INTERMEDIATE_BELT_2IN` (`build_intermediate_belt_2in`) splits one packed row into intermediate and product machines joined by a belt along its face (both draw from `two_stage_recipes`); `REACH_OVER_3IN`/`REACH_OVER_4IN` (`build_factory_n_ingredients`) pack the same column as `REACH_OVER_2IN` but carry two ingredients per belt — a two-lane near belt tapped by plain inserters and a far belt reached over it by long-handed inserters; the `TRIAL_RECIPE_TREE_DEPTH_N` **trials** (`build_recipe_tree_trial`; flagged by `LessonKind::is_trial` / Python `LESSON_IS_TRIAL`) place ONLY markers, always on the grid edge working inward (sources face their wall's inward normal, the sink faces outward so it pulls from its interior neighbour) — a random craftable sink item (any assembler tier) plus one source per item of a randomly-expanded frontier of its ingredient tree, where the deepest expanded chain is exactly `N` crafting stages — with a reserved free working cell per marker; `total_entities == 0` and `max_throughput` is the sink recipe's single-assembler output rate.
- **Underground lessons**: `MOVE_VIA_UG_BELT` forces underground belts with an `UNAVAILABLE`-footprint wall; `CROSS_UNDER_BELT` gives a real, protected belt line (the "obstruction" — a winding edge-to-edge cut) the crossing must tunnel _under_ via `factory_gen.rs::ug_aware_belt_path` (a Dijkstra that may dip under blocked cells; minimal tunnel span, no 180° reversals or tile reuse, source→UG-down / UG-up→sink shortcuts).
- **No-orphan invariant**: no lesson's solved factory may contain orphan tiles — throughput must report `unreachable == 0` (checked by `factory_gen.rs::tests::test_no_orphan_tiles_every_lesson` across every non-trial `LessonKind`; a trial's world is only the unconnected markers). Unreachability is judged per ENTITY (an entity is an orphan iff none of its lane nodes lie on a source→sink path) since e.g. an inserter-fed belt legitimately has one forever-empty lane. The **one opt-out** is `LessonKind::allows_orphans()` (exposed to Python as `factorion_rs.py_lesson_allows_orphans()`): `SPLITTER_MERGE_SIDELOADED` returns `true` because its side-load decoy belts are intentional protected orphans, and `ROUTE_BELTS`/`ROUTE_BELTS_BUS` because they leave sources whose item no sink takes unconnected; the test then requires every orphan to be a protected tile or such a source (`unreachable == protected_positions.len() + unmatched sources`), so no _accidental_ orphan slips through.
- **Jargon**: the orientation-relative sides of a directional entity are **left, right, fore, aft** (a north-facing belt's left side is west, its fore side north). Use these consistently in code comments and docs.
- **Dual-lane belts** (`wiki/two-lane-system.md`, #65): graph nodes are NOT 1:1 with entities — belt-ish tiles (belt, UG, each splitter tile) emit one node per `Lane` (`NodeId.lane`), each capped at half the tile's `Item::flow_rate()` (belt tile 15 → 7.5/lane; a splitter is two belt tiles, so `flow_rate(Splitter)` is 15 per tile, 4 lane pools × 7.5 = 30 total); inserter/assembler/source/sink stay single-node. Node references use ONE canonical format everywhere (engine labels, `py_build_graph`, and fixture `graph:` blocks): `<entity_char>@x,y` plus `:L`/`:R` for lane nodes (grid-registry chars: `b@1,0:L`, `i@0,1`, `d`/`u`, `S`, `K`); fixture graph blocks may elide lane markers per document (entity-level, lane-projected comparison). All connection logic funnels through `entities.rs::calc_lane_aware_edges`: straight/curve/tunnel/splitter flow is lane-preserving; a perpendicular feed onto a belt with any other belt-connectable input sideloads onto the near-side lane (`belt_feeders`/`is_curved_belt` — lone side feed = curve; UG mouths and splitters never curve; a side feed onto a UG tile connects only the feeder lane over its surface half — aft of an entrance, fore of an exit); inserters drop onto exactly one lane (far when perpendicular, belt's right when in-line) but pick up from _both_, splitting their rate evenly over the lanes they reach and then over the items on each lane in proportion to their rates (`throughput.rs::pickup_input`, max-min fair across lanes so a lane with nothing to give leaves its share to the other) — so an inserter cannot unmix a belt, and no item it can reach starves another.
- **Rendering factories to ASCII** — when asked to "render"/"show"/"draw" factories, use the real renderer, never hand-roll one: `from factorion import render_factory; print(render_factory(factory))` (accepts a `Factory` or a `(C, W, H)` world tensor; Rust binding `factorion_rs.render_factory(world_WHC)` lives in `factorion_rs/src/render.rs`). It returns the two-char grid format the textual fixtures use — `b`=belt, `i`=inserter, `l`=long-handed inserter, `a`=assembler box, `Y`=splitter, `d`/`u`=underground down/up, `S`=source, `K`=sink, `e`/`E`=small/medium pole, `Z`=substation box, each plus a facing marker `^>v<` (`.` for none); `..` is an empty tile. To also show each factory's recipe, read the assembler's tagged item from the `ITEMS` channel (or the sink's carried item).
- **Placing entities in a generator** (`factory_gen.rs`) — mutate the `World` through the existing `place_*` helpers, never hand-write `world.set(Channel::Entities/Direction, …)` for a belt/marker/splitter (same rule as the renderer above: don't re-implement a helper that exists). `place_belts(world, &[(x, y, dir, Misc)])` lays plain belts **and** underground tunnel ends, one tuple per tile; `place_marker` for sources/sinks, `place_splitter` for the 2-tile splitter, `place_assembler` for the 3×3. Route belt runs with `find_belt_path`/`find_belt_paths` (the UG-aware Dijkstra) and feed the returned placements straight into `place_belts`. Raw `world.set` on the entity channel is only for post-hoc tweaks (e.g. neutralising a tile inside a validation check), not layout.
- **`factorion.apply_placement_action`** — the one placement path for the builder UI and the mod server: placements **overwrite** — every entity the footprint touches is removed whole first (unit grouping is `graph.rs::unit_anchors`, via `factorion_rs.py_unit_tiles`), so `empty` deletes and anything else replaces; sources/sinks and walls can't be edited, and a lesson's `protected_positions` are generated as `UNAVAILABLE` footprint so they're walls too. Returns `None` or the rejection reason.
- **Reward** (`denoise.score_world`, what RL maximises): `score = thput_raw * cost_efficiency / max_throughput`, with `cost_efficiency = 1 / (1 + entity_cost_scale * entity_cost)` and entity cost = per-output recursively-expanded raw-item quantity + craft time (`denoise.entity_cost`). Entity cost enters ONLY as a multiplicative discount on achieved throughput — at `thput_raw = 0` the score is 0 no matter what is on the grid; there is no standalone cost penalty. The score is uncapped above 1: out-producing the reference `max_throughput` keeps paying (#426).
- **Generation eval** (`denoise.run_denoise_eval`, via `sft.generation_metrics`; SFT's `val/`, RL's `eval/`): every factory of `sft.build_eval_factories` (`eval_seeds_per_kind` per training kind, trials included, from seeds disjoint from training) is blanked to its markers, gets `eval_best_of` chains generated from noise, and scores its best legalized chain (ties: lower `entity_cost`, then lower chain index); `thput@1` reads the first chain. Also logged: `converged_frac@N` (chains that reached a fixed point), `refine_steps@N`, and SFT's `val/t{level}/{type,recipe,part}_acc` and `val/identity` (a clean factory left untouched). SFT checkpoints are selected on `val/thput@8`, RL's on `eval/thput@8`.

## W&B dashboards

Runs are named by a hyperparameter signature, not a timestamp (`rl.py:_signature`, `sft.py:_artifact_name`), e.g. `raft-s15-lr0.00025-g64x32-t1.5-sft0.3-fromdr17bmik-seed1`. `global_step` (ancestral denoising steps of builds) is the RL x-axis; `train/samples_seen` is SFT's. RL logs once per iteration into these sections:

- **`eval/`** — the generation eval (`eval/thput@N`, `eval/{LESSON}/thput@N`; every metric from it, here and in SFT's `val/`, is logged as `<key>@N`, N = `eval_best_of`, and an unsuffixed metric is best-of-1), every `--eval-every` iterations and at iteration 0, so a run's first point is its base's own `val/`. **This is the headline progress signal**, and the sweep metric (`ci/sweep_rl.yaml`). The same eval also logs `eval/inserters_no_input@N` / `eval/inserters_no_output@N` (+ `eval/trial_*`; SFT: `val/…`): raw counts of inserters nothing feeds / that feed nothing across the eval set (`factorion.count_dangling_inserters`), which should be zero — the belt-stops-beside-the-inserter and inserter-feeds-inserter failures, visible without rendering factories. It also logs `eval/unpowered@N` (+ `eval/trial_unpowered@N`; SFT: `val/…`): inserters and assemblers no pole powers in the selected builds (`factorion.count_unpowered`), which should also be zero.
- **`rollout/`** — the iteration's training builds: `thput`, `thput@G` (best of each group of `group_size`), `reward`, `entity_cost`, overall and per lesson (`rollout/{LESSON}/…`).
- **Lessons vs trials are pooled separately.** The `eval/thput@N` and `rollout/*` aggregates cover **lessons only**; trials go to `eval/trial_thput@N` and the parallel `rollout/trial_*` block. This keeps the headline comparable to the SFT baseline. Per-lesson `{eval,rollout}/{TRIAL_*}/…` keys are logged as usual, so a trial's own curve is always visible. **`eval/trial_thput@N` is the metric that actually matters** — it scores building a factory from nothing but the markers, which is the real problem; every lesson is a proxy for it. Expect it to stay low for a long time (#339).
- **`losses/`** `raft` (toward the best builds), `sft` (toward the lesson references); **`optim/`** `lr,grad_norm`; **`perf/`** `sps,rollout_seconds,update_seconds,eval_seconds`.

## Throughput metric (`thput`)

Per-factory throughput is raw items/sec ÷ that factory's `Factory.max_throughput`, so a perfectly-rebuilt factory scores 1.0 regardless of belt speed. Neither it nor the reward is clipped: the references are not optimal, so out-producing one scores above 1. The denominator is supplied by the generator: the solved world's simulated rate for lessons, an analytic single-assembler ceiling for trials (which have no reference solution to simulate). `thput@N` is the generation eval's mean over held-out lessons of each factory's best-of-N build; `trial_thput@N` the same over trials.

**The RL goal is for RL's `eval/thput@N` to exceed its own SFT base's `val/thput@N` on the same lesson mix.** The longer-run goal is `eval/trial_thput@N` climbing well above zero — nothing imitative can produce it.

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
uv run python sft.py --track --num-samples 2000000
uv run python rl.py --track --start-from <sft_run_id> --total-timesteps 3000000
```

Run a W&B sweep (creates the sweep from `ci/sweep_rl.yaml`/`ci/sweep_sft.yaml`
at the given commit and drains it on self-terminating RunPod pods; or comment
`/ci sweep rl` on a PR):

```bash
uv run python -m ci sweep rl --ref main
```

## GPU training via CI (`/ci` PR comments)

**When asked to "do an SFT/RL run" or "compare X vs main", use this — never
launch pods by hand.** GPU jobs run on fire-and-forget, self-terminating
RunPod pods, triggered by commenting on a PR (the comment runs the **PR's
head commit**, so the branch must contain the `ci/` directory — merge main
into stale branches first). Full details: `ci/README.md`; `/ci help` posts
the grammar.

The workflow: push a branch, open a PR, then comment one of

```
/ci sft --num-samples 200000              # SFT smoke (~minutes); omit flag for production-sized
/ci rl --start-from <sft_run_id> --total-timesteps 100000
/ci compare rl --start-from <sft_run_id> --seeds 3 --total-timesteps 250000
assert pr:eval/thput@8 > main:eval/thput@8
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
- **Results as PR comments**: sft/rl runs get a headline-metrics comment
  from the reporter cron (≤30 min after the run finishes); compares post
  the seed-paired every-metric report + commit status from the waiting
  workflow itself (~2 min after the last run finishes).

Pods cannot leak (EXIT trap, in-pod deadline timer, hourly watchdog reaping
by the deadline encoded in the pod name) — never babysit them; `/ci pods`
to check state. Only whitelisted knobs are overridable (SFT `num_samples`,
RL `start_from`/`total_timesteps`, compare `seeds`); everything else comes
from `training_config.py` at the launched commit.

### Reading a compare

Judge a compare on the **difference curve**, never on endpoint values or the
report table alone. Pull both sides' full history of the metric from W&B,
interpolate onto a common x-axis (`train/samples_seen` for SFT, `global_step`
for RL), take PR − main, smooth it (rolling window ≈ 10% of the run), and
report over the second half of the run: the mean smoothed difference, the
fraction of points where the raw difference is positive, and the difference's
own point-to-point std. A gap that keeps one sign at ≳ 90% of points and
clears the diff's std is real even at one seed per side; one that flips sign
or decays toward zero is not. Say whether the gap is growing, steady, or
closing. Endpoint deltas understate a steady gap by about half and invert small
per-lesson ones; seed std measured at another sample budget or architecture
does not transfer.

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

0. ASSERT that the code being merged should be

- [ ] minimal, lean, trimmed
- [ ] not have excessive comments or tests,
- [ ] not have excessive helpers,
- [ ] not have functions that could be inlined,
- [ ] should contain _ZERO_ references to previous states of the codebase

1. **Rust format + lint**: `cd factorion_rs && cargo fmt && cargo clippy --all-targets -- -D warnings && cd ..`
2. **Rust tests**: `cd factorion_rs && cargo test && cd ..`
3. **Build the Rust extension**: `cd factorion_rs && maturin develop --release && cd ..`
4. **Python tests**: `WANDB_MODE=disabled WANDB_DISABLED=true uv run python -m pytest tests/ -v`
5. **Python linter**: `uv run ruff check .`
6. **Type checker**: `uv run ty check .`

All 7 must pass before the work is considered complete.

## Code Conventions

- CLI arguments are defined via the `RlArgs`/`SftArgs` dataclasses (in `training_config.py`) parsed by `tyro`.
- Factory state is represented as 3D tensors with semantic channels.
- Experiment runs are tracked with Weights & Biases.
- Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`; set up the env with `uv sync`.
- **Comments say _why_, never _what_ or _when_.** Never leave a comment describing how the code used to be, or narrating a change you just made ("now does X", "removed Y", "was previously Z") — the diff and git history record that. Don't restate what the code plainly says either. A comment earns its place only by explaining _why_ the code is the way it is (a non-obvious constraint, tradeoff, or gotcha), and it should be terse.

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

- New code must be accompanied by end-to-end tests.
- **Small, incremental commits.** Make each commit as small as possible while still being a logical, self-contained unit of change (e.g., "expand entity head" is one commit, "sample all lesson kinds" is another, "update tests for new return shape" is another). Do not bundle unrelated changes. Each commit should pass the pre-completion checklist on its own.
- **Easy-to-review diffs, minimal churn.** Keep the diff as small as the change requires and no larger. Touch only lines relevant to the task — no changes to unrelated code, no reformatting/renaming/reordering for taste, no "while I'm here" cleanups, and don't reflow whitespace or rewrite comments the change doesn't force. A one-line behaviour change should be a one-line diff, not a one-liner buried in twenty lines of churn (and don't answer a +1/-1 change with a paragraph of new comments). If you spot an unrelated improvement worth making, mention it separately rather than folding it in.
