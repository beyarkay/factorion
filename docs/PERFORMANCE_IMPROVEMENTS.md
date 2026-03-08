# Performance Improvements

This document catalogues ideas for improving training wall-clock time. The core
problem is **GPU starvation**: the model is tiny (~135K params) and GPU batch
sizes are microscopic. With `num_envs=16`, every inference call during rollout
processes a batch of 16 samples — an A100 can handle thousands. The GPU is idle
95%+ of the time, waiting for the CPU to step 16 serial Python environments.

The bottleneck is CPU-bound env stepping, not GPU compute.

Ideas are ordered by estimated impact.

---

## Table of Contents

- [High-impact ideas](#high-impact-ideas)
  - [H1: Increase num_envs (64-256+)](#h1-increase-num_envs-64-256)
  - [H2: AsyncVectorEnv](#h2-asyncvectorenv)
  - [H3: Move the entire env to Rust (vectorized)](#h3-move-the-entire-env-to-rust-vectorized)
  - [H4: Reduce num_minibatches / increase minibatch size](#h4-reduce-num_minibatches--increase-minibatch-size)
  - [H5: Mixed precision (bf16 on A100)](#h5-mixed-precision-bf16-on-a100)
- [Medium-impact ideas](#medium-impact-ideas)
  - [M1: Action masking for invalid actions](#m1-action-masking-for-invalid-actions)
  - [M2: Move video recording out of the training loop](#m2-move-video-recording-out-of-the-training-loop)
  - [M3: Reduce update_epochs from 8 to 4](#m3-reduce-update_epochs-from-8-to-4)
  - [M4: Pre-allocate numpy buffer for Rust calls](#m4-pre-allocate-numpy-buffer-for-rust-calls)
- [Recommended implementation order](#recommended-implementation-order)

---

## High-impact ideas

### H1: Increase `num_envs` (64-256+)

**Estimated impact:** High | **Complexity:** Zero (CLI arg change)

The single biggest lever. Current defaults: `num_envs=16`, `batch_size=4096`,
`minibatch_size=128`.

Benefits of more envs:

- **Larger GPU batches during rollout**: inference batch goes from 16 to 256,
  much better GPU utilisation.
- **More episodes per iteration**: faster curriculum signal accumulation. The
  `moving_average_length=500` throughput buffer fills faster, so the agent
  levels up sooner.
- **Larger batch_size**: `256 * 256 = 65K` batch gives smoother gradients.

With the model this small (~135K params), GPU memory is not the constraint —
even 512 envs would fit easily on an A100 80GB.

The catch: with `SyncVectorEnv`, more envs means proportionally more serial CPU
time per rollout step. This motivates H2.

**How to test:**

```bash
python ppo.py --num-envs 128 --num-minibatches 8 --total-timesteps 500000
```

### H2: AsyncVectorEnv

**Estimated impact:** High | **Complexity:** Easy (one-line change + reset fix)

`SyncVectorEnv` (`ppo.py:940`) calls each env's `step()` serially, so all Rust
`simulate_throughput` calls and Python reward computation happen sequentially.
`AsyncVectorEnv` runs each env in its own subprocess, parallelising all per-env
work across CPU cores.

Combined with H1 (say 128 envs), you'd have 128 subprocesses stepping in
parallel. On a machine with 32+ cores, this gives ~16-32x faster env stepping.

```python
# Current (ppo.py:940):
envs = gym.vector.SyncVectorEnv([...])

# Proposed:
envs = gym.vector.AsyncVectorEnv([...])
```

**Implementation notes:**

- Gymnasium's `AsyncVectorEnv` uses subprocesses by default. No Rust changes
  needed — each subprocess loads its own copy of the extension.
- The manual per-env reset loop at `ppo.py:1068-1072` needs reworking.
  `AsyncVectorEnv` handles autoreset internally; you can't call
  `envs.envs[idx].reset()` across a process boundary. Instead, pass
  `num_missing_entities` via the env's `options` dict and let the autoreset
  mechanism handle it.
- IPC serialization overhead is the risk. At 8x8 grids each observation is
  ~256 floats, so serialisation cost should be negligible. Worth a quick
  benchmark to confirm.

Also documented as P5 in `docs/EXPERIMENTS.md`.

### H3: Move the entire env to Rust (vectorized)

**Estimated impact:** Very high | **Complexity:** High
| **Status:** Feasibility confirmed (PyO3 proof-of-concept built and tested)

The highest-ceiling optimization. Currently each `step()` call involves:

1. Python dict unpacking and 8+ validity checks in Python
2. Torch tensor mutations (`_world_CWH[channel, x, y] = value`)
3. `.permute(1, 2, 0).to(torch.int64).numpy()` conversion per step (`ppo.py:450`)
4. Rust FFI call to `simulate_throughput`
5. Python reward computation with multiple tensor operations
6. Solution match computation (`_compute_solution_match`)
7. Python info dict construction (~20 keys)

With AsyncVectorEnv (H2), each subprocess still runs all this Python. The
subprocess IPC overhead (serializing obs/actions/rewards) adds further latency.

A Rust vectorized env eliminates all of it:

```python
# Python side (ppo.py) — replaces AsyncVectorEnv entirely:
env = factorion_rs.VecFactorioEnv(num_envs=256, size=8, max_steps=16)
obs = env.reset(seed=1)                 # returns numpy (256, 5, 8, 8)
obs, rewards, dones, infos = env.step(actions)  # actions: numpy (256, 6)
env.set_max_missing(3)                  # curriculum update, instant
```

| Overhead | Current (Python + AsyncVectorEnv) | After H3 (Rust VecEnv) |
|---|---|---|
| Per-step logic | Python validity checks, reward math, dict building | Rust, compiled, zero-overhead |
| Tensor ↔ numpy | `.permute().to().numpy()` per step per env | Not needed — Rust writes directly to pre-allocated buffers |
| Subprocess IPC | Serialize/deserialize obs+actions per step | None — single process |
| GIL | Held during all Python env logic | Released via `py.allow_threads()` |
| `simulate_throughput` FFI | One cross-boundary call per env per step | Internal Rust function call (inlined) |
| Memory allocation | New numpy array per env per step | Pre-allocated, reused across steps |
| Parallelism | OS-level subprocess scheduling | rayon thread pool, work-stealing, no IPC |

#### Feasibility (confirmed)

A proof-of-concept `VecEnvProof` was built and tested: a `#[pyclass]` Rust
struct holding N flat arrays, with `reset()` and `step()` methods returning
numpy arrays via PyO3. Confirmed working:

- `#[pyclass]` struct holds mutable env state (`Vec<f32>`, `Vec<i32>`)
- `reset()` → `(N, C, W, H)` numpy array via `PyArray1::from_vec().reshape()`
- `step()` → `(obs, rewards, dones)` tuple of numpy arrays
- Returned arrays are regular mutable numpy — `torch.as_tensor()` works
- Pattern: ~80 lines of Rust for the full PyO3 plumbing

#### What already exists in `factorion_rs`

The crate already has substantial infrastructure to build on:

| Component | Status | Location |
|---|---|---|
| `World` struct (flat `Vec<i64>`, WHC layout) | Done | `world.rs` |
| `World::from_numpy()` | Done | `world.rs` |
| `Channel`, `Direction`, `EntityKind`, `Item`, `Misc` enums | Done | `types.rs` |
| `Direction::delta()`, `Direction::opposite()` | Done | `types.rs` |
| `EntityKind::flow_rate()`, `EntityKind::name()` | Done | `types.rs` |
| `build_graph()` (world → factory graph) | Done | `graph.rs` |
| `calc_throughput()` (graph → throughput) | Done | `throughput.rs` |
| Entity connection logic (belt, inserter, underground) | Done | `entities.rs` |
| Recipes (`get_recipe`) | Done | `types.rs` |
| `simulate_throughput()` PyO3 binding | Done | `lib.rs` |

#### What needs to be built

##### Phase 1: `generate_lesson` (~250 lines of Python → Rust)

Port `generate_lesson` and `find_belt_paths_with_source_sink_orient` from
`factorion.py:1314-1569`. This is the hardest piece:

1. **Random source/sink placement** — sample two distinct positions on the
   grid, assign random directions. Uses `random.seed(seed)` for
   determinism; Rust equivalent: seed a `rand::rngs::StdRng`.

2. **BFS pathfinding** — `find_belt_paths_with_source_sink_orient` does a
   multi-path BFS from source front to sink back, finding all shortest
   belt paths. ~100 lines, straightforward BFS with parent tracking and
   backtracking. Standard `VecDeque` in Rust.

3. **Path selection + belt placement** — choose a random path, place
   transport belts with correct directions along it.

4. **Entity removal for curriculum** — randomly remove N belts from the
   placed path to create the puzzle.

**Determinism requirement:** Same seed must produce identical worlds from
both Python and Rust. This enables parity tests: generate worlds from both
implementations and assert equality.

**Testing strategy:** For each seed in 0..100, generate a lesson from Python
and from Rust, assert the world tensors are identical. This catches any
divergence in RNG sequences, BFS ordering, or path selection.

##### Phase 2: `step()` + `reset()` (~200 lines of Python → Rust)

Port `FactorioEnv.step()` from `ppo.py:351-589`:

1. **Action validation** — 8 checks (masked tile, source/sink, entity+direction
   combos, bounds). All simple integer comparisons.

2. **World mutation** — set 4 channels at `(x, y)`. One array write per channel.

3. **Throughput computation** — call existing `build_graph()` +
   `calc_throughput()` directly (no FFI boundary).

4. **Reward computation** — `frac_reachable`, `material_cost`,
   `final_dir_reward`, PBRS deltas (`_compute_solution_match`). All
   arithmetic over the world array.

5. **Termination logic** — `throughput >= 1.0` or `steps > max_steps`.

Port `FactorioEnv.reset()` from `ppo.py:308-349`:

1. Call `generate_lesson` twice (solved + incomplete).
2. Initialize step counter, prev_match, cum_reward.

##### Phase 3: `VecFactorioEnv` wrapper + rayon parallelism

```rust
// factorion_rs/src/vec_env.rs

use rayon::prelude::*;

#[pyclass]
pub struct VecFactorioEnv {
    num_envs: usize,
    size: usize,
    max_steps: usize,
    channels: usize,

    // Per-env state, contiguous in memory
    worlds: Vec<i64>,            // (N * C * W * H) flat
    solved_worlds: Vec<i64>,     // (N * C * W * H) flat
    steps: Vec<usize>,           // (N,)
    seeds: Vec<u64>,             // (N,)
    max_missing: Vec<usize>,     // (N,)
    prev_match: Vec<[f64; 3]>,   // (N,) for PBRS deltas
    cum_rewards: Vec<f64>,       // (N,)

    // Pre-allocated output buffers (reused across steps)
    obs_buf: Vec<f32>,           // (N * C * W * H) flat
    reward_buf: Vec<f32>,        // (N,)
    done_buf: Vec<bool>,         // (N,)
}

#[pymethods]
impl VecFactorioEnv {
    #[new]
    fn new(num_envs: usize, size: usize, max_steps: usize) -> Self { ... }

    fn reset(&mut self, py: Python, seed: u64) -> PyArray4<f32> {
        // For each env (parallelised with rayon):
        //   1. generate_lesson(solved, seed+i)
        //   2. generate_lesson(incomplete, seed+i)
        //   3. init step counter, prev_match
        // Copy world data into obs_buf, return as numpy
    }

    fn step(
        &mut self,
        py: Python,
        actions: PyReadonlyArray2<i64>,  // (N, 6)
    ) -> (PyArray4<f32>, PyArray1<f32>, PyArray1<bool>, ...) {
        py.allow_threads(|| {
            // Release the GIL for the entire batch computation
            self.worlds
                .par_chunks_mut(self.env_size())
                .zip(self.reward_buf.par_iter_mut())
                .zip(self.done_buf.par_iter_mut())
                .enumerate()
                .for_each(|(i, ((world, reward), done))| {
                    // 1. Validate action
                    // 2. Mutate world
                    // 3. build_graph + calc_throughput (already Rust)
                    // 4. Compute reward, PBRS deltas
                    // 5. Check termination, auto-reset if done
                });
        });
        // Return pre-allocated buffers as numpy views
    }

    fn set_max_missing(&mut self, value: usize) {
        self.max_missing.fill(value);
    }
}
```

Key design decisions:

- **Pre-allocated buffers**: `obs_buf`, `reward_buf`, `done_buf` are allocated
  once in `new()` and reused every step. No allocation in the hot loop.
- **`py.allow_threads()`**: Releases the GIL during the parallel computation.
  Python can't call into the env while it's stepping, but that's fine — the
  training loop is sequential (step → inference → step).
- **rayon `par_chunks_mut`**: Each env's world slice is processed in parallel
  with work-stealing. No manual thread management.
- **Observation layout**: Store as `(N, C, W, H)` in the Rust buffer, matching
  PyTorch's expected layout. No transpose needed.

##### Phase 4: Integration with `ppo.py`

Replace the gym/AsyncVectorEnv setup with the Rust env:

```python
# Before (current):
envs = gym.vector.AsyncVectorEnv([make_env(...) for i in range(N)])
obs, info = envs.reset(seed=seed, options={...})
obs, reward, term, trunc, infos = envs.step(action_dict)

# After:
env = factorion_rs.VecFactorioEnv(num_envs=N, size=8, max_steps=16)
obs = env.reset(seed=seed)
obs, rewards, dones, infos = env.step(actions_array)
```

Changes to `ppo.py`:

- Remove `FactorioEnv` class entirely (or keep for testing/rendering only)
- Remove `make_env`, `_ensure_env_registered`, gym registration
- Remove `gym.vector.AsyncVectorEnv` setup
- Change action format from dict to flat `(N, 6)` int64 array
- Episode statistics tracking moves to Rust (or a thin Python wrapper
  that reads from the info arrays)
- `AgentCNN.__init__` takes explicit `width`, `height`, `channels` args
  instead of reading from `envs.single_observation_space`
- Video rendering kept in Python (infrequent, uses a separate
  `SyncVectorEnv` with the Python `FactorioEnv`)

##### Phase 5: Info dict handling

The current `step()` returns ~20 info keys per env. Options:

**Option A (simple):** Return a flat numpy array of info values per env, with
a fixed schema. Python unpacks by index:
```python
obs, rewards, dones, info_array = env.step(actions)
# info_array shape: (N, 20) — throughput, frac_reachable, etc.
throughputs = info_array[:, 0]
```

**Option B (structured):** Return a Python dict of numpy arrays:
```python
{"throughput": np.array([...]), "frac_reachable": np.array([...]), ...}
```

Option A is faster (no dict construction in the hot loop). Option B is more
readable. Start with Option A, add convenience accessors if needed.

#### Testing strategy

1. **`generate_lesson` parity**: For seeds 0..100, generate worlds from both
   Python and Rust, assert tensors are identical.

2. **`step()` parity**: For seeds 0..20, reset both Python and Rust envs with
   same seed, step with same random actions for 16 steps, assert observations,
   rewards, and dones match at every step.

3. **`VecFactorioEnv` consistency**: Reset with N=64, step 1000 times, verify
   no panics, all rewards finite, all obs in valid range.

4. **Determinism**: Same seed → same trajectory, verified across 10 runs.

5. **Benchmark**: Compare SPS between AsyncVectorEnv (H2) and VecFactorioEnv
   (H3) at N=16, 64, 256 envs.

#### Dependencies

- `rayon` — parallel iterators (add to `[dependencies]` in `Cargo.toml`)
- `rand` — seeded RNG for `generate_lesson` (add to `[dependencies]`)
- Existing: `pyo3`, `numpy`

#### Implementation order

1. Port `generate_lesson` + BFS pathfinder to Rust, with parity tests
2. Port `step()` + `reset()` to a single-env `FactorioEnv` Rust struct, with
   parity tests
3. Wrap in `VecFactorioEnv` with rayon, expose via PyO3
4. Update `ppo.py` to use the Rust env
5. Benchmark and profile

### H4: Reduce `num_minibatches` / increase minibatch size

**Estimated impact:** High (with H1) | **Complexity:** Zero (CLI arg change)

Current: `num_minibatches=32`, `batch_size=4096` → `minibatch_size=128`.

A minibatch of 128 is tiny for GPU — kernel launch overhead dominates. Options:

| `num_envs` | `batch_size` | `num_minibatches` | `minibatch_size` | Notes |
|---|---|---|---|---|
| 16 | 4,096 | 32 | 128 | Current (too small) |
| 16 | 4,096 | 4 | 1,024 | Quick test, no env change |
| 128 | 32,768 | 8 | 4,096 | With H1 (good GPU payload) |
| 256 | 65,536 | 8 | 8,192 | With H1 (great GPU payload) |

Even without changing `num_envs`, dropping `num_minibatches` from 32 to 4 gives
8x larger minibatches and fewer kernel launches per epoch.

### H5: Mixed precision (bf16 on A100)

**Estimated impact:** Medium (after H1/H4) | **Complexity:** Easy

A100s have massive bf16 throughput (312 TFLOPS bf16 vs 19.5 TFLOPS fp32). The
model currently uses fp32 everywhere.

```python
scaler = torch.amp.GradScaler('cuda')

# During rollout inference:
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    action_ED, logprobs_E, _entropy_E, value_E = agent.get_action_and_value(next_obs_ECWH)

# During PPO update:
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    _action_BA, newlogprobs_B, entropy_B, newvalue_B = agent.get_action_and_value(
        obs_B[idxs], actions_B.long()[idxs]
    )
    # ... loss computation ...

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
scaler.step(optimizer)
scaler.update()
```

Won't help much at current batch sizes (GPU isn't the bottleneck), but becomes
meaningful once `num_envs` and `minibatch_size` are increased via H1/H4.

Note: bf16 doesn't need `GradScaler` on A100 (bf16 has the same exponent range
as fp32). You can simplify to just `torch.amp.autocast` without the scaler.

---

## Medium-impact ideas

### M1: Action masking for invalid actions

**Estimated impact:** Medium-high (sample efficiency) | **Complexity:** Medium

330 of 640 effective actions (51.6%) are invalid with the current
`num_entities=2` action space. Invalid actions are silently rejected
(`ppo.py:384-425`) — they don't modify the world, waste an entire step, and
produce zero PBRS delta.

With masking, every step modifies the world, so every step produces a non-zero
PBRS delta signal. This roughly doubles the effective reward signal density.

This is primarily a **sample efficiency** improvement (learning faster per
timestep) rather than a wall-clock speed improvement, but the two compound:
faster learning per step means fewer total steps needed.

Set invalid action logits to `-inf` before the softmax:

```python
# In get_action_and_value, after computing tile_logits:
tile_mask = get_tile_mask(x_BCWH)  # True for valid tiles
tile_logits_BN[~tile_mask.reshape(B, -1)] = float('-inf')

# After selecting a tile and computing entity logits:
ent_mask = get_entity_mask(...)  # True for valid entities
logits_e_BE[~ent_mask] = float('-inf')
```

Masks must be applied during both rollout and PPO update (otherwise the
importance sampling ratio `pi_new / pi_old` is computed over different
distributions).

Also documented as P2 in `docs/EXPERIMENTS.md`.

### M2: Move video recording out of the training loop

**Estimated impact:** Medium | **Complexity:** Easy

Lines `ppo.py:1262-1319`: every 50 iterations, training blocks while 5 new
envs are created, stepped 10 times with rendering, saved as PNGs, and encoded
to MP4 via ffmpeg. This is synchronous.

Options (pick one):

- **Background thread/process:** Spawn the recording in a separate process so
  training continues immediately.
- **Reduce frequency:** Record every 200 iterations instead of 50.
- **Final-only:** Only record on the last iteration.
- **Skip entirely during benchmarks:** Add a `--no-video` flag.

### M3: Reduce `update_epochs` from 8 to 4

**Estimated impact:** Medium | **Complexity:** Zero (CLI arg change)

Each update epoch is a full forward+backward pass over the entire batch. 8
epochs means the backward pass runs `8 * num_minibatches` times per iteration.

With larger batches (H1/H4), each epoch already sees more data, so fewer epochs
are needed for the same gradient quality. PPO typically uses 3-10 epochs; 4 is
a common choice.

This halves the time spent in the PPO update phase. Must validate empirically
that throughput doesn't regress — fewer epochs means less policy refinement per
batch of experience.

```bash
python ppo.py --update-epochs 4
```

### M4: Pre-allocate numpy buffer for Rust calls

**Estimated impact:** Low-medium | **Complexity:** Easy

Every `step()` converts the world tensor for Rust:

```python
self._world_CWH.permute(1, 2, 0).to(torch.int64).numpy()  # ppo.py:450
```

This runs 4,096 times per iteration (16 envs x 256 steps), allocating a new
numpy array each time. Pre-allocate and do an incremental update:

```python
# In __init__:
self._world_WHC_np = np.zeros((size, size, num_channels), dtype=np.int64)

# In step(), after mutating _world_CWH at (x, y):
self._world_WHC_np[x, y, :] = self._world_CWH[:, x, y].numpy()

# In reset():
self._world_WHC_np[:] = self._world_CWH.permute(1, 2, 0).to(torch.int64).numpy()

# Then:
throughput, num_unreachable = factorion_rs.simulate_throughput(self._world_WHC_np)
```

Risk: must keep `_world_WHC_np` perfectly in sync with `_world_CWH`. A missed
update causes silent correctness bugs.

Also documented as P3 in `docs/EXPERIMENTS.md`.

---

## Recommended implementation order

The highest bang-for-buck sequence:

1. **H1 + H4**: Increase `num_envs` to 64-128 and decrease `num_minibatches`
   to 4-8. Zero code changes — just CLI args. Benchmark immediately.

2. **H2**: Switch to `AsyncVectorEnv`. One-line change plus fixing the manual
   reset loop. Combined with step 1, this parallelises env stepping across
   cores.

3. **M1**: Action masking. Moderate code change, big sample efficiency gain.
   Every step now produces a useful learning signal.

4. **Profile**: After steps 1-3, profile to see where time is actually spent
   before investing in H3 (full Rust env). Use `py-spy` or PyTorch profiler.

5. **H3**: If env stepping is still the bottleneck after H2, move the full env
   to Rust with a batched interface. This is the highest-effort change but
   removes all remaining Python overhead.

6. **H5 + M3**: Mixed precision and fewer update epochs. These become
   meaningful once GPU is actually doing work (after H1/H4).

7. **M2 + M4**: Quality-of-life improvements. Small gains but easy to do.
