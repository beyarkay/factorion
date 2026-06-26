#!/usr/bin/env bash
#
# run.sh — reproducible fixed-cost PPO speed run.
#
# A single, deterministic PPO training invocation for benchmarking. It is NOT a
# learning run (reported throughput will be ~0) — it just exercises the full
# loop (RL rollouts + optimiser steps) so we can measure how fast it goes.
#
# Designed to be wrapped by hyperfine, e.g.:
#   hyperfine --warmup 1 --runs 5 ./run.sh
#
# Override the amount of work with TOTAL_TIMESTEPS (must be a multiple of
# batch_size = num_envs * num_steps = 4096):
#   TOTAL_TIMESTEPS=16384 ./run.sh     # 4 iterations instead of the default 8
#
# Config mirrors a real training command (same net dims / num_envs / num_steps /
# num_minibatches / update_epochs / grid size) so the compute profile matches
# production. Deliberate differences: critic-warmup 0 (measure full PPO steps),
# eval-every 0 (eval isn't part of the loop), no --start-from (arch is fixed by
# --layer{1,2,3}, so it doesn't affect speed and dropping it keeps this offline),
# and WANDB disabled (no network/logging overhead).
#
# Iteration count: real training runs go 6+ hours over 100s of iterations, so
# the *steady-state per-iteration* cost is what matters — startup (Python import
# + one-time torch.compile() warmup, a few seconds total) is negligible there.
# We run 8 iterations so that per-iteration compute dominates the measured wall
# time and we optimise the right thing, rather than fixating on the one-off
# startup cost a 2-iteration benchmark over-weights. ppo.py also reports a
# "steady" mean (dropping iteration 1) in the summary JSON for a startup-free
# view. Keep TOTAL_TIMESTEPS fixed across runs you compare.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# batch_size = num_envs (16) * num_steps (256) = 4096; default = 8 iterations.
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-32768}"

WANDB_MODE=disabled WANDB_DISABLED=true uv run ppo.py \
  --adam-epsilon 6.866e-6 \
  --clip-coef 0.2746 \
  --critic-warmup 0 \
  --ent-coef-end 7.92625e-4 \
  --ent-coef-start 7.05347e-4 \
  --gae-lambda 0.9021936994100002 \
  --gamma 0.9957335539938416 \
  --layer1 93 --layer2 69 --layer3 96 \
  --learning-rate 1.619489860053545e-4 \
  --max-grad-norm 1.979 \
  --num-envs 16 \
  --num-minibatches 32 \
  --num-steps 256 \
  --seed 1 \
  --size 11 \
  --target-kl 0.02 \
  --tile-head-std 0.06503 \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --update-epochs 8 \
  --vf-coef 0.7426 \
  --eval-every 0 \
  --summary-path "${SUMMARY_PATH:-/tmp/run_ppo_summary.json}"
