#!/usr/bin/env bash
#
# sft_bench_run.sh — one reproducible SFT TRAINING-SPEED benchmark run.
#
# Times the SFT training loop (data path + forward/backward + the per-epoch
# val pass) to a FIXED, deterministic policy quality, holding the computation
# constant. This is the SFT analogue of BENCHMARK.md's run.sh: a PURE-SPEED
# benchmark, so a change is only valid if it leaves `val_loss` BIT-IDENTICAL
# (the invariance signature) — i.e. it made the same training faster without
# altering the math or skipping a step.
#
# Setup mirrors the PPO quality benchmark's offline cached checkpoint: the
# (state,action) dataset is generated once and cached (--dataset-cache), so the
# timed run skips the ~26s build_factory data generation (that CPU cost is a
# separate axis — see the Rust generation port) and measures the TRAINING loop,
# which is the SFT-specific optimisation target. The rollout eval is disabled
# (eval-rollouts-every-n-epochs 0): it is a noisy held-out *diagnostic*, not a
# loss calculation, and including it would swamp training-loop speedups.
#
# Fixed config: size 11 + the real 93-69-96 encoder (so speedups transfer),
# 60k samples x 10 epochs. Baseline: val_loss = 1.6888, ~90 s on this box.
#
# Any extra flag is forwarded to sft.py and overrides the defaults below.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

CACHE="${CACHE:-checkpoints/sft_bench_ds_60k.pt}"

WANDB_MODE=disabled WANDB_DISABLED=true uv run python sft.py \
  --seed 1 \
  --size 11 \
  --num-samples 60000 \
  --epochs 10 \
  --batch-size 512 \
  --layer1 93 --layer2 69 --layer3 96 \
  --eval-rollouts-every-n-epochs 0 \
  --dataset-cache "$CACHE" \
  --checkpoint-path "${CKPT:-/tmp/sft_bench_ckpt.pt}" \
  --summary-path "${SUMMARY_PATH:-/tmp/sft_bench_summary.json}" \
  "$@"
