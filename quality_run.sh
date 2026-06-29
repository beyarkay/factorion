#!/usr/bin/env bash
#
# quality_run.sh — reproducible TIME-TO-QUALITY benchmark run.
#
# Unlike run.sh (which times a fixed iteration count), this finetunes the
# canonical SFT policy with PPO and STOPS the moment an EMA-smoothed quality
# metric crosses a threshold, recording the wall-clock time-to-quality. This is
# the benchmark for changes that are allowed to alter the *computation* — number
# of envs, batch size, learning rate, numeric precision (AMP/bf16), etc. — where
# "faster" means "reaches the same policy quality in less wall-clock time".
#
# Fully OFFLINE: WANDB disabled, eval disabled (eval/* needs an extra forward
# pass and is noisy); the quality signal is the on-policy rollout reward, which
# is logged every iteration at no extra cost and is comparable across
# env/batch/LR/precision changes (those don't change the reward scale; only
# step_penalty / throughput_reward_scale would, and we hold those fixed). Starts
# from the on-disk SFT checkpoint so no network is needed.
#
# Determinism / noise: for a FIXED seed the trajectory (hence the crossing
# iteration) is reproducible, so repeated runs of the SAME config differ only by
# wall-time jitter. But changing batch size / precision resamples the trajectory
# (FP reduction order, gradients), so a single seed is one draw from a noisy
# distribution — compare CONFIGS across several seeds (quality_measure.sh sweeps
# the seed) and look at the mean, not a single run.
#
# Any flag is forwarded to ppo.py and OVERRIDES the defaults below (tyro takes
# the last value), so sweeping a knob is just:
#   ./quality_run.sh --learning-rate 3e-4 --num-envs 32
#   ./quality_run.sh --seed 3
#   ./quality_run.sh --target-value -0.10
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

CKPT="${CKPT:-checkpoints/sft_j0s5y2mc.pt}"
if [ ! -f "$CKPT" ]; then
  echo "ERROR: SFT checkpoint not found at $CKPT" >&2
  echo "Recreate it offline-cacheable via _resolve_wandb_checkpoint('j0s5y2mc')." >&2
  exit 1
fi

# Canonical SFT->PPO finetune recipe (the j0s5y2mc reference run) + the
# time-to-quality target. Everything here is a DEFAULT: append flags on the
# command line to override any of them.
WANDB_MODE=disabled WANDB_DISABLED=true uv run ppo.py \
  --seed 1 \
  --size 11 \
  --start-from "$CKPT" \
  --num-envs 16 \
  --num-steps 256 \
  --num-minibatches 32 \
  --update-epochs 8 \
  --learning-rate 1.619489860053545e-4 \
  --ent-coef-start 7.05347e-4 --ent-coef-end 7.92625e-4 \
  --gae-lambda 0.9021936994100002 --gamma 0.9957335539938416 \
  --max-grad-norm 1.979 --clip-coef 0.2746 --target-kl 0.02 \
  --critic-warmup 10 --tile-head-std 0.06503 --adam-epsilon 6.866e-6 \
  --layer1 93 --layer2 69 --layer3 96 \
  --eval-every 0 \
  --target-metric rollout/reward \
  --target-value -0.15 \
  --quality-ema-alpha 0.4 \
  --max-seconds 300 \
  --total-timesteps 100000000 \
  --summary-path "${SUMMARY_PATH:-/tmp/quality_ppo_summary.json}" \
  "$@"
