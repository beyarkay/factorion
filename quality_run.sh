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
# is logged every iteration at no extra cost and — crucially — is comparable
# across env/batch/LR/precision changes (those don't change the reward scale;
# only step_penalty / throughput_reward_scale would, and we hold those fixed).
#
# Determinism: fixed seed + torch_deterministic => the training trajectory (and
# therefore the *iteration* at which quality is reached) is reproducible; only
# per-iteration wall-time jitter varies between runs. That is what makes the
# time-to-quality number repeatable enough for hyperfine --runs 5.
#
# Starts from the SFT checkpoint cached on disk (checkpoints/sft_j0s5y2mc.pt;
# re-create with: scripts-free, see EXPERIMENTS / _resolve_wandb_checkpoint) so
# no network access is needed.
#
# Override any sweepable knob via env vars, e.g.:
#   LR=3e-4 NUM_ENVS=32 ./quality_run.sh
#   TARGET_VALUE=-0.10 ./quality_run.sh
#   EXTRA_ARGS="--amp" ./quality_run.sh        # once an AMP flag exists
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

CKPT="${CKPT:-checkpoints/sft_j0s5y2mc.pt}"
if [ ! -f "$CKPT" ]; then
  echo "ERROR: SFT checkpoint not found at $CKPT" >&2
  echo "Recreate it offline-cacheable via _resolve_wandb_checkpoint('j0s5y2mc')." >&2
  exit 1
fi

# Quality target (EMA of --target-metric). -0.15 is reached ~iter 31 (~110s on
# the current box) — deep in the learning phase, monotonic crossing.
TARGET_METRIC="${TARGET_METRIC:-rollout/reward}"
TARGET_VALUE="${TARGET_VALUE:--0.15}"
QUALITY_EMA_ALPHA="${QUALITY_EMA_ALPHA:-0.4}"
MAX_SECONDS="${MAX_SECONDS:-300}"

# Canonical SFT->PPO finetune recipe (the j0s5y2mc reference run). Sweepable
# knobs are env-overridable; everything else is fixed so comparisons are clean.
NUM_ENVS="${NUM_ENVS:-16}"
NUM_STEPS="${NUM_STEPS:-256}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-32}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-8}"
LR="${LR:-1.619489860053545e-4}"

WANDB_MODE=disabled WANDB_DISABLED=true uv run ppo.py \
  --seed 1 \
  --size 11 \
  --start-from "$CKPT" \
  --num-envs "$NUM_ENVS" \
  --num-steps "$NUM_STEPS" \
  --num-minibatches "$NUM_MINIBATCHES" \
  --update-epochs "$UPDATE_EPOCHS" \
  --learning-rate "$LR" \
  --ent-coef-start 7.05347e-4 --ent-coef-end 7.92625e-4 \
  --gae-lambda 0.9021936994100002 --gamma 0.9957335539938416 \
  --max-grad-norm 1.979 --clip-coef 0.2746 --target-kl 0.02 \
  --critic-warmup 10 --tile-head-std 0.06503 --adam-epsilon 6.866e-6 \
  --layer1 93 --layer2 69 --layer3 96 \
  --eval-every 0 \
  --target-metric "$TARGET_METRIC" \
  --target-value "$TARGET_VALUE" \
  --quality-ema-alpha "$QUALITY_EMA_ALPHA" \
  --max-seconds "$MAX_SECONDS" \
  --total-timesteps 100000000 \
  --summary-path "${SUMMARY_PATH:-/tmp/quality_ppo_summary.json}"
