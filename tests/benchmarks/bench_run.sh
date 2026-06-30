#!/usr/bin/env bash
#
# bench_run.sh <kind> [extra ppo.py/sft.py flags...] — one benchmark execution.
#
#   kind = ppo-speed    fixed-iteration PPO loop (pure-speed; throughput ~0)
#          ppo-quality  PPO finetune of the cached SFT ckpt to a reward threshold
#          sft          SFT training to a fixed val_loss (cached dataset)
#
# Wrapped by bench_measure.sh; runnable directly for a single timing/check.
# Any trailing flags are forwarded to the underlying ppo.py / sft.py (tyro takes
# the last value, so they override the recipe). See tests/benchmarks/CLAUDE.md
# for the full playbook and EXPERIMENT_LOG.md for what's been tried.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root

KIND="${1:?usage: bench_run.sh <ppo-speed|ppo-quality|sft> [flags]}"; shift || true

case "$KIND" in
  ppo-speed)
    # batch_size = num_envs(16) * num_steps(256) = 4096; default = 8 iterations.
    TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-32768}"
    WANDB_MODE=disabled WANDB_DISABLED=true uv run ppo.py \
      --seed 1 --size 11 \
      --num-envs 16 --num-steps 256 --num-minibatches 32 --update-epochs 8 \
      --learning-rate 1.619489860053545e-4 \
      --ent-coef-start 7.05347e-4 --ent-coef-end 7.92625e-4 \
      --gae-lambda 0.9021936994100002 --gamma 0.9957335539938416 \
      --max-grad-norm 1.979 --clip-coef 0.2746 --target-kl 0.02 --vf-coef 0.7426 \
      --critic-warmup 0 --tile-head-std 0.06503 --adam-epsilon 6.866e-6 \
      --layer1 93 --layer2 69 --layer3 96 \
      --eval-every 0 \
      --total-timesteps "$TOTAL_TIMESTEPS" \
      --summary-path "${SUMMARY_PATH:-/tmp/bench_ppo_speed.json}" "$@"
    ;;
  ppo-quality)
    CKPT="${CKPT:-checkpoints/sft_j0s5y2mc.pt}"
    if [ ! -f "$CKPT" ]; then
      echo "ERROR: SFT checkpoint not found at $CKPT" >&2
      echo "Recreate offline via _resolve_wandb_checkpoint('j0s5y2mc')." >&2
      exit 1
    fi
    WANDB_MODE=disabled WANDB_DISABLED=true uv run ppo.py \
      --seed 1 --size 11 --start-from "$CKPT" \
      --num-envs 16 --num-steps 256 --num-minibatches 32 --update-epochs 8 \
      --learning-rate 7e-4 \
      --ent-coef-start 7.05347e-4 --ent-coef-end 7.92625e-4 \
      --gae-lambda 0.9021936994100002 --gamma 0.9957335539938416 \
      --max-grad-norm 1.979 --clip-coef 0.2746 --target-kl 0.02 \
      --critic-warmup 5 --tile-head-std 0.06503 --adam-epsilon 6.866e-6 \
      --layer1 93 --layer2 69 --layer3 96 \
      --eval-every 0 \
      --target-metric rollout/reward --target-value -0.15 \
      --quality-ema-alpha 0.4 --max-seconds 300 --total-timesteps 100000000 \
      --summary-path "${SUMMARY_PATH:-/tmp/bench_ppo_quality.json}" "$@"
    ;;
  sft)
    # Times the SFT training loop (real 93-69-96 net, 60k samples x 10 epochs) to
    # a fixed deterministic val_loss. --dataset-cache skips the ~26s build_factory
    # data gen (a separate axis) so the timed run measures training; rollout eval
    # off (it's a noisy diagnostic, not a loss).
    CACHE="${CACHE:-checkpoints/sft_bench_ds_60k.pt}"
    WANDB_MODE=disabled WANDB_DISABLED=true uv run python sft.py \
      --seed 1 --size 11 --num-samples 60000 --epochs 10 --batch-size 512 \
      --layer1 93 --layer2 69 --layer3 96 \
      --eval-rollouts-every-n-epochs 0 \
      --dataset-cache "$CACHE" \
      --checkpoint-path "${CKPT:-/tmp/bench_sft_ckpt.pt}" \
      --summary-path "${SUMMARY_PATH:-/tmp/bench_sft.json}" "$@"
    ;;
  *)
    echo "unknown kind: $KIND (expected ppo-speed|ppo-quality|sft)" >&2
    exit 1
    ;;
esac
