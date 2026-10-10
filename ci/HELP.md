## `/ci` — copy-paste cheatsheet

Each job runs **this PR's head commit** on a self-terminating pod and reports
back here as a comment. Only the flags shown are overridable; everything else
comes from `training_config.py`. Full grammar: `ci/README.md`.

```text
# sweeps                            (also: --agents-per-pod N)
/ci sweep sft --pods 4
/ci sweep ppo --pods 4

# single runs
/ci sft --num-samples 5M
/ci ppo --start-from abc123 --total-timesteps 40M

# comparisons — N seeds/side; add `assert` lines for a pass/fail status
/ci compare sft --seeds 3
assert pr:val/thput@8 > main:val/thput@8
assert pr:val/acc >= 0.5

/ci compare ppo --start-from abc123 --total-timesteps 40M --seeds 3
assert pr:eval/thput@8 > main:eval/thput@8

# pods
/ci pods
/ci kill abc123                    # or: /ci kill --all
/ci watchdog --dry-run
```

`--num-samples`/`--total-timesteps` take a plain count or an SI-ish
suffix: `500k`, `5M`, `4.5M`, `2B`.

`assert` sides: `pr:`/`test:` = this branch, `main:`/`base:` = baseline; ops
`< > <= >= == ~=` (`~=` ≈ equal, append `+- tol`). Bare numbers are thresholds.

A `compare` posts two comments: the every-metric table, then a factory diff —
both sides rebuilding the same factories, rendered side by side wherever their
throughput disagrees.

## GPUs: `--gpu-type`

`sft`, `ppo`, `compare` and `sweep` all take `--gpu-type "<RunPod GPU id>"`.
SFT work (`sft`, `compare sft`, `sweep sft`) defaults to the RTX 6000 Ada;
PPO defaults to the RTX 2000 Ada, because its rollout is CPU-bound and a bigger
card buys nothing. A card in the lineup (`GPU_FALLBACKS` in `ci/config.py`)
falls back to the lineup entries after it when it's unavailable. Any other id
is used exactly and fails if it can't be scheduled. A `compare`'s or `sweep`'s
pods all run on the card the first one landed on. Check the actual card in
W&B (`env/gpu_name`).

```text
/ci sft --num-samples 5M --gpu-type "NVIDIA RTX A4000"     # cheapest per sample
/ci compare sft --gpu-type "NVIDIA RTX A6000"              # pinned: last in the lineup
```

SFT at 15×15 with bf16 + compile, training step only (evals add ~10–20%, pod
setup ~10–20 min). Measured on #463, 2026-10; prices are RunPod's quote at the
time.

| `--gpu-type` | $/hr | s / 100k samples | samples/s | $ / 1M samples | 45M-sample run |
|---|---|---|---|---|---|
| `NVIDIA RTX A4000` | 0.17 | 21.2 | 4,700 | 0.010 | 2.7 h, $0.45 |
| `NVIDIA A40` | 0.49 | 14.7 | 6,800 | 0.020 | 1.8 h, $0.90 |
| `NVIDIA RTX A6000` | 0.53 | 13.4 | 7,500 | 0.020 | 1.7 h, $0.89 |
| `NVIDIA L40S` | 1.09 | 10.1 | 9,900 | 0.031 | 1.3 h, $1.38 |
| `NVIDIA RTX 6000 Ada Generation` (SFT default) | 0.74 | 7.9 | 12,700 | 0.016 | 1.0 h, $0.73 |
| `NVIDIA A100 80GB PCIe` | 1.59 | 7.4 | 13,500 | 0.033 | 0.9 h, $1.47 |

For scale, fp32 without compile on the RTX 2000 Ada is 144 s per 100k. Two
`NVIDIA H100 80GB HBM3` pods never got to training. Same-card pods vary ~18%
in speed, so pin the card (an id outside the lineup, or the A6000) when
comparing speed.
