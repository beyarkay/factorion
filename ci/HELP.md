## `/ci` — copy-paste cheatsheet

Each job runs **this PR's head commit** on a self-terminating pod and reports
back here as a comment. Only the flags shown are overridable; everything else
comes from `training_config.py`. Full grammar: `ci/README.md`.

```text
# sweeps                            (also: --agents-per-pod N)
/ci sweep sft --pods 4
/ci sweep ppo --pods 4

# single runs
/ci sft --num-samples 5000000
/ci ppo --start-from abc123 --total-timesteps 40000000

# comparisons — N seeds/side; add `assert` lines for a pass/fail status
/ci compare sft --seeds 3
assert pr:val/thput > main:val/thput
assert pr:val/acc >= 0.5

/ci compare ppo --start-from abc123 --total-timesteps 40000000 --seeds 3
assert pr:val/thput > main:val/thput

# pods
/ci pods
/ci kill abc123                    # or: /ci kill --all
/ci watchdog --dry-run
```

`assert` sides: `pr:`/`test:` = this branch, `main:`/`base:` = baseline; ops
`< > <= >= == ~=` (`~=` ≈ equal, append `+- tol`). Bare numbers are thresholds.
