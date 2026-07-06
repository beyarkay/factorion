#!/bin/bash
# ci/runner.sh — pod-side setup, run from the repo root at the requested
# commit by the bootstrap that ci/launch.py bakes into the pod's docker
# command. Builds the Rust extension and hands off to the Python job
# dispatcher; the bootstrap's EXIT trap terminates the pod afterwards no
# matter how this script ends. Job parameters travel exclusively via the
# FCI_JOB_B64 spec consumed by `python -m ci.jobs` — nothing is configured
# here.
set -euo pipefail

export PATH="/root/.cargo/bin:${PATH}"
# Required by torch.use_deterministic_algorithms on CUDA.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Second safety layer (first: the bootstrap EXIT trap; third: the 6-hourly
# GH-cron watchdog reading the deadline encoded in the pod name): hard-kill
# the pod at FCI_DEADLINE even if the job wedges.
if [ -n "${FCI_DEADLINE:-}" ]; then
    cat > /workspace/deadline_watchdog.py << 'PY'
import os
import time

import runpod

remaining = int(os.environ["FCI_DEADLINE"]) - time.time()
if remaining > 0:
    time.sleep(remaining)
runpod.api_key = os.environ["RUNPOD_API_KEY"]
runpod.terminate_pod(os.environ["RUNPOD_POD_ID"])
PY
    nohup python /workspace/deadline_watchdog.py &> /dev/null &
    echo ">>> Deadline watchdog armed (epoch ${FCI_DEADLINE})"
fi

echo ">>> GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2> /dev/null || echo unknown)"
echo ">>> Building factorion_rs..."
(cd factorion_rs && maturin build --release --out dist && pip install --force-reinstall dist/*.whl)

echo ">>> Starting job..."
python -m ci.jobs 2>&1 | tee /workspace/job.log
