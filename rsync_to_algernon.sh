#!/bin/bash
REMOTE_HOST="192.168.0.218"
REMOTE_DIR="/home/brk/projects/factorion"

# Rsync command (preserves permissions, excludes .git and other temp files)
echo "Syncing files to brk@${REMOTE_HOST}:${REMOTE_DIR}..."
rsync -avz \
    --progress \
    --exclude='.git/' \
    "ppo.py" \
    "factorion.py" \
    "run_sweep.sh" \
    "sweep.yaml" \
    "pyproject.toml" \
    "uv.lock" \
    "brk@${REMOTE_HOST}:${REMOTE_DIR}"

# SSH command to run remotely after sync. `uv run` syncs deps from the lockfile
# (preserving the maturin-built factorion_rs) and runs inside the project venv.
echo "Running remote command: '$1'"
ssh "brk@${REMOTE_HOST}" "cd ${REMOTE_DIR} && nohup uv run $1 > $(date +'logs/factorion.%Y-%m-%dT%H%M%S.log') 2>&1 &"


