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
    "requirements.txt" \
    "brk@${REMOTE_HOST}:${REMOTE_DIR}"

# SSH command to run remotely after sync
echo "Running remote command: '$1'"
ssh "brk@${REMOTE_HOST}" "cd ${REMOTE_DIR} && source .venv/bin/activate && nohup $1 > $(date +'logs/factorion.%Y-%m-%dT%H%M%S.log') 2>&1 &"


