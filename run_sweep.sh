#!/bin/bash
source .venv/bin/activate
python cleanrl/ppo.py "$@"
