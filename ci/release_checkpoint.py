"""Fetch the checkpoint a release ships and prove the release's code runs it.

The shipped checkpoint is the mod's default (`checkpoint` in
factorion-mod/mod/inference.lua), copied to `<out_dir>/<run_id>.pt` for the
GitHub release, where the server finds it without W&B. The release workflow
runs this against the installed wheel (`python -I`, outside the checkout), so
an architecture change that orphans the checkpoint fails the release.

    python -I ci/release_checkpoint.py factorion-mod/mod/inference.lua dist/
"""

import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from denoise import load_denoiser
from factorion import Channel, str2ent
from factorion_server.server import best_build
from sft import resolve_checkpoint
from training_config import SftArgs


def main(inference_lua: Path, out_dir: Path) -> None:
    match = re.search(r'checkpoint = "([^"]+)"', inference_lua.read_text())
    assert match, f"no default checkpoint in {inference_lua}"
    run_id = match.group(1)
    path, _ = resolve_checkpoint(f"beyarkay/factorion/{run_id}", SftArgs.wandb_project_name)

    size = SftArgs.size
    obs = np.zeros((len(Channel), size, size), dtype=np.float32)
    obs[Channel.FOOTPRINT.value] = 1
    obs[Channel.ENTITIES.value, 0, size // 2] = str2ent("source").value
    obs[Channel.ENTITIES.value, size - 1, size // 2] = str2ent("sink").value
    best_build(load_denoiser(path).eval(), obs, 1.0, 0, torch.device("cpu"))

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, out_dir / f"{run_id}.pt")
    print(f"{run_id} loads and generates; copied to {out_dir / f'{run_id}.pt'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
