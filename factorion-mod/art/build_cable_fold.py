"""Rebuild the single Cable fold model and the installed mod's sprite layers.

uv run --no-sync python factorion-mod/art/build_cable_fold.py
  [--blender /path/to/blender] [--device CPU|METAL|CUDA|OPTIX|HIP]

Builds once and renders both rotations in one Blender process.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    art = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blender",
        default=shutil.which("blender")
        or "/Applications/Blender.app/Contents/MacOS/Blender",
    )
    parser.add_argument(
        "--device", default="METAL" if sys.platform == "darwin" else "CPU"
    )
    parser.add_argument("--samples", type=int, default=64)
    args = parser.parse_args()
    sprites = art.parent / "mod/graphics/entity"
    out = art / "cable-fold"
    subprocess.run(
        [
            args.blender,
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
            str(art / "cable_fold.py"),
            "--",
            str(out),
            "both",
            "--device",
            args.device,
            "--samples",
            str(args.samples),
        ],
        check=True,
    )
    for orientation in ("horizontal", "vertical"):
        out = art / "cable-fold" / orientation
        pack = [sys.executable, str(art / "pack_cable_fold.py"), str(out), str(sprites)]
        if orientation == "horizontal":
            pack += [
                "--icon-out",
                str(art.parent / "mod/graphics/icons/gpu-server.png"),
            ]
        subprocess.run(pack, check=True)


if __name__ == "__main__":
    main()
