"""Visualise SFT training data as a browseable HTML file.

Generates the same expert demonstrations the SFT pipeline trains on, but
writes each (state, action) pair to an HTML page instead of feeding it to
a neural network. Lets you eyeball what the model is being trained on
before committing to a long run.

Reuses generate_lesson + extract_expert_actions + world2html. No
data-gen logic is duplicated.

Usage:
    uv run python scripts/visualise_sft_data.py --num-samples 30
    open sft_training_data.html
"""

import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import factorion  # noqa: E402
from sft import extract_expert_actions  # noqa: E402

_, _objs = factorion.datatypes.run()
_, _fns = factorion.functions.run()

Channel = _objs["Channel"]
LessonKind = _objs["LessonKind"]
generate_lesson = _fns["generate_lesson"]
world2html = _fns["world2html"]

TARGET_COLOR = "rgba(40, 200, 80, 0.55)"  # green: the single tile this pair trains on


@dataclass
class VizArgs:
    num_samples: int = 200
    """number of (state, action) pairs to render"""
    size: int = 8
    """grid size"""
    max_level: int = 0
    """max curriculum level (0 = auto: 2*size)"""
    seed: int = 1
    """random seed"""
    output_path: str = "sft_training_data.html"
    """where to write the HTML"""
    kind: str = ""
    """if set, only render this LessonKind (e.g. ASSEMBLE_1IN_1OUT)"""
    final_only: bool = False
    """if set, render one solved (completed) factory per lesson instead of
    each (state, action) pair — useful for eyeballing variety in
    generated lessons rather than the SFT training stream"""




def main(args: VizArgs) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    max_level = args.max_level if args.max_level > 0 else 2 * args.size
    if args.kind:
        kinds = [LessonKind[args.kind]]
    else:
        kinds = list(LessonKind)

    sections: list[str] = []
    kind_counts: dict[str, int] = {k.name: 0 for k in kinds}
    seed = args.seed
    samples_so_far = 0

    while samples_so_far < args.num_samples:
        kind = random.choice(kinds)
        level = random.randint(1, max_level)
        seed += 1

        try:
            solved, _ = generate_lesson(
                size=args.size, kind=kind, num_missing_entities=0, seed=seed,
            )
            if not args.final_only:
                task, _ = generate_lesson(
                    size=args.size, kind=kind, num_missing_entities=level, seed=seed,
                )
        except Exception:
            continue

        if args.final_only:
            world_html = world2html(solved.permute(1, 2, 0)).text
            sections.append(
                f"""
                <div class="card">
                  <div class="card-head">
                    #{samples_so_far + 1} · {kind.name} · seed={seed}
                  </div>
                  {world_html}
                </div>
                """
            )
            kind_counts[kind.name] += 1
            samples_so_far += 1
            continue

        pairs = extract_expert_actions(solved, task)
        n_pairs = len(pairs)
        for pair_idx, (obs, tile_idx, entity_id, direction_id, _) in enumerate(pairs):
            x, y = tile_idx // args.size, tile_idx % args.size
            # Build a "what the model should output" world: input state with
            # the target entity+direction placed on the target tile. The
            # other channels (items, misc) at that tile stay empty — those
            # aren't part of the SFT prediction targets.
            view = obs.clone()
            view[Channel.ENTITIES.value, x, y] = entity_id
            view[Channel.DIRECTION.value, x, y] = direction_id
            world_html = world2html(
                view.permute(1, 2, 0), highlights={(x, y): TARGET_COLOR}
            ).text
            # Header gives the full reproducer: kind + level + lesson seed +
            # pair index within that lesson. Reproduce in a REPL with
            #   solved, _ = generate_lesson(size=N, kind=KIND, num_missing_entities=0, seed=SEED)
            #   task,   _ = generate_lesson(size=N, kind=KIND, num_missing_entities=L, seed=SEED)
            #   pair = extract_expert_actions(solved, task)[PAIR_IDX]
            sections.append(
                f"""
                <div class="card">
                  <div class="card-head">
                    #{samples_so_far + 1} · {kind.name} · L{level} ·
                    seed={seed} · pair {pair_idx + 1}/{n_pairs} · target=({x},{y})
                  </div>
                  {world_html}
                </div>
                """
            )
            kind_counts[kind.name] += 1
            samples_so_far += 1
            if samples_so_far >= args.num_samples:
                break

    breakdown = " · ".join(
        f"{k}: {v}" for k, v in sorted(kind_counts.items()) if v > 0
    )
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>SFT training data</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1600px; margin: 1em auto; padding: 0 1em; color: #222; }}
  h1 {{ margin: 0 0 0.2em 0; }}
  .meta {{ color: #555; margin-bottom: 0.5em; }}
  .legend {{ font-size: 0.85em; color: #555; margin: 0 0 1em 0; }}
  .swatch {{ display: inline-block; width: 0.9em; height: 0.9em; vertical-align: middle; border: 1px solid #999; margin-right: 0.2em; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 1em; }}
  .card {{ border: 1px solid #ddd; padding: 0.4em; border-radius: 4px; }}
  .card-head {{ font-size: 0.8em; color: #666; margin-bottom: 0.3em; font-family: monospace; }}
</style></head><body>
<h1>SFT training data preview</h1>
<p class="meta">
  {samples_so_far} samples · size={args.size} · max_level={max_level} · seed={args.seed} · {breakdown}
</p>
<p class="legend">
  Each card is one (state, action) training pair. All non-green cells are
  the model's <strong>input state</strong>; the
  <span class="swatch" style="background: {TARGET_COLOR};"></span>
  green cell is the single (entity, direction) the model is being trained to
  predict at this step (icon and arrow shown there are the SFT target, not
  part of the input).<br>
  Grey cross-hatched cells are <strong>UNAVAILABLE</strong> (FOOTPRINT
  channel = unbuildable). The lesson generators set this for every cell
  that's empty in the solved layout, so the agent's input always
  identifies the buildable region as part of the training data.
</p>
<div class="grid">
{"".join(sections)}
</div>
</body></html>"""

    out = Path(args.output_path)
    out.write_text(page)
    print(f"Wrote {samples_so_far} samples to {out}")


if __name__ == "__main__":
    main(tyro.cli(VizArgs))
