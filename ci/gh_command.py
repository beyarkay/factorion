"""Dispatcher for `/ci ...` PR comments (invoked by ci-command.yml).

The workflow hands over the comment body plus PR context via env vars; this
module parses the command, launches pods with the repo's secrets, and posts
the outcome back to the PR — comments are the backbone of CI reporting.
`/ci help` posts HELP below, which doubles as the grammar reference.

Env: COMMENT_BODY, COMMENT_URL, PR_NUMBER, HEAD_SHA, GITHUB_TOKEN,
GITHUB_REPOSITORY, RUNPOD_API_KEY, WANDB_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
import traceback
from typing import Optional

from ci import github_api
from ci.config import (
    AGENTS_PER_POD_DEFAULT,
    COMPARE_NUM_SAMPLES_DEFAULT,
    COMPARE_SEEDS_DEFAULT,
    GPU_FALLBACKS,
    WANDB_PROJECT,
    PpoJob,
    SftJob,
    compare_group,
    compare_nonce,
    pod_emoji,
    pod_url,
    ppo_budget_seconds,
    sft_budget_seconds,
)
from ci.launch import (
    create_sweep,
    launch,
    launch_compare,
    read_sweep_config,
    resolve_ref,
    sweep_summary_line,
)

COMPARE_STATUS_CONTEXT = "factorion-ci/compare"
# Compare waits in-workflow so the commit status lands promptly; cap the wait
# under the workflow's own timeout.
MAX_WAIT_SECONDS = 320 * 60

HELP = """\
## `/ci` — GPU training jobs from PR comments

Every job runs **this PR's head commit** on a self-terminating RunPod pod and
reports back here as a comment. Hyperparameters come from `training_config.py`;
the flags below are the entire override surface (see `ci/README.md`).

### Train

- `/ci sft` — SFT from scratch (production-sized: `SftArgs.num_samples`)
- `/ci sft --num-samples 200000` — quick smoke run (~minutes)
- `/ci ppo --start-from j0s5y2mc` — PPO from an SFT checkpoint (W&B run id);
  optional `--total-timesteps N`

The result comment (headline metrics + all metrics) lands when the run
finishes, however long it takes.

### Compare this branch to main

- `/ci compare sft` — N seeds each of this branch and main (one pod per run),
  then a seed-paired diff of every logged metric. Options: `--seeds 3`,
  `--num-samples 5000000`, `--base-ref main`
- `/ci compare ppo --start-from j0s5y2mc` — same, comparing PPO finetuning
  from that checkpoint on both commits; optional `--total-timesteps N`

Add `assert` lines to turn the comparison into a pass/fail commit status:

```
/ci compare sft --seeds 3
assert pr:val/thput > main:val/thput
assert pr:val/acc >= 0.5
```

(`pr:`/`test:` = this branch, `main:`/`base:` = the baseline; bare numbers
are thresholds. Comparators on group means: `<` `>` `<=` `>=`, plus `==` /
`~=` meaning approximately equal — |lhs − rhs| <= 1e-3 by default, or append
a tolerance like `assert pr:val/acc == main:val/acc +- 0.01`. A missing
metric or unparseable line fails the check.)

### Sweeps

- `/ci sweep sft` or `/ci sweep ppo` — W&B sweep from this commit's
  `ci/sweep_<algo>.yaml`. Options: `--pods 1`, `--agents-per-pod 1`

### Pod management

- `/ci pods` — list CI pods (status, cost, deadline)
- `/ci kill <pod_id>` or `/ci kill --all` — terminate CI pods
- `/ci watchdog --dry-run` — show what the leaked-pod reaper would do
- `/ci help` — this message
"""


def parse_comment(body: str) -> tuple[list[str], list[str]]:
    """Extract (argv tokens, assertion expressions) from a comment body."""
    tokens: list[str] = []
    assertions: list[str] = []
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("/ci") and not tokens:
            tokens = shlex.split(line)[1:]
        elif line.lower().startswith("assert "):
            assertions.append(line[len("assert ") :].strip())
    return tokens, assertions


_ENTITY_CACHE: list = []


def _wandb_entity():
    """Cached W&B entity for building run URLs; None if unresolvable."""
    if not _ENTITY_CACHE:
        try:
            import wandb

            _ENTITY_CACHE.append(wandb.Api().default_entity)
        except Exception:
            _ENTITY_CACHE.append(None)
    return _ENTITY_CACHE[0]


def _commit_link(sha: str) -> str:
    repo = os.environ.get("GITHUB_REPOSITORY")
    if repo:
        return f"[`{sha[:7]}`](https://github.com/{repo}/commit/{sha})"
    return f"`{sha[:7]}`"


def _group_link(group: str) -> str:
    entity = _wandb_entity()
    if entity:
        return f"[`{group}`](https://wandb.ai/{entity}/{WANDB_PROJECT}/groups/{group})"
    return f"`{group}`"


def _project_link(sha7: str) -> str:
    """Markdown link to the W&B project (filter by tag sha:<sha7> there)."""
    entity = _wandb_entity()
    url = f"https://wandb.ai/{entity}/{WANDB_PROJECT}" if entity else "https://wandb.ai"
    return f"[W&B project]({url}) (runs tagged `sha:{sha7}`)"


PENDING = ("⏳", "⏳")  # (pod, run) states before anything has happened
_RUN_EMOJI = {"finished": "✅", "crashed": "❌", "failed": "❌", "killed": "❌"}
_STATUS_LEGEND = (
    "⏳ pending · 🚀 running · ✅ finished · ❌ crashed · "
    "💀 pod past deadline · 🗑 pod terminated"
)


def _launch_statuses(infos: list[dict]) -> dict[str, tuple[str, str]]:
    """Live (pod_emoji, run_emoji) per launched pod, from RunPod + W&B."""
    from ci import runpod_api

    import wandb

    pods = {p["id"]: p for p in runpod_api.list_ci_pods()}
    api = wandb.Api()
    entity = _wandb_entity()
    out = {}
    for info in infos:
        pod_state = pod_emoji(pods.get(info["pod_id"]))
        run_emoji = "⏳"
        run_id = info.get("wandb_run_id")
        if run_id and entity:
            try:
                run = api.run(f"{entity}/{WANDB_PROJECT}/{run_id}")
                run_emoji = _RUN_EMOJI.get(run.state, "🚀")
            except Exception:
                run_emoji = "⏳"  # wandb.init hasn't created the run yet
        out[info["pod_id"]] = (pod_state, run_emoji)
    return out


def _launched_comment(
    title: str,
    infos: list[dict],
    footer: str = "",
    statuses: Optional[dict[str, tuple[str, str]]] = None,
) -> str:
    """statuses: live (pod, run) emoji per pod id — only for comments that
    something keeps editing (compare); a fire-and-forget launch comment must
    not show a ⏳ nobody will ever update."""
    lines = [f"## &#x1F440; {title}", ""]
    entity = _wandb_entity() if any(i.get("wandb_run_id") for i in infos) else None
    for info in infos:
        pod_emoji, run_emoji = (
            statuses.get(info["pod_id"], PENDING) if statuses is not None else ("", "")
        )
        line = (
            f"- {pod_emoji} pod [`{info['pod_id']}`]({pod_url(info['pod_id'])}) "
            f"(`{info['pod_name']}`)"
        )
        run_id = info.get("wandb_run_id")
        if run_id and entity:
            line += (
                f" &rarr; {run_emoji} [W&B run `{run_id}`]"
                f"(https://wandb.ai/{entity}/{WANDB_PROJECT}/runs/{run_id})"
            )
        elif run_id:
            line += f" &rarr; {run_emoji} W&B run `{run_id}`"
        lines.append(line.replace("  ", " "))
    if statuses is not None:
        lines += ["", f"_{_STATUS_LEGEND}_"]
    spec = {k: v for k, v in infos[0]["job"].items() if k != "extra_tags"}
    lines += [
        "",
        "Job spec (every other hyperparameter comes from `training_config.py` "
        "at this commit; for compares, `seed` varies per pod):",
        "```json",
        json.dumps(spec, indent=2),
        "```",
    ]
    if footer:
        lines += ["", footer]
    return "\n".join(lines)


def _with_trigger_footer(ctx, body: str) -> str:
    """Every CI comment links back to the /ci comment that triggered it —
    PRs accumulate many CI comments and the trail matters."""
    url = ctx.get("comment_url")
    if url:
        return f"{body.rstrip()}\n\n_Originally triggered by [this comment]({url})_"
    return body


def _post(ctx, body: str) -> int:
    comment_id = github_api.post_pr_comment(ctx["pr"], _with_trigger_footer(ctx, body))
    # Tell the workflow we replied: its dispatch-failure fallback comment is
    # only for runs that die BEFORE reaching the PR (a deliberate exit 1 — a
    # failed compare assertion — already reported itself).
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a") as fh:
            fh.write("replied=true\n")
    return comment_id


def cmd_sft(args, ctx) -> None:
    job = SftJob(
        sha=ctx["sha"], num_samples=args.num_samples, extra_tags=[f"pr:{ctx['pr']}"]
    )
    info = launch(job, args.gpu_type, wait=False)
    _post(
        ctx,
        _launched_comment(
            f"SFT run launched at {_commit_link(ctx['sha'])}",
            [info],
            footer=f"Results land here as a comment when the run finishes. {_project_link(ctx['sha'][:7])}",
        ),
    )


def cmd_ppo(args, ctx) -> None:
    job = PpoJob(
        sha=ctx["sha"],
        start_from=args.start_from,
        total_timesteps=args.total_timesteps,
        extra_tags=[f"pr:{ctx['pr']}"],
    )
    info = launch(job, args.gpu_type, wait=False)
    _post(
        ctx,
        _launched_comment(
            f"PPO run launched at {_commit_link(ctx['sha'])} (from `{args.start_from}`)",
            [info],
            footer=f"Results land here as a comment when the run finishes. {_project_link(ctx['sha'][:7])}",
        ),
    )


def _compare_wait_seconds(args) -> int:
    if args.algo == "sft":
        budget = sft_budget_seconds(args.num_samples, 1)
    else:
        from training_config import PpoArgs

        budget = ppo_budget_seconds(args.total_timesteps or PpoArgs().total_timesteps)
    return min(budget + 45 * 60, MAX_WAIT_SECONDS)


def _missing_run_warnings(infos: list[dict]) -> str:
    """Name the pods whose pre-assigned W&B run never appeared, so a
    short-handed report explains itself (pod died, or is still stuck pulling
    the image — the pod page tells which)."""
    try:
        import wandb

        api = wandb.Api()
        entity = _wandb_entity()
        if not entity:
            return ""
        missing = []
        for info in infos:
            run_id = info.get("wandb_run_id")
            if not run_id:
                continue
            try:
                api.run(f"{entity}/{WANDB_PROJECT}/{run_id}")
            except Exception:
                missing.append(info)
        if not missing:
            return ""
        lines = [
            "> [!WARNING]",
            "> Pod(s) never produced their W&B run — died, or still stuck "
            "pulling the image (see the pod page). Rerun the /ci command to "
            "fill the gap:",
        ]
        for info in missing:
            job = info.get("job", {})
            lines.append(
                f"> - pod [`{info['pod_id']}`]({pod_url(info['pod_id'])}) "
                f"(seed {job.get('seed')}, group `{job.get('group')}`) — "
                f"run `{info['wandb_run_id']}` never appeared"
            )
        return "\n".join(lines) + "\n\n"
    except Exception:
        return ""


def _post_compare_outcome(
    ctx,
    assertions: list[str],
    main_group: str,
    pr_group: str,
    infos: Optional[list[dict]] = None,
) -> None:
    from ci.report import compare_report

    md, ok = compare_report(
        main_group=main_group,
        pr_group=pr_group,
        assertions=assertions,
    )
    _post(ctx, _missing_run_warnings(infos or []) + md)
    state = "success" if ok else "failure"
    description = (
        "all assertions passed"
        if ok and assertions
        else "comparison posted"
        if ok
        else "assertion failed or no runs to compare"
    )
    github_api.set_commit_status(ctx["sha"], state, COMPARE_STATUS_CONTEXT, description)
    if not ok:
        sys.exit(1)


def cmd_compare(args, ctx) -> None:
    nonce = compare_nonce()
    pr_group = compare_group(ctx["sha"], args.algo, nonce, "pr")
    main_group = compare_group(ctx["sha"], args.algo, nonce, "main")
    infos = launch_compare(
        algo=args.algo,
        sha=ctx["sha"],
        base_sha=resolve_ref(args.base_ref),
        nonce=nonce,
        seeds=args.seeds,
        num_samples=args.num_samples,
        start_from=args.start_from,
        total_timesteps=args.total_timesteps,
        gpu_type=args.gpu_type,
        extra_tags=[f"pr:{ctx['pr']}"],
    )
    assertion_note = (
        "\n".join(f"- `{a}`" for a in ctx["assertions"]) or "_none — report only_"
    )
    title = (
        f"Compare launched: {_commit_link(ctx['sha'])} vs `{args.base_ref}` "
        f"({args.algo}, {args.seeds} seeds x 2 sides, one pod per run)"
    )
    footer = (
        f"W&B groups: {_group_link(pr_group)} vs "
        f"{_group_link(main_group)}\n\n"
        f"Assertions:\n{assertion_note}"
    )
    comment_id = _post(
        ctx,
        _launched_comment(
            title, infos, footer=footer, statuses={i["pod_id"]: PENDING for i in infos}
        ),
    )

    def refresh_launch_comment() -> None:
        # Best-effort: a failed edit must never take down the compare wait.
        try:
            body = _launched_comment(
                title, infos, footer=footer, statuses=_launch_statuses(infos)
            )
            github_api.update_pr_comment(comment_id, _with_trigger_footer(ctx, body))
        except Exception as e:
            print(f"could not refresh the launch comment: {e}", flush=True)

    github_api.set_commit_status(
        ctx["sha"], "pending", COMPARE_STATUS_CONTEXT, "compare runs in flight"
    )

    from ci.report import wait_for_groups

    wait_for_groups(
        main_group=main_group,
        pr_group=pr_group,
        expect_each=args.seeds,
        timeout_seconds=_compare_wait_seconds(args),
        pod_ids=[i["pod_id"] for i in infos if i["pod_id"]],
        on_poll=refresh_launch_comment,
    )
    refresh_launch_comment()  # final ✅/❌/🗑 states next to the report
    _post_compare_outcome(ctx, ctx["assertions"], main_group, pr_group, infos=infos)


def cmd_sweep(args, ctx) -> None:
    algo = args.algo
    sweep_path = create_sweep(algo, ctx["sha"])
    from ci.config import SweepJob

    infos = [
        launch(
            SweepJob(
                sha=ctx["sha"],
                algo=algo,
                sweep_path=sweep_path,
                agents_per_pod=args.agents_per_pod,
                extra_tags=[f"pr:{ctx['pr']}"],
            ),
            args.gpu_type,
            wait=False,
        )
        for _ in range(args.pods)
    ]
    entity, project, sweep_id = sweep_path.split("/")
    sweep_url = f"https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}"
    sweep_line = sweep_summary_line(read_sweep_config(algo, ctx["sha"]))
    _post(
        ctx,
        _launched_comment(
            f"{algo.upper()} sweep launched at {_commit_link(ctx['sha'])} "
            f"({args.pods} pod(s) x {args.agents_per_pod} agents)",
            infos,
            footer=(
                f"Sweeping: {sweep_line}\n\n"
                f"[View sweep on W&B]({sweep_url}) — report follows when the sweep drains."
            ),
        ),
    )

    # Wait for the sweep to drain (run_cap from the yaml), then report.
    from ci.report import sweep_report

    _wait_for_sweep(sweep_path)
    _post(ctx, sweep_report(sweep_path))


def _wait_for_sweep(sweep_path: str, timeout_seconds: int = MAX_WAIT_SECONDS) -> None:
    import wandb

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        sweep = wandb.Api().sweep(sweep_path)
        if str(sweep.state).upper() in ("FINISHED", "CANCELLED", "CANCELED"):
            return
        print(f"sweep state: {sweep.state}", flush=True)
        time.sleep(300)
    print(f"sweep still running after {timeout_seconds}s; posting a partial report")


def _pods_table(pods: list[dict]) -> str:
    from ci.cli import pod_summary

    lines = [
        "| Pod | Name | Status | GPU | Uptime | $/hr | Deadline |",
        "|---|---|---|---|---|---|---|",
    ]
    for pod in pods:
        s = pod_summary(pod)
        lines.append(
            f"| [`{s['id']}`]({s['url']}) | `{s['name']}` | {s['status']} "
            f"| {s['gpu']} | {s['uptime']} | ${s['cost_hr']} | {s['deadline']} |"
        )
    return "\n".join(lines)


def cmd_pods(args, ctx) -> None:
    from ci import runpod_api

    ci_pods = runpod_api.list_ci_pods()
    body = _pods_table(ci_pods) if ci_pods else "No CI pods running."
    _post(ctx, body)


def cmd_kill(args, ctx) -> None:
    from ci import runpod_api

    targets = (
        runpod_api.list_ci_pods()
        if args.all
        else [{"id": args.pod_id}]
        if args.pod_id
        else []
    )
    if not targets:
        _post(ctx, "Nothing to kill: pass a pod id or `--all`.")
        return
    lines = []
    for pod in targets:
        runpod_api.terminate_with_retry(pod["id"])
        lines.append(f"- terminated [`{pod['id']}`]({pod_url(pod['id'])})")
    _post(ctx, "\n".join(lines))


def cmd_watchdog(args, ctx) -> None:
    from ci import runpod_api
    from ci.watchdog import decide_terminations

    pods = runpod_api.list_pods()
    doomed = decide_terminations(pods, now=time.time())
    verb = "would terminate" if args.dry_run else "terminating"
    lines = [
        f"{len(pods)} pod(s) total, "
        f"{sum(1 for p in pods if (p.get('name') or '').startswith('factorion-ci-'))} CI pod(s)."
    ]
    for pod, reason in doomed:
        lines.append(f"- {verb} [`{pod['id']}`]({pod_url(pod['id'])}) `{pod.get('name')}`: {reason}")
        if not args.dry_run:
            runpod_api.terminate_with_retry(pod["id"])
    if not doomed:
        lines.append("Nothing to reap.")
    _post(ctx, "\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="/ci", add_help=False)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--gpu-type", default=GPU_FALLBACKS[0])

    sp = sub.add_parser("sft", add_help=False)
    sp.add_argument("--num-samples", type=int, default=None)
    common(sp)

    sp = sub.add_parser("ppo", add_help=False)
    sp.add_argument("--start-from", required=True)
    sp.add_argument("--total-timesteps", type=int, default=None)
    common(sp)

    sp = sub.add_parser("compare", add_help=False)
    sp.add_argument("algo", nargs="?", choices=("sft", "ppo"), default="sft")
    sp.add_argument("--base-ref", default="main")
    sp.add_argument("--seeds", type=int, default=COMPARE_SEEDS_DEFAULT)
    sp.add_argument("--num-samples", type=int, default=COMPARE_NUM_SAMPLES_DEFAULT)
    sp.add_argument("--start-from", default=None)
    sp.add_argument("--total-timesteps", type=int, default=None)
    common(sp)

    sp = sub.add_parser("sweep", add_help=False)
    sp.add_argument("algo", choices=("sft", "ppo"))
    sp.add_argument("--pods", type=int, default=1)
    sp.add_argument("--agents-per-pod", type=int, default=AGENTS_PER_POD_DEFAULT)
    common(sp)

    sub.add_parser("pods", add_help=False)

    sp = sub.add_parser("kill", add_help=False)
    sp.add_argument("pod_id", nargs="?", default="")
    sp.add_argument("--all", action="store_true")

    sp = sub.add_parser("watchdog", add_help=False)
    sp.add_argument("--dry-run", action="store_true")

    sub.add_parser("help", add_help=False)
    return p


def main() -> None:
    body = os.environ["COMMENT_BODY"]
    ctx = {
        "pr": int(os.environ["PR_NUMBER"]),
        "sha": os.environ["HEAD_SHA"],
        "comment_url": os.environ.get("COMMENT_URL", ""),
    }
    tokens, assertions = parse_comment(body)
    ctx["assertions"] = assertions

    if not tokens or tokens[0] == "help":
        _post(ctx, HELP)
        return

    try:
        args = build_parser().parse_args(tokens)
    except SystemExit:
        _post(
            ctx,
            f"## &#x274C; could not parse `/ci {' '.join(tokens)}`\n\n{HELP}",
        )
        raise

    try:
        dispatch = {
            "sft": cmd_sft,
            "ppo": cmd_ppo,
            "compare": cmd_compare,
            "sweep": cmd_sweep,
            "pods": cmd_pods,
            "kill": cmd_kill,
            "watchdog": cmd_watchdog,
        }
        dispatch[args.command](args, ctx)
    except SystemExit:
        raise
    except Exception:
        _post(
            ctx,
            "## &#x274C; `/ci` command failed\n\n"
            f"```\n{traceback.format_exc()[-3000:]}\n```",
        )
        raise


if __name__ == "__main__":
    main()
