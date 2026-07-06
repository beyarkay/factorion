"""Dispatcher for `/ci ...` PR comments (invoked by ci-command.yml).

The workflow hands over the comment body plus PR context via env vars; this
module parses the command, launches pods with the repo's secrets, and posts
the outcome back to the PR — comments are the backbone of CI reporting.

Comment grammar (first line + optional assert lines):

    /ci sft [--num-samples N]
    /ci ppo --start-from <wandb_run_id> [--total-timesteps N]
    /ci compare [--algo sft|ppo] [--seeds N] [--num-samples N]
                [--start-from ID] [--total-timesteps N]
    assert pr:val/thput > main:val/thput
    assert pr:val/acc >= 0.5
    /ci compare-report            (re-post the compare report for this PR head)
    /ci sweep-sft [--pods N] [--agents-per-pod N]
    /ci sweep-ppo [--pods N] [--agents-per-pod N]
    /ci pods | /ci kill --all | /ci watchdog | /ci help

Env: COMMENT_BODY, PR_NUMBER, HEAD_SHA, GITHUB_TOKEN, GITHUB_REPOSITORY,
RUNPOD_API_KEY, WANDB_API_KEY.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import shlex
import sys
import time
import traceback

from ci import github_api
from ci.config import (
    COMPARE_NUM_SAMPLES_DEFAULT,
    COMPARE_SEEDS_DEFAULT,
    GPU_FALLBACKS,
    WANDB_PROJECT,
    PpoJob,
    SftJob,
    compare_group,
    ppo_budget_seconds,
    sft_budget_seconds,
)
from ci.launch import create_sweep, launch, launch_compare, resolve_ref

COMPARE_STATUS_CONTEXT = "factorion-ci/compare"
# Compare waits in-workflow so the commit status lands promptly; cap the wait
# under the workflow's own timeout.
MAX_WAIT_SECONDS = 320 * 60

USAGE = __doc__ or ""


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


def _wandb_runs_url(sha7: str) -> str:
    return f"https://wandb.ai/?project={WANDB_PROJECT} (runs tagged `sha:{sha7}`)"


def _launched_comment(title: str, infos: list[dict], footer: str = "") -> str:
    lines = [f"## &#x1F680; {title}", ""]
    for info in infos:
        lines.append(f"- pod `{info['pod_id']}` (`{info['pod_name']}`)")
    lines += [
        "",
        "Pods terminate themselves when done (EXIT trap + deadline timer + "
        "6-hourly watchdog).",
    ]
    if footer:
        lines += ["", footer]
    return "\n".join(lines)


def cmd_sft(args, ctx) -> None:
    job = SftJob(
        sha=ctx["sha"], num_samples=args.num_samples, extra_tags=[f"pr:{ctx['pr']}"]
    )
    info = launch(job, args.gpu_type, wait=False)
    github_api.post_pr_comment(
        ctx["pr"],
        _launched_comment(
            f"SFT run launched at `{ctx['sha'][:7]}`",
            [info],
            footer=f"Results land here as a comment when the run finishes. {_wandb_runs_url(ctx['sha'][:7])}",
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
    github_api.post_pr_comment(
        ctx["pr"],
        _launched_comment(
            f"PPO run launched at `{ctx['sha'][:7]}` (from `{args.start_from}`)",
            [info],
            footer=f"Results land here as a comment when the run finishes. {_wandb_runs_url(ctx['sha'][:7])}",
        ),
    )


def _compare_wait_seconds(args) -> int:
    if args.algo == "sft":
        budget = sft_budget_seconds(args.num_samples, 1)
    else:
        from training_config import PpoArgs

        budget = ppo_budget_seconds(args.total_timesteps or PpoArgs().total_timesteps)
    return min(budget + 45 * 60, MAX_WAIT_SECONDS)


def _post_compare_outcome(ctx, assertions: list[str]) -> None:
    from ci.report import compare_report

    md, ok = compare_report(
        base_group=compare_group(ctx["sha"], "base"),
        test_group=compare_group(ctx["sha"], "test"),
        assertions=assertions,
    )
    github_api.post_pr_comment(ctx["pr"], md)
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
    infos = launch_compare(
        algo=args.algo,
        sha=ctx["sha"],
        base_sha=resolve_ref(args.base_ref),
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
    github_api.post_pr_comment(
        ctx["pr"],
        _launched_comment(
            f"Compare launched: `{ctx['sha'][:7]}` vs `{args.base_ref}` "
            f"({args.algo}, {args.seeds} seeds x 2 sides, one pod per run)",
            infos,
            footer=f"Assertions:\n{assertion_note}",
        ),
    )
    github_api.set_commit_status(
        ctx["sha"], "pending", COMPARE_STATUS_CONTEXT, "compare runs in flight"
    )

    from ci.report import wait_for_groups

    wait_for_groups(
        base_group=compare_group(ctx["sha"], "base"),
        test_group=compare_group(ctx["sha"], "test"),
        expect_each=args.seeds,
        timeout_seconds=_compare_wait_seconds(args),
    )
    _post_compare_outcome(ctx, ctx["assertions"])


def cmd_compare_report(args, ctx) -> None:
    _post_compare_outcome(ctx, ctx["assertions"])


def cmd_sweep(args, ctx, algo: str) -> None:
    sweep_path = create_sweep(algo, ctx["sha"])
    from ci.config import SweepJob

    infos = [
        launch(
            SweepJob(
                sha=ctx["sha"],
                algo=algo,
                sweep_path=sweep_path,
                agents_per_pod=args.agents_per_pod,
            ),
            args.gpu_type,
            wait=False,
        )
        for _ in range(args.pods)
    ]
    entity, project, sweep_id = sweep_path.split("/")
    sweep_url = f"https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}"
    github_api.post_pr_comment(
        ctx["pr"],
        _launched_comment(
            f"{algo.upper()} sweep launched at `{ctx['sha'][:7]}` "
            f"({args.pods} pod(s) x {args.agents_per_pod} agents)",
            infos,
            footer=f"[View sweep on W&B]({sweep_url}) — report follows when the sweep drains.",
        ),
    )

    # Wait for the sweep to drain (run_cap from the yaml), then report.
    from ci.report import sweep_report

    import wandb

    deadline = time.time() + MAX_WAIT_SECONDS
    while time.time() < deadline:
        sweep = wandb.Api().sweep(sweep_path)
        if str(sweep.state).upper() in ("FINISHED", "CANCELLED", "CANCELED"):
            break
        print(f"sweep state: {sweep.state}", flush=True)
        time.sleep(300)
    github_api.post_pr_comment(ctx["pr"], sweep_report(sweep_path))


def _captured(fn, *args, **kwargs) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args, **kwargs)
    return buf.getvalue()


def cmd_pods(args, ctx) -> None:
    from ci import cli

    out = _captured(cli.pods) or "(no output)"
    github_api.post_pr_comment(ctx["pr"], f"```\n{out}\n```")


def cmd_kill(args, ctx) -> None:
    from ci import cli

    out = _captured(cli.kill, pod_id=args.pod_id, all=args.all) or "(no output)"
    github_api.post_pr_comment(ctx["pr"], f"```\n{out}\n```")


def cmd_watchdog(args, ctx) -> None:
    from ci import watchdog

    out = _captured(watchdog.run, dry_run=args.dry_run) or "(no output)"
    github_api.post_pr_comment(ctx["pr"], f"```\n{out}\n```")


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
    sp.add_argument("--algo", choices=("sft", "ppo"), default="sft")
    sp.add_argument("--base-ref", default="main")
    sp.add_argument("--seeds", type=int, default=COMPARE_SEEDS_DEFAULT)
    sp.add_argument("--num-samples", type=int, default=COMPARE_NUM_SAMPLES_DEFAULT)
    sp.add_argument("--start-from", default=None)
    sp.add_argument("--total-timesteps", type=int, default=None)
    common(sp)

    sub.add_parser("compare-report", add_help=False)

    for name in ("sweep-sft", "sweep-ppo"):
        sp = sub.add_parser(name, add_help=False)
        sp.add_argument("--pods", type=int, default=1)
        sp.add_argument("--agents-per-pod", type=int, default=5)
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
    }
    tokens, assertions = parse_comment(body)
    ctx["assertions"] = assertions

    if not tokens or tokens[0] == "help":
        github_api.post_pr_comment(ctx["pr"], f"```\n{USAGE}\n```")
        return

    try:
        args = build_parser().parse_args(tokens)
    except SystemExit:
        github_api.post_pr_comment(
            ctx["pr"],
            f"## &#x274C; could not parse `/ci {' '.join(tokens)}`\n\n```\n{USAGE}\n```",
        )
        raise

    try:
        dispatch = {
            "sft": cmd_sft,
            "ppo": cmd_ppo,
            "compare": cmd_compare,
            "compare-report": cmd_compare_report,
            "sweep-sft": lambda a, c: cmd_sweep(a, c, "sft"),
            "sweep-ppo": lambda a, c: cmd_sweep(a, c, "ppo"),
            "pods": cmd_pods,
            "kill": cmd_kill,
            "watchdog": cmd_watchdog,
        }
        dispatch[args.command](args, ctx)
    except SystemExit:
        raise
    except Exception:
        github_api.post_pr_comment(
            ctx["pr"],
            "## &#x274C; `/ci` command failed\n\n"
            f"```\n{traceback.format_exc()[-3000:]}\n```",
        )
        raise


if __name__ == "__main__":
    main()
