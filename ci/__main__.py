"""Factorion CI entry point: `uv run python -m ci <command>`."""

import tyro

from ci import cli


def main() -> None:
    tyro.extras.subcommand_cli_from_dict(
        {
            "sft": cli.sft,
            "ppo": cli.ppo,
            "sweep": cli.sweep,
            "compare": cli.compare,
            "pods": cli.pods,
            "kill": cli.kill,
            "watchdog": cli.watchdog,
            "compare-report": cli.compare_report,
            "sweep-report": cli.sweep_report,
            "history": cli.history,
            "post-pending-reports": cli.post_pending_reports,
        },
        description="Launch and manage factorion GPU training jobs on RunPod.",
    )


if __name__ == "__main__":
    main()
