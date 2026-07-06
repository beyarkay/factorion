"""Minimal GitHub REST helpers for CI: PR comments and commit statuses.

Used by the /ci comment workflow and the reporter cron; authenticates with
the workflow's GITHUB_TOKEN. Kept to plain `requests` so nothing else is
needed on a runner.

Required env vars: GITHUB_TOKEN, GITHUB_REPOSITORY (owner/repo — set
automatically on GitHub Actions runners).
"""

from __future__ import annotations

import os

import requests

API = "https://api.github.com"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _repo() -> str:
    return os.environ["GITHUB_REPOSITORY"]


def post_pr_comment(pr_number: int, body: str) -> None:
    r = requests.post(
        f"{API}/repos/{_repo()}/issues/{pr_number}/comments",
        headers=_headers(),
        json={"body": body},
        timeout=30,
    )
    r.raise_for_status()


def list_pr_comment_bodies(pr_number: int, max_pages: int = 10) -> list[str]:
    bodies: list[str] = []
    for page in range(1, max_pages + 1):
        r = requests.get(
            f"{API}/repos/{_repo()}/issues/{pr_number}/comments",
            headers=_headers(),
            params={"per_page": 100, "page": page},
            timeout=30,
        )
        r.raise_for_status()
        batch = r.json()
        bodies.extend(c.get("body", "") for c in batch)
        if len(batch) < 100:
            break
    return bodies


def set_commit_status(sha: str, state: str, context: str, description: str) -> None:
    """state: success | failure | error | pending. Shows as a PR check."""
    r = requests.post(
        f"{API}/repos/{_repo()}/statuses/{sha}",
        headers=_headers(),
        json={"state": state, "context": context, "description": description[:140]},
        timeout=30,
    )
    r.raise_for_status()
