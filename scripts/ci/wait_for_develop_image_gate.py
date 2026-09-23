#!/usr/bin/env python3
"""Wait for the exact develop push to publish its tested runtime image pair.

The PR gate cannot use a mutable ``develop`` tag or assume GitHub scheduled
push CI before pull-request CI. This metadata-only wait consumes no accelerator
lease; the later E2E job pulls source-pinned tags and reauthenticates labels.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time


def github_json(endpoint: str, *fields: str) -> dict:
    """Read GitHub API metadata with the workflow's short-lived token."""
    command = ["gh", "api", "--method", "GET", endpoint]
    for field in fields:
        command.extend(("-f", field))
    return json.loads(subprocess.check_output(command, text=True))


def matching_ci_runs(response: dict, revision: str) -> list[dict]:
    """Keep only exact develop push runs; never accept a nearby branch tag."""
    return [run for run in response.get("workflow_runs", [])
            if run.get("head_sha") == revision and run.get("head_branch") == "develop"
            and run.get("event") == "push"]


def wait_for_image_gate(repository: str, revision: str, timeout_seconds: int,
                        poll_seconds: int = 20) -> int:
    """Return the successful CI run ID or fail on a final red exact-source run."""
    branch = github_json(f"repos/{repository}/branches/develop")
    if branch.get("commit", {}).get("sha") != revision:
        raise ValueError("PR head is no longer the current develop branch tip")
    deadline = time.monotonic() + timeout_seconds
    observed = None
    while True:
        response = github_json(
            f"repos/{repository}/actions/workflows/ci.yml/runs",
            "branch=develop", f"head_sha={revision}", "event=push", "per_page=20")
        runs = matching_ci_runs(response, revision)
        complete = [run for run in runs if run.get("status") == "completed"]
        passed = [run for run in complete if run.get("conclusion") == "success"]
        if passed:
            identifier = int(passed[0]["id"])
            print(f"[master-pr] exact develop image gate passed run={identifier} revision={revision}",
                  flush=True)
            return identifier
        active = [run for run in runs if run.get("status") != "completed"]
        if complete and not active:
            raise ValueError("exact develop image gate completed without publishing both ISAs")
        state = (runs[0].get("status"), runs[0].get("conclusion")) if runs else ("queued", None)
        if state != observed:
            print(f"[master-pr] waiting for exact develop image gate revision={revision} "
                  f"state={state[0]} conclusion={state[1]}", flush=True)
            observed = state
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("exact develop image gate did not complete within the PR wait budget")
        time.sleep(min(poll_seconds, remaining))


def main(argv: list[str] | None = None) -> int:
    """Validate immutable identity before waiting for the ordinary develop CI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=4 * 60 * 60)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repository):
        parser.error("--repository requires OWNER/REPO")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision requires a full Git SHA")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    wait_for_image_gate(args.repository, args.revision, args.timeout_seconds)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, TimeoutError, OSError, subprocess.SubprocessError) as error:
        print(f"[master-pr] ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
