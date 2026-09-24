#!/usr/bin/env python3
"""Post an idempotent red/amber/green benchmark receipt on a master PR.

The authoritative pass/fail decision stays in the image-bound benchmark
runner. This presentation layer reads only its completed, authenticated
aggregate report; it cannot turn a red measurement green or advance a mark.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
import sys

MARKER = "<!-- llaminar-pr-benchmark-rag -->"
ISAS = ("AVX512", "AVX2")


def traffic_light(comparison: dict) -> str:
    """Classify one measured phase using the ratchet's own decision."""
    if not comparison["passed"]:
        return "🔴"
    prior = comparison["high_water"]
    if prior is not None and comparison["current"] < prior:
        return "🟡"
    return "🟢"


def render(report: dict | None, revision: str, run_url: str) -> str:
    """Show every cell/phase and its high-water delta, not a global average."""
    lines = [MARKER, "## Llaminar benchmark PR gate", "",
             f"Source: `{revision}` · [workflow evidence]({run_url})", ""]
    if report is None:
        return "\n".join(lines + ["🔴 Benchmark evidence is unavailable; inspect the workflow logs.", ""])
    if (report.get("source", {}).get("revision") != revision
            or set(report.get("variants", {})) != set(ISAS)
            or report.get("scope") != "full-http-e2e-and-benchmarks"
            or not isinstance(report.get("regression_threshold_pct"), (int, float))):
        raise ValueError("RAG report does not belong to this exact PR image pair")
    comparisons = [comparison for isa in ISAS
                   for comparison in report["variants"][isa]["comparisons"]]
    if not comparisons:
        raise ValueError("RAG report has no measured benchmark phases")
    colors = Counter(traffic_light(comparison) for comparison in comparisons)
    all_passed = all(comparison["passed"] for comparison in comparisons)
    if report.get("passed") is not all_passed:
        raise ValueError("RAG status disagrees with canonical benchmark evidence")
    lines.extend([f"{'🟢 PASS' if all_passed else '🔴 BLOCKED'} · "
                  f"{colors['🟢']} green / {colors['🟡']} amber / {colors['🔴']} red "
                  f"phase measurements across {len(comparisons) // 2} ISA-cells.",
                  f"Amber is below a prior high-water mark but inside the "
                  f"{report['regression_threshold_pct']}% noise tolerance; "
                  "red exceeds that tolerance and blocks the PR.", ""])
    for isa in ISAS:
        rows = report["variants"][isa]["comparisons"]
        lines.extend([f"<details><summary>{isa}: {len(rows) // 2} cells</summary>", "",
                      "| RAG | Cell | Phase | Measured tok/s | High water tok/s | Delta |",
                      "|:---:|---|---|---:|---:|---:|"])
        for row in rows:
            prior = row["high_water"]
            delta = "new" if prior is None else f"{100 * (row['current'] / prior - 1):+.1f}%"
            cell = row["case"].rsplit("/", 1)[-1].replace("|", "\\|")
            lines.append(f"| {traffic_light(row)} | `{cell}` | {row['phase']} | "
                         f"{row['current']:.2f} | "
                         f"{'new' if prior is None else f'{prior:.2f}'} | {delta} |")
        lines.extend(["", "</details>", ""])
    return "\n".join(lines)


def github_json(endpoint: str, *, method: str = "GET", payload: dict | None = None) -> object:
    """Use the workflow token without putting a comment body on the command line."""
    command = ["gh", "api", "--method", method, endpoint]
    if payload is not None:
        command.extend(("--input", "-"))
    completed = subprocess.run(command, input=json.dumps(payload) if payload else None,
                               text=True, capture_output=True, check=True)
    return json.loads(completed.stdout)


def publish(repository: str, number: int, body: str) -> None:
    """Update our prior bot comment, otherwise create exactly one PR comment."""
    endpoint = f"repos/{repository}/issues/{number}/comments"
    comments = []
    for page in range(1, 101):
        batch = github_json(endpoint + f"?per_page=100&page={page}")
        if not isinstance(batch, list):
            raise ValueError("PR comment inventory is not a list")
        comments.extend(batch)
        if len(batch) < 100:
            break
    else:
        raise ValueError("PR comment inventory exceeded the bounded review window")
    owned = [comment for comment in comments if MARKER in comment.get("body", "")
             and comment.get("user", {}).get("login") == "github-actions[bot]"]
    if owned:
        github_json(f"repos/{repository}/issues/comments/{owned[-1]['id']}",
                    method="PATCH", payload={"body": body})
    else:
        github_json(endpoint, method="POST", payload={"body": body})


def main(argv: list[str] | None = None) -> int:
    """Render a durable PR summary even when benchmark execution was red."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repository):
        parser.error("--repository requires OWNER/REPO")
    if args.pr <= 0 or not re.fullmatch(r"[0-9a-f]{40}", args.source_revision):
        parser.error("--pr and --source-revision require exact positive identities")
    report = json.loads(args.report.read_text()) if args.report.exists() else None
    body = render(report, args.source_revision, args.run_url)
    publish(args.repository, args.pr, body)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a") as output:
            output.write(body + "\n")
    print("[benchmark-rag] posted PR measurement table", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        print(f"[benchmark-rag] ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
