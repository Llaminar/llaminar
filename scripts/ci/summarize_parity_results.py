#!/usr/bin/env python3
"""
Summarize parity test results into a Markdown table for GitHub Actions.

Walks tests/v2/integration/parity/results/<git-hash>/ for per-test directories.
Each directory typically contains:
  - prefill_summary.csv  : LM_HEAD aggregate (cosine, KL, top-1, top-5, passed)
  - decode_steps.csv     : per-decode-step LM_HEAD metrics (averaged)

Writes a Markdown table to:
  - stdout (always), and
  - $GITHUB_STEP_SUMMARY when set (GitHub Actions step summary).

Usage:
    python3 scripts/ci/summarize_parity_results.py [--results-root DIR] \
        [--artifact-name NAME]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def fmt_float(v: str | None, digits: int = 4) -> str:
    if v is None or v == "":
        return "-"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return v


def fmt_pass(v: str | None) -> str:
    s = (v or "").strip().lower()
    if s in ("true", "1", "yes", "pass", "passed"):
        return "✅"
    if s in ("false", "0", "no", "fail", "failed"):
        return "❌"
    return "❔"


def passed_bool(v: str | None) -> bool | None:
    s = (v or "").strip().lower()
    if s in ("true", "1", "yes", "pass", "passed"):
        return True
    if s in ("false", "0", "no", "fail", "failed"):
        return False
    return None


def avg(rows: list[dict[str, str]], key: str) -> str:
    vals = []
    for r in rows:
        try:
            vals.append(float(r.get(key, "")))
        except (TypeError, ValueError):
            pass
    if not vals:
        return "-"
    return f"{sum(vals) / len(vals):.4f}"


def all_passed(rows: list[dict[str, str]], key: str = "passed") -> str:
    if not rows:
        return "-"
    bools = [passed_bool(r.get(key)) for r in rows]
    if any(b is False for b in bools):
        return "❌"
    if all(b is True for b in bools):
        return "✅"
    return "❔"


def summarize(results_root: Path, artifact_name: str) -> tuple[str, int, int]:
    """Returns (markdown, total_count, fail_count)."""
    if not results_root.is_dir():
        return (f"_No parity results directory found at `{results_root}`._\n", 0, 0)

    # Per-commit subdirs: tests/.../results/<git-hash>/<test-dir>/...
    commit_dirs = sorted([d for d in results_root.iterdir() if d.is_dir()])
    if not commit_dirs:
        return (f"_No commit subdirectories under `{results_root}`._\n", 0, 0)

    out: list[str] = []
    total = 0
    failed = 0
    for commit_dir in commit_dirs:
        test_dirs = sorted([d for d in commit_dir.iterdir() if d.is_dir()])
        if not test_dirs:
            continue

        out.append(f"### Commit `{commit_dir.name}`\n")
        out.append("")

        # Prefill table
        prefill_rows: list[str] = []
        for tdir in test_dirs:
            summary = read_csv(tdir / "prefill_summary.csv")
            if not summary:
                continue
            row = summary[0]
            csv_rel = (tdir / "prefill_summary.csv").relative_to(results_root.parent)
            status = fmt_pass(row.get("overall_passed"))
            total += 1
            if passed_bool(row.get("overall_passed")) is False:
                failed += 1
            prefill_rows.append(
                f"| `{tdir.name}` "
                f"| {fmt_float(row.get('lm_head_cosine'))} "
                f"| {fmt_float(row.get('lm_head_kl'))} "
                f"| {fmt_float(row.get('lm_head_top1'), 2)} "
                f"| {fmt_float(row.get('lm_head_top5'), 2)} "
                f"| {row.get('total_layers_passed', '-')} "
                f"| {status} "
                f"| `{csv_rel}` |"
            )

        if prefill_rows:
            out.append("#### Prefill")
            out.append("")
            out.append(
                "| Test | Cosine | KL Div | Top-1 | Top-5 | Layers Pass | Status | CSV |"
            )
            out.append(
                "|------|-------:|-------:|------:|------:|------------:|:------:|-----|"
            )
            out.extend(prefill_rows)
            out.append("")

        # Decode table (aggregated across steps)
        decode_rows: list[str] = []
        for tdir in test_dirs:
            steps = read_csv(tdir / "decode_steps.csv")
            if not steps:
                continue
            csv_rel = (tdir / "decode_steps.csv").relative_to(results_root.parent)
            status = all_passed(steps, "passed")
            total += 1
            if status == "❌":
                failed += 1
            decode_rows.append(
                f"| `{tdir.name}` "
                f"| {len(steps)} "
                f"| {avg(steps, 'cosine')} "
                f"| {avg(steps, 'kl_divergence')} "
                f"| {avg(steps, 'top1_overlap')} "
                f"| {avg(steps, 'top5_overlap')} "
                f"| {status} "
                f"| `{csv_rel}` |"
            )

        if decode_rows:
            out.append("#### Decode (averaged across steps)")
            out.append("")
            out.append(
                "| Test | Steps | Avg Cosine | Avg KL | Avg Top-1 | Avg Top-5 | Status | CSV |"
            )
            out.append(
                "|------|------:|-----------:|-------:|----------:|----------:|:------:|-----|"
            )
            out.extend(decode_rows)
            out.append("")

    if not out:
        return (f"_No parity CSV summaries found under `{results_root}`._\n", 0, 0)

    header = [
        "## Parity Test Results",
        "",
        f"_Detailed CSVs available in the **`{artifact_name}`** workflow artifact._",
        "",
    ]
    return ("\n".join(header + out) + "\n", total, failed)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-root",
        default="tests/v2/integration/parity/results",
        help="Path to parity results root (default: %(default)s)",
    )
    p.add_argument(
        "--artifact-name",
        default="parity-results",
        help="Artifact name to reference in the summary (default: %(default)s)",
    )
    args = p.parse_args()

    md, total, failed = summarize(Path(args.results_root), args.artifact_name)
    sys.stdout.write(md)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(md)
            fh.write(f"\n_Total tests: **{total}**, Failed: **{failed}**_\n")

    return 0  # informational only — never fail the step


if __name__ == "__main__":
    sys.exit(main())
