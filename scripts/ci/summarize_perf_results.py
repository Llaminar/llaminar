#!/usr/bin/env python3
"""
Render benchmark JSON files as Markdown tables for the GitHub Actions summary.

Looks for benchmark_results.json files under --results-root (which contains
<git-hash>/benchmark_results.json subdirs). Picks the most recent run and
renders one table per model: device | prefill (tok/s) | decode (tok/s) |
prefill Δ vs baseline | decode Δ vs baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def fmt_tok(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.1f}"
    except (TypeError, ValueError):
        return "—"


def fmt_delta(cur, base) -> str:
    if cur is None or base in (None, 0):
        return "—"
    try:
        delta_pct = (float(cur) - float(base)) / float(base) * 100.0
    except (TypeError, ValueError, ZeroDivisionError):
        return "—"
    if abs(delta_pct) < 0.1:
        return "≈ 0%"
    arrow = "▲" if delta_pct > 0 else "▼"
    return f"{arrow} {delta_pct:+.1f}%"


def render_table(model: dict) -> list[str]:
    out = [f"### {model.get('name', '?')}", ""]
    out.append(f"_Model file: `{model.get('model', '?')}`_")
    out.append("")
    out.append("| Device | Prefill (tok/s) | Decode (tok/s) | Prefill Δ | Decode Δ |")
    out.append("|--------|----------------:|---------------:|----------:|---------:|")
    for d in model.get("devices", []):
        out.append(
            f"| `{d.get('device', '?')}` "
            f"| {fmt_tok(d.get('prefill_tok_s'))} "
            f"| {fmt_tok(d.get('decode_tok_s'))} "
            f"| {fmt_delta(d.get('prefill_tok_s'), d.get('baseline_prefill_tok_s'))} "
            f"| {fmt_delta(d.get('decode_tok_s'), d.get('baseline_decode_tok_s'))} |"
        )
    out.append("")
    return out


def find_latest_json(results_root: Path) -> Path | None:
    if not results_root.is_dir():
        return None
    candidates = list(results_root.glob("*/benchmark_results.json"))
    if not candidates:
        # Also try a flat file at the root.
        flat = results_root / "benchmark_results.json"
        if flat.is_file():
            return flat
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-root",
        default="benchmark_results",
        help="Path containing <commit>/benchmark_results.json subdirs",
    )
    args = p.parse_args()

    json_path = find_latest_json(Path(args.results_root))
    if json_path is None:
        msg = (
            f"## Performance Results\n\n"
            f"_No benchmark JSON found under `{args.results_root}`._\n"
        )
        sys.stdout.write(msg)
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a", encoding="utf-8") as fh:
                fh.write(msg)
        return 0

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"## Performance Results\n\n_Failed to parse `{json_path}`: {exc}_\n"
        sys.stdout.write(msg)
        return 0

    out: list[str] = [
        "## Performance Results",
        "",
        f"_Commit: `{data.get('commit', '?')}` · {data.get('timestamp', '')}_",
        "",
    ]
    for model in data.get("models", []):
        out.extend(render_table(model))

    md = "\n".join(out) + "\n"
    sys.stdout.write(md)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(md)

    return 0


if __name__ == "__main__":
    sys.exit(main())
