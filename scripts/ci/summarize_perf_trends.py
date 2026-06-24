#!/usr/bin/env python3
"""
Generate trend charts for benchmark results across the last N commits.

Walks <results-root>/<git-hash>/benchmark_results.csv, sorts commits by
directory mtime (newest first), takes the most recent N (default 10),
and produces one matplotlib line chart per metric (prefill tok/s,
decode tok/s) with one line per `model:device` configuration.

Output:
  - PNG files under <out-dir>
  - Markdown appended to $GITHUB_STEP_SUMMARY (if set), with each PNG
    embedded as a base64 data URI so the summary is self-contained.

Usage:
    python3 scripts/ci/summarize_perf_trends.py \\
        --results-root benchmark_results \\
        --out-dir perf-trends \\
        --history 10
"""
from __future__ import annotations

import argparse
import base64
import csv
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _select_commits(results_root: Path, history: int) -> list[Path]:
    """Newest -> oldest, then return oldest -> newest for plotting."""
    if not results_root.is_dir():
        return []
    commit_dirs = [
        d for d in results_root.iterdir()
        if d.is_dir() and (d / "benchmark_results.csv").is_file()
    ]
    commit_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    selected = commit_dirs[:history]
    selected.sort(key=lambda p: p.stat().st_mtime)  # oldest first for x-axis
    return selected


def _to_float(s: str | None) -> float | None:
    if s is None or s == "" or s.lower() == "null":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _is_direct_cpu_device(device: str) -> bool:
    return device == "cpu" or device.startswith("cpu:")


def _device_label(row: dict[str, str]) -> str:
    device = row.get("device", "?")
    cpu_isa = row.get("cpu_isa", "")
    if cpu_isa and _is_direct_cpu_device(device):
        return f"{device}[{cpu_isa}]"
    return device


def _collect_series(
    commits: list[Path],
    metric_col: str,
) -> dict[str, list[tuple[str, float]]]:
    """Returns {label: [(commit_short, value), ...]} where label = 'Model · device'."""
    out: dict[str, list[tuple[str, float]]] = {}
    for commit_dir in commits:
        commit_short = commit_dir.name[:8]
        rows = _read_csv(commit_dir / "benchmark_results.csv")
        for r in rows:
            value = _to_float(r.get(metric_col))
            if value is None:
                continue
            model_name = r.get("model_name", "?")
            device = _device_label(r)
            label = f"{model_name} · {device}"
            out.setdefault(label, []).append((commit_short, value))
    return out


def _plot_metric(
    metric_label: str,
    series: dict[str, list[tuple[str, float]]],
    out_path: Path,
) -> bool:
    """Render a single line chart. Returns True if any data was plotted."""
    if not series:
        return False

    # Build a stable x-axis from the union of commits (in insertion order).
    x_order: list[str] = []
    seen = set()
    for points in series.values():
        for c, _ in points:
            if c not in seen:
                seen.add(c)
                x_order.append(c)
    if not x_order:
        return False

    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.18 * len(series) + 4)))
    x_idx = {c: i for i, c in enumerate(x_order)}

    plotted = 0
    for label in sorted(series.keys()):
        points = series[label]
        xs = [x_idx[c] for c, _ in points]
        ys = [v for _, v in points]
        if not xs:
            continue
        legend_label = label if len(label) <= 60 else label[:57] + "..."
        ax.plot(
            xs, ys, marker="o", markersize=4, linewidth=1.2,
            label=legend_label, alpha=0.9,
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return False

    ax.set_title(f"{metric_label} trend across last {len(x_order)} commits")
    ax.set_xlabel("Commit (oldest → newest)")
    ax.set_ylabel(metric_label)
    ax.set_xticks(list(range(len(x_order))))
    ax.set_xticklabels(x_order, rotation=30, ha="right", fontsize=8)
    ax.grid(True, alpha=0.3)

    if plotted > 8:
        ax.legend(
            loc="center left", bbox_to_anchor=(1.01, 0.5),
            fontsize=7, frameon=False,
        )
        fig.tight_layout(rect=(0, 0, 0.78, 1))
    else:
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return True


def _embed_png(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f'<img alt="{path.stem}" src="data:image/png;base64,{b64}" />'


METRICS: list[tuple[str, str, str]] = [
    # (column_name, axis_label, output_slug)
    ("prefill_tok_s", "Prefill (tok/s)", "prefill_tok_s.png"),
    ("decode_tok_s", "Decode (tok/s)", "decode_tok_s.png"),
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-root",
        default="benchmark_results",
        help="Path to benchmark results root (default: %(default)s)",
    )
    p.add_argument(
        "--out-dir",
        default="perf-trends",
        help="Directory to write trend PNGs to (default: %(default)s)",
    )
    p.add_argument(
        "--history",
        type=int,
        default=10,
        help="Number of most-recent commits to include (default: %(default)s)",
    )
    args = p.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    commits = _select_commits(results_root, args.history)
    if not commits:
        msg = (
            "## Performance Trends\n\n"
            f"_No commit subdirectories with `benchmark_results.csv` found "
            f"under `{results_root}`._\n"
        )
        sys.stdout.write(msg)
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a", encoding="utf-8") as fh:
                fh.write(msg)
        return 0

    md: list[str] = ["## Performance Trends", ""]
    md.append(
        f"_Last **{len(commits)}** commits "
        f"(`{commits[0].name[:8]}` → `{commits[-1].name[:8]}`)._"
    )
    md.append("")

    for col, axis_label, slug in METRICS:
        series = _collect_series(commits, col)
        png_path = out_dir / slug
        if not _plot_metric(axis_label, series, png_path):
            continue
        md.append(f"#### {axis_label}")
        md.append("")
        md.append(_embed_png(png_path))
        md.append("")

    out_md = "\n".join(md) + "\n"
    sys.stdout.write(out_md)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(out_md)

    return 0


if __name__ == "__main__":
    sys.exit(main())
