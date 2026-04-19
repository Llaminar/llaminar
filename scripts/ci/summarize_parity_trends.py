#!/usr/bin/env python3
"""
Generate trend charts for parity test metrics across the last N commits.

Walks <results-root>/<git-hash>/<test-name>/prefill_summary.csv and
decode_steps.csv, sorts commits by directory mtime (newest first), takes
the most recent N, and produces one matplotlib line chart per metric
(cosine, KL, top-1, top-5) with one line per test name.

Output:
  - PNG files under <out-dir>
  - Markdown appended to $GITHUB_STEP_SUMMARY (if set), with each PNG
    embedded as a base64 data URI so the summary is self-contained.

Usage:
    python3 scripts/ci/summarize_parity_trends.py \\
        --results-root tests/v2/integration/parity/results \\
        --out-dir parity-trends \\
        --history 10
"""
from __future__ import annotations

import argparse
import base64
import csv
import os
import sys
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _avg(rows: list[dict[str, str]], key: str) -> float | None:
    vals: list[float] = []
    for r in rows:
        try:
            vals.append(float(r.get(key, "")))
        except (TypeError, ValueError):
            continue
    if not vals:
        return None
    return sum(vals) / len(vals)


def _first(rows: list[dict[str, str]], key: str) -> float | None:
    if not rows:
        return None
    try:
        return float(rows[0].get(key, ""))
    except (TypeError, ValueError):
        return None


# Metric extractors: (test_dir, csv_filename) -> float|None
PrefillExtractor = Callable[[list[dict[str, str]]], float | None]

PREFILL_METRICS: dict[str, tuple[str, PrefillExtractor]] = {
    "Cosine": ("prefill_summary.csv", lambda rows: _first(rows, "lm_head_cosine")),
    "KL Divergence": ("prefill_summary.csv", lambda rows: _first(rows, "lm_head_kl")),
    "Top-1": ("prefill_summary.csv", lambda rows: _first(rows, "lm_head_top1")),
    "Top-5": ("prefill_summary.csv", lambda rows: _first(rows, "lm_head_top5")),
}

DECODE_METRICS: dict[str, tuple[str, PrefillExtractor]] = {
    "Cosine": ("decode_steps.csv", lambda rows: _avg(rows, "cosine")),
    "KL Divergence": ("decode_steps.csv", lambda rows: _avg(rows, "kl_divergence")),
    "Top-1": ("decode_steps.csv", lambda rows: _avg(rows, "top1_overlap")),
    "Top-5": ("decode_steps.csv", lambda rows: _avg(rows, "top5_overlap")),
}


def _select_commits(results_root: Path, history: int) -> list[Path]:
    """Newest -> oldest, then return oldest -> newest for plotting."""
    if not results_root.is_dir():
        return []
    commit_dirs = [d for d in results_root.iterdir() if d.is_dir()]
    commit_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    selected = commit_dirs[:history]
    selected.sort(key=lambda p: p.stat().st_mtime)  # oldest first for x-axis
    return selected


def _collect_series(
    commits: list[Path],
    metrics: dict[str, tuple[str, PrefillExtractor]],
) -> dict[str, dict[str, list[tuple[str, float]]]]:
    """Returns {metric_name: {test_name: [(commit_short, value), ...]}}."""
    out: dict[str, dict[str, list[tuple[str, float]]]] = {m: {} for m in metrics}
    for commit_dir in commits:
        commit_short = commit_dir.name[:8]
        test_dirs = [d for d in commit_dir.iterdir() if d.is_dir()]
        for tdir in test_dirs:
            for metric_name, (csv_name, extractor) in metrics.items():
                rows = _read_csv(tdir / csv_name)
                if not rows:
                    continue
                value = extractor(rows)
                if value is None:
                    continue
                out[metric_name].setdefault(tdir.name, []).append((commit_short, value))
    return out


def _plot_metric(
    metric_name: str,
    series: dict[str, list[tuple[str, float]]],
    out_path: Path,
    phase_label: str,
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

    # Sort tests for deterministic legend; truncate very long names for legend.
    plotted = 0
    for test_name in sorted(series.keys()):
        points = series[test_name]
        xs = [x_idx[c] for c, _ in points]
        ys = [v for _, v in points]
        if not xs:
            continue
        label = test_name if len(test_name) <= 60 else test_name[:57] + "..."
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.0, label=label, alpha=0.85)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return False

    ax.set_title(f"{phase_label} · {metric_name} trend across last {len(x_order)} commits")
    ax.set_xlabel("Commit (oldest → newest)")
    ax.set_ylabel(metric_name)
    ax.set_xticks(list(range(len(x_order))))
    ax.set_xticklabels(x_order, rotation=30, ha="right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Legend outside if too many lines.
    if plotted > 8:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=7,
            frameon=False,
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
    """Return an HTML img tag with the PNG inlined as base64."""
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f'<img alt="{path.stem}" src="data:image/png;base64,{b64}" />'


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-root",
        default="tests/v2/integration/parity/results",
        help="Path to parity results root (default: %(default)s)",
    )
    p.add_argument(
        "--out-dir",
        default="parity-trends",
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
            "## Parity Trends\n\n"
            f"_No commit subdirectories found under `{results_root}`._\n"
        )
        sys.stdout.write(msg)
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a", encoding="utf-8") as fh:
                fh.write(msg)
        return 0

    md: list[str] = ["## Parity Trends", ""]
    md.append(
        f"_Last **{len(commits)}** commits "
        f"(`{commits[0].name[:8]}` → `{commits[-1].name[:8]}`)._"
    )
    md.append("")

    for phase_label, metrics in (("Prefill", PREFILL_METRICS), ("Decode", DECODE_METRICS)):
        series_by_metric = _collect_series(commits, metrics)
        section_md: list[str] = []
        for metric_name in metrics:
            series = series_by_metric.get(metric_name, {})
            slug = (
                f"{phase_label.lower()}_"
                f"{metric_name.lower().replace(' ', '_')}.png"
            )
            png_path = out_dir / slug
            ok = _plot_metric(metric_name, series, png_path, phase_label)
            if not ok:
                continue
            section_md.append(f"#### {phase_label} · {metric_name}")
            section_md.append("")
            section_md.append(_embed_png(png_path))
            section_md.append("")

        if section_md:
            md.append(f"### {phase_label}")
            md.append("")
            md.extend(section_md)

    out_md = "\n".join(md) + "\n"
    sys.stdout.write(out_md)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(out_md)

    return 0


if __name__ == "__main__":
    sys.exit(main())
