#!/usr/bin/env python3
"""Merge ISA-scoped benchmark result artifacts into one commit result.

The benchmark hook emits one result JSON per run. CI runs the hook against both
the full AVX2 and full AVX512 runtime images. This helper merges those artifacts
into the canonical tracked files:

  benchmark_results/<short-sha>/benchmark_results.json
  benchmark_results/<short-sha>/benchmark_results.csv

Direct CPU devices keep one row per ISA. Non-CPU devices keep one row, preferring
the requested ISA run to avoid duplicate GPU trend lines.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def is_direct_cpu_device(device: str) -> bool:
    return device == "cpu" or device.startswith("cpu:")


def load_jsons(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        if item.is_file() and item.name == "benchmark_results.json":
            paths.append(item)
        elif item.is_dir():
            paths.extend(sorted(item.rglob("benchmark_results.json")))
    return sorted(dict.fromkeys(paths))


def iter_rows(data: dict[str, Any], source_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_runs = data.get("runs")
    runs: list[dict[str, Any]] = []
    if isinstance(raw_runs, list):
        for raw_run in raw_runs:
            if not isinstance(raw_run, dict):
                continue
            run = dict(raw_run)
            run.setdefault("source", data.get("source", "unknown"))
            run.setdefault("image", data.get("image"))
            run.setdefault("cpu_isa", data.get("container_cpu_isa"))
            runs.append(run)

    if not runs:
        runs = [
            {
                "source": data.get("source", "unknown"),
                "image": data.get("image"),
                "cpu_isa": data.get("container_cpu_isa"),
            }
        ]

    rows: list[dict[str, Any]] = []
    for model in data.get("models", []):
        model_name = model.get("name", "?")
        model_path = model.get("model", "?")
        for dev in model.get("devices", []):
            row = dict(dev)
            row["model_name"] = model_name
            row["model_path"] = model_path
            row.setdefault("source", data.get("source", "unknown"))
            row.setdefault("image", data.get("image"))
            row.setdefault("container_cpu_isa", data.get("container_cpu_isa"))
            row.setdefault("cpu_isa", None)
            row["source_path"] = str(source_path)
            rows.append(row)
    return list(runs), rows


def row_preference_score(row: dict[str, Any], preferred_non_cpu_isa: str) -> tuple[int, str]:
    container_isa = str(row.get("container_cpu_isa") or "")
    preferred = 1 if container_isa == preferred_non_cpu_isa else 0
    return preferred, str(row.get("source_path") or "")


def merge_rows(all_rows: list[dict[str, Any]], preferred_non_cpu_isa: str) -> list[dict[str, Any]]:
    cpu_rows: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    non_cpu_rows: dict[tuple[str, str, str], dict[str, Any]] = {}

    for row in all_rows:
        device = str(row.get("device", "?"))
        model_name = str(row.get("model_name", "?"))
        model_path = str(row.get("model_path", "?"))
        if is_direct_cpu_device(device):
            isa = str(row.get("cpu_isa") or row.get("container_cpu_isa") or "unknown")
            row["cpu_isa"] = isa
            cpu_rows[(model_name, model_path, device, isa)] = row
        else:
            key = (model_name, model_path, device)
            current = non_cpu_rows.get(key)
            if current is None or row_preference_score(row, preferred_non_cpu_isa) > row_preference_score(current, preferred_non_cpu_isa):
                non_cpu_rows[key] = row

    merged = list(cpu_rows.values()) + list(non_cpu_rows.values())
    merged.sort(
        key=lambda r: (
            str(r.get("model_name", "")),
            str(r.get("device", "")),
            str(r.get("cpu_isa") or ""),
            str(r.get("container_cpu_isa") or ""),
        )
    )
    return merged


def group_models(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["model_name"]), str(row["model_path"]))
        grouped.setdefault(key, []).append(row)

    models: list[dict[str, Any]] = []
    for (name, model), model_rows in sorted(grouped.items()):
        devices = []
        for row in model_rows:
            devices.append(
                {
                    "device": row.get("device"),
                    "cpu_isa": row.get("cpu_isa"),
                    "container_cpu_isa": row.get("container_cpu_isa"),
                    "source": row.get("source"),
                    "image": row.get("image"),
                    "prefill_tok_s": row.get("prefill_tok_s"),
                    "decode_tok_s": row.get("decode_tok_s"),
                    "baseline_prefill_tok_s": row.get("baseline_prefill_tok_s"),
                    "baseline_decode_tok_s": row.get("baseline_decode_tok_s"),
                }
            )
        models.append({"name": name, "model": model, "devices": devices})
    return models


def write_csv(path: Path, commit: str, timestamp: str, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "commit",
                "timestamp",
                "source",
                "image",
                "container_cpu_isa",
                "model_name",
                "model_path",
                "device",
                "cpu_isa",
                "prefill_tok_s",
                "decode_tok_s",
                "baseline_prefill_tok_s",
                "baseline_decode_tok_s",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "commit": commit,
                    "timestamp": timestamp,
                    "source": row.get("source"),
                    "image": row.get("image"),
                    "container_cpu_isa": row.get("container_cpu_isa"),
                    "model_name": row.get("model_name"),
                    "model_path": row.get("model_path"),
                    "device": row.get("device"),
                    "cpu_isa": row.get("cpu_isa"),
                    "prefill_tok_s": row.get("prefill_tok_s"),
                    "decode_tok_s": row.get("decode_tok_s"),
                    "baseline_prefill_tok_s": row.get("baseline_prefill_tok_s"),
                    "baseline_decode_tok_s": row.get("baseline_decode_tok_s"),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True, help="Artifact file or directory. Repeatable.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write merged result files.")
    parser.add_argument("--commit", required=True, help="Commit hash/short hash for the merged record.")
    parser.add_argument("--preferred-non-cpu-isa", default="AVX512", choices=("AVX2", "AVX512"))
    args = parser.parse_args()

    json_paths = load_jsons(args.input)
    if not json_paths:
        raise SystemExit("merge_benchmark_results: no benchmark_results.json files found")

    runs: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for path in json_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        file_runs, file_rows = iter_rows(data, path)
        runs.extend(file_runs)
        rows.extend(file_rows)

    merged_rows = merge_rows(rows, args.preferred_non_cpu_isa)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    output = {
        "schema": "llaminar.benchmark.v2",
        "commit": args.commit,
        "timestamp": timestamp,
        "runs": runs,
        "models": group_models(merged_rows),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "benchmark_results.json"
    csv_path = args.output_dir / "benchmark_results.csv"
    json_path.write_text(json.dumps(output, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    write_csv(csv_path, args.commit, timestamp, merged_rows)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
