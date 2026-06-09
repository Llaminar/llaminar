#!/usr/bin/env python3
"""Summarize MTP perfstats for iteration benchmark matrix rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


FIELDS = (
    "decode_step_ms",
    "verifier_ms",
    "correction_ms",
    "correction_count",
    "publish_ms",
    "main_verifier_warmup",
    "main_verifier_capture",
    "main_verifier_replay",
    "replay_resets",
    "replay_preserves",
)


def _records(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"{path}: expected perfstats 'records' list")
    return [record for record in records if isinstance(record, dict)]


def _tags(record: dict[str, Any]) -> dict[str, str]:
    tags = record.get("tags", {})
    if not isinstance(tags, dict):
        return {}
    return {str(k): str(v) for k, v in tags.items()}


def _matches(
    record: dict[str, Any],
    *,
    domain: str,
    name: str,
    phase: str | None = None,
    tags: dict[str, str] | None = None,
) -> bool:
    if record.get("domain") != domain or record.get("name") != name:
        return False
    if phase is not None and record.get("phase") != phase:
        return False
    if tags:
        record_tags = _tags(record)
        for key, value in tags.items():
            if record_tags.get(key) != value:
                return False
    return True


def _sum_total_ms(records: Iterable[dict[str, Any]], domain: str, name: str) -> float:
    total = 0.0
    for record in records:
        if _matches(record, domain=domain, name=name):
            total += float(record.get("total_ms", 0.0) or 0.0)
    return total


def _sum_count(
    records: Iterable[dict[str, Any]],
    domain: str,
    name: str,
    *,
    phase: str | None = None,
    tags: dict[str, str] | None = None,
) -> int:
    total = 0
    for record in records:
        if _matches(record, domain=domain, name=name, phase=phase, tags=tags):
            total += int(record.get("count", 0) or 0)
    return total


def summarize(path: Path | None) -> dict[str, float | int]:
    records = _records(path)
    return {
        "decode_step_ms": _sum_total_ms(records, "mtp", "decode_step_total"),
        "verifier_ms": _sum_total_ms(records, "mtp", "verifier_forward"),
        "correction_ms": _sum_total_ms(records, "mtp", "all_position_correction_forward"),
        "correction_count": _sum_count(records, "mtp", "all_position_correction_forward"),
        "publish_ms": _sum_total_ms(records, "mtp", "all_position_publish_accepted_state"),
        "main_verifier_warmup": _sum_count(
            records,
            "forward_graph",
            "decode_segmented_phase",
            phase="decode",
            tags={"context": "main_verifier", "phase": "warmup"},
        ),
        "main_verifier_capture": _sum_count(
            records,
            "forward_graph",
            "decode_segmented_phase",
            phase="decode",
            tags={"context": "main_verifier", "phase": "capture"},
        ),
        "main_verifier_replay": _sum_count(
            records,
            "forward_graph",
            "decode_segmented_phase",
            phase="decode",
            tags={"context": "main_verifier", "phase": "replay"},
        ),
        "replay_resets": _sum_count(
            records,
            "mtp",
            "live_prefix_replay_state_after_mutation",
            phase="decode",
            tags={"replay_state": "reset"},
        ),
        "replay_preserves": _sum_count(
            records,
            "mtp",
            "live_prefix_replay_state_after_mutation",
            phase="decode",
            tags={"replay_state": "preserved"},
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("perfstats", nargs="?", type=Path)
    parser.add_argument("--format", choices=("tsv", "json"), default="tsv")
    args = parser.parse_args()

    summary = summarize(args.perfstats)
    if args.format == "json":
        print(json.dumps(summary, sort_keys=True))
    else:
        print("\t".join(str(summary[field]) for field in FIELDS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
