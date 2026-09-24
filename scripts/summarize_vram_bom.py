#!/usr/bin/env python3
"""Summarize LLAMINAR_VRAM_BOM log rows."""

from __future__ import annotations

import argparse
import collections
import shlex
from pathlib import Path


def mib(value: int) -> float:
    return value / (1024.0 * 1024.0)


def parse_fields(payload: str) -> dict[str, str]:
    row: dict[str, str] = {}
    for token in shlex.split(payload):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        row[key] = value
    return row


def row_bytes(row: dict[str, str]) -> int:
    kind = row.get("kind", "")
    if kind == "weight_preflight":
        keys = ("required_bytes",)
    elif kind == "workspace_model_floor":
        keys = ("floor_bytes", "min_budget_bytes")
    elif kind == "weight_pool_summary":
        keys = ("total_bytes", "persistent_bytes")
    elif kind == "weight_pool_weight":
        keys = ("persistent_bytes",)
    else:
        keys = (
            "bytes",
            "total_bytes",
            "needed_bytes",
            "required_bytes",
            "persistent_bytes",
            "planned_weights_bytes",
        )

    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return 0


def row_device(row: dict[str, str]) -> str:
    return row.get("device") or row.get("backend") or row.get("device_id") or "n/a"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path, help="Log file containing [VRAM_BOM] rows")
    parser.add_argument("--top", type=int, default=40, help="Number of largest rows to print")
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    with args.log.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            marker = "[VRAM_BOM]"
            pos = line.find(marker)
            if pos < 0:
                continue
            row = parse_fields(line[pos + len(marker) :].strip())
            if row:
                rows.append(row)

    if not rows:
        print("No [VRAM_BOM] rows found.")
        return 1

    by_kind: collections.Counter[str] = collections.Counter()
    by_kind_device: collections.Counter[tuple[str, str]] = collections.Counter()
    detail_rows: list[tuple[int, dict[str, str]]] = []
    detail_kinds = {
        "arena_buffer",
        "workspace_buffer",
        "kv_cache_buffer",
        "dynamic_tensor_buffer",
        "weight_pool_weight",
        "collective_temp_buffer",
        "collective_temp_reservation",
    }

    for row in rows:
        kind = row.get("kind", "unknown")
        size = row_bytes(row)
        by_kind[kind] += size
        by_kind_device[(kind, row_device(row))] += size
        if kind in detail_kinds:
            detail_rows.append((size, row))

    print("Totals by kind:")
    for kind, total in by_kind.most_common():
        print(f"  {kind:32s} {total:14d} bytes {mib(total):10.3f} MiB")

    print("\nTotals by kind/device:")
    for (kind, device), total in by_kind_device.most_common():
        print(f"  {kind:32s} {device:16s} {total:14d} bytes {mib(total):10.3f} MiB")

    print(f"\nTop {args.top} buffer rows:")
    for size, row in sorted(detail_rows, key=lambda item: item[0], reverse=True)[: args.top]:
        kind = row.get("kind", "unknown")
        device = row_device(row)
        name = row.get("name") or row.get("label") or row.get("source") or ""
        extra = ""
        if "side" in row:
            extra += f" side={row['side']}"
        if "layer" in row:
            extra += f" layer={row['layer']}"
        if "seq" in row:
            extra += f" seq={row['seq']}"
        if "rows" in row and "cols" in row:
            extra += f" shape={row['rows']}x{row['cols']}"
        print(f"  {mib(size):10.3f} MiB  {kind:26s} {device:16s} {name}{extra}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
