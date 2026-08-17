#!/usr/bin/env python3
"""Evaluate Least-Loaded Expert Parallelism candidates on MoE trace corpora."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TRAIN_SEEDS = {101, 202, 303, 404}
VALIDATION_SEEDS = {505}
HOLDOUT_SEEDS = {606}


def parse_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def seed_split(backend: str, seed: int, n_predict: int) -> str:
    if seed in TRAIN_SEEDS:
        return "train"
    if seed in VALIDATION_SEEDS:
        return "validation"
    if seed in HOLDOUT_SEEDS:
        return "holdout"

    digest = hashlib.sha1(f"{backend}:{seed}:{n_predict}".encode("utf-8")).digest()
    bucket = digest[0] % 10
    if bucket < 7:
        return "train"
    if bucket < 8:
        return "validation"
    return "holdout"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(parsed)
    return rows


def parse_int_list(value: str) -> list[int]:
    result: list[int] = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        result.append(int(stripped))
    return result


def read_summary(summary_path: Path) -> list[dict[str, str]]:
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        return 0
    return numerator // denominator + (1 if numerator % denominator else 0)


def is_balanced_enough(
    total_load: int,
    max_expert_load: int,
    expert_count: int,
    lambda_numerator: int,
    lambda_denominator: int,
) -> bool:
    if total_load <= 0 or expert_count <= 0:
        return True
    if lambda_numerator <= 0 or lambda_denominator <= 0:
        return False
    return (
        max_expert_load * expert_count * lambda_denominator
        < total_load * lambda_numerator
    )


def available_capacity(capacity: int, assigned: int, pending: int) -> int:
    if assigned >= capacity:
        return 0
    remaining = capacity - assigned
    return 0 if pending >= remaining else remaining - pending


def load_stats(participant_loads: list[int]) -> dict[str, int]:
    if not participant_loads:
        return {"total": 0, "min": 0, "max": 0, "spread": 0}
    total = sum(participant_loads)
    minimum = min(participant_loads)
    maximum = max(participant_loads)
    return {
        "total": total,
        "min": minimum,
        "max": maximum,
        "spread": maximum - minimum,
    }


def standard_participant_loads(
    expert_loads: list[int],
    owners: list[int],
    participant_count: int,
) -> list[int]:
    loads = [0 for _ in range(participant_count)]
    for expert, load in enumerate(expert_loads):
        owner = owners[expert] if expert < len(owners) else -1
        if 0 <= owner < participant_count:
            loads[owner] += load
    return loads


def append_span(
    spans: list[dict[str, int]],
    transfers: set[tuple[int, int, int]],
    status: Counter[str],
    expert: int,
    owner: int,
    destination: int,
    begin: int,
    end: int,
    forced: bool,
) -> None:
    if end <= begin:
        return
    foreign = destination != owner
    spans.append(
        {
            "expert": expert,
            "owner": owner,
            "destination": destination,
            "begin": begin,
            "end": end,
            "foreign": 1 if foreign else 0,
            "forced": 1 if forced else 0,
        }
    )
    rows = end - begin
    if foreign:
        status["spilled_rows"] += rows
        transfers.add((expert, owner, destination))
    else:
        status["native_rows"] += rows


def least_loaded_other(
    native_participant: int,
    participant_count: int,
    pending: list[int],
    assigned: list[int],
) -> int:
    best = -1
    best_load = 0
    for participant in range(participant_count):
        if participant == native_participant and participant_count > 1:
            continue
        load = pending[participant] + assigned[participant]
        if best < 0 or load < best_load or (load == best_load and participant < best):
            best = participant
            best_load = load
    return native_participant if best < 0 else best


def spill_least_loaded(
    spans: list[dict[str, int]],
    transfers: set[tuple[int, int, int]],
    status: Counter[str],
    assigned: list[int],
    pending: list[int],
    participant_count: int,
    capacity: int,
    min_chunk_tokens: int,
    expert: int,
    owner: int,
    remaining_rows: int,
    route_row_offset: int,
) -> None:
    while remaining_rows > 0:
        assigned_this_round = False
        skipped: set[int] = set()

        while len(skipped) < participant_count:
            best = -1
            best_load = 0
            for participant in range(participant_count):
                if participant == owner and participant_count > 1:
                    continue
                if participant in skipped:
                    continue
                load = assigned[participant] + pending[participant]
                if best < 0 or load < best_load or (
                    load == best_load and participant < best
                ):
                    best = participant
                    best_load = load

            if best < 0:
                break

            available = available_capacity(capacity, assigned[best], pending[best])
            chunk = min(remaining_rows, available)
            if (
                chunk == 0
                or (
                    min_chunk_tokens > 0
                    and chunk < min_chunk_tokens
                    and remaining_rows > chunk
                )
            ):
                status["min_chunk_skips"] += 1
                skipped.add(best)
                continue

            append_span(
                spans,
                transfers,
                status,
                expert,
                owner,
                best,
                route_row_offset,
                route_row_offset + chunk,
                False,
            )
            assigned[best] += chunk
            route_row_offset += chunk
            remaining_rows -= chunk
            assigned_this_round = True
            break

        if not assigned_this_round:
            forced = least_loaded_other(owner, participant_count, pending, assigned)
            append_span(
                spans,
                transfers,
                status,
                expert,
                owner,
                forced,
                route_row_offset,
                route_row_offset + remaining_rows,
                True,
            )
            assigned[forced] += remaining_rows
            remaining_rows = 0
            status["forced_spills"] += 1


def plan_llep(
    expert_loads: list[int],
    owners: list[int],
    participant_count: int,
    *,
    min_chunk_tokens: int,
    alpha_numerator: int,
    alpha_denominator: int,
    lambda_numerator: int,
    lambda_denominator: int,
    min_spread_improvement: int,
    min_spread_improvement_per_critical_path_slot: int,
    enable_balanced_skip: bool,
) -> dict[str, Any]:
    status: Counter[str] = Counter()
    expert_count = len(expert_loads)
    pending = [0 for _ in range(participant_count)]
    assigned = [0 for _ in range(participant_count)]
    spans: list[dict[str, int]] = []
    transfers: set[tuple[int, int, int]] = set()

    if (
        expert_count <= 0
        or participant_count <= 0
        or participant_count > 64
        or alpha_numerator <= 0
        or alpha_denominator <= 0
        or len(owners) < expert_count
    ):
        status["invalid_config"] = 1
        return {"status": status, "post_loads": assigned, "spans": spans}

    for expert, load in enumerate(expert_loads):
        owner = owners[expert]
        if owner < 0 or owner >= participant_count:
            status["invalid_config"] = 1
            return {"status": status, "post_loads": assigned, "spans": spans}
        pending[owner] += load
        status["total_load"] += load
        status["max_expert_load"] = max(status["max_expert_load"], load)

    standard_loads = list(pending)
    standard = load_stats(standard_loads)
    if enable_balanced_skip and is_balanced_enough(
        status["total_load"],
        status["max_expert_load"],
        expert_count,
        lambda_numerator,
        lambda_denominator,
    ):
        status["skipped_balanced"] = 1
        status["standard_ep_selected"] = 1
        return {
            "status": status,
            "post_loads": standard_loads,
            "spans": spans,
        }

    status["capacity_per_participant"] = ceil_div(
        status["total_load"] * alpha_numerator,
        participant_count * alpha_denominator,
    )

    for expert in sorted(range(expert_count), key=lambda idx: (-expert_loads[idx], idx)):
        load = expert_loads[expert]
        if load <= 0:
            continue
        owner = owners[expert]
        pending[owner] = max(0, pending[owner] - load)

        native_available = available_capacity(
            status["capacity_per_participant"],
            assigned[owner],
            pending[owner],
        )
        if native_available >= load:
            append_span(spans, transfers, status, expert, owner, owner, 0, load, False)
            assigned[owner] += load
            continue

        route_offset = 0
        remaining = load
        if native_available > 0:
            append_span(
                spans,
                transfers,
                status,
                expert,
                owner,
                owner,
                0,
                native_available,
                False,
            )
            assigned[owner] += native_available
            route_offset = native_available
            remaining -= native_available

        spill_least_loaded(
            spans,
            transfers,
            status,
            assigned,
            pending,
            participant_count,
            status["capacity_per_participant"],
            min_chunk_tokens,
            expert,
            owner,
            remaining,
            route_offset,
        )

    status["span_count"] = len(spans)
    status["weight_transfer_count"] = len(transfers)
    outgoing = Counter(transfer[1] for transfer in transfers)
    incoming = Counter(transfer[2] for transfer in transfers)
    critical_path_transfer_slots = max(
        [0, *outgoing.values(), *incoming.values()]
    )
    status["critical_path_transfer_slots"] = critical_path_transfer_slots
    planned = load_stats(assigned)
    improvement = max(0, standard["spread"] - planned["spread"])
    required_improvement = max(0, min_spread_improvement) + (
        max(0, min_spread_improvement_per_critical_path_slot)
        * critical_path_transfer_slots
    )
    status["required_spread_improvement"] = required_improvement
    status["assigned_load_spread_improvement"] = improvement
    if required_improvement > 0 and improvement < required_improvement:
        status["skipped_insufficient_spread_improvement"] = 1
        status["standard_ep_selected"] = 1
        for key in (
            "native_rows",
            "spilled_rows",
            "span_count",
            "weight_transfer_count",
            "min_chunk_skips",
            "forced_spills",
        ):
            status[key] = 0
        spans.clear()
        transfers.clear()
        assigned = standard_loads
    return {"status": status, "post_loads": assigned, "spans": spans}


def owner_map_by_wave(trace: dict[str, Any]) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for row in trace.get("expert_owner_participants", []) or []:
        wave_layer = parse_int(row.get("wave_layer"), -1)
        owners = row.get("owners", [])
        if wave_layer >= 0 and isinstance(owners, list):
            result[wave_layer] = [parse_int(owner, -1) for owner in owners]
    return result


def global_loads_by_wave(trace: dict[str, Any]) -> dict[int, list[int]]:
    histogram = trace.get("gathered_histogram")
    if not isinstance(histogram, dict):
        return {}

    expert_count = parse_int(histogram.get("num_experts"))
    wave_layer_count = parse_int(histogram.get("wave_layer_count"))
    if expert_count <= 0 or wave_layer_count <= 0:
        return {}

    loads = {
        wave_layer: [0 for _ in range(expert_count)]
        for wave_layer in range(wave_layer_count)
    }
    global_entries = histogram.get("global_expert_loads", [])
    if isinstance(global_entries, list) and global_entries:
        for entry in global_entries:
            if not isinstance(entry, dict):
                continue
            wave_layer = parse_int(entry.get("wave_layer"), -1)
            expert = parse_int(entry.get("expert"), -1)
            if wave_layer in loads and 0 <= expert < expert_count:
                loads[wave_layer][expert] += parse_int(entry.get("count"))
        return loads

    for entry in histogram.get("nonzero", []) or []:
        if not isinstance(entry, dict):
            continue
        wave_layer = parse_int(entry.get("wave_layer"), -1)
        expert = parse_int(entry.get("expert"), -1)
        if wave_layer in loads and 0 <= expert < expert_count:
            loads[wave_layer][expert] += parse_int(entry.get("count"))
    return loads


def trace_input_signature(
    loads_by_wave: dict[int, list[int]],
    owners_by_wave: dict[int, list[int]],
) -> str:
    digest = hashlib.sha1()
    for wave_layer, loads in sorted(loads_by_wave.items()):
        digest.update(f"L{wave_layer}:".encode("utf-8"))
        digest.update(",".join(str(value) for value in loads).encode("utf-8"))
        digest.update(b";")
    for wave_layer, owners in sorted(owners_by_wave.items()):
        digest.update(f"O{wave_layer}:".encode("utf-8"))
        digest.update(",".join(str(value) for value in owners).encode("utf-8"))
        digest.update(b";")
    return digest.hexdigest()


def evaluate_trace_window(
    summary: dict[str, str],
    trace: dict[str, Any],
    row_index: int,
    wave_layer: int,
    expert_loads: list[int],
    owners: list[int],
    args: argparse.Namespace,
    corpus_dir: Path,
) -> dict[str, Any]:
    config = trace.get("config", {})
    tags = trace.get("tags", {})
    status = trace.get("status", {})
    participant_count = parse_int(config.get("participant_count"))

    standard_loads = standard_participant_loads(expert_loads, owners, participant_count)
    standard = load_stats(standard_loads)
    plan = plan_llep(
        expert_loads,
        owners,
        participant_count,
        min_chunk_tokens=args.min_chunk_tokens,
        alpha_numerator=args.alpha_numerator,
        alpha_denominator=args.alpha_denominator,
        lambda_numerator=args.lambda_numerator,
        lambda_denominator=args.lambda_denominator,
        min_spread_improvement=args.min_spread_improvement,
        min_spread_improvement_per_critical_path_slot=(
            args.min_spread_improvement_per_critical_path_slot
        ),
        enable_balanced_skip=not args.disable_balanced_skip,
    )
    llep = load_stats(plan["post_loads"])
    plan_status: Counter[str] = plan["status"]
    improvement = max(0, standard["spread"] - llep["spread"])
    total = standard["total"]

    return {
        "backend": summary.get("backend", ""),
        "placement": summary.get("placement", ""),
        "case": summary.get("case", ""),
        "n_predict": parse_int(summary.get("n_predict")),
        "seed": parse_int(summary.get("seed")),
        "split": seed_split(
            summary.get("backend", ""),
            parse_int(summary.get("seed")),
            parse_int(summary.get("n_predict")),
        ),
        "rep": parse_int(summary.get("rep")),
        "row_index": row_index,
        "wave_layer": wave_layer,
        "maintenance_graph_kind": tags.get("maintenance_graph_kind", ""),
        "decode_tokens_seen": parse_int(tags.get("decode_tokens_seen")),
        "expert_count": len(expert_loads),
        "participant_count": participant_count,
        "total_load": total,
        "max_expert_load": max(expert_loads) if expert_loads else 0,
        "standard_load_min": standard["min"],
        "standard_load_max": standard["max"],
        "standard_load_spread": standard["spread"],
        "llep_load_min": llep["min"],
        "llep_load_max": llep["max"],
        "llep_load_spread": llep["spread"],
        "spread_improvement": improvement,
        "spread_improvement_fraction": (
            "" if standard["spread"] <= 0 else f"{improvement / standard['spread']:.9g}"
        ),
        "spread_improvement_per_critical_path_slot": (
            ""
            if plan_status["critical_path_transfer_slots"] <= 0
            else f"{improvement / plan_status['critical_path_transfer_slots']:.9g}"
        ),
        "required_spread_improvement": plan_status["required_spread_improvement"],
        "min_spread_improvement": max(0, args.min_spread_improvement),
        "min_spread_improvement_per_critical_path_slot": max(
            0, args.min_spread_improvement_per_critical_path_slot
        ),
        "capacity_per_participant": plan_status["capacity_per_participant"],
        "span_count": plan_status["span_count"],
        "weight_transfer_count": plan_status["weight_transfer_count"],
        "critical_path_transfer_slots": plan_status[
            "critical_path_transfer_slots"
        ],
        "spilled_rows": plan_status["spilled_rows"],
        "native_rows": plan_status["native_rows"],
        "min_chunk_skips": plan_status["min_chunk_skips"],
        "forced_spills": plan_status["forced_spills"],
        "invalid_config": plan_status["invalid_config"],
        "skipped_balanced": plan_status["skipped_balanced"],
        "skipped_insufficient_spread_improvement": plan_status[
            "skipped_insufficient_spread_improvement"
        ],
        "standard_ep_selected": plan_status["standard_ep_selected"],
        "current_policy_planned_arrivals": parse_int(status.get("planned_arrivals")),
        "current_policy_payload_slots": parse_int(
            status.get("payload_bucket_requested_slots")
        ),
        "rebalance_trace_jsonl": relative_path(
            Path(summary.get("rebalance_trace_jsonl", "")),
            corpus_dir,
        ),
    }


def build_tables(
    corpus_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows = read_summary(corpus_dir / "summary.tsv")
    run_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []

    for summary in summary_rows:
        trace_path = Path(summary.get("rebalance_trace_jsonl", ""))
        traces = load_jsonl(trace_path)
        aggregate: Counter[str] = Counter()
        missing_trace_rows = 0
        empty_wave_layers = 0
        seen_trace_inputs: set[str] = set()

        for row_index, trace in enumerate(traces):
            loads_by_wave = global_loads_by_wave(trace)
            owners_by_wave = owner_map_by_wave(trace)
            if not loads_by_wave or not owners_by_wave:
                missing_trace_rows += 1
                continue

            signature = trace_input_signature(loads_by_wave, owners_by_wave)
            if signature in seen_trace_inputs:
                aggregate["duplicate_trace_rows"] += 1
                continue
            seen_trace_inputs.add(signature)

            aggregate["eligible_trace_rows"] += 1
            for wave_layer, expert_loads in sorted(loads_by_wave.items()):
                owners = owners_by_wave.get(wave_layer)
                if owners is None:
                    aggregate["missing_owner_wave_layers"] += 1
                    continue
                if sum(expert_loads) <= 0:
                    empty_wave_layers += 1
                    continue

                row = evaluate_trace_window(
                    summary,
                    trace,
                    row_index,
                    wave_layer,
                    expert_loads,
                    owners,
                    args,
                    corpus_dir,
                )
                window_rows.append(row)
                aggregate["eligible_wave_layers"] += 1
                aggregate["total_load"] += row["total_load"]
                aggregate["standard_load_spread"] += row["standard_load_spread"]
                aggregate["llep_load_spread"] += row["llep_load_spread"]
                aggregate["spread_improvement"] += row["spread_improvement"]
                aggregate["weight_transfer_count"] += row["weight_transfer_count"]
                aggregate["critical_path_transfer_slots"] += row[
                    "critical_path_transfer_slots"
                ]
                aggregate["required_spread_improvement"] += row[
                    "required_spread_improvement"
                ]
                aggregate["spilled_rows"] += row["spilled_rows"]
                aggregate["native_rows"] += row["native_rows"]
                aggregate["span_count"] += row["span_count"]
                aggregate["min_chunk_skips"] += row["min_chunk_skips"]
                aggregate["forced_spills"] += row["forced_spills"]
                aggregate["invalid_config"] += row["invalid_config"]
                aggregate["skipped_balanced"] += row["skipped_balanced"]
                aggregate["skipped_insufficient_spread_improvement"] += row[
                    "skipped_insufficient_spread_improvement"
                ]
                aggregate["current_policy_planned_arrivals"] += row[
                    "current_policy_planned_arrivals"
                ]
                aggregate["current_policy_payload_slots"] += row[
                    "current_policy_payload_slots"
                ]

        standard_spread = aggregate["standard_load_spread"]
        run_rows.append(
            {
                "backend": summary.get("backend", ""),
                "placement": summary.get("placement", ""),
                "case": summary.get("case", ""),
                "n_predict": parse_int(summary.get("n_predict")),
                "seed": parse_int(summary.get("seed")),
                "split": seed_split(
                    summary.get("backend", ""),
                    parse_int(summary.get("seed")),
                    parse_int(summary.get("n_predict")),
                ),
                "rep": parse_int(summary.get("rep")),
                "trace_rows": len(traces),
                "missing_trace_rows": missing_trace_rows,
                "duplicate_trace_rows": aggregate["duplicate_trace_rows"],
                "eligible_trace_rows": aggregate["eligible_trace_rows"],
                "eligible_wave_layers": aggregate["eligible_wave_layers"],
                "empty_wave_layers": empty_wave_layers,
                "missing_owner_wave_layers": aggregate["missing_owner_wave_layers"],
                "total_load": aggregate["total_load"],
                "standard_load_spread": standard_spread,
                "llep_load_spread": aggregate["llep_load_spread"],
                "spread_improvement": aggregate["spread_improvement"],
                "spread_improvement_fraction": (
                    ""
                    if standard_spread <= 0
                    else f"{aggregate['spread_improvement'] / standard_spread:.9g}"
                ),
                "spread_improvement_per_critical_path_slot": (
                    ""
                    if aggregate["critical_path_transfer_slots"] <= 0
                    else f"{aggregate['spread_improvement'] / aggregate['critical_path_transfer_slots']:.9g}"
                ),
                "required_spread_improvement": aggregate["required_spread_improvement"],
                "weight_transfer_count": aggregate["weight_transfer_count"],
                "critical_path_transfer_slots": aggregate[
                    "critical_path_transfer_slots"
                ],
                "spilled_rows": aggregate["spilled_rows"],
                "native_rows": aggregate["native_rows"],
                "span_count": aggregate["span_count"],
                "min_chunk_skips": aggregate["min_chunk_skips"],
                "forced_spills": aggregate["forced_spills"],
                "invalid_config": aggregate["invalid_config"],
                "skipped_balanced": aggregate["skipped_balanced"],
                "skipped_insufficient_spread_improvement": aggregate[
                    "skipped_insufficient_spread_improvement"
                ],
                "current_policy_planned_arrivals": aggregate[
                    "current_policy_planned_arrivals"
                ],
                "current_policy_payload_slots": aggregate["current_policy_payload_slots"],
                "rebalance_trace_jsonl": relative_path(trace_path, corpus_dir),
            }
        )

    return run_rows, window_rows, grouped_summary(run_rows)


def grouped_summary(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        groups[
            (
                row["backend"],
                row["n_predict"],
                row["split"],
                row["case"],
            )
        ].append(row)

    output: list[dict[str, Any]] = []
    for (backend, n_predict, split, case), rows in sorted(groups.items()):
        standard_spread = sum(row["standard_load_spread"] for row in rows)
        improvement = sum(row["spread_improvement"] for row in rows)
        output.append(
            {
                "backend": backend,
                "n_predict": n_predict,
                "split": split,
                "case": case,
                "runs": len(rows),
                "eligible_wave_layers": sum(row["eligible_wave_layers"] for row in rows),
                "missing_trace_rows": sum(row["missing_trace_rows"] for row in rows),
                "duplicate_trace_rows": sum(row["duplicate_trace_rows"] for row in rows),
                "standard_load_spread": standard_spread,
                "llep_load_spread": sum(row["llep_load_spread"] for row in rows),
                "spread_improvement": improvement,
                "spread_improvement_fraction": (
                    "" if standard_spread <= 0 else f"{improvement / standard_spread:.9g}"
                ),
                "spread_improvement_per_critical_path_slot": (
                    ""
                    if sum(row["critical_path_transfer_slots"] for row in rows) <= 0
                    else f"{improvement / sum(row['critical_path_transfer_slots'] for row in rows):.9g}"
                ),
                "required_spread_improvement": sum(
                    row["required_spread_improvement"] for row in rows
                ),
                "weight_transfer_count": sum(row["weight_transfer_count"] for row in rows),
                "critical_path_transfer_slots": sum(
                    row["critical_path_transfer_slots"] for row in rows
                ),
                "spilled_rows": sum(row["spilled_rows"] for row in rows),
                "skipped_insufficient_spread_improvement": sum(
                    row["skipped_insufficient_spread_improvement"] for row in rows
                ),
                "current_policy_planned_arrivals": sum(
                    row["current_policy_planned_arrivals"] for row in rows
                ),
            }
        )
    return output


def build_roi_sweep(
    corpus_dir: Path,
    args: argparse.Namespace,
    per_critical_path_slot_thresholds: list[int],
) -> list[dict[str, Any]]:
    sweep_rows: list[dict[str, Any]] = []
    for threshold in per_critical_path_slot_thresholds:
        sweep_args = argparse.Namespace(**vars(args))
        sweep_args.min_spread_improvement_per_critical_path_slot = threshold
        run_rows, _window_rows, split_rows = build_tables(corpus_dir, sweep_args)
        for row in split_rows:
            transfers = row["weight_transfer_count"]
            critical_path_slots = row["critical_path_transfer_slots"]
            improvement = row["spread_improvement"]
            sweep_rows.append(
                {
                    "min_spread_improvement_per_critical_path_slot": max(
                        0, threshold
                    ),
                    "backend": row["backend"],
                    "n_predict": row["n_predict"],
                    "split": row["split"],
                    "case": row["case"],
                    "runs": row["runs"],
                    "eligible_wave_layers": row["eligible_wave_layers"],
                    "standard_load_spread": row["standard_load_spread"],
                    "llep_load_spread": row["llep_load_spread"],
                    "spread_improvement": improvement,
                    "spread_improvement_fraction": row["spread_improvement_fraction"],
                    "weight_transfer_count": transfers,
                    "critical_path_transfer_slots": critical_path_slots,
                    "spread_improvement_per_critical_path_slot": (
                        ""
                        if critical_path_slots <= 0
                        else f"{improvement / critical_path_slots:.9g}"
                    ),
                    "spilled_rows": row["spilled_rows"],
                    "skipped_insufficient_spread_improvement": row[
                        "skipped_insufficient_spread_improvement"
                    ],
                }
            )

        if not run_rows:
            sweep_rows.append(
                {
                    "min_spread_improvement_per_critical_path_slot": max(
                        0, threshold
                    ),
                    "backend": "",
                    "n_predict": 0,
                    "split": "",
                    "case": "",
                    "runs": 0,
                    "eligible_wave_layers": 0,
                    "standard_load_spread": 0,
                    "llep_load_spread": 0,
                    "spread_improvement": 0,
                    "spread_improvement_fraction": "",
                    "weight_transfer_count": 0,
                    "critical_path_transfer_slots": 0,
                    "spread_improvement_per_critical_path_slot": "",
                    "spilled_rows": 0,
                    "skipped_insufficient_spread_improvement": 0,
                }
            )
    return sweep_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No LLEP-evaluable trace rows found.")
        return
    print(
        "backend n_predict split case runs eligible_layers missing_trace_rows "
        "duplicate_trace_rows standard_spread llep_spread improvement improvement_frac transfers spilled_rows"
    )
    for row in rows:
        print(
            f"{row['backend']} {row['n_predict']} {row['split']} {row['case']} "
            f"{row['runs']} {row['eligible_wave_layers']} {row['missing_trace_rows']} "
            f"{row['duplicate_trace_rows']} "
            f"{row['standard_load_spread']} {row['llep_load_spread']} "
            f"{row['spread_improvement']} {row['spread_improvement_fraction']} "
            f"{row['weight_transfer_count']} {row['spilled_rows']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-chunk-tokens", type=int, default=64)
    parser.add_argument("--alpha-numerator", type=int, default=1)
    parser.add_argument("--alpha-denominator", type=int, default=1)
    parser.add_argument("--lambda-numerator", type=int, default=13)
    parser.add_argument("--lambda-denominator", type=int, default=10)
    parser.add_argument("--min-spread-improvement", type=int, default=0)
    parser.add_argument(
        "--min-spread-improvement-per-critical-path-slot",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--sweep-min-spread-improvement-per-critical-path-slot",
        default="",
        help=(
            "Comma-separated per-critical-path-slot ROI thresholds to evaluate "
            "into roi_sweep.csv"
        ),
    )
    parser.add_argument("--disable-balanced-skip", action="store_true")
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or args.corpus_dir / "llep_analysis"
    run_rows, window_rows, split_rows = build_tables(args.corpus_dir, args)
    write_csv(output_dir / "run_summary.csv", run_rows)
    write_csv(output_dir / "window_llep.csv", window_rows)
    write_csv(output_dir / "split_summary.csv", split_rows)
    if args.sweep_min_spread_improvement_per_critical_path_slot:
        thresholds = parse_int_list(
            args.sweep_min_spread_improvement_per_critical_path_slot
        )
        write_csv(
            output_dir / "roi_sweep.csv",
            build_roi_sweep(args.corpus_dir, args, thresholds),
        )
    if args.print_summary:
        print_summary(split_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
