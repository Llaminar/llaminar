#!/usr/bin/env python3
"""Summarize MoE device-rebalance trace corpora for policy experiments."""

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
REAL_COMMAND_OPS = {1, 2}


def parse_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_optional_float(value: float | None) -> str:
    return "" if value is None else f"{value:.9g}"


def load_spread_ratio(spread: int, total: int) -> float | None:
    if total <= 0:
        return None
    return max(0, spread) / total


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


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def read_summary(summary_path: Path) -> list[dict[str, str]]:
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def benchmark_fields(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "success": False,
            "prefill_tok_s": 0.0,
            "decode_tok_s": 0.0,
            "overall_tok_s": 0.0,
            "prefill_ms": 0.0,
            "decode_ms": 0.0,
            "total_ms": 0.0,
            "decode_mean_ms": 0.0,
            "decode_p50_ms": 0.0,
            "decode_p90_ms": 0.0,
            "decode_latency_samples": 0,
        }

    data = load_json(path)
    throughput = data.get("throughput_tokens_per_sec", {})
    timing = data.get("timing_ms", {})
    latency = data.get("decode_latency_ms", {})
    return {
        "success": bool(data.get("success", False)),
        "prefill_tok_s": parse_float(throughput.get("prefill")),
        "decode_tok_s": parse_float(throughput.get("decode")),
        "overall_tok_s": parse_float(throughput.get("overall")),
        "prefill_ms": parse_float(timing.get("prefill")),
        "decode_ms": parse_float(timing.get("decode")),
        "total_ms": parse_float(timing.get("total")),
        "decode_mean_ms": parse_float(latency.get("mean")),
        "decode_p50_ms": parse_float(latency.get("p50")),
        "decode_p90_ms": parse_float(latency.get("p90")),
        "decode_latency_samples": parse_int(latency.get("samples")),
    }


def command_counts(commands: list[dict[str, Any]]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for command in commands:
        op = parse_int(command.get("op"))
        if op in REAL_COMMAND_OPS:
            counts[op] += 1
    return counts


def build_tables(corpus_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_path = corpus_dir / "summary.tsv"
    summary_rows = read_summary(summary_path)
    run_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []

    for summary in summary_rows:
        backend = summary.get("backend", "")
        n_predict = parse_int(summary.get("n_predict"))
        seed = parse_int(summary.get("seed"))
        split = seed_split(backend, seed, n_predict)
        benchmark_path = Path(summary.get("benchmark_json", ""))
        trace_path = Path(summary.get("rebalance_trace_jsonl", ""))
        benchmark = benchmark_fields(benchmark_path)
        trace_rows = load_jsonl(trace_path)

        aggregate = Counter()
        max_payload_bucket_slots = 0
        max_payload_bucket_requested_slots = 0
        max_epoch = 0
        trace_errors = 0
        first_planned_decode_token: int | None = None
        last_planned_decode_token: int | None = None
        first_applied_decode_token: int | None = None
        last_applied_decode_token: int | None = None

        for row_index, trace in enumerate(trace_rows):
            status = trace.get("status", {})
            apply_status = trace.get("apply_status", {})
            tags = trace.get("tags", {})
            commands = [
                command
                for command in trace.get("commands", [])
                if parse_int(command.get("op")) in REAL_COMMAND_OPS
            ]
            op_counts = command_counts(commands)
            planned_arrivals = parse_int(status.get("planned_arrivals"))
            selected_replicas = parse_int(status.get("selected_replicas"))
            accepted_improvement = parse_int(
                status.get("accepted_load_spread_improvement_total")
            )
            payload_bucket_slots = parse_int(status.get("payload_bucket_slots"))
            payload_requested_slots = parse_int(
                status.get("payload_bucket_requested_slots")
            )
            epoch = parse_int(status.get("last_epoch"))
            decode_tokens_seen = parse_int(tags.get("decode_tokens_seen"))
            decode_tokens_remaining = max(0, n_predict - decode_tokens_seen)
            row_error = 1 if parse_int(status.get("status_code")) != 0 else 0
            pre_load_min = parse_int(status.get("pre_policy_load_min"))
            pre_load_max = parse_int(status.get("pre_policy_load_max"))
            post_load_min = parse_int(status.get("post_policy_load_min"))
            post_load_max = parse_int(status.get("post_policy_load_max"))
            pre_load_total = parse_int(status.get("pre_policy_load_total"))
            post_load_total = parse_int(status.get("post_policy_load_total"))
            pre_load_spread = max(0, pre_load_max - pre_load_min)
            post_load_spread = max(0, post_load_max - post_load_min)
            post_wave_load_total = parse_int(status.get("post_wave_load_total"))
            post_wave_load_spread = parse_int(status.get("post_wave_load_spread"))
            llep_assignment_spans = parse_int(
                status.get("llep_assignment_span_count")
            )
            llep_weight_transfers = parse_int(
                status.get("llep_weight_transfer_count")
            )
            llep_native_rows = parse_int(status.get("llep_native_rows"))
            llep_spilled_rows = parse_int(status.get("llep_spilled_rows"))
            llep_total_rows = llep_native_rows + llep_spilled_rows
            llep_spilled_row_ratio = load_spread_ratio(
                llep_spilled_rows,
                llep_total_rows,
            )
            llep_spilled_rows_per_transfer = (
                None
                if llep_weight_transfers <= 0
                else llep_spilled_rows / llep_weight_transfers
            )
            pre_imbalance_ratio = load_spread_ratio(pre_load_spread, pre_load_total)
            post_imbalance_ratio = load_spread_ratio(post_load_spread, post_load_total)
            post_wave_imbalance_ratio = load_spread_ratio(
                post_wave_load_spread,
                post_wave_load_total,
            )
            post_apply_multi_resident_experts = parse_int(
                apply_status.get("post_apply_multi_resident_experts")
            )
            applied_arrivals = parse_int(apply_status.get("applied_arrivals"))
            maintenance_kind = str(tags.get("maintenance_graph_kind", ""))
            include_in_run_aggregate = maintenance_kind != "payload"

            if include_in_run_aggregate and planned_arrivals > 0:
                if first_planned_decode_token is None:
                    first_planned_decode_token = decode_tokens_seen
                last_planned_decode_token = decode_tokens_seen
            if applied_arrivals > 0:
                if first_applied_decode_token is None:
                    first_applied_decode_token = decode_tokens_seen
                last_applied_decode_token = decode_tokens_seen

            aggregate["trace_rows"] += 1
            trace_errors += row_error
            max_payload_bucket_slots = max(max_payload_bucket_slots, payload_bucket_slots)
            max_payload_bucket_requested_slots = max(
                max_payload_bucket_requested_slots, payload_requested_slots
            )
            max_epoch = max(max_epoch, epoch)
            if include_in_run_aggregate:
                aggregate["economic_rows"] += 1
                aggregate["planned_arrivals"] += planned_arrivals
                aggregate["selected_replicas"] += selected_replicas
                aggregate["changed_layers"] += parse_int(status.get("changed_layers"))
                aggregate["apply_post_apply_multi_resident_experts"] += (
                    post_apply_multi_resident_experts
                )
                aggregate["candidate_considered"] += parse_int(
                    status.get("candidate_arrivals_considered")
                )
                aggregate["candidate_pruned_by_count_bound"] += parse_int(
                    status.get("candidate_arrivals_pruned_by_count_bound")
                )
                aggregate["candidate_below_floor"] += parse_int(
                    status.get("candidate_arrivals_below_floor")
                )
                aggregate["llep_assignment_spans"] += llep_assignment_spans
                aggregate["llep_weight_transfers"] += llep_weight_transfers
                aggregate["llep_native_rows"] += llep_native_rows
                aggregate["llep_spilled_rows"] += llep_spilled_rows
                aggregate["llep_standard_ep_selected"] += parse_int(
                    status.get("llep_standard_ep_selected")
                )
                aggregate["llep_skipped_balanced"] += parse_int(
                    status.get("llep_skipped_balanced")
                )
                aggregate["llep_skipped_insufficient_spread_improvement"] += parse_int(
                    status.get("llep_skipped_insufficient_spread_improvement")
                )
                aggregate["llep_skipped_insufficient_foreign_rows"] += parse_int(
                    status.get("llep_skipped_insufficient_foreign_rows")
                )
                aggregate["llep_min_chunk_skips"] += parse_int(
                    status.get("llep_min_chunk_skips")
                )
                aggregate["llep_forced_spills"] += parse_int(
                    status.get("llep_forced_spills")
                )
                aggregate["llep_required_spread_improvement"] += parse_int(
                    status.get("llep_required_spread_improvement")
                )
                aggregate["llep_required_foreign_rows"] += parse_int(
                    status.get("llep_required_foreign_rows")
                )
                aggregate["candidate_improvement"] += parse_int(
                    status.get("candidate_load_spread_improvement_total")
                )
                aggregate["candidate_improvement_max"] = max(
                    aggregate["candidate_improvement_max"],
                    parse_int(status.get("candidate_load_spread_improvement_max")),
                )
                aggregate["accepted_improvement"] += accepted_improvement
                aggregate["accepted_improvement_max"] = max(
                    aggregate["accepted_improvement_max"],
                    parse_int(status.get("accepted_load_spread_improvement_max")),
                )
                aggregate["skipped_wave_cost_floor"] += parse_int(
                    status.get("skipped_wave_cost_floor")
                )
                aggregate["skipped_low_router_benefit"] += parse_int(
                    status.get("skipped_low_router_benefit")
                )
                aggregate["skipped_post_load_spread_ceiling"] += parse_int(
                    status.get("skipped_post_load_spread_ceiling")
                )
                aggregate["router_eligible"] += parse_int(
                    status.get("router_hot_cache_eligible_dispatches")
                )
                aggregate["router_used"] += parse_int(
                    status.get("router_hot_cache_used_dispatches")
                )
                aggregate["router_improved"] += parse_int(
                    status.get("router_hot_cache_improved_dispatches")
                )
                aggregate["router_default_spread"] += parse_int(
                    status.get("router_hot_cache_default_load_spread_total")
                )
                aggregate["router_actual_spread"] += parse_int(
                    status.get("router_hot_cache_actual_load_spread_total")
                )
                aggregate["router_spread_improvement"] += parse_int(
                    status.get("router_hot_cache_load_spread_improvement_total")
                )
                aggregate["router_active"] += parse_int(
                    status.get("router_hot_cache_active_dispatches")
                )
                aggregate["router_miss"] += parse_int(
                    status.get("router_hot_cache_miss_dispatches")
                )
                aggregate["router_selected_slots"] += parse_int(
                    status.get("router_hot_cache_selected_expert_slots")
                )
                aggregate["router_replicated_selected_slots"] += parse_int(
                    status.get("router_hot_cache_replicated_selected_expert_slots")
                )
                aggregate["pre_policy_load_total"] += pre_load_total
                aggregate["post_policy_load_total"] += post_load_total
                aggregate["pre_policy_load_spread"] += pre_load_spread
                aggregate["post_policy_load_spread"] += post_load_spread
                aggregate["pre_policy_load_spread_max"] = max(
                    aggregate["pre_policy_load_spread_max"],
                    pre_load_spread,
                )
                aggregate["post_policy_load_spread_max"] = max(
                    aggregate["post_policy_load_spread_max"],
                    post_load_spread,
                )
                if pre_imbalance_ratio is not None:
                    aggregate["pre_policy_imbalance_ratio_sum"] += pre_imbalance_ratio
                    aggregate["pre_policy_imbalance_ratio_samples"] += 1
                    aggregate["pre_policy_imbalance_ratio_max"] = max(
                        aggregate["pre_policy_imbalance_ratio_max"],
                        pre_imbalance_ratio,
                    )
                if post_imbalance_ratio is not None:
                    aggregate["post_policy_imbalance_ratio_sum"] += post_imbalance_ratio
                    aggregate["post_policy_imbalance_ratio_samples"] += 1
                    aggregate["post_policy_imbalance_ratio_max"] = max(
                        aggregate["post_policy_imbalance_ratio_max"],
                        post_imbalance_ratio,
                    )
                if post_wave_imbalance_ratio is not None:
                    aggregate["post_wave_imbalance_ratio_sum"] += post_wave_imbalance_ratio
                    aggregate["post_wave_imbalance_ratio_samples"] += 1
                    aggregate["post_wave_imbalance_ratio_max"] = max(
                        aggregate["post_wave_imbalance_ratio_max"],
                        post_wave_imbalance_ratio,
                    )
                aggregate["post_wave_load_total"] += post_wave_load_total
                aggregate["post_wave_load_spread"] += post_wave_load_spread
                aggregate["post_wave_load_spread_max"] = max(
                    aggregate["post_wave_load_spread_max"],
                    post_wave_load_spread,
                )
                aggregate["noop_command_entries"] += parse_int(
                    trace.get("noop_command_entries")
                )
                aggregate["command_entries"] += sum(op_counts.values())
                aggregate["transfer_arrivals"] += op_counts[1]
                aggregate["resident_replicas"] += op_counts[2]

            window_rows.append(
                {
                    "backend": backend,
                    "placement": summary.get("placement", ""),
                    "case": summary.get("case", ""),
                    "n_predict": n_predict,
                    "seed": seed,
                    "split": split,
                    "rep": parse_int(summary.get("rep")),
                    "row_index": row_index,
                    "maintenance_graph_kind": tags.get("maintenance_graph_kind", ""),
                    "decode_tokens_seen": decode_tokens_seen,
                    "decode_tokens_remaining": decode_tokens_remaining,
                    "window": parse_int(tags.get("window")),
                    "epoch": epoch,
                    "status_code": parse_int(status.get("status_code")),
                    "planned_arrivals": planned_arrivals,
                    "planned_remaining_tokens": (
                        decode_tokens_remaining
                        if include_in_run_aggregate and planned_arrivals > 0
                        else -1
                    ),
                    "selected_replicas": selected_replicas,
                    "changed_layers": parse_int(status.get("changed_layers")),
                    "candidate_considered": parse_int(
                        status.get("candidate_arrivals_considered")
                    ),
                    "candidate_pruned_by_count_bound": parse_int(
                        status.get("candidate_arrivals_pruned_by_count_bound")
                    ),
                    "candidate_below_floor": parse_int(
                        status.get("candidate_arrivals_below_floor")
                    ),
                    "llep_assignment_spans": llep_assignment_spans,
                    "llep_weight_transfers": llep_weight_transfers,
                    "llep_native_rows": llep_native_rows,
                    "llep_spilled_rows": llep_spilled_rows,
                    "llep_spilled_row_ratio": format_optional_float(
                        llep_spilled_row_ratio
                    ),
                    "llep_spilled_rows_per_transfer": format_optional_float(
                        llep_spilled_rows_per_transfer
                    ),
                    "llep_standard_ep_selected": parse_int(
                        status.get("llep_standard_ep_selected")
                    ),
                    "llep_skipped_balanced": parse_int(
                        status.get("llep_skipped_balanced")
                    ),
                    "llep_skipped_insufficient_spread_improvement": parse_int(
                        status.get("llep_skipped_insufficient_spread_improvement")
                    ),
                    "llep_skipped_insufficient_foreign_rows": parse_int(
                        status.get("llep_skipped_insufficient_foreign_rows")
                    ),
                    "llep_min_chunk_skips": parse_int(
                        status.get("llep_min_chunk_skips")
                    ),
                    "llep_forced_spills": parse_int(
                        status.get("llep_forced_spills")
                    ),
                    "llep_required_spread_improvement": parse_int(
                        status.get("llep_required_spread_improvement")
                    ),
                    "llep_required_foreign_rows": parse_int(
                        status.get("llep_required_foreign_rows")
                    ),
                    "candidate_improvement": parse_int(
                        status.get("candidate_load_spread_improvement_total")
                    ),
                    "candidate_improvement_max": parse_int(
                        status.get("candidate_load_spread_improvement_max")
                    ),
                    "accepted_improvement": accepted_improvement,
                    "accepted_improvement_max": parse_int(
                        status.get("accepted_load_spread_improvement_max")
                    ),
                    "skipped_wave_cost_floor": parse_int(
                        status.get("skipped_wave_cost_floor")
                    ),
                    "skipped_low_router_benefit": parse_int(
                        status.get("skipped_low_router_benefit")
                    ),
                    "skipped_post_load_spread_ceiling": parse_int(
                        status.get("skipped_post_load_spread_ceiling")
                    ),
                    "payload_bucket_requested_slots": payload_requested_slots,
                    "payload_bucket_slots": payload_bucket_slots,
                    "payload_edge_mask": parse_int(status.get("payload_edge_mask")),
                    "router_eligible": parse_int(
                        status.get("router_hot_cache_eligible_dispatches")
                    ),
                    "router_used": parse_int(
                        status.get("router_hot_cache_used_dispatches")
                    ),
                    "router_improved": parse_int(
                        status.get("router_hot_cache_improved_dispatches")
                    ),
                    "router_default_spread": parse_int(
                        status.get("router_hot_cache_default_load_spread_total")
                    ),
                    "router_actual_spread": parse_int(
                        status.get("router_hot_cache_actual_load_spread_total")
                    ),
                    "router_spread_improvement": parse_int(
                        status.get("router_hot_cache_load_spread_improvement_total")
                    ),
                    "router_active": parse_int(
                        status.get("router_hot_cache_active_dispatches")
                    ),
                    "router_miss": parse_int(
                        status.get("router_hot_cache_miss_dispatches")
                    ),
                    "router_selected_slots": parse_int(
                        status.get("router_hot_cache_selected_expert_slots")
                    ),
                    "router_replicated_selected_slots": parse_int(
                        status.get("router_hot_cache_replicated_selected_expert_slots")
                    ),
                    "pre_policy_load_total": parse_int(
                        status.get("pre_policy_load_total")
                    ),
                    "pre_policy_load_min": pre_load_min,
                    "pre_policy_load_max": pre_load_max,
                    "pre_policy_load_spread": pre_load_spread,
                    "pre_policy_imbalance_ratio": format_optional_float(
                        pre_imbalance_ratio
                    ),
                    "post_policy_load_total": post_load_total,
                    "post_policy_load_min": post_load_min,
                    "post_policy_load_max": post_load_max,
                    "post_policy_load_spread": post_load_spread,
                    "post_policy_imbalance_ratio": format_optional_float(
                        post_imbalance_ratio
                    ),
                    "post_wave_load_total": post_wave_load_total,
                    "post_wave_load_spread": post_wave_load_spread,
                    "post_wave_imbalance_ratio": format_optional_float(
                        post_wave_imbalance_ratio
                    ),
                    "apply_post_apply_multi_resident_experts": (
                        post_apply_multi_resident_experts
                    ),
                    "applied_arrivals": applied_arrivals,
                    "applied_remaining_tokens": (
                        decode_tokens_remaining if applied_arrivals > 0 else -1
                    ),
                    "command_entries": sum(op_counts.values()),
                    "transfer_arrivals": op_counts[1],
                    "resident_replicas": op_counts[2],
                    "noop_command_entries": parse_int(
                        trace.get("noop_command_entries")
                    ),
                    "benchmark_json": relative_path(benchmark_path, corpus_dir),
                    "rebalance_trace_jsonl": relative_path(trace_path, corpus_dir),
                }
            )

        first_planned_remaining = (
            max(0, n_predict - first_planned_decode_token)
            if first_planned_decode_token is not None
            else -1
        )
        last_planned_remaining = (
            max(0, n_predict - last_planned_decode_token)
            if last_planned_decode_token is not None
            else -1
        )
        first_applied_remaining = (
            max(0, n_predict - first_applied_decode_token)
            if first_applied_decode_token is not None
            else -1
        )
        last_applied_remaining = (
            max(0, n_predict - last_applied_decode_token)
            if last_applied_decode_token is not None
            else -1
        )
        pre_imbalance_samples = aggregate["pre_policy_imbalance_ratio_samples"]
        post_imbalance_samples = aggregate["post_policy_imbalance_ratio_samples"]
        post_wave_imbalance_samples = aggregate["post_wave_imbalance_ratio_samples"]
        llep_total_rows = aggregate["llep_native_rows"] + aggregate["llep_spilled_rows"]

        run_row = {
            "backend": backend,
            "placement": summary.get("placement", ""),
            "case": summary.get("case", ""),
            "n_predict": n_predict,
            "seed": seed,
            "split": split,
            "rep": parse_int(summary.get("rep")),
            "exit_code": summary.get("exit_code", ""),
            **benchmark,
            "trace_rows": aggregate["trace_rows"],
            "economic_rows": aggregate["economic_rows"],
            "trace_errors": trace_errors,
            "max_epoch": max_epoch,
            "first_planned_decode_token": (
                first_planned_decode_token
                if first_planned_decode_token is not None
                else -1
            ),
            "last_planned_decode_token": (
                last_planned_decode_token
                if last_planned_decode_token is not None
                else -1
            ),
            "first_planned_remaining_tokens": first_planned_remaining,
            "last_planned_remaining_tokens": last_planned_remaining,
            "first_applied_decode_token": (
                first_applied_decode_token
                if first_applied_decode_token is not None
                else -1
            ),
            "last_applied_decode_token": (
                last_applied_decode_token
                if last_applied_decode_token is not None
                else -1
            ),
            "first_applied_remaining_tokens": first_applied_remaining,
            "last_applied_remaining_tokens": last_applied_remaining,
            "planned_arrivals": aggregate["planned_arrivals"],
            "selected_replicas": aggregate["selected_replicas"],
            "changed_layers": aggregate["changed_layers"],
            "apply_post_apply_multi_resident_experts": aggregate[
                "apply_post_apply_multi_resident_experts"
            ],
            "candidate_considered": aggregate["candidate_considered"],
            "candidate_pruned_by_count_bound": aggregate[
                "candidate_pruned_by_count_bound"
            ],
            "candidate_below_floor": aggregate["candidate_below_floor"],
            "llep_assignment_spans": aggregate["llep_assignment_spans"],
            "llep_weight_transfers": aggregate["llep_weight_transfers"],
            "llep_native_rows": aggregate["llep_native_rows"],
            "llep_spilled_rows": aggregate["llep_spilled_rows"],
            "llep_spilled_row_ratio": format_optional_float(
                load_spread_ratio(aggregate["llep_spilled_rows"], llep_total_rows)
            ),
            "llep_spilled_rows_per_transfer": format_optional_float(
                None
                if aggregate["llep_weight_transfers"] <= 0
                else aggregate["llep_spilled_rows"] / aggregate["llep_weight_transfers"]
            ),
            "llep_standard_ep_selected": aggregate["llep_standard_ep_selected"],
            "llep_skipped_balanced": aggregate["llep_skipped_balanced"],
            "llep_skipped_insufficient_spread_improvement": aggregate[
                "llep_skipped_insufficient_spread_improvement"
            ],
            "llep_skipped_insufficient_foreign_rows": aggregate[
                "llep_skipped_insufficient_foreign_rows"
            ],
            "llep_min_chunk_skips": aggregate["llep_min_chunk_skips"],
            "llep_forced_spills": aggregate["llep_forced_spills"],
            "llep_required_spread_improvement": aggregate[
                "llep_required_spread_improvement"
            ],
            "llep_required_foreign_rows": aggregate["llep_required_foreign_rows"],
            "candidate_improvement": aggregate["candidate_improvement"],
            "candidate_improvement_max": aggregate["candidate_improvement_max"],
            "accepted_improvement": aggregate["accepted_improvement"],
            "accepted_improvement_max": aggregate["accepted_improvement_max"],
            "skipped_wave_cost_floor": aggregate["skipped_wave_cost_floor"],
            "skipped_low_router_benefit": aggregate["skipped_low_router_benefit"],
            "skipped_post_load_spread_ceiling": aggregate[
                "skipped_post_load_spread_ceiling"
            ],
            "router_eligible": aggregate["router_eligible"],
            "router_used": aggregate["router_used"],
            "router_improved": aggregate["router_improved"],
            "router_default_spread": aggregate["router_default_spread"],
            "router_actual_spread": aggregate["router_actual_spread"],
            "router_spread_improvement": aggregate["router_spread_improvement"],
            "router_active": aggregate["router_active"],
            "router_miss": aggregate["router_miss"],
            "router_selected_slots": aggregate["router_selected_slots"],
            "router_replicated_selected_slots": aggregate[
                "router_replicated_selected_slots"
            ],
            "pre_policy_load_total": aggregate["pre_policy_load_total"],
            "post_policy_load_total": aggregate["post_policy_load_total"],
            "pre_policy_load_spread": aggregate["pre_policy_load_spread"],
            "post_policy_load_spread": aggregate["post_policy_load_spread"],
            "pre_policy_load_spread_max": aggregate["pre_policy_load_spread_max"],
            "post_policy_load_spread_max": aggregate["post_policy_load_spread_max"],
            "pre_policy_imbalance_ratio_avg": format_optional_float(
                None
                if pre_imbalance_samples <= 0
                else aggregate["pre_policy_imbalance_ratio_sum"] / pre_imbalance_samples
            ),
            "pre_policy_imbalance_ratio_max": format_optional_float(
                None
                if pre_imbalance_samples <= 0
                else aggregate["pre_policy_imbalance_ratio_max"]
            ),
            "pre_policy_imbalance_ratio_samples": pre_imbalance_samples,
            "post_policy_imbalance_ratio_avg": format_optional_float(
                None
                if post_imbalance_samples <= 0
                else aggregate["post_policy_imbalance_ratio_sum"] / post_imbalance_samples
            ),
            "post_policy_imbalance_ratio_max": format_optional_float(
                None
                if post_imbalance_samples <= 0
                else aggregate["post_policy_imbalance_ratio_max"]
            ),
            "post_policy_imbalance_ratio_samples": post_imbalance_samples,
            "post_wave_load_total": aggregate["post_wave_load_total"],
            "post_wave_load_spread": aggregate["post_wave_load_spread"],
            "post_wave_load_spread_max": aggregate["post_wave_load_spread_max"],
            "post_wave_imbalance_ratio_avg": format_optional_float(
                None
                if post_wave_imbalance_samples <= 0
                else aggregate["post_wave_imbalance_ratio_sum"] / post_wave_imbalance_samples
            ),
            "post_wave_imbalance_ratio_max": format_optional_float(
                None
                if post_wave_imbalance_samples <= 0
                else aggregate["post_wave_imbalance_ratio_max"]
            ),
            "post_wave_imbalance_ratio_samples": post_wave_imbalance_samples,
            "command_entries": aggregate["command_entries"],
            "transfer_arrivals": aggregate["transfer_arrivals"],
            "resident_replicas": aggregate["resident_replicas"],
            "noop_command_entries": aggregate["noop_command_entries"],
            "payload_bucket_slots_max": max_payload_bucket_slots,
            "payload_bucket_requested_slots_max": max_payload_bucket_requested_slots,
            "benchmark_json": relative_path(benchmark_path, corpus_dir),
            "rebalance_trace_jsonl": relative_path(trace_path, corpus_dir),
        }
        run_rows.append(run_row)

    return run_rows, window_rows


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


def average_numeric_field(rows: list[dict[str, Any]], field: str) -> str:
    values = [
        parse_float(row[field])
        for row in rows
        if field in row and row[field] != ""
    ]
    if not values:
        return ""
    return f"{sum(values) / len(values):.9g}"


def grouped_summary(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        groups[(row["backend"], row["n_predict"], row["split"])].append(row)

    output: list[dict[str, Any]] = []
    for (backend, n_predict, split), rows in sorted(groups.items()):
        count = len(rows)
        output.append(
            {
                "backend": backend,
                "n_predict": n_predict,
                "split": split,
                "runs": count,
                "decode_tok_s_mean": sum(row["decode_tok_s"] for row in rows) / count,
                "decode_mean_ms_mean": sum(row["decode_mean_ms"] for row in rows) / count,
                "decode_p50_ms_mean": sum(row["decode_p50_ms"] for row in rows) / count,
                "decode_p90_ms_mean": sum(row["decode_p90_ms"] for row in rows) / count,
                "accepted_improvement_total": sum(
                    row["accepted_improvement"] for row in rows
                ),
                "llep_assignment_spans_total": sum(
                    row["llep_assignment_spans"] for row in rows
                ),
                "llep_weight_transfers_total": sum(
                    row["llep_weight_transfers"] for row in rows
                ),
                "llep_spilled_rows_total": sum(
                    row["llep_spilled_rows"] for row in rows
                ),
                "llep_spilled_row_ratio_avg_mean": average_numeric_field(
                    rows,
                    "llep_spilled_row_ratio",
                ),
                "llep_spilled_rows_per_transfer_avg_mean": average_numeric_field(
                    rows,
                    "llep_spilled_rows_per_transfer",
                ),
                "pre_policy_imbalance_ratio_avg_mean": average_numeric_field(
                    rows,
                    "pre_policy_imbalance_ratio_avg",
                ),
                "post_policy_imbalance_ratio_avg_mean": average_numeric_field(
                    rows,
                    "post_policy_imbalance_ratio_avg",
                ),
                "post_wave_imbalance_ratio_avg_mean": average_numeric_field(
                    rows,
                    "post_wave_imbalance_ratio_avg",
                ),
                "transfer_arrivals_total": sum(row["transfer_arrivals"] for row in rows),
                "resident_replicas_total": sum(row["resident_replicas"] for row in rows),
                "router_used_total": sum(row["router_used"] for row in rows),
                "router_improved_total": sum(row["router_improved"] for row in rows),
                "router_active_total": sum(row["router_active"] for row in rows),
                "router_miss_total": sum(row["router_miss"] for row in rows),
                "router_selected_slots_total": sum(
                    row["router_selected_slots"] for row in rows
                ),
                "router_replicated_selected_slots_total": sum(
                    row["router_replicated_selected_slots"] for row in rows
                ),
            }
        )
    return output


def print_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No completed runs found.")
        return
    header = (
        "backend n_predict split runs decode_tok_s_mean mean_ms_mean "
        "p50_ms_mean p90_ms_mean pre_imbalance_avg post_imbalance_avg "
        "llep_spilled_rows llep_weight_transfers transfers resident router_used "
        "accepted_improvement"
    )
    print(header)
    for row in rows:
        print(
            f"{row['backend']} {row['n_predict']} {row['split']} {row['runs']} "
            f"{row['decode_tok_s_mean']:.3f} {row['decode_mean_ms_mean']:.3f} "
            f"{row['decode_p50_ms_mean']:.3f} {row['decode_p90_ms_mean']:.3f} "
            f"{row['pre_policy_imbalance_ratio_avg_mean']} "
            f"{row['post_policy_imbalance_ratio_avg_mean']} "
            f"{row['llep_spilled_rows_total']} "
            f"{row['llep_weight_transfers_total']} "
            f"{row['transfer_arrivals_total']} "
            f"{row['resident_replicas_total']} {row['router_used_total']} "
            f"{row['accepted_improvement_total']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CSV outputs; defaults to <corpus_dir>/analysis.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print grouped split summary to stdout.",
    )
    args = parser.parse_args()

    corpus_dir = args.corpus_dir
    output_dir = args.output_dir or corpus_dir / "analysis"
    run_rows, window_rows = build_tables(corpus_dir)
    split_rows = grouped_summary(run_rows)

    write_csv(output_dir / "run_summary.csv", run_rows)
    write_csv(output_dir / "window_metrics.csv", window_rows)
    write_csv(output_dir / "split_summary.csv", split_rows)
    if args.print_summary:
        print_summary(split_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
