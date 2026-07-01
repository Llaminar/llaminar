#!/usr/bin/env python3
"""Compare dynamic-hot MoE rebalance traces against paired baseline runs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


KEY_FIELDS = ("backend", "n_predict", "seed", "rep")
NUMERIC_FIELDS = (
    "decode_tok_s",
    "decode_mean_ms",
    "decode_p50_ms",
    "decode_p90_ms",
    "transfer_arrivals",
    "resident_replicas",
    "router_used",
    "accepted_improvement",
)

FEATURE_FIELDS = (
    "planned_arrivals",
    "changed_layers",
    "apply_post_apply_multi_resident_experts",
    "transfer_arrivals",
    "resident_replicas",
    "router_active",
    "router_miss",
    "router_eligible",
    "router_used",
    "router_improved",
    "router_selected_slots",
    "router_replicated_selected_slots",
    "router_default_spread",
    "router_actual_spread",
    "router_spread_improvement",
    "accepted_improvement",
    "accepted_improvement_max",
    "first_planned_decode_token",
    "last_planned_decode_token",
    "first_planned_remaining_tokens",
    "last_planned_remaining_tokens",
    "first_applied_decode_token",
    "last_applied_decode_token",
    "first_applied_remaining_tokens",
    "last_applied_remaining_tokens",
    "skipped_wave_cost_floor",
    "skipped_low_router_benefit",
    "skipped_post_load_spread_ceiling",
    "candidate_improvement",
    "candidate_improvement_max",
    "candidate_below_floor",
    "selected_replicas",
    "command_entries",
    "pre_policy_load_total",
    "post_policy_load_total",
    "pre_policy_load_spread",
    "post_policy_load_spread",
    "pre_policy_load_spread_max",
    "post_policy_load_spread_max",
    "pre_policy_imbalance_ratio_avg",
    "pre_policy_imbalance_ratio_max",
    "pre_policy_imbalance_ratio_samples",
    "post_policy_imbalance_ratio_avg",
    "post_policy_imbalance_ratio_max",
    "post_policy_imbalance_ratio_samples",
    "post_wave_load_total",
    "post_wave_load_spread",
    "post_wave_load_spread_max",
    "post_wave_imbalance_ratio_avg",
    "post_wave_imbalance_ratio_max",
    "post_wave_imbalance_ratio_samples",
    "payload_bucket_slots_max",
    "payload_bucket_requested_slots_max",
)

DERIVED_FEATURE_FIELDS = (
    "resident_fraction",
    "apply_multi_resident_per_transfer",
    "accepted_improvement_per_transfer",
    "accepted_improvement_per_command",
    "router_miss_rate",
    "router_eligible_rate",
    "router_use_rate_active",
    "router_replicated_selected_slot_ratio",
    "router_selected_slots_per_active_dispatch",
    "router_replicated_slots_per_active_dispatch",
    "router_spread_improvement_per_used_dispatch",
    "router_spread_improvement_per_eligible_dispatch",
    "router_spread_improvement_per_selected_slot",
    "router_spread_improvement_per_replicated_selected_slot",
    "post_vs_pre_load_spread_ratio",
    "pre_policy_load_spread_fraction",
    "post_policy_load_spread_fraction",
    "post_policy_load_spread_per_transfer",
    "post_wave_load_spread_fraction",
    "post_wave_load_spread_per_transfer",
    "post_vs_pre_imbalance_ratio",
    "post_wave_vs_pre_imbalance_ratio",
    "imbalance_ratio_delta",
    "load_spread_delta",
    "first_planned_remaining_fraction",
    "last_planned_remaining_fraction",
    "first_applied_remaining_fraction",
    "last_applied_remaining_fraction",
    "first_plan_to_apply_lag_tokens",
    "last_plan_to_apply_lag_tokens",
    "router_use_rate",
    "router_improve_rate",
)

LABEL_FIELDS = (
    "hot_minus_static_decode_tok_s",
    "hot_minus_dynamic_decode_tok_s",
    "dynamic_minus_static_decode_tok_s",
)


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def index_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str, str], dict[str, str]]:
    indexed: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in KEY_FIELDS) + (row.get("case", ""),)
        indexed[key] = row
    return indexed


def lookup(
    indexed: dict[tuple[str, str, str, str, str], dict[str, str]],
    hot_row: dict[str, str],
    case: str,
) -> dict[str, str] | None:
    key = tuple(hot_row.get(field, "") for field in KEY_FIELDS) + (case,)
    return indexed.get(key)


def lookup_by_key(
    indexed: dict[tuple[str, str, str, str, str], dict[str, str]],
    key_row: dict[str, str],
    case: str,
) -> dict[str, str] | None:
    key = tuple(key_row.get(field, "") for field in KEY_FIELDS) + (case,)
    return indexed.get(key)


def delta(hot_row: dict[str, str], baseline_row: dict[str, str] | None, field: str) -> str:
    if baseline_row is None:
        return ""
    hot = parse_float(hot_row.get(field))
    baseline = parse_float(baseline_row.get(field))
    if hot is None or baseline is None:
        return ""
    return f"{hot - baseline:.9g}"


def format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.9g}"


def value(row: dict[str, str] | None, field: str) -> str:
    return "" if row is None else row.get(field, "")


def numeric_value(row: dict[str, str] | None, field: str) -> float:
    if row is None:
        return 0.0
    return parse_float(row.get(field)) or 0.0


def ratio(numerator: float, denominator: float) -> str:
    if denominator <= 0.0:
        return ""
    return f"{numerator / denominator:.9g}"


def build_delta_rows(
    dynamic_hot_rows: list[dict[str, str]],
    baseline_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    baseline_index = index_rows(baseline_rows)
    output: list[dict[str, str]] = []
    for hot in dynamic_hot_rows:
        if hot.get("case") != "dynamic_hot10":
            continue
        static = lookup(baseline_index, hot, "static")
        dynamic = lookup(baseline_index, hot, "dynamic")
        row: dict[str, str] = {
            "backend": hot.get("backend", ""),
            "n_predict": hot.get("n_predict", ""),
            "seed": hot.get("seed", ""),
            "split": hot.get("split", ""),
            "rep": hot.get("rep", ""),
            "hot_decode_tok_s": hot.get("decode_tok_s", ""),
            "static_decode_tok_s": value(static, "decode_tok_s"),
            "dynamic_decode_tok_s": value(dynamic, "decode_tok_s"),
            "hot_minus_static_decode_tok_s": delta(hot, static, "decode_tok_s"),
            "hot_minus_dynamic_decode_tok_s": delta(hot, dynamic, "decode_tok_s"),
            "dynamic_minus_static_decode_tok_s": delta(dynamic or {}, static, "decode_tok_s"),
            "hot_decode_mean_ms": hot.get("decode_mean_ms", ""),
            "static_decode_mean_ms": value(static, "decode_mean_ms"),
            "dynamic_decode_mean_ms": value(dynamic, "decode_mean_ms"),
            "hot_minus_static_decode_mean_ms": delta(hot, static, "decode_mean_ms"),
            "hot_minus_dynamic_decode_mean_ms": delta(hot, dynamic, "decode_mean_ms"),
            "hot_minus_static_decode_p90_ms": delta(hot, static, "decode_p90_ms"),
            "hot_minus_dynamic_decode_p90_ms": delta(hot, dynamic, "decode_p90_ms"),
        }
        for field in (
            "transfer_arrivals",
            "resident_replicas",
            "router_used",
            "accepted_improvement",
        ):
            row[f"hot_{field}"] = hot.get(field, "")
        output.append(row)
    return sorted(output, key=lambda r: (r["backend"], int(r["n_predict"]), int(r["seed"])))


def build_feature_label_rows(
    dynamic_hot_rows: list[dict[str, str]],
    baseline_rows: list[dict[str, str]],
    feature_rows: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Join clean policy labels to controller-observable trace features.

    dynamic_hot_rows are treated as the clean label source. feature_rows can come
    from a traced run of the same backend/length/seed/rep matrix; when omitted,
    the dynamic_hot_rows themselves are used as both labels and features.
    """

    baseline_index = index_rows(baseline_rows)
    feature_index = index_rows(feature_rows or dynamic_hot_rows)
    output: list[dict[str, str]] = []

    for hot in dynamic_hot_rows:
        if hot.get("case") != "dynamic_hot10":
            continue
        static = lookup_by_key(baseline_index, hot, "static")
        dynamic = lookup_by_key(baseline_index, hot, "dynamic")
        features = lookup_by_key(feature_index, hot, "dynamic_hot10")
        row: dict[str, str] = {
            "backend": hot.get("backend", ""),
            "n_predict": hot.get("n_predict", ""),
            "seed": hot.get("seed", ""),
            "split": hot.get("split", ""),
            "rep": hot.get("rep", ""),
            "hot_decode_tok_s": hot.get("decode_tok_s", ""),
            "static_decode_tok_s": value(static, "decode_tok_s"),
            "dynamic_decode_tok_s": value(dynamic, "decode_tok_s"),
            "hot_minus_static_decode_tok_s": delta(hot, static, "decode_tok_s"),
            "hot_minus_dynamic_decode_tok_s": delta(hot, dynamic, "decode_tok_s"),
            "dynamic_minus_static_decode_tok_s": delta(dynamic or {}, static, "decode_tok_s"),
            "hot_decode_mean_ms": hot.get("decode_mean_ms", ""),
            "static_decode_mean_ms": value(static, "decode_mean_ms"),
            "dynamic_decode_mean_ms": value(dynamic, "decode_mean_ms"),
            "hot_minus_static_decode_mean_ms": delta(hot, static, "decode_mean_ms"),
            "hot_minus_dynamic_decode_mean_ms": delta(hot, dynamic, "decode_mean_ms"),
        }
        for field in FEATURE_FIELDS:
            row[field] = value(features, field)

        if features is None:
            for field in DERIVED_FEATURE_FIELDS:
                row[field] = ""
            output.append(row)
            continue

        transfer_arrivals = numeric_value(features, "transfer_arrivals")
        resident_replicas = numeric_value(features, "resident_replicas")
        command_entries = numeric_value(features, "command_entries")
        apply_multi_resident = numeric_value(
            features,
            "apply_post_apply_multi_resident_experts",
        )
        router_active = numeric_value(features, "router_active")
        router_miss = numeric_value(features, "router_miss")
        router_eligible = numeric_value(features, "router_eligible")
        router_used = numeric_value(features, "router_used")
        router_selected_slots = numeric_value(features, "router_selected_slots")
        router_replicated_selected_slots = numeric_value(
            features,
            "router_replicated_selected_slots",
        )
        accepted_improvement = numeric_value(features, "accepted_improvement")
        router_spread_improvement = numeric_value(features, "router_spread_improvement")
        pre_policy_load_spread = numeric_value(features, "pre_policy_load_spread")
        post_policy_load_spread = numeric_value(features, "post_policy_load_spread")
        pre_policy_load_total = numeric_value(features, "pre_policy_load_total")
        post_policy_load_total = numeric_value(features, "post_policy_load_total")
        post_wave_load_spread = numeric_value(features, "post_wave_load_spread")
        post_wave_load_total = numeric_value(features, "post_wave_load_total")
        pre_policy_imbalance_ratio_avg = numeric_value(
            features,
            "pre_policy_imbalance_ratio_avg",
        )
        post_policy_imbalance_ratio_avg = numeric_value(
            features,
            "post_policy_imbalance_ratio_avg",
        )
        post_wave_imbalance_ratio_avg = numeric_value(
            features,
            "post_wave_imbalance_ratio_avg",
        )
        n_predict_value = numeric_value(features or hot, "n_predict")
        first_planned_token = numeric_value(features, "first_planned_decode_token")
        last_planned_token = numeric_value(features, "last_planned_decode_token")
        first_planned_remaining = numeric_value(features, "first_planned_remaining_tokens")
        last_planned_remaining = numeric_value(features, "last_planned_remaining_tokens")
        first_applied_token = numeric_value(features, "first_applied_decode_token")
        last_applied_token = numeric_value(features, "last_applied_decode_token")
        first_applied_remaining = numeric_value(features, "first_applied_remaining_tokens")
        last_applied_remaining = numeric_value(features, "last_applied_remaining_tokens")
        row["resident_fraction"] = ratio(resident_replicas, command_entries)
        row["apply_multi_resident_per_transfer"] = ratio(
            apply_multi_resident,
            transfer_arrivals,
        )
        row["accepted_improvement_per_transfer"] = ratio(
            accepted_improvement,
            max(transfer_arrivals, 1.0),
        )
        row["accepted_improvement_per_command"] = ratio(
            accepted_improvement,
            max(command_entries, 1.0),
        )
        row["router_miss_rate"] = ratio(router_miss, router_active)
        row["router_eligible_rate"] = ratio(router_eligible, router_active)
        row["router_use_rate_active"] = ratio(router_used, router_active)
        row["router_replicated_selected_slot_ratio"] = ratio(
            router_replicated_selected_slots,
            router_selected_slots,
        )
        row["router_selected_slots_per_active_dispatch"] = ratio(
            router_selected_slots,
            router_active,
        )
        row["router_replicated_slots_per_active_dispatch"] = ratio(
            router_replicated_selected_slots,
            router_active,
        )
        row["router_spread_improvement_per_used_dispatch"] = ratio(
            router_spread_improvement,
            router_used,
        )
        row["router_spread_improvement_per_eligible_dispatch"] = ratio(
            router_spread_improvement,
            router_eligible,
        )
        row["router_spread_improvement_per_selected_slot"] = ratio(
            router_spread_improvement,
            router_selected_slots,
        )
        row["router_spread_improvement_per_replicated_selected_slot"] = ratio(
            router_spread_improvement,
            router_replicated_selected_slots,
        )
        row["post_vs_pre_load_spread_ratio"] = ratio(
            post_policy_load_spread,
            pre_policy_load_spread,
        )
        row["pre_policy_load_spread_fraction"] = ratio(
            pre_policy_load_spread,
            pre_policy_load_total,
        )
        row["post_policy_load_spread_fraction"] = ratio(
            post_policy_load_spread,
            post_policy_load_total,
        )
        row["post_policy_load_spread_per_transfer"] = ratio(
            post_policy_load_spread,
            max(transfer_arrivals, 1.0),
        )
        row["post_wave_load_spread_fraction"] = ratio(
            post_wave_load_spread,
            post_wave_load_total,
        )
        row["post_wave_load_spread_per_transfer"] = ratio(
            post_wave_load_spread,
            max(transfer_arrivals, 1.0),
        )
        row["post_vs_pre_imbalance_ratio"] = ratio(
            post_policy_imbalance_ratio_avg,
            pre_policy_imbalance_ratio_avg,
        )
        row["post_wave_vs_pre_imbalance_ratio"] = ratio(
            post_wave_imbalance_ratio_avg,
            pre_policy_imbalance_ratio_avg,
        )
        row["imbalance_ratio_delta"] = (
            format_float(post_policy_imbalance_ratio_avg - pre_policy_imbalance_ratio_avg)
            if pre_policy_imbalance_ratio_avg > 0.0
            else ""
        )
        row["load_spread_delta"] = format_float(
            post_policy_load_spread - pre_policy_load_spread
        )
        row["first_planned_remaining_fraction"] = ratio(
            first_planned_remaining,
            n_predict_value,
        ) if first_planned_remaining >= 0.0 else ""
        row["last_planned_remaining_fraction"] = ratio(
            last_planned_remaining,
            n_predict_value,
        ) if last_planned_remaining >= 0.0 else ""
        row["first_applied_remaining_fraction"] = ratio(
            first_applied_remaining,
            n_predict_value,
        ) if first_applied_remaining >= 0.0 else ""
        row["last_applied_remaining_fraction"] = ratio(
            last_applied_remaining,
            n_predict_value,
        ) if last_applied_remaining >= 0.0 else ""
        row["first_plan_to_apply_lag_tokens"] = (
            format_float(first_applied_token - first_planned_token)
            if first_planned_token >= 0.0 and first_applied_token >= 0.0
            else ""
        )
        row["last_plan_to_apply_lag_tokens"] = (
            format_float(last_applied_token - last_planned_token)
            if last_planned_token >= 0.0 and last_applied_token >= 0.0
            else ""
        )
        row["router_use_rate"] = ratio(router_used, router_eligible)
        row["router_improve_rate"] = ratio(
            numeric_value(features, "router_improved"),
            router_used,
        )
        output.append(row)

    return sorted(output, key=lambda r: (r["backend"], int(r["n_predict"]), int(r["seed"])))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def numeric(row: dict[str, str], field: str) -> float | None:
    return parse_float(row.get(field))


def threshold_summary(feature_label_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    thresholds: list[dict[str, str]] = []
    feature_fields = FEATURE_FIELDS + DERIVED_FEATURE_FIELDS
    for label in LABEL_FIELDS:
        labelled_rows = [row for row in feature_label_rows if numeric(row, label) is not None]
        for feature in feature_fields:
            values = sorted(
                {
                    value
                    for value in (numeric(row, feature) for row in labelled_rows)
                    if value is not None
                }
            )
            for threshold in values:
                for direction in (">=", "<="):
                    selected = [
                        row
                        for row in labelled_rows
                        if (numeric(row, feature) is not None)
                        and (
                            numeric(row, feature) >= threshold
                            if direction == ">="
                            else numeric(row, feature) <= threshold
                        )
                    ]
                    if len(selected) < 3:
                        continue
                    label_values = [
                        numeric(row, label)
                        for row in selected
                        if numeric(row, label) is not None
                    ]
                    if not label_values:
                        continue
                    positives = [value for value in label_values if value > 0.0]
                    thresholds.append(
                        {
                            "label": label,
                            "feature": feature,
                            "direction": direction,
                            "threshold": format_float(threshold),
                            "selected_rows": str(len(selected)),
                            "coverage": format_float(
                                len(selected) / len(labelled_rows)
                                if labelled_rows
                                else None
                            ),
                            "positive_rate": format_float(
                                len(positives) / len(label_values)
                            ),
                            "label_delta_mean": format_float(
                                sum(label_values) / len(label_values)
                            ),
                        }
                    )

    return sorted(
        thresholds,
        key=lambda row: (
            row["label"],
            -(parse_float(row["label_delta_mean"]) or 0.0),
            -(parse_float(row["positive_rate"]) or 0.0),
            -(parse_float(row["coverage"]) or 0.0),
        ),
    )


def summarize(delta_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in delta_rows:
        groups[(row["backend"], row["n_predict"], row["split"])].append(row)

    output: list[dict[str, str]] = []
    for (backend, n_predict, split), rows in sorted(groups.items()):
        summary: dict[str, str] = {
            "backend": backend,
            "n_predict": n_predict,
            "split": split,
            "hot_rows": str(len(rows)),
        }
        static_pairs = [
            row for row in rows if numeric(row, "hot_minus_static_decode_tok_s") is not None
        ]
        dynamic_pairs = [
            row for row in rows if numeric(row, "hot_minus_dynamic_decode_tok_s") is not None
        ]
        summary["static_pairs"] = str(len(static_pairs))
        summary["dynamic_pairs"] = str(len(dynamic_pairs))
        for field in (
            "hot_minus_static_decode_tok_s",
            "hot_minus_dynamic_decode_tok_s",
            "dynamic_minus_static_decode_tok_s",
            "hot_minus_static_decode_mean_ms",
            "hot_minus_dynamic_decode_mean_ms",
            "hot_minus_static_decode_p90_ms",
            "hot_minus_dynamic_decode_p90_ms",
        ):
            values = [numeric(row, field) for row in rows]
            values = [v for v in values if v is not None]
            summary[f"{field}_mean"] = (
                "" if not values else f"{sum(values) / len(values):.9g}"
            )
        output.append(summary)
    return output


def print_summary(rows: list[dict[str, str]]) -> None:
    if not rows:
        print("No paired rows yet.")
        return
    print(
        "backend n_predict split hot_rows static_pairs dynamic_pairs "
        "hot-static_tok_s hot-dynamic_tok_s "
        "dynamic-static_tok_s hot-static_mean_ms hot-static_p90_ms"
    )
    for row in rows:
        print(
            f"{row['backend']} {row['n_predict']} {row['split']} "
            f"{row['hot_rows']} {row['static_pairs']} {row['dynamic_pairs']} "
            f"{row['hot_minus_static_decode_tok_s_mean']} "
            f"{row['hot_minus_dynamic_decode_tok_s_mean']} "
            f"{row['dynamic_minus_static_decode_tok_s_mean']} "
            f"{row['hot_minus_static_decode_mean_ms_mean']} "
            f"{row['hot_minus_static_decode_p90_ms_mean']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic-hot-run-summary", required=True, type=Path)
    parser.add_argument("--baseline-run-summary", required=True, type=Path)
    parser.add_argument(
        "--feature-run-summary",
        type=Path,
        default=None,
        help=(
            "Optional traced dynamic-hot run_summary.csv used as the feature source. "
            "Defaults to --dynamic-hot-run-summary."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()

    hot_rows = read_rows(args.dynamic_hot_run_summary)
    baseline_rows = read_rows(args.baseline_run_summary)
    feature_rows = read_rows(args.feature_run_summary) if args.feature_run_summary else hot_rows
    delta_rows = build_delta_rows(hot_rows, baseline_rows)
    feature_label_rows = build_feature_label_rows(hot_rows, baseline_rows, feature_rows)
    threshold_rows = threshold_summary(feature_label_rows)
    summary_rows = summarize(delta_rows)
    write_csv(args.output_dir / "policy_delta_by_seed.csv", delta_rows)
    write_csv(args.output_dir / "policy_delta_summary.csv", summary_rows)
    write_csv(args.output_dir / "policy_feature_labels.csv", feature_label_rows)
    write_csv(args.output_dir / "policy_feature_thresholds.csv", threshold_rows)
    if args.print_summary:
        print_summary(summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
