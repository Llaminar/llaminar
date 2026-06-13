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
    "stochastic_physical_verify_rows",
    "stochastic_semantic_verify_rows",
    "stochastic_post_reject_rows",
    "condition_ms",
    "condition_count",
    "condition_skipped_ready",
    "correction_ms",
    "correction_count",
    "deferred_corrections",
    "rejection_no_ready",
    "publish_ms",
    "publish_count",
    "publish_avg_ms",
    "sidecar_ms",
    "sidecar_depth0_decode_ms",
    "shifted_initial_ms",
    "shifted_initial_commits",
    "shifted_initial_reused",
    "shifted_prefix_ms",
    "shifted_deferred_ms",
    "shifted_row_ms",
    "shifted_kv_ready_events",
    "shifted_kv_ready_waits",
    "shifted_kv_syncs_deferred",
    "sampling_ms",
    "sampling_enqueue_ms",
    "stochastic_batch_outcome_ms",
    "stochastic_batch_d2h_sync_ms",
    "greedy_summary_ms",
    "checkpoint_ms",
    "sidecar_graph_hits",
    "sidecar_graph_misses",
    "main_decode_warmup",
    "main_decode_capture",
    "main_decode_replay",
    "main_verifier_warmup",
    "main_verifier_capture",
    "main_verifier_replay",
    "replay_resets",
    "replay_preserves",
    "replay_reset_caches",
    "replay_rebind_caches",
    "replay_ordinary_decode_resets",
    "replay_verifier_rebinds",
    "replay_other_rebinds",
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


def _sum_total_ms(
    records: Iterable[dict[str, Any]],
    domain: str,
    name: str,
    *,
    phase: str | None = None,
) -> float:
    total = 0.0
    for record in records:
        if _matches(record, domain=domain, name=name, phase=phase):
            total += float(record.get("total_ms", 0.0) or 0.0)
    return total


def _sum_total_ms_many(
    records: Iterable[dict[str, Any]],
    domain: str,
    names: Iterable[str],
    *,
    phase: str | None = None,
) -> float:
    name_set = set(names)
    total = 0.0
    for record in records:
        name = record.get("name")
        if name not in name_set:
            continue
        if _matches(record, domain=domain, name=str(name), phase=phase):
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


def _sum_value(
    records: Iterable[dict[str, Any]],
    domain: str,
    name: str,
    *,
    phase: str | None = None,
    tags: dict[str, str] | None = None,
) -> float:
    total = 0.0
    for record in records:
        if _matches(record, domain=domain, name=name, phase=phase, tags=tags):
            total += float(record.get("value", 0.0) or 0.0)
    return total


def _sum_tagged_int(
    records: Iterable[dict[str, Any]],
    domain: str,
    name: str,
    tag_name: str,
    *,
    phase: str | None = None,
    tags: dict[str, str] | None = None,
) -> int:
    total = 0
    for record in records:
        if not _matches(record, domain=domain, name=name, phase=phase, tags=tags):
            continue
        record_tags = _tags(record)
        try:
            tagged_value = int(record_tags.get(tag_name, "0") or 0)
        except ValueError:
            tagged_value = 0
        total += tagged_value * int(record.get("count", 0) or 0)
    return total


def summarize(path: Path | None) -> dict[str, float | int]:
    records = _records(path)
    publish_ms = _sum_total_ms(
        records,
        "mtp",
        "all_position_publish_accepted_state",
        phase="decode",
    )
    publish_count = _sum_count(
        records,
        "mtp",
        "all_position_publish_accepted_state",
        phase="decode",
    )
    shifted_row_ms = _sum_total_ms_many(
        records,
        "mtp",
        (
            "shifted_row_commit",
            "shifted_row_device_target_commit",
            "shifted_row_sequential_commit",
        ),
        phase="decode",
    )
    sampling_ms = _sum_total_ms_many(
        records,
        "mtp",
        (
            "all_position_stochastic_device_batch_outcome",
            "all_position_stochastic_host_target_distribution",
            "all_position_verifier_greedy_device_summary",
            "all_position_verifier_sample_rows",
            "sample_first_token_device",
            "sample_first_token_host",
            "sample_first_token_stochastic",
            "sample_first_token_stochastic_device",
            "sample_mtp_token_device",
            "sample_mtp_token_host",
            "sample_mtp_token_stochastic",
            "sample_mtp_token_stochastic_device",
            "sample_mtp_token_stochastic_distribution",
            "sample_stochastic_distribution_enqueue",
        ),
        phase="decode",
    )
    sampling_enqueue_ms = _sum_total_ms_many(
        records,
        "mtp",
        (
            "sample_stochastic_distribution_enqueue",
            "stochastic_distribution_batch_build_enqueue",
            "stochastic_batch_verify_enqueue",
            "stochastic_batch_bonus_sample_enqueue",
            "stochastic_batch_summary_enqueue",
        ),
        phase="decode",
    )
    stochastic_batch_outcome_ms = _sum_total_ms(
        records,
        "mtp",
        "all_position_stochastic_device_batch_outcome",
        phase="decode",
    )
    stochastic_batch_d2h_sync_ms = _sum_total_ms(
        records,
        "mtp",
        "stochastic_batch_summary_d2h_sync",
        phase="decode",
    )
    greedy_summary_ms = _sum_total_ms(
        records,
        "mtp",
        "all_position_verifier_greedy_device_summary",
        phase="decode",
    )
    checkpoint_ms = _sum_total_ms_many(
        records,
        "mtp",
        (
            "capture_live_prefix_state",
            "capture_post_sidecar_prefix_state",
            "capture_verifier_base_prefix_state",
            "live_prefix_checkpoint_hybrid_export",
            "live_prefix_checkpoint_hybrid_storage",
            "live_prefix_checkpoint_layout",
        ),
        phase="decode",
    )
    return {
        "decode_step_ms": _sum_total_ms(records, "mtp", "decode_step_total"),
        "verifier_ms": _sum_total_ms(records, "mtp", "verifier_forward"),
        "stochastic_physical_verify_rows": _sum_value(
            records,
            "mtp",
            "stochastic_device_physical_verify_rows",
            phase="decode",
        ),
        "stochastic_semantic_verify_rows": _sum_value(
            records,
            "mtp",
            "stochastic_device_semantic_verify_rows",
            phase="decode",
        ),
        "stochastic_post_reject_rows": _sum_value(
            records,
            "mtp",
            "stochastic_device_post_reject_rows",
            phase="decode",
        ),
        "condition_ms": _sum_total_ms(records, "mtp", "condition_forward"),
        "condition_count": _sum_count(records, "mtp", "condition_forward"),
        "condition_skipped_ready": _sum_count(
            records,
            "mtp",
            "condition_forward_skipped_ready_logits",
            phase="decode",
        ),
        "correction_ms": _sum_total_ms(records, "mtp", "all_position_correction_forward"),
        "correction_count": _sum_count(records, "mtp", "all_position_correction_forward"),
        "deferred_corrections": _sum_count(
            records,
            "mtp",
            "all_position_deferred_correction_condition_tokens",
            phase="decode",
        ),
        "rejection_no_ready": _sum_count(
            records,
            "mtp",
            "all_position_rejection_without_ready_token",
            phase="decode",
        ),
        "publish_ms": publish_ms,
        "publish_count": publish_count,
        "publish_avg_ms": publish_ms / publish_count if publish_count else 0.0,
        "sidecar_ms": _sum_total_ms(records, "mtp", "sidecar_forward", phase="decode"),
        "sidecar_depth0_decode_ms": _sum_total_ms(
            records,
            "mtp",
            "sidecar_depth0_total",
            phase="decode",
        ),
        "shifted_initial_ms": _sum_total_ms(
            records,
            "mtp",
            "all_position_initial_shifted_commit",
            phase="decode",
        ),
        "shifted_initial_commits": _sum_count(
            records,
            "mtp",
            "all_position_initial_shifted_commits",
            phase="decode",
        ),
        "shifted_initial_reused": _sum_count(
            records,
            "mtp",
            "all_position_initial_shifted_reused_sidecar_rows",
            phase="decode",
        ),
        "shifted_prefix_ms": _sum_total_ms(
            records,
            "mtp",
            "all_position_shifted_prefix_commit",
            phase="decode",
        ),
        "shifted_deferred_ms": _sum_total_ms(
            records,
            "mtp",
            "all_position_deferred_correction_shifted_commit",
            phase="decode",
        ),
        "shifted_row_ms": shifted_row_ms,
        "shifted_kv_ready_events": _sum_count(
            records,
            "mtp",
            "shifted_mtp_kv_ready_events",
            phase="decode",
        ),
        "shifted_kv_ready_waits": _sum_count(
            records,
            "mtp",
            "shifted_mtp_kv_ready_waits",
            phase="decode",
        ),
        "shifted_kv_syncs_deferred": _sum_count(
            records,
            "mtp",
            "shifted_mtp_kv_stream_syncs_deferred",
            phase="decode",
        ),
        "sampling_ms": sampling_ms,
        "sampling_enqueue_ms": sampling_enqueue_ms,
        "stochastic_batch_outcome_ms": stochastic_batch_outcome_ms,
        "stochastic_batch_d2h_sync_ms": stochastic_batch_d2h_sync_ms,
        "greedy_summary_ms": greedy_summary_ms,
        "checkpoint_ms": checkpoint_ms,
        "sidecar_graph_hits": _sum_count(
            records,
            "mtp",
            "sidecar_graph_cache_hits",
            phase="decode",
        ),
        "sidecar_graph_misses": _sum_count(
            records,
            "mtp",
            "sidecar_graph_cache_misses",
            phase="decode",
        ),
        "main_decode_warmup": _sum_count(
            records,
            "forward_graph",
            "decode_segmented_phase",
            phase="decode",
            tags={"context": "main_decode", "phase": "warmup"},
        ),
        "main_decode_capture": _sum_count(
            records,
            "forward_graph",
            "decode_segmented_phase",
            phase="decode",
            tags={"context": "main_decode", "phase": "capture"},
        ),
        "main_decode_replay": _sum_count(
            records,
            "forward_graph",
            "decode_segmented_phase",
            phase="decode",
            tags={"context": "main_decode", "phase": "replay"},
        ),
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
        "replay_reset_caches": _sum_tagged_int(
            records,
            "mtp",
            "live_prefix_replay_state_after_mutation",
            "forward_replay_reset_cache_count",
            phase="decode",
            tags={"replay_state": "reset"},
        ),
        "replay_rebind_caches": _sum_tagged_int(
            records,
            "mtp",
            "live_prefix_replay_state_after_mutation",
            "forward_replay_stream_rebind_cache_count",
            phase="decode",
            tags={"replay_state": "reset"},
        ),
        "replay_ordinary_decode_resets": _sum_tagged_int(
            records,
            "mtp",
            "live_prefix_replay_state_after_mutation",
            "forward_replay_ordinary_decode_reset_count",
            phase="decode",
            tags={"replay_state": "reset"},
        ),
        "replay_verifier_rebinds": _sum_tagged_int(
            records,
            "mtp",
            "live_prefix_replay_state_after_mutation",
            "forward_replay_all_position_verifier_rebind_count",
            phase="decode",
            tags={"replay_state": "reset"},
        ),
        "replay_other_rebinds": _sum_tagged_int(
            records,
            "mtp",
            "live_prefix_replay_state_after_mutation",
            "forward_replay_other_rebind_count",
            phase="decode",
            tags={"replay_state": "reset"},
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("perfstats", nargs="*", type=Path)
    parser.add_argument("--format", choices=("tsv", "json"), default="tsv")
    args = parser.parse_args()

    if len(args.perfstats) <= 1:
        summary = summarize(args.perfstats[0] if args.perfstats else None)
        if args.format == "json":
            print(json.dumps(summary, sort_keys=True))
        else:
            print("\t".join(str(summary[field]) for field in FIELDS))
        return 0

    summaries = [
        {"path": str(path), **summarize(path)}
        for path in args.perfstats
    ]
    if args.format == "json":
        print(json.dumps(summaries, sort_keys=True))
    else:
        print("\t".join(("path", *FIELDS)))
        for summary in summaries:
            print("\t".join(str(summary[field]) for field in ("path", *FIELDS)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
