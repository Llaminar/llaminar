#!/usr/bin/env python3
"""Regression tests for MoE rebalance trace corpus analysis."""

from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "analyze_moe_rebalance_trace_corpus.py"

spec = importlib.util.spec_from_file_location("moe_trace_analyzer", SCRIPT)
assert spec is not None
moe_trace_analyzer = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(moe_trace_analyzer)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class MoERebalanceTraceCorpusAnalyzerTest(unittest.TestCase):
    def test_seed_split_uses_canonical_policy_sets(self) -> None:
        self.assertEqual(moe_trace_analyzer.seed_split("cuda", 101, 1024), "train")
        self.assertEqual(moe_trace_analyzer.seed_split("cuda", 404, 2048), "train")
        self.assertEqual(
            moe_trace_analyzer.seed_split("rocm", 505, 1024),
            "validation",
        )
        self.assertEqual(moe_trace_analyzer.seed_split("rocm", 606, 2048), "holdout")

    def test_build_tables_filters_noop_commands_and_summarizes_router_benefit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp)
            run_dir = corpus / "cuda" / "twocard" / "dynamic_hot10" / "n_1024" / "seed_101" / "rep_1"
            benchmark = run_dir / "benchmark.json"
            trace = run_dir / "rebalance_trace.jsonl"
            write_json(
                benchmark,
                {
                    "success": True,
                    "throughput_tokens_per_sec": {
                        "prefill": 3000.0,
                        "decode": 120.0,
                        "overall": 200.0,
                    },
                    "timing_ms": {
                        "prefill": 200.0,
                        "decode": 8500.0,
                        "total": 8700.0,
                    },
                    "decode_latency_ms": {
                        "mean": 8.5,
                        "p50": 8.0,
                        "p90": 9.0,
                        "samples": 1024,
                    },
                },
            )
            trace.parent.mkdir(parents=True, exist_ok=True)
            trace.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "tags": {
                                    "maintenance_graph_kind": "plan",
                                    "decode_tokens_seen": "322",
                                    "window": "64",
                                },
                                "status": {
                                    "status_code": 0,
                                    "last_epoch": 7,
                                    "planned_arrivals": 3,
                                    "selected_replicas": 2,
                                    "changed_layers": 1,
                                    "candidate_arrivals_considered": 5,
                                    "candidate_arrivals_below_floor": 2,
                                    "candidate_arrivals_pruned_by_count_bound": 11,
                                    "llep_assignment_span_count": 17,
                                    "llep_weight_transfer_count": 2,
                                    "llep_native_rows": 90,
                                    "llep_spilled_rows": 30,
                                    "llep_standard_ep_selected": 0,
                                    "llep_skipped_balanced": 0,
                                    "llep_skipped_insufficient_spread_improvement": 1,
                                    "llep_skipped_insufficient_foreign_rows": 0,
                                    "llep_min_chunk_skips": 4,
                                    "llep_forced_spills": 1,
                                    "llep_required_spread_improvement": 22,
                                    "llep_required_foreign_rows": 10,
                                    "candidate_load_spread_improvement_total": 123,
                                    "candidate_load_spread_improvement_max": 77,
                                    "accepted_load_spread_improvement_total": 99,
                                    "accepted_load_spread_improvement_max": 66,
                                    "skipped_wave_cost_floor": 0,
                                    "skipped_low_router_benefit": 0,
                                    "skipped_post_load_spread_ceiling": 1,
                                    "payload_bucket_requested_slots": 1,
                                    "payload_bucket_slots": 1,
                                    "payload_edge_mask": 256,
                                    "router_hot_cache_eligible_dispatches": 13,
                                    "router_hot_cache_used_dispatches": 8,
                                    "router_hot_cache_improved_dispatches": 8,
                                    "router_hot_cache_default_load_spread_total": 40,
                                    "router_hot_cache_actual_load_spread_total": 9,
                                    "router_hot_cache_load_spread_improvement_total": 31,
                                    "router_hot_cache_active_dispatches": 21,
                                    "router_hot_cache_miss_dispatches": 3,
                                    "router_hot_cache_selected_expert_slots": 42,
                                    "router_hot_cache_replicated_selected_expert_slots": 10,
                                    "pre_policy_load_total": 101,
                                    "pre_policy_load_min": 12,
                                    "pre_policy_load_max": 30,
                                    "post_policy_load_total": 101,
                                    "post_policy_load_min": 18,
                                    "post_policy_load_max": 24,
                                    "post_wave_load_total": 101,
                                    "post_wave_load_spread": 6,
                                },
                                "apply_status": {
                                    "applied_arrivals": 0,
                                    "post_apply_multi_resident_experts": 2,
                                },
                                "commands": [
                                    {"op": 0},
                                    {"op": 1},
                                    {"op": 2},
                                    {"op": 99},
                                ],
                                "noop_command_entries": 4,
                            }
                        ),
                        json.dumps(
                            {
                                "tags": {
                                    "maintenance_graph_kind": "payload",
                                    "decode_tokens_seen": "323",
                                    "window": "64",
                                },
                                "status": {
                                    "status_code": 0,
                                    "last_epoch": 7,
                                    "planned_arrivals": 3,
                                    "selected_replicas": 2,
                                    "changed_layers": 1,
                                    "candidate_arrivals_considered": 5,
                                    "candidate_arrivals_below_floor": 2,
                                    "candidate_arrivals_pruned_by_count_bound": 11,
                                    "llep_assignment_span_count": 999,
                                    "llep_weight_transfer_count": 999,
                                    "llep_native_rows": 999,
                                    "llep_spilled_rows": 999,
                                    "llep_standard_ep_selected": 999,
                                    "llep_skipped_balanced": 999,
                                    "llep_skipped_insufficient_spread_improvement": 999,
                                    "llep_skipped_insufficient_foreign_rows": 999,
                                    "llep_min_chunk_skips": 999,
                                    "llep_forced_spills": 999,
                                    "llep_required_spread_improvement": 999,
                                    "llep_required_foreign_rows": 999,
                                    "candidate_load_spread_improvement_total": 123,
                                    "candidate_load_spread_improvement_max": 77,
                                    "accepted_load_spread_improvement_total": 99,
                                    "accepted_load_spread_improvement_max": 66,
                                    "skipped_wave_cost_floor": 0,
                                    "skipped_low_router_benefit": 0,
                                    "skipped_post_load_spread_ceiling": 1,
                                    "payload_bucket_requested_slots": 1,
                                    "payload_bucket_slots": 1,
                                    "payload_edge_mask": 256,
                                    "router_hot_cache_eligible_dispatches": 13,
                                    "router_hot_cache_used_dispatches": 8,
                                    "router_hot_cache_improved_dispatches": 8,
                                    "router_hot_cache_default_load_spread_total": 40,
                                    "router_hot_cache_actual_load_spread_total": 9,
                                    "router_hot_cache_load_spread_improvement_total": 31,
                                    "router_hot_cache_active_dispatches": 21,
                                    "router_hot_cache_miss_dispatches": 3,
                                    "router_hot_cache_selected_expert_slots": 42,
                                    "router_hot_cache_replicated_selected_expert_slots": 10,
                                    "pre_policy_load_total": 101,
                                    "pre_policy_load_min": 12,
                                    "pre_policy_load_max": 30,
                                    "post_policy_load_total": 101,
                                    "post_policy_load_min": 18,
                                    "post_policy_load_max": 24,
                                    "post_wave_load_total": 101,
                                    "post_wave_load_spread": 6,
                                },
                                "apply_status": {
                                    "applied_arrivals": 1,
                                    "post_apply_multi_resident_experts": 9,
                                },
                                "commands": [
                                    {"op": 1},
                                    {"op": 2},
                                ],
                                "noop_command_entries": 4,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            summary = corpus / "summary.tsv"
            with summary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    delimiter="\t",
                    fieldnames=[
                        "backend",
                        "placement",
                        "case",
                        "n_predict",
                        "seed",
                        "rep",
                        "exit_code",
                        "benchmark_json",
                        "perf_json",
                        "perf_csv",
                        "rebalance_trace_jsonl",
                        "log",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "backend": "cuda",
                        "placement": "twocard",
                        "case": "dynamic_hot10",
                        "n_predict": "1024",
                        "seed": "101",
                        "rep": "1",
                        "exit_code": "0",
                        "benchmark_json": str(benchmark),
                        "perf_json": "",
                        "perf_csv": "",
                        "rebalance_trace_jsonl": str(trace),
                        "log": "",
                    }
                )

            run_rows, window_rows = moe_trace_analyzer.build_tables(corpus)

        self.assertEqual(len(run_rows), 1)
        self.assertEqual(len(window_rows), 2)
        self.assertEqual(run_rows[0]["split"], "train")
        self.assertEqual(run_rows[0]["decode_tok_s"], 120.0)
        self.assertEqual(run_rows[0]["decode_mean_ms"], 8.5)
        self.assertEqual(run_rows[0]["trace_rows"], 2)
        self.assertEqual(run_rows[0]["economic_rows"], 1)
        self.assertEqual(run_rows[0]["first_planned_decode_token"], 322)
        self.assertEqual(run_rows[0]["last_planned_decode_token"], 322)
        self.assertEqual(run_rows[0]["first_planned_remaining_tokens"], 702)
        self.assertEqual(run_rows[0]["last_planned_remaining_tokens"], 702)
        self.assertEqual(run_rows[0]["first_applied_decode_token"], 323)
        self.assertEqual(run_rows[0]["last_applied_decode_token"], 323)
        self.assertEqual(run_rows[0]["first_applied_remaining_tokens"], 701)
        self.assertEqual(run_rows[0]["last_applied_remaining_tokens"], 701)
        self.assertEqual(run_rows[0]["command_entries"], 2)
        self.assertEqual(run_rows[0]["transfer_arrivals"], 1)
        self.assertEqual(run_rows[0]["resident_replicas"], 1)
        self.assertEqual(run_rows[0]["noop_command_entries"], 4)
        self.assertEqual(run_rows[0]["router_used"], 8)
        self.assertEqual(run_rows[0]["accepted_improvement"], 99)
        self.assertEqual(run_rows[0]["apply_post_apply_multi_resident_experts"], 2)
        self.assertEqual(run_rows[0]["candidate_below_floor"], 2)
        self.assertEqual(run_rows[0]["llep_assignment_spans"], 17)
        self.assertEqual(run_rows[0]["llep_weight_transfers"], 2)
        self.assertEqual(run_rows[0]["llep_native_rows"], 90)
        self.assertEqual(run_rows[0]["llep_spilled_rows"], 30)
        self.assertEqual(run_rows[0]["llep_spilled_row_ratio"], "0.25")
        self.assertEqual(run_rows[0]["llep_spilled_rows_per_transfer"], "15")
        self.assertEqual(run_rows[0]["llep_standard_ep_selected"], 0)
        self.assertEqual(run_rows[0]["llep_skipped_balanced"], 0)
        self.assertEqual(run_rows[0]["llep_skipped_insufficient_spread_improvement"], 1)
        self.assertEqual(run_rows[0]["llep_skipped_insufficient_foreign_rows"], 0)
        self.assertEqual(run_rows[0]["llep_min_chunk_skips"], 4)
        self.assertEqual(run_rows[0]["llep_forced_spills"], 1)
        self.assertEqual(run_rows[0]["llep_required_spread_improvement"], 22)
        self.assertEqual(run_rows[0]["llep_required_foreign_rows"], 10)
        self.assertEqual(run_rows[0]["candidate_improvement"], 123)
        self.assertEqual(run_rows[0]["candidate_improvement_max"], 77)
        self.assertEqual(run_rows[0]["accepted_improvement_max"], 66)
        self.assertEqual(run_rows[0]["skipped_post_load_spread_ceiling"], 1)
        self.assertEqual(run_rows[0]["router_default_spread"], 40)
        self.assertEqual(run_rows[0]["router_actual_spread"], 9)
        self.assertEqual(run_rows[0]["router_spread_improvement"], 31)
        self.assertEqual(run_rows[0]["router_active"], 21)
        self.assertEqual(run_rows[0]["router_miss"], 3)
        self.assertEqual(run_rows[0]["router_selected_slots"], 42)
        self.assertEqual(run_rows[0]["router_replicated_selected_slots"], 10)
        self.assertEqual(run_rows[0]["pre_policy_load_total"], 101)
        self.assertEqual(run_rows[0]["post_policy_load_total"], 101)
        self.assertEqual(run_rows[0]["pre_policy_load_spread"], 18)
        self.assertEqual(run_rows[0]["post_policy_load_spread"], 6)
        self.assertEqual(run_rows[0]["pre_policy_load_spread_max"], 18)
        self.assertEqual(run_rows[0]["post_policy_load_spread_max"], 6)
        self.assertEqual(run_rows[0]["pre_policy_imbalance_ratio_avg"], "0.178217822")
        self.assertEqual(run_rows[0]["pre_policy_imbalance_ratio_max"], "0.178217822")
        self.assertEqual(run_rows[0]["pre_policy_imbalance_ratio_samples"], 1)
        self.assertEqual(run_rows[0]["post_policy_imbalance_ratio_avg"], "0.0594059406")
        self.assertEqual(run_rows[0]["post_policy_imbalance_ratio_max"], "0.0594059406")
        self.assertEqual(run_rows[0]["post_policy_imbalance_ratio_samples"], 1)
        self.assertEqual(run_rows[0]["post_wave_load_total"], 101)
        self.assertEqual(run_rows[0]["post_wave_load_spread"], 6)
        self.assertEqual(run_rows[0]["post_wave_load_spread_max"], 6)
        self.assertEqual(run_rows[0]["post_wave_imbalance_ratio_avg"], "0.0594059406")
        self.assertEqual(run_rows[0]["post_wave_imbalance_ratio_max"], "0.0594059406")
        self.assertEqual(run_rows[0]["post_wave_imbalance_ratio_samples"], 1)
        self.assertEqual(window_rows[0]["command_entries"], 2)
        self.assertEqual(window_rows[0]["decode_tokens_remaining"], 702)
        self.assertEqual(window_rows[0]["planned_remaining_tokens"], 702)
        self.assertEqual(window_rows[0]["applied_arrivals"], 0)
        self.assertEqual(window_rows[0]["applied_remaining_tokens"], -1)
        self.assertEqual(window_rows[0]["candidate_below_floor"], 2)
        self.assertEqual(window_rows[0]["llep_assignment_spans"], 17)
        self.assertEqual(window_rows[0]["llep_weight_transfers"], 2)
        self.assertEqual(window_rows[0]["llep_native_rows"], 90)
        self.assertEqual(window_rows[0]["llep_spilled_rows"], 30)
        self.assertEqual(window_rows[0]["llep_spilled_row_ratio"], "0.25")
        self.assertEqual(window_rows[0]["llep_spilled_rows_per_transfer"], "15")
        self.assertEqual(window_rows[0]["llep_skipped_insufficient_spread_improvement"], 1)
        self.assertEqual(window_rows[0]["skipped_post_load_spread_ceiling"], 1)
        self.assertEqual(window_rows[0]["router_spread_improvement"], 31)
        self.assertEqual(window_rows[0]["router_active"], 21)
        self.assertEqual(window_rows[0]["router_miss"], 3)
        self.assertEqual(window_rows[0]["router_selected_slots"], 42)
        self.assertEqual(window_rows[0]["router_replicated_selected_slots"], 10)
        self.assertEqual(window_rows[0]["pre_policy_load_spread"], 18)
        self.assertEqual(window_rows[0]["post_policy_load_spread"], 6)
        self.assertEqual(window_rows[0]["post_wave_load_spread"], 6)
        self.assertEqual(window_rows[0]["pre_policy_imbalance_ratio"], "0.178217822")
        self.assertEqual(window_rows[0]["post_policy_imbalance_ratio"], "0.0594059406")
        self.assertEqual(window_rows[0]["post_wave_imbalance_ratio"], "0.0594059406")
        self.assertEqual(window_rows[0]["apply_post_apply_multi_resident_experts"], 2)
        self.assertEqual(window_rows[1]["maintenance_graph_kind"], "payload")
        self.assertEqual(window_rows[1]["command_entries"], 2)
        self.assertEqual(window_rows[1]["apply_post_apply_multi_resident_experts"], 9)
        self.assertEqual(window_rows[1]["planned_remaining_tokens"], -1)
        self.assertEqual(window_rows[1]["llep_spilled_rows"], 999)
        self.assertEqual(window_rows[1]["applied_arrivals"], 1)
        self.assertEqual(window_rows[1]["applied_remaining_tokens"], 701)


if __name__ == "__main__":
    unittest.main()
