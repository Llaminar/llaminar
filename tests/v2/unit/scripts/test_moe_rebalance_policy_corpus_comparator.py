#!/usr/bin/env python3
"""Regression tests for MoE rebalance policy corpus comparison."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "compare_moe_rebalance_policy_corpora.py"

spec = importlib.util.spec_from_file_location("moe_policy_comparator", SCRIPT)
assert spec is not None
moe_policy_comparator = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(moe_policy_comparator)


class MoERebalancePolicyCorpusComparatorTest(unittest.TestCase):
    def test_build_delta_rows_joins_by_backend_length_seed_rep_and_case(self) -> None:
        hot_rows = [
            {
                "backend": "cuda",
                "n_predict": "1024",
                "seed": "101",
                "split": "train",
                "rep": "1",
                "case": "dynamic_hot10",
                "decode_tok_s": "126.0",
                "decode_mean_ms": "7.9",
                "decode_p90_ms": "8.3",
                "transfer_arrivals": "24",
                "resident_replicas": "8",
                "router_used": "388",
                "accepted_improvement": "1600",
            }
        ]
        baseline_rows = [
            {
                "backend": "cuda",
                "n_predict": "1024",
                "seed": "101",
                "split": "train",
                "rep": "1",
                "case": "static",
                "decode_tok_s": "120.0",
                "decode_mean_ms": "8.4",
                "decode_p90_ms": "8.9",
            },
            {
                "backend": "cuda",
                "n_predict": "1024",
                "seed": "101",
                "split": "train",
                "rep": "1",
                "case": "dynamic",
                "decode_tok_s": "122.5",
                "decode_mean_ms": "8.1",
                "decode_p90_ms": "8.5",
            },
        ]

        rows = moe_policy_comparator.build_delta_rows(hot_rows, baseline_rows)
        summary = moe_policy_comparator.summarize(rows)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["hot_minus_static_decode_tok_s"], "6")
        self.assertEqual(rows[0]["hot_minus_dynamic_decode_tok_s"], "3.5")
        self.assertEqual(rows[0]["dynamic_minus_static_decode_tok_s"], "2.5")
        self.assertEqual(rows[0]["hot_minus_static_decode_mean_ms"], "-0.5")
        self.assertEqual(rows[0]["hot_minus_static_decode_p90_ms"], "-0.6")
        self.assertEqual(rows[0]["hot_transfer_arrivals"], "24")
        self.assertEqual(summary[0]["hot_rows"], "1")
        self.assertEqual(summary[0]["static_pairs"], "1")
        self.assertEqual(summary[0]["dynamic_pairs"], "1")
        self.assertEqual(summary[0]["hot_minus_static_decode_tok_s_mean"], "6")

    def test_feature_label_rows_join_clean_labels_to_traced_features(self) -> None:
        hot_rows = [
            {
                "backend": "rocm",
                "n_predict": "2048",
                "seed": "303",
                "split": "train",
                "rep": "1",
                "case": "dynamic_hot10",
                "decode_tok_s": "66.5",
                "decode_mean_ms": "15.0",
            },
            {
                "backend": "rocm",
                "n_predict": "2048",
                "seed": "404",
                "split": "train",
                "rep": "1",
                "case": "dynamic_hot10",
                "decode_tok_s": "63.0",
                "decode_mean_ms": "16.0",
            }
        ]
        baseline_rows = [
            {
                "backend": "rocm",
                "n_predict": "2048",
                "seed": "303",
                "split": "train",
                "rep": "1",
                "case": "static",
                "decode_tok_s": "60.0",
                "decode_mean_ms": "16.5",
            },
            {
                "backend": "rocm",
                "n_predict": "2048",
                "seed": "303",
                "split": "train",
                "rep": "1",
                "case": "dynamic",
                "decode_tok_s": "64.0",
                "decode_mean_ms": "15.6",
            },
            {
                "backend": "rocm",
                "n_predict": "2048",
                "seed": "404",
                "split": "train",
                "rep": "1",
                "case": "static",
                "decode_tok_s": "62.0",
                "decode_mean_ms": "16.4",
            },
            {
                "backend": "rocm",
                "n_predict": "2048",
                "seed": "404",
                "split": "train",
                "rep": "1",
                "case": "dynamic",
                "decode_tok_s": "64.0",
                "decode_mean_ms": "15.6",
            },
        ]
        feature_rows = [
            {
                "backend": "rocm",
                "n_predict": "2048",
                "seed": "303",
                "split": "train",
                "rep": "1",
                "case": "dynamic_hot10",
                "decode_tok_s": "51.0",
                "planned_arrivals": "32",
                "changed_layers": "4",
                "apply_post_apply_multi_resident_experts": "12",
                "transfer_arrivals": "30",
                "resident_replicas": "10",
                "command_entries": "40",
                "accepted_improvement": "2400",
                "first_planned_decode_token": "512",
                "last_planned_decode_token": "1536",
                "first_planned_remaining_tokens": "1536",
                "last_planned_remaining_tokens": "512",
                "first_applied_decode_token": "513",
                "last_applied_decode_token": "1540",
                "first_applied_remaining_tokens": "1535",
                "last_applied_remaining_tokens": "508",
                "skipped_wave_cost_floor": "0",
                "skipped_low_router_benefit": "0",
                "skipped_post_load_spread_ceiling": "1",
                "router_active": "240",
                "router_miss": "40",
                "router_eligible": "200",
                "router_used": "150",
                "router_improved": "120",
                "router_selected_slots": "400",
                "router_replicated_selected_slots": "150",
                "router_default_spread": "1000",
                "router_actual_spread": "700",
                "router_spread_improvement": "300",
                "selected_replicas": "20",
                "candidate_below_floor": "3",
                "llep_assignment_spans": "50",
                "llep_weight_transfers": "5",
                "llep_native_rows": "900",
                "llep_spilled_rows": "100",
                "llep_spilled_row_ratio": "0.1",
                "llep_spilled_rows_per_transfer": "20",
                "llep_standard_ep_selected": "0",
                "llep_skipped_balanced": "0",
                "llep_skipped_insufficient_spread_improvement": "1",
                "llep_skipped_insufficient_foreign_rows": "0",
                "llep_min_chunk_skips": "2",
                "llep_forced_spills": "1",
                "llep_required_spread_improvement": "25",
                "llep_required_foreign_rows": "80",
                "candidate_improvement": "2600",
                "candidate_improvement_max": "500",
                "accepted_improvement_max": "400",
                "pre_policy_load_total": "10000",
                "post_policy_load_total": "10000",
                "pre_policy_load_spread": "100",
                "post_policy_load_spread": "40",
                "pre_policy_load_spread_max": "14",
                "post_policy_load_spread_max": "8",
                "pre_policy_imbalance_ratio_avg": "0.01",
                "pre_policy_imbalance_ratio_max": "0.014",
                "pre_policy_imbalance_ratio_samples": "2",
                "post_policy_imbalance_ratio_avg": "0.004",
                "post_policy_imbalance_ratio_max": "0.008",
                "post_policy_imbalance_ratio_samples": "2",
                "post_wave_load_total": "10000",
                "post_wave_load_spread": "40",
                "post_wave_load_spread_max": "8",
                "post_wave_imbalance_ratio_avg": "0.006",
                "post_wave_imbalance_ratio_max": "0.009",
                "post_wave_imbalance_ratio_samples": "2",
                "payload_bucket_slots_max": "2",
                "payload_bucket_requested_slots_max": "1",
            }
        ]

        rows = moe_policy_comparator.build_feature_label_rows(
            hot_rows,
            baseline_rows,
            feature_rows,
        )
        thresholds = moe_policy_comparator.threshold_summary(rows * 3)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["hot_minus_static_decode_tok_s"], "6.5")
        self.assertEqual(rows[0]["hot_minus_dynamic_decode_tok_s"], "2.5")
        self.assertEqual(rows[0]["dynamic_minus_static_decode_tok_s"], "4")
        self.assertEqual(rows[0]["planned_arrivals"], "32")
        self.assertEqual(rows[0]["changed_layers"], "4")
        self.assertEqual(rows[0]["apply_post_apply_multi_resident_experts"], "12")
        self.assertEqual(rows[0]["transfer_arrivals"], "30")
        self.assertEqual(rows[0]["resident_replicas"], "10")
        self.assertEqual(rows[0]["resident_fraction"], "0.25")
        self.assertEqual(rows[0]["apply_multi_resident_per_transfer"], "0.4")
        self.assertEqual(rows[0]["accepted_improvement_per_transfer"], "80")
        self.assertEqual(rows[0]["accepted_improvement_per_command"], "60")
        self.assertEqual(rows[0]["first_planned_decode_token"], "512")
        self.assertEqual(rows[0]["first_planned_remaining_fraction"], "0.75")
        self.assertEqual(rows[0]["last_planned_remaining_fraction"], "0.25")
        self.assertEqual(rows[0]["first_applied_decode_token"], "513")
        self.assertEqual(rows[0]["first_applied_remaining_fraction"], "0.749511719")
        self.assertEqual(rows[0]["last_applied_remaining_fraction"], "0.248046875")
        self.assertEqual(rows[0]["first_plan_to_apply_lag_tokens"], "1")
        self.assertEqual(rows[0]["last_plan_to_apply_lag_tokens"], "4")
        self.assertEqual(rows[0]["router_active"], "240")
        self.assertEqual(rows[0]["router_miss"], "40")
        self.assertEqual(rows[0]["router_selected_slots"], "400")
        self.assertEqual(rows[0]["router_replicated_selected_slots"], "150")
        self.assertEqual(rows[0]["router_miss_rate"], "0.166666667")
        self.assertEqual(rows[0]["router_eligible_rate"], "0.833333333")
        self.assertEqual(rows[0]["router_use_rate_active"], "0.625")
        self.assertEqual(rows[0]["router_replicated_selected_slot_ratio"], "0.375")
        self.assertEqual(
            rows[0]["router_selected_slots_per_active_dispatch"],
            "1.66666667",
        )
        self.assertEqual(
            rows[0]["router_replicated_slots_per_active_dispatch"],
            "0.625",
        )
        self.assertEqual(rows[0]["router_spread_improvement_per_used_dispatch"], "2")
        self.assertEqual(
            rows[0]["router_spread_improvement_per_eligible_dispatch"],
            "1.5",
        )
        self.assertEqual(rows[0]["router_spread_improvement_per_selected_slot"], "0.75")
        self.assertEqual(
            rows[0]["router_spread_improvement_per_replicated_selected_slot"],
            "2",
        )
        self.assertEqual(rows[0]["post_vs_pre_load_spread_ratio"], "0.4")
        self.assertEqual(rows[0]["pre_policy_load_spread_fraction"], "0.01")
        self.assertEqual(rows[0]["post_policy_load_spread_fraction"], "0.004")
        self.assertEqual(rows[0]["pre_policy_imbalance_ratio_avg"], "0.01")
        self.assertEqual(rows[0]["post_policy_imbalance_ratio_avg"], "0.004")
        self.assertEqual(rows[0]["post_wave_imbalance_ratio_avg"], "0.006")
        self.assertEqual(rows[0]["post_vs_pre_imbalance_ratio"], "0.4")
        self.assertEqual(rows[0]["post_wave_vs_pre_imbalance_ratio"], "0.6")
        self.assertEqual(rows[0]["imbalance_ratio_delta"], "-0.006")
        self.assertEqual(
            rows[0]["post_policy_load_spread_per_transfer"],
            "1.33333333",
        )
        self.assertEqual(rows[0]["post_wave_load_spread_fraction"], "0.004")
        self.assertEqual(
            rows[0]["post_wave_load_spread_per_transfer"],
            "1.33333333",
        )
        self.assertEqual(rows[0]["skipped_post_load_spread_ceiling"], "1")
        self.assertEqual(rows[0]["llep_assignment_spans"], "50")
        self.assertEqual(rows[0]["llep_weight_transfers"], "5")
        self.assertEqual(rows[0]["llep_spilled_rows"], "100")
        self.assertEqual(rows[0]["llep_spilled_row_ratio"], "0.1")
        self.assertEqual(rows[0]["llep_spilled_rows_per_transfer"], "20")
        self.assertEqual(rows[0]["llep_spans_per_weight_transfer"], "10")
        self.assertEqual(rows[0]["llep_spilled_rows_per_span"], "2")
        self.assertEqual(rows[0]["llep_spilled_rows_per_planned_arrival"], "3.125")
        self.assertEqual(rows[0]["load_spread_delta"], "-60")
        self.assertEqual(rows[0]["router_use_rate"], "0.75")
        self.assertEqual(rows[0]["router_improve_rate"], "0.8")
        self.assertTrue(
            any(
                row["label"] == "hot_minus_dynamic_decode_tok_s"
                and row["feature"] == "resident_fraction"
                for row in thresholds
            )
        )
        self.assertEqual(rows[1]["seed"], "404")
        self.assertEqual(rows[1]["transfer_arrivals"], "")
        self.assertEqual(rows[1]["first_applied_remaining_fraction"], "")


if __name__ == "__main__":
    unittest.main()
