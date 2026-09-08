#!/usr/bin/env python3
"""Regression tests for the LLEP trace corpus evaluator."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "evaluate_llep_trace_corpus.py"

spec = importlib.util.spec_from_file_location("moe_llep_evaluator", SCRIPT)
assert spec is not None
moe_llep_evaluator = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(moe_llep_evaluator)


def evaluator_args(**overrides: object) -> argparse.Namespace:
    values = {
        "min_chunk_tokens": 0,
        "alpha_numerator": 1,
        "alpha_denominator": 1,
        "lambda_numerator": 13,
        "lambda_denominator": 10,
        "min_spread_improvement": 0,
        "min_spread_improvement_per_critical_path_slot": 0,
        "sweep_min_spread_improvement_per_critical_path_slot": "",
        "disable_balanced_skip": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def write_summary(corpus: Path, trace: Path, seed: str = "101") -> None:
    with (corpus / "summary.tsv").open("w", encoding="utf-8", newline="") as handle:
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
                "seed": seed,
                "rep": "1",
                "exit_code": "0",
                "benchmark_json": "",
                "perf_json": "",
                "perf_csv": "",
                "rebalance_trace_jsonl": str(trace),
                "log": "",
            }
        )


class MoELLEPTraceCorpusEvaluatorTest(unittest.TestCase):
    def test_reciprocal_wave_prices_one_critical_path_slot(self) -> None:
        plan = moe_llep_evaluator.plan_llep(
            [5, 15, 20, 20],
            [0, 0, 1, 1],
            2,
            min_chunk_tokens=0,
            alpha_numerator=3,
            alpha_denominator=4,
            lambda_numerator=13,
            lambda_denominator=10,
            min_spread_improvement=0,
            min_spread_improvement_per_critical_path_slot=5,
            enable_balanced_skip=False,
        )

        status = plan["status"]
        self.assertEqual(status["weight_transfer_count"], 2)
        self.assertEqual(status["critical_path_transfer_slots"], 1)
        self.assertEqual(status["assigned_load_spread_improvement"], 6)
        self.assertEqual(status["required_spread_improvement"], 5)
        self.assertEqual(status["skipped_insufficient_spread_improvement"], 0)
        self.assertEqual(status["standard_ep_selected"], 0)

    def test_evaluates_llep_assignment_from_gathered_histogram_and_owner_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp)
            trace = corpus / "cuda" / "twocard" / "dynamic_hot10" / "trace.jsonl"
            trace.parent.mkdir(parents=True, exist_ok=True)
            trace_row = {
                "config": {
                    "participant_count": 2,
                    "num_experts": 4,
                },
                "tags": {
                    "maintenance_graph_kind": "plan",
                    "decode_tokens_seen": "256",
                },
                "status": {
                    "planned_arrivals": 1,
                    "payload_bucket_requested_slots": 1,
                },
                "gathered_histogram": {
                    "layout": "participant,wave_layer,expert",
                    "participant_count": 2,
                    "wave_layer_count": 1,
                    "num_experts": 4,
                    "global_expert_loads": [
                        {"wave_layer": 0, "expert": 0, "count": 80},
                        {"wave_layer": 0, "expert": 1, "count": 20},
                    ],
                },
                "expert_owner_participants": [
                    {"wave_layer": 0, "owners": [0, 0, 1, 1]},
                ],
            }
            duplicate_trace_row = dict(trace_row)
            duplicate_trace_row["device"] = "CUDA:1"
            trace.write_text(
                json.dumps(trace_row)
                + "\n"
                + json.dumps(duplicate_trace_row)
                + "\n",
                encoding="utf-8",
            )
            write_summary(corpus, trace)

            run_rows, window_rows, split_rows = moe_llep_evaluator.build_tables(
                corpus,
                evaluator_args(),
            )

        self.assertEqual(len(window_rows), 1)
        self.assertEqual(window_rows[0]["standard_load_spread"], 100)
        self.assertEqual(window_rows[0]["llep_load_spread"], 0)
        self.assertEqual(window_rows[0]["spread_improvement"], 100)
        self.assertEqual(
            window_rows[0]["spread_improvement_per_critical_path_slot"], "100"
        )
        self.assertEqual(window_rows[0]["required_spread_improvement"], 0)
        self.assertEqual(window_rows[0]["capacity_per_participant"], 50)
        self.assertEqual(window_rows[0]["native_rows"], 50)
        self.assertEqual(window_rows[0]["spilled_rows"], 50)
        self.assertEqual(window_rows[0]["weight_transfer_count"], 1)
        self.assertEqual(window_rows[0]["skipped_insufficient_spread_improvement"], 0)
        self.assertEqual(window_rows[0]["current_policy_planned_arrivals"], 1)

        self.assertEqual(run_rows[0]["eligible_wave_layers"], 1)
        self.assertEqual(run_rows[0]["duplicate_trace_rows"], 1)
        self.assertEqual(run_rows[0]["spread_improvement"], 100)
        self.assertEqual(run_rows[0]["spread_improvement_fraction"], "1")
        self.assertEqual(
            run_rows[0]["spread_improvement_per_critical_path_slot"], "100"
        )
        self.assertEqual(run_rows[0]["required_spread_improvement"], 0)
        self.assertEqual(run_rows[0]["skipped_insufficient_spread_improvement"], 0)
        self.assertEqual(split_rows[0]["eligible_wave_layers"], 1)
        self.assertEqual(split_rows[0]["duplicate_trace_rows"], 1)
        self.assertEqual(split_rows[0]["spread_improvement_fraction"], "1")

    def test_roi_gate_can_reject_low_value_llep_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp)
            trace = corpus / "cuda" / "twocard" / "dynamic_hot10" / "trace.jsonl"
            trace.parent.mkdir(parents=True, exist_ok=True)
            trace.write_text(
                json.dumps(
                    {
                        "config": {
                            "participant_count": 2,
                            "num_experts": 4,
                        },
                        "tags": {
                            "maintenance_graph_kind": "plan",
                            "decode_tokens_seen": "512",
                        },
                        "status": {},
                        "gathered_histogram": {
                            "layout": "participant,wave_layer,expert",
                            "participant_count": 2,
                            "wave_layer_count": 1,
                            "num_experts": 4,
                            "global_expert_loads": [
                                {"wave_layer": 0, "expert": 0, "count": 80},
                                {"wave_layer": 0, "expert": 1, "count": 20},
                            ],
                        },
                        "expert_owner_participants": [
                            {"wave_layer": 0, "owners": [0, 0, 1, 1]},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            write_summary(corpus, trace)

            run_rows, window_rows, split_rows = moe_llep_evaluator.build_tables(
                corpus,
                evaluator_args(
                    min_spread_improvement_per_critical_path_slot=128
                ),
            )

        self.assertEqual(len(window_rows), 1)
        self.assertEqual(window_rows[0]["standard_load_spread"], 100)
        self.assertEqual(window_rows[0]["llep_load_spread"], 100)
        self.assertEqual(window_rows[0]["spread_improvement"], 0)
        self.assertEqual(window_rows[0]["required_spread_improvement"], 128)
        self.assertEqual(window_rows[0]["weight_transfer_count"], 0)
        self.assertEqual(window_rows[0]["spilled_rows"], 0)
        self.assertEqual(window_rows[0]["native_rows"], 0)
        self.assertEqual(window_rows[0]["skipped_insufficient_spread_improvement"], 1)
        self.assertEqual(window_rows[0]["standard_ep_selected"], 1)

        self.assertEqual(run_rows[0]["spread_improvement"], 0)
        self.assertEqual(run_rows[0]["required_spread_improvement"], 128)
        self.assertEqual(run_rows[0]["skipped_insufficient_spread_improvement"], 1)
        self.assertEqual(split_rows[0]["skipped_insufficient_spread_improvement"], 1)

    def test_roi_sweep_reports_threshold_tradeoffs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp)
            trace = corpus / "cuda" / "twocard" / "dynamic_hot10" / "trace.jsonl"
            trace.parent.mkdir(parents=True, exist_ok=True)
            trace.write_text(
                json.dumps(
                    {
                        "config": {
                            "participant_count": 2,
                            "num_experts": 4,
                        },
                        "tags": {"maintenance_graph_kind": "plan"},
                        "status": {},
                        "gathered_histogram": {
                            "layout": "participant,wave_layer,expert",
                            "participant_count": 2,
                            "wave_layer_count": 1,
                            "num_experts": 4,
                            "global_expert_loads": [
                                {"wave_layer": 0, "expert": 0, "count": 80},
                                {"wave_layer": 0, "expert": 1, "count": 20},
                            ],
                        },
                        "expert_owner_participants": [
                            {"wave_layer": 0, "owners": [0, 0, 1, 1]},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            write_summary(corpus, trace)
            args = evaluator_args()

            rows = moe_llep_evaluator.build_roi_sweep(corpus, args, [0, 128])

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            rows[0]["min_spread_improvement_per_critical_path_slot"], 0
        )
        self.assertEqual(rows[0]["spread_improvement"], 100)
        self.assertEqual(rows[0]["weight_transfer_count"], 1)
        self.assertEqual(
            rows[0]["spread_improvement_per_critical_path_slot"], "100"
        )
        self.assertEqual(
            rows[1]["min_spread_improvement_per_critical_path_slot"], 128
        )
        self.assertEqual(rows[1]["spread_improvement"], 0)
        self.assertEqual(rows[1]["weight_transfer_count"], 0)
        self.assertEqual(rows[1]["skipped_insufficient_spread_improvement"], 1)

    def test_reports_missing_histogram_inputs_without_fabricating_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp)
            trace = corpus / "cuda" / "twocard" / "dynamic_hot10" / "trace.jsonl"
            trace.parent.mkdir(parents=True, exist_ok=True)
            trace.write_text(
                json.dumps(
                    {
                        "config": {
                            "participant_count": 2,
                            "num_experts": 4,
                        },
                        "tags": {"maintenance_graph_kind": "plan"},
                        "status": {"planned_arrivals": 1},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            write_summary(corpus, trace, seed="606")

            run_rows, window_rows, split_rows = moe_llep_evaluator.build_tables(
                corpus,
                evaluator_args(),
            )

        self.assertEqual(window_rows, [])
        self.assertEqual(run_rows[0]["missing_trace_rows"], 1)
        self.assertEqual(run_rows[0]["duplicate_trace_rows"], 0)
        self.assertEqual(run_rows[0]["eligible_wave_layers"], 0)
        self.assertEqual(run_rows[0]["spread_improvement"], 0)
        self.assertEqual(run_rows[0]["split"], "holdout")
        self.assertEqual(split_rows[0]["missing_trace_rows"], 1)


if __name__ == "__main__":
    unittest.main()
