#!/usr/bin/env python3
"""Unit checks for scripts/summarize_mtp_perfstats.py."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "summarize_mtp_perfstats.py"
EXPECTED_FIELD_COUNT = len(runpy.run_path(str(SCRIPT))["FIELDS"])


class MTPPerfStatsSummaryTest(unittest.TestCase):
    def test_extracts_verifier_capture_and_replay_health(self) -> None:
        payload = {
            "schema": "llaminar.perf_stats.v1",
            "records": [
                {
                    "domain": "mtp",
                    "name": "decode_step_total",
                    "phase": "decode",
                    "count": 3,
                    "total_ms": 30.5,
                },
                {
                    "domain": "mtp",
                    "name": "verifier_forward",
                    "phase": "decode",
                    "count": 3,
                    "total_ms": 18.25,
                },
                {
                    "domain": "mtp",
                    "name": "request_batch_stochastic_verifier_forward",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 5.5,
                },
                {
                    "domain": "mtp",
                    "name": "grouped_outcome_stochastic_verifier_forward",
                    "phase": "decode",
                    "count": 2,
                    "total_ms": 7.25,
                },
                {
                    "domain": "mtp",
                    "name": "stochastic_device_physical_verify_rows",
                    "phase": "decode",
                    "count": 2,
                    "value": 6,
                },
                {
                    "domain": "mtp",
                    "name": "stochastic_device_physical_verify_rows",
                    "phase": "decode",
                    "count": 1,
                    "value": 3,
                },
                {
                    "domain": "mtp",
                    "name": "stochastic_device_semantic_verify_rows",
                    "phase": "decode",
                    "count": 3,
                    "value": 7,
                },
                {
                    "domain": "mtp",
                    "name": "stochastic_device_post_reject_rows",
                    "phase": "decode",
                    "count": 3,
                    "value": 2,
                },
                {
                    "domain": "mtp",
                    "name": "stochastic_seeded_device_threshold_rows",
                    "phase": "decode",
                    "count": 2,
                    "value": 5,
                },
                {
                    "domain": "mtp",
                    "name": "verifier_economy_capability",
                    "phase": "decode",
                    "device": "cuda:0",
                    "count": 1,
                    "value": 4,
                    "tags": {
                        "lane": "dense",
                        "perf_gate_status": "correct_serial_fallback_not_economical",
                        "max_rows": "4",
                        "serial_decode_equivalent_fallback": "true",
                        "grouped_decode_equivalent": "false",
                        "row_indexed_lm_head": "false",
                        "device_resident_input": "false",
                        "device_resident_outcome": "false",
                        "device_resident_publication": "false",
                        "host_bridge_free_hot_path": "false",
                        "graph_capturable": "false",
                        "greedy": "true",
                        "stochastic": "true",
                    },
                },
                {
                    "domain": "mtp",
                    "name": "verifier_economy_capability",
                    "phase": "decode",
                    "device": "cuda:0",
                    "count": 1,
                    "value": 3,
                    "tags": {
                        "lane": "moe",
                        "perf_gate_status": "grouped_promoted",
                        "max_rows": "3",
                        "serial_decode_equivalent_fallback": "false",
                        "grouped_decode_equivalent": "true",
                        "row_indexed_lm_head": "true",
                        "device_resident_input": "true",
                        "device_resident_outcome": "true",
                        "device_resident_publication": "true",
                        "host_bridge_free_hot_path": "true",
                        "graph_capturable": "true",
                        "greedy": "true",
                        "stochastic": "true",
                    },
                },
                {
                    "domain": "mtp",
                    "name": "condition_forward",
                    "phase": "decode",
                    "count": 2,
                    "total_ms": 9.75,
                },
                {
                    "domain": "mtp",
                    "name": "condition_forward_skipped_ready_logits",
                    "phase": "decode",
                    "count": 5,
                },
                {
                    "domain": "mtp",
                    "name": "condition_forward_skipped_pending_condition",
                    "phase": "decode",
                    "count": 6,
                },
                {
                    "domain": "mtp",
                    "name": "pending_condition_verifier_rows",
                    "phase": "decode",
                    "count": 2,
                    "value": 4,
                },
                {
                    "domain": "mtp",
                    "name": "first_token_pending_condition_rows",
                    "phase": "decode",
                    "count": 3,
                    "value": 3,
                },
                {
                    "domain": "mtp",
                    "name": "all_position_correction_forward",
                    "phase": "decode",
                    "count": 2,
                    "total_ms": 7.5,
                },
                {
                    "domain": "mtp",
                    "name": "all_position_deferred_correction_condition_tokens",
                    "phase": "decode",
                    "count": 3,
                    "total_ms": 0,
                },
                {
                    "domain": "mtp",
                    "name": "all_position_rejection_without_ready_token",
                    "phase": "decode",
                    "count": 4,
                    "total_ms": 0,
                },
                {
                    "domain": "mtp",
                    "name": "all_position_publish_accepted_state",
                    "phase": "decode",
                    "count": 3,
                    "total_ms": 1.25,
                },
                {
                    "domain": "mtp",
                    "name": "grouped_outcome_publish_accepted_state_device_resident",
                    "phase": "decode",
                    "count": 2,
                    "total_ms": 2.0,
                },
                {
                    "domain": "mtp",
                    "name": "sidecar_forward",
                    "phase": "decode",
                    "count": 3,
                    "total_ms": 2.5,
                },
                {
                    "domain": "mtp",
                    "name": "sidecar_depth0_total",
                    "phase": "decode",
                    "count": 3,
                    "total_ms": 3.5,
                },
                {
                    "domain": "mtp",
                    "name": "sidecar_depth0_total",
                    "phase": "prefill",
                    "count": 99,
                    "total_ms": 999.0,
                },
                {
                    "domain": "mtp",
                    "name": "all_position_initial_shifted_commit",
                    "phase": "decode",
                    "count": 3,
                    "total_ms": 4.0,
                },
                {
                    "domain": "mtp",
                    "name": "all_position_initial_shifted_commits",
                    "phase": "decode",
                    "count": 2,
                },
                {
                    "domain": "mtp",
                    "name": "all_position_initial_shifted_reused_sidecar_rows",
                    "phase": "decode",
                    "count": 5,
                },
                {
                    "domain": "mtp",
                    "name": "all_position_shifted_prefix_commit",
                    "phase": "decode",
                    "count": 3,
                    "total_ms": 5.0,
                },
                {
                    "domain": "mtp",
                    "name": "all_position_deferred_correction_shifted_commit",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 6.0,
                },
                {
                    "domain": "mtp",
                    "name": "shifted_row_commit",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 1.0,
                },
                {
                    "domain": "mtp",
                    "name": "shifted_row_device_target_commit",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 2.0,
                },
                {
                    "domain": "mtp",
                    "name": "shifted_row_sequential_commit",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 3.0,
                },
                {
                    "domain": "mtp",
                    "name": "shifted_mtp_kv_ready_events",
                    "phase": "decode",
                    "count": 4,
                },
                {
                    "domain": "mtp",
                    "name": "shifted_mtp_kv_ready_waits",
                    "phase": "decode",
                    "count": 5,
                },
                {
                    "domain": "mtp",
                    "name": "shifted_mtp_kv_stream_syncs_deferred",
                    "phase": "decode",
                    "count": 6,
                },
                {
                    "domain": "mtp",
                    "name": "sample_first_token_stochastic_device",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 0.25,
                },
                {
                    "domain": "mtp",
                    "name": "sample_mtp_token_stochastic_distribution",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 0.75,
                },
                {
                    "domain": "mtp",
                    "name": "sample_stochastic_distribution_enqueue",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 0.5,
                },
                {
                    "domain": "mtp",
                    "name": "all_position_stochastic_device_batch_outcome",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 10.0,
                },
                {
                    "domain": "mtp",
                    "name": "stochastic_distribution_build_gpu",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 8.0,
                },
                {
                    "domain": "mtp",
                    "name": "stochastic_distribution_batch_build_gpu",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 9.0,
                },
                {
                    "domain": "mtp",
                    "name": "stochastic_processed_rows_build_gpu",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 9.5,
                },
                {
                    "domain": "mtp",
                    "name": "all_position_stochastic_device_resident_outcome_enqueue",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 11.0,
                },
                {
                    "domain": "mtp",
                    "name": "grouped_outcome_stochastic_device_resident_outcome_enqueue",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 1.0,
                },
                {
                    "domain": "mtp",
                    "name": "all_position_stochastic_device_outcome_host_bridge",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 12.0,
                },
                {
                    "domain": "mtp",
                    "name": "grouped_outcome_stochastic_device_outcome_host_bridge",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 2.0,
                },
                {
                    "domain": "mtp",
                    "name": "stochastic_request_batch_summary_gpu_reducer",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 13.0,
                },
                {
                    "domain": "mtp",
                    "name": "stochastic_first_sidecar_prelaunch_enqueue",
                    "phase": "decode",
                    "count": 2,
                    "total_ms": 1.5,
                },
                {
                    "domain": "mtp",
                    "name": "stochastic_first_sidecar_prelaunches",
                    "phase": "decode",
                    "count": 3,
                },
                {
                    "domain": "mtp",
                    "name": "stochastic_first_sidecar_prelaunch_reuses",
                    "phase": "decode",
                    "count": 2,
                },
                {
                    "domain": "mtp",
                    "name": "stochastic_prelaunched_first_sidecar_dropped",
                    "phase": "decode",
                    "count": 1,
                },
                {
                    "domain": "mtp",
                    "name": "stochastic_first_sidecar_prelaunch_discarded_complete",
                    "phase": "decode",
                    "count": 2,
                },
                {
                    "domain": "mtp",
                    "name": "stochastic_first_sidecar_resident_ready_inputs",
                    "phase": "decode",
                    "count": 4,
                },
                {
                    "domain": "mtp",
                    "name": "stochastic_first_sidecar_resident_condition_inputs",
                    "phase": "decode",
                    "count": 5,
                },
                {
                    "domain": "mtp",
                    "name": "mtp_sidecar_device_token_inputs",
                    "phase": "decode",
                    "count": 3,
                    "tags": {"source": "host"},
                },
                {
                    "domain": "mtp",
                    "name": "mtp_sidecar_device_token_inputs",
                    "phase": "decode",
                    "count": 7,
                    "tags": {"source": "device"},
                },
                {
                    "domain": "mtp",
                    "name": "all_position_stochastic_device_outcome_catchup_plan",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 0.75,
                },
                {
                    "domain": "mtp",
                    "name": "all_position_transaction_plan_build",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 0.5,
                },
                {
                    "domain": "mtp",
                    "name": "grouped_outcome_transaction_plan_build",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 0.25,
                },
                {
                    "domain": "mtp",
                    "name": "device_resident_publication_host_adoption",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 0.25,
                },
                {
                    "domain": "mtp",
                    "name": "grouped_outcome_device_resident_host_adoption",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 0.125,
                },
                {
                    "domain": "mtp",
                    "name": "transaction_output_commit",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 0.125,
                },
                {
                    "domain": "mtp",
                    "name": "stochastic_batch_summary_d2h_sync",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 20.0,
                },
                {
                    "domain": "mtp",
                    "name": "stochastic_batch_summary_d2h_enqueue",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 0.25,
                },
                {
                    "domain": "mtp",
                    "name": "stochastic_batch_summary_d2h_wait",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 19.75,
                },
                {
                    "domain": "mtp",
                    "name": "stochastic_request_batch_summary_d2h_sync",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 3.0,
                },
                {
                    "domain": "mtp",
                    "name": "stochastic_request_batch_summary_response_ready_wait",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 2.25,
                },
                {
                    "domain": "mtp",
                    "name": "stochastic_request_batch_summary_d2h_enqueue",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 0.5,
                },
                {
                    "domain": "mtp",
                    "name": "stochastic_request_batch_summary_d2h_wait",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 2.5,
                },
                {
                    "domain": "mtp",
                    "name": "stochastic_request_batch_summary_bridge_stream_create",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 1.5,
                },
                {
                    "domain": "mtp",
                    "name": "stochastic_request_batch_summary_bridge_stream_creations",
                    "phase": "decode",
                    "count": 1,
                    "value": 1.0,
                },
                {
                    "domain": "mtp",
                    "name": "stochastic_request_batch_summary_bridge_stream_reuses",
                    "phase": "decode",
                    "count": 1,
                    "value": 4.0,
                },
                {
                    "domain": "stage_gpu",
                    "name": "graph_replay.total",
                    "phase": "decode",
                    "count": 2,
                    "total_ms": 14.0,
                    "tags": {"context": "main_decode"},
                },
                {
                    "domain": "stage_gpu",
                    "name": "graph_replay.total",
                    "phase": "decode",
                    "count": 5,
                    "total_ms": 35.0,
                    "tags": {"context": "main_verifier"},
                },
                {
                    "domain": "stage_gpu",
                    "name": "total",
                    "phase": "decode",
                    "count": 5,
                    "total_ms": 31.0,
                    "tags": {"context": "main_verifier"},
                },
                {
                    "domain": "stage_gpu",
                    "name": "type.MOE_EXPERT_FFN",
                    "phase": "decode",
                    "count": 5,
                    "total_ms": 21.0,
                    "tags": {"context": "main_verifier"},
                },
                {
                    "domain": "stage_gpu",
                    "name": "type.MOE_ROUTER",
                    "phase": "decode",
                    "count": 5,
                    "total_ms": 4.0,
                    "tags": {"context": "main_verifier"},
                },
                {
                    "domain": "stage_gpu",
                    "name": "type.GDN_PROJECTION",
                    "phase": "decode",
                    "count": 5,
                    "total_ms": 3.0,
                    "tags": {"context": "main_verifier"},
                },
                {
                    "domain": "stage_gpu",
                    "name": "type.GDN_RECURRENCE",
                    "phase": "decode",
                    "count": 5,
                    "total_ms": 2.0,
                    "tags": {"context": "main_verifier"},
                },
                {
                    "domain": "stage_gpu",
                    "name": "type.ATTENTION",
                    "phase": "decode",
                    "count": 5,
                    "total_ms": 1.5,
                    "tags": {"context": "main_verifier"},
                },
                {
                    "domain": "stage_gpu",
                    "name": "type.LM_HEAD",
                    "phase": "decode",
                    "count": 5,
                    "total_ms": 1.25,
                    "tags": {"context": "main_verifier"},
                },
                {
                    "domain": "stage_gpu",
                    "name": "type.MOE_EXPERT_FFN",
                    "phase": "prefill",
                    "count": 99,
                    "total_ms": 999.0,
                    "tags": {"context": "main_verifier"},
                },
                {
                    "domain": "stage_gpu",
                    "name": "graph_replay.total",
                    "phase": "decode",
                    "count": 4,
                    "total_ms": 6.0,
                    "tags": {"context": "mtp_decode_sidecar_resident_logical_state"},
                },
                {
                    "domain": "mtp",
                    "name": "sidecar_replay_reset",
                    "phase": "decode",
                    "count": 4,
                    "total_ms": 44.0,
                    "tags": {
                        "reset_scope": "after_spec_publication",
                        "mutation_reason": "accepted_spec_publication",
                    },
                },
                {
                    "domain": "stage_gpu",
                    "name": "graph_replay.total",
                    "phase": "decode",
                    "count": 10,
                    "total_ms": 99.0,
                    "tags": {"context": "mtp_shifted_prefill"},
                },
                {
                    "domain": "mtp",
                    "name": "all_position_verifier_greedy_device_summary",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 30.0,
                },
                {
                    "domain": "mtp",
                    "name": "sample_stochastic_distribution_d2h_sync",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 40.0,
                },
                {
                    "domain": "mtp",
                    "name": "capture_live_prefix_state",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 1.0,
                },
                {
                    "domain": "mtp",
                    "name": "capture_verifier_base_prefix_state",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 2.0,
                },
                {
                    "domain": "mtp",
                    "name": "live_prefix_checkpoint_hybrid_export",
                    "phase": "decode",
                    "count": 1,
                    "total_ms": 3.0,
                },
                {
                    "domain": "mtp",
                    "name": "sidecar_graph_cache_hits",
                    "phase": "decode",
                    "count": 7,
                },
                {
                    "domain": "mtp",
                    "name": "sidecar_graph_cache_hits",
                    "phase": "prefill",
                    "count": 77,
                },
                {
                    "domain": "mtp",
                    "name": "sidecar_graph_cache_misses",
                    "phase": "decode",
                    "count": 8,
                },
                {
                    "domain": "forward_graph",
                    "name": "decode_segmented_phase",
                    "phase": "decode",
                    "count": 6,
                    "tags": {"context": "main_decode", "phase": "warmup"},
                },
                {
                    "domain": "forward_graph",
                    "name": "decode_segmented_phase",
                    "phase": "decode",
                    "count": 2,
                    "tags": {"context": "main_decode", "phase": "capture"},
                },
                {
                    "domain": "forward_graph",
                    "name": "decode_segmented_phase",
                    "phase": "decode",
                    "count": 9,
                    "tags": {"context": "main_decode", "phase": "replay"},
                },
                {
                    "domain": "forward_graph",
                    "name": "decode_segmented_phase",
                    "phase": "decode",
                    "count": 4,
                    "tags": {"context": "main_verifier", "phase": "warmup"},
                },
                {
                    "domain": "forward_graph",
                    "name": "decode_segmented_phase",
                    "phase": "decode",
                    "count": 1,
                    "tags": {"context": "main_verifier", "phase": "capture"},
                },
                {
                    "domain": "forward_graph",
                    "name": "decode_segmented_phase",
                    "phase": "decode",
                    "count": 5,
                    "tags": {"context": "main_verifier", "phase": "replay"},
                },
                {
                    "domain": "forward_graph",
                    "name": "decode_segmented_phase",
                    "phase": "decode",
                    "count": 99,
                    "tags": {"context": "other_decode", "phase": "replay"},
                },
                {
                    "domain": "mtp",
                    "name": "live_prefix_replay_state_after_mutation",
                    "phase": "decode",
                    "count": 2,
                    "tags": {
                        "replay_state": "reset",
                        "forward_replay_reset_cache_count": "0",
                        "forward_replay_stream_rebind_cache_count": "3",
                        "forward_replay_ordinary_decode_reset_count": "0",
                        "forward_replay_all_position_verifier_rebind_count": "1",
                        "forward_replay_other_rebind_count": "2",
                    },
                },
                {
                    "domain": "mtp",
                    "name": "live_prefix_replay_state_after_mutation",
                    "phase": "decode",
                    "count": 6,
                    "tags": {"replay_state": "preserved"},
                },
                {
                    "domain": "mtp",
                    "name": "live_prefix_replay_state_after_mutation",
                    "phase": "decode",
                    "count": 4,
                    "tags": {"sidecar_replay_state": "reset_after_spec_publication"},
                },
                {
                    "domain": "mtp",
                    "name": "live_prefix_replay_state_after_mutation",
                    "phase": "decode",
                    "count": 3,
                    "tags": {"sidecar_replay_state": "preserved_for_spec_publication"},
                },
            ],
        }

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(payload, handle)
            path = Path(handle.name)

        try:
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(path), "--format", "json"],
                check=True,
                text=True,
                capture_output=True,
            )
        finally:
            path.unlink(missing_ok=True)

        summary = json.loads(result.stdout)
        self.assertEqual(summary["decode_step_ms"], 30.5)
        self.assertEqual(summary["verifier_ms"], 31.0)
        self.assertEqual(summary["stochastic_physical_verify_rows"], 9.0)
        self.assertEqual(summary["stochastic_semantic_verify_rows"], 7.0)
        self.assertEqual(summary["stochastic_post_reject_rows"], 2.0)
        self.assertEqual(summary["stochastic_seeded_device_threshold_rows"], 5.0)
        self.assertEqual(
            summary["verifier_economy_dense"],
            "status=correct_serial_fallback_not_economical;rows=4;serial=true;grouped=false;row_lm=false;resident=false/false/false;bridge_free=false;graph=false;greedy=true;stochastic=true",
        )
        self.assertEqual(
            summary["verifier_economy_moe"],
            "status=grouped_promoted;rows=3;serial=false;grouped=true;row_lm=true;resident=true/true/true;bridge_free=true;graph=true;greedy=true;stochastic=true",
        )
        self.assertEqual(summary["condition_ms"], 9.75)
        self.assertEqual(summary["condition_count"], 2)
        self.assertEqual(summary["condition_skipped_ready"], 5)
        self.assertEqual(summary["condition_skipped_pending"], 6)
        self.assertEqual(summary["pending_condition_rows"], 4.0)
        self.assertEqual(summary["first_token_pending_condition_rows"], 3.0)
        self.assertEqual(summary["correction_ms"], 7.5)
        self.assertEqual(summary["correction_count"], 2)
        self.assertEqual(summary["deferred_corrections"], 3)
        self.assertEqual(summary["rejection_no_ready"], 4)
        self.assertEqual(summary["publish_ms"], 3.25)
        self.assertEqual(summary["publish_count"], 5)
        self.assertAlmostEqual(summary["publish_avg_ms"], 3.25 / 5.0)
        self.assertEqual(summary["sidecar_ms"], 2.5)
        self.assertEqual(summary["sidecar_depth0_decode_ms"], 3.5)
        self.assertEqual(summary["shifted_initial_ms"], 4.0)
        self.assertEqual(summary["shifted_initial_commits"], 2)
        self.assertEqual(summary["shifted_initial_reused"], 5)
        self.assertEqual(summary["shifted_prefix_ms"], 5.0)
        self.assertEqual(summary["shifted_deferred_ms"], 6.0)
        self.assertEqual(summary["shifted_row_ms"], 6.0)
        self.assertEqual(summary["shifted_kv_ready_events"], 4)
        self.assertEqual(summary["shifted_kv_ready_waits"], 5)
        self.assertEqual(summary["shifted_kv_syncs_deferred"], 6)
        self.assertEqual(summary["sampling_ms"], 41.5)
        self.assertEqual(summary["sampling_enqueue_ms"], 0.5)
        self.assertEqual(summary["stochastic_distribution_build_gpu_ms"], 8.0)
        self.assertEqual(summary["stochastic_distribution_batch_build_gpu_ms"], 9.0)
        self.assertEqual(summary["stochastic_processed_rows_build_gpu_ms"], 9.5)
        self.assertEqual(summary["stochastic_batch_outcome_ms"], 10.0)
        self.assertEqual(summary["resident_outcome_enqueue_ms"], 12.0)
        self.assertEqual(summary["resident_outcome_host_bridge_ms"], 14.0)
        self.assertEqual(summary["stochastic_batch_gpu_reducer_ms"], 13.0)
        self.assertEqual(summary["first_sidecar_prelaunch_ms"], 1.5)
        self.assertEqual(summary["first_sidecar_prelaunches"], 3)
        self.assertEqual(summary["first_sidecar_prelaunch_reuses"], 2)
        self.assertEqual(summary["first_sidecar_prelaunch_drops"], 1)
        self.assertEqual(summary["first_sidecar_prelaunch_discarded_complete"], 2)
        self.assertEqual(summary["first_sidecar_resident_ready_inputs"], 4)
        self.assertEqual(summary["first_sidecar_resident_condition_inputs"], 5)
        self.assertEqual(summary["sidecar_device_token_inputs"], 10)
        self.assertEqual(summary["sidecar_device_token_inputs_from_host"], 3)
        self.assertEqual(summary["sidecar_device_token_inputs_from_device"], 7)
        self.assertEqual(summary["outcome_catchup_plan_ms"], 0.75)
        self.assertEqual(summary["transaction_plan_ms"], 0.75)
        self.assertEqual(summary["host_state_adoption_ms"], 0.375)
        self.assertEqual(summary["transaction_output_commit_ms"], 0.125)
        self.assertEqual(summary["stochastic_batch_d2h_sync_ms"], 23.0)
        self.assertEqual(summary["stochastic_batch_response_ready_wait_ms"], 2.25)
        self.assertEqual(summary["stochastic_batch_d2h_enqueue_ms"], 0.75)
        self.assertEqual(summary["stochastic_batch_d2h_wait_ms"], 22.25)
        self.assertEqual(summary["bridge_stream_create_ms"], 1.5)
        self.assertEqual(summary["bridge_stream_creations"], 1.0)
        self.assertEqual(summary["bridge_stream_reuses"], 4.0)
        self.assertEqual(summary["main_decode_graph_replay_gpu_ms"], 14.0)
        self.assertEqual(summary["main_verifier_graph_replay_gpu_ms"], 35.0)
        self.assertEqual(summary["main_verifier_stage_sample_gpu_ms"], 31.0)
        self.assertEqual(summary["main_verifier_moe_expert_ffn_gpu_ms"], 21.0)
        self.assertEqual(summary["main_verifier_moe_router_gpu_ms"], 4.0)
        self.assertEqual(summary["main_verifier_gdn_projection_gpu_ms"], 3.0)
        self.assertEqual(summary["main_verifier_gdn_recurrence_gpu_ms"], 2.0)
        self.assertEqual(summary["main_verifier_attention_gpu_ms"], 1.5)
        self.assertEqual(summary["main_verifier_lm_head_gpu_ms"], 1.25)
        self.assertEqual(summary["sidecar_graph_replay_gpu_ms"], 6.0)
        self.assertEqual(summary["sidecar_replay_reset_ms"], 44.0)
        self.assertEqual(summary["greedy_summary_ms"], 30.0)
        self.assertEqual(summary["checkpoint_ms"], 6.0)
        self.assertEqual(summary["sidecar_graph_hits"], 7)
        self.assertEqual(summary["sidecar_graph_misses"], 8)
        self.assertEqual(summary["main_decode_warmup"], 6)
        self.assertEqual(summary["main_decode_capture"], 2)
        self.assertEqual(summary["main_decode_replay"], 9)
        self.assertEqual(summary["main_verifier_warmup"], 4)
        self.assertEqual(summary["main_verifier_capture"], 1)
        self.assertEqual(summary["main_verifier_replay"], 5)
        self.assertEqual(summary["replay_resets"], 2)
        self.assertEqual(summary["replay_preserves"], 6)
        self.assertEqual(summary["sidecar_replay_reset_after_spec_publication"], 4)
        self.assertEqual(summary["sidecar_replay_preserved_for_spec_publication"], 3)
        self.assertEqual(summary["replay_reset_caches"], 0)
        self.assertEqual(summary["replay_rebind_caches"], 6)
        self.assertEqual(summary["replay_ordinary_decode_resets"], 0)
        self.assertEqual(summary["replay_verifier_rebinds"], 2)
        self.assertEqual(summary["replay_other_rebinds"], 4)

    def test_missing_path_emits_zero_tsv(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "/tmp/does-not-exist-llaminar-perfstats.json"],
            check=True,
            text=True,
            capture_output=True,
        )
        values = result.stdout.strip().split("\t")
        self.assertEqual(len(values), EXPECTED_FIELD_COUNT)
        self.assertTrue(all(value in ("0", "0.0") for value in values))

    def test_multiple_paths_emit_table_for_matrix_comparison(self) -> None:
        paths: list[Path] = []
        for total in (11.0, 22.0):
            payload = {
                "schema": "llaminar.perf_stats.v1",
                "records": [
                    {
                        "domain": "mtp",
                        "name": "verifier_forward",
                        "phase": "decode",
                        "count": 1,
                        "total_ms": total,
                    },
                ],
            }
            handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
            with handle:
                json.dump(payload, handle)
                paths.append(Path(handle.name))

        try:
            result = subprocess.run(
                [sys.executable, str(SCRIPT), *(str(path) for path in paths)],
                check=True,
                text=True,
                capture_output=True,
            )
        finally:
            for path in paths:
                path.unlink(missing_ok=True)

        lines = result.stdout.strip().splitlines()
        self.assertEqual(lines[0].split("\t")[0], "path")
        self.assertIn("\t11.0\t", lines[1])
        self.assertIn("\t22.0\t", lines[2])


if __name__ == "__main__":
    unittest.main()
