#!/usr/bin/env python3
"""Unit checks for scripts/summarize_mtp_perfstats.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "summarize_mtp_perfstats.py"


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
                        "forward_replay_reset_cache_count": "1",
                        "forward_replay_stream_rebind_cache_count": "2",
                        "forward_replay_ordinary_decode_reset_count": "1",
                        "forward_replay_all_position_verifier_rebind_count": "1",
                        "forward_replay_other_rebind_count": "1",
                    },
                },
                {
                    "domain": "mtp",
                    "name": "live_prefix_replay_state_after_mutation",
                    "phase": "decode",
                    "count": 6,
                    "tags": {"replay_state": "preserved"},
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
        self.assertEqual(summary["verifier_ms"], 18.25)
        self.assertEqual(summary["condition_ms"], 9.75)
        self.assertEqual(summary["condition_count"], 2)
        self.assertEqual(summary["condition_skipped_ready"], 5)
        self.assertEqual(summary["correction_ms"], 7.5)
        self.assertEqual(summary["correction_count"], 2)
        self.assertEqual(summary["deferred_corrections"], 3)
        self.assertEqual(summary["rejection_no_ready"], 4)
        self.assertEqual(summary["publish_ms"], 1.25)
        self.assertEqual(summary["publish_count"], 3)
        self.assertAlmostEqual(summary["publish_avg_ms"], 1.25 / 3.0)
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
        self.assertEqual(summary["sampling_ms"], 1.5)
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
        self.assertEqual(summary["replay_reset_caches"], 2)
        self.assertEqual(summary["replay_rebind_caches"], 4)
        self.assertEqual(summary["replay_ordinary_decode_resets"], 2)
        self.assertEqual(summary["replay_verifier_rebinds"], 2)
        self.assertEqual(summary["replay_other_rebinds"], 2)

    def test_missing_path_emits_zero_tsv(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "/tmp/does-not-exist-llaminar-perfstats.json"],
            check=True,
            text=True,
            capture_output=True,
        )
        values = result.stdout.strip().split("\t")
        self.assertEqual(len(values), 40)
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
