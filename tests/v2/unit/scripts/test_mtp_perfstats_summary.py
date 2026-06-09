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
                    "name": "all_position_correction_forward",
                    "phase": "decode",
                    "count": 2,
                    "total_ms": 7.5,
                },
                {
                    "domain": "mtp",
                    "name": "all_position_publish_accepted_state",
                    "phase": "decode",
                    "count": 3,
                    "total_ms": 1.25,
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
                    "tags": {"context": "main_decode", "phase": "replay"},
                },
                {
                    "domain": "mtp",
                    "name": "live_prefix_replay_state_after_mutation",
                    "phase": "decode",
                    "count": 2,
                    "tags": {"replay_state": "reset"},
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
        self.assertEqual(summary["correction_ms"], 7.5)
        self.assertEqual(summary["correction_count"], 2)
        self.assertEqual(summary["publish_ms"], 1.25)
        self.assertEqual(summary["main_verifier_warmup"], 4)
        self.assertEqual(summary["main_verifier_capture"], 1)
        self.assertEqual(summary["main_verifier_replay"], 5)
        self.assertEqual(summary["replay_resets"], 2)
        self.assertEqual(summary["replay_preserves"], 6)

    def test_missing_path_emits_zero_tsv(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "/tmp/does-not-exist-llaminar-perfstats.json"],
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertEqual(
            result.stdout.strip(),
            "0.0\t0.0\t0.0\t0\t0.0\t0\t0\t0\t0\t0",
        )


if __name__ == "__main__":
    unittest.main()
