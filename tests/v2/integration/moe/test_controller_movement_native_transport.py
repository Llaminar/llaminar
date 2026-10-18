#!/usr/bin/env python3
"""Check actual C++ completed-movement exports with the independent Python policy.

The required executable uses production publishers and JSON projections with
device-free metadata. Its ordered evidence must reject corruption even when
counter totals, endpoint ranges and the original HTTP history are unchanged.
No payload, model or GPU is involved; a missing executable is a hard failure.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/ci"))
sys.path.insert(0, str(ROOT / "tests/v2/e2e/server"))
from generation_movement_ledger import validate_movement_transport_mirrors, validate_controller_movement_transport
from runtime_feature_perf_policy import MovementEvidence, validate_runtime_feature_policy
from server_execution_contract import RuntimeFeaturePolicy


class NativeControllerMovementTransport(unittest.TestCase):
    """The wire fixture is emitted once by the real C++ projection functions."""

    @classmethod
    def setUpClass(cls):
        """Require the registered native executable; no synthetic replacement is allowed."""
        result = subprocess.run([cls.fixture], check=True, capture_output=True, text=True, timeout=20)
        cls.document = json.loads(result.stdout)

    def test_native_cpp_export_preserves_large_ids_followers_and_http_cutoff(self):
        """Full terminal history may legitimately extend an earlier response."""
        data = self.document
        self.assertEqual(len(data["records"]), 24)
        self.assertGreater(data["terminal_movement"][0]["device_publications"][0]["physical_payload_bytes"], 2**54)
        self.assertEqual(data["terminal_movement"][1]["movement"]["economy"], [])
        validate_movement_transport_mirrors(data["http_movement"], data["records"], data["terminal_movement"])
        self.assertIsNone(validate_runtime_feature_policy(data["records"], RuntimeFeaturePolicy(),
            MovementEvidence.REQUIRED, terminal_movement=data["terminal_movement"]))

    def test_native_cpp_export_rejects_modified_interior_history_and_capacity(self):
        """Every mutation alters only metadata that used to live in per-transaction keys."""
        for mutation in ("actual_bytes", "middle_wave", "edge", "backend", "capacity", "economy", "follower", "counter"):
            data = copy.deepcopy(self.document)
            terminal = data["terminal_movement"][0]
            publication = terminal["device_publications"][1]
            if mutation == "actual_bytes":
                publication["physical_payload_bytes"] += 1
            elif mutation == "middle_wave":
                del terminal["device_publications"][1]
            elif mutation == "edge":
                del terminal["movement"]["edges"][2]
            elif mutation == "backend":
                terminal["movement"]["edges"][2]["source_device"] = "CUDA:0"
            elif mutation == "capacity":
                publication["controller"]["participant_flow_violations"] = 1
            elif mutation == "economy":
                publication["controller"]["projected_net_benefit_ns"] += 1
            elif mutation == "follower":
                data["terminal_movement"][1]["movement"]["economy"] = terminal["movement"]["economy"]
            else:
                data["records"][1]["sequence_digest_lo"] ^= 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_controller_movement_transport(data["records"], data["terminal_movement"])


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "--fixture":
        raise SystemExit("--fixture requires the registered native movement executable")
    NativeControllerMovementTransport.fixture = sys.argv[2]
    del sys.argv[1:3]
    unittest.main()
