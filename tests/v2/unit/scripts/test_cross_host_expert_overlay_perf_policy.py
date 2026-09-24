#!/usr/bin/env python3
"""Adversarial, device-free regressions for remote CPU computation certification.

Fixtures reproduce the production collector's record shapes. They are not live
cloud/inference evidence; the tests prove that an idle rank, header-only wire,
foreign phase, missing endpoint or local shared-page transport cannot pass the
post-shutdown observer.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "e2e/server"))
from cross_host_expert_overlay_perf_policy import validate_cross_host_expert_execution


def example(remote_hosts: int = 2, authority: int = 1, backend: str = "rocm",
            decode_phase: str = "grouped_verifier") -> tuple[dict, dict]:
    """Build paired endpoint observations, with nonzero authority by default."""
    size = remote_hosts + 1
    topology = {"kind": "cross-host-expert-overlay", "remote_cpu_hosts": remote_hosts,
                "execution_ranks": size, "cpu_ranks_per_host": 1, "continuation_devices": 1,
                "continuation_backend": backend}
    data = {"schema": "llaminar.perf_stats.v1", "world_size": size,
            "authority_rank": authority, "records": []}
    records = data["records"]
    for rank in range(size):
        records.append({"domain": "server", "name": "rank_membership", "kind": "counter",
            "phase": "startup", "count": 1, "value": 1.0, "rank": rank,
            "tags": {"rank": str(rank), "world_size": str(size), "authority_rank": str(authority),
                "node_id": str(rank), "local_rank": "0", "hostname": "reused-container-label",
                "identity_source": "communicator_cluster_inventory"}})
        if rank == authority:
            continue
        for endpoint, observer in (("source", authority), ("target", rank)):
            records.append({"domain": "forward_graph", "name": "moe_overlay_graph_completion_sequence",
                "kind": "ordered_sequence", "rank": observer, "phase": "inference", "device": "mpi",
                "count": 2, "value": 2.0, "sequence_word_count": 28,
                "sequence_digest_lo": 93, "sequence_digest_hi": 71,
                "tags": {"source_world_rank": str(authority), "target_world_rank": str(rank),
                         "endpoint_role": endpoint}})
        phases = ("prefill", decode_phase)
        for phase in phases:
            records.append({"domain": "forward_graph", "name": "moe_overlay_local_expert_active_routes",
                "kind": "counter", "rank": rank, "phase": "moe_overlay", "count": 1, "value": 4.0,
                "device": "cpu:0", "tags": {"completion": "local_expert_packet_complete",
                    "identity_source": "sparse_collective_key", "device_kind": "CPU", "tier": "1",
                    "input_rows": "2", "output_rows": "2", "service_source": phase}})
        for direction, label in (("Dispatch", "dispatch"), ("ReturnReduce", "return")):
            for endpoint, observer in (("source", authority), ("target", rank)):
                for depth in sorted({0 if phase == "grouped_verifier" else -1 for phase in phases}):
                    grouped = [phase for phase in phases if (0 if phase == "grouped_verifier" else -1) == depth]
                    tags = {"source_world_rank": str(authority), "target_world_rank": str(rank),
                            "domain_ordinal": "2", "layer": "7", "mtp_depth": str(depth), "tier": "1",
                            "direction": direction, "endpoint_role": endpoint, "transport": "mpi",
                            "participant_count": "1"}
                    base = {"domain": "forward_graph", "device": "mpi", "rank": observer, "tags": tags}
                    records.extend({**base, "name": name, "phase": "moe_overlay", "kind": "counter",
                        "count": len(grouped), "value": value} for name, value in (
                            (f"moe_overlay_rank_batch_{label}_transactions", float(len(grouped))),
                            ("moe_overlay_rank_batch_payload_bytes", len(grouped) * (4096.0 if label == "dispatch" else 2048.0))))
                    records.extend({**base, "name": "moe_overlay_rank_batch_sequence", "phase": phase,
                        "kind": "ordered_sequence", "count": 1, "value": 1.0,
                        "sequence_word_count": 4, "sequence_digest_lo": 1234,
                        "sequence_digest_hi": 5678} for phase in grouped)
    # Each JSON record has independent tags as it would after parsing actual
    # output. Mutating one witness must not accidentally edit its peer fixture.
    import json
    return json.loads(json.dumps(data)), topology


def first(data: dict, suffix: str, *, endpoint: str | None = None) -> dict:
    """Select one exact witness for a targeted corruption, never patch all peers."""
    return next(row for row in data["records"] if row["name"].endswith(suffix)
                and (endpoint is None or row["tags"].get("endpoint_role") == endpoint))


def add_empty_outcomes(data: dict) -> None:
    """Interleave an empty dispatch after each real one without inventing return bytes."""
    additions = []
    for row in data["records"]:
        tags = row["tags"]
        if tags.get("direction") == "ReturnReduce":
            if row["name"] == "moe_overlay_rank_batch_sequence":
                empty = deepcopy(row)
                empty["name"] = "moe_overlay_rank_batch_empty_return_sequence"
                empty["sequence_digest_lo"] += 17
                additions.append(empty)
            elif row["name"] == "moe_overlay_rank_batch_return_transactions":
                empty = deepcopy(row)
                empty["name"] = "moe_overlay_rank_batch_empty_return_transactions"
                additions.append(empty)
        elif tags.get("direction") == "Dispatch":
            if row["name"] == "moe_overlay_rank_batch_sequence":
                row["count"] *= 2
                row["value"] *= 2
                row["sequence_word_count"] *= 2
                row["sequence_digest_lo"] += 23
            else:
                row["value"] += row["count"] * (112 if row["name"].endswith("payload_bytes") else 1)
                row["count"] *= 2
    data["records"].extend(additions)


class CrossHostExpertExecutionTests(unittest.TestCase):
    """Missing or inconsistent computation/transport evidence is a hard failure."""

    def test_both_backends_host_counts_and_authority_positions(self):
        for backend in ("cuda", "rocm"):
            for count in (1, 2, 3):
                for authority in range(count + 1):
                    with self.subTest(backend=backend, count=count, authority=authority):
                        data, topology = example(count, authority, backend)
                        original = deepcopy(data)
                        proof = validate_cross_host_expert_execution(data, topology)
                        self.assertEqual(data, original)
                        self.assertEqual(proof["authority_rank"], authority)
                        self.assertEqual(len(proof["remote_cpu_hosts"]), count)
                        for rank in proof["remote_cpu_hosts"]:
                            self.assertNotEqual(rank["rank"], authority)
                            self.assertEqual(rank["prefill_routes"], 4)
                            self.assertEqual(rank["decode_routes"], 4)
                            self.assertEqual(rank["dispatch_transactions"], 2)
                            self.assertEqual(rank["return_transactions"], 2)
                            self.assertEqual(rank["dispatch_bytes"], 8192)  # Never double-count endpoints.
                            self.assertEqual(rank["return_bytes"], 4096)

    def test_serial_decode_also_qualifies(self):
        data, topology = example(decode_phase="decode")
        validate_cross_host_expert_execution(data, topology)

    def test_empty_outcomes_are_paired_without_counting_them_as_wire_returns(self):
        for phase in ("decode", "grouped_verifier"):
            data, topology = example(decode_phase=phase)
            add_empty_outcomes(data)
            proof = validate_cross_host_expert_execution(data, topology)
            for rank in proof["remote_cpu_hosts"]:
                self.assertEqual(rank["dispatch_transactions"], 4)
                self.assertEqual(rank["return_transactions"], 2)
                self.assertEqual(rank["empty_return_outcomes"], 2)
                self.assertEqual(rank["return_bytes"], 4096)
                self.assertEqual(rank["dispatch_bytes"], 8192 + 224)
                self.assertEqual(rank["completed_graph_transactions"], 2)

    def test_empty_outcomes_cannot_mask_missing_mismatched_or_extra_evidence(self):
        data, topology = example()
        add_empty_outcomes(data)
        for index, row in enumerate(data["records"]):
            if "empty_return" not in row["name"]:
                continue
            for mutation in ("remove", "duplicate", "value", "direction", "layer"):
                with self.subTest(index=index, mutation=mutation):
                    broken = deepcopy(data)
                    if mutation == "remove":
                        broken["records"].pop(index)
                    elif mutation == "duplicate":
                        broken["records"].append(deepcopy(row))
                    elif mutation == "value":
                        broken["records"][index]["value"] += 1
                    else:
                        broken["records"][index]["tags"][mutation] = "Dispatch" if mutation == "direction" else "8"
                    with self.assertRaises(ValueError):
                        validate_cross_host_expert_execution(broken, topology)

    def test_entire_empty_layer_has_no_physical_return_group(self):
        data, topology = example()
        additions = []
        for row in data["records"]:
            if row["tags"].get("mtp_depth") != "-1":
                continue
            extra = deepcopy(row)
            extra["tags"]["layer"] = "8"
            if extra["tags"].get("direction") == "ReturnReduce":
                if extra["name"] == "moe_overlay_rank_batch_sequence":
                    extra["name"] = "moe_overlay_rank_batch_empty_return_sequence"
                elif extra["name"] == "moe_overlay_rank_batch_return_transactions":
                    extra["name"] = "moe_overlay_rank_batch_empty_return_transactions"
                else:
                    continue  # An elided return has no payload-byte observation.
            elif extra["name"] == "moe_overlay_rank_batch_payload_bytes":
                extra["value"] = 112.0
            additions.append(extra)
        data["records"].extend(additions)
        proof = validate_cross_host_expert_execution(data, topology)
        for rank in proof["remote_cpu_hosts"]:
            self.assertEqual(rank["empty_return_outcomes"], 1)
            self.assertEqual(rank["return_transactions"], 2)
            self.assertEqual(rank["dispatch_transactions"], 3)
        # Removing the real prefill returns cannot be hidden by the empty layer.
        data["records"] = [row for row in data["records"] if not (
            row["tags"].get("direction") == "ReturnReduce" and row["tags"].get("mtp_depth") == "-1"
            and row["tags"].get("layer") == "7")]
        with self.assertRaises(ValueError):
            validate_cross_host_expert_execution(data, topology)

    def test_graph_completion_is_required_independently_of_numerical_outcomes(self):
        for mutation in ("remove", "duplicate", "sequence_digest_lo", "sequence_word_count", "count", "rank", "phase"):
            data, topology = example()
            add_empty_outcomes(data)
            row = first(data, "graph_completion_sequence")
            if mutation == "remove":
                data["records"].remove(row)
            elif mutation == "duplicate":
                data["records"].append(deepcopy(row))
            elif mutation == "phase":
                row["phase"] = "decode"
            else:
                row[mutation] += 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_cross_host_expert_execution(data, topology)

    def test_depth_marker_is_canonical_and_matches_phase(self):
        for value in ("-2", "-01", "-0", "+0", "01", "2147483648", True, None):
            with self.subTest(value=value):
                data, topology = example()
                for row in data["records"]:
                    if "mtp_depth" in row["tags"]:
                        row["tags"]["mtp_depth"] = value
                with self.assertRaises(ValueError):
                    validate_cross_host_expert_execution(data, topology)
        for phase, depth in (("prefill", "0"), ("decode", "0"), ("grouped_verifier", "-1")):
            data, topology = example(decode_phase="decode" if phase == "decode" else "grouped_verifier")
            for row in data["records"]:
                if row["phase"] == phase and "mtp_depth" in row["tags"]:
                    row["tags"]["mtp_depth"] = depth
            with self.assertRaises(ValueError):
                validate_cross_host_expert_execution(data, topology)

    def test_every_individual_witness_is_required(self):
        data, topology = example()
        for index, row in enumerate(data["records"]):
            with self.subTest(index=index, name=row["name"]):
                broken = deepcopy(data)
                broken["records"].pop(index)
                with self.assertRaises(ValueError):
                    validate_cross_host_expert_execution(broken, topology)

    def test_idle_cpu_rank_and_header_only_payload_cannot_pass(self):
        for idle_phase in (None, "prefill", "grouped_verifier"):
            with self.subTest(phase=idle_phase):
                data, topology = example()
                for row in data["records"]:
                    if (row["rank"] == 0 and row["name"] == "moe_overlay_local_expert_active_routes"
                            and (idle_phase is None or row["tags"]["service_source"] == idle_phase)):
                        row["value"] = 0.0
                with self.assertRaisesRegex(ValueError, "rank 0.*prefill and decode"):
                    validate_cross_host_expert_execution(data, topology)

    def test_same_node_relabelled_as_different_hosts_is_not_remote(self):
        data, topology = example()
        for row in data["records"]:
            if row["name"] == "rank_membership":
                row["tags"]["hostname"] = f"pretend-{row['rank']}"
                row["tags"]["node_id"] = "0"
        with self.assertRaisesRegex(ValueError, "share a physical host"):
            validate_cross_host_expert_execution(data, topology)

    def test_malformed_membership(self):
        for field, values in {"identity_source": (None, "hostname"), "node_id": (None, "-1", "01"),
                "local_rank": (None, "1"), "hostname": (None, "", "bad\nhost"),
                "world_size": (None, "2"), "rank": (None, "1"), "authority_rank": (None, "0")}.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    data, topology = example()
                    first(data, "rank_membership")["tags"][field] = value
                    with self.assertRaises(ValueError):
                        validate_cross_host_expert_execution(data, topology)

    def test_missing_and_duplicate_membership(self):
        for mutation in ("duplicate", "value", "count", "rank", "phase"):
            with self.subTest(mutation=mutation):
                data, topology = example()
                row = first(data, "rank_membership")
                if mutation == "duplicate":
                    data["records"].append(deepcopy(row))
                else:
                    row[mutation] = {"value": True, "count": 2, "rank": 3, "phase": "inference"}[mutation]
                with self.assertRaises(ValueError):
                    validate_cross_host_expert_execution(data, topology)

    def test_topology_count_and_backend_must_be_explicit(self):
        for field in ("remote_cpu_hosts", "execution_ranks", "continuation_devices", "cpu_ranks_per_host"):
            for value in (None, True, 0, -1, "1", 1.5):
                with self.subTest(field=field, value=value):
                    data, topology = example()
                    topology[field] = value
                    with self.assertRaises(ValueError):
                        validate_cross_host_expert_execution(data, topology)
        for backend in (None, "CPU", "cuda+rocm"):
            data, topology = example()
            topology["continuation_backend"] = backend
            with self.assertRaises(ValueError):
                validate_cross_host_expert_execution(data, topology)

    def test_invalid_counters_cannot_supply_positive_work(self):
        for suffix in ("active_routes", "payload_bytes", "dispatch_transactions"):
            for value in (True, -1, "4", float("nan"), float("inf"), 1.5, 2**53, 2**1024):
                with self.subTest(suffix=suffix, value=value):
                    data, topology = example()
                    first(data, suffix)["value"] = value
                    with self.assertRaises(ValueError):
                        validate_cross_host_expert_execution(data, topology)

    def test_route_completion_identity_and_phase_cannot_be_substituted(self):
        for field, value in (("completion", "dispatch_only"), ("identity_source", "host_shadow"),
                ("device_kind", "ROCm"), ("service_source", "synthetic_test"),
                ("service_source", "decode"), ("input_rows", "0"), ("output_rows", "0"), ("tier", "7")):
            with self.subTest(field=field):
                data, topology = example()
                first(data, "active_routes")["tags"][field] = value
                with self.assertRaises(ValueError):
                    validate_cross_host_expert_execution(data, topology)

    def test_local_channel_can_never_claim_cross_host_work(self):
        data, topology = example()
        for row in data["records"]:
            if "transport" in row["tags"]:
                row["tags"]["transport"] = "node_local_shared_rows"
                row["device"] = "node_local_shared_rows"
        with self.assertRaisesRegex(ValueError, "node-local activation channel"):
            validate_cross_host_expert_execution(data, topology)

    def test_foreign_or_relabelled_transport_geometry(self):
        for field, value in (("source_world_rank", "2"), ("target_world_rank", "1"),
                ("endpoint_role", "target"), ("domain_ordinal", "3"), ("layer", "8"),
                ("mtp_depth", "1"), ("tier", "0"), ("participant_count", "0"),
                ("participant_count", "2"), ("direction", "ReturnReduce")):
            with self.subTest(field=field):
                data, topology = example()
                first(data, "rank_batch_sequence", endpoint="source")["tags"][field] = value
                with self.assertRaises(ValueError):
                    validate_cross_host_expert_execution(data, topology)

    def test_sequence_abi_digests_and_counts_are_matched(self):
        for field, value in (("count", 2), ("count", True), ("value", True), ("value", 2),
                ("sequence_word_count", 3), ("sequence_digest_lo", 42), ("sequence_digest_hi", 42),
                ("sequence_digest_hi", 2**64), ("phase", "decode"), ("kind", "counter")):
            with self.subTest(field=field):
                data, topology = example()
                first(data, "rank_batch_sequence", endpoint="target")[field] = value
                with self.assertRaises(ValueError):
                    validate_cross_host_expert_execution(data, topology)

    def test_unreturned_dispatch_cannot_pass_even_when_other_phases_complete(self):
        data, topology = example()
        data["records"] = [row for row in data["records"] if row["tags"].get("direction") != "ReturnReduce"]
        with self.assertRaisesRegex(ValueError, "no completed matching return"):
            validate_cross_host_expert_execution(data, topology)

    def test_duplicate_transport_or_counter_is_rejected(self):
        for suffix in ("rank_batch_sequence", "payload_bytes", "dispatch_transactions"):
            data, topology = example()
            data["records"].append(deepcopy(first(data, suffix)))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                validate_cross_host_expert_execution(data, topology)

    def test_counter_and_sequence_totals_cannot_differ(self):
        for suffix in ("payload_bytes", "dispatch_transactions"):
            for field in ("value", "count"):
                data, topology = example()
                first(data, suffix, endpoint="target")[field] += 1
                with self.assertRaisesRegex(ValueError, "counters or bytes differ"):
                    validate_cross_host_expert_execution(data, topology)

    def test_harness_rejects_incomplete_remote_admission_before_startup(self):
        """Exercise real shell option parsing; no model, Docker or Azure is started."""
        root = Path(__file__).resolve().parents[4]
        harness = root / "tests/v2/e2e/server/test_server_e2e.sh"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            config, argv = path / "config.json", path / "args.json"
            config.write_text(json.dumps({"schema": 1, "scope": "cross-host-e2e",
                "source_revision": "revision", "cells": [{"configuration": {
                    "model_parity_schema": 1, "id": "unrequested", "model": "none",
                    "e2e": {}, "cross_host_e2e": []}}]}))
            argv.write_text(json.dumps(["--only-strategies", "expert-overlay"]))
            base = ["bash", str(harness), "--suite", "absent-model|tp", "--server-args-file", str(argv)]
            for options, profiling, expected in (
                (["--cross-host-case", "absent"], "1", "Cross-host evidence requires"),
                (["--cross-host-configuration", str(config)], "1", "Cross-host evidence requires"),
                (["--cross-host-configuration", str(config), "--cross-host-case", "absent"],
                 "0", "Cross-host evidence requires"),
                (["--cross-host-configuration", str(config), "--cross-host-case", "absent"],
                 "1", "absent or duplicated in the canonical manifest"),
            ):
                with self.subTest(options=options, profiling=profiling):
                    completed = subprocess.run(base + options, cwd=root, text=True, capture_output=True,
                        timeout=10, env={**os.environ, "LLAMINAR_E2E_PERF_STATS": profiling,
                                        "LLAMINAR_E2E_LOG_DIR": str(path / "logs")})
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn(expected, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
