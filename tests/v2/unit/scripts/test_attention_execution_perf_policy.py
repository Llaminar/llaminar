#!/usr/bin/env python3
"""Device-free regressions for declared FA/QSA ownership and native evidence.

Tiny authored observations model exact runtime boundaries. No model artifact,
accelerator, backend initialization or copied production JSON is needed. Mixed
families deliberately share an owner so a dense predictor cannot hide a missing
sparse trunk. Full graph lifecycle remains the separate graph contract's gate.
"""
from __future__ import annotations

import copy
import contextlib
from contextlib import ExitStack
import io
import os
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import sys
import unittest

SERVER = Path(__file__).resolve().parents[2] / "e2e/server"
sys.path.insert(0, str(SERVER))
from attention_execution_perf_policy import AttentionExecutionValidation, validate_attention_execution_policy
from server_execution_contract import validate_server_execution_contract
from test_server_execution_contract import evidence
sys.path.insert(0, str(SERVER))


def cell(device="ROCm:0", *, dense=False, rank_count=1):
    """Explicit family obligations repeat per rank even when ordinals repeat."""
    data = evidence([(device, device)] * rank_count)
    for row in data["records"]:
        if row["name"] == "execution_topology":
            row["tags"].update(flash_attention_devices=device if dense else "", sparse_attention_devices=device)
    return data


def qsa(device="ROCm:0", *, rank=0, phase="decode", rows="1", signature="17"):
    """Setup and actual recording carry the same immutable native signature."""
    policy = "fp32" if device in {"CPU", "CUDA:0"} else "fp16_tile_33" if phase == "decode" else "fp16_tile"
    base = {"rank": rank, "device": device, "value": 1, "tags": {
        "phase": phase, "policy": policy, "program": policy,
        "parts": "33" if policy == "fp16_tile_33" else "8", "rows": rows,
        "query_heads": "6", "kv_heads": "1", "query_head_start": "0", "kv_head_start": "0",
        "capture_signature": signature}}
    setup = {**copy.deepcopy(base), "domain": "qsa", "name": "attention_arithmetic_policy", "phase": "setup"}
    actual = {**copy.deepcopy(base), "domain": "qsa" if device == "CPU" else "gpu_graph_inventory",
              "name": "attention_program_executions" if device == "CPU" else "qsa_attention_program_captures",
              "phase": "execute" if device == "CPU" else "capture_setup"}
    return [setup, actual]


def inventory(device="ROCm:0", rank=0):
    """The durable native graph inventory proves the captured sparse stage."""
    return {"domain": "forward_graph", "name": "full_graph_capture_stage_types", "rank": rank,
        "device": device, "phase": "decode", "value": 12, "tags": {
        "source": "full_graph_capture", "attribution": "graph_replay_metadata",
        "graph_capture_scope": "full_graph_capture_plan", "stage_type": "QWEN4EXP_QSA_ATTENTION",
        "type": "captured_executable", "context": "main_decode"}}


def flash(device="ROCm:0", rank=0):
    """Small valid dense FA plan exercises the unchanged canonical validator."""
    backend = device.partition(":")[0].lower()
    return {"domain": "gpu_graph_inventory", "name": backend + "_fa2_parallel_plan_selections",
        "phase": "capture_setup", "device": device, "rank": rank, "value": 1, "tags": {
        "requested_axis": "geometry_selected", "selected_axis": "query_sequence",
        "batch_size": "1", "query_rows": "64", "local_query_heads": "6", "head_dim": "256",
        "kv_capacity": "512", "query_grid_blocks": "24", "kv_storage": "FP16",
        "context_partitions": "0", "context_partition_slots": "0", "device_direct_partition_limit": "0",
        "tile_q": "16", "tile_kv": "32", "context_phase_block_slots": "0",
        "reducer_dimension_wavefronts": "0", "reducer_block_slots": "0",
        "query_warp_groups": "4", "reducer_dimension_warps": "0"}}


def complete(device="ROCm:0", *, dense=False, rank_count=1):
    """Minimal successful evidence for both semantic phases and every owner."""
    data = cell(device, dense=dense, rank_count=rank_count)
    for rank in range(rank_count):
        data["records"] += qsa(device, rank=rank) + qsa(device, rank=rank, phase="prefill", rows="64", signature="18")
        if device != "CPU":
            data["records"].append(inventory(device, rank))
            data["records"].append({"domain": "forward_graph", "name": "prefill_graph_phase",
                "rank": rank, "device": device, "phase": "prefill", "value": 1,
                "tags": {"capture_phase": "replay", "bucket_seq_len": "64"}})
        if dense and device != "CPU":
            data["records"].append(flash(device, rank))
    return data


def validate(data):
    """Execute the shared production wrapper after authenticating startup policy."""
    return validate_attention_execution_policy(data["records"], validate_server_execution_contract(data))


class AttentionExecutionPolicyTests(unittest.TestCase):
    """Every supported backend/family obeys exact ownership and semantic phase."""

    def test_cpu_cuda_rocm_sparse_and_mixed_attention(self):
        for device in ("CPU", "CUDA:0", "ROCm:0"):
            for dense in (False, True):
                with self.subTest(device=device, dense=dense):
                    result = validate(complete(device, dense=dense, rank_count=2))
                    self.assertIsNone(result.error)
                    self.assertEqual(result.sparse_participants, 2)
                    self.assertEqual(result.flash_participants, 2 if dense else 0)

    def test_dense_only_keeps_canonical_flash_attention_policy(self):
        for device in ("CPU", "CUDA:0", "ROCm:0"):
            data = cell(device, dense=True)
            data["records"][0]["tags"]["sparse_attention_devices"] = ""
            if device != "CPU":
                data["records"].append(flash(device))
            self.assertIsNone(validate(data).error)
            if device != "CPU":
                data["records"][-1]["tags"]["requested_axis"] = "query_sequence"
                self.assertIsNotNone(validate(data).error)

    def test_absent_qsa_fails_even_with_surviving_dense_sidecar(self):
        for device in ("CPU", "CUDA:0", "ROCm:0"):
            data = cell(device, dense=True)
            if device != "CPU":
                data["records"].append(flash(device))
            self.assertIn("no QSA", validate(data).error)

    def test_setup_and_graph_stage_without_actual_program_capture_fail(self):
        data = complete()
        data["records"] = [r for r in data["records"] if r["name"] != "qsa_attention_program_captures"]
        self.assertIn("no QSA", validate(data).error)

    def test_rank_device_and_phase_cannot_borrow_neighbor_evidence(self):
        for device in ("CPU", "CUDA:0", "ROCm:0"):
            for field, value in (("rank", 1), ("rank", True), ("device", "ROCm:7"), ("phase", "setup")):
                with self.subTest(device=device, field=field):
                    data = complete(device)
                    data["records"][3][field] = value
                    self.assertIsNotNone(validate(data).error)

    def test_missing_rank_and_semantic_phase_fail(self):
        data = complete(rank_count=2)
        data["records"] = [r for r in data["records"] if r["name"] != "qsa_attention_program_captures" or r["rank"] == 0]
        self.assertIsNotNone(validate(data).error)
        for phase in ("prefill", "decode"):
            data = complete()
            data["records"] = [r for r in data["records"] if r["name"] != "qsa_attention_program_captures" or r["tags"]["phase"] != phase]
            self.assertIn("phase", validate(data).error)

    def test_malformed_policy_geometry_and_nonfinite_counts_fail(self):
        for tags in ({"phase": "unspecified"}, {"policy": "automatic"}, {"program": "fp32"},
                     {"parts": "8"}, {"rows": "0"}, {"query_heads": "bad"}, {"kv_heads": "0"},
                     {"query_head_start": "-1"}, {"kv_head_start": "01"}, {"capture_signature": "0"}):
            with self.subTest(tags=tags):
                data = complete(); data["records"][3]["tags"].update(tags)
                self.assertIsNotNone(validate(data).error)
        for value in (0, -1, float("nan"), float("inf"), "1", True):
            data = complete(); data["records"][3]["value"] = value
            self.assertIsNotNone(validate(data).error)

    def test_stale_setup_or_entirely_missing_setup_fails(self):
        data = complete(); data["records"][3]["tags"]["capture_signature"] = "123"
        self.assertIn("stale", validate(data).error)
        data = complete(); data["records"] = [r for r in data["records"] if r["name"] != "attention_arithmetic_policy"]
        self.assertIn("stale", validate(data).error)

    def test_large_to_small_prefill_cannot_reuse_an_unrelated_capture(self):
        data = complete(); data["records"][-1]["tags"]["bucket_seq_len"] = "32"
        self.assertIn("prefill bucket", validate(data).error)

    def test_native_inventory_requires_real_capture_provenance(self):
        for change in ({"type": "capturable"}, {"source": "plan"}, {"attribution": "guess"}, {"context": ""}, {"graph_capture_scope": "plan"}):
            data = complete(); data["records"][-2]["tags"].update(change)
            self.assertIsNotNone(validate(data).error)
        data = complete(); data["records"] = [r for r in data["records"] if r["name"] != "full_graph_capture_stage_types"]
        self.assertIn("stage inventory", validate(data).error)

    def test_native_segmented_child_keeps_its_declared_capture_boundary(self):
        """Heterogeneous admission remains the separate graph validator's job."""
        data = complete()
        row = data["records"][-2]
        row["name"] = "segmented_graph_capture_stage_types"
        row["tags"].update(source="segmented_graph_capture", graph_capture_scope="segmented_capture_plan",
                            type="captured_child_template")
        self.assertIsNone(validate(data).error)

    def test_multirow_mtp_is_decode_by_explicit_phase(self):
        data = complete()
        for record in data["records"]:
            if record["name"] in {"attention_arithmetic_policy", "qsa_attention_program_captures"} and record["tags"]["phase"] == "decode":
                record["tags"]["rows"] = "15"
        self.assertIsNone(validate(data).error)

    def test_shell_observer_requires_sparse_family_on_cpu_cuda_rocm(self):
        """Execute the real embedded consumer while mocking only unrelated gates."""
        script = (SERVER / "test_server_e2e.sh").read_text()
        block = script.split("validate_perf_stats() {", 1)[1].split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        for device in ("CPU", "CUDA:0", "ROCm:0"):
            data = cell(device, dense=True)
            if device != "CPU": data["records"].append(flash(device))
            output = io.StringIO()
            graph_result = SimpleNamespace(error=None, device_kinds=frozenset({device.partition(":")[0].lower()}))
            args = ["validator", "/artifact", "auto", "", "false", "e2e-certification", str(SERVER), "", "", "", "", ""]
            with ExitStack() as stack:
                stack.enter_context(patch("ranked_perf_artifacts.collect_and_publish_ranked_perf_stats", return_value=data))
                stack.enter_context(patch("ranked_perf_artifacts.validate_memory_authority"))
                for target in ("graph_capture_perf_policy.validate_graph_capture_policy",
                               "request_input_lifetime_perf_policy.validate_request_input_lifetime_policy",
                               "gpu_host_transfer_perf_policy.validate_gpu_host_transfer_policy"):
                    stack.enter_context(patch(target, return_value=graph_result))
                stack.enter_context(patch.object(sys, "argv", args))
                stack.enter_context(patch.dict(os.environ, {"LLAMINAR_E2E_MOVEMENT_EVIDENCE": "not_applicable"}))
                stack.enter_context(contextlib.redirect_stdout(output))
                with self.assertRaises(SystemExit):
                    exec(compile(block, "http-attention-observer", "exec"), {})
            self.assertIn("no QSA", output.getvalue())

    def test_opencode_observer_requires_sparse_family_independently(self):
        """App sessions consume exactly the same family observer on every backend."""
        import opencode_cell_runtime as runtime
        configuration = {"runtime": {"generation": {"mtp_policy": "off", "mtp_verify_mode": "greedy"}},
            "e2e": {"planning": {"mode": "auto", "strategy": "single", "device_counts": {"cpu": 1}, "mpi_ranks": 1}}}
        for device in ("CPU", "CUDA:0", "ROCm:0"):
            data = cell(device, dense=True)
            data["records"][-1]["tags"]["prefix_cache"] = "true"
            if device != "CPU": data["records"].append(flash(device))
            with ExitStack() as stack:
                stack.enter_context(patch.object(runtime, "validate_memory_authority"))
                stack.enter_context(patch.object(runtime, "validate_automatic_selection", return_value={}))
                stack.enter_context(patch.object(runtime, "validate_runtime_feature_policy", return_value=None))
                stack.enter_context(patch.object(runtime, "app_archive_compaction_assessment", return_value={"passed": True}))
                for name in ("validate_graph_capture_policy", "validate_gpu_host_transfer_policy"):
                    stack.enter_context(patch.object(runtime, name, return_value=AttentionExecutionValidation(None, 0, 0, 0)))
                result = runtime.runtime_evidence(data, configuration)
            self.assertFalse(result["passed"])
            self.assertIn("no QSA", result["checks"]["attention"]["error"])

    def test_family_contract_rejects_missing_and_inconsistent_obligations(self):
        for mutation in (lambda t: t.pop("sparse_attention_devices"),
                         lambda t: t.update(sparse_attention_devices=""),
                         lambda t: t.update(flash_attention_devices="ROCm:7")):
            data = cell(); mutation(data["records"][0]["tags"])
            with self.assertRaises(ValueError):
                validate(data)


if __name__ == "__main__":
    unittest.main()
