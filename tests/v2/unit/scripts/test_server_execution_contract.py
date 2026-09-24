#!/usr/bin/env python3
"""Device-free regressions for auto/saved-plan server certification.

The immutable startup projection is the only topology/policy input to the
observers. These tests also execute the actual shell-embedded validator so a
future CLI-only shortcut cannot silently drop MTP, prefix or movement checks.
No model, MPI process, driver or cloud resource is used.
"""
from __future__ import annotations

import contextlib
import copy
from dataclasses import asdict
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[2] / "e2e/server"
sys.path.insert(0, str(SERVER_DIR))
from server_execution_contract import (RuntimeFeaturePolicy, validate_server_execution_contract,
                                       automatic_selection_policy, validate_automatic_selection)
from graph_capture_perf_policy import validate_graph_capture_policy
from flash_attention_perf_policy import validate_flash_attention_plan_policy
import test_server_graph_capture_perf_policy as graph_fixtures


def evidence(devices: list[tuple[str, str]], *, authority: int = 0,
             features: RuntimeFeaturePolicy = RuntimeFeaturePolicy()) -> dict:
    """Build exact startup observations with explicit per-rank attention roles."""
    rows = []
    for rank, (selected, attention) in enumerate(devices):
        rows.append({"domain": "server", "name": "execution_topology", "rank": rank,
                     "value": 1.0, "phase": "startup", "tags": {
                         "schema": "1", "source": "resolved_execution_plan",
                         "devices": selected, "attention_devices": attention}})
    policy = {key: str(value).lower() if isinstance(value, bool) else str(value)
              for key, value in asdict(features).items()}
    rows.append({"domain": "server", "name": "execution_policy", "rank": authority,
                 "value": 1.0, "phase": "startup", "tags": {
                     "schema": "1", "source": "resolved_execution_plan", **policy}})
    return {"world_size": len(devices), "authority_rank": authority, "records": rows}


def automatic_evidence(devices: list[str], *, strategy: str, nodes: list[int] | None = None) -> dict:
    """Add physical execution witnesses; visible GPU ordinals share UUIDs on one node."""
    data = evidence([(devices, devices) for devices in devices])
    data["records"][-1]["tags"]["execution_strategy"] = strategy
    for rank, selected in enumerate(devices):
        node = str(nodes[rank] if nodes is not None else 0)
        data["records"].append({"domain": "server", "name": "rank_membership", "rank": rank,
            "value": 1, "phase": "startup", "tags": {"rank": str(rank), "node_id": node,
                "world_size": str(len(devices)), "authority_rank": "0",
                "identity_source": "communicator_cluster_inventory"}})
        for device in selected.split(",") if selected else []:
            backend = device.partition(":")[0]
            data["records"].append({"domain": "server", "name": "execution_participant", "rank": rank,
                "value": 1, "phase": "startup", "device": device, "tags": {
                    "schema": "1", "source": "resolved_execution_plan",
                    "identity_source": "communicator_cluster_inventory", "node_id": node,
                    "backend": backend, "physical_id": f"numa:{rank}" if backend == "CPU" else f"uuid-{device}"}})
    return data


class TestPipelineDomainEvidence(unittest.TestCase):
    """A GPU count is not evidence of the requested TP-over-PP composition."""

    @staticmethod
    def selection():
        """Expected domain shapes do not prescribe ordering, ordinals or splits."""
        return {"mode": "auto", "strategy": "pp", "mpi_ranks": 1,
                "device_counts": {"cuda": 2, "rocm": 2},
                "pipeline_layers": 64,
                "pipeline_domains": [{"backend": "cuda", "devices": 2}, {"backend": "rocm", "devices": 2}]}

    @staticmethod
    def observed(groups=("CUDA:1,CUDA:0", "ROCm:3,ROCm:2"), split=17):
        """Construct observation from concrete stage membership, not its name."""
        data = automatic_evidence([",".join(groups)], strategy="pp")
        for index, group in enumerate(groups):
            data["records"].append({"domain": "server", "name": "execution_pipeline_domain", "rank": 0,
                "value": 1, "phase": "startup", "tags": {"schema": "1", "source": "resolved_execution_plan",
                    "scope": "rank_local", "stage": str(index), "stages": str(len(groups)),
                    "first_layer": str(0 if index == 0 else split),
                    "end_layer": str(split if index == 0 else 64), "devices": group}})
        return data

    def test_accepts_either_vendor_order_and_nonuniform_layer_splits(self):
        for groups in (("CUDA:1,CUDA:0", "ROCm:3,ROCm:2"), ("ROCm:3,ROCm:2", "CUDA:1,CUDA:0")):
            for split in (1, 17, 63):
                with self.subTest(groups=groups, split=split):
                    observed = validate_automatic_selection(self.observed(groups, split), self.selection())
                    self.assertEqual(observed["pipeline_layers"], 64)
                    self.assertCountEqual(observed["pipeline_domains"], self.selection()["pipeline_domains"])

    def test_recurrent_only_domain_retains_compute_and_capture_obligations(self):
        """A one-layer GDN domain is compute, not missing FlashAttention work."""
        for groups in (("CUDA:1,CUDA:0", "ROCm:3,ROCm:2"), ("ROCm:3,ROCm:2", "CUDA:1,CUDA:0")):
            with self.subTest(groups=groups):
                data = self.observed(groups, split=1)
                data["records"][0]["tags"]["attention_devices"] = groups[1]
                contract = validate_server_execution_contract(data)
                self.assertEqual(contract.device_kinds, {"cuda", "rocm"})
                backend = groups[1].partition(":")[0].lower()
                self.assertEqual(contract.attention_device_kinds, {backend})
                observed = validate_automatic_selection(data, self.selection())
                self.assertEqual(observed["device_counts"], {"cuda": 2, "rocm": 2})
                missing = validate_flash_attention_plan_policy([], contract.attention_device_kinds)
                self.assertEqual(missing.expected_backends, {backend})
                self.assertIsNotNone(missing.error)  # The actual FA owner is still mandatory.
                self.assertIsNotNone(validate_graph_capture_policy([], contract.device_kinds).error)

    def test_missing_false_and_partial_domain_proof_fails(self):
        mutations = (
            lambda d: d["records"].pop(),
            lambda d: d["records"].append(copy.deepcopy(d["records"][-1])),
            lambda d: d["records"][-1]["tags"].update(devices="ROCm:3"),
            lambda d: d["records"][-1]["tags"].update(devices="CUDA:0,ROCm:3"),
            lambda d: d["records"][-1]["tags"].update(devices="ROCm:7,ROCm:8"),
            lambda d: d["records"][-1]["tags"].update(devices="CUDA:1,CUDA:0"),
            lambda d: d["records"][-1]["tags"].update(first_layer="18"),
            lambda d: d["records"][-1]["tags"].update(first_layer="16"),
            lambda d: d["records"][-1]["tags"].update(end_layer="63"),
            lambda d: d["records"][-1]["tags"].update(stage="2"),
            lambda d: d["records"][-1]["tags"].update(stages="3"),
            lambda d: d["records"][-1]["tags"].update(source="cli"),
            lambda d: d["records"][-1].update(rank=True),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                data = self.observed()
                mutate(data)
                validate_automatic_selection(data, self.selection())

    def test_four_singletons_do_not_prove_two_tensor_parallel_domains(self):
        data = self.observed(("CUDA:1", "CUDA:0", "ROCm:3", "ROCm:2"))
        with self.assertRaises(ValueError):
            validate_automatic_selection(data, self.selection())

    def test_contract_rejects_incomplete_and_contradictory_shapes(self):
        selection = self.selection()
        for change in ({"mpi_ranks": 0}, {"mpi_ranks": True}, {"pipeline_layers": 0},
                       {"pipeline_layers": True}, {"pipeline_domains": []},
                       {"pipeline_domains": [{"backend": "cuda", "devices": 1}]},
                       {"pipeline_domains": [{"backend": "cuda", "devices": 2}, {"backend": "rocm", "devices": True}]},
                       {"strategy": "tp"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                automatic_selection_policy({"planning": {**selection, **change}})


class TestServerExecutionContract(unittest.TestCase):
    """Authenticate roles without conflating discoverable devices and execution."""

    def test_all_backends_and_reversed_nonzero_authority(self):
        for kind, label in (("cpu", "CPU"), ("cuda", "CUDA:2"), ("rocm", "ROCm:3")):
            with self.subTest(kind=kind):
                contract = validate_server_execution_contract(evidence([(label, label)]))
                self.assertEqual(contract.device_kinds, {kind})
                self.assertEqual(contract.attention_device_kinds, {kind})
                self.assertEqual(contract.uses_gpu, kind != "cpu")
        contract = validate_server_execution_contract(evidence(
            [("CPU", ""), ("ROCm:1,ROCm:2", "ROCm:1,ROCm:2"), ("", "")], authority=1))
        self.assertEqual(contract.device_kinds, {"cpu", "rocm"})
        self.assertEqual(contract.attention_device_kinds, {"rocm"})

    def test_automatic_selection_requires_real_backend_counts_and_execution_family(self):
        for devices, strategy, counts in (
                (["CPU", "CPU"], "tp", {"cpu": 2}),
                (["CUDA:0,CUDA:1"], "tp", {"cuda": 2}),
                (["ROCm:2,ROCm:3"], "pp", {"rocm": 2}),
                (["CUDA:0,CUDA:1,ROCm:2,ROCm:3"], "pp", {"cuda": 2, "rocm": 2})):
            with self.subTest(devices=devices, strategy=strategy):
                selection = {"mode": "auto", "strategy": strategy,
                             "device_counts": counts, "mpi_ranks": len(devices)}
                data = automatic_evidence(devices, strategy=strategy)
                self.assertEqual(validate_automatic_selection(data, selection),
                                 {"strategy": strategy, "device_counts": counts,
                                  "mpi_ranks": len(devices)})
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    validate_automatic_selection(data, {**selection, "device_counts": {"cuda": 1}})
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    validate_automatic_selection(data, {**selection, "strategy": "single"})

    def test_automatic_selection_deduplicates_visibility_not_distinct_physical_hosts(self):
        selection = {"mode": "auto", "strategy": "tp", "device_counts": {"cuda": 2},
                     "mpi_ranks": 2}
        with self.assertRaisesRegex(ValueError, "mismatch"):
            validate_automatic_selection(automatic_evidence(["CUDA:0", "CUDA:0"], strategy="tp"), selection)
        observed = validate_automatic_selection(
            automatic_evidence(["CUDA:0", "CUDA:0"], strategy="tp", nodes=[0, 1]), selection)
        self.assertEqual(observed["device_counts"], {"cuda": 2})

    def test_two_rank_cell_rejects_single_rank_with_same_physical_device_count(self):
        selection = {"mode": "auto", "strategy": "tp", "device_counts": {"cuda": 2},
                     "mpi_ranks": 2}
        with self.assertRaisesRegex(ValueError, "MPI rank mismatch"):
            validate_automatic_selection(
                automatic_evidence(["CUDA:0,CUDA:1"], strategy="tp"), selection)
        observed = validate_automatic_selection(
            automatic_evidence(["CUDA:0", "CUDA:1"], strategy="tp"), selection)
        self.assertEqual(observed["mpi_ranks"], 2)

    def test_automatic_selection_rejects_idle_missing_duplicate_and_forged_participants(self):
        original = automatic_evidence(["CUDA:0,CUDA:1"], strategy="tp")
        selection = {"mode": "auto", "strategy": "tp", "device_counts": {"cuda": 2},
                     "mpi_ranks": 1}
        for mutate in (
                lambda d: d["records"].pop(),
                lambda d: d["records"].append(copy.deepcopy(d["records"][-1])),
                lambda d: d["records"][-1].update(device="CUDA:7"),
                lambda d: d["records"][-1]["tags"].update(physical_id=""),
                lambda d: d["records"][-1]["tags"].update(node_id="1"),
                lambda d: d["records"][-1]["tags"].update(backend="ROCm"),
                lambda d: d["records"][-1]["tags"].update(identity_source="measured_latency")):
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                data = copy.deepcopy(original)
                mutate(data)
                validate_automatic_selection(data, selection)
        for planning in (None, {}, {**selection, "mode": "apply"}, {**selection, "strategy": "future"},
                         {key: value for key, value in selection.items() if key != "mpi_ranks"},
                         {**selection, "device_counts": {}}, {**selection, "device_counts": {"cuda": True}},
                         {**selection, "device_counts": {"cuda": 0}}, {**selection, "device_counts": {"cuda": "2"}}):
            with self.subTest(planning=planning), self.assertRaises(ValueError):
                automatic_selection_policy({"planning": planning})

    def test_missing_duplicate_foreign_and_future_records_fail(self):
        mutations = [
            lambda d: d["records"].pop(0),
            lambda d: d["records"].append(copy.deepcopy(d["records"][0])),
            lambda d: d["records"].append(copy.deepcopy(d["records"][-1])),
            lambda d: d["records"][-1].update(rank=1),
            lambda d: d["records"][0].update(rank=9),
            lambda d: d["records"][0].update(phase="decode"),
            lambda d: d["records"][0]["tags"].update(schema="2"),
            lambda d: d["records"][0]["tags"].update(source="cli"),
            lambda d: d["records"][0]["tags"].update(devices="CUDA:0,CUDA:0"),
            lambda d: d["records"][0]["tags"].update(devices="auto"),
            lambda d: d["records"][0]["tags"].update(attention_devices="ROCm:0"),
            lambda d: d["records"][-1]["tags"].pop("mtp"),
            lambda d: d["records"][-1]["tags"].update(mtp="maybe"),
            lambda d: d["records"][-1]["tags"].update(mtp_depth_policy="future"),
            lambda d: d["records"][-1]["tags"].update(mtp_min_depth="4", mtp_max_depth="2"),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(ValueError):
                data = evidence([("CUDA:0", "CUDA:0"), ("CPU", "")])
                mutate(data)
                validate_server_execution_contract(data)

    def test_runtime_topology_does_not_waive_graph_boundary_proof(self):
        for device in ("cuda:0", "rocm:0"):
            rows = graph_fixtures.TestServerGraphCapturePerfPolicy.local_ticket_boundary_records(device)
            label = "CUDA:0" if device.startswith("cuda") else "ROCm:0"
            mixed = validate_server_execution_contract(evidence([(label, label), ("CPU", "")]))
            self.assertIsNone(validate_graph_capture_policy(rows, mixed.device_kinds).error)
            homogeneous = validate_server_execution_contract(evidence([(label, label)]))
            self.assertIn("homogeneous", validate_graph_capture_policy(rows, homogeneous.device_kinds).error)
            self.assertIsNotNone(validate_graph_capture_policy(rows[:1], mixed.device_kinds).error)

    def test_expert_tiers_do_not_acquire_attention_obligations(self):
        for model, expert in (("CUDA:0", "ROCm:0"), ("ROCm:0", "CUDA:0"), ("CPU", "CUDA:0")):
            contract = validate_server_execution_contract(evidence([(model, model), (expert, "")]))
            check = validate_flash_attention_plan_policy([], contract.attention_device_kinds)
            expected = model.partition(":")[0].lower()
            self.assertEqual(check.expected_backends, set() if expected == "cpu" else {expected})

    def test_actual_harness_keeps_features_with_saved_config_and_auto(self):
        """Only external artifact collection is mocked; execute the real observer."""
        script = (SERVER_DIR / "test_server_e2e.sh").read_text()
        block = script.split("validate_perf_stats() {", 1)[1].split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        cases = [
            (RuntimeFeaturePolicy(mtp=True), "not_applicable", "attempted and accepted"),
            (RuntimeFeaturePolicy(prefix_cache=True), "not_applicable", "harvested prefix"),
            (RuntimeFeaturePolicy(residency_maintenance="dynamic"), "required", "committed physical"),
        ]
        for flags in ("--config /saved/plan.json", "--auto", ""):
            for features, movement, error in cases:
                with self.subTest(flags=flags, error=error):
                    data = evidence([("CPU", "CPU")], features=features)
                    # The MTP domain-presence guard must pass to reach the
                    # stronger attempted/accepted check; presence is not work.
                    data["records"].append({"domain": "mtp", "name": "placeholder", "value": 1})
                    args = ["validator", "/artifact", "auto", flags, "false", "e2e-certification",
                            str(SERVER_DIR), "", "", "", "", ""]
                    output = io.StringIO()
                    with patch("ranked_perf_artifacts.collect_and_publish_ranked_perf_stats", return_value=data), \
                         patch("ranked_perf_artifacts.validate_memory_authority"), \
                         patch.object(sys, "argv", args), \
                         patch.dict(os.environ, {"LLAMINAR_E2E_MOVEMENT_EVIDENCE": movement}), \
                         contextlib.redirect_stdout(output), self.assertRaises(SystemExit):
                        exec(compile(block, "server-evidence-validator", "exec"), {})
                    self.assertIn(error, output.getvalue())

    def test_prefix_workload_runs_without_explicit_cli_flags(self):
        """Execute the harness's request selection with saved/default intent."""
        import subprocess
        script = (SERVER_DIR / "test_server_e2e.sh").read_text()
        block = script.split("    # ─── Prefix Cache Probe", 1)[1]
        block = block[block.index("\n"):].split(
            "    if suite_runs_prefix_cache_rebalance_clear_probe", 1)[0]
        for flags in ("", "--auto", "--config /saved/plan.json", "--prefix-cache"):
            with self.subTest(flags=flags):
                command = ('set -euo pipefail\ntag=t port=1 max_tokens=32 thinking_model=false\n'
                           'extra_flags="$1"\nrun_prefix_cache_checks() { echo PREFIX_PROBE; }\n' + block)
                result = subprocess.run(["bash", "-c", command, "probe", flags],
                                        capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), "PREFIX_PROBE")


if __name__ == "__main__":
    unittest.main()
