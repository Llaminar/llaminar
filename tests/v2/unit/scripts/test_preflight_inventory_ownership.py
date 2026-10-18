#!/usr/bin/env python3
"""Prove real CTest registrations retain their complete startup ownership.

MPI cluster discovery and CPU-reference setup can query both vendors' physical
memory before a backend-specific test starts. Conversely, CPU graph-construction
fixtures must inherit the same startup guard in Unit and focused preflight.
These checks inspect generated CTest commands and the production lane classifier
without executing a GPU command or substituting source-text assertions.
"""
from __future__ import annotations

import argparse
import copy
import fnmatch
import json
from pathlib import Path
import subprocess
import sys
import unittest


SOURCE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SOURCE / "scripts/ci"))
from run_production_parity_campaigns import PreflightExecutionLane, _preflight_execution_lane


TARGETS = {
    "common": frozenset(("v2_integration_moe_router_exchange", "v2_perf_moe_router_exchange",
                         "v2_integration_moe_projection_pipeline", "v2_perf_moe_projection_pipeline",
                         "v2_integration_transfer_engine_copy_activation")),
    "cuda": frozenset(("v2_integration_cuda_gemm_parity", "v2_integration_cuda_gemm_batch_invariance",
                       "v2_integration_cuda_flash_attention_parity", "v2_integration_cuda_gdn_padded_real_length",
                       "v2_integration_cuda_gdn_projection_capture")),
    "rocm": frozenset(("v2_integration_rocm_gdn_projection_capture",)),
}
DEVICE_FREE_TARGETS = frozenset(("v2_test_mtp_graph_construction", "v2_test_device_graph_orchestrator",
                                 "v2_test_device_registry", "v2_test_backend_startup_config",
                                 "v2_test_moe_runtime_table"))
NATIVE_GRAPH_CASE = "Test__Qwen35MoEGraph.ROCmPhaseSplitDecodeUsesCanonicalRouteSlotAllreduce"
BUILD: Path
BACKENDS: list[str]


def validate_inventory(document: dict, backends: list[str]) -> list[str]:
    """Require each actual fixture registration to claim its process-wide resources."""
    selected_targets = frozenset().union(*(TARGETS[backend] for backend in ("common", *backends)))
    matched, owners = [], set()
    for test in document["tests"]:
        executables = {Path(argument).name for argument in test.get("command", [])}
        targets = executables & selected_targets
        owners.update(targets)
        properties = {item["name"]: item["value"] for item in test["properties"]}
        labels = frozenset(properties.get("LABELS", []))
        if "v2_test_qwen35moe_graph" in executables and "Unit" in labels:
            filters = [argument.partition("=")[2] for argument in test.get("command", [])
                       if argument.startswith("--gtest_filter=")]
            include, _, exclude = (filters[0] if len(filters) == 1 else "*").partition("-")
            selected = any(fnmatch.fnmatchcase(NATIVE_GRAPH_CASE, pattern)
                           for pattern in (include or "*").split(":"))
            excluded = any(fnmatch.fnmatchcase(NATIVE_GRAPH_CASE, pattern)
                           for pattern in exclude.split(":"))
            if selected and not excluded:
                raise ValueError(f"{test['name']} admits a native GPU graph case into Unit")
        if executables & DEVICE_FREE_TARGETS:
            owners.update(executables & DEVICE_FREE_TARGETS)
            if not labels & {"Unit", "DeviceFree"}:
                raise ValueError(f"{test['name']} omits device-free ownership")
            guards = [entry for entry in properties.get("ENVIRONMENT", [])
                      if entry.startswith("LLAMINAR_FORCE_CPU_ONLY_STARTUP=")]
            if not guards or set(guards) != {"LLAMINAR_FORCE_CPU_ONLY_STARTUP=1"}:
                raise ValueError(f"{test['name']} omits an unambiguous CPU-only startup guard")
            matched.append(test["name"])
        if not targets:
            continue
        if "FullDeviceInventory" not in labels:
            raise ValueError(f"{test['name']} omits full inventory ownership")
        _, lane = _preflight_execution_lane(test["name"], labels, test["command"],
                                            properties.get("RESOURCE_LOCK", []))
        if lane is not PreflightExecutionLane.EXCLUSIVE:
            raise ValueError(f"{test['name']} can overlap an unselected GPU vendor")
        matched.append(test["name"])
    required = {target for target in selected_targets if not target.startswith("v2_perf_")} | DEVICE_FREE_TARGETS
    if not required <= owners:
        raise ValueError(f"Missing real fixture registrations: {sorted(required - owners)}")
    return matched


class PreflightInventoryOwnershipTests(unittest.TestCase):
    """Exercise configured registrations and adversarial ownership omissions."""

    @classmethod
    def setUpClass(cls) -> None:
        """Read CTest's actual inventory without executing any registered command."""
        cls.document = json.loads(subprocess.check_output(
            ["ctest", "--test-dir", str(BUILD), "--show-only=json-v1"], text=True, timeout=10))

    def test_every_filter_retains_complete_process_ownership(self) -> None:
        """CUDA, ROCm and performance filters inherit their executable's scope."""
        self.assertTrue(validate_inventory(self.document, BACKENDS))

    def test_missing_scope_cannot_be_hidden_by_backend_labels(self) -> None:
        """Removing the scope from one real registration must reject its whole inventory."""
        for backend in ("common", *BACKENDS):
            for target in TARGETS[backend]:
                if target.startswith("v2_perf_"):
                    continue
                with self.subTest(target=target):
                    document = copy.deepcopy(self.document)
                    selected = next(test for test in document["tests"] if target in
                                    {Path(argument).name for argument in test.get("command", [])})
                    for property in selected["properties"]:
                        if property["name"] == "LABELS":
                            property["value"] = [label for label in property["value"]
                                                 if label != "FullDeviceInventory"]
                    with self.assertRaisesRegex(ValueError, "omits full inventory ownership"):
                        validate_inventory(document, BACKENDS)

    def test_missing_or_conflicting_cpu_guard_rejects_device_free_fixture(self) -> None:
        """A CPU-only label without enforced startup isolation must not certify a run."""
        for replacement in ([], ["LLAMINAR_FORCE_CPU_ONLY_STARTUP=0"],
                            ["LLAMINAR_FORCE_CPU_ONLY_STARTUP=1", "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0"]):
            with self.subTest(replacement=replacement):
                document = copy.deepcopy(self.document)
                selected = next(test for test in document["tests"]
                                if test["name"] == "V2_Unit_DeviceRegistry")
                for property in selected["properties"]:
                    if property["name"] == "ENVIRONMENT":
                        property["value"] = [entry for entry in property["value"]
                            if not entry.startswith("LLAMINAR_FORCE_CPU_ONLY_STARTUP=")] + replacement
                with self.assertRaisesRegex(ValueError, "CPU-only startup guard"):
                    validate_inventory(document, BACKENDS)

    def test_absent_fixture_cannot_produce_an_empty_green_check(self) -> None:
        """An obsolete target selector must fail instead of certifying zero work."""
        with self.assertRaisesRegex(ValueError, "Missing real fixture registrations"):
            validate_inventory({"tests": []}, BACKENDS)

    def test_native_graph_case_is_excluded_from_unit_and_retained_in_preflight(self) -> None:
        """A mixed fixture must keep native GPU work in its explicit accelerator lane."""
        document = copy.deepcopy(self.document)
        unit = next(test for test in document["tests"] if test["name"] == "V2_Unit_Qwen35MoEGraph")
        unit["command"] = [argument for argument in unit["command"]
                           if not argument.startswith("--gtest_filter=")]
        with self.assertRaisesRegex(ValueError, "native GPU graph case into Unit"):
            validate_inventory(document, BACKENDS)
        if "rocm" in BACKENDS:
            native = next(test for test in self.document["tests"]
                          if test["name"] == "V2_Integration_ROCmCanonicalRouteSlotAllreduce")
            properties = {item["name"]: item["value"] for item in native["properties"]}
            self.assertIn("--gtest_filter=" + NATIVE_GRAPH_CASE, native["command"])
            self.assertTrue({"ROCm", "ProductionTestPreflight"} <= set(properties["LABELS"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("cuda", "rocm"), action="append", required=True)
    args, remaining = parser.parse_known_args()
    BUILD = args.build_dir.resolve(strict=True)
    BACKENDS = args.backend
    unittest.main(argv=[sys.argv[0], *remaining])
