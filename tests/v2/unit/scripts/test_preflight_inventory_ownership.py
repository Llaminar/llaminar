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
MTP_TARGET = "v2_test_mtp_graph_construction"
MTP_NATIVE_CASES = (
    "GPUSidecarGraphCacheMaterializesAndUsesFullGraphFromFirstInvocation",
    "GPUDeviceTargetAndDraftSidecarCachesAreIndependent",
    "GPUShiftedPrefillSidecarPolicyUsesShiftedPrefillContext",
    "GPUStochasticRequestBatchScratchScalesWithConfiguredCapacity",
)


def selected_by_gtest(test: dict, case: str) -> bool:
    """Project Google's include/exclude grammar onto an actual native case name."""
    filters = [arg.partition("=")[2] for arg in test.get("command", [])
               if arg.startswith("--gtest_filter=")]
    include, _, exclude = (filters[0] if len(filters) == 1 else "*").partition("-")
    return (any(fnmatch.fnmatchcase(case, pattern) for pattern in (include or "*").split(":"))
            and not any(fnmatch.fnmatchcase(case, pattern) for pattern in exclude.split(":")))


BUILD: Path
BACKENDS: list[str]


def copy_inventory_row(document: dict, predicate) -> tuple[dict, dict]:
    """Isolate one adversarial row while retaining every original inventory entry.

    Only the selected row is mutable in these tests. Sharing all other rows
    avoids repeatedly cloning the complete generated command/property payload.
    The top-level mapping and tests list are independent, so removal is safe.
    """
    tests = list(document["tests"])
    index = next(index for index, test in enumerate(tests) if predicate(test))
    tests[index] = copy.deepcopy(tests[index])
    return {**document, "tests": tests}, tests[index]


def validate_inventory(document: dict, backends: list[str]) -> list[str]:
    """Require each actual fixture registration to claim its process-wide resources."""
    selected_targets = frozenset().union(*(TARGETS[backend] for backend in ("common", *backends)))
    matched, owners, mtp_native_backends = [], set(), set()
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
        mtp_native = MTP_TARGET in executables and any(
            selected_by_gtest(test, f"{backend}/MTPNativeGraphConstruction.{case}/0")
            for backend in ("CUDA", "ROCm") for case in MTP_NATIVE_CASES)
        if mtp_native:
            if "Unit" in labels or "DeviceFree" in labels:
                raise ValueError(f"{test['name']} admits a native GPU graph case into Unit/device-free scope")
            backend = next((name for name in ("CUDA", "ROCm")
                            if test["name"] == f"V2_Integration_MTPNativeGraphConstruction_{name}"), None)
            expected_filter = f"--gtest_filter={backend}/MTPNativeGraphConstruction.*"
            filters = [arg for arg in test.get("command", []) if arg.startswith("--gtest_filter=")]
            if (backend is None or filters != [expected_filter] or
                    not {"GPU", backend, "Integration", "ProductionTestPreflight"} <= labels or
                    ({"CUDA", "ROCm"} & labels) != {backend}):
                raise ValueError(f"{test['name']} omits exact native MTP backend ownership")
            environment = properties.get("ENVIRONMENT", [])
            expected = {"LLAMINAR_FORCE_CPU_ONLY_STARTUP=0",
                        f"LLAMINAR_SKIP_CUDA_STARTUP={0 if backend == 'CUDA' else 1}",
                        f"LLAMINAR_SKIP_ROCM_STARTUP={0 if backend == 'ROCm' else 1}"}
            actual = [entry for entry in environment if entry.partition("=")[0] in
                      {item.partition("=")[0] for item in expected}]
            if set(actual) != expected or len(actual) != len(expected):
                raise ValueError(f"{test['name']} omits exact native MTP startup scope")
            _, native_lane = _preflight_execution_lane(test["name"], labels, test["command"],
                                                        properties.get("RESOURCE_LOCK", []))
            expected_lane = (PreflightExecutionLane.EXCLUSIVE if "FullDeviceInventory" in labels
                             else PreflightExecutionLane.CUDA_SOCKET if backend == "CUDA"
                             else PreflightExecutionLane.ROCM_SOCKET)
            if native_lane is not expected_lane:
                raise ValueError(f"{test['name']} omits exact native MTP execution scope")
            mtp_native_backends.add(backend.lower())
            matched.append(test["name"])
        if executables & DEVICE_FREE_TARGETS and not mtp_native:
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
    if owners and not set(backends) <= mtp_native_backends:
        raise ValueError(f"Missing native MTP backend registrations: {sorted(set(backends) - mtp_native_backends)}")
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
                    document, selected = copy_inventory_row(self.document, lambda test: target in
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
                document, selected = copy_inventory_row(self.document,
                    lambda test: test["name"] == "V2_Unit_DeviceRegistry")
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

    def test_mtp_native_selection_cannot_leak_into_unit(self) -> None:
        """Every native MTP body stays excluded from the real device-free Unit command."""
        for case in MTP_NATIVE_CASES:
            for backend in ("CUDA", "ROCm"):
                document, unit = copy_inventory_row(self.document,
                    lambda test: test["name"] == "V2_Unit_MTPGraphConstruction")
                unit["command"] = [arg for arg in unit["command"] if not arg.startswith("--gtest_filter=")]
                unit["command"].append(f"--gtest_filter={backend}/MTPNativeGraphConstruction.{case}/0")
                with self.assertRaisesRegex(ValueError, "native GPU graph case into Unit"):
                    validate_inventory(document, BACKENDS)

    def test_mtp_native_backend_scope_and_presence_are_required(self) -> None:
        """A hidden backend, broad filter, CPU guard or missing native cell fails closed."""
        for backend in BACKENDS:
            native_name = f"V2_Integration_MTPNativeGraphConstruction_{'CUDA' if backend == 'cuda' else 'ROCm'}"
            for mutation in ("absent", "labels", "filter", "guard"):
                with self.subTest(backend=backend, mutation=mutation):
                    document, native = copy_inventory_row(self.document,
                        lambda test: test["name"] == native_name)
                    if mutation == "absent":
                        document["tests"].remove(native)
                    elif mutation == "filter":
                        native["command"] = [arg for arg in native["command"] if not arg.startswith("--gtest_filter=")]
                    else:
                        for prop in native["properties"]:
                            if mutation == "labels" and prop["name"] == "LABELS":
                                prop["value"] = [label for label in prop["value"] if label not in {"CUDA", "ROCm"}]
                            if mutation == "guard" and prop["name"] == "ENVIRONMENT":
                                prop["value"] = [entry for entry in prop["value"] if not entry.startswith("LLAMINAR_FORCE_CPU_ONLY_STARTUP=")] + ["LLAMINAR_FORCE_CPU_ONLY_STARTUP=1"]
                    with self.assertRaisesRegex(ValueError, "native MTP"):
                        validate_inventory(document, BACKENDS)

    def test_native_graph_case_is_excluded_from_unit_and_retained_in_preflight(self) -> None:
        """A mixed fixture must keep native GPU work in its explicit accelerator lane."""
        document, unit = copy_inventory_row(self.document,
            lambda test: test["name"] == "V2_Unit_Qwen35MoEGraph")
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
