#!/usr/bin/env python3
## @file test_develop_image_gate.py
#  @brief Exercise PR and image publication commands, hardware scope and failure evidence.
"""Device-free contracts for model-free develop PR and image-publication gates.

The PR workflow consumes the same image driver in independent AVX512 and AVX2
jobs without publication. Its actual matrix command is parsed and exercised
here so ISA coverage, model exclusion and read-only cache access stay explicit.
"""
from __future__ import annotations

import json
from dataclasses import replace
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/ci"))

import run_develop_image_gate as gate
import run_develop_pr_tests as pr_runner


def images(isa: str) -> dict:
    """Return just the immutable IDs needed by the outer develop transaction."""
    return {"test-runner": {"id": f"test-runner-{isa.lower()}"},
            "runtime": {"id": f"runtime-{isa.lower()}"}}


def pr_report(scope: pr_runner.PRTestScope = pr_runner.PRTestScope.HOST) -> dict:
    """Build a scoped PR receipt; full accelerator preflight remains unproven."""
    report = {"schema": 1, "scope": scope.value, "complete": True, "return_code": 0,
            "lanes": {lane: {"complete": True, "return_code": 0,
                              "tests": [f"V2_{lane}_Contract"], "test_count": 1}
                      for lane in scope.lanes}}
    if scope is pr_runner.PRTestScope.GPU:
        registered = ["V2_host_Contract", *(name for lane in report["lanes"].values() for name in lane["tests"])]
        report["qualification"] = {"schema": 1, "registered": registered, "selected": registered,
                                   "requirements": {}, "hardware": None, "unqualified": {}}
    return report


def hardware(cuda: int = 1, rocm: int = 1) -> dict:
    """Observe physical cards with valid unknown PCI locality on a one-node host."""
    return {"qfn_source_hardware_schema": 2, "scope": "local-host-all-ranks",
            "devices": [{"backend": "cpu", "ordinal": 0, "numa_node": 0, "socket_id": 0},
                        *({"backend": backend, "ordinal": index, "numa_node": -1,
                           "uuid": f"{backend}-{index}", "native_isa": "gfx906:sramecc+:xnack-"}
                          for backend, count in (("cuda", cuda), ("rocm", rocm))
                          for index in range(count))], "directed_peer_access": []}


class DevelopImageGateTests(unittest.TestCase):
    """Keep develop publishing smaller than, and incapable of replacing, certification."""

    def test_hardware_preflight_covers_every_supported_lane(self):
        """Replay physical cardinality without excluding mixed or multi-rank work."""
        rows = tuple(pr_runner.PreflightRegistration(
            name=f"V2_Integration_{vendor}",
            labels=frozenset({"Integration", "ProductionTestPreflight", vendor}),
            mpi_ranks=1, lane=lane, ctest_index=index)
            for index, (vendor, lane) in enumerate((
                ("CUDA", pr_runner.PreflightExecutionLane.CUDA_SOCKET),
                ("ROCm", pr_runner.PreflightExecutionLane.ROCM_SOCKET)), start=1))
        rows += (replace(rows[0], name="V2_Integration_MultiRank", mpi_ranks=2,
                         lane=pr_runner.PreflightExecutionLane.EXCLUSIVE, ctest_index=3),
                 replace(rows[0], name="V2_Integration_Mixed", mpi_ranks=2,
                         labels=rows[0].labels | {"ROCm"},
                         lane=pr_runner.PreflightExecutionLane.EXCLUSIVE, ctest_index=4),
                 replace(rows[0], name="V2_Integration_TwoCUDA", ctest_index=5,
                         labels=rows[0].labels | {"HardwareOptional", "LocalHostDevices", "Devices2"}))
        expected = {"cuda": (rows[0].name,), "rocm": (rows[1].name,),
                    "exclusive": (rows[3].name, rows[2].name)}
        for _ in range(3):
            plan, qualification = pr_runner.hardware_preflight_plan(rows, hardware())
            self.assertEqual(plan, expected)
            self.assertEqual(set(qualification["unqualified"]), {rows[-1].name})
            self.assertEqual(len(qualification["registered"]), len(rows))
        plan, qualification = pr_runner.hardware_preflight_plan(rows, hardware(cuda=2))
        self.assertIn(rows[-1].name, plan["cuda"])
        self.assertFalse(qualification["unqualified"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hardware-skips.xml"
            pr_runner.write_hardware_skips(path, qualification)
            self.assertEqual(ET.parse(path).getroot().get("skipped"), "0")
        for incomplete in (hardware(cuda=0), hardware(rocm=0)):
            with self.assertRaisesRegex(ValueError, "both CUDA and ROCm"):
                pr_runner.hardware_preflight_plan(rows, incomplete)

    def test_unknown_gpu_locality_preserves_count_tests_and_rejects_topology_claims(self):
        """Unknown PCI locality is valid evidence, never an invented NUMA cohort."""
        from run_qfn_source_admission import observed_native_isa_cohorts
        observed = pr_runner.source_hardware_devices(hardware())
        self.assertEqual(observed_native_isa_cohorts(observed, "rocm", "gfx906"), {})
        base = pr_runner.PreflightRegistration("V2_Integration_Local", frozenset({
            "Integration", "ProductionTestPreflight", "HardwareOptional", "CUDA", "ROCm",
            "Devices2", "CUDADevices1", "ROCmDevices1", "LocalHostDevices"}),
            2, pr_runner.PreflightExecutionLane.EXCLUSIVE, 1)
        strict = replace(base, name="V2_Integration_Strict", ctest_index=2,
                         labels=(base.labels - {"LocalHostDevices"}) | {"DistinctNUMARanks"})
        plan, qualification = pr_runner.hardware_preflight_plan((base, strict), hardware())
        self.assertEqual(plan["exclusive"], (base.name,))
        self.assertEqual(set(qualification["unqualified"]), {strict.name})
        for constraint in ("NativeISA-gfx906", "StageSizes1x1", "DirectedPeerEdge", "DistinctNUMARanks"):
            with self.subTest(constraint=constraint), self.assertRaises(RuntimeError):
                pr_runner.hardware_preflight_plan((replace(base, labels=base.labels | {constraint}),), hardware())
        aliased = hardware(cuda=2)
        aliased["devices"][2]["uuid"] = aliased["devices"][1]["uuid"]
        with self.assertRaisesRegex(ValueError, "unique native GPU"):
            pr_runner.hardware_preflight_plan((base,), aliased)
        # A valid missing NUMA observation must not relax physical identity checks.
        for mutation in ("missing-uuid", "duplicate-uuid", "invalid-numa"):
            malformed = hardware(rocm=2)
            if mutation == "missing-uuid":
                malformed["devices"][-1].pop("uuid")
            elif mutation == "duplicate-uuid":
                malformed["devices"][-1]["uuid"] = malformed["devices"][-2]["uuid"]
            else:
                malformed["devices"][-1]["numa_node"] = -2
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                observed_native_isa_cohorts(pr_runner.source_hardware_devices(malformed), "rocm", "gfx906")

    def test_gpu_scope_executes_exact_ctest_inventory_and_rejects_skips(self):
        """Replay native CTest indices with poisoned unselected cases and failed GPUs."""
        for outcome in (0, 9, 77):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                build = root / "build"
                build.mkdir()
                # This device-free launcher fixture preserves explicit MPI argv
                # while executing one Python child, with no MPI or GPU runtime.
                launcher = root / "mpirun"
                launcher.write_text(f'#!{sys.executable}\nimport os, sys\n'
                                    'assert sys.argv[1:3] == ["-np", "1"]\n'
                                    'os.execv(sys.argv[3], sys.argv[3:])\n')
                launcher.chmod(0o755)
                forbidden = root / "unselected-ran"
                lines = []
                for name, vendor, selected in (("Host", "", False), ("ROCm", "ROCm", True),
                                                ("Mixed", "CUDA;ROCm", True),
                                                ("Wide", "CUDA;HardwareOptional;LocalHostDevices;Devices2", False),
                                                ("CUDA", "CUDA", True)):
                    code = (f"raise SystemExit({outcome if vendor == 'ROCm' else 0})" if selected else
                            f"from pathlib import Path; Path('{forbidden}').touch(); raise SystemExit(99)")
                    labels = f"Integration;ProductionTestPreflight;{vendor}"
                    skip = "" if "HardwareOptional" in labels else "SKIP_RETURN_CODE 77"
                    lines += [f'add_test(V2_Integration_{name} "{launcher}" -np 1 '
                              f'"{sys.executable}" -c "{code}")',
                              f'set_tests_properties(V2_Integration_{name} PROPERTIES '
                              f'LABELS "{labels}" TIMEOUT 30 {skip})']
                (build / "CTestTestfile.cmake").write_text("\n".join(lines) + "\n")
                output = root / "out"
                arguments = ["--scope", "hardware-preflight", "--build-dir", str(build),
                             "--installed-build-receipt", str(root / "receipt"), "--output", str(output)]
                with patch.object(pr_runner, "validate"), \
                     patch.object(pr_runner, "observe_preflight_hardware", return_value=hardware()):
                    self.assertEqual(pr_runner.main(arguments), 0 if outcome == 0 else 8)
                self.assertFalse(forbidden.exists())
                report = json.loads((output / "pr-tests.json").read_text())
                if outcome == 77:
                    self.assertIn("without skipping", report["lanes"]["rocm"]["validation_error"])
                self.assertEqual(report["scope"], "hardware-preflight")
                self.assertEqual(report["complete"], outcome == 0)
                self.assertEqual(report["lanes"]["cuda"]["tests"], ["V2_Integration_CUDA"])
                self.assertEqual(report["lanes"]["rocm"]["tests"], ["V2_Integration_ROCm"])
                self.assertEqual(set(report["qualification"]["unqualified"]), {"V2_Integration_Wide"})
                suite = ET.parse(output / "hardware-skips.xml").getroot()
                self.assertEqual((suite.get("tests"), suite.get("skipped")), ("1", "1"))
                case, = suite.findall("testcase")
                self.assertEqual((case.get("name"), case.get("status")), ("V2_Integration_Wide", "notrun"))
                skip = case.find("skipped")
                self.assertEqual(skip.get("message"), "requires 2 CUDA GPUs, runner has 1")
                self.assertEqual(json.loads(skip.text), report["qualification"]["unqualified"][case.get("name")])
                self.assertEqual(report["lanes"]["exclusive"]["tests"], ["V2_Integration_Mixed"])
                self.assertTrue(report["lanes"]["exclusive"]["complete"])

    def test_gpu_checks_use_separate_required_container_and_receipt(self):
        """Keep Unit device-free and require both subsequent vendor observations."""
        for mutation in (None, "missing-rocm", "skipped-rocm", "failed-rocm", "omit-exclusive", "forge-qualification"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                scopes = []

                def execute(command, log):
                    scope = (pr_runner.PRTestScope.GPU if "--gpus" in command
                             else pr_runner.PRTestScope.HOST)
                    scopes.append(scope)
                    self.assertEqual(command[command.index("--ulimit") + 1], "core=0:0")
                    prefix = "pr-tests/gpu" if scope is pr_runner.PRTestScope.GPU else "pr-tests"
                    report = pr_report(scope)
                    if scope is pr_runner.PRTestScope.GPU:
                        self.assertEqual(command[-2:], ["--scope", "hardware-preflight"])
                        if mutation == "missing-rocm":
                            report["lanes"].pop("rocm")
                        elif mutation == "skipped-rocm":
                            report["lanes"]["rocm"]["complete"] = False
                        elif mutation == "failed-rocm":
                            report["lanes"]["rocm"]["return_code"] = 8
                        elif mutation == "omit-exclusive":
                            report["lanes"]["exclusive"]["tests"] = []
                            report["lanes"]["exclusive"]["test_count"] = 0
                        elif mutation == "forge-qualification":
                            report["qualification"]["selected"] = ["V2_host_Contract"]
                    gate.write_json(directory / prefix / "pr-tests.json", report)

                with patch.object(gate.docker_paths, "device_args",
                                  side_effect=[[], ["--gpus", "all", "--device", "/dev/kfd"]]) as devices, \
                     patch.object(gate.docker_paths, "mounts", return_value=[]), \
                     patch.object(gate, "run", side_effect=execute), \
                     patch.object(gate.subprocess, "run"):
                    if mutation is None:
                        report = gate.run_pr_tests(images("AVX2"), directory, gate.PRGPUCoverage.CUDA_ROCM)
                        self.assertEqual(set(report["lanes"]), {"unit", "host", "cuda", "rocm", "exclusive"})
                        self.assertEqual(report["scope"], "unit-and-hardware-preflight")
                    else:
                        with self.assertRaises(ValueError):
                            gate.run_pr_tests(images("AVX2"), directory, gate.PRGPUCoverage.CUDA_ROCM)
                    self.assertEqual([call.args[1] for call in devices.call_args_list], ["CPU", "CUDA+ROCm"])
                self.assertEqual(scopes, [pr_runner.PRTestScope.HOST, pr_runner.PRTestScope.GPU])
                self.assertEqual((directory / "pr-tests/pr-validation.json").exists(), mutation is None)

    def test_completed_host_failure_still_exercises_gpus_and_keeps_pr_red(self):
        """A completed CPU test failure preserves both the red job and GPU evidence."""
        with patch.object(gate, "_run_pr_scope", side_effect=[
                subprocess.CalledProcessError(8, "ctest"), pr_report(pr_runner.PRTestScope.GPU)]) as run:
            with self.assertRaises(subprocess.CalledProcessError) as failed:
                gate.run_pr_tests(images("AVX2"), Path("/unused"), gate.PRGPUCoverage.CUDA_ROCM)
        self.assertEqual(failed.exception.returncode, 8)
        self.assertEqual([call.args[2] for call in run.call_args_list],
                         [pr_runner.PRTestScope.HOST, pr_runner.PRTestScope.GPU])

    def test_gpu_requirements_cannot_be_ignored_by_build_only(self):
        """Reject a requested GPU gate when the transaction would not execute tests."""
        with self.assertRaises(SystemExit):
            gate.parse_arguments(["--output", "/unused", "--build-only", "--pr-gpus", "CUDA+ROCm"])

    def test_pr_driver_runs_exact_unit_and_host_inventories_and_rejects_skips(self):
        """Replay real CTest selection while poison accelerator cases stay unrun."""
        for unit_outcome, host_outcome in ((0, 0), (7, 0), (77, 0), (0, 7), (0, 77)):
            with self.subTest(unit=unit_outcome, host=host_outcome), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                build = root / "build"
                build.mkdir()
                forbidden = root / "accelerator-ran"
                registrations = []
                for name, labels, outcome in (
                        ("V2_Unit_Contract", "Unit", unit_outcome),
                        ("V2_Integration_HostContract", "Integration;ProductionTestPreflight", host_outcome)):
                    registrations += [
                        f'add_test({name} "{sys.executable}" "-c" "raise SystemExit({outcome})")',
                        f'set_tests_properties({name} PROPERTIES LABELS "{labels}" TIMEOUT 30 SKIP_RETURN_CODE 77)']
                for name, labels in (("CUDA", "CUDA"), ("ROCm", "ROCm"),
                                     ("Mixed", "CUDA;ROCm"), ("Inventory", "FullDeviceInventory")):
                    registrations += [
                        f'add_test(V2_Integration_{name} "{sys.executable}" "-c" '
                        f'"from pathlib import Path; Path(\'{forbidden}\').touch(); raise SystemExit(99)")',
                        f'set_tests_properties(V2_Integration_{name} PROPERTIES '
                        f'LABELS "Integration;ProductionTestPreflight;{labels}" TIMEOUT 30)']
                (build / "CTestTestfile.cmake").write_text("\n".join(registrations) + "\n")
                output = root / "results"
                arguments = ["--build-dir", str(build), "--installed-build-receipt",
                             str(root / "receipt"), "--output", str(output)]
                with patch.object(pr_runner, "validate") as validate:
                    code = pr_runner.main(arguments)
                    self.assertEqual(code, 0 if unit_outcome == host_outcome == 0 else 8)
                validate.assert_called_once_with(root / "receipt", build,
                                                 pr_runner.InstalledTestScope.PRODUCTION)
                self.assertFalse(forbidden.exists())
                report = json.loads((output / "pr-tests.json").read_text())
                self.assertEqual(report["scope"], "unit-and-host")
                self.assertEqual(report["complete"], unit_outcome == host_outcome == 0)
                self.assertEqual(report["lanes"]["unit"]["tests"], ["V2_Unit_Contract"])
                if unit_outcome == 0:
                    self.assertEqual(report["lanes"]["host"]["tests"], ["V2_Integration_HostContract"])

    def test_pr_driver_rejects_an_empty_host_lane(self):
        """A GPU-only preflight inventory must not become a green host gate."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = root / "build"
            build.mkdir()
            (build / "CTestTestfile.cmake").write_text(
                f'add_test(V2_Unit_Contract "{sys.executable}" "-c" "pass")\n'
                'set_tests_properties(V2_Unit_Contract PROPERTIES LABELS "Unit" TIMEOUT 30)\n'
                f'add_test(V2_Integration_GPU "{sys.executable}" "-c" "raise SystemExit(99)")\n'
                'set_tests_properties(V2_Integration_GPU PROPERTIES '
                'LABELS "Integration;ProductionTestPreflight;CUDA" TIMEOUT 30)\n')
            with patch.object(pr_runner, "validate"), self.assertRaisesRegex(ValueError, "nonempty"):
                pr_runner.main(["--build-dir", str(build), "--installed-build-receipt",
                                str(root / "receipt"), "--output", str(root / "out")])

    def test_unit_results_reject_missing_duplicate_and_foreign_observations(self):
        """A zero CTest exit status alone cannot prove the complete inventory."""
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "unit.xml"
            for names in ([], ["V2_Unit_A", "V2_Unit_A"], ["V2_Integration_A"]):
                report.write_text('<testsuite>' + ''.join(
                    f'<testcase name="{name}" status="run"/>' for name in names) + '</testsuite>')
                with self.assertRaisesRegex(ValueError, "exact CTest inventory"):
                    pr_runner.validate_results(report, ("V2_Unit_A",))

    def test_pr_cpu_admission_requires_each_selected_isa_on_every_cpu(self):
        """Reject missing VNNI and heterogeneous flags without changing the ISA."""
        avx2 = gate.CPU_ISA_FEATURES[gate.CPUISA.AVX2]
        avx512 = gate.CPU_ISA_FEATURES[gate.CPUISA.AVX512]
        for isa, flags in (("AVX2", avx2), ("AVX512", avx512)):
            with patch.object(gate.Path, "read_text", return_value="flags: " + " ".join(flags)):
                gate.require_host_cpu_isa(isa)
        for flags in (avx2, avx512 - {"avx512_vnni"}):
            with patch.object(gate.Path, "read_text", return_value="flags: " + " ".join(flags)), \
                 self.assertRaisesRegex(ValueError, "AVX512.*missing CPU flags.*avx512_vnni"):
                gate.require_host_cpu_isa("AVX512")
        rows = "flags: " + " ".join(avx512) + "\nflags: " + " ".join(avx2)
        with patch.object(gate.Path, "read_text", return_value=rows), \
             self.assertRaisesRegex(ValueError, "missing CPU flags"):
            gate.require_host_cpu_isa("AVX512")
        with patch.object(gate.Path, "read_text", return_value="flags:\n" + rows), \
             self.assertRaisesRegex(ValueError, "missing CPU flags"):
            gate.require_host_cpu_isa("AVX2")
        with patch.object(gate.Path, "read_text", return_value=""), \
             self.assertRaisesRegex(ValueError, "explicit x86 CPU flags"):
            gate.require_host_cpu_isa("AVX2")

    def test_unsupported_pr_isa_fails_before_docker_build(self):
        """A size label cannot authorize expensive compilation for an unusable VM."""
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "out"
            with patch.object(gate, "require_host_cpu_isa", side_effect=ValueError("missing CPU flags")), \
                 patch.object(gate, "build") as build, self.assertRaisesRegex(ValueError, "missing CPU flags"):
                gate.main(["--output", str(output), "--cpu-isa", "AVX512"])
            build.assert_not_called()
            self.assertFalse(output.exists())

    def test_publish_requires_both_shipping_isas(self):
        """A partial local diagnosis can never overwrite a public develop tag."""
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(SystemExit) as error:
                gate.parse_arguments(["--output", str(Path(temporary) / "out"), "--publish",
                                      "--image", "ghcr.io/llaminar/llaminar:develop",
                                      "--cpu-isa", "AVX2"])
        self.assertEqual(error.exception.code, 2)

    def test_build_only_publishes_both_runtimes_without_tests(self):
        """Publication builds only runtimes and cannot execute a test container."""
        source = {"revision": "a" * 40, "tree": "tree", "dirty": False}
        events = []
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"

            def build(args, admitted, directory, *, roles, test_inventory):
                self.assertEqual(admitted, source)
                self.assertEqual(roles, (gate.ImageRole.RUNTIME,))
                events.append(("build", args.cpu_isa))
                return {"runtime": images(args.cpu_isa)["runtime"]}

            def publish(built, isa, tag, revision, directory):
                if isa == "AVX512":
                    self.assertEqual(events, [("build", "AVX512"), ("build", "AVX2")])
                events.append(("publish", isa))
                return {"tag": gate.runtime_tag(tag, isa),
                        "ref_tag": gate.revision_tag(tag, isa, revision),
                        "image": built["runtime"]["id"]}

            with patch.object(gate, "source_identity", return_value=source), \
                 patch.object(gate, "build", side_effect=build), \
                 patch.object(gate, "require_image"), \
                 patch.object(gate, "require_host_cpu_isa") as cpu_admission, \
                 patch.object(gate, "run_pr_tests") as tests, \
                 patch.object(gate, "publish_runtime", side_effect=publish):
                self.assertEqual(gate.main(["--output", str(output), "--build-only", "--publish",
                                            "--image", "ghcr.io/llaminar/llaminar:develop"]), 0)
            tests.assert_not_called()
            cpu_admission.assert_not_called()
            self.assertEqual(events, [("build", "AVX512"), ("build", "AVX2"),
                                      ("publish", "AVX512"), ("publish", "AVX2")])
            receipt = json.loads((output / "develop-image-gate.json").read_text())
            self.assertTrue(receipt["complete"])
            self.assertTrue(receipt["published"])
            self.assertEqual(receipt["validation"], "build-only")
            for variant in receipt["variants"].values():
                self.assertEqual(set(variant), {"runtime_image", "publication"})
            self.assertEqual(receipt["variants"]["AVX2"]["publication"]["ref_tag"],
                             "ghcr.io/llaminar/llaminar:develop-" + "a" * 40 + "-avx2")

    def test_second_build_failure_prevents_every_registry_mutation(self):
        """A successful first ISA cannot conceal failure of the second build."""
        source = {"revision": "a" * 40, "tree": "tree", "dirty": False}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            with patch.object(gate, "source_identity", return_value=source), \
                 patch.object(gate, "build", side_effect=[images("AVX512"), RuntimeError("build red")]), \
                 patch.object(gate, "require_image"), \
                 patch.object(gate, "publish_runtime") as publish:
                with self.assertRaisesRegex(RuntimeError, "build red"):
                    gate.main(["--output", str(output), "--build-only", "--publish",
                               "--image", "ghcr.io/llaminar/llaminar:develop"])
            publish.assert_not_called()
            self.assertFalse(json.loads((output / "develop-image-gate.json").read_text())["complete"])

    def test_hosted_stage_builds_and_uploads_one_runtime(self):
        """Staging has an ISA-specific receipt and leaves the develop aliases to the publisher."""
        source = {"revision": "a" * 40, "tree": "tree", "dirty": False}
        base = "ghcr.io/llaminar/llaminar:develop"
        runtime = "sha256:" + "1" * 64
        manifest = "sha256:" + "2" * 64
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "stage"
            with patch.object(gate, "source_identity", return_value=source), \
                 patch.object(gate, "build", return_value={"runtime": {"id": runtime}}) as build, \
                 patch.object(gate, "require_image"), \
                 patch.object(gate, "require_host_cpu_isa") as cpu, \
                 patch.object(gate, "run_pr_tests") as tests, \
                 patch.object(gate, "run") as run, \
                 patch.object(gate, "remote_image_id", return_value=runtime) as remote, \
                 patch.object(gate.subprocess, "check_output", return_value=json.dumps(manifest)):
                self.assertEqual(gate.main(["--output", str(output), "--build-only", "--stage",
                                            "--image", base, "--cpu-isa", "AVX2"]), 0)
            self.assertEqual(build.call_args.args[0].cpu_isa, "AVX2")
            self.assertEqual(build.call_args.kwargs["roles"], (gate.ImageRole.RUNTIME,))
            cpu.assert_not_called()
            tests.assert_not_called()
            self.assertEqual([call.args[0] for call in run.call_args_list], [
                ["docker", "tag", runtime, base + "-staging-avx2"],
                ["docker", "push", base + "-staging-avx2"],
            ])
            remote.assert_called_once_with(base.rsplit(":", 1)[0] + "@" + manifest)
            receipt = json.loads((output / "develop-image-gate.json").read_text())
            self.assertTrue(receipt["complete"])
            self.assertFalse(receipt["published"])
            self.assertEqual(receipt["variants"], {"AVX2": {
                "runtime_image": runtime, "staged_manifest": manifest}})

    def test_hosted_publish_verifies_both_sources_before_copying_registry_manifests(self):
        """A missing, stale, mislabeled or conflicting lane cannot publish either ISA."""
        source = {"revision": "a" * 40, "tree": "tree", "dirty": False}
        base = "ghcr.io/llaminar/llaminar:develop"
        for defect in (None, "missing", "foreign-tree", "foreign-isa", "different-config", "pinned-conflict"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                registry, configs, expected = {}, {}, {}
                for index, isa in enumerate(gate.SHIPPING_ISAS):
                    if defect == "missing" and isa == "AVX2":
                        continue
                    manifest = "sha256:" + str(index + 1) * 64
                    runtime = "sha256:" + str(index + 3) * 64
                    ref = f"{base.rsplit(':', 1)[0]}@{manifest}"
                    expected[isa] = runtime
                    negative = defect if isa == "AVX2" else None
                    registry[ref] = "sha256:" + "9" * 64 if negative == "different-config" else runtime
                    configs[ref] = {"config": {"Labels": {
                        "org.opencontainers.image.revision": source["revision"],
                        "org.llaminar.source_tree": source["tree"],
                        "org.llaminar.build_type": "Release", "org.llaminar.image_role": "runtime",
                        "org.llaminar.cpu_isa": "AVX512" if negative == "foreign-isa" else isa,
                        "org.llaminar.cuda": "ON", "org.llaminar.rocm": "ON",
                    }}}
                    if negative == "pinned-conflict":
                        registry[gate.revision_tag(base, isa, source["revision"])] = "sha256:" + "f" * 64
                    artifact = root / f"develop-stage-{isa}"
                    artifact.mkdir()
                    gate.write_json(artifact / "develop-image-gate.json", {
                        "schema": 1, "source": {**source, "tree": "foreign"} if negative == "foreign-tree" else source,
                        "image": base, "validation": "build-only", "complete": True, "published": False,
                        "variants": {isa: {"runtime_image": runtime, "staged_manifest": manifest}},
                    })

                def copy(command, log):
                    self.assertEqual(command[:5], ["docker", "buildx", "imagetools", "create", "--prefer-index=false"])
                    registry[command[-2]] = registry[command[-1]]

                output = root / "published"
                with patch.object(gate, "source_identity", return_value=source), \
                     patch.object(gate, "remote_image_id", side_effect=registry.get), \
                     patch.object(gate.subprocess, "check_output", side_effect=lambda cmd, **_: json.dumps(configs[cmd[-1]])), \
                     patch.object(gate, "run", side_effect=copy) as run, \
                     patch.object(gate, "build") as build:
                    args = ["--output", str(output), "--image", base, "--publish-staged", str(root)]
                    if defect is not None:
                        with self.assertRaises((ValueError, FileNotFoundError)):
                            gate.main(args)
                        run.assert_not_called()
                    else:
                        self.assertEqual(gate.main(args), 0)
                        self.assertEqual([call.args[0][-2] for call in run.call_args_list], [
                            *(gate.revision_tag(base, isa, source["revision"]) for isa in gate.SHIPPING_ISAS),
                            *(gate.runtime_tag(base, isa) for isa in gate.SHIPPING_ISAS),
                        ])
                    build.assert_not_called()
                receipt = json.loads((output / "develop-image-gate.json").read_text())
                self.assertEqual(receipt["complete"], defect is None)
                self.assertEqual(receipt["published"], defect is None)
                if defect is None:
                    self.assertEqual({isa: variant["runtime_image"] for isa, variant in receipt["variants"].items()}, expected)

    def test_ref_tag_is_immutable_and_published_before_mutable_branch(self):
        source = "a" * 40
        runtime = {"runtime": {"id": "sha256:" + "1" * 64}}
        base = "ghcr.io/llaminar/llaminar:develop"
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with patch.object(gate, "remote_image_id", return_value=None), \
                 patch.object(gate, "run") as execute, \
                 patch.object(gate, "image_identity", return_value=runtime["runtime"]):
                result = gate.publish_runtime(runtime, "AVX2", base, source, directory)
            pinned = base + "-" + source + "-avx2"
            mutable = base + "-avx2"
            self.assertEqual(result["ref_tag"], pinned)
            self.assertEqual([call.args[0][:2] for call in execute.call_args_list],
                             [["docker", "tag"], ["docker", "push"],
                              ["docker", "tag"], ["docker", "push"]])
            self.assertEqual(execute.call_args_list[1].args[0][-1], pinned)
            self.assertEqual(execute.call_args_list[3].args[0][-1], mutable)
            with patch.object(gate, "remote_image_id", return_value="sha256:" + "2" * 64), \
                 patch.object(gate, "run") as execute:
                with self.assertRaisesRegex(ValueError, "different image bytes"):
                    gate.publish_runtime(runtime, "AVX2", base, source, directory)
                execute.assert_not_called()

    def test_registry_probe_distinguishes_missing_from_transport_failure(self):
        manifest = '{"config":{"digest":"sha256:' + "1" * 64 + '"}}'
        with patch.object(gate.subprocess, "run", return_value=type("Result", (), {
                "returncode": 0, "stdout": manifest, "stderr": ""})()):
            self.assertEqual(gate.remote_image_id("example:ref"), "sha256:" + "1" * 64)
        with patch.object(gate.subprocess, "run", return_value=type("Result", (), {
                "returncode": 1, "stdout": "", "stderr": "manifest unknown",
                "args": ["docker"]})()):
            self.assertIsNone(gate.remote_image_id("example:missing"))
        with patch.object(gate.subprocess, "run", return_value=type("Result", (), {
                "returncode": 1, "stdout": "", "stderr": "network unavailable",
                "args": ["docker"]})()):
            with self.assertRaises(gate.subprocess.CalledProcessError):
                gate.remote_image_id("example:failed")

    def test_canceled_build_retires_exact_child_group(self):
        """SIGTERM of the Actions driver cannot orphan its Buildx-style child."""
        driver_source = (
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "from run_production_pipeline import run\n"
            "child = 'import os,sys,time;from pathlib import Path;"
            "Path(sys.argv[1]).write_text(str(os.getpid()));time.sleep(30)'\n"
            "run([sys.executable, '-c', child, sys.argv[2]], Path(sys.argv[3]))\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pid_file, log = root / "child.pid", root / "build.log"
            driver = subprocess.Popen([sys.executable, "-c", driver_source,
                                       str(ROOT / "scripts/ci"), str(pid_file), str(log)],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                      start_new_session=True)
            child_pid = None
            try:
                deadline = time.monotonic() + 5
                while not pid_file.exists() and driver.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(pid_file.exists(), "build child never reached its ready edge")
                child_pid = int(pid_file.read_text())
                os.kill(driver.pid, signal.SIGTERM)
                os.kill(driver.pid, signal.SIGTERM)
                self.assertNotEqual(driver.wait(timeout=5), 0)
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
            finally:
                if driver.poll() is None:
                    driver.terminate()
                    driver.wait(timeout=5)
                if child_pid is not None:
                    try:
                        os.kill(child_pid, 0)
                    except ProcessLookupError:
                        pass
                    else:
                        os.killpg(child_pid, signal.SIGKILL)

    def test_ci_replaces_the_step_shell_with_the_cancellable_driver(self):
        """The selected CI steps deliver cancellation directly to the image gate."""
        workflows = ROOT / ".github/workflows"
        driver = "python3 scripts/ci/run_develop_image_gate.py"
        for workflow, job in (
            ("develop-pr-tests.yml", "unit"),
            ("develop-image-isa.yml", "runtime"),
            ("ci.yml", "publish"),
        ):
            with self.subTest(workflow=workflow, job=job):
                policy = yaml.load((workflows / workflow).read_text(), Loader=yaml.BaseLoader)
                commands = [
                    step["run"]
                    for step in policy["jobs"][job]["steps"]
                    if driver in step.get("run", "")
                ]
                self.assertEqual(len(commands), 1)
                self.assertTrue(
                    any(line.startswith(f"exec {driver} ") for line in commands[0].splitlines()),
                    "the image gate must replace its step shell",
                )

    def test_ci_stages_each_isa_after_its_own_toolchain_job(self):
        workflow = (ROOT / ".github/workflows/ci.yml").read_text()
        self.assertIn("exec python3 scripts/ci/run_develop_image_gate.py", workflow)
        policy = yaml.load(workflow, Loader=yaml.BaseLoader)
        job = policy["jobs"]["develop-images"]
        self.assertEqual(job["strategy"], {"fail-fast": "false", "matrix": {"cpu_isa": list(gate.SHIPPING_ISAS)}})
        self.assertEqual(job["uses"], "./.github/workflows/develop-image-isa.yml")
        self.assertEqual(job["with"], {"cpu-isa": "${{ matrix.cpu_isa }}"})
        self.assertEqual(policy["concurrency"], {
            "group": "llaminar-develop-images", "cancel-in-progress": "true"})
        reusable_path = ROOT / job["uses"]
        reusable = yaml.load(reusable_path.read_text(), Loader=yaml.BaseLoader)["jobs"]
        preparation, runtime = reusable["toolchain"], reusable["runtime"]
        self.assertEqual(runtime["needs"], "toolchain")
        for phase in (preparation, runtime):
            self.assertEqual(phase["runs-on"], "ubuntu-24.04")
            self.assertEqual(phase["timeout-minutes"], "240")
            self.assertEqual(phase["steps"][0]["with"]["ref"], "${{ github.sha }}")
        prepare_steps, steps = preparation["steps"], runtime["steps"]
        buildx = next(step for step in steps if step.get("uses") == "docker/setup-buildx-action@v4")
        self.assertEqual(buildx["with"]["driver"], "docker-container")
        self.assertNotEqual(buildx["with"].get("keep-state"), "true")
        cold_build, = [step for step in prepare_steps if step.get("uses") == "./.github/actions/cache-develop-toolchain"]
        self.assertEqual(cold_build["with"], {"cpu-isa": "${{ inputs.cpu-isa }}"})
        self.assertEqual(cold_build["if"], "steps.lookup.outputs.cache-hit != 'true'")
        first_lookup, ready = [step for step in prepare_steps if step.get("uses") == "actions/cache/restore@v5"]
        restore, = [step for step in steps if step.get("uses") == "actions/cache/restore@v5"]
        handoff = {"key": "${{ steps.toolchain-cache.outputs.key }}",
                   "path": "${{ steps.toolchain-cache.outputs.path }}"}
        self.assertEqual(first_lookup["with"], {**handoff, "lookup-only": "true"})
        self.assertEqual(ready["with"], {**handoff, "lookup-only": "true", "fail-on-cache-miss": "true"})
        self.assertLess(prepare_steps.index(first_lookup), prepare_steps.index(cold_build))
        self.assertLess(prepare_steps.index(cold_build), prepare_steps.index(ready))
        self.assertEqual(restore["with"], {**handoff, "fail-on-cache-miss": "true"})
        verify = next(step for step in steps if "toolchain_cache.py verify" in step.get("run", ""))
        staging = next(step for step in steps if "--stage" in step.get("run", ""))
        self.assertLess(steps.index(restore), steps.index(verify))
        self.assertLess(steps.index(verify), steps.index(staging))
        self.assertIn("LLAMINAR_DOCKER_TOOLCHAIN_ROOT", verify["run"])
        self.assertIn("exec python3 scripts/ci/run_develop_image_gate.py", staging["run"])
        self.assertIn("--build-only", staging["run"])
        self.assertEqual(staging["env"]["LLAMINAR_DOCKER_BUILD_CACHE_ACCESS"], "read-write")
        self.assertIn(':build-cache-develop"', staging["run"])
        self.assertIn('--cpu-isa "$CPU_ISA"', staging["run"])
        self.assertIn("kernel.core_pattern=/proc/llaminar-core-disabled", reusable_path.read_text())
        publisher = policy["jobs"]["publish"]
        self.assertEqual(publisher["runs-on"], "ubuntu-24.04")
        self.assertEqual(publisher["needs"], "develop-images")
        self.assertTrue(any("--publish-staged" in step.get("run", "") for step in publisher["steps"]))

    def test_failed_job_logs_survive_a_killed_build_and_a_missing_sibling_log(self):
        """Replay a killed build's GitHub transcript without needing a live runner or network."""
        workflows = ROOT / ".github/workflows"
        for caller, build in (("develop-pr.yml", "unit"), ("ci.yml", "develop-images")):
            job = yaml.load((workflows / caller).read_text(), Loader=yaml.BaseLoader)["jobs"]["failed-job-logs"]
            if caller == "ci.yml":
                self.assertEqual(job["needs"], [build, "publish"])
                self.assertEqual(job["if"], "always() && (contains(needs.*.result, 'failure') || "
                                 "contains(needs.*.result, 'cancelled'))")
            else:
                self.assertEqual(job["needs"], build)
                self.assertEqual(job["if"], f"always() && (needs.{build}.result == 'failure' || needs.{build}.result == 'cancelled')")
            self.assertEqual(job["permissions"], {"actions": "read"})
            self.assertEqual(job["uses"], "./.github/workflows/failed-job-logs.yml")
        workflow = yaml.load((workflows / "failed-job-logs.yml").read_text(), Loader=yaml.BaseLoader)
        collect, upload, summary = workflow["jobs"]["collect"]["steps"]
        self.assertIn('.conclusion == "cancelled"', collect["run"])
        self.assertEqual(upload["if"], "${{ always() && steps.transcripts.outputs.expected == 'true' }}")
        self.assertIn("github.run_attempt", upload["with"]["name"])
        self.assertIn("steps.upload.outputs.artifact-url", summary["env"]["ARTIFACT_URL"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # The API lists two failed jobs. One has no transcript yet. The
            # other includes both a signed cache URL and the runner's exit code.
            gh = root / "gh"
            gh.write_text("#!/bin/sh\nprintf '12\\tAVX2\\n13\\tAVX512\\n'\n")
            curl = root / "curl"
            curl.write_text("""#!/bin/sh
while [ "$1" != --output ]; do shift; done
case "$3" in
  */12/logs) : > "$2" ;;
  */13/logs) printf '%s\\n' 'https://cache.example/blob?sig=private' 'exit code 137' > "$2" ;;
  *) exit 1 ;;
esac
""")
            gh.chmod(0o755)
            curl.chmod(0o755)
            env = dict(os.environ, PATH=f"{root}:{os.environ['PATH']}", RUNNER_TEMP=directory,
                       GITHUB_OUTPUT=str(root / "outputs"), GH_TOKEN="unused", GITHUB_API_URL="https://api.example", GITHUB_REPOSITORY="test/repo",
                       GITHUB_RUN_ID="41", GITHUB_RUN_ATTEMPT="2")
            result = subprocess.run(["bash", "-c", collect["run"]], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("job 12", result.stdout)
            logs = root / "failed-job-logs"
            self.assertEqual([p.name for p in logs.iterdir()], ["AVX512-13.log"])
            self.assertEqual((logs / "AVX512-13.log").read_text(), "https://cache.example/blob?[redacted]\nexit code 137\n")

    def test_develop_pr_runs_both_isas_on_separate_dedicated_runners(self):
        """Exercise the real PR command, not a separately invented argument set."""
        workflow = yaml.load((ROOT / ".github/workflows/develop-pr.yml").read_text(),
                             Loader=yaml.BaseLoader)
        self.assertEqual(set(workflow["on"]), {"pull_request"})
        self.assertEqual(workflow["on"]["pull_request"]["branches"], ["develop"])
        self.assertEqual(set(workflow["on"]["pull_request"]["types"]),
                         {"opened", "synchronize", "reopened", "ready_for_review"})
        self.assertEqual(workflow["permissions"], {"contents": "read"})
        self.assertNotIn("concurrency", workflow)
        self.assertEqual(set(workflow["jobs"]), {"unit", "failed-job-logs"})
        job = workflow["jobs"]["unit"]
        self.assertEqual(job["uses"],
                         "Llaminar/llaminar/.github/workflows/develop-pr-tests.yml@develop")
        # Drafts allocate no runner, while ready_for_review enters this gate.
        self.assertEqual(job["if"],
                         "github.event.pull_request.head.repo.full_name == github.repository && "
                         "!github.event.pull_request.draft")
        reusable = yaml.load((ROOT / ".github/workflows/develop-pr-tests.yml").read_text(),
                             Loader=yaml.BaseLoader)
        self.assertEqual(set(reusable["on"]), {"workflow_call"})
        self.assertEqual(reusable["permissions"], {"contents": "read"})
        caller = job
        job = reusable["jobs"]["unit"]
        self.assertEqual(job["concurrency"], {
            "group": "llaminar-develop-pr-${{ github.event.pull_request.number }}-${{ matrix.cpu_isa }}",
            "cancel-in-progress": "true"})
        self.assertEqual(job["runs-on"], "${{ matrix.runner }}")
        self.assertEqual(job["strategy"], {"fail-fast": "false", "max-parallel": "2",
            "matrix": {"cpu_isa": ["AVX2", "AVX512"], "include": [
                {"cpu_isa": "AVX2", "runner": "llaminar-develop-avx2", "pr_gpus": "CUDA+ROCm",
                 "check_name": "Unit + hardware preflight (AVX2)"},
                {"cpu_isa": "AVX512", "runner": "llaminar-avx512", "pr_gpus": "none",
                 "check_name": "Unit + host (AVX512)"}]}})
        self.assertEqual(len({row["runner"] for row in job["strategy"]["matrix"]["include"]}), 2)
        self.assertEqual(f"{caller['name']} / {job['name']}",
                         "Develop PR / ${{ matrix.check_name }}")
        self.assertIn("github.event_name == 'pull_request'", job["if"])
        self.assertIn("github.base_ref == 'develop'", job["if"])
        self.assertIn("github.event.pull_request.head.repo.full_name == github.repository", job["if"])
        self.assertIn("!github.event.pull_request.draft", job["if"])
        steps = job["steps"]
        actions = {step["uses"] for step in steps if "uses" in step}
        self.assertEqual(actions, {"actions/checkout@v6", "docker/setup-buildx-action@v4",
                                   "actions/upload-artifact@v4",
                                   "./.github/actions/cache-develop-toolchain"})
        checkout = next(step for step in steps if step.get("uses") == "actions/checkout@v6")
        self.assertEqual(checkout["with"], {
            "submodules": "false", "lfs": "false", "persist-credentials": "false"})
        buildx = next(step for step in steps if step.get("uses") == "docker/setup-buildx-action@v4")
        for key, value in {"driver": "docker-container", "cache-binary": "false"}.items():
            self.assertEqual(buildx["with"][key], value)
        commands = [step["run"] for step in steps if "run" in step]
        self.assertEqual(len(commands), 2)
        self.assertEqual(buildx["with"]["keep-state"], "true")
        self.assertEqual(buildx["with"]["name"], "llaminar-develop")
        self.assertNotIn("cleanup", buildx["with"])
        self.assertNotIn("sudo", "\n".join(commands))
        build_step = next(step for step in steps if "exec python3" in step.get("run", ""))
        self.assertEqual(build_step["env"], {"CPU_ISA": "${{ matrix.cpu_isa }}",
                                            "PR_GPUS": "${{ matrix.pr_gpus }}",
                                            "LLAMINAR_DOCKER_BUILD_CACHE_ACCESS": "import-only"})
        self.assertIn("LLAMINAR_DOCKER_BUILD_CACHE_IMAGE", commands[-1])
        artifact = next(step for step in steps if step.get("uses") == "actions/upload-artifact@v4")
        self.assertIn("${{ matrix.cpu_isa }}", artifact["with"]["name"])
        self.assertIn("/*/pr-tests/**", artifact["with"]["path"])
        for isa in job["strategy"]["matrix"]["cpu_isa"]:
            command = shlex.split(commands[-1].split("exec ", 1)[1].replace("\\\n", ""))
            self.assertEqual(command[:2], ["python3", "scripts/ci/run_develop_image_gate.py"])
            arguments = command[2:]
            arguments[arguments.index("--cpu-isa") + 1] = isa
            gpu_policy = next(row["pr_gpus"] for row in job["strategy"]["matrix"]["include"]
                              if row["cpu_isa"] == isa)
            arguments[arguments.index("--pr-gpus") + 1] = gpu_policy
            policy = gate.parse_arguments(arguments)
            self.assertEqual(policy.pr_gpus.value, gpu_policy)
            # develop's protected caller supplies only the ISA and output on
            # the first PR introducing GPU coverage. Exercise its actual argv.
            protected = gate.parse_arguments(["--cpu-isa", isa, "--output", "/unused"])
            self.assertIs(protected.pr_gpus, policy.pr_gpus)
            self.assertEqual(policy.cpu_isas, [isa])
            self.assertFalse(policy.publish)
            self.assertIsNone(policy.image)

            source = {"revision": "a" * 40, "tree": "merge-tree", "dirty": False}
            events = []
            with tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "output"
                arguments[arguments.index("--output") + 1] = str(output)

                def build(args, admitted, directory, *, roles, test_inventory):
                    self.assertEqual(admitted, source)
                    self.assertEqual(test_inventory, gate.TestRunnerInventory.MODEL_FREE)
                    self.assertEqual(roles, (gate.ImageRole.TEST_RUNNER,))
                    events.append(("build", args.cpu_isa))
                    return images(args.cpu_isa)

                def unit_tests(built, directory, gpu_coverage):
                    events.append(("unit", built["test-runner"]["id"]))
                    self.assertEqual(gpu_coverage.value, gpu_policy)
                    report = pr_report()
                    report["scope"] = gpu_coverage.validation
                    if gpu_coverage is gate.PRGPUCoverage.CUDA_ROCM:
                        report["lanes"].update(pr_report(pr_runner.PRTestScope.GPU)["lanes"])
                    return report

                with patch.object(gate, "require_host_cpu_isa") as cpu_admission, \
                     patch.object(gate, "source_identity", return_value=source), \
                     patch.object(gate, "build", side_effect=build), \
                     patch.object(gate, "require_image"), \
                     patch.object(gate, "run_pr_tests", side_effect=unit_tests), \
                     patch.object(gate, "publish_runtime") as publish:
                    self.assertEqual(gate.main(arguments), 0)
                publish.assert_not_called()
                cpu_admission.assert_called_once_with(isa)
                self.assertEqual(events, [("build", isa),
                                         ("unit", f"test-runner-{isa.lower()}")])
                receipt = json.loads((output / "develop-image-gate.json").read_text())
                self.assertTrue(receipt["complete"])
                self.assertFalse(receipt["published"])
                self.assertEqual(set(receipt["variants"]), {isa})
                self.assertEqual(receipt["validation"], policy.pr_gpus.validation)
                self.assertEqual(receipt["variants"][isa]["unit_test_count"], 1)
                self.assertEqual(receipt["variants"][isa]["host_test_count"], 1)

    def test_develop_auto_merge_waits_for_protected_pr_gate(self):
        """Only trusted, non-draft PRs arm native auto-merge; no PR code runs."""
        workflow = yaml.load((ROOT / ".github/workflows/develop-auto-merge.yml").read_text(),
                             Loader=yaml.BaseLoader)
        self.assertEqual(set(workflow["on"]), {"pull_request_target"})
        self.assertEqual(workflow["on"]["pull_request_target"]["branches"], ["develop"])
        self.assertEqual(set(workflow["on"]["pull_request_target"]["types"]),
                         {"opened", "reopened", "synchronize", "ready_for_review", "edited"})
        self.assertEqual(workflow["permissions"],
                         {"contents": "read", "pull-requests": "read"})
        self.assertEqual(set(workflow["jobs"]), {"arm", "arm-owner"})
        job = workflow["jobs"]["arm"]
        self.assertIn("head.repo.full_name == github.repository", job["if"])
        self.assertIn("!github.event.pull_request.draft", job["if"])
        self.assertIn("github.event.pull_request.user.id != 3513358", job["if"])
        self.assertEqual(len(job["steps"]), 1)
        owner = workflow["jobs"]["arm-owner"]
        self.assertIn("github.event.pull_request.user.id == 3513358", owner["if"])
        self.assertIn("head.repo.full_name == github.repository", owner["if"])
        self.assertIn("!github.event.pull_request.draft", owner["if"])
        self.assertEqual(owner["environment"], "llaminar-bot")
        mint, arm = owner["steps"]
        self.assertRegex(mint["uses"], r"^actions/create-github-app-token@[0-9a-f]{40}$")
        self.assertEqual(mint["with"], {
            "client-id": "${{ vars.LLAMINAR_BOT_CLIENT_ID }}",
            "private-key": "${{ secrets.LLAMINAR_BOT_PRIVATE_KEY }}",
            "permission-contents": "write", "permission-pull-requests": "write"})
        self.assertEqual(arm["env"]["GH_TOKEN"], "${{ steps.bot.outputs.token }}")
        self.assertIn("--auto --squash", arm["run"])
        self.assertIn('--match-head-commit "$PR_HEAD_SHA"', arm["run"])
        self.assertNotIn("--admin", str(workflow))
        self.assertNotIn("checkout", str(owner))
        step = job["steps"][0]
        self.assertEqual(step["env"]["GH_TOKEN"], "${{ secrets.DEVELOP_AUTOMERGE_TOKEN }}")
        self.assertEqual(step["env"]["PR_NUMBER"],
                         "${{ github.event.pull_request.number }}")
        self.assertEqual(step["env"]["PR_HEAD_SHA"], "${{ github.event.pull_request.head.sha }}")
        self.assertIn('gh pr merge "$PR_NUMBER" --repo "$GITHUB_REPOSITORY" --auto --squash', step["run"])
        self.assertIn('--match-head-commit "$PR_HEAD_SHA"', step["run"])
        self.assertNotIn("checkout", str(job))

    def test_owner_job_migrates_native_merge_identity_for_the_current_head(self):
        """Execute the shipped workflow shell for new and previously armed PRs."""
        workflow = yaml.load((ROOT / ".github/workflows/develop-auto-merge.yml").read_text(),
                             Loader=yaml.BaseLoader)
        command = workflow["jobs"]["arm-owner"]["steps"][-1]["run"]
        repo = "Llaminar/llaminar"
        sha = "a" * 40
        arm = f"pr merge 64 --repo {repo} --auto --squash --match-head-commit {sha}"
        disable = f"pr merge 64 --repo {repo} --disable-auto"
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            gh = directory / "gh"
            gh.write_text('#!/bin/sh\nif [ "$1" = api ]; then cat "$PR_JSON"; '
                          'else echo "$*" >> "$MERGE_LOG"; fi\n')
            gh.chmod(0o755)
            fixture, log = directory / "pr.json", directory / "calls"
            for login, action, author, source, expected in [
                    (None, "opened", 3513358, sha, [arm]),
                    ("dbsanfte", "edited", 3513358, sha, [disable, arm]),
                    (None, "edited", 3513358, sha, []),
                    ("llaminar-bot[bot]", "synchronize", 3513358, sha, []),
                    ("dbsanfte", "edited", 42, sha, []),
                    ("dbsanfte", "edited", 3513358, "b" * 40, [])]:
                with self.subTest(login=login, action=action, author=author, source=source):
                    fixture.write_text(json.dumps({"head": {"sha": source, "repo": {"full_name": repo}},
                        "base": {"ref": "develop"}, "user": {"id": author}, "draft": False,
                        "state": "open", "auto_merge": {"enabled_by": {"login": login}} if login else None}))
                    log.write_text("")
                    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True,
                        env={**os.environ, "PATH": temp + os.pathsep + os.environ["PATH"],
                             "PR_JSON": str(fixture), "MERGE_LOG": str(log), "GITHUB_REPOSITORY": repo,
                             "PR_NUMBER": "64", "PR_HEAD_SHA": sha, "PR_EVENT_ACTION": action})
                    self.assertEqual(result.returncode == 0, author == 3513358 and source == sha)
                    self.assertEqual(log.read_text().splitlines(), expected)

    def test_all_host_ci_jobs_share_a_serial_non_discarding_queue(self):
        """Production certification jobs cannot contend for this host's GPUs."""
        values = yaml.load((ROOT / "scripts/ci/arc/llaminar-xeon-host-docker-values.yaml")
                           .read_text(), Loader=yaml.BaseLoader)
        self.assertEqual(values["maxRunners"], "1")
        expected = {"group": "llaminar-develop-image-gate",
                    "cancel-in-progress": "false", "queue": "max"}
        admitted = []
        for path in (ROOT / ".github/workflows").glob("*.yml"):
            workflow = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
            for name, job in workflow.get("jobs", {}).items():
                if job.get("runs-on") not in (["llaminar-xeon-host"], "self-hosted"):
                    continue
                admitted.append(f"{path.name}:{name}")
                self.assertEqual(job.get("concurrency", workflow.get("concurrency")),
                                 expected, admitted[-1])
        self.assertGreaterEqual(len(admitted), 5)

    def test_pr_executes_in_exact_test_image_without_devices_or_models(self):
        """A PR must test its built image, not a host checkout or older runtime."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)

            def run(command, log):
                self.assertEqual(command[:3], ["docker", "run", "--rm"])
                start = command.index("test-runner-avx512")
                self.assertNotIn("--gpus", command)
                self.assertNotIn("--device", command)
                invocation = command[start + 1:]
                # The log wrapper receives the real canonical prerequisite
                # command as positional arguments, retaining its exit status.
                self.assertEqual(invocation[-8:], [
                    "python3", "scripts/ci/run_develop_pr_tests.py",
                    "--build-dir", "build_v2_integration",
                    "--installed-build-receipt", "/src/installed-tests.json",
                    "--output", "/ci-results/pr-tests"])
                gate.write_json(directory / "pr-tests/pr-tests.json",
                                pr_report())

            with patch.object(gate.docker_paths, "device_args", return_value=[]) as devices, \
                 patch.object(gate.docker_paths, "mounts", return_value=[]) as mounts, \
                 patch.object(gate, "run", side_effect=run), \
                 patch.object(gate.subprocess, "run") as retire:
                self.assertEqual(gate.run_pr_tests(images("AVX512"), directory),
                                 pr_report())
            devices.assert_called_once_with("test-runner-avx512", "CPU",
                                            user=f"{os.getuid()}:{os.getgid()}")
            mounts.assert_called_once_with([(directory, "/ci-results", False)])
            retirement = retire.call_args.args[0]
            self.assertEqual(retirement[:3], ["docker", "rm", "-f"])
            self.assertEqual(len(retirement), 4)
            self.assertTrue(retirement[3].startswith("llaminar-develop-pr-tests-"))

    def test_pr_receipt_cannot_omit_or_skip_the_host_lane(self):
        """The outer image transaction rejects incomplete and broader receipts."""
        mutations = (
            lambda report: report["lanes"].pop("host"),
            lambda report: report["lanes"]["host"].update(complete=False),
            lambda report: report["lanes"]["host"].update(return_code=1),
            lambda report: report["lanes"]["host"].update(tests=[], test_count=0),
            lambda report: report.update(scope="production"),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                report = pr_report()
                mutate(report)
                gate.write_json(directory / "pr-tests/pr-tests.json", report)
                with patch.object(gate.docker_paths, "mounts", return_value=[]), \
                     patch.object(gate, "run"), patch.object(gate.subprocess, "run"), \
                     self.assertRaisesRegex(ValueError, "incomplete"):
                    gate.run_pr_tests(images("AVX2"), directory)


if __name__ == "__main__":
    unittest.main()
