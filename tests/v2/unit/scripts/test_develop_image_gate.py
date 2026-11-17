#!/usr/bin/env python3
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
                    if outcome == 77:
                        with self.assertRaisesRegex(ValueError, "without skipping"):
                            pr_runner.main(arguments)
                    else:
                        self.assertEqual(pr_runner.main(arguments) == 0, outcome == 0)
                self.assertFalse(forbidden.exists())
                report = json.loads((output / "pr-tests.json").read_text())
                self.assertEqual(report["scope"], "hardware-preflight")
                self.assertEqual(report["complete"], outcome == 0)
                self.assertEqual(report["lanes"]["cuda"]["tests"], ["V2_Integration_CUDA"])
                self.assertEqual(report["lanes"]["rocm"]["tests"], ["V2_Integration_ROCm"])
                self.assertEqual(set(report["qualification"]["unqualified"]), {"V2_Integration_Wide"})
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
                    prefix = "pr-gpu-tests" if scope is pr_runner.PRTestScope.GPU else "pr-tests"
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
                self.assertEqual((directory / "pr-validation.json").exists(), mutation is None)

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
                    if 77 in (unit_outcome, host_outcome):
                        with self.assertRaisesRegex(ValueError, "without skipping"):
                            pr_runner.main(arguments)
                    else:
                        code = pr_runner.main(arguments)
                        self.assertEqual(code == 0, unit_outcome == host_outcome == 0)
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
        workflow = (ROOT / ".github/workflows/ci.yml").read_text()
        self.assertIn("exec python3 scripts/ci/run_develop_image_gate.py", workflow)
        policy = yaml.load(workflow, Loader=yaml.BaseLoader)
        self.assertEqual(policy["jobs"]["develop-images"]["runs-on"],
                         {"group": "Develop CI", "labels": "llaminar-develop-avx2"})
        self.assertEqual(policy["concurrency"], {
            "group": "llaminar-develop-images", "cancel-in-progress": "true"})
        self.assertIn("--build-only", workflow)
        self.assertNotIn("prerequisites/", workflow)
        self.assertNotIn("prune_docker_build_cache", workflow)
        buildx = next(step for step in policy["jobs"]["develop-images"]["steps"]
                      if step.get("uses") == "docker/setup-buildx-action@v4")
        self.assertEqual(buildx["with"]["name"], "llaminar-develop")
        self.assertEqual(buildx["with"]["keep-state"], "true")
        self.assertNotIn("sudo", workflow)

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
        self.assertEqual(set(workflow["jobs"]), {"unit"})
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
                {"cpu_isa": "AVX2", "runner": "llaminar-develop-avx2", "pr_gpus": "CUDA+ROCm"},
                {"cpu_isa": "AVX512", "runner": "llaminar-avx512", "pr_gpus": "none"}]}})
        self.assertEqual(len({row["runner"] for row in job["strategy"]["matrix"]["include"]}), 2)
        self.assertEqual(f"{caller['name']} / {job['name']}",
                         "Develop PR / Unit + host (${{ matrix.cpu_isa }})")
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
                         {"opened", "reopened", "synchronize", "ready_for_review"})
        self.assertEqual(workflow["permissions"],
                         {"contents": "read", "pull-requests": "read"})
        self.assertEqual(set(workflow["jobs"]), {"arm"})
        job = workflow["jobs"]["arm"]
        self.assertIn("head.repo.full_name == github.repository", job["if"])
        self.assertIn("!github.event.pull_request.draft", job["if"])
        self.assertEqual(len(job["steps"]), 1)
        step = job["steps"][0]
        self.assertEqual(step["env"]["GH_TOKEN"], "${{ secrets.DEVELOP_AUTOMERGE_TOKEN }}")
        self.assertEqual(step["env"]["PR_NUMBER"],
                         "${{ github.event.pull_request.number }}")
        self.assertEqual(step["env"]["PR_HEAD_SHA"], "${{ github.event.pull_request.head.sha }}")
        self.assertIn('gh pr merge "$PR_NUMBER" --repo "$GITHUB_REPOSITORY" --auto --squash', step["run"])
        self.assertIn('--match-head-commit "$PR_HEAD_SHA"', step["run"])
        self.assertNotIn("checkout", str(job))

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
