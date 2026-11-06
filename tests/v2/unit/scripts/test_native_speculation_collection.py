#!/usr/bin/env python3
## @file test_native_speculation_collection.py
## @brief Synthetic collector contracts through the actual native launch adapter.
## The process boundary, native output and prerequisite admissions are mocked.
## No model, GPU, MPI worker, backend server or real inference is executed. Tests
## retain exact argument/environment/lifetime assertions without running kill
## operations: process retirement is intercepted by a strict mock as well.
"""Device-free collection contracts; all observations are synthetic fixtures."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/ci"))
collection = importlib.import_module("collect_native_speculation")
evaluation = importlib.import_module("evaluate_native_speculation")
benchmark = importlib.import_module("run_model_parity_benchmarks")
fixtures = importlib.import_module("test_native_speculation_evaluation")


class CollectionContractTests(unittest.TestCase):
    """Use synthetic native-shaped data and intercept every process/admission edge."""

    def setUp(self):
        """Create fake local deployment files, never a runnable inference binary."""
        self.fixture = fixtures.EvidenceContractTests("test_valid_costs_distinguish_ar_stock_and_useful_tokens")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.directory = self.fixture.directory
        self.plan_path = self.fixture.plan_path
        self.document = self.fixture.plan_document
        self.model = self.directory / "synthetic-model.gguf"
        self.model.write_bytes(b"not model weights; fixture storage only")
        self.hardware = {"synthetic": "passive cpu/pci identity"}
        self.execution = {"schema": collection.EXECUTION_SCHEMA,
                          "environment": {"PATH": "/usr/bin:/bin", "FROZEN_SETTING": "recorded"}, "roles": {}}
        self.policy_paths = {}
        self.roots = {}
        for name in ("stock", "candidate"):
            root = self.directory / ("source-" + name)
            (root / "src/v2").mkdir(parents=True)
            self.roots[name] = root
            for kind in ("release", "integration"):
                build = self.directory / (name + "-" + kind)
                build.mkdir()
                (build / "CMakeCache.txt").write_text(
                    f'CMAKE_HOME_DIRECTORY:INTERNAL={root / "src/v2"}\n'
                    f'CMAKE_BUILD_TYPE:STRING={"Release" if kind == "release" else "Integration"}\n'
                    'LLAMINAR_CPU_ISA:STRING=AVX512\n'
                    'HAVE_CUDA:BOOL=OFF\nHAVE_ROCM:BOOL=OFF\nCMAKE_CXX_COMPILER:FILEPATH=/synthetic/c++\n')
                if kind == "release":
                    binary = build / "llaminar2"
                    # Invalid shebang is intentional: even a lost Popen patch
                    # cannot accidentally run this as a model or shell program.
                    binary.write_text("#!/synthetic-nonexistent-interpreter\n" + name + "\n")
                    binary.chmod(0o700)
                else:
                    for filename in (".ninja_log", "build.ninja", "CTestTestfile.cmake"):
                        (build / filename).write_text("synthetic build boundary\n")
        for role in evaluation.Role:
            name = role.value
            implementation = "candidate" if role is evaluation.Role.CANDIDATE else "stock"
            binary = self.directory / (implementation + "-release/llaminar2")
            build = {**self.fixture.build,
                     "source_revision": ("b" if implementation == "candidate" else "a") * 40,
                     "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
            self.document["implementations"][name]["build"] = self.fixture.save(name + "-build.json", build)
            receipt = {"schema": 1, "mode": "model-free-prerequisites", "preflight_return_code": 0,
                       "certification_eligible": False, "preflight_tests": ["V2_Unit_Fixture", "V2_Integration_Fixture"],
                       "preflight_test_count": 2}
            policy = {"kind": "llaminar.orchestration-config", "schema_version": 1, "configuration": {
                "planning_mode": "Apply", "hostfile": "", "mpi_no_bootstrap": False, "deterministic": False,
                "model_path": str(self.model), "max_seq_len": 8192,
                "mtp_activation_policy": "Disabled" if role is evaluation.Role.AR else "Enabled",
                "mtp": {"enabled": role is not evaluation.Role.AR, "max_request_batch": 1}}}
            reference = self.fixture.save(name + "-apply.json", policy)
            self.policy_paths[str(self.directory / reference["path"])] = role
            self.execution["roles"][name] = {
                "binary": str(binary), "configuration": reference,
                "preflight_build_directory": str(self.directory / (implementation + "-integration")),
                "prerequisites": self.fixture.save(name + "-prerequisites.json", receipt)}
        self.repin_setup()
        self.admission = self.patch(collection.prerequisites, "reuse_unchanged_production_test_preflight",
                                    return_value=(0, 0.0, ("V2_Unit_Fixture", "V2_Integration_Fixture")))
        self.freshness = self.patch(collection.prerequisites, "validate_source_freshness")
        self.patch(collection, "source_revision", side_effect=lambda root: ("b" if root.name.endswith("candidate") else "a") * 40)
        self.hardware_probe = self.patch(benchmark, "hardware_identity", side_effect=lambda: copy.deepcopy(self.hardware))
        self.patch(collection.os, "sched_getaffinity", return_value={0, 1})
        self.launch = self.patch(benchmark.subprocess, "Popen", side_effect=self.synthetic_process)
        self.retire = self.patch(benchmark.os, "killpg")
        self.processes = []
        self.process_hook = None

    def patch(self, target, name, **kwargs):
        """Retain explicit mock ownership and restore all global process edges."""
        started = patch.object(target, name, **kwargs)
        value = started.start()
        self.addCleanup(started.stop)
        return value

    def repin_setup(self):
        """Freeze evaluator-owned synthetic setup bytes before each experiment."""
        self.document["deployment"] = {
            "model": self.fixture.save("model.json", {
                "files": [str(self.model)],
                "stat_identity": collection.model_identities({"cells": [{"model_files": [str(self.model)]}]})}),
            "hardware": self.fixture.save("hardware.json", self.hardware),
            "execution": self.fixture.save("execution.json", self.execution)}
        reference = self.fixture.save("plan.json", self.document)
        self.plan_sha = reference["sha256"]
        self.plan = evaluation.load_plan(self.plan_path, self.plan_sha)

    def freeze(self):
        """Use the actual freeze interface with mocked qualification observations."""
        return collection.freeze(self.plan_path, self.plan_sha, self.directory / "frozen")

    def run_collection(self, frozen=None, output="collected"):
        """Run the production collector with the real frozen file digest."""
        frozen = frozen or self.freeze()
        return collection.collect(frozen, evaluation.read_document(frozen)[1], self.directory / output)

    def synthetic_process(self, argv, **kwargs):
        """Intercept native Popen, emit synthetic native JSON and return a fake owner."""
        self.assertEqual(argv[1], "benchmark")
        self.assertTrue(kwargs["start_new_session"])
        self.assertNotIn("shell", kwargs)
        self.assertEqual(kwargs["env"]["FROZEN_SETTING"], "recorded")
        role = self.policy_paths[argv[argv.index("--config") + 1]]
        prompt = argv[argv.index("--prompt") + 1]
        request = next(row for row in self.document["requests"]
                       if row["workload"]["prompt_unit"] * row["workload"]["prompt_repetitions"] == prompt)
        work = request["workload"]
        self.assertEqual(kwargs["env"]["LLAMINAR_BENCHMARK_ITERATIONS"], str(work["measurement_iterations"]))
        self.assertEqual(kwargs["env"]["LLAMINAR_BENCHMARK_WARMUP_ITERATIONS"], str(work["warmup_iterations"]))
        output = Path(argv[argv.index("--benchmark-json-output") + 1])
        data = self.fixture.measurement(request, role)
        data["fixture_identity"] = str(output)
        output.write_text(json.dumps(data))
        process = Mock(pid=50_001 + len(self.processes))
        process.wait.return_value = 0
        self.processes.append(process)
        if self.process_hook is not None:
            self.process_hook(data, output, process)
        return process

    def test_complete_collection_is_directly_evaluator_ready(self):
        """All planned synthetic blocks traverse the canonical launcher once."""
        frozen = self.freeze()
        self.launch.assert_not_called()
        result = self.run_collection(frozen)
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["certification_eligible"])
        self.assertEqual(self.launch.call_count, 18)
        self.assertEqual(len(result["observations"]), 18)
        self.assertEqual(result["planned_observations"], 18)
        self.assertNotIn("active", result)
        self.retire.assert_not_called()
        report = evaluation.evaluate(self.plan, self.directory / "collected/evidence.json")
        self.assertEqual(report["correctness"]["emitted_token_ids"], "exact_ar_match")
        self.assertFalse(report["certification_eligible"])
        self.assertEqual(self.admission.call_count, 9)  # freeze + admission + completion, three roles
        for call in self.freshness.call_args_list:
            self.assertIn(call.kwargs["source_root"], self.roots.values())
        for process in self.processes:
            process.wait.assert_called_once_with(timeout=benchmark.e2e.parity.EXACT_CELL_TIMEOUT_SECONDS)

    def test_schedule_balances_roles_and_keeps_holdout_last(self):
        """Rotation is frozen, not selected from measured outcomes."""
        self.document["requests"].reverse()
        self.repin_setup()
        blocks = collection.schedule(self.plan)
        splits = [self.plan.requests[row["request"]]["split"] for row in blocks]
        self.assertEqual(splits, ["development"] * 3 + ["held_out"] * 3)
        for request in self.plan.requests:
            first_roles = [row["roles"][0] for row in blocks if row["request"] == request]
            self.assertEqual(set(first_roles), {role.value for role in evaluation.Role})

    def test_ambient_environment_is_not_inherited(self):
        """Hidden profiler/loader variables cannot leak into a frozen child env."""
        with patch.dict(os.environ, {"UNRECORDED_SECRET": "never-publish", "LD_PRELOAD": "/synthetic/profiler.so"}):
            self.run_collection()
        for call in self.launch.call_args_list:
            self.assertNotIn("UNRECORDED_SECRET", call.kwargs["env"])
            self.assertNotIn("LD_PRELOAD", call.kwargs["env"])

    def test_interruption_or_timeout_retires_only_the_fake_owner(self):
        """Exercise actual launcher cleanup with every kill operation mocked."""
        for error in (subprocess.TimeoutExpired(["synthetic"], 900), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                self.processes.clear()
                self.launch.reset_mock()
                self.retire.reset_mock()
                def interrupt(data, path, process):
                    process.wait.side_effect = [error, 0]
                self.process_hook = interrupt
                output = "failed-" + type(error).__name__
                frozen = self.directory / "frozen/collection.json"
                if not frozen.exists():
                    frozen = self.freeze()
                with self.assertRaises(type(error)):
                    self.run_collection(frozen, output)
                status = json.loads((self.directory / output / "status.json").read_text())
                self.assertEqual(status["status"], "failed")
                self.assertEqual(status["observations"], [])
                self.assertFalse((self.directory / output / "evidence.json").exists())
                self.assertEqual(self.launch.call_count, 1)
                self.retire.assert_called_once_with(self.processes[0].pid, benchmark.signal.SIGKILL)
                self.assertEqual(self.processes[0].wait.call_count, 2)

    def test_nonzero_missing_or_malformed_output_never_retries(self):
        """Retain first failure and never switch policy or fabricate a sample."""
        cases = {
            "nonzero": lambda data, path, process: setattr(process.wait, "return_value", 7),
            "missing": lambda data, path, process: path.unlink(),
            "malformed": lambda data, path, process: path.write_text('{"schema":"old"}'),
            "truncated": lambda data, path, process: path.write_text(json.dumps({**data, "iterations": []})),
        }
        frozen = self.freeze()
        for name, change in cases.items():
            with self.subTest(case=name):
                self.launch.reset_mock()
                self.process_hook = change
                with self.assertRaises((ValueError, KeyError, OSError, subprocess.SubprocessError)):
                    self.run_collection(frozen, name)
                self.assertEqual(self.launch.call_count, 1)
                status = json.loads((self.directory / name / "status.json").read_text())
                self.assertEqual(status["status"], "failed")
                self.assertIn("failure", status)
                self.assertFalse((self.directory / name / "evidence.json").exists())

    def test_input_change_during_process_fails_before_observation(self):
        """Model storage mutation cannot be accepted under the prelaunch digest."""
        def change(data, path, process):
            self.model.write_bytes(b"changed synthetic storage")
        self.process_hook = change
        with self.assertRaisesRegex(ValueError, "inputs changed during launch"):
            self.run_collection()
        status = json.loads((self.directory / "collected/status.json").read_text())
        self.assertEqual(status["observations"], [])
        self.assertEqual(self.launch.call_count, 1)

    def test_wrong_tokens_fail_at_first_drift(self):
        """Exact seeded output drift ends collection instead of running more blocks."""
        def change(data, path, process):
            if len(self.processes) == 2:
                for row in data["iterations"]:
                    row["generated_token_ids"][0] = 99
                path.write_text(json.dumps(data))
        self.process_hook = change
        with self.assertRaisesRegex(ValueError, "exact token mismatch"):
            self.run_collection()
        self.assertEqual(self.launch.call_count, 2)
        self.assertFalse((self.directory / "collected/evidence.json").exists())

    def test_non_executable_binary_cannot_enter_collection(self):
        """An existing digest alone does not make a native binary launchable."""
        path = Path(self.execution["roles"]["candidate"]["binary"])
        path.chmod(0o600)
        with self.assertRaisesRegex(ValueError, "native binary is not executable"):
            self.freeze()
        self.launch.assert_not_called()

    def test_stale_or_failed_prerequisites_block_every_launch(self):
        """Canonical receipt rejection is a hard admission boundary."""
        frozen = self.freeze()
        self.admission.side_effect = ValueError("synthetic stale canonical receipt")
        with self.assertRaisesRegex(ValueError, "stale canonical receipt"):
            self.run_collection(frozen)
        self.launch.assert_not_called()
        self.assertEqual(json.loads((self.directory / "collected/status.json").read_text())["status"], "failed")

    def test_changed_hardware_or_binary_blocks_admission(self):
        """Independently mutate physical identity and executable bytes."""
        frozen = self.freeze()
        original = copy.deepcopy(self.hardware)
        self.hardware["synthetic"] = "different host"
        with self.assertRaisesRegex(ValueError, "hardware identity"):
            self.run_collection(frozen, "hardware-changed")
        self.hardware = original
        Path(self.execution["roles"]["candidate"]["binary"]).write_text("changed binary bytes")
        with self.assertRaisesRegex(ValueError, "binary digest"):
            self.run_collection(frozen, "binary-changed")
        self.launch.assert_not_called()

    def test_changed_collection_plan_or_source_bindings_fail(self):
        """External digest and native source identity remain mandatory controls."""
        frozen = self.freeze()
        digest = evaluation.read_document(frozen)[1]
        with frozen.open("a") as handle:
            handle.write(" ")
        with self.assertRaisesRegex(ValueError, "artifact digest mismatch"):
            collection.collect(frozen, digest, self.directory / "bad-manifest")
        with patch.object(collection, "source_revision", return_value="0" * 40):
            with self.assertRaisesRegex(ValueError, "source revision differs"):
                self.run_collection(frozen, "source-changed")
        self.launch.assert_not_called()

    def test_prerequisite_backend_isa_and_compiler_must_match_release(self):
        """A CPU-only or other-ISA receipt cannot admit another build profile."""
        path = self.directory / "candidate-integration/CMakeCache.txt"
        original = path.read_text()
        for index, (old, new) in enumerate((
            ("HAVE_CUDA:BOOL=OFF", "HAVE_CUDA:BOOL=ON\nCMAKE_CUDA_COMPILER:FILEPATH=/synthetic/nvcc"),
            ("HAVE_ROCM:BOOL=OFF", "HAVE_ROCM:BOOL=ON\nCMAKE_HIP_COMPILER:FILEPATH=/synthetic/hipcc"),
            ("LLAMINAR_CPU_ISA:STRING=AVX512", "LLAMINAR_CPU_ISA:STRING=AVX2"),
            ("CMAKE_CXX_COMPILER:FILEPATH=/synthetic/c++", "CMAKE_CXX_COMPILER:FILEPATH=/different/c++"),
        )):
            with self.subTest(changed=old):
                path.write_text(original.replace(old, new))
                with self.assertRaisesRegex(ValueError, "ISA/backend/compiler contracts differ"):
                    collection.freeze(self.plan_path, self.plan_sha, self.directory / f"mismatched-build-{index}")
        self.launch.assert_not_called()

    def test_batch_and_activation_intent_have_independent_guards(self):
        """Matching enabled booleans cannot conceal another activation or batch policy."""
        binding = self.execution["roles"]["candidate"]
        reference = binding["configuration"]
        original = json.loads((self.directory / reference["path"]).read_text())
        for index, update in enumerate((
            lambda p: p.update(mtp_activation_policy="Automatic"),
            lambda p: p["mtp"].update(enabled=False),
            lambda p: p["mtp"].update(max_request_batch=2),
            lambda p: p["mtp"].update(max_request_batch=True),
        )):
            with self.subTest(case=index):
                policy = copy.deepcopy(original)
                update(policy["configuration"])
                binding["configuration"] = self.fixture.save(reference["path"], policy)
                self.repin_setup()
                with self.assertRaisesRegex(ValueError, "single-request AR/MTP role intent"):
                    collection.freeze(self.plan_path, self.plan_sha, self.directory / f"bad-role-{index}")
        self.launch.assert_not_called()

    def test_fixed_placement_sampler_and_single_request_contracts(self):
        """Every collector-specific placement constraint has its own mutation."""
        binding = self.execution["roles"]["candidate"]
        reference = binding["configuration"]
        path = self.directory / reference["path"]
        original = json.loads(path.read_text())
        cases = [
            ("planning_mode", "Automatic", "fixed local apply"),
            ("hostfile", "/synthetic/remote-hosts", "fixed local apply"),
            ("mpi_no_bootstrap", True, "fixed local apply"),
            ("deterministic", True, "fixed local apply"),
            ("model_path", "/wrong.gguf", "another primary model"),
            ("max_seq_len", 16384, "placement or non-MTP"),
        ]
        for index, (field, value, message) in enumerate(cases):
            with self.subTest(field=field):
                data = copy.deepcopy(original)
                data["configuration"][field] = value
                binding["configuration"] = self.fixture.save(reference["path"], data)
                self.repin_setup()
                with self.assertRaisesRegex(ValueError, message):
                    collection.freeze(self.plan_path, self.plan_sha, self.directory / f"bad-freeze-{index}")
        self.launch.assert_not_called()

    def test_profile_and_iteration_overrides_are_not_collected(self):
        """Each pre-existing forbidden environment predicate has an isolated fixture."""
        for index, key in enumerate(("LLAMINAR_PROFILE_KERNELS", "LLAMINAR_DETERMINISTIC", "LLAMINAR_BENCHMARK_ITERATIONS",
                                     "OMPI_COMM_WORLD_SIZE", "PMI_RANK", "PMIX_RANK", "NSYS_TEST", "NCU_TEST", "ROCPROF_TEST", "LD_PRELOAD",
                                     "LD_AUDIT", "CUDA_INJECTION64_PATH", "CUDA_PROFILE", "HSA_TOOLS_LIB", "ROCP_TOOL_LIBRARIES",
                                     "LLAMINAR_GPU_STAGE_TIMING", "LLAMINAR_TRACE_STAGES", "LLAMINAR_STAGE_DUMP_ENABLED",
                                     "LLAMINAR_MTP_DEVICE_PHASE_SNAPSHOTS", "LLAMINAR_DEBUG_KV_CACHE_SNAPSHOT")):
            with self.subTest(key=key):
                self.execution["environment"] = {"PATH": "/usr/bin:/bin", key: "1"}
                self.repin_setup()
                with self.assertRaisesRegex(ValueError, "overrides are not collectable"):
                    collection.freeze(self.plan_path, self.plan_sha, self.directory / f"bad-env-{index}")
        self.launch.assert_not_called()

    def test_known_instrumentation_is_rejected_even_when_set_to_zero(self):
        """Presence cannot be interpreted using another runtime's truthy grammar."""
        self.execution["environment"]["LLAMINAR_GPU_STAGE_TIMING"] = "0"
        self.repin_setup()
        with self.assertRaisesRegex(ValueError, "overrides are not collectable"):
            self.freeze()
        self.launch.assert_not_called()

    def test_native_iteration_clamp_and_receipt_shape_rejected_before_launch(self):
        """Unsupported counts and incomplete canonical receipts cannot enter timing."""
        self.document["requests"][0]["workload"]["measurement_iterations"] = 101
        self.repin_setup()
        with self.assertRaisesRegex(ValueError, "clamped at 100"):
            self.freeze()
        self.document["requests"][0]["workload"]["measurement_iterations"] = 3
        binding = self.execution["roles"]["ar"]
        reference = binding["prerequisites"]
        receipt = json.loads((self.directory / reference["path"]).read_text())
        receipt["preflight_tests"].pop()
        binding["prerequisites"] = self.fixture.save(reference["path"], receipt)
        self.repin_setup()
        with self.assertRaisesRegex(ValueError, "incomplete or failed"):
            self.freeze()
        self.launch.assert_not_called()

    def test_final_validation_failure_never_publishes_completed_evidence(self):
        """Completion still uses the actual evaluator, not a collector pass flag."""
        frozen = self.freeze()
        with patch.object(evaluation, "evaluate", side_effect=ValueError("synthetic final evaluator rejection")) as evaluate:
            with self.assertRaisesRegex(ValueError, "final evaluator rejection"):
                self.run_collection(frozen)
        evaluate.assert_called_once()
        self.assertEqual(self.launch.call_count, 18)
        self.assertFalse((self.directory / "collected/evidence.json").exists())
        self.assertFalse((self.directory / "collected/comparison.json").exists())
        status = json.loads((self.directory / "collected/status.json").read_text())
        self.assertEqual(status["status"], "failed")

    def test_last_admission_change_rejects_all_collected_observations(self):
        """An input change at completion is not excused by valid earlier rows."""
        frozen = self.freeze()
        original = collection.inspect_inputs
        calls = 0
        def changed_at_completion(*args):
            nonlocal calls
            calls += 1
            result = original(*args)
            if calls == 2:
                result[2]["cpu_affinity"] = [9]
            return result
        with patch.object(collection, "inspect_inputs", side_effect=changed_at_completion):
            with self.assertRaisesRegex(ValueError, "changed before completion"):
                self.run_collection(frozen)
        self.assertEqual(self.launch.call_count, 18)
        self.assertFalse((self.directory / "collected/evidence.json").exists())

    def test_cli_freeze_and_collect_failures_return_nonzero_without_overwrite(self):
        """The documented commands preserve raw failures and require external pins."""
        frozen = self.directory / "frozen/collection.json"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(collection.main(["freeze", "--plan", str(self.plan_path), "--plan-sha256", self.plan_sha,
                                              "--output", str(frozen.parent)]), 0)
            collection_sha = evaluation.read_document(frozen)[1]
            arguments = ["collect", "--collection", str(frozen), "--collection-sha256", collection_sha,
                         "--output", str(self.directory / "cli-results")]
            self.assertEqual(collection.main(arguments), 0)
            result = (self.directory / "cli-results/evidence.json").read_bytes()
            self.assertEqual(collection.main(arguments), 1)
            self.assertEqual((self.directory / "cli-results/evidence.json").read_bytes(), result)
            arguments[arguments.index("--collection-sha256") + 1] = "0" * 64
            arguments[-1] = str(self.directory / "bad-cli-results")
            self.assertEqual(collection.main(arguments), 1)
            self.assertFalse((self.directory / "bad-cli-results").exists())
        for observation in json.loads(result)["observations"]:
            self.assertFalse(Path(observation["benchmark"]["path"]).is_absolute())

    def test_output_directories_are_exclusive_and_outside_checkouts(self):
        """A rerun cannot overwrite success or invalidate source admission in place."""
        frozen = self.freeze()
        with self.assertRaises(FileExistsError):
            collection.freeze(self.plan_path, self.plan_sha, frozen.parent)
        with self.assertRaisesRegex(ValueError, "outside source checkouts"):
            collection.collect(frozen, evaluation.read_document(frozen)[1], self.roots["stock"] / "results")
        target = self.directory / "already-exists"
        target.mkdir()
        (target / "sentinel").write_text("keep")
        with self.assertRaises(FileExistsError):
            self.run_collection(frozen, "already-exists")
        self.assertEqual((target / "sentinel").read_text(), "keep")
        self.launch.assert_not_called()


class SourceFreshnessContractTests(unittest.TestCase):
    """Exercise the actual canonical freshness implementation on synthetic files."""

    def test_explicit_source_root_and_wrong_source_rejection(self):
        """Another checkout uses the same complete freshness rule, never a bypass."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            (root / "src/v2").mkdir(parents=True)
            source = root / "src/v2/synthetic.cpp"
            source.write_text("synthetic interpreted-source boundary")
            build = Path(directory) / "build"
            build.mkdir()
            (build / "CMakeCache.txt").write_text(f'CMAKE_HOME_DIRECTORY:INTERNAL={root / "src/v2"}\n')
            receipt = Path(directory) / "receipt.json"
            receipt.write_text(json.dumps({"preflight_completed_ns": 2**63 - 1}))
            with patch.object(collection.prerequisites.subprocess, "check_output", return_value=b"src/v2/synthetic.cpp\0") as listing:
                collection.prerequisites.validate_source_freshness(build, receipt, source_root=root)
                self.assertEqual(listing.call_args.kwargs["cwd"], root)
                with self.assertRaisesRegex(ValueError, "this checkout's configured build"):
                    collection.prerequisites.validate_source_freshness(build, receipt)
                receipt.write_text(json.dumps({"preflight_completed_ns": 1}))
                with self.assertRaisesRegex(ValueError, "source changed after"):
                    collection.prerequisites.validate_source_freshness(build, receipt, source_root=root)


if __name__ == "__main__":
    unittest.main()
