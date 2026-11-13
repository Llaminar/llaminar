#!/usr/bin/env python3
"""Device-free negative traces for exact compiler/dependency artifact ownership.

These typed fixtures certify validation/dispatch behavior only. Actual compiler
records and original-code controls are produced by the independent SDK build.
No test manufactures native readiness or launches an accelerator kernel.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import shlex
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
HELPERS = ROOT / "scripts/docker/llvm-lto-linker"

def load(name, path):
    """Import the actual builder/guard implementation into a device-free fixture."""
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value

artifact = load("lto_artifact_test_body", HELPERS / "artifact.py")
builder = load("lto_builder_test_body", HELPERS / "build.py")
guard = artifact.load_guard(ROOT / "scripts/build/gpu_spill_guard.py")

class OutputRoleTest(unittest.TestCase):
    """Original output identity distinguishes the collective from ordinary host targets."""
    def test_configure_preserves_parallel_device_link_with_other_offload_options(self):
        """CMake must retain the partition operand beside RCCL's other linker pair."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "receipt.json"
            receipt.write_text("authored compiler identity fixture")
            args = argparse.Namespace(pins=ROOT / "scripts/docker/rocm-release.env",
                linker_prefix=root / "fixed", spill_guard=ROOT / "scripts/build/gpu_spill_guard.py",
                architectures="gfx906;gfx1100", library=root / "librccl.so.1.0",
                sdk=root / "sdk", write_cache=root / "link.cmake")
            with patch.object(artifact, "compiler_artifact", return_value=(receipt,
                    {"architectures": ["gfx906", "gfx1100"], "linker_sha256": "authored"})):
                artifact.configure(args)
            hook = (root / "llaminar-final-allocation/target-link-dependencies.cmake").read_text()
            self.assertIn("[====[SHELL:-Xoffload-linker --lto-partitions=8]====]", hook)
            # The compiler-owned native check independently rejects a lost or
            # changed partition count; this protects its generated input policy.
            self.assertNotIn("--lto-partitions", args.write_cache.read_text())

    def test_shared_host_fmt_is_not_a_collective(self):
        argv = ["hipcc", "-shared", "-o", "_deps/fmt-build/libfmt.so.10.2.1", "fmt.o"]
        self.assertIn("-shared", argv)  # Original all-shared selection negative control.
        self.assertEqual(artifact.classify_output(argv, "/build/librccl.so.1.0", "/build"), artifact.ArtifactRole.HOST)

    def test_exact_collective_relative_and_absolute_output(self):
        for output in ("librccl.so.1.0", "/build/librccl.so.1.0"):
            self.assertEqual(artifact.classify_output(["hipcc", "-shared", "-o", output],
                             "/build/librccl.so.1.0", "/build"), artifact.ArtifactRole.COLLECTIVE)

    def test_foreign_collective_missing_output_and_wrong_link_role_reject(self):
        for argv in (["hipcc", "-shared", "-o", "/foreign/librccl.so.1.0"],
                     ["hipcc", "-shared", "-o", "librccl.so.2"], ["hipcc", "-shared"],
                     ["hipcc", "-o", "librccl.so.1.0"],
                     ["hipcc", "-shared", "-o", "a", "-o", "b"]):
            with self.subTest(argv=argv), self.assertRaises(RuntimeError):
                artifact.classify_output(argv, "/build/librccl.so.1.0", "/build")

    def test_host_never_borrows_native_driver_admission_and_collective_cannot_omit_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("ld.lld", "receipt"):
                (root / name).write_bytes(name.encode())
            spec = {"linker": str(root / "ld.lld"), "linker_sha256": artifact.digest(root / "ld.lld"),
                    "linker_receipt": str(root / "receipt"), "linker_receipt_sha256": artifact.digest(root / "receipt"),
                    "library": str(root / "librccl.so.1.0"), "architectures": ["gfx906", "gfx1100"],
                    "wrapper": str(root / "clang-linker-wrapper")}
            with patch.object(artifact.subprocess, "run") as native:
                result = artifact.authenticate_driver(spec, ["hipcc", "-shared", "-o", "libfmt.so"], root)
                self.assertEqual(result["role"], artifact.ArtifactRole.HOST.value)
                native.assert_not_called()
            response = argparse.Namespace(stdout="", stderr="", returncode=0)
            with patch.object(artifact.subprocess, "run", return_value=response), self.assertRaisesRegex(RuntimeError, "native architecture"):
                artifact.authenticate_driver(spec, ["hipcc", "-shared", "-o", "librccl.so.1.0"], root)

    def test_collective_target_entry_cannot_execute_a_foreign_host_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("ld.lld", "receipt"):
                (root / name).write_bytes(name.encode())
            spec = {"linker": str(root / "ld.lld"), "linker_sha256": artifact.digest(root / "ld.lld"),
                    "linker_receipt": str(root / "receipt"), "linker_receipt_sha256": artifact.digest(root / "receipt"),
                    "library": str(root / "librccl.so.1.0")}
            path = root / "spec.json"
            path.write_text(json.dumps(spec))
            argv = ["artifact.py", "gate", "--spec", str(path), "--", "hipcc", "-shared", "-o", str(root / "libother.so")]
            with patch.object(sys, "argv", argv), patch.object(artifact.os, "execv") as execute, \
                    self.assertRaisesRegex(RuntimeError, "non-collective output"):
                artifact.main()
            execute.assert_not_called()

    def test_stale_partition_cannot_fill_missing_current_record(self):
        """Old eight-file success cannot certify a later link producing only seven files."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {"record_root": str(root), "architectures": ["gfx906", "gfx1100"],
                    "library": str(root / "librccl.so.1.0")}
            foreign = Path(str(builder.final_record_base(spec["library"], "gfx999")) + ".part.7.obj.yaml")
            foreign.write_text("foreign target")
            for arch in spec["architectures"]:
                for index in range(8):
                    for kind in ("obj", "asm"):
                        Path(str(builder.final_record_base(spec["library"], arch)) + f".part.{index}.{kind}.yaml").write_text("old")
                self.assertEqual(len(artifact.object_records(spec, arch)), 8)  # Original stale-file acceptance.
            with artifact.record_transaction(spec):
                with self.assertRaisesRegex(RuntimeError, "Concurrent"):
                    with artifact.record_transaction(spec):
                        self.fail("Overlapping link/proof transaction admitted")
                artifact.withdraw_records(spec)
                self.assertEqual(foreign.read_text(), "foreign target")
                for arch in spec["architectures"]:
                    for index in range(7):
                        Path(str(builder.final_record_base(spec["library"], arch)) + f".part.{index}.obj.yaml").write_text("current")
                    with self.assertRaisesRegex(RuntimeError, "record is missing"):
                        artifact.object_records(spec, arch)
                    self.assertFalse(Path(str(builder.final_record_base(spec["library"], arch)) + ".part.7.asm.yaml").exists())
                    Path(str(builder.final_record_base(spec["library"], arch)) + ".part.7.obj.yaml").write_text("current")
                    self.assertEqual(len(artifact.object_records(spec, arch)), 8)


class WrapperDriverTest(unittest.TestCase):
    """Actual two-level argv must retain exact final linker, targets and records."""

    def setUp(self):
        """Create real executable file identities without invoking any native program."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for directory, names in ((self.root / "sdk/lib/llvm/bin", ("clang-linker-wrapper", "clang", "ld.lld")),
                                 (self.root / "fixed/bin", ("ld.lld",))):
            directory.mkdir(parents=True)
            for name in names:
                executable = directory / name
                executable.write_text("identity fixture: " + str(executable))
                executable.chmod(0o755)

    def fixture(self):
        """Return typed printed-command observations, never a compiler success proof."""
        wrapper = str(self.root / "sdk/lib/llvm/bin/clang-linker-wrapper")
        linker = str(self.root / "fixed/bin/ld.lld")
        output = "/build/librccl.so.1.0"
        arches = ["gfx906", "gfx1100"]
        outer = [wrapper, "--linker-path=" + str(self.root / "sdk/lib/llvm/bin/ld.lld"), "-o", output, "input.o"]
        inner = [[str(self.root / "sdk/lib/llvm/bin/clang"), "--no-default-config", "--target=amdgcn-amd-amdhsa",
                  "-mcpu=" + arch, "--ld-path=" + linker, "input-" + arch + ".bc"] for arch in arches]
        native = [[linker, "--lto-partitions=8", "-plugin-opt=mcpu=" + arch,
                   "-plugin-opt=opt-remarks-filename=" + str(builder.final_record_base(output, arch)),
                   "-plugin-opt=pass-remarks-analysis=stack-frame-layout|kernel-resource-usage"] for arch in arches]
        return wrapper, linker, output, arches, outer, inner, native

    def invoke(self, fixture, outer=None, inner=None, native=None, cwd="/build", argv_output=None,
               wrapper_diagnostics=None):
        """Mock only tool observations while executing the real admission body."""
        wrapper, linker, output, arches, expected_outer, expected_inner, expected_native = fixture
        outer = expected_outer if outer is None else outer
        inner = expected_inner if inner is None else inner
        native = expected_native if native is None else native
        texts = [shlex.join(outer), "\n".join(shlex.join(command) for command in inner)
                 if wrapper_diagnostics is None else wrapper_diagnostics]
        texts.extend(shlex.join(command) for command in native)
        responses = [argparse.Namespace(stdout="", stderr=text, returncode=0) for text in texts]
        with patch.object(builder.subprocess, "run", side_effect=responses) as calls:
            argv = ["hipcc", "-shared", "-o", output if argv_output is None else argv_output]
            result = builder.authenticate_wrapper(argv, cwd, wrapper, linker, arches, 8, output)
        self.assertEqual(calls.call_args_list[0].args[0], [*argv, "-###"])
        options_end = outer.index("--") if "--" in outer else len(outer)
        self.assertEqual(calls.call_args_list[1].args[0],
                         [outer[0], "--dry-run", "--save-temps", *outer[1:options_end],
                          "--wrapper-jobs=1", *outer[options_end:]])
        self.assertTrue(all(call.args[0][-1] == "-###" for index, call in enumerate(calls.call_args_list) if index != 1))
        for index, arch in enumerate(arches):
            original = next(command for command in inner if "-mcpu=" + arch in command)
            self.assertEqual(calls.call_args_list[index + 2].args[0], [*original, "-###"])
        return result

    def test_preview_serializes_after_original_job_policy_without_mutating_link_argv(self):
        """Last-argument policy wins over original parallel/jobserver values only in preview."""
        fixture = self.fixture()
        for jobs in ([], ["--wrapper-jobs=16"], ["--wrapper-jobs=jobserver"],
                     ["--wrapper-jobs=4", "--wrapper-jobs=16"],
                     ["--wrapper-jobs=16", "--", "--wrapper-jobs=99", "literal-input.o"]):
            outer = [*fixture[-3], *jobs]
            original = copy.deepcopy(outer)
            with self.subTest(jobs=jobs):
                result = self.invoke(fixture, outer=outer)
                self.assertEqual(outer, original)
                self.assertEqual(result["native_commands"], fixture[-1])
                self.assertTrue(all("--lto-partitions=8" in command
                                    for command in result["native_commands"]))
                self.assertFalse(any(arg.startswith("--wrapper-jobs=")
                                     for command in result["native_commands"] for arg in command))

    def test_torn_parallel_preview_commands_are_rejected_without_reassembly(self):
        """The actual two-target/one-line race shape must still fail strict architecture admission."""
        fixture = self.fixture()
        first, second = (shlex.join(command) for command in fixture[-2])
        # Two complete target identities appear in one physical line; the next
        # line is an orphaned input continuation, matching the captured failure.
        second = second + " extra-first.bc extra-last.bc"
        split = second.index(" extra-last.bc")
        torn = first + " " + second[:split] + "\n" + second[split:]
        with self.assertRaisesRegex(RuntimeError, "lost or repeated a native architecture compiler"):
            self.invoke(fixture, wrapper_diagnostics=torn)

    def test_complete_two_level_native_admission(self):
        fixture = self.fixture()
        result = self.invoke(fixture)
        self.assertEqual(result["native_commands"], fixture[-1])
        self.assertEqual(len(result["inner_driver_diagnostics"]), 2)
        self.assertNotEqual(builder.final_record_base(fixture[2], "gfx906"),
                            builder.final_record_base(fixture[2], "gfx1100"))

    def test_relative_record_identity_uses_original_link_cwd_without_rewriting_argv(self):
        """Same owned output resolves locally; wrong cwd/foreign sibling rejects."""
        fixture = self.fixture()
        outer, native = copy.deepcopy(fixture[-3]), copy.deepcopy(fixture[-1])
        outer[outer.index("-o") + 1] = "librccl.so.1.0"
        for command, arch in zip(native, fixture[3]):
            command[3] = ("-plugin-opt=opt-remarks-filename="
                          + str(builder.final_record_base("librccl.so.1.0", arch)))
        result = self.invoke(fixture, outer=outer, native=native, argv_output="librccl.so.1.0")
        self.assertEqual(result["native_commands"], native)
        with self.assertRaisesRegex(RuntimeError, "allocation policy"):
            self.invoke(fixture, outer=outer, native=native, cwd="/foreign", argv_output="librccl.so.1.0")
        foreign = copy.deepcopy(native)
        foreign[0][3] = foreign[0][3].replace("filename=", "filename=../foreign/")
        with self.assertRaisesRegex(RuntimeError, "allocation policy"):
            self.invoke(fixture, outer=outer, native=foreign, argv_output="librccl.so.1.0")
        duplicate = copy.deepcopy(native)
        duplicate[0].append(duplicate[0][3])
        with self.assertRaisesRegex(RuntimeError, "unique final allocation filename"):
            self.invoke(fixture, outer=outer, native=duplicate, argv_output="librccl.so.1.0")

    def test_generic_driver_basename_cannot_be_admitted_as_the_canonical_linker(self):
        """No old generic-driver alias or automatic alternative is supported."""
        fixture = list(self.fixture())
        fixture[1] = "/fixed/bin/lld"
        with self.assertRaisesRegex(RuntimeError, "canonical ld.lld basename"):
            self.invoke(tuple(fixture))

    def test_sdk_and_fixed_linker_aliases_authenticate_same_files_without_rewriting_commands(self):
        """Both configured-alias/reported-real and configured-real/reported-alias work."""
        fixture = self.fixture()
        sdk_alias, fixed_alias = self.root / "sdk-alias", self.root / "fixed-alias"
        sdk_alias.symlink_to(self.root / "sdk", target_is_directory=True)
        fixed_alias.symlink_to(self.root / "fixed", target_is_directory=True)
        def aliases(command):
            return [value.replace(str(self.root / "sdk"), str(sdk_alias))
                         .replace(str(self.root / "fixed"), str(fixed_alias)) for value in command]
        outer = aliases(fixture[-3])
        inner = [aliases(command) for command in fixture[-2]]
        native = [aliases(command) for command in fixture[-1]]
        result = self.invoke(fixture, outer=outer, inner=inner, native=native)
        self.assertEqual(result["native_commands"], native)
        configured = list(fixture)
        configured[0], configured[1] = outer[0], aliases([fixture[1]])[0]
        result = self.invoke(tuple(configured), outer=fixture[-3], inner=fixture[-2], native=fixture[-1])
        self.assertEqual(result["native_commands"], fixture[-1])

    def test_missing_foreign_and_basename_changing_aliases_reject(self):
        """Real symlinks cannot hide a foreign executable or change a driver role."""
        fixture = self.fixture()
        foreign = self.root / "foreign"
        foreign.mkdir()
        for name, original in (("clang-linker-wrapper", fixture[0]),
                               ("clang", str(self.root / "sdk/lib/llvm/bin/clang")), ("ld.lld", fixture[1])):
            target = foreign / name
            target.write_bytes(Path(original).read_bytes())  # Same bytes are a distinct authority.
            target.chmod(0o755)
        missing = self.root / "dangling"
        missing.symlink_to(self.root / "does-not-exist", target_is_directory=True)
        generic = self.root / "lld"
        generic.symlink_to(fixture[1])
        for kind in ("foreign_wrapper", "missing_wrapper", "foreign_clang", "missing_clang",
                     "foreign_option", "missing_option", "foreign_native", "missing_native",
                     "generic_option", "generic_native", "duplicate_option"):
            outer, inner, native = copy.deepcopy(fixture[-3:])
            if kind.endswith("wrapper"):
                outer[0] = str((foreign if kind.startswith("foreign") else missing) / "clang-linker-wrapper")
            elif kind.endswith("clang"):
                inner[0][0] = str((foreign if kind.startswith("foreign") else missing) / "clang")
            elif kind == "duplicate_option":
                inner[0].append(inner[0][4])
            else:
                target = generic if kind.startswith("generic") else (foreign if kind.startswith("foreign") else missing) / "ld.lld"
                if kind.endswith("option"):
                    inner[0][4] = "--ld-path=" + str(target)
                else:
                    native[0][0] = str(target)
            with self.subTest(kind=kind), self.assertRaises(RuntimeError):
                self.invoke(fixture, outer=outer, inner=inner, native=native)

        for role in ("wrapper", "linker"):
            for directory in (foreign, missing):
                configured = list(fixture)
                configured[0 if role == "wrapper" else 1] = str(directory / ("clang-linker-wrapper" if role == "wrapper" else "ld.lld"))
                with self.subTest(configured_role=role, configured_directory=directory), self.assertRaises(RuntimeError):
                    self.invoke(tuple(configured), outer=fixture[-3], inner=fixture[-2], native=fixture[-1])

    def test_missing_foreign_repeated_or_changed_driver_policy_rejects(self):
        cases = []
        fixture = self.fixture()
        for kind in ("missing_wrapper", "foreign_wrapper", "missing_inner", "repeated_inner",
                     "wrong_target", "missing_fixed_linker", "wrong_native_linker", "wrong_partitions",
                     "wrong_record_name", "missing_allocation_pass", "extra_native_target", "external_gold_plugin"):
            outer, inner, native = copy.deepcopy(fixture[-3:])
            if kind == "missing_wrapper":
                outer = []
            elif kind == "foreign_wrapper":
                outer[0] = "/foreign/clang-linker-wrapper"
            elif kind == "missing_inner":
                inner.pop()
            elif kind == "repeated_inner":
                inner[1] = copy.deepcopy(inner[0])
            elif kind == "wrong_target":
                inner[0][3] = "-mcpu=gfx999"
            elif kind == "missing_fixed_linker":
                inner[0][4] = "--ld-path=/foreign/ld.lld"
            elif kind == "wrong_native_linker":
                native[0][0] = str(self.root / "sdk/lib/llvm/bin/ld.lld")
            elif kind == "wrong_partitions":
                native[0][1] = "--lto-partitions=2"
            elif kind == "wrong_record_name":
                native[0][3] = "-plugin-opt=opt-remarks-filename=shared.opt.ld.yaml"
            elif kind == "missing_allocation_pass":
                native[0][-1] = "-plugin-opt=pass-remarks-analysis=other"
            elif kind == "extra_native_target":
                native[0] = native[0] + ["-plugin-opt=mcpu=gfx999"]
            elif kind == "external_gold_plugin":
                native[0].extend(["-plugin", "/sdk/lib/LLVMgold.so"])
            cases.append((kind, outer, inner, native))
        for kind, outer, inner, native in cases:
            with self.subTest(kind=kind), self.assertRaises(RuntimeError):
                self.invoke(fixture, outer=outer, inner=inner, native=native)

class CompilerProofTest(unittest.TestCase):
    """Reject incomplete partition records and changed compiler observations."""
    @staticmethod
    def gvn_filter_proof():
        """Typed validation traces only; the builder supplies actual linked execution."""
        variants = []
        for variant in ("original", "patched"):
            cases = []
            for name, stream, selection, terminal in (
                    ("filtered", True, "stack-frame-layout", False),
                    ("gvn_selected", True, "gvn", False), ("unfiltered", True, "", False),
                    ("terminal_override", True, "stack-frame-layout", True),
                    ("terminal_only", False, "", True), ("disabled", False, "", False)):
                constructed = int(name != "disabled" and (name != "filtered" or variant == "original"))
                cases.append({"name": name, "stream_enabled": stream, "filter": selection,
                    "terminal_gvn": terminal, "constructed": constructed,
                    "messages": ["actual diagnostic fixture"] if constructed else [],
                    "serialized": "LoadClobbered fixture" if name in ("gvn_selected", "unfiltered") else "",
                    "optimized_ir": "fixed optimized witness IR"})
            variants.append({"variant": variant, "gvn_object_sha256": ("a" if variant == "original" else "b") * 64,
                "scalar_archive_sha256": "c" * 64, "executable_sha256": "d" * 64,
                "observation": {"schema": 1, "cases": cases}})
        return {"schema": 1, "variants": variants}

    @staticmethod
    def report():
        rows = []
        emitted = sorted([*builder.KERNELS, "retained_helper_a", "retained_helper_b"])
        for arch in ("gfx906", "gfx1100"):
            for count in (2, 8):
                kernels = [{"name": name, "sgpr": 0, "vgpr": 0, "private_bytes": 0, "dynamic_stack": False}
                           for name in sorted(builder.KERNELS)]
                for variant in ("original", "patched"):
                    repaired = variant == "patched"
                    rows.append({"architecture": arch, "partitions": count, "variant": variant,
                        "metadata_partitions": count, "kernels": 16, "code_sha256": "1" * 64,
                        "emitted_functions": copy.deepcopy(emitted),
                        "allocation_functions": len(emitted) if repaired else 0,
                        "record_files": count if repaired else 0, "assembly_record_files": count if repaired else 0,
                        "kernels_metadata": copy.deepcopy(kernels),
                        "allocations": {name: {"instances": 1, "slots": []} for name in emitted} if repaired else {},
                        "guard_accepted_spill_free_frames": repaired})
        return {"schema": 1, "certificate": True, "architectures": ["gfx906", "gfx1100"],
                "toolchain": {"linker_sha256": "4" * 64, "gvn_filter_proof": CompilerProofTest.gvn_filter_proof()},
                "observations": rows}

    def test_actual_guard_proof_rejects_excluded_work_lost_remarks_or_changed_ir(self):
        """The executable original/repaired observations are mandatory, not a source scan."""
        changes = [lambda p: p.pop("variants"),
                   lambda p: p["variants"].append(copy.deepcopy(p["variants"][0])),
                   lambda p: p["variants"][0].pop("gvn_object_sha256"),
                   lambda p: p["variants"][1].update(gvn_object_sha256=p["variants"][0]["gvn_object_sha256"]),
                   lambda p: p["variants"][0]["observation"]["cases"][0].update(constructed=0, messages=[]),
                   lambda p: p["variants"][1]["observation"]["cases"][0].update(constructed=1, messages=["unrequested"]),
                   lambda p: p["variants"][1]["observation"]["cases"][1].update(constructed=0, messages=[]),
                   lambda p: p["variants"][1]["observation"]["cases"][2].update(serialized=""),
                   lambda p: p["variants"][1]["observation"]["cases"][3].update(messages=["changed"]),
                   lambda p: p["variants"][1]["observation"]["cases"][4].update(optimized_ir="changed"),
                   lambda p: p["variants"][1]["observation"]["cases"][5].update(constructed=1, messages=["disabled"])]
        for change in changes:
            proof = self.gvn_filter_proof()
            change(proof)
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                builder.verify_gvn_filter_proof(proof)
        report = self.report()
        report["toolchain"].pop("gvn_filter_proof")
        with self.assertRaises(RuntimeError):
            builder.verify(report, ["gfx906", "gfx1100"], guard=guard)

    def test_complete_typed_partition_trace(self):
        builder.verify(self.report(), ["gfx906", "gfx1100"], guard=guard)

    def test_complete_emitted_roster_includes_helpers_and_rejects_missing_extra_records(self):
        """A non-kernel function needs the same allocation proof as every kernel."""
        report = self.report()
        self.assertGreater(len(report["observations"][1]["emitted_functions"]), len(builder.KERNELS))
        changes = [lambda rows: rows[1].pop("emitted_functions"),
                   lambda rows: rows[1]["emitted_functions"].append(rows[1]["emitted_functions"][0]),
                   lambda rows: rows[1]["emitted_functions"].reverse(),
                   lambda rows: rows[1]["emitted_functions"].append(""),
                   lambda rows: rows[0]["emitted_functions"].remove("retained_helper_a"),
                   lambda rows: rows[1]["allocations"].pop("retained_helper_a"),
                   lambda rows: rows[1].update(allocation_functions=16),
                   lambda rows: rows[1]["allocations"].update(unemitted_helper={"instances": 1, "slots": []})]
        for change in changes:
            report = self.report()
            change(report["observations"])
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                builder.verify(report, ["gfx906", "gfx1100"], guard=guard)
        # Even if the scalar count is edited to hide the omitted helper, the
        # observed ELF roster still requires its exact allocation record.
        report = self.report()
        repaired = report["observations"][1]
        repaired["allocations"].pop("retained_helper_a")
        repaired["allocation_functions"] -= 1
        with self.assertRaises(RuntimeError):
            builder.verify(report, ["gfx906", "gfx1100"], guard=guard)
        report = self.report()
        for row in report["observations"][:2]:
            row["emitted_functions"].remove("partition_probe_0")
        with self.assertRaises(RuntimeError):
            builder.verify(report, ["gfx906", "gfx1100"], guard=guard)

    def test_actual_symbol_table_parser_covers_every_defined_function_binding(self):
        """Use the SDK JSON schema; undefined functions and data are separate."""
        def symbol(name, kind=2, section=8, binding=0):
            return {"Symbol": {"Name": {"Name": name, "Value": 1},
                    "Type": {"Name": "Function" if kind == 2 else "Object", "Value": kind},
                    "Section": {"Name": ".text" if section else "Undefined", "Value": section},
                    "Binding": {"Name": "Weak" if binding == 2 else "Local", "Value": binding}}}
        objects = [{"Symbols": [symbol("local_helper"), symbol("weak_helper", binding=2),
                    symbol("global_kernel", binding=1), symbol("undefined_function", section=0),
                    symbol("global_kernel.kd", kind=1)]}]
        self.assertEqual(builder.parse_emitted_functions(objects),
                         ["global_kernel", "local_helper", "weak_helper"])
        changes = [lambda value: value.append(copy.deepcopy(value[0])),
                   lambda value: value[0].pop("Symbols"),
                   lambda value: value[0]["Symbols"].append(copy.deepcopy(value[0]["Symbols"][0])),
                   lambda value: value[0]["Symbols"][0]["Symbol"]["Type"].pop("Value"),
                   lambda value: value[0]["Symbols"][0]["Symbol"]["Section"].update(Value=True),
                   lambda value: value[0]["Symbols"][0]["Symbol"]["Name"].update(Name="")]
        for change in changes:
            value = copy.deepcopy(objects)
            change(value)
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                builder.parse_emitted_functions(value)

    def test_missing_duplicate_or_changed_partition_trace_rejects(self):
        for change in (lambda rows: rows.pop(), lambda rows: rows.append(copy.deepcopy(rows[0])),
                       lambda rows: rows[1].update(record_files=1),
                       lambda rows: rows[1].update(code_sha256="2" * 64),
                       lambda rows: rows[0].update(allocation_functions=16),
                       lambda rows: rows[1]["allocations"].pop("partition_probe_7"),
                       lambda rows: rows[1]["kernels_metadata"][0].update(vgpr="unknown")):
            report = self.report()
            change(report["observations"])
            with self.assertRaises(RuntimeError):
                builder.verify(report, ["gfx906", "gfx1100"], guard=guard)

class FinalResourceEvidenceTest(unittest.TestCase):
    """External dependency evidence reports physical resources without certifying separate gates."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.library = self.root / "library"
        self.receipt = self.root / "receipt"
        self.library.write_bytes(b"exact selected library identity fixture")
        self.receipt.write_bytes(b"exact selected build receipt fixture")
        self.args = argparse.Namespace(report=self.root / "proof.json", library=self.library,
            receipt=self.receipt, spill_guard=ROOT / "scripts/build/gpu_spill_guard.py", architectures="gfx906;gfx1100")

    def tearDown(self):
        self.temporary.cleanup()

    def report(self):
        rows = []
        for arch in ("gfx906", "gfx1100"):
            rows.append({"architecture": arch, "partitions": 8,
                "record_files": [{"sha256": "3" * 64, "bytes": 17} for _ in range(8)],
                "kernels": [{"name": "kernel", "sgpr": 1, "vgpr": 0, "private_bytes": 16, "dynamic_stack": False}],
                "allocations": {"kernel": {"instances": 1, "slots": [{"kind": "Variable", "size": 16}]}}})
        for row in rows:
            functions = {name: guard.FunctionAllocation([guard.StackSlot(**slot) for slot in value["slots"]],
                         value["instances"]) for name, value in row["allocations"].items()}
            row["resource_usage"] = artifact.resource_usage(functions, [guard.KernelSpills(**v) for v in row["kernels"]])
        canonical = artifact.pins(HELPERS.parent / "rocm-release.env")
        compiler = CompilerProofTest.report()
        compiler.update(source_inputs={
            path.name: artifact.digest(path) for path in
            (HELPERS / "build.py", HELPERS / "main.cpp", HELPERS / "PressureProbe.hip", HELPERS / "GVNRemarkProbe.cpp", self.args.spill_guard)},
            llvm_revision=canonical["LLAMINAR_ROCM_LLVM_REVISION"],
            llvm_patch_sha256=artifact.digest(HELPERS.parent / "patches/llvm-parallel-lto-remarks.patch"),
            backend_source_sha256=canonical["LLAMINAR_ROCM_LLVM_LTO_BACKEND_SHA256"],
            driver_source_sha256=canonical["LLAMINAR_ROCM_LLVM_LLD_DRIVER_SHA256"],
            gvn_source_sha256=canonical["LLAMINAR_ROCM_LLVM_GVN_SOURCE_SHA256"],
            gvn_patch_sha256=canonical["LLAMINAR_ROCM_LLVM_GVN_PATCH_SHA256"])
        proof_json = json.dumps(compiler)
        compiler_receipt = {"schema": 1, "linker_sha256": "4" * 64, "architectures": ["gfx906", "gfx1100"],
                            "proof_sha256": hashlib.sha256(proof_json.encode()).hexdigest()}
        receipt_json = json.dumps(compiler_receipt)
        return {"schema": 2, "scope": "external-rccl-resource-evidence",
                "resource_evidence_complete": True, "zero_spill_qualified": False,
                "native_runtime_qualified": False, "performance_qualified": False,
                "producer_sha256": artifact.digest(HELPERS / "artifact.py"), "library_sha256": artifact.digest(self.library),
                "receipt_sha256": artifact.digest(self.receipt), "architectures": ["gfx906", "gfx1100"],
                "observations": rows, "linker_sha256": "4" * 64,
                "linker_receipt_sha256": hashlib.sha256(receipt_json.encode()).hexdigest(),
                "compiler_artifact": {"receipt_json": receipt_json, "proof_json": proof_json}}

    def verify(self, report):
        self.args.report.write_text(json.dumps(report))
        artifact.verify_proof(self.args)

    def test_intentional_private_frames_and_register_moves_remain_legal(self):
        self.verify(self.report())

    def test_external_spills_reported_while_project_kernel_guard_remains_fatal(self):
        report = self.report()
        for row in report["observations"]:
            row["allocations"]["kernel"]["slots"] = [{"kind": "Spill", "size": 4}]
            functions = {"kernel": guard.FunctionAllocation([guard.StackSlot("Spill", 4)], 1)}
            kernels = [guard.KernelSpills(**value) for value in row["kernels"]]
            row["resource_usage"] = artifact.resource_usage(functions, kernels)
            self.assertEqual(row["resource_usage"]["physical_spills"]["bytes"], 4)
            with self.assertRaises(guard.SpillGuardError):
                guard.audit_device_allocations(functions, kernels)
        self.verify(report)
        self.assertFalse(report["zero_spill_qualified"])
        self.assertFalse(report["native_runtime_qualified"])
        self.assertFalse(report["performance_qualified"])

    def test_missing_conflicting_or_misreported_resources_reject(self):
        changes = [lambda r: r.update(schema=1),
                   lambda r: r.update(zero_spill_qualified=True),
                   lambda r: r.update(native_runtime_qualified=True),
                   lambda r: r.update(performance_qualified=True),
                   lambda r: r["observations"][0].pop("resource_usage"),
                   lambda r: r["observations"][0]["resource_usage"]["physical_spills"].update(bytes=1),
                   lambda r: r.pop("producer_sha256"),
                   lambda r: r.update(producer_sha256="0" * 64),
                   lambda r: r.pop("compiler_artifact"),
                   lambda r: r["compiler_artifact"].update(proof_json="{}"),
                   lambda r: r.update(linker_sha256="0" * 64),
                   lambda r: r.update(library_sha256="0" * 64),
                   lambda r: r.update(receipt_sha256="0" * 64),
                   lambda r: r["observations"].pop(),
                   lambda r: r["observations"][0]["record_files"].pop(),
                   lambda r: r["observations"][0]["record_files"][0].pop("sha256"),
                   lambda r: r["observations"][0]["kernels"].append(copy.deepcopy(r["observations"][0]["kernels"][0])),
                   lambda r: r["observations"][0]["kernels"][0].pop("private_bytes"),
                   lambda r: r["observations"][0]["kernels"][0].update(vgpr="unknown"),
                   lambda r: r["observations"][0]["allocations"].clear(),
                   lambda r: r["observations"][0]["allocations"]["kernel"]["slots"][0].update(kind="Spill"),
                   lambda r: r["observations"][0]["allocations"]["kernel"]["slots"][0].update(kind="Unknown")]
        for change in changes:
            report = self.report()
            change(report)
            with self.subTest(change=change), self.assertRaises((RuntimeError, guard.SpillGuardError)):
                self.verify(report)

if __name__ == "__main__":
    unittest.main()
