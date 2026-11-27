#!/usr/bin/env python3
"""Static selection boundaries and explicit GNU14 native acquisition controls.

The native class is never silently skipped: its separate integration registration
fails if GNU14 is missing. Parser controls are not native qualification evidence.
"""
import contextlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[2] / "cmake" / "CPUNativeTileRegisterEvidence.py"
SPEC = importlib.util.spec_from_file_location("static_checker", PATH)
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


class StaticSelectionBoundaryTest(unittest.TestCase):
    """Compiler-free controls for non-authority manifests and bounded input grammars."""

    def test_manifest_and_enum_cannot_construct_authority(self):
        for key in (None, {}, "static-executable", "0" * 64):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "native verifier"):
                C.TypedStaticExecutableSelection(key, None, "f", {}, {})
        u = C.Unit("x", {"f": ["ret"]}, "", {"f": "weak"}, frozenset(), frozenset(), {"f": "default"})
        self.assertIsNotNone(C.CallModel.select(u, u, "f", dict(binding="static-executable")))
        self.assertIsNotNone(C.CallModel.select(u, u, "f", dict(binding="direct", nonpreemptible="static-executable",
                                                                             link_evidence="0" * 64)))

    def test_rewriting_options_include_forwarded_forms(self):
        for option in ("-flto", "-flto=auto", "--gc-sections", "--icf=all", "-r", "--relocatable", "-shared"):
            for arg in (option, "-Wl," + option):
                with self.subTest(arg=arg), self.assertRaisesRegex(ValueError, "unsupported static"):
                    C._static_options([arg])

    def test_response_contents_are_inputs_and_recursive_response_fails(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "args").write_text("-Wl,--gc-sections")
            files = {}
            expanded = C._static_response(["@args"], root, files)
            self.assertEqual(files[str((root / "args").resolve())], C._static_hash((root / "args").read_bytes()))
            with self.assertRaisesRegex(ValueError, "unsupported static"):
                C._static_options(expanded)
            (root / "args").write_text("@args")
            with self.assertRaisesRegex(ValueError, "recursive"):
                C._static_response(["@args"], root, {})

    def test_failed_capture_retires_old_recipe(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            proof = root / "selection-recipe.json"
            proof.write_text('{"success": true}')
            with self.assertRaises(ValueError):
                C.capture_static_link(["compiler", "-flto"], root, root, root)
            self.assertFalse(proof.exists())

    def test_thin_archive_and_truncated_elf_fail(self):
        with self.assertRaisesRegex(ValueError, "regular archive"):
            C._static_members(b"!<thin>\n")
        with self.assertRaisesRegex(ValueError, "ELF64"):
            C._static_elf(b"not elf")

    def test_archive_duplicate_identity_retains_ordinal_and_offset(self):
        def member(body):
            return b"same.o/         " + b"0           0     0     100644  " + str(len(body)).encode().ljust(10) + b"`\n" + body
        members = C._static_members(b"!<arch>\n" + member(b"aa") + member(b"bb"))
        self.assertEqual([m[3] for m in members], ["same.o", "same.o"])
        self.assertNotEqual(members[0][:2], members[1][:2])
        self.assertEqual([m[4] for m in members], [b"aa", b"bb"])


    def test_emulation_comes_from_actual_linker_inventory(self):
        observed = "GNU ld (test)\n  Supported emulations:\n   aarch64linux\n   aarch64elf\n"
        for tokens in (["-maarch64linux"], ["-m", "aarch64linux"]):
            self.assertEqual(C._static_emulation(tokens, observed), "aarch64linux")
        for tokens in ([], ["-m"], ["-mforeign"], ["-maarch64linux", "-maarch64elf"]):
            with self.assertRaises(ValueError):
                C._static_emulation(tokens, observed)

    def test_wrapped_map_kept_and_discarded_are_distinct(self):
        name = ".text." + "long_mangled_name" * 12
        text = ("Discarded input sections\n " + name + "\n  0x0 0x8 loser.a(loser.o)\n"
                "Memory Configuration\nLinker script and memory map\n " + name +
                "\n  0x1000 0x8 winner.a(winner.o)\n")
        discarded, kept = C._static_map_records(text)
        self.assertEqual(discarded, [(name, 0, 8, "loser.a(loser.o)")])
        self.assertEqual(kept, [(name, 4096, 8, "winner.a(winner.o)")])
        for broken in (text + "Linker script and memory map", text.replace("  0x1000", "not an address")):
            with self.assertRaises(ValueError):
                C._static_map_records(broken)

    def test_trace_member_forms_canonicalize_and_unknown_records_fail(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root).resolve()
            header = b"m.o/            " + b"0           0     0     100644  " + b"2         " + b"`\n"
            archive = root / "a.a"
            archive.write_bytes(b"!<arch>\n" + header + b"aa")
            for line in (str(archive), f"({archive})m.o", f"{archive}(m.o)"):
                self.assertEqual(C._static_trace_paths(line, root), {archive})
            for line in (f"{archive}(absent.o)", "ld: warning: unknown", f"({archive})m.o(extra)"):
                with self.assertRaises(ValueError):
                    C._static_trace_paths(line, root)


class StaticSelectionNativeTest(unittest.TestCase):
    """Real linker acquisition; host acquisition is not AArch64 selection qualification."""

    def test_native_gnu14_capture_and_nonbfd_gc_controls(self):
        compiler = os.environ.get("LLAMINAR_STATIC_TEST_GXX", "g++-14")
        compiler = shutil.which(compiler)
        self.assertIsNotNone(compiler, "native static acquisition requires GNU14 g++ (LLAMINAR_STATIC_TEST_GXX)")
        base = os.environ.get("LLAMINAR_STATIC_TEST_ARTIFACTS")
        if base:
            Path(base).mkdir(parents=True, exist_ok=True)
        with contextlib.nullcontext(tempfile.mkdtemp(prefix="static-native-", dir=base)) as root:
            root = Path(root)
            print(f"Retained native static artifacts: {root}", flush=True)
            (root / "f.cpp").write_text('extern "C" __attribute__((weak)) int f(){return 7;}\n')
            (root / "g.cpp").write_text('extern "C" __attribute__((weak)) int f(){return 9;}\n')
            (root / "main.cpp").write_text('extern "C" int f(); int main(){return f();}\n')
            for name in ("f", "g", "main"):
                subprocess.run([compiler, "-O2", "-ffunction-sections", "-c", str(root / (name + ".cpp")),
                                "-o", str(root / (name + ".o"))], check=True)
            ar = subprocess.check_output([compiler, "-print-prog-name=ar"], text=True).strip()
            for name in ("f", "g"):
                subprocess.run([ar, "rcs", str(root / (name + ".a")), str(root / (name + ".o"))], check=True)
            args = [compiler, "-fuse-ld=bfd", str(root / "main.o"), "-Wl,--whole-archive",
                    str(root / "f.a"), str(root / "g.a"), "-Wl,--no-whole-archive",
                    f"-Wl,--dependency-file={root / 'production.d'}", "-o", str(root / "app")]
            subprocess.run(args, check=True)
            result = C.capture_static_link(args, root, root / "app", root / "capture")
            self.assertIn(str(root / "f.a"), result["files"])
            self.assertTrue(any("crt" in p for p in result["files"]))
            for flag in ("-Wl,--gc-sections", "-Wl,--icf=all", "-flto"):
                with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "unsupported static"):
                    C.capture_static_link(args + [flag], root, root / "app", root / "rejected")
            # A forged map/manifest cannot be passed to the typed constructor.
            with self.assertRaisesRegex(ValueError, "native verifier"):
                C.TypedStaticExecutableSelection(result, None, "f", {}, {})


if __name__ == "__main__":
    unittest.main()
