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
from unittest import mock

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
            without_dep = [a for a in args if not a.startswith("-Wl,--dependency-file=")]
            duplicated = args + [next(a for a in args if a.startswith("-Wl,--dependency-file="))]
            for bad in (without_dep, duplicated):
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    C.capture_static_link(bad, root, root / "app", root / "bad-dependency-token")
            for flag in ("-Wl,--gc-sections", "-Wl,--icf=all", "-flto"):
                with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "unsupported static"):
                    C.capture_static_link(args + [flag], root, root / "app", root / "rejected")
            # A forged map/manifest cannot be passed to the typed constructor.
            with self.assertRaisesRegex(ValueError, "native verifier"):
                C.TypedStaticExecutableSelection(result, None, "f", {}, {})


    def test_actual_aarch64_comdat_selection_and_negative_boundaries(self):
        compiler = shutil.which(os.environ.get("LLAMINAR_STATIC_TEST_GXX", "g++-14"))
        self.assertIsNotNone(compiler, "native selected-object test requires GNU14")
        self.assertEqual(os.uname().machine, "aarch64", "selected-object native gate requires real AArch64")
        base = os.environ.get("LLAMINAR_STATIC_TEST_ARTIFACTS")
        if base:
            Path(base).mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="static-comdat-", dir=base))
        print(f"Retained native COMDAT selection: {root}", flush=True)
        template = "template<int N> __attribute__((noinline,noipa)) int chosen(){return N;}\n"
        for name, wrapper in (("f", "use"), ("g", "other")):
            (root / (name + ".cpp")).write_text(template + f"int {wrapper}(){{return chosen<7>();}}\n")
        (root / "main.cpp").write_text("extern int use(); int main(){return use();}\n")
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
        obj = (root / "f.o").read_bytes()
        objdump = subprocess.check_output([compiler, "-print-prog-name=objdump"], text=True).strip()
        listing, _, _ = C.disassemble(objdump, root / "f.o")
        (root / "f.disassembly").write_text(listing)
        image = C.read_object(obj)
        unit = C.make_unit("winner", image, C.normalize_disassembly(listing, image))
        target = "_Z6chosenILi7EEiv"
        self.assertEqual(unit.functions[target], "weak")
        # Original direct vocabulary rejects this actual default-visible winner.
        self.assertEqual(C.CallModel.select(unit, unit, target, dict(binding="direct")),
                         "direct binding without a non-preemptibility proof")
        authority = C.verify_static_executable_selection(args, root, root / "app", root / "valid", unit, obj, obj, target)
        model = C.CallModel({"winner": unit}, {target: authority})
        self.assertEqual(model.resolve(unit, target), ("helper", unit, target))
        discarded, kept = C._static_map_records((root / "valid/native.map").read_text())
        self.assertTrue(any(r[0] == ".text." + target and "g.a(g.o)" in r[3] for r in discarded))
        self.assertTrue(any(r[0] == ".text." + target and "f.a(f.o)" in r[3] for r in kept))
        original_capture = C.capture_static_link
        original_app = (root / "app").read_bytes()

        def inject(label, edit, expected):
            # Fault injection happens after a REAL native capture. No invented
            # manifest, digest, boolean or direct-proof enum creates authority.
            def captured(*a, **kw):
                recipe = original_capture(*a, **kw)
                edit(Path(a[3]))
                return recipe
            with mock.patch.object(C, "capture_static_link", captured):
                with self.subTest(boundary=label), self.assertRaisesRegex(ValueError, expected):
                    C.verify_static_executable_selection(args, root, root / "app", root / label, unit, obj, obj, target)
            # Keep injected output bytes as evidence; restore only the test fixture
            # executable for the next independent adversarial experiment.
            (root / label / "injected.elf").write_bytes((root / "app").read_bytes())
            (root / "app").write_bytes(original_app)

        def map_edit(change):
            def edit(directory):
                p = directory / "native.map"
                p.write_text(change(p.read_text()))
            return edit
        inject("wrong-winner", map_edit(lambda t: t.replace("f.a(f.o)", "g.a(g.o)")), "kept member/section")
        inject("stale-map", map_edit(lambda t: t + "\nforged trailing observation\n"), "artifact changed")
        inject("discarded-only", map_edit(lambda t: t.split("Linker script and memory map")[0] +
               "Linker script and memory map\n"), "ambiguous/missing kept owner")
        def elf_edit(change):
            def edit(directory):
                data = bytearray((root / "app").read_bytes())
                change(data)
                (root / "app").write_bytes(data)
            return edit
        def clear_pie(d):
            headers, _ = C._static_elf(d)
            self.assertEqual(int.from_bytes(d[16:18], "little"), 3)
            for h in headers:
                if h["type"] == 6:
                    for b in range(h["offset"], h["offset"]+h["size"], 16):
                        if int.from_bytes(d[b:b+8], "little") == 0x6ffffffb:
                            flags = int.from_bytes(d[b+8:b+16], "little") & ~0x8000000
                            d[b+8:b+16] = flags.to_bytes(8, "little")
        def clear_interpreter(d):
            offset = int.from_bytes(d[32:40], "little")
            for i in range(int.from_bytes(d[56:58], "little")):
                b = offset + i * 56
                if int.from_bytes(d[b:b+4], "little") == 3:
                    d[b:b+4] = bytes(4)
        inject("ordinary-dso", elf_edit(clear_pie), "ET_EXEC or PIE")
        inject("missing-interpreter", elf_edit(clear_interpreter), "ET_EXEC or PIE")
        inject("stale-elf", elf_edit(lambda d: d.__setitem__(-1, d[-1] ^ 1)), "artifact changed")
        def symbol_field(field, value):
            def change(d):
                headers, _ = C._static_elf(d)
                for h in headers:
                    if h["type"] != 2:
                        continue
                    strings = headers[h["link"]]
                    for b in range(h["offset"]+24, h["offset"]+h["size"], 24):
                        name = C._string(d, strings["offset"]+int.from_bytes(d[b:b+4], "little"),
                                         strings["offset"]+strings["size"])
                        if name == target:
                            if field == "binding":
                                d[b+4] = value
                            elif field == "visibility":
                                d[b+5] = value
                            else:
                                offset = b + (8 if field == "address" else 16)
                                old = int.from_bytes(d[offset:offset+8], "little")
                                d[offset:offset+8] = (old+value).to_bytes(8, "little")
                            return
                self.fail("native final symbol missing")
            return change
        inject("wrong-address", elf_edit(symbol_field("address", 4)), "^static input/final symbol extent mismatch$")
        inject("wrong-size", elf_edit(symbol_field("size", 4)), "^static native nm disagrees with selected/final symbol extent$")
        inject("global-binding", elf_edit(symbol_field("binding", 0x12)), "weak/default")
        inject("hidden-binding", elf_edit(symbol_field("visibility", 2)), "weak/default")
        def ambiguous_map(text):
            discarded, kept = C._static_map_records(text)
            n, a, size, owner = next(r for r in kept if r[0] == ".text." + target)
            return text + f"\n {n} 0x{a:x} 0x{size:x} {owner}\n"
        inject("ambiguous-kept", map_edit(ambiguous_map), "ambiguous/missing kept owner")
        # Duplicate names are real archive headers, not a caller-provided ordinal.
        saved_archive = (root / "f.a").read_bytes()
        def duplicate_member(directory):
            data = (root / "f.a").read_bytes()
            member = C._static_members(data)[0]
            start, length = member[1], len(member[4])
            (root / "f.a").write_bytes(data + data[start:start+60+length+length%2])
        inject("duplicate-members", duplicate_member, "ambiguous archive member")
        (root / "duplicate-members/injected-archive.a").write_bytes((root / "f.a").read_bytes())
        (root / "f.a").write_bytes(saved_archive)
        def mutate_member(directory):
            data = bytearray(saved_archive)
            member = C._static_members(data)[0]
            headers, names = C._static_elf(member[4])
            code = headers[names.index(".text." + target)]
            data[member[1]+60+code["offset"]] ^= 1
            (root / "f.a").write_bytes(data)
        inject("mutated-member", mutate_member, "kept member/section")
        (root / "mutated-member/injected-archive.a").write_bytes((root / "f.a").read_bytes())
        (root / "f.a").write_bytes(saved_archive)
        exported_args = args.copy()
        exported_args[exported_args.index("-o")+1] = str(root / "exported-app")
        exported_args = [f"-Wl,--dependency-file={root / 'exported.d'}" if a.startswith("-Wl,--dependency-file=")
                         else a for a in exported_args] + ["-Wl,--export-dynamic"]
        subprocess.run(exported_args, check=True)
        with self.assertRaisesRegex(ValueError, "dynsym"):
            C.verify_static_executable_selection(exported_args, root, root / "exported-app", root / "dynsym", unit, obj, obj, target)
        # Identity controls use the actual selected object, not synthetic booleans.
        with self.assertRaisesRegex(ValueError, "audited Unit/B1"):
            C.verify_static_executable_selection(args, root, root / "app", root / "wrong-replay", unit, obj,
                                                (root / "g.o").read_bytes(), target)
        with self.assertRaisesRegex(ValueError, "Unit/target"):
            authority.check(unit._replace(copy="other"), target)
        with mock.patch.dict(os.environ, {"LIBRARY_PATH": "/changed"}):
            with self.assertRaisesRegex(ValueError, "environment"):
                authority.check(unit, target)


if __name__ == "__main__":
    unittest.main()
