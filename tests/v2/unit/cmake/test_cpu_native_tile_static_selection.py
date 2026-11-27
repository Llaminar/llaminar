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

    def test_unit_fingerprint_binds_exact_line_types_and_body_state(self):
        bodies = C.Bodies({"f": ["ret"]})
        unit = C.Unit("x", bodies, "", {"f": "weak"}, frozenset(), frozenset(), {"f": "default"})
        plain = C._static_unit_fingerprint(unit)
        class UnknownInstruction(str):
            pass
        bodies["f"][0] = UnknownInstruction("ret")
        with self.assertRaisesRegex(ValueError, "instruction class"):
            C._static_unit_fingerprint(unit)
        bodies["f"][0] = "ret"
        self.assertEqual(C._static_unit_fingerprint(unit), plain)
        for marker in (C.TLSIEPage, C.TLSIEOffsetLoad):
            bodies["f"][0] = marker("ret")
            typed = C._static_unit_fingerprint(unit)
            self.assertNotEqual(typed, plain)
            bodies["f"][0] = "ret"
            self.assertNotEqual(C._static_unit_fingerprint(unit), typed)
        bodies.extra = True
        with self.assertRaisesRegex(ValueError, "container state"):
            C._static_unit_fingerprint(unit)
        del bodies.extra
        class UnknownBodies(C.Bodies):
            pass
        with self.assertRaisesRegex(ValueError, "container state"):
            C._static_unit_fingerprint(unit._replace(bodies=UnknownBodies(bodies)))

    def test_ninja_record_requires_exact_output_valid_mtime_and_complete_nonempty_count(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root).resolve()
            (root / "input.o").write_bytes(b"object")
            valid = "app: #deps 1, deps mtime 123 (VALID)\n    input.o\n\n"
            self.assertEqual(C._static_ninja_record(valid, "app", 123, root), {root / "input.o"})
            for text in ("", valid.replace("app:", "other:"), valid.replace("VALID", "STALE"),
                         valid.replace("mtime 123", "mtime 124"), valid.replace("#deps 1", "#deps 2"),
                         "app: #deps 0, deps mtime 123 (VALID)\n", valid + valid):
                with self.subTest(record=text), self.assertRaises(ValueError):
                    C._static_ninja_record(text, "app", 123, root)

    def test_rewriting_options_include_forwarded_forms(self):
        for option in ("-flto", "-flto=auto", "--gc-sections", "--icf=all", "-r", "--relocatable", "-shared", "-T", "-Tscript.ld", "--script=script.ld"):
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
        first = unit.bodies[target][0]
        for marker in (C.TLSIEPage, C.TLSIEOffsetLoad):
            unit.bodies[target][0] = marker(str(first))
            with self.assertRaisesRegex(ValueError, "Unit/target changed"):
                authority.check(unit, target)
            self.assertEqual(model.resolve(unit, target)[0], "fail")
            unit.bodies[target][0] = first
            authority.check(unit, target)
        unit.bodies.extra = True
        with self.assertRaisesRegex(ValueError, "container state"):
            authority.check(unit, target)
        del unit.bodies.extra
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


    def test_native_complete_closure_mutation_controls(self):
        compiler = shutil.which(os.environ.get("LLAMINAR_STATIC_TEST_GXX", "g++-14"))
        self.assertIsNotNone(compiler, "native closure controls require GNU14")
        base = os.environ.get("LLAMINAR_STATIC_TEST_ARTIFACTS")
        if base:
            Path(base).mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="static-closure-", dir=base))
        print(f"Retained native closure controls: {root}", flush=True)
        shadow = root / "shadow"
        shadow.mkdir()
        # These are real native tool/input copies, never changes to installed SDK
        # files. -B makes the fixture's production link actually select them.
        for name in ("Scrt1.o", "crti.o", "crtbeginS.o", "crtendS.o", "crtn.o", "liblto_plugin.so", "libm.so"):
            source = subprocess.check_output([compiler, "-print-file-name=" + name], text=True).strip()
            self.assertTrue(Path(source).is_file(), source)
            shutil.copy2(source, shadow / name)
        collect = subprocess.check_output([compiler, "-print-prog-name=collect2"], text=True).strip()
        shutil.copy2(collect, shadow / "collect2")
        (root / "main.cpp").write_text("int main(){return 0;}\n")
        subprocess.run([compiler, "-c", str(root / "main.cpp"), "-o", str(root / "main.o")], check=True)
        script = shadow / "libm.so"
        options = ["-B" + str(shadow) + "/", "-fuse-ld=bfd", str(root / "main.o"),
                   "-Wl,--dependency-file=" + str(root / "production.d"),
                   "-o", str(root / "app")]
        response = root / "link.rsp"
        response.write_text(" ".join(options))
        args = [compiler, "@" + str(response)]
        subprocess.run(args, check=True)
        baseline = C.capture_static_link(args, root, root / "app", root / "valid")
        for name in ("Scrt1.o", "liblto_plugin.so", "collect2"):
            self.assertIn(str(shadow / name), baseline["files"], f"native production did not select {name}")
        self.assertIn(str(script), baseline["files"])
        originals = {p: p.read_bytes() for p in (shadow / "Scrt1.o", shadow / "liblto_plugin.so",
                                                shadow / "collect2", response, script, root / "production.d")}
        native_run = subprocess.run

        def fault(label, when, mutate, expected):
            directory = root / label
            fired = False
            def invoke(command, **kw):
                nonlocal fired
                result = native_run(command, **kw)
                if not fired and when(command):
                    fired = True
                    mutate(directory, command, result)
                return result
            with mock.patch.dict(os.environ, dict(os.environ), clear=True):
                with mock.patch.object(C.subprocess, "run", invoke):
                    with self.subTest(boundary=label), self.assertRaisesRegex(ValueError, expected):
                        C.capture_static_link(args, root, root / "app", directory)
            self.assertTrue(fired, label)
            # Preserve the exact injected bytes; fixture reset is outside capture.
            for p, data in originals.items():
                if p.read_bytes() != data:
                    (directory / ("injected-" + p.name)).write_bytes(p.read_bytes())
                    p.write_bytes(data)
            args[:] = [compiler, "@" + str(response)]

        def transaction(label):
            return lambda command: "-o" in command and Path(command[command.index("-o")+1]).name == label + ".elf"
        def append(path):
            return lambda directory, command, result: path.write_bytes(path.read_bytes() + b"\nchanged\n")
        fault("mutated-crt", transaction("map"), append(shadow / "Scrt1.o"), "input changed between transactions")
        fault("mutated-plugin", transaction("map"), append(shadow / "liblto_plugin.so"), "static input changed:")
        fault("mutated-tool", transaction("map"), append(shadow / "collect2"), "static input changed:")
        fault("mutated-response", transaction("map"), append(response), "static input changed:")
        fault("mutated-script", transaction("map"), append(script), "input changed between transactions")
        fault("mutated-production-deps", transaction("map"), append(root / "production.d"), "static input changed:")
        fault("mutated-default-script", lambda c: "--verbose" in c and (root / "mutated-default-script/default-script.txt").exists(),
              lambda directory, command, result: append(directory / "default-script.txt")(directory, command, result),
              "static input changed:")
        fault("environment-during-capture", transaction("control"),
              lambda directory, command, result: os.environ.__setitem__("LIBRARY_PATH", "/injected"),
              "environment changed during capture")
        fault("argv-during-capture", transaction("control"),
              lambda directory, command, result: args.append("-O3"), "original argv changed during capture")
        for label in ("control", "map"):
            def change_elf(directory, command, result):
                p = Path(command[command.index("-o")+1])
                data = bytearray(p.read_bytes())
                data[-1] ^= 1
                p.write_bytes(data)
            fault("different-" + label, transaction(label), change_elf, f"static {label} differs from production")
        def drop_trace(directory, command, result):
            (directory / "unmodified-native-trace.stdout").write_bytes(result.stdout)
            lines = result.stdout.splitlines(keepends=True)
            self.assertGreater(len(lines), 2)
            result.stdout = b"".join(lines[1:])
        fault("trace-dependency-mismatch", transaction("instrumented"), drop_trace,
              "trace/dependency input set mismatch")
        # A genuinely incomplete production dependency inventory is rejected before
        # any claimed closure, even though the original executable is unchanged.
        dep = root / "production.d"
        text = dep.read_text()
        implicit = next(p for p in baseline["files"] if Path(p).name == "Scrt1.o")
        dep.write_text(text.replace(implicit, ""))
        with self.assertRaisesRegex(ValueError, "debug dependency set differs from production"):
            C.capture_static_link(args, root, root / "app", root / "dropped-implicit-crt")
        (root / "dropped-implicit-crt/injected-production.d").write_bytes(dep.read_bytes())
        dep.write_bytes(originals[dep])
        extra = root / "extra.ld"
        extra.write_text("SECTIONS { .proof_note : { BYTE(0x73) } } INSERT AFTER .comment;\n")
        explicit_args = [compiler, *options, "-Wl,-T," + str(extra)]
        explicit_args[explicit_args.index("-o")+1] = str(root / "explicit-script-app")
        explicit_args = ["-Wl,--dependency-file=" + str(root / "explicit-script.d")
                         if a.startswith("-Wl,--dependency-file=") else a for a in explicit_args]
        native_run(explicit_args, check=True)
        with self.assertRaisesRegex(ValueError, "unsupported static link option: -T"):
            C.capture_static_link(explicit_args, root, root / "explicit-script-app", root / "unsupported-explicit-script")


    def test_native_ninja_consumed_dependency_record_controls(self):
        compiler = shutil.which(os.environ.get("LLAMINAR_STATIC_TEST_GXX", "g++-14"))
        ninja = shutil.which("ninja")
        self.assertIsNotNone(compiler, "native Ninja fixture requires GNU14")
        self.assertIsNotNone(ninja, "native Ninja fixture requires actual Ninja")
        base = os.environ.get("LLAMINAR_STATIC_TEST_ARTIFACTS")
        if base:
            Path(base).mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="static-ninja-", dir=base))
        print(f"Retained native Ninja controls: {root}", flush=True)
        (root / "main.cpp").write_text("int main(){return 0;}\n")
        subprocess.run([compiler, "-c", str(root / "main.cpp"), "-o", str(root / "main.o")], check=True)
        args = [compiler, "main.o", "-Wl,--dependency-file=link.d", "-o", "app"]
        (root / "build.ninja").write_text("rule link\n  command = " + " ".join(args) +
                "\n  depfile = link.d\n  deps = gcc\nbuild app: link main.o\n")
        subprocess.run([ninja, "app"], cwd=root, check=True)
        self.assertFalse((root / "link.d").exists(), "Ninja deps=gcc must actually consume the side output")
        with self.assertRaisesRegex(ValueError, "requires explicit native Ninja"):
            C.capture_static_link(args, root, root / "app", root / "no-explicit-ninja")
        C.capture_static_link(args, root, root / "app", root / "valid", ninja=Path(ninja))
        output = root / "app"
        mtime = output.stat().st_mtime_ns
        os.utime(output, ns=(mtime+1, mtime+1))
        with self.assertRaisesRegex(ValueError, "stale or wrong-output"):
            C.capture_static_link(args, root, output, root / "stale", ninja=Path(ninja))
        os.utime(output, ns=(mtime, mtime))
        shutil.copy2(output, root / "wrong-app")
        wrong = args.copy()
        wrong[-1] = "wrong-app"
        with self.assertRaisesRegex(ValueError, "native Ninja deps query failed"):
            C.capture_static_link(wrong, root, root / "wrong-app", root / "wrong-target", ninja=Path(ninja))
        native_run = subprocess.run
        database = root / ".ninja_deps"
        original_database = database.read_bytes()
        copied_ninja = root / "ninja"
        shutil.copy2(ninja, copied_ninja)
        for label, path in (("mutated-database", database), ("mutated-ninja-tool", copied_ninja)):
            def invoke(command, **kw):
                result = native_run(command, **kw)
                if "-o" in command and Path(command[command.index("-o")+1]).name == "map.elf":
                    path.write_bytes(path.read_bytes() + b"changed")
                return result
            with mock.patch.object(C.subprocess, "run", invoke):
                with self.assertRaisesRegex(ValueError, "static input changed:"):
                    C.capture_static_link(args, root, output, root / label, ninja=copied_ninja)
            (root / label / "injected-native-artifact").write_bytes(path.read_bytes())
            database.write_bytes(original_database)
            shutil.copy2(ninja, copied_ninja)
        # Actual Ninja log with an incomplete producer side output: valid record
        # status alone must not be mistaken for a complete native linker closure.
        rule = (root / "build.ninja").read_text()
        rule = rule.replace("\n  depfile", " && sed -i '/Scrt1.o/d' link.d\n  depfile")
        (root / "build.ninja").write_text(rule)
        native_run([ninja, "-t", "clean"], cwd=root, check=True)
        native_run([ninja, "app"], cwd=root, check=True)
        with self.assertRaisesRegex(ValueError, "debug dependency set differs from production"):
            C.capture_static_link(args, root, output, root / "different-native-record-set", ninja=Path(ninja))


if __name__ == "__main__":
    unittest.main()
