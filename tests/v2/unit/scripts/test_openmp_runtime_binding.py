#!/usr/bin/env python3
"""Prove the real executable loads its compiler-selected CPU OpenMP runtime.

ELF dependency tracing asks the executable's native loader to resolve its whole
closure without entering main, initializing an accelerator, or loading a model.
The expected SONAME comes from FindOpenMP's selected library at configuration;
absolute compiler paths and untrusted same-filename aliases are not authority.
The shadow control uses a harmless DSO and a temporary alias to reproduce the
SDK's libgomp-to-libomp mistake without modifying any installed library.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


class RuntimeBindingError(RuntimeError):
    """A selected OpenMP ABI is missing, duplicated, or replaced by an alias."""


def parse_loader_runtime(trace: str, expected_soname: str) -> Path:
    """Resolve the required ABI from loader output, rejecting missing bindings."""
    pattern = re.compile(r"^\s*(\S+)\s+=>\s+(.+?)\s+\(0x[0-9a-fA-F]+\)\s*$")
    matches = []
    for line in trace.splitlines():
        match = pattern.fullmatch(line)
        if match and match[1] == expected_soname:
            matches.append(Path(match[2]))
    if len(matches) != 1 or not matches[0].is_absolute():
        raise RuntimeBindingError(
            f"OpenMP runtime binding requires one resolved {expected_soname}; "
            f"observed {matches}")
    return matches[0]


def parse_soname(dynamic_section: str) -> str:
    """Require one real ELF identity instead of trusting the alias's filename."""
    matches = re.findall(r"\(SONAME\)[^\n]*\[([^\]]+)\]", dynamic_section)
    if len(matches) != 1:
        raise RuntimeBindingError("OpenMP runtime DSO has no unique ELF SONAME")
    return matches[0]


def require_identity(expected_soname: str, actual_soname: str, runtime: Path) -> None:
    """Reject an SDK compatibility runtime posing as the selected native ABI."""
    if actual_soname != expected_soname:
        raise RuntimeBindingError(
            f"OpenMP runtime shadowed: expected {expected_soname}, "
            f"loaded {actual_soname} from {runtime}")


def verify_binding(binary: Path, expected_soname: str, readelf: str,
                   environment: dict[str, str] | None = None) -> Path:
    """Authenticate the loader's actual selection without executing inference."""
    env = dict(os.environ if environment is None else environment)
    env["LD_TRACE_LOADED_OBJECTS"] = "1"
    trace = subprocess.run([str(binary)], env=env, text=True, capture_output=True,
                           check=True, timeout=30)
    runtime = parse_loader_runtime(trace.stdout, expected_soname)
    # Dereferencing the selected file is essential: an alias called libgomp
    # can contain LLVM libomp even though the dependency trace names libgomp.
    dynamic = subprocess.run([readelf, "-d", str(runtime)], text=True,
                             capture_output=True, check=True, timeout=30)
    require_identity(expected_soname, parse_soname(dynamic.stdout), runtime)
    return runtime


def verify_shadow_rejection(binary: Path, expected_soname: str, readelf: str,
                            fixture: Path) -> None:
    """Prove same-filename substitution is rejected through the native loader."""
    with tempfile.TemporaryDirectory(prefix="llaminar-openmp-shadow-") as directory:
        alias = Path(directory) / expected_soname
        alias.symlink_to(fixture.resolve(strict=True))
        env = dict(os.environ)
        # The isolated child may shadow a dependency; no production environment
        # or installed SDK is changed. Preloads would defeat this negative test.
        env.pop("LD_PRELOAD", None)
        env["LD_LIBRARY_PATH"] = directory + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        try:
            verify_binding(binary, expected_soname, readelf, env)
        except RuntimeBindingError as error:
            if "OpenMP runtime shadowed:" in str(error):
                return
            raise
        raise RuntimeBindingError("Same-filename OpenMP shadow was not rejected")


class OpenMPRuntimeBindingTests(unittest.TestCase):
    """Fast device-free negative controls for loader and ELF identity parsing."""

    def test_selected_runtime_resolves_once(self):
        """Unrelated dependencies must not replace the configured OpenMP ABI."""
        trace = "libc.so.6 => /lib/libc.so.6 (0x1)\nlibgomp.so.1 => /lib/libgomp.so.1 (0x2)"
        self.assertEqual(parse_loader_runtime(trace, "libgomp.so.1"),
                         Path("/lib/libgomp.so.1"))

    def test_missing_relative_and_duplicate_bindings_fail(self):
        """Incomplete closure and ambiguous identities fail instead of guessing."""
        for trace in ("libgomp.so.1 => not found",
                      "libgomp.so.1 => relative/libgomp.so.1 (0x1)",
                      "libgomp.so.1 => /a/libgomp.so.1 (0x1)\n"
                      "libgomp.so.1 => /b/libgomp.so.1 (0x2)"):
            with self.subTest(trace=trace), self.assertRaises(RuntimeBindingError):
                parse_loader_runtime(trace, "libgomp.so.1")

    def test_native_soname_is_required(self):
        """A symlink's name is not evidence for the file's real ELF identity."""
        self.assertEqual(parse_soname("0xe (SONAME) Library soname: [libomp.so]"),
                         "libomp.so")
        for text in ("", "(SONAME) [libgomp.so.1]\n(SONAME) [libomp.so]"):
            with self.subTest(text=text), self.assertRaises(RuntimeBindingError):
                parse_soname(text)

    def test_compatibility_alias_is_rejected_symmetrically(self):
        """GNU and LLVM bindings both reject substitution by the other runtime."""
        for expected, actual in (("libgomp.so.1", "libomp.so"),
                                 ("libomp.so", "libgomp.so.1")):
            with self.subTest(expected=expected), self.assertRaisesRegex(
                    RuntimeBindingError, "OpenMP runtime shadowed"):
                require_identity(expected, actual, Path("/sdk/libgomp.so.1"))
        require_identity("libgomp.so.1", "libgomp.so.1", Path("/lib/libgomp.so.1"))


def main() -> int:
    """Run parser controls or a source-configured real-executable preflight."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--expected-soname")
    parser.add_argument("--readelf", default="readelf")
    parser.add_argument("--wrong-runtime-fixture", type=Path)
    args = parser.parse_args()
    if args.binary is None:
        if args.expected_soname or args.wrong_runtime_fixture:
            parser.error("runtime binding options require --binary")
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(OpenMPRuntimeBindingTests))
        return 0 if result.wasSuccessful() else 1
    if not args.expected_soname:
        parser.error("--binary requires the configure-owned --expected-soname")
    runtime = verify_binding(args.binary.resolve(strict=True), args.expected_soname,
                             args.readelf)
    if args.wrong_runtime_fixture:
        verify_shadow_rejection(args.binary.resolve(strict=True), args.expected_soname,
                                args.readelf, args.wrong_runtime_fixture)
    print(f"PASS OpenMP runtime binding: {args.expected_soname} -> {runtime}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
