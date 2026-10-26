#!/usr/bin/env python3
"""Authenticate HIP runtime selection without entering either executable's main.

The installer-owned source/version/ordered-patch receipt and actual resolved DSO
are one artifact contract. Trace both the inherited environment and the clean
ELF default, because a qualified override cannot repair a stale default (and a
correct default cannot excuse a stale override). The canonical SDK installation and exported runtime component use the same
lib/../share layout. Resolve that exact
physical owner, never search alternate receipts or infer a repair from SONAME.

Unit controls use tiny synthetic files and mocked loader traces. The integration
entrypoint reuses OpenMPRuntimeBinding's LD_TRACE_LOADED_OBJECTS helper; it does
not initialize HIP, MPI, an accelerator, or a model. Only the selected HIP DSO
and small source/receipt files are hashed, never application/core/model payloads.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from test_openmp_runtime_binding import RuntimeBindingError as ELFBindingError, parse_soname, trace_loader

ROOT = Path(__file__).resolve().parents[4]


class HIPRuntimeBindingError(RuntimeError):
    """A real loader choice lacks its current canonical HIP installation proof."""


def file_digest(path: Path) -> str:
    """Hash one bounded dependency artifact while rejecting concurrent replacement."""
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    fields = lambda value: (value.st_dev, value.st_ino, value.st_size,
                            value.st_mtime_ns, value.st_ctime_ns)
    if fields(before) != fields(after):
        raise HIPRuntimeBindingError("HIP artifact changed while authenticating: " + str(path))
    return digest.hexdigest()


@dataclass(frozen=True)
class HIPRuntimeAuthority:
    """Current installer-owned version and ordered patch identity, not a runtime path list."""
    soname: str
    identity: str
    patches: tuple[tuple[str, str], ...]

    @classmethod
    def from_source(cls, source_root: Path) -> "HIPRuntimeAuthority":
        """Read the canonical installer array/pins directly; reject unknown array syntax."""
        docker = source_root / "scripts/docker"
        pins = {}
        for line in (docker / "rocm-release.env").read_text().splitlines():
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                if key in pins:
                    raise HIPRuntimeBindingError("Duplicate HIP release pin: " + key)
                pins[key] = value
        keys = ("LLAMINAR_ROCM_VERSION", "LLAMINAR_ROCM_DEB_VERSION",
                "LLAMINAR_ROCM_SYSTEMS_REVISION", "LLAMINAR_HIP_LIBRARY_VERSION")
        if any(not pins.get(key) or ":" in pins[key] for key in keys):
            raise HIPRuntimeBindingError("Incomplete HIP source/version identity")
        version = pins[keys[-1]]
        if not re.fullmatch(r"\d+(?:\.\d+)+", version):
            raise HIPRuntimeBindingError("Invalid HIP library version")
        declaration = re.search(r"^hip_patches=\(\n(.*?)^\)",
                                (docker / "install-hip-graph-runtime.sh").read_text(), re.M | re.S)
        if not declaration:
            raise HIPRuntimeBindingError("Canonical installer omitted ordered HIP repairs")
        patches = []
        for line in declaration[1].splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.fullmatch(r'"\$\{hip_script_dir\}/(patches/[^"\n]+)"', line)
            if not match:
                raise HIPRuntimeBindingError("Unknown canonical HIP patch declaration")
            relative = PurePosixPath(match[1])
            if ".." in relative.parts or relative.is_absolute() or relative.suffix != ".patch":
                raise HIPRuntimeBindingError("Invalid canonical HIP patch path")
            patches.append((relative.name, file_digest(docker / relative)))
        if not patches or len({name for name, _ in patches}) != len(patches):
            raise HIPRuntimeBindingError("Missing or duplicated canonical HIP repairs")
        identity = ":".join([*(pins[key] for key in keys), *(digest for _, digest in patches)])
        return cls("libamdhip64.so." + version.split(".")[0], identity, tuple(patches))


def parse_hip_runtime(trace: str, expected_soname: str) -> Path:
    """Require exactly one resolved HIP ABI and no unresolved dependency in the closure."""
    if re.search(r"=>\s+not found(?:\s|$)", trace):
        raise HIPRuntimeBindingError("Native loader closure has an unresolved dependency")
    names = re.compile(r"libamdhip64\.so(?:\.\d+)*")
    resolved = []
    for line in trace.splitlines():
        fields = line.strip().split()
        if not fields or not names.fullmatch(Path(fields[0]).name):
            continue
        if Path(fields[0]).name != expected_soname:
            raise HIPRuntimeBindingError("Native loader requested a foreign HIP ABI")
        if len(fields) >= 3 and fields[1] == "=>":
            candidate = Path(fields[2])
        else:
            candidate = Path(fields[0])
        if not candidate.is_absolute():
            raise HIPRuntimeBindingError("Native loader did not resolve an absolute HIP DSO")
        resolved.append(candidate)
    if len(resolved) != 1:
        raise HIPRuntimeBindingError("Native loader requires one unique HIP DSO")
    return resolved[0]


def require_installed_receipt(runtime: Path, authority: HIPRuntimeAuthority) -> None:
    """Authenticate the actual physical DSO and its paired local/shipping receipt and patches."""
    runtime = runtime.resolve(strict=True)
    if runtime.parent.name != "lib":
        raise HIPRuntimeBindingError("HIP DSO is outside its canonical installed lib owner")
    prefix = runtime.parent.parent
    receipt = prefix / "share/llaminar/hip-graph-runtime.txt"
    try:
        marker = receipt.read_text().strip()
    except OSError as error:
        raise HIPRuntimeBindingError("Loaded HIP DSO has no paired installation receipt") from error
    if not marker.startswith(authority.identity + ":"):
        raise HIPRuntimeBindingError("Loaded HIP receipt has stale source/version/ordered-patch identity")
    declared = marker[len(authority.identity) + 1:]
    if not re.fullmatch(r"[0-9a-f]{64}", declared) or file_digest(runtime) != declared:
        raise HIPRuntimeBindingError("Loaded HIP DSO differs from its installation receipt")
    for name, digest in authority.patches:
        installed = prefix / "share/licenses/llaminar-hip" / name
        if not installed.is_file() or file_digest(installed) != digest:
            raise HIPRuntimeBindingError("Loaded HIP installed repair differs from current source: " + name)
    if receipt.read_text().strip() != marker:
        raise HIPRuntimeBindingError("HIP installation receipt changed during authentication")


def verify_binding(binary: Path, authority: HIPRuntimeAuthority, readelf: str,
                   environment: dict[str, str]) -> Path:
    """Trace the real ELF closure before main and validate its actual HIP artifact identity."""
    trace = trace_loader(binary, environment)
    runtime = parse_hip_runtime(trace, authority.soname)
    dynamic = subprocess.run([readelf, "-d", str(runtime)], text=True,
                             capture_output=True, check=True, timeout=30)
    try:
        soname = parse_soname(dynamic.stdout)
    except ELFBindingError as error:
        raise HIPRuntimeBindingError("Loaded HIP runtime has no unique ELF SONAME") from error
    if soname != authority.soname:
        raise HIPRuntimeBindingError("Loaded HIP runtime has a foreign ELF SONAME")
    require_installed_receipt(runtime, authority)
    return runtime.resolve(strict=True)


def verify_environments(binary: Path, authority: HIPRuntimeAuthority, readelf: str,
                        environment: dict[str, str]) -> dict[str, Path]:
    """Check inherited and clean/default resolution independently, allowing authenticated copies."""
    clean = {key: value for key, value in environment.items()
             if key not in ("LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT")}
    return {"inherited": verify_binding(binary, authority, readelf, dict(environment)),
            "clean_default": verify_binding(binary, authority, readelf, clean)}


def verify_shadow_rejection(binary: Path, authority: HIPRuntimeAuthority, readelf: str,
                            fixture: Path) -> None:
    """Use the existing harmless same-filename DSO fixture; never execute native inference."""
    with tempfile.TemporaryDirectory(prefix="llaminar-hip-shadow-") as directory:
        alias = Path(directory) / authority.soname
        alias.symlink_to(fixture.resolve(strict=True))
        env = dict(os.environ)
        env.pop("LD_PRELOAD", None)
        env.pop("LD_AUDIT", None)
        env["LD_LIBRARY_PATH"] = directory + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        try:
            verify_binding(binary, authority, readelf, env)
        except HIPRuntimeBindingError as error:
            if "foreign ELF SONAME" in str(error):
                return
            raise
        raise HIPRuntimeBindingError("Same-filename foreign HIP runtime was not rejected")


class HIPRuntimeBindingTests(unittest.TestCase):
    """Device-free metadata controls; tiny fixture bytes are not native HIP libraries."""

    def fixture(self, directory: Path) -> tuple[HIPRuntimeAuthority, Path]:
        """Author a source-owned installer and installed artifact without compiling/loading it."""
        source = directory / "source/scripts/docker"
        (source / "patches").mkdir(parents=True)
        (source / "rocm-release.env").write_text("LLAMINAR_ROCM_VERSION=10.0.0\n"
            "LLAMINAR_ROCM_DEB_VERSION=10.0.0-4\nLLAMINAR_ROCM_SYSTEMS_REVISION=revision\n"
            "LLAMINAR_HIP_LIBRARY_VERSION=7.15.26333\n")
        (source / "install-hip-graph-runtime.sh").write_text('hip_patches=(\n'
            '    "${hip_script_dir}/patches/first.patch"\n'
            '    "${hip_script_dir}/patches/latest.patch"\n)\n')
        for name in ("first.patch", "latest.patch"):
            (source / "patches" / name).write_text("owned repair " + name)
        authority = HIPRuntimeAuthority.from_source(directory / "source")
        prefix = directory / "installed"
        (prefix / "lib").mkdir(parents=True)
        (prefix / "share/llaminar").mkdir(parents=True)
        (prefix / "share/licenses/llaminar-hip").mkdir(parents=True)
        library = prefix / "lib/libamdhip64.so.7.15.26333"
        library.write_bytes(b"tiny authenticated artifact fixture")
        for name, _ in authority.patches:
            shutil.copyfile(source / "patches" / name, prefix / "share/licenses/llaminar-hip" / name)
        (prefix / "share/llaminar/hip-graph-runtime.txt").write_text(authority.identity + ":" + file_digest(library) + "\n")
        return authority, library

    def test_source_receipt_accepts_local_symlink_and_sdk_component_copy(self):
        """Different deployment inodes with the same source-certified bytes remain valid."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authority, library = self.fixture(root)
            alias = library.parent / authority.soname
            alias.symlink_to(library.name)
            require_installed_receipt(alias, authority)
            shipped = root / "opt/rocm-sdk"
            shutil.copytree(library.parent.parent, shipped)
            require_installed_receipt(shipped / "lib" / authority.soname, authority)
            self.assertNotEqual(library.stat().st_ino, (shipped / "lib" / library.name).stat().st_ino)

    def test_stale_repair_missing_receipt_and_changed_artifact_reject(self):
        """A six-of-seven-style receipt, missing marker or replaced DSO cannot authorize a loader."""
        for mutation in ("missing", "stale_patch", "artifact", "installed_patch"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                authority, library = self.fixture(Path(directory))
                marker = library.parent.parent / "share/llaminar/hip-graph-runtime.txt"
                if mutation == "missing":
                    marker.unlink()
                elif mutation == "stale_patch":
                    marker.write_text(":".join([*authority.identity.split(":")[:-1], file_digest(library)]))
                elif mutation == "artifact":
                    library.write_bytes(b"replacement has the same public filename")
                else:
                    (library.parent.parent / "share/licenses/llaminar-hip/latest.patch").write_text("foreign")
                with self.assertRaises(HIPRuntimeBindingError):
                    require_installed_receipt(library, authority)

    def test_source_array_and_version_are_closed(self):
        """Unsupported shell syntax, duplicates and changed source repairs fail before any loader."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authority, library = self.fixture(root)
            source = root / "source/scripts/docker"
            (source / "patches/latest.patch").write_text("new source repair")
            changed = HIPRuntimeAuthority.from_source(root / "source")
            with self.assertRaisesRegex(HIPRuntimeBindingError, "stale source"):
                require_installed_receipt(library, changed)
            (source / "patches/latest.patch").write_text("owned repair latest.patch")
            pin_file = source / "rocm-release.env"
            original_pins = pin_file.read_text()
            for old, new in (("10.0.0\n", "10.1.0\n"), ("10.0.0-4", "10.0.0-5"),
                             ("=revision", "=foreign"), ("7.15.26333", "7.15.99999")):
                pin_file.write_text(original_pins.replace(old, new))
                with self.subTest(pin=old), self.assertRaisesRegex(HIPRuntimeBindingError, "stale source"):
                    require_installed_receipt(library, HIPRuntimeAuthority.from_source(root / "source"))
            pin_file.write_text(original_pins)
            for line in ('    "${other}/patches/first.patch"',
                         '    "${hip_script_dir}/patches/../first.patch"',
                         '    "${hip_script_dir}/patches/first.patch"\n    "${hip_script_dir}/patches/first.patch"'):
                (source / "install-hip-graph-runtime.sh").write_text("hip_patches=(\n" + line + "\n)\n")
                with self.subTest(line=line), self.assertRaises(HIPRuntimeBindingError):
                    HIPRuntimeAuthority.from_source(root / "source")

    def test_loader_requires_one_current_resolved_abi(self):
        """Missing, mixed, duplicate, unresolved and relative loader records are all fatal."""
        soname = "libamdhip64.so.7"
        good = soname + " => /sdk/lib/" + soname + " (0x1)"
        self.assertEqual(parse_hip_runtime(good, soname), Path("/sdk/lib") / soname)
        self.assertEqual(parse_hip_runtime("/sdk/lib/" + soname + " (0x1)", soname), Path("/sdk/lib") / soname)
        for trace in ("", soname + " => not found", soname + " => relative.so (0x1)",
                      good + "\n" + good, good + "\nlibamdhip64.so.6 => /old/lib.so (0x2)",
                      good + "\nlibforeign.so => not found"):
            with self.subTest(trace=trace), self.assertRaises(HIPRuntimeBindingError):
                parse_hip_runtime(trace, soname)

    def test_inherited_and_default_are_separate_authentication_boundaries(self):
        """A valid override cannot hide a broken default, nor the reverse."""
        authority = HIPRuntimeAuthority("libamdhip64.so.7", "identity", ())
        env = {"LD_LIBRARY_PATH": "/override", "LD_PRELOAD": "/preload", "LD_AUDIT": "/audit", "OTHER": "kept"}
        with patch(__name__ + ".verify_binding", side_effect=[Path("/first"), Path("/second")]) as verify:
            result = verify_environments(Path("/app"), authority, "readelf", env)
        self.assertEqual(result, {"inherited": Path("/first"), "clean_default": Path("/second")})
        self.assertEqual(verify.call_args_list[0].args[3], env)
        self.assertEqual(verify.call_args_list[1].args[3], {"OTHER": "kept"})
        for failures in ((HIPRuntimeBindingError("override"),), (Path("/valid"), HIPRuntimeBindingError("default"))):
            with self.subTest(failures=failures), patch(__name__ + ".verify_binding", side_effect=failures), self.assertRaises(HIPRuntimeBindingError):
                verify_environments(Path("/app"), authority, "readelf", env)

    def test_foreign_soname_and_same_filename_shadow_reject(self):
        """ELF identity is checked before accepting a similarly named path or receipt."""
        authority = HIPRuntimeAuthority("libamdhip64.so.7", "identity", ())
        completed = subprocess.CompletedProcess([], 0, "(SONAME) [libforeign.so]", "")
        with patch(__name__ + ".trace_loader", return_value="libamdhip64.so.7 => /sdk/lib/libamdhip64.so.7 (0x1)"), \
             patch(__name__ + ".subprocess.run", return_value=completed), \
             self.assertRaisesRegex(HIPRuntimeBindingError, "foreign ELF SONAME"):
            verify_binding(Path("/app"), authority, "readelf", {})
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "harmless.so"
            fixture.write_bytes(b"never loaded in a Unit test")
            with patch(__name__ + ".verify_binding", side_effect=HIPRuntimeBindingError("foreign ELF SONAME")):
                verify_shadow_rejection(Path("/app"), authority, "readelf", fixture)
            with patch(__name__ + ".verify_binding", return_value=fixture), self.assertRaisesRegex(HIPRuntimeBindingError, "not rejected"):
                verify_shadow_rejection(Path("/app"), authority, "readelf", fixture)


    def staged_component(self, root: Path) -> tuple[HIPRuntimeAuthority, Path]:
        """Exercise the real installer's export boundary with tiny authenticated file fixtures."""
        authority = HIPRuntimeAuthority.from_source(ROOT)
        version = authority.identity.split(":")[3]
        prefix = root / "builder-sdk"
        (prefix / ".info").mkdir(parents=True)
        (prefix / ".info/version").write_text(authority.identity.split(":")[0] + "\n")
        (prefix / "lib").mkdir()
        (prefix / "share/llaminar").mkdir(parents=True)
        licenses = prefix / "share/licenses/llaminar-hip"
        licenses.mkdir(parents=True)
        library = prefix / "lib" / ("libamdhip64.so." + version)
        library.write_bytes(b"synthetic authenticated HIP component, not a native DSO")
        for alias, target in ((authority.soname, library.name),
                              (library.name + "-0000000", library.name),
                              ("libamdhip64.so", authority.soname)):
            (prefix / "lib" / alias).symlink_to(target)
        (licenses / "LICENSE.md").write_text("synthetic component license fixture")
        for name, _ in authority.patches:
            shutil.copyfile(ROOT / "scripts/docker/patches" / name, licenses / name)
        (prefix / "share/llaminar/hip-graph-runtime.txt").write_text(
            authority.identity + ":" + file_digest(library) + "\n")
        component = root / "runtime-hip"
        env = dict(os.environ, ROCM_PATH=str(prefix), HIP_RUNTIME_INSTALL_PREFIX=str(prefix),
                   HIP_RUNTIME_STAGE_PREFIX=str(component))
        result = subprocess.run(["bash", str(ROOT / "scripts/docker/install-hip-graph-runtime.sh")],
                                env=env, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("component staged", result.stdout)
        return authority, component

    def test_sdk_component_replaces_stock_aliases_and_authenticates_both_loader_modes(self):
        """One staged SDK owner must satisfy inherited and clean loaders after stock install."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authority, component = self.staged_component(root)
            sdk = root / "runtime/opt/rocm-sdk"
            (sdk / "lib").mkdir(parents=True)
            version = authority.identity.split(":")[3]
            package = "libamdhip64.so." + version + "-0000000"
            (sdk / "lib" / package).write_bytes(b"stock package file must be replaced")
            (sdk / "lib" / authority.soname).symlink_to(package)
            (sdk / "lib/libamdhip64.so").symlink_to(authority.soname)
            subprocess.run(["cp", "-a", str(component) + "/.", str(sdk)], check=True, timeout=30)
            expected = sdk / "lib" / ("libamdhip64.so." + version)
            for name in (expected.name, package, authority.soname, "libamdhip64.so"):
                self.assertEqual((sdk / "lib" / name).resolve(strict=True), expected)
                require_installed_receipt(sdk / "lib" / name, authority)
            trace = authority.soname + " => " + str(sdk / "lib" / authority.soname) + " (0x1)"
            with patch(__name__ + ".trace_loader", return_value=trace), \
                    patch(__name__ + ".subprocess.run", return_value=subprocess.CompletedProcess(
                        [], 0, "(SONAME) [" + authority.soname + "]", "")):
                result = verify_environments(Path("/never-enter-main"), authority, "readelf",
                    {"LD_LIBRARY_PATH": "/usr/local/lib:" + str(sdk / "lib")})
            self.assertEqual(result, {"inherited": expected, "clean_default": expected})

    def test_old_two_prefix_shipping_layout_rejects_clean_stock_sdk(self):
        """The original /usr/local repair cannot excuse the SDK's stock clean-loader choice."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authority, component = self.staged_component(root)
            local = root / "runtime/usr/local"
            shutil.copytree(component, local, symlinks=True)
            sdk = root / "runtime/opt/rocm-sdk"
            (sdk / "lib").mkdir(parents=True)
            stock = sdk / "lib" / authority.soname
            stock.write_bytes(b"unrepaired SDK HIP with matching public SONAME")
            local_alias = local / "lib" / authority.soname
            require_installed_receipt(local_alias, authority)
            def loader(_binary, env):
                selected = local_alias if env.get("LD_LIBRARY_PATH") else stock
                return authority.soname + " => " + str(selected) + " (0x1)"
            with patch(__name__ + ".trace_loader", side_effect=loader), \
                    patch(__name__ + ".subprocess.run", return_value=subprocess.CompletedProcess(
                        [], 0, "(SONAME) [" + authority.soname + "]", "")):
                with self.assertRaisesRegex(HIPRuntimeBindingError, "no paired installation receipt"):
                    verify_environments(Path("/never-enter-main"), authority, "readelf",
                        {"LD_LIBRARY_PATH": str(local / "lib")})

    def test_sdk_component_staging_rejects_stale_source_without_building(self):
        """Export cannot fall through to a compiler/download when its existing receipt is stale."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authority, _ = self.staged_component(root)
            prefix = root / "builder-sdk"
            (prefix / "share/llaminar/hip-graph-runtime.txt").write_text("stale identity")
            target = root / "must-not-exist"
            result = subprocess.run(["bash", str(ROOT / "scripts/docker/install-hip-graph-runtime.sh")],
                env=dict(os.environ, ROCM_PATH=str(prefix), HIP_RUNTIME_INSTALL_PREFIX=str(prefix),
                         HIP_RUNTIME_STAGE_PREFIX=str(target)), text=True, capture_output=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("staging requires an authenticated installed runtime", result.stderr)
            self.assertFalse(target.exists())

    @staticmethod
    def tree_identity(prefix: Path) -> dict:
        """Observe synthetic tree bytes/links/mtime without following foreign destination links."""
        result = {}
        if not prefix.exists() and not prefix.is_symlink():
            return result
        for path in [prefix, *sorted(prefix.rglob("*"))] if prefix.is_dir() and not prefix.is_symlink() else [prefix]:
            relative = str(path.relative_to(prefix))
            if path.is_symlink():
                result[relative] = ("link", os.readlink(path), path.lstat().st_mtime_ns)
            elif path.is_file():
                result[relative] = ("file", path.read_bytes(), path.stat().st_mtime_ns)
            else:
                result[relative] = ("directory",)
        return result

    def test_sdk_component_accepts_existing_empty_docker_staging_root(self):
        """Docker precreates only the component root; absent lib/share are owned by export."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runtime-hip").mkdir()
            authority, component = self.staged_component(root)
            require_installed_receipt(component / "lib" / authority.soname, authority)

    def test_sdk_component_rejects_resolved_source_and_occupied_destinations_without_writes(self):
        """Source aliases, existing components and dangling links cannot authorize any copy."""
        cases = ("source", "source_dot", "source_link", "source_child", "lib_dir", "share_dir",
                 "lib_file", "share_file", "lib_link", "share_link", "lib_dangling", "share_dangling",
                 "root_file", "root_link", "root_dangling")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.staged_component(root)
                source = root / "builder-sdk"
                target = root / "destination"
                external = root / "external"
                external.mkdir()
                (external / "sentinel").write_bytes(b"foreign destination bytes must remain")
                if case == "source":
                    target = source
                elif case == "source_dot":
                    target = source / ".." / "builder-sdk"
                elif case == "source_link":
                    target.symlink_to(source, target_is_directory=True)
                elif case == "source_child":
                    target = source / "new-component"
                elif case == "root_file":
                    target.write_bytes(b"owned file")
                elif case in ("root_link", "root_dangling"):
                    target.symlink_to(external if case == "root_link" else root / "absent", target_is_directory=True)
                else:
                    target.mkdir()
                    component, kind = case.split("_", 1)
                    path = target / component
                    if kind == "dir":
                        path.mkdir()
                        (path / "sentinel").write_bytes(b"existing component data")
                    elif kind == "file":
                        path.write_bytes(b"existing component file")
                    else:
                        path.symlink_to(external if kind == "link" else root / "absent", target_is_directory=True)
                before = {"source": self.tree_identity(source), "target": self.tree_identity(target),
                          "external": self.tree_identity(external)}
                result = subprocess.run(["bash", str(ROOT / "scripts/docker/install-hip-graph-runtime.sh")],
                    env=dict(os.environ, ROCM_PATH=str(source), HIP_RUNTIME_INSTALL_PREFIX=str(source),
                             HIP_RUNTIME_STAGE_PREFIX=str(target)), text=True, capture_output=True, timeout=30)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("HIP staging", result.stderr)
                self.assertEqual(before, {"source": self.tree_identity(source),
                    "target": self.tree_identity(target), "external": self.tree_identity(external)})

def main() -> int:
    """Run fast synthetic controls or authenticate both real preflight executables before main."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, action="append", default=[])
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--readelf", default="readelf")
    parser.add_argument("--wrong-runtime-fixture", type=Path)
    args = parser.parse_args()
    if not args.binary:
        if args.wrong_runtime_fixture:
            parser.error("--wrong-runtime-fixture requires --binary")
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(HIPRuntimeBindingTests))
        return 0 if result.wasSuccessful() else 1
    authority = HIPRuntimeAuthority.from_source(args.source_root.resolve(strict=True))
    for binary in args.binary:
        binary = binary.resolve(strict=True)
        observations = verify_environments(binary, authority, args.readelf, dict(os.environ))
        for mode, runtime in observations.items():
            print(f"PASS HIP runtime binding: {binary} {mode} {authority.soname} -> {runtime}")
        if args.wrong_runtime_fixture:
            verify_shadow_rejection(binary, authority, args.readelf, args.wrong_runtime_fixture)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
