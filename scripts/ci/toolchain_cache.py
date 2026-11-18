#!/usr/bin/env python3
"""Build and authenticate the dependency-only OCI context on the selected BuildKit worker.

Local BuildKit layer reuse is optional. A complete per-ISA image and a verified
receipt are mandatory; remote cache availability never admits or vetoes them.
The ordinary Dockerfile remains the single source for direct and gated builds.
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
TOOLCHAIN_ROOT_ENV = "LLAMINAR_DOCKER_TOOLCHAIN_ROOT"
ISAS = ("AVX512", "AVX2")
SCHEMA = 2
# Bind invocation/admission policy as well as the actual bytes visible to the recipe.
POLICY = ("scripts/ci/toolchain_cache.py", ".github/actions/cache-develop-toolchain/action.yml")
INPUT_LABEL = "org.llaminar.toolchain_inputs"
ISA_LABEL = "org.llaminar.toolchain_isa"
INDEX_TYPE = "application/vnd.oci.image.index.v1+json"
MANIFEST_TYPE = "application/vnd.oci.image.manifest.v1+json"


def git(*arguments: str) -> bytes:
    """Read committed objects without modifying the checkout or its index."""
    return subprocess.check_output(["git", "-C", str(ROOT), *arguments])


@dataclass(frozen=True)
class DependencyContext:
    """One committed recipe and its local inputs, used for both identity and staging."""

    recipe: bytes
    files: tuple[tuple[str, str, str], ...]  # path, Git file mode, blob OID

    @classmethod
    def from_tree(cls, tree: str) -> DependencyContext:
        """Project the canonical recipe at its named source-build boundary, not a line count.

        Dependency COPYs use literal, repository-relative file or directory operands.
        Expand whole directories in Git (including modes), not from a second installer
        list or a dirty checkout. Unsupported input forms must not produce a guessed key.
        """
        dockerfile = git("show", f"{tree}:Dockerfile")
        sections = re.split(rb"(?im)^[ \t]*FROM[ \t]+toolchain[ \t]+AS[ \t]+builder[ \t]*\r?$", dockerfile)
        if len(sections) != 2:
            raise ValueError("toolchain recipe requires one FROM toolchain AS builder boundary")
        recipe = sections[0]
        required = {".dockerignore", *POLICY}
        lines = recipe.decode().splitlines()
        if any(re.match(r"\s*#\s*escape\s*=", line, re.I) for line in lines):
            raise ValueError("toolchain recipe requires the default Dockerfile escape")
        code = "\n".join(line for line in lines if not line.lstrip().startswith("#"))
        for line in code.replace("\\\n", "").splitlines():
            words = line.split()
            if not words:
                continue
            op = words[0].upper()
            if op in ("ADD", "ONBUILD") or "<<" in line or (
                    op == "RUN" and any(word.startswith("--mount=") and
                    (not word.startswith("--mount=type=cache,") or ",from=" in word) for word in words)):
                raise ValueError(f"unsupported toolchain context input: {line[:100]}")
            if op != "COPY":
                continue
            if len(words) < 3:
                raise ValueError("toolchain COPY requires literal local sources and a destination")
            for source in words[1:-1]:
                path = source.rstrip("/")
                if (not re.fullmatch(r"[\w.][\w./-]*", path, re.ASCII) or
                        any(part in ("", ".", "..") for part in path.split("/")) or path == "Dockerfile"):
                    raise ValueError(f"unsupported toolchain COPY input: {source}")
                required.add(path)
        raw = git("--literal-pathspecs", "ls-tree", "-rz", tree, "--", *sorted(required), "Dockerfile.dockerignore")
        files = []
        for entry in raw.split(b"\0"):
            if entry:
                header, path = entry.decode().split("\t", 1)
                mode, kind, oid = header.split()
                if kind != "blob" or mode not in ("100644", "100755"):
                    raise ValueError(f"toolchain input must be a regular committed file: {path}")
                files.append((path, mode, oid))
        paths = {path for path, _, _ in files}
        if not {".dockerignore", *POLICY}.issubset(paths):
            raise ValueError("missing committed toolchain ignore or invocation policy file")
        for source in sorted(required):
            if source not in paths and not any(path.startswith(source + "/") for path in paths):
                raise ValueError(f"missing committed toolchain input: {source}")
        return cls(recipe, tuple(files))

    def identity(self, isa: str) -> dict:
        """Bind recipe bytes, complete local input closure, invocation policy and ISA."""
        if isa not in ISAS:
            raise ValueError("toolchain requires AVX512 or AVX2")
        value = {"schema": SCHEMA, "cpu_isa": isa, "objects": self.files,
                 "recipe_sha256": hashlib.sha256(self.recipe).hexdigest()}
        value["fingerprint"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        return value

    def stage(self, path: Path) -> None:
        """Materialize exactly the fingerprinted blobs and modes, never a second recipe."""
        path.mkdir()
        (path / "Dockerfile").write_bytes(self.recipe)
        for name, mode, oid in self.files:
            target = path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(git("cat-file", "blob", oid))
            target.chmod(int(mode[-3:], 8))


def identity(tree: str, isa: str) -> dict:
    """Recompute the required toolchain identity independently for a source snapshot."""
    return DependencyContext.from_tree(tree).identity(isa)


def regular(path: Path) -> None:
    """Reject missing blobs and symlink substitutions in restored cache data."""
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"toolchain OCI file is missing or is a symlink: {path}")


def blob(layout: Path, descriptor: dict, *, contents: bool = True) -> bytes:
    """Authenticate one OCI descriptor before parsing or admitting its payload."""
    digest = descriptor.get("digest", "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError("toolchain OCI descriptor requires a SHA256 digest")
    path = layout / "blobs" / "sha256" / digest.split(":")[1]
    regular(path)
    if type(descriptor.get("size")) is not int or path.stat().st_size != descriptor["size"]:
        raise ValueError(f"toolchain OCI blob size mismatch: {digest}")
    if not contents:
        return b""
    if descriptor["size"] > 16 * 1024 * 1024:
        raise ValueError("toolchain OCI manifest/config is unexpectedly large")
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    if hasher.hexdigest() != digest.split(":")[1]:
        raise ValueError(f"toolchain OCI blob digest mismatch: {digest}")
    return path.read_bytes()


def admit(layout: Path, expected: dict, *, verify_layers: bool = False) -> dict:
    """Admit a single Linux/amd64 image bound to the exact dependency identity.

    Local OCI export validates all layer digests once. The source image driver
    rechecks the manifest/config, receipt and layer extents; BuildKit owns validation of
    the layer bytes as it imports the pinned OCI context into its content store.
    """
    if layout.is_symlink() or (layout / "blobs").is_symlink() or (layout / "blobs/sha256").is_symlink():
        raise ValueError("toolchain OCI directories must not be symlinks")
    for name in ("oci-layout", "index.json"):
        regular(layout / name)
    if json.loads((layout / "oci-layout").read_text()).get("imageLayoutVersion") != "1.0.0":
        raise ValueError("unsupported toolchain OCI layout version")
    index = json.loads((layout / "index.json").read_text())
    manifests = index.get("manifests", [])
    if index.get("schemaVersion") != 2 or len(manifests) != 1:
        raise ValueError("toolchain OCI layout requires exactly one image")
    descriptor = manifests[0]
    # Buildx may wrap a single-platform image in an additional OCI index.
    if descriptor.get("mediaType") == INDEX_TYPE:
        manifests = json.loads(blob(layout, descriptor)).get("manifests", [])
        if len(manifests) != 1:
            raise ValueError("toolchain OCI index requires one unattested image")
        descriptor = manifests[0]
    if descriptor.get("mediaType") != MANIFEST_TYPE:
        raise ValueError("toolchain artifact is not an OCI image manifest")
    manifest = json.loads(blob(layout, descriptor))
    config = json.loads(blob(layout, manifest["config"]))
    if config.get("os") != "linux" or config.get("architecture") != "amd64":
        raise ValueError("toolchain artifact must target linux/amd64")
    labels = config.get("config", {}).get("Labels", {})
    if labels.get(INPUT_LABEL) != expected["fingerprint"] or labels.get(ISA_LABEL) != expected["cpu_isa"]:
        raise ValueError("toolchain artifact does not match the admitted dependency inputs/ISA")
    layers = manifest.get("layers", [])
    if not layers:
        raise ValueError("toolchain OCI image has no filesystem layers")
    for layer in layers:
        if verify_layers:
            # Stream large layers; do not retain multi-GiB compiler images in RAM.
            path = layout / "blobs/sha256" / layer.get("digest", "").removeprefix("sha256:")
            blob(layout, layer, contents=False)
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if "sha256:" + actual != layer["digest"]:
                raise ValueError("toolchain OCI layer digest mismatch")
        else:
            blob(layout, layer, contents=False)
    return {"schema": SCHEMA, "cpu_isa": expected["cpu_isa"],
            "fingerprint": expected["fingerprint"], "manifest": descriptor["digest"],
            "layer_bytes": sum(layer["size"] for layer in layers), "layer_count": len(layers)}


def build_context_arguments(isa: str, source_tree: str) -> list[str]:
    """Require the prepared artifact whenever cloud CI declares its authority."""
    raw = os.environ.get(TOOLCHAIN_ROOT_ENV)
    if raw is None:
        return []
    root = Path(raw)
    if not raw or not root.is_absolute() or not root.is_dir():
        raise ValueError(f"{TOOLCHAIN_ROOT_ENV} must name the prepared absolute directory")
    layout = root / isa.lower()
    admitted = admit(layout, identity(source_tree, isa))
    receipt = root / f"{isa.lower()}-toolchain.json"
    regular(receipt)
    verified = json.loads(receipt.read_bytes())
    if not isinstance(verified, dict) or any(verified.get(key) != value for key, value in admitted.items()):
        raise ValueError("toolchain receipt does not match the admitted OCI artifact")
    print(f"[toolchain-cache] consume isa={isa} manifest={admitted['manifest']} "
          f"inputs={admitted['fingerprint']}", flush=True)
    return ["--build-context", f"toolchain=oci-layout://{layout}@{admitted['manifest']}"]


def verify(root: Path, expected: dict) -> None:
    """Issue a receipt only after hashing all layers of a completed local export."""
    started = time.monotonic()
    suffix = expected["cpu_isa"].lower()
    path = root / f"{suffix}-toolchain.json"
    path.unlink(missing_ok=True)  # A rejected re-verification cannot leave yesterday's proof.
    receipt = admit(root / suffix, expected, verify_layers=True)
    receipt["validation_seconds"] = time.monotonic() - started
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print("[toolchain-cache] " + json.dumps(receipt), flush=True)


def build_toolchains(root: Path, isa: str) -> None:
    """Solve the real dependency stage on the retained worker and verify each ISA in turn.

    Local BuildKit DAG hits reuse SDK/RCCL layers; there is no separate cache-only
    solve, GHA importer/exporter or remote-archive round trip. Cold workers must
    build the same full stage. A failed solve never authorizes an old export.
    """
    if not root.is_absolute() or isa not in (*ISAS, "both"):
        raise ValueError("toolchain build requires an absolute output root and a shipping ISA or both")
    root.mkdir(parents=True, exist_ok=False)
    context = root / "inputs"
    inputs = DependencyContext.from_tree(git("rev-parse", "HEAD^{tree}").decode().strip())
    inputs.stage(context)
    for selected in ISAS if isa == "both" else (isa,):
        expected = inputs.identity(selected)
        layout = root / selected.lower()
        subprocess.run(["docker", "buildx", "build", "--progress=plain", "--network=host",
                        "--platform=linux/amd64", "--target=toolchain", "--provenance=false", "--sbom=false",
                        "--build-arg", f"LLAMINAR_CPU_ISA={selected}", "--label", f"{ISA_LABEL}={selected}",
                        "--label", f"{INPUT_LABEL}={expected['fingerprint']}",
                        "--output", f"type=oci,dest={layout},tar=false", str(context)], check=True)
        verify(root, expected)


def main() -> None:
    """Prepare the dependency authority consumed by the existing source-image driver."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cpu-isa", choices=(*ISAS, "both"), required=True)
    args = parser.parse_args()
    build_toolchains(args.root, args.cpu_isa)


if __name__ == "__main__":
    main()
