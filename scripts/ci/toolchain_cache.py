#!/usr/bin/env python3
"""Admit complete, dependency-bound OCI toolchains for ephemeral cloud builders.

The cache key describes only the committed dependency context and CPU ISA.
Building that restricted context makes an undeclared dependency a hard error.
An exact cache hit restores a complete image, which Buildx consumes as the
``toolchain`` named context; intermediate solver cache hits are not required.
OCI content digests and image labels bind the restored bytes to those inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
TOOLCHAIN_ROOT_ENV = "LLAMINAR_DOCKER_TOOLCHAIN_ROOT"
ISAS = ("AVX512", "AVX2")
SCHEMA = 1
# This is the complete available context, not merely a cache-key approximation.
# Add dependencies here when adding a toolchain COPY outside these roots.
INPUTS = ("Dockerfile", ".dockerignore", "requirements.txt", "scripts/docker",
          "scripts/build", "scripts/ci/toolchain_cache.py",
          ".github/actions/cache-develop-toolchain/action.yml")
INPUT_LABEL = "org.llaminar.toolchain_inputs"
ISA_LABEL = "org.llaminar.toolchain_isa"
INDEX_TYPE = "application/vnd.oci.image.index.v1+json"
MANIFEST_TYPE = "application/vnd.oci.image.manifest.v1+json"


def git(*arguments: str) -> bytes:
    """Read committed objects without modifying the checkout or its index."""
    return subprocess.check_output(["git", "-C", str(ROOT), *arguments])


def identity(tree: str, isa: str) -> dict:
    """Bind the available dependency context, policy version and explicit ISA."""
    if isa not in ISAS:
        raise ValueError("toolchain cache requires AVX512 or AVX2")
    entries = git("ls-tree", "-r", "-z", tree, "--", *INPUTS).split(b"\0")
    objects = [entry.decode() for entry in entries if entry]
    paths = [entry.split("\t", 1)[1] for entry in objects]
    if any(required not in paths for required in ("Dockerfile", ".dockerignore", "requirements.txt")):
        raise ValueError("toolchain dependency context is incomplete")
    if any(entry.split()[0] not in ("100644", "100755") for entry in objects):
        raise ValueError("toolchain inputs must be regular committed files")
    value = {"schema": SCHEMA, "cpu_isa": isa, "objects": objects}
    value["fingerprint"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value


def prepare(root: Path, isa: str, github_output: Path) -> None:
    """Stage one restricted source context and immutable per-ISA cache keys."""
    root.mkdir(parents=True, exist_ok=False)
    context = root / "inputs"
    context.mkdir()
    tree = git("rev-parse", "HEAD^{tree}").decode().strip()
    archive = git("archive", "--format=tar", tree, "--", *INPUTS)
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        source.extractall(context, filter="data")
    with github_output.open("a") as output:
        output.write(f"context={context}\nroot={root}\n")
        for selected in ISAS if isa == "both" else (isa,):
            admitted = identity(tree, selected)
            suffix = selected.lower()
            output.write(f"key_{suffix}=llaminar-toolchain-oci-v{SCHEMA}-{suffix}-{admitted['fingerprint']}\n")
            output.write(f"fingerprint_{suffix}={admitted['fingerprint']}\n")
            output.write(f"path_{suffix}={root / suffix}\n")


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

    Cache restoration validates all layer digests once. The source image driver
    rechecks the manifest/config and layer extents; BuildKit owns validation of
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
    print(f"[toolchain-cache] consume isa={isa} manifest={admitted['manifest']} "
          f"inputs={admitted['fingerprint']}", flush=True)
    return ["--build-context", f"toolchain=oci-layout://{layout}@{admitted['manifest']}"]


def main() -> None:
    """Prepare keys or verify/save an independently restored toolchain image."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cpu-isa", choices=(*ISAS, "both"), required=True)
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--cache-hit", choices=("true", "false"), default="false")
    args = parser.parse_args()
    if not args.root.is_absolute():
        parser.error("--root must be absolute")
    if args.command == "prepare":
        if args.github_output is None:
            parser.error("prepare requires --github-output")
        prepare(args.root, args.cpu_isa, args.github_output)
    else:
        if args.cpu_isa == "both":
            parser.error("verify requires one explicit ISA")
        started = time.monotonic()
        tree = git("rev-parse", "HEAD^{tree}").decode().strip()
        receipt = admit(args.root / args.cpu_isa.lower(), identity(tree, args.cpu_isa), verify_layers=True)
        receipt.update(cache_restored=args.cache_hit == "true", validation_seconds=time.monotonic() - started)
        (args.root / f"{args.cpu_isa.lower()}-toolchain.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print("[toolchain-cache] " + json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
