"""Shared executable-closure provenance for production tuning corpora.

Production benchmark binaries link most dispatch and kernel-launch behavior
through ``libllaminar2_core.so``. A trustworthy corpus therefore identifies
both the small benchmark ELF and its resolved core shared object, together with
the CMake build type that produced them. These helpers centralize that contract
so dense and routed-MoE transactions cannot drift into different notions of a
valid producer.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Mapping


TRAINER_PROVENANCE_SCHEMA = "native-vnni-trainer-closure-v1"


def sha256_file(path: Path) -> str:
    """Return a streaming content digest for one immutable corpus input."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def mapping_digest(payload: Mapping[str, object]) -> str:
    """Digest a JSON mapping using one stable canonical representation."""

    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _cmake_cache_value(cache: Path, name: str) -> str | None:
    """Read one exact CMake cache assignment without invoking CMake."""

    prefix = f"{name}:"
    with cache.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(prefix) and "=" in line:
                return line.rstrip("\n").split("=", maxsplit=1)[1]
    return None


def _find_cmake_cache(binary: Path) -> Path | None:
    """Find the build-tree cache governing an executable."""

    resolved = Path(binary).resolve()
    for directory in (resolved.parent, *resolved.parents):
        candidate = directory / "CMakeCache.txt"
        if candidate.is_file():
            return candidate
    return None


def _linked_llaminar_core(binary: Path) -> Path | None:
    """Resolve the exact project core shared object loaded by a trainer."""

    completed = subprocess.run(
        ("ldd", str(Path(binary).resolve())),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        text=True,
    )
    match = re.search(
        r"^\s*libllaminar2_core\.so\s+=>\s+(\S+)\s+\(",
        completed.stdout,
        flags=re.MULTILINE,
    )
    if match is None:
        return None
    resolved = Path(match.group(1)).resolve()
    return resolved if resolved.is_file() else None


def trainer_provenance(binary: Path, backend: str) -> dict[str, object]:
    """Fingerprint the exact executable closure that measures a GPU corpus."""

    normalized = backend.strip().lower()
    if normalized not in {"cuda", "rocm"}:
        raise ValueError(f"unsupported production GPU backend {backend!r}")
    resolved_binary = Path(binary).resolve()
    if not resolved_binary.is_file() or not os.access(resolved_binary, os.X_OK):
        raise ValueError(
            f"{normalized} trainer binary is not executable: {binary}"
        )
    cache = _find_cmake_cache(resolved_binary)
    core = _linked_llaminar_core(resolved_binary)
    return {
        "schema_version": TRAINER_PROVENANCE_SCHEMA,
        "backend": normalized,
        "binary_name": resolved_binary.name,
        "binary_sha256": sha256_file(resolved_binary),
        "core_library_name": core.name if core is not None else None,
        "core_library_sha256": (
            sha256_file(core) if core is not None else None
        ),
        "cmake_build_type": (
            _cmake_cache_value(cache, "CMAKE_BUILD_TYPE")
            if cache is not None else None
        ),
    }


def require_release_trainer(provenance: Mapping[str, object]) -> None:
    """Reject evidence that cannot represent the deployed Release binary."""

    if provenance.get("schema_version") != TRAINER_PROVENANCE_SCHEMA:
        raise ValueError("trainer provenance schema is unsupported")
    if provenance.get("cmake_build_type") != "Release":
        raise ValueError(
            "production NativeVNNI corpora require a Release trainer; "
            f"observed {provenance.get('cmake_build_type')!r}"
        )
    if not provenance.get("core_library_sha256"):
        raise ValueError(
            "trainer does not resolve a hashed libllaminar2_core.so"
        )
