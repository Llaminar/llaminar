"""Seal and verify immutable NativeVNNI benchmark corpora.

A corpus is the expensive evidence transaction, not one particular policy fit.
Canonical timing rows, isolated profiler evidence, route manifests, refinement
plans, and their raw profiler reports are therefore retained together and may
be mined repeatedly without launching kernels again.  The small JSON manifest
is ordinary Git metadata; large payloads beneath the canonical corpus root are
tracked by Git LFS.

The bundle format intentionally preserves refresh-script filenames.  A fit-only
transaction can materialize the directory into a disposable workspace and run
the existing backend compiler without translating or hand-selecting evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

from .shape_manifest import load_shape_manifest


SCHEMA_VERSION = "native-vnni-corpus-bundle-v1"
MANIFEST_NAME = "corpus.manifest.json"
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"
SUPPORTED_BACKENDS = frozenset(("cpu", "cpu-prefill", "cuda", "rocm"))
_IGNORED_DIRECTORY_NAMES = frozenset(("fit-cache", "policy_fit_cache"))
_PRODUCTION_REQUIRED_PAYLOADS: dict[str, tuple[str, ...]] = {
    "cuda": (
        "CUDANativeVNNIGemvDispatchHeuristicGenerated.inc",
        "cuda_decode_m1.development.csv",
        "cuda_decode_m1.development.timing.csv",
        "cuda_decode_m1.sealed.csv",
        "cuda_decode_m1.sealed.timing.csv",
        "cuda_decode_verifier.csv",
        "cuda_decode_verifier.timing.csv",
        "cuda_decode_m1_common_observations.csv",
        "cuda_profiler_requests.json",
        "cuda_profiler_evidence.json",
        "cuda_decode_common_observations.csv",
        "cuda_final_profiler_requests.json",
        "cuda_final_profiler_evidence.json",
        "cuda_decode_policy.json",
    ),
    "rocm": (
        "ROCmNativeVNNIDecodeDispatchGenerated.inc",
        "rocm_decode_fast.development.csv",
        "rocm_decode_fast.development.timing.csv",
        "rocm_decode_fast.sealed.csv",
        "rocm_decode_fast.sealed.timing.csv",
        "rocm_decode_verifier.csv",
        "rocm_decode_verifier.timing.csv",
        "rocm_decode_fast_common_observations.csv",
        "rocm_profiler_requests.json",
        "rocm_profiler_evidence.json",
        "rocm_decode_common_observations.csv",
        "rocm_final_profiler_requests.json",
        "rocm_final_profiler_evidence.json",
        "rocm_decode_policy.json",
    ),
    "cpu": (
        "CPUNativeVNNIVerifierRowsPolicyGenerated.inc",
        "cpu_verifier_rows.development.csv",
        "cpu_verifier_rows.development.timing.csv",
        "cpu_verifier_rows.sealed.csv",
        "cpu_verifier_rows.sealed.timing.csv",
        "cpu_verifier_rows_common_observations.csv",
        "cpu_profiler_requests.json",
        "cpu_profiler_evidence.json",
        "cpu_final_profiler_requests.json",
        "cpu_final_profiler_evidence.json",
        "cpu_verifier_rows_policy.json",
    ),
    "cpu-prefill": (
        "CPUNativeVNNIPrefillPolicyGenerated.inc",
        "cpu_prefill_sweep.development-v8.csv",
        "cpu_prefill_sweep.development-v8.timing.csv",
        "cpu_prefill_sweep.sealed-v8.csv",
        "cpu_prefill_sweep.sealed-v8.timing.csv",
        "cpu_prefill_common_observations.v8.csv",
        "cpu_prefill_profiler_source_observations.csv",
        "cpu_prefill_profiler_requests.json",
        "cpu_prefill_profiler_evidence.json",
        "cpu_prefill_final_profiler_requests.json",
        "cpu_prefill_final_profiler_evidence.json",
        "cpu_prefill_frozen_policy.v8.json",
        "cpu_prefill_sealed_witness_plan.v8.json",
        "cpu_prefill_policy.v8.json",
    ),
}


class CorpusBundleError(ValueError):
    """Raised when corpus evidence is incomplete, mutable, or unauthenticated."""


@dataclass(frozen=True, order=True)
class CorpusFile:
    """One content-addressed regular file owned by a corpus bundle."""

    path: str
    size_bytes: int
    sha256: str

    def to_json(self) -> dict[str, object]:
        """Return the canonical JSON representation used by the manifest."""

        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


def _sha256(path: Path) -> str:
    """Hash one file without loading an LFS-sized payload into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _is_lfs_pointer(path: Path) -> bool:
    """Return whether a checkout contains pointer text instead of evidence."""

    try:
        with path.open("rb") as handle:
            return handle.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX
    except OSError as error:
        raise CorpusBundleError(f"cannot read corpus file {path}: {error}") from error


def _validate_relative_path(raw: str) -> PurePosixPath:
    """Reject absolute and parent-traversing paths before filesystem access."""

    path = PurePosixPath(raw)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise CorpusBundleError(f"unsafe corpus-relative path: {raw!r}")
    return path


def _is_sha256(value: object) -> bool:
    """Return whether ``value`` is one canonical lowercase SHA-256 identity."""

    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    hexadecimal = value.removeprefix("sha256:")
    return len(hexadecimal) == 64 and all(
        character in "0123456789abcdef" for character in hexadecimal
    )


def _iter_payloads(directory: Path) -> Iterable[Path]:
    """Yield stable corpus payload paths while rejecting partial publication."""

    for root, directory_names, file_names in os.walk(directory):
        directory_names[:] = sorted(
            name
            for name in directory_names
            if name not in _IGNORED_DIRECTORY_NAMES
        )
        root_path = Path(root)
        for name in sorted(file_names):
            path = root_path / name
            relative = path.relative_to(directory)
            if relative.as_posix() == MANIFEST_NAME:
                continue
            if name.endswith(".inprogress") or name.endswith(".lock"):
                raise CorpusBundleError(
                    f"cannot seal partial corpus artifact: {relative}"
                )
            if path.is_symlink() or not path.is_file():
                raise CorpusBundleError(
                    f"corpus payload must be a regular file: {relative}"
                )
            yield path


def _source_record(path: Path, repository_root: Path) -> dict[str, str]:
    """Bind one checked-in policy inventory source to its repository path."""

    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError as error:
        raise CorpusBundleError(
            f"shape inventory source is outside repository: {path}"
        ) from error
    if not resolved.is_file():
        raise CorpusBundleError(f"shape inventory source is missing: {path}")
    return {"path": relative, "sha256": _sha256(resolved)}


def _manifest_digest(payload: dict[str, object]) -> str:
    """Hash policy-relevant manifest content without timestamps or self-ID."""

    authenticated = {
        key: value
        for key, value in payload.items()
        if key not in {"corpus_id", "created_utc"}
    }
    encoded = json.dumps(
        authenticated,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def configuration_digest(refresh_arguments: Sequence[str]) -> str:
    """Return an unambiguous identity for forwarded collection arguments."""

    encoded = json.dumps(
        list(refresh_arguments),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def required_payloads(backend: str, profile: str) -> tuple[str, ...]:
    """Return artifacts proving one backend transaction reached certification."""

    if profile != "all":
        return ()
    return _PRODUCTION_REQUIRED_PAYLOADS[backend]


def _validate_backend_completeness(
    backend: str,
    profile: str,
    payload_paths: Iterable[str],
) -> None:
    """Reject checkpoints and fit diagnostics masquerading as full corpora."""

    present = set(payload_paths)
    missing = sorted(set(required_payloads(backend, profile)) - present)
    if missing:
        raise CorpusBundleError(
            "production corpus is incomplete; missing payloads: "
            + ", ".join(missing)
        )


def seal_corpus(
    directory: Path,
    *,
    backend: str,
    profile: str,
    architecture: str,
    repository_root: Path,
    inventory_sources: Sequence[Path],
    refresh_arguments: Sequence[str] = (),
) -> dict[str, object]:
    """Validate and atomically seal a complete refresh output directory.

    The caller must invoke this only after collection, fitting, certification,
    and optional installation have succeeded.  Once sealed, any payload change
    invalidates the manifest and requires a new corpus generation directory.
    """

    if backend not in SUPPORTED_BACKENDS:
        raise CorpusBundleError(f"unsupported corpus backend: {backend}")
    if not isinstance(profile, str) or not profile.strip():
        raise CorpusBundleError("corpus profile identity must be non-empty")
    if not isinstance(architecture, str) or not architecture.strip():
        raise CorpusBundleError("corpus architecture identity must be non-empty")
    directory = directory.resolve()
    if not directory.is_dir():
        raise CorpusBundleError(f"corpus directory does not exist: {directory}")
    if (directory / MANIFEST_NAME).exists():
        raise CorpusBundleError(
            "corpus is already sealed; create a new generation instead"
        )
    files = []
    for path in _iter_payloads(directory):
        relative = path.relative_to(directory).as_posix()
        if _is_lfs_pointer(path):
            raise CorpusBundleError(
                f"cannot seal unresolved Git LFS pointer: {relative}"
            )
        files.append(CorpusFile(relative, path.stat().st_size, _sha256(path)))
    if not files:
        raise CorpusBundleError("cannot seal an empty corpus")
    _validate_backend_completeness(
        backend,
        profile,
        (entry.path for entry in files),
    )

    sources = sorted(
        (_source_record(path, repository_root) for path in inventory_sources),
        key=lambda record: record["path"],
    )
    if not sources:
        raise CorpusBundleError("at least one shape inventory source is required")
    shape_inventory_digest = load_shape_manifest().digest()
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "backend": backend,
        "profile": profile,
        "architecture": architecture,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "shape_inventory_digest": shape_inventory_digest,
        "shape_inventory_sources": sources,
        "refresh_arguments": list(refresh_arguments),
        "configuration_digest": configuration_digest(refresh_arguments),
        "files": [entry.to_json() for entry in sorted(files)],
    }
    payload["corpus_id"] = _manifest_digest(payload)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = directory / f"{MANIFEST_NAME}.inprogress"
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(directory / MANIFEST_NAME)
    return payload


def verify_corpus(
    manifest_path: Path,
    *,
    repository_root: Path,
    require_current_inventory: bool = True,
) -> dict[str, object]:
    """Authenticate every bundle payload and its shared shape inventory."""

    if _is_lfs_pointer(manifest_path):
        raise CorpusBundleError("corpus manifest itself is an unresolved LFS pointer")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CorpusBundleError(f"cannot read corpus manifest: {error}") from error
    required = {
        "schema_version",
        "backend",
        "profile",
        "architecture",
        "created_utc",
        "shape_inventory_digest",
        "shape_inventory_sources",
        "refresh_arguments",
        "configuration_digest",
        "files",
        "corpus_id",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise CorpusBundleError("corpus manifest root schema is invalid")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise CorpusBundleError("corpus manifest schema version is unsupported")
    if payload["backend"] not in SUPPORTED_BACKENDS:
        raise CorpusBundleError("corpus manifest backend is unsupported")
    if not isinstance(payload["profile"], str) or not payload["profile"].strip():
        raise CorpusBundleError("corpus manifest profile is invalid")
    if (
        not isinstance(payload["architecture"], str)
        or not payload["architecture"].strip()
    ):
        raise CorpusBundleError("corpus manifest architecture is invalid")
    if not isinstance(payload["created_utc"], str):
        raise CorpusBundleError("corpus creation timestamp is invalid")
    try:
        created = datetime.fromisoformat(payload["created_utc"])
    except ValueError as error:
        raise CorpusBundleError("corpus creation timestamp is invalid") from error
    if created.tzinfo is None:
        raise CorpusBundleError("corpus creation timestamp must include a timezone")
    for digest_name in (
        "shape_inventory_digest",
        "configuration_digest",
        "corpus_id",
    ):
        if not _is_sha256(payload[digest_name]):
            raise CorpusBundleError(f"corpus manifest {digest_name} is invalid")
    refresh_arguments = payload["refresh_arguments"]
    if not isinstance(refresh_arguments, list) or not all(
        isinstance(value, str) for value in refresh_arguments
    ):
        raise CorpusBundleError("corpus refresh arguments must be strings")
    if payload["configuration_digest"] != configuration_digest(refresh_arguments):
        raise CorpusBundleError("corpus collection configuration digest changed")
    directory = manifest_path.resolve().parent
    records = payload["files"]
    if not isinstance(records, list) or not records:
        raise CorpusBundleError("corpus manifest contains no payload files")
    seen = set()
    for raw in records:
        if not isinstance(raw, dict) or set(raw) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise CorpusBundleError("corpus file record schema is invalid")
        if (
            not isinstance(raw["path"], str)
            or type(raw["size_bytes"]) is not int
            or raw["size_bytes"] < 0
            or not _is_sha256(raw["sha256"])
        ):
            raise CorpusBundleError("corpus file record values are invalid")
        relative = _validate_relative_path(raw["path"])
        if relative.as_posix() in seen:
            raise CorpusBundleError(f"duplicate corpus file: {relative}")
        seen.add(relative.as_posix())
        path = directory.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file():
            raise CorpusBundleError(f"corpus payload is missing: {relative}")
        if _is_lfs_pointer(path):
            raise CorpusBundleError(
                f"unresolved Git LFS pointer; run git lfs pull: {relative}"
            )
        if path.stat().st_size != raw["size_bytes"]:
            raise CorpusBundleError(f"corpus payload size changed: {relative}")
        if _sha256(path) != raw["sha256"]:
            raise CorpusBundleError(f"corpus payload digest changed: {relative}")
    _validate_backend_completeness(
        str(payload["backend"]),
        str(payload["profile"]),
        seen,
    )

    sources = payload["shape_inventory_sources"]
    if not isinstance(sources, list) or not sources:
        raise CorpusBundleError("corpus has no shape inventory sources")
    for raw in sources:
        if not isinstance(raw, dict) or set(raw) != {"path", "sha256"}:
            raise CorpusBundleError("shape inventory source schema is invalid")
        if not isinstance(raw["path"], str) or not _is_sha256(raw["sha256"]):
            raise CorpusBundleError("shape inventory source values are invalid")
        relative = _validate_relative_path(raw["path"])
        source = repository_root.joinpath(*relative.parts)
        if not source.is_file() or _sha256(source) != raw["sha256"]:
            if require_current_inventory:
                raise CorpusBundleError(
                    f"shape inventory source changed: {relative}"
                )
    if (
        require_current_inventory
        and payload["shape_inventory_digest"] != load_shape_manifest().digest()
    ):
        raise CorpusBundleError(
            "corpus predates the current resolved shape inventory; collect a "
            "new generation for the missing overlays"
        )
    if payload["corpus_id"] != _manifest_digest(payload):
        raise CorpusBundleError("corpus manifest identity digest does not match")
    return payload


def _default_inventory_sources() -> tuple[Path, ...]:
    """Return every checked-in source contributing to the resolved inventory."""

    from .qwen_release_geometry import QWEN_RELEASE_CATALOG_PATH
    from .shape_manifest import MANIFEST_PATH

    return MANIFEST_PATH, QWEN_RELEASE_CATALOG_PATH


def main(argv: Sequence[str] | None = None) -> int:
    """Provide seal, verify, and identity commands for shell orchestration."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal = subparsers.add_parser("seal")
    seal.add_argument("--directory", type=Path, required=True)
    seal.add_argument("--backend", choices=sorted(SUPPORTED_BACKENDS), required=True)
    seal.add_argument("--profile", default="all")
    seal.add_argument("--architecture", required=True)
    seal.add_argument("--repository-root", type=Path, required=True)
    seal.add_argument("--inventory-source", type=Path, action="append")
    seal.add_argument("--refresh-argument", action="append", default=[])

    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--repository-root", type=Path, required=True)
    verify.add_argument("--allow-historical-inventory", action="store_true")

    subparsers.add_parser("inventory-id")
    configuration = subparsers.add_parser("configuration-id")
    configuration.add_argument("--refresh-argument", action="append", default=[])

    arguments = parser.parse_args(argv)
    if arguments.command == "seal":
        payload = seal_corpus(
            arguments.directory,
            backend=arguments.backend,
            profile=arguments.profile,
            architecture=arguments.architecture,
            repository_root=arguments.repository_root,
            inventory_sources=tuple(arguments.inventory_source or _default_inventory_sources()),
            refresh_arguments=tuple(arguments.refresh_argument),
        )
        print(payload["corpus_id"])
    elif arguments.command == "verify":
        payload = verify_corpus(
            arguments.manifest,
            repository_root=arguments.repository_root,
            require_current_inventory=not arguments.allow_historical_inventory,
        )
        print(payload["corpus_id"])
    elif arguments.command == "inventory-id":
        print(load_shape_manifest().digest())
    else:
        print(configuration_digest(tuple(arguments.refresh_argument)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
