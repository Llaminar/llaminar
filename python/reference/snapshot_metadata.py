#!/usr/bin/env python3
"""Content identity and atomic metadata publication for parity references.

Production parity snapshots are expensive, long-lived artifacts.  A directory
name or model pathname is not sufficient proof that the NumPy checkpoints came
from the GGUF bytes exercised by the native runner.  This module gives every
reference pack an exact model, prompt, tokenizer-output, and decode-depth
identity and publishes ``metadata.txt`` only after all snapshot files exist.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REFERENCE_IDENTITY_VERSION = 1
REFERENCE_ENGINE = "pytorch"
REFERENCE_DEVICE = "cpu"
REFERENCE_DTYPE = "float32"
_HASH_CHUNK_BYTES = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of one regular file using bounded memory."""

    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(
            f"production parity model identity requires a regular file: {resolved}"
        )

    digest = hashlib.sha256()
    with resolved.open("rb") as source:
        while chunk := source.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    """Hash the exact UTF-8 bytes of a prompt or canonical scalar field."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReferenceIdentity:
    """Exact provenance fields shared by every parity snapshot generator."""

    model_sha256: str
    prompt_sha256: str
    token_ids_sha256: str
    decode_steps: int

    def metadata_lines(self) -> list[str]:
        """Serialize stable scalar fields in the C++ loader's line format."""

        return [
            f"reference_identity_version: {REFERENCE_IDENTITY_VERSION}",
            f"reference_engine: {REFERENCE_ENGINE}",
            f"reference_device: {REFERENCE_DEVICE}",
            f"reference_dtype: {REFERENCE_DTYPE}",
            f"model_sha256: {self.model_sha256}",
            f"prompt_sha256: {self.prompt_sha256}",
            f"token_ids_sha256: {self.token_ids_sha256}",
            f"decode_steps: {self.decode_steps}",
        ]


def build_reference_identity(
    model_path: str | Path,
    prompt: str,
    token_ids: Iterable[int],
    decode_steps: int,
) -> ReferenceIdentity:
    """Build a reference identity from the exact inference inputs."""

    if decode_steps < 0:
        raise ValueError("decode_steps must be non-negative")
    canonical_tokens = ",".join(str(int(token)) for token in token_ids)
    if not canonical_tokens:
        raise ValueError("reference tokenizer produced no token IDs")
    return ReferenceIdentity(
        model_sha256=sha256_file(Path(model_path)),
        prompt_sha256=sha256_text(prompt),
        token_ids_sha256=sha256_text(canonical_tokens),
        decode_steps=decode_steps,
    )


def write_metadata_atomically(path: Path, lines: Iterable[str]) -> None:
    """Publish a complete metadata marker with flush/fsync/rename ordering."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(line.rstrip("\n") for line in lines) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
