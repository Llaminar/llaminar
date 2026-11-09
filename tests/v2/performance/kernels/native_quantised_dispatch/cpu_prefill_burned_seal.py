"""Preserve a failed CPU-prefill seal as typed development evidence.

A sealed corpus becomes development evidence as soon as its performance result
is inspected.  Reusing those measurements is desirable, but appending their raw
CSV to an older development transaction would falsely relabel the rows with the
older build provenance.  This module binds the raw aggregate, timing sidecar,
post-freeze witness plan, and the provenance of the process that actually
measured them into one content-addressed transaction.

The transaction does not make a failed seal installable.  It only permits a
later learner generation to authenticate and adapt the measurements under
their original identity.  That changed policy must still be certified against
a new untouched seal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence


SCHEMA_VERSION = "cpu-prefill-burned-seal-development-v1"


def _sha256_file(path: Path) -> str:
    """Return the streaming SHA-256 identity of one immutable payload."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_digest(value: object) -> str:
    """Hash one JSON-compatible mapping using the repository convention."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: object, context: str) -> str:
    """Return a canonical SHA-256 string or reject the manifest."""

    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(
            character not in "0123456789abcdef"
            for character in value.removeprefix("sha256:")
        )
    ):
        raise ValueError(f"{context} is not a canonical SHA-256 identity")
    return value


def _require_mapping(
    value: object,
    required: set[str],
    context: str,
) -> Mapping[str, object]:
    """Return one strict object so omitted provenance cannot be inferred."""

    if not isinstance(value, dict) or set(value) != required:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(
            f"{context} schema is invalid: expected={sorted(required)} "
            f"actual={actual}"
        )
    return value


@dataclass(frozen=True)
class BurnedSealArtifact:
    """One manifest-relative immutable evidence payload."""

    path: PurePosixPath
    sha256: str

    @classmethod
    def from_mapping(
        cls,
        raw: object,
        context: str,
    ) -> "BurnedSealArtifact":
        """Parse one strict artifact and reject unsafe path syntax."""

        value = _require_mapping(raw, {"path", "sha256"}, context)
        raw_path = value["path"]
        if (
            not isinstance(raw_path, str)
            or not raw_path
            or any(character in raw_path for character in "\0\n\r\t")
        ):
            raise ValueError(f"{context} path is invalid")
        path = PurePosixPath(raw_path)
        if path.is_absolute():
            raise ValueError(f"{context} path must be manifest-relative")
        return cls(
            path=path,
            sha256=_require_sha256(value["sha256"], f"{context}.sha256"),
        )

    def resolve(self, manifest_path: Path) -> Path:
        """Resolve this payload relative to its transaction manifest."""

        return manifest_path.resolve().parent.joinpath(*self.path.parts).resolve()

    def canonical_mapping(self) -> dict[str, str]:
        """Return this artifact's stable serialized form."""

        return {"path": self.path.as_posix(), "sha256": self.sha256}


@dataclass(frozen=True)
class BurnedSealProvenance:
    """The exact process and serial-oracle identity of one failed seal."""

    run_id: str
    git_revision: str
    build_id: str
    compiler_id: str
    architecture_class: str
    device_name: str
    driver_runtime: str
    serial_m1_policy_hash: str

    @classmethod
    def from_mapping(cls, raw: object) -> "BurnedSealProvenance":
        """Parse complete provenance without fallback to development fields."""

        fields = {
            "run_id",
            "git_revision",
            "build_id",
            "compiler_id",
            "architecture_class",
            "device_name",
            "driver_runtime",
            "serial_m1_policy_hash",
        }
        value = _require_mapping(raw, fields, "burned seal provenance")
        if any(not isinstance(value[field], str) or not value[field] for field in fields):
            raise ValueError("burned seal provenance fields must be non-empty strings")
        _require_sha256(
            value["serial_m1_policy_hash"],
            "burned seal serial M=1 policy hash",
        )
        return cls(**{field: str(value[field]) for field in fields})

    def canonical_mapping(self) -> dict[str, str]:
        """Return every provenance discriminator in stable field order."""

        return {
            "run_id": self.run_id,
            "git_revision": self.git_revision,
            "build_id": self.build_id,
            "compiler_id": self.compiler_id,
            "architecture_class": self.architecture_class,
            "device_name": self.device_name,
            "driver_runtime": self.driver_runtime,
            "serial_m1_policy_hash": self.serial_m1_policy_hash,
        }


@dataclass(frozen=True)
class CPUPrefillBurnedSealTransaction:
    """Authenticated failed-seal payloads admitted only as development data."""

    path: Path
    transaction_digest: str
    aggregate: BurnedSealArtifact
    timing: BurnedSealArtifact
    witness_plan: BurnedSealArtifact
    provenance: BurnedSealProvenance


def read_cpu_prefill_burned_seal_transaction(
    path: Path,
) -> CPUPrefillBurnedSealTransaction:
    """Read, schema-check, and authenticate one burned-seal transaction."""

    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    root = _require_mapping(
        raw,
        {
            "schema_version",
            "transaction_digest",
            "aggregate",
            "timing",
            "witness_plan",
            "provenance",
        },
        "burned seal transaction",
    )
    if root["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported CPU prefill burned-seal transaction")
    authenticated = dict(root)
    claimed_digest = _require_sha256(
        authenticated.pop("transaction_digest"),
        "burned seal transaction digest",
    )
    if _canonical_digest(authenticated) != claimed_digest:
        raise ValueError("CPU prefill burned-seal transaction digest changed")

    transaction = CPUPrefillBurnedSealTransaction(
        path=path,
        transaction_digest=claimed_digest,
        aggregate=BurnedSealArtifact.from_mapping(root["aggregate"], "aggregate"),
        timing=BurnedSealArtifact.from_mapping(root["timing"], "timing"),
        witness_plan=BurnedSealArtifact.from_mapping(
            root["witness_plan"], "witness_plan"
        ),
        provenance=BurnedSealProvenance.from_mapping(root["provenance"]),
    )
    for artifact in (
        transaction.aggregate,
        transaction.timing,
        transaction.witness_plan,
    ):
        payload = artifact.resolve(path)
        if payload.is_symlink() or not payload.is_file():
            raise ValueError(f"burned seal payload is missing: {payload}")
        if _sha256_file(payload) != artifact.sha256:
            raise ValueError(f"burned seal payload digest changed: {payload}")
    return transaction


def write_cpu_prefill_burned_seal_transaction(
    output: Path,
    aggregate: Path,
    timing: Path,
    witness_plan: Path,
    provenance: BurnedSealProvenance,
) -> CPUPrefillBurnedSealTransaction:
    """Atomically author one transaction from already-retained seal files."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def artifact(path: Path) -> dict[str, str]:
        resolved = Path(path).resolve()
        return {
            "path": PurePosixPath(os.path.relpath(resolved, output.parent.resolve())).as_posix(),
            "sha256": _sha256_file(resolved),
        }

    authenticated = {
        "schema_version": SCHEMA_VERSION,
        "aggregate": artifact(aggregate),
        "timing": artifact(timing),
        "witness_plan": artifact(witness_plan),
        "provenance": provenance.canonical_mapping(),
    }
    payload = {
        **authenticated,
        "transaction_digest": _canonical_digest(authenticated),
    }
    temporary = output.with_name(f"{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return read_cpu_prefill_burned_seal_transaction(output)


def main(argv: Sequence[str] | None = None) -> int:
    """Create or authenticate a burned-seal development transaction."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--aggregate", type=Path, required=True)
    create.add_argument("--timing", type=Path, required=True)
    create.add_argument("--witness-plan", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    for field in BurnedSealProvenance.__dataclass_fields__:
        create.add_argument(f"--{field.replace('_', '-')}", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    arguments = parser.parse_args(argv)

    if arguments.command == "create":
        provenance = BurnedSealProvenance(**{
            field: getattr(arguments, field)
            for field in BurnedSealProvenance.__dataclass_fields__
        })
        transaction = write_cpu_prefill_burned_seal_transaction(
            arguments.output,
            arguments.aggregate,
            arguments.timing,
            arguments.witness_plan,
            provenance,
        )
    else:
        transaction = read_cpu_prefill_burned_seal_transaction(
            arguments.manifest
        )
    print(
        "authenticated CPU prefill burned seal "
        f"{transaction.transaction_digest} -> {transaction.path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
