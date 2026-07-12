"""Shape-group development/sealed split manifests and opaque sealed handles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .corpus import ObservationCorpus


class Partition(str, Enum):
    """Permitted corpus partition for one indivisible logical shape group."""

    DEVELOPMENT = "development"
    SEALED = "sealed"


@dataclass(frozen=True)
class SplitManifest:
    """Versioned immutable assignment of every shape group to one partition."""

    version: str
    seed: str
    assignments: Mapping[str, Partition]

    def validate(self, corpus: ObservationCorpus) -> None:
        """Require one complete, non-overlapping assignment for the corpus."""

        groups = set(corpus.shape_groups())
        assigned = set(self.assignments)
        missing = sorted(groups - assigned)
        unexpected = sorted(assigned - groups)
        if missing or unexpected:
            raise ValueError(
                f"split manifest mismatch: missing={missing} unexpected={unexpected}"
            )
        if not any(value == Partition.DEVELOPMENT for value in self.assignments.values()):
            raise ValueError("split manifest has no development shape groups")
        if not any(value == Partition.SEALED for value in self.assignments.values()):
            raise ValueError("split manifest has no sealed shape groups")

    def digest(self) -> str:
        """Hash the exact shape assignment used by a policy certificate."""

        payload = {
            "version": self.version,
            "seed": self.seed,
            "assignments": {
                key: value.value for key, value in sorted(self.assignments.items())
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def partition(self, corpus: ObservationCorpus) -> tuple[ObservationCorpus, "SealedPartition"]:
        """Return visible development rows and an unopened sealed commitment."""

        self.validate(corpus)
        development = ObservationCorpus(
            row
            for row in corpus
            if self.assignments[row.shape_group_id] == Partition.DEVELOPMENT
        )
        sealed = ObservationCorpus(
            row
            for row in corpus
            if self.assignments[row.shape_group_id] == Partition.SEALED
        )
        return development, SealedPartition(sealed, self.digest())


def deterministic_split_manifest(
    corpus: ObservationCorpus,
    *,
    seed: str,
    sealed_fraction: float = 0.2,
    version: str = "native-vnni-shape-split-v1",
) -> SplitManifest:
    """Create a deterministic shape-group split without candidate-row leakage.

    This helper is intended for development and fixture corpora. Production
    refreshes should check in an explicit reviewed manifest that also proves
    per-leaf boundary/tail coverage before measurements are collected.
    """

    if not 0.0 < sealed_fraction < 1.0:
        raise ValueError("sealed_fraction must be strictly between zero and one")
    groups = corpus.shape_groups()
    if len(groups) < 3:
        raise ValueError("at least three shape groups are required for a sealed split")

    scored = sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{version}\0{seed}\0{group}".encode()).digest(),
    )
    sealed_count = max(1, min(len(scored) - 2, round(len(scored) * sealed_fraction)))
    sealed = set(scored[:sealed_count])
    return SplitManifest(
        version=version,
        seed=seed,
        assignments={
            group: Partition.SEALED if group in sealed else Partition.DEVELOPMENT
            for group in groups
        },
    )


class SealedPartition:
    """Opaque sealed rows that can open only against a frozen generic digest."""

    def __init__(self, corpus: ObservationCorpus, manifest_digest: str):
        self.__corpus = corpus
        self.manifest_digest = manifest_digest
        self.commitment = corpus.digest()
        self.__opened_for_digest: str | None = None

    @property
    def opened(self) -> bool:
        """Return whether certification has irreversibly opened this handle."""

        return self.__opened_for_digest is not None

    def open(self, frozen_generic_policy_digest: str) -> ObservationCorpus:
        """Open once after the final generic policy has a stable digest."""

        digest = frozen_generic_policy_digest.strip()
        if not digest.startswith("sha256:"):
            raise ValueError("sealed partition requires a frozen sha256 policy digest")
        if self.__opened_for_digest is not None and self.__opened_for_digest != digest:
            raise ValueError("sealed partition was already opened for a different policy")
        self.__opened_for_digest = digest
        return self.__corpus
