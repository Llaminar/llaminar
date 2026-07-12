"""Validated common NativeVNNI observation corpus and runtime key types."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Iterator

from .schema import (
    AspectBucket,
    Backend,
    ExecutionMode,
    NativeVNNIObservation,
    SemanticContract,
)


@dataclass(frozen=True, order=True)
class RuntimeKey:
    """Every discriminator available to an exact production resolver."""

    backend: Backend
    architecture_class: str
    semantic_contract: SemanticContract
    operation_kind: str
    bundle_signature: str
    projection_n_vector: tuple[int, ...]
    prepared_family_id: str
    packing_abi: str
    runtime_codebook_id: int
    m: int
    aggregate_n: int
    k: int


@dataclass(frozen=True, order=True)
class GenericDomain:
    """Dimensions shared by one aspect/work segmented generic policy."""

    backend: Backend
    architecture_class: str
    semantic_contract: SemanticContract
    operation_kind: str
    bundle_signature: str
    prepared_family_id: str
    packing_abi: str
    runtime_codebook_id: int
    m: int
    aspect_bucket: AspectBucket


@dataclass(frozen=True, order=True)
class SurfaceKey:
    """Alias and execution-mode surface invisible to policy ABI v1."""

    source_format: str
    execution_mode: ExecutionMode


@dataclass(frozen=True)
class CandidateIdentity:
    """Normalized candidate metadata that must be stable across observations."""

    effective_candidate_id: str
    candidate_family: str
    config_json: str
    arithmetic_fingerprint: str


def runtime_key(observation: NativeVNNIObservation) -> RuntimeKey:
    """Project one observation onto the exact runtime dispatch surface."""

    return RuntimeKey(
        backend=observation.backend,
        architecture_class=observation.architecture_class,
        semantic_contract=observation.semantic_contract,
        operation_kind=observation.operation_kind,
        bundle_signature=observation.bundle_signature,
        projection_n_vector=observation.projection_n_vector,
        prepared_family_id=observation.prepared_family_id,
        packing_abi=observation.packing_abi,
        runtime_codebook_id=observation.runtime_codebook_id,
        m=observation.m,
        aggregate_n=observation.aggregate_n,
        k=observation.k,
    )


def generic_domain(observation: NativeVNNIObservation) -> GenericDomain:
    """Project one observation onto the v1 generic aspect/work domain."""

    return GenericDomain(
        backend=observation.backend,
        architecture_class=observation.architecture_class,
        semantic_contract=observation.semantic_contract,
        operation_kind=observation.operation_kind,
        bundle_signature=observation.bundle_signature,
        prepared_family_id=observation.prepared_family_id,
        packing_abi=observation.packing_abi,
        runtime_codebook_id=observation.runtime_codebook_id,
        m=observation.m,
        aspect_bucket=observation.aspect_bucket,
    )


class ObservationCorpus:
    """Immutable validated corpus with normalized candidate consistency checks."""

    def __init__(self, observations: Iterable[NativeVNNIObservation]):
        rows = tuple(observations)
        if not rows:
            raise ValueError("NativeVNNI corpus must contain at least one observation")
        for observation in rows:
            observation.validate()
        self._observations = rows
        self._validate_candidate_identities()

    @property
    def observations(self) -> tuple[NativeVNNIObservation, ...]:
        """Return the validated aggregate observations in deterministic order."""

        return self._observations

    def _validate_candidate_identities(self) -> None:
        identities: dict[tuple[Backend, str, str], CandidateIdentity] = {}
        for row in self._observations:
            key = (row.backend, row.architecture_class, row.effective_candidate_id)
            identity = CandidateIdentity(
                effective_candidate_id=row.effective_candidate_id,
                candidate_family=row.candidate_family,
                config_json=json.dumps(row.config_json, sort_keys=True, separators=(",", ":")),
                arithmetic_fingerprint=row.arithmetic_fingerprint,
            )
            previous = identities.setdefault(key, identity)
            if previous != identity:
                raise ValueError(
                    "effective candidate identity changed across observations: "
                    f"{key}"
                )

    def rows_for_runtime_key(self, key: RuntimeKey) -> tuple[NativeVNNIObservation, ...]:
        """Return every alias, mode, and candidate observation for an exact key."""

        return tuple(row for row in self._observations if runtime_key(row) == key)

    def rows_for_generic_domain(self, domain: GenericDomain) -> tuple[NativeVNNIObservation, ...]:
        """Return every measured shape and candidate in one generic bucket."""

        return tuple(row for row in self._observations if generic_domain(row) == domain)

    def runtime_keys(self) -> tuple[RuntimeKey, ...]:
        """Return sorted unique exact runtime keys represented by the corpus."""

        return tuple(sorted({runtime_key(row) for row in self._observations}))

    def generic_domains(self) -> tuple[GenericDomain, ...]:
        """Return sorted unique v1 generic policy domains."""

        return tuple(sorted({generic_domain(row) for row in self._observations}))

    def subset(self, predicate) -> "ObservationCorpus":
        """Create a validated corpus containing rows accepted by ``predicate``."""

        return ObservationCorpus(row for row in self._observations if predicate(row))

    def shape_groups(self) -> tuple[str, ...]:
        """Return sorted logical shape groups used as split units."""

        return tuple(sorted({row.shape_group_id for row in self._observations}))

    def digest(self) -> str:
        """Hash all canonical observations independent of input file ordering."""

        serialized_rows = sorted(
            json.dumps(
                row.canonical_mapping(), sort_keys=True, separators=(",", ":")
            )
            for row in self._observations
        )
        encoded = ("[" + ",".join(serialized_rows) + "]").encode()
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def __iter__(self) -> Iterator[NativeVNNIObservation]:
        return iter(self._observations)

    def __len__(self) -> int:
        return len(self._observations)


def observed_surfaces(rows: Iterable[NativeVNNIObservation]) -> frozenset[SurfaceKey]:
    """Return alias/mode surfaces present in a runtime-key row set."""

    return frozenset(
        SurfaceKey(row.source_format, row.execution_mode) for row in rows
    )
