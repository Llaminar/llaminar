"""Validated common NativeVNNI observation corpus and runtime key types."""

from __future__ import annotations

import hashlib
import heapq
import json
import multiprocessing
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Iterable, Iterator

from .schema import (
    AspectBucket,
    Backend,
    ExecutionMode,
    NativeVNNIObservation,
    SemanticContract,
)


# Large policy corpora contain hundreds of thousands of immutable Python
# objects.  Fork workers can read those objects through copy-on-write mappings
# without serializing every observation through a multiprocessing pipe.  The
# parent publishes this tuple only for the lifetime of one digest operation;
# worker tasks receive integer ranges and return sorted canonical JSON shards.
_PARALLEL_DIGEST_ROWS: tuple[NativeVNNIObservation, ...] = ()


def _serialize_canonical_digest_range(
    bounds: tuple[int, int],
) -> tuple[str, ...]:
    """Serialize and sort one disjoint observation range for corpus hashing."""

    begin, end = bounds
    return tuple(sorted(
        _PARALLEL_DIGEST_ROWS[index]._cached_canonical_json
        for index in range(begin, end)
    ))


def _digest_serialized_rows(
    serialized_shards: Iterable[Iterable[str]],
    *,
    distinguish_execution_mode: bool,
    distinguish_aspect_bucket: bool,
) -> str:
    """Hash sorted shards as the exact historical canonical JSON array.

    ``heapq.merge`` gives the same lexical ordering as sorting the complete
    list.  Streaming punctuation and row bytes avoids constructing a second
    corpus-sized string while preserving the old digest byte sequence exactly.
    """

    digest = hashlib.sha256()
    if not distinguish_aspect_bucket:
        digest.update(b"aspect-collapsed:")
    if not distinguish_execution_mode:
        digest.update(b"mode-collapsed:")
    digest.update(b"[")
    first = True
    for serialized_row in heapq.merge(*serialized_shards):
        if not first:
            digest.update(b",")
        digest.update(serialized_row.encode())
        first = False
    digest.update(b"]")
    return "sha256:" + digest.hexdigest()


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
    execution_mode: ExecutionMode
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
    execution_mode: ExecutionMode
    m: int
    aspect_bucket: AspectBucket
    all_aspects: bool = False


@dataclass(frozen=True, order=True)
class SurfaceKey:
    """Source alias observed within one mode-specific policy-v2 runtime key."""

    source_format: str
    execution_mode: ExecutionMode


@dataclass(frozen=True)
class CandidateIdentity:
    """Policy-selectable candidate metadata stable across runtime shapes.

    A shape-resolved candidate may map to a different concrete
    ``effective_candidate_id`` at each runtime key. Its nominal candidate ID,
    configuration, and arithmetic formula must remain invariant.
    """

    candidate_id: str
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
        execution_mode=observation.execution_mode,
        m=observation.m,
        aggregate_n=observation.aggregate_n,
        k=observation.k,
    )


def generic_domain(observation: NativeVNNIObservation) -> GenericDomain:
    """Project one observation onto the v2 mode/aspect/work policy domain."""

    return GenericDomain(
        backend=observation.backend,
        architecture_class=observation.architecture_class,
        semantic_contract=observation.semantic_contract,
        operation_kind=observation.operation_kind,
        bundle_signature=observation.bundle_signature,
        prepared_family_id=observation.prepared_family_id,
        packing_abi=observation.packing_abi,
        runtime_codebook_id=observation.runtime_codebook_id,
        execution_mode=observation.execution_mode,
        m=observation.m,
        aspect_bucket=observation.aspect_bucket,
    )


class ObservationCorpus:
    """Immutable validated corpus with explicit runtime-mode discrimination.

    Most generated ABIs, including CUDA, can select different policies for eager
    and graph-captured execution. ROCm's current NativeVNNI ABI intentionally
    cannot. Setting ``distinguish_execution_mode=False`` preserves the original
    mode on every observation and therefore in every timing surface, while
    canonicalizing only runtime keys and generic domains. The learner then picks
    one worst-mode-robust candidate rather than allowing table order to resolve
    duplicate mode-specific entries.
    """

    def __init__(
        self,
        observations: Iterable[NativeVNNIObservation],
        *,
        distinguish_execution_mode: bool = True,
        distinguish_aspect_bucket: bool = True,
    ):
        self._initialize(
            observations,
            distinguish_execution_mode=distinguish_execution_mode,
            distinguish_aspect_bucket=distinguish_aspect_bucket,
            validate_rows=True,
        )

    @classmethod
    def _from_validated(
        cls,
        observations: Iterable[NativeVNNIObservation],
        *,
        distinguish_execution_mode: bool = True,
        distinguish_aspect_bucket: bool = True,
    ) -> "ObservationCorpus":
        """Index immutable rows whose complete schema validation already ran.

        This private constructor is for ownership-preserving transformations:
        strict CSV parsing, filtering an existing corpus, and projection code
        that validates each newly created row before publication.  Candidate
        identity consistency and all runtime/domain indices are still rebuilt.
        Callers must never use it for untrusted or mutated row objects.
        """

        corpus = cls.__new__(cls)
        corpus._initialize(
            observations,
            distinguish_execution_mode=distinguish_execution_mode,
            distinguish_aspect_bucket=distinguish_aspect_bucket,
            validate_rows=False,
        )
        return corpus

    def _initialize(
        self,
        observations: Iterable[NativeVNNIObservation],
        *,
        distinguish_execution_mode: bool,
        distinguish_aspect_bucket: bool,
        validate_rows: bool,
    ) -> None:
        """Validate ownership as requested and build every canonical index."""

        rows = tuple(observations)
        if not rows:
            raise ValueError("NativeVNNI corpus must contain at least one observation")
        if validate_rows:
            for observation in rows:
                observation.validate()
        self._observations = rows
        self._distinguish_execution_mode = distinguish_execution_mode
        self._distinguish_aspect_bucket = distinguish_aspect_bucket
        self._validate_candidate_identities()
        rows_by_runtime_key: dict[
            RuntimeKey, list[NativeVNNIObservation]
        ] = defaultdict(list)
        rows_by_generic_domain: dict[
            GenericDomain, list[NativeVNNIObservation]
        ] = defaultdict(list)
        for row in rows:
            rows_by_runtime_key[self.runtime_key_for(row)].append(row)
            rows_by_generic_domain[self.generic_domain_for(row)].append(row)
        self._rows_by_runtime_key = {
            key: tuple(value) for key, value in rows_by_runtime_key.items()
        }
        self._rows_by_generic_domain = {
            key: tuple(value) for key, value in rows_by_generic_domain.items()
        }
        self._runtime_keys = tuple(sorted(self._rows_by_runtime_key))
        self._generic_domains = tuple(sorted(self._rows_by_generic_domain))
        # The corpus is immutable after construction.  Content hashing is
        # relatively expensive for trainer-scale evidence, so every later
        # provenance check on this object can reuse the first verified result.
        self._digest_cache: str | None = None

    @property
    def observations(self) -> tuple[NativeVNNIObservation, ...]:
        """Return the validated aggregate observations in deterministic order."""

        return self._observations

    def _validate_candidate_identities(self) -> None:
        identities: dict[tuple[Backend, str, str], CandidateIdentity] = {}
        for row in self._observations:
            key = (row.backend, row.architecture_class, row.candidate_id)
            identity = CandidateIdentity(
                candidate_id=row.candidate_id,
                candidate_family=row.candidate_family,
                config_json=json.dumps(row.config_json, sort_keys=True, separators=(",", ":")),
                arithmetic_fingerprint=row.arithmetic_fingerprint,
            )
            previous = identities.setdefault(key, identity)
            if previous != identity:
                raise ValueError(
                    "policy candidate identity changed across observations: "
                    f"{key}"
                )

    def rows_for_runtime_key(self, key: RuntimeKey) -> tuple[NativeVNNIObservation, ...]:
        """Return every alias, mode, and candidate observation for an exact key."""

        return self._rows_by_runtime_key.get(key, ())

    @property
    def distinguishes_execution_mode(self) -> bool:
        """Return whether execution mode participates in generated runtime keys."""

        return self._distinguish_execution_mode

    @property
    def distinguishes_aspect_bucket(self) -> bool:
        """Return whether coarse aspect participates in generic domains."""

        return self._distinguish_aspect_bucket

    def runtime_key_for(self, row: NativeVNNIObservation) -> RuntimeKey:
        """Project one row through this corpus's runtime visibility policy."""

        key = runtime_key(row)
        if self._distinguish_execution_mode:
            return key
        return replace(key, execution_mode=ExecutionMode.EAGER)

    def generic_domain_for(self, row: NativeVNNIObservation) -> GenericDomain:
        """Project one row through this corpus's generic-domain visibility."""

        domain = generic_domain(row)
        if not self._distinguish_execution_mode:
            domain = replace(domain, execution_mode=ExecutionMode.EAGER)
        if not self._distinguish_aspect_bucket:
            domain = replace(
                domain,
                aspect_bucket=AspectBucket.BALANCED,
                all_aspects=True,
            )
        return domain

    def with_collapsed_aspect_domains(self) -> "ObservationCorpus":
        """Reindex validated rows into cross-aspect generic policy domains.

        Exact runtime keys and observation aspect labels remain unchanged. Only
        the generic-domain index is collapsed, allowing aspect ratio to act as
        a learned tree feature instead of duplicating a hard outer partition.
        """

        return ObservationCorpus._from_validated(
            self._observations,
            distinguish_execution_mode=self._distinguish_execution_mode,
            distinguish_aspect_bucket=False,
        )

    def rows_for_generic_domain(self, domain: GenericDomain) -> tuple[NativeVNNIObservation, ...]:
        """Return every measured shape and candidate in one generic bucket."""

        return self._rows_by_generic_domain.get(domain, ())

    def runtime_keys(self) -> tuple[RuntimeKey, ...]:
        """Return sorted unique exact runtime keys represented by the corpus."""

        return self._runtime_keys

    def generic_domains(self) -> tuple[GenericDomain, ...]:
        """Return sorted unique v2 generic policy domains."""

        return self._generic_domains

    def subset(self, predicate) -> "ObservationCorpus":
        """Create a validated corpus containing rows accepted by ``predicate``."""

        return ObservationCorpus._from_validated(
            (row for row in self._observations if predicate(row)),
            distinguish_execution_mode=self._distinguish_execution_mode,
            distinguish_aspect_bucket=self._distinguish_aspect_bucket,
        )

    def shape_groups(self) -> tuple[str, ...]:
        """Return sorted logical shape groups used as split units."""

        return tuple(sorted({row.shape_group_id for row in self._observations}))

    def digest(self) -> str:
        """Hash canonical observations with deterministic parallel reduction.

        Serialization is independent per row, while the historical ABI sorts
        every serialized row before hashing.  Large corpora therefore split
        serialization and local sorting across fork workers, then merge those
        sorted shards in the parent.  The parent streams exactly the bytes the
        former monolithic ``json.dumps``/``join`` implementation produced, so
        existing manifests and fit-cache identities remain valid.

        Small unit-test corpora stay in-process because process startup would
        cost more than the work.  ``LLAMINAR_NATIVE_VNNI_CORPUS_DIGEST_WORKERS``
        can make benchmark experiments explicit; its value must be positive.
        """

        if self._digest_cache is not None:
            return self._digest_cache

        global _PARALLEL_DIGEST_ROWS
        row_count = len(self._observations)
        requested_workers = int(os.environ.get(
            "LLAMINAR_NATIVE_VNNI_CORPUS_DIGEST_WORKERS",
            str(os.cpu_count() or 1),
        ))
        if requested_workers < 1:
            raise ValueError("corpus digest worker count must be positive")

        # At least 4,096 rows per task keeps fork/IPC overhead small while a
        # production CPU corpus still fans out across both sockets.  One task
        # per worker bounds the returned shard inventory and keeps reduction
        # ordering independent of worker completion order.
        worker_count = min(requested_workers, max(1, row_count // 4096))
        if worker_count == 1:
            serialized_shards = (
                tuple(sorted(
                    row._cached_canonical_json
                    for row in self._observations
                )),
            )
        else:
            base = row_count // worker_count
            remainder = row_count % worker_count
            bounds = []
            begin = 0
            for worker_index in range(worker_count):
                size = base + (1 if worker_index < remainder else 0)
                bounds.append((begin, begin + size))
                begin += size
            _PARALLEL_DIGEST_ROWS = self._observations
            try:
                with ProcessPoolExecutor(
                    max_workers=worker_count,
                    mp_context=multiprocessing.get_context("fork"),
                ) as executor:
                    serialized_shards = tuple(executor.map(
                        _serialize_canonical_digest_range,
                        bounds,
                    ))
            finally:
                _PARALLEL_DIGEST_ROWS = ()

        self._digest_cache = _digest_serialized_rows(
            serialized_shards,
            distinguish_execution_mode=self._distinguish_execution_mode,
            distinguish_aspect_bucket=self._distinguish_aspect_bucket,
        )
        return self._digest_cache

    def __iter__(self) -> Iterator[NativeVNNIObservation]:
        return iter(self._observations)

    def __len__(self) -> int:
        return len(self._observations)


def observed_surfaces(rows: Iterable[NativeVNNIObservation]) -> frozenset[SurfaceKey]:
    """Return alias/mode surfaces present in a runtime-key row set."""

    return frozenset(
        SurfaceKey(row.source_format, row.execution_mode) for row in rows
    )
