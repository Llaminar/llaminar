"""Cross-cutting corpus inventory and policy contract validation helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .candidate_registry import CandidateRegistry
from .corpus import ObservationCorpus, RuntimeKey, SurfaceKey, runtime_key
from .format_registry import FORMAT_SPECS, runtime_aliases
from .schema import ExecutionMode, SemanticContract


# Fifteen speculative drafts plus the terminal target row is the default
# certification envelope. M=31 catches implementations that accidentally bake
# that graph default into a kernel or trainer admission check.
CANONICAL_VERIFIER_M = frozenset((*range(2, 17), 31))


def require_canonical_alias_coverage(
    corpus: ObservationCorpus,
    *,
    required_execution_modes: Iterable[ExecutionMode] | None = None,
) -> None:
    """Require every alias per mode and every requested policy-v2 mode key."""

    mode_override = tuple(required_execution_modes or ())
    failures = []
    for key in corpus.runtime_keys():
        rows = corpus.rows_for_runtime_key(key)
        expected_aliases = set(runtime_aliases(key.backend.value, key.runtime_codebook_id))
        expected_modes = {key.execution_mode}
        expected = {
            SurfaceKey(alias, mode)
            for alias in expected_aliases
            for mode in expected_modes
        }
        actual = {SurfaceKey(row.source_format, row.execution_mode) for row in rows}
        missing = sorted(expected - actual)
        if missing:
            failures.append((key, missing))
    if mode_override:
        represented_modes = defaultdict(set)
        for key in corpus.runtime_keys():
            identity = (
                key.backend,
                key.architecture_class,
                key.semantic_contract,
                key.operation_kind,
                key.bundle_signature,
                key.projection_n_vector,
                key.prepared_family_id,
                key.packing_abi,
                key.runtime_codebook_id,
                key.m,
                key.aggregate_n,
                key.k,
            )
            represented_modes[identity].add(key.execution_mode)
        required_modes = set(mode_override)
        for identity, modes in represented_modes.items():
            missing_modes = sorted(required_modes - modes, key=lambda mode: mode.value)
            if missing_modes:
                failures.append((identity, missing_modes))
    if failures:
        first_key, first_missing = failures[0]
        raise ValueError(
            f"canonical alias/mode coverage is incomplete for {len(failures)} key(s); "
            f"first={first_key} missing={first_missing}"
        )


def require_verifier_m_matrix(corpus: ObservationCorpus) -> None:
    """Require the complete runtime-M certification inventory per surface."""

    grouped = defaultdict(set)
    for row in corpus:
        if row.semantic_contract != SemanticContract.VERIFIER_SERIAL_M1_BITWISE:
            continue
        identity = (
            row.backend,
            row.architecture_class,
            row.operation_kind,
            row.bundle_signature,
            row.projection_n_vector,
            row.prepared_family_id,
            row.runtime_codebook_id,
            row.source_format,
            row.execution_mode,
            row.shape_group_id,
        )
        grouped[identity].add(row.m)
    missing = [
        (identity, sorted(CANONICAL_VERIFIER_M - values))
        for identity, values in grouped.items()
        if not CANONICAL_VERIFIER_M.issubset(values)
    ]
    if missing:
        raise ValueError(
            f"verifier M matrix incomplete for {len(missing)} surface(s); first={missing[0]}"
        )


def require_candidate_matrix_complete(corpus: ObservationCorpus) -> None:
    """Reject candidates silently absent on one alias/mode of an exact key."""

    failures = []
    for key in corpus.runtime_keys():
        rows = corpus.rows_for_runtime_key(key)
        surfaces = {SurfaceKey(row.source_format, row.execution_mode) for row in rows}
        candidates = {row.effective_candidate_id for row in rows}
        present = {
            (row.effective_candidate_id, SurfaceKey(row.source_format, row.execution_mode))
            for row in rows
        }
        for candidate in candidates:
            absent = sorted(
                surface for surface in surfaces if (candidate, surface) not in present
            )
            if absent:
                failures.append((key, candidate, absent))
    if failures:
        raise ValueError(
            f"candidate matrix has {len(failures)} incomplete key/candidate row(s); "
            f"first={failures[0]}"
        )


def require_registry_candidate_coverage(
    corpus: ObservationCorpus,
    registry: CandidateRegistry,
) -> None:
    """Require every contract-compatible registry candidate at every exact key.

    Matrix completeness alone cannot detect a candidate omitted from every
    alias.  This registry-backed check closes that gap.  Unsupported launches
    must still be represented by explicit ``supported=false`` observations so
    the corpus records why the candidate was unavailable.
    """

    failures = []
    for key in corpus.runtime_keys():
        if key.backend != registry.backend:
            continue
        required = {
            entry.effective_candidate_id
            for entry in registry.entries
            if entry.supports_contract(key.semantic_contract)
        }
        actual = {
            row.effective_candidate_id for row in corpus.rows_for_runtime_key(key)
        }
        missing = sorted(required - actual)
        unexpected = sorted(actual - required)
        if missing or unexpected:
            failures.append((key, missing, unexpected))
    if failures:
        key, missing, unexpected = failures[0]
        raise ValueError(
            f"registry candidate coverage is incomplete for {len(failures)} key(s); "
            f"first={key} missing={missing} unexpected={unexpected}"
        )
