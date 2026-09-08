"""Cross-cutting corpus inventory and policy contract validation helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable

from .candidate_registry import CandidateRegistry
from .corpus import ObservationCorpus, RuntimeKey, SurfaceKey, runtime_key
from .format_registry import FORMAT_SPECS, runtime_aliases
from .schema import ExecutionMode, SemanticContract
from .shape_manifest import NativeVNNIShapeManifest


# Fifteen speculative drafts plus the terminal target row is the default
# certification envelope. M=31 catches implementations that accidentally bake
# that graph default into a kernel or trainer admission check.
CANONICAL_VERIFIER_M = frozenset((*range(2, 17), 31))


def require_exact_overlay_scope(
    corpus: ObservationCorpus,
    manifest: NativeVNNIShapeManifest,
    *,
    shape_names: tuple[str, ...],
    m_values: tuple[int, ...],
    execution_modes: tuple[ExecutionMode, ...],
    contract: SemanticContract,
    active_rows: tuple[int, ...] | None = None,
) -> None:
    """Prove an explicitly declared additive refresh, including every format.

    A partial refresh must not infer its intended coverage from surviving CSV
    rows: a missing shape, depth, mode or entire codebook would then disappear
    silently. Resolve geometry through the shared production manifest and
    compare its Cartesian surface inventory with authenticated observations.
    Candidate reachability remains the backend registry's responsibility.
    Unmeasured keys outside this declaration retain their installed decisions;
    this gate is not a generic-policy or full-matrix certificate.
    """

    for name, values in (("shapes", shape_names), ("M", m_values),
                         ("modes", execution_modes)):
        if not values or len(set(values)) != len(values):
            raise ValueError(f"exact refresh {name} must be nonempty and unique")
    if any(m < (2 if contract == SemanticContract.VERIFIER_SERIAL_M1_BITWISE else 1)
           for m in m_values):
        raise ValueError("exact refresh M is outside its semantic contract")
    shapes = tuple(manifest.by_name(name) for name in shape_names)
    if any(not shape.exact_overlay for shape in shapes):
        raise ValueError("exact refresh only accepts production overlay shapes")
    expected = {
        (shape.name, shape.n, shape.k, m, spec.label, mode, active)
        for shape in shapes for m in m_values for spec in FORMAT_SPECS
        for mode in execution_modes
        for active in ((None,) if active_rows is None else active_rows)
    }
    if active_rows is not None and (
        not active_rows or len(set(active_rows)) != len(active_rows)
        or any(type(active) is not int or not 1 <= active <= min(m_values)
               for active in active_rows)
    ):
        raise ValueError("exact refresh active rows must be unique positive prefixes within every M")
    actual = {
        (row.shape_name, row.aggregate_n, row.k, row.m, row.source_format,
         row.execution_mode, row.active_rows)
        for row in corpus
    }
    if actual != expected:
        raise ValueError(
            "exact refresh surface inventory is incomplete or out of scope: "
            f"missing={sorted(expected - actual, key=repr)[:8]} "
            f"unexpected={sorted(actual - expected, key=repr)[:8]}"
        )
    if {row.semantic_contract for row in corpus} != {contract}:
        raise ValueError("exact refresh mixes arithmetic contracts")
    if len({(row.backend, row.architecture_class) for row in corpus}) != 1:
        raise ValueError("exact refresh must target one backend architecture")
    require_canonical_alias_coverage(corpus)
    require_candidate_matrix_complete(corpus)


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
            SurfaceKey(alias, mode, occupancy)
            for alias in expected_aliases
            for mode in expected_modes
            for occupancy in {row.active_rows or 0 for row in rows}
        }
        actual = {SurfaceKey.from_observation(row) for row in rows}
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
        surfaces = {SurfaceKey.from_observation(row) for row in rows}
        candidates = {row.effective_candidate_id for row in rows}
        present = {
            (row.effective_candidate_id, SurfaceKey.from_observation(row))
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
    *,
    candidate_ids_for_runtime_key: (
        Callable[[RuntimeKey], Iterable[str]] | None
    ) = None,
) -> None:
    """Require every forceable registry candidate at every exact runtime key.

    Matrix completeness alone cannot detect a candidate omitted from every
    alias.  This registry-backed check closes that gap.  Unsupported launches
    may be represented by explicit ``supported=false`` observations when that
    negative evidence is useful. A surface whose nominal registry entries
    collapse to fewer physical schedules must supply
    ``candidate_ids_for_runtime_key``; aliases of one launch are then required
    exactly once under their canonical physical candidate ID.
    """

    failures = []
    for key in corpus.runtime_keys():
        if key.backend != registry.backend:
            continue
        registered = {
            entry.effective_candidate_id
            for entry in registry.entries
            if entry.supports_contract(key.semantic_contract)
        }
        required = (
            registered
            if candidate_ids_for_runtime_key is None
            else set(candidate_ids_for_runtime_key(key))
        )
        foreign = sorted(required - registered)
        if foreign:
            raise ValueError(
                "runtime candidate resolver returned IDs outside the "
                f"contract-compatible registry: {foreign}"
            )
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
