#!/usr/bin/env python3
"""Compile strong CUDA NativeVNNI decode evidence through the common policy core.

The CUDA trainer measures explicit public-M1 schedules and the grouped verifier's
``INHERIT_SERIAL_M1`` implementation on eager and graph-captured production
surfaces.  This analyzer adapts those rows into the backend-neutral evidence
schema, chooses mode-specific alias-robust exact winners, learns bounded aspect/work
rules, and emits the production CUDA selector ABI.

Only Fast M=1 decisions are emitted.  Grouped verifier M=2..16 and M=31 rows
certify that the grouped kernel inherits the complete frozen M=1 arithmetic
identity, including its exact K-partition count; they never publish an
independently tuned verifier schedule.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path


KERNEL_PERF_ROOT = Path(__file__).resolve().parents[2]
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.adapters.cuda_decode import (  # noqa: E402
    CUDADecodeAdapterContext,
    adapt_cuda_decode_csv,
    raw_corpus_id,
)
from native_vnni_dispatch.candidate_observation import (  # noqa: E402
    read_observation_csv,
    write_observation_csv,
)
from native_vnni_dispatch.candidate_registry import (  # noqa: E402
    candidate_registry_digest,
    cuda_native_vnni_gemv_registry,
)
from native_vnni_dispatch.certification import CertificationReport  # noqa: E402
from native_vnni_dispatch.compiler import (  # noqa: E402
    CompiledPolicy,
    FrozenPolicy,
    certify_frozen_policy,
    freeze_policy,
)
from native_vnni_dispatch.corpus import (  # noqa: E402
    ObservationCorpus,
    RuntimeKey,
    runtime_key,
)
from native_vnni_dispatch.cpp_predicates import (  # noqa: E402
    aspect_condition,
    generic_rule_sort_key,
    predicate_condition,
    render_if_header,
)
from native_vnni_dispatch.cuda_shape_resolved import (  # noqa: E402
    project_cuda_shape_resolved_candidates,
)
from native_vnni_dispatch.exact_oracle import (  # noqa: E402
    ExactWinner,
    build_exact_winners,
)
from native_vnni_dispatch.format_registry import FORMAT_SPECS  # noqa: E402
from native_vnni_dispatch.measurement_plan import (  # noqa: E402
    MEASUREMENT_PLAN_PATH,
    NativeVNNIGPUMeasurementPlan,
    load_gpu_measurement_plan,
)
from native_vnni_dispatch.profiles import MeasurementProfile  # noqa: E402
from native_vnni_dispatch.profiler_model import (  # noqa: E402
    ProfilerFeatureCatalog,
    load_profiler_feature_catalog,
)
from native_vnni_dispatch.paired_confirmation import (  # noqa: E402
    PairedCellKey,
    PairedTimingComparison,
    paired_timing_comparisons,
    read_paired_confirmation_csv,
)
from native_vnni_dispatch.paired_requests import (  # noqa: E402
    paired_comparison_digest,
)
from native_vnni_dispatch.policy_ir import PolicyIR  # noqa: E402
from native_vnni_dispatch.policy_artifact import (  # noqa: E402
    validate_installable_policy_artifact,
    validate_frozen_policy_file,
    write_compiled_policy,
    write_frozen_policy,
)
from native_vnni_dispatch.schema import (  # noqa: E402
    AspectBucket,
    Backend,
    ExecutionMode,
    SemanticContract,
)
from native_vnni_dispatch.segmented_policy import (  # noqa: E402
    GenericDispatchRule,
    fit_generic_policy,
)
from native_vnni_dispatch.shape_manifest import (  # noqa: E402
    MANIFEST_PATH,
    NativeVNNIShapeManifest,
    ShapePartition,
    load_shape_manifest,
    partition_assignments,
)
from native_vnni_dispatch.validation import (  # noqa: E402
    CANONICAL_VERIFIER_M,
    require_candidate_matrix_complete,
    require_canonical_alias_coverage,
    require_verifier_m_matrix,
)


@dataclass(frozen=True, order=True)
class FastEntry:
    """One alias-robust, execution-mode-specific public-M1 decision."""

    codebook: int
    execution_mode: ExecutionMode
    n: int
    k: int
    family: str
    tile_n: int
    cpt: int
    target_waves: int
    min_kgroups_per_cta: int
    max_kb: int
    force_two_phase: int
    exact_kb: int
    candidate_id: str
    shape_name: str
    max_surface_regret: float
    max_cv: float


def pack_shape_key(
    execution_mode: ExecutionMode,
    m: int,
    n: int,
    k: int,
) -> int:
    """Pack the mode-aware CUDA policy-v2 exact key without lossy hashing."""

    captured = int(execution_mode == ExecutionMode.GRAPH_CAPTURED)
    return (
        (captured << 63)
        | ((m & 0x7F) << 56)
        | ((k & 0xFFFFFF) << 28)
        | (n & 0x0FFFFFFF)
    )


def _serial_hashes(
    corpus: ObservationCorpus,
    serial_m1_policy_hash: str,
) -> dict[RuntimeKey, str]:
    """Bind each represented verifier key to the frozen public-M1 artifact."""

    return {
        key: serial_m1_policy_hash
        for key in corpus.runtime_keys()
        if key.semantic_contract == SemanticContract.VERIFIER_SERIAL_M1_BITWISE
    }


def _fast_config(candidate_id: str) -> dict[str, object]:
    """Resolve and validate one forceable public-M1 candidate configuration."""

    candidate = cuda_native_vnni_gemv_registry().resolve(candidate_id)
    if not candidate.supports_contract(SemanticContract.FAST):
        raise ValueError(f"{candidate_id} is not a Fast CUDA decode candidate")
    config = candidate.config_json
    family = str(config["family"])
    if family not in {"wide", "direct", "kpar", "kpar_formula"}:
        raise ValueError(f"unsupported CUDA generated family {family!r}")
    if family == "kpar" and int(config["exact_kb"]) <= 0:
        raise ValueError("CUDA KPAR policy must publish an exact positive KB")
    return config


def select_fast_entries(
    corpus: ObservationCorpus,
    serial_m1_policy_hash: str,
) -> tuple[list[FastEntry], dict[RuntimeKey, ExactWinner]]:
    """Run the common exact oracle and encode only public-M1 decisions."""

    exact = build_exact_winners(
        corpus,
        serial_m1_hashes=_serial_hashes(corpus, serial_m1_policy_hash),
    )
    entries = []
    for key, winner in sorted(exact.items()):
        if key.semantic_contract == SemanticContract.VERIFIER_SERIAL_M1_BITWISE:
            if winner.candidate_id != (
                "cuda.nvnni.decode.verifier.inherit_serial_m1"
            ):
                raise ValueError(
                    "grouped CUDA verifier winner does not inherit serial M1"
                )
            continue
        if key.semantic_contract != SemanticContract.FAST or key.m != 1:
            raise ValueError(f"unsupported CUDA decode exact policy key {key}")
        if len(key.projection_n_vector) != 1:
            raise ValueError("CUDA decode exact entry is not one projection")
        rows = corpus.rows_for_runtime_key(key)
        shape_names = sorted({row.shape_name for row in rows})
        if len(shape_names) != 1:
            raise ValueError(
                f"multiple shape names collapse onto CUDA exact key {key}: "
                f"{shape_names}"
            )
        config = _fast_config(winner.candidate_id)
        entries.append(FastEntry(
            codebook=key.runtime_codebook_id,
            execution_mode=key.execution_mode,
            n=key.aggregate_n,
            k=key.k,
            family=str(config["family"]),
            tile_n=int(config["tile_n"]),
            cpt=int(config["cpt"]),
            target_waves=int(config["target_waves"]),
            min_kgroups_per_cta=int(config["min_kgroups_per_cta"]),
            max_kb=int(config["max_kb"]),
            force_two_phase=int(config["force_two_phase"]),
            exact_kb=int(config["exact_kb"]),
            candidate_id=winner.candidate_id,
            shape_name=shape_names[0],
            max_surface_regret=winner.max_surface_regret,
            max_cv=winner.max_cv,
        ))
    return entries, exact


def select_fast_generic_rules(
    corpus: ObservationCorpus,
    serial_m1_policy_hash: str,
) -> list[GenericDispatchRule]:
    """Fit the shared bounded-regret learner and retain public-M1 domains."""

    generic = fit_generic_policy(
        corpus,
        serial_m1_hashes=_serial_hashes(corpus, serial_m1_policy_hash),
    )
    return [
        rule
        for rule in generic.rules
        if rule.domain.semantic_contract == SemanticContract.FAST
        and rule.domain.m == 1
    ]


def _fast_m1_corpus(corpus: ObservationCorpus) -> ObservationCorpus:
    """Project a combined transaction onto the independent Fast-M1 phase."""

    rows = tuple(
        row
        for row in corpus
        if row.semantic_contract == SemanticContract.FAST and row.m == 1
    )
    if not rows:
        raise ValueError("CUDA policy transaction contains no Fast M=1 evidence")
    return ObservationCorpus(rows)


def _require_manifest_surface_complete(
    corpus: ObservationCorpus,
    manifest: NativeVNNIShapeManifest,
    *,
    required_surfaces: frozenset[tuple[str, str, ExecutionMode]],
) -> None:
    """Require every planned shape/format/mode with declared dimensions."""

    observed_names = {row.shape_name for row in corpus}
    expected_names = {surface[0] for surface in required_surfaces}
    missing = sorted(expected_names - observed_names)
    unexpected = sorted(observed_names - expected_names)
    if missing or unexpected:
        raise ValueError(
            "CUDA M1 shape-manifest coverage is incomplete: "
            f"missing={missing} unexpected={unexpected}"
        )
    for row in corpus:
        shape = manifest.by_name(row.shape_name)
        if row.aggregate_n != shape.n or row.k != shape.k:
            raise ValueError(
                f"{row.shape_name}: corpus dimensions "
                f"{row.aggregate_n}x{row.k} disagree with manifest "
                f"{shape.n}x{shape.k}"
            )

    observed_surfaces = {
        (row.shape_name, row.source_format, row.execution_mode)
        for row in corpus
    }
    missing_surfaces = sorted(
        required_surfaces - observed_surfaces,
        key=lambda item: (item[0], item[1], item[2].value),
    )
    unexpected_surfaces = sorted(
        observed_surfaces - required_surfaces,
        key=lambda item: (item[0], item[1], item[2].value),
    )
    if missing_surfaces or unexpected_surfaces:
        raise ValueError(
            "CUDA M1 shape/format/mode Cartesian surface is incomplete: "
            f"missing_count={len(missing_surfaces)} "
            f"first_missing={missing_surfaces[:1]} "
            f"unexpected_count={len(unexpected_surfaces)} "
            f"first_unexpected={unexpected_surfaces[:1]}"
        )


def _fast_partition_surfaces(
    manifest: NativeVNNIShapeManifest,
    measurement_plan: NativeVNNIGPUMeasurementPlan,
    partition: ShapePartition,
) -> frozenset[tuple[str, str, ExecutionMode]]:
    """Return the exact reviewed CUDA Fast-M1 physical surfaces."""

    surfaces = {
        (shape_name, spec.label, mode)
        for shape_name in (
            measurement_plan.common_development_shapes
            if partition == ShapePartition.DEVELOPMENT
            else manifest.partition_names(
                verifier=False,
                partition=ShapePartition.SEALED,
            )
        )
        for spec in FORMAT_SPECS
        for mode in (ExecutionMode.EAGER, ExecutionMode.GRAPH_CAPTURED)
    }
    if partition == ShapePartition.DEVELOPMENT:
        for extension in measurement_plan.fast_development_extensions:
            if extension.backend != Backend.CUDA:
                continue
            surfaces.update(
                (shape_name, source_format, mode)
                for shape_name in extension.shape_names
                for source_format in extension.source_formats
                for mode in (
                    ExecutionMode.EAGER,
                    ExecutionMode.GRAPH_CAPTURED,
                )
            )
    return frozenset(surfaces)


def _fast_sealed_commitment(
    manifest: NativeVNNIShapeManifest,
    measurement_plan: NativeVNNIGPUMeasurementPlan,
) -> str:
    """Commit to the untouched Fast shape inventory before measurement."""

    payload = {
        "protocol": "cuda-native-vnni-fast-m1-sealed-v1",
        "manifest_digest": manifest.digest(),
        "measurement_plan_digest": measurement_plan.digest(manifest),
        "shape_names": list(manifest.partition_names(
            verifier=False,
            partition=ShapePartition.SEALED,
        )),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _require_fast_partition(
    corpus: ObservationCorpus,
    manifest: NativeVNNIShapeManifest,
    measurement_plan: NativeVNNIGPUMeasurementPlan,
    partition: ShapePartition,
) -> ObservationCorpus:
    """Validate one physically separate Fast-M1 measurement partition."""

    direct = _fast_m1_corpus(corpus)
    required_surfaces = _fast_partition_surfaces(
        manifest,
        measurement_plan,
        partition,
    )
    _require_manifest_surface_complete(
        direct,
        manifest,
        required_surfaces=required_surfaces,
    )
    assignments = partition_assignments(
        ((row.shape_group_id, row.shape_name) for row in direct),
        verifier=False,
        manifest=manifest,
    )
    if set(assignments.values()) != {partition}:
        raise ValueError(
            f"CUDA M1 {partition.value} input crosses a manifest partition"
        )
    return direct


def freeze_fast_policy(
    development_corpus: ObservationCorpus,
    manifest: NativeVNNIShapeManifest,
    measurement_plan: NativeVNNIGPUMeasurementPlan,
    *,
    paired_development_comparisons: dict[
        PairedCellKey, tuple[PairedTimingComparison, ...]
    ] | None = None,
    profiler_feature_catalog: ProfilerFeatureCatalog | None = None,
) -> FrozenPolicy:
    """Fit CUDA Fast M1 from development rows without accepting sealed data."""

    direct_development = _require_fast_partition(
        development_corpus,
        manifest,
        measurement_plan,
        ShapePartition.DEVELOPMENT,
    )
    development = project_cuda_shape_resolved_candidates(
        direct_development
    )
    frozen = freeze_policy(
        development,
        sealed_commitment=_fast_sealed_commitment(manifest, measurement_plan),
        split_manifest_digest=measurement_plan.digest(manifest),
        paired_development_comparisons=paired_development_comparisons,
        profiler_feature_catalog=profiler_feature_catalog,
        metadata={
            "backend": "cuda",
            "semantic_contract": SemanticContract.FAST.value,
            "shape_manifest_schema": manifest.schema_version,
            "shape_manifest_digest": manifest.digest(),
            "measurement_plan_schema": measurement_plan.schema_version,
            "measurement_plan_digest": measurement_plan.digest(manifest),
            "paired_development_evidence_digest": paired_comparison_digest(
                paired_development_comparisons or {}
            ),
        },
    )
    exact_overlay_names = {
        shape.name for shape in manifest.shapes if shape.exact_overlay
    }
    exact_overlay_keys = {
        runtime_key(row)
        for row in direct_development
        if row.shape_name in exact_overlay_names
    }
    filtered = replace(
        frozen.policy_ir,
        exact_entries=tuple(
            entry
            for entry in frozen.policy_ir.exact_entries
            if entry.key in exact_overlay_keys
        ),
    )
    if filtered.digest(generic_only=True) != frozen.generic_digest:
        raise ValueError("filtering exact overlays changed frozen CUDA policy")
    return FrozenPolicy(
        filtered,
        promotion_diagnostics=frozen.promotion_diagnostics,
    )


def certify_fast_policy(
    frozen: FrozenPolicy,
    development_corpus: ObservationCorpus,
    sealed_corpus: ObservationCorpus,
    manifest: NativeVNNIShapeManifest,
    measurement_plan: NativeVNNIGPUMeasurementPlan,
) -> CompiledPolicy:
    """Open Fast sealed rows and certify an already-frozen CUDA policy."""

    direct_development = _require_fast_partition(
        development_corpus,
        manifest,
        measurement_plan,
        ShapePartition.DEVELOPMENT,
    )
    direct_sealed = _require_fast_partition(
        sealed_corpus,
        manifest,
        measurement_plan,
        ShapePartition.SEALED,
    )
    development = project_cuda_shape_resolved_candidates(
        direct_development
    )
    sealed = project_cuda_shape_resolved_candidates(direct_sealed)
    compiled = certify_frozen_policy(
        frozen,
        development,
        sealed,
    )
    exact_overlay_names = {
        shape.name for shape in manifest.shapes if shape.exact_overlay
    }
    exact_overlay_keys = {
        runtime_key(row)
        for row in (*direct_development.observations, *direct_sealed.observations)
        if row.shape_name in exact_overlay_names
    }
    filtered = replace(
        compiled.policy_ir,
        exact_entries=tuple(
            entry
            for entry in compiled.policy_ir.exact_entries
            if entry.key in exact_overlay_keys
        ),
    )
    if filtered.digest(generic_only=True) != frozen.generic_digest:
        raise ValueError("sealed exact filtering changed frozen CUDA policy")
    return CompiledPolicy(filtered, compiled.certification)


def validate_policy_ir_inputs(
    policy_ir: PolicyIR,
    entries: list[FastEntry],
    generic_rules: list[GenericDispatchRule],
) -> None:
    """Prove the backend emitter receives the exact accepted common IR."""

    ir_exact = {
        (
            entry.key.runtime_codebook_id,
            entry.key.execution_mode,
            entry.key.aggregate_n,
            entry.key.k,
        ): entry.candidate_id
        for entry in policy_ir.exact_entries
    }
    emitted_exact = {
        (
            entry.codebook,
            entry.execution_mode,
            entry.n,
            entry.k,
        ): entry.candidate_id
        for entry in entries
    }
    if emitted_exact != ir_exact:
        missing = sorted(set(ir_exact) - set(emitted_exact))
        unexpected = sorted(set(emitted_exact) - set(ir_exact))
        differing = sorted(
            key
            for key in set(ir_exact) & set(emitted_exact)
            if ir_exact[key] != emitted_exact[key]
        )
        raise ValueError(
            "CUDA exact emitter inputs disagree with common IR: "
            f"missing={missing[:1]} unexpected={unexpected[:1]} "
            f"differing={differing[:1]}"
        )
    if tuple(generic_rules) != policy_ir.generic_rules:
        raise ValueError("CUDA generic emitter inputs disagree with common IR")


def validate_emitter_inputs(
    compiled: CompiledPolicy,
    entries: list[FastEntry],
    generic_rules: list[GenericDispatchRule],
) -> None:
    """Validate a sealed-certified policy before production emission."""

    validate_policy_ir_inputs(compiled.policy_ir, entries, generic_rules)


def validate_complete(
    corpus: ObservationCorpus,
    manifest: NativeVNNIShapeManifest,
    measurement_plan: NativeVNNIGPUMeasurementPlan,
    *,
    require_full_inventory: bool,
) -> None:
    """Require every applicable alias, mode, candidate, shape, and depth."""

    require_canonical_alias_coverage(
        corpus,
        required_execution_modes=(
            ExecutionMode.EAGER,
            ExecutionMode.GRAPH_CAPTURED,
        ),
    )
    require_verifier_m_matrix(corpus)
    require_candidate_matrix_complete(corpus)
    _require_shape_reachable_candidate_coverage(corpus)

    architecture_classes = {row.architecture_class for row in corpus}
    if len(architecture_classes) != 1:
        raise ValueError(
            "one CUDA generated artifact must describe exactly one architecture "
            f"class, got {sorted(architecture_classes)}"
        )

    by_shape: dict[tuple, set[tuple[SemanticContract, int]]] = {}
    names_by_shape: dict[tuple, str] = {}
    for row in corpus:
        identity = (
            row.runtime_codebook_id,
            row.source_format,
            row.execution_mode,
            row.shape_group_id,
        )
        previous_name = names_by_shape.setdefault(identity, row.shape_name)
        if previous_name != row.shape_name:
            raise ValueError(
                f"CUDA shape group maps to multiple manifest names: {identity}"
            )
        by_shape.setdefault(identity, set()).add((row.semantic_contract, row.m))

    failures = []
    for identity, represented in by_shape.items():
        shape_name = names_by_shape[identity]
        manifest.by_name(shape_name)
        source_format = identity[1]
        required: set[tuple[SemanticContract, int]] = set()
        if measurement_plan.fast_shape_applies(
            manifest,
            backend=Backend.CUDA,
            source_format=source_format,
            shape_name=shape_name,
        ):
            required.add((SemanticContract.FAST, 1))
        if measurement_plan.verifier_shape_applies(
            manifest,
            shape_name=shape_name,
        ):
            required.update(
                (SemanticContract.VERIFIER_SERIAL_M1_BITWISE, m)
                for m in CANONICAL_VERIFIER_M
            )
        missing = required - represented
        unexpected = represented - required
        if missing or unexpected:
            order_key = lambda item: (item[0].value, item[1])
            failures.append((
                identity,
                sorted(missing, key=order_key),
                sorted(unexpected, key=order_key),
            ))
    if failures:
        raise ValueError(
            "CUDA decode shape applicability/depth matrix is invalid for "
            f"{len(failures)} surface(s); first={failures[0]}"
        )

    if require_full_inventory:
        expected = measurement_plan.expected_decode_surfaces(
            manifest,
            backend=Backend.CUDA,
            source_formats=tuple(spec.label for spec in FORMAT_SPECS),
            execution_modes=(
                ExecutionMode.EAGER,
                ExecutionMode.GRAPH_CAPTURED,
            ),
            verifier_m_values=CANONICAL_VERIFIER_M,
        )
        actual = {
            (
                row.semantic_contract,
                row.m,
                row.shape_name,
                row.source_format,
                row.execution_mode,
            )
            for row in corpus
        }
        if actual != expected:
            order_key = lambda item: (
                item[2], item[3], item[4].value, item[0].value, item[1]
            )
            missing = sorted(expected - actual, key=order_key)
            unexpected = sorted(actual - expected, key=order_key)
            raise ValueError(
                "CUDA production physical surface inventory is incomplete: "
                f"missing_count={len(missing)} first_missing={missing[:1]} "
                f"unexpected_count={len(unexpected)} "
                f"first_unexpected={unexpected[:1]}"
            )


def validate_fast_m1_complete(corpus: ObservationCorpus) -> None:
    """Require a complete standalone public-M1 candidate matrix.

    M1 training is the first half of the ordered verifier transaction, so it
    cannot require dependent M=2+ evidence. It must still prove every canonical
    source alias, eager/captured surface, forceable candidate, shape, and exact
    K-partition before the generated artifact may be staged.
    """

    invalid = [
        key
        for key in corpus.runtime_keys()
        if key.semantic_contract != SemanticContract.FAST or key.m != 1
    ]
    if invalid:
        raise ValueError(
            "standalone CUDA M1 staging corpus contains non-M1 key "
            f"{invalid[0]}"
        )
    require_canonical_alias_coverage(
        corpus,
        required_execution_modes=(
            ExecutionMode.EAGER,
            ExecutionMode.GRAPH_CAPTURED,
        ),
    )
    require_candidate_matrix_complete(corpus)
    _require_shape_reachable_candidate_coverage(corpus)
    architecture_classes = {row.architecture_class for row in corpus}
    if len(architecture_classes) != 1:
        raise ValueError(
            "one CUDA M1 artifact must describe exactly one architecture "
            f"class, got {sorted(architecture_classes)}"
        )


def _require_shape_reachable_candidate_coverage(
    corpus: ObservationCorpus,
) -> None:
    """Require every candidate whose exact partials fit the shape workspace.

    CUDA allocates one KPAR partial slot per 32-value K group. Exact KB values
    are reduced to one representative per distinct blocks-per-partition width:
    within such a class, every larger KB performs the same useful arithmetic
    plus extra empty CTAs and zero reductions, so it is strictly dominated.
    Exact non-divisors remain represented and publish explicit zero tails.
    """

    registry = cuda_native_vnni_gemv_registry()
    failures = []
    for key in corpus.runtime_keys():
        required = set()
        for candidate in registry.entries:
            if not candidate.supports_contract(key.semantic_contract):
                continue
            if candidate.config_json.get("family") == "kpar_formula":
                # Formula costs are projected from this directly measured
                # exact matrix after completeness validation.
                continue
            exact_kb = int(candidate.config_json.get("exact_kb", 0))
            if exact_kb and exact_kb not in _economical_exact_kblocks(key.k // 32):
                continue
            required.add(candidate.effective_candidate_id)
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
            "CUDA shape-reachable candidate coverage is incomplete for "
            f"{len(failures)} key(s); first={key} missing={missing} "
            f"unexpected={unexpected}"
        )


def _economical_exact_kblocks(k_groups: int) -> frozenset[int]:
    """Return one smallest KB for each distinct partition-width geometry."""

    represented_widths = set()
    result = set()
    for kb in range(1, min(k_groups, 256) + 1):
        blocks_per_partition = (k_groups + kb - 1) // kb
        if blocks_per_partition in represented_widths:
            continue
        represented_widths.add(blocks_per_partition)
        result.add(kb)
    return frozenset(result)

def _shape_enum(family: str) -> str:
    """Map one common candidate family into the CUDA runtime enum."""

    return {
        "wide": "NativeGemvShape::WIDE",
        "direct": "NativeGemvShape::DIRECT",
        "kpar": "NativeGemvShape::KPAR",
        "kpar_formula": "NativeGemvShape::KPAR",
    }[family]


def _tuning_literal(config: dict[str, object]) -> str:
    """Render the CUDA ABI, keeping exact KB after legacy aggregate fields."""

    return (
        "{" + ", ".join(str(int(config[field])) for field in (
            "tile_n",
            "cpt",
            "target_waves",
            "min_kgroups_per_cta",
            "max_kb",
            "force_two_phase",
            "exact_kb",
        )) + "}"
    )


def _entry_line(entry: FastEntry) -> str:
    """Render one exact public-M1 table row with trainer provenance."""

    config = {
        "tile_n": entry.tile_n,
        "cpt": entry.cpt,
        "target_waves": entry.target_waves,
        "min_kgroups_per_cta": entry.min_kgroups_per_cta,
        "max_kb": entry.max_kb,
        "force_two_phase": entry.force_two_phase,
        "exact_kb": entry.exact_kb,
    }
    return (
        f"            {{0x{pack_shape_key(entry.execution_mode, 1, entry.n, entry.k):016x}ULL, "
        f"{_shape_enum(entry.family)}, {_tuning_literal(config)}}}, "
        f"// CB={entry.codebook} mode={entry.execution_mode.value} M=1 "
        f"{entry.n}x{entry.k} "
        f"{entry.candidate_id} {entry.shape_name} "
        f"max-regret={entry.max_surface_regret:.4%} max-cv={entry.max_cv:.4%}"
    )


def _resolved_tuning_lines(config: dict[str, object]) -> list[str]:
    """Render a concrete tuning assignment for a static or formula candidate."""

    if config["family"] != "kpar_formula":
        return [
            f"            tuning = GeneratedDispatchTuning{_tuning_literal(config)};"
        ]
    tile_n = int(config["tile_n"])
    cpt = int(config["cpt"])
    maximum = int(config["max_kb"])
    formula_kind = str(config["formula_kind"])
    if formula_kind == "target_blocks":
        resolver = (
            "resolveGeneratedTargetBlocksKBlocks("
            f"(n + {tile_n} - 1) / {tile_n}, k / 32, "
            f"{int(config['target_blocks'])}, "
            f"{int(config['min_kgroups_per_cta'])}, {maximum})"
        )
    elif formula_kind == "canonical_target_blocks":
        resolver = (
            "resolveGeneratedCanonicalTargetBlocksKBlocks("
            f"(n + {tile_n} - 1) / {tile_n}, k / 32, "
            f"{int(config['target_blocks'])}, "
            f"{int(config['min_kgroups_per_cta'])}, {maximum})"
        )
    elif formula_kind == "blocks_per_partition":
        resolver = (
            "resolveGeneratedBlocksPerPartitionKBlocks("
            f"k / 32, {int(config['blocks_per_partition'])}, {maximum})"
        )
    else:
        raise ValueError(f"unsupported CUDA formula kind {formula_kind!r}")
    return [
        f"            const int resolved_kb = {resolver};",
        "            if (resolved_kb <= 0)",
        "                return false;",
        (
            "            tuning = GeneratedDispatchTuning"
            f"{{{tile_n}, {cpt}, 0, 0, 0, 1, resolved_kb}};"
        ),
    ]


def generate_include(
    entries: list[FastEntry],
    generic_rules: list[GenericDispatchRule],
    *,
    corpus_digest: str,
    registry_digest: str,
    profile: MeasurementProfile,
    exact_only: bool = False,
    certification: CertificationReport | None = None,
    policy_digest: str = "",
    shape_manifest_digest: str = "",
    measurement_plan_digest: str = "",
) -> str:
    """Render exact and bounded generic decisions into the CUDA selector ABI."""

    entries_by_codebook: dict[int, list[FastEntry]] = {}
    for entry in entries:
        entries_by_codebook.setdefault(entry.codebook, []).append(entry)
    for rows in entries_by_codebook.values():
        rows.sort(key=lambda item: pack_shape_key(
            item.execution_mode, 1, item.n, item.k
        ))

    rules_by_codebook: dict[int, list[GenericDispatchRule]] = {}
    for rule in generic_rules:
        rules_by_codebook.setdefault(
            rule.domain.runtime_codebook_id, []
        ).append(rule)

    if certification is not None:
        decision_comment = (
            "// Decisions: common alias-robust, mode-specific exact oracle + "
            "frozen development policy with generic-only sealed certification."
        )
    elif exact_only:
        decision_comment = (
            "// Decisions: common alias-robust, mode-specific exact oracle; "
            "generic policy intentionally absent pending sealed certification."
        )
    else:
        decision_comment = (
            "// Decisions: common alias-robust, mode-specific exact oracle + "
            "development-only segmented-regret learner; this artifact is not "
            "installable without sealed certification."
        )

    lines = [
        "// Auto-generated by analyze_cuda_native_vnni_decode_trainer.py. DO NOT EDIT.",
        decision_comment,
        "// Grouped verifier M=2..16 and M=31 inherit this exact public-M1 policy.",
        "// Uncovered runtime keys return false; production has no default route.",
        f"// Measurement profile: {profile.value}",
        f"// Common corpus digest: {corpus_digest}",
        f"// Candidate registry digest: {registry_digest}",
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "struct GeneratedDispatchTuning",
        "{",
        "    int tile_n = 0;",
        "    int cpt = 0;",
        "    int target_waves = 0;",
        "    int mkg = 0;",
        "    int max_kb = 0;",
        "    int force_two_phase = 0;",
        "    // Exact KB is part of the batch-invariant arithmetic identity.",
        "    int exact_kb = 0;",
        "};",
        "",
        "struct GeneratedDispatchEntry",
        "{",
        "    uint64_t key;",
        "    NativeGemvShape shape;",
        "    GeneratedDispatchTuning tuning;",
        "};",
        "",
        "inline constexpr uint64_t packGeneratedDispatchKey(",
        "    bool graph_captured, int m, int n, int k)",
        "{",
        "    return (static_cast<uint64_t>(graph_captured ? 1 : 0) << 63) |",
        "           (static_cast<uint64_t>(m & 0x7F) << 56) |",
        "           (static_cast<uint64_t>(k & 0xFFFFFF) << 28) |",
        "           static_cast<uint64_t>(n & 0x0FFFFFFF);",
        "}",
        "",
        "template <size_t Count>",
        "inline bool findGeneratedDispatchEntry(",
        "    const GeneratedDispatchEntry (&table)[Count], uint64_t key,",
        "    NativeGemvShape &shape, GeneratedDispatchTuning &tuning)",
        "{",
        "    size_t lo = 0;",
        "    size_t hi = Count;",
        "    while (lo < hi)",
        "    {",
        "        const size_t mid = lo + ((hi - lo) / 2);",
        "        if (table[mid].key == key)",
        "        {",
        "            shape = table[mid].shape;",
        "            tuning = table[mid].tuning;",
        "            return true;",
        "        }",
        "        if (table[mid].key < key)",
        "            lo = mid + 1;",
        "        else",
        "            hi = mid;",
        "    }",
        "    return false;",
        "}",
        "",
        "inline int resolveGeneratedTargetBlocksKBlocks(",
        "    int grid_n, int k_groups, int target_blocks,",
        "    int min_kgroups_per_cta, int max_kb)",
        "{",
        "    if (grid_n <= 0 || k_groups <= 0 || target_blocks <= 0 ||",
        "        min_kgroups_per_cta <= 0 || max_kb <= 0)",
        "        return 0;",
        "    int kb = (target_blocks + grid_n - 1) / grid_n;",
        "    if (kb < 2)",
        "        kb = 2;",
        "    int kb_max = k_groups / min_kgroups_per_cta;",
        "    if (kb_max < 2)",
        "        kb_max = 2;",
        "    if (kb > kb_max)",
        "        kb = kb_max;",
        "    if (k_groups % kb != 0)",
        "    {",
        "        int lower = -1;",
        "        int upper = -1;",
        "        for (int distance = 1; distance < kb; ++distance)",
        "        {",
        "            const int candidate = kb - distance;",
        "            if (candidate >= 2 && k_groups % candidate == 0)",
        "            {",
        "                lower = candidate;",
        "                break;",
        "            }",
        "        }",
        "        for (int distance = 1; kb + distance <= kb_max; ++distance)",
        "        {",
        "            const int candidate = kb + distance;",
        "            if (k_groups % candidate == 0)",
        "            {",
        "                upper = candidate;",
        "                break;",
        "            }",
        "        }",
        "        if (lower > 0 && upper > 0)",
        "        {",
        "            long long lower_distance =",
        "                static_cast<long long>(grid_n) * lower - target_blocks;",
        "            long long upper_distance =",
        "                static_cast<long long>(grid_n) * upper - target_blocks;",
        "            if (lower_distance < 0)",
        "                lower_distance = -lower_distance;",
        "            if (upper_distance < 0)",
        "                upper_distance = -upper_distance;",
        "            kb = upper_distance < lower_distance ? upper : lower;",
        "        }",
        "        else if (lower > 0)",
        "            kb = lower;",
        "        else if (upper > 0)",
        "            kb = upper;",
        "    }",
        "    if (kb > max_kb)",
        "        kb = max_kb;",
        "    if (kb > k_groups)",
        "        kb = k_groups;",
        "    return kb > 0 ? kb : 0;",
        "}",
        "",
        "inline int resolveGeneratedCanonicalTargetBlocksKBlocks(",
        "    int grid_n, int k_groups, int target_blocks,",
        "    int min_kgroups_per_cta, int max_kb)",
        "{",
        "    if (grid_n <= 0 || k_groups <= 0 || target_blocks <= 0 ||",
        "        min_kgroups_per_cta <= 0 || max_kb <= 0)",
        "        return 0;",
        "    int desired_kb = (target_blocks + grid_n - 1) / grid_n;",
        "    if (desired_kb < 2)",
        "        desired_kb = 2;",
        "    int kb_max = k_groups / min_kgroups_per_cta;",
        "    if (kb_max < 1)",
        "        kb_max = 1;",
        "    if (desired_kb > kb_max)",
        "        desired_kb = kb_max;",
        "    if (desired_kb > max_kb)",
        "        desired_kb = max_kb;",
        "    if (desired_kb > k_groups)",
        "        desired_kb = k_groups;",
        "    const int groups_per_partition =",
        "        (k_groups + desired_kb - 1) / desired_kb;",
        "    int canonical_kb =",
        "        (k_groups + groups_per_partition - 1) / groups_per_partition;",
        "    if (canonical_kb > max_kb)",
        "        canonical_kb = max_kb;",
        "    if (canonical_kb > k_groups)",
        "        canonical_kb = k_groups;",
        "    return canonical_kb > 0 ? canonical_kb : 0;",
        "}",
        "",
        "inline int resolveGeneratedBlocksPerPartitionKBlocks(",
        "    int k_groups, int blocks_per_partition, int max_kb)",
        "{",
        "    if (k_groups <= 0 || blocks_per_partition <= 0 || max_kb <= 0)",
        "        return 0;",
        "    int kb = (k_groups + blocks_per_partition - 1) /",
        "             blocks_per_partition;",
        "    if (kb > max_kb)",
        "        kb = max_kb;",
        "    if (kb > k_groups)",
        "        kb = k_groups;",
        "    return kb > 0 ? kb : 0;",
        "}",
        "",
        "template <uint8_t CB>",
        "inline bool selectGeneratedDispatch(",
        "    bool graph_captured, int m, int n, int k, NativeGemvShape &shape,",
        "    GeneratedDispatchTuning &tuning)",
        "{",
        "    if (m != 1)",
        "        return false;",
        "    const uint64_t key = packGeneratedDispatchKey(",
        "        graph_captured, m, n, k);",
    ]

    for codebook in sorted(entries_by_codebook):
        lines.extend([
            f"    if constexpr (CB == {codebook})",
            "    {",
            "        static constexpr GeneratedDispatchEntry kTable[] = {",
        ])
        lines.extend(_entry_line(entry) for entry in entries_by_codebook[codebook])
        lines.extend([
            "        };",
            "        if (findGeneratedDispatchEntry(kTable, key, shape, tuning))",
            "            return true;",
            "    }",
        ])

    if rules_by_codebook:
        lines.extend([
            "",
            "    const long long work_items =",
            "        static_cast<long long>(n) * static_cast<long long>(k);",
    ])
    if certification is not None:
        lines[7:7] = [
            f"// Common policy digest: {policy_digest}",
            f"// Shape manifest digest: {shape_manifest_digest}",
            f"// Measurement plan digest: {measurement_plan_digest}",
            (
                "// Frozen generic policy digest: "
                f"{certification.frozen_generic_policy_digest}"
            ),
            (
                "// Sealed generic certificate: coverage="
                f"{certification.covered_cell_count}/"
                f"{certification.required_cell_count} max-regret="
                f"{certification.max_observed_regret:.6%} max-simultaneous-ucb="
                f"{certification.max_simultaneous_95pct_upper_regret:.6%}"
            ),
        ]
    for codebook in sorted(rules_by_codebook):
        lines.extend([
            f"    if constexpr (CB == {codebook})",
            "    {",
        ])
        for rule in sorted(
            rules_by_codebook[codebook],
            key=lambda item: (
                item.domain.execution_mode.value,
                generic_rule_sort_key(item),
            ),
        ):
            config = _fast_config(rule.candidate_id)
            mode_condition = (
                "graph_captured"
                if rule.domain.execution_mode == ExecutionMode.GRAPH_CAPTURED
                else "!graph_captured"
            )
            conditions = [
                mode_condition,
                aspect_condition(rule.domain.aspect_bucket),
                *(predicate_condition(predicate) for predicate in rule.predicates),
            ]
            if config["family"] == "kpar":
                conditions.append(f"k / 32 >= {int(config['exact_kb'])}")
            lines.extend(render_if_header(conditions, indent="        "))
            lines.extend([
                "        {",
                f"            shape = {_shape_enum(str(config['family']))};",
                *_resolved_tuning_lines(config),
                "            return true;",
                "        }",
            ])
        lines.append("    }")

    lines.extend([
        "    return false;",
        "}",
        "",
    ])
    return "\n".join(lines)


def _canonical_mapping_digest(mapping: dict[str, object]) -> str:
    """Hash one already-normalized policy mapping exactly like ``PolicyIR``."""

    encoded = json.dumps(
        mapping,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def validate_certified_m1_artifacts(
    policy_path: Path,
    include_path: Path,
    manifest: NativeVNNIShapeManifest,
    measurement_plan: NativeVNNIGPUMeasurementPlan,
) -> None:
    """Bind verifier evidence to the previously certified staged M1 bytes.

    The verifier transaction is deliberately unable to invoke the learner. It
    receives only the immutable common-IR certificate and the exact include that
    was compiled into the verifier trainer. This validator recomputes both IR
    digests, enforces every sealed gate, and requires the include's provenance
    comments to name the same manifest and generic digest before those exact
    include bytes may be republished.
    """

    payload = validate_installable_policy_artifact(
        policy_path,
        include_path=include_path,
    )
    policy = payload["policy"]
    certification = payload["certification"]
    if not isinstance(policy, dict) or not isinstance(certification, dict):
        raise ValueError("certified CUDA M1 policy payload is malformed")
    generic_fields = {
        "policy_abi",
        "learner_version",
        "feature_schema_version",
        "generic_rules",
        "unpromoted_domains",
        "cross_validation",
    }
    if not generic_fields.issubset(policy):
        raise ValueError("certified CUDA M1 policy omits generic IR fields")
    generic_mapping = {name: policy[name] for name in generic_fields}
    frozen_digest = _canonical_mapping_digest(generic_mapping)
    if frozen_digest != payload["frozen_generic_policy_digest"]:
        raise ValueError("certified CUDA M1 generic digest does not match its IR")

    metadata = policy.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("certified CUDA M1 policy omits metadata")
    if metadata.get("shape_manifest_digest") != manifest.digest():
        raise ValueError("certified CUDA M1 policy names a different shape manifest")
    if metadata.get("measurement_plan_digest") != measurement_plan.digest(manifest):
        raise ValueError("certified CUDA M1 policy names a different measurement plan")
    if metadata.get("frozen_generic_policy_digest") != frozen_digest:
        raise ValueError("certified CUDA M1 metadata changed after generic freeze")
    if not metadata.get("development_corpus_digest"):
        raise ValueError("certified CUDA M1 policy omits its development digest")
    if not metadata.get("sealed_corpus_digest"):
        raise ValueError("certified CUDA M1 policy omits its sealed digest")

    include = include_path.read_text(encoding="utf-8")
    required_comments = (
        f"// Shape manifest digest: {manifest.digest()}",
        f"// Measurement plan digest: {measurement_plan.digest(manifest)}",
        f"// Frozen generic policy digest: {frozen_digest}",
        "// Sealed generic certificate: coverage=",
    )
    missing = [comment for comment in required_comments if comment not in include]
    if missing:
        raise ValueError(
            "certified CUDA M1 include is not bound to its policy certificate: "
            f"missing={missing}"
        )


def write_summary(
    path: Path,
    entries: list[FastEntry],
    exact: dict[RuntimeKey, ExactWinner],
) -> None:
    """Write selected M1 schedules and the number of certified verifier keys."""

    verifier_count = sum(
        key.semantic_contract == SemanticContract.VERIFIER_SERIAL_M1_BITWISE
        for key in exact
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "execution_codebook", "execution_mode", "m", "n", "k",
            "candidate_id", "family",
            "tile_n", "cpt", "exact_kb", "shape", "max_surface_regret",
            "max_cv", "certified_verifier_key_count",
        ])
        for entry in sorted(entries):
            writer.writerow([
                entry.codebook,
                entry.execution_mode.value,
                1,
                entry.n,
                entry.k,
                entry.candidate_id,
                entry.family,
                entry.tile_n,
                entry.cpt,
                entry.exact_kb,
                entry.shape_name,
                f"{entry.max_surface_regret:.9f}",
                f"{entry.max_cv:.9f}",
                verifier_count,
            ])


def _context_from_args(
    args: argparse.Namespace,
    inputs: tuple[Path, ...],
    timing_sidecars: tuple[Path, ...],
    *,
    run_id: str | None = None,
    build_id: str | None = None,
    serial_m1_policy_hash: str | None = None,
) -> CUDADecodeAdapterContext:
    """Build conspicuous smoke provenance or strict installable provenance."""

    corpus_id = raw_corpus_id((*inputs, *timing_sidecars))
    profile = MeasurementProfile(args.profile)
    if not profile.installable:
        return CUDADecodeAdapterContext.workflow_smoke(corpus_id=corpus_id)
    return CUDADecodeAdapterContext(
        profile=profile,
        run_id=run_id if run_id is not None else args.run_id,
        corpus_id=corpus_id,
        git_revision=args.git_revision,
        build_id=build_id if build_id is not None else args.build_id,
        compiler_id=args.compiler_id,
        architecture_class=args.architecture_class,
        device_name=args.device_name,
        driver_runtime=args.driver_runtime,
        serial_m1_policy_hash=(
            serial_m1_policy_hash
            if serial_m1_policy_hash is not None
            else args.serial_m1_policy_hash
        ),
        raw_timing_sidecar_retained=bool(timing_sidecars),
    )


def main() -> int:
    """Adapt strong evidence and emit one CUDA production selector artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument(
        "--input", nargs="+", type=Path, dest="input_options",
        help="Strong CUDA trainer CSV shard(s)",
    )
    parser.add_argument(
        "--timing-sidecar", action="append", type=Path, default=[],
        help="Raw timing CSV shard; repeat for every aggregate shard",
    )
    parser.add_argument(
        "--m1-input", nargs="+", type=Path,
        help="Frozen first-phase M1 aggregate shard(s) for a final compile",
    )
    parser.add_argument(
        "--m1-timing-sidecar", action="append", type=Path, default=[],
        help="First-phase M1 timing shard; repeat for every aggregate shard",
    )
    parser.add_argument(
        "--verifier-input", nargs="+", type=Path,
        help="Second-phase verifier aggregate shard(s) for a final compile",
    )
    parser.add_argument(
        "--verifier-timing-sidecar", action="append", type=Path, default=[],
        help="Second-phase verifier timing shard; repeat per aggregate shard",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", "--summary-csv", dest="summary", type=Path)
    parser.add_argument("--common-observations", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--require-fast-m1-complete",
        action="store_true",
        help="Validate the independent all-format M1 staging matrix",
    )
    parser.add_argument(
        "--exact-only",
        action="store_true",
        help="Emit exact M1 entries without uncertified generic rules",
    )
    parser.add_argument(
        "--certify-generic",
        action="store_true",
        help=(
            "Recompute and validate a previously frozen development policy, "
            "then open a physically separate sealed corpus exactly once"
        ),
    )
    parser.add_argument(
        "--freeze-generic",
        action="store_true",
        help=(
            "Fit and publish development-only generic IR without reading any "
            "sealed aggregate or timing file"
        ),
    )
    parser.add_argument(
        "--development-input",
        nargs="+",
        type=Path,
        help="Frozen development aggregate shard(s) for separate certification",
    )
    parser.add_argument(
        "--development-timing-sidecar",
        action="append",
        type=Path,
        default=[],
        help="Development timing shard; repeat for every aggregate shard",
    )
    parser.add_argument(
        "--sealed-input",
        nargs="+",
        type=Path,
        help="Opaque sealed aggregate shard(s) opened only after digest validation",
    )
    parser.add_argument(
        "--sealed-timing-sidecar",
        action="append",
        type=Path,
        default=[],
        help="Sealed timing shard; repeat for every aggregate shard",
    )
    parser.add_argument(
        "--frozen-policy-json",
        type=Path,
        help="Development policy artifact published before sealed measurement",
    )
    parser.add_argument(
        "--shape-manifest",
        type=Path,
        default=MANIFEST_PATH,
        help="Reviewed cross-backend shape and split manifest",
    )
    parser.add_argument(
        "--measurement-plan",
        type=Path,
        default=MEASUREMENT_PLAN_PATH,
        help="Reviewed bounded and backend-scoped GPU timing plan",
    )
    parser.add_argument(
        "--policy-json",
        type=Path,
        help="Write common policy IR and sealed certificate JSON",
    )
    parser.add_argument(
        "--paired-development-csv",
        action="append",
        type=Path,
        default=[],
        help=(
            "Development-only paired tournament CSV; repeat for every shard. "
            "These ratios may refine CV but never sealed evaluation."
        ),
    )
    parser.add_argument(
        "--development-profiler-requests",
        type=Path,
        help=(
            "Authenticated isolated profiler requests derived from the "
            "development common-observation corpus"
        ),
    )
    parser.add_argument(
        "--development-profiler-evidence",
        type=Path,
        help=(
            "Complete per-candidate Nsight evidence for development fitting"
        ),
    )
    parser.add_argument(
        "--development-profiler-observations",
        type=Path,
        help="Original common CSV bound to reusable profiler sidecars",
    )
    parser.add_argument(
        "--certified-m1-include",
        type=Path,
        help="Exact sealed-certified M1 include compiled into the verifier trainer",
    )
    parser.add_argument(
        "--certified-m1-policy-json",
        type=Path,
        help="Sealed certificate and common IR bound to --certified-m1-include",
    )
    parser.add_argument(
        "--profile", choices=[profile.value for profile in MeasurementProfile],
        default=MeasurementProfile.QUICK.value,
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument("--git-revision", default="")
    parser.add_argument("--build-id", default="")
    parser.add_argument("--m1-build-id", default="")
    parser.add_argument("--compiler-id", default="")
    parser.add_argument("--architecture-class", default="")
    parser.add_argument("--device-name", default="")
    parser.add_argument("--driver-runtime", default="")
    parser.add_argument("--serial-m1-policy-hash", default="")
    parser.add_argument("--m1-baseline-policy-hash", default="")
    args = parser.parse_args()

    if args.freeze_generic and args.certify_generic:
        parser.error("--freeze-generic and --certify-generic are mutually exclusive")
    if args.exact_only and (args.certify_generic or args.freeze_generic):
        parser.error(
            "--exact-only cannot be combined with generic freeze/certification"
        )
    if (args.certify_generic or args.freeze_generic) and (
        args.profile != MeasurementProfile.PRODUCTION.value
    ):
        parser.error("generic freeze/certification requires --profile production")
    if args.policy_json and not (args.certify_generic or args.freeze_generic):
        parser.error("--policy-json requires generic freeze/certification")
    if args.paired_development_csv and not (
        args.certify_generic or args.freeze_generic
    ):
        parser.error(
            "--paired-development-csv requires generic freeze/certification"
        )
    if bool(args.development_profiler_requests) != bool(
        args.development_profiler_evidence
    ):
        parser.error(
            "development profiler requests and evidence are required together"
        )
    if args.development_profiler_observations and not (
        args.development_profiler_requests
        and args.development_profiler_evidence
    ):
        parser.error(
            "development profiler observations require requests and evidence"
        )
    if (args.freeze_generic or args.certify_generic) and not (
        args.development_profiler_requests
        and args.development_profiler_evidence
    ):
        parser.error(
            "production CUDA freeze/certification requires complete "
            "development profiler evidence"
        )
    certified_replay = bool(
        args.certified_m1_include or args.certified_m1_policy_json
    )
    if bool(args.certified_m1_include) != bool(args.certified_m1_policy_json):
        parser.error(
            "--certified-m1-include and --certified-m1-policy-json are required "
            "together"
        )
    if certified_replay and (args.certify_generic or args.freeze_generic):
        parser.error(
            "certified M1 replay cannot invoke generic freeze/certification"
        )

    positional = tuple(args.inputs)
    optional = tuple(args.input_options or ())
    if positional and optional:
        parser.error("use positional inputs or --input, not both")
    inputs = positional or optional
    separate_certification = bool(
        args.development_input
        or args.sealed_input
        or args.development_timing_sidecar
        or args.sealed_timing_sidecar
        or args.frozen_policy_json
    )
    if separate_certification:
        if not args.certify_generic:
            parser.error(
                "separate development/sealed inputs require --certify-generic"
            )
        if inputs or args.timing_sidecar or args.m1_input or args.verifier_input:
            parser.error(
                "separate certification cannot mix normal or two-phase inputs"
            )
        if not args.development_input or not args.sealed_input:
            parser.error(
                "separate certification requires development and sealed inputs"
            )
        if not args.frozen_policy_json:
            parser.error("separate certification requires --frozen-policy-json")
    elif args.certify_generic:
        parser.error(
            "--certify-generic requires physically separate --development-input, "
            "--sealed-input, and --frozen-policy-json artifacts"
        )

    paired_development_comparisons = (
        paired_timing_comparisons(read_paired_confirmation_csv(
            tuple(args.paired_development_csv)
        ))
        if args.paired_development_csv
        else {}
    )

    if args.freeze_generic:
        if not inputs:
            parser.error("--freeze-generic requires development --input shards")
        if args.m1_input or args.verifier_input or separate_certification:
            parser.error("--freeze-generic accepts development input only")
        if not args.policy_json:
            parser.error("--freeze-generic requires --policy-json")
        timing_sidecars = tuple(args.timing_sidecar)
        context = _context_from_args(args, inputs, timing_sidecars)
        corpus = adapt_cuda_decode_csv(
            inputs,
            context,
            timing_sidecars=timing_sidecars,
        )
        profiler_catalog = load_profiler_feature_catalog(
            corpus,
            args.development_profiler_requests,
            args.development_profiler_evidence,
            source_corpus=(
                read_observation_csv((args.development_profiler_observations,))
                if args.development_profiler_observations
                else None
            ),
        )
        manifest = load_shape_manifest(args.shape_manifest)
        measurement_plan = load_gpu_measurement_plan(
            args.measurement_plan,
            manifest=manifest,
        )
        if args.require_fast_m1_complete:
            validate_fast_m1_complete(corpus)
        frozen = freeze_fast_policy(
            corpus,
            manifest,
            measurement_plan,
            paired_development_comparisons=paired_development_comparisons,
            profiler_feature_catalog=profiler_catalog,
        )
        entries, exact = select_fast_entries(
            corpus, context.serial_m1_policy_hash
        )
        exact_overlay_names = {
            shape.name for shape in manifest.shapes if shape.exact_overlay
        }
        entries = [
            entry for entry in entries
            if entry.shape_name in exact_overlay_names
        ]
        generic_rules = list(frozen.policy_ir.generic_rules)
        validate_policy_ir_inputs(frozen.policy_ir, entries, generic_rules)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(generate_include(
            entries,
            generic_rules,
            corpus_digest=corpus.digest(),
            registry_digest=candidate_registry_digest(),
            profile=context.profile,
            exact_only=False,
            certification=None,
            policy_digest=frozen.policy_ir.digest(),
            shape_manifest_digest=manifest.digest(),
            measurement_plan_digest=measurement_plan.digest(manifest),
        ), encoding="utf-8")
        write_frozen_policy(args.policy_json, frozen)
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            write_summary(args.summary, entries, exact)
        if args.common_observations:
            args.common_observations.parent.mkdir(parents=True, exist_ok=True)
            write_observation_csv(args.common_observations, corpus)
        print(
            f"froze {len(corpus)} development observations as "
            f"{frozen.generic_digest} -> {args.output}"
        )
        return 0

    if separate_certification:
        development_inputs = tuple(args.development_input)
        development_timing = tuple(args.development_timing_sidecar)
        development_context = _context_from_args(
            args, development_inputs, development_timing
        )
        development_corpus = adapt_cuda_decode_csv(
            development_inputs,
            development_context,
            timing_sidecars=development_timing,
        )
        profiler_catalog = load_profiler_feature_catalog(
            development_corpus,
            args.development_profiler_requests,
            args.development_profiler_evidence,
            source_corpus=(
                read_observation_csv((args.development_profiler_observations,))
                if args.development_profiler_observations
                else None
            ),
        )
        manifest = load_shape_manifest(args.shape_manifest)
        measurement_plan = load_gpu_measurement_plan(
            args.measurement_plan,
            manifest=manifest,
        )
        if args.require_fast_m1_complete:
            validate_fast_m1_complete(development_corpus)
        frozen = freeze_fast_policy(
            development_corpus,
            manifest,
            measurement_plan,
            paired_development_comparisons=paired_development_comparisons,
            profiler_feature_catalog=profiler_catalog,
        )
        # This comparison happens before the first sealed file is opened.
        validate_frozen_policy_file(args.frozen_policy_json, frozen)

        sealed_inputs = tuple(args.sealed_input)
        sealed_timing = tuple(args.sealed_timing_sidecar)
        sealed_context = _context_from_args(
            args,
            sealed_inputs,
            sealed_timing,
            run_id=f"{args.run_id}-sealed",
        )
        sealed_corpus = adapt_cuda_decode_csv(
            sealed_inputs,
            sealed_context,
            timing_sidecars=sealed_timing,
        )
        if args.require_fast_m1_complete:
            validate_fast_m1_complete(sealed_corpus)
        compiled = certify_fast_policy(
            frozen,
            development_corpus,
            sealed_corpus,
            manifest,
            measurement_plan,
        )
        corpus = ObservationCorpus((
            *development_corpus.observations,
            *sealed_corpus.observations,
        ))
        entries, exact = select_fast_entries(
            corpus, development_context.serial_m1_policy_hash
        )
        exact_overlay_names = {
            shape.name for shape in manifest.shapes if shape.exact_overlay
        }
        entries = [
            entry for entry in entries
            if entry.shape_name in exact_overlay_names
        ]
        generic_rules = list(compiled.policy_ir.generic_rules)
        validate_emitter_inputs(compiled, entries, generic_rules)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(generate_include(
            entries,
            generic_rules,
            corpus_digest=corpus.digest(),
            registry_digest=candidate_registry_digest(),
            profile=development_context.profile,
            certification=compiled.certification,
            policy_digest=compiled.policy_ir.digest(),
            shape_manifest_digest=manifest.digest(),
            measurement_plan_digest=measurement_plan.digest(manifest),
        ), encoding="utf-8")
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            write_summary(args.summary, entries, exact)
        if args.common_observations:
            args.common_observations.parent.mkdir(parents=True, exist_ok=True)
            write_observation_csv(args.common_observations, corpus)
        if args.policy_json:
            write_compiled_policy(args.policy_json, compiled)
        print(
            f"certified frozen policy {frozen.generic_digest} against "
            f"{len(sealed_corpus)} sealed observations -> {args.output}"
        )
        return 0

    two_phase = bool(args.m1_input or args.verifier_input)
    if certified_replay and not two_phase:
        parser.error("certified M1 replay requires the two-phase verifier inputs")
    if two_phase:
        if inputs or args.timing_sidecar:
            parser.error(
                "two-phase compile cannot mix --m1/--verifier inputs with "
                "positional/--input evidence"
            )
        if not args.m1_input or not args.verifier_input:
            parser.error("two-phase compile requires both --m1-input and --verifier-input")
        if not args.m1_build_id or not args.m1_baseline_policy_hash:
            parser.error(
                "two-phase compile requires --m1-build-id and "
                "--m1-baseline-policy-hash"
            )
        m1_inputs = tuple(args.m1_input)
        verifier_inputs = tuple(args.verifier_input)
        m1_timing = tuple(args.m1_timing_sidecar)
        verifier_timing = tuple(args.verifier_timing_sidecar)
        m1_context = _context_from_args(
            args,
            m1_inputs,
            m1_timing,
            run_id=f"{args.run_id}-m1",
            build_id=args.m1_build_id,
            serial_m1_policy_hash=args.m1_baseline_policy_hash,
        )
        verifier_context = _context_from_args(
            args,
            verifier_inputs,
            verifier_timing,
            run_id=f"{args.run_id}-verifier",
            build_id=args.build_id,
            serial_m1_policy_hash=args.serial_m1_policy_hash,
        )
        m1_corpus = adapt_cuda_decode_csv(
            m1_inputs,
            m1_context,
            timing_sidecars=m1_timing,
        )
        verifier_corpus = adapt_cuda_decode_csv(
            verifier_inputs,
            verifier_context,
            timing_sidecars=verifier_timing,
        )
        corpus = ObservationCorpus((
            *m1_corpus.observations,
            *verifier_corpus.observations,
        ))
        context = verifier_context
    else:
        if not inputs:
            parser.error("at least one strong trainer CSV is required")
        timing_sidecars = tuple(args.timing_sidecar)
        context = _context_from_args(args, inputs, timing_sidecars)
        corpus = adapt_cuda_decode_csv(
            inputs,
            context,
            timing_sidecars=timing_sidecars,
        )
    entries, exact = select_fast_entries(
        corpus,
        context.serial_m1_policy_hash,
    )
    generic_rules = (
        []
        if args.exact_only or certified_replay
        else select_fast_generic_rules(corpus, context.serial_m1_policy_hash)
    )
    if args.require_fast_m1_complete:
        validate_fast_m1_complete(corpus)
    if args.require_complete:
        complete_manifest = load_shape_manifest(args.shape_manifest)
        validate_complete(
            corpus,
            complete_manifest,
            load_gpu_measurement_plan(
                args.measurement_plan,
                manifest=complete_manifest,
            ),
            require_full_inventory=(
                context.profile == MeasurementProfile.PRODUCTION
            ),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if certified_replay:
        manifest = load_shape_manifest(args.shape_manifest)
        measurement_plan = load_gpu_measurement_plan(
            args.measurement_plan,
            manifest=manifest,
        )
        validate_certified_m1_artifacts(
            args.certified_m1_policy_json,
            args.certified_m1_include,
            manifest,
            measurement_plan,
        )
        generated = args.certified_m1_include.read_text(encoding="utf-8")
    else:
        generated = generate_include(
            entries,
            generic_rules,
            corpus_digest=corpus.digest(),
            registry_digest=candidate_registry_digest(),
            profile=context.profile,
            exact_only=args.exact_only,
        )
    args.output.write_text(generated, encoding="utf-8")
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        write_summary(args.summary, entries, exact)
    if args.common_observations:
        args.common_observations.parent.mkdir(parents=True, exist_ok=True)
        write_observation_csv(args.common_observations, corpus)
    verifier_count = sum(
        key.semantic_contract == SemanticContract.VERIFIER_SERIAL_M1_BITWISE
        for key in exact
    )
    print(
        f"adapted {len(corpus)} strong observations; selected {len(entries)} "
        f"Fast M=1 exact entries and certified {verifier_count} verifier keys "
        f"-> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
