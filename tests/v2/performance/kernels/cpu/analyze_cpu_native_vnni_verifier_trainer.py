#!/usr/bin/env python3
"""Compile strong CPU grouped-verifier evidence through the common policy core.

Byte equality, effective route identity, repeat stability, alias coverage, and
candidate selection are shared with CUDA/ROCm.  The backend emitter only maps
the common Pairwise/WideRows decisions into the existing CPU selector ABI; it
does not apply independent cosine, L2, KL, or speedup filters.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


KERNEL_PERF_ROOT = Path(__file__).resolve().parents[1]
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.adapters.cpu_verifier import (  # noqa: E402
    CPUVerifierAdapterContext,
    adapt_cpu_verifier_csv,
    adapt_cpu_verifier_csv_to_common,
    raw_corpus_id,
)
from native_vnni_dispatch.candidate_observation import (  # noqa: E402
    read_observation_csv,
    write_observation_csv,
)
from native_vnni_dispatch.candidate_registry import (  # noqa: E402
    candidate_registry_digest,
    cpu_native_vnni_verifier_registry,
)
from native_vnni_dispatch.certification import CertificationReport  # noqa: E402
from native_vnni_dispatch.compiler import (  # noqa: E402
    CompiledPolicy,
    FrozenPolicy,
    finalize_frozen_policy_certificate,
    freeze_policy,
)
from native_vnni_dispatch.cpu_grouped_decode_sealed_plan import (  # noqa: E402
    build_cpu_grouped_sealed_plan,
    certify_cpu_grouped_sealed_pairs,
    cpu_grouped_burned_seal_costs,
    cpu_grouped_sealed_reserve_commitment,
    read_cpu_grouped_sealed_plan,
    validate_cpu_grouped_sealed_plan,
    write_cpu_grouped_sealed_plan,
)
from native_vnni_dispatch.cpu_sealed_paired import (  # noqa: E402
    load_cpu_burned_seal_development,
    paired_evidence_digest,
    resolve_sealed_paired_evidence_paths,
)
from native_vnni_dispatch.corpus import (  # noqa: E402
    GenericDomain,
    ObservationCorpus,
    RuntimeKey,
)
from native_vnni_dispatch.cpp_predicates import (  # noqa: E402
    aspect_condition,
    generic_rule_sort_key,
    predicate_condition,
    render_if_header,
)
from native_vnni_dispatch.exact_oracle import build_exact_winners  # noqa: E402
from native_vnni_dispatch.policy_artifact import (  # noqa: E402
    validate_frozen_policy_file,
    write_certification_diagnostic,
    write_compiled_policy,
    write_frozen_policy,
)
from native_vnni_dispatch.policy_ir import PolicyIR  # noqa: E402
from native_vnni_dispatch.paired_confirmation import (  # noqa: E402
    PairedCellKey,
    PairedTimingComparison,
    paired_timing_comparisons,
    read_paired_confirmation_csv,
)
from native_vnni_dispatch.paired_requests import (  # noqa: E402
    paired_comparison_digest,
)
from native_vnni_dispatch.profiles import (  # noqa: E402
    MIN_PROMOTION_SAMPLES,
    MIN_PROMOTION_WARMUPS,
    MeasurementProfile,
)
from native_vnni_dispatch.profiler_model import (  # noqa: E402
    ProfilerFeatureCatalog,
    load_profiler_feature_catalog,
)
from native_vnni_dispatch.schema import SemanticContract  # noqa: E402
from native_vnni_dispatch.segmented_policy import (  # noqa: E402
    CandidatePointCost,
    DEFAULT_TREE_LEAVES,
    GenericDispatchRule,
    MAX_TREE_LEAVES,
    PolicyFitCache,
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
    require_candidate_matrix_complete,
    require_canonical_alias_coverage,
    require_registry_candidate_coverage,
    require_verifier_m_matrix,
)


@dataclass(frozen=True, order=True)
class PolicyEntry:
    """One common exact decision encoded for the CPU generated selector."""

    build_isa: str
    runtime_isa: str
    threads: int
    codebook: int
    m: int
    n: int
    k: int
    policy: str
    candidate_id: str
    shape_names: tuple[str, ...]
    max_surface_regret: float
    max_cv: float


@dataclass(frozen=True)
class CPUGenericDispatchRule:
    """One generic rule plus the CPU runtime surface that certified it."""

    build_isa: str
    runtime_isa: str
    threads: int
    rule: GenericDispatchRule


CPU_VERIFIER_POLICIES = (
    "Pairwise",
    "WideRows",
    "FullKRowChunkGrid",
    "FullKTwoRowNbc1",
    "FullKTwoRowNbc2",
    "FullKTwoRowPairGridNbc1",
    "FullKTwoRowPairGridNbc2",
    "FullKTwoRowPairGridNbc4",
    "FullKTwoRowPairGridNbc8",
)
CPU_VERIFIER_POLICY_ORDINAL = {
    policy: ordinal for ordinal, policy in enumerate(CPU_VERIFIER_POLICIES)
}


def _cpu_runtime_surface(architecture_class: str) -> tuple[str, str, int]:
    """Decode the adapter-owned ISA/thread suffix from an architecture key."""

    parts = architecture_class.rsplit("|", 3)
    if len(parts) != 4:
        raise ValueError(
            "CPU architecture class lacks build/runtime/thread identity: "
            f"{architecture_class!r}"
        )
    build_part, runtime_part, threads_part = parts[1:]
    if not build_part.startswith("build=") or not runtime_part.startswith("runtime="):
        raise ValueError(f"malformed CPU architecture class {architecture_class!r}")
    if not threads_part.startswith("threads="):
        raise ValueError(f"malformed CPU thread class {architecture_class!r}")
    build_isa = build_part.removeprefix("build=")
    runtime_isa = runtime_part.removeprefix("runtime=")
    threads = int(threads_part.removeprefix("threads="))
    if build_isa not in {"AVX2", "AVX512"}:
        raise ValueError(f"unsupported CPU build ISA {build_isa!r}")
    if runtime_isa not in {"AVX2", "AVX512"}:
        raise ValueError(f"unsupported CPU runtime ISA {runtime_isa!r}")
    if build_isa == "AVX2" and runtime_isa != "AVX2":
        raise ValueError("AVX2 build cannot own an AVX512 runtime policy")
    if threads <= 0:
        raise ValueError("CPU policy thread count must be positive")
    return build_isa, runtime_isa, threads


def _pack_key(codebook: int, m: int, n: int, k: int) -> int:
    return (
        ((codebook & 0xFF) << 56)
        | ((m & 0xFF) << 48)
        | ((k & 0xFFFFFF) << 24)
        | (n & 0xFFFFFF)
    )


def _candidate_policy(candidate_id: str) -> str:
    candidate = cpu_native_vnni_verifier_registry().resolve(candidate_id)
    policy = str(candidate.config_json["policy"])
    if policy not in CPU_VERIFIER_POLICY_ORDINAL:
        raise ValueError(f"unknown CPU verifier policy {policy!r}")
    return policy


def _serial_hashes(
    corpus: ObservationCorpus,
    serial_m1_policy_hash: str,
) -> dict[RuntimeKey, str]:
    return {
        key: serial_m1_policy_hash
        for key in corpus.runtime_keys()
        if key.semantic_contract == SemanticContract.VERIFIER_SERIAL_M1_BITWISE
    }


def select_entries(
    corpus: ObservationCorpus,
    serial_m1_policy_hash: str,
) -> list[PolicyEntry]:
    """Select exact candidates with the common alias-robust oracle."""

    winners = build_exact_winners(
        corpus,
        serial_m1_hashes=_serial_hashes(corpus, serial_m1_policy_hash),
    )
    entries = []
    for key, winner in sorted(winners.items()):
        if key.semantic_contract != SemanticContract.VERIFIER_SERIAL_M1_BITWISE:
            raise ValueError(f"unexpected CPU semantic contract in {key}")
        if len(key.projection_n_vector) != 1:
            raise ValueError("CPU verifier policy only supports one projection")
        rows = corpus.rows_for_runtime_key(key)
        shape_names = tuple(sorted({row.shape_name for row in rows}))
        build_isa, runtime_isa, threads = _cpu_runtime_surface(
            key.architecture_class
        )
        entries.append(PolicyEntry(
            build_isa=build_isa,
            runtime_isa=runtime_isa,
            threads=threads,
            codebook=key.runtime_codebook_id,
            m=key.m,
            n=key.aggregate_n,
            k=key.k,
            policy=_candidate_policy(winner.candidate_id),
            candidate_id=winner.candidate_id,
            shape_names=shape_names,
            max_surface_regret=winner.max_surface_regret,
            max_cv=winner.max_cv,
        ))
    return entries


def select_generic_rules(
    corpus: ObservationCorpus,
    serial_m1_policy_hash: str,
) -> list[CPUGenericDispatchRule]:
    """Fit shared aspect/work rules without a backend-local label learner."""

    generic = fit_generic_policy(
        corpus,
        serial_m1_hashes=_serial_hashes(corpus, serial_m1_policy_hash),
    )
    return [
        CPUGenericDispatchRule(
            *_cpu_runtime_surface(rule.domain.architecture_class),
            rule=rule,
        )
        for rule in generic.rules
    ]


def _emit_generic_rules(
    rules: tuple[GenericDispatchRule, ...],
) -> list[CPUGenericDispatchRule]:
    """Attach CPU ISA/thread surfaces to already-frozen common IR rules."""

    return [
        CPUGenericDispatchRule(
            *_cpu_runtime_surface(rule.domain.architecture_class),
            rule=rule,
        )
        for rule in rules
    ]


def _require_partition(
    corpus: ObservationCorpus,
    manifest: NativeVNNIShapeManifest,
    partition: ShapePartition,
) -> ObservationCorpus:
    """Reject cross-partition or incomplete CPU verifier evidence."""

    assignments = partition_assignments(
        ((row.shape_group_id, row.shape_name) for row in corpus),
        verifier=True,
        manifest=manifest,
    )
    if set(assignments.values()) != {partition}:
        raise ValueError(
            f"CPU verifier {partition.value} input crosses a manifest partition"
        )
    validate_complete(
        corpus,
        manifest,
        require_full_inventory=True,
        expected_partition=partition,
    )
    return corpus


def freeze_cpu_verifier_policy(
    development: ObservationCorpus,
    manifest: NativeVNNIShapeManifest,
    serial_m1_policy_hash: str,
    profiler_feature_catalog: ProfilerFeatureCatalog | None = None,
    fit_cache_directory: Path | None = None,
    paired_development_comparisons: dict[
        PairedCellKey, tuple[PairedTimingComparison, ...]
    ] | None = None,
    supplemental_development_costs: dict[
        GenericDomain, tuple[CandidatePointCost, ...]
    ] | None = None,
    burned_seal_evidence_digests: tuple[str, ...] = (),
    max_leaves: int = DEFAULT_TREE_LEAVES,
    minimum_promotion_warmups: int = MIN_PROMOTION_WARMUPS,
    minimum_promotion_samples: int = MIN_PROMOTION_SAMPLES,
) -> FrozenPolicy:
    """Fit grouped-verifier rules without receiving any sealed observations."""

    corpus = _require_partition(
        development,
        manifest,
        ShapePartition.DEVELOPMENT,
    )
    fit_cache = (
        PolicyFitCache(directory=fit_cache_directory)
        if fit_cache_directory is not None
        else None
    )
    return freeze_policy(
        corpus,
        sealed_commitment=cpu_grouped_sealed_reserve_commitment(manifest),
        split_manifest_digest=manifest.digest(),
        serial_m1_hashes=_serial_hashes(corpus, serial_m1_policy_hash),
        paired_development_comparisons=paired_development_comparisons,
        supplemental_development_costs=supplemental_development_costs,
        profiler_feature_catalog=profiler_feature_catalog,
        fit_cache=fit_cache,
        max_leaves=max_leaves,
        minimum_promotion_warmups=minimum_promotion_warmups,
        minimum_promotion_samples=minimum_promotion_samples,
        metadata={
            "backend": "cpu",
            "semantic_contract": SemanticContract.VERIFIER_SERIAL_M1_BITWISE.value,
            "shape_manifest_schema": manifest.schema_version,
            "shape_manifest_digest": manifest.digest(),
            "split_surface": "cpu_verifier",
            "paired_development_evidence_digest": paired_comparison_digest(
                paired_development_comparisons or {}
            ),
            "burned_seal_development_evidence_digests": list(
                burned_seal_evidence_digests
            ),
            "generic_max_leaves": max_leaves,
        },
    )


def certify_cpu_verifier_policy(
    frozen: FrozenPolicy,
    development: ObservationCorpus,
    sealed_plan_path: Path,
    sealed_paired_csvs: tuple[Path, ...],
    manifest: NativeVNNIShapeManifest,
    sealed_build_id: str,
    supplemental_dimensions_by_group: dict[str, tuple[int, int]] | None = None,
) -> CompiledPolicy:
    """Certify frozen grouped leaves with fresh byte-proven paired evidence."""

    development = _require_partition(
        development,
        manifest,
        ShapePartition.DEVELOPMENT,
    )
    plan = read_cpu_grouped_sealed_plan(sealed_plan_path)
    validate_cpu_grouped_sealed_plan(
        plan,
        frozen.policy_ir.generic_rules,
        development,
        manifest,
        sealed_build_id,
        supplemental_dimensions_by_group,
    )
    certification = certify_cpu_grouped_sealed_pairs(
        frozen.policy_ir.generic_rules,
        frozen.generic_digest,
        plan,
        sealed_paired_csvs,
    )
    return finalize_frozen_policy_certificate(
        frozen,
        development,
        certification,
        sealed_evidence_digest=paired_evidence_digest(
            plan.plan_digest, sealed_paired_csvs
        ),
    )


def _load_burned_seal_development(
    development: ObservationCorpus,
    plan_paths: tuple[Path, ...],
    evidence_directories: tuple[Path, ...],
) -> tuple[
    dict[GenericDomain, tuple[CandidatePointCost, ...]],
    tuple[str, ...],
    dict[str, tuple[int, int]],
]:
    """Load inspected grouped seals as generic-only development evidence."""

    return load_cpu_burned_seal_development(
        development,
        plan_paths,
        evidence_directories,
        surface_name="grouped decode",
        read_plan=read_cpu_grouped_sealed_plan,
        build_costs=cpu_grouped_burned_seal_costs,
    )


def _load_reused_development_common(
    path: Path,
    context: CPUVerifierAdapterContext,
) -> ObservationCorpus:
    """Authenticate one grouped adaptation from the exact raw transaction.

    The adapt-only phase binds every common row to the aggregate/timing digest
    and complete CPU build/runtime provenance. Freeze and certification may
    reuse that immutable result instead of converting the same large CSVs
    again, but only after every row proves it belongs to this transaction and
    ISA-specific build identity.
    """

    context.validate()
    corpus = read_observation_csv((path,))
    for row in corpus:
        context.validate_promotion_timing(
            row.warmup_count,
            row.sample_count,
            forced_route_ok=row.forced_route_ok,
        )
        build_isa, _runtime_isa, _threads = _cpu_runtime_surface(
            row.architecture_class
        )
        architecture_prefix = row.architecture_class.rsplit("|", 3)[0]
        expected = {
            "run_id": context.run_id,
            "corpus_id": context.corpus_id,
            "git_revision": context.git_revision,
            "build_id": f"{context.build_id}|cpu_isa={build_isa}",
            "compiler_id": context.compiler_id,
            "architecture_class": context.architecture_class,
            "device_name": context.device_name,
            "driver_runtime": context.driver_runtime,
            "serial_m1_policy_hash": context.serial_m1_policy_hash,
        }
        observed = {
            "run_id": row.run_id,
            "corpus_id": row.corpus_id,
            "git_revision": row.git_revision,
            "build_id": row.build_id,
            "compiler_id": row.compiler_id,
            "architecture_class": architecture_prefix,
            "device_name": row.device_name,
            "driver_runtime": row.driver_runtime,
            "serial_m1_policy_hash": row.serial_m1_policy_hash,
        }
        if observed != expected:
            changed = sorted(
                name for name in expected if observed[name] != expected[name]
            )
            raise ValueError(
                "reused CPU grouped common corpus changed raw/build "
                f"provenance fields {changed}"
            )
    return corpus


def _load_profiler_catalog(
    corpus: ObservationCorpus,
    args: argparse.Namespace,
) -> ProfilerFeatureCatalog:
    """Authenticate and cache exact grouped profiler model descriptors."""

    return load_profiler_feature_catalog(
        corpus,
        args.development_profiler_requests,
        args.development_profiler_evidence,
        source_corpus_path=args.development_profiler_observations,
        cache_path=(
            args.fit_cache_dir / "profiler_feature_catalog_v1.json"
            if args.fit_cache_dir is not None else None
        ),
    )


def validate_emitter_inputs(
    policy_ir: PolicyIR,
    entries: list[PolicyEntry],
    generic_rules: list[CPUGenericDispatchRule],
) -> None:
    """Prove that backend emission consumes the exact certified common IR."""

    # Compare architecture surfaces in their decoded representation so the C++
    # emitter cannot accidentally collapse AVX2-build and AVX512-dispatch rows.
    decoded_ir_exact = {
        (
            *_cpu_runtime_surface(entry.key.architecture_class),
            entry.key.runtime_codebook_id,
            entry.key.m,
            entry.key.aggregate_n,
            entry.key.k,
        ): entry.candidate_id
        for entry in policy_ir.exact_entries
    }
    decoded_emitted_exact = {
        (
            entry.build_isa,
            entry.runtime_isa,
            entry.threads,
            entry.codebook,
            entry.m,
            entry.n,
            entry.k,
        ): entry.candidate_id
        for entry in entries
    }
    if decoded_ir_exact != decoded_emitted_exact:
        raise ValueError("CPU verifier emitter exact entries diverge from common IR")
    if tuple(item.rule for item in generic_rules) != policy_ir.generic_rules:
        raise ValueError("CPU verifier emitter generic rules diverge from common IR")


def validate_isa_regime_matrix(corpus: ObservationCorpus) -> None:
    """Require every supported CPU build/runtime dispatch combination."""

    observed_regimes = {
        _cpu_runtime_surface(key.architecture_class)[:2]
        for key in corpus.runtime_keys()
    }
    required_regimes = {
        ("AVX2", "AVX2"),
        ("AVX512", "AVX2"),
        ("AVX512", "AVX512"),
    }
    missing_regimes = sorted(required_regimes.difference(observed_regimes))
    if missing_regimes:
        rendered = ", ".join(
            f"{build}/{runtime}" for build, runtime in missing_regimes
        )
        raise ValueError(f"CPU ISA regime matrix incomplete: {rendered}")


def validate_complete(
    corpus: ObservationCorpus,
    manifest: NativeVNNIShapeManifest,
    *,
    require_full_inventory: bool,
    expected_partition: ShapePartition | None = None,
) -> None:
    """Validate the grouped verifier matrix and reviewed shape applicability."""

    require_canonical_alias_coverage(corpus)
    require_verifier_m_matrix(corpus)
    require_candidate_matrix_complete(corpus)
    require_registry_candidate_coverage(
        corpus,
        cpu_native_vnni_verifier_registry(),
    )
    observed_names = {row.shape_name for row in corpus}
    disallowed = sorted(
        name
        for name in observed_names
        if manifest.by_name(name).verifier_partition is None
    )
    if disallowed:
        raise ValueError(
            "CPU verifier corpus contains Fast-only shapes: "
            f"{disallowed[:3]}"
        )
    if require_full_inventory:
        expected_names = set(manifest.cpu_measurement_names(
            verifier=True,
            partition=expected_partition,
        ))
        if observed_names != expected_names:
            raise ValueError(
                "CPU production verifier shape inventory is incomplete: "
                f"missing={sorted(expected_names - observed_names)[:3]} "
                f"unexpected={sorted(observed_names - expected_names)[:3]}"
            )
    validate_isa_regime_matrix(corpus)


def _parse_required_key(raw: str) -> tuple[int, int, int, int, str]:
    parts = raw.split(":")
    if len(parts) != 4:
        raise ValueError(f"invalid --require-key {raw!r}; expected FORMAT:M:N:K")
    from native_vnni_dispatch.format_registry import format_spec

    spec = format_spec(parts[0])
    m, n, k = (int(value, 0) for value in parts[1:])
    if m < 2 or n <= 0 or k <= 0:
        raise ValueError(f"invalid --require-key dimensions in {raw!r}")
    return spec.runtime_codebook("cpu"), m, n, k, spec.label


def validate_required_keys(entries: list[PolicyEntry], required: list[str]) -> None:
    present = {
        (row.build_isa, row.runtime_isa, row.codebook, row.m, row.n, row.k)
        for row in entries
    }
    regimes = {
        ("AVX2", "AVX2"),
        ("AVX512", "AVX2"),
        ("AVX512", "AVX512"),
    }
    missing = []
    for raw in required:
        codebook, m, n, k, label = _parse_required_key(raw)
        for build_isa, runtime_isa in sorted(regimes):
            if (build_isa, runtime_isa, codebook, m, n, k) not in present:
                missing.append(
                    f"{build_isa}/{runtime_isa}:{label}:M{m}:N{n}:K{k}"
                )
    if missing:
        raise ValueError("missing required CPU verifier policy row(s): " + ", ".join(missing))


def required_keys_from_inventory(
    formats_csv: str,
    shapes_csv: str,
    m_values_csv: str,
    manifest: NativeVNNIShapeManifest,
) -> list[str]:
    """Expand one compact exact-overlay inventory inside the certifier.

    Keeping the three independent dimensions on the command line avoids an
    ``ARG_MAX`` failure when their Cartesian product contains hundreds of
    thousands of required policy keys. Duplicate geometries and runtime-format
    aliases remain harmless and are deduplicated before validation.
    """

    supplied = tuple(bool(value.strip()) for value in (
        formats_csv, shapes_csv, m_values_csv
    ))
    if any(supplied) and not all(supplied):
        raise ValueError(
            "required CPU verifier inventory needs formats, shapes, and M values"
        )
    if not any(supplied):
        return []

    formats = tuple(value for value in formats_csv.split(",") if value)
    shape_names = tuple(value for value in shapes_csv.split(",") if value)
    try:
        m_values = tuple(
            int(value, 0) for value in m_values_csv.split(",") if value
        )
    except ValueError as error:
        raise ValueError("required CPU verifier inventory has invalid M") from error
    if not formats or not shape_names or not m_values:
        raise ValueError("required CPU verifier inventory contains an empty axis")

    keys = set()
    for format_name in formats:
        # Reuse the normal key parser to authenticate every format label.
        _parse_required_key(f"{format_name}:2:1:32")
        for shape_name in shape_names:
            shape = manifest.by_name(shape_name)
            for m in m_values:
                if m >= 2:
                    keys.add(f"{format_name}:{m}:{shape.n}:{shape.k}")
    return sorted(keys)


def _exact_table_identifier(build_isa: str, runtime_isa: str, threads: int) -> str:
    """Return the stable C++ suffix for one exact-overlay runtime surface."""

    return f"{build_isa}{runtime_isa}T{threads}"


def _append_uint64_initializer(
    lines: list[str],
    values: list[int],
    *,
    values_per_line: int = 4,
) -> None:
    """Append a compact, reviewable hexadecimal initializer to ``lines``."""

    for begin in range(0, len(values), values_per_line):
        rendered = ", ".join(
            f"0x{value:016x}ULL"
            for value in values[begin : begin + values_per_line]
        )
        lines.append(f"    {rendered},")


def _append_uint8_initializer(
    lines: list[str],
    values: list[int],
    *,
    values_per_line: int = 16,
) -> None:
    """Append one compact byte-valued generated policy table."""

    for begin in range(0, len(values), values_per_line):
        rendered = ", ".join(
            str(value) for value in values[begin : begin + values_per_line]
        )
        lines.append(f"    {rendered},")


def generate_include(
    entries: list[PolicyEntry],
    generic_rules: list[CPUGenericDispatchRule],
    *,
    corpus_digest: str,
    registry_digest: str,
    profile: MeasurementProfile,
    minimum_promotion_warmups: int = MIN_PROMOTION_WARMUPS,
    minimum_promotion_samples: int = MIN_PROMOTION_SAMPLES,
    policy_digest: str = "",
    certification: CertificationReport | None = None,
) -> str:
    """Render exact/generic decisions into an ISA-aware CPU selector."""

    if certification is None:
        decision_comment = (
            "// Decisions: common NativeVNNI alias-robust exact oracle plus "
            "development-only generic policy; this artifact is not installable "
            "without sealed certification."
        )
    else:
        decision_comment = (
            "// Decisions: common NativeVNNI alias-robust exact oracle plus "
            "frozen development policy with generic-only sealed certification."
        )
    lines = [
        "// Auto-generated by analyze_cpu_native_vnni_verifier_trainer.py. DO NOT EDIT.",
        decision_comment,
        f"// Measurement profile: {profile.value}",
        "// Installable timing evidence floor: "
        f"{minimum_promotion_warmups} warmup(s), "
        f"{minimum_promotion_samples} sample(s)",
        f"// Common corpus digest: {corpus_digest}",
        f"// Candidate registry digest: {registry_digest}",
        "#pragma once",
        "#define LLAMINAR_CPU_NVNNI_VERIFIER_POLICY_ABI 3",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace llaminar2::cpu::native_vnni::generated",
        "{",
        "enum class CPUNativeVNNIVerifierRowsPolicy : uint8_t",
        "{",
        *(
            f"    {policy} = {ordinal},"
            for ordinal, policy in enumerate(CPU_VERIFIER_POLICIES)
        ),
        "};",
        "",
        "enum class CPUNativeVNNIBuildISA : uint8_t",
        "{",
        "    AVX2 = 0,",
        "    AVX512 = 1,",
        "};",
        "",
        "enum class CPUNativeVNNIRuntimeISA : uint8_t",
        "{",
        "    AVX2 = 0,",
        "    AVX512 = 1,",
        "};",
        "",
        "inline constexpr uint64_t packCPUNativeVNNIVerifierRowsPolicyKey(",
        "    uint8_t codebook, int m, int n, int k)",
        "{",
        "    return (static_cast<uint64_t>(codebook) << 56) |",
        "           (static_cast<uint64_t>(m & 0xFF) << 48) |",
        "           (static_cast<uint64_t>(k & 0xFFFFFF) << 24) |",
        "           static_cast<uint64_t>(n & 0xFFFFFF);",
        "}",
        "",
        "/** Map every positive grouped runtime M onto measured policy support. */",
        "inline constexpr int cpuNativeVNNIVerifierPolicyM(int m)",
        "{",
        "    if (m < 2)",
        "        return 0;",
        "    return m <= 16 ? m : 31;",
        "}",
    ]
    if certification is not None:
        lines[5:5] = [
            f"// Common policy digest: {policy_digest}",
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

    group_keys = sorted({
        (entry.build_isa, entry.runtime_isa, entry.threads)
        for entry in entries
    } | {
        (item.build_isa, item.runtime_isa, item.threads)
        for item in generic_rules
    })
    entries_by_group = {
        group: sorted(
            (
                entry
                for entry in entries
                if (entry.build_isa, entry.runtime_isa, entry.threads) == group
            ),
            key=lambda entry: _pack_key(
                entry.codebook, entry.m, entry.n, entry.k
            ),
        )
        for group in group_keys
    }
    rules_by_group = {
        group: [
            item
            for item in generic_rules
            if (item.build_isa, item.runtime_isa, item.threads) == group
        ]
        for group in group_keys
    }

    for build_isa, runtime_isa, threads in group_keys:
        group = (build_isa, runtime_isa, threads)
        group_entries = entries_by_group[group]
        if not group_entries:
            continue
        packed_keys = [
            _pack_key(entry.codebook, entry.m, entry.n, entry.k)
            for entry in group_entries
        ]
        if len(set(packed_keys)) != len(packed_keys):
            raise ValueError(
                "CPU verifier exact overlays contain duplicate packed keys for "
                f"{build_isa}/{runtime_isa}/T{threads}"
            )
        policy_values = []
        for entry in group_entries:
            try:
                policy_values.append(CPU_VERIFIER_POLICY_ORDINAL[entry.policy])
            except KeyError as error:
                raise ValueError(
                    f"unknown exact CPU verifier policy {entry.policy!r}"
                ) from error

        identifier = _exact_table_identifier(build_isa, runtime_isa, threads)
        lines.extend([
            "",
            "/**",
            f" * Sorted exact-overlay keys for {build_isa} build, "
            f"{runtime_isa} runtime, and {threads} worker threads.",
            " *",
            " * Entry i owns the byte-valued policy at index i in the adjacent",
            " * array. Keeping keys and policy bytes separate retains compact",
            " * binary search while supporting every registered grouped route.",
            " */",
            f"inline constexpr uint64_t kCPUNativeVNNIVerifierExact{identifier}Keys[] =",
            "{",
        ])
        _append_uint64_initializer(lines, packed_keys)
        lines.extend([
            "};",
            f"inline constexpr uint8_t kCPUNativeVNNIVerifierExact{identifier}Policies[] =",
            "{",
        ])
        _append_uint8_initializer(lines, policy_values)
        lines.append("};")

    lines.extend([
        "",
        "/**",
        " * Resolve one exact overlay without constructing a giant C++ switch.",
        " *",
        " * The generator sorts every key and emits one compact policy byte.",
        " * Binary search keeps lookup logarithmic while dramatically reducing",
        " * compiler memory, object size, and generated source volume.",
        " */",
        "template <std::size_t KeyCount, std::size_t PolicyCount>",
        "inline bool selectCPUNativeVNNIVerifierRowsExactPolicy(",
        "    const uint64_t (&keys)[KeyCount],",
        "    const uint8_t (&policies)[PolicyCount],",
        "    uint64_t key, CPUNativeVNNIVerifierRowsPolicy &policy)",
        "{",
        "    static_assert(PolicyCount == KeyCount);",
        "    std::size_t first = 0;",
        "    std::size_t last = KeyCount;",
        "    while (first < last)",
        "    {",
        "        const std::size_t middle = first + (last - first) / 2U;",
        "        if (keys[middle] < key)",
        "            first = middle + 1U;",
        "        else",
        "            last = middle;",
        "    }",
        "    if (first == KeyCount || keys[first] != key)",
        "        return false;",
        f"    if (policies[first] > {len(CPU_VERIFIER_POLICIES) - 1}U)",
        "        return false;",
        "    policy = static_cast<CPUNativeVNNIVerifierRowsPolicy>(policies[first]);",
        "    return true;",
        "}",
        "",
        "inline bool selectCPUNativeVNNIVerifierRowsGeneratedPolicy(",
        "    CPUNativeVNNIBuildISA build_isa,",
        "    CPUNativeVNNIRuntimeISA runtime_isa, int threads,",
        "    uint8_t codebook, int m, int n, int k, int k_tiles,",
        "    CPUNativeVNNIVerifierRowsPolicy &policy, float *measured_speedup = nullptr)",
        "{",
        "    const int policy_m = cpuNativeVNNIVerifierPolicyM(m);",
        "    if (policy_m == 0 || n <= 0 || k <= 0 || k_tiles < 0)",
        "        return false;",
        "    const uint64_t key =",
        "        packCPUNativeVNNIVerifierRowsPolicyKey(codebook, m, n, k);",
    ])
    for build_isa, runtime_isa, threads in group_keys:
        group = (build_isa, runtime_isa, threads)
        group_entries = entries_by_group[group]
        group_rules = rules_by_group[group]
        lines.extend([
            f"    if (build_isa == CPUNativeVNNIBuildISA::{build_isa} &&",
            f"        runtime_isa == CPUNativeVNNIRuntimeISA::{runtime_isa} &&",
            f"        threads == {threads})",
            "    {",
        ])
        if group_entries:
            identifier = _exact_table_identifier(
                build_isa, runtime_isa, threads
            )
            lines.extend([
                "        if (m <= 255 && selectCPUNativeVNNIVerifierRowsExactPolicy(",
                f"                kCPUNativeVNNIVerifierExact{identifier}Keys,",
                f"                kCPUNativeVNNIVerifierExact{identifier}Policies,",
                "                key, policy))",
                "        {",
                "            if (measured_speedup)",
                "                *measured_speedup = 0.0f;",
                "            return true;",
                "        }",
            ])
        if group_rules:
            lines.extend([
                "        const long long work_items =",
                "            static_cast<long long>(n) * "
                "static_cast<long long>(k);",
            ])
            for item in sorted(
                group_rules,
                key=lambda value: (
                    value.rule.domain.runtime_codebook_id,
                    value.rule.domain.m,
                    generic_rule_sort_key(value.rule),
                ),
            ):
                rule = item.rule
                conditions = [
                    f"codebook == {rule.domain.runtime_codebook_id}",
                    f"policy_m == {rule.domain.m}",
                    aspect_condition(rule.domain.aspect_bucket),
                    *(
                        predicate_condition(
                            predicate,
                            k_tiles_expression="k_tiles",
                        )
                        for predicate in rule.predicates
                    ),
                ]
                lines.extend(render_if_header(conditions, indent="        "))
                lines.extend([
                    "        {",
                    "            policy = "
                    "CPUNativeVNNIVerifierRowsPolicy::"
                    f"{_candidate_policy(rule.candidate_id)};",
                    "            if (measured_speedup)",
                    "                *measured_speedup = 0.0f;",
                    "            return true;",
                    "        }",
                ])
        lines.append("    }")
    lines.extend([
        "    return false;",
        "}",
        "",
        "} // namespace llaminar2::cpu::native_vnni::generated",
        "",
    ])
    return "\n".join(lines)


def write_summary(path: Path, entries: list[PolicyEntry]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "build_isa", "runtime_isa", "threads", "execution_codebook",
            "m", "n", "k", "policy", "candidate_id", "shape_names",
            "max_surface_regret", "max_cv",
        ])
        for entry in sorted(entries):
            writer.writerow([
                entry.build_isa, entry.runtime_isa, entry.threads,
                entry.codebook, entry.m, entry.n, entry.k, entry.policy,
                entry.candidate_id, ";".join(entry.shape_names),
                f"{entry.max_surface_regret:.9f}", f"{entry.max_cv:.9f}",
            ])


def _context_from_args(
    args: argparse.Namespace,
    inputs: tuple[Path, ...],
    timing_sidecars: tuple[Path, ...],
    *,
    run_id: str | None = None,
) -> CPUVerifierAdapterContext:
    corpus_id = raw_corpus_id((*inputs, *timing_sidecars))
    profile = MeasurementProfile(args.profile)
    if not profile.installable:
        return CPUVerifierAdapterContext.workflow_smoke(
            corpus_id=corpus_id,
            minimum_promotion_warmups=args.minimum_promotion_warmups,
            minimum_promotion_samples=args.minimum_promotion_samples,
        )
    return CPUVerifierAdapterContext(
        profile=profile,
        run_id=run_id or args.run_id,
        corpus_id=corpus_id,
        git_revision=args.git_revision,
        build_id=args.build_id,
        compiler_id=args.compiler_id,
        architecture_class=args.architecture_class,
        device_name=args.device_name,
        driver_runtime=args.driver_runtime,
        serial_m1_policy_hash=args.serial_m1_policy_hash,
        raw_timing_sidecar_retained=bool(timing_sidecars),
        minimum_promotion_warmups=args.minimum_promotion_warmups,
        minimum_promotion_samples=args.minimum_promotion_samples,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--input", nargs="+", type=Path, dest="input_options")
    parser.add_argument("--timing-sidecar", action="append", type=Path, default=[])
    parser.add_argument("--development-input", action="append", type=Path, default=[])
    parser.add_argument(
        "--development-timing-sidecar", action="append", type=Path, default=[]
    )
    parser.add_argument("--freeze-generic", action="store_true")
    parser.add_argument("--certify-generic", action="store_true")
    parser.add_argument("--adapt-only", action="store_true")
    parser.add_argument("--frozen-policy-json", type=Path)
    parser.add_argument("--policy-json", type=Path)
    parser.add_argument("--sealed-plan-json", type=Path)
    parser.add_argument("--sealed-request-dir", type=Path)
    parser.add_argument(
        "--sealed-paired-csv", action="append", type=Path, default=[]
    )
    parser.add_argument(
        "--sealed-paired-dir", action="append", type=Path, default=[]
    )
    parser.add_argument(
        "--certification-diagnostic",
        type=Path,
        help="Write the complete non-installable grouped seal report",
    )
    parser.add_argument(
        "--certification-diagnostic-only",
        action="store_true",
        help="Write the grouped seal diagnostic without publishing policy output",
    )
    parser.add_argument("--development-profiler-requests", type=Path)
    parser.add_argument("--development-profiler-evidence", type=Path)
    parser.add_argument(
        "--development-profiler-observations",
        type=Path,
        help="Original common CSV bound to reusable profiler sidecars",
    )
    parser.add_argument(
        "--paired-development-csv",
        action="append",
        type=Path,
        default=[],
        help="Retained grouped paired-development evidence shard",
    )
    parser.add_argument(
        "--burned-sealed-plan-json",
        action="append",
        type=Path,
        default=[],
        help="Inspected prior grouped seal promoted to generic development",
    )
    parser.add_argument(
        "--burned-sealed-paired-dir",
        action="append",
        type=Path,
        default=[],
        help="Paired CSV directory matched positionally to a burned plan",
    )
    parser.add_argument(
        "--fit-cache-dir",
        type=Path,
        help="Persistent content-addressed candidate-cost and CV cache",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", "--summary-csv", dest="summary", type=Path)
    parser.add_argument("--common-observations", type=Path)
    parser.add_argument(
        "--reuse-development-common",
        action="store_true",
        help=(
            "Authenticate and reuse --common-observations for grouped "
            "development instead of repeating raw timing adaptation"
        ),
    )
    parser.add_argument("--require-key", action="append", default=[])
    parser.add_argument("--require-inventory-formats", default="")
    parser.add_argument("--require-inventory-shapes", default="")
    parser.add_argument("--require-inventory-m-values", default="")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--require-isa-matrix", action="store_true")
    parser.add_argument(
        "--profile", choices=[profile.value for profile in MeasurementProfile],
        default=MeasurementProfile.QUICK.value,
    )
    parser.add_argument(
        "--minimum-promotion-warmups",
        type=int,
        default=MIN_PROMOTION_WARMUPS,
        help="Minimum warmups required for installable fixed-timing evidence",
    )
    parser.add_argument(
        "--minimum-promotion-samples",
        type=int,
        default=MIN_PROMOTION_SAMPLES,
        help="Minimum samples required for installable fixed-timing evidence",
    )
    parser.add_argument(
        "--generic-max-leaves",
        type=int,
        default=DEFAULT_TREE_LEAVES,
        help=(
            "Maximum grouped generic-tree leaves. The default is "
            f"{DEFAULT_TREE_LEAVES}; an explicit best-effort transaction may "
            "reduce this while retaining total dispatch and every correctness "
            "gate."
        ),
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument("--git-revision", default="")
    parser.add_argument("--build-id", default="")
    parser.add_argument("--compiler-id", default="")
    parser.add_argument("--architecture-class", default="")
    parser.add_argument("--device-name", default="")
    parser.add_argument("--driver-runtime", default="")
    parser.add_argument("--serial-m1-policy-hash", default="")
    parser.add_argument(
        "--sealed-build-id",
        default="",
        help=(
            "Combined AVX2/AVX512 trainer digest that physically executes "
            "the post-freeze grouped paired seal"
        ),
    )
    parser.add_argument(
        "--shape-manifest",
        type=Path,
        default=MANIFEST_PATH,
        help="Reviewed grouped-verifier shape applicability",
    )
    args = parser.parse_args()
    sealed_paired_csvs: tuple[Path, ...] = ()
    if args.certify_generic:
        try:
            sealed_paired_csvs = resolve_sealed_paired_evidence_paths(
                args.sealed_paired_csv,
                args.sealed_paired_dir,
            )
        except ValueError as error:
            parser.error(str(error))

    if not 1 <= args.generic_max_leaves <= MAX_TREE_LEAVES:
        parser.error(
            f"--generic-max-leaves must be in [1, {MAX_TREE_LEAVES}]"
        )

    positional = tuple(args.inputs)
    optional = tuple(args.input_options or ())
    if positional and optional:
        parser.error("use positional inputs or --input, not both")
    inputs = positional or optional
    if args.freeze_generic and args.certify_generic:
        parser.error("--freeze-generic and --certify-generic are mutually exclusive")
    if args.certification_diagnostic and not args.certify_generic:
        parser.error("--certification-diagnostic requires --certify-generic")
    if args.certification_diagnostic_only and not args.certification_diagnostic:
        parser.error(
            "--certification-diagnostic-only requires "
            "--certification-diagnostic"
        )
    if args.adapt_only and (args.freeze_generic or args.certify_generic):
        parser.error("--adapt-only cannot freeze or certify a policy")
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
    if args.paired_development_csv and not (
        args.freeze_generic or args.certify_generic
    ):
        parser.error(
            "--paired-development-csv requires generic freeze/certification"
        )
    if len(args.burned_sealed_plan_json) != len(
        args.burned_sealed_paired_dir
    ):
        parser.error(
            "burned grouped plans and paired directories must pair by position"
        )
    if args.burned_sealed_plan_json and not (
        args.freeze_generic or args.certify_generic
    ):
        parser.error(
            "burned grouped development requires freeze or certification"
        )
    if args.reuse_development_common and not (
        args.freeze_generic or args.certify_generic
    ):
        parser.error(
            "--reuse-development-common requires freeze or certification"
        )
    if args.reuse_development_common and args.common_observations is None:
        parser.error(
            "--reuse-development-common requires --common-observations"
        )
    if (args.freeze_generic or args.certify_generic) and not (
        args.development_profiler_requests
        and args.development_profiler_evidence
    ):
        parser.error(
            "production CPU freeze/certification requires complete "
            "development profiler evidence"
        )
    if (
        args.freeze_generic or args.certify_generic
    ) and not args.sealed_build_id.startswith("sha256:"):
        parser.error(
            "CPU grouped freeze/certification requires --sealed-build-id"
        )
    separate_certification = bool(
        args.development_input
        or args.development_timing_sidecar
        or args.frozen_policy_json
    )
    if args.freeze_generic:
        if not inputs or args.development_input:
            parser.error("--freeze-generic accepts development --input only")
        if not args.policy_json:
            parser.error("--freeze-generic requires --policy-json")
        if not args.sealed_plan_json or not args.sealed_request_dir:
            parser.error(
                "--freeze-generic requires sealed plan and request directory"
            )
    elif args.certify_generic:
        if inputs or args.timing_sidecar:
            parser.error("--certify-generic accepts separate partition inputs only")
        if not args.development_input:
            parser.error("certification requires development inputs")
        if not (
            args.frozen_policy_json
            and args.policy_json
            and args.sealed_plan_json
            and sealed_paired_csvs
        ):
            parser.error(
                "certification requires frozen policy, compiled policy output, "
                "sealed plan, and paired CSV evidence"
            )
    elif separate_certification:
        parser.error("separate partition inputs require --certify-generic")
    elif not inputs:
        parser.error("at least one strong CPU trainer CSV is required")
    if (
        MeasurementProfile(args.profile) == MeasurementProfile.PRODUCTION
        and not args.freeze_generic
        and not args.certify_generic
        and not args.adapt_only
    ):
        parser.error(
            "production CPU policy emission requires development freeze and "
            "separate sealed certification"
        )

    manifest = load_shape_manifest(args.shape_manifest)
    try:
        required_keys = [
            *args.require_key,
            *required_keys_from_inventory(
                args.require_inventory_formats,
                args.require_inventory_shapes,
                args.require_inventory_m_values,
                manifest,
            ),
        ]
    except (KeyError, ValueError) as error:
        parser.error(str(error))
    paired_cells = []
    for paired_path in args.paired_development_csv:
        paired_cells.extend(read_paired_confirmation_csv((paired_path,)))
    paired_development_comparisons = (
        paired_timing_comparisons(paired_cells) if paired_cells else {}
    )
    if args.adapt_only:
        if not inputs or separate_certification:
            parser.error("--adapt-only accepts development --input only")
        timing_sidecars = tuple(args.timing_sidecar)
        context = _context_from_args(args, inputs, timing_sidecars)
        corpus = (
            adapt_cpu_verifier_csv_to_common(
                inputs,
                context,
                args.common_observations,
                timing_sidecars=timing_sidecars,
            )
            if args.common_observations is not None
            else adapt_cpu_verifier_csv(
                inputs,
                context,
                timing_sidecars=timing_sidecars,
            )
        )
        validate_complete(
            corpus,
            manifest,
            require_full_inventory=(
                context.profile == MeasurementProfile.PRODUCTION
            ),
            expected_partition=ShapePartition.DEVELOPMENT,
        )
        entries = select_entries(corpus, context.serial_m1_policy_hash)
        validate_required_keys(entries, required_keys)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(generate_include(
            entries,
            [],
            corpus_digest=corpus.digest(),
            registry_digest=candidate_registry_digest(),
            profile=context.profile,
            minimum_promotion_warmups=context.minimum_promotion_warmups,
            minimum_promotion_samples=context.minimum_promotion_samples,
        ), encoding="utf-8")
        print(
            f"adapted {len(corpus)} CPU verifier development observations "
            f"without fitting -> {args.output}"
        )
        return 0

    if args.freeze_generic:
        timing_sidecars = tuple(args.timing_sidecar)
        context = _context_from_args(args, inputs, timing_sidecars)
        corpus = (
            _load_reused_development_common(
                args.common_observations,
                context,
            )
            if args.reuse_development_common
            else adapt_cpu_verifier_csv(
                inputs,
                context,
                timing_sidecars=timing_sidecars,
            )
        )
        profiler_catalog = _load_profiler_catalog(corpus, args)
        supplemental_costs, burned_digests, supplemental_dimensions = (
            _load_burned_seal_development(
                corpus,
                tuple(args.burned_sealed_plan_json),
                tuple(args.burned_sealed_paired_dir),
            )
        )
        frozen = freeze_cpu_verifier_policy(
            corpus,
            manifest,
            context.serial_m1_policy_hash,
            profiler_catalog,
            args.fit_cache_dir,
            paired_development_comparisons,
            supplemental_costs,
            burned_digests,
            args.generic_max_leaves,
            context.minimum_promotion_warmups,
            context.minimum_promotion_samples,
        )
        entries = select_entries(corpus, context.serial_m1_policy_hash)
        generic_rules = _emit_generic_rules(frozen.policy_ir.generic_rules)
        validate_emitter_inputs(frozen.policy_ir, entries, generic_rules)
        validate_required_keys(entries, required_keys)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(generate_include(
            entries,
            generic_rules,
            corpus_digest=corpus.digest(),
            registry_digest=candidate_registry_digest(),
            profile=context.profile,
            minimum_promotion_warmups=context.minimum_promotion_warmups,
            minimum_promotion_samples=context.minimum_promotion_samples,
            policy_digest=frozen.policy_ir.digest(),
        ), encoding="utf-8")
        write_frozen_policy(args.policy_json, frozen)
        plan = build_cpu_grouped_sealed_plan(
            frozen.policy_ir.generic_rules,
            frozen.generic_digest,
            corpus,
            manifest,
            args.sealed_build_id,
            supplemental_dimensions,
        )
        validate_cpu_grouped_sealed_plan(
            plan,
            frozen.policy_ir.generic_rules,
            corpus,
            manifest,
            args.sealed_build_id,
            supplemental_dimensions,
        )
        shards = write_cpu_grouped_sealed_plan(
            args.sealed_plan_json,
            plan,
            args.sealed_request_dir,
        )
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            write_summary(args.summary, entries)
        if args.common_observations and not args.reuse_development_common:
            args.common_observations.parent.mkdir(parents=True, exist_ok=True)
            write_observation_csv(args.common_observations, corpus)
        print(
            f"froze {len(corpus)} CPU verifier development observations as "
            f"{frozen.generic_digest}; planned {len(plan.rule_witnesses)} "
            f"fresh leaves as {len(plan.requests)} paired edges in "
            f"{len(shards)} shards -> {args.sealed_plan_json}"
        )
        return 0

    if args.certify_generic:
        development_inputs = tuple(args.development_input)
        development_timing = tuple(args.development_timing_sidecar)
        development_context = _context_from_args(
            args,
            development_inputs,
            development_timing,
        )
        development = (
            _load_reused_development_common(
                args.common_observations,
                development_context,
            )
            if args.reuse_development_common
            else adapt_cpu_verifier_csv(
                development_inputs,
                development_context,
                timing_sidecars=development_timing,
            )
        )
        profiler_catalog = _load_profiler_catalog(development, args)
        supplemental_costs, burned_digests, supplemental_dimensions = (
            _load_burned_seal_development(
                development,
                tuple(args.burned_sealed_plan_json),
                tuple(args.burned_sealed_paired_dir),
            )
        )
        frozen = freeze_cpu_verifier_policy(
            development,
            manifest,
            development_context.serial_m1_policy_hash,
            profiler_catalog,
            args.fit_cache_dir,
            paired_development_comparisons,
            supplemental_costs,
            burned_digests,
            args.generic_max_leaves,
            development_context.minimum_promotion_warmups,
            development_context.minimum_promotion_samples,
        )
        # This byte comparison is deliberately complete before the first
        # sealed CSV or timing sidecar is opened.
        validate_frozen_policy_file(args.frozen_policy_json, frozen)

        compiled = certify_cpu_verifier_policy(
            frozen,
            development,
            args.sealed_plan_json,
            sealed_paired_csvs,
            manifest,
            args.sealed_build_id,
            supplemental_dimensions,
        )
        if args.certification_diagnostic:
            write_certification_diagnostic(
                args.certification_diagnostic,
                compiled,
            )
            if args.certification_diagnostic_only:
                print(
                    "wrote non-installable CPU grouped certification "
                    f"diagnostic -> {args.certification_diagnostic}"
                )
                return 0
        compiled.certification.require_promotable()
        corpus = development
        entries = select_entries(corpus, development_context.serial_m1_policy_hash)
        generic_rules = _emit_generic_rules(compiled.policy_ir.generic_rules)
        validate_emitter_inputs(compiled.policy_ir, entries, generic_rules)
        validate_required_keys(entries, required_keys)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(generate_include(
            entries,
            generic_rules,
            corpus_digest=corpus.digest(),
            registry_digest=candidate_registry_digest(),
            profile=development_context.profile,
            minimum_promotion_warmups=(
                development_context.minimum_promotion_warmups
            ),
            minimum_promotion_samples=(
                development_context.minimum_promotion_samples
            ),
            policy_digest=compiled.policy_ir.digest(),
            certification=compiled.certification,
        ), encoding="utf-8")
        write_compiled_policy(args.policy_json, compiled)
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            write_summary(args.summary, entries)
        if args.common_observations and not args.reuse_development_common:
            args.common_observations.parent.mkdir(parents=True, exist_ok=True)
            write_observation_csv(args.common_observations, corpus)
        print(
            f"certified frozen CPU verifier policy {frozen.generic_digest} "
            f"against {len(sealed_paired_csvs)} paired shard(s) -> "
            f"{args.output}"
        )
        return 0

    timing_sidecars = tuple(args.timing_sidecar)
    context = _context_from_args(args, inputs, timing_sidecars)
    corpus = adapt_cpu_verifier_csv(
        inputs,
        context,
        timing_sidecars=timing_sidecars,
    )
    entries = select_entries(corpus, context.serial_m1_policy_hash)
    generic_rules = select_generic_rules(corpus, context.serial_m1_policy_hash)
    if args.require_complete:
        validate_complete(
            corpus,
            manifest,
            require_full_inventory=(
                args.profile == MeasurementProfile.PRODUCTION.value
            ),
        )
    elif args.require_isa_matrix:
        validate_isa_regime_matrix(corpus)
    validate_required_keys(entries, required_keys)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        generate_include(
            entries,
            generic_rules,
            corpus_digest=corpus.digest(),
            registry_digest=candidate_registry_digest(),
            profile=context.profile,
            minimum_promotion_warmups=context.minimum_promotion_warmups,
            minimum_promotion_samples=context.minimum_promotion_samples,
        ),
        encoding="utf-8",
    )
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        write_summary(args.summary, entries)
    if args.common_observations:
        args.common_observations.parent.mkdir(parents=True, exist_ok=True)
        write_observation_csv(args.common_observations, corpus)
    print(
        f"adapted {len(corpus)} strong CPU observations; generated "
        f"{len(entries)} exact policies -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
