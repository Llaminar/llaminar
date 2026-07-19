#!/usr/bin/env python3
"""Compile strict CPU NativeVNNI M=1 evidence into a sealed selector.

The CPU decode policy owns only physical OpenMP N-chunk task granularity.  It
does not alter the independently frozen full-K/K-part arithmetic schedule, so
every candidate admitted here must already be byte-identical to that serial M=1
oracle.  Development fitting, sealed certification, exact overlays, and
generic rules all use the common NativeVNNI policy core.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path


KERNEL_PERF_ROOT = Path(__file__).resolve().parents[1]
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.adapters.cpu_decode import (  # noqa: E402
    CPUDecodeAdapterContext,
    adapt_cpu_decode_csv,
)
from native_vnni_dispatch.adapters.evidence import raw_corpus_id  # noqa: E402
from native_vnni_dispatch.candidate_observation import (  # noqa: E402
    read_observation_csv,
    write_observation_csv,
)
from native_vnni_dispatch.candidate_registry import (  # noqa: E402
    candidate_registry_digest,
    cpu_native_vnni_decode_registry,
)
from native_vnni_dispatch.certification import CertificationReport  # noqa: E402
from native_vnni_dispatch.compiler import (  # noqa: E402
    CompiledPolicy,
    FrozenPolicy,
    certify_frozen_policy,
    freeze_policy,
)
from native_vnni_dispatch.corpus import ObservationCorpus  # noqa: E402
from native_vnni_dispatch.cpp_predicates import (  # noqa: E402
    aspect_condition,
    generic_rule_sort_key,
    predicate_condition,
    render_if_header,
)
from native_vnni_dispatch.exact_oracle import build_exact_winners  # noqa: E402
from native_vnni_dispatch.format_registry import FORMAT_SPECS  # noqa: E402
from native_vnni_dispatch.policy_artifact import (  # noqa: E402
    validate_frozen_policy_file,
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
from native_vnni_dispatch.profiles import MeasurementProfile  # noqa: E402
from native_vnni_dispatch.profiler_model import (  # noqa: E402
    ProfilerFeatureCatalog,
    load_profiler_feature_catalog,
)
from native_vnni_dispatch.schema import (  # noqa: E402
    AspectBucket,
    ExecutionMode,
    SemanticContract,
)
from native_vnni_dispatch.segmented_policy import (  # noqa: E402
    GenericDispatchRule,
    PolicyFitCache,
    fit_generic_policy,
    validate_generic_rule_partition,
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
)


SERIAL_FULL_K_BUNDLE = "single-native-vnni-decode:serial-full-k:fp32-output:v1"
SERIAL_KPART_BUNDLE = "single-native-vnni-decode:serial-kpart:fp32-output:v1"
DECODE_BUNDLES = (SERIAL_FULL_K_BUNDLE, SERIAL_KPART_BUNDLE)
REQUIRED_ISA_REGIMES = frozenset({
    ("AVX2", "AVX2"),
    ("AVX512", "AVX2"),
    ("AVX512", "AVX512"),
})


@dataclass(frozen=True, order=True)
class DecodePolicyEntry:
    """One exact CPU M=1 decision encoded for the runtime selector ABI."""

    build_isa: str
    runtime_isa: str
    threads: int
    codebook: int
    n: int
    k: int
    serial_kpart: bool
    policy: str
    candidate_id: str
    shape_names: tuple[str, ...]
    max_surface_regret: float
    max_cv: float


@dataclass(frozen=True)
class CPUDecodeGenericRule:
    """One generic common-policy leaf plus its CPU runtime surface."""

    build_isa: str
    runtime_isa: str
    threads: int
    rule: GenericDispatchRule


def _cpu_runtime_surface(architecture_class: str) -> tuple[str, str, int]:
    """Decode adapter-owned build ISA, runtime ISA, and thread width."""

    parts = architecture_class.rsplit("|", 3)
    if len(parts) != 4:
        raise ValueError(
            "CPU architecture class lacks build/runtime/thread identity: "
            f"{architecture_class!r}"
        )
    build = parts[1].removeprefix("build=")
    runtime = parts[2].removeprefix("runtime=")
    thread_text = parts[3].removeprefix("threads=")
    if parts[1] != f"build={build}" or parts[2] != f"runtime={runtime}":
        raise ValueError(f"malformed CPU architecture class {architecture_class!r}")
    if parts[3] != f"threads={thread_text}":
        raise ValueError(f"malformed CPU thread class {architecture_class!r}")
    threads = int(thread_text)
    if (build, runtime) not in REQUIRED_ISA_REGIMES or threads <= 0:
        raise ValueError(f"unsupported CPU decode surface {build}/{runtime}/{threads}")
    return build, runtime, threads


def _candidate_policy(candidate_id: str) -> str:
    """Map one forceable registry candidate to the generated enum spelling."""

    candidate = cpu_native_vnni_decode_registry().resolve(candidate_id)
    n_block_chunks = int(candidate.config_json["n_block_chunks"])
    if n_block_chunks not in {1, 2, 4, 8, 16}:
        raise ValueError(f"unsupported CPU decode NBC width {n_block_chunks}")
    return f"Nbc{n_block_chunks}"


def _bundle_serial_kpart(bundle_signature: str) -> bool:
    """Decode the frozen arithmetic regime represented by one common domain."""

    if bundle_signature == SERIAL_FULL_K_BUNDLE:
        return False
    if bundle_signature == SERIAL_KPART_BUNDLE:
        return True
    raise ValueError(f"unknown CPU decode arithmetic bundle {bundle_signature!r}")


def _pack_key(codebook: int, serial_kpart: bool, n: int, k: int) -> int:
    """Pack the exact overlay key without losing supported dimensions."""

    if codebook < 0 or codebook > 0xFF or n <= 0 or k <= 0:
        raise ValueError("CPU decode exact key has invalid dimensions")
    if n > 0xFFFFFF or k > 0x7FFFFFFF:
        raise ValueError(f"CPU decode exact key exceeds ABI: N={n} K={k}")
    return (
        ((codebook & 0xFF) << 56)
        | ((1 if serial_kpart else 0) << 55)
        | ((k & 0x7FFFFFFF) << 24)
        | (n & 0xFFFFFF)
    )


def select_entries(corpus: ObservationCorpus) -> list[DecodePolicyEntry]:
    """Select exact byte-safe M=1 winners through the common oracle."""

    winners = build_exact_winners(corpus)
    entries = []
    for key, winner in sorted(winners.items()):
        if key.semantic_contract != SemanticContract.FAST or key.m != 1:
            raise ValueError(f"unexpected CPU decode policy key {key}")
        if len(key.projection_n_vector) != 1:
            raise ValueError("CPU decode exact policy supports one projection")
        rows = corpus.rows_for_runtime_key(key)
        build, runtime, threads = _cpu_runtime_surface(key.architecture_class)
        entries.append(DecodePolicyEntry(
            build_isa=build,
            runtime_isa=runtime,
            threads=threads,
            codebook=key.runtime_codebook_id,
            n=key.aggregate_n,
            k=key.k,
            serial_kpart=_bundle_serial_kpart(key.bundle_signature),
            policy=_candidate_policy(winner.candidate_id),
            candidate_id=winner.candidate_id,
            shape_names=tuple(sorted({row.shape_name for row in rows})),
            max_surface_regret=winner.max_surface_regret,
            max_cv=winner.max_cv,
        ))
    return entries


def select_generic_rules(corpus: ObservationCorpus) -> list[CPUDecodeGenericRule]:
    """Fit generic geometry rules with the shared bounded-regret learner."""

    generic = fit_generic_policy(corpus.with_collapsed_aspect_domains())
    return _emit_generic_rules(generic.rules)


def _emit_generic_rules(
    rules: tuple[GenericDispatchRule, ...],
) -> list[CPUDecodeGenericRule]:
    """Attach CPU runtime surfaces to already-frozen common rules."""

    return [
        CPUDecodeGenericRule(
            *_cpu_runtime_surface(rule.domain.architecture_class),
            rule=rule,
        )
        for rule in rules
    ]


def _sealed_commitment(manifest: NativeVNNIShapeManifest) -> str:
    """Commit to untouched bounded CPU Fast-M1 shapes before fitting."""

    payload = {
        "schema": "cpu-native-vnni-decode-fast-m1-split-v1",
        "shape_manifest_digest": manifest.digest(),
        "sealed_shapes": manifest.cpu_measurement_names(
            verifier=False,
            partition=ShapePartition.SEALED,
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def validate_isa_regime_matrix(corpus: ObservationCorpus) -> None:
    """Require all three supported CPU build/runtime dispatch regimes."""

    observed = {
        _cpu_runtime_surface(key.architecture_class)[:2]
        for key in corpus.runtime_keys()
    }
    missing = sorted(REQUIRED_ISA_REGIMES - observed)
    if missing:
        raise ValueError(f"CPU decode ISA regime matrix incomplete: {missing}")


def validate_complete(
    corpus: ObservationCorpus,
    manifest: NativeVNNIShapeManifest,
    *,
    require_full_inventory: bool,
    expected_partition: ShapePartition | None = None,
) -> None:
    """Validate all-format M=1 candidates, geometry, partition, and ISA axes."""

    if any(
        row.semantic_contract != SemanticContract.FAST
        or row.m != 1
        or row.execution_mode != ExecutionMode.EAGER
        for row in corpus
    ):
        raise ValueError("CPU decode corpus contains a non-eager/non-Fast-M1 row")
    require_canonical_alias_coverage(
        corpus,
        required_execution_modes=(ExecutionMode.EAGER,),
    )
    require_candidate_matrix_complete(corpus)
    require_registry_candidate_coverage(
        corpus,
        cpu_native_vnni_decode_registry(),
    )
    observed_names = {row.shape_name for row in corpus}
    disallowed = sorted(
        name for name in observed_names
        if manifest.by_name(name).fast_partition is None
    )
    if disallowed:
        raise ValueError(f"CPU decode corpus contains non-Fast shapes: {disallowed[:3]}")
    for row in corpus:
        shape = manifest.by_name(row.shape_name)
        if (row.aggregate_n, row.k) != (shape.n, shape.k):
            raise ValueError(
                f"{row.shape_name}: CPU decode dimensions disagree with manifest"
            )
    if require_full_inventory:
        expected_names = set(manifest.cpu_measurement_names(
            verifier=False,
            partition=expected_partition,
        ))
        if observed_names != expected_names:
            raise ValueError(
                "CPU production decode shape inventory is incomplete: "
                f"missing={sorted(expected_names-observed_names)[:3]} "
                f"unexpected={sorted(observed_names-expected_names)[:3]}"
            )
    validate_isa_regime_matrix(corpus)


def _require_partition(
    corpus: ObservationCorpus,
    manifest: NativeVNNIShapeManifest,
    partition: ShapePartition,
) -> ObservationCorpus:
    """Reject mixed or incomplete Fast-M1 development/sealed evidence."""

    assignments = partition_assignments(
        ((row.shape_group_id, row.shape_name) for row in corpus),
        verifier=False,
        manifest=manifest,
    )
    if set(assignments.values()) != {partition}:
        raise ValueError(f"CPU decode {partition.value} input crosses a partition")
    validate_complete(
        corpus,
        manifest,
        require_full_inventory=True,
        expected_partition=partition,
    )
    return corpus


def freeze_cpu_decode_policy(
    development: ObservationCorpus,
    manifest: NativeVNNIShapeManifest,
    profiler_feature_catalog: ProfilerFeatureCatalog | None = None,
    fit_cache_directory: Path | None = None,
    paired_development_comparisons: dict[
        PairedCellKey, tuple[PairedTimingComparison, ...]
    ] | None = None,
) -> FrozenPolicy:
    """Fit generic Fast-M1 rules without opening sealed observations."""

    corpus = _require_partition(
        development,
        manifest,
        ShapePartition.DEVELOPMENT,
    ).with_collapsed_aspect_domains()
    fit_cache = (
        PolicyFitCache(directory=fit_cache_directory)
        if fit_cache_directory is not None
        else None
    )
    return freeze_policy(
        corpus,
        sealed_commitment=_sealed_commitment(manifest),
        split_manifest_digest=manifest.digest(),
        paired_development_comparisons=paired_development_comparisons,
        profiler_feature_catalog=profiler_feature_catalog,
        fit_cache=fit_cache,
        metadata={
            "backend": "cpu",
            "semantic_contract": SemanticContract.FAST.value,
            "shape_manifest_schema": manifest.schema_version,
            "shape_manifest_digest": manifest.digest(),
            "split_surface": "cpu_decode_fast_m1",
            "arithmetic_policy": "frozen_serial_full_k_or_ordered_kpart",
            "paired_development_evidence_digest": paired_comparison_digest(
                paired_development_comparisons or {}
            ),
        },
    )


def certify_cpu_decode_policy(
    frozen: FrozenPolicy,
    development: ObservationCorpus,
    sealed: ObservationCorpus,
    manifest: NativeVNNIShapeManifest,
) -> CompiledPolicy:
    """Certify the immutable Fast-M1 policy on physically sealed rows."""

    development = _require_partition(
        development,
        manifest,
        ShapePartition.DEVELOPMENT,
    ).with_collapsed_aspect_domains()
    sealed = _require_partition(
        sealed, manifest, ShapePartition.SEALED
    ).with_collapsed_aspect_domains()
    return certify_frozen_policy(frozen, development, sealed)


def validate_emitter_inputs(
    policy_ir: PolicyIR,
    entries: list[DecodePolicyEntry],
    generic_rules: list[CPUDecodeGenericRule],
) -> None:
    """Prove C++ emission consumes the exact certified common policy IR."""

    ir_exact = {
        (
            *_cpu_runtime_surface(entry.key.architecture_class),
            entry.key.runtime_codebook_id,
            entry.key.aggregate_n,
            entry.key.k,
            _bundle_serial_kpart(entry.key.bundle_signature),
        ): entry.candidate_id
        for entry in policy_ir.exact_entries
    }
    emitted_exact = {
        (
            entry.build_isa,
            entry.runtime_isa,
            entry.threads,
            entry.codebook,
            entry.n,
            entry.k,
            entry.serial_kpart,
        ): entry.candidate_id
        for entry in entries
    }
    if ir_exact != emitted_exact:
        raise ValueError("CPU decode exact emitter inputs diverge from common IR")
    if tuple(item.rule for item in generic_rules) != policy_ir.generic_rules:
        raise ValueError("CPU decode generic emitter inputs diverge from common IR")


def validate_total_policy(generic_rules: list[CPUDecodeGenericRule]) -> None:
    """Prove every ISA/codebook/arithmetic domain has one cross-aspect tree."""

    groups = {
        (item.build_isa, item.runtime_isa, item.threads)
        for item in generic_rules
    }
    observed_regimes = {(build, runtime) for build, runtime, _ in groups}
    if observed_regimes != REQUIRED_ISA_REGIMES:
        raise ValueError(
            "CPU decode generated policy has incomplete ISA groups: "
            f"missing={sorted(REQUIRED_ISA_REGIMES-observed_regimes)}"
        )
    codebooks = sorted({spec.runtime_codebook("cpu") for spec in FORMAT_SPECS})
    missing = []
    malformed = []
    for build, runtime, threads in sorted(groups):
        for codebook in codebooks:
            for bundle in DECODE_BUNDLES:
                key = (build, runtime, threads, codebook, bundle)
                domain_rules = [
                    item.rule for item in generic_rules
                    if item.build_isa == build
                    and item.runtime_isa == runtime
                    and item.threads == threads
                    and item.rule.domain.runtime_codebook_id == codebook
                    and item.rule.domain.bundle_signature == bundle
                    and item.rule.domain.m == 1
                    and item.rule.domain.all_aspects
                ]
                if not domain_rules:
                    missing.append(key)
                    continue
                domains = {rule.domain for rule in domain_rules}
                if len(domains) != 1:
                    malformed.append((*key, "multiple-domains"))
                    continue
                try:
                    validate_generic_rule_partition(domain_rules)
                except ValueError as error:
                    malformed.append((*key, str(error)))
    if missing or malformed:
        raise ValueError(
            "CPU decode generated generic policy is not total: "
            f"missing_count={len(missing)} missing={missing} "
            f"malformed_count={len(malformed)} malformed={malformed}"
        )


def generate_include(
    entries: list[DecodePolicyEntry],
    generic_rules: list[CPUDecodeGenericRule],
    *,
    corpus_digest: str,
    registry_digest: str,
    profile: MeasurementProfile,
    policy_digest: str = "",
    certification: CertificationReport | None = None,
) -> str:
    """Render exact overlays first, followed by total generic rules."""

    certified = 1 if certification is not None else 0
    lines = [
        "/**",
        " * @file CPUNativeVNNIDecodePolicyGenerated.inc",
        " * @brief Generated byte-exact CPU NativeVNNI M=1 schedule policy.",
        " *",
        " * Exact measured overlays take precedence over certified generic",
        " * geometry rules. Both select only task ownership; the frozen serial",
        " * full-K/K-part arithmetic and ascending reductions remain unchanged.",
        " */",
        "// Auto-generated by analyze_cpu_native_vnni_decode_trainer.py. DO NOT EDIT.",
        f"// Measurement profile: {profile.value}",
        f"// Common corpus digest: {corpus_digest}",
        f"// Candidate registry digest: {registry_digest}",
        "#pragma once",
        "#include <cstdint>",
        "",
        "#define LLAMINAR_CPU_NVNNI_DECODE_POLICY_ABI 1",
        f"#define LLAMINAR_CPU_NVNNI_DECODE_POLICY_CERTIFIED {certified}",
        "",
        "namespace llaminar2::cpu::native_vnni::generated",
        "{",
        "enum class CPUNativeVNNIDecodePolicy : uint8_t",
        "{",
        "    Nbc1 = 0, Nbc2 = 1, Nbc4 = 2, Nbc8 = 3, Nbc16 = 4,",
        "};",
        "enum class CPUNativeVNNIDecodeBuildISA : uint8_t",
        "{",
        "    AVX2 = 0, AVX512 = 1,",
        "};",
        "enum class CPUNativeVNNIDecodeRuntimeISA : uint8_t",
        "{",
        "    AVX2 = 0, AVX512 = 1,",
        "};",
        "",
        "inline constexpr uint64_t packCPUNativeVNNIDecodePolicyKey(",
        "    uint8_t codebook, bool serial_kpart, int n, int k)",
        "{",
        "    return (static_cast<uint64_t>(codebook) << 56) |",
        "           (static_cast<uint64_t>(serial_kpart ? 1 : 0) << 55) |",
        "           (static_cast<uint64_t>(k & 0x7FFFFFFF) << 24) |",
        "           static_cast<uint64_t>(n & 0xFFFFFF);",
        "}",
        "",
        "inline bool selectCPUNativeVNNIDecodeGeneratedPolicy(",
        "    CPUNativeVNNIDecodeBuildISA build_isa,",
        "    CPUNativeVNNIDecodeRuntimeISA runtime_isa, int threads,",
        "    uint8_t codebook, int n, int k, bool serial_kpart,",
        "    CPUNativeVNNIDecodePolicy &policy)",
        "{",
        "    const uint64_t key = packCPUNativeVNNIDecodePolicyKey(",
        "        codebook, serial_kpart, n, k);",
    ]
    if certification is not None:
        lines[12:12] = [
            f"// Common policy digest: {policy_digest}",
            "// Frozen generic policy digest: "
            f"{certification.frozen_generic_policy_digest}",
            "// Sealed generic certificate: coverage="
            f"{certification.covered_cell_count}/"
            f"{certification.required_cell_count} max-regret="
            f"{certification.max_observed_regret:.6%} max-simultaneous-ucb="
            f"{certification.max_simultaneous_95pct_upper_regret:.6%}",
        ]

    groups = sorted({
        (entry.build_isa, entry.runtime_isa, entry.threads)
        for entry in entries
    } | {
        (item.build_isa, item.runtime_isa, item.threads)
        for item in generic_rules
    })
    for build, runtime, threads in groups:
        group_entries = [
            entry for entry in entries
            if (entry.build_isa, entry.runtime_isa, entry.threads)
            == (build, runtime, threads)
        ]
        group_rules = [
            item for item in generic_rules
            if (item.build_isa, item.runtime_isa, item.threads)
            == (build, runtime, threads)
        ]
        lines.extend([
            f"    if (build_isa == CPUNativeVNNIDecodeBuildISA::{build} &&",
            f"        runtime_isa == CPUNativeVNNIDecodeRuntimeISA::{runtime} &&",
            f"        threads == {threads})",
            "    {",
        ])
        if group_entries:
            lines.extend(["        switch (key)", "        {"])
            for entry in sorted(group_entries):
                shape_text = ",".join(entry.shape_names)
                lines.extend([
                    f"        case 0x{_pack_key(entry.codebook, entry.serial_kpart, entry.n, entry.k):016x}ULL:",
                    f"            // CB={entry.codebook} {shape_text} "
                    f"{entry.candidate_id} max-regret={entry.max_surface_regret:.4%}",
                    "            policy = "
                    f"CPUNativeVNNIDecodePolicy::{entry.policy};",
                    "            return true;",
                ])
            lines.extend(["        default:", "            break;", "        }"])
        if group_rules:
            lines.extend([
                "        const long long work_items =",
                "            static_cast<long long>(n) * static_cast<long long>(k);",
            ])
            for item in sorted(
                group_rules,
                key=lambda value: (
                    value.rule.domain.runtime_codebook_id,
                    value.rule.domain.bundle_signature,
                    generic_rule_sort_key(value.rule),
                ),
            ):
                rule = item.rule
                conditions = [
                    f"codebook == {rule.domain.runtime_codebook_id}",
                    "serial_kpart" if _bundle_serial_kpart(
                        rule.domain.bundle_signature
                    ) else "!serial_kpart",
                    *(() if rule.domain.all_aspects else (
                        aspect_condition(rule.domain.aspect_bucket),
                    )),
                    *(predicate_condition(value) for value in rule.predicates),
                ]
                lines.extend(render_if_header(conditions, indent="        "))
                lines.extend([
                    "        {",
                    "            policy = "
                    f"CPUNativeVNNIDecodePolicy::{_candidate_policy(rule.candidate_id)};",
                    "            return true;",
                    "        }",
                ])
        lines.append("    }")
    lines.extend([
        "    return false;",
        "}",
        "} // namespace llaminar2::cpu::native_vnni::generated",
        "",
    ])
    return "\n".join(lines)


def write_summary(path: Path, entries: list[DecodePolicyEntry]) -> None:
    """Write reviewable exact winners without affecting policy selection."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "build_isa", "runtime_isa", "threads", "codebook", "n", "k",
            "serial_kpart", "policy", "candidate_id", "shape_names",
            "max_surface_regret", "max_cv",
        ])
        for entry in entries:
            writer.writerow([
                entry.build_isa, entry.runtime_isa, entry.threads,
                entry.codebook, entry.n, entry.k, int(entry.serial_kpart),
                entry.policy, entry.candidate_id, ";".join(entry.shape_names),
                entry.max_surface_regret, entry.max_cv,
            ])


def _context_from_args(
    args: argparse.Namespace,
    inputs: tuple[Path, ...],
    timing_sidecars: tuple[Path, ...],
    *,
    run_id: str | None = None,
) -> CPUDecodeAdapterContext:
    """Construct provenance for one physical evidence partition."""

    corpus_id = raw_corpus_id((*inputs, *timing_sidecars))
    profile = MeasurementProfile(args.profile)
    if not profile.installable:
        return CPUDecodeAdapterContext(
            profile=profile,
            run_id="workflow-smoke",
            corpus_id=corpus_id,
            git_revision="workflow-smoke",
            build_id="workflow-smoke",
            compiler_id="workflow-smoke",
            architecture_class="workflow-smoke",
            device_name="workflow-smoke",
            driver_runtime="workflow-smoke",
            frozen_serial_policy_hash=(
                args.serial_m1_policy_hash or "sha256:" + "0" * 64
            ),
            raw_timing_sidecar_retained=bool(timing_sidecars),
        )
    return CPUDecodeAdapterContext(
        profile=profile,
        run_id=run_id or args.run_id,
        corpus_id=corpus_id,
        git_revision=args.git_revision,
        build_id=args.build_id,
        compiler_id=args.compiler_id,
        architecture_class=args.architecture_class,
        device_name=args.device_name,
        driver_runtime=args.driver_runtime,
        frozen_serial_policy_hash=args.serial_m1_policy_hash,
        raw_timing_sidecar_retained=bool(timing_sidecars),
    )


def _load_profiler_catalog(
    corpus: ObservationCorpus,
    args: argparse.Namespace,
) -> ProfilerFeatureCatalog:
    """Authenticate exact per-candidate profiler evidence for development."""

    return load_profiler_feature_catalog(
        corpus,
        args.development_profiler_requests,
        args.development_profiler_evidence,
        source_corpus=(
            read_observation_csv((args.development_profiler_observations,))
            if args.development_profiler_observations else None
        ),
        cache_path=(
            args.fit_cache_dir / "profiler_feature_catalog_v1.json"
            if args.fit_cache_dir is not None else None
        ),
    )


def _write_outputs(
    args: argparse.Namespace,
    corpus: ObservationCorpus,
    entries: list[DecodePolicyEntry],
    rules: list[CPUDecodeGenericRule],
    *,
    policy_digest: str = "",
    certification: CertificationReport | None = None,
) -> None:
    """Publish one generated include and optional review artifacts."""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generate_include(
        entries,
        rules,
        corpus_digest=corpus.digest(),
        registry_digest=candidate_registry_digest(),
        profile=MeasurementProfile(args.profile),
        policy_digest=policy_digest,
        certification=certification,
    ), encoding="utf-8")
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        write_summary(args.summary, entries)
    if args.common_observations:
        args.common_observations.parent.mkdir(parents=True, exist_ok=True)
        write_observation_csv(args.common_observations, corpus)


def main() -> int:
    """Run adaptation, development freeze, or untouched certification."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--input", nargs="+", type=Path, dest="input_options")
    parser.add_argument("--timing-sidecar", action="append", type=Path, default=[])
    parser.add_argument("--development-input", action="append", type=Path, default=[])
    parser.add_argument("--development-timing-sidecar", action="append", type=Path, default=[])
    parser.add_argument("--sealed-input", action="append", type=Path, default=[])
    parser.add_argument("--sealed-timing-sidecar", action="append", type=Path, default=[])
    parser.add_argument("--freeze-generic", action="store_true")
    parser.add_argument("--certify-generic", action="store_true")
    parser.add_argument("--adapt-only", action="store_true")
    parser.add_argument("--frozen-policy-json", type=Path)
    parser.add_argument("--policy-json", type=Path)
    parser.add_argument("--development-profiler-requests", type=Path)
    parser.add_argument("--development-profiler-evidence", type=Path)
    parser.add_argument("--development-profiler-observations", type=Path)
    parser.add_argument(
        "--paired-development-csv",
        action="append",
        type=Path,
        default=[],
        help=(
            "Development-only CPU paired tournament CSV; repeat for every "
            "retained shard. Ratios may refine development costs but never "
            "sealed observations."
        ),
    )
    parser.add_argument(
        "--fit-cache-dir",
        type=Path,
        help="Persistent content-addressed candidate-cost and CV cache",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--common-observations", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--require-isa-matrix", action="store_true")
    parser.add_argument(
        "--profile",
        choices=[profile.value for profile in MeasurementProfile],
        default=MeasurementProfile.QUICK.value,
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument("--git-revision", default="")
    parser.add_argument("--build-id", default="")
    parser.add_argument("--compiler-id", default="")
    parser.add_argument("--architecture-class", default="")
    parser.add_argument("--device-name", default="")
    parser.add_argument("--driver-runtime", default="")
    parser.add_argument("--serial-m1-policy-hash", default="")
    parser.add_argument("--shape-manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()

    positional = tuple(args.inputs)
    optional = tuple(args.input_options or ())
    if positional and optional:
        parser.error("use positional inputs or --input, not both")
    inputs = positional or optional
    if sum((args.freeze_generic, args.certify_generic, args.adapt_only)) > 1:
        parser.error("adapt, freeze, and certify modes are mutually exclusive")
    profiler_pair = bool(args.development_profiler_requests), bool(
        args.development_profiler_evidence
    )
    if profiler_pair[0] != profiler_pair[1]:
        parser.error("development profiler requests and evidence are required together")
    if args.development_profiler_observations and not all(profiler_pair):
        parser.error("profiler observations require requests and evidence")
    if args.paired_development_csv and not (
        args.freeze_generic or args.certify_generic
    ):
        parser.error(
            "--paired-development-csv requires generic freeze/certification"
        )
    if (args.freeze_generic or args.certify_generic) and not all(profiler_pair):
        parser.error("production CPU decode fitting requires profiler evidence")
    separate = bool(
        args.development_input or args.development_timing_sidecar
        or args.sealed_input or args.sealed_timing_sidecar
        or args.frozen_policy_json
    )
    if args.freeze_generic:
        if not inputs or separate or not args.policy_json:
            parser.error("freeze requires --input and --policy-json only")
    elif args.certify_generic:
        if inputs or args.timing_sidecar:
            parser.error("certification accepts separate partition inputs only")
        if not args.development_input or not args.sealed_input:
            parser.error("certification requires development and sealed inputs")
        if not args.frozen_policy_json or not args.policy_json:
            parser.error("certification requires frozen and compiled policy JSON")
    elif separate:
        parser.error("separate partition inputs require --certify-generic")
    elif not inputs:
        parser.error("at least one strong CPU decode CSV is required")
    if (
        MeasurementProfile(args.profile) == MeasurementProfile.PRODUCTION
        and not (args.freeze_generic or args.certify_generic or args.adapt_only)
    ):
        parser.error("production emission requires freeze and sealed certification")

    manifest = load_shape_manifest(args.shape_manifest)
    paired_cells = []
    for paired_path in args.paired_development_csv:
        paired_cells.extend(read_paired_confirmation_csv((paired_path,)))
    paired_development_comparisons = (
        paired_timing_comparisons(paired_cells) if paired_cells else {}
    )
    if args.certify_generic:
        development_inputs = tuple(args.development_input)
        development_timing = tuple(args.development_timing_sidecar)
        development_context = _context_from_args(
            args, development_inputs, development_timing
        )
        development = adapt_cpu_decode_csv(
            development_inputs,
            development_context,
            timing_sidecars=development_timing,
        )
        frozen = freeze_cpu_decode_policy(
            development,
            manifest,
            _load_profiler_catalog(development, args),
            args.fit_cache_dir,
            paired_development_comparisons,
        )
        validate_frozen_policy_file(args.frozen_policy_json, frozen)
        sealed_inputs = tuple(args.sealed_input)
        sealed_timing = tuple(args.sealed_timing_sidecar)
        sealed = adapt_cpu_decode_csv(
            sealed_inputs,
            _context_from_args(
                args,
                sealed_inputs,
                sealed_timing,
                run_id=f"{args.run_id}-sealed",
            ),
            timing_sidecars=sealed_timing,
        )
        compiled = certify_cpu_decode_policy(
            frozen, development, sealed, manifest
        )
        corpus = ObservationCorpus((
            *development.observations,
            *sealed.observations,
        ))
        entries = select_entries(corpus)
        rules = _emit_generic_rules(compiled.policy_ir.generic_rules)
        validate_emitter_inputs(compiled.policy_ir, entries, rules)
        validate_total_policy(rules)
        _write_outputs(
            args,
            corpus,
            entries,
            rules,
            policy_digest=compiled.policy_ir.digest(),
            certification=compiled.certification,
        )
        write_compiled_policy(args.policy_json, compiled)
        print(
            f"certified frozen CPU decode policy {frozen.generic_digest} "
            f"against {len(sealed)} sealed observations -> {args.output}"
        )
        return 0

    timing = tuple(args.timing_sidecar)
    context = _context_from_args(args, inputs, timing)
    corpus = adapt_cpu_decode_csv(inputs, context, timing_sidecars=timing)
    if args.adapt_only:
        validate_complete(
            corpus,
            manifest,
            require_full_inventory=context.profile.installable,
            expected_partition=(
                ShapePartition.DEVELOPMENT if context.profile.installable else None
            ),
        )
        _write_outputs(args, corpus, select_entries(corpus), [])
        print(f"adapted {len(corpus)} CPU decode observations -> {args.output}")
        return 0

    if args.freeze_generic:
        profiler_catalog = _load_profiler_catalog(corpus, args)
        frozen = freeze_cpu_decode_policy(
            corpus,
            manifest,
            profiler_catalog,
            args.fit_cache_dir,
            paired_development_comparisons,
        )
        entries = select_entries(corpus)
        rules = _emit_generic_rules(frozen.policy_ir.generic_rules)
        validate_emitter_inputs(frozen.policy_ir, entries, rules)
        validate_total_policy(rules)
        _write_outputs(
            args,
            corpus,
            entries,
            rules,
            policy_digest=frozen.policy_ir.digest(),
        )
        write_frozen_policy(args.policy_json, frozen)
        print(
            f"froze {len(corpus)} CPU decode observations as "
            f"{frozen.generic_digest} -> {args.output}"
        )
        return 0

    if args.require_complete:
        validate_complete(corpus, manifest, require_full_inventory=False)
    elif args.require_isa_matrix:
        validate_isa_regime_matrix(corpus)
    entries = select_entries(corpus)
    rules = select_generic_rules(corpus)
    _write_outputs(args, corpus, entries, rules)
    print(
        f"adapted {len(corpus)} CPU decode observations; generated "
        f"{len(entries)} exact policies -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
