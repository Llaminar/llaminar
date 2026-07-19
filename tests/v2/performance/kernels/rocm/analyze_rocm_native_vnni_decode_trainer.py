#!/usr/bin/env python3
"""Compile strong ROCm NativeVNNI decode evidence through the common policy core.

The analyzer deliberately performs no backend-local modal-label learning.
Every raw row is first adapted into the strict common schema; byte/route
eligibility, source-alias robustness, eager/captured robustness, and generic
regret learning then come from the shared NativeVNNI implementation.

Only Fast M=1 candidates encode a KB in the current ROCm runtime ABI.  The
grouped runtime-M verifier candidate is ``INHERIT_SERIAL_M1`` and is certified as
an independent execution surface, but it never emits a conflicting split-K
entry.  Production verifier resolution therefore consumes the exact same
frozen M=1 policy that defined its serial oracle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path


KERNEL_PERF_ROOT = Path(__file__).resolve().parents[1]
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.adapters.rocm_decode import (  # noqa: E402
    ROCmDecodeAdapterContext,
    adapt_rocm_decode_csv,
    raw_corpus_id,
)
from native_vnni_dispatch.candidate_observation import (  # noqa: E402
    read_observation_csv,
    write_observation_csv,
)
from native_vnni_dispatch.candidate_registry import (  # noqa: E402
    candidate_registry_digest,
    rocm_native_vnni_decode_registry,
)
from native_vnni_dispatch.certification import CertificationReport  # noqa: E402
from native_vnni_dispatch.compiler import (  # noqa: E402
    CompiledPolicy,
    FrozenPolicy,
    certify_frozen_policy,
    freeze_policy,
)
from native_vnni_dispatch.corpus import ObservationCorpus, RuntimeKey  # noqa: E402
from native_vnni_dispatch.cpp_predicates import (  # noqa: E402
    aspect_condition,
    generic_rule_sort_key,
    predicate_condition,
    render_if_header,
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
from native_vnni_dispatch.policy_artifact import (  # noqa: E402
    validate_installable_policy_artifact,
    validate_frozen_policy_file,
    write_compiled_policy,
    write_frozen_policy,
)
from native_vnni_dispatch.policy_ir import PolicyIR  # noqa: E402
from native_vnni_dispatch.schema import (  # noqa: E402
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
    require_registry_candidate_coverage,
    require_verifier_m_matrix,
)


CANONICAL_TARGET_WAVES = 4
OVERLAY_BEGIN = "    // BEGIN COMMON ROCM NATIVEVNNI DECODE EXACT OVERLAY"
OVERLAY_END = "    // END COMMON ROCM NATIVEVNNI DECODE EXACT OVERLAY"


@dataclass(frozen=True, order=True)
class FastEntry:
    """One alias/mode-robust Fast M=1 exact decision for the ROCm ABI."""

    codebook: int
    n: int
    k: int
    kb: int
    candidate_id: str
    shape_name: str
    max_surface_regret: float
    max_cv: float


def pack_shape_key(m: int, n: int, k: int) -> int:
    """Pack the existing ROCm decode exact-key ABI without lossy hashing."""

    return ((m & 0xFF) << 56) | ((k & 0xFFFFFF) << 28) | (n & 0x0FFFFFFF)


def _mode_robust_policy_corpus(corpus: ObservationCorpus) -> ObservationCorpus:
    """Preserve mode timing surfaces while collapsing the ROCm runtime key."""

    if not corpus.distinguishes_execution_mode:
        return corpus
    return ObservationCorpus(
        corpus.observations,
        distinguish_execution_mode=False,
    )


def _candidate_kb(candidate_id: str) -> int:
    """Resolve one common Fast candidate and return its effective KB."""

    candidate = rocm_native_vnni_decode_registry().resolve(candidate_id)
    if not candidate.supports_contract(SemanticContract.FAST):
        raise ValueError(f"{candidate_id} is not a Fast ROCm decode candidate")
    return int(candidate.config_json["kb"])


def _serial_hashes(
    corpus: ObservationCorpus,
    serial_m1_policy_hash: str,
) -> dict[RuntimeKey, str]:
    """Bind every represented verifier key to the frozen M=1 artifact."""

    return {
        key: serial_m1_policy_hash
        for key in corpus.runtime_keys()
        if key.semantic_contract == SemanticContract.VERIFIER_SERIAL_M1_BITWISE
    }


def select_fast_entries(
    corpus: ObservationCorpus,
    serial_m1_policy_hash: str,
) -> tuple[list[FastEntry], dict[RuntimeKey, ExactWinner]]:
    """Run the common exact oracle and encode only Fast M=1 KB decisions."""

    policy_corpus = _mode_robust_policy_corpus(corpus)
    exact = build_exact_winners(
        policy_corpus,
        serial_m1_hashes=_serial_hashes(policy_corpus, serial_m1_policy_hash),
    )
    entries = []
    for key, winner in sorted(exact.items()):
        if key.semantic_contract == SemanticContract.VERIFIER_SERIAL_M1_BITWISE:
            if winner.candidate_id != (
                "rocm.nvnni.decode.verifier.inherit_serial_m1"
            ):
                raise ValueError(
                    "grouped verifier exact winner does not inherit serial M1"
                )
            continue
        if key.semantic_contract != SemanticContract.FAST or key.m != 1:
            raise ValueError(f"unsupported ROCm decode exact policy key {key}")
        if len(key.projection_n_vector) != 1:
            raise ValueError("ROCm decode exact entry is not a single projection")
        rows = policy_corpus.rows_for_runtime_key(key)
        shape_names = sorted({row.shape_name for row in rows})
        if len(shape_names) != 1:
            raise ValueError(
                f"multiple shape names collapse onto ROCm exact key {key}: {shape_names}"
            )
        entries.append(FastEntry(
            codebook=key.runtime_codebook_id,
            n=key.aggregate_n,
            k=key.k,
            kb=_candidate_kb(winner.candidate_id),
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
    """Fit the shared bounded regret learner and retain Fast M=1 domains."""

    policy_corpus = _mode_robust_policy_corpus(corpus)
    generic = fit_generic_policy(
        policy_corpus,
        serial_m1_hashes=_serial_hashes(policy_corpus, serial_m1_policy_hash),
    )
    return [
        rule
        for rule in generic.rules
        if rule.domain.semantic_contract == SemanticContract.FAST
        and rule.domain.m == 1
    ]


def _fast_m1_corpus(corpus: ObservationCorpus) -> ObservationCorpus:
    """Project one physical transaction onto ROCm public Fast M=1 rows."""

    rows = tuple(
        row
        for row in corpus
        if row.semantic_contract == SemanticContract.FAST and row.m == 1
    )
    if not rows:
        raise ValueError("ROCm policy transaction contains no Fast M=1 evidence")
    return ObservationCorpus(rows)


def _fast_partition_surfaces(
    manifest: NativeVNNIShapeManifest,
    measurement_plan: NativeVNNIGPUMeasurementPlan,
    partition: ShapePartition,
) -> frozenset[tuple[str, str, ExecutionMode]]:
    """Return the reviewed ROCm shape/format/mode partition surface."""

    shape_names = (
        measurement_plan.common_development_shapes
        if partition == ShapePartition.DEVELOPMENT
        else manifest.partition_names(
            verifier=False,
            partition=ShapePartition.SEALED,
        )
    )
    surfaces = {
        (shape_name, spec.label, mode)
        for shape_name in shape_names
        for spec in FORMAT_SPECS
        for mode in (ExecutionMode.EAGER, ExecutionMode.GRAPH_CAPTURED)
    }
    if partition == ShapePartition.DEVELOPMENT:
        for extension in measurement_plan.fast_development_extensions:
            if extension.backend != Backend.ROCM:
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


def _require_fast_partition(
    corpus: ObservationCorpus,
    manifest: NativeVNNIShapeManifest,
    measurement_plan: NativeVNNIGPUMeasurementPlan,
    partition: ShapePartition,
) -> ObservationCorpus:
    """Reject mixed, incomplete, or dimension-stale ROCm Fast evidence."""

    direct = _fast_m1_corpus(corpus)
    require_canonical_alias_coverage(
        direct,
        required_execution_modes=(
            ExecutionMode.EAGER,
            ExecutionMode.GRAPH_CAPTURED,
        ),
    )
    require_candidate_matrix_complete(direct)
    require_registry_candidate_coverage(
        direct,
        rocm_native_vnni_decode_registry(),
    )
    required = _fast_partition_surfaces(manifest, measurement_plan, partition)
    actual = {
        (row.shape_name, row.source_format, row.execution_mode)
        for row in direct
    }
    if actual != required:
        order = lambda item: (item[0], item[1], item[2].value)
        missing = sorted(required - actual, key=order)
        unexpected = sorted(actual - required, key=order)
        raise ValueError(
            "ROCm Fast partition surface is incomplete: "
            f"missing_count={len(missing)} first_missing={missing[:1]} "
            f"unexpected_count={len(unexpected)} "
            f"first_unexpected={unexpected[:1]}"
        )
    for row in direct:
        shape = manifest.by_name(row.shape_name)
        if (row.aggregate_n, row.k) != (shape.n, shape.k):
            raise ValueError(
                f"{row.shape_name}: ROCm evidence dimensions disagree with "
                "the reviewed shape manifest"
            )
    assignments = partition_assignments(
        ((row.shape_group_id, row.shape_name) for row in direct),
        verifier=False,
        manifest=manifest,
    )
    if set(assignments.values()) != {partition}:
        raise ValueError(
            f"ROCm Fast {partition.value} input crosses a manifest partition"
        )
    return direct


def _fast_sealed_commitment(
    manifest: NativeVNNIShapeManifest,
    measurement_plan: NativeVNNIGPUMeasurementPlan,
) -> str:
    """Commit to the untouched ROCm Fast sealed inventory before fitting."""

    payload = {
        "protocol": "rocm-native-vnni-fast-m1-sealed-v1",
        "manifest_digest": manifest.digest(),
        "measurement_plan_digest": measurement_plan.digest(manifest),
        "shape_names": list(manifest.partition_names(
            verifier=False,
            partition=ShapePartition.SEALED,
        )),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def freeze_fast_policy(
    development_corpus: ObservationCorpus,
    manifest: NativeVNNIShapeManifest,
    measurement_plan: NativeVNNIGPUMeasurementPlan,
    profiler_feature_catalog: ProfilerFeatureCatalog,
) -> FrozenPolicy:
    """Fit mode-robust ROCm Fast rules without receiving sealed rows."""

    direct = _require_fast_partition(
        development_corpus,
        manifest,
        measurement_plan,
        ShapePartition.DEVELOPMENT,
    )
    development = _mode_robust_policy_corpus(direct)
    return freeze_policy(
        development,
        sealed_commitment=_fast_sealed_commitment(manifest, measurement_plan),
        split_manifest_digest=measurement_plan.digest(manifest),
        profiler_feature_catalog=profiler_feature_catalog,
        metadata={
            "backend": "rocm",
            "semantic_contract": SemanticContract.FAST.value,
            "shape_manifest_schema": manifest.schema_version,
            "shape_manifest_digest": manifest.digest(),
            "measurement_plan_schema": measurement_plan.schema_version,
            "measurement_plan_digest": measurement_plan.digest(manifest),
            "execution_mode_policy": "alias_robust_collapsed",
        },
    )


def certify_fast_policy(
    frozen: FrozenPolicy,
    development_corpus: ObservationCorpus,
    sealed_corpus: ObservationCorpus,
    manifest: NativeVNNIShapeManifest,
    measurement_plan: NativeVNNIGPUMeasurementPlan,
) -> CompiledPolicy:
    """Open ROCm sealed rows and certify the already-frozen generic IR."""

    development = _mode_robust_policy_corpus(_require_fast_partition(
        development_corpus,
        manifest,
        measurement_plan,
        ShapePartition.DEVELOPMENT,
    ))
    sealed = _mode_robust_policy_corpus(_require_fast_partition(
        sealed_corpus,
        manifest,
        measurement_plan,
        ShapePartition.SEALED,
    ))
    return certify_frozen_policy(frozen, development, sealed)


def validate_emitter_inputs(
    policy_ir: PolicyIR,
    entries: list[FastEntry],
    generic_rules: list[GenericDispatchRule],
) -> None:
    """Prove ROCm emission consumes the exact certified common policy IR."""

    ir_exact = {
        (
            entry.key.runtime_codebook_id,
            entry.key.aggregate_n,
            entry.key.k,
        ): entry.candidate_id
        for entry in policy_ir.exact_entries
    }
    emitted_exact = {
        (entry.codebook, entry.n, entry.k): entry.candidate_id
        for entry in entries
    }
    if emitted_exact != ir_exact:
        raise ValueError("ROCm exact emitter inputs disagree with common IR")
    if tuple(generic_rules) != policy_ir.generic_rules:
        raise ValueError("ROCm generic emitter inputs disagree with common IR")


def validate_complete(
    corpus: ObservationCorpus,
    manifest: NativeVNNIShapeManifest,
    measurement_plan: NativeVNNIGPUMeasurementPlan,
    *,
    require_full_inventory: bool,
) -> None:
    """Require every applicable surface, depth, mode, and candidate.

    The bounded measurement plan, rather than the broad supported-shape
    manifest, owns physical surface applicability. Treating every geometry as
    both contracts would turn a local M1 refinement into unrelated all-format
    verifier sweeps.
    """

    require_canonical_alias_coverage(
        corpus,
        required_execution_modes=(
            ExecutionMode.EAGER,
            ExecutionMode.GRAPH_CAPTURED,
        ),
    )
    require_verifier_m_matrix(corpus)
    require_candidate_matrix_complete(corpus)
    require_registry_candidate_coverage(
        corpus,
        rocm_native_vnni_decode_registry(),
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
                f"ROCm shape group maps to multiple manifest names: {identity}"
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
            backend=Backend.ROCM,
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
            "ROCm decode shape applicability/depth matrix is invalid for "
            f"{len(failures)} surface(s); first={failures[0]}"
        )

    if require_full_inventory:
        expected = measurement_plan.expected_decode_surfaces(
            manifest,
            backend=Backend.ROCM,
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
                "ROCm production physical surface inventory is incomplete: "
                f"missing_count={len(missing)} first_missing={missing[:1]} "
                f"unexpected_count={len(unexpected)} "
                f"first_unexpected={unexpected[:1]}"
            )


def validate_verifier_complete(
    corpus: ObservationCorpus,
    manifest: NativeVNNIShapeManifest,
    measurement_plan: NativeVNNIGPUMeasurementPlan,
) -> None:
    """Validate the staged-policy grouped verifier transaction by itself."""

    if any(
        row.semantic_contract != SemanticContract.VERIFIER_SERIAL_M1_BITWISE
        for row in corpus
    ):
        raise ValueError("ROCm verifier transaction contains Fast observations")
    require_canonical_alias_coverage(
        corpus,
        required_execution_modes=(
            ExecutionMode.EAGER,
            ExecutionMode.GRAPH_CAPTURED,
        ),
    )
    require_verifier_m_matrix(corpus)
    require_candidate_matrix_complete(corpus)
    expected = {
        surface
        for surface in measurement_plan.expected_decode_surfaces(
            manifest,
            backend=Backend.ROCM,
            source_formats=tuple(spec.label for spec in FORMAT_SPECS),
            execution_modes=(
                ExecutionMode.EAGER,
                ExecutionMode.GRAPH_CAPTURED,
            ),
            verifier_m_values=CANONICAL_VERIFIER_M,
        )
        if surface[0] == SemanticContract.VERIFIER_SERIAL_M1_BITWISE
    }
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
        raise ValueError(
            "ROCm verifier physical surface inventory is incomplete: "
            f"missing_count={len(expected - actual)} "
            f"unexpected_count={len(actual - expected)}"
        )


def _entry_line(entry: FastEntry, *, indent: str) -> str:
    """Render one current-ABI exact entry with common provenance."""

    return (
        f"{indent}{{0x{pack_shape_key(1, entry.n, entry.k):016x}ULL, "
        f"{{{entry.kb}, {CANONICAL_TARGET_WAVES}}}}}, "
        f"// CB={entry.codebook} M=1 {entry.n}x{entry.k} "
        f"{entry.candidate_id} {entry.shape_name} "
        f"max-regret={entry.max_surface_regret:.4%} max-cv={entry.max_cv:.4%}"
    )


def generate_include(
    entries: list[FastEntry],
    generic_rules: list[GenericDispatchRule],
    *,
    corpus_digest: str,
    registry_digest: str,
    profile: MeasurementProfile,
    policy_digest: str = "",
    certification: CertificationReport | None = None,
) -> str:
    """Render exact and common-regret generic decisions into the current ABI."""

    entries_by_codebook: dict[int, list[FastEntry]] = {}
    for entry in entries:
        entries_by_codebook.setdefault(entry.codebook, []).append(entry)
    for rows in entries_by_codebook.values():
        rows.sort(key=lambda item: pack_shape_key(1, item.n, item.k))

    rules_by_codebook: dict[int, list[GenericDispatchRule]] = {}
    for rule in generic_rules:
        rules_by_codebook.setdefault(rule.domain.runtime_codebook_id, []).append(rule)

    decision_comment = (
        "// Decisions: common alias/mode-robust exact oracle plus frozen "
        "development policy with generic-only sealed certification."
        if certification is not None
        else "// Decisions: development-only common alias/mode-robust exact "
        "oracle and profiler-informed generic policy; this artifact is not "
        "installable."
    )
    lines = [
        "// Auto-generated by analyze_rocm_native_vnni_decode_trainer.py. DO NOT EDIT.",
        decision_comment,
        "// Runtime-M verifier rows inherit this serial M=1 policy; no independent verifier KB is emitted.",
        f"// Measurement profile: {profile.value}",
        f"// Common corpus digest: {corpus_digest}",
        f"// Candidate registry digest: {registry_digest}",
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace llaminar2::rocm::generated",
        "{",
        "struct ROCmNativeVNNIDecodeDispatchConfig",
        "{",
        "    uint8_t kb = 0;",
        "    // ABI compatibility only: explicit KB fully determines the launch.",
        "    uint8_t target_waves_per_cu = 0;",
        "};",
        "",
        "struct ROCmNativeVNNIDecodeTuningEntry",
        "{",
        "    uint64_t key;",
        "    ROCmNativeVNNIDecodeDispatchConfig config;",
        "};",
        "",
        "inline constexpr uint64_t packROCmNativeVNNIDecodeDispatchKey(int m, int n, int k)",
        "{",
        "    return (static_cast<uint64_t>(m & 0xFF) << 56) |",
        "           (static_cast<uint64_t>(k & 0xFFFFFF) << 28) |",
        "           static_cast<uint64_t>(n & 0x0FFFFFFF);",
        "}",
        "",
        "template <size_t Count>",
        "inline bool findROCmNativeVNNIDecodeDispatchEntry(",
        "    const ROCmNativeVNNIDecodeTuningEntry (&table)[Count],",
        "    uint64_t key,",
        "    ROCmNativeVNNIDecodeDispatchConfig &out)",
        "{",
        "    size_t lo = 0;",
        "    size_t hi = Count;",
        "    while (lo < hi)",
        "    {",
        "        const size_t mid = lo + ((hi - lo) / 2);",
        "        if (table[mid].key == key)",
        "        {",
        "            out = table[mid].config;",
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
        "inline bool selectROCmNativeVNNIDecodeGenerated(",
        "    uint8_t codebook_id, int m, int n, int k, ROCmNativeVNNIDecodeDispatchConfig &out)",
        "{",
        "    const uint64_t key = packROCmNativeVNNIDecodeDispatchKey(m, n, k);",
    ]
    if certification is not None:
        lines[6:6] = [
            f"// Common policy digest: {policy_digest}",
            (
                "// Frozen generic policy digest: "
                f"{certification.frozen_generic_policy_digest}"
            ),
            (
                "// Sealed generic certificate: coverage="
                f"{certification.covered_cell_count}/"
                f"{certification.required_cell_count} max-regret="
                f"{certification.max_observed_regret:.6%} "
                "max-simultaneous-ucb="
                f"{certification.max_simultaneous_95pct_upper_regret:.6%}"
            ),
        ]

    for codebook in sorted(entries_by_codebook):
        lines.extend([
            f"    if (codebook_id == {codebook})",
            "    {",
            "        static constexpr ROCmNativeVNNIDecodeTuningEntry kTable[] = {",
        ])
        lines.extend(_entry_line(entry, indent="            ") for entry in entries_by_codebook[codebook])
        lines.extend([
            "        };",
            "        if (findROCmNativeVNNIDecodeDispatchEntry(kTable, key, out))",
            "            return true;",
            "    }",
        ])

    lines.extend([
        "",
        "    if (m != 1)",
        "        return false;",
    ])
    if rules_by_codebook:
        lines.extend([
            "    const long long work_items =",
            "        static_cast<long long>(n) * static_cast<long long>(k);",
        ])
    for codebook in sorted(rules_by_codebook):
        lines.append(f"    if (codebook_id == {codebook})")
        lines.append("    {")
        for rule in sorted(
            rules_by_codebook[codebook],
            key=generic_rule_sort_key,
        ):
            kb = _candidate_kb(rule.candidate_id)
            conditions = [
                aspect_condition(rule.domain.aspect_bucket),
                *(predicate_condition(predicate) for predicate in rule.predicates),
            ]
            lines.extend(render_if_header(conditions, indent="        "))
            lines.extend([
                "        {",
                f"            out = ROCmNativeVNNIDecodeDispatchConfig{{{kb}, {CANONICAL_TARGET_WAVES}}};",
                "            return true;",
                "        }",
            ])
        lines.append("    }")
    lines.extend([
        "    return false;",
        "}",
        "",
        "} // namespace llaminar2::rocm::generated",
        "",
    ])
    return "\n".join(lines)


def _strip_existing_overlay(text: str) -> str:
    """Remove one previous partial exact overlay before inserting a refresh."""

    begin = text.find(OVERLAY_BEGIN)
    if begin < 0:
        return text
    end = text.find(OVERLAY_END, begin)
    if end < 0:
        raise ValueError("base include has an unterminated common exact overlay")
    end += len(OVERLAY_END)
    if end < len(text) and text[end] == "\n":
        end += 1
    return text[:begin] + text[end:]


def emit_overlay(entries: list[FastEntry], output: Path, base: Path) -> None:
    """Layer common exact M=1 winners above a broader checked-in policy."""

    if not base.is_file():
        raise ValueError(f"base include not found: {base}")
    text = _strip_existing_overlay(base.read_text(encoding="utf-8"))
    marker = (
        "    const uint64_t key = "
        "packROCmNativeVNNIDecodeDispatchKey(m, n, k);\n"
    )
    location = text.find(marker)
    if location < 0:
        raise ValueError("base include lacks the ROCm selector key marker")
    insert_at = location + len(marker)
    overlay = [OVERLAY_BEGIN]
    for entry in sorted(entries):
        overlay.extend([
            f"    if (codebook_id == {entry.codebook} &&",
            f"        key == 0x{pack_shape_key(1, entry.n, entry.k):016x}ULL)",
            "    {",
            f"        out = ROCmNativeVNNIDecodeDispatchConfig{{{entry.kb}, {CANONICAL_TARGET_WAVES}}};",
            "        return true;",
            "    }",
        ])
    overlay.append(OVERLAY_END)
    updated = text[:insert_at] + "\n".join(overlay) + "\n" + text[insert_at:]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(updated, encoding="utf-8")


def write_summary(
    path: Path,
    entries: list[FastEntry],
    exact: dict[RuntimeKey, ExactWinner],
) -> None:
    """Write selected Fast entries plus verifier certification counts."""

    verifier_count = sum(
        key.semantic_contract == SemanticContract.VERIFIER_SERIAL_M1_BITWISE
        for key in exact
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "execution_codebook", "m", "n", "k", "candidate_id", "kb",
            "target_waves", "shape", "max_surface_regret", "max_cv",
            "certified_verifier_key_count",
        ])
        for entry in sorted(entries):
            writer.writerow([
                entry.codebook,
                1,
                entry.n,
                entry.k,
                entry.candidate_id,
                entry.kb,
                CANONICAL_TARGET_WAVES,
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
) -> ROCmDecodeAdapterContext:
    """Build conspicuous smoke provenance or strict installable provenance."""

    corpus_id = raw_corpus_id((*inputs, *timing_sidecars))
    profile = MeasurementProfile(args.profile)
    if not profile.installable:
        return ROCmDecodeAdapterContext.workflow_smoke(corpus_id=corpus_id)
    return ROCmDecodeAdapterContext(
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
    )


def main() -> int:
    """CLI entry point for common adaptation and current-ABI ROCm emission."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument(
        "--input", nargs="+", type=Path, dest="input_options",
        help="Legacy spelling for strong trainer CSV shard(s)",
    )
    parser.add_argument(
        "--timing-sidecar", action="append", type=Path, default=[],
        help="Raw timing CSV shard; repeat for every aggregate shard",
    )
    parser.add_argument(
        "--development-input", action="append", type=Path, default=[]
    )
    parser.add_argument(
        "--development-timing-sidecar", action="append", type=Path, default=[]
    )
    parser.add_argument("--sealed-input", action="append", type=Path, default=[])
    parser.add_argument(
        "--sealed-timing-sidecar", action="append", type=Path, default=[]
    )
    parser.add_argument("--freeze-generic", action="store_true")
    parser.add_argument("--certify-generic", action="store_true")
    parser.add_argument("--adapt-only", action="store_true")
    parser.add_argument("--frozen-policy-json", type=Path)
    parser.add_argument("--policy-json", type=Path)
    parser.add_argument("--development-profiler-requests", type=Path)
    parser.add_argument("--development-profiler-evidence", type=Path)
    parser.add_argument(
        "--development-profiler-observations",
        type=Path,
        help="Original common CSV bound to reusable profiler sidecars",
    )
    parser.add_argument("--certified-policy-include", type=Path)
    parser.add_argument("--certified-policy-json", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", "--summary-csv", dest="summary", type=Path)
    parser.add_argument("--common-observations", type=Path)
    parser.add_argument("--base-include", type=Path)
    parser.add_argument(
        "--max-generated-kb", type=int, default=64,
        help="Current ABI guard; values below the graph-safe cap are rejected",
    )
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--profile", choices=[profile.value for profile in MeasurementProfile],
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
    parser.add_argument(
        "--shape-manifest",
        type=Path,
        default=MANIFEST_PATH,
        help="Reviewed shape partitions and semantic-surface applicability",
    )
    parser.add_argument(
        "--measurement-plan",
        type=Path,
        default=MEASUREMENT_PLAN_PATH,
        help="Reviewed bounded and backend-scoped GPU timing plan",
    )
    args = parser.parse_args()

    positional = tuple(args.inputs)
    optional = tuple(args.input_options or ())
    if positional and optional:
        parser.error("use positional inputs or --input, not both")
    inputs = positional or optional
    if args.max_generated_kb != 64:
        parser.error(
            "--max-generated-kb must remain 64; silently dropping forceable "
            "candidates makes the common matrix incomplete"
        )
    if args.freeze_generic and args.certify_generic:
        parser.error("--freeze-generic and --certify-generic are mutually exclusive")
    if args.adapt_only and (args.freeze_generic or args.certify_generic):
        parser.error("--adapt-only cannot freeze or certify a policy")
    certified_replay = bool(
        args.certified_policy_include or args.certified_policy_json
    )
    if bool(args.certified_policy_include) != bool(args.certified_policy_json):
        parser.error(
            "certified policy include and JSON are required together"
        )
    if certified_replay and (
        args.adapt_only or args.freeze_generic or args.certify_generic
    ):
        parser.error("certified verifier replay is a separate transaction")
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
            "production ROCm freeze/certification requires complete "
            "development profiler evidence"
        )
    separate_certification = bool(
        args.development_input
        or args.development_timing_sidecar
        or args.sealed_input
        or args.sealed_timing_sidecar
        or args.frozen_policy_json
    )
    if args.freeze_generic:
        if not inputs or separate_certification:
            parser.error("--freeze-generic accepts development --input only")
        if not args.policy_json:
            parser.error("--freeze-generic requires --policy-json")
    elif args.certify_generic:
        if inputs or args.timing_sidecar:
            parser.error("--certify-generic accepts separate partition inputs only")
        if not args.development_input or not args.sealed_input:
            parser.error("certification requires development and sealed inputs")
        if not args.frozen_policy_json or not args.policy_json:
            parser.error("certification requires frozen and compiled policy JSON")
    elif separate_certification:
        parser.error("separate partition inputs require --certify-generic")
    elif not inputs:
        parser.error("at least one strong trainer CSV is required")
    if (
        MeasurementProfile(args.profile) == MeasurementProfile.PRODUCTION
        and not args.freeze_generic
        and not args.certify_generic
        and not args.adapt_only
        and not certified_replay
    ):
        parser.error(
            "production ROCm policy emission requires development freeze and "
            "separate sealed certification"
        )

    manifest = load_shape_manifest(args.shape_manifest)
    measurement_plan = load_gpu_measurement_plan(
        args.measurement_plan,
        manifest=manifest,
    )

    if args.adapt_only:
        if not inputs or separate_certification:
            parser.error("--adapt-only accepts development --input only")
        timing = tuple(args.timing_sidecar)
        context = _context_from_args(args, inputs, timing)
        corpus = adapt_rocm_decode_csv(inputs, context, timing_sidecars=timing)
        direct = _require_fast_partition(
            corpus,
            manifest,
            measurement_plan,
            ShapePartition.DEVELOPMENT,
        )
        policy_corpus = _mode_robust_policy_corpus(direct)
        entries, exact = select_fast_entries(
            policy_corpus,
            context.serial_m1_policy_hash,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(generate_include(
            entries,
            [],
            corpus_digest=policy_corpus.digest(),
            registry_digest=candidate_registry_digest(),
            profile=context.profile,
        ), encoding="utf-8")
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            write_summary(args.summary, entries, exact)
        if args.common_observations:
            args.common_observations.parent.mkdir(parents=True, exist_ok=True)
            write_observation_csv(args.common_observations, corpus)
        print(
            f"adapted {len(corpus)} ROCm Fast development observations "
            f"without fitting -> {args.output}"
        )
        return 0

    if args.freeze_generic:
        timing = tuple(args.timing_sidecar)
        context = _context_from_args(args, inputs, timing)
        corpus = adapt_rocm_decode_csv(inputs, context, timing_sidecars=timing)
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
        frozen = freeze_fast_policy(
            corpus,
            manifest,
            measurement_plan,
            profiler_catalog,
        )
        policy_corpus = _mode_robust_policy_corpus(_fast_m1_corpus(corpus))
        entries, exact = select_fast_entries(
            policy_corpus,
            context.serial_m1_policy_hash,
        )
        generic_rules = list(frozen.policy_ir.generic_rules)
        validate_emitter_inputs(frozen.policy_ir, entries, generic_rules)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(generate_include(
            entries,
            generic_rules,
            corpus_digest=policy_corpus.digest(),
            registry_digest=candidate_registry_digest(),
            profile=context.profile,
            policy_digest=frozen.policy_ir.digest(),
        ), encoding="utf-8")
        write_frozen_policy(args.policy_json, frozen)
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            write_summary(args.summary, entries, exact)
        if args.common_observations:
            args.common_observations.parent.mkdir(parents=True, exist_ok=True)
            write_observation_csv(args.common_observations, corpus)
        print(
            f"froze {len(corpus)} ROCm development observations as "
            f"{frozen.generic_digest} -> {args.output}"
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
        development = adapt_rocm_decode_csv(
            development_inputs,
            development_context,
            timing_sidecars=development_timing,
        )
        profiler_catalog = load_profiler_feature_catalog(
            development,
            args.development_profiler_requests,
            args.development_profiler_evidence,
            source_corpus=(
                read_observation_csv((args.development_profiler_observations,))
                if args.development_profiler_observations
                else None
            ),
        )
        frozen = freeze_fast_policy(
            development,
            manifest,
            measurement_plan,
            profiler_catalog,
        )
        # This complete byte comparison precedes the first sealed file read.
        validate_frozen_policy_file(args.frozen_policy_json, frozen)

        sealed_inputs = tuple(args.sealed_input)
        sealed_timing = tuple(args.sealed_timing_sidecar)
        sealed_context = _context_from_args(
            args,
            sealed_inputs,
            sealed_timing,
            run_id=f"{args.run_id}-sealed",
        )
        sealed = adapt_rocm_decode_csv(
            sealed_inputs,
            sealed_context,
            timing_sidecars=sealed_timing,
        )
        compiled = certify_fast_policy(
            frozen,
            development,
            sealed,
            manifest,
            measurement_plan,
        )
        development_fast = _fast_m1_corpus(development)
        sealed_fast = _fast_m1_corpus(sealed)
        policy_corpus = _mode_robust_policy_corpus(ObservationCorpus((
            *development_fast.observations,
            *sealed_fast.observations,
        )))
        entries, exact = select_fast_entries(
            policy_corpus,
            development_context.serial_m1_policy_hash,
        )
        generic_rules = list(compiled.policy_ir.generic_rules)
        validate_emitter_inputs(compiled.policy_ir, entries, generic_rules)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(generate_include(
            entries,
            generic_rules,
            corpus_digest=policy_corpus.digest(),
            registry_digest=candidate_registry_digest(),
            profile=development_context.profile,
            policy_digest=compiled.policy_ir.digest(),
            certification=compiled.certification,
        ), encoding="utf-8")
        write_compiled_policy(args.policy_json, compiled)
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            write_summary(args.summary, entries, exact)
        if args.common_observations:
            combined = ObservationCorpus((
                *development.observations,
                *sealed.observations,
            ))
            args.common_observations.parent.mkdir(parents=True, exist_ok=True)
            write_observation_csv(args.common_observations, combined)
        print(
            f"certified frozen ROCm policy {frozen.generic_digest} against "
            f"{len(sealed)} sealed observations -> {args.output}"
        )
        return 0

    timing = tuple(args.timing_sidecar)
    context = _context_from_args(args, inputs, timing)
    corpus = adapt_rocm_decode_csv(
        inputs,
        context,
        timing_sidecars=timing,
    )
    if certified_replay:
        validate_verifier_complete(corpus, manifest, measurement_plan)
        policy_corpus = _mode_robust_policy_corpus(corpus)
        entries, exact = select_fast_entries(
            policy_corpus,
            context.serial_m1_policy_hash,
        )
        if entries:
            raise ValueError("ROCm verifier replay unexpectedly emitted Fast entries")
        validate_installable_policy_artifact(
            args.certified_policy_json,
            include_path=args.certified_policy_include,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            args.certified_policy_include.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            write_summary(args.summary, entries, exact)
        if args.common_observations:
            args.common_observations.parent.mkdir(parents=True, exist_ok=True)
            write_observation_csv(args.common_observations, corpus)
        print(
            f"validated {len(corpus)} ROCm grouped verifier observations "
            f"against certified M1 policy -> {args.output}"
        )
        return 0
    policy_corpus = _mode_robust_policy_corpus(corpus)
    entries, exact = select_fast_entries(
        policy_corpus,
        context.serial_m1_policy_hash,
    )
    generic_rules = select_fast_generic_rules(
        policy_corpus,
        context.serial_m1_policy_hash,
    )
    if args.require_complete:
        manifest = load_shape_manifest(args.shape_manifest)
        validate_complete(
            corpus,
            manifest,
            load_gpu_measurement_plan(
                args.measurement_plan,
                manifest=manifest,
            ),
            require_full_inventory=(
                args.profile == MeasurementProfile.PRODUCTION.value
            ),
        )

    if args.base_include:
        emit_overlay(entries, args.output, args.base_include)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            generate_include(
                entries,
                generic_rules,
                corpus_digest=policy_corpus.digest(),
                registry_digest=candidate_registry_digest(),
                profile=context.profile,
            ),
            encoding="utf-8",
        )
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        write_summary(args.summary, entries, exact)
    if args.common_observations:
        args.common_observations.parent.mkdir(parents=True, exist_ok=True)
        write_observation_csv(args.common_observations, corpus)
    print(
        f"adapted {len(corpus)} strong observations; selected {len(entries)} "
        f"Fast M=1 exact entries and certified "
        f"{sum(key.semantic_contract == SemanticContract.VERIFIER_SERIAL_M1_BITWISE for key in exact)} "
        f"verifier keys -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
