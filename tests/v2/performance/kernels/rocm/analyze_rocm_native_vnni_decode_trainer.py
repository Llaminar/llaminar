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
import sys
from dataclasses import dataclass
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
    write_observation_csv,
)
from native_vnni_dispatch.candidate_registry import (  # noqa: E402
    candidate_registry_digest,
    rocm_native_vnni_decode_registry,
)
from native_vnni_dispatch.corpus import ObservationCorpus, RuntimeKey  # noqa: E402
from native_vnni_dispatch.exact_oracle import (  # noqa: E402
    ExactWinner,
    build_exact_winners,
)
from native_vnni_dispatch.profiles import MeasurementProfile  # noqa: E402
from native_vnni_dispatch.schema import (  # noqa: E402
    AspectBucket,
    ExecutionMode,
    SemanticContract,
)
from native_vnni_dispatch.segmented_policy import (  # noqa: E402
    GenericDispatchRule,
    fit_generic_policy,
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

    exact = build_exact_winners(
        corpus,
        serial_m1_hashes=_serial_hashes(corpus, serial_m1_policy_hash),
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
        rows = corpus.rows_for_runtime_key(key)
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


def validate_complete(corpus: ObservationCorpus) -> None:
    """Require all aliases, modes, M depths, and registry candidates."""

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
    for row in corpus:
        identity = (
            row.runtime_codebook_id,
            row.source_format,
            row.execution_mode,
            row.shape_group_id,
        )
        by_shape.setdefault(identity, set()).add((row.semantic_contract, row.m))
    required = {(SemanticContract.FAST, 1)} | {
        (SemanticContract.VERIFIER_SERIAL_M1_BITWISE, m)
        for m in CANONICAL_VERIFIER_M
    }
    missing = [
        (identity, sorted(required - represented, key=lambda item: (item[0].value, item[1])))
        for identity, represented in by_shape.items()
        if not required.issubset(represented)
    ]
    if missing:
        raise ValueError(
            f"ROCm decode Fast/verifier depth matrix is incomplete for "
            f"{len(missing)} surface(s); first={missing[0]}"
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


def _aspect_condition(bucket: AspectBucket) -> str:
    return {
        AspectBucket.VERY_WIDE: "aspect_ratio >= 16.0f",
        AspectBucket.WIDE: "aspect_ratio >= 2.0f",
        AspectBucket.BALANCED: "aspect_ratio >= 0.75f",
        AspectBucket.TALL: "true",
    }[bucket]


def generate_include(
    entries: list[FastEntry],
    generic_rules: list[GenericDispatchRule],
    *,
    corpus_digest: str,
    registry_digest: str,
    profile: MeasurementProfile,
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

    lines = [
        "// Auto-generated by analyze_rocm_native_vnni_decode_trainer.py. DO NOT EDIT.",
        "// Decisions: common alias/mode-robust exact oracle + segmented-regret learner.",
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
            "    const float aspect_ratio =",
            "        k > 0 ? static_cast<float>(n) / static_cast<float>(k) : 0.0f;",
            "    const long long work_items =",
            "        static_cast<long long>(n) * static_cast<long long>(k);",
        ])
    for codebook in sorted(rules_by_codebook):
        lines.append(f"    if (codebook_id == {codebook})")
        lines.append("    {")
        for rule in sorted(
            rules_by_codebook[codebook],
            key=lambda item: (
                list(AspectBucket).index(item.domain.aspect_bucket),
                item.min_work_items,
            ),
        ):
            kb = _candidate_kb(rule.candidate_id)
            lines.extend([
                f"        if ({_aspect_condition(rule.domain.aspect_bucket)} &&",
                f"            work_items >= {rule.min_work_items}LL &&",
                f"            work_items <= {rule.max_work_items}LL)",
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
) -> ROCmDecodeAdapterContext:
    """Build conspicuous smoke provenance or strict installable provenance."""

    corpus_id = raw_corpus_id((*inputs, *args.timing_sidecar))
    profile = MeasurementProfile(args.profile)
    if not profile.installable:
        return ROCmDecodeAdapterContext.workflow_smoke(corpus_id=corpus_id)
    return ROCmDecodeAdapterContext(
        profile=profile,
        run_id=args.run_id,
        corpus_id=corpus_id,
        git_revision=args.git_revision,
        build_id=args.build_id,
        compiler_id=args.compiler_id,
        architecture_class=args.architecture_class,
        device_name=args.device_name,
        driver_runtime=args.driver_runtime,
        serial_m1_policy_hash=args.serial_m1_policy_hash,
        raw_timing_sidecar_retained=bool(args.timing_sidecar),
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
    args = parser.parse_args()

    positional = tuple(args.inputs)
    optional = tuple(args.input_options or ())
    if positional and optional:
        parser.error("use positional inputs or --input, not both")
    inputs = positional or optional
    if not inputs:
        parser.error("at least one strong trainer CSV is required")
    if args.max_generated_kb != 64:
        parser.error(
            "--max-generated-kb must remain 64; silently dropping forceable "
            "candidates makes the common matrix incomplete"
        )

    context = _context_from_args(args, inputs)
    corpus = adapt_rocm_decode_csv(
        inputs,
        context,
        timing_sidecars=args.timing_sidecar,
    )
    entries, exact = select_fast_entries(
        corpus,
        context.serial_m1_policy_hash,
    )
    generic_rules = select_fast_generic_rules(
        corpus,
        context.serial_m1_policy_hash,
    )
    if args.require_complete:
        validate_complete(corpus)

    if args.base_include:
        emit_overlay(entries, args.output, args.base_include)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            generate_include(
                entries,
                generic_rules,
                corpus_digest=corpus.digest(),
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
