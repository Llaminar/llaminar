#!/usr/bin/env python3
"""Compile strong ROCm grouped-MoE measurements into an anchor dispatch table.

This backend adapter no longer performs independent median or modal winner
selection.  It first converts every measured candidate into the strict common
NativeVNNI observation schema, then delegates eligibility and alias-robust
winner selection to :mod:`native_vnni_dispatch.exact_oracle`.

The emitted C++ table retains the current ROCm runtime ABI: execution codebook,
projection role, seven N:K anchors, and trained M anchors.  This is a backend
encoding of common decisions, not a second learner.  The broader common generic
aspect/work policy and sealed certificate are emitted separately as that
runtime ABI is migrated.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path


KERNEL_PERF_ROOT = Path(__file__).resolve().parents[1]
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.adapters.rocm_moe import (  # noqa: E402
    ROCmMoEAdapterContext,
    adapt_rocm_moe_csv,
    raw_corpus_id,
)
from native_vnni_dispatch.candidate_observation import (  # noqa: E402
    write_observation_csv,
)
from native_vnni_dispatch.candidate_registry import (  # noqa: E402
    candidate_registry_digest,
    rocm_moe_grouped_prefill_registry,
)
from native_vnni_dispatch.corpus import ObservationCorpus  # noqa: E402
from native_vnni_dispatch.exact_oracle import build_exact_winners  # noqa: E402
from native_vnni_dispatch.profiles import MeasurementProfile  # noqa: E402
from native_vnni_dispatch.schema import SemanticContract  # noqa: E402
from native_vnni_dispatch.validation import (  # noqa: E402
    require_candidate_matrix_complete,
    require_canonical_alias_coverage,
    require_registry_candidate_coverage,
)


CANONICAL_CODEBOOKS = (0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19)
CANONICAL_ROLES = ("gateup", "down")
CANONICAL_RATIO_EXPONENTS = (-3, -2, -1, 0, 1, 2, 3)
DEFAULT_M_ANCHORS = (12, 16, 24, 32, 64, 128, 256)


@dataclass(frozen=True, order=True)
class PolicyKey:
    """Current runtime-visible dimensions for one generated table entry."""

    codebook: int
    role: str
    ratio_exponent: int
    m_anchor: int


@dataclass(frozen=True)
class Winner:
    """One common-oracle decision encoded for the ROCm anchor-table ABI."""

    policy: PolicyKey
    candidate_id: str
    tile_m: int
    tile_n: int
    median_graph_us: float
    sample_count: int
    source_format_count: int
    max_surface_regret: float
    p95_surface_regret: float
    max_cv: float


def parse_int_list(raw: str) -> tuple[int, ...]:
    """Parse a sorted comma-separated positive integer list."""

    values = tuple(int(token.strip()) for token in raw.split(",") if token.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("M anchors must be positive integers")
    if tuple(sorted(set(values))) != values:
        raise argparse.ArgumentTypeError("M anchors must be sorted and unique")
    return values


def ratio_exponent(n: int, k: int) -> int:
    """Classify one projection N:K by nearest power-of-two exponent in [-3, 3]."""

    if n <= 0 or k <= 0:
        raise ValueError(f"invalid projection shape N={n} K={k}")
    exponent = int(round(math.log2(float(n) / float(k))))
    return max(CANONICAL_RATIO_EXPONENTS[0], min(CANONICAL_RATIO_EXPONENTS[-1], exponent))


def _role_from_operation(operation_kind: str) -> str:
    """Map the common operation identity back to the current launcher role."""

    if operation_kind == "MoEGroupedGateUpBundle":
        return "gateup"
    if operation_kind == "MoEGroupedDownProjection":
        return "down"
    raise ValueError(f"unknown ROCm MoE common operation {operation_kind!r}")


def select_winners(
    corpus: ObservationCorpus,
    anchors: tuple[int, ...],
    serial_m1_policy_hash: str,
) -> dict[PolicyKey, Winner]:
    """Encode common alias-robust exact winners into runtime anchor keys."""

    serial_hashes = {
        key: serial_m1_policy_hash
        for key in corpus.runtime_keys()
        if key.semantic_contract == SemanticContract.VERIFIER_SERIAL_M1_BITWISE
    }
    exact = build_exact_winners(corpus, serial_m1_hashes=serial_hashes)
    winners: dict[PolicyKey, Winner] = {}
    for runtime_key, exact_winner in exact.items():
        if runtime_key.m not in anchors:
            raise ValueError(
                f"ROCm MoE anchor emitter received untrained M={runtime_key.m}; "
                f"declared anchors={anchors}"
            )
        role = _role_from_operation(runtime_key.operation_kind)
        # Gate/up is represented as the honest (N,N) homogeneous bundle in the
        # common IR. The current C++ ABI receives one N because both projections
        # share that dimension and tile policy.
        projection_n = runtime_key.projection_n_vector[0]
        if role == "gateup" and runtime_key.projection_n_vector != (
            projection_n, projection_n
        ):
            raise ValueError("gate/up common bundle is not homogeneous")
        policy = PolicyKey(
            codebook=runtime_key.runtime_codebook_id,
            role=role,
            ratio_exponent=ratio_exponent(projection_n, runtime_key.k),
            m_anchor=runtime_key.m,
        )
        if policy in winners:
            raise ValueError(
                f"multiple exact shapes collapse onto current ROCm policy key {policy}; "
                "the backend ABI must be widened before emitting this corpus"
            )
        config = exact_winner.config_json
        winners[policy] = Winner(
            policy=policy,
            candidate_id=exact_winner.candidate_id,
            tile_m=int(config["tile_m"]),
            tile_n=int(config["tile_n"]),
            median_graph_us=statistics.median(
                surface.median_us for surface in exact_winner.surfaces
            ),
            sample_count=sum(
                surface.sample_count for surface in exact_winner.surfaces
            ),
            source_format_count=len({
                surface.surface.source_format for surface in exact_winner.surfaces
            }),
            max_surface_regret=exact_winner.max_surface_regret,
            p95_surface_regret=exact_winner.p95_surface_regret,
            max_cv=exact_winner.max_cv,
        )
    return winners


def validate_complete(
    corpus: ObservationCorpus,
    winners: dict[PolicyKey, Winner],
    anchors: tuple[int, ...],
) -> None:
    """Require the full runtime, alias, and explicit-candidate Cartesian product."""

    expected = {
        PolicyKey(codebook, role, ratio, m)
        for codebook in CANONICAL_CODEBOOKS
        for role in CANONICAL_ROLES
        for ratio in CANONICAL_RATIO_EXPONENTS
        for m in anchors
    }
    missing = sorted(expected.difference(winners))
    unexpected = sorted(set(winners).difference(expected))
    if missing or unexpected:
        fragments = []
        if missing:
            fragments.append(f"missing {len(missing)} policies; first={missing[:8]}")
        if unexpected:
            fragments.append(
                f"unexpected {len(unexpected)} policies; first={unexpected[:8]}"
            )
        raise ValueError("incomplete grouped-prefill sweep: " + "; ".join(fragments))

    require_canonical_alias_coverage(corpus)
    require_candidate_matrix_complete(corpus)
    require_registry_candidate_coverage(
        corpus, rocm_moe_grouped_prefill_registry()
    )


def generate_include(
    winners: dict[PolicyKey, Winner],
    anchors: tuple[int, ...],
    *,
    corpus_digest: str,
    registry_digest: str,
    profile: MeasurementProfile,
) -> str:
    """Render the current C++ ABI without changing common policy decisions."""

    lines = [
        "// Auto-generated by analyze_rocm_moe_grouped_prefill_trainer.py. DO NOT EDIT.",
        "// Decisions: common NativeVNNI alias-robust exact oracle.",
        f"// Measurement profile: {profile.value}",
        f"// Common corpus digest: {corpus_digest}",
        f"// Candidate registry digest: {registry_digest}",
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        "namespace llaminar2::rocm::generated",
        "{",
        "    struct ROCmMoEGroupedPrefillConfig",
        "    {",
        "        uint8_t tile_m = 0;",
        "        uint16_t tile_n = 0;",
        "    };",
        "",
        "    struct ROCmMoEGroupedPrefillEntry",
        "    {",
        "        uint8_t codebook;",
        "        uint8_t role; // 0=gate/up, 1=down",
        "        int8_t ratio_exponent;",
        "        uint16_t m_anchor;",
        "        ROCmMoEGroupedPrefillConfig config;",
        "    };",
        "",
        "    inline constexpr uint16_t kROCmMoEGroupedPrefillMAnchors[] = {",
        "        " + ", ".join(str(anchor) for anchor in anchors),
        "    };",
        "",
        "    inline constexpr ROCmMoEGroupedPrefillEntry kROCmMoEGroupedPrefillEntries[] = {",
    ]

    role_ids = {"gateup": 0, "down": 1}
    for policy in sorted(winners):
        winner = winners[policy]
        lines.append(
            "        {"
            f"{policy.codebook}, {role_ids[policy.role]}, {policy.ratio_exponent}, "
            f"{policy.m_anchor}, {{{winner.tile_m}, {winner.tile_n}}}"
            "},"
        )

    lines.extend([
        "    };",
        "",
        "    inline int rocmMoEGroupedPrefillRatioExponent(int n, int k)",
        "    {",
        "        const int64_t n64 = n;",
        "        const int64_t k64 = k;",
        "        const int64_t n_squared = n64 * n64;",
        "        const int64_t k_squared = k64 * k64;",
        "        if (n64 >= k64)",
        "        {",
        "            if (n_squared >= 32 * k_squared) return 3;",
        "            if (n_squared >= 8 * k_squared) return 2;",
        "            if (n_squared >= 2 * k_squared) return 1;",
        "            return 0;",
        "        }",
        "        if (k_squared >= 32 * n_squared) return -3;",
        "        if (k_squared >= 8 * n_squared) return -2;",
        "        if (k_squared >= 2 * n_squared) return -1;",
        "        return 0;",
        "    }",
        "",
        "    inline uint16_t rocmMoEGroupedPrefillMAnchor(int m)",
        "    {",
        "        uint16_t best = kROCmMoEGroupedPrefillMAnchors[0];",
        "        int best_distance = m >= best ? m - best : best - m;",
        "        for (uint16_t candidate : kROCmMoEGroupedPrefillMAnchors)",
        "        {",
        "            const int distance = m >= candidate ? m - candidate : candidate - m;",
        "            if (distance < best_distance)",
        "            {",
        "                best = candidate;",
        "                best_distance = distance;",
        "            }",
        "        }",
        "        return best;",
        "    }",
        "",
        "    inline bool selectROCmMoEGroupedPrefillGenerated(",
        "        uint8_t codebook,",
        "        uint8_t role,",
        "        int m,",
        "        int n,",
        "        int k,",
        "        ROCmMoEGroupedPrefillConfig &config)",
        "    {",
        "        if (m <= 0 || n <= 0 || k <= 0 || role > 1)",
        "            return false;",
        "        const int ratio = rocmMoEGroupedPrefillRatioExponent(n, k);",
        "        const uint16_t m_anchor = rocmMoEGroupedPrefillMAnchor(m);",
        "        for (const auto &entry : kROCmMoEGroupedPrefillEntries)",
        "        {",
        "            if (entry.codebook == codebook &&",
        "                entry.role == role &&",
        "                entry.ratio_exponent == ratio &&",
        "                entry.m_anchor == m_anchor)",
        "            {",
        "                config = entry.config;",
        "                return true;",
        "            }",
        "        }",
        "        return false;",
        "    }",
        "} // namespace llaminar2::rocm::generated",
        "",
    ])
    return "\n".join(lines)


def write_summary(path: Path, winners: dict[PolicyKey, Winner]) -> None:
    """Write selected candidates and robust surface diagnostics for review."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "execution_codebook", "role", "ratio_exponent", "m_anchor",
            "candidate_id", "tile_m", "tile_n", "median_graph_us",
            "sample_count", "source_format_count", "max_surface_regret",
            "p95_surface_regret", "max_cv",
        ])
        for policy in sorted(winners):
            winner = winners[policy]
            writer.writerow([
                policy.codebook,
                policy.role,
                policy.ratio_exponent,
                policy.m_anchor,
                winner.candidate_id,
                winner.tile_m,
                winner.tile_n,
                f"{winner.median_graph_us:.6f}",
                winner.sample_count,
                winner.source_format_count,
                f"{winner.max_surface_regret:.9f}",
                f"{winner.p95_surface_regret:.9f}",
                f"{winner.max_cv:.9f}",
            ])


def _context_from_args(args: argparse.Namespace) -> ROCmMoEAdapterContext:
    """Build either conspicuous smoke provenance or strict installable context."""

    corpus_id = raw_corpus_id((*args.inputs, *args.timing_sidecar))
    profile = MeasurementProfile(args.profile)
    if not profile.installable:
        return ROCmMoEAdapterContext.workflow_smoke(corpus_id=corpus_id)
    return ROCmMoEAdapterContext(
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
    """CLI entry point for common adaptation and ROCm backend emission."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Strong trainer CSV shard(s)")
    parser.add_argument(
        "--timing-sidecar", action="append", type=Path, default=[],
        help="Raw timing CSV shard; repeat for each retained shard",
    )
    parser.add_argument("--output", type=Path, required=True, help="Generated C++ include")
    parser.add_argument("--summary-csv", type=Path, help="Selected-winner diagnostics")
    parser.add_argument(
        "--common-observations", type=Path,
        help="Optional strict common-schema corpus output",
    )
    parser.add_argument(
        "--m-anchors", type=parse_int_list, default=DEFAULT_M_ANCHORS,
        help="Sorted trained M anchors (default: 12,16,24,32,64,128,256)",
    )
    parser.add_argument(
        "--require-complete", action="store_true",
        help="Require canonical codebook/role/aspect/M/alias/candidate coverage",
    )
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

    context = _context_from_args(args)
    corpus = adapt_rocm_moe_csv(
        args.inputs,
        context,
        timing_sidecars=args.timing_sidecar,
    )
    winners = select_winners(corpus, args.m_anchors, context.serial_m1_policy_hash)
    if args.require_complete:
        validate_complete(corpus, winners, args.m_anchors)

    args.output.write_text(
        generate_include(
            winners,
            args.m_anchors,
            corpus_digest=corpus.digest(),
            registry_digest=candidate_registry_digest(),
            profile=context.profile,
        ),
        encoding="utf-8",
    )
    if args.summary_csv:
        write_summary(args.summary_csv, winners)
    if args.common_observations:
        write_observation_csv(args.common_observations, corpus)
    print(
        f"adapted {len(corpus)} strong observations; generated {len(winners)} "
        f"ROCm grouped-prefill policies -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
