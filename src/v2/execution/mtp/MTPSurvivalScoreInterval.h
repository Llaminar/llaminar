/**
 * @file MTPSurvivalScoreInterval.h
 * @brief Exact integer prefix/progress bounds for one supported MTP window.
 *
 * This allocation-free host/device calculation consumes existing R/A/Z/F. It
 * has no live state, histogram, depth selector, sampling policy or cost table.
 * Bounds describe truncation/extension of the SAME supported inputs, not the
 * unseen frontiers of another runtime policy. Costs are borrowed positive
 * immutable estimates; a supported score is not a throughput guarantee.
 */
#pragma once

#include "MTPReversibleProposalGate.h"
#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MTP_SURVIVAL_HD __host__ __device__ __forceinline__
#else
#define LLAMINAR_MTP_SURVIVAL_HD inline
#endif

namespace llaminar2
{
    /** @brief Caller-owned proof that counts describe supported state progress. */
    enum class MTPPrefixEvidenceScope : uint32_t
    {
        Unsupported,
        NonStoppedUnclippedVerifiedState,
    };

    /**
     * @brief Borrowed same-depth eligible window; no new control/trace ABI words.
     *
     * A is the sum of accepted prefixes, Z counts zero prefixes, F counts full
     * prefixes. The commit owner must prove V=Y+carry_in-carry_out=1+prefix for
     * every included transaction. Stops/clipping are not representable by this
     * supported scope. This function cannot establish that proof from counts.
     * accepted is widened on input, not stored as new device state.
     */
    struct MTPPrefixWindowEvidence
    {
        MTPPrefixEvidenceScope scope = MTPPrefixEvidenceScope::Unsupported;
        uint32_t depth = 0;
        uint32_t runs = 0;
        uint64_t accepted = 0;
        uint32_t zero = 0;
        uint32_t full = 0;
    };

    /** @brief Fatal malformed/unsupported inputs; invalid results carry no score. */
    enum class MTPSurvivalScoreError : uint32_t
    {
        None, UnsupportedEvidence, InvalidDepth, EmptyWindow,
        InconsistentCounts, InvalidCandidateDepth, InvalidCost,
    };

    /**
     * @brief Evidence ordering relative to the same window's incumbent score.
     *
     * Dominated includes a tie: the candidate upper score cannot strictly win.
     * Supported means its LOWER score strictly wins on this exposure only.
     * Unresolved means the score interval overlaps a possible strict win; an
     * upper-bound win is not an observed improvement or permission to promote.
     */
    enum class MTPSurvivalScoreClass : uint32_t
    {
        Invalid, Incumbent, Dominated, Supported, Unresolved,
    };

    /** @brief Candidate geometry, independent of score classification. */
    enum class MTPPrefixCandidateDirection : uint32_t { Same, Lower, Higher };

    /**
     * @brief Exact total verified-state progress, not emitted response tokens.
     *
     * For uint32 R and d<=15, every member is <=16*UINT32_MAX<2^36. Counts,
     * products by depths and the integer extrema therefore fit uint64 exactly.
     */
    struct MTPVerifiedProgressInterval
    {
        uint64_t incumbent = 0;
        uint64_t lower = 0;
        uint64_t upper = 0;
    };

    /** @brief Immutable classification; intentionally has no chosen-depth field. */
    struct MTPSurvivalScoreResult
    {
        MTPSurvivalScoreError error = MTPSurvivalScoreError::UnsupportedEvidence;
        MTPSurvivalScoreClass classification = MTPSurvivalScoreClass::Invalid;
        MTPPrefixCandidateDirection direction = MTPPrefixCandidateDirection::Same;
        MTPVerifiedProgressInterval progress{};

        /** @return Whether all evidence, geometry and costs passed validation. */
        LLAMINAR_MTP_SURVIVAL_HD bool valid() const noexcept
        {
            return error == MTPSurvivalScoreError::None;
        }
    };
    static_assert(std::is_trivially_copyable_v<MTPPrefixWindowEvidence>);
    static_assert(std::is_trivially_copyable_v<MTPSurvivalScoreResult>);

    /**
     * @brief Classify a same-exposure score interval without choosing an action.
     * @param evidence Actual same-depth eligible R/A/Z/F and typed commit proof.
     * @param candidate_depth Positive hypothetical prefix depth within1..15.
     * @param incumbent_cost Positive immutable cost at evidence.depth.
     * @param candidate_cost Positive immutable cost at candidate_depth.
     * @return Exact progress interval/classification, or error with zero interval.
     *
     * At d>=3 the interior n=R-Z-F values lie in[1,d-1] with sum S=A-dF.
     * Concentrating S-n excess into full d-2 loads and one remainder minimizes
     * the concave min(L,k). Distributing below/above k attains the upper bound.
     * At k>d only F censored-full samples can extend; their future is unknown.
     * Scores V/(R*C) compare as V/C because this SAME R is common. The existing
     * Euclidean comparator uses division/remainders, avoiding products that can
     * exceed uint64 at maximum runs/cost. Cost monotonicity is not assumed.
     * The helper never allocates, mutates evidence or reads device/host state.
     */
    LLAMINAR_MTP_SURVIVAL_HD MTPSurvivalScoreResult evaluateMTPPrefixSurvivalScore(
        const MTPPrefixWindowEvidence &evidence, uint32_t candidate_depth,
        uint32_t incumbent_cost, uint32_t candidate_cost) noexcept
    {
        MTPSurvivalScoreResult result;
        if (evidence.scope != MTPPrefixEvidenceScope::NonStoppedUnclippedVerifiedState)
            return result;
        if (evidence.depth < 1 || evidence.depth > 15)
        {
            result.error = MTPSurvivalScoreError::InvalidDepth;
            return result;
        }
        if (evidence.runs == 0)
        {
            result.error = MTPSurvivalScoreError::EmptyWindow;
            return result;
        }
        const uint64_t d = evidence.depth, r = evidence.runs;
        const uint64_t a = evidence.accepted, z = evidence.zero, f = evidence.full;
        if (z > r || f > r - z || a > d * r || a < d * f)
        {
            result.error = MTPSurvivalScoreError::InconsistentCounts;
            return result;
        }
        const uint64_t n = r - z - f, s = a - d * f;
        if ((d == 1 && (n != 0 || a != f)) ||
            (d != 1 && (s < n || s > (d - 1) * n)))
        {
            result.error = MTPSurvivalScoreError::InconsistentCounts;
            return result;
        }
        if (candidate_depth < 1 || candidate_depth > 15)
        {
            result.error = MTPSurvivalScoreError::InvalidCandidateDepth;
            return result;
        }
        if (incumbent_cost == 0 || candidate_cost == 0 ||
            (candidate_depth == evidence.depth && candidate_cost != incumbent_cost))
        {
            result.error = MTPSurvivalScoreError::InvalidCost;
            return result;
        }

        const uint64_t k = candidate_depth;
        uint64_t lower = a, upper = a;
        if (k > d)
        {
            upper += (k - d) * f;
            result.direction = MTPPrefixCandidateDirection::Higher;
        }
        else if (k < d)
        {
            result.direction = MTPPrefixCandidateDirection::Lower;
            if (d == 2)
                lower = upper = f + n; // The only interior value and k are1.
            else
            {
                const uint64_t excess = s - n;
                const uint64_t q = excess / (d - 2), rem = excess % (d - 2);
                lower = k * f + n + q * (k - 1) + (rem < k - 1 ? rem : k - 1);
                upper = k * f + (s < k * n ? s : k * n);
            }
        }

        result.progress = {r + a, r + lower, r + upper};
        result.error = MTPSurvivalScoreError::None;
        if (k == d)
            result.classification = MTPSurvivalScoreClass::Incumbent;
        else if (compareMTPUtilityFractions(result.progress.upper, candidate_cost,
                     result.progress.incumbent, incumbent_cost) != MTPUtilityOrder::Greater)
            result.classification = MTPSurvivalScoreClass::Dominated;
        else if (compareMTPUtilityFractions(result.progress.lower, candidate_cost,
                     result.progress.incumbent, incumbent_cost) == MTPUtilityOrder::Greater)
            result.classification = MTPSurvivalScoreClass::Supported;
        else
            result.classification = MTPSurvivalScoreClass::Unresolved;
        return result;
    }
}
#undef LLAMINAR_MTP_SURVIVAL_HD
