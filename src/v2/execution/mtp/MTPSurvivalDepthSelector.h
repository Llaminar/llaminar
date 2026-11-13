/**
 * @file MTPSurvivalDepthSelector.h
 * @brief Pure bounded proposal ranking from exact same-exposure score intervals.
 *
 * The caller supplies actual verified-progress evidence, admitted bounds and a
 * borrowed immutable cost profile. A typed confirmation transition then requires
 * matching targets on consumed disjoint windows before an actual trial. These
 * helpers own no persistent controller state or cost
 * table. Every nonhold result is a proposal requiring an actual reversible trial,
 * including SupportedProbe; no projected interval certifies runtime throughput.
 */
#pragma once

#include "MTPSurvivalScoreInterval.h"
#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MTP_SELECTOR_HD __host__ __device__ __forceinline__
#else
#define LLAMINAR_MTP_SELECTOR_HD inline
#endif

namespace llaminar2
{
    /** @brief Optional borrowed failed-target comparison; not a live phase word. */
    enum class MTPRejectedVerifiedComparisonKind : uint32_t { None, RejectedTarget };

    /**
     * @brief One failed target's actual verified-state sample, never emitted Y.
     *
     * None requires zero payload. RejectedTarget requires a different admitted
     * target and positive runs/progress in[runs,(target+1)*runs]. The gate owner
     * projects its real rejected reference here; Pending owns no new proposal.
     */
    struct MTPRejectedVerifiedComparison
    {
        MTPRejectedVerifiedComparisonKind kind = MTPRejectedVerifiedComparisonKind::None;
        uint32_t target_depth = 0;
        uint32_t reference_runs = 0;
        uint64_t reference_verified_progress = 0;
    };

    /** @brief Pure proposal, requiring the real gate for every nonhold action. */
    enum class MTPSurvivalProposalKind : uint32_t { Hold, SupportedProbe, UnresolvedProbe };

    /** @brief Invalid input never returns a usable hold or partial proposal. */
    enum class MTPSurvivalSelectorError : uint32_t
    {
        None, InvalidBounds, InvalidEvidence, InvalidCost, InvalidRejectedComparison,
    };

    /** @brief A proposal and its evidence; no next-depth mutation is performed. */
    struct MTPSurvivalDepthProposal
    {
        MTPSurvivalSelectorError error = MTPSurvivalSelectorError::InvalidEvidence;
        MTPSurvivalProposalKind kind = MTPSurvivalProposalKind::Hold;
        uint32_t target_depth = 0;
        MTPVerifiedProgressInterval progress{};

        /** @return Whether complete evidence, bounds, costs and cache validated. */
        LLAMINAR_MTP_SELECTOR_HD bool valid() const noexcept
        {
            return error == MTPSurvivalSelectorError::None;
        }
    };

    /**
     * @brief Deterministic tie order shared by supported and unresolved ranks.
     * @param current Actual incumbent depth. @param candidate New candidate.
     * @param candidate_cost Candidate's borrowed positive immutable cost.
     * @param retained Current proposal depth. @param retained_cost Its cost.
     * @return Whether candidate wins nearest-depth, lower-cost, smaller-depth order.
     */
    LLAMINAR_MTP_SELECTOR_HD bool preferMTPPrefixProposalTie(uint32_t current,
        uint32_t candidate, uint32_t candidate_cost, uint32_t retained,
        uint32_t retained_cost) noexcept
    {
        const uint32_t distance = candidate > current ? candidate-current : current-candidate;
        const uint32_t retained_distance = retained > current ? retained-current : current-retained;
        return distance < retained_distance ||
            (distance == retained_distance && (candidate_cost < retained_cost ||
             (candidate_cost == retained_cost && candidate < retained)));
    }

    /**
     * @brief Consider every admitted depth; filter only one authenticated target.
     * @tparam CostProfile Borrowed immutable authority with uint32_t costAt(uint32_t).
     * @param evidence Actual same-depth supported R/A/Z/F window.
     * @param minimum_depth Admitted positive minimum within1..15.
     * @param maximum_depth Admitted maximum within1..15, including current depth.
     * @param costs Existing cost authority; every admitted depth must be positive.
     * @param rejected Optional actual failed verified-state sample from gate owner.
     * @return SupportedProbe with greatest lower/C; else nearest UnresolvedProbe;
     *         else incumbent Hold. All errors return zero depth/progress.
     *
     * Supported score ties use nearest depth, lower cost, smaller depth. Unresolved
     * intervals use that tie order directly, never the optimistic bound magnitude.
     * A rejected target alone is blocked when current V/(R*Ccurrent) is at least
     * its failed actual V/(referenceR*Ctarget); other candidates remain eligible.
     * The same authoritative costs serve filtering and ranking. Denominator
     * products uint32R*uint32C fit uint64; the Euclidean comparator avoids larger
     * cross-products. No arrays, allocations, clocks, mutation or policy state.
     */
    template <typename CostProfile>
    LLAMINAR_MTP_SELECTOR_HD MTPSurvivalDepthProposal selectMTPSurvivalDepthProposal(
        const MTPPrefixWindowEvidence &evidence, uint32_t minimum_depth,
        uint32_t maximum_depth, const CostProfile &costs,
        const MTPRejectedVerifiedComparison &rejected = {}) noexcept
    {
        static_assert(std::is_same_v<decltype(costs.costAt(uint32_t{})),uint32_t>,
            "Borrowed cost authority must return uint32_t; zero means unavailable");
        if (minimum_depth < 1 || maximum_depth > 15 || minimum_depth > maximum_depth ||
            evidence.depth < minimum_depth || evidence.depth > maximum_depth)
            return {MTPSurvivalSelectorError::InvalidBounds};
        const uint32_t current_cost = costs.costAt(evidence.depth);
        if (current_cost == 0)
            return {MTPSurvivalSelectorError::InvalidCost};
        const auto incumbent = evaluateMTPPrefixSurvivalScore(
            evidence,evidence.depth,current_cost,current_cost);
        if (!incumbent.valid())
            return {MTPSurvivalSelectorError::InvalidEvidence};

        bool block_target = false;
        if (rejected.kind == MTPRejectedVerifiedComparisonKind::None)
        {
            if (rejected.target_depth != 0 || rejected.reference_runs != 0 ||
                rejected.reference_verified_progress != 0)
                return {MTPSurvivalSelectorError::InvalidRejectedComparison};
        }
        else if (rejected.kind == MTPRejectedVerifiedComparisonKind::RejectedTarget)
        {
            if (rejected.target_depth < minimum_depth || rejected.target_depth > maximum_depth ||
                rejected.target_depth == evidence.depth || rejected.reference_runs == 0 ||
                rejected.reference_verified_progress < rejected.reference_runs ||
                rejected.reference_verified_progress >
                    static_cast<uint64_t>(rejected.reference_runs)*(rejected.target_depth+1))
                return {MTPSurvivalSelectorError::InvalidRejectedComparison};
            const uint32_t target_cost = costs.costAt(rejected.target_depth);
            if (target_cost == 0)
                return {MTPSurvivalSelectorError::InvalidCost};
            block_target = compareMTPUtilityFractions(incumbent.progress.incumbent,
                static_cast<uint64_t>(evidence.runs)*current_cost,
                rejected.reference_verified_progress,
                static_cast<uint64_t>(rejected.reference_runs)*target_cost) != MTPUtilityOrder::Less;
        }
        else
            return {MTPSurvivalSelectorError::InvalidRejectedComparison};

        MTPSurvivalDepthProposal result{MTPSurvivalSelectorError::None,
            MTPSurvivalProposalKind::Hold,evidence.depth,incumbent.progress};
        uint32_t retained_cost = current_cost;
        for (uint32_t candidate = minimum_depth; candidate <= maximum_depth; ++candidate)
        {
            const uint32_t candidate_cost = costs.costAt(candidate);
            // Validate even a blocked target; a cache cannot hide missing costs.
            if (candidate_cost == 0)
                return {MTPSurvivalSelectorError::InvalidCost};
            if (candidate == evidence.depth ||
                (block_target && candidate == rejected.target_depth))
                continue;
            const auto score = evaluateMTPPrefixSurvivalScore(
                evidence,candidate,current_cost,candidate_cost);
            if (!score.valid())
                return {MTPSurvivalSelectorError::InvalidEvidence};
            bool retain = false;
            if (score.classification == MTPSurvivalScoreClass::Supported)
            {
                if (result.kind != MTPSurvivalProposalKind::SupportedProbe)
                    retain = true;
                else
                {
                    const auto order = compareMTPUtilityFractions(score.progress.lower,
                        candidate_cost,result.progress.lower,retained_cost);
                    retain = order == MTPUtilityOrder::Greater ||
                        (order == MTPUtilityOrder::Equal && preferMTPPrefixProposalTie(evidence.depth,
                            candidate,candidate_cost,result.target_depth,retained_cost));
                }
                if (retain) result.kind = MTPSurvivalProposalKind::SupportedProbe;
            }
            else if (score.classification == MTPSurvivalScoreClass::Unresolved &&
                result.kind != MTPSurvivalProposalKind::SupportedProbe)
            {
                retain = result.kind == MTPSurvivalProposalKind::Hold ||
                    preferMTPPrefixProposalTie(evidence.depth,candidate,candidate_cost,
                        result.target_depth,retained_cost);
                if (retain) result.kind = MTPSurvivalProposalKind::UnresolvedProbe;
            }
            if (retain)
            {
                result.target_depth = candidate;
                result.progress = score.progress;
                retained_cost = candidate_cost;
            }
        }
        return result;
    }

    /**
     * @brief Sole confirmation history for completed, disjoint cost-policy windows.
     *
     * Host owns this value directly. Resident control projects target_depth onto
     * LastRecommendedDepth and completed_windows onto PromotionStreak. Zero
     * streak names the current selector; no duplicated host device mirror exists.
     */
    struct MTPProposalConfirmation
    {
        uint32_t target_depth = 0;
        uint32_t completed_windows = 0;
    };

    /** @brief Confirmation rejects corruption before observation counters change. */
    enum class MTPProposalConfirmationError : uint32_t { None, InvalidPolicy, InvalidState, InvalidTarget };
    /** @brief Only TrialReady permits a scored proposal to reach the real gate. */
    enum class MTPProposalConfirmationAction : uint32_t { Hold, AwaitingConfirmation, TrialReady };
    /** @brief Pure transition result; input history remains unchanged on error. */
    struct MTPProposalConfirmationResult
    {
        MTPProposalConfirmationError error = MTPProposalConfirmationError::None;
        MTPProposalConfirmationAction action = MTPProposalConfirmationAction::Hold;
        MTPProposalConfirmation next{};
        /** @return Whether all history, admitted bounds and target validated. */
        LLAMINAR_MTP_SELECTOR_HD bool valid() const noexcept
        { return error == MTPProposalConfirmationError::None; }
    };

    /**
     * @brief Validate measured history before any real observation mutation.
     * @param state Sole persistent candidate and number of completed windows.
     * @param current Actual current selector. @param minimum Admitted lower bound.
     * @param maximum Admitted upper bound. @param required Admitted consecutive windows.
     * @param pending Whether the independent reversible gate owns an actual trial.
     * @return False for poison, stale selector, overflow-prone streak or pending history.
     */
    LLAMINAR_MTP_SELECTOR_HD bool validMTPProposalConfirmation(
        const MTPProposalConfirmation &state, uint32_t current, uint32_t minimum,
        uint32_t maximum, uint32_t required, bool pending) noexcept
    {
        if (minimum < 1 || maximum > 15 || minimum > maximum || current < minimum ||
            current > maximum || required == 0 || state.completed_windows >= required ||
            state.target_depth < minimum || state.target_depth > maximum)
            return false;
        if (state.completed_windows == 0)
            return state.target_depth == current;
        return !pending && state.target_depth != current;
    }

    /**
     * @brief Confirm one actual completed window, never an overlapping prefix.
     * @param state Sole prior history; copied, never mutated by this helper.
     * @param current Actual selector shared by every frame in the window.
     * @param minimum Admitted lower bound. @param maximum Admitted upper bound.
     * @param required Existing promote_consecutive_windows, unchanged by this policy.
     * @param proposed_target Actual cache-filtered selector result, or current for Hold.
     * @return Hold clears history; a changed candidate starts at one; the required
     *         matching completed window returns TrialReady with empty history.
     *
     * The caller must consume this window's authoritative R/A/Z/F and actual Y
     * exactly once even when AwaitingConfirmation. Pending trials bypass this
     * helper. TrialReady uses only this window as its incumbent utility sample;
     * earlier windows establish target stability, not a synthetic pooled reward.
     */
    LLAMINAR_MTP_SELECTOR_HD MTPProposalConfirmationResult confirmMTPSurvivalProposal(
        const MTPProposalConfirmation &state, uint32_t current, uint32_t minimum,
        uint32_t maximum, uint32_t required, uint32_t proposed_target) noexcept
    {
        if (minimum < 1 || maximum > 15 || minimum > maximum || current < minimum ||
            current > maximum || required == 0)
            return {MTPProposalConfirmationError::InvalidPolicy};
        if (!validMTPProposalConfirmation(state,current,minimum,maximum,required,false))
            return {MTPProposalConfirmationError::InvalidState};
        if (proposed_target < minimum || proposed_target > maximum)
            return {MTPProposalConfirmationError::InvalidTarget};
        const MTPProposalConfirmation empty{current,0};
        if (proposed_target == current)
            return {MTPProposalConfirmationError::None,MTPProposalConfirmationAction::Hold,empty};
        const uint32_t count = state.target_depth == proposed_target
            ? state.completed_windows + 1 : 1;
        if (count == required)
            return {MTPProposalConfirmationError::None,MTPProposalConfirmationAction::TrialReady,empty};
        return {MTPProposalConfirmationError::None,MTPProposalConfirmationAction::AwaitingConfirmation,
            {proposed_target,count}};
    }
}
#undef LLAMINAR_MTP_SELECTOR_HD
