/**
 * @file MTPReversibleProposalGate.h
 * @brief Shared host/device experiment gate around an actual depth proposal.
 *
 * The CPU and resident controllers borrow actual scored proposals. This
 * allocation-free gate compares actual published verifier-state progress from
 * consecutive eligible windows using an immutable, admission-authenticated
 * transaction cost profile. It is an estimated utility comparison, never a device clock or an
 * identical-history counterfactual. Its five words belong to the real controller;
 * diagnostic storage and host mirrors must never serve as live state. Every
 * non-hold scored proposal starts an actual reversible trial. Actual response
 * emission remains a separate window quantity. The publication owner validates
 * each eligible state-progress input; this gate borrows the cumulative validated
 * quantity and stores only its reference sample, never a second current window.
 */
#pragma once

#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MTP_PROPOSAL_HD __host__ __device__ __forceinline__
#else
#define LLAMINAR_MTP_PROPOSAL_HD inline
#endif

namespace llaminar2
{
    /** @brief Meaning of the single saved reference sample. */
    enum class MTPReversibleProposalPhase : uint32_t
    {
        Clear = 0,
        ProbePending = 1,        ///< Reference is the incumbent before departure.
        RejectedComparison = 2, ///< Reference is the failed target after return.
    };

    /** @brief Five live words: one verified reference and actual current emission. */
    struct MTPReversibleProposalState
    {
        MTPReversibleProposalPhase phase = MTPReversibleProposalPhase::Clear;
        uint32_t reference_depth = 0;
        uint32_t reference_runs = 0;
        uint32_t reference_verified_progress = 0; ///< Actual published states in the saved sample.
        uint32_t window_emitted = 0; ///< Actual output Y; never a second V accumulator.
    };
    static_assert(sizeof(MTPReversibleProposalState) == 5 * sizeof(uint32_t));

    /** @brief Inactive preserves ordinary ownership; Observe never starts a probe. */
    enum class MTPReversibleProposalMode : uint32_t { Inactive, Observe, Dynamic };

    /**
     * @brief Transient authority of a cost-scored proposal; never live state.
     *
     * SupportedScore has a conservative same-input utility advantage.
     * UnresolvedScore has only a positive upper-bound opportunity and needs
     * real feedback. Neither origin permits a direct selector change. Pending
     * owns comparison before any further scored proposal is considered.
     */
    enum class MTPDepthProposalOrigin : uint32_t
    {
        Unevaluated,
        SupportedScore,
        UnresolvedScore,
        Hold,
    };

    /**
     * @brief Borrowed cumulative verified state progress, distinct from emission.
     *
     * The actual publication owner must validate every eligible transaction's
     * published rows before appending its canonical R/A counters. Consequently
     * their transient R+A is this quantity; no duplicate live V bank is stored.
     * This tag does not grant authority to infer publication from a trace or
     * unvalidated acceptance count.
     */
    struct MTPVerifiedProgress
    {
        uint32_t states = 0;
    };

    /** @brief Immutable admission inputs, with no new cadence or confidence knob. */
    struct MTPReversibleProposalPolicy
    {
        MTPReversibleProposalMode mode = MTPReversibleProposalMode::Inactive;
        uint32_t minimum_depth = 1;
        uint32_t maximum_depth = 15;
        uint32_t minimum_samples = 1;
    };

    /**
     * @brief Actual emission plus a borrowed post-append verified progress view.
     *
     * window_runs is authoritative eligible R AFTER this transaction. Append
     * owns only newly_emitted_tokens and ignores verified_progress. Decision
     * borrows cumulative verified_progress, validated by the real publication
     * owner, and never substitutes emitted Y for published V. Stopped/clipped
     * transactions are ineligible; rollback alone remains eligible. A Pending
     * trial ignores supplied proposal fields until comparison completes.
     */
    struct MTPReversibleProposalInput
    {
        uint32_t executed_depth = 0;
        uint32_t window_runs = 0;
        uint32_t newly_emitted_tokens = 0;
        bool eligible = false;
        MTPDepthProposalOrigin proposal_origin = MTPDepthProposalOrigin::Unevaluated;
        uint32_t parent_target = 0;
        MTPVerifiedProgress verified_progress;
    };

    /** @brief Exact event; production probe/keep/return counters remain separate. */
    enum class MTPReversibleProposalAction : uint32_t
    {
        Inactive,
        Ineligible,
        Accumulating,
        ParentHold,
        WouldProbe,
        ProbeStarted,
        ProbeKept,
        ProbeReturned,
        RejectedTargetBlocked,
    };

    /** @brief Fatal contract errors; an error never mutates any gate word. */
    enum class MTPReversibleProposalError : uint32_t
    {
        None, InvalidPolicy, InvalidState, InvalidObservation, InvalidCost, CounterOverflow
    };

    /** @brief Ordering of two exact estimated utilities. */
    enum class MTPUtilityOrder : int { Less = -1, Equal = 0, Greater = 1 };

    /**
     * @brief Compare nonnegative fractions without overflowing a cross-product.
     * @param left_numerator First numerator, including zero.
     * @param left_denominator First strictly positive denominator.
     * @param right_numerator Second numerator, including zero.
     * @param right_denominator Second strictly positive denominator.
     * @return Exact order; positive denominators are required by the typed caller.
     *
     * Quotients decide immediately when unequal. Otherwise the remainders are
     * compared through reciprocal fractions, reversing the order each time. The
     * Euclidean algorithm terminates and never multiplies the two denominators.
     */
    LLAMINAR_MTP_PROPOSAL_HD MTPUtilityOrder compareMTPUtilityFractions(
        uint64_t left_numerator, uint64_t left_denominator,
        uint64_t right_numerator, uint64_t right_denominator)
    {
        bool reversed = false;
        for (;;)
        {
            const uint64_t left_quotient = left_numerator / left_denominator;
            const uint64_t right_quotient = right_numerator / right_denominator;
            if (left_quotient != right_quotient)
            {
                const bool less = left_quotient < right_quotient;
                return less != reversed ? MTPUtilityOrder::Less : MTPUtilityOrder::Greater;
            }
            const uint64_t left_remainder = left_numerator % left_denominator;
            const uint64_t right_remainder = right_numerator % right_denominator;
            if (left_remainder == 0 || right_remainder == 0)
            {
                if (left_remainder == right_remainder)
                    return MTPUtilityOrder::Equal;
                const bool less = left_remainder == 0;
                return less != reversed ? MTPUtilityOrder::Less : MTPUtilityOrder::Greater;
            }
            left_numerator = left_denominator;
            left_denominator = left_remainder;
            right_numerator = right_denominator;
            right_denominator = right_remainder;
            reversed = !reversed;
        }
    }

    /**
     * @brief Compare verified V/(R*C), whose denominator product fits exactly in uint64.
     * @param left_verified_progress First actual verified-state progress sum.
     * @param left_runs First strictly positive eligible run count.
     * @param left_cost First strictly positive immutable transaction cost.
     * @param right_verified_progress Second actual verified-state progress sum.
     * @param right_runs Second strictly positive eligible run count.
     * @param right_cost Second strictly positive immutable transaction cost.
     * @return Exact utility order without a potentially 96-bit triple product.
     */
    LLAMINAR_MTP_PROPOSAL_HD MTPUtilityOrder compareMTPVerifiedProgressUtility(
        uint32_t left_verified_progress, uint32_t left_runs, uint32_t left_cost,
        uint32_t right_verified_progress, uint32_t right_runs, uint32_t right_cost)
    {
        return compareMTPUtilityFractions(left_verified_progress,
            static_cast<uint64_t>(left_runs) * left_cost,
            right_verified_progress, static_cast<uint64_t>(right_runs) * right_cost);
    }

    /** @brief Selector/window effects; only actual changes reset normal cooldown. */
    struct MTPReversibleProposalResult
    {
        MTPReversibleProposalError error = MTPReversibleProposalError::None;
        MTPReversibleProposalAction action = MTPReversibleProposalAction::Accumulating;
        uint32_t next_depth = 0;
        bool consume_window = false;
        bool reset_cooldown = false;

        /** @return Whether the gate accepted this input without a fatal error. */
        LLAMINAR_MTP_PROPOSAL_HD bool valid() const
        {
            return error == MTPReversibleProposalError::None;
        }
    };

    /**
     * @brief Validate a phase-specific reference without reading a cost table.
     * @param state Controller-owned five-word state.
     * @param policy Admitted bounds/mode/cadence.
     * @param current_depth Actual sole selector at the transaction boundary.
     * @return Whether the state has one coherent reference meaning.
     */
    LLAMINAR_MTP_PROPOSAL_HD bool validMTPReversibleProposalState(
        const MTPReversibleProposalState &state,
        const MTPReversibleProposalPolicy &policy, uint32_t current_depth)
    {
        if (state.phase == MTPReversibleProposalPhase::Clear)
            return state.reference_depth == 0 && state.reference_runs == 0 &&
                   state.reference_verified_progress == 0;
        if ((state.phase != MTPReversibleProposalPhase::ProbePending &&
             state.phase != MTPReversibleProposalPhase::RejectedComparison) ||
            policy.mode != MTPReversibleProposalMode::Dynamic ||
            state.reference_depth < policy.minimum_depth ||
            state.reference_depth > policy.maximum_depth ||
            state.reference_depth == current_depth || state.reference_runs < policy.minimum_samples ||
            state.reference_verified_progress < state.reference_runs)
            return false;
        return static_cast<uint64_t>(state.reference_verified_progress) <=
            static_cast<uint64_t>(state.reference_runs) * (state.reference_depth + 1);
    }

    /**
     * @brief Append actual Y before pending completion or scored-proposal evaluation.
     * @param policy Admitted mode/bounds/minimum samples.
     * @param state Sole live gate words; modified only after successful validation.
     * @param input Actual eligible transaction and post-append canonical window R.
     * @return Inactive/Ineligible without writes, Accumulating on append, or error.
     *
     * This stage owns only window_emitted. The real controller already appended
     * its canonical R/A/Z/F. Rollback alone is eligible; stopped and clipped
     * transactions remain physical work but do not change the eligible sample.
     * This helper does not validate model publication; that stays at its owner.
     */
    LLAMINAR_MTP_PROPOSAL_HD MTPReversibleProposalResult appendMTPReversibleProposalEmission(
        const MTPReversibleProposalPolicy &policy, MTPReversibleProposalState &state,
        const MTPReversibleProposalInput &input)
    {
        MTPReversibleProposalResult result;
        result.next_depth = input.executed_depth;
        if (policy.mode != MTPReversibleProposalMode::Inactive &&
            policy.mode != MTPReversibleProposalMode::Observe &&
            policy.mode != MTPReversibleProposalMode::Dynamic)
            result.error = MTPReversibleProposalError::InvalidPolicy;
        if (!result.valid())
            return result;
        if (policy.mode == MTPReversibleProposalMode::Inactive)
        {
            result.action = MTPReversibleProposalAction::Inactive;
            return result;
        }
        if (policy.minimum_depth == 0 || policy.maximum_depth > 15 ||
            policy.minimum_depth > policy.maximum_depth || policy.minimum_samples == 0)
        {
            result.error = MTPReversibleProposalError::InvalidPolicy;
            return result;
        }
        if (input.executed_depth < policy.minimum_depth ||
            input.executed_depth > policy.maximum_depth)
            result.error = MTPReversibleProposalError::InvalidObservation;
        else if (!validMTPReversibleProposalState(state, policy, input.executed_depth))
            result.error = MTPReversibleProposalError::InvalidState;
        if (!result.valid())
            return result;
        if (!input.eligible)
        {
            result.action = MTPReversibleProposalAction::Ineligible;
            return result;
        }
        if (input.window_runs == 0 || input.newly_emitted_tokens == 0 ||
            input.newly_emitted_tokens > input.executed_depth + 1)
        {
            result.error = MTPReversibleProposalError::InvalidObservation;
            return result;
        }
        if (input.newly_emitted_tokens > UINT32_MAX - state.window_emitted)
        {
            result.error = MTPReversibleProposalError::CounterOverflow;
            return result;
        }
        auto next = state;
        next.window_emitted += input.newly_emitted_tokens;
        if (next.window_emitted < input.window_runs || static_cast<uint64_t>(next.window_emitted) >
            static_cast<uint64_t>(input.window_runs) * (input.executed_depth + 1))
        {
            result.error = MTPReversibleProposalError::InvalidObservation;
            return result;
        }
        state = next;
        return result;
    }

    /**
     * @brief Readiness of an already validated pending probe, before new proposal work.
     * @param policy Admitted existing minimum_samples.
     * @param state Actual phase; callers must validate it through append first.
     * @param window_runs Existing eligible R, including the just-appended sample.
     * @return Whether a pending comparison must precede any new proposal evaluation.
     */
    LLAMINAR_MTP_PROPOSAL_HD bool mtpReversibleProposalPendingReady(
        const MTPReversibleProposalPolicy &policy,
        const MTPReversibleProposalState &state, uint32_t window_runs)
    {
        return state.phase == MTPReversibleProposalPhase::ProbePending &&
               window_runs >= policy.minimum_samples;
    }

    /**
     * @brief Decide from borrowed verified V, appended actual Y and immutable costs.
     * @tparam CostProfile Immutable admission-bound type with uint32_t costAt(depth)
     * returning a positive cost or zero for an unavailable depth. Host and device
     * use the same integers; the profile owns its storage and complete scope.
     * @param policy Admitted mode, bounds and existing minimum_samples.
     * @param state Sole live five-word owner, updated atomically only on success.
     * @param input Actual post-append window/proposal; newly_emitted_tokens is ignored.
     * @param profile Immutable transaction costs; no clocks or profile mutation.
     * @return Typed action, selector and reset effects, or a fatal typed error.
     *
     * The caller must apply consume_window to its existing R/A/Z/F bank exactly
     * once, after this function returns; this gate owns only the matching Y word.
     * Call appendMTPReversibleProposalEmission exactly once before this method.
     * Pending comparison precedes new proposals even during cooldown. The
     * reference stores actual verified V; current V is borrowed, while actual
     * window Y is independently retained and consumed with R/A/Z/F. A keep
     * consumes one trial without a second proposal. A return caches the failed
     * target's verified rate and resets cooldown as a real selector change.
     * Every scored move in either direction starts a real trial. A hold consumes
     * actual Y while preserving a cached reference. Ranking should skip a vetoed
     * same target; this function defensively enforces that unchanged comparison.
     */
    template<class CostProfile>
    LLAMINAR_MTP_PROPOSAL_HD MTPReversibleProposalResult decideMTPReversibleProposalGate(
        const MTPReversibleProposalPolicy &policy, MTPReversibleProposalState &state,
        const MTPReversibleProposalInput &input, const CostProfile &profile)
    {
        static_assert(std::is_same_v<decltype(profile.costAt(uint32_t{})), uint32_t>,
            "immutable cost profile must return exact uint32_t costs");
        MTPReversibleProposalResult result;
        result.next_depth = input.executed_depth;
        if (policy.mode != MTPReversibleProposalMode::Inactive &&
            policy.mode != MTPReversibleProposalMode::Observe &&
            policy.mode != MTPReversibleProposalMode::Dynamic)
            result.error = MTPReversibleProposalError::InvalidPolicy;
        if (!result.valid())
            return result;
        if (policy.mode == MTPReversibleProposalMode::Inactive)
        {
            result.action = MTPReversibleProposalAction::Inactive;
            return result;
        }
        if (policy.minimum_depth == 0 || policy.maximum_depth > 15 ||
            policy.minimum_depth > policy.maximum_depth || policy.minimum_samples == 0)
        {
            result.error = MTPReversibleProposalError::InvalidPolicy;
            return result;
        }
        if (input.executed_depth < policy.minimum_depth ||
            input.executed_depth > policy.maximum_depth)
            result.error = MTPReversibleProposalError::InvalidObservation;
        else if (!validMTPReversibleProposalState(state, policy, input.executed_depth))
            result.error = MTPReversibleProposalError::InvalidState;
        if (!result.valid())
            return result;
        if (!input.eligible)
        {
            result.action = MTPReversibleProposalAction::Ineligible;
            return result;
        }
        if (input.window_runs == 0 || state.window_emitted < input.window_runs ||
            static_cast<uint64_t>(state.window_emitted) >
                static_cast<uint64_t>(input.window_runs) * (input.executed_depth + 1) ||
            input.verified_progress.states < input.window_runs ||
            static_cast<uint64_t>(input.verified_progress.states) >
                static_cast<uint64_t>(input.window_runs) * (input.executed_depth + 1))
        {
            result.error = MTPReversibleProposalError::InvalidObservation;
            return result;
        }
        auto next = state;
        if (next.phase == MTPReversibleProposalPhase::ProbePending)
        {
            if (input.window_runs < policy.minimum_samples)
            {
                state = next;
                return result;
            }
            const uint32_t current_cost = profile.costAt(input.executed_depth);
            const uint32_t reference_cost = profile.costAt(next.reference_depth);
            if (current_cost == 0 || reference_cost == 0)
            {
                result.error = MTPReversibleProposalError::InvalidCost;
                return result;
            }
            if (compareMTPVerifiedProgressUtility(input.verified_progress.states, input.window_runs, current_cost,
                    next.reference_verified_progress, next.reference_runs, reference_cost) == MTPUtilityOrder::Greater)
            {
                next = {};
                result.action = MTPReversibleProposalAction::ProbeKept;
            }
            else
            {
                result.next_depth = next.reference_depth;
                next.phase = MTPReversibleProposalPhase::RejectedComparison;
                next.reference_depth = input.executed_depth;
                next.reference_runs = input.window_runs;
                next.reference_verified_progress = input.verified_progress.states;
                next.window_emitted = 0;
                result.action = MTPReversibleProposalAction::ProbeReturned;
                result.reset_cooldown = true;
            }
            result.consume_window = true;
            state = next;
            return result;
        }
        if (input.proposal_origin == MTPDepthProposalOrigin::Unevaluated)
        {
            state = next;
            return result;
        }
        if (input.window_runs < policy.minimum_samples ||
            input.parent_target < policy.minimum_depth || input.parent_target > policy.maximum_depth)
        {
            result.error = MTPReversibleProposalError::InvalidObservation;
            return result;
        }
        const bool scored = input.proposal_origin == MTPDepthProposalOrigin::SupportedScore ||
                            input.proposal_origin == MTPDepthProposalOrigin::UnresolvedScore;
        const bool hold = input.proposal_origin == MTPDepthProposalOrigin::Hold;
        if ((!scored && !hold) ||
            (hold && input.parent_target != input.executed_depth) ||
            (scored && input.parent_target == input.executed_depth))
        {
            result.error = MTPReversibleProposalError::InvalidObservation;
            return result;
        }
        result.consume_window = true;
        if (hold)
        {
            next.window_emitted = 0;
            result.action = MTPReversibleProposalAction::ParentHold;
            state = next;
            return result;
        }
        if (policy.mode == MTPReversibleProposalMode::Observe)
        {
            next.window_emitted = 0;
            result.action = MTPReversibleProposalAction::WouldProbe;
            state = next;
            return result;
        }
        const uint32_t current_cost = profile.costAt(input.executed_depth);
        const uint32_t target_cost = profile.costAt(input.parent_target);
        if (current_cost == 0 || target_cost == 0)
        {
            result.error = MTPReversibleProposalError::InvalidCost;
            return result;
        }
        if (next.phase == MTPReversibleProposalPhase::RejectedComparison &&
            input.parent_target == next.reference_depth &&
            compareMTPVerifiedProgressUtility(input.verified_progress.states, input.window_runs, current_cost,
                next.reference_verified_progress, next.reference_runs, target_cost) != MTPUtilityOrder::Less)
        {
            next.window_emitted = 0;
            result.action = MTPReversibleProposalAction::RejectedTargetBlocked;
            state = next;
            return result;
        }
        next.phase = MTPReversibleProposalPhase::ProbePending;
        next.reference_depth = input.executed_depth;
        next.reference_runs = input.window_runs;
        next.reference_verified_progress = input.verified_progress.states;
        next.window_emitted = 0;
        result.action = MTPReversibleProposalAction::ProbeStarted;
        result.next_depth = input.parent_target;
        result.reset_cooldown = true;
        state = next;
        return result;
    }
    /**
     * @brief Atomic convenience composition of append and decision for one input.
     * @tparam CostProfile Immutable uint32_t costAt(depth) injection.
     * @param policy Admitted gate policy.
     * @param state Sole live state, unchanged on either stage's error.
     * @param input Actual observation plus already-computed scored proposal.
     * @param profile Immutable admission-authenticated costs.
     * @return The same event as separated append/decide, without double counting.
     *
     * Production may use the separate stages to run its scored proposal
     * between them only when phase is not ProbePending. This convenience method
     * never runs a selector and never manufactures proposal feedback.
     */
    template<class CostProfile>
    LLAMINAR_MTP_PROPOSAL_HD MTPReversibleProposalResult applyMTPReversibleProposalGate(
        const MTPReversibleProposalPolicy &policy, MTPReversibleProposalState &state,
        const MTPReversibleProposalInput &input, const CostProfile &profile)
    {
        auto next = state;
        auto result = appendMTPReversibleProposalEmission(policy, next, input);
        if (!result.valid() || result.action == MTPReversibleProposalAction::Inactive ||
            result.action == MTPReversibleProposalAction::Ineligible)
            return result;
        result = decideMTPReversibleProposalGate(policy, next, input, profile);
        if (result.valid())
            state = next;
        return result;
    }

}
#undef LLAMINAR_MTP_PROPOSAL_HD
