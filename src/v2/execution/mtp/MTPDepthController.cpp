/**
 * @file MTPDepthController.cpp
 * @brief Deterministic rolling-window MTP depth decisions for host execution.
 *
 * Configuration resolves automatic thresholds before any window is evaluated.
 * GPU callers seal the same topology-bound values and trained initial depth
 * into the device ABI instead of consulting this host controller for live
 * device-owned decisions. Startup policy is resolved here once for both owners.
 * A measured scope ranks every admitted depth using exact prefix-survival
 * bounds, confirms targets on disjoint completed windows, then prices actual
 * reversible trials in verified-state progress. Deferred windows consume their
 * real observations without changing the selector, cooldown or failed reference.
 * Physical emitted outputs retain their independent publication authority.
 */
#include "MTPDepthController.h"
#include "MTPDepthLearnedPolicy.h"
#include "MTPSurvivalDepthSelector.h"

#include <algorithm>
#include <optional>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        /** @brief Host diagnostic projection of the shared learned match. */
        struct MTPGeneratedDepthPolicyMatch
        {
            int target_depth = 1;
            MTPDepthDecisionReason reason = MTPDepthDecisionReason::Hold;
        };

        /**
         * @brief Project the single actual cost-policy proposal into the trial gate.
         * @param decision Result of the shared survival-score selector.
         * @return Typed evidence for a real trial, or an explicit hold.
         */
        MTPDepthProposalOrigin proposalOrigin(const MTPDepthDecision &decision)
        {
            switch (decision.reason)
            {
            case MTPDepthDecisionReason::SurvivalSupportedProbe:
                return MTPDepthProposalOrigin::SupportedScore;
            case MTPDepthDecisionReason::SurvivalUnresolvedProbe:
                return MTPDepthProposalOrigin::UnresolvedScore;
            default:
                return MTPDepthProposalOrigin::Hold;
            }
        }

        /**
         * @brief Rank every admitted depth using one actual eligible window.
         * @param config Admitted bounds, without a second cadence authority.
         * @param depth Sole current selector used by every observation in window.
         * @param window Existing R/A/Z/F whose publication contract was checked.
         * @param state Sole trial/rejection owner; pending trials bypass selection.
         * @param costs Immutable costs for this complete measured execution scope.
         * @return A proposal, never a prediction of end-to-end speed.
         * @throws std::invalid_argument For malformed/overflowing observations.
         */
        MTPSurvivalDepthProposal survivalProposal(
            const MTPDepthPolicyConfig &config, int depth, const MTPDepthWindow &window,
            const MTPReversibleProposalState &state, const MTPDepthCostProfile &costs)
        {
            if (window.verifier_runs > UINT32_MAX || window.zero_accepts > UINT32_MAX ||
                window.full_accepts > UINT32_MAX || window.accepted_draft_tokens > UINT32_MAX ||
                window.verifier_runs > UINT32_MAX - window.accepted_draft_tokens)
                throw std::invalid_argument("MTP verified-progress window exceeds its controller capacity");
            MTPRejectedVerifiedComparison rejected;
            if (state.phase == MTPReversibleProposalPhase::RejectedComparison)
                rejected = {.kind = MTPRejectedVerifiedComparisonKind::RejectedTarget,
                    .target_depth = state.reference_depth, .reference_runs = state.reference_runs,
                    .reference_verified_progress = state.reference_verified_progress};
            const auto proposal = selectMTPSurvivalDepthProposal(
                {.scope = MTPPrefixEvidenceScope::NonStoppedUnclippedVerifiedState,
                 .depth = static_cast<uint32_t>(depth),
                 .runs = static_cast<uint32_t>(window.verifier_runs),
                 .accepted = window.accepted_draft_tokens,
                 .zero = static_cast<uint32_t>(window.zero_accepts),
                 .full = static_cast<uint32_t>(window.full_accepts)},
                static_cast<uint32_t>(config.min_depth), static_cast<uint32_t>(config.max_depth),
                costs, rejected);
            if (!proposal.valid())
                throw std::invalid_argument("invalid measured MTP survival-score proposal");
            return proposal;
        }

        /**
         * @brief Return the first generated rule that matches this window.
         *
         * Rules are ordered by the generator.  They are intentionally advisory:
         * callers still clamp to request bounds and can ignore the match when
         * the dynamic policy is disabled.
         */
        std::optional<MTPGeneratedDepthPolicyMatch> matchGeneratedDepthPolicy(
            const MTPDepthPolicyConfig &config,
            MTPVerifyMode verify_mode,
            int current_depth,
            const MTPDepthWindow &window)
        {
            const auto match = matchMTPLearnedDepthPolicy(
                {.enabled = config.use_generated_policy && config.mode == MTPDepthPolicyMode::Dynamic,
                 .backend = config.backend, .model_class = config.model_class,
                 .verify_mode = verify_mode, .scope = config.scope},
                current_depth,
                {.attempted = window.attempted_draft_tokens,
                 .accepted = window.accepted_draft_tokens, .runs = window.verifier_runs,
                 .zero_accepts = window.zero_accepts, .full_accepts = window.full_accepts});
            if (!match.matched)
                return std::nullopt;
            return MTPGeneratedDepthPolicyMatch{
                .target_depth = std::clamp(current_depth + match.depth_delta,
                                          config.min_depth, config.max_depth),
                .reason = match.depth_delta > 0 ? MTPDepthDecisionReason::GeneratedPolicyPromote
                    : match.depth_delta < 0 ? MTPDepthDecisionReason::GeneratedPolicyDemote
                                           : MTPDepthDecisionReason::GeneratedPolicyHold};
        }

        /**
         * @brief Return the explicit measured warm start for this request class.
         *
         * A depth-four hold on one prompt does not make depth four a better
         * starting point than depth three across the training workload. The
         * trainer owns that separate economic decision. Do not infer it from
         * the deepest transition row. Explicit caller bounds still apply; an
         * untrained domain uses the ordinary portable controller initialization.
         * @return The measured startup depth, or -1 if no bounded choice exists.
         */
        int resolveGeneratedInitialDepth(
            const MTPDepthPolicyConfig &config,
            MTPVerifyMode verify_mode)
        {
            if (!config.use_generated_policy ||
                config.mode != MTPDepthPolicyMode::Dynamic)
            {
                return -1;
            }

            return resolveMTPLearnedStartup(
                {.enabled = true, .backend = config.backend, .model_class = config.model_class,
                 .verify_mode = verify_mode, .scope = config.scope},
                config.min_depth, config.max_depth);
        }

        /** @brief Whether the live depth is the measured admission winner. */
        bool isGeneratedBestDepth(
            const MTPDepthPolicyConfig &config,
            MTPVerifyMode verify_mode,
            int current_depth)
        {
            return resolveGeneratedInitialDepth(config, verify_mode) == current_depth;
        }
    } // namespace

    int resolveMTPDepthPolicyInitialDepth(
        const MTPDepthPolicyConfig &config,
        int configured_draft_tokens,
        MTPVerifyMode verify_mode)
    {
        if (config.mode == MTPDepthPolicyMode::Fixed)
            return configured_draft_tokens;
        if (config.initial_depth > 0)
            return config.initial_depth;
        if (config.min_depth == 0)
            return 0;

        // A zero authored ceiling means automatic capacity, not an empty
        // learned-policy search interval. Preserve that intent in the caller.
        auto bounded = config;
        if (bounded.max_depth <= 0)
            bounded.max_depth = defaultMTPAdaptiveMaximumDraftDepth();
        if (bounded.min_depth < 0 || bounded.max_depth < bounded.min_depth)
            return -1; // Let the owning admission contract reject invalid bounds.
        const int trained = resolveGeneratedInitialDepth(bounded, verify_mode);
        if (trained > 0)
            return trained;
        return verify_mode == MTPVerifyMode::SpeculativeSampling
            ? config.min_depth
            : std::clamp(2, config.min_depth, bounded.max_depth);
    }

    const char *toString(MTPDepthDecisionReason reason)
    {
        switch (reason)
        {
        case MTPDepthDecisionReason::FixedMode:
            return "fixed_mode";
        case MTPDepthDecisionReason::WindowNotReady:
            return "window_not_ready";
        case MTPDepthDecisionReason::BudgetLimited:
            return "budget_limited";
        case MTPDepthDecisionReason::ModelStopped:
            return "model_stopped";
        case MTPDepthDecisionReason::SurvivalSupportedProbe:
            return "survival_supported_probe";
        case MTPDepthDecisionReason::SurvivalUnresolvedProbe:
            return "survival_unresolved_probe";
        case MTPDepthDecisionReason::SurvivalAwaitingConfirmation:
            return "survival_awaiting_confirmation";
        case MTPDepthDecisionReason::SurvivalHold:
            return "survival_hold";
        case MTPDepthDecisionReason::CooldownActive:
            return "cooldown_active";
        case MTPDepthDecisionReason::PromotionHysteresisActive:
            return "promotion_hysteresis_active";
        case MTPDepthDecisionReason::PromoteFullAcceptRate:
            return "promote_full_accept_rate";
        case MTPDepthDecisionReason::DemoteZeroAcceptRate:
            return "demote_zero_accept_rate";
        case MTPDepthDecisionReason::DemoteLowAcceptanceRate:
            return "demote_low_acceptance_rate";
        case MTPDepthDecisionReason::GeneratedPolicyPromote:
            return "generated_policy_promote";
        case MTPDepthDecisionReason::GeneratedPolicyDemote:
            return "generated_policy_demote";
        case MTPDepthDecisionReason::GeneratedPolicyHold:
            return "generated_policy_hold";
        case MTPDepthDecisionReason::GeneratedBestDepthGraceWindow:
            return "generated_best_depth_grace_window";
        case MTPDepthDecisionReason::ProbeHigherBeforeDemote:
            return "probe_higher_before_demote";
        case MTPDepthDecisionReason::DepthZeroBypass:
            return "depth_zero_bypass";
        case MTPDepthDecisionReason::Hold:
            return "hold";
        case MTPDepthDecisionReason::UtilityProbeStarted:
            return "utility_probe_started";
        case MTPDepthDecisionReason::UtilityProbeKept:
            return "utility_probe_kept";
        case MTPDepthDecisionReason::UtilityProbeReturned:
            return "utility_probe_returned";
        case MTPDepthDecisionReason::UtilityRejectedComparisonHold:
            return "utility_rejected_comparison_hold";
        default:
            return "unknown";
        }
    }

    MTPDepthController::MTPDepthController(
        MTPDepthPolicyConfig config,
        int configured_draft_tokens,
        MTPVerifyMode verify_mode)
    {
        configure(config, configured_draft_tokens, verify_mode);
    }

    void MTPDepthController::configure(
        MTPDepthPolicyConfig config,
        int configured_draft_tokens,
        MTPVerifyMode verify_mode)
    {
        const MTPLearnedDepthPolicyContext identity{
            .backend = config.backend, .model_class = config.model_class,
            .verify_mode = verify_mode, .scope = config.scope};
        if (!identity.valid())
            throw std::invalid_argument("MTP depth policy requires valid immutable learned scope and model keys");
        verify_mode_ = verify_mode;
        // Standalone CPU callers have no GPU domain. Orchestration supplies an
        // already-resolved profile value; explicit values are never replaced.
        config.demote_zero_accept_rate = config.demote_zero_accept_rate.value_or(
            defaultMTPZeroAcceptDemotionRate(MTPDepthDefaultsProfile::Portable));
        if (configured_draft_tokens < 1)
            throw std::invalid_argument("configured MTP draft tokens must be > 0");

        if (config.mode == MTPDepthPolicyMode::Fixed)
        {
            config.min_depth = configured_draft_tokens;
            config.max_depth = configured_draft_tokens;
            config.initial_depth = configured_draft_tokens;
        }
        else if (config.max_depth <= 0)
        {
            // An absent adaptive ceiling means every production-supported
            // depth is eligible.  It must not inherit the unrelated fixed
            // depth, whose default of one would silently disable adaptation.
            config.max_depth = defaultMTPAdaptiveMaximumDraftDepth();
        }
        if (config.initial_depth <= 0)
        {
            config.initial_depth = resolveMTPDepthPolicyInitialDepth(
                config, configured_draft_tokens, verify_mode_);
        }
        if (config.min_depth < 0)
            throw std::invalid_argument("MTP depth policy min_depth must be >= 0");
        if (config.mode == MTPDepthPolicyMode::Fixed && config.min_depth < 1)
            throw std::invalid_argument("MTP fixed depth policy min_depth must be > 0");
        if (config.max_depth < config.min_depth)
            throw std::invalid_argument("MTP depth policy max_depth must be >= min_depth");
        if (config.initial_depth < config.min_depth ||
            config.initial_depth > config.max_depth)
        {
            throw std::invalid_argument("MTP depth policy initial_depth must be within [min_depth, max_depth]");
        }
        if (config.mode != MTPDepthPolicyMode::Fixed)
        {
            if (config.window_size <= 0)
                throw std::invalid_argument("MTP depth policy window_size must be > 0");
            if (config.min_samples <= 0)
                throw std::invalid_argument("MTP depth policy min_samples must be > 0");
            if (config.cooldown_steps < 0)
                throw std::invalid_argument("MTP depth policy cooldown_steps must be >= 0");
            if (config.promote_consecutive_windows <= 0)
            {
                throw std::invalid_argument(
                    "MTP depth policy promote_consecutive_windows must be > 0");
            }
            auto rate_valid = [](double value)
            {
                return value >= 0.0 && value <= 1.0;
            };
            if (!rate_valid(config.promote_full_accept_rate) ||
                !rate_valid(*config.demote_zero_accept_rate) ||
                !rate_valid(config.demote_acceptance_rate))
            {
                throw std::invalid_argument("MTP depth policy thresholds must be in [0, 1]");
            }
        }

        const auto cost_profile = resolveMTPDepthCostProfile(
            {.enabled = config.mode == MTPDepthPolicyMode::Dynamic && config.use_generated_policy,
             .backend = config.backend, .model_class = config.model_class,
             .verify_mode = verify_mode_, .scope = config.scope});
        if (cost_profile.active() && (config.min_depth <= 0 || config.max_depth > 15))
            throw std::invalid_argument("measured MTP utility profile requires positive depth bounds within1..15");
        config_ = config;
        cost_profile_ = cost_profile;
        current_depth_ = config_.initial_depth;
        steps_since_change_ = config_.cooldown_steps;
        if (cost_profile_.active())
            proposal_history_ = MTPProposalConfirmation{static_cast<uint32_t>(current_depth_),0};
        else
            proposal_history_ = LegacyProposalHistory{};
        window_ = {};
        utility_state_ = {};
        last_decision_ = {};
        last_decision_.old_depth = current_depth_;
        last_decision_.new_depth = current_depth_;
        last_decision_.recommended_depth = current_depth_;
        stats_ = {};
        rejected_depths_.assign(
            static_cast<size_t>(std::max(0, config_.max_depth) + 1),
            uint8_t{0});
    }

    bool MTPDepthController::depthZeroProbeReady() const
    {
        return config_.mode != MTPDepthPolicyMode::Fixed &&
               current_depth_ == 0 &&
               steps_since_change_ >= config_.cooldown_steps;
    }

    int MTPDepthController::requestedDepthForStep() const
    {
        if (current_depth_ > 0)
            return current_depth_;
        return depthZeroProbeReady() ? std::min(1, config_.max_depth) : 0;
    }

    void MTPDepthController::reset()
    {
        current_depth_ = config_.initial_depth;
        steps_since_change_ = config_.cooldown_steps;
        if (cost_profile_.active())
            proposal_history_ = MTPProposalConfirmation{static_cast<uint32_t>(current_depth_),0};
        else
            proposal_history_ = LegacyProposalHistory{};
        window_ = {};
        utility_state_ = {};
        last_decision_ = {};
        last_decision_.old_depth = current_depth_;
        last_decision_.new_depth = current_depth_;
        last_decision_.recommended_depth = current_depth_;
        stats_ = {};
        std::fill(rejected_depths_.begin(), rejected_depths_.end(), uint8_t{0});
    }

    bool MTPDepthController::depthRejected(int depth) const
    {
        return depth >= 0 &&
               static_cast<size_t>(depth) < rejected_depths_.size() &&
               rejected_depths_[static_cast<size_t>(depth)] != 0;
    }

    void MTPDepthController::setDepthRejected(int depth, bool rejected)
    {
        if (depth < 0 || static_cast<size_t>(depth) >= rejected_depths_.size())
            return;
        rejected_depths_[static_cast<size_t>(depth)] = rejected ? uint8_t{1} : uint8_t{0};
    }

    int MTPDepthController::nextUnrejectedDepthAbove(int depth) const
    {
        for (int candidate = depth + 1; candidate <= config_.max_depth; ++candidate)
        {
            if (!depthRejected(candidate))
                return candidate;
        }
        return depth;
    }

    bool MTPDepthController::windowReady() const
    {
        // A pending experiment completes at its admitted minimum, before any
        // ordinary parent matcher, full-window cadence or cooldown is consulted.
        if (cost_profile_.active() && utility_state_.phase == MTPReversibleProposalPhase::ProbePending)
            return window_.verifier_runs >= static_cast<uint64_t>(config_.min_samples);
        const uint64_t required = static_cast<uint64_t>(
            std::max(config_.window_size, config_.min_samples));
        if (window_.verifier_runs >= required)
            return true;

        if (cost_profile_.active())
        {
            // Pending was handled above. A supported/unresolved proposal may
            // use the existing minimum window, while holds retain the full
            // configured window. No generated hold can hide a priced action.
            return steps_since_change_ >= config_.cooldown_steps &&
                window_.verifier_runs >= static_cast<uint64_t>(config_.min_samples) &&
                survivalProposal(config_, current_depth_, window_, utility_state_, cost_profile_).kind !=
                    MTPSurvivalProposalKind::Hold;
        }

        if (config_.mode == MTPDepthPolicyMode::Fixed ||
            steps_since_change_ < config_.cooldown_steps ||
            window_.attempted_draft_tokens == 0)
        {
            return false;
        }

        const double acceptance_rate =
            static_cast<double>(window_.accepted_draft_tokens) /
            static_cast<double>(window_.attempted_draft_tokens);
        const double zero_accept_rate =
            static_cast<double>(window_.zero_accepts) /
            static_cast<double>(window_.verifier_runs);

        if (window_.verifier_runs < static_cast<uint64_t>(config_.min_samples))
            return false;

        if (auto generated = matchGeneratedDepthPolicy(
                config_,
                verify_mode_,
                current_depth_,
                window_))
        {
            /*
             * Generated promote/demote rows may evaluate after the smaller
             * min-sample window.  Generated hold rows are guardrails: they
             * suppress an eager handwritten demotion once a window is otherwise
             * ready, but they must not reset healthy evidence early.
             */
            return generated->target_depth != current_depth_;
        }

        const bool perfect_probe =
            current_depth_ < config_.max_depth &&
            !depthRejected(current_depth_ + 1) &&
            window_.full_accepts == window_.verifier_runs &&
            window_.zero_accepts == 0;
        if (perfect_probe)
            return true;

        if (current_depth_ <= config_.min_depth)
            return false;

        const bool demote_ready =
            zero_accept_rate >= *config_.demote_zero_accept_rate ||
            (current_depth_ > std::max(config_.min_depth, 1) &&
             acceptance_rate < config_.demote_acceptance_rate);
        if (!demote_ready)
            return false;

        if (isGeneratedBestDepth(config_, verify_mode_, current_depth_) &&
            window_.verifier_runs < static_cast<uint64_t>(config_.window_size))
        {
            /*
             * The generated startup lane is selected from complete training
             * requests. Do not let one small noisy window evict it;
             * wait for the ordinary full window unless the partial window is
             * completely unproductive.
             */
            return zero_accept_rate >= 1.0;
        }

        return true;
    }

    MTPDepthDecision MTPDepthController::windowDecision() const
    {
        MTPDepthDecision decision;
        decision.evaluated = true;
        decision.old_depth = current_depth_;
        decision.new_depth = current_depth_;
        decision.recommended_depth = current_depth_;
        decision.window = window_;

        if (window_.attempted_draft_tokens > 0)
        {
            decision.acceptance_rate =
                static_cast<double>(window_.accepted_draft_tokens) /
                static_cast<double>(window_.attempted_draft_tokens);
        }
        if (window_.verifier_runs > 0)
        {
            decision.zero_accept_rate =
                static_cast<double>(window_.zero_accepts) /
                static_cast<double>(window_.verifier_runs);
            decision.full_accept_rate =
                static_cast<double>(window_.full_accepts) /
                static_cast<double>(window_.verifier_runs);
        }

        return decision;
    }

    MTPDepthDecision MTPDepthController::evaluateWindow() const
    {
        auto decision = windowDecision();
        if (config_.mode == MTPDepthPolicyMode::Fixed)
        {
            decision.reason = MTPDepthDecisionReason::FixedMode;
            return decision;
        }
        if (steps_since_change_ < config_.cooldown_steps)
        {
            decision.reason = MTPDepthDecisionReason::CooldownActive;
            return decision;
        }

        if (cost_profile_.active())
        {
            const auto proposal = survivalProposal(config_, current_depth_, window_, utility_state_, cost_profile_);
            decision.recommended_depth = static_cast<int>(proposal.target_depth);
            decision.new_depth = decision.recommended_depth;
            decision.changed = decision.new_depth != current_depth_;
            decision.reason = proposal.kind == MTPSurvivalProposalKind::SupportedProbe
                ? MTPDepthDecisionReason::SurvivalSupportedProbe
                : proposal.kind == MTPSurvivalProposalKind::UnresolvedProbe
                    ? MTPDepthDecisionReason::SurvivalUnresolvedProbe
                    : MTPDepthDecisionReason::SurvivalHold;
            return decision;
        }

        int proposed_depth = current_depth_;
        /*
         * Low acceptance can shrink deeper drafts down to depth 1, but depth 0
         * is a qualitatively different bypass mode.  Enter it only on the
         * dedicated zero-acceptance signal so a noisy stochastic window does
         * not throw away the cheap depth-1 probe that keeps the controller
         * connected to MTP speedup opportunities.
         */
        const bool perfect_accept_window =
            window_.verifier_runs > 0 &&
            window_.full_accepts == window_.verifier_runs &&
            window_.zero_accepts == 0;

        if (auto generated = matchGeneratedDepthPolicy(
                config_,
                verify_mode_,
                current_depth_,
                window_))
        {
            decision.reason = generated->reason;
            decision.recommended_depth = generated->target_depth;
            if (config_.mode == MTPDepthPolicyMode::Observe)
            {
                decision.observe_recommendation =
                    generated->target_depth != current_depth_;
                decision.new_depth = current_depth_;
                return decision;
            }
            decision.new_depth = generated->target_depth;
            decision.changed = generated->target_depth != current_depth_;
            return decision;
        }

        const bool zero_accept_demote =
            current_depth_ > config_.min_depth &&
            decision.zero_accept_rate >= *config_.demote_zero_accept_rate;
        const bool low_accept_demote =
            current_depth_ > std::max(config_.min_depth, 1) &&
            decision.acceptance_rate < config_.demote_acceptance_rate;
        const bool highest_unrejected_depth =
            current_depth_ < config_.max_depth &&
            nextUnrejectedDepthAbove(current_depth_) == current_depth_;
        const double catastrophic_zero_accept_rate =
            *config_.demote_zero_accept_rate +
            (1.0 - *config_.demote_zero_accept_rate) * 0.5;
        const bool generated_best_depth_grace =
            isGeneratedBestDepth(config_, verify_mode_, current_depth_) &&
            (zero_accept_demote || low_accept_demote) &&
            decision.zero_accept_rate < catastrophic_zero_accept_rate &&
            std::get<LegacyProposalHistory>(proposal_history_).generated_best_bad_streak == 0;

        if (generated_best_depth_grace)
        {
            /*
             * The generated policy warm-starts at the measured request-wide
             * winner. A single noisy full window can be
             * much worse than the request average, especially at MoE depth 3.
             * Give the learned winner one grace window unless zero-accept
             * pressure is catastrophic; a second consecutive bad window still
             * demotes through the normal branch below.
             */
            decision.reason = MTPDepthDecisionReason::GeneratedBestDepthGraceWindow;
        }
        else if (zero_accept_demote ||
            (low_accept_demote && !highest_unrejected_depth))
        {
            const int upward_probe_depth = nextUnrejectedDepthAbove(current_depth_);
            /*
             * Probing past a weak intermediate depth is useful only when the
             * signal is ambiguous.  A window dominated by zero-accept steps is
             * already telling us the current draft depth is too expensive for
             * this request, so spending another window at an even deeper draft
             * repeats the same mistake.  The cutoff is derived from the
             * configured zero-accept demotion threshold: halfway from that
             * threshold to a completely zero-accept window is "catastrophic".
             */
            const bool ambiguous_demote_signal =
                decision.zero_accept_rate < catastrophic_zero_accept_rate;
            /*
             * A bad intermediate depth proves this candidate is poor, but it
             * does not always prove deeper candidates are poor.  Probe
             * unrejected higher depths once before settling downward. The
             * configured maximum is an ordinary candidate: a trained policy
             * may hold below it for measured economy, but generic policy must
             * not make that final row unreachable merely because it is last.
             */
            if (config_.mode == MTPDepthPolicyMode::Dynamic &&
                current_depth_ > std::max(config_.min_depth, 1) &&
                upward_probe_depth > current_depth_ &&
                ambiguous_demote_signal)
            {
                proposed_depth = upward_probe_depth;
                decision.reason = MTPDepthDecisionReason::ProbeHigherBeforeDemote;
            }
            else
            {
                proposed_depth = current_depth_ - 1;
                decision.reason = zero_accept_demote
                                      ? MTPDepthDecisionReason::DemoteZeroAcceptRate
                                      : MTPDepthDecisionReason::DemoteLowAcceptanceRate;
            }
        }
        else if (low_accept_demote && highest_unrejected_depth)
        {
            /*
             * Once a deeper depth has been rejected, the highest remaining
             * candidate is often still the best throughput lane even with
             * imperfect token acceptance.  Demoting on a merely low-acceptance
             * window makes the controller abandon the best fixed-depth lane
             * after it has already learned that going deeper is bad.  Keep the
             * stronger zero-accept demotion above for truly unproductive
             * windows; otherwise hold and gather another window at this depth.
             */
            decision.reason = MTPDepthDecisionReason::Hold;
        }
        else if (current_depth_ < config_.max_depth &&
                 decision.full_accept_rate >= config_.promote_full_accept_rate &&
                 window_.zero_accepts == 0)
        {
            const bool next_depth_was_rejected = depthRejected(current_depth_ + 1);
            /*
             * Once the deepest depth has failed during this request, the
             * handwritten fallback should not keep rediscovering the same
             * expensive loser.  Shallower retries are still useful: they let a
             * depth-zero bypass recover and let depth 1 retest depth 2 after
             * fresh hysteresis.
             */
            const bool blocked_rejected_deepest_retry =
                next_depth_was_rejected &&
                current_depth_ + 1 == config_.max_depth &&
                config_.max_depth >= 3;
            if (blocked_rejected_deepest_retry)
            {
                decision.reason = MTPDepthDecisionReason::Hold;
            }
            else if ((perfect_accept_window && !next_depth_was_rejected) ||
                     std::get<LegacyProposalHistory>(proposal_history_).promotion_streak + 1 >= config_.promote_consecutive_windows)
            {
                proposed_depth = current_depth_ + 1;
                decision.reason = MTPDepthDecisionReason::PromoteFullAcceptRate;
            }
            else
            {
                decision.reason = MTPDepthDecisionReason::PromotionHysteresisActive;
            }
        }
        else if (config_.min_depth >= 1 &&
                 current_depth_ == config_.min_depth &&
                 current_depth_ < config_.max_depth &&
                 decision.acceptance_rate >= config_.promote_full_accept_rate)
        {
            /*
             * Depth 1 is the cheapest useful speculative lane.  Climbing from
             * it is intentionally stricter than "not bad enough to demote":
             * a deeper probe pays extra sidecar and verifier work, so require
             * the same promotion threshold that governs ordinary depth growth.
             * Operators can still lower promote_full_accept_rate to explore
             * noisier stochastic/code prompts, while the default sticks near
             * fixed d1 unless depth 1 is essentially perfect.
             */
            if (std::get<LegacyProposalHistory>(proposal_history_).promotion_streak + 1 >= config_.promote_consecutive_windows)
            {
                proposed_depth = current_depth_ + 1;
                decision.reason = MTPDepthDecisionReason::PromoteFullAcceptRate;
            }
            else
            {
                decision.reason = MTPDepthDecisionReason::PromotionHysteresisActive;
            }
        }
        else
        {
            decision.reason = MTPDepthDecisionReason::Hold;
        }

        proposed_depth = std::clamp(proposed_depth, config_.min_depth, config_.max_depth);
        decision.recommended_depth = proposed_depth;
        if (config_.mode == MTPDepthPolicyMode::Observe)
        {
            decision.observe_recommendation = proposed_depth != current_depth_;
            decision.new_depth = current_depth_;
            return decision;
        }

        decision.new_depth = proposed_depth;
        decision.changed = proposed_depth != current_depth_;
        return decision;
    }

    MTPDepthDecision MTPDepthController::recordStep(const MTPDepthObservation &observation)
    {
        MTPDepthDecision decision;
        decision.old_depth = current_depth_;
        decision.new_depth = current_depth_;
        decision.recommended_depth = cost_profile_.active() &&
                std::holds_alternative<MTPProposalConfirmation>(proposal_history_)
            ? static_cast<int>(std::get<MTPProposalConfirmation>(proposal_history_).target_depth)
            : current_depth_;

        if (cost_profile_.active())
        {
            const auto *confirmation = std::get_if<MTPProposalConfirmation>(&proposal_history_);
            if (!confirmation || !validMTPProposalConfirmation(*confirmation,
                    static_cast<uint32_t>(current_depth_),static_cast<uint32_t>(config_.min_depth),
                    static_cast<uint32_t>(config_.max_depth),
                    static_cast<uint32_t>(config_.promote_consecutive_windows),
                    utility_state_.phase == MTPReversibleProposalPhase::ProbePending))
                throw std::invalid_argument("invalid measured MTP proposal confirmation before observation");
        }

        if (cost_profile_.active() && (!observation.newly_emitted_tokens ||
            *observation.newly_emitted_tokens == 0 ||
            *observation.newly_emitted_tokens > static_cast<uint32_t>(current_depth_ + 1) ||
            observation.accepted_speculative_prefix < 0 || observation.accepted_speculative_prefix > current_depth_))
            throw std::invalid_argument("measured MTP utility requires actual positive emitted progress and valid prefix");

        if (observation.publication && (!observation.publication->valid() ||
            !observation.newly_emitted_tokens || observation.publication->newlyEmittedTokens() !=
                *observation.newly_emitted_tokens))
            throw std::invalid_argument("MTP observation disagrees with its actual committed publication");
        if (cost_profile_.active() && !observation.publication)
            throw std::invalid_argument("measured MTP requires an actual committed publication witness");

        if (config_.mode == MTPDepthPolicyMode::Fixed)
        {
            decision.reason = MTPDepthDecisionReason::FixedMode;
            last_decision_ = decision;
            return decision;
        }
        if (observation.publication && observation.publication->modelStopped())
        {
            decision.reason = MTPDepthDecisionReason::ModelStopped;
            last_decision_ = decision;
            return decision;
        }

        if (observation.budget_limited || observation.effective_depth <= 0)
        {
            decision.reason = MTPDepthDecisionReason::BudgetLimited;
            last_decision_ = decision;
            return decision;
        }

        const int effective_depth = std::clamp(
            observation.effective_depth,
            config_.min_depth,
            config_.max_depth);
        const int accepted_prefix = std::clamp(
            observation.accepted_speculative_prefix,
            0,
            effective_depth);

        // Only actual non-stopped, unclipped publication supports the prefix
        // identity. Keep physical emitted Y separate from verified state V;
        // carried outputs may change Y by one without changing accepted work.
        if (observation.publication && observation.publication->publishedStateTokens() !=
                static_cast<uint32_t>(accepted_prefix + 1))
            throw std::invalid_argument("MTP verified progress violates its eligible prefix contract");

        const bool utility_pending = cost_profile_.active() &&
            utility_state_.phase == MTPReversibleProposalPhase::ProbePending;
        if (cost_profile_.active())
        {
            if (!observation.newly_emitted_tokens ||
                observation.effective_depth != current_depth_ || observation.requested_depth != current_depth_ ||
                window_.verifier_runs >= UINT32_MAX ||
                steps_since_change_ == std::numeric_limits<int>::max())
                throw std::invalid_argument("measured MTP utility requires actual bounded emitted progress/current-depth observation");
            auto next = utility_state_;
            const auto appended = appendMTPReversibleProposalEmission(
                {.mode = MTPReversibleProposalMode::Dynamic,
                 .minimum_depth = static_cast<uint32_t>(config_.min_depth),
                 .maximum_depth = static_cast<uint32_t>(config_.max_depth),
                 .minimum_samples = static_cast<uint32_t>(config_.min_samples)}, next,
                {.executed_depth = static_cast<uint32_t>(current_depth_),
                 .window_runs = static_cast<uint32_t>(window_.verifier_runs + 1),
                 .newly_emitted_tokens = *observation.newly_emitted_tokens, .eligible = true});
            if (!appended.valid())
                throw std::invalid_argument("invalid/overflowing measured MTP utility observation");
            utility_state_ = next;
        }

        ++window_.verifier_runs;
        window_.attempted_draft_tokens += static_cast<uint64_t>(effective_depth);
        window_.accepted_draft_tokens += static_cast<uint64_t>(accepted_prefix);
        window_.rejected_draft_tokens += static_cast<uint64_t>(effective_depth - accepted_prefix);
        window_.accepted_prefix_sum += static_cast<uint64_t>(accepted_prefix);
        if (accepted_prefix == 0)
            ++window_.zero_accepts;
        if (accepted_prefix == effective_depth)
            ++window_.full_accepts;
        if (observation.rollback)
            ++window_.rollbacks;
        ++steps_since_change_;

        if (!windowReady())
        {
            decision.reason = MTPDepthDecisionReason::WindowNotReady;
            decision.window = window_;
            last_decision_ = decision;
            return decision;
        }

        decision = utility_pending ? windowDecision() : evaluateWindow();
        ++stats_.windows;
        // Parent heuristics keep their own history only when actually evaluated.
        // A pending comparison must not manufacture a second parent observation.
        if (!utility_pending && !cost_profile_.active())
        {
            if (decision.observe_recommendation)
            {
                ++stats_.observe_recommendations;
            }
            if (decision.reason == MTPDepthDecisionReason::PromotionHysteresisActive)
            {
                ++std::get<LegacyProposalHistory>(proposal_history_).promotion_streak;
            }
            else if (decision.reason == MTPDepthDecisionReason::PromoteFullAcceptRate ||
                     decision.reason == MTPDepthDecisionReason::DemoteZeroAcceptRate ||
                     decision.reason == MTPDepthDecisionReason::DemoteLowAcceptanceRate ||
                     decision.reason == MTPDepthDecisionReason::GeneratedPolicyPromote ||
                     decision.reason == MTPDepthDecisionReason::GeneratedPolicyDemote ||
                     decision.reason == MTPDepthDecisionReason::GeneratedPolicyHold ||
                     decision.reason == MTPDepthDecisionReason::GeneratedBestDepthGraceWindow ||
                     decision.reason == MTPDepthDecisionReason::ProbeHigherBeforeDemote ||
                     decision.reason == MTPDepthDecisionReason::Hold)
            {
                std::get<LegacyProposalHistory>(proposal_history_).promotion_streak = 0;
            }
            if (decision.evaluated)
            {
                if (decision.reason == MTPDepthDecisionReason::GeneratedBestDepthGraceWindow)
                    ++std::get<LegacyProposalHistory>(proposal_history_).generated_best_bad_streak;
                else
                    std::get<LegacyProposalHistory>(proposal_history_).generated_best_bad_streak = 0;
            }
            if (config_.mode == MTPDepthPolicyMode::Dynamic)
            {
                if (decision.reason == MTPDepthDecisionReason::DemoteZeroAcceptRate ||
                    decision.reason == MTPDepthDecisionReason::DemoteLowAcceptanceRate ||
                    decision.reason == MTPDepthDecisionReason::GeneratedPolicyDemote ||
                    decision.reason == MTPDepthDecisionReason::ProbeHigherBeforeDemote)
                {
                    setDepthRejected(decision.old_depth, true);
                }
                /*
                 * Generated promotions are explicit trained retests.  Ordinary
                 * fallback promotion no longer targets rejected depths, so this is
                 * normally a no-op for it.  ProbeHigherBeforeDemote is different:
                 * it is a diagnostic jump taken from a bad intermediate window.
                 * Do not clear a previously rejected destination depth merely
                 * because we are probing upward; otherwise the controller can
                 * churn back into an expensive bad depth.
                 */
                if (decision.changed &&
                    decision.new_depth > decision.old_depth &&
                    (decision.reason == MTPDepthDecisionReason::PromoteFullAcceptRate ||
                     decision.reason == MTPDepthDecisionReason::GeneratedPolicyPromote))
                {
                    setDepthRejected(decision.new_depth, false);
                }
            }
        }
        if (cost_profile_.active())
        {
            if (window_.accepted_draft_tokens > UINT32_MAX ||
                window_.verifier_runs > UINT32_MAX - window_.accepted_draft_tokens)
                throw std::invalid_argument("MTP verified progress exceeds comparison capacity");
            auto &confirmation = std::get<MTPProposalConfirmation>(proposal_history_);
            auto next_confirmation = MTPProposalConfirmation{static_cast<uint32_t>(current_depth_),0};
            bool awaiting_confirmation = false;
            if (!utility_pending)
            {
                const auto confirmed = confirmMTPSurvivalProposal(confirmation,
                    static_cast<uint32_t>(current_depth_),static_cast<uint32_t>(config_.min_depth),
                    static_cast<uint32_t>(config_.max_depth),
                    static_cast<uint32_t>(config_.promote_consecutive_windows),
                    static_cast<uint32_t>(decision.recommended_depth));
                if (!confirmed.valid())
                    throw std::invalid_argument("invalid completed MTP proposal confirmation");
                next_confirmation = confirmed.next;
                awaiting_confirmation = confirmed.action == MTPProposalConfirmationAction::AwaitingConfirmation;
            }
            const auto gate = decideMTPReversibleProposalGate(
                {.mode = MTPReversibleProposalMode::Dynamic,
                 .minimum_depth = static_cast<uint32_t>(config_.min_depth),
                 .maximum_depth = static_cast<uint32_t>(config_.max_depth),
                 .minimum_samples = static_cast<uint32_t>(config_.min_samples)}, utility_state_,
                {.executed_depth = static_cast<uint32_t>(current_depth_),
                 .window_runs = static_cast<uint32_t>(window_.verifier_runs), .eligible = true,
                 .proposal_origin = utility_pending ? MTPDepthProposalOrigin::Unevaluated
                                                    : awaiting_confirmation ? MTPDepthProposalOrigin::Hold : proposalOrigin(decision),
                 .parent_target = static_cast<uint32_t>(awaiting_confirmation ? current_depth_ : decision.recommended_depth),
                 .verified_progress = {.states = static_cast<uint32_t>(
                     window_.verifier_runs + window_.accepted_draft_tokens)}}, cost_profile_);
            if (!gate.valid() || !gate.consume_window)
                throw std::invalid_argument("invalid measured MTP utility decision/window ownership");
            decision.new_depth = static_cast<int>(gate.next_depth);
            if (awaiting_confirmation)
            {
                confirmation = next_confirmation;
                decision.reason = MTPDepthDecisionReason::SurvivalAwaitingConfirmation;
            }
            else
            {
                confirmation = MTPProposalConfirmation{gate.next_depth,0};
                decision.recommended_depth = decision.new_depth;
            }
            decision.changed = decision.new_depth != decision.old_depth;
            switch (gate.action)
            {
            case MTPReversibleProposalAction::ProbeStarted:
                ++stats_.utility_probes; decision.reason = MTPDepthDecisionReason::UtilityProbeStarted; break;
            case MTPReversibleProposalAction::ProbeKept:
                ++stats_.utility_keeps; decision.reason = MTPDepthDecisionReason::UtilityProbeKept; break;
            case MTPReversibleProposalAction::ProbeReturned:
                ++stats_.utility_returns; decision.reason = MTPDepthDecisionReason::UtilityProbeReturned; break;
            case MTPReversibleProposalAction::RejectedTargetBlocked:
                decision.reason = MTPDepthDecisionReason::UtilityRejectedComparisonHold; break;
            default: break;
            }
        }
        if (decision.changed)
        {
            ++stats_.updates;
            if (decision.new_depth > decision.old_depth)
                ++stats_.promotions;
            else
                ++stats_.demotions;
            current_depth_ = decision.new_depth;
            steps_since_change_ = 0;
        }
        else if (current_depth_ == 0 && decision.evaluated)
        {
            steps_since_change_ = 0;
        }
        window_ = {};
        last_decision_ = decision;
        return decision;
    }

    MTPDepthDecision MTPDepthController::recordBypassStep()
    {
        MTPDepthDecision decision;
        decision.old_depth = current_depth_;
        decision.new_depth = current_depth_;
        decision.recommended_depth = current_depth_;

        if (config_.mode == MTPDepthPolicyMode::Fixed)
        {
            decision.reason = MTPDepthDecisionReason::FixedMode;
            last_decision_ = decision;
            return decision;
        }

        if (current_depth_ != 0)
        {
            decision.reason = MTPDepthDecisionReason::Hold;
            last_decision_ = decision;
            return decision;
        }

        ++steps_since_change_;
        decision.reason = MTPDepthDecisionReason::DepthZeroBypass;
        last_decision_ = decision;
        return decision;
    }

} // namespace llaminar2
