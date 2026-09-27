/**
 * @file MTPDepthController.cpp
 * @brief Deterministic rolling-window MTP depth decisions for host execution.
 *
 * Configuration resolves automatic thresholds before any window is evaluated.
 * GPU callers seal the same topology-bound values and trained initial depth
 * into the device ABI instead of consulting this host controller for live
 * device-owned decisions. Startup policy is resolved here once for both owners.
 */
#include "MTPDepthController.h"
#include "MTPDepthLearnedPolicy.h"

#include <algorithm>
#include <optional>
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
                 .verify_mode = verify_mode},
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

            for (const auto &choice : kMTPGeneratedDepthPolicyStartups)
            {
                if (choice.verify_mode != verify_mode)
                    continue;
                if (choice.backend != MTPDepthPolicyBackend::Any &&
                    choice.backend != config.backend)
                    continue;
                if (choice.model_class != MTPDepthPolicyModelClass::Any &&
                    choice.model_class != config.model_class)
                    continue;
                if (choice.initial_depth < config.min_depth ||
                    choice.initial_depth > config.max_depth)
                    continue;
                return choice.initial_depth;
            }
            return -1;
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

        config_ = config;
        current_depth_ = config_.initial_depth;
        steps_since_change_ = config_.cooldown_steps;
        promotion_streak_ = 0;
        generated_best_bad_streak_ = 0;
        window_ = {};
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
        promotion_streak_ = 0;
        generated_best_bad_streak_ = 0;
        window_ = {};
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
        const uint64_t required = static_cast<uint64_t>(
            std::max(config_.window_size, config_.min_samples));
        if (window_.verifier_runs >= required)
            return true;

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
            if (generated->target_depth != current_depth_)
                return true;
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

    MTPDepthDecision MTPDepthController::evaluateWindow() const
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
            generated_best_bad_streak_ == 0;

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
                     promotion_streak_ + 1 >= config_.promote_consecutive_windows)
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
            if (promotion_streak_ + 1 >= config_.promote_consecutive_windows)
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
        decision.recommended_depth = current_depth_;

        if (config_.mode == MTPDepthPolicyMode::Fixed)
        {
            decision.reason = MTPDepthDecisionReason::FixedMode;
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

        decision = evaluateWindow();
        ++stats_.windows;
        if (decision.observe_recommendation)
        {
            ++stats_.observe_recommendations;
        }
        if (decision.reason == MTPDepthDecisionReason::PromotionHysteresisActive)
        {
            ++promotion_streak_;
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
            promotion_streak_ = 0;
        }
        if (decision.evaluated)
        {
            if (decision.reason == MTPDepthDecisionReason::GeneratedBestDepthGraceWindow)
                ++generated_best_bad_streak_;
            else
                generated_best_bad_streak_ = 0;
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
