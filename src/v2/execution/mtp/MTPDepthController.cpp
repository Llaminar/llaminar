#include "MTPDepthController.h"

#include <algorithm>
#include <stdexcept>

namespace llaminar2
{

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
        int configured_draft_tokens)
    {
        configure(config, configured_draft_tokens);
    }

    void MTPDepthController::configure(
        MTPDepthPolicyConfig config,
        int configured_draft_tokens)
    {
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
            config.max_depth = configured_draft_tokens;
        }
        if (config.initial_depth <= 0)
        {
            config.initial_depth =
                config.mode == MTPDepthPolicyMode::Dynamic
                    ? config.min_depth
                    : config.max_depth;
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
            !rate_valid(config.demote_zero_accept_rate) ||
            !rate_valid(config.demote_acceptance_rate))
        {
            throw std::invalid_argument("MTP depth policy thresholds must be in [0, 1]");
        }

        config_ = config;
        current_depth_ = config_.initial_depth;
        steps_since_change_ = config_.cooldown_steps;
        promotion_streak_ = 0;
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

        const bool perfect_probe =
            current_depth_ < config_.max_depth &&
            !depthRejected(current_depth_ + 1) &&
            window_.full_accepts == window_.verifier_runs &&
            window_.zero_accepts == 0;
        if (perfect_probe)
            return true;

        if (current_depth_ <= config_.min_depth)
            return false;

        return zero_accept_rate >= config_.demote_zero_accept_rate ||
               (current_depth_ > std::max(config_.min_depth, 1) &&
                acceptance_rate < config_.demote_acceptance_rate);
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

        const bool zero_accept_demote =
            current_depth_ > config_.min_depth &&
            decision.zero_accept_rate >= config_.demote_zero_accept_rate;
        const bool low_accept_demote =
            current_depth_ > std::max(config_.min_depth, 1) &&
            decision.acceptance_rate < config_.demote_acceptance_rate;
        const bool highest_unrejected_depth =
            current_depth_ < config_.max_depth &&
            nextUnrejectedDepthAbove(current_depth_) == current_depth_;

        if (zero_accept_demote ||
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
            const double catastrophic_zero_accept_rate =
                config_.demote_zero_accept_rate +
                (1.0 - config_.demote_zero_accept_rate) * 0.5;
            const bool ambiguous_demote_signal =
                decision.zero_accept_rate < catastrophic_zero_accept_rate;
            /*
             * A bad intermediate depth proves this candidate is poor, but it
             * does not prove deeper candidates are poor.  Probe each untested
             * deeper depth once before settling downward; rejected depths can
             * be retried later through the normal promotion hysteresis.
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
            if ((perfect_accept_window && !next_depth_was_rejected) ||
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
                 decision.reason == MTPDepthDecisionReason::ProbeHigherBeforeDemote ||
                 decision.reason == MTPDepthDecisionReason::Hold)
        {
            promotion_streak_ = 0;
        }
        if (config_.mode == MTPDepthPolicyMode::Dynamic)
        {
            if (decision.reason == MTPDepthDecisionReason::DemoteZeroAcceptRate ||
                decision.reason == MTPDepthDecisionReason::DemoteLowAcceptanceRate ||
                decision.reason == MTPDepthDecisionReason::ProbeHigherBeforeDemote)
            {
                setDepthRejected(decision.old_depth, true);
            }
            /*
             * A normal promotion is earned by healthy lower-depth windows, so
             * it is allowed to forgive the destination depth and retest it.
             * ProbeHigherBeforeDemote is different: it is a diagnostic jump
             * taken from a bad intermediate window.  Do not clear a previously
             * rejected destination depth merely because we are probing upward;
             * otherwise the controller can churn back into an expensive bad
             * depth on every ambiguous demotion window.
             */
            if (decision.changed &&
                decision.new_depth > decision.old_depth &&
                decision.reason == MTPDepthDecisionReason::PromoteFullAcceptRate)
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
