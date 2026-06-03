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
            config.initial_depth = config.max_depth;
        if (config.min_depth < 1)
            throw std::invalid_argument("MTP depth policy min_depth must be > 0");
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
    }

    bool MTPDepthController::windowReady() const
    {
        const uint64_t required = static_cast<uint64_t>(
            std::max(config_.window_size, config_.min_samples));
        if (window_.verifier_runs >= required)
            return true;

        if (config_.mode == MTPDepthPolicyMode::Fixed ||
            current_depth_ <= config_.min_depth ||
            steps_since_change_ < config_.cooldown_steps ||
            window_.verifier_runs < static_cast<uint64_t>(config_.min_samples) ||
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

        return zero_accept_rate >= config_.demote_zero_accept_rate ||
               acceptance_rate < config_.demote_acceptance_rate;
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
        if (current_depth_ > config_.min_depth &&
            decision.zero_accept_rate >= config_.demote_zero_accept_rate)
        {
            proposed_depth = current_depth_ - 1;
            decision.reason = MTPDepthDecisionReason::DemoteZeroAcceptRate;
        }
        else if (current_depth_ > config_.min_depth &&
                 decision.acceptance_rate < config_.demote_acceptance_rate)
        {
            proposed_depth = config_.min_depth;
            decision.reason = MTPDepthDecisionReason::DemoteLowAcceptanceRate;
        }
        else if (current_depth_ < config_.max_depth &&
                 decision.full_accept_rate >= config_.promote_full_accept_rate &&
                 decision.zero_accept_rate < config_.demote_zero_accept_rate)
        {
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
                 decision.reason == MTPDepthDecisionReason::Hold)
        {
            promotion_streak_ = 0;
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
        window_ = {};
        last_decision_ = decision;
        return decision;
    }

} // namespace llaminar2
