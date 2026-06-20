#include "MTPDepthController.h"

#include <algorithm>
#include <optional>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        /**
         * @brief One row in the generated dynamic-depth policy table.
         *
         * The table is produced offline from benchmark summaries.  Runtime only
         * evaluates simple numeric predicates so the controller stays
         * deterministic and cheap.
         */
        struct MTPGeneratedDepthPolicyRule
        {
            MTPVerifyMode verify_mode = MTPVerifyMode::Greedy;
            MTPDepthPolicyBackend backend = MTPDepthPolicyBackend::Any;
            MTPDepthPolicyModelClass model_class = MTPDepthPolicyModelClass::Any;
            int current_depth = 1;
            double min_acceptance_rate = 0.0;
            double max_acceptance_rate = 1.0;
            double max_zero_accept_rate = 1.0;
            double min_full_accept_rate = 0.0;
            int depth_delta = 0;
            const char *label = "";
        };

#include "MTPDepthPolicyGenerated.inc"

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
            double acceptance_rate,
            double zero_accept_rate,
            double full_accept_rate)
        {
            if (!config.use_generated_policy ||
                config.mode != MTPDepthPolicyMode::Dynamic)
            {
                return std::nullopt;
            }

            for (const auto &rule : kMTPGeneratedDepthPolicyRules)
            {
                if (rule.verify_mode != verify_mode)
                    continue;
                if (rule.backend != MTPDepthPolicyBackend::Any &&
                    rule.backend != config.backend)
                    continue;
                if (rule.model_class != MTPDepthPolicyModelClass::Any &&
                    rule.model_class != config.model_class)
                    continue;
                if (rule.current_depth != current_depth)
                    continue;
                if (acceptance_rate < rule.min_acceptance_rate ||
                    acceptance_rate > rule.max_acceptance_rate ||
                    zero_accept_rate > rule.max_zero_accept_rate ||
                    full_accept_rate < rule.min_full_accept_rate)
                {
                    continue;
                }

                const int target = std::clamp(
                    current_depth + rule.depth_delta,
                    config.min_depth,
                    config.max_depth);
                return MTPGeneratedDepthPolicyMatch{
                    .target_depth = target,
                    .reason = rule.depth_delta > 0
                                  ? MTPDepthDecisionReason::GeneratedPolicyPromote
                                  : (rule.depth_delta < 0
                                         ? MTPDepthDecisionReason::GeneratedPolicyDemote
                                         : MTPDepthDecisionReason::GeneratedPolicyHold),
                };
            }

            return std::nullopt;
        }

        /**
         * @brief Return the learned best fixed-depth lane for this request class.
         *
         * The generated trainer emits a delta-zero hold row for the fixed-depth
         * lane that won a benchmark group.  Dynamic mode can use that row as a
         * warm start so short generations do not spend their first windows
         * rediscovering the same profitable depth through noisy local
         * acceptance.  If no generated hold row matches, callers keep the
         * handwritten default.
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

            int best_depth = -1;
            for (const auto &rule : kMTPGeneratedDepthPolicyRules)
            {
                if (rule.depth_delta != 0 ||
                    rule.verify_mode != verify_mode)
                {
                    continue;
                }
                if (rule.backend != MTPDepthPolicyBackend::Any &&
                    rule.backend != config.backend)
                    continue;
                if (rule.model_class != MTPDepthPolicyModelClass::Any &&
                    rule.model_class != config.model_class)
                    continue;
                if (rule.current_depth < config.min_depth ||
                    rule.current_depth > config.max_depth)
                    continue;
                best_depth = std::max(best_depth, rule.current_depth);
            }
            return best_depth;
        }

        bool isGeneratedBestDepth(
            const MTPDepthPolicyConfig &config,
            MTPVerifyMode verify_mode,
            int current_depth)
        {
            return resolveGeneratedInitialDepth(config, verify_mode) == current_depth;
        }
    } // namespace

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
            config.initial_depth = resolveGeneratedInitialDepth(config, verify_mode_);
            if (config.initial_depth <= 0)
            {
                config.initial_depth =
                    resolveMTPDepthPolicyInitialDepth(
                        config,
                        configured_draft_tokens,
                        verify_mode_);
            }
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
                !rate_valid(config.demote_zero_accept_rate) ||
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
                acceptance_rate,
                zero_accept_rate,
                static_cast<double>(window_.full_accepts) /
                    static_cast<double>(window_.verifier_runs)))
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
            zero_accept_rate >= config_.demote_zero_accept_rate ||
            (current_depth_ > std::max(config_.min_depth, 1) &&
             acceptance_rate < config_.demote_acceptance_rate);
        if (!demote_ready)
            return false;

        if (isGeneratedBestDepth(config_, verify_mode_, current_depth_) &&
            window_.verifier_runs < static_cast<uint64_t>(config_.window_size))
        {
            /*
             * A generated best-depth lane is selected from full fixed-depth
             * benchmark evidence.  Do not let one small noisy window evict it;
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
                decision.acceptance_rate,
                decision.zero_accept_rate,
                decision.full_accept_rate))
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
            decision.zero_accept_rate >= config_.demote_zero_accept_rate;
        const bool low_accept_demote =
            current_depth_ > std::max(config_.min_depth, 1) &&
            decision.acceptance_rate < config_.demote_acceptance_rate;
        const bool highest_unrejected_depth =
            current_depth_ < config_.max_depth &&
            nextUnrejectedDepthAbove(current_depth_) == current_depth_;
        const double catastrophic_zero_accept_rate =
            config_.demote_zero_accept_rate +
            (1.0 - config_.demote_zero_accept_rate) * 0.5;
        const bool generated_best_depth_grace =
            isGeneratedBestDepth(config_, verify_mode_, current_depth_) &&
            (zero_accept_demote || low_accept_demote) &&
            decision.zero_accept_rate < catastrophic_zero_accept_rate &&
            generated_best_bad_streak_ == 0;

        if (generated_best_depth_grace)
        {
            /*
             * The generated policy warm-starts at the fixed-depth lane that
             * won whole-request benchmarks.  A single noisy full window can be
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
            const bool upward_probe_enters_deepest =
                upward_probe_depth == config_.max_depth &&
                config_.max_depth >= 3;
            /*
             * A bad intermediate depth proves this candidate is poor, but it
             * does not always prove deeper candidates are poor.  Probe
             * shallower intermediate depths once before settling downward.
             * The deepest lane is expensive enough that entering it is a
             * generated-policy decision, not a handwritten fallback guess.
             */
            if (config_.mode == MTPDepthPolicyMode::Dynamic &&
                current_depth_ > std::max(config_.min_depth, 1) &&
                upward_probe_depth > current_depth_ &&
                ambiguous_demote_signal &&
                !upward_probe_enters_deepest)
            {
                proposed_depth = upward_probe_depth;
                decision.reason = MTPDepthDecisionReason::ProbeHigherBeforeDemote;
            }
            else if (config_.mode == MTPDepthPolicyMode::Dynamic &&
                     current_depth_ > std::max(config_.min_depth, 1) &&
                     upward_probe_depth > current_depth_ &&
                     ambiguous_demote_signal &&
                     upward_probe_enters_deepest)
            {
                proposed_depth = current_depth_;
                decision.reason = MTPDepthDecisionReason::Hold;
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
            const bool fallback_enters_deepest =
                current_depth_ + 1 == config_.max_depth &&
                config_.max_depth >= 3;
            if (blocked_rejected_deepest_retry)
            {
                decision.reason = MTPDepthDecisionReason::Hold;
            }
            else if (fallback_enters_deepest)
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
