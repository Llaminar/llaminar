#pragma once

#include "../config/RuntimeConfig.h"

#include <cstdint>
#include <string>

namespace llaminar2
{

    enum class MTPDepthDecisionReason
    {
        FixedMode,
        WindowNotReady,
        BudgetLimited,
        CooldownActive,
        PromotionHysteresisActive,
        PromoteFullAcceptRate,
        DemoteZeroAcceptRate,
        DemoteLowAcceptanceRate,
        Hold,
    };

    const char *toString(MTPDepthDecisionReason reason);

    struct MTPDepthWindow
    {
        uint64_t verifier_runs = 0;
        uint64_t attempted_draft_tokens = 0;
        uint64_t accepted_draft_tokens = 0;
        uint64_t rejected_draft_tokens = 0;
        uint64_t rollbacks = 0;
        uint64_t full_accepts = 0;
        uint64_t zero_accepts = 0;
        uint64_t accepted_prefix_sum = 0;
    };

    struct MTPDepthObservation
    {
        int requested_depth = 1;
        int effective_depth = 1;
        int accepted_speculative_prefix = 0;
        bool budget_limited = false;
        bool rollback = false;
    };

    struct MTPDepthDecision
    {
        bool evaluated = false;
        bool changed = false;
        bool observe_recommendation = false;
        MTPDepthDecisionReason reason = MTPDepthDecisionReason::WindowNotReady;
        int old_depth = 1;
        int new_depth = 1;
        int recommended_depth = 1;
        MTPDepthWindow window;
        double acceptance_rate = 0.0;
        double zero_accept_rate = 0.0;
        double full_accept_rate = 0.0;
    };

    struct MTPDepthControllerStats
    {
        uint64_t windows = 0;
        uint64_t updates = 0;
        uint64_t promotions = 0;
        uint64_t demotions = 0;
        uint64_t observe_recommendations = 0;
    };

    class MTPDepthController
    {
    public:
        MTPDepthController() = default;
        MTPDepthController(MTPDepthPolicyConfig config, int configured_draft_tokens);

        void configure(MTPDepthPolicyConfig config, int configured_draft_tokens);
        void reset();

        int currentDepth() const { return current_depth_; }
        int minDepth() const { return config_.min_depth; }
        int maxDepth() const { return config_.max_depth; }
        MTPDepthPolicyMode mode() const { return config_.mode; }
        const MTPDepthControllerStats &stats() const { return stats_; }
        const MTPDepthDecision &lastDecision() const { return last_decision_; }

        MTPDepthDecision recordStep(const MTPDepthObservation &observation);

    private:
        MTPDepthDecision evaluateWindow() const;
        bool windowReady() const;

        MTPDepthPolicyConfig config_;
        int current_depth_ = 1;
        int steps_since_change_ = 0;
        int promotion_streak_ = 0;
        MTPDepthWindow window_;
        MTPDepthDecision last_decision_;
        MTPDepthControllerStats stats_;
    };

} // namespace llaminar2
