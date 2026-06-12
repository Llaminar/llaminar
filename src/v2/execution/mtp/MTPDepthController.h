#pragma once

#include "../config/RuntimeConfig.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Human-readable reason for the last adaptive depth decision.
     *
     * These values are exported through request summaries and benchmark JSON.
     * Keep them stable enough that tuning scripts can group by the string form
     * returned by @ref toString.
     */
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
        GeneratedPolicyPromote,
        GeneratedPolicyDemote,
        GeneratedPolicyHold,
        GeneratedBestDepthGraceWindow,
        ProbeHigherBeforeDemote,
        DepthZeroBypass,
        Hold,
    };

    const char *toString(MTPDepthDecisionReason reason);

    /**
     * @brief Rolling observation window used by dynamic MTP depth selection.
     *
     * The controller intentionally records only cheap counters from completed
     * verifier steps.  It does not read clocks or backend-specific timing; the
     * offline trainer learns timing-sensitive policies from benchmark TSVs and
     * emits simple acceptance-window predicates.
     */
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

    /**
     * @brief Per-step signal recorded after a speculative verifier step.
     *
     * @c requested_depth is what the controller asked for, @c effective_depth is
     * the number of draft rows actually attempted after budget clamping, and
     * @c accepted_speculative_prefix is the contiguous accepted prefix.
     */
    struct MTPDepthObservation
    {
        int requested_depth = 1;
        int effective_depth = 1;
        int accepted_speculative_prefix = 0;
        bool budget_limited = false;
        bool rollback = false;
    };

    /**
     * @brief Result of evaluating one rolling window.
     *
     * Most decode steps return @c evaluated=false because the window is still
     * accumulating evidence.  When evaluated, @c changed tells callers whether
     * the live request depth moved and @c reason explains why.
     */
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

    /**
     * @brief Aggregate adaptive-depth counters emitted into request stats.
     */
    struct MTPDepthControllerStats
    {
        uint64_t windows = 0;
        uint64_t updates = 0;
        uint64_t promotions = 0;
        uint64_t demotions = 0;
        uint64_t observe_recommendations = 0;
    };

    /**
     * @brief Deterministic online controller for fixed, observe, and dynamic
     * MTP draft depth.
     *
     * Dynamic mode combines hand-written safety hysteresis with an optional
     * generated policy table.  The generated policy can only promote/demote
     * inside the configured min/max bounds and is disabled automatically for
     * fixed mode, so fixed-depth benchmark lanes remain hard pinned.
     */
    class MTPDepthController
    {
    public:
        MTPDepthController() = default;
        MTPDepthController(
            MTPDepthPolicyConfig config,
            int configured_draft_tokens,
            MTPVerifyMode verify_mode = MTPVerifyMode::Greedy);

        void configure(
            MTPDepthPolicyConfig config,
            int configured_draft_tokens,
            MTPVerifyMode verify_mode = MTPVerifyMode::Greedy);
        void reset();

        int currentDepth() const { return current_depth_; }
        int requestedDepthForStep() const;
        int minDepth() const { return config_.min_depth; }
        int maxDepth() const { return config_.max_depth; }
        MTPDepthPolicyMode mode() const { return config_.mode; }
        const MTPDepthControllerStats &stats() const { return stats_; }
        const MTPDepthDecision &lastDecision() const { return last_decision_; }

        MTPDepthDecision recordStep(const MTPDepthObservation &observation);
        MTPDepthDecision recordBypassStep();

    private:
        /**
         * @brief Return true when a previously tried depth produced a demotion
         * signal and should not be retried through the fast perfect-probe path.
         */
        bool depthRejected(int depth) const;

        /**
         * @brief Mark a draft depth as rejected or clear that rejection.
         *
         * Dynamic mode uses this small memory to distinguish "untested" from
         * "known bad for the current request".  A rejected depth can still be
         * retried later, but only through the normal promotion hysteresis.
         */
        void setDepthRejected(int depth, bool rejected);

        /**
         * @brief Find the nearest un-rejected deeper draft depth.
         *
         * This lets a bad intermediate probe try the next candidate once before
         * settling downward.  It is deliberately depth-only bookkeeping: timing
         * remains in the benchmark harness, not in the online controller.
         */
        int nextUnrejectedDepthAbove(int depth) const;

        MTPDepthDecision evaluateWindow() const;
        bool windowReady() const;
        bool depthZeroProbeReady() const;

        MTPDepthPolicyConfig config_;
        /** @brief Active verifier mode used to select generated policy rows. */
        MTPVerifyMode verify_mode_ = MTPVerifyMode::Greedy;
        int current_depth_ = 1;
        int steps_since_change_ = 0;
        int promotion_streak_ = 0;
        /**
         * @brief Consecutive non-catastrophic bad windows at the learned best
         * fixed-depth lane.
         *
         * Benchmark-trained hold rows can identify a depth that wins over a
         * whole request even though individual 16-step windows are noisy.  One
         * bad window gets grace; a second consecutive bad window is treated as
         * request-local evidence that this prompt differs from the trained
         * fixed-depth lane.
         */
        int generated_best_bad_streak_ = 0;
        MTPDepthWindow window_;
        MTPDepthDecision last_decision_;
        MTPDepthControllerStats stats_;
        std::vector<uint8_t> rejected_depths_;
    };

} // namespace llaminar2
