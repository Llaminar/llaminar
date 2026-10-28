/**
 * @file MTPDepthController.h
 * @brief Host-owned adaptive depth observations and explicit admission policy.
 *
 * Request admission selects a measured startup independently of the learned
 * acceptance-window transitions. The controller owns only CPU request state;
 * GPU admission seals the same policy into the device-owned generation ABI.
 * Measured adaptation confirms targets using one mode-exclusive CPU history;
 * partial observations cannot overwrite or duplicate that target authority.
 * Neither startup selection nor adaptation changes the retained graph capacity.
 */
#pragma once

#include "../config/RuntimeConfig.h"
#include "MTPDepthCostProfile.h"
#include "MTPPublicationWitness.h"
#include "MTPReversibleProposalGate.h"
#include "MTPSurvivalDepthSelector.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include <variant>

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
        ModelStopped,
        SurvivalSupportedProbe,
        SurvivalUnresolvedProbe,
        SurvivalHold,
        SurvivalAwaitingConfirmation,
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
        UtilityProbeStarted,
        UtilityProbeKept,
        UtilityProbeReturned,
        UtilityRejectedComparisonHold,
    };

    /** @return Stable diagnostic spelling of a depth decision reason. */
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
        /** Actual response publication delta. Required by utility comparison and
         * trace export; an absent synthetic legacy observation is not inferred
         * from accepted drafts or from the verifier's leading condition row.
         */
        std::optional<uint32_t> newly_emitted_tokens;
        /** Actual publisher-owned progress and debt. Production always supplies
         * this value; absence is retained only for synthetic generic fixtures.
         * Measured progress policy and v3 trace export reject missing evidence.
         */
        std::optional<MTPPublicationWitness> publication;
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
        uint64_t utility_probes = 0;
        uint64_t utility_keeps = 0;
        uint64_t utility_returns = 0;
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
        /** @brief Construct portable bookkeeping for subsequent configuration. */
        MTPDepthController() = default;
        /**
         * @brief Validate and admit one request's depth policy.
         * @param config Authored bounds, immutable domain identity and hysteresis.
         * @param configured_draft_tokens Positive fixed depth, not an adaptive ceiling.
         * @param verify_mode Active request sampling specialization.
         * @throws std::invalid_argument If bounds or policy thresholds are invalid.
         */
        MTPDepthController(
            MTPDepthPolicyConfig config,
            int configured_draft_tokens,
            MTPVerifyMode verify_mode = MTPVerifyMode::Greedy);

        /**
         * @brief Replace policy and reset all observations for a new request.
         * @param config Authored bounds, domain identity and hysteresis.
         * @param configured_draft_tokens Positive fixed-mode draft count.
         * @param verify_mode Active request sampling specialization.
         * @throws std::invalid_argument If the requested policy is inconsistent.
         */
        void configure(
            MTPDepthPolicyConfig config,
            int configured_draft_tokens,
            MTPVerifyMode verify_mode = MTPVerifyMode::Greedy);
        /** @brief Restore the admitted startup depth without changing capacity. */
        void reset();

        /** @return Current live draft depth, including an explicitly enabled zero. */
        int currentDepth() const { return current_depth_; }
        /** @return Next depth, including a scheduled probe out of explicit bypass. */
        int requestedDepthForStep() const;
        /** @return Admitted lower bound. */
        int minDepth() const { return config_.min_depth; }
        /** @return Admitted upper bound, independent of measured startup. */
        int maxDepth() const { return config_.max_depth; }
        /** @return Fixed, observing, or adaptive execution intent. */
        MTPDepthPolicyMode mode() const { return config_.mode; }
        /** @return Request-owned aggregate observation and transition counts. */
        const MTPDepthControllerStats &stats() const { return stats_; }
        /** @return Actual resolved host policy for a bounded replay diagnostic.
         * Automatic thresholds/startup and frozen scope are already resolved by
         * this controller; observers must not re-resolve them through GPU rules.
         */
        const MTPDepthPolicyConfig &resolvedConfig() const noexcept { return config_; }
        /** @return Actual admitted host verification law for exact replay. */
        MTPVerifyMode verifyMode() const noexcept { return verify_mode_; }
        /** @return Actual host-owned reversible comparison, never a GPU mirror. */
        const MTPReversibleProposalState &utilityState() const noexcept { return utility_state_; }
        /** @return Borrowed actual measured confirmation, or null for legacy mode.
         * The pointer is an observation of the sole mode-exclusive host owner;
         * configure/reset may replace that owner, so callers must not retain it.
         */
        const MTPProposalConfirmation *proposalConfirmation() const noexcept
        { return std::get_if<MTPProposalConfirmation>(&proposal_history_); }
        /** @return Immutable measured profile selected from the admitted context. */
        MTPDepthCostProfile costProfile() const noexcept { return cost_profile_; }
        /** @return Last evaluated or deferred transition, for diagnostics. */
        const MTPDepthDecision &lastDecision() const { return last_decision_; }

        /**
         * @brief Observe one completed verifier transaction and evaluate if due.
         * @param observation Actual execution depth and accepted prefix, not capacity.
         * @return Decision after applying policy, hysteresis and explicit bounds.
         */
        MTPDepthDecision recordStep(const MTPDepthObservation &observation);
        /** @return Transition after a step in an explicitly admitted depth-zero mode. */
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

        /** @return Actual window/rates without evaluating a second parent proposal. */
        MTPDepthDecision windowDecision() const;
        /** @return Learned or portable recommendation for the accumulated window. */
        MTPDepthDecision evaluateWindow() const;
        /** @return Whether sufficient unbiased observations permit evaluation. */
        bool windowReady() const;
        /** @return Whether an explicitly admitted depth-zero lane may probe again. */
        bool depthZeroProbeReady() const;

        MTPDepthPolicyConfig config_;
        /** Immutable cost authority; no table storage or live timing state. */
        MTPDepthCostProfile cost_profile_;
        /** Sole actual host comparison; current depth and R remain existing owners. */
        MTPReversibleProposalState utility_state_;
        /** @brief Active verifier mode used to select generated policy rows. */
        MTPVerifyMode verify_mode_ = MTPVerifyMode::Greedy;
        int current_depth_ = 1;
        int steps_since_change_ = 0;
        /** @brief Legacy history is inactive while measured confirmation owns this storage. */
        struct LegacyProposalHistory
        {
            int promotion_streak = 0;
            int generated_best_bad_streak = 0;
        };
        /**
         * @brief Mode-exclusive CPU history, selected at configure/reset.
         *
         * Measured mode owns a persistent confirmation target independent of
         * last_decision_, which is overwritten on every partial observation.
         * The unmeasured mode retains its original two legacy history values.
         */
        std::variant<LegacyProposalHistory,MTPProposalConfirmation> proposal_history_{LegacyProposalHistory{}};
        MTPDepthWindow window_;
        MTPDepthDecision last_decision_;
        MTPDepthControllerStats stats_;
        std::vector<uint8_t> rejected_depths_;
    };

} // namespace llaminar2
