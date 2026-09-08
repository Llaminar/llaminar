/**
 * @file MoEOptimizationStatus.h
 * @brief Typed, passive lifecycle status for adaptive MoE optimization.
 *
 * Runtime policy remains owned by the host residency authority or by the
 * device-resident controller. This value is a read-only projection of that
 * owner state for correctness gates and diagnostics. It is deliberately
 * independent of PerfStats: disabling instrumentation must never change when
 * certification, movement, or publication is considered complete.
 */

#pragma once

#include "backends/DeviceId.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace llaminar2
{
    /** Exact owner of a live ExpertOverlay optimization lifecycle. */
    enum class MoEOptimizationAuthority : std::uint8_t
    {
        None,   ///< No ExpertOverlay execution authority exists.
        Host,   ///< A host-owned multi-tier or heterogeneous authority is live.
        Device, ///< A homogeneous all-GPU device authority is live.
    };

    /** Explicit lifecycle state of adaptive ExpertOverlay movement. */
    enum class MoEOptimizationLifecycleState : std::uint8_t
    {
        NotApplicable,    ///< The model does not use ExpertOverlay.
        MovementDisabled, ///< ExpertOverlay is live under a non-moving policy.
        LearningEconomy,  ///< Measured economy is not yet active in policy.
        Active,           ///< Measured economy is installed and movement may run.
        Drained,          ///< Admission is closed and terminal evidence is stable.
        Failed,           ///< The owning lifecycle reached a terminal failure.
    };

    /**
     * @brief Exact background activity owned by the live optimization authority.
     *
     * Lifecycle readiness and current work are deliberately separate. An
     * `Active` Dynamic policy can be collecting a new demand window, exchanging
     * an already-frozen proposal, moving bytes, or publishing a new residency.
     * Callers that need an immutable measurement epoch must inspect this typed
     * state rather than guessing from a movement counter or optional telemetry.
     */
    enum class MoEOptimizationActivityState : std::uint8_t
    {
        NotApplicable, ///< No ExpertOverlay optimization authority exists.
        Dormant,       ///< Movement is disabled by the selected policy.
        LearningEconomy, ///< Measured policy inputs are not yet installed.
        CollectingDemand, ///< No complete demand window or wave is admitted.
        ReconcilingDemand, ///< The owner is checking for queued demand/work.
        PlanningMovement, ///< One frozen demand generation is being optimized.
        AwaitingAuthorityProposal, ///< A follower owns only its passive mailbox.
        ExchangingProposal, ///< A canonical distributed plan is in flight.
        MovingWeights, ///< Preparation, transfer, or retirement is active.
        PublishingResidency, ///< A prepared placement is becoming selectable.
        Draining, ///< New work is closed while admitted work is reaped.
        Failed,   ///< The activity owner failed terminally.
    };

    /**
     * @brief Monotonic totals published by the actual movement owner.
     *
     * These values describe completed, durable placement changes only. They
     * are operational state projected by the host authority or device
     * controller, not reconstructed from optional telemetry. A caller may
     * subtract two observations to attribute movement to a request interval.
     */
    struct MoEOptimizationMovementTotals
    {
        std::uint64_t transactions = 0u; ///< Physically completed movement waves.
        std::uint64_t commands = 0u; ///< Expert migration commands completed.
        std::uint64_t physical_bytes = 0u; ///< Durable projection payload bytes.
        std::uint64_t promotions = 0u; ///< Moves toward a smaller tier priority.
        std::uint64_t demotions = 0u; ///< Moves toward a larger tier priority.
        std::uint64_t same_priority_moves = 0u; ///< In-tier skew rebalances.
    };

    /** Thermal direction of one completed physical expert movement edge. */
    enum class MoEOptimizationMovementDirection : std::uint8_t
    {
        Promotion,    ///< Destination has a smaller numeric tier priority.
        Demotion,     ///< Destination has a larger numeric tier priority.
        SamePriority, ///< Movement rebalances participants within one priority.
    };

    /**
     * @brief Logical placement objective advanced by one completed move cycle.
     *
     * Physical direction and optimization intent are deliberately independent.
     * A participant rebalance can be folded into the destination of a
     * promotion or demotion, while a same-priority edge can merely close the
     * capacity cycle required by a tier-residency change. Correctness gates
     * must therefore inspect this typed axis instead of inferring intent from
     * endpoint priorities or optional telemetry.
     */
    enum class MoEOptimizationMovementAxis : std::uint8_t
    {
        TierResidency,       ///< Improves expert assignment between priorities.
        ParticipantPlacement, ///< Reduces skew inside an apportioned tier.
        Combined, ///< The enclosing closed cycle advances both objectives.
    };

    /** @return Whether @p axis advances cross-tier residency. */
    [[nodiscard]] constexpr bool advancesTierResidency(
        MoEOptimizationMovementAxis axis) noexcept
    {
        return axis == MoEOptimizationMovementAxis::TierResidency ||
               axis == MoEOptimizationMovementAxis::Combined;
    }

    /** @return Whether @p axis advances within-tier participant placement. */
    [[nodiscard]] constexpr bool advancesParticipantPlacement(
        MoEOptimizationMovementAxis axis) noexcept
    {
        return axis == MoEOptimizationMovementAxis::ParticipantPlacement ||
               axis == MoEOptimizationMovementAxis::Combined;
    }

    /**
     * @brief Typed immutable identity of one durably completed expert move.
     *
     * This record is published by the same host or device authority that owns
     * placement. It is suitable for correctness attribution and CSV export;
     * optional PerfStats rows may mirror it but are never reconstructed into
     * this type.
     */
    struct MoEOptimizationMovementEdge
    {
        MoEOptimizationAuthority authority =
            MoEOptimizationAuthority::None;
        std::uint64_t transaction = 0u;
        std::uint64_t candidate_epoch = 0u;
        int layer = -1;
        int expert = -1;
        std::size_t cycle_index = 0u;
        std::size_t cycle_size = 0u;
        MoEOptimizationMovementDirection direction =
            MoEOptimizationMovementDirection::SamePriority;
        /** Logical objective of this edge's authoritative closed cycle. */
        MoEOptimizationMovementAxis axis =
            MoEOptimizationMovementAxis::TierResidency;
        int source_participant = -1;
        int destination_participant = -1;
        int source_priority = 0;
        int destination_priority = 0;
        DeviceId source_device = DeviceId::invalid();
        DeviceId destination_device = DeviceId::invalid();
        int source_world_rank = -1;
        int destination_world_rank = -1;
        bool source_world_rank_known = false;
        bool destination_world_rank_known = false;
        std::uint64_t estimated_weight_bytes = 0u;
        std::uint64_t activation_count = 0u;
        bool blocking_inference = false;

        /** @brief Compare the complete authoritative movement identity. */
        bool operator==(const MoEOptimizationMovementEdge &) const = default;

        /** @return Whether every identity required by parity is valid. */
        [[nodiscard]] bool valid() const noexcept
        {
            return authority != MoEOptimizationAuthority::None &&
                   transaction > 0u && candidate_epoch > 0u && layer >= 0 &&
                   expert >= 0 && cycle_size > 0u &&
                   source_participant >= 0 &&
                   destination_participant >= 0 &&
                   source_device.is_valid() && destination_device.is_valid();
        }
    };

    /**
     * @brief Immutable economic proof for one committed movement transaction.
     *
     * Only the policy authority publishes this record. Distributed execution
     * followers retain the same movement edges but do not manufacture local
     * service economics. This distinction makes a single coordinator-owned
     * proof available even when PerfStats is disabled on every rank.
     */
    struct MoEOptimizationMovementEconomy
    {
        MoEOptimizationAuthority authority =
            MoEOptimizationAuthority::None;
        std::uint64_t transaction = 0u;
        std::uint64_t candidate_epoch = 0u;
        std::uint64_t command_count = 0u;
        std::uint64_t cycle_count = 0u;
        std::uint64_t projected_service_gain_ns = 0u;
        std::uint64_t projected_transfer_and_repack_ns = 0u;
        std::uint64_t projected_inference_interference_ns = 0u;
        std::uint64_t projected_net_benefit_ns = 0u;

        /** @brief Compare the complete authoritative economy identity. */
        bool operator==(const MoEOptimizationMovementEconomy &) const = default;

        /** @return Whether identity and exact net-benefit arithmetic agree. */
        [[nodiscard]] bool valid() const noexcept
        {
            if (authority == MoEOptimizationAuthority::None ||
                transaction == 0u || candidate_epoch == 0u ||
                command_count == 0u || cycle_count == 0u ||
                projected_service_gain_ns == 0u ||
                projected_net_benefit_ns == 0u ||
                projected_transfer_and_repack_ns >
                    std::numeric_limits<std::uint64_t>::max() -
                        projected_inference_interference_ns)
            {
                return false;
            }
            const std::uint64_t projected_cost_ns =
                projected_transfer_and_repack_ns +
                projected_inference_interference_ns;
            return projected_service_gain_ns > projected_cost_ns &&
                   projected_net_benefit_ns ==
                       projected_service_gain_ns - projected_cost_ns;
        }
    };

    /**
     * @brief Exclusive cycle counts for the two ExpertOverlay placement axes.
     *
     * `combined` names one closed cycle that advances both objectives. It is
     * deliberately not also counted in either pure-axis bucket, so `total()`
     * is the exact number of classified cycles.
     */
    struct MoEOptimizationCycleAxisCounts
    {
        std::uint64_t tier_residency = 0u;
        std::uint64_t participant_placement = 0u;
        std::uint64_t combined = 0u;

        /** @return The exact number of exclusively classified cycles. */
        [[nodiscard]] constexpr std::uint64_t total() const noexcept
        {
            return tier_residency + participant_placement + combined;
        }

        /** @return Whether the buckets sum to @p expected without overflow. */
        [[nodiscard]] constexpr bool matchesTotal(
            std::uint64_t expected) const noexcept
        {
            if (tier_residency > expected)
                return false;
            expected -= tier_residency;
            if (participant_placement > expected)
                return false;
            expected -= participant_placement;
            return combined == expected;
        }

        /** @brief Compare all exclusive axis buckets. */
        bool operator==(const MoEOptimizationCycleAxisCounts &) const = default;
    };

    /** Exact interpretation of a host-authority migration-cycle capacity. */
    enum class MoEOptimizationCycleCapacityKind : std::uint8_t
    {
        Bounded,   ///< A positive preallocated production transfer-slot BOM.
        Unbounded, ///< Protocol-only planning with no physical slot ceiling.
    };

    /**
     * @brief Typed host-policy proof for one admitted movement wave.
     *
     * Only the heterogeneous host policy authority creates this value.
     * Followers execute the authenticated selected plan and retain its movement
     * edges, but must not manufacture coordinator-owned candidate or rejection
     * accounting. Optional PerfStats records may mirror this proof; correctness
     * callers consume this type directly.
     */
    struct MoEOptimizationHostMovementAdmission
    {
        MoEOptimizationAuthority authority =
            MoEOptimizationAuthority::None;
        std::uint64_t transaction = 0u;
        std::uint64_t candidate_epoch = 0u;
        MoEOptimizationCycleCapacityKind cycle_capacity_kind =
            MoEOptimizationCycleCapacityKind::Bounded;
        std::uint64_t maximum_concurrent_cycles = 0u;

        std::uint64_t candidate_cycles = 0u;
        std::uint64_t policy_eligible_cycles = 0u;
        MoEOptimizationCycleAxisCounts policy_eligible_axes;

        std::uint64_t admitted_candidate_cycles = 0u;
        MoEOptimizationCycleAxisCounts admitted_candidate_axes;

        std::uint64_t admitted_physical_cycles = 0u;
        MoEOptimizationCycleAxisCounts admitted_physical_axes;

        std::uint64_t individual_policy_rejected_cycles = 0u;
        std::uint64_t dependent_payoff_rejected_cycles = 0u;
        std::uint64_t capacity_rejected_cycles = 0u;
        std::uint64_t participant_axis_budget_rejected_cycles = 0u;

        /** Dependency combinations considered outside cycle classification. */
        std::uint64_t dependent_cohort_candidates = 0u;
        /** Candidate dependency combinations rejected by joint economics. */
        std::uint64_t dependent_cohort_payoff_rejections = 0u;

        bool physical_cycle_recomposition = false;
        bool capacity_bounded = false;
        bool policy_bounded = false;

        /** @brief Compare the complete authority-owned admission proof. */
        bool operator==(
            const MoEOptimizationHostMovementAdmission &) const = default;

        /**
         * @return Whether identity, total classification, rejection accounting,
         *         flags, and physical transfer capacity agree exactly.
         */
        [[nodiscard]] bool valid() const noexcept
        {
            const bool valid_capacity =
                (cycle_capacity_kind ==
                     MoEOptimizationCycleCapacityKind::Bounded &&
                 maximum_concurrent_cycles > 0u &&
                 admitted_physical_cycles <= maximum_concurrent_cycles) ||
                (cycle_capacity_kind ==
                     MoEOptimizationCycleCapacityKind::Unbounded &&
                 maximum_concurrent_cycles == 0u);
            if (authority != MoEOptimizationAuthority::Host ||
                transaction == 0u || candidate_epoch != transaction ||
                !valid_capacity || candidate_cycles == 0u ||
                admitted_candidate_cycles == 0u ||
                admitted_physical_cycles == 0u ||
                !policy_eligible_axes.matchesTotal(policy_eligible_cycles) ||
                !admitted_candidate_axes.matchesTotal(
                    admitted_candidate_cycles) ||
                !admitted_physical_axes.matchesTotal(
                    admitted_physical_cycles) ||
                dependent_cohort_payoff_rejections >
                    dependent_cohort_candidates)
            {
                return false;
            }

            /* Subtraction-based checks avoid accepting wrapped counter sums. */
            if (individual_policy_rejected_cycles > candidate_cycles ||
                candidate_cycles - individual_policy_rejected_cycles !=
                    policy_eligible_cycles)
            {
                return false;
            }
            if (admitted_candidate_cycles > policy_eligible_cycles)
                return false;
            std::uint64_t unadmitted =
                policy_eligible_cycles - admitted_candidate_cycles;
            const auto consume_rejection = [&unadmitted](
                                               std::uint64_t rejected) noexcept
            {
                if (rejected > unadmitted)
                    return false;
                unadmitted -= rejected;
                return true;
            };
            if (!consume_rejection(dependent_payoff_rejected_cycles) ||
                !consume_rejection(capacity_rejected_cycles) ||
                !consume_rejection(
                    participant_axis_budget_rejected_cycles) ||
                unadmitted != 0u)
            {
                return false;
            }

            return physical_cycle_recomposition ==
                       (admitted_physical_cycles !=
                        admitted_candidate_cycles) &&
                   capacity_bounded == (capacity_rejected_cycles != 0u) &&
                   policy_bounded ==
                       (individual_policy_rejected_cycles != 0u ||
                        dependent_payoff_rejected_cycles != 0u ||
                        participant_axis_budget_rejected_cycles != 0u);
        }
    };

    /**
     * @brief Race-safe model-lifetime movement evidence snapshot.
     *
     * Implementations retain complete edges for the runner lifetime. A future
     * bounded implementation must increase `discarded_edges`; correctness
     * campaigns reject such a truncated snapshot rather than silently losing
     * the edge that selected a numerical witness.
     */
    struct MoEOptimizationMovementLedger
    {
        std::vector<MoEOptimizationMovementEdge> edges;
        std::uint64_t discarded_edges = 0u;
        /** Coordinator/device-policy proofs, one per committed movement wave. */
        std::vector<MoEOptimizationMovementEconomy> economy;
        std::uint64_t discarded_economy_records = 0u;
        /** Host-policy admission proofs; distributed followers omit these. */
        std::vector<MoEOptimizationHostMovementAdmission> host_admissions;
        std::uint64_t discarded_host_admission_records = 0u;

        /** @return Whether the complete runner-lifetime ledger is present. */
        [[nodiscard]] bool complete() const noexcept
        {
            return discarded_edges == 0u &&
                   discarded_economy_records == 0u &&
                   discarded_host_admission_records == 0u;
        }
    };

    /**
     * @brief Passive occupancy of the authority's active routing-demand window.
     *
     * This value describes the exact RCU histogram bank accepting new routed
     * rows. It is not a reservation: a caller may use it to admit an immutable
     * measurement cohort only while that caller owns exclusive inference
     * admission. Keeping the capacity beside the count prevents callers from
     * copying a controller window constant or inferring lifecycle state from
     * optional PerfStats counters.
     */
    struct MoEOptimizationDemandWindow
    {
        std::uint64_t generation = 0u; ///< Active RCU histogram generation.
        std::uint64_t collected_routed_rows = 0u; ///< Rows already admitted.
        std::uint64_t capacity_routed_rows = 0u; ///< Decision threshold.

        /** @return Whether the authority published a usable window geometry. */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            return capacity_routed_rows > 0u;
        }

        /**
         * @return Rows that can be added before the window reaches its decision
         *         threshold, or zero when the threshold is already complete.
         */
        [[nodiscard]] constexpr std::uint64_t remainingRoutedRows() const noexcept
        {
            return valid() && collected_routed_rows < capacity_routed_rows
                       ? capacity_routed_rows - collected_routed_rows
                       : 0u;
        }

        /**
         * @brief Test whether one exclusive cohort stays below the next decision.
         * @param routed_rows Exact routed rows the cohort will execute.
         * @return True only when adding every row remains strictly below the
         *         window threshold.
         *
         * Equality is deliberately rejected: the final row would complete the
         * demand window and allow asynchronous proposal/publication while the
         * caller is still validating the cohort's terminal epoch.
         */
        [[nodiscard]] constexpr bool canAdmitExclusiveCohort(
            std::uint64_t routed_rows) const noexcept
        {
            return routed_rows > 0u && routed_rows < remainingRoutedRows();
        }
    };

    /** Monotonic relationship between two optimization progress snapshots. */
    enum class MoEOptimizationProgressRelation : std::uint8_t
    {
        Regressed, ///< At least one authoritative counter moved backwards.
        Unchanged, ///< No durable, demand, or reconciliation edge advanced.
        Advanced,  ///< At least one authoritative lifecycle edge advanced.
    };

    /**
     * @brief Compact monotonic stamp for a live optimization lifecycle.
     *
     * Long-running placement decisions and asynchronous histogram drains may
     * span more than one wall-clock interval while still completing an
     * independently bounded wave on every interval.  This stamp lets a
     * lifecycle watchdog distinguish that healthy progress from a stalled
     * authority without treating transient activity enum changes as progress.
     * Demand occupancy is ordered lexicographically by RCU generation and row
     * count because rotating a full bank deliberately resets its row count.
     */
    struct MoEOptimizationProgressStamp
    {
        std::uint64_t published_movement_waves = 0u;
        std::uint64_t completed_transactions = 0u;
        std::uint64_t completed_commands = 0u;
        std::uint64_t completed_physical_bytes = 0u;
        std::uint64_t completed_promotions = 0u;
        std::uint64_t completed_demotions = 0u;
        std::uint64_t completed_same_priority_moves = 0u;
        std::uint64_t demand_generation = 0u;
        std::uint64_t demand_rows = 0u;
        std::uint64_t published_progress_generation = 0u;
        std::uint64_t reconciled_progress_generation = 0u;

        /**
         * @brief Classify this observation relative to an earlier stamp.
         * @param previous Earlier observation from the same authority lifetime.
         * @return Regressed, unchanged, or advanced monotonic progress.
         */
        [[nodiscard]] constexpr MoEOptimizationProgressRelation relationTo(
            const MoEOptimizationProgressStamp &previous) const noexcept
        {
            const bool scalar_regressed =
                published_movement_waves <
                    previous.published_movement_waves ||
                completed_transactions < previous.completed_transactions ||
                completed_commands < previous.completed_commands ||
                completed_physical_bytes <
                    previous.completed_physical_bytes ||
                completed_promotions < previous.completed_promotions ||
                completed_demotions < previous.completed_demotions ||
                completed_same_priority_moves <
                    previous.completed_same_priority_moves ||
                published_progress_generation <
                    previous.published_progress_generation ||
                reconciled_progress_generation <
                    previous.reconciled_progress_generation;
            const bool demand_regressed =
                demand_generation < previous.demand_generation ||
                (demand_generation == previous.demand_generation &&
                 demand_rows < previous.demand_rows);
            if (scalar_regressed || demand_regressed)
                return MoEOptimizationProgressRelation::Regressed;

            const bool scalar_advanced =
                published_movement_waves >
                    previous.published_movement_waves ||
                completed_transactions > previous.completed_transactions ||
                completed_commands > previous.completed_commands ||
                completed_physical_bytes >
                    previous.completed_physical_bytes ||
                completed_promotions > previous.completed_promotions ||
                completed_demotions > previous.completed_demotions ||
                completed_same_priority_moves >
                    previous.completed_same_priority_moves ||
                published_progress_generation >
                    previous.published_progress_generation ||
                reconciled_progress_generation >
                    previous.reconciled_progress_generation;
            const bool demand_advanced =
                demand_generation > previous.demand_generation ||
                (demand_generation == previous.demand_generation &&
                 demand_rows > previous.demand_rows);
            return scalar_advanced || demand_advanced
                       ? MoEOptimizationProgressRelation::Advanced
                       : MoEOptimizationProgressRelation::Unchanged;
        }
    };

    /**
     * @brief Immutable observation of the sole MoE optimization authority.
     *
     * `published_movement_waves` is zero at the initial placement and advances
     * once for each durable placement publication on every authority kind. A
     * device authority may publish a new RCU-selectable residency before its
     * asynchronous source retirement completes; callers requiring completed
     * physical work must additionally observe
     * `completed_movement.transactions`. The counter is intentionally
     * normalized rather than exposing host placement epochs (which begin at
     * one) beside device movement counters (which begin at zero). Callers may
     * observe progress but cannot mutate or advance it.
     */
    struct MoEOptimizationStatus
    {
        MoEOptimizationAuthority authority =
            MoEOptimizationAuthority::None;
        MoEOptimizationLifecycleState state =
            MoEOptimizationLifecycleState::NotApplicable;
        MoEOptimizationActivityState activity =
            MoEOptimizationActivityState::NotApplicable;
        std::uint64_t published_movement_waves = 0u;
        /** Cumulative completed physical movement from the owning authority. */
        MoEOptimizationMovementTotals completed_movement;
        /** Active authority-owned demand window, when host policy owns one. */
        MoEOptimizationDemandWindow demand_window;
        /**
         * Latest externally published maintenance-progress generation.
         *
         * Host inference increments this generation before waking its
         * background authority. Device-only authorities leave both progress
         * generations at zero because their captured controller owns the
         * complete decision boundary.
         */
        std::uint64_t published_progress_generation = 0u;
        /**
         * Latest progress generation for which the host authority observed no
         * complete demand window after polling every active source.
         */
        std::uint64_t reconciled_progress_generation = 0u;
        std::string diagnostic;

        /** @return Whether a Dynamic authority may currently publish movement. */
        [[nodiscard]] bool active() const noexcept
        {
            return state == MoEOptimizationLifecycleState::Active;
        }

        /**
         * @return Monotonic lifecycle stamp suitable for a no-progress watchdog.
         *
         * Activity is intentionally absent: `ReconcilingDemand` and
         * `MovingWeights` may alternate while retrying one unchanged edge.
         * Only durable movement, RCU demand, or published reconciliation
         * authority can renew a bounded watchdog.
         */
        [[nodiscard]] constexpr MoEOptimizationProgressStamp
        progressStamp() const noexcept
        {
            return {
                .published_movement_waves = published_movement_waves,
                .completed_transactions =
                    completed_movement.transactions,
                .completed_commands = completed_movement.commands,
                .completed_physical_bytes =
                    completed_movement.physical_bytes,
                .completed_promotions = completed_movement.promotions,
                .completed_demotions = completed_movement.demotions,
                .completed_same_priority_moves =
                    completed_movement.same_priority_moves,
                .demand_generation = demand_window.generation,
                .demand_rows = demand_window.collected_routed_rows,
                .published_progress_generation =
                    published_progress_generation,
                .reconciled_progress_generation =
                    reconciled_progress_generation,
            };
        }

        /**
         * @return Whether no complete demand window or admitted wave is active.
         *
         * A distributed follower's preposted mailbox is also quiescent: it
         * owns no policy decision or physical transaction until the authority
         * publishes a proposal. A host policy authority additionally requires
         * every inference/event notification visible in this snapshot to have
         * been reconciled. This prevents a stale `CollectingDemand` state from
         * masquerading as a boundary immediately after inference fills a new
         * histogram window.
         */
        [[nodiscard]] bool quiescentBetweenWaves() const noexcept
        {
            return active() &&
                   published_progress_generation ==
                       reconciled_progress_generation &&
                   (activity ==
                        MoEOptimizationActivityState::CollectingDemand ||
                    activity == MoEOptimizationActivityState::
                        AwaitingAuthorityProposal);
        }

        /**
         * @brief Decide whether an exclusive cohort may begin without movement.
         * @param routed_rows Exact routed rows executed by the complete cohort.
         * @return True only at a reconciled between-wave boundary with enough
         *         typed demand-window headroom.
         *
         * This is a passive admission fact, not a lease. The caller must own
         * exclusive request admission from this observation through the cohort;
         * ordinary concurrent servers should never use it as a synchronization
         * primitive.
         */
        [[nodiscard]] bool canBeginExclusiveCohort(
            std::uint64_t routed_rows) const noexcept
        {
            return quiescentBetweenWaves() &&
                   demand_window.canAdmitExclusiveCohort(routed_rows);
        }

        /** @return Whether measured Dynamic economy is still being learned. */
        [[nodiscard]] bool learning() const noexcept
        {
            return state ==
                MoEOptimizationLifecycleState::LearningEconomy;
        }

        /** @return Whether admission closed with immutable terminal evidence. */
        [[nodiscard]] bool drained() const noexcept
        {
            return state == MoEOptimizationLifecycleState::Drained;
        }

        /** @return Whether the owning lifecycle has failed terminally. */
        [[nodiscard]] bool failed() const noexcept
        {
            return state == MoEOptimizationLifecycleState::Failed;
        }
    };
} // namespace llaminar2
