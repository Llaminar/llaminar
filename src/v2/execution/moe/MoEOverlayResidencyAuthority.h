/**
 * @file MoEOverlayResidencyAuthority.h
 * @brief Versioned histogram-driven residency for heterogeneous MoE overlays.
 *
 * ExpertOverlay dispatch crosses explicit device and process boundaries.  A
 * placement update therefore cannot be represented as a collection of mutable
 * masks: every dispatch ticket must observe one complete owner map, and every
 * destination must own prepared expert weights before that map becomes live.
 * This file defines the single publication authority and its transactional
 * transport boundary.  It is deliberately device-free so the lifecycle can be
 * proved exhaustively in unit tests; production transports bind the protocol to
 * prepared-weight services and exact backend transfer streams.
 */

#pragma once

#include "DecodeExpertHistogram.h"
#include "DeviceMoERebalancePolicyShared.h"
#include "MoEExpertOwnerMap.h"
#include "MoELayeredExpertOwnership.h"
#include "MoEOptimizationStatus.h"
#include "MoERoutedExpertPlacementPlanner.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace llaminar2
{
    /** @brief Thermal direction of one complete expert residency move. */
    enum class MoEOverlayTierMigrationDirection
    {
        Promotion,
        Demotion,
        SamePriority,
    };

    /**
     * @brief Exact source and destination for one layer/expert weight triplet.
     *
     * Tier priority is a compute-capacity ordering: a smaller numeric priority
     * is hotter.  Keeping both resolved owners in the record lets transports
     * distinguish same-rank peer copies, cross-backend copies, and true
     * cross-domain/rank transfers without reconstructing topology from strings.
     */
    struct MoEOverlayTierMigration
    {
        int layer_idx = -1;
        int expert_id = -1;
        uint64_t activation_count = 0;
        size_t estimated_weight_bytes = 0;
        MoEOverlayTierMigrationDirection direction =
            MoEOverlayTierMigrationDirection::SamePriority;
        /** Objective of this edge's closed capacity-preserving cycle. */
        MoEOptimizationMovementAxis axis =
            MoEOptimizationMovementAxis::TierResidency;
        MoEExpertOwner source;
        MoEExpertOwner destination;

        bool crossesTier() const noexcept
        {
            return source.tier_idx != destination.tier_idx;
        }

        bool crossesDomain() const noexcept
        {
            return source.domain_name != destination.domain_name;
        }

        bool crossesWorldRank() const noexcept
        {
            return source.owner_world_rank_known &&
                   destination.owner_world_rank_known &&
                   source.owner_world_rank != destination.owner_world_rank;
        }

        bool crossesBackend() const noexcept
        {
            return source.device.type != destination.device.type;
        }
    };

    /**
     * @brief Exact slot-flow proof for one immutable migration edge set.
     *
     * Thermal direction counts are diagnostic only: a closed three-tier cycle
     * may contain one direct promotion and two stepwise demotions. Capacity is
     * preserved when every `(layer, participant)` and `(layer, tier)` vertex
     * has equal incoming and outgoing expert slots. This typed summary keeps
     * that graph invariant distinct from priority-direction accounting.
     */
    struct MoEOverlayMigrationCapacityEvidence
    {
        size_t edges_checked = 0;
        size_t participant_coordinates_checked = 0;
        size_t tier_coordinates_checked = 0;
        size_t malformed_edges = 0;
        size_t participant_flow_violations = 0;
        size_t tier_flow_violations = 0;

        /** @return Whether every physical and logical capacity flow balances. */
        [[nodiscard]] bool capacityPreserved() const noexcept
        {
            return malformed_edges == 0 &&
                   participant_flow_violations == 0 &&
                   tier_flow_violations == 0;
        }
    };

    /**
     * @brief Prove per-layer participant and tier slot-flow conservation.
     * @param migrations Complete immutable edge set for one candidate wave.
     * @return Typed conservation evidence independent of direction counts.
     */
    [[nodiscard]] MoEOverlayMigrationCapacityEvidence
    analyzeMoEOverlayMigrationCapacity(
        std::span<const MoEOverlayTierMigration> migrations) noexcept;

    /**
     * @brief One closed, capacity-preserving migration cycle within a layer.
     *
     * Indices refer to the enclosing transaction's immutable migration vector.
     * The destination participant of each indexed edge equals the source
     * participant of the next edge, including the last-to-first wrap. A
     * one-edge cycle represents a logical relocation whose physical endpoint
     * is unchanged. Cycles are the smallest units that may be selected for a
     * bounded-shadow publication wave without changing any physical
     * participant's logical expert capacity.
     */
    struct MoEOverlayTierMigrationCycle
    {
        int layer_idx = -1;
        std::vector<size_t> migration_indices;

        /**
         * @brief Validate closure, layer identity, bounds, and unique edges.
         * @param migrations Enclosing transaction's complete movement vector.
         * @return Whether this is one well-formed directed participant cycle.
         */
        [[nodiscard]] bool valid(
            const std::vector<MoEOverlayTierMigration> &migrations) const noexcept;
    };

    /**
     * @brief Inactive expert slots required at one exact destination endpoint.
     *
     * A destination cannot overwrite an active slot because old-epoch tickets
     * may still execute there after publication. The transport must reserve at
     * least this many endpoint-owned inactive slots before enqueueing the wave.
     */
    struct MoEOverlayTierShadowRequirement
    {
        int layer_idx = -1;
        int tier_idx = -1;
        int destination_participant = -1;
        size_t slot_count = 0;

        /** @brief Compare the complete physical endpoint and capacity. */
        bool operator==(const MoEOverlayTierShadowRequirement &) const = default;
    };

    /**
     * @brief Measured cost of moving one complete expert between endpoints.
     *
     * One row covers gate, up, and down projections for the exact model layer,
     * prepared codebook, staging geometry, and directed physical endpoint pair.
     * Calibration observes the direction while its capacity-preserving reverse
     * edge is live, so reciprocal rows describe one paired-wave critical path
     * rather than two additive byte-serial charges. A reciprocal two-edge cycle
     * therefore takes the slower row once. Longer cycles and multiple admitted
     * cycles remain conservatively additive until a profile supplies an exact
     * composite-wave contention identity. The transfer term includes conversion
     * or repacking where required; interference remains separate so overlap cost
     * is observable.
     */
    struct MoEOverlayParticipantLayerMigrationCost
    {
        int source_participant = -1;
        int destination_participant = -1;
        int layer = -1;
        uint64_t transfer_and_repack_ns = 0;
        uint64_t inference_interference_ns = 0;
    };

    /**
     * @brief Complete setup-certified directed endpoint migration profile.
     *
     * A multi-participant profile contains exactly one row for every distinct
     * source/destination participant pair and model layer. Participant ids are
     * resolved topology ids, not assumptions about backend/socket placement.
     */
    struct MoEOverlayMigrationCostProfile
    {
        std::string identity;
        std::vector<MoEOverlayParticipantLayerMigrationCost> costs;
    };

    /**
     * @brief Deterministic smoothing, payoff, and anti-oscillation policy.
     *
     * A cycle is eligible only when projected service savings over the
     * token-denominated payoff horizon exceed its certified movement cost,
     * measured inference interference, and the configured minimum net benefit.
     * Reciprocal pair swaps use their measured parallel critical path; shapes
     * without matching composite calibration are priced conservatively.
     * Recently moved experts remain pinned for a whole number of histogram
     * generations.
     */
    struct MoEOverlayMigrationEconomyPolicy
    {
        uint32_t historical_window_weight = 3;
        uint32_t current_window_weight = 1;
        /** Expected routed-token lifetime over which movement must amortize. */
        uint64_t payoff_horizon_tokens =
            moe_rebalance_policy::kDefaultMigrationPayoffHorizonTokens;
        uint64_t minimum_net_benefit_ns = 0;
        uint64_t minimum_residency_generations = 2;

        /** @return Whether every arithmetic and lifecycle field is usable. */
        [[nodiscard]] bool valid() const noexcept;
    };

    /**
     * @brief Histogram-driven ownership policy inside one unchanged tier.
     *
     * ExpertOverlay tier assignment controls which domain may host an expert;
     * this policy controls the exact apportioned participant inside that
     * domain. Every accepted choice is a paired whole-expert swap, preserving
     * both tier quota and participant resident-slot capacity. The planner is a
     * subordinate part of one ExpertOverlay epoch and never publishes its own
     * runtime table.
     */
    struct MoEOverlayParticipantRebalancePolicy
    {
        bool enabled = true;
        uint32_t imbalance_threshold_per_mille =
            moe_rebalance_policy::kDefaultDynamicImbalanceThresholdPerMille;
        uint32_t minimum_improvement_per_mille =
            moe_rebalance_policy::kDefaultDynamicMinImprovementPerMille;
        uint32_t maximum_swaps_per_layer =
            moe_rebalance_policy::kDefaultDynamicMaxSwapsPerLayer;
        uint32_t maximum_plan_entries_per_wave =
            moe_rebalance_policy::kDefaultDynamicMaxPlanEntriesPerWave;
        uint64_t minimum_window_activations =
            moe_rebalance_policy::kDefaultDynamicMinWindowActivations;

        /** @return Whether thresholds and paired-entry capacity are coherent. */
        [[nodiscard]] bool valid() const noexcept
        {
            return !enabled ||
                   (imbalance_threshold_per_mille >=
                        moe_rebalance_policy::
                            kMinimumDynamicImbalanceThresholdPerMille &&
                   maximum_swaps_per_layer > 0u &&
                   maximum_plan_entries_per_wave >= 2u);
        }
    };

    /**
     * @brief Pointer-free economic proof attached to one candidate wave.
     *
     * The coordinator's policy-audit fingerprint includes this record. The
     * separately typed execution fingerprint deliberately excludes it so a
     * follower can execute the selected plan without manufacturing a local
     * copy of coordinator-owned measurements.
     */
    struct MoEOverlayMigrationEconomyEvidence
    {
        bool enabled = false;
        std::string service_profile_identity;
        std::string migration_profile_identity;
        uint64_t smoothed_through_generation = 0;
        uint32_t historical_window_weight = 0;
        uint32_t current_window_weight = 0;
        uint64_t payoff_horizon_tokens = 0;
        uint64_t minimum_net_benefit_ns = 0;
        uint64_t minimum_residency_generations = 0;
        uint64_t projected_service_gain_ns = 0;
        uint64_t projected_transfer_and_repack_ns = 0;
        uint64_t projected_inference_interference_ns = 0;
        uint64_t projected_net_benefit_ns = 0;
        uint64_t payoff_rejected_cycles = 0;
        uint64_t residency_rejected_cycles = 0;

        /** @return Whether disabled-zero or enabled evidence is coherent. */
        [[nodiscard]] bool valid(bool has_migrations) const noexcept;
    };

    /** @brief Immutable placement state consumed by one or more dispatch tickets. */
    struct MoEOverlayResidencySnapshot
    {
        uint64_t epoch = 0;
        std::shared_ptr<const MoERoutedExpertPlacementPlan> placement_plan;
        MoEExpertOwnerMap owner_map;
        MoELayeredExpertOwnership layered_ownership;

        bool valid() const noexcept
        {
            return epoch > 0 && placement_plan != nullptr &&
                   !layered_ownership.empty();
        }
    };

    /** @brief Typed intent separating publishable placement from calibration. */
    enum class MoEOverlayResidencyTransactionPurpose : std::uint8_t
    {
        PlacementChange,   ///< A candidate epoch that may become authoritative.
        EconomyCalibration, ///< A staged copy that must be aborted, never committed.
        /**
         * Restore the model-preparation placement before its context is reused.
         *
         * This is a terminal runner-lifecycle transaction. It uses the same
         * asynchronous physical transport and RCU publication protocol as an
         * ordinary placement change, but it is not a histogram/economy
         * decision and must not be counted as an inference-time optimization.
         */
        PreparedContextRestoration,
    };

    /**
     * @brief One complete, capacity-preserving candidate residency wave.
     *
     * `previous` and `candidate` are retained for the whole transport wave.
     * A stale transaction is rejected if another epoch was published after it
     * was planned.  Transports must stage every migration in `migrations`
     * before acknowledging the staging phase.
     */
    struct MoEOverlayResidencyTransaction
    {
        MoEOverlayResidencyTransactionPurpose purpose =
            MoEOverlayResidencyTransactionPurpose::PlacementChange;
        /** Unique setup-owned identity for a repeated calibration sample. */
        uint64_t calibration_sequence = 0;
        uint64_t expected_epoch = 0;
        uint64_t histogram_generation = 0;
        /**
         * Exact immutable planning evidence used for every placement, movement,
         * and economy calculation in this transaction. When temporal smoothing
         * is enabled this is the smoothed window, not the just-rotated raw input
         * bank. The raw bank remains private input to the authority's smoothing
         * state and must never be published beside migrations derived from a
         * different window.
         */
        std::shared_ptr<const DecodeExpertHistogramWindow> histogram_window;
        std::shared_ptr<const MoEOverlayResidencySnapshot> previous;
        std::shared_ptr<const MoEOverlayResidencySnapshot> candidate;
        std::vector<MoEOverlayTierMigration> migrations;
        std::vector<MoEOverlayTierMigrationCycle> migration_cycles;
        std::vector<MoEOverlayTierShadowRequirement> shadow_requirements;
        /** Coordinator-only typed candidate/admission accounting. */
        std::optional<MoEOptimizationHostMovementAdmission> host_admission;
        MoEOverlayMigrationEconomyEvidence economy;

        bool empty() const noexcept { return migrations.empty(); }

        /**
         * @brief Validate epoch identity and the complete cycle/reservation plan.
         * @return Whether every movement appears in exactly one closed cycle and
         *         every destination is covered by an exact shadow requirement.
         */
        [[nodiscard]] bool valid() const noexcept;
    };

    /**
     * @brief One dense root-authored candidate coordinate for distributed use.
     *
     * Every model `(layer, expert)` coordinate names its candidate tier and
     * physical participant.  Changed coordinates additionally retain the
     * exact execution metadata used by migration staging.  Unchanged entries
     * keep that metadata zero so the wire representation has one canonical
     * form and cannot hide a second sparse change set.
     */
    struct MoEOverlayAuthoritativeResidencyEntry
    {
        int candidate_tier_idx = -1;
        int candidate_owner_participant = -1;
        bool changed = false;
        MoEOptimizationMovementAxis axis =
            MoEOptimizationMovementAxis::TierResidency;
        uint64_t activation_count = 0;
        size_t estimated_weight_bytes = 0;

        /** @return Whether this is one canonical dense proposal entry. */
        [[nodiscard]] bool valid() const noexcept;

        bool operator==(
            const MoEOverlayAuthoritativeResidencyEntry &) const = default;
    };

    /**
     * @brief Canonical execution proposal authored by one distributed root.
     *
     * This value intentionally excludes economy profiles.  Economy is policy
     * evidence owned by the continuation/root authority; followers need the
     * selected executable candidate, not a local copy of the measurements
     * which selected it.  The distributed protocol authenticates this value
     * and separately carries the root's complete policy-audit fingerprint.
     */
    struct MoEOverlayAuthoritativeResidencyPlan
    {
        uint64_t expected_epoch = 0;
        int num_layers = 0;
        int num_experts = 0;
        std::shared_ptr<const DecodeExpertHistogramWindow> histogram_window;
        std::vector<MoEOverlayAuthoritativeResidencyEntry> entries;

        /** @return Whether geometry, evidence, and dense entries are coherent. */
        [[nodiscard]] bool valid() const noexcept;

        /** @return Flattened index for one checked layer/expert coordinate. */
        [[nodiscard]] size_t offset(int layer_idx, int expert_id) const;
    };

    /** @brief Non-blocking readiness returned by a background migration wave. */
    enum class MoEOverlayResidencyWaveProgress
    {
        Pending, ///< Device/network work remains in flight; do not publish.
        Ready,   ///< The phase's exact completion event has become visible.
        Deferred, ///< Global staging backpressure requires whole-wave retry.
        Failed,  ///< The phase failed and the unpublished candidate must abort.
    };

    /** @brief Whether exact host admission for the old epoch remains open. */
    enum class MoEOverlayRetirementAdmissionState : std::uint8_t
    {
        Open = 0, ///< A captured old-epoch ticket may still become a host reader.
        Closed,   ///< The device grace period completed; no new reader can enter.
    };

    /** @brief Process-local old-epoch host reader-count state. */
    enum class MoEOverlayRetirementReaderState : std::uint8_t
    {
        Active = 0, ///< At least one admitted old-epoch host reader remains.
        Drained,    ///< The process-local old-epoch reader count is zero.
    };

    /** @brief Complete source-owned local state supplied to a retirement poll. */
    struct MoEOverlayLocalRetirementState
    {
        MoEOverlayRetirementAdmissionState admission =
            MoEOverlayRetirementAdmissionState::Open;
        MoEOverlayRetirementReaderState readers =
            MoEOverlayRetirementReaderState::Active;
    };

    /** @brief Typed result unique to the two-step old-bank grace period. */
    enum class MoEOverlayRetirementFenceProgress : std::uint8_t
    {
        Pending = 0, ///< A local/device/rank readiness edge is still outstanding.
        ReadyToCloseAdmission, ///< Device and current-reader grace periods align.
        ReadyToRetire, ///< Admission is closed and every rank's readers drained.
        Failed, ///< The publication/retirement lifecycle is invalid.
    };

    /**
     * @brief Absolute host-monotonic interval for one completed staging wave.
     *
     * The values are meaningful only inside the process that observed them;
     * distributed wrappers publish their own interval spanning local staging
     * and all-rank readiness consensus. They are never compared across clocks.
     */
    struct MoEOverlayResidencyWaveInterval
    {
        std::uint64_t begin_steady_nanoseconds = 0;
        std::uint64_t end_steady_nanoseconds = 0;

        /** @return Whether this is one positive monotonic interval. */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            return begin_steady_nanoseconds != 0 &&
                   end_steady_nanoseconds > begin_steady_nanoseconds;
        }

        /** @return Whether the supplied interval is wholly inside this wave. */
        [[nodiscard]] constexpr bool contains(
            std::uint64_t begin_nanoseconds,
            std::uint64_t end_nanoseconds) const noexcept
        {
            return valid() && begin_nanoseconds >= begin_steady_nanoseconds &&
                   end_nanoseconds <= end_steady_nanoseconds &&
                   end_nanoseconds > begin_nanoseconds;
        }
    };

    /**
     * @brief Owned asynchronous work for one candidate residency epoch.
     *
     * A wave pins all source engines and owns every inactive destination slot,
     * staging buffer, event, and transport lifetime needed by its transaction.
     * Poll methods query readiness only; they must never synchronize a stream,
     * wait for a collective, or run work on an inference thread. Preparation
     * builds and authenticates every inactive host/device runtime bank without
     * changing admission. Publication then flips the already-ready device
     * selectors. Only after publication readiness is globally observed may the
     * residency authority advance the public admission floor.
     */
    class IMoEOverlayResidencyWave
    {
    public:
        virtual ~IMoEOverlayResidencyWave() = default;

        /**
         * @brief Query background preparation/transfer readiness without waiting.
         * @param error Receives a precise diagnostic when `Failed` is returned.
         * @return Current staging progress.
         */
        virtual MoEOverlayResidencyWaveProgress pollStage(
            std::string *error) noexcept = 0;

        /**
         * @brief Return this process's exact interval after staging is Ready.
         *
         * Device-free test doubles may omit timing. Economy calibration treats
         * a missing interval as unusable evidence and never estimates it.
         */
        [[nodiscard]] virtual std::optional<
            MoEOverlayResidencyWaveInterval>
        completedStageInterval() const noexcept
        {
            return std::nullopt;
        }

        /**
         * @brief Enqueue construction of every inactive destination bank.
         * @param error Receives a precise enqueue/validation failure.
         * @return Whether preparation work was enqueued without blocking.
         */
        virtual bool beginPrepare(std::string *error) noexcept = 0;

        /**
         * @brief Query inactive-bank preparation readiness without waiting.
         * @param error Receives a precise diagnostic when `Failed` is returned.
         * @return Current preparation progress.
         */
        virtual MoEOverlayResidencyWaveProgress pollPrepare(
            std::string *error) noexcept = 0;

        /**
         * @brief Enqueue the inference-visible selector publication fan-out.
         * @param error Receives a precise enqueue/validation failure.
         * @return Whether every process-local publication was submitted.
         *
         * The candidate epoch is already exact-addressable through the host
         * authority before this method is called. Once any selector can have
         * changed, failure is fatal: rollback could invalidate an admitted
         * device ticket and is therefore not a legal lifecycle transition.
         */
        virtual bool beginPublication(std::string *error) noexcept = 0;

        /**
         * @brief Query selector publication and all-rank consensus readiness.
         * @param error Receives a precise diagnostic when `Failed` is returned.
         * @return Current publication progress.
         */
        virtual MoEOverlayResidencyWaveProgress pollPublication(
            std::string *error) noexcept = 0;

        /** @brief Abort and recycle an unpublished candidate asynchronously. */
        virtual void abortStaged() noexcept = 0;

        /**
         * @brief Query whether asynchronous abort cleanup has quiesced.
         * @param error Receives a precise cleanup failure diagnostic.
         * @return Pending while resources remain referenced by background work,
         *         Ready when this wave may be destroyed, or Failed on a broken
         *         cleanup/event edge.
         *
         * The default is appropriate for device-free waves whose `abortStaged()`
         * completes synchronously. Device and network waves override this and
         * retain their buffers, events, source pins, and destination slots until
         * the exact completion authorities report Ready.
         */
        virtual MoEOverlayResidencyWaveProgress pollAbort(
            std::string *error) noexcept
        {
            (void)error;
            return MoEOverlayResidencyWaveProgress::Ready;
        }

        /**
         * @brief Observe the authority's successful public-floor transition.
         *
         * This control-plane callback runs immediately after publication. It
         * must not launch work, allocate, synchronize, or fail; physical bank
         * visibility was already proved by `pollPublication()`. Device-free waves
         * need no notification, so the default is intentionally empty.
         */
        virtual void markAuthorityPublished() noexcept {}

        /**
         * @brief Poll the cross-participant lease-drain fence for the old epoch.
         *
         * The authority calls this throughout the local grace period. Before
         * exact admission closes, the wave first proves that every rank has no
         * current host readers and no device producer capable of materializing
         * a delayed old-epoch ticket. The authority then closes admission and a
         * second aligned generation proves that any acquisition racing the close
         * has departed. This keeps ranks in identical asynchronous generations
         * without rejecting valid delayed device tickets or blocking inference.
         *
         * @param local_state Source-owned admission and host-reader state.
         * @param error Receives a precise retirement-fence failure.
         * @return A typed request to close admission, final retirement readiness,
         *         pending progress, or failure.
         */
        virtual MoEOverlayRetirementFenceProgress pollRetirementFence(
            MoEOverlayLocalRetirementState local_state,
            std::string *error) noexcept
        {
            if (error)
                error->clear();
            if (local_state.readers ==
                MoEOverlayRetirementReaderState::Active)
            {
                return MoEOverlayRetirementFenceProgress::Pending;
            }
            return local_state.admission ==
                           MoEOverlayRetirementAdmissionState::Open
                       ? MoEOverlayRetirementFenceProgress::
                             ReadyToCloseAdmission
                       : MoEOverlayRetirementFenceProgress::ReadyToRetire;
        }

        /**
         * @brief Retire old engines after the previous epoch's leases drain.
         *
         * Called only by the maintenance worker. The method may enqueue ordered
         * cleanup but must not synchronize or run destruction on final return.
         */
        virtual void retirePrevious() noexcept = 0;
    };

    /** @brief Result of reserving and enqueueing one background migration wave. */
    enum class MoEOverlayResidencyStageStartStatus
    {
        Started,  ///< Source pins, inactive slots, and work were installed.
        Deferred, ///< Shadow capacity/backpressure requires a later retry.
        Failed,   ///< Validation or enqueue failed; the wave is not retryable.
    };

    /** @brief Typed ownership returned by a transport's non-blocking start. */
    struct MoEOverlayResidencyStageStart
    {
        MoEOverlayResidencyStageStartStatus status =
            MoEOverlayResidencyStageStartStatus::Failed;
        std::unique_ptr<IMoEOverlayResidencyWave> wave;
        /** Partially enqueued unpublished work retained only for async cleanup. */
        std::unique_ptr<IMoEOverlayResidencyWave> cleanup_wave;
        std::string error;

        /** @return Whether status and wave ownership form a valid start result. */
        [[nodiscard]] bool valid() const noexcept
        {
            switch (status)
            {
            case MoEOverlayResidencyStageStartStatus::Started:
                return wave != nullptr && cleanup_wave == nullptr;
            case MoEOverlayResidencyStageStartStatus::Deferred:
                return wave == nullptr && cleanup_wave == nullptr;
            case MoEOverlayResidencyStageStartStatus::Failed:
                return wave == nullptr;
            }
            return false;
        }
    };

    /**
     * @brief Factory boundary for event-driven prepared-weight migration.
     *
     * `beginStage()` runs on a maintenance worker, pins immutable sources,
     * reserves inactive destinations, and enqueues work on persistent background
     * lanes. It must return without waiting for GPU, CPU, network, or tickets.
     */
    class IMoEOverlayResidencyTransport
    {
    public:
        virtual ~IMoEOverlayResidencyTransport() = default;

        /**
         * @brief Start an asynchronous candidate wave.
         * @param transaction Exact old/new epoch and migration set.
         * @return Typed started, deferred, or failed result with owned wave.
         */
        virtual MoEOverlayResidencyStageStart beginStage(
            const MoEOverlayResidencyTransaction &transaction) = 0;

        /**
         * @brief Return bytes belonging to completed durable placement waves.
         *
         * The default supports structural test transports that do not move a
         * physical payload. Production transports override this with their
         * typed transfer ledger; callers must not reconstruct the value from
         * PerfStats.
         */
        [[nodiscard]] virtual std::uint64_t
        completedPlacementPayloadBytes() const noexcept
        {
            return 0u;
        }
    };

    /** @brief Result of attempting to publish one residency transaction. */
    enum class MoEOverlayResidencyApplyStatus
    {
        Started,
        Staging,
        Preparing,
        Publishing,
        Published,
        DynamicNoMovement,
        StaticNoMovement,
        Idle,
        Deferred,
        Busy,
        Stale,
        StageFailed,
        PreparationFailed,
        PublicationFailed,
        RetirementFailed,
    };

    struct MoEOverlayResidencyApplyResult
    {
        MoEOverlayResidencyApplyStatus status =
            MoEOverlayResidencyApplyStatus::DynamicNoMovement;
        uint64_t published_epoch = 0;
        size_t migration_count = 0;
        std::string error;

        bool ok() const noexcept
        {
            return status == MoEOverlayResidencyApplyStatus::Started ||
                   status == MoEOverlayResidencyApplyStatus::Staging ||
                   status == MoEOverlayResidencyApplyStatus::Preparing ||
                   status == MoEOverlayResidencyApplyStatus::Publishing ||
                   status == MoEOverlayResidencyApplyStatus::Published ||
                   status == MoEOverlayResidencyApplyStatus::DynamicNoMovement ||
                   status == MoEOverlayResidencyApplyStatus::StaticNoMovement ||
                   status == MoEOverlayResidencyApplyStatus::Idle ||
                   status == MoEOverlayResidencyApplyStatus::Deferred;
        }
    };

    /** @brief Process-local counters complementing exported PerfStats evidence. */
    struct MoEOverlayResidencyAuthorityStats
    {
        uint64_t checks = 0;
        uint64_t capacity_bounded_proposals = 0;
        /** Full candidate snapshots materialized while filling bounded waves. */
        uint64_t bounded_candidate_snapshot_builds = 0;
        uint64_t target_migrations_omitted = 0;
        uint64_t target_cycles_omitted = 0;
        uint64_t economy_proposals = 0;
        uint64_t payoff_rejected_cycles = 0;
        uint64_t residency_rejected_cycles = 0;
        uint64_t participant_rebalance_checks = 0;
        uint64_t participant_rebalance_proposals = 0;
        uint64_t participant_rebalance_owner_changes = 0;
        uint64_t static_no_movement_checks = 0;
        uint64_t dynamic_no_movement_checks = 0;
        uint64_t committed_waves = 0;
        uint64_t committed_migrations = 0;
        uint64_t committed_cycles = 0;
        /** Terminal waves used only to restore a reusable prepared context. */
        uint64_t prepared_context_restoration_waves = 0;
        /** Expert moves excluded from live Dynamic movement totals. */
        uint64_t prepared_context_restoration_migrations = 0;
        /** Closed cycles excluded from live Dynamic movement totals. */
        uint64_t prepared_context_restoration_cycles = 0;
        uint64_t promotions = 0;
        uint64_t demotions = 0;
        uint64_t same_priority_moves = 0;
        uint64_t cross_domain_migrations = 0;
        uint64_t cross_rank_migrations = 0;
        uint64_t cross_backend_migrations = 0;
        uint64_t busy_rejections = 0;
        uint64_t stale_rejections = 0;
        uint64_t stage_failures = 0;
        uint64_t commit_failures = 0;
        uint64_t background_waves_started = 0;
        uint64_t deferred_waves = 0;
        uint64_t published_with_old_tickets = 0;
        uint64_t old_epoch_retirements = 0;
        uint64_t aborted_waves_reaped = 0;
        uint64_t ticket_acquire_retries = 0;
        /** Request-local LLEP leases successfully pinned to a durable epoch. */
        uint64_t current_batch_llep_leases_acquired = 0;
        /** Request-local LLEP leases released after transient residency restore. */
        uint64_t current_batch_llep_leases_released = 0;
        /** LLEP lease requests rejected for topology or owner-map disagreement. */
        uint64_t current_batch_llep_lease_rejections = 0;
    };

    /** @brief Non-blocking preparation state for one immutable routing window. */
    enum class MoEOverlayHistogramWindowProgress
    {
        Waiting, ///< The host-authoritative token boundary is not full.
        Pending, ///< At least one device evidence bank is still draining.
        Ready,   ///< A complete immutable window is returned to the caller.
    };

    /** @brief Typed result from one background window-preparation poll. */
    struct MoEOverlayHistogramWindowResult
    {
        MoEOverlayHistogramWindowProgress progress =
            MoEOverlayHistogramWindowProgress::Waiting;
        std::shared_ptr<const DecodeExpertHistogramWindow> window;
    };

    /** @brief Non-blocking status of the one-time certification-demand rebase. */
    enum class MoEOverlayHistogramRebaseProgress : std::uint8_t
    {
        Pending,  ///< At least one runtime histogram source is still draining.
        Complete, ///< All pre-certification demand was frozen and discarded.
    };

    /** @brief Result of advancing economy activation at a public request edge. */
    enum class MoEOverlayDemandActivationResult : std::uint8_t
    {
        NotReady,      ///< Economy evidence or its quarantine rebase is incomplete.
        Activated,     ///< This boundary opened the first live demand generation.
        AlreadyActive, ///< A prior request boundary already performed activation.
    };

    /**
     * @brief Single authority for live ExpertOverlay tier residency.
     *
     * Tickets retain exact immutable epochs while maintenance prepares the next
     * epoch on background streams. Publication atomically switches new tickets
     * to a ready inactive bank without closing admission; old tickets continue
     * against their retained epoch. Old engines retire later on the maintenance
     * worker after that epoch's lease counter reaches zero.
     */
    class MoEOverlayResidencyAuthority final
    {
        struct PublishedEpochState;
        struct ActiveBackgroundWave;
        struct PendingRetirement;
        struct PendingAbort;
        struct EconomyState;

    public:
        /** @brief Semantic reader class retained by one epoch lease. */
        enum class TicketLeasePurpose : std::uint8_t
        {
            InferenceDispatch, ///< Ordinary decode, prefill, or verifier reader.
            InferenceGraphSequence, ///< Complete serial or speculative graph sequence.
            CurrentBatchLLEP,  ///< Request-scoped transient-residency child transaction.
        };

        struct Config
        {
            /**
             * Declarative epoch-one layout, including any explicit or
             * adversarial placement supplied by the caller. Runtime
             * maintenance must never rewrite this policy before publishing
             * the initial snapshot.
             */
            MoERoutedExpertPlacementPlan initial_plan;
            MoERoutedExpertModelMetadata model_metadata;
            /**
             * Durable maintenance intent, independent of epoch-one placement.
             * `Observe` records routing evidence without permitting movement;
             * `Dynamic` permits histogram-driven tier and participant changes.
             */
            MoERebalanceRuntimeMode maintenance_mode =
                MoERebalanceRuntimeMode::Off;
            DecodeExpertHistogram *histogram = nullptr;
            /** Ceiling for host-authority adaptive demand windows; zero fixes it. */
            std::uint64_t histogram_max_window_tokens = 0u;
            /** Multiplier applied after each host-authority RCU rotation. */
            double histogram_window_growth_factor = 1.0;
            /**
             * Immutable setup-certified phase service costs.
             *
             * The authority retains shared ownership so distributed proposal
             * generations and background waves can never observe a replaced
             * profile. A null profile preserves raw-frequency planning but is
             * not sufficient evidence for cost-optimal production migration.
             */
            std::shared_ptr<const MoERoutedTierServiceProfile>
                phase_service_profile;
            /** Complete measured directed endpoint movement costs. */
            std::shared_ptr<const MoEOverlayMigrationCostProfile>
                migration_cost_profile;
            /**
             * Optional production economy gate. Supplying it requires both
             * service and movement profiles; partial measured policy is fatal.
             */
            std::optional<MoEOverlayMigrationEconomyPolicy>
                migration_economy_policy;
            /** Same-tier participant skew policy owned by this authority. */
            MoEOverlayParticipantRebalancePolicy participant_rebalance_policy;
            /**
             * Maximum inactive arrivals at one endpoint/layer in a wave.
             * Zero preserves the complete target wave and is intended for
             * protocol tests. Production supplies its preallocated slot BOM.
             */
            std::size_t shadow_slots_per_endpoint_layer = 0;
            /**
             * Maximum closed migration cycles admitted concurrently.
             * Zero is unbounded; production uses a positive bandwidth budget.
             */
            std::size_t max_concurrent_cycles = 0;
            std::string perf_device;
        };

        /** @brief RAII lease binding one dispatch ticket to an immutable epoch. */
        class TicketLease final
        {
        public:
            TicketLease() = default;
            ~TicketLease();
            TicketLease(const TicketLease &) = delete;
            TicketLease &operator=(const TicketLease &) = delete;
            TicketLease(TicketLease &&other) noexcept;
            TicketLease &operator=(TicketLease &&other) noexcept;

            const MoEOverlayResidencySnapshot *operator->() const noexcept
            {
                return snapshot_.get();
            }
            const MoEOverlayResidencySnapshot &operator*() const
            {
                return *snapshot_;
            }
            explicit operator bool() const noexcept
            {
                return snapshot_ != nullptr;
            }

            /** @return Durable epoch pinned by this reader. */
            [[nodiscard]] uint64_t epoch() const noexcept
            {
                return snapshot_ ? snapshot_->epoch : 0;
            }

            /** @return Typed reason this reader prevents epoch retirement. */
            [[nodiscard]] TicketLeasePurpose purpose() const noexcept
            {
                return purpose_;
            }

        private:
            friend class MoEOverlayResidencyAuthority;
            TicketLease(
                MoEOverlayResidencyAuthority *authority,
                std::shared_ptr<PublishedEpochState> epoch_state,
                TicketLeasePurpose purpose);
            void release() noexcept;

            MoEOverlayResidencyAuthority *authority_ = nullptr;
            std::shared_ptr<PublishedEpochState> epoch_state_;
            std::shared_ptr<const MoEOverlayResidencySnapshot> snapshot_;
            TicketLeasePurpose purpose_ =
                TicketLeasePurpose::InferenceDispatch;
        };

        explicit MoEOverlayResidencyAuthority(Config config);
        ~MoEOverlayResidencyAuthority();

        MoEOverlayResidencyAuthority(
            const MoEOverlayResidencyAuthority &) = delete;
        MoEOverlayResidencyAuthority &operator=(
            const MoEOverlayResidencyAuthority &) = delete;

        /** @brief Bind a dispatch ticket to the race-safe current epoch. */
        std::optional<TicketLease> tryAcquireTicketSnapshot();

        /**
         * @brief Pin the current epoch for one complete sparse graph sequence.
         *
         * Per-layer dispatch tickets remain short lived, but a segmented
         * prefill or multi-graph MTP transaction must keep their common epoch
         * exact-addressable between those dispatches. The transaction
         * coordinator owns this lease until every sparse return in the
         * sequence has retired.
         *
         * @return Race-safe lease for the current publication.
         */
        std::optional<TicketLease> tryAcquireGraphSequenceSnapshot();

        /**
         * @brief Bind a device-selected ticket to one exact retained epoch.
         *
         * A captured heterogeneous producer may select placement epoch @p epoch
         * before its pinned ticket reaches the host boundary.  Publication can
         * advance in that interval, so reacquiring "current" would pair the
         * device computation with a different owner map.  This overload pins
         * the fully prepared successor, current publication, or single two-bank
         * retirement generation without taking the maintenance mutex or waiting
         * for a stream. The successor slot closes the bounded interval in which
         * one GPU has published E+1 while the public host admission floor remains
         * E. It returns no lease once retirement has closed admission for that
         * epoch.
         *
         * @param epoch Positive residency epoch copied from the device ticket.
         * @return Exact lease when the epoch remains addressable; otherwise empty.
         */
        std::optional<TicketLease> tryAcquireTicketSnapshot(uint64_t epoch);

        /**
         * @brief Pin the current durable epoch for one CPU current-batch LLEP child.
         *
         * The method derives the exact domain-local owner row from the RCU
         * snapshot and writes it into caller-owned graph storage. It performs
         * no allocation, transfer, or placement mutation. The returned lease
         * must remain alive until all transient expert copies have been removed;
         * this prevents retirement of the parent epoch while the child still
         * borrows its prepared source engines.
         *
         * @param layer_idx Routed model layer whose owner row is required.
         * @param domain_name Exact routed domain executing the LLEP transaction.
         * @param domain_participant_count Domain-local participant cardinality.
         * @param owner_participants_out Preallocated `[expert]` domain-local owners.
         * @param error Optional precise rejection diagnostic.
         * @return Typed epoch lease, or empty when topology/ownership disagrees.
         */
        [[nodiscard]] std::optional<TicketLease>
        tryAcquireCurrentBatchLLEPLease(
            int layer_idx,
            std::string_view domain_name,
            int domain_participant_count,
            std::span<uint32_t> owner_participants_out,
            std::string *error = nullptr);

        /** @brief Read the current immutable snapshot outside ticket execution. */
        std::shared_ptr<const MoEOverlayResidencySnapshot> snapshot() const;

        /** @brief Return the unmodified epoch-one placement policy. */
        [[nodiscard]] RoutedExpertResidencyPolicy residencyPolicy() const noexcept;

        /**
         * @brief Return the durable runtime-maintenance intent.
         * @return Exact configured Off, Observe, or Dynamic mode.
         */
        [[nodiscard]] MoERebalanceRuntimeMode maintenanceMode() const noexcept;

        /**
         * @brief Return whether inference records live routing histograms.
         * @return True for Observe and Dynamic maintenance.
         */
        [[nodiscard]] bool observesHistogram() const noexcept;

        /**
         * @brief Return whether a successor residency epoch may be published.
         * @return True only for Dynamic maintenance.
         */
        [[nodiscard]] bool migrationEnabled() const noexcept;

        /**
         * @brief Seal exact setup-certified service and migration economics.
         *
         * Graph construction must first publish the exact prepared expert
         * engines.  A setup profiler can then measure those engines and the
         * real directed transfer lanes before this method installs the
         * immutable results.  Installation is one-shot and is accepted only
         * before the first dynamic proposal; inference ticket acquisition may
         * already have exercised the unchanged initial residency epoch.
         *
         * @param service Complete participant-reduced tier/layer/phase costs.
         * @param migration Complete directed participant/layer movement costs.
         * @param policy Smoothing, payoff, and minimum-residency policy.
         * @throws std::invalid_argument for static policy or invalid evidence.
         * @throws std::logic_error after a proposal or prior certification.
         */
        void installEconomyCertification(
            std::shared_ptr<const MoERoutedTierServiceProfile> service,
            std::shared_ptr<const MoEOverlayMigrationCostProfile> migration,
            MoEOverlayMigrationEconomyPolicy policy);

        /**
         * @brief Query whether full service/movement economics are sealed.
         * @return True after constructor-time or delayed certification.
         */
        [[nodiscard]] bool hasEconomyCertification() const noexcept;

        /**
         * @brief Query whether post-certificate route demand is admitted.
         * @return True only after a public request boundary opened the live bank.
         */
        [[nodiscard]] bool optimizationDemandActive() const noexcept;

        /**
         * @brief Activate sealed economics at the next public request boundary.
         *
         * Installation and demand admission are intentionally separate. The
         * request that completed service certification remains quarantined in
         * its entirety; a later prefill admission opens CPU and exact GPU
         * writer states before that new request executes any model graph.
         *
         * @return Typed no-op, activation, or already-active result.
         * @throws std::runtime_error when a device writer rejects activation.
         */
        [[nodiscard]] MoEOverlayDemandActivationResult
        activateOptimizationDemandAtRequestBoundary();

        /**
         * @brief Query whether a dynamic proposal has accumulated a full window.
         * @return True only when a dynamic policy's active histogram bank is full.
         *
         * The query is allocation-free and never synchronizes device histogram
         * sources. Device evidence is merged later by `proposeFromHistogram()`
         * on the maintenance worker, after the host token boundary authorizes a
         * rotation.
         */
        [[nodiscard]] bool maintenanceWindowReady() const noexcept;

        /**
         * @brief Observe the active routing-demand bank without advancing it.
         * @return Exact generation, routed-row occupancy, and decision capacity,
         *         or an invalid zero-capacity value when no histogram exists.
         *
         * This is the sole host-policy source for public demand headroom. It
         * neither synchronizes runtime histogram drains nor reserves capacity.
         */
        [[nodiscard]] MoEOptimizationDemandWindow
        optimizationDemandWindow() const noexcept;

        /**
         * @brief Start or poll exact device evidence, then rotate the host bank.
         *
         * The first call is authorized only by a full host token window. Once
         * started, later polls continue even after request-local activity
         * changes. Pending never waits for a stream or device; failures throw
         * with the owning drain's diagnostic.
         */
        [[nodiscard]] MoEOverlayHistogramWindowResult
        progressHistogramWindow();

        /**
         * @brief Drain and discard demand collected before economy activation.
         *
         * Service certification deliberately exercises broad startup traffic.
         * Those routes measure compute cost but are not a production placement
         * window. The certification owner polls this method after composing the
         * immutable profiles and before installing them. It uses the same exact
         * runtime-histogram events and generation rotation as a normal proposal,
         * but never exposes the frozen calibration window to policy.
         *
         * @return Pending without waiting, or Complete after the new empty
         *         post-certification generation is authoritative.
         * @throws std::logic_error for the wrong authority/lifecycle.
         * @throws std::runtime_error when a runtime drain cannot complete.
         */
        [[nodiscard]] MoEOverlayHistogramRebaseProgress
        progressEconomyEvidenceRebase();

        /** @brief Build the deterministic hottest-first candidate from current evidence. */
        MoEOverlayResidencyTransaction proposeFromHistogram();

        /**
         * @brief Return whether the exact model-preparation placement is live.
         *
         * The comparison includes logical tier placement and exact participant
         * ownership. It deliberately says nothing about which recyclable
         * physical slot backs an expert; the model registry is rebound from
         * the published participant banks after this condition becomes true.
         *
         * @return True when the current publication matches epoch one's
         *         prepared placement, independent of its current epoch number.
         */
        [[nodiscard]] bool initialPreparedPlacementPublished() const;

        /**
         * @brief Build one bounded wave toward the prepared model placement.
         *
         * This terminal lifecycle operation bypasses histogram payoff policy:
         * context reuse requires exact prepared-weight identity, so restoration
         * is mandatory rather than an optional performance decision. The wave
         * remains capacity preserving and honors the same shadow-slot and
         * concurrent-cycle BOM as live maintenance.
         *
         * @return A valid restoration transaction. It is empty when the exact
         *         prepared placement is already published.
         * @throws std::logic_error unless host-owned Dynamic publication is
         *         available and no malformed lifecycle state is observed.
         */
        MoEOverlayResidencyTransaction
        proposeInitialPreparedPlacementRestoration();

        /**
         * @brief Freeze the authoritative local routing window for distribution.
         * @return Immutable, validated window after runtime sources are merged.
         * @throws std::logic_error Unless Dynamic maintenance is enabled.
         * @throws std::runtime_error When device histogram synchronization fails.
         *
         * A distributed maintenance coordinator calls this once, broadcasts
         * the returned value, and then asks every rank to derive its proposal
         * with @ref proposeFromFrozenHistogramWindow. Non-coordinator ranks must
         * never substitute their process-local partial routing evidence.
         */
        [[nodiscard]] std::shared_ptr<const DecodeExpertHistogramWindow>
        freezeAndRotateHistogramWindow();

        /**
         * @brief Derive a candidate from one coordinator-published frozen window.
         * @param window Valid immutable model-wide routing evidence.
         * @return Deterministic transaction based on this rank's current epoch.
         * @throws std::invalid_argument For invalid or mismatched geometry.
         * @throws std::logic_error Unless Dynamic maintenance and the current
         *         epoch are valid.
         *
         * The method performs no histogram synchronization or rotation. Every
         * rank given identical topology, epoch, and window therefore constructs
         * the same pointer-independent transaction fingerprint.
         */
        MoEOverlayResidencyTransaction proposeFromFrozenHistogramWindow(
            std::shared_ptr<const DecodeExpertHistogramWindow> window);

        /**
         * @brief Export the exact executable part of a root-owned proposal.
         * @param transaction Valid live placement transaction from this authority.
         * @return Dense pointer-free candidate suitable for authenticated publication.
         * @throws std::invalid_argument For a malformed or non-live transaction.
         * @throws std::logic_error When its migration set does not exactly cover
         *         the immutable snapshot delta.
         */
        [[nodiscard]] MoEOverlayAuthoritativeResidencyPlan
        exportAuthoritativeResidencyPlan(
            const MoEOverlayResidencyTransaction &transaction) const;

        /**
         * @brief Materialize one root-authored proposal without rerunning policy.
         * @param plan Authenticated dense candidate received from the root.
         * @return Exact rank-local physical transaction for distributed staging.
         * @throws std::invalid_argument For malformed geometry or movement data.
         * @throws std::logic_error For stale epoch, topology disagreement, or a
         *         candidate which is not capacity preserving.
         *
         * Followers never smooth histograms, evaluate economy, select tiers,
         * or rebalance participants here.  They resolve the root's declarative
         * participant IDs against their own immutable topology and validate the
         * resulting execution identity before entering consensus.
         */
        [[nodiscard]] MoEOverlayResidencyTransaction
        adoptAuthoritativeResidencyPlan(
            const MoEOverlayAuthoritativeResidencyPlan &plan);

        /**
         * @brief Enqueue a proposal on the background transport domain.
         * @param transaction Capacity-preserving candidate based on current epoch.
         * @param transport Factory that owns exact streams, events, and slots.
         * @return Started, deferred, no-movement, busy, stale, or failed result.
         */
        MoEOverlayResidencyApplyResult beginApply(
            const MoEOverlayResidencyTransaction &transaction,
            IMoEOverlayResidencyTransport &transport);

        /**
         * @brief Advance background state and reap old epochs without waiting.
         * @return Current progress or a terminal publication/failure result.
         *
         * This method belongs on a maintenance worker. Every wave poll is an
         * event query; calling it must never synchronize an inference stream.
         */
        MoEOverlayResidencyApplyResult advanceBackground();

        /** @brief Return a race-safe copy of protocol counters. */
        MoEOverlayResidencyAuthorityStats stats() const noexcept;

        /**
         * @brief Snapshot exact live-placement edges committed by this authority.
         *
         * Prepared-context restoration edges are intentionally excluded: they
         * belong to teardown, not to the optimization epoch whose numerical
         * behavior a production request must certify.
         */
        [[nodiscard]] MoEOptimizationMovementLedger movementLedger() const;

        uint64_t activeTicketCount() const noexcept
        {
            return active_ticket_count_.load(std::memory_order_acquire);
        }

        /** @return Number of published old epochs awaiting lease-safe retirement. */
        size_t pendingRetirementCount() const noexcept;

        /** @return Number of unpublished waves awaiting abort-event quiescence. */
        size_t pendingAbortCount() const noexcept;

        /** @return Whether one unpublished candidate wave is currently active. */
        bool hasActiveBackgroundWave() const noexcept;

    private:
        /** Sole typed owner of the shared runtime-histogram drain lane. */
        enum class HistogramDrainState : std::uint8_t
        {
            Idle,
            ProposalWindow,
            CertificationRebase,
        };

        /** Typed host-economy activation state; no certificate may skip rebase. */
        enum class EconomyActivationState : std::uint8_t
        {
            CollectingEvidence,
            RebasingRoutingEvidence,
            ReadyForCertification,
            CertifiedAwaitingRequestBoundary,
            Active,
        };

        /** @brief Common non-blocking drain/merge/rotate implementation. */
        [[nodiscard]] MoEOverlayHistogramWindowResult
        progressHistogramDrain(HistogramDrainState requested_state);

        /**
         * @brief Advance the next host demand window after one proposal rotation.
         *
         * Certification rebase is excluded: it removes synthetic startup
         * demand before the first production window and therefore must not
         * consume an adaptive step. Every ordinary proposal rotation advances
         * exactly once, including an economically rejected proposal, matching
         * the established Dynamic controller semantics.
         */
        void growHistogramWindowAfterProposalRotation();

        /**
         * @brief Reject host planning/publication for a device-owned dynamic authority.
         *
         * Homogeneous GPU topology retains this object as the immutable setup
         * catalog and ticket source used while graphs are built. Its live
         * epoch is published by the captured device arena, so entering any
         * host proposal or apply method would create a second durable writer.
         * Unresolved test plans retain host behavior; production plans are
         * frozen before authority construction.
         *
         * CPU participant memory is host memory, so a homogeneous CPU tier
         * legitimately executes this implementation. Only CUDA/ROCm topology
         * is rejected here.
         *
         * @param operation Stable diagnostic name of the attempted host action.
         * @throws std::logic_error when the frozen executor is accelerator-resident.
         */
        void requireHostDynamicPublicationAuthority(
            const char *operation) const;

        /**
         * @brief Validate and materialize one immutable residency epoch.
         * @param epoch Positive publication generation.
         * @param plan Complete logical tier placement for the generation.
         * @param metadata Exact routed model geometry.
         * @param previous_owners Prior epoch ownership for a stable transition,
         *        or null only while constructing the initial epoch.
         * @param explicit_ownership Exact participant owners selected by the
         *        candidate planner, or null to derive cold-start/stable owners.
         * @return Fully validated snapshot with physical and layered ownership.
         * @throws std::invalid_argument for invalid plan, geometry, or topology.
         */
        static std::shared_ptr<const MoEOverlayResidencySnapshot> buildSnapshot(
            uint64_t epoch,
            MoERoutedExpertPlacementPlan plan,
            const MoERoutedExpertModelMetadata &metadata,
            const MoEExpertOwnerMap *previous_owners,
            const MoELayeredExpertOwnership *explicit_ownership = nullptr);
        /** @brief Validate profiles and build allocation-free lookup state. */
        static std::unique_ptr<EconomyState> buildEconomyState(
            const Config &config,
            const MoEOverlayResidencySnapshot &initial_snapshot);
        /**
         * @brief Materialize exact physical edges and policy-authored intent.
         * @param previous Currently published immutable placement.
         * @param candidate Proposed next immutable placement.
         * @param histogram_window Optional authenticated demand view for edge
         *        evidence. The borrowed window must outlive this call.
         * @param estimated_weight_bytes Complete packed expert payload size.
         * @param participant_changes Exact participant-planner destinations;
         *        entries absent from a bounded candidate are ignored.
         * @return Canonically ordered physical movement edges.
         */
        static std::vector<MoEOverlayTierMigration> buildMigrations(
            const MoEOverlayResidencySnapshot &previous,
            const MoEOverlayResidencySnapshot &candidate,
            const ValidatedDecodeExpertHistogramWindowView *histogram_window,
            size_t estimated_weight_bytes,
            std::span<const MoELayeredExpertOwnershipChange>
                participant_changes = {});
        /**
         * @brief Decompose edges and normalize each cycle's objective axis.
         * @param migrations Mutable canonical edges; every edge in a closed
         *        cycle receives that cycle's aggregate typed objective.
         * @return Complete deterministic capacity-preserving cycle cover.
         */
        static std::vector<MoEOverlayTierMigrationCycle> buildMigrationCycles(
            std::vector<MoEOverlayTierMigration> &migrations);
        static std::vector<MoEOverlayTierShadowRequirement>
        buildShadowRequirements(
            const std::vector<MoEOverlayTierMigration> &migrations);
        static void requireCapacityPreserving(
            const MoEOverlayResidencySnapshot &previous,
            const MoEOverlayResidencySnapshot &candidate);
        /** @brief Acquire the current publication for one typed reader class. */
        std::optional<TicketLease> tryAcquireCurrentSnapshot(
            TicketLeasePurpose purpose);
        void releaseTicket(
            const std::shared_ptr<PublishedEpochState> &epoch_state,
            TicketLeasePurpose purpose) noexcept;
        /**
         * @brief Progress lease-drain fences and reclaim globally safe epochs.
         * @param error Receives a precise distributed fence failure.
         * @return False only when a retirement protocol irrecoverably failed.
         */
        bool reapReadyRetirementsLocked(std::string *error = nullptr);
        void reapReadyAbortsLocked() noexcept;
        void abortAndRetainWaveLocked(
            std::unique_ptr<IMoEOverlayResidencyWave> work,
            uint64_t epoch) noexcept;
        void recordAbortedWaveReapedLocked(uint64_t epoch) noexcept;
        MoEOverlayResidencyApplyResult failActiveWaveLocked(
            MoEOverlayResidencyApplyStatus status,
            std::string error) noexcept;
        void recordNoMovement(bool static_policy);
        void recordCommitted(
            const MoEOverlayResidencyTransaction &transaction);

        Config config_;
        MoERoutedExpertPlacementPlan planning_template_;
        /** Immutable epoch-one logical/physical-owner identity for reuse sealing. */
        std::shared_ptr<const MoEOverlayResidencySnapshot> initial_snapshot_;
        std::atomic<std::shared_ptr<PublishedEpochState>> published_epoch_;
        /**
         * Fully prepared successor accepted by exact device-selected tickets.
         *
         * This slot becomes visible before any GPU selector may flip and is
         * removed only after the same state object becomes @ref published_epoch_.
         * It is not a second policy authority: ordinary host admission never
         * reads it, while exact heterogeneous tickets may name it during the
         * bounded selector fan-out interval.
         */
        std::atomic<std::shared_ptr<PublishedEpochState>> candidate_epoch_;
        /**
         * One lock-free lookup for the old half of the two-bank RCU pair.
         * The maintenance protocol admits no successor wave until this pointer
         * is cleared, so an array or mutex-protected epoch map is unnecessary.
         */
        std::atomic<std::shared_ptr<PublishedEpochState>> retiring_epoch_;
        std::atomic<uint64_t> active_ticket_count_{0};
        mutable std::mutex maintenance_mutex_;
        std::unique_ptr<ActiveBackgroundWave> active_wave_;
        std::vector<std::unique_ptr<PendingRetirement>> pending_retirements_;
        std::vector<std::unique_ptr<PendingAbort>> pending_aborts_;
        mutable std::mutex economy_mutex_;
        std::unique_ptr<EconomyState> economy_state_;
        std::atomic<EconomyActivationState> economy_activation_state_{
            EconomyActivationState::CollectingEvidence};
        std::atomic<HistogramDrainState> histogram_drain_state_{
            HistogramDrainState::Idle};

        std::atomic<uint64_t> checks_{0};
        std::atomic<uint64_t> capacity_bounded_proposals_{0};
        std::atomic<uint64_t> bounded_candidate_snapshot_builds_{0};
        std::atomic<uint64_t> target_migrations_omitted_{0};
        std::atomic<uint64_t> target_cycles_omitted_{0};
        std::atomic<uint64_t> economy_proposals_{0};
        std::atomic<uint64_t> payoff_rejected_cycles_{0};
        std::atomic<uint64_t> residency_rejected_cycles_{0};
        std::atomic<uint64_t> participant_rebalance_checks_{0};
        std::atomic<uint64_t> participant_rebalance_proposals_{0};
        std::atomic<uint64_t> participant_rebalance_owner_changes_{0};
        std::atomic<uint64_t> static_no_movement_checks_{0};
        std::atomic<uint64_t> dynamic_no_movement_checks_{0};
        std::atomic<uint64_t> committed_waves_{0};
        std::atomic<uint64_t> committed_migrations_{0};
        std::atomic<uint64_t> committed_cycles_{0};
        /** Terminal context-restoration work, excluded from live movement totals. */
        std::atomic<uint64_t> prepared_context_restoration_waves_{0};
        std::atomic<uint64_t> prepared_context_restoration_migrations_{0};
        std::atomic<uint64_t> prepared_context_restoration_cycles_{0};
        std::atomic<uint64_t> promotions_{0};
        std::atomic<uint64_t> demotions_{0};
        std::atomic<uint64_t> same_priority_moves_{0};
        std::atomic<uint64_t> cross_domain_migrations_{0};
        std::atomic<uint64_t> cross_rank_migrations_{0};
        std::atomic<uint64_t> cross_backend_migrations_{0};
        std::atomic<uint64_t> busy_rejections_{0};
        std::atomic<uint64_t> stale_rejections_{0};
        std::atomic<uint64_t> stage_failures_{0};
        std::atomic<uint64_t> commit_failures_{0};
        std::atomic<uint64_t> background_waves_started_{0};
        std::atomic<uint64_t> deferred_waves_{0};
        std::atomic<uint64_t> published_with_old_tickets_{0};
        std::atomic<uint64_t> old_epoch_retirements_{0};
        std::atomic<uint64_t> aborted_waves_reaped_{0};
        /** Exact committed edge identities, independent of optional telemetry. */
        mutable std::mutex movement_ledger_mutex_;
        std::vector<MoEOptimizationMovementEdge> movement_ledger_;
        /** Coordinator-owned economy proofs; followers intentionally omit them. */
        std::vector<MoEOptimizationMovementEconomy> movement_economy_;
        /** Coordinator-owned cycle-admission proofs; followers omit them. */
        std::vector<MoEOptimizationHostMovementAdmission>
            movement_host_admissions_;
        std::atomic<uint64_t> ticket_acquire_retries_{0};
        std::atomic<uint64_t> current_batch_llep_leases_acquired_{0};
        std::atomic<uint64_t> current_batch_llep_leases_released_{0};
        std::atomic<uint64_t> current_batch_llep_lease_rejections_{0};
    };

} // namespace llaminar2
