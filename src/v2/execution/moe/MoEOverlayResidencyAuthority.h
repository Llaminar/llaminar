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
     * @brief One closed, capacity-preserving migration cycle within a layer.
     *
     * Indices refer to the enclosing transaction's immutable migration vector.
     * The destination tier of each indexed edge equals the source tier of the
     * next edge, including the last-to-first wrap. A one-edge cycle represents
     * relocation between participants of the same tier. Cycles are the smallest
     * units that may be selected for a bounded-shadow publication wave without
     * changing any physical participant's logical expert capacity.
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
                   (imbalance_threshold_per_mille >= 1000u &&
                   maximum_swaps_per_layer > 0u &&
                   maximum_plan_entries_per_wave >= 2u);
        }
    };

    /**
     * @brief Pointer-free economic proof attached to one candidate wave.
     *
     * Distributed fingerprints include this record so ranks using different
     * measured profiles or policy cannot accidentally vote for one epoch.
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
        std::shared_ptr<const DecodeExpertHistogramWindow> histogram_window;
        std::shared_ptr<const MoEOverlayResidencySnapshot> previous;
        std::shared_ptr<const MoEOverlayResidencySnapshot> candidate;
        std::vector<MoEOverlayTierMigration> migrations;
        std::vector<MoEOverlayTierMigrationCycle> migration_cycles;
        std::vector<MoEOverlayTierShadowRequirement> shadow_requirements;
        MoEOverlayMigrationEconomyEvidence economy;

        bool empty() const noexcept { return migrations.empty(); }

        /**
         * @brief Validate epoch identity and the complete cycle/reservation plan.
         * @return Whether every movement appears in exactly one closed cycle and
         *         every destination is covered by an exact shadow requirement.
         */
        [[nodiscard]] bool valid() const noexcept;
    };

    /** @brief Non-blocking readiness returned by a background migration wave. */
    enum class MoEOverlayResidencyWaveProgress
    {
        Pending, ///< Device/network work remains in flight; do not publish.
        Ready,   ///< The phase's exact completion event has become visible.
        Deferred, ///< Global staging backpressure requires whole-wave retry.
        Failed,  ///< The phase failed and the unpublished candidate must abort.
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
     * wait for a collective, or run work on an inference thread. Commit builds
     * and authenticates an inactive runtime bank. Only the residency authority
     * publishes the public epoch after commit readiness is observed.
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
         * @brief Enqueue publication of a complete inactive destination bank.
         * @param error Receives a precise enqueue/validation failure.
         * @return Whether commit work was enqueued without blocking.
         */
        virtual bool beginCommit(std::string *error) noexcept = 0;

        /**
         * @brief Query inactive-bank commit readiness without waiting.
         * @param error Receives a precise diagnostic when `Failed` is returned.
         * @return Current commit progress.
         */
        virtual MoEOverlayResidencyWaveProgress pollCommit(
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
         * @brief Observe the authority's successful candidate-epoch CAS.
         *
         * This control-plane callback runs immediately after publication. It
         * must not launch work, allocate, synchronize, or fail; physical bank
         * visibility was already proved by `pollCommit()`. Device-free waves
         * need no notification, so the default is intentionally empty.
         */
        virtual void markPublished() noexcept {}

        /**
         * @brief Poll the cross-participant lease-drain fence for the old epoch.
         *
         * The authority calls this only after every process-local ticket for
         * the previous epoch has drained.  A single-process wave is therefore
         * immediately ready.  Distributed waves override the method and vote
         * on a private non-blocking consensus lane so no rank can reclaim an
         * old participant bank while another rank still has an in-flight
         * sparse dispatch stamped with that epoch.
         *
         * @param error Receives a precise retirement-fence failure.
         * @return Pending until all ranks report local lease drainage, Ready
         *         when physical retirement is safe, or Failed on a protocol
         *         error.
         */
        virtual MoEOverlayResidencyWaveProgress pollRetirementFence(
            std::string *error) noexcept
        {
            if (error)
                error->clear();
            return MoEOverlayResidencyWaveProgress::Ready;
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
    };

    /** @brief Result of attempting to publish one residency transaction. */
    enum class MoEOverlayResidencyApplyStatus
    {
        Started,
        Staging,
        Committing,
        Committed,
        DynamicNoMovement,
        StaticNoMovement,
        Idle,
        Deferred,
        Busy,
        Stale,
        StageFailed,
        CommitFailed,
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
                   status == MoEOverlayResidencyApplyStatus::Committing ||
                   status == MoEOverlayResidencyApplyStatus::Committed ||
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
         * @brief Bind a device-selected ticket to one exact retained epoch.
         *
         * A captured heterogeneous producer may select placement epoch @p epoch
         * before its pinned ticket reaches the host boundary.  Publication can
         * advance in that interval, so reacquiring "current" would pair the
         * device computation with a different owner map.  This overload pins
         * either the current publication or the single two-bank retirement
         * generation without taking the maintenance mutex or waiting for a
         * stream.  It returns no lease once retirement has closed admission for
         * that epoch.
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
         * @brief Start or poll exact device evidence, then rotate the host bank.
         *
         * The first call is authorized only by a full host token window. Once
         * started, later polls continue even after request-local activity
         * changes. Pending never waits for a stream or device; failures throw
         * with the owning drain's diagnostic.
         */
        [[nodiscard]] MoEOverlayHistogramWindowResult
        progressHistogramWindow();

        /** @brief Build the deterministic hottest-first candidate from current evidence. */
        MoEOverlayResidencyTransaction proposeFromHistogram();

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
        static std::vector<MoEOverlayTierMigration> buildMigrations(
            const MoEOverlayResidencySnapshot &previous,
            const MoEOverlayResidencySnapshot &candidate,
            const DecodeExpertHistogramWindow *histogram_window,
            size_t estimated_weight_bytes);
        static std::vector<MoEOverlayTierMigrationCycle> buildMigrationCycles(
            const std::vector<MoEOverlayTierMigration> &migrations);
        static std::vector<MoEOverlayTierShadowRequirement>
        buildShadowRequirements(
            const std::vector<MoEOverlayTierMigration> &migrations);
        static void requireCapacityPreserving(
            const MoEOverlayResidencySnapshot &previous,
            const MoEOverlayResidencySnapshot &candidate);
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
        std::atomic<std::shared_ptr<PublishedEpochState>> published_epoch_;
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
        std::atomic<bool> economy_certified_{false};
        std::atomic<bool> histogram_drain_active_{false};

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
        std::atomic<uint64_t> ticket_acquire_retries_{0};
        std::atomic<uint64_t> current_batch_llep_leases_acquired_{0};
        std::atomic<uint64_t> current_batch_llep_leases_released_{0};
        std::atomic<uint64_t> current_batch_llep_lease_rejections_{0};
    };

} // namespace llaminar2
