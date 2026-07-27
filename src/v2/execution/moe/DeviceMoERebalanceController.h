/**
 * @file DeviceMoERebalanceController.h
 * @brief Graph-capturable device-side MoE rebalance publish/apply ABI.
 */

#pragma once

#include "DeviceMoERebalancePolicyShared.h"
#include "MoERuntimeTable.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace llaminar2
{
    inline constexpr uint32_t kDeviceMoERebalanceMagic = 0x4d4f4552u; // "MOER"
    inline constexpr uint32_t kDeviceMoERebalanceVersion = 3;
    inline constexpr uint32_t kDeviceMoERebalanceAssignmentStaticOwner = 0;
    inline constexpr uint32_t kDeviceMoERebalanceAssignmentLeastLoadedResident = 1;

    enum class DeviceMoERebalanceFlags : uint32_t
    {
        None = 0,
        HotReplicaCache = 1u << 0,
        ResetHistogramsAfterApply = 1u << 1,
        PlanMissingArrivals = 1u << 2,
        DeferRuntimeApply = 1u << 3,
        CollectLoadStats = 1u << 4,
    };

    enum class DeviceMoERebalancePlanOp : uint32_t
    {
        None = 0,
        ExpertPayloadArrival = 1,
        ResidentExpertAssignment = 2,
        OwnershipTransfer = 3,
    };

    enum class DeviceMoERebalancePipelinePhase : uint32_t
    {
        /**
         * Pack local runtime histograms plus resident expert descriptors and
         * gather them across the homogeneous LocalTP domain.
         */
        CollectState = 1,

        /**
         * Run the device policy kernel and write fixed-size command-buffer
         * entries for this participant.
         */
        PlanAssignments = 2,

        /**
         * Execute arrival-copy commands into preallocated transfer slots on an
         * explicit transfer stream.
         */
        StageArrivals = 3,

        /**
         * Publish completed transfer slots into the live mirrored runtime table
         * immediately before routed expert compute.
         */
        ApplyLayer = 4,
    };

    enum class DeviceMoERebalanceWaveLifecycle : uint32_t
    {
        Idle = 0,
        Planning = 1,
        TransferInFlight = 2,
        ReadyToApply = 3,
        Applying = 4,
        Applied = 5,
        Error = 6,
    };

    enum class DeviceMoERebalanceTransferMode : uint32_t
    {
        /**
         * Graph-captured steady-state mode for homogeneous GPU domains.
         *
         * The async maintenance graph gathers wave-scoped histogram state and
         * publishes assignments for experts already resident on this
         * participant. The steady decode graph only polls/apply ready waves; it
         * never moves packed expert payload bytes, never allocates transfer
         * slots, and must not set PlanMissingArrivals.
         */
        ResidentOnly = 0,

        /**
         * Compact same-backend GPU arrival mode.
         *
         * The controller may plan ExpertPayloadArrival commands for experts that
         * are not yet resident on this participant. The graph gathers compact
         * command metadata, each source packs only planned non-empty arrivals
         * into a fixed-capacity staging lane, and NCCL/RCCL grouped collectives
         * move that lane on an explicit transfer stream. Destination kernels
         * unpack only real plan entries into preallocated transfer slots.
         *
         * The captured collective has a fixed maximum slot capacity, but it
         * must never gather a full expert directory and device kernels must not
         * dereference peer device pointers directly.
         */
        CompactTransferSlots = 1,

        /**
         * Obsolete fixed-arena payload sideband mode.
         *
         * The mode is still represented so older stage/unit scaffolding can
         * account for fixed payload capacity, but graph-side rebalance must not
         * select it. Perfstats showed that the captured collective arena mostly
         * moves empty expert slots. A future payload mode must use a compact
         * non-empty arrival list instead.
         */
        CollectiveSidebandPayload = 2,

        /**
         * Obsolete legacy transfer mode for homogeneous GPU domains.
         *
         * This path launches rebalance-specific raw allgather calls from
         * MoEDeviceRebalanceStage and moves fixed payload arenas. Graph-side
         * rebalance must reject this mode for the same empty-slot reason as
         * CollectiveSidebandPayload.
         */
        LegacyCollectiveAllGather = 3,
    };

    inline constexpr uint32_t kDeviceMoEInvalidSlot = 0xffffffffu;

    struct DeviceMoERebalanceCommandBufferHeader
    {
        uint32_t magic = kDeviceMoERebalanceMagic;
        uint32_t version = kDeviceMoERebalanceVersion;
        uint32_t epoch = 0;
        uint32_t phase = static_cast<uint32_t>(DeviceMoERebalancePipelinePhase::PlanAssignments);
        uint32_t command_count = 0;
        uint32_t command_capacity = 0;
        uint32_t participant_id = 0;
        uint32_t participant_count = 1;
    };

    struct DeviceMoERebalanceWaveState
    {
        uint32_t magic = kDeviceMoERebalanceMagic;
        uint32_t version = kDeviceMoERebalanceVersion;
        uint32_t epoch = 0;
        uint32_t next_start_layer = 0;
        uint32_t planned_start_layer = 0;
        uint32_t planned_layer_count = 0;
        uint32_t command_capacity = 0;
        uint32_t participant_id = 0;
        uint32_t participant_count = 1;
        uint32_t requested_payload_slots = 0;
        uint32_t payload_bucket_slots = 0;
        uint32_t payload_bucket_index = 0;
        uint32_t payload_bucket_overflow = 0;
        uint32_t reserved[3] = {};
    };

    /**
     * Device-resident lifecycle for one rebalance wave.
     *
     * This is the handoff contract between a separately captured maintenance
     * graph and the decode graph.  The maintenance graph owns CollectState,
     * PlanAssignments, and StageArrivals; after transfer kernels publish
     * copy_complete_epoch with a device fence, decode-side ApplyLayer kernels
     * may poll this state and apply only matching ready epochs.  No host publish
     * callback, graph recapture, or decode-graph rewarm is part of this path.
     */
    struct DeviceMoERebalanceWaveProgress
    {
        uint32_t magic = kDeviceMoERebalanceMagic;
        uint32_t version = kDeviceMoERebalanceVersion;
        uint32_t epoch = 0;
        uint32_t state = static_cast<uint32_t>(DeviceMoERebalanceWaveLifecycle::Idle);
        uint32_t planned_start_layer = 0;
        uint32_t planned_layer_count = 0;
        uint32_t command_count = 0;
        uint32_t copied_arrivals = 0;
        uint32_t applied_arrivals = 0;
        uint32_t applied_layer_count = 0;
        uint32_t error_code = 0;
        uint32_t requested_payload_slots = 0;
        uint32_t payload_bucket_slots = 0;
        uint32_t payload_bucket_index = 0;
        uint32_t payload_bucket_overflow = 0;
        /**
         * Participant whose publication first poisoned this wave.
         *
         * This field is meaningful only when state is Error. Keeping it on the
         * wave makes the failing edge visible after later graph replays without
         * requiring a host-side diagnostic mirror in the execution path.
         */
        uint32_t error_participant = kDeviceMoEInvalidSlot;
    };

    /**
     * Fixed persistent controller state shared by captured maintenance and
     * decode graphs.  V1 uses two waves so the next rebalance transfer can run
     * while decode applies the previous ready wave.  Larger domains keep the
     * same ABI; participant_count is variable and never assumes two cards.
     */
    struct DeviceMoERebalanceGraphControllerState
    {
        uint32_t magic = kDeviceMoERebalanceMagic;
        uint32_t version = kDeviceMoERebalanceVersion;
        uint32_t participant_id = 0;
        uint32_t participant_count = 1;
        uint32_t next_epoch = 1;
        uint32_t active_wave = 0;
        uint32_t wave_count = 2;
        uint32_t maintenance_launches = 0;
        uint32_t decode_apply_polls = 0;
        uint32_t decode_apply_hits = 0;
        /**
         * First fatal controller error. A non-zero value permanently poisons
         * this request-owned controller until explicit request teardown.
         */
        uint32_t last_error_code = 0;
        /// Command-buffer wave that first published last_error_code.
        uint32_t last_error_wave_index = kDeviceMoEInvalidSlot;
        /// Command epoch associated with the first fatal publication.
        uint32_t last_error_epoch = 0;
        /// Payload arrivals required from the failing participant.
        uint32_t last_error_expected_arrivals = 0;
        /// Payload arrivals actually reported by the failing participant.
        uint32_t last_error_copied_arrivals = 0;
        /// Participant-local copy status code observed at the failure.
        uint32_t last_error_copy_status_code = 0;
        DeviceMoERebalanceWaveProgress waves[2];
    };

    /**
     * Fixed-size graph-captured rebalance command.
     *
     * This is the command buffer consumed by StageArrivals and ApplyLayer. The
     * host may allocate and inspect the buffer for diagnostics, but homogeneous
     * GPU rebalance replay must produce and consume these entries entirely from
     * captured collectives, kernels, explicit streams, and events.
     */
    struct DeviceMoERebalancePlanEntry
    {
        uint32_t op = static_cast<uint32_t>(DeviceMoERebalancePlanOp::None);
        uint32_t layer = 0;
        uint32_t expert = 0;
        uint32_t source_participant = 0;
        uint32_t destination_participant = 0;
        uint32_t source_resident_mask = 0;
        uint32_t flags = 0;
        uint32_t destination_slot = kDeviceMoEInvalidSlot;
        uint32_t payload_slot = kDeviceMoEInvalidSlot;

        /**
         * Logical occupant observed by the destination participant when it
         * leased destination_slot for this command. These fields turn a bare
         * array index into an explicit compare-and-replace transaction. The
         * unpack kernel must observe the same occupant and generation before
         * mutating the shared transfer directory.
         */
        uint32_t destination_previous_layer = kDeviceMoEInvalidSlot;
        uint32_t destination_previous_expert = kDeviceMoEInvalidSlot;
        uint32_t destination_generation = 0;
    };

    enum class DeviceMoERebalanceDirectoryFlags : uint32_t
    {
        None = 0,
        Valid = 1u << 0,
        Resident = 1u << 1,
        LocalCompute = 1u << 2,
        TransferSlot = 1u << 3,
        CopyComplete = 1u << 4,
    };

    struct DeviceMoEExpertDirectoryEntry
    {
        DeviceMoEExpertDescriptor descriptor;
        uint32_t layer = 0;
        uint32_t expert = 0;
        uint32_t participant = 0;
        uint32_t resident_mask = 0;
        uint32_t epoch = 0;
        uint32_t flags = 0;
        uint32_t slot_index = kDeviceMoEInvalidSlot;
        uint32_t generation = 0;
    };

    enum class DeviceMoERebalanceApplyStatusCode : uint32_t
    {
        Ok = 0,
        InvalidConfig = 1,
        InvalidRuntime = 2,
        MissingPlan = 3,
        MissingDirectory = 4,
        InProgress = 5,
    };

    struct DeviceMoERebalanceApplyStatus
    {
        uint32_t magic = kDeviceMoERebalanceMagic;
        uint32_t version = kDeviceMoERebalanceVersion;
        uint32_t status_code = static_cast<uint32_t>(DeviceMoERebalanceApplyStatusCode::Ok);
        uint32_t plan_entries_seen = 0;
        uint32_t applied_arrivals = 0;
        uint32_t skipped_wrong_destination = 0;
        uint32_t invalid_plan_entries = 0;
        uint32_t missing_source_descriptors = 0;
        uint32_t missing_destination_slots = 0;
        uint32_t descriptor_mismatches = 0;
        uint32_t changed_layers = 0;
        uint32_t copied_arrivals = 0;
        uint32_t copy_incomplete = 0;
        uint32_t post_apply_multi_resident_experts = 0;
        uint32_t required_local_arrivals = 0;
        uint32_t ready_local_arrivals = 0;
    };

    constexpr DeviceMoERebalanceFlags operator|(DeviceMoERebalanceFlags lhs,
                                                DeviceMoERebalanceFlags rhs) noexcept
    {
        return static_cast<DeviceMoERebalanceFlags>(
            static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
    }

    constexpr bool hasDeviceMoERebalanceFlag(uint32_t flags,
                                             DeviceMoERebalanceFlags flag) noexcept
    {
        return (flags & static_cast<uint32_t>(flag)) != 0u;
    }

    enum class DeviceMoERebalanceStatusCode : uint32_t
    {
        Ok = 0,
        WindowNotReady = 1,
        InvalidConfig = 2,
        InvalidRuntime = 3,
        MissingHistogram = 4,
        MissingTransferCompletion = 5,
    };

    struct DeviceMoERebalanceConfig
    {
        uint32_t magic = kDeviceMoERebalanceMagic;
        uint32_t version = kDeviceMoERebalanceVersion;
        uint32_t num_layers = 0;
        uint32_t num_experts = 0;
        uint32_t top_k = 0;
        uint32_t participant_id = 0;
        uint32_t participant_count = 1;
        uint32_t root_participant = 0;
        uint32_t window_size_tokens = 0;
        uint32_t max_hot_replicas_per_participant = 0;
        /**
         * Device-side planning may operate on a bounded rolling layer window so
         * transfer waves can be staged ahead of the layer that will consume
         * them. layer_window_count == 0 means "all layers" for compatibility.
         *
         * layer_wave_count caps how many layers a single captured controller
         * replay may plan inside that window. A value of zero means "the whole
         * window"; production graph-captured rebalance should use a small
         * nonzero wave so transfer staging can overlap with decode without
         * turning each maintenance replay into a full-model rebalance.
         */
        uint32_t layer_window_start = 0;
        uint32_t layer_window_count = 0;
        uint32_t layer_wave_count = 0;
        uint32_t flags = static_cast<uint32_t>(DeviceMoERebalanceFlags::HotReplicaCache) |
                         static_cast<uint32_t>(DeviceMoERebalanceFlags::ResetHistogramsAfterApply);
        uint32_t min_load_spread_improvement = 0;
        uint32_t min_load_spread_improvement_divisor = 0;
        uint32_t min_wave_spread_improvement_per_payload_slot = 0;
        uint32_t min_foreign_rows_per_transfer = 0;
        uint32_t min_router_spread_improvement_per_payload_slot = 0;
        uint32_t max_post_wave_load_spread_per_mille = 0;
        /**
         * @brief LLEP capacity multiplier used by graph-captured device planners.
         *
         * The shared LLEP assignment policy caps native owner work at
         * ceil(total_rows * alpha_numerator / (participants * alpha_denominator)).
         * Keeping these fields in the device config makes decode-maintenance
         * waves and request-local prefill migrations obey the same policy.
         */
        uint32_t llep_alpha_numerator = 1;
        uint32_t llep_alpha_denominator = 1;
        /**
         * @brief Balanced-load static-owner skip threshold for LLEP.
         *
         * When `llep_enable_balanced_skip` is non-zero, LLEP may intentionally
         * publish no transfers if max_expert_load / mean_expert_load is below
         * lambda_numerator / lambda_denominator.  Explicit migration tests set
         * the enable flag to zero so the LLEP path must do observable work.
         */
        uint32_t llep_lambda_numerator = 13;
        uint32_t llep_lambda_denominator = 10;
        uint32_t llep_enable_balanced_skip = 1;
        uint32_t dynamic_imbalance_threshold_per_mille =
            moe_rebalance_policy::kDefaultDynamicImbalanceThresholdPerMille;
        uint32_t dynamic_min_improvement_per_mille =
            moe_rebalance_policy::kDefaultDynamicMinImprovementPerMille;
        uint32_t dynamic_max_swaps_per_layer =
            moe_rebalance_policy::kDefaultDynamicMaxSwapsPerLayer;
        uint32_t dynamic_max_plan_entries_per_wave =
            moe_rebalance_policy::kDefaultDynamicMaxPlanEntriesPerWave;
        uint32_t dynamic_min_window_activations =
            static_cast<uint32_t>(moe_rebalance_policy::kDefaultDynamicMinWindowActivations);
        uint32_t routed_assignment_policy = kDeviceMoERebalanceAssignmentStaticOwner;
    };

    struct DeviceMoERebalanceStatus
    {
        uint32_t magic = kDeviceMoERebalanceMagic;
        uint32_t version = kDeviceMoERebalanceVersion;
        uint32_t status_code = static_cast<uint32_t>(DeviceMoERebalanceStatusCode::Ok);
        uint32_t last_epoch = 0;
        uint32_t windows_observed = 0;
        uint32_t windows_applied = 0;
        uint32_t changed_layers = 0;
        uint32_t selected_replicas = 0;
        uint32_t skipped_not_ready = 0;
        uint32_t skipped_no_resident = 0;
        uint32_t invalid_runtime_layers = 0;
        uint32_t planned_arrivals = 0;
        uint32_t plan_overflow = 0;
        uint32_t payload_bucket_requested_slots = 0;
        uint32_t payload_bucket_slots = 0;
        uint32_t payload_bucket_index = 0;
        uint32_t payload_bucket_overflow = 0;
        uint32_t skipped_no_improvement = 0;
        uint32_t dynamic_ownership_swap_attempts = 0;
        uint32_t dynamic_ownership_swap_accepts = 0;
        uint32_t dynamic_ownership_swap_rejections = 0;
        uint32_t candidate_arrivals_considered = 0;
        uint32_t candidate_arrivals_below_floor = 0;
        uint32_t candidate_arrivals_pruned_by_count_bound = 0;
        uint32_t skipped_busy_wave = 0;
        uint32_t llep_assignment_span_count = 0;
        uint32_t llep_weight_transfer_count = 0;
        uint32_t llep_standard_ep_selected = 0;
        uint32_t llep_skipped_balanced = 0;
        uint32_t llep_skipped_insufficient_spread_improvement = 0;
        uint32_t llep_skipped_insufficient_foreign_rows = 0;
        uint32_t llep_min_chunk_skips = 0;
        uint32_t llep_forced_spills = 0;
        uint64_t candidate_load_spread_improvement_total = 0;
        uint64_t candidate_load_spread_improvement_max = 0;
        uint64_t accepted_load_spread_improvement_total = 0;
        uint64_t accepted_load_spread_improvement_max = 0;
        uint64_t llep_native_rows = 0;
        uint64_t llep_spilled_rows = 0;
        uint64_t llep_required_spread_improvement = 0;
        uint64_t llep_required_foreign_rows = 0;
        uint64_t router_hot_cache_eligible_dispatches = 0;
        uint64_t router_hot_cache_used_dispatches = 0;
        uint64_t router_hot_cache_improved_dispatches = 0;
        uint64_t router_hot_cache_default_load_spread_total = 0;
        uint64_t router_hot_cache_actual_load_spread_total = 0;
        uint64_t router_hot_cache_load_spread_improvement_total = 0;
        uint64_t router_hot_cache_active_dispatches = 0;
        uint64_t router_hot_cache_miss_dispatches = 0;
        uint64_t router_hot_cache_selected_expert_slots = 0;
        uint64_t router_hot_cache_replicated_selected_expert_slots = 0;
        uint64_t pre_policy_load_total = 0;
        uint64_t pre_policy_load_min = 0;
        uint64_t pre_policy_load_max = 0;
        uint64_t pre_policy_imbalance_numerator = 0;
        uint64_t pre_policy_imbalance_denominator = 0;
        uint64_t post_policy_load_total = 0;
        uint64_t post_policy_load_min = 0;
        uint64_t post_policy_load_max = 0;
        uint64_t post_policy_imbalance_numerator = 0;
        uint64_t post_policy_imbalance_denominator = 0;
        uint64_t pre_policy_participant_load[kDeviceMoEMaxParticipants] = {};
        uint64_t post_policy_participant_load[kDeviceMoEMaxParticipants] = {};
        uint32_t window_ready_slots = 0;
        uint32_t window_required_slots = 0;
        uint32_t skipped_wave_cost_floor = 0;
        uint32_t skipped_low_router_benefit = 0;
        uint32_t skipped_post_load_spread_ceiling = 0;
        uint32_t payload_source_participant_mask = 0;
        uint32_t payload_destination_participant_mask = 0;
        uint64_t payload_edge_mask = 0;
        /**
         * @brief Sum of routed rows represented by every layer before this wave.
         *
         * This value is retained even when the wave is rejected and the public
         * post-policy fields are restored to the pre-wave placement.  Together
         * with @ref post_wave_load_total it identifies accounting mismatches in
         * a proposed transfer wave without requiring host reconstruction.
         */
        uint64_t pre_wave_load_total = 0;
        /**
         * @brief Sum of per-layer participant load spreads before this wave.
         *
         * Unlike pre_policy_load_max - pre_policy_load_min, this metric keeps
         * layer boundaries intact.  It is the left-hand side of the aggregate
         * wave-economy comparison used by both GPU backends.
         */
        uint64_t pre_wave_load_spread = 0;
        uint64_t post_wave_load_total = 0;
        uint64_t post_wave_load_spread = 0;
        /** @brief Nonzero when aggregate participant placement did not improve. */
        uint32_t skipped_participant_load_spread = 0;
        /** @brief Nonzero when the sum of per-layer load spreads did not improve. */
        uint32_t skipped_aggregate_load_spread = 0;
        /** @brief Nonzero when the configured post-wave spread ceiling rejected the wave. */
        uint32_t skipped_configured_load_spread_ceiling = 0;
        /**
         * @brief Candidates not selected because the configured transfer wave was full.
         *
         * This is scheduling backpressure, not command corruption. The planner
         * records it before accepting the candidate and leaves the command
         * buffer internally complete. By contrast, @ref plan_overflow is
         * reserved for a command that could not be represented and is fatal.
         */
        uint32_t capacity_limited_candidates = 0;
        /**
         * @brief Applied prefill experts currently backed by transient payload slots.
         *
         * GPU controllers inspect the already-active placement before planning
         * the next maintenance wave. A nonzero value therefore proves that an
         * earlier prefill LLEP/dynamic transfer was materialized, applied, and
         * published for local execution. The field is exported only through the
         * existing request-boundary status readback; recording it introduces no
         * hot-path host access, allocation, or synchronization.
         */
        uint32_t prefill_active_transfer_slot_experts = 0;
    };

    static_assert(std::is_trivially_copyable_v<DeviceMoERebalanceConfig>);
    static_assert(std::is_trivially_copyable_v<DeviceMoERebalanceStatus>);
    static_assert(std::is_trivially_copyable_v<DeviceMoERebalanceCommandBufferHeader>);
    static_assert(std::is_trivially_copyable_v<DeviceMoERebalanceWaveState>);
    static_assert(std::is_trivially_copyable_v<DeviceMoERebalanceWaveProgress>);
    static_assert(std::is_trivially_copyable_v<DeviceMoERebalanceGraphControllerState>);
    static_assert(std::is_trivially_copyable_v<DeviceMoERebalancePlanEntry>);
    static_assert(std::is_trivially_copyable_v<DeviceMoEExpertDirectoryEntry>);
    static_assert(std::is_trivially_copyable_v<DeviceMoERebalanceApplyStatus>);

    struct DeviceMoERebalanceTransferCostEstimate
    {
        uint64_t command_buffer_count = 0;
        uint64_t transfer_plan_capacity = 0;
        uint64_t payload_slot_capacity = 0;
        uint64_t payload_slot_count = 0;
        uint64_t slot_payload_bytes = 0;
        uint64_t plan_local_bytes = 0;
        uint64_t plan_gathered_bytes = 0;
        uint64_t header_local_bytes = 0;
        uint64_t header_gathered_bytes = 0;
        uint64_t source_descriptor_local_bytes = 0;
        uint64_t source_descriptor_gathered_bytes = 0;
        uint64_t selected_payload_bucket_slots = 0;
        uint64_t selected_payload_slot_count = 0;
        uint64_t selected_payload_local_capacity_bytes = 0;
        uint64_t selected_payload_gathered_capacity_bytes = 0;
        uint64_t selected_payload_transport_edge_count = 0;
        uint64_t selected_payload_transport_capacity_bytes = 0;
        uint64_t payload_local_capacity_bytes = 0;
        uint64_t payload_gathered_capacity_bytes = 0;
        uint64_t captured_payload_slack_bytes = 0;
        uint64_t useful_payload_bytes = 0;
        uint64_t wasted_payload_capacity_bytes = 0;
        uint64_t wasted_payload_transport_capacity_bytes = 0;
    };

    inline uint64_t deviceMoERebalanceSaturatingMul(uint64_t lhs, uint64_t rhs) noexcept
    {
        if (lhs == 0 || rhs == 0)
            return 0;
        constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
        return lhs > (kMax / rhs) ? kMax : lhs * rhs;
    }

    inline uint32_t deviceMoERebalanceLayerWindowCount(
        const DeviceMoERebalanceConfig &config) noexcept
    {
        return config.layer_window_count == 0
                   ? config.num_layers
                   : std::min(config.layer_window_count, config.num_layers);
    }

    inline uint32_t deviceMoERebalanceLayerWaveCount(
        const DeviceMoERebalanceConfig &config) noexcept
    {
        const uint32_t window_count = deviceMoERebalanceLayerWindowCount(config);
        return config.layer_wave_count == 0
                   ? window_count
                   : std::min(config.layer_wave_count, window_count);
    }

    inline uint32_t deviceMoERebalancePayloadBucketSlots(
        uint32_t requested_slots,
        uint32_t max_slot_capacity) noexcept
    {
        return moe_rebalance_policy::payloadBucketSlots(
            requested_slots,
            max_slot_capacity);
    }

    inline uint32_t deviceMoERebalancePayloadBucketIndex(
        uint32_t bucket_slots) noexcept
    {
        return moe_rebalance_policy::payloadBucketIndex(bucket_slots);
    }

    inline uint64_t deviceMoELoadSpread(uint64_t min_load, uint64_t max_load) noexcept
    {
        return max_load > min_load ? max_load - min_load : 0;
    }

    inline bool deviceMoERebalanceModeMovesFixedPayloadCapacity(
        DeviceMoERebalanceTransferMode mode) noexcept
    {
        return mode == DeviceMoERebalanceTransferMode::CollectiveSidebandPayload ||
               mode == DeviceMoERebalanceTransferMode::LegacyCollectiveAllGather;
    }

    inline bool deviceMoERebalanceModeUsesCollectivePayloadLane(
        DeviceMoERebalanceTransferMode mode) noexcept
    {
        return mode == DeviceMoERebalanceTransferMode::CompactTransferSlots ||
               deviceMoERebalanceModeMovesFixedPayloadCapacity(mode);
    }

    inline bool deviceMoERebalanceModeUsesTransferSlots(
        DeviceMoERebalanceTransferMode mode) noexcept
    {
        return mode == DeviceMoERebalanceTransferMode::CompactTransferSlots ||
               deviceMoERebalanceModeMovesFixedPayloadCapacity(mode);
    }

    inline bool deviceMoERebalanceModePlansMissingArrivals(
        DeviceMoERebalanceTransferMode mode) noexcept
    {
        return deviceMoERebalanceModeUsesTransferSlots(mode);
    }

    inline uint64_t deviceMoERebalanceCommandPlanCapacity(
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceTransferMode mode) noexcept
    {
        const uint64_t layer_count =
            std::max<uint32_t>(1u, deviceMoERebalanceLayerWaveCount(config));
        const uint64_t replicas_per_participant =
            std::max<uint32_t>(1u, config.max_hot_replicas_per_participant);
        uint64_t entries =
            deviceMoERebalanceSaturatingMul(layer_count, replicas_per_participant);

        if (mode == DeviceMoERebalanceTransferMode::CompactTransferSlots)
        {
            const uint64_t participants =
                std::max<uint32_t>(1u, config.participant_count);
            entries = deviceMoERebalanceSaturatingMul(
                entries,
                deviceMoERebalanceSaturatingMul(participants, participants));
        }

        return entries;
    }

    /**
     * @brief Size a merged prefill-LLEP command plan without truncation.
     *
     * Every participant materializes the transfers requested by its own
     * destination-local runtime bank. Domain projection then allgathers those
     * plans and selects participant P's destination commands from participant
     * P's record. Consequently, the merged plan can contain one full payload
     * bucket per participant even when each individual local plan is smaller.
     *
     * Payload bytes remain bounded by @p payload_slots_per_participant for each
     * source participant. Only command metadata needs the domain-wide sum.
     * Keeping this distinction explicit prevents the projection kernel from
     * filling a local-sized plan, marking overflow, and applying a truncated
     * transfer set whose assignment spans still describe the complete wave.
     *
     * @param config                        Rebalance participant topology.
     * @param local_plan_capacity           Capacity needed by one local plan.
     * @param payload_slots_per_participant Captured payload slots per source.
     * @return Saturating domain-wide command capacity.
     */
    inline uint64_t deviceMoEPrefillLLEPMergedPlanCapacity(
        const DeviceMoERebalanceConfig &config,
        uint64_t local_plan_capacity,
        uint32_t payload_slots_per_participant) noexcept
    {
        const uint64_t participant_count =
            std::max<uint32_t>(1u, config.participant_count);
        const uint64_t domain_payload_commands =
            deviceMoERebalanceSaturatingMul(
                participant_count,
                static_cast<uint64_t>(payload_slots_per_participant));
        return std::max(local_plan_capacity, domain_payload_commands);
    }

    inline uint64_t estimateDeviceMoERebalanceTransferPlanCapacity(
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceTransferMode mode,
        uint32_t local_transfer_slot_count,
        uint64_t collective_payload_slot_bytes) noexcept
    {
        if (!deviceMoERebalanceModeUsesTransferSlots(mode) ||
            local_transfer_slot_count == 0)
        {
            return 0;
        }
        if (deviceMoERebalanceModeUsesCollectivePayloadLane(mode) &&
            collective_payload_slot_bytes == 0)
        {
            return 0;
        }

        return deviceMoERebalanceCommandPlanCapacity(config, mode);
    }

    inline uint64_t estimateDeviceMoERebalancePayloadSlotCapacity(
        uint64_t transfer_plan_capacity,
        DeviceMoERebalanceTransferMode mode,
        uint32_t local_transfer_slot_count,
        uint32_t collective_payload_slot_capacity = 0) noexcept
    {
        if (!deviceMoERebalanceModeUsesCollectivePayloadLane(mode) ||
            transfer_plan_capacity == 0 ||
            local_transfer_slot_count == 0)
        {
            return 0;
        }
        const uint32_t captured_slot_capacity =
            collective_payload_slot_capacity == 0
                ? local_transfer_slot_count
                : std::min(collective_payload_slot_capacity, local_transfer_slot_count);
        return std::min<uint64_t>(
            transfer_plan_capacity,
            static_cast<uint64_t>(captured_slot_capacity));
    }

    inline DeviceMoERebalanceTransferCostEstimate estimateDeviceMoERebalanceTransferCost(
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceTransferMode mode,
        uint32_t local_transfer_slot_count,
        uint64_t collective_payload_slot_bytes,
        uint32_t copied_arrivals,
        uint32_t collective_payload_slot_capacity = 0,
        uint32_t selected_payload_bucket_slots = std::numeric_limits<uint32_t>::max(),
        uint32_t selected_payload_edge_count = std::numeric_limits<uint32_t>::max()) noexcept
    {
        DeviceMoERebalanceTransferCostEstimate estimate{};
        estimate.transfer_plan_capacity =
            estimateDeviceMoERebalanceTransferPlanCapacity(
                config,
                mode,
                local_transfer_slot_count,
                collective_payload_slot_bytes);
        if (estimate.transfer_plan_capacity == 0)
            return estimate;

        estimate.command_buffer_count = 2;
        estimate.slot_payload_bytes =
            deviceMoERebalanceModeUsesCollectivePayloadLane(mode)
                ? collective_payload_slot_bytes
                : 0;
        estimate.payload_slot_capacity =
            estimateDeviceMoERebalancePayloadSlotCapacity(
                estimate.transfer_plan_capacity,
                mode,
                local_transfer_slot_count,
                collective_payload_slot_capacity);
        estimate.payload_slot_count =
            deviceMoERebalanceModeUsesCollectivePayloadLane(mode)
                ? (mode == DeviceMoERebalanceTransferMode::CompactTransferSlots
                       ? estimate.payload_slot_capacity
                       : deviceMoERebalanceSaturatingMul(
                             estimate.payload_slot_capacity,
                             config.participant_count))
                : 0;
        const bool have_selected_payload_bucket =
            selected_payload_bucket_slots != std::numeric_limits<uint32_t>::max();
        estimate.selected_payload_bucket_slots =
            deviceMoERebalanceModeUsesCollectivePayloadLane(mode)
                ? std::min<uint64_t>(
                      have_selected_payload_bucket
                          ? static_cast<uint64_t>(selected_payload_bucket_slots)
                          : estimate.payload_slot_capacity,
                      estimate.payload_slot_capacity)
                : 0;
        estimate.selected_payload_slot_count =
            deviceMoERebalanceModeUsesCollectivePayloadLane(mode)
                ? (mode == DeviceMoERebalanceTransferMode::CompactTransferSlots
                       ? estimate.selected_payload_bucket_slots
                       : deviceMoERebalanceSaturatingMul(
                             estimate.selected_payload_bucket_slots,
                             config.participant_count))
                : 0;
        estimate.plan_local_bytes =
            deviceMoERebalanceSaturatingMul(
                deviceMoERebalanceSaturatingMul(
                    estimate.command_buffer_count,
                    estimate.transfer_plan_capacity),
                sizeof(DeviceMoERebalancePlanEntry));
        estimate.plan_gathered_bytes =
            deviceMoERebalanceSaturatingMul(
                estimate.plan_local_bytes,
                config.participant_count);
        estimate.header_local_bytes =
            deviceMoERebalanceSaturatingMul(
                estimate.command_buffer_count,
                sizeof(DeviceMoERebalanceCommandBufferHeader));
        estimate.header_gathered_bytes =
            deviceMoERebalanceSaturatingMul(
                estimate.header_local_bytes,
                config.participant_count);
        if (mode == DeviceMoERebalanceTransferMode::CompactTransferSlots)
        {
            estimate.source_descriptor_local_bytes =
                deviceMoERebalanceSaturatingMul(
                    deviceMoERebalanceSaturatingMul(
                        deviceMoERebalanceSaturatingMul(
                            config.participant_count,
                            estimate.command_buffer_count),
                        estimate.transfer_plan_capacity),
                    sizeof(DeviceMoEExpertDirectoryEntry));
            estimate.source_descriptor_gathered_bytes = 0;
        }
        if (deviceMoERebalanceModeUsesCollectivePayloadLane(mode))
        {
            estimate.selected_payload_local_capacity_bytes =
                deviceMoERebalanceSaturatingMul(
                    estimate.selected_payload_slot_count,
                    estimate.slot_payload_bytes);
            estimate.selected_payload_gathered_capacity_bytes =
                deviceMoERebalanceSaturatingMul(
                    estimate.selected_payload_local_capacity_bytes,
                    config.participant_count);
            const uint64_t transport_edge_count =
                selected_payload_edge_count == std::numeric_limits<uint32_t>::max()
                    ? static_cast<uint64_t>(config.participant_count)
                    : static_cast<uint64_t>(selected_payload_edge_count);
            estimate.selected_payload_transport_edge_count =
                transport_edge_count;
            estimate.selected_payload_transport_capacity_bytes =
                deviceMoERebalanceSaturatingMul(
                    deviceMoERebalanceSaturatingMul(
                        estimate.selected_payload_bucket_slots,
                        transport_edge_count),
                    estimate.slot_payload_bytes);
            estimate.payload_local_capacity_bytes =
                deviceMoERebalanceSaturatingMul(
                    estimate.payload_slot_count,
                    estimate.slot_payload_bytes);
            estimate.payload_gathered_capacity_bytes =
                deviceMoERebalanceSaturatingMul(
                    estimate.payload_local_capacity_bytes,
                    config.participant_count);
            estimate.captured_payload_slack_bytes =
                estimate.selected_payload_bucket_slots != 0 &&
                        estimate.payload_gathered_capacity_bytes >
                            estimate.selected_payload_gathered_capacity_bytes
                    ? estimate.payload_gathered_capacity_bytes -
                          estimate.selected_payload_gathered_capacity_bytes
                    : 0;
            estimate.useful_payload_bytes =
                deviceMoERebalanceSaturatingMul(
                    copied_arrivals,
                    estimate.slot_payload_bytes);
            estimate.wasted_payload_capacity_bytes =
                estimate.selected_payload_gathered_capacity_bytes > estimate.useful_payload_bytes
                    ? estimate.selected_payload_gathered_capacity_bytes - estimate.useful_payload_bytes
                    : 0;
            estimate.wasted_payload_transport_capacity_bytes =
                estimate.selected_payload_transport_capacity_bytes > estimate.useful_payload_bytes
                    ? estimate.selected_payload_transport_capacity_bytes - estimate.useful_payload_bytes
                    : 0;
        }
        return estimate;
    }

    inline bool validateDeviceMoERebalanceConfig(const DeviceMoERebalanceConfig &config) noexcept
    {
        return config.magic == kDeviceMoERebalanceMagic &&
               config.version == kDeviceMoERebalanceVersion &&
               config.num_layers > 0 &&
               config.num_experts > 0 &&
               config.num_experts <= kDeviceMoEMaxExperts &&
               config.top_k > 0 &&
               config.top_k <= kDeviceMoEMaxTopK &&
               config.participant_count > 0 &&
               config.participant_count <= kDeviceMoEMaxParticipants &&
               config.participant_id < config.participant_count &&
               moe_rebalance_policy::hasValidRootParticipant(config) &&
               config.window_size_tokens > 0 &&
               config.llep_alpha_numerator > 0 &&
               config.llep_alpha_denominator > 0 &&
               config.llep_lambda_numerator > 0 &&
               config.llep_lambda_denominator > 0 &&
               (config.layer_window_count == 0 ||
                config.layer_window_start < config.num_layers);
    }

    inline bool deviceMoEIsRootParticipant(const DeviceMoERebalanceConfig &config) noexcept
    {
        return moe_rebalance_policy::isRootParticipant(config);
    }

    inline uint32_t deviceMoEParticipantBit(uint32_t participant) noexcept
    {
        return moe_rebalance_policy::participantBit(participant);
    }

    inline uint32_t deviceMoEValidParticipantMask(uint32_t participant_count) noexcept
    {
        return moe_rebalance_policy::validParticipantMask(participant_count);
    }

    inline bool deviceMoEPlacementHasLocalHotReplica(
        const DeviceMoEPlacementBank &bank,
        const DeviceMoERebalanceConfig &config,
        uint32_t valid_mask) noexcept
    {
        const uint32_t participant_bit = deviceMoEParticipantBit(config.participant_id);
        for (uint32_t expert = 0; expert < config.num_experts; ++expert)
        {
            const auto &desc = bank.experts[expert];
            uint32_t resident_mask = bank.resident_participant_mask[expert] & valid_mask;
            if (desc.owner_participant >= 0 &&
                desc.owner_participant < static_cast<int32_t>(config.participant_count))
            {
                resident_mask |= deviceMoEParticipantBit(
                    static_cast<uint32_t>(desc.owner_participant));
            }
            const bool local_resident =
                (resident_mask & participant_bit) != 0u ||
                bank.local_compute_mask[expert] != 0u;
            const bool owner_local =
                desc.owner_participant == static_cast<int32_t>(config.participant_id);
            const bool multi_resident =
                (resident_mask & (resident_mask - 1u)) != 0u;
            if (local_resident && !owner_local && multi_resident)
                return true;
        }
        return false;
    }

    inline uint64_t deviceMoEDirectoryIndex(
        const DeviceMoERebalanceConfig &config,
        uint32_t participant,
        uint32_t layer,
        uint32_t expert) noexcept
    {
        const uint64_t participant_stride =
            static_cast<uint64_t>(config.num_layers) * static_cast<uint64_t>(config.num_experts);
        return static_cast<uint64_t>(participant) * participant_stride +
               static_cast<uint64_t>(layer) * static_cast<uint64_t>(config.num_experts) +
               static_cast<uint64_t>(expert);
    }

    inline bool deviceMoEDescriptorReady(const DeviceMoEExpertDescriptor &desc) noexcept
    {
        return desc.logical_expert_id >= 0 &&
               desc.gate.valid() &&
               desc.up.valid() &&
               desc.down.valid();
    }

    inline bool deviceMoETransferDescriptorReady(const DeviceMoEExpertDescriptor &desc) noexcept
    {
        return desc.gate.valid() &&
               desc.up.valid() &&
               desc.down.valid();
    }

    inline bool deviceMoENativeVnniFormatForCodebook(
        uint8_t codebook_id,
        uint8_t &payload_bytes_per_block,
        uint8_t &is_asymmetric,
        uint8_t &has_emins) noexcept
    {
        payload_bytes_per_block = 0;
        is_asymmetric = 0;
        has_emins = 0;
        switch (codebook_id)
        {
        case 0:
        case 4:
            payload_bytes_per_block = 16;
            return true;
        case 5:
            payload_bytes_per_block = 16;
            is_asymmetric = 1;
            return true;
        case 6:
            payload_bytes_per_block = 20;
            return true;
        case 7:
            payload_bytes_per_block = 20;
            is_asymmetric = 1;
            return true;
        case 8:
            payload_bytes_per_block = 24;
            is_asymmetric = 1;
            return true;
        case 9:
            payload_bytes_per_block = 12;
            is_asymmetric = 1;
            return true;
        case 10:
            payload_bytes_per_block = 8;
            is_asymmetric = 1;
            has_emins = 1;
            return true;
        case 11:
            payload_bytes_per_block = 13;
            return true;
        case 12:
            payload_bytes_per_block = 12;
            return true;
        case 13:
        case 14:
            payload_bytes_per_block = 9;
            is_asymmetric = 1;
            return true;
        case 15:
            payload_bytes_per_block = 8;
            return true;
        case 16:
        case 17:
            payload_bytes_per_block = 6;
            is_asymmetric = 1;
            return true;
        case 19:
        case 20:
            payload_bytes_per_block = 32;
            return true;
        default:
            return false;
        }
    }

    inline bool deviceMoEProjectionFormat(
        const DeviceNativeVNNIMatrixDesc &desc,
        uint8_t &payload_bytes_per_block,
        uint8_t &is_asymmetric,
        uint8_t &has_emins) noexcept
    {
        return deviceMoENativeVnniFormatForCodebook(
            desc.codebook_id,
            payload_bytes_per_block,
            is_asymmetric,
            has_emins);
    }

    /**
     * @brief Resolve immutable allocation capacity for a NativeVNNI descriptor.
     *
     * Model weights have an exact-format allocation and therefore leave the
     * explicit capacity fields at zero. Transfer slots carry non-zero capacity
     * metadata because their active codebook can change without reallocating or
     * changing any graph-captured pointer.
     */
    inline bool deviceMoEProjectionAllocationCapacity(
        const DeviceNativeVNNIMatrixDesc &desc,
        uint8_t &payload_bytes_per_block,
        uint8_t &has_mins,
        uint8_t &has_emins) noexcept
    {
        if (desc.allocation_payload_bytes_per_block != 0u)
        {
            payload_bytes_per_block = desc.allocation_payload_bytes_per_block;
            has_mins = desc.allocation_has_mins;
            has_emins = desc.allocation_has_emins;
            return true;
        }

        uint8_t is_asymmetric = 0;
        if (!deviceMoEProjectionFormat(
                desc,
                payload_bytes_per_block,
                is_asymmetric,
                has_emins))
        {
            return false;
        }
        has_mins = is_asymmetric;
        return true;
    }

    /**
     * @brief Test whether one formatted matrix fits a stable transfer allocation.
     */
    inline bool deviceMoEMatrixFitsTransferCapacity(
        const DeviceNativeVNNIMatrixDesc &src,
        const DeviceNativeVNNIMatrixDesc &dst) noexcept
    {
        if (src.n != dst.n ||
            src.k != dst.k ||
            src.blocks_per_row != dst.blocks_per_row)
        {
            return false;
        }

        uint8_t src_payload_bytes = 0;
        uint8_t src_is_asymmetric = 0;
        uint8_t src_has_emins = 0;
        uint8_t dst_payload_capacity = 0;
        uint8_t dst_has_mins = 0;
        uint8_t dst_has_emins = 0;
        return deviceMoEProjectionFormat(
                   src,
                   src_payload_bytes,
                   src_is_asymmetric,
                   src_has_emins) &&
               deviceMoEProjectionAllocationCapacity(
                   dst,
                   dst_payload_capacity,
                   dst_has_mins,
                   dst_has_emins) &&
               src_payload_bytes <= dst_payload_capacity &&
               (src_is_asymmetric == 0u || dst_has_mins != 0u) &&
               (src_has_emins == 0u || dst_has_emins != 0u);
    }

    /**
     * @brief Retarget a reusable allocation to the source matrix's active format.
     *
     * Allocation pointers and capacity metadata are intentionally preserved.
     * Only the logical matrix geometry and codebook metadata consumed by grouped
     * kernels are updated.
     */
    inline void deviceMoERetargetTransferMatrixFormat(
        DeviceNativeVNNIMatrixDesc &dst,
        const DeviceNativeVNNIMatrixDesc &src) noexcept
    {
        dst.n = src.n;
        dst.k = src.k;
        dst.blocks_per_row = src.blocks_per_row;
        dst.codebook_id = src.codebook_id;
    }

    inline uint64_t deviceMoEMatrixBlockCount(const DeviceNativeVNNIMatrixDesc &desc) noexcept
    {
        return static_cast<uint64_t>(desc.blocks_per_row) * static_cast<uint64_t>(desc.n);
    }

    inline uint64_t deviceMoEMatrixPayloadBytes(
        const DeviceNativeVNNIMatrixDesc &desc,
        const DeviceMoEExpertDirectoryEntry &entry) noexcept
    {
        (void)entry;
        uint8_t payload_bytes_per_block = 0;
        uint8_t is_asymmetric = 0;
        uint8_t has_emins = 0;
        if (!deviceMoEProjectionFormat(
                desc,
                payload_bytes_per_block,
                is_asymmetric,
                has_emins))
        {
            return 0;
        }
        return deviceMoEMatrixBlockCount(desc) * static_cast<uint64_t>(payload_bytes_per_block);
    }

    inline uint64_t deviceMoEMatrixScalesBytes(const DeviceNativeVNNIMatrixDesc &desc) noexcept
    {
        return deviceMoEMatrixBlockCount(desc) * sizeof(uint16_t);
    }

    inline uint64_t deviceMoEMatrixMinsBytes(
        const DeviceNativeVNNIMatrixDesc &desc,
        const DeviceMoEExpertDirectoryEntry &entry) noexcept
    {
        (void)entry;
        uint8_t payload_bytes_per_block = 0;
        uint8_t is_asymmetric = 0;
        uint8_t has_emins = 0;
        if (!deviceMoEProjectionFormat(
                desc,
                payload_bytes_per_block,
                is_asymmetric,
                has_emins))
        {
            return 0;
        }
        return is_asymmetric != 0u ? deviceMoEMatrixScalesBytes(desc) : 0u;
    }

    inline uint64_t deviceMoEMatrixEminsBytes(
        const DeviceNativeVNNIMatrixDesc &desc,
        const DeviceMoEExpertDirectoryEntry &entry) noexcept
    {
        (void)entry;
        uint8_t payload_bytes_per_block = 0;
        uint8_t is_asymmetric = 0;
        uint8_t has_emins = 0;
        if (!deviceMoEProjectionFormat(
                desc,
                payload_bytes_per_block,
                is_asymmetric,
                has_emins))
        {
            return 0;
        }
        return has_emins != 0u
                   ? deviceMoEMatrixBlockCount(desc) * sizeof(uint32_t)
                   : 0u;
    }

    inline bool deviceMoEMatrixCopyReady(
        const DeviceNativeVNNIMatrixDesc &desc,
        const DeviceMoEExpertDirectoryEntry &entry) noexcept
    {
        (void)entry;
        uint8_t payload_bytes_per_block = 0;
        uint8_t is_asymmetric = 0;
        uint8_t has_emins = 0;
        if (!desc.valid() ||
            !deviceMoEProjectionFormat(
                desc,
                payload_bytes_per_block,
                is_asymmetric,
                has_emins) ||
            payload_bytes_per_block == 0u)
            return false;
        if (is_asymmetric != 0u && !desc.mins)
            return false;
        if (has_emins != 0u && !desc.emins)
            return false;
        return true;
    }

    inline bool deviceMoEDirectoryCopyReady(const DeviceMoEExpertDirectoryEntry &entry) noexcept
    {
        return deviceMoEMatrixCopyReady(entry.descriptor.gate, entry) &&
               deviceMoEMatrixCopyReady(entry.descriptor.up, entry) &&
               deviceMoEMatrixCopyReady(entry.descriptor.down, entry);
    }

    inline bool deviceMoEDirectoryEntryReady(
        const DeviceMoEExpertDirectoryEntry &entry,
        uint32_t participant,
        uint32_t layer,
        uint32_t expert) noexcept
    {
        const uint32_t required =
            static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::Valid) |
            static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::Resident);
        return entry.participant == participant &&
               entry.layer == layer &&
               entry.expert == expert &&
               (entry.flags & required) == required &&
               deviceMoEDescriptorReady(entry.descriptor) &&
               deviceMoEDirectoryCopyReady(entry);
    }

    inline bool deviceMoETransferSlotReadyForCopy(
        const DeviceMoEExpertDirectoryEntry &entry,
        uint32_t participant,
        uint32_t layer,
        uint32_t expert) noexcept
    {
        (void)layer;
        (void)expert;
        const uint32_t required =
            static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::Valid) |
            static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::TransferSlot);
        return entry.participant == participant &&
               (entry.flags & required) == required &&
               deviceMoETransferDescriptorReady(entry.descriptor) &&
               deviceMoEDirectoryCopyReady(entry);
    }

    inline bool deviceMoETransferSlotCopyComplete(
        const DeviceMoEExpertDirectoryEntry &entry,
        uint32_t participant,
        uint32_t layer,
        uint32_t expert) noexcept
    {
        return deviceMoETransferSlotReadyForCopy(entry, participant, layer, expert) &&
               entry.layer == layer &&
               entry.expert == expert &&
               entry.descriptor.logical_expert_id == static_cast<int32_t>(expert) &&
               (entry.flags & static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::Resident)) != 0u &&
               (entry.flags & static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::CopyComplete)) != 0u;
    }

    inline bool deviceMoEDirectoryFitsTransferCapacity(
        const DeviceMoEExpertDirectoryEntry &src,
        const DeviceMoEExpertDirectoryEntry &dst) noexcept
    {
        return deviceMoEMatrixFitsTransferCapacity(
                   src.descriptor.gate,
                   dst.descriptor.gate) &&
               deviceMoEMatrixFitsTransferCapacity(
                   src.descriptor.up,
                   dst.descriptor.up) &&
               deviceMoEMatrixFitsTransferCapacity(
                   src.descriptor.down,
                   dst.descriptor.down);
    }

    /**
     * @brief Retarget all projection formats in a reusable expert transfer slot.
     */
    inline void deviceMoERetargetTransferDirectoryFormats(
        DeviceMoEExpertDirectoryEntry &dst,
        const DeviceMoEExpertDirectoryEntry &src) noexcept
    {
        deviceMoERetargetTransferMatrixFormat(
            dst.descriptor.gate,
            src.descriptor.gate);
        deviceMoERetargetTransferMatrixFormat(
            dst.descriptor.up,
            src.descriptor.up);
        deviceMoERetargetTransferMatrixFormat(
            dst.descriptor.down,
            src.descriptor.down);
    }

    inline int32_t deviceMoEFirstResidentParticipant(
        uint32_t resident_mask,
        uint32_t participant_count,
        int32_t preferred_participant = -1,
        int32_t excluded_participant = -1) noexcept
    {
        return moe_rebalance_policy::firstResidentParticipant(
            resident_mask,
            participant_count,
            preferred_participant,
            excluded_participant);
    }

    inline bool deviceMoERebalanceCandidateCanAffectLocalCompute(
        const DeviceMoEExpertDescriptor &desc,
        uint32_t resident_mask,
        const DeviceMoERebalanceConfig &config,
        uint32_t participant_bit,
        bool plan_missing_arrivals) noexcept
    {
        (void)participant_bit;
        return moe_rebalance_policy::candidateCanAffectLocalCompute(
            desc,
            resident_mask,
            config,
            plan_missing_arrivals);
    }

    inline bool deviceMoERebalanceCandidateCanAffectDomainCompute(
        const DeviceMoEExpertDescriptor &desc,
        uint32_t resident_mask,
        const DeviceMoERebalanceConfig &config,
        bool plan_missing_arrivals) noexcept
    {
        return moe_rebalance_policy::candidateCanAffectDomainCompute(
            desc,
            resident_mask,
            config,
            plan_missing_arrivals);
    }

    inline moe_rebalance_policy::DestinationChoice deviceMoEBestMissingResidentDestination(
        const uint64_t *current_participant_load,
        uint64_t expert_count,
        uint32_t current_resident_mask,
        const DeviceMoERebalanceConfig &config,
        const uint32_t *planned_replicas_per_participant,
        uint32_t max_replicas_per_participant) noexcept
    {
        return moe_rebalance_policy::bestDynamicMissingResidentDestination(
            current_participant_load,
            expert_count,
            current_resident_mask,
            config.participant_count,
            -1,
            planned_replicas_per_participant,
            max_replicas_per_participant,
            config.window_size_tokens,
            config.min_load_spread_improvement,
            config.min_load_spread_improvement_divisor);
    }

    inline bool deviceMoEAppendPlanEntry(
        DeviceMoERebalancePlanEntry *plan_entries,
        uint32_t *plan_count,
        uint32_t plan_capacity,
        const DeviceMoERebalancePlanEntry &entry,
        DeviceMoERebalanceStatus *status) noexcept
    {
        if (!plan_entries || !plan_count)
            return false;
        if (*plan_count >= plan_capacity)
        {
            if (status)
                ++status->plan_overflow;
            return false;
        }
        plan_entries[*plan_count] = entry;
        ++(*plan_count);
        if (status)
            ++status->planned_arrivals;
        return true;
    }

    inline bool packDeviceMoELocalDirectoryHost(
        const DeviceMoELayerRuntime *runtime_layers,
        DeviceMoEExpertDirectoryEntry *local_directory,
        const DeviceMoERebalanceConfig &config) noexcept
    {
        if (!validateDeviceMoERebalanceConfig(config) ||
            !runtime_layers ||
            !local_directory)
        {
            return false;
        }

        const uint32_t participant_bit = deviceMoEParticipantBit(config.participant_id);
        const uint32_t valid_mask = deviceMoEValidParticipantMask(config.participant_count);
        for (uint32_t layer = 0; layer < config.num_layers; ++layer)
        {
            const auto &runtime = runtime_layers[layer];
            const bool runtime_ok =
                runtime.active_bank <= 1u &&
                runtime.expert_count == config.num_experts &&
                runtime.top_k == config.top_k &&
                runtime.participant_id == config.participant_id &&
                runtime.participant_count == config.participant_count;
            const auto &bank = runtime.banks[runtime.active_bank <= 1u ? runtime.active_bank : 0u];

            for (uint32_t expert = 0; expert < config.num_experts; ++expert)
            {
                auto &entry = local_directory[
                    static_cast<uint64_t>(layer) * static_cast<uint64_t>(config.num_experts) +
                    static_cast<uint64_t>(expert)];
                entry = DeviceMoEExpertDirectoryEntry{};
                entry.layer = layer;
                entry.expert = expert;
                entry.participant = config.participant_id;
                entry.epoch = runtime.active_epoch;
                entry.generation = runtime.active_epoch;

                if (!runtime_ok)
                    continue;

                const uint32_t resident_mask =
                    bank.resident_participant_mask[expert] & valid_mask;
                const bool local_resident =
                    (resident_mask & participant_bit) != 0u ||
                    bank.local_compute_mask[expert] != 0u;
                const auto &desc = bank.experts[expert];
                entry.resident_mask = resident_mask;
                entry.slot_index = desc.local_slot >= 0
                                       ? static_cast<uint32_t>(desc.local_slot)
                                       : kDeviceMoEInvalidSlot;

                if (!local_resident || !deviceMoEDescriptorReady(desc))
                    continue;

                entry.descriptor = desc;
                if (!deviceMoEDirectoryCopyReady(entry))
                {
                    entry.descriptor = DeviceMoEExpertDescriptor{};
                    continue;
                }
                entry.flags =
                    static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::Valid) |
                    static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::Resident);
                if (bank.local_compute_mask[expert] != 0u)
                {
                    entry.flags |=
                        static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::LocalCompute);
                }
            }
        }

        return true;
    }

    inline bool applyDeviceMoERebalanceArrivalsHost(
        DeviceMoELayerRuntime *runtime_layers,
        const DeviceMoERebalancePlanEntry *plan_entries,
        uint32_t plan_count,
        const DeviceMoEExpertDirectoryEntry *gathered_directory,
        const DeviceMoEExpertDirectoryEntry *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceApplyStatus *status) noexcept
    {
        if (status)
            *status = DeviceMoERebalanceApplyStatus{};
        if (!validateDeviceMoERebalanceConfig(config))
        {
            if (status)
                status->status_code = static_cast<uint32_t>(DeviceMoERebalanceApplyStatusCode::InvalidConfig);
            return false;
        }
        if (!runtime_layers)
        {
            if (status)
                status->status_code = static_cast<uint32_t>(DeviceMoERebalanceApplyStatusCode::InvalidRuntime);
            return false;
        }
        if (!plan_entries && plan_count > 0)
        {
            if (status)
                status->status_code = static_cast<uint32_t>(DeviceMoERebalanceApplyStatusCode::MissingPlan);
            return false;
        }
        uint8_t changed_layer[kDeviceMoEMaxExperts] = {};
        static_assert(kDeviceMoEMaxExperts >= kDeviceMoEMaxParticipants);
        const uint32_t participant_bit = deviceMoEParticipantBit(config.participant_id);
        const uint32_t valid_mask = deviceMoEValidParticipantMask(config.participant_count);

        for (uint32_t i = 0; i < plan_count; ++i)
        {
            if (status)
                ++status->plan_entries_seen;
            const auto &plan = plan_entries[i];
            if (plan.op == static_cast<uint32_t>(DeviceMoERebalancePlanOp::None))
                continue;
            if ((plan.op != static_cast<uint32_t>(DeviceMoERebalancePlanOp::ExpertPayloadArrival) &&
                 plan.op != static_cast<uint32_t>(DeviceMoERebalancePlanOp::ResidentExpertAssignment) &&
                 plan.op != static_cast<uint32_t>(DeviceMoERebalancePlanOp::OwnershipTransfer)) ||
                plan.layer >= config.num_layers ||
                plan.expert >= config.num_experts ||
                plan.source_participant >= config.participant_count ||
                plan.destination_participant >= config.participant_count)
            {
                if (status)
                    ++status->invalid_plan_entries;
                continue;
            }
            const bool destination_local =
                plan.destination_participant == config.participant_id;
            const uint32_t destination_bit =
                deviceMoEParticipantBit(plan.destination_participant);

            auto &runtime = runtime_layers[plan.layer];
            if (runtime.active_bank > 1u ||
                runtime.expert_count != config.num_experts ||
                runtime.top_k != config.top_k ||
                runtime.participant_id != config.participant_id ||
                runtime.participant_count != config.participant_count)
            {
                if (status)
                    ++status->invalid_plan_entries;
                continue;
            }

            auto &bank = runtime.banks[runtime.active_bank];
            auto desc = bank.experts[plan.expert];
            uint32_t resident_mask =
                (bank.resident_participant_mask[plan.expert] |
                 plan.source_resident_mask |
                 destination_bit) &
                valid_mask;
            const bool ownership_transfer =
                plan.op == static_cast<uint32_t>(DeviceMoERebalancePlanOp::OwnershipTransfer);
            if (ownership_transfer)
                resident_mask = destination_bit & valid_mask;
            if ((plan.op == static_cast<uint32_t>(DeviceMoERebalancePlanOp::ExpertPayloadArrival) ||
                 ownership_transfer) &&
                destination_local)
            {
                if (status)
                    ++status->required_local_arrivals;
                if (!gathered_directory || !local_transfer_slots)
                {
                    if (status)
                        status->status_code =
                            static_cast<uint32_t>(DeviceMoERebalanceApplyStatusCode::MissingDirectory);
                    return false;
                }

                const auto &source_entry = gathered_directory[
                    deviceMoEDirectoryIndex(config, plan.source_participant, plan.layer, plan.expert)];
                if (!deviceMoEDirectoryEntryReady(
                        source_entry,
                        plan.source_participant,
                        plan.layer,
                        plan.expert) ||
                    !deviceMoEDirectoryCopyReady(source_entry))
                {
                    if (status)
                        ++status->missing_source_descriptors;
                    continue;
                }

                if (plan.destination_slot >= local_transfer_slot_count)
                {
                    if (status)
                        ++status->missing_destination_slots;
                    continue;
                }

                const auto &slot_entry = local_transfer_slots[plan.destination_slot];
                if (!deviceMoETransferSlotReadyForCopy(
                        slot_entry,
                        config.participant_id,
                        plan.layer,
                        plan.expert))
                {
                    if (status)
                        ++status->missing_destination_slots;
                    continue;
                }
                if (!deviceMoETransferSlotCopyComplete(
                        slot_entry,
                        config.participant_id,
                        plan.layer,
                        plan.expert))
                {
                    if (status)
                        ++status->copy_incomplete;
                    continue;
                }
                if (status)
                    ++status->ready_local_arrivals;
                if (slot_entry.descriptor.logical_expert_id != static_cast<int32_t>(plan.expert))
                {
                    if (status)
                        ++status->descriptor_mismatches;
                    continue;
                }
                desc = slot_entry.descriptor;
                desc.owner_participant = ownership_transfer
                                             ? static_cast<int32_t>(plan.destination_participant)
                                             : bank.experts[plan.expert].owner_participant;
                resident_mask = ownership_transfer
                                    ? destination_bit & valid_mask
                                    : resident_mask | participant_bit;
            }
            else if (plan.op == static_cast<uint32_t>(DeviceMoERebalancePlanOp::ResidentExpertAssignment) &&
                     destination_local &&
                     (resident_mask & participant_bit) == 0u)
            {
                if (status)
                    ++status->missing_source_descriptors;
                continue;
            }
            else if (ownership_transfer)
            {
                desc.owner_participant = static_cast<int32_t>(plan.destination_participant);
            }

            if (ownership_transfer)
                desc.flags &= ~toMoEExpertFlags(DeviceMoEExpertFlags::Replicated);
            else
                desc.flags |= toMoEExpertFlags(DeviceMoEExpertFlags::Replicated);
            if (destination_local)
            {
                desc.flags |= toMoEExpertFlags(DeviceMoEExpertFlags::Valid |
                                               DeviceMoEExpertFlags::Resident |
                                               DeviceMoEExpertFlags::LocalCompute);
                bank.local_compute_mask[plan.expert] = 1u;
                bank.replica_role[plan.expert] =
                    ownership_transfer ||
                            desc.owner_participant == static_cast<int32_t>(config.participant_id)
                        ? static_cast<uint8_t>(DeviceMoEReplicaRole::Primary)
                        : static_cast<uint8_t>(DeviceMoEReplicaRole::Replica);
                if (status)
                    ++status->applied_arrivals;
            }
            else if (ownership_transfer)
            {
                desc.flags &= ~toMoEExpertFlags(DeviceMoEExpertFlags::Resident |
                                                DeviceMoEExpertFlags::LocalCompute);
                bank.local_compute_mask[plan.expert] = 0u;
                bank.replica_role[plan.expert] =
                    static_cast<uint8_t>(DeviceMoEReplicaRole::None);
            }
            bank.experts[plan.expert] = desc;
            bank.resident_participant_mask[plan.expert] =
                resident_mask & valid_mask;
            if (plan.op == static_cast<uint32_t>(
                               DeviceMoERebalancePlanOp::ExpertPayloadArrival) ||
                ownership_transfer)
            {
                /*
                 * Every participant consumes the same gathered plan, while
                 * only the destination owns the transfer-slot descriptor.
                 * Publish this sticky layer marker from the global command so
                 * prefix capture cannot make a participant-local durability
                 * decision.
                 */
                bank.transient_placement_observed = 1u;
            }
            if (plan.layer < kDeviceMoEMaxExperts)
                changed_layer[plan.layer] = 1u;
        }

        if (status)
        {
            for (uint32_t layer = 0; layer < config.num_layers && layer < kDeviceMoEMaxExperts; ++layer)
            {
                if (changed_layer[layer] == 0u)
                    continue;

                auto &bank = runtime_layers[layer].banks[runtime_layers[layer].active_bank];
                uint32_t multi_resident = 0;
                for (uint32_t expert = 0; expert < config.num_experts; ++expert)
                {
                    const uint32_t resident_mask =
                        bank.resident_participant_mask[expert] & valid_mask;
                    if (moe_rebalance_policy::residentCount(
                            resident_mask, config.participant_count) > 1u)
                    {
                        ++multi_resident;
                    }
                }
                bank.multi_resident_expert_count = multi_resident;
                ++status->changed_layers;
                status->post_apply_multi_resident_experts += multi_resident;
            }
            status->status_code = static_cast<uint32_t>(DeviceMoERebalanceApplyStatusCode::Ok);
        }
        return true;
    }

    inline uint32_t deviceMoEPackedHistogramLayerCount(
        const DeviceMoERebalanceConfig &config) noexcept
    {
        return deviceMoERebalanceLayerWaveCount(config);
    }

    inline uint64_t deviceMoEGatheredHistogramCount(
        const uint64_t *gathered_histograms,
        const DeviceMoERebalanceConfig &config,
        uint32_t participant,
        uint32_t wave_layer,
        uint32_t expert) noexcept
    {
        const uint64_t participant_stride =
            static_cast<uint64_t>(deviceMoEPackedHistogramLayerCount(config)) *
            static_cast<uint64_t>(config.num_experts);
        const uint64_t layer_stride = static_cast<uint64_t>(config.num_experts);
        return gathered_histograms[static_cast<uint64_t>(participant) * participant_stride +
                                   static_cast<uint64_t>(wave_layer) * layer_stride +
                                   static_cast<uint64_t>(expert)];
    }

    inline uint64_t deviceMoEGlobalExpertCount(
        const uint64_t *gathered_histograms,
        const DeviceMoERebalanceConfig &config,
        uint32_t wave_layer,
        uint32_t expert) noexcept
    {
        uint64_t total = 0;
        for (uint32_t participant = 0; participant < config.participant_count; ++participant)
            total += deviceMoEGatheredHistogramCount(
                gathered_histograms, config, participant, wave_layer, expert);
        return total;
    }

    inline uint64_t deviceMoEWindowObservedSlots(
        const uint64_t *gathered_histograms,
        const DeviceMoERebalanceConfig &config) noexcept
    {
        if (!gathered_histograms || config.num_layers == 0 ||
            config.top_k == 0 || config.participant_count == 0)
        {
            return 0;
        }

        const uint32_t packed_layer_count = deviceMoEPackedHistogramLayerCount(config);
        if (packed_layer_count == 0)
            return 0;

        uint64_t last_layer_slots = 0;
        const uint32_t last_layer = packed_layer_count - 1u;
        for (uint32_t participant = 0; participant < config.participant_count; ++participant)
        {
            for (uint32_t expert = 0; expert < config.num_experts; ++expert)
            {
                last_layer_slots += deviceMoEGatheredHistogramCount(
                    gathered_histograms, config, participant, last_layer, expert);
            }
        }

        return last_layer_slots;
    }

    inline uint64_t deviceMoEWindowRequiredSlots(
        const DeviceMoERebalanceConfig &config) noexcept
    {
        return static_cast<uint64_t>(config.window_size_tokens) *
               static_cast<uint64_t>(config.top_k);
    }

    inline uint32_t deviceMoEClampU64ToU32(uint64_t value) noexcept
    {
        return value > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
                   ? std::numeric_limits<uint32_t>::max()
                   : static_cast<uint32_t>(value);
    }

    inline bool deviceMoEWindowReady(
        const uint64_t *gathered_histograms,
        const DeviceMoERebalanceConfig &config) noexcept
    {
        const uint64_t last_layer_slots =
            deviceMoEWindowObservedSlots(gathered_histograms, config);
        const uint64_t slots_per_token = static_cast<uint64_t>(config.top_k);
        return slots_per_token > 0 &&
               last_layer_slots >= deviceMoEWindowRequiredSlots(config);
    }

    inline bool applyDeviceMoERebalancePolicyHost(
        DeviceMoELayerRuntime *runtime_layers,
        const uint64_t *gathered_histograms,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceStatus *status,
        DeviceMoERebalancePlanEntry *plan_entries = nullptr,
        uint32_t *plan_count = nullptr,
        uint32_t plan_capacity = 0)
    {
        if (status)
        {
            *status = DeviceMoERebalanceStatus{};
        }
        if (plan_count)
            *plan_count = 0;
        if (!validateDeviceMoERebalanceConfig(config))
        {
            if (status)
                status->status_code = static_cast<uint32_t>(DeviceMoERebalanceStatusCode::InvalidConfig);
            return false;
        }
        if (!runtime_layers)
        {
            if (status)
                status->status_code = static_cast<uint32_t>(DeviceMoERebalanceStatusCode::InvalidRuntime);
            return false;
        }
        if (!gathered_histograms)
        {
            if (status)
                status->status_code = static_cast<uint32_t>(DeviceMoERebalanceStatusCode::MissingHistogram);
            return false;
        }
        if (!deviceMoEWindowReady(gathered_histograms, config))
        {
            if (status)
            {
                status->window_ready_slots =
                    deviceMoEClampU64ToU32(
                        deviceMoEWindowObservedSlots(gathered_histograms, config));
                status->window_required_slots =
                    deviceMoEClampU64ToU32(deviceMoEWindowRequiredSlots(config));
                status->status_code = static_cast<uint32_t>(DeviceMoERebalanceStatusCode::WindowNotReady);
                status->skipped_not_ready = 1;
                status->skipped_busy_wave = 0;
            }
            return true;
        }

        uint32_t changed_layers = 0;
        uint32_t selected_replicas = 0;
        uint32_t skipped_no_resident = 0;
        uint32_t skipped_no_improvement = 0;
        uint32_t dynamic_ownership_swap_attempts = 0;
        uint32_t dynamic_ownership_swap_accepts = 0;
        uint32_t dynamic_ownership_swap_rejections = 0;
        uint32_t candidate_arrivals_considered = 0;
        uint32_t candidate_arrivals_below_floor = 0;
        uint32_t candidate_arrivals_pruned_by_count_bound = 0;
        uint64_t candidate_load_spread_improvement_total = 0;
        uint64_t candidate_load_spread_improvement_max = 0;
        uint64_t accepted_load_spread_improvement_total = 0;
        uint64_t accepted_load_spread_improvement_max = 0;
        uint64_t router_hot_cache_eligible_dispatches = 0;
        uint64_t router_hot_cache_used_dispatches = 0;
        uint64_t router_hot_cache_improved_dispatches = 0;
        uint64_t router_hot_cache_default_load_spread_total = 0;
        uint64_t router_hot_cache_actual_load_spread_total = 0;
        uint64_t router_hot_cache_load_spread_improvement_total = 0;
        uint64_t router_hot_cache_active_dispatches = 0;
        uint64_t router_hot_cache_miss_dispatches = 0;
        uint64_t router_hot_cache_selected_expert_slots = 0;
        uint64_t router_hot_cache_replicated_selected_expert_slots = 0;
        uint64_t post_wave_load_total = 0;
        uint64_t post_wave_load_spread = 0;
        uint32_t hot_cache_active_layers = 0;
        uint32_t prefill_active_transfer_slot_experts = 0;
        uint32_t last_epoch = 0;
        uint64_t pre_policy_load[kDeviceMoEMaxParticipants] = {};
        uint64_t post_policy_load[kDeviceMoEMaxParticipants] = {};
        const uint32_t participant_bit = deviceMoEParticipantBit(config.participant_id);
        const uint32_t valid_mask = deviceMoEValidParticipantMask(config.participant_count);
        const bool plan_missing_arrivals = hasDeviceMoERebalanceFlag(
            config.flags,
            DeviceMoERebalanceFlags::PlanMissingArrivals);
        const bool collect_load_stats = hasDeviceMoERebalanceFlag(
            config.flags,
            DeviceMoERebalanceFlags::CollectLoadStats);
        const uint32_t layer_window_count =
            config.layer_window_count == 0
                ? config.num_layers
                : std::min(config.layer_window_count, config.num_layers);
        const uint32_t layer_wave_count =
            config.layer_wave_count == 0
                ? layer_window_count
                : std::min(config.layer_wave_count, layer_window_count);
        const uint32_t layer_window_start =
            config.num_layers == 0 ? 0 : (config.layer_window_start % config.num_layers);

        /*
         * Inspect every layer rather than only the rolling maintenance wave.
         * Prefill placement publication is model-wide, while the maintenance
         * cursor intentionally sees only a bounded layer slice per replay.
         */
        for (uint32_t layer = 0; layer < config.num_layers; ++layer)
        {
            prefill_active_transfer_slot_experts +=
                deviceMoELayerActiveTransferSlotExpertCount(runtime_layers[layer]);
        }

        for (uint32_t window_index = 0; window_index < layer_wave_count; ++window_index)
        {
            const uint32_t layer =
                (layer_window_start + window_index) % config.num_layers;
            auto &runtime = runtime_layers[layer];
            if (runtime.active_bank > 1u ||
                runtime.expert_count != config.num_experts ||
                runtime.top_k != config.top_k ||
                runtime.participant_id != config.participant_id ||
                runtime.participant_count != config.participant_count)
            {
                if (status)
                    ++status->invalid_runtime_layers;
                continue;
            }
            router_hot_cache_eligible_dispatches +=
                runtime.router_hot_cache_eligible_dispatches;
            router_hot_cache_used_dispatches +=
                runtime.router_hot_cache_used_dispatches;
            router_hot_cache_improved_dispatches +=
                runtime.router_hot_cache_improved_dispatches;
            router_hot_cache_default_load_spread_total +=
                runtime.router_hot_cache_default_load_spread_total;
            router_hot_cache_actual_load_spread_total +=
                runtime.router_hot_cache_actual_load_spread_total;
            router_hot_cache_load_spread_improvement_total +=
                runtime.router_hot_cache_load_spread_improvement_total;
            router_hot_cache_active_dispatches +=
                runtime.router_hot_cache_active_dispatches;
            router_hot_cache_miss_dispatches +=
                runtime.router_hot_cache_miss_dispatches;
            router_hot_cache_selected_expert_slots +=
                runtime.router_hot_cache_selected_expert_slots;
            router_hot_cache_replicated_selected_expert_slots +=
                runtime.router_hot_cache_replicated_selected_expert_slots;
            if (hasDeviceMoERebalanceFlag(config.flags,
                                          DeviceMoERebalanceFlags::ResetHistogramsAfterApply))
            {
                runtime.router_hot_cache_eligible_dispatches = 0;
                runtime.router_hot_cache_used_dispatches = 0;
                runtime.router_hot_cache_improved_dispatches = 0;
                runtime.router_hot_cache_default_load_spread_total = 0;
                runtime.router_hot_cache_actual_load_spread_total = 0;
                runtime.router_hot_cache_load_spread_improvement_total = 0;
                runtime.router_hot_cache_active_dispatches = 0;
                runtime.router_hot_cache_miss_dispatches = 0;
                runtime.router_hot_cache_selected_expert_slots = 0;
                runtime.router_hot_cache_replicated_selected_expert_slots = 0;
            }

            const uint32_t inactive_bank = 1u - runtime.active_bank;
            const auto &active = runtime.banks[runtime.active_bank];
            auto &next = runtime.banks[inactive_bank];
            if (deviceMoEPlacementHasLocalHotReplica(active, config, valid_mask))
                ++hot_cache_active_layers;

            if (collect_load_stats)
            {
                for (uint32_t expert = 0; expert < config.num_experts; ++expert)
                {
                    const auto &desc = active.experts[expert];
                    uint32_t resident_mask =
                        active.resident_participant_mask[expert] & valid_mask;
                    if (desc.owner_participant >= 0 &&
                        desc.owner_participant < static_cast<int32_t>(config.participant_count))
                    {
                        resident_mask |=
                            deviceMoEParticipantBit(static_cast<uint32_t>(desc.owner_participant));
                    }
                    const uint64_t count = deviceMoEGlobalExpertCount(
                        gathered_histograms, config, window_index, expert);
                    for (uint32_t participant = 0; participant < config.participant_count; ++participant)
                    {
                        pre_policy_load[participant] +=
                            moe_rebalance_policy::projectedParticipantLoadForExpert(
                                count,
                                resident_mask,
                                config.participant_count,
                                participant);
                    }
                }
            }

            next = active;
            next.epoch = runtime.active_epoch + 1u;
            next.expert_count = config.num_experts;
            uint32_t post_policy_resident_mask[kDeviceMoEMaxExperts] = {};
            uint64_t current_policy_load[kDeviceMoEMaxParticipants] = {};
            uint64_t candidate_policy_load[kDeviceMoEMaxParticipants] = {};
            uint64_t current_policy_total = 0;

            for (uint32_t expert = 0; expert < config.num_experts; ++expert)
            {
                auto &desc = next.experts[expert];
                uint32_t resident_mask = next.resident_participant_mask[expert] & valid_mask;
                if (desc.owner_participant >= 0 &&
                    desc.owner_participant < static_cast<int32_t>(config.participant_count))
                {
                    resident_mask |=
                        deviceMoEParticipantBit(static_cast<uint32_t>(desc.owner_participant));
                }
                const bool owner_local = desc.owner_participant == static_cast<int32_t>(config.participant_id);
                const bool local_resident = (resident_mask & participant_bit) != 0u;
                const bool multi_resident =
                    moe_rebalance_policy::residentCount(
                        resident_mask, config.participant_count) > 1u;
                if (multi_resident)
                    desc.flags |= toMoEExpertFlags(DeviceMoEExpertFlags::Replicated);
                else
                    desc.flags &= ~toMoEExpertFlags(DeviceMoEExpertFlags::Replicated);
                if (local_resident)
                {
                    desc.flags |= toMoEExpertFlags(DeviceMoEExpertFlags::Valid |
                                                  DeviceMoEExpertFlags::Resident |
                                                  DeviceMoEExpertFlags::LocalCompute);
                }
                else
                {
                    desc.flags &= ~toMoEExpertFlags(DeviceMoEExpertFlags::LocalCompute);
                }
                next.local_compute_mask[expert] = local_resident ? 1u : 0u;
                next.replica_role[expert] =
                    local_resident
                        ? (owner_local
                               ? static_cast<uint8_t>(DeviceMoEReplicaRole::Primary)
                               : static_cast<uint8_t>(DeviceMoEReplicaRole::Replica))
                        : static_cast<uint8_t>(DeviceMoEReplicaRole::None);
                next.resident_participant_mask[expert] = resident_mask;
                post_policy_resident_mask[expert] =
                    next.resident_participant_mask[expert] & valid_mask;
            }
            for (uint32_t expert = 0; expert < config.num_experts; ++expert)
            {
                const auto &desc = next.experts[expert];
                uint32_t resident_mask =
                    post_policy_resident_mask[expert] & valid_mask;
                if (desc.owner_participant >= 0 &&
                    desc.owner_participant < static_cast<int32_t>(config.participant_count))
                {
                    resident_mask |=
                        deviceMoEParticipantBit(static_cast<uint32_t>(desc.owner_participant));
                }
                const uint64_t count = deviceMoEGlobalExpertCount(
                    gathered_histograms, config, window_index, expert);
                for (uint32_t participant = 0; participant < config.participant_count; ++participant)
                {
                    current_policy_load[participant] +=
                        moe_rebalance_policy::projectedParticipantLoadForExpert(
                            count,
                            resident_mask,
                            config.participant_count,
                            participant);
                }
            }
            for (uint32_t participant = 0; participant < config.participant_count; ++participant)
                current_policy_total += current_policy_load[participant];

            uint32_t selected[kDeviceMoEMaxExperts] = {};
            uint32_t selected_count = 0;
            uint32_t local_replicas = 0;
            const uint32_t max_local_replicas = hasDeviceMoERebalanceFlag(
                                                    config.flags,
                                                    DeviceMoERebalanceFlags::HotReplicaCache)
                                                    ? config.max_hot_replicas_per_participant
                                                    : 0u;

            for (uint32_t rank = 0; rank < config.num_experts && local_replicas < max_local_replicas; ++rank)
            {
                if (plan_count &&
                    moe_rebalance_policy::commandBufferFull(*plan_count, plan_capacity))
                {
                    if (status)
                        ++status->plan_overflow;
                    break;
                }

                uint32_t best_expert = kDeviceMoEMaxExperts;
                uint64_t best_value = 0;
                uint64_t best_count = 0;
                for (uint32_t expert = 0; expert < config.num_experts; ++expert)
                {
                    bool already_selected = false;
                    for (uint32_t i = 0; i < selected_count; ++i)
                    {
                        if (selected[i] == expert)
                        {
                            already_selected = true;
                            break;
                        }
                    }
                    if (already_selected)
                        continue;
                    const auto &candidate_desc = next.experts[expert];
                    uint32_t candidate_resident_mask =
                        next.resident_participant_mask[expert] & valid_mask;
                    if (candidate_desc.owner_participant >= 0 &&
                        candidate_desc.owner_participant < static_cast<int32_t>(config.participant_count))
                    {
                        candidate_resident_mask |=
                            deviceMoEParticipantBit(static_cast<uint32_t>(candidate_desc.owner_participant));
                    }
                    if (!deviceMoERebalanceCandidateCanAffectLocalCompute(
                            candidate_desc,
                            candidate_resident_mask,
                            config,
                            participant_bit,
                            plan_missing_arrivals))
                    {
                        continue;
                    }

                    const uint64_t count = deviceMoEGlobalExpertCount(
                        gathered_histograms, config, window_index, expert);
                    if (count == 0)
                        continue;
                    const bool candidate_local_resident =
                        (candidate_resident_mask & participant_bit) != 0u;
                    const bool candidate_owner_local =
                        candidate_desc.owner_participant ==
                        static_cast<int32_t>(config.participant_id);
                    uint64_t candidate_value = count;
                    if (plan_missing_arrivals &&
                        !candidate_local_resident &&
                        !candidate_owner_local)
                    {
                        if (!moe_rebalance_policy::expertCountCanMeetLoadSpreadFloor(
                                count,
                                current_policy_total,
                                config.min_load_spread_improvement,
                                config.min_load_spread_improvement_divisor))
                        {
                            ++candidate_arrivals_pruned_by_count_bound;
                            ++skipped_no_improvement;
                            continue;
                        }
                        const int32_t source_participant = deviceMoEFirstResidentParticipant(
                            candidate_resident_mask,
                            config.participant_count,
                            candidate_desc.owner_participant,
                            static_cast<int32_t>(config.participant_id));
	                        const uint32_t proposed_resident_mask =
	                            (candidate_resident_mask | participant_bit) & valid_mask;
	                        const auto delta =
	                            source_participant >= 0
	                                ? moe_rebalance_policy::evaluateAddingResidentDynamicSpread(
	                                      current_policy_load,
	                                      count,
	                                      candidate_resident_mask,
	                                      proposed_resident_mask,
	                                      config.participant_count,
	                                      static_cast<uint32_t>(source_participant),
	                                      config.participant_id,
	                                      config.window_size_tokens,
	                                      config.min_load_spread_improvement,
	                                      config.min_load_spread_improvement_divisor)
	                                : moe_rebalance_policy::LoadSpreadDelta{};
                        ++candidate_arrivals_considered;
                        candidate_load_spread_improvement_total += delta.improvement;
                        candidate_load_spread_improvement_max =
                            std::max(candidate_load_spread_improvement_max,
                                     delta.improvement);
                        if (!delta.meets_floor)
                        {
                            ++candidate_arrivals_below_floor;
                            ++skipped_no_improvement;
                            continue;
                        }
                        candidate_value = delta.improvement;
                    }
                    if (moe_rebalance_policy::candidateValueIsBetter(
                            candidate_value,
                            count,
                            expert,
                            best_value,
                            best_count,
                            best_expert))
                    {
                        best_value = candidate_value;
                        best_count = count;
                        best_expert = expert;
                    }
                }

                if (best_expert >= config.num_experts || best_count == 0)
                    break;
                selected[selected_count++] = best_expert;

                auto &desc = next.experts[best_expert];
                uint32_t resident_mask = next.resident_participant_mask[best_expert] & valid_mask;
                if (desc.owner_participant >= 0 &&
                    desc.owner_participant < static_cast<int32_t>(config.participant_count))
                {
                    resident_mask |= deviceMoEParticipantBit(static_cast<uint32_t>(desc.owner_participant));
                }
                next.resident_participant_mask[best_expert] = resident_mask;
                const bool local_resident = (resident_mask & participant_bit) != 0u;
                const bool owner_local = desc.owner_participant == static_cast<int32_t>(config.participant_id);
                const bool replicated = (resident_mask & (resident_mask - 1u)) != 0u;
                if (plan_missing_arrivals && !local_resident && !owner_local)
                {
                    const uint32_t proposed_resident_mask =
                        (resident_mask | participant_bit) & valid_mask;
                    const uint64_t count = deviceMoEGlobalExpertCount(
                        gathered_histograms, config, window_index, best_expert);
                    const int32_t source_participant = deviceMoEFirstResidentParticipant(
                        resident_mask,
                        config.participant_count,
                        desc.owner_participant,
                        static_cast<int32_t>(config.participant_id));
	                    const auto delta =
	                        source_participant >= 0
	                            ? moe_rebalance_policy::evaluateAddingResidentDynamicSpread(
	                                  current_policy_load,
	                                  count,
	                                  resident_mask,
	                                  proposed_resident_mask,
	                                  config.participant_count,
	                                  static_cast<uint32_t>(source_participant),
	                                  config.participant_id,
	                                  config.window_size_tokens,
	                                  config.min_load_spread_improvement,
	                                  config.min_load_spread_improvement_divisor)
	                            : moe_rebalance_policy::LoadSpreadDelta{};
                    if (!delta.meets_floor)
                    {
                        ++skipped_no_improvement;
                        continue;
                    }
                    if (!moe_rebalance_policy::addingResidentImprovesDynamicSpread(
                            current_policy_load,
                            count,
                            resident_mask,
                            proposed_resident_mask,
                            config.participant_count,
                            static_cast<uint32_t>(source_participant),
                            config.participant_id,
                            candidate_policy_load,
                            config.window_size_tokens,
                            config.min_load_spread_improvement,
                            config.min_load_spread_improvement_divisor))
                    {
                        ++skipped_no_improvement;
                        continue;
                    }

                    DeviceMoERebalancePlanEntry entry;
                    entry.op = static_cast<uint32_t>(DeviceMoERebalancePlanOp::ExpertPayloadArrival);
                    entry.layer = layer;
                    entry.expert = best_expert;
                    entry.source_participant = static_cast<uint32_t>(std::max<int32_t>(0, source_participant));
                    entry.destination_participant = config.participant_id;
                    entry.source_resident_mask = resident_mask;
                    entry.destination_slot = plan_count ? *plan_count : kDeviceMoEInvalidSlot;
                    entry.payload_slot = entry.destination_slot;
                    if (source_participant >= 0 &&
                        deviceMoEAppendPlanEntry(plan_entries, plan_count, plan_capacity, entry, status))
                    {
                        post_policy_resident_mask[best_expert] =
                            proposed_resident_mask;
                        for (uint32_t participant = 0; participant < config.participant_count; ++participant)
                            current_policy_load[participant] = candidate_policy_load[participant];
                        accepted_load_spread_improvement_total += delta.improvement;
                        accepted_load_spread_improvement_max =
                            std::max(accepted_load_spread_improvement_max,
                                     delta.improvement);
                        ++local_replicas;
                    }
                    else
                    {
                        ++skipped_no_resident;
                    }
                    continue;
                }
                if (!replicated)
                {
                    ++skipped_no_resident;
                    continue;
                }

                desc.flags |= toMoEExpertFlags(DeviceMoEExpertFlags::Replicated);
                if (local_resident)
                {
                    next.local_compute_mask[best_expert] = 1u;
                    next.replica_role[best_expert] = owner_local
                                                        ? static_cast<uint8_t>(DeviceMoEReplicaRole::Primary)
                                                        : static_cast<uint8_t>(DeviceMoEReplicaRole::Replica);
                    if (!owner_local)
                        ++local_replicas;
                    ++selected_replicas;
                }
                post_policy_resident_mask[best_expert] =
                    next.resident_participant_mask[best_expert] & valid_mask;
            }

            runtime.active_bank = inactive_bank;
            runtime.active_epoch = next.epoch;
            last_epoch = runtime.active_epoch;
            ++changed_layers;

            if (hasDeviceMoERebalanceFlag(config.flags,
                                          DeviceMoERebalanceFlags::ResetHistogramsAfterApply))
            {
                std::fill(runtime.decode_histogram,
                          runtime.decode_histogram + config.num_experts,
                          0ULL);
                std::fill(runtime.decode_local_histogram,
                          runtime.decode_local_histogram + config.num_experts,
                          0ULL);
            }

            uint64_t layer_post_total = 0;
            uint64_t layer_post_min = 0;
            uint64_t layer_post_max = 0;
            moe_rebalance_policy::finalizeLoadSpread(
                current_policy_load,
                config.participant_count,
                layer_post_total,
                layer_post_min,
                layer_post_max);
            post_wave_load_total += layer_post_total;
            post_wave_load_spread += layer_post_max - layer_post_min;

            if (collect_load_stats)
            {
                for (uint32_t expert = 0; expert < config.num_experts; ++expert)
                {
                    const auto &desc = next.experts[expert];
                    uint32_t resident_mask =
                        post_policy_resident_mask[expert] & valid_mask;
                    if (desc.owner_participant >= 0 &&
                        desc.owner_participant < static_cast<int32_t>(config.participant_count))
                    {
                        resident_mask |=
                            deviceMoEParticipantBit(static_cast<uint32_t>(desc.owner_participant));
                    }
                    const uint64_t count = deviceMoEGlobalExpertCount(
                        gathered_histograms, config, window_index, expert);
                    for (uint32_t participant = 0; participant < config.participant_count; ++participant)
                    {
                        post_policy_load[participant] +=
                            moe_rebalance_policy::projectedParticipantLoadForExpert(
                                count,
                                resident_mask,
                                config.participant_count,
                                participant);
                    }
                }
            }
        }

        if (status)
        {
            uint64_t pre_total = 0;
            uint64_t pre_min = 0;
            uint64_t pre_max = 0;
            uint64_t post_total = 0;
            uint64_t post_min = 0;
            uint64_t post_max = 0;
            if (collect_load_stats)
            {
                moe_rebalance_policy::finalizeLoadSpread(
                    pre_policy_load,
                    config.participant_count,
                    pre_total,
                    pre_min,
                    pre_max);
                moe_rebalance_policy::finalizeLoadSpread(
                    post_policy_load,
                    config.participant_count,
                    post_total,
                    post_min,
                    post_max);
            }
            status->status_code = static_cast<uint32_t>(DeviceMoERebalanceStatusCode::Ok);
            status->windows_observed = 1;
            status->windows_applied = changed_layers > 0 ? 1u : 0u;
            status->changed_layers = changed_layers;
            status->selected_replicas = selected_replicas;
            status->skipped_no_resident = skipped_no_resident;
            status->skipped_no_improvement = skipped_no_improvement;
            status->dynamic_ownership_swap_attempts = dynamic_ownership_swap_attempts;
            status->dynamic_ownership_swap_accepts = dynamic_ownership_swap_accepts;
            status->dynamic_ownership_swap_rejections = dynamic_ownership_swap_rejections;
            status->candidate_arrivals_considered = candidate_arrivals_considered;
            status->candidate_arrivals_below_floor = candidate_arrivals_below_floor;
            status->candidate_arrivals_pruned_by_count_bound =
                candidate_arrivals_pruned_by_count_bound;
            status->candidate_load_spread_improvement_total =
                candidate_load_spread_improvement_total;
            status->candidate_load_spread_improvement_max =
                candidate_load_spread_improvement_max;
            status->accepted_load_spread_improvement_total =
                accepted_load_spread_improvement_total;
            status->accepted_load_spread_improvement_max =
                accepted_load_spread_improvement_max;
            status->router_hot_cache_eligible_dispatches =
                router_hot_cache_eligible_dispatches;
            status->router_hot_cache_used_dispatches =
                router_hot_cache_used_dispatches;
            status->router_hot_cache_improved_dispatches =
                router_hot_cache_improved_dispatches;
            status->router_hot_cache_default_load_spread_total =
                router_hot_cache_default_load_spread_total;
            status->router_hot_cache_actual_load_spread_total =
                router_hot_cache_actual_load_spread_total;
            status->router_hot_cache_load_spread_improvement_total =
                router_hot_cache_load_spread_improvement_total;
            status->router_hot_cache_active_dispatches =
                router_hot_cache_active_dispatches;
            status->router_hot_cache_miss_dispatches =
                router_hot_cache_miss_dispatches;
            status->router_hot_cache_selected_expert_slots =
                router_hot_cache_selected_expert_slots;
            status->router_hot_cache_replicated_selected_expert_slots =
                router_hot_cache_replicated_selected_expert_slots;
            status->prefill_active_transfer_slot_experts =
                prefill_active_transfer_slot_experts;
            status->last_epoch = last_epoch;
            status->window_ready_slots =
                deviceMoEClampU64ToU32(
                    deviceMoEWindowObservedSlots(gathered_histograms, config));
            status->window_required_slots =
                deviceMoEClampU64ToU32(deviceMoEWindowRequiredSlots(config));
            status->post_wave_load_total = post_wave_load_total;
            status->post_wave_load_spread = post_wave_load_spread;
            if (collect_load_stats)
            {
                status->pre_policy_load_total = pre_total;
                status->pre_policy_load_min = pre_min;
                status->pre_policy_load_max = pre_max;
                status->pre_policy_imbalance_numerator = pre_max - pre_min;
                status->pre_policy_imbalance_denominator = pre_total;
                status->post_policy_load_total = post_total;
                status->post_policy_load_min = post_min;
                status->post_policy_load_max = post_max;
                status->post_policy_imbalance_numerator = post_max - post_min;
                status->post_policy_imbalance_denominator = post_total;
                for (uint32_t participant = 0;
                     participant < kDeviceMoEMaxParticipants;
                     ++participant)
                {
                    status->pre_policy_participant_load[participant] =
                        participant < config.participant_count
                            ? pre_policy_load[participant]
                            : 0ULL;
                    status->post_policy_participant_load[participant] =
                        participant < config.participant_count
                            ? post_policy_load[participant]
                            : 0ULL;
                }
            }
        }
        return true;
    }

} // namespace llaminar2
