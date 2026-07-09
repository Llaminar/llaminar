/**
 * @file ROCmMoEKernels.hip
 * @brief HIP __global__ kernels for MoE operations
 *
 * Contains HIP kernels for device-native MoE execution:
 * 1. gate_logits_kernel    — GEMV: hidden × gate_weights^T → logits
 * 2. softmax_topk_kernel   — Row-wise softmax + top-k selection
 * 3. gather_tokens_kernel  — Gather token rows into contiguous batch
 * 4. scatter_add_kernel    — Weighted scatter-add expert outputs
 * 5. shared_expert_gate_kernel — sigmoid(dot) * scale per token
 * 6. swiglu_kernel         — silu(gate) * up activation
 *
 * Compiled by hipcc. Only includes HIP headers + HipDeviceGuard.
 * Linkage to host C++ is via extern "C" wrapper functions.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>
#include <cstdio>
#include <cfloat>

#include "execution/moe/DeviceMoERebalancePolicyShared.h"
#include "execution/moe/LeastLoadedExpertAssignment.h"
#include "../../../backends/rocm/HipDeviceGuard.h"

namespace
{
    constexpr int kDeviceMoEMaxExperts = 256;
    constexpr int kDeviceMoEMaxTopK = 16;
    constexpr int kDeviceMoEMaxParticipants = 8;
    constexpr int kGateLogitsExpertsPerBlock = 4;
    constexpr int kGateLogitsGroupedThreads = 256;
    constexpr int kDeviceMoERebalanceApplyThreads = kGateLogitsGroupedThreads;
    constexpr uint32_t kDeviceMoEFlagValid = 1u << 0;
    constexpr uint32_t kDeviceMoEFlagResident = 1u << 1;
    constexpr uint32_t kDeviceMoEFlagReplicated = 1u << 2;
    constexpr uint32_t kDeviceMoEFlagLocalCompute = 1u << 4;
    constexpr uint32_t kDeviceMoEFlagTransferSlot = 1u << 5;
    constexpr uint32_t kDeviceMoEDirectoryFlagValid = 1u << 0;
    constexpr uint32_t kDeviceMoEDirectoryFlagResident = 1u << 1;
    constexpr uint32_t kDeviceMoEDirectoryFlagLocalCompute = 1u << 2;
    constexpr uint32_t kDeviceMoEDirectoryFlagTransferSlot = 1u << 3;
    constexpr uint32_t kDeviceMoEDirectoryFlagCopyComplete = 1u << 4;
    constexpr uint32_t kDeviceMoEReplicaRoleNone = 0u;
    constexpr uint32_t kDeviceMoEReplicaRolePrimary = 1u;
    constexpr uint32_t kDeviceMoEReplicaRoleReplica = 2u;
    constexpr uint32_t kDeviceMoERebalanceMagic = 0x4d4f4552u;
    constexpr uint32_t kDeviceMoERebalanceVersion = 1u;
    constexpr uint32_t kDeviceMoERebalanceFlagHotReplicaCache = 1u << 0;
    constexpr uint32_t kDeviceMoERebalanceFlagResetHistograms = 1u << 1;
    constexpr uint32_t kDeviceMoERebalanceFlagPlanMissingArrivals = 1u << 2;
    constexpr uint32_t kDeviceMoERebalanceFlagDeferRuntimeApply = 1u << 3;
    constexpr uint32_t kDeviceMoERebalanceFlagCollectLoadStats = 1u << 4;
    constexpr uint32_t kDeviceMoERebalanceStatusOk = 0u;
    constexpr uint32_t kDeviceMoERebalanceStatusWindowNotReady = 1u;
    constexpr uint32_t kDeviceMoERebalanceStatusInvalidConfig = 2u;
    constexpr uint32_t kDeviceMoERebalanceStatusInvalidRuntime = 3u;
    constexpr uint32_t kDeviceMoERebalanceStatusMissingHistogram = 4u;
    constexpr uint32_t kDeviceMoERebalanceStatusMissingTransferCompletion = 5u;
    constexpr uint32_t kDeviceMoERebalanceApplyStatusInProgress = 5u;
    constexpr uint32_t kDeviceMoERebalancePlanExpertPayloadArrival = 1u;
    constexpr uint32_t kDeviceMoERebalancePlanResidentExpertAssignment = 2u;
    constexpr uint32_t kDeviceMoERebalancePlanOwnershipTransfer = 3u;
    constexpr uint32_t kDeviceMoERebalanceAssignmentLeastLoadedEP = 1u;
    constexpr uint32_t kDeviceMoERebalancePhasePlanAssignments = 2u;
    constexpr uint32_t kDeviceMoERebalanceLifecycleIdle = 0u;
    constexpr uint32_t kDeviceMoERebalanceLifecyclePlanning = 1u;
    constexpr uint32_t kDeviceMoERebalanceLifecycleTransferInFlight = 2u;
    constexpr uint32_t kDeviceMoERebalanceLifecycleReadyToApply = 3u;
    constexpr uint32_t kDeviceMoERebalanceLifecycleApplying = 4u;
    constexpr uint32_t kDeviceMoERebalanceLifecycleApplied = 5u;
    constexpr uint32_t kDeviceMoERebalanceLifecycleError = 6u;
    constexpr uint32_t kDeviceMoEInvalidSlot = 0xffffffffu;

    __device__ __forceinline__ bool rebalance_plan_requires_payload(uint32_t op)
    {
        return op == kDeviceMoERebalancePlanExpertPayloadArrival ||
               op == kDeviceMoERebalancePlanOwnershipTransfer;
    }

    __device__ __forceinline__ bool rebalance_plan_applies_runtime(uint32_t op)
    {
        return rebalance_plan_requires_payload(op) ||
               op == kDeviceMoERebalancePlanResidentExpertAssignment;
    }

    __device__ inline float route_weight_to_float(__half value)
    {
        return __half2float(value);
    }

    __device__ inline float route_weight_to_float(hip_bfloat16 value)
    {
        return static_cast<float>(value);
    }

    struct DeviceNativeVNNIMatrixDescView
    {
        const uint8_t *payload;
        const void *scales;
        const void *mins;
        const void *emins;
        int n;
        int k;
        uint32_t blocks_per_row;
        uint8_t codebook_id;
        uint8_t reserved[3];
    };

    struct DeviceMoEExpertDescriptorView
    {
        DeviceNativeVNNIMatrixDescView gate;
        DeviceNativeVNNIMatrixDescView up;
        DeviceNativeVNNIMatrixDescView down;
        int32_t logical_expert_id;
        int32_t owner_participant;
        int32_t local_slot;
        uint32_t flags;
    };

    struct DeviceMoEPlacementBankView
    {
        DeviceMoEExpertDescriptorView experts[kDeviceMoEMaxExperts];
        uint8_t local_compute_mask[kDeviceMoEMaxExperts];
        uint8_t replica_role[kDeviceMoEMaxExperts];
        uint32_t resident_participant_mask[kDeviceMoEMaxExperts];
        uint32_t epoch;
        uint32_t expert_count;
        uint32_t reserved[2];
    };

    struct DeviceMoELayerRuntimeView
    {
        uint32_t active_bank;
        uint32_t active_epoch;
        uint32_t expert_count;
        uint32_t top_k;
        DeviceMoEPlacementBankView banks[2];
        int32_t topk_expert_ids[kDeviceMoEMaxTopK];
        float topk_weights[kDeviceMoEMaxTopK];
        uint64_t decode_histogram[kDeviceMoEMaxExperts];
        uint64_t decode_local_histogram[kDeviceMoEMaxExperts];
        uint64_t router_hot_cache_eligible_dispatches;
        uint64_t router_hot_cache_used_dispatches;
        uint64_t router_hot_cache_improved_dispatches;
        uint64_t router_hot_cache_default_load_spread_total;
        uint64_t router_hot_cache_actual_load_spread_total;
        uint64_t router_hot_cache_load_spread_improvement_total;
        uint64_t router_hot_cache_active_dispatches;
        uint64_t router_hot_cache_miss_dispatches;
        uint64_t router_hot_cache_selected_expert_slots;
        uint64_t router_hot_cache_replicated_selected_expert_slots;
        int32_t *route_expert_ids;
        float *route_weights;
        int32_t *route_participant_ids;
        int32_t *expert_counts;
        int32_t *expert_offsets;
        int32_t *grouped_token_ids;
        float *grouped_route_weights;
        float *grouped_gate_scratch;
        float *grouped_up_scratch;
        float *grouped_output_partials;
        void *decode_scratch;
        void *reserved_ptrs[3];
        uint64_t reserved_u64[4];
        uint32_t prefill_token_capacity;
        uint32_t prefill_route_capacity;
        uint32_t participant_id;
        uint32_t participant_count;
    };

    struct DeviceMoERebalanceConfigView
    {
        uint32_t magic;
        uint32_t version;
        uint32_t num_layers;
        uint32_t num_experts;
        uint32_t top_k;
        uint32_t participant_id;
        uint32_t participant_count;
        uint32_t root_participant;
        uint32_t window_size_tokens;
        uint32_t max_hot_replicas_per_participant;
        uint32_t layer_window_start;
        uint32_t layer_window_count;
        uint32_t layer_wave_count;
        uint32_t flags;
        uint32_t min_load_spread_improvement;
        uint32_t min_load_spread_improvement_divisor;
        uint32_t min_wave_spread_improvement_per_payload_slot;
        uint32_t min_foreign_rows_per_transfer;
        uint32_t min_router_spread_improvement_per_payload_slot;
        uint32_t max_post_wave_load_spread_per_mille;
        uint32_t llep_alpha_numerator;
        uint32_t llep_alpha_denominator;
        uint32_t llep_lambda_numerator;
        uint32_t llep_lambda_denominator;
        uint32_t llep_enable_balanced_skip;
        uint32_t dynamic_imbalance_threshold_per_mille;
        uint32_t dynamic_min_improvement_per_mille;
        uint32_t dynamic_max_swaps_per_layer;
        uint32_t dynamic_max_plan_entries_per_wave;
        uint32_t dynamic_min_window_activations;
        uint32_t routed_assignment_policy;
    };

    struct DeviceMoERebalanceStatusView
    {
        uint32_t magic;
        uint32_t version;
        uint32_t status_code;
        uint32_t last_epoch;
        uint32_t windows_observed;
        uint32_t windows_applied;
        uint32_t changed_layers;
        uint32_t selected_replicas;
        uint32_t skipped_not_ready;
        uint32_t skipped_no_resident;
        uint32_t invalid_runtime_layers;
        uint32_t planned_arrivals;
        uint32_t plan_overflow;
        uint32_t payload_bucket_requested_slots;
        uint32_t payload_bucket_slots;
        uint32_t payload_bucket_index;
        uint32_t payload_bucket_overflow;
        uint32_t skipped_no_improvement;
        uint32_t dynamic_ownership_swap_attempts;
        uint32_t dynamic_ownership_swap_accepts;
        uint32_t dynamic_ownership_swap_rejections;
        uint32_t candidate_arrivals_considered;
        uint32_t candidate_arrivals_below_floor;
        uint32_t candidate_arrivals_pruned_by_count_bound;
        uint32_t skipped_busy_wave;
        uint32_t llep_assignment_span_count;
        uint32_t llep_weight_transfer_count;
        uint32_t llep_standard_ep_selected;
        uint32_t llep_skipped_balanced;
        uint32_t llep_skipped_insufficient_spread_improvement;
        uint32_t llep_skipped_insufficient_foreign_rows;
        uint32_t llep_min_chunk_skips;
        uint32_t llep_forced_spills;
        uint64_t candidate_load_spread_improvement_total;
        uint64_t candidate_load_spread_improvement_max;
        uint64_t accepted_load_spread_improvement_total;
        uint64_t accepted_load_spread_improvement_max;
        uint64_t llep_native_rows;
        uint64_t llep_spilled_rows;
        uint64_t llep_required_spread_improvement;
        uint64_t llep_required_foreign_rows;
        uint64_t router_hot_cache_eligible_dispatches;
        uint64_t router_hot_cache_used_dispatches;
        uint64_t router_hot_cache_improved_dispatches;
        uint64_t router_hot_cache_default_load_spread_total;
        uint64_t router_hot_cache_actual_load_spread_total;
        uint64_t router_hot_cache_load_spread_improvement_total;
        uint64_t router_hot_cache_active_dispatches;
        uint64_t router_hot_cache_miss_dispatches;
        uint64_t router_hot_cache_selected_expert_slots;
        uint64_t router_hot_cache_replicated_selected_expert_slots;
        uint64_t pre_policy_load_total;
        uint64_t pre_policy_load_min;
        uint64_t pre_policy_load_max;
        uint64_t pre_policy_imbalance_numerator;
        uint64_t pre_policy_imbalance_denominator;
        uint64_t post_policy_load_total;
        uint64_t post_policy_load_min;
        uint64_t post_policy_load_max;
        uint64_t post_policy_imbalance_numerator;
        uint64_t post_policy_imbalance_denominator;
        uint64_t pre_policy_participant_load[kDeviceMoEMaxParticipants];
        uint64_t post_policy_participant_load[kDeviceMoEMaxParticipants];
        uint32_t window_ready_slots;
        uint32_t window_required_slots;
        uint32_t skipped_wave_cost_floor;
        uint32_t skipped_low_router_benefit;
        uint32_t skipped_post_load_spread_ceiling;
        uint32_t payload_source_participant_mask;
        uint32_t payload_destination_participant_mask;
        uint64_t payload_edge_mask;
        uint64_t post_wave_load_total;
        uint64_t post_wave_load_spread;
    };

    __device__ __forceinline__ void fail_fast_incomplete_llep_transfer()
    {
        __builtin_trap();
    }

    __device__ __forceinline__ bool prefill_llep_transfer_status_complete(
        const DeviceMoERebalanceStatusView *__restrict__ status,
        uint64_t expected_transfer_count,
        uint64_t expected_span_count)
    {
        if (!status ||
            status->magic != kDeviceMoERebalanceMagic ||
            status->version != kDeviceMoERebalanceVersion ||
            status->status_code != kDeviceMoERebalanceStatusOk ||
            status->plan_overflow != 0u ||
            status->payload_bucket_overflow != 0u ||
            expected_transfer_count > 0xffffffffULL ||
            expected_span_count > 0xffffffffULL)
        {
            return false;
        }

        const uint32_t transfer_count =
            static_cast<uint32_t>(expected_transfer_count);
        const uint32_t span_count =
            static_cast<uint32_t>(expected_span_count);
        return status->llep_weight_transfer_count == transfer_count &&
               status->planned_arrivals == transfer_count &&
               status->llep_assignment_span_count == span_count;
    }

    struct DeviceMoERebalancePlanEntryView
    {
        uint32_t op;
        uint32_t layer;
        uint32_t expert;
        uint32_t source_participant;
        uint32_t destination_participant;
        uint32_t source_resident_mask;
        uint32_t flags;
        uint32_t destination_slot;
        uint32_t payload_slot;
    };

    struct DeviceMoERebalanceCommandBufferHeaderView
    {
        uint32_t magic;
        uint32_t version;
        uint32_t epoch;
        uint32_t phase;
        uint32_t command_count;
        uint32_t command_capacity;
        uint32_t participant_id;
        uint32_t participant_count;
    };

    struct DeviceMoERebalanceWaveStateView
    {
        uint32_t magic;
        uint32_t version;
        uint32_t epoch;
        uint32_t next_start_layer;
        uint32_t planned_start_layer;
        uint32_t planned_layer_count;
        uint32_t command_capacity;
        uint32_t participant_id;
        uint32_t participant_count;
        uint32_t requested_payload_slots;
        uint32_t payload_bucket_slots;
        uint32_t payload_bucket_index;
        uint32_t payload_bucket_overflow;
        uint32_t reserved[3];
    };

    struct DeviceMoERebalanceWaveProgressView
    {
        uint32_t magic;
        uint32_t version;
        uint32_t epoch;
        uint32_t state;
        uint32_t planned_start_layer;
        uint32_t planned_layer_count;
        uint32_t command_count;
        uint32_t copied_arrivals;
        uint32_t applied_arrivals;
        uint32_t applied_layer_count;
        uint32_t error_code;
        uint32_t requested_payload_slots;
        uint32_t payload_bucket_slots;
        uint32_t payload_bucket_index;
        uint32_t payload_bucket_overflow;
        uint32_t reserved[1];
    };

    struct DeviceMoERebalanceGraphControllerStateView
    {
        uint32_t magic;
        uint32_t version;
        uint32_t participant_id;
        uint32_t participant_count;
        uint32_t next_epoch;
        uint32_t active_wave;
        uint32_t wave_count;
        uint32_t maintenance_launches;
        uint32_t decode_apply_polls;
        uint32_t decode_apply_hits;
        uint32_t last_error_code;
        uint32_t reserved[5];
        DeviceMoERebalanceWaveProgressView waves[2];
    };

    struct DeviceMoEExpertDirectoryEntryView
    {
        DeviceMoEExpertDescriptorView descriptor;
        uint32_t layer;
        uint32_t expert;
        uint32_t participant;
        uint32_t resident_mask;
        uint32_t epoch;
        uint32_t flags;
        uint32_t slot_index;
        uint32_t generation;
        uint8_t payload_bytes_per_block;
        uint8_t is_asymmetric;
        uint8_t has_emins;
        uint8_t reserved_u8;
        uint32_t reserved;
    };

    struct DeviceMoERebalanceApplyStatusView
    {
        uint32_t magic;
        uint32_t version;
        uint32_t status_code;
        uint32_t plan_entries_seen;
        uint32_t applied_arrivals;
        uint32_t skipped_wrong_destination;
        uint32_t invalid_plan_entries;
        uint32_t missing_source_descriptors;
        uint32_t missing_destination_slots;
        uint32_t descriptor_mismatches;
        uint32_t changed_layers;
        uint32_t copied_arrivals;
        uint32_t copy_incomplete;
        uint32_t post_apply_multi_resident_experts;
        uint32_t required_local_arrivals;
        uint32_t ready_local_arrivals;
    };

    __device__ __forceinline__ bool prefill_llep_apply_status_complete(
        const DeviceMoERebalanceApplyStatusView *__restrict__ status,
        uint64_t expected_transfer_count)
    {
        if (!status ||
            status->magic != kDeviceMoERebalanceMagic ||
            status->version != kDeviceMoERebalanceVersion ||
            status->status_code != 0u ||
            expected_transfer_count > 0xffffffffULL)
        {
            return false;
        }

        const uint32_t transfer_count =
            static_cast<uint32_t>(expected_transfer_count);
        if (transfer_count > 0u &&
            (status->plan_entries_seen < transfer_count ||
             status->required_local_arrivals > transfer_count ||
             status->ready_local_arrivals > transfer_count))
        {
            return false;
        }

        return status->required_local_arrivals == status->ready_local_arrivals &&
               status->invalid_plan_entries == 0u &&
               status->missing_source_descriptors == 0u &&
               status->missing_destination_slots == 0u &&
               status->descriptor_mismatches == 0u &&
               status->copy_incomplete == 0u;
    }

    __device__ __forceinline__ bool rebalance_config_ok(
        const DeviceMoERebalanceConfigView &config)
    {
        return config.magic == kDeviceMoERebalanceMagic &&
               config.version == kDeviceMoERebalanceVersion &&
               config.num_layers > 0u &&
               config.num_experts > 0u &&
               config.num_experts <= static_cast<uint32_t>(kDeviceMoEMaxExperts) &&
               config.top_k > 0u &&
               config.top_k <= static_cast<uint32_t>(kDeviceMoEMaxTopK) &&
               config.participant_count > 0u &&
               config.participant_count <= static_cast<uint32_t>(kDeviceMoEMaxParticipants) &&
               config.participant_id < config.participant_count &&
               llaminar2::moe_rebalance_policy::hasValidRootParticipant(config) &&
               config.window_size_tokens > 0u &&
               (config.layer_window_count == 0u ||
                config.layer_window_start < config.num_layers);
    }

    __device__ __forceinline__ uint32_t rebalance_command_buffer_count(
        uint32_t requested)
    {
        if (requested <= 1u)
            return 1u;
        return min(requested, 2u);
    }

    __device__ __forceinline__ bool rebalance_graph_controller_state_basic_ok(
        const DeviceMoERebalanceGraphControllerStateView *state,
        const DeviceMoERebalanceConfigView &config)
    {
        return state &&
               state->magic == kDeviceMoERebalanceMagic &&
               state->version == kDeviceMoERebalanceVersion &&
               state->participant_id == config.participant_id &&
               state->participant_count == config.participant_count &&
               state->wave_count == 2u;
    }

    __device__ __forceinline__ uint32_t rebalance_active_command_wave_index(
        const DeviceMoERebalanceGraphControllerStateView *state,
        const DeviceMoERebalanceConfigView &config,
        uint32_t command_buffer_count)
    {
        const uint32_t count = rebalance_command_buffer_count(command_buffer_count);
        if (count == 1u)
            return 0u;
        if (!rebalance_graph_controller_state_basic_ok(state, config))
            return 0u;
        return state->active_wave % count;
    }

    __device__ __forceinline__ bool rebalance_active_wave_busy_for_new_plan(
        const DeviceMoERebalanceGraphControllerStateView *state,
        const DeviceMoERebalanceConfigView &config,
        uint32_t command_wave_index)
    {
        if (!rebalance_graph_controller_state_basic_ok(state, config) ||
            command_wave_index >= state->wave_count)
        {
            return false;
        }
        const auto &wave = state->waves[command_wave_index];
        if (wave.magic != kDeviceMoERebalanceMagic ||
            wave.version != kDeviceMoERebalanceVersion ||
            wave.epoch == 0u)
        {
            return false;
        }
        return wave.state == kDeviceMoERebalanceLifecyclePlanning ||
               wave.state == kDeviceMoERebalanceLifecycleTransferInFlight ||
               wave.state == kDeviceMoERebalanceLifecycleReadyToApply ||
               wave.state == kDeviceMoERebalanceLifecycleApplying;
    }

    __device__ __forceinline__ void reset_rebalance_wave_progress_device(
        DeviceMoERebalanceWaveProgressView &wave)
    {
        DeviceMoERebalanceWaveProgressView zero{};
        wave = zero;
        wave.magic = kDeviceMoERebalanceMagic;
        wave.version = kDeviceMoERebalanceVersion;
        wave.state = kDeviceMoERebalanceLifecycleIdle;
    }

    __device__ __forceinline__ void clear_rebalance_command_header_device(
        DeviceMoERebalanceCommandBufferHeaderView *header)
    {
        if (!header)
            return;
        const uint32_t command_capacity = header->command_capacity;
        const uint32_t participant_id = header->participant_id;
        const uint32_t participant_count = header->participant_count;
        header->magic = kDeviceMoERebalanceMagic;
        header->version = kDeviceMoERebalanceVersion;
        header->epoch = 0u;
        header->phase = kDeviceMoERebalancePhasePlanAssignments;
        header->command_count = 0u;
        header->command_capacity = command_capacity;
        header->participant_id = participant_id;
        header->participant_count = participant_count;
    }

    __device__ __forceinline__ uint32_t rebalance_layer_window_count(
        const DeviceMoERebalanceConfigView &config)
    {
        return config.layer_window_count == 0u
                   ? config.num_layers
                   : ((config.layer_window_count < config.num_layers)
                          ? config.layer_window_count
                          : config.num_layers);
    }

    __device__ __forceinline__ uint32_t rebalance_layer_wave_count(
        const DeviceMoERebalanceConfigView &config)
    {
        const uint32_t window_count = rebalance_layer_window_count(config);
        return config.layer_wave_count == 0u
                   ? window_count
                   : ((config.layer_wave_count < window_count)
                          ? config.layer_wave_count
                          : window_count);
    }

    __device__ __forceinline__ uint32_t rebalance_wave_start_offset(
        const DeviceMoERebalanceConfigView &config,
        const DeviceMoERebalanceWaveStateView *wave_state)
    {
        const uint32_t window_count = rebalance_layer_window_count(config);
        if (!wave_state ||
            window_count == 0u ||
            config.num_layers == 0u ||
            wave_state->magic != kDeviceMoERebalanceMagic ||
            wave_state->version != kDeviceMoERebalanceVersion ||
            wave_state->participant_id != config.participant_id ||
            wave_state->participant_count != config.participant_count)
        {
            return 0u;
        }

        const uint32_t layer_window_start = config.layer_window_start % config.num_layers;
        const uint32_t candidate = wave_state->next_start_layer % config.num_layers;
        const uint32_t candidate_offset =
            (candidate + config.num_layers - layer_window_start) % config.num_layers;
        return candidate_offset < window_count ? candidate_offset : 0u;
    }

    __device__ __forceinline__ uint32_t rebalance_actual_layer_for_wave_index(
        const DeviceMoERebalanceConfigView &config,
        const DeviceMoERebalanceWaveStateView *wave_state,
        uint32_t wave_layer)
    {
        const uint32_t window_count = rebalance_layer_window_count(config);
        if (window_count == 0u || config.num_layers == 0u)
            return 0u;
        const uint32_t layer_window_start = config.layer_window_start % config.num_layers;
        const uint32_t start_offset = rebalance_wave_start_offset(config, wave_state);
        return (layer_window_start + ((start_offset + wave_layer) % window_count)) %
               config.num_layers;
    }

    __device__ __forceinline__ unsigned long long rebalance_histogram_count(
        const unsigned long long *histograms,
        const DeviceMoERebalanceConfigView &config,
        uint32_t participant,
        uint32_t wave_layer,
        uint32_t expert)
    {
        const unsigned long long participant_stride =
            static_cast<unsigned long long>(rebalance_layer_wave_count(config)) *
            static_cast<unsigned long long>(config.num_experts);
        const unsigned long long layer_stride =
            static_cast<unsigned long long>(config.num_experts);
        return histograms[static_cast<unsigned long long>(participant) * participant_stride +
                          static_cast<unsigned long long>(wave_layer) * layer_stride +
                          static_cast<unsigned long long>(expert)];
    }

    __device__ __forceinline__ unsigned long long rebalance_global_count(
        const unsigned long long *histograms,
        const DeviceMoERebalanceConfigView &config,
        uint32_t wave_layer,
        uint32_t expert)
    {
        unsigned long long total = 0;
        for (uint32_t participant = 0; participant < config.participant_count; ++participant)
            total += rebalance_histogram_count(histograms, config, participant, wave_layer, expert);
        return total;
    }

    __device__ __forceinline__ uint32_t rebalance_clamp_u64_to_u32(
        unsigned long long value)
    {
        return value > 0xffffffffULL ? 0xffffffffu : static_cast<uint32_t>(value);
    }

    __device__ __forceinline__ unsigned long long rebalance_window_observed_slots(
        const unsigned long long *histograms,
        const DeviceMoERebalanceConfigView &config)
    {
        if (!histograms || config.num_layers == 0u || config.top_k == 0u ||
            config.participant_count == 0u)
        {
            return 0ULL;
        }
        const uint32_t wave_count = rebalance_layer_wave_count(config);
        if (wave_count == 0u)
            return 0ULL;
        unsigned long long last_layer_slots = 0;
        const uint32_t last_layer = wave_count - 1u;
        for (uint32_t participant = 0; participant < config.participant_count; ++participant)
            for (uint32_t expert = 0; expert < config.num_experts; ++expert)
                last_layer_slots += rebalance_histogram_count(
                    histograms, config, participant, last_layer, expert);
        return last_layer_slots;
    }

    __device__ __forceinline__ unsigned long long rebalance_window_required_slots(
        const DeviceMoERebalanceConfigView &config)
    {
        return static_cast<unsigned long long>(config.window_size_tokens) *
               static_cast<unsigned long long>(config.top_k);
    }

    __device__ __forceinline__ bool runtime_shape_ok(
        const DeviceMoELayerRuntimeView *runtime,
        int num_experts,
        int top_k)
    {
        return runtime &&
               runtime->active_bank <= 1u &&
               runtime->active_epoch != 0u &&
               runtime->expert_count == static_cast<uint32_t>(num_experts) &&
               runtime->top_k == static_cast<uint32_t>(top_k);
    }

    __device__ __forceinline__ int runtime_expert_owner(
        const DeviceMoEPlacementBankView &bank,
        int expert_id)
    {
        return bank.experts[expert_id].owner_participant;
    }

    __device__ __forceinline__ uint32_t runtime_participant_bit(int participant)
    {
        return llaminar2::moe_rebalance_policy::participantBit(
            static_cast<uint32_t>(participant));
    }

    __device__ __forceinline__ uint32_t runtime_valid_participant_mask(uint32_t participant_count)
    {
        return llaminar2::moe_rebalance_policy::validParticipantMask(participant_count);
    }

    __device__ __forceinline__ int rebalance_first_resident_participant(
        uint32_t resident_mask,
        uint32_t participant_count,
        int preferred_participant,
        int excluded_participant)
    {
        return llaminar2::moe_rebalance_policy::firstResidentParticipant(
            resident_mask,
            participant_count,
            preferred_participant,
            excluded_participant);
    }

    __device__ __forceinline__ bool rebalance_candidate_can_affect_local_compute(
        const DeviceMoEExpertDescriptorView &desc,
        uint32_t resident_mask,
        const DeviceMoERebalanceConfigView &config,
        uint32_t participant_bit,
        bool plan_missing_arrivals)
    {
        (void)participant_bit;
        return llaminar2::moe_rebalance_policy::candidateCanAffectLocalCompute(
            desc,
            resident_mask,
            config,
            plan_missing_arrivals);
    }

    __device__ __forceinline__ bool rebalance_candidate_can_affect_domain_compute(
        const DeviceMoEExpertDescriptorView &desc,
        uint32_t resident_mask,
        const DeviceMoERebalanceConfigView &config,
        bool plan_missing_arrivals)
    {
        return llaminar2::moe_rebalance_policy::candidateCanAffectDomainCompute(
            desc,
            resident_mask,
            config,
            plan_missing_arrivals);
    }

    __device__ __forceinline__ bool rebalance_append_plan_entry(
        DeviceMoERebalancePlanEntryView *plan_entries,
        uint32_t *plan_count,
        uint32_t plan_capacity,
        const DeviceMoERebalancePlanEntryView &entry,
        DeviceMoERebalanceStatusView *status)
    {
        if (!plan_entries || !plan_count)
            return false;
        const uint32_t index = *plan_count;
        if (index >= plan_capacity)
        {
            if (status)
                ++status->plan_overflow;
            return false;
        }
        plan_entries[index] = entry;
        *plan_count = index + 1u;
        if (status)
            ++status->planned_arrivals;
        return true;
    }

    __device__ __forceinline__ void rebalance_command_layer_span(
        const DeviceMoERebalancePlanEntryView *plan_entries,
        uint32_t command_count,
        const DeviceMoERebalanceConfigView &config,
        uint32_t layer_window_start,
        uint32_t start_offset,
        uint32_t layer_window_count,
        uint32_t layer_wave_count,
        uint32_t *planned_start_layer,
        uint32_t *planned_layer_count)
    {
        if (!planned_start_layer || !planned_layer_count ||
            !plan_entries || command_count == 0u ||
            config.num_layers == 0u ||
            layer_window_count == 0u ||
            layer_wave_count == 0u)
        {
            return;
        }

        uint32_t first_wave_offset = layer_wave_count;
        uint32_t last_wave_offset = 0u;
        for (uint32_t i = 0; i < command_count; ++i)
        {
            const uint32_t layer = plan_entries[i].layer % config.num_layers;
            const uint32_t window_offset =
                (layer + config.num_layers - layer_window_start) % config.num_layers;
            if (window_offset >= layer_window_count)
                continue;
            const uint32_t wave_offset =
                (window_offset + layer_window_count - start_offset) % layer_window_count;
            if (wave_offset >= layer_wave_count)
                continue;
            if (first_wave_offset > wave_offset)
                first_wave_offset = wave_offset;
            if (last_wave_offset < wave_offset)
                last_wave_offset = wave_offset;
        }

        if (first_wave_offset >= layer_wave_count ||
            last_wave_offset < first_wave_offset)
        {
            return;
        }

        *planned_start_layer =
            (layer_window_start + ((start_offset + first_wave_offset) % layer_window_count)) %
            config.num_layers;
        *planned_layer_count = last_wave_offset - first_wave_offset + 1u;
    }

    __device__ __forceinline__ uint32_t runtime_expert_resident_mask(
        const DeviceMoELayerRuntimeView *runtime,
        const DeviceMoEPlacementBankView &bank,
        int expert_id)
    {
        const uint32_t participant_count = runtime->participant_count;
        uint32_t mask = bank.resident_participant_mask[expert_id] &
                        runtime_valid_participant_mask(participant_count);
        if (bank.local_compute_mask[expert_id] != 0u &&
            runtime->participant_id < participant_count)
        {
            mask |= runtime_participant_bit(static_cast<int>(runtime->participant_id));
        }
        return mask;
    }

    __device__ __forceinline__ int runtime_resident_count(
        uint32_t resident_mask,
        uint32_t participant_count)
    {
        int count = 0;
        for (int participant = 0; participant < static_cast<int>(participant_count); ++participant)
        {
            if ((resident_mask & runtime_participant_bit(participant)) != 0u)
                ++count;
        }
        return count;
    }

    __device__ __forceinline__ bool runtime_expert_replicated(
        const DeviceMoELayerRuntimeView *runtime,
        const DeviceMoEPlacementBankView &bank,
        int expert_id)
    {
        if (!runtime)
            return false;
        const uint32_t resident_mask =
            runtime_expert_resident_mask(runtime, bank, expert_id);
        return runtime_resident_count(resident_mask, runtime->participant_count) > 1;
    }

    __device__ __forceinline__ uint32_t runtime_multi_resident_expert_count(
        const DeviceMoEPlacementBankView &bank)
    {
        return bank.reserved[0];
    }

    __device__ __forceinline__ uint32_t runtime_compute_multi_resident_expert_count(
        const DeviceMoEPlacementBankView &bank,
        uint32_t expert_count,
        uint32_t participant_count)
    {
        uint32_t count = 0u;
        const uint32_t valid_mask = runtime_valid_participant_mask(participant_count);
        for (uint32_t expert = 0; expert < expert_count; ++expert)
        {
            if (runtime_resident_count(bank.resident_participant_mask[expert] & valid_mask,
                                       participant_count) > 1)
            {
                ++count;
            }
        }
        return count;
    }

    __device__ __forceinline__ bool runtime_has_local_hot_replica(
        const DeviceMoELayerRuntimeView *runtime,
        const DeviceMoEPlacementBankView &bank,
        uint32_t expert_count)
    {
        for (uint32_t expert = 0; expert < expert_count; ++expert)
        {
            const DeviceMoEExpertDescriptorView &desc = bank.experts[expert];
            const uint32_t resident_mask =
                runtime_expert_resident_mask(runtime, bank, static_cast<int>(expert));
            const bool local_resident =
                (resident_mask & runtime_participant_bit(
                                     static_cast<int>(runtime->participant_id))) != 0u;
            const bool owner_local =
                desc.owner_participant == static_cast<int32_t>(runtime->participant_id);
            const bool multi_resident =
                (resident_mask & (resident_mask - 1u)) != 0u;
            if (local_resident && !owner_local && multi_resident)
                return true;
        }
        return false;
    }

    __device__ __forceinline__ int runtime_nth_resident_participant(
        uint32_t resident_mask,
        uint32_t participant_count,
        int ordinal)
    {
        for (int participant = 0; participant < static_cast<int>(participant_count); ++participant)
        {
            if ((resident_mask & runtime_participant_bit(participant)) == 0u)
                continue;
            if (ordinal == 0)
                return participant;
            --ordinal;
        }
        return -1;
    }

    __device__ __forceinline__ uint64_t runtime_same_expert_prior_occurrences(
        const int *selected_experts,
        int selected_slot,
        int expert_id)
    {
        uint64_t occurrences = 0;
        for (int slot = 0; slot < selected_slot; ++slot)
        {
            if (selected_experts[slot] == expert_id)
                ++occurrences;
        }
        return occurrences;
    }

    __device__ __forceinline__ int runtime_choose_replicated_participant(
        const DeviceMoELayerRuntimeView *runtime,
        const DeviceMoEPlacementBankView &bank,
        const int *selected_experts,
        int selected_slot,
        int expert_id,
        const int *load)
    {
        const uint32_t participant_count = runtime->participant_count;
        const uint32_t resident_mask =
            runtime_expert_resident_mask(runtime, bank, expert_id);
        const int resident_count = runtime_resident_count(resident_mask, participant_count);
        if (resident_count <= 0)
            return -1;

        const uint64_t turn =
            runtime->decode_histogram[expert_id] +
            runtime_same_expert_prior_occurrences(selected_experts, selected_slot, expert_id);
        const int preferred =
            runtime_nth_resident_participant(
                resident_mask,
                participant_count,
                static_cast<int>(turn % static_cast<uint64_t>(resident_count)));

        int best = -1;
        for (int participant = 0; participant < static_cast<int>(participant_count); ++participant)
        {
            if ((resident_mask & runtime_participant_bit(participant)) == 0u)
                continue;
            if (best < 0 ||
                load[participant] < load[best] ||
                (load[participant] == load[best] && participant == preferred))
            {
                best = participant;
            }
        }
        return best;
    }

    __device__ __forceinline__ uint64_t runtime_participant_load_spread(
        const int *load,
        uint32_t participant_count)
    {
        if (!load || participant_count == 0u)
            return 0ULL;
        int min_load = load[0];
        int max_load = load[0];
        for (uint32_t participant = 1; participant < participant_count; ++participant)
        {
            const int value = load[participant];
            if (value < min_load)
                min_load = value;
            if (value > max_load)
                max_load = value;
        }
        return static_cast<uint64_t>(max_load - min_load);
    }

    __device__ __forceinline__ void runtime_resolve_decode_dispatch(
        DeviceMoELayerRuntimeView *runtime,
        const int *selected_experts,
        int num_experts,
        int top_k,
        bool record_balance,
        bool *local_compute_flags)
    {
        if (!runtime_shape_ok(runtime, num_experts, top_k) ||
            !selected_experts ||
            !local_compute_flags)
        {
            return;
        }

        const auto &bank = runtime->banks[runtime->active_bank];
        const uint32_t participant_count = runtime->participant_count;
        const uint32_t participant_id = runtime->participant_id;
        const bool valid_participants =
            participant_count > 0u &&
            participant_count <= static_cast<uint32_t>(kDeviceMoEMaxParticipants) &&
            participant_id < participant_count;
        const bool has_multi_resident_experts =
            runtime_multi_resident_expert_count(bank) != 0u;
        if (!has_multi_resident_experts)
        {
            for (int slot = 0; slot < top_k; ++slot)
            {
                const int expert_id = selected_experts[slot];
                local_compute_flags[slot] =
                    expert_id >= 0 &&
                    expert_id < num_experts &&
                    bank.local_compute_mask[expert_id] != 0u;
            }
            return;
        }
        const bool track_balance =
            record_balance &&
            valid_participants;

        int default_load[kDeviceMoEMaxParticipants] = {};
        int actual_load[kDeviceMoEMaxParticipants] = {};
        bool eligible = false;
        bool used_remote_replica = false;
        uint64_t selected_expert_slots = 0ULL;
        uint64_t replicated_selected_expert_slots = 0ULL;
        if (track_balance)
            ++runtime->router_hot_cache_active_dispatches;

        for (int slot = 0; slot < top_k; ++slot)
        {
            const int expert_id = selected_experts[slot];
            if (expert_id < 0 || expert_id >= num_experts)
                continue;
            if (track_balance)
                ++selected_expert_slots;

            const bool local_resident = bank.local_compute_mask[expert_id] != 0u;
            const bool replicated = runtime_expert_replicated(runtime, bank, expert_id);
            const int owner = runtime_expert_owner(bank, expert_id);
            const bool owner_valid =
                valid_participants &&
                owner >= 0 &&
                owner < static_cast<int>(participant_count);

            if (track_balance && owner_valid)
                ++default_load[owner];

            if (!replicated || !valid_participants || !owner_valid)
            {
                local_compute_flags[slot] = local_resident;
                if (!replicated && owner_valid)
                    ++actual_load[owner];
                continue;
            }

            if (track_balance)
            {
                const uint32_t resident_mask =
                    runtime_expert_resident_mask(runtime, bank, expert_id);
                if (runtime_resident_count(resident_mask, participant_count) > 1)
                {
                    eligible = true;
                    ++replicated_selected_expert_slots;
                }
            }
        }

        if (track_balance)
        {
            runtime->router_hot_cache_selected_expert_slots += selected_expert_slots;
            runtime->router_hot_cache_replicated_selected_expert_slots +=
                replicated_selected_expert_slots;
            if (!eligible)
                ++runtime->router_hot_cache_miss_dispatches;
        }

        for (int slot = 0; slot < top_k; ++slot)
        {
            const int expert_id = selected_experts[slot];
            if (expert_id < 0 || expert_id >= num_experts ||
                !runtime_expert_replicated(runtime, bank, expert_id))
            {
                continue;
            }

            const bool local_resident = bank.local_compute_mask[expert_id] != 0u;
            const int owner = runtime_expert_owner(bank, expert_id);
            if (!valid_participants ||
                owner < 0 ||
                owner >= static_cast<int>(participant_count))
            {
                local_compute_flags[slot] = local_resident;
                continue;
            }

            const int selected_participant =
                runtime_choose_replicated_participant(
                    runtime, bank, selected_experts, slot, expert_id, actual_load);
            if (selected_participant >= 0 &&
                selected_participant < static_cast<int>(participant_count))
            {
                local_compute_flags[slot] =
                    local_resident &&
                    selected_participant == static_cast<int>(participant_id);
                ++actual_load[selected_participant];
                if (track_balance && selected_participant != owner)
                    used_remote_replica = true;
            }
            else if (track_balance)
            {
                ++actual_load[owner];
            }
        }

        if (!track_balance || !eligible)
            return;

        const uint64_t default_spread =
            runtime_participant_load_spread(default_load, participant_count);
        const uint64_t actual_spread =
            runtime_participant_load_spread(actual_load, participant_count);
        ++runtime->router_hot_cache_eligible_dispatches;
        runtime->router_hot_cache_default_load_spread_total += default_spread;
        runtime->router_hot_cache_actual_load_spread_total += actual_spread;
        if (!used_remote_replica)
            return;
        ++runtime->router_hot_cache_used_dispatches;
        if (actual_spread < default_spread)
        {
            ++runtime->router_hot_cache_improved_dispatches;
            runtime->router_hot_cache_load_spread_improvement_total +=
                default_spread - actual_spread;
        }
    }

    __device__ __forceinline__ void runtime_publish_decode_dispatch(
        DeviceMoELayerRuntimeView *runtime,
        const int *selected_ids,
        const float *selected_weights,
        int num_experts,
        int top_k,
        float *legacy_indices,
        float *legacy_weights,
        bool write_legacy_outputs,
        bool update_runtime_histogram)
    {
        const bool shape_ok = runtime_shape_ok(runtime, num_experts, top_k);
        bool local_compute_flags[kDeviceMoEMaxTopK] = {};
        if (shape_ok)
        {
            runtime_resolve_decode_dispatch(
                runtime,
                selected_ids,
                num_experts,
                top_k,
                update_runtime_histogram,
                local_compute_flags);
        }
        for (int slot = 0; slot < top_k; ++slot)
        {
            const int expert_id = selected_ids[slot];
            const float weight = selected_weights[slot];

            if (shape_ok)
            {
                const bool local_compute = local_compute_flags[slot];
                runtime->topk_expert_ids[slot] = local_compute ? expert_id : -1;
                runtime->topk_weights[slot] = local_compute ? weight : 0.0f;
            }

            if (write_legacy_outputs && legacy_indices && legacy_weights)
            {
                legacy_indices[slot] = static_cast<float>(expert_id);
                legacy_weights[slot] = weight;
            }
        }

        if (!update_runtime_histogram)
            return;

        for (int slot = 0; slot < top_k; ++slot)
        {
            const int expert_id = selected_ids[slot];
            if (expert_id < 0 || expert_id >= num_experts)
                continue;
            atomicAdd(reinterpret_cast<unsigned long long *>(&runtime->decode_histogram[expert_id]), 1ULL);
            if (shape_ok && local_compute_flags[slot])
            {
                atomicAdd(reinterpret_cast<unsigned long long *>(&runtime->decode_local_histogram[expert_id]), 1ULL);
            }
        }
    }

    __global__ void device_rebalance_controller_kernel(
        DeviceMoELayerRuntimeView *runtime_layers,
        const unsigned long long *gathered_histograms,
        DeviceMoERebalanceStatusView *status,
        DeviceMoERebalanceConfigView config,
        DeviceMoERebalancePlanEntryView *plan_entries,
        uint32_t *plan_count,
        uint32_t plan_capacity,
        uint32_t payload_slot_capacity,
        DeviceMoERebalanceCommandBufferHeaderView *command_header,
        DeviceMoERebalanceWaveStateView *wave_state,
        DeviceMoERebalanceGraphControllerStateView *controller_state,
        uint32_t command_buffer_count)
    {
        if (blockIdx.x != 0)
            return;

        const uint32_t lane = threadIdx.x;
        const bool leader = lane == 0u;
        __shared__ uint32_t shared_command_wave_index;
        __shared__ uint32_t shared_abort;
        __shared__ uint32_t shared_layer_invalid;
        __shared__ uint32_t shared_selected[kDeviceMoEMaxExperts];
        __shared__ uint32_t shared_selected_count;
        __shared__ uint32_t shared_local_replicas;
        __shared__ uint32_t shared_plan_epoch;
        __shared__ unsigned long long shared_candidate_values[kDeviceMoEMaxExperts];
        __shared__ unsigned long long shared_candidate_counts[kDeviceMoEMaxExperts];
        __shared__ unsigned long long shared_candidate_improvements[kDeviceMoEMaxExperts];
        __shared__ uint32_t shared_candidate_experts[kDeviceMoEMaxExperts];
        __shared__ uint32_t shared_candidate_arrival_considered[kDeviceMoEMaxExperts];
        __shared__ uint32_t shared_candidate_arrival_below_floor[kDeviceMoEMaxExperts];
        __shared__ uint32_t shared_stop_rank;
        __shared__ uint32_t shared_post_policy_resident_mask[kDeviceMoEMaxExperts];
        __shared__ uint32_t shared_physical_source_resident_mask[kDeviceMoEMaxExperts];
        __shared__ uint64_t shared_current_policy_load[kDeviceMoEMaxParticipants];
        __shared__ uint64_t shared_candidate_policy_load[kDeviceMoEMaxParticipants];
        __shared__ uint64_t shared_pre_policy_load[kDeviceMoEMaxParticipants];
        __shared__ uint64_t shared_post_policy_load[kDeviceMoEMaxParticipants];
        __shared__ uint64_t shared_owner_policy_load[kDeviceMoEMaxParticipants];
        __shared__ uint64_t shared_expert_counts[kDeviceMoEMaxExperts];
        __shared__ int32_t shared_expert_owners[kDeviceMoEMaxExperts];
        __shared__ uint64_t shared_required_load_spread_improvement;
        __shared__ uint32_t shared_count_bound_prunes;
        __shared__ uint32_t shared_destination_replica_counts[kDeviceMoEMaxParticipants];
        __shared__ uint32_t shared_destination_transfer_slot_counts[kDeviceMoEMaxParticipants];
        __shared__ uint32_t shared_source_payload_slot_counts[kDeviceMoEMaxParticipants];

        if (leader)
        {
            shared_command_wave_index =
                rebalance_active_command_wave_index(controller_state, config, command_buffer_count);
            shared_abort = 0u;
        }
        for (uint32_t participant = lane;
             participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants);
            participant += blockDim.x)
        {
            shared_current_policy_load[participant] = 0ULL;
            shared_candidate_policy_load[participant] = 0ULL;
            shared_pre_policy_load[participant] = 0ULL;
            shared_post_policy_load[participant] = 0ULL;
            shared_owner_policy_load[participant] = 0ULL;
            shared_destination_replica_counts[participant] = 0u;
            shared_destination_transfer_slot_counts[participant] = 0u;
            shared_source_payload_slot_counts[participant] = 0u;
        }
        for (int expert = static_cast<int>(lane);
             expert < kDeviceMoEMaxExperts;
             expert += static_cast<int>(blockDim.x))
        {
            shared_expert_counts[expert] = 0ULL;
            shared_expert_owners[expert] = -1;
        }
        __syncthreads();

        const uint32_t command_wave_index = shared_command_wave_index;
        if (plan_entries)
            plan_entries += static_cast<unsigned long long>(command_wave_index) * plan_capacity;
        if (plan_count)
            plan_count += command_wave_index;
        if (command_header)
            command_header += command_wave_index;
        if (wave_state)
            wave_state += command_wave_index;

        if (leader)
        {
            if (status)
            {
                DeviceMoERebalanceStatusView zero_status{};
                *status = zero_status;
                status->magic = kDeviceMoERebalanceMagic;
                status->version = kDeviceMoERebalanceVersion;
                status->status_code = kDeviceMoERebalanceStatusOk;
            }
            else
            {
                shared_abort = 1u;
            }

            if (shared_abort == 0u &&
                rebalance_active_wave_busy_for_new_plan(
                    controller_state,
                    config,
                    command_wave_index))
            {
                status->status_code = kDeviceMoERebalanceStatusWindowNotReady;
                status->skipped_not_ready = 1u;
                status->skipped_busy_wave = 1u;
                shared_abort = 1u;
            }

            if (plan_count)
                *plan_count = 0u;

            if (command_header)
            {
                command_header->magic = kDeviceMoERebalanceMagic;
                command_header->version = kDeviceMoERebalanceVersion;
                command_header->epoch = 0u;
                command_header->phase = kDeviceMoERebalancePhasePlanAssignments;
                command_header->command_count = 0u;
                command_header->command_capacity = plan_capacity;
                command_header->participant_id = config.participant_id;
                command_header->participant_count = config.participant_count;
            }

            if (wave_state)
            {
                const bool keep_cursor =
                    wave_state->magic == kDeviceMoERebalanceMagic &&
                    wave_state->version == kDeviceMoERebalanceVersion &&
                    wave_state->participant_id == config.participant_id &&
                    wave_state->participant_count == config.participant_count;
                const uint32_t next_start = keep_cursor ? wave_state->next_start_layer : 0u;
                wave_state->magic = kDeviceMoERebalanceMagic;
                wave_state->version = kDeviceMoERebalanceVersion;
                wave_state->epoch = 0u;
                wave_state->next_start_layer = next_start;
                wave_state->planned_start_layer = 0u;
                wave_state->planned_layer_count = 0u;
                wave_state->command_capacity = plan_capacity;
                wave_state->participant_id = config.participant_id;
                wave_state->participant_count = config.participant_count;
                wave_state->requested_payload_slots = 0u;
                wave_state->payload_bucket_slots = 0u;
                wave_state->payload_bucket_index = 0u;
                wave_state->payload_bucket_overflow = 0u;
            }

            if (shared_abort == 0u && !rebalance_config_ok(config))
            {
                status->status_code = kDeviceMoERebalanceStatusInvalidConfig;
                shared_abort = 1u;
            }
            if (shared_abort == 0u && !runtime_layers)
            {
                status->status_code = kDeviceMoERebalanceStatusInvalidRuntime;
                shared_abort = 1u;
            }
            if (shared_abort == 0u && !gathered_histograms)
            {
                status->status_code = kDeviceMoERebalanceStatusMissingHistogram;
                shared_abort = 1u;
            }
            if (shared_abort == 0u)
            {
                const unsigned long long observed_slots =
                    rebalance_window_observed_slots(gathered_histograms, config);
                const unsigned long long required_slots =
                    rebalance_window_required_slots(config);
                status->window_ready_slots =
                    rebalance_clamp_u64_to_u32(observed_slots);
                status->window_required_slots =
                    rebalance_clamp_u64_to_u32(required_slots);
                if (observed_slots < required_slots)
                {
                    status->status_code = kDeviceMoERebalanceStatusWindowNotReady;
                    status->skipped_not_ready = 1u;
                    status->skipped_busy_wave = 0u;
                    shared_abort = 1u;
                }
            }
        }
        __syncthreads();
        if (shared_abort != 0u)
            return;

        const uint32_t participant_bit = runtime_participant_bit(static_cast<int>(config.participant_id));
        const uint32_t valid_mask = runtime_valid_participant_mask(config.participant_count);
        const bool plan_missing_arrivals =
            (config.flags & kDeviceMoERebalanceFlagPlanMissingArrivals) != 0u;
        const bool defer_runtime_apply =
            (config.flags & kDeviceMoERebalanceFlagDeferRuntimeApply) != 0u;
        const bool domain_root_planning = defer_runtime_apply && plan_missing_arrivals;
        const bool least_loaded_assignment =
            config.routed_assignment_policy == kDeviceMoERebalanceAssignmentLeastLoadedEP;
        const bool hot_replica_cache =
            (config.flags & kDeviceMoERebalanceFlagHotReplicaCache) != 0u;
        const bool collect_load_stats =
            (config.flags & kDeviceMoERebalanceFlagCollectLoadStats) != 0u;
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
        uint32_t llep_assignment_span_count = 0u;
        uint32_t llep_weight_transfer_count = 0u;
        uint32_t llep_standard_ep_selected = 0u;
        uint32_t llep_skipped_balanced = 0u;
        uint32_t llep_skipped_insufficient_spread_improvement = 0u;
        uint32_t llep_skipped_insufficient_foreign_rows = 0u;
        uint32_t llep_min_chunk_skips = 0u;
        uint32_t llep_forced_spills = 0u;
        uint64_t candidate_load_spread_improvement_total = 0ULL;
        uint64_t candidate_load_spread_improvement_max = 0ULL;
        uint64_t accepted_load_spread_improvement_total = 0ULL;
        uint64_t accepted_load_spread_improvement_max = 0ULL;
        uint64_t llep_native_rows = 0ULL;
        uint64_t llep_spilled_rows = 0ULL;
        uint64_t llep_required_spread_improvement = 0ULL;
        uint64_t llep_required_foreign_rows = 0ULL;
        uint64_t router_hot_cache_eligible_dispatches = 0ULL;
        uint64_t router_hot_cache_used_dispatches = 0ULL;
        uint64_t router_hot_cache_improved_dispatches = 0ULL;
        uint64_t router_hot_cache_default_load_spread_total = 0ULL;
        uint64_t router_hot_cache_actual_load_spread_total = 0ULL;
        uint64_t router_hot_cache_load_spread_improvement_total = 0ULL;
        uint64_t router_hot_cache_active_dispatches = 0ULL;
        uint64_t router_hot_cache_miss_dispatches = 0ULL;
        uint64_t router_hot_cache_selected_expert_slots = 0ULL;
        uint64_t router_hot_cache_replicated_selected_expert_slots = 0ULL;
        uint32_t hot_cache_active_layers = 0u;
        uint64_t pre_wave_load_total = 0ULL;
        uint64_t pre_wave_load_spread = 0ULL;
        uint64_t post_wave_load_total = 0ULL;
        uint64_t post_wave_load_spread = 0ULL;
        uint32_t invalid_layers = 0;
        uint32_t last_epoch = 0;
        const uint32_t layer_window_count =
            config.layer_window_count == 0u
                ? config.num_layers
                : ((config.layer_window_count < config.num_layers)
                       ? config.layer_window_count
                       : config.num_layers);
        const uint32_t layer_wave_count =
            config.layer_wave_count == 0u
                ? layer_window_count
                : ((config.layer_wave_count < layer_window_count)
                       ? config.layer_wave_count
                       : layer_window_count);
        const uint32_t layer_window_start =
            config.num_layers == 0u ? 0u : (config.layer_window_start % config.num_layers);
        uint32_t start_offset = 0u;
        if (wave_state && layer_window_count > 0u)
        {
            const uint32_t candidate = wave_state->next_start_layer % config.num_layers;
            const uint32_t candidate_offset =
                (candidate + config.num_layers - layer_window_start) % config.num_layers;
            if (candidate_offset < layer_window_count)
                start_offset = candidate_offset;
        }
        if (leader)
        {
            shared_plan_epoch =
                (defer_runtime_apply &&
                 rebalance_graph_controller_state_basic_ok(controller_state, config) &&
                 controller_state->next_epoch > 0u)
                    ? controller_state->next_epoch
                    : 1u;
        }
        __syncthreads();

        /*
         * Router-benefit counters describe the previously active replica set.
         * Aggregate them over the full routed layer window before planning the
         * next layer wave so rolling waves cannot advance past their own payback
         * signal.
         */
        if (leader)
        {
            for (uint32_t window_index = 0; window_index < layer_window_count; ++window_index)
            {
                const uint32_t layer =
                    (layer_window_start + window_index) % config.num_layers;
                DeviceMoELayerRuntimeView &runtime = runtime_layers[layer];
                const bool runtime_ok =
                    runtime.active_bank <= 1u &&
                    runtime.expert_count == config.num_experts &&
                    runtime.top_k == config.top_k &&
                    runtime.participant_id == config.participant_id &&
                    runtime.participant_count == config.participant_count;
                if (!runtime_ok)
                    continue;

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

                const DeviceMoEPlacementBankView &active = runtime.banks[runtime.active_bank];
                if (runtime_has_local_hot_replica(&runtime, active, config.num_experts))
                    ++hot_cache_active_layers;

                if ((config.flags & kDeviceMoERebalanceFlagResetHistograms) != 0u)
                {
                    runtime.router_hot_cache_eligible_dispatches = 0ULL;
                    runtime.router_hot_cache_used_dispatches = 0ULL;
                    runtime.router_hot_cache_improved_dispatches = 0ULL;
                    runtime.router_hot_cache_default_load_spread_total = 0ULL;
                    runtime.router_hot_cache_actual_load_spread_total = 0ULL;
                    runtime.router_hot_cache_load_spread_improvement_total = 0ULL;
                    runtime.router_hot_cache_active_dispatches = 0ULL;
                    runtime.router_hot_cache_miss_dispatches = 0ULL;
                    runtime.router_hot_cache_selected_expert_slots = 0ULL;
                    runtime.router_hot_cache_replicated_selected_expert_slots = 0ULL;
                }
            }
        }
        __syncthreads();

        /*
         * In deferred transfer-slot mode the root participant produces the
         * domain-wide assignment plan and project_rebalance_domain_commands()
         * later broadcasts that root plan into every participant's local apply
         * ABI. Non-root participants must still advance the rolling wave cursor
         * so their next histogram pack stays aligned with the root, but running
         * the full candidate-selection loop here is wasted work.
         */
        if (domain_root_planning && config.participant_id != config.root_participant)
        {
            if (leader)
            {
                if (status)
                {
                    status->status_code = kDeviceMoERebalanceStatusOk;
                    status->windows_observed = 1u;
                    status->windows_applied = 0u;
                    status->changed_layers = 0u;
                    status->last_epoch = 0u;
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
                }
                if (command_header)
                {
                    command_header->epoch = 0u;
                    command_header->command_count = 0u;
                }
                if (wave_state)
                {
                    const uint32_t start_layer =
                        (layer_window_start + start_offset) % config.num_layers;
                    wave_state->epoch = 0u;
                    wave_state->planned_start_layer = start_layer;
                    wave_state->planned_layer_count = layer_wave_count;
                    const uint32_t next_offset =
                        layer_window_count > 0u
                            ? ((start_offset + layer_wave_count) % layer_window_count)
                            : 0u;
                    wave_state->next_start_layer =
                        (layer_window_start + next_offset) % config.num_layers;
                    wave_state->requested_payload_slots = 0u;
                    wave_state->payload_bucket_slots = 0u;
                    wave_state->payload_bucket_index = 0u;
                    wave_state->payload_bucket_overflow = 0u;
                }
            }
            return;
        }

        for (uint32_t window_index = 0; window_index < layer_wave_count; ++window_index)
        {
            const uint32_t layer =
                (layer_window_start + ((start_offset + window_index) % layer_window_count)) %
                config.num_layers;
            DeviceMoELayerRuntimeView &runtime = runtime_layers[layer];
            if (leader)
            {
                shared_layer_invalid =
                    (runtime.active_bank > 1u ||
                     runtime.expert_count != config.num_experts ||
                     runtime.top_k != config.top_k ||
                     runtime.participant_id != config.participant_id ||
                     runtime.participant_count != config.participant_count)
                        ? 1u
                        : 0u;
                if (shared_layer_invalid != 0u)
                    ++invalid_layers;
            }
            __syncthreads();
            if (shared_layer_invalid != 0u)
                continue;

            uint32_t layer_command_count_before = 0u;
            if (leader && defer_runtime_apply && plan_count)
                layer_command_count_before =
                    *plan_count < plan_capacity ? *plan_count : plan_capacity;

            const uint32_t inactive_bank = 1u - runtime.active_bank;
            const DeviceMoEPlacementBankView &active = runtime.banks[runtime.active_bank];
            DeviceMoEPlacementBankView &next = runtime.banks[inactive_bank];
            if (leader)
            {
                next.epoch = runtime.active_epoch + 1u;
                next.expert_count = config.num_experts;
                next.reserved[0] = active.reserved[0];
                next.reserved[1] = active.reserved[1];
            }

            for (uint32_t expert = lane; expert < config.num_experts; expert += blockDim.x)
            {
                next.experts[expert] = active.experts[expert];
                DeviceMoEExpertDescriptorView &desc = next.experts[expert];
                uint32_t resident_mask =
                    active.resident_participant_mask[expert] & valid_mask;
                const bool owner_local =
                    desc.owner_participant == static_cast<int32_t>(config.participant_id);
                const bool local_resident = (resident_mask & participant_bit) != 0u;
                const bool multi_resident =
                    runtime_resident_count(resident_mask, config.participant_count) > 1;
                if (multi_resident)
                    desc.flags |= kDeviceMoEFlagReplicated;
                else
                    desc.flags &= ~kDeviceMoEFlagReplicated;
                if (local_resident)
                {
                    desc.flags |= kDeviceMoEFlagValid |
                                  kDeviceMoEFlagResident |
                                  kDeviceMoEFlagLocalCompute;
                }
                else
                {
                    desc.flags &= ~kDeviceMoEFlagLocalCompute;
                }
                next.local_compute_mask[expert] = local_resident ? 1u : 0u;
                next.replica_role[expert] =
                    local_resident
                        ? (owner_local
                               ? static_cast<uint8_t>(kDeviceMoEReplicaRolePrimary)
                               : static_cast<uint8_t>(kDeviceMoEReplicaRoleReplica))
                        : static_cast<uint8_t>(kDeviceMoEReplicaRoleNone);
                next.resident_participant_mask[expert] = resident_mask;
                /*
                 * Transfer commands are published after the controller has
                 * optimistically updated next.resident_participant_mask for
                 * this wave.  Keep a separate wave-start physical mask for
                 * choosing copy sources, otherwise a later command can try to
                 * copy from a participant that only becomes resident after an
                 * earlier command in the same wave.
                 */
                shared_physical_source_resident_mask[expert] = resident_mask;
                shared_post_policy_resident_mask[expert] =
                    next.resident_participant_mask[expert] & valid_mask;
            }
            __syncthreads();
            if (leader)
            {
                for (uint32_t participant = 0; participant < config.participant_count; ++participant)
                {
                    shared_current_policy_load[participant] = 0ULL;
                    shared_owner_policy_load[participant] = 0ULL;
                }
                for (uint32_t expert = 0; expert < config.num_experts; ++expert)
                {
                    const DeviceMoEExpertDescriptorView &desc = next.experts[expert];
                    uint32_t resident_mask =
                        shared_post_policy_resident_mask[expert] & valid_mask;
                    const unsigned long long count =
                        rebalance_global_count(gathered_histograms, config, window_index, expert);
                    shared_expert_counts[expert] = count;
                    shared_expert_owners[expert] =
                        desc.owner_participant >= 0 &&
                                desc.owner_participant < static_cast<int32_t>(config.participant_count)
                            ? desc.owner_participant
                            : -1;
                    if (shared_expert_owners[expert] >= 0)
                    {
                        shared_owner_policy_load[
                            static_cast<uint32_t>(shared_expert_owners[expert])] += count;
                    }
                    for (uint32_t participant = 0; participant < config.participant_count; ++participant)
                    {
                        shared_current_policy_load[participant] +=
                            llaminar2::moe_rebalance_policy::projectedParticipantLoadForExpert(
                                count,
                                resident_mask,
                                config.participant_count,
                                participant);
                    }
                }
                uint64_t layer_pre_total = 0ULL;
                uint64_t layer_pre_min = 0ULL;
                uint64_t layer_pre_max = 0ULL;
                llaminar2::moe_rebalance_policy::finalizeLoadSpread(
                    shared_current_policy_load,
                    config.participant_count,
                    layer_pre_total,
                    layer_pre_min,
                    layer_pre_max);
                for (uint32_t participant = 0; participant < config.participant_count; ++participant)
                    shared_pre_policy_load[participant] += shared_current_policy_load[participant];
                pre_wave_load_total += layer_pre_total;
                pre_wave_load_spread += layer_pre_max - layer_pre_min;
            }
            __syncthreads();

            if (leader &&
                least_loaded_assignment &&
                domain_root_planning &&
                plan_missing_arrivals &&
                payload_slot_capacity > 0u)
            {
                uint32_t owner_participants[kDeviceMoEMaxExperts];
                uint32_t resident_participant_masks[kDeviceMoEMaxExperts];
                for (uint32_t expert = 0; expert < config.num_experts; ++expert)
                {
                    uint32_t resident_mask =
                        shared_post_policy_resident_mask[expert] & valid_mask;
                    const int32_t owner = shared_expert_owners[expert];
                    if (owner >= 0 && owner < static_cast<int32_t>(config.participant_count))
                    {
                        owner_participants[expert] = static_cast<uint32_t>(owner);
                    }
                    else
                    {
                        const int first_resident = rebalance_first_resident_participant(
                            resident_mask,
                            config.participant_count,
                            -1,
                            -1);
                        owner_participants[expert] =
                            first_resident >= 0 ? static_cast<uint32_t>(first_resident) : 0u;
                    }
                    resident_participant_masks[expert] = resident_mask;
                }

                llaminar2::least_loaded_ep::LeastLoadedExpertWeightTransfer
                    weight_transfers[kDeviceMoEMaxExperts];
                llaminar2::least_loaded_ep::LeastLoadedExpertAssignmentConfig llep_config{};
                llep_config.expert_count = config.num_experts;
                llep_config.participant_count = config.participant_count;
                llep_config.alpha_numerator = config.llep_alpha_numerator;
                llep_config.alpha_denominator = config.llep_alpha_denominator;
                llep_config.lambda_numerator = config.llep_lambda_numerator;
                llep_config.lambda_denominator = config.llep_lambda_denominator;
                llep_config.min_spread_improvement =
                    config.min_load_spread_improvement;
                llep_config.min_spread_improvement_divisor =
                    config.min_load_spread_improvement_divisor;
                llep_config.min_spread_improvement_per_transfer =
                    config.min_wave_spread_improvement_per_payload_slot;
                llep_config.min_foreign_rows_per_transfer =
                    config.min_foreign_rows_per_transfer;
                llep_config.enable_balanced_skip =
                    config.llep_enable_balanced_skip != 0u;

                llaminar2::least_loaded_ep::LeastLoadedExpertAssignmentWorkspace workspace{};
                workspace.sorted_experts = shared_selected;
                workspace.pending_load = shared_owner_policy_load;
                workspace.assigned_load = shared_candidate_policy_load;

                llaminar2::least_loaded_ep::LeastLoadedExpertAssignmentStatus llep_status{};
                const bool planned_llep =
                    llaminar2::least_loaded_ep::planLeastLoadedExpertWeightTransfers(
                        shared_expert_counts,
                        owner_participants,
                        llep_config,
                        workspace,
                        weight_transfers,
                        kDeviceMoEMaxExperts,
                        &llep_status,
                        resident_participant_masks);
                if (!planned_llep || llep_status.overflow != 0u)
                {
                    if (status)
                        ++status->plan_overflow;
                }
                candidate_arrivals_considered += llep_status.weight_transfer_count;
                llep_assignment_span_count += llep_status.span_count;
                llep_weight_transfer_count += llep_status.weight_transfer_count;
                llep_standard_ep_selected += llep_status.standard_ep_selected;
                llep_skipped_balanced += llep_status.skipped_balanced;
                llep_skipped_insufficient_spread_improvement +=
                    llep_status.skipped_insufficient_spread_improvement;
                llep_skipped_insufficient_foreign_rows +=
                    llep_status.skipped_insufficient_foreign_rows;
                llep_min_chunk_skips += llep_status.min_chunk_skips;
                llep_forced_spills += llep_status.forced_spills;
                llep_native_rows += llep_status.native_rows;
                llep_spilled_rows += llep_status.spilled_rows;
                llep_required_spread_improvement +=
                    llep_status.required_spread_improvement;
                llep_required_foreign_rows += llep_status.required_foreign_rows;
                candidate_load_spread_improvement_total +=
                    llep_status.assigned_load_spread_improvement;
                if (llep_status.assigned_load_spread_improvement >
                    candidate_load_spread_improvement_max)
                {
                    candidate_load_spread_improvement_max =
                        llep_status.assigned_load_spread_improvement;
                }

                uint32_t accepted_transfers = 0u;
                if (planned_llep &&
                    llep_status.overflow == 0u &&
                    llep_status.weight_transfer_count > 0u)
                {
                    for (uint32_t transfer_index = 0;
                         transfer_index < llep_status.weight_transfer_count;
                         ++transfer_index)
                    {
                        const auto transfer = weight_transfers[transfer_index];
                        if (transfer.expert >= config.num_experts ||
                            transfer.source_participant >= config.participant_count ||
                            transfer.destination_participant >= config.participant_count ||
                            transfer.source_participant == transfer.destination_participant)
                        {
                            ++skipped_no_resident;
                            continue;
                        }

                        const uint32_t destination_bit =
                            runtime_participant_bit(static_cast<int>(transfer.destination_participant));
                        const uint32_t source_bit =
                            runtime_participant_bit(static_cast<int>(transfer.source_participant));
                        uint32_t resident_mask =
                            shared_post_policy_resident_mask[transfer.expert] & valid_mask;
                        const uint32_t physical_source_resident_mask =
                            shared_physical_source_resident_mask[transfer.expert] & valid_mask;
                        if ((physical_source_resident_mask & source_bit) == 0u)
                        {
                            /*
                             * The LLEP transfer planner can only name a useful
                             * source if that participant already owns resident
                             * bytes for this expert before this wave starts.
                             * Keep the command stream honest by rejecting
                             * owner-only or same-wave sources instead of
                             * manufacturing residency in the transfer metadata.
                             */
                            ++skipped_no_resident;
                            continue;
                        }
                        if ((resident_mask & destination_bit) != 0u)
                            continue;

                        if (transfer.destination_participant >=
                                static_cast<uint32_t>(kDeviceMoEMaxParticipants) ||
                            transfer.source_participant >=
                                static_cast<uint32_t>(kDeviceMoEMaxParticipants) ||
                            shared_destination_transfer_slot_counts[
                                transfer.destination_participant] >= payload_slot_capacity ||
                            shared_source_payload_slot_counts[
                                transfer.source_participant] >= payload_slot_capacity)
                        {
                            if (status)
                                ++status->plan_overflow;
                            break;
                        }

                        const bool ownership_transfer = !hot_replica_cache;
                        DeviceMoERebalancePlanEntryView entry{};
                        entry.op = ownership_transfer
                                       ? kDeviceMoERebalancePlanOwnershipTransfer
                                       : kDeviceMoERebalancePlanExpertPayloadArrival;
                        entry.layer = layer;
                        entry.expert = transfer.expert;
                        entry.source_participant = transfer.source_participant;
                        entry.destination_participant = transfer.destination_participant;
                        entry.source_resident_mask =
                            ownership_transfer ? physical_source_resident_mask : resident_mask;
                        entry.destination_slot =
                            shared_destination_transfer_slot_counts[
                                transfer.destination_participant];
                        entry.payload_slot =
                            shared_source_payload_slot_counts[
                                transfer.source_participant];
                        if (!rebalance_append_plan_entry(
                                plan_entries,
                                plan_count,
                                plan_capacity,
                                entry,
                                status))
                        {
                            break;
                        }

                        ++shared_destination_transfer_slot_counts[
                            transfer.destination_participant];
                        ++shared_source_payload_slot_counts[
                            transfer.source_participant];
                        resident_mask =
                            ownership_transfer
                                ? (destination_bit & valid_mask)
                                : ((resident_mask | destination_bit) & valid_mask);
                        shared_post_policy_resident_mask[transfer.expert] =
                            resident_mask;
                        next.resident_participant_mask[transfer.expert] =
                            resident_mask;
                        if (ownership_transfer)
                        {
                            next.experts[transfer.expert].owner_participant =
                                static_cast<int32_t>(transfer.destination_participant);
                            shared_expert_owners[transfer.expert] =
                                static_cast<int32_t>(transfer.destination_participant);
                        }
                        ++accepted_transfers;
                    }
                }

                if (accepted_transfers > 0u)
                {
                    uint64_t accepted_before_total = 0ULL;
                    uint64_t accepted_before_min = 0ULL;
                    uint64_t accepted_before_max = 0ULL;
                    llaminar2::moe_rebalance_policy::finalizeLoadSpread(
                        shared_current_policy_load,
                        config.participant_count,
                        accepted_before_total,
                        accepted_before_min,
                        accepted_before_max);
                    (void)accepted_before_total;

                    for (uint32_t participant = 0;
                         participant < config.participant_count;
                         ++participant)
                    {
                        shared_candidate_policy_load[participant] = 0ULL;
                    }
                    for (uint32_t expert = 0; expert < config.num_experts; ++expert)
                    {
                        uint32_t resident_mask =
                            shared_post_policy_resident_mask[expert] & valid_mask;
                        for (uint32_t participant = 0;
                             participant < config.participant_count;
                             ++participant)
                        {
                            shared_candidate_policy_load[participant] +=
                                llaminar2::moe_rebalance_policy::projectedParticipantLoadForExpert(
                                    shared_expert_counts[expert],
                                    resident_mask,
                                    config.participant_count,
                                    participant);
                        }
                    }

                    uint64_t accepted_after_total = 0ULL;
                    uint64_t accepted_after_min = 0ULL;
                    uint64_t accepted_after_max = 0ULL;
                    llaminar2::moe_rebalance_policy::finalizeLoadSpread(
                        shared_candidate_policy_load,
                        config.participant_count,
                        accepted_after_total,
                        accepted_after_min,
                        accepted_after_max);
                    (void)accepted_after_total;
                    const uint64_t accepted_before_spread =
                        accepted_before_max >= accepted_before_min
                            ? accepted_before_max - accepted_before_min
                            : 0ULL;
                    const uint64_t accepted_after_spread =
                        accepted_after_max >= accepted_after_min
                            ? accepted_after_max - accepted_after_min
                            : 0ULL;
                    const uint64_t accepted_improvement =
                        accepted_before_spread > accepted_after_spread
                            ? accepted_before_spread - accepted_after_spread
                            : 0ULL;

                    for (uint32_t participant = 0;
                         participant < config.participant_count;
                         ++participant)
                    {
                        shared_current_policy_load[participant] =
                            shared_candidate_policy_load[participant];
                    }
                    accepted_load_spread_improvement_total += accepted_improvement;
                    if (accepted_improvement > accepted_load_spread_improvement_max)
                    {
                        accepted_load_spread_improvement_max = accepted_improvement;
                    }
                    if (hot_replica_cache)
                        selected_replicas += accepted_transfers;
                }
                else if (llep_status.standard_ep_selected != 0u ||
                         llep_status.weight_transfer_count == 0u)
                {
                    ++skipped_no_improvement;
                }
            }
            __syncthreads();

            if (leader && !least_loaded_assignment && plan_missing_arrivals && payload_slot_capacity > 0u)
            {
                const uint32_t max_swaps =
                    config.dynamic_max_swaps_per_layer == 0u
                        ? 0u
                        : config.dynamic_max_swaps_per_layer;
                const uint32_t max_entries =
                    config.dynamic_max_plan_entries_per_wave == 0u
                        ? plan_capacity
                        : min(config.dynamic_max_plan_entries_per_wave, plan_capacity);
                for (uint32_t swap_iter = 0; swap_iter < max_swaps; ++swap_iter)
                {
                    const uint32_t command_count =
                        plan_count ? ((*plan_count < plan_capacity) ? *plan_count : plan_capacity) : 0u;
                    if (command_count + 2u > plan_capacity ||
                        command_count + 2u > max_entries)
                    {
                        if (status)
                            ++status->plan_overflow;
                        break;
                    }

                    ++dynamic_ownership_swap_attempts;
                    const auto swap_choice =
                        llaminar2::moe_rebalance_policy::bestDynamicOwnershipSwap(
                            shared_owner_policy_load,
                            shared_expert_counts,
                            shared_expert_owners,
                            config.num_experts,
                            config.participant_count,
                            config.dynamic_imbalance_threshold_per_mille,
                            config.dynamic_min_improvement_per_mille,
                            config.dynamic_min_window_activations);
                    if (!swap_choice.valid)
                    {
                        ++dynamic_ownership_swap_rejections;
                        break;
                    }
                    const uint32_t heavy_source = swap_choice.overloaded_participant;
                    const uint32_t heavy_destination = swap_choice.underloaded_participant;
                    const uint32_t light_source = swap_choice.underloaded_participant;
                    const uint32_t light_destination = swap_choice.overloaded_participant;
                    const uint32_t heavy_physical_source_mask =
                        shared_physical_source_resident_mask[swap_choice.heavy_expert] & valid_mask;
                    const uint32_t light_physical_source_mask =
                        shared_physical_source_resident_mask[swap_choice.light_expert] & valid_mask;
                    const uint32_t heavy_source_bit =
                        runtime_participant_bit(static_cast<int>(heavy_source));
                    const uint32_t light_source_bit =
                        runtime_participant_bit(static_cast<int>(light_source));
                    if (heavy_source >= static_cast<uint32_t>(kDeviceMoEMaxParticipants) ||
                        heavy_destination >= static_cast<uint32_t>(kDeviceMoEMaxParticipants) ||
                        light_source >= static_cast<uint32_t>(kDeviceMoEMaxParticipants) ||
                        light_destination >= static_cast<uint32_t>(kDeviceMoEMaxParticipants) ||
                        shared_source_payload_slot_counts[heavy_source] >= payload_slot_capacity ||
                        shared_source_payload_slot_counts[light_source] >= payload_slot_capacity ||
                        shared_destination_transfer_slot_counts[heavy_destination] >= payload_slot_capacity ||
                        shared_destination_transfer_slot_counts[light_destination] >= payload_slot_capacity)
                    {
                        if (status)
                            ++status->plan_overflow;
                        break;
                    }
                    if ((heavy_physical_source_mask & heavy_source_bit) == 0u ||
                        (light_physical_source_mask & light_source_bit) == 0u)
                    {
                        /*
                         * Ownership swaps still move full expert payloads through the
                         * compact transfer path.  A logical old owner or same-wave
                         * destination is not enough; each source participant must
                         * physically hold the payload before we publish the paired swap.
                         */
                        ++skipped_no_resident;
                        ++dynamic_ownership_swap_rejections;
                        break;
                    }
                    ++dynamic_ownership_swap_accepts;

                    DeviceMoERebalancePlanEntryView heavy_entry{};
                    heavy_entry.op = kDeviceMoERebalancePlanOwnershipTransfer;
                    heavy_entry.layer = layer;
                    heavy_entry.expert = swap_choice.heavy_expert;
                    heavy_entry.source_participant = heavy_source;
                    heavy_entry.destination_participant = heavy_destination;
                    heavy_entry.source_resident_mask = heavy_physical_source_mask;
                    heavy_entry.destination_slot =
                        shared_destination_transfer_slot_counts[heavy_destination];
                    heavy_entry.payload_slot =
                        shared_source_payload_slot_counts[heavy_source];

                    DeviceMoERebalancePlanEntryView light_entry{};
                    light_entry.op = kDeviceMoERebalancePlanOwnershipTransfer;
                    light_entry.layer = layer;
                    light_entry.expert = swap_choice.light_expert;
                    light_entry.source_participant = light_source;
                    light_entry.destination_participant = light_destination;
                    light_entry.source_resident_mask = light_physical_source_mask;
                    light_entry.destination_slot =
                        shared_destination_transfer_slot_counts[light_destination];
                    light_entry.payload_slot =
                        shared_source_payload_slot_counts[light_source];

                    if (!rebalance_append_plan_entry(
                            plan_entries,
                            plan_count,
                            plan_capacity,
                            heavy_entry,
                            status) ||
                        !rebalance_append_plan_entry(
                            plan_entries,
                            plan_count,
                            plan_capacity,
                            light_entry,
                            status))
                    {
                        break;
                    }

                    ++shared_source_payload_slot_counts[heavy_source];
                    ++shared_source_payload_slot_counts[light_source];
                    ++shared_destination_transfer_slot_counts[heavy_destination];
                    ++shared_destination_transfer_slot_counts[light_destination];

                    next.experts[swap_choice.heavy_expert].owner_participant =
                        static_cast<int32_t>(heavy_destination);
                    next.experts[swap_choice.light_expert].owner_participant =
                        static_cast<int32_t>(light_destination);
                    shared_post_policy_resident_mask[swap_choice.heavy_expert] =
                        runtime_participant_bit(static_cast<int>(heavy_destination));
                    shared_post_policy_resident_mask[swap_choice.light_expert] =
                        runtime_participant_bit(static_cast<int>(light_destination));
                    next.resident_participant_mask[swap_choice.heavy_expert] =
                        shared_post_policy_resident_mask[swap_choice.heavy_expert];
                    next.resident_participant_mask[swap_choice.light_expert] =
                        shared_post_policy_resident_mask[swap_choice.light_expert];

                    llaminar2::moe_rebalance_policy::applyDynamicOwnershipSwap(
                        shared_owner_policy_load,
                        shared_expert_owners,
                        swap_choice);
                    for (uint32_t participant = 0; participant < config.participant_count; ++participant)
                        shared_current_policy_load[participant] = shared_owner_policy_load[participant];

                    accepted_load_spread_improvement_total += swap_choice.improvement;
                    if (swap_choice.improvement > accepted_load_spread_improvement_max)
                        accepted_load_spread_improvement_max = swap_choice.improvement;
                }
            }
            __syncthreads();

            const uint32_t max_local_replicas =
                (config.flags & kDeviceMoERebalanceFlagHotReplicaCache) != 0u
                    ? config.max_hot_replicas_per_participant
                    : 0u;
            const uint32_t max_selected_replicas =
                domain_root_planning
                    ? max_local_replicas * config.participant_count
                    : max_local_replicas;
            if (leader)
            {
                shared_selected_count = 0u;
                shared_local_replicas = 0u;
                for (uint32_t participant = 0; participant < config.participant_count; ++participant)
                    shared_destination_replica_counts[participant] = 0u;
            }
            __syncthreads();

            for (uint32_t rank = 0; !least_loaded_assignment && rank < config.num_experts; ++rank)
            {
                if (shared_local_replicas >= max_selected_replicas)
                    break;
                if (leader)
                {
                    shared_stop_rank = 0u;
                    shared_count_bound_prunes = 0u;
                    uint64_t current_policy_total = 0ULL;
                    for (uint32_t participant = 0; participant < config.participant_count; ++participant)
                        current_policy_total += shared_current_policy_load[participant];
                    shared_required_load_spread_improvement =
                        llaminar2::moe_rebalance_policy::requiredLoadSpreadImprovement(
                            current_policy_total,
                            config.min_load_spread_improvement,
                            config.min_load_spread_improvement_divisor);
                    if (plan_count &&
                        llaminar2::moe_rebalance_policy::commandBufferFull(
                            *plan_count,
                            plan_capacity))
                    {
                        if (status)
                            ++status->plan_overflow;
                        shared_stop_rank = 1u;
                    }
                }
                __syncthreads();
                if (shared_stop_rank != 0u)
                    break;

                uint32_t candidate_expert = static_cast<uint32_t>(kDeviceMoEMaxExperts);
                unsigned long long candidate_value = 0ULL;
                unsigned long long candidate_count = 0ULL;
                unsigned long long candidate_improvement = 0ULL;
                uint32_t candidate_arrival_considered = 0u;
                uint32_t candidate_arrival_below_floor = 0u;
                if (lane < config.num_experts)
                {
                    bool already_selected = false;
                    for (uint32_t i = 0; i < shared_selected_count; ++i)
                    {
                        if (shared_selected[i] == lane)
                        {
                            already_selected = true;
                            break;
                        }
                    }
                    if (!already_selected)
                    {
                        const DeviceMoEExpertDescriptorView &candidate_desc =
                            next.experts[lane];
                        uint32_t candidate_resident_mask =
                            next.resident_participant_mask[lane] & valid_mask;
                        const uint32_t candidate_physical_source_mask =
                            shared_physical_source_resident_mask[lane] & valid_mask;
                        const bool candidate_can_affect =
                            domain_root_planning
                                ? rebalance_candidate_can_affect_domain_compute(
                                      candidate_desc,
                                      candidate_resident_mask,
                                      config,
                                      plan_missing_arrivals)
                                : rebalance_candidate_can_affect_local_compute(
                                      candidate_desc,
                                      candidate_resident_mask,
                                      config,
                                      participant_bit,
                                      plan_missing_arrivals);
                        if (candidate_can_affect)
                        {
                            const unsigned long long count =
                                rebalance_global_count(gathered_histograms, config, window_index, lane);
                            if (count > 0ULL)
                            {
                                const bool candidate_local_resident =
                                    (candidate_resident_mask & participant_bit) != 0u;
                                const bool candidate_owner_local =
                                    candidate_desc.owner_participant ==
                                    static_cast<int32_t>(config.participant_id);
                                if (domain_root_planning)
                                {
                                    if (shared_required_load_spread_improvement != 0ULL &&
                                        count < shared_required_load_spread_improvement)
                                    {
                                        atomicAdd(&shared_count_bound_prunes, 1u);
                                    }
                                    else
                                    {
                                        const auto destination_choice =
                                            llaminar2::moe_rebalance_policy::bestDynamicMissingResidentDestination(
                                                shared_current_policy_load,
                                                count,
                                                candidate_resident_mask,
                                                config.participant_count,
                                                candidate_desc.owner_participant,
                                                shared_destination_replica_counts,
                                                max_local_replicas,
                                                config.window_size_tokens,
                                                config.min_load_spread_improvement,
                                                config.min_load_spread_improvement_divisor);
                                        candidate_arrival_considered = 1u;
                                        candidate_improvement = destination_choice.delta.improvement;
                                        const int source_participant =
                                            destination_choice.valid
                                                ? rebalance_first_resident_participant(
                                                      candidate_physical_source_mask,
                                                      config.participant_count,
                                                      candidate_desc.owner_participant,
                                                      static_cast<int>(destination_choice.destination_participant))
                                                : -1;
                                        const bool destination_has_transfer_slot =
                                            destination_choice.valid &&
                                            destination_choice.destination_participant < kDeviceMoEMaxParticipants &&
                                            shared_destination_transfer_slot_counts[
                                                destination_choice.destination_participant] < payload_slot_capacity;
                                        const bool source_has_payload_slot =
                                            source_participant >= 0 &&
                                            static_cast<uint32_t>(source_participant) < kDeviceMoEMaxParticipants &&
                                            shared_source_payload_slot_counts[
                                                static_cast<uint32_t>(source_participant)] < payload_slot_capacity;
                                        const bool source_is_physically_resident =
                                            source_participant >= 0 &&
                                            static_cast<uint32_t>(source_participant) < config.participant_count &&
                                            (candidate_physical_source_mask &
                                             runtime_participant_bit(source_participant)) != 0u;
                                        const auto source_delta =
                                            source_is_physically_resident
                                                ? llaminar2::moe_rebalance_policy::evaluateAddingResidentDynamicSpread(
                                                      shared_current_policy_load,
                                                      count,
                                                      candidate_resident_mask,
                                                      destination_choice.proposed_resident_mask,
                                                      config.participant_count,
                                                      static_cast<uint32_t>(source_participant),
                                                      destination_choice.destination_participant,
                                                      config.window_size_tokens,
                                                      config.min_load_spread_improvement,
                                                      config.min_load_spread_improvement_divisor)
                                                : llaminar2::moe_rebalance_policy::LoadSpreadDelta{};
                                        candidate_improvement = source_delta.improvement;
                                        if (destination_choice.valid &&
                                            destination_has_transfer_slot &&
                                            source_has_payload_slot &&
                                            source_is_physically_resident &&
                                            source_delta.meets_floor)
                                        {
                                            candidate_value = source_delta.improvement;
                                            candidate_count = count;
                                            candidate_expert = lane;
                                        }
                                        else if ((candidate_resident_mask & (candidate_resident_mask - 1u)) == 0u)
                                        {
                                            candidate_arrival_below_floor = 1u;
                                        }
                                        else
                                        {
                                            candidate_value = count;
                                            candidate_count = count;
                                            candidate_expert = lane;
                                        }
                                    }
                                }
                                else if (plan_missing_arrivals &&
                                    !candidate_local_resident &&
                                    !candidate_owner_local)
                                {
                                    if (shared_required_load_spread_improvement != 0ULL &&
                                        count < shared_required_load_spread_improvement)
                                    {
                                        atomicAdd(&shared_count_bound_prunes, 1u);
                                    }
                                    else
                                    {
                                        const int source_participant =
                                            rebalance_first_resident_participant(
                                                candidate_resident_mask,
                                                config.participant_count,
                                                candidate_desc.owner_participant,
                                                static_cast<int>(config.participant_id));
                                        const uint32_t proposed_resident_mask =
                                            (candidate_resident_mask | participant_bit) & valid_mask;
	                                        const auto delta =
	                                            source_participant >= 0
	                                                ? llaminar2::moe_rebalance_policy::evaluateAddingResidentDynamicSpread(
	                                                      shared_current_policy_load,
	                                                      count,
	                                                      candidate_resident_mask,
	                                                      proposed_resident_mask,
	                                                      config.participant_count,
	                                                      static_cast<uint32_t>(source_participant),
	                                                      config.participant_id,
	                                                      config.window_size_tokens,
	                                                      config.min_load_spread_improvement,
	                                                      config.min_load_spread_improvement_divisor)
	                                                : llaminar2::moe_rebalance_policy::LoadSpreadDelta{};
                                        candidate_arrival_considered = 1u;
                                        candidate_improvement = delta.improvement;
                                        if (!delta.meets_floor)
                                        {
                                            candidate_arrival_below_floor = 1u;
                                        }
                                        else
                                        {
                                            const bool source_has_payload_slot =
                                                source_participant >= 0 &&
                                                static_cast<uint32_t>(source_participant) < kDeviceMoEMaxParticipants &&
                                                shared_source_payload_slot_counts[
                                                    static_cast<uint32_t>(source_participant)] < payload_slot_capacity;
                                            if (config.participant_id < kDeviceMoEMaxParticipants &&
                                                shared_destination_transfer_slot_counts[config.participant_id] <
                                                    payload_slot_capacity &&
                                                source_has_payload_slot)
                                            {
                                                candidate_value = delta.improvement;
                                                candidate_count = count;
                                                candidate_expert = lane;
                                            }
                                        }
                                    }
                                }
                                else
                                {
                                    candidate_value = count;
                                    candidate_count = count;
                                    candidate_expert = lane;
                                }
                            }
                        }
                    }
                }

                shared_candidate_values[lane] = candidate_value;
                shared_candidate_counts[lane] = candidate_count;
                shared_candidate_improvements[lane] = candidate_improvement;
                shared_candidate_experts[lane] = candidate_expert;
                shared_candidate_arrival_considered[lane] = candidate_arrival_considered;
                shared_candidate_arrival_below_floor[lane] = candidate_arrival_below_floor;
                __syncthreads();

                if (leader)
                {
                    for (uint32_t expert = 0; expert < config.num_experts; ++expert)
                    {
                        candidate_arrivals_considered +=
                            shared_candidate_arrival_considered[expert];
                        candidate_arrivals_below_floor +=
                            shared_candidate_arrival_below_floor[expert];
                        skipped_no_improvement +=
                            shared_candidate_arrival_below_floor[expert];
                        candidate_load_spread_improvement_total +=
                            shared_candidate_improvements[expert];
                        if (shared_candidate_improvements[expert] >
                            candidate_load_spread_improvement_max)
                        {
                            candidate_load_spread_improvement_max =
                                shared_candidate_improvements[expert];
                        }
                    }
                }
                __syncthreads();

                for (uint32_t stride = kDeviceMoEMaxExperts / 2u; stride > 0u; stride >>= 1u)
                {
                    if (lane < stride)
                    {
                        const uint32_t other = lane + stride;
                        const unsigned long long lhs_value = shared_candidate_values[lane];
                        const unsigned long long rhs_value = shared_candidate_values[other];
                        const unsigned long long lhs_count = shared_candidate_counts[lane];
                        const unsigned long long rhs_count = shared_candidate_counts[other];
                        const uint32_t lhs_expert = shared_candidate_experts[lane];
                        const uint32_t rhs_expert = shared_candidate_experts[other];
                        if (llaminar2::moe_rebalance_policy::candidateValueIsBetter(
                                rhs_value,
                                rhs_count,
                                rhs_expert,
                                lhs_value,
                                lhs_count,
                                lhs_expert))
                        {
                            shared_candidate_values[lane] = rhs_value;
                            shared_candidate_counts[lane] = rhs_count;
                            shared_candidate_experts[lane] = rhs_expert;
                        }
                    }
                    __syncthreads();
                }

                if (leader)
                {
                    const uint32_t best_expert = shared_candidate_experts[0];
                    const unsigned long long best_count = shared_candidate_counts[0];
                    shared_stop_rank =
                        (best_expert >= config.num_experts || best_count == 0ULL) ? 1u : 0u;
                    candidate_arrivals_pruned_by_count_bound += shared_count_bound_prunes;
                    skipped_no_improvement += shared_count_bound_prunes;
                    if (shared_stop_rank == 0u)
                        shared_selected[shared_selected_count++] = best_expert;
                }
                __syncthreads();
                if (shared_stop_rank != 0u)
                    break;

                if (leader)
                {
                    const uint32_t best_expert = shared_selected[shared_selected_count - 1u];
                    DeviceMoEExpertDescriptorView &desc = next.experts[best_expert];
                    uint32_t resident_mask = next.resident_participant_mask[best_expert] & valid_mask;
                    next.resident_participant_mask[best_expert] = resident_mask;

                    const bool replicated = (resident_mask & (resident_mask - 1u)) != 0u;
                    const bool local_resident = (resident_mask & participant_bit) != 0u;
                    const bool owner_local =
                        desc.owner_participant == static_cast<int32_t>(config.participant_id);
                    if (domain_root_planning)
                    {
                        const uint32_t physical_source_resident_mask =
                            shared_physical_source_resident_mask[best_expert] & valid_mask;
                        const unsigned long long count =
                            rebalance_global_count(gathered_histograms, config, window_index, best_expert);
                        const auto destination_choice =
                            llaminar2::moe_rebalance_policy::bestDynamicMissingResidentDestination(
                                shared_current_policy_load,
                                count,
                                resident_mask,
                                config.participant_count,
                                desc.owner_participant,
                                shared_destination_replica_counts,
                                max_local_replicas,
                                config.window_size_tokens,
                                config.min_load_spread_improvement,
                                config.min_load_spread_improvement_divisor);
                        if (destination_choice.valid)
                        {
                            const int source_participant =
                                rebalance_first_resident_participant(
                                    physical_source_resident_mask,
                                    config.participant_count,
                                    desc.owner_participant,
                                    static_cast<int>(destination_choice.destination_participant));
                            const bool source_is_physically_resident =
                                source_participant >= 0 &&
                                static_cast<uint32_t>(source_participant) < config.participant_count &&
                                (physical_source_resident_mask &
                                 runtime_participant_bit(source_participant)) != 0u;
                            if (!source_is_physically_resident)
                            {
                                /*
                                 * A transfer command is only valid when the chosen source
                                 * already owns resident bytes in this runtime bank.  Static
                                 * ownership can guide preference, but it cannot authorize a
                                 * compact payload copy by itself.
                                 */
                                ++skipped_no_resident;
                            }
                            else if (!llaminar2::moe_rebalance_policy::addingResidentImprovesDynamicSpread(
                                    shared_current_policy_load,
                                    count,
                                    resident_mask,
                                    destination_choice.proposed_resident_mask,
                                    config.participant_count,
                                    static_cast<uint32_t>(source_participant),
                                    destination_choice.destination_participant,
                                    shared_candidate_policy_load,
                                    config.window_size_tokens,
                                    config.min_load_spread_improvement,
                                    config.min_load_spread_improvement_divisor))
                            {
                                ++skipped_no_improvement;
                            }
                            else
                            {
                                uint32_t destination_slot = kDeviceMoEInvalidSlot;
                                uint32_t payload_slot = kDeviceMoEInvalidSlot;
                                if (destination_choice.destination_participant < kDeviceMoEMaxParticipants &&
                                    shared_destination_transfer_slot_counts[
                                        destination_choice.destination_participant] < payload_slot_capacity &&
                                    source_participant >= 0 &&
                                    static_cast<uint32_t>(source_participant) < kDeviceMoEMaxParticipants &&
                                    shared_source_payload_slot_counts[
                                        static_cast<uint32_t>(source_participant)] < payload_slot_capacity)
                                {
                                    destination_slot =
                                        shared_destination_transfer_slot_counts[
                                            destination_choice.destination_participant];
                                    payload_slot =
                                        shared_source_payload_slot_counts[
                                            static_cast<uint32_t>(source_participant)];
                                }
                                DeviceMoERebalancePlanEntryView entry{};
                                entry.op = kDeviceMoERebalancePlanExpertPayloadArrival;
                                entry.layer = layer;
                                entry.expert = best_expert;
                                entry.source_participant =
                                    source_participant >= 0 ? static_cast<uint32_t>(source_participant) : 0u;
                                entry.destination_participant =
                                    destination_choice.destination_participant;
                                entry.source_resident_mask = resident_mask;
                                entry.destination_slot = destination_slot;
                                entry.payload_slot = payload_slot;
                                if (destination_slot != kDeviceMoEInvalidSlot &&
                                    payload_slot != kDeviceMoEInvalidSlot &&
                                    source_participant >= 0 &&
                                    rebalance_append_plan_entry(
                                        plan_entries,
                                        plan_count,
                                        plan_capacity,
                                        entry,
                                        status))
                                {
                                    ++shared_destination_transfer_slot_counts[
                                        destination_choice.destination_participant];
                                    ++shared_source_payload_slot_counts[
                                        static_cast<uint32_t>(source_participant)];
                                    shared_post_policy_resident_mask[best_expert] =
                                        destination_choice.proposed_resident_mask;
                                    for (uint32_t participant = 0; participant < config.participant_count; ++participant)
                                        shared_current_policy_load[participant] =
                                            shared_candidate_policy_load[participant];
                                    accepted_load_spread_improvement_total +=
                                        destination_choice.delta.improvement;
                                    if (destination_choice.delta.improvement >
                                        accepted_load_spread_improvement_max)
                                    {
                                        accepted_load_spread_improvement_max =
                                            destination_choice.delta.improvement;
                                    }
                                    ++shared_destination_replica_counts[
                                        destination_choice.destination_participant];
                                    ++shared_local_replicas;
                                    ++selected_replicas;
                                }
                                else
                                {
                                    if (destination_slot == kDeviceMoEInvalidSlot && status)
                                        ++status->plan_overflow;
                                    ++skipped_no_resident;
                                }
                            }
                        }
                        else if (!replicated)
                        {
                            ++skipped_no_resident;
                        }
                        else
                        {
                            bool resident_plan_ready = false;
                            for (uint32_t participant = 0; participant < config.participant_count; ++participant)
                            {
                                if ((resident_mask & runtime_participant_bit(static_cast<int>(participant))) == 0u)
                                    continue;
                                DeviceMoERebalancePlanEntryView entry{};
                                entry.op = kDeviceMoERebalancePlanResidentExpertAssignment;
                                entry.layer = layer;
                                entry.expert = best_expert;
                                entry.source_participant =
                                    desc.owner_participant >= 0
                                        ? static_cast<uint32_t>(desc.owner_participant)
                                        : participant;
                                entry.destination_participant = participant;
                                entry.source_resident_mask = resident_mask;
                                entry.destination_slot = kDeviceMoEInvalidSlot;
                                entry.payload_slot = kDeviceMoEInvalidSlot;
                                if (rebalance_append_plan_entry(
                                        plan_entries,
                                        plan_count,
                                        plan_capacity,
                                        entry,
                                        status))
                                {
                                    resident_plan_ready = true;
                                    if (desc.owner_participant != static_cast<int32_t>(participant))
                                    {
                                        ++shared_destination_replica_counts[participant];
                                        ++shared_local_replicas;
                                    }
                                }
                            }
                            if (resident_plan_ready)
                            {
                                shared_post_policy_resident_mask[best_expert] =
                                    resident_mask & valid_mask;
                                ++selected_replicas;
                            }
                            else
                            {
                                ++skipped_no_resident;
                            }
                        }
                    }
                    else if (plan_missing_arrivals && !local_resident && !owner_local)
                    {
                        const uint32_t physical_source_resident_mask =
                            shared_physical_source_resident_mask[best_expert] & valid_mask;
                        const unsigned long long count =
                            rebalance_global_count(gathered_histograms, config, window_index, best_expert);
                        const int source_participant = rebalance_first_resident_participant(
                            physical_source_resident_mask,
                            config.participant_count,
                            desc.owner_participant,
                            static_cast<int>(config.participant_id));
                        const uint32_t proposed_resident_mask =
                            (resident_mask | participant_bit) & valid_mask;
	                        const auto delta =
	                            source_participant >= 0
	                                ? llaminar2::moe_rebalance_policy::evaluateAddingResidentDynamicSpread(
	                                      shared_current_policy_load,
	                                      count,
	                                      resident_mask,
	                                      proposed_resident_mask,
	                                      config.participant_count,
	                                      static_cast<uint32_t>(source_participant),
	                                      config.participant_id,
	                                      config.window_size_tokens,
	                                      config.min_load_spread_improvement,
	                                      config.min_load_spread_improvement_divisor)
	                                : llaminar2::moe_rebalance_policy::LoadSpreadDelta{};
                        if (!delta.meets_floor)
                        {
                            ++skipped_no_improvement;
                        }
                        else if (!llaminar2::moe_rebalance_policy::addingResidentImprovesDynamicSpread(
                                shared_current_policy_load,
                                count,
                                resident_mask,
                                proposed_resident_mask,
                                config.participant_count,
                                static_cast<uint32_t>(source_participant),
                                config.participant_id,
                                shared_candidate_policy_load,
                                config.window_size_tokens,
                                config.min_load_spread_improvement,
                                config.min_load_spread_improvement_divisor))
                        {
                            ++skipped_no_improvement;
                        }
                        else
                        {
                            uint32_t destination_slot = kDeviceMoEInvalidSlot;
                            uint32_t payload_slot = kDeviceMoEInvalidSlot;
                            if (config.participant_id < kDeviceMoEMaxParticipants &&
                                shared_destination_transfer_slot_counts[config.participant_id] < payload_slot_capacity &&
                                source_participant >= 0 &&
                                static_cast<uint32_t>(source_participant) < kDeviceMoEMaxParticipants &&
                                shared_source_payload_slot_counts[
                                    static_cast<uint32_t>(source_participant)] < payload_slot_capacity)
                            {
                                destination_slot =
                                    shared_destination_transfer_slot_counts[config.participant_id];
                                payload_slot =
                                    shared_source_payload_slot_counts[
                                        static_cast<uint32_t>(source_participant)];
                            }
                            DeviceMoERebalancePlanEntryView entry{};
                            entry.op = kDeviceMoERebalancePlanExpertPayloadArrival;
                            entry.layer = layer;
                            entry.expert = best_expert;
                            entry.source_participant =
                                source_participant >= 0 ? static_cast<uint32_t>(source_participant) : 0u;
                            entry.destination_participant = config.participant_id;
                            entry.source_resident_mask = resident_mask;
                            entry.destination_slot = destination_slot;
                            entry.payload_slot = payload_slot;
                            if (destination_slot != kDeviceMoEInvalidSlot &&
                                payload_slot != kDeviceMoEInvalidSlot &&
                                source_participant >= 0 &&
                                rebalance_append_plan_entry(
                                    plan_entries,
                                    plan_count,
                                    plan_capacity,
                                    entry,
                                    status))
                            {
                                ++shared_destination_transfer_slot_counts[config.participant_id];
                                ++shared_source_payload_slot_counts[
                                    static_cast<uint32_t>(source_participant)];
                                shared_post_policy_resident_mask[best_expert] =
                                    proposed_resident_mask;
                                for (uint32_t participant = 0; participant < config.participant_count; ++participant)
                                    shared_current_policy_load[participant] =
                                        shared_candidate_policy_load[participant];
                                accepted_load_spread_improvement_total += delta.improvement;
                                if (delta.improvement > accepted_load_spread_improvement_max)
                                    accepted_load_spread_improvement_max = delta.improvement;
                                ++shared_local_replicas;
                            }
                            else
                            {
                                if (destination_slot == kDeviceMoEInvalidSlot && status)
                                    ++status->plan_overflow;
                                ++skipped_no_resident;
                            }
                        }
                    }
                    else if (!replicated)
                    {
                        ++skipped_no_resident;
                    }
                    else
                    {
                        desc.flags |= kDeviceMoEFlagReplicated;
                        if (local_resident)
                        {
                            bool resident_plan_ready = true;
                            if (defer_runtime_apply)
                            {
                                DeviceMoERebalancePlanEntryView entry{};
                                entry.op = kDeviceMoERebalancePlanResidentExpertAssignment;
                                entry.layer = layer;
                                entry.expert = best_expert;
                                entry.source_participant =
                                    desc.owner_participant >= 0
                                        ? static_cast<uint32_t>(desc.owner_participant)
                                        : config.participant_id;
                                entry.destination_participant = config.participant_id;
                                entry.source_resident_mask = resident_mask;
                                entry.destination_slot = kDeviceMoEInvalidSlot;
                                entry.payload_slot = kDeviceMoEInvalidSlot;
                                if (!rebalance_append_plan_entry(
                                        plan_entries,
                                        plan_count,
                                        plan_capacity,
                                        entry,
                                        status))
                                {
                                    ++skipped_no_resident;
                                    resident_plan_ready = false;
                                }
                            }
                            if (resident_plan_ready)
                            {
                                next.local_compute_mask[best_expert] = 1u;
                                next.replica_role[best_expert] =
                                    owner_local
                                        ? static_cast<uint8_t>(kDeviceMoEReplicaRolePrimary)
                                        : static_cast<uint8_t>(kDeviceMoEReplicaRoleReplica);
                                if (!owner_local)
                                    ++shared_local_replicas;
                                ++selected_replicas;
                            }
                            shared_post_policy_resident_mask[best_expert] =
                                next.resident_participant_mask[best_expert] & valid_mask;
                        }
                    }
                }
                __syncthreads();
            }

            if (leader)
            {
                if (defer_runtime_apply)
                {
                    const uint32_t command_count =
                        plan_count ? ((*plan_count < plan_capacity) ? *plan_count : plan_capacity) : 0u;
                    if (command_count > layer_command_count_before)
                    {
                        last_epoch = shared_plan_epoch;
                        ++changed_layers;
                    }
                }
                else
                {
                    next.reserved[0] =
                        runtime_compute_multi_resident_expert_count(
                            next, config.num_experts, config.participant_count);
                    runtime.active_bank = inactive_bank;
                    runtime.active_epoch = next.epoch;
                    last_epoch = runtime.active_epoch;
                    ++changed_layers;
                }
            }

            if (!defer_runtime_apply &&
                (config.flags & kDeviceMoERebalanceFlagResetHistograms) != 0u)
            {
                for (uint32_t expert = lane; expert < config.num_experts; expert += blockDim.x)
                {
                    runtime.decode_histogram[expert] = 0ULL;
                    runtime.decode_local_histogram[expert] = 0ULL;
                }
                if (leader)
                {
                    runtime.router_hot_cache_eligible_dispatches = 0ULL;
                    runtime.router_hot_cache_used_dispatches = 0ULL;
                    runtime.router_hot_cache_improved_dispatches = 0ULL;
                    runtime.router_hot_cache_default_load_spread_total = 0ULL;
                    runtime.router_hot_cache_actual_load_spread_total = 0ULL;
                    runtime.router_hot_cache_load_spread_improvement_total = 0ULL;
                    runtime.router_hot_cache_active_dispatches = 0ULL;
                    runtime.router_hot_cache_miss_dispatches = 0ULL;
                    runtime.router_hot_cache_selected_expert_slots = 0ULL;
                    runtime.router_hot_cache_replicated_selected_expert_slots = 0ULL;
                }
            }
            if (leader)
            {
                uint64_t layer_post_total = 0ULL;
                uint64_t layer_post_min = 0ULL;
                uint64_t layer_post_max = 0ULL;
                llaminar2::moe_rebalance_policy::finalizeLoadSpread(
                    shared_current_policy_load,
                    config.participant_count,
                    layer_post_total,
                    layer_post_min,
                    layer_post_max);
                for (uint32_t participant = 0; participant < config.participant_count; ++participant)
                    shared_post_policy_load[participant] += shared_current_policy_load[participant];
                post_wave_load_total += layer_post_total;
                post_wave_load_spread += layer_post_max - layer_post_min;
            }
            __syncthreads();
        }

        if (leader)
        {
            uint64_t pre_total = 0ULL;
            uint64_t pre_min = 0ULL;
            uint64_t pre_max = 0ULL;
            uint64_t post_total = 0ULL;
            uint64_t post_min = 0ULL;
            uint64_t post_max = 0ULL;
            llaminar2::moe_rebalance_policy::finalizeLoadSpread(
                shared_pre_policy_load,
                config.participant_count,
                pre_total,
                pre_min,
                pre_max);
            llaminar2::moe_rebalance_policy::finalizeLoadSpread(
                shared_post_policy_load,
                config.participant_count,
                post_total,
                post_min,
                post_max);
            status->status_code = invalid_layers > 0u
                                      ? kDeviceMoERebalanceStatusInvalidRuntime
                                      : kDeviceMoERebalanceStatusOk;
            status->windows_observed = 1u;
            status->windows_applied = changed_layers > 0u ? 1u : 0u;
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
            status->llep_assignment_span_count = llep_assignment_span_count;
            status->llep_weight_transfer_count = llep_weight_transfer_count;
            status->llep_standard_ep_selected = llep_standard_ep_selected;
            status->llep_skipped_balanced = llep_skipped_balanced;
            status->llep_skipped_insufficient_spread_improvement =
                llep_skipped_insufficient_spread_improvement;
            status->llep_skipped_insufficient_foreign_rows =
                llep_skipped_insufficient_foreign_rows;
            status->llep_min_chunk_skips = llep_min_chunk_skips;
            status->llep_forced_spills = llep_forced_spills;
            status->candidate_load_spread_improvement_total =
                candidate_load_spread_improvement_total;
            status->candidate_load_spread_improvement_max =
                candidate_load_spread_improvement_max;
            status->accepted_load_spread_improvement_total =
                accepted_load_spread_improvement_total;
            status->accepted_load_spread_improvement_max =
                accepted_load_spread_improvement_max;
            status->llep_native_rows = llep_native_rows;
            status->llep_spilled_rows = llep_spilled_rows;
            status->llep_required_spread_improvement =
                llep_required_spread_improvement;
            status->llep_required_foreign_rows = llep_required_foreign_rows;
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
            status->invalid_runtime_layers = invalid_layers;
            status->last_epoch = last_epoch;
            status->window_ready_slots =
                rebalance_clamp_u64_to_u32(
                    rebalance_window_observed_slots(gathered_histograms, config));
            status->window_required_slots =
                rebalance_clamp_u64_to_u32(rebalance_window_required_slots(config));
            if (leader && collect_load_stats)
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
                     participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants);
                     ++participant)
                {
                    status->pre_policy_participant_load[participant] =
                        participant < config.participant_count
                            ? shared_pre_policy_load[participant]
                            : 0ULL;
                    status->post_policy_participant_load[participant] =
                        participant < config.participant_count
                            ? shared_post_policy_load[participant]
                            : 0ULL;
                }
            }

            uint32_t command_count =
                plan_count ? ((*plan_count < plan_capacity) ? *plan_count : plan_capacity) : 0u;
            uint32_t requested_by_source[kDeviceMoEMaxParticipants] = {};
            uint32_t requested_payload_slots = 0u;
            uint32_t payload_source_participant_mask = 0u;
            uint32_t payload_destination_participant_mask = 0u;
            uint64_t payload_edge_mask = 0ULL;
            if (plan_entries)
            {
                for (uint32_t i = 0; i < command_count; ++i)
                {
                    const auto &plan = plan_entries[i];
                    if (rebalance_plan_requires_payload(plan.op) &&
                        plan.source_participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants) &&
                        plan.destination_participant < config.participant_count &&
                        plan.destination_participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants) &&
                        plan.payload_slot != kDeviceMoEInvalidSlot)
                    {
                        const uint32_t requested = plan.payload_slot + 1u;
                        if (requested_by_source[plan.source_participant] < requested)
                            requested_by_source[plan.source_participant] = requested;
                        payload_source_participant_mask |=
                            llaminar2::moe_rebalance_policy::participantBit(
                                plan.source_participant);
                        payload_destination_participant_mask |=
                            llaminar2::moe_rebalance_policy::participantBit(
                                plan.destination_participant);
                        payload_edge_mask |=
                            llaminar2::moe_rebalance_policy::directedParticipantEdgeBit(
                                plan.source_participant,
                                plan.destination_participant,
                                static_cast<uint32_t>(kDeviceMoEMaxParticipants));
                    }
                }
                for (uint32_t participant = 0;
                     participant < config.participant_count &&
                     participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants);
                     ++participant)
                {
                    if (requested_by_source[participant] > requested_payload_slots)
                        requested_payload_slots = requested_by_source[participant];
                }
            }
            uint32_t payload_bucket_slots =
                llaminar2::moe_rebalance_policy::payloadBucketSlots(
                    requested_payload_slots,
                    payload_slot_capacity);
            uint32_t payload_bucket_index =
                llaminar2::moe_rebalance_policy::payloadBucketIndex(
                    payload_bucket_slots);
            uint32_t payload_bucket_overflow =
                (status->plan_overflow != 0u ||
                 requested_payload_slots > payload_bucket_slots)
                    ? 1u
                    : 0u;
            const bool wave_cost_floor_rejected =
                requested_payload_slots > 0u &&
                !llaminar2::moe_rebalance_policy::transferWaveMeetsSpreadImprovementFloor(
                    accepted_load_spread_improvement_total,
                    requested_payload_slots,
                    config.min_wave_spread_improvement_per_payload_slot);
            const bool router_benefit_floor_rejected =
                requested_payload_slots > 0u &&
                !llaminar2::moe_rebalance_policy::transferWaveMeetsRealizedRouterBenefitFloor(
                    router_hot_cache_load_spread_improvement_total,
                    requested_payload_slots,
                    config.min_router_spread_improvement_per_payload_slot,
                    hot_cache_active_layers > 0u);
            const bool realized_router_payback =
                requested_payload_slots > 0u &&
                router_hot_cache_used_dispatches > 0ULL &&
                llaminar2::moe_rebalance_policy::transferWaveMeetsRealizedRouterBenefitFloor(
                    router_hot_cache_load_spread_improvement_total,
                    requested_payload_slots,
                    config.min_router_spread_improvement_per_payload_slot,
                    true);
            const bool durable_llep_ownership_wave =
                least_loaded_assignment && !hot_replica_cache;
            const bool post_load_spread_ceiling_rejected =
                requested_payload_slots > 0u &&
                (!llaminar2::moe_rebalance_policy::transferWaveParticipantSpreadIsAcceptable(
                     pre_max - pre_min,
                     post_max - post_min,
                     pre_total,
                     post_total,
                     requested_payload_slots,
                     realized_router_payback) ||
                 !llaminar2::moe_rebalance_policy::transferWaveImprovesAggregateLoadSpread(
                     pre_wave_load_spread,
                     post_wave_load_spread,
                     pre_wave_load_total,
                     post_wave_load_total,
                     requested_payload_slots) ||
                 (!durable_llep_ownership_wave &&
                  !llaminar2::moe_rebalance_policy::transferWaveMeetsPostLoadSpreadCeiling(
                     post_wave_load_spread,
                     post_wave_load_total,
                     requested_payload_slots,
                     config.max_post_wave_load_spread_per_mille)));
            const bool wave_rejected =
                wave_cost_floor_rejected ||
                router_benefit_floor_rejected ||
                post_load_spread_ceiling_rejected;
            if (wave_rejected)
            {
                uint32_t resident_command_count = 0u;
                command_count =
                    llaminar2::moe_rebalance_policy::prunePayloadArrivalsPreservingResidentAssignments(
                        plan_entries,
                        command_count,
                        &resident_command_count);
                if (plan_count)
                    *plan_count = command_count;
                skipped_no_improvement +=
                    selected_replicas > resident_command_count
                        ? selected_replicas - resident_command_count
                        : 1u;
                selected_replicas = resident_command_count;
                accepted_load_spread_improvement_total = 0ULL;
                accepted_load_spread_improvement_max = 0ULL;
                requested_payload_slots = 0u;
                payload_bucket_slots = 0u;
                payload_bucket_index = 0u;
                payload_bucket_overflow = 0u;
                payload_source_participant_mask = 0u;
                payload_destination_participant_mask = 0u;
                payload_edge_mask = 0ULL;
                if (command_count == 0u)
                {
                    changed_layers = 0u;
                    last_epoch = 0u;
                }
            }
            if (wave_rejected)
            {
                status->windows_applied = command_count > 0u ? 1u : 0u;
                status->changed_layers = changed_layers;
                status->selected_replicas = selected_replicas;
                status->skipped_no_improvement = skipped_no_improvement;
                status->accepted_load_spread_improvement_total =
                    accepted_load_spread_improvement_total;
                status->accepted_load_spread_improvement_max =
                    accepted_load_spread_improvement_max;
                status->last_epoch = last_epoch;
                status->planned_arrivals = 0u;
                if (collect_load_stats)
                {
                    status->post_policy_load_total = status->pre_policy_load_total;
                    status->post_policy_load_min = status->pre_policy_load_min;
                    status->post_policy_load_max = status->pre_policy_load_max;
                    status->post_policy_imbalance_numerator =
                        status->pre_policy_imbalance_numerator;
                    status->post_policy_imbalance_denominator =
                        status->pre_policy_imbalance_denominator;
                    for (uint32_t participant = 0;
                         participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants);
                         ++participant)
                    {
                        status->post_policy_participant_load[participant] =
                            status->pre_policy_participant_load[participant];
                    }
                }
            }
            status->payload_bucket_requested_slots = requested_payload_slots;
            status->payload_bucket_slots = payload_bucket_slots;
            status->payload_bucket_index = payload_bucket_index;
            status->payload_bucket_overflow = payload_bucket_overflow;
            status->skipped_wave_cost_floor = wave_cost_floor_rejected ? 1u : 0u;
            status->skipped_low_router_benefit =
                router_benefit_floor_rejected ? 1u : 0u;
            status->skipped_post_load_spread_ceiling =
                post_load_spread_ceiling_rejected ? 1u : 0u;
            status->payload_source_participant_mask = payload_source_participant_mask;
            status->payload_destination_participant_mask =
                payload_destination_participant_mask;
            status->payload_edge_mask = payload_edge_mask;
            status->post_wave_load_total = post_wave_load_total;
            status->post_wave_load_spread = post_wave_load_spread;
            const uint32_t start_layer =
                (layer_window_start + start_offset) % config.num_layers;
            uint32_t planned_start_layer = start_layer;
            uint32_t planned_layer_count = layer_wave_count;
            if (command_count > 0u)
            {
                rebalance_command_layer_span(
                    plan_entries,
                    command_count,
                    config,
                    layer_window_start,
                    start_offset,
                    layer_window_count,
                    layer_wave_count,
                    &planned_start_layer,
                    &planned_layer_count);
            }
            if (command_header)
            {
                command_header->epoch = defer_runtime_apply && command_count > 0u
                                            ? shared_plan_epoch
                                            : last_epoch;
                command_header->command_count = command_count;
            }
            if (wave_state)
            {
                wave_state->epoch = last_epoch;
                wave_state->planned_start_layer = planned_start_layer;
                wave_state->planned_layer_count = planned_layer_count;
                wave_state->requested_payload_slots = requested_payload_slots;
                wave_state->payload_bucket_slots = payload_bucket_slots;
                wave_state->payload_bucket_index = payload_bucket_index;
                wave_state->payload_bucket_overflow = payload_bucket_overflow;
                const uint32_t next_offset =
                    layer_window_count > 0u
                        ? ((start_offset + layer_wave_count) % layer_window_count)
                        : 0u;
                wave_state->next_start_layer =
                    (layer_window_start + next_offset) % config.num_layers;
            }
            if (defer_runtime_apply &&
                command_count > 0u &&
                controller_state &&
                rebalance_graph_controller_state_basic_ok(controller_state, config) &&
                command_wave_index < controller_state->wave_count)
            {
                const bool transfer_payload_required = requested_payload_slots > 0u;
                auto &wave = controller_state->waves[command_wave_index];
                wave.magic = kDeviceMoERebalanceMagic;
                wave.version = kDeviceMoERebalanceVersion;
                wave.epoch = shared_plan_epoch;
                wave.state = kDeviceMoERebalanceLifecyclePlanning;
                wave.planned_start_layer =
                    wave_state ? wave_state->planned_start_layer : planned_start_layer;
                wave.planned_layer_count =
                    wave_state ? wave_state->planned_layer_count : planned_layer_count;
                wave.command_count = command_count;
                wave.copied_arrivals = transfer_payload_required ? 0u : command_count;
                wave.applied_arrivals = 0u;
                wave.applied_layer_count = 0u;
                wave.error_code = 0u;
                wave.requested_payload_slots = requested_payload_slots;
                wave.payload_bucket_slots = payload_bucket_slots;
                wave.payload_bucket_index = payload_bucket_index;
                wave.payload_bucket_overflow = payload_bucket_overflow;
                __threadfence();
                wave.state = transfer_payload_required
                                 ? kDeviceMoERebalanceLifecyclePlanning
                                 : kDeviceMoERebalanceLifecycleReadyToApply;
                controller_state->next_epoch = shared_plan_epoch + 1u;
            }
            else if (defer_runtime_apply &&
                     command_count == 0u &&
                     controller_state &&
                     rebalance_graph_controller_state_basic_ok(controller_state, config) &&
                     command_wave_index < controller_state->wave_count)
            {
                reset_rebalance_wave_progress_device(controller_state->waves[command_wave_index]);
            }
        }
    }

    __global__ void device_rebalance_dynamic_ownership_controller_kernel(
        DeviceMoELayerRuntimeView *runtime_layers,
        const unsigned long long *gathered_histograms,
        DeviceMoERebalanceStatusView *status,
        DeviceMoERebalanceConfigView config,
        DeviceMoERebalancePlanEntryView *plan_entries,
        uint32_t *plan_count,
        uint32_t plan_capacity,
        uint32_t payload_slot_capacity,
        DeviceMoERebalanceCommandBufferHeaderView *command_header,
        DeviceMoERebalanceWaveStateView *wave_state,
        DeviceMoERebalanceGraphControllerStateView *controller_state,
        uint32_t command_buffer_count)
    {
        if (blockIdx.x != 0) return;

        const uint32_t lane = threadIdx.x;
        const bool leader = lane == 0u;
        __shared__ uint32_t shared_command_wave_index;
        __shared__ uint32_t shared_abort;
        __shared__ uint32_t shared_layer_invalid;
        __shared__ uint32_t shared_plan_epoch;
        __shared__ uint64_t shared_owner_policy_load[kDeviceMoEMaxParticipants];
        __shared__ uint64_t shared_pre_policy_load[kDeviceMoEMaxParticipants];
        __shared__ uint64_t shared_post_policy_load[kDeviceMoEMaxParticipants];
        __shared__ uint64_t shared_expert_counts[kDeviceMoEMaxExperts];
        __shared__ int32_t shared_expert_owners[kDeviceMoEMaxExperts];
        __shared__ uint32_t shared_destination_transfer_slot_counts[kDeviceMoEMaxParticipants];
        __shared__ uint32_t shared_source_payload_slot_counts[kDeviceMoEMaxParticipants];

        if (leader)
        {
            shared_command_wave_index =
                rebalance_active_command_wave_index(controller_state, config, command_buffer_count);
            shared_abort = 0u;
        }
        for (uint32_t participant = lane;
             participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants);
             participant += blockDim.x)
        {
            shared_owner_policy_load[participant] = 0ULL;
            shared_pre_policy_load[participant] = 0ULL;
            shared_post_policy_load[participant] = 0ULL;
            shared_destination_transfer_slot_counts[participant] = 0u;
            shared_source_payload_slot_counts[participant] = 0u;
        }
        for (uint32_t expert = lane;
             expert < static_cast<uint32_t>(kDeviceMoEMaxExperts);
             expert += blockDim.x)
        {
            shared_expert_counts[expert] = 0ULL;
            shared_expert_owners[expert] = -1;
        }
        __syncthreads();

        const uint32_t command_wave_index = shared_command_wave_index;
        if (plan_entries)
            plan_entries += static_cast<unsigned long long>(command_wave_index) * plan_capacity;
        if (plan_count)
            plan_count += command_wave_index;
        if (command_header)
            command_header += command_wave_index;
        if (wave_state)
            wave_state += command_wave_index;

        if (leader)
        {
            if (status)
            {
                DeviceMoERebalanceStatusView zero_status{};
                *status = zero_status;
                status->magic = kDeviceMoERebalanceMagic;
                status->version = kDeviceMoERebalanceVersion;
                status->status_code = kDeviceMoERebalanceStatusOk;
            }
            else
            {
                shared_abort = 1u;
            }

            if (shared_abort == 0u &&
                rebalance_active_wave_busy_for_new_plan(
                    controller_state,
                    config,
                    command_wave_index))
            {
                status->status_code = kDeviceMoERebalanceStatusWindowNotReady;
                status->skipped_not_ready = 1u;
                status->skipped_busy_wave = 1u;
                shared_abort = 1u;
            }

            if (plan_count)
                *plan_count = 0u;

            if (command_header)
            {
                command_header->magic = kDeviceMoERebalanceMagic;
                command_header->version = kDeviceMoERebalanceVersion;
                command_header->epoch = 0u;
                command_header->phase = kDeviceMoERebalancePhasePlanAssignments;
                command_header->command_count = 0u;
                command_header->command_capacity = plan_capacity;
                command_header->participant_id = config.participant_id;
                command_header->participant_count = config.participant_count;
            }

            if (wave_state)
            {
                const bool keep_cursor =
                    wave_state->magic == kDeviceMoERebalanceMagic &&
                    wave_state->version == kDeviceMoERebalanceVersion &&
                    wave_state->participant_id == config.participant_id &&
                    wave_state->participant_count == config.participant_count;
                const uint32_t next_start = keep_cursor ? wave_state->next_start_layer : 0u;
                wave_state->magic = kDeviceMoERebalanceMagic;
                wave_state->version = kDeviceMoERebalanceVersion;
                wave_state->epoch = 0u;
                wave_state->next_start_layer = next_start;
                wave_state->planned_start_layer = 0u;
                wave_state->planned_layer_count = 0u;
                wave_state->command_capacity = plan_capacity;
                wave_state->participant_id = config.participant_id;
                wave_state->participant_count = config.participant_count;
                wave_state->requested_payload_slots = 0u;
                wave_state->payload_bucket_slots = 0u;
                wave_state->payload_bucket_index = 0u;
                wave_state->payload_bucket_overflow = 0u;
            }

            if (shared_abort == 0u && !rebalance_config_ok(config))
            {
                status->status_code = kDeviceMoERebalanceStatusInvalidConfig;
                shared_abort = 1u;
            }
            if (shared_abort == 0u && (!runtime_layers || !gathered_histograms))
            {
                status->status_code = !runtime_layers
                                          ? kDeviceMoERebalanceStatusInvalidRuntime
                                          : kDeviceMoERebalanceStatusMissingHistogram;
                shared_abort = 1u;
            }
            if (shared_abort == 0u)
            {
                const unsigned long long observed_slots =
                    rebalance_window_observed_slots(gathered_histograms, config);
                const unsigned long long required_slots =
                    rebalance_window_required_slots(config);
                status->window_ready_slots =
                    rebalance_clamp_u64_to_u32(observed_slots);
                status->window_required_slots =
                    rebalance_clamp_u64_to_u32(required_slots);
                if (observed_slots < required_slots)
                {
                    status->status_code = kDeviceMoERebalanceStatusWindowNotReady;
                    status->skipped_not_ready = 1u;
                    status->skipped_busy_wave = 0u;
                    shared_abort = 1u;
                }
            }
        }
        __syncthreads();
        if (shared_abort != 0u)
            return;

        const uint32_t valid_mask = runtime_valid_participant_mask(config.participant_count);
        const bool collect_load_stats =
            (config.flags & kDeviceMoERebalanceFlagCollectLoadStats) != 0u;
        const uint32_t layer_window_count =
            config.layer_window_count == 0u
                ? config.num_layers
                : ((config.layer_window_count < config.num_layers)
                       ? config.layer_window_count
                       : config.num_layers);
        const uint32_t layer_wave_count =
            config.layer_wave_count == 0u
                ? layer_window_count
                : ((config.layer_wave_count < layer_window_count)
                       ? config.layer_wave_count
                       : layer_window_count);
        const uint32_t layer_window_start =
            config.num_layers == 0u ? 0u : (config.layer_window_start % config.num_layers);
        uint32_t start_offset = 0u;
        if (wave_state && layer_window_count > 0u)
        {
            const uint32_t candidate = wave_state->next_start_layer % config.num_layers;
            const uint32_t candidate_offset =
                (candidate + config.num_layers - layer_window_start) % config.num_layers;
            if (candidate_offset < layer_window_count)
                start_offset = candidate_offset;
        }

        if (leader)
        {
            shared_plan_epoch =
                (rebalance_graph_controller_state_basic_ok(controller_state, config) &&
                 controller_state->next_epoch > 0u)
                    ? controller_state->next_epoch
                    : 1u;
        }
        __syncthreads();

        uint32_t changed_layers = 0u;
        uint32_t invalid_layers = 0u;
        uint32_t last_epoch = 0u;
        uint32_t dynamic_ownership_swap_attempts = 0u;
        uint32_t dynamic_ownership_swap_accepts = 0u;
        uint32_t dynamic_ownership_swap_rejections = 0u;
        uint32_t skipped_no_resident = 0u;
        uint32_t skipped_no_improvement = 0u;
        uint64_t accepted_load_spread_improvement_total = 0ULL;
        uint64_t accepted_load_spread_improvement_max = 0ULL;
        uint64_t pre_wave_load_total = 0ULL;
        uint64_t pre_wave_load_spread = 0ULL;
        uint64_t post_wave_load_total = 0ULL;
        uint64_t post_wave_load_spread = 0ULL;

        if (config.participant_id != config.root_participant)
        {
            if (leader)
            {
                status->status_code = kDeviceMoERebalanceStatusOk;
                status->windows_observed = 1u;
                if (command_header)
                    command_header->command_count = 0u;
                if (wave_state)
                {
                    const uint32_t start_layer =
                        (layer_window_start + start_offset) % config.num_layers;
                    wave_state->planned_start_layer = start_layer;
                    wave_state->planned_layer_count = layer_wave_count;
                    const uint32_t next_offset =
                        layer_window_count > 0u
                            ? ((start_offset + layer_wave_count) % layer_window_count)
                            : 0u;
                    wave_state->next_start_layer =
                        (layer_window_start + next_offset) % config.num_layers;
                }
            }
            return;
        }

        for (uint32_t window_index = 0; window_index < layer_wave_count; ++window_index)
        {
            const uint32_t layer =
                (layer_window_start + ((start_offset + window_index) % layer_window_count)) %
                config.num_layers;
            DeviceMoELayerRuntimeView &runtime = runtime_layers[layer];
            if (leader)
            {
                shared_layer_invalid =
                    (runtime.active_bank > 1u ||
                     runtime.expert_count != config.num_experts ||
                     runtime.top_k != config.top_k ||
                     runtime.participant_id != config.participant_id ||
                     runtime.participant_count != config.participant_count)
                        ? 1u
                        : 0u;
                if (shared_layer_invalid != 0u)
                    ++invalid_layers;
            }
            __syncthreads();
            if (shared_layer_invalid != 0u)
                continue;

            uint32_t layer_command_count_before = 0u;
            if (leader && plan_count)
                layer_command_count_before =
                    *plan_count < plan_capacity ? *plan_count : plan_capacity;

            const DeviceMoEPlacementBankView &active = runtime.banks[runtime.active_bank];
            if (leader)
            {
                for (uint32_t participant = 0; participant < config.participant_count; ++participant)
                    shared_owner_policy_load[participant] = 0ULL;
                for (uint32_t expert = 0; expert < config.num_experts; ++expert)
                {
                    const DeviceMoEExpertDescriptorView &desc = active.experts[expert];
                    const unsigned long long count =
                        rebalance_global_count(gathered_histograms, config, window_index, expert);
                    shared_expert_counts[expert] = count;
                    shared_expert_owners[expert] =
                        desc.owner_participant >= 0 &&
                                desc.owner_participant < static_cast<int32_t>(config.participant_count)
                            ? desc.owner_participant
                            : -1;
                    if (shared_expert_owners[expert] >= 0)
                        shared_owner_policy_load[static_cast<uint32_t>(shared_expert_owners[expert])] += count;
                }

                uint64_t layer_pre_total = 0ULL;
                uint64_t layer_pre_min = 0ULL;
                uint64_t layer_pre_max = 0ULL;
                llaminar2::moe_rebalance_policy::finalizeLoadSpread(
                    shared_owner_policy_load,
                    config.participant_count,
                    layer_pre_total,
                    layer_pre_min,
                    layer_pre_max);
                for (uint32_t participant = 0; participant < config.participant_count; ++participant)
                    shared_pre_policy_load[participant] += shared_owner_policy_load[participant];
                pre_wave_load_total += layer_pre_total;
                pre_wave_load_spread += layer_pre_max - layer_pre_min;

                const uint32_t max_swaps = config.dynamic_max_swaps_per_layer;
                const uint32_t max_entries =
                    config.dynamic_max_plan_entries_per_wave == 0u
                        ? plan_capacity
                        : ((config.dynamic_max_plan_entries_per_wave < plan_capacity)
                               ? config.dynamic_max_plan_entries_per_wave
                               : plan_capacity);
                for (uint32_t swap_iter = 0; swap_iter < max_swaps; ++swap_iter)
                {
                    const uint32_t command_count =
                        plan_count ? ((*plan_count < plan_capacity) ? *plan_count : plan_capacity) : 0u;
                    if (command_count + 2u > plan_capacity ||
                        command_count + 2u > max_entries)
                    {
                        if (status)
                            ++status->plan_overflow;
                        break;
                    }

                    ++dynamic_ownership_swap_attempts;
                    const auto swap_choice =
                        llaminar2::moe_rebalance_policy::bestDynamicOwnershipSwap(
                            shared_owner_policy_load,
                            shared_expert_counts,
                            shared_expert_owners,
                            config.num_experts,
                            config.participant_count,
                            config.dynamic_imbalance_threshold_per_mille,
                            config.dynamic_min_improvement_per_mille,
                            config.dynamic_min_window_activations);
                    if (!swap_choice.valid)
                    {
                        ++dynamic_ownership_swap_rejections;
                        break;
                    }

                    const uint32_t heavy_source = swap_choice.overloaded_participant;
                    const uint32_t heavy_destination = swap_choice.underloaded_participant;
                    const uint32_t light_source = swap_choice.underloaded_participant;
                    const uint32_t light_destination = swap_choice.overloaded_participant;
                    const uint32_t heavy_physical_source_mask =
                        active.resident_participant_mask[swap_choice.heavy_expert] & valid_mask;
                    const uint32_t light_physical_source_mask =
                        active.resident_participant_mask[swap_choice.light_expert] & valid_mask;
                    const uint32_t heavy_source_bit =
                        runtime_participant_bit(static_cast<int>(heavy_source));
                    const uint32_t light_source_bit =
                        runtime_participant_bit(static_cast<int>(light_source));
                    if (heavy_source >= static_cast<uint32_t>(kDeviceMoEMaxParticipants) ||
                        heavy_destination >= static_cast<uint32_t>(kDeviceMoEMaxParticipants) ||
                        light_source >= static_cast<uint32_t>(kDeviceMoEMaxParticipants) ||
                        light_destination >= static_cast<uint32_t>(kDeviceMoEMaxParticipants) ||
                        shared_source_payload_slot_counts[heavy_source] >= payload_slot_capacity ||
                        shared_source_payload_slot_counts[light_source] >= payload_slot_capacity ||
                        shared_destination_transfer_slot_counts[heavy_destination] >= payload_slot_capacity ||
                        shared_destination_transfer_slot_counts[light_destination] >= payload_slot_capacity)
                    {
                        if (status)
                            ++status->plan_overflow;
                        break;
                    }
                    if ((heavy_physical_source_mask & heavy_source_bit) == 0u ||
                        (light_physical_source_mask & light_source_bit) == 0u)
                    {
                        /*
                         * The graph-captured controller publishes commands that the
                         * payload packer consumes later.  Reject an ownership swap
                         * immediately if either old owner is not a physical resident.
                         */
                        ++skipped_no_resident;
                        ++dynamic_ownership_swap_rejections;
                        break;
                    }

                    DeviceMoERebalancePlanEntryView heavy_entry{};
                    heavy_entry.op = kDeviceMoERebalancePlanOwnershipTransfer;
                    heavy_entry.layer = layer;
                    heavy_entry.expert = swap_choice.heavy_expert;
                    heavy_entry.source_participant = heavy_source;
                    heavy_entry.destination_participant = heavy_destination;
                    heavy_entry.source_resident_mask = heavy_physical_source_mask;
                    heavy_entry.destination_slot =
                        shared_destination_transfer_slot_counts[heavy_destination];
                    heavy_entry.payload_slot =
                        shared_source_payload_slot_counts[heavy_source];

                    DeviceMoERebalancePlanEntryView light_entry{};
                    light_entry.op = kDeviceMoERebalancePlanOwnershipTransfer;
                    light_entry.layer = layer;
                    light_entry.expert = swap_choice.light_expert;
                    light_entry.source_participant = light_source;
                    light_entry.destination_participant = light_destination;
                    light_entry.source_resident_mask = light_physical_source_mask;
                    light_entry.destination_slot =
                        shared_destination_transfer_slot_counts[light_destination];
                    light_entry.payload_slot =
                        shared_source_payload_slot_counts[light_source];

                    if (!rebalance_append_plan_entry(
                            plan_entries,
                            plan_count,
                            plan_capacity,
                            heavy_entry,
                            status) ||
                        !rebalance_append_plan_entry(
                            plan_entries,
                            plan_count,
                            plan_capacity,
                            light_entry,
                            status))
                    {
                        break;
                    }

                    ++dynamic_ownership_swap_accepts;
                    ++shared_source_payload_slot_counts[heavy_source];
                    ++shared_source_payload_slot_counts[light_source];
                    ++shared_destination_transfer_slot_counts[heavy_destination];
                    ++shared_destination_transfer_slot_counts[light_destination];
                    llaminar2::moe_rebalance_policy::applyDynamicOwnershipSwap(
                        shared_owner_policy_load,
                        shared_expert_owners,
                        swap_choice);
                    accepted_load_spread_improvement_total += swap_choice.improvement;
                    if (swap_choice.improvement > accepted_load_spread_improvement_max)
                        accepted_load_spread_improvement_max = swap_choice.improvement;
                }

                uint64_t layer_post_total = 0ULL;
                uint64_t layer_post_min = 0ULL;
                uint64_t layer_post_max = 0ULL;
                llaminar2::moe_rebalance_policy::finalizeLoadSpread(
                    shared_owner_policy_load,
                    config.participant_count,
                    layer_post_total,
                    layer_post_min,
                    layer_post_max);
                for (uint32_t participant = 0; participant < config.participant_count; ++participant)
                    shared_post_policy_load[participant] += shared_owner_policy_load[participant];
                post_wave_load_total += layer_post_total;
                post_wave_load_spread += layer_post_max - layer_post_min;

                const uint32_t command_count =
                    plan_count ? ((*plan_count < plan_capacity) ? *plan_count : plan_capacity) : 0u;
                if (command_count > layer_command_count_before)
                {
                    last_epoch = shared_plan_epoch;
                    ++changed_layers;
                }
            }
            __syncthreads();
        }

        if (leader)
        {
            uint64_t pre_total = 0ULL;
            uint64_t pre_min = 0ULL;
            uint64_t pre_max = 0ULL;
            uint64_t post_total = 0ULL;
            uint64_t post_min = 0ULL;
            uint64_t post_max = 0ULL;
            llaminar2::moe_rebalance_policy::finalizeLoadSpread(
                shared_pre_policy_load,
                config.participant_count,
                pre_total,
                pre_min,
                pre_max);
            llaminar2::moe_rebalance_policy::finalizeLoadSpread(
                shared_post_policy_load,
                config.participant_count,
                post_total,
                post_min,
                post_max);

            uint32_t command_count =
                plan_count ? ((*plan_count < plan_capacity) ? *plan_count : plan_capacity) : 0u;
            uint32_t requested_by_source[kDeviceMoEMaxParticipants] = {};
            uint32_t requested_payload_slots = 0u;
            uint32_t payload_source_participant_mask = 0u;
            uint32_t payload_destination_participant_mask = 0u;
            uint64_t payload_edge_mask = 0ULL;
            for (uint32_t i = 0; plan_entries && i < command_count; ++i)
            {
                const auto &plan = plan_entries[i];
                if (rebalance_plan_requires_payload(plan.op) &&
                    plan.source_participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants) &&
                    plan.destination_participant < config.participant_count &&
                    plan.destination_participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants) &&
                    plan.payload_slot != kDeviceMoEInvalidSlot)
                {
                    const uint32_t requested = plan.payload_slot + 1u;
                    if (requested_by_source[plan.source_participant] < requested)
                        requested_by_source[plan.source_participant] = requested;
                    payload_source_participant_mask |=
                        llaminar2::moe_rebalance_policy::participantBit(
                            plan.source_participant);
                    payload_destination_participant_mask |=
                        llaminar2::moe_rebalance_policy::participantBit(
                            plan.destination_participant);
                    payload_edge_mask |=
                        llaminar2::moe_rebalance_policy::directedParticipantEdgeBit(
                            plan.source_participant,
                            plan.destination_participant,
                            static_cast<uint32_t>(kDeviceMoEMaxParticipants));
                }
            }
            for (uint32_t participant = 0;
                 participant < config.participant_count &&
                 participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants);
                 ++participant)
            {
                if (requested_by_source[participant] > requested_payload_slots)
                    requested_payload_slots = requested_by_source[participant];
            }
            uint32_t payload_bucket_slots =
                llaminar2::moe_rebalance_policy::payloadBucketSlots(
                    requested_payload_slots,
                    payload_slot_capacity);
            uint32_t payload_bucket_index =
                llaminar2::moe_rebalance_policy::payloadBucketIndex(
                    payload_bucket_slots);
            uint32_t payload_bucket_overflow =
                (status->plan_overflow != 0u ||
                 requested_payload_slots > payload_bucket_slots)
                    ? 1u
                    : 0u;

            const bool wave_cost_floor_rejected =
                requested_payload_slots > 0u &&
                !llaminar2::moe_rebalance_policy::transferWaveMeetsSpreadImprovementFloor(
                    accepted_load_spread_improvement_total,
                    requested_payload_slots,
                    config.min_wave_spread_improvement_per_payload_slot);
            const bool post_load_spread_ceiling_rejected =
                requested_payload_slots > 0u &&
                (!llaminar2::moe_rebalance_policy::transferWaveParticipantSpreadIsAcceptable(
                     pre_max - pre_min,
                     post_max - post_min,
                     pre_total,
                     post_total,
                     requested_payload_slots,
                     false) ||
                 !llaminar2::moe_rebalance_policy::transferWaveImprovesAggregateLoadSpread(
                     pre_wave_load_spread,
                     post_wave_load_spread,
                     pre_wave_load_total,
                     post_wave_load_total,
                     requested_payload_slots) ||
                 !llaminar2::moe_rebalance_policy::transferWaveMeetsPostLoadSpreadCeiling(
                     post_wave_load_spread,
                     post_wave_load_total,
                     requested_payload_slots,
                     config.max_post_wave_load_spread_per_mille));
            const bool wave_rejected =
                wave_cost_floor_rejected || post_load_spread_ceiling_rejected;
            if (wave_rejected)
            {
                command_count = 0u;
                if (plan_count)
                    *plan_count = 0u;
                skipped_no_improvement += 1u;
                changed_layers = 0u;
                last_epoch = 0u;
                accepted_load_spread_improvement_total = 0ULL;
                accepted_load_spread_improvement_max = 0ULL;
                requested_payload_slots = 0u;
                payload_bucket_slots = 0u;
                payload_bucket_index = 0u;
                payload_bucket_overflow = 0u;
                payload_source_participant_mask = 0u;
                payload_destination_participant_mask = 0u;
                payload_edge_mask = 0ULL;
                post_total = pre_total;
                post_min = pre_min;
                post_max = pre_max;
            }

            status->status_code = invalid_layers > 0u
                                      ? kDeviceMoERebalanceStatusInvalidRuntime
                                      : kDeviceMoERebalanceStatusOk;
            status->windows_observed = 1u;
            status->windows_applied = command_count > 0u ? 1u : 0u;
            status->changed_layers = changed_layers;
            status->skipped_no_resident = skipped_no_resident;
            status->skipped_no_improvement = skipped_no_improvement;
            status->dynamic_ownership_swap_attempts = dynamic_ownership_swap_attempts;
            status->dynamic_ownership_swap_accepts = dynamic_ownership_swap_accepts;
            status->dynamic_ownership_swap_rejections = dynamic_ownership_swap_rejections;
            status->accepted_load_spread_improvement_total =
                accepted_load_spread_improvement_total;
            status->accepted_load_spread_improvement_max =
                accepted_load_spread_improvement_max;
            status->invalid_runtime_layers = invalid_layers;
            status->last_epoch = last_epoch;
            status->window_ready_slots =
                rebalance_clamp_u64_to_u32(
                    rebalance_window_observed_slots(gathered_histograms, config));
            status->window_required_slots =
                rebalance_clamp_u64_to_u32(rebalance_window_required_slots(config));
            status->payload_bucket_requested_slots = requested_payload_slots;
            status->payload_bucket_slots = payload_bucket_slots;
            status->payload_bucket_index = payload_bucket_index;
            status->payload_bucket_overflow = payload_bucket_overflow;
            status->skipped_wave_cost_floor = wave_cost_floor_rejected ? 1u : 0u;
            status->skipped_post_load_spread_ceiling =
                post_load_spread_ceiling_rejected ? 1u : 0u;
            status->payload_source_participant_mask = payload_source_participant_mask;
            status->payload_destination_participant_mask =
                payload_destination_participant_mask;
            status->payload_edge_mask = payload_edge_mask;
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
                     participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants);
                     ++participant)
                {
                    status->pre_policy_participant_load[participant] =
                        participant < config.participant_count
                            ? shared_pre_policy_load[participant]
                            : 0ULL;
                    status->post_policy_participant_load[participant] =
                        participant < config.participant_count
                            ? shared_post_policy_load[participant]
                            : 0ULL;
                }
            }

            const uint32_t start_layer =
                (layer_window_start + start_offset) % config.num_layers;
            uint32_t planned_start_layer = start_layer;
            uint32_t planned_layer_count = layer_wave_count;
            if (command_count > 0u)
            {
                rebalance_command_layer_span(
                    plan_entries,
                    command_count,
                    config,
                    layer_window_start,
                    start_offset,
                    layer_window_count,
                    layer_wave_count,
                    &planned_start_layer,
                    &planned_layer_count);
            }
            if (command_header)
            {
                command_header->epoch = command_count > 0u ? shared_plan_epoch : last_epoch;
                command_header->command_count = command_count;
            }
            if (wave_state)
            {
                wave_state->epoch = last_epoch;
                wave_state->planned_start_layer = planned_start_layer;
                wave_state->planned_layer_count = planned_layer_count;
                wave_state->requested_payload_slots = requested_payload_slots;
                wave_state->payload_bucket_slots = payload_bucket_slots;
                wave_state->payload_bucket_index = payload_bucket_index;
                wave_state->payload_bucket_overflow = payload_bucket_overflow;
                const uint32_t next_offset =
                    layer_window_count > 0u
                        ? ((start_offset + layer_wave_count) % layer_window_count)
                        : 0u;
                wave_state->next_start_layer =
                    (layer_window_start + next_offset) % config.num_layers;
            }
            if (command_count > 0u &&
                controller_state &&
                rebalance_graph_controller_state_basic_ok(controller_state, config) &&
                command_wave_index < controller_state->wave_count)
            {
                auto &wave = controller_state->waves[command_wave_index];
                wave.magic = kDeviceMoERebalanceMagic;
                wave.version = kDeviceMoERebalanceVersion;
                wave.epoch = shared_plan_epoch;
                wave.state = kDeviceMoERebalanceLifecyclePlanning;
                wave.planned_start_layer =
                    wave_state ? wave_state->planned_start_layer : planned_start_layer;
                wave.planned_layer_count =
                    wave_state ? wave_state->planned_layer_count : planned_layer_count;
                wave.command_count = command_count;
                wave.copied_arrivals = requested_payload_slots > 0u ? 0u : command_count;
                wave.applied_arrivals = 0u;
                wave.applied_layer_count = 0u;
                wave.error_code = 0u;
                wave.requested_payload_slots = requested_payload_slots;
                wave.payload_bucket_slots = payload_bucket_slots;
                wave.payload_bucket_index = payload_bucket_index;
                wave.payload_bucket_overflow = payload_bucket_overflow;
                __threadfence();
                wave.state = requested_payload_slots > 0u
                                 ? kDeviceMoERebalanceLifecyclePlanning
                                 : kDeviceMoERebalanceLifecycleReadyToApply;
                controller_state->next_epoch = shared_plan_epoch + 1u;
            }
            else if (command_count == 0u &&
                     controller_state &&
                     rebalance_graph_controller_state_basic_ok(controller_state, config) &&
                     command_wave_index < controller_state->wave_count)
            {
                reset_rebalance_wave_progress_device(controller_state->waves[command_wave_index]);
            }
        }
    }

    __global__ void pack_rebalance_histograms_kernel(
        const DeviceMoELayerRuntimeView *runtime_layers,
        unsigned long long *local_histograms,
        DeviceMoERebalanceConfigView config,
        const DeviceMoERebalanceWaveStateView *wave_state,
        const DeviceMoERebalanceGraphControllerStateView *controller_state,
        uint32_t command_buffer_count)
    {
        if (!runtime_layers || !local_histograms || !rebalance_config_ok(config))
            return;

        const uint32_t command_wave_index =
            rebalance_active_command_wave_index(controller_state, config, command_buffer_count);
        const DeviceMoERebalanceWaveStateView *selected_wave_state =
            wave_state ? wave_state + command_wave_index : nullptr;
        const uint32_t packed_layers = rebalance_layer_wave_count(config);
        const uint32_t total =
            packed_layers * config.num_experts;
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        while (idx < total)
        {
            const uint32_t wave_layer = idx / config.num_experts;
            const uint32_t expert = idx - wave_layer * config.num_experts;
            const uint32_t layer =
                rebalance_actual_layer_for_wave_index(config, selected_wave_state, wave_layer);
            const DeviceMoELayerRuntimeView &runtime = runtime_layers[layer];
            unsigned long long value = 0ULL;
            if (runtime.expert_count == config.num_experts &&
                runtime.top_k == config.top_k &&
                runtime.participant_id == config.participant_id &&
                runtime.participant_count == config.participant_count)
            {
                value = static_cast<unsigned long long>(
                    runtime.decode_local_histogram[expert]);
            }
            local_histograms[idx] = value;
            idx += blockDim.x * gridDim.x;
        }
    }

    __device__ __forceinline__ bool rebalance_matrix_desc_ready(
        const DeviceNativeVNNIMatrixDescView &desc)
    {
        return desc.payload != nullptr &&
               desc.scales != nullptr &&
               desc.n > 0 &&
               desc.k > 0 &&
               desc.blocks_per_row > 0u;
    }

    __device__ __forceinline__ bool rebalance_expert_desc_ready(
        const DeviceMoEExpertDescriptorView &desc)
    {
        return desc.logical_expert_id >= 0 &&
               rebalance_matrix_desc_ready(desc.gate) &&
               rebalance_matrix_desc_ready(desc.up) &&
               rebalance_matrix_desc_ready(desc.down);
    }

    __device__ __forceinline__ bool rebalance_transfer_desc_ready(
        const DeviceMoEExpertDescriptorView &desc)
    {
        return rebalance_matrix_desc_ready(desc.gate) &&
               rebalance_matrix_desc_ready(desc.up) &&
               rebalance_matrix_desc_ready(desc.down);
    }

    __device__ __forceinline__ bool rebalance_format_for_codebook(
        uint8_t codebook_id,
        uint8_t &payload_bytes,
        uint8_t &is_asymmetric,
        uint8_t &has_emins)
    {
        payload_bytes = 0u;
        is_asymmetric = 0u;
        has_emins = 0u;
        switch (codebook_id)
        {
        case 0:
        case 4:
            payload_bytes = 16u;
            return true;
        case 5:
            payload_bytes = 16u;
            is_asymmetric = 1u;
            return true;
        case 6:
            payload_bytes = 20u;
            return true;
        case 7:
            payload_bytes = 20u;
            is_asymmetric = 1u;
            return true;
        case 8:
            payload_bytes = 24u;
            is_asymmetric = 1u;
            return true;
        case 9:
            payload_bytes = 12u;
            is_asymmetric = 1u;
            return true;
        case 10:
            payload_bytes = 8u;
            is_asymmetric = 1u;
            has_emins = 1u;
            return true;
        case 11:
            payload_bytes = 13u;
            return true;
        case 12:
            payload_bytes = 12u;
            return true;
        case 13:
        case 14:
            payload_bytes = 9u;
            is_asymmetric = 1u;
            return true;
        case 15:
            payload_bytes = 8u;
            return true;
        case 16:
        case 17:
            payload_bytes = 6u;
            is_asymmetric = 1u;
            return true;
        case 19:
        case 20:
            payload_bytes = 32u;
            return true;
        default:
            return false;
        }
    }

    __device__ __forceinline__ unsigned long long rebalance_block_count(
        const DeviceNativeVNNIMatrixDescView &desc)
    {
        return static_cast<unsigned long long>(desc.blocks_per_row) *
               static_cast<unsigned long long>(desc.n);
    }

    __device__ __forceinline__ bool rebalance_projection_format(
        const DeviceNativeVNNIMatrixDescView &desc,
        uint8_t &payload_bytes,
        uint8_t &is_asymmetric,
        uint8_t &has_emins)
    {
        return rebalance_format_for_codebook(
            desc.codebook_id,
            payload_bytes,
            is_asymmetric,
            has_emins);
    }

    __device__ __forceinline__ bool rebalance_matrix_copy_ready(
        const DeviceNativeVNNIMatrixDescView &desc,
        const DeviceMoEExpertDirectoryEntryView &entry)
    {
        (void)entry;
        uint8_t payload_bytes = 0u;
        uint8_t is_asymmetric = 0u;
        uint8_t has_emins = 0u;
        if (!rebalance_matrix_desc_ready(desc) ||
            !rebalance_projection_format(desc, payload_bytes, is_asymmetric, has_emins) ||
            payload_bytes == 0u)
            return false;
        if (is_asymmetric != 0u && desc.mins == nullptr)
            return false;
        if (has_emins != 0u && desc.emins == nullptr)
            return false;
        return true;
    }

    __device__ __forceinline__ bool rebalance_directory_copy_ready(
        const DeviceMoEExpertDirectoryEntryView &entry)
    {
        return rebalance_matrix_copy_ready(entry.descriptor.gate, entry) &&
               rebalance_matrix_copy_ready(entry.descriptor.up, entry) &&
               rebalance_matrix_copy_ready(entry.descriptor.down, entry);
    }

    __device__ __forceinline__ bool rebalance_source_entry_ready(
        const DeviceMoEExpertDirectoryEntryView &entry,
        uint32_t participant,
        uint32_t layer,
        uint32_t expert)
    {
        const uint32_t required = kDeviceMoEDirectoryFlagValid |
                                  kDeviceMoEDirectoryFlagResident;
        return entry.participant == participant &&
               entry.layer == layer &&
               entry.expert == expert &&
               (entry.resident_mask &
                runtime_participant_bit(static_cast<int>(participant))) != 0u &&
               entry.slot_index != kDeviceMoEInvalidSlot &&
               entry.descriptor.local_slot >= 0 &&
               (entry.flags & required) == required &&
               rebalance_expert_desc_ready(entry.descriptor) &&
               rebalance_directory_copy_ready(entry);
    }

    __device__ __forceinline__ bool rebalance_transfer_slot_ready(
        const DeviceMoEExpertDirectoryEntryView &entry,
        uint32_t participant,
        uint32_t layer,
        uint32_t expert)
    {
        (void)layer;
        (void)expert;
        const uint32_t required = kDeviceMoEDirectoryFlagValid |
                                  kDeviceMoEDirectoryFlagTransferSlot;
        return entry.participant == participant &&
               entry.slot_index != kDeviceMoEInvalidSlot &&
               entry.descriptor.local_slot >= 0 &&
               (entry.flags & required) == required &&
               rebalance_transfer_desc_ready(entry.descriptor) &&
               rebalance_directory_copy_ready(entry);
    }

    __device__ __forceinline__ bool rebalance_transfer_slot_copy_complete(
        const DeviceMoEExpertDirectoryEntryView &entry,
        uint32_t participant,
        uint32_t layer,
        uint32_t expert)
    {
        return rebalance_transfer_slot_ready(entry, participant, layer, expert) &&
               entry.layer == layer &&
               entry.expert == expert &&
               entry.descriptor.logical_expert_id == static_cast<int32_t>(expert) &&
               (entry.flags & kDeviceMoEDirectoryFlagResident) != 0u &&
               (entry.flags & kDeviceMoEDirectoryFlagCopyComplete) != 0u;
    }

    __device__ __forceinline__ bool rebalance_directory_compatible(
        const DeviceMoEExpertDirectoryEntryView &src,
        const DeviceMoEExpertDirectoryEntryView &dst)
    {
        return src.descriptor.gate.n == dst.descriptor.gate.n &&
               src.descriptor.gate.k == dst.descriptor.gate.k &&
               src.descriptor.gate.blocks_per_row == dst.descriptor.gate.blocks_per_row &&
               src.descriptor.gate.codebook_id == dst.descriptor.gate.codebook_id &&
               src.descriptor.up.n == dst.descriptor.up.n &&
               src.descriptor.up.k == dst.descriptor.up.k &&
               src.descriptor.up.blocks_per_row == dst.descriptor.up.blocks_per_row &&
               src.descriptor.up.codebook_id == dst.descriptor.up.codebook_id &&
               src.descriptor.down.n == dst.descriptor.down.n &&
               src.descriptor.down.k == dst.descriptor.down.k &&
               src.descriptor.down.blocks_per_row == dst.descriptor.down.blocks_per_row &&
	               src.descriptor.down.codebook_id == dst.descriptor.down.codebook_id;
    }

    __device__ __forceinline__ DeviceMoEExpertDirectoryEntryView make_rebalance_source_entry(
        const DeviceMoELayerRuntimeView *runtime_layers,
        const DeviceMoERebalanceConfigView &config,
        uint32_t layer,
        uint32_t expert)
    {
        DeviceMoEExpertDirectoryEntryView entry{};
        entry.layer = layer;
        entry.expert = expert;
        entry.participant = config.participant_id;
        entry.slot_index = kDeviceMoEInvalidSlot;
        if (!runtime_layers ||
            !rebalance_config_ok(config) ||
            layer >= config.num_layers ||
            expert >= config.num_experts)
        {
            return entry;
        }

        const uint32_t participant_bit = runtime_participant_bit(static_cast<int>(config.participant_id));
        const uint32_t valid_mask = runtime_valid_participant_mask(config.participant_count);
        const DeviceMoELayerRuntimeView &runtime = runtime_layers[layer];
        const bool runtime_ok =
            runtime.active_bank <= 1u &&
            runtime.expert_count == config.num_experts &&
            runtime.top_k == config.top_k &&
            runtime.participant_id == config.participant_id &&
            runtime.participant_count == config.participant_count;
        if (!runtime_ok)
            return entry;

        const DeviceMoEPlacementBankView &bank = runtime.banks[runtime.active_bank];
        const DeviceMoEExpertDescriptorView &desc = bank.experts[expert];
        const uint32_t resident_mask =
            runtime_expert_resident_mask(&runtime, bank, static_cast<int>(expert)) & valid_mask;
        const bool local_resident = (resident_mask & participant_bit) != 0u;
        const bool local_slot_resident = desc.local_slot >= 0;
        entry.epoch = runtime.active_epoch;
        entry.generation = runtime.active_epoch;
        entry.resident_mask = resident_mask;
        if (desc.local_slot >= 0)
            entry.slot_index = static_cast<uint32_t>(desc.local_slot);

        if (local_resident && local_slot_resident && rebalance_expert_desc_ready(desc))
        {
            entry.descriptor = desc;
            if (rebalance_format_for_codebook(
                    desc.gate.codebook_id,
                    entry.payload_bytes_per_block,
                    entry.is_asymmetric,
                    entry.has_emins) &&
                rebalance_directory_copy_ready(entry))
            {
                entry.flags = kDeviceMoEDirectoryFlagValid |
                              kDeviceMoEDirectoryFlagResident;
                if (bank.local_compute_mask[expert] != 0u)
                    entry.flags |= kDeviceMoEDirectoryFlagLocalCompute;
            }
            else
            {
                entry.descriptor = DeviceMoEExpertDescriptorView{};
            }
        }
        return entry;
    }

    __global__ void pack_rebalance_directory_kernel(
        const DeviceMoELayerRuntimeView *runtime_layers,
        DeviceMoEExpertDirectoryEntryView *local_directory,
        DeviceMoERebalanceConfigView config)
    {
        if (!runtime_layers || !local_directory || !rebalance_config_ok(config))
            return;

        const uint32_t total = config.num_layers * config.num_experts;
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        while (idx < total)
        {
            const uint32_t layer = idx / config.num_experts;
            const uint32_t expert = idx - layer * config.num_experts;
            local_directory[idx] = make_rebalance_source_entry(
                runtime_layers,
                config,
                layer,
                expert);
            idx += blockDim.x * gridDim.x;
        }
    }

    __device__ __forceinline__ void init_rebalance_apply_status_device(DeviceMoERebalanceApplyStatusView *status)
    {
        DeviceMoERebalanceApplyStatusView zero{};
        *status = zero;
        status->magic = kDeviceMoERebalanceMagic;
        status->version = kDeviceMoERebalanceVersion;
        status->status_code = 0u;
    }

    __device__ __forceinline__ void init_rebalance_apply_status_in_progress_device(DeviceMoERebalanceApplyStatusView *status)
    {
        init_rebalance_apply_status_device(status);
        status->status_code = kDeviceMoERebalanceApplyStatusInProgress;
    }

    __global__ void init_rebalance_apply_status_kernel(DeviceMoERebalanceApplyStatusView *status)
    {
        if (!status || blockIdx.x != 0 || threadIdx.x != 0)
            return;
        init_rebalance_apply_status_device(status);
    }

    __device__ __forceinline__ bool rebalance_command_header_ok(
        const DeviceMoERebalanceCommandBufferHeaderView *command_header,
        const DeviceMoERebalanceConfigView &config);

    __device__ __forceinline__ bool rebalance_graph_controller_state_ok(
        const DeviceMoERebalanceGraphControllerStateView *state,
        const DeviceMoERebalanceConfigView &config)
    {
        return state &&
               state->magic == kDeviceMoERebalanceMagic &&
               state->version == kDeviceMoERebalanceVersion &&
               state->participant_id == config.participant_id &&
               state->participant_count == config.participant_count &&
               state->wave_count == 2u;
    }

    __device__ void init_rebalance_graph_controller_state_device(
        DeviceMoERebalanceGraphControllerStateView *state,
        const DeviceMoERebalanceConfigView &config)
    {
        DeviceMoERebalanceGraphControllerStateView zero{};
        *state = zero;
        state->magic = kDeviceMoERebalanceMagic;
        state->version = kDeviceMoERebalanceVersion;
        state->participant_id = config.participant_id;
        state->participant_count = config.participant_count;
        state->next_epoch = 1u;
        state->active_wave = 0u;
        state->wave_count = 2u;
        for (uint32_t i = 0; i < 2u; ++i)
        {
            auto &wave = state->waves[i];
            wave.magic = kDeviceMoERebalanceMagic;
            wave.version = kDeviceMoERebalanceVersion;
            wave.state = kDeviceMoERebalanceLifecycleIdle;
        }
    }

    __global__ void init_rebalance_graph_controller_state_kernel(
        DeviceMoERebalanceGraphControllerStateView *state,
        DeviceMoERebalanceConfigView config)
    {
        if (blockIdx.x != 0 || threadIdx.x != 0 || !state)
            return;
        if (!rebalance_config_ok(config))
        {
            state->last_error_code = kDeviceMoERebalanceStatusInvalidConfig;
            return;
        }
        if (!rebalance_graph_controller_state_ok(state, config))
            init_rebalance_graph_controller_state_device(state, config);
    }

    __device__ __forceinline__ uint32_t rebalance_wave_next_layer(
        const DeviceMoERebalanceWaveProgressView &wave,
        const DeviceMoERebalanceConfigView &config)
    {
        if (config.num_layers == 0u || wave.planned_layer_count == 0u)
            return config.num_layers;
        const uint32_t layer_offset =
            wave.applied_layer_count < wave.planned_layer_count
                ? wave.applied_layer_count
                : wave.planned_layer_count;
        if (layer_offset >= wave.planned_layer_count)
            return config.num_layers;
        return (wave.planned_start_layer + layer_offset) % config.num_layers;
    }

    struct DeviceMoEReadyWaveSelection
    {
        DeviceMoERebalanceWaveProgressView *wave;
        uint32_t wave_index;
    };

    __device__ DeviceMoEReadyWaveSelection rebalance_find_ready_wave(
        DeviceMoERebalanceGraphControllerStateView *state,
        const DeviceMoERebalanceCommandBufferHeaderView *command_headers,
        const DeviceMoERebalanceConfigView &config,
        int target_layer,
        uint32_t command_buffer_count)
    {
        DeviceMoEReadyWaveSelection selection{nullptr, 0u};
        const uint32_t count = rebalance_command_buffer_count(command_buffer_count);
        if (!rebalance_graph_controller_state_ok(state, config) ||
            !command_headers)
        {
            return selection;
        }

        for (uint32_t i = 0; i < state->wave_count && i < count; ++i)
        {
            auto &wave = state->waves[i];
            const auto &command_header = command_headers[i];
            if (wave.magic != kDeviceMoERebalanceMagic ||
                wave.version != kDeviceMoERebalanceVersion ||
                !rebalance_command_header_ok(&command_header, config) ||
                command_header.epoch == 0u ||
                wave.epoch != command_header.epoch ||
                wave.state != kDeviceMoERebalanceLifecycleReadyToApply)
            {
                continue;
            }
            if (target_layer >= 0)
            {
                const uint32_t next_layer = rebalance_wave_next_layer(wave, config);
                if (next_layer != static_cast<uint32_t>(target_layer))
                    continue;
            }
            selection.wave = &wave;
            selection.wave_index = i;
            return selection;
        }
        return selection;
    }

    /**
     * @brief Return true only when a participant's transfer copy pass is clean.
     *
     * The transfer-complete publisher consumes an allgather of per-participant
     * copy status records.  A participant can have no destination arrivals for a
     * wave and still be the source for another participant's payload slot, so
     * checking only `copied_arrivals` is not enough.  Source-side pack failures
     * such as missing descriptors must poison the whole transfer wave before any
     * apply kernel can observe an empty payload slot as valid work.
     */
    __device__ __forceinline__ bool rebalance_copy_status_ok(
        const DeviceMoERebalanceApplyStatusView &status)
    {
        return status.magic == kDeviceMoERebalanceMagic &&
               status.version == kDeviceMoERebalanceVersion &&
               status.status_code == 0u &&
               status.invalid_plan_entries == 0u &&
               status.missing_source_descriptors == 0u &&
               status.missing_destination_slots == 0u &&
               status.descriptor_mismatches == 0u &&
               status.copy_incomplete == 0u;
    }

    __device__ __forceinline__ uint32_t rebalance_expected_payload_arrivals(
        const DeviceMoERebalancePlanEntryView *plan_entries,
        uint32_t command_count,
        uint32_t participant,
        const DeviceMoERebalanceConfigView &config)
    {
        if (!plan_entries || participant >= config.participant_count)
            return 0u;
        uint32_t expected = 0u;
        for (uint32_t i = 0; i < command_count; ++i)
        {
            const auto &plan = plan_entries[i];
            if (plan.op == 0u ||
                !rebalance_plan_requires_payload(plan.op) ||
                plan.layer >= config.num_layers ||
                plan.expert >= config.num_experts ||
                plan.source_participant >= config.participant_count ||
                plan.destination_participant != participant)
            {
                continue;
            }
            ++expected;
        }
        return expected;
    }

    __global__ void publish_rebalance_transfer_complete_kernel(
        DeviceMoERebalanceGraphControllerStateView *state,
        const DeviceMoERebalanceCommandBufferHeaderView *command_header,
        const DeviceMoERebalanceWaveStateView *wave_state,
        const DeviceMoERebalanceApplyStatusView *copy_status,
        const DeviceMoERebalancePlanEntryView *plan_entries,
        uint32_t plan_capacity,
        const DeviceMoERebalanceApplyStatusView *gathered_copy_status,
        DeviceMoERebalanceConfigView config,
        uint32_t command_buffer_count)
    {
        if (blockIdx.x != 0 || threadIdx.x != 0 || !state)
            return;
        if (!rebalance_config_ok(config))
        {
            state->last_error_code = kDeviceMoERebalanceStatusInvalidConfig;
            return;
        }
        if (!rebalance_graph_controller_state_ok(state, config))
            init_rebalance_graph_controller_state_device(state, config);
        ++state->maintenance_launches;
        const uint32_t wave_index =
            rebalance_active_command_wave_index(state, config, command_buffer_count);
        if (command_header)
            command_header += wave_index;
        if (wave_state)
            wave_state += wave_index;
        if (!command_header || !wave_state || !copy_status ||
            !plan_entries || plan_capacity == 0u || !gathered_copy_status ||
            !rebalance_command_header_ok(command_header, config))
        {
            state->last_error_code = kDeviceMoERebalanceStatusInvalidRuntime;
            return;
        }
        const uint32_t command_count =
            command_header->command_count <
                    (command_header->command_capacity < plan_capacity
                         ? command_header->command_capacity
                         : plan_capacity)
                ? command_header->command_count
                : (command_header->command_capacity < plan_capacity
                       ? command_header->command_capacity
                       : plan_capacity);
        if (command_header->epoch == 0u || command_count == 0u)
        {
            if (wave_index < state->wave_count)
            {
                reset_rebalance_wave_progress_device(state->waves[wave_index]);
            }
            return;
        }

        auto &wave = state->waves[wave_index];
        const uint32_t requested_payload_slots = wave_state->requested_payload_slots;
        const uint32_t payload_bucket_slots = wave_state->payload_bucket_slots;
        const uint32_t payload_bucket_index = wave_state->payload_bucket_index;
        const uint32_t payload_bucket_overflow = wave_state->payload_bucket_overflow;
        if (wave.state == kDeviceMoERebalanceLifecycleReadyToApply &&
            wave.epoch != command_header->epoch)
        {
            state->last_error_code = kDeviceMoERebalanceStatusMissingTransferCompletion;
            wave.error_code = kDeviceMoERebalanceStatusMissingTransferCompletion;
            wave.state = kDeviceMoERebalanceLifecycleError;
            return;
        }
        const auto *wave_plan_entries =
            plan_entries + static_cast<unsigned long long>(wave_index) * plan_capacity;
        bool transfer_complete = true;
        uint32_t copied_arrivals_total = 0u;
        for (uint32_t participant = 0; participant < config.participant_count; ++participant)
        {
            const auto &peer_status = gathered_copy_status[participant];
            if (!rebalance_copy_status_ok(peer_status))
            {
                transfer_complete = false;
                break;
            }
            copied_arrivals_total += peer_status.copied_arrivals;
            const uint32_t expected =
                rebalance_expected_payload_arrivals(
                    wave_plan_entries,
                    command_count,
                    participant,
                    config);
            if (peer_status.copied_arrivals < expected)
            {
                transfer_complete = false;
                break;
            }
        }

        wave.magic = kDeviceMoERebalanceMagic;
        wave.version = kDeviceMoERebalanceVersion;
        wave.epoch = command_header->epoch;
        wave.state = kDeviceMoERebalanceLifecycleTransferInFlight;
        wave.planned_start_layer = wave_state->planned_start_layer;
        wave.planned_layer_count = wave_state->planned_layer_count;
        wave.command_count = command_count;
        wave.copied_arrivals = copied_arrivals_total;
        wave.applied_arrivals = 0u;
        wave.applied_layer_count = 0u;
        wave.error_code = 0u;
        wave.requested_payload_slots = requested_payload_slots;
        wave.payload_bucket_slots = payload_bucket_slots;
        wave.payload_bucket_index = payload_bucket_index;
        wave.payload_bucket_overflow = payload_bucket_overflow;
        __threadfence();
        if (!transfer_complete)
        {
            state->last_error_code = kDeviceMoERebalanceStatusMissingTransferCompletion;
            wave.error_code = kDeviceMoERebalanceStatusMissingTransferCompletion;
            wave.state = kDeviceMoERebalanceLifecycleError;
            return;
        }
        wave.state = kDeviceMoERebalanceLifecycleReadyToApply;
        state->next_epoch =
            state->next_epoch > command_header->epoch + 1u ? state->next_epoch : command_header->epoch + 1u;
    }

    __device__ __forceinline__ bool rebalance_command_header_ok(
        const DeviceMoERebalanceCommandBufferHeaderView *command_header,
        const DeviceMoERebalanceConfigView &config)
    {
        return !command_header ||
               (command_header->magic == kDeviceMoERebalanceMagic &&
                command_header->version == kDeviceMoERebalanceVersion &&
                command_header->phase == kDeviceMoERebalancePhasePlanAssignments &&
                command_header->participant_id == config.participant_id &&
               command_header->participant_count == config.participant_count);
    }

    __device__ __forceinline__ bool rebalance_command_header_for_participant_ok(
        const DeviceMoERebalanceCommandBufferHeaderView &command_header,
        uint32_t participant,
        const DeviceMoERebalanceConfigView &config)
    {
        return command_header.magic == kDeviceMoERebalanceMagic &&
               command_header.version == kDeviceMoERebalanceVersion &&
               command_header.phase == kDeviceMoERebalancePhasePlanAssignments &&
               command_header.participant_id == participant &&
               command_header.participant_count == config.participant_count;
    }

    __device__ __forceinline__ uint32_t rebalance_command_count(
        const uint32_t *plan_count,
        uint32_t plan_capacity,
        const DeviceMoERebalanceCommandBufferHeaderView *command_header)
    {
        if (command_header)
        {
            const uint32_t header_capacity =
                command_header->command_capacity < plan_capacity
                    ? command_header->command_capacity
                    : plan_capacity;
            return command_header->command_count < header_capacity
                       ? command_header->command_count
                       : header_capacity;
        }
        return plan_count
                   ? ((*plan_count < plan_capacity) ? *plan_count : plan_capacity)
                   : 0u;
    }

    __device__ __forceinline__ unsigned long long rebalance_source_descriptor_local_index(
        const DeviceMoERebalanceConfigView &config,
        uint32_t command_buffer_count,
        uint32_t plan_capacity,
        uint32_t destination_participant,
        uint32_t command_buffer_index,
        uint32_t plan_index)
    {
        (void)config;
        return ((static_cast<unsigned long long>(destination_participant) *
                 static_cast<unsigned long long>(command_buffer_count) +
                 static_cast<unsigned long long>(command_buffer_index)) *
                static_cast<unsigned long long>(plan_capacity)) +
               static_cast<unsigned long long>(plan_index);
    }

    __global__ void project_rebalance_domain_commands_kernel(
        const DeviceMoERebalancePlanEntryView *gathered_plan_entries,
        const DeviceMoERebalanceCommandBufferHeaderView *gathered_command_headers,
        uint32_t plan_capacity,
        DeviceMoERebalancePlanEntryView *local_plan_entries,
        DeviceMoERebalanceCommandBufferHeaderView *local_command_headers,
        DeviceMoERebalanceStatusView *status,
        uint32_t payload_slot_capacity,
        DeviceMoERebalanceConfigView config,
        uint32_t command_buffer_count,
        const DeviceMoERebalanceWaveStateView *gathered_wave_states,
        DeviceMoERebalanceWaveStateView *local_wave_states)
    {
        if (!gathered_plan_entries ||
            !gathered_command_headers ||
            !local_plan_entries ||
            !local_command_headers ||
            !rebalance_config_ok(config) ||
            plan_capacity == 0u)
        {
            return;
        }

        const uint32_t metadata_buffer_count =
            rebalance_command_buffer_count(command_buffer_count);
        const uint32_t root_participant = config.root_participant;
        if (root_participant >= config.participant_count)
            return;

        for (uint32_t buffer_index = threadIdx.x;
             buffer_index < metadata_buffer_count;
             buffer_index += blockDim.x)
        {
            const auto &root_header =
                gathered_command_headers[static_cast<unsigned long long>(root_participant) *
                                             static_cast<unsigned long long>(metadata_buffer_count) +
                                         static_cast<unsigned long long>(buffer_index)];
            auto projected = DeviceMoERebalanceCommandBufferHeaderView{};
            projected.magic = kDeviceMoERebalanceMagic;
            projected.version = kDeviceMoERebalanceVersion;
            projected.phase = kDeviceMoERebalancePhasePlanAssignments;
            projected.command_capacity = plan_capacity;
            projected.participant_id = config.participant_id;
            projected.participant_count = config.participant_count;
            if (rebalance_command_header_for_participant_ok(
                    root_header,
                    root_participant,
                    config))
            {
                projected.epoch = root_header.epoch;
                const uint32_t header_capacity =
                    root_header.command_capacity < plan_capacity
                        ? root_header.command_capacity
                        : plan_capacity;
                projected.command_count =
                    root_header.command_count < header_capacity
                        ? root_header.command_count
                        : header_capacity;
            }
            local_command_headers[buffer_index] = projected;

            if (gathered_wave_states && local_wave_states)
            {
                const auto &root_wave =
                    gathered_wave_states[static_cast<unsigned long long>(root_participant) *
                                             static_cast<unsigned long long>(metadata_buffer_count) +
                                         static_cast<unsigned long long>(buffer_index)];
                auto projected_wave = DeviceMoERebalanceWaveStateView{};
                projected_wave.magic = kDeviceMoERebalanceMagic;
                projected_wave.version = kDeviceMoERebalanceVersion;
                projected_wave.command_capacity = plan_capacity;
                projected_wave.participant_id = config.participant_id;
                projected_wave.participant_count = config.participant_count;
                if (root_wave.magic == kDeviceMoERebalanceMagic &&
                    root_wave.version == kDeviceMoERebalanceVersion &&
                    root_wave.participant_id == root_participant &&
                    root_wave.participant_count == config.participant_count)
                {
                    projected_wave = root_wave;
                    projected_wave.command_capacity = plan_capacity;
                    projected_wave.participant_id = config.participant_id;
                    projected_wave.participant_count = config.participant_count;
                }
                local_wave_states[buffer_index] = projected_wave;
            }
        }

        const unsigned long long total_entries =
            static_cast<unsigned long long>(metadata_buffer_count) *
            static_cast<unsigned long long>(plan_capacity);
        for (unsigned long long idx = static_cast<unsigned long long>(threadIdx.x);
             idx < total_entries;
             idx += static_cast<unsigned long long>(blockDim.x))
        {
            local_plan_entries[idx] = DeviceMoERebalancePlanEntryView{};
        }
        __syncthreads();

        /**
         * @brief Compact root-domain transfer commands after checking physical source residency.
         *
         * The root planner decides which experts should move, but only a
         * participant with a resident descriptor in the active runtime bank can
         * pack the payload.  Copying owner-only root commands into every local
         * command buffer makes invalid sources look runnable until the pack
         * kernel dereferences empty descriptor payloads.  Filtering here keeps
         * the command buffer, wave payload metadata, and status masks aligned
         * with work that can really be transferred.
         */
        if (threadIdx.x == 0)
        {
            uint32_t total_command_count = 0u;
            uint32_t last_projected_epoch = 0u;
            uint32_t requested_by_source[kDeviceMoEMaxParticipants] = {};
            uint32_t payload_source_participant_mask = 0u;
            uint32_t payload_destination_participant_mask = 0u;
            uint64_t payload_edge_mask = 0ULL;
            uint32_t invalid_runtime_layers = 0u;
            for (uint32_t buffer_index = 0; buffer_index < metadata_buffer_count; ++buffer_index)
            {
                uint32_t output_count = 0u;
                uint32_t buffer_requested_by_source[kDeviceMoEMaxParticipants] = {};
                const auto &root_header =
                    gathered_command_headers[static_cast<unsigned long long>(root_participant) *
                                                 static_cast<unsigned long long>(metadata_buffer_count) +
                                             static_cast<unsigned long long>(buffer_index)];
                if (rebalance_command_header_for_participant_ok(
                        root_header,
                        root_participant,
                        config))
                {
                    const uint32_t header_capacity =
                        root_header.command_capacity < plan_capacity
                            ? root_header.command_capacity
                            : plan_capacity;
                    const uint32_t command_count =
                        root_header.command_count < header_capacity
                        ? root_header.command_count
                        : header_capacity;
                    if (root_header.epoch != 0u)
                        last_projected_epoch = root_header.epoch;
                    for (uint32_t plan_index = 0; plan_index < command_count; ++plan_index)
                    {
                        const unsigned long long root_plan_index =
                            (static_cast<unsigned long long>(root_participant) *
                                 static_cast<unsigned long long>(metadata_buffer_count) +
                             static_cast<unsigned long long>(buffer_index)) *
                                static_cast<unsigned long long>(plan_capacity) +
                            static_cast<unsigned long long>(plan_index);
                        const auto &plan = gathered_plan_entries[root_plan_index];
                        const bool source_in_range =
                            plan.source_participant < config.participant_count &&
                            plan.source_participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants);
                        const bool destination_in_range =
                            plan.destination_participant < config.participant_count &&
                            plan.destination_participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants);
                        const uint32_t source_participant_bit =
                            source_in_range
                                ? llaminar2::moe_rebalance_policy::participantBit(
                                      plan.source_participant)
                                : 0u;
                        const bool valid =
                            rebalance_plan_requires_payload(plan.op) &&
                            plan.layer < config.num_layers &&
                            plan.expert < config.num_experts &&
                            source_in_range &&
                            destination_in_range &&
                            plan.source_participant != plan.destination_participant &&
                            plan.payload_slot != kDeviceMoEInvalidSlot &&
                            (plan.source_resident_mask & source_participant_bit) != 0u;
                        if (!valid)
                        {
                            ++invalid_runtime_layers;
                            continue;
                        }
                        if (output_count >= plan_capacity)
                        {
                            ++invalid_runtime_layers;
                            continue;
                        }

                        const unsigned long long local_plan_index =
                            static_cast<unsigned long long>(buffer_index) *
                                static_cast<unsigned long long>(plan_capacity) +
                            static_cast<unsigned long long>(output_count);
                        local_plan_entries[local_plan_index] = plan;
                        ++output_count;
                        ++total_command_count;

                        const uint32_t requested = plan.payload_slot + 1u;
                        if (requested_by_source[plan.source_participant] < requested)
                            requested_by_source[plan.source_participant] = requested;
                        if (buffer_requested_by_source[plan.source_participant] < requested)
                            buffer_requested_by_source[plan.source_participant] = requested;
                        payload_source_participant_mask |=
                            llaminar2::moe_rebalance_policy::participantBit(
                                plan.source_participant);
                        payload_destination_participant_mask |=
                            llaminar2::moe_rebalance_policy::participantBit(
                                plan.destination_participant);
                        payload_edge_mask |=
                            llaminar2::moe_rebalance_policy::directedParticipantEdgeBit(
                                plan.source_participant,
                                plan.destination_participant,
                                static_cast<uint32_t>(kDeviceMoEMaxParticipants));
                    }
                }

                local_command_headers[buffer_index].command_count = output_count;
                uint32_t buffer_requested_payload_slots = 0u;
                for (uint32_t participant = 0;
                     participant < config.participant_count &&
                     participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants);
                     ++participant)
                {
                    if (buffer_requested_by_source[participant] > buffer_requested_payload_slots)
                        buffer_requested_payload_slots = buffer_requested_by_source[participant];
                }
                if (local_wave_states)
                {
                    auto wave = local_wave_states[buffer_index];
                    wave.requested_payload_slots = buffer_requested_payload_slots;
                    wave.payload_bucket_slots =
                        llaminar2::moe_rebalance_policy::payloadBucketSlots(
                            buffer_requested_payload_slots,
                            payload_slot_capacity);
                    wave.payload_bucket_index =
                        llaminar2::moe_rebalance_policy::payloadBucketIndex(
                            wave.payload_bucket_slots);
                    wave.payload_bucket_overflow =
                        buffer_requested_payload_slots > wave.payload_bucket_slots ? 1u : 0u;
                    local_wave_states[buffer_index] = wave;
                }
            }

            if (status)
            {
                uint32_t requested_payload_slots = 0u;
                for (uint32_t participant = 0;
                     participant < config.participant_count &&
                     participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants);
                     ++participant)
                {
                    if (requested_by_source[participant] > requested_payload_slots)
                        requested_payload_slots = requested_by_source[participant];
                }
                const uint32_t payload_bucket_slots =
                    llaminar2::moe_rebalance_policy::payloadBucketSlots(
                        requested_payload_slots,
                        payload_slot_capacity);
                const uint32_t payload_bucket_index =
                    llaminar2::moe_rebalance_policy::payloadBucketIndex(
                        payload_bucket_slots);
                const uint32_t payload_bucket_overflow =
                    requested_payload_slots > payload_bucket_slots ? 1u : 0u;

                if (status->magic != kDeviceMoERebalanceMagic ||
                    status->version != kDeviceMoERebalanceVersion)
                {
                    DeviceMoERebalanceStatusView zero_status{};
                    *status = zero_status;
                    status->magic = kDeviceMoERebalanceMagic;
                    status->version = kDeviceMoERebalanceVersion;
                    status->status_code = kDeviceMoERebalanceStatusOk;
                }
                if (status->status_code == kDeviceMoERebalanceStatusOk)
                {
                    status->windows_observed = status->windows_observed > 1u ? status->windows_observed : 1u;
                    status->planned_arrivals = total_command_count;
                    status->invalid_runtime_layers += invalid_runtime_layers;
                    if (total_command_count != 0u)
                    {
                        status->windows_applied = status->windows_applied > 1u ? status->windows_applied : 1u;
                        if (status->last_epoch == 0u)
                            status->last_epoch = last_projected_epoch;
                    }
                    else
                    {
                        status->windows_applied = 0u;
                    }
                    status->payload_bucket_requested_slots = requested_payload_slots;
                    status->payload_bucket_slots = payload_bucket_slots;
                    status->payload_bucket_index = payload_bucket_index;
                    status->payload_bucket_overflow = payload_bucket_overflow;
                    status->payload_source_participant_mask = payload_source_participant_mask;
                    status->payload_destination_participant_mask =
                        payload_destination_participant_mask;
                    status->payload_edge_mask = payload_edge_mask;
                }
            }
        }
    }

    __global__ void project_prefill_llep_domain_commands_kernel(
        const DeviceMoERebalancePlanEntryView *gathered_plan_entries,
        const DeviceMoERebalanceCommandBufferHeaderView *gathered_command_headers,
        uint32_t plan_capacity,
        DeviceMoERebalancePlanEntryView *local_plan_entries,
        uint32_t *local_plan_count,
        DeviceMoERebalanceCommandBufferHeaderView *local_command_headers,
        DeviceMoERebalanceStatusView *status,
        uint32_t payload_slot_capacity,
        DeviceMoERebalanceConfigView config,
        uint32_t command_buffer_count)
    {
        if (!gathered_plan_entries ||
            !gathered_command_headers ||
            !local_plan_entries ||
            !local_plan_count ||
            !local_command_headers ||
            !status ||
            !rebalance_config_ok(config) ||
            plan_capacity == 0u ||
            payload_slot_capacity == 0u)
        {
            return;
        }

        const uint32_t metadata_buffer_count =
            rebalance_command_buffer_count(command_buffer_count);
        const unsigned long long total_entries =
            static_cast<unsigned long long>(metadata_buffer_count) *
            static_cast<unsigned long long>(plan_capacity);
        for (unsigned long long idx = static_cast<unsigned long long>(threadIdx.x);
             idx < total_entries;
             idx += static_cast<unsigned long long>(blockDim.x))
        {
            local_plan_entries[idx] = DeviceMoERebalancePlanEntryView{};
        }
        for (uint32_t buffer_index = threadIdx.x;
             buffer_index < metadata_buffer_count;
             buffer_index += blockDim.x)
        {
            DeviceMoERebalanceCommandBufferHeaderView header{};
            header.magic = kDeviceMoERebalanceMagic;
            header.version = kDeviceMoERebalanceVersion;
            header.phase = kDeviceMoERebalancePhasePlanAssignments;
            header.command_capacity = plan_capacity;
            header.participant_id = config.participant_id;
            header.participant_count = config.participant_count;
            local_command_headers[buffer_index] = header;
            local_plan_count[buffer_index] = 0u;
        }
        __syncthreads();

        if (threadIdx.x != 0u)
            return;

        const DeviceMoERebalanceStatusView prior_status = *status;
        DeviceMoERebalanceStatusView projected_status{};
        projected_status.magic = kDeviceMoERebalanceMagic;
        projected_status.version = kDeviceMoERebalanceVersion;
        projected_status.status_code =
            (prior_status.magic == kDeviceMoERebalanceMagic &&
             prior_status.version == kDeviceMoERebalanceVersion)
                ? prior_status.status_code
                : kDeviceMoERebalanceStatusOk;
        projected_status.windows_observed = 1u;
        projected_status.plan_overflow =
            (prior_status.magic == kDeviceMoERebalanceMagic &&
             prior_status.version == kDeviceMoERebalanceVersion)
                ? prior_status.plan_overflow
                : 0u;
        projected_status.payload_bucket_overflow =
            (prior_status.magic == kDeviceMoERebalanceMagic &&
             prior_status.version == kDeviceMoERebalanceVersion)
                ? prior_status.payload_bucket_overflow
                : 0u;

        uint32_t total_command_count = 0u;
        uint32_t last_epoch = 0u;
        uint32_t requested_payload_slots = 0u;
        uint32_t payload_source_participant_mask = 0u;
        uint32_t payload_destination_participant_mask = 0u;
        uint64_t payload_edge_mask = 0ULL;

        for (uint32_t buffer_index = 0u;
             buffer_index < metadata_buffer_count;
             ++buffer_index)
        {
            uint32_t source_payload_counts[kDeviceMoEMaxParticipants] = {};
            uint32_t destination_slot_counts[kDeviceMoEMaxParticipants] = {};
            uint32_t output_count = 0u;
            uint32_t buffer_epoch = 0u;

            for (uint32_t participant = 0u;
                 participant < config.participant_count &&
                 participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants);
                 ++participant)
            {
                const unsigned long long participant_buffer_base =
                    static_cast<unsigned long long>(participant) *
                        static_cast<unsigned long long>(metadata_buffer_count) +
                    static_cast<unsigned long long>(buffer_index);
                const auto &header =
                    gathered_command_headers[participant_buffer_base];
                if (!rebalance_command_header_for_participant_ok(
                        header,
                        participant,
                        config))
                {
                    continue;
                }
                if (header.epoch != 0u)
                    buffer_epoch = max(buffer_epoch, header.epoch);
                const uint32_t command_count =
                    min(header.command_count,
                        min(header.command_capacity, plan_capacity));
                for (uint32_t plan_index = 0u;
                     plan_index < command_count;
                     ++plan_index)
                {
                    const auto &plan =
                        gathered_plan_entries[
                            participant_buffer_base *
                                static_cast<unsigned long long>(plan_capacity) +
                            static_cast<unsigned long long>(plan_index)];
                    const uint32_t source_participant_bit =
                        llaminar2::moe_rebalance_policy::participantBit(
                            plan.source_participant);
                    const bool valid =
                        rebalance_plan_requires_payload(plan.op) &&
                        plan.layer < config.num_layers &&
                        plan.expert < config.num_experts &&
                        plan.source_participant < config.participant_count &&
                        plan.destination_participant == participant &&
                        plan.destination_participant < config.participant_count &&
                        plan.source_participant != plan.destination_participant &&
                        plan.source_participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants) &&
                        plan.destination_participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants) &&
                        (plan.source_resident_mask & source_participant_bit) != 0u;
                    if (!valid)
                    {
                        ++projected_status.invalid_runtime_layers;
                        continue;
                    }

                    if (output_count >= plan_capacity ||
                        source_payload_counts[plan.source_participant] >= payload_slot_capacity ||
                        destination_slot_counts[plan.destination_participant] >= payload_slot_capacity)
                    {
                        projected_status.plan_overflow = 1u;
                        projected_status.payload_bucket_overflow = 1u;
                        continue;
                    }

                    DeviceMoERebalancePlanEntryView projected = plan;
                    projected.destination_slot =
                        destination_slot_counts[plan.destination_participant]++;
                    projected.payload_slot =
                        source_payload_counts[plan.source_participant]++;

                    local_plan_entries[
                        static_cast<unsigned long long>(buffer_index) *
                            static_cast<unsigned long long>(plan_capacity) +
                        static_cast<unsigned long long>(output_count)] =
                        projected;
                    ++output_count;

                    total_command_count += 1u;
                    requested_payload_slots =
                        max(requested_payload_slots, projected.payload_slot + 1u);
                    payload_source_participant_mask |=
                        llaminar2::moe_rebalance_policy::participantBit(
                            projected.source_participant);
                    payload_destination_participant_mask |=
                        llaminar2::moe_rebalance_policy::participantBit(
                            projected.destination_participant);
                    payload_edge_mask |=
                        llaminar2::moe_rebalance_policy::directedParticipantEdgeBit(
                            projected.source_participant,
                            projected.destination_participant,
                            static_cast<uint32_t>(kDeviceMoEMaxParticipants));
                }
            }

            local_plan_count[buffer_index] = output_count;
            local_command_headers[buffer_index].epoch = buffer_epoch;
            local_command_headers[buffer_index].command_count = output_count;
            last_epoch = max(last_epoch, buffer_epoch);
        }

        const uint32_t payload_bucket_slots =
            llaminar2::moe_rebalance_policy::payloadBucketSlots(
                requested_payload_slots,
                payload_slot_capacity);
        projected_status.last_epoch =
            prior_status.last_epoch != 0u ? prior_status.last_epoch : last_epoch;
        projected_status.planned_arrivals = prior_status.planned_arrivals;
        projected_status.windows_applied =
            prior_status.windows_applied != 0u
                ? prior_status.windows_applied
                : (prior_status.planned_arrivals != 0u ? 1u : 0u);
        projected_status.payload_bucket_requested_slots = requested_payload_slots;
        projected_status.payload_bucket_slots = payload_bucket_slots;
        projected_status.payload_bucket_index =
            llaminar2::moe_rebalance_policy::payloadBucketIndex(
                payload_bucket_slots);
        if (requested_payload_slots > payload_bucket_slots ||
            projected_status.plan_overflow != 0u)
        {
            projected_status.payload_bucket_overflow = 1u;
        }
        projected_status.payload_source_participant_mask =
            payload_source_participant_mask;
        projected_status.payload_destination_participant_mask =
            payload_destination_participant_mask;
        projected_status.payload_edge_mask = payload_edge_mask;
        projected_status.candidate_arrivals_considered = total_command_count;
        projected_status.llep_assignment_span_count =
            prior_status.llep_assignment_span_count;
        projected_status.llep_weight_transfer_count =
            prior_status.llep_weight_transfer_count;
        *status = projected_status;
    }

    __global__ void materialize_prefill_llep_transfer_commands_kernel(
        const DeviceMoELayerRuntimeView *__restrict__ runtime,
        DeviceMoERebalancePlanEntryView *__restrict__ plan_entries,
        uint32_t *__restrict__ plan_count,
        uint32_t plan_capacity,
        DeviceMoERebalanceCommandBufferHeaderView *__restrict__ command_headers,
        DeviceMoERebalanceStatusView *__restrict__ status,
        DeviceMoERebalanceConfigView config,
        uint32_t payload_slot_capacity,
        uint32_t layer_idx,
        uint32_t command_buffer_count)
    {
        __shared__ uint32_t source_payload_counts[kDeviceMoEMaxParticipants];
        __shared__ uint32_t destination_slot_counts[kDeviceMoEMaxParticipants];
        __shared__ uint32_t compact_command_count;
        __shared__ uint32_t overflow;
        __shared__ uint32_t valid;
        __shared__ uint32_t transfer_count;
        __shared__ uint32_t span_count;
        __shared__ uint32_t fast_payload_source_participant_mask;
        __shared__ uint32_t fast_payload_destination_participant_mask;
        __shared__ unsigned long long fast_payload_edge_mask;
        __shared__ uint32_t shared_transfer_experts[256];
        __shared__ uint32_t shared_transfer_sources[256];
        __shared__ uint32_t shared_transfer_destinations[256];

        const uint32_t lane = threadIdx.x;
        const uint32_t metadata_buffer_count =
            rebalance_command_buffer_count(command_buffer_count);

        for (uint32_t participant = lane;
             participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants);
             participant += blockDim.x)
        {
            source_payload_counts[participant] = 0u;
            destination_slot_counts[participant] = 0u;
        }

        for (uint32_t idx = lane; idx < metadata_buffer_count; idx += blockDim.x)
        {
            DeviceMoERebalanceCommandBufferHeaderView header{};
            header.magic = kDeviceMoERebalanceMagic;
            header.version = kDeviceMoERebalanceVersion;
            header.phase = kDeviceMoERebalancePhasePlanAssignments;
            header.command_capacity = plan_capacity;
            header.participant_id = config.participant_id;
            header.participant_count = config.participant_count;
            command_headers[idx] = header;
        }

        if (lane == 0u)
        {
            compact_command_count = 0u;
            overflow = 0u;
            valid = 0u;
            transfer_count = 0u;
            span_count = 0u;
            fast_payload_source_participant_mask = 0u;
            fast_payload_destination_participant_mask = 0u;
            fast_payload_edge_mask = 0ULL;
            if (plan_count)
                *plan_count = 0u;
            if (status)
            {
                *status = DeviceMoERebalanceStatusView{};
                status->magic = kDeviceMoERebalanceMagic;
                status->version = kDeviceMoERebalanceVersion;
                status->status_code = kDeviceMoERebalanceStatusOk;
            }

            if (runtime &&
                plan_entries &&
                plan_count &&
                command_headers &&
                status &&
                rebalance_config_ok(config) &&
                payload_slot_capacity > 0u &&
                plan_capacity > 0u &&
                layer_idx < config.num_layers &&
                runtime->active_bank <= 1u &&
                runtime->participant_count == config.participant_count &&
                runtime->participant_count <= static_cast<uint32_t>(kDeviceMoEMaxParticipants) &&
                runtime->reserved_ptrs[2])
            {
                transfer_count =
                    runtime->reserved_u64[3] > 0xffffffffULL
                        ? 0xffffffffu
                        : static_cast<uint32_t>(runtime->reserved_u64[3]);
                span_count =
                    runtime->reserved_u64[2] > 0xffffffffULL
                        ? 0xffffffffu
                        : static_cast<uint32_t>(runtime->reserved_u64[2]);
                valid = 1u;
            }
            else if (status)
            {
                status->status_code = kDeviceMoERebalanceStatusInvalidRuntime;
                status->invalid_runtime_layers = 1u;
            }
        }
        __syncthreads();

        if (valid == 0u)
            return;

    const auto *transfers =
        static_cast<const llaminar2::least_loaded_ep::LeastLoadedExpertWeightTransfer *>(
            runtime->reserved_ptrs[2]);

    if (transfer_count <= plan_capacity &&
        transfer_count <= payload_slot_capacity &&
        transfer_count <= static_cast<uint32_t>(blockDim.x))
    {
        if (lane < transfer_count)
        {
            const auto transfer = transfers[lane];
            shared_transfer_experts[lane] = transfer.expert;
            shared_transfer_sources[lane] = transfer.source_participant;
            shared_transfer_destinations[lane] = transfer.destination_participant;
        }
        __syncthreads();

        for (uint32_t transfer_idx = lane; transfer_idx < transfer_count; transfer_idx += blockDim.x)
        {
            const uint32_t transfer_expert = shared_transfer_experts[transfer_idx];
            const uint32_t transfer_source = shared_transfer_sources[transfer_idx];
            const uint32_t transfer_destination = shared_transfer_destinations[transfer_idx];
            uint32_t resident_mask = 0u;
            if (runtime->active_bank <= 1u &&
                transfer_expert < static_cast<uint32_t>(kDeviceMoEMaxExperts))
            {
                resident_mask =
                    runtime->banks[runtime->active_bank]
                        .resident_participant_mask[transfer_expert];
            }
            const uint32_t transfer_source_bit =
                llaminar2::moe_rebalance_policy::participantBit(
                    transfer_source);
            const bool transfer_valid =
                transfer_expert < config.num_experts &&
                transfer_source < config.participant_count &&
                transfer_destination < config.participant_count &&
                transfer_source < static_cast<uint32_t>(kDeviceMoEMaxParticipants) &&
                transfer_destination < static_cast<uint32_t>(kDeviceMoEMaxParticipants) &&
                transfer_source != transfer_destination &&
                (resident_mask & transfer_source_bit) != 0u;
            if (!transfer_valid)
            {
                atomicOr(&overflow, 1u);
                continue;
            }

            uint32_t output_idx = 0u;
            uint32_t destination_slot = 0u;
            uint32_t payload_slot = 0u;
            for (uint32_t prev_idx = 0u; prev_idx < transfer_idx; ++prev_idx)
            {
                const uint32_t prev_expert = shared_transfer_experts[prev_idx];
                const uint32_t prev_source = shared_transfer_sources[prev_idx];
                const uint32_t prev_destination = shared_transfer_destinations[prev_idx];
                const bool prev_valid =
                    prev_expert < config.num_experts &&
                    prev_source < config.participant_count &&
                    prev_destination < config.participant_count &&
                    prev_source < static_cast<uint32_t>(kDeviceMoEMaxParticipants) &&
                    prev_destination < static_cast<uint32_t>(kDeviceMoEMaxParticipants) &&
                    prev_source != prev_destination;
                if (!prev_valid)
                    continue;
                ++output_idx;
                if (prev_destination == transfer_destination)
                    ++destination_slot;
                if (prev_source == transfer_source)
                    ++payload_slot;
            }

            DeviceMoERebalancePlanEntryView entry{};
            entry.op = kDeviceMoERebalancePlanExpertPayloadArrival;
            entry.layer = layer_idx;
            entry.expert = transfer_expert;
            entry.source_participant = transfer_source;
            entry.destination_participant = transfer_destination;
            entry.source_resident_mask = resident_mask;
            entry.destination_slot = destination_slot;
            entry.payload_slot = payload_slot;
            plan_entries[output_idx] = entry;
            atomicMax(source_payload_counts + transfer_source, payload_slot + 1u);
            atomicOr(
                &fast_payload_source_participant_mask,
                llaminar2::moe_rebalance_policy::participantBit(transfer_source));
            atomicOr(
                &fast_payload_destination_participant_mask,
                llaminar2::moe_rebalance_policy::participantBit(transfer_destination));
            atomicOr(
                &fast_payload_edge_mask,
                static_cast<unsigned long long>(
                    llaminar2::moe_rebalance_policy::directedParticipantEdgeBit(
                        transfer_source,
                        transfer_destination,
                        static_cast<uint32_t>(kDeviceMoEMaxParticipants))));
            atomicAdd(&compact_command_count, 1u);
        }
        __syncthreads();

        if (lane == 0u)
        {
            const uint32_t command_count =
                compact_command_count < plan_capacity
                    ? compact_command_count
                    : plan_capacity;
            *plan_count = command_count;
            command_headers[0].command_count = command_count;
            command_headers[0].epoch = runtime ? runtime->active_epoch : 0u;

            uint32_t requested_payload_slots = 0u;
            for (uint32_t participant = 0u;
                 participant < config.participant_count &&
                 participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants);
                 ++participant)
            {
                if (requested_payload_slots < source_payload_counts[participant])
                    requested_payload_slots = source_payload_counts[participant];
            }
            const uint32_t payload_bucket_slots =
                llaminar2::moe_rebalance_policy::payloadBucketSlots(
                    requested_payload_slots,
                    payload_slot_capacity);

            status->last_epoch = command_headers[0].epoch;
            status->planned_arrivals = command_count;
            status->plan_overflow =
                overflow != 0u || transfer_count > command_count ? 1u : 0u;
            status->payload_bucket_requested_slots = requested_payload_slots;
            status->payload_bucket_slots = payload_bucket_slots;
            status->payload_bucket_index =
                llaminar2::moe_rebalance_policy::payloadBucketIndex(
                    payload_bucket_slots);
            status->payload_bucket_overflow =
                requested_payload_slots > payload_bucket_slots ||
                        status->plan_overflow != 0u
                    ? 1u
                    : 0u;
            status->payload_source_participant_mask =
                fast_payload_source_participant_mask;
            status->payload_destination_participant_mask =
                fast_payload_destination_participant_mask;
            status->payload_edge_mask = static_cast<uint64_t>(fast_payload_edge_mask);
            status->llep_assignment_span_count = span_count;
            status->llep_weight_transfer_count = transfer_count;
        }
        return;
    }

    if (lane == 0u)
    {
        for (uint32_t transfer_idx = 0u; transfer_idx < transfer_count; ++transfer_idx)
            {
                const auto transfer = transfers[transfer_idx];
                uint32_t resident_mask = 0u;
                if (runtime->active_bank <= 1u &&
                    transfer.expert < static_cast<uint32_t>(kDeviceMoEMaxExperts))
                {
                    resident_mask =
                        runtime->banks[runtime->active_bank]
                            .resident_participant_mask[transfer.expert];
                }
                const uint32_t transfer_source_bit =
                    llaminar2::moe_rebalance_policy::participantBit(
                        transfer.source_participant);
                const bool transfer_valid =
                    transfer.expert < config.num_experts &&
                    transfer.source_participant < config.participant_count &&
                    transfer.destination_participant < config.participant_count &&
                    transfer.source_participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants) &&
                    transfer.destination_participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants) &&
                    transfer.source_participant != transfer.destination_participant &&
                    (resident_mask & transfer_source_bit) != 0u;
                if (!transfer_valid)
                {
                    ++overflow;
                    continue;
                }

                const uint32_t destination_slot =
                    destination_slot_counts[transfer.destination_participant]++;
                const uint32_t payload_slot =
                    source_payload_counts[transfer.source_participant]++;
                if (destination_slot >= payload_slot_capacity ||
                    payload_slot >= payload_slot_capacity)
                {
                    ++overflow;
                    continue;
                }

                const uint32_t output_idx = compact_command_count++;
                if (output_idx >= plan_capacity)
                {
                    ++overflow;
                    continue;
                }

                DeviceMoERebalancePlanEntryView entry{};
                entry.op = kDeviceMoERebalancePlanExpertPayloadArrival;
                entry.layer = layer_idx;
                entry.expert = transfer.expert;
                entry.source_participant = transfer.source_participant;
                entry.destination_participant = transfer.destination_participant;
                entry.source_resident_mask = resident_mask;
                entry.destination_slot = destination_slot;
                entry.payload_slot = payload_slot;
                plan_entries[output_idx] = entry;
            }

            const uint32_t command_count =
                compact_command_count < plan_capacity
                    ? compact_command_count
                    : plan_capacity;
            *plan_count = command_count;
            command_headers[0].command_count = command_count;
            command_headers[0].epoch = runtime ? runtime->active_epoch : 0u;

            uint32_t requested_by_source[kDeviceMoEMaxParticipants] = {};
            uint32_t requested_payload_slots = 0u;
            uint32_t payload_source_participant_mask = 0u;
            uint32_t payload_destination_participant_mask = 0u;
            uint64_t payload_edge_mask = 0ULL;
            for (uint32_t idx = 0u; idx < command_count; ++idx)
            {
                const auto &entry = plan_entries[idx];
                const uint32_t requested = entry.payload_slot + 1u;
                if (entry.source_participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants) &&
                    requested_by_source[entry.source_participant] < requested)
                {
                    requested_by_source[entry.source_participant] = requested;
                }
                payload_source_participant_mask |=
                    llaminar2::moe_rebalance_policy::participantBit(
                        entry.source_participant);
                payload_destination_participant_mask |=
                    llaminar2::moe_rebalance_policy::participantBit(
                        entry.destination_participant);
                payload_edge_mask |=
                    llaminar2::moe_rebalance_policy::directedParticipantEdgeBit(
                        entry.source_participant,
                        entry.destination_participant,
                        static_cast<uint32_t>(kDeviceMoEMaxParticipants));
            }
            for (uint32_t participant = 0u;
                 participant < config.participant_count &&
                 participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants);
                 ++participant)
            {
                if (requested_payload_slots < requested_by_source[participant])
                    requested_payload_slots = requested_by_source[participant];
            }
            const uint32_t payload_bucket_slots =
                llaminar2::moe_rebalance_policy::payloadBucketSlots(
                    requested_payload_slots,
                    payload_slot_capacity);

            status->last_epoch = command_headers[0].epoch;
            status->planned_arrivals = command_count;
            status->plan_overflow =
                overflow != 0u || transfer_count > command_count ? 1u : 0u;
            status->payload_bucket_requested_slots = requested_payload_slots;
            status->payload_bucket_slots = payload_bucket_slots;
            status->payload_bucket_index =
                llaminar2::moe_rebalance_policy::payloadBucketIndex(
                    payload_bucket_slots);
            status->payload_bucket_overflow =
                requested_payload_slots > payload_bucket_slots ||
                        status->plan_overflow != 0u
                    ? 1u
                    : 0u;
            status->payload_source_participant_mask =
                payload_source_participant_mask;
            status->payload_destination_participant_mask =
                payload_destination_participant_mask;
            status->payload_edge_mask = payload_edge_mask;
            status->llep_assignment_span_count = span_count;
            status->llep_weight_transfer_count = transfer_count;
        }
    }

    __global__ void pack_rebalance_source_descriptors_kernel(
        const DeviceMoELayerRuntimeView *runtime_layers,
        const DeviceMoERebalancePlanEntryView *plan_entries,
        const DeviceMoERebalanceCommandBufferHeaderView *command_headers,
        uint32_t plan_capacity,
        DeviceMoEExpertDirectoryEntryView *local_source_descriptors,
        DeviceMoERebalanceConfigView config,
        DeviceMoERebalanceGraphControllerStateView *controller_state,
        uint32_t command_buffer_count)
    {
        (void)controller_state;
        if (!runtime_layers ||
            !plan_entries ||
            !command_headers ||
            !local_source_descriptors ||
            !rebalance_config_ok(config) ||
            plan_capacity == 0u)
        {
            return;
        }

        const uint32_t metadata_buffer_count =
            rebalance_command_buffer_count(command_buffer_count);
        const unsigned long long total =
            static_cast<unsigned long long>(config.participant_count) *
            static_cast<unsigned long long>(metadata_buffer_count) *
            static_cast<unsigned long long>(plan_capacity);
        unsigned long long idx = static_cast<unsigned long long>(blockIdx.x) *
                                 static_cast<unsigned long long>(blockDim.x) +
                                 static_cast<unsigned long long>(threadIdx.x);
        while (idx < total)
        {
            local_source_descriptors[idx] = DeviceMoEExpertDirectoryEntryView{};

            const uint32_t plan_index =
                static_cast<uint32_t>(idx % static_cast<unsigned long long>(plan_capacity));
            const unsigned long long tmp =
                idx / static_cast<unsigned long long>(plan_capacity);
            const uint32_t command_buffer_index =
                static_cast<uint32_t>(tmp % static_cast<unsigned long long>(metadata_buffer_count));
            const uint32_t destination_participant =
                static_cast<uint32_t>(tmp / static_cast<unsigned long long>(metadata_buffer_count));
            const auto &header =
                command_headers[static_cast<unsigned long long>(command_buffer_index)];
            if (!rebalance_command_header_ok(&header, config))
            {
                idx += static_cast<unsigned long long>(blockDim.x) *
                       static_cast<unsigned long long>(gridDim.x);
                continue;
            }
            const uint32_t header_capacity =
                header.command_capacity < plan_capacity ? header.command_capacity : plan_capacity;
            const uint32_t command_count =
                header.command_count < header_capacity ? header.command_count : header_capacity;
            if (plan_index >= command_count)
            {
                idx += static_cast<unsigned long long>(blockDim.x) *
                       static_cast<unsigned long long>(gridDim.x);
                continue;
            }

            const auto &plan =
                plan_entries[static_cast<unsigned long long>(command_buffer_index) *
                                 static_cast<unsigned long long>(plan_capacity) +
                             static_cast<unsigned long long>(plan_index)];
            if (rebalance_plan_requires_payload(plan.op) &&
                plan.source_participant == config.participant_id &&
                plan.destination_participant == destination_participant &&
                plan.layer < config.num_layers &&
                plan.expert < config.num_experts)
            {
                local_source_descriptors[idx] = make_rebalance_source_entry(
                    runtime_layers,
                    config,
                    plan.layer,
                    plan.expert);
            }

            idx += static_cast<unsigned long long>(blockDim.x) *
                   static_cast<unsigned long long>(gridDim.x);
        }
    }

    __device__ void rebalance_copy_bytes(const void *src, const void *dst_const, unsigned long long bytes)
    {
        const auto *src_bytes = static_cast<const uint8_t *>(src);
        auto *dst_bytes = const_cast<uint8_t *>(static_cast<const uint8_t *>(dst_const));
        if (!src_bytes || !dst_bytes || bytes == 0ULL)
            return;
        if ((((reinterpret_cast<uintptr_t>(src_bytes) |
               reinterpret_cast<uintptr_t>(dst_bytes) |
               static_cast<uintptr_t>(bytes)) &
              0xFULL) == 0u))
        {
            const auto *src_vec = reinterpret_cast<const uint4 *>(src_bytes);
            auto *dst_vec = reinterpret_cast<uint4 *>(dst_bytes);
            const unsigned long long vec_count = bytes / sizeof(uint4);
            for (unsigned long long i = threadIdx.x; i < vec_count; i += blockDim.x)
                dst_vec[i] = src_vec[i];
            return;
        }
        if ((((reinterpret_cast<uintptr_t>(src_bytes) |
               reinterpret_cast<uintptr_t>(dst_bytes) |
               static_cast<uintptr_t>(bytes)) &
              0x7ULL) == 0u))
        {
            const auto *src_vec = reinterpret_cast<const uint2 *>(src_bytes);
            auto *dst_vec = reinterpret_cast<uint2 *>(dst_bytes);
            const unsigned long long vec_count = bytes / sizeof(uint2);
            for (unsigned long long i = threadIdx.x; i < vec_count; i += blockDim.x)
                dst_vec[i] = src_vec[i];
            return;
        }
        if ((((reinterpret_cast<uintptr_t>(src_bytes) |
               reinterpret_cast<uintptr_t>(dst_bytes) |
               static_cast<uintptr_t>(bytes)) &
              0x3ULL) == 0u))
        {
            const auto *src_vec = reinterpret_cast<const uint32_t *>(src_bytes);
            auto *dst_vec = reinterpret_cast<uint32_t *>(dst_bytes);
            const unsigned long long vec_count = bytes / sizeof(uint32_t);
            for (unsigned long long i = threadIdx.x; i < vec_count; i += blockDim.x)
                dst_vec[i] = src_vec[i];
            return;
        }
        if ((((reinterpret_cast<uintptr_t>(src_bytes) |
               reinterpret_cast<uintptr_t>(dst_bytes) |
               static_cast<uintptr_t>(bytes)) &
              0x1ULL) == 0u))
        {
            const auto *src_vec = reinterpret_cast<const uint16_t *>(src_bytes);
            auto *dst_vec = reinterpret_cast<uint16_t *>(dst_bytes);
            const unsigned long long vec_count = bytes / sizeof(uint16_t);
            for (unsigned long long i = threadIdx.x; i < vec_count; i += blockDim.x)
                dst_vec[i] = src_vec[i];
            return;
        }
        for (unsigned long long i = threadIdx.x; i < bytes; i += blockDim.x)
            dst_bytes[i] = src_bytes[i];
    }

    __device__ __forceinline__ unsigned long long rebalance_projection_payload_bytes(
        const DeviceNativeVNNIMatrixDescView &desc,
        const DeviceMoEExpertDirectoryEntryView &entry)
    {
        (void)entry;
        uint8_t payload_bytes_per_block = 0u;
        uint8_t is_asymmetric = 0u;
        uint8_t has_emins = 0u;
        if (!rebalance_projection_format(desc, payload_bytes_per_block, is_asymmetric, has_emins))
            return 0ULL;
        const unsigned long long blocks = rebalance_block_count(desc);
        const unsigned long long scales_bytes = blocks * sizeof(uint16_t);
        return blocks * static_cast<unsigned long long>(payload_bytes_per_block) +
               scales_bytes +
               (is_asymmetric != 0u ? scales_bytes : 0ULL) +
               (has_emins != 0u ? blocks * sizeof(uint32_t) : 0ULL);
    }

    __device__ __forceinline__ unsigned long long rebalance_expert_payload_bytes(
        const DeviceMoEExpertDirectoryEntryView &entry)
    {
        return rebalance_projection_payload_bytes(entry.descriptor.gate, entry) +
               rebalance_projection_payload_bytes(entry.descriptor.up, entry) +
               rebalance_projection_payload_bytes(entry.descriptor.down, entry);
    }

    __device__ unsigned long long rebalance_pack_projection_to_payload(
        const DeviceNativeVNNIMatrixDescView &src,
        const DeviceMoEExpertDirectoryEntryView &entry,
        uint8_t *payload)
    {
        (void)entry;
        uint8_t payload_bytes_per_block = 0u;
        uint8_t is_asymmetric = 0u;
        uint8_t has_emins = 0u;
        if (!rebalance_projection_format(src, payload_bytes_per_block, is_asymmetric, has_emins))
            return 0ULL;
        const unsigned long long blocks = rebalance_block_count(src);
        const unsigned long long payload_bytes =
            blocks * static_cast<unsigned long long>(payload_bytes_per_block);
        const unsigned long long scales_bytes = blocks * sizeof(uint16_t);
        const unsigned long long mins_bytes =
            is_asymmetric != 0u ? scales_bytes : 0ULL;
        const unsigned long long emins_bytes =
            has_emins != 0u ? blocks * sizeof(uint32_t) : 0ULL;

        unsigned long long offset = 0ULL;
        rebalance_copy_bytes(src.payload, payload + offset, payload_bytes);
        offset += payload_bytes;
        rebalance_copy_bytes(src.scales, payload + offset, scales_bytes);
        offset += scales_bytes;
        if (mins_bytes > 0ULL)
        {
            rebalance_copy_bytes(src.mins, payload + offset, mins_bytes);
            offset += mins_bytes;
        }
        if (emins_bytes > 0ULL)
        {
            rebalance_copy_bytes(src.emins, payload + offset, emins_bytes);
            offset += emins_bytes;
        }
        return offset;
    }

    __device__ unsigned long long rebalance_unpack_projection_from_payload(
        const uint8_t *payload,
        const DeviceNativeVNNIMatrixDescView &dst,
        const DeviceMoEExpertDirectoryEntryView &entry)
    {
        (void)entry;
        uint8_t payload_bytes_per_block = 0u;
        uint8_t is_asymmetric = 0u;
        uint8_t has_emins = 0u;
        if (!rebalance_projection_format(dst, payload_bytes_per_block, is_asymmetric, has_emins))
            return 0ULL;
        const unsigned long long blocks = rebalance_block_count(dst);
        const unsigned long long payload_bytes =
            blocks * static_cast<unsigned long long>(payload_bytes_per_block);
        const unsigned long long scales_bytes = blocks * sizeof(uint16_t);
        const unsigned long long mins_bytes =
            is_asymmetric != 0u ? scales_bytes : 0ULL;
        const unsigned long long emins_bytes =
            has_emins != 0u ? blocks * sizeof(uint32_t) : 0ULL;

        unsigned long long offset = 0ULL;
        rebalance_copy_bytes(payload + offset, dst.payload, payload_bytes);
        offset += payload_bytes;
        rebalance_copy_bytes(payload + offset, dst.scales, scales_bytes);
        offset += scales_bytes;
        if (mins_bytes > 0ULL)
        {
            rebalance_copy_bytes(payload + offset, dst.mins, mins_bytes);
            offset += mins_bytes;
        }
        if (emins_bytes > 0ULL)
        {
            rebalance_copy_bytes(payload + offset, dst.emins, emins_bytes);
            offset += emins_bytes;
        }
        return offset;
    }

    __global__ void pack_rebalance_collective_payloads_kernel(
        const DeviceMoERebalancePlanEntryView *gathered_plan_entries,
        const DeviceMoERebalanceCommandBufferHeaderView *gathered_command_headers,
        uint32_t plan_capacity,
        const DeviceMoEExpertDirectoryEntryView *local_directory,
        uint8_t *local_payload,
        uint32_t local_payload_slot_count,
        unsigned long long payload_slot_bytes,
        DeviceMoERebalanceConfigView config,
        DeviceMoERebalanceApplyStatusView *status,
        DeviceMoERebalanceGraphControllerStateView *controller_state,
        uint32_t command_buffer_count)
    {
        if (!status || !gathered_plan_entries || !gathered_command_headers ||
            !local_directory || !local_payload || plan_capacity == 0u ||
            payload_slot_bytes <= sizeof(DeviceMoEExpertDirectoryEntryView))
            return;
        const uint32_t global_slot = blockIdx.x;
        if (global_slot >= local_payload_slot_count)
            return;

    uint8_t *payload_slot =
        local_payload + static_cast<unsigned long long>(global_slot) * payload_slot_bytes;
    if (threadIdx.x == 0)
        *reinterpret_cast<DeviceMoEExpertDirectoryEntryView *>(payload_slot) =
            DeviceMoEExpertDirectoryEntryView{};
    __syncthreads();

    const uint32_t destination_participant = global_slot / plan_capacity;
    const uint32_t plan_index = global_slot % plan_capacity;
        if (destination_participant >= config.participant_count)
            return;
        const uint32_t metadata_buffer_count =
            rebalance_command_buffer_count(command_buffer_count);
        const uint32_t active_wave =
            rebalance_active_command_wave_index(controller_state, config, metadata_buffer_count);
        const unsigned long long metadata_participant_base =
            static_cast<unsigned long long>(destination_participant) *
            static_cast<unsigned long long>(metadata_buffer_count);
        const auto &header = gathered_command_headers[metadata_participant_base + active_wave];
        if (!rebalance_command_header_for_participant_ok(
                header,
                destination_participant,
                config))
            return;
        const uint32_t header_capacity =
            header.command_capacity < plan_capacity ? header.command_capacity : plan_capacity;
        const uint32_t count =
            header.command_count < header_capacity ? header.command_count : header_capacity;
        if (plan_index >= count)
            return;

        const auto &plan =
            gathered_plan_entries[(metadata_participant_base + active_wave) *
                                      static_cast<unsigned long long>(plan_capacity) +
                                  static_cast<unsigned long long>(plan_index)];
        const uint32_t source_bit =
            runtime_participant_bit(static_cast<int>(config.participant_id));
        if (plan.op == 0u ||
            !rebalance_plan_requires_payload(plan.op) ||
            plan.destination_participant != destination_participant ||
            plan.source_participant != config.participant_id ||
            plan.layer >= config.num_layers ||
            plan.expert >= config.num_experts)
        {
            return;
        }
        if ((plan.source_resident_mask & source_bit) == 0u)
        {
            if (threadIdx.x == 0)
                atomicAdd(&status->invalid_plan_entries, 1u);
            return;
        }

        const auto &src =
            local_directory[static_cast<unsigned long long>(plan.layer) *
                                static_cast<unsigned long long>(config.num_experts) +
                            static_cast<unsigned long long>(plan.expert)];
        if (!rebalance_source_entry_ready(src, config.participant_id, plan.layer, plan.expert))
        {
            if (threadIdx.x == 0)
                atomicAdd(&status->missing_source_descriptors, 1u);
            return;
        }
        const unsigned long long payload_data_bytes =
            payload_slot_bytes - sizeof(DeviceMoEExpertDirectoryEntryView);
        if (rebalance_expert_payload_bytes(src) > payload_data_bytes)
        {
            if (threadIdx.x == 0)
                atomicAdd(&status->descriptor_mismatches, 1u);
            return;
        }

        if (threadIdx.x == 0)
            *reinterpret_cast<DeviceMoEExpertDirectoryEntryView *>(payload_slot) = src;
        __syncthreads();

        unsigned long long offset = 0ULL;
        uint8_t *payload_data = payload_slot + sizeof(DeviceMoEExpertDirectoryEntryView);
        offset += rebalance_pack_projection_to_payload(src.descriptor.gate, src, payload_data + offset);
        offset += rebalance_pack_projection_to_payload(src.descriptor.up, src, payload_data + offset);
        offset += rebalance_pack_projection_to_payload(src.descriptor.down, src, payload_data + offset);
        (void)offset;
    }

    __global__ void pack_rebalance_compact_payloads_kernel(
        const DeviceMoERebalancePlanEntryView *plan_entries,
        const DeviceMoERebalanceCommandBufferHeaderView *command_headers,
        uint32_t plan_capacity,
        const DeviceMoEExpertDirectoryEntryView *local_source_descriptors,
        uint8_t *local_payload,
        uint32_t local_payload_slot_count,
        unsigned long long payload_slot_bytes,
        DeviceMoERebalanceConfigView config,
        DeviceMoERebalanceApplyStatusView *status,
        DeviceMoERebalanceGraphControllerStateView *controller_state,
        uint32_t command_buffer_count)
    {
        if (!status || !plan_entries || !command_headers ||
            !local_source_descriptors || !local_payload ||
            !rebalance_config_ok(config) || plan_capacity == 0u ||
            payload_slot_bytes <= sizeof(DeviceMoEExpertDirectoryEntryView))
            return;

        const uint32_t global_slot = blockIdx.x;
        if (global_slot >= local_payload_slot_count)
            return;

    uint8_t *payload_slot =
        local_payload + static_cast<unsigned long long>(global_slot) * payload_slot_bytes;
    if (threadIdx.x == 0)
        *reinterpret_cast<DeviceMoEExpertDirectoryEntryView *>(payload_slot) =
            DeviceMoEExpertDirectoryEntryView{};
    __syncthreads();

    const uint32_t metadata_buffer_count =
        rebalance_command_buffer_count(command_buffer_count);
        const uint32_t active_wave =
            rebalance_active_command_wave_index(controller_state, config, metadata_buffer_count);
        const auto &header = command_headers[active_wave];
        if (!rebalance_command_header_ok(&header, config))
            return;
        const uint32_t command_count =
            min(header.command_count,
                min(header.command_capacity, plan_capacity));
        const uint32_t source_bit =
            runtime_participant_bit(static_cast<int>(config.participant_id));

        __shared__ uint32_t shared_plan_index;
        if (threadIdx.x == 0)
            shared_plan_index = kDeviceMoEInvalidSlot;
        __syncthreads();
        for (uint32_t plan_index = threadIdx.x;
             plan_index < command_count;
             plan_index += blockDim.x)
        {
            const auto &plan =
                plan_entries[static_cast<unsigned long long>(active_wave) *
                                 static_cast<unsigned long long>(plan_capacity) +
                             static_cast<unsigned long long>(plan_index)];
            const bool matches_local_source_slot =
                rebalance_plan_requires_payload(plan.op) &&
                plan.source_participant == config.participant_id &&
                plan.destination_participant < config.participant_count &&
                plan.payload_slot == global_slot &&
                plan.layer < config.num_layers &&
                plan.expert < config.num_experts;
            if (matches_local_source_slot &&
                (plan.source_resident_mask & source_bit) == 0u)
            {
                atomicAdd(&status->invalid_plan_entries, 1u);
                continue;
            }
            if (matches_local_source_slot)
            {
                atomicCAS(&shared_plan_index, kDeviceMoEInvalidSlot, plan_index);
            }
        }
        __syncthreads();
        if (shared_plan_index == kDeviceMoEInvalidSlot)
            return;

        const auto &selected_plan =
            plan_entries[static_cast<unsigned long long>(active_wave) *
                             static_cast<unsigned long long>(plan_capacity) +
                         static_cast<unsigned long long>(shared_plan_index)];
        const auto &src = local_source_descriptors[
            rebalance_source_descriptor_local_index(
                config,
                metadata_buffer_count,
                plan_capacity,
                selected_plan.destination_participant,
                active_wave,
                shared_plan_index)];
        if (!rebalance_source_entry_ready(src,
                                          config.participant_id,
                                          selected_plan.layer,
                                          selected_plan.expert))
        {
            if (threadIdx.x == 0)
            {
                atomicAdd(&status->missing_source_descriptors, 1u);
            }
            return;
        }

        const unsigned long long payload_data_bytes =
            payload_slot_bytes - sizeof(DeviceMoEExpertDirectoryEntryView);
        if (rebalance_expert_payload_bytes(src) > payload_data_bytes)
        {
            if (threadIdx.x == 0)
                atomicAdd(&status->descriptor_mismatches, 1u);
            return;
        }

        if (threadIdx.x == 0)
            *reinterpret_cast<DeviceMoEExpertDirectoryEntryView *>(payload_slot) = src;
        __syncthreads();

        unsigned long long offset = 0ULL;
        uint8_t *payload_data = payload_slot + sizeof(DeviceMoEExpertDirectoryEntryView);
        offset += rebalance_pack_projection_to_payload(src.descriptor.gate, src, payload_data + offset);
        offset += rebalance_pack_projection_to_payload(src.descriptor.up, src, payload_data + offset);
        offset += rebalance_pack_projection_to_payload(src.descriptor.down, src, payload_data + offset);
        (void)offset;
    }

    __global__ void unpack_rebalance_collective_payloads_kernel(
        const DeviceMoERebalancePlanEntryView *plan_entries,
        const uint32_t *plan_count,
        uint32_t plan_capacity,
        const DeviceMoERebalanceCommandBufferHeaderView *command_header,
        const uint8_t *gathered_payload,
        uint32_t local_payload_slot_count,
        unsigned long long payload_slot_bytes,
        DeviceMoEExpertDirectoryEntryView *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        DeviceMoERebalanceConfigView config,
        DeviceMoERebalanceApplyStatusView *status,
        DeviceMoERebalanceGraphControllerStateView *controller_state,
        uint32_t command_buffer_count)
    {
        const uint32_t command_wave_index =
            rebalance_active_command_wave_index(controller_state, config, command_buffer_count);
        if (plan_entries)
            plan_entries += static_cast<unsigned long long>(command_wave_index) * plan_capacity;
        if (plan_count)
            plan_count += command_wave_index;
        if (command_header)
            command_header += command_wave_index;

        if (!status || !plan_entries || (!plan_count && !command_header) ||
            !gathered_payload || !local_transfer_slots ||
            plan_capacity == 0u ||
            payload_slot_bytes <= sizeof(DeviceMoEExpertDirectoryEntryView))
            return;

        const uint32_t count = rebalance_command_count(plan_count, plan_capacity, command_header);

        if (blockIdx.x == 0)
        {
            if (threadIdx.x == 0)
                atomicAdd(&status->plan_entries_seen, count);
            for (uint32_t plan_index = threadIdx.x;
                 plan_index < count;
                 plan_index += blockDim.x)
            {
                const auto &plan = plan_entries[plan_index];
                if (plan.op == 0u)
                    continue;
                if (!rebalance_plan_requires_payload(plan.op) ||
                    plan.layer >= config.num_layers ||
                    plan.expert >= config.num_experts ||
                    plan.source_participant >= config.participant_count ||
                    (plan.source_resident_mask &
                     runtime_participant_bit(static_cast<int>(plan.source_participant))) == 0u)
                {
                    atomicAdd(&status->invalid_plan_entries, 1u);
                    continue;
                }
                if (plan.destination_participant != config.participant_id)
                {
                    atomicAdd(&status->skipped_wrong_destination, 1u);
                    continue;
                }
                if (plan.destination_slot >= local_transfer_slot_count)
                {
                    atomicAdd(&status->missing_destination_slots, 1u);
                    continue;
                }
                if (plan.payload_slot >= local_payload_slot_count)
                {
                    atomicAdd(&status->invalid_plan_entries, 1u);
                    continue;
                }
            }
            return;
        }

        const uint32_t destination_slot = blockIdx.x - 1u;
        if (destination_slot >= local_transfer_slot_count)
            return;

        __shared__ uint32_t shared_plan_index;
        if (threadIdx.x == 0)
            shared_plan_index = kDeviceMoEInvalidSlot;
        __syncthreads();

        for (uint32_t plan_index = threadIdx.x;
             plan_index < count;
             plan_index += blockDim.x)
        {
            const auto &plan = plan_entries[plan_index];
            if (plan.op != 0u &&
                rebalance_plan_requires_payload(plan.op) &&
                plan.layer < config.num_layers &&
                plan.expert < config.num_experts &&
                plan.source_participant < config.participant_count &&
                (plan.source_resident_mask &
                 runtime_participant_bit(static_cast<int>(plan.source_participant))) != 0u &&
                plan.destination_participant == config.participant_id &&
                plan.destination_slot == destination_slot &&
                plan.payload_slot < local_payload_slot_count)
            {
                atomicMin(&shared_plan_index, plan_index);
            }
        }
        __syncthreads();
        if (shared_plan_index == kDeviceMoEInvalidSlot)
            return;

        const auto &plan = plan_entries[shared_plan_index];

        const unsigned long long gathered_slot =
            (static_cast<unsigned long long>(plan.source_participant) *
                 static_cast<unsigned long long>(local_payload_slot_count) +
             static_cast<unsigned long long>(plan.payload_slot));
        const uint8_t *payload_slot = gathered_payload + gathered_slot * payload_slot_bytes;
        const auto &src =
            *reinterpret_cast<const DeviceMoEExpertDirectoryEntryView *>(payload_slot);
        const unsigned long long payload_data_bytes =
            payload_slot_bytes - sizeof(DeviceMoEExpertDirectoryEntryView);
        auto &dst = local_transfer_slots[destination_slot];
        if (threadIdx.x == 0)
        {
            dst.layer = kDeviceMoEInvalidSlot;
            dst.expert = kDeviceMoEInvalidSlot;
            dst.descriptor.logical_expert_id = -1;
            dst.resident_mask &= ~runtime_participant_bit(static_cast<int>(config.participant_id));
            dst.flags &= ~(kDeviceMoEDirectoryFlagResident |
                           kDeviceMoEDirectoryFlagCopyComplete);
        }
        __syncthreads();

        if (!rebalance_source_entry_ready(src, plan.source_participant, plan.layer, plan.expert))
        {
            if (threadIdx.x == 0)
                atomicAdd(&status->missing_source_descriptors, 1u);
            return;
        }
        if (!rebalance_transfer_slot_ready(dst, config.participant_id, plan.layer, plan.expert))
        {
            if (threadIdx.x == 0)
                atomicAdd(&status->missing_destination_slots, 1u);
            return;
        }
        if (!rebalance_directory_compatible(src, dst) ||
            rebalance_expert_payload_bytes(src) > payload_data_bytes)
        {
            if (threadIdx.x == 0)
                atomicAdd(&status->descriptor_mismatches, 1u);
            return;
        }

        unsigned long long offset = 0ULL;
        const uint8_t *payload_data = payload_slot + sizeof(DeviceMoEExpertDirectoryEntryView);
        offset += rebalance_unpack_projection_from_payload(payload_data + offset, dst.descriptor.gate, src);
        offset += rebalance_unpack_projection_from_payload(payload_data + offset, dst.descriptor.up, src);
        offset += rebalance_unpack_projection_from_payload(payload_data + offset, dst.descriptor.down, src);
        (void)offset;
        __syncthreads();
        __threadfence();
        __syncthreads();

        if (threadIdx.x == 0)
        {
            dst.layer = plan.layer;
            dst.expert = plan.expert;
            dst.descriptor.logical_expert_id = static_cast<int32_t>(plan.expert);
            dst.descriptor.owner_participant = src.descriptor.owner_participant;
            dst.descriptor.flags |= kDeviceMoEFlagTransferSlot;
            dst.resident_mask |= runtime_participant_bit(static_cast<int>(config.participant_id));
            dst.flags |= kDeviceMoEDirectoryFlagResident |
                         kDeviceMoEDirectoryFlagCopyComplete;
            atomicAdd(&status->copied_arrivals, 1u);
        }
    }

    __global__ void apply_rebalance_arrivals_kernel(
        DeviceMoELayerRuntimeView *runtime_layers,
        const DeviceMoERebalancePlanEntryView *plan_entries,
        const uint32_t *plan_count,
        uint32_t plan_capacity,
        DeviceMoERebalanceCommandBufferHeaderView *command_header,
        const DeviceMoEExpertDirectoryEntryView *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        DeviceMoERebalanceConfigView config,
        DeviceMoERebalanceApplyStatusView *status,
        int target_layer,
        DeviceMoERebalanceGraphControllerStateView *controller_state,
        bool require_ready_wave,
        uint32_t command_buffer_count)
    {
        if (blockIdx.x != 0 || !status)
            return;

        __shared__ uint8_t changed_layer[kDeviceMoEMaxExperts];
        __shared__ uint32_t reduce_scratch[kDeviceMoERebalanceApplyThreads];
        __shared__ uint32_t shared_valid;
        __shared__ uint32_t shared_count;
        __shared__ uint32_t shared_participant_bit;
        __shared__ uint32_t shared_valid_mask;
        __shared__ DeviceMoERebalanceWaveProgressView *shared_ready_wave;
        __shared__ uint32_t shared_ready_wave_index;
        __shared__ const DeviceMoERebalancePlanEntryView *shared_plan_entries;
        __shared__ const uint32_t *shared_plan_count;
        __shared__ DeviceMoERebalanceCommandBufferHeaderView *shared_command_header;

        for (uint32_t layer = threadIdx.x;
             layer < static_cast<uint32_t>(kDeviceMoEMaxExperts);
             layer += blockDim.x)
        {
            changed_layer[layer] = 0u;
        }
        reduce_scratch[threadIdx.x] = 0u;
        __syncthreads();

        if (threadIdx.x == 0)
        {
            if (require_ready_wave)
                init_rebalance_apply_status_in_progress_device(status);

            bool setup_ok = true;
            shared_valid = 0u;
            shared_count = 0u;
            shared_participant_bit = 0u;
            shared_valid_mask = 0u;
            shared_ready_wave = nullptr;
            shared_ready_wave_index = 0u;
            shared_plan_entries = plan_entries;
            shared_plan_count = plan_count;
            shared_command_header = command_header;

            if (require_ready_wave)
            {
                if (!controller_state)
                {
                    setup_ok = false;
                }
                else
                {
                    if ((config.flags & kDeviceMoERebalanceFlagCollectLoadStats) != 0u &&
                        rebalance_graph_controller_state_ok(controller_state, config))
                        ++controller_state->decode_apply_polls;
                    const DeviceMoEReadyWaveSelection ready = rebalance_find_ready_wave(
                        controller_state,
                        command_header,
                        config,
                        target_layer,
                        command_buffer_count);
                    shared_ready_wave = ready.wave;
                    shared_ready_wave_index = ready.wave_index;
                    if (!shared_ready_wave)
                    {
                        setup_ok = false;
                    }
                    else
                    {
                        shared_plan_entries =
                            plan_entries + static_cast<unsigned long long>(shared_ready_wave_index) * plan_capacity;
                        if (plan_count)
                            shared_plan_count = plan_count + shared_ready_wave_index;
                        if (command_header)
                            shared_command_header = command_header + shared_ready_wave_index;
                        shared_ready_wave->state = kDeviceMoERebalanceLifecycleApplying;
                        ++controller_state->decode_apply_hits;
                    }
                }
            }

            if (setup_ok && !rebalance_config_ok(config))
            {
                status->status_code = 1u;
                if (shared_ready_wave)
                {
                    shared_ready_wave->error_code = 1u;
                    shared_ready_wave->state = kDeviceMoERebalanceLifecycleError;
                    controller_state->last_error_code = 1u;
                }
                setup_ok = false;
            }
            if (setup_ok && !runtime_layers)
            {
                status->status_code = 2u;
                if (shared_ready_wave)
                {
                    shared_ready_wave->error_code = 2u;
                    shared_ready_wave->state = kDeviceMoERebalanceLifecycleError;
                    controller_state->last_error_code = 2u;
                }
                setup_ok = false;
            }
            if (setup_ok &&
                (!shared_plan_entries || (!shared_plan_count && !shared_command_header) ||
                 !rebalance_command_header_ok(shared_command_header, config)))
            {
                status->status_code = 3u;
                if (shared_ready_wave)
                {
                    shared_ready_wave->error_code = 3u;
                    shared_ready_wave->state = kDeviceMoERebalanceLifecycleError;
                    controller_state->last_error_code = 3u;
                }
                setup_ok = false;
            }

            if (setup_ok)
            {
                shared_count = rebalance_command_count(shared_plan_count, plan_capacity, shared_command_header);
                shared_participant_bit = runtime_participant_bit(static_cast<int>(config.participant_id));
                shared_valid_mask = runtime_valid_participant_mask(config.participant_count);
                shared_valid = 1u;
            }
        }
        __syncthreads();
        if (shared_valid == 0u)
            return;

        if (threadIdx.x == 0)
        {
            for (uint32_t i = 0; i < shared_count; ++i)
            {
                ++status->plan_entries_seen;
                const auto &plan = shared_plan_entries[i];
                if (plan.op == 0u)
                    continue;
                if (target_layer >= 0 &&
                    plan.layer != static_cast<uint32_t>(target_layer))
                    continue;
                if (!rebalance_plan_applies_runtime(plan.op) ||
                    plan.layer >= config.num_layers ||
                    plan.expert >= config.num_experts ||
                    plan.source_participant >= config.participant_count ||
                    plan.destination_participant >= config.participant_count)
                {
                    ++status->invalid_plan_entries;
                    continue;
                }
                const bool destination_local =
                    plan.destination_participant == config.participant_id;
                const uint32_t destination_bit =
                    runtime_participant_bit(static_cast<int>(plan.destination_participant));

                auto &runtime = runtime_layers[plan.layer];
                if (runtime.active_bank > 1u ||
                    runtime.expert_count != config.num_experts ||
                    runtime.top_k != config.top_k ||
                    runtime.participant_id != config.participant_id ||
                    runtime.participant_count != config.participant_count)
                {
                    ++status->invalid_plan_entries;
                    continue;
                }

                const auto &active = runtime.banks[runtime.active_bank];
                uint32_t resident_mask =
                    (active.resident_participant_mask[plan.expert] |
                     plan.source_resident_mask |
                     destination_bit) &
                    shared_valid_mask;
                if (rebalance_plan_requires_payload(plan.op) &&
                    destination_local)
                {
                    ++status->required_local_arrivals;
                    if (!local_transfer_slots ||
                        plan.destination_slot >= local_transfer_slot_count)
                    {
                        ++status->missing_destination_slots;
                        continue;
                    }

                    const auto &slot = local_transfer_slots[plan.destination_slot];
                    if (!rebalance_transfer_slot_copy_complete(slot, config.participant_id, plan.layer, plan.expert))
                    {
                        ++status->copy_incomplete;
                        continue;
                    }
                    ++status->ready_local_arrivals;
                    if (plan.op == kDeviceMoERebalancePlanOwnershipTransfer)
                        resident_mask = destination_bit & shared_valid_mask;
                    else
                        resident_mask |= shared_participant_bit;
                }
                else if (plan.op == kDeviceMoERebalancePlanResidentExpertAssignment &&
                         destination_local &&
                         (resident_mask & shared_participant_bit) == 0u)
                {
                    ++status->missing_source_descriptors;
                    continue;
                }

                changed_layer[plan.layer] = 1u;
            }
        }
        __syncthreads();

        for (uint32_t layer = 0;
             layer < config.num_layers && layer < static_cast<uint32_t>(kDeviceMoEMaxExperts);
             ++layer)
        {
            if (changed_layer[layer] == 0u)
                continue;

            auto &runtime = runtime_layers[layer];
            const uint32_t inactive_bank = 1u - runtime.active_bank;
            const auto &active = runtime.banks[runtime.active_bank];
            auto &next = runtime.banks[inactive_bank];
            if (threadIdx.x == 0)
            {
                next.epoch = runtime.active_epoch + 1u;
                next.expert_count = config.num_experts;
                next.reserved[0] = active.reserved[0];
                next.reserved[1] = active.reserved[1];
            }
            for (uint32_t expert = threadIdx.x; expert < config.num_experts; expert += blockDim.x)
            {
                next.experts[expert] = active.experts[expert];
                auto &base_desc = next.experts[expert];
                uint32_t resident_mask =
                    active.resident_participant_mask[expert] & shared_valid_mask;
                const bool owner_local =
                    base_desc.owner_participant == static_cast<int32_t>(config.participant_id);
                const bool local_resident = (resident_mask & shared_participant_bit) != 0u;
                const bool multi_resident =
                    runtime_resident_count(resident_mask, config.participant_count) > 1;
                if (multi_resident)
                    base_desc.flags |= kDeviceMoEFlagReplicated;
                else
                    base_desc.flags &= ~kDeviceMoEFlagReplicated;
                if (local_resident)
                {
                    base_desc.flags |= kDeviceMoEFlagValid |
                                       kDeviceMoEFlagResident |
                                       kDeviceMoEFlagLocalCompute;
                }
                else
                {
                    base_desc.flags &= ~kDeviceMoEFlagLocalCompute;
                }
                next.local_compute_mask[expert] = local_resident ? 1u : 0u;
                next.replica_role[expert] =
                    local_resident
                        ? (owner_local
                               ? static_cast<uint8_t>(kDeviceMoEReplicaRolePrimary)
                               : static_cast<uint8_t>(kDeviceMoEReplicaRoleReplica))
                        : static_cast<uint8_t>(kDeviceMoEReplicaRoleNone);
                next.resident_participant_mask[expert] = resident_mask;
            }
            __syncthreads();
        }

        if (threadIdx.x == 0)
        {
            for (uint32_t i = 0; i < shared_count; ++i)
            {
                const auto &plan = shared_plan_entries[i];
                if (plan.op == 0u)
                    continue;
                if (target_layer >= 0 &&
                    plan.layer != static_cast<uint32_t>(target_layer))
                    continue;
                if (!rebalance_plan_applies_runtime(plan.op) ||
                    plan.layer >= config.num_layers ||
                    plan.expert >= config.num_experts ||
                    plan.source_participant >= config.participant_count ||
                    plan.destination_participant >= config.participant_count ||
                    changed_layer[plan.layer] == 0u)
                {
                    continue;
                }
                const bool destination_local =
                    plan.destination_participant == config.participant_id;
                const uint32_t destination_bit =
                    runtime_participant_bit(static_cast<int>(plan.destination_participant));
                const bool ownership_transfer =
                    plan.op == kDeviceMoERebalancePlanOwnershipTransfer;
                auto &runtime = runtime_layers[plan.layer];
                if (runtime.active_bank > 1u ||
                    runtime.expert_count != config.num_experts ||
                    runtime.top_k != config.top_k ||
                    runtime.participant_id != config.participant_id ||
                    runtime.participant_count != config.participant_count)
                {
                    continue;
                }

                const auto &active = runtime.banks[runtime.active_bank];
                auto desc = active.experts[plan.expert];
                uint32_t resident_mask =
                    (active.resident_participant_mask[plan.expert] |
                     plan.source_resident_mask |
                     destination_bit) &
                    shared_valid_mask;
                if (ownership_transfer)
                    resident_mask = destination_bit & shared_valid_mask;
                if (rebalance_plan_requires_payload(plan.op) &&
                    destination_local)
                {
                    if (!local_transfer_slots ||
                        plan.destination_slot >= local_transfer_slot_count)
                        continue;
                    const auto &slot = local_transfer_slots[plan.destination_slot];
                    if (!rebalance_transfer_slot_copy_complete(slot, config.participant_id, plan.layer, plan.expert))
                        continue;
                    desc = slot.descriptor;
                    desc.owner_participant =
                        ownership_transfer
                            ? static_cast<int32_t>(plan.destination_participant)
                            : active.experts[plan.expert].owner_participant;
                    resident_mask = ownership_transfer
                                        ? destination_bit & shared_valid_mask
                                        : resident_mask | shared_participant_bit;
                }
                else if (plan.op == kDeviceMoERebalancePlanResidentExpertAssignment &&
                         destination_local &&
                         (resident_mask & shared_participant_bit) == 0u)
                {
                    continue;
                }

                const uint32_t inactive_bank = 1u - runtime.active_bank;
                auto &next = runtime.banks[inactive_bank];
                if (ownership_transfer)
                {
                    desc.owner_participant =
                        static_cast<int32_t>(plan.destination_participant);
                    desc.flags &= ~kDeviceMoEFlagReplicated;
                }
                else
                {
                    desc.flags |= kDeviceMoEFlagReplicated;
                }
                if (destination_local)
                {
                    desc.flags |= kDeviceMoEFlagValid |
                                  kDeviceMoEFlagResident |
                                  kDeviceMoEFlagLocalCompute;
                    next.local_compute_mask[plan.expert] = 1u;
                    next.replica_role[plan.expert] =
                        ownership_transfer ||
                                desc.owner_participant == static_cast<int32_t>(config.participant_id)
                            ? static_cast<uint8_t>(kDeviceMoEReplicaRolePrimary)
                            : static_cast<uint8_t>(kDeviceMoEReplicaRoleReplica);
                    ++status->applied_arrivals;
                }
                else if (ownership_transfer)
                {
                    desc.flags &= ~(kDeviceMoEFlagResident | kDeviceMoEFlagLocalCompute);
                    next.local_compute_mask[plan.expert] = 0u;
                    next.replica_role[plan.expert] =
                        static_cast<uint8_t>(kDeviceMoEReplicaRoleNone);
                }
                next.experts[plan.expert] = desc;
                next.resident_participant_mask[plan.expert] =
                    resident_mask & shared_valid_mask;
            }
        }
        __syncthreads();

        for (uint32_t layer = 0;
             layer < config.num_layers && layer < static_cast<uint32_t>(kDeviceMoEMaxExperts);
             ++layer)
        {
            if (changed_layer[layer] == 0u)
                continue;

            auto &runtime = runtime_layers[layer];
            const uint32_t inactive_bank = 1u - runtime.active_bank;
            auto &next = runtime.banks[inactive_bank];
            uint32_t local_multi_resident = 0u;
            for (uint32_t expert = threadIdx.x; expert < config.num_experts; expert += blockDim.x)
            {
                const uint32_t mask = next.resident_participant_mask[expert] & shared_valid_mask;
                if (runtime_resident_count(mask, config.participant_count) > 1u)
                    ++local_multi_resident;
                if ((config.flags & kDeviceMoERebalanceFlagResetHistograms) != 0u)
                {
                    runtime.decode_histogram[expert] = 0ULL;
                    runtime.decode_local_histogram[expert] = 0ULL;
                }
            }
            reduce_scratch[threadIdx.x] = local_multi_resident;
            __syncthreads();
            for (uint32_t stride = blockDim.x / 2u; stride > 0u; stride >>= 1u)
            {
                if (threadIdx.x < stride)
                    reduce_scratch[threadIdx.x] += reduce_scratch[threadIdx.x + stride];
                __syncthreads();
            }
            if (threadIdx.x == 0)
            {
                next.reserved[0] = reduce_scratch[0];
            }
            __syncthreads();
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0)
            {
                runtime.active_bank = inactive_bank;
                runtime.active_epoch = next.epoch;
                ++status->changed_layers;
                status->post_apply_multi_resident_experts += reduce_scratch[0];
            }
            __syncthreads();
        }

        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0)
        {
            status->status_code = 0u;
            if (shared_ready_wave)
            {
                shared_ready_wave->applied_arrivals += status->applied_arrivals;
                const uint32_t layer_increment = target_layer >= 0 ? 1u : shared_ready_wave->planned_layer_count;
                const uint32_t next_applied = shared_ready_wave->applied_layer_count + layer_increment;
                shared_ready_wave->applied_layer_count =
                    next_applied < shared_ready_wave->planned_layer_count
                        ? next_applied
                        : shared_ready_wave->planned_layer_count;
                if (shared_ready_wave->applied_layer_count >= shared_ready_wave->planned_layer_count)
                {
                    clear_rebalance_command_header_device(shared_command_header);
                    shared_ready_wave->state = kDeviceMoERebalanceLifecycleApplied;
                    controller_state->active_wave =
                        (controller_state->active_wave + 1u) %
                        rebalance_command_buffer_count(command_buffer_count);
                }
                else
                {
                    shared_ready_wave->state = kDeviceMoERebalanceLifecycleReadyToApply;
                }
            }
            __threadfence();
        }
    }

    __device__ __forceinline__ void try_apply_ready_rebalance_wave_for_layer_thread0(
        DeviceMoELayerRuntimeView *runtime_layers,
        const DeviceMoERebalancePlanEntryView *plan_entries,
        uint32_t plan_capacity,
        DeviceMoERebalanceCommandBufferHeaderView *command_headers,
        const DeviceMoEExpertDirectoryEntryView *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        DeviceMoERebalanceConfigView config,
        DeviceMoERebalanceApplyStatusView *status,
        DeviceMoERebalanceGraphControllerStateView *controller_state,
        int target_layer,
        uint32_t command_buffer_count)
    {
        if (threadIdx.x != 0 || !runtime_layers ||
            !plan_entries || !command_headers || !controller_state ||
            !rebalance_config_ok(config))
        {
            return;
        }

        if ((config.flags & kDeviceMoERebalanceFlagCollectLoadStats) != 0u &&
            rebalance_graph_controller_state_ok(controller_state, config))
            ++controller_state->decode_apply_polls;
        const DeviceMoEReadyWaveSelection ready = rebalance_find_ready_wave(
            controller_state,
            command_headers,
            config,
            target_layer,
            command_buffer_count);
        if (!ready.wave)
            return;

        DeviceMoERebalanceWaveProgressView *wave = ready.wave;
        const uint32_t wave_index = ready.wave_index;
        auto *command_header = command_headers + wave_index;
        if (!rebalance_command_header_ok(command_header, config))
            return;

        const auto *wave_plan_entries =
            plan_entries + static_cast<unsigned long long>(wave_index) * plan_capacity;
        const uint32_t command_count =
            rebalance_command_count(nullptr, plan_capacity, command_header);
        if (command_count == 0u)
            return;

        if (status)
            init_rebalance_apply_status_in_progress_device(status);
        wave->state = kDeviceMoERebalanceLifecycleApplying;
        ++controller_state->decode_apply_hits;

        uint32_t layer = target_layer >= 0
                             ? static_cast<uint32_t>(target_layer)
                             : config.num_layers;
        if (target_layer < 0)
        {
            const uint32_t layer_offset =
                min(wave->applied_layer_count, wave->planned_layer_count);
            if (layer_offset < wave->planned_layer_count && config.num_layers > 0u)
            {
                layer =
                    (wave->planned_start_layer + layer_offset) %
                    config.num_layers;
            }
            if (layer >= config.num_layers)
            {
                clear_rebalance_command_header_device(command_header);
                wave->state = kDeviceMoERebalanceLifecycleApplied;
                controller_state->active_wave =
                    (controller_state->active_wave + 1u) %
                    rebalance_command_buffer_count(command_buffer_count);
                __threadfence();
                return;
            }
        }
        const uint32_t participant_bit =
            runtime_participant_bit(static_cast<int>(config.participant_id));
        const uint32_t valid_mask = runtime_valid_participant_mask(config.participant_count);
        auto &runtime = runtime_layers[layer];
        if (runtime.active_bank > 1u ||
            runtime.expert_count != config.num_experts ||
            runtime.top_k != config.top_k ||
            runtime.participant_id != config.participant_id ||
            runtime.participant_count != config.participant_count)
        {
            if (status)
                status->status_code = 2u;
            wave->error_code = 2u;
            wave->state = kDeviceMoERebalanceLifecycleError;
            controller_state->last_error_code = 2u;
            return;
        }

        const auto &active = runtime.banks[runtime.active_bank];
        uint32_t valid_plan_count = 0u;
        uint32_t ready_plan_count = 0u;
        bool blocked_on_arrival = false;

        for (uint32_t i = 0; i < command_count; ++i)
        {
            const auto &plan = wave_plan_entries[i];
            if (status)
                ++status->plan_entries_seen;
            if (plan.op == 0u || plan.layer != layer)
                continue;
            if (!rebalance_plan_applies_runtime(plan.op) ||
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
            ++valid_plan_count;

            const uint32_t destination_bit =
                runtime_participant_bit(static_cast<int>(plan.destination_participant));
            uint32_t resident_mask =
                (active.resident_participant_mask[plan.expert] |
                 plan.source_resident_mask |
                 destination_bit) &
                valid_mask;
            if (plan.op == kDeviceMoERebalancePlanOwnershipTransfer)
                resident_mask = destination_bit & valid_mask;

            if (rebalance_plan_requires_payload(plan.op) &&
                destination_local)
            {
                if (status)
                    ++status->required_local_arrivals;
                if (!local_transfer_slots ||
                    plan.destination_slot >= local_transfer_slot_count)
                {
                    if (status)
                        ++status->missing_destination_slots;
                    blocked_on_arrival = true;
                    continue;
                }
                const auto &slot = local_transfer_slots[plan.destination_slot];
                if (!rebalance_transfer_slot_copy_complete(
                        slot, config.participant_id, plan.layer, plan.expert))
                {
                    if (status)
                        ++status->copy_incomplete;
                    blocked_on_arrival = true;
                    continue;
                }
                if (status)
                    ++status->ready_local_arrivals;
            }
            else if (plan.op == kDeviceMoERebalancePlanResidentExpertAssignment &&
                     destination_local &&
                     (resident_mask & participant_bit) == 0u)
            {
                if (status)
                    ++status->missing_source_descriptors;
                blocked_on_arrival = true;
                continue;
            }

            ++ready_plan_count;
        }

        if (valid_plan_count == 0u)
        {
            wave->applied_layer_count =
                min(wave->planned_layer_count, wave->applied_layer_count + 1u);
            wave->state =
                wave->applied_layer_count >= wave->planned_layer_count
                    ? kDeviceMoERebalanceLifecycleApplied
                    : kDeviceMoERebalanceLifecycleReadyToApply;
            if (wave->state == kDeviceMoERebalanceLifecycleApplied)
            {
                clear_rebalance_command_header_device(command_header);
                controller_state->active_wave =
                    (controller_state->active_wave + 1u) %
                    rebalance_command_buffer_count(command_buffer_count);
            }
            __threadfence();
            return;
        }
        if (blocked_on_arrival || ready_plan_count == 0u)
        {
            wave->state = kDeviceMoERebalanceLifecycleReadyToApply;
            __threadfence();
            return;
        }

        const uint32_t inactive_bank = 1u - runtime.active_bank;
        auto &next = runtime.banks[inactive_bank];
        next.epoch = runtime.active_epoch + 1u;
        next.expert_count = config.num_experts;
        next.reserved[0] = active.reserved[0];
        next.reserved[1] = active.reserved[1];
        for (uint32_t expert = 0; expert < config.num_experts; ++expert)
        {
            next.experts[expert] = active.experts[expert];
            auto &base_desc = next.experts[expert];
            uint32_t resident_mask =
                active.resident_participant_mask[expert] & valid_mask;
            const bool owner_local =
                base_desc.owner_participant == static_cast<int32_t>(config.participant_id);
            const bool local_resident = (resident_mask & participant_bit) != 0u;
            const bool multi_resident =
                runtime_resident_count(resident_mask, config.participant_count) > 1;
            if (multi_resident)
                base_desc.flags |= kDeviceMoEFlagReplicated;
            else
                base_desc.flags &= ~kDeviceMoEFlagReplicated;
            if (local_resident)
            {
                base_desc.flags |= kDeviceMoEFlagValid |
                                   kDeviceMoEFlagResident |
                                   kDeviceMoEFlagLocalCompute;
            }
            else
            {
                base_desc.flags &= ~kDeviceMoEFlagLocalCompute;
            }
            next.local_compute_mask[expert] = local_resident ? 1u : 0u;
            next.replica_role[expert] =
                local_resident
                    ? (owner_local
                           ? static_cast<uint8_t>(kDeviceMoEReplicaRolePrimary)
                           : static_cast<uint8_t>(kDeviceMoEReplicaRoleReplica))
                    : static_cast<uint8_t>(kDeviceMoEReplicaRoleNone);
            next.resident_participant_mask[expert] = resident_mask;
        }

        uint32_t applied_arrivals = 0u;
        for (uint32_t i = 0; i < command_count; ++i)
        {
            const auto &plan = wave_plan_entries[i];
            if (plan.op == 0u || plan.layer != layer)
                continue;
            if (!rebalance_plan_applies_runtime(plan.op) ||
                plan.expert >= config.num_experts ||
                plan.source_participant >= config.participant_count ||
                plan.destination_participant >= config.participant_count)
            {
                continue;
            }

            const bool destination_local =
                plan.destination_participant == config.participant_id;
            const uint32_t destination_bit =
                runtime_participant_bit(static_cast<int>(plan.destination_participant));
            const bool ownership_transfer =
                plan.op == kDeviceMoERebalancePlanOwnershipTransfer;
            auto desc = active.experts[plan.expert];
            uint32_t resident_mask =
                (active.resident_participant_mask[plan.expert] |
                 plan.source_resident_mask |
                 destination_bit) &
                valid_mask;
            if (ownership_transfer)
                resident_mask = destination_bit & valid_mask;

            if (rebalance_plan_requires_payload(plan.op) &&
                destination_local)
            {
                if (!local_transfer_slots ||
                    plan.destination_slot >= local_transfer_slot_count)
                {
                    if (status)
                        ++status->missing_destination_slots;
                    continue;
                }
                const auto &slot = local_transfer_slots[plan.destination_slot];
                if (!rebalance_transfer_slot_copy_complete(
                        slot, config.participant_id, plan.layer, plan.expert))
                {
                    if (status)
                        ++status->copy_incomplete;
                    continue;
                }
                desc = slot.descriptor;
                desc.owner_participant =
                    ownership_transfer
                        ? static_cast<int32_t>(plan.destination_participant)
                        : active.experts[plan.expert].owner_participant;
                resident_mask = ownership_transfer
                                    ? destination_bit & valid_mask
                                    : resident_mask | participant_bit;
            }
            else if (plan.op == kDeviceMoERebalancePlanResidentExpertAssignment &&
                     destination_local &&
                     (resident_mask & participant_bit) == 0u)
            {
                if (status)
                    ++status->missing_source_descriptors;
                continue;
            }

            if (ownership_transfer)
            {
                desc.owner_participant =
                    static_cast<int32_t>(plan.destination_participant);
                desc.flags &= ~kDeviceMoEFlagReplicated;
            }
            else
            {
                desc.flags |= kDeviceMoEFlagReplicated;
            }
            if (destination_local)
            {
                desc.flags |= kDeviceMoEFlagValid |
                              kDeviceMoEFlagResident |
                              kDeviceMoEFlagLocalCompute;
                next.local_compute_mask[plan.expert] = 1u;
                next.replica_role[plan.expert] =
                    ownership_transfer ||
                            desc.owner_participant == static_cast<int32_t>(config.participant_id)
                        ? static_cast<uint8_t>(kDeviceMoEReplicaRolePrimary)
                        : static_cast<uint8_t>(kDeviceMoEReplicaRoleReplica);
                ++applied_arrivals;
            }
            else if (ownership_transfer)
            {
                desc.flags &= ~(kDeviceMoEFlagResident | kDeviceMoEFlagLocalCompute);
                next.local_compute_mask[plan.expert] = 0u;
                next.replica_role[plan.expert] =
                    static_cast<uint8_t>(kDeviceMoEReplicaRoleNone);
            }
            next.experts[plan.expert] = desc;
            next.resident_participant_mask[plan.expert] = resident_mask & valid_mask;
        }

        uint32_t multi_resident = 0u;
        for (uint32_t expert = 0; expert < config.num_experts; ++expert)
        {
            const uint32_t mask = next.resident_participant_mask[expert] & valid_mask;
            if (runtime_resident_count(mask, config.participant_count) > 1u)
                ++multi_resident;
            if ((config.flags & kDeviceMoERebalanceFlagResetHistograms) != 0u)
            {
                runtime.decode_histogram[expert] = 0ULL;
                runtime.decode_local_histogram[expert] = 0ULL;
            }
        }
        next.reserved[0] = multi_resident;
        runtime.active_bank = inactive_bank;
        runtime.active_epoch = next.epoch;
        if (status)
        {
            status->status_code = 0u;
            status->changed_layers = 1u;
            status->applied_arrivals = applied_arrivals;
            status->post_apply_multi_resident_experts = multi_resident;
        }
        wave->applied_arrivals += applied_arrivals;
        const uint32_t next_applied = wave->applied_layer_count + 1u;
        wave->applied_layer_count =
            next_applied < wave->planned_layer_count
                ? next_applied
                : wave->planned_layer_count;
        if (wave->applied_layer_count >= wave->planned_layer_count)
        {
            clear_rebalance_command_header_device(command_header);
            wave->state = kDeviceMoERebalanceLifecycleApplied;
            controller_state->active_wave =
                (controller_state->active_wave + 1u) %
                rebalance_command_buffer_count(command_buffer_count);
        }
        else
        {
            wave->state = kDeviceMoERebalanceLifecycleReadyToApply;
        }
        __threadfence();
    }

    constexpr int kMoEPointerStageMax = 16;

    struct MoEPointerStagePayload
    {
        const float *const_ptrs_a[kMoEPointerStageMax];
        const float *const_ptrs_b[kMoEPointerStageMax];
        float *mutable_ptrs_a[kMoEPointerStageMax];
        float *mutable_ptrs_b[kMoEPointerStageMax];
    };

    __global__ void moe_stage_const_pointer_arrays_kernel(
        MoEPointerStagePayload payload,
        const float **dst_a,
        const float **dst_b,
        int count)
    {
        const int i = static_cast<int>(threadIdx.x);
        if (i >= count)
            return;
        dst_a[i] = payload.const_ptrs_a[i];
        dst_b[i] = payload.const_ptrs_b[i];
    }

    __global__ void moe_stage_mutable_pointer_arrays_kernel(
        MoEPointerStagePayload payload,
        float **dst_a,
        float **dst_b,
        int count)
    {
        const int i = static_cast<int>(threadIdx.x);
        if (i >= count)
            return;
        dst_a[i] = payload.mutable_ptrs_a[i];
        dst_b[i] = payload.mutable_ptrs_b[i];
    }

    __device__ __forceinline__ bool moe_topk_pair_better(
        float candidate_value,
        int candidate_id,
        float best_value,
        int best_id)
    {
        return candidate_value > best_value ||
               (candidate_value == best_value && candidate_id < best_id);
    }

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx906__)
#define LLAMINAR_ROCM_MOE_GFX906_DPP 1
#endif

#if defined(LLAMINAR_ROCM_MOE_GFX906_DPP)
#define LLAMINAR_ROCM_MOE_DPP_F32(name, barrier, dpp_ctrl, vop_instr)       \
    __device__ __forceinline__ float name(float x)                          \
    {                                                                       \
        float r;                                                            \
        asm volatile(                                                       \
            barrier                                                         \
            vop_instr " %0, %1, %1 " dpp_ctrl " row_mask:0xf bank_mask:0xf" \
            : "=v"(r)                                                       \
            : "v"(x)                                                        \
            : "memory");                                                    \
        return r;                                                           \
    }

    LLAMINAR_ROCM_MOE_DPP_F32(rocm_moe_dpp_add_xor1_f32, "s_nop 4\n", "quad_perm:[1,0,3,2]", "v_add_f32_dpp")
    LLAMINAR_ROCM_MOE_DPP_F32(rocm_moe_dpp_add_xor2_f32, "s_nop 1\n", "quad_perm:[2,3,0,1]", "v_add_f32_dpp")
    LLAMINAR_ROCM_MOE_DPP_F32(rocm_moe_dpp_add_xor8_f32, "s_nop 1\n", "row_ror:8", "v_add_f32_dpp")
    LLAMINAR_ROCM_MOE_DPP_F32(rocm_moe_dpp_max_xor1_f32, "s_nop 4\n", "quad_perm:[1,0,3,2]", "v_max_f32_dpp")
    LLAMINAR_ROCM_MOE_DPP_F32(rocm_moe_dpp_max_xor2_f32, "s_nop 1\n", "quad_perm:[2,3,0,1]", "v_max_f32_dpp")
    LLAMINAR_ROCM_MOE_DPP_F32(rocm_moe_dpp_max_xor8_f32, "s_nop 1\n", "row_ror:8", "v_max_f32_dpp")

#undef LLAMINAR_ROCM_MOE_DPP_F32

    template <int off>
    __device__ __forceinline__ int rocm_moe_dpp_xfer_xor_i32(int x)
    {
        int d;
        if constexpr (off == 1)
        {
            asm volatile("s_nop 4\n"
                         "v_mov_b32_dpp %0, %1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"
                         : "=v"(d)
                         : "v"(x)
                         : "memory");
        }
        else if constexpr (off == 2)
        {
            asm volatile("s_nop 4\n"
                         "v_mov_b32_dpp %0, %1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf"
                         : "=v"(d)
                         : "v"(x)
                         : "memory");
        }
        else if constexpr (off == 4)
        {
            asm volatile("v_mov_b32 %0, %1\n"
                         "s_nop 1\n"
                         "v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0x5\n"
                         "v_mov_b32_dpp %0, %1 row_shr:4 row_mask:0xf bank_mask:0xa\n"
                         : "=v"(d)
                         : "v"(x)
                         : "memory");
        }
        else if constexpr (off == 8)
        {
            asm volatile("s_nop 4\n"
                         "v_mov_b32_dpp %0, %1 row_ror:8 row_mask:0xf bank_mask:0xf"
                         : "=v"(d)
                         : "v"(x)
                         : "memory");
        }
        else
        {
            static_assert(off == 16, "unsupported xor offset");
            asm volatile("ds_swizzle_b32 %0, %1 offset:swizzle(SWAP,16)\n"
                         "s_waitcnt lgkmcnt(0)\n"
                         : "=v"(d)
                         : "v"(x)
                         : "memory");
        }
        return d;
    }

    __device__ __forceinline__ float rocm_moe_dpp_xfer_xor4_f32(float x)
    {
        return __int_as_float(rocm_moe_dpp_xfer_xor_i32<4>(__float_as_int(x)));
    }

    __device__ __forceinline__ float rocm_moe_dpp_xfer_xor16_f32(float x)
    {
        return __int_as_float(rocm_moe_dpp_xfer_xor_i32<16>(__float_as_int(x)));
    }
#endif

    __device__ __forceinline__ float rocm_moe_wave_reduce_sum_f32(float x)
    {
#if defined(LLAMINAR_ROCM_MOE_GFX906_DPP)
        x = rocm_moe_dpp_add_xor1_f32(x);
        x = rocm_moe_dpp_add_xor2_f32(x);
        x += rocm_moe_dpp_xfer_xor4_f32(x);
        x = rocm_moe_dpp_add_xor8_f32(x);
        x += rocm_moe_dpp_xfer_xor16_f32(x);
        x += __shfl_xor(x, 32);
#else
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            x += __shfl_down(x, offset);
#endif
        return x;
    }

    __device__ __forceinline__ float rocm_moe_wave_reduce_max_f32(float x)
    {
#if defined(LLAMINAR_ROCM_MOE_GFX906_DPP)
        x = rocm_moe_dpp_max_xor1_f32(x);
        x = rocm_moe_dpp_max_xor2_f32(x);
        x = fmaxf(x, rocm_moe_dpp_xfer_xor4_f32(x));
        x = rocm_moe_dpp_max_xor8_f32(x);
        x = fmaxf(x, rocm_moe_dpp_xfer_xor16_f32(x));
        x = fmaxf(x, __shfl_xor(x, 32));
#else
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            x = fmaxf(x, __shfl_down(x, offset));
#endif
        return x;
    }

    __device__ void moe_select_topk_probabilities_block(
        float *__restrict__ expert_probs,
        int num_experts,
        int top_k,
        int *__restrict__ selected_ids,
        float *__restrict__ selected_weights)
    {
        __shared__ float warp_best_values[8];
        __shared__ int warp_best_ids[8];

        const int lane = threadIdx.x % warpSize;
        const int warp_id = threadIdx.x / warpSize;
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;

        for (int slot = 0; slot < top_k; ++slot)
        {
            float best_value = -1.0f;
            int best_id = kDeviceMoEMaxExperts;

            if (threadIdx.x < num_experts)
            {
                best_value = expert_probs[threadIdx.x];
                best_id = threadIdx.x;
            }

            for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            {
                const float other_value = __shfl_down(best_value, offset);
                const int other_id = __shfl_down(best_id, offset);
                if (moe_topk_pair_better(other_value, other_id, best_value, best_id))
                {
                    best_value = other_value;
                    best_id = other_id;
                }
            }

            if (lane == 0)
            {
                warp_best_values[warp_id] = best_value;
                warp_best_ids[warp_id] = best_id;
            }
            __syncthreads();

            if (warp_id == 0)
            {
                best_value = (lane < num_warps) ? warp_best_values[lane] : -1.0f;
                best_id = (lane < num_warps) ? warp_best_ids[lane] : kDeviceMoEMaxExperts;
                for (int offset = warpSize / 2; offset > 0; offset >>= 1)
                {
                    const float other_value = __shfl_down(best_value, offset);
                    const int other_id = __shfl_down(best_id, offset);
                    if (moe_topk_pair_better(other_value, other_id, best_value, best_id))
                    {
                        best_value = other_value;
                        best_id = other_id;
                    }
                }

                if (lane == 0)
                {
                    selected_ids[slot] = best_id;
                    selected_weights[slot] = best_value;
                    if (best_id >= 0 && best_id < num_experts)
                    {
                        expert_probs[best_id] = -1.0f;
                    }
                }
            }
            __syncthreads();
        }
    }
}

// =========================================================================
// Kernel 1: Single-token Gate Logits
//
// Decode routes one token at a time. hipBLAS is efficient for prefill GEMM but
// pays too much fixed overhead for a 1 x d_model by num_experts router GEMV.
// This path computes one expert logit per block and feeds the existing
// softmax/top-k kernel.
// =========================================================================

__global__ void rocm_moe_gate_logits_single_token_kernel(
    const float *__restrict__ hidden,       // [d_model]
    const float *__restrict__ gate_weights, // [num_experts, d_model]
    float *__restrict__ logits,             // [num_experts]
    int d_model, int num_experts)
{
    const int expert = blockIdx.x;
    if (expert >= num_experts) return;

    const float *w = gate_weights + expert * d_model;
    float sum = 0.0f;

    if ((d_model & 3) == 0)
    {
        const int vec_count = d_model >> 2;
        const float4 *hidden4 = reinterpret_cast<const float4 *>(hidden);
        const float4 *w4 = reinterpret_cast<const float4 *>(w);
        for (int i = threadIdx.x; i < vec_count; i += blockDim.x)
        {
            const float4 h = hidden4[i];
            const float4 ww = w4[i];
            sum += h.x * ww.x + h.y * ww.y + h.z * ww.z + h.w * ww.w;
        }
    }
    else
    {
        for (int i = threadIdx.x; i < d_model; i += blockDim.x)
        {
            sum += hidden[i] * w[i];
        }
    }

    sum = rocm_moe_wave_reduce_sum_f32(sum);

    __shared__ float warp_sums[8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0)
    {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        sum = (lane < nw) ? warp_sums[lane] : 0.0f;
        sum = rocm_moe_wave_reduce_sum_f32(sum);
        if (lane == 0) logits[expert] = sum;
    }
}

__global__ void rocm_moe_gate_logits_small_m_kernel(
    const float *__restrict__ hidden,       // [seq_len, d_model]
    const float *__restrict__ gate_weights, // [num_experts, d_model]
    float *__restrict__ logits,             // [seq_len, num_experts]
    int seq_len, int d_model, int num_experts,
    bool use_vec4)
{
    extern __shared__ float4 hidden_shared_vec4[];
    float *hidden_shared = reinterpret_cast<float *>(hidden_shared_vec4);

    const int row = blockIdx.y;
    if (row >= seq_len) return;

    const float *h = hidden + static_cast<size_t>(row) * d_model;

    if (use_vec4)
    {
        const int vec_count = d_model >> 2;
        const float4 *hidden4 = reinterpret_cast<const float4 *>(h);
        for (int i = threadIdx.x; i < vec_count; i += blockDim.x)
        {
            hidden_shared_vec4[i] = hidden4[i];
        }
    }
    else
    {
        for (int i = threadIdx.x; i < d_model; i += blockDim.x)
            hidden_shared[i] = h[i];
    }
    __syncthreads();

    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (warp_id >= kGateLogitsExpertsPerBlock)
        return;

    const int expert = blockIdx.x * kGateLogitsExpertsPerBlock + warp_id;
    if (expert >= num_experts)
        return;

    const float *w = gate_weights + static_cast<size_t>(expert) * d_model;
    float sum = 0.0f;

    if (use_vec4)
    {
        const int vec_count = d_model >> 2;
        const float4 *hidden4 = reinterpret_cast<const float4 *>(hidden_shared);
        const float4 *w4 = reinterpret_cast<const float4 *>(w);
        for (int i = lane; i < vec_count; i += warpSize)
        {
            const float4 hh = hidden4[i];
            const float4 ww = w4[i];
            sum += hh.x * ww.x + hh.y * ww.y + hh.z * ww.z + hh.w * ww.w;
        }
    }
    else
    {
        for (int i = lane; i < d_model; i += warpSize)
            sum += hidden_shared[i] * w[i];
    }

    sum = rocm_moe_wave_reduce_sum_f32(sum);

    if (lane == 0)
        logits[static_cast<size_t>(row) * num_experts + expert] = sum;
}

template <typename GateWeightT>
__global__ void rocm_moe_gate_logits_small_m_mixed_weights_kernel(
    const float *__restrict__ hidden,              // [seq_len, d_model]
    const GateWeightT *__restrict__ gate_weights,  // [num_experts, d_model]
    float *__restrict__ logits,                    // [seq_len, num_experts]
    int seq_len, int d_model, int num_experts)
{
    extern __shared__ float hidden_shared[];

    const int row = blockIdx.y;
    if (row >= seq_len) return;

    const float *h = hidden + static_cast<size_t>(row) * d_model;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x)
        hidden_shared[i] = h[i];
    __syncthreads();

    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (warp_id >= kGateLogitsExpertsPerBlock)
        return;

    const int expert = blockIdx.x * kGateLogitsExpertsPerBlock + warp_id;
    if (expert >= num_experts)
        return;

    const GateWeightT *w =
        gate_weights + static_cast<size_t>(expert) * d_model;
    float sum = 0.0f;
    for (int i = lane; i < d_model; i += warpSize)
        sum += hidden_shared[i] * route_weight_to_float(w[i]);

    sum = rocm_moe_wave_reduce_sum_f32(sum);

    if (lane == 0)
        logits[static_cast<size_t>(row) * num_experts + expert] = sum;
}

__global__ void rocm_moe_gate_logits_single_token_kpart_kernel(
    const float *__restrict__ hidden,       // [d_model]
    const float *__restrict__ gate_weights, // [num_experts, d_model]
    float *__restrict__ partials,           // [num_experts, k_partitions]
    int d_model, int num_experts, int k_partitions,
    bool use_vec4)
{
    const int expert = blockIdx.x;
    const int partition = blockIdx.y;
    if (expert >= num_experts || partition >= k_partitions) return;

    const int block_count = (d_model + 31) / 32;
    const int blocks_per_partition = (block_count + k_partitions - 1) / k_partitions;
    const int block_begin = partition * blocks_per_partition;
    const int block_end_unclamped = block_begin + blocks_per_partition;
    const int block_end = (block_end_unclamped < block_count) ? block_end_unclamped : block_count;
    const int element_begin = block_begin * 32;
    const int element_end_unclamped = block_end * 32;
    const int element_end = (element_end_unclamped < d_model) ? element_end_unclamped : d_model;

    const float *w = gate_weights + expert * d_model;
    float sum = 0.0f;

    if (use_vec4)
    {
        const int vec_begin = element_begin >> 2;
        const int vec_end = element_end >> 2;
        const float4 *hidden4 = reinterpret_cast<const float4 *>(hidden);
        const float4 *w4 = reinterpret_cast<const float4 *>(w);
        for (int i = vec_begin + threadIdx.x; i < vec_end; i += blockDim.x)
        {
            const float4 h = hidden4[i];
            const float4 ww = w4[i];
            sum += h.x * ww.x + h.y * ww.y + h.z * ww.z + h.w * ww.w;
        }
    }
    else
    {
        for (int i = element_begin + threadIdx.x; i < element_end; i += blockDim.x)
        {
            sum += hidden[i] * w[i];
        }
    }

    sum = rocm_moe_wave_reduce_sum_f32(sum);

    __shared__ float warp_sums[8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0)
    {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        sum = (lane < nw) ? warp_sums[lane] : 0.0f;
        sum = rocm_moe_wave_reduce_sum_f32(sum);
        if (lane == 0) partials[expert * k_partitions + partition] = sum;
    }
}

__global__ void rocm_moe_gate_logits_kpart_reduce_kernel(
    const float *__restrict__ partials, // [num_experts, k_partitions]
    float *__restrict__ logits,         // [num_experts]
    int num_experts, int k_partitions)
{
    const int expert = blockIdx.x;
    if (expert >= num_experts) return;

    const float *expert_partials = partials + expert * k_partitions;
    float sum = 0.0f;
    for (int partition = threadIdx.x; partition < k_partitions; partition += blockDim.x)
    {
        sum += expert_partials[partition];
    }

    sum = rocm_moe_wave_reduce_sum_f32(sum);

    __shared__ float warp_sums[8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0)
    {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        sum = (lane < nw) ? warp_sums[lane] : 0.0f;
        sum = rocm_moe_wave_reduce_sum_f32(sum);
        if (lane == 0) logits[expert] = sum;
    }
}

__global__ void rocm_moe_fp32_to_fp16_kernel(
    const float *__restrict__ input,
    __half *__restrict__ output,
    int count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    output[idx] = __float2half(input[idx]);
}

__global__ void rocm_moe_router_gate_quantize_q8_kernel(
    const float *__restrict__ gate_weights, // [num_experts, d_model]
    int8_t *__restrict__ gate_weights_q8,   // [num_experts, blocks_per_row, 32]
    float *__restrict__ gate_scales,        // [num_experts, blocks_per_row]
    int d_model, int num_experts)
{
    constexpr int BLOCK_SIZE = 32;
    const int expert = blockIdx.x;
    const int block_idx = blockIdx.y;
    const int lane = threadIdx.x;
    if (expert >= num_experts || lane >= BLOCK_SIZE) return;

    const int blocks_per_row = d_model / BLOCK_SIZE;
    const int k = block_idx * BLOCK_SIZE + lane;
    const size_t scale_idx = static_cast<size_t>(expert) * blocks_per_row + block_idx;
    const float value = (k < d_model) ? gate_weights[static_cast<size_t>(expert) * d_model + k] : 0.0f;

    float max_abs = fabsf(value);
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        max_abs = fmaxf(max_abs, __shfl_xor(max_abs, mask));
    }

    const float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
    if (lane == 0)
    {
        gate_scales[scale_idx] = scale;
    }

    const float q_value = value / scale;
    gate_weights_q8[scale_idx * BLOCK_SIZE + lane] = static_cast<int8_t>(
        rintf(fminf(127.0f, fmaxf(-127.0f, q_value))));
}

__global__ void rocm_moe_router_hidden_quantize_q8_kernel(
    const float *__restrict__ hidden,       // [d_model]
    int8_t *__restrict__ hidden_q8,         // [d_model]
    float *__restrict__ hidden_scales,      // [blocks_per_row]
    int d_model)
{
    constexpr int BLOCK_SIZE = 32;
    const int block_idx = blockIdx.x;
    const int lane = threadIdx.x;
    const int k = block_idx * BLOCK_SIZE + lane;
    if (lane >= BLOCK_SIZE || k >= d_model) return;

    const float value = hidden[k];
    float max_abs = fabsf(value);
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        max_abs = fmaxf(max_abs, __shfl_xor(max_abs, mask));
    }

    const float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
    if (lane == 0)
    {
        hidden_scales[block_idx] = scale;
    }

    const float q_value = value / scale;
    hidden_q8[k] = static_cast<int8_t>(
        rintf(fminf(127.0f, fmaxf(-127.0f, q_value))));
}

__global__ void rocm_moe_gate_logits_single_token_q8_weights_kernel(
    const int8_t *__restrict__ hidden_q8,       // [d_model]
    const float *__restrict__ hidden_scales,    // [blocks_per_row]
    const int8_t *__restrict__ gate_weights_q8, // [num_experts, blocks_per_row, 32]
    const float *__restrict__ gate_scales,      // [num_experts, blocks_per_row]
    float *__restrict__ logits,                 // [num_experts]
    int d_model, int num_experts)
{
    constexpr int BLOCK_SIZE = 32;
    const int expert = blockIdx.x;
    if (expert >= num_experts) return;

    const int blocks_per_row = d_model / BLOCK_SIZE;
    float sum = 0.0f;
    for (int block_idx = threadIdx.x; block_idx < blocks_per_row; block_idx += blockDim.x)
    {
        const int32_t *h4 = reinterpret_cast<const int32_t *>(hidden_q8 + block_idx * BLOCK_SIZE);
        const size_t scale_idx = static_cast<size_t>(expert) * blocks_per_row + block_idx;
        const int32_t *w4 = reinterpret_cast<const int32_t *>(gate_weights_q8 + scale_idx * BLOCK_SIZE);

        int32_t block_acc = 0;
        #pragma unroll
        for (int group = 0; group < 8; ++group)
        {
            block_acc = __builtin_amdgcn_sdot4(h4[group], w4[group], block_acc, false);
        }
        sum += static_cast<float>(block_acc) * hidden_scales[block_idx] * gate_scales[scale_idx];
    }

    sum = rocm_moe_wave_reduce_sum_f32(sum);

    __shared__ float warp_sums[8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0)
    {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        sum = (lane < nw) ? warp_sums[lane] : 0.0f;
        sum = rocm_moe_wave_reduce_sum_f32(sum);
        if (lane == 0) logits[expert] = sum;
    }
}

/**
 * @brief Quantize every verifier hidden row with serial Q8 decode scales.
 *
 * The grid is `[blocks_per_row, verifier_row]`; each HIP block computes the
 * exact same 32-column scale and int8 payload that the serial one-row router
 * would have produced for that row.  Keeping this row-local quantization is the
 * key serial-equivalence constraint before the grouped Q8 logits kernel below
 * shares gate loads across rows.
 */
__global__ void rocm_moe_router_hidden_quantize_q8_rows_kernel(
    const float *__restrict__ hidden,
    int8_t *__restrict__ hidden_q8,
    float *__restrict__ hidden_scales,
    int seq_len,
    int d_model)
{
    constexpr int BLOCK_SIZE = 32;
    const int block_idx = blockIdx.x;
    const int row = blockIdx.y;
    const int lane = threadIdx.x;
    const int k = block_idx * BLOCK_SIZE + lane;
    if (row >= seq_len || lane >= BLOCK_SIZE || k >= d_model) return;

    const int blocks_per_row = d_model / BLOCK_SIZE;
    const size_t row_offset = static_cast<size_t>(row) * static_cast<size_t>(d_model);
    const float value = hidden[row_offset + k];
    float max_abs = fabsf(value);
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        max_abs = fmaxf(max_abs, __shfl_xor(max_abs, mask));
    }

    const float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
    if (lane == 0)
    {
        hidden_scales[static_cast<size_t>(row) * blocks_per_row + block_idx] = scale;
    }

    const float q_value = value / scale;
    hidden_q8[row_offset + k] = static_cast<int8_t>(
        rintf(fminf(127.0f, fmaxf(-127.0f, q_value))));
}

/**
 * @brief Economical grouped Q8 router logits for verifier rows.
 *
 * One block owns one expert and accumulates up to four row logits.  Gate Q8
 * payload and scales are loaded once per K block and reused for every verifier
 * row, while each row keeps the same K traversal and reduction order as serial
 * decode.  This is the grouped production primitive the verifier route calls;
 * row replay remains only in diagnostic test oracles.
 */
__global__ void rocm_moe_gate_logits_q8_grouped_verifier_kernel(
    const int8_t *__restrict__ hidden_q8,
    const float *__restrict__ hidden_scales,
    const int8_t *__restrict__ gate_weights_q8,
    const float *__restrict__ gate_scales,
    float *__restrict__ logits,
    int seq_len,
    int d_model,
    int num_experts)
{
    constexpr int BLOCK_SIZE = 32;
    const int expert = blockIdx.x;
    if (expert >= num_experts) return;

    const int blocks_per_row = d_model / BLOCK_SIZE;
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int block_idx = threadIdx.x; block_idx < blocks_per_row; block_idx += blockDim.x)
    {
        const size_t scale_idx = static_cast<size_t>(expert) * blocks_per_row + block_idx;
        const int32_t *w4 = reinterpret_cast<const int32_t *>(gate_weights_q8 + scale_idx * BLOCK_SIZE);
        const float gate_scale = gate_scales[scale_idx];

        int32_t w_values[8];
        #pragma unroll
        for (int group = 0; group < 8; ++group)
        {
            w_values[group] = w4[group];
        }

        for (int row = 0; row < seq_len; ++row)
        {
            const int8_t *row_hidden =
                hidden_q8 + static_cast<size_t>(row) * static_cast<size_t>(d_model);
            const float *row_scales =
                hidden_scales + static_cast<size_t>(row) * static_cast<size_t>(blocks_per_row);
            const int32_t *h4 = reinterpret_cast<const int32_t *>(row_hidden + block_idx * BLOCK_SIZE);

            int32_t block_acc = 0;
            #pragma unroll
            for (int group = 0; group < 8; ++group)
            {
                block_acc = __builtin_amdgcn_sdot4(h4[group], w_values[group], block_acc, false);
            }
            sums[row] += static_cast<float>(block_acc) * row_scales[block_idx] * gate_scale;
        }
    }

    __shared__ float warp_sums[4][8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    for (int row = 0; row < seq_len; ++row)
    {
        float reduced = rocm_moe_wave_reduce_sum_f32(sums[row]);
        if (lane == 0) warp_sums[row][warp_id] = reduced;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        for (int row = 0; row < seq_len; ++row)
        {
            float reduced = (lane < nw) ? warp_sums[row][lane] : 0.0f;
            reduced = rocm_moe_wave_reduce_sum_f32(reduced);
            if (lane == 0)
            {
                logits[static_cast<size_t>(row) * num_experts + expert] = reduced;
            }
        }
    }
}

__global__ void rocm_moe_gate_logits_single_token_fp16_weights_kernel(
    const float *__restrict__ hidden,          // [d_model]
    const __half *__restrict__ gate_weights,   // [num_experts, d_model]
    float *__restrict__ logits,                // [num_experts]
    int d_model, int num_experts)
{
    const int expert = blockIdx.x;
    if (expert >= num_experts) return;

    const __half *w = gate_weights + expert * d_model;
    float sum = 0.0f;

    if ((d_model & 1) == 0)
    {
        const int pair_count = d_model >> 1;
        const __half2 *w2 = reinterpret_cast<const __half2 *>(w);
        for (int i = threadIdx.x; i < pair_count; i += blockDim.x)
        {
            const float2 ww = __half22float2(w2[i]);
            const int h = i << 1;
            sum += hidden[h] * ww.x + hidden[h + 1] * ww.y;
        }
    }
    else
    {
        for (int i = threadIdx.x; i < d_model; i += blockDim.x)
        {
            sum += hidden[i] * __half2float(w[i]);
        }
    }

    sum = rocm_moe_wave_reduce_sum_f32(sum);

    __shared__ float warp_sums[8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0)
    {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        sum = (lane < nw) ? warp_sums[lane] : 0.0f;
        sum = rocm_moe_wave_reduce_sum_f32(sum);
        if (lane == 0) logits[expert] = sum;
    }
}

__global__ void rocm_moe_gate_logits_single_token_bf16_weights_kernel(
    const float *__restrict__ hidden,              // [d_model]
    const hip_bfloat16 *__restrict__ gate_weights, // [num_experts, d_model]
    float *__restrict__ logits,                    // [num_experts]
    int d_model, int num_experts)
{
    const int expert = blockIdx.x;
    if (expert >= num_experts) return;

    const hip_bfloat16 *w = gate_weights + expert * d_model;
    float sum = 0.0f;

    for (int i = threadIdx.x; i < d_model; i += blockDim.x)
    {
        sum += hidden[i] * static_cast<float>(w[i]);
    }

    sum = rocm_moe_wave_reduce_sum_f32(sum);

    __shared__ float warp_sums[8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0)
    {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        sum = (lane < nw) ? warp_sums[lane] : 0.0f;
        sum = rocm_moe_wave_reduce_sum_f32(sum);
        if (lane == 0) logits[expert] = sum;
    }
}

/**
 * @brief Decode-equivalent grouped FP32 router logits for verifier rows.
 *
 * One HIP block owns one expert and computes all MTP verifier rows for that
 * expert.  Gate values are loaded once per K position and reused across rows,
 * but each row keeps the exact vectorized loop and wave/block reduction shape
 * used by rocm_moe_gate_logits_single_token_kernel().
 */
__global__ void rocm_moe_gate_logits_fp32_grouped_verifier_kernel(
    const float *__restrict__ hidden,
    const float *__restrict__ gate_weights,
    float *__restrict__ logits,
    int seq_len,
    int d_model,
    int num_experts)
{
    const int expert = blockIdx.x;
    if (expert >= num_experts) return;

    const float *w = gate_weights + static_cast<size_t>(expert) * d_model;
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    if ((d_model & 3) == 0)
    {
        const int vec_count = d_model >> 2;
        const float4 *w4 = reinterpret_cast<const float4 *>(w);
        for (int i = threadIdx.x; i < vec_count; i += blockDim.x)
        {
            const float4 ww = w4[i];
            const int h = i << 2;
            for (int row = 0; row < seq_len; ++row)
            {
                const float4 hh =
                    reinterpret_cast<const float4 *>(hidden + static_cast<size_t>(row) * d_model)[i];
                sums[row] += hh.x * ww.x + hh.y * ww.y + hh.z * ww.z + hh.w * ww.w;
            }
            (void)h;
        }
    }
    else
    {
        for (int i = threadIdx.x; i < d_model; i += blockDim.x)
        {
            const float ww = w[i];
            for (int row = 0; row < seq_len; ++row)
                sums[row] += hidden[static_cast<size_t>(row) * d_model + i] * ww;
        }
    }

    __shared__ float warp_sums[4][8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    for (int row = 0; row < seq_len; ++row)
    {
        float reduced = rocm_moe_wave_reduce_sum_f32(sums[row]);
        if (lane == 0) warp_sums[row][warp_id] = reduced;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        for (int row = 0; row < seq_len; ++row)
        {
            float reduced = (lane < nw) ? warp_sums[row][lane] : 0.0f;
            reduced = rocm_moe_wave_reduce_sum_f32(reduced);
            if (lane == 0)
                logits[static_cast<size_t>(row) * num_experts + expert] = reduced;
        }
    }
}

/**
 * @brief Decode-equivalent grouped FP16-router logits for verifier rows.
 *
 * Mirrors rocm_moe_gate_logits_single_token_fp16_weights_kernel() per row,
 * including the half2 loop for even K, while sharing each gate load across the
 * verifier rows handled by the expert-owned block.
 */
__global__ void rocm_moe_gate_logits_fp16_grouped_verifier_kernel(
    const float *__restrict__ hidden,
    const __half *__restrict__ gate_weights,
    float *__restrict__ logits,
    int seq_len,
    int d_model,
    int num_experts)
{
    const int expert = blockIdx.x;
    if (expert >= num_experts) return;

    const __half *w = gate_weights + static_cast<size_t>(expert) * d_model;
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    if ((d_model & 1) == 0)
    {
        const int pair_count = d_model >> 1;
        const __half2 *w2 = reinterpret_cast<const __half2 *>(w);
        for (int i = threadIdx.x; i < pair_count; i += blockDim.x)
        {
            const float2 ww = __half22float2(w2[i]);
            const int h = i << 1;
            for (int row = 0; row < seq_len; ++row)
            {
                const float *row_hidden = hidden + static_cast<size_t>(row) * d_model;
                sums[row] += row_hidden[h] * ww.x + row_hidden[h + 1] * ww.y;
            }
        }
    }
    else
    {
        for (int i = threadIdx.x; i < d_model; i += blockDim.x)
        {
            const float ww = __half2float(w[i]);
            for (int row = 0; row < seq_len; ++row)
                sums[row] += hidden[static_cast<size_t>(row) * d_model + i] * ww;
        }
    }

    __shared__ float warp_sums[4][8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    for (int row = 0; row < seq_len; ++row)
    {
        float reduced = rocm_moe_wave_reduce_sum_f32(sums[row]);
        if (lane == 0) warp_sums[row][warp_id] = reduced;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        for (int row = 0; row < seq_len; ++row)
        {
            float reduced = (lane < nw) ? warp_sums[row][lane] : 0.0f;
            reduced = rocm_moe_wave_reduce_sum_f32(reduced);
            if (lane == 0)
                logits[static_cast<size_t>(row) * num_experts + expert] = reduced;
        }
    }
}

/**
 * @brief Decode-equivalent grouped BF16-router logits for verifier rows.
 */
__global__ void rocm_moe_gate_logits_bf16_grouped_verifier_kernel(
    const float *__restrict__ hidden,
    const hip_bfloat16 *__restrict__ gate_weights,
    float *__restrict__ logits,
    int seq_len,
    int d_model,
    int num_experts)
{
    const int expert = blockIdx.x;
    if (expert >= num_experts) return;

    const hip_bfloat16 *w = gate_weights + static_cast<size_t>(expert) * d_model;
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = threadIdx.x; i < d_model; i += blockDim.x)
    {
        const float ww = static_cast<float>(w[i]);
        for (int row = 0; row < seq_len; ++row)
            sums[row] += hidden[static_cast<size_t>(row) * d_model + i] * ww;
    }

    __shared__ float warp_sums[4][8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    for (int row = 0; row < seq_len; ++row)
    {
        float reduced = rocm_moe_wave_reduce_sum_f32(sums[row]);
        if (lane == 0) warp_sums[row][warp_id] = reduced;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        for (int row = 0; row < seq_len; ++row)
        {
            float reduced = (lane < nw) ? warp_sums[row][lane] : 0.0f;
            reduced = rocm_moe_wave_reduce_sum_f32(reduced);
            if (lane == 0)
                logits[static_cast<size_t>(row) * num_experts + expert] = reduced;
        }
    }
}

// =========================================================================
// Kernel 1b: Single-token Gate Logits, Grouped Experts
//
// Each CTA handles a contiguous group of experts and loads the decode hidden
// vector into LDS once. On gfx906 this maps four wave64 waves to four experts,
// avoiding cross-wave reductions while preserving the original logits layout.
// =========================================================================

__global__ void rocm_moe_gate_logits_single_token_grouped_kernel(
    const float *__restrict__ hidden,       // [d_model]
    const float *__restrict__ gate_weights, // [num_experts, d_model]
    float *__restrict__ logits,             // [num_experts]
    int d_model, int num_experts,
    bool use_vec4)
{
    extern __shared__ float4 hidden_shared_vec4[];
    float *hidden_shared = reinterpret_cast<float *>(hidden_shared_vec4);

    if (use_vec4)
    {
        const int vec_count = d_model >> 2;
        const float4 *hidden4 = reinterpret_cast<const float4 *>(hidden);
        for (int i = threadIdx.x; i < vec_count; i += blockDim.x)
        {
            hidden_shared_vec4[i] = hidden4[i];
        }
    }
    else
    {
        for (int i = threadIdx.x; i < d_model; i += blockDim.x)
        {
            hidden_shared[i] = hidden[i];
        }
    }
    __syncthreads();

    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (warp_id >= kGateLogitsExpertsPerBlock)
        return;

    const int expert = blockIdx.x * kGateLogitsExpertsPerBlock + warp_id;
    if (expert >= num_experts)
        return;

    const float *w = gate_weights + expert * d_model;
    float sum = 0.0f;

    if (use_vec4)
    {
        const int vec_count = d_model >> 2;
        const float4 *hidden4 = reinterpret_cast<const float4 *>(hidden_shared);
        const float4 *w4 = reinterpret_cast<const float4 *>(w);
        for (int i = lane; i < vec_count; i += warpSize)
        {
            const float4 h = hidden4[i];
            const float4 ww = w4[i];
            sum += h.x * ww.x + h.y * ww.y + h.z * ww.z + h.w * ww.w;
        }
    }
    else
    {
        for (int i = lane; i < d_model; i += warpSize)
        {
            sum += hidden_shared[i] * w[i];
        }
    }

    sum = rocm_moe_wave_reduce_sum_f32(sum);

    if (lane == 0)
        logits[expert] = sum;
}

// =========================================================================
// Kernel 2: Row-wise Softmax + Top-K Selection
//
// For each token:
//   1. Find max logit (numerical stability)
//   2. Compute exp(logit - max) and sum
//   3. Normalize to probabilities
//   4. Select top-k experts by probability
//   5. Optionally renormalize top-k weights
//
// One block per token. Uses shared memory for the logits array.
// =========================================================================

__global__ void rocm_moe_softmax_topk_kernel(
    float *__restrict__ logits,             // [seq_len, num_experts] IN/OUT (probabilities)
    int *__restrict__ expert_indices,       // [seq_len * top_k] OUT
    float *__restrict__ expert_weights,     // [seq_len * top_k] OUT
    int seq_len, int num_experts, int top_k,
    bool normalize_weights,
    const int *__restrict__ effective_seq_len_ptr)
{
    const int token = blockIdx.x;
    if (token >= seq_len) return;

    int effective_seq_len = seq_len;
    if (effective_seq_len_ptr)
    {
        const int raw_effective = *effective_seq_len_ptr;
        effective_seq_len = raw_effective < 1 ? 1 : (raw_effective > seq_len ? seq_len : raw_effective);
    }
    if (token >= effective_seq_len)
    {
        if (threadIdx.x == 0)
        {
            for (int k = 0; k < top_k; ++k)
            {
                const size_t out = static_cast<size_t>(token) * top_k + k;
                expert_indices[out] = -1;
                expert_weights[out] = 0.0f;
            }
        }
        for (int e = threadIdx.x; e < num_experts; e += blockDim.x)
            logits[static_cast<size_t>(token) * num_experts + e] = 0.0f;
        return;
    }

    float *row = logits + token * num_experts;
    __shared__ float values[kDeviceMoEMaxExperts];
    __shared__ float reductions[256];
    __shared__ int reduction_indices[256];
    __shared__ int selected_indices[kDeviceMoEMaxTopK];
    __shared__ float selected_weights[kDeviceMoEMaxTopK];

    // Step 1: Find max (thread-parallel)
    float local_max = -FLT_MAX;
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x)
    {
        local_max = fmaxf(local_max, row[e]);
    }
    // Warp reduction for max
    local_max = rocm_moe_wave_reduce_max_f32(local_max);

    __shared__ float shared_max[8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) shared_max[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0)
    {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        local_max = (lane < nw) ? shared_max[lane] : -FLT_MAX;
        local_max = rocm_moe_wave_reduce_max_f32(local_max);
        if (lane == 0) shared_max[0] = local_max;
    }
    __syncthreads();
    float max_val = shared_max[0];

    // Step 2: Compute exp and sum
    float local_sum = 0.0f;
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x)
    {
        float val = expf(row[e] - max_val);
        row[e] = val;
        local_sum += val;
    }
    // Warp reduction for sum
    local_sum = rocm_moe_wave_reduce_sum_f32(local_sum);

    __shared__ float shared_sum[8];
    if (lane == 0) shared_sum[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0)
    {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        local_sum = (lane < nw) ? shared_sum[lane] : 0.0f;
        local_sum = rocm_moe_wave_reduce_sum_f32(local_sum);
        if (lane == 0) shared_sum[0] = local_sum;
    }
    __syncthreads();
    float sum_val = shared_sum[0];
    float inv_sum = (sum_val > 0.0f) ? 1.0f / sum_val : 0.0f;

    // Step 3: Normalize probabilities
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x)
    {
        const float prob = row[e] * inv_sum;
        row[e] = prob;
        values[e] = prob;
    }
    __syncthreads();

    /**
     * Select top-k with a block-wide argmax reduction.
     *
     * The old ROCm path did the whole top_k * num_experts scan on lane 0 after
     * softmax. MTP verifier rows execute this kernel for every MoE layer, so
     * leaving 255 lanes idle made routing dominate ROCm verifier time. This
     * mirrors the CUDA contract: each round finds one winner in parallel, masks
     * it from the shared-memory copy only, and leaves row[] intact for callers
     * that inspect router probabilities. Ties intentionally choose the lowest
     * expert id, matching the previous ascending serial scan.
     */
    float topk_sum = 0.0f;
    for (int k = 0; k < top_k; ++k)
    {
        float best_value = -1.0f;
        int best_idx = 0;
        for (int e = threadIdx.x; e < num_experts; e += blockDim.x)
        {
            const float value = values[e];
            if (value > best_value)
            {
                best_value = value;
                best_idx = e;
            }
        }
        reductions[threadIdx.x] = best_value;
        reduction_indices[threadIdx.x] = best_idx;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
            {
                const float other = reductions[threadIdx.x + stride];
                const int other_idx = reduction_indices[threadIdx.x + stride];
                const float current = reductions[threadIdx.x];
                const int current_idx = reduction_indices[threadIdx.x];
                if (other > current || (other == current && other_idx < current_idx))
                {
                    reductions[threadIdx.x] = other;
                    reduction_indices[threadIdx.x] = other_idx;
                }
            }
            __syncthreads();
        }

        const int winner = reduction_indices[0];
        const float winner_value = reductions[0];
        topk_sum += winner_value;
        if (threadIdx.x == 0)
        {
            selected_indices[k] = winner;
            selected_weights[k] = winner_value;
            values[winner] = -1.0f;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        int *out_idx = expert_indices + token * top_k;
        float *out_wt = expert_weights + token * top_k;
        const float inv_topk_sum = (normalize_weights && topk_sum > 0.0f) ? 1.0f / topk_sum : 1.0f;
        for (int k = 0; k < top_k; ++k)
        {
            out_idx[k] = selected_indices[k];
            out_wt[k] = selected_weights[k] * inv_topk_sum;
        }
    }
}

// =========================================================================
// Kernel 2b: Decode Softmax + Top-K + Runtime Write
//
// Single-token decode variant that keeps routing entirely device-side: it
// converts logits to probabilities, selects top-k, writes the stable runtime
// table, optionally mirrors legacy FP32 routing tensors, and updates the
// decode histogram in one launch.
// =========================================================================

__global__ void rocm_moe_softmax_topk_decode_runtime_kernel(
    float *__restrict__ logits,                       // [num_experts] IN/OUT (probabilities)
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    DeviceMoELayerRuntimeView *__restrict__ runtime_layers,
    float *__restrict__ legacy_indices,               // optional [top_k]
    float *__restrict__ legacy_weights,               // optional [top_k]
    int num_experts, int top_k,
    bool normalize_weights,
    bool write_legacy_outputs,
    bool update_runtime_histogram,
    const DeviceMoERebalancePlanEntryView *__restrict__ rebalance_plan_entries,
    uint32_t rebalance_plan_capacity,
    DeviceMoERebalanceCommandBufferHeaderView *__restrict__ rebalance_command_header,
    const DeviceMoEExpertDirectoryEntryView *__restrict__ rebalance_local_transfer_slots,
    uint32_t rebalance_local_transfer_slot_count,
    DeviceMoERebalanceConfigView rebalance_config,
    DeviceMoERebalanceApplyStatusView *__restrict__ rebalance_apply_status,
    DeviceMoERebalanceGraphControllerStateView *__restrict__ rebalance_controller_state,
    int rebalance_target_layer,
    uint32_t rebalance_command_buffer_count)
{
    if (!logits || !runtime || num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK)
    {
        return;
    }

    try_apply_ready_rebalance_wave_for_layer_thread0(
        runtime_layers,
        rebalance_plan_entries,
        rebalance_plan_capacity,
        rebalance_command_header,
        rebalance_local_transfer_slots,
        rebalance_local_transfer_slot_count,
        rebalance_config,
        rebalance_apply_status,
        rebalance_controller_state,
        rebalance_target_layer,
        rebalance_command_buffer_count);

    float local_max = -FLT_MAX;
    for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
    {
        local_max = fmaxf(local_max, logits[expert]);
    }
    local_max = rocm_moe_wave_reduce_max_f32(local_max);

    __shared__ float shared_max[8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) shared_max[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0)
    {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        local_max = (lane < num_warps) ? shared_max[lane] : -FLT_MAX;
        local_max = rocm_moe_wave_reduce_max_f32(local_max);
        if (lane == 0) shared_max[0] = local_max;
    }
    __syncthreads();
    const float max_val = shared_max[0];

    float local_sum = 0.0f;
    for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
    {
        const float value = expf(logits[expert] - max_val);
        logits[expert] = value;
        local_sum += value;
    }
    local_sum = rocm_moe_wave_reduce_sum_f32(local_sum);

    __shared__ float shared_sum[8];
    if (lane == 0) shared_sum[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0)
    {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        local_sum = (lane < num_warps) ? shared_sum[lane] : 0.0f;
        local_sum = rocm_moe_wave_reduce_sum_f32(local_sum);
        if (lane == 0) shared_sum[0] = local_sum;
    }
    __syncthreads();

    const float inv_sum = (shared_sum[0] > 0.0f) ? 1.0f / shared_sum[0] : 0.0f;
    for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
    {
        logits[expert] *= inv_sum;
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        int selected_ids[kDeviceMoEMaxTopK];
        float selected_weights[kDeviceMoEMaxTopK];

        for (int slot = 0; slot < top_k; ++slot)
        {
            selected_ids[slot] = -1;
            selected_weights[slot] = 0.0f;
        }

        for (int slot = 0; slot < top_k; ++slot)
        {
            float best_value = -1.0f;
            int best_expert = 0;
            for (int expert = 0; expert < num_experts; ++expert)
            {
                if (logits[expert] > best_value)
                {
                    bool already_selected = false;
                    for (int previous = 0; previous < slot; ++previous)
                    {
                        if (selected_ids[previous] == expert)
                        {
                            already_selected = true;
                            break;
                        }
                    }
                    if (!already_selected)
                    {
                        best_value = logits[expert];
                        best_expert = expert;
                    }
                }
            }
            selected_ids[slot] = best_expert;
            selected_weights[slot] = best_value;
        }

        if (normalize_weights)
        {
            float topk_sum = 0.0f;
            for (int slot = 0; slot < top_k; ++slot)
            {
                topk_sum += selected_weights[slot];
            }
            if (topk_sum > 0.0f)
            {
                const float inv_topk_sum = 1.0f / topk_sum;
                for (int slot = 0; slot < top_k; ++slot)
                {
                    selected_weights[slot] *= inv_topk_sum;
                }
            }
        }

        runtime_publish_decode_dispatch(
            runtime, selected_ids, selected_weights, num_experts, top_k,
            legacy_indices, legacy_weights, write_legacy_outputs, update_runtime_histogram);
    }
}

__global__ void rocm_moe_softmax_topk_decode_runtime_wave64_kernel(
    const float *__restrict__ logits,                 // [num_experts] input logits, left unmodified
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    DeviceMoELayerRuntimeView *__restrict__ runtime_layers,
    float *__restrict__ legacy_indices,               // optional [top_k]
    float *__restrict__ legacy_weights,               // optional [top_k]
    int num_experts, int top_k,
    bool normalize_weights,
    bool write_legacy_outputs,
    bool update_runtime_histogram,
    const DeviceMoERebalancePlanEntryView *__restrict__ rebalance_plan_entries,
    uint32_t rebalance_plan_capacity,
    DeviceMoERebalanceCommandBufferHeaderView *__restrict__ rebalance_command_header,
    const DeviceMoEExpertDirectoryEntryView *__restrict__ rebalance_local_transfer_slots,
    uint32_t rebalance_local_transfer_slot_count,
    DeviceMoERebalanceConfigView rebalance_config,
    DeviceMoERebalanceApplyStatusView *__restrict__ rebalance_apply_status,
    DeviceMoERebalanceGraphControllerStateView *__restrict__ rebalance_controller_state,
    int rebalance_target_layer,
    uint32_t rebalance_command_buffer_count)
{
    if (!logits || !runtime || num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK)
    {
        return;
    }

    try_apply_ready_rebalance_wave_for_layer_thread0(
        runtime_layers,
        rebalance_plan_entries,
        rebalance_plan_capacity,
        rebalance_command_header,
        rebalance_local_transfer_slots,
        rebalance_local_transfer_slot_count,
        rebalance_config,
        rebalance_apply_status,
        rebalance_controller_state,
        rebalance_target_layer,
        rebalance_command_buffer_count);

    __shared__ float expert_probs[kDeviceMoEMaxExperts];
    __shared__ float shared_max[8];
    __shared__ float shared_sum[8];

    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;

    float max_val = -FLT_MAX;
    for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
    {
        max_val = fmaxf(max_val, logits[expert]);
    }
    max_val = rocm_moe_wave_reduce_max_f32(max_val);
    if (lane == 0) shared_max[warp_id] = max_val;
    __syncthreads();

    if (warp_id == 0)
    {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        max_val = (lane < num_warps) ? shared_max[lane] : -FLT_MAX;
        max_val = rocm_moe_wave_reduce_max_f32(max_val);
        if (lane == 0) shared_max[0] = max_val;
    }
    __syncthreads();
    max_val = shared_max[0];

    float sum = 0.0f;
    for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
    {
        const float value = expf(logits[expert] - max_val);
        expert_probs[expert] = value;
        sum += value;
    }
    sum = rocm_moe_wave_reduce_sum_f32(sum);
    if (lane == 0) shared_sum[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0)
    {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        sum = (lane < num_warps) ? shared_sum[lane] : 0.0f;
        sum = rocm_moe_wave_reduce_sum_f32(sum);
        if (lane == 0) shared_sum[0] = sum;
    }
    __syncthreads();

    const float inv_sum = (shared_sum[0] > 0.0f) ? 1.0f / shared_sum[0] : 0.0f;
    for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
    {
        expert_probs[expert] *= inv_sum;
    }
    __syncthreads();

    __shared__ int selected_ids[kDeviceMoEMaxTopK];
    __shared__ float selected_weights[kDeviceMoEMaxTopK];
    moe_select_topk_probabilities_block(expert_probs, num_experts, top_k, selected_ids, selected_weights);

    if (threadIdx.x == 0)
    {
        if (normalize_weights)
        {
            float topk_sum = 0.0f;
            for (int slot = 0; slot < top_k; ++slot)
            {
                topk_sum += selected_weights[slot];
            }
            if (topk_sum > 0.0f)
            {
                const float inv_topk_sum = 1.0f / topk_sum;
                for (int slot = 0; slot < top_k; ++slot)
                {
                    selected_weights[slot] *= inv_topk_sum;
                }
            }
        }

        runtime_publish_decode_dispatch(
            runtime, selected_ids, selected_weights, num_experts, top_k,
            legacy_indices, legacy_weights, write_legacy_outputs, update_runtime_histogram);
    }
}

/**
 * @brief Decode-equivalent softmax/top-k for grouped MTP verifier rows.
 *
 * The MTP verifier is allowed to publish grouped rows into live decode state, so
 * its routing weights must be byte-identical to running each row through the
 * production single-token decode router.  This kernel keeps the grouped launch
 * economical (one block per verifier row) while intentionally mirroring
 * rocm_moe_softmax_topk_decode_runtime_wave64_kernel's probability reduction,
 * top-k selection, and top-k renormalization order inside each block.
 */
__global__ void rocm_moe_softmax_topk_decode_equivalent_rows_kernel(
    const float *__restrict__ logits,                 // [seq_len, num_experts] input logits, left unmodified
    int *__restrict__ expert_indices,                 // [seq_len, top_k] OUT
    float *__restrict__ expert_weights,               // [seq_len, top_k] OUT
    int seq_len, int num_experts, int top_k,
    bool normalize_weights)
{
    const int token = blockIdx.x;
    if (!logits || !expert_indices || !expert_weights ||
        token >= seq_len || num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK)
    {
        return;
    }

    const float *row_logits = logits + static_cast<size_t>(token) * num_experts;

    __shared__ float expert_probs[kDeviceMoEMaxExperts];
    __shared__ float shared_max[8];
    __shared__ float shared_sum[8];

    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;

    float max_val = -FLT_MAX;
    for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
    {
        max_val = fmaxf(max_val, row_logits[expert]);
    }
    max_val = rocm_moe_wave_reduce_max_f32(max_val);
    if (lane == 0) shared_max[warp_id] = max_val;
    __syncthreads();

    if (warp_id == 0)
    {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        max_val = (lane < num_warps) ? shared_max[lane] : -FLT_MAX;
        max_val = rocm_moe_wave_reduce_max_f32(max_val);
        if (lane == 0) shared_max[0] = max_val;
    }
    __syncthreads();
    max_val = shared_max[0];

    float sum = 0.0f;
    for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
    {
        const float value = expf(row_logits[expert] - max_val);
        expert_probs[expert] = value;
        sum += value;
    }
    sum = rocm_moe_wave_reduce_sum_f32(sum);
    if (lane == 0) shared_sum[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0)
    {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        sum = (lane < num_warps) ? shared_sum[lane] : 0.0f;
        sum = rocm_moe_wave_reduce_sum_f32(sum);
        if (lane == 0) shared_sum[0] = sum;
    }
    __syncthreads();

    const float inv_sum = (shared_sum[0] > 0.0f) ? 1.0f / shared_sum[0] : 0.0f;
    for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
    {
        expert_probs[expert] *= inv_sum;
    }
    __syncthreads();

    __shared__ int selected_ids[kDeviceMoEMaxTopK];
    __shared__ float selected_weights[kDeviceMoEMaxTopK];
    moe_select_topk_probabilities_block(expert_probs, num_experts, top_k, selected_ids, selected_weights);

    if (threadIdx.x == 0)
    {
        if (normalize_weights)
        {
            float topk_sum = 0.0f;
            for (int slot = 0; slot < top_k; ++slot)
            {
                topk_sum += selected_weights[slot];
            }
            if (topk_sum > 0.0f)
            {
                const float inv_topk_sum = 1.0f / topk_sum;
                for (int slot = 0; slot < top_k; ++slot)
                {
                    selected_weights[slot] *= inv_topk_sum;
                }
            }
        }

        int *out_idx = expert_indices + static_cast<size_t>(token) * top_k;
        float *out_wt = expert_weights + static_cast<size_t>(token) * top_k;
        for (int slot = 0; slot < top_k; ++slot)
        {
            out_idx[slot] = selected_ids[slot];
            out_wt[slot] = selected_weights[slot];
        }
    }
}

__global__ void rocm_moe_router_kpart_reduce_softmax_topk_decode_runtime_kernel(
    const float *__restrict__ partials,                // [num_experts, k_partitions]
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    DeviceMoELayerRuntimeView *__restrict__ runtime_layers,
    float *__restrict__ legacy_indices,                // optional [top_k]
    float *__restrict__ legacy_weights,                // optional [top_k]
    int num_experts, int k_partitions, int top_k,
    bool normalize_weights,
    bool write_legacy_outputs,
    bool update_runtime_histogram,
    const DeviceMoERebalancePlanEntryView *__restrict__ rebalance_plan_entries,
    uint32_t rebalance_plan_capacity,
    DeviceMoERebalanceCommandBufferHeaderView *__restrict__ rebalance_command_header,
    const DeviceMoEExpertDirectoryEntryView *__restrict__ rebalance_local_transfer_slots,
    uint32_t rebalance_local_transfer_slot_count,
    DeviceMoERebalanceConfigView rebalance_config,
    DeviceMoERebalanceApplyStatusView *__restrict__ rebalance_apply_status,
    DeviceMoERebalanceGraphControllerStateView *__restrict__ rebalance_controller_state,
    int rebalance_target_layer,
    uint32_t rebalance_command_buffer_count)
{
    if (!partials || !runtime || num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        k_partitions <= 0 || top_k <= 0 || top_k > kDeviceMoEMaxTopK)
    {
        return;
    }

    try_apply_ready_rebalance_wave_for_layer_thread0(
        runtime_layers,
        rebalance_plan_entries,
        rebalance_plan_capacity,
        rebalance_command_header,
        rebalance_local_transfer_slots,
        rebalance_local_transfer_slot_count,
        rebalance_config,
        rebalance_apply_status,
        rebalance_controller_state,
        rebalance_target_layer,
        rebalance_command_buffer_count);

    __shared__ float expert_probs[kDeviceMoEMaxExperts];

    float local_max = -FLT_MAX;
    for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
    {
        const float *expert_partials = partials + expert * k_partitions;
        float sum = 0.0f;
        for (int partition = 0; partition < k_partitions; ++partition)
        {
            sum += expert_partials[partition];
        }
        expert_probs[expert] = sum;
        local_max = fmaxf(local_max, sum);
    }

    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    local_max = rocm_moe_wave_reduce_max_f32(local_max);

    __shared__ float shared_max[8];
    if (lane == 0) shared_max[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0)
    {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        local_max = (lane < num_warps) ? shared_max[lane] : -FLT_MAX;
        local_max = rocm_moe_wave_reduce_max_f32(local_max);
        if (lane == 0) shared_max[0] = local_max;
    }
    __syncthreads();
    const float max_val = shared_max[0];

    float local_sum = 0.0f;
    for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
    {
        const float value = expf(expert_probs[expert] - max_val);
        expert_probs[expert] = value;
        local_sum += value;
    }
    local_sum = rocm_moe_wave_reduce_sum_f32(local_sum);

    __shared__ float shared_sum[8];
    if (lane == 0) shared_sum[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0)
    {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        local_sum = (lane < num_warps) ? shared_sum[lane] : 0.0f;
        local_sum = rocm_moe_wave_reduce_sum_f32(local_sum);
        if (lane == 0) shared_sum[0] = local_sum;
    }
    __syncthreads();

    const float inv_sum = (shared_sum[0] > 0.0f) ? 1.0f / shared_sum[0] : 0.0f;
    for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
    {
        expert_probs[expert] *= inv_sum;
    }
    __syncthreads();

    __shared__ int selected_ids[kDeviceMoEMaxTopK];
    __shared__ float selected_weights[kDeviceMoEMaxTopK];
    moe_select_topk_probabilities_block(expert_probs, num_experts, top_k, selected_ids, selected_weights);

    if (threadIdx.x == 0)
    {
        if (normalize_weights)
        {
            float topk_sum = 0.0f;
            for (int slot = 0; slot < top_k; ++slot)
            {
                topk_sum += selected_weights[slot];
            }
            if (topk_sum > 0.0f)
            {
                const float inv_topk_sum = 1.0f / topk_sum;
                for (int slot = 0; slot < top_k; ++slot)
                {
                    selected_weights[slot] *= inv_topk_sum;
                }
            }
        }

        runtime_publish_decode_dispatch(
            runtime, selected_ids, selected_weights, num_experts, top_k,
            legacy_indices, legacy_weights, write_legacy_outputs, update_runtime_histogram);
    }
}

// =========================================================================
// Kernel 2c: Decode Route Select Runtime Copy
//
// Copies single-token top-k ids/weights into the stable device runtime table,
// optionally mirrors them to legacy routing tensors, and records a device-side
// decode histogram. The routing tensors remain necessary until the expert
// decode kernels consume runtime top-k directly.
// =========================================================================

__global__ void rocm_moe_decode_route_select_runtime_kernel(
    const int *__restrict__ expert_indices,       // [top_k]
    const float *__restrict__ expert_weights,     // [top_k]
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    float *__restrict__ legacy_indices,           // optional [top_k]
    float *__restrict__ legacy_weights,           // optional [top_k]
    int num_experts, int top_k,
    bool write_legacy_outputs,
    bool update_runtime_histogram)
{
    if (threadIdx.x != 0 || !runtime) return;
    runtime_publish_decode_dispatch(
        runtime, expert_indices, expert_weights, num_experts, top_k,
        legacy_indices, legacy_weights, write_legacy_outputs, update_runtime_histogram);
}

// =========================================================================
// Kernel 3: Gather Tokens
//
// Copy selected token rows into a contiguous batch buffer.
// Each block handles one token copy; threads cover the d_model dimension.
// =========================================================================

__global__ void rocm_moe_gather_tokens_kernel(
    const float *__restrict__ hidden,        // [seq_len, d_model]
    float *__restrict__ batch_buffer,        // [num_tokens, d_model]
    const int *__restrict__ token_indices,   // [num_tokens]
    int num_tokens, int d_model)
{
    const int token = blockIdx.x;
    if (token >= num_tokens) return;

    const int src_row = token_indices[token];
    const float *src = hidden + src_row * d_model;
    float *dst = batch_buffer + token * d_model;

    // Vectorized copy: try float4 first
    const int d_model4 = d_model >> 2;
    const float4 *src4 = reinterpret_cast<const float4 *>(src);
    float4 *dst4 = reinterpret_cast<float4 *>(dst);

    for (int i = threadIdx.x; i < d_model4; i += blockDim.x)
    {
        dst4[i] = src4[i];
    }

    // Handle tail elements
    const int tail_start = d_model4 << 2;
    for (int i = tail_start + threadIdx.x; i < d_model; i += blockDim.x)
    {
        dst[i] = src[i];
    }
}

/**
 * @brief Copy one row using a by-value row index.
 *
 * Verifier replay uses this instead of the general gather kernel so graph and
 * stream replay do not depend on an async H2D copy from a stack-owned host row
 * index.  The row width is normally a multiple of four, but the scalar tail
 * keeps the helper valid for routing rows such as top_k=8 as well.
 */
__global__ void rocm_moe_copy_row_kernel(
    const float *__restrict__ source,
    float *__restrict__ row_buffer,
    int row_index,
    int row_width)
{
    if (row_index < 0 || row_width <= 0)
        return;

    const float *src = source + static_cast<size_t>(row_index) * row_width;
    const int row_width4 = row_width >> 2;
    const float4 *src4 = reinterpret_cast<const float4 *>(src);
    float4 *dst4 = reinterpret_cast<float4 *>(row_buffer);
    for (int i = threadIdx.x; i < row_width4; i += blockDim.x)
        dst4[i] = src4[i];

    const int tail_start = row_width4 << 2;
    for (int i = tail_start + threadIdx.x; i < row_width; i += blockDim.x)
        row_buffer[i] = src[i];
}

// =========================================================================
// Kernel 4: Scatter-Add Weighted
//
// output[token_indices[i]] += weights[i] * expert_output[i]
// Each block handles one token; threads cover d_model.
// =========================================================================

__global__ void rocm_moe_scatter_add_kernel(
    float *__restrict__ output,             // [seq_len, d_model]
    const float *__restrict__ expert_output, // [num_tokens, d_model]
    const int *__restrict__ token_indices,   // [num_tokens]
    const float *__restrict__ weights,       // [num_tokens]
    int num_tokens, int d_model)
{
    const int token = blockIdx.x;
    if (token >= num_tokens) return;

    const int dst_row = token_indices[token];
    const float w = weights[token];
    const float *src = expert_output + token * d_model;
    float *dst = output + dst_row * d_model;

    for (int i = threadIdx.x; i < d_model; i += blockDim.x)
    {
        atomicAdd(&dst[i], w * src[i]);
    }
}

/**
 * @brief Store one scratch row into a destination row selected by value.
 *
 * The operation is an overwrite, not an add.  Decode-equivalent verifier replay
 * owns each destination row exactly once, so direct store avoids atomics and
 * avoids staging a `{row, weight}` pair through host memory.
 */
__global__ void rocm_moe_write_row_kernel(
    float *__restrict__ destination,
    const float *__restrict__ row_buffer,
    int row_index,
    int row_width)
{
    if (row_index < 0 || row_width <= 0)
        return;

    float *dst = destination + static_cast<size_t>(row_index) * row_width;
    const int row_width4 = row_width >> 2;
    const float4 *src4 = reinterpret_cast<const float4 *>(row_buffer);
    float4 *dst4 = reinterpret_cast<float4 *>(dst);
    for (int i = threadIdx.x; i < row_width4; i += blockDim.x)
        dst4[i] = src4[i];

    const int tail_start = row_width4 << 2;
    for (int i = tail_start + threadIdx.x; i < row_width; i += blockDim.x)
        dst[i] = row_buffer[i];
}

// =========================================================================
// Kernel 5: Shared Expert Gate
//
// For each token:
//   gate = sigmoid(dot(gate_inp, input[t]))
//   shared_output[t] *= gate
//
// Two-phase: first compute dot products (reduction), then scale.
// =========================================================================

__global__ void rocm_moe_shared_expert_gate_dot_kernel(
    const float *__restrict__ input,        // [seq_len, d_model]
    const float *__restrict__ gate_inp,     // [d_model]
    float *__restrict__ gate_values,        // [seq_len] scratch
    int seq_len, int d_model)
{
    const int token = blockIdx.x;
    if (token >= seq_len) return;

    const float *x = input + token * d_model;

    /*
     * MTP verifier rows compare the grouped M=2..4 shared-gate path against
     * serial M=1 decode row replay.  The serial ROCm decode kernel below
     * accumulates four contiguous FP32 products at a time before participating
     * in the block reduction.  Use the same per-lane accumulation order here;
     * a scalar stride loop is mathematically equivalent, but it changes FP32
     * rounding by a few ulps and those differences can amplify after later MoE
     * and residual stages.
     */
    float sum = 0.0f;
    if ((d_model & 3) == 0)
    {
        const int d4 = d_model >> 2;
        const float4 *__restrict__ input4 = reinterpret_cast<const float4 *>(x);
        const float4 *__restrict__ gate4 = reinterpret_cast<const float4 *>(gate_inp);
        for (int i = threadIdx.x; i < d4; i += blockDim.x)
        {
            const float4 xv = input4[i];
            const float4 gv = gate4[i];
            sum += xv.x * gv.x + xv.y * gv.y + xv.z * gv.z + xv.w * gv.w;
        }
    }
    else
    {
        for (int i = threadIdx.x; i < d_model; i += blockDim.x)
        {
            sum += x[i] * gate_inp[i];
        }
    }

    // Wave reduction
    sum = rocm_moe_wave_reduce_sum_f32(sum);

    __shared__ float warp_sums[8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0)
    {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        sum = (lane < nw) ? warp_sums[lane] : 0.0f;
        sum = rocm_moe_wave_reduce_sum_f32(sum);
        if (lane == 0)
        {
            // Apply sigmoid
            gate_values[token] = 1.0f / (1.0f + expf(-sum));
        }
    }
}

__device__ __forceinline__ int rocm_moe_clamp_effective_seq_len(
    int seq_len,
    const int *__restrict__ device_effective_seq_len)
{
    if (!device_effective_seq_len)
        return seq_len;
    const int raw = *device_effective_seq_len;
    return raw < 0 ? 0 : (raw > seq_len ? seq_len : raw);
}

__global__ void rocm_moe_shared_expert_gate_scale_kernel(
    float *__restrict__ shared_output,      // [seq_len, d_model]
    const float *__restrict__ gate_values,  // [seq_len]
    int seq_len, int d_model,
    const int *__restrict__ device_effective_seq_len = nullptr)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = seq_len * d_model;
    if (idx >= total) return;

    const int token = idx / d_model;
    const int effective_seq_len =
        rocm_moe_clamp_effective_seq_len(seq_len, device_effective_seq_len);
    if (token >= effective_seq_len)
    {
        shared_output[idx] = 0.0f;
        return;
    }
    shared_output[idx] *= gate_values[token];
}

__global__ void rocm_moe_shared_expert_gate_decode_fused_kernel(
    const float *__restrict__ input,        // [d_model], token 0
    const float *__restrict__ gate_inp,     // [d_model]
    float *__restrict__ shared_output,      // [d_model], token 0
    int d_model)
{
    float sum = 0.0f;
    if ((d_model & 3) == 0)
    {
        const int d4 = d_model >> 2;
        const float4 *__restrict__ input4 = reinterpret_cast<const float4 *>(input);
        const float4 *__restrict__ gate4 = reinterpret_cast<const float4 *>(gate_inp);
        for (int i = threadIdx.x; i < d4; i += blockDim.x)
        {
            const float4 x = input4[i];
            const float4 g = gate4[i];
            sum += x.x * g.x + x.y * g.y + x.z * g.z + x.w * g.w;
        }
    }
    else
    {
        for (int i = threadIdx.x; i < d_model; i += blockDim.x)
        {
            sum += input[i] * gate_inp[i];
        }
    }

    sum = rocm_moe_wave_reduce_sum_f32(sum);

    __shared__ float warp_sums[16];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0)
    {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        float block_sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        block_sum = rocm_moe_wave_reduce_sum_f32(block_sum);
        if (lane == 0)
            warp_sums[0] = 1.0f / (1.0f + expf(-block_sum));
    }
    __syncthreads();

    const float gate = warp_sums[0];
    if ((d_model & 3) == 0)
    {
        const int d4 = d_model >> 2;
        float4 *__restrict__ shared4 = reinterpret_cast<float4 *>(shared_output);
        for (int i = threadIdx.x; i < d4; i += blockDim.x)
        {
            const float4 s = shared4[i];
            shared4[i] = make_float4(gate * s.x, gate * s.y, gate * s.z, gate * s.w);
        }
    }
    else
    {
        for (int i = threadIdx.x; i < d_model; i += blockDim.x)
        {
            shared_output[i] *= gate;
        }
    }
}

__global__ void rocm_moe_shared_expert_gate_add_kernel(
    const float *__restrict__ input,            // [seq_len, d_model]
    const float *__restrict__ gate_inp,         // [d_model]
    float *__restrict__ shared_output,          // [seq_len, d_model]
    const float *__restrict__ routed_residual,  // [seq_len, d_model]
    float *__restrict__ combined_output,        // [seq_len, d_model]
    int seq_len, int d_model,
    const int *__restrict__ device_effective_seq_len = nullptr)
{
    const int token = blockIdx.x;
    if (token >= seq_len) return;

    const size_t row_offset = static_cast<size_t>(token) * d_model;
    const int effective_seq_len =
        rocm_moe_clamp_effective_seq_len(seq_len, device_effective_seq_len);
    if (token >= effective_seq_len)
    {
        if ((d_model & 3) == 0)
        {
            const int d4 = d_model >> 2;
            float4 *__restrict__ shared4 =
                reinterpret_cast<float4 *>(shared_output + row_offset);
            float4 *__restrict__ combined4 =
                reinterpret_cast<float4 *>(combined_output + row_offset);
            const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int i = threadIdx.x; i < d4; i += blockDim.x)
            {
                shared4[i] = zero;
                combined4[i] = zero;
            }
        }
        else
        {
            for (int i = threadIdx.x; i < d_model; i += blockDim.x)
            {
                shared_output[row_offset + i] = 0.0f;
                combined_output[row_offset + i] = 0.0f;
            }
        }
        return;
    }
    const float *x = input + row_offset;

    float sum = 0.0f;
    if ((d_model & 3) == 0)
    {
        const int d4 = d_model >> 2;
        const float4 *__restrict__ x4 = reinterpret_cast<const float4 *>(x);
        const float4 *__restrict__ gate4 = reinterpret_cast<const float4 *>(gate_inp);
        for (int i = threadIdx.x; i < d4; i += blockDim.x)
        {
            const float4 xv = x4[i];
            const float4 gv = gate4[i];
            sum += xv.x * gv.x + xv.y * gv.y + xv.z * gv.z + xv.w * gv.w;
        }
    }
    else
    {
        for (int i = threadIdx.x; i < d_model; i += blockDim.x)
            sum += x[i] * gate_inp[i];
    }

    sum = rocm_moe_wave_reduce_sum_f32(sum);

    __shared__ float warp_sums[16];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0)
    {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        float block_sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        block_sum = rocm_moe_wave_reduce_sum_f32(block_sum);
        if (lane == 0)
            warp_sums[0] = 1.0f / (1.0f + expf(-block_sum));
    }
    __syncthreads();

    const float gate = warp_sums[0];
    if ((d_model & 3) == 0)
    {
        const int d4 = d_model >> 2;
        float4 *__restrict__ shared4 =
            reinterpret_cast<float4 *>(shared_output + row_offset);
        const float4 *__restrict__ routed4 =
            reinterpret_cast<const float4 *>(routed_residual + row_offset);
        float4 *__restrict__ combined4 =
            reinterpret_cast<float4 *>(combined_output + row_offset);
        for (int i = threadIdx.x; i < d4; i += blockDim.x)
        {
            const float4 s = shared4[i];
            const float4 r = routed4[i];
            const float4 gated = make_float4(
                gate * s.x, gate * s.y, gate * s.z, gate * s.w);
            shared4[i] = gated;
            combined4[i] = make_float4(
                r.x + gated.x, r.y + gated.y, r.z + gated.z, r.w + gated.w);
        }
    }
    else
    {
        for (int i = threadIdx.x; i < d_model; i += blockDim.x)
        {
            const float gated_shared = gate * shared_output[row_offset + i];
            shared_output[row_offset + i] = gated_shared;
            combined_output[row_offset + i] = routed_residual[row_offset + i] + gated_shared;
        }
    }
}

// =========================================================================
// Kernel 6: SwiGLU — silu(gate) * up
//
// gate[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]
// Vectorized with float4.
// =========================================================================

__global__ void rocm_moe_swiglu_kernel(
    float *__restrict__ gate,
    const float *__restrict__ up,
    int count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    float g = gate[idx];
    float silu = g / (1.0f + expf(-g));
    gate[idx] = silu * up[idx];
}

// =========================================================================
// Kernel 6b: Weighted Add — output += weight * input
//
// Simple element-wise weighted accumulation for expert output aggregation.
// =========================================================================

__global__ void rocm_moe_weighted_add_kernel(
    float *__restrict__ output,
    const float *__restrict__ input,
    float weight,
    int count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    output[idx] += weight * input[idx];
}

/**
 * @brief Launch the small-M router for FP16/BF16 gate weights.
 *
 * Request-batched MTP verification produces M=2..4 router rows. Those rows need
 * the same dtype support as decode's single-token router, but the C bridge must
 * stay non-templated so host code can link it predictably.
 */
template <typename GateWeightT>
bool launch_gate_logits_small_m_mixed_weights(
    const float *hidden, const void *gate_weights, float *logits,
    int seq_len, int d_model, int num_experts,
    int device_idx, void *stream,
    const char *label)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!stream || !hidden || !gate_weights || !logits ||
        seq_len < 2 || seq_len > 4 || d_model <= 0 || num_experts <= 0)
    {
        fprintf(stderr, "HIP MoE %s invalid arguments\n", label);
        return false;
    }

    constexpr size_t kSmallMGroupedSharedCapBytes = 48 * 1024;
    const size_t shared_mem_bytes = static_cast<size_t>(d_model) * sizeof(float);
    if (shared_mem_bytes > kSmallMGroupedSharedCapBytes)
    {
        fprintf(stderr, "HIP MoE %s shared memory too large: requested=%zu cap=%zu\n",
                label, shared_mem_bytes, kSmallMGroupedSharedCapBytes);
        return false;
    }

    const int blocks =
        (num_experts + kGateLogitsExpertsPerBlock - 1) /
        kGateLogitsExpertsPerBlock;
    hipLaunchKernelGGL(rocm_moe_gate_logits_small_m_mixed_weights_kernel<GateWeightT>,
        dim3(blocks, seq_len), dim3(kGateLogitsGroupedThreads), shared_mem_bytes,
        static_cast<hipStream_t>(stream),
        hidden, static_cast<const GateWeightT *>(gate_weights), logits,
        seq_len, d_model, num_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE %s kernel launch failed: %s\n",
                label, hipGetErrorString(err));
        return false;
    }
    return true;
}

// =========================================================================
// Extern "C" bridge functions — called from ROCmMoEKernel.cpp
// =========================================================================

extern "C"
{

bool hipMoE_gate_logits_single_token(
    const float *hidden, const float *gate_weights, float *logits,
    int d_model, int num_experts,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 128;
    hipLaunchKernelGGL(rocm_moe_gate_logits_single_token_kernel,
        dim3(num_experts), dim3(threads), 0, static_cast<hipStream_t>(stream),
        hidden, gate_weights, logits, d_model, num_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE gate_logits_single_token kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_gate_logits_small_m(
    const float *hidden, const float *gate_weights, float *logits,
    int seq_len, int d_model, int num_experts,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!stream || !hidden || !gate_weights || !logits ||
        seq_len < 2 || seq_len > 4 || d_model <= 0 || num_experts <= 0)
    {
        fprintf(stderr, "HIP MoE gate_logits_small_m invalid arguments\n");
        return false;
    }

    constexpr size_t kSmallMGroupedSharedCapBytes = 48 * 1024;
    const size_t shared_mem_bytes = static_cast<size_t>(d_model) * sizeof(float);
    if (shared_mem_bytes > kSmallMGroupedSharedCapBytes)
    {
        fprintf(stderr, "HIP MoE gate_logits_small_m shared memory too large: requested=%zu cap=%zu\n",
                shared_mem_bytes, kSmallMGroupedSharedCapBytes);
        return false;
    }

    const bool use_vec4 = ((d_model & 3) == 0) &&
                          ((reinterpret_cast<uintptr_t>(hidden) & 0xfu) == 0) &&
                          ((reinterpret_cast<uintptr_t>(gate_weights) & 0xfu) == 0);
    const int blocks = (num_experts + kGateLogitsExpertsPerBlock - 1) / kGateLogitsExpertsPerBlock;
    hipLaunchKernelGGL(rocm_moe_gate_logits_small_m_kernel,
        dim3(blocks, seq_len), dim3(kGateLogitsGroupedThreads), shared_mem_bytes, static_cast<hipStream_t>(stream),
        hidden, gate_weights, logits, seq_len, d_model, num_experts, use_vec4);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE gate_logits_small_m kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_gate_logits_small_m_fp16_weights(
    const float *hidden, const void *gate_weights_fp16, float *logits,
    int seq_len, int d_model, int num_experts,
    int device_idx, void *stream)
{
    return launch_gate_logits_small_m_mixed_weights<__half>(
        hidden, gate_weights_fp16, logits, seq_len, d_model, num_experts,
        device_idx, stream, "gate_logits_small_m_fp16_weights");
}

bool hipMoE_gate_logits_small_m_bf16_weights(
    const float *hidden, const void *gate_weights_bf16, float *logits,
    int seq_len, int d_model, int num_experts,
    int device_idx, void *stream)
{
    return launch_gate_logits_small_m_mixed_weights<hip_bfloat16>(
        hidden, gate_weights_bf16, logits, seq_len, d_model, num_experts,
        device_idx, stream, "gate_logits_small_m_bf16_weights");
}

bool hipMoE_gate_logits_single_token_kpart(
    const float *hidden, const float *gate_weights, float *logits,
    float *partials,
    int d_model, int num_experts, int k_partitions,
    int device_idx, void *stream)
{
    if (!hidden || !gate_weights || !logits || !partials ||
        d_model <= 0 || num_experts <= 0 || k_partitions <= 0)
    {
        fprintf(stderr, "HIP MoE K-part gate logits invalid args\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const bool use_vec4 = ((d_model & 3) == 0) &&
                          ((reinterpret_cast<uintptr_t>(hidden) & 0xfu) == 0) &&
                          ((reinterpret_cast<uintptr_t>(gate_weights) & 0xfu) == 0);
    const int partial_threads = 128;
    hipLaunchKernelGGL(rocm_moe_gate_logits_single_token_kpart_kernel,
        dim3(num_experts, k_partitions), dim3(partial_threads), 0, static_cast<hipStream_t>(stream),
        hidden, gate_weights, partials, d_model, num_experts, k_partitions, use_vec4);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE gate_logits_single_token_kpart kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    const int reduce_threads = 64;
    hipLaunchKernelGGL(rocm_moe_gate_logits_kpart_reduce_kernel,
        dim3(num_experts), dim3(reduce_threads), 0, static_cast<hipStream_t>(stream),
        partials, logits, num_experts, k_partitions);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE gate_logits_kpart_reduce kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_gate_logits_single_token_kpart_partials(
    const float *hidden, const float *gate_weights, float *partials,
    int d_model, int num_experts, int k_partitions,
    int device_idx, void *stream)
{
    if (!hidden || !gate_weights || !partials ||
        d_model <= 0 || num_experts <= 0 || k_partitions <= 0)
    {
        fprintf(stderr, "HIP MoE K-part gate logits partials invalid args\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const bool use_vec4 = ((d_model & 3) == 0) &&
                          ((reinterpret_cast<uintptr_t>(hidden) & 0xfu) == 0) &&
                          ((reinterpret_cast<uintptr_t>(gate_weights) & 0xfu) == 0);
    const int partial_threads = 128;
    hipLaunchKernelGGL(rocm_moe_gate_logits_single_token_kpart_kernel,
        dim3(num_experts, k_partitions), dim3(partial_threads), 0, static_cast<hipStream_t>(stream),
        hidden, gate_weights, partials, d_model, num_experts, k_partitions, use_vec4);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE gate_logits_single_token_kpart_partials kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_fp32_to_fp16(
    const float *input, void *output_fp16, int count,
    int device_idx, void *stream)
{
    if (!input || !output_fp16 || count < 0)
    {
        fprintf(stderr, "HIP MoE fp32_to_fp16 invalid args\n");
        return false;
    }
    if (count == 0)
        return true;

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;
    hipLaunchKernelGGL(rocm_moe_fp32_to_fp16_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        input, static_cast<__half *>(output_fp16), count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE fp32_to_fp16 kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_gate_logits_single_token_fp16_weights(
    const float *hidden, const void *gate_weights_fp16, float *logits,
    int d_model, int num_experts,
    int device_idx, void *stream)
{
    if (!hidden || !gate_weights_fp16 || !logits || d_model <= 0 || num_experts <= 0)
    {
        fprintf(stderr, "HIP MoE fp16 gate logits invalid args\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 128;
    hipLaunchKernelGGL(rocm_moe_gate_logits_single_token_fp16_weights_kernel,
        dim3(num_experts), dim3(threads), 0, static_cast<hipStream_t>(stream),
        hidden, static_cast<const __half *>(gate_weights_fp16), logits, d_model, num_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE gate_logits_single_token_fp16_weights kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_gate_logits_single_token_bf16_weights(
    const float *hidden, const void *gate_weights_bf16, float *logits,
    int d_model, int num_experts,
    int device_idx, void *stream)
{
    if (!hidden || !gate_weights_bf16 || !logits || d_model <= 0 || num_experts <= 0)
    {
        fprintf(stderr, "HIP MoE bf16 gate logits invalid args\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 128;
    hipLaunchKernelGGL(rocm_moe_gate_logits_single_token_bf16_weights_kernel,
        dim3(num_experts), dim3(threads), 0, static_cast<hipStream_t>(stream),
        hidden, static_cast<const hip_bfloat16 *>(gate_weights_bf16), logits, d_model, num_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE gate_logits_single_token_bf16_weights kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_gate_logits_fp32_decode_equivalent_rows(
    const float *hidden, const float *gate_weights, float *logits,
    int seq_len, int d_model, int num_experts,
    int device_idx, void *stream)
{
    if (!hidden || !gate_weights || !logits || !stream ||
        seq_len <= 0 || seq_len > 4 || d_model <= 0 || num_experts <= 0)
    {
        fprintf(stderr, "HIP MoE FP32 grouped verifier gate logits invalid args\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 128;
    hipLaunchKernelGGL(rocm_moe_gate_logits_fp32_grouped_verifier_kernel,
        dim3(num_experts), dim3(threads), 0, static_cast<hipStream_t>(stream),
        hidden, gate_weights, logits, seq_len, d_model, num_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE FP32 grouped verifier gate logits kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_gate_logits_fp16_decode_equivalent_rows(
    const float *hidden, const void *gate_weights_fp16, float *logits,
    int seq_len, int d_model, int num_experts,
    int device_idx, void *stream)
{
    if (!hidden || !gate_weights_fp16 || !logits || !stream ||
        seq_len <= 0 || seq_len > 4 || d_model <= 0 || num_experts <= 0)
    {
        fprintf(stderr, "HIP MoE FP16 grouped verifier gate logits invalid args\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 128;
    hipLaunchKernelGGL(rocm_moe_gate_logits_fp16_grouped_verifier_kernel,
        dim3(num_experts), dim3(threads), 0, static_cast<hipStream_t>(stream),
        hidden, static_cast<const __half *>(gate_weights_fp16), logits,
        seq_len, d_model, num_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE FP16 grouped verifier gate logits kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_gate_logits_bf16_decode_equivalent_rows(
    const float *hidden, const void *gate_weights_bf16, float *logits,
    int seq_len, int d_model, int num_experts,
    int device_idx, void *stream)
{
    if (!hidden || !gate_weights_bf16 || !logits || !stream ||
        seq_len <= 0 || seq_len > 4 || d_model <= 0 || num_experts <= 0)
    {
        fprintf(stderr, "HIP MoE BF16 grouped verifier gate logits invalid args\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 128;
    hipLaunchKernelGGL(rocm_moe_gate_logits_bf16_grouped_verifier_kernel,
        dim3(num_experts), dim3(threads), 0, static_cast<hipStream_t>(stream),
        hidden, static_cast<const hip_bfloat16 *>(gate_weights_bf16), logits,
        seq_len, d_model, num_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE BF16 grouped verifier gate logits kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_quantize_router_gate_q8(
    const float *gate_weights, int8_t *gate_weights_q8, float *gate_scales,
    int d_model, int num_experts,
    int device_idx, void *stream)
{
    if (!gate_weights || !gate_weights_q8 || !gate_scales ||
        d_model <= 0 || num_experts <= 0 || (d_model % 32) != 0)
    {
        fprintf(stderr, "HIP MoE Q8 router gate quantize invalid args\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int blocks_per_row = d_model / 32;
    hipLaunchKernelGGL(rocm_moe_router_gate_quantize_q8_kernel,
        dim3(num_experts, blocks_per_row), dim3(32), 0, static_cast<hipStream_t>(stream),
        gate_weights, gate_weights_q8, gate_scales, d_model, num_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE Q8 router gate quantize kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_gate_logits_single_token_q8_weights(
    const float *hidden, int8_t *hidden_q8, float *hidden_scales,
    const int8_t *gate_weights_q8, const float *gate_scales, float *logits,
    int d_model, int num_experts,
    int device_idx, void *stream)
{
    if (!hidden || !hidden_q8 || !hidden_scales || !gate_weights_q8 || !gate_scales || !logits ||
        d_model <= 0 || num_experts <= 0 || (d_model % 32) != 0)
    {
        fprintf(stderr, "HIP MoE Q8 gate logits invalid args\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    (void)hipGetLastError();

    const int blocks_per_row = d_model / 32;
    hipLaunchKernelGGL(rocm_moe_router_hidden_quantize_q8_kernel,
        dim3(blocks_per_row), dim3(32), 0, hip_stream,
        hidden, hidden_q8, hidden_scales, d_model);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE Q8 hidden quantize kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    const int threads = 128;
    hipLaunchKernelGGL(rocm_moe_gate_logits_single_token_q8_weights_kernel,
        dim3(num_experts), dim3(threads), 0, hip_stream,
        hidden_q8, hidden_scales, gate_weights_q8, gate_scales, logits,
        d_model, num_experts);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE Q8 gate logits kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

/**
 * @brief Grouped Q8 router logits for MTP verifier rows.
 *
 * This is intentionally decode-equivalent rather than a generic prefill GEMM:
 * each hidden row is quantized independently with serial decode's block scales,
 * then one expert-owned block reuses gate Q8 data while accumulating all rows.
 */
bool hipMoE_gate_logits_q8_weights_decode_equivalent_rows(
    const float *hidden, int8_t *hidden_q8, float *hidden_scales,
    const int8_t *gate_weights_q8, const float *gate_scales, float *logits,
    int seq_len, int d_model, int num_experts,
    int device_idx, void *stream)
{
    if (!hidden || !hidden_q8 || !hidden_scales || !gate_weights_q8 || !gate_scales || !logits ||
        !stream || seq_len <= 0 || seq_len > 4 ||
        d_model <= 0 || num_experts <= 0 || (d_model % 32) != 0)
    {
        fprintf(stderr, "HIP MoE Q8 grouped verifier gate logits invalid args\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    (void)hipGetLastError();

    const int blocks_per_row = d_model / 32;
    hipLaunchKernelGGL(rocm_moe_router_hidden_quantize_q8_rows_kernel,
        dim3(blocks_per_row, seq_len), dim3(32), 0, hip_stream,
        hidden, hidden_q8, hidden_scales, seq_len, d_model);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE Q8 grouped hidden quantize kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    const int threads = 128;
    hipLaunchKernelGGL(rocm_moe_gate_logits_q8_grouped_verifier_kernel,
        dim3(num_experts), dim3(threads), 0, hip_stream,
        hidden_q8, hidden_scales, gate_weights_q8, gate_scales, logits,
        seq_len, d_model, num_experts);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE Q8 grouped gate logits kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_gate_logits_single_token_grouped(
    const float *hidden, const float *gate_weights, float *logits,
    int d_model, int num_experts, size_t shared_mem_bytes,
    int device_idx, void *stream)
{
    if (!hidden || !gate_weights || !logits || d_model <= 0 || num_experts <= 0)
    {
        fprintf(stderr, "HIP MoE grouped gate logits invalid args\n");
        return false;
    }

    const size_t required_shared = static_cast<size_t>(d_model) * sizeof(float);
    if (shared_mem_bytes < required_shared)
    {
        fprintf(stderr, "HIP MoE grouped gate logits shared memory too small: requested=%zu required=%zu\n",
                shared_mem_bytes, required_shared);
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const bool use_vec4 = ((d_model & 3) == 0) &&
                          ((reinterpret_cast<uintptr_t>(hidden) & 0xfu) == 0) &&
                          ((reinterpret_cast<uintptr_t>(gate_weights) & 0xfu) == 0);
    const int blocks = (num_experts + kGateLogitsExpertsPerBlock - 1) / kGateLogitsExpertsPerBlock;
    hipLaunchKernelGGL(rocm_moe_gate_logits_single_token_grouped_kernel,
        dim3(blocks), dim3(kGateLogitsGroupedThreads), shared_mem_bytes, static_cast<hipStream_t>(stream),
        hidden, gate_weights, logits, d_model, num_experts, use_vec4);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE grouped gate_logits_single_token kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_softmax_topk(
    float *logits,
    int *expert_indices, float *expert_weights,
    int seq_len, int num_experts, int top_k,
    bool normalize_weights,
    int device_idx, void *stream,
    const int *device_effective_seq_len)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!stream || !logits || !expert_indices || !expert_weights ||
        seq_len <= 0 || num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK || top_k > num_experts)
    {
        fprintf(stderr, "HIP MoE softmax_topk invalid arguments\n");
        return false;
    }

    // One block per token, 256 threads for parallel softmax
    const int threads = 256;
    hipLaunchKernelGGL(rocm_moe_softmax_topk_kernel,
        dim3(seq_len), dim3(threads), 0, static_cast<hipStream_t>(stream),
        logits, expert_indices, expert_weights,
        seq_len, num_experts, top_k, normalize_weights,
        device_effective_seq_len);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE softmax_topk kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_softmax_topk_decode_runtime(
    float *logits,
    void *runtime,
    float *legacy_indices,
    float *legacy_weights,
    int num_experts, int top_k,
    bool normalize_weights,
    bool write_legacy_outputs,
    bool update_runtime_histogram,
    void *runtime_layers,
    const void *rebalance_plan_entries,
    uint32_t rebalance_plan_capacity,
    void *rebalance_command_header,
    const void *rebalance_local_transfer_slots,
    uint32_t rebalance_local_transfer_slot_count,
    const void *rebalance_config,
    void *rebalance_apply_status,
    void *rebalance_controller_state,
    int rebalance_target_layer,
    uint32_t rebalance_command_buffer_count,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!logits || !runtime || num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK)
    {
        return false;
    }
    if (write_legacy_outputs && (!legacy_indices || !legacy_weights))
    {
        return false;
    }

    const int threads = 256;
    DeviceMoERebalanceConfigView rebalance_cfg{};
    if (rebalance_config)
        rebalance_cfg = *static_cast<const DeviceMoERebalanceConfigView *>(rebalance_config);
    hipLaunchKernelGGL(rocm_moe_softmax_topk_decode_runtime_kernel,
        dim3(1), dim3(threads), 0, static_cast<hipStream_t>(stream),
        logits,
        static_cast<DeviceMoELayerRuntimeView *>(runtime),
        static_cast<DeviceMoELayerRuntimeView *>(runtime_layers),
        legacy_indices, legacy_weights,
        num_experts, top_k,
        normalize_weights,
        write_legacy_outputs,
        update_runtime_histogram,
        static_cast<const DeviceMoERebalancePlanEntryView *>(rebalance_plan_entries),
        rebalance_plan_capacity,
        static_cast<DeviceMoERebalanceCommandBufferHeaderView *>(rebalance_command_header),
        static_cast<const DeviceMoEExpertDirectoryEntryView *>(rebalance_local_transfer_slots),
        rebalance_local_transfer_slot_count,
        rebalance_cfg,
        static_cast<DeviceMoERebalanceApplyStatusView *>(rebalance_apply_status),
        static_cast<DeviceMoERebalanceGraphControllerStateView *>(rebalance_controller_state),
        rebalance_target_layer,
        rebalance_command_buffer_count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE softmax_topk_decode_runtime kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_softmax_topk_decode_runtime_wave64(
    const float *logits,
    void *runtime,
    float *legacy_indices,
    float *legacy_weights,
    int num_experts, int top_k,
    bool normalize_weights,
    bool write_legacy_outputs,
    bool update_runtime_histogram,
    void *runtime_layers,
    const void *rebalance_plan_entries,
    uint32_t rebalance_plan_capacity,
    void *rebalance_command_header,
    const void *rebalance_local_transfer_slots,
    uint32_t rebalance_local_transfer_slot_count,
    const void *rebalance_config,
    void *rebalance_apply_status,
    void *rebalance_controller_state,
    int rebalance_target_layer,
    uint32_t rebalance_command_buffer_count,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!logits || !runtime || num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK)
    {
        return false;
    }
    if (write_legacy_outputs && (!legacy_indices || !legacy_weights))
    {
        return false;
    }

    const int threads = 256;
    DeviceMoERebalanceConfigView rebalance_cfg{};
    if (rebalance_config)
        rebalance_cfg = *static_cast<const DeviceMoERebalanceConfigView *>(rebalance_config);
    hipLaunchKernelGGL(rocm_moe_softmax_topk_decode_runtime_wave64_kernel,
        dim3(1), dim3(threads), 0, static_cast<hipStream_t>(stream),
        logits,
        static_cast<DeviceMoELayerRuntimeView *>(runtime),
        static_cast<DeviceMoELayerRuntimeView *>(runtime_layers),
        legacy_indices, legacy_weights,
        num_experts, top_k,
        normalize_weights,
        write_legacy_outputs,
        update_runtime_histogram,
        static_cast<const DeviceMoERebalancePlanEntryView *>(rebalance_plan_entries),
        rebalance_plan_capacity,
        static_cast<DeviceMoERebalanceCommandBufferHeaderView *>(rebalance_command_header),
        static_cast<const DeviceMoEExpertDirectoryEntryView *>(rebalance_local_transfer_slots),
        rebalance_local_transfer_slot_count,
        rebalance_cfg,
        static_cast<DeviceMoERebalanceApplyStatusView *>(rebalance_apply_status),
        static_cast<DeviceMoERebalanceGraphControllerStateView *>(rebalance_controller_state),
        rebalance_target_layer,
        rebalance_command_buffer_count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE softmax_topk_decode_runtime_wave64 kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_softmax_topk_decode_equivalent_rows(
    const float *logits,
    int *expert_indices, float *expert_weights,
    int seq_len, int num_experts, int top_k,
    bool normalize_weights,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!logits || !expert_indices || !expert_weights || !stream ||
        seq_len <= 0 || seq_len > 4 ||
        num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK || top_k > num_experts)
    {
        fprintf(stderr, "HIP MoE decode-equivalent rows softmax_topk invalid arguments\n");
        return false;
    }

    const int threads = 256;
    hipLaunchKernelGGL(rocm_moe_softmax_topk_decode_equivalent_rows_kernel,
        dim3(seq_len), dim3(threads), 0, static_cast<hipStream_t>(stream),
        logits, expert_indices, expert_weights,
        seq_len, num_experts, top_k, normalize_weights);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE decode-equivalent rows softmax_topk kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_router_kpart_reduce_softmax_topk_decode_runtime(
    const float *partials,
    void *runtime,
    float *legacy_indices,
    float *legacy_weights,
    int num_experts, int k_partitions, int top_k,
    bool normalize_weights,
    bool write_legacy_outputs,
    bool update_runtime_histogram,
    void *runtime_layers,
    const void *rebalance_plan_entries,
    uint32_t rebalance_plan_capacity,
    void *rebalance_command_header,
    const void *rebalance_local_transfer_slots,
    uint32_t rebalance_local_transfer_slot_count,
    const void *rebalance_config,
    void *rebalance_apply_status,
    void *rebalance_controller_state,
    int rebalance_target_layer,
    uint32_t rebalance_command_buffer_count,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!partials || !runtime || num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        k_partitions <= 0 || top_k <= 0 || top_k > kDeviceMoEMaxTopK)
    {
        return false;
    }
    if (write_legacy_outputs && (!legacy_indices || !legacy_weights))
    {
        return false;
    }

    const int threads = 256;
    DeviceMoERebalanceConfigView rebalance_cfg{};
    if (rebalance_config)
        rebalance_cfg = *static_cast<const DeviceMoERebalanceConfigView *>(rebalance_config);
    hipLaunchKernelGGL(rocm_moe_router_kpart_reduce_softmax_topk_decode_runtime_kernel,
        dim3(1), dim3(threads), 0, static_cast<hipStream_t>(stream),
        partials,
        static_cast<DeviceMoELayerRuntimeView *>(runtime),
        static_cast<DeviceMoELayerRuntimeView *>(runtime_layers),
        legacy_indices, legacy_weights,
        num_experts, k_partitions, top_k,
        normalize_weights,
        write_legacy_outputs,
        update_runtime_histogram,
        static_cast<const DeviceMoERebalancePlanEntryView *>(rebalance_plan_entries),
        rebalance_plan_capacity,
        static_cast<DeviceMoERebalanceCommandBufferHeaderView *>(rebalance_command_header),
        static_cast<const DeviceMoEExpertDirectoryEntryView *>(rebalance_local_transfer_slots),
        rebalance_local_transfer_slot_count,
        rebalance_cfg,
        static_cast<DeviceMoERebalanceApplyStatusView *>(rebalance_apply_status),
        static_cast<DeviceMoERebalanceGraphControllerStateView *>(rebalance_controller_state),
        rebalance_target_layer,
        rebalance_command_buffer_count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE router_kpart_reduce_softmax_topk_decode_runtime kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_decode_route_select_runtime(
    const int *expert_indices,
    const float *expert_weights,
    void *runtime,
    float *legacy_indices,
    float *legacy_weights,
    int num_experts, int top_k,
    bool write_legacy_outputs,
    bool update_runtime_histogram,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!expert_indices || !expert_weights || !runtime || top_k <= 0 || top_k > kDeviceMoEMaxTopK)
        return false;
    if (write_legacy_outputs && (!legacy_indices || !legacy_weights))
        return false;

    const int threads = 32;
    hipLaunchKernelGGL(rocm_moe_decode_route_select_runtime_kernel,
        dim3(1), dim3(threads), 0, static_cast<hipStream_t>(stream),
        expert_indices, expert_weights,
        static_cast<DeviceMoELayerRuntimeView *>(runtime),
        legacy_indices, legacy_weights,
        num_experts, top_k,
        write_legacy_outputs,
        update_runtime_histogram);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE decode_route_select_runtime kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_gather_tokens(
    const float *hidden, float *batch_buffer,
    const int *token_indices,
    int num_tokens, int d_model,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    hipLaunchKernelGGL(rocm_moe_gather_tokens_kernel,
        dim3(num_tokens), dim3(threads), 0, static_cast<hipStream_t>(stream),
        hidden, batch_buffer, token_indices, num_tokens, d_model);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE gather_tokens kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_copy_token_row(
    const float *source, float *row_buffer,
    int row_index, int row_width,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!source || !row_buffer || row_index < 0 || row_width <= 0 || !stream)
        return false;

    const int threads = 256;
    hipLaunchKernelGGL(rocm_moe_copy_row_kernel,
        dim3(1), dim3(threads), 0, static_cast<hipStream_t>(stream),
        source, row_buffer, row_index, row_width);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE copy_token_row kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_scatter_add(
    float *output, const float *expert_output,
    const int *token_indices, const float *weights,
    int num_tokens, int d_model,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    hipLaunchKernelGGL(rocm_moe_scatter_add_kernel,
        dim3(num_tokens), dim3(threads), 0, static_cast<hipStream_t>(stream),
        output, expert_output, token_indices, weights, num_tokens, d_model);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE scatter_add kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_write_token_row(
    float *destination, const float *row_buffer,
    int row_index, int row_width,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!destination || !row_buffer || row_index < 0 || row_width <= 0 || !stream)
        return false;

    const int threads = 256;
    hipLaunchKernelGGL(rocm_moe_write_row_kernel,
        dim3(1), dim3(threads), 0, static_cast<hipStream_t>(stream),
        destination, row_buffer, row_index, row_width);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE write_token_row kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_shared_expert_gate(
    const float *input, const float *gate_inp,
    float *shared_output, float *gate_scratch,
    int seq_len, int d_model,
    int device_idx, void *stream)
{
    // TIMING: skip device guard and error clear to test if they cause sync
    // llaminar2::HipDeviceGuard::setDevice(device_idx);
    // (void)hipGetLastError();

    // Phase 1: Compute dot products → sigmoid → gate_scratch[seq_len]
    {
        const int threads = 256;
        hipLaunchKernelGGL(rocm_moe_shared_expert_gate_dot_kernel,
            dim3(seq_len), dim3(threads), 0, static_cast<hipStream_t>(stream),
            input, gate_inp, gate_scratch, seq_len, d_model);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "HIP MoE shared_expert_gate dot kernel launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
    }

    // Phase 2: Scale shared_output by gate values
    {
        const int total = seq_len * d_model;
        const int threads = 256;
        const int blocks = (total + threads - 1) / threads;
        hipLaunchKernelGGL(rocm_moe_shared_expert_gate_scale_kernel,
            dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
            shared_output, gate_scratch, seq_len, d_model);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "HIP MoE shared_expert_gate scale kernel launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
    }

    return true;
}

bool hipMoE_shared_expert_gate_effective_seq_len(
    const float *input, const float *gate_inp,
    float *shared_output, float *gate_scratch,
    int seq_len, int d_model,
    const int *device_effective_seq_len,
    int device_idx, void *stream)
{
    if (!input || !gate_inp || !shared_output || !gate_scratch ||
        !device_effective_seq_len || seq_len <= 0 || d_model <= 0 || !stream)
    {
        fprintf(stderr, "HIP MoE shared_expert_gate_effective_seq_len invalid args\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    hipLaunchKernelGGL(rocm_moe_shared_expert_gate_dot_kernel,
        dim3(seq_len), dim3(threads), 0, static_cast<hipStream_t>(stream),
        input, gate_inp, gate_scratch, seq_len, d_model);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE shared_expert_gate_effective_seq_len dot kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    const int total = seq_len * d_model;
    const int blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(rocm_moe_shared_expert_gate_scale_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        shared_output, gate_scratch, seq_len, d_model, device_effective_seq_len);
    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE shared_expert_gate_effective_seq_len scale kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    return true;
}

bool hipMoE_shared_expert_gate_decode_fused(
    const float *input, const float *gate_inp,
    float *shared_output, int d_model,
    int device_idx, void *stream)
{
    if (!input || !gate_inp || !shared_output || d_model <= 0)
    {
        fprintf(stderr, "HIP MoE shared_expert_gate_decode_fused invalid args\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    hipLaunchKernelGGL(rocm_moe_shared_expert_gate_decode_fused_kernel,
        dim3(1), dim3(threads), 0, static_cast<hipStream_t>(stream),
        input, gate_inp, shared_output, d_model);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE shared_expert_gate_decode_fused kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    return true;
}

bool hipMoE_shared_expert_gate_add(
    const float *input, const float *gate_inp,
    float *shared_output, const float *routed_residual,
    float *combined_output, int seq_len, int d_model,
    int device_idx, void *stream)
{
    if (!input || !gate_inp || !shared_output || !routed_residual ||
        !combined_output || seq_len <= 0 || d_model <= 0)
    {
        fprintf(stderr, "HIP MoE shared_expert_gate_add invalid args\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    hipLaunchKernelGGL(rocm_moe_shared_expert_gate_add_kernel,
        dim3(seq_len), dim3(threads), 0, static_cast<hipStream_t>(stream),
        input, gate_inp, shared_output, routed_residual, combined_output,
        seq_len, d_model);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE shared_expert_gate_add kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    return true;
}

bool hipMoE_shared_expert_gate_add_effective_seq_len(
    const float *input, const float *gate_inp,
    float *shared_output, const float *routed_residual,
    float *combined_output, int seq_len, int d_model,
    const int *device_effective_seq_len,
    int device_idx, void *stream)
{
    if (!input || !gate_inp || !shared_output || !routed_residual ||
        !combined_output || !device_effective_seq_len ||
        seq_len <= 0 || d_model <= 0 || !stream)
    {
        fprintf(stderr, "HIP MoE shared_expert_gate_add_effective_seq_len invalid args\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    hipLaunchKernelGGL(rocm_moe_shared_expert_gate_add_kernel,
        dim3(seq_len), dim3(threads), 0, static_cast<hipStream_t>(stream),
        input, gate_inp, shared_output, routed_residual, combined_output,
        seq_len, d_model, device_effective_seq_len);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE shared_expert_gate_add_effective_seq_len kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    return true;
}

bool hipMoE_swiglu(
    float *gate, const float *up,
    int count,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;
    if (blocks == 0 && count > 0) return true; // No-op for empty

    hipLaunchKernelGGL(rocm_moe_swiglu_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        gate, up, count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE swiglu kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_weighted_add(
    float *output, const float *input,
    float weight, int count,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (count <= 0) return true;

    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;

    hipLaunchKernelGGL(rocm_moe_weighted_add_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        output, input, weight, count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE weighted_add kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

} // extern "C"

// =========================================================================
// Phase 2 Kernels: Histogram + Expert Mask
// =========================================================================

// =========================================================================
// Kernel 7: Histogram Record
//
// Atomically increments per-expert counters based on routing indices.
// One block per token; threads cover the top_k dimension.
// =========================================================================

__global__ void rocm_moe_histogram_record_kernel(
    const int *__restrict__ routing_indices, // [seq_len * top_k]
    unsigned long long *__restrict__ histogram, // [max_layers * num_experts]
    int seq_len, int num_experts, int top_k,
    int layer_offset) // = layer_idx * num_experts
{
    const int token = blockIdx.x;
    if (token >= seq_len) return;

    for (int k = threadIdx.x; k < top_k; k += blockDim.x)
    {
        int expert_idx = routing_indices[token * top_k + k];
        if (expert_idx >= 0 && expert_idx < num_experts)
        {
            atomicAdd(&histogram[layer_offset + expert_idx], 1ULL);
        }
    }
}

// =========================================================================
// Kernel 8: Apply Expert Mask
//
// Zero out routing weights for masked-off experts.
// One block per token; threads cover the top_k dimension.
// =========================================================================

__global__ void rocm_moe_apply_expert_mask_kernel(
    float *__restrict__ routing_weights,       // [seq_len * top_k]
    const int *__restrict__ routing_indices,   // [seq_len * top_k]
    const bool *__restrict__ expert_mask,      // [num_experts]
    int seq_len, int top_k)
{
    const int token = blockIdx.x;
    if (token >= seq_len) return;

    for (int k = threadIdx.x; k < top_k; k += blockDim.x)
    {
        int idx = token * top_k + k;
        int expert_idx = routing_indices[idx];
        if (!expert_mask[expert_idx])
        {
            routing_weights[idx] = 0.0f;
        }
    }
}

// =========================================================================
// Kernel 9: Histogram Reset
//
// Zero a layer's slice of the histogram buffer.
// =========================================================================

__global__ void rocm_moe_histogram_reset_kernel(
    unsigned long long *__restrict__ histogram, // [max_layers * num_experts]
    int layer_offset, int num_experts)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_experts)
    {
        histogram[layer_offset + idx] = 0ULL;
    }
}

// =========================================================================
// Phase 2 extern "C" bridge functions
// =========================================================================

extern "C"
{

bool hipMoE_histogram_record(
    const int *routing_indices, unsigned long long *histogram,
    int seq_len, int num_experts, int top_k, int layer_idx,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = (top_k <= 256) ? ((top_k + 31) / 32 * 32) : 256;
    const int actual_threads = (threads < 32) ? 32 : threads;
    const int layer_offset = layer_idx * num_experts;

    hipLaunchKernelGGL(rocm_moe_histogram_record_kernel,
        dim3(seq_len), dim3(actual_threads), 0, static_cast<hipStream_t>(stream),
        routing_indices, histogram,
        seq_len, num_experts, top_k, layer_offset);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE histogram_record kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_apply_expert_mask(
    float *routing_weights, const int *routing_indices,
    const bool *expert_mask,
    int seq_len, int top_k,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = (top_k <= 256) ? ((top_k + 31) / 32 * 32) : 256;
    const int actual_threads = (threads < 32) ? 32 : threads;

    hipLaunchKernelGGL(rocm_moe_apply_expert_mask_kernel,
        dim3(seq_len), dim3(actual_threads), 0, static_cast<hipStream_t>(stream),
        routing_weights, routing_indices, expert_mask,
        seq_len, top_k);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE apply_expert_mask kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_histogram_reset(
    unsigned long long *histogram, int layer_idx, int num_experts,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int layer_offset = layer_idx * num_experts;
    const int threads = 256;
    const int blocks = (num_experts + threads - 1) / threads;

    hipLaunchKernelGGL(rocm_moe_histogram_reset_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        histogram, layer_offset, num_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE histogram_reset kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_device_rebalance_controller(
    void *runtime_layers,
    const unsigned long long *gathered_histograms,
    void *status,
    const void *config,
    void *plan_entries,
    uint32_t *plan_count,
    uint32_t plan_capacity,
    uint32_t payload_slot_capacity,
        void *command_header,
        void *wave_state,
        void *controller_state,
        uint32_t command_buffer_count,
        int device_idx,
        void *stream)
{
    if (!runtime_layers || !gathered_histograms || !status || !config || !stream)
        return false;

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
    const bool dynamic_ownership_fast_path =
        cfg.routed_assignment_policy != kDeviceMoERebalanceAssignmentLeastLoadedEP &&
        (cfg.flags & kDeviceMoERebalanceFlagHotReplicaCache) == 0u &&
        (cfg.flags & kDeviceMoERebalanceFlagPlanMissingArrivals) != 0u &&
        (cfg.flags & kDeviceMoERebalanceFlagDeferRuntimeApply) != 0u;
    if (dynamic_ownership_fast_path)
    {
        hipLaunchKernelGGL(device_rebalance_dynamic_ownership_controller_kernel,
            dim3(1), dim3(kDeviceMoEMaxExperts), 0, static_cast<hipStream_t>(stream),
            static_cast<DeviceMoELayerRuntimeView *>(runtime_layers),
            gathered_histograms,
            static_cast<DeviceMoERebalanceStatusView *>(status),
            cfg,
            static_cast<DeviceMoERebalancePlanEntryView *>(plan_entries),
            plan_count,
            plan_capacity,
            payload_slot_capacity,
            static_cast<DeviceMoERebalanceCommandBufferHeaderView *>(command_header),
            static_cast<DeviceMoERebalanceWaveStateView *>(wave_state),
            static_cast<DeviceMoERebalanceGraphControllerStateView *>(controller_state),
            command_buffer_count);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "HIP MoE device_rebalance_dynamic_ownership_controller kernel launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        return true;
    }
    hipLaunchKernelGGL(device_rebalance_controller_kernel,
        dim3(1), dim3(kDeviceMoEMaxExperts), 0, static_cast<hipStream_t>(stream),
        static_cast<DeviceMoELayerRuntimeView *>(runtime_layers),
        gathered_histograms,
        static_cast<DeviceMoERebalanceStatusView *>(status),
        cfg,
        static_cast<DeviceMoERebalancePlanEntryView *>(plan_entries),
        plan_count,
            plan_capacity,
            payload_slot_capacity,
            static_cast<DeviceMoERebalanceCommandBufferHeaderView *>(command_header),
            static_cast<DeviceMoERebalanceWaveStateView *>(wave_state),
            static_cast<DeviceMoERebalanceGraphControllerStateView *>(controller_state),
            command_buffer_count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE device_rebalance_controller kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_pack_rebalance_histograms(
    void *runtime_layers,
    unsigned long long *local_histograms,
    const void *config,
    const void *wave_state,
    const void *controller_state,
    uint32_t command_buffer_count,
    int device_idx,
    void *stream)
{
    if (!runtime_layers || !local_histograms || !config || !stream)
        return false;

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
    const uint32_t window_count =
        cfg.layer_window_count == 0u
            ? cfg.num_layers
            : ((cfg.layer_window_count < cfg.num_layers)
                   ? cfg.layer_window_count
                   : cfg.num_layers);
    const uint32_t wave_count =
        cfg.layer_wave_count == 0u
            ? window_count
            : ((cfg.layer_wave_count < window_count)
                   ? cfg.layer_wave_count
                   : window_count);
    const int total = static_cast<int>(wave_count * cfg.num_experts);
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    hipLaunchKernelGGL(pack_rebalance_histograms_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        static_cast<const DeviceMoELayerRuntimeView *>(runtime_layers),
        local_histograms,
        cfg,
        static_cast<const DeviceMoERebalanceWaveStateView *>(wave_state),
        static_cast<const DeviceMoERebalanceGraphControllerStateView *>(controller_state),
        command_buffer_count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE pack_rebalance_histograms kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_pack_rebalance_directory(
    void *runtime_layers,
    void *local_directory,
    const void *config,
    int device_idx,
    void *stream)
{
    if (!runtime_layers || !local_directory || !config || !stream)
        return false;

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
    const int total = static_cast<int>(cfg.num_layers * cfg.num_experts);
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    hipLaunchKernelGGL(pack_rebalance_directory_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        static_cast<const DeviceMoELayerRuntimeView *>(runtime_layers),
        static_cast<DeviceMoEExpertDirectoryEntryView *>(local_directory),
        cfg);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE pack_rebalance_directory kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_pack_rebalance_source_descriptors(
    void *runtime_layers,
    const void *gathered_plan_entries,
    const void *gathered_command_headers,
    uint32_t plan_capacity,
    void *local_source_descriptors,
    const void *config,
    void *controller_state,
    uint32_t command_buffer_count,
    int device_idx,
    void *stream)
{
    if (!runtime_layers || !gathered_plan_entries || !gathered_command_headers ||
        !local_source_descriptors || !config || !stream)
        return false;

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
    const uint32_t metadata_buffer_count =
        command_buffer_count <= 1u ? 1u : (command_buffer_count < 2u ? command_buffer_count : 2u);
    const unsigned long long total =
        static_cast<unsigned long long>(cfg.participant_count) *
        static_cast<unsigned long long>(metadata_buffer_count) *
        static_cast<unsigned long long>(plan_capacity);
    const int threads = 256;
    const int blocks =
        static_cast<int>((total + static_cast<unsigned long long>(threads) - 1ULL) /
                         static_cast<unsigned long long>(threads));
    if (blocks > 0)
    {
        hipLaunchKernelGGL(pack_rebalance_source_descriptors_kernel,
            dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
            static_cast<const DeviceMoELayerRuntimeView *>(runtime_layers),
            static_cast<const DeviceMoERebalancePlanEntryView *>(gathered_plan_entries),
            static_cast<const DeviceMoERebalanceCommandBufferHeaderView *>(gathered_command_headers),
            plan_capacity,
            static_cast<DeviceMoEExpertDirectoryEntryView *>(local_source_descriptors),
            cfg,
            static_cast<DeviceMoERebalanceGraphControllerStateView *>(controller_state),
            command_buffer_count);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE pack_rebalance_source_descriptors kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_project_rebalance_domain_commands(
    const void *gathered_plan_entries,
    const void *gathered_command_headers,
    uint32_t plan_capacity,
    void *local_plan_entries,
    void *local_command_headers,
    const void *config,
    void *status,
    uint32_t payload_slot_capacity,
    uint32_t command_buffer_count,
    const void *gathered_wave_states,
    void *local_wave_states,
    int device_idx,
    void *stream)
{
    if (!gathered_plan_entries || !gathered_command_headers ||
        !local_plan_entries || !local_command_headers ||
        !config || !stream || plan_capacity == 0u)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
    hipLaunchKernelGGL(project_rebalance_domain_commands_kernel,
        dim3(1), dim3(256), 0, static_cast<hipStream_t>(stream),
        static_cast<const DeviceMoERebalancePlanEntryView *>(gathered_plan_entries),
        static_cast<const DeviceMoERebalanceCommandBufferHeaderView *>(gathered_command_headers),
        plan_capacity,
        static_cast<DeviceMoERebalancePlanEntryView *>(local_plan_entries),
        static_cast<DeviceMoERebalanceCommandBufferHeaderView *>(local_command_headers),
        static_cast<DeviceMoERebalanceStatusView *>(status),
        payload_slot_capacity,
        cfg,
        command_buffer_count,
        static_cast<const DeviceMoERebalanceWaveStateView *>(gathered_wave_states),
        static_cast<DeviceMoERebalanceWaveStateView *>(local_wave_states));

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE project_rebalance_domain_commands kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_project_prefill_llep_domain_commands(
    const void *gathered_plan_entries,
    const void *gathered_command_headers,
    uint32_t plan_capacity,
    void *local_plan_entries,
    uint32_t *local_plan_count,
    void *local_command_header,
    const void *config,
    void *status,
    uint32_t payload_slot_capacity,
    uint32_t command_buffer_count,
    int device_idx,
    void *stream)
{
    if (!gathered_plan_entries || !gathered_command_headers ||
        !local_plan_entries || !local_plan_count || !local_command_header ||
        !config || !status || !stream || plan_capacity == 0u ||
        payload_slot_capacity == 0u)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
    hipLaunchKernelGGL(project_prefill_llep_domain_commands_kernel,
        dim3(1), dim3(256), 0, static_cast<hipStream_t>(stream),
        static_cast<const DeviceMoERebalancePlanEntryView *>(gathered_plan_entries),
        static_cast<const DeviceMoERebalanceCommandBufferHeaderView *>(gathered_command_headers),
        plan_capacity,
        static_cast<DeviceMoERebalancePlanEntryView *>(local_plan_entries),
        local_plan_count,
        static_cast<DeviceMoERebalanceCommandBufferHeaderView *>(local_command_header),
        static_cast<DeviceMoERebalanceStatusView *>(status),
        payload_slot_capacity,
        cfg,
        command_buffer_count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE project_prefill_llep_domain_commands kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_materialize_prefill_llep_transfer_commands(
    const void *runtime_layer,
    void *plan_entries,
    uint32_t *plan_count,
    uint32_t plan_capacity,
    void *command_header,
    void *status,
    const void *config,
    uint32_t payload_slot_capacity,
    uint32_t layer_idx,
    uint32_t command_buffer_count,
    int device_idx,
    void *stream)
{
    if (!runtime_layer || !plan_entries || !plan_count || !command_header ||
        !status || !config || !stream || plan_capacity == 0u ||
        payload_slot_capacity == 0u)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
    hipLaunchKernelGGL(materialize_prefill_llep_transfer_commands_kernel,
        dim3(1), dim3(256), 0, static_cast<hipStream_t>(stream),
        static_cast<const DeviceMoELayerRuntimeView *>(runtime_layer),
        static_cast<DeviceMoERebalancePlanEntryView *>(plan_entries),
        plan_count,
        plan_capacity,
        static_cast<DeviceMoERebalanceCommandBufferHeaderView *>(command_header),
        static_cast<DeviceMoERebalanceStatusView *>(status),
        cfg,
        payload_slot_capacity,
        layer_idx,
        command_buffer_count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE materialize_prefill_llep_transfer_commands kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_pack_rebalance_compact_payloads(
    const void *plan_entries,
    const void *command_headers,
    uint32_t plan_capacity,
    const void *local_source_descriptors,
    void *local_payload,
    uint32_t local_payload_slot_count,
    unsigned long long payload_slot_bytes,
    const void *config,
    void *status,
    void *controller_state,
    uint32_t command_buffer_count,
    int device_idx,
    void *stream)
{
    if (!plan_entries || !command_headers ||
        !local_source_descriptors || !local_payload || !config || !status || !stream ||
        plan_capacity == 0u || local_payload_slot_count == 0u ||
        payload_slot_bytes <= sizeof(DeviceMoEExpertDirectoryEntryView))
        return false;

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
    // Pack starts the per-wave copy-status record.  The unpack phase later
    // accumulates into this same record so source-side pack failures cannot
    // be erased before transfer-complete publication inspects the gathered
    // statuses.
    hipLaunchKernelGGL(init_rebalance_apply_status_kernel,
        dim3(1), dim3(1), 0, static_cast<hipStream_t>(stream),
        static_cast<DeviceMoERebalanceApplyStatusView *>(status));

    const int threads = 256;
    hipLaunchKernelGGL(pack_rebalance_compact_payloads_kernel,
        dim3(static_cast<int>(local_payload_slot_count)), dim3(threads),
        0, static_cast<hipStream_t>(stream),
        static_cast<const DeviceMoERebalancePlanEntryView *>(plan_entries),
        static_cast<const DeviceMoERebalanceCommandBufferHeaderView *>(command_headers),
        plan_capacity,
        static_cast<const DeviceMoEExpertDirectoryEntryView *>(local_source_descriptors),
        static_cast<uint8_t *>(local_payload),
        local_payload_slot_count,
        payload_slot_bytes,
        cfg,
        static_cast<DeviceMoERebalanceApplyStatusView *>(status),
        static_cast<DeviceMoERebalanceGraphControllerStateView *>(controller_state),
        command_buffer_count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE pack_rebalance_compact_payloads kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_pack_rebalance_collective_payloads(
    const void *gathered_plan_entries,
    const void *gathered_command_headers,
    uint32_t plan_capacity,
    const void *local_directory,
    void *local_payload,
    uint32_t local_payload_slot_count,
    unsigned long long payload_slot_bytes,
        const void *config,
        void *status,
        void *controller_state,
        uint32_t command_buffer_count,
        int device_idx,
        void *stream)
{
    if (!gathered_plan_entries || !gathered_command_headers ||
        !local_directory || !local_payload || !config || !status ||
        !stream || plan_capacity == 0u || local_payload_slot_count == 0u ||
        payload_slot_bytes <= sizeof(DeviceMoEExpertDirectoryEntryView))
        return false;

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
    hipLaunchKernelGGL(init_rebalance_apply_status_kernel,
        dim3(1), dim3(1), 0, static_cast<hipStream_t>(stream),
        static_cast<DeviceMoERebalanceApplyStatusView *>(status));
    hipLaunchKernelGGL(pack_rebalance_collective_payloads_kernel,
        dim3(local_payload_slot_count), dim3(256), 0, static_cast<hipStream_t>(stream),
        static_cast<const DeviceMoERebalancePlanEntryView *>(gathered_plan_entries),
        static_cast<const DeviceMoERebalanceCommandBufferHeaderView *>(gathered_command_headers),
        plan_capacity,
        static_cast<const DeviceMoEExpertDirectoryEntryView *>(local_directory),
        static_cast<uint8_t *>(local_payload),
        local_payload_slot_count,
            payload_slot_bytes,
            cfg,
            static_cast<DeviceMoERebalanceApplyStatusView *>(status),
            static_cast<DeviceMoERebalanceGraphControllerStateView *>(controller_state),
            command_buffer_count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE pack_rebalance_collective_payloads kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_unpack_rebalance_collective_payloads(
    const void *plan_entries,
    const uint32_t *plan_count,
    uint32_t plan_capacity,
    const void *command_header,
    const void *gathered_payload,
    uint32_t local_payload_slot_count,
    unsigned long long payload_slot_bytes,
    void *local_transfer_slots,
    uint32_t local_transfer_slot_count,
        const void *config,
        void *status,
        void *controller_state,
        uint32_t command_buffer_count,
        int device_idx,
        void *stream)
{
    if (!plan_entries || (!plan_count && !command_header) ||
        !gathered_payload || !local_transfer_slots || !config || !status ||
        !stream || plan_capacity == 0u || local_payload_slot_count == 0u ||
        payload_slot_bytes <= sizeof(DeviceMoEExpertDirectoryEntryView))
        return false;

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
    if (local_transfer_slot_count == 0xffffffffu)
        return false;
    const uint32_t unpack_block_count = local_transfer_slot_count + 1u;
    // Deliberately preserve the status initialized by the matching pack call.
    // Pack and unpack are two halves of one transfer wave and both must
    // contribute counters to the same status record.
    hipLaunchKernelGGL(unpack_rebalance_collective_payloads_kernel,
        dim3(unpack_block_count), dim3(256), 0, static_cast<hipStream_t>(stream),
        static_cast<const DeviceMoERebalancePlanEntryView *>(plan_entries),
        plan_count,
        plan_capacity,
        const_cast<DeviceMoERebalanceCommandBufferHeaderView *>(
            static_cast<const DeviceMoERebalanceCommandBufferHeaderView *>(command_header)),
        static_cast<const uint8_t *>(gathered_payload),
        local_payload_slot_count,
        payload_slot_bytes,
        static_cast<DeviceMoEExpertDirectoryEntryView *>(local_transfer_slots),
            local_transfer_slot_count,
            cfg,
            static_cast<DeviceMoERebalanceApplyStatusView *>(status),
            static_cast<DeviceMoERebalanceGraphControllerStateView *>(controller_state),
            command_buffer_count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE unpack_rebalance_collective_payloads kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_apply_rebalance_arrivals(
    void *runtime_layers,
    const void *plan_entries,
    const uint32_t *plan_count,
    uint32_t plan_capacity,
    const void *command_header,
    const void *local_transfer_slots,
    uint32_t local_transfer_slot_count,
    const void *config,
    void *status,
    int target_layer,
    int device_idx,
    void *stream)
{
    if (!runtime_layers || !plan_entries || (!plan_count && !command_header) ||
        !local_transfer_slots || !config || !status || !stream)
        return false;

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
    hipLaunchKernelGGL(init_rebalance_apply_status_kernel,
        dim3(1), dim3(1), 0, static_cast<hipStream_t>(stream),
        static_cast<DeviceMoERebalanceApplyStatusView *>(status));
    hipLaunchKernelGGL(apply_rebalance_arrivals_kernel,
        dim3(1), dim3(kDeviceMoERebalanceApplyThreads), 0, static_cast<hipStream_t>(stream),
        static_cast<DeviceMoELayerRuntimeView *>(runtime_layers),
        static_cast<const DeviceMoERebalancePlanEntryView *>(plan_entries),
        plan_count,
        plan_capacity,
        const_cast<DeviceMoERebalanceCommandBufferHeaderView *>(
            static_cast<const DeviceMoERebalanceCommandBufferHeaderView *>(command_header)),
        static_cast<const DeviceMoEExpertDirectoryEntryView *>(local_transfer_slots),
        local_transfer_slot_count,
        cfg,
            static_cast<DeviceMoERebalanceApplyStatusView *>(status),
            target_layer,
            nullptr,
            false,
            1u);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE apply_rebalance_arrivals kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_init_rebalance_graph_controller_state(
    void *controller_state,
    const void *config,
    int device_idx,
    void *stream)
{
    if (!controller_state || !config || !stream)
        return false;

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
    hipLaunchKernelGGL(init_rebalance_graph_controller_state_kernel,
        dim3(1), dim3(1), 0, static_cast<hipStream_t>(stream),
        static_cast<DeviceMoERebalanceGraphControllerStateView *>(controller_state),
        cfg);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE init_rebalance_graph_controller_state kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_publish_rebalance_transfer_complete(
    void *controller_state,
    const void *command_header,
    const void *wave_state,
    const void *copy_status,
    const void *plan_entries,
    uint32_t plan_capacity,
    const void *gathered_copy_status,
    const void *config,
    uint32_t command_buffer_count,
    int device_idx,
    void *stream)
{
    if (!controller_state || !command_header || !wave_state ||
        !copy_status || !plan_entries || plan_capacity == 0u ||
        !gathered_copy_status || !config || !stream)
        return false;

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
    hipLaunchKernelGGL(publish_rebalance_transfer_complete_kernel,
        dim3(1), dim3(1), 0, static_cast<hipStream_t>(stream),
            static_cast<DeviceMoERebalanceGraphControllerStateView *>(controller_state),
            static_cast<const DeviceMoERebalanceCommandBufferHeaderView *>(command_header),
            static_cast<const DeviceMoERebalanceWaveStateView *>(wave_state),
            static_cast<const DeviceMoERebalanceApplyStatusView *>(copy_status),
            static_cast<const DeviceMoERebalancePlanEntryView *>(plan_entries),
            plan_capacity,
            static_cast<const DeviceMoERebalanceApplyStatusView *>(gathered_copy_status),
            cfg,
            command_buffer_count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE publish_rebalance_transfer_complete kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_apply_ready_rebalance_wave(
    void *runtime_layers,
    const void *plan_entries,
    const uint32_t *plan_count,
    uint32_t plan_capacity,
    void *command_header,
    const void *local_transfer_slots,
    uint32_t local_transfer_slot_count,
    const void *config,
        void *status,
        void *controller_state,
        int target_layer,
        uint32_t command_buffer_count,
        int device_idx,
        void *stream)
{
    if (!runtime_layers || !plan_entries || (!plan_count && !command_header) ||
        !config || !status || !controller_state || !stream)
        return false;

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
    hipLaunchKernelGGL(apply_rebalance_arrivals_kernel,
        dim3(1), dim3(kDeviceMoERebalanceApplyThreads), 0, static_cast<hipStream_t>(stream),
        static_cast<DeviceMoELayerRuntimeView *>(runtime_layers),
        static_cast<const DeviceMoERebalancePlanEntryView *>(plan_entries),
        plan_count,
        plan_capacity,
        static_cast<DeviceMoERebalanceCommandBufferHeaderView *>(command_header),
        static_cast<const DeviceMoEExpertDirectoryEntryView *>(local_transfer_slots),
        local_transfer_slot_count,
        cfg,
        static_cast<DeviceMoERebalanceApplyStatusView *>(status),
            target_layer,
            static_cast<DeviceMoERebalanceGraphControllerStateView *>(controller_state),
            true,
            command_buffer_count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE apply_ready_rebalance_wave kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

} // extern "C" (Phase 2)

// =========================================================================
// Phase 3 Kernels: Device-Side Token Grouping
// =========================================================================

// =========================================================================
// Kernel 10: Count Per Expert
//
// Counts how many token-slots go to each expert via atomicAdd.
// Grid: ceil(total_slots / 256), Block: 256
// =========================================================================

__global__ void rocm_moe_count_per_expert_kernel(
    const int *__restrict__ routing_indices, // [total_slots]
    int *__restrict__ expert_counts,         // [num_experts]
    int total_slots, int num_experts)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_slots) return;

    int expert_id = routing_indices[idx];
    if (expert_id >= 0 && expert_id < num_experts)
    {
        atomicAdd(&expert_counts[expert_id], 1);
    }
}

// =========================================================================
// Kernel 11: Exclusive Scan
//
// Computes exclusive prefix sum on expert_counts → expert_offsets.
// Single block, single thread (num_experts ≤ 256; serial cost negligible).
// Also preserves expert_counts (the original counts are needed by caller).
// =========================================================================

__global__ void rocm_moe_exclusive_scan_kernel(
    int *__restrict__ expert_counts,   // [num_experts] IN (read, then overwritten with original counts)
    int *__restrict__ expert_offsets,   // [num_experts] OUT (exclusive prefix sum)
    int num_experts)
{
    if (threadIdx.x != 0) return;

    int running = 0;
    for (int e = 0; e < num_experts; ++e)
    {
        int count = expert_counts[e];
        expert_offsets[e] = running;
        running += count;
        // expert_counts[e] already contains the correct count — no modification needed
    }
}

// =========================================================================
// Kernel 11b: Max Expert Count (device-side reduction)
//
// Computes max(expert_counts[0..num_experts-1]) entirely on device.
// Single wavefront (64 threads), warp-shuffle reduction.
// Grid: 1 block, Block: 64 threads
// =========================================================================

__global__ void rocm_moe_max_expert_count_kernel(
    const int *__restrict__ expert_counts,  // [num_experts]
    int *__restrict__ d_max_out,            // [1] output scalar
    int num_experts)
{
    const int tid = threadIdx.x;
    int local_max = 0;

    // Each thread processes multiple experts (stride 64)
    for (int e = tid; e < num_experts; e += 64)
        local_max = max(local_max, expert_counts[e]);

    // Warp-shuffle reduction within a single wavefront (64 threads)
    for (int offset = 32; offset > 0; offset >>= 1)
        local_max = max(local_max, __shfl_down(local_max, offset));

    if (tid == 0)
        *d_max_out = local_max;
}

__global__ void rocm_moe_build_active_expert_list_kernel(
    const int *__restrict__ expert_counts,
    int *__restrict__ active_expert_ids,
    int num_experts,
    int max_active_experts)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    int active = 0;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        const int count = expert_counts[expert];
        if (count > 0 && active < max_active_experts)
            active_expert_ids[active++] = expert;
    }
    for (int slot = active; slot < max_active_experts; ++slot)
        active_expert_ids[slot] = -1;
}

__global__ void rocm_moe_build_active_expert_list_runtime_kernel(
    const DeviceMoELayerRuntimeView *__restrict__ runtime,
    int *__restrict__ active_expert_ids,
    int num_experts,
    int max_active_experts)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    if (!runtime || !active_expert_ids ||
        num_experts <= 0 || max_active_experts <= 0 ||
        runtime->active_bank > 1u ||
        !runtime->expert_counts)
    {
        fail_fast_incomplete_llep_transfer();
        return;
    }

    const DeviceMoEPlacementBankView &bank = runtime->banks[runtime->active_bank];
    const uint32_t local_bit =
        runtime_participant_bit(static_cast<int>(runtime->participant_id));
    int active = 0;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        const int count = runtime->expert_counts[expert];
        if (count <= 0)
            continue;

        const DeviceMoEExpertDescriptorView &desc = bank.experts[expert];
        const bool local_ready =
            bank.local_compute_mask[expert] != 0u &&
            (bank.resident_participant_mask[expert] & local_bit) != 0u &&
            desc.local_slot >= 0 &&
            rebalance_expert_desc_ready(desc);
        if (!local_ready)
        {
            printf("runtime_prefill_active_expert_not_ready "
                   "participant=%u expert=%d count=%d active_bank=%u "
                   "local_mask=%u resident_mask=%u local_bit=%u "
                   "local_slot=%d logical=%d owner=%d\\n",
                   runtime->participant_id,
                   expert,
                   count,
                   runtime->active_bank,
                   bank.local_compute_mask[expert],
                   bank.resident_participant_mask[expert],
                   local_bit,
                   desc.local_slot,
                   desc.logical_expert_id,
                   desc.owner_participant);
            fail_fast_incomplete_llep_transfer();
            return;
        }
        if (active < max_active_experts)
            active_expert_ids[active++] = expert;
    }
    for (int slot = active; slot < max_active_experts; ++slot)
        active_expert_ids[slot] = -1;
}

/**
 * @brief Build the route-slot to grouped-slot map for runtime grouped prefill.
 *
 * LLEP runtime grouping stores original route slots in @c grouped_token_ids
 * after it filters the batch down to this participant's assigned routes.  The
 * ordered scatter stage needs the inverse map so every token accumulates its
 * top-k choices in router order instead of in grouped expert order.  Invalid
 * scratch state traps immediately because continuing would publish numerically
 * unstable MoE output behind an apparently successful prefill.
 */
__global__ void rocm_moe_build_runtime_original_to_grouped_kernel(
    const DeviceMoELayerRuntimeView *__restrict__ runtime,
    int *__restrict__ original_to_grouped,
    int current_slots,
    int max_slots,
    int num_experts)
{
    const int expert = blockIdx.x;
    if (!runtime || !original_to_grouped ||
        current_slots < 0 || max_slots <= 0 || current_slots > max_slots ||
        num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        runtime->active_bank > 1u ||
        runtime->prefill_route_capacity < static_cast<uint32_t>(max_slots) ||
        !runtime->route_expert_ids ||
        !runtime->route_participant_ids ||
        !runtime->expert_counts ||
        !runtime->expert_offsets ||
        !runtime->grouped_token_ids)
    {
        if (expert == 0 && threadIdx.x == 0)
        {
            printf("rocm_runtime_original_to_grouped_invalid_contract "
                   "current_slots=%d max_slots=%d num_experts=%d route_capacity=%u "
                   "route_experts=%p route_participants=%p counts=%p offsets=%p grouped=%p\n",
                   current_slots,
                   max_slots,
                   num_experts,
                   runtime ? runtime->prefill_route_capacity : 0u,
                   runtime ? runtime->route_expert_ids : nullptr,
                   runtime ? runtime->route_participant_ids : nullptr,
                   runtime ? runtime->expert_counts : nullptr,
                   runtime ? runtime->expert_offsets : nullptr,
                   runtime ? runtime->grouped_token_ids : nullptr);
        }
        fail_fast_incomplete_llep_transfer();
        return;
    }
    if (expert >= num_experts)
        return;

    const int count = runtime->expert_counts[expert];
    const int offset = runtime->expert_offsets[expert];
    if (count < 0 || offset < 0 || offset + count > max_slots)
    {
        if (threadIdx.x == 0)
        {
            printf("rocm_runtime_original_to_grouped_invalid_expert_range "
                   "participant=%u expert=%d count=%d offset=%d max_slots=%d\n",
                   runtime->participant_id,
                   expert,
                   count,
                   offset,
                   max_slots);
        }
        fail_fast_incomplete_llep_transfer();
        return;
    }

    const int local_participant = static_cast<int>(runtime->participant_id);
    for (int row = threadIdx.x; row < count; row += blockDim.x)
    {
        const int grouped_slot = offset + row;
        const int route_slot = runtime->grouped_token_ids[grouped_slot];
        if (route_slot < 0 ||
            route_slot >= current_slots ||
            route_slot >= max_slots ||
            runtime->route_expert_ids[route_slot] != expert ||
            runtime->route_participant_ids[route_slot] != local_participant)
        {
            printf("rocm_runtime_original_to_grouped_invalid_row "
                   "participant=%u expert=%d row=%d grouped_slot=%d route_slot=%d "
                   "route_expert=%d route_participant=%d\n",
                   runtime->participant_id,
                   expert,
                   row,
                   grouped_slot,
                   route_slot,
                   (route_slot >= 0 && route_slot < max_slots)
                       ? runtime->route_expert_ids[route_slot]
                       : -1,
                   (route_slot >= 0 && route_slot < max_slots)
                       ? runtime->route_participant_ids[route_slot]
                       : -1);
            fail_fast_incomplete_llep_transfer();
            return;
        }
        original_to_grouped[route_slot] = grouped_slot;
    }
}

// =========================================================================
// Kernel 12: Scatter Tokens
//
// Scatters token indices and weights into expert-grouped arrays.
// Each thread handles one (token, slot) pair, uses atomicAdd on write_heads.
// Grid: ceil(total_slots / 256), Block: 256
// =========================================================================

__global__ void rocm_moe_scatter_tokens_kernel(
    const int *__restrict__ routing_indices,     // [total_slots]
    const float *__restrict__ routing_weights,   // [total_slots]
    const int *__restrict__ expert_offsets,       // [num_experts]
    int *__restrict__ write_heads,               // [num_experts] (init to 0)
    int *__restrict__ grouped_token_indices,     // [total_slots] OUT
    int *__restrict__ original_to_grouped,        // [total_slots] OUT, optional
    float *__restrict__ grouped_weights,         // [total_slots] OUT
    int total_slots, int num_experts, int top_k)
{
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= total_slots) return;

    int token_idx = slot / top_k;
    int expert_id = routing_indices[slot];

    if (expert_id >= 0 && expert_id < num_experts)
    {
        int write_pos = atomicAdd(&write_heads[expert_id], 1);
        int dest = expert_offsets[expert_id] + write_pos;
        grouped_token_indices[dest] = token_idx;
        if (original_to_grouped)
            original_to_grouped[slot] = dest;
        grouped_weights[dest] = routing_weights[slot];
    }
}

__global__ void rocm_moe_group_tokens_small_float_kernel(
    const float *__restrict__ routing_indices,
    const float *__restrict__ routing_weights,
    int *__restrict__ expert_counts,
    int *__restrict__ expert_offsets,
    int *__restrict__ grouped_token_indices,
    int *__restrict__ original_to_grouped,
    int *__restrict__ original_expert_ids,
    float *__restrict__ grouped_weights,
    int *__restrict__ active_expert_ids,
    int total_slots,
    int num_experts,
    int top_k,
    int max_active_experts)
{
    __shared__ int counts[kDeviceMoEMaxExperts];
    __shared__ int offsets[kDeviceMoEMaxExperts];
    __shared__ int write_heads[kDeviceMoEMaxExperts];

    const int tid = threadIdx.x;
    if (tid < num_experts)
    {
        counts[tid] = 0;
        offsets[tid] = 0;
        write_heads[tid] = 0;
    }
    __syncthreads();

    if (tid < total_slots)
    {
        grouped_token_indices[tid] = 0;
        original_to_grouped[tid] = -1;
        /*
         * The small-M planner feeds two later grouped verifier contracts:
         * ordered scatter needs route-slot -> grouped-slot, while parallel
         * down publication needs route-slot -> expert-id.  Keep both maps
         * authored by this single grouping kernel so verifier replay cannot
         * accidentally consume stale scratch from a previous batch.
         */
        original_expert_ids[tid] = -1;
        grouped_weights[tid] = 0.0f;
        const int expert_id = static_cast<int>(routing_indices[tid]);
        original_expert_ids[tid] =
            (expert_id >= 0 && expert_id < num_experts) ? expert_id : -1;
        if (expert_id >= 0 && expert_id < num_experts)
            atomicAdd(&counts[expert_id], 1);
    }
    __syncthreads();

    if (tid == 0)
    {
        int running = 0;
        int active = 0;
        for (int expert = 0; expert < num_experts; ++expert)
        {
            offsets[expert] = running;
            const int count = counts[expert];
            if (count > 0 && active < max_active_experts)
                active_expert_ids[active++] = expert;
            running += count;
        }
        for (int slot = active; slot < max_active_experts; ++slot)
            active_expert_ids[slot] = -1;
    }
    __syncthreads();

    if (tid < num_experts)
    {
        expert_counts[tid] = counts[tid];
        expert_offsets[tid] = offsets[tid];
    }

    if (tid < total_slots)
    {
        const int expert_id = static_cast<int>(routing_indices[tid]);
        if (expert_id >= 0 && expert_id < num_experts)
        {
            const int local = atomicAdd(&write_heads[expert_id], 1);
            const int dest = offsets[expert_id] + local;
            grouped_token_indices[dest] = tid / top_k;
            original_to_grouped[tid] = dest;
            grouped_weights[dest] = routing_weights[tid];
        }
    }
}

__global__ void rocm_moe_prepare_shared_expert_group_kernel(
    int *__restrict__ expert_offsets,
    int *__restrict__ expert_counts,
    int *__restrict__ grouped_token_indices,
    int *__restrict__ original_to_grouped,
    float *__restrict__ grouped_weights,
    int *__restrict__ active_expert_ids,
    int seq_len)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0)
    {
        expert_offsets[0] = 0;
        expert_counts[0] = seq_len;
        active_expert_ids[0] = 0;
    }
    if (idx < seq_len)
    {
        grouped_token_indices[idx] = idx;
        original_to_grouped[idx] = idx;
        grouped_weights[idx] = 1.0f;
    }
}

__global__ void rocm_moe_prefill_group_clear_kernel(
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    int max_slots, int num_experts, int clear_route_metadata)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (!runtime) return;

    if (idx < num_experts && runtime->expert_counts && runtime->expert_offsets)
    {
        runtime->expert_counts[idx] = 0;
        runtime->expert_offsets[idx] = 0;
    }

    if (idx < max_slots)
    {
        if (clear_route_metadata && runtime->route_expert_ids) runtime->route_expert_ids[idx] = -1;
        if (clear_route_metadata && runtime->route_weights) runtime->route_weights[idx] = 0.0f;
        if (clear_route_metadata && runtime->route_participant_ids) runtime->route_participant_ids[idx] = -1;
        if (runtime->grouped_token_ids) runtime->grouped_token_ids[idx] = 0;
        if (runtime->grouped_route_weights) runtime->grouped_route_weights[idx] = 0.0f;
    }
}

/**
 * @brief Return whether a runtime-prefill route can be computed by this shard.
 *
 * StaticOwner LocalTP verifier prefill is intentionally a local partial.  The
 * router still produces global expert ids, but this participant must group
 * only experts whose active placement-bank descriptor is resident and marked
 * LocalCompute.  LLEP planning does not use this predicate during its first
 * grouping pass because it needs global route counts before assigning rows.
 */
__device__ __forceinline__ bool rocm_moe_prefill_static_local_ready(
    const DeviceMoELayerRuntimeView *__restrict__ runtime,
    int expert_id,
    int num_experts)
{
    if (!runtime ||
        expert_id < 0 ||
        expert_id >= num_experts ||
        expert_id >= kDeviceMoEMaxExperts ||
        runtime->active_bank > 1u)
    {
        return false;
    }

    const DeviceMoEPlacementBankView &bank = runtime->banks[runtime->active_bank];
    const DeviceMoEExpertDescriptorView &desc = bank.experts[expert_id];
    const uint32_t local_bit =
        runtime_participant_bit(static_cast<int>(runtime->participant_id));
    return bank.local_compute_mask[expert_id] != 0u &&
           (bank.resident_participant_mask[expert_id] & local_bit) != 0u &&
           desc.local_slot >= 0 &&
           rebalance_expert_desc_ready(desc);
}

__global__ void rocm_moe_prefill_group_cast_count_kernel(
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    const float *__restrict__ routing_indices,
    const float *__restrict__ routing_weights,
    int current_slots, int max_slots, int num_experts,
    int filter_to_local_runtime_experts)
{
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (!runtime || slot >= max_slots) return;
    if (!runtime->route_expert_ids || !runtime->route_weights || !runtime->route_participant_ids) return;
    if (runtime->prefill_route_capacity < static_cast<uint32_t>(max_slots)) return;

    int expert_id = -1;
    float weight = 0.0f;
    int participant_id = -1;
    if (slot < current_slots)
    {
        expert_id = static_cast<int>(routing_indices[slot]);
        weight = routing_weights[slot];
        if (expert_id < 0 || expert_id >= num_experts ||
            (filter_to_local_runtime_experts != 0 &&
             !rocm_moe_prefill_static_local_ready(runtime, expert_id, num_experts)))
        {
            expert_id = -1;
            weight = 0.0f;
        }
        else
        {
            participant_id = static_cast<int>(runtime->participant_id);
        }
    }

    runtime->route_expert_ids[slot] = expert_id;
    runtime->route_weights[slot] = weight;
    runtime->route_participant_ids[slot] = participant_id;
}

__global__ void rocm_moe_prefill_group_count_assigned_kernel(
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    int current_slots, int max_slots, int num_experts)
{
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (!runtime || slot >= max_slots || slot >= current_slots) return;
    if (!runtime->route_expert_ids || !runtime->route_participant_ids || !runtime->expert_counts) return;
    if (runtime->prefill_route_capacity < static_cast<uint32_t>(max_slots)) return;

    const int expert_id = runtime->route_expert_ids[slot];
    const int participant_id = runtime->route_participant_ids[slot];
    if (expert_id >= 0 &&
        expert_id < num_experts &&
        participant_id == static_cast<int>(runtime->participant_id))
    {
        atomicAdd(&runtime->expert_counts[expert_id], 1);
    }
}

__global__ void rocm_moe_prefill_group_exclusive_scan_kernel(
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    int num_experts)
{
    if (!runtime || threadIdx.x != 0) return;
    if (!runtime->expert_counts || !runtime->expert_offsets) return;

    int running = 0;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        const int count = runtime->expert_counts[expert];
        runtime->expert_offsets[expert] = running;
        running += count;
    }
}

__global__ void rocm_moe_prefill_group_scatter_deterministic_kernel(
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    int current_slots, int max_slots, int num_experts, int top_k)
{
    const int expert = blockIdx.x;
    if (!runtime || expert >= num_experts) return;
    if (!runtime->route_expert_ids || !runtime->route_weights ||
        !runtime->route_participant_ids ||
        !runtime->expert_offsets || !runtime->expert_counts ||
        !runtime->grouped_token_ids || !runtime->grouped_route_weights) return;
    if (runtime->prefill_route_capacity < static_cast<uint32_t>(max_slots)) return;

    __shared__ int local_offsets[kGateLogitsGroupedThreads];
    __shared__ int running_count;
    __shared__ int expert_offset;
    __shared__ int expert_end;
    __shared__ int participant_id;

    if (threadIdx.x == 0)
    {
        running_count = 0;
        expert_offset = runtime->expert_offsets[expert];
        expert_end = runtime->expert_offsets[expert] + runtime->expert_counts[expert];
        participant_id = static_cast<int>(runtime->participant_id);
    }
    __syncthreads();

    for (int chunk_start = 0; chunk_start < max_slots; chunk_start += blockDim.x)
    {
        const int slot = chunk_start + threadIdx.x;
        const int keep =
            slot < current_slots &&
            runtime->route_expert_ids[slot] == expert &&
            runtime->route_participant_ids[slot] == participant_id;

        local_offsets[threadIdx.x] = keep;
        __syncthreads();

        if (threadIdx.x == 0)
        {
            int chunk_running = running_count;
            for (int lane = 0; lane < blockDim.x; ++lane)
            {
                const int count = local_offsets[lane];
                local_offsets[lane] = chunk_running;
                chunk_running += count;
            }
            running_count = chunk_running;
        }
        __syncthreads();

        if (keep)
        {
            const int dest = expert_offset + local_offsets[threadIdx.x];
            if (dest < expert_end)
            {
                runtime->grouped_token_ids[dest] = slot;
                runtime->grouped_route_weights[dest] = runtime->route_weights[slot];
            }
        }
        __syncthreads();
    }
}

__global__ void rocm_moe_prefill_llep_count_expert_routes_kernel(
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    int current_slots, int max_slots, int num_experts)
{
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (!runtime || slot >= current_slots || slot >= max_slots) return;
    if (!runtime->route_expert_ids || !runtime->expert_counts) return;
    if (runtime->prefill_route_capacity < static_cast<uint32_t>(max_slots)) return;

    const int expert_id = runtime->route_expert_ids[slot];
    if (expert_id >= 0 && expert_id < num_experts)
        atomicAdd(&runtime->expert_counts[expert_id], 1);
}

__global__ void rocm_moe_prefill_llep_clear_assignment_kernel(
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    int max_slots, int num_experts)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (!runtime) return;

    if (idx < num_experts && runtime->expert_counts && runtime->expert_offsets)
    {
        runtime->expert_counts[idx] = 0;
        runtime->expert_offsets[idx] = -1;
    }

    if (idx < max_slots)
    {
        if (runtime->grouped_token_ids) runtime->grouped_token_ids[idx] = 0;
        if (runtime->grouped_route_weights) runtime->grouped_route_weights[idx] = 0.0f;
    }

    int32_t *split_ends = static_cast<int32_t *>(runtime->reserved_ptrs[0]);
    const int split_items = num_experts * static_cast<int>(kDeviceMoEMaxParticipants);
    if (idx < split_items && split_ends)
        split_ends[idx] = 0;

    if (idx == 0)
    {
        runtime->reserved_u64[2] = 0ULL;
        runtime->reserved_u64[3] = 0ULL;
    }
}

__global__ void rocm_moe_prefill_llep_clear_current_batch_plan_kernel(
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    int num_experts)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (!runtime) return;

    if (idx < num_experts && runtime->expert_counts)
        runtime->expert_counts[idx] = 0;

    int32_t *split_ends = static_cast<int32_t *>(runtime->reserved_ptrs[0]);
    const int split_items = num_experts * static_cast<int>(kDeviceMoEMaxParticipants);
    if (idx < split_items && split_ends)
        split_ends[idx] = 0;

    if (idx == 0)
    {
        runtime->reserved_u64[2] = 0ULL;
        runtime->reserved_u64[3] = 0ULL;
    }
}

__global__ void rocm_moe_prefill_llep_plan_resident_splits_kernel(
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    int current_slots, int max_slots,
    int num_experts)
{
    if (!runtime || blockIdx.x != 0 || threadIdx.x != 0) return;
    if (!runtime->route_expert_ids || !runtime->route_participant_ids ||
        !runtime->expert_counts || !runtime->expert_offsets || !runtime->reserved_ptrs[0]) return;
    if (runtime->active_bank > 1u ||
        runtime->participant_count == 0u ||
        runtime->participant_count > kDeviceMoEMaxParticipants ||
        current_slots < 0 ||
        max_slots <= 0 ||
        current_slots > max_slots ||
        runtime->prefill_route_capacity < static_cast<uint32_t>(max_slots))
    {
        return;
    }
    uint64_t participant_load[kDeviceMoEMaxParticipants];
    for (uint32_t participant = 0; participant < kDeviceMoEMaxParticipants; ++participant)
        participant_load[participant] = 0ULL;

    const uint32_t participant_count = runtime->participant_count;
    const DeviceMoEPlacementBankView &bank = runtime->banks[runtime->active_bank];
    auto *split_ends = static_cast<int32_t *>(runtime->reserved_ptrs[0]);

    for (int order = 0; order < num_experts; ++order)
    {
        const int best_expert =
            llaminar2::least_loaded_ep::selectHighestLoadUnassignedExpert(
                runtime->expert_counts,
                runtime->expert_offsets,
                static_cast<uint32_t>(num_experts));
        if (best_expert < 0)
            break;
        const int best_load = runtime->expert_counts[best_expert];
        runtime->expert_offsets[best_expert] = 0;
        if (best_load <= 0)
            continue;

        const auto &desc = bank.experts[best_expert];
        const int owner = desc.owner_participant;
        const uint32_t resident_mask =
            llaminar2::least_loaded_ep::normalizeResidentParticipantMask(
                bank.resident_participant_mask[best_expert],
                owner,
                runtime->participant_id,
                participant_count);
        const uint32_t default_participant =
            (owner >= 0 && static_cast<uint32_t>(owner) < participant_count)
                ? static_cast<uint32_t>(owner)
                : (runtime->participant_id < participant_count ? runtime->participant_id : 0u);

        uint32_t destination_counts[kDeviceMoEMaxParticipants];
        for (uint32_t participant = 0; participant < kDeviceMoEMaxParticipants; ++participant)
            destination_counts[participant] = 0u;

        llaminar2::least_loaded_ep::assignLeastLoadedResidentSplitCounts(
            static_cast<uint64_t>(best_load),
            resident_mask,
            participant_load,
            participant_count,
            default_participant,
            destination_counts);

        int cumulative = 0;
        const int split_base = best_expert * static_cast<int>(kDeviceMoEMaxParticipants);
        for (uint32_t participant = 0; participant < kDeviceMoEMaxParticipants; ++participant)
        {
            if (participant < participant_count)
                cumulative += static_cast<int>(destination_counts[participant]);
            split_ends[split_base + static_cast<int>(participant)] = cumulative;
        }
    }
}

__global__ void rocm_moe_prefill_llep_assign_routes_from_splits_kernel(
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    int current_slots, int max_slots,
    int num_experts)
{
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (!runtime || slot >= current_slots || slot >= max_slots) return;
    if (!runtime->route_expert_ids || !runtime->route_participant_ids ||
        !runtime->expert_offsets || !runtime->reserved_ptrs[0]) return;
    if (runtime->active_bank > 1u ||
        runtime->participant_count == 0u ||
        runtime->participant_count > kDeviceMoEMaxParticipants ||
        runtime->prefill_route_capacity < static_cast<uint32_t>(max_slots))
    {
        return;
    }
    const int expert_id = runtime->route_expert_ids[slot];
    if (expert_id < 0 || expert_id >= num_experts)
        return;

    const uint32_t participant_count = runtime->participant_count;
    const DeviceMoEPlacementBankView &bank = runtime->banks[runtime->active_bank];
    const auto &desc = bank.experts[expert_id];
    const int owner = desc.owner_participant;
    const uint32_t default_participant =
        (owner >= 0 && static_cast<uint32_t>(owner) < participant_count)
            ? static_cast<uint32_t>(owner)
            : (runtime->participant_id < participant_count ? runtime->participant_id : 0u);

    auto *split_ends = static_cast<int32_t *>(runtime->reserved_ptrs[0]);
    const int ordinal = atomicAdd(runtime->expert_offsets + expert_id, 1);
    const int split_base = expert_id * static_cast<int>(kDeviceMoEMaxParticipants);
    uint32_t assigned_participant = default_participant;
    for (uint32_t participant = 0; participant < participant_count; ++participant)
    {
        if (ordinal < split_ends[split_base + static_cast<int>(participant)])
        {
            assigned_participant = participant;
            break;
        }
    }
    runtime->route_participant_ids[slot] = static_cast<int32_t>(assigned_participant);
}

__global__ void rocm_moe_prefill_llep_plan_current_batch_kernel(
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    int current_slots, int max_slots,
    int num_experts,
    uint32_t min_chunk_tokens,
    uint32_t alpha_numerator,
    uint32_t alpha_denominator,
    uint32_t lambda_numerator,
    uint32_t lambda_denominator,
    uint64_t min_spread_improvement,
    uint32_t min_spread_improvement_divisor,
    uint64_t min_spread_improvement_per_transfer,
    uint64_t min_foreign_rows_per_transfer,
    uint32_t max_weight_transfers,
    int enable_balanced_skip)
{
    __shared__ uint64_t expert_loads[kDeviceMoEMaxExperts];
    __shared__ uint32_t owner_participants[kDeviceMoEMaxExperts];
    __shared__ uint32_t resident_participant_masks[kDeviceMoEMaxExperts];
    __shared__ uint32_t sorted_experts[kDeviceMoEMaxExperts];
    __shared__ uint64_t pending_load[kDeviceMoEMaxParticipants];
    __shared__ uint64_t assigned_load[kDeviceMoEMaxParticipants];

    if (!runtime) return;

    const int lane = threadIdx.x;
    if (lane == 0)
    {
        runtime->reserved_u64[2] = 0ULL;
        runtime->reserved_u64[3] = 0ULL;
    }

    if (runtime->active_bank > 1u ||
        runtime->participant_count == 0u ||
        runtime->participant_count > kDeviceMoEMaxParticipants ||
        current_slots < 0 ||
        max_slots <= 0 ||
        current_slots > max_slots ||
        num_experts <= 0 ||
        num_experts > kDeviceMoEMaxExperts ||
        runtime->prefill_route_capacity < static_cast<uint32_t>(max_slots) ||
        !runtime->expert_counts ||
        !runtime->reserved_ptrs[1] ||
        !runtime->reserved_ptrs[2])
    {
        return;
    }

    const uint32_t participant_count = runtime->participant_count;
    const DeviceMoEPlacementBankView &bank = runtime->banks[runtime->active_bank];

    if (lane < kDeviceMoEMaxParticipants)
    {
        pending_load[lane] = 0ULL;
        assigned_load[lane] = 0ULL;
    }

    for (int expert = lane; expert < num_experts; expert += blockDim.x)
    {
        const int raw_count = runtime->expert_counts[expert];
        expert_loads[expert] = raw_count > 0 ? static_cast<uint64_t>(raw_count) : 0ULL;
        const int owner = bank.experts[expert].owner_participant;
        owner_participants[expert] =
            (owner >= 0 && static_cast<uint32_t>(owner) < participant_count)
                ? static_cast<uint32_t>(owner)
                : (runtime->participant_id < participant_count ? runtime->participant_id : 0u);
        uint32_t resident_mask =
            bank.resident_participant_mask[expert] &
            runtime_valid_participant_mask(participant_count);
        resident_participant_masks[expert] = resident_mask;
        sorted_experts[expert] = static_cast<uint32_t>(expert);
    }
    __syncthreads();

    if (lane != 0)
        return;

    auto *spans = static_cast<llaminar2::least_loaded_ep::LeastLoadedExpertAssignmentSpan *>(
        runtime->reserved_ptrs[1]);
    auto *transfers = static_cast<llaminar2::least_loaded_ep::LeastLoadedExpertWeightTransfer *>(
        runtime->reserved_ptrs[2]);
    const uint64_t span_capacity_u64 = runtime->reserved_u64[0];
    const uint64_t transfer_capacity_u64 = runtime->reserved_u64[1];
    const uint32_t span_capacity =
        span_capacity_u64 > 0xffffffffULL ? 0xffffffffu : static_cast<uint32_t>(span_capacity_u64);
    const uint32_t transfer_capacity =
        transfer_capacity_u64 > 0xffffffffULL ? 0xffffffffu : static_cast<uint32_t>(transfer_capacity_u64);

    llaminar2::least_loaded_ep::LeastLoadedExpertAssignmentConfig config{};
    config.expert_count = static_cast<uint32_t>(num_experts);
    config.participant_count = participant_count;
    config.min_chunk_tokens = min_chunk_tokens;
    config.alpha_numerator = alpha_numerator;
    config.alpha_denominator = alpha_denominator;
    config.lambda_numerator = lambda_numerator;
    config.lambda_denominator = lambda_denominator;
    config.min_spread_improvement = min_spread_improvement;
    config.min_spread_improvement_divisor = min_spread_improvement_divisor;
    config.min_spread_improvement_per_transfer = min_spread_improvement_per_transfer;
    config.min_foreign_rows_per_transfer = min_foreign_rows_per_transfer;
    config.max_weight_transfers = max_weight_transfers;
    config.enable_balanced_skip = enable_balanced_skip != 0;

    llaminar2::least_loaded_ep::LeastLoadedExpertAssignmentWorkspace workspace{};
    workspace.sorted_experts = sorted_experts;
    workspace.pending_load = pending_load;
    workspace.assigned_load = assigned_load;

    llaminar2::least_loaded_ep::LeastLoadedExpertAssignmentStatus status{};
    const bool ok = llaminar2::least_loaded_ep::planLeastLoadedExpertAssignment(
        expert_loads,
        owner_participants,
        config,
        workspace,
        spans,
        span_capacity,
        transfers,
        transfer_capacity,
        &status,
        resident_participant_masks);
    if (ok && status.overflow == 0u && status.invalid_config == 0u)
    {
        runtime->reserved_u64[2] = status.span_count;
        runtime->reserved_u64[3] = status.weight_transfer_count;
        int32_t *span_bounds = static_cast<int32_t *>(runtime->reserved_ptrs[0]);
        if (span_bounds)
        {
            for (int expert = 0; expert < num_experts; ++expert)
            {
                span_bounds[2 * expert] = -1;
                span_bounds[2 * expert + 1] = -1;
            }
            for (uint32_t idx = 0; idx < status.span_count; ++idx)
            {
                const uint32_t expert = spans[idx].expert;
                if (expert >= static_cast<uint32_t>(num_experts))
                    continue;
                int32_t &begin = span_bounds[2 * static_cast<int>(expert)];
                int32_t &end = span_bounds[2 * static_cast<int>(expert) + 1];
                if (begin < 0)
                    begin = static_cast<int32_t>(idx);
                end = static_cast<int32_t>(idx + 1u);
            }
        }
    }
}

__device__ __forceinline__ int rocm_moe_prefill_llep_default_owner_participant(
    const DeviceMoELayerRuntimeView *__restrict__ runtime,
    int expert)
{
    if (!runtime ||
        runtime->active_bank > 1u ||
        runtime->participant_count == 0u ||
        runtime->participant_count > kDeviceMoEMaxParticipants ||
        expert < 0 ||
        expert >= kDeviceMoEMaxExperts)
    {
        return -1;
    }
    const DeviceMoEPlacementBankView &bank = runtime->banks[runtime->active_bank];
    const int owner = bank.experts[expert].owner_participant;
    return (owner >= 0 && static_cast<uint32_t>(owner) < runtime->participant_count)
               ? owner
               : static_cast<int>(
                     runtime->participant_id < runtime->participant_count
                         ? runtime->participant_id
                 : 0u);
}

__device__ __forceinline__ bool rocm_moe_prefill_llep_local_expert_ready(
    const DeviceMoELayerRuntimeView *__restrict__ runtime,
    const DeviceMoEPlacementBankView &bank,
    int expert)
{
    if (!runtime ||
        expert < 0 ||
        expert >= static_cast<int>(runtime->expert_count) ||
        expert >= kDeviceMoEMaxExperts)
    {
        return false;
    }
    const auto &desc = bank.experts[expert];
    const uint32_t local_bit =
        runtime_participant_bit(static_cast<int>(runtime->participant_id));
    return bank.local_compute_mask[expert] != 0u &&
           (bank.resident_participant_mask[expert] & local_bit) != 0u &&
           desc.local_slot >= 0 &&
           rebalance_expert_desc_ready(desc);
}

__device__ __forceinline__ bool rocm_moe_prefill_llep_assignment_ready(
    const DeviceMoELayerRuntimeView *__restrict__ runtime,
    const DeviceMoEPlacementBankView &active_bank,
    int expert,
    int row,
    int assigned_participant)
{
    if (!runtime ||
        expert < 0 ||
        expert >= static_cast<int>(runtime->expert_count) ||
        expert >= kDeviceMoEMaxExperts ||
        assigned_participant < 0 ||
        static_cast<uint32_t>(assigned_participant) >= runtime->participant_count)
    {
        if (threadIdx.x == 0)
        {
            printf("prefill_llep_invalid_assignment_participant "
                   "participant=%u expert=%d row=%d assigned=%d participant_count=%u\\n",
                   runtime ? runtime->participant_id : 0u,
                   expert,
                   row,
                   assigned_participant,
                   runtime ? runtime->participant_count : 0u);
        }
        return false;
    }

    if (assigned_participant != static_cast<int32_t>(runtime->participant_id))
        return true;

    const uint32_t assigned_bit = runtime_participant_bit(assigned_participant);
    const int owner_participant =
        rocm_moe_prefill_llep_default_owner_participant(runtime, expert);
    const uint32_t resident_mask =
        llaminar2::least_loaded_ep::normalizeResidentParticipantMask(
            active_bank.resident_participant_mask[expert],
            owner_participant,
            runtime->participant_id,
            runtime->participant_count);
    if ((resident_mask & assigned_bit) == 0u)
    {
        if (threadIdx.x == 0)
        {
            printf("prefill_llep_assignment_not_resident "
                   "participant=%u expert=%d row=%d assigned=%d assigned_bit=%u "
                   "resident_mask=%u raw_resident_mask=%u owner=%d local_ready=%u\\n",
                   runtime->participant_id,
                   expert,
                   row,
                   assigned_participant,
                   assigned_bit,
                   resident_mask,
                   active_bank.resident_participant_mask[expert],
                   owner_participant,
                   rocm_moe_prefill_llep_local_expert_ready(runtime, active_bank, expert) ? 1u : 0u);
        }
        return false;
    }

    if (assigned_participant == static_cast<int32_t>(runtime->participant_id) &&
        !rocm_moe_prefill_llep_local_expert_ready(runtime, active_bank, expert))
    {
        if (threadIdx.x == 0)
        {
            printf("prefill_llep_assignment_local_not_ready "
                   "participant=%u expert=%d row=%d assigned=%d resident_mask=%u "
                   "raw_resident_mask=%u owner=%d\\n",
                   runtime->participant_id,
                   expert,
                   row,
                   assigned_participant,
                   resident_mask,
                   active_bank.resident_participant_mask[expert],
                   owner_participant);
        }
        return false;
    }
    return true;
}

__global__ void rocm_moe_prefill_llep_assign_routes_from_current_batch_spans_kernel(
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    int current_slots,
    int max_slots,
    int num_experts,
    int top_k,
    int allow_pending_transfers,
    const DeviceMoERebalanceStatusView *__restrict__ transfer_status,
    const DeviceMoERebalanceApplyStatusView *__restrict__ apply_status)
{
    const int expert = blockIdx.x;
    if (!runtime || expert >= num_experts)
        return;
    if (!runtime->route_expert_ids || !runtime->route_participant_ids ||
        !runtime->expert_counts || !runtime->expert_offsets ||
        !runtime->grouped_token_ids || !runtime->reserved_ptrs[1])
    {
        return;
    }
    if (top_k <= 0 ||
        runtime->prefill_route_capacity < static_cast<uint32_t>(max_slots))
        return;
    if (allow_pending_transfers)
    {
        const bool transfer_complete =
            prefill_llep_transfer_status_complete(
                transfer_status,
                runtime->reserved_u64[3],
                runtime->reserved_u64[2]);
        const bool apply_complete =
            prefill_llep_apply_status_complete(
                apply_status,
                runtime->reserved_u64[3]);
        if (!transfer_complete || !apply_complete)
        {
            if (expert == 0 && threadIdx.x == 0)
            {
                printf("prefill_llep_after_transfer_incomplete "
                       "participant=%u expected_transfers=%llu expected_spans=%llu "
                       "transfer_ok=%u transfer_magic=%u transfer_version=%u transfer_status=%u "
                       "plan_overflow=%u payload_overflow=%u planned=%u transfer_count=%u span_count=%u "
                       "apply_ok=%u apply_magic=%u apply_version=%u apply_status=%u seen=%u applied=%u "
                       "required_local=%u ready_local=%u invalid=%u missing_src=%u missing_dst=%u "
                       "mismatch=%u copy_incomplete=%u copied=%u\\n",
                       runtime->participant_id,
                       static_cast<unsigned long long>(runtime->reserved_u64[3]),
                       static_cast<unsigned long long>(runtime->reserved_u64[2]),
                       transfer_complete ? 1u : 0u,
                       transfer_status ? transfer_status->magic : 0u,
                       transfer_status ? transfer_status->version : 0u,
                       transfer_status ? transfer_status->status_code : 0u,
                       transfer_status ? transfer_status->plan_overflow : 0u,
                       transfer_status ? transfer_status->payload_bucket_overflow : 0u,
                       transfer_status ? transfer_status->planned_arrivals : 0u,
                       transfer_status ? transfer_status->llep_weight_transfer_count : 0u,
                       transfer_status ? transfer_status->llep_assignment_span_count : 0u,
                       apply_complete ? 1u : 0u,
                       apply_status ? apply_status->magic : 0u,
                       apply_status ? apply_status->version : 0u,
                       apply_status ? apply_status->status_code : 0u,
                       apply_status ? apply_status->plan_entries_seen : 0u,
                       apply_status ? apply_status->applied_arrivals : 0u,
                       apply_status ? apply_status->required_local_arrivals : 0u,
                       apply_status ? apply_status->ready_local_arrivals : 0u,
                       apply_status ? apply_status->invalid_plan_entries : 0u,
                       apply_status ? apply_status->missing_source_descriptors : 0u,
                       apply_status ? apply_status->missing_destination_slots : 0u,
                       apply_status ? apply_status->descriptor_mismatches : 0u,
                       apply_status ? apply_status->copy_incomplete : 0u,
                       apply_status ? apply_status->copied_arrivals : 0u);
            }
            fail_fast_incomplete_llep_transfer();
            return;
        }
    }
    else if (runtime->reserved_u64[3] != 0ULL)
    {
        return;
    }

    const uint64_t span_count_u64 = runtime->reserved_u64[2];
    const uint64_t span_capacity_u64 = runtime->reserved_u64[0];
    const uint32_t span_count =
        span_count_u64 > span_capacity_u64
            ? static_cast<uint32_t>(span_capacity_u64)
            : static_cast<uint32_t>(span_count_u64);
    const int expert_count = runtime->expert_counts[expert];
    const int expert_offset = runtime->expert_offsets[expert];
    if (expert_count <= 0 || expert_offset < 0)
        return;

    if (span_count == 0u)
    {
        const int assigned_participant =
            rocm_moe_prefill_llep_default_owner_participant(runtime, expert);
        if (assigned_participant < 0)
            return;
        const DeviceMoEPlacementBankView &active_bank = runtime->banks[runtime->active_bank];

        for (int row = threadIdx.x; row < expert_count; row += blockDim.x)
        {
            if (!rocm_moe_prefill_llep_assignment_ready(
                    runtime, active_bank, expert, row, assigned_participant))
            {
                fail_fast_incomplete_llep_transfer();
                return;
            }
            const int slot = runtime->grouped_token_ids[expert_offset + row];
            if (slot >= 0 && slot < current_slots && slot < max_slots &&
                runtime->route_expert_ids[slot] == expert)
            {
                runtime->route_participant_ids[slot] = assigned_participant;
            }
        }
        return;
    }

    const auto *spans =
        static_cast<const llaminar2::least_loaded_ep::LeastLoadedExpertAssignmentSpan *>(
            runtime->reserved_ptrs[1]);
    const auto *span_bounds = static_cast<const int32_t *>(runtime->reserved_ptrs[0]);
    int span_begin = 0;
    int span_end = static_cast<int>(span_count);
    if (span_bounds)
    {
        span_begin = span_bounds[2 * expert];
        span_end = span_bounds[2 * expert + 1];
        if (span_begin < 0 || span_end <= span_begin)
        {
            const int assigned_participant =
                rocm_moe_prefill_llep_default_owner_participant(runtime, expert);
            if (assigned_participant < 0)
                return;
            const DeviceMoEPlacementBankView &active_bank = runtime->banks[runtime->active_bank];

            for (int row = threadIdx.x; row < expert_count; row += blockDim.x)
            {
                if (!rocm_moe_prefill_llep_assignment_ready(
                        runtime, active_bank, expert, row, assigned_participant))
                {
                    fail_fast_incomplete_llep_transfer();
                    return;
                }
                const int slot = runtime->grouped_token_ids[expert_offset + row];
                if (slot >= 0 && slot < current_slots && slot < max_slots &&
                    runtime->route_expert_ids[slot] == expert)
                {
                    runtime->route_participant_ids[slot] = assigned_participant;
                }
            }
            return;
        }
        if (span_begin > static_cast<int>(span_count))
            span_begin = static_cast<int>(span_count);
        if (span_end > static_cast<int>(span_count))
            span_end = static_cast<int>(span_count);
    }

    for (int row = threadIdx.x; row < expert_count; row += blockDim.x)
    {
        int32_t assigned_participant = -1;
        for (int idx = span_begin; idx < span_end; ++idx)
        {
            const auto &span = spans[idx];
            if (span.expert != static_cast<uint32_t>(expert))
                continue;
            if (static_cast<uint32_t>(row) >= span.route_row_begin &&
                static_cast<uint32_t>(row) < span.route_row_end)
            {
                assigned_participant = static_cast<int32_t>(span.destination_participant);
                break;
            }
        }
        if (assigned_participant < 0)
        {
            assigned_participant =
                rocm_moe_prefill_llep_default_owner_participant(runtime, expert);
            if (assigned_participant < 0)
                continue;
        }
        const DeviceMoEPlacementBankView &active_bank = runtime->banks[runtime->active_bank];
        if (!rocm_moe_prefill_llep_assignment_ready(
                runtime, active_bank, expert, row, assigned_participant))
        {
            fail_fast_incomplete_llep_transfer();
            return;
        }

        const int slot = runtime->grouped_token_ids[expert_offset + row];
        if (slot >= 0 && slot < current_slots && slot < max_slots &&
            runtime->route_expert_ids[slot] == expert)
        {
            runtime->route_participant_ids[slot] = assigned_participant;
        }
    }
}

__global__ void rocm_moe_materialize_runtime_prefill_descriptor_tables_kernel(
    const DeviceMoELayerRuntimeView *__restrict__ runtime,
    DeviceNativeVNNIMatrixDescView *__restrict__ gate_descs,
    DeviceNativeVNNIMatrixDescView *__restrict__ up_descs,
    DeviceNativeVNNIMatrixDescView *__restrict__ down_descs,
    int num_experts)
{
    const int expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (!runtime || !gate_descs || !up_descs || !down_descs) return;
    if (expert >= num_experts || expert >= kDeviceMoEMaxExperts) return;
    if (runtime->active_bank > 1u) return;

    const DeviceMoEPlacementBankView &bank = runtime->banks[runtime->active_bank];
    const DeviceMoEExpertDescriptorView &desc = bank.experts[expert];
    const uint32_t local_bit =
        runtime_participant_bit(static_cast<int>(runtime->participant_id));
    const bool local_ready =
        bank.local_compute_mask[expert] != 0u &&
        (bank.resident_participant_mask[expert] & local_bit) != 0u &&
        desc.local_slot >= 0 &&
        rebalance_expert_desc_ready(desc);
    if (local_ready)
    {
        gate_descs[expert] = desc.gate;
        up_descs[expert] = desc.up;
        down_descs[expert] = desc.down;
    }
    else
    {
        gate_descs[expert] = DeviceNativeVNNIMatrixDescView{};
        up_descs[expert] = DeviceNativeVNNIMatrixDescView{};
        down_descs[expert] = DeviceNativeVNNIMatrixDescView{};
    }
}

__global__ void rocm_moe_prefill_gather_expert_runtime_kernel(
    const DeviceMoELayerRuntimeView *__restrict__ runtime,
    const float *__restrict__ hidden,
    float *__restrict__ batch_buffer,
    int expert_id, int max_tokens, int d_model)
{
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = max_tokens * d_model;
    if (!runtime || !hidden || !batch_buffer || linear >= total) return;
    if (!runtime->expert_counts || !runtime->expert_offsets || !runtime->grouped_token_ids) return;
    if (expert_id < 0 || expert_id >= static_cast<int>(runtime->expert_count)) return;
    if (runtime->prefill_token_capacity < static_cast<uint32_t>(max_tokens)) return;

    const int row = linear / d_model;
    const int col = linear - row * d_model;
    const int count = runtime->expert_counts[expert_id];
    const int offset = runtime->expert_offsets[expert_id];

    float value = 0.0f;
    if (row < count)
    {
        const int route_slot = runtime->grouped_token_ids[offset + row];
        const int token_id =
            runtime->top_k > 0u ? route_slot / static_cast<int>(runtime->top_k) : -1;
        if (token_id >= 0 && token_id < max_tokens)
            value = hidden[token_id * d_model + col];
    }

    batch_buffer[linear] = value;
}

__global__ void rocm_moe_prefill_scatter_expert_runtime_kernel(
    float *__restrict__ output,
    const float *__restrict__ expert_output,
    const DeviceMoELayerRuntimeView *__restrict__ runtime,
    int expert_id, int max_tokens, int d_model)
{
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = max_tokens * d_model;
    if (!runtime || !output || !expert_output || linear >= total) return;
    if (!runtime->expert_counts || !runtime->expert_offsets ||
        !runtime->grouped_token_ids || !runtime->grouped_route_weights) return;
    if (expert_id < 0 || expert_id >= static_cast<int>(runtime->expert_count)) return;
    if (runtime->prefill_token_capacity < static_cast<uint32_t>(max_tokens)) return;

    const int row = linear / d_model;
    const int col = linear - row * d_model;
    const int count = runtime->expert_counts[expert_id];
    if (row >= count) return;

    const int offset = runtime->expert_offsets[expert_id];
    const int route_slot = runtime->grouped_token_ids[offset + row];
    const int token_id =
        runtime->top_k > 0u ? route_slot / static_cast<int>(runtime->top_k) : -1;
    if (token_id < 0 || token_id >= max_tokens) return;

    const float weight = runtime->grouped_route_weights[offset + row];
    atomicAdd(&output[token_id * d_model + col], weight * expert_output[linear]);
}

// =========================================================================
// Phase 3 extern "C" bridge functions
// =========================================================================

extern "C"
{

bool hipMoE_count_per_expert(
    const int *routing_indices, int *expert_counts,
    int total_slots, int num_experts,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    const int blocks = (total_slots + threads - 1) / threads;

    hipLaunchKernelGGL(rocm_moe_count_per_expert_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        routing_indices, expert_counts, total_slots, num_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE count_per_expert kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_exclusive_scan(
    int *expert_counts, int *expert_offsets,
    int num_experts,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    hipLaunchKernelGGL(rocm_moe_exclusive_scan_kernel,
        dim3(1), dim3(1), 0, static_cast<hipStream_t>(stream),
        expert_counts, expert_offsets, num_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE exclusive_scan kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_max_expert_count(
    const int *expert_counts, int *d_max_out,
    int num_experts,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    hipLaunchKernelGGL(rocm_moe_max_expert_count_kernel,
        dim3(1), dim3(64), 0, static_cast<hipStream_t>(stream),
        expert_counts, d_max_out, num_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE max_expert_count kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_build_active_expert_list(
    const int *expert_counts,
    int *active_expert_ids,
    int num_experts,
    int max_active_experts,
    int device_idx,
    void *stream)
{
    if (!expert_counts || !active_expert_ids || num_experts <= 0 ||
        max_active_experts <= 0 || max_active_experts > num_experts || !stream)
        return false;

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    hipLaunchKernelGGL(rocm_moe_build_active_expert_list_kernel,
        dim3(1), dim3(1), 0, static_cast<hipStream_t>(stream),
        expert_counts, active_expert_ids, num_experts, max_active_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE build_active_expert_list kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_scatter_tokens(
    const int *routing_indices, const float *routing_weights,
    const int *expert_offsets, int *write_heads,
    int *grouped_token_indices, int *original_to_grouped,
    float *grouped_weights,
    int total_slots, int num_experts, int top_k,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    const int blocks = (total_slots + threads - 1) / threads;

    hipLaunchKernelGGL(rocm_moe_scatter_tokens_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        routing_indices, routing_weights, expert_offsets, write_heads,
        grouped_token_indices, original_to_grouped, grouped_weights,
        total_slots, num_experts, top_k);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE scatter_tokens kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_build_active_expert_list_runtime(
    const void *runtime,
    int *active_expert_ids,
    int num_experts,
    int max_active_experts,
    int device_idx,
    void *stream)
{
    if (!runtime || !active_expert_ids || num_experts <= 0 ||
        max_active_experts <= 0 || max_active_experts > num_experts || !stream)
        return false;

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    hipLaunchKernelGGL(rocm_moe_build_active_expert_list_runtime_kernel,
        dim3(1), dim3(1), 0, static_cast<hipStream_t>(stream),
        static_cast<const DeviceMoELayerRuntimeView *>(runtime),
        active_expert_ids,
        num_experts,
        max_active_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE build_active_expert_list_runtime kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_build_runtime_original_to_grouped(
    const void *runtime,
    int *original_to_grouped,
    int current_slots,
    int max_slots,
    int num_experts,
    int top_k,
    int device_idx,
    void *stream)
{
    if (!runtime || !original_to_grouped || !stream ||
        current_slots < 0 || max_slots <= 0 || current_slots > max_slots ||
        num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    auto hip_stream = static_cast<hipStream_t>(stream);
    hipError_t err = hipMemsetAsync(
        original_to_grouped,
        0xff,
        static_cast<size_t>(max_slots) * sizeof(int),
        hip_stream);
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE runtime original_to_grouped memset failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    hipLaunchKernelGGL(rocm_moe_build_runtime_original_to_grouped_kernel,
        dim3(num_experts), dim3(256), 0, hip_stream,
        static_cast<const DeviceMoELayerRuntimeView *>(runtime),
        original_to_grouped,
        current_slots,
        max_slots,
        num_experts);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE build_runtime_original_to_grouped kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_group_tokens_small_float(
    const float *routing_indices, const float *routing_weights,
    int *expert_counts, int *expert_offsets,
    int *grouped_token_indices, int *original_to_grouped,
    int *original_expert_ids,
    float *grouped_weights,
    int *active_expert_ids,
    int total_slots, int num_experts, int top_k,
    int max_active_experts,
    int device_idx, void *stream)
{
    /*
     * This ABI is intentionally stricter than the legacy prefill grouping
     * helper.  MTP verifier rows may publish through the original-route
     * parallel down kernel, so the caller must provide original_expert_ids
     * alongside original_to_grouped.  Returning success without that map would
     * make the grouped path descriptor-dependent on old workspace contents.
     */
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!stream ||
        !routing_indices || !routing_weights || !expert_counts || !expert_offsets ||
        !grouped_token_indices || !original_to_grouped ||
        !original_expert_ids ||
        !grouped_weights || !active_expert_ids ||
        total_slots <= 0 || total_slots > 64 ||
        num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK ||
        max_active_experts <= 0 || max_active_experts > total_slots ||
        max_active_experts > num_experts)
    {
        fprintf(stderr, "HIP MoE small float grouping invalid arguments\n");
        return false;
    }

    hipLaunchKernelGGL(rocm_moe_group_tokens_small_float_kernel,
        dim3(1), dim3(256), 0, static_cast<hipStream_t>(stream),
        routing_indices, routing_weights,
        expert_counts, expert_offsets,
        grouped_token_indices, original_to_grouped, original_expert_ids, grouped_weights,
        active_expert_ids,
        total_slots, num_experts, top_k,
        max_active_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE small float grouping kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_prepare_shared_expert_group(
    int *expert_offsets,
    int *expert_counts,
    int *grouped_token_indices,
    int *original_to_grouped,
    float *grouped_weights,
    int *active_expert_ids,
    int seq_len,
    int device_idx,
    void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!stream || !expert_offsets || !expert_counts ||
        !grouped_token_indices || !original_to_grouped ||
        !grouped_weights || !active_expert_ids ||
        seq_len <= 0)
    {
        fprintf(stderr, "HIP MoE shared expert grouping invalid arguments\n");
        return false;
    }

    const int threads = 256;
    const int blocks = (seq_len + threads - 1) / threads;
    hipLaunchKernelGGL(rocm_moe_prepare_shared_expert_group_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        expert_offsets, expert_counts,
        grouped_token_indices, original_to_grouped, grouped_weights,
        active_expert_ids, seq_len);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE shared expert grouping kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_group_prefill_routes_runtime(
    const float *routing_indices, const float *routing_weights,
    void *runtime,
    int current_slots, int max_slots, int num_experts, int top_k,
    int filter_to_local_runtime_experts,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!runtime || !routing_indices || !routing_weights ||
        current_slots < 0 || max_slots <= 0 || current_slots > max_slots ||
        num_experts <= 0 || num_experts > kDeviceMoEMaxExperts || top_k <= 0)
    {
        fprintf(stderr, "HIP MoE group_prefill_routes invalid arguments\n");
        return false;
    }

    auto *runtime_view = static_cast<DeviceMoELayerRuntimeView *>(runtime);
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    const int threads = 256;
    const int clear_items = (max_slots > num_experts) ? max_slots : num_experts;
    const int clear_blocks = (clear_items + threads - 1) / threads;

    hipLaunchKernelGGL(rocm_moe_prefill_group_clear_kernel,
        dim3(clear_blocks), dim3(threads), 0, hip_stream,
        runtime_view, max_slots, num_experts, 1);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill group clear kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    const int slot_blocks = (max_slots + threads - 1) / threads;
    hipLaunchKernelGGL(rocm_moe_prefill_group_cast_count_kernel,
        dim3(slot_blocks), dim3(threads), 0, hip_stream,
        runtime_view, routing_indices, routing_weights,
        current_slots, max_slots, num_experts,
        filter_to_local_runtime_experts);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill group cast/count kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    hipLaunchKernelGGL(rocm_moe_prefill_group_count_assigned_kernel,
        dim3(slot_blocks), dim3(threads), 0, hip_stream,
        runtime_view, current_slots, max_slots, num_experts);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill group assigned-count kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    hipLaunchKernelGGL(rocm_moe_prefill_group_exclusive_scan_kernel,
        dim3(1), dim3(1), 0, hip_stream,
        runtime_view, num_experts);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill group scan kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    hipLaunchKernelGGL(rocm_moe_prefill_group_scatter_deterministic_kernel,
        dim3(num_experts), dim3(threads), 0, hip_stream,
        runtime_view, current_slots, max_slots, num_experts, top_k);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill group scatter kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    return true;
}

bool hipMoE_regroup_prefill_routes_runtime_assignments(
    void *runtime,
    int current_slots, int max_slots, int num_experts, int top_k,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!runtime || !stream ||
        current_slots < 0 || max_slots <= 0 || current_slots > max_slots ||
        num_experts <= 0 || num_experts > kDeviceMoEMaxExperts || top_k <= 0)
    {
        fprintf(stderr, "HIP MoE regroup_prefill_routes invalid arguments\n");
        return false;
    }

    auto *runtime_view = static_cast<DeviceMoELayerRuntimeView *>(runtime);
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    const int threads = 256;
    const int split_items = num_experts * static_cast<int>(kDeviceMoEMaxParticipants);
    int clear_items = (max_slots > num_experts) ? max_slots : num_experts;
    clear_items = (clear_items > split_items) ? clear_items : split_items;
    const int clear_blocks = (clear_items + threads - 1) / threads;
    const int slot_blocks = (max_slots + threads - 1) / threads;

    hipLaunchKernelGGL(rocm_moe_prefill_group_clear_kernel,
        dim3(clear_blocks), dim3(threads), 0, hip_stream,
        runtime_view, max_slots, num_experts, 0);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill regroup clear kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    hipLaunchKernelGGL(rocm_moe_prefill_group_count_assigned_kernel,
        dim3(slot_blocks), dim3(threads), 0, hip_stream,
        runtime_view, current_slots, max_slots, num_experts);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill regroup assigned-count kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    hipLaunchKernelGGL(rocm_moe_prefill_group_exclusive_scan_kernel,
        dim3(1), dim3(1), 0, hip_stream,
        runtime_view, num_experts);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill regroup scan kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    hipLaunchKernelGGL(rocm_moe_prefill_group_scatter_deterministic_kernel,
        dim3(num_experts), dim3(threads), 0, hip_stream,
        runtime_view, current_slots, max_slots, num_experts, top_k);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill regroup scatter kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    return true;
}

bool hipMoE_assign_prefill_routes_least_loaded_resident(
    void *runtime,
    int current_slots, int max_slots, int num_experts, int top_k,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!runtime || !stream ||
        current_slots < 0 || max_slots <= 0 || current_slots > max_slots ||
        num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK)
    {
        fprintf(stderr, "HIP MoE assign_prefill_routes_least_loaded_resident invalid arguments\n");
        return false;
    }

    auto *runtime_view = static_cast<DeviceMoELayerRuntimeView *>(runtime);
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    const int threads = 256;
    const int clear_items = (max_slots > num_experts) ? max_slots : num_experts;
    const int clear_blocks = (clear_items + threads - 1) / threads;
    const int slot_blocks = (max_slots + threads - 1) / threads;

    hipLaunchKernelGGL(rocm_moe_prefill_llep_clear_assignment_kernel,
        dim3(clear_blocks), dim3(threads), 0, hip_stream,
        runtime_view, max_slots, num_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill LLEP clear kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    hipLaunchKernelGGL(rocm_moe_prefill_llep_count_expert_routes_kernel,
        dim3(slot_blocks), dim3(threads), 0, hip_stream,
        runtime_view, current_slots, max_slots, num_experts);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill LLEP count kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    hipLaunchKernelGGL(rocm_moe_prefill_llep_plan_resident_splits_kernel,
        dim3(1), dim3(1), 0, hip_stream,
        runtime_view, current_slots, max_slots, num_experts);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill LLEP resident-split planning kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    hipLaunchKernelGGL(rocm_moe_prefill_llep_assign_routes_from_splits_kernel,
        dim3(slot_blocks), dim3(threads), 0, hip_stream,
        runtime_view, current_slots, max_slots, num_experts);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill LLEP split assignment kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    return true;
}

bool hipMoE_plan_prefill_routes_least_loaded_current_batch(
    void *runtime,
    int current_slots, int max_slots, int num_experts, int top_k,
    uint32_t min_chunk_tokens,
    uint32_t alpha_numerator,
    uint32_t alpha_denominator,
    uint32_t lambda_numerator,
    uint32_t lambda_denominator,
    uint64_t min_spread_improvement,
    uint32_t min_spread_improvement_divisor,
    uint64_t min_spread_improvement_per_transfer,
    uint64_t min_foreign_rows_per_transfer,
    uint32_t max_weight_transfers,
    int enable_balanced_skip,
    int device_idx, void *stream)
{
    (void)top_k;
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!runtime || !stream ||
        current_slots < 0 || max_slots <= 0 || current_slots > max_slots ||
        num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        alpha_numerator == 0u || alpha_denominator == 0u ||
        lambda_numerator == 0u || lambda_denominator == 0u)
    {
        fprintf(stderr, "HIP MoE plan_prefill_routes_least_loaded_current_batch invalid arguments\n");
        return false;
    }

    auto *runtime_view = static_cast<DeviceMoELayerRuntimeView *>(runtime);
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    const int threads = 256;
    const int split_items = num_experts * static_cast<int>(kDeviceMoEMaxParticipants);
    const int clear_items = (split_items > num_experts) ? split_items : num_experts;
    const int clear_blocks = (clear_items + threads - 1) / threads;
    const int slot_blocks = (max_slots + threads - 1) / threads;

    hipLaunchKernelGGL(rocm_moe_prefill_llep_clear_current_batch_plan_kernel,
        dim3(clear_blocks), dim3(threads), 0, hip_stream,
        runtime_view, num_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill LLEP current-batch clear kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    hipLaunchKernelGGL(rocm_moe_prefill_llep_count_expert_routes_kernel,
        dim3(slot_blocks), dim3(threads), 0, hip_stream,
        runtime_view, current_slots, max_slots, num_experts);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill LLEP current-batch count kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    hipLaunchKernelGGL(rocm_moe_prefill_llep_plan_current_batch_kernel,
        dim3(1), dim3(threads), 0, hip_stream,
        runtime_view,
        current_slots,
        max_slots,
        num_experts,
        min_chunk_tokens,
        alpha_numerator,
        alpha_denominator,
        lambda_numerator,
        lambda_denominator,
        min_spread_improvement,
        min_spread_improvement_divisor,
        min_spread_improvement_per_transfer,
        min_foreign_rows_per_transfer,
        max_weight_transfers,
        enable_balanced_skip);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill LLEP current-batch plan kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    return true;
}

bool hipMoE_assign_prefill_routes_from_llep_current_batch_plan_no_transfers(
    void *runtime,
    int current_slots, int max_slots, int num_experts, int top_k,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!runtime || !stream ||
        current_slots < 0 || max_slots <= 0 || current_slots > max_slots ||
        num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK)
    {
        fprintf(stderr, "HIP MoE assign_prefill_routes_from_llep_current_batch_plan_no_transfers invalid arguments\n");
        return false;
    }

    auto *runtime_view = static_cast<DeviceMoELayerRuntimeView *>(runtime);
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    const int threads = 256;

    hipLaunchKernelGGL(rocm_moe_prefill_llep_assign_routes_from_current_batch_spans_kernel,
        dim3(num_experts), dim3(threads), 0, hip_stream,
        runtime_view, current_slots, max_slots, num_experts, top_k, 0, nullptr, nullptr);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill LLEP current-batch span assignment kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    return true;
}

bool hipMoE_assign_prefill_routes_from_llep_current_batch_plan_after_transfers(
    void *runtime,
    int current_slots, int max_slots, int num_experts, int top_k,
    const void *transfer_status,
    const void *apply_status,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!runtime || !transfer_status || !apply_status || !stream ||
        current_slots < 0 || max_slots <= 0 || current_slots > max_slots ||
        num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK)
    {
        fprintf(stderr, "HIP MoE assign_prefill_routes_from_llep_current_batch_plan_after_transfers invalid arguments\n");
        return false;
    }

    auto *runtime_view = static_cast<DeviceMoELayerRuntimeView *>(runtime);
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    const int threads = 256;

    hipLaunchKernelGGL(rocm_moe_prefill_llep_assign_routes_from_current_batch_spans_kernel,
        dim3(num_experts), dim3(threads), 0, hip_stream,
        runtime_view,
        current_slots,
        max_slots,
        num_experts,
        top_k,
        1,
        static_cast<const DeviceMoERebalanceStatusView *>(transfer_status),
        static_cast<const DeviceMoERebalanceApplyStatusView *>(apply_status));

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill LLEP after-transfer span assignment kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    return true;
}

__attribute__((visibility("default"))) bool hipMoE_materialize_runtime_prefill_descriptor_tables(
    const void *runtime,
    DeviceNativeVNNIMatrixDescView *gate_descs,
    DeviceNativeVNNIMatrixDescView *up_descs,
    DeviceNativeVNNIMatrixDescView *down_descs,
    int num_experts,
    int device_idx,
    void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!runtime || !gate_descs || !up_descs || !down_descs || !stream ||
        num_experts <= 0 || num_experts > kDeviceMoEMaxExperts)
    {
        fprintf(stderr, "HIP MoE materialize_runtime_prefill_descriptor_tables invalid arguments\n");
        return false;
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    const int threads = kGateLogitsGroupedThreads;
    const int blocks = (num_experts + threads - 1) / threads;
    hipLaunchKernelGGL(rocm_moe_materialize_runtime_prefill_descriptor_tables_kernel,
        dim3(blocks), dim3(threads), 0, hip_stream,
        static_cast<const DeviceMoELayerRuntimeView *>(runtime),
        gate_descs,
        up_descs,
        down_descs,
        num_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE runtime prefill descriptor materialize kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    return true;
}

bool hipMoE_prefill_gather_expert_runtime(
    const void *runtime,
    const float *hidden,
    float *batch_buffer,
    int expert_id, int max_tokens, int d_model,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!runtime || !hidden || !batch_buffer || expert_id < 0 || max_tokens <= 0 || d_model <= 0)
    {
        fprintf(stderr, "HIP MoE prefill gather runtime invalid arguments\n");
        return false;
    }

    const int threads = 256;
    const int total = max_tokens * d_model;
    const int blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(rocm_moe_prefill_gather_expert_runtime_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        static_cast<const DeviceMoELayerRuntimeView *>(runtime),
        hidden, batch_buffer, expert_id, max_tokens, d_model);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill gather runtime launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_prefill_scatter_expert_runtime(
    float *output,
    const float *expert_output,
    const void *runtime,
    int expert_id, int max_tokens, int d_model,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!output || !expert_output || !runtime || expert_id < 0 || max_tokens <= 0 || d_model <= 0)
    {
        fprintf(stderr, "HIP MoE prefill scatter runtime invalid arguments\n");
        return false;
    }

    const int threads = 256;
    const int total = max_tokens * d_model;
    const int blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(rocm_moe_prefill_scatter_expert_runtime_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        output, expert_output,
        static_cast<const DeviceMoELayerRuntimeView *>(runtime),
        expert_id, max_tokens, d_model);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill scatter runtime launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

} // extern "C" (Phase 3)

// =========================================================================
// Phase 4: Tensor-aware utility kernels
// =========================================================================

/// Convert int array to float array on device (for routing index tensors)
__global__ void rocm_moe_int_to_float_kernel(
    const int *__restrict__ input, float *__restrict__ output, int count)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < count)
        output[idx] = static_cast<float>(input[idx]);
}

/// Convert float array to int array on device (reverse: for GPU-side token grouping)
__global__ void rocm_moe_float_to_int_kernel(
    const float *__restrict__ input, int *__restrict__ output, int count)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < count)
        output[idx] = static_cast<int>(input[idx]);
}

__global__ void rocm_moe_float_to_masked_int_kernel(
    const float *__restrict__ input,
    int *__restrict__ output,
    const uint8_t *__restrict__ expert_mask,
    int count,
    int num_experts)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= count)
        return;
    const int expert = static_cast<int>(input[idx]);
    output[idx] =
        (expert >= 0 && expert < num_experts && expert_mask && expert_mask[expert] != 0u)
            ? expert
            : -1;
}

extern "C" {

bool hipMoE_stage_mutable_pointer_arrays(
    float **d_a,
    float **d_b,
    float *const *h_a,
    float *const *h_b,
    int count,
    int device_idx,
    void *stream)
{
    if (!d_a || !d_b || !h_a || !h_b || count <= 0 ||
        count > kMoEPointerStageMax || !stream)
    {
        fprintf(stderr, "HIP MoE mutable pointer staging invalid arguments: count=%d stream=%p\n", count, stream);
        return false;
    }

    MoEPointerStagePayload payload{};
    for (int i = 0; i < count; ++i)
    {
        if (!h_a[i] || !h_b[i])
        {
            fprintf(stderr, "HIP MoE mutable pointer staging null pointer at slot %d\n", i);
            return false;
        }
        payload.mutable_ptrs_a[i] = h_a[i];
        payload.mutable_ptrs_b[i] = h_b[i];
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();
    hipLaunchKernelGGL(moe_stage_mutable_pointer_arrays_kernel,
        dim3(1), dim3(kMoEPointerStageMax), 0, static_cast<hipStream_t>(stream),
        payload, d_a, d_b, count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE mutable pointer staging launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_stage_const_pointer_arrays(
    const float **d_a,
    const float **d_b,
    const float *const *h_a,
    const float *const *h_b,
    int count,
    int device_idx,
    void *stream)
{
    if (!d_a || !d_b || !h_a || !h_b || count <= 0 ||
        count > kMoEPointerStageMax || !stream)
    {
        fprintf(stderr, "HIP MoE const pointer staging invalid arguments: count=%d stream=%p\n", count, stream);
        return false;
    }

    MoEPointerStagePayload payload{};
    for (int i = 0; i < count; ++i)
    {
        if (!h_a[i] || !h_b[i])
        {
            fprintf(stderr, "HIP MoE const pointer staging null pointer at slot %d\n", i);
            return false;
        }
        payload.const_ptrs_a[i] = h_a[i];
        payload.const_ptrs_b[i] = h_b[i];
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();
    hipLaunchKernelGGL(moe_stage_const_pointer_arrays_kernel,
        dim3(1), dim3(kMoEPointerStageMax), 0, static_cast<hipStream_t>(stream),
        payload, d_a, d_b, count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE const pointer staging launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_int_to_float(
    const int *d_input, float *d_output, int count,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;

    hipLaunchKernelGGL(rocm_moe_int_to_float_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        d_input, d_output, count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE int_to_float kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_float_to_int(
    const float *d_input, int *d_output, int count,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;

    // Reuse int_to_float kernel pattern with reversed types —
    // a trivial __device__ lambda would be ideal but HIP doesn't support
    // device lambdas in extern "C".  Instead, add a dedicated kernel below.
    // For now, use a trivial inline kernel via hipLaunchKernelGGL.
    // The kernel is defined right after this bridge.
    hipLaunchKernelGGL(rocm_moe_float_to_int_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        d_input, d_output, count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE float_to_int kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_float_to_masked_int(
    const float *d_input,
    int *d_output,
    const uint8_t *d_expert_mask,
    int count,
    int num_experts,
    int device_idx,
    void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;

    hipLaunchKernelGGL(rocm_moe_float_to_masked_int_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        d_input, d_output, d_expert_mask, count, num_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE float_to_masked_int kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

} // extern "C" (Phase 4)
