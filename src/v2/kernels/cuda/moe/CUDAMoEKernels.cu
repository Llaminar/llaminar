/**
 * @file CUDAMoEKernels.cu
 * @brief CUDA launch bridges for MoE routing, grouping, and scatter/gather primitives.
 *
 * These kernels intentionally cover the non-GEMM MoE glue. Expert gate/up/down
 * projections continue to use the dedicated CUDA GEMM kernels. The launch
 * wrappers are C ABI functions consumed by `CUDAMoEKernel.cpp`, matching the
 * split used by the rest of the CUDA backend.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "kernels/cuda/gemm/CUDANativeVNNIDecodeCommon.cuh"
#include "execution/moe/DeviceMoERebalancePolicyShared.h"
#include "execution/moe/LeastLoadedExpertAssignment.h"
#include "utils/DebugEnv.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace
{
    constexpr int kThreads = 256;
    constexpr int kMaxExperts = 1024;
    constexpr int kDeviceMoEMaxExperts = 256;
    constexpr int kDeviceMoEMaxParticipants = 8;
    constexpr int kMaxTopK = 16;
    constexpr uint8_t kMixedCodebookSentinel = 0xffu;
    constexpr uint32_t kDeviceMoEFlagValid = 1u << 0;
    constexpr uint32_t kDeviceMoEFlagResident = 1u << 1;
    constexpr uint32_t kDeviceMoEFlagReplicated = 1u << 2;
    constexpr uint32_t kDeviceMoEFlagLocalCompute = 1u << 4;
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

    struct DeviceNativeVNNIMatrixDesc
    {
        const uint8_t *payload = nullptr;
        const void *scales = nullptr;
        const void *mins = nullptr;
        const void *emins = nullptr;
        int n = 0;
        int k = 0;
        uint32_t blocks_per_row = 0;
        uint8_t codebook_id = 0;
        uint8_t reserved[3] = {0, 0, 0};
    };

    struct DeviceMoEExpertDescriptorView
    {
        DeviceNativeVNNIMatrixDesc gate;
        DeviceNativeVNNIMatrixDesc up;
        DeviceNativeVNNIMatrixDesc down;
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
        int32_t topk_expert_ids[kMaxTopK];
        float topk_weights[kMaxTopK];
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
        uint32_t min_router_spread_improvement_per_payload_slot;
        uint32_t max_post_wave_load_spread_per_mille;
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
        uint64_t candidate_load_spread_improvement_total;
        uint64_t candidate_load_spread_improvement_max;
        uint64_t accepted_load_spread_improvement_total;
        uint64_t accepted_load_spread_improvement_max;
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
        uint32_t reserved[2];
    };

    __device__ __forceinline__ bool rebalance_config_ok(
        const DeviceMoERebalanceConfigView &config)
    {
        return config.magic == kDeviceMoERebalanceMagic &&
               config.version == kDeviceMoERebalanceVersion &&
               config.num_layers > 0u &&
               config.num_experts > 0u &&
               config.num_experts <= static_cast<uint32_t>(kDeviceMoEMaxExperts) &&
               config.top_k > 0u &&
               config.top_k <= static_cast<uint32_t>(kMaxTopK) &&
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
                   : min(config.layer_window_count, config.num_layers);
    }

    __device__ __forceinline__ uint32_t rebalance_layer_wave_count(
        const DeviceMoERebalanceConfigView &config)
    {
        const uint32_t window_count = rebalance_layer_window_count(config);
        return config.layer_wave_count == 0u
                   ? window_count
                   : min(config.layer_wave_count, window_count);
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

    __device__ __forceinline__ uint32_t runtime_expert_resident_mask(
        const DeviceMoELayerRuntimeView *runtime,
        const DeviceMoEPlacementBankView &bank,
        int expert_id)
    {
        const uint32_t participant_count = runtime->participant_count;
        uint32_t mask = bank.resident_participant_mask[expert_id] &
                        runtime_valid_participant_mask(participant_count);
        const int owner = runtime_expert_owner(bank, expert_id);
        if (owner >= 0 && owner < static_cast<int>(participant_count))
            mask |= runtime_participant_bit(owner);
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
        uint64_t candidate_load_spread_improvement_total = 0ULL;
        uint64_t candidate_load_spread_improvement_max = 0ULL;
        uint64_t accepted_load_spread_improvement_total = 0ULL;
        uint64_t accepted_load_spread_improvement_max = 0ULL;
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
                : min(config.layer_window_count, config.num_layers);
        const uint32_t layer_wave_count =
            config.layer_wave_count == 0u
                ? layer_window_count
                : min(config.layer_wave_count, layer_window_count);
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
         * Router-benefit counters measure the effect of the previously active
         * replica set.  Aggregate them over the full routed layer window before
         * planning the next layer wave; otherwise a rolling wave can miss its
         * own payback signal when the next planning slice has already advanced.
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
                if (desc.owner_participant >= 0 &&
                    desc.owner_participant < static_cast<int32_t>(config.participant_count))
                {
                    resident_mask |= runtime_participant_bit(desc.owner_participant);
                }
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
                    if (desc.owner_participant >= 0 &&
                        desc.owner_participant < static_cast<int32_t>(config.participant_count))
                    {
                        resident_mask |= runtime_participant_bit(desc.owner_participant);
                    }
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
                for (uint32_t expert = 0; expert < config.num_experts; ++expert)
                {
                    uint32_t resident_mask =
                        shared_post_policy_resident_mask[expert] & valid_mask;
                    const int32_t owner = shared_expert_owners[expert];
                    if (owner >= 0 && owner < static_cast<int32_t>(config.participant_count))
                    {
                        owner_participants[expert] = static_cast<uint32_t>(owner);
                        resident_mask |= runtime_participant_bit(owner);
                    }
                    else
                    {
                        const int fallback = rebalance_first_resident_participant(
                            resident_mask,
                            config.participant_count,
                            -1,
                            -1);
                        owner_participants[expert] =
                            fallback >= 0 ? static_cast<uint32_t>(fallback) : 0u;
                    }
                }

                llaminar2::least_loaded_ep::LeastLoadedExpertWeightTransfer
                    weight_transfers[kDeviceMoEMaxExperts];
                llaminar2::least_loaded_ep::LeastLoadedExpertAssignmentConfig llep_config{};
                llep_config.expert_count = config.num_experts;
                llep_config.participant_count = config.participant_count;
                llep_config.min_spread_improvement =
                    config.min_load_spread_improvement;
                llep_config.min_spread_improvement_divisor =
                    config.min_load_spread_improvement_divisor;
                llep_config.min_spread_improvement_per_transfer =
                    config.min_wave_spread_improvement_per_payload_slot;
                llep_config.enable_balanced_skip = true;

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
                        &llep_status);
                if (!planned_llep || llep_status.overflow != 0u)
                {
                    if (status)
                        ++status->plan_overflow;
                }
                candidate_arrivals_considered += llep_status.weight_transfer_count;
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
                            (shared_post_policy_resident_mask[transfer.expert] |
                             source_bit) &
                            valid_mask;
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

                        DeviceMoERebalancePlanEntryView entry{};
                        entry.op = kDeviceMoERebalancePlanExpertPayloadArrival;
                        entry.layer = layer;
                        entry.expert = transfer.expert;
                        entry.source_participant = transfer.source_participant;
                        entry.destination_participant = transfer.destination_participant;
                        entry.source_resident_mask = resident_mask;
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
                        resident_mask = (resident_mask | destination_bit) & valid_mask;
                        shared_post_policy_resident_mask[transfer.expert] =
                            resident_mask;
                        next.resident_participant_mask[transfer.expert] =
                            resident_mask;
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
                        const int32_t owner = shared_expert_owners[expert];
                        if (owner >= 0 && owner < static_cast<int32_t>(config.participant_count))
                            resident_mask |= runtime_participant_bit(owner);
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
                    ++dynamic_ownership_swap_accepts;

                    const uint32_t heavy_source = swap_choice.overloaded_participant;
                    const uint32_t heavy_destination = swap_choice.underloaded_participant;
                    const uint32_t light_source = swap_choice.underloaded_participant;
                    const uint32_t light_destination = swap_choice.overloaded_participant;
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

                    DeviceMoERebalancePlanEntryView heavy_entry{};
                    heavy_entry.op = kDeviceMoERebalancePlanOwnershipTransfer;
                    heavy_entry.layer = layer;
                    heavy_entry.expert = swap_choice.heavy_expert;
                    heavy_entry.source_participant = heavy_source;
                    heavy_entry.destination_participant = heavy_destination;
                    heavy_entry.source_resident_mask =
                        runtime_participant_bit(static_cast<int>(heavy_source));
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
                    light_entry.source_resident_mask =
                        runtime_participant_bit(static_cast<int>(light_source));
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
                        if (candidate_desc.owner_participant >= 0 &&
                            candidate_desc.owner_participant < static_cast<int32_t>(config.participant_count))
                        {
                            candidate_resident_mask |=
                                runtime_participant_bit(candidate_desc.owner_participant);
                        }
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
                                                ? static_cast<int>(destination_choice.source_participant)
                                                : rebalance_first_resident_participant(
                                                      candidate_resident_mask,
                                                      config.participant_count,
                                                      candidate_desc.owner_participant,
                                                      static_cast<int>(destination_choice.destination_participant));
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
                                        if (destination_choice.valid &&
                                            destination_has_transfer_slot &&
                                            source_has_payload_slot)
                                        {
                                            candidate_value = destination_choice.delta.improvement;
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
                    if (desc.owner_participant >= 0 &&
                        desc.owner_participant < static_cast<int32_t>(config.participant_count))
                    {
                        resident_mask |= runtime_participant_bit(desc.owner_participant);
                    }
                    next.resident_participant_mask[best_expert] = resident_mask;

                    const bool replicated = (resident_mask & (resident_mask - 1u)) != 0u;
                    const bool local_resident = (resident_mask & participant_bit) != 0u;
                    const bool owner_local =
                        desc.owner_participant == static_cast<int32_t>(config.participant_id);
                    if (domain_root_planning)
                    {
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
                            if (!llaminar2::moe_rebalance_policy::addingResidentImprovesDynamicSpread(
                                    shared_current_policy_load,
                                    count,
                                    resident_mask,
                                    destination_choice.proposed_resident_mask,
                                    config.participant_count,
                                    destination_choice.source_participant,
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
                                const int source_participant =
                                    static_cast<int>(destination_choice.source_participant);
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
                        const unsigned long long count =
                            rebalance_global_count(gathered_histograms, config, window_index, best_expert);
                        const int source_participant = rebalance_first_resident_participant(
                            resident_mask,
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
                    if (command_count > 0u)
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
                 !llaminar2::moe_rebalance_policy::transferWaveMeetsPostLoadSpreadCeiling(
                     post_wave_load_spread,
                     post_wave_load_total,
                     requested_payload_slots,
                     config.max_post_wave_load_spread_per_mille));
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
            if (command_header)
            {
                command_header->epoch = defer_runtime_apply && command_count > 0u
                                            ? shared_plan_epoch
                                            : last_epoch;
                command_header->command_count = command_count;
            }
            if (wave_state)
            {
                const uint32_t start_layer =
                    (layer_window_start + start_offset) % config.num_layers;
                wave_state->epoch = last_epoch;
                wave_state->planned_start_layer = start_layer;
                wave_state->planned_layer_count = layer_wave_count;
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
                    wave_state ? wave_state->planned_start_layer : layer_window_start;
                wave.planned_layer_count =
                    wave_state ? wave_state->planned_layer_count : layer_wave_count;
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
        const DeviceNativeVNNIMatrixDesc &desc)
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
        const DeviceNativeVNNIMatrixDesc &desc)
    {
        return static_cast<unsigned long long>(desc.blocks_per_row) *
               static_cast<unsigned long long>(desc.n);
    }

    __device__ __forceinline__ bool rebalance_projection_format(
        const DeviceNativeVNNIMatrixDesc &desc,
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
        const DeviceNativeVNNIMatrixDesc &desc,
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
        entry.epoch = runtime.active_epoch;
        entry.generation = runtime.active_epoch;
        entry.resident_mask = resident_mask;
        if (desc.local_slot >= 0)
            entry.slot_index = static_cast<uint32_t>(desc.local_slot);

        if (local_resident && rebalance_expert_desc_ready(desc))
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

    __device__ __forceinline__ bool rebalance_wave_covers_layer(
        const DeviceMoERebalanceWaveProgressView &wave,
        const DeviceMoERebalanceConfigView &config,
        uint32_t layer)
    {
        if (layer >= config.num_layers || wave.planned_layer_count == 0u)
            return false;
        const uint32_t count = min(wave.planned_layer_count, config.num_layers);
        const uint32_t start = config.num_layers == 0u ? 0u : wave.planned_start_layer % config.num_layers;
        for (uint32_t i = 0; i < count; ++i)
        {
            if (((start + i) % config.num_layers) == layer)
                return true;
        }
        return false;
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
            if (target_layer >= 0 &&
                !rebalance_wave_covers_layer(wave, config, static_cast<uint32_t>(target_layer)))
            {
                continue;
            }
            selection.wave = &wave;
            selection.wave_index = i;
            return selection;
        }
        return selection;
    }

    __global__ void publish_rebalance_transfer_complete_kernel(
        DeviceMoERebalanceGraphControllerStateView *state,
        const DeviceMoERebalanceCommandBufferHeaderView *command_header,
        const DeviceMoERebalanceWaveStateView *wave_state,
        const DeviceMoERebalanceApplyStatusView *copy_status,
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
            !rebalance_command_header_ok(command_header, config))
        {
            state->last_error_code = 3u;
            return;
        }
        const uint32_t command_count =
            min(command_header->command_count, command_header->command_capacity);
        if (command_header->epoch == 0u || command_count == 0u)
        {
            if (wave_index < state->wave_count)
            {
                reset_rebalance_wave_progress_device(state->waves[wave_index]);
            }
            return;
        }

        auto &wave = state->waves[wave_index];
        const uint32_t requested_payload_slots = wave.requested_payload_slots;
        const uint32_t payload_bucket_slots = wave.payload_bucket_slots;
        const uint32_t payload_bucket_index = wave.payload_bucket_index;
        const uint32_t payload_bucket_overflow = wave.payload_bucket_overflow;
        if (wave.state == kDeviceMoERebalanceLifecycleReadyToApply &&
            wave.epoch != command_header->epoch)
        {
            state->last_error_code = 5u;
            wave.error_code = 5u;
            wave.state = kDeviceMoERebalanceLifecycleError;
            return;
        }

        wave.magic = kDeviceMoERebalanceMagic;
        wave.version = kDeviceMoERebalanceVersion;
        wave.epoch = command_header->epoch;
        wave.state = kDeviceMoERebalanceLifecycleTransferInFlight;
        wave.planned_start_layer = wave_state->planned_start_layer;
        wave.planned_layer_count = wave_state->planned_layer_count;
        wave.command_count = command_count;
        wave.copied_arrivals = copy_status->copied_arrivals;
        wave.applied_arrivals = 0u;
        wave.applied_layer_count = 0u;
        wave.error_code = 0u;
        wave.requested_payload_slots = requested_payload_slots;
        wave.payload_bucket_slots = payload_bucket_slots;
        wave.payload_bucket_index = payload_bucket_index;
        wave.payload_bucket_overflow = payload_bucket_overflow;
        __threadfence();
        wave.state = kDeviceMoERebalanceLifecycleReadyToApply;
        state->next_epoch = max(state->next_epoch, command_header->epoch + 1u);
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
            return min(command_header->command_count,
                       min(command_header->command_capacity, plan_capacity));
        return plan_count ? min(*plan_count, plan_capacity) : 0u;
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
        uint32_t command_buffer_count)
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
                projected.command_count =
                    min(root_header.command_count,
                        min(root_header.command_capacity, plan_capacity));
            }
            local_command_headers[buffer_index] = projected;
        }

        const unsigned long long total_entries =
            static_cast<unsigned long long>(metadata_buffer_count) *
            static_cast<unsigned long long>(plan_capacity);
        for (unsigned long long idx = static_cast<unsigned long long>(threadIdx.x);
             idx < total_entries;
             idx += static_cast<unsigned long long>(blockDim.x))
        {
            const uint32_t buffer_index =
                static_cast<uint32_t>(idx / static_cast<unsigned long long>(plan_capacity));
            const uint32_t plan_index =
                static_cast<uint32_t>(idx % static_cast<unsigned long long>(plan_capacity));
            const auto &root_header =
                gathered_command_headers[static_cast<unsigned long long>(root_participant) *
                                             static_cast<unsigned long long>(metadata_buffer_count) +
                                         static_cast<unsigned long long>(buffer_index)];
            DeviceMoERebalancePlanEntryView entry{};
            if (rebalance_command_header_for_participant_ok(
                    root_header,
                    root_participant,
                    config))
            {
                const uint32_t command_count =
                    min(root_header.command_count,
                        min(root_header.command_capacity, plan_capacity));
                if (plan_index < command_count)
                {
                    const unsigned long long root_plan_index =
                        (static_cast<unsigned long long>(root_participant) *
                             static_cast<unsigned long long>(metadata_buffer_count) +
                         static_cast<unsigned long long>(buffer_index)) *
                            static_cast<unsigned long long>(plan_capacity) +
                        static_cast<unsigned long long>(plan_index);
                    entry = gathered_plan_entries[root_plan_index];
                }
            }
            local_plan_entries[idx] = entry;
        }

        if (threadIdx.x == 0 && status)
        {
            uint32_t total_command_count = 0u;
            uint32_t last_projected_epoch = 0u;
            uint32_t requested_by_source[kDeviceMoEMaxParticipants] = {};
            uint32_t payload_source_participant_mask = 0u;
            uint32_t payload_destination_participant_mask = 0u;
            uint64_t payload_edge_mask = 0ULL;
            for (uint32_t buffer_index = 0; buffer_index < metadata_buffer_count; ++buffer_index)
            {
                const auto &root_header =
                    gathered_command_headers[static_cast<unsigned long long>(root_participant) *
                                                 static_cast<unsigned long long>(metadata_buffer_count) +
                                             static_cast<unsigned long long>(buffer_index)];
                if (rebalance_command_header_for_participant_ok(
                        root_header,
                        root_participant,
                        config))
                {
                    const uint32_t command_count =
                        min(root_header.command_count,
                            min(root_header.command_capacity, plan_capacity));
                    total_command_count += command_count;
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
                }
            }

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
                status->windows_observed = max(status->windows_observed, 1u);
                status->planned_arrivals = total_command_count;
                if (total_command_count != 0u)
                {
                    status->windows_applied = max(status->windows_applied, 1u);
                    if (status->last_epoch == 0u)
                        status->last_epoch = last_projected_epoch;
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
            const uint32_t command_count =
                min(header.command_count, min(header.command_capacity, plan_capacity));
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
        for (unsigned long long i = threadIdx.x; i < bytes; i += blockDim.x)
            dst_bytes[i] = src_bytes[i];
    }

    __device__ __forceinline__ unsigned long long rebalance_projection_payload_bytes(
        const DeviceNativeVNNIMatrixDesc &desc,
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
        const DeviceNativeVNNIMatrixDesc &src,
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
        const DeviceNativeVNNIMatrixDesc &dst,
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

    __device__ void rebalance_zero_payload_slot(uint8_t *slot, unsigned long long bytes)
    {
        for (unsigned long long i = threadIdx.x; i < bytes; i += blockDim.x)
            slot[i] = 0u;
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
        rebalance_zero_payload_slot(payload_slot, payload_slot_bytes);

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
        const uint32_t count = min(header.command_count,
                                   min(header.command_capacity, plan_capacity));
        if (plan_index >= count)
            return;

        const auto &plan =
            gathered_plan_entries[(metadata_participant_base + active_wave) *
                                      static_cast<unsigned long long>(plan_capacity) +
                                  static_cast<unsigned long long>(plan_index)];
        if (plan.op == 0u ||
            !rebalance_plan_requires_payload(plan.op) ||
            plan.destination_participant != destination_participant ||
            plan.source_participant != config.participant_id ||
            plan.layer >= config.num_layers ||
            plan.expert >= config.num_experts)
        {
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
        rebalance_zero_payload_slot(payload_slot, payload_slot_bytes);

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
            if (rebalance_plan_requires_payload(plan.op) &&
                plan.source_participant == config.participant_id &&
                plan.destination_participant < config.participant_count &&
                plan.payload_slot == global_slot &&
                plan.layer < config.num_layers &&
                plan.expert < config.num_experts)
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
        if (!rebalance_source_entry_ready(src, config.participant_id, src.layer, src.expert))
            return;

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

        const uint32_t plan_index = blockIdx.x;
        const uint32_t count = rebalance_command_count(plan_count, plan_capacity, command_header);
        if (plan_index >= count)
            return;
        const auto &plan = plan_entries[plan_index];
        if (threadIdx.x == 0)
            atomicAdd(&status->plan_entries_seen, 1u);

        if (plan.op == 0u)
            return;
        if (!rebalance_plan_requires_payload(plan.op) ||
            plan.layer >= config.num_layers ||
            plan.expert >= config.num_experts ||
            plan.source_participant >= config.participant_count)
        {
            if (threadIdx.x == 0)
                atomicAdd(&status->invalid_plan_entries, 1u);
            return;
        }
        if (plan.destination_participant != config.participant_id)
        {
            if (threadIdx.x == 0)
                atomicAdd(&status->skipped_wrong_destination, 1u);
            return;
        }
        if (plan.destination_slot >= local_transfer_slot_count)
        {
            if (threadIdx.x == 0)
                atomicAdd(&status->missing_destination_slots, 1u);
            return;
        }
        if (plan.payload_slot >= local_payload_slot_count)
        {
            if (threadIdx.x == 0)
                atomicAdd(&status->invalid_plan_entries, 1u);
            return;
        }

        const unsigned long long gathered_slot =
            (static_cast<unsigned long long>(plan.source_participant) *
                 static_cast<unsigned long long>(local_payload_slot_count) +
             static_cast<unsigned long long>(plan.payload_slot));
        const uint8_t *payload_slot = gathered_payload + gathered_slot * payload_slot_bytes;
        const auto &src =
            *reinterpret_cast<const DeviceMoEExpertDirectoryEntryView *>(payload_slot);
        const unsigned long long payload_data_bytes =
            payload_slot_bytes - sizeof(DeviceMoEExpertDirectoryEntryView);
        auto &dst = local_transfer_slots[plan.destination_slot];
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

        if (threadIdx.x == 0)
        {
            dst.layer = plan.layer;
            dst.expert = plan.expert;
            dst.descriptor.logical_expert_id = static_cast<int32_t>(plan.expert);
            dst.descriptor.owner_participant = src.descriptor.owner_participant;
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
        __shared__ uint32_t reduce_scratch[kThreads];
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
                init_rebalance_apply_status_device(status);

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
                if (base_desc.owner_participant >= 0 &&
                    base_desc.owner_participant < static_cast<int32_t>(config.participant_count))
                {
                    resident_mask |= runtime_participant_bit(base_desc.owner_participant);
                }
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
                    desc.flags &= ~kDeviceMoEFlagReplicated;
                else
                    desc.flags |= kDeviceMoEFlagReplicated;
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
                runtime.active_bank = inactive_bank;
                runtime.active_epoch = next.epoch;
                ++status->changed_layers;
                status->post_apply_multi_resident_experts += reduce_scratch[0];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0)
        {
            status->status_code = 0u;
            if (shared_ready_wave)
            {
                shared_ready_wave->applied_arrivals += status->applied_arrivals;
                const uint32_t layer_increment = target_layer >= 0 ? 1u : shared_ready_wave->planned_layer_count;
                shared_ready_wave->applied_layer_count =
                    min(shared_ready_wave->planned_layer_count,
                        shared_ready_wave->applied_layer_count + layer_increment);
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
                __threadfence();
            }
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
            init_rebalance_apply_status_device(status);
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
        uint32_t valid_plan_count = 0u;
        const uint32_t participant_bit =
            runtime_participant_bit(static_cast<int>(config.participant_id));
        const uint32_t valid_mask = runtime_valid_participant_mask(config.participant_count);

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
            ++valid_plan_count;
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

        const uint32_t inactive_bank = 1u - runtime.active_bank;
        const auto &active = runtime.banks[runtime.active_bank];
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
            if (base_desc.owner_participant >= 0 &&
                base_desc.owner_participant < static_cast<int32_t>(config.participant_count))
            {
                resident_mask |= runtime_participant_bit(base_desc.owner_participant);
            }
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
                desc.flags &= ~kDeviceMoEFlagReplicated;
            else
                desc.flags |= kDeviceMoEFlagReplicated;
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

    __device__ __forceinline__ bool grouped_desc_supports_codebook(uint8_t codebook_id)
    {
        switch (codebook_id)
        {
        case 0:
        case 4:
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
        case 10:
        case 11:
        case 12:
        case 13:
        case 14:
        case 15:
        case 16:
        case 17:
        case 19:
            return true;
        default:
            return false;
        }
    }

    __device__ __forceinline__ bool grouped_desc_requires_mins(uint8_t codebook_id)
    {
        switch (codebook_id)
        {
        case 5:
        case 7:
        case 8:
        case 9:
        case 10:
        case 13:
        case 14:
        case 16:
        case 17:
            return true;
        default:
            return false;
        }
    }

    __device__ __forceinline__ bool grouped_desc_requires_emins(uint8_t codebook_id)
    {
        return codebook_id == 10;
    }

    __device__ __forceinline__ bool native_vnni_desc_shape_ok_dynamic(
        const DeviceNativeVNNIMatrixDesc &desc,
        int N,
        int K)
    {
        return desc.payload &&
               desc.scales &&
               desc.n == N &&
               desc.k == K &&
               desc.blocks_per_row == static_cast<uint32_t>(K / 32) &&
               grouped_desc_supports_codebook(desc.codebook_id) &&
               (!grouped_desc_requires_mins(desc.codebook_id) || desc.mins) &&
               (!grouped_desc_requires_emins(desc.codebook_id) || desc.emins);
    }

    template <uint8_t CodebookId>
    __device__ __forceinline__ bool native_vnni_desc_shape_ok(
        const DeviceNativeVNNIMatrixDesc &desc,
        int N,
        int K)
    {
        if constexpr (CodebookId == kMixedCodebookSentinel)
        {
            return native_vnni_desc_shape_ok_dynamic(desc, N, K);
        }
        else
        {
            return desc.payload &&
                   desc.scales &&
                   desc.n == N &&
                   desc.k == K &&
                   desc.blocks_per_row == static_cast<uint32_t>(K / 32) &&
                   desc.codebook_id == CodebookId &&
                   (!(llaminar2::cuda_native_vnni::CodebookTraits<CodebookId>::is_asymmetric ||
                      llaminar2::cuda_native_vnni::CodebookTraits<CodebookId>::is_dual_scale) ||
                    desc.mins) &&
                   (!llaminar2::cuda_native_vnni::CodebookTraits<CodebookId>::is_dual_scale_asym ||
                    desc.emins);
        }
    }

    __device__ __forceinline__ float silu(float x)
    {
        return x / (1.0f + expf(-x));
    }

    bool finishLaunch(const char *name)
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::fprintf(stderr, "%s launch failed: %s\n", name, cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    // Router GEMV: one block per (expert, token) computes logit = dot(hidden_token, gate_expert).
    //
    // Optimized memory-bound dot product. The previous implementation used scalar
    // loads (one element/thread/iter) followed by a full log2(blockDim)-step shared
    // memory tree reduction, leaving it ~15x off the memory-bound limit. This version:
    //   1. Loads gate/hidden as coalesced float4 vectors (4x fewer load instructions).
    //   2. Reduces within each warp via __shfl_down_sync (no __syncthreads, no shared
    //      traffic for the intra-warp reduction).
    //   3. Combines the (few) per-warp partials with a single short shared-memory step.
    // The block-per-expert grid is preserved so decode (num_experts blocks) keeps full
    // SM coverage.
    __global__ void route_logits_kernel(
        const float *__restrict__ hidden,
        const float *__restrict__ gate_weights,
        float *__restrict__ logits,
        int seq_len, int d_model, int num_experts)
    {
        const int expert = blockIdx.x;
        const int token = blockIdx.y;
        if (expert >= num_experts || token >= seq_len)
            return;

        const float *h = hidden + static_cast<size_t>(token) * d_model;
        const float *g = gate_weights + static_cast<size_t>(expert) * d_model;

        float sum = 0.0f;
        const bool can_vectorize =
            ((reinterpret_cast<std::uintptr_t>(h) |
              reinterpret_cast<std::uintptr_t>(g)) &
             0x0fu) == 0u;
        if (can_vectorize)
        {
            // Vectorized main loop: consecutive threads read consecutive float4 chunks
            // (fully coalesced). The row width is a multiple of 4 for supported
            // models, but graph/arena suballocations can still shift the base pointer.
            // Only use float4 loads when both row pointers are actually 16B-aligned.
            const int vec4 = d_model >> 2;
            for (int v = threadIdx.x; v < vec4; v += blockDim.x)
            {
                const float4 hv = reinterpret_cast<const float4 *>(h)[v];
                const float4 gv = reinterpret_cast<const float4 *>(g)[v];
                sum += hv.x * gv.x + hv.y * gv.y + hv.z * gv.z + hv.w * gv.w;
            }
            // Scalar tail for any d_model not divisible by 4.
            for (int j = (vec4 << 2) + threadIdx.x; j < d_model; j += blockDim.x)
                sum += h[j] * g[j];
        }
        else
        {
            for (int j = threadIdx.x; j < d_model; j += blockDim.x)
                sum += h[j] * g[j];
        }

        // Intra-warp reduction via shuffle (no shared memory, no barriers).
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);

        // Combine per-warp partials. blockDim is a multiple of 32 and <= 1024, so at
        // most 32 warps; warp 0 reduces the per-warp sums in a single shuffle pass.
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int num_warps = blockDim.x >> 5;
        __shared__ float warp_sums[32];
        if (lane == 0)
            warp_sums[warp] = sum;
        __syncthreads();
        if (warp == 0)
        {
            float v = (lane < num_warps) ? warp_sums[lane] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1)
                v += __shfl_down_sync(0xffffffffu, v, offset);
            if (lane == 0)
                logits[static_cast<size_t>(token) * num_experts + expert] = v;
        }
    }

    __global__ void route_logits_bf16_kernel(
        const float *__restrict__ hidden,
        const __nv_bfloat16 *__restrict__ gate_weights,
        float *__restrict__ logits,
        int seq_len, int d_model, int num_experts)
    {
        const int expert = blockIdx.x;
        const int token = blockIdx.y;
        if (expert >= num_experts || token >= seq_len)
            return;

        const float *h = hidden + static_cast<size_t>(token) * d_model;
        const __nv_bfloat16 *g = gate_weights + static_cast<size_t>(expert) * d_model;

        float sum = 0.0f;
        for (int j = threadIdx.x; j < d_model; j += blockDim.x)
            sum += h[j] * __bfloat162float(g[j]);

        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);

        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int num_warps = blockDim.x >> 5;
        __shared__ float warp_sums[32];
        if (lane == 0)
            warp_sums[warp] = sum;
        __syncthreads();
        if (warp == 0)
        {
            float v = (lane < num_warps) ? warp_sums[lane] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1)
                v += __shfl_down_sync(0xffffffffu, v, offset);
            if (lane == 0)
                logits[static_cast<size_t>(token) * num_experts + expert] = v;
        }
    }

    // Tiled SGEMM for the PREFILL router logits: logits[S, E] = hidden[S, D] · gateᵀ[D, E].
    //
    // The block-per-(expert,token) route_logits_kernel above has zero data reuse: it
    // re-reads hidden[token] once per expert (E× redundant) and gate[expert] once per
    // token (S× redundant), saturating L2 (~85% L2 throughput, ~1% DRAM, ~0.7 TFLOP/s).
    // This classic shared-memory tiled GEMM stages BM×BK hidden and BN×BK gate tiles
    // into smem and reuses each loaded element across the BN (resp. BM) tile dimension,
    // cutting L2 traffic by ~min(BM,BN)× and converting the kernel from L2-bound to
    // compute-bound.
    //
    // Tile geometry: BM=64 tokens × BN=64 experts per block, BK=16 along d_model.
    // Block = 16×16 = 256 threads, each thread computes a TM×TN = 4×4 output micro-tile.
    // Decode (seq_len == 1) still uses the warp-reduction kernel above for SM coverage.
    template <int BM, int BN, int BK, int TM, int TN>
    __global__ void route_logits_tiled_kernel(
        const float *__restrict__ hidden,       // A: [seq_len, d_model] row-major
        const float *__restrict__ gate_weights, // B: [num_experts, d_model] row-major
        float *__restrict__ logits,             // C: [seq_len, num_experts] row-major
        int seq_len, int d_model, int num_experts)
    {
        // Shared tiles, stored K-major so the compute loop reads a full column of
        // BM (resp. BN) values contiguously for each k step.
        __shared__ float As[BK * BM]; // As[k * BM + m] = hidden[blockM + m, kk + k]
        __shared__ float Bs[BK * BN]; // Bs[k * BN + n] = gate  [blockN + n, kk + k]

        // Origin of this block's output tile in (token, expert) space.
        const int blockM = blockIdx.y * BM;
        const int blockN = blockIdx.x * BN;

        // 16×16 thread grid; each thread owns a TM×TN micro-tile of the output.
        const int threadRow = threadIdx.x / (BN / TN); // 0..15
        const int threadCol = threadIdx.x % (BN / TN); // 0..15

        // Per-thread accumulators for the TM×TN micro-tile (lives in registers).
        float acc[TM][TN];
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
            for (int j = 0; j < TN; ++j)
                acc[i][j] = 0.0f;

        // March across the K (d_model) dimension one BK-wide strip at a time.
        for (int kk = 0; kk < d_model; kk += BK)
        {
            // Cooperative load of the hidden tile into smem (K-major). Consecutive
            // threads read consecutive k → fully coalesced within each hidden row.
            for (int idx = threadIdx.x; idx < BM * BK; idx += blockDim.x)
            {
                const int m = idx / BK;
                const int k = idx % BK;
                const int gm = blockM + m;
                const int gk = kk + k;
                As[k * BM + m] = (gm < seq_len && gk < d_model)
                                     ? hidden[static_cast<size_t>(gm) * d_model + gk]
                                     : 0.0f;
            }
            // Cooperative load of the gate tile into smem (K-major), same coalescing.
            for (int idx = threadIdx.x; idx < BN * BK; idx += blockDim.x)
            {
                const int n = idx / BK;
                const int k = idx % BK;
                const int gn = blockN + n;
                const int gk = kk + k;
                Bs[k * BN + n] = (gn < num_experts && gk < d_model)
                                     ? gate_weights[static_cast<size_t>(gn) * d_model + gk]
                                     : 0.0f;
            }
            __syncthreads();

            // Multiply the staged strip: for each k, broadcast TM hidden values and
            // TN gate values from smem into registers and accumulate the outer product.
#pragma unroll
            for (int k = 0; k < BK; ++k)
            {
                float regM[TM];
                float regN[TN];
#pragma unroll
                for (int i = 0; i < TM; ++i)
                    regM[i] = As[k * BM + threadRow * TM + i];
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    regN[j] = Bs[k * BN + threadCol * TN + j];
#pragma unroll
                for (int i = 0; i < TM; ++i)
#pragma unroll
                    for (int j = 0; j < TN; ++j)
                        acc[i][j] += regM[i] * regN[j];
            }
            __syncthreads();
        }

        // Write the micro-tile back to global memory, guarding the ragged token edge
        // (num_experts is a multiple of BN for all supported models, but guard anyway).
#pragma unroll
        for (int i = 0; i < TM; ++i)
        {
            const int m = blockM + threadRow * TM + i;
            if (m >= seq_len)
                continue;
#pragma unroll
            for (int j = 0; j < TN; ++j)
            {
                const int n = blockN + threadCol * TN + j;
                if (n < num_experts)
                    logits[static_cast<size_t>(m) * num_experts + n] = acc[i][j];
            }
        }
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

    __device__ void moe_select_topk_probabilities_block(
        float *__restrict__ values,
        int num_experts,
        int top_k,
        int *__restrict__ selected,
        float *__restrict__ selected_weights,
        float *__restrict__ reductions,
        int *__restrict__ reduction_indices)
    {
        for (int k = 0; k < top_k; ++k)
        {
            float best_value = -1.0f;
            int best = kMaxExperts;
            for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
            {
                const float value = values[expert];
                if (moe_topk_pair_better(value, expert, best_value, best))
                {
                    best_value = value;
                    best = expert;
                }
            }
            reductions[threadIdx.x] = best_value;
            reduction_indices[threadIdx.x] = best;
            __syncthreads();

            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
            {
                if (threadIdx.x < stride)
                {
                    const float other_value = reductions[threadIdx.x + stride];
                    const int other_idx = reduction_indices[threadIdx.x + stride];
                    const float current_value = reductions[threadIdx.x];
                    const int current_idx = reduction_indices[threadIdx.x];
                    if (moe_topk_pair_better(other_value, other_idx, current_value, current_idx))
                    {
                        reductions[threadIdx.x] = other_value;
                        reduction_indices[threadIdx.x] = other_idx;
                    }
                }
                __syncthreads();
            }

            if (threadIdx.x == 0)
            {
                const int winner = reduction_indices[0];
                selected[k] = winner;
                selected_weights[k] = reductions[0];
                if (winner >= 0 && winner < num_experts)
                    values[winner] = -1.0f;
            }
            __syncthreads();
        }
    }

    __global__ void softmax_topk_kernel(
        float *__restrict__ logits,
        int *__restrict__ expert_indices,
        float *__restrict__ expert_weights,
        int seq_len, int num_experts, int top_k,
        bool normalize_weights,
        const int *__restrict__ effective_seq_len_ptr)
    {
        const int token = blockIdx.x;
        if (token >= seq_len)
            return;

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
            for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
                logits[static_cast<size_t>(token) * num_experts + expert] = 0.0f;
            return;
        }

        __shared__ float values[kMaxExperts];
        __shared__ float reductions[kThreads];
        __shared__ int red_idx[kThreads];
        __shared__ int selected[kMaxTopK];
        __shared__ float selected_weights[kMaxTopK];

        const size_t row_offset = static_cast<size_t>(token) * num_experts;
        float local_max = -INFINITY;
        for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
            local_max = fmaxf(local_max, logits[row_offset + expert]);
        reductions[threadIdx.x] = local_max;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
                reductions[threadIdx.x] = fmaxf(reductions[threadIdx.x], reductions[threadIdx.x + stride]);
            __syncthreads();
        }
        const float max_value = reductions[0];

        float local_sum = 0.0f;
        for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
        {
            const float prob = expf(logits[row_offset + expert] - max_value);
            values[expert] = prob;
            local_sum += prob;
        }
        reductions[threadIdx.x] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
                reductions[threadIdx.x] += reductions[threadIdx.x + stride];
            __syncthreads();
        }
        const float denom = reductions[0];
        for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
        {
            const float prob = denom > 0.0f ? values[expert] / denom : 0.0f;
            values[expert] = prob;
            logits[row_offset + expert] = prob;
        }
        __syncthreads();

        // Parallel top-k selection. The original implementation ran the entire
        // top_k * num_experts argmax scan on thread 0 with the other 255 lanes idle.
        // Here each selection round does a block-wide parallel argmax reduction over
        // the (still-unselected) experts, masks the winner, and repeats. Ties break
        // toward the lower expert index to match the original serial scan exactly.
        float topk_sum = 0.0f;
        for (int k = 0; k < top_k; ++k)
        {
            // Each thread finds the best (value,index) over its strided expert subset.
            float best_value = -1.0f;
            int best = 0;
            for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
            {
                const float value = values[expert];
                if (value > best_value)
                {
                    best_value = value;
                    best = expert;
                }
            }
            reductions[threadIdx.x] = best_value;
            red_idx[threadIdx.x] = best;
            __syncthreads();

            // Tree reduction to the global argmax for this round.
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
            {
                if (threadIdx.x < stride)
                {
                    const float other = reductions[threadIdx.x + stride];
                    const int other_idx = red_idx[threadIdx.x + stride];
                    const float cur = reductions[threadIdx.x];
                    const int cur_idx = red_idx[threadIdx.x];
                    if (other > cur || (other == cur && other_idx < cur_idx))
                    {
                        reductions[threadIdx.x] = other;
                        red_idx[threadIdx.x] = other_idx;
                    }
                }
                __syncthreads();
            }

            const int winner = red_idx[0];
            const float winner_value = reductions[0];
            topk_sum += winner_value;
            if (threadIdx.x == 0)
            {
                selected[k] = winner;
                selected_weights[k] = winner_value;
                values[winner] = -1.0f; // mask out for the next round
            }
            __syncthreads(); // ensure the mask is visible before the next round
        }

        // Thread 0 writes the final indices + (optionally normalized) weights.
        if (threadIdx.x == 0)
        {
            for (int k = 0; k < top_k; ++k)
            {
                const size_t out = static_cast<size_t>(token) * top_k + k;
                expert_indices[out] = selected[k];
                expert_weights[out] = normalize_weights && topk_sum > 0.0f
                                          ? selected_weights[k] / topk_sum
                                          : selected_weights[k];
            }
        }
    }

    __global__ void softmax_topk_decode_runtime_kernel(
        float *__restrict__ logits,
        DeviceMoELayerRuntimeView *__restrict__ runtime,
        DeviceMoELayerRuntimeView *__restrict__ runtime_layers,
        float *legacy_indices,
        float *legacy_weights,
        int num_experts, int top_k,
        bool normalize_weights,
        bool write_legacy_outputs,
        bool update_runtime_histogram,
        const DeviceMoERebalancePlanEntryView *rebalance_plan_entries,
        uint32_t rebalance_plan_capacity,
        DeviceMoERebalanceCommandBufferHeaderView *rebalance_command_header,
        const DeviceMoEExpertDirectoryEntryView *rebalance_local_transfer_slots,
        uint32_t rebalance_local_transfer_slot_count,
        DeviceMoERebalanceConfigView rebalance_config,
        DeviceMoERebalanceApplyStatusView *rebalance_apply_status,
        DeviceMoERebalanceGraphControllerStateView *rebalance_controller_state,
        int rebalance_target_layer,
        uint32_t rebalance_command_buffer_count)
    {
        __shared__ float values[kMaxExperts];
        __shared__ float reductions[kThreads];
        __shared__ int red_idx[kThreads];
        __shared__ int selected[kMaxTopK];
        __shared__ float selected_weights[kMaxTopK];

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

        float local_max = -INFINITY;
        for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
            local_max = fmaxf(local_max, logits[expert]);
        reductions[threadIdx.x] = local_max;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
                reductions[threadIdx.x] = fmaxf(reductions[threadIdx.x], reductions[threadIdx.x + stride]);
            __syncthreads();
        }
        const float max_value = reductions[0];

        float local_sum = 0.0f;
        for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
        {
            const float prob = expf(logits[expert] - max_value);
            values[expert] = prob;
            local_sum += prob;
        }
        reductions[threadIdx.x] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
                reductions[threadIdx.x] += reductions[threadIdx.x + stride];
            __syncthreads();
        }
        const float denom = reductions[0];
        for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
        {
            const float prob = denom > 0.0f ? values[expert] / denom : 0.0f;
            values[expert] = prob;
            logits[expert] = prob;
        }
        __syncthreads();

        moe_select_topk_probabilities_block(
            values, num_experts, top_k, selected, selected_weights, reductions, red_idx);

        if (threadIdx.x == 0)
        {
            const bool shape_ok = runtime_shape_ok(runtime, num_experts, top_k);
            float topk_sum = 0.0f;
            for (int k = 0; k < top_k; ++k)
                topk_sum += selected_weights[k];

            bool local_compute_flags[kMaxTopK] = {};
            if (shape_ok)
            {
                runtime_resolve_decode_dispatch(
                    runtime,
                    selected,
                    num_experts,
                    top_k,
                    update_runtime_histogram,
                    local_compute_flags);
            }
            for (int k = 0; k < top_k; ++k)
            {
                const float weight = normalize_weights && topk_sum > 0.0f
                                         ? selected_weights[k] / topk_sum
                                         : selected_weights[k];
                if (shape_ok)
                {
                    const bool local_compute = local_compute_flags[k];
                    runtime->topk_expert_ids[k] = local_compute ? selected[k] : -1;
                    runtime->topk_weights[k] = local_compute ? weight : 0.0f;
                }
                if (write_legacy_outputs)
                {
                    legacy_indices[k] = static_cast<float>(selected[k]);
                    legacy_weights[k] = weight;
                }
            }
            if (update_runtime_histogram)
            {
                for (int k = 0; k < top_k; ++k)
                {
                    const int expert = selected[k];
                    if (expert < 0 || expert >= kDeviceMoEMaxExperts)
                        continue;
                    atomicAdd(reinterpret_cast<unsigned long long *>(&runtime->decode_histogram[expert]),
                              static_cast<unsigned long long>(1));
                    if (shape_ok && local_compute_flags[k])
                    {
                        atomicAdd(reinterpret_cast<unsigned long long *>(&runtime->decode_local_histogram[expert]),
                                  static_cast<unsigned long long>(1));
                    }
                }
            }
        }
    }

    __global__ void decode_route_select_runtime_kernel(
        const int *__restrict__ expert_indices,
        const float *__restrict__ expert_weights,
        DeviceMoELayerRuntimeView *__restrict__ runtime,
        float *legacy_indices,
        float *legacy_weights,
        int num_experts,
        int top_k,
        bool write_legacy_outputs,
        bool update_runtime_histogram)
    {
        if (threadIdx.x != 0)
            return;

        const bool shape_ok = runtime_shape_ok(runtime, num_experts, top_k);
        bool local_compute_flags[kMaxTopK] = {};
        if (shape_ok)
        {
            runtime_resolve_decode_dispatch(
                runtime,
                expert_indices,
                num_experts,
                top_k,
                update_runtime_histogram,
                local_compute_flags);
        }
        for (int k = 0; k < top_k; ++k)
        {
            const int expert = expert_indices[k];
            const float weight = expert_weights[k];
            if (shape_ok)
            {
                const bool local_compute = local_compute_flags[k];
                runtime->topk_expert_ids[k] = local_compute ? expert : -1;
                runtime->topk_weights[k] = local_compute ? weight : 0.0f;
            }
            if (write_legacy_outputs)
            {
                legacy_indices[k] = static_cast<float>(expert);
                legacy_weights[k] = weight;
            }
        }

        if (update_runtime_histogram)
        {
            for (int k = 0; k < top_k; ++k)
            {
                const int expert = expert_indices[k];
                if (expert < 0 || expert >= kDeviceMoEMaxExperts)
                    continue;
                atomicAdd(reinterpret_cast<unsigned long long *>(&runtime->decode_histogram[expert]),
                          static_cast<unsigned long long>(1));
                if (shape_ok && local_compute_flags[k])
                {
                    atomicAdd(reinterpret_cast<unsigned long long *>(&runtime->decode_local_histogram[expert]),
                              static_cast<unsigned long long>(1));
                }
            }
        }
    }

    __global__ void int_to_float_kernel(const int *__restrict__ input, float *__restrict__ output, int count)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < count)
            output[idx] = static_cast<float>(input[idx]);
    }

    __global__ void float_to_int_kernel(const float *__restrict__ input, int *__restrict__ output, int count)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < count)
            output[idx] = static_cast<int>(input[idx]);
    }

    __global__ void float_to_masked_int_kernel(
        const float *__restrict__ input,
        int *__restrict__ output,
        const uint8_t *__restrict__ expert_mask,
        int count,
        int num_experts)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count)
            return;
        const int expert = static_cast<int>(input[idx]);
        output[idx] =
            (expert >= 0 && expert < num_experts && expert_mask && expert_mask[expert] != 0u)
                ? expert
                : -1;
    }

    __global__ void gather_tokens_kernel(
        const float *__restrict__ hidden,
        float *__restrict__ batch_buffer,
        const int *__restrict__ token_indices,
        int num_tokens, int d_model)
    {
        // 2D grid: blockIdx.y selects the token row, threads cover the d_model columns
        // as float4. The original flat 1D layout paid an integer div+mod (idx/d_model,
        // idx%d_model) per element and used scalar copies; this version removes both.
        // d_model is a multiple of 32 (enforced) → safe to treat the row as float4.
        const int token_slot = blockIdx.y;
        if (token_slot >= num_tokens)
            return;
        const int n4 = d_model >> 2;
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n4)
            return;
        const int token = token_indices[token_slot];
        const float4 *src = reinterpret_cast<const float4 *>(hidden + static_cast<size_t>(token) * d_model);
        float4 *dst = reinterpret_cast<float4 *>(batch_buffer + static_cast<size_t>(token_slot) * d_model);
        dst[i] = src[i];
    }

    __global__ void gather_tokens_scalar_kernel(
        const float *__restrict__ hidden,
        float *__restrict__ batch_buffer,
        const int *__restrict__ token_indices,
        int total_elements, int d_model)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_elements)
            return;

        const int token_slot = idx / d_model;
        const int col = idx - token_slot * d_model;
        const int token = token_indices[token_slot];
        batch_buffer[idx] = hidden[static_cast<size_t>(token) * d_model + col];
    }

    /**
     * @brief Copy one logical row without staging a host token-index buffer.
     *
     * MTP verifier replay uses this primitive to keep row selection owned by
     * the captured GPU stream.  The scalar tail keeps routing rows such as
     * top_k=8 valid even though model-width rows use the vectorized float4 path.
     */
    __global__ void copy_token_row_kernel(
        const float *__restrict__ source,
        float *__restrict__ row_buffer,
        int row_index,
        int row_width)
    {
        if (row_index < 0 || row_width <= 0)
            return;
        const float *src = source + static_cast<size_t>(row_index) * row_width;
        const int n4 = row_width >> 2;
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n4)
        {
            reinterpret_cast<float4 *>(row_buffer)[i] =
                reinterpret_cast<const float4 *>(src)[i];
        }
        const int tail_start = n4 << 2;
        for (int col = tail_start + i; col < row_width; col += gridDim.x * blockDim.x)
            row_buffer[col] = src[col];
    }

    __global__ void scatter_add_kernel(
        float *__restrict__ output,
        const float *__restrict__ expert_output,
        const int *__restrict__ token_indices,
        const float *__restrict__ weights,
        int num_tokens, int d_model)
    {
        // 2D grid + float4, mirroring gather_tokens_kernel. Each token_slot maps to a
        // distinct output token (caller guarantees uniqueness — original used a plain
        // non-atomic +=, preserved here), so the read-modify-write per float4 is race
        // free. Removes the per-element div/mod and vectorizes the accumulate.
        const int token_slot = blockIdx.y;
        if (token_slot >= num_tokens)
            return;
        const int n4 = d_model >> 2;
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n4)
            return;
        const int token = token_indices[token_slot];
        const float w = weights[token_slot];
        const float4 *ev = reinterpret_cast<const float4 *>(expert_output + static_cast<size_t>(token_slot) * d_model);
        float4 *ov = reinterpret_cast<float4 *>(output + static_cast<size_t>(token) * d_model);
        const float4 e = ev[i];
        float4 o = ov[i];
        o.x += w * e.x;
        o.y += w * e.y;
        o.z += w * e.z;
        o.w += w * e.w;
        ov[i] = o;
    }

    __global__ void scatter_add_scalar_kernel(
        float *__restrict__ output,
        const float *__restrict__ expert_output,
        const int *__restrict__ token_indices,
        const float *__restrict__ weights,
        int total_elements, int d_model)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_elements)
            return;

        const int token_slot = idx / d_model;
        const int col = idx - token_slot * d_model;
        const int token = token_indices[token_slot];
        output[static_cast<size_t>(token) * d_model + col] += weights[token_slot] * expert_output[idx];
    }

    /**
     * @brief Overwrite one destination row from a one-row scratch tensor.
     *
     * Decode-equivalent verifier replay writes each semantic row exactly once,
     * so this direct store avoids atomics and avoids async H2D staging of
     * `{row, weight}` metadata.
     */
    __global__ void write_token_row_kernel(
        float *__restrict__ destination,
        const float *__restrict__ row_buffer,
        int row_index,
        int row_width)
    {
        if (row_index < 0 || row_width <= 0)
            return;
        float *dst = destination + static_cast<size_t>(row_index) * row_width;
        const int n4 = row_width >> 2;
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n4)
        {
            reinterpret_cast<float4 *>(dst)[i] =
                reinterpret_cast<const float4 *>(row_buffer)[i];
        }
        const int tail_start = n4 << 2;
        for (int col = tail_start + i; col < row_width; col += gridDim.x * blockDim.x)
            dst[col] = row_buffer[col];
    }

    __device__ __forceinline__ int clamp_effective_seq_len(
        int seq_len,
        const int *__restrict__ device_effective_seq_len)
    {
        if (!device_effective_seq_len)
            return seq_len;
        const int raw = *device_effective_seq_len;
        return raw < 0 ? 0 : (raw > seq_len ? seq_len : raw);
    }

    __global__ void shared_expert_gate_kernel(
        const float *__restrict__ input,
        const float *__restrict__ gate_inp,
        float *__restrict__ shared_output,
        int seq_len, int d_model,
        const int *__restrict__ device_effective_seq_len = nullptr)
    {
        const int token = blockIdx.x;
        if (token >= seq_len)
            return;

        const int effective_seq_len = clamp_effective_seq_len(seq_len, device_effective_seq_len);
        const size_t row_offset = static_cast<size_t>(token) * d_model;
        if (token >= effective_seq_len)
        {
            float4 *out4 = reinterpret_cast<float4 *>(shared_output + row_offset);
            const int n4 = d_model >> 2;
            for (int i = threadIdx.x; i < n4; i += blockDim.x)
                out4[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int i = (n4 << 2) + threadIdx.x; i < d_model; i += blockDim.x)
                shared_output[row_offset + i] = 0.0f;
            return;
        }

        __shared__ float partial[kThreads];
        const float *x = input + row_offset;

        // d_model is always a multiple of 32 (enforced upstream), so it is also a
        // multiple of 4 → we can process the row as float4 to cut the load/store
        // instruction count 4× (the original scalar stride loop was MIO-bound at 65%
        // memory throughput). Both rows start at token*d_model*4 bytes, which is
        // 16-byte aligned for d_model multiple of 4.
        const int n4 = d_model >> 2;
        const float4 *x4 = reinterpret_cast<const float4 *>(x);
        const float4 *g4 = reinterpret_cast<const float4 *>(gate_inp);

        float dot = 0.0f;
        for (int i = threadIdx.x; i < n4; i += blockDim.x)
        {
            const float4 a = x4[i];
            const float4 b = g4[i];
            dot += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
                partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }

        const float gate = 1.0f / (1.0f + expf(-partial[0]));
        float4 *out4 = reinterpret_cast<float4 *>(shared_output + row_offset);
        for (int i = threadIdx.x; i < n4; i += blockDim.x)
        {
            float4 v = out4[i];
            v.x *= gate;
            v.y *= gate;
            v.z *= gate;
            v.w *= gate;
            out4[i] = v;
        }
        for (int i = (n4 << 2) + threadIdx.x; i < d_model; i += blockDim.x)
            shared_output[row_offset + i] *= gate;
    }

    __global__ void shared_expert_gate_add_kernel(
        const float *__restrict__ input,
        const float *__restrict__ gate_inp,
        float *__restrict__ shared_output,
        const float *__restrict__ routed_residual,
        float *__restrict__ combined_output,
        int seq_len, int d_model,
        const int *__restrict__ device_effective_seq_len = nullptr)
    {
        const int token = blockIdx.x;
        if (token >= seq_len)
            return;

        __shared__ float partial[kThreads];
        const size_t row_offset = static_cast<size_t>(token) * d_model;
        const int effective_seq_len = clamp_effective_seq_len(seq_len, device_effective_seq_len);
        if (token >= effective_seq_len)
        {
            float4 *shared4 = reinterpret_cast<float4 *>(shared_output + row_offset);
            float4 *out4 = reinterpret_cast<float4 *>(combined_output + row_offset);
            const int n4 = d_model >> 2;
            for (int i = threadIdx.x; i < n4; i += blockDim.x)
            {
                shared4[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                out4[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            for (int i = (n4 << 2) + threadIdx.x; i < d_model; i += blockDim.x)
            {
                shared_output[row_offset + i] = 0.0f;
                combined_output[row_offset + i] = 0.0f;
            }
            return;
        }
        const float *x = input + row_offset;

        const int n4 = d_model >> 2;
        const float4 *x4 = reinterpret_cast<const float4 *>(x);
        const float4 *g4 = reinterpret_cast<const float4 *>(gate_inp);

        float dot = 0.0f;
        for (int i = threadIdx.x; i < n4; i += blockDim.x)
        {
            const float4 a = x4[i];
            const float4 b = g4[i];
            dot += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
        for (int i = (n4 << 2) + threadIdx.x; i < d_model; i += blockDim.x)
            dot += x[i] * gate_inp[i];

        partial[threadIdx.x] = dot;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
                partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }

        const float gate = 1.0f / (1.0f + expf(-partial[0]));
        float4 *shared4 = reinterpret_cast<float4 *>(shared_output + row_offset);
        const float4 *residual4 = reinterpret_cast<const float4 *>(routed_residual + row_offset);
        float4 *out4 = reinterpret_cast<float4 *>(combined_output + row_offset);
        for (int i = threadIdx.x; i < n4; i += blockDim.x)
        {
            const float4 s = shared4[i];
            const float4 r = residual4[i];
            const float4 gated = make_float4(gate * s.x, gate * s.y, gate * s.z, gate * s.w);
            shared4[i] = gated;
            out4[i] = make_float4(
                r.x + gated.x,
                r.y + gated.y,
                r.z + gated.z,
                r.w + gated.w);
        }
        for (int i = (n4 << 2) + threadIdx.x; i < d_model; i += blockDim.x)
        {
            const float gated = gate * shared_output[row_offset + i];
            shared_output[row_offset + i] = gated;
            combined_output[row_offset + i] = routed_residual[row_offset + i] + gated;
        }
    }

    __global__ void swiglu_kernel(float *__restrict__ gate, const float *__restrict__ up, int count)
    {
        // Process four contiguous elements per thread via float4 to cut the load/store
        // instruction count 4×. count (= m*intermediate) is not guaranteed to be a
        // multiple of 4, so vectorize the bulk and handle the ragged tail with scalars.
        const int n4 = count >> 2;
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n4)
        {
            float4 g = reinterpret_cast<float4 *>(gate)[idx];
            const float4 u = reinterpret_cast<const float4 *>(up)[idx];
            g.x = silu(g.x) * u.x;
            g.y = silu(g.y) * u.y;
            g.z = silu(g.z) * u.z;
            g.w = silu(g.w) * u.w;
            reinterpret_cast<float4 *>(gate)[idx] = g;
        }
        // Tail: the last (count & 3) elements. Only the first few threads do work.
        const int tail_base = n4 << 2;
        const int tail_idx = tail_base + idx;
        if (idx < (count & 3) && tail_idx < count)
            gate[tail_idx] = silu(gate[tail_idx]) * up[tail_idx];
    }

    __global__ void weighted_add_kernel(float *__restrict__ output, const float *__restrict__ input, float weight, int count)
    {
        // float4-vectorized fused multiply-add (output += weight*input). count is
        // typically d_model (a multiple of 32), but we still handle a scalar tail so
        // the kernel is safe for any count.
        const int n4 = count >> 2;
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n4)
        {
            float4 o = reinterpret_cast<float4 *>(output)[idx];
            const float4 in = reinterpret_cast<const float4 *>(input)[idx];
            o.x += weight * in.x;
            o.y += weight * in.y;
            o.z += weight * in.z;
            o.w += weight * in.w;
            reinterpret_cast<float4 *>(output)[idx] = o;
        }
        const int tail_base = n4 << 2;
        const int tail_idx = tail_base + idx;
        if (idx < (count & 3) && tail_idx < count)
            output[tail_idx] += weight * input[tail_idx];
    }

    __global__ void count_per_expert_kernel(
        const int *__restrict__ routing_indices,
        int *__restrict__ expert_counts,
        int total_slots, int num_experts)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_slots)
            return;
        const int expert = routing_indices[idx];
        if (expert >= 0 && expert < num_experts)
            atomicAdd(expert_counts + expert, 1);
    }

    __global__ void exclusive_scan_kernel(const int *__restrict__ expert_counts, int *__restrict__ expert_offsets, int num_experts)
    {
        // The prefix sum is inherently serial, but the original implementation ran it
        // on thread 0 reading/writing global memory — num_experts (256) dependent
        // global loads at ~400-cycle latency dominated the ~9µs runtime. Stage the
        // counts into shared memory with a coalesced strided load, run the serial scan
        // in smem (~20-cycle latency), then write the offsets back coalesced. Arithmetic
        // order is identical to the original, so results are bit-exact.
        __shared__ int s_counts[kMaxExperts];

        for (int i = threadIdx.x; i < num_experts; i += blockDim.x)
            s_counts[i] = expert_counts[i];
        __syncthreads();

        if (threadIdx.x == 0)
        {
            int running = 0;
            for (int expert = 0; expert < num_experts; ++expert)
            {
                const int c = s_counts[expert];
                s_counts[expert] = running; // in-place exclusive prefix
                running += c;
            }
        }
        __syncthreads();

        for (int i = threadIdx.x; i < num_experts; i += blockDim.x)
            expert_offsets[i] = s_counts[i];
    }

    __global__ void scatter_tokens_kernel(
        const int *__restrict__ routing_indices,
        const float *__restrict__ routing_weights,
        int *__restrict__ write_heads,
        const int *__restrict__ expert_offsets,
        int *__restrict__ grouped_token_indices,
        float *__restrict__ grouped_weights,
        int total_slots, int top_k, int num_experts)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_slots)
            return;
        const int expert = routing_indices[idx];
        if (expert < 0 || expert >= num_experts)
            return;
        const int local = atomicAdd(write_heads + expert, 1);
        const int dest = expert_offsets[expert] + local;
        grouped_token_indices[dest] = idx / top_k;
        grouped_weights[dest] = routing_weights[idx];
    }

    __global__ void build_active_expert_list_kernel(
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

    __global__ void scatter_tokens_deterministic_kernel(
        const int *__restrict__ routing_indices,
        const float *__restrict__ routing_weights,
        const int *__restrict__ expert_offsets,
        const int *__restrict__ expert_counts,
        int *__restrict__ grouped_token_indices,
        int *__restrict__ original_to_grouped,
        int *__restrict__ original_expert_ids,
        float *__restrict__ grouped_weights,
        int total_slots,
        int top_k,
        int num_experts)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_slots)
            return;

        const int expert = routing_indices[idx];
        if (expert < 0 || expert >= num_experts)
            return;

        int local = 0;
        for (int prev = 0; prev < idx; ++prev)
        {
            if (routing_indices[prev] == expert)
                ++local;
        }
        if (local >= expert_counts[expert])
            return;

        const int dest = expert_offsets[expert] + local;
        grouped_token_indices[dest] = idx / top_k;
        original_to_grouped[idx] = dest;
        original_expert_ids[idx] = expert;
        grouped_weights[dest] = routing_weights[idx];
    }

    __global__ void group_tokens_small_float_kernel(
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
        if (threadIdx.x != 0 || blockIdx.x != 0)
            return;

        for (int expert = 0; expert < num_experts; ++expert)
        {
            expert_counts[expert] = 0;
            expert_offsets[expert] = 0;
        }
        for (int idx = 0; idx < total_slots; ++idx)
        {
            original_to_grouped[idx] = -1;
            original_expert_ids[idx] = -1;
        }
        for (int slot = 0; slot < max_active_experts; ++slot)
            active_expert_ids[slot] = -1;

        for (int idx = 0; idx < total_slots; ++idx)
        {
            const int expert = static_cast<int>(routing_indices[idx]);
            if (expert >= 0 && expert < num_experts)
                ++expert_counts[expert];
        }

        int running = 0;
        int active_count = 0;
        for (int expert = 0; expert < num_experts; ++expert)
        {
            expert_offsets[expert] = running;
            const int count = expert_counts[expert];
            if (count > 0 && active_count < max_active_experts)
                active_expert_ids[active_count++] = expert;
            running += count;
        }

        for (int idx = 0; idx < total_slots; ++idx)
        {
            const int expert = static_cast<int>(routing_indices[idx]);
            if (expert < 0 || expert >= num_experts)
                continue;

            int local = 0;
            for (int prev = 0; prev < idx; ++prev)
            {
                if (static_cast<int>(routing_indices[prev]) == expert)
                    ++local;
            }

            const int dest = expert_offsets[expert] + local;
            grouped_token_indices[dest] = idx / top_k;
            original_to_grouped[idx] = dest;
            original_expert_ids[idx] = expert;
            grouped_weights[dest] = routing_weights[idx];
        }
    }

    __global__ void prepare_shared_expert_group_kernel(
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

    __global__ void prefill_group_clear_runtime_kernel(
        DeviceMoELayerRuntimeView *__restrict__ runtime,
        int max_slots,
        int num_experts,
        int clear_route_metadata)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (!runtime)
            return;

        if (idx < num_experts && runtime->expert_counts && runtime->expert_offsets)
        {
            runtime->expert_counts[idx] = 0;
            runtime->expert_offsets[idx] = 0;
        }

        if (idx < max_slots)
        {
            if (clear_route_metadata && runtime->route_expert_ids)
                runtime->route_expert_ids[idx] = -1;
            if (clear_route_metadata && runtime->route_weights)
                runtime->route_weights[idx] = 0.0f;
            if (clear_route_metadata && runtime->route_participant_ids)
                runtime->route_participant_ids[idx] = -1;
            if (runtime->grouped_token_ids)
                runtime->grouped_token_ids[idx] = 0;
            if (runtime->grouped_route_weights)
                runtime->grouped_route_weights[idx] = 0.0f;
        }
    }

    __global__ void prefill_group_cast_count_runtime_kernel(
        DeviceMoELayerRuntimeView *__restrict__ runtime,
        const float *__restrict__ routing_indices,
        const float *__restrict__ routing_weights,
        int current_slots,
        int max_slots,
        int num_experts)
    {
        const int slot = blockIdx.x * blockDim.x + threadIdx.x;
        if (!runtime || slot >= max_slots)
            return;
        if (!runtime->route_expert_ids || !runtime->route_weights || !runtime->route_participant_ids)
            return;
        if (runtime->prefill_route_capacity < static_cast<uint32_t>(max_slots))
            return;

        int expert_id = -1;
        float weight = 0.0f;
        int participant_id = -1;
        if (slot < current_slots)
        {
            expert_id = static_cast<int>(routing_indices[slot]);
            weight = routing_weights[slot];
            if (expert_id < 0 || expert_id >= num_experts)
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

    __global__ void prefill_group_count_assigned_runtime_kernel(
        DeviceMoELayerRuntimeView *__restrict__ runtime,
        int current_slots,
        int max_slots,
        int num_experts)
    {
        const int slot = blockIdx.x * blockDim.x + threadIdx.x;
        if (!runtime || slot >= max_slots || slot >= current_slots)
            return;
        if (!runtime->route_expert_ids || !runtime->route_participant_ids || !runtime->expert_counts)
            return;
        if (runtime->prefill_route_capacity < static_cast<uint32_t>(max_slots))
            return;

        const int expert_id = runtime->route_expert_ids[slot];
        const int participant_id = runtime->route_participant_ids[slot];
        if (expert_id >= 0 &&
            expert_id < num_experts &&
            participant_id == static_cast<int>(runtime->participant_id))
        {
            atomicAdd(runtime->expert_counts + expert_id, 1);
        }
    }

    __global__ void prefill_group_exclusive_scan_runtime_kernel(
        DeviceMoELayerRuntimeView *__restrict__ runtime,
        int num_experts)
    {
        if (!runtime || threadIdx.x != 0)
            return;
        if (!runtime->expert_counts || !runtime->expert_offsets)
            return;

        int running = 0;
        for (int expert = 0; expert < num_experts; ++expert)
        {
            const int count = runtime->expert_counts[expert];
            runtime->expert_offsets[expert] = running;
            running += count;
        }
    }

    __global__ void prefill_group_scatter_deterministic_runtime_kernel(
        DeviceMoELayerRuntimeView *__restrict__ runtime,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k)
    {
        const int expert = blockIdx.x;
        if (!runtime || expert >= num_experts || threadIdx.x != 0)
            return;
        if (!runtime->route_expert_ids || !runtime->route_weights ||
            !runtime->route_participant_ids ||
            !runtime->expert_offsets || !runtime->expert_counts ||
            !runtime->grouped_token_ids || !runtime->grouped_route_weights)
            return;
        if (runtime->prefill_route_capacity < static_cast<uint32_t>(max_slots))
            return;

        int write = runtime->expert_offsets[expert];
        const int end = runtime->expert_offsets[expert] + runtime->expert_counts[expert];
        for (int slot = 0; slot < max_slots; ++slot)
        {
            if (slot >= current_slots)
                continue;
            if (runtime->route_expert_ids[slot] != expert)
                continue;
            if (runtime->route_participant_ids[slot] != static_cast<int>(runtime->participant_id))
                continue;
            if (write < end)
            {
                runtime->grouped_token_ids[write] = slot / top_k;
                runtime->grouped_route_weights[write] = runtime->route_weights[slot];
                ++write;
            }
        }
    }

    __global__ void prefill_llep_count_expert_routes_runtime_kernel(
        DeviceMoELayerRuntimeView *__restrict__ runtime,
        int current_slots,
        int max_slots,
        int num_experts)
    {
        const int slot = blockIdx.x * blockDim.x + threadIdx.x;
        if (!runtime)
            return;
        if (slot >= current_slots || slot >= max_slots)
            return;
        if (!runtime->route_expert_ids || !runtime->expert_counts)
            return;
        if (runtime->prefill_route_capacity < static_cast<uint32_t>(max_slots))
            return;

        const int expert_id = runtime->route_expert_ids[slot];
        if (expert_id >= 0 && expert_id < num_experts)
            atomicAdd(runtime->expert_counts + expert_id, 1);
    }

    __global__ void prefill_llep_clear_assignment_runtime_kernel(
        DeviceMoELayerRuntimeView *__restrict__ runtime,
        int max_slots,
        int num_experts)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (!runtime)
            return;

        if (idx < num_experts && runtime->expert_counts && runtime->expert_offsets)
        {
            runtime->expert_counts[idx] = 0;
            runtime->expert_offsets[idx] = -1;
        }

        if (idx < max_slots)
        {
            if (runtime->grouped_token_ids)
                runtime->grouped_token_ids[idx] = 0;
            if (runtime->grouped_route_weights)
                runtime->grouped_route_weights[idx] = 0.0f;
        }

        int32_t *split_ends = static_cast<int32_t *>(runtime->reserved_ptrs[0]);
        const int split_items = num_experts * static_cast<int>(kDeviceMoEMaxParticipants);
        if (idx < split_items && split_ends)
            split_ends[idx] = 0;
    }

    __global__ void prefill_llep_plan_resident_splits_runtime_kernel(
        DeviceMoELayerRuntimeView *__restrict__ runtime,
        int current_slots,
        int max_slots,
        int num_experts)
    {
        if (!runtime || blockIdx.x != 0 || threadIdx.x != 0)
            return;
        if (!runtime->route_expert_ids || !runtime->route_participant_ids ||
            !runtime->expert_counts || !runtime->expert_offsets || !runtime->reserved_ptrs[0])
            return;
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
            const uint32_t fallback =
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
                fallback,
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

    __global__ void prefill_llep_assign_routes_from_splits_runtime_kernel(
        DeviceMoELayerRuntimeView *__restrict__ runtime,
        int current_slots,
        int max_slots,
        int num_experts)
    {
        const int slot = blockIdx.x * blockDim.x + threadIdx.x;
        if (!runtime || slot >= current_slots || slot >= max_slots)
            return;
        if (!runtime->route_expert_ids || !runtime->route_participant_ids ||
            !runtime->expert_offsets || !runtime->reserved_ptrs[0])
            return;
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
        const uint32_t fallback =
            (owner >= 0 && static_cast<uint32_t>(owner) < participant_count)
                ? static_cast<uint32_t>(owner)
                : (runtime->participant_id < participant_count ? runtime->participant_id : 0u);

        auto *split_ends = static_cast<int32_t *>(runtime->reserved_ptrs[0]);
        const int ordinal = atomicAdd(runtime->expert_offsets + expert_id, 1);
        const int split_base = expert_id * static_cast<int>(kDeviceMoEMaxParticipants);
        uint32_t assigned_participant = fallback;
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

    __global__ void materialize_runtime_prefill_descriptor_tables_kernel(
        const DeviceMoELayerRuntimeView *__restrict__ runtime,
        DeviceNativeVNNIMatrixDesc *__restrict__ gate_descs,
        DeviceNativeVNNIMatrixDesc *__restrict__ up_descs,
        DeviceNativeVNNIMatrixDesc *__restrict__ down_descs,
        int num_experts)
    {
        const int expert = blockIdx.x * blockDim.x + threadIdx.x;
        if (!runtime || !gate_descs || !up_descs || !down_descs)
            return;
        if (expert >= num_experts || expert >= kDeviceMoEMaxExperts)
            return;
        if (runtime->active_bank > 1u)
            return;

        const DeviceMoEPlacementBankView &bank = runtime->banks[runtime->active_bank];
        const DeviceMoEExpertDescriptorView &desc = bank.experts[expert];
        gate_descs[expert] = desc.gate;
        up_descs[expert] = desc.up;
        down_descs[expert] = desc.down;
    }

    __global__ void prefill_gather_expert_runtime_kernel(
        const DeviceMoELayerRuntimeView *__restrict__ runtime,
        const float *__restrict__ hidden,
        float *__restrict__ batch_buffer,
        int expert_id,
        int max_tokens,
        int d_model)
    {
        const int linear = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = max_tokens * d_model;
        if (!runtime || !hidden || !batch_buffer || linear >= total)
            return;
        if (!runtime->expert_counts || !runtime->expert_offsets || !runtime->grouped_token_ids)
            return;
        if (expert_id < 0 || expert_id >= static_cast<int>(runtime->expert_count))
            return;
        if (runtime->prefill_token_capacity < static_cast<uint32_t>(max_tokens))
            return;

        const int row = linear / d_model;
        const int col = linear - row * d_model;
        const int count = runtime->expert_counts[expert_id];
        const int offset = runtime->expert_offsets[expert_id];

        float value = 0.0f;
        if (row < count)
        {
            const int token_id = runtime->grouped_token_ids[offset + row];
            if (token_id >= 0 && token_id < max_tokens)
                value = hidden[token_id * d_model + col];
        }
        batch_buffer[linear] = value;
    }

    __global__ void prefill_scatter_expert_runtime_kernel(
        float *__restrict__ output,
        const float *__restrict__ expert_output,
        const DeviceMoELayerRuntimeView *__restrict__ runtime,
        int expert_id,
        int max_tokens,
        int d_model)
    {
        const int linear = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = max_tokens * d_model;
        if (!runtime || !output || !expert_output || linear >= total)
            return;
        if (!runtime->expert_counts || !runtime->expert_offsets ||
            !runtime->grouped_token_ids || !runtime->grouped_route_weights)
            return;
        if (runtime->prefill_token_capacity < static_cast<uint32_t>(max_tokens))
            return;
        if (expert_id < 0 || expert_id >= static_cast<int>(runtime->expert_count))
            return;

        const int row = linear / d_model;
        const int col = linear - row * d_model;
        const int count = runtime->expert_counts[expert_id];
        if (row >= count)
            return;

        const int offset = runtime->expert_offsets[expert_id];
        const int token_id = runtime->grouped_token_ids[offset + row];
        if (token_id < 0 || token_id >= max_tokens)
            return;
        const float weight = runtime->grouped_route_weights[offset + row];
        atomicAdd(output + token_id * d_model + col, weight * expert_output[linear]);
    }

    __global__ void gather_expert_fixed_kernel(
        const float *__restrict__ hidden,
        float *__restrict__ batch_buffer,
        const int *__restrict__ expert_offsets,
        const int *__restrict__ expert_counts,
        const int *__restrict__ grouped_token_indices,
        int expert_id,
        int max_tokens,
        int d_model)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = max_tokens * d_model;
        if (idx >= total)
            return;

        const int token_slot = idx / d_model;
        const int col = idx % d_model;
        const int count = expert_counts[expert_id];
        float value = 0.0f;
        if (token_slot < count)
        {
            const int grouped = expert_offsets[expert_id] + token_slot;
            const int token = grouped_token_indices[grouped];
            value = hidden[static_cast<size_t>(token) * d_model + col];
        }
        batch_buffer[idx] = value;
    }

    __global__ void scatter_expert_fixed_kernel(
        float *__restrict__ output,
        const float *__restrict__ expert_output,
        const int *__restrict__ expert_offsets,
        const int *__restrict__ expert_counts,
        const int *__restrict__ grouped_token_indices,
        const float *__restrict__ grouped_weights,
        int expert_id,
        int max_tokens,
        int d_model)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = max_tokens * d_model;
        if (idx >= total)
            return;

        const int token_slot = idx / d_model;
        const int col = idx % d_model;
        const int count = expert_counts[expert_id];
        if (token_slot >= count)
            return;

        const int grouped = expert_offsets[expert_id] + token_slot;
        const int token = grouped_token_indices[grouped];
        const float weight = grouped_weights[grouped];
        atomicAdd(output + static_cast<size_t>(token) * d_model + col,
                  weight * expert_output[idx]);
    }

    __global__ void grouped_hidden_quantize_blockwise_kernel(
        const float *__restrict__ hidden,
        int8_t *__restrict__ A_int8,
        float *__restrict__ scales_A_blockwise,
        int K)
    {
        constexpr int kBlockSize = 32;
        const int block_idx = blockIdx.x;
        const int lane = threadIdx.x;
        const int col = block_idx * kBlockSize + lane;
        if (lane >= kBlockSize || col >= K)
            return;

        float abs_value = fabsf(hidden[col]);
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            abs_value = fmaxf(abs_value, __shfl_xor_sync(0xffffffffu, abs_value, mask));

        const float scale = (abs_value > 0.0f) ? (abs_value / 127.0f) : 1.0f;
        if (lane == 0)
            scales_A_blockwise[block_idx] = scale;

        const float q = hidden[col] / scale;
        A_int8[col] = static_cast<int8_t>(rintf(fminf(127.0f, fmaxf(-127.0f, q))));
    }

    __global__ void router_gate_quantize_q8_kernel(
        const float *__restrict__ gate_weights,
        int8_t *__restrict__ gate_weights_q8,
        float *__restrict__ gate_scales,
        int K,
        int num_experts)
    {
        constexpr int kBlockSize = 32;
        const int expert = blockIdx.x;
        const int block_idx = blockIdx.y;
        const int lane = threadIdx.x;
        if (expert >= num_experts || lane >= kBlockSize)
            return;

        const int blocks_per_row = K / kBlockSize;
        const int col = block_idx * kBlockSize + lane;
        const size_t scale_idx = static_cast<size_t>(expert) * blocks_per_row + block_idx;
        const float value = (col < K)
                                ? gate_weights[static_cast<size_t>(expert) * K + col]
                                : 0.0f;

        float abs_value = fabsf(value);
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            abs_value = fmaxf(abs_value, __shfl_xor_sync(0xffffffffu, abs_value, mask));

        const float scale = (abs_value > 0.0f) ? (abs_value / 127.0f) : 1.0f;
        if (lane == 0)
            gate_scales[scale_idx] = scale;

        const float q = value / scale;
        gate_weights_q8[scale_idx * kBlockSize + lane] =
            static_cast<int8_t>(rintf(fminf(127.0f, fmaxf(-127.0f, q))));
    }

    __global__ void router_gate_logits_single_token_q8_kernel(
        const int8_t *__restrict__ hidden_q8,
        const float *__restrict__ hidden_scales,
        const int8_t *__restrict__ gate_weights_q8,
        const float *__restrict__ gate_scales,
        float *__restrict__ logits,
        int K,
        int num_experts)
    {
        constexpr int kBlockSize = 32;
        const int expert = blockIdx.x;
        if (expert >= num_experts)
            return;

        const int blocks_per_row = K / kBlockSize;
        float sum = 0.0f;
        for (int block_idx = threadIdx.x; block_idx < blocks_per_row; block_idx += blockDim.x)
        {
            const int *h4 = reinterpret_cast<const int *>(hidden_q8 + block_idx * kBlockSize);
            const size_t scale_idx = static_cast<size_t>(expert) * blocks_per_row + block_idx;
            const int *w4 = reinterpret_cast<const int *>(gate_weights_q8 + scale_idx * kBlockSize);

            int block_acc = 0;
#pragma unroll
            for (int group = 0; group < 8; ++group)
                block_acc = __dp4a(h4[group], w4[group], block_acc);
            sum += static_cast<float>(block_acc) * hidden_scales[block_idx] * gate_scales[scale_idx];
        }

        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);

        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int num_warps = blockDim.x >> 5;
        __shared__ float warp_sums[32];
        if (lane == 0)
            warp_sums[warp] = sum;
        __syncthreads();
        if (warp == 0)
        {
            float v = (lane < num_warps) ? warp_sums[lane] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1)
                v += __shfl_down_sync(0xffffffffu, v, offset);
            if (lane == 0)
                logits[expert] = v;
        }
    }

    __global__ void grouped_swiglu_quantize_blockwise_kernel(
        const float *const *__restrict__ gate_ptrs,
        const float *const *__restrict__ up_ptrs,
        const int *__restrict__ expert_ids,
        int8_t *__restrict__ A_int8,
        float *__restrict__ scales_A_blockwise,
        int num_active,
        int K)
    {
        const int slot = blockIdx.x;
        if (slot >= num_active)
            return;
        if (expert_ids && expert_ids[slot] < 0)
            return;

        constexpr int kBlockSize = 32;
        constexpr int kWarps = 8;
        const int lane = threadIdx.x & 31;
        const int warp_id = threadIdx.x >> 5;
        const int blocks_per_row = K / kBlockSize;
        const float *gate = gate_ptrs[slot];
        const float *up = up_ptrs[slot];
        int8_t *row_int8 = A_int8 + static_cast<size_t>(slot) * K;
        float *row_scales = scales_A_blockwise + static_cast<size_t>(slot) * blocks_per_row;

        for (int block_idx = warp_id; block_idx < blocks_per_row; block_idx += kWarps)
        {
            const int col = block_idx * kBlockSize + lane;
            const float g = gate[col];
            const float value = (g / (1.0f + expf(-g))) * up[col];

            float abs_value = fabsf(value);
#pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1)
                abs_value = fmaxf(abs_value, __shfl_xor_sync(0xffffffffu, abs_value, mask));

            const float scale = (abs_value > 0.0f) ? (abs_value / 127.0f) : 1.0f;
            if (lane == 0)
                row_scales[block_idx] = scale;

            const float q = value / scale;
            row_int8[col] = static_cast<int8_t>(rintf(fminf(127.0f, fmaxf(-127.0f, q))));
        }
    }

    // Gather + blockwise-int8 quantize for the prefill expert pipeline.
    //
    // Each 32-lane warp owns exactly one 32-element quant block: it loads the block,
    // computes the absmax via warp shuffle, derives the per-block scale, and writes the
    // quantized int8 + scale. The original launch used block=32 (a single warp), which
    // caps occupancy at one warp per block (profiled ~21% occupancy). Packing
    // kWarpsPerQuantBlock warps into each CUDA block lets the scheduler run many warps
    // per SM with the same per-warp work, lifting occupancy ~8× with no extra arithmetic.
    static constexpr int kWarpsPerQuantBlock = 8; // 8 warps = 256-thread block
    __global__ void grouped_prefill_gather_quantize_blockwise_kernel(
        const float *__restrict__ hidden,
        int8_t *__restrict__ A_int8,
        float *__restrict__ scales_A_blockwise,
        const int *__restrict__ grouped_token_indices,
        int total_slots,
        int K)
    {
        const int slot = blockIdx.y;
        if (slot >= total_slots)
            return;

        // Map this thread to (warp, lane); each warp handles a distinct 32-col block.
        const int warp = threadIdx.x >> 5;  // 0..kWarpsPerQuantBlock-1
        const int lane = threadIdx.x & 31;  // 0..31
        const int block_idx = blockIdx.x * kWarpsPerQuantBlock + warp;
        const int col = block_idx * 32 + lane;
        const int blocks_per_row = (K + 31) / 32;
        if (block_idx >= blocks_per_row || col >= K)
            return;

        const int source_token = grouped_token_indices[slot];
        const float value = hidden[static_cast<size_t>(source_token) * K + col];

        // Warp-local absmax reduction (each warp owns its own 32-element block).
        float abs_value = fabsf(value);
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            abs_value = fmaxf(abs_value, __shfl_xor_sync(0xffffffffu, abs_value, mask));

        const float scale = (abs_value > 0.0f) ? (abs_value / 127.0f) : 1.0f;
        if (lane == 0)
            scales_A_blockwise[static_cast<size_t>(slot) * blocks_per_row + block_idx] = scale;

        const float q = value / scale;
        A_int8[static_cast<size_t>(slot) * K + col] =
            static_cast<int8_t>(rintf(fminf(127.0f, fmaxf(-127.0f, q))));
    }

    template <uint8_t CodebookId, int TileM>
    __device__ __forceinline__ void accumulate_prefill_dot_block(
        const int32_t (&packed_groups)[8],
        const uint16_t *__restrict__ scale_base,
        const uint16_t *__restrict__ min_base,
        const uint32_t *__restrict__ emin_base,
        const uint8_t *__restrict__ payload,
        size_t linear,
        const int32_t *__restrict__ a4_base,
        int a4_stride_i32,
        const float *__restrict__ scale_a_base,
        int scale_a_stride,
        int tokens_in_group,
        float (&acc)[TileM])
    {
#pragma unroll
        for (int m = 0; m < TileM; ++m)
        {
            if (m >= tokens_in_group)
                break;

            // Per-token activation row: a4_base/scale_a_base point at a staged shared-memory
            // tile (contiguous, stride = 8 int32 / 1 scale). The 8 int32 are read as 2x128-bit
            // vector loads to cut the LDS instruction count 4x (relieves the MIO pipe).
            // Requires a4_base + m*stride to be 16-byte aligned (s_a is __align__(16), stride=8).
            const int4 *a4v = reinterpret_cast<const int4 *>(a4_base + static_cast<size_t>(m) * a4_stride_i32);
            const int4 av0 = a4v[0];
            const int4 av1 = a4v[1];
            const int32_t a4[8] = {av0.x, av0.y, av0.z, av0.w, av1.x, av1.y, av1.z, av1.w};
            const float scale_a = scale_a_base[static_cast<size_t>(m) * scale_a_stride];

            if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CodebookId>::is_dual_scale)
            {
                int dot_lo = 0;
                int dot_hi = 0;
                int sum_lo = 0;
                int sum_hi = 0;
#pragma unroll
                for (int group = 0; group < 4; ++group)
                {
                    dot_lo = __dp4a(a4[group], packed_groups[group], dot_lo);
                    sum_lo += llaminar2::cuda_native_vnni::sum_packed_i8(a4[group]);
                }
#pragma unroll
                for (int group = 4; group < 8; ++group)
                {
                    dot_hi = __dp4a(a4[group], packed_groups[group], dot_hi);
                    sum_hi += llaminar2::cuda_native_vnni::sum_packed_i8(a4[group]);
                }

                const float scale_lo = llaminar2::cuda_native_vnni::fp16_bits_to_float(scale_base[linear]);
                const float scale_hi = min_base ? llaminar2::cuda_native_vnni::fp16_bits_to_float(min_base[linear]) : 0.0f;
                acc[m] += scale_a * (scale_lo * static_cast<float>(dot_lo) +
                                     scale_hi * static_cast<float>(dot_hi));

                if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CodebookId>::is_dual_scale_asym)
                {
                    const uint32_t emin = emin_base ? emin_base[linear] : 0u;
                    const float min_lo = llaminar2::cuda_native_vnni::fp16_bits_to_float(static_cast<uint16_t>(emin));
                    const float min_hi = llaminar2::cuda_native_vnni::fp16_bits_to_float(static_cast<uint16_t>(emin >> 16));
                    acc[m] += scale_a * (min_lo * static_cast<float>(sum_lo) +
                                         min_hi * static_cast<float>(sum_hi));
                }

                if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CodebookId>::is_iq1_m)
                {
                    constexpr float kIQ1SDelta = 0.125f;
                    const uint8_t qh0 = payload[4];
                    const uint8_t qh1 = payload[5];
                    const int sg0 = llaminar2::cuda_native_vnni::sum_packed_i8(a4[0]) +
                                    llaminar2::cuda_native_vnni::sum_packed_i8(a4[1]);
                    const int sg1 = llaminar2::cuda_native_vnni::sum_packed_i8(a4[2]) +
                                    llaminar2::cuda_native_vnni::sum_packed_i8(a4[3]);
                    const int sg2 = llaminar2::cuda_native_vnni::sum_packed_i8(a4[4]) +
                                    llaminar2::cuda_native_vnni::sum_packed_i8(a4[5]);
                    const int sg3 = llaminar2::cuda_native_vnni::sum_packed_i8(a4[6]) +
                                    llaminar2::cuda_native_vnni::sum_packed_i8(a4[7]);
                    const float delta0 = (qh0 & 0x08) ? -kIQ1SDelta : kIQ1SDelta;
                    const float delta1 = (qh0 & 0x80) ? -kIQ1SDelta : kIQ1SDelta;
                    const float delta2 = (qh1 & 0x08) ? -kIQ1SDelta : kIQ1SDelta;
                    const float delta3 = (qh1 & 0x80) ? -kIQ1SDelta : kIQ1SDelta;
                    acc[m] += scale_a * ((delta0 * static_cast<float>(sg0) + delta1 * static_cast<float>(sg1)) * scale_lo +
                                         (delta2 * static_cast<float>(sg2) + delta3 * static_cast<float>(sg3)) * scale_hi);
                }
            }
            else
            {
                int dot = 0;
                int sum_a = 0;
#pragma unroll
                for (int group = 0; group < 8; ++group)
                {
                    dot = __dp4a(a4[group], packed_groups[group], dot);
                    sum_a += llaminar2::cuda_native_vnni::sum_packed_i8(a4[group]);
                }

                const float scale_b = llaminar2::cuda_native_vnni::fp16_bits_to_float(scale_base[linear]);
                acc[m] += scale_a * scale_b * static_cast<float>(dot);

                if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CodebookId>::is_asymmetric)
                {
                    const float min_b = min_base ? llaminar2::cuda_native_vnni::fp16_bits_to_float(min_base[linear]) : 0.0f;
                    acc[m] += scale_a * min_b * static_cast<float>(sum_a);
                }
            }
        }
    }

    template <uint8_t CodebookId, int kTileM, int kTileN>
    __global__ void grouped_native_vnni_gate_up_prefill_kernel(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        const DeviceNativeVNNIMatrixDesc *__restrict__ gate_descs,
        const DeviceNativeVNNIMatrixDesc *__restrict__ up_descs,
        const int *__restrict__ expert_counts,
        const int *__restrict__ expert_offsets,
        const int *__restrict__ active_expert_ids,
        int active_expert_slots,
        float *__restrict__ gate_output,
        float *__restrict__ up_output,
        int N,
        int K)
    {
        // kTileN columns per block (one per thread); kTileM tokens processed per block.
        // Larger kTileM amortizes the expensive IQ-codebook weight decode (done once per
        // (block_idx, n)) over more tokens and reduces redundant cross-group re-decode.
        const int n = blockIdx.x * kTileN + threadIdx.x;
        const int token_group = blockIdx.y;
        const int expert_id = (active_expert_slots > 0)
                                  ? active_expert_ids[blockIdx.z]
                                  : static_cast<int>(blockIdx.z);
        if (expert_id < 0)
            return;

        const int count = expert_counts[expert_id];
        const int first_token = token_group * kTileM;
        if (first_token >= count)
            return; // Whole-block uniform exit (token_group/expert from blockIdx) — barrier-safe.

        const int tokens_in_group = min(kTileM, count - first_token);
        const int first_slot = expert_offsets[expert_id] + first_token;
        const int blocks_per_row = K / 32;

        // Per-thread output column may be out of range when N is not a multiple of kTileN.
        // We must NOT early-return such threads: they still participate in the cooperative
        // shared-memory staging + __syncthreads below. Guard only the weight work / writes.
        const bool active = (n < N);

        const DeviceNativeVNNIMatrixDesc gate_desc = gate_descs[expert_id];
        const DeviceNativeVNNIMatrixDesc up_desc = up_descs[expert_id];
        if (!native_vnni_desc_shape_ok<CodebookId>(gate_desc, N, K) ||
            !native_vnni_desc_shape_ok<CodebookId>(up_desc, N, K))
            return;

        const uint8_t *gate_payload_base = gate_desc.payload;
        const uint16_t *gate_scale_base = static_cast<const uint16_t *>(gate_desc.scales);
        const uint16_t *gate_min_base = static_cast<const uint16_t *>(gate_desc.mins);
        const uint32_t *gate_emin_base = static_cast<const uint32_t *>(gate_desc.emins);
        const uint8_t *up_payload_base = up_desc.payload;
        const uint16_t *up_scale_base = static_cast<const uint16_t *>(up_desc.scales);
        const uint16_t *up_min_base = static_cast<const uint16_t *>(up_desc.mins);
        const uint32_t *up_emin_base = static_cast<const uint32_t *>(up_desc.emins);

        // Staged activation tile for the current K-block: kTileM tokens × 32 int8 (= 8 int32)
        // plus one blockwise scale per token. Staging once per block (cooperatively) removes
        // the ~kTileN-way redundant L1 loads that all N-threads previously issued.
        // 16-byte aligned so the per-token 8×int32 row can be read as 2× int4 (128-bit) loads.
        __shared__ __align__(16) int32_t s_a[kTileM * 8];
        __shared__ float s_scale[kTileM];

        float gate_acc[kTileM] = {};
        float up_acc[kTileM] = {};

        // Weight decode is hoisted above the activation-staging barrier: for IQ3 codebooks
        // the decode is a data-dependent payload-byte load followed by a grid-table lookup
        // (a long-scoreboard latency chain). Issuing it before the cooperative smem staging
        // + __syncthreads lets the load latency overlap the staging loads and barrier wait,
        // with no extra register double-buffer (which would cost occupancy).
        constexpr int kPayloadBytes =
            llaminar2::cuda_native_vnni::CodebookTraits<CodebookId>::payload_bytes;

#pragma unroll 1
        for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx)
        {
            const size_t linear = static_cast<size_t>(block_idx) * N + static_cast<size_t>(n);

            // Decode this block's weights first so the global-load + grid-lookup latency
            // overlaps the activation staging + barrier below.
            int32_t gate_groups[8];
            int32_t up_groups[8];
            const uint8_t *gate_payload = gate_payload_base + linear * kPayloadBytes;
            const uint8_t *up_payload = up_payload_base + linear * kPayloadBytes;
            if (active)
            {
                llaminar2::cuda_native_vnni::decode_groups_vec<CodebookId>(gate_payload, gate_groups);
                llaminar2::cuda_native_vnni::decode_groups_vec<CodebookId>(up_payload, up_groups);
            }

            // Cooperatively load the activation tile for this K-block into shared memory.
            const int a_count = tokens_in_group * 8;
            for (int idx = threadIdx.x; idx < a_count; idx += kTileN)
            {
                const int m = idx >> 3;
                const int g = idx & 7;
                const int slot = first_slot + m;
                s_a[idx] = reinterpret_cast<const int32_t *>(
                    A_int8 + static_cast<size_t>(slot) * K + block_idx * 32)[g];
            }
            for (int m = threadIdx.x; m < tokens_in_group; m += kTileN)
            {
                const int slot = first_slot + m;
                s_scale[m] = scales_A_blockwise[static_cast<size_t>(slot) * blocks_per_row + block_idx];
            }
            __syncthreads();

            if (active)
            {
                accumulate_prefill_dot_block<CodebookId, kTileM>(
                    gate_groups, gate_scale_base, gate_min_base, gate_emin_base, gate_payload,
                    linear, s_a, /*a4_stride_i32=*/8, s_scale, /*scale_a_stride=*/1,
                    tokens_in_group, gate_acc);
                accumulate_prefill_dot_block<CodebookId, kTileM>(
                    up_groups, up_scale_base, up_min_base, up_emin_base, up_payload,
                    linear, s_a, /*a4_stride_i32=*/8, s_scale, /*scale_a_stride=*/1,
                    tokens_in_group, up_acc);
            }

            // Barrier before the next iteration overwrites the staged tile.
            __syncthreads();
        }

        if (active)
        {
#pragma unroll
            for (int m = 0; m < kTileM; ++m)
            {
                if (m >= tokens_in_group)
                    break;
                const int slot = first_slot + m;
                gate_output[static_cast<size_t>(slot) * N + n] = gate_acc[m];
                up_output[static_cast<size_t>(slot) * N + n] = up_acc[m];
            }
        }
    }

    // Fused gate/up GEMM + SwiGLU + blockwise int8 quantization.
    //
    // This is the fused-epilogue variant of grouped_native_vnni_gate_up_prefill_kernel: after
    // computing gate_acc/up_acc for each token in the tile, it directly evaluates
    // silu(gate)*up and blockwise-quantizes the result to int8, writing the down-projection
    // input (swiglu_int8 + swiglu_scales) in place. This eliminates the FP32 gate/up global
    // round-trip and the separate grouped_prefill_swiglu_quantize_blockwise_kernel launch.
    //
    // Quantization layout: each warp (32 lanes) spans exactly one aligned 32-wide block of the
    // intermediate (N) dimension (kTileN=128 is a multiple of 32, and N % 32 == 0), so the
    // per-block absmax is a warp-shuffle reduction with no cross-warp synchronization.
    template <uint8_t CodebookId, int kTileM, int kTileN>
    __global__ void grouped_native_vnni_gate_up_swiglu_prefill_kernel(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        const DeviceNativeVNNIMatrixDesc *__restrict__ gate_descs,
        const DeviceNativeVNNIMatrixDesc *__restrict__ up_descs,
        const int *__restrict__ expert_counts,
        const int *__restrict__ expert_offsets,
        const int *__restrict__ active_expert_ids,
        int active_expert_slots,
        int8_t *__restrict__ swiglu_int8,
        float *__restrict__ swiglu_scales,
        int N,
        int K)
    {
        const int n = blockIdx.x * kTileN + threadIdx.x;
        const int token_group = blockIdx.y;
        const int expert_id = (active_expert_slots > 0)
                                  ? active_expert_ids[blockIdx.z]
                                  : static_cast<int>(blockIdx.z);
        if (expert_id < 0)
            return;

        const int count = expert_counts[expert_id];
        const int first_token = token_group * kTileM;
        if (first_token >= count)
            return; // Whole-block uniform exit — barrier-safe.

        const int tokens_in_group = min(kTileM, count - first_token);
        const int first_slot = expert_offsets[expert_id] + first_token;
        const int blocks_per_row = K / 32;

        // Threads with n >= N still participate in cooperative staging + __syncthreads.
        const bool active = (n < N);

        const DeviceNativeVNNIMatrixDesc gate_desc = gate_descs[expert_id];
        const DeviceNativeVNNIMatrixDesc up_desc = up_descs[expert_id];
        if (!native_vnni_desc_shape_ok<CodebookId>(gate_desc, N, K) ||
            !native_vnni_desc_shape_ok<CodebookId>(up_desc, N, K))
            return;

        const uint8_t *gate_payload_base = gate_desc.payload;
        const uint16_t *gate_scale_base = static_cast<const uint16_t *>(gate_desc.scales);
        const uint16_t *gate_min_base = static_cast<const uint16_t *>(gate_desc.mins);
        const uint32_t *gate_emin_base = static_cast<const uint32_t *>(gate_desc.emins);
        const uint8_t *up_payload_base = up_desc.payload;
        const uint16_t *up_scale_base = static_cast<const uint16_t *>(up_desc.scales);
        const uint16_t *up_min_base = static_cast<const uint16_t *>(up_desc.mins);
        const uint32_t *up_emin_base = static_cast<const uint32_t *>(up_desc.emins);

        __shared__ __align__(16) int32_t s_a[kTileM * 8];
        __shared__ float s_scale[kTileM];

        float gate_acc[kTileM] = {};
        float up_acc[kTileM] = {};

        constexpr int kPayloadBytes =
            llaminar2::cuda_native_vnni::CodebookTraits<CodebookId>::payload_bytes;

#pragma unroll 1
        for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx)
        {
            const size_t linear = static_cast<size_t>(block_idx) * N + static_cast<size_t>(n);

            // Hoisted weight decode (overlaps activation staging + barrier latency).
            int32_t gate_groups[8];
            int32_t up_groups[8];
            const uint8_t *gate_payload = gate_payload_base + linear * kPayloadBytes;
            const uint8_t *up_payload = up_payload_base + linear * kPayloadBytes;
            if (active)
            {
                llaminar2::cuda_native_vnni::decode_groups_vec<CodebookId>(gate_payload, gate_groups);
                llaminar2::cuda_native_vnni::decode_groups_vec<CodebookId>(up_payload, up_groups);
            }

            // Cooperatively stage the activation tile for this K-block into shared memory.
            const int a_count = tokens_in_group * 8;
            for (int idx = threadIdx.x; idx < a_count; idx += kTileN)
            {
                const int m = idx >> 3;
                const int g = idx & 7;
                const int slot = first_slot + m;
                s_a[idx] = reinterpret_cast<const int32_t *>(
                    A_int8 + static_cast<size_t>(slot) * K + block_idx * 32)[g];
            }
            for (int m = threadIdx.x; m < tokens_in_group; m += kTileN)
            {
                const int slot = first_slot + m;
                s_scale[m] = scales_A_blockwise[static_cast<size_t>(slot) * blocks_per_row + block_idx];
            }
            __syncthreads();

            if (active)
            {
                accumulate_prefill_dot_block<CodebookId, kTileM>(
                    gate_groups, gate_scale_base, gate_min_base, gate_emin_base, gate_payload,
                    linear, s_a, /*a4_stride_i32=*/8, s_scale, /*scale_a_stride=*/1,
                    tokens_in_group, gate_acc);
                accumulate_prefill_dot_block<CodebookId, kTileM>(
                    up_groups, up_scale_base, up_min_base, up_emin_base, up_payload,
                    linear, s_a, /*a4_stride_i32=*/8, s_scale, /*scale_a_stride=*/1,
                    tokens_in_group, up_acc);
            }

            __syncthreads();
        }

        // Fused SwiGLU + blockwise int8 quant epilogue. N == intermediate here.
        const int lane = threadIdx.x & 31;
        const int blocks_per_row_out = N / 32;
        const int quant_block = n >> 5; // aligned 32-wide block index along intermediate
#pragma unroll
        for (int m = 0; m < kTileM; ++m)
        {
            if (m >= tokens_in_group)
                break; // tokens_in_group is block-uniform → all lanes break together (shfl-safe).

            const int slot = first_slot + m;
            // Inactive lanes (n >= N) contribute 0 to the warp absmax and skip the write.
            const float value = active ? (silu(gate_acc[m]) * up_acc[m]) : 0.0f;

            float abs_value = fabsf(value);
#pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1)
                abs_value = fmaxf(abs_value, __shfl_xor_sync(0xffffffffu, abs_value, mask));

            const float scale = (abs_value > 0.0f) ? (abs_value / 127.0f) : 1.0f;
            if (active)
            {
                if (lane == 0)
                    swiglu_scales[static_cast<size_t>(slot) * blocks_per_row_out + quant_block] = scale;
                const float q = value / scale;
                swiglu_int8[static_cast<size_t>(slot) * N + n] =
                    static_cast<int8_t>(rintf(fminf(127.0f, fmaxf(-127.0f, q))));
            }
        }
    }

    __global__ void grouped_prefill_swiglu_quantize_blockwise_kernel(
        const float *__restrict__ gate,
        const float *__restrict__ up,
        int8_t *__restrict__ A_int8,
        float *__restrict__ scales_A_blockwise,
        int total_slots,
        int K)
    {
        const int block_idx = blockIdx.x;
        const int slot = blockIdx.y;
        const int lane = threadIdx.x;
        const int col = block_idx * 32 + lane;
        if (slot >= total_slots || lane >= 32 || col >= K)
            return;

        const size_t idx = static_cast<size_t>(slot) * K + col;
        const float g = gate[idx];
        const float value = silu(g) * up[idx];

        float abs_value = fabsf(value);
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            abs_value = fmaxf(abs_value, __shfl_xor_sync(0xffffffffu, abs_value, mask));

        const float scale = (abs_value > 0.0f) ? (abs_value / 127.0f) : 1.0f;
        const int blocks_per_row = (K + 31) / 32;
        if (lane == 0)
            scales_A_blockwise[static_cast<size_t>(slot) * blocks_per_row + block_idx] = scale;

        const float q = value / scale;
        A_int8[idx] = static_cast<int8_t>(rintf(fminf(127.0f, fmaxf(-127.0f, q))));
    }

    template <uint8_t CodebookId, int kTileM, int kTileN>
    __global__ void grouped_native_vnni_down_prefill_kernel(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        const DeviceNativeVNNIMatrixDesc *__restrict__ descs,
        const int *__restrict__ expert_counts,
        const int *__restrict__ expert_offsets,
        const int *__restrict__ active_expert_ids,
        int active_expert_slots,
        float *__restrict__ output,
        int N,
        int K)
    {
        // kTileM tokens processed per block; larger values amortize weight decode (see gate_up).
        const int n = blockIdx.x * kTileN + threadIdx.x;
        const int token_group = blockIdx.y;
        const int expert_id = (active_expert_slots > 0)
                                  ? active_expert_ids[blockIdx.z]
                                  : static_cast<int>(blockIdx.z);
        if (expert_id < 0)
            return;

        const int count = expert_counts[expert_id];
        const int first_token = token_group * kTileM;
        if (first_token >= count)
            return; // Whole-block uniform exit — barrier-safe.

        const int tokens_in_group = min(kTileM, count - first_token);
        const int first_slot = expert_offsets[expert_id] + first_token;
        const int blocks_per_row = K / 32;

        // Per-thread output column may exceed N; such threads must still participate in the
        // cooperative staging + __syncthreads. Guard only weight decode / output writes.
        const bool active = (n < N);

        const DeviceNativeVNNIMatrixDesc desc = descs[expert_id];
        if (!native_vnni_desc_shape_ok<CodebookId>(desc, N, K))
            return;

        const uint8_t *payload_base = desc.payload;
        const uint16_t *scale_base = static_cast<const uint16_t *>(desc.scales);
        const uint16_t *min_base = static_cast<const uint16_t *>(desc.mins);
        const uint32_t *emin_base = static_cast<const uint32_t *>(desc.emins);

        // Staged activation tile (see gate_up kernel) to remove redundant L1 loads.
        // 16-byte aligned for vectorized 2× int4 (128-bit) per-token reads.
        __shared__ __align__(16) int32_t s_a[kTileM * 8];
        __shared__ float s_scale[kTileM];

        float acc[kTileM] = {};

#pragma unroll 1
        for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx)
        {
            // Cooperatively stage the activation tile for this K-block into shared memory.
            const int a_count = tokens_in_group * 8;
            for (int idx = threadIdx.x; idx < a_count; idx += kTileN)
            {
                const int m = idx >> 3;
                const int g = idx & 7;
                const int slot = first_slot + m;
                s_a[idx] = reinterpret_cast<const int32_t *>(
                    A_int8 + static_cast<size_t>(slot) * K + block_idx * 32)[g];
            }
            for (int m = threadIdx.x; m < tokens_in_group; m += kTileN)
            {
                const int slot = first_slot + m;
                s_scale[m] = scales_A_blockwise[static_cast<size_t>(slot) * blocks_per_row + block_idx];
            }
            __syncthreads();

            if (active)
            {
                const size_t linear = static_cast<size_t>(block_idx) * N + static_cast<size_t>(n);
                int32_t packed_groups[8];
                const uint8_t *payload = payload_base +
                    linear * llaminar2::cuda_native_vnni::payload_bytes_for_codebook<CodebookId>();
                llaminar2::cuda_native_vnni::decode_groups_vec<CodebookId>(payload, packed_groups);
                accumulate_prefill_dot_block<CodebookId, kTileM>(
                    packed_groups, scale_base, min_base, emin_base, payload,
                    linear, s_a, /*a4_stride_i32=*/8, s_scale, /*scale_a_stride=*/1,
                    tokens_in_group, acc);
            }

            __syncthreads();
        }

        if (active)
        {
#pragma unroll
            for (int m = 0; m < kTileM; ++m)
            {
                if (m >= tokens_in_group)
                    break;
                const int slot = first_slot + m;
                output[static_cast<size_t>(slot) * N + n] = acc[m];
            }
        }
    }

    __global__ void grouped_prefill_scatter_weighted_kernel(
        float *__restrict__ output,
        const float *__restrict__ expert_output,
        const int *__restrict__ grouped_token_indices,
        const float *__restrict__ grouped_weights,
        int total_slots,
        int d_model)
    {
        constexpr int kTileN = 64;
        const int col = blockIdx.x * kTileN + threadIdx.x;
        const int slot = blockIdx.y;
        if (slot >= total_slots || col >= d_model)
            return;

        const int token = grouped_token_indices[slot];
        const float weight = grouped_weights[slot];
        const float value = expert_output[static_cast<size_t>(slot) * d_model + col];
        atomicAdd(output + static_cast<size_t>(token) * d_model + col, weight * value);
    }

    __global__ void grouped_prefill_scatter_weighted_ordered_kernel(
        float *__restrict__ output,
        const float *__restrict__ expert_output,
        const int *__restrict__ original_to_grouped,
        const float *__restrict__ grouped_weights,
        int seq_len,
        int top_k,
        int d_model)
    {
        constexpr int kTileN = 64;
        const int col = blockIdx.x * kTileN + threadIdx.x;
        const int token = blockIdx.y;
        if (token >= seq_len || col >= d_model)
            return;

        float sum = 0.0f;
#pragma unroll 1
        for (int k = 0; k < top_k; ++k)
        {
            const int original_slot = token * top_k + k;
            const int grouped_slot = original_to_grouped[original_slot];
            if (grouped_slot < 0)
                continue;
            const float weight = grouped_weights[grouped_slot];
            const float value = expert_output[static_cast<size_t>(grouped_slot) * d_model + col];
            sum += weight * value;
        }
        output[static_cast<size_t>(token) * d_model + col] = sum;
    }

    /**
     * @brief Compute a partial dot product over a K-block subrange [b_start, b_end).
     *
     * This is the split-K building block: each caller reduces only a slice of the
     * K dimension so that multiple thread blocks can cooperate on a single output
     * column. Passing [0, K/32) reproduces the full reduction.
     *
     * @param desc                 Native-VNNI weight descriptor (column n of B).
     * @param n                    Output column index into the weight matrix.
     * @param A_int8               Quantized activation row (int8, K elements).
     * @param scales_A_blockwise   Per-K-block activation scales.
     * @param N                    Weight matrix column count (stride for payload).
     * @param K                    Reduction dimension length.
     * @param b_start              First K-block (inclusive) this call reduces.
     * @param b_end                Last K-block (exclusive) this call reduces.
     * @return Partial accumulated dot product over the requested K-block range.
     */
    template <uint8_t CodebookId>
    __device__ __forceinline__ float native_vnni_dot_desc_range(
        const DeviceNativeVNNIMatrixDesc &desc,
        int n,
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        int N,
        int K,
        int b_start,
        int b_end)
    {
        const int blocks_per_row = K / 32;
        if (!A_int8 ||
            !scales_A_blockwise ||
            n < 0 ||
            n >= N ||
            blocks_per_row <= 0 ||
            !native_vnni_desc_shape_ok<CodebookId>(desc, N, K))
        {
            return 0.0f;
        }
        const uint8_t *payload_base = desc.payload;
        const uint16_t *scale_base = static_cast<const uint16_t *>(desc.scales);
        const uint16_t *min_base = static_cast<const uint16_t *>(desc.mins);
        const uint32_t *emin_base = static_cast<const uint32_t *>(desc.emins);
        float acc = 0.0f;

        // Clamp the requested range to the valid K-block span.
        if (b_start < 0)
            b_start = 0;
        if (b_end > blocks_per_row)
            b_end = blocks_per_row;

#pragma unroll 1
        for (int block_idx = b_start; block_idx < b_end; ++block_idx)
        {
            const int32_t *a4 = reinterpret_cast<const int32_t *>(A_int8 + block_idx * 32);
            const size_t linear = static_cast<size_t>(block_idx) * N + static_cast<size_t>(n);
            const uint8_t *payload = payload_base +
                                     linear * llaminar2::cuda_native_vnni::payload_bytes_for_codebook<CodebookId>();

            int32_t packed_groups[8];
            llaminar2::cuda_native_vnni::decode_groups_vec<CodebookId>(payload, packed_groups);

            const float scale_a = scales_A_blockwise[block_idx];
            if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CodebookId>::is_dual_scale)
            {
                int dot_lo = 0;
                int dot_hi = 0;
                int sum_lo = 0;
                int sum_hi = 0;
#pragma unroll
                for (int group = 0; group < 4; ++group)
                {
                    dot_lo = __dp4a(a4[group], packed_groups[group], dot_lo);
                    dot_hi = __dp4a(a4[group + 4], packed_groups[group + 4], dot_hi);
                    sum_lo += llaminar2::cuda_native_vnni::sum_packed_i8(a4[group]);
                    sum_hi += llaminar2::cuda_native_vnni::sum_packed_i8(a4[group + 4]);
                }

                const float scale_lo = llaminar2::cuda_native_vnni::fp16_bits_to_float(scale_base[linear]);
                const float scale_hi = min_base ? llaminar2::cuda_native_vnni::fp16_bits_to_float(min_base[linear]) : 0.0f;
                acc += scale_a * (scale_lo * static_cast<float>(dot_lo) +
                                  scale_hi * static_cast<float>(dot_hi));

                if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CodebookId>::is_dual_scale_asym)
                {
                    const uint32_t emin = emin_base ? emin_base[linear] : 0u;
                    const float min_lo = llaminar2::cuda_native_vnni::fp16_bits_to_float(static_cast<uint16_t>(emin));
                    const float min_hi = llaminar2::cuda_native_vnni::fp16_bits_to_float(static_cast<uint16_t>(emin >> 16));
                    acc += scale_a * (min_lo * static_cast<float>(sum_lo) +
                                      min_hi * static_cast<float>(sum_hi));
                }

                if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CodebookId>::is_iq1_m)
                {
                    constexpr float kIQ1SDelta = 0.125f;
                    const uint8_t qh0 = payload[4];
                    const uint8_t qh1 = payload[5];
                    const int sg0 = llaminar2::cuda_native_vnni::sum_packed_i8(a4[0]) +
                                    llaminar2::cuda_native_vnni::sum_packed_i8(a4[1]);
                    const int sg1 = llaminar2::cuda_native_vnni::sum_packed_i8(a4[2]) +
                                    llaminar2::cuda_native_vnni::sum_packed_i8(a4[3]);
                    const int sg2 = llaminar2::cuda_native_vnni::sum_packed_i8(a4[4]) +
                                    llaminar2::cuda_native_vnni::sum_packed_i8(a4[5]);
                    const int sg3 = llaminar2::cuda_native_vnni::sum_packed_i8(a4[6]) +
                                    llaminar2::cuda_native_vnni::sum_packed_i8(a4[7]);
                    const float delta0 = (qh0 & 0x08) ? -kIQ1SDelta : kIQ1SDelta;
                    const float delta1 = (qh0 & 0x80) ? -kIQ1SDelta : kIQ1SDelta;
                    const float delta2 = (qh1 & 0x08) ? -kIQ1SDelta : kIQ1SDelta;
                    const float delta3 = (qh1 & 0x80) ? -kIQ1SDelta : kIQ1SDelta;
                    acc += scale_a * ((delta0 * static_cast<float>(sg0) + delta1 * static_cast<float>(sg1)) * scale_lo +
                                      (delta2 * static_cast<float>(sg2) + delta3 * static_cast<float>(sg3)) * scale_hi);
                }
            }
            else
            {
                int dot = 0;
                int sum_a = 0;
#pragma unroll
                for (int group = 0; group < 8; ++group)
                {
                    dot = __dp4a(a4[group], packed_groups[group], dot);
                    sum_a += llaminar2::cuda_native_vnni::sum_packed_i8(a4[group]);
                }

                const float scale_b = llaminar2::cuda_native_vnni::fp16_bits_to_float(scale_base[linear]);
                acc += scale_a * scale_b * static_cast<float>(dot);

                if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CodebookId>::is_asymmetric)
                {
                    const float min_b = min_base ? llaminar2::cuda_native_vnni::fp16_bits_to_float(min_base[linear]) : 0.0f;
                    acc += scale_a * min_b * static_cast<float>(sum_a);
                }
            }
        }

        return acc;
    }

    /**
     * @brief Full-K dot product wrapper (reduces all K-blocks). Preserves the
     *        original single-shot reduction semantics for non-split-K callers.
     */
    template <uint8_t CodebookId>
    __device__ __forceinline__ float native_vnni_dot_desc(
        const DeviceNativeVNNIMatrixDesc &desc,
        int n,
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        int N,
        int K)
    {
        return native_vnni_dot_desc_range<CodebookId>(
            desc, n, A_int8, scales_A_blockwise, N, K, 0, K / 32);
    }

    __device__ __forceinline__ float native_vnni_dot_desc_range_dynamic(
        const DeviceNativeVNNIMatrixDesc &desc,
        int n,
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        int N,
        int K,
        int b_start,
        int b_end)
    {
        switch (desc.codebook_id)
        {
        case 0:
            return native_vnni_dot_desc_range<0>(desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        case 4:
            return native_vnni_dot_desc_range<4>(desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        case 5:
            return native_vnni_dot_desc_range<5>(desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        case 6:
            return native_vnni_dot_desc_range<6>(desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        case 7:
            return native_vnni_dot_desc_range<7>(desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        case 8:
            return native_vnni_dot_desc_range<8>(desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        case 9:
            return native_vnni_dot_desc_range<9>(desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        case 10:
            return native_vnni_dot_desc_range<10>(desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        case 11:
            return native_vnni_dot_desc_range<11>(desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        case 12:
            return native_vnni_dot_desc_range<12>(desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        case 13:
            return native_vnni_dot_desc_range<13>(desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        case 14:
            return native_vnni_dot_desc_range<14>(desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        case 15:
            return native_vnni_dot_desc_range<15>(desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        case 16:
            return native_vnni_dot_desc_range<16>(desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        case 17:
            return native_vnni_dot_desc_range<17>(desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        case 19:
            return native_vnni_dot_desc_range<19>(desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        default:
            return 0.0f;
        }
    }

    template <uint8_t CodebookId>
    __device__ __forceinline__ float native_vnni_dot_desc_range_dispatch(
        const DeviceNativeVNNIMatrixDesc &desc,
        int n,
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        int N,
        int K,
        int b_start,
        int b_end)
    {
        if constexpr (CodebookId == kMixedCodebookSentinel)
        {
            return native_vnni_dot_desc_range_dynamic(
                desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        }
        else
        {
            return native_vnni_dot_desc_range<CodebookId>(
                desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        }
    }

    template <uint8_t CodebookId>
    __device__ __forceinline__ float native_vnni_dot_desc_dispatch(
        const DeviceNativeVNNIMatrixDesc &desc,
        int n,
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        int N,
        int K)
    {
        return native_vnni_dot_desc_range_dispatch<CodebookId>(
            desc, n, A_int8, scales_A_blockwise, N, K, 0, K / 32);
    }

    /**
     * @brief Split-K scatter kernel for grouped gate/up decode projection.
     *
     * Each block reduces one K-partition of one output column for one expert
     * slot, writing its partial to a [slot][k_part][N] scratch buffer. A separate
     * reduce kernel sums the partials. This raises occupancy versus the single
     * full-K kernel (which launches only num_active * ceil(N/64) blocks) by
     * multiplying the block count by k_partitions, exposing far more warps to
     * hide global-memory latency on the weight payload loads.
     *
     * Grid:  ((N + 63)/64, k_partitions, num_active)   Block: (64)
     */
    template <uint8_t CodebookId>
    __global__ void grouped_native_vnni_gate_up_kpart_decode_kernel(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        const DeviceNativeVNNIMatrixDesc *__restrict__ gate_descs,
        const DeviceNativeVNNIMatrixDesc *__restrict__ up_descs,
        const int *__restrict__ expert_ids,
        float *__restrict__ gate_partials,
        float *__restrict__ up_partials,
        int num_active,
        int N,
        int K,
        int num_experts,
        int k_partitions)
    {
        constexpr int kTileN = 64;
        const int n = blockIdx.x * kTileN + threadIdx.x;
        const int k_part = blockIdx.y;
        const int slot = blockIdx.z;
        if (slot >= num_active || k_part >= k_partitions || n >= N)
            return;

        // Linear index into the [num_active][k_partitions][N] partials buffer.
        const size_t partial_index =
            (static_cast<size_t>(slot) * static_cast<size_t>(k_partitions) +
             static_cast<size_t>(k_part)) *
                static_cast<size_t>(N) +
            static_cast<size_t>(n);

        const int expert_id = expert_ids[slot];
        if (expert_id < 0)
            return;
        assert(expert_id < num_experts);
        if (expert_id >= num_experts)
        {
            gate_partials[partial_index] = 0.0f;
            up_partials[partial_index] = 0.0f;
            return;
        }

        // Evenly split the K-blocks across the k_partitions; this block owns
        // [b_start, b_end).
        const int blocks_per_row = K / 32;
        const int blocks_per_part = (blocks_per_row + k_partitions - 1) / k_partitions;
        const int b_start = k_part * blocks_per_part;
        int b_end = b_start + blocks_per_part;
        if (b_end > blocks_per_row)
            b_end = blocks_per_row;

        if (b_start >= b_end)
        {
            gate_partials[partial_index] = 0.0f;
            up_partials[partial_index] = 0.0f;
            return;
        }

        const DeviceNativeVNNIMatrixDesc gate_desc = gate_descs[expert_id];
        const DeviceNativeVNNIMatrixDesc up_desc = up_descs[expert_id];
        gate_partials[partial_index] = native_vnni_dot_desc_range_dispatch<CodebookId>(
            gate_desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        up_partials[partial_index] = native_vnni_dot_desc_range_dispatch<CodebookId>(
            up_desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
    }

    /**
     * @brief Reduce K-partition partials into the final grouped gate/up outputs.
     *
     * Sums the k_partitions partial contributions for each (slot, n) produced by
     * grouped_native_vnni_gate_up_kpart_decode_kernel and writes the result to
     * the per-slot gate/up output buffers.
     *
     * Grid: ((N + 63)/64, num_active)   Block: (64)
     */
    __global__ void grouped_native_vnni_gate_up_kpart_reduce_kernel(
        const float *__restrict__ gate_partials,
        const float *__restrict__ up_partials,
        const int *__restrict__ expert_ids,
        float *const *__restrict__ gate_outputs,
        float *const *__restrict__ up_outputs,
        int num_active,
        int N,
        int k_partitions)
    {
        constexpr int kTileN = 64;
        const int n = blockIdx.x * kTileN + threadIdx.x;
        const int slot = blockIdx.y;
        if (slot >= num_active || n >= N)
            return;
        if (expert_ids && expert_ids[slot] < 0)
            return;

        float gate_sum = 0.0f;
        float up_sum = 0.0f;
        const size_t slot_base =
            static_cast<size_t>(slot) * static_cast<size_t>(k_partitions) * static_cast<size_t>(N);
        for (int k_part = 0; k_part < k_partitions; ++k_part)
        {
            const size_t idx =
                slot_base + static_cast<size_t>(k_part) * static_cast<size_t>(N) + static_cast<size_t>(n);
            gate_sum += gate_partials[idx];
            up_sum += up_partials[idx];
        }

        gate_outputs[slot][n] = gate_sum;
        up_outputs[slot][n] = up_sum;
    }

    /**
     * @brief Split-K gate/up projection for tiny grouped verifier-prefill rows.
     *
     * The ordinary grouped-prefill gate/up kernel launches one full-K CTA for
     * each (expert, M-tile, N-tile).  For MTP verifier batches M is only 2..4,
     * so that leaves the GPU under-filled even though each CTA has a long
     * quantized-weight latency chain.  This variant mirrors the single-row
     * decode split-K strategy: each CTA owns one K partition and writes partial
     * gate/up sums into [grouped_slot][k_partition][N].  A second tiny reduce
     * kernel sums partitions back into the normal grouped-prefill FP32
     * gate/up scratch buffers.
     */
    template <uint8_t CodebookId, int kTileM>
    __global__ void grouped_native_vnni_gate_up_prefill_kpart_kernel(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        const DeviceNativeVNNIMatrixDesc *__restrict__ gate_descs,
        const DeviceNativeVNNIMatrixDesc *__restrict__ up_descs,
        const int *__restrict__ expert_counts,
        const int *__restrict__ expert_offsets,
        const int *__restrict__ active_expert_ids,
        int active_expert_slots,
        float *__restrict__ gate_partials,
        float *__restrict__ up_partials,
        int N,
        int K,
        int k_partitions)
    {
        constexpr int kTileN = 64;
        const int n = blockIdx.x * kTileN + threadIdx.x;
        const int token_group = blockIdx.y;
        const int expert_grid_index = blockIdx.z / k_partitions;
        const int k_part = blockIdx.z - expert_grid_index * k_partitions;
        if (n >= N || k_part >= k_partitions)
            return;

        const int expert_id = (active_expert_slots > 0)
                                  ? active_expert_ids[expert_grid_index]
                                  : expert_grid_index;
        if (expert_id < 0)
            return;

        const int count = expert_counts[expert_id];
        const int first_token = token_group * kTileM;
        if (first_token >= count)
            return;

        const int tokens_in_group = min(kTileM, count - first_token);
        const int first_slot = expert_offsets[expert_id] + first_token;
        const int blocks_per_row = K / 32;
        const int blocks_per_part = (blocks_per_row + k_partitions - 1) / k_partitions;
        const int b_start = k_part * blocks_per_part;
        int b_end = b_start + blocks_per_part;
        if (b_end > blocks_per_row)
            b_end = blocks_per_row;

        const DeviceNativeVNNIMatrixDesc gate_desc = gate_descs[expert_id];
        const DeviceNativeVNNIMatrixDesc up_desc = up_descs[expert_id];
        if (!native_vnni_desc_shape_ok<CodebookId>(gate_desc, N, K) ||
            !native_vnni_desc_shape_ok<CodebookId>(up_desc, N, K))
            return;

#pragma unroll
        for (int m = 0; m < kTileM; ++m)
        {
            if (m >= tokens_in_group)
                break;
            const int slot = first_slot + m;
            const size_t partial_index =
                (static_cast<size_t>(slot) * static_cast<size_t>(k_partitions) +
                 static_cast<size_t>(k_part)) *
                    static_cast<size_t>(N) +
                static_cast<size_t>(n);
            if (b_start >= b_end)
            {
                gate_partials[partial_index] = 0.0f;
                up_partials[partial_index] = 0.0f;
                continue;
            }
            const int8_t *slot_A = A_int8 + static_cast<size_t>(slot) * K;
            const float *slot_scales =
                scales_A_blockwise + static_cast<size_t>(slot) * blocks_per_row;
            gate_partials[partial_index] = native_vnni_dot_desc_range<CodebookId>(
                gate_desc, n, slot_A, slot_scales, N, K, b_start, b_end);
            up_partials[partial_index] = native_vnni_dot_desc_range<CodebookId>(
                up_desc, n, slot_A, slot_scales, N, K, b_start, b_end);
        }
    }

    /**
     * @brief Reduce split-K gate/up partials and quantize SwiGLU in one pass.
     *
     * Each lane sums k-partitions in the same order used by the former
     * reduce-then-quantize sequence, computes the same SwiGLU value, and writes
     * the same blockwise INT8 row layout for the down-projection kernel.  The
     * benefit is architectural: verifier graphs avoid a launch and a global
     * FP32 gate/up scratch round trip.
     */
    __global__ void grouped_native_vnni_gate_up_prefill_kpart_reduce_swiglu_kernel(
        const float *__restrict__ gate_partials,
        const float *__restrict__ up_partials,
        int8_t *__restrict__ swiglu_int8,
        float *__restrict__ swiglu_scales,
        int total_slots,
        int N,
        int k_partitions)
    {
        constexpr int kTileN = 32;
        const int lane = threadIdx.x;
        const int block_idx = blockIdx.x;
        const int slot = blockIdx.y;
        const int n = block_idx * kTileN + lane;
        if (slot >= total_slots)
            return;

        const bool active = n < N;
        const size_t slot_base =
            static_cast<size_t>(slot) * static_cast<size_t>(k_partitions) * static_cast<size_t>(N);

        float gate_sum = 0.0f;
        float up_sum = 0.0f;
        if (active)
        {
            for (int k_part = 0; k_part < k_partitions; ++k_part)
            {
                const size_t idx =
                    slot_base + static_cast<size_t>(k_part) * static_cast<size_t>(N) + static_cast<size_t>(n);
                gate_sum += gate_partials[idx];
                up_sum += up_partials[idx];
            }
        }

        const float value = active ? (silu(gate_sum) * up_sum) : 0.0f;
        float abs_value = fabsf(value);
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            abs_value = fmaxf(abs_value, __shfl_xor_sync(0xffffffffu, abs_value, mask));

        const float scale = (abs_value > 0.0f) ? (abs_value / 127.0f) : 1.0f;
        const int blocks_per_row = (N + kTileN - 1) / kTileN;
        if (lane == 0)
            swiglu_scales[static_cast<size_t>(slot) * static_cast<size_t>(blocks_per_row) +
                          static_cast<size_t>(block_idx)] = scale;

        if (active)
        {
            const float q = value / scale;
            swiglu_int8[static_cast<size_t>(slot) * static_cast<size_t>(N) + static_cast<size_t>(n)] =
                static_cast<int8_t>(rintf(fminf(127.0f, fmaxf(-127.0f, q))));
        }
    }

    template <uint8_t CodebookId>
    __global__ void grouped_native_vnni_gate_up_decode_kernel(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        const DeviceNativeVNNIMatrixDesc *__restrict__ gate_descs,
        const DeviceNativeVNNIMatrixDesc *__restrict__ up_descs,
        const int *__restrict__ expert_ids,
        float *const *__restrict__ gate_outputs,
        float *const *__restrict__ up_outputs,
        int num_active,
        int N,
        int K)
    {
        constexpr int kTileN = 64;
        const int n = blockIdx.x * kTileN + threadIdx.x;
        const int slot = blockIdx.y;
        if (slot >= num_active || n >= N)
            return;

        const int expert_id = expert_ids[slot];
        if (expert_id < 0)
            return;

        const DeviceNativeVNNIMatrixDesc gate_desc = gate_descs[expert_id];
        const DeviceNativeVNNIMatrixDesc up_desc = up_descs[expert_id];
        gate_outputs[slot][n] = native_vnni_dot_desc_dispatch<CodebookId>(
            gate_desc, n, A_int8, scales_A_blockwise, N, K);
        up_outputs[slot][n] = native_vnni_dot_desc_dispatch<CodebookId>(
            up_desc, n, A_int8, scales_A_blockwise, N, K);
    }

    template <uint8_t CodebookId>
    __global__ void grouped_native_vnni_down_decode_kernel(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        const DeviceNativeVNNIMatrixDesc *__restrict__ descs,
        const int *__restrict__ expert_ids,
        const float *__restrict__ route_weights,
        float *__restrict__ output,
        int num_active,
        int N,
        int K)
    {
        constexpr int kTileN = 64;
        const int n = blockIdx.x * kTileN + threadIdx.x;
        if (n >= N)
            return;

        const int blocks_per_row = K / 32;
        float total = 0.0f;
#pragma unroll 1
        for (int slot = 0; slot < num_active; ++slot)
        {
            const int expert_id = expert_ids[slot];
            if (expert_id < 0)
                continue;

            const DeviceNativeVNNIMatrixDesc desc = descs[expert_id];
            const int8_t *slot_A = A_int8 + static_cast<size_t>(slot) * K;
            const float *slot_scales = scales_A_blockwise + static_cast<size_t>(slot) * blocks_per_row;
            const float expert_value = native_vnni_dot_desc_dispatch<CodebookId>(
                desc, n, slot_A, slot_scales, N, K);
            total += route_weights[slot] * expert_value;
        }
        output[n] = total;
    }

    /**
     * @brief Split-K scatter kernel for the grouped SwiGLU down projection.
     *
     * Each block reduces one K-partition of one output column, summing the
     * route-weighted partial contributions of all active experts for that
     * K-range, and writes the partial to a [k_partitions][N] scratch buffer.
     * A separate reduce kernel sums the partials. The down projection launches
     * only ceil(N/64) blocks in the serial path (N = d_model), leaving the GPU
     * heavily under-occupied; multiplying the block count by k_partitions
     * exposes enough warps to hide the weight-payload global-memory latency.
     *
     * The expert sum and the K-partition sum commute because each is a linear
     * accumulation, so summing experts within a K-range and then summing the
     * K-ranges yields the same result as the serial full-K expert sum.
     *
     * Grid: ((N + 63)/64, k_partitions)   Block: (64)
     */
    template <uint8_t CodebookId>
    __global__ void grouped_native_vnni_down_kpart_decode_kernel(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        const DeviceNativeVNNIMatrixDesc *__restrict__ descs,
        const int *__restrict__ expert_ids,
        const float *__restrict__ route_weights,
        float *__restrict__ partials,
        int num_active,
        int N,
        int K,
        int num_experts,
        int k_partitions)
    {
        constexpr int kTileN = 64;
        const int n = blockIdx.x * kTileN + threadIdx.x;
        const int k_part = blockIdx.y;
        if (k_part >= k_partitions || n >= N)
            return;

        // Linear index into the [k_partitions][N] partials buffer.
        const size_t partial_index =
            static_cast<size_t>(k_part) * static_cast<size_t>(N) + static_cast<size_t>(n);

        // Evenly split the K-blocks across the partitions; this block owns
        // [b_start, b_end).
        const int blocks_per_row = K / 32;
        const int blocks_per_part = (blocks_per_row + k_partitions - 1) / k_partitions;
        const int b_start = k_part * blocks_per_part;
        int b_end = b_start + blocks_per_part;
        if (b_end > blocks_per_row)
            b_end = blocks_per_row;
        if (b_start >= b_end)
        {
            partials[partial_index] = 0.0f;
            return;
        }

        // Accumulate the route-weighted expert contributions for this K-range.
        float total = 0.0f;
#pragma unroll 1
        for (int slot = 0; slot < num_active; ++slot)
        {
            const int expert_id = expert_ids[slot];
            if (expert_id < 0)
                continue;
            assert(expert_id < num_experts);
            if (expert_id >= num_experts)
                continue;

            const DeviceNativeVNNIMatrixDesc desc = descs[expert_id];
            const int8_t *slot_A = A_int8 + static_cast<size_t>(slot) * K;
            const float *slot_scales = scales_A_blockwise + static_cast<size_t>(slot) * blocks_per_row;
            const float expert_value = native_vnni_dot_desc_range_dispatch<CodebookId>(
                desc, n, slot_A, slot_scales, N, K, b_start, b_end);
            total += route_weights[slot] * expert_value;
        }
        partials[partial_index] = total;
    }

    /**
     * @brief Reduce K-partition partials into the final grouped down output.
     *
     * Sums the k_partitions partial contributions for each output column n
     * produced by grouped_native_vnni_down_kpart_decode_kernel.
     *
     * Grid: ((N + 63)/64)   Block: (64)
     */
    __global__ void grouped_native_vnni_down_kpart_reduce_kernel(
        const float *__restrict__ partials,
        float *__restrict__ output,
        int N,
        int k_partitions)
    {
        constexpr int kTileN = 64;
        const int n = blockIdx.x * kTileN + threadIdx.x;
        if (n >= N)
            return;

        float sum = 0.0f;
        for (int k_part = 0; k_part < k_partitions; ++k_part)
            sum += partials[static_cast<size_t>(k_part) * static_cast<size_t>(N) + static_cast<size_t>(n)];
        output[n] = sum;
    }

    template <uint8_t CodebookId>
    __global__ void grouped_native_vnni_gate_up_decode_runtime_kernel(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        const DeviceMoELayerRuntimeView *__restrict__ runtime,
        const int *__restrict__ expert_ids,
        float *const *__restrict__ gate_outputs,
        float *const *__restrict__ up_outputs,
        int num_active,
        int N,
        int K,
        int num_experts)
    {
        constexpr int kTileN = 64;
        const int n = blockIdx.x * kTileN + threadIdx.x;
        const int slot = blockIdx.y;
        if (slot >= num_active || n >= N)
            return;

        const int expert_id = expert_ids[slot];
        if (expert_id < 0 || expert_id >= num_experts)
        {
            gate_outputs[slot][n] = 0.0f;
            up_outputs[slot][n] = 0.0f;
            return;
        }

        const DeviceMoEPlacementBankView &bank = runtime->banks[runtime->active_bank];
        const DeviceNativeVNNIMatrixDesc gate_desc = bank.experts[expert_id].gate;
        const DeviceNativeVNNIMatrixDesc up_desc = bank.experts[expert_id].up;
        gate_outputs[slot][n] = native_vnni_dot_desc_dispatch<CodebookId>(
            gate_desc, n, A_int8, scales_A_blockwise, N, K);
        up_outputs[slot][n] = native_vnni_dot_desc_dispatch<CodebookId>(
            up_desc, n, A_int8, scales_A_blockwise, N, K);
    }

    template <uint8_t CodebookId>
    __global__ void grouped_native_vnni_gate_up_kpart_decode_runtime_kernel(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        const DeviceMoELayerRuntimeView *__restrict__ runtime,
        const int *__restrict__ expert_ids,
        float *__restrict__ gate_partials,
        float *__restrict__ up_partials,
        int num_active,
        int N,
        int K,
        int num_experts,
        int k_partitions)
    {
        constexpr int kTileN = 64;
        const int n = blockIdx.x * kTileN + threadIdx.x;
        const int k_part = blockIdx.y;
        const int slot = blockIdx.z;
        if (slot >= num_active || k_part >= k_partitions || n >= N)
            return;

        const size_t partial_index =
            (static_cast<size_t>(slot) * static_cast<size_t>(k_partitions) +
             static_cast<size_t>(k_part)) *
                static_cast<size_t>(N) +
            static_cast<size_t>(n);

        const int expert_id = expert_ids[slot];
        if (expert_id < 0 || expert_id >= num_experts)
        {
            gate_partials[partial_index] = 0.0f;
            up_partials[partial_index] = 0.0f;
            return;
        }

        const int blocks_per_row = K / 32;
        const int blocks_per_part = (blocks_per_row + k_partitions - 1) / k_partitions;
        const int b_start = k_part * blocks_per_part;
        int b_end = b_start + blocks_per_part;
        if (b_end > blocks_per_row)
            b_end = blocks_per_row;
        if (b_start >= b_end)
        {
            gate_partials[partial_index] = 0.0f;
            up_partials[partial_index] = 0.0f;
            return;
        }

        const DeviceMoEPlacementBankView &bank = runtime->banks[runtime->active_bank];
        const DeviceNativeVNNIMatrixDesc gate_desc = bank.experts[expert_id].gate;
        const DeviceNativeVNNIMatrixDesc up_desc = bank.experts[expert_id].up;
        gate_partials[partial_index] = native_vnni_dot_desc_range_dispatch<CodebookId>(
            gate_desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        up_partials[partial_index] = native_vnni_dot_desc_range_dispatch<CodebookId>(
            up_desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
    }

    template <uint8_t CodebookId>
    __global__ void grouped_native_vnni_down_decode_runtime_kernel(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        const DeviceMoELayerRuntimeView *__restrict__ runtime,
        const int *__restrict__ expert_ids,
        const float *__restrict__ route_weights,
        float *__restrict__ output,
        int num_active,
        int N,
        int K,
        int num_experts)
    {
        constexpr int kTileN = 64;
        const int n = blockIdx.x * kTileN + threadIdx.x;
        if (n >= N)
            return;

        const int blocks_per_row = K / 32;
        const DeviceMoEPlacementBankView &bank = runtime->banks[runtime->active_bank];
        float total = 0.0f;
#pragma unroll 1
        for (int slot = 0; slot < num_active; ++slot)
        {
            const int expert_id = expert_ids[slot];
            if (expert_id < 0 || expert_id >= num_experts)
                continue;

            const DeviceNativeVNNIMatrixDesc desc = bank.experts[expert_id].down;
            const int8_t *slot_A = A_int8 + static_cast<size_t>(slot) * K;
            const float *slot_scales = scales_A_blockwise + static_cast<size_t>(slot) * blocks_per_row;
            const float expert_value = native_vnni_dot_desc_dispatch<CodebookId>(
                desc, n, slot_A, slot_scales, N, K);
            total += route_weights[slot] * expert_value;
        }
        output[n] = total;
    }

    template <uint8_t CodebookId>
    __global__ void grouped_native_vnni_down_kpart_decode_runtime_kernel(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        const DeviceMoELayerRuntimeView *__restrict__ runtime,
        const int *__restrict__ expert_ids,
        const float *__restrict__ route_weights,
        float *__restrict__ partials,
        int num_active,
        int N,
        int K,
        int num_experts,
        int k_partitions)
    {
        constexpr int kTileN = 64;
        const int n = blockIdx.x * kTileN + threadIdx.x;
        const int k_part = blockIdx.y;
        if (k_part >= k_partitions || n >= N)
            return;

        const size_t partial_index =
            static_cast<size_t>(k_part) * static_cast<size_t>(N) + static_cast<size_t>(n);

        const int blocks_per_row = K / 32;
        const int blocks_per_part = (blocks_per_row + k_partitions - 1) / k_partitions;
        const int b_start = k_part * blocks_per_part;
        int b_end = b_start + blocks_per_part;
        if (b_end > blocks_per_row)
            b_end = blocks_per_row;
        if (b_start >= b_end)
        {
            partials[partial_index] = 0.0f;
            return;
        }

        const DeviceMoEPlacementBankView &bank = runtime->banks[runtime->active_bank];
        float total = 0.0f;
#pragma unroll 1
        for (int slot = 0; slot < num_active; ++slot)
        {
            const int expert_id = expert_ids[slot];
            if (expert_id < 0 || expert_id >= num_experts)
                continue;

            const DeviceNativeVNNIMatrixDesc desc = bank.experts[expert_id].down;
            const int8_t *slot_A = A_int8 + static_cast<size_t>(slot) * K;
            const float *slot_scales = scales_A_blockwise + static_cast<size_t>(slot) * blocks_per_row;
            const float expert_value = native_vnni_dot_desc_range_dispatch<CodebookId>(
                desc, n, slot_A, slot_scales, N, K, b_start, b_end);
            total += route_weights[slot] * expert_value;
        }
        partials[partial_index] = total;
    }

    int blocksFor(int count)
    {
        return (count + kThreads - 1) / kThreads;
    }

    int select_grouped_prefill_tile_m(int requested_tile_m, int max_tokens_per_expert)
    {
        switch (requested_tile_m)
        {
        case 2:
        case 4:
        case 8:
        case 16:
            return requested_tile_m;
        default:
            break;
        }

        // Tiny MTP verifier groups (M=2..4) are fastest with the compact
        // two-row template on CUDA. Wider M tiles add per-block work without
        // improving occupancy for these active expert groups.
        if (max_tokens_per_expert <= 4)
            return 2;
        if (max_tokens_per_expert <= 8)
            return 8;
        return 16;
    }
}

extern "C"
{
    bool cudaMoE_quantize_router_gate_q8(
        const float *gate_weights, int8_t *gate_weights_q8, float *gate_scales,
        int d_model, int num_experts,
        int device_idx, void *stream)
    {
        if (!gate_weights || !gate_weights_q8 || !gate_scales || !stream ||
            d_model <= 0 || num_experts <= 0 || (d_model % 32) != 0)
        {
            std::fprintf(stderr, "[cudaMoE_quantize_router_gate_q8] invalid arguments\n");
            return false;
        }

        cudaSetDevice(device_idx);
        const int blocks_per_row = d_model / 32;
        router_gate_quantize_q8_kernel<<<dim3(num_experts, blocks_per_row), dim3(32), 0,
                                          static_cast<cudaStream_t>(stream)>>>(
            gate_weights, gate_weights_q8, gate_scales, d_model, num_experts);
        return finishLaunch("cudaMoE_quantize_router_gate_q8");
    }

    bool cudaMoE_gate_logits_single_token_q8_weights(
        const float *hidden, int8_t *hidden_q8, float *hidden_scales,
        const int8_t *gate_weights_q8, const float *gate_scales, float *logits,
        int d_model, int num_experts,
        int device_idx, void *stream)
    {
        if (!hidden || !hidden_q8 || !hidden_scales ||
            !gate_weights_q8 || !gate_scales || !logits || !stream ||
            d_model <= 0 || num_experts <= 0 || (d_model % 32) != 0)
        {
            std::fprintf(stderr, "[cudaMoE_gate_logits_single_token_q8_weights] invalid arguments\n");
            return false;
        }

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        const int blocks_per_row = d_model / 32;
        grouped_hidden_quantize_blockwise_kernel<<<blocks_per_row, 32, 0, cuda_stream>>>(
            hidden, hidden_q8, hidden_scales, d_model);
        if (!finishLaunch("cudaMoE_router_hidden_quantize_q8"))
            return false;

        constexpr int kRouterThreads = 128;
        router_gate_logits_single_token_q8_kernel<<<num_experts, kRouterThreads, 0, cuda_stream>>>(
            hidden_q8, hidden_scales, gate_weights_q8, gate_scales, logits, d_model, num_experts);
        return finishLaunch("cudaMoE_gate_logits_single_token_q8_weights");
    }

    bool cudaMoE_route_logits(const float *hidden, const float *gate_weights, float *logits,
                              int seq_len, int d_model, int num_experts,
                              int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);

        // PREFILL: with many tokens this is a genuine GEMM. The naive block-per-
        // (expert,token) kernel is L2-bandwidth-bound (re-reads hidden E× and gate S×).
        // Above this token threshold the tiled SGEMM (smem reuse) is a large win; below
        // it (decode, seq_len==1) the warp-reduction kernel keeps full SM coverage.
        constexpr int kRouteTiledMinTokens = 16;
        if (seq_len >= kRouteTiledMinTokens)
        {
            // Tile geometry must match the template instantiation below. BM=BN=64
            // (256-thread block) empirically beats smaller tiles here: although it
            // yields only 44 blocks (occupancy-bound on this M=679,N=256 GEMM), the
            // 256-thread block's load efficiency and ILP outperform 32×32 (1918) and
            // 64×32 (1931) configs that produce more blocks but fewer threads each.
            constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
            constexpr int kTiledThreads = (BM / TM) * (BN / TN); // 16×16 = 256
            dim3 grid((num_experts + BN - 1) / BN, (seq_len + BM - 1) / BM);
            route_logits_tiled_kernel<BM, BN, BK, TM, TN>
                <<<grid, kTiledThreads, 0, static_cast<cudaStream_t>(stream)>>>(
                    hidden, gate_weights, logits, seq_len, d_model, num_experts);
            return finishLaunch("cudaMoE_route_logits_tiled");
        }

        dim3 grid(num_experts, seq_len);
        route_logits_kernel<<<grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            hidden, gate_weights, logits, seq_len, d_model, num_experts);
        return finishLaunch("cudaMoE_route_logits");
    }

    bool cudaMoE_route_logits_bf16(const float *hidden, const void *gate_weights, float *logits,
                                   int seq_len, int d_model, int num_experts,
                                   int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        dim3 grid(num_experts, seq_len);
        route_logits_bf16_kernel<<<grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            hidden, static_cast<const __nv_bfloat16 *>(gate_weights), logits,
            seq_len, d_model, num_experts);
        return finishLaunch("cudaMoE_route_logits_bf16");
    }

    bool cudaMoE_softmax_topk(float *logits, int *expert_indices, float *expert_weights,
                              int seq_len, int num_experts, int top_k, bool normalize_weights,
                              int device_idx, void *stream,
                              const int *device_effective_seq_len)
    {
        if (num_experts > kMaxExperts || top_k > kMaxTopK)
            return false;
        cudaSetDevice(device_idx);
        softmax_topk_kernel<<<seq_len, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            logits, expert_indices, expert_weights, seq_len, num_experts, top_k,
            normalize_weights, device_effective_seq_len);
        return finishLaunch("cudaMoE_softmax_topk");
    }

    bool cudaMoE_softmax_topk_decode_runtime(float *logits,
                                             void *runtime_layer,
                                             float *legacy_indices, float *legacy_weights,
                                             int num_experts, int top_k, bool normalize_weights,
                                             bool write_legacy_outputs, bool update_runtime_histogram,
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
        if (!runtime_layer ||
            num_experts <= 0 ||
            num_experts > kDeviceMoEMaxExperts ||
            top_k <= 0 ||
            top_k > kMaxTopK ||
            top_k > num_experts)
            return false;
        cudaSetDevice(device_idx);
        DeviceMoERebalanceConfigView rebalance_cfg{};
        if (rebalance_config)
            rebalance_cfg = *static_cast<const DeviceMoERebalanceConfigView *>(rebalance_config);
        softmax_topk_decode_runtime_kernel<<<1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            logits, static_cast<DeviceMoELayerRuntimeView *>(runtime_layer),
            static_cast<DeviceMoELayerRuntimeView *>(runtime_layers),
            legacy_indices, legacy_weights, num_experts, top_k, normalize_weights,
            write_legacy_outputs, update_runtime_histogram,
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
        return finishLaunch("cudaMoE_softmax_topk_decode_runtime");
    }

    bool cudaMoE_decode_route_select_runtime(const int *expert_indices, const float *expert_weights,
                                             void *runtime_layer,
                                             float *legacy_indices, float *legacy_weights,
                                             int num_experts, int top_k, bool write_legacy_outputs,
                                             bool update_runtime_histogram, int device_idx, void *stream)
    {
        if (!runtime_layer || num_experts > kDeviceMoEMaxExperts || top_k > kMaxTopK)
            return false;
        cudaSetDevice(device_idx);
        decode_route_select_runtime_kernel<<<1, kMaxTopK, 0, static_cast<cudaStream_t>(stream)>>>(
            expert_indices, expert_weights, static_cast<DeviceMoELayerRuntimeView *>(runtime_layer),
            legacy_indices, legacy_weights, num_experts, top_k, write_legacy_outputs, update_runtime_histogram);
        return finishLaunch("cudaMoE_decode_route_select_runtime");
    }

    bool cudaMoE_device_rebalance_controller(
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
        cudaSetDevice(device_idx);
        const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
        device_rebalance_controller_kernel<<<1, kDeviceMoEMaxExperts, 0, static_cast<cudaStream_t>(stream)>>>(
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
        return finishLaunch("cudaMoE_device_rebalance_controller");
    }

    bool cudaMoE_pack_rebalance_histograms(
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
        cudaSetDevice(device_idx);
        const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
        const uint32_t window_count =
            cfg.layer_window_count == 0u
                ? cfg.num_layers
                : (cfg.layer_window_count < cfg.num_layers ? cfg.layer_window_count : cfg.num_layers);
        const uint32_t wave_count =
            cfg.layer_wave_count == 0u
                ? window_count
                : (cfg.layer_wave_count < window_count ? cfg.layer_wave_count : window_count);
        const int total = static_cast<int>(wave_count * cfg.num_experts);
        const int blocks = (total + kThreads - 1) / kThreads;
        pack_rebalance_histograms_kernel<<<blocks, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const DeviceMoELayerRuntimeView *>(runtime_layers),
            local_histograms,
            cfg,
            static_cast<const DeviceMoERebalanceWaveStateView *>(wave_state),
            static_cast<const DeviceMoERebalanceGraphControllerStateView *>(controller_state),
            command_buffer_count);
        return finishLaunch("cudaMoE_pack_rebalance_histograms");
    }

    bool cudaMoE_pack_rebalance_directory(
        void *runtime_layers,
        void *local_directory,
        const void *config,
        int device_idx,
        void *stream)
    {
        if (!runtime_layers || !local_directory || !config || !stream)
            return false;
        cudaSetDevice(device_idx);
        const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
        const int total = static_cast<int>(cfg.num_layers * cfg.num_experts);
        const int blocks = (total + kThreads - 1) / kThreads;
        pack_rebalance_directory_kernel<<<blocks, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const DeviceMoELayerRuntimeView *>(runtime_layers),
            static_cast<DeviceMoEExpertDirectoryEntryView *>(local_directory),
            cfg);
        return finishLaunch("cudaMoE_pack_rebalance_directory");
    }

    bool cudaMoE_pack_rebalance_source_descriptors(
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
        cudaSetDevice(device_idx);
        const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
        const uint32_t metadata_buffer_count =
            command_buffer_count <= 1u ? 1u : (command_buffer_count < 2u ? command_buffer_count : 2u);
        const unsigned long long total =
            static_cast<unsigned long long>(cfg.participant_count) *
            static_cast<unsigned long long>(metadata_buffer_count) *
            static_cast<unsigned long long>(plan_capacity);
        const int blocks =
            static_cast<int>((total + static_cast<unsigned long long>(kThreads) - 1ULL) /
                             static_cast<unsigned long long>(kThreads));
        if (blocks > 0)
        {
            pack_rebalance_source_descriptors_kernel<<<blocks, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
                static_cast<const DeviceMoELayerRuntimeView *>(runtime_layers),
                static_cast<const DeviceMoERebalancePlanEntryView *>(gathered_plan_entries),
                static_cast<const DeviceMoERebalanceCommandBufferHeaderView *>(gathered_command_headers),
                plan_capacity,
                static_cast<DeviceMoEExpertDirectoryEntryView *>(local_source_descriptors),
                cfg,
                static_cast<DeviceMoERebalanceGraphControllerStateView *>(controller_state),
                command_buffer_count);
        }
        return finishLaunch("cudaMoE_pack_rebalance_source_descriptors");
    }

    bool cudaMoE_project_rebalance_domain_commands(
        const void *gathered_plan_entries,
        const void *gathered_command_headers,
        uint32_t plan_capacity,
        void *local_plan_entries,
        void *local_command_headers,
        const void *config,
        void *status,
        uint32_t payload_slot_capacity,
        uint32_t command_buffer_count,
        int device_idx,
        void *stream)
    {
        if (!gathered_plan_entries ||
            !gathered_command_headers ||
            !local_plan_entries ||
            !local_command_headers ||
            !config ||
            !stream ||
            plan_capacity == 0u)
        {
            return false;
        }
        cudaSetDevice(device_idx);
        const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
        project_rebalance_domain_commands_kernel<<<1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const DeviceMoERebalancePlanEntryView *>(gathered_plan_entries),
            static_cast<const DeviceMoERebalanceCommandBufferHeaderView *>(gathered_command_headers),
            plan_capacity,
            static_cast<DeviceMoERebalancePlanEntryView *>(local_plan_entries),
            static_cast<DeviceMoERebalanceCommandBufferHeaderView *>(local_command_headers),
            static_cast<DeviceMoERebalanceStatusView *>(status),
            payload_slot_capacity,
            cfg,
            command_buffer_count);
        return finishLaunch("cudaMoE_project_rebalance_domain_commands");
    }

    bool cudaMoE_pack_rebalance_compact_payloads(
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
        cudaSetDevice(device_idx);
        const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
        init_rebalance_apply_status_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<DeviceMoERebalanceApplyStatusView *>(status));
        pack_rebalance_compact_payloads_kernel<<<
            static_cast<int>(local_payload_slot_count),
            kThreads,
            0,
            static_cast<cudaStream_t>(stream)>>>(
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
        return finishLaunch("cudaMoE_pack_rebalance_compact_payloads");
    }

    bool cudaMoE_pack_rebalance_collective_payloads(
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
        cudaSetDevice(device_idx);
        const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
        init_rebalance_apply_status_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<DeviceMoERebalanceApplyStatusView *>(status));
        pack_rebalance_collective_payloads_kernel<<<
            static_cast<int>(local_payload_slot_count),
            kThreads,
            0,
            static_cast<cudaStream_t>(stream)>>>(
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
        return finishLaunch("cudaMoE_pack_rebalance_collective_payloads");
    }

    bool cudaMoE_unpack_rebalance_collective_payloads(
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
        cudaSetDevice(device_idx);
        const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
        init_rebalance_apply_status_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<DeviceMoERebalanceApplyStatusView *>(status));
        unpack_rebalance_collective_payloads_kernel<<<
            static_cast<int>(plan_capacity),
            kThreads,
            0,
            static_cast<cudaStream_t>(stream)>>>(
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
        return finishLaunch("cudaMoE_unpack_rebalance_collective_payloads");
    }

    bool cudaMoE_apply_rebalance_arrivals(
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
        cudaSetDevice(device_idx);
        const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
        init_rebalance_apply_status_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<DeviceMoERebalanceApplyStatusView *>(status));
        apply_rebalance_arrivals_kernel<<<1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
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
        return finishLaunch("cudaMoE_apply_rebalance_arrivals");
    }

    bool cudaMoE_init_rebalance_graph_controller_state(
        void *controller_state,
        const void *config,
        int device_idx,
        void *stream)
    {
        if (!controller_state || !config || !stream)
            return false;
        cudaSetDevice(device_idx);
        const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
        init_rebalance_graph_controller_state_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<DeviceMoERebalanceGraphControllerStateView *>(controller_state),
            cfg);
        return finishLaunch("cudaMoE_init_rebalance_graph_controller_state");
    }

    bool cudaMoE_publish_rebalance_transfer_complete(
        void *controller_state,
        const void *command_header,
        const void *wave_state,
        const void *copy_status,
        const void *config,
        uint32_t command_buffer_count,
        int device_idx,
        void *stream)
    {
        if (!controller_state || !command_header || !wave_state ||
            !copy_status || !config || !stream)
        {
            return false;
        }
        cudaSetDevice(device_idx);
        const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
        publish_rebalance_transfer_complete_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<DeviceMoERebalanceGraphControllerStateView *>(controller_state),
            static_cast<const DeviceMoERebalanceCommandBufferHeaderView *>(command_header),
            static_cast<const DeviceMoERebalanceWaveStateView *>(wave_state),
            static_cast<const DeviceMoERebalanceApplyStatusView *>(copy_status),
            cfg,
            command_buffer_count);
        return finishLaunch("cudaMoE_publish_rebalance_transfer_complete");
    }

    bool cudaMoE_apply_ready_rebalance_wave(
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
        {
            return false;
        }
        cudaSetDevice(device_idx);
        const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
        apply_rebalance_arrivals_kernel<<<1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
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
        return finishLaunch("cudaMoE_apply_ready_rebalance_wave");
    }

    bool cudaMoE_int_to_float(const int *input, float *output, int count, int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        int_to_float_kernel<<<blocksFor(count), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(input, output, count);
        return finishLaunch("cudaMoE_int_to_float");
    }

    bool cudaMoE_float_to_int(const float *input, int *output, int count, int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        float_to_int_kernel<<<blocksFor(count), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(input, output, count);
        return finishLaunch("cudaMoE_float_to_int");
    }

    bool cudaMoE_float_to_masked_int(
        const float *input,
        int *output,
        const uint8_t *expert_mask,
        int count,
        int num_experts,
        int device_idx,
        void *stream)
    {
        cudaSetDevice(device_idx);
        float_to_masked_int_kernel<<<blocksFor(count), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            input, output, expert_mask, count, num_experts);
        return finishLaunch("cudaMoE_float_to_masked_int");
    }

    bool cudaMoE_gather_tokens(const float *hidden, float *batch_buffer, const int *token_indices,
                               int num_tokens, int d_model, int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        if (num_tokens <= 0 || d_model <= 0)
            return true;

        if ((d_model & 3) == 0)
        {
            // 2D grid: x covers d_model/4 float4 columns, y selects the token row.
            const int n4 = d_model >> 2;
            dim3 grid((n4 + kThreads - 1) / kThreads, num_tokens);
            gather_tokens_kernel<<<grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
                hidden, batch_buffer, token_indices, num_tokens, d_model);
        }
        else
        {
            const int total_elements = num_tokens * d_model;
            gather_tokens_scalar_kernel<<<blocksFor(total_elements), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
                hidden, batch_buffer, token_indices, total_elements, d_model);
        }
        return finishLaunch("cudaMoE_gather_tokens");
    }

    bool cudaMoE_copy_token_row(const float *source, float *row_buffer,
                                int row_index, int row_width, int device_idx, void *stream)
    {
        if (!source || !row_buffer || row_index < 0 || row_width <= 0 || !stream)
            return false;
        cudaSetDevice(device_idx);
        const int vector_columns = (row_width + 3) >> 2;
        copy_token_row_kernel<<<blocksFor(vector_columns), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            source, row_buffer, row_index, row_width);
        return finishLaunch("cudaMoE_copy_token_row");
    }

    bool cudaMoE_scatter_add(float *output, const float *expert_output, const int *token_indices,
                             const float *weights, int num_tokens, int d_model, int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        if (num_tokens <= 0 || d_model <= 0)
            return true;

        if ((d_model & 3) == 0)
        {
            // 2D grid: x covers d_model/4 float4 columns, y selects the token row.
            const int n4 = d_model >> 2;
            dim3 grid((n4 + kThreads - 1) / kThreads, num_tokens);
            scatter_add_kernel<<<grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
                output, expert_output, token_indices, weights, num_tokens, d_model);
        }
        else
        {
            const int total_elements = num_tokens * d_model;
            scatter_add_scalar_kernel<<<blocksFor(total_elements), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
                output, expert_output, token_indices, weights, total_elements, d_model);
        }
        return finishLaunch("cudaMoE_scatter_add");
    }

    bool cudaMoE_write_token_row(float *destination, const float *row_buffer,
                                 int row_index, int row_width, int device_idx, void *stream)
    {
        if (!destination || !row_buffer || row_index < 0 || row_width <= 0 || !stream)
            return false;
        cudaSetDevice(device_idx);
        const int vector_columns = (row_width + 3) >> 2;
        write_token_row_kernel<<<blocksFor(vector_columns), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            destination, row_buffer, row_index, row_width);
        return finishLaunch("cudaMoE_write_token_row");
    }

    bool cudaMoE_shared_expert_gate(const float *input, const float *gate_inp, float *shared_output,
                                    int seq_len, int d_model, int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        shared_expert_gate_kernel<<<seq_len, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            input, gate_inp, shared_output, seq_len, d_model);
        return finishLaunch("cudaMoE_shared_expert_gate");
    }

    bool cudaMoE_shared_expert_gate_effective_seq_len(
        const float *input, const float *gate_inp, float *shared_output,
        int seq_len, int d_model, const int *device_effective_seq_len,
        int device_idx, void *stream)
    {
        if (!input || !gate_inp || !shared_output || !device_effective_seq_len ||
            seq_len <= 0 || d_model <= 0 || !stream)
            return false;
        cudaSetDevice(device_idx);
        shared_expert_gate_kernel<<<seq_len, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            input, gate_inp, shared_output, seq_len, d_model, device_effective_seq_len);
        return finishLaunch("cudaMoE_shared_expert_gate_effective_seq_len");
    }

    bool cudaMoE_shared_expert_gate_add(
        const float *input, const float *gate_inp, float *shared_output,
        const float *routed_residual, float *combined_output,
        int seq_len, int d_model, int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        shared_expert_gate_add_kernel<<<seq_len, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            input, gate_inp, shared_output, routed_residual, combined_output, seq_len, d_model);
        return finishLaunch("cudaMoE_shared_expert_gate_add");
    }

    bool cudaMoE_shared_expert_gate_add_effective_seq_len(
        const float *input, const float *gate_inp, float *shared_output,
        const float *routed_residual, float *combined_output,
        int seq_len, int d_model, const int *device_effective_seq_len,
        int device_idx, void *stream)
    {
        if (!input || !gate_inp || !shared_output || !routed_residual ||
            !combined_output || !device_effective_seq_len ||
            seq_len <= 0 || d_model <= 0 || !stream)
            return false;
        cudaSetDevice(device_idx);
        shared_expert_gate_add_kernel<<<seq_len, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            input, gate_inp, shared_output, routed_residual, combined_output,
            seq_len, d_model, device_effective_seq_len);
        return finishLaunch("cudaMoE_shared_expert_gate_add_effective_seq_len");
    }

    bool cudaMoE_swiglu(float *gate, const float *up, int count, int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        // One thread per float4 lane; ceil(count/4) threads cover both the vectorized
        // bulk and the (<4) scalar tail.
        swiglu_kernel<<<blocksFor((count + 3) >> 2), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(gate, up, count);
        return finishLaunch("cudaMoE_swiglu");
    }

    bool cudaMoE_weighted_add(float *output, const float *input, float weight, int count, int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        weighted_add_kernel<<<blocksFor((count + 3) >> 2), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(output, input, weight, count);
        return finishLaunch("cudaMoE_weighted_add");
    }

    bool cudaMoE_count_per_expert(const int *routing_indices, int *expert_counts, int total_slots,
                                  int num_experts, int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        count_per_expert_kernel<<<blocksFor(total_slots), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            routing_indices, expert_counts, total_slots, num_experts);
        return finishLaunch("cudaMoE_count_per_expert");
    }

    bool cudaMoE_exclusive_scan(const int *expert_counts, int *expert_offsets,
                                int num_experts, int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        exclusive_scan_kernel<<<1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(expert_counts, expert_offsets, num_experts);
        return finishLaunch("cudaMoE_exclusive_scan");
    }

    bool cudaMoE_build_active_expert_list(
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
        cudaSetDevice(device_idx);
        build_active_expert_list_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
            expert_counts, active_expert_ids, num_experts, max_active_experts);
        return finishLaunch("cudaMoE_build_active_expert_list");
    }

    bool cudaMoE_scatter_tokens(const int *routing_indices, const float *routing_weights,
                                int *write_heads, const int *expert_offsets,
                                int *grouped_token_indices, float *grouped_weights,
                                int total_slots, int top_k, int num_experts,
                                int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        scatter_tokens_kernel<<<blocksFor(total_slots), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            routing_indices, routing_weights, write_heads, expert_offsets,
            grouped_token_indices, grouped_weights, total_slots, top_k, num_experts);
        return finishLaunch("cudaMoE_scatter_tokens");
    }

    bool cudaMoE_scatter_tokens_deterministic(
        const int *routing_indices,
        const float *routing_weights,
        const int *expert_offsets,
        const int *expert_counts,
        int *grouped_token_indices,
        int *original_to_grouped,
        int *original_expert_ids,
        float *grouped_weights,
        int total_slots,
        int top_k,
        int num_experts,
        int device_idx,
        void *stream)
    {
        if (!stream)
            return false;
        cudaSetDevice(device_idx);
        scatter_tokens_deterministic_kernel<<<blocksFor(total_slots), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            routing_indices, routing_weights, expert_offsets, expert_counts,
            grouped_token_indices, original_to_grouped, original_expert_ids, grouped_weights,
            total_slots, top_k, num_experts);
        return finishLaunch("cudaMoE_scatter_tokens_deterministic");
    }

    bool cudaMoE_group_tokens_small_float(
        const float *routing_indices,
        const float *routing_weights,
        int *expert_counts,
        int *expert_offsets,
        int *grouped_token_indices,
        int *original_to_grouped,
        int *original_expert_ids,
        float *grouped_weights,
        int *active_expert_ids,
        int total_slots,
        int num_experts,
        int top_k,
        int max_active_experts,
        int device_idx,
        void *stream)
    {
        if (!stream)
            return false;
        cudaSetDevice(device_idx);
        group_tokens_small_float_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
            routing_indices, routing_weights, expert_counts, expert_offsets,
            grouped_token_indices, original_to_grouped, original_expert_ids, grouped_weights,
            active_expert_ids, total_slots, num_experts, top_k, max_active_experts);
        return finishLaunch("cudaMoE_group_tokens_small_float");
    }

    bool cudaMoE_prepare_shared_expert_group(
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
        cudaSetDevice(device_idx);
        prepare_shared_expert_group_kernel<<<blocksFor(seq_len), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            expert_offsets, expert_counts, grouped_token_indices, original_to_grouped,
            grouped_weights, active_expert_ids, seq_len);
        return finishLaunch("cudaMoE_prepare_shared_expert_group");
    }

    bool cudaMoE_group_prefill_routes_runtime(
        const float *routing_indices,
        const float *routing_weights,
        void *runtime,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        int device_idx,
        void *stream)
    {
        if (!runtime || !routing_indices || !routing_weights || !stream ||
            current_slots < 0 || max_slots <= 0 || current_slots > max_slots ||
            num_experts <= 0 || num_experts > kDeviceMoEMaxExperts || top_k <= 0)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        auto *runtime_view = static_cast<DeviceMoELayerRuntimeView *>(runtime);

        const int clear_items = max_slots > num_experts ? max_slots : num_experts;
        prefill_group_clear_runtime_kernel<<<blocksFor(clear_items), kThreads, 0, cuda_stream>>>(
            runtime_view, max_slots, num_experts, 1);
        if (!finishLaunch("cudaMoE_prefill_group_clear_runtime"))
            return false;

        prefill_group_cast_count_runtime_kernel<<<blocksFor(max_slots), kThreads, 0, cuda_stream>>>(
            runtime_view, routing_indices, routing_weights,
            current_slots, max_slots, num_experts);
        if (!finishLaunch("cudaMoE_prefill_group_cast_count_runtime"))
            return false;

        prefill_group_count_assigned_runtime_kernel<<<blocksFor(max_slots), kThreads, 0, cuda_stream>>>(
            runtime_view, current_slots, max_slots, num_experts);
        if (!finishLaunch("cudaMoE_prefill_group_count_assigned_runtime"))
            return false;

        prefill_group_exclusive_scan_runtime_kernel<<<1, 1, 0, cuda_stream>>>(
            runtime_view, num_experts);
        if (!finishLaunch("cudaMoE_prefill_group_scan_runtime"))
            return false;

        prefill_group_scatter_deterministic_runtime_kernel<<<num_experts, 1, 0, cuda_stream>>>(
            runtime_view, current_slots, max_slots, num_experts, top_k);
        return finishLaunch("cudaMoE_prefill_group_scatter_runtime");
    }

    bool cudaMoE_regroup_prefill_routes_runtime_assignments(
        void *runtime,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        int device_idx,
        void *stream)
    {
        if (!runtime || !stream ||
            current_slots < 0 || max_slots <= 0 || current_slots > max_slots ||
            num_experts <= 0 || num_experts > kDeviceMoEMaxExperts || top_k <= 0)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        auto *runtime_view = static_cast<DeviceMoELayerRuntimeView *>(runtime);

        const int clear_items = max_slots > num_experts ? max_slots : num_experts;
        prefill_group_clear_runtime_kernel<<<blocksFor(clear_items), kThreads, 0, cuda_stream>>>(
            runtime_view, max_slots, num_experts, 0);
        if (!finishLaunch("cudaMoE_prefill_regroup_clear_runtime"))
            return false;

        prefill_group_count_assigned_runtime_kernel<<<blocksFor(max_slots), kThreads, 0, cuda_stream>>>(
            runtime_view, current_slots, max_slots, num_experts);
        if (!finishLaunch("cudaMoE_prefill_regroup_count_assigned_runtime"))
            return false;

        prefill_group_exclusive_scan_runtime_kernel<<<1, 1, 0, cuda_stream>>>(
            runtime_view, num_experts);
        if (!finishLaunch("cudaMoE_prefill_regroup_scan_runtime"))
            return false;

        prefill_group_scatter_deterministic_runtime_kernel<<<num_experts, 1, 0, cuda_stream>>>(
            runtime_view, current_slots, max_slots, num_experts, top_k);
        return finishLaunch("cudaMoE_prefill_regroup_scatter_runtime");
    }

    bool cudaMoE_assign_prefill_routes_least_loaded_resident(
        void *runtime,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        int device_idx,
        void *stream)
    {
        (void)top_k;
        if (!runtime || !stream ||
            current_slots < 0 || max_slots <= 0 || current_slots > max_slots ||
            num_experts <= 0 || num_experts > kDeviceMoEMaxExperts)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        auto *runtime_view = static_cast<DeviceMoELayerRuntimeView *>(runtime);

        const int split_items = num_experts * static_cast<int>(kDeviceMoEMaxParticipants);
        int clear_items = max_slots > num_experts ? max_slots : num_experts;
        clear_items = clear_items > split_items ? clear_items : split_items;
        prefill_llep_clear_assignment_runtime_kernel<<<blocksFor(clear_items), kThreads, 0, cuda_stream>>>(
            runtime_view, max_slots, num_experts);
        if (!finishLaunch("cudaMoE_prefill_llep_clear_runtime"))
            return false;

        prefill_llep_count_expert_routes_runtime_kernel<<<blocksFor(max_slots), kThreads, 0, cuda_stream>>>(
            runtime_view, current_slots, max_slots, num_experts);
        if (!finishLaunch("cudaMoE_prefill_llep_count_routes"))
            return false;

        prefill_llep_plan_resident_splits_runtime_kernel<<<1, 1, 0, cuda_stream>>>(
            runtime_view, current_slots, max_slots, num_experts);
        if (!finishLaunch("cudaMoE_prefill_llep_plan_resident_splits"))
            return false;

        prefill_llep_assign_routes_from_splits_runtime_kernel<<<blocksFor(max_slots), kThreads, 0, cuda_stream>>>(
            runtime_view, current_slots, max_slots, num_experts);
        return finishLaunch("cudaMoE_prefill_llep_assign_routes_from_splits");
    }

    __attribute__((visibility("default"))) bool cudaMoE_materialize_runtime_prefill_descriptor_tables(
        const void *runtime,
        DeviceNativeVNNIMatrixDesc *gate_descs,
        DeviceNativeVNNIMatrixDesc *up_descs,
        DeviceNativeVNNIMatrixDesc *down_descs,
        int num_experts,
        int device_idx,
        void *stream)
    {
        if (!runtime || !gate_descs || !up_descs || !down_descs || !stream ||
            num_experts <= 0 || num_experts > kDeviceMoEMaxExperts)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        materialize_runtime_prefill_descriptor_tables_kernel<<<blocksFor(num_experts), kThreads, 0,
                                                               static_cast<cudaStream_t>(stream)>>>(
            static_cast<const DeviceMoELayerRuntimeView *>(runtime),
            gate_descs,
            up_descs,
            down_descs,
            num_experts);
        return finishLaunch("cudaMoE_materialize_runtime_prefill_descriptor_tables");
    }

    bool cudaMoE_prefill_gather_expert_runtime(
        const void *runtime,
        const float *hidden,
        float *batch_buffer,
        int expert_id,
        int max_tokens,
        int d_model,
        int device_idx,
        void *stream)
    {
        if (!runtime || !hidden || !batch_buffer || !stream ||
            expert_id < 0 || max_tokens <= 0 || d_model <= 0)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        const int total = max_tokens * d_model;
        prefill_gather_expert_runtime_kernel<<<blocksFor(total), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const DeviceMoELayerRuntimeView *>(runtime),
            hidden,
            batch_buffer,
            expert_id,
            max_tokens,
            d_model);
        return finishLaunch("cudaMoE_prefill_gather_expert_runtime");
    }

    bool cudaMoE_prefill_scatter_expert_runtime(
        float *output,
        const float *expert_output,
        const void *runtime,
        int expert_id,
        int max_tokens,
        int d_model,
        int device_idx,
        void *stream)
    {
        if (!output || !expert_output || !runtime || !stream ||
            expert_id < 0 || max_tokens <= 0 || d_model <= 0)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        const int total = max_tokens * d_model;
        prefill_scatter_expert_runtime_kernel<<<blocksFor(total), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            output,
            expert_output,
            static_cast<const DeviceMoELayerRuntimeView *>(runtime),
            expert_id,
            max_tokens,
            d_model);
        return finishLaunch("cudaMoE_prefill_scatter_expert_runtime");
    }

    bool cudaMoE_gather_expert_fixed(const float *hidden, float *batch_buffer,
                                     const int *expert_offsets, const int *expert_counts,
                                     const int *grouped_token_indices,
                                     int expert_id, int max_tokens, int d_model,
                                     int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        const int total = max_tokens * d_model;
        gather_expert_fixed_kernel<<<blocksFor(total), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            hidden, batch_buffer, expert_offsets, expert_counts, grouped_token_indices,
            expert_id, max_tokens, d_model);
        return finishLaunch("cudaMoE_gather_expert_fixed");
    }

    bool cudaMoE_scatter_expert_fixed(float *output, const float *expert_output,
                                      const int *expert_offsets, const int *expert_counts,
                                      const int *grouped_token_indices,
                                      const float *grouped_weights,
                                      int expert_id, int max_tokens, int d_model,
                                      int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        const int total = max_tokens * d_model;
        scatter_expert_fixed_kernel<<<blocksFor(total), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            output, expert_output, expert_offsets, expert_counts, grouped_token_indices,
            grouped_weights, expert_id, max_tokens, d_model);
        return finishLaunch("cudaMoE_scatter_expert_fixed");
    }

    bool cudaMoE_grouped_gate_up_native_vnni_decode_table(
        const float *d_hidden,
        const DeviceNativeVNNIMatrixDesc *d_gate_desc_table,
        const DeviceNativeVNNIMatrixDesc *d_up_desc_table,
        const int *d_expert_ids,
        float *const *d_gate_outputs,
        float *const *d_up_outputs,
        int8_t *d_hidden_int8,
        float *d_hidden_scales,
        bool hidden_prequantized,
        int num_active,
        int N,
        int K,
        uint8_t codebook_id,
        int device_idx,
        void *stream)
    {
        if (!d_hidden || !d_gate_desc_table || !d_up_desc_table || !d_expert_ids ||
            !d_gate_outputs || !d_up_outputs || !d_hidden_int8 || !d_hidden_scales ||
            num_active <= 0 || N <= 0 || K <= 0 || (K % 32) != 0)
        {
            std::fprintf(stderr, "[cudaMoE_grouped_gate_up_native_vnni_decode_table] invalid arguments\n");
            return false;
        }

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        if (!hidden_prequantized)
        {
            const int blocks_per_row = K / 32;
            grouped_hidden_quantize_blockwise_kernel<<<blocks_per_row, 32, 0, cuda_stream>>>(
                d_hidden, d_hidden_int8, d_hidden_scales, K);
            if (!finishLaunch("cudaMoE_grouped_gate_up_hidden_quantize"))
                return false;
        }

        constexpr int kTileN = 64;
        dim3 grid((N + kTileN - 1) / kTileN, num_active);
        dim3 block(kTileN);

#define LAUNCH_GROUPED_GATE_UP(CB)                                                                  \
    grouped_native_vnni_gate_up_decode_kernel<CB><<<grid, block, 0, cuda_stream>>>(                 \
        d_hidden_int8, d_hidden_scales, d_gate_desc_table, d_up_desc_table, d_expert_ids,           \
        d_gate_outputs, d_up_outputs, num_active, N, K)

        switch (codebook_id)
        {
        case 0:  LAUNCH_GROUPED_GATE_UP(0); break;
        case 4:  LAUNCH_GROUPED_GATE_UP(4); break;
        case 5:  LAUNCH_GROUPED_GATE_UP(5); break;
        case 6:  LAUNCH_GROUPED_GATE_UP(6); break;
        case 7:  LAUNCH_GROUPED_GATE_UP(7); break;
        case 8:  LAUNCH_GROUPED_GATE_UP(8); break;
        case 9:  LAUNCH_GROUPED_GATE_UP(9); break;
        case 10: LAUNCH_GROUPED_GATE_UP(10); break;
        case 11: LAUNCH_GROUPED_GATE_UP(11); break;
        case 12: LAUNCH_GROUPED_GATE_UP(12); break;
        case 13: LAUNCH_GROUPED_GATE_UP(13); break;
        case 14: LAUNCH_GROUPED_GATE_UP(14); break;
        case 15: LAUNCH_GROUPED_GATE_UP(15); break;
        case 16: LAUNCH_GROUPED_GATE_UP(16); break;
        case 17: LAUNCH_GROUPED_GATE_UP(17); break;
        case 19: LAUNCH_GROUPED_GATE_UP(19); break;
        case kMixedCodebookSentinel: LAUNCH_GROUPED_GATE_UP(kMixedCodebookSentinel); break;
        default:
            std::fprintf(stderr, "[cudaMoE_grouped_gate_up_native_vnni_decode_table] unsupported codebook_id=%u\n",
                         static_cast<unsigned>(codebook_id));
            return false;
        }

#undef LAUNCH_GROUPED_GATE_UP

        return finishLaunch("cudaMoE_grouped_gate_up_native_vnni_decode_table");
    }

    bool cudaMoE_grouped_gate_up_native_vnni_decode_table_kpart(
        const float *d_hidden,
        const DeviceNativeVNNIMatrixDesc *d_gate_desc_table,
        const DeviceNativeVNNIMatrixDesc *d_up_desc_table,
        const int *d_expert_ids,
        float *const *d_gate_outputs,
        float *const *d_up_outputs,
        int8_t *d_hidden_int8,
        float *d_hidden_scales,
        bool hidden_prequantized,
        float *d_gate_partials,
        float *d_up_partials,
        int num_active,
        int N,
        int K,
        int num_experts,
        uint8_t codebook_id,
        int k_partitions,
        int device_idx,
        void *stream)
    {
        const bool valid_k_partitions = (k_partitions == 2 || k_partitions == 4 || k_partitions == 8 ||
                                         k_partitions == 16 || k_partitions == 32);
        if (!d_hidden || !d_gate_desc_table || !d_up_desc_table || !d_expert_ids ||
            !d_gate_outputs || !d_up_outputs || !d_hidden_int8 || !d_hidden_scales ||
            !d_gate_partials || !d_up_partials ||
            num_active <= 0 || N <= 0 || K <= 0 || num_experts <= 0 ||
            (K % 32) != 0 || !valid_k_partitions)
        {
            std::fprintf(stderr, "[cudaMoE_grouped_gate_up_native_vnni_decode_table_kpart] invalid arguments\n");
            return false;
        }

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

        // Step 1: blockwise-quantize the shared hidden activation row once.
        if (!hidden_prequantized)
        {
            const int blocks_per_row = K / 32;
            grouped_hidden_quantize_blockwise_kernel<<<blocks_per_row, 32, 0, cuda_stream>>>(
                d_hidden, d_hidden_int8, d_hidden_scales, K);
            if (!finishLaunch("cudaMoE_grouped_gate_up_kpart_hidden_quantize"))
                return false;
        }

        // Step 2: split-K scatter — each (n-tile, k-partition, expert-slot) block
        // reduces its K-slice into the partials buffer.
        constexpr int kTileN = 64;
        dim3 partial_grid((N + kTileN - 1) / kTileN, k_partitions, num_active);
        dim3 block(kTileN);

#define LAUNCH_GROUPED_GATE_UP_KPART(CB)                                                            \
    grouped_native_vnni_gate_up_kpart_decode_kernel<CB><<<partial_grid, block, 0, cuda_stream>>>(   \
        d_hidden_int8, d_hidden_scales, d_gate_desc_table, d_up_desc_table, d_expert_ids,           \
        d_gate_partials, d_up_partials, num_active, N, K, num_experts, k_partitions)

        switch (codebook_id)
        {
        case 0:  LAUNCH_GROUPED_GATE_UP_KPART(0); break;
        case 4:  LAUNCH_GROUPED_GATE_UP_KPART(4); break;
        case 5:  LAUNCH_GROUPED_GATE_UP_KPART(5); break;
        case 6:  LAUNCH_GROUPED_GATE_UP_KPART(6); break;
        case 7:  LAUNCH_GROUPED_GATE_UP_KPART(7); break;
        case 8:  LAUNCH_GROUPED_GATE_UP_KPART(8); break;
        case 9:  LAUNCH_GROUPED_GATE_UP_KPART(9); break;
        case 10: LAUNCH_GROUPED_GATE_UP_KPART(10); break;
        case 11: LAUNCH_GROUPED_GATE_UP_KPART(11); break;
        case 12: LAUNCH_GROUPED_GATE_UP_KPART(12); break;
        case 13: LAUNCH_GROUPED_GATE_UP_KPART(13); break;
        case 14: LAUNCH_GROUPED_GATE_UP_KPART(14); break;
        case 15: LAUNCH_GROUPED_GATE_UP_KPART(15); break;
        case 16: LAUNCH_GROUPED_GATE_UP_KPART(16); break;
        case 17: LAUNCH_GROUPED_GATE_UP_KPART(17); break;
        case 19: LAUNCH_GROUPED_GATE_UP_KPART(19); break;
        case kMixedCodebookSentinel: LAUNCH_GROUPED_GATE_UP_KPART(kMixedCodebookSentinel); break;
        default:
            std::fprintf(stderr, "[cudaMoE_grouped_gate_up_native_vnni_decode_table_kpart] unsupported codebook_id=%u\n",
                         static_cast<unsigned>(codebook_id));
            return false;
        }

#undef LAUNCH_GROUPED_GATE_UP_KPART

        if (!finishLaunch("cudaMoE_grouped_gate_up_kpart_partial"))
            return false;

        // Step 3: reduce the k_partitions partials into the final gate/up outputs.
        dim3 reduce_grid((N + kTileN - 1) / kTileN, num_active);
        grouped_native_vnni_gate_up_kpart_reduce_kernel<<<reduce_grid, block, 0, cuda_stream>>>(
            d_gate_partials, d_up_partials, d_expert_ids, d_gate_outputs, d_up_outputs,
            num_active, N, k_partitions);

        return finishLaunch("cudaMoE_grouped_gate_up_native_vnni_decode_table_kpart");
    }

    bool cudaMoE_grouped_gate_up_native_vnni_decode_runtime(
        const float *d_hidden,
        const void *d_runtime_layer,
        const int *d_expert_ids,
        float *const *d_gate_outputs,
        float *const *d_up_outputs,
        int8_t *d_hidden_int8,
        float *d_hidden_scales,
        bool hidden_prequantized,
        int num_active,
        int N,
        int K,
        int num_experts,
        uint8_t codebook_id,
        int device_idx,
        void *stream)
    {
        if (!d_hidden || !d_runtime_layer || !d_expert_ids ||
            !d_gate_outputs || !d_up_outputs || !d_hidden_int8 || !d_hidden_scales ||
            num_active <= 0 || N <= 0 || K <= 0 || num_experts <= 0 || (K % 32) != 0)
        {
            std::fprintf(stderr, "[cudaMoE_grouped_gate_up_native_vnni_decode_runtime] invalid arguments\n");
            return false;
        }

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        if (!hidden_prequantized)
        {
            const int blocks_per_row = K / 32;
            grouped_hidden_quantize_blockwise_kernel<<<blocks_per_row, 32, 0, cuda_stream>>>(
                d_hidden, d_hidden_int8, d_hidden_scales, K);
            if (!finishLaunch("cudaMoE_grouped_gate_up_runtime_hidden_quantize"))
                return false;
        }

        constexpr int kTileN = 64;
        dim3 grid((N + kTileN - 1) / kTileN, num_active);
        dim3 block(kTileN);
        const auto *runtime = static_cast<const DeviceMoELayerRuntimeView *>(d_runtime_layer);

#define LAUNCH_GROUPED_GATE_UP_RUNTIME(CB)                                                       \
    grouped_native_vnni_gate_up_decode_runtime_kernel<CB><<<grid, block, 0, cuda_stream>>>(       \
        d_hidden_int8, d_hidden_scales, runtime, d_expert_ids,                                    \
        d_gate_outputs, d_up_outputs, num_active, N, K, num_experts)

        switch (codebook_id)
        {
        case 0:  LAUNCH_GROUPED_GATE_UP_RUNTIME(0); break;
        case 4:  LAUNCH_GROUPED_GATE_UP_RUNTIME(4); break;
        case 5:  LAUNCH_GROUPED_GATE_UP_RUNTIME(5); break;
        case 6:  LAUNCH_GROUPED_GATE_UP_RUNTIME(6); break;
        case 7:  LAUNCH_GROUPED_GATE_UP_RUNTIME(7); break;
        case 8:  LAUNCH_GROUPED_GATE_UP_RUNTIME(8); break;
        case 9:  LAUNCH_GROUPED_GATE_UP_RUNTIME(9); break;
        case 10: LAUNCH_GROUPED_GATE_UP_RUNTIME(10); break;
        case 11: LAUNCH_GROUPED_GATE_UP_RUNTIME(11); break;
        case 12: LAUNCH_GROUPED_GATE_UP_RUNTIME(12); break;
        case 13: LAUNCH_GROUPED_GATE_UP_RUNTIME(13); break;
        case 14: LAUNCH_GROUPED_GATE_UP_RUNTIME(14); break;
        case 15: LAUNCH_GROUPED_GATE_UP_RUNTIME(15); break;
        case 16: LAUNCH_GROUPED_GATE_UP_RUNTIME(16); break;
        case 17: LAUNCH_GROUPED_GATE_UP_RUNTIME(17); break;
        case 19: LAUNCH_GROUPED_GATE_UP_RUNTIME(19); break;
        case kMixedCodebookSentinel: LAUNCH_GROUPED_GATE_UP_RUNTIME(kMixedCodebookSentinel); break;
        default:
            std::fprintf(stderr, "[cudaMoE_grouped_gate_up_native_vnni_decode_runtime] unsupported codebook_id=%u\n",
                         static_cast<unsigned>(codebook_id));
            return false;
        }

#undef LAUNCH_GROUPED_GATE_UP_RUNTIME

        return finishLaunch("cudaMoE_grouped_gate_up_native_vnni_decode_runtime");
    }

    bool cudaMoE_grouped_gate_up_native_vnni_decode_runtime_kpart(
        const float *d_hidden,
        const void *d_runtime_layer,
        const int *d_expert_ids,
        float *const *d_gate_outputs,
        float *const *d_up_outputs,
        int8_t *d_hidden_int8,
        float *d_hidden_scales,
        bool hidden_prequantized,
        float *d_gate_partials,
        float *d_up_partials,
        int num_active,
        int N,
        int K,
        int num_experts,
        uint8_t codebook_id,
        int k_partitions,
        int device_idx,
        void *stream)
    {
        const bool valid_k_partitions = (k_partitions == 2 || k_partitions == 4 || k_partitions == 8 ||
                                         k_partitions == 16 || k_partitions == 32);
        if (!d_hidden || !d_runtime_layer || !d_expert_ids ||
            !d_gate_outputs || !d_up_outputs || !d_hidden_int8 || !d_hidden_scales ||
            !d_gate_partials || !d_up_partials ||
            num_active <= 0 || N <= 0 || K <= 0 || num_experts <= 0 ||
            (K % 32) != 0 || !valid_k_partitions)
        {
            std::fprintf(stderr, "[cudaMoE_grouped_gate_up_native_vnni_decode_runtime_kpart] invalid arguments\n");
            return false;
        }

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        if (!hidden_prequantized)
        {
            const int blocks_per_row = K / 32;
            grouped_hidden_quantize_blockwise_kernel<<<blocks_per_row, 32, 0, cuda_stream>>>(
                d_hidden, d_hidden_int8, d_hidden_scales, K);
            if (!finishLaunch("cudaMoE_grouped_gate_up_runtime_kpart_hidden_quantize"))
                return false;
        }

        constexpr int kTileN = 64;
        dim3 partial_grid((N + kTileN - 1) / kTileN, k_partitions, num_active);
        dim3 block(kTileN);
        const auto *runtime = static_cast<const DeviceMoELayerRuntimeView *>(d_runtime_layer);

#define LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(CB)                                                   \
    grouped_native_vnni_gate_up_kpart_decode_runtime_kernel<CB><<<partial_grid, block, 0, cuda_stream>>>( \
        d_hidden_int8, d_hidden_scales, runtime, d_expert_ids,                                      \
        d_gate_partials, d_up_partials, num_active, N, K, num_experts, k_partitions)

        switch (codebook_id)
        {
        case 0:  LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(0); break;
        case 4:  LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(4); break;
        case 5:  LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(5); break;
        case 6:  LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(6); break;
        case 7:  LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(7); break;
        case 8:  LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(8); break;
        case 9:  LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(9); break;
        case 10: LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(10); break;
        case 11: LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(11); break;
        case 12: LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(12); break;
        case 13: LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(13); break;
        case 14: LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(14); break;
        case 15: LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(15); break;
        case 16: LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(16); break;
        case 17: LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(17); break;
        case 19: LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(19); break;
        case kMixedCodebookSentinel: LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(kMixedCodebookSentinel); break;
        default:
            std::fprintf(stderr, "[cudaMoE_grouped_gate_up_native_vnni_decode_runtime_kpart] unsupported codebook_id=%u\n",
                         static_cast<unsigned>(codebook_id));
            return false;
        }

#undef LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART

        if (!finishLaunch("cudaMoE_grouped_gate_up_native_vnni_decode_runtime_kpart partial"))
            return false;

        dim3 reduce_grid((N + kTileN - 1) / kTileN, num_active);
        grouped_native_vnni_gate_up_kpart_reduce_kernel<<<reduce_grid, block, 0, cuda_stream>>>(
            d_gate_partials, d_up_partials, d_expert_ids, d_gate_outputs, d_up_outputs,
            num_active, N, k_partitions);

        return finishLaunch("cudaMoE_grouped_gate_up_native_vnni_decode_runtime_kpart reduce");
    }

    bool cudaMoE_grouped_swiglu_down_native_vnni_decode_table(
        const float *const *d_gate_ptrs,
        const float *const *d_up_ptrs,
        const DeviceNativeVNNIMatrixDesc *d_desc_table,
        const int *d_expert_ids,
        const float *d_weights,
        int8_t *d_swiglu_int8,
        float *d_swiglu_scales,
        float *d_output,
        int num_active,
        int N,
        int K,
        uint8_t codebook_id,
        int device_idx,
        void *stream)
    {
        if (!d_gate_ptrs || !d_up_ptrs || !d_desc_table || !d_expert_ids || !d_weights ||
            !d_swiglu_int8 || !d_swiglu_scales || !d_output ||
            num_active <= 0 || N <= 0 || K <= 0 || (K % 32) != 0)
        {
            std::fprintf(stderr, "[cudaMoE_grouped_swiglu_down_native_vnni_decode_table] invalid arguments\n");
            return false;
        }

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        grouped_swiglu_quantize_blockwise_kernel<<<num_active, kThreads, 0, cuda_stream>>>(
            d_gate_ptrs, d_up_ptrs, d_expert_ids, d_swiglu_int8, d_swiglu_scales, num_active, K);
        if (!finishLaunch("cudaMoE_grouped_swiglu_quantize"))
            return false;

        constexpr int kTileN = 64;
        dim3 grid((N + kTileN - 1) / kTileN);
        dim3 block(kTileN);

#define LAUNCH_GROUPED_DOWN(CB)                                                                  \
    grouped_native_vnni_down_decode_kernel<CB><<<grid, block, 0, cuda_stream>>>(                 \
        d_swiglu_int8, d_swiglu_scales, d_desc_table, d_expert_ids, d_weights,                   \
        d_output, num_active, N, K)

        switch (codebook_id)
        {
        case 0:  LAUNCH_GROUPED_DOWN(0); break;
        case 4:  LAUNCH_GROUPED_DOWN(4); break;
        case 5:  LAUNCH_GROUPED_DOWN(5); break;
        case 6:  LAUNCH_GROUPED_DOWN(6); break;
        case 7:  LAUNCH_GROUPED_DOWN(7); break;
        case 8:  LAUNCH_GROUPED_DOWN(8); break;
        case 9:  LAUNCH_GROUPED_DOWN(9); break;
        case 10: LAUNCH_GROUPED_DOWN(10); break;
        case 11: LAUNCH_GROUPED_DOWN(11); break;
        case 12: LAUNCH_GROUPED_DOWN(12); break;
        case 13: LAUNCH_GROUPED_DOWN(13); break;
        case 14: LAUNCH_GROUPED_DOWN(14); break;
        case 15: LAUNCH_GROUPED_DOWN(15); break;
        case 16: LAUNCH_GROUPED_DOWN(16); break;
        case 17: LAUNCH_GROUPED_DOWN(17); break;
        case 19: LAUNCH_GROUPED_DOWN(19); break;
        case kMixedCodebookSentinel: LAUNCH_GROUPED_DOWN(kMixedCodebookSentinel); break;
        default:
            std::fprintf(stderr, "[cudaMoE_grouped_swiglu_down_native_vnni_decode_table] unsupported codebook_id=%u\n",
                         static_cast<unsigned>(codebook_id));
            return false;
        }

#undef LAUNCH_GROUPED_DOWN

        return finishLaunch("cudaMoE_grouped_swiglu_down_native_vnni_decode_table");
    }

    bool cudaMoE_grouped_swiglu_down_native_vnni_decode_table_kpart(
        const float *const *d_gate_ptrs,
        const float *const *d_up_ptrs,
        const DeviceNativeVNNIMatrixDesc *d_desc_table,
        const int *d_expert_ids,
        const float *d_weights,
        int8_t *d_swiglu_int8,
        float *d_swiglu_scales,
        float *d_down_partials,
        float *d_output,
        int num_active,
        int d_model,
        int intermediate,
        int num_experts,
        uint8_t codebook_id,
        int k_partitions,
        int device_idx,
        void *stream)
    {
        const bool valid_k_partitions =
            (k_partitions == 2 || k_partitions == 4 || k_partitions == 8 ||
             k_partitions == 16);
        if (!d_gate_ptrs || !d_up_ptrs || !d_desc_table || !d_expert_ids || !d_weights ||
            !d_swiglu_int8 || !d_swiglu_scales || !d_down_partials || !d_output ||
            num_active <= 0 || d_model <= 0 || intermediate <= 0 || num_experts <= 0 ||
            (intermediate % 32) != 0 || !valid_k_partitions)
        {
            std::fprintf(stderr, "[cudaMoE_grouped_swiglu_down_native_vnni_decode_table_kpart] invalid arguments\n");
            return false;
        }

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

        // Step 1: quantize the per-slot SwiGLU activations (shared with serial path).
        grouped_swiglu_quantize_blockwise_kernel<<<num_active, kThreads, 0, cuda_stream>>>(
            d_gate_ptrs, d_up_ptrs, d_expert_ids, d_swiglu_int8, d_swiglu_scales, num_active, intermediate);
        if (!finishLaunch("cudaMoE_grouped_swiglu_quantize"))
            return false;

        constexpr int kTileN = 64;
        const int N = d_model;
        const int K = intermediate;
        dim3 scatter_grid((N + kTileN - 1) / kTileN, k_partitions);
        dim3 reduce_grid((N + kTileN - 1) / kTileN);
        dim3 block(kTileN);

        // Step 2: scatter — each (n, k_part) block writes a route-weighted partial.
#define LAUNCH_GROUPED_DOWN_KPART(CB)                                                            \
    grouped_native_vnni_down_kpart_decode_kernel<CB><<<scatter_grid, block, 0, cuda_stream>>>(   \
        d_swiglu_int8, d_swiglu_scales, d_desc_table, d_expert_ids, d_weights,                   \
        d_down_partials, num_active, N, K, num_experts, k_partitions)

        switch (codebook_id)
        {
        case 0:  LAUNCH_GROUPED_DOWN_KPART(0); break;
        case 4:  LAUNCH_GROUPED_DOWN_KPART(4); break;
        case 5:  LAUNCH_GROUPED_DOWN_KPART(5); break;
        case 6:  LAUNCH_GROUPED_DOWN_KPART(6); break;
        case 7:  LAUNCH_GROUPED_DOWN_KPART(7); break;
        case 8:  LAUNCH_GROUPED_DOWN_KPART(8); break;
        case 9:  LAUNCH_GROUPED_DOWN_KPART(9); break;
        case 10: LAUNCH_GROUPED_DOWN_KPART(10); break;
        case 11: LAUNCH_GROUPED_DOWN_KPART(11); break;
        case 12: LAUNCH_GROUPED_DOWN_KPART(12); break;
        case 13: LAUNCH_GROUPED_DOWN_KPART(13); break;
        case 14: LAUNCH_GROUPED_DOWN_KPART(14); break;
        case 15: LAUNCH_GROUPED_DOWN_KPART(15); break;
        case 16: LAUNCH_GROUPED_DOWN_KPART(16); break;
        case 17: LAUNCH_GROUPED_DOWN_KPART(17); break;
        case 19: LAUNCH_GROUPED_DOWN_KPART(19); break;
        case kMixedCodebookSentinel: LAUNCH_GROUPED_DOWN_KPART(kMixedCodebookSentinel); break;
        default:
            std::fprintf(stderr, "[cudaMoE_grouped_swiglu_down_native_vnni_decode_table_kpart] unsupported codebook_id=%u\n",
                         static_cast<unsigned>(codebook_id));
            return false;
        }

#undef LAUNCH_GROUPED_DOWN_KPART

        if (!finishLaunch("cudaMoE_grouped_swiglu_down_native_vnni_decode_table_kpart scatter"))
            return false;

        // Step 3: reduce — sum the k_partitions partials into the final output.
        grouped_native_vnni_down_kpart_reduce_kernel<<<reduce_grid, block, 0, cuda_stream>>>(
            d_down_partials, d_output, N, k_partitions);

        return finishLaunch("cudaMoE_grouped_swiglu_down_native_vnni_decode_table_kpart reduce");
    }

    bool cudaMoE_grouped_swiglu_down_native_vnni_decode_runtime(
        const float *const *d_gate_ptrs,
        const float *const *d_up_ptrs,
        const void *d_runtime_layer,
        const int *d_expert_ids,
        const float *d_weights,
        int8_t *d_swiglu_int8,
        float *d_swiglu_scales,
        float *d_output,
        int num_active,
        int N,
        int K,
        int num_experts,
        uint8_t codebook_id,
        int device_idx,
        void *stream)
    {
        if (!d_gate_ptrs || !d_up_ptrs || !d_runtime_layer || !d_expert_ids || !d_weights ||
            !d_swiglu_int8 || !d_swiglu_scales || !d_output ||
            num_active <= 0 || N <= 0 || K <= 0 || num_experts <= 0 || (K % 32) != 0)
        {
            std::fprintf(stderr, "[cudaMoE_grouped_swiglu_down_native_vnni_decode_runtime] invalid arguments\n");
            return false;
        }

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        grouped_swiglu_quantize_blockwise_kernel<<<num_active, kThreads, 0, cuda_stream>>>(
            d_gate_ptrs, d_up_ptrs, d_expert_ids, d_swiglu_int8, d_swiglu_scales, num_active, K);
        if (!finishLaunch("cudaMoE_grouped_swiglu_runtime_quantize"))
            return false;

        constexpr int kTileN = 64;
        dim3 grid((N + kTileN - 1) / kTileN);
        dim3 block(kTileN);
        const auto *runtime = static_cast<const DeviceMoELayerRuntimeView *>(d_runtime_layer);

#define LAUNCH_GROUPED_DOWN_RUNTIME(CB)                                                        \
    grouped_native_vnni_down_decode_runtime_kernel<CB><<<grid, block, 0, cuda_stream>>>(       \
        d_swiglu_int8, d_swiglu_scales, runtime, d_expert_ids, d_weights,                      \
        d_output, num_active, N, K, num_experts)

        switch (codebook_id)
        {
        case 0:  LAUNCH_GROUPED_DOWN_RUNTIME(0); break;
        case 4:  LAUNCH_GROUPED_DOWN_RUNTIME(4); break;
        case 5:  LAUNCH_GROUPED_DOWN_RUNTIME(5); break;
        case 6:  LAUNCH_GROUPED_DOWN_RUNTIME(6); break;
        case 7:  LAUNCH_GROUPED_DOWN_RUNTIME(7); break;
        case 8:  LAUNCH_GROUPED_DOWN_RUNTIME(8); break;
        case 9:  LAUNCH_GROUPED_DOWN_RUNTIME(9); break;
        case 10: LAUNCH_GROUPED_DOWN_RUNTIME(10); break;
        case 11: LAUNCH_GROUPED_DOWN_RUNTIME(11); break;
        case 12: LAUNCH_GROUPED_DOWN_RUNTIME(12); break;
        case 13: LAUNCH_GROUPED_DOWN_RUNTIME(13); break;
        case 14: LAUNCH_GROUPED_DOWN_RUNTIME(14); break;
        case 15: LAUNCH_GROUPED_DOWN_RUNTIME(15); break;
        case 16: LAUNCH_GROUPED_DOWN_RUNTIME(16); break;
        case 17: LAUNCH_GROUPED_DOWN_RUNTIME(17); break;
        case 19: LAUNCH_GROUPED_DOWN_RUNTIME(19); break;
        case kMixedCodebookSentinel: LAUNCH_GROUPED_DOWN_RUNTIME(kMixedCodebookSentinel); break;
        default:
            std::fprintf(stderr, "[cudaMoE_grouped_swiglu_down_native_vnni_decode_runtime] unsupported codebook_id=%u\n",
                         static_cast<unsigned>(codebook_id));
            return false;
        }

#undef LAUNCH_GROUPED_DOWN_RUNTIME

        return finishLaunch("cudaMoE_grouped_swiglu_down_native_vnni_decode_runtime");
    }

    bool cudaMoE_grouped_swiglu_down_native_vnni_decode_runtime_kpart(
        const float *const *d_gate_ptrs,
        const float *const *d_up_ptrs,
        const void *d_runtime_layer,
        const int *d_expert_ids,
        const float *d_weights,
        int8_t *d_swiglu_int8,
        float *d_swiglu_scales,
        float *d_down_partials,
        float *d_output,
        int num_active,
        int d_model,
        int intermediate,
        int num_experts,
        uint8_t codebook_id,
        int k_partitions,
        int device_idx,
        void *stream)
    {
        const bool valid_k_partitions =
            (k_partitions == 2 || k_partitions == 4 || k_partitions == 8 ||
             k_partitions == 16);
        if (!d_gate_ptrs || !d_up_ptrs || !d_runtime_layer || !d_expert_ids || !d_weights ||
            !d_swiglu_int8 || !d_swiglu_scales || !d_down_partials || !d_output ||
            num_active <= 0 || d_model <= 0 || intermediate <= 0 || num_experts <= 0 ||
            (intermediate % 32) != 0 || !valid_k_partitions)
        {
            std::fprintf(stderr, "[cudaMoE_grouped_swiglu_down_native_vnni_decode_runtime_kpart] invalid arguments\n");
            return false;
        }

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        grouped_swiglu_quantize_blockwise_kernel<<<num_active, kThreads, 0, cuda_stream>>>(
            d_gate_ptrs, d_up_ptrs, d_expert_ids, d_swiglu_int8, d_swiglu_scales, num_active, intermediate);
        if (!finishLaunch("cudaMoE_grouped_swiglu_runtime_quantize"))
            return false;

        constexpr int kTileN = 64;
        const int N = d_model;
        const int K = intermediate;
        dim3 scatter_grid((N + kTileN - 1) / kTileN, k_partitions);
        dim3 reduce_grid((N + kTileN - 1) / kTileN);
        dim3 block(kTileN);
        const auto *runtime = static_cast<const DeviceMoELayerRuntimeView *>(d_runtime_layer);

#define LAUNCH_GROUPED_DOWN_RUNTIME_KPART(CB)                                                   \
    grouped_native_vnni_down_kpart_decode_runtime_kernel<CB><<<scatter_grid, block, 0, cuda_stream>>>( \
        d_swiglu_int8, d_swiglu_scales, runtime, d_expert_ids, d_weights,                       \
        d_down_partials, num_active, N, K, num_experts, k_partitions)

        switch (codebook_id)
        {
        case 0:  LAUNCH_GROUPED_DOWN_RUNTIME_KPART(0); break;
        case 4:  LAUNCH_GROUPED_DOWN_RUNTIME_KPART(4); break;
        case 5:  LAUNCH_GROUPED_DOWN_RUNTIME_KPART(5); break;
        case 6:  LAUNCH_GROUPED_DOWN_RUNTIME_KPART(6); break;
        case 7:  LAUNCH_GROUPED_DOWN_RUNTIME_KPART(7); break;
        case 8:  LAUNCH_GROUPED_DOWN_RUNTIME_KPART(8); break;
        case 9:  LAUNCH_GROUPED_DOWN_RUNTIME_KPART(9); break;
        case 10: LAUNCH_GROUPED_DOWN_RUNTIME_KPART(10); break;
        case 11: LAUNCH_GROUPED_DOWN_RUNTIME_KPART(11); break;
        case 12: LAUNCH_GROUPED_DOWN_RUNTIME_KPART(12); break;
        case 13: LAUNCH_GROUPED_DOWN_RUNTIME_KPART(13); break;
        case 14: LAUNCH_GROUPED_DOWN_RUNTIME_KPART(14); break;
        case 15: LAUNCH_GROUPED_DOWN_RUNTIME_KPART(15); break;
        case 16: LAUNCH_GROUPED_DOWN_RUNTIME_KPART(16); break;
        case 17: LAUNCH_GROUPED_DOWN_RUNTIME_KPART(17); break;
        case 19: LAUNCH_GROUPED_DOWN_RUNTIME_KPART(19); break;
        case kMixedCodebookSentinel: LAUNCH_GROUPED_DOWN_RUNTIME_KPART(kMixedCodebookSentinel); break;
        default:
            std::fprintf(stderr, "[cudaMoE_grouped_swiglu_down_native_vnni_decode_runtime_kpart] unsupported codebook_id=%u\n",
                         static_cast<unsigned>(codebook_id));
            return false;
        }

#undef LAUNCH_GROUPED_DOWN_RUNTIME_KPART

        if (!finishLaunch("cudaMoE_grouped_swiglu_down_native_vnni_decode_runtime_kpart scatter"))
            return false;

        grouped_native_vnni_down_kpart_reduce_kernel<<<reduce_grid, block, 0, cuda_stream>>>(
            d_down_partials, d_output, N, k_partitions);

        return finishLaunch("cudaMoE_grouped_swiglu_down_native_vnni_decode_runtime_kpart reduce");
    }

    bool cudaMoE_grouped_prefill_pipeline(
        const float *d_hidden,
        const DeviceNativeVNNIMatrixDesc *d_gate_desc_table,
        const DeviceNativeVNNIMatrixDesc *d_up_desc_table,
        const DeviceNativeVNNIMatrixDesc *d_down_desc_table,
        const int *d_group_counts,
        const int *d_group_offsets,
        const int *d_group_token_indices,
        const int *d_original_to_grouped,
        const int *d_active_expert_ids,
        const float *d_group_weights,
        int8_t *d_scratch_A_int8,
        float *d_scratch_scales,
        float *d_scratch_gate,
        float *d_scratch_up,
        float *d_gate_partials,
        float *d_up_partials,
        int8_t *d_scratch_swiglu_int8,
        float *d_scratch_swiglu_scales,
        float *d_scratch_down_out,
        float *d_output,
        int num_experts,
        int d_model,
        int intermediate,
        int max_tokens_per_expert,
        int total_slots,
        int top_k,
        int active_expert_slots,
        uint8_t gateup_codebook_id,
        uint8_t down_codebook_id,
        uint32_t gateup_codebook_mask,
        uint32_t down_codebook_mask,
        int gateup_k_partitions,
        int device_idx,
        void *stream)
    {
        if (!d_hidden || !d_gate_desc_table || !d_up_desc_table || !d_down_desc_table ||
            !d_group_counts || !d_group_offsets || !d_group_token_indices || !d_group_weights ||
            !d_scratch_A_int8 || !d_scratch_scales || !d_scratch_gate || !d_scratch_up ||
            !d_scratch_swiglu_int8 || !d_scratch_swiglu_scales || !d_scratch_down_out || !d_output ||
            num_experts <= 0 || d_model <= 0 || intermediate <= 0 ||
            max_tokens_per_expert <= 0 || total_slots <= 0 || top_k <= 0 ||
            active_expert_slots < 0 ||
            (d_model % 32) != 0 || (intermediate % 32) != 0)
        {
            std::fprintf(stderr, "[cudaMoE_grouped_prefill_pipeline] invalid arguments\n");
            return false;
        }
        const bool use_active_expert_grid = active_expert_slots > 0;
        const bool use_gateup_kpart =
            use_active_expert_grid &&
            max_tokens_per_expert <= 4 &&
            gateup_k_partitions > 1 &&
            d_gate_partials &&
            d_up_partials;
        if (use_active_expert_grid &&
            (!d_active_expert_ids ||
             active_expert_slots > total_slots ||
             active_expert_slots > num_experts))
            return false;
        const int expert_grid = use_active_expert_grid ? active_expert_slots : num_experts;

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

        {
            // One warp per 32-col quant block; pack kWarpsPerQuantBlock warps per CUDA
            // block so each block covers kWarpsPerQuantBlock*32 columns. grid.x rounds up.
            const int blocks_per_row = d_model / 32;
            dim3 grid((blocks_per_row + kWarpsPerQuantBlock - 1) / kWarpsPerQuantBlock, total_slots);
            dim3 block(kWarpsPerQuantBlock * 32);
            grouped_prefill_gather_quantize_blockwise_kernel<<<grid, block, 0, cuda_stream>>>(
                d_hidden, d_scratch_A_int8, d_scratch_scales,
                d_group_token_indices, total_slots, d_model);
            if (!finishLaunch("cudaMoE_grouped_prefill_gather_quantize"))
                return false;
        }

        {
            const int requestedTileM =
                llaminar2::debugEnv().gemm.cuda_moe_prefill_tile_m;
            const int kTileM =
                select_grouped_prefill_tile_m(requestedTileM, max_tokens_per_expert);
            const bool tiny_active_verifier =
                active_expert_slots > 0 && max_tokens_per_expert <= 4;
            const int kTileN = tiny_active_verifier ? 64 : 128;
            // When fusion is enabled the gate/up kernel computes SwiGLU + blockwise int8 quant
            // in its epilogue, writing the down-projection input directly (no FP32 gate/up
            // round-trip, no separate swiglu_quantize launch).
            const bool fuse_swiglu = llaminar2::debugEnv().gemm.cuda_moe_prefill_fuse_swiglu;
            dim3 grid((intermediate + kTileN - 1) / kTileN,
                      (max_tokens_per_expert + kTileM - 1) / kTileM,
                      expert_grid);
            dim3 block(kTileN);

#define LAUNCH_GROUPED_GATEUP_PREFILL_TM(CB, TM)                                                   \
    do {                                                                                           \
        dim3 cb_grid(grid.x, grid.y, expert_grid);                                                  \
        if (kTileN == 64)                                                                            \
            grouped_native_vnni_gate_up_prefill_kernel<CB, TM, 64><<<cb_grid, block, 0, cuda_stream>>>( \
                d_scratch_A_int8, d_scratch_scales, d_gate_desc_table, d_up_desc_table,             \
                d_group_counts, d_group_offsets, d_active_expert_ids, active_expert_slots,          \
                d_scratch_gate, d_scratch_up, intermediate, d_model);                               \
        else                                                                                         \
            grouped_native_vnni_gate_up_prefill_kernel<CB, TM, 128><<<cb_grid, block, 0, cuda_stream>>>( \
                d_scratch_A_int8, d_scratch_scales, d_gate_desc_table, d_up_desc_table,             \
                d_group_counts, d_group_offsets, d_active_expert_ids, active_expert_slots,          \
                d_scratch_gate, d_scratch_up, intermediate, d_model);                               \
    } while (0)
#define LAUNCH_GROUPED_GATEUP_KPART_PREFILL_TM(CB, TM)                                             \
    do {                                                                                           \
        dim3 cb_grid(grid.x, grid.y, expert_grid * gateup_k_partitions);                           \
        grouped_native_vnni_gate_up_prefill_kpart_kernel<CB, TM><<<cb_grid, block, 0, cuda_stream>>>( \
            d_scratch_A_int8, d_scratch_scales, d_gate_desc_table, d_up_desc_table,                 \
            d_group_counts, d_group_offsets, d_active_expert_ids, active_expert_slots,              \
            d_gate_partials, d_up_partials, intermediate, d_model, gateup_k_partitions);            \
    } while (0)
#define LAUNCH_GROUPED_GATEUP_SWIGLU_PREFILL_TM(CB, TM)                                            \
    do {                                                                                           \
        dim3 cb_grid(grid.x, grid.y, expert_grid);                                                  \
        if (kTileN == 64)                                                                            \
            grouped_native_vnni_gate_up_swiglu_prefill_kernel<CB, TM, 64><<<cb_grid, block, 0, cuda_stream>>>( \
                d_scratch_A_int8, d_scratch_scales, d_gate_desc_table, d_up_desc_table,             \
                d_group_counts, d_group_offsets, d_active_expert_ids, active_expert_slots,          \
                d_scratch_swiglu_int8, d_scratch_swiglu_scales,                                    \
                intermediate, d_model);                                                             \
        else                                                                                         \
            grouped_native_vnni_gate_up_swiglu_prefill_kernel<CB, TM, 128><<<cb_grid, block, 0, cuda_stream>>>( \
                d_scratch_A_int8, d_scratch_scales, d_gate_desc_table, d_up_desc_table,             \
                d_group_counts, d_group_offsets, d_active_expert_ids, active_expert_slots,          \
                d_scratch_swiglu_int8, d_scratch_swiglu_scales,                                    \
                intermediate, d_model);                                                             \
    } while (0)
#define LAUNCH_GROUPED_GATEUP_PREFILL(CB)                                                          \
    do {                                                                                           \
        if (use_gateup_kpart) {                                                                     \
            if (kTileM == 16)      LAUNCH_GROUPED_GATEUP_KPART_PREFILL_TM(CB, 16);                 \
            else if (kTileM == 8)  LAUNCH_GROUPED_GATEUP_KPART_PREFILL_TM(CB, 8);                  \
            else if (kTileM == 4)  LAUNCH_GROUPED_GATEUP_KPART_PREFILL_TM(CB, 4);                  \
            else                   LAUNCH_GROUPED_GATEUP_KPART_PREFILL_TM(CB, 2);                  \
        } else if (fuse_swiglu) {                                                                   \
            if (kTileM == 16)      LAUNCH_GROUPED_GATEUP_SWIGLU_PREFILL_TM(CB, 16);                 \
            else if (kTileM == 8)  LAUNCH_GROUPED_GATEUP_SWIGLU_PREFILL_TM(CB, 8);                  \
            else if (kTileM == 4)  LAUNCH_GROUPED_GATEUP_SWIGLU_PREFILL_TM(CB, 4);                  \
            else                   LAUNCH_GROUPED_GATEUP_SWIGLU_PREFILL_TM(CB, 2);                  \
        } else {                                                                                    \
            if (kTileM == 16)      LAUNCH_GROUPED_GATEUP_PREFILL_TM(CB, 16);                        \
            else if (kTileM == 8)  LAUNCH_GROUPED_GATEUP_PREFILL_TM(CB, 8);                         \
            else if (kTileM == 4)  LAUNCH_GROUPED_GATEUP_PREFILL_TM(CB, 4);                         \
            else                   LAUNCH_GROUPED_GATEUP_PREFILL_TM(CB, 2);                         \
        }                                                                                           \
    } while (0)

            bool launched_gateup = false;
#define LAUNCH_GROUPED_GATEUP_IF_PRESENT(CB)                                                       \
    do {                                                                                           \
        if (gateup_codebook_mask & (uint32_t{1} << (CB))) {                                        \
            LAUNCH_GROUPED_GATEUP_PREFILL(CB);                                                      \
            if (!finishLaunch("cudaMoE_grouped_gate_up_prefill"))                                  \
                return false;                                                                       \
            launched_gateup = true;                                                                 \
        }                                                                                           \
    } while (0)

            LAUNCH_GROUPED_GATEUP_IF_PRESENT(0);
            LAUNCH_GROUPED_GATEUP_IF_PRESENT(4);
            LAUNCH_GROUPED_GATEUP_IF_PRESENT(5);
            LAUNCH_GROUPED_GATEUP_IF_PRESENT(6);
            LAUNCH_GROUPED_GATEUP_IF_PRESENT(7);
            LAUNCH_GROUPED_GATEUP_IF_PRESENT(8);
            LAUNCH_GROUPED_GATEUP_IF_PRESENT(9);
            LAUNCH_GROUPED_GATEUP_IF_PRESENT(10);
            LAUNCH_GROUPED_GATEUP_IF_PRESENT(11);
            LAUNCH_GROUPED_GATEUP_IF_PRESENT(12);
            LAUNCH_GROUPED_GATEUP_IF_PRESENT(13);
            LAUNCH_GROUPED_GATEUP_IF_PRESENT(14);
            LAUNCH_GROUPED_GATEUP_IF_PRESENT(15);
            LAUNCH_GROUPED_GATEUP_IF_PRESENT(16);
            LAUNCH_GROUPED_GATEUP_IF_PRESENT(17);
            LAUNCH_GROUPED_GATEUP_IF_PRESENT(19);
            if (!launched_gateup)
            {
                std::fprintf(stderr,
                             "[cudaMoE_grouped_prefill_pipeline] unsupported gate/up codebook_id=%u mask=0x%x\n",
                             static_cast<unsigned>(gateup_codebook_id),
                             static_cast<unsigned>(gateup_codebook_mask));
                return false;
            }

            if (use_gateup_kpart)
            {
                constexpr int kReduceTileN = 32;
                dim3 reduce_grid((intermediate + kReduceTileN - 1) / kReduceTileN, total_slots);
                dim3 reduce_block(kReduceTileN);
                grouped_native_vnni_gate_up_prefill_kpart_reduce_swiglu_kernel<<<
                    reduce_grid, reduce_block, 0, cuda_stream>>>(
                    d_gate_partials, d_up_partials,
                    d_scratch_swiglu_int8, d_scratch_swiglu_scales,
                    total_slots, intermediate, gateup_k_partitions);
                if (!finishLaunch("cudaMoE_grouped_gate_up_prefill_kpart_reduce_swiglu"))
                    return false;
            }

#undef LAUNCH_GROUPED_GATEUP_IF_PRESENT
#undef LAUNCH_GROUPED_GATEUP_PREFILL
#undef LAUNCH_GROUPED_GATEUP_KPART_PREFILL_TM
#undef LAUNCH_GROUPED_GATEUP_SWIGLU_PREFILL_TM
#undef LAUNCH_GROUPED_GATEUP_PREFILL_TM
        }

        // Separate SwiGLU + blockwise-quant pass. Skipped when fusion is enabled — the fused
        // gate/up kernel already produced d_scratch_swiglu_int8 / d_scratch_swiglu_scales.
        if (!use_gateup_kpart && !llaminar2::debugEnv().gemm.cuda_moe_prefill_fuse_swiglu)
        {
            dim3 grid(intermediate / 32, total_slots);
            dim3 block(32);
            grouped_prefill_swiglu_quantize_blockwise_kernel<<<grid, block, 0, cuda_stream>>>(
                d_scratch_gate, d_scratch_up,
                d_scratch_swiglu_int8, d_scratch_swiglu_scales,
                total_slots, intermediate);
            if (!finishLaunch("cudaMoE_grouped_swiglu_quantize_prefill"))
                return false;
        }

        {
            const int requestedTileM =
                llaminar2::debugEnv().gemm.cuda_moe_prefill_tile_m;
            const int kTileM =
                select_grouped_prefill_tile_m(requestedTileM, max_tokens_per_expert);
            const bool tiny_active_verifier =
                active_expert_slots > 0 && max_tokens_per_expert <= 4;
            const int kTileN = tiny_active_verifier ? 64 : 128;
            dim3 grid((d_model + kTileN - 1) / kTileN,
                      (max_tokens_per_expert + kTileM - 1) / kTileM,
                      expert_grid);
            dim3 block(kTileN);

#define LAUNCH_GROUPED_DOWN_PREFILL_TM(CB, TM)                                                     \
    do {                                                                                           \
        dim3 cb_grid(grid.x, grid.y, expert_grid);                                                  \
        if (kTileN == 64)                                                                            \
            grouped_native_vnni_down_prefill_kernel<CB, TM, 64><<<cb_grid, block, 0, cuda_stream>>>( \
                d_scratch_swiglu_int8, d_scratch_swiglu_scales, d_down_desc_table,                  \
                d_group_counts, d_group_offsets, d_active_expert_ids, active_expert_slots,          \
                d_scratch_down_out, d_model, intermediate);                                         \
        else                                                                                         \
            grouped_native_vnni_down_prefill_kernel<CB, TM, 128><<<cb_grid, block, 0, cuda_stream>>>( \
                d_scratch_swiglu_int8, d_scratch_swiglu_scales, d_down_desc_table,                  \
                d_group_counts, d_group_offsets, d_active_expert_ids, active_expert_slots,          \
                d_scratch_down_out, d_model, intermediate);                                         \
    } while (0)
#define LAUNCH_GROUPED_DOWN_PREFILL(CB)                                                            \
    do {                                                                                           \
        if (kTileM == 16)      LAUNCH_GROUPED_DOWN_PREFILL_TM(CB, 16);                              \
        else if (kTileM == 8)  LAUNCH_GROUPED_DOWN_PREFILL_TM(CB, 8);                               \
        else if (kTileM == 4)  LAUNCH_GROUPED_DOWN_PREFILL_TM(CB, 4);                               \
        else                   LAUNCH_GROUPED_DOWN_PREFILL_TM(CB, 2);                               \
    } while (0)

            bool launched_down = false;
#define LAUNCH_GROUPED_DOWN_IF_PRESENT(CB)                                                         \
    do {                                                                                           \
        if (down_codebook_mask & (uint32_t{1} << (CB))) {                                          \
            LAUNCH_GROUPED_DOWN_PREFILL(CB);                                                        \
            if (!finishLaunch("cudaMoE_grouped_down_prefill"))                                    \
                return false;                                                                       \
            launched_down = true;                                                                   \
        }                                                                                           \
    } while (0)

            LAUNCH_GROUPED_DOWN_IF_PRESENT(0);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(4);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(5);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(6);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(7);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(8);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(9);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(10);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(11);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(12);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(13);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(14);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(15);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(16);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(17);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(19);
            if (!launched_down)
            {
                std::fprintf(stderr,
                             "[cudaMoE_grouped_prefill_pipeline] unsupported down codebook_id=%u mask=0x%x\n",
                             static_cast<unsigned>(down_codebook_id),
                             static_cast<unsigned>(down_codebook_mask));
                return false;
            }

#undef LAUNCH_GROUPED_DOWN_IF_PRESENT
#undef LAUNCH_GROUPED_DOWN_PREFILL
#undef LAUNCH_GROUPED_DOWN_PREFILL_TM
        }

        {
            constexpr int kTileN = 64;
            dim3 block(kTileN);
            if (d_original_to_grouped)
            {
                const int seq_len = total_slots / top_k;
                dim3 grid((d_model + kTileN - 1) / kTileN, seq_len);
                grouped_prefill_scatter_weighted_ordered_kernel<<<grid, block, 0, cuda_stream>>>(
                    d_output, d_scratch_down_out,
                    d_original_to_grouped, d_group_weights,
                    seq_len, top_k, d_model);
            }
            else
            {
                dim3 grid((d_model + kTileN - 1) / kTileN, total_slots);
                grouped_prefill_scatter_weighted_kernel<<<grid, block, 0, cuda_stream>>>(
                    d_output, d_scratch_down_out,
                    d_group_token_indices, d_group_weights,
                    total_slots, d_model);
            }
            if (!finishLaunch("cudaMoE_grouped_scatter_prefill"))
                return false;
        }

        return true;
    }
}
