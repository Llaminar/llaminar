/**
 * @file CUDAMoEDecodeRouterBody.cuh
 * @brief Shared inline CUDA decode softmax/top-k runtime router bodies for RouterAndGather and RetainedHCFFN.
 */
#pragma once
#include "CUDAMoERebalanceCommon.cuh"

namespace
{
    /**
     * @brief Publish one decode route with compile-time maintenance policy.
     *
     * Static graphs instantiate `<false, false>` so the compiler removes both
     * global histogram atomics and every ready-wave control read. Observe and
     * Dynamic graphs retain the exact evidence/apply work they own. Keeping
     * these as template policies prevents a captured Static graph from paying
     * register and branch costs for maintenance pointers that are always null.
     */
    template <bool UpdateRuntimeHistogram, bool ApplyReadyRebalance, int ExpertCapacity = kMaxExperts>
    __device__ __forceinline__ void softmax_topk_decode_runtime_body(
        float *__restrict__ logits,
        DeviceMoELayerRuntimeView *__restrict__ runtime,
        DeviceMoELayerRuntimeView *__restrict__ runtime_layers,
        float *legacy_indices,
        float *legacy_weights,
        int num_experts, int top_k,
        bool normalize_weights,
        bool write_legacy_outputs,
        llaminar2::RoutedExpertRowExecutionPolicy row_execution_policy,
        const int32_t *__restrict__ absolute_position_ids,
        const DeviceMoERebalancePlanEntryView *rebalance_plan_entries,
        uint32_t rebalance_plan_capacity,
        DeviceMoERebalanceCommandBufferHeaderView *rebalance_command_header,
        const DeviceMoEExpertDirectoryEntryView *rebalance_local_transfer_slots,
        uint32_t rebalance_local_transfer_slot_count,
        DeviceMoERebalanceConfigView rebalance_config,
        DeviceMoERebalanceApplyStatusView *rebalance_apply_status,
        DeviceMoERebalanceGraphControllerStateView *rebalance_controller_state,
        int rebalance_target_layer,
        uint32_t rebalance_command_buffer_count,
        const int32_t *__restrict__ active_row_count_device)
    {
        __shared__ float values[ExpertCapacity];
        __shared__ float reductions[kThreads];
        __shared__ int red_idx[kThreads];
        __shared__ int selected[kMaxTopK];
        __shared__ float selected_weights[kMaxTopK];
        __shared__ RuntimeDecodeDispatchScratch dispatch;

        if constexpr (ApplyReadyRebalance)
        {
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
        }

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

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if constexpr (ExpertCapacity == 512)
            retainedTopK512(values, top_k, selected, selected_weights, reductions, red_idx);
        else
#endif
            moe_select_topk_probabilities_block(
                values, num_experts, top_k, selected, selected_weights, reductions, red_idx);

        if (threadIdx.x == 0)
        {
            runtime_require_decode_row_policy(runtime, row_execution_policy);
            const bool row_active = runtime_decode_row_active(active_row_count_device);
            const uint32_t execution_bank =
                runtime_require_decode_execution_bank(runtime);
            if (execution_bank > 1u)
                return;
            const bool shape_ok = runtime_shape_ok(
                runtime, execution_bank, num_experts, top_k);
            if (!runtime_route_assignment_ledger_ok(runtime, top_k))
            {
                FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                    "decode route assignment ledger is absent or undersized");
                return;
            }
            float topk_sum = 0.0f;
            for (int k = 0; k < top_k; ++k)
                topk_sum += selected_weights[k];

            if (shape_ok && row_active)
            {
                runtime_resolve_decode_dispatch(
                    runtime,
                    execution_bank,
                    selected,
                    num_experts,
                    top_k,
                    absolute_position_ids
                        ? absolute_position_ids[0]
                        : -1,
                    row_execution_policy,
                    UpdateRuntimeHistogram,
                    dispatch);
            }
            for (int k = 0; k < top_k; ++k)
            {
                const float weight = normalize_weights && topk_sum > 0.0f
                                         ? selected_weights[k] / topk_sum
                                         : selected_weights[k];
                if (shape_ok)
                {
                    const bool local_compute = row_active && dispatch.local_compute_flags[k];
                    runtime->topk_expert_ids[k] = local_compute ? selected[k] : -1;
                    runtime->topk_weights[k] = local_compute ? weight : 0.0f;
                    runtime->route_participant_ids[k] =
                        row_active ? dispatch.assigned_participants[k] : -1;
                }
                if (write_legacy_outputs)
                {
                    legacy_indices[k] = row_active ? static_cast<float>(selected[k]) : -1.0f;
                    legacy_weights[k] = row_active ? weight : 0.0f;
                }
            }
            if constexpr (UpdateRuntimeHistogram)
            {
                if (!row_active || !runtime_histogram_admits_rows(*runtime))
                    return;
                for (int k = 0; k < top_k; ++k)
                {
                    const int expert = selected[k];
                    if (expert < 0 || expert >= kDeviceMoEMaxExperts)
                        continue;
                    atomicAdd(reinterpret_cast<unsigned long long *>(
                                  &runtime_selected_histogram(*runtime, 0u)[expert]),
                              static_cast<unsigned long long>(1));
                    if (shape_ok && dispatch.local_compute_flags[k])
                    {
                        atomicAdd(reinterpret_cast<unsigned long long *>(
                                      &runtime_local_histogram(*runtime, 0u)[expert]),
                                  static_cast<unsigned long long>(1));
                    }
                }
            }
        }
    }


    template <bool UpdateRuntimeHistogram,
              llaminar2::RoutedExpertRowExecutionPolicy RowPolicy = llaminar2::RoutedExpertRowExecutionPolicy::ParticipantAssigned,
              bool PublishDescriptors = false, int ExpertCapacity = kMaxExperts>
    __device__ __forceinline__ void softmax_topk_decode_independent_body(
        float *__restrict__ logits,
        DeviceMoELayerRuntimeView *__restrict__ runtime,
        float *legacy_indices,
        float *legacy_weights,
        int num_experts,
        int top_k,
        bool normalize_weights,
        bool write_legacy_outputs,
        const int32_t *__restrict__ active_row_count_device,
        llaminar2::MoEDecodeDescriptorPublication publication)
    {
        __shared__ float values[ExpertCapacity];
        __shared__ float reductions[kThreads];
        __shared__ int red_idx[kThreads];
        __shared__ int selected[kMaxTopK];
        __shared__ float selected_weights[kMaxTopK];

        float local_max = -INFINITY;
        for (int expert = threadIdx.x; expert < num_experts;
             expert += blockDim.x)
        {
            local_max = fmaxf(local_max, logits[expert]);
        }
        reductions[threadIdx.x] = local_max;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
                reductions[threadIdx.x] = fmaxf(
                    reductions[threadIdx.x],
                    reductions[threadIdx.x + stride]);
            __syncthreads();
        }
        const float max_value = reductions[0];

        float local_sum = 0.0f;
        for (int expert = threadIdx.x; expert < num_experts;
             expert += blockDim.x)
        {
            const float probability = expf(logits[expert] - max_value);
            values[expert] = probability;
            local_sum += probability;
        }
        reductions[threadIdx.x] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
                reductions[threadIdx.x] += reductions[threadIdx.x + stride];
            __syncthreads();
        }
        const float denominator = reductions[0];
        for (int expert = threadIdx.x; expert < num_experts;
             expert += blockDim.x)
        {
            const float probability =
                denominator > 0.0f ? values[expert] / denominator : 0.0f;
            values[expert] = probability;
            logits[expert] = probability;
        }
        __syncthreads();

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if constexpr (ExpertCapacity == 512)
            retainedTopK512(values, top_k, selected, selected_weights, reductions, red_idx);
        else
#endif
            moe_select_topk_probabilities_block(
                values, num_experts, top_k, selected, selected_weights, reductions, red_idx);

        static_assert(RowPolicy == llaminar2::RoutedExpertRowExecutionPolicy::ParticipantAssigned ||
                      RowPolicy == llaminar2::RoutedExpertRowExecutionPolicy::IntermediateShardLocal);
        if (threadIdx.x == 0)
        {
            float topk_sum = 0.0f;
            for (int slot = 0; slot < top_k; ++slot)
                topk_sum += selected_weights[slot];
            if (normalize_weights && topk_sum > 0.0f)
                for (int slot = 0; slot < top_k; ++slot)
                    selected_weights[slot] = selected_weights[slot] / topk_sum;
        }
        // Only metadata publication is parallel: preserve the serial FP32
        // sum/division and make its exact bits visible to every selected lane.
        __syncthreads();
        const int slot = static_cast<int>(threadIdx.x);
        if (slot >= top_k) return;

        const uint32_t execution_bank =
            runtime_require_decode_execution_bank(runtime);
        if (execution_bank > 1u)
            return;
        const bool shape_ok = runtime_shape_ok(
            runtime, execution_bank, num_experts, top_k);
        if (!runtime_route_assignment_ledger_ok(runtime, top_k))
        {
            FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                "unique-owner decode route ledger is absent or undersized");
            return;
        }
        const auto &bank = runtime_placement_banks(runtime)[execution_bank];
        if (RowPolicy == llaminar2::RoutedExpertRowExecutionPolicy::ParticipantAssigned &&
            shape_ok && runtime_multi_resident_expert_count(bank) != 0u)
        {
            FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                "unique-owner decode graph observed a replicated expert");
            return;
        }

        runtime_require_decode_row_policy(runtime, RowPolicy);
        const bool row_active = runtime_decode_row_active(active_row_count_device);
        const bool collect_histogram =
            UpdateRuntimeHistogram &&
            runtime_histogram_admits_rows(*runtime);
        {
            const int expert_id = row_active ? selected[slot] : -1;
            const float weight = selected_weights[slot];
            bool local_compute = false;
            if (shape_ok)
            {
                int participant = -1;
                if constexpr (RowPolicy == llaminar2::RoutedExpertRowExecutionPolicy::IntermediateShardLocal)
                {
                    if (expert_id != -1)
                    {
                        const auto selection = runtime_intermediate_shard_row(runtime, bank, expert_id);
                        local_compute = selection.disposition == llaminar2::MoEIntermediateShardRowDisposition::Local;
                        participant = selection.participant;
                    }
                }
                else
                {
                    const bool selected = expert_id >= 0 && expert_id < num_experts;
                    local_compute = selected && bank.local_compute_mask[expert_id] != 0u;
                    participant = selected ? runtime_expert_owner(bank, expert_id) : -1;
                }
                runtime->topk_expert_ids[slot] =
                    local_compute ? expert_id : -1;
                runtime->topk_weights[slot] =
                    local_compute ? weight : 0.0f;
                runtime->route_participant_ids[slot] =
                    participant;
                if constexpr (PublishDescriptors)
                {
                    if (local_compute)
                    {
                        const auto &descriptor = bank.experts[expert_id];
                        bool ready = descriptor.weight_format == publication.format;
                        if constexpr (RowPolicy == llaminar2::RoutedExpertRowExecutionPolicy::ParticipantAssigned)
                            ready = ready && descriptor.local_slot >= 0 &&
                                (bank.resident_participant_mask[expert_id] & runtime_participant_bit(runtime->participant_id)) &&
                                llaminar2::deviceMoEExpertProjectionReady(descriptor, publication.projections);
                        llaminar2::publishMoEDecodeDescriptor(publication, descriptor, expert_id, ready);
                    }
                }
            }
            if (write_legacy_outputs)
            {
                legacy_indices[slot] = static_cast<float>(expert_id);
                legacy_weights[slot] = row_active ? weight : 0.0f;
            }
            if constexpr (UpdateRuntimeHistogram)
            {
                if (!collect_histogram || expert_id < 0 ||
                    expert_id >= num_experts)
                {
                    return;
                }
                atomicAdd(
                    reinterpret_cast<unsigned long long *>(
                        &runtime_selected_histogram(*runtime, 0u)[expert_id]),
                    static_cast<unsigned long long>(1));
                if (shape_ok && local_compute)
                {
                    atomicAdd(
                        reinterpret_cast<unsigned long long *>(
                            &runtime_local_histogram(*runtime, 0u)[expert_id]),
                        static_cast<unsigned long long>(1));
                }
            }
        }
    }

} // namespace
