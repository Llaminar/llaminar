#pragma once

#include "../local_execution/device/WorkspaceDescriptor.h"
#include "../../tensors/TensorKernels.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace llaminar2
{
    namespace MoEWorkspaceBuffers
    {
        constexpr const char *STAGING_INDICES = "moe_staging_indices";
        constexpr const char *STAGING_WEIGHTS = "moe_staging_weights";
        constexpr const char *ROUTE_LOGITS = "moe_route_logits";
        constexpr const char *ROUTE_INDICES = "moe_route_indices";
        constexpr const char *ROUTE_WEIGHTS = "moe_route_weights";

        constexpr const char *GROUP_INT_INDICES = "moe_group_int_indices";
        constexpr const char *GROUP_OFFSETS = "moe_group_offsets";
        constexpr const char *GROUP_COUNTS = "moe_group_counts";
        constexpr const char *GROUP_TOKEN_INDICES = "moe_group_token_indices";
        constexpr const char *GROUP_ORIGINAL_TO_GROUPED = "moe_group_original_to_grouped";
        constexpr const char *GROUP_ORIGINAL_EXPERT_IDS = "moe_group_original_expert_ids";
        constexpr const char *GROUP_WRITE_HEADS = "moe_group_write_heads";
        constexpr const char *GROUP_WEIGHTS = "moe_group_weights";
        constexpr const char *GROUP_ACTIVE_EXPERT_IDS = "moe_group_active_expert_ids";

        constexpr const char *PREFILL_A_INT8 = "moe_prefill_a_int8";
        constexpr const char *PREFILL_A_SCALES = "moe_prefill_a_scales";
        constexpr const char *PREFILL_SWIGLU_INT8 = "moe_prefill_swiglu_int8";
        constexpr const char *PREFILL_SWIGLU_SCALES = "moe_prefill_swiglu_scales";
        constexpr const char *PREFILL_GATE = "moe_prefill_gate";
        constexpr const char *PREFILL_UP = "moe_prefill_up";

        constexpr const char *DECODE_HIDDEN_INT8 = "moe_decode_hidden_int8";
        constexpr const char *DECODE_HIDDEN_SCALES = "moe_decode_hidden_scales";
        constexpr const char *GATEUP_GATE_PARTIALS = "moe_grouped_gateup_gate_partials";
        constexpr const char *GATEUP_UP_PARTIALS = "moe_grouped_gateup_up_partials";
        constexpr const char *DOWN_PARTIALS = "moe_grouped_down_partials";
        constexpr const char *DECODE_SWIGLU_INT8 = "moe_decode_swiglu_int8";
        constexpr const char *DECODE_SWIGLU_SCALES = "moe_decode_swiglu_scales";
        constexpr const char *DECODE_EXPERT_IDS = "moe_grouped_decode_expert_ids";
        constexpr const char *DECODE_WEIGHTS = "moe_grouped_decode_weights";
        constexpr const char *CUDA_DECODE_GATEUP_GATE_PTRS = "cuda_moe_decode_gateup_gate_ptrs";
        constexpr const char *CUDA_DECODE_GATEUP_UP_PTRS = "cuda_moe_decode_gateup_up_ptrs";
        constexpr const char *CUDA_DECODE_DOWN_GATE_PTRS = "cuda_moe_decode_down_gate_ptrs";
        constexpr const char *CUDA_DECODE_DOWN_UP_PTRS = "cuda_moe_decode_down_up_ptrs";

        constexpr const char *ROCM_SHARED_GATE = "rocm_moe_shared_gate";
        constexpr const char *ROCM_ROUTE_LOGITS_PARTIALS = "rocm_moe_route_logits_partials";
        constexpr const char *ROCM_ROUTER_Q8_HIDDEN = "rocm_moe_router_q8_hidden";
        constexpr const char *ROCM_ROUTER_Q8_SCALES = "rocm_moe_router_q8_scales";
        constexpr const char *ROCM_GROUP_MAX_TOKENS = "rocm_moe_group_max_tokens";
        constexpr const char *ROCM_DECODE_GATE_PTRS = "rocm_moe_decode_gate_ptrs";
        constexpr const char *ROCM_DECODE_UP_PTRS = "rocm_moe_decode_up_ptrs";
        constexpr const char *ROCM_DECODE_GATE_OUTPUT_PTRS = "rocm_moe_decode_gate_output_ptrs";
        constexpr const char *ROCM_DECODE_UP_OUTPUT_PTRS = "rocm_moe_decode_up_output_ptrs";
        constexpr const char *ROCM_DECODE_DOWN_DESCS = "rocm_moe_decode_down_descs";

        constexpr int kRuntimePointerTableSlots = 1024;
        constexpr int kRuntimePointerWorkspaceScopes = 3;
        constexpr int kRuntimePointerWorkspaceEntries =
            kRuntimePointerTableSlots * kRuntimePointerWorkspaceScopes;
        constexpr int kRuntimePointerArrayMaxTopK = 16;

        inline int ceilDiv(int value, int divisor)
        {
            return (value + divisor - 1) / divisor;
        }

        inline void add(WorkspaceRequirements &reqs, const char *name, std::size_t bytes)
        {
            if (bytes == 0)
                return;
            reqs.buffers.push_back({name, bytes, 256, true});
        }

        inline WorkspaceRequirements routing(int max_seq_len, int num_experts, int top_k)
        {
            WorkspaceRequirements reqs;
            max_seq_len = std::max(1, max_seq_len);
            num_experts = std::max(1, num_experts);
            top_k = std::max(1, top_k);

            const std::size_t tokens = static_cast<std::size_t>(max_seq_len);
            const std::size_t route_slots = tokens * static_cast<std::size_t>(top_k);
            add(reqs, ROUTE_LOGITS, tokens * static_cast<std::size_t>(num_experts) * sizeof(float));
            add(reqs, ROUTE_INDICES, route_slots * sizeof(int));
            add(reqs, ROUTE_WEIGHTS, route_slots * sizeof(float));
            return reqs;
        }

        inline WorkspaceRequirements expertExecution(
            int max_seq_len,
            int d_model,
            int intermediate,
            int num_experts,
            int top_k)
        {
            WorkspaceRequirements reqs;
            max_seq_len = std::max(1, max_seq_len);
            d_model = std::max(1, d_model);
            intermediate = std::max(1, intermediate);
            num_experts = std::max(1, num_experts);
            top_k = std::max(1, top_k);

            const std::size_t tokens = static_cast<std::size_t>(max_seq_len);
            const std::size_t total_slots = tokens * static_cast<std::size_t>(top_k);
            const std::size_t active_expert_id_slots = std::max(
                total_slots,
                static_cast<std::size_t>(num_experts));
            const int max_dim = std::max(d_model, intermediate);
            const int max_blocks = ceilDiv(max_dim, 32);
            const int d_model_blocks = ceilDiv(d_model, 32);
            const int intermediate_blocks = ceilDiv(intermediate, 32);

            add(reqs, STAGING_INDICES, tokens * sizeof(int));
            add(reqs, STAGING_WEIGHTS, tokens * sizeof(float));

            add(reqs, GROUP_INT_INDICES, total_slots * sizeof(int));
            add(reqs, GROUP_TOKEN_INDICES, total_slots * sizeof(int));
            add(reqs, GROUP_ORIGINAL_TO_GROUPED, total_slots * sizeof(int));
            add(reqs, GROUP_ORIGINAL_EXPERT_IDS, total_slots * sizeof(int));
            add(reqs, GROUP_WEIGHTS, total_slots * sizeof(float));
            add(reqs, GROUP_ACTIVE_EXPERT_IDS, active_expert_id_slots * sizeof(int));
            add(reqs, GROUP_OFFSETS, static_cast<std::size_t>(num_experts) * sizeof(int));
            add(reqs, GROUP_COUNTS, static_cast<std::size_t>(num_experts) * sizeof(int));
            add(reqs, GROUP_WRITE_HEADS, static_cast<std::size_t>(num_experts) * sizeof(int));

            add(reqs, PREFILL_A_INT8, total_slots * static_cast<std::size_t>(max_dim) * sizeof(int8_t));
            add(reqs, PREFILL_A_SCALES, total_slots * static_cast<std::size_t>(max_blocks) * sizeof(float));
            add(reqs, PREFILL_SWIGLU_INT8, total_slots * static_cast<std::size_t>(intermediate) * sizeof(int8_t));
            add(reqs, PREFILL_SWIGLU_SCALES, total_slots * static_cast<std::size_t>(intermediate_blocks) * sizeof(float));
            add(reqs, PREFILL_GATE, total_slots * static_cast<std::size_t>(max_dim) * sizeof(float));
            add(reqs, PREFILL_UP, total_slots * static_cast<std::size_t>(intermediate) * sizeof(float));

            constexpr int kMaxGateUpPartitions = 32;
            constexpr int kMaxDownPartitions = 16;
            constexpr int kMaxVerifierRows = 4;
            const std::size_t decode_slots = static_cast<std::size_t>(top_k);
            const std::size_t verifier_splitk_slots =
                static_cast<std::size_t>(std::min(max_seq_len, kMaxVerifierRows)) *
                static_cast<std::size_t>(top_k);
            const std::size_t gateup_partial_slots =
                std::max(decode_slots, verifier_splitk_slots);

            add(reqs, DECODE_HIDDEN_INT8, static_cast<std::size_t>(d_model) * sizeof(int8_t));
            add(reqs, DECODE_HIDDEN_SCALES, static_cast<std::size_t>(d_model_blocks) * sizeof(float));
            add(reqs, GATEUP_GATE_PARTIALS,
                gateup_partial_slots * kMaxGateUpPartitions * static_cast<std::size_t>(intermediate) * sizeof(float));
            add(reqs, GATEUP_UP_PARTIALS,
                gateup_partial_slots * kMaxGateUpPartitions * static_cast<std::size_t>(intermediate) * sizeof(float));
            add(reqs, DOWN_PARTIALS,
                kMaxDownPartitions * static_cast<std::size_t>(d_model) * sizeof(float));
            add(reqs, DECODE_SWIGLU_INT8,
                decode_slots * static_cast<std::size_t>(intermediate) * sizeof(int8_t));
            add(reqs, DECODE_SWIGLU_SCALES,
                decode_slots * static_cast<std::size_t>(intermediate_blocks) * sizeof(float));
            add(reqs, DECODE_EXPERT_IDS, decode_slots * sizeof(int));
            add(reqs, DECODE_WEIGHTS, decode_slots * sizeof(float));
            add(reqs, CUDA_DECODE_GATEUP_GATE_PTRS,
                static_cast<std::size_t>(kRuntimePointerWorkspaceEntries) *
                    kRuntimePointerArrayMaxTopK * sizeof(float *));
            add(reqs, CUDA_DECODE_GATEUP_UP_PTRS,
                static_cast<std::size_t>(kRuntimePointerWorkspaceEntries) *
                    kRuntimePointerArrayMaxTopK * sizeof(float *));
            add(reqs, CUDA_DECODE_DOWN_GATE_PTRS,
                static_cast<std::size_t>(kRuntimePointerWorkspaceEntries) *
                    kRuntimePointerArrayMaxTopK * sizeof(const float *));
            add(reqs, CUDA_DECODE_DOWN_UP_PTRS,
                static_cast<std::size_t>(kRuntimePointerWorkspaceEntries) *
                    kRuntimePointerArrayMaxTopK * sizeof(const float *));
            return reqs;
        }

        inline WorkspaceRequirements cudaMoE(
            int max_seq_len,
            int d_model,
            int intermediate,
            int num_experts,
            int top_k)
        {
            WorkspaceRequirements reqs = routing(max_seq_len, num_experts, top_k);
            reqs.merge(expertExecution(max_seq_len, d_model, intermediate, num_experts, top_k));
            return reqs;
        }

        inline WorkspaceRequirements rocmRouting(
            int max_seq_len,
            int d_model,
            int num_experts,
            int top_k)
        {
            WorkspaceRequirements reqs = routing(max_seq_len, num_experts, top_k);
            max_seq_len = std::max(1, max_seq_len);
            d_model = std::max(1, d_model);
            num_experts = std::max(1, num_experts);
            top_k = std::max(1, top_k);

            const int d_model_blocks = ceilDiv(d_model, 32);
            constexpr int kMaxRouterPartitions = 16;

            add(reqs, ROCM_ROUTE_LOGITS_PARTIALS,
                static_cast<std::size_t>(num_experts) * kMaxRouterPartitions * sizeof(float));
            add(reqs, ROCM_ROUTER_Q8_HIDDEN, static_cast<std::size_t>(d_model) * sizeof(int8_t));
            add(reqs, ROCM_ROUTER_Q8_SCALES, static_cast<std::size_t>(d_model_blocks) * sizeof(float));
            return reqs;
        }

        inline WorkspaceRequirements rocmMoE(
            int max_seq_len,
            int d_model,
            int intermediate,
            int num_experts,
            int top_k)
        {
            WorkspaceRequirements reqs = rocmRouting(max_seq_len, d_model, num_experts, top_k);
            reqs.merge(expertExecution(max_seq_len, d_model, intermediate, num_experts, top_k));
            max_seq_len = std::max(1, max_seq_len);
            top_k = std::max(1, top_k);

            const std::size_t decode_slots = static_cast<std::size_t>(top_k);

            add(reqs, ROCM_SHARED_GATE, static_cast<std::size_t>(max_seq_len) * sizeof(float));
            add(reqs, ROCM_GROUP_MAX_TOKENS, sizeof(int));
            /*
             * Grouped decode pointer arrays are captured by value as device
             * addresses. A full decode graph contains many MoE stages, so ROCm
             * reserves deterministic graph-owned pointer slots; one mutable slot
             * would make all captured stages replay the last staged scratch
             * pointer set.
             */
            add(reqs, ROCM_DECODE_GATE_PTRS,
                static_cast<std::size_t>(kRuntimePointerWorkspaceEntries) *
                    kRuntimePointerArrayMaxTopK * sizeof(const float *));
            add(reqs, ROCM_DECODE_UP_PTRS,
                static_cast<std::size_t>(kRuntimePointerWorkspaceEntries) *
                    kRuntimePointerArrayMaxTopK * sizeof(const float *));
            add(reqs, ROCM_DECODE_GATE_OUTPUT_PTRS,
                static_cast<std::size_t>(kRuntimePointerWorkspaceEntries) *
                    kRuntimePointerArrayMaxTopK * sizeof(float *));
            add(reqs, ROCM_DECODE_UP_OUTPUT_PTRS,
                static_cast<std::size_t>(kRuntimePointerWorkspaceEntries) *
                    kRuntimePointerArrayMaxTopK * sizeof(float *));
            add(reqs, ROCM_DECODE_DOWN_DESCS,
                decode_slots * sizeof(DeviceNativeVNNIMatrixDesc));
            return reqs;
        }
    } // namespace MoEWorkspaceBuffers
} // namespace llaminar2
