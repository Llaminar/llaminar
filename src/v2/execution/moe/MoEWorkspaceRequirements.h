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
        constexpr const char *PREFILL_EFFECTIVE_SEQ_LEN = "moe_prefill_effective_seq_len";

        constexpr const char *GROUP_INT_INDICES = "moe_group_int_indices";
        constexpr const char *GROUP_OFFSETS = "moe_group_offsets";
        constexpr const char *GROUP_COUNTS = "moe_group_counts";
        constexpr const char *GROUP_TOKEN_INDICES = "moe_group_token_indices";
        constexpr const char *GROUP_ORIGINAL_TO_GROUPED = "moe_group_original_to_grouped";
        constexpr const char *GROUP_ORIGINAL_EXPERT_IDS = "moe_group_original_expert_ids";
        constexpr const char *GROUP_WRITE_HEADS = "moe_group_write_heads";
        constexpr const char *GROUP_WEIGHTS = "moe_group_weights";
        constexpr const char *GROUP_ACTIVE_EXPERT_IDS = "moe_group_active_expert_ids";
        constexpr const char *GROUP_EXPERT_MASK = "moe_group_expert_mask";

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
        constexpr const char *CUDA_ROUTING_DECODE_EXPERT_IDS = "cuda_moe_routing_decode_expert_ids";
        constexpr const char *CUDA_DECODE_GATEUP_GATE_PTRS = "cuda_moe_decode_gateup_gate_ptrs";
        constexpr const char *CUDA_DECODE_GATEUP_UP_PTRS = "cuda_moe_decode_gateup_up_ptrs";
        constexpr const char *CUDA_DECODE_DOWN_GATE_PTRS = "cuda_moe_decode_down_gate_ptrs";
        constexpr const char *CUDA_DECODE_DOWN_UP_PTRS = "cuda_moe_decode_down_up_ptrs";
        constexpr const char *CUDA_GROUPED_GATE_DESC_TABLES = "cuda_moe_grouped_gate_desc_tables";
        constexpr const char *CUDA_GROUPED_UP_DESC_TABLES = "cuda_moe_grouped_up_desc_tables";
        constexpr const char *CUDA_GROUPED_DOWN_DESC_TABLES = "cuda_moe_grouped_down_desc_tables";
        constexpr const char *CUDA_RUNTIME_PREFILL_GATE_DESC_TABLE = "cuda_moe_runtime_prefill_gate_desc_table";
        constexpr const char *CUDA_RUNTIME_PREFILL_UP_DESC_TABLE = "cuda_moe_runtime_prefill_up_desc_table";
        constexpr const char *CUDA_RUNTIME_PREFILL_DOWN_DESC_TABLE = "cuda_moe_runtime_prefill_down_desc_table";
        constexpr const char *CUDA_ROUTER_Q8_GATE_WEIGHTS = "cuda_moe_router_q8_gate_weights";
        constexpr const char *CUDA_ROUTER_Q8_GATE_SCALES = "cuda_moe_router_q8_gate_scales";

        constexpr const char *ROCM_SHARED_GATE = "rocm_moe_shared_gate";
        constexpr const char *ROCM_ROUTE_LOGITS_PARTIALS = "rocm_moe_route_logits_partials";
        constexpr const char *ROCM_ROUTER_Q8_HIDDEN = "rocm_moe_router_q8_hidden";
        constexpr const char *ROCM_ROUTER_Q8_SCALES = "rocm_moe_router_q8_scales";
        constexpr const char *ROCM_ROUTER_Q8_GATE_WEIGHTS = "rocm_moe_router_q8_gate_weights";
        constexpr const char *ROCM_ROUTER_Q8_GATE_SCALES = "rocm_moe_router_q8_gate_scales";
        constexpr const char *ROCM_ROUTER_FP16_GATE_WEIGHTS = "rocm_moe_router_fp16_gate_weights";
        constexpr const char *ROCM_GROUP_MAX_TOKENS = "rocm_moe_group_max_tokens";
        constexpr const char *ROCM_DECODE_GATE_PTRS = "rocm_moe_decode_gate_ptrs";
        constexpr const char *ROCM_DECODE_UP_PTRS = "rocm_moe_decode_up_ptrs";
        constexpr const char *ROCM_DECODE_GATE_OUTPUT_PTRS = "rocm_moe_decode_gate_output_ptrs";
        constexpr const char *ROCM_DECODE_UP_OUTPUT_PTRS = "rocm_moe_decode_up_output_ptrs";
        constexpr const char *ROCM_DECODE_DOWN_DESCS = "rocm_moe_decode_down_descs";
        constexpr const char *ROCM_GROUPED_GATE_DESC_TABLES = "rocm_moe_grouped_gate_desc_tables";
        constexpr const char *ROCM_GROUPED_UP_DESC_TABLES = "rocm_moe_grouped_up_desc_tables";
        constexpr const char *ROCM_GROUPED_DOWN_DESC_TABLES = "rocm_moe_grouped_down_desc_tables";
        constexpr const char *ROCM_RUNTIME_PREFILL_GATE_DESC_TABLE = "rocm_moe_runtime_prefill_gate_desc_table";
        constexpr const char *ROCM_RUNTIME_PREFILL_UP_DESC_TABLE = "rocm_moe_runtime_prefill_up_desc_table";
        constexpr const char *ROCM_RUNTIME_PREFILL_DOWN_DESC_TABLE = "rocm_moe_runtime_prefill_down_desc_table";

        /**
         * @brief Maximum number of MoE transformer layers resident on one device.
         *
         * Router-weight caches and histogram tables are owned once per model
         * layer. Keeping this architectural bound named separately prevents
         * graph-role metadata from accidentally inheriting the smaller
         * one-owner-per-layer capacity again.
         */
        constexpr int kMaximumMoELayersPerDevice = 128;
        /**
         * @brief Maximum retained grouped-descriptor identities per MoE layer.
         *
         * One model layer can simultaneously retain descriptor identities for
         * ordinary prefill, serial decode, grouped verifier, MTP sidecars, and
         * graph-bucket/topology variants. These owners coexist because captured
         * graphs record descriptor addresses for their complete lifetime.
         *
         * This is an ownership bound, not an execution-depth or request-count
         * bound. Slot acquisition occurs while graph metadata is prepared; graph
         * launch and replay only dereference the already leased device address.
         */
        constexpr int kMaximumGroupedDescriptorOwnersPerMoELayer = 8;
        constexpr int kGroupedDescriptorTableSlots =
            kMaximumMoELayersPerDevice *
            kMaximumGroupedDescriptorOwnersPerMoELayer;
        constexpr int kRuntimePointerTableSlots = kGroupedDescriptorTableSlots;
        constexpr int kRuntimePointerWorkspaceScopes = 3;
        constexpr int kRuntimePointerWorkspaceEntries =
            kRuntimePointerTableSlots * kRuntimePointerWorkspaceScopes;
        constexpr int kRuntimePointerArrayMaxTopK = 16;
        /**
         * @brief Per-device capacity for graph-lifetime MoE metadata owners.
         *
         * Descriptor tables and fixed-topology masks are immutable inputs to a
         * captured graph. They therefore use one leased slot per live MoE
         * pipeline owner instead of the request scratch region shared by
         * sequential stages.
         */
        constexpr int kRouterGateCacheSlots = kMaximumMoELayersPerDevice;
        static_assert(
            kGroupedDescriptorTableSlots <= kRuntimePointerTableSlots,
            "Every persistent descriptor identity needs a matching runtime pointer-table identity");
        /**
         * @brief Reusable row tile used by verifier split-K partial buffers.
         *
         * This is a workspace tile, not an MTP depth limit. Runtime verifier
         * rows beyond this tile are processed by repeated grouped tiles while
         * preserving the serial decode reduction order inside each row.
         */
        constexpr int kVerifierSplitKTileRows = 16;

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
            add(reqs, PREFILL_EFFECTIVE_SEQ_LEN, sizeof(int));
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
            add(reqs, GROUP_EXPERT_MASK,
                static_cast<std::size_t>(kGroupedDescriptorTableSlots) *
                    static_cast<std::size_t>(num_experts) * sizeof(uint8_t));
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
            const std::size_t decode_slots = static_cast<std::size_t>(top_k);
            const std::size_t verifier_splitk_slots =
                static_cast<std::size_t>(std::min(max_seq_len, kVerifierSplitKTileRows)) *
                static_cast<std::size_t>(top_k);
            const std::size_t gateup_partial_slots =
                std::max(decode_slots, verifier_splitk_slots);

            add(reqs, DECODE_HIDDEN_INT8,
                static_cast<std::size_t>(max_seq_len) *
                    static_cast<std::size_t>(d_model) * sizeof(int8_t));
            add(reqs, DECODE_HIDDEN_SCALES,
                static_cast<std::size_t>(max_seq_len) *
                    static_cast<std::size_t>(d_model_blocks) * sizeof(float));
            add(reqs, GATEUP_GATE_PARTIALS,
                gateup_partial_slots * kMaxGateUpPartitions * static_cast<std::size_t>(intermediate) * sizeof(float));
            add(reqs, GATEUP_UP_PARTIALS,
                gateup_partial_slots * kMaxGateUpPartitions * static_cast<std::size_t>(intermediate) * sizeof(float));
            const std::size_t down_partial_slots =
                std::max<std::size_t>(1u, verifier_splitk_slots);
            add(reqs, DOWN_PARTIALS,
                down_partial_slots * kMaxDownPartitions * static_cast<std::size_t>(d_model) * sizeof(float));
            add(reqs, DECODE_SWIGLU_INT8,
                decode_slots * static_cast<std::size_t>(intermediate) * sizeof(int8_t));
            add(reqs, DECODE_SWIGLU_SCALES,
                decode_slots * static_cast<std::size_t>(intermediate_blocks) * sizeof(float));
            add(reqs, DECODE_EXPERT_IDS, decode_slots * sizeof(int));
            add(reqs, DECODE_WEIGHTS, decode_slots * sizeof(float));
            add(reqs, CUDA_ROUTING_DECODE_EXPERT_IDS, decode_slots * sizeof(int));
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
            d_model = std::max(1, d_model);
            num_experts = std::max(1, num_experts);
            const int d_model_blocks = ceilDiv(d_model, 32);
            const std::size_t table_descs =
                static_cast<std::size_t>(kGroupedDescriptorTableSlots) *
                static_cast<std::size_t>(num_experts) *
                sizeof(DeviceNativeVNNIMatrixDesc);
            add(reqs, CUDA_GROUPED_GATE_DESC_TABLES, table_descs);
            add(reqs, CUDA_GROUPED_UP_DESC_TABLES, table_descs);
            add(reqs, CUDA_GROUPED_DOWN_DESC_TABLES, table_descs);
            /*
             * Runtime-placement grouped graphs materialize the current
             * device-resident expert pointers before launching their GEMMs.
             * Those mutable descriptor values are graph-owner state: a main
             * verifier graph, an MTP sidecar, and a rebalance topology graph
             * may all remain live at once.  Give every retained immutable
             * descriptor identity a matching fixed-stride runtime slot so one
             * graph can never overwrite another graph's captured addresses.
             */
            const std::size_t runtime_prefill_descs = table_descs;
            add(reqs, CUDA_RUNTIME_PREFILL_GATE_DESC_TABLE, runtime_prefill_descs);
            add(reqs, CUDA_RUNTIME_PREFILL_UP_DESC_TABLE, runtime_prefill_descs);
            add(reqs, CUDA_RUNTIME_PREFILL_DOWN_DESC_TABLE, runtime_prefill_descs);
            add(reqs, CUDA_ROUTER_Q8_GATE_WEIGHTS,
                static_cast<std::size_t>(kRouterGateCacheSlots) *
                    static_cast<std::size_t>(num_experts) *
                    static_cast<std::size_t>(d_model) * sizeof(int8_t));
            add(reqs, CUDA_ROUTER_Q8_GATE_SCALES,
                static_cast<std::size_t>(kRouterGateCacheSlots) *
                    static_cast<std::size_t>(num_experts) *
                    static_cast<std::size_t>(d_model_blocks) * sizeof(float));
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
            add(reqs, ROCM_ROUTER_Q8_HIDDEN,
                static_cast<std::size_t>(max_seq_len) *
                    static_cast<std::size_t>(d_model) * sizeof(int8_t));
            add(reqs, ROCM_ROUTER_Q8_SCALES,
                static_cast<std::size_t>(max_seq_len) *
                    static_cast<std::size_t>(d_model_blocks) * sizeof(float));
            add(reqs, ROCM_ROUTER_Q8_GATE_WEIGHTS,
                static_cast<std::size_t>(kRouterGateCacheSlots) *
                    static_cast<std::size_t>(num_experts) *
                    static_cast<std::size_t>(d_model) * sizeof(int8_t));
            add(reqs, ROCM_ROUTER_Q8_GATE_SCALES,
                static_cast<std::size_t>(kRouterGateCacheSlots) *
                    static_cast<std::size_t>(num_experts) *
                    static_cast<std::size_t>(d_model_blocks) * sizeof(float));
            add(reqs, ROCM_ROUTER_FP16_GATE_WEIGHTS,
                static_cast<std::size_t>(kRouterGateCacheSlots) *
                    static_cast<std::size_t>(num_experts) *
                    static_cast<std::size_t>(d_model) * sizeof(uint16_t));
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
            const std::size_t table_descs =
                static_cast<std::size_t>(kGroupedDescriptorTableSlots) *
                static_cast<std::size_t>(num_experts) *
                sizeof(DeviceNativeVNNIMatrixDesc);
            add(reqs, ROCM_GROUPED_GATE_DESC_TABLES, table_descs);
            add(reqs, ROCM_GROUPED_UP_DESC_TABLES, table_descs);
            add(reqs, ROCM_GROUPED_DOWN_DESC_TABLES, table_descs);
            /*
             * Runtime-placement grouped graphs materialize mutable descriptor
             * values on device.  The destination is owned by the persistent
             * descriptor lease retained by that graph, not by the device-wide
             * MoE kernel singleton.  Match the immutable table arena's slot
             * count and stride so concurrently retained verifier/rebalance
             * graphs cannot alias one another.
             */
            const std::size_t runtime_prefill_descs = table_descs;
            add(reqs, ROCM_RUNTIME_PREFILL_GATE_DESC_TABLE, runtime_prefill_descs);
            add(reqs, ROCM_RUNTIME_PREFILL_UP_DESC_TABLE, runtime_prefill_descs);
            add(reqs, ROCM_RUNTIME_PREFILL_DOWN_DESC_TABLE, runtime_prefill_descs);
            return reqs;
        }
    } // namespace MoEWorkspaceBuffers
} // namespace llaminar2
