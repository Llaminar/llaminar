/**
 * @file ROCmMoEKernel.h
 * @brief ROCm/HIP implementation of MoE kernel operations
 *
 * Implements IMoEKernel for ROCm GPU execution using HIP kernels.
 * Follows the three-file pattern: .h (class), .cpp (bridge), .hip (kernels).
 *
 * Operations:
 * - Router: gate logits (GEMV) → softmax → top-k selection
 * - Token gather: parallel row copy to expert batch buffer
 * - Scatter-add: weighted accumulation of expert outputs
 * - Shared expert gate: sigmoid dot + elementwise scale
 * - SwiGLU: silu(gate) * up activation
 *
 * All operations are dispatched on the kernel's bound HIP stream.
 */

#pragma once

#include "../../IMoEKernel.h"
#include "../ROCmKernelBase.h"
#include "../../../tensors/TensorType.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace llaminar2
{
    namespace rocm
    {
        class HipBLASGemmKernel;
    }

    /**
     * @brief ROCm GPU implementation of MoE kernel operations
     *
     * Uses HIP kernels for device-native MoE operations. All methods
     * expect device pointers and dispatch work on the bound GPU stream.
     *
     * Constructor requires a device ordinal. The kernel obtains its
     * HIP stream from GPUDeviceContextPool.
     */
    class ROCmMoEKernel : public IMoEKernel, public ROCmKernelBase
    {
    public:
        /**
         * @brief Construct a ROCm MoE kernel for the given device
         * @param device_ordinal ROCm GPU ordinal (0-based)
         */
        explicit ROCmMoEKernel(int device_ordinal);
        ~ROCmMoEKernel() override;

        void bindWorkspace(DeviceWorkspaceManager *workspace) override;
        void unbindWorkspace() override { bindWorkspace(nullptr); }

        /**
         * @brief Reset request-shaped host metadata while preserving device buffers.
         *
         * Session resets clear histogram counts and CPU-side grouping metadata.
         * Device scratch, runtime pointer arrays, descriptor tables, and router
         * weight conversion caches stay resident because captured prefill graphs
         * can reference those stable device addresses across requests.
         */
        void resetDynamicState() override;

        // =================================================================
        // IMoEKernel interface
        // =================================================================

        bool route(
            const float *hidden,
            const float *gate_weights,
            int seq_len, int d_model,
            int num_experts, int top_k,
            bool normalize_weights,
            MoERoutingResult &result) override;

        void gatherTokenBatch(
            const float *hidden,
            float *batch_buffer,
            const int *token_indices,
            int num_tokens, int d_model) override;

        void scatterAddWeighted(
            float *output,
            const float *expert_output,
            const int *token_indices,
            const float *weights,
            int num_tokens, int d_model) override;

        void sharedExpertGate(
            const float *input,
            const float *gate_inp,
            float *shared_output,
            int seq_len, int d_model) override;

        void swiGLU(float *gate, const float *up, int count) override;

        void weightedAdd(float *output, const float *input,
                         float weight, int count) override;

        // =================================================================
        // Tensor-aware API overrides (GPU implementations)
        // =================================================================

        bool routeWithTensors(
            ITensor *hidden, ITensor *gate_weights,
            int seq_len, int d_model, int num_experts, int top_k,
            bool normalize_weights,
            ITensor *output_indices, ITensor *output_weights,
            MoERoutingResult &host_result) override;

        bool decodeRouteSelect(
            DeviceMoELayerRuntime *runtime_layer,
            ITensor *hidden, ITensor *gate_weights,
            int d_model, int num_experts, int top_k,
            bool normalize_weights,
            ITensor *output_indices, ITensor *output_weights,
            bool write_legacy_outputs,
            bool update_runtime_histogram) override;

        void zeroBuffer(ITensor *tensor, size_t bytes) override;

        void gatherTokenBatchFromTensors(
            ITensor *hidden, ITensor *batch_buffer,
            const int *host_token_indices, int num_tokens, int d_model) override;

        void scatterAddWeightedFromTensors(
            ITensor *output, ITensor *expert_output,
            const int *host_token_indices, const float *host_weights,
            int num_tokens, int d_model) override;

        void sharedExpertGateFromTensors(
            ITensor *input, ITensor *gate_inp, ITensor *shared_output,
            int seq_len, int d_model) override;

        void swiGLUFromTensors(ITensor *gate, ITensor *up, int count) override;

        void weightedAddFromTensors(
            ITensor *output, ITensor *input, float weight, int count) override;

        bool groupedExpertDownDecode(
            ITensor *const *gate_tensors,
            ITensor *const *up_tensors,
            const int *expert_ids,
            const float *expert_weights,
            const DeviceNativeVNNIMatrixDesc *down_descs,
            int num_active,
            ITensor *output,
            int d_model,
            int intermediate) override;

        int uploadGroupedExpertDownDescriptorTable(
            const DeviceNativeVNNIMatrixDesc *down_descs,
            int num_experts,
            int d_model,
            int intermediate) override;

        int uploadGroupedExpertGateUpDescriptorTables(
            const DeviceNativeVNNIMatrixDesc *gate_descs,
            const DeviceNativeVNNIMatrixDesc *up_descs,
            int num_experts,
            int d_model,
            int intermediate) override;

        bool groupedExpertGateUpDecodeFromTable(
            const TensorBase *input,
            const int *expert_ids,
            int descriptor_table_id,
            int num_active,
            ITensor *const *gate_outputs,
            ITensor *const *up_outputs,
            int d_model,
            int intermediate) override;

        bool groupedExpertGateUpDecodeFromRouting(
            const TensorBase *input,
            ITensor *routing_indices,
            int descriptor_table_id,
            int top_k,
            ITensor *const *gate_outputs,
            ITensor *const *up_outputs,
            int d_model,
            int intermediate) override;

        bool groupedExpertGateUpDecodeFromRuntime(
            DeviceMoELayerRuntime *runtime_layer,
            const TensorBase *input,
            int descriptor_table_id,
            int top_k,
            ITensor *const *gate_outputs,
            ITensor *const *up_outputs,
            int d_model,
            int intermediate) override;

        bool groupedExpertDownDecodeFromTable(
            ITensor *const *gate_tensors,
            ITensor *const *up_tensors,
            const int *expert_ids,
            const float *expert_weights,
            int descriptor_table_id,
            int num_active,
            ITensor *output,
            int d_model,
            int intermediate) override;

        bool groupedExpertDownDecodeFromRouting(
            ITensor *const *gate_tensors,
            ITensor *const *up_tensors,
            ITensor *routing_indices,
            ITensor *routing_weights,
            int descriptor_table_id,
            int top_k,
            ITensor *output,
            int d_model,
            int intermediate) override;

        bool groupedExpertDownDecodeFromRuntime(
            ITensor *const *gate_tensors,
            ITensor *const *up_tensors,
            DeviceMoELayerRuntime *runtime_layer,
            int descriptor_table_id,
            int top_k,
            ITensor *output,
            int d_model,
            int intermediate) override;

        // =================================================================
        // Phase 2: Device-resident histogram + expert mask
        // =================================================================

        void recordHistogramDevice(
            const int *d_routing_indices, int seq_len, int top_k, int layer_idx) override;

        void syncHistogramToHost(
            uint64_t *host_counts, int layer_idx, int num_experts) override;

        void resetHistogramDevice(int layer_idx, int num_experts) override;

        void updateExpertMaskDevice(const bool *mask, int num_experts) override;

        void applyExpertMaskDevice(
            float *d_routing_weights, const int *d_routing_indices,
            int seq_len, int top_k) override;

        // =================================================================
        // Phase 3: Device-side token grouping (prefill optimization)
        // =================================================================

        bool groupTokensByExpertDevice(
            const int *d_routing_indices,
            const float *d_routing_weights,
            int seq_len, int num_experts, int top_k,
            int *d_expert_offsets,
            int *d_expert_counts,
            int *d_grouped_token_indices,
            float *d_grouped_weights) override;

        bool groupPrefillRoutes(
            DeviceMoELayerRuntime *runtime_layer,
            ITensor *routing_indices, ITensor *routing_weights,
            int current_tokens, int max_tokens,
            int num_experts, int top_k) override;

        bool gatherPrefillExpertBatchFromRuntime(
            DeviceMoELayerRuntime *runtime_layer,
            ITensor *hidden, ITensor *batch_buffer,
            int expert_id, int max_tokens, int d_model) override;

        bool scatterPrefillExpertResultsFromRuntime(
            ITensor *output, ITensor *expert_results,
            DeviceMoELayerRuntime *runtime_layer,
            int expert_id, int max_tokens, int d_model) override;

        // =================================================================
        // Phase 4: GPU-side expert dispatch for prefill
        // =================================================================

        bool prepareExpertGroups(
            ITensor *routing_indices, ITensor *routing_weights,
            int seq_len, int num_experts, int top_k) override;

        int getExpertTokenCount(int expert_id) const override;

        void gatherExpertBatch(
            ITensor *hidden, ITensor *batch_buffer,
            int expert_id, int d_model) override;

        void scatterExpertResults(
            ITensor *output, ITensor *expert_results,
            int expert_id, int d_model) override;

        // =================================================================
        // Phase 5: Fully-grouped MoE prefill pipeline (graph-capturable)
        // =================================================================

        bool prepareExpertGroupsAsync(
            ITensor *routing_indices, ITensor *routing_weights,
            int seq_len, int num_experts, int top_k) override;

        bool prepareSharedExpertPrefillGroup(int seq_len) override;

        bool executeGroupedPrefillPipeline(
            ITensor *hidden, ITensor *output,
            int gateup_desc_table_id,
            int down_desc_table_id,
            int seq_len, int d_model, int intermediate,
            int num_experts, int top_k) override;

        bool hasGroupedPrefillScratchCapacity(int total_slots, int d_model, int intermediate) const
        {
            return total_slots <= prefill_slots_cap_ &&
                   d_model <= prefill_d_model_cap_ &&
                   intermediate <= prefill_intermediate_cap_;
        }

        bool hasGroupingBufferCapacity(int total_slots, int num_experts) const
        {
            return total_slots <= group_slots_cap_ &&
                   num_experts <= group_experts_cap_ &&
                   d_write_heads_ != nullptr &&
                   num_experts <= max_write_heads_experts_;
        }

        // =================================================================
        // ITensorKernel interface
        // =================================================================

        bool supports_device(int device_idx) const override
        {
            return device_idx >= 0; // GPU only
        }

        KernelSnapshotInfo getKernelSnapshotInfo() const override
        {
            return KernelSnapshotInfo::passthrough();
        }

    protected:
        /// Propagate stream changes to the child hipBLAS kernel
        void setGPUStream(void *stream) override
        {
            ROCmKernelBase::setGPUStream(stream);
            syncBlasStream();
        }

    private:
        static constexpr std::size_t kRuntimePointerArrayMaxTopK = 16;

        void syncBlasStream();
        void allocateHistogramBuffers(int num_layers, int num_experts);
        bool ensureStagingCapacity(int count);
        bool ensureGroupedDecodeCapacity(int num_active, int intermediate);
        bool ensureGroupedGateUpCapacity(int num_active, int d_model);
        bool ensureGroupedGateUpKPartScratchCapacity(int num_active, int k_partitions, int intermediate);
        bool stageRuntimeGateUpPointerArrays(
            int descriptor_table_id,
            int top_k,
            const std::array<float *, kRuntimePointerArrayMaxTopK> &gate_ptrs,
            const std::array<float *, kRuntimePointerArrayMaxTopK> &up_ptrs,
            float ***d_gate_ptrs,
            float ***d_up_ptrs);
        bool stageRuntimeDownPointerArrays(
            int descriptor_table_id,
            int top_k,
            const std::array<const float *, kRuntimePointerArrayMaxTopK> &gate_ptrs,
            const std::array<const float *, kRuntimePointerArrayMaxTopK> &up_ptrs,
            const float ***d_gate_ptrs,
            const float ***d_up_ptrs);
        bool ensureSharedGateScratchCapacity(int seq_len);
        bool ensureRouteBufferCapacity(size_t logits_count, size_t topk_count);
        bool ensureRouteLogitsPartialsCapacity(size_t partial_count);
        bool ensureRouterQ8HiddenScratchCapacity(int d_model);
        bool bindWorkspaceBuffer(void **ptr, const char *name, size_t bytes, const char *context);
        void clearWorkspaceScratchBindings() noexcept;
        struct RouterQ8GateCacheEntry;
        const RouterQ8GateCacheEntry *getOrCreateQ8RouterGateCache(
            ITensor *gate_weights,
            const float *gate_device_ptr,
            int d_model,
            int num_experts);
        const void *getOrCreateFP16RouterGateCache(
            ITensor *gate_weights,
            const float *gate_device_ptr,
            int d_model,
            int num_experts);

        /// Core GPU routing: gate logits GEMM + softmax + top-k.
        /// Returns device buffers (caller must D2H and hipFree).
        struct DeviceRouteBuffers
        {
            float *d_logits = nullptr;  ///< [seq_len * num_experts]
            int *d_indices = nullptr;   ///< [seq_len * top_k]
            float *d_weights = nullptr; ///< [seq_len * top_k]
            size_t logits_count = 0;
            size_t topk_count = 0;
        };
        bool routeCore(const float *hidden, const void *gate_weights, TensorType gate_type,
                       int seq_len, int d_model, int num_experts, int top_k,
                       bool normalize_weights, DeviceRouteBuffers &bufs);

        struct GroupedDownDescriptorTable
        {
            DeviceNativeVNNIMatrixDesc *device_descs = nullptr;
            std::vector<DeviceNativeVNNIMatrixDesc> host_descs;
            int num_experts = 0;
            int d_model = 0;
            int intermediate = 0;
            uint8_t codebook_id = 0;
            bool valid = false;
        };

        struct GroupedGateUpDescriptorTable
        {
            DeviceNativeVNNIMatrixDesc *device_gate_descs = nullptr;
            DeviceNativeVNNIMatrixDesc *device_up_descs = nullptr;
            std::vector<DeviceNativeVNNIMatrixDesc> host_gate_descs;
            std::vector<DeviceNativeVNNIMatrixDesc> host_up_descs;
            int num_experts = 0;
            int d_model = 0;
            int intermediate = 0;
            uint8_t codebook_id = 0;
            bool valid = false;
        };

        struct RouterFP16GateCacheEntry
        {
            std::uintptr_t source_tensor = 0;
            std::uintptr_t source_device_ptr = 0;
            int d_model = 0;
            int num_experts = 0;
            size_t element_count = 0;
            void *d_gate_weights_fp16 = nullptr;
        };

        struct RouterQ8GateCacheEntry
        {
            std::uintptr_t source_tensor = 0;
            std::uintptr_t source_device_ptr = 0;
            int d_model = 0;
            int num_experts = 0;
            int blocks_per_row = 0;
            size_t element_count = 0;
            size_t scale_count = 0;
            int8_t *d_gate_weights_q8 = nullptr;
            float *d_gate_scales = nullptr;
        };

        int device_ordinal_;
        std::unique_ptr<rocm::HipBLASGemmKernel> blas_gemm_;

        // Phase 2: device-resident histogram and expert mask
        uint64_t *d_histogram_ = nullptr; ///< [max_layers_ * max_experts_] on device
        bool *d_expert_mask_ = nullptr;   ///< [max_experts_] on device
        int max_experts_ = 0;
        int max_layers_ = 0;

        // Phase 3: write_heads scratch buffer for token grouping
        int *d_write_heads_ = nullptr; ///< [max_write_heads_experts_] on device
        int max_write_heads_experts_ = 0;

        // Staging buffers for tensor-aware gather/scatter (H2D of small host arrays)
        int *d_staging_indices_ = nullptr;   ///< [staging_capacity_] ints on device
        float *d_staging_weights_ = nullptr; ///< [staging_capacity_] floats on device
        int staging_capacity_ = 0;

        // Grouped decode staging for ROCm native-VNNI MoE down path.
        const float **d_grouped_gate_ptrs_ = nullptr;
        const float **d_grouped_up_ptrs_ = nullptr;
        int *d_grouped_expert_ids_ = nullptr;
        float *d_grouped_decode_weights_ = nullptr;
        DeviceNativeVNNIMatrixDesc *d_grouped_down_descs_ = nullptr;
        int8_t *d_grouped_swiglu_int8_ = nullptr;
        float *d_grouped_swiglu_scales_ = nullptr;
        int grouped_decode_active_cap_ = 0;
        int grouped_decode_intermediate_cap_ = 0;
        std::vector<GroupedDownDescriptorTable> grouped_down_desc_tables_;

        // Grouped decode staging for ROCm native-VNNI MoE gate/up path.
        float **d_grouped_gate_output_ptrs_ = nullptr;
        float **d_grouped_up_output_ptrs_ = nullptr;
        int *d_grouped_gateup_expert_ids_ = nullptr;
        int8_t *d_grouped_hidden_int8_ = nullptr;
        float *d_grouped_hidden_scales_ = nullptr;
        float *d_grouped_gateup_gate_partials_ = nullptr;
        float *d_grouped_gateup_up_partials_ = nullptr;
        int grouped_gateup_active_cap_ = 0;
        int grouped_gateup_d_model_cap_ = 0;
        int grouped_gateup_kpart_active_cap_ = 0;
        int grouped_gateup_kpart_partitions_cap_ = 0;
        int grouped_gateup_kpart_intermediate_cap_ = 0;
        std::vector<GroupedGateUpDescriptorTable> grouped_gateup_desc_tables_;
        std::vector<float *> host_grouped_gate_output_ptrs_;
        std::vector<float *> host_grouped_up_output_ptrs_;
        std::vector<int> host_grouped_gateup_expert_ids_;

        // Reusable scratch for sharedExpertGate() gate values.
        float *d_shared_gate_scratch_ = nullptr; ///< [shared_gate_scratch_capacity_] floats on device
        int shared_gate_scratch_capacity_ = 0;

        // Reusable routing buffers for routeCore().
        float *d_route_logits_ = nullptr;            ///< [route_logits_capacity_] floats on device
        int *d_route_indices_ = nullptr;             ///< [route_topk_capacity_] ints on device
        float *d_route_weights_ = nullptr;           ///< [route_topk_capacity_] floats on device
        float *d_route_logits_partials_ = nullptr;   ///< [route_logits_partials_capacity_] floats on device
        int8_t *d_router_q8_hidden_ = nullptr;       ///< [router_q8_hidden_d_model_cap_] int8 values on device
        float *d_router_q8_hidden_scales_ = nullptr; ///< [router_q8_hidden_blocks_cap_] floats on device
        size_t route_logits_capacity_ = 0;
        size_t route_topk_capacity_ = 0;
        size_t route_logits_partials_capacity_ = 0;
        int router_q8_hidden_d_model_cap_ = 0;
        int router_q8_hidden_blocks_cap_ = 0;
        std::vector<RouterFP16GateCacheEntry> router_fp16_gate_cache_;
        std::vector<RouterQ8GateCacheEntry> router_q8_gate_cache_;

        // Phase 4: GPU-side expert grouping state (for prepareExpertGroups)
        int *d_group_int_indices_ = nullptr;   ///< float→int converted routing indices
        int *d_group_offsets_ = nullptr;       ///< [num_experts] exclusive prefix sums
        int *d_group_counts_ = nullptr;        ///< [num_experts] per-expert token counts
        int *d_group_max_tokens_ = nullptr;    ///< [1] device-side max(d_group_counts_) (async reduction)
        int *d_group_token_indices_ = nullptr; ///< [total_slots] grouped token indices
        float *d_group_weights_ = nullptr;     ///< [total_slots] grouped routing weights
        int *d_group_active_expert_ids_ = nullptr; ///< [min(total_slots,num_experts)] compact active expert ids
        int group_active_expert_slots_ = 0;    ///< Fixed active-expert launch slots from the last small grouping pass
        int group_slots_cap_ = 0;              ///< capacity for total_slots buffers
        int group_experts_cap_ = 0;            ///< capacity for num_experts buffers

        // Phase 5: Grouped prefill pipeline scratch buffers.
        // The kernels execute sequentially on one stream, so A/scales are reused
        // for SwiGLU quantization and gate is reused for down-projection output.
        int8_t *d_prefill_A_int8_ = nullptr;  ///< [prefill_slots_cap_, max(d_model,intermediate)]
        float *d_prefill_A_scales_ = nullptr; ///< [prefill_slots_cap_, max_blocks_per_row]
        float *d_prefill_gate_ = nullptr;     ///< [prefill_slots_cap_, max(d_model,intermediate)]
        float *d_prefill_up_ = nullptr;       ///< [prefill_slots_cap_, intermediate]
        int prefill_slots_cap_ = 0;           ///< Current capacity (total_slots)
        int prefill_d_model_cap_ = 0;         ///< Current d_model capacity
        int prefill_intermediate_cap_ = 0;    ///< Current intermediate capacity

        bool scratch_workspace_bound_ = false;
        uint64_t bound_workspace_id_ = 0;

        bool ensureGroupedPrefillScratchCapacity(int total_slots, int d_model, int intermediate);
    };

} // namespace llaminar2
