#pragma once

/**
 * @file CUDAMoEKernel.h
 * @brief CUDA implementation of device-resident MoE routing and dispatch primitives.
 *
 * Provides the CUDA backend for `IMoEKernel`, keeping MoE routing results,
 * expert gather/scatter buffers, shared expert gates, and fallback SwiGLU
 * activations on-device. GEMM remains handled by the existing tensor-aware
 * CUDA GEMM engines; this class owns only the non-GEMM MoE glue kernels and
 * persistent scratch needed by those kernels.
 *
 * Lifecycle: Instances are created and cached by `KernelFactory` per CUDA
 * device. Scratch allocations are retained across calls and released when the
 * cached kernel is destroyed.
 */

#include "../../IMoEKernel.h"
#include "../CUDAKernelBase.h"
#include "../gemm/CUDADeviceWorkspace.h"
#include "../../../tensors/TensorType.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace llaminar2
{
    /**
     * @brief CUDA backend for MoE router, grouping, gather/scatter, and elementwise glue.
     *
     * The compute stages remain backend-neutral and call this through
     * `IMoEKernel`. All CUDA runtime interaction is isolated here and in the
     * companion `.cu` bridge file.
     */
    class CUDAMoEKernel final : public IMoEKernel, public CUDAKernelBase
    {
    public:
        /// @brief Construct a CUDA MoE kernel bound to one CUDA ordinal.
        explicit CUDAMoEKernel(int device_ordinal);

        /// @brief Release persistent CUDA scratch buffers.
        ~CUDAMoEKernel() override;

        /// @brief Clear request-shaped host grouping metadata while retaining device scratch.
        void resetDynamicState() override;

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

        bool groupTokensByExpertDevice(
            const int *d_routing_indices,
            const float *d_routing_weights,
            int seq_len, int num_experts, int top_k,
            int *d_expert_offsets,
            int *d_expert_counts,
            int *d_grouped_token_indices,
            float *d_grouped_weights) override;

        bool prepareExpertGroups(
            ITensor *routing_indices, ITensor *routing_weights,
            int seq_len, int num_experts, int top_k) override;

        /// @brief Upload persistent down-projection descriptors for grouped CUDA prefill.
        int uploadGroupedExpertDownDescriptorTable(
            const DeviceNativeVNNIMatrixDesc *down_descs,
            int num_experts,
            int d_model,
            int intermediate) override;

        /// @brief Upload persistent gate/up descriptor tables for grouped CUDA prefill.
        int uploadGroupedExpertGateUpDescriptorTables(
            const DeviceNativeVNNIMatrixDesc *gate_descs,
            const DeviceNativeVNNIMatrixDesc *up_descs,
            int num_experts,
            int d_model,
            int intermediate) override;

        /// @brief Prepare device-only expert grouping metadata for graph-captured prefill.
        bool prepareExpertGroupsAsync(
            ITensor *routing_indices, ITensor *routing_weights,
            int seq_len, int num_experts, int top_k) override;

        /// @brief Prepare grouped prefill metadata for the always-active shared expert.
        bool prepareSharedExpertPrefillGroup(int seq_len) override;

        /// @brief Execute fixed-topology grouped MoE prefill without host synchronization.
        bool executeGroupedPrefillPipeline(
            ITensor *hidden, ITensor *output,
            int gateup_desc_table_id,
            int down_desc_table_id,
            int seq_len, int d_model, int intermediate,
            int num_experts, int top_k) override;

        /// @brief Execute grouped gate/up decode from a persistent descriptor table and static host ids.
        bool groupedExpertGateUpDecodeFromTable(
            const TensorBase *input,
            const int *expert_ids,
            int descriptor_table_id,
            int num_active,
            ITensor *const *gate_outputs,
            ITensor *const *up_outputs,
            int d_model,
            int intermediate) override;

        /// @brief Execute grouped SwiGLU/down decode from a persistent descriptor table and static host ids/weights.
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

        /// @brief Execute graph-capturable grouped gate/up decode from runtime-table top-k ids.
        bool groupedExpertGateUpDecodeFromRuntime(
            DeviceMoELayerRuntime *runtime_layer,
            const TensorBase *input,
            int table_id,
            int top_k,
            ITensor *const *gate_outputs,
            ITensor *const *up_outputs,
            int d_model,
            int intermediate) override;

        /// @brief Execute graph-capturable grouped SwiGLU/down decode from runtime-table ids and weights.
        bool groupedExpertDownDecodeFromRuntime(
            ITensor *const *gate_tensors,
            ITensor *const *up_tensors,
            DeviceMoELayerRuntime *runtime_layer,
            int table_id,
            int top_k,
            ITensor *output,
            int d_model,
            int intermediate) override;

        /// @brief Execute fused graph-capturable runtime-routed gate/up, SwiGLU, and down decode.
        bool groupedExpertDecodeFromRuntime(
            DeviceMoELayerRuntime *runtime_layer,
            const TensorBase *input,
            int gateup_table_id,
            int down_table_id,
            int top_k,
            ITensor *output,
            int d_model,
            int intermediate) override;

        int getExpertTokenCount(int expert_id) const override;

        void gatherExpertBatch(
            ITensor *hidden, ITensor *batch_buffer,
            int expert_id, int d_model) override;

        void scatterExpertResults(
            ITensor *output, ITensor *expert_results,
            int expert_id, int d_model) override;

        bool supports_device(int device_idx) const override
        {
            return device_idx == device_ordinal_ || device_idx >= 0;
        }

        KernelSnapshotInfo getKernelSnapshotInfo() const override
        {
            return KernelSnapshotInfo::passthrough();
        }

        /// @brief Forward stage stream binding into the CUDA base class.
        void setGPUStream(void *stream) override { CUDAKernelBase::setGPUStream(stream); }

        /// @brief Bind graph-owned workspace scratch used by routing/grouped MoE kernels.
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;

        void unbindWorkspace() override { bindWorkspace(nullptr); }

    private:
        static constexpr std::size_t kRuntimePointerArrayMaxTopK = 16;

        struct DeviceRouteBuffers
        {
            float *d_logits = nullptr;
            int *d_indices = nullptr;
            float *d_weights = nullptr;
            size_t logits_count = 0;
            size_t topk_count = 0;
        };

        DeviceId deviceId() const { return DeviceId::cuda(device_ordinal_); }

        bool routeCore(const float *hidden, const void *gate_weights, TensorType gate_type,
                       int seq_len, int d_model, int num_experts, int top_k,
                       bool normalize_weights, DeviceRouteBuffers &buffers);
        bool routeLogitsCuBLAS(const float *hidden, const float *gate_weights, float *logits,
                               int seq_len, int d_model, int num_experts);

        bool ensureStagingCapacity(int count);
        bool ensureRouteBufferCapacity(size_t logits_count, size_t topk_count);
        bool ensureGroupingBufferCapacity(int total_slots, int num_experts);
        bool ensureGroupedPrefillScratchCapacity(int total_slots, int d_model, int intermediate);
        bool ensureGroupedGateUpDecodeCapacity(int top_k, int d_model);
        bool ensureGroupedGateUpKPartScratchCapacity(int top_k, int k_partitions, int intermediate);
        bool ensureGroupedDownKPartScratchCapacity(int k_partitions, int d_model, int slots = 1);
        bool ensureGroupedDownDecodeCapacity(int top_k, int intermediate);
        bool ensureGroupedDecodeMetadata(
            const int *expert_ids,
            const float *expert_weights,
            int num_active,
            bool include_weights);
        bool groupTokensByExpertDeviceMapped(
            const int *d_routing_indices,
            const float *d_routing_weights,
            int seq_len, int num_experts, int top_k,
            int *d_expert_offsets,
            int *d_expert_counts,
            int *d_grouped_token_indices,
            int *d_original_to_grouped,
            int *d_original_expert_ids,
            float *d_grouped_weights);
        bool ensureRuntimeGateUpPointerArrays(
            int table_id,
            int top_k,
            const std::array<float *, kRuntimePointerArrayMaxTopK> &gate_ptrs,
            const std::array<float *, kRuntimePointerArrayMaxTopK> &up_ptrs,
            float ***d_gate_ptrs,
            float ***d_up_ptrs);
        bool ensureRuntimeDownPointerArrays(
            int table_id,
            int top_k,
            const std::array<const float *, kRuntimePointerArrayMaxTopK> &gate_ptrs,
            const std::array<const float *, kRuntimePointerArrayMaxTopK> &up_ptrs,
            const float ***d_gate_ptrs,
            const float ***d_up_ptrs);
        bool bindWorkspaceBuffer(void **ptr, const char *name, size_t bytes, const char *context);
        void clearWorkspaceScratchBindings() noexcept;
        void releaseDeviceBuffers() noexcept;

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

        struct RuntimeGateUpPointerCacheEntry
        {
            int table_id = -1;
            int top_k = 0;
            std::array<std::uintptr_t, kRuntimePointerArrayMaxTopK> gate_ptr_values = {};
            std::array<std::uintptr_t, kRuntimePointerArrayMaxTopK> up_ptr_values = {};
            float **d_gate_ptrs = nullptr;
            float **d_up_ptrs = nullptr;
        };

        struct RuntimeDownPointerCacheEntry
        {
            int table_id = -1;
            int top_k = 0;
            std::array<std::uintptr_t, kRuntimePointerArrayMaxTopK> gate_ptr_values = {};
            std::array<std::uintptr_t, kRuntimePointerArrayMaxTopK> up_ptr_values = {};
            const float **d_gate_ptrs = nullptr;
            const float **d_up_ptrs = nullptr;
        };

        int device_ordinal_ = 0;
        void *router_cublas_handle_ = nullptr;

        int *d_staging_indices_ = nullptr;
        float *d_staging_weights_ = nullptr;
        int staging_capacity_ = 0;

        float *d_route_logits_ = nullptr;
        int *d_route_indices_ = nullptr;
        float *d_route_weights_ = nullptr;
        size_t route_logits_capacity_ = 0;
        size_t route_topk_capacity_ = 0;

        int *d_group_int_indices_ = nullptr;
        int *d_group_offsets_ = nullptr;
        int *d_group_counts_ = nullptr;
        int *d_group_token_indices_ = nullptr;
        int *d_group_original_to_grouped_ = nullptr;
        int *d_group_original_expert_ids_ = nullptr;
        int *d_group_write_heads_ = nullptr;
        float *d_group_weights_ = nullptr;
        int *d_group_active_expert_ids_ = nullptr;
        int group_active_expert_slots_ = 0;
        int group_slots_cap_ = 0;
        int group_experts_cap_ = 0;

        std::vector<GroupedDownDescriptorTable> grouped_down_desc_tables_;
        std::vector<GroupedGateUpDescriptorTable> grouped_gateup_desc_tables_;
        std::vector<RuntimeGateUpPointerCacheEntry> runtime_gateup_pointer_cache_;
        std::vector<RuntimeDownPointerCacheEntry> runtime_down_pointer_cache_;

        int8_t *d_prefill_A_int8_ = nullptr;
        float *d_prefill_A_scales_ = nullptr;
        int8_t *d_prefill_swiglu_int8_ = nullptr;
        float *d_prefill_swiglu_scales_ = nullptr;
        float *d_prefill_gate_ = nullptr;
        float *d_prefill_up_ = nullptr;
        int prefill_slots_cap_ = 0;
        int prefill_d_model_cap_ = 0;
        int prefill_intermediate_cap_ = 0;

        int8_t *d_decode_hidden_int8_ = nullptr;
        float *d_decode_hidden_scales_ = nullptr;
        int decode_gateup_topk_cap_ = 0;
        int decode_gateup_d_model_cap_ = 0;

        // Split-K partials scratch for the grouped gate/up decode projection.
        // Layout: [top_k][k_partitions][intermediate] per buffer.
        float *d_grouped_gateup_gate_partials_ = nullptr;
        float *d_grouped_gateup_up_partials_ = nullptr;
        int grouped_gateup_kpart_active_cap_ = 0;
        int grouped_gateup_kpart_partitions_cap_ = 0;
        int grouped_gateup_kpart_intermediate_cap_ = 0;

        int8_t *d_decode_swiglu_int8_ = nullptr;
        float *d_decode_swiglu_scales_ = nullptr;
        int decode_down_topk_cap_ = 0;
        int decode_down_intermediate_cap_ = 0;

        int *d_grouped_decode_expert_ids_ = nullptr;
        float *d_grouped_decode_weights_ = nullptr;
        int grouped_decode_metadata_cap_ = 0;
        std::vector<int> grouped_decode_cached_expert_ids_;
        std::vector<float> grouped_decode_cached_weights_;

        // Split-K partials scratch for the grouped SwiGLU down projection.
        // Decode layout: [1][k_partitions][d_model].
        // Verifier prefill layout: [total_slots][k_partitions][d_model].
        float *d_grouped_down_partials_ = nullptr;
        int grouped_down_kpart_partitions_cap_ = 0;
        int grouped_down_kpart_d_model_cap_ = 0;
        int grouped_down_kpart_slots_cap_ = 0;

        bool scratch_workspace_bound_ = false;
    };

} // namespace llaminar2
