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

#include <cstdint>
#include <memory>

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
        void syncBlasStream();
        void allocateHistogramBuffers(int num_layers, int num_experts);
        void ensureStagingCapacity(int count);

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
        bool routeCore(const float *hidden, const float *gate_weights,
                       int seq_len, int d_model, int num_experts, int top_k,
                       bool normalize_weights, DeviceRouteBuffers &bufs);

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

        // Phase 4: GPU-side expert grouping state (for prepareExpertGroups)
        int *d_group_int_indices_ = nullptr;   ///< float→int converted routing indices
        int *d_group_offsets_ = nullptr;       ///< [num_experts] exclusive prefix sums
        int *d_group_counts_ = nullptr;        ///< [num_experts] per-expert token counts
        int *d_group_token_indices_ = nullptr; ///< [total_slots] grouped token indices
        float *d_group_weights_ = nullptr;     ///< [total_slots] grouped routing weights
        int group_slots_cap_ = 0;              ///< capacity for total_slots buffers
        int group_experts_cap_ = 0;            ///< capacity for num_experts buffers
    };

} // namespace llaminar2
