/**
 * @file CPUMoEKernel.h
 * @brief CPU implementation of MoE kernel operations
 *
 * Implements IMoEKernel for CPU execution using ISA-dispatched vector
 * primitives (AVX2/AVX-512). This is the reference implementation;
 * GPU implementations (CUDA/ROCm) can override for device-native execution.
 */

#pragma once

#include "../../IMoEKernel.h"
#include "../CPUKernelBase.h"

namespace llaminar2
{

    /**
     * @brief CPU implementation of MoE kernel operations
     *
     * Uses ISA-dispatched vector primitives for:
     * - Router: vec_dot for gate logits, scalar softmax + partial_sort top-k
     * - Gather: memcpy-based token collection
     * - Scatter: vec_axpy for weighted accumulation
     * - Shared expert gate: vec_dot + sigmoid + vec_scale
     * - SwiGLU: ISA-dispatched silu(gate) * up
     */
    class CPUMoEKernel : public IMoEKernel, public CPUKernelBase
    {
    public:
        CPUMoEKernel() = default;
        ~CPUMoEKernel() override = default;

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

        // =================================================================
        // ITensorKernel interface
        // =================================================================

        bool supports_device(int device_idx) const override
        {
            return device_idx < 0; // CPU only (device_idx == -1)
        }

        KernelSnapshotInfo getKernelSnapshotInfo() const override
        {
            return KernelSnapshotInfo::passthrough();
        }
    };

} // namespace llaminar2
