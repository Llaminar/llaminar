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
#include "../../../tensors/BlockStructures.h"

#include <vector>

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

        /**
         * @brief Return the router-owned Q8_1 rows for the next grouped expert pass.
         *
         * CPU NativeVNNI expert projections consume Q8_1 activations.  Routing
         * and expert execution therefore share one canonical transform instead
         * of quantizing the same hidden row again for every routed expert.  A
         * publication is valid only for the exact FP32 source address, requested
         * row count, and model width that produced it.  Callers must treat a null
         * result as a broken production ordering contract; grouped CPU verifier
         * execution has no requantization fallback.
         *
         * @param source FP32 hidden-row base passed to route().
         * @param rows Number of contiguous verifier rows the consumer needs.
         * @param d_model Number of FP32 values in each row.
         * @return Contiguous `[rows, ceil(d_model / 32)]` Q8_1 blocks, or nullptr
         *         when no matching router publication exists.
         */
        const Q8_1Block *publishedRouterQ8Hidden(
            const float *source,
            int rows,
            int d_model) const noexcept;

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

    private:
        /** Revoke publication metadata before beginning a new routing call. */
        void invalidateRouterQ8HiddenPublication() noexcept;

        /**
         * @brief Quantize and publish runtime-M router inputs for NativeVNNI experts.
         *
         * The implementation deliberately uses the same Q8_1 block primitives
         * as serial NativeVNNI decode.  Keeping this transform in the router
         * makes grouped expert execution both economical and byte-equivalent.
         */
        bool publishRouterQ8Hidden(
            const float *source,
            int rows,
            int d_model);

        std::vector<Q8_1Block> router_q8_hidden_; ///< Canonical contiguous verifier rows.
        const float *router_q8_hidden_source_ = nullptr; ///< FP32 producer identity.
        int router_q8_hidden_rows_ = 0; ///< Number of published rows.
        int router_q8_hidden_d_model_ = 0; ///< Published row width.
        bool router_q8_hidden_valid_ = false; ///< Metadata and payload are current.
    };

} // namespace llaminar2
