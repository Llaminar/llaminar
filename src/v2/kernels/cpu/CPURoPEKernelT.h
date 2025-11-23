/**
 * @file CPURoPEKernelT.h
 * @brief CPU implementation of rotary position embeddings (uses vectorized primitives)
 *
 * Optimized RoPE implementation with:
 * - AVX2/AVX512 vectorization (8-16× speedup)
 * - Cached inverse frequencies
 * - Persistent decode state (complex recurrence)
 * - OpenMP parallelization for prefill
 * - Angle recurrence across tokens
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../../tensors/TensorKernels.h"
#include "primitives/RoPEPrimitives.h"
#include "CPUKernelBase.h"
#include <vector>
#include <cmath>

namespace llaminar2
{
    // Forward declarations
    class FP32Tensor;
    class BF16Tensor;
    class FP16Tensor;
    class Q8_1Tensor;

    /**
     * @brief CPU implementation of RoPE kernel
     *
     * Uses vectorized primitives from V1.
     */
    template <typename TensorT>
    class CPURoPEKernelT : public ITensorRoPE, public CPUKernelBase
    {
    public:
        CPURoPEKernelT() = default;
        ~CPURoPEKernelT() override = default;

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1; // CPU only
        }

        bool apply(
            float *Q, float *K,
            const int *position_ids,
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            float rope_theta = 10000.0f,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        bool apply_bf16(
            uint16_t *Q_bf16, uint16_t *K_bf16,
            const int *position_ids,
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            float rope_theta = 10000.0f,
            int device_idx = -1) override;

        bool apply_fp16(
            uint16_t *Q_fp16, uint16_t *K_fp16,
            const int *position_ids,
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            float rope_theta = 10000.0f,
            int device_idx = -1) override;

        bool apply_q8_1(
            void *Q_q8_1, void *K_q8_1,
            const int *position_ids,
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            float rope_theta = 10000.0f,
            int device_idx = -1) override;

    private:
        // Thread-local state for decode (uses primitives type)
        thread_local static primitives::RoPEPersistentState tls_state_;

        // Inverse frequency cache (delegates to primitives)
        static const std::vector<float> &get_inv_freq_cached(int head_dim, float freq_base);

        // Core rotation implementation (delegates to primitives)
        static void apply_rotation(
            float *q, float *k,
            int seq_len, int head_dim,
            int q_heads, int k_heads,
            int n_past, float freq_base);

        // Kernel contract (operator fusion framework)
        KernelContract get_contract() const override
        {
            return KernelContract{
                .accepted_input_formats = {TensorFormat::FP32, TensorFormat::BF16, TensorFormat::FP16},
                .output_format = TensorFormat::FP32,
                .supports_inplace = true,
                .is_fusable = true};
        }
    };

    // Backward compatibility alias
    using CPURoPEKernel = CPURoPEKernelT<FP32Tensor>;

} // namespace llaminar2
