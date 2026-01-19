/**
 * @file ROCmRoPEKernelT.cpp
 * @brief ROCm RoPE kernel implementation
 *
 * Implementation of the ROCmRoPEKernelT template specializations.
 * Calls the extern "C" HIP wrapper functions defined in ROCmRoPEKernels.hip.
 *
 * @author Llaminar Team
 * @date 2025-01-17
 */

#include "ROCmRoPEKernelT.h"
#include "../../../tensors/Tensors.h"
#include <hip/hip_runtime.h>

// Forward declare extern "C" HIP wrappers
extern "C"
{
    bool hipOps_rope_fp32(
        float *Q,
        float *K,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        int device_idx);

    bool hipOps_rope_bf16(
        uint16_t *Q,
        uint16_t *K,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        int device_idx);

    bool hipOps_rope_fp16(
        uint16_t *Q,
        uint16_t *K,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        int device_idx);
}

namespace llaminar2
{
    namespace rocm
    {

        // =========================================================================
        // ROCmRoPEKernelT<FP32> Implementation
        // =========================================================================

        bool ROCmRoPEKernelT<ActivationPrecision::FP32>::apply(
            float *data, float *output,
            const int *pos_ids,
            int batch_size, int seq_len, int head_dim, int num_heads,
            float theta_base, bool interleaved,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            (void)output;
            (void)batch_size;
            (void)interleaved; // TODO: support interleaved layout
            (void)mpi_ctx;
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            return apply_typed(data, nullptr, pos_ids, seq_len, num_heads, num_heads, head_dim, theta_base, dev);
        }

        bool ROCmRoPEKernelT<ActivationPrecision::FP32>::apply_typed(
            float *Q,
            float *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            int device_idx)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            bool ok = hipOps_rope_fp32(Q, K, position_ids, seq_len, n_heads, n_kv_heads,
                                       head_dim, rope_theta, dev);
            if (ok)
                hipDeviceSynchronize();
            return ok;
        }

        // =========================================================================
        // ROCmRoPEKernelT<BF16> Implementation
        // =========================================================================

        bool ROCmRoPEKernelT<ActivationPrecision::BF16>::apply_bf16(
            uint16_t *data, uint16_t *output,
            const int *pos_ids,
            int batch_size, int seq_len, int head_dim, int num_heads,
            float theta_base, int device_idx)
        {
            (void)output;
            (void)batch_size;
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            return apply_typed(data, nullptr, pos_ids, seq_len, num_heads, num_heads, head_dim, theta_base, dev);
        }

        bool ROCmRoPEKernelT<ActivationPrecision::BF16>::apply_typed(
            uint16_t *Q,
            uint16_t *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            int device_idx)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            bool ok = hipOps_rope_bf16(Q, K, position_ids, seq_len, n_heads, n_kv_heads,
                                       head_dim, rope_theta, dev);
            if (ok)
                hipDeviceSynchronize();
            return ok;
        }

        // =========================================================================
        // ROCmRoPEKernelT<FP16> Implementation
        // =========================================================================

        bool ROCmRoPEKernelT<ActivationPrecision::FP16>::apply_fp16(
            uint16_t *data, uint16_t *output,
            const int *pos_ids,
            int batch_size, int seq_len, int head_dim, int num_heads,
            float theta_base, int device_idx)
        {
            (void)output;
            (void)batch_size;
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            return apply_typed(data, nullptr, pos_ids, seq_len, num_heads, num_heads, head_dim, theta_base, dev);
        }

        bool ROCmRoPEKernelT<ActivationPrecision::FP16>::apply_typed(
            uint16_t *Q,
            uint16_t *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            int device_idx)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            bool ok = hipOps_rope_fp16(Q, K, position_ids, seq_len, n_heads, n_kv_heads,
                                       head_dim, rope_theta, dev);
            if (ok)
                hipDeviceSynchronize();
            return ok;
        }

    } // namespace rocm
} // namespace llaminar2
