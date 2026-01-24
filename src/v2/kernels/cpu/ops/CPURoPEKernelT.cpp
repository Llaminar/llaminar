/**
 * @file CPURoPEKernelT.cpp
 * @brief Implementation of typed RoPE kernel with ActivationPrecision support
 * @author David Sanftenberg
 *
 * This file implements the CPURoPEKernelT template specializations for
 * FP32, BF16, FP16, Q8_1, and Q16_1 precision types.
 *
 * Q16_1 supports variable block sizes (32, 64, 128, 192) via templated dispatch.
 *
 * Implementation mirrors CPURoPEKernelT - uses existing primitives with n_past
 * and handles position_ids at the kernel level.
 */

#include "CPURoPEKernelT.h"
#include "../primitives/RoPEPrimitives.h"
#include "../../../tensors/BlockStructures.h"
#include "../../../tensors/Tensors.h" // For FP32Tensor, BF16Tensor, etc.
#include "../../../utils/Logger.h"
#include <cstring>

namespace llaminar2
{

    // Thread-local state for decode optimization (mirrors CPURoPEKernelT)
    static thread_local primitives::RoPEPersistentState tls_state_;

    // ============================================================================
    // FP32 Specialization Implementation
    // ============================================================================

    bool CPURoPEKernelT<ActivationPrecision::FP32>::apply_typed(
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
        (void)device_idx; // Unused for CPU kernel

        if (!Q)
        {
            LOG_ERROR("CPURoPEKernelT<FP32>: Null Q pointer");
            return false;
        }

        if (head_dim % 2 != 0)
        {
            LOG_ERROR("CPURoPEKernelT<FP32>: head_dim must be even");
            return false;
        }

        if (seq_len <= 0)
        {
            return true; // Nothing to do
        }

        const int q_stride = n_heads * head_dim;
        const int k_stride = n_kv_heads * head_dim;

        // Check for contiguous positions to enable optimized block processing
        bool contiguous = true;
        int start_pos = 0;

        if (position_ids)
        {
            start_pos = position_ids[0];
            if (seq_len > 1)
            {
                for (int i = 1; i < seq_len; ++i)
                {
                    if (position_ids[i] != start_pos + i)
                    {
                        contiguous = false;
                        break;
                    }
                }
            }
        }

        // If positions are contiguous, use optimized block processing
        if (contiguous && start_pos >= 0)
        {
            primitives::apply_rope_vectorized(
                Q, K,
                seq_len, head_dim,
                n_heads, n_kv_heads,
                start_pos, rope_theta,
                (seq_len == 1) ? &tls_state_ : nullptr);
            return true;
        }

        // Fall back to per-token processing for non-contiguous positions
        for (int tok = 0; tok < seq_len; ++tok)
        {
            int position = position_ids ? position_ids[tok] : tok;

            // Skip RoPE for padding tokens (position_id = -1)
            if (position < 0)
            {
                continue;
            }

            float *q_token = Q + tok * q_stride;
            float *k_token = K ? (K + tok * k_stride) : nullptr;

            // Apply RoPE to single token
            primitives::apply_rope_vectorized(
                q_token, k_token,
                1, head_dim,
                n_heads, n_kv_heads,
                position, rope_theta,
                &tls_state_);
        }

        return true;
    }

    // ============================================================================
    // BF16 Specialization Implementation
    // ============================================================================

    bool CPURoPEKernelT<ActivationPrecision::BF16>::apply_typed(
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
        (void)device_idx; // Unused for CPU kernel

        if (!Q)
        {
            LOG_ERROR("CPURoPEKernelT<BF16>: Null Q pointer");
            return false;
        }

        if (head_dim % 2 != 0)
        {
            LOG_ERROR("CPURoPEKernelT<BF16>: head_dim must be even");
            return false;
        }

        if (seq_len <= 0)
        {
            return true; // Nothing to do
        }

        const int q_stride = n_heads * head_dim;
        const int k_stride = n_kv_heads * head_dim;

        // Check for contiguous positions to enable optimized block processing
        bool contiguous = true;
        int start_pos = 0;

        if (position_ids)
        {
            start_pos = position_ids[0];
            if (seq_len > 1)
            {
                for (int i = 1; i < seq_len; ++i)
                {
                    if (position_ids[i] != start_pos + i)
                    {
                        contiguous = false;
                        break;
                    }
                }
            }
        }

        // If positions are contiguous, use optimized block processing
        if (contiguous && start_pos >= 0)
        {
            primitives::apply_rope_bf16(
                Q, K,
                seq_len, head_dim,
                n_heads, n_kv_heads,
                start_pos, rope_theta,
                (seq_len == 1) ? &tls_state_ : nullptr);
            return true;
        }

        // Fall back to per-token processing for non-contiguous positions
        for (int tok = 0; tok < seq_len; ++tok)
        {
            int position = position_ids ? position_ids[tok] : tok;

            // Skip RoPE for padding tokens (position_id = -1)
            if (position < 0)
            {
                continue;
            }

            uint16_t *q_token = Q + tok * q_stride;
            uint16_t *k_token = K ? (K + tok * k_stride) : nullptr;

            // Apply RoPE to single token
            primitives::apply_rope_bf16(
                q_token, k_token,
                1, head_dim,
                n_heads, n_kv_heads,
                position, rope_theta,
                &tls_state_);
        }

        return true;
    }

    // ============================================================================
    // FP16 Specialization Implementation
    // ============================================================================

    bool CPURoPEKernelT<ActivationPrecision::FP16>::apply_typed(
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
        (void)device_idx; // Unused for CPU kernel

        if (!Q)
        {
            LOG_ERROR("CPURoPEKernelT<FP16>: Null Q pointer");
            return false;
        }

        if (head_dim % 2 != 0)
        {
            LOG_ERROR("CPURoPEKernelT<FP16>: head_dim must be even");
            return false;
        }

        if (seq_len <= 0)
        {
            return true; // Nothing to do
        }

        const int q_stride = n_heads * head_dim;
        const int k_stride = n_kv_heads * head_dim;

        // Check for contiguous positions to enable optimized block processing
        bool contiguous = true;
        int start_pos = 0;

        if (position_ids)
        {
            start_pos = position_ids[0];
            if (seq_len > 1)
            {
                for (int i = 1; i < seq_len; ++i)
                {
                    if (position_ids[i] != start_pos + i)
                    {
                        contiguous = false;
                        break;
                    }
                }
            }
        }

        // If positions are contiguous, use optimized block processing
        if (contiguous && start_pos >= 0)
        {
            primitives::apply_rope_fp16(
                Q, K,
                seq_len, head_dim,
                n_heads, n_kv_heads,
                start_pos, rope_theta,
                (seq_len == 1) ? &tls_state_ : nullptr);
            return true;
        }

        // Fall back to per-token processing for non-contiguous positions
        for (int tok = 0; tok < seq_len; ++tok)
        {
            int position = position_ids ? position_ids[tok] : tok;

            // Skip RoPE for padding tokens (position_id = -1)
            if (position < 0)
            {
                continue;
            }

            uint16_t *q_token = Q + tok * q_stride;
            uint16_t *k_token = K ? (K + tok * k_stride) : nullptr;

            // Apply RoPE to single token
            primitives::apply_rope_fp16(
                q_token, k_token,
                1, head_dim,
                n_heads, n_kv_heads,
                position, rope_theta,
                &tls_state_);
        }

        return true;
    }

    // ============================================================================
    // Q8_1 Specialization Implementation (Pure-Integer)
    // ============================================================================

    bool CPURoPEKernelT<ActivationPrecision::Q8_1>::apply_typed(
        Q8_1Block *Q,
        Q8_1Block *K,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        int device_idx)
    {
        (void)device_idx; // Unused for CPU kernel

        if (!Q)
        {
            LOG_ERROR("CPURoPEKernelT<Q8_1>: Null Q pointer");
            return false;
        }

        // Validate head_dim divisibility by Q8_1 block size
        if (head_dim % 32 != 0)
        {
            LOG_ERROR("CPURoPEKernelT<Q8_1>: head_dim ("
                      << head_dim << ") must be divisible by 32 (Q8_1 block size)");
            return false;
        }

        // Call pure-integer Q8_1 RoPE primitive
        primitives::apply_rope_q8_1_integer(
            Q, K,
            position_ids,
            seq_len,
            n_heads, n_kv_heads,
            head_dim,
            rope_theta,
            (seq_len == 1) ? &tls_state_ : nullptr);

        return true;
    }

    // ============================================================================
    // ITensorRoPE Interface Implementations
    // ============================================================================

    // --- FP32 apply() ---
    bool CPURoPEKernelT<ActivationPrecision::FP32>::apply(
        float *data, float *output,
        const int *pos_ids,
        int batch_size, int seq_len, int head_dim, int num_heads,
        float theta_base, bool interleaved,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)output;      // In-place operation
        (void)interleaved; // Not used in this implementation
        (void)mpi_ctx;     // Not used for CPU kernel

        // The old apply() signature has batch_size but apply_typed uses seq_len
        // We treat batch_size * seq_len as the total sequence length
        int total_seq = batch_size * seq_len;
        return apply_typed(data, nullptr, pos_ids, total_seq, num_heads, num_heads, head_dim, theta_base, device_idx);
    }

    // --- FP32 apply_tensor() ---
    bool CPURoPEKernelT<ActivationPrecision::FP32>::apply_tensor(
        TensorBase *Q,
        TensorBase *K,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        const MPIContext *mpi_ctx,
        int device_idx,
        int pos_offset)
    {
        (void)mpi_ctx;
        (void)pos_offset; // CPU kernel doesn't need this optimization

        if (!Q || Q->native_type() != TensorType::FP32)
        {
            LOG_ERROR("CPURoPEKernelT<FP32>::apply_tensor: Q must be FP32Tensor");
            return false;
        }

        auto *q_fp32 = dynamic_cast<FP32Tensor *>(Q);
        FP32Tensor *k_fp32 = nullptr;
        if (K)
        {
            if (K->native_type() != TensorType::FP32)
            {
                LOG_ERROR("CPURoPEKernelT<FP32>::apply_tensor: K must be FP32Tensor");
                return false;
            }
            k_fp32 = dynamic_cast<FP32Tensor *>(K);
        }

        return apply_typed(
            q_fp32->mutable_data(),
            k_fp32 ? k_fp32->mutable_data() : nullptr,
            position_ids, seq_len, n_heads, n_kv_heads, head_dim, rope_theta, device_idx);
    }

    // --- BF16 apply_bf16() ---
    bool CPURoPEKernelT<ActivationPrecision::BF16>::apply_bf16(
        uint16_t *data, uint16_t *output,
        const int *pos_ids,
        int batch_size, int seq_len, int head_dim, int num_heads,
        float theta_base, int device_idx)
    {
        (void)output; // In-place operation
        int total_seq = batch_size * seq_len;
        return apply_typed(data, nullptr, pos_ids, total_seq, num_heads, num_heads, head_dim, theta_base, device_idx);
    }

    // --- BF16 apply_tensor() ---
    bool CPURoPEKernelT<ActivationPrecision::BF16>::apply_tensor(
        TensorBase *Q,
        TensorBase *K,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        const MPIContext *mpi_ctx,
        int device_idx,
        int pos_offset)
    {
        (void)mpi_ctx;
        (void)pos_offset; // CPU kernel doesn't need this optimization

        if (!Q || Q->native_type() != TensorType::BF16)
        {
            LOG_ERROR("CPURoPEKernelT<BF16>::apply_tensor: Q must be BF16Tensor");
            return false;
        }

        auto *q_bf16 = dynamic_cast<BF16Tensor *>(Q);
        BF16Tensor *k_bf16 = nullptr;
        if (K)
        {
            if (K->native_type() != TensorType::BF16)
            {
                LOG_ERROR("CPURoPEKernelT<BF16>::apply_tensor: K must be BF16Tensor");
                return false;
            }
            k_bf16 = dynamic_cast<BF16Tensor *>(K);
        }

        return apply_typed(
            q_bf16->mutable_typed_data(),
            k_bf16 ? k_bf16->mutable_typed_data() : nullptr,
            position_ids, seq_len, n_heads, n_kv_heads, head_dim, rope_theta, device_idx);
    }

    // --- FP16 apply_fp16() ---
    bool CPURoPEKernelT<ActivationPrecision::FP16>::apply_fp16(
        uint16_t *data, uint16_t *output,
        const int *pos_ids,
        int batch_size, int seq_len, int head_dim, int num_heads,
        float theta_base, int device_idx)
    {
        (void)output; // In-place operation
        int total_seq = batch_size * seq_len;
        return apply_typed(data, nullptr, pos_ids, total_seq, num_heads, num_heads, head_dim, theta_base, device_idx);
    }

    // --- FP16 apply_tensor() ---
    bool CPURoPEKernelT<ActivationPrecision::FP16>::apply_tensor(
        TensorBase *Q,
        TensorBase *K,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        const MPIContext *mpi_ctx,
        int device_idx,
        int pos_offset)
    {
        (void)mpi_ctx;
        (void)pos_offset; // CPU kernel doesn't need this optimization

        if (!Q || Q->native_type() != TensorType::FP16)
        {
            LOG_ERROR("CPURoPEKernelT<FP16>::apply_tensor: Q must be FP16Tensor");
            return false;
        }

        auto *q_fp16 = dynamic_cast<FP16Tensor *>(Q);
        FP16Tensor *k_fp16 = nullptr;
        if (K)
        {
            if (K->native_type() != TensorType::FP16)
            {
                LOG_ERROR("CPURoPEKernelT<FP16>::apply_tensor: K must be FP16Tensor");
                return false;
            }
            k_fp16 = dynamic_cast<FP16Tensor *>(K);
        }

        return apply_typed(
            q_fp16->mutable_typed_data(),
            k_fp16 ? k_fp16->mutable_typed_data() : nullptr,
            position_ids, seq_len, n_heads, n_kv_heads, head_dim, rope_theta, device_idx);
    }

    // --- Q8_1 apply_q8_1() ---
    bool CPURoPEKernelT<ActivationPrecision::Q8_1>::apply_q8_1(
        void *data, void *output,
        const int *pos_ids,
        int batch_size, int seq_len, int head_dim, int num_heads,
        float theta_base, int device_idx)
    {
        (void)output; // In-place operation
        int total_seq = batch_size * seq_len;
        return apply_typed(
            static_cast<Q8_1Block *>(data),
            nullptr,
            pos_ids, total_seq, num_heads, num_heads, head_dim, theta_base, device_idx);
    }

    // --- Q8_1 apply_tensor() ---
    bool CPURoPEKernelT<ActivationPrecision::Q8_1>::apply_tensor(
        TensorBase *Q,
        TensorBase *K,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        const MPIContext *mpi_ctx,
        int device_idx,
        int pos_offset)
    {
        (void)mpi_ctx;
        (void)pos_offset; // CPU kernel doesn't need this optimization

        if (!Q || Q->native_type() != TensorType::Q8_1)
        {
            LOG_ERROR("CPURoPEKernelT<Q8_1>::apply_tensor: Q must be Q8_1Tensor");
            return false;
        }

        auto *q_q8 = dynamic_cast<Q8_1Tensor *>(Q);
        Q8_1Tensor *k_q8 = nullptr;
        if (K)
        {
            if (K->native_type() != TensorType::Q8_1)
            {
                LOG_ERROR("CPURoPEKernelT<Q8_1>::apply_tensor: K must be Q8_1Tensor");
                return false;
            }
            k_q8 = dynamic_cast<Q8_1Tensor *>(K);
        }

        return apply_typed(
            q_q8->mutable_typed_data(),
            k_q8 ? k_q8->mutable_typed_data() : nullptr,
            position_ids, seq_len, n_heads, n_kv_heads, head_dim, rope_theta, device_idx);
    }

    // --- Q8_1 apply_q8_1_to_fp32() (Hybrid mode) ---
    bool CPURoPEKernelT<ActivationPrecision::Q8_1>::apply_q8_1_to_fp32(
        TensorBase *Q_in,
        TensorBase *K_in,
        TensorBase *Q_out,
        TensorBase *K_out,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)mpi_ctx;
        (void)device_idx;

        // Validate inputs
        if (!Q_in || Q_in->native_type() != TensorType::Q8_1)
        {
            LOG_ERROR("CPURoPEKernelT<Q8_1>::apply_q8_1_to_fp32: Q_in must be Q8_1Tensor");
            return false;
        }
        if (!Q_out || Q_out->native_type() != TensorType::FP32)
        {
            LOG_ERROR("CPURoPEKernelT<Q8_1>::apply_q8_1_to_fp32: Q_out must be FP32Tensor");
            return false;
        }

        auto *q_in_q8 = dynamic_cast<Q8_1Tensor *>(Q_in);
        auto *q_out_fp32 = dynamic_cast<FP32Tensor *>(Q_out);

        const Q8_1Block *k_in_blocks = nullptr;
        float *k_out_data = nullptr;

        if (K_in && K_out)
        {
            if (K_in->native_type() != TensorType::Q8_1)
            {
                LOG_ERROR("CPURoPEKernelT<Q8_1>::apply_q8_1_to_fp32: K_in must be Q8_1Tensor");
                return false;
            }
            if (K_out->native_type() != TensorType::FP32)
            {
                LOG_ERROR("CPURoPEKernelT<Q8_1>::apply_q8_1_to_fp32: K_out must be FP32Tensor");
                return false;
            }
            auto *k_in_q8 = dynamic_cast<Q8_1Tensor *>(K_in);
            auto *k_out_fp32 = dynamic_cast<FP32Tensor *>(K_out);
            k_in_blocks = k_in_q8->typed_data();
            k_out_data = k_out_fp32->mutable_typed_data();
        }

        // Call the primitive
        primitives::apply_rope_q8_1_to_fp32(
            q_in_q8->typed_data(),
            k_in_blocks,
            q_out_fp32->mutable_typed_data(),
            k_out_data,
            position_ids,
            seq_len,
            n_heads,
            n_kv_heads,
            head_dim,
            rope_theta);

        return true;
    }

    // --- Q8_1 apply_q8_1_to_q16_1() (HybridQ16 mode) ---
    bool CPURoPEKernelT<ActivationPrecision::Q8_1>::apply_q8_1_to_q16_1(
        TensorBase *Q_in,
        TensorBase *K_in,
        TensorBase *Q_out,
        TensorBase *K_out,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)mpi_ctx;
        (void)device_idx;

        // Validate Q inputs
        if (!Q_in || Q_in->native_type() != TensorType::Q8_1)
        {
            LOG_ERROR("CPURoPEKernelT<Q8_1>::apply_q8_1_to_q16_1: Q_in must be Q8_1Tensor");
            return false;
        }
        if (!Q_out || Q_out->native_type() != TensorType::Q16_1)
        {
            LOG_ERROR("CPURoPEKernelT<Q8_1>::apply_q8_1_to_q16_1: Q_out must be Q16_1Tensor");
            return false;
        }

        auto *q_in_q8 = dynamic_cast<Q8_1Tensor *>(Q_in);
        auto *q_out_q16 = dynamic_cast<Q16_1Tensor *>(Q_out);

        const Q8_1Block *k_in_blocks = nullptr;
        Q16_1Block *k_out_blocks = nullptr;

        if (K_in && K_out)
        {
            if (K_in->native_type() != TensorType::Q8_1)
            {
                LOG_ERROR("CPURoPEKernelT<Q8_1>::apply_q8_1_to_q16_1: K_in must be Q8_1Tensor");
                return false;
            }
            if (K_out->native_type() != TensorType::Q16_1)
            {
                LOG_ERROR("CPURoPEKernelT<Q8_1>::apply_q8_1_to_q16_1: K_out must be Q16_1Tensor");
                return false;
            }
            auto *k_in_q8 = dynamic_cast<Q8_1Tensor *>(K_in);
            auto *k_out_q16 = dynamic_cast<Q16_1Tensor *>(K_out);
            k_in_blocks = k_in_q8->typed_data();
            k_out_blocks = k_out_q16->mutable_typed_data();
        }

        // Call the primitive
        primitives::apply_rope_q8_1_to_q16_1(
            q_in_q8->typed_data(),
            k_in_blocks,
            q_out_q16->mutable_typed_data(),
            k_out_blocks,
            position_ids,
            seq_len,
            n_heads,
            n_kv_heads,
            head_dim,
            rope_theta);

        return true;
    }

    // --- Q8_1 apply_q8_1_to_q16() (HybridQ16 mode with head scales) ---
    // Uses pure integer arithmetic (Q8 fixed-point ratios, Q15 sin/cos)
    bool CPURoPEKernelT<ActivationPrecision::Q8_1>::apply_q8_1_to_q16(
        TensorBase *Q_in,
        TensorBase *K_in,
        TensorBase *Q_out,
        TensorBase *K_out,
        float *Q_head_scales,
        float *K_head_scales,
        Q16BlockSize block_size,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)mpi_ctx;
        (void)device_idx;

        // Validate Q inputs
        if (!Q_in || Q_in->native_type() != TensorType::Q8_1)
        {
            LOG_ERROR("CPURoPEKernelT<Q8_1>::apply_q8_1_to_q16: Q_in must be Q8_1Tensor");
            return false;
        }
        if (!Q_out || Q_out->native_type() != TensorType::Q16_1)
        {
            LOG_ERROR("CPURoPEKernelT<Q8_1>::apply_q8_1_to_q16: Q_out must be Q16_1Tensor");
            return false;
        }

        auto *q_in_q8 = dynamic_cast<Q8_1Tensor *>(Q_in);
        auto *q_out_q16 = dynamic_cast<Q16_1Tensor *>(Q_out);

        const Q8_1Block *k_in_blocks = nullptr;
        void *k_out_ptr = nullptr;

        if (K_in && K_out)
        {
            if (K_in->native_type() != TensorType::Q8_1)
            {
                LOG_ERROR("CPURoPEKernelT<Q8_1>::apply_q8_1_to_q16: K_in must be Q8_1Tensor");
                return false;
            }
            if (K_out->native_type() != TensorType::Q16_1)
            {
                LOG_ERROR("CPURoPEKernelT<Q8_1>::apply_q8_1_to_q16: K_out must be Q16_1Tensor");
                return false;
            }
            auto *k_in_q8 = dynamic_cast<Q8_1Tensor *>(K_in);
            auto *k_out_q16 = dynamic_cast<Q16_1Tensor *>(K_out);
            k_in_blocks = k_in_q8->typed_data();
            k_out_ptr = k_out_q16->mutable_data();
        }

        // Call the dispatch primitive with runtime block size selection
        primitives::apply_rope_q8_1_to_q16_dispatch(
            q_in_q8->typed_data(),
            k_in_blocks,
            q_out_q16->mutable_data(),
            k_out_ptr,
            Q_head_scales,
            K_head_scales,
            block_size,
            position_ids,
            seq_len,
            n_heads,
            n_kv_heads,
            head_dim,
            rope_theta);

        return true;
    }

    // --- Q8_1 apply_q8_1_to_q16_fixed_scale() (HybridQ16 mode with FIXED scale) ---
    // All output blocks have d = kv_cache_scale / 32767 for integer attention
    bool CPURoPEKernelT<ActivationPrecision::Q8_1>::apply_q8_1_to_q16_fixed_scale(
        TensorBase *Q_in,
        TensorBase *K_in,
        TensorBase *Q_out,
        TensorBase *K_out,
        Q16BlockSize block_size,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        float kv_cache_scale,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)mpi_ctx;
        (void)device_idx;

        // Validate Q inputs
        if (!Q_in || Q_in->native_type() != TensorType::Q8_1)
        {
            LOG_ERROR("CPURoPEKernelT<Q8_1>::apply_q8_1_to_q16_fixed_scale: Q_in must be Q8_1Tensor");
            return false;
        }
        if (!Q_out || Q_out->native_type() != TensorType::Q16_1)
        {
            LOG_ERROR("CPURoPEKernelT<Q8_1>::apply_q8_1_to_q16_fixed_scale: Q_out must be Q16_1Tensor");
            return false;
        }

        auto *q_in_q8 = dynamic_cast<Q8_1Tensor *>(Q_in);
        auto *q_out_q16 = dynamic_cast<Q16_1Tensor *>(Q_out);

        const Q8_1Block *k_in_blocks = nullptr;
        void *k_out_ptr = nullptr;

        if (K_in && K_out)
        {
            if (K_in->native_type() != TensorType::Q8_1)
            {
                LOG_ERROR("CPURoPEKernelT<Q8_1>::apply_q8_1_to_q16_fixed_scale: K_in must be Q8_1Tensor");
                return false;
            }
            if (K_out->native_type() != TensorType::Q16_1)
            {
                LOG_ERROR("CPURoPEKernelT<Q8_1>::apply_q8_1_to_q16_fixed_scale: K_out must be Q16_1Tensor");
                return false;
            }
            auto *k_in_q8 = dynamic_cast<Q8_1Tensor *>(K_in);
            auto *k_out_q16 = dynamic_cast<Q16_1Tensor *>(K_out);
            k_in_blocks = k_in_q8->typed_data();
            k_out_ptr = k_out_q16->mutable_typed_data();
        }

        // Call the fixed-scale dispatch primitive
        primitives::apply_rope_q8_1_to_q16_fixed_scale_dispatch(
            q_in_q8->typed_data(),
            k_in_blocks,
            q_out_q16->mutable_typed_data(),
            k_out_ptr,
            block_size,
            position_ids,
            seq_len,
            n_heads,
            n_kv_heads,
            head_dim,
            rope_theta,
            kv_cache_scale);

        return true;
    }

    // --- Q8_1 apply_mixed_q8_k16_to_q16() (HybridQ16 K precision fix mode) ---
    // Q input: Q8_1 (from GEMM) → Q16_1 output with FIXED kv_cache_scale
    // K input: Q16_1 (from GEMM with higher precision) → Q16_1 output with DYNAMIC per-head scale
    //
    // This method handles the K precision fix case where GEMM outputs K as Q16_1
    // (instead of Q8_1) to preserve small values. Q is still processed as Q8_1→Q16_1
    // with fixed scale, but K uses Q16_1→Q16_1 dynamic scale path.
    bool CPURoPEKernelT<ActivationPrecision::Q8_1>::apply_mixed_q8_k16_to_q16(
        TensorBase *Q_in,     // Q8_1 input
        TensorBase *K_in,     // Q16_1 input (from GEMM K precision fix!)
        TensorBase *Q_out,    // Q16_1 output (fixed scale)
        TensorBase *K_out,    // Q16_1 output (dynamic scale)
        float *K_head_scales, // Output: per-head K scales [seq_len * n_kv_heads]
        Q16BlockSize block_size,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        float kv_cache_scale,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)mpi_ctx;
        (void)device_idx;

        // Validate Q inputs (Q8_1 → Q16_1 fixed scale path)
        if (!Q_in || Q_in->native_type() != TensorType::Q8_1)
        {
            LOG_ERROR("CPURoPEKernelT<Q8_1>::apply_mixed_q8_k16_to_q16: Q_in must be Q8_1Tensor");
            return false;
        }
        if (!Q_out || Q_out->native_type() != TensorType::Q16_1)
        {
            LOG_ERROR("CPURoPEKernelT<Q8_1>::apply_mixed_q8_k16_to_q16: Q_out must be Q16_1Tensor");
            return false;
        }

        // Validate K inputs (Q16_1 → Q16_1 dynamic scale path - K precision fix!)
        if (!K_in || K_in->native_type() != TensorType::Q16_1)
        {
            LOG_ERROR("CPURoPEKernelT<Q8_1>::apply_mixed_q8_k16_to_q16: K_in must be Q16_1Tensor (K precision fix mode)");
            return false;
        }
        if (!K_out || K_out->native_type() != TensorType::Q16_1)
        {
            LOG_ERROR("CPURoPEKernelT<Q8_1>::apply_mixed_q8_k16_to_q16: K_out must be Q16_1Tensor");
            return false;
        }
        if (!K_head_scales)
        {
            LOG_ERROR("CPURoPEKernelT<Q8_1>::apply_mixed_q8_k16_to_q16: K_head_scales must not be nullptr");
            return false;
        }

        auto *q_in_q8 = dynamic_cast<Q8_1Tensor *>(Q_in);
        auto *q_out_q16 = dynamic_cast<Q16_1Tensor *>(Q_out);
        auto *k_in_q16 = dynamic_cast<Q16_1Tensor *>(K_in);
        auto *k_out_q16 = dynamic_cast<Q16_1Tensor *>(K_out);

        // Step 1: Process Q with Q8_1 → Q16_1 fixed scale (same as standard HybridQ16)
        // K_in=nullptr and K_out=nullptr since we handle K separately
        primitives::apply_rope_q8_1_to_q16_fixed_scale_dispatch(
            q_in_q8->typed_data(),
            nullptr, // K_in - handled separately
            q_out_q16->mutable_typed_data(),
            nullptr, // K_out - handled separately
            block_size,
            position_ids,
            seq_len,
            n_heads,
            n_kv_heads, // Doesn't matter for K since we pass nullptr
            head_dim,
            rope_theta,
            kv_cache_scale);

        // Step 2: Process K with Q16_1 → Q16_1 dynamic scale (K precision fix path)
        // This preserves the higher precision from GEMM and outputs per-head scales
        primitives::apply_rope_q16_to_q16_dynamic_scale_dispatch(
            k_in_q16->typed_data(),
            k_out_q16->mutable_typed_data(),
            block_size,
            position_ids,
            seq_len,
            n_kv_heads,
            head_dim,
            rope_theta,
            K_head_scales); // Per-head scales [seq_len * n_kv_heads]

        LOG_DEBUG("[CPURoPEKernelT<Q8_1>::apply_mixed_q8_k16_to_q16] K precision fix applied: "
                  << "Q: Q8_1→Q16_1 fixed scale=" << kv_cache_scale
                  << ", K: Q16_1→Q16_1 dynamic scale (head_scales[0]=" << K_head_scales[0] << ")");

        return true;
    }

    // ============================================================================
    // Q16_1 Specialization Implementation (High-Precision Integer)
    // ============================================================================

    // 32-element block apply_typed (backward compatibility)
    bool CPURoPEKernelT<ActivationPrecision::Q16_1>::apply_typed(
        Q16_1Block *Q,
        Q16_1Block *K,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        int device_idx)
    {
        (void)device_idx; // Unused for CPU kernel

        if (!Q)
        {
            LOG_ERROR("CPURoPEKernelT<Q16_1>: Null Q pointer");
            return false;
        }

        // Validate head_dim divisibility by Q16_1 block size
        if (head_dim % 32 != 0)
        {
            LOG_ERROR("CPURoPEKernelT<Q16_1>: head_dim ("
                      << head_dim << ") must be divisible by 32 (Q16_1 block size)");
            return false;
        }

        // Call Q16_1 RoPE primitive (32-element blocks)
        primitives::apply_rope_q16_integer<Q16_1Block>(
            Q, K,
            position_ids,
            seq_len,
            n_heads, n_kv_heads,
            head_dim,
            rope_theta,
            (seq_len == 1) ? &tls_state_ : nullptr);

        return true;
    }

    // Templated apply for variable block sizes
    template <typename BlockType>
    bool CPURoPEKernelT<ActivationPrecision::Q16_1>::apply_typed_block(
        BlockType *Q,
        BlockType *K,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        int device_idx)
    {
        (void)device_idx; // Unused for CPU kernel

        if (!Q)
        {
            LOG_ERROR("CPURoPEKernelT<Q16_1>: Null Q pointer");
            return false;
        }

        constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;

        // Validate head_dim divisibility by block size
        if (head_dim % BLOCK_SIZE != 0)
        {
            LOG_ERROR("CPURoPEKernelT<Q16_1>: head_dim ("
                      << head_dim << ") must be divisible by " << BLOCK_SIZE);
            return false;
        }

        // Call templated Q16 RoPE primitive
        primitives::apply_rope_q16_integer<BlockType>(
            Q, K,
            position_ids,
            seq_len,
            n_heads, n_kv_heads,
            head_dim,
            rope_theta,
            (seq_len == 1) ? &tls_state_ : nullptr);

        return true;
    }

    // Explicit template instantiations for all block sizes
    template bool CPURoPEKernelT<ActivationPrecision::Q16_1>::apply_typed_block<Q16_1Block>(
        Q16_1Block *, Q16_1Block *, const int *, int, int, int, int, float, int);
    template bool CPURoPEKernelT<ActivationPrecision::Q16_1>::apply_typed_block<Q16_1Block_64>(
        Q16_1Block_64 *, Q16_1Block_64 *, const int *, int, int, int, int, float, int);
    template bool CPURoPEKernelT<ActivationPrecision::Q16_1>::apply_typed_block<Q16_1Block_128>(
        Q16_1Block_128 *, Q16_1Block_128 *, const int *, int, int, int, int, float, int);

    // --- Q16_1 apply_q16_1() ---
    bool CPURoPEKernelT<ActivationPrecision::Q16_1>::apply_q16_1(
        void *Q_data, void *K_data,
        const int *pos_ids,
        int seq_len, int n_heads, int n_kv_heads, int head_dim,
        float theta_base, int device_idx)
    {
        return apply_typed(
            static_cast<Q16_1Block *>(Q_data),
            static_cast<Q16_1Block *>(K_data),
            pos_ids, seq_len, n_heads, n_kv_heads, head_dim, theta_base, device_idx);
    }

    // --- Q16_1 apply_tensor() ---
    bool CPURoPEKernelT<ActivationPrecision::Q16_1>::apply_tensor(
        TensorBase *Q,
        TensorBase *K,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        const MPIContext *mpi_ctx,
        int device_idx,
        int pos_offset)
    {
        (void)mpi_ctx;
        (void)pos_offset; // CPU kernel doesn't need this optimization

        if (!Q || Q->native_type() != TensorType::Q16_1)
        {
            LOG_ERROR("CPURoPEKernelT<Q16_1>::apply_tensor: Q must be Q16_1Tensor");
            return false;
        }

        auto *q_q16 = dynamic_cast<Q16_1Tensor *>(Q);
        Q16_1Tensor *k_q16 = nullptr;
        if (K)
        {
            if (K->native_type() != TensorType::Q16_1)
            {
                LOG_ERROR("CPURoPEKernelT<Q16_1>::apply_tensor: K must be Q16_1Tensor");
                return false;
            }
            k_q16 = dynamic_cast<Q16_1Tensor *>(K);

            // Ensure both tensors have the same block size
            if (k_q16->q16_block_size() != q_q16->q16_block_size())
            {
                LOG_ERROR("CPURoPEKernelT<Q16_1>::apply_tensor: Q and K must have same block size");
                return false;
            }
        }

        // Dispatch based on block size
        // Note: raw_mutable_data() returns void*, we cast to the correct block type
        void *q_raw = q_q16->raw_mutable_data();
        void *k_raw = k_q16 ? k_q16->raw_mutable_data() : nullptr;

        switch (q_q16->q16_block_size())
        {
        case Q16BlockSize::BLOCK_32:
            return apply_typed_block<Q16_1Block>(
                static_cast<Q16_1Block *>(q_raw),
                static_cast<Q16_1Block *>(k_raw),
                position_ids, seq_len, n_heads, n_kv_heads, head_dim, rope_theta, device_idx);

        case Q16BlockSize::BLOCK_64:
            return apply_typed_block<Q16_1Block_64>(
                static_cast<Q16_1Block_64 *>(q_raw),
                static_cast<Q16_1Block_64 *>(k_raw),
                position_ids, seq_len, n_heads, n_kv_heads, head_dim, rope_theta, device_idx);

        case Q16BlockSize::BLOCK_128:
            return apply_typed_block<Q16_1Block_128>(
                static_cast<Q16_1Block_128 *>(q_raw),
                static_cast<Q16_1Block_128 *>(k_raw),
                position_ids, seq_len, n_heads, n_kv_heads, head_dim, rope_theta, device_idx);

        default:
            LOG_ERROR("CPURoPEKernelT<Q16_1>::apply_tensor: Unknown block size");
            return false;
        }
    }

    // ============================================================================
    // Destructor Definitions (required for vtable emission)
    // ============================================================================

    CPURoPEKernelT<ActivationPrecision::FP32>::~CPURoPEKernelT() = default;
    CPURoPEKernelT<ActivationPrecision::BF16>::~CPURoPEKernelT() = default;
    CPURoPEKernelT<ActivationPrecision::FP16>::~CPURoPEKernelT() = default;
    CPURoPEKernelT<ActivationPrecision::Q8_1>::~CPURoPEKernelT() = default;
    // Note: Q16_1 destructor is defaulted in-class

    // ============================================================================
    // Explicit Template Instantiations
    // ============================================================================

    template class CPURoPEKernelT<ActivationPrecision::FP32>;
    template class CPURoPEKernelT<ActivationPrecision::BF16>;
    template class CPURoPEKernelT<ActivationPrecision::FP16>;
    template class CPURoPEKernelT<ActivationPrecision::Q8_1>;
    template class CPURoPEKernelT<ActivationPrecision::Q16_1>;

} // namespace llaminar2
