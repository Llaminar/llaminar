/**
 * @file GQAAttention.cpp
 * @brief Grouped Query Attention (GQA) implementation
 * @author David Sanftenberg
 */

#include "GQAAttention.h"
#include "../AttentionUtils.h"
#include "../../utils/Logger.h"
#include "../../utils/DebugAssert.h"
#include "../../utils/MPIStager.h"
#include "../../tensors/TensorFactory.h"
#include "../../tensors/Tensors.h"
#include "../../kernels/cpu/primitives/SoftmaxPrimitives.h"
#include <cstring>
#include <cmath>
#include <sstream>
#include <omp.h>
#include <atomic>

namespace llaminar2
{
    namespace
    {
        bool should_build_mask(const GQAAttentionConfig &config,
                               int batch_size,
                               const std::vector<int> *sequence_lengths)
        {
            const bool has_lengths = sequence_lengths && !sequence_lengths->empty();
            // Only build mask if we actually need one:
            // - Causal masking required, OR
            // - Sliding window enabled, OR
            // - Padding mask needed (sequence_lengths provided)
            // NOTE: batch_size > 1 alone does NOT require a mask (equal-length batches with no padding)
            return config.causal || config.window_size > 0 || has_lengths;
        }

        bool build_sequence_mask(TensorBase *mask_tensor,
                                 int batch_size,
                                 int seq_len,
                                 const std::vector<int> *sequence_lengths,
                                 const GQAAttentionConfig &config)
        {
            if (!mask_tensor)
            {
                LOG_ERROR("[GQAAttention] mask tensor not provided");
                return false;
            }

            float *mask_data = mask_tensor->mutable_data();
            if (!mask_data)
            {
                LOG_ERROR("[GQAAttention] mask tensor has no storage");
                return false;
            }

            if (batch_size <= 1)
            {
                attention_utils::create_causal_mask(mask_data, seq_len, config.window_size);
                return true;
            }

            // For batched attention, choose appropriate mask builder:
            // - If sequence_lengths provided: Use combined mask (padding + optional causal)
            // - Otherwise:
            //   - If causal: Use batch causal mask
            //   - If non-causal: Use padding-only mask
            if (sequence_lengths && !sequence_lengths->empty())
            {
                attention_utils::create_combined_batch_mask(mask_data,
                                                            batch_size,
                                                            seq_len,
                                                            sequence_lengths->data(),
                                                            config.causal,
                                                            config.window_size);
            }
            else
            {
                // No sequence_lengths provided - choose mask type based on causal flag
                const int *seq_ptr = sequence_lengths ? sequence_lengths->data() : nullptr;
                if (config.causal)
                {
                    // Causal attention: mask future tokens
                    attention_utils::create_batch_causal_mask(mask_data,
                                                              batch_size,
                                                              seq_len,
                                                              seq_ptr,
                                                              config.window_size);
                }
                else
                {
                    // Non-causal attention: bi-directional, only mask padding
                    attention_utils::create_batch_padding_mask(mask_data,
                                                               batch_size,
                                                               seq_len,
                                                               seq_ptr,
                                                               config.window_size);
                }
            }
            return true;
        }

        bool build_combined_batch_mask(TensorBase *mask_tensor,
                                       int batch_size,
                                       int seq_len,
                                       const std::vector<int> &actual_lengths,
                                       const GQAAttentionConfig &config)
        {
            if (!mask_tensor)
            {
                LOG_ERROR("[GQAAttention] mask tensor not provided");
                return false;
            }

            float *mask_data = mask_tensor->mutable_data();
            if (!mask_data)
            {
                LOG_ERROR("[GQAAttention] mask tensor has no storage");
                return false;
            }

            attention_utils::create_combined_batch_mask(mask_data,
                                                        batch_size,
                                                        seq_len,
                                                        actual_lengths.data(),
                                                        config.causal,
                                                        config.window_size);
            return true;
        }
    } // namespace

    bool GQAAttention::compute(
        TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
        const GQAAttentionConfig &config,
        int batch_size,
        const std::vector<int> *sequence_lengths)
    {
        if (!validate_inputs(Q, K, V, output, config))
        {
            return false;
        }

        const auto &q_shape = Q->shape();
        if (q_shape.empty())
        {
            LOG_ERROR("[GQAAttention] compute: Q tensor shape is empty");
            return false;
        }

        const int total_tokens = static_cast<int>(q_shape[0]);
        const int effective_batch_size = (batch_size > 0) ? batch_size : 1;

        if (total_tokens % effective_batch_size != 0)
        {
            LOG_ERROR("[GQAAttention] compute: total tokens (" << total_tokens
                                                               << ") not divisible by batch size (" << effective_batch_size << ")");
            return false;
        }

        const int seq_len = total_tokens / effective_batch_size;

        auto *activation_output = dynamic_cast<IActivationTensor *>(output);
        if (!activation_output)
        {
            LOG_ERROR("[GQAAttention] compute: output tensor does not implement IActivationTensor");
            return false;
        }

        auto attention_kernel = activation_output->createAttention();
        if (!attention_kernel)
        {
            LOG_ERROR("[GQAAttention] compute: failed to create attention kernel");
            return false;
        }

        TensorBase *mask_tensor = nullptr;
        if (should_build_mask(config, effective_batch_size, sequence_lengths))
        {
            mask_tensor = config.workspace_mask.get();
            if (!build_sequence_mask(mask_tensor, effective_batch_size, seq_len, sequence_lengths, config))
            {
                return false;
            }
        }

        // Choose correct kernel path based on batch_size
        bool success;
        if (effective_batch_size > 1)
        {
            // Batch path: Call compute_batch with separate batch_size and seq_len
            success = attention_kernel->compute_batch(
                Q->data(),
                K->data(),
                V->data(),
                output->mutable_data(),
                effective_batch_size,
                seq_len,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                config.causal,
                config.window_size,
                config.workspace_scores.get(),
                config.workspace_qkv_buffer.get(),
                config.workspace_context.get(),
                mask_tensor,
                (config.precision == ActivationPrecision::BF16),
                config.mpi_ctx.get(),
                -1);
        }
        else
        {
            // Single sequence path: Call compute with total_tokens as seq_len
            success = attention_kernel->compute(
                Q->data(),
                K->data(),
                V->data(),
                output->mutable_data(),
                total_tokens,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                config.causal,
                config.window_size,
                config.workspace_scores.get(),
                config.workspace_qkv_buffer.get(),
                config.workspace_context.get(),
                mask_tensor,
                (config.precision == ActivationPrecision::BF16),
                config.mpi_ctx.get(),
                -1);
        }

        if (!success)
        {
            LOG_ERROR("[GQAAttention] compute: attention kernel invocation failed");
        }

        return success;
    }

    bool GQAAttention::compute_batch(
        TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
        const std::vector<int> &actual_lengths,
        int batch_size, int seq_len,
        const GQAAttentionConfig &config)
    {
        if (!validate_inputs(Q, K, V, output, config))
        {
            return false;
        }

        if (batch_size <= 0)
        {
            LOG_ERROR("[GQAAttention] compute_batch: invalid batch size " << batch_size);
            return false;
        }

        if (static_cast<int>(actual_lengths.size()) != batch_size)
        {
            LOG_ERROR("[GQAAttention] compute_batch: actual_lengths size (" << actual_lengths.size()
                                                                            << ") != batch_size (" << batch_size << ")");
            return false;
        }

        const auto &q_shape = Q->shape();
        if (q_shape.size() < 2)
        {
            LOG_ERROR("[GQAAttention] compute_batch: invalid Q tensor rank");
            return false;
        }

        const int total_seq_len = batch_size * seq_len;
        if (static_cast<int>(q_shape[0]) != total_seq_len)
        {
            LOG_ERROR("[GQAAttention] compute_batch: Q shape[0] (" << q_shape[0]
                                                                   << ") != batch_size * seq_len (" << total_seq_len << ")");
            return false;
        }

        auto *activation_output = dynamic_cast<IActivationTensor *>(output);
        if (!activation_output)
        {
            LOG_ERROR("[GQAAttention] compute_batch: output tensor does not implement IActivationTensor");
            return false;
        }

        auto attention_kernel = activation_output->createAttention();
        if (!attention_kernel)
        {
            LOG_ERROR("[GQAAttention] compute_batch: failed to create attention kernel");
            return false;
        }

        TensorBase *mask_tensor = nullptr;
        if (config.causal || !actual_lengths.empty())
        {
            mask_tensor = config.workspace_mask.get();
            if (!build_combined_batch_mask(mask_tensor, batch_size, seq_len, actual_lengths, config))
            {
                return false;
            }
        }

        const bool success = attention_kernel->compute_batch(
            Q->data(),
            K->data(),
            V->data(),
            output->mutable_data(),
            batch_size,
            seq_len,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            config.causal,
            config.window_size,
            config.workspace_scores.get(),
            config.workspace_qkv_buffer.get(),
            config.workspace_context.get(),
            mask_tensor,
            (config.precision == ActivationPrecision::BF16),
            config.mpi_ctx.get(),
            -1);

        if (!success)
        {
            LOG_ERROR("[GQAAttention] compute_batch: attention kernel invocation failed");
        }

        return success;
    }

    bool GQAAttention::compute_mpi(
        TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
        const GQAAttentionConfig &config,
        int batch_size,
        const std::vector<int> *sequence_lengths)
    {
        // The orchestrator now handles MPI fan-out and reuses the single-rank kernel
        // implementation. Retain this entry point for callers that still expect it.
        return compute(Q, K, V, output, config, batch_size, sequence_lengths);
    }

    bool GQAAttention::compute_tensor_parallel(
        TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
        const GQAAttentionConfig &config,
        int batch_size,
        const std::vector<int> *sequence_lengths)
    {
        LOG_ERROR("[GQAAttention] Tensor-parallel execution is orchestrated upstream and should not invoke compute_tensor_parallel directly");
        return false;
    }

    // ============================================================================
    // Private Helper Functions (Testable Atomic Operations)
    // ============================================================================

    bool GQAAttention::validate_inputs(
        const TensorBase *Q, const TensorBase *K, const TensorBase *V,
        const TensorBase *output, const GQAAttentionConfig &config)
    {
        // Check for null pointers
        if (!Q || !K || !V || !output)
        {
            LOG_ERROR("[GQAAttention] validate_inputs: null pointer detected");
            return false;
        }

        // Validate head configuration
        if (config.n_heads % config.n_kv_heads != 0)
        {
            LOG_ERROR("[GQAAttention] validate_inputs: n_heads (" << config.n_heads
                                                                  << ") must be divisible by n_kv_heads (" << config.n_kv_heads << ")");
            return false;
        }

        // Validate tensor dimensions
        const auto &q_shape = Q->shape();
        const auto &k_shape = K->shape();
        const auto &v_shape = V->shape();
        const auto &out_shape = output->shape();

        if (q_shape.size() != 2)
        {
            LOG_ERROR("[GQAAttention] validate_inputs: Q must be 2D, got " << q_shape.size() << "D");
            return false;
        }

        // Validate Q dimensions
        int expected_q_dim = config.n_heads * config.head_dim;
        if (q_shape[1] != (size_t)expected_q_dim)
        {
            LOG_ERROR("[GQAAttention] validate_inputs: Q dimension mismatch. Expected "
                      << expected_q_dim << ", got " << q_shape[1]);
            return false;
        }

        // Validate K/V dimensions
        int expected_kv_dim = config.n_kv_heads * config.head_dim;
        if (k_shape.size() != 2 || k_shape[1] != (size_t)expected_kv_dim)
        {
            LOG_ERROR("[GQAAttention] validate_inputs: K dimension mismatch. Expected [*, "
                      << expected_kv_dim << "], got [" << k_shape[0] << ", " << k_shape[1] << "]");
            return false;
        }

        if (v_shape.size() != 2 || v_shape[1] != (size_t)expected_kv_dim)
        {
            LOG_ERROR("[GQAAttention] validate_inputs: V dimension mismatch. Expected [*, "
                      << expected_kv_dim << "], got [" << v_shape[0] << ", " << v_shape[1] << "]");
            return false;
        }

        // Validate sequence length consistency
        if (q_shape[0] != k_shape[0] || q_shape[0] != v_shape[0])
        {
            LOG_ERROR("[GQAAttention] validate_inputs: Sequence length mismatch. Q="
                      << q_shape[0] << ", K=" << k_shape[0] << ", V=" << v_shape[0]);
            return false;
        }

        // Validate output dimensions
        if (out_shape != q_shape)
        {
            LOG_ERROR("[GQAAttention] validate_inputs: Output shape mismatch. Expected "
                      << "[" << q_shape[0] << ", " << q_shape[1] << "], got ["
                      << out_shape[0] << ", " << out_shape[1] << "]");
            return false;
        }

        return true;
    }

    void GQAAttention::broadcast_kv_heads_if_needed(
        const float *K_in, const float *V_in,
        std::vector<float> &K_out, std::vector<float> &V_out,
        int seq_len, int n_heads, int n_kv_heads, int head_dim)
    {
        if (n_kv_heads >= n_heads)
        {
            // No broadcasting needed (MHA) - just copy input
            size_t total_elements = seq_len * n_heads * head_dim;
            K_out.assign(K_in, K_in + total_elements);
            V_out.assign(V_in, V_in + total_elements);
            return;
        }

        // GQA or MQA: Broadcast K/V heads
        K_out.resize(seq_len * n_heads * head_dim);
        V_out.resize(seq_len * n_heads * head_dim);

        attention_utils::broadcast_kv_heads(
            K_in, K_out.data(),
            seq_len, n_heads, n_kv_heads, head_dim);

        attention_utils::broadcast_kv_heads(
            V_in, V_out.data(),
            seq_len, n_heads, n_kv_heads, head_dim);
    }

    void GQAAttention::extract_head_data(
        const float *strided_data, float *contiguous_out,
        int seq_len, int head_dim, int n_heads, int head_idx,
        int batch_offset)
    {
        // Extract contiguous head data from strided multi-head layout
        for (int s = 0; s < seq_len; ++s)
        {
#pragma omp simd
            for (int d = 0; d < head_dim; ++d)
            {
                const int src_idx = (batch_offset + s) * n_heads * head_dim + head_idx * head_dim + d;
                const int dst_idx = s * head_dim + d;
                contiguous_out[dst_idx] = strided_data[src_idx];
            }
        }
    }

    bool GQAAttention::compute_attention_scores(
        const float *Q, const float *K, float *scores,
        int seq_len, int head_dim, ActivationPrecision precision)
    {
        // GEMM: scores = Q @ K^T
        // Q: [seq_len, head_dim], K: [seq_len, head_dim]
        // scores: [seq_len, seq_len]

        if (precision == ActivationPrecision::BF16)
        {
            // BF16: Create temporary BF16Tensor view of K to get auto-tuned GEMM kernel
            BF16Tensor K_tensor({static_cast<size_t>(seq_len), static_cast<size_t>(head_dim)});
            K_tensor.from_fp32(K, seq_len * head_dim);

            auto gemm_kernel = K_tensor.createGemm();
            return gemm_kernel->multiply_activations(
                Q, K_tensor.data(),
                scores,
                seq_len,  // m
                seq_len,  // n
                head_dim, // k
                true,     // transpose_B (K^T)
                1.0f,     // alpha
                0.0f);    // beta
        }
        else
        {
            // FP32 or FP16 (both use FP32 path with auto-tuner)
            // Create temporary FP32Tensor view of K to get auto-tuned GEMM kernel
            FP32Tensor K_tensor({static_cast<size_t>(seq_len), static_cast<size_t>(head_dim)}, -1);
            std::memcpy(K_tensor.mutable_data(), K, seq_len * head_dim * sizeof(float));

            auto gemm_kernel = K_tensor.createGemm();
            return gemm_kernel->multiply_activations(
                Q, K_tensor.data(),
                scores,
                seq_len,  // m
                seq_len,  // n
                head_dim, // k
                true,     // transpose_B (K^T)
                1.0f,     // alpha
                0.0f);    // beta
        }
    }

    void GQAAttention::scale_scores_inplace(
        float *scores, int size, int head_dim)
    {
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

#pragma omp parallel for if (size > 8192)
        for (int i = 0; i < size; ++i)
        {
            scores[i] *= scale;
        }
    }

    void GQAAttention::apply_attention_mask(
        float *scores, int seq_len, int batch_size,
        const int *seq_lengths, bool causal, int window_size,
        const GQAAttentionConfig &config)
    {
        if (!causal && !seq_lengths)
        {
            // No masking needed
            return;
        }

        // Use workspace mask buffer
        if (!config.workspace_mask)
        {
            LOG_ERROR("[GQAAttention] apply_attention_mask: workspace_mask not provided");
            return;
        }

        float *mask = config.workspace_mask->mutable_data();

        if (batch_size == 1)
        {
            // Single sequence: standard causal mask
            if (causal)
            {
                attention_utils::create_causal_mask(mask, seq_len, window_size);
                attention_utils::apply_attention_mask(scores, mask, seq_len, seq_len);
            }
        }
        else
        {
            // Batched sequences: block-diagonal mask with padding
            attention_utils::create_batch_causal_mask(
                mask, batch_size, seq_len, seq_lengths, window_size);

            // Apply to all heads (mask is shared across heads within a batch)
            attention_utils::apply_attention_mask(scores, mask, batch_size * seq_len, seq_len);
        }
    }

    void GQAAttention::apply_softmax(
        float *scores, int rows, int cols)
    {
        primitives::SoftmaxRowArgs softmax_args;
        softmax_args.causal = false; // Mask already applied
        softmax_args.scale = 1.0f;   // Scaling already done
        softmax_args.rows = rows;
        softmax_args.cols = cols;
        softmax_args.scores = scores;

        primitives::softmax_row_major_vectorized(softmax_args);
    }

    bool GQAAttention::compute_context_from_scores(
        const float *scores, const float *V, float *context,
        int seq_len, int head_dim, ActivationPrecision precision)
    {
        // GEMM: context = scores @ V
        // scores: [seq_len, seq_len], V: [seq_len, head_dim]
        // context: [seq_len, head_dim]

        // For context GEMM we treat V as an activation matrix and
        // use the activation-activation GEMM interface. This avoids
        // weight-only restrictions (e.g., transpose-only layouts)
        // and matches the shapes used in attention tests.

        if (precision == ActivationPrecision::BF16)
        {
            BF16Tensor V_tensor({static_cast<size_t>(seq_len), static_cast<size_t>(head_dim)});
            V_tensor.from_fp32(V, seq_len * head_dim);

            auto gemm_kernel = V_tensor.createGemm();
            return gemm_kernel->multiply_activations(
                scores,
                V_tensor.data(),
                context,
                /*m=*/seq_len,
                /*n=*/head_dim,
                /*k=*/seq_len,
                /*transpose_B=*/false,
                /*alpha=*/1.0f,
                /*beta=*/0.0f);
        }
        else
        {
            FP32Tensor V_tensor({static_cast<size_t>(seq_len), static_cast<size_t>(head_dim)}, -1);
            std::memcpy(V_tensor.mutable_data(), V, seq_len * head_dim * sizeof(float));

            auto gemm_kernel = V_tensor.createGemm();
            return gemm_kernel->multiply_activations(
                scores,
                V_tensor.data(),
                context,
                /*m=*/seq_len,
                /*n=*/head_dim,
                /*k=*/seq_len,
                /*transpose_B=*/false,
                /*alpha=*/1.0f,
                /*beta=*/0.0f);
        }
    }

    void GQAAttention::write_context_to_output(
        const float *context, float *output,
        int seq_len, int head_dim, int n_heads, int head_idx,
        int batch_offset)
    {
        // Write contiguous context to strided multi-head output
        for (int s = 0; s < seq_len; ++s)
        {
#pragma omp simd
            for (int d = 0; d < head_dim; ++d)
            {
                const int src_idx = s * head_dim + d;
                const int dst_idx = (batch_offset + s) * n_heads * head_dim + head_idx * head_dim + d;
                output[dst_idx] = context[src_idx];
            }
        }
    }

} // namespace llaminar2
