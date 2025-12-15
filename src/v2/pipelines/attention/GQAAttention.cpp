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
#include "../../utils/OpenMPUtils.h"
#include "../../tensors/TensorFactory.h"
#include "../../tensors/Tensors.h"
#include "../../tensors/SIMDHelpers.h"
#include "../../kernels/cpu/primitives/SoftmaxPrimitives.h"
#include "../../kernels/cpu/attention/CPUAttentionKernelTyped.h"
#include <cstring>
#include <cmath>
#include <sstream>
#include <omp.h>
#include <atomic>

namespace llaminar2
{
    namespace
    {
        /**
         * @brief Factory function to create attention kernel based on precision
         *
         * This decouples kernel selection from output tensor type, allowing
         * Q8_1 inputs with FP32 output to still use the Q8_1 kernel.
         */
        std::unique_ptr<ITensorAttention> createAttentionKernelForPrecision(ActivationPrecision precision)
        {
            switch (precision)
            {
            case ActivationPrecision::Q8_1:
                return std::make_unique<CPUAttentionKernelTyped<ActivationPrecision::Q8_1>>();
            case ActivationPrecision::BF16:
                return std::make_unique<CPUAttentionKernelTyped<ActivationPrecision::BF16>>();
            case ActivationPrecision::FP16:
                return std::make_unique<CPUAttentionKernelTyped<ActivationPrecision::FP16>>();
            case ActivationPrecision::FP32:
            default:
                return std::make_unique<CPUAttentionKernelTyped<ActivationPrecision::FP32>>();
            }
        }
    } // namespace

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
        LOG_TRACE("[GQAAttention::compute] Called with precision=" << static_cast<int>(config.precision));

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

        // Create attention kernel based on config.precision, NOT output tensor type
        // This allows Q8_1 inputs to use the Q8_1 kernel even with FP32 output
        auto attention_kernel = createAttentionKernelForPrecision(config.precision);
        if (!attention_kernel)
        {
            LOG_ERROR("[GQAAttention] compute: failed to create attention kernel for precision "
                      << static_cast<int>(config.precision));
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

        // Get data pointers based on precision
        // For Q8_1, we need to pass the raw Q8_1 block pointers (cast as float*) to the kernel
        // The kernel will interpret these as Q8_1Block* internally
        const float *Q_ptr = nullptr;
        const float *K_ptr = nullptr;
        const float *V_ptr = nullptr;
        float *output_ptr = nullptr;

        // Determine effective precision based on actual tensor types
        // This handles the case where Q is Q8_1 but K/V come from FP32 KV cache
        ActivationPrecision effective_precision = config.precision;

        // Temporary buffers for mixed precision scenarios
        std::unique_ptr<FP32Tensor> temp_fp32_output_buffer; // For FP32 fallback with Q8_1 output tensor
        std::unique_ptr<Q8_1Tensor> temp_q8_1_output_buffer; // For Q8_1 compute with FP32 output tensor
        Q8_1Tensor *out_q8_1_for_requant = nullptr;          // Q8_1 output needing requant from FP32
        FP32Tensor *out_fp32_for_dequant = nullptr;          // FP32 output needing dequant from Q8_1

        if (config.precision == ActivationPrecision::Q8_1)
        {
            LOG_TRACE("[GQAAttention] Q8_1 precision requested - checking tensor types");

            // Check if tensors are actually Q8_1
            auto *Q_q8_1 = dynamic_cast<Q8_1Tensor *>(Q);
            auto *K_q8_1 = dynamic_cast<Q8_1Tensor *>(K);
            auto *V_q8_1 = dynamic_cast<Q8_1Tensor *>(V);
            auto *out_q8_1 = dynamic_cast<Q8_1Tensor *>(output);
            auto *out_fp32 = dynamic_cast<FP32Tensor *>(output);

            LOG_TRACE("[GQAAttention] Tensor types: Q=" << (Q_q8_1 ? "Q8_1" : "other")
                                                        << ", K=" << (K_q8_1 ? "Q8_1" : "other")
                                                        << ", V=" << (V_q8_1 ? "Q8_1" : "other")
                                                        << ", output=" << (out_q8_1 ? "Q8_1" : (out_fp32 ? "FP32" : "other")));

            if (Q_q8_1 && K_q8_1 && V_q8_1)
            {
                // All inputs are Q8_1: use native Q8_1 path
                Q_ptr = reinterpret_cast<const float *>(Q_q8_1->q8_1_blocks());
                K_ptr = reinterpret_cast<const float *>(K_q8_1->q8_1_blocks());
                V_ptr = reinterpret_cast<const float *>(V_q8_1->q8_1_blocks());

                if (out_q8_1)
                {
                    // All tensors Q8_1: native Q8_1 throughout
                    LOG_TRACE("[GQAAttention] Native Q8_1 path: Q/K/V/output all Q8_1");
                    output_ptr = reinterpret_cast<float *>(out_q8_1->mutable_q8_1_blocks());
                }
                else if (out_fp32)
                {
                    // Q/K/V are Q8_1 but output is FP32: run Q8_1 kernel to temp, then dequant
                    LOG_TRACE("[GQAAttention] Q8_1 compute with FP32 output: using temp Q8_1 buffer + dequant");
                    const auto &out_shape = output->shape();
                    temp_q8_1_output_buffer = std::make_unique<Q8_1Tensor>(out_shape);
                    output_ptr = reinterpret_cast<float *>(temp_q8_1_output_buffer->mutable_q8_1_blocks());
                    out_fp32_for_dequant = out_fp32;
                }
                else
                {
                    // Unknown output type: fall back to FP32
                    LOG_WARN("[GQAAttention] Q8_1 inputs but unknown output type: falling back to FP32");
                    effective_precision = ActivationPrecision::FP32;
                    Q_ptr = Q->fp32_data(); // Explicit dequantization
                    K_ptr = K->fp32_data();
                    V_ptr = V->fp32_data();
                    output_ptr = output->mutable_data();
                }
            }
            else
            {
                // Mixed precision case: Q/K/V might be Q8_1 or FP32 (from KV cache)
                // Use fp32_data() for explicit dequantization (also works for FP32 tensors)
                LOG_TRACE("[GQAAttention] Mixed precision detected: using FP32 path with explicit dequantization");
                effective_precision = ActivationPrecision::FP32;
                Q_ptr = Q->fp32_data(); // Explicit dequantization if Q8_1
                K_ptr = K->fp32_data(); // Already FP32 (from cache) or explicitly dequantizes
                V_ptr = V->fp32_data();

                // Check if output is Q8_1 - if so, need temp buffer and requantization
                if (out_q8_1)
                {
                    // Output is Q8_1: create temp FP32 buffer, requantize after kernel
                    const auto &out_shape = output->shape();
                    temp_fp32_output_buffer = std::make_unique<FP32Tensor>(out_shape);
                    output_ptr = temp_fp32_output_buffer->mutable_data();
                    out_q8_1_for_requant = out_q8_1;
                }
                else
                {
                    // Output is already FP32: use directly
                    output_ptr = output->mutable_data();
                }
            }
        }
        else
        {
            // FP32/BF16/FP16 path: Use fp32_data() which returns FP32 pointer
            // For non-Q8_1 tensors, this is equivalent to data()
            Q_ptr = Q->fp32_data();
            K_ptr = K->fp32_data();
            V_ptr = V->fp32_data();
            output_ptr = output->mutable_data();
        }

        // Choose correct kernel path based on batch_size
        bool success;
        if (effective_batch_size > 1)
        {
            // Batch path: Call compute_batch with separate batch_size and seq_len
            success = attention_kernel->compute_batch(
                Q_ptr,
                K_ptr,
                V_ptr,
                output_ptr,
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
                Q_ptr,
                K_ptr,
                V_ptr,
                output_ptr,
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
            return false;
        }

        // Post-compute precision conversions

        // Case 1: FP32 compute with Q8_1 output → requantize
        if (out_q8_1_for_requant && temp_fp32_output_buffer)
        {
            // Quantize FP32 output to Q8_1
            const float *fp32_data = temp_fp32_output_buffer->data();
            Q8_1Block *out_blocks = out_q8_1_for_requant->mutable_q8_1_blocks();
            const auto &shape = temp_fp32_output_buffer->shape();
            size_t n_elements = 1;
            for (size_t d : shape)
                n_elements *= d;
            simd::quantize_fp32_to_q8_1_blocks(fp32_data, out_blocks, n_elements);
            LOG_TRACE("[GQAAttention] Requantized attention output to Q8_1 (" << n_elements << " elements)");
        }

        // Case 2: Q8_1 compute with FP32 output → dequantize
        if (out_fp32_for_dequant && temp_q8_1_output_buffer)
        {
            // Dequantize Q8_1 output to FP32
            const Q8_1Block *q8_blocks = temp_q8_1_output_buffer->q8_1_blocks();
            float *fp32_data = out_fp32_for_dequant->mutable_data();
            const auto &shape = temp_q8_1_output_buffer->shape();
            size_t n_elements = 1;
            for (size_t d : shape)
                n_elements *= d;
            simd::dequantize_q8_1_to_fp32(q8_blocks, fp32_data, n_elements);
            LOG_TRACE("[GQAAttention] Dequantized Q8_1 attention output to FP32 (" << n_elements << " elements)");
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

        // Create attention kernel based on config.precision, NOT output tensor type
        auto attention_kernel = createAttentionKernelForPrecision(config.precision);
        if (!attention_kernel)
        {
            LOG_ERROR("[GQAAttention] compute_batch: failed to create attention kernel for precision "
                      << static_cast<int>(config.precision));
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

        // Get data pointers based on precision
        // For Q8_1, we need to pass the raw Q8_1 block pointers (cast as float*) to the kernel
        const float *Q_ptr = nullptr;
        const float *K_ptr = nullptr;
        const float *V_ptr = nullptr;
        float *output_ptr = nullptr;

        // Temporary buffer for Q8_1 compute with FP32 output
        std::unique_ptr<Q8_1Tensor> temp_q8_1_output_buffer;
        FP32Tensor *out_fp32_for_dequant = nullptr;

        if (config.precision == ActivationPrecision::Q8_1)
        {
            // Q8_1 path: Pass raw Q8_1 block pointers
            auto *Q_q8_1 = dynamic_cast<Q8_1Tensor *>(Q);
            auto *K_q8_1 = dynamic_cast<Q8_1Tensor *>(K);
            auto *V_q8_1 = dynamic_cast<Q8_1Tensor *>(V);
            auto *out_q8_1 = dynamic_cast<Q8_1Tensor *>(output);
            auto *out_fp32 = dynamic_cast<FP32Tensor *>(output);

            if (!Q_q8_1 || !K_q8_1 || !V_q8_1)
            {
                LOG_ERROR("[GQAAttention] compute_batch: Q8_1 precision configured but input tensors are not Q8_1Tensor");
                return false;
            }

            Q_ptr = reinterpret_cast<const float *>(Q_q8_1->q8_1_blocks());
            K_ptr = reinterpret_cast<const float *>(K_q8_1->q8_1_blocks());
            V_ptr = reinterpret_cast<const float *>(V_q8_1->q8_1_blocks());

            if (out_q8_1)
            {
                // All tensors Q8_1: native path
                output_ptr = reinterpret_cast<float *>(out_q8_1->mutable_q8_1_blocks());
            }
            else if (out_fp32)
            {
                // Q/K/V are Q8_1 but output is FP32: compute to temp Q8_1, then dequant
                LOG_TRACE("[GQAAttention] compute_batch: Q8_1 compute with FP32 output");
                const auto &out_shape = output->shape();
                temp_q8_1_output_buffer = std::make_unique<Q8_1Tensor>(out_shape);
                output_ptr = reinterpret_cast<float *>(temp_q8_1_output_buffer->mutable_q8_1_blocks());
                out_fp32_for_dequant = out_fp32;
            }
            else
            {
                LOG_ERROR("[GQAAttention] compute_batch: Q8_1 precision but unknown output type");
                return false;
            }
        }
        else
        {
            Q_ptr = Q->fp32_data(); // Use explicit dequantization for non-Q8_1 paths
            K_ptr = K->fp32_data();
            V_ptr = V->fp32_data();
            output_ptr = output->mutable_data();
        }

        const bool success = attention_kernel->compute_batch(
            Q_ptr,
            K_ptr,
            V_ptr,
            output_ptr,
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
            return false;
        }

        // Dequantize Q8_1 temp output to FP32 if needed
        if (out_fp32_for_dequant && temp_q8_1_output_buffer)
        {
            const Q8_1Block *q8_blocks = temp_q8_1_output_buffer->q8_1_blocks();
            float *fp32_data = out_fp32_for_dequant->mutable_data();
            const auto &shape = temp_q8_1_output_buffer->shape();
            size_t n_elements = 1;
            for (size_t d : shape)
                n_elements *= d;
            simd::dequantize_q8_1_to_fp32(q8_blocks, fp32_data, n_elements);
            LOG_TRACE("[GQAAttention] compute_batch: Dequantized Q8_1 output to FP32 (" << n_elements << " elements)");
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

        bool do_parallel = (size > 8192);
        auto scale_work = [&]()
        {
#pragma omp for schedule(static)
            for (int i = 0; i < size; ++i)
            {
                scores[i] *= scale;
            }
        };
        OMP_WORKSHARE_REGION_IF(scale_work, do_parallel);
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
