/**
 * @file MpiAttentionOrchestrator.cpp
 * @brief Grouped Query Attention (GQA) implementation
 * @author David Sanftenberg
 */

#include "MpiAttentionOrchestrator.h"
#include "GQAAttention.h"
#include "../AttentionUtils.h"
#include "../../utils/Logger.h"
#include "../../utils/DebugAssert.h"
#include "../../utils/MPIStager.h"
#include "../../tensors/TensorFactory.h"
#include "../../tensors/Tensors.h"
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

        /**
         * @brief Detect the actual precision of input tensors
         *
         * Returns the precision that matches the actual tensor types.
         * This allows us to select the appropriate native path based on
         * the runtime tensor types, not just the config setting.
         */
        ActivationPrecision detectTensorPrecision(const TensorBase *tensor)
        {
            if (!tensor)
            {
                LOG_DEBUG("[detectTensorPrecision] tensor is null, returning FP32");
                return ActivationPrecision::FP32;
            }

            if (dynamic_cast<const Q8_1Tensor *>(tensor))
            {
                LOG_TRACE("[detectTensorPrecision] detected Q8_1Tensor");
                return ActivationPrecision::Q8_1;
            }
            if (dynamic_cast<const BF16Tensor *>(tensor))
            {
                LOG_TRACE("[detectTensorPrecision] detected BF16Tensor");
                return ActivationPrecision::BF16;
            }
            if (dynamic_cast<const FP16Tensor *>(tensor))
            {
                LOG_TRACE("[detectTensorPrecision] detected FP16Tensor");
                return ActivationPrecision::FP16;
            }
            // Default to FP32 (FP32Tensor or any other type that provides float data())
            LOG_DEBUG("[detectTensorPrecision] defaulting to FP32, tensor type=" << typeid(*tensor).name());
            return ActivationPrecision::FP32;
        }

        /**
         * @brief Check if native tensor-parallel path should be used for the given precision
         *
         * Returns true if:
         * - Q, K, V all have the same precision type
         * - output tensor matches (or can receive) that precision
         * - head_dim meets precision-specific requirements
         *
         * For Q8_1: head_dim must be multiple of 32, and 64 or 128 (JIT kernel)
         * For BF16/FP16: no special requirements (element-wise ops)
         * For FP32: always supported
         */
        bool shouldUseNativeTensorParallel(
            const TensorBase *Q, const TensorBase *K, const TensorBase *V,
            const TensorBase *output,
            const MpiAttentionConfig &config,
            ActivationPrecision &detected_precision)
        {
            // Detect input precision from Q tensor
            detected_precision = detectTensorPrecision(Q);

            // Verify K and V match
            if (detectTensorPrecision(K) != detected_precision ||
                detectTensorPrecision(V) != detected_precision)
            {
                // Mixed precision inputs - fall back to FP32
                detected_precision = ActivationPrecision::FP32;
                return false;
            }

            // Check output tensor compatibility
            ActivationPrecision output_precision = detectTensorPrecision(output);

            // Native path requires matching input/output precision
            // (or FP32 output for precision types that don't have native output support yet)
            if (output_precision != detected_precision && output_precision != ActivationPrecision::FP32)
            {
                detected_precision = ActivationPrecision::FP32;
                return false;
            }

            // Precision-specific requirements
            switch (detected_precision)
            {
            case ActivationPrecision::Q8_1:
                // Q8_1 requires block-aligned head_dim
                if (config.head_dim % Q8_1Block::BLOCK_SIZE != 0)
                    return false;
                // JIT kernel only supports 64 or 128
                if (config.head_dim != 64 && config.head_dim != 128)
                    return false;
                return true;

            case ActivationPrecision::BF16:
            case ActivationPrecision::FP16:
                // BF16/FP16 native paths require even head_dim for SIMD
                if (config.head_dim % 2 != 0)
                    return false;
                return true;

            case ActivationPrecision::FP32:
                // FP32 always supported
                return true;

            default:
                return false;
            }
        }

        /**
         * @brief Slice Q8_1 blocks for a range of heads
         *
         * Q8_1 tensor layout: [seq_len, n_heads * head_dim_blocks] where head_dim_blocks = head_dim / 32
         * This extracts blocks for heads [start_head, start_head + local_n_heads).
         *
         * @param src Source Q8_1 blocks (full tensor)
         * @param dst Destination Q8_1 blocks (local heads only)
         * @param seq_len Sequence length
         * @param n_heads Total number of heads
         * @param head_dim Head dimension (must be multiple of 32)
         * @param start_head First head index for this rank
         * @param local_n_heads Number of heads for this rank
         */
        void sliceQ8_1HeadBlocks(
            const Q8_1Block *src, Q8_1Block *dst,
            int seq_len, int n_heads, int head_dim,
            size_t start_head, size_t local_n_heads)
        {
            const int head_dim_blocks = head_dim / Q8_1Block::BLOCK_SIZE; // blocks per head
            const int src_blocks_per_row = n_heads * head_dim_blocks;
            const int dst_blocks_per_row = static_cast<int>(local_n_heads) * head_dim_blocks;

#pragma omp parallel for
            for (int t = 0; t < seq_len; ++t)
            {
                const Q8_1Block *src_row = src + t * src_blocks_per_row;
                Q8_1Block *dst_row = dst + t * dst_blocks_per_row;

                for (size_t local_h = 0; local_h < local_n_heads; ++local_h)
                {
                    size_t global_h = start_head + local_h;
                    const Q8_1Block *src_head = src_row + global_h * head_dim_blocks;
                    Q8_1Block *dst_head = dst_row + local_h * head_dim_blocks;

                    // Copy all blocks for this head
                    std::memcpy(dst_head, src_head, head_dim_blocks * sizeof(Q8_1Block));
                }
            }
        }

        /**
         * @brief Dequantize Q8_1 blocks to FP32
         *
         * Converts Q8_1 block array to FP32 float array.
         * output[i] = q8_1.qs[i % 32] * fp16_to_fp32(q8_1.d)
         *
         * @param src Q8_1 blocks
         * @param dst FP32 output
         * @param total_blocks Total number of Q8_1 blocks
         */
        void dequantizeQ8_1ToFP32(const Q8_1Block *src, float *dst, size_t total_blocks)
        {
#pragma omp parallel for
            for (size_t b = 0; b < total_blocks; ++b)
            {
                const Q8_1Block &block = src[b];
                float *out = dst + b * Q8_1Block::BLOCK_SIZE;
                float d = fp16_to_fp32(block.d);
                for (int i = 0; i < 32; ++i)
                {
                    out[i] = static_cast<float>(block.qs[i]) * d;
                }
            }
        }

        /**
         * @brief Broadcast K/V Q8_1 blocks to match Q heads (GQA expansion)
         *
         * When n_heads > n_kv_heads (GQA/MQA), each KV head is replicated to multiple Q heads.
         * This creates local K/V buffers where each KV head is broadcast to match local_n_heads.
         *
         * @param K_src Source K blocks [seq_len, n_kv_heads, head_dim_blocks]
         * @param V_src Source V blocks [seq_len, n_kv_heads, head_dim_blocks]
         * @param K_dst Destination K blocks [seq_len, local_n_heads, head_dim_blocks]
         * @param V_dst Destination V blocks [seq_len, local_n_heads, head_dim_blocks]
         * @param seq_len Sequence length
         * @param n_kv_heads Number of KV heads in source
         * @param head_dim Head dimension (must be multiple of 32)
         * @param start_head First Q head index for this rank
         * @param local_n_heads Number of Q heads for this rank
         * @param n_heads Total number of Q heads
         */
        void broadcastQ8_1KVHeads(
            const Q8_1Block *K_src, const Q8_1Block *V_src,
            Q8_1Block *K_dst, Q8_1Block *V_dst,
            int seq_len, int n_kv_heads, int head_dim,
            size_t start_head, size_t local_n_heads, int n_heads)
        {
            const int head_dim_blocks = head_dim / Q8_1Block::BLOCK_SIZE;
            const int kv_blocks_per_row = n_kv_heads * head_dim_blocks;
            const int dst_blocks_per_row = static_cast<int>(local_n_heads) * head_dim_blocks;
            const int heads_per_kv = n_heads / n_kv_heads;

#pragma omp parallel for
            for (int t = 0; t < seq_len; ++t)
            {
                const Q8_1Block *K_row = K_src + t * kv_blocks_per_row;
                const Q8_1Block *V_row = V_src + t * kv_blocks_per_row;
                Q8_1Block *K_dst_row = K_dst + t * dst_blocks_per_row;
                Q8_1Block *V_dst_row = V_dst + t * dst_blocks_per_row;

                for (size_t local_h = 0; local_h < local_n_heads; ++local_h)
                {
                    size_t global_h = start_head + local_h;
                    int kv_h = static_cast<int>(global_h) / heads_per_kv; // Which KV head to use

                    const Q8_1Block *K_head = K_row + kv_h * head_dim_blocks;
                    const Q8_1Block *V_head = V_row + kv_h * head_dim_blocks;
                    Q8_1Block *K_dst_head = K_dst_row + local_h * head_dim_blocks;
                    Q8_1Block *V_dst_head = V_dst_row + local_h * head_dim_blocks;

                    // Copy all blocks for this head
                    std::memcpy(K_dst_head, K_head, head_dim_blocks * sizeof(Q8_1Block));
                    std::memcpy(V_dst_head, V_head, head_dim_blocks * sizeof(Q8_1Block));
                }
            }
        }

        /**
         * @brief Native Q8_1 tensor-parallel attention implementation
         *
         * Performs tensor-parallel attention entirely in Q8_1 domain:
         *   1. Slices Q Q8_1 blocks for local heads (no dequantization)
         *   2. Broadcasts K/V Q8_1 blocks to match local Q heads (GQA expansion)
         *   3. Runs native Q8_1 attention kernel (JIT GEMM with fused softmax)
         *   4. Uses MPI_Allgatherv to combine Q8_1 blocks from all ranks
         *   5. Output remains in Q8_1 format (no dequant→requant round trip!)
         *
         * This avoids the expensive FP32 dequantization of Q/K/V tensors AND
         * keeps the output in Q8_1 format, eliminating quantization noise from
         * unnecessary format conversions.
         */
        bool compute_tensor_parallel_q8_1_native(
            TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
            const MpiAttentionConfig &config,
            int rank, int world_size,
            int total_tokens, int effective_batch_size, int seq_len, int padded_seq_len,
            size_t start_head, size_t local_n_heads,
            const std::vector<int> *sequence_lengths)
        {
            (void)padded_seq_len; // Not used in Q8_1 path yet

            // Cast inputs to Q8_1Tensor
            auto *Q_q8 = dynamic_cast<Q8_1Tensor *>(Q);
            auto *K_q8 = dynamic_cast<Q8_1Tensor *>(K);
            auto *V_q8 = dynamic_cast<Q8_1Tensor *>(V);
            auto *output_q8 = dynamic_cast<Q8_1Tensor *>(output);

            if (!Q_q8 || !K_q8 || !V_q8)
            {
                LOG_ERROR("[MPI TP Q8_1] Input tensors are not Q8_1Tensor");
                return false;
            }

            // Check if output supports Q8_1 native path
            const bool output_is_q8_1 = (output_q8 != nullptr);
            LOG_TRACE("[MPI TP Q8_1] Output tensor type: " << (output_is_q8_1 ? "Q8_1" : "FP32"));

            // Get Q8_1 block pointers (native format, no dequantization!)
            const Q8_1Block *Q_blocks = Q_q8->q8_1_blocks();
            const Q8_1Block *K_blocks = K_q8->q8_1_blocks();
            const Q8_1Block *V_blocks = V_q8->q8_1_blocks();

            if (!Q_blocks || !K_blocks || !V_blocks)
            {
                LOG_ERROR("[MPI TP Q8_1] Null Q8_1 block pointers");
                return false;
            }

            // Block layout parameters
            const int head_dim_blocks = config.head_dim / Q8_1Block::BLOCK_SIZE; // blocks per head

            // Calculate local buffer sizes (in blocks)
            // Q: sliced to local_n_heads
            // K/V: broadcast from n_kv_heads to local_n_heads (GQA expansion)
            const size_t q_local_blocks = total_tokens * local_n_heads * head_dim_blocks;
            const size_t kv_local_blocks = total_tokens * local_n_heads * head_dim_blocks; // Broadcast to match Q!

            // Allocate local Q8_1 buffers
            std::vector<Q8_1Block> Q_local(q_local_blocks);
            std::vector<Q8_1Block> K_local(kv_local_blocks);
            std::vector<Q8_1Block> V_local(kv_local_blocks);
            std::vector<Q8_1Block> output_q8_local(q_local_blocks);

            // Slice Q heads for this rank
            sliceQ8_1HeadBlocks(Q_blocks, Q_local.data(), total_tokens, config.n_heads, config.head_dim, start_head, local_n_heads);

            // Broadcast K/V from n_kv_heads to local_n_heads (GQA expansion)
            // This replicates each KV head to the corresponding Q heads
            broadcastQ8_1KVHeads(
                K_blocks, V_blocks, K_local.data(), V_local.data(),
                total_tokens, config.n_kv_heads, config.head_dim,
                start_head, local_n_heads, config.n_heads);

            LOG_TRACE("[MPI TP Q8_1] Rank " << rank << ": sliced Q to " << q_local_blocks
                                            << " blocks, broadcast K/V to " << kv_local_blocks << " blocks (GQA ratio="
                                            << (config.n_heads / config.n_kv_heads) << ")");

            // Create attention mask if needed
            const bool needs_mask = config.causal || (sequence_lengths && !sequence_lengths->empty());
            TensorBase *mask_tensor = config.workspace_mask.get();
            LOG_TRACE("[MPI TP Q8_1] needs_mask=" << needs_mask << " config.causal=" << config.causal
                                                  << " mask_tensor=" << (mask_tensor ? "valid" : "nullptr"));
            if (needs_mask && !mask_tensor)
            {
                LOG_ERROR("[MPI TP Q8_1] workspace_mask is required for causal/padded attention");
                return false;
            }

            if (needs_mask)
            {
                float *mask_data = mask_tensor->mutable_data();
                if (effective_batch_size == 1)
                {
                    if (config.causal)
                    {
                        LOG_TRACE("[MPI TP Q8_1] Creating causal mask for " << total_tokens << " tokens");
                        attention_utils::create_causal_mask(mask_data, total_tokens, config.window_size);
                        // Debug: print mask contents for 3x3
                        if (total_tokens <= 5 && rank == 0)
                        {
                            std::ostringstream oss;
                            oss << "[MPI TP Q8_1] Causal mask contents (first " << total_tokens << " rows):" << std::endl;
                            for (int i = 0; i < total_tokens; ++i)
                            {
                                oss << "  row " << i << ": [";
                                for (int j = 0; j < total_tokens; ++j)
                                {
                                    float val = mask_data[i * total_tokens + j];
                                    if (std::isinf(val))
                                        oss << "-inf";
                                    else
                                        oss << val;
                                    if (j < total_tokens - 1)
                                        oss << ", ";
                                }
                                oss << "]" << std::endl;
                            }
                            LOG_INFO(oss.str());
                        }
                    }
                    else
                    {
                        std::fill_n(mask_data, total_tokens * total_tokens, 0.0f);
                    }
                }
                else
                {
                    const int *seq_lens_ptr = sequence_lengths ? sequence_lengths->data() : nullptr;
                    if (config.causal)
                    {
                        attention_utils::create_batch_causal_mask(
                            mask_data, effective_batch_size, seq_len, seq_lens_ptr, config.window_size);
                    }
                    else
                    {
                        attention_utils::create_batch_padding_mask(
                            mask_data, effective_batch_size, seq_len, seq_lens_ptr, config.window_size);
                    }
                }
            }

            // Create Q8_1 attention kernel
            auto attention_kernel = std::make_unique<CPUAttentionKernelTyped<ActivationPrecision::Q8_1>>();

            // Execute native Q8_1 attention using the clean Q8_1 API
            // IMPORTANT: K/V are already broadcast to local_n_heads, so pass local_n_heads for both!
            bool local_success;
            if (effective_batch_size > 1)
            {
                // Batch path - use native Q8_1 batch API
                local_success = attention_kernel->compute_batch_q8_1(
                    Q_local.data(),
                    K_local.data(),
                    V_local.data(),
                    output_q8_local.data(),
                    effective_batch_size,
                    seq_len,
                    static_cast<int>(local_n_heads),
                    static_cast<int>(local_n_heads), // K/V already broadcast per head!
                    config.head_dim,
                    config.causal,
                    config.window_size,
                    config.workspace_scores.get(),
                    mask_tensor,
                    config.mpi_ctx.get(),
                    -1);
            }
            else
            {
                // Single sequence path - use native Q8_1 API
                local_success = attention_kernel->compute_q8_1(
                    Q_local.data(),
                    K_local.data(),
                    V_local.data(),
                    output_q8_local.data(),
                    total_tokens,
                    static_cast<int>(local_n_heads),
                    static_cast<int>(local_n_heads), // K/V already broadcast per head!
                    config.head_dim,
                    config.causal,
                    config.window_size,
                    config.workspace_scores.get(),
                    mask_tensor,
                    config.mpi_ctx.get(),
                    -1);
            }

            // Check for kernel failures across all ranks
            float local_ok_flag = local_success ? 1.0f : 0.0f;
            float global_ok_flag = 0.0f;
            config.mpi_ctx->allreduce_sum(&local_ok_flag, &global_ok_flag, 1);
            bool global_success = (global_ok_flag == static_cast<float>(world_size));
            if (!global_success)
            {
                if (rank == 0)
                {
                    LOG_ERROR("[MPI TP Q8_1] Aborting due to kernel failure on at least one rank");
                }
                return false;
            }

            // ============================================================================
            // MPI Communication: Allgather Q8_1 blocks from all ranks
            // ============================================================================
            // Each rank computed local_n_heads worth of output. We need to gather all heads
            // from all ranks into the output tensor.
            //
            // For Q8_1 output: Use byte-level allgatherv to avoid dequant→requant round trip
            // For FP32 output: Dequantize and use float allreduce (legacy path)

            if (output_is_q8_1)
            {
                // Native Q8_1 path: Use allgatherv to gather Q8_1 blocks directly
                Q8_1Block *output_blocks = output_q8->mutable_q8_1_blocks();
                if (!output_blocks)
                {
                    LOG_ERROR("[MPI TP Q8_1] Null output Q8_1 block pointer");
                    return false;
                }

                // Calculate per-rank block counts and displacements
                // Each rank contributes: total_tokens * local_n_heads * head_dim_blocks blocks
                // Block layout in output: [total_tokens, n_heads, head_dim_blocks]
                const size_t blocks_per_rank = q_local_blocks;
                const size_t bytes_per_rank = blocks_per_rank * sizeof(Q8_1Block);

                // Prepare send buffer with correct layout for allgatherv
                // Need to rearrange from [tokens, local_heads, blocks] to fit into global layout
                std::vector<Q8_1Block> send_blocks(blocks_per_rank);

                // Copy local output blocks (already in correct per-token, per-head layout)
                std::memcpy(send_blocks.data(), output_q8_local.data(), bytes_per_rank);

                // Calculate recv_counts and displacements for each rank
                std::vector<int> recv_counts(world_size);
                std::vector<int> displs(world_size);

                // Each rank sends the same number of bytes (assuming equal head distribution)
                for (int r = 0; r < world_size; ++r)
                {
                    recv_counts[r] = static_cast<int>(bytes_per_rank);
                    displs[r] = r * static_cast<int>(bytes_per_rank);
                }

                // Allocate temporary receive buffer for allgatherv
                // Layout after gather: [world_size, total_tokens, local_n_heads, head_dim_blocks]
                std::vector<Q8_1Block> recv_buffer(world_size * blocks_per_rank);

                // Perform byte-level allgatherv
                config.mpi_ctx->allgatherv_bytes(
                    send_blocks.data(), static_cast<int>(bytes_per_rank),
                    recv_buffer.data(), recv_counts.data(), displs.data());

                // Rearrange from [world_size, tokens, local_heads, blocks] to [tokens, n_heads, blocks]
                // This interleaves the heads from each rank into the correct global positions
                const size_t output_head_stride = head_dim_blocks;
                const size_t output_token_stride = config.n_heads * head_dim_blocks;

#pragma omp parallel for collapse(3) if (total_tokens * world_size * local_n_heads > 64)
                for (int t = 0; t < total_tokens; ++t)
                {
                    for (int r = 0; r < world_size; ++r)
                    {
                        for (size_t local_h = 0; local_h < local_n_heads; ++local_h)
                        {
                            // Source: recv_buffer[r * blocks_per_rank + t * local_n_heads * head_dim_blocks + local_h * head_dim_blocks]
                            size_t src_offset = r * blocks_per_rank +
                                                t * local_n_heads * head_dim_blocks +
                                                local_h * head_dim_blocks;

                            // Destination: output_blocks[t * n_heads * head_dim_blocks + (r * local_n_heads + local_h) * head_dim_blocks]
                            size_t global_head = r * local_n_heads + local_h;
                            size_t dst_offset = t * output_token_stride + global_head * output_head_stride;

                            // Copy head_dim_blocks worth of Q8_1Block
                            std::memcpy(&output_blocks[dst_offset], &recv_buffer[src_offset],
                                        head_dim_blocks * sizeof(Q8_1Block));
                        }
                    }
                }

                LOG_TRACE("[MPI TP Q8_1] Rank " << rank << ": Q8_1 allgatherv complete, "
                                                << blocks_per_rank << " blocks per rank");
            }
            else
            {
                // Legacy FP32 path: Dequantize and use float allreduce
                float *output_data = output->mutable_data();
                if (!output_data)
                {
                    LOG_ERROR("[MPI TP Q8_1] Null FP32 output data pointer");
                    return false;
                }

                // Dequantize local Q8_1 output to FP32 for allreduce
                std::vector<float> local_output_fp32(total_tokens * local_n_heads * config.head_dim);
                dequantizeQ8_1ToFP32(output_q8_local.data(), local_output_fp32.data(), q_local_blocks);

                LOG_DEBUG("[MPI TP Q8_1] Rank " << rank << ": using FP32 fallback (output tensor is FP32)");

                // Prepare send buffer for allreduce (full size, only local heads filled)
                std::vector<float> send_buffer(total_tokens * config.n_heads * config.head_dim, 0.0f);

                // Copy local heads to correct position in send buffer
                if (effective_batch_size > 1)
                {
                    // Batch path: local_output is [batch_size, seq_len, local_n_heads, head_dim]
                    const size_t seq_head_stride = static_cast<size_t>(config.n_heads) * config.head_dim;
                    const size_t batch_stride = static_cast<size_t>(seq_len) * seq_head_stride;
                    const size_t local_seq_head_stride = static_cast<size_t>(local_n_heads) * config.head_dim;
                    const size_t local_batch_stride = static_cast<size_t>(seq_len) * local_seq_head_stride;

#pragma omp parallel for collapse(3) if (effective_batch_size * seq_len * local_n_heads > 64)
                    for (int b = 0; b < effective_batch_size; ++b)
                    {
                        for (int s = 0; s < seq_len; ++s)
                        {
                            for (size_t local_h = 0; local_h < local_n_heads; ++local_h)
                            {
                                size_t global_h = start_head + local_h;
                                const float *src = local_output_fp32.data() + b * local_batch_stride + s * local_seq_head_stride + local_h * config.head_dim;
                                float *dst = send_buffer.data() + b * batch_stride + s * seq_head_stride + global_h * config.head_dim;
#pragma omp simd
                                for (int d = 0; d < config.head_dim; ++d)
                                {
                                    dst[d] = src[d];
                                }
                            }
                        }
                    }
                }
                else
                {
                    // Single-sequence path
#pragma omp parallel for collapse(2) if (total_tokens * local_n_heads > 64)
                    for (int t = 0; t < total_tokens; ++t)
                    {
                        for (size_t local_h = 0; local_h < local_n_heads; ++local_h)
                        {
                            size_t global_h = start_head + local_h;
#pragma omp simd
                            for (int d = 0; d < config.head_dim; ++d)
                            {
                                send_buffer[t * config.n_heads * config.head_dim + global_h * config.head_dim + d] =
                                    local_output_fp32[t * local_n_heads * config.head_dim + local_h * config.head_dim + d];
                            }
                        }
                    }
                }

                // Allreduce to combine results (zeros + local values = concatenation effect)
                config.mpi_ctx->allreduce_sum(
                    send_buffer.data(),
                    output_data,
                    total_tokens * config.n_heads * config.head_dim);
            }

            // Barrier to ensure all ranks complete
            config.mpi_ctx->barrier();

            return true;
        }
    } // namespace

    namespace
    {
        GQAAttentionConfig to_gqa_config(const MpiAttentionConfig &config)
        {
            GQAAttentionConfig gqa_cfg;
            gqa_cfg.n_heads = config.n_heads;
            gqa_cfg.n_kv_heads = config.n_kv_heads;
            gqa_cfg.head_dim = config.head_dim;
            gqa_cfg.causal = config.causal;
            gqa_cfg.window_size = config.window_size;
            gqa_cfg.precision = config.precision;
            gqa_cfg.mpi_ctx = config.mpi_ctx;
            gqa_cfg.mpi_strategy = MPIStrategy::None;
            gqa_cfg.verbose_logging = config.verbose_logging;
            gqa_cfg.workspace_scores = config.workspace_scores;
            gqa_cfg.workspace_qkv_buffer = config.workspace_qkv_buffer;
            gqa_cfg.workspace_context = config.workspace_context;
            gqa_cfg.workspace_mask = config.workspace_mask;
            return gqa_cfg;
        }
    } // namespace

    bool MpiAttentionOrchestrator::compute(
        TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
        const MpiAttentionConfig &config,
        int batch_size,
        const std::vector<int> *sequence_lengths)
    {
        if (!validate_inputs(Q, K, V, output, config))
        {
            return false;
        }

        GQAAttentionConfig gqa_cfg = to_gqa_config(config);
        return GQAAttention::compute(Q, K, V, output, gqa_cfg, batch_size, sequence_lengths);
    }

    bool MpiAttentionOrchestrator::compute_batch(
        TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
        const std::vector<int> &actual_lengths,
        int batch_size, int seq_len,
        const MpiAttentionConfig &config)
    {
        if (!validate_inputs(Q, K, V, output, config))
        {
            return false;
        }

        if (static_cast<int>(actual_lengths.size()) != batch_size)
        {
            LOG_ERROR("[MpiAttentionOrchestrator] compute_batch: actual_lengths size ("
                      << actual_lengths.size() << ") != batch_size (" << batch_size << ")");
            return false;
        }

        const auto &q_shape = Q->shape();
        const int total_seq_len = batch_size * seq_len;
        if (static_cast<int>(q_shape[0]) != total_seq_len)
        {
            LOG_ERROR("[MpiAttentionOrchestrator] compute_batch: Q shape[0] ("
                      << q_shape[0] << ") != batch_size * seq_len (" << total_seq_len << ")");
            return false;
        }

        GQAAttentionConfig gqa_cfg = to_gqa_config(config);
        return GQAAttention::compute_batch(Q, K, V, output, actual_lengths, batch_size, seq_len, gqa_cfg);
    }

    bool MpiAttentionOrchestrator::compute_mpi(
        TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
        const MpiAttentionConfig &config,
        int batch_size,
        const std::vector<int> *sequence_lengths)
    {
        // Fast path: No MPI or single-rank execution
        if (!config.mpi_ctx || config.mpi_ctx->world_size() == 1 || config.mpi_strategy == MPIStrategy::None)
        {
            return compute(Q, K, V, output, config, batch_size, sequence_lengths);
        }

        // Dispatch based on MPI strategy
        switch (config.mpi_strategy)
        {
        case MPIStrategy::TensorParallel:
            return compute_tensor_parallel(Q, K, V, output, config, batch_size, sequence_lengths);

        case MPIStrategy::SequenceParallel:
            // TODO: Implement sequence-parallel attention (Phase 6)
            LOG_ERROR("[MpiAttentionOrchestrator] SequenceParallel attention not yet implemented");
            return false;

        case MPIStrategy::PipelineParallel:
            // Pipeline-parallel doesn't change attention (distributes layers instead)
            return compute(Q, K, V, output, config, batch_size, sequence_lengths);

        case MPIStrategy::Hybrid:
            // TODO: Implement hybrid strategy (Phase 6)
            LOG_ERROR("[MpiAttentionOrchestrator] Hybrid strategy not yet implemented");
            return false;

        default:
            LOG_ERROR("[MpiAttentionOrchestrator] Unknown MPI strategy: " << static_cast<int>(config.mpi_strategy));
            return false;
        }
    }

    bool MpiAttentionOrchestrator::compute_tensor_parallel(
        TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
        const MpiAttentionConfig &config,
        int batch_size,
        const std::vector<int> *sequence_lengths)
    {
        LOG_TRACE("[MpiAttentionOrchestrator::compute_tensor_parallel] precision="
                  << static_cast<int>(config.precision)
                  << ", Q type=" << (dynamic_cast<Q8_1Tensor *>(Q) ? "Q8_1" : "FP32")
                  << ", output type=" << (dynamic_cast<Q8_1Tensor *>(output) ? "Q8_1" : "FP32"));

        // Validate MPI context
        if (!config.mpi_ctx)
        {
            LOG_ERROR("[MpiAttentionOrchestrator] Tensor-parallel attention requires MPI context");
            return false;
        }

        int rank = config.mpi_ctx->rank();
        int world_size = config.mpi_ctx->world_size();

        // 1. Validate inputs
        if (!validate_inputs(Q, K, V, output, config))
        {
            return false;
        }

        // Validate divisibility for tensor parallelism
        if (config.n_heads % world_size != 0)
        {
            LOG_ERROR("[MpiAttentionOrchestrator] Tensor-parallel requires n_heads (" << config.n_heads
                                                                                      << ") divisible by world_size (" << world_size << ")");
            return false;
        }

        // Infer dimensions from Q shape and batch_size
        const auto &q_shape = Q->shape();
        int total_tokens = static_cast<int>(q_shape[0]);
        int effective_batch_size = (batch_size > 0) ? batch_size : 1;
        int seq_len = total_tokens / effective_batch_size;

        if (total_tokens % effective_batch_size != 0)
        {
            LOG_ERROR("[MpiAttentionOrchestrator] total_tokens (" << total_tokens
                                                                  << ") not divisible by batch_size (" << effective_batch_size << ")");
            return false;
        }

        LOG_TRACE("[MPI TP] Batch-aware attention: total_tokens=" << total_tokens
                                                                  << " batch_size=" << effective_batch_size << " seq_len_per_batch=" << seq_len);

        int padded_seq_len = seq_len;

        // Distribute attention heads across ranks
        auto [start_head, local_n_heads] = config.mpi_ctx->get_local_slice(static_cast<size_t>(config.n_heads));

        // ============================================================================
        // Native Precision Tensor-Parallel Path
        // ============================================================================
        // Check if we can use a native precision path which avoids unnecessary
        // format conversions. The native path:
        //   1. Operates on tensors in their native format (Q8_1, BF16, FP16, FP32)
        //   2. Uses precision-appropriate MPI communication (byte-level for non-FP32)
        //   3. Keeps output in native format when output tensor matches input precision
        //
        // This eliminates dequant→allreduce→requant round trips that add noise.
        //
        ActivationPrecision detected_precision;
        if (shouldUseNativeTensorParallel(Q, K, V, output, config, detected_precision))
        {
            LOG_TRACE("[MPI TP] Using native " << static_cast<int>(detected_precision)
                                               << " tensor-parallel path");

            switch (detected_precision)
            {
            case ActivationPrecision::Q8_1:
                return compute_tensor_parallel_q8_1_native(
                    Q, K, V, output,
                    config, rank, world_size,
                    total_tokens, effective_batch_size, seq_len, padded_seq_len,
                    start_head, local_n_heads, sequence_lengths);

            case ActivationPrecision::BF16:
            case ActivationPrecision::FP16:
                // TODO: Implement native BF16/FP16 tensor-parallel paths
                // For now, fall through to FP32 path (data() handles conversion)
                LOG_DEBUG("[MPI TP] BF16/FP16 native path not yet implemented, using FP32 fallback");
                break;

            case ActivationPrecision::FP32:
                // FP32 uses the standard path below
                break;
            }
        }

        // ============================================================================
        // FP32 Fallback Path (also handles BF16/FP16 via dequantization)
        // ============================================================================
        // This path converts inputs to FP32 via data() calls, then uses FP32 kernels.
        // Works for all precision types but may not be optimal for non-FP32.
        LOG_DEBUG("[MPI TP] Using FP32 fallback tensor-parallel path");

        // Get tensor data pointers
        const float *Q_data = Q->data();
        const float *K_data = K->data();
        const float *V_data = V->data();
        float *output_data = output->mutable_data();

        if (!Q_data || !K_data || !V_data || !output_data)
        {
            LOG_ERROR("[MpiAttentionOrchestrator] Null pointer in attention tensors");
            return false;
        }

        // 2. Broadcast K/V heads to match Q heads (if needed)
        std::vector<float> K_broadcast, V_broadcast;
        const float *K_expanded = K_data;
        const float *V_expanded = V_data;

        broadcast_kv_heads_if_needed(
            K_data, V_data, K_broadcast, V_broadcast,
            total_tokens, config.n_heads, config.n_kv_heads, config.head_dim);

        if (!K_broadcast.empty())
        {
            K_expanded = K_broadcast.data();
            V_expanded = V_broadcast.data();
        }

        // Prepare contiguous rank-local Q/K/V buffers for kernel invocation
        const size_t local_tensor_elements = static_cast<size_t>(total_tokens) * local_n_heads * config.head_dim;
        std::vector<float> Q_local(local_tensor_elements);
        std::vector<float> K_local(local_tensor_elements);
        std::vector<float> V_local(local_tensor_elements);
        std::vector<float> local_output(local_tensor_elements, 0.0f);

        // For batch path, we need to preserve [batch_size, seq_len, n_heads, head_dim] layout
        // For single-sequence path, total_tokens = seq_len, so batch structure doesn't matter
        auto copy_head_slice = [&](const float *src, float *dst)
        {
            if (effective_batch_size > 1)
            {
                // Batch path: src is [batch_size, seq_len, n_heads, head_dim]
                // dst should be [batch_size, seq_len, local_n_heads, head_dim]
                const size_t seq_head_stride = static_cast<size_t>(config.n_heads) * config.head_dim;
                const size_t batch_stride = static_cast<size_t>(seq_len) * seq_head_stride;
                const size_t local_seq_head_stride = static_cast<size_t>(local_n_heads) * config.head_dim;
                const size_t local_batch_stride = static_cast<size_t>(seq_len) * local_seq_head_stride;

                for (int b = 0; b < effective_batch_size; ++b)
                {
                    const float *batch_src = src + b * batch_stride;
                    float *batch_dst = dst + b * local_batch_stride;

                    for (int s = 0; s < seq_len; ++s)
                    {
                        const float *seq_src = batch_src + static_cast<size_t>(s) * seq_head_stride;
                        float *seq_dst = batch_dst + static_cast<size_t>(s) * local_seq_head_stride;

                        for (size_t local_h = 0; local_h < local_n_heads; ++local_h)
                        {
                            const size_t global_h = start_head + local_h;
                            const float *src_head = seq_src + global_h * config.head_dim;
                            float *dst_head = seq_dst + local_h * config.head_dim;
                            std::memcpy(dst_head, src_head, static_cast<size_t>(config.head_dim) * sizeof(float));
                        }
                    }
                }
            }
            else
            {
                // Single-sequence path: src is [seq_len, n_heads, head_dim]
                for (int t = 0; t < total_tokens; ++t)
                {
                    const float *src_row = src + static_cast<size_t>(t) * config.n_heads * config.head_dim;
                    float *dst_row = dst + static_cast<size_t>(t) * local_n_heads * config.head_dim;
                    for (size_t local_h = 0; local_h < local_n_heads; ++local_h)
                    {
                        const size_t global_h = start_head + local_h;
                        const float *src_head = src_row + global_h * config.head_dim;
                        float *dst_head = dst_row + local_h * config.head_dim;
                        std::memcpy(dst_head, src_head, static_cast<size_t>(config.head_dim) * sizeof(float));
                    }
                }
            }
        };

        copy_head_slice(Q_data, Q_local.data());
        copy_head_slice(K_expanded, K_local.data());
        copy_head_slice(V_expanded, V_local.data());

        // NOTE: For tensor-parallel attention, we extract raw float data from Q/K/V
        // and create local float buffers. This means Q8_1 tensors are dequantized
        // via data() calls above. The kernel type still matters for internal
        // operations like softmax precision.
        //
        // Create kernel based on config.precision, not output tensor type
        auto attention_kernel = createAttentionKernelForPrecision(config.precision);
        if (!attention_kernel)
        {
            LOG_ERROR("[MpiAttentionOrchestrator] Failed to create attention kernel for tensor-parallel path (precision="
                      << static_cast<int>(config.precision) << ")");
            return false;
        }

        const bool needs_mask = config.causal || (sequence_lengths && !sequence_lengths->empty());
        TensorBase *mask_tensor = config.workspace_mask.get();
        if (needs_mask && !mask_tensor)
        {
            LOG_ERROR("[MpiAttentionOrchestrator] workspace_mask is required for causal/padded tensor-parallel attention");
            return false;
        }

        if (needs_mask)
        {
            float *mask_data = mask_tensor->mutable_data();
            if (effective_batch_size == 1)
            {
                // Single sequence: respect causal flag (fixes E2E parity tests)
                if (config.causal)
                {
                    attention_utils::create_causal_mask(mask_data, total_tokens, config.window_size);
                }
                else
                {
                    // Non-causal single sequence: no mask needed (all tokens can attend to all tokens)
                    // Fill with zeros (no masking)
                    std::fill_n(mask_data, total_tokens * total_tokens, 0.0f);
                }
            }
            else
            {
                const int *seq_lens_ptr = sequence_lengths ? sequence_lengths->data() : nullptr;

                // Use appropriate mask based on causal flag
                if (config.causal)
                {
                    // Causal attention: mask future tokens + padding
                    attention_utils::create_batch_causal_mask(
                        mask_data, effective_batch_size, padded_seq_len, seq_lens_ptr, config.window_size);
                }
                else
                {
                    // Non-causal attention: mask only padding tokens (bi-directional)
                    attention_utils::create_batch_padding_mask(
                        mask_data, effective_batch_size, padded_seq_len, seq_lens_ptr, config.window_size);
                }
            }
        }

        // Choose correct kernel path based on batch_size
        bool local_success;
        if (effective_batch_size > 1)
        {
            // Batch path: Call compute_batch with separate batch_size and seq_len
            // Layout: Q_local/K_local/V_local are [batch_size, seq_len, local_n_heads, head_dim] (flattened)
            local_success = attention_kernel->compute_batch(
                Q_local.data(),
                K_local.data(),
                V_local.data(),
                local_output.data(),
                effective_batch_size,
                seq_len,
                static_cast<int>(local_n_heads),
                static_cast<int>(local_n_heads), // K/V already broadcast per head
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
            local_success = attention_kernel->compute(
                Q_local.data(),
                K_local.data(),
                V_local.data(),
                local_output.data(),
                total_tokens,
                static_cast<int>(local_n_heads),
                static_cast<int>(local_n_heads), // K/V already broadcast per head
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

        float local_ok_flag = local_success ? 1.0f : 0.0f;
        float global_ok_flag = 0.0f;
        config.mpi_ctx->allreduce_sum(&local_ok_flag, &global_ok_flag, 1);
        bool global_success = (global_ok_flag == static_cast<float>(world_size));
        if (!global_success)
        {
            if (rank == 0)
            {
                LOG_ERROR("[MpiAttentionOrchestrator] Aborting tensor-parallel attention due to kernel failure on at least one rank");
            }
            return false;
        }

        // Debug: Check FP32 local output stats
        if (rank == 0 && Logger::getInstance().shouldLog(LogLevel::VERBOSITY_DEBUG))
        {
            float min_val = local_output[0], max_val = local_output[0];
            double sum = 0.0;
            for (size_t i = 0; i < local_output.size(); ++i)
            {
                min_val = std::min(min_val, local_output[i]);
                max_val = std::max(max_val, local_output[i]);
                sum += local_output[i];
            }
            LOG_DEBUG("[MPI TP FP32] Rank " << rank << ": local_output min=" << min_val
                                            << " max=" << max_val << " mean=" << (sum / local_output.size())
                                            << " first10=[" << local_output[0] << "," << local_output[1]
                                            << "," << local_output[2] << "," << local_output[3]
                                            << "," << local_output[4] << "," << local_output[5]
                                            << "," << local_output[6] << "," << local_output[7]
                                            << "," << local_output[8] << "," << local_output[9] << "]");
        }

        LOG_DEBUG("[MPI TP] Rank " << rank << ": local_output[0]=" << local_output[0]);

        // 8. Allreduce: Sum local outputs from all ranks
        std::memset(output_data, 0, total_tokens * config.n_heads * config.head_dim * sizeof(float));

        // Create temporary buffer for allreduce (each rank's contribution)
        std::vector<float> send_buffer(total_tokens * config.n_heads * config.head_dim, 0.0f);

        // Copy local heads to correct position in send buffer
        // Must preserve batch structure if effective_batch_size > 1
        if (effective_batch_size > 1)
        {
            // Batch path: local_output is [batch_size, seq_len, local_n_heads, head_dim]
            // send_buffer should be [batch_size, seq_len, n_heads, head_dim] with only start_head..start_head+local_n_heads filled
            const size_t seq_head_stride = static_cast<size_t>(config.n_heads) * config.head_dim;
            const size_t batch_stride = static_cast<size_t>(seq_len) * seq_head_stride;
            const size_t local_seq_head_stride = static_cast<size_t>(local_n_heads) * config.head_dim;
            const size_t local_batch_stride = static_cast<size_t>(seq_len) * local_seq_head_stride;

#pragma omp parallel for collapse(3) if (effective_batch_size * seq_len * local_n_heads > 64)
            for (int b = 0; b < effective_batch_size; ++b)
            {
                for (int s = 0; s < seq_len; ++s)
                {
                    for (size_t local_h = 0; local_h < local_n_heads; ++local_h)
                    {
                        size_t global_h = start_head + local_h;
                        const float *src = local_output.data() + b * local_batch_stride + s * local_seq_head_stride + local_h * config.head_dim;
                        float *dst = send_buffer.data() + b * batch_stride + s * seq_head_stride + global_h * config.head_dim;
#pragma omp simd
                        for (int d = 0; d < config.head_dim; ++d)
                        {
                            dst[d] = src[d];
                        }
                    }
                }
            }
        }
        else
        {
            // Single-sequence path: local_output is [seq_len, local_n_heads, head_dim]
#pragma omp parallel for collapse(2) if (total_tokens * local_n_heads > 64)
            for (int t = 0; t < total_tokens; ++t)
            {
                for (size_t local_h = 0; local_h < local_n_heads; ++local_h)
                {
                    size_t global_h = start_head + local_h;
#pragma omp simd
                    for (int d = 0; d < config.head_dim; ++d)
                    {
                        send_buffer[t * config.n_heads * config.head_dim + global_h * config.head_dim + d] =
                            local_output[t * local_n_heads * config.head_dim + local_h * config.head_dim + d];
                    }
                }
            }
        }

        LOG_DEBUG("[MPI TP] Rank " << rank << ": send_buffer[0]=" << send_buffer[0]
                                   << " (rank computes heads " << start_head << "-" << (start_head + local_n_heads - 1) << ")");

        // Bounds-safe debug logging (only construct string if logging enabled)
        if (Logger::getInstance().shouldLog(LogLevel::VERBOSITY_DEBUG))
        {
            const size_t buffer_size = send_buffer.size();
            if (start_head * config.head_dim < buffer_size)
            {
                LOG_DEBUG("[MPI TP] Rank " << rank << ": send_buffer[" << (start_head * config.head_dim) << "]="
                                           << send_buffer[start_head * config.head_dim] << " (first element of head " << start_head << ")");
            }

            std::ostringstream debug_msg;
            debug_msg << "[MPI TP] Rank " << rank << " BEFORE allreduce (buffer_size=" << buffer_size << "):";
            if (buffer_size > 100)
                debug_msg << " send_buffer[100]=" << send_buffer[100];
            if (buffer_size > 1000)
                debug_msg << " send_buffer[1000]=" << send_buffer[1000];
            if (buffer_size > 8000)
                debug_msg << " send_buffer[8000]=" << send_buffer[8000];
            LOG_DEBUG(debug_msg.str());
        }

        // Stage send_buffer and output to host if needed (for GPU tensors)
        // For CPU tensors (current default), this is a no-op
        // When GPU backends enabled, this will:
        //   1. Copy send_buffer from GPU to host (if needed)
        //   2. Perform MPI Allreduce on CPU
        //   3. Copy result back to GPU output tensor

        // Check if we need GPU staging (output tensor on device)
        bool requires_staging = MPIStager::requiresStaging(output);

        // For now, send_buffer is always std::vector<float> (CPU memory)
        // But output tensor might be on GPU in the future

        float *mpi_output_buffer = output_data; // Default: use output tensor directly
        std::vector<float> host_output_staging; // Only allocated if GPU staging needed

        if (requires_staging)
        {
            // GPU tensor case: allocate staging buffer
            host_output_staging.resize(total_tokens * config.n_heads * config.head_dim);
            mpi_output_buffer = host_output_staging.data();

            LOG_DEBUG("[MPI TP] Rank " << rank << ": GPU staging required, using host buffer");
        }

        // Allreduce: Sum contributions from all ranks
        // Use total_tokens (not seq_len) to cover all batches
        config.mpi_ctx->allreduce_sum(
            send_buffer.data(),
            mpi_output_buffer,
            total_tokens * config.n_heads * config.head_dim);

        // Stage result back to GPU if needed
        if (requires_staging)
        {
            MPIStager::toDevice(host_output_staging, output);
            LOG_DEBUG("[MPI TP] Rank " << rank << ": Staged result back to GPU");
        }

        // Bounds-safe debug logging after allreduce (only construct string if logging enabled)
        if (Logger::getInstance().shouldLog(LogLevel::VERBOSITY_DEBUG))
        {
            const size_t output_size = total_tokens * config.n_heads * config.head_dim;
            std::ostringstream debug_msg_after;
            debug_msg_after << "[MPI TP] Rank " << rank << " AFTER allreduce (output_size=" << output_size << "):";

            // Log from the buffer that contains the result
            const float *result_buffer = requires_staging ? host_output_staging.data() : output_data;
            debug_msg_after << " output[0]=" << result_buffer[0];
            if (output_size > 100)
                debug_msg_after << " output[100]=" << result_buffer[100];
            if (output_size > 1000)
                debug_msg_after << " output[1000]=" << result_buffer[1000];
            if (output_size > 8000)
                debug_msg_after << " output[8000]=" << result_buffer[8000];
            if (output_size > 0)
                debug_msg_after << " output[" << (output_size - 1) << "]=" << result_buffer[output_size - 1];
            LOG_DEBUG(debug_msg_after.str());
        }

        // 10. Barrier to ensure all ranks complete
        config.mpi_ctx->barrier();

        return true;
    }

    // ============================================================================
    // Private Helper Functions (Testable Atomic Operations)
    // ============================================================================

    bool MpiAttentionOrchestrator::validate_inputs(
        const TensorBase *Q, const TensorBase *K, const TensorBase *V,
        const TensorBase *output, const MpiAttentionConfig &config)
    {
        // Check for null pointers
        if (!Q || !K || !V || !output)
        {
            LOG_ERROR("[MpiAttentionOrchestrator] validate_inputs: null pointer detected");
            return false;
        }

        // Validate head configuration
        if (config.n_heads % config.n_kv_heads != 0)
        {
            LOG_ERROR("[MpiAttentionOrchestrator] validate_inputs: n_heads (" << config.n_heads
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
            LOG_ERROR("[MpiAttentionOrchestrator] validate_inputs: Q must be 2D, got " << q_shape.size() << "D");
            return false;
        }

        // Validate Q dimensions
        int expected_q_dim = config.n_heads * config.head_dim;
        if (q_shape[1] != (size_t)expected_q_dim)
        {
            LOG_ERROR("[MpiAttentionOrchestrator] validate_inputs: Q dimension mismatch. Expected "
                      << expected_q_dim << ", got " << q_shape[1]);
            return false;
        }

        // Validate K/V dimensions
        int expected_kv_dim = config.n_kv_heads * config.head_dim;
        if (k_shape.size() != 2 || k_shape[1] != (size_t)expected_kv_dim)
        {
            LOG_ERROR("[MpiAttentionOrchestrator] validate_inputs: K dimension mismatch. Expected [*, "
                      << expected_kv_dim << "], got [" << k_shape[0] << ", " << k_shape[1] << "]");
            return false;
        }

        if (v_shape.size() != 2 || v_shape[1] != (size_t)expected_kv_dim)
        {
            LOG_ERROR("[MpiAttentionOrchestrator] validate_inputs: V dimension mismatch. Expected [*, "
                      << expected_kv_dim << "], got [" << v_shape[0] << ", " << v_shape[1] << "]");
            return false;
        }

        // Validate sequence length consistency
        if (q_shape[0] != k_shape[0] || q_shape[0] != v_shape[0])
        {
            LOG_ERROR("[MpiAttentionOrchestrator] validate_inputs: Sequence length mismatch. Q="
                      << q_shape[0] << ", K=" << k_shape[0] << ", V=" << v_shape[0]);
            return false;
        }

        // Validate output dimensions
        if (out_shape != q_shape)
        {
            LOG_ERROR("[MpiAttentionOrchestrator] validate_inputs: Output shape mismatch. Expected "
                      << "[" << q_shape[0] << ", " << q_shape[1] << "], got ["
                      << out_shape[0] << ", " << out_shape[1] << "]");
            return false;
        }

        return true;
    }

    void MpiAttentionOrchestrator::broadcast_kv_heads_if_needed(
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

    void MpiAttentionOrchestrator::extract_head_data(
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

    bool MpiAttentionOrchestrator::compute_attention_scores(
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

    void MpiAttentionOrchestrator::scale_scores_inplace(
        float *scores, int size, int head_dim)
    {
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

#pragma omp parallel for if (size > 8192)
        for (int i = 0; i < size; ++i)
        {
            scores[i] *= scale;
        }
    }

    void MpiAttentionOrchestrator::apply_attention_mask(
        float *scores, int seq_len, int batch_size,
        const int *seq_lengths, bool causal, int window_size,
        const MpiAttentionConfig &config)
    {
        if (!causal && !seq_lengths)
        {
            // No masking needed
            return;
        }

        // Use workspace mask buffer
        if (!config.workspace_mask)
        {
            LOG_ERROR("[MpiAttentionOrchestrator] apply_attention_mask: workspace_mask not provided");
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
            // Use appropriate mask based on causal flag
            if (causal)
            {
                // Causal attention: mask future tokens + padding
                attention_utils::create_batch_causal_mask(
                    mask, batch_size, seq_len, seq_lengths, window_size);
            }
            else
            {
                // Non-causal attention: mask only padding tokens (bi-directional)
                attention_utils::create_batch_padding_mask(
                    mask, batch_size, seq_len, seq_lengths, window_size);
            }

            // Apply to all heads (mask is shared across heads within a batch)
            attention_utils::apply_attention_mask(scores, mask, batch_size * seq_len, seq_len);
        }
    }

    void MpiAttentionOrchestrator::apply_softmax(
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

    bool MpiAttentionOrchestrator::compute_context_from_scores(
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

    void MpiAttentionOrchestrator::write_context_to_output(
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
