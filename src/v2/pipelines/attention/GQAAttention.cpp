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
#include "../../kernels/cpu/primitives/SoftmaxPrimitives.h"
#include "../../kernels/cpu/FP32GemmKernel.h"
#include "../../kernels/cpu/BF16GemmKernel.h"
#include <cstring>
#include <cmath>
#include <sstream>
#include <omp.h>

namespace llaminar2
{

    bool GQAAttention::compute(
        TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
        const GQAAttentionConfig &config,
        int batch_size,
        const std::vector<int> *sequence_lengths)
    {
        // 1. Validate inputs
        if (!validate_inputs(Q, K, V, output, config))
        {
            return false;
        }

        // Infer seq_len from Q shape: [seq_len, n_heads * head_dim]
        const auto &q_shape = Q->shape();
        int seq_len = static_cast<int>(q_shape[0]);

        // Get tensor data pointers
        const float *Q_data = Q->data();
        const float *K_data = K->data();
        const float *V_data = V->data();
        float *output_data = output->mutable_data();

        // 2. Broadcast K/V heads to match Q heads (if needed)
        std::vector<float> K_broadcast, V_broadcast;
        const float *K_expanded = K_data;
        const float *V_expanded = V_data;

        broadcast_kv_heads_if_needed(
            K_data, V_data, K_broadcast, V_broadcast,
            seq_len, config.n_heads, config.n_kv_heads, config.head_dim);

        if (!K_broadcast.empty())
        {
            K_expanded = K_broadcast.data();
            V_expanded = V_broadcast.data();
        }

        // Validate workspace buffers
        if (!config.workspace_scores || !config.workspace_qkv_buffer || !config.workspace_context)
        {
            LOG_ERROR("[GQAAttention] compute: workspace buffers not provided");
            return false;
        }

        float *scores = config.workspace_scores->mutable_data();

        // 3. Compute attention scores per head: Q @ K^T
        // Parallelize over heads (independent operations)
#pragma omp parallel if (config.n_heads > 1)
        {
            // Thread-local extraction buffers (from workspace, partitioned per thread)
            const int thread_id = omp_get_thread_num();
            const size_t buf_offset = thread_id * seq_len * config.head_dim * 3;
            float *Q_h = config.workspace_qkv_buffer->mutable_data() + buf_offset;
            float *K_h = Q_h + seq_len * config.head_dim;

#pragma omp for
            for (int h = 0; h < config.n_heads; ++h)
            {
                float *scores_h = scores + h * seq_len * seq_len;

                // Extract Q and K for this head
                extract_head_data(Q_data, Q_h, seq_len, config.head_dim, config.n_heads, h, 0);
                extract_head_data(K_expanded, K_h, seq_len, config.head_dim, config.n_heads, h, 0);

                LOG_DEBUG("[GQAAttention] Head " << h << ": Q_h[0]=" << Q_h[0] << " K_h[0]=" << K_h[0]);

                // GEMM: scores[h] = Q_h @ K_h^T
                if (!compute_attention_scores(Q_h, K_h, scores_h, seq_len, config.head_dim, config.precision))
                {
                    LOG_ERROR("[GQAAttention] compute: Q·K^T GEMM failed for head " << h);
                }

                LOG_DEBUG("[GQAAttention] Head " << h << ": scores[0]=" << scores_h[0]);
            }
        }

        // 4. Scale scores by 1/sqrt(head_dim)
        LOG_DEBUG("[GQAAttention] Scaling scores by 1/sqrt(" << config.head_dim << ")");
        scale_scores_inplace(scores, config.n_heads * seq_len * seq_len, config.head_dim);
        LOG_DEBUG("[GQAAttention] After scaling: scores[0]=" << scores[0]);

        // 5. Apply causal mask (if enabled)
        if (config.causal || sequence_lengths)
        {
            LOG_DEBUG("[GQAAttention] Applying attention mask (batch_size=" << batch_size << ")");

            // Use workspace mask buffer
            if (!config.workspace_mask)
            {
                LOG_ERROR("[GQAAttention] compute: workspace_mask buffer not provided");
                return false;
            }
            float *mask = config.workspace_mask->mutable_data();

            if (batch_size == 1)
            {
                // Single sequence: standard causal mask
                attention_utils::create_causal_mask(mask, seq_len, config.window_size);
            }
            else
            {
                // Batched sequences: block-diagonal causal mask with padding
                int padded_seq_len = seq_len / batch_size;
                const int *seq_lens_ptr = sequence_lengths ? sequence_lengths->data() : nullptr;
                attention_utils::create_batch_causal_mask(
                    mask, batch_size, padded_seq_len, seq_lens_ptr, config.window_size);
            }

            // Apply mask to each head separately
#pragma omp parallel for if (config.n_heads > 1)
            for (int h = 0; h < config.n_heads; ++h)
            {
                float *scores_h = scores + h * seq_len * seq_len;
                attention_utils::apply_attention_mask(scores_h, mask, seq_len, seq_len);
            }

            LOG_DEBUG("[GQAAttention] After masking: scores[0]=" << scores[0]);
        }

        // 6. Apply softmax
        apply_softmax(scores, config.n_heads * seq_len, seq_len);
        LOG_DEBUG("[GQAAttention] After softmax: scores[0]=" << scores[0]);

        // 7. Compute context: scores @ V
        std::memset(output_data, 0, seq_len * config.n_heads * config.head_dim * sizeof(float));

#pragma omp parallel if (config.n_heads > 1)
        {
            // Thread-local buffers
            const int thread_id = omp_get_thread_num();
            const size_t buf_offset = thread_id * seq_len * config.head_dim * 3;
            float *V_h = config.workspace_qkv_buffer->mutable_data() + buf_offset + 2 * seq_len * config.head_dim;

            const size_t ctx_offset = thread_id * seq_len * config.head_dim;
            float *context_h = config.workspace_context->mutable_data() + ctx_offset;

#pragma omp for
            for (int h = 0; h < config.n_heads; ++h)
            {
                const float *scores_h = scores + h * seq_len * seq_len;

                // Extract V for this head
                extract_head_data(V_expanded, V_h, seq_len, config.head_dim, config.n_heads, h, 0);

                // GEMM: context_h = scores[h] @ V_h
                if (!compute_context_from_scores(scores_h, V_h, context_h, seq_len, config.head_dim, config.precision))
                {
                    LOG_ERROR("[GQAAttention] compute: scores·V GEMM failed for head " << h);
                }

                // Write context back to strided output
                write_context_to_output(context_h, output_data, seq_len, config.head_dim, config.n_heads, h, 0);
            }
        }

        return true;
    }

    bool GQAAttention::compute_batch(
        TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
        const std::vector<int> &actual_lengths,
        int batch_size, int seq_len,
        const GQAAttentionConfig &config)
    {
        // 1. Validate inputs
        if (!validate_inputs(Q, K, V, output, config))
        {
            return false;
        }

        if (static_cast<int>(actual_lengths.size()) != batch_size)
        {
            LOG_ERROR("[GQAAttention] compute_batch: actual_lengths size ("
                      << actual_lengths.size() << ") != batch_size (" << batch_size << ")");
            return false;
        }

        // Validate tensor shapes
        const auto &q_shape = Q->shape();
        const int total_seq_len = batch_size * seq_len;
        if (static_cast<int>(q_shape[0]) != total_seq_len)
        {
            LOG_ERROR("[GQAAttention] compute_batch: Q shape[0] ("
                      << q_shape[0] << ") != batch_size * seq_len (" << total_seq_len << ")");
            return false;
        }

        // Get tensor data pointers
        const float *Q_data = Q->data();
        const float *K_data = K->data();
        const float *V_data = V->data();
        float *output_data = output->mutable_data();

        // 2. Broadcast K/V heads to match Q heads (if needed)
        std::vector<float> K_broadcast, V_broadcast;
        const float *K_expanded = K_data;
        const float *V_expanded = V_data;

        broadcast_kv_heads_if_needed(
            K_data, V_data, K_broadcast, V_broadcast,
            total_seq_len, config.n_heads, config.n_kv_heads, config.head_dim);

        if (!K_broadcast.empty())
        {
            K_expanded = K_broadcast.data();
            V_expanded = V_broadcast.data();
        }

        // Validate workspace buffers
        if (!config.workspace_scores || !config.workspace_qkv_buffer ||
            !config.workspace_context || !config.workspace_mask)
        {
            LOG_ERROR("[GQAAttention] compute_batch: workspace buffers not provided");
            return false;
        }

        float *scores = config.workspace_scores->mutable_data();

        // Create combined causal + padding mask [batch_size, seq_len, seq_len]
        float *combined_mask = config.workspace_mask->mutable_data();
        attention_utils::create_combined_batch_mask(
            combined_mask, batch_size, seq_len,
            actual_lengths.data(), config.causal, config.window_size);

        // 3. Compute attention scores per batch and head: Q @ K^T
        // Parallelize over batch×heads
#pragma omp parallel if (batch_size * config.n_heads > 1)
        {
            // Thread-local extraction buffers
            const int thread_id = omp_get_thread_num();
            const size_t buf_offset = thread_id * seq_len * config.head_dim * 2;
            float *Q_bh = config.workspace_qkv_buffer->mutable_data() + buf_offset;
            float *K_bh = Q_bh + seq_len * config.head_dim;

#pragma omp for collapse(2)
            for (int b = 0; b < batch_size; ++b)
            {
                for (int h = 0; h < config.n_heads; ++h)
                {
                    const int batch_offset = b * seq_len;
                    float *scores_bh = scores + (b * config.n_heads + h) * seq_len * seq_len;

                    // Extract Q and K for this batch and head
                    extract_head_data(Q_data, Q_bh, seq_len, config.head_dim, config.n_heads, h, batch_offset);
                    extract_head_data(K_expanded, K_bh, seq_len, config.head_dim, config.n_heads, h, batch_offset);

                    // GEMM: scores[b,h] = Q_bh @ K_bh^T
                    if (!compute_attention_scores(Q_bh, K_bh, scores_bh, seq_len, config.head_dim, config.precision))
                    {
                        LOG_ERROR("[GQAAttention] compute_batch: Q·K^T GEMM failed for batch " << b << " head " << h);
                    }
                }
            }
        }

        // 4. Scale scores by 1/sqrt(head_dim)
        scale_scores_inplace(scores, batch_size * config.n_heads * seq_len * seq_len, config.head_dim);

        // 5. Apply combined causal + padding mask
        // Broadcast mask across heads: mask[b, seq_len, seq_len] -> all heads for batch b
#pragma omp parallel for collapse(2) if (batch_size * config.n_heads > 1)
        for (int b = 0; b < batch_size; ++b)
        {
            for (int h = 0; h < config.n_heads; ++h)
            {
                const float *mask_b = combined_mask + b * seq_len * seq_len;
                float *scores_bh = scores + (b * config.n_heads + h) * seq_len * seq_len;
                attention_utils::apply_attention_mask(scores_bh, mask_b, seq_len, seq_len);
            }
        }

        // 6. Apply softmax
        apply_softmax(scores, batch_size * config.n_heads * seq_len, seq_len);

        // 7. Compute context: scores @ V
        std::memset(output_data, 0, total_seq_len * config.n_heads * config.head_dim * sizeof(float));

#pragma omp parallel if (batch_size * config.n_heads > 1)
        {
            // Thread-local buffers
            const int thread_id = omp_get_thread_num();
            const size_t buf_offset = thread_id * seq_len * config.head_dim;
            float *V_bh = config.workspace_qkv_buffer->mutable_data() + buf_offset;

            const size_t ctx_offset = thread_id * seq_len * config.head_dim;
            float *context_bh = config.workspace_context->mutable_data() + ctx_offset;

#pragma omp for collapse(2)
            for (int b = 0; b < batch_size; ++b)
            {
                for (int h = 0; h < config.n_heads; ++h)
                {
                    const int batch_offset = b * seq_len;
                    const float *scores_bh = scores + (b * config.n_heads + h) * seq_len * seq_len;

                    // Extract V for this batch and head
                    extract_head_data(V_expanded, V_bh, seq_len, config.head_dim, config.n_heads, h, batch_offset);

                    // GEMM: context_bh = scores[b,h] @ V_bh
                    if (!compute_context_from_scores(scores_bh, V_bh, context_bh, seq_len, config.head_dim, config.precision))
                    {
                        LOG_ERROR("[GQAAttention] compute_batch: scores·V GEMM failed for batch " << b << " head " << h);
                    }

                    // Write context back to strided output
                    write_context_to_output(context_bh, output_data, seq_len, config.head_dim, config.n_heads, h, batch_offset);
                }
            }
        }

        return true;
    }

    bool GQAAttention::compute_mpi(
        TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
        const GQAAttentionConfig &config,
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
            LOG_ERROR("[GQAAttention] SequenceParallel attention not yet implemented");
            return false;

        case MPIStrategy::PipelineParallel:
            // Pipeline-parallel doesn't change attention (distributes layers instead)
            return compute(Q, K, V, output, config, batch_size, sequence_lengths);

        case MPIStrategy::Hybrid:
            // TODO: Implement hybrid strategy (Phase 6)
            LOG_ERROR("[GQAAttention] Hybrid strategy not yet implemented");
            return false;

        default:
            LOG_ERROR("[GQAAttention] Unknown MPI strategy: " << static_cast<int>(config.mpi_strategy));
            return false;
        }
    }

    bool GQAAttention::compute_tensor_parallel(
        TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
        const GQAAttentionConfig &config,
        int batch_size,
        const std::vector<int> *sequence_lengths)
    {
        // Validate MPI context
        if (!config.mpi_ctx)
        {
            LOG_ERROR("[GQAAttention] Tensor-parallel attention requires MPI context");
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
            LOG_ERROR("[GQAAttention] Tensor-parallel requires n_heads (" << config.n_heads
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
            LOG_ERROR("[GQAAttention] total_tokens (" << total_tokens
                                                      << ") not divisible by batch_size (" << effective_batch_size << ")");
            return false;
        }

        LOG_DEBUG("[MPI TP] Batch-aware attention: total_tokens=" << total_tokens
                                                                  << " batch_size=" << effective_batch_size << " seq_len_per_batch=" << seq_len);

        int padded_seq_len = seq_len;

        // Distribute attention heads across ranks
        auto [start_head, local_n_heads] = config.mpi_ctx->get_local_slice(static_cast<size_t>(config.n_heads));

        if (config.verbose_logging && rank == 0)
        {
            LOG_INFO("[MPI TensorParallel] Attention: n_heads=" << config.n_heads
                                                                << ", world_size=" << world_size << ", local_n_heads=" << local_n_heads);
        }

        if (rank == 0 || config.verbose_logging)
        {
            LOG_INFO("[MPI TensorParallel] Rank " << rank << "/" << world_size
                                                  << ": Computing heads [" << start_head << ", " << (start_head + local_n_heads - 1) << "]");
        }

        // Get tensor data pointers
        const float *Q_data = Q->data();
        const float *K_data = K->data();
        const float *V_data = V->data();
        float *output_data = output->mutable_data();

        if (!Q_data || !K_data || !V_data || !output_data)
        {
            LOG_ERROR("[GQAAttention] Null pointer in attention tensors");
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

        // Allocate local output buffer for this rank's heads
        std::vector<float> local_output(total_tokens * local_n_heads * config.head_dim, 0.0f);

        // Temporary scores for local heads
        auto local_scores_tensor = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(local_n_heads * total_tokens), static_cast<size_t>(total_tokens)});
        float *local_scores = local_scores_tensor->mutable_data();

        // 3. Compute attention for local heads only
#pragma omp parallel for if (local_n_heads > 1)
        for (size_t local_h = 0; local_h < local_n_heads; ++local_h)
        {
            size_t global_h = start_head + local_h;
            float *scores_h = local_scores + local_h * total_tokens * total_tokens;

            // Extract Q and K for this head
            std::vector<float> Q_h(total_tokens * config.head_dim);
            std::vector<float> K_h(total_tokens * config.head_dim);

            extract_head_data(Q_data, Q_h.data(), total_tokens, config.head_dim, config.n_heads, global_h, 0);
            extract_head_data(K_expanded, K_h.data(), total_tokens, config.head_dim, config.n_heads, global_h, 0);

            LOG_DEBUG("[MPI TP] Rank " << rank << " Head " << global_h << " (local " << local_h << "): Q[0]=" << Q_h[0]);

            // GEMM: scores[local_h] = Q_h @ K_h^T
            if (!compute_attention_scores(Q_h.data(), K_h.data(), scores_h, total_tokens, config.head_dim, config.precision))
            {
                LOG_ERROR("[GQAAttention] Q·K^T GEMM failed for local head " << local_h);
            }

            LOG_DEBUG("[MPI TP] Rank " << rank << " Head " << global_h << ": scores[0]=" << scores_h[0]);
        }

        // 4. Scale scores by 1/sqrt(head_dim)
        LOG_DEBUG("[MPI TP] Rank " << rank << ": Scaling scores by 1/sqrt(" << config.head_dim << ")");
        scale_scores_inplace(local_scores, local_n_heads * total_tokens * total_tokens, config.head_dim);
        LOG_DEBUG("[MPI TP] Rank " << rank << " after scaling: scores[0]=" << local_scores[0]);

        // 5. Apply causal mask (if enabled)
        if (config.causal || sequence_lengths)
        {
            LOG_DEBUG("[MPI TP] Rank " << rank << ": Applying causal mask (batch_size=" << batch_size << ")");
            std::vector<float> mask(total_tokens * total_tokens);

            if (effective_batch_size == 1)
            {
                // Single sequence: standard causal mask
                attention_utils::create_causal_mask(mask.data(), total_tokens, config.window_size);
            }
            else
            {
                // Batched sequences: block-diagonal causal mask with padding
                const int *seq_lens_ptr = sequence_lengths ? sequence_lengths->data() : nullptr;
                attention_utils::create_batch_causal_mask(
                    mask.data(), effective_batch_size, padded_seq_len, seq_lens_ptr, config.window_size);
            }

            // Apply mask to local heads
#pragma omp parallel for if (local_n_heads > 1)
            for (size_t local_h = 0; local_h < local_n_heads; ++local_h)
            {
                float *scores_h = local_scores + local_h * total_tokens * total_tokens;
                attention_utils::apply_attention_mask(scores_h, mask.data(), total_tokens, total_tokens);
            }

            LOG_DEBUG("[MPI TP] Rank " << rank << " after masking: scores[0]=" << local_scores[0]);
        }

        // 6. Apply softmax
        apply_softmax(local_scores, local_n_heads * total_tokens, total_tokens);
        LOG_DEBUG("[MPI TP] Rank " << rank << " after softmax: scores[0]=" << local_scores[0]);

        if (config.verbose_logging && rank == 0)
        {
            LOG_INFO("[MPI TensorParallel] Applied vectorized softmax to " << local_n_heads << " heads");
        }

        // 7. Compute context: scores @ V for local heads
#pragma omp parallel for if (local_n_heads > 1)
        for (size_t local_h = 0; local_h < local_n_heads; ++local_h)
        {
            size_t global_h = start_head + local_h;
            const float *scores_h = local_scores + local_h * total_tokens * total_tokens;

            // Extract V for this head
            std::vector<float> V_h(total_tokens * config.head_dim);
            extract_head_data(V_expanded, V_h.data(), total_tokens, config.head_dim, config.n_heads, global_h, 0);

            // Compute context
            std::vector<float> context_h(total_tokens * config.head_dim);
            if (!compute_context_from_scores(scores_h, V_h.data(), context_h.data(), total_tokens, config.head_dim, config.precision))
            {
                LOG_ERROR("[GQAAttention] scores·V GEMM failed for local head " << local_h);
            }

            LOG_DEBUG("[MPI TP] Rank " << rank << " Head " << global_h << ": context[0]=" << context_h[0]);

            // Write to local_output buffer (contiguous for this rank's heads)
            for (int t = 0; t < total_tokens; ++t)
            {
#pragma omp simd
                for (int d = 0; d < config.head_dim; ++d)
                {
                    local_output[t * local_n_heads * config.head_dim + local_h * config.head_dim + d] =
                        context_h[t * config.head_dim + d];
                }
            }
        }

        LOG_DEBUG("[MPI TP] Rank " << rank << ": local_output[0]=" << local_output[0]);

        // 8. Allreduce: Sum local outputs from all ranks
        std::memset(output_data, 0, total_tokens * config.n_heads * config.head_dim * sizeof(float));

        // Create temporary buffer for allreduce (each rank's contribution)
        std::vector<float> send_buffer(total_tokens * config.n_heads * config.head_dim, 0.0f);

        // Copy local heads to correct position in send buffer
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
        int seq_len, int head_dim, ComputePrecision precision)
    {
        // GEMM: scores = Q @ K^T
        // Q: [seq_len, head_dim], K: [seq_len, head_dim]
        // scores: [seq_len, seq_len]

        // Select GEMM kernel based on precision
        std::unique_ptr<ITensorGemm> gemm_kernel;

        if (precision == ComputePrecision::BF16)
        {
            // Use BF16 GEMM for activation-activation multiply
            gemm_kernel = std::make_unique<BF16GemmKernel>(nullptr);
        }
        else if (precision == ComputePrecision::FP16)
        {
            // FP16 support: fallback to FP32 for now (TODO: implement FP16GemmKernel)
            gemm_kernel = std::make_unique<FP32GemmKernel>(nullptr);
        }
        else
        {
            // FP32 (default)
            gemm_kernel = std::make_unique<FP32GemmKernel>(nullptr);
        }

        return gemm_kernel->multiply_activations(
            Q, K, scores,
            seq_len, seq_len, head_dim,
            true,  // transpose_b (K^T)
            1.0f,  // alpha
            0.0f); // beta
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
        int seq_len, int head_dim, ComputePrecision precision)
    {
        // GEMM: context = scores @ V
        // scores: [seq_len, seq_len], V: [seq_len, head_dim]
        // context: [seq_len, head_dim]

        // Select GEMM kernel based on precision
        std::unique_ptr<ITensorGemm> gemm_kernel;

        if (precision == ComputePrecision::BF16)
        {
            // Use BF16 GEMM for activation-activation multiply
            gemm_kernel = std::make_unique<BF16GemmKernel>(nullptr);
        }
        else if (precision == ComputePrecision::FP16)
        {
            // FP16 support: fallback to FP32 for now (TODO: implement FP16GemmKernel)
            gemm_kernel = std::make_unique<FP32GemmKernel>(nullptr);
        }
        else
        {
            // FP32 (default)
            gemm_kernel = std::make_unique<FP32GemmKernel>(nullptr);
        }

        return gemm_kernel->multiply_activations(
            scores, V, context,
            seq_len, head_dim, seq_len,
            false, // no transpose
            1.0f,  // alpha
            0.0f); // beta
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
