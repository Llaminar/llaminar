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
#include <cstring>
#include <cmath>
#include <sstream>
#include <omp.h>
#include <atomic>

namespace llaminar2
{

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

        auto *activation_output = dynamic_cast<IActivationTensor *>(output);
        if (!activation_output)
        {
            LOG_ERROR("[MpiAttentionOrchestrator] Tensor-parallel attention requires activation tensor output");
            return false;
        }

        auto attention_kernel = activation_output->createAttention();
        if (!attention_kernel)
        {
            LOG_ERROR("[MpiAttentionOrchestrator] Failed to create attention kernel for tensor-parallel path");
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
