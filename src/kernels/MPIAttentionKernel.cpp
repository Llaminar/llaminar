/**
 * @file MPIAttentionKernel.cpp
 * @brief Multi-head self-attention (prefill / decode) with optional grouped (GQA) key/value heads in MPI context.
 *
 * @section Contract
 * Inputs:
 *  - inputs[0]: Query tensor Q [seq_len_q, n_heads, head_dim].
 *  - inputs[1]: Key tensor K   [seq_len_k, n_kv_heads, head_dim].
 *  - inputs[2]: Value tensor V [seq_len_k, n_kv_heads, head_dim].
 *  - inputs[3] (optional): Causal mask or attention bias structure.
 * Outputs:
 *  - outputs[0]: Context tensor C [seq_len_q, n_heads, head_dim].
 * Semantics:
 *  - Scaled dot-product attention with RoPE already applied upstream.
 *  - Grouped attention: heads mapped so each query head attends corresponding kv group (n_heads multiple of n_kv_heads).
 * Scaling & Masking:
 *  - Score_ij = (Q_i · K_j) / sqrt(head_dim).
 *  - If causal, j > i positions masked (set to -inf before softmax).
 * Numerical Expectations:
 *  - Softmax stability: Max-subtraction per row; overflow-safe for typical head_dim <= 256.
 *  - Relative error vs reference < 5e-5 (accumulation order dependent) for float32.
 * Distribution Strategy (current):
 *  - Replicated computation across ranks (future: sequence or head partition + AllReduce).
 * Error Modes:
 *  - Dimension mismatch, invalid head grouping (n_heads % n_kv_heads != 0), null inputs.
 *  - seq_len_k < seq_len_q under causal mask unsupported.
 * Threading:
 *  - Parallelized across (seq_len_q * n_heads) independent softmax+matmul tasks.
 * Performance Notes:
 *  - Potential optimization: fuse QK^T + softmax + PV with block tiling and cache-friendly layout.
 * Future Extensions:
 *  - Flash-attention style block-sparse kernel, distributed head partition.
 *  - Mixed precision (FP16/BF16) with accumulation in float32.
 * Safety:
 *  - Checks for NaN/Inf after softmax optionally (debug build) could be enabled via env flag (TBD).
 * @warning Ensure RoPE applied before this kernel for correct positional alignment.
 * @todo Introduce streaming KV cache update path for decode stage.
 * @author David Sanftenberg
 */
#include "MPIAttentionKernel.h"
#include "../utils/perf_counters.h"
#include "../utils/debug_env.h"
#include "../utils/debug_sharding.h"
#include "kernels/common/attention_primitives.h" // shared fast paths (RoPE, QK scores, scores@V)
#include "../adaptive_matmul.h"
#include "../backends/prefill_backend.h"
#include "../backends/inference_backend.h"
#include "../tp_policy.h"
#include "../logger.h"
#include "../performance_timer.h"
#include "../tensors/tensor_factory.h"
#include "../tensors/sharded_simple_tensor.h"
#include "../tensors/shard_spec.h"
#include "../tensors/tp_output_projection_executor.h" // TP simulation executor
#include <cmath>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <iostream>
#include <sstream>
#include <mpi.h>
#include "tensors/tp_partition.h"
#include "../qwen_pipeline.h" // for LayerTokenDiffRow instrumentation

namespace llaminar
{

    MPIAttentionKernel::MPIAttentionKernel(int n_head, int n_head_kv, int head_dim, float rope_freq_base, DistributionStrategy strategy)
        : n_head_(n_head), n_head_kv_(n_head_kv), head_dim_(head_dim), rope_freq_base_(rope_freq_base), strategy_(strategy)
    {
        if (head_dim_ <= 0 || n_head_ <= 0 || n_head_kv_ <= 0)
        {
            throw std::invalid_argument("MPIAttentionKernel: invalid constructor parameters");
        }
        if (n_head_kv_ > n_head_)
        {
            throw std::invalid_argument("MPIAttentionKernel: n_head_kv cannot exceed n_head");
        }
    }

    bool MPIAttentionKernel::execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                     std::vector<std::shared_ptr<TensorBase>> &outputs)
    {
        // Execution timing scaffolding (per-phase breakdown)
        auto t_exec_start = std::chrono::high_resolution_clock::now();
        double t_distribute_ms = 0.0, t_proj_ms = 0.0, t_rope_ms = 0.0, t_attn_ms = 0.0;
        double t_reconstruct_wo_ms = 0.0, t_output_proj_ms = 0.0, t_gather_pre_ms = 0.0, t_gather_post_ms = 0.0, t_emit_ms = 0.0;
        const auto &perf_env = debugEnv().performance;
        auto time_block = [](auto &&fn)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            auto t1 = std::chrono::high_resolution_clock::now();
            return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
        };

        if (!validate(inputs, outputs))
        {
            return false;
        }

        // Extract inputs: input, wq, wk, wv, wo, k_cache, v_cache
        auto global_input = inputs[0];
        auto global_wq = inputs[1];
        auto global_wk = inputs[2];
        auto global_wv = inputs[3];
        auto global_wo = inputs[4];
        auto k_cache = inputs[5]; // TODO: Handle KV cache in future version
        auto v_cache = inputs[6]; // TODO: Handle KV cache in future version
        auto global_output = outputs[0];

        size_t seq_len = static_cast<size_t>(global_input->shape()[0]);
        size_t d_model = static_cast<size_t>(global_input->shape()[1]);

        // Heuristic auto-switch: if user did NOT explicitly request a mode via env var earlier,
        // choose GatherHeadsPreProjection when sequence length exceeds threshold.
        // Threshold controlled by LLAMINAR_ATTN_GATHER_THRESHOLD (default 1024).
        static bool env_mode_forced = debugEnv().attention.output_mode_forced;
        if (!env_mode_forced)
        {
            int threshold = 1024;
            if (debugEnv().attention.gather_threshold >= 1)
                threshold = debugEnv().attention.gather_threshold;
            AttentionOutputMode decided = output_mode_; // current (likely LocalHeads)
            if (seq_len >= (size_t)threshold)
            {
                decided = AttentionOutputMode::GatherHeadsPreProjection;
            }
            if (decided != output_mode_ && getRank() == 0)
            {
                LOG_INFO("Attention heuristic switching mode -> gather_pre (seq_len=" << seq_len
                                                                                      << ", threshold=" << threshold << ")");
            }
            output_mode_ = decided;
        }

        // Ensure input is replicated across all ranks for tests that only initialize on rank 0.
        // This avoids undefined values on other ranks influencing local computations.
        if (getSize() > 1)
        {
            MPI_Bcast(global_input->data(), (int)global_input->size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
        }

        // Get local head distribution
        auto [local_heads, head_offset] = getHeadDistribution();
        size_t local_head_dim = static_cast<size_t>(local_heads * head_dim_);

        // Detect pre-sharded weights (Head axis) to bypass legacy distributeInputs.
        auto is_head_sharded = [](const std::shared_ptr<TensorBase> &t) -> bool
        {
            if (!t)
                return false;
            // Dynamic cast to ShardedSimpleTensor if available.
            if (auto *raw = dynamic_cast<ShardedSimpleTensor *>(t.get()))
            {
                const auto &spec = raw->shard_spec();
                return spec.is_sharded() && spec.axis == ShardSpec::Axis::Heads;
            }
            return false;
        };

        bool pre_sharded = is_head_sharded(global_wq) && is_head_sharded(global_wk) &&
                           is_head_sharded(global_wv) && is_head_sharded(global_wo);

        std::shared_ptr<TensorBase> local_wq;
        std::shared_ptr<TensorBase> local_wk;
        std::shared_ptr<TensorBase> local_wv;
        std::shared_ptr<TensorBase> local_wo;

        if (local_heads == 0)
        {
            // No heads assigned to this rank (imbalanced head/world configuration). Allocate minimal tensors and skip work.
            local_wq = createLocalSimpleTensor({d_model, 0});
            local_wk = createLocalSimpleTensor({d_model, 0});
            local_wv = createLocalSimpleTensor({d_model, 0});
            local_wo = createLocalSimpleTensor({0, d_model});
            // Still need empty Q/K/V to satisfy downstream shapes
            auto local_q = createLocalSimpleTensor({seq_len, 0});
            auto local_k = createLocalSimpleTensor({seq_len, 0});
            auto local_v = createLocalSimpleTensor({seq_len, 0});
            // Produce a zeroed attention output (residual path will just add zero)
            if (!outputs.empty() && outputs[0])
            {
                auto &attn_out_zero = outputs[0];
                if (attn_out_zero->shape().size() == 2 && attn_out_zero->shape()[0] == (int)seq_len && attn_out_zero->shape()[1] == d_model)
                {
                    memset(attn_out_zero->data(), 0, seq_len * d_model * sizeof(float));
                }
            }
            if (getRank() == 0)
                LOG_WARN("MPIAttentionKernel: rank has zero local heads; producing zero contribution");
            return true;
        }

        if (pre_sharded)
        {
            // Assume each provided weight already holds only this rank's slice.
            // Validate slice dimensionality matches our expected local_head_dim.
            size_t wq_cols = static_cast<size_t>(global_wq->shape()[1]);
            size_t wk_cols = static_cast<size_t>(global_wk->shape()[1]);
            size_t wv_cols = static_cast<size_t>(global_wv->shape()[1]);
            size_t wo_rows = static_cast<size_t>(global_wo->shape()[0]);
            if (wq_cols != local_head_dim || wk_cols != local_head_dim || wv_cols != local_head_dim || wo_rows != local_head_dim)
            {
                LOG_ERROR("MPIAttentionKernel: pre-sharded weight dims mismatch local expectation (expected " << local_head_dim << ")");
                return false;
            }
            local_wq = global_wq;
            local_wk = global_wk;
            local_wv = global_wv;
            local_wo = global_wo;
            if (getRank() == 0)
            {
                LOG_DEBUG("MPIAttentionKernel: detected pre-sharded head-axis weights; skipping distributeInputs");
            }
        }
        else
        {
            // Legacy path: allocate local slices then copy/distribute.
            local_wq = createLocalSimpleTensor({d_model, local_head_dim});
            local_wk = createLocalSimpleTensor({d_model, local_head_dim});
            local_wv = createLocalSimpleTensor({d_model, local_head_dim});
            local_wo = createLocalSimpleTensor({local_head_dim, d_model});
            {
                PERF_SCOPED_TIMER("MPIAttentionKernel::distributeInputs");
                t_distribute_ms = time_block([&]
                                             { distributeInputs(global_input, global_wq, global_wk, global_wv, global_wo,
                                                                local_wq, local_wk, local_wv, local_wo, seq_len, d_model); });
            }
        }
        // Create local projection tensors
        auto local_q = createLocalSimpleTensor({seq_len, local_head_dim});
        auto local_k = createLocalSimpleTensor({seq_len, local_head_dim});
        auto local_v = createLocalSimpleTensor({seq_len, local_head_dim});

        // Compute Q, K, V projections for local heads using COSMA
        {
            PERF_SCOPED_TIMER("MPIAttentionKernel::computeLocalProjections");
            t_proj_ms = time_block([&]
                                   { computeLocalProjections(global_input, local_wq, local_wk, local_wv,
                                                             local_q, local_k, local_v, seq_len, d_model); });
        }
        if (debugEnv().attention.internal_diff && debugEnv().pipeline.layer_token_diff && getRank() == 0 && seq_len > 0)
        {
            size_t slice = (size_t)local_heads * head_dim_;
            size_t offset = (seq_len - 1) * slice;
            bool incr = (seq_len == 1 && n_past_ > 0);
            int capture_seq_len_meta = incr ? (n_past_ + 1) : (int)seq_len;
            QwenPipeline::appendInternalAttnRow(nullptr,
                                                layer_index_, capture_seq_len_meta, incr,
                                                "attn_int_q_proj", local_q->data() + offset, slice);
            QwenPipeline::appendInternalAttnRow(nullptr,
                                                layer_index_, capture_seq_len_meta, incr,
                                                "attn_int_k_proj", local_k->data() + offset, slice);
        }
        // Apply RoPE to local Q and K
        {
            PERF_SCOPED_TIMER("MPIAttentionKernel::applyLocalRoPE");
            t_rope_ms = time_block([&]
                                   { applyLocalRoPE(local_q->data(), local_k->data(), seq_len, local_heads); });
        }
        if (debugEnv().attention.internal_diff && debugEnv().pipeline.layer_token_diff && getRank() == 0 && seq_len > 0)
        {
            size_t slice = (size_t)local_heads * head_dim_;
            size_t offset = (seq_len - 1) * slice;
            bool incr = (seq_len == 1 && n_past_ > 0);
            int capture_seq_len_meta = incr ? (n_past_ + 1) : (int)seq_len;
            QwenPipeline::appendInternalAttnRow(nullptr, layer_index_, capture_seq_len_meta, incr, "attn_int_q_rope", local_q->data() + offset, slice);
            QwenPipeline::appendInternalAttnRow(nullptr, layer_index_, capture_seq_len_meta, incr, "attn_int_k_rope", local_k->data() + offset, slice);
        }

        // Prefill KV cache population (only once at sequence start). Support both full and grouped KV heads.
        // For grouped KV (multi-query) we map each query head to its KV head via (q_head % n_head_kv_). We may
        // write duplicate KV slices when multiple query heads map to the same KV head; duplicates are identical
        // and therefore harmless for correctness in this diagnostic path.
        if (n_past_ == 0 && seq_len > 1)
        {
            float *k_cache_ptr = k_cache->data();
            float *v_cache_ptr = v_cache->data();
            size_t kv_dim = static_cast<size_t>(n_head_kv_) * head_dim_;
            const float *local_k_ptr = local_k->data();
            const float *local_v_ptr = local_v->data();
            int group_size = (n_head_kv_ == 0) ? 1 : (n_head_ / n_head_kv_);
            for (size_t t = 0; t < seq_len; ++t)
            {
                const float *k_row_local = local_k_ptr + t * local_head_dim;
                const float *v_row_local = local_v_ptr + t * local_head_dim;
                for (int lh = 0; lh < local_heads; ++lh)
                {
                    size_t global_q_head = head_offset + lh;
                    size_t kv_head = static_cast<size_t>(global_q_head % n_head_kv_);
                    // For grouped KV ensure only canonical query head (lowest in group) writes to cache to avoid
                    // clobbering with potentially different numeric projections across ranks.
                    if (n_head_kv_ != n_head_)
                    {
                        if ((global_q_head % group_size) != 0)
                            continue; // skip non-canonical heads
                    }
                    float *k_dest = k_cache_ptr + t * kv_dim + kv_head * head_dim_;
                    float *v_dest = v_cache_ptr + t * kv_dim + kv_head * head_dim_;
                    const float *k_src = k_row_local + lh * head_dim_;
                    const float *v_src = v_row_local + lh * head_dim_;
                    std::memcpy(k_dest, k_src, head_dim_ * sizeof(float));
                    std::memcpy(v_dest, v_src, head_dim_ * sizeof(float));
                }
            }
            if (getRank() == 0)
                LOG_DEBUG("MPIAttentionKernel: populated KV cache for prefill seq_len=" << seq_len << (n_head_kv_ == n_head_ ? "" : " (grouped)"));
        }

        // Create local attended output tensor. For GatherHeadsPreProjection optimization we store
        // directly in heads-major layout [local_head_dim, seq_len] to skip a reorder later.
        // Treat Replicated as an alias of GatherHeadsPreProjection (single global projection path)
        bool pre_mode = (output_mode_ == AttentionOutputMode::GatherHeadsPreProjection ||
                         output_mode_ == AttentionOutputMode::Replicated);
        bool incremental_single = (seq_len == 1 && n_past_ > 0); // single-token decode with history
        int capture_seq_len_meta = incremental_single ? (n_past_ + 1) : (int)seq_len;
        auto local_attended_output = pre_mode
                                         ? createLocalSimpleTensor({local_head_dim, seq_len})
                                         : createLocalSimpleTensor({seq_len, local_head_dim});

        // Incremental decode specialized path: use KV cache (inputs[5], inputs[6]) to attend over all past tokens + current
        if (incremental_single)
        {
            PERF_SCOPED_TIMER("MPIAttentionKernel::incrementalAttention");
            t_attn_ms = time_block([&]
                                   {
                // Support grouped KV incremental decode (n_head_kv_ <= n_head_). When grouped, all query heads map
                // to a (possibly smaller) set of KV heads. We re-use the populated KV cache and perform attention
                // over all past tokens + current.

                size_t kv_dim = static_cast<size_t>(n_head_kv_) * head_dim_;
                // Safety check on cache shapes (optional)
                if ((size_t)k_cache->shape()[1] != kv_dim || (size_t)v_cache->shape()[1] != kv_dim)
                {
                    LOG_ERROR("MPIAttentionKernel: KV cache dim mismatch for incremental path");
                    throw std::runtime_error("KV cache dimension mismatch");
                }

                float *k_cache_ptr = k_cache->data();
                float *v_cache_ptr = v_cache->data();
                const float *q_cur = local_q->data(); // [1, local_head_dim]
                const float *k_cur = local_k->data();
                const float *v_cur = local_v->data();
                // Write current token K/V into cache for each (possibly grouped) KV head mapping.
                // Duplicate writes for grouped heads are harmless (identical data) and keep logic simple.
                int group_size = (n_head_kv_ == 0) ? 1 : (n_head_ / n_head_kv_);
                for (int lh = 0; lh < local_heads; ++lh)
                {
                    size_t global_q_head = head_offset + lh;
                    size_t kv_head = static_cast<size_t>(global_q_head % n_head_kv_);
                    if (n_head_kv_ != n_head_)
                    {
                        if ((global_q_head % group_size) != 0)
                            continue; // only canonical query head writes
                    }
                    float *k_dest = k_cache_ptr + n_past_ * kv_dim + kv_head * head_dim_;
                    float *v_dest = v_cache_ptr + n_past_ * kv_dim + kv_head * head_dim_;
                    const float *k_src = k_cur + lh * head_dim_;
                    const float *v_src = v_cur + lh * head_dim_;
                    std::memcpy(k_dest, k_src, head_dim_ * sizeof(float));
                    std::memcpy(v_dest, v_src, head_dim_ * sizeof(float));
                }

                // For each local query head, compute attention over tokens [0, n_past_] using its mapped KV head.
                const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
                for (int lh = 0; lh < local_heads; ++lh)
                {
                    size_t global_q_head = head_offset + lh;
                    size_t kv_head = static_cast<size_t>(global_q_head % n_head_kv_);
                    const float *q_vec = q_cur + lh * head_dim_;
                    // Temporary score buffer (stack) limited by max seq len (assumed small in tests)
                    std::vector<float> scores(n_past_ + 1);
                    float max_score = -std::numeric_limits<float>::infinity();
                    for (int t = 0; t < n_past_; ++t)
                    {
                        const float *k_vec = k_cache_ptr + t * kv_dim + kv_head * head_dim_;
                        float dot = 0.f;
                        for (int d = 0; d < head_dim_; ++d)
                            dot += q_vec[d] * k_vec[d];
                        dot *= scale;
                        scores[t] = dot;
                        if (dot > max_score)
                            max_score = dot;
                    }
                    // Current token score
                    {
                        const float *k_vec = k_cache_ptr + n_past_ * kv_dim + kv_head * head_dim_;
                        float dot = 0.f;
                        for (int d = 0; d < head_dim_; ++d)
                            dot += q_vec[d] * k_vec[d];
                        dot *= scale;
                        scores[n_past_] = dot;
                        if (dot > max_score)
                            max_score = dot;
                    }
                    // Softmax
                    float sum_exp = 0.f;
                    for (float &s : scores)
                    {
                        s = std::exp(s - max_score);
                        sum_exp += s;
                    }
                    float inv_sum = 1.f / (sum_exp + 1e-9f);
                    for (float &s : scores)
                        s *= inv_sum;
                    // Weighted value accumulation
                    float *out_base = pre_mode ? (local_attended_output->data() + lh * head_dim_ * seq_len) : local_attended_output->data();
                    float *out_vec = pre_mode ? (out_base) : (out_base + lh * head_dim_); // seq_len==1 in incremental
                    std::fill(out_vec, out_vec + head_dim_, 0.f);
                    for (int t = 0; t < n_past_; ++t)
                    {
                        const float *v_vec = v_cache_ptr + t * kv_dim + kv_head * head_dim_;
                        float w = scores[t];
                        for (int d = 0; d < head_dim_; ++d)
                            out_vec[d] += w * v_vec[d];
                    }
                    // Current token contribution
                    {
                        const float *v_vec = v_cache_ptr + n_past_ * kv_dim + kv_head * head_dim_;
                        float w = scores[n_past_];
                        for (int d = 0; d < head_dim_; ++d)
                            out_vec[d] += w * v_vec[d];
                    }
                }

                // NOTE: We already wrote current token K/V for (canonical) heads above.
                // The previous implementation redundantly attempted to memcpy a contiguous
                // block for all local heads assuming a per-query-head cache layout. That
                // layout is only valid when n_head_kv_ == n_head_. For grouped KV (GQA)
                // it caused out-of-bounds writes (global_head >= n_head_kv_) and numerical
                // drift. We remove it and rely solely on the canonical per-kv-head writes.

                size_t T = (size_t)n_past_ + 1; // total tokens including current
                size_t expected_T = expected_total_window_ ? expected_total_window_ : T;
                const auto &ade = debugEnv().attention_decode;
                if(ade.decode_diag && getRank()==0) {
                    int cache_rows = k_cache ? k_cache->shape()[0] : -1;
                    if(cache_rows >=0 && (size_t)cache_rows < T) {
                        LOG_WARN("[DecodeAttnDiag] cache underrun: layer="<<layer_index_<<" token_pos="<<n_past_
                                 <<" cache_rows="<<cache_rows<<" expected>="<<T);
                    }
                    LOG_WARN("[DecodeAttnDiag] layer="<<layer_index_<<" token_pos="<<n_past_
                             <<" local_heads="<<local_heads<<" head_offset="<<head_offset
                             <<" T="<<T<<" expected_T="<<expected_T<<" head_dim="<<head_dim_<<" cache_rows="<<cache_rows);
                    if(expected_total_window_ && T != expected_T) {
                        LOG_WARN("[DecodeAttnDiag] WINDOW_MISMATCH layer="<<layer_index_<<" token_pos="<<n_past_
                                 <<" actual_T="<<T<<" expected_T="<<expected_T);
                    }
                }
                // Per-instance monotonic check (resets on new decode stream when expected_T==1)
                if(expected_T == 1) {
                    last_seen_decode_T_ = 0; // new stream (first token)
                }
                if(ade.decode_diag && T < last_seen_decode_T_ && getRank()==0) {
                    LOG_WARN("[DecodeAttnDiag] NON_MONOTONIC_INSTANCE_T previous="<<last_seen_decode_T_<<" current="<<T
                             <<" layer="<<layer_index_<<" token_pos="<<n_past_);
                }
                if(T > last_seen_decode_T_) last_seen_decode_T_ = T;
                if(ade.decode_diag && k_cache && getRank()==0) {
                    int cache_rows = k_cache->shape()[0];
                    if((int)T > cache_rows) {
                        LOG_WARN("[DecodeAttnDiag] T exceeds cache_rows: T="<<T<<" cache_rows="<<cache_rows<<" layer="<<layer_index_);
                    }
                }
                // Build contiguous Q/K/V blocks representing the full causal window [0..T-1]
                // so we can invoke the standard computeLocalAttention path for numerical parity.
                size_t local_head_dim_sz = local_head_dim;
                auto full_q = createLocalSimpleTensor({T, local_head_dim_sz});
                auto full_k = createLocalSimpleTensor({T, local_head_dim_sz});
                auto full_v = createLocalSimpleTensor({T, local_head_dim_sz});
                float *fq = full_q->data();
                float *fk = full_k->data();
                float *fv = full_v->data();
                // Populate K/V from cache for past tokens (0..T-1)
                for (size_t t = 0; t < T; ++t)
                {
                    for (int lh = 0; lh < local_heads; ++lh)
                    {
                        int global_head = head_offset + lh;
                        int kv_head = (n_head_kv_ == n_head_) ? global_head : (global_head % n_head_kv_);
                        const float *k_src = k_cache_ptr + t * kv_dim + kv_head * head_dim_;
                        const float *v_src = v_cache_ptr + t * kv_dim + kv_head * head_dim_;
                        float *k_dst = fk + t * local_head_dim_sz + lh * head_dim_;
                        float *v_dst = fv + t * local_head_dim_sz + lh * head_dim_;
                        std::memcpy(k_dst, k_src, head_dim_ * sizeof(float));
                        std::memcpy(v_dst, v_src, head_dim_ * sizeof(float));
                    }
                }
                // Populate Q: only last row (T-1) has real query; earlier rows unused for our output row.
                std::memset(fq, 0, T * local_head_dim_sz * sizeof(float));
                std::memcpy(fq + (T - 1) * local_head_dim_sz, q_cur, local_head_dim_sz * sizeof(float));

                // Compute attention across window using existing numerically stable path.
                auto full_out = createLocalSimpleTensor({T, local_head_dim_sz});
                computeLocalAttention(fq, fk, fv, full_out->data(), T, local_heads);
                // Internal diff capture: full window context (reconstructed incremental path)
                if(debugEnv().attention.internal_diff && debugEnv().pipeline.layer_token_diff && getRank()==0) {
                    size_t slice = (size_t)local_heads * head_dim_;
                    const float *last_row_full = full_out->data() + (T - 1) * slice;
                    QwenPipeline::appendInternalAttnRow(nullptr, layer_index_, (int)T, true, "attn_int_context_full", last_row_full, slice);
                }
                if(ade.decode_diag && ade.dump_full_qkv && getRank()==0 && (int)local_head_dim_sz <= ade.dump_limit) {
                    auto dump_vec = [&](const char* tag, const float* ptr){ std::ostringstream oss; oss<<"[DecodeAttnDump] "<<tag<<":"; for(size_t i=0;i<local_head_dim_sz;++i){ oss<<ptr[i]; if(i+1<local_head_dim_sz) oss<<","; } LOG_WARN(oss.str()); };
                    dump_vec("q_last", fq + (T-1)*local_head_dim_sz);
                    if(T>1) dump_vec("k_prev", fk + (T-2)*local_head_dim_sz);
                    dump_vec("k_last", fk + (T-1)*local_head_dim_sz);
                }

                // Extract last row into expected output layout.
                const float *last_row = full_out->data() + (T - 1) * local_head_dim_sz;
                if (pre_mode)
                {
                    float *dstA = local_attended_output->data(); // heads-major [local_head_dim, 1]
                    for (size_t h = 0; h < local_head_dim_sz; ++h)
                        dstA[h * seq_len + 0] = last_row[h];
                }
                else
                {
                    float *dst = local_attended_output->data(); // [1, local_head_dim]
                    std::memcpy(dst, last_row, local_head_dim_sz * sizeof(float));
                } });
            // Internal diff capture: incremental context after mapping to local_attended_output layout
            if (debugEnv().attention.internal_diff && debugEnv().pipeline.layer_token_diff && getRank() == 0)
            {
                size_t slice = (size_t)local_heads * head_dim_;
                std::vector<float> tmp(slice);
                const float *src_base = local_attended_output->data();
                if (pre_mode)
                {
                    for (size_t h = 0; h < slice; ++h)
                        tmp[h] = src_base[h * seq_len + 0];
                }
                else
                {
                    std::memcpy(tmp.data(), src_base, slice * sizeof(float));
                }
                QwenPipeline::appendInternalAttnRow(nullptr, layer_index_, capture_seq_len_meta, true, "attn_int_context", tmp.data(), slice);
            }
        }
        else
        {
            // Standard (prefill or trivial single-token) attention over provided seq_len window
            PERF_SCOPED_TIMER("MPIAttentionKernel::computeLocalAttention");
            t_attn_ms = time_block([&]
                                   {
                if (pre_mode)
                {
                    auto tmp_std = createLocalSimpleTensor({seq_len, local_head_dim});
                    computeLocalAttention(local_q->data(), local_k->data(), local_v->data(),
                                          tmp_std->data(), seq_len, local_heads);
                    const float *srcA = tmp_std->data();
                    float *dstA = local_attended_output->data();
                    for (size_t s = 0; s < seq_len; ++s)
                    {
                        const float *row = srcA + s * local_head_dim;
                        for (size_t h = 0; h < local_head_dim; ++h)
                            dstA[h * seq_len + s] = row[h];
                    }
                }
                else
                {
                    computeLocalAttention(local_q->data(), local_k->data(), local_v->data(),
                                          local_attended_output->data(), seq_len, local_heads);
                } });
            // Internal diff capture: prefill / standard context last token row
            if (debugEnv().attention.internal_diff && debugEnv().pipeline.layer_token_diff && getRank() == 0 && seq_len > 0)
            {
                size_t slice = (size_t)local_heads * head_dim_;
                std::vector<float> tmp(slice);
                if (pre_mode)
                {
                    const float *src = local_attended_output->data();
                    size_t last_idx = (size_t)(seq_len - 1);
                    for (size_t h = 0; h < slice; ++h)
                        tmp[h] = src[h * seq_len + last_idx];
                }
                else
                {
                    const float *src = local_attended_output->data() + (seq_len - 1) * slice;
                    std::memcpy(tmp.data(), src, slice * sizeof(float));
                }
                QwenPipeline::appendInternalAttnRow(nullptr, layer_index_, capture_seq_len_meta, (seq_len == 1 && n_past_ > 0), "attn_int_context", tmp.data(), slice);
            }
        }

        // Pre-projection gather path: gather head contexts BEFORE output projection so we do the
        // expensive WO matmul only once globally (optionally TP split later). We skip computing
        // local output projection in this branch.
        if (output_mode_ == AttentionOutputMode::GatherHeadsPreProjection ||
            output_mode_ == AttentionOutputMode::Replicated)
        {
            PERF_SCOPED_TIMER("MPIAttentionKernel::gatherHeadsPreProjection");
            size_t total_head_dim = static_cast<size_t>(n_head_) * head_dim_;
            size_t local_head_dim = static_cast<size_t>(local_heads * head_dim_);
            // local_attended_output is already heads-major if optimization active; if not, transpose.
            std::shared_ptr<TensorBase> local_heads_major = local_attended_output;
            if (local_attended_output->shape().size() == 2 && local_attended_output->shape()[0] == (int)seq_len)
            {
                // Need transpose to heads-major
                auto tmp_heads_major = createLocalSimpleTensor({local_head_dim, seq_len});
                const float *src = local_attended_output->data();
                float *dst = tmp_heads_major->data();
                for (size_t s = 0; s < seq_len; ++s)
                {
                    const float *row = src + s * local_head_dim;
                    for (size_t h = 0; h < local_head_dim; ++h)
                        dst[h * seq_len + s] = row[h];
                }
                local_heads_major = tmp_heads_major;
            }

            auto global_heads_major = createLocalSimpleTensor({total_head_dim, seq_len});
            std::vector<int> recvcounts(getSize());
            std::vector<int> displs(getSize());
            size_t running = 0;
            for (int r = 0; r < getSize(); ++r)
            {
                auto [r_heads, r_off] = getHeadDistribution(r);
                size_t r_head_dim = static_cast<size_t>(r_heads * head_dim_);
                size_t elems = r_head_dim * seq_len;
                recvcounts[r] = static_cast<int>(elems);
                displs[r] = static_cast<int>(running);
                running += elems;
            }
            size_t send_elems = local_head_dim * seq_len;
            (void)pre_mode; // silence unused warning
            MPI_Allgatherv(local_heads_major->data(), (int)send_elems, MPI_FLOAT,
                           global_heads_major->data(), recvcounts.data(), displs.data(), MPI_FLOAT, MPI_COMM_WORLD);

            // 3. Reorder heads-major global buffer back to [seq_len, total_head_dim]
            auto full_attended = createLocalSimpleTensor({seq_len, total_head_dim});
            float *dst_full = full_attended->data();
            const float *src_heads_major = global_heads_major->data();
            for (size_t h = 0; h < total_head_dim; ++h)
            {
                const float *head_vec = src_heads_major + h * seq_len;
                for (size_t s = 0; s < seq_len; ++s)
                {
                    dst_full[s * total_head_dim + h] = head_vec[s];
                }
            }

            // 4. Ensure we have full (replicated) WO. If input WO was head-sharded, reconstruct it.
            bool wo_head_sharded = false;
            if (auto *pwo = dynamic_cast<ShardedSimpleTensor *>(global_wo.get()))
            {
                if (pwo->shard_spec().is_sharded() && pwo->shard_spec().axis == ShardSpec::Axis::Heads)
                {
                    wo_head_sharded = true;
                }
            }
            std::shared_ptr<TensorBase> full_wo = global_wo;
            if (wo_head_sharded)
            {
                t_reconstruct_wo_ms = time_block([&]
                                                 {
                    PERF_SCOPED_TIMER("MPIAttentionKernel::reconstructWO");
                    full_wo = createLocalSimpleTensor({total_head_dim, d_model});
                    std::vector<int> w_recvcounts(getSize());
                    std::vector<int> w_displs(getSize());
                    size_t wrunning = 0;
                    for (int r = 0; r < getSize(); ++r)
                    {
                        auto [r_heads, r_off] = getHeadDistribution(r);
                        size_t r_head_dim = static_cast<size_t>(r_heads * head_dim_);
                        size_t elems = r_head_dim * d_model;
                        w_recvcounts[r] = (int)elems;
                        w_displs[r] = (int)wrunning;
                        wrunning += elems;
                    }
                    size_t w_send_elems = local_head_dim * d_model;
                    MPI_Allgatherv(local_wo->data(), (int)w_send_elems, MPI_FLOAT,
                                   full_wo->data(), w_recvcounts.data(), w_displs.data(), MPI_FLOAT, MPI_COMM_WORLD); });
            }

            // 5. Single (global) output projection: [seq_len, total_head_dim] * [total_head_dim, d_model]
            t_output_proj_ms = time_block([&]
                                          {
                PerfMatmulPhaseScope phase(2,4); // Attention output projection (gather-pre path)
                if (!adaptive_matmul(full_attended->data(), full_wo->data(), global_output->data(),
                                     seq_len, d_model, total_head_dim, false))
                {
                    LOG_ERROR("MPIAttentionKernel: pre-projection gathered output projection failed");
                    throw std::runtime_error("Attention output projection failed");
                } });

            // Populate metadata early and return.
            // Preserve the originally requested mode (Replicated remains distinguishable in metadata)
            last_meta_.mode = output_mode_;
            last_meta_.local_head_offset = head_offset;
            last_meta_.local_head_count = local_heads;
            last_meta_.concatenated = true;
            last_meta_.replicated = true; // full projection identical across ranks
            auto end_pre = std::chrono::high_resolution_clock::now();
            double ms_pre = std::chrono::duration<double, std::milli>(end_pre - t_exec_start).count();
            t_gather_pre_ms = ms_pre; // aggregate path timing (full since start).
            if (getRank() == 0)
            {
                if (output_mode_ == AttentionOutputMode::Replicated)
                    LOG_DEBUG("MPIAttention (Replicated alias -> gather_pre) executed in " << ms_pre << " ms heads=" << n_head_);
                else
                    LOG_DEBUG("MPIAttention (GatherHeadsPreProjection) executed in " << ms_pre << " ms heads=" << n_head_);
                LOG_DEBUG("AttentionResultMeta mode=" << (int)last_meta_.mode
                                                      << " heads_off=" << last_meta_.local_head_offset
                                                      << " heads_cnt=" << last_meta_.local_head_count
                                                      << " concat=" << last_meta_.concatenated
                                                      << " repl=" << last_meta_.replicated);
            }
            if (perf_env.layer_attention && getRank() == perf_env.log_rank)
            {
                double total_ms = ms_pre;
                double accounted = t_distribute_ms + t_proj_ms + t_rope_ms + t_attn_ms + t_reconstruct_wo_ms + t_output_proj_ms;
                double other = std::max(0.0, total_ms - accounted);
                LOG_INFO("ATTN_LAYER_TIMING seq_len=" << seq_len
                                                      << " heads=" << n_head_
                                                      << " head_dim=" << head_dim_
                                                      << " local_heads=" << local_heads
                                                      << " mode=pre_projection"
                                                      << " distribute_ms=" << t_distribute_ms
                                                      << " proj_ms=" << t_proj_ms
                                                      << " rope_ms=" << t_rope_ms
                                                      << " attn_ms=" << t_attn_ms
                                                      << " recon_wo_ms=" << t_reconstruct_wo_ms
                                                      << " out_proj_ms=" << t_output_proj_ms
                                                      << " total_ms=" << total_ms
                                                      << " other_ms=" << other
                                                      << " layer_idx=" << PerformanceCounters::tl_phase_.layer_index);
            }
            return true;
        }

        // Create local final output tensor
        auto local_final_output = createLocalSimpleTensor({seq_len, d_model});

        // Compute output projection for local heads using COSMA
        {
            PERF_SCOPED_TIMER("MPIAttentionKernel::computeLocalOutputProjection");
            t_output_proj_ms = time_block([&]
                                          { PerfMatmulPhaseScope phase(2,4); computeLocalOutputProjection(local_attended_output, local_wo,
                                         local_final_output, seq_len, local_heads, d_model); });
        }
        if (debugEnv().attention.internal_diff && debugEnv().pipeline.layer_token_diff && getRank() == 0 && seq_len > 0)
        {
            const float *src = local_final_output->data() + (seq_len - 1) * d_model;
            QwenPipeline::appendInternalAttnRow(nullptr, layer_index_, capture_seq_len_meta, (seq_len == 1 && n_past_ > 0), "attn_int_out_partial", src, (size_t)d_model);
        }

        bool performed_collective = false;
        if (output_mode_ == AttentionOutputMode::GatherHeadsPostProjection)
        {
            PERF_SCOPED_TIMER("MPIAttentionKernel::gatherHeadsPostProjection");
            // Row-partition of W_o yields additive partial contributions; sum to reconstruct.
            if (getSize() > 1)
            {
                t_gather_post_ms = time_block([&]
                                              { PerfAllreduce(local_final_output->data(), global_output->data(),
                                                              (int)(seq_len * d_model), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD); });
                performed_collective = true;
            }
            else
            {
                t_gather_post_ms = time_block([&]
                                              { std::copy(local_final_output->data(),
                                                          local_final_output->data() + seq_len * d_model,
                                                          global_output->data()); });
            }
            if (getRank() == 0)
                LOG_DEBUG("MPIAttentionKernel: GatherHeadsPostProjection reconstructed full hidden via "
                          << (performed_collective ? "Allreduce (SUM)" : "local copy"));
            // Metadata: full hidden now available & replicated
            last_meta_.concatenated = true; // assembled full hidden dimension
            last_meta_.replicated = true;   // identical across ranks
        }
        else if (output_mode_ == AttentionOutputMode::LocalHeads)
        {
            PERF_SCOPED_TIMER("MPIAttentionKernel::emitLocalPartial");
            t_emit_ms = time_block([&]
                                   { std::copy(local_final_output->data(),
                                               local_final_output->data() + seq_len * d_model,
                                               global_output->data()); });
            if (getShardingDebugConfig().assert_replicated_misuse)
            {
                global_output->data()[seq_len * d_model - 1] += (float)(1e-30 * (getRank() + 1));
            }
            if (getRank() == 0)
                LOG_DEBUG("MPIAttentionKernel: emitted local partial head contribution (LocalHeads mode)");
        }
        else
        {
            // Unsupported modes (pre-gather, replicated) fallback to LocalHeads behavior for now.
            PERF_SCOPED_TIMER("MPIAttentionKernel::emitLocalPartialFallback");
            t_emit_ms = time_block([&]
                                   { std::copy(local_final_output->data(),
                                               local_final_output->data() + seq_len * d_model,
                                               global_output->data()); });
            if (getRank() == 0)
                LOG_WARN("MPIAttentionKernel: output mode not implemented (" << (int)output_mode_ << ") falling back to LocalHeads partial");
        }

        auto end = std::chrono::high_resolution_clock::now();
        double execution_time = std::chrono::duration<double, std::milli>(end - t_exec_start).count();

        LOG_DEBUG("MPIAttention executed in " + std::to_string(execution_time) +
                  " ms on rank " + std::to_string(getRank()) +
                  " (local_heads=" + std::to_string(local_heads) + ")");

        // Populate result metadata including final (possibly heuristic-adjusted) mode.
        last_meta_.mode = output_mode_;
        last_meta_.local_head_offset = head_offset;
        last_meta_.local_head_count = local_heads;
        // If we performed a gather above, metadata already set. Ensure LocalHeads defaults remain false.
        if (output_mode_ == AttentionOutputMode::LocalHeads)
        {
            last_meta_.concatenated = false;
            last_meta_.replicated = false;
        }
        else if (output_mode_ == AttentionOutputMode::GatherHeadsPostProjection && !performed_collective)
        {
            // Single-rank edge case
            last_meta_.concatenated = true;
            last_meta_.replicated = true;
        }
        // (GatherHeadsPreProjection / Replicated) early-returned above with metadata populated.
        if (getRank() == 0)
        {
            LOG_DEBUG("AttentionResultMeta mode=" << (int)last_meta_.mode
                                                  << " heads_off=" << last_meta_.local_head_offset
                                                  << " heads_cnt=" << last_meta_.local_head_count
                                                  << " concat=" << last_meta_.concatenated
                                                  << " repl=" << last_meta_.replicated);
        }
        if (perf_env.layer_attention && getRank() == perf_env.log_rank)
        {
            double accounted = t_distribute_ms + t_proj_ms + t_rope_ms + t_attn_ms + t_output_proj_ms + t_gather_post_ms + t_emit_ms;
            double other = std::max(0.0, execution_time - accounted);
            std::string mode_str = (output_mode_ == AttentionOutputMode::GatherHeadsPostProjection ? "gather_post" : output_mode_ == AttentionOutputMode::LocalHeads ? "local_heads"
                                                                                                                                                                     : "other");
            LOG_INFO("ATTN_LAYER_TIMING seq_len=" << seq_len
                                                  << " heads=" << n_head_
                                                  << " head_dim=" << head_dim_
                                                  << " local_heads=" << local_heads
                                                  << " mode=" << mode_str
                                                  << " distribute_ms=" << t_distribute_ms
                                                  << " proj_ms=" << t_proj_ms
                                                  << " rope_ms=" << t_rope_ms
                                                  << " attn_ms=" << t_attn_ms
                                                  << " out_proj_ms=" << t_output_proj_ms
                                                  << " gather_post_ms=" << t_gather_post_ms
                                                  << " emit_ms=" << t_emit_ms
                                                  << " total_ms=" << execution_time
                                                  << " other_ms=" << other
                                                  << " layer_idx=" << PerformanceCounters::tl_phase_.layer_index);
        }
        return true;
    }

    bool MPIAttentionKernel::validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                      const std::vector<std::shared_ptr<TensorBase>> &outputs) const
    {
        if (inputs.size() != 7)
        {
            LOG_ERROR("MPIAttentionKernel: Expected 7 inputs (input, wq, wk, wv, wo, k_cache, v_cache), got " << inputs.size());
            return false;
        }

        if (outputs.size() != 1)
        {
            LOG_ERROR("MPIAttentionKernel: Expected 1 output, got " << outputs.size());
            return false;
        }

        auto input = inputs[0];
        auto wq = inputs[1];
        auto wk = inputs[2];
        auto wv = inputs[3];
        auto wo = inputs[4];
        auto output = outputs[0];

        if (input->shape().size() != 2)
        {
            LOG_ERROR("MPIAttentionKernel: Input must be 2D [seq_len, d_model], got shape size " << input->shape().size());
            return false;
        }

        size_t seq_len = static_cast<size_t>(input->shape()[0]);
        size_t d_model = static_cast<size_t>(input->shape()[1]);
        size_t total_head_dim = static_cast<size_t>(n_head_ * head_dim_);
        bool wq_head_sharded = false, wk_head_sharded = false, wv_head_sharded = false, wo_head_sharded = false;
        auto chk = [](const std::shared_ptr<TensorBase> &t) -> const ShardSpec *
        {
            if(!t) return nullptr; auto *p = dynamic_cast<ShardedSimpleTensor*>(t.get()); return p ? &p->shard_spec() : nullptr; };
        const ShardSpec *spec_wq = chk(wq);
        if (spec_wq && spec_wq->axis == ShardSpec::Axis::Heads)
            wq_head_sharded = spec_wq->is_sharded();
        const ShardSpec *spec_wk = chk(wk);
        if (spec_wk && spec_wk->axis == ShardSpec::Axis::Heads)
            wk_head_sharded = spec_wk->is_sharded();
        const ShardSpec *spec_wv = chk(wv);
        if (spec_wv && spec_wv->axis == ShardSpec::Axis::Heads)
            wv_head_sharded = spec_wv->is_sharded();
        const ShardSpec *spec_wo = chk(wo);
        if (spec_wo && spec_wo->axis == ShardSpec::Axis::Heads)
            wo_head_sharded = spec_wo->is_sharded();

        bool any_sharded = wq_head_sharded || wk_head_sharded || wv_head_sharded || wo_head_sharded;
        if (any_sharded)
        {
            // For head-sharded mode accept local dimensions instead of global aggregate.
            auto [local_heads, head_offset] = getHeadDistribution();
            size_t local_head_dim = static_cast<size_t>(local_heads * head_dim_);
            if (wq_head_sharded)
            {
                if (wq->shape().size() != 2 || (size_t)wq->shape()[0] != d_model || (size_t)wq->shape()[1] != local_head_dim)
                {
                    LOG_ERROR("MPIAttentionKernel: pre-sharded wq shape mismatch");
                    return false;
                }
            }
            else if (wq->shape().size() != 2 || (size_t)wq->shape()[0] != d_model || (size_t)wq->shape()[1] != total_head_dim)
            {
                LOG_ERROR("MPIAttentionKernel: Query weight dimension mismatch");
                return false;
            }
            if (wk_head_sharded)
            {
                if (wk->shape().size() != 2 || (size_t)wk->shape()[0] != d_model || (size_t)wk->shape()[1] != local_head_dim)
                {
                    LOG_ERROR("MPIAttentionKernel: pre-sharded wk shape mismatch");
                    return false;
                }
            }
            else if (wk->shape().size() != 2 || (size_t)wk->shape()[0] != d_model || (size_t)wk->shape()[1] != (size_t)(n_head_kv_ * head_dim_))
            {
                LOG_ERROR("MPIAttentionKernel: Key weight dimension mismatch");
                return false;
            }
            if (wv_head_sharded)
            {
                if (wv->shape().size() != 2 || (size_t)wv->shape()[0] != d_model || (size_t)wv->shape()[1] != local_head_dim)
                {
                    LOG_ERROR("MPIAttentionKernel: pre-sharded wv shape mismatch");
                    return false;
                }
            }
            else if (wv->shape().size() != 2 || (size_t)wv->shape()[0] != d_model || (size_t)wv->shape()[1] != (size_t)(n_head_kv_ * head_dim_))
            {
                LOG_ERROR("MPIAttentionKernel: Value weight dimension mismatch");
                return false;
            }
            if (wo_head_sharded)
            {
                if (wo->shape().size() != 2 || (size_t)wo->shape()[0] != local_head_dim || (size_t)wo->shape()[1] != d_model)
                {
                    LOG_ERROR("MPIAttentionKernel: pre-sharded wo shape mismatch");
                    return false;
                }
            }
            else if (wo->shape().size() != 2 || (size_t)wo->shape()[0] != total_head_dim || (size_t)wo->shape()[1] != d_model)
            {
                LOG_ERROR("MPIAttentionKernel: Output weight dimension mismatch");
                return false;
            }
        }
        else
        {
            // Original global dimension checks (unchanged)
            if (wq->shape().size() != 2 || static_cast<size_t>(wq->shape()[0]) != d_model ||
                static_cast<size_t>(wq->shape()[1]) != total_head_dim)
            {
                LOG_ERROR("MPIAttentionKernel: Query weight dimension mismatch");
                return false;
            }
            if (wk->shape().size() != 2 || static_cast<size_t>(wk->shape()[0]) != d_model ||
                static_cast<size_t>(wk->shape()[1]) != static_cast<size_t>(n_head_kv_ * head_dim_))
            {
                LOG_ERROR("MPIAttentionKernel: Key weight dimension mismatch");
                return false;
            }
            if (wv->shape().size() != 2 || static_cast<size_t>(wv->shape()[0]) != d_model ||
                static_cast<size_t>(wv->shape()[1]) != static_cast<size_t>(n_head_kv_ * head_dim_))
            {
                LOG_ERROR("MPIAttentionKernel: Value weight dimension mismatch");
                return false;
            }
            if (wo->shape().size() != 2 || static_cast<size_t>(wo->shape()[0]) != total_head_dim ||
                static_cast<size_t>(wo->shape()[1]) != d_model)
            {
                LOG_ERROR("MPIAttentionKernel: Output weight dimension mismatch");
                return false;
            }
        }

        // (Removed duplicate global-dimension validation for sharded path.)

        // Validate output dimensions
        if (output->shape().size() != 2 || static_cast<size_t>(output->shape()[0]) != seq_len ||
            static_cast<size_t>(output->shape()[1]) != d_model)
        {
            LOG_ERROR("MPIAttentionKernel: Output dimension mismatch");
            return false;
        }

        return true;
    }

    void MPIAttentionKernel::setHeadDimensions(int n_head, int n_head_kv, int head_dim)
    {
        n_head_ = n_head;
        n_head_kv_ = n_head_kv;
        head_dim_ = head_dim;

        if (n_head_ % getSize() != 0)
        {
            LOG_WARN("Number of heads (" << n_head_ << ") not evenly divisible by MPI size ("
                                         << getSize() << "). Load balancing may be suboptimal.");
        }
    }

    std::pair<int, int> MPIAttentionKernel::getHeadDistribution() const
    {
        return getHeadDistribution(getRank());
    }

    std::pair<int, int> MPIAttentionKernel::getHeadDistribution(int rank) const
    {
        int heads_per_rank = n_head_ / getSize();
        int remainder = n_head_ % getSize();

        int local_heads = heads_per_rank + (rank < remainder ? 1 : 0);
        int head_offset = rank * heads_per_rank + std::min(rank, remainder);

        return {local_heads, head_offset};
    }

    void MPIAttentionKernel::distributeInputs(const std::shared_ptr<TensorBase> &global_input,
                                              const std::shared_ptr<TensorBase> &global_wq,
                                              const std::shared_ptr<TensorBase> &global_wk,
                                              const std::shared_ptr<TensorBase> &global_wv,
                                              const std::shared_ptr<TensorBase> &global_wo,
                                              std::shared_ptr<TensorBase> &local_wq,
                                              std::shared_ptr<TensorBase> &local_wk,
                                              std::shared_ptr<TensorBase> &local_wv,
                                              std::shared_ptr<TensorBase> &local_wo,
                                              size_t seq_len, size_t d_model)
    {
        auto [local_heads, head_offset] = getHeadDistribution();
        size_t local_head_dim = static_cast<size_t>(local_heads * head_dim_);
        size_t head_offset_dim = static_cast<size_t>(head_offset * head_dim_);

        const float *global_wq_ptr = global_wq ? global_wq->data() : nullptr;
        const float *global_wk_ptr = global_wk ? global_wk->data() : nullptr;
        const float *global_wv_ptr = global_wv ? global_wv->data() : nullptr;
        const float *global_wo_ptr = global_wo ? global_wo->data() : nullptr;
        float *local_wq_ptr = local_wq ? local_wq->data() : nullptr;
        float *local_wk_ptr = local_wk ? local_wk->data() : nullptr;
        float *local_wv_ptr = local_wv ? local_wv->data() : nullptr;
        float *local_wo_ptr = local_wo ? local_wo->data() : nullptr;

        auto require_data = [&](const char *name, const float *ptr)
        {
            if (ptr)
                return;
            LOG_ERROR("MPIAttentionKernel::distributeInputs null data pointer for " << name << " on rank " << getRank());
            throw std::runtime_error("Null tensor data pointer");
        };
        auto require_data_mut = [&](const char *name, float *ptr)
        {
            if (ptr)
                return;
            LOG_ERROR("MPIAttentionKernel::distributeInputs null writable pointer for " << name << " on rank " << getRank());
            throw std::runtime_error("Null tensor data pointer");
        };

        require_data("global_wq", global_wq_ptr);
        require_data("global_wk", global_wk_ptr);
        require_data("global_wv", global_wv_ptr);
        require_data("global_wo", global_wo_ptr);
        require_data_mut("local_wq", local_wq_ptr);
        require_data_mut("local_wk", local_wk_ptr);
        require_data_mut("local_wv", local_wv_ptr);
        require_data_mut("local_wo", local_wo_ptr);

        // Extract local query weights (columns for assigned heads)
        for (size_t i = 0; i < d_model; ++i)
        {
            const float *global_row = global_wq_ptr + i * n_head_ * head_dim_;
            float *local_row = local_wq_ptr + i * local_head_dim;
            memcpy(local_row, global_row + head_offset_dim, local_head_dim * sizeof(float));
        }

        // SIMPLE FIX: For grouped attention where n_head_kv != n_head
        // Just replicate the available KV heads to match the local Q heads
        // This is not optimal but prevents buffer overruns and NaN values

        for (size_t i = 0; i < d_model; ++i)
        {
            const float *global_k_row = global_wk_ptr + i * n_head_kv_ * head_dim_;
            const float *global_v_row = global_wv_ptr + i * n_head_kv_ * head_dim_;
            float *local_k_row = local_wk_ptr + i * local_head_dim;
            float *local_v_row = local_wv_ptr + i * local_head_dim;

            // For each local Q head, assign the corresponding KV head
            // Use modulo to handle the case where we have more Q heads than KV heads
            for (int local_head = 0; local_head < local_heads; ++local_head)
            {
                int global_q_head = head_offset + local_head;
                int kv_head = global_q_head % n_head_kv_; // Map Q head to KV head

                const float *src_k = global_k_row + kv_head * head_dim_;
                const float *src_v = global_v_row + kv_head * head_dim_;
                float *dst_k = local_k_row + local_head * head_dim_;
                float *dst_v = local_v_row + local_head * head_dim_;

                memcpy(dst_k, src_k, head_dim_ * sizeof(float));
                memcpy(dst_v, src_v, head_dim_ * sizeof(float));
            }
        }

        // Extract local output weights (rows for assigned heads)
        for (size_t i = 0; i < local_head_dim; ++i)
        {
            const float *global_row = global_wo_ptr + (head_offset_dim + i) * d_model;
            float *local_row = local_wo_ptr + i * d_model;
            memcpy(local_row, global_row, d_model * sizeof(float));
        }

        LOG_DEBUG("Distributed weights: local_heads=" << local_heads << ", head_offset=" << head_offset << " on rank " << getRank());
    }

    void MPIAttentionKernel::computeLocalProjections(const std::shared_ptr<TensorBase> &input,
                                                     const std::shared_ptr<TensorBase> &local_wq,
                                                     const std::shared_ptr<TensorBase> &local_wk,
                                                     const std::shared_ptr<TensorBase> &local_wv,
                                                     std::shared_ptr<TensorBase> &local_q,
                                                     std::shared_ptr<TensorBase> &local_k,
                                                     std::shared_ptr<TensorBase> &local_v,
                                                     size_t seq_len, size_t d_model)
    {
        const auto &attnEnv = debugEnv().attention;
        bool force_scalar = attnEnv.force_scalar;
        bool validate_proj = attnEnv.validate_proj;
        auto scalar_matmul = [](const float *A, const float *B, float *C, size_t M, size_t N, size_t K)
        {
            for (size_t i = 0; i < M; ++i)
            {
                const float *a_row = A + i * K;
                float *c_row = C + i * N;
                for (size_t j = 0; j < N; ++j)
                    c_row[j] = 0.f;
                for (size_t k = 0; k < K; ++k)
                {
                    float a = a_row[k];
                    const float *b_col = B + k * N;
                    for (size_t j = 0; j < N; ++j)
                        c_row[j] += a * b_col[j];
                }
            }
        };

        auto do_projection = [&](const char *tag, const std::shared_ptr<TensorBase> &W, std::shared_ptr<TensorBase> &OUT, int phase_id)
        {
            auto [lh, _ho] = getHeadDistribution();
            size_t local_head_dim = static_cast<size_t>(lh * head_dim_);
            if (force_scalar)
            {
                scalar_matmul(input->data(), W->data(), OUT->data(), seq_len, local_head_dim, d_model);
            }
            else
            {
                PerfMatmulPhaseScope phase(2, phase_id);
                bool prefill_like = seq_len >= (size_t)debugEnv().cosma.prefill_threshold;
                if (prefill_like)
                {
                    static auto prefill_backend = PrefillBackendFactory::create();
                    PrefillOpDesc desc;
                    desc.kind = PrefillOpKind::MatMul;
                    desc.M = seq_len;
                    desc.N = local_head_dim;
                    desc.K = d_model;
                    desc.is_prefill = true;
                    PrefillLaunchContext ctx{input->data(), W->data(), OUT->data()};
                    auto decision = prefill_backend->launch(desc, ctx);
                    if (decision.status != PrefillStatus::Success)
                    {
                        LOG_WARN(tag << " projection abstraction fallback status=" << (int)decision.status << " reason=" << decision.reason);
                        if (!adaptive_matmul(input->data(), W->data(), OUT->data(), seq_len, local_head_dim, d_model, false))
                        {
                            LOG_ERROR(tag << " projection failed on rank " << getRank());
                            return false;
                        }
                    }
                }
                else
                {
                    static auto infer_backend = InferenceBackendFactory::create();
                    InferenceOpDesc desc;
                    desc.kind = InferenceOpKind::MatMul;
                    desc.M = seq_len;
                    desc.N = local_head_dim;
                    desc.K = d_model;
                    desc.latency_critical = true;
                    InferenceLaunchContext ctx{input->data(), W->data(), OUT->data()};
                    auto decision = infer_backend->launch(desc, ctx);
                    if (decision.status != InferenceStatus::Success)
                    {
                        LOG_WARN(tag << " projection inference fallback status=" << (int)decision.status << " reason=" << decision.reason);
                        if (!adaptive_matmul(input->data(), W->data(), OUT->data(), seq_len, local_head_dim, d_model, false))
                        {
                            LOG_ERROR(tag << " projection failed on rank " << getRank());
                            return false;
                        }
                    }
                }
            }
            if (validate_proj || force_scalar)
            {
                auto [lh2, _] = getHeadDistribution();
                size_t lhd2 = (size_t)lh2 * head_dim_;
                std::vector<float> ref(seq_len * lhd2);
                scalar_matmul(input->data(), W->data(), ref.data(), seq_len, lhd2, d_model);
                // Lightweight diff metrics (log only)
                double max_abs = 0.0, sum_sq_ref = 0.0, sum_sq_diff = 0.0;
                const float *dst = OUT->data();
                for (size_t i = 0; i < ref.size(); ++i)
                {
                    double r = ref[i];
                    double g = dst[i];
                    double d = std::fabs(r - g);
                    if (d > max_abs)
                        max_abs = d;
                    sum_sq_ref += r * r;
                    sum_sq_diff += d * d;
                }
                double rel_l2 = (sum_sq_ref == 0.0) ? 0.0 : std::sqrt(sum_sq_diff / sum_sq_ref);
                if (max_abs > 1e-5 || rel_l2 > 1e-5)
                {
                    if (getRank() == 0)
                        LOG_WARN("Projection validation diverged tag=" << tag << " max_abs=" << max_abs << " rel_l2=" << rel_l2);
                }
            }
            return true;
        };

        // Q, K, V
        if (!do_projection("Q", local_wq, local_q, 1))
            return;
        if (!do_projection("K", local_wk, local_k, 2))
            return;
        if (!do_projection("V", local_wv, local_v, 3))
            return;
        auto [lh_final, _off] = getHeadDistribution();
        LOG_DEBUG("Computed local projections using COSMA/adaptive for " << lh_final << " heads on rank " << getRank());
    }

    void MPIAttentionKernel::computeLocalAttention(const float *local_q, const float *local_k, const float *local_v,
                                                   float *local_output, size_t seq_len, int local_heads)
    {
        // Create temporary storage for attention scores
        size_t scores_size = seq_len * seq_len * static_cast<size_t>(local_heads);
        auto scores = std::make_unique<float[]>(scores_size);

        // Compute attention scores and apply softmax
        computeLocalAttentionScores(local_q, local_k, scores.get(), seq_len, local_heads);

        // Apply attention to values
        applyLocalAttention(scores.get(), local_v, local_output, seq_len, local_heads);

        // Optional debug validation: re-run scalar reference primitives on already RoPE-rotated Q/K
        // and compare with kernel path. Enabled when LLAMINAR_ATTN_PRIMITIVES_VALIDATE is set.
        if (debugEnv().attention.validate_primitives)
        {
            // We intentionally perform this only on rank 0 to avoid duplicated logs.
            if (getRank() == 0)
            {
                // Primitive layout matches: sequence-major rows, heads contiguous per row.
                // local_q/k/v layout here is [seq_len, local_heads * head_dim_].
                // Build a temporary contiguous copy so we can treat as primitives input.
                size_t frame_elems = seq_len * static_cast<size_t>(local_heads) * head_dim_;
                std::vector<float> q_copy(frame_elems), k_copy(frame_elems), v_copy(frame_elems);
                std::memcpy(q_copy.data(), local_q, frame_elems * sizeof(float));
                std::memcpy(k_copy.data(), local_k, frame_elems * sizeof(float));
                std::memcpy(v_copy.data(), local_v, frame_elems * sizeof(float));

                // Fused primitive attention (always causal in kernel today)
                std::vector<float> fused_out(frame_elems, 0.f);
                llaminar::attn::fused_attention(q_copy.data(), k_copy.data(), v_copy.data(), fused_out.data(),
                                                static_cast<int>(seq_len), head_dim_, local_heads, /*causal=*/true);

                // Compute simple diff metrics
                double max_abs = 0.0, sum_sq_ref = 0.0, sum_sq_diff = 0.0;
                for (size_t i = 0; i < frame_elems; ++i)
                {
                    double ref = fused_out[i];
                    double got = local_output[i];
                    double diff = std::fabs(ref - got);
                    if (diff > max_abs)
                        max_abs = diff;
                    sum_sq_ref += ref * ref;
                    sum_sq_diff += diff * diff;
                }
                double rel_l2 = (sum_sq_ref == 0.0) ? 0.0 : std::sqrt(sum_sq_diff / sum_sq_ref);
                if (max_abs > 1e-5 || rel_l2 > 1e-5)
                {
                    LOG_WARN("MPIAttentionKernel primitives validation divergence max_abs=" << max_abs
                                                                                            << " rel_l2=" << rel_l2
                                                                                            << " (seq_len=" << seq_len << ", local_heads=" << local_heads << ")");
                }
                else
                {
                    LOG_DEBUG("MPIAttentionKernel primitives validation OK max_abs=" << max_abs
                                                                                     << " rel_l2=" << rel_l2);
                }
            }
        }

        LOG_DEBUG("Computed local attention for " << local_heads << " heads on rank " << getRank());
    }

    void MPIAttentionKernel::applyLocalRoPE(float *local_q, float *local_k, size_t seq_len, int local_heads)
    {
        // Primitive applies in-place RoPE to Q/K (causal offset n_past_)
        const auto &env = debugEnv();
        if (getRank() == 0)
        {
            static int gate_count = 0;
            if (gate_count < 8)
            {
                LOG_WARN("[RoPEDiagGate] internal_diff=" << env.attention.internal_diff
                                                         << " layer_token_diff=" << env.pipeline.layer_token_diff
                                                         << " layer_replay_compare=" << env.pipeline.layer_replay_compare
                                                         << " seq_len=" << seq_len << " n_past=" << n_past_ << " local_heads=" << local_heads);
                ++gate_count;
            }
        }
        // We want diagnostics even if replay compare snapshot not yet set when called from replay pipeline.
        bool diag = env.attention.internal_diff && getRank() == 0;
        if (diag)
        {
            static int invoke_count = 0;
            ++invoke_count;
            if (invoke_count <= 12)
            {
                LOG_INFO("[RoPEDiagInvoke] call=" << invoke_count << " seq_len=" << seq_len << " n_past=" << n_past_ << " local_heads=" << local_heads << " head_dim=" << head_dim_ << " layer_replay_compare=" << (env.pipeline.layer_replay_compare ? 1 : 0));
            }
        }
        if (!env.pipeline.layer_replay_compare && diag)
        {
            // Lightweight marker to confirm gating condition; only emits once per run per rank due to static.
            static bool warned = false;
            if (!warned)
            {
                LOG_INFO("[RoPEDiag] note=replay_compare_flag_off but collecting due to internal_diff");
                warned = true;
            }
        }
        // Determine whether this invocation is incremental single-token (seq_len==1 with history)
        bool incr = (seq_len == 1 && n_past_ > 0);
        int head_dim = head_dim_;
        int preview = std::min(head_dim * local_heads, 8); // first few floats across first head slice
        // Capture BEFORE snapshot for first head last token only (the token whose rotation parity we care about)
        std::array<float, 8> q_before{};
        std::array<float, 8> k_before{};
        q_before.fill(0.f);
        k_before.fill(0.f);
        if (diag && seq_len > 0 && preview > 0)
        {
            // Layout: contiguous [seq_len, local_heads*head_dim]
            size_t row_offset = (seq_len - 1) * (size_t)local_heads * head_dim; // last token row inside local_q/k
            const float *q_row = local_q + row_offset;
            const float *k_row = local_k + row_offset;
            for (int i = 0; i < preview; ++i)
            {
                q_before[i] = q_row[i];
                k_before[i] = k_row[i];
            }
        }

        llaminar::attn::apply_rope(local_q, local_k, (int)seq_len, head_dim_, local_heads, (int)n_past_, rope_freq_base_);

        if (diag && seq_len > 0)
        {
            if (preview == 0)
            {
                LOG_INFO("[RoPEDiag] mode=" << (incr ? "INC" : "REPLAY")
                                            << " n_past=" << n_past_ << " seq_len=" << seq_len
                                            << " heads_local=" << local_heads << " head_dim=" << head_dim
                                            << " note=preview_zero_unexpected");
            }
            else
            {
                std::array<float, 8> q_after{};
                std::array<float, 8> k_after{};
                q_after.fill(0.f);
                k_after.fill(0.f);
                size_t row_offset = (seq_len - 1) * (size_t)local_heads * head_dim; // last token row
                const float *q_row_after = local_q + row_offset;
                const float *k_row_after = local_k + row_offset;
                long double l2_q_b = 0, l2_q_a = 0, l2_k_b = 0, l2_k_a = 0, l2_q_delta = 0, l2_k_delta = 0;
                for (int i = 0; i < preview; ++i)
                {
                    q_after[i] = q_row_after[i];
                    k_after[i] = k_row_after[i];
                    long double qb = q_before[i], qa = q_after[i];
                    long double kb = k_before[i], ka = k_after[i];
                    l2_q_b += qb * qb;
                    l2_q_a += qa * qa;
                    l2_k_b += kb * kb;
                    l2_k_a += ka * ka;
                    long double dq = qa - qb;
                    long double dk = ka - kb;
                    l2_q_delta += dq * dq;
                    l2_k_delta += dk * dk;
                }
                double rel_move_q = (l2_q_b > 0) ? std::sqrt((double)l2_q_delta / (double)l2_q_b) : 0.0;
                double rel_move_k = (l2_k_b > 0) ? std::sqrt((double)l2_k_delta / (double)l2_k_b) : 0.0;
                int pos_last = (int)n_past_ + (int)seq_len - 1; // semantic position of this final token
                // Recompute first two angles used (pair 0) for transparency
                int pairs = head_dim / 2;
                float theta0 = (pairs > 0) ? (1.f / std::pow(rope_freq_base_, (2.f * 0) / head_dim)) : 0.f;
                float angle_pos = theta0 * pos_last;
                float cs = std::cos(angle_pos), sn = std::sin(angle_pos);
                std::ostringstream oss;
                oss << "[RoPEDiag] mode=" << (incr ? "INC" : "REPLAY")
                    << " n_past=" << n_past_ << " seq_len=" << seq_len
                    << " pos_last=" << pos_last
                    << " heads_local=" << local_heads
                    << " head_dim=" << head_dim
                    << " freq_base=" << rope_freq_base_
                    << " theta0=" << theta0
                    << " angle0=" << angle_pos
                    << " cos0=" << cs << " sin0=" << sn
                    << " q_before=";
                for (int i = 0; i < preview; ++i)
                {
                    oss << q_before[i];
                    if (i + 1 < preview)
                        oss << ",";
                }
                oss << " q_after=";
                for (int i = 0; i < preview; ++i)
                {
                    oss << q_after[i];
                    if (i + 1 < preview)
                        oss << ",";
                }
                oss << " rel_move_q=" << rel_move_q
                    << " k_before=";
                for (int i = 0; i < preview; ++i)
                {
                    oss << k_before[i];
                    if (i + 1 < preview)
                        oss << ",";
                }
                oss << " k_after=";
                for (int i = 0; i < preview; ++i)
                {
                    oss << k_after[i];
                    if (i + 1 < preview)
                        oss << ",";
                }
                oss << " rel_move_k=" << rel_move_k;
                LOG_INFO(oss.str());
            }
        }
        LOG_DEBUG("Applied RoPE for " << local_heads << " heads on rank " << getRank());
    }

    void MPIAttentionKernel::computeLocalAttentionScores(const float *local_q, const float *local_k, float *scores,
                                                         size_t seq_len, int local_heads)
    {
        llaminar::attn::compute_qk_scores(local_q, local_k, scores, (int)seq_len, head_dim_, local_heads, /*causal=*/true, /*apply_softmax=*/true);
        LOG_DEBUG("Computed local attention scores for " << local_heads << " heads on rank " << getRank());
    }

    void MPIAttentionKernel::applyLocalAttention(const float *scores, const float *local_v, float *local_attended_output,
                                                 size_t seq_len, int local_heads)
    {
        const auto &env = debugEnv();
        llaminar::attn::apply_scores_to_v(scores, local_v, local_attended_output, (int)seq_len, head_dim_, local_heads);
        if (env.attention.dump_attention && getRank() == 0 && seq_len <= 4 && local_heads <= 2)
        {
            std::cerr << "[AttnApplyDump] head0 row0:";
            for (size_t j = 0; j < std::min(seq_len, (size_t)8); ++j)
                std::cerr << ' ' << scores[j];
            std::cerr << '\n';
        }
        LOG_DEBUG("Applied attention for " << local_heads << " heads on rank " << getRank());
    }

    void MPIAttentionKernel::computeLocalOutputProjection(const std::shared_ptr<TensorBase> &local_attended_output,
                                                          const std::shared_ptr<TensorBase> &local_wo,
                                                          std::shared_ptr<TensorBase> &local_final_output,
                                                          size_t seq_len, int local_heads, size_t d_model)
    {
        size_t local_head_dim = static_cast<size_t>(local_heads * head_dim_);
        bool tp_sim_done = false; // if simulation path executes successfully, skip baseline path

        // ---------------------------------------------------------------------
        // Tensor-Parallel (simulation) path (Task 2)
        // When enabled via tp_sim.* env snapshot we synthesize an intra-process
        // tensor parallel execution of the output projection, partitioning either
        // rows (M dimension / sequence length) or columns (N / d_model) and
        // reconstructing the full output. This is a correctness exploration tool
        // that does not (yet) overlap or distribute work across MPI ranks.
        // ---------------------------------------------------------------------
        const auto &tpSim = debugEnv().tp_sim;
        if (tpSim.enable && tpSim.partitions > 1)
        {
            PERF_SCOPED_TIMER("OutputProjection::tp_sim");
            int parts = tpSim.partitions;
            bool row_split = false;
            switch (tpSim.mode)
            {
            case DebugEnvSnapshot::TPSimEnv::Mode::Row:
                row_split = true;
                break;
            case DebugEnvSnapshot::TPSimEnv::Mode::Col:
                row_split = false;
                break;
            case DebugEnvSnapshot::TPSimEnv::Mode::Auto:
            default:
                // Heuristic: prefer column split if d_model divisible, else row if seq_len divisible, else column fallback.
                if (d_model % static_cast<size_t>(parts) == 0)
                    row_split = false;
                else if (seq_len % static_cast<size_t>(parts) == 0)
                    row_split = true;
                else
                    row_split = false;
                break;
            }

            auto matmul_fn = [&](const float *A, const float *B, float *C, std::size_t M, std::size_t N, std::size_t K) -> bool
            {
                return adaptive_matmul(A, B, C, (int)M, (int)N, (int)K, false);
            };

            // Build executor configs for each simulated partition and run sequentially (single process simulation).
            std::vector<TPOutputLocalResult> local_parts;
            local_parts.reserve(parts);
            bool failed = false;
            for (int p = 0; p < parts; ++p)
            {
                TPOutputExecConfig cfg;
                cfg.tp_size = parts;
                cfg.tp_rank = p;
                cfg.row_split = row_split;
                try
                {
                    TPOutputProjectionExecutor exec(matmul_fn, cfg, seq_len, d_model, local_head_dim);
                    local_parts.push_back(exec.run(local_attended_output->data(), local_wo->data()));
                }
                catch (const std::exception &e)
                {
                    failed = true;
                    if (getRank() == 0)
                        LOG_ERROR("[TP-Sim] partition execution failed p=" << p << " err=" << e.what());
                    break;
                }
            }
            if (!failed)
            {
                // Reconstruct into final output buffer.
                try
                {
                    if (row_split)
                        TPOutputProjectionExecutor::reconstruct_rows(local_parts, local_final_output->data(), seq_len, d_model);
                    else
                        TPOutputProjectionExecutor::reconstruct_columns(local_parts, local_final_output->data(), seq_len, d_model);
                    if (getRank() == 0)
                        LOG_DEBUG("[TP-Sim] output projection reconstructed mode=" << (row_split ? "row" : "col")
                                                                                   << " parts=" << parts
                                                                                   << " seq_len=" << seq_len
                                                                                   << " d_model=" << d_model
                                                                                   << " local_head_dim=" << local_head_dim);
                }
                catch (const std::exception &e)
                {
                    failed = true;
                    if (getRank() == 0)
                        LOG_ERROR("[TP-Sim] reconstruction failed: " << e.what());
                }
            }
            if (!failed)
            {
                // Optional: reference scalar validation may still run below if attention.validate_output enabled.
                if (debugEnv().attention.validate_output && getRank() == 0)
                    LOG_DEBUG("[TP-Sim] validation (scalar ref) will run after simulation path");
                tp_sim_done = true; // ensure baseline logic skipped
            }
            else
            {
                // Fallback: execute baseline path if simulation failed.
                if (getRank() == 0)
                    LOG_WARN("[TP-Sim] falling back to baseline single-path output projection due to prior errors");
                // Continue into baseline logic below (no early return).
            }
            // NOTE: We intentionally do not 'return' here so scalar validation below still applies.
            // Baseline logic guarded further down by absence of simulation gating state.
        }
        if (!tp_sim_done)
        {
            const auto &attnEnv3 = debugEnv().attention;
            int tp_parts = attnEnv3.tp_partitions;
            bool tp_disable = attnEnv3.tp_disable;
            bool auto_escalated = false;
            if (!tp_disable && tp_parts == 1 && attnEnv3.tp_auto && d_model >= 2048 && (d_model % 2 == 0))
            {
                tp_parts = 2;
                auto_escalated = true;
            }
            size_t op_elems_proxy = seq_len * d_model * local_head_dim; // approximate cost proxy
            if (auto_escalated && op_elems_proxy < 4096ULL)
            {
                // Revert tiny auto-escalations – explicit user tp_partitions still honored.
                tp_parts = 1;
                auto_escalated = false;
            }
            bool use_tp = (tp_parts > 1) && !tp_disable;
            if (!use_tp)
            {
                PERF_SCOPED_TIMER("OutputProjection::single_gemm");
                {
                    bool is_prefill_like = seq_len >= static_cast<size_t>(debugEnv().cosma.prefill_threshold);
                    if (is_prefill_like)
                    {
                        static auto prefill_backend = PrefillBackendFactory::create();
                        PrefillOpDesc desc;
                        desc.kind = PrefillOpKind::MatMul;
                        desc.M = seq_len;
                        desc.N = d_model;
                        desc.K = local_head_dim;
                        desc.is_prefill = true;
                        PrefillLaunchContext ctx{local_attended_output->data(), local_wo->data(), local_final_output->data()};
                        auto decision = prefill_backend->launch(desc, ctx);
                        if (decision.status != PrefillStatus::Success)
                        {
                            LOG_WARN("Output projection prefill fallback status=" << (int)decision.status << " reason=" << decision.reason);
                            if (!adaptive_matmul(local_attended_output->data(), local_wo->data(), local_final_output->data(), seq_len, d_model, local_head_dim, false))
                            {
                                LOG_ERROR("Output projection failed on rank " << getRank());
                                return;
                            }
                        }
                    }
                    else
                    {
                        static auto infer_backend = InferenceBackendFactory::create();
                        InferenceOpDesc desc;
                        desc.kind = InferenceOpKind::MatMul;
                        desc.M = seq_len;
                        desc.N = d_model;
                        desc.K = local_head_dim;
                        desc.latency_critical = true;
                        InferenceLaunchContext ctx{local_attended_output->data(), local_wo->data(), local_final_output->data()};
                        auto decision = infer_backend->launch(desc, ctx);
                        if (decision.status != InferenceStatus::Success)
                        {
                            LOG_WARN("Output projection inference fallback status=" << (int)decision.status << " reason=" << decision.reason);
                            if (!adaptive_matmul(local_attended_output->data(), local_wo->data(), local_final_output->data(), seq_len, d_model, local_head_dim, false))
                            {
                                LOG_ERROR("Output projection failed on rank " << getRank());
                                return;
                            }
                        }
                    }
                }
            }
            else
            {
                PERF_SCOPED_TIMER("OutputProjection::tp_partition");
                float *C = local_final_output->data();
                const float *B = local_wo->data();
                auto do_part = [&](int part)
                {
                    auto spec = compute_tp_partition(d_model, tp_parts, part, TPPartitionSpec::Axis::Col);
                    int n_sub = (int)spec.local_dim;
                    int col_off = (int)spec.local_offset;
                    const float *B_sub = B + col_off;
                    float *C_sub = C + col_off;
                    if (!adaptive_matmul(local_attended_output->data(), B_sub, C_sub, seq_len, n_sub, local_head_dim, false))
                    {
                        LOG_ERROR("[TP] partition projection failed part=" << part);
                    }
                };
                for (int p = 0; p < tp_parts; ++p)
                    do_part(p);
            }
            // Logging only if baseline executed
            if (use_tp)
                LOG_DEBUG("Computed output projection (TP col) heads=" << local_heads << " parts=" << tp_parts << " rank=" << getRank());
            else
                LOG_DEBUG("Computed output projection (single GEMM) heads=" << local_heads << " rank=" << getRank());
        }

        // Optional scalar reference validation for output projection (pre-gather).
        // Enabled via LLAMINAR_ATTN_OUTPUT_VALIDATE=1 (rank 0 logs divergences >1e-6).
        if (debugEnv().attention.validate_output)
        {
            // Recompute output projection with naive scalar matmul to validate adaptive_matmul path.
            // local_attended_output: [seq_len, local_head_dim]
            // local_wo: [local_head_dim, d_model]
            size_t local_head_dim_e = local_head_dim;
            std::vector<float> ref(seq_len * d_model, 0.f);
            const float *A = local_attended_output->data();
            const float *B = local_wo->data();
            for (size_t i = 0; i < seq_len; ++i)
            {
                const float *a_row = A + i * local_head_dim_e;
                float *c_row = ref.data() + i * d_model;
                for (size_t k = 0; k < local_head_dim_e; ++k)
                {
                    float aval = a_row[k];
                    const float *b_col = B + k * d_model;
                    for (size_t j = 0; j < d_model; ++j)
                    {
                        c_row[j] += aval * b_col[j];
                    }
                }
            }
            double max_abs = 0.0, sum_sq_ref = 0.0, sum_sq_diff = 0.0;
            const float *C = local_final_output->data();
            for (size_t idx = 0; idx < ref.size(); ++idx)
            {
                double r = ref[idx];
                double g = C[idx];
                double d = std::fabs(r - g);
                if (d > max_abs)
                    max_abs = d;
                sum_sq_ref += r * r;
                sum_sq_diff += d * d;
            }
            double rel_l2 = (sum_sq_ref == 0.0) ? 0.0 : std::sqrt(sum_sq_diff / sum_sq_ref);
            if (getRank() == 0)
            {
                if (max_abs > 1e-6 || rel_l2 > 1e-6)
                {
                    LOG_WARN("MPIAttentionKernel output projection validation divergence max_abs=" << max_abs
                                                                                                   << " rel_l2=" << rel_l2
                                                                                                   << " (seq_len=" << seq_len << ", local_heads=" << local_heads << ")");
                }
                else
                {
                    LOG_DEBUG("MPIAttentionKernel output projection validation OK max_abs=" << max_abs
                                                                                            << " rel_l2=" << rel_l2);
                }
            }
        }

        if (tp_sim_done && getRank() == 0)
            LOG_DEBUG("[TP-Sim] output projection complete (validation stage passed)");
    }

    std::shared_ptr<TensorBase> MPIAttentionKernel::createLocalSimpleTensor(const std::vector<size_t> &shape) const
    {
        std::vector<int> int_shape(shape.begin(), shape.end());
        return TensorFactory::create_simple(int_shape);
    }

} // namespace llaminar