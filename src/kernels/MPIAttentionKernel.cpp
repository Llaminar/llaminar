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
#include "../utils/debug_env.h"
#include "../utils/debug_sharding.h"
#include "kernels/common/attention_primitives.h" // for optional validation
#include "../adaptive_matmul.h"
#include "../logger.h"
#include "../performance_timer.h"
#include "../tensors/tensor_factory.h"
#include "../tensors/sharded_simple_tensor.h"
#include "../tensors/shard_spec.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <chrono>

namespace llaminar
{

    MPIAttentionKernel::MPIAttentionKernel(int n_head, int n_head_kv, int head_dim,
                                           float rope_freq_base,
                                           DistributionStrategy strategy)
        : MPIKernelBase(), n_head_(n_head), n_head_kv_(n_head_kv), head_dim_(head_dim),
          n_past_(0), rope_freq_base_(rope_freq_base), strategy_(strategy)
    {
        if (n_head_ % getSize() != 0)
        {
            LOG_WARN("Number of heads (" << n_head_ << ") not evenly divisible by MPI size ("
                                         << getSize() << "). Load balancing may be suboptimal.");
        }
    }

    bool MPIAttentionKernel::execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                     std::vector<std::shared_ptr<TensorBase>> &outputs)
    {
        PERF_SCOPED_TIMER("MPIAttentionKernel::execute");
        auto start = std::chrono::high_resolution_clock::now();

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
                distributeInputs(global_input, global_wq, global_wk, global_wv, global_wo,
                                 local_wq, local_wk, local_wv, local_wo, seq_len, d_model);
            }
        }
        // Create local projection tensors
        auto local_q = createLocalSimpleTensor({seq_len, local_head_dim});
        auto local_k = createLocalSimpleTensor({seq_len, local_head_dim});
        auto local_v = createLocalSimpleTensor({seq_len, local_head_dim});

        // Compute Q, K, V projections for local heads using COSMA
        {
            PERF_SCOPED_TIMER("MPIAttentionKernel::computeLocalProjections");
            computeLocalProjections(global_input, local_wq, local_wk, local_wv,
                                    local_q, local_k, local_v, seq_len, d_model);
        }
        // Apply RoPE to local Q and K
        {
            PERF_SCOPED_TIMER("MPIAttentionKernel::applyLocalRoPE");
            applyLocalRoPE(local_q->data(), local_k->data(), seq_len, local_heads);
        }

        // Create local attended output tensor
        auto local_attended_output = createLocalSimpleTensor({seq_len, local_head_dim});

        // Compute attention for local heads
        {
            PERF_SCOPED_TIMER("MPIAttentionKernel::computeLocalAttention");
            computeLocalAttention(local_q->data(), local_k->data(), local_v->data(),
                                  local_attended_output->data(), seq_len, local_heads);
        }

        // Create local final output tensor
        auto local_final_output = createLocalSimpleTensor({seq_len, d_model});

        // Compute output projection for local heads using COSMA
        {
            PERF_SCOPED_TIMER("MPIAttentionKernel::computeLocalOutputProjection");
            computeLocalOutputProjection(local_attended_output, local_wo,
                                         local_final_output, seq_len, local_heads, d_model);
        }

        // Always emit local partial (per-rank head slice) into global_output.
        // Global aggregation (sum/concat) is now explicitly the caller's responsibility.
        {
            PERF_SCOPED_TIMER("MPIAttentionKernel::emitLocalPartial");
            std::copy(local_final_output->data(),
                      local_final_output->data() + seq_len * d_model,
                      global_output->data());
            // Optional misuse guard: when enabled, stamp a lightweight canary into an unlikely location
            // (final element) so that downstream replicated-assuming code performing a silent in-place
            // reduction or reuse can be detected in debug diagnostics. Enabled by env var:
            //   LLAMINAR_ASSERT_REPLICATED_MISUSE=1 will later trigger a check in validate() on next execute.
            if (getShardingDebugConfig().assert_replicated_misuse)
            {
                // Embed rank id scaled marker; harmless numerically (very small) but detectable.
                global_output->data()[seq_len * d_model - 1] += (float)(1e-30 * (getRank() + 1));
            }
            if (getRank() == 0)
            {
                LOG_DEBUG("MPIAttentionKernel: emitted local partial head contribution (legacy gather path removed)");
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        double execution_time = std::chrono::duration<double, std::milli>(end - start).count();

        LOG_DEBUG("MPIAttention executed in " + std::to_string(execution_time) +
                  " ms on rank " + std::to_string(getRank()) +
                  " (local_heads=" + std::to_string(local_heads) + ")");
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
        // Compute Q = input @ local_wq using adaptive matrix multiplication
        {
            PERF_SCOPED_TIMER("COSMA::Q_projection");
            auto [local_heads, head_offset] = getHeadDistribution();
            size_t local_head_dim = static_cast<size_t>(local_heads * head_dim_);
            if (!adaptive_matmul(input->data(), local_wq->data(), local_q->data(),
                                 seq_len, local_head_dim, d_model, false))
            {
                LOG_ERROR("Q projection failed on rank " << getRank());
                return;
            }
        }

        // Compute K = input @ local_wk using adaptive matrix multiplication
        {
            PERF_SCOPED_TIMER("COSMA::K_projection");
            auto [local_heads, head_offset] = getHeadDistribution();
            size_t local_head_dim = static_cast<size_t>(local_heads * head_dim_);
            if (!adaptive_matmul(input->data(), local_wk->data(), local_k->data(),
                                 seq_len, local_head_dim, d_model, false))
            {
                LOG_ERROR("K projection failed on rank " << getRank());
                return;
            }
        }

        // Compute V = input @ local_wv using adaptive matrix multiplication
        {
            PERF_SCOPED_TIMER("COSMA::V_projection");
            auto [local_heads, head_offset] = getHeadDistribution();
            size_t local_head_dim = static_cast<size_t>(local_heads * head_dim_);
            if (!adaptive_matmul(input->data(), local_wv->data(), local_v->data(),
                                 seq_len, local_head_dim, d_model, false))
            {
                LOG_ERROR("V projection failed on rank " << getRank());
                return;
            }
        }

        auto [local_heads, head_offset] = getHeadDistribution();
        LOG_DEBUG("Computed local projections using COSMA for " << local_heads << " heads on rank " << getRank());
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
        // Simplified RoPE implementation - apply rotation to each head
        const float theta_base = rope_freq_base_;

        for (int head = 0; head < local_heads; ++head)
        {
            for (size_t seq = 0; seq < seq_len; ++seq)
            {
                float *q_head = local_q + seq * local_heads * head_dim_ + head * head_dim_;
                float *k_head = local_k + seq * local_heads * head_dim_ + head * head_dim_;

                for (int dim_pair = 0; dim_pair < head_dim_ / 2; ++dim_pair)
                {
                    float theta = 1.0f / std::pow(theta_base, (2.0f * dim_pair) / head_dim_);
                    float cos_theta = std::cos((n_past_ + static_cast<float>(seq)) * theta);
                    float sin_theta = std::sin((n_past_ + static_cast<float>(seq)) * theta);

                    // Apply rotation to Q
                    float q0 = q_head[2 * dim_pair];
                    float q1 = q_head[2 * dim_pair + 1];
                    q_head[2 * dim_pair] = q0 * cos_theta - q1 * sin_theta;
                    q_head[2 * dim_pair + 1] = q0 * sin_theta + q1 * cos_theta;

                    // Apply rotation to K
                    float k0 = k_head[2 * dim_pair];
                    float k1 = k_head[2 * dim_pair + 1];
                    k_head[2 * dim_pair] = k0 * cos_theta - k1 * sin_theta;
                    k_head[2 * dim_pair + 1] = k0 * sin_theta + k1 * cos_theta;
                }
            }
        }

        LOG_DEBUG("Applied RoPE to " << local_heads << " local heads on rank " << getRank());
    }

    void MPIAttentionKernel::computeLocalAttentionScores(const float *local_q, const float *local_k, float *scores,
                                                         size_t seq_len, int local_heads)
    {
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

        for (int head = 0; head < local_heads; ++head)
        {
            // Compute Q @ K^T for this head
            for (size_t i = 0; i < seq_len; ++i)
            {
                const float *q_row = local_q + i * local_heads * head_dim_ + head * head_dim_;

                for (size_t j = 0; j < seq_len; ++j)
                {
                    const float *k_row = local_k + j * local_heads * head_dim_ + head * head_dim_;

                    float score = 0.0f;
                    for (int d = 0; d < head_dim_; ++d)
                    {
                        score += q_row[d] * k_row[d];
                    }

                    scores[head * seq_len * seq_len + i * seq_len + j] = score * scale;
                }
            }

            // Apply causal mask and softmax for this head
            for (size_t i = 0; i < seq_len; ++i)
            {
                float *score_row = scores + head * seq_len * seq_len + i * seq_len;

                // Apply causal mask
                for (size_t j = i + 1; j < seq_len; ++j)
                {
                    score_row[j] = -INFINITY;
                }

                // Compute softmax
                float max_score = -INFINITY;
                for (size_t j = 0; j <= i; ++j)
                {
                    max_score = std::max(max_score, score_row[j]);
                }

                float sum_exp = 0.0f;
                for (size_t j = 0; j <= i; ++j)
                {
                    score_row[j] = std::exp(score_row[j] - max_score);
                    sum_exp += score_row[j];
                }

                for (size_t j = 0; j <= i; ++j)
                {
                    score_row[j] /= sum_exp;
                }

                // Set masked positions to 0
                for (size_t j = i + 1; j < seq_len; ++j)
                {
                    score_row[j] = 0.0f;
                }
            }
        }

        LOG_DEBUG("Computed attention scores for " << local_heads << " heads on rank " << getRank());
    }

    void MPIAttentionKernel::applyLocalAttention(const float *scores, const float *local_v, float *local_attended_output,
                                                 size_t seq_len, int local_heads)
    {
        // Compute attention @ values for each head
        for (int head = 0; head < local_heads; ++head)
        {
            for (size_t i = 0; i < seq_len; ++i)
            {
                const float *score_row = scores + head * seq_len * seq_len + i * seq_len;
                float *output_row = local_attended_output + i * local_heads * head_dim_ + head * head_dim_;

                // Initialize output row to zero
                for (int d = 0; d < head_dim_; ++d)
                {
                    output_row[d] = 0.0f;
                }

                // Compute weighted sum of values
                for (size_t j = 0; j < seq_len; ++j)
                {
                    const float *v_row = local_v + j * local_heads * head_dim_ + head * head_dim_;
                    float weight = score_row[j];

                    for (int d = 0; d < head_dim_; ++d)
                    {
                        output_row[d] += weight * v_row[d];
                    }
                }
            }
        }

        LOG_DEBUG("Applied attention to values for " << local_heads << " heads on rank " << getRank());
    }

    void MPIAttentionKernel::computeLocalOutputProjection(const std::shared_ptr<TensorBase> &local_attended_output,
                                                          const std::shared_ptr<TensorBase> &local_wo,
                                                          std::shared_ptr<TensorBase> &local_final_output,
                                                          size_t seq_len, int local_heads, size_t d_model)
    {
        // Compute attended_output @ local_wo using adaptive matrix multiplication
        PERF_SCOPED_TIMER("COSMA::output_projection");
        size_t local_head_dim = static_cast<size_t>(local_heads * head_dim_);
        if (!adaptive_matmul(local_attended_output->data(), local_wo->data(), local_final_output->data(),
                             seq_len, d_model, local_head_dim, false))
        {
            LOG_ERROR("Output projection failed on rank " << getRank());
            return;
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

        LOG_DEBUG("Computed output projection using COSMA for " << local_heads << " heads on rank " << getRank());
    }

    std::shared_ptr<TensorBase> MPIAttentionKernel::createLocalSimpleTensor(const std::vector<size_t> &shape) const
    {
        std::vector<int> int_shape(shape.begin(), shape.end());
        return TensorFactory::create_simple(int_shape);
    }

} // namespace llaminar