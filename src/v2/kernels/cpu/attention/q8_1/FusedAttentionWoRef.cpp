/**
 * @file FusedAttentionWoRef.cpp
 * @brief Reference implementation of fused attention + Wo projection kernel.
 *
 * Composes microkernels (Q8DotProduct, OnlineSoftmax, VWeightedAccum, WoProjection)
 * into a complete attention + output projection kernel.
 *
 * Supports:
 * - Single sequence prefill
 * - Single sequence decode (with position_offset)
 * - Batched decode with uniform or variable KV lengths
 *
 * @author David Sanftenberg
 * @date December 2025
 */

#include "FusedAttentionWoRef.h"
#include "microkernels/Q8DotProduct.h"
#include "microkernels/OnlineSoftmax.h"
#include "microkernels/VWeightedAccum.h"
#include "microkernels/WoProjection.h"
#include "microkernels/FastExp.h"
#include "../../../../utils/Logger.h"

#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

namespace llaminar::v2::kernels
{

    bool FusedAttentionWoRef::validate_params(const FusedAttentionWoParams &params)
    {
        // Check required tensors
        if (params.Q == nullptr || params.K == nullptr || params.V == nullptr)
        {
            LOG_ERROR("FusedAttentionWoRef: Q, K, V tensors must not be null");
            return false;
        }

        if (params.output == nullptr)
        {
            LOG_ERROR("FusedAttentionWoRef: output buffer must not be null");
            return false;
        }

        // Check Wo
        if (params.Wo == nullptr)
        {
            LOG_ERROR("FusedAttentionWoRef: Wo weights must not be null");
            return false;
        }

        // Check batch size
        if (params.batch_size <= 0)
        {
            LOG_ERROR("FusedAttentionWoRef: batch_size must be positive");
            return false;
        }

        // Check dimensions
        if (params.seq_len <= 0 || params.kv_seq_len <= 0)
        {
            LOG_ERROR("FusedAttentionWoRef: seq_len and kv_seq_len must be positive");
            return false;
        }

        if (params.num_heads <= 0 || params.num_kv_heads <= 0)
        {
            LOG_ERROR("FusedAttentionWoRef: num_heads and num_kv_heads must be positive");
            return false;
        }

        if (params.num_heads % params.num_kv_heads != 0)
        {
            LOG_ERROR("FusedAttentionWoRef: num_heads must be divisible by num_kv_heads (GQA)");
            return false;
        }

        if (params.head_dim <= 0 || params.head_dim % 32 != 0)
        {
            LOG_ERROR("FusedAttentionWoRef: head_dim must be positive and divisible by 32");
            return false;
        }

        if (params.d_model <= 0)
        {
            LOG_ERROR("FusedAttentionWoRef: d_model must be positive");
            return false;
        }

        // Validate variable KV lengths if provided
        if (params.kv_seq_lens != nullptr)
        {
            for (int b = 0; b < params.batch_size; ++b)
            {
                if (params.kv_seq_lens[b] <= 0)
                {
                    LOG_ERROR("FusedAttentionWoRef: kv_seq_lens[" << b << "] must be positive");
                    return false;
                }
                if (params.kv_seq_lens[b] > params.kv_seq_len)
                {
                    LOG_ERROR("FusedAttentionWoRef: kv_seq_lens[" << b << "] exceeds max kv_seq_len");
                    return false;
                }
            }
        }

        return true;
    }

    FusedAttentionWoRef::BatchOffsets FusedAttentionWoRef::compute_batch_offsets(
        const FusedAttentionWoParams &params)
    {
        BatchOffsets offsets;
        const int num_blocks = params.head_dim / 32;

        // Q: [batch_size, seq_len, num_heads, num_blocks]
        offsets.q_batch_stride = static_cast<size_t>(params.seq_len) *
                                 params.num_heads * num_blocks;

        // K/V: [batch_size, kv_seq_len, num_kv_heads, num_blocks]
        offsets.kv_batch_stride = static_cast<size_t>(params.kv_seq_len) *
                                  params.num_kv_heads * num_blocks;

        // Output: [batch_size, seq_len, d_model]
        offsets.output_batch_stride = static_cast<size_t>(params.seq_len) * params.d_model;

        return offsets;
    }

    bool FusedAttentionWoRef::execute(const FusedAttentionWoParams &params)
    {
        // Validate parameters
        if (!validate_params(params))
        {
            return false;
        }

        const int batch_size = params.batch_size;
        const int head_dim = params.head_dim;
        const int d_model = params.d_model;
        const int seq_len = params.seq_len;

        // Compute batch strides
        const BatchOffsets offsets = compute_batch_offsets(params);

        // Zero entire output buffer (all batch items, all heads accumulate into it)
        std::memset(params.output, 0,
                    static_cast<size_t>(batch_size) * seq_len * d_model * sizeof(float));

        // Allocate context buffer for intermediate FP32 attention output
        std::vector<float> context_buffer(head_dim);

        // Process each batch item
        for (int b = 0; b < batch_size; ++b)
        {
            process_batch_item(params, b, context_buffer.data());
        }

        return true;
    }

    void FusedAttentionWoRef::process_batch_item(
        const FusedAttentionWoParams &params,
        int batch_idx,
        float *context_buffer)
    {
        const int seq_len = params.seq_len;
        const int num_heads = params.num_heads;
        const int head_dim = params.head_dim;
        const int d_model = params.d_model;

        // Compute output offset for this batch item
        const BatchOffsets offsets = compute_batch_offsets(params);
        float *output_base = params.output + batch_idx * offsets.output_batch_stride;

        // Process each query position
        for (int m = 0; m < seq_len; ++m)
        {
            float *output_row = output_base + m * d_model;

            // Process each attention head
            for (int h = 0; h < num_heads; ++h)
            {
                process_head(params, batch_idx, m, h, context_buffer);

                // Project context through Wo and accumulate into output
                // μK4: WoProjection
                microkernels::WoProjectionParams wo_params;
                wo_params.context = context_buffer;
                wo_params.wo_weights = params.Wo;
                wo_params.wo_type = params.wo_type;
                wo_params.head_dim = head_dim;
                wo_params.d_model = d_model;
                wo_params.head_idx = h;
                wo_params.n_heads = num_heads;
                wo_params.output = output_row;
                wo_params.accumulate = true; // Multiple heads contribute

                microkernels::wo_projection_ref(wo_params);
            }
        }
    }

    void FusedAttentionWoRef::execute_single_head(
        const FusedAttentionWoParams &params,
        int query_pos,
        int head_idx,
        float *context_buffer,
        float *output_buffer,
        int batch_idx)
    {
        // Process this specific head
        process_head(params, batch_idx, query_pos, head_idx, context_buffer);

        // Project through Wo
        microkernels::WoProjectionParams wo_params;
        wo_params.context = context_buffer;
        wo_params.wo_weights = params.Wo;
        wo_params.wo_type = params.wo_type;
        wo_params.head_dim = params.head_dim;
        wo_params.d_model = params.d_model;
        wo_params.head_idx = head_idx;
        wo_params.n_heads = params.num_heads;
        wo_params.output = output_buffer;
        wo_params.accumulate = true;

        microkernels::wo_projection_ref(wo_params);
    }

    void FusedAttentionWoRef::process_head(
        const FusedAttentionWoParams &params,
        int batch_idx, // Batch index
        int m,         // Query position within batch
        int h,         // Head index
        float *context // Output: [head_dim] FP32 context
    )
    {
        using namespace microkernels;

        const int head_dim = params.head_dim;
        const int num_blocks = head_dim / 32; // Q8_1 block size is 32
        const int num_kv_heads = params.num_kv_heads;
        const int kv_head = get_kv_head(h, params.num_heads, num_kv_heads);

        // Compute batch offsets
        const BatchOffsets offsets = compute_batch_offsets(params);

        // Get Q row for this batch, position, and head
        // Layout: Q[batch_size, seq_len, num_heads, num_blocks]
        const Q8_1Block *Q_batch = params.Q + batch_idx * offsets.q_batch_stride;
        const Q8_1Block *Q_row = Q_batch +
                                 (static_cast<size_t>(m) * params.num_heads + h) * num_blocks;

        // Get K/V base for this batch
        const Q8_1Block *K_batch = params.K + batch_idx * offsets.kv_batch_stride;
        const Q8_1Block *V_batch = params.V + batch_idx * offsets.kv_batch_stride;

        // Initialize online softmax state
        OnlineSoftmaxState softmax_state = online_softmax_init();

        // Zero context accumulator
        std::memset(context, 0, head_dim * sizeof(float));

        // Get KV length and position offset for this batch item
        const int kv_len = params.get_kv_len(batch_idx);
        const int pos_offset = params.get_position_offset(batch_idx);

        // Determine attention range
        // For causal attention: can only attend to positions [0, m + position_offset]
        // For full attention: attend to all positions [0, kv_len)
        int max_kv_pos;
        if (params.causal)
        {
            // In decode mode: m might be 0 while pos_offset indicates true position
            max_kv_pos = std::min(m + pos_offset + 1, kv_len);
        }
        else
        {
            max_kv_pos = kv_len;
        }

        // Iterate over all K/V positions
        for (int n = 0; n < max_kv_pos; ++n)
        {
            // Get K row for this position and KV head
            // Layout within batch: K[kv_seq_len, num_kv_heads, num_blocks]
            const Q8_1Block *K_row = K_batch +
                                     (static_cast<size_t>(n) * num_kv_heads + kv_head) * num_blocks;

            // μK1: Compute Q·K score
            Q8DotProductParams dot_params;
            dot_params.q_blocks = Q_row;
            dot_params.k_blocks = K_row;
            dot_params.num_blocks = num_blocks;
            dot_params.global_scale = params.scale;

            float score = q8_dot_product_ref(dot_params).score;

            // μK2: Online softmax update
            // Store old max to compute correction factor
            float old_max = softmax_state.max_score;
            float weight = online_softmax_update(softmax_state, score);

            // Compute correction factor for previously accumulated values
            float correction = 1.0f;
            if (softmax_state.initialized && n > 0 && softmax_state.max_score > old_max)
            {
                // Max changed, need to rescale existing accumulation
                correction = online_softmax_correction(old_max, softmax_state.max_score);
            }

            // μK3: Accumulate weighted V
            const Q8_1Block *V_row = V_batch +
                                     (static_cast<size_t>(n) * num_kv_heads + kv_head) * num_blocks;

            VWeightedAccumParams accum_params;
            accum_params.v_blocks = V_row;
            accum_params.weight = weight;
            accum_params.correction = correction;
            accum_params.context = context;
            accum_params.num_blocks = num_blocks;

            v_weighted_accum_ref(accum_params);
        }

        // Normalize context by 1/sum_exp
        float inv_sum = online_softmax_finalize(softmax_state);
        for (int d = 0; d < head_dim; ++d)
        {
            context[d] *= inv_sum;
        }
    }

} // namespace llaminar::v2::kernels
