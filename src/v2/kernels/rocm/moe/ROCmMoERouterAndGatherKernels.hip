/**
 * @file ROCmMoERouterAndGatherKernels.hip
 * @brief ROCm MoE router gate logits, softmax/top-k, gather/scatter, shared expert gate, SwiGLU, and canonical rank bank kernels.
 */
#include "ROCmMoERebalanceCommon.hip.inc"

namespace
{
    /**
     * @brief Compare router candidates without divergent short-circuit branches.
     * @param candidate_value Original probability bits, including unordered values.
     * @param candidate_id Original expert ID used to resolve equal probabilities.
     * @param best_value Current tournament probability.
     * @param best_id Current tournament expert ID.
     * @return Greater probability, or lower expert ID for an equal probability.
     *
     * All operands are already in registers and every comparison is pure. Use
     * boolean bit operations so LLVM need not mask lanes between the comparisons
     * in each tournament round. NaNs, signed zero and ID ties retain the same
     * comparison results; no floating-point arithmetic or reduction is changed.
     */
    __device__ __forceinline__ bool moe_topk_pair_better(
        float candidate_value,
        int candidate_id,
        float best_value,
        int best_id)
    {
        return (candidate_value > best_value) |
               ((candidate_value == best_value) & (candidate_id < best_id));
    }

    // Both router logits and softmax consume one production arithmetic authority.
    using llaminar2::rocm::moe_wave_contract::rocm_moe_wave_reduce_sum_f32;
    using llaminar2::rocm::moe_wave_contract::rocm_moe_wave_reduce_max_f32;

    __device__ void moe_select_topk_probabilities_block(
        float *__restrict__ expert_probs,
        int num_experts,
        int top_k,
        int *__restrict__ selected_ids,
        float *__restrict__ selected_weights)
    {
        const int lane = threadIdx.x % warpSize;
        const int warp_id = threadIdx.x / warpSize;

        /*
         * One wave64 scans the complete expert row.  Qwen3.6 has 256 experts,
         * so each lane owns four candidates per round.  Thousands of
         * independent row work-groups provide grid-level occupancy; involving
         * all four waves in each top-k slot only added two work-group barriers
         * per winner.  The single-wave tournament retains the exact stable
         * comparison order while reducing the whole selection to one final
         * publication barrier.
         */
        if (warp_id == 0)
        {
            for (int slot = 0; slot < top_k; ++slot)
            {
                float best_value = -INFINITY;
                int best_id = kDeviceMoEMaxExperts;
                for (int expert = lane;
                     expert < num_experts;
                     expert += warpSize)
                {
                    const float value = expert_probs[expert];
                    if (moe_topk_pair_better(
                            value,
                            expert,
                            best_value,
                            best_id))
                    {
                        best_value = value;
                        best_id = expert;
                    }
                }

                for (int offset = warpSize / 2; offset > 0; offset >>= 1)
                {
                    const float other_value = __shfl_down(best_value, offset);
                    const int other_id = __shfl_down(best_id, offset);
                    if (moe_topk_pair_better(other_value, other_id, best_value, best_id))
                    {
                        best_value = other_value;
                        best_id = other_id;
                    }
                }

                const int winner = __shfl(best_id, 0);
                if (lane == 0)
                {
                    selected_ids[slot] = best_id;
                    selected_weights[slot] = best_value;
                    if (winner >= 0 && winner < num_experts)
                        expert_probs[winner] = -INFINITY;
                }
                __builtin_amdgcn_wave_barrier();
            }
        }
        __syncthreads();
    }

    /**
     * @brief Select wave64 top-k from at most 512 experts without rescanning.
     *
     * Each lane owns four or eight expert ids congruent to its lane id.  It loads
     * and sorts those candidates once, then advances its register-resident
     * queue only when that lane wins the wave tournament.  The former router
     * reread all 256 shared probabilities for each of the eight Qwen top-k
     * slots and required a wave barrier after mutating the winning entry.
     *
     * The pair comparator is the same value-descending/id-ascending total
     * order used by @ref moe_select_topk_probabilities_block, so this changes
     * neither winner ids nor weights.  NaNs retain the former "not a
     * candidate" semantics by carrying the sentinel id.  Only thread zero
     * consumes the selected shared arrays after this helper, therefore no CTA
     * publication barrier is needed.
     * @tparam Candidates Original four/eight candidates per wave lane.
     * @tparam SelectQueueSwaps Use identical comparator decisions with bit-preserving
     *         selects in the explicit fused candidate; the original control retains
     *         its conditional swaps, including nonfinite and signed-zero behavior.
     */
    template <int Candidates = 4, bool SelectQueueSwaps = false>
    __device__ __forceinline__ void
    moe_select_topk_probabilities_wave64_register_queue(
        const float *__restrict__ expert_probs,
        int num_experts,
        int top_k,
        int *__restrict__ selected_ids,
        float *__restrict__ selected_weights)
    {
        static_assert(Candidates == 4 || Candidates == 8);
        constexpr int kWaveWidth = 64;
        const int lane = threadIdx.x & (kWaveWidth - 1);
        if (threadIdx.x >= kWaveWidth)
            return;

        int id0 = lane;
        int id1 = lane + kWaveWidth;
        int id2 = lane + 2 * kWaveWidth;
        int id3 = lane + 3 * kWaveWidth;
        int id4 = lane + 4 * kWaveWidth;
        int id5 = lane + 5 * kWaveWidth;
        int id6 = lane + 6 * kWaveWidth;
        int id7 = lane + 7 * kWaveWidth;
        float value0 = id0 < num_experts ? expert_probs[id0] : -INFINITY;
        float value1 = id1 < num_experts ? expert_probs[id1] : -INFINITY;
        float value2 = id2 < num_experts ? expert_probs[id2] : -INFINITY;
        float value3 = id3 < num_experts ? expert_probs[id3] : -INFINITY;
        float value4 = -INFINITY, value5 = -INFINITY;
        float value6 = -INFINITY, value7 = -INFINITY;
        if constexpr (Candidates == 8)
        {
            value4 = id4 < num_experts ? expert_probs[id4] : -INFINITY;
            value5 = id5 < num_experts ? expert_probs[id5] : -INFINITY;
            value6 = id6 < num_experts ? expert_probs[id6] : -INFINITY;
            value7 = id7 < num_experts ? expert_probs[id7] : -INFINITY;
        }

        if (id0 >= num_experts || value0 != value0)
            id0 = kDeviceMoEMaxExperts;
        if (id1 >= num_experts || value1 != value1)
            id1 = kDeviceMoEMaxExperts;
        if (id2 >= num_experts || value2 != value2)
            id2 = kDeviceMoEMaxExperts;
        if (id3 >= num_experts || value3 != value3)
            id3 = kDeviceMoEMaxExperts;
        if constexpr (Candidates == 8)
        {
            if (id4 >= num_experts || value4 != value4) id4 = kDeviceMoEMaxExperts;
            if (id5 >= num_experts || value5 != value5) id5 = kDeviceMoEMaxExperts;
            if (id6 >= num_experts || value6 != value6) id6 = kDeviceMoEMaxExperts;
            if (id7 >= num_experts || value7 != value7) id7 = kDeviceMoEMaxExperts;
        }

#define LLAMINAR_MOE_SORT_PAIR_DESC(A, B)                                    \
        do                                                                    \
        {                                                                     \
            const bool replace = moe_topk_pair_better(value##B, id##B, value##A, id##A); \
            if constexpr (SelectQueueSwaps)                                   \
            {                                                                 \
                const float original_value = value##A;                         \
                const int original_id = id##A;                                 \
                value##A = replace ? value##B : original_value;                 \
                id##A = replace ? id##B : original_id;                           \
                value##B = replace ? original_value : value##B;                 \
                id##B = replace ? original_id : id##B;                           \
            }                                                                 \
            else if (replace)                                                 \
            {                                                                 \
                const float swap_value = value##A;                            \
                const int swap_id = id##A;                                    \
                value##A = value##B;                                          \
                id##A = id##B;                                                \
                value##B = swap_value;                                        \
                id##B = swap_id;                                              \
            }                                                                 \
        } while (false)

        /* Five compare-swaps form a complete descending network for four. */
        LLAMINAR_MOE_SORT_PAIR_DESC(0, 1);
        LLAMINAR_MOE_SORT_PAIR_DESC(2, 3);
        LLAMINAR_MOE_SORT_PAIR_DESC(0, 2);
        LLAMINAR_MOE_SORT_PAIR_DESC(1, 3);
        LLAMINAR_MOE_SORT_PAIR_DESC(1, 2);
        if constexpr (Candidates == 8)
        {
            // Two sorted four-element queues merge with a fixed nine-comparator
            // odd/even network. All values/IDs stay in scalar registers; no
            // thread-local expert-sized storage or repeated LDS scan is needed.
            LLAMINAR_MOE_SORT_PAIR_DESC(4, 5);
            LLAMINAR_MOE_SORT_PAIR_DESC(6, 7);
            LLAMINAR_MOE_SORT_PAIR_DESC(4, 6);
            LLAMINAR_MOE_SORT_PAIR_DESC(5, 7);
            LLAMINAR_MOE_SORT_PAIR_DESC(5, 6);
            LLAMINAR_MOE_SORT_PAIR_DESC(0, 4);
            LLAMINAR_MOE_SORT_PAIR_DESC(2, 6);
            LLAMINAR_MOE_SORT_PAIR_DESC(2, 4);
            LLAMINAR_MOE_SORT_PAIR_DESC(1, 5);
            LLAMINAR_MOE_SORT_PAIR_DESC(3, 7);
            LLAMINAR_MOE_SORT_PAIR_DESC(3, 5);
            LLAMINAR_MOE_SORT_PAIR_DESC(1, 2);
            LLAMINAR_MOE_SORT_PAIR_DESC(3, 4);
            LLAMINAR_MOE_SORT_PAIR_DESC(5, 6);
        }
#undef LLAMINAR_MOE_SORT_PAIR_DESC

        for (int slot = 0; slot < top_k; ++slot)
        {
            float best_value = value0;
            int best_id = id0;
            for (int offset = kWaveWidth / 2; offset > 0; offset >>= 1)
            {
                const float other_value = __shfl_down(best_value, offset);
                const int other_id = __shfl_down(best_id, offset);
                if (moe_topk_pair_better(
                        other_value,
                        other_id,
                        best_value,
                        best_id))
                {
                    best_value = other_value;
                    best_id = other_id;
                }
            }

            const int winner = __shfl(best_id, 0);
            if (lane == 0)
            {
                selected_ids[slot] = best_id;
                selected_weights[slot] = best_value;
            }
            if (winner < num_experts &&
                lane == (winner & (kWaveWidth - 1)))
            {
                value0 = value1;
                id0 = id1;
                value1 = value2;
                id1 = id2;
                value2 = value3;
                id2 = id3;
                if constexpr (Candidates == 8)
                {
                    value3 = value4; id3 = id4;
                    value4 = value5; id4 = id5;
                    value5 = value6; id5 = id6;
                    value6 = value7; id6 = id7;
                    value7 = -INFINITY; id7 = kDeviceMoEMaxExperts;
                }
                else
                {
                    value3 = -INFINITY;
                    id3 = kDeviceMoEMaxExperts;
                }
            }
        }
    }
} // namespace

// =========================================================================
// Kernel 1: Single-token Gate Logits
//
// Decode routes one token at a time. hipBLAS is efficient for prefill GEMM but
// pays too much fixed overhead for a 1 x d_model by num_experts router GEMV.
// This path computes one expert logit per block and feeds the existing
// softmax/top-k kernel.
// =========================================================================

/**
 * @brief Bit-exact multi-expert LDS-cached FP32 router logits kernel for M=1 decode.
 *
 * Loads hidden[0..d_model-1] once per 8-expert CTA into dynamic LDS as float4,
 * then executes 4 teams of 128 threads x 2 interleaved experts using the exact
 * stride-128 float4 order and wave64/2-warp reduction order as the baseline kernel.
 */
__global__ __launch_bounds__(512) void rocm_moe_gate_logits_single_token_fp32_lds_kernel(
    const float *__restrict__ hidden,       // [d_model]
    const float *__restrict__ gate_weights, // [num_experts, d_model]
    float *__restrict__ logits,             // [num_experts]
    int d_model, int num_experts)
{
    extern __shared__ float4 sh_hidden4[];
    const int vec_count = d_model >> 2;
    const float4 *hidden4 = reinterpret_cast<const float4 *>(hidden);
    for (int i = threadIdx.x; i < vec_count; i += 512)
        sh_hidden4[i] = hidden4[i];
    __syncthreads();

    const int team = threadIdx.x >> 7;        // 0..3 (4 teams of 128 threads)
    const int tid128 = threadIdx.x & 127;     // 0..127 (exact baseline logical lane)
    const int e0 = static_cast<int>(blockIdx.x) * 8 + team;
    const int e1 = e0 + 4;
    const bool has_e0 = (e0 < num_experts);
    const bool has_e1 = (e1 < num_experts);

    const float4 *w0 = has_e0 ? reinterpret_cast<const float4 *>(gate_weights + static_cast<size_t>(e0) * d_model) : nullptr;
    const float4 *w1 = has_e1 ? reinterpret_cast<const float4 *>(gate_weights + static_cast<size_t>(e1) * d_model) : nullptr;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    if (has_e0 || has_e1)
    {
        for (int i = tid128; i < vec_count; i += 128)
        {
            const float4 h = sh_hidden4[i];
            if (has_e0)
            {
                const float4 ww0 = w0[i];
                sum0 += h.x * ww0.x + h.y * ww0.y + h.z * ww0.z + h.w * ww0.w;
            }
            if (has_e1)
            {
                const float4 ww1 = w1[i];
                sum1 += h.x * ww1.x + h.y * ww1.y + h.z * ww1.z + h.w * ww1.w;
            }
        }
    }

    sum0 = rocm_moe_wave_reduce_sum_f32(sum0);
    sum1 = rocm_moe_wave_reduce_sum_f32(sum1);

    __shared__ float team_warp_sums[4][2][2];
    const int lane = tid128 & 63;
    const int warp_in_team = tid128 >> 6; // 0 or 1
    if (lane == 0)
    {
        team_warp_sums[team][0][warp_in_team] = sum0;
        team_warp_sums[team][1][warp_in_team] = sum1;
    }
    __syncthreads();

    if (warp_in_team == 0)
    {
        if (has_e0)
        {
            float s0 = (lane < 2) ? team_warp_sums[team][0][lane] : 0.0f;
            s0 = rocm_moe_wave_reduce_sum_f32(s0);
            if (lane == 0) logits[e0] = s0;
        }
        if (has_e1)
        {
            float s1 = (lane < 2) ? team_warp_sums[team][1][lane] : 0.0f;
            s1 = rocm_moe_wave_reduce_sum_f32(s1);
            if (lane == 0) logits[e1] = s1;
        }
    }
}

__global__ void rocm_moe_gate_logits_single_token_kernel(
    const float *__restrict__ hidden,       // [d_model]
    const float *__restrict__ gate_weights, // [num_experts, d_model]
    float *__restrict__ logits,             // [num_experts]
    int d_model, int num_experts)
{
    const int expert = blockIdx.x;
    if (expert >= num_experts) return;

    const float *w = gate_weights + expert * d_model;
    float sum = 0.0f;

    if ((d_model & 3) == 0)
    {
        const int vec_count = d_model >> 2;
        const float4 *hidden4 = reinterpret_cast<const float4 *>(hidden);
        const float4 *w4 = reinterpret_cast<const float4 *>(w);
        for (int i = threadIdx.x; i < vec_count; i += blockDim.x)
        {
            const float4 h = hidden4[i];
            const float4 ww = w4[i];
            sum += h.x * ww.x + h.y * ww.y + h.z * ww.z + h.w * ww.w;
        }
    }
    else
    {
        for (int i = threadIdx.x; i < d_model; i += blockDim.x)
        {
            sum += hidden[i] * w[i];
        }
    }

    sum = rocm_moe_wave_reduce_sum_f32(sum);

    __shared__ float warp_sums[8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0)
    {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        sum = (lane < nw) ? warp_sums[lane] : 0.0f;
        sum = rocm_moe_wave_reduce_sum_f32(sum);
        if (lane == 0) logits[expert] = sum;
    }
}

__global__ void rocm_moe_gate_logits_single_token_kpart_kernel(
    const float *__restrict__ hidden,       // [d_model]
    const float *__restrict__ gate_weights, // [num_experts, d_model]
    float *__restrict__ partials,           // [num_experts, k_partitions]
    int d_model, int num_experts, int k_partitions,
    bool use_vec4)
{
    const int expert = blockIdx.x;
    const int partition = blockIdx.y;
    if (expert >= num_experts || partition >= k_partitions) return;

    const int block_count = (d_model + 31) / 32;
    const int blocks_per_partition = (block_count + k_partitions - 1) / k_partitions;
    const int block_begin = partition * blocks_per_partition;
    const int block_end_unclamped = block_begin + blocks_per_partition;
    const int block_end = (block_end_unclamped < block_count) ? block_end_unclamped : block_count;
    const int element_begin = block_begin * 32;
    const int element_end_unclamped = block_end * 32;
    const int element_end = (element_end_unclamped < d_model) ? element_end_unclamped : d_model;

    const float *w = gate_weights + expert * d_model;
    float sum = 0.0f;

    if (use_vec4)
    {
        const int vec_begin = element_begin >> 2;
        const int vec_end = element_end >> 2;
        const float4 *hidden4 = reinterpret_cast<const float4 *>(hidden);
        const float4 *w4 = reinterpret_cast<const float4 *>(w);
        for (int i = vec_begin + threadIdx.x; i < vec_end; i += blockDim.x)
        {
            const float4 h = hidden4[i];
            const float4 ww = w4[i];
            sum += h.x * ww.x + h.y * ww.y + h.z * ww.z + h.w * ww.w;
        }
    }
    else
    {
        for (int i = element_begin + threadIdx.x; i < element_end; i += blockDim.x)
        {
            sum += hidden[i] * w[i];
        }
    }

    sum = rocm_moe_wave_reduce_sum_f32(sum);

    __shared__ float warp_sums[8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0)
    {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        sum = (lane < nw) ? warp_sums[lane] : 0.0f;
        sum = rocm_moe_wave_reduce_sum_f32(sum);
        if (lane == 0) partials[expert * k_partitions + partition] = sum;
    }
}

__global__ void rocm_moe_gate_logits_kpart_reduce_kernel(
    const float *__restrict__ partials, // [num_experts, k_partitions]
    float *__restrict__ logits,         // [num_experts]
    int num_experts, int k_partitions)
{
    const int expert = blockIdx.x;
    if (expert >= num_experts) return;

    const float *expert_partials = partials + expert * k_partitions;
    float sum = 0.0f;
    for (int partition = threadIdx.x; partition < k_partitions; partition += blockDim.x)
    {
        sum += expert_partials[partition];
    }

    sum = rocm_moe_wave_reduce_sum_f32(sum);

    __shared__ float warp_sums[8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0)
    {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        sum = (lane < nw) ? warp_sums[lane] : 0.0f;
        sum = rocm_moe_wave_reduce_sum_f32(sum);
        if (lane == 0) logits[expert] = sum;
    }
}


/**
 * @brief Quantize one FP32 value with the canonical serial-decode Q8 arithmetic.
 *
 * The ordinary NativeQuantised M=1 decoder computes a reciprocal once per 32-value
 * block and multiplies each value by that reciprocal.  Writing the equivalent
 * expression as `value / scale` changes FP32 rounding for values near an integer
 * half-way point; `rintf` can then select a different int8 byte.  Router-owned Q8
 * hidden rows are consumed directly by grouped MTP expert kernels, so they must
 * preserve the serial decoder's operation order, not merely its real-number
 * formula. Router gate-weight caching uses this same contract so identical
 * FP32 model weights publish identical cached Q8 bytes on every GPU backend.
 *
 * @param value FP32 source value from the hidden activation row.
 * @param scale Positive block scale, or the canonical all-zero scale of 1.
 * @return Signed Q8 byte using round-to-nearest-even after saturation.
 */
__device__ __forceinline__ int8_t
rocm_moe_quantize_q8_decode_equivalent(float value, float scale)
{
    return static_cast<int8_t>(
        llaminar2::rocm::decode_equivalent::quantizeQ8FromScale(
            value, scale));
}

__global__ void rocm_moe_router_gate_quantize_q8_kernel(
    const float *__restrict__ gate_weights, // [num_experts, d_model]
    int8_t *__restrict__ gate_weights_q8,   // [num_experts, blocks_per_row, 32]
    float *__restrict__ gate_scales,        // [num_experts, blocks_per_row]
    int d_model, int num_experts)
{
    constexpr int BLOCK_SIZE = 32;
    const int expert = blockIdx.x;
    const int block_idx = blockIdx.y;
    const int lane = threadIdx.x;
    if (expert >= num_experts || lane >= BLOCK_SIZE) return;

    const int blocks_per_row = d_model / BLOCK_SIZE;
    const int k = block_idx * BLOCK_SIZE + lane;
    const size_t scale_idx = static_cast<size_t>(expert) * blocks_per_row + block_idx;
    const float value = (k < d_model) ? gate_weights[static_cast<size_t>(expert) * d_model + k] : 0.0f;

    float max_abs = fabsf(value);
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        max_abs = fmaxf(max_abs, __shfl_xor(max_abs, mask));
    }

    const float scale =
        llaminar2::rocm::decode_equivalent::q8Scale(max_abs);
    if (lane == 0)
    {
        gate_scales[scale_idx] = scale;
    }

    gate_weights_q8[scale_idx * BLOCK_SIZE + lane] =
        rocm_moe_quantize_q8_decode_equivalent(value, scale);
}

__global__ void rocm_moe_router_hidden_quantize_q8_kernel(
    const float *__restrict__ hidden,       // [d_model]
    int8_t *__restrict__ hidden_q8,         // [d_model]
    float *__restrict__ hidden_scales,      // [blocks_per_row]
    int d_model)
{
    constexpr int BLOCK_SIZE = 32;
    const int block_idx = blockIdx.x;
    const int lane = threadIdx.x;
    const int k = block_idx * BLOCK_SIZE + lane;
    if (lane >= BLOCK_SIZE || k >= d_model) return;

    const float value = hidden[k];
    float max_abs = fabsf(value);
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        max_abs = fmaxf(max_abs, __shfl_xor(max_abs, mask));
    }

    const float scale =
        llaminar2::rocm::decode_equivalent::q8Scale(max_abs);
    if (lane == 0)
    {
        hidden_scales[block_idx] = scale;
    }

    hidden_q8[k] = rocm_moe_quantize_q8_decode_equivalent(value, scale);
}

/**
 * @brief Fused hidden Q8 quantization (Original or SignedHadamard32) +
 *        2-lanes-per-K32-block coalesced int4 Q8 router logits kernel (wave64).
 *
 * Replaces up to three separate single-token router launches (Hadamard Q8
 * quantize, Q8 gate logits, and original-basis Q8 re-quantize) with one
 * 512-thread (8 experts/CTA) launch. Block 0 publishes the original-basis Q8
 * hidden bank in parallel while every CTA quantizes into LDS and computes
 * 1024-byte coalesced wave64 int4 dot products with the exact two-wave FP32
 * reduction order (blocks 0..63 + blocks 64..127) of the baseline kernel.
 */
template <bool SignedHadamard>
__global__ __launch_bounds__(512) void rocm_moe_gate_logits_single_token_q8_fused_int4_kernel(
    const float *__restrict__ hidden,           // [d_model]
    int8_t *__restrict__ out_orig_hidden_q8,    // [d_model]
    float *__restrict__ out_orig_hidden_scales, // [blocks_per_row]
    const int8_t *__restrict__ gate_weights_q8, // [num_experts, blocks_per_row, 32]
    const float *__restrict__ gate_scales,      // [num_experts, blocks_per_row]
    float *__restrict__ logits,                 // [num_experts]
    int d_model, int num_experts)
{
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__GFX9__)
    constexpr int kMaxBlocks = 128;
    const int blocks_per_row = d_model >> 5;
    __shared__ int4 sh_h4[kMaxBlocks * 2];
    __shared__ float sh_hscales[kMaxBlocks];

    const int sub_warp = threadIdx.x >> 5; // 0..15 (16 x 32-lane sub-warps)
    const int sub_lane = threadIdx.x & 31; // 0..31
    for (int blk = sub_warp; blk < blocks_per_row; blk += 16)
    {
        const int k = (blk << 5) + sub_lane;
        const float orig_value = hidden[k];
        if (blockIdx.x == 0 || !SignedHadamard)
        {
            float max_abs = fabsf(orig_value);
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1)
                max_abs = fmaxf(max_abs, __shfl_xor(max_abs, mask, 32));
            const float scale = llaminar2::rocm::decode_equivalent::q8Scale(max_abs);
            const int8_t q = rocm_moe_quantize_q8_decode_equivalent(orig_value, scale);
            if (blockIdx.x == 0)
            {
                if (sub_lane == 0)
                    out_orig_hidden_scales[blk] = scale;
                out_orig_hidden_q8[k] = q;
            }
            if constexpr (!SignedHadamard)
            {
                if (sub_lane == 0)
                    sh_hscales[blk] = scale;
                reinterpret_cast<int8_t *>(sh_h4)[k] = q;
            }
        }
        if constexpr (SignedHadamard)
        {
            namespace fp = llaminar2::device_fp32_contract;
            float value = fp::floatFromBits(
                fp::bitsFromFloat(orig_value) ^
                (((llaminar2::router_hadamard_q8::negative_signs >> sub_lane) & 1u) << 31u));
            #pragma unroll
            for (int stride = 1; stride < 32; stride *= 2)
            {
                const float other = llaminar2::router_hadamard_q8::exchange(value, stride);
                value = (sub_lane & stride) ? fp::add(other, -value) : fp::add(value, other);
            }
            value = fp::multiply(value, 0x1.6a09e6p-3f);
            float maximum = llaminar2::device_q8_activation_contract::maximumAbsoluteValue(value, 0.0f);
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1)
                maximum = llaminar2::device_q8_activation_contract::maximumAbsoluteValue(
                    maximum, llaminar2::router_hadamard_q8::exchange(maximum, mask));
            const float scale = llaminar2::device_q8_activation_contract::scale(maximum);
            if (sub_lane == 0)
                sh_hscales[blk] = scale;
            reinterpret_cast<int8_t *>(sh_h4)[k] =
                static_cast<int8_t>(llaminar2::device_q8_activation_contract::quantize(value, scale));
        }
    }
    __syncthreads();

    const int wave = threadIdx.x >> 6; // 0..7 (8 experts per CTA)
    const int lane = threadIdx.x & 63; // 0..63
    const int expert = static_cast<int>(blockIdx.x) * 8 + wave;
    if (expert >= num_experts)
        return;

    const int blk_in_32 = lane >> 1; // 0..31
    const int half = lane & 1;       // 0..1
    const size_t scale_base = static_cast<size_t>(expert) * blocks_per_row;
    const int4 *w4_row = reinterpret_cast<const int4 *>(gate_weights_q8 + scale_base * 32u);

    auto compute_32 = [&](int base_blk) -> float {
        const int blk = base_blk + blk_in_32;
        if (blk >= blocks_per_row)
            return 0.0f;
        const int idx = (blk << 1) + half;
        const int4 w = w4_row[idx];
        const int4 h = sh_h4[idx];
        int32_t d = ::llaminar2::rocm_integer_contract::signedDot4(h.x, w.x, 0);
        d = ::llaminar2::rocm_integer_contract::signedDot4(h.y, w.y, d);
        d = ::llaminar2::rocm_integer_contract::signedDot4(h.z, w.z, d);
        d = ::llaminar2::rocm_integer_contract::signedDot4(h.w, w.w, d);
        const int32_t full = d + __shfl_xor(d, 1, 64);
        return static_cast<float>(full) * sh_hscales[blk] * gate_scales[scale_base + blk];
    };

    const float t0 = compute_32(0);
    const float t1 = (blocks_per_row > 32) ? compute_32(32) : 0.0f;
    const float t2 = (blocks_per_row > 64) ? compute_32(64) : 0.0f;
    const float t3 = (blocks_per_row > 96) ? compute_32(96) : 0.0f;

    const int src_lane = (lane & 31) << 1;
    const float lo0 = __shfl(t0, src_lane, 64);
    const float hi0 = __shfl(t1, src_lane, 64);
    float sum0 = (lane < 32) ? lo0 : hi0;

    const float lo1 = __shfl(t2, src_lane, 64);
    const float hi1 = __shfl(t3, src_lane, 64);
    float sum1 = (lane < 32) ? lo1 : hi1;

    sum0 = rocm_moe_wave_reduce_sum_f32(sum0);
    sum1 = rocm_moe_wave_reduce_sum_f32(sum1);
    if (lane == 0)
        logits[expert] = sum0 + sum1;
#else
    __builtin_trap();
#endif
}

/**
 * @brief Coalesced LDS-cached Q8 router logits kernel for M=1 decode (wave64).
 *
 * Loads hidden_q8 and hidden_scales once per 8-expert CTA into LDS, then assigns
 * one wave64 per expert with 8 contiguous int32 lanes per K32 block (256-byte
 * coalesced reads, zero idle lanes). Preserves the exact baseline 128-thread
 * two-wave FP32 reduction order for blocks_per_row <= 128.
 */
__global__ __launch_bounds__(512) void rocm_moe_gate_logits_single_token_q8_coalesced_lds_kernel(
    const int8_t *__restrict__ hidden_q8,       // [d_model]
    const float *__restrict__ hidden_scales,    // [blocks_per_row]
    const int8_t *__restrict__ gate_weights_q8, // [num_experts, blocks_per_row, 32]
    const float *__restrict__ gate_scales,      // [num_experts, blocks_per_row]
    float *__restrict__ logits,                 // [num_experts]
    int d_model, int num_experts)
{
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__GFX9__)
    constexpr int kMaxBlocks = 128;
    const int blocks_per_row = d_model >> 5;
    const int words_per_row = d_model >> 2;
    __shared__ int32_t sh_h4[kMaxBlocks * 8];
    __shared__ float sh_hscales[kMaxBlocks];
    __shared__ float sh_terms[8][kMaxBlocks];

    const int32_t *h4_src = reinterpret_cast<const int32_t *>(hidden_q8);
    for (int i = threadIdx.x; i < words_per_row; i += 512)
        sh_h4[i] = h4_src[i];
    if (threadIdx.x < blocks_per_row)
        sh_hscales[threadIdx.x] = hidden_scales[threadIdx.x];
    __syncthreads();

    const int wave = threadIdx.x >> 6;     // 0..7 (8 experts per CTA)
    const int lane = threadIdx.x & 63;     // 0..63
    const int sub = lane & 7;              // 0..7 (int32 word within K32 block)
    const int blk_in_step = lane >> 3;     // 0..7 (8 K32 blocks per wave64 step)
    const int expert = static_cast<int>(blockIdx.x) * 8 + wave;
    const bool active = (expert < num_experts);

    if (active)
    {
        const size_t scale_base = static_cast<size_t>(expert) * blocks_per_row;
        const int32_t *w4_row = reinterpret_cast<const int32_t *>(
            gate_weights_q8 + scale_base * 32u);
        for (int blk_base = 0; blk_base < blocks_per_row; blk_base += 8)
        {
            const int blk = blk_base + blk_in_step;
            int32_t dot = 0;
            if (blk < blocks_per_row)
            {
                const int w_idx = (blk << 3) + sub;
                dot = ::llaminar2::rocm_integer_contract::signedDot4(sh_h4[w_idx], w4_row[w_idx], 0);
            }
            dot += __shfl_down(dot, 4, 8);
            dot += __shfl_down(dot, 2, 8);
            dot += __shfl_down(dot, 1, 8);
            if (sub == 0 && blk < blocks_per_row)
            {
                sh_terms[wave][blk] = static_cast<float>(dot) *
                    sh_hscales[blk] * gate_scales[scale_base + blk];
            }
        }
    }
    __syncthreads();

    if (active)
    {
        float sum0 = (lane < blocks_per_row) ? sh_terms[wave][lane] : 0.0f;
        float sum1 = (lane + 64 < blocks_per_row) ? sh_terms[wave][lane + 64] : 0.0f;
        sum0 = rocm_moe_wave_reduce_sum_f32(sum0);
        sum1 = rocm_moe_wave_reduce_sum_f32(sum1);
        if (lane == 0)
            logits[expert] = sum0 + sum1;
    }
#else
    __builtin_trap();
#endif
}

__global__ void rocm_moe_gate_logits_single_token_q8_weights_kernel(
    const int8_t *__restrict__ hidden_q8,       // [d_model]
    const float *__restrict__ hidden_scales,    // [blocks_per_row]
    const int8_t *__restrict__ gate_weights_q8, // [num_experts, blocks_per_row, 32]
    const float *__restrict__ gate_scales,      // [num_experts, blocks_per_row]
    float *__restrict__ logits,                 // [num_experts]
    int d_model, int num_experts)
{
    constexpr int BLOCK_SIZE = 32;
    const int expert = blockIdx.x;
    if (expert >= num_experts) return;

    const int blocks_per_row = d_model / BLOCK_SIZE;
    float sum = 0.0f;
    for (int block_idx = threadIdx.x; block_idx < blocks_per_row; block_idx += blockDim.x)
    {
        const int32_t *h4 = reinterpret_cast<const int32_t *>(hidden_q8 + block_idx * BLOCK_SIZE);
        const size_t scale_idx = static_cast<size_t>(expert) * blocks_per_row + block_idx;
        const int32_t *w4 = reinterpret_cast<const int32_t *>(gate_weights_q8 + scale_idx * BLOCK_SIZE);

        int32_t block_acc = 0;
        #pragma unroll
        for (int group = 0; group < 8; ++group)
        {
            block_acc = ::llaminar2::rocm_integer_contract::signedDot4(h4[group], w4[group], block_acc);
        }
        sum += static_cast<float>(block_acc) * hidden_scales[block_idx] * gate_scales[scale_idx];
    }

    sum = rocm_moe_wave_reduce_sum_f32(sum);

    __shared__ float warp_sums[8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0)
    {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        sum = (lane < nw) ? warp_sums[lane] : 0.0f;
        sum = rocm_moe_wave_reduce_sum_f32(sum);
        if (lane == 0) logits[expert] = sum;
    }
}

/**
 * @brief Quantize every active prefill row with serial Q8 decode scales.
 *
 * Each 32-lane subgroup owns one original quantization block. Large prefill
 * launches pack eight such subgroups into a workgroup, using both halves of an
 * AMD wave instead of scheduling a separate half-empty wave for every block.
 * Small verifier/decode launches keep one subgroup per workgroup. Neither form
 * changes a row's scale, rounding, or output address; the published bytes are
 * also consumed directly by routed and shared experts without requantization.
 */
template<bool PublishSum>
__device__ __forceinline__ void rocm_moe_router_hidden_quantize_q8_rows_body(
    const float* hidden, int8_t* hidden_q8, float* hidden_scales, int seq_len,
    int d_model,const int* device_effective_seq_len, std::int16_t* sums)
{
    constexpr int BLOCK_SIZE = 32;
    const int block_idx = blockIdx.x * (blockDim.x / BLOCK_SIZE) + threadIdx.x / BLOCK_SIZE;
    const int row = blockIdx.y;
    const int lane = threadIdx.x % BLOCK_SIZE;
    const int k = block_idx * BLOCK_SIZE + lane;
    const int active_rows = device_effective_seq_len
                                ? max(0, min(seq_len, *device_effective_seq_len))
                                : seq_len;
    // Widths are multiples of 32: every lane of a tail subgroup exits together.
    // No shuffle may mix adjacent subgroups, including the halves of wave64.
    if (row >= active_rows || k >= d_model) return;

    const int blocks_per_row = d_model / BLOCK_SIZE;
    const size_t row_offset = static_cast<size_t>(row) * static_cast<size_t>(d_model);
    const float value = hidden[row_offset + k];
    float max_abs = fabsf(value);
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        max_abs = fmaxf(max_abs, __shfl_xor(max_abs, mask, BLOCK_SIZE));
    }

    const float scale =
        llaminar2::rocm::decode_equivalent::q8Scale(max_abs);
    if (lane == 0)
    {
        hidden_scales[static_cast<size_t>(row) * blocks_per_row + block_idx] = scale;
    }

    const auto code=rocm_moe_quantize_q8_decode_equivalent(value, scale);
    hidden_q8[row_offset + k] = code;
    if constexpr(PublishSum) {
        int sum=code;
#pragma unroll
        for(int mask=16;mask>0;mask>>=1)sum+=__shfl_xor(sum,mask,BLOCK_SIZE);
        if(lane==0)sums[static_cast<std::size_t>(row)*blocks_per_row+block_idx]=static_cast<std::int16_t>(sum);
    }
}


/** @brief Original launch identity over the same exact quantizer. */
__global__ void rocm_moe_router_hidden_quantize_q8_rows_kernel(
    const float* hidden, int8_t* hidden_q8, float* hidden_scales, int seq_len,
    int d_model,const int* device_effective_seq_len)
{
    rocm_moe_router_hidden_quantize_q8_rows_body<false>(
        hidden,hidden_q8,hidden_scales,seq_len,d_model,device_effective_seq_len,nullptr);
}

/** @brief Closed frontier's extra immutable integer plane, in the original quantizer node. */
__global__ void rocm_moe_router_hidden_quantize_q8_rows_and_sums_kernel(
    const float* hidden, int8_t* codes,float* scales,const int* live,std::int16_t* sums)
{
    rocm_moe_router_hidden_quantize_q8_rows_body<true>(hidden,codes,scales,4,2560,live,sums);
}

/**
 * @brief Economical grouped Q8 router logits for arbitrary prefill rows.
 *
 * One block owns one expert and a fixed tile of up to sixteen rows. Gate Q8 payload
 * and scales are loaded once per K block and reused inside that tile, while each
 * row keeps the same K traversal and reduction order as serial decode. Larger
 * prefill buckets add independent row tiles in `grid.y`; they never change a
 * row's numerical reduction shape.
 * @tparam Threads Physical lanes: 64 when serial lanes 64..127 own no blocks,
 * otherwise all 128 serial owners. The K stride is always 128, independently
 * of this scheduling choice. Wave32 targets retain the inter-wave reduction.
 * @param hidden_q8 Quantized hidden rows with 32-byte block boundaries.
 * @param hidden_scales Independently quantized FP32 scale for each row block.
 * @param gate_weights_q8 Prepared expert-major router bytes.
 * @param gate_scales Prepared FP32 scale for each router weight block.
 * @param logits Live expert-major output columns within each token row.
 * @param seq_len Captured row capacity; not the current live extent.
 * @param d_model Hidden width, a positive multiple of 32.
 * @param num_experts Number of router outputs per row.
 * @param device_effective_seq_len Optional device-owned live count, clamped
 * to the captured capacity before any row-local access.
 */
template<int Threads, bool Partitioned = false>
__global__ __launch_bounds__(Threads, 2)
void rocm_moe_gate_logits_q8_grouped_verifier_kernel(
    const int8_t *__restrict__ hidden_q8,
    const float *__restrict__ hidden_scales,
    const int8_t *__restrict__ gate_weights_q8,
    const float *__restrict__ gate_scales,
    float *__restrict__ logits,
    int seq_len,
    int d_model,
    int num_experts,
    const int *__restrict__ device_effective_seq_len,
    llaminar2::DeviceRowPartition partition = {})
{
    static_assert(Threads == 64 || Threads == 128);
    if constexpr (Partitioned)
    {
        // Only ownership changes. Keep full local Q8 rows available to later
        // experts, and publish this owner's logits contiguously from row zero.
        const auto rows = partition.resolve(seq_len, device_effective_seq_len);
        hidden_q8 += static_cast<size_t>(rows.first) * d_model;
        hidden_scales += static_cast<size_t>(rows.first) * (d_model / 32);
        seq_len = rows.count;
        device_effective_seq_len = nullptr;
    }
    constexpr int BLOCK_SIZE = 32;
    constexpr int ROWS_PER_BLOCK = 16;
    constexpr int SERIAL_K_PARTITIONS = 128;
    const int expert = blockIdx.x;
    if (expert >= num_experts) return;

    const int active_rows = device_effective_seq_len
                                ? max(0, min(seq_len, *device_effective_seq_len))
                                : seq_len;
    const int row_base = static_cast<int>(blockIdx.y) * ROWS_PER_BLOCK;
    const int rows_in_block = min(ROWS_PER_BLOCK, active_rows - row_base);
    if (rows_in_block <= 0) return;

    const int blocks_per_row = d_model / BLOCK_SIZE;
    float sums[ROWS_PER_BLOCK] = {};
    for (int block_idx = threadIdx.x; block_idx < blocks_per_row; block_idx += SERIAL_K_PARTITIONS)
    {
        const size_t scale_idx = static_cast<size_t>(expert) * blocks_per_row + block_idx;
        const int32_t *w4 = reinterpret_cast<const int32_t *>(gate_weights_q8 + scale_idx * BLOCK_SIZE);
        const float gate_scale = gate_scales[scale_idx];

        int32_t w_values[8];
        #pragma unroll
        for (int group = 0; group < 8; ++group)
        {
            w_values[group] = w4[group];
        }

        for (int local_row = 0; local_row < rows_in_block; ++local_row)
        {
            const int row = row_base + local_row;
            const int8_t *row_hidden =
                hidden_q8 + static_cast<size_t>(row) * static_cast<size_t>(d_model);
            const float *row_scales =
                hidden_scales + static_cast<size_t>(row) * static_cast<size_t>(blocks_per_row);
            const int32_t *h4 = reinterpret_cast<const int32_t *>(row_hidden + block_idx * BLOCK_SIZE);

            int32_t block_acc = 0;
            #pragma unroll
            for (int group = 0; group < 8; ++group)
            {
                block_acc = ::llaminar2::rocm_integer_contract::signedDot4(h4[group], w_values[group], block_acc);
            }
            sums[local_row] +=
                static_cast<float>(block_acc) * row_scales[block_idx] * gate_scale;
        }
    }

    // The omitted serial wave contributes +0, not another K block. On wave64
    // hardware its contribution needs neither an idle wave nor an LDS barrier.
    // Keeping the explicit addition also preserves signed-zero behavior. A
    // wave32 target still has two nonempty waves and uses the reduction below.
    if (Threads == warpSize)
    {
        for (int local_row = 0; local_row < rows_in_block; ++local_row)
        {
            float reduced = rocm_moe_wave_reduce_sum_f32(sums[local_row]);
            reduced = __fadd_rn(reduced, 0.0f);
            if (threadIdx.x == 0)
                logits[static_cast<size_t>(row_base + local_row) * num_experts + expert] = reduced;
        }
        return;
    }

    __shared__ float warp_sums[ROWS_PER_BLOCK][8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    for (int local_row = 0; local_row < rows_in_block; ++local_row)
    {
        float reduced = rocm_moe_wave_reduce_sum_f32(sums[local_row]);
        if (lane == 0) warp_sums[local_row][warp_id] = reduced;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        for (int local_row = 0; local_row < rows_in_block; ++local_row)
        {
            float reduced = (lane < nw) ? warp_sums[local_row][lane] : 0.0f;
            reduced = rocm_moe_wave_reduce_sum_f32(reduced);
            if (lane == 0)
            {
                const int row = row_base + local_row;
                logits[static_cast<size_t>(row) * num_experts + expert] = reduced;
            }
        }
    }
}

/**
 * @brief Round original router hidden to a physical half register, then widen.
 * @param value Original FP32 hidden; expert consumers retain these source bits.
 * @return Half-RNE operand widened exactly for FP32 products and accumulation.
 */
__device__ __forceinline__ float router_fp16_operand(float value)
{
    return __half2float(__float2half_rn(value));
}

/**
 * @brief Bit-exact 2-expert-per-CTA no-LDS FP16 router logits kernel for M=1 decode.
 *
 * Uses 256 CTAs x 128 threads (for 512 experts) with L1-cached FP32->FP16 hidden
 * loads and 2-expert register preloading, hiding HBM2 latency across 4+ waves/CU
 * with zero dynamic LDS while preserving the exact baseline stride-128 pair order
 * and two-wave FP32 reduction order.
 */
__global__ __launch_bounds__(128) void rocm_moe_gate_logits_single_token_fp16_weights_lds_kernel(
    const float *__restrict__ hidden,        // [d_model]
    const __half *__restrict__ gate_weights, // [num_experts, d_model]
    float *__restrict__ logits,              // [num_experts]
    int d_model, int num_experts)
{
    const int tid = threadIdx.x;
    const int lane = tid & 63;
    const int warp = tid >> 6; // 0 or 1
    const int e0 = static_cast<int>(blockIdx.x) * 2;
    const int e1 = e0 + 1;
    const bool has_e0 = (e0 < num_experts);
    const bool has_e1 = (e1 < num_experts);
    if (!has_e0)
        return;

    const int pair_count = d_model >> 1;
    const __half2 *w0 = reinterpret_cast<const __half2 *>(
        gate_weights + static_cast<size_t>(e0) * d_model);
    const __half2 *w1 = has_e1 ? reinterpret_cast<const __half2 *>(
        gate_weights + static_cast<size_t>(e1) * d_model) : nullptr;

    float s0 = 0.0f;
    float s1 = 0.0f;
    if (pair_count == 1280 && has_e1)
    {
        __half2 r0[10], r1[10];
        float2 rh[10];
        #pragma unroll
        for (int k = 0; k < 10; ++k)
        {
            const int i = tid + k * 128;
            r0[k] = w0[i];
            r1[k] = w1[i];
            rh[k] = {router_fp16_operand(hidden[i << 1]),
                     router_fp16_operand(hidden[(i << 1) + 1])};
        }
        #pragma unroll
        for (int k = 0; k < 10; ++k)
        {
            const float2 ww0 = __half22float2(r0[k]);
            const float2 ww1 = __half22float2(r1[k]);
            s0 += rh[k].x * ww0.x + rh[k].y * ww0.y;
            s1 += rh[k].x * ww1.x + rh[k].y * ww1.y;
        }
    }
    else
    {
        int i = tid;
        for (; i + 384 < pair_count; i += 512)
        {
            const __half2 a0 = w0[i], a1 = w0[i + 128], a2 = w0[i + 256], a3 = w0[i + 384];
            __half2 b0{}, b1{}, b2{}, b3{};
            if (has_e1)
            {
                b0 = w1[i]; b1 = w1[i + 128]; b2 = w1[i + 256]; b3 = w1[i + 384];
            }
            const int idxs[4] = {i, i + 128, i + 256, i + 384};
            const __half2 wa[4] = {a0, a1, a2, a3};
            const __half2 wb[4] = {b0, b1, b2, b3};
            #pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                const int h = idxs[k] << 1;
                const float hx = router_fp16_operand(hidden[h]);
                const float hy = router_fp16_operand(hidden[h + 1]);
                const float2 ww0 = __half22float2(wa[k]);
                s0 += hx * ww0.x + hy * ww0.y;
                if (has_e1)
                {
                    const float2 ww1 = __half22float2(wb[k]);
                    s1 += hx * ww1.x + hy * ww1.y;
                }
            }
        }
        for (; i < pair_count; i += 128)
        {
            const int h = i << 1;
            const float hx = router_fp16_operand(hidden[h]);
            const float hy = router_fp16_operand(hidden[h + 1]);
            const float2 ww0 = __half22float2(w0[i]);
            s0 += hx * ww0.x + hy * ww0.y;
            if (has_e1)
            {
                const float2 ww1 = __half22float2(w1[i]);
                s1 += hx * ww1.x + hy * ww1.y;
            }
        }
    }

    s0 = rocm_moe_wave_reduce_sum_f32(s0);
    if (has_e1)
        s1 = rocm_moe_wave_reduce_sum_f32(s1);

    __shared__ float tw[2][2];
    if (lane == 0)
    {
        tw[0][warp] = s0;
        if (has_e1)
            tw[1][warp] = s1;
    }
    __syncthreads();

    if (warp == 0)
    {
        float v0 = (lane < 2) ? tw[0][lane] : 0.0f;
        v0 = rocm_moe_wave_reduce_sum_f32(v0);
        if (lane == 0)
            logits[e0] = v0;
        if (has_e1)
        {
            float v1 = (lane < 2) ? tw[1][lane] : 0.0f;
            v1 = rocm_moe_wave_reduce_sum_f32(v1);
            if (lane == 0)
                logits[e1] = v1;
        }
    }
}

__global__ void rocm_moe_gate_logits_single_token_fp16_weights_kernel(
    const float *__restrict__ hidden,          // [d_model]
    const __half *__restrict__ gate_weights,   // [num_experts, d_model]
    float *__restrict__ logits,                // [num_experts]
    int d_model, int num_experts)
{
    const int expert = blockIdx.x;
    if (expert >= num_experts) return;

    const __half *w = gate_weights + expert * d_model;
    float sum = 0.0f;

    if ((d_model & 1) == 0)
    {
        const int pair_count = d_model >> 1;
        const __half2 *w2 = reinterpret_cast<const __half2 *>(w);
        for (int i = threadIdx.x; i < pair_count; i += blockDim.x)
        {
            const float2 ww = __half22float2(w2[i]);
            const int h = i << 1;
            sum += router_fp16_operand(hidden[h]) * ww.x + router_fp16_operand(hidden[h + 1]) * ww.y;
        }
    }
    else
    {
        for (int i = threadIdx.x; i < d_model; i += blockDim.x)
        {
            sum += router_fp16_operand(hidden[i]) * __half2float(w[i]);
        }
    }

    sum = rocm_moe_wave_reduce_sum_f32(sum);

    __shared__ float warp_sums[8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0)
    {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        sum = (lane < nw) ? warp_sums[lane] : 0.0f;
        sum = rocm_moe_wave_reduce_sum_f32(sum);
        if (lane == 0) logits[expert] = sum;
    }
}

__global__ void rocm_moe_gate_logits_single_token_bf16_weights_kernel(
    const float *__restrict__ hidden,              // [d_model]
    const hip_bfloat16 *__restrict__ gate_weights, // [num_experts, d_model]
    float *__restrict__ logits,                    // [num_experts]
    int d_model, int num_experts)
{
    const int expert = blockIdx.x;
    if (expert >= num_experts) return;

    const hip_bfloat16 *w = gate_weights + expert * d_model;
    float sum = 0.0f;

    for (int i = threadIdx.x; i < d_model; i += blockDim.x)
    {
        sum += hidden[i] * static_cast<float>(w[i]);
    }

    sum = rocm_moe_wave_reduce_sum_f32(sum);

    __shared__ float warp_sums[8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0)
    {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        sum = (lane < nw) ? warp_sums[lane] : 0.0f;
        sum = rocm_moe_wave_reduce_sum_f32(sum);
        if (lane == 0) logits[expert] = sum;
    }
}

/**
 * @brief Decode-equivalent grouped FP32 router logits for prefill rows.
 *
 * One HIP block owns one expert and a fixed sixteen-row tile. Gate values are
 * loaded once per K position and reused inside the tile, but each row keeps the
 * exact vectorized loop and wave/block reduction shape used by
 * rocm_moe_gate_logits_single_token_kernel(). Additional prefill rows add
 * independent `grid.y` tiles rather than changing the reduction tree.
 */
template<bool Partitioned = false>
__global__ __launch_bounds__(128, 2)
void rocm_moe_gate_logits_fp32_grouped_verifier_kernel(
    const float *__restrict__ hidden,
    const float *__restrict__ gate_weights,
    float *__restrict__ logits,
    int seq_len,
    int d_model,
    int num_experts,
    const int *__restrict__ device_effective_seq_len,
    llaminar2::DeviceRowPartition partition = {})
{
    if constexpr (Partitioned)
    {
        const auto rows = partition.resolve(seq_len, device_effective_seq_len);
        hidden += static_cast<size_t>(rows.first) * d_model;
        seq_len = rows.count;
        device_effective_seq_len = nullptr;
    }
    constexpr int ROWS_PER_BLOCK = 16;
    const int expert = blockIdx.x;
    if (expert >= num_experts) return;

    const int active_rows = device_effective_seq_len
                                ? max(0, min(seq_len, *device_effective_seq_len))
                                : seq_len;
    const int row_base = static_cast<int>(blockIdx.y) * ROWS_PER_BLOCK;
    const int rows_in_block = min(ROWS_PER_BLOCK, active_rows - row_base);
    if (rows_in_block <= 0) return;

    const float *w = gate_weights + static_cast<size_t>(expert) * d_model;
    float sums[ROWS_PER_BLOCK] = {};

    if ((d_model & 3) == 0)
    {
        const int vec_count = d_model >> 2;
        const float4 *w4 = reinterpret_cast<const float4 *>(w);
        for (int i = threadIdx.x; i < vec_count; i += blockDim.x)
        {
            const float4 ww = w4[i];
            for (int local_row = 0; local_row < rows_in_block; ++local_row)
            {
                const int row = row_base + local_row;
                const float4 hh =
                    reinterpret_cast<const float4 *>(hidden + static_cast<size_t>(row) * d_model)[i];
                sums[local_row] +=
                    hh.x * ww.x + hh.y * ww.y + hh.z * ww.z + hh.w * ww.w;
            }
        }
    }
    else
    {
        for (int i = threadIdx.x; i < d_model; i += blockDim.x)
        {
            const float ww = w[i];
            for (int local_row = 0; local_row < rows_in_block; ++local_row)
            {
                const int row = row_base + local_row;
                sums[local_row] += hidden[static_cast<size_t>(row) * d_model + i] * ww;
            }
        }
    }

    __shared__ float warp_sums[ROWS_PER_BLOCK][8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    for (int local_row = 0; local_row < rows_in_block; ++local_row)
    {
        float reduced = rocm_moe_wave_reduce_sum_f32(sums[local_row]);
        if (lane == 0) warp_sums[local_row][warp_id] = reduced;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        for (int local_row = 0; local_row < rows_in_block; ++local_row)
        {
            float reduced = (lane < nw) ? warp_sums[local_row][lane] : 0.0f;
            reduced = rocm_moe_wave_reduce_sum_f32(reduced);
            if (lane == 0)
            {
                const int row = row_base + local_row;
                logits[static_cast<size_t>(row) * num_experts + expert] = reduced;
            }
        }
    }
}

/**
 * @brief Decode-equivalent half-A/half-B router logits for live prefill/verifier rows.
 *
 * Mirrors rocm_moe_gate_logits_single_token_fp16_weights_kernel() per row,
 * including the half2 loop for even K, while sharing each gate load across the
 * sixteen-row tile handled by the expert-owned block.
 */
template<bool Partitioned, int FixedRows>
__device__ __forceinline__ void groupedFP16RouterRows(
    const float *__restrict__ hidden,
    const __half *__restrict__ gate_weights,
    float *__restrict__ logits,
    int seq_len,
    int d_model,
    int num_experts,
    const int *__restrict__ device_effective_seq_len,
    llaminar2::DeviceRowPartition partition = {})
{
    if constexpr (Partitioned)
    {
        const auto rows = partition.resolve(seq_len, device_effective_seq_len);
        hidden += static_cast<size_t>(rows.first) * d_model;
        seq_len = rows.count;
        device_effective_seq_len = nullptr;
    }
    static_assert(!Partitioned || FixedRows == 0);
    constexpr int ROWS_PER_BLOCK = FixedRows ? FixedRows : 16;
    const int width = FixedRows ? 2560 : d_model;
    const int experts = FixedRows ? 512 : num_experts;
    const int expert = blockIdx.x;
    if (expert >= experts) return;

    const int active_rows = FixedRows
        ? __builtin_amdgcn_readfirstlane(*device_effective_seq_len)
        : (device_effective_seq_len ? *device_effective_seq_len : seq_len);
    if (active_rows < 0 || active_rows > seq_len) __builtin_trap();
    const int row_base = FixedRows ? 0 : static_cast<int>(blockIdx.y) * ROWS_PER_BLOCK;
    const int rows_in_block = min(ROWS_PER_BLOCK, active_rows - row_base);
    if (rows_in_block <= 0) return;

    // Compare the original device-owned prefix before each row's input or output touch.
    // A fixed tile can retain one independently ordered accumulator per row.
    const auto visitRows = [rows_in_block](auto visit) {
        if constexpr (FixedRows)
        {
#pragma unroll
            for (int row = 0; row < ROWS_PER_BLOCK; ++row)
                if (row < rows_in_block) visit(row);
        }
        else
            for (int row = 0; row < rows_in_block; ++row)
                visit(row);
    };
    const __half *w = gate_weights + static_cast<size_t>(expert) * width;
    float sums[ROWS_PER_BLOCK] = {};

    if ((width & 1) == 0)
    {
        const int pair_count = width >> 1;
        const __half2 *w2 = reinterpret_cast<const __half2 *>(w);
        for (int i = threadIdx.x; i < pair_count; i += blockDim.x)
        {
            const float2 ww = __half22float2(w2[i]);
            const int h = i << 1;
            visitRows([&](int local_row)
            {
                const int row = row_base + local_row;
                const float *row_hidden = hidden + static_cast<size_t>(row) * width;
                sums[local_row] += router_fp16_operand(row_hidden[h]) * ww.x + router_fp16_operand(row_hidden[h + 1]) * ww.y;
            });
        }
    }
    else
    {
        for (int i = threadIdx.x; i < width; i += blockDim.x)
        {
            const float ww = __half2float(w[i]);
            visitRows([&](int local_row)
            {
                const int row = row_base + local_row;
                sums[local_row] += router_fp16_operand(hidden[static_cast<size_t>(row) * width + i]) * ww;
            });
        }
    }

    __shared__ float warp_sums[ROWS_PER_BLOCK][8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    visitRows([&](int local_row)
    {
        float reduced = rocm_moe_wave_reduce_sum_f32(sums[local_row]);
        if (lane == 0) warp_sums[local_row][warp_id] = reduced;
    });
    __syncthreads();

    if (warp_id == 0)
    {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        visitRows([&](int local_row)
        {
            float reduced = (lane < nw) ? warp_sums[local_row][lane] : 0.0f;
            reduced = rocm_moe_wave_reduce_sum_f32(reduced);
            if (lane == 0)
            {
                const int row = row_base + local_row;
                logits[static_cast<size_t>(row) * experts + expert] = reduced;
            }
        });
    }
}

/** @brief Preserve the original runtime-tiled grouped and partitioned entry. */
template<bool Partitioned = false>
__global__ __launch_bounds__(128, 2)
void rocm_moe_gate_logits_fp16_grouped_verifier_kernel(
    const float *__restrict__ hidden, const __half *__restrict__ gate_weights,
    float *__restrict__ logits, int seq_len, int d_model, int num_experts,
    const int *__restrict__ count, llaminar2::DeviceRowPartition partition = {})
{
    groupedFP16RouterRows<Partitioned, 0>(
        hidden, gate_weights, logits, seq_len, d_model, num_experts, count, partition);
}

/** @brief Compile the original FP32 accumulation and wave order for one counted tile. */
template<int Rows>
__global__ __launch_bounds__(128, 2)
void qfn_grouped_fp16_router_fixed_rows(
    const float *__restrict__ hidden, const __half *__restrict__ gate_weights,
    float *__restrict__ logits, const int *__restrict__ count, int capacity)
{
    groupedFP16RouterRows<false, Rows>(hidden, gate_weights, logits, capacity, 2560, 512, count);
}

/**
 * @brief Share half operands across four rows and eight independent experts.
 *
 * Even K retains the serial half2 pair expression; odd K retains scalar order.
 * Every output keeps the original 128-lane partition and the backend's exact
 * wave/block tree. The original hidden allocation is never modified.
 * @tparam Partitioned Resolve participant live intervals before operand loads.
 * @param hidden Original FP32 input, rounded to half RNE in registers.
 * @param gate_weights Model-owned physical half gate rows.
 * @param logits Participant-local FP32 output; inactive suffix stays untouched.
 * @param seq_len Global admitted capacity. @param d_model Reduction width.
 * @param num_experts Expert count. @param live_rows Optional device live count.
 * @param partition Immutable row ownership, resolved entirely on device.
 */
template<bool Partitioned>
__global__ __launch_bounds__(128, 2)
void rocm_moe_gate_logits_fp16_expert_tile_kernel(
    const float *__restrict__ hidden, const __half *__restrict__ gate_weights,
    float *__restrict__ logits, int seq_len, int d_model, int num_experts,
    const int *live_rows, llaminar2::DeviceRowPartition partition)
{
    if constexpr (Partitioned)
    {
        const auto rows = partition.resolve(seq_len, live_rows);
        hidden += static_cast<size_t>(rows.first) * d_model;
        seq_len = rows.count;
    }
    else if (live_rows)
    {
        const int count = *live_rows;
        if (count < 0 || count > seq_len) __builtin_trap();
        seq_len = count;
    }
    constexpr int Rows = 4, Experts = 8;
    const int first_row = blockIdx.y * Rows, first_expert = blockIdx.x * Experts;
    if (first_row >= seq_len || first_expert >= num_experts) return;
    float sums[Rows][Experts] = {};
    if ((d_model & 1) == 0)
    {
        for (int pair = threadIdx.x; pair < d_model / 2; pair += 128)
        {
            float2 weights[Experts];
            #pragma unroll
            for (int e = 0; e < Experts; ++e)
            {
                weights[e] = {0, 0};
                if (first_expert + e < num_experts)
                    weights[e] = __half22float2(reinterpret_cast<const __half2 *>(
                        gate_weights + static_cast<size_t>(first_expert + e) * d_model)[pair]);
            }
            #pragma unroll
            for (int r = 0; r < Rows; ++r)
                if (first_row + r < seq_len)
                {
                    const float *h = hidden + static_cast<size_t>(first_row + r) * d_model + pair * 2;
                    const float a = router_fp16_operand(h[0]), b = router_fp16_operand(h[1]);
                    #pragma unroll
                    for (int e = 0; e < Experts; ++e) sums[r][e] += a * weights[e].x + b * weights[e].y;
                }
        }
    }
    else
    {
        for (int j = threadIdx.x; j < d_model; j += 128)
        {
            float weights[Experts];
            #pragma unroll
            for (int e = 0; e < Experts; ++e)
                weights[e] = first_expert + e < num_experts
                    ? __half2float(gate_weights[static_cast<size_t>(first_expert + e) * d_model + j]) : 0.0f;
            #pragma unroll
            for (int r = 0; r < Rows; ++r)
                if (first_row + r < seq_len)
                {
                    const float h = router_fp16_operand(hidden[static_cast<size_t>(first_row + r) * d_model + j]);
                    #pragma unroll
                    for (int e = 0; e < Experts; ++e) sums[r][e] += h * weights[e];
                }
        }
    }
    // Four slots also represent wave32 targets; gfx906 consumes exactly two.
    __shared__ float partial[Rows][Experts][4];
    const int lane = threadIdx.x % warpSize, wave = threadIdx.x / warpSize;
    #pragma unroll
    for (int r = 0; r < Rows; ++r)
    {
        #pragma unroll
        for (int e = 0; e < Experts; ++e)
        {
            const float value = rocm_moe_wave_reduce_sum_f32(sums[r][e]);
            if (lane == 0) partial[r][e][wave] = value;
        }
    }
    __syncthreads();
    if (wave == 0)
    {
        #pragma unroll
        for (int r = 0; r < Rows; ++r)
        {
            #pragma unroll
            for (int e = 0; e < Experts; ++e)
            {
                float value = lane < 128 / warpSize ? partial[r][e][lane] : 0.0f;
                value = rocm_moe_wave_reduce_sum_f32(value);
                if (lane == 0 && first_row + r < seq_len && first_expert + e < num_experts)
                    logits[static_cast<size_t>(first_row + r) * num_experts + first_expert + e] = value;
            }
        }
    }
}

/**
 * @brief Select measured small-row or shared-expert geometry before capture.
 * @tparam Partitioned Resolve participant rows inside the retained kernel.
 * @param hidden Original FP32 rows. @param gate Model-owned physical half gate.
 * @param logits Participant-local FP32 output. @param capacity Whole-input capacity.
 * @param local_capacity Admitted owner rows. @param width Reduction width.
 * @param experts Expert count. @param live_rows Optional device-owned live count.
 * @param partition Immutable row ownership.
 * @param stream Exact non-null stream authenticated by the public launch bridge.
 *
 * M1/M8 prefer the existing single-expert tile. M16 through M2048 benefit from
 * eight-expert reuse; selection never downloads a device-owned live count.
 */
template<bool Partitioned>
void launch_fp16_router_rows(const float *hidden, const __half *gate, float *logits,
    int capacity, int local_capacity, int width, int experts, const int *live_rows,
    llaminar2::DeviceRowPartition partition, hipStream_t stream)
{
    if (local_capacity >= 16 && experts >= 8)
        hipLaunchKernelGGL((rocm_moe_gate_logits_fp16_expert_tile_kernel<Partitioned>),
            dim3((experts + 7) / 8, (local_capacity + 3) / 4), dim3(128), 0, stream,
            hidden, gate, logits, capacity, width, experts, live_rows, partition);
    else
        hipLaunchKernelGGL((rocm_moe_gate_logits_fp16_grouped_verifier_kernel<Partitioned>),
            dim3(experts, (local_capacity + 15) / 16), dim3(128), 0, stream,
            hidden, gate, logits, capacity, width, experts, live_rows, partition);
}

/**
 * @brief Decode-equivalent grouped BF16-router logits for prefill rows.
 */
template<bool Partitioned = false>
__global__ __launch_bounds__(128, 2)
void rocm_moe_gate_logits_bf16_grouped_verifier_kernel(
    const float *__restrict__ hidden,
    const hip_bfloat16 *__restrict__ gate_weights,
    float *__restrict__ logits,
    int seq_len,
    int d_model,
    int num_experts,
    const int *__restrict__ device_effective_seq_len,
    llaminar2::DeviceRowPartition partition = {})
{
    if constexpr (Partitioned)
    {
        const auto rows = partition.resolve(seq_len, device_effective_seq_len);
        hidden += static_cast<size_t>(rows.first) * d_model;
        seq_len = rows.count;
        device_effective_seq_len = nullptr;
    }
    constexpr int ROWS_PER_BLOCK = 16;
    const int expert = blockIdx.x;
    if (expert >= num_experts) return;

    const int active_rows = device_effective_seq_len
                                ? max(0, min(seq_len, *device_effective_seq_len))
                                : seq_len;
    const int row_base = static_cast<int>(blockIdx.y) * ROWS_PER_BLOCK;
    const int rows_in_block = min(ROWS_PER_BLOCK, active_rows - row_base);
    if (rows_in_block <= 0) return;

    const hip_bfloat16 *w = gate_weights + static_cast<size_t>(expert) * d_model;
    float sums[ROWS_PER_BLOCK] = {};
    for (int i = threadIdx.x; i < d_model; i += blockDim.x)
    {
        const float ww = static_cast<float>(w[i]);
        for (int local_row = 0; local_row < rows_in_block; ++local_row)
        {
            const int row = row_base + local_row;
            sums[local_row] += hidden[static_cast<size_t>(row) * d_model + i] * ww;
        }
    }

    __shared__ float warp_sums[ROWS_PER_BLOCK][8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    for (int local_row = 0; local_row < rows_in_block; ++local_row)
    {
        float reduced = rocm_moe_wave_reduce_sum_f32(sums[local_row]);
        if (lane == 0) warp_sums[local_row][warp_id] = reduced;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        for (int local_row = 0; local_row < rows_in_block; ++local_row)
        {
            float reduced = (lane < nw) ? warp_sums[local_row][lane] : 0.0f;
            reduced = rocm_moe_wave_reduce_sum_f32(reduced);
            if (lane == 0)
            {
                const int row = row_base + local_row;
                logits[static_cast<size_t>(row) * num_experts + expert] = reduced;
            }
        }
    }
}

// =========================================================================
// Kernel 1b: Single-token Gate Logits, Grouped Experts
//
// Each CTA handles a contiguous group of experts and loads the decode hidden
// vector into LDS once. On gfx906 this maps four wave64 waves to four experts,
// avoiding cross-wave reductions while preserving the original logits layout.
// =========================================================================

__global__ void rocm_moe_gate_logits_grouped_decode_router_rows_kernel(
    const float *__restrict__ hidden,       // [seq_len, d_model]
    const float *__restrict__ gate_weights, // [num_experts, d_model]
    float *__restrict__ logits,             // [seq_len, num_experts]
    int d_model, int num_experts,
    bool use_vec4,
    int seq_len,
    const int *__restrict__ device_effective_seq_len)
{
    const int row = static_cast<int>(blockIdx.y);
    const int active_rows = device_effective_seq_len
                                ? max(0, min(seq_len, *device_effective_seq_len))
                                : seq_len;
    if (row >= active_rows)
        return;

    const float *row_hidden =
        hidden + static_cast<size_t>(row) * static_cast<size_t>(d_model);
    extern __shared__ float4 hidden_shared_vec4[];
    float *hidden_shared = reinterpret_cast<float *>(hidden_shared_vec4);

    if (use_vec4)
    {
        const int vec_count = d_model >> 2;
        const float4 *hidden4 = reinterpret_cast<const float4 *>(row_hidden);
        for (int i = threadIdx.x; i < vec_count; i += blockDim.x)
        {
            hidden_shared_vec4[i] = hidden4[i];
        }
    }
    else
    {
        for (int i = threadIdx.x; i < d_model; i += blockDim.x)
        {
            hidden_shared[i] = row_hidden[i];
        }
    }
    __syncthreads();

    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (warp_id >= kGateLogitsExpertsPerBlock)
        return;

    const int expert = blockIdx.x * kGateLogitsExpertsPerBlock + warp_id;
    if (expert >= num_experts)
        return;

    const float *w = gate_weights + expert * d_model;
    float sum = 0.0f;

    if (use_vec4)
    {
        const int vec_count = d_model >> 2;
        const float4 *hidden4 = reinterpret_cast<const float4 *>(hidden_shared);
        const float4 *w4 = reinterpret_cast<const float4 *>(w);
        for (int i = lane; i < vec_count; i += warpSize)
        {
            const float4 h = hidden4[i];
            const float4 ww = w4[i];
            sum += h.x * ww.x + h.y * ww.y + h.z * ww.z + h.w * ww.w;
        }
    }
    else
    {
        for (int i = lane; i < d_model; i += warpSize)
        {
            sum += hidden_shared[i] * w[i];
        }
    }

    sum = rocm_moe_wave_reduce_sum_f32(sum);

    if (lane == 0)
    {
        logits[static_cast<size_t>(row) * static_cast<size_t>(num_experts) +
               static_cast<size_t>(expert)] = sum;
    }
}

// =========================================================================
// Kernel 2b: Decode Softmax + Top-K + Runtime Write
//
// Single-token decode variant that keeps routing entirely device-side: it
// converts logits to probabilities, selects top-k, writes the stable runtime
// table, optionally mirrors legacy FP32 routing tensors, and updates the
// decode histogram in one launch.
// =========================================================================

// Runtime router bridges use the same 256-lane expert reduction geometry.
__global__ __launch_bounds__(kDeviceMoEMaxExperts) void rocm_moe_softmax_topk_decode_runtime_kernel(
    float *__restrict__ logits,                       // [num_experts] IN/OUT (probabilities)
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    DeviceMoELayerRuntimeView *__restrict__ runtime_layers,
    float *__restrict__ legacy_indices,               // optional [top_k]
    float *__restrict__ legacy_weights,               // optional [top_k]
    int num_experts, int top_k,
    bool normalize_weights,
    bool write_legacy_outputs,
    bool update_runtime_histogram,
    llaminar2::RoutedExpertRowExecutionPolicy row_execution_policy,
    const int32_t *__restrict__ absolute_position_ids,
    const DeviceMoERebalancePlanEntryView *__restrict__ rebalance_plan_entries,
    uint32_t rebalance_plan_capacity,
    DeviceMoERebalanceCommandBufferHeaderView *__restrict__ rebalance_command_header,
    const DeviceMoEExpertDirectoryEntryView *__restrict__ rebalance_local_transfer_slots,
    uint32_t rebalance_local_transfer_slot_count,
    DeviceMoERebalanceConfigView rebalance_config,
    DeviceMoERebalanceApplyStatusView *__restrict__ rebalance_apply_status,
    DeviceMoERebalanceGraphControllerStateView *__restrict__ rebalance_controller_state,
    int rebalance_target_layer,
    uint32_t rebalance_command_buffer_count,
        const int32_t *__restrict__ active_row_count_device)
{
    if (!logits || !runtime || num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK)
    {
        return;
    }

    try_apply_ready_rebalance_wave_for_layer_thread0(
        runtime_layers,
        rebalance_plan_entries,
        rebalance_plan_capacity,
        rebalance_command_header,
        rebalance_local_transfer_slots,
        rebalance_local_transfer_slot_count,
        rebalance_config,
        rebalance_apply_status,
        rebalance_controller_state,
        rebalance_target_layer,
        rebalance_command_buffer_count);

    float local_max = -FLT_MAX;
    for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
    {
        local_max = fmaxf(local_max, logits[expert]);
    }
    local_max = rocm_moe_wave_reduce_max_f32(local_max);

    __shared__ float shared_max[8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) shared_max[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0)
    {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        local_max = (lane < num_warps) ? shared_max[lane] : -FLT_MAX;
        local_max = rocm_moe_wave_reduce_max_f32(local_max);
        if (lane == 0) shared_max[0] = local_max;
    }
    __syncthreads();
    const float max_val = shared_max[0];

    float local_sum = 0.0f;
    for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
    {
        const float value = expf(logits[expert] - max_val);
        logits[expert] = value;
        local_sum += value;
    }
    local_sum = rocm_moe_wave_reduce_sum_f32(local_sum);

    __shared__ float shared_sum[8];
    if (lane == 0) shared_sum[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0)
    {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        local_sum = (lane < num_warps) ? shared_sum[lane] : 0.0f;
        local_sum = rocm_moe_wave_reduce_sum_f32(local_sum);
        if (lane == 0) shared_sum[0] = local_sum;
    }
    __syncthreads();

    const float inv_sum = (shared_sum[0] > 0.0f) ? 1.0f / shared_sum[0] : 0.0f;
    for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
    {
        logits[expert] *= inv_sum;
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        int selected_ids[kDeviceMoEMaxTopK];
        float selected_weights[kDeviceMoEMaxTopK];

        for (int slot = 0; slot < top_k; ++slot)
        {
            selected_ids[slot] = -1;
            selected_weights[slot] = 0.0f;
        }

        for (int slot = 0; slot < top_k; ++slot)
        {
            float best_value = -1.0f;
            int best_expert = 0;
            for (int expert = 0; expert < num_experts; ++expert)
            {
                if (logits[expert] > best_value)
                {
                    bool already_selected = false;
                    for (int previous = 0; previous < slot; ++previous)
                    {
                        if (selected_ids[previous] == expert)
                        {
                            already_selected = true;
                            break;
                        }
                    }
                    if (!already_selected)
                    {
                        best_value = logits[expert];
                        best_expert = expert;
                    }
                }
            }
            selected_ids[slot] = best_expert;
            selected_weights[slot] = best_value;
        }

        if (normalize_weights)
        {
            float topk_sum = 0.0f;
            for (int slot = 0; slot < top_k; ++slot)
            {
                topk_sum += selected_weights[slot];
            }
            if (topk_sum > 0.0f)
            {
                const float inv_topk_sum = 1.0f / topk_sum;
                for (int slot = 0; slot < top_k; ++slot)
                {
                    selected_weights[slot] *= inv_topk_sum;
                }
            }
        }

        runtime_publish_decode_dispatch(
            runtime, selected_ids, selected_weights, num_experts, top_k,
            legacy_indices, legacy_weights, write_legacy_outputs,
            update_runtime_histogram,
            row_execution_policy,
            absolute_position_ids ? absolute_position_ids[0] : -1, active_row_count_device);
    }
}

/**
 * @brief Publish one wave64 decode route with compile-time maintenance policy.
 *
 * Static graphs instantiate `<false, false>` so LLVM can remove every
 * histogram atomic, ready-wave pointer, and rebalance control branch from the
 * captured router.  Observe/Dynamic graphs retain exactly the evidence and
 * apply work they own.  This mirrors CUDA's production policy split and keeps
 * inactive maintenance state from consuming SGPRs or scratch on every routed
 * layer.
 */
template <bool UpdateRuntimeHistogram, bool ApplyReadyRebalance>
__global__ __launch_bounds__(kDeviceMoEMaxExperts) void rocm_moe_softmax_topk_decode_runtime_wave64_kernel(
    float *__restrict__ logits,                       // [num_experts] IN logits, OUT probabilities
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    DeviceMoELayerRuntimeView *__restrict__ runtime_layers,
    float *__restrict__ legacy_indices,               // optional [top_k]
    float *__restrict__ legacy_weights,               // optional [top_k]
    int num_experts, int top_k,
    bool normalize_weights,
    bool write_legacy_outputs,
    llaminar2::RoutedExpertRowExecutionPolicy row_execution_policy,
    const int32_t *__restrict__ absolute_position_ids,
    const DeviceMoERebalancePlanEntryView *__restrict__ rebalance_plan_entries,
    uint32_t rebalance_plan_capacity,
    DeviceMoERebalanceCommandBufferHeaderView *__restrict__ rebalance_command_header,
    const DeviceMoEExpertDirectoryEntryView *__restrict__ rebalance_local_transfer_slots,
    uint32_t rebalance_local_transfer_slot_count,
    DeviceMoERebalanceConfigView rebalance_config,
    DeviceMoERebalanceApplyStatusView *__restrict__ rebalance_apply_status,
    DeviceMoERebalanceGraphControllerStateView *__restrict__ rebalance_controller_state,
    int rebalance_target_layer,
    uint32_t rebalance_command_buffer_count,
        const int32_t *__restrict__ active_row_count_device)
{
    if (!logits || !runtime || num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK)
    {
        return;
    }

    if constexpr (ApplyReadyRebalance)
    {
        try_apply_ready_rebalance_wave_for_layer_thread0(
            runtime_layers,
            rebalance_plan_entries,
            rebalance_plan_capacity,
            rebalance_command_header,
            rebalance_local_transfer_slots,
            rebalance_local_transfer_slot_count,
            rebalance_config,
            rebalance_apply_status,
            rebalance_controller_state,
            rebalance_target_layer,
            rebalance_command_buffer_count);
    }

    __shared__ float expert_probs[kDeviceMoEMaxExperts];
    __shared__ float shared_max[8];
    __shared__ float shared_sum[8];

    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;

    float max_val = -FLT_MAX;
    for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
    {
        max_val = fmaxf(max_val, logits[expert]);
    }
    max_val = rocm_moe_wave_reduce_max_f32(max_val);
    if (lane == 0) shared_max[warp_id] = max_val;
    __syncthreads();

    if (warp_id == 0)
    {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        max_val = (lane < num_warps) ? shared_max[lane] : -FLT_MAX;
        max_val = rocm_moe_wave_reduce_max_f32(max_val);
        if (lane == 0) shared_max[0] = max_val;
    }
    __syncthreads();
    max_val = shared_max[0];

    float sum = 0.0f;
    for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
    {
        const float value = expf(logits[expert] - max_val);
        expert_probs[expert] = value;
        sum += value;
    }
    sum = rocm_moe_wave_reduce_sum_f32(sum);
    if (lane == 0) shared_sum[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0)
    {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        sum = (lane < num_warps) ? shared_sum[lane] : 0.0f;
        sum = rocm_moe_wave_reduce_sum_f32(sum);
        if (lane == 0) shared_sum[0] = sum;
    }
    __syncthreads();

    const float inv_sum = (shared_sum[0] > 0.0f) ? 1.0f / shared_sum[0] : 0.0f;
    for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
    {
        const float probability = expert_probs[expert] * inv_sum;
        expert_probs[expert] = probability;
        logits[expert] = probability;
    }
    __syncthreads();

    __shared__ int selected_ids[kDeviceMoEMaxTopK];
    __shared__ float selected_weights[kDeviceMoEMaxTopK];
    moe_select_topk_probabilities_block(expert_probs, num_experts, top_k, selected_ids, selected_weights);

    if (threadIdx.x == 0)
    {
        if (normalize_weights)
        {
            float topk_sum = 0.0f;
            for (int slot = 0; slot < top_k; ++slot)
            {
                topk_sum += selected_weights[slot];
            }
            if (topk_sum > 0.0f)
            {
                const float inv_topk_sum = 1.0f / topk_sum;
                for (int slot = 0; slot < top_k; ++slot)
                {
                    selected_weights[slot] *= inv_topk_sum;
                }
            }
        }

        runtime_publish_decode_dispatch(
            runtime, selected_ids, selected_weights, num_experts, top_k,
            legacy_indices, legacy_weights, write_legacy_outputs,
            UpdateRuntimeHistogram,
            row_execution_policy,
            absolute_position_ids ? absolute_position_ids[0] : -1, active_row_count_device);
    }
}

/**
 * @brief Minimal wave64 router for independently published owner or fragment slots.
 *
 * Unlike the replica-aware kernel, this entrypoint does not carry rebalance
 * pointers, position state, or participant load-balancing scratch in its HIP
 * kernel ABI.  That distinction is material on gfx906: unused formal kernel
 * arguments still consume SGPRs, and the inlined replica scheduler otherwise
 * spills into private memory even in a Static graph.
 *
 * The probability reduction and top-k selection are intentionally identical
 * to @ref rocm_moe_softmax_topk_decode_runtime_wave64_kernel so selecting the
 * economical resident-set specialization cannot change routing arithmetic.
 *
 * @tparam RowPolicy Original whole-owner or intermediate-fragment row contract.
 * @tparam UpdateRuntimeHistogram Whether the captured graph owns histogram
 *         publication for Dynamic maintenance.

 * @tparam PublishDescriptors Publish original arena views in the authenticated route lane.
 * @tparam SelectQueueSwaps Preserve exact queue comparisons using selected tuple writes.
 * @param logits Original FP32 probabilities in place.
 * @param runtime Original device-owned runtime and acquired bank.
 * @param legacy_indices Complete selected IDs when requested.
 * @param legacy_weights Complete selected weight bits when requested.
 * @param num_experts Exact admitted source expert extent.
 * @param top_k Exact selected packet width.
 * @param normalize_weights Retain the original serial FP32 normalization.
 * @param write_legacy_outputs Whether complete selected outputs are required.
 * @param active_row_count_device Optional live count for this physical row.
 * @param publication Original arena outputs; unused in the control specialization.
 */
constexpr int kQFNRegisterSelectorExperts = 512;
constexpr int kQFNRegisterSelectorTopK = 10;
__device__ __forceinline__ void moe_grouped_topk_register_merge(
    const float* __restrict__ probabilities, int* __restrict__ selected_ids,
    float* __restrict__ selected_weights);

template <bool UpdateRuntimeHistogram,
          llaminar2::RoutedExpertRowExecutionPolicy RowPolicy = llaminar2::RoutedExpertRowExecutionPolicy::ParticipantAssigned,
          bool PublishDescriptors = false, bool SelectQueueSwaps = false>
__device__ __forceinline__ void rocm_moe_softmax_topk_decode_independent_body(
    float *__restrict__ logits,
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    float *__restrict__ legacy_indices,
    float *__restrict__ legacy_weights,
    int num_experts,
    int top_k,
    bool normalize_weights,
    bool write_legacy_outputs,
        const int32_t *__restrict__ active_row_count_device,
        llaminar2::MoEDecodeDescriptorPublication publication)
{
    if (!logits || !runtime || num_experts <= 0 ||
        num_experts > kDeviceMoEMaxExperts || top_k <= 0 ||
        top_k > kDeviceMoEMaxTopK)
    {
        return;
    }

    __shared__ float expert_probs[kDeviceMoEMaxExperts];
    __shared__ float shared_max[8];
    __shared__ float shared_sum[8];

    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    float max_val = -FLT_MAX;
    for (int expert = threadIdx.x; expert < num_experts;
         expert += blockDim.x)
    {
        max_val = fmaxf(max_val, logits[expert]);
    }
    max_val = rocm_moe_wave_reduce_max_f32(max_val);
    if (lane == 0)
        shared_max[warp_id] = max_val;
    __syncthreads();

    if (warp_id == 0)
    {
        const int num_warps =
            (blockDim.x + warpSize - 1) / warpSize;
        max_val = lane < num_warps ? shared_max[lane] : -FLT_MAX;
        max_val = rocm_moe_wave_reduce_max_f32(max_val);
        if (lane == 0)
            shared_max[0] = max_val;
    }
    __syncthreads();
    max_val = shared_max[0];

    float sum = 0.0f;
    for (int expert = threadIdx.x; expert < num_experts;
         expert += blockDim.x)
    {
        const float value = expf(logits[expert] - max_val);
        expert_probs[expert] = value;
        sum += value;
    }
    sum = rocm_moe_wave_reduce_sum_f32(sum);
    if (lane == 0)
        shared_sum[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0)
    {
        const int num_warps =
            (blockDim.x + warpSize - 1) / warpSize;
        sum = lane < num_warps ? shared_sum[lane] : 0.0f;
        sum = rocm_moe_wave_reduce_sum_f32(sum);
        if (lane == 0)
            shared_sum[0] = sum;
    }
    __syncthreads();

    const float inv_sum =
        shared_sum[0] > 0.0f ? 1.0f / shared_sum[0] : 0.0f;
    for (int expert = threadIdx.x; expert < num_experts;
         expert += blockDim.x)
    {
        const float probability = expert_probs[expert] * inv_sum;
        expert_probs[expert] = probability;
        logits[expert] = probability;
    }
    __syncthreads();

    __shared__ int selected_ids[kDeviceMoEMaxTopK];
    __shared__ float selected_weights[kDeviceMoEMaxTopK];
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__GFX9__)
    if (num_experts == kQFNRegisterSelectorExperts &&
        top_k == kQFNRegisterSelectorTopK &&
        blockDim.x == 256)
    {
        moe_grouped_topk_register_merge(
            expert_probs, selected_ids, selected_weights);
    }
    else
#endif
    if (num_experts > 256)
        moe_select_topk_probabilities_wave64_register_queue<8, SelectQueueSwaps>(
            expert_probs, num_experts, top_k, selected_ids, selected_weights);
    else
        moe_select_topk_probabilities_wave64_register_queue<4, SelectQueueSwaps>(
            expert_probs, num_experts, top_k, selected_ids, selected_weights);

    if (threadIdx.x == 0)
    {
        if (normalize_weights)
        {
            float topk_sum = 0.0f;
            for (int slot = 0; slot < top_k; ++slot)
                topk_sum += selected_weights[slot];
            if (topk_sum > 0.0f)
            {
                const float inv_topk_sum = 1.0f / topk_sum;
                for (int slot = 0; slot < top_k; ++slot)
                    selected_weights[slot] *= inv_topk_sum;
            }
        }
    }
    // Keep the serial FP32 normalization above unchanged. Every selected
    // slot's lane must observe its final bits before publishing owner metadata
    // and exact integer demand. Inactive lanes publish nothing.
    __syncthreads();
    runtime_publish_independent_decode_dispatch<UpdateRuntimeHistogram, RowPolicy, PublishDescriptors>(
        runtime, selected_ids, selected_weights, num_experts, top_k,
        legacy_indices, legacy_weights, write_legacy_outputs, active_row_count_device, publication);
}

/** @brief Original independent router entry; retains the control's exact kernel ABI. */
template <bool UpdateRuntimeHistogram,
          llaminar2::RoutedExpertRowExecutionPolicy RowPolicy = llaminar2::RoutedExpertRowExecutionPolicy::ParticipantAssigned>
__global__ void rocm_moe_softmax_topk_decode_independent_wave64_kernel(
    float *__restrict__ logits,
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    float *__restrict__ legacy_indices,
    float *__restrict__ legacy_weights,
    int num_experts,
    int top_k,
    bool normalize_weights,
    bool write_legacy_outputs,
        const int32_t *__restrict__ active_row_count_device)
{
    rocm_moe_softmax_topk_decode_independent_body<UpdateRuntimeHistogram, RowPolicy, false>(
        logits, runtime, legacy_indices, legacy_weights, num_experts, top_k,
        normalize_weights, write_legacy_outputs, active_row_count_device, {});
}

/** @brief Fused entry publishes arena descriptors from the same authenticated route lanes. */
template <bool UpdateRuntimeHistogram,
          llaminar2::RoutedExpertRowExecutionPolicy RowPolicy = llaminar2::RoutedExpertRowExecutionPolicy::ParticipantAssigned>
__global__ void rocm_moe_softmax_topk_decode_descriptors_kernel(
    float *__restrict__ logits,
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    float *__restrict__ legacy_indices,
    float *__restrict__ legacy_weights,
    int num_experts,
    int top_k,
    bool normalize_weights,
    bool write_legacy_outputs,
        const int32_t *__restrict__ active_row_count_device,
        llaminar2::MoEDecodeDescriptorPublication publication)
{
    rocm_moe_softmax_topk_decode_independent_body<UpdateRuntimeHistogram, RowPolicy, true>(
        logits, runtime, legacy_indices, legacy_weights, num_experts, top_k,
        normalize_weights, write_legacy_outputs, active_row_count_device, publication);
}

/**
 * @brief Paired original FP16 dot teams followed by last-producer route publication.
 * @tparam UpdateRuntimeHistogram Original integer demand-publication policy.
 * @tparam RowPolicy Original independent placement/fragment execution contract.
 * @param hidden Original FP32 hidden row. @param gate_weights Native FP16 rows.
 * @param logits Original public logits/probabilities, unchanged in extent.
 * @param runtime Original acquired device runtime.
 * @param legacy_indices Original complete selected IDs. @param legacy_weights Original selected weights.
 * @param d_model Original reduction width. @param num_experts Exact source expert extent.
 * @param top_k Original selected width. @param normalize_weights Serial normalization policy.
 * @param write_legacy_outputs Original complete-output publication policy.
 * @param active_row_count_device Device row count; inactive rows still write probabilities.
 * @param publication Original arena destinations and descriptor precision.
 * @param completion One initialized exclusive counter for these physical producers.
 * @details Each128-lane team keeps its original half2/scalar K stride and
 * two-wave reduction. The odd final team reaches every CTA barrier. Release
 * arrival publishes both logits; only the final producer acquires and enters
 * the unchanged256-lane softmax body. Graph order separates counter epochs.
 */
template <bool UpdateRuntimeHistogram, llaminar2::RoutedExpertRowExecutionPolicy RowPolicy>
__global__ void rocm_moe_fp16_projection_route_descriptors_kernel(
    const float *__restrict__ hidden, const __half *__restrict__ gate_weights,
    float *__restrict__ logits, DeviceMoELayerRuntimeView *__restrict__ runtime,
    float *__restrict__ legacy_indices, float *__restrict__ legacy_weights,
    int d_model, int num_experts, int top_k, bool normalize_weights,
    bool write_legacy_outputs, const int32_t *__restrict__ active_row_count_device,
    llaminar2::MoEDecodeDescriptorPublication publication,
    llaminar2::NativeTileCompletionView completion)
{
    constexpr int TeamThreads = 128;
    const int team = threadIdx.x / TeamThreads;
    const int team_lane = threadIdx.x % TeamThreads;
    const int expert = static_cast<int>(blockIdx.x) * 2 + team;
    float sum = 0.0f;
    if (expert < num_experts)
    {
        const __half *w = gate_weights + static_cast<size_t>(expert) * d_model;
        if ((d_model & 1) == 0)
        {
            const int pair_count = d_model >> 1;
            const __half2 *w2 = reinterpret_cast<const __half2 *>(w);
            for (int i = team_lane; i < pair_count; i += TeamThreads)
            {
                const float2 ww = __half22float2(w2[i]);
                const int h = i << 1;
                sum += router_fp16_operand(hidden[h]) * ww.x + router_fp16_operand(hidden[h + 1]) * ww.y;
            }
        }
        else
        {
            for (int i = team_lane; i < d_model; i += TeamThreads)
                sum += router_fp16_operand(hidden[i]) * __half2float(w[i]);
        }
    }
    sum = rocm_moe_wave_reduce_sum_f32(sum);
    __shared__ float team_wave_sums[4];
    const int lane = team_lane % warpSize;
    const int team_wave = team_lane / warpSize;
    if (lane == 0) team_wave_sums[team * 2 + team_wave] = sum;
    __syncthreads();
    if (team_wave == 0)
    {
        sum = lane < 2 ? team_wave_sums[team * 2 + lane] : 0.0f;
        sum = rocm_moe_wave_reduce_sum_f32(sum);
        if (lane == 0 && expert < num_experts) logits[expert] = sum;
    }
    // Every writer joins before arrival. No producer waits for blocks that
    // might not yet be resident; their scheduling order cannot change math.
    __syncthreads();
    __shared__ llaminar2::NativeTileCompletionDecision decision;
    if (threadIdx.x == 0)
        decision = llaminar2::native_tile_completion::arriveForPublication(
            completion.counters, completion.contributors);
    __syncthreads();
    if (decision == llaminar2::NativeTileCompletionDecision::Invalid) __builtin_trap();
    if (decision != llaminar2::NativeTileCompletionDecision::Publish) return;
    rocm_moe_softmax_topk_decode_independent_body<UpdateRuntimeHistogram, RowPolicy, true, true>(
        logits, runtime, legacy_indices, legacy_weights, num_experts, top_k,
        normalize_weights, write_legacy_outputs, active_row_count_device, publication);
    // Publication returns for lanes without selected slots. The caller still
    // joins every lane before resetting the counter and retiring this epoch.
    __syncthreads();
    if (threadIdx.x == 0) llaminar2::native_tile_completion::reset(completion.counters);
}

/**
 * @brief LDS-cached 8-expert/CTA FP16 router projection + last-producer softmax/topk.
 *
 * Loads and half-RNE-converts hidden[0..d_model-1] once per 8-expert CTA into dynamic LDS,
 * then executes 2 teams of 128 threads x 4 interleaved experts using the exact same
 * stride-128 pair order and wave64/2-warp reduction order as the baseline kernel.
 * Uses 256 threads/CTA so the final arriving producer runs the unchanged 256-lane
 * softmax/top-k/descriptor-publication body with zero barrier divergence.
 */
template <bool UpdateRuntimeHistogram, llaminar2::RoutedExpertRowExecutionPolicy RowPolicy>
__global__ __launch_bounds__(256) void rocm_moe_fp16_projection_route_descriptors_lds_kernel(
    const float *__restrict__ hidden, const __half *__restrict__ gate_weights,
    float *__restrict__ logits, DeviceMoELayerRuntimeView *__restrict__ runtime,
    float *__restrict__ legacy_indices, float *__restrict__ legacy_weights,
    int d_model, int num_experts, int top_k, bool normalize_weights,
    bool write_legacy_outputs, const int32_t *__restrict__ active_row_count_device,
    llaminar2::MoEDecodeDescriptorPublication publication,
    llaminar2::NativeTileCompletionView completion)
{
    extern __shared__ float sh_hidden[];
    for (int i = threadIdx.x; i < d_model; i += 256)
        sh_hidden[i] = router_fp16_operand(hidden[i]);
    __syncthreads();

    constexpr int TeamThreads = 128;
    const int team = threadIdx.x >> 7;        // 0 or 1
    const int team_lane = threadIdx.x & 127;  // 0..127 (exact baseline logical lane)
    const int base_e = static_cast<int>(blockIdx.x) * 8 + team * 4;
    const int e0 = base_e + 0;
    const int e1 = base_e + 1;
    const int e2 = base_e + 2;
    const int e3 = base_e + 3;
    const bool has0 = (e0 < num_experts);
    const bool has1 = (e1 < num_experts);
    const bool has2 = (e2 < num_experts);
    const bool has3 = (e3 < num_experts);

    const int pair_count = d_model >> 1;
    const __half2 *w0 = has0 ? reinterpret_cast<const __half2 *>(gate_weights + static_cast<size_t>(e0) * d_model) : nullptr;
    const __half2 *w1 = has1 ? reinterpret_cast<const __half2 *>(gate_weights + static_cast<size_t>(e1) * d_model) : nullptr;
    const __half2 *w2 = has2 ? reinterpret_cast<const __half2 *>(gate_weights + static_cast<size_t>(e2) * d_model) : nullptr;
    const __half2 *w3 = has3 ? reinterpret_cast<const __half2 *>(gate_weights + static_cast<size_t>(e3) * d_model) : nullptr;

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    if (has0)
    {
        for (int i = team_lane; i < pair_count; i += TeamThreads)
        {
            const int h = i << 1;
            const float h0 = sh_hidden[h];
            const float h1 = sh_hidden[h + 1];
            const float2 ww0 = __half22float2(w0[i]);
            sum0 += h0 * ww0.x + h1 * ww0.y;
            if (has1) { const float2 ww1 = __half22float2(w1[i]); sum1 += h0 * ww1.x + h1 * ww1.y; }
            if (has2) { const float2 ww2 = __half22float2(w2[i]); sum2 += h0 * ww2.x + h1 * ww2.y; }
            if (has3) { const float2 ww3 = __half22float2(w3[i]); sum3 += h0 * ww3.x + h1 * ww3.y; }
        }
    }

    sum0 = rocm_moe_wave_reduce_sum_f32(sum0);
    sum1 = rocm_moe_wave_reduce_sum_f32(sum1);
    sum2 = rocm_moe_wave_reduce_sum_f32(sum2);
    sum3 = rocm_moe_wave_reduce_sum_f32(sum3);

    __shared__ float team_wave_sums[2][4][2];
    const int lane = team_lane & 63;
    const int team_wave = team_lane >> 6; // 0 or 1
    if (lane == 0)
    {
        team_wave_sums[team][0][team_wave] = sum0;
        team_wave_sums[team][1][team_wave] = sum1;
        team_wave_sums[team][2][team_wave] = sum2;
        team_wave_sums[team][3][team_wave] = sum3;
    }
    __syncthreads();

    if (team_wave == 0)
    {
        if (has0) { float s = lane < 2 ? team_wave_sums[team][0][lane] : 0.0f; s = rocm_moe_wave_reduce_sum_f32(s); if (lane == 0) logits[e0] = s; }
        if (has1) { float s = lane < 2 ? team_wave_sums[team][1][lane] : 0.0f; s = rocm_moe_wave_reduce_sum_f32(s); if (lane == 0) logits[e1] = s; }
        if (has2) { float s = lane < 2 ? team_wave_sums[team][2][lane] : 0.0f; s = rocm_moe_wave_reduce_sum_f32(s); if (lane == 0) logits[e2] = s; }
        if (has3) { float s = lane < 2 ? team_wave_sums[team][3][lane] : 0.0f; s = rocm_moe_wave_reduce_sum_f32(s); if (lane == 0) logits[e3] = s; }
    }
    __syncthreads();
    __shared__ llaminar2::NativeTileCompletionDecision decision;
    if (threadIdx.x == 0)
        decision = llaminar2::native_tile_completion::arriveForPublication(
            completion.counters, completion.contributors);
    __syncthreads();
    if (decision == llaminar2::NativeTileCompletionDecision::Invalid) __builtin_trap();
    if (decision != llaminar2::NativeTileCompletionDecision::Publish) return;
    rocm_moe_softmax_topk_decode_independent_body<UpdateRuntimeHistogram, RowPolicy, true, true>(
        logits, runtime, legacy_indices, legacy_weights, num_experts, top_k,
        normalize_weights, write_legacy_outputs, active_row_count_device, publication);
    __syncthreads();
    if (threadIdx.x == 0) llaminar2::native_tile_completion::reset(completion.counters);
}

/** @brief Admitted native wave64 bitonic register-merge top-10 selector over 512 experts. */
__device__ __forceinline__ void moe_grouped_topk_register_merge(
    const float* __restrict__ probabilities, int* __restrict__ selected_ids,
    float* __restrict__ selected_weights)
{
    using Key = unsigned long long;
    constexpr int wave_width = 64;
    constexpr int frontier_per_wave = 16;
    constexpr int id_bits = 10;  // Includes the nonzero tie key of expert zero.
    static_assert(kQFNRegisterSelectorExperts < (1u << id_bits));
    const unsigned lane = threadIdx.x % wave_width;
    const unsigned wave = threadIdx.x / wave_width;

    const auto pack = [&](int id) -> Key {
        const float probability = probabilities[id];
        if (!(probability >= 0.0f)) return 0;
        const unsigned bits = probability == 0.0f ? 0u : __float_as_uint(probability);
        return (static_cast<Key>(bits) << id_bits) | static_cast<unsigned>(kQFNRegisterSelectorExperts - id);
    };
    const auto exchange = [&](Key key, int distance, bool keep_high) -> Key {
        const Key other = __shfl_xor(key, distance, wave_width);
        return keep_high ? max(key, other) : min(key, other);
    };

    Key first = pack(threadIdx.x);
    Key second = pack(threadIdx.x + kQFNRegisterSelectorExperts / 2);
    // Sorting the two runs in opposite directions forms one bitonic 128-key run.
    #pragma unroll
    for (int width = 2; width <= wave_width; width *= 2) {
        #pragma unroll
        for (int distance = width / 2; distance > 0; distance /= 2) {
            const bool high = ((lane & width) == 0) == ((lane & distance) == 0);
            first = exchange(first, distance, high);
            second = exchange(second, distance, !high);
        }
    }
    first = max(first, second);
    #pragma unroll
    for (int distance = wave_width / 2; distance > 0; distance /= 2)
        first = exchange(first, distance, (lane & distance) == 0);

    __shared__ Key frontiers[wave_width];
    if (lane < frontier_per_wave) {
        const unsigned in_run = (wave & 1) ? frontier_per_wave - 1 - lane : lane;
        frontiers[wave * frontier_per_wave + in_run] = first;
    }
    __syncthreads();
    if (wave == 0) {
        Key key = frontiers[lane];
        // Each adjacent pair is already bitonic. Two merges complete the sort.
        #pragma unroll
        for (int width = 2 * frontier_per_wave; width <= wave_width; width *= 2)
            #pragma unroll
            for (int distance = width / 2; distance > 0; distance /= 2)
                key = exchange(key, distance, ((lane & width) == 0) == ((lane & distance) == 0));

        const int id = key ? kQFNRegisterSelectorExperts - int(key & ((1u << id_bits) - 1))
                           : kDeviceMoEMaxExperts;
        int consumed_id = id;
        if (__ballot(lane < kQFNRegisterSelectorTopK && !key)) {
            // Serial selection marks each consumed score -infinity. If too few
            // finite experts exist, its smallest consumed ID wins the remaining
            // rounds; an all-NaN row retains the original sentinel.
            consumed_id = lane < kQFNRegisterSelectorTopK ? id : kDeviceMoEMaxExperts;
            #pragma unroll
            for (int distance = wave_width / 2; distance > 0; distance /= 2)
                consumed_id = min(consumed_id, __shfl_xor(consumed_id, distance, wave_width));
        }
        if (lane < kQFNRegisterSelectorTopK) {
            selected_ids[lane] = key ? id : consumed_id;
            selected_weights[lane] = key ? probabilities[id] : -INFINITY;
        }
    }
    // The normalization and ledger publisher consumes all ten wave results.
    __syncthreads();
}

/**
 * @brief Decode-equivalent softmax/top-k for grouped prefill rows.
 *
 * The MTP verifier is allowed to publish grouped rows into live decode state, so
 * its routing weights must be byte-identical to running each row through the
 * production single-token decode router.  This kernel keeps the grouped launch
 * economical (one block per verifier row) while intentionally mirroring
 * rocm_moe_softmax_topk_decode_runtime_wave64_kernel's probability reduction,
 * top-k selection, and top-k renormalization order inside each block.
 */
template<bool Partitioned, bool RegisterSelector>
__device__ __forceinline__ void grouped_softmax_rows_body(
    float *__restrict__ logits,                       // [seq_len, num_experts] IN logits, OUT probabilities
    float *__restrict__ expert_indices,               // [seq_len, top_k] OUT
    float *__restrict__ expert_weights,               // [seq_len, top_k] OUT
    int seq_len, int num_experts, int top_k,
    bool normalize_weights,
    const int *__restrict__ device_effective_seq_len,
    DeviceMoELayerRuntimeView *__restrict__ deferred_selected_route_ledger,
    llaminar2::DeviceRowPartition partition = {},
    llaminar2::MoERouterSelectedRoute *selected_packet = nullptr,
    std::uint64_t *selected_bytes = nullptr)
{
    if constexpr (Partitioned)
    {
        // Complete-domain history can only be published after the explicit
        // selection exchange, never from this participant's compact fragment.
        seq_len = partition.resolve(seq_len, device_effective_seq_len).count;
        device_effective_seq_len = nullptr;
        // Zero-row owners must still advance the collective protocol. The
        // stream dependency publishes the complete packet, not this scalar
        // alone, after all CTAs finish their unchanged top-k arithmetic.
        if (blockIdx.x == 0 && threadIdx.x == 0)
            *selected_bytes = std::uint64_t(seq_len) * top_k * sizeof(*selected_packet);
    }
    const int token = blockIdx.x;
    const int active_rows = device_effective_seq_len
                                ? (seq_len == 1 ? (runtime_decode_row_active(device_effective_seq_len) ? 1 : 0)
                                                : max(0, min(seq_len, *device_effective_seq_len)))
                                : seq_len;
    if (!logits || (!Partitioned && (!expert_indices || !expert_weights)) ||
        token >= seq_len || num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK)
    {
        return;
    }

    const int total_slots = seq_len * top_k;
    const bool ledger_is_complete =
        deferred_selected_route_ledger == nullptr ||
        (deferred_selected_route_ledger
             ->deferred_verifier_route_expert_ids != nullptr &&
         deferred_selected_route_ledger
             ->deferred_verifier_route_participant_ids != nullptr &&
         deferred_selected_route_ledger->deferred_verifier_route_capacity >=
             static_cast<uint32_t>(total_slots));
    if (!ledger_is_complete)
    {
        if (token == 0 && threadIdx.x == 0)
        {
            FAIL_FAST_INVALID_GROUPED_VERIFIER_COMMIT(
                "overlay verifier routing requires a complete per-layer "
                "deferred selected-route ledger");
        }
        return;
    }

    if (token >= active_rows)
    {
        for (int slot = threadIdx.x; slot < top_k; slot += blockDim.x)
        {
            const size_t output_slot = static_cast<size_t>(token) * top_k + slot;
            expert_indices[output_slot] = -1.0f;
            expert_weights[output_slot] = 0.0f;
            if (deferred_selected_route_ledger)
            {
                deferred_selected_route_ledger
                    ->deferred_verifier_route_expert_ids[output_slot] = -1;
                deferred_selected_route_ledger
                    ->deferred_verifier_route_participant_ids[output_slot] = -1;
            }
        }
        for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
        {
            logits[static_cast<size_t>(token) * num_experts + expert] = 0.0f;
        }
        return;
    }

    float *row_logits = logits + static_cast<size_t>(token) * num_experts;

    __shared__ float expert_probs[kDeviceMoEMaxExperts];
    __shared__ float shared_max[8];
    __shared__ float shared_sum[8];

    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;

    float max_val = -FLT_MAX;
    for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
    {
        max_val = fmaxf(max_val, row_logits[expert]);
    }
    max_val = rocm_moe_wave_reduce_max_f32(max_val);
    if (lane == 0) shared_max[warp_id] = max_val;
    __syncthreads();

    if (warp_id == 0)
    {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        max_val = (lane < num_warps) ? shared_max[lane] : -FLT_MAX;
        max_val = rocm_moe_wave_reduce_max_f32(max_val);
        if (lane == 0) shared_max[0] = max_val;
    }
    __syncthreads();
    max_val = shared_max[0];

    float sum = 0.0f;
    for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
    {
        const float value = expf(row_logits[expert] - max_val);
        expert_probs[expert] = value;
        sum += value;
    }
    sum = rocm_moe_wave_reduce_sum_f32(sum);
    if (lane == 0) shared_sum[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0)
    {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        sum = (lane < num_warps) ? shared_sum[lane] : 0.0f;
        sum = rocm_moe_wave_reduce_sum_f32(sum);
        if (lane == 0) shared_sum[0] = sum;
    }
    __syncthreads();

    const float inv_sum = (shared_sum[0] > 0.0f) ? 1.0f / shared_sum[0] : 0.0f;
    for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
    {
        const float probability = expert_probs[expert] * inv_sum;
        expert_probs[expert] = probability;
        row_logits[expert] = probability;
    }
    __syncthreads();

    __shared__ int selected_ids[kDeviceMoEMaxTopK];
    __shared__ float selected_weights[kDeviceMoEMaxTopK];
    if constexpr (RegisterSelector)
        moe_grouped_topk_register_merge(expert_probs, selected_ids, selected_weights);
    else
        moe_select_topk_probabilities_block(expert_probs, num_experts, top_k, selected_ids, selected_weights);

    if (threadIdx.x == 0)
    {
        if (normalize_weights)
        {
            float topk_sum = 0.0f;
            for (int slot = 0; slot < top_k; ++slot)
            {
                topk_sum += selected_weights[slot];
            }
            if (topk_sum > 0.0f)
            {
                const float inv_topk_sum = 1.0f / topk_sum;
                for (int slot = 0; slot < top_k; ++slot)
                {
                    selected_weights[slot] *= inv_topk_sum;
                }
            }
        }

        for (int slot = 0; slot < top_k; ++slot)
        {
            const size_t out = static_cast<size_t>(token) * top_k + slot;
            if constexpr (Partitioned)
                selected_packet[out] = {static_cast<float>(selected_ids[slot]), selected_weights[slot]};
            else
            {
                expert_indices[out] = static_cast<float>(selected_ids[slot]);
                expert_weights[out] = selected_weights[slot];
            }
            if (deferred_selected_route_ledger)
            {
                /*
                 * Ticket dispatch resolves ownership outside this GPU graph.
                 * Record global selection and leave local ownership unknown.
                 */
                const size_t output_slot =
                    static_cast<size_t>(token) * top_k + slot;
                deferred_selected_route_ledger
                    ->deferred_verifier_route_expert_ids[output_slot] =
                    selected_ids[slot];
                deferred_selected_route_ledger
                    ->deferred_verifier_route_participant_ids[output_slot] = -1;
            }
        }
    }
}

/** @brief Original grouped and partitioned softmax/top-k launch entry. */
template<bool Partitioned = false>
__global__ void rocm_moe_softmax_topk_decode_equivalent_rows_kernel(
    float *__restrict__ logits,                       // [seq_len, num_experts] IN logits, OUT probabilities
    float *__restrict__ expert_indices,               // [seq_len, top_k] OUT
    float *__restrict__ expert_weights,               // [seq_len, top_k] OUT
    int seq_len, int num_experts, int top_k,
    bool normalize_weights,
    const int *__restrict__ device_effective_seq_len,
    DeviceMoELayerRuntimeView *__restrict__ deferred_selected_route_ledger,
    llaminar2::DeviceRowPartition partition = {},
    llaminar2::MoERouterSelectedRoute *selected_packet = nullptr,
    std::uint64_t *selected_bytes = nullptr)
{
    grouped_softmax_rows_body<Partitioned, false>(logits, expert_indices, expert_weights, seq_len,
        num_experts, top_k, normalize_weights, device_effective_seq_len,
        deferred_selected_route_ledger, partition, selected_packet, selected_bytes);
}

/** @brief Qualified gfx906 register selector; all other arithmetic uses the common body. */
__global__ void qfn_grouped_softmax_register_selector_kernel(
    float *__restrict__ logits,                       // [seq_len, num_experts] IN logits, OUT probabilities
    float *__restrict__ expert_indices,               // [seq_len, top_k] OUT
    float *__restrict__ expert_weights,               // [seq_len, top_k] OUT
    int seq_len, int num_experts, int top_k,
    bool normalize_weights,
    const int *__restrict__ device_effective_seq_len,
    DeviceMoELayerRuntimeView *__restrict__ deferred_selected_route_ledger,
    llaminar2::DeviceRowPartition partition = {},
    llaminar2::MoERouterSelectedRoute *selected_packet = nullptr,
    std::uint64_t *selected_bytes = nullptr)
{
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__GFX9__)
    grouped_softmax_rows_body<false, true>(logits, expert_indices, expert_weights, seq_len, num_experts,
        top_k, normalize_weights, device_effective_seq_len, deferred_selected_route_ledger,
        partition, selected_packet, selected_bytes);
#else
    // An incompatible code object can never be selected by the host.
    __builtin_trap();
#endif
}

__global__ __launch_bounds__(kDeviceMoEMaxExperts) void rocm_moe_router_kpart_reduce_softmax_topk_decode_runtime_kernel(
    const float *__restrict__ partials,                // [num_experts, k_partitions]
    float *__restrict__ router_probabilities,          // [num_experts] OUT
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    DeviceMoELayerRuntimeView *__restrict__ runtime_layers,
    float *__restrict__ legacy_indices,                // optional [top_k]
    float *__restrict__ legacy_weights,                // optional [top_k]
    int num_experts, int k_partitions, int top_k,
    bool normalize_weights,
    bool write_legacy_outputs,
    bool update_runtime_histogram,
    llaminar2::RoutedExpertRowExecutionPolicy row_execution_policy,
    const int32_t *__restrict__ absolute_position_ids,
    const DeviceMoERebalancePlanEntryView *__restrict__ rebalance_plan_entries,
    uint32_t rebalance_plan_capacity,
    DeviceMoERebalanceCommandBufferHeaderView *__restrict__ rebalance_command_header,
    const DeviceMoEExpertDirectoryEntryView *__restrict__ rebalance_local_transfer_slots,
    uint32_t rebalance_local_transfer_slot_count,
    DeviceMoERebalanceConfigView rebalance_config,
    DeviceMoERebalanceApplyStatusView *__restrict__ rebalance_apply_status,
    DeviceMoERebalanceGraphControllerStateView *__restrict__ rebalance_controller_state,
    int rebalance_target_layer,
    uint32_t rebalance_command_buffer_count,
        const int32_t *__restrict__ active_row_count_device)
{
    if (!partials || !router_probabilities || !runtime ||
        num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        k_partitions <= 0 || top_k <= 0 || top_k > kDeviceMoEMaxTopK)
    {
        return;
    }

    try_apply_ready_rebalance_wave_for_layer_thread0(
        runtime_layers,
        rebalance_plan_entries,
        rebalance_plan_capacity,
        rebalance_command_header,
        rebalance_local_transfer_slots,
        rebalance_local_transfer_slot_count,
        rebalance_config,
        rebalance_apply_status,
        rebalance_controller_state,
        rebalance_target_layer,
        rebalance_command_buffer_count);

    __shared__ float expert_probs[kDeviceMoEMaxExperts];

    float local_max = -FLT_MAX;
    for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
    {
        const float *expert_partials = partials + expert * k_partitions;
        float sum = 0.0f;
        for (int partition = 0; partition < k_partitions; ++partition)
        {
            sum += expert_partials[partition];
        }
        expert_probs[expert] = sum;
        local_max = fmaxf(local_max, sum);
    }

    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    local_max = rocm_moe_wave_reduce_max_f32(local_max);

    __shared__ float shared_max[8];
    if (lane == 0) shared_max[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0)
    {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        local_max = (lane < num_warps) ? shared_max[lane] : -FLT_MAX;
        local_max = rocm_moe_wave_reduce_max_f32(local_max);
        if (lane == 0) shared_max[0] = local_max;
    }
    __syncthreads();
    const float max_val = shared_max[0];

    float local_sum = 0.0f;
    for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
    {
        const float value = expf(expert_probs[expert] - max_val);
        expert_probs[expert] = value;
        local_sum += value;
    }
    local_sum = rocm_moe_wave_reduce_sum_f32(local_sum);

    __shared__ float shared_sum[8];
    if (lane == 0) shared_sum[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0)
    {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        local_sum = (lane < num_warps) ? shared_sum[lane] : 0.0f;
        local_sum = rocm_moe_wave_reduce_sum_f32(local_sum);
        if (lane == 0) shared_sum[0] = local_sum;
    }
    __syncthreads();

    const float inv_sum = (shared_sum[0] > 0.0f) ? 1.0f / shared_sum[0] : 0.0f;
    for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
    {
        const float probability = expert_probs[expert] * inv_sum;
        expert_probs[expert] = probability;
        router_probabilities[expert] = probability;
    }
    __syncthreads();

    __shared__ int selected_ids[kDeviceMoEMaxTopK];
    __shared__ float selected_weights[kDeviceMoEMaxTopK];
    moe_select_topk_probabilities_block(expert_probs, num_experts, top_k, selected_ids, selected_weights);

    if (threadIdx.x == 0)
    {
        if (normalize_weights)
        {
            float topk_sum = 0.0f;
            for (int slot = 0; slot < top_k; ++slot)
            {
                topk_sum += selected_weights[slot];
            }
            if (topk_sum > 0.0f)
            {
                const float inv_topk_sum = 1.0f / topk_sum;
                for (int slot = 0; slot < top_k; ++slot)
                {
                    selected_weights[slot] *= inv_topk_sum;
                }
            }
        }

        runtime_publish_decode_dispatch(
            runtime, selected_ids, selected_weights, num_experts, top_k,
            legacy_indices, legacy_weights, write_legacy_outputs,
            update_runtime_histogram,
            row_execution_policy,
            absolute_position_ids ? absolute_position_ids[0] : -1, active_row_count_device);
    }
}

// =========================================================================
// Kernel 2c: Decode Route Select Runtime Copy
//
// Copies single-token top-k ids/weights into the stable device runtime table,
// optionally mirrors them to legacy routing tensors, and records a device-side
// decode histogram. The routing tensors remain necessary until the expert
// decode kernels consume runtime top-k directly.
// =========================================================================

// The bridge launches 32 lanes, not HIP's default 1024-lane maximum.
__global__ __launch_bounds__(32) void rocm_moe_decode_route_select_runtime_kernel(
    const int *__restrict__ expert_indices,       // [top_k]
    const float *__restrict__ expert_weights,     // [top_k]
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    float *__restrict__ legacy_indices,           // optional [top_k]
    float *__restrict__ legacy_weights,           // optional [top_k]
    int num_experts, int top_k,
    bool write_legacy_outputs,
    bool update_runtime_histogram,
        llaminar2::RoutedExpertRowExecutionPolicy row_execution_policy,
    const int32_t *__restrict__ absolute_position_ids)
{
    if (threadIdx.x != 0 || !runtime) return;
    runtime_publish_decode_dispatch(
        runtime, expert_indices, expert_weights, num_experts, top_k,
        legacy_indices, legacy_weights, write_legacy_outputs,
        update_runtime_histogram,
        row_execution_policy,
        absolute_position_ids ? absolute_position_ids[0] : -1, nullptr);
}

// =========================================================================
// Kernel 3: Gather Tokens
//
// Copy selected token rows into a contiguous batch buffer.
// Each block handles one token copy; threads cover the d_model dimension.
// =========================================================================

__global__ void rocm_moe_gather_tokens_kernel(
    const float *__restrict__ hidden,        // [seq_len, d_model]
    float *__restrict__ batch_buffer,        // [num_tokens, d_model]
    const int *__restrict__ token_indices,   // [num_tokens]
    int num_tokens, int d_model)
{
    const int token = blockIdx.x;
    if (token >= num_tokens) return;

    const int src_row = token_indices[token];
    const float *src = hidden + src_row * d_model;
    float *dst = batch_buffer + token * d_model;

    // Vectorized copy: try float4 first
    const int d_model4 = d_model >> 2;
    const float4 *src4 = reinterpret_cast<const float4 *>(src);
    float4 *dst4 = reinterpret_cast<float4 *>(dst);

    for (int i = threadIdx.x; i < d_model4; i += blockDim.x)
    {
        dst4[i] = src4[i];
    }

    // Handle tail elements
    const int tail_start = d_model4 << 2;
    for (int i = tail_start + threadIdx.x; i < d_model; i += blockDim.x)
    {
        dst[i] = src[i];
    }
}

/**
 * @brief Copy one row using a by-value row index.
 *
 * Verifier replay uses this instead of the general gather kernel so graph and
 * stream replay do not depend on an async H2D copy from a stack-owned host row
 * index.  The row width is normally a multiple of four, but the scalar tail
 * keeps the helper valid for routing rows such as top_k=8 as well.
 */
__global__ void rocm_moe_copy_row_kernel(
    const float *__restrict__ source,
    float *__restrict__ row_buffer,
    int row_index,
    int row_width)
{
    if (row_index < 0 || row_width <= 0)
        return;

    const float *src = source + static_cast<size_t>(row_index) * row_width;
    const int row_width4 = row_width >> 2;
    const float4 *src4 = reinterpret_cast<const float4 *>(src);
    float4 *dst4 = reinterpret_cast<float4 *>(row_buffer);
    for (int i = threadIdx.x; i < row_width4; i += blockDim.x)
        dst4[i] = src4[i];

    const int tail_start = row_width4 << 2;
    for (int i = tail_start + threadIdx.x; i < row_width; i += blockDim.x)
        row_buffer[i] = src[i];
}

// =========================================================================
// Kernel 4: Scatter-Add Weighted
//
// output[token_indices[i]] += weights[i] * expert_output[i]
// Each block handles one token; threads cover d_model.
// =========================================================================

__global__ void rocm_moe_scatter_add_kernel(
    float *__restrict__ output,             // [seq_len, d_model]
    const float *__restrict__ expert_output, // [num_tokens, d_model]
    const int *__restrict__ token_indices,   // [num_tokens]
    const float *__restrict__ weights,       // [num_tokens]
    int num_tokens, int d_model)
{
    const int token = blockIdx.x;
    if (token >= num_tokens) return;

    const int dst_row = token_indices[token];
    const float w = weights[token];
    const float *src = expert_output + token * d_model;
    float *dst = output + dst_row * d_model;

    for (int i = threadIdx.x; i < d_model; i += blockDim.x)
    {
        atomicAdd(&dst[i], w * src[i]);
    }
}

/**
 * @brief Store one scratch row into a destination row selected by value.
 *
 * The operation is an overwrite, not an add.  Decode-equivalent verifier replay
 * owns each destination row exactly once, so direct store avoids atomics and
 * avoids staging a `{row, weight}` pair through host memory.
 */
__global__ void rocm_moe_write_row_kernel(
    float *__restrict__ destination,
    const float *__restrict__ row_buffer,
    int row_index,
    int row_width)
{
    if (row_index < 0 || row_width <= 0)
        return;

    float *dst = destination + static_cast<size_t>(row_index) * row_width;
    const int row_width4 = row_width >> 2;
    const float4 *src4 = reinterpret_cast<const float4 *>(row_buffer);
    float4 *dst4 = reinterpret_cast<float4 *>(dst);
    for (int i = threadIdx.x; i < row_width4; i += blockDim.x)
        dst4[i] = src4[i];

    const int tail_start = row_width4 << 2;
    for (int i = tail_start + threadIdx.x; i < row_width; i += blockDim.x)
        dst[i] = row_buffer[i];
}

// =========================================================================
// Kernel 5: Shared Expert Gate
//
// For each token:
//   gate = sigmoid(dot(gate_inp, input[t]))
//   shared_output[t] *= gate
//
// Two-phase: first compute dot products (reduction), then scale.
// =========================================================================

__global__ void rocm_moe_shared_expert_gate_dot_kernel(
    const float *__restrict__ input,        // [seq_len, d_model]
    const float *__restrict__ gate_inp,     // [d_model]
    float *__restrict__ gate_values,        // [seq_len] scratch
    int seq_len, int d_model)
{
    const int token = blockIdx.x;
    if (token >= seq_len) return;

    const float *x = input + token * d_model;

    /*
     * MTP verifier rows compare the grouped runtime-M shared-gate path against
     * serial M=1 decode row replay.  The serial ROCm decode kernel below
     * accumulates four contiguous FP32 products at a time before participating
     * in the block reduction.  Use the same per-lane accumulation order here;
     * a scalar stride loop is mathematically equivalent, but it changes FP32
     * rounding by a few ulps and those differences can amplify after later MoE
     * and residual stages.
     */
    float sum = 0.0f;
    if ((d_model & 3) == 0)
    {
        const int d4 = d_model >> 2;
        const float4 *__restrict__ input4 = reinterpret_cast<const float4 *>(x);
        const float4 *__restrict__ gate4 = reinterpret_cast<const float4 *>(gate_inp);
        for (int i = threadIdx.x; i < d4; i += blockDim.x)
        {
            const float4 xv = input4[i];
            const float4 gv = gate4[i];
            sum += xv.x * gv.x + xv.y * gv.y + xv.z * gv.z + xv.w * gv.w;
        }
    }
    else
    {
        for (int i = threadIdx.x; i < d_model; i += blockDim.x)
        {
            sum += x[i] * gate_inp[i];
        }
    }

    // Wave reduction
    sum = rocm_moe_wave_reduce_sum_f32(sum);

    __shared__ float warp_sums[8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0)
    {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        sum = (lane < nw) ? warp_sums[lane] : 0.0f;
        sum = rocm_moe_wave_reduce_sum_f32(sum);
        if (lane == 0)
        {
            // Apply sigmoid
            gate_values[token] = 1.0f / (1.0f + expf(-sum));
        }
    }
}

__device__ __forceinline__ int rocm_moe_clamp_effective_seq_len(
    int seq_len,
    const int *__restrict__ device_effective_seq_len)
{
    if (!device_effective_seq_len)
        return seq_len;
    const int raw = *device_effective_seq_len;
    return raw < 0 ? 0 : (raw > seq_len ? seq_len : raw);
}

__global__ void rocm_moe_shared_expert_gate_scale_kernel(
    float *__restrict__ shared_output,      // [seq_len, d_model]
    const float *__restrict__ gate_values,  // [seq_len]
    int seq_len, int d_model,
    const int *__restrict__ device_effective_seq_len = nullptr)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = seq_len * d_model;
    if (idx >= total) return;

    const int token = idx / d_model;
    const int effective_seq_len =
        rocm_moe_clamp_effective_seq_len(seq_len, device_effective_seq_len);
    if (token >= effective_seq_len)
    {
        shared_output[idx] = 0.0f;
        return;
    }
    shared_output[idx] = llaminar2::device_fp32_contract::multiply(
        gate_values[token], shared_output[idx]);
}

/**
 * @brief Preserve scalar shared-gate arithmetic while honoring an optional device-owned live row.
 * @param input Original full hidden row; unread when the clamped count is zero.
 * @param gate_inp Prepared FP32 gate vector in the same original reduction order.
 * @param shared_output In-place gated row; overwritten with positive zero when inactive.
 * @param d_model Complete model width.
 * @param device_effective_seq_len Optional stable device INT32; null is ordinary scalar decode.
 */
__global__ void rocm_moe_shared_expert_gate_decode_fused_kernel(
    const float *__restrict__ input,        // [d_model], token 0
    const float *__restrict__ gate_inp,     // [d_model]
    float *__restrict__ shared_output,      // [d_model], token 0
    int d_model,
    const int *__restrict__ device_effective_seq_len)
{
    // Preserve the established shared-gate clamp contract while keeping the
    // live=1 dot/tree identical to uncounted scalar decode. Empty work never
    // reads the hidden/gate row and overwrites prior shared bytes explicitly.
    if (rocm_moe_clamp_effective_seq_len(1, device_effective_seq_len) == 0)
    {
        for (int i = threadIdx.x; i < d_model; i += blockDim.x)
            shared_output[i] = 0.0f;
        return;
    }
    float sum = 0.0f;
    if ((d_model & 3) == 0)
    {
        const int d4 = d_model >> 2;
        const float4 *__restrict__ input4 = reinterpret_cast<const float4 *>(input);
        const float4 *__restrict__ gate4 = reinterpret_cast<const float4 *>(gate_inp);
        for (int i = threadIdx.x; i < d4; i += blockDim.x)
        {
            const float4 x = input4[i];
            const float4 g = gate4[i];
            sum += x.x * g.x + x.y * g.y + x.z * g.z + x.w * g.w;
        }
    }
    else
    {
        for (int i = threadIdx.x; i < d_model; i += blockDim.x)
        {
            sum += input[i] * gate_inp[i];
        }
    }

    sum = rocm_moe_wave_reduce_sum_f32(sum);

    __shared__ float warp_sums[16];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0)
    {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        float block_sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        block_sum = rocm_moe_wave_reduce_sum_f32(block_sum);
        if (lane == 0)
            warp_sums[0] = 1.0f / (1.0f + expf(-block_sum));
    }
    __syncthreads();

    const float gate = warp_sums[0];
    if ((d_model & 3) == 0)
    {
        const int d4 = d_model >> 2;
        float4 *__restrict__ shared4 = reinterpret_cast<float4 *>(shared_output);
        for (int i = threadIdx.x; i < d4; i += blockDim.x)
        {
            const float4 s = shared4[i];
            shared4[i] = make_float4(
                llaminar2::device_fp32_contract::multiply(gate, s.x),
                llaminar2::device_fp32_contract::multiply(gate, s.y),
                llaminar2::device_fp32_contract::multiply(gate, s.z),
                llaminar2::device_fp32_contract::multiply(gate, s.w));
        }
    }
    else
    {
        for (int i = threadIdx.x; i < d_model; i += blockDim.x)
        {
            shared_output[i] = llaminar2::device_fp32_contract::multiply(gate, shared_output[i]);
        }
    }
}

/**
 * @brief Gate a full row or its disjoint output-column slice with the identical full-width dot.
 * @tparam ColumnSlice Select disjoint output columns without changing arithmetic.
 *
 * Input stride and sigmoid reduction always use the full model width. Only
 * output addressing narrows; inactive rows overwrite both outputs with zero.
 * Standalone/full/column paths share explicit RNE product and sum boundaries.
 */
template<bool ColumnSlice = false>
__global__ void rocm_moe_shared_expert_gate_add_kernel(
    const float *__restrict__ input,            // [seq_len, d_model]
    const float *__restrict__ gate_inp,         // [d_model]
    float *__restrict__ shared_output,          // [seq_len, d_model]
    const float *__restrict__ routed_residual,  // [seq_len, d_model]
    float *__restrict__ combined_output,        // [seq_len, d_model]
    int seq_len, int d_model,
    const int *__restrict__ device_effective_seq_len = nullptr,
    int output_columns = 0)
{
    llaminar2::rocm::moe_shared_gate::gateAdd<ColumnSlice>(input,gate_inp,shared_output,routed_residual,
        combined_output,seq_len,d_model,int(blockIdx.x),
        rocm_moe_clamp_effective_seq_len(seq_len,device_effective_seq_len),output_columns);
}

/**
 * @brief Publish one shared partial into a rank-addressed canonical bank.
 *
 * Route slots occupy the prefix of @p canonical_publication and are owned by
 * the routed-expert pipeline. This kernel overwrites the complete participant
 * suffix: the local participant bank receives @p shared_output and every peer
 * bank receives exact zero. A following RCCL reduce therefore transports
 * shared evidence without allowing RCCL to select its FP32 addition order.
 */
__global__ __launch_bounds__(256)
void rocm_moe_publish_shared_expert_rank_bank_kernel(
    const float *__restrict__ shared_output,
    float *__restrict__ canonical_publication,
    int seq_len,
    int top_k,
    int d_model,
    int participant_index,
    int participant_count,
    const int *__restrict__ device_effective_seq_len)
{
    const int bank_row = blockIdx.y;
    const int participant = bank_row / seq_len;
    const int token = bank_row - participant * seq_len;
    if (participant >= participant_count || token >= seq_len)
        return;

    const int effective_seq_len =
        rocm_moe_clamp_effective_seq_len(
            seq_len, device_effective_seq_len);
    const bool copy_local =
        participant == participant_index && token < effective_seq_len;
    const size_t row_elements =
        static_cast<size_t>(seq_len) * static_cast<size_t>(d_model);
    const size_t route_elements =
        row_elements * static_cast<size_t>(top_k);
    const size_t source_offset =
        static_cast<size_t>(token) * static_cast<size_t>(d_model);
    const size_t destination_offset =
        route_elements +
        static_cast<size_t>(participant) * row_elements + source_offset;

    if ((d_model & 3) == 0)
    {
        const int d4 = d_model >> 2;
        const float4 *__restrict__ source4 =
            reinterpret_cast<const float4 *>(shared_output + source_offset);
        float4 *__restrict__ destination4 =
            reinterpret_cast<float4 *>(
                canonical_publication + destination_offset);
        // copy_local is uniform for this bank row. Separate value stores
        // avoid materializing a named zero vector in private memory and then
        // selecting that lvalue address instead of a direct register-zero store.
        if (copy_local)
        {
            for (int index = blockIdx.x * blockDim.x + threadIdx.x;
                 index < d4; index += gridDim.x * blockDim.x)
                destination4[index] = source4[index];
        }
        else
        {
            for (int index = blockIdx.x * blockDim.x + threadIdx.x;
                 index < d4; index += gridDim.x * blockDim.x)
                destination4[index] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        return;
    }

    for (int column = blockIdx.x * blockDim.x + threadIdx.x;
         column < d_model;
         column += gridDim.x * blockDim.x)
    {
        canonical_publication[destination_offset + column] =
            copy_local ? shared_output[source_offset + column] : 0.0f;
    }
}

/**
 * @brief Finalize routed slots and shared rank banks in deterministic order.
 *
 * The gate-dot geometry and wave reduction exactly match
 * rocm_moe_shared_expert_gate_add_kernel. A two-dimensional launch gives each
 * token/output-column tile its own workgroup. Every tile recomputes the same
 * fixed-order gate value, avoiding both inter-workgroup publication and a
 * second launch while preserving a bit-identical gate. Route slots are then
 * folded in router order and participant banks in ascending rank order using
 * explicit round-to-nearest FP32 operations. The operation is atomic-free,
 * allocation-free, transfer-free, and graph-capturable.
 */
__global__ __launch_bounds__(256)
void rocm_moe_finalize_canonical_publication_kernel(
    const float *__restrict__ input,
    const float *__restrict__ gate_inp,
    const float *__restrict__ canonical_publication,
    float *__restrict__ routed_output,
    float *__restrict__ shared_output,
    float *__restrict__ combined_output,
    int seq_len,
    int top_k,
    int d_model,
    int participant_count,
    const int *__restrict__ device_effective_seq_len)
{
    const int output_tile = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int first_column =
        output_tile * kCanonicalMoEOutputTileColumns;
    const int unbounded_last_column =
        first_column + kCanonicalMoEOutputTileColumns;
    const int last_column =
        unbounded_last_column < d_model
            ? unbounded_last_column
            : d_model;
    if (token >= seq_len || first_column >= d_model)
        return;

    const size_t row_offset =
        static_cast<size_t>(token) * static_cast<size_t>(d_model);
    const int effective_seq_len =
        rocm_moe_clamp_effective_seq_len(
            seq_len, device_effective_seq_len);
    if (token >= effective_seq_len)
    {
        if ((d_model & 3) == 0)
        {
            const int first4 = first_column >> 2;
            const int last4 = last_column >> 2;
            float4 *__restrict__ routed4 =
                reinterpret_cast<float4 *>(routed_output + row_offset);
            float4 *__restrict__ shared4 =
                reinterpret_cast<float4 *>(shared_output + row_offset);
            float4 *__restrict__ combined4 =
                reinterpret_cast<float4 *>(combined_output + row_offset);
            const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int index = first4 + threadIdx.x;
                 index < last4;
                 index += blockDim.x)
            {
                routed4[index] = zero;
                shared4[index] = zero;
                combined4[index] = zero;
            }
        }
        else
        {
            for (int column = first_column + threadIdx.x;
                 column < last_column;
                 column += blockDim.x)
            {
                const size_t output_index =
                    row_offset + static_cast<size_t>(column);
                routed_output[output_index] = 0.0f;
                shared_output[output_index] = 0.0f;
                combined_output[output_index] = 0.0f;
            }
        }
        return;
    }

    const float *row_input = input + row_offset;
    float gate_dot = 0.0f;
    if ((d_model & 3) == 0)
    {
        const int d4 = d_model >> 2;
        const float4 *__restrict__ input4 =
            reinterpret_cast<const float4 *>(row_input);
        const float4 *__restrict__ gate4 =
            reinterpret_cast<const float4 *>(gate_inp);
        for (int index = threadIdx.x; index < d4; index += blockDim.x)
        {
            const float4 x = input4[index];
            const float4 g = gate4[index];
            gate_dot +=
                x.x * g.x + x.y * g.y + x.z * g.z + x.w * g.w;
        }
    }
    else
    {
        for (int column = threadIdx.x;
             column < d_model;
             column += blockDim.x)
        {
            gate_dot += row_input[column] * gate_inp[column];
        }
    }

    gate_dot = rocm_moe_wave_reduce_sum_f32(gate_dot);
    __shared__ float wave_sums[16];
    const int lane = threadIdx.x % warpSize;
    const int wave = threadIdx.x / warpSize;
    if (lane == 0)
        wave_sums[wave] = gate_dot;
    __syncthreads();

    if (wave == 0)
    {
        const int wave_count =
            (blockDim.x + warpSize - 1) / warpSize;
        float block_sum = lane < wave_count ? wave_sums[lane] : 0.0f;
        block_sum = rocm_moe_wave_reduce_sum_f32(block_sum);
        if (lane == 0)
            wave_sums[0] = 1.0f / (1.0f + expf(-block_sum));
    }
    __syncthreads();
    const float gate = wave_sums[0];

    const size_t row_elements =
        static_cast<size_t>(seq_len) * static_cast<size_t>(d_model);
    const size_t route_elements =
        row_elements * static_cast<size_t>(top_k);
    const float *rank_banks = canonical_publication + route_elements;

    if ((d_model & 3) == 0)
    {
        const int first4 = first_column >> 2;
        const int last4 = last_column >> 2;
        float4 *__restrict__ routed4 =
            reinterpret_cast<float4 *>(routed_output + row_offset);
        float4 *__restrict__ shared4 =
            reinterpret_cast<float4 *>(shared_output + row_offset);
        float4 *__restrict__ combined4 =
            reinterpret_cast<float4 *>(combined_output + row_offset);
        for (int index = first4 + threadIdx.x;
             index < last4;
             index += blockDim.x)
        {
            float4 routed = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll 1
            for (int route = 0; route < top_k; ++route)
            {
                const size_t route_row =
                    (static_cast<size_t>(token) *
                         static_cast<size_t>(top_k) +
                     static_cast<size_t>(route)) *
                    static_cast<size_t>(d_model);
                const float4 contribution =
                    reinterpret_cast<const float4 *>(
                        canonical_publication + route_row)[index];
                routed.x = rocm_native_quantised_accumulate_rn(
                    routed.x, contribution.x);
                routed.y = rocm_native_quantised_accumulate_rn(
                    routed.y, contribution.y);
                routed.z = rocm_native_quantised_accumulate_rn(
                    routed.z, contribution.z);
                routed.w = rocm_native_quantised_accumulate_rn(
                    routed.w, contribution.w);
            }

            float4 shared = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll 1
            for (int participant = 0;
                 participant < participant_count;
                 ++participant)
            {
                const size_t bank_row =
                    static_cast<size_t>(participant) * row_elements +
                    row_offset;
                const float4 contribution =
                    reinterpret_cast<const float4 *>(
                        rank_banks + bank_row)[index];
                shared.x = rocm_native_quantised_accumulate_rn(
                    shared.x, contribution.x);
                shared.y = rocm_native_quantised_accumulate_rn(
                    shared.y, contribution.y);
                shared.z = rocm_native_quantised_accumulate_rn(
                    shared.z, contribution.z);
                shared.w = rocm_native_quantised_accumulate_rn(
                    shared.w, contribution.w);
            }

            const float4 gated = make_float4(
                __fmul_rn(gate, shared.x),
                __fmul_rn(gate, shared.y),
                __fmul_rn(gate, shared.z),
                __fmul_rn(gate, shared.w));
            routed4[index] = routed;
            shared4[index] = gated;
            combined4[index] = make_float4(
                __fadd_rn(routed.x, gated.x),
                __fadd_rn(routed.y, gated.y),
                __fadd_rn(routed.z, gated.z),
                __fadd_rn(routed.w, gated.w));
        }
        return;
    }

    for (int column = first_column + threadIdx.x;
         column < last_column;
         column += blockDim.x)
    {
        float routed = 0.0f;
#pragma unroll 1
        for (int route = 0; route < top_k; ++route)
        {
            const size_t route_index =
                (static_cast<size_t>(token) *
                     static_cast<size_t>(top_k) +
                 static_cast<size_t>(route)) *
                    static_cast<size_t>(d_model) +
                static_cast<size_t>(column);
            routed = rocm_native_quantised_accumulate_rn(
                routed, canonical_publication[route_index]);
        }
        float shared = 0.0f;
#pragma unroll 1
        for (int participant = 0;
             participant < participant_count;
             ++participant)
        {
            const size_t shared_index =
                static_cast<size_t>(participant) * row_elements +
                row_offset + static_cast<size_t>(column);
            shared = rocm_native_quantised_accumulate_rn(
                shared, rank_banks[shared_index]);
        }
        const size_t output_index =
            row_offset + static_cast<size_t>(column);
        const float gated = __fmul_rn(gate, shared);
        routed_output[output_index] = routed;
        shared_output[output_index] = gated;
        combined_output[output_index] = __fadd_rn(routed, gated);
    }
}

// =========================================================================
// Kernel 6: SwiGLU — silu(gate) * up
//
// gate[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]
// Vectorized with float4.
// =========================================================================

__global__ void rocm_moe_swiglu_kernel(
    float *__restrict__ gate,
    const float *__restrict__ up,
    int count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const float g = gate[idx];
    gate[idx] =
        llaminar2::rocm::decode_equivalent::swigluValue(g, up[idx]);
}

// =========================================================================
// Kernel 6b: Decode-equivalent weighted route accumulation
//
// The independently rounded multiply and add are shared with grouped route
// publication. This prevents compiler contraction from making serial M=1 and
// grouped verifier rows observe different FP32 expression trees.
// =========================================================================

__global__ void rocm_moe_weighted_add_kernel(
    float *__restrict__ output,
    const float *__restrict__ input,
    float weight,
    int count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const float weighted_route = rocm_native_quantised_weight_route_rn(
        weight, input[idx]);
    output[idx] = rocm_native_quantised_accumulate_rn(
        output[idx], weighted_route);
}

// =========================================================================
// Extern "C" bridge functions — called from ROCmMoEKernel.cpp
// =========================================================================

extern "C"
{

bool hipMoE_publish_current_batch_llep_evidence_to_generation_control(
    const void *runtime_layers,
    int layer_count,
    int *generation_control,
    int generation_control_stride,
    int request_count,
    int movement_layer_count_index,
    int non_owner_assignment_layer_count_index,
    int device_idx,
    void *stream)
{
    if (!runtime_layers || layer_count <= 0 || !generation_control ||
        generation_control_stride <= 0 || request_count <= 0 ||
        movement_layer_count_index < 0 ||
        movement_layer_count_index >= generation_control_stride ||
        non_owner_assignment_layer_count_index < 0 ||
        non_owner_assignment_layer_count_index >= generation_control_stride ||
        movement_layer_count_index == non_owner_assignment_layer_count_index ||
        !stream)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();
    constexpr int kEvidenceThreads = 256;
    hipLaunchKernelGGL(
        publish_current_batch_llep_evidence_to_generation_control_kernel,
        dim3(1),
        dim3(kEvidenceThreads),
        0,
        static_cast<hipStream_t>(stream),
        static_cast<const DeviceMoELayerRuntimeView *>(runtime_layers),
        layer_count,
        generation_control,
        generation_control_stride,
        request_count,
        movement_layer_count_index,
        non_owner_assignment_layer_count_index);
    const hipError_t error = hipGetLastError();
    if (error != hipSuccess)
    {
        fprintf(
            stderr,
            "HIP current-batch LLEP evidence publication failed: %s\n",
            hipGetErrorString(error));
        return false;
    }
    return true;
}

bool hipMoE_gate_logits_single_token(
    const float *hidden, const float *gate_weights, float *logits,
    int d_model, int num_experts,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if ((d_model & 3) == 0 && d_model <= 8192 &&
        (reinterpret_cast<std::uintptr_t>(hidden) & 15u) == 0u &&
        (reinterpret_cast<std::uintptr_t>(gate_weights) & 15u) == 0u)
    {
        const int blocks = (num_experts + 7) / 8;
        const size_t shmem_bytes = static_cast<size_t>(d_model) * sizeof(float);
        hipLaunchKernelGGL(rocm_moe_gate_logits_single_token_fp32_lds_kernel,
            dim3(blocks), dim3(512), shmem_bytes, static_cast<hipStream_t>(stream),
            hidden, gate_weights, logits, d_model, num_experts);
    }
    else
    {
        const int threads = 128;
        hipLaunchKernelGGL(rocm_moe_gate_logits_single_token_kernel,
            dim3(num_experts), dim3(threads), 0, static_cast<hipStream_t>(stream),
            hidden, gate_weights, logits, d_model, num_experts);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE gate_logits_single_token kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_gate_logits_single_token_kpart(
    const float *hidden, const float *gate_weights, float *logits,
    float *partials,
    int d_model, int num_experts, int k_partitions,
    int device_idx, void *stream)
{
    if (!hidden || !gate_weights || !logits || !partials ||
        d_model <= 0 || num_experts <= 0 || k_partitions <= 0)
    {
        fprintf(stderr, "HIP MoE K-part gate logits invalid args\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const bool use_vec4 = ((d_model & 3) == 0) &&
                          ((reinterpret_cast<uintptr_t>(hidden) & 0xfu) == 0) &&
                          ((reinterpret_cast<uintptr_t>(gate_weights) & 0xfu) == 0);
    const int partial_threads = 128;
    hipLaunchKernelGGL(rocm_moe_gate_logits_single_token_kpart_kernel,
        dim3(num_experts, k_partitions), dim3(partial_threads), 0, static_cast<hipStream_t>(stream),
        hidden, gate_weights, partials, d_model, num_experts, k_partitions, use_vec4);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE gate_logits_single_token_kpart kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    const int reduce_threads = 64;
    hipLaunchKernelGGL(rocm_moe_gate_logits_kpart_reduce_kernel,
        dim3(num_experts), dim3(reduce_threads), 0, static_cast<hipStream_t>(stream),
        partials, logits, num_experts, k_partitions);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE gate_logits_kpart_reduce kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_gate_logits_single_token_kpart_partials(
    const float *hidden, const float *gate_weights, float *partials,
    int d_model, int num_experts, int k_partitions,
    int device_idx, void *stream)
{
    if (!hidden || !gate_weights || !partials ||
        d_model <= 0 || num_experts <= 0 || k_partitions <= 0)
    {
        fprintf(stderr, "HIP MoE K-part gate logits partials invalid args\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const bool use_vec4 = ((d_model & 3) == 0) &&
                          ((reinterpret_cast<uintptr_t>(hidden) & 0xfu) == 0) &&
                          ((reinterpret_cast<uintptr_t>(gate_weights) & 0xfu) == 0);
    const int partial_threads = 128;
    hipLaunchKernelGGL(rocm_moe_gate_logits_single_token_kpart_kernel,
        dim3(num_experts, k_partitions), dim3(partial_threads), 0, static_cast<hipStream_t>(stream),
        hidden, gate_weights, partials, d_model, num_experts, k_partitions, use_vec4);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE gate_logits_single_token_kpart_partials kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_gate_logits_single_token_fp16_weights(
    const float *hidden, const void *gate_weights_fp16, float *logits,
    int d_model, int num_experts,
    int device_idx, void *stream)
{
    if (!hidden || !gate_weights_fp16 || !logits || d_model <= 0 || num_experts <= 0 || !stream)
    {
        fprintf(stderr, "HIP MoE fp16 gate logits invalid args\n");
        return false;
    }

    if (static_cast<hipError_t>(llaminar2::HipDeviceGuard::setDevice(device_idx)) != hipSuccess)
        return false;
    (void)hipGetLastError();

    if ((d_model & 1) == 0 && num_experts >= 2)
    {
        const int blocks = (num_experts + 1) / 2;
        hipLaunchKernelGGL(rocm_moe_gate_logits_single_token_fp16_weights_lds_kernel,
            dim3(blocks), dim3(128), 0, static_cast<hipStream_t>(stream),
            hidden, static_cast<const __half *>(gate_weights_fp16), logits, d_model, num_experts);
    }
    else
    {
        const int threads = 128;
        hipLaunchKernelGGL(rocm_moe_gate_logits_single_token_fp16_weights_kernel,
            dim3(num_experts), dim3(threads), 0, static_cast<hipStream_t>(stream),
            hidden, static_cast<const __half *>(gate_weights_fp16), logits, d_model, num_experts);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE gate_logits_single_token_fp16_weights kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    llaminar2::PerfStatsCollector::addCounter("kernel", "rocm_moe_router_fp16_operand_calls", 1.0,
        "router", "rocm:" + std::to_string(device_idx),
        {{"m", "1"}, {"d_model", std::to_string(d_model)}, {"num_experts", std::to_string(num_experts)},
         {"activation", "fp16_rne"}, {"weight", "fp16"}, {"accumulation", "fp32"}, {"output", "fp32"}, {"route", "serial_fixed"}});
    return true;
}

bool hipMoE_gate_logits_single_token_bf16_weights(
    const float *hidden, const void *gate_weights_bf16, float *logits,
    int d_model, int num_experts,
    int device_idx, void *stream)
{
    if (!hidden || !gate_weights_bf16 || !logits || d_model <= 0 || num_experts <= 0)
    {
        fprintf(stderr, "HIP MoE bf16 gate logits invalid args\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 128;
    hipLaunchKernelGGL(rocm_moe_gate_logits_single_token_bf16_weights_kernel,
        dim3(num_experts), dim3(threads), 0, static_cast<hipStream_t>(stream),
        hidden, static_cast<const hip_bfloat16 *>(gate_weights_bf16), logits, d_model, num_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE gate_logits_single_token_bf16_weights kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_gate_logits_fp32_decode_equivalent_rows(
    const float *hidden, const float *gate_weights, float *logits,
    int seq_len, int d_model, int num_experts,
    int device_idx, void *stream,
    const int *device_effective_seq_len)
{
    if (!hidden || !gate_weights || !logits || !stream ||
        seq_len <= 0 || d_model <= 0 || num_experts <= 0)
    {
        fprintf(stderr, "HIP MoE FP32 batch-invariant grouped gate logits invalid args\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    constexpr int rows_per_block = 16;
    const int threads = 128;
    hipLaunchKernelGGL(rocm_moe_gate_logits_fp32_grouped_verifier_kernel,
        dim3(num_experts, (seq_len + rows_per_block - 1) / rows_per_block),
        dim3(threads), 0, static_cast<hipStream_t>(stream),
        hidden, gate_weights, logits, seq_len, d_model, num_experts,
        device_effective_seq_len);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE FP32 grouped verifier gate logits kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_gate_logits_fp16_decode_equivalent_rows_with_policy(
    const float *hidden, const void *gate_weights_fp16, float *logits,
    int seq_len, int d_model, int num_experts,
    int device_idx, void *stream,
    const int *device_effective_seq_len, bool qualified_default)
{
    if (!hidden || !gate_weights_fp16 || !logits || !stream ||
        seq_len <= 0 || d_model <= 0 || num_experts <= 0)
    {
        fprintf(stderr, "HIP MoE FP16 batch-invariant grouped gate logits invalid args\n");
        return false;
    }

    if (static_cast<hipError_t>(llaminar2::HipDeviceGuard::setDevice(device_idx)) != hipSuccess)
        return false;
    (void)hipGetLastError();

    bool fixed_rows = false;
    if (device_effective_seq_len && seq_len >= 2 && seq_len <= 8 &&
        d_model == 2560 && num_experts == 512 && llaminar2::qfnGroupedFP16RouterFixedRowsOptIn(qualified_default))
    {
        const auto hardware = llaminar2::rocm::nativeQuantisedHardwareForDevice(device_idx);
        fixed_rows = llaminar2::native_quantised::selectVerifierDispatchTable(hardware) ==
            llaminar2::native_quantised::VerifierDispatchTable::MI50;
    }
    if (fixed_rows)
    {
        if (reinterpret_cast<std::uintptr_t>(stream) <= 2 ||
            (reinterpret_cast<std::uintptr_t>(gate_weights_fp16) & 3u))
            return false;
        const auto native = static_cast<hipStream_t>(stream);
        const auto *gate = static_cast<const __half *>(gate_weights_fp16);
        // Size the register accumulator array for the admitted row capacity.
        // The device count still bounds every load and store in the final tile.
        if (seq_len <= 4)
            hipLaunchKernelGGL(qfn_grouped_fp16_router_fixed_rows<4>, dim3(512), dim3(128), 0,
                native, hidden, gate, logits, device_effective_seq_len, seq_len);
        else
            hipLaunchKernelGGL(qfn_grouped_fp16_router_fixed_rows<8>, dim3(512), dim3(128), 0,
                native, hidden, gate, logits, device_effective_seq_len, seq_len);
    }
    else
        launch_fp16_router_rows<false>(hidden, static_cast<const __half *>(gate_weights_fp16), logits,
            seq_len, seq_len, d_model, num_experts, device_effective_seq_len, {}, static_cast<hipStream_t>(stream));

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE FP16 grouped verifier gate logits kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    if (fixed_rows)
        llaminar2::PerfStatsCollector::addCounter("kernel", "qfn_grouped_fp16_router_fixed_rows_nodes", 1.0,
            "router", "rocm:" + std::to_string(device_idx), {{"physical_rows", std::to_string(seq_len)}});
    llaminar2::PerfStatsCollector::addCounter("kernel", "rocm_moe_router_fp16_operand_calls", 1.0,
        "router", "rocm:" + std::to_string(device_idx),
        {{"m", std::to_string(seq_len)}, {"d_model", std::to_string(d_model)}, {"num_experts", std::to_string(num_experts)},
         {"activation", "fp16_rne"}, {"weight", "fp16"}, {"accumulation", "fp32"}, {"output", "fp32"}, {"route", "grouped_fixed"}});
    return true;
}

bool hipMoE_gate_logits_fp16_decode_equivalent_rows(
    const float *hidden, const void *gate_weights_fp16, float *logits,
    int seq_len, int d_model, int num_experts,
    int device_idx, void *stream,
    const int *device_effective_seq_len)
{
    return hipMoE_gate_logits_fp16_decode_equivalent_rows_with_policy(hidden, gate_weights_fp16, logits, seq_len, d_model, num_experts, device_idx, stream, device_effective_seq_len, false);
}

bool hipMoE_gate_logits_bf16_decode_equivalent_rows(
    const float *hidden, const void *gate_weights_bf16, float *logits,
    int seq_len, int d_model, int num_experts,
    int device_idx, void *stream,
    const int *device_effective_seq_len)
{
    if (!hidden || !gate_weights_bf16 || !logits || !stream ||
        seq_len <= 0 || d_model <= 0 || num_experts <= 0)
    {
        fprintf(stderr, "HIP MoE BF16 batch-invariant grouped gate logits invalid args\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    constexpr int rows_per_block = 16;
    const int threads = 128;
    hipLaunchKernelGGL(rocm_moe_gate_logits_bf16_grouped_verifier_kernel,
        dim3(num_experts, (seq_len + rows_per_block - 1) / rows_per_block),
        dim3(threads), 0, static_cast<hipStream_t>(stream),
        hidden, static_cast<const hip_bfloat16 *>(gate_weights_bf16), logits,
        seq_len, d_model, num_experts, device_effective_seq_len);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE BF16 grouped verifier gate logits kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_quantize_router_gate_q8(
    const float *gate_weights, int8_t *gate_weights_q8, float *gate_scales,
    int d_model, int num_experts,
    int device_idx, void *stream, llaminar2::RouterQ8Basis basis)
{
    if (!gate_weights || !gate_weights_q8 || !gate_scales ||
        !stream || (basis != llaminar2::RouterQ8Basis::Original &&
            basis != llaminar2::RouterQ8Basis::SignedHadamard32) ||
            d_model <= 0 || num_experts <= 0 || (d_model % 32) != 0)
    {
        fprintf(stderr, "HIP MoE Q8 router gate quantize invalid args\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

        if (basis == llaminar2::RouterQ8Basis::SignedHadamard32)
            return llaminar2::router_hadamard_q8::launch(
                gate_weights, gate_weights_q8, gate_scales, num_experts, d_model, stream);
    const int blocks_per_row = d_model / 32;
    hipLaunchKernelGGL(rocm_moe_router_gate_quantize_q8_kernel,
        dim3(num_experts, blocks_per_row), dim3(32), 0, static_cast<hipStream_t>(stream),
        gate_weights, gate_weights_q8, gate_scales, d_model, num_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE Q8 router gate quantize kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_gate_logits_single_token_q8_weights(
    const float *hidden, int8_t *hidden_q8, float *hidden_scales,
    const int8_t *gate_weights_q8, const float *gate_scales, float *logits,
    int d_model, int num_experts,
    int device_idx, void *stream, llaminar2::RouterQ8Basis basis)
{
    if (!hidden || !hidden_q8 || !hidden_scales || !gate_weights_q8 || !gate_scales || !logits ||
        !stream || (basis != llaminar2::RouterQ8Basis::Original &&
            basis != llaminar2::RouterQ8Basis::SignedHadamard32) ||
            d_model <= 0 || num_experts <= 0 || (d_model % 32) != 0)
    {
        fprintf(stderr, "HIP MoE Q8 gate logits invalid args\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    (void)hipGetLastError();

    const int blocks_per_row = d_model / 32;
    const bool is_wave64 =
        llaminar2::rocmHardwareMetadata().at(device_idx).native_wave_width == 64;
    const bool use_fused_wave64 =
        blocks_per_row <= 128 &&
        ((reinterpret_cast<std::uintptr_t>(gate_weights_q8) & 15u) == 0u) &&
        is_wave64;
    if (use_fused_wave64)
    {
        const int blocks = (num_experts + 7) / 8;
        if (basis == llaminar2::RouterQ8Basis::SignedHadamard32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(rocm_moe_gate_logits_single_token_q8_fused_int4_kernel<true>),
                dim3(blocks), dim3(512), 0, hip_stream,
                hidden, hidden_q8, hidden_scales, gate_weights_q8, gate_scales, logits,
                d_model, num_experts);
        }
        else
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(rocm_moe_gate_logits_single_token_q8_fused_int4_kernel<false>),
                dim3(blocks), dim3(512), 0, hip_stream,
                hidden, hidden_q8, hidden_scales, gate_weights_q8, gate_scales, logits,
                d_model, num_experts);
        }
        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "HIP MoE fused Q8 gate logits kernel launch failed: %s\n", hipGetErrorString(err));
            return false;
        }
        return true;
    }

    if (basis == llaminar2::RouterQ8Basis::SignedHadamard32)
        {
            if (!llaminar2::router_hadamard_q8::launch(
                    hidden, hidden_q8, hidden_scales, 1, d_model, stream)) return false;
        }
        else
        {
            hipLaunchKernelGGL(rocm_moe_router_hidden_quantize_q8_kernel,
        dim3(blocks_per_row), dim3(32), 0, hip_stream,
        hidden, hidden_q8, hidden_scales, d_model);
        }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE Q8 hidden quantize kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    if (blocks_per_row <= 128 && is_wave64)
    {
        const int blocks = (num_experts + 7) / 8;
        hipLaunchKernelGGL(rocm_moe_gate_logits_single_token_q8_coalesced_lds_kernel,
            dim3(blocks), dim3(512), 0, hip_stream,
            hidden_q8, hidden_scales, gate_weights_q8, gate_scales, logits,
            d_model, num_experts);
    }
    else
    {
        const int threads = 128;
        hipLaunchKernelGGL(rocm_moe_gate_logits_single_token_q8_weights_kernel,
            dim3(num_experts), dim3(threads), 0, hip_stream,
            hidden_q8, hidden_scales, gate_weights_q8, gate_scales, logits,
            d_model, num_experts);
    }

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE Q8 gate logits kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    // Reuse the bank after the final router reader, retaining original expert input.
    if (basis == llaminar2::RouterQ8Basis::SignedHadamard32)
    {
        hipLaunchKernelGGL(rocm_moe_router_hidden_quantize_q8_kernel,
        dim3(blocks_per_row), dim3(32), 0, hip_stream,
        hidden, hidden_q8, hidden_scales, d_model);
        if (hipGetLastError() != hipSuccess) return false;
    }
    return true;
}

/** @copydoc hipMoE_publish_original_hidden_q8_with_k32_sums */
bool hipMoE_publish_original_hidden_q8_with_k32_sums(const float* hidden,std::int8_t* codes,
    float* scales,std::int16_t* sums,int device,void* stream,const int* live_rows)
{
    if (!hidden || !codes || !scales || !sums || !live_rows ||
        reinterpret_cast<std::uintptr_t>(stream)<=2u || device<0 ||
        static_cast<hipError_t>(llaminar2::HipDeviceGuard::setDevice(device))!=hipSuccess)
        return false;
    hipLaunchKernelGGL(rocm_moe_router_hidden_quantize_q8_rows_and_sums_kernel,
        dim3(80,4),dim3(32),0,static_cast<hipStream_t>(stream),hidden,codes,scales,live_rows,sums);
    return hipGetLastError()==hipSuccess;
}

/** @copydoc hipMoE_publish_original_hidden_q8_rows */
bool hipMoE_publish_original_hidden_q8_rows(
    const float *hidden, int8_t *codes, float *scales,
    int rows, int width, int device, void *stream, const int *live_rows)
{
    if (!hidden || !codes || !scales ||
        reinterpret_cast<std::uintptr_t>(stream) <= 2u || rows <= 0 ||
        width <= 0 || width % 32 != 0 || device < 0 ||
        static_cast<hipError_t>(llaminar2::HipDeviceGuard::setDevice(device)) != hipSuccess)
        return false;
    const int threads = (rows > 16 || (width / 32) >= 8) ? 256 : 32;
    const int groups = threads / 32;
    hipLaunchKernelGGL(rocm_moe_router_hidden_quantize_q8_rows_kernel,
        dim3((width / 32 + groups - 1) / groups, rows), dim3(threads), 0,
        static_cast<hipStream_t>(stream), hidden, codes, scales, rows, width, live_rows);
    return hipGetLastError() == hipSuccess;
}

/** @brief Record a successfully enqueued projection, including capture recording.
 * @param selection Exact resolved engine and original geometry used by this submission.
 *
 * Retained graph replays do not execute this host function. This counter proves
 * submitted/recorded topology, not live arithmetic or the number of GPU replays.
 */
static void record_rocm_router_q8_projection_submission(
    const llaminar2::rocm::RouterQ8PrefillSelection &selection)
{
    using namespace llaminar2;
    const char *name = selection.usesWmma() ? "rocm_router_q8_wmma_projection_submissions" :
        "rocm_router_q8_dot4_projection_submissions";
    if (!PerfStatsCollector::isRecordFamilyEnabled("kernel",name)) return;
    const auto &shape = selection.shape();
    PerfStatsCollector::addCounter("kernel",name,1.0,"router","rocm:" + std::to_string(selection.device()),
        {{"capacity",std::to_string(shape.capacity)}, {"width",std::to_string(shape.width)},
         {"experts",std::to_string(shape.experts)},
         {"basis",shape.basis == RouterQ8Basis::Original ? "original" : "signed_hadamard32"},
         {"ownership",shape.ownership == rocm::RouterQ8RowOwnership::WholeRows ? "whole_rows" : "compact_owned"},
         {"k_waves",std::to_string(selection.usesWmma() ? rocm::routerQ8WaveCount(selection.nativeTile()) : 0)}});
}

/**
 * @brief Grouped Q8 router logits for batch-invariant prefill rows.
 *
 * This is intentionally decode-equivalent rather than a generic prefill GEMM:
 * each hidden row is quantized independently with serial decode's block scales,
 * then one expert-owned block reuses gate Q8 data while accumulating all rows.
 * The 64-thread geometry is admitted only when the second half of the serial
 * K partition is empty. Wider routers retain all 128 owners; merely reducing
 * their thread count would change K accumulation order and MTP token choices.
 *
 * @param hidden FP32 hidden rows owned by the caller's retained graph.
 * @param hidden_q8 Persistent byte workspace for independent Q8 row blocks.
 * @param hidden_scales Persistent FP32 workspace for half-rounded Q8 scales.
 * @param gate_weights_q8 Immutable prepared router bytes, expert-major.
 * @param gate_scales Prepared router scales, one per 32-column block.
 * @param logits Output rows; inactive rows are left untouched.
 * @param seq_len Admitted physical row capacity embedded in graph identity.
 * @param d_model Positive hidden width divisible by 32.
 * @param num_experts Positive router output width.
 * @param device_idx Device owning every binding and the exact launch stream.
 * @param stream Explicit non-null producer stream.
 * @param device_effective_seq_len Optional device-owned live count.
 * @return Whether both production launches were accepted by HIP.
 */
bool hipMoE_gate_logits_q8_weights_decode_equivalent_rows(
    const float *hidden, int8_t *hidden_q8, float *hidden_scales,
    const int8_t *gate_weights_q8, const float *gate_scales, float *logits,
    int seq_len, int d_model, int num_experts,
    int device_idx, void *stream,
    const int *device_effective_seq_len, llaminar2::RouterQ8Basis basis)
{
    if (!hidden || !hidden_q8 || !hidden_scales || !gate_weights_q8 || !gate_scales || !logits ||
        !stream || stream == hipStreamLegacy || stream == hipStreamPerThread || seq_len <= 0 ||
        (basis != llaminar2::RouterQ8Basis::Original &&
            basis != llaminar2::RouterQ8Basis::SignedHadamard32) ||
        d_model <= 0 || num_experts <= 0 || (d_model % 32) != 0)
    {
        fprintf(stderr, "HIP MoE Q8 batch-invariant grouped gate logits invalid args\n");
        return false;
    }

    if (llaminar2::HipDeviceGuard::setDevice(device_idx) != hipSuccess) return false;
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    int stream_device = -1;
    if (hipStreamGetDevice(hip_stream, &stream_device) != hipSuccess || stream_device != device_idx)
        return false;
    (void)hipGetLastError();

    const auto selection = llaminar2::rocm::selectRouterQ8Prefill(device_idx,
        {seq_len,d_model,num_experts,basis,llaminar2::rocm::RouterQ8RowOwnership::WholeRows},
        device_effective_seq_len ? llaminar2::DeviceRowRange::deviceCounted(seq_len,device_effective_seq_len) :
            llaminar2::DeviceRowRange::fullyActive(seq_len));
    const int blocks_per_row = d_model / 32;
    // Pack independent scales only when there is prefill-sized work. This is
    // launch geometry, not a different quantizer or a changed router K tree.
    const int quantize_threads = seq_len > 16 ? 256 : 32;
    const int quantize_groups = quantize_threads / 32;
    if (basis == llaminar2::RouterQ8Basis::SignedHadamard32)
        {
            if (!llaminar2::router_hadamard_q8::launch(
                    hidden, hidden_q8, hidden_scales, seq_len, d_model, stream, device_effective_seq_len)) return false;
        }
        else
        {
            hipLaunchKernelGGL(rocm_moe_router_hidden_quantize_q8_rows_kernel,
        dim3((blocks_per_row + quantize_groups - 1) / quantize_groups, seq_len),
        dim3(quantize_threads), 0, hip_stream,
        hidden, hidden_q8, hidden_scales, seq_len, d_model,
        device_effective_seq_len);
        }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE Q8 grouped hidden quantize kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    constexpr int rows_per_block = 16;
    const dim3 grid(num_experts, (seq_len + rows_per_block - 1) / rows_per_block);
    // The choice uses immutable geometry, never a live-count readback. At
    // K=3072 the second serial wave owns 32 blocks and must remain present.
    if (selection.usesWmma())
    {
        if (!llaminar2::rocm::routerQ8Prefill({
                .hidden_codes = hidden_q8, .hidden_scales = hidden_scales,
                .gate_codes = gate_weights_q8, .gate_scales = gate_scales, .logits = logits,
                .capacity = seq_len, .width = d_model, .experts = num_experts,
                .device = device_idx, .live_rows = device_effective_seq_len, .basis = basis}, selection, stream)) return false;
    }
    else if (blocks_per_row <= 64)
        hipLaunchKernelGGL((rocm_moe_gate_logits_q8_grouped_verifier_kernel<64>),
            grid, dim3(64), 0, hip_stream,
            hidden_q8, hidden_scales, gate_weights_q8, gate_scales, logits,
            seq_len, d_model, num_experts, device_effective_seq_len);
    else
        hipLaunchKernelGGL((rocm_moe_gate_logits_q8_grouped_verifier_kernel<128>),
            grid, dim3(128), 0, hip_stream,
            hidden_q8, hidden_scales, gate_weights_q8, gate_scales, logits,
            seq_len, d_model, num_experts, device_effective_seq_len);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE Q8 grouped gate logits kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    record_rocm_router_q8_projection_submission(selection);
    // Reuse the bank after the final router reader, retaining original expert input.
    if (basis == llaminar2::RouterQ8Basis::SignedHadamard32)
    {
        hipLaunchKernelGGL(rocm_moe_router_hidden_quantize_q8_rows_kernel,
        dim3((blocks_per_row + quantize_groups - 1) / quantize_groups, seq_len),
        dim3(quantize_threads), 0, hip_stream,
        hidden, hidden_q8, hidden_scales, seq_len, d_model,
        device_effective_seq_len);
        if (hipGetLastError() != hipSuccess) return false;
    }
    return true;
}

bool hipMoE_gate_logits_single_token_grouped(
    const float *hidden, const float *gate_weights, float *logits,
    int d_model, int num_experts, size_t shared_mem_bytes,
    int device_idx, void *stream)
{
    if (!hidden || !gate_weights || !logits || d_model <= 0 || num_experts <= 0)
    {
        fprintf(stderr, "HIP MoE grouped gate logits invalid args\n");
        return false;
    }

    const size_t required_shared = static_cast<size_t>(d_model) * sizeof(float);
    if (shared_mem_bytes < required_shared)
    {
        fprintf(stderr, "HIP MoE grouped gate logits shared memory too small: requested=%zu required=%zu\n",
                shared_mem_bytes, required_shared);
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const bool use_vec4 = ((d_model & 3) == 0) &&
                          ((reinterpret_cast<uintptr_t>(hidden) & 0xfu) == 0) &&
                          ((reinterpret_cast<uintptr_t>(gate_weights) & 0xfu) == 0);
    const int blocks = (num_experts + kGateLogitsExpertsPerBlock - 1) / kGateLogitsExpertsPerBlock;
    hipLaunchKernelGGL(rocm_moe_gate_logits_grouped_decode_router_rows_kernel,
        dim3(blocks), dim3(kGateLogitsGroupedThreads), shared_mem_bytes, static_cast<hipStream_t>(stream),
        hidden, gate_weights, logits, d_model, num_experts, use_vec4,
        /*seq_len=*/1, /*device_effective_seq_len=*/nullptr);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE grouped gate_logits_single_token kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

/**
 * @brief Launch the selected shared-memory decode router for every grouped row.
 *
 * `grid.y` adds independent rows without changing the block shape, shared
 * hidden staging, lane-to-K mapping, or wave reduction used by M=1 decode.
 * Consequently each grouped row produces the exact serial-decode logit bytes
 * while all rows are submitted in one launch.
 */
bool hipMoE_gate_logits_grouped_decode_router_rows(
    const float *hidden,
    const float *gate_weights,
    float *logits,
    int seq_len,
    int d_model,
    int num_experts,
    size_t shared_mem_bytes,
    int device_idx,
    void *stream,
    const int *device_effective_seq_len)
{
    if (!hidden || !gate_weights || !logits || !stream ||
        seq_len <= 0 || d_model <= 0 || num_experts <= 1)
    {
        fprintf(stderr, "HIP MoE grouped decode-router rows invalid args\n");
        return false;
    }

    const size_t required_shared =
        static_cast<size_t>(d_model) * sizeof(float);
    if (shared_mem_bytes < required_shared)
    {
        fprintf(stderr,
                "HIP MoE grouped decode-router rows shared memory too small: requested=%zu required=%zu\n",
                shared_mem_bytes,
                required_shared);
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const bool use_vec4 =
        ((d_model & 3) == 0) &&
        ((reinterpret_cast<uintptr_t>(hidden) & 0xfu) == 0) &&
        ((reinterpret_cast<uintptr_t>(gate_weights) & 0xfu) == 0);
    const int expert_blocks =
        (num_experts + kGateLogitsExpertsPerBlock - 1) /
        kGateLogitsExpertsPerBlock;
    hipLaunchKernelGGL(
        rocm_moe_gate_logits_grouped_decode_router_rows_kernel,
        dim3(expert_blocks, seq_len),
        dim3(kGateLogitsGroupedThreads),
        shared_mem_bytes,
        static_cast<hipStream_t>(stream),
        hidden,
        gate_weights,
        logits,
        d_model,
        num_experts,
        use_vec4,
        seq_len,
        device_effective_seq_len);

    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr,
                "HIP MoE grouped decode-router rows launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_softmax_topk_decode_runtime(
    float *logits,
    void *runtime,
    float *legacy_indices,
    float *legacy_weights,
    int num_experts, int top_k,
    bool normalize_weights,
    bool write_legacy_outputs,
    bool update_runtime_histogram,
    llaminar2::RoutedExpertRowExecutionPolicy row_execution_policy,
    void *runtime_layers,
    const void *rebalance_plan_entries,
    uint32_t rebalance_plan_capacity,
    void *rebalance_command_header,
    const void *rebalance_local_transfer_slots,
    uint32_t rebalance_local_transfer_slot_count,
    const void *rebalance_config,
    void *rebalance_apply_status,
    void *rebalance_controller_state,
    int rebalance_target_layer,
    uint32_t rebalance_command_buffer_count,
    const int32_t *absolute_position_ids,
    int device_idx, void *stream,
        const int32_t *active_row_count_device)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!logits || !runtime || num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK)
    {
        return false;
    }
    if (write_legacy_outputs && (!legacy_indices || !legacy_weights))
    {
        return false;
    }

    const int threads = 256;
    DeviceMoERebalanceConfigView rebalance_cfg{};
    if (rebalance_config)
        rebalance_cfg = *static_cast<const DeviceMoERebalanceConfigView *>(rebalance_config);
    hipLaunchKernelGGL(rocm_moe_softmax_topk_decode_runtime_kernel,
        dim3(1), dim3(threads), 0, static_cast<hipStream_t>(stream),
        logits,
        static_cast<DeviceMoELayerRuntimeView *>(runtime),
        static_cast<DeviceMoELayerRuntimeView *>(runtime_layers),
        legacy_indices, legacy_weights,
        num_experts, top_k,
        normalize_weights,
        write_legacy_outputs,
        update_runtime_histogram,
        row_execution_policy,
        absolute_position_ids,
        static_cast<const DeviceMoERebalancePlanEntryView *>(rebalance_plan_entries),
        rebalance_plan_capacity,
        static_cast<DeviceMoERebalanceCommandBufferHeaderView *>(rebalance_command_header),
        static_cast<const DeviceMoEExpertDirectoryEntryView *>(rebalance_local_transfer_slots),
        rebalance_local_transfer_slot_count,
        rebalance_cfg,
        static_cast<DeviceMoERebalanceApplyStatusView *>(rebalance_apply_status),
        static_cast<DeviceMoERebalanceGraphControllerStateView *>(rebalance_controller_state),
        rebalance_target_layer,
        rebalance_command_buffer_count, active_row_count_device);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE softmax_topk_decode_runtime kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_softmax_topk_decode_runtime_wave64(
    float *logits,
    void *runtime,
    float *legacy_indices,
    float *legacy_weights,
    int num_experts, int top_k,
    bool normalize_weights,
    bool write_legacy_outputs,
    bool update_runtime_histogram,
    llaminar2::RoutedExpertRowExecutionPolicy row_execution_policy,
    void *runtime_layers,
    const void *rebalance_plan_entries,
    uint32_t rebalance_plan_capacity,
    void *rebalance_command_header,
    const void *rebalance_local_transfer_slots,
    uint32_t rebalance_local_transfer_slot_count,
    const void *rebalance_config,
    void *rebalance_apply_status,
    void *rebalance_controller_state,
    int rebalance_target_layer,
    uint32_t rebalance_command_buffer_count,
    const int32_t *absolute_position_ids,
    int device_idx, void *stream,
        const int32_t *active_row_count_device)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!logits || !runtime || num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK)
    {
        return false;
    }
    if (write_legacy_outputs && (!legacy_indices || !legacy_weights))
    {
        return false;
    }

    const bool apply_ready_rebalance = runtime_layers != nullptr;
    const bool complete_rebalance_binding =
        runtime_layers && rebalance_plan_entries &&
        rebalance_plan_capacity > 0u && rebalance_command_header &&
        rebalance_config && rebalance_apply_status &&
        rebalance_controller_state && rebalance_command_buffer_count > 0u;
    const bool any_rebalance_binding =
        runtime_layers || rebalance_plan_entries ||
        rebalance_plan_capacity > 0u || rebalance_command_header ||
        rebalance_local_transfer_slots ||
        rebalance_local_transfer_slot_count > 0u || rebalance_config ||
        rebalance_apply_status || rebalance_controller_state ||
        rebalance_target_layer >= 0;
    if (any_rebalance_binding && !complete_rebalance_binding)
        return false;

    // Fragment slots do not share replica load-balancing state. Reuse the
    // independent-slot kernel while retaining each slot's acquired-bank proof.
    // A ready-wave transaction still uses its explicitly fused apply kernel.
    if (row_execution_policy == llaminar2::RoutedExpertRowExecutionPolicy::IntermediateShardLocal &&
        !apply_ready_rebalance)
    {
        if (!stream || top_k > num_experts) return false;
#define LLAMINAR_LAUNCH_FRAGMENT_ROUTE(HISTOGRAM) \
        hipLaunchKernelGGL(HIP_KERNEL_NAME( \
            rocm_moe_softmax_topk_decode_independent_wave64_kernel<HISTOGRAM, \
                llaminar2::RoutedExpertRowExecutionPolicy::IntermediateShardLocal>), \
            dim3(1), dim3(256), 0, static_cast<hipStream_t>(stream), \
            logits, static_cast<DeviceMoELayerRuntimeView *>(runtime), \
            legacy_indices, legacy_weights, num_experts, top_k, \
            normalize_weights, write_legacy_outputs, active_row_count_device)
        if (update_runtime_histogram)
            LLAMINAR_LAUNCH_FRAGMENT_ROUTE(true);
        else
            LLAMINAR_LAUNCH_FRAGMENT_ROUTE(false);
#undef LLAMINAR_LAUNCH_FRAGMENT_ROUTE
        return hipGetLastError() == hipSuccess;
    }

    const int threads = 256;
    DeviceMoERebalanceConfigView rebalance_cfg{};
    if (rebalance_config)
        rebalance_cfg = *static_cast<const DeviceMoERebalanceConfigView *>(rebalance_config);

#define LLAMINAR_LAUNCH_ROCM_WAVE64_DECODE_ROUTE(HISTOGRAM, APPLY)              \
    hipLaunchKernelGGL(                                                         \
        HIP_KERNEL_NAME(                                                        \
            rocm_moe_softmax_topk_decode_runtime_wave64_kernel<                 \
                HISTOGRAM, APPLY>),                                             \
        dim3(1), dim3(threads), 0, static_cast<hipStream_t>(stream),             \
        logits,                                                                 \
        static_cast<DeviceMoELayerRuntimeView *>(runtime),                      \
        static_cast<DeviceMoELayerRuntimeView *>(runtime_layers),               \
        legacy_indices, legacy_weights,                                         \
        num_experts, top_k,                                                     \
        normalize_weights,                                                      \
        write_legacy_outputs,                                                   \
        row_execution_policy,                                            \
        absolute_position_ids,                                                  \
        static_cast<const DeviceMoERebalancePlanEntryView *>(                   \
            rebalance_plan_entries),                                            \
        rebalance_plan_capacity,                                                \
        static_cast<DeviceMoERebalanceCommandBufferHeaderView *>(               \
            rebalance_command_header),                                          \
        static_cast<const DeviceMoEExpertDirectoryEntryView *>(                 \
            rebalance_local_transfer_slots),                                    \
        rebalance_local_transfer_slot_count,                                    \
        rebalance_cfg,                                                          \
        static_cast<DeviceMoERebalanceApplyStatusView *>(                       \
            rebalance_apply_status),                                            \
        static_cast<DeviceMoERebalanceGraphControllerStateView *>(              \
            rebalance_controller_state),                                        \
        rebalance_target_layer,                                                 \
        rebalance_command_buffer_count, active_row_count_device)

    if (update_runtime_histogram)
    {
        if (apply_ready_rebalance)
            LLAMINAR_LAUNCH_ROCM_WAVE64_DECODE_ROUTE(true, true);
        else
            LLAMINAR_LAUNCH_ROCM_WAVE64_DECODE_ROUTE(true, false);
    }
    else if (apply_ready_rebalance)
    {
        LLAMINAR_LAUNCH_ROCM_WAVE64_DECODE_ROUTE(false, true);
    }
    else
    {
        LLAMINAR_LAUNCH_ROCM_WAVE64_DECODE_ROUTE(false, false);
    }
#undef LLAMINAR_LAUNCH_ROCM_WAVE64_DECODE_ROUTE

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE softmax_topk_decode_runtime_wave64 kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

    /**
     * @brief Publish the exact decode route and its descriptors in one retained kernel.
     * @param logits Original FP32 router logits, overwritten with probabilities.
     * @param runtime Original device runtime retaining the acquired placement bank.
     * @param legacy_indices Optional complete selected ID output.
     * @param legacy_weights Optional complete selected weight output.
     * @param num_experts Exact original source extent.
     * @param top_k Exact selected packet width.
     * @param normalize_weights Preserve the backend's original serial normalization.
     * @param write_legacy_outputs Whether both complete selected outputs are bound.
     * @param update_runtime_histogram Whether this graph owns demand evidence.
     * @param row_policy Original independent whole-owner or local-fragment contract.
     * @param publication Existing arena descriptor outputs with complete identity.
     * @param device_idx Original rank-local device ordinal.
     * @param stream Exact non-default producer and consumer stream.
     * @param active_row_count_device Optional device-owned live row count.
     * @return Whether validation and native submission succeeded.
     */
__attribute__((visibility("default"))) bool hipMoE_softmax_topk_decode_selected_descriptors(
        float *logits, void *runtime, float *legacy_indices, float *legacy_weights,
        int num_experts, int top_k, bool normalize_weights, bool write_legacy_outputs,
        bool update_runtime_histogram, llaminar2::RoutedExpertRowExecutionPolicy row_policy,
        llaminar2::MoEDecodeDescriptorPublication publication,
        int device_idx, void *stream, const int32_t *active_row_count_device)
{
    using Policy = llaminar2::RoutedExpertRowExecutionPolicy;
    using Projections = llaminar2::DeviceMoEProjectionSet;
    if (!logits || !runtime || reinterpret_cast<std::uintptr_t>(stream) <= 2u ||
        num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK || top_k > num_experts ||
        !publication.valid() || publication.expert_count != num_experts ||
        (write_legacy_outputs && (!legacy_indices || !legacy_weights)) ||
        (row_policy != Policy::ParticipantAssigned && row_policy != Policy::IntermediateShardLocal) ||
        (row_policy == Policy::IntermediateShardLocal) != (publication.projections == Projections::IntermediateShard))
        return false;
    llaminar2::HipDeviceGuard::setDevice(device_idx);
#define LLAMINAR_LAUNCH_ROUTE_DESCRIPTORS(HISTOGRAM, POLICY) \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(rocm_moe_softmax_topk_decode_descriptors_kernel<HISTOGRAM, POLICY>), \
        dim3(1), dim3(256), 0, static_cast<hipStream_t>(stream), \
        logits, static_cast<DeviceMoELayerRuntimeView *>(runtime), legacy_indices, legacy_weights, \
        num_experts, top_k, normalize_weights, write_legacy_outputs, active_row_count_device, publication)
    if (row_policy == Policy::IntermediateShardLocal)
    {
        if (update_runtime_histogram) LLAMINAR_LAUNCH_ROUTE_DESCRIPTORS(true, Policy::IntermediateShardLocal);
        else LLAMINAR_LAUNCH_ROUTE_DESCRIPTORS(false, Policy::IntermediateShardLocal);
    }
    else
    {
        if (update_runtime_histogram) LLAMINAR_LAUNCH_ROUTE_DESCRIPTORS(true, Policy::ParticipantAssigned);
        else LLAMINAR_LAUNCH_ROUTE_DESCRIPTORS(false, Policy::ParticipantAssigned);
    }
#undef LLAMINAR_LAUNCH_ROUTE_DESCRIPTORS
    return hipGetLastError() == hipSuccess;
}

/** @copydoc hipMoE_fp16_projection_route_selected_descriptors */
__attribute__((visibility("default"))) bool hipMoE_fp16_projection_route_selected_descriptors(
    const float *hidden, const void *gate_weights, float *logits, void *runtime,
    float *legacy_indices, float *legacy_weights, int d_model, int num_experts,
    int top_k, bool normalize_weights, bool write_legacy_outputs,
    bool update_runtime_histogram, llaminar2::RoutedExpertRowExecutionPolicy row_policy,
    llaminar2::MoEDecodeDescriptorPublication publication,
    llaminar2::NativeTileCompletionView completion,
    int device_idx, void *stream, const int32_t *active_row_count_device)
{
    using Policy = llaminar2::RoutedExpertRowExecutionPolicy;
    using Projections = llaminar2::DeviceMoEProjectionSet;
    if (!hidden || !gate_weights || !logits || !runtime || d_model <= 0 ||
        reinterpret_cast<std::uintptr_t>(stream) <= 2u || num_experts <= 0 ||
        num_experts > kDeviceMoEMaxExperts || top_k <= 0 ||
        top_k > kDeviceMoEMaxTopK || top_k > num_experts ||
        !publication.valid() || publication.expert_count != num_experts ||
        (write_legacy_outputs && (!legacy_indices || !legacy_weights)) ||
        (row_policy != Policy::ParticipantAssigned && row_policy != Policy::IntermediateShardLocal) ||
        (row_policy == Policy::IntermediateShardLocal) != (publication.projections == Projections::IntermediateShard) ||
        !completion.counters || reinterpret_cast<std::uintptr_t>(completion.counters) % alignof(std::uint32_t) ||
        completion.tiles != 1u || completion.routes != 1u ||
        completion.contributors != static_cast<std::uint32_t>((num_experts + 1) / 2))
        return false;
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    const bool use_lds = (d_model & 1) == 0 && d_model <= 8192;
    const std::uint32_t lds_blocks = static_cast<std::uint32_t>((num_experts + 7) / 8);
    const std::size_t lds_bytes = static_cast<std::size_t>(d_model) * sizeof(float);
    auto lds_completion = completion;
    lds_completion.contributors = lds_blocks;
#define LLAMINAR_LAUNCH_FP16_ROUTE_COMPLETION(HISTOGRAM, POLICY) \
    do { \
        if (use_lds) { \
            hipLaunchKernelGGL(HIP_KERNEL_NAME(rocm_moe_fp16_projection_route_descriptors_lds_kernel<HISTOGRAM, POLICY>), \
                dim3(lds_blocks), dim3(256), lds_bytes, static_cast<hipStream_t>(stream), \
                hidden, static_cast<const __half *>(gate_weights), logits, \
                static_cast<DeviceMoELayerRuntimeView *>(runtime), legacy_indices, legacy_weights, \
                d_model, num_experts, top_k, normalize_weights, write_legacy_outputs, \
                active_row_count_device, publication, lds_completion); \
        } else { \
            hipLaunchKernelGGL(HIP_KERNEL_NAME(rocm_moe_fp16_projection_route_descriptors_kernel<HISTOGRAM, POLICY>), \
                dim3(completion.contributors), dim3(256), 0, static_cast<hipStream_t>(stream), \
                hidden, static_cast<const __half *>(gate_weights), logits, \
                static_cast<DeviceMoELayerRuntimeView *>(runtime), legacy_indices, legacy_weights, \
                d_model, num_experts, top_k, normalize_weights, write_legacy_outputs, \
                active_row_count_device, publication, completion); \
        } \
    } while (false)
    if (row_policy == Policy::IntermediateShardLocal)
    {
        if (update_runtime_histogram) LLAMINAR_LAUNCH_FP16_ROUTE_COMPLETION(true, Policy::IntermediateShardLocal);
        else LLAMINAR_LAUNCH_FP16_ROUTE_COMPLETION(false, Policy::IntermediateShardLocal);
    }
    else
    {
        if (update_runtime_histogram) LLAMINAR_LAUNCH_FP16_ROUTE_COMPLETION(true, Policy::ParticipantAssigned);
        else LLAMINAR_LAUNCH_FP16_ROUTE_COMPLETION(false, Policy::ParticipantAssigned);
    }
#undef LLAMINAR_LAUNCH_FP16_ROUTE_COMPLETION
    return hipGetLastError() == hipSuccess;
}

/**
 * @brief Launch the spill-free unique-owner wave64 decode router.
 *
 * The caller must have prepared a graph whose capacity contract excludes
 * replicas.  The device kernel re-authenticates that invariant against the
 * active runtime bank and traps instead of selecting the replica-aware path if
 * the contract is violated.
 */
bool hipMoE_softmax_topk_decode_unique_owner_wave64(
    float *logits,
    void *runtime,
    float *legacy_indices,
    float *legacy_weights,
    int num_experts,
    int top_k,
    bool normalize_weights,
    bool write_legacy_outputs,
    bool update_runtime_histogram,
    int device_idx,
    void *stream,
        const int32_t *active_row_count_device)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();
    if (!logits || !runtime || !stream || num_experts <= 0 ||
        num_experts > kDeviceMoEMaxExperts || top_k <= 0 ||
        top_k > kDeviceMoEMaxTopK || top_k > num_experts)
    {
        return false;
    }
    if (write_legacy_outputs && (!legacy_indices || !legacy_weights))
        return false;

    /*
     * Give the router one wave for every populated 64-expert interval.  A
     * fixed four-wave workgroup made Qwen's 128-expert router carry two idle
     * waves through every workgroup barrier and through both second-level
     * reductions.  Removing only wholly inactive waves preserves the exact
     * reduction tree for every populated expert (the omitted inputs were
     * respectively -INF and +0), while materially reducing decode latency on
     * gfx906.  Keep this launch rule shared by the histogram and static
     * specializations so changing maintenance policy cannot change routing
     * arithmetic.
     */
    const int threads =
        llaminar2::moe_runtime_abi::wave64RouterThreads(num_experts);
    if (update_runtime_histogram)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                rocm_moe_softmax_topk_decode_independent_wave64_kernel<true>),
            dim3(1),
            dim3(threads),
            0,
            static_cast<hipStream_t>(stream),
            logits,
            static_cast<DeviceMoELayerRuntimeView *>(runtime),
            legacy_indices,
            legacy_weights,
            num_experts,
            top_k,
            normalize_weights,
            write_legacy_outputs, active_row_count_device);
    }
    else
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                rocm_moe_softmax_topk_decode_independent_wave64_kernel<false>),
            dim3(1),
            dim3(threads),
            0,
            static_cast<hipStream_t>(stream),
            logits,
            static_cast<DeviceMoELayerRuntimeView *>(runtime),
            legacy_indices,
            legacy_weights,
            num_experts,
            top_k,
            normalize_weights,
            write_legacy_outputs, active_row_count_device);
    }

    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(
            stderr,
            "HIP unique-owner MoE decode router launch failed: %s\n",
            hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_softmax_topk_decode_equivalent_rows_with_policy(
    float *logits,
    float *expert_indices, float *expert_weights,
    int seq_len, int num_experts, int top_k,
    bool normalize_weights,
    int device_idx, void *stream,
    const int *device_effective_seq_len,
    void *deferred_selected_route_ledger, bool qualified_default)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!logits || !expert_indices || !expert_weights || !stream ||
        seq_len <= 0 ||
        num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK || top_k > num_experts)
    {
        fprintf(stderr, "HIP MoE decode-equivalent rows softmax_topk invalid arguments\n");
        return false;
    }

    const int threads = 256;
    // This selection depends only on immutable graph geometry and discovery;
    // the live count remains a device-owned input in the shared body.
    bool use_register_selector = false;
    if (num_experts == kQFNRegisterSelectorExperts && top_k == kQFNRegisterSelectorTopK)
    {
        const bool is_wave64 =
            llaminar2::rocmHardwareMetadata().at(device_idx).native_wave_width == 64;
        if (is_wave64 &&
            ((seq_len == 1 && llaminar2::qfnGroupedTopKRegisterOptIn(true)) ||
             (device_effective_seq_len && seq_len >= 2 && seq_len <= 8 &&
              llaminar2::qfnGroupedTopKRegisterOptIn(qualified_default))))
        {
            use_register_selector = true;
        }
    }
    if (use_register_selector)
    {
        hipLaunchKernelGGL(qfn_grouped_softmax_register_selector_kernel,
            dim3(seq_len), dim3(threads), 0, static_cast<hipStream_t>(stream),
            logits, expert_indices, expert_weights,
            seq_len, num_experts, top_k, normalize_weights,
            device_effective_seq_len,
            static_cast<DeviceMoELayerRuntimeView *>(
                deferred_selected_route_ledger));
    }
    else
    {
        hipLaunchKernelGGL(rocm_moe_softmax_topk_decode_equivalent_rows_kernel,
            dim3(seq_len), dim3(threads), 0, static_cast<hipStream_t>(stream),
            logits, expert_indices, expert_weights,
            seq_len, num_experts, top_k, normalize_weights,
            device_effective_seq_len,
            static_cast<DeviceMoELayerRuntimeView *>(
                deferred_selected_route_ledger));
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE decode-equivalent rows softmax_topk kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    if (use_register_selector)
        llaminar2::PerfStatsCollector::addCounter("rocm_moe", "grouped_register_topk_calls", 1.0,
            "verifier", "ROCm:" + std::to_string(device_idx), {{"physical_rows", std::to_string(seq_len)}});
    return true;
}

bool hipMoE_softmax_topk_decode_equivalent_rows(
    float *logits,
    float *expert_indices, float *expert_weights,
    int seq_len, int num_experts, int top_k,
    bool normalize_weights,
    int device_idx, void *stream,
    const int *device_effective_seq_len,
    void *deferred_selected_route_ledger)
{
    return hipMoE_softmax_topk_decode_equivalent_rows_with_policy(logits, expert_indices, expert_weights, seq_len, num_experts, top_k, normalize_weights, device_idx, stream, device_effective_seq_len, deferred_selected_route_ledger, false);
}

bool hipMoE_router_kpart_reduce_softmax_topk_decode_runtime(
    const float *partials,
    float *router_probabilities,
    void *runtime,
    float *legacy_indices,
    float *legacy_weights,
    int num_experts, int k_partitions, int top_k,
    bool normalize_weights,
    bool write_legacy_outputs,
    bool update_runtime_histogram,
    llaminar2::RoutedExpertRowExecutionPolicy row_execution_policy,
    void *runtime_layers,
    const void *rebalance_plan_entries,
    uint32_t rebalance_plan_capacity,
    void *rebalance_command_header,
    const void *rebalance_local_transfer_slots,
    uint32_t rebalance_local_transfer_slot_count,
    const void *rebalance_config,
    void *rebalance_apply_status,
    void *rebalance_controller_state,
    int rebalance_target_layer,
    uint32_t rebalance_command_buffer_count,
    const int32_t *absolute_position_ids,
    int device_idx, void *stream,
        const int32_t *active_row_count_device)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!partials || !router_probabilities || !runtime ||
        num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        k_partitions <= 0 || top_k <= 0 || top_k > kDeviceMoEMaxTopK)
    {
        return false;
    }
    if (write_legacy_outputs && (!legacy_indices || !legacy_weights))
    {
        return false;
    }

    const int threads = 256;
    DeviceMoERebalanceConfigView rebalance_cfg{};
    if (rebalance_config)
        rebalance_cfg = *static_cast<const DeviceMoERebalanceConfigView *>(rebalance_config);
    hipLaunchKernelGGL(rocm_moe_router_kpart_reduce_softmax_topk_decode_runtime_kernel,
        dim3(1), dim3(threads), 0, static_cast<hipStream_t>(stream),
        partials,
        router_probabilities,
        static_cast<DeviceMoELayerRuntimeView *>(runtime),
        static_cast<DeviceMoELayerRuntimeView *>(runtime_layers),
        legacy_indices, legacy_weights,
        num_experts, k_partitions, top_k,
        normalize_weights,
        write_legacy_outputs,
        update_runtime_histogram,
        row_execution_policy,
        absolute_position_ids,
        static_cast<const DeviceMoERebalancePlanEntryView *>(rebalance_plan_entries),
        rebalance_plan_capacity,
        static_cast<DeviceMoERebalanceCommandBufferHeaderView *>(rebalance_command_header),
        static_cast<const DeviceMoEExpertDirectoryEntryView *>(rebalance_local_transfer_slots),
        rebalance_local_transfer_slot_count,
        rebalance_cfg,
        static_cast<DeviceMoERebalanceApplyStatusView *>(rebalance_apply_status),
        static_cast<DeviceMoERebalanceGraphControllerStateView *>(rebalance_controller_state),
        rebalance_target_layer,
        rebalance_command_buffer_count, active_row_count_device);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE router_kpart_reduce_softmax_topk_decode_runtime kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_decode_route_select_runtime(
    const int *expert_indices,
    const float *expert_weights,
    void *runtime,
    float *legacy_indices,
    float *legacy_weights,
    int num_experts, int top_k,
    bool write_legacy_outputs,
    bool update_runtime_histogram, llaminar2::RoutedExpertRowExecutionPolicy row_execution_policy,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!expert_indices || !expert_weights || !runtime || !stream || num_experts <= 0 ||
        num_experts > kDeviceMoEMaxExperts || top_k <= 0 || top_k > kDeviceMoEMaxTopK)
        return false;
    if (write_legacy_outputs && (!legacy_indices || !legacy_weights))
        return false;

    const int threads = 32;
    hipLaunchKernelGGL(rocm_moe_decode_route_select_runtime_kernel,
        dim3(1), dim3(threads), 0, static_cast<hipStream_t>(stream),
        expert_indices, expert_weights,
        static_cast<DeviceMoELayerRuntimeView *>(runtime),
        legacy_indices, legacy_weights,
        num_experts, top_k,
        write_legacy_outputs,
        update_runtime_histogram,
        row_execution_policy, /*absolute_position_ids=*/nullptr);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE decode_route_select_runtime kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_gather_tokens(
    const float *hidden, float *batch_buffer,
    const int *token_indices,
    int num_tokens, int d_model,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    hipLaunchKernelGGL(rocm_moe_gather_tokens_kernel,
        dim3(num_tokens), dim3(threads), 0, static_cast<hipStream_t>(stream),
        hidden, batch_buffer, token_indices, num_tokens, d_model);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE gather_tokens kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_copy_token_row(
    const float *source, float *row_buffer,
    int row_index, int row_width,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!source || !row_buffer || row_index < 0 || row_width <= 0 || !stream)
        return false;

    const int threads = 256;
    hipLaunchKernelGGL(rocm_moe_copy_row_kernel,
        dim3(1), dim3(threads), 0, static_cast<hipStream_t>(stream),
        source, row_buffer, row_index, row_width);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE copy_token_row kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_scatter_add(
    float *output, const float *expert_output,
    const int *token_indices, const float *weights,
    int num_tokens, int d_model,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    hipLaunchKernelGGL(rocm_moe_scatter_add_kernel,
        dim3(num_tokens), dim3(threads), 0, static_cast<hipStream_t>(stream),
        output, expert_output, token_indices, weights, num_tokens, d_model);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE scatter_add kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_write_token_row(
    float *destination, const float *row_buffer,
    int row_index, int row_width,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!destination || !row_buffer || row_index < 0 || row_width <= 0 || !stream)
        return false;

    const int threads = 256;
    hipLaunchKernelGGL(rocm_moe_write_row_kernel,
        dim3(1), dim3(threads), 0, static_cast<hipStream_t>(stream),
        destination, row_buffer, row_index, row_width);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE write_token_row kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_shared_expert_gate(
    const float *input, const float *gate_inp,
    float *shared_output, float *gate_scratch,
    int seq_len, int d_model,
    int device_idx, void *stream)
{
    // TIMING: skip device guard and error clear to test if they cause sync
    // llaminar2::HipDeviceGuard::setDevice(device_idx);
    // (void)hipGetLastError();

    // Phase 1: Compute dot products → sigmoid → gate_scratch[seq_len]
    {
        const int threads = 256;
        hipLaunchKernelGGL(rocm_moe_shared_expert_gate_dot_kernel,
            dim3(seq_len), dim3(threads), 0, static_cast<hipStream_t>(stream),
            input, gate_inp, gate_scratch, seq_len, d_model);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "HIP MoE shared_expert_gate dot kernel launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
    }

    // Phase 2: Scale shared_output by gate values
    {
        const int total = seq_len * d_model;
        const int threads = 256;
        const int blocks = (total + threads - 1) / threads;
        hipLaunchKernelGGL(rocm_moe_shared_expert_gate_scale_kernel,
            dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
            shared_output, gate_scratch, seq_len, d_model);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "HIP MoE shared_expert_gate scale kernel launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
    }

    return true;
}

bool hipMoE_shared_expert_gate_effective_seq_len(
    const float *input, const float *gate_inp,
    float *shared_output, float *gate_scratch,
    int seq_len, int d_model,
    const int *device_effective_seq_len,
    int device_idx, void *stream)
{
    if (!input || !gate_inp || !shared_output || !gate_scratch ||
        !device_effective_seq_len || seq_len <= 0 || d_model <= 0 || !stream)
    {
        fprintf(stderr, "HIP MoE shared_expert_gate_effective_seq_len invalid args\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    hipLaunchKernelGGL(rocm_moe_shared_expert_gate_dot_kernel,
        dim3(seq_len), dim3(threads), 0, static_cast<hipStream_t>(stream),
        input, gate_inp, gate_scratch, seq_len, d_model);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE shared_expert_gate_effective_seq_len dot kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    const int total = seq_len * d_model;
    const int blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(rocm_moe_shared_expert_gate_scale_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        shared_output, gate_scratch, seq_len, d_model, device_effective_seq_len);
    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE shared_expert_gate_effective_seq_len scale kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    return true;
}

/**
 * @brief Enqueue the existing scalar tree with an optional borrowed live-count authority.
 * @param input Resident full hidden row.
 * @param gate_inp Resident FP32 gate vector.
 * @param shared_output Resident in-place shared row.
 * @param d_model Positive full hidden width.
 * @param device_idx Original worker's HIP ordinal.
 * @param stream Exact non-null ordered producer stream.
 * @param device_effective_seq_len Optional stable device INT32, retaining the shared clamp contract.
 * @return Whether the exact native launch was accepted.
 */
bool hipMoE_shared_expert_gate_decode_fused(
    const float *input, const float *gate_inp,
    float *shared_output, int d_model,
    int device_idx, void *stream, const int *device_effective_seq_len)
{
    if (!input || !gate_inp || !shared_output || d_model <= 0 || !stream)
    {
        fprintf(stderr, "HIP MoE shared_expert_gate_decode_fused invalid args\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    hipLaunchKernelGGL(rocm_moe_shared_expert_gate_decode_fused_kernel,
        dim3(1), dim3(threads), 0, static_cast<hipStream_t>(stream),
        input, gate_inp, shared_output, d_model, device_effective_seq_len);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE shared_expert_gate_decode_fused kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    return true;
}

bool hipMoE_shared_expert_gate_add(
    const float *input, const float *gate_inp,
    float *shared_output, const float *routed_residual,
    float *combined_output, int seq_len, int d_model,
    int device_idx, void *stream)
{
    if (!input || !gate_inp || !shared_output || !routed_residual ||
        !combined_output || seq_len <= 0 || d_model <= 0)
    {
        fprintf(stderr, "HIP MoE shared_expert_gate_add invalid args\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(rocm_moe_shared_expert_gate_add_kernel<false>),
        dim3(seq_len), dim3(threads), 0, static_cast<hipStream_t>(stream),
        input, gate_inp, shared_output, routed_residual, combined_output,
        seq_len, d_model);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE shared_expert_gate_add kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    return true;
}

bool hipMoE_shared_expert_gate_add_effective_seq_len(
    const float *input, const float *gate_inp,
    float *shared_output, const float *routed_residual,
    float *combined_output, int seq_len, int d_model,
    const int *device_effective_seq_len,
    int device_idx, void *stream)
{
    if (!input || !gate_inp || !shared_output || !routed_residual ||
        !combined_output || !device_effective_seq_len ||
        seq_len <= 0 || d_model <= 0 || !stream)
    {
        fprintf(stderr, "HIP MoE shared_expert_gate_add_effective_seq_len invalid args\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(rocm_moe_shared_expert_gate_add_kernel<false>),
        dim3(seq_len), dim3(threads), 0, static_cast<hipStream_t>(stream),
        input, gate_inp, shared_output, routed_residual, combined_output,
        seq_len, d_model, device_effective_seq_len);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE shared_expert_gate_add_effective_seq_len kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    return true;
}

/** @brief Gate and combine owned columns using the canonical full-width dot and padded-row mask. */
bool hipMoE_shared_expert_column_gate_add(
    const float *input, const float *gate_inp, float *shared_output,
    const float *routed_columns, float *combined_columns,
    int rows, int model_columns, int local_columns, const int *active_rows,
    int ordinal, void *stream)
{
    if (!input || !gate_inp || !shared_output || !routed_columns || !combined_columns ||
        rows <= 0 || local_columns <= 0 || local_columns > model_columns || !stream)
        return false;
    if (hipSetDevice(ordinal) != hipSuccess || hipGetLastError() != hipSuccess) return false;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(rocm_moe_shared_expert_gate_add_kernel<true>),
        dim3(rows), dim3(256), 0, static_cast<hipStream_t>(stream),
        input, gate_inp, shared_output, routed_columns, combined_columns,
        rows, model_columns, active_rows, local_columns);
    return hipGetLastError() == hipSuccess;
}

bool rocmMoE_publish_shared_expert_rank_bank(
    const float *shared_output,
    float *canonical_publication,
    int seq_len,
    int top_k,
    int d_model,
    int participant_index,
    int participant_count,
    const int *device_effective_seq_len,
    int device_idx,
    void *stream)
{
    if (!shared_output || !canonical_publication || !stream ||
        seq_len <= 0 || top_k <= 0 || d_model <= 0 ||
        participant_count <= 0 || participant_index < 0 ||
        participant_index >= participant_count)
    {
        fprintf(
            stderr,
            "HIP MoE canonical shared rank-bank publication has invalid arguments\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    constexpr int kThreads = 256;
    const int column_work =
        (d_model & 3) == 0 ? (d_model >> 2) : d_model;
    const int column_blocks =
        (column_work + kThreads - 1) / kThreads;
    const dim3 grid(
        column_blocks > 0 ? column_blocks : 1,
        seq_len * participant_count);
    hipLaunchKernelGGL(
        rocm_moe_publish_shared_expert_rank_bank_kernel,
        grid,
        dim3(kThreads),
        0,
        static_cast<hipStream_t>(stream),
        shared_output,
        canonical_publication,
        seq_len,
        top_k,
        d_model,
        participant_index,
        participant_count,
        device_effective_seq_len);

    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(
            stderr,
            "HIP MoE canonical shared rank-bank publication launch failed: %s\n",
            hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmMoE_finalize_canonical_publication(
    const float *input,
    const float *gate_inp,
    const float *canonical_publication,
    float *routed_output,
    float *shared_output,
    float *combined_output,
    int seq_len,
    int top_k,
    int d_model,
    int participant_count,
    const int *device_effective_seq_len,
    int device_idx,
    void *stream)
{
    if (!input || !gate_inp || !canonical_publication ||
        !routed_output || !shared_output || !combined_output || !stream ||
        seq_len <= 0 || top_k <= 0 || d_model <= 0 ||
        participant_count <= 0)
    {
        fprintf(
            stderr,
            "HIP MoE canonical publication finalizer has invalid arguments\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    constexpr int kThreads = 256;
    const dim3 grid(
        (d_model + kCanonicalMoEOutputTileColumns - 1) /
            kCanonicalMoEOutputTileColumns,
        seq_len);
    hipLaunchKernelGGL(
        rocm_moe_finalize_canonical_publication_kernel,
        grid,
        dim3(kThreads),
        0,
        static_cast<hipStream_t>(stream),
        input,
        gate_inp,
        canonical_publication,
        routed_output,
        shared_output,
        combined_output,
        seq_len,
        top_k,
        d_model,
        participant_count,
        device_effective_seq_len);

    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(
            stderr,
            "HIP MoE canonical publication finalizer launch failed: %s\n",
            hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_swiglu(
    float *gate, const float *up,
    int count,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;
    if (blocks == 0 && count > 0) return true; // No-op for empty

    hipLaunchKernelGGL(rocm_moe_swiglu_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        gate, up, count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE swiglu kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_weighted_add(
    float *output, const float *input,
    float weight, int count,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (count <= 0) return true;

    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;

    hipLaunchKernelGGL(rocm_moe_weighted_add_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        output, input, weight, count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE weighted_add kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

} // extern "C"


namespace llaminar2::rocm
{
    /** @brief Share installed router arithmetic while assigning each live row once. */
    bool routeOwnedRows(const MoERouterOwnedRowsLaunch &launch, void *stream)
    {
        if (!stream || stream == hipStreamLegacy || stream == hipStreamPerThread ||
            !launch.valid() || launch.experts > kDeviceMoEMaxExperts || launch.top_k > kDeviceMoEMaxTopK)
            return false;
        const auto native_stream = static_cast<hipStream_t>(stream);
        const int rows = launch.partition.capacityFor(launch.capacity);
        const dim3 grid(launch.experts, (rows + 15) / 16);
        switch (launch.format)
        {
        case MoERouterPreparedFormat::BlockQ8:
        {
            const auto &hardware = requireROCmCurrentHardware(
                [](int *ordinal) { return hipGetDevice(ordinal); },
                [](int ordinal) -> const ROCmDeviceHardware & { return rocmHardwareMetadata().at(ordinal); });
            int stream_device = -1;
            if (hipStreamGetDevice(native_stream, &stream_device) != hipSuccess || stream_device != hardware.ordinal)
                return false;
            const auto selection = selectRouterQ8Prefill(hardware.ordinal,
                {launch.capacity,launch.width,launch.experts,launch.q8_basis,RouterQ8RowOwnership::CompactOwned},
                launch.live_rows ? DeviceRowRange::deviceCounted(launch.capacity,launch.live_rows) :
                    DeviceRowRange::fullyActive(launch.capacity));
            // Expert consumers still borrow ALL local hidden-Q8 rows. Row
            // ownership removes only logits/top-k duplication, not that input.
            const int groups = launch.capacity > 16 ? 8 : 1;
            if (launch.q8_basis == RouterQ8Basis::SignedHadamard32 && rows > 0)
            {
                if (!router_hadamard_q8::launch(launch.hidden, launch.hidden_q8, launch.hidden_scales,
                        launch.capacity, launch.width, stream, launch.live_rows)) return false;
            }
            else
            {
                hipLaunchKernelGGL(rocm_moe_router_hidden_quantize_q8_rows_kernel,
                dim3((launch.width / 32 + groups - 1) / groups, launch.capacity),
                dim3(groups * 32), 0, native_stream,
                launch.hidden, launch.hidden_q8, launch.hidden_scales,
                launch.capacity, launch.width, launch.live_rows);
            }
            if (hipGetLastError() != hipSuccess) return false;
            if (rows == 0) break;
            if (selection.usesWmma())
            {
                if (!routerQ8Prefill({
                        .hidden_codes = launch.hidden_q8, .hidden_scales = launch.hidden_scales,
                        .gate_codes = static_cast<const int8_t *>(launch.gate), .gate_scales = launch.gate_scales,
                        .logits = launch.logits, .capacity = launch.capacity, .width = launch.width,
                        .experts = launch.experts, .device = hardware.ordinal,
                        .live_rows = launch.live_rows, .partition = launch.partition, .basis = launch.q8_basis}, selection, stream)) return false;
            }
            else if (launch.width / 32 <= 64)
                hipLaunchKernelGGL((rocm_moe_gate_logits_q8_grouped_verifier_kernel<64, true>),
                    grid, dim3(64), 0, native_stream, launch.hidden_q8, launch.hidden_scales,
                    static_cast<const int8_t *>(launch.gate), launch.gate_scales, launch.logits,
                    launch.capacity, launch.width, launch.experts, launch.live_rows, launch.partition);
            else
                hipLaunchKernelGGL((rocm_moe_gate_logits_q8_grouped_verifier_kernel<128, true>),
                    grid, dim3(128), 0, native_stream, launch.hidden_q8, launch.hidden_scales,
                    static_cast<const int8_t *>(launch.gate), launch.gate_scales, launch.logits,
                    launch.capacity, launch.width, launch.experts, launch.live_rows, launch.partition);
            if (hipGetLastError() != hipSuccess) return false;
            record_rocm_router_q8_projection_submission(selection);
            if (launch.q8_basis == RouterQ8Basis::SignedHadamard32)
            {
                // Publish ALL live original rows for experts, including non-owned rows.
                hipLaunchKernelGGL(rocm_moe_router_hidden_quantize_q8_rows_kernel,
                dim3((launch.width / 32 + groups - 1) / groups, launch.capacity),
                dim3(groups * 32), 0, native_stream,
                launch.hidden, launch.hidden_q8, launch.hidden_scales,
                launch.capacity, launch.width, launch.live_rows);
                if (hipGetLastError() != hipSuccess) return false;
            }
            break;
        }
        case MoERouterPreparedFormat::FP32:
            if (rows == 0) break;
            hipLaunchKernelGGL((rocm_moe_gate_logits_fp32_grouped_verifier_kernel<true>),
                grid, dim3(128), 0, native_stream, launch.hidden,
                static_cast<const float *>(launch.gate), launch.logits,
                launch.capacity, launch.width, launch.experts, launch.live_rows, launch.partition);
            break;
        case MoERouterPreparedFormat::FP16:
            if (rows == 0) break;
            launch_fp16_router_rows<true>(launch.hidden, static_cast<const __half *>(launch.gate), launch.logits,
                launch.capacity, rows, launch.width, launch.experts, launch.live_rows, launch.partition, native_stream);
            break;
        case MoERouterPreparedFormat::BF16:
            if (rows == 0) break;
            hipLaunchKernelGGL((rocm_moe_gate_logits_bf16_grouped_verifier_kernel<true>),
                grid, dim3(128), 0, native_stream, launch.hidden,
                static_cast<const hip_bfloat16 *>(launch.gate), launch.logits,
                launch.capacity, launch.width, launch.experts, launch.live_rows, launch.partition);
            break;
        default:
            return false;
        }
        if (hipGetLastError() != hipSuccess) return false;
        hipLaunchKernelGGL((rocm_moe_softmax_topk_decode_equivalent_rows_kernel<true>),
            dim3(std::max(1, rows)), dim3(256), 0, native_stream, launch.logits, nullptr, nullptr,
            launch.capacity, launch.experts, launch.top_k, launch.normalize,
            launch.live_rows, nullptr, launch.partition, launch.selected, launch.selected_bytes);
        if (hipGetLastError() != hipSuccess) return false;
        if (launch.format == MoERouterPreparedFormat::FP16)
        {
            int ordinal = -1;
            if (hipGetDevice(&ordinal) != hipSuccess) return false;
            PerfStatsCollector::addCounter("kernel", "rocm_moe_router_fp16_operand_calls", 1.0,
                "router", "rocm:" + std::to_string(ordinal),
                {{"m", std::to_string(launch.capacity)}, {"d_model", std::to_string(launch.width)},
                 {"num_experts", std::to_string(launch.experts)}, {"activation", "fp16_rne"},
                 {"weight", "fp16"}, {"accumulation", "fp32"}, {"output", "fp32"}, {"route", "owned_fixed"}});
        }
        return true;
    }

    bool publishRouterOwnedRows(const MoERouterRowPublication &publication, void *stream)
    {
        if (!stream || !publication.valid()) return false;
        const auto slots = std::max(std::size_t(publication.layout.capacity) * publication.layout.top_k,
            std::size_t(publication.layout.partition.participants()));
        if ((slots + 127) / 128 > std::size_t(INT_MAX)) return false;
        hipLaunchKernelGGL(publish_router_owned_rows_kernel, dim3(unsigned((slots + 127) / 128)),
            dim3(128), 0, static_cast<hipStream_t>(stream), publication, publication.layout.packetBytes());
        return hipGetLastError() == hipSuccess;
    }
}
