/**
 * @file ROCmVerifierAttentionWmma.hip.inc
 * @brief Lane-exact FP16 grouped decode for one query row and four shared-KV heads.
 *
 * A CTA shares one original row policy across four adjacent query heads. All
 * sixteen waves cooperatively stage keys, independently of the original
 * scalar wave count. Sparse matrix products retain all sixty-four original
 * score lanes; each consumer keeps its serial wave-strided key order, online
 * softmax and PV state. Barriers retire score reads before merge-bank reuse.
 */

namespace verifier_attention_wmma_detail
{
/** @brief Rotate transposed score-plane reads across the LDS banks. */
inline constexpr int kScorePlaneStride = 65;
/** @brief Preserve half2 alignment while rotating query heads across LDS banks. */
inline constexpr int kQueryRowStride = 258;
/** @brief Rotate cooperatively staged key rows across LDS banks. */
inline constexpr int kKeyRowStride = 258;
/** @brief A split owns no bytes, publishes neutral state, or consumes keys. */
enum class RowWork { Outside, Neutral, Active };
/** @brief One original query row's immutable split and ring policy. */
struct Row { RowWork work; int begin, end, origin, capacity, waves; };
/** @brief CTA storage with explicit score-to-output lifetime reuse. */
struct alignas(16) Storage
{
    __half queries[4 * kQueryRowStride];
    __half keys[16 * kKeyRowStride];
    float score_or_merge[64 * kScorePlaneStride];
    float maxima[16], sums[16], scales[16];
};
static_assert(sizeof(Storage) == 27152);

/** @brief Translate an original logical key without changing ring traversal order.
 * @param logical Original key position. @param origin Published physical origin.
 * @param capacity Positive ring capacity, or zero for linear storage.
 * @return The original physical row, using unsigned addition like scalar decode. */
__device__ __forceinline__ unsigned physicalKey(int logical, int origin, int capacity)
{
    return capacity > 0
        ? (static_cast<unsigned>(origin) + static_cast<unsigned>(logical)) % static_cast<unsigned>(capacity)
        : static_cast<unsigned>(logical);
}

/** @brief Preserve one original wave's lane-local score and online softmax/PV updates.
 * @param partial Original lane's two-fdot2 result. @param values This key's FP16 V row.
 * @param lane Original wave lane. @param scale Original softmax scale.
 * @param maximum Lane-owned running maximum. @param sum Lane-owned probability sum.
 * @param output Four original lane-owned output accumulators. */
__device__ __forceinline__ void consumeKey(float partial, const __half *values, int lane,
    float scale, float &maximum, float &sum, float (&output)[4])
{
    const float score = wavefrontReduceSum(partial) * scale;
    const float next_maximum = fmaxf(maximum, score);
    const float scale_old = fast_expf(maximum - next_maximum);
    const float probability = fast_expf(score - next_maximum);
    sum = sum * scale_old + probability;
#pragma unroll
    for (int index = 0; index < 4; ++index)
    {
        const float product = probability * __half2float(values[lane + index * 64]);
        // Scalar decode lowers this update with scale_old as the first FMA
        // operand. Both multiplicands may carry different NaN payloads; LLVM
        // may commute a C++ multiply even when contraction/order is otherwise
        // unchanged. Preserve the actual scalar instruction's operand order.
        float next_output;
        asm("v_fma_f32 %0, %1, %2, %3"
            : "=v"(next_output) : "v"(scale_old), "v"(output[index]), "v"(product));
        output[index] = next_output;
    }
    maximum = next_maximum;
}

/** @brief Read one original row policy into registers, without sharing host state.
 * @param params Original per-row publication. @param row Physical query row.
 * @param rows Retained row capacity. @param heads Original local query-head count.
 * @param split This CTA's split. @param splits Original split-capacity stride.
 * @param cus Original device policy input. @param force_splits Split override presence.
 * @param forced_splits Requested split count. @param force_threads Wave override presence.
 * @param forced_threads Requested original thread count. @return Exact original row policy. */
__device__ __forceinline__ Row rowPolicy(const llaminar2::attention::AttentionDeviceParams *params,
    int row, int rows, int heads, int split, int splits, int cus,
    bool force_splits, int forced_splits, bool force_threads, int forced_threads)
{
    if (row >= rows) return {};
    const auto p = params[row];
    const auto config = selectDecodeRowConfig(p.kv_len, 256, heads, cus, splits,
        force_splits, forced_splits, force_threads, forced_threads);
    const int size = (p.kv_len + config.splits - 1) / config.splits;
    return {split >= config.splits ? RowWork::Outside :
                split * size >= p.kv_len ? RowWork::Neutral : RowWork::Active,
            split * size, min((split + 1) * size, p.kv_len),
            p.ring_row_origin, p.ring_row_capacity > 0 ? p.ring_row_capacity : 0,
            config.threads_per_block / 64};
}

/** @brief Matrix producer for four heads sharing one original query-row policy.
 * @param Q Original FP32 queries, rounded once to the scalar FP16 operand policy.
 * @param K Shared FP16 key bank. @param V Shared FP16 value bank.
 * @param O_partial Original dimension-major split outputs. @param m_partial Split maxima.
 * @param l_partial Split probability sums. @param params Original device row publication.
 * @param row_count Retained query rows, including production sentinel rows.
 * @param heads Local query heads. @param kv_heads Visible shared-cache heads.
 * @param split_capacity Original split stride. @param num_cus Original device policy input.
 * @param scale Original softmax scale. @param forced_splits_present Split override presence.
 * @param forced_splits Requested split count. @param forced_tpb_present Wave override presence.
 * @param forced_tpb Requested original thread count. @param head_start Replicated head offset.
 * @param gqa_n_rep Replicated head ratio, or zero for ordinary local GQA mapping.
 */
__global__ __launch_bounds__(1024)
void flash_decode_verifier_wmma_fp16_hd256_kernel(const float *__restrict__ Q,
    const __half *__restrict__ K, const __half *__restrict__ V,
    float *__restrict__ O_partial, float *__restrict__ m_partial, float *__restrict__ l_partial,
    const llaminar2::attention::AttentionDeviceParams *__restrict__ params,
    int row_count, int heads, int kv_heads, int split_capacity, int num_cus, float scale,
    bool forced_splits_present, int forced_splits, bool forced_tpb_present, int forced_tpb,
    int head_start, int gqa_n_rep)
{
#if defined(__gfx1100__)
    using Half16 = _Float16 __attribute__((ext_vector_type(16)));
    using Float4 = float __attribute__((ext_vector_type(4)));
    __shared__ Storage shared;
    const int tid = threadIdx.x, lane = tid % 64, producer_wave = tid / 64;
    const int wave = producer_wave % 4, query = producer_wave / 4;
    const int ratio = gqa_n_rep > 0 ? gqa_n_rep : heads / kv_heads;
    const int groups_per_kv = (ratio - 1) / 4 + 1;
    const int kv_head = blockIdx.x / groups_per_kv;
    const int first_kv_head = (blockIdx.x % groups_per_kv) * 4;
    const int first_head = kv_head * ratio + first_kv_head - (gqa_n_rep > 0 ? head_start : 0);
    const int head_count = min(4, ratio - first_kv_head);
    // This test is uniform. Individual unowned heads must still stage keys
    // and reach every barrier for the other heads in their CTA.
    if (first_head >= heads || first_head + head_count <= 0) return;
    const int global_row = blockIdx.z, split = blockIdx.y, head = first_head + query;
    const bool owns_head = query < head_count && head >= 0 && head < heads;
    const Row row = rowPolicy(params, global_row, row_count, heads, split, split_capacity,
        num_cus, forced_splits_present, forced_splits, forced_tpb_present, forced_tpb);
    const size_t partial_index = (static_cast<size_t>(global_row) * heads + head) * split_capacity + split;
    if (row.work == RowWork::Neutral && owns_head && wave == 0)
    {
        if (lane == 0) { m_partial[partial_index] = -FLT_MAX; l_partial[partial_index] = 0.0f; }
#pragma unroll
        for (int d = 0; d < 4; ++d)
            O_partial[((static_cast<size_t>(global_row) * heads + head) * 256 + lane + d * 64) * split_capacity + split] = 0.0f;
    }
    if (row.work != RowWork::Active) return; // The original row policy is uniform.
    const int dimension = tid % 256;
    shared.queries[query * kQueryRowStride + dimension] = owns_head
        ? __float2half(Q[(static_cast<size_t>(global_row) * heads + head) * 256 + dimension])
        : __float2half(0.0f);
    __syncthreads();
    const int steps = (row.end - row.begin + 15) / 16;
    const size_t key_stride = static_cast<size_t>(kv_heads) * 256;
    const size_t head_offset = static_cast<size_t>(kv_head) * 256;
    const int column = lane % 16;
    float output[4]{}, maximum = -FLT_MAX, sum = 0.0f;
#pragma unroll 1
    for (int step = 0; step < steps; ++step)
    {
        // All sixteen physical waves supply keys even when the original
        // scalar implementation uses only one, two or three consumer waves.
        const int key = row.begin + step * 16 + producer_wave;
        auto *staged = reinterpret_cast<__half2 *>(shared.keys + producer_wave * kKeyRowStride);
        __half2 first{}, second{};
        if (key < row.end)
        {
            const unsigned physical = physicalKey(key, row.origin, row.capacity);
            const auto *source = reinterpret_cast<const __half2 *>(K + physical * key_stride + head_offset);
            first = source[lane]; second = source[lane + 64];
        }
        staged[lane] = first; staged[lane + 64] = second;
        __syncthreads();
#pragma unroll 1
        for (int item = 0; item < 4; ++item)
        {
            const int plane = producer_wave * 4 + item;
            Half16 a{}, b{};
            if (column < 4)
            {
                const auto *q = reinterpret_cast<const __half2 *>(shared.queries + column * kQueryRowStride);
                const auto q0 = __builtin_bit_cast(_Float16_2, q[plane]);
                const auto q1 = __builtin_bit_cast(_Float16_2, q[plane + 64]);
                a[12] = q0[0]; a[13] = q0[1]; a[14] = q1[0]; a[15] = q1[1];
            }
            const auto *k = reinterpret_cast<const __half2 *>(shared.keys + column * kKeyRowStride);
            const auto k0 = __builtin_bit_cast(_Float16_2, k[plane]);
            const auto k1 = __builtin_bit_cast(_Float16_2, k[plane + 64]);
            b[12] = k0[0]; b[13] = k0[1]; b[14] = k1[0]; b[15] = k1[1];
            shared.score_or_merge[plane * kScorePlaneStride + lane] =
                __builtin_amdgcn_wmma_f32_16x16x16_f16_w64(a, b, Float4{})[0];
        }
        __syncthreads();
        if (owns_head && wave < row.waves)
        {
            // A sixteen-key boundary changes the first column only for
            // three-wave rows. Preserve begin + wave + j*waves exactly,
            // including nonzero split beginnings and ring wraparound.
            int first_column = wave;
            if (row.waves == 3)
            {
                first_column -= step % 3;
                if (first_column < 0) first_column += 3;
            }
            for (int key_column = first_column; key_column < 16; key_column += row.waves)
            {
                const int logical = row.begin + step * 16 + key_column;
                if (logical >= row.end) break;
                const float partial = shared.score_or_merge[lane * kScorePlaneStride + query * 16 + key_column];
                const unsigned physical = physicalKey(logical, row.origin, row.capacity);
                consumeKey(partial, V + physical * key_stride + head_offset, lane, scale, maximum, sum, output);
            }
        }
        __syncthreads(); // Retire score readers before the next tile or merge.
    }
    if (owns_head && wave < row.waves)
    {
        if (lane == 0) { shared.maxima[query * 4 + wave] = maximum; shared.sums[query * 4 + wave] = sum; }
#pragma unroll
        for (int d = 0; d < 4; ++d)
            shared.score_or_merge[(query * 4 + wave) * 256 + lane + d * 64] = output[d];
    }
    __syncthreads();
    if (owns_head && wave == 0 && lane == 0)
    {
        const int base = query * 4;
        float final_maximum = shared.maxima[base], final_sum = shared.sums[base];
        shared.scales[base] = 1.0f;
        for (int w = 1; w < row.waves; ++w)
        {
            const float other_maximum = shared.maxima[base + w], other_sum = shared.sums[base + w];
            const float next = fmaxf(final_maximum, other_maximum);
            const float self_scale = fast_expf(final_maximum - next), other_scale = fast_expf(other_maximum - next);
            for (int previous = 0; previous < w; ++previous) shared.scales[base + previous] *= self_scale;
            shared.scales[base + w] = other_scale;
            final_sum = self_scale * final_sum + other_scale * other_sum;
            final_maximum = next;
        }
        m_partial[partial_index] = final_maximum; l_partial[partial_index] = final_sum;
    }
    __syncthreads();
    if (owns_head && wave == 0)
#pragma unroll
        for (int d = 0; d < 4; ++d)
        {
            float value = 0.0f;
            for (int w = 0; w < row.waves; ++w)
            {
                const float factor = shared.scales[query * 4 + w];
                const float prior = shared.score_or_merge[(query * 4 + w) * 256 + lane + d * 64];
                asm("v_fma_f32 %0, %1, %2, %3" : "=v"(value) : "v"(factor), "v"(prior), "v"(value));
            }
            O_partial[((static_cast<size_t>(global_row) * heads + head) * 256 + lane + d * 64) * split_capacity + split] = value;
        }
#endif
}
}
