/**
 * @file ROCmVerifierAttentionWmma.hip.inc
 * @brief Lane-exact FP16 grouped decode across two query rows and four heads.
 *
 * Every CTA owns one four-head interval within one KV head. Its eight
 * logical queries share sparse matrix products while retaining all sixty-four
 * original score lanes and two independent query states per physical wave.
 * The sole 32KiB shared bank progresses through score, maxima/sums, scales and
 * wave-output roles. Explicit barriers retire each role before reuse; queries,
 * row metadata and saved merge scales remain in registers.
 */

namespace verifier_attention_wmma_detail
{
/** @brief A split either owns no bytes, publishes neutral state, or consumes keys. */
enum class RowWork { Outside, Neutral, Active };

/** @brief One original row's immutable metadata for this split invocation. */
struct Row
{
    RowWork work;
    int begin, end, origin, capacity, waves;
};

/** @brief The sole CTA allocation; every role fits the same 32KiB bank exactly. */
struct alignas(16) Storage { float values[64 * 128]; };
static_assert(sizeof(Storage) == 32768);

/** @brief A bijection that spreads transposed lane reads without padding.
 * @param plane Original scalar score lane. @param output Query/key matrix index.
 * @return Address in the 32KiB bank, with no collisions across planes or outputs. */
__device__ __forceinline__ int scoreAddress(int plane, int output)
{ return plane * 128 + (output ^ plane); }

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

/** @brief Matrix producer for two query rows and four heads sharing one KV head.
 * @param Q Original FP32 queries, rounded once to the scalar FP16 operand policy.
 * @param K Original shared FP16 key bank. @param V Original shared FP16 value bank.
 * @param O_partial Original dimension-major split outputs. @param m_partial Original split maxima.
 * @param l_partial Original split probability sums. @param params Original per-row device publication.
 * @param row_count Physical query rows, including production sentinel rows.
 * @param heads Local query heads. @param kv_heads Visible shared-cache heads.
 * @param split_capacity Original retained split stride. @param num_cus Original device policy input.
 * @param scale Original softmax scale. @param forced_splits_present Effective scalar split override.
 * @param forced_splits Requested split count. @param forced_tpb_present Effective scalar wave override.
 * @param forced_tpb Requested physical thread count. @param head_start Replicated global head offset.
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
    const int wave = producer_wave % 4, quartet = producer_wave / 4;
    const int ratio = gqa_n_rep > 0 ? gqa_n_rep : heads / kv_heads;
    const int groups_per_kv = (ratio - 1) / 4 + 1;
    const int kv_head = blockIdx.x / groups_per_kv;
    const int first_kv_head = (blockIdx.x % groups_per_kv) * 4;
    const int first_head = kv_head * ratio + first_kv_head - (gqa_n_rep > 0 ? head_start : 0);
    const int head_count = min(4, ratio - first_kv_head);
    // Empty replicated-shard groups are uniformly outside the local interval.
    if (first_head >= heads || first_head + head_count <= 0) return;
    const int first_row = blockIdx.z * 2, split = blockIdx.y;
    const Row row0 = rowPolicy(params, first_row, row_count, heads, split, split_capacity,
        num_cus, forced_splits_present, forced_splits, forced_tpb_present, forced_tpb);
    const Row row1 = rowPolicy(params, first_row + 1, row_count, heads, split, split_capacity,
        num_cus, forced_splits_present, forced_splits, forced_tpb_present, forced_tpb);
    const bool active0 = row0.work == RowWork::Active, active1 = row1.work == RowWork::Active;
#pragma unroll
    for (int item = 0; item < 2; ++item)
    {
        const int q = quartet + item * 4, head = first_head + q % 4;
        const Row row = q < 4 ? row0 : row1;
        if (row.work == RowWork::Neutral && wave == 0 && q % 4 < head_count && head >= 0 && head < heads)
        {
            const size_t index = (static_cast<size_t>(first_row + q / 4) * heads + head) * split_capacity + split;
            if (lane == 0) { m_partial[index] = -FLT_MAX; l_partial[index] = 0.0f; }
#pragma unroll
            for (int d = 0; d < 4; ++d)
                O_partial[((static_cast<size_t>(first_row + q / 4) * heads + head) * 256 + lane + d * 64) * split_capacity + split] = 0.0f;
        }
    }
    if (!active0 && !active1) return;
    const Row common_row = active0 ? row0 : row1;
    const bool common = (!active0 || row0.waves == 4) && (!active1 || row1.waves == 4) &&
        (!active0 || !active1 || (row0.begin == row1.begin && row0.capacity == row1.capacity &&
            (row0.capacity == 0 || row0.origin == row1.origin)));
    const int common_end = max(active0 ? row0.end : 0, active1 ? row1.end : 0);
    const int steps = common ? (common_end - common_row.begin + 15) / 16 : max(
        active0 ? (row0.end - row0.begin + row0.waves * 2 - 1) / (row0.waves * 2) : 0,
        active1 ? (row1.end - row1.begin + row1.waves * 2 - 1) / (row1.waves * 2) : 0);
    const size_t key_stride = static_cast<size_t>(kv_heads) * 256;
    const size_t head_offset = static_cast<size_t>(kv_head) * 256;
    const int column = lane % 16;
    const Row operand_row = column < 4 ? row0 : row1;
    const int operand_head = first_head + column % 4;
    __half2 queries0[4]{}, queries1[4]{};
#pragma unroll
    for (int item = 0; item < 4; ++item)
    {
        const int plane = producer_wave * 4 + item;
        if (operand_row.work == RowWork::Active && column < 8 && column % 4 < head_count && operand_head >= 0 && operand_head < heads)
        {
            const float *q = Q + (static_cast<size_t>(first_row + column / 4) * heads + operand_head) * 256;
            queries0[item] = __halves2half2(__float2half(q[plane * 2]), __float2half(q[plane * 2 + 1]));
            queries1[item] = __halves2half2(__float2half(q[plane * 2 + 128]), __float2half(q[plane * 2 + 129]));
        }
    }
    float maxima[2]{-FLT_MAX, -FLT_MAX}, sums[2]{}, outputs[2][4]{};
#pragma unroll 1
    for (int step = 0; step < steps; ++step)
    {
        const Row key_row = column < 8 ? row0 : row1;
        const int logical = common ? common_row.begin + step * 16 + column :
            key_row.begin + column % 4 + (step * 2 + (column % 8) / 4) * key_row.waves;
        const bool valid = common ? logical < common_end : key_row.work == RowWork::Active &&
            column % 4 < key_row.waves && logical < key_row.end;
        const unsigned physical = physicalKey(logical, common ? common_row.origin : key_row.origin,
            common ? common_row.capacity : key_row.capacity);
#pragma unroll
        for (int item = 0; item < 4; ++item)
        {
            int plane = producer_wave * 4 + item;
            auto query0 = __builtin_bit_cast(uint32_t, queries0[item]);
            auto query1 = __builtin_bit_cast(uint32_t, queries1[item]);
            // Keep sparse tuples and swizzled store addresses local to this
            // matrix issue. Hoisting four tuples retains 24 zero VGPRs across
            // the PV loop. This empty identity emits no instruction and
            // changes no operand bits; it only bounds compiler live ranges.
            asm volatile("" : "+v"(query0), "+v"(query1), "+v"(plane));
            Half16 a{}, b{};
            const auto q0 = __builtin_bit_cast(_Float16_2, query0);
            const auto q1 = __builtin_bit_cast(_Float16_2, query1);
            a[12] = q0[0]; a[13] = q0[1]; a[14] = q1[0]; a[15] = q1[1];
            if (valid)
            {
                const auto *k = reinterpret_cast<const __half2 *>(K + physical * key_stride + head_offset);
                const auto k0 = __builtin_bit_cast(_Float16_2, k[plane]);
                const auto k1 = __builtin_bit_cast(_Float16_2, k[plane + 64]);
                b[12] = k0[0]; b[13] = k0[1]; b[14] = k1[0]; b[15] = k1[1];
            }
            const Float4 result = __builtin_amdgcn_wmma_f32_16x16x16_f16_w64(a, b, Float4{});
#pragma unroll
            for (int element = 0; element < 2; ++element)
                shared.values[scoreAddress(plane, element * 64 + lane)] = result[element];
        }
        __syncthreads();
#pragma unroll
        for (int item = 0; item < 2; ++item)
        {
            const int q = quartet + item * 4, head = first_head + q % 4;
            const Row row = q < 4 ? row0 : row1;
            if (row.work == RowWork::Active && wave < row.waves && q % 4 < head_count && head >= 0 && head < heads)
                for (int key = 0; key < (common ? 4 : 2); ++key)
                {
                    const int at = common ? common_row.begin + step * 16 + wave + key * 4 :
                        row.begin + wave + (step * 2 + key) * row.waves;
                    if (at >= row.end) break;
                    const int key_column = common ? wave + key * 4 : (q / 4) * 8 + key * 4 + wave;
                    const float partial = shared.values[scoreAddress(lane, q * 16 + key_column)];
                    const unsigned value_row = physicalKey(at, row.origin, row.capacity);
                    consumeKey(partial, V + value_row * key_stride + head_offset, lane,
                        scale, maxima[item], sums[item], outputs[item]);
                }
        }
        __syncthreads(); // Retire score reads before another tile or the merge roles.
    }
#pragma unroll
    for (int item = 0; item < 2; ++item)
    {
        const int q = quartet + item * 4, head = first_head + q % 4;
        const Row row = q < 4 ? row0 : row1;
        if (row.work == RowWork::Active && wave < row.waves && q % 4 < head_count && head >= 0 && head < heads && lane == 0)
        {
            shared.values[q * 4 + wave] = maxima[item];
            shared.values[64 + q * 4 + wave] = sums[item];
        }
    }
    __syncthreads();
#pragma unroll
    for (int item = 0; item < 2; ++item)
    {
        const int q = quartet + item * 4, head = first_head + q % 4;
        const Row row = q < 4 ? row0 : row1;
        if (row.work == RowWork::Active && wave == 0 && q % 4 < head_count && head >= 0 && head < heads && lane == 0)
        {
            const int base = q * 4;
            float maximum = shared.values[base], sum = shared.values[64 + base];
            shared.values[128 + base] = 1.0f;
            for (int w = 1; w < row.waves; ++w)
            {
                const float other_maximum = shared.values[base + w], other_sum = shared.values[64 + base + w];
                const float next = fmaxf(maximum, other_maximum);
                const float self_scale = fast_expf(maximum - next), other_scale = fast_expf(other_maximum - next);
                for (int previous = 0; previous < w; ++previous) shared.values[128 + base + previous] *= self_scale;
                shared.values[128 + base + w] = other_scale;
                sum = self_scale * sum + other_scale * other_sum;
                maximum = next;
            }
            const size_t index = (static_cast<size_t>(first_row + q / 4) * heads + head) * split_capacity + split;
            m_partial[index] = maximum;
            l_partial[index] = sum;
        }
    }
    __syncthreads();
    // Four owner waves each retain two queries' scales distributed over lanes
    // zero through three. Shuffles later recover their original ordered tree.
    float scales[2]{};
#pragma unroll
    for (int item = 0; item < 2; ++item)
    {
        const int q = quartet + item * 4, head = first_head + q % 4;
        const Row row = q < 4 ? row0 : row1;
        if (row.work == RowWork::Active && wave == 0 && lane < row.waves && q % 4 < head_count && head >= 0 && head < heads)
            scales[item] = shared.values[128 + q * 4 + lane];
    }
    __syncthreads(); // Every scale reader must finish before any O overwrite.
#pragma unroll
    for (int item = 0; item < 2; ++item)
    {
        const int q = quartet + item * 4, head = first_head + q % 4;
        const Row row = q < 4 ? row0 : row1;
        if (row.work == RowWork::Active && wave < row.waves && q % 4 < head_count && head >= 0 && head < heads)
#pragma unroll
            for (int d = 0; d < 4; ++d)
                shared.values[(q * 4 + wave) * 256 + lane + d * 64] = outputs[item][d];
    }
    __syncthreads();
#pragma unroll
    for (int item = 0; item < 2; ++item)
    {
        const int q = quartet + item * 4, head = first_head + q % 4;
        const Row row = q < 4 ? row0 : row1;
        if (row.work == RowWork::Active && wave == 0 && q % 4 < head_count && head >= 0 && head < heads)
#pragma unroll
            for (int d = 0; d < 4; ++d)
            {
                float value = 0.0f;
                for (int w = 0; w < row.waves; ++w)
                {
                    const float factor = __shfl(scales[item], w);
                    const float prior = shared.values[(q * 4 + w) * 256 + lane + d * 64];
                    asm("v_fma_f32 %0, %1, %2, %3" : "=v"(value) : "v"(factor), "v"(prior), "v"(value));
                }
                O_partial[((static_cast<size_t>(first_row + q / 4) * heads + head) * 256 + lane + d * 64) * split_capacity + split] = value;
            }
    }
#endif
}
}
