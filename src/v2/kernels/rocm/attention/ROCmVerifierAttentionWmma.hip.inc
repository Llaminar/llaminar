/**
 * @file ROCmVerifierAttentionWmma.hip.inc
 * @brief Lane-exact FP16 grouped decode using sparse gfx1100 matrix products.
 *
 * Included after the original scalar decode policy/helpers. Four query rows
 * share one CTA, but retain independent split ranges, ring mappings, original
 * wave-strided key lists and lane-owned softmax/PV state. Matrix instructions
 * produce every original two-fdot2 partial; the original XOR tree consumes all
 * sixty-four partials, including lane-specific NaN payloads. A shared root is
 * not interchangeable with those lanes. Score storage becomes merge storage
 * only after every score reader reaches the final retirement barrier.
 */

namespace verifier_attention_wmma_detail
{
/** @brief Rotate score planes across LDS banks when consumers transpose lane ownership. */
inline constexpr int kScorePlaneStride = 65;

/** @brief Keep half2 alignment while rotating the four query rows across LDS banks. */
inline constexpr int kQueryRowStride = 258;

/** @brief Pad a cooperative key row so matrix consumers address distinct LDS banks. */
inline constexpr int kKeyRowStride = 258;

/** @brief A split either owns no bytes, publishes neutral state, or consumes keys. */
enum class RowWork { Outside, Neutral, Active };

/** @brief One original row's immutable metadata for this split invocation. */
struct Row
{
    RowWork work;
    int begin, end, origin, capacity, waves;
};

/** @brief CTA-owned transient storage; the flat scratch bank has one live role at a time. */
struct alignas(16) Storage
{
    Row rows[4];
    // Matrix lanes broadcast four different query rows. A 256-half stride
    // places all four distinct words in one bank; 258 halves rotate them by
    // one DWORD while retaining the half2 loads' four-byte alignment.
    __half queries[4 * kQueryRowStride];
    // One producer wave loads each complete key contiguously. Matrix lanes
    // then reuse its exact half bits instead of issuing scattered global
    // half2 loads separately for every sparse arithmetic plane.
    __half keys[16 * kKeyRowStride];
    // Producers write consecutive columns; consumers read one column from
    // every original lane plane. A 64-float stride aliases those reads onto
    // one gfx1100 LDS bank. The extra word rotates all 64 lanes across banks.
    float score_or_merge[64 * kScorePlaneStride];
    float maxima[16], sums[16], scales[16];
    int first_active, steps, common_begin, common_end, common_origin, common_capacity;
    bool common_keys;
};

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

/** @brief Matrix producer for four shared-KV query rows with independent scalar decode metadata.
 * @param Q Original FP32 queries, converted once to the scalar FP16 operand policy.
 * @param K Original shared FP16 key bank. @param V Original shared FP16 value bank.
 * @param O_partial Original dimension-major split outputs. @param m_partial Original split maxima.
 * @param l_partial Original split probability sums. @param params Original per-row device publication.
 * @param row_count Physical query rows, including the production sentinel rows.
 * @param heads Local query heads. @param kv_heads Visible shared-cache heads.
 * @param split_capacity Original retained split stride. @param num_cus Original device policy input.
 * @param scale Original softmax scale. @param forced_splits_present Effective scalar split override.
 * @param forced_splits Original requested split count. @param forced_tpb_present Effective scalar wave override.
 * @param forced_tpb Original requested physical thread count. @param head_start Replicated global head offset.
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
    const int tid = threadIdx.x, lane = tid % 64, wave = (tid / 64) % 4, query = tid / 256;
    const int first_row = blockIdx.z * 4, head = blockIdx.x, split = blockIdx.y;
    if (tid < 4)
    {
        Row row{};
        if (first_row + tid < row_count)
        {
            const auto p = params[first_row + tid];
            const auto config = selectDecodeRowConfig(p.kv_len, 256, heads, num_cus, split_capacity,
                forced_splits_present, forced_splits, forced_tpb_present, forced_tpb);
            const int size = (p.kv_len + config.splits - 1) / config.splits;
            row = {split >= config.splits ? RowWork::Outside :
                       split * size >= p.kv_len ? RowWork::Neutral : RowWork::Active,
                   split * size, min((split + 1) * size, p.kv_len),
                   p.ring_row_origin, p.ring_row_capacity > 0 ? p.ring_row_capacity : 0,
                   config.threads_per_block / 64};
        }
        shared.rows[tid] = row;
    }
    __syncthreads();
    const Row row = shared.rows[query];
    const int local_dimension = tid % 256;
    shared.queries[query * kQueryRowStride + local_dimension] = row.work == RowWork::Active
        ? __float2half(Q[(static_cast<size_t>(first_row + query) * heads + head) * 256 + local_dimension])
        : __float2half(0.0f);
    if (tid == 0)
    {
        shared.first_active = -1;
        shared.steps = 0;
        shared.common_keys = true;
        for (int q = 0; q < 4; ++q)
        {
            const Row candidate = shared.rows[q];
            if (candidate.work != RowWork::Active) continue;
            shared.steps = max(shared.steps, (candidate.end - candidate.begin + candidate.waves - 1) / candidate.waves);
            if (shared.first_active < 0)
            {
                shared.first_active = q;
                shared.common_begin = candidate.begin;
                shared.common_end = candidate.end;
                shared.common_origin = candidate.origin;
                shared.common_capacity = candidate.capacity;
            }
            shared.common_end = max(shared.common_end, candidate.end);
            shared.common_keys = shared.common_keys && candidate.waves == 4 &&
                candidate.begin == shared.common_begin && candidate.capacity == shared.common_capacity &&
                (candidate.capacity == 0 || candidate.origin == shared.common_origin);
        }
        if (shared.first_active >= 0 && shared.common_keys)
            shared.steps = (shared.common_end - shared.common_begin + 15) / 16;
    }
    // Original empty splits publish neutral bytes; surplus and physical tails own no output.
    const int global_row = first_row + query;
    const size_t partial_index = (static_cast<size_t>(global_row) * heads + head) * split_capacity + split;
    if (row.work == RowWork::Neutral && wave == 0)
    {
        if (lane == 0) { m_partial[partial_index] = -FLT_MAX; l_partial[partial_index] = 0.0f; }
#pragma unroll
        for (int index = 0; index < 4; ++index)
            O_partial[((static_cast<size_t>(global_row) * heads + head) * 256 + lane + index * 64) * split_capacity + split] = 0.0f;
    }
    __syncthreads();
    if (shared.first_active < 0) return; // Uniform across the entire CTA.

    const int ratio = gqa_n_rep > 0 ? gqa_n_rep : heads / kv_heads;
    const int kv_head = gqa_n_rep > 0 ? (head_start + head) / ratio : head / ratio;
    const size_t key_stride = static_cast<size_t>(kv_heads) * 256;
    const size_t head_offset = static_cast<size_t>(kv_head) * 256;
    const int column = lane % 16, producer_wave = tid / 64;
    float output[4]{}, maximum = -FLT_MAX, sum = 0.0f;
#pragma unroll 1
    for (int step = 0; step < shared.steps; ++step)
    {
        // Every wave owns a key column, independently of the query row it
        // consumes later. Inactive query waves must still supply other rows'
        // keys and reach the publication barrier.
        const Row key_row = shared.rows[producer_wave / 4];
        const int key = shared.common_keys ? shared.common_begin + step * 16 + producer_wave :
            key_row.begin + producer_wave % 4 + step * key_row.waves;
        const bool key_valid = shared.common_keys ? key < shared.common_end :
            key_row.work == RowWork::Active && producer_wave % 4 < key_row.waves && key < key_row.end;
        const unsigned physical = physicalKey(key,
            shared.common_keys ? shared.common_origin : key_row.origin,
            shared.common_keys ? shared.common_capacity : key_row.capacity);
        auto *staged = reinterpret_cast<__half2 *>(shared.keys + producer_wave * kKeyRowStride);
        __half2 first{}, second{};
        if (key_valid)
        {
            const auto *source = reinterpret_cast<const __half2 *>(K + physical * key_stride + head_offset);
            first = source[lane];
            second = source[lane + 64];
        }
        staged[lane] = first;
        staged[lane + 64] = second;
        __syncthreads(); // Publish every key column before matrix consumers read it.
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
        if (row.work == RowWork::Active && wave < row.waves)
        {
            const int count = shared.common_keys ? 4 : 1;
            for (int item = 0; item < count; ++item)
            {
                const int logical = shared.common_keys ? shared.common_begin + step * 16 + wave + item * 4 :
                    row.begin + wave + step * row.waves;
                if (logical >= row.end) break;
                const int key_column = shared.common_keys ? wave + item * 4 : query * 4 + wave;
                const float partial = shared.score_or_merge[lane * kScorePlaneStride + query * 16 + key_column];
                const unsigned value_row = physicalKey(logical, row.origin, row.capacity);
                consumeKey(partial, V + value_row * key_stride + head_offset, lane, scale, maximum, sum, output);
            }
        }
        __syncthreads(); // Retire all reads before the next matrix tile or merge publication.
    }
    if (row.work == RowWork::Active && wave < row.waves)
    {
        if (lane == 0) { shared.maxima[query * 4 + wave] = maximum; shared.sums[query * 4 + wave] = sum; }
#pragma unroll
        for (int index = 0; index < 4; ++index)
            shared.score_or_merge[(query * 4 + wave) * 256 + lane + index * 64] = output[index];
    }
    __syncthreads();
    if (row.work == RowWork::Active && wave == 0 && lane == 0)
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
        m_partial[partial_index] = final_maximum;
        l_partial[partial_index] = final_sum;
    }
    __syncthreads();
    if (row.work == RowWork::Active && wave == 0)
        for (int dimension = lane; dimension < 256; dimension += 64)
        {
            float value = 0.0f;
            for (int w = 0; w < row.waves; ++w)
                value += shared.scales[query * 4 + w] * shared.score_or_merge[(query * 4 + w) * 256 + dimension];
            O_partial[((static_cast<size_t>(global_row) * heads + head) * 256 + dimension) * split_capacity + split] = value;
        }
#endif
}
}
