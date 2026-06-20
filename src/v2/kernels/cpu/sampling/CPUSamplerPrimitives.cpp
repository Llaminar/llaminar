#include "CPUSamplerPrimitives.h"

#include "../../../utils/CPUFeatures.h"

#include <algorithm>
#include <limits>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

namespace llaminar2::cpu_sampling
{
    namespace
    {
        inline int sanitize_topk(int vocab_size, int top_k)
        {
            if (vocab_size <= 0 || top_k <= 0)
            {
                return 0;
            }
            return std::min(vocab_size, top_k);
        }

        inline void initialize_outputs(float *out_logits, int *out_ids, int top_k)
        {
            for (int i = 0; i < top_k; ++i)
            {
                out_logits[i] = -std::numeric_limits<float>::infinity();
                out_ids[i] = -1;
            }
        }

        inline void insert_candidate(
            float value,
            int token_id,
            float *out_logits,
            int *out_ids,
            int &count,
            int top_k)
        {
            if (top_k <= 0)
            {
                return;
            }

            if (count == top_k && !(value > out_logits[top_k - 1]))
            {
                return;
            }

            int pos = count < top_k ? count++ : top_k - 1;
            while (pos > 0 && value > out_logits[pos - 1])
            {
                out_logits[pos] = out_logits[pos - 1];
                out_ids[pos] = out_ids[pos - 1];
                --pos;
            }

            out_logits[pos] = value;
            out_ids[pos] = token_id;
        }

        inline int select_topk_serial_scan(
            const float *logits,
            int vocab_size,
            int top_k,
            float *out_logits,
            int *out_ids)
        {
            top_k = sanitize_topk(vocab_size, top_k);
            if (!logits || !out_logits || !out_ids || top_k == 0)
            {
                return 0;
            }

            initialize_outputs(out_logits, out_ids, top_k);

            int count = 0;
            for (int i = 0; i < vocab_size; ++i)
            {
                insert_candidate(logits[i], i, out_logits, out_ids, count, top_k);
            }
            return count;
        }
    } // namespace

    int select_topk_scalar(
        const float *logits,
        int vocab_size,
        int top_k,
        float *out_logits,
        int *out_ids)
    {
        return select_topk_serial_scan(logits, vocab_size, top_k, out_logits, out_ids);
    }

#if defined(__AVX2__)
    int select_topk_avx2(
        const float *logits,
        int vocab_size,
        int top_k,
        float *out_logits,
        int *out_ids)
    {
        top_k = sanitize_topk(vocab_size, top_k);
        if (!logits || !out_logits || !out_ids || top_k == 0)
        {
            return 0;
        }

        initialize_outputs(out_logits, out_ids, top_k);

        alignas(32) float lane_values[8];
        int count = 0;
        int i = 0;
        const int vec_end = vocab_size & ~7;
        for (; i < vec_end; i += 8)
        {
            const float threshold =
                count == top_k ? out_logits[top_k - 1] : -std::numeric_limits<float>::infinity();
            const __m256 values = _mm256_loadu_ps(logits + i);
            int mask = 0xFF;
            if (count == top_k)
            {
                const __m256 limit = _mm256_set1_ps(threshold);
                mask = _mm256_movemask_ps(_mm256_cmp_ps(values, limit, _CMP_GT_OQ));
                if (mask == 0)
                {
                    continue;
                }
            }

            _mm256_store_ps(lane_values, values);
            for (int lane = 0; lane < 8; ++lane)
            {
                if (mask & (1 << lane))
                {
                    insert_candidate(lane_values[lane], i + lane, out_logits, out_ids, count, top_k);
                }
            }
        }

        for (; i < vocab_size; ++i)
        {
            insert_candidate(logits[i], i, out_logits, out_ids, count, top_k);
        }
        return count;
    }
#else
    int select_topk_avx2(
        const float *logits,
        int vocab_size,
        int top_k,
        float *out_logits,
        int *out_ids)
    {
        return select_topk_scalar(logits, vocab_size, top_k, out_logits, out_ids);
    }
#endif

#if defined(__AVX512F__)
    int select_topk_avx512(
        const float *logits,
        int vocab_size,
        int top_k,
        float *out_logits,
        int *out_ids)
    {
        top_k = sanitize_topk(vocab_size, top_k);
        if (!logits || !out_logits || !out_ids || top_k == 0)
        {
            return 0;
        }

        initialize_outputs(out_logits, out_ids, top_k);

        alignas(64) float lane_values[16];
        int count = 0;
        int i = 0;
        const int vec_end = vocab_size & ~15;
        for (; i < vec_end; i += 16)
        {
            const float threshold =
                count == top_k ? out_logits[top_k - 1] : -std::numeric_limits<float>::infinity();
            const __m512 values = _mm512_loadu_ps(logits + i);
            __mmask16 mask = 0xFFFF;
            if (count == top_k)
            {
                const __m512 limit = _mm512_set1_ps(threshold);
                mask = _mm512_cmp_ps_mask(values, limit, _CMP_GT_OQ);
                if (mask == 0)
                {
                    continue;
                }
            }

            _mm512_store_ps(lane_values, values);
            for (int lane = 0; lane < 16; ++lane)
            {
                if (mask & (static_cast<__mmask16>(1) << lane))
                {
                    insert_candidate(lane_values[lane], i + lane, out_logits, out_ids, count, top_k);
                }
            }
        }

        for (; i < vocab_size; ++i)
        {
            insert_candidate(logits[i], i, out_logits, out_ids, count, top_k);
        }
        return count;
    }
#else
    int select_topk_avx512(
        const float *logits,
        int vocab_size,
        int top_k,
        float *out_logits,
        int *out_ids)
    {
        return select_topk_avx2(logits, vocab_size, top_k, out_logits, out_ids);
    }
#endif

    int select_topk(
        const float *logits,
        int vocab_size,
        int top_k,
        float *out_logits,
        int *out_ids)
    {
        return ISA_DISPATCH_RETVAL(select_topk, logits, vocab_size, top_k, out_logits, out_ids);
    }

} // namespace llaminar2::cpu_sampling
