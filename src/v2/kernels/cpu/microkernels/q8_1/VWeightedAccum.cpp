/**
 * @file VWeightedAccum.cpp
 * @brief Reference and SIMD implementations of weighted V accumulation
 * @author David Sanftenberg
 */

#include "VWeightedAccum.h"
#include "../../../../tensors/FP16Utils.h"
#include "../../../../utils/CPUFeatures.h"

#include <cstring>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace llaminar::v2::kernels::microkernels
{

    void v_weighted_accum_ref(const VWeightedAccumParams &params)
    {
        const int head_dim = params.num_blocks * 32;

        // Apply correction factor to existing accumulation (if max changed)
        if (params.correction != 1.0f)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                params.context[d] *= params.correction;
            }
        }

        // Add weighted V (dequantized from Q8_1)
        for (int b = 0; b < params.num_blocks; ++b)
        {
            const Q8_1Block &v_block = params.v_blocks[b];
            float d_v = llaminar2::fp16_to_fp32(v_block.d);
            float weighted_scale = params.weight * d_v;

            float *ctx = params.context + b * 32;
            for (int i = 0; i < 32; ++i)
            {
                float v_val = static_cast<float>(v_block.qs[i]) * weighted_scale;
                ctx[i] += v_val;
            }
        }
    }

    void apply_softmax_correction_ref(float *context, float correction, int head_dim)
    {
        for (int d = 0; d < head_dim; ++d)
        {
            context[d] *= correction;
        }
    }

#if defined(__AVX512F__)

    void v_weighted_accum_avx512(const VWeightedAccumParams &params)
    {
        const int head_dim = params.num_blocks * 32;

        // Broadcast correction factor
        __m512 corr = _mm512_set1_ps(params.correction);

        // Apply correction if needed (vectorized)
        if (params.correction != 1.0f)
        {
            for (int d = 0; d < head_dim; d += 16)
            {
                __m512 ctx = _mm512_loadu_ps(params.context + d);
                ctx = _mm512_mul_ps(ctx, corr);
                _mm512_storeu_ps(params.context + d, ctx);
            }
        }

        // Broadcast weight
        __m512 weight = _mm512_set1_ps(params.weight);

        // Process each block
        for (int b = 0; b < params.num_blocks; ++b)
        {
            const Q8_1Block &v_block = params.v_blocks[b];
            float d_v = llaminar2::fp16_to_fp32(v_block.d);

            // Combined weight * scale
            __m512 weighted_scale = _mm512_set1_ps(params.weight * d_v);

            float *ctx = params.context + b * 32;

            // Load 32 int8 values and convert to float
            // Process in two chunks of 16

            // First 16 elements
            __m128i v_i8_lo = _mm_loadu_si128(reinterpret_cast<const __m128i *>(v_block.qs));
            __m512i v_i32_lo = _mm512_cvtepi8_epi32(v_i8_lo);
            __m512 v_fp32_lo = _mm512_cvtepi32_ps(v_i32_lo);

            // Multiply by weighted scale and accumulate
            __m512 ctx_lo = _mm512_loadu_ps(ctx);
            ctx_lo = _mm512_fmadd_ps(v_fp32_lo, weighted_scale, ctx_lo);
            _mm512_storeu_ps(ctx, ctx_lo);

            // Second 16 elements
            __m128i v_i8_hi = _mm_loadu_si128(reinterpret_cast<const __m128i *>(v_block.qs + 16));
            __m512i v_i32_hi = _mm512_cvtepi8_epi32(v_i8_hi);
            __m512 v_fp32_hi = _mm512_cvtepi32_ps(v_i32_hi);

            __m512 ctx_hi = _mm512_loadu_ps(ctx + 16);
            ctx_hi = _mm512_fmadd_ps(v_fp32_hi, weighted_scale, ctx_hi);
            _mm512_storeu_ps(ctx + 16, ctx_hi);
        }
    }

    void apply_softmax_correction_avx512(float *context, float correction, int head_dim)
    {
        __m512 corr = _mm512_set1_ps(correction);

        int d = 0;
        // Vectorized loop (16 floats at a time)
        for (; d + 16 <= head_dim; d += 16)
        {
            __m512 ctx = _mm512_loadu_ps(context + d);
            ctx = _mm512_mul_ps(ctx, corr);
            _mm512_storeu_ps(context + d, ctx);
        }

        // Scalar tail
        for (; d < head_dim; ++d)
        {
            context[d] *= correction;
        }
    }

#else

    void v_weighted_accum_avx512(const VWeightedAccumParams &params)
    {
        v_weighted_accum_ref(params);
    }

    void apply_softmax_correction_avx512(float *context, float correction, int head_dim)
    {
        apply_softmax_correction_ref(context, correction, head_dim);
    }

#endif

    void v_weighted_accum(const VWeightedAccumParams &params)
    {
#if defined(__AVX512F__)
        if (llaminar2::cpu_supports_avx512())
        {
            v_weighted_accum_avx512(params);
            return;
        }
#endif
        v_weighted_accum_ref(params);
    }

} // namespace llaminar::v2::kernels::microkernels
