/**
 * @file SwiGLUPrimitives.cpp
 * @brief SwiGLU primitives implementation
 * @author GitHub Copilot
 */

#include "SwiGLUPrimitives.h"
#include <cmath>
#include <omp.h>
#include <algorithm>
#include "../../../tensors/SIMDHelpers.h"
#include "../../../utils/OpenMPUtils.h"

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#include "../../../utils/CPUFeatures.h"

// Libmvec declarations for vectorized exp
#if defined(__GLIBC__)
extern "C"
{
#if defined(__AVX512F__)
    __m512 _ZGVeN16v_expf(__m512) __attribute__((weak));
#endif
#if defined(__AVX2__)
    __m256 _ZGVcN8v_expf(__m256) __attribute__((weak));
#endif
}
#endif

#include "../../../tensors/FP16Utils.h"
#include "../../../tensors/BlockStructures.h"

namespace llaminar2::primitives
{

    // Scalar implementation
    inline float silu_scalar(float x)
    {
        return x / (1.0f + std::exp(-x));
    }

    static void compute_swiglu_scalar(const float *gate, const float *up, float *output, int size)
    {
        for (int i = 0; i < size; ++i)
        {
            output[i] = silu_scalar(gate[i]) * up[i];
        }
    }

#if defined(__AVX2__)
    // Fast exp approximation (AVX2)
    // Based on Schraudolph/Polynomial approximation
    static inline __m256 fast_exp256(__m256 x)
    {
        __m256 log2e = _mm256_set1_ps(1.4426950408f);
        __m256 ln2 = _mm256_set1_ps(0.69314718056f);
        __m256 max_input = _mm256_set1_ps(88.0f);
        __m256 min_input = _mm256_set1_ps(-88.0f);

        // Clamp input to avoid overflow/underflow
        x = _mm256_min_ps(x, max_input);
        x = _mm256_max_ps(x, min_input);

        // k = round(x * log2(e))
        __m256 k_f = _mm256_round_ps(_mm256_mul_ps(x, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256i k_i = _mm256_cvtps_epi32(k_f);

        // r = x - k * ln2
        __m256 r = _mm256_fnmadd_ps(k_f, ln2, x); // x - k*ln2

        // Polynomial approximation for e^r on [-0.5, 0.5]
        // p(r) = 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120
        __m256 c2 = _mm256_set1_ps(0.5f);
        __m256 c3 = _mm256_set1_ps(0.166666667f);
        __m256 c4 = _mm256_set1_ps(0.041666667f);
        __m256 c5 = _mm256_set1_ps(0.008333333f);
        __m256 one = _mm256_set1_ps(1.0f);

        __m256 p = _mm256_fmadd_ps(c5, r, c4);
        p = _mm256_fmadd_ps(p, r, c3);
        p = _mm256_fmadd_ps(p, r, c2);
        p = _mm256_fmadd_ps(p, r, one);
        p = _mm256_fmadd_ps(p, r, one);

        // 2^k
        __m256i bias = _mm256_set1_epi32(127);
        __m256i k_biased = _mm256_add_epi32(k_i, bias);
        __m256i k_shifted = _mm256_slli_epi32(k_biased, 23);
        __m256 two_k = _mm256_castsi256_ps(k_shifted);

        return _mm256_mul_ps(p, two_k);
    }

    static void compute_swiglu_avx2(const float *gate, const float *up, float *output, int size)
    {
        int i = 0;
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 zero = _mm256_setzero_ps();
        __m256 two = _mm256_set1_ps(2.0f);

        for (; i + 8 <= size; i += 8)
        {
            __m256 g = _mm256_loadu_ps(gate + i);
            __m256 u = _mm256_loadu_ps(up + i);

            // silu(g) = g / (1 + exp(-g))  [applies activation to GATE, per HuggingFace]
            __m256 neg_g = _mm256_sub_ps(zero, g);
            __m256 exp_neg_g = fast_exp256(neg_g);
            __m256 denom = _mm256_add_ps(one, exp_neg_g);

            // Fast reciprocal with Newton-Raphson
            __m256 rcp = _mm256_rcp_ps(denom);
            __m256 term = _mm256_fnmadd_ps(denom, rcp, two); // 2 - d*rcp
            __m256 sigmoid_g = _mm256_mul_ps(rcp, term);

            __m256 silu_g = _mm256_mul_ps(g, sigmoid_g);

            // output = silu(g) * u  [HuggingFace FFN formula: act_fn(gate_proj) * up_proj]
            __m256 out = _mm256_mul_ps(silu_g, u);
            _mm256_storeu_ps(output + i, out);
        }

        // Tail
        for (; i < size; ++i)
        {
            float g = gate[i];
            float u = up[i];
            output[i] = silu_scalar(g) * u;
        }
    }
#endif

#if defined(__AVX512F__)
    static inline __m512 fast_exp512(__m512 x)
    {
        __m512 log2e = _mm512_set1_ps(1.4426950408f);
        __m512 ln2 = _mm512_set1_ps(0.69314718056f);
        __m512 max_input = _mm512_set1_ps(88.0f);
        __m512 min_input = _mm512_set1_ps(-88.0f);

        x = _mm512_min_ps(x, max_input);
        x = _mm512_max_ps(x, min_input);

        __m512 k_f = _mm512_roundscale_ps(_mm512_mul_ps(x, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m512i k_i = _mm512_cvtps_epi32(k_f);

        __m512 r = _mm512_fnmadd_ps(k_f, ln2, x);

        __m512 c2 = _mm512_set1_ps(0.5f);
        __m512 c3 = _mm512_set1_ps(0.166666667f);
        __m512 c4 = _mm512_set1_ps(0.041666667f);
        __m512 c5 = _mm512_set1_ps(0.008333333f);
        __m512 one = _mm512_set1_ps(1.0f);

        __m512 p = _mm512_fmadd_ps(c5, r, c4);
        p = _mm512_fmadd_ps(p, r, c3);
        p = _mm512_fmadd_ps(p, r, c2);
        p = _mm512_fmadd_ps(p, r, one);
        p = _mm512_fmadd_ps(p, r, one);

        __m512i bias = _mm512_set1_epi32(127);
        __m512i k_biased = _mm512_add_epi32(k_i, bias);
        __m512i k_shifted = _mm512_slli_epi32(k_biased, 23);
        __m512 two_k = _mm512_castsi512_ps(k_shifted);

        return _mm512_mul_ps(p, two_k);
    }

    static void compute_swiglu_avx512(const float *gate, const float *up, float *output, int size)
    {
        int i = 0;
        __m512 one = _mm512_set1_ps(1.0f);
        __m512 zero = _mm512_setzero_ps();
        __m512 two = _mm512_set1_ps(2.0f);

        for (; i + 16 <= size; i += 16)
        {
            __m512 g = _mm512_loadu_ps(gate + i);
            __m512 u = _mm512_loadu_ps(up + i);

            // silu(g) = g / (1 + exp(-g))  [applies activation to GATE, per HuggingFace]
            __m512 neg_g = _mm512_sub_ps(zero, g);
            __m512 exp_neg_g = fast_exp512(neg_g);
            __m512 denom = _mm512_add_ps(one, exp_neg_g);

            __m512 rcp = _mm512_rcp14_ps(denom); // rcp14 is more accurate than rcp
            // Newton-Raphson for better precision (optional, rcp14 might be enough)
            // x1 = x0 * (2 - d * x0)
            __m512 term = _mm512_fnmadd_ps(denom, rcp, two);
            __m512 sigmoid_g = _mm512_mul_ps(rcp, term);

            __m512 silu_g = _mm512_mul_ps(g, sigmoid_g);

            // output = silu(g) * u  [HuggingFace FFN formula: act_fn(gate_proj) * up_proj]
            __m512 out = _mm512_mul_ps(silu_g, u);
            _mm512_storeu_ps(output + i, out);
        }

        for (; i < size; ++i)
        {
            float g = gate[i];
            float u = up[i];
            output[i] = silu_scalar(g) * u; // silu on gate, per HuggingFace
        }
    }
#endif

// Stubs for portability when ISA unavailable at compile time
#if !defined(__AVX2__)
    static void compute_swiglu_avx2(const float *gate, const float *up, float *output, int size)
    {
        compute_swiglu_scalar(gate, up, output, size);
    }
#endif
#if !defined(__AVX512F__)
    static void compute_swiglu_avx512(const float *gate, const float *up, float *output, int size)
    {
        compute_swiglu_avx2(gate, up, output, size);
    }
#endif

    void compute_swiglu(const float *gate, const float *up, float *output, int size)
    {
        // OpenMP parallelization
        // Use a chunk size that fits in L1/L2 cache and provides enough work
        const int chunk_size = 1024;

        // Lambda for loop body - used by both parallel and worksharing paths
        auto process_chunk = [&](int i)
        {
            int current_chunk = std::min(chunk_size, size - i);
            const float *g_ptr = gate + i;
            const float *u_ptr = up + i;
            float *o_ptr = output + i;

            ISA_DISPATCH_VOID(compute_swiglu, g_ptr, u_ptr, o_ptr, current_chunk);
        };

        // Layer-fusion support: avoid nested parallel region if already parallel
        auto do_work = [&]()
        {
#pragma omp for schedule(static)
            for (int i = 0; i < size; i += chunk_size)
            {
                process_chunk(i);
            }
        };
        OMP_WORKSHARE_REGION(do_work);
    }

    void compute_swiglu_bf16(const uint16_t *gate, const uint16_t *up, uint16_t *output, int size)
    {
        // Layer-fusion support: avoid nested parallel region if already parallel
        auto do_work = [&]()
        {
#pragma omp for schedule(static)
            for (int i = 0; i < size; ++i)
            {
                float g = simd::bf16_to_fp32(gate[i]);
                float u = simd::bf16_to_fp32(up[i]);
                output[i] = simd::fp32_to_bf16(silu_scalar(g) * u);
            }
        };
        OMP_WORKSHARE_REGION(do_work);
    }

    void compute_swiglu_fp16(const uint16_t *gate, const uint16_t *up, uint16_t *output, int size)
    {
        // Layer-fusion support: avoid nested parallel region if already parallel
        auto do_work = [&]()
        {
#pragma omp for schedule(static)
            for (int i = 0; i < size; ++i)
            {
                float g = simd::fp16_to_fp32(gate[i]);
                float u = simd::fp16_to_fp32(up[i]);
                output[i] = simd::fp32_to_fp16(silu_scalar(g) * u);
            }
        };
        OMP_WORKSHARE_REGION(do_work);
    }

#if defined(__AVX2__)
    static void compute_swiglu_q8_1_avx2(const void *gate, const void *up, void *output, int size)
    {
        const Q8_1Block *g_blocks = static_cast<const Q8_1Block *>(gate);
        const Q8_1Block *u_blocks = static_cast<const Q8_1Block *>(up);
        Q8_1Block *o_blocks = static_cast<Q8_1Block *>(output);
        int num_blocks = size / Q8_1Block::BLOCK_SIZE;

        // Constants for SIMD operations
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 zero = _mm256_setzero_ps();
        __m256 two = _mm256_set1_ps(2.0f);
        __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        __m256 max_val_127 = _mm256_set1_ps(127.0f);
        __m256 min_val_neg127 = _mm256_set1_ps(-127.0f);
        __m256i perm_idx = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        __m256i offset_128 = _mm256_set1_epi8(-128);

        // Lambda for processing a single block - captures SIMD constants by reference
        auto process_block = [&](int b)
        {
            const Q8_1Block &gb = g_blocks[b];
            const Q8_1Block &ub = u_blocks[b];
            Q8_1Block &ob = o_blocks[b];

            float g_scale_f = simd::fp16_to_fp32(gb.d);
            float u_scale_f = simd::fp16_to_fp32(ub.d);

            __m256 g_scale = _mm256_set1_ps(g_scale_f);
            __m256 u_scale = _mm256_set1_ps(u_scale_f);

            // Load 32 int8s
            __m128i g_lo = _mm_loadu_si128((const __m128i *)gb.qs);
            __m128i g_hi = _mm_loadu_si128((const __m128i *)(gb.qs + 16));
            __m128i u_lo = _mm_loadu_si128((const __m128i *)ub.qs);
            __m128i u_hi = _mm_loadu_si128((const __m128i *)(ub.qs + 16));

            // Convert to floats (4 groups of 8)
            __m256 g0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_lo));
            __m256 g1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_lo, 8)));
            __m256 g2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_hi));
            __m256 g3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_hi, 8)));

            __m256 u0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_lo));
            __m256 u1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_lo, 8)));
            __m256 u2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_hi));
            __m256 u3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_hi, 8)));

            // Apply scales
            g0 = _mm256_mul_ps(g0, g_scale);
            g1 = _mm256_mul_ps(g1, g_scale);
            g2 = _mm256_mul_ps(g2, g_scale);
            g3 = _mm256_mul_ps(g3, g_scale);

            u0 = _mm256_mul_ps(u0, u_scale);
            u1 = _mm256_mul_ps(u1, u_scale);
            u2 = _mm256_mul_ps(u2, u_scale);
            u3 = _mm256_mul_ps(u3, u_scale);

            // Compute SwiGLU: silu(g) * u  [HuggingFace FFN formula: act_fn(gate_proj) * up_proj]
            auto compute_silu_op = [&](__m256 val)
            {
                __m256 neg_val = _mm256_sub_ps(zero, val);
                __m256 exp_neg_val = fast_exp256(neg_val);
                __m256 denom = _mm256_add_ps(one, exp_neg_val);
                __m256 rcp = _mm256_rcp_ps(denom);
                __m256 term = _mm256_fnmadd_ps(denom, rcp, two);
                __m256 sigmoid = _mm256_mul_ps(rcp, term);
                return _mm256_mul_ps(val, sigmoid);
            };

            __m256 res0 = _mm256_mul_ps(compute_silu_op(g0), u0);
            __m256 res1 = _mm256_mul_ps(compute_silu_op(g1), u1);
            __m256 res2 = _mm256_mul_ps(compute_silu_op(g2), u2);
            __m256 res3 = _mm256_mul_ps(compute_silu_op(g3), u3);

            // Max abs
            __m256 max_v = _mm256_and_ps(res0, abs_mask);
            max_v = _mm256_max_ps(max_v, _mm256_and_ps(res1, abs_mask));
            max_v = _mm256_max_ps(max_v, _mm256_and_ps(res2, abs_mask));
            max_v = _mm256_max_ps(max_v, _mm256_and_ps(res3, abs_mask));

            // Horizontal max
            __m256 perm = _mm256_permute_ps(max_v, _MM_SHUFFLE(2, 3, 0, 1));
            max_v = _mm256_max_ps(max_v, perm);
            perm = _mm256_permute_ps(max_v, _MM_SHUFFLE(1, 0, 3, 2));
            max_v = _mm256_max_ps(max_v, perm);
            __m128 max_128 = _mm256_castps256_ps128(max_v);
            __m128 max_high = _mm256_extractf128_ps(max_v, 1);
            max_128 = _mm_max_ps(max_128, max_high);
            float max_abs = _mm_cvtss_f32(max_128);

            // Requantize
            float d = max_abs / 127.0f;
            ob.d = simd::fp32_to_fp16(d);
            float inv_d_f = (d > 1e-10f) ? 1.0f / d : 0.0f;
            __m256 inv_d = _mm256_set1_ps(inv_d_f);

            auto quantize_op = [&](__m256 val)
            {
                val = _mm256_mul_ps(val, inv_d);
                val = _mm256_max_ps(min_val_neg127, _mm256_min_ps(max_val_127, val));
                return _mm256_cvtps_epi32(_mm256_round_ps(val, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            };

            __m256i q0 = quantize_op(res0);
            __m256i q1 = quantize_op(res1);
            __m256i q2 = quantize_op(res2);
            __m256i q3 = quantize_op(res3);

            // Pack back to int8
            __m256i p0 = _mm256_packs_epi32(q0, q1);
            __m256i p1 = _mm256_packs_epi32(q2, q3);
            __m256i packed_bytes = _mm256_packs_epi16(p0, p1);
            __m256i result = _mm256_permutevar8x32_epi32(packed_bytes, perm_idx);

            _mm256_storeu_si256((__m256i *)ob.qs, result);

            // Compute sum
            __m256i u_shifted = _mm256_add_epi8(result, offset_128);
            __m256i sums = _mm256_sad_epu8(u_shifted, _mm256_setzero_si256());

            int64_t sum_val = _mm256_extract_epi64(sums, 0) + _mm256_extract_epi64(sums, 1) +
                              _mm256_extract_epi64(sums, 2) + _mm256_extract_epi64(sums, 3);
            ob.sum_qs = (int16_t)(sum_val - 4096);
        };

        // Layer-fusion support: use OMP_WORKSHARE_REGION for nested-safe parallelization
        auto do_work = [&]()
        {
#pragma omp for schedule(static)
            for (int b = 0; b < num_blocks; ++b)
            {
                process_block(b);
            }
        };
        OMP_WORKSHARE_REGION(do_work);
    }
#endif

#if defined(__AVX512F__)
    static void compute_swiglu_q8_1_avx512(const void *gate, const void *up, void *output, int size)
    {
        const Q8_1Block *g_blocks = static_cast<const Q8_1Block *>(gate);
        const Q8_1Block *u_blocks = static_cast<const Q8_1Block *>(up);
        Q8_1Block *o_blocks = static_cast<Q8_1Block *>(output);
        int num_blocks = size / Q8_1Block::BLOCK_SIZE;

        __m512 one = _mm512_set1_ps(1.0f);
        __m512 zero = _mm512_setzero_ps();
        __m512 two = _mm512_set1_ps(2.0f);
        __m512 max_val_127 = _mm512_set1_ps(127.0f);
        __m512 min_val_neg127 = _mm512_set1_ps(-127.0f);

        int b = 0;

        // Process 2 blocks at a time to expose ILP (4 vectors in flight)
        // AVX512 has 32 registers, so we can easily handle this without spilling.
        for (; b < num_blocks - 1; b += 2)
        {
            // --- Block 0 ---
            const Q8_1Block &gb0 = g_blocks[b];
            const Q8_1Block &ub0 = u_blocks[b];
            Q8_1Block &ob0 = o_blocks[b];

            // --- Block 1 ---
            const Q8_1Block &gb1 = g_blocks[b + 1];
            const Q8_1Block &ub1 = u_blocks[b + 1];
            Q8_1Block &ob1 = o_blocks[b + 1];

            // Load scales
            __m512 g_scale0 = _mm512_set1_ps(simd::fp16_to_fp32(gb0.d));
            __m512 u_scale0 = _mm512_set1_ps(simd::fp16_to_fp32(ub0.d));
            __m512 g_scale1 = _mm512_set1_ps(simd::fp16_to_fp32(gb1.d));
            __m512 u_scale1 = _mm512_set1_ps(simd::fp16_to_fp32(ub1.d));

            // Load Block 0 (Low/High)
            __m128i bytes0_lo = _mm_loadu_si128((const __m128i *)gb0.qs);
            __m512 g0_lo = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(bytes0_lo));
            bytes0_lo = _mm_loadu_si128((const __m128i *)ub0.qs);
            __m512 u0_lo = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(bytes0_lo));

            __m128i bytes0_hi = _mm_loadu_si128((const __m128i *)(gb0.qs + 16));
            __m512 g0_hi = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(bytes0_hi));
            bytes0_hi = _mm_loadu_si128((const __m128i *)(ub0.qs + 16));
            __m512 u0_hi = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(bytes0_hi));

            // Load Block 1 (Low/High)
            __m128i bytes1_lo = _mm_loadu_si128((const __m128i *)gb1.qs);
            __m512 g1_lo = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(bytes1_lo));
            bytes1_lo = _mm_loadu_si128((const __m128i *)ub1.qs);
            __m512 u1_lo = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(bytes1_lo));

            __m128i bytes1_hi = _mm_loadu_si128((const __m128i *)(gb1.qs + 16));
            __m512 g1_hi = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(bytes1_hi));
            bytes1_hi = _mm_loadu_si128((const __m128i *)(ub1.qs + 16));
            __m512 u1_hi = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(bytes1_hi));

            // Apply scales
            g0_lo = _mm512_mul_ps(g0_lo, g_scale0);
            g0_hi = _mm512_mul_ps(g0_hi, g_scale0);
            u0_lo = _mm512_mul_ps(u0_lo, u_scale0);
            u0_hi = _mm512_mul_ps(u0_hi, u_scale0);

            g1_lo = _mm512_mul_ps(g1_lo, g_scale1);
            g1_hi = _mm512_mul_ps(g1_hi, g_scale1);
            u1_lo = _mm512_mul_ps(u1_lo, u_scale1);
            u1_hi = _mm512_mul_ps(u1_hi, u_scale1);

            // Compute SwiGLU (lambda for reuse)
            // HuggingFace FFN formula: act_fn(gate_proj) * up_proj = silu(g) * u
            auto swiglu_op = [&](__m512 g, __m512 u)
            {
                __m512 neg_g = _mm512_sub_ps(zero, g);
                __m512 exp_neg_g = fast_exp512(neg_g);
                __m512 denom = _mm512_add_ps(one, exp_neg_g);
                __m512 rcp = _mm512_rcp14_ps(denom);
                __m512 term = _mm512_fnmadd_ps(denom, rcp, two);
                __m512 sigmoid_g = _mm512_mul_ps(rcp, term);
                __m512 silu_g = _mm512_mul_ps(g, sigmoid_g);
                return _mm512_mul_ps(silu_g, u);
            };

            __m512 res0_lo = swiglu_op(g0_lo, u0_lo);
            __m512 res0_hi = swiglu_op(g0_hi, u0_hi);
            __m512 res1_lo = swiglu_op(g1_lo, u1_lo);
            __m512 res1_hi = swiglu_op(g1_hi, u1_hi);

            // Max abs (reduce locally first)
            __m512 abs0 = _mm512_max_ps(_mm512_abs_ps(res0_lo), _mm512_abs_ps(res0_hi));
            float max_abs0 = _mm512_reduce_max_ps(abs0);

            __m512 abs1 = _mm512_max_ps(_mm512_abs_ps(res1_lo), _mm512_abs_ps(res1_hi));
            float max_abs1 = _mm512_reduce_max_ps(abs1);

            // Requantize Block 0
            {
                float d = max_abs0 / 127.0f;
                ob0.d = simd::fp32_to_fp16(d);
                float inv_d_f = (d > 1e-10f) ? 1.0f / d : 0.0f;
                __m512 inv_d = _mm512_set1_ps(inv_d_f);

                auto quant_op = [&](__m512 val)
                {
                    val = _mm512_mul_ps(val, inv_d);
                    val = _mm512_max_ps(min_val_neg127, _mm512_min_ps(max_val_127, val));
                    return _mm512_cvtps_epi32(_mm512_roundscale_ps(val, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                };

                __m512i q_lo = quant_op(res0_lo);
                __m512i q_hi = quant_op(res0_hi);

                _mm512_mask_cvtsepi32_storeu_epi8(ob0.qs, 0xFFFF, q_lo);
                _mm512_mask_cvtsepi32_storeu_epi8(ob0.qs + 16, 0xFFFF, q_hi);

                int32_t sum = _mm512_reduce_add_epi32(q_lo) + _mm512_reduce_add_epi32(q_hi);
                ob0.sum_qs = (int16_t)sum;
            }

            // Requantize Block 1
            {
                float d = max_abs1 / 127.0f;
                ob1.d = simd::fp32_to_fp16(d);
                float inv_d_f = (d > 1e-10f) ? 1.0f / d : 0.0f;
                __m512 inv_d = _mm512_set1_ps(inv_d_f);

                auto quant_op = [&](__m512 val)
                {
                    val = _mm512_mul_ps(val, inv_d);
                    val = _mm512_max_ps(min_val_neg127, _mm512_min_ps(max_val_127, val));
                    return _mm512_cvtps_epi32(_mm512_roundscale_ps(val, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                };

                __m512i q_lo = quant_op(res1_lo);
                __m512i q_hi = quant_op(res1_hi);

                _mm512_mask_cvtsepi32_storeu_epi8(ob1.qs, 0xFFFF, q_lo);
                _mm512_mask_cvtsepi32_storeu_epi8(ob1.qs + 16, 0xFFFF, q_hi);

                int32_t sum = _mm512_reduce_add_epi32(q_lo) + _mm512_reduce_add_epi32(q_hi);
                ob1.sum_qs = (int16_t)sum;
            }
        }

        // Tail loop (1 block)
        for (; b < num_blocks; ++b)
        {
            const Q8_1Block &gb = g_blocks[b];
            const Q8_1Block &ub = u_blocks[b];
            Q8_1Block &ob = o_blocks[b];

            float g_scale_f = simd::fp16_to_fp32(gb.d);
            float u_scale_f = simd::fp16_to_fp32(ub.d);

            __m512 g_scale = _mm512_set1_ps(g_scale_f);
            __m512 u_scale = _mm512_set1_ps(u_scale_f);

            // Process 32 elements in 2 chunks of 16
            __m512 res[2];
            float max_abs = 0.0f;

            // Unrolled inner loop
            {
                // Chunk 0
                __m128i bytes0 = _mm_loadu_si128((const __m128i *)gb.qs);
                __m512 g0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(bytes0));
                bytes0 = _mm_loadu_si128((const __m128i *)ub.qs);
                __m512 u0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(bytes0));

                g0 = _mm512_mul_ps(g0, g_scale);
                u0 = _mm512_mul_ps(u0, u_scale);

                // silu(g) * u  [HuggingFace FFN formula: act_fn(gate_proj) * up_proj]
                __m512 neg_g0 = _mm512_sub_ps(zero, g0);
                __m512 exp_neg_g0 = fast_exp512(neg_g0);
                __m512 denom0 = _mm512_add_ps(one, exp_neg_g0);
                __m512 rcp0 = _mm512_rcp14_ps(denom0);
                __m512 term0 = _mm512_fnmadd_ps(denom0, rcp0, two);
                __m512 sigmoid_g0 = _mm512_mul_ps(rcp0, term0);
                __m512 silu_g0 = _mm512_mul_ps(g0, sigmoid_g0);
                res[0] = _mm512_mul_ps(silu_g0, u0);

                // Chunk 1
                __m128i bytes1 = _mm_loadu_si128((const __m128i *)(gb.qs + 16));
                __m512 g1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(bytes1));
                bytes1 = _mm_loadu_si128((const __m128i *)(ub.qs + 16));
                __m512 u1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(bytes1));

                g1 = _mm512_mul_ps(g1, g_scale);
                u1 = _mm512_mul_ps(u1, u_scale);

                // silu(g) * u  [HuggingFace FFN formula: act_fn(gate_proj) * up_proj]
                __m512 neg_g1 = _mm512_sub_ps(zero, g1);
                __m512 exp_neg_g1 = fast_exp512(neg_g1);
                __m512 denom1 = _mm512_add_ps(one, exp_neg_g1);
                __m512 rcp1 = _mm512_rcp14_ps(denom1);
                __m512 term1 = _mm512_fnmadd_ps(denom1, rcp1, two);
                __m512 sigmoid_g1 = _mm512_mul_ps(rcp1, term1);
                __m512 silu_g1 = _mm512_mul_ps(g1, sigmoid_g1);
                res[1] = _mm512_mul_ps(silu_g1, u1);
            }

            // Max abs
            __m512 abs_res = _mm512_max_ps(_mm512_abs_ps(res[0]), _mm512_abs_ps(res[1]));
            max_abs = _mm512_reduce_max_ps(abs_res);

            float d = max_abs / 127.0f;
            ob.d = simd::fp32_to_fp16(d);
            float inv_d_f = (d > 1e-10f) ? 1.0f / d : 0.0f;
            __m512 inv_d = _mm512_set1_ps(inv_d_f);

            int32_t sum = 0;

            // Requantize
            {
                // Chunk 0
                __m512 val0 = _mm512_mul_ps(res[0], inv_d);
                val0 = _mm512_max_ps(min_val_neg127, _mm512_min_ps(max_val_127, val0));
                __m512i q0 = _mm512_cvtps_epi32(_mm512_roundscale_ps(val0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                _mm512_mask_cvtsepi32_storeu_epi8(ob.qs, 0xFFFF, q0);
                sum += _mm512_reduce_add_epi32(q0);

                // Chunk 1
                __m512 val1 = _mm512_mul_ps(res[1], inv_d);
                val1 = _mm512_max_ps(min_val_neg127, _mm512_min_ps(max_val_127, val1));
                __m512i q1 = _mm512_cvtps_epi32(_mm512_roundscale_ps(val1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                _mm512_mask_cvtsepi32_storeu_epi8(ob.qs + 16, 0xFFFF, q1);
                sum += _mm512_reduce_add_epi32(q1);
            }
            ob.sum_qs = (int16_t)sum;
        }
    }
#endif

    void compute_swiglu_q8_1(const void *gate, const void *up, void *output, int size)
    {
#if defined(__AVX512F__)
        compute_swiglu_q8_1_avx512(gate, up, output, size);
#elif defined(__AVX2__)
        compute_swiglu_q8_1_avx2(gate, up, output, size);
#else
        const Q8_1Block *g_blocks = static_cast<const Q8_1Block *>(gate);
        const Q8_1Block *u_blocks = static_cast<const Q8_1Block *>(up);
        Q8_1Block *o_blocks = static_cast<Q8_1Block *>(output);

        // size is total elements. Q8_1Block::BLOCK_SIZE is 32.
        int num_blocks = size / Q8_1Block::BLOCK_SIZE;

        // Lambda for processing a single block
        auto process_block = [&](int b)
        {
            const Q8_1Block &gb = g_blocks[b];
            const Q8_1Block &ub = u_blocks[b];
            Q8_1Block &ob = o_blocks[b];

            float g_scale = simd::fp16_to_fp32(gb.d);
            float u_scale = simd::fp16_to_fp32(ub.d);

            float temp[Q8_1Block::BLOCK_SIZE];
            float max_abs = 0.0f;

            for (int i = 0; i < Q8_1Block::BLOCK_SIZE; ++i)
            {
                float g = g_scale * gb.qs[i];
                float u = u_scale * ub.qs[i];
                float res = silu_scalar(g) * u; // silu on gate, per HuggingFace
                temp[i] = res;
                max_abs = std::max(max_abs, std::abs(res));
            }

            // Requantize
            float d = max_abs / 127.0f;
            ob.d = simd::fp32_to_fp16(d);
            float inv_d = (d > 1e-10f) ? 1.0f / d : 0.0f;
            int32_t sum = 0;

            for (int i = 0; i < Q8_1Block::BLOCK_SIZE; ++i)
            {
                int8_t q = (int8_t)std::round(std::max(-127.0f, std::min(127.0f, temp[i] * inv_d)));
                ob.qs[i] = q;
                sum += q;
            }
            ob.sum_qs = sum;
        };

        // Layer-fusion support: avoid nested parallel region if already parallel
        auto do_work = [&]()
        {
#pragma omp for schedule(static)
            for (int b = 0; b < num_blocks; ++b)
            {
                process_block(b);
            }
        };
        OMP_WORKSHARE_REGION(do_work);
#endif
    }

} // namespace llaminar2::primitives
