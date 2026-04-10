/**
 * @file CPUShortConvolution.cpp
 * @brief CPU implementation of causal depthwise conv1d + SiLU
 *
 * Two execution paths:
 *   Prefill (seq_len > 1): Full causal conv1d with zero-padding, stores tail in conv_state
 *   Decode  (seq_len == 1): Conv1d update using conv_state history
 *
 * Prefill is AVX-512 vectorized (16-wide) with pre-transposed weights for
 * contiguous SIMD loads across channels. Decode is AVX-512 vectorized across
 * the channel dimension.
 *
 * Reference: torch_causal_conv1d_update() and F.conv1d() in HuggingFace transformers
 */

#include "CPUShortConvolution.h"
#include "../../../utils/CPUFeatures.h"
#include "../../../utils/OpenMPUtils.h"

#include <cmath>
#include <cstring>
#include <vector>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#include "../simd/AVX2Helpers.h"

namespace llaminar2
{

    bool CPUShortConvolution::forward(
        const float *input, const float *weight, const float *bias,
        float *output, float *conv_state,
        int seq_len, int channels, int kernel_size,
        bool apply_silu)
    {
        if (seq_len == 1)
        {
            return executeDecode(input, weight, bias, output, conv_state,
                                 channels, kernel_size, apply_silu);
        }
        else
        {
            return executePrefill(input, weight, bias, output, conv_state,
                                  seq_len, channels, kernel_size, apply_silu);
        }
    }

#ifdef __AVX512F__
    // Fast SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)) using AVX-512.
    // Uses range-reduced exp polynomial: exp(x) = 2^n * P(f) where n = round(x*log2e).
    static inline __m512 avx512_silu(__m512 vx)
    {
        const __m512 vone = _mm512_set1_ps(1.0f);
        const __m512 vlog2e = _mm512_set1_ps(1.4426950408889634f);
        const __m512 vln2 = _mm512_set1_ps(0.6931471805599453f);

        // Clamp -x to [-88, 88] to avoid overflow in exp
        __m512 neg_x = _mm512_sub_ps(_mm512_setzero_ps(), vx);
        neg_x = _mm512_max_ps(_mm512_set1_ps(-88.0f), _mm512_min_ps(_mm512_set1_ps(88.0f), neg_x));

        // Range reduction: exp(-x) = 2^n * 2^f
        __m512 neg_x_scaled = _mm512_mul_ps(neg_x, vlog2e);
        __m512 vn = _mm512_roundscale_ps(neg_x_scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m512 vf = _mm512_sub_ps(neg_x_scaled, vn);

        // Polynomial approximation of 2^f for f in [-0.5, 0.5]
        __m512 vp = _mm512_fmadd_ps(_mm512_set1_ps(0.0013333558146428f), vf, _mm512_set1_ps(0.0096181291076285f));
        vp = _mm512_fmadd_ps(vp, vf, _mm512_set1_ps(0.0555041086648216f));
        vp = _mm512_fmadd_ps(vp, vf, _mm512_set1_ps(0.2402265069591007f));
        vp = _mm512_fmadd_ps(vp, vf, vln2);
        vp = _mm512_fmadd_ps(vp, vf, vone);

        // Reconstruct 2^n via integer exponent
        __m512i vi_n = _mm512_add_epi32(_mm512_cvtps_epi32(vn), _mm512_set1_epi32(127));
        __m512 v2n = _mm512_castsi512_ps(_mm512_slli_epi32(vi_n, 23));
        __m512 vexp = _mm512_mul_ps(vp, v2n);

        // SiLU = x * sigmoid(x) = x / (1 + exp(-x))
        __m512 vsig = _mm512_div_ps(vone, _mm512_add_ps(vone, vexp));
        return _mm512_mul_ps(vx, vsig);
    }
#endif

    // ========================================================================
    // Named ISA implementations: executePrefill
    // ========================================================================

    static void shortconv_prefill_scalar(
        const float *input, const float *weight, const float *bias,
        float *output, float *conv_state,
        int seq_len, int channels, int kernel_size,
        bool apply_silu)
    {
        const int state_len = kernel_size - 1;

        auto do_work = [&]()
        {
#pragma omp for schedule(static)
            for (int c = 0; c < channels; ++c)
            {
                const float *w = weight + c * kernel_size;
                const float b = bias ? bias[c] : 0.0f;

                if (conv_state)
                {
                    for (int s = 0; s < state_len; ++s)
                    {
                        const int src_t = seq_len - state_len + s;
                        conv_state[c * state_len + s] =
                            (src_t >= 0) ? input[src_t * channels + c] : 0.0f;
                    }
                }

                for (int t = seq_len - 1; t >= 0; --t)
                {
                    float sum = b;
                    for (int k = 0; k < kernel_size; ++k)
                    {
                        const int input_t = t - state_len + k;
                        if (input_t >= 0)
                            sum += w[k] * input[input_t * channels + c];
                    }
                    if (apply_silu)
                    {
                        const float sig = 1.0f / (1.0f + std::exp(-sum));
                        output[t * channels + c] = sum * sig;
                    }
                    else
                    {
                        output[t * channels + c] = sum;
                    }
                }
            }
        };
        OMP_WORKSHARE_REGION(do_work);
    }

#if defined(__AVX2__)
    static void shortconv_prefill_avx2(
        const float *input, const float *weight, const float *bias,
        float *output, float *conv_state,
        int seq_len, int channels, int kernel_size,
        bool apply_silu)
    {
        const int state_len = kernel_size - 1;

        std::vector<float> wt(static_cast<size_t>(kernel_size) * channels);
        for (int k = 0; k < kernel_size; ++k)
            for (int c = 0; c < channels; ++c)
                wt[k * channels + c] = weight[c * kernel_size + k];

        auto do_work = [&]()
        {
            if (conv_state)
            {
#pragma omp for schedule(static) nowait
                for (int c = 0; c < channels; ++c)
                {
                    for (int s = 0; s < state_len; ++s)
                    {
                        const int src_t = seq_len - state_len + s;
                        conv_state[c * state_len + s] =
                            (src_t >= 0) ? input[src_t * channels + c] : 0.0f;
                    }
                }
            }

            const int n_blocks = (channels + 7) / 8;

#pragma omp for schedule(static)
            for (int blk = 0; blk < n_blocks; ++blk)
            {
                const int c_start = blk * 8;
                const int c_width = (c_start + 8 <= channels) ? 8 : (channels - c_start);

                if (c_width == 8)
                {
                    __m256 vbias = bias
                                       ? _mm256_loadu_ps(bias + c_start)
                                       : _mm256_setzero_ps();

                    __m256 vw[8];
                    for (int k = 0; k < kernel_size && k < 8; ++k)
                        vw[k] = _mm256_loadu_ps(&wt[k * channels + c_start]);

                    for (int t = seq_len - 1; t >= 0; --t)
                    {
                        __m256 vsum = vbias;

                        for (int k = 0; k < kernel_size; ++k)
                        {
                            const int input_t = t - state_len + k;
                            if (input_t >= 0)
                            {
                                __m256 vin = _mm256_loadu_ps(&input[input_t * channels + c_start]);
                                vsum = _mm256_fmadd_ps(vw[k], vin, vsum);
                            }
                        }

                        if (apply_silu)
                            vsum = avx2::fast_silu(vsum);

                        _mm256_storeu_ps(&output[t * channels + c_start], vsum);
                    }
                }
                else
                {
                    for (int ci = 0; ci < c_width; ++ci)
                    {
                        const int c = c_start + ci;
                        const float b = bias ? bias[c] : 0.0f;

                        for (int t = seq_len - 1; t >= 0; --t)
                        {
                            float sum = b;
                            for (int k = 0; k < kernel_size; ++k)
                            {
                                const int input_t = t - state_len + k;
                                if (input_t >= 0)
                                    sum += wt[k * channels + c] * input[input_t * channels + c];
                            }
                            if (apply_silu)
                            {
                                const float sig = 1.0f / (1.0f + std::exp(-sum));
                                output[t * channels + c] = sum * sig;
                            }
                            else
                            {
                                output[t * channels + c] = sum;
                            }
                        }
                    }
                }
            }
        };
        OMP_WORKSHARE_REGION(do_work);
    }
#endif

#if defined(__AVX512F__)
    static void shortconv_prefill_avx512(
        const float *input, const float *weight, const float *bias,
        float *output, float *conv_state,
        int seq_len, int channels, int kernel_size,
        bool apply_silu)
    {
        const int state_len = kernel_size - 1;

        std::vector<float> wt(static_cast<size_t>(kernel_size) * channels);
        for (int k = 0; k < kernel_size; ++k)
            for (int c = 0; c < channels; ++c)
                wt[k * channels + c] = weight[c * kernel_size + k];

        auto do_work = [&]()
        {
            if (conv_state)
            {
#pragma omp for schedule(static) nowait
                for (int c = 0; c < channels; ++c)
                {
                    for (int s = 0; s < state_len; ++s)
                    {
                        const int src_t = seq_len - state_len + s;
                        conv_state[c * state_len + s] =
                            (src_t >= 0) ? input[src_t * channels + c] : 0.0f;
                    }
                }
            }

            const int n_blocks = (channels + 15) / 16;

#pragma omp for schedule(static)
            for (int blk = 0; blk < n_blocks; ++blk)
            {
                const int c_start = blk * 16;
                const int c_width = (c_start + 16 <= channels) ? 16 : (channels - c_start);

                if (c_width == 16)
                {
                    __m512 vbias = bias
                                       ? _mm512_loadu_ps(bias + c_start)
                                       : _mm512_setzero_ps();

                    __m512 vw[8];
                    for (int k = 0; k < kernel_size && k < 8; ++k)
                        vw[k] = _mm512_loadu_ps(&wt[k * channels + c_start]);

                    for (int t = seq_len - 1; t >= 0; --t)
                    {
                        __m512 vsum = vbias;

                        for (int k = 0; k < kernel_size; ++k)
                        {
                            const int input_t = t - state_len + k;
                            if (input_t >= 0)
                            {
                                __m512 vin = _mm512_loadu_ps(&input[input_t * channels + c_start]);
                                vsum = _mm512_fmadd_ps(vw[k], vin, vsum);
                            }
                        }

                        if (apply_silu)
                            vsum = avx512_silu(vsum);

                        _mm512_storeu_ps(&output[t * channels + c_start], vsum);
                    }
                }
                else
                {
                    for (int ci = 0; ci < c_width; ++ci)
                    {
                        const int c = c_start + ci;
                        const float b = bias ? bias[c] : 0.0f;

                        for (int t = seq_len - 1; t >= 0; --t)
                        {
                            float sum = b;
                            for (int k = 0; k < kernel_size; ++k)
                            {
                                const int input_t = t - state_len + k;
                                if (input_t >= 0)
                                    sum += wt[k * channels + c] * input[input_t * channels + c];
                            }
                            if (apply_silu)
                            {
                                const float sig = 1.0f / (1.0f + std::exp(-sum));
                                output[t * channels + c] = sum * sig;
                            }
                            else
                            {
                                output[t * channels + c] = sum;
                            }
                        }
                    }
                }
            }
        };
        OMP_WORKSHARE_REGION(do_work);
    }
#endif

// Stubs for when ISA is unavailable at compile time
#if !defined(__AVX2__)
    static void shortconv_prefill_avx2(
        const float *input, const float *weight, const float *bias,
        float *output, float *conv_state,
        int seq_len, int channels, int kernel_size,
        bool apply_silu)
    {
        shortconv_prefill_scalar(input, weight, bias, output, conv_state, seq_len, channels, kernel_size, apply_silu);
    }
#endif
#if !defined(__AVX512F__)
    static void shortconv_prefill_avx512(
        const float *input, const float *weight, const float *bias,
        float *output, float *conv_state,
        int seq_len, int channels, int kernel_size,
        bool apply_silu)
    {
        shortconv_prefill_avx2(input, weight, bias, output, conv_state, seq_len, channels, kernel_size, apply_silu);
    }
#endif

    bool CPUShortConvolution::executePrefill(
        const float *input, const float *weight, const float *bias,
        float *output, float *conv_state,
        int seq_len, int channels, int kernel_size,
        bool apply_silu)
    {
        ISA_DISPATCH_VOID(shortconv_prefill, input, weight, bias, output, conv_state, seq_len, channels, kernel_size, apply_silu);
        return true;
    }

    // ========================================================================
    // Named ISA implementations: executeDecode
    // ========================================================================

    static void shortconv_decode_scalar(
        const float *input, const float *weight, const float *bias,
        float *output, float *conv_state,
        int channels, int kernel_size,
        bool apply_silu)
    {
        const int state_len = kernel_size - 1;

        auto do_work = [&]()
        {
#pragma omp for schedule(static)
            for (int c = 0; c < channels; ++c)
            {
                const float *w = weight + c * kernel_size;
                float *state = conv_state + c * state_len;
                const float b = bias ? bias[c] : 0.0f;

                float sum = b;
                for (int k = 0; k < state_len; ++k)
                    sum += w[k] * state[k];
                sum += w[state_len] * input[c];

                for (int k = 0; k < state_len - 1; ++k)
                    state[k] = state[k + 1];
                if (state_len > 0)
                    state[state_len - 1] = input[c];

                if (apply_silu)
                {
                    const float sig = 1.0f / (1.0f + std::exp(-sum));
                    output[c] = sum * sig;
                }
                else
                {
                    output[c] = sum;
                }
            }
        };
        OMP_WORKSHARE_REGION(do_work);
    }

#if defined(__AVX2__)
    static void shortconv_decode_avx2(
        const float *input, const float *weight, const float *bias,
        float *output, float *conv_state,
        int channels, int kernel_size,
        bool apply_silu)
    {
        const int state_len = kernel_size - 1;

        auto do_work = [&]()
        {
            const int n_blocks = (channels + 7) / 8;

#pragma omp for schedule(static)
            for (int blk = 0; blk < n_blocks; ++blk)
            {
                const int c_start = blk * 8;
                const int c_end = (c_start + 8 <= channels) ? c_start + 8 : channels;
                const int c_width = c_end - c_start;

                alignas(32) float sums[8];
                for (int ci = 0; ci < c_width; ++ci)
                {
                    const int c = c_start + ci;
                    const float *w = weight + c * kernel_size;
                    float *state = conv_state + c * state_len;
                    const float b = bias ? bias[c] : 0.0f;

                    float sum = b;
                    for (int k = 0; k < state_len; ++k)
                        sum += w[k] * state[k];
                    sum += w[state_len] * input[c];

                    for (int k = 0; k < state_len - 1; ++k)
                        state[k] = state[k + 1];
                    if (state_len > 0)
                        state[state_len - 1] = input[c];

                    sums[ci] = sum;
                }

                if (apply_silu && c_width == 8)
                {
                    __m256 vsum = _mm256_load_ps(sums);
                    vsum = avx2::fast_silu(vsum);
                    _mm256_storeu_ps(&output[c_start], vsum);
                }
                else if (apply_silu)
                {
                    for (int ci = 0; ci < c_width; ++ci)
                    {
                        const float s = sums[ci];
                        const float sig = 1.0f / (1.0f + std::exp(-s));
                        output[c_start + ci] = s * sig;
                    }
                }
                else
                {
                    for (int ci = 0; ci < c_width; ++ci)
                        output[c_start + ci] = sums[ci];
                }
            }
        };
        OMP_WORKSHARE_REGION(do_work);
    }
#endif

#if defined(__AVX512F__)
    static void shortconv_decode_avx512(
        const float *input, const float *weight, const float *bias,
        float *output, float *conv_state,
        int channels, int kernel_size,
        bool apply_silu)
    {
        const int state_len = kernel_size - 1;

        auto do_work = [&]()
        {
            const int n_blocks = (channels + 15) / 16;

#pragma omp for schedule(static)
            for (int blk = 0; blk < n_blocks; ++blk)
            {
                const int c_start = blk * 16;
                const int c_end = (c_start + 16 <= channels) ? c_start + 16 : channels;
                const int c_width = c_end - c_start;

                alignas(64) float sums[16];
                for (int ci = 0; ci < c_width; ++ci)
                {
                    const int c = c_start + ci;
                    const float *w = weight + c * kernel_size;
                    float *state = conv_state + c * state_len;
                    const float b = bias ? bias[c] : 0.0f;

                    float sum = b;
                    for (int k = 0; k < state_len; ++k)
                        sum += w[k] * state[k];
                    sum += w[state_len] * input[c];

                    for (int k = 0; k < state_len - 1; ++k)
                        state[k] = state[k + 1];
                    if (state_len > 0)
                        state[state_len - 1] = input[c];

                    sums[ci] = sum;
                }

                if (apply_silu && c_width == 16)
                {
                    __m512 vsum = _mm512_load_ps(sums);
                    vsum = avx512_silu(vsum);
                    _mm512_storeu_ps(&output[c_start], vsum);
                }
                else if (apply_silu)
                {
                    for (int ci = 0; ci < c_width; ++ci)
                    {
                        const float s = sums[ci];
                        const float sig = 1.0f / (1.0f + std::exp(-s));
                        output[c_start + ci] = s * sig;
                    }
                }
                else
                {
                    for (int ci = 0; ci < c_width; ++ci)
                        output[c_start + ci] = sums[ci];
                }
            }
        };
        OMP_WORKSHARE_REGION(do_work);
    }
#endif

// Stubs for when ISA is unavailable at compile time
#if !defined(__AVX2__)
    static void shortconv_decode_avx2(
        const float *input, const float *weight, const float *bias,
        float *output, float *conv_state,
        int channels, int kernel_size,
        bool apply_silu)
    {
        shortconv_decode_scalar(input, weight, bias, output, conv_state, channels, kernel_size, apply_silu);
    }
#endif
#if !defined(__AVX512F__)
    static void shortconv_decode_avx512(
        const float *input, const float *weight, const float *bias,
        float *output, float *conv_state,
        int channels, int kernel_size,
        bool apply_silu)
    {
        shortconv_decode_avx2(input, weight, bias, output, conv_state, channels, kernel_size, apply_silu);
    }
#endif

    bool CPUShortConvolution::executeDecode(
        const float *input, const float *weight, const float *bias,
        float *output, float *conv_state,
        int channels, int kernel_size,
        bool apply_silu)
    {
        if (!conv_state)
            return false;

        ISA_DISPATCH_VOID(shortconv_decode, input, weight, bias, output, conv_state, channels, kernel_size, apply_silu);
        return true;
    }

} // namespace llaminar2
