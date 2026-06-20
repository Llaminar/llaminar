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
#include <algorithm>
#include <vector>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#include "../simd/AVX2Helpers.h"

namespace llaminar2
{
#ifdef __AVX512F__
    static inline __m512 avx512_silu(__m512 vx);
#endif

    /**
     * @brief Compute one causal short-conv decode channel and publish its state.
     *
     * The grouped verifier path and ordinary one-token decode both call this
     * helper so they share the exact same scalar accumulation and state-update
     * order.  Qwen3.6 GDN layers are sensitive enough that a few ULPs in the
     * short-conv output can be amplified by later recurrent/projection layers,
     * so verifier chunks must not carry a near-duplicate arithmetic sequence.
     */
    static inline float shortconv_decode_sum_and_update_channel(
        const float *input_row,
        const float *weight,
        const float *bias,
        float *conv_state,
        int channel,
        int kernel_size)
    {
        const int state_len = kernel_size - 1;
        const float *w = weight + static_cast<size_t>(channel) * kernel_size;
        float *state = conv_state + static_cast<size_t>(channel) * state_len;
        const float current_input = input_row[channel];

        float sum = bias ? bias[channel] : 0.0f;
        for (int k = 0; k < state_len; ++k)
            sum += w[k] * state[k];
        sum += w[state_len] * current_input;

        for (int k = 0; k < state_len - 1; ++k)
            state[k] = state[k + 1];
        if (state_len > 0)
            state[state_len - 1] = current_input;

        return sum;
    }

    /**
     * @brief Store a block of decode sums through the same SiLU path everywhere.
     *
     * The helper keeps grouped verifier rows concurrent over channel blocks but
     * makes their publication numerically identical to serial decode.  Full
     * AVX512/AVX2 blocks use the existing vector approximations; tails use the
     * scalar reference form used by ordinary decode tails.
     */
    static inline void shortconv_store_decode_block(
        const float *sums,
        float *output,
        int width,
        bool apply_silu)
    {
        if (!apply_silu)
        {
            for (int lane = 0; lane < width; ++lane)
                output[lane] = sums[lane];
            return;
        }

#if defined(__AVX512F__)
        if (width == 16)
        {
            __m512 vsum = _mm512_load_ps(sums);
            vsum = avx512_silu(vsum);
            _mm512_storeu_ps(output, vsum);
            return;
        }
#endif
#if defined(__AVX2__)
        if (width == 8)
        {
            __m256 vsum = _mm256_load_ps(sums);
            vsum = avx2::fast_silu(vsum);
            _mm256_storeu_ps(output, vsum);
            return;
        }
#endif

        for (int lane = 0; lane < width; ++lane)
        {
            const float sum = sums[lane];
            output[lane] = sum / (1.0f + std::exp(-sum));
        }
    }

    void CPUShortConvolution::bindVerifierStateCaptureWorkspace(float *workspace, int rows, int state_size)
    {
        verifier_state_capture_ = workspace;
        verifier_state_capture_rows_ = rows;
        verifier_state_capture_size_ = state_size;
    }

    void CPUShortConvolution::bindSpeculativeStateWorkspace(float *workspace, int state_size)
    {
        speculative_state_work_ = workspace;
        speculative_state_work_size_ = state_size;
    }

    bool CPUShortConvolution::restoreVerifierStateCaptureRow(float *dst_state, int row, void *stream)
    {
        if (row < 0 || row >= verifier_state_capture_rows_)
            return false;
        return restoreStateFromSnapshot(
            dst_state,
            verifier_state_capture_,
            row,
            verifier_state_capture_size_,
            verifier_state_capture_size_,
            stream);
    }

    float *CPUShortConvolution::prepareSpeculativeState(float *live_state, int state_floats)
    {
        if (!live_state || state_floats <= 0)
            return nullptr;

        float *work = nullptr;
        if (speculative_state_work_ && speculative_state_work_size_ >= state_floats)
        {
            work = speculative_state_work_;
        }
        else
        {
            owned_speculative_state_work_.resize(static_cast<size_t>(state_floats));
            work = owned_speculative_state_work_.data();
            speculative_state_work_size_ = std::max(speculative_state_work_size_, state_floats);
        }

        std::memcpy(work, live_state, static_cast<size_t>(state_floats) * sizeof(float));
        return work;
    }

    bool CPUShortConvolution::forward(
        const float *input, const float *weight, const float *bias,
        float *output, float *conv_state,
        int seq_len, int channels, int kernel_size,
        bool apply_silu)
    {
        const int state_len = kernel_size - 1;
        const int state_floats = channels * std::max(0, state_len);
        const bool grouped_verifier_capture_active =
            seq_len > 1 &&
            verifier_state_capture_ &&
            verifier_state_capture_rows_ > 0 &&
            verifier_state_capture_size_ >= state_floats &&
            state_floats > 0;
        if (grouped_verifier_capture_active)
        {
            /*
             * Capture slots mean "run this multi-row verifier chunk
             * speculatively."  They must not change ordinary one-token decode:
             * serial decode is the live-state oracle and must mutate
             * conv_state in-place even when a verifier graph previously bound
             * capture workspace on this shared kernel object.
             */
            float *speculative_state = prepareSpeculativeState(conv_state, state_floats);
            if (!speculative_state)
                return false;
            return forwardWithStateSnapshots(
                input, weight, bias, output, speculative_state,
                seq_len, channels, kernel_size,
                verifier_state_capture_,
                verifier_state_capture_size_,
                verifier_state_capture_rows_,
                apply_silu);
        }

        if (seq_len == 1)
        {
            return executeDecode(input, weight, bias, output, conv_state,
                                 channels, kernel_size, apply_silu);
        }
        else
        {
            const int state_len = kernel_size - 1;
            const int state_floats = channels * state_len;
            if (verifier_state_capture_ &&
                verifier_state_capture_rows_ > 0 &&
                verifier_state_capture_size_ >= state_floats)
            {
                return forwardWithStateSnapshots(
                    input, weight, bias,
                    output, conv_state,
                    seq_len, channels, kernel_size,
                    verifier_state_capture_,
                    verifier_state_capture_size_,
                    verifier_state_capture_rows_,
                    apply_silu);
            }

            const bool ok = executePrefillPreservingInPlaceTail(
                input, weight, bias, output, conv_state,
                seq_len, channels, kernel_size, apply_silu);
            return ok;
        }
    }

    bool CPUShortConvolution::forwardWithStateSnapshots(
        const float *input, const float *weight, const float *bias,
        float *output, float *conv_state,
        int seq_len, int channels, int kernel_size,
        float *state_snapshots, int snapshot_stride_floats,
        int max_snapshot_rows,
        bool apply_silu)
    {
        if (!conv_state || !state_snapshots || seq_len <= 0 || channels <= 0 || kernel_size <= 0)
            return false;

        const int state_len = kernel_size - 1;
        const int state_floats = channels * state_len;
        if (state_len <= 0 || snapshot_stride_floats < state_floats || max_snapshot_rows <= 0)
            return false;

        std::vector<float> raw_input_copy;
        const float *raw_input = input;
        if (input == output)
        {
            raw_input_copy.assign(input, input + static_cast<size_t>(seq_len) * channels);
            raw_input = raw_input_copy.data();
        }

        /**
         * Publishable MTP verifier rows must be decode-equivalent, but calling
         * the one-token decode helper once per row repeatedly opens OpenMP
         * worksharing regions and is the wrong shape for M=2..4 verifier work.
         *
         * The causal dependency is per channel only.  This grouped path assigns
         * channel blocks to worker threads and then walks all verifier rows for
         * that block through the same mutable-state update order as ordinary
         * decode: compute from the current state, shift the state, append the
         * current input row, apply the ISA-specific SiLU path, then snapshot the
         * post-row state.  Keeping the mutable update sequence matters because
         * Qwen3.6 GDN layers can amplify even sub-ULP differences across later
         * attention and FFN stages.
         */
        int channel_block_width = 1;
#if defined(__AVX512F__)
        if (activeISALevel() == ISALevel::AVX512)
        {
            channel_block_width = 16;
        }
        else
#endif
#if defined(__AVX2__)
        if (activeISALevel() == ISALevel::AVX2)
        {
            channel_block_width = 8;
        }
#endif

        auto grouped_decode_equivalent = [&]()
        {
#pragma omp for schedule(static)
            for (int block = 0; block < (channels + channel_block_width - 1) / channel_block_width; ++block)
            {
                const int c_start = block * channel_block_width;
                const int c_width = std::min(channel_block_width, channels - c_start);
                alignas(64) float sums[16];

                for (int t = 0; t < seq_len; ++t)
                {
                    const float *input_row =
                        raw_input + static_cast<size_t>(t) * channels;
                    for (int ci = 0; ci < c_width; ++ci)
                    {
                        const int c = c_start + ci;
                        sums[ci] = shortconv_decode_sum_and_update_channel(
                            input_row, weight, bias, conv_state, c, kernel_size);
                    }

                    float *output_row =
                        output + static_cast<size_t>(t) * channels + c_start;
                    shortconv_store_decode_block(
                        sums, output_row, c_width, apply_silu);

                    if (t < max_snapshot_rows)
                    {
                        for (int ci = 0; ci < c_width; ++ci)
                        {
                            const int c = c_start + ci;
                            const float *state = conv_state + static_cast<size_t>(c) * state_len;
                            float *snapshot =
                                state_snapshots + static_cast<size_t>(t) * snapshot_stride_floats +
                                static_cast<size_t>(c) * state_len;
                            std::memcpy(snapshot, state, static_cast<size_t>(state_len) * sizeof(float));
                        }
                    }
                }
            }
        };
        OMP_WORKSHARE_REGION(grouped_decode_equivalent);
        return true;
    }

    bool CPUShortConvolution::executePrefillPreservingInPlaceTail(
        const float *input, const float *weight, const float *bias,
        float *output, float *conv_state,
        int seq_len, int channels, int kernel_size,
        bool apply_silu)
    {
        const int state_len = kernel_size - 1;
        const bool in_place_with_state =
            input == output && conv_state && state_len > 0;
        std::vector<float> raw_tail;
        if (in_place_with_state)
        {
            raw_tail.resize(static_cast<size_t>(channels) * state_len);
            for (int c = 0; c < channels; ++c)
            {
                for (int s = 0; s < state_len; ++s)
                {
                    const int src_t = seq_len - state_len + s;
                    raw_tail[static_cast<size_t>(c) * state_len + s] =
                        (src_t >= 0) ? input[static_cast<size_t>(src_t) * channels + c]
                                     : conv_state[c * state_len + state_len + src_t];
                }
            }
        }

        const bool ok = executePrefill(input, weight, bias, output, conv_state,
                                       seq_len, channels, kernel_size, apply_silu);
        if (ok && in_place_with_state)
        {
            std::memcpy(conv_state,
                        raw_tail.data(),
                        raw_tail.size() * sizeof(float));
        }
        return ok;
    }

    bool CPUShortConvolution::restoreStateFromSnapshot(
        float *state, const float *state_snapshots,
        int snapshot_row, int snapshot_stride_floats,
        int state_floats, void *stream)
    {
        (void)stream;
        if (!state || !state_snapshots || snapshot_row < 0 ||
            snapshot_stride_floats < state_floats || state_floats < 0)
            return false;

        std::memcpy(state,
                    state_snapshots + static_cast<size_t>(snapshot_row) * snapshot_stride_floats,
                    static_cast<size_t>(state_floats) * sizeof(float));
        return true;
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

                for (int t = seq_len - 1; t >= 0; --t)
                {
                    float sum = b;
                    for (int k = 0; k < kernel_size; ++k)
                    {
                        const int input_t = t - state_len + k;
                        if (input_t >= 0)
                            sum += w[k] * input[input_t * channels + c];
                        else if (conv_state)
                            sum += w[k] * conv_state[c * state_len + state_len + input_t];
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

                if (conv_state)
                {
                    float *state = conv_state + c * state_len;
                    for (int s = 0; s < state_len; ++s)
                    {
                        const int src_t = seq_len - state_len + s;
                        state[s] = (src_t >= 0) ? input[src_t * channels + c]
                                                 : state[state_len + src_t];
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
                            else if (conv_state)
                            {
                                alignas(32) float hist[8];
                                const int state_idx = state_len + input_t;
                                for (int lane = 0; lane < 8; ++lane)
                                    hist[lane] = conv_state[(c_start + lane) * state_len + state_idx];
                                __m256 vh = _mm256_load_ps(hist);
                                vsum = _mm256_fmadd_ps(vw[k], vh, vsum);
                            }
                        }

                        if (apply_silu)
                            vsum = avx2::fast_silu(vsum);

                        _mm256_storeu_ps(&output[t * channels + c_start], vsum);
                    }

                    if (conv_state)
                    {
                        for (int ci = 0; ci < c_width; ++ci)
                        {
                            const int c = c_start + ci;
                            float *state = conv_state + c * state_len;
                            for (int s = 0; s < state_len; ++s)
                            {
                                const int src_t = seq_len - state_len + s;
                                state[s] = (src_t >= 0) ? input[src_t * channels + c]
                                                         : state[state_len + src_t];
                            }
                        }
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
                                else if (conv_state)
                                    sum += wt[k * channels + c] * conv_state[c * state_len + state_len + input_t];
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

                        if (conv_state)
                        {
                            float *state = conv_state + c * state_len;
                            for (int s = 0; s < state_len; ++s)
                            {
                                const int src_t = seq_len - state_len + s;
                                state[s] = (src_t >= 0) ? input[src_t * channels + c]
                                                         : state[state_len + src_t];
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
                            else if (conv_state)
                            {
                                alignas(64) float hist[16];
                                const int state_idx = state_len + input_t;
                                for (int lane = 0; lane < 16; ++lane)
                                    hist[lane] = conv_state[(c_start + lane) * state_len + state_idx];
                                __m512 vh = _mm512_load_ps(hist);
                                vsum = _mm512_fmadd_ps(vw[k], vh, vsum);
                            }
                        }

                        if (apply_silu)
                            vsum = avx512_silu(vsum);

                        _mm512_storeu_ps(&output[t * channels + c_start], vsum);
                    }

                    if (conv_state)
                    {
                        for (int ci = 0; ci < c_width; ++ci)
                        {
                            const int c = c_start + ci;
                            float *state = conv_state + c * state_len;
                            for (int s = 0; s < state_len; ++s)
                            {
                                const int src_t = seq_len - state_len + s;
                                state[s] = (src_t >= 0) ? input[src_t * channels + c]
                                                         : state[state_len + src_t];
                            }
                        }
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
                                else if (conv_state)
                                    sum += wt[k * channels + c] * conv_state[c * state_len + state_len + input_t];
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

                        if (conv_state)
                        {
                            float *state = conv_state + c * state_len;
                            for (int s = 0; s < state_len; ++s)
                            {
                                const int src_t = seq_len - state_len + s;
                                state[s] = (src_t >= 0) ? input[src_t * channels + c]
                                                         : state[state_len + src_t];
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
        auto do_work = [&]()
        {
#pragma omp for schedule(static)
            for (int c = 0; c < channels; ++c)
            {
                const float sum = shortconv_decode_sum_and_update_channel(
                    input, weight, bias, conv_state, c, kernel_size);
                shortconv_store_decode_block(&sum, output + c, 1, apply_silu);
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
                    sums[ci] = shortconv_decode_sum_and_update_channel(
                        input, weight, bias, conv_state, c, kernel_size);
                }

                shortconv_store_decode_block(
                    sums, output + c_start, c_width, apply_silu);
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
                    sums[ci] = shortconv_decode_sum_and_update_channel(
                        input, weight, bias, conv_state, c, kernel_size);
                }

                shortconv_store_decode_block(
                    sums, output + c_start, c_width, apply_silu);
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
