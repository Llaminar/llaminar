/**
 * @file Test__ROCmGDNPaddedRealLength.cpp
 * @brief ROCm integration coverage for padded GDN real-length semantics.
 *
 * Exercises the real ROCm GatedDeltaNet and short-convolution kernels directly,
 * without model loading or graph orchestration. The tests compare padded bucket
 * prefill with an effective real length against an unpadded reference prefill
 * followed by a decode step, proving that padding rows do not corrupt the GPU
 * recurrence or convolution state carried into decode.
 */

#include <gtest/gtest.h>

#ifdef HAVE_ROCM
#include "kernels/rocm/gdn/ROCmGatedDeltaNet.h"
#include "kernels/rocm/gdn/ROCmShortConvolution.h"
#include <hip/hip_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llaminar2;

namespace
{
#ifdef HAVE_ROCM
    /// @brief Throws with HIP's diagnostic string when a runtime call fails.
    void checkHip(hipError_t status, const char *operation)
    {
        if (status != hipSuccess)
            throw std::runtime_error(std::string(operation) + " failed: " + hipGetErrorString(status));
    }

    /// @brief Returns true when at least one ROCm device is visible to HIP.
    bool hasROCm()
    {
        int count = 0;
        return hipGetDeviceCount(&count) == hipSuccess && count > 0;
    }

    /// @brief RAII wrapper for an FP32 HIP device buffer used by direct kernel calls.
    struct HipFloatBuffer
    {
        float *ptr = nullptr; ///< Device pointer owned by this buffer.
        size_t count = 0;     ///< Number of FP32 elements allocated.

        explicit HipFloatBuffer(size_t n) : count(n)
        {
            if (count > 0)
                checkHip(hipMalloc(reinterpret_cast<void **>(&ptr), count * sizeof(float)), "hipMalloc(float)");
        }

        explicit HipFloatBuffer(const std::vector<float> &host) : HipFloatBuffer(host.size())
        {
            copyFrom(host);
        }

        HipFloatBuffer(size_t n, float value) : HipFloatBuffer(n)
        {
            fill(value);
        }

        ~HipFloatBuffer()
        {
            if (ptr)
                (void)hipFree(ptr);
        }

        HipFloatBuffer(const HipFloatBuffer &) = delete;
        HipFloatBuffer &operator=(const HipFloatBuffer &) = delete;

        /// @brief Copies a host vector into the owned device buffer.
        void copyFrom(const std::vector<float> &host)
        {
            ASSERT_EQ(host.size(), count);
            if (count > 0)
            {
                checkHip(hipMemcpy(ptr, host.data(), count * sizeof(float), hipMemcpyHostToDevice),
                         "hipMemcpy host-to-device(float)");
            }
        }

        /// @brief Fills the buffer through a host staging vector so the exact FP32 value is stored.
        void fill(float value)
        {
            std::vector<float> host(count, value);
            copyFrom(host);
        }

        /// @brief Copies the device buffer back to host memory.
        std::vector<float> toHost() const
        {
            std::vector<float> host(count);
            if (count > 0)
            {
                checkHip(hipMemcpy(host.data(), ptr, count * sizeof(float), hipMemcpyDeviceToHost),
                         "hipMemcpy device-to-host(float)");
            }
            return host;
        }
    };

    /// @brief RAII wrapper for int metadata stored on a HIP device.
    struct HipIntBuffer
    {
        int *ptr = nullptr; ///< Device pointer owned by this buffer.

        explicit HipIntBuffer(int value)
        {
            checkHip(hipMalloc(reinterpret_cast<void **>(&ptr), sizeof(int)), "hipMalloc(int)");
            checkHip(hipMemcpy(ptr, &value, sizeof(int), hipMemcpyHostToDevice), "hipMemcpy host-to-device(int)");
        }

        explicit HipIntBuffer(std::initializer_list<int> values)
        {
            std::vector<int> host(values);
            checkHip(hipMalloc(reinterpret_cast<void **>(&ptr), host.size() * sizeof(int)), "hipMalloc(int[])");
            checkHip(hipMemcpy(ptr, host.data(), host.size() * sizeof(int), hipMemcpyHostToDevice),
                     "hipMemcpy host-to-device(int[])");
        }

        ~HipIntBuffer()
        {
            if (ptr)
                (void)hipFree(ptr);
        }

        HipIntBuffer(const HipIntBuffer &) = delete;
        HipIntBuffer &operator=(const HipIntBuffer &) = delete;
    };

    /// @brief RAII wrapper for a non-blocking HIP stream used by direct kernel tests.
    struct HipStreamHandle
    {
        hipStream_t stream = nullptr; ///< HIP stream owned by this wrapper.

        HipStreamHandle()
        {
            checkHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "hipStreamCreateWithFlags");
        }

        ~HipStreamHandle()
        {
            if (stream)
                (void)hipStreamDestroy(stream);
        }

        HipStreamHandle(const HipStreamHandle &) = delete;
        HipStreamHandle &operator=(const HipStreamHandle &) = delete;
    };

    /**
     * @brief Builds row-major sequence data with hostile padding rows.
     *
     * Rows before real_len are low-magnitude real prompt tokens, rows between
     * real_len and bucket_len are high-magnitude padding, and the last row is
     * the decode token shared by both padded and reference paths.
     */
    std::vector<float> makeSequenceRows(
        int total_len,
        int width,
        int real_len,
        int bucket_len,
        float real_scale,
        float pad_scale,
        float decode_scale)
    {
        std::vector<float> values(static_cast<size_t>(total_len) * static_cast<size_t>(width));

        for (int row = 0; row < total_len; ++row)
        {
            for (int col = 0; col < width; ++col)
            {
                const size_t index = static_cast<size_t>(row) * static_cast<size_t>(width) + static_cast<size_t>(col);
                const float col_wave = static_cast<float>((col % 11) - 5);
                const float row_wave = static_cast<float>((row % 7) + 1);

                if (row < real_len)
                {
                    values[index] = real_scale * row_wave * static_cast<float>((col % 5) + 1) +
                                    0.0007f * col_wave;
                }
                else if (row < bucket_len)
                {
                    const float sign = (col % 2 == 0) ? 1.0f : -1.0f;
                    values[index] = sign * pad_scale * static_cast<float>((col % 3) + 1) +
                                    0.03125f * static_cast<float>(row + 1);
                }
                else
                {
                    values[index] = decode_scale * (0.5f * col_wave + row_wave);
                }
            }
        }

        return values;
    }

    /// @brief Creates deterministic per-channel short-convolution weights.
    std::vector<float> makeShortConvWeights(int channels, int kernel_size)
    {
        std::vector<float> weights(static_cast<size_t>(channels) * static_cast<size_t>(kernel_size));
        for (int channel = 0; channel < channels; ++channel)
        {
            for (int tap = 0; tap < kernel_size; ++tap)
            {
                const size_t index = static_cast<size_t>(channel) * static_cast<size_t>(kernel_size) + static_cast<size_t>(tap);
                weights[index] = 0.0175f * static_cast<float>((channel % 5) - 2) +
                                 0.045f * static_cast<float>(tap + 1);
            }
        }
        return weights;
    }

    /// @brief Creates deterministic bias values for short-convolution tests.
    std::vector<float> makeBias(int count)
    {
        std::vector<float> values(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i)
            values[static_cast<size_t>(i)] = 0.0025f * static_cast<float>((i % 7) - 3);
        return values;
    }

    /// @brief Creates deterministic nonzero recurrent/conv state.
    std::vector<float> makeInitialState(size_t count, float scale)
    {
        std::vector<float> values(count);
        for (size_t i = 0; i < count; ++i)
        {
            const float wave = static_cast<float>(static_cast<int>(i % 23) - 11);
            const float slow = static_cast<float>(static_cast<int>((i / 23) % 17) - 8);
            values[i] = scale * (0.7f * wave + 0.13f * slow);
        }
        return values;
    }

    /// @brief Computes maximum absolute difference and relative L2 over a contiguous span.
    std::pair<float, double> diffStats(
        const std::vector<float> &actual,
        const std::vector<float> &expected,
        size_t offset,
        size_t count)
    {
        float max_abs = 0.0f;
        double sum_sq_diff = 0.0;
        double sum_sq_ref = 0.0;

        for (size_t i = 0; i < count; ++i)
        {
            const float diff = actual[offset + i] - expected[offset + i];
            max_abs = std::max(max_abs, std::abs(diff));
            sum_sq_diff += static_cast<double>(diff) * static_cast<double>(diff);
            sum_sq_ref += static_cast<double>(expected[offset + i]) * static_cast<double>(expected[offset + i]);
        }

        const double rel_l2 = std::sqrt(sum_sq_diff / std::max(sum_sq_ref, 1e-30));
        return {max_abs, rel_l2};
    }

    /// @brief Returns the largest absolute value in a contiguous output span.
    float maxAbsSpan(const std::vector<float> &values, size_t offset, size_t count)
    {
        float max_abs = 0.0f;
        for (size_t i = 0; i < count; ++i)
            max_abs = std::max(max_abs, std::abs(values[offset + i]));
        return max_abs;
    }
#endif
} // namespace

#ifdef HAVE_ROCM

TEST(Test__ROCmGDNPaddedRealLength, RecurrenceEffectivePrefillMatchesUnpaddedDecode)
{
    if (!hasROCm())
        GTEST_SKIP() << "No ROCm device available";
    checkHip(hipSetDevice(0), "hipSetDevice");

    constexpr int n_heads = 2;
    constexpr int d_k = 128;
    constexpr int d_v = 128;
    constexpr int bucket_len = 13;
    constexpr int decode_row = bucket_len;
    constexpr int total_len = bucket_len + 1;
    constexpr int qk_stride = n_heads * d_k;
    constexpr int v_stride = n_heads * d_v;
    constexpr size_t output_elems = static_cast<size_t>(total_len) * static_cast<size_t>(v_stride);

    for (int real_len : {11, 7})
    {
        const auto Q = makeSequenceRows(total_len, qk_stride, real_len, bucket_len, 0.0021f, 0.19f, 0.0035f);
        const auto K = makeSequenceRows(total_len, qk_stride, real_len, bucket_len, -0.0019f, 0.17f, -0.0027f);
        const auto V = makeSequenceRows(total_len, v_stride, real_len, bucket_len, 0.0025f, 0.23f, 0.0041f);
        const auto alpha = makeSequenceRows(total_len, n_heads, real_len, bucket_len, 0.025f, 0.8f, 0.031f);
        const auto beta = makeSequenceRows(total_len, n_heads, real_len, bucket_len, -0.021f, 0.7f, -0.029f);
        const std::vector<float> A_log(static_cast<size_t>(n_heads), -0.5f);
        const std::vector<float> dt_bias(static_cast<size_t>(n_heads), 0.1f);

        HipFloatBuffer d_Q(Q);
        HipFloatBuffer d_K(K);
        HipFloatBuffer d_V(V);
        HipFloatBuffer d_alpha(alpha);
        HipFloatBuffer d_beta(beta);
        HipFloatBuffer d_A_log(A_log);
        HipFloatBuffer d_dt_bias(dt_bias);
        HipFloatBuffer d_padded_out(output_elems, 123.0f);
        HipFloatBuffer d_ref_out(output_elems, -57.0f);
        HipIntBuffer d_effective_len(real_len);

        ROCmGatedDeltaNet padded_kernel(0);
        padded_kernel.allocateGPUState(n_heads * d_k * d_v);
        ASSERT_TRUE(padded_kernel.chunkForwardWithEffectiveSeqLen(
            d_Q.ptr, d_K.ptr, d_V.ptr, d_alpha.ptr, d_beta.ptr, d_A_log.ptr, d_dt_bias.ptr,
            d_padded_out.ptr, nullptr,
            bucket_len, n_heads, d_k, d_v,
            /*chunk_size=*/64, /*use_qk_l2norm=*/true,
            d_effective_len.ptr));
        checkHip(hipDeviceSynchronize(), "hipDeviceSynchronize(padded recurrence prefill)");
        ASSERT_TRUE(padded_kernel.recurrent_step(
            d_Q.ptr + static_cast<size_t>(decode_row) * qk_stride,
            d_K.ptr + static_cast<size_t>(decode_row) * qk_stride,
            d_V.ptr + static_cast<size_t>(decode_row) * v_stride,
            d_alpha.ptr + static_cast<size_t>(decode_row) * n_heads,
            d_beta.ptr + static_cast<size_t>(decode_row) * n_heads,
            d_A_log.ptr,
            d_dt_bias.ptr,
            d_padded_out.ptr + static_cast<size_t>(decode_row) * v_stride,
            nullptr,
            n_heads, d_k, d_v,
            /*use_qk_l2norm=*/true));
        checkHip(hipDeviceSynchronize(), "hipDeviceSynchronize(padded recurrence decode)");

        ROCmGatedDeltaNet ref_kernel(0);
        ref_kernel.allocateGPUState(n_heads * d_k * d_v);
        ASSERT_TRUE(ref_kernel.chunk_forward(
            d_Q.ptr, d_K.ptr, d_V.ptr, d_alpha.ptr, d_beta.ptr, d_A_log.ptr, d_dt_bias.ptr,
            d_ref_out.ptr, nullptr,
            real_len, n_heads, d_k, d_v,
            /*chunk_size=*/64, /*use_qk_l2norm=*/true));
        checkHip(hipDeviceSynchronize(), "hipDeviceSynchronize(reference recurrence prefill)");
        ASSERT_TRUE(ref_kernel.recurrent_step(
            d_Q.ptr + static_cast<size_t>(decode_row) * qk_stride,
            d_K.ptr + static_cast<size_t>(decode_row) * qk_stride,
            d_V.ptr + static_cast<size_t>(decode_row) * v_stride,
            d_alpha.ptr + static_cast<size_t>(decode_row) * n_heads,
            d_beta.ptr + static_cast<size_t>(decode_row) * n_heads,
            d_A_log.ptr,
            d_dt_bias.ptr,
            d_ref_out.ptr + static_cast<size_t>(decode_row) * v_stride,
            nullptr,
            n_heads, d_k, d_v,
            /*use_qk_l2norm=*/true));
        checkHip(hipDeviceSynchronize(), "hipDeviceSynchronize(reference recurrence decode)");

        const auto padded = d_padded_out.toHost();
        const auto ref = d_ref_out.toHost();
        const auto prefix = diffStats(padded, ref, 0, static_cast<size_t>(real_len) * v_stride);
        const auto decode = diffStats(padded, ref, static_cast<size_t>(decode_row) * v_stride, v_stride);
        const float tail_abs = maxAbsSpan(
            padded,
            static_cast<size_t>(real_len) * v_stride,
            static_cast<size_t>(bucket_len - real_len) * v_stride);

        EXPECT_LT(prefix.first, 2e-4f) << "real_len=" << real_len;
        EXPECT_LT(prefix.second, 1e-4) << "real_len=" << real_len;
        EXPECT_LT(decode.first, 2e-4f) << "real_len=" << real_len;
        EXPECT_LT(decode.second, 1e-4) << "real_len=" << real_len;
        EXPECT_EQ(tail_abs, 0.0f) << "ROCm recurrence padding rows must be inert for real_len=" << real_len;
    }
}

TEST(Test__ROCmGDNPaddedRealLength, ShortConvEffectivePrefillPreservesDecodeState)
{
    if (!hasROCm())
        GTEST_SKIP() << "No ROCm device available";
    checkHip(hipSetDevice(0), "hipSetDevice");

    constexpr int channels = 32;
    constexpr int kernel_size = 4;
    constexpr int bucket_len = 13;
    constexpr int decode_row = bucket_len;
    constexpr int total_len = bucket_len + 1;
    constexpr size_t output_elems = static_cast<size_t>(total_len) * static_cast<size_t>(channels);

    const auto weight = makeShortConvWeights(channels, kernel_size);
    const auto bias = makeBias(channels);

    for (int real_len : {11, 7})
    {
        const auto input = makeSequenceRows(total_len, channels, real_len, bucket_len, 0.015f, 3.0f, 0.027f);

        HipFloatBuffer d_input(input);
        HipFloatBuffer d_weight(weight);
        HipFloatBuffer d_bias(bias);
        HipFloatBuffer d_padded_out(output_elems, 91.0f);
        HipFloatBuffer d_ref_out(output_elems, -37.0f);
        HipIntBuffer d_effective_len(real_len);

        ROCmShortConvolution padded_kernel(0);
        padded_kernel.allocateGPUState(channels * (kernel_size - 1));
        ASSERT_TRUE(padded_kernel.forwardWithEffectiveSeqLen(
            d_input.ptr, d_weight.ptr, d_bias.ptr,
            d_padded_out.ptr, nullptr,
            bucket_len, channels, kernel_size,
            d_effective_len.ptr,
            /*apply_silu=*/true));
        checkHip(hipDeviceSynchronize(), "hipDeviceSynchronize(padded short-conv prefill)");
        ASSERT_TRUE(padded_kernel.forward(
            d_input.ptr + static_cast<size_t>(decode_row) * channels,
            d_weight.ptr,
            d_bias.ptr,
            d_padded_out.ptr + static_cast<size_t>(decode_row) * channels,
            nullptr,
            1, channels, kernel_size,
            /*apply_silu=*/true));
        checkHip(hipDeviceSynchronize(), "hipDeviceSynchronize(padded short-conv decode)");

        ROCmShortConvolution ref_kernel(0);
        ref_kernel.allocateGPUState(channels * (kernel_size - 1));
        ASSERT_TRUE(ref_kernel.forward(
            d_input.ptr, d_weight.ptr, d_bias.ptr,
            d_ref_out.ptr, nullptr,
            real_len, channels, kernel_size,
            /*apply_silu=*/true));
        checkHip(hipDeviceSynchronize(), "hipDeviceSynchronize(reference short-conv prefill)");
        ASSERT_TRUE(ref_kernel.forward(
            d_input.ptr + static_cast<size_t>(decode_row) * channels,
            d_weight.ptr,
            d_bias.ptr,
            d_ref_out.ptr + static_cast<size_t>(decode_row) * channels,
            nullptr,
            1, channels, kernel_size,
            /*apply_silu=*/true));
        checkHip(hipDeviceSynchronize(), "hipDeviceSynchronize(reference short-conv decode)");

        const auto padded = d_padded_out.toHost();
        const auto ref = d_ref_out.toHost();
        const auto prefix = diffStats(padded, ref, 0, static_cast<size_t>(real_len) * channels);
        const auto decode = diffStats(padded, ref, static_cast<size_t>(decode_row) * channels, channels);
        const float tail_abs = maxAbsSpan(
            padded,
            static_cast<size_t>(real_len) * channels,
            static_cast<size_t>(bucket_len - real_len) * channels);

        EXPECT_LT(prefix.first, 1e-5f) << "real_len=" << real_len;
        EXPECT_LT(prefix.second, 1e-5) << "real_len=" << real_len;
        EXPECT_LT(decode.first, 1e-5f) << "real_len=" << real_len;
        EXPECT_LT(decode.second, 1e-5) << "real_len=" << real_len;
        EXPECT_EQ(tail_abs, 0.0f) << "ROCm short-conv padding rows must be inert for real_len=" << real_len;
    }
}

TEST(Test__ROCmGDNPaddedRealLength, RecurrenceVerifierStateSnapshotRestoresAcceptedRow)
{
    if (!hasROCm())
        GTEST_SKIP() << "No ROCm device available";
    checkHip(hipSetDevice(0), "hipSetDevice");

    constexpr int n_heads = 2;
    constexpr int d_k = 128;
    constexpr int d_v = 128;
    constexpr int verifier_len = 4;
    constexpr int accepted_rows = 2;
    constexpr int continuation_row = accepted_rows;
    constexpr int qk_stride = n_heads * d_k;
    constexpr int v_stride = n_heads * d_v;
    constexpr int state_floats = n_heads * d_k * d_v;
    constexpr size_t output_elems = static_cast<size_t>(verifier_len) * static_cast<size_t>(v_stride);

    const auto Q = makeSequenceRows(verifier_len, qk_stride, verifier_len, verifier_len, 0.0021f, 0.0f, 0.0021f);
    const auto K = makeSequenceRows(verifier_len, qk_stride, verifier_len, verifier_len, -0.0019f, 0.0f, -0.0019f);
    const auto V = makeSequenceRows(verifier_len, v_stride, verifier_len, verifier_len, 0.0025f, 0.0f, 0.0025f);
    const auto alpha = makeSequenceRows(verifier_len, n_heads, verifier_len, verifier_len, 0.025f, 0.0f, 0.025f);
    const auto beta = makeSequenceRows(verifier_len, n_heads, verifier_len, verifier_len, -0.021f, 0.0f, -0.021f);
    const std::vector<float> A_log(static_cast<size_t>(n_heads), -0.5f);
    const std::vector<float> dt_bias(static_cast<size_t>(n_heads), 0.1f);

    HipFloatBuffer d_Q(Q);
    HipFloatBuffer d_K(K);
    HipFloatBuffer d_V(V);
    HipFloatBuffer d_alpha(alpha);
    HipFloatBuffer d_beta(beta);
    HipFloatBuffer d_Q_cont(Q);
    HipFloatBuffer d_K_cont(K);
    HipFloatBuffer d_V_cont(V);
    HipFloatBuffer d_alpha_cont(alpha);
    HipFloatBuffer d_beta_cont(beta);
    HipFloatBuffer d_Q_ref(Q);
    HipFloatBuffer d_K_ref(K);
    HipFloatBuffer d_V_ref(V);
    HipFloatBuffer d_alpha_ref(alpha);
    HipFloatBuffer d_beta_ref(beta);
    HipFloatBuffer d_A_log(A_log);
    HipFloatBuffer d_dt_bias(dt_bias);
    HipFloatBuffer d_verifier_out(output_elems, 0.0f);
    HipFloatBuffer d_restored_next(static_cast<size_t>(v_stride), 0.0f);
    HipFloatBuffer d_ref_prefix(static_cast<size_t>(accepted_rows) * static_cast<size_t>(v_stride), 0.0f);
    HipFloatBuffer d_ref_next(static_cast<size_t>(v_stride), 0.0f);
    HipFloatBuffer d_snapshots(static_cast<size_t>(verifier_len) * static_cast<size_t>(state_floats), -99.0f);
    HipFloatBuffer d_speculative_state_work(static_cast<size_t>(state_floats), 0.0f);
    HipStreamHandle stream;

    ROCmGatedDeltaNet verifier_kernel(0);
    verifier_kernel.allocateGPUState(state_floats);
    verifier_kernel.setGPUStream(stream.stream);
    std::vector<float> initial_live_state(static_cast<size_t>(state_floats));
    ASSERT_TRUE(verifier_kernel.exportState(initial_live_state.data(), nullptr, nullptr));
    verifier_kernel.bindVerifierStateCaptureWorkspace(d_snapshots.ptr, verifier_len, state_floats);
    verifier_kernel.bindSpeculativeStateWorkspace(d_speculative_state_work.ptr, state_floats);
    ASSERT_TRUE(verifier_kernel.chunk_forward(
        d_Q.ptr, d_K.ptr, d_V.ptr, d_alpha.ptr, d_beta.ptr, d_A_log.ptr, d_dt_bias.ptr,
        d_verifier_out.ptr, nullptr,
        verifier_len, n_heads, d_k, d_v,
        /*chunk_size=*/64, /*use_qk_l2norm=*/true));
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(verifier recurrence capture)");
    std::vector<float> live_state_before_publish(static_cast<size_t>(state_floats));
    ASSERT_TRUE(verifier_kernel.exportState(live_state_before_publish.data(), nullptr, nullptr));
    ASSERT_TRUE(verifier_kernel.restoreVerifierStateCaptureRow(
        nullptr, accepted_rows - 1, stream.stream));
    verifier_kernel.bindVerifierStateCaptureWorkspace(nullptr, 0, state_floats);
    verifier_kernel.bindSpeculativeStateWorkspace(nullptr, state_floats);
    ASSERT_TRUE(verifier_kernel.recurrent_step(
        d_Q_cont.ptr + static_cast<size_t>(continuation_row) * qk_stride,
        d_K_cont.ptr + static_cast<size_t>(continuation_row) * qk_stride,
        d_V_cont.ptr + static_cast<size_t>(continuation_row) * v_stride,
        d_alpha_cont.ptr + static_cast<size_t>(continuation_row) * n_heads,
        d_beta_cont.ptr + static_cast<size_t>(continuation_row) * n_heads,
        d_A_log.ptr,
        d_dt_bias.ptr,
        d_restored_next.ptr,
        nullptr,
        n_heads, d_k, d_v,
        /*use_qk_l2norm=*/true));
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(restored recurrence decode)");

    ROCmGatedDeltaNet ref_kernel(0);
    ref_kernel.allocateGPUState(state_floats);
    ASSERT_TRUE(ref_kernel.chunk_forward(
        d_Q_ref.ptr, d_K_ref.ptr, d_V_ref.ptr, d_alpha_ref.ptr, d_beta_ref.ptr, d_A_log.ptr, d_dt_bias.ptr,
        d_ref_prefix.ptr, nullptr,
        accepted_rows, n_heads, d_k, d_v,
        /*chunk_size=*/64, /*use_qk_l2norm=*/true));
    ASSERT_TRUE(ref_kernel.recurrent_step(
        d_Q_ref.ptr + static_cast<size_t>(continuation_row) * qk_stride,
        d_K_ref.ptr + static_cast<size_t>(continuation_row) * qk_stride,
        d_V_ref.ptr + static_cast<size_t>(continuation_row) * v_stride,
        d_alpha_ref.ptr + static_cast<size_t>(continuation_row) * n_heads,
        d_beta_ref.ptr + static_cast<size_t>(continuation_row) * n_heads,
        d_A_log.ptr,
        d_dt_bias.ptr,
        d_ref_next.ptr,
        nullptr,
        n_heads, d_k, d_v,
        /*use_qk_l2norm=*/true));
    checkHip(hipDeviceSynchronize(), "hipDeviceSynchronize(reference recurrence decode)");

    const auto restored = d_restored_next.toHost();
    const auto ref = d_ref_next.toHost();
    const auto diff = diffStats(restored, ref, 0, restored.size());
    const auto live_state_diff =
        diffStats(live_state_before_publish, initial_live_state, 0, initial_live_state.size());
    EXPECT_LT(diff.first, 2e-4f);
    EXPECT_LT(diff.second, 1e-4);
    EXPECT_LT(live_state_diff.first, 1e-7f);
    EXPECT_LT(live_state_diff.second, 1e-7)
        << "ROCm verifier capture must not mutate live recurrence state before publication";
}

TEST(Test__ROCmGDNPaddedRealLength, RecurrenceDeviceMetadataRestoreMatchesAcceptedPrefix)
{
    if (!hasROCm())
        GTEST_SKIP() << "No ROCm device available";
    checkHip(hipSetDevice(0), "hipSetDevice");

    constexpr int n_heads = 2;
    constexpr int d_k = 128;
    constexpr int d_v = 128;
    constexpr int verifier_len = 4;
    constexpr int accepted_rows = 2;
    constexpr int continuation_row = accepted_rows;
    constexpr int qk_stride = n_heads * d_k;
    constexpr int v_stride = n_heads * d_v;
    constexpr int state_floats = n_heads * d_k * d_v;
    constexpr size_t output_elems = static_cast<size_t>(verifier_len) * static_cast<size_t>(v_stride);

    const auto Q = makeSequenceRows(verifier_len, qk_stride, verifier_len, verifier_len, 0.0021f, 0.0f, 0.0021f);
    const auto K = makeSequenceRows(verifier_len, qk_stride, verifier_len, verifier_len, -0.0019f, 0.0f, -0.0019f);
    const auto V = makeSequenceRows(verifier_len, v_stride, verifier_len, verifier_len, 0.0025f, 0.0f, 0.0025f);
    const auto alpha = makeSequenceRows(verifier_len, n_heads, verifier_len, verifier_len, 0.025f, 0.0f, 0.025f);
    const auto beta = makeSequenceRows(verifier_len, n_heads, verifier_len, verifier_len, -0.021f, 0.0f, -0.021f);
    const std::vector<float> A_log(static_cast<size_t>(n_heads), -0.5f);
    const std::vector<float> dt_bias(static_cast<size_t>(n_heads), 0.1f);

    HipFloatBuffer d_Q(Q);
    HipFloatBuffer d_K(K);
    HipFloatBuffer d_V(V);
    HipFloatBuffer d_alpha(alpha);
    HipFloatBuffer d_beta(beta);
    HipFloatBuffer d_Q_cont(Q);
    HipFloatBuffer d_K_cont(K);
    HipFloatBuffer d_V_cont(V);
    HipFloatBuffer d_alpha_cont(alpha);
    HipFloatBuffer d_beta_cont(beta);
    HipFloatBuffer d_Q_ref(Q);
    HipFloatBuffer d_K_ref(K);
    HipFloatBuffer d_V_ref(V);
    HipFloatBuffer d_alpha_ref(alpha);
    HipFloatBuffer d_beta_ref(beta);
    HipFloatBuffer d_A_log(A_log);
    HipFloatBuffer d_dt_bias(dt_bias);
    HipFloatBuffer d_verifier_out(output_elems, 0.0f);
    HipFloatBuffer d_restored_next(static_cast<size_t>(v_stride), 0.0f);
    HipFloatBuffer d_ref_prefix(static_cast<size_t>(accepted_rows) * static_cast<size_t>(v_stride), 0.0f);
    HipFloatBuffer d_ref_next(static_cast<size_t>(v_stride), 0.0f);
    HipFloatBuffer d_snapshots(static_cast<size_t>(verifier_len) * static_cast<size_t>(state_floats), -99.0f);
    HipFloatBuffer d_speculative_state_work(static_cast<size_t>(state_floats), 0.0f);
    HipIntBuffer d_accepted_state_slots({-1, -1, accepted_rows - 1});
    constexpr int metadata_request_index = 2;
    HipStreamHandle stream;

    ROCmGatedDeltaNet verifier_kernel(0);
    verifier_kernel.allocateGPUState(state_floats);
    verifier_kernel.setGPUStream(stream.stream);
    verifier_kernel.bindVerifierStateCaptureWorkspace(d_snapshots.ptr, verifier_len, state_floats);
    verifier_kernel.bindSpeculativeStateWorkspace(d_speculative_state_work.ptr, state_floats);
    ASSERT_TRUE(verifier_kernel.chunk_forward(
        d_Q.ptr, d_K.ptr, d_V.ptr, d_alpha.ptr, d_beta.ptr, d_A_log.ptr, d_dt_bias.ptr,
        d_verifier_out.ptr, nullptr,
        verifier_len, n_heads, d_k, d_v,
        /*chunk_size=*/64, /*use_qk_l2norm=*/true));
    ASSERT_FALSE(verifier_kernel.publishAcceptedSpeculativeStateFromDeviceMetadata(
        nullptr, d_accepted_state_slots.ptr, metadata_request_index, nullptr));
    ASSERT_TRUE(verifier_kernel.publishAcceptedSpeculativeStateFromDeviceMetadata(
        nullptr, d_accepted_state_slots.ptr, metadata_request_index, stream.stream));
    ASSERT_TRUE(verifier_kernel.recurrent_step(
        d_Q_cont.ptr + static_cast<size_t>(continuation_row) * qk_stride,
        d_K_cont.ptr + static_cast<size_t>(continuation_row) * qk_stride,
        d_V_cont.ptr + static_cast<size_t>(continuation_row) * v_stride,
        d_alpha_cont.ptr + static_cast<size_t>(continuation_row) * n_heads,
        d_beta_cont.ptr + static_cast<size_t>(continuation_row) * n_heads,
        d_A_log.ptr,
        d_dt_bias.ptr,
        d_restored_next.ptr,
        nullptr,
        n_heads, d_k, d_v,
        /*use_qk_l2norm=*/true));
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(restored recurrence metadata)");

    ROCmGatedDeltaNet ref_kernel(0);
    ref_kernel.allocateGPUState(state_floats);
    ref_kernel.setGPUStream(stream.stream);
    ASSERT_TRUE(ref_kernel.chunk_forward(
        d_Q_ref.ptr, d_K_ref.ptr, d_V_ref.ptr, d_alpha_ref.ptr, d_beta_ref.ptr, d_A_log.ptr, d_dt_bias.ptr,
        d_ref_prefix.ptr, nullptr,
        accepted_rows, n_heads, d_k, d_v,
        /*chunk_size=*/64, /*use_qk_l2norm=*/true));
    ASSERT_TRUE(ref_kernel.recurrent_step(
        d_Q_ref.ptr + static_cast<size_t>(continuation_row) * qk_stride,
        d_K_ref.ptr + static_cast<size_t>(continuation_row) * qk_stride,
        d_V_ref.ptr + static_cast<size_t>(continuation_row) * v_stride,
        d_alpha_ref.ptr + static_cast<size_t>(continuation_row) * n_heads,
        d_beta_ref.ptr + static_cast<size_t>(continuation_row) * n_heads,
        d_A_log.ptr,
        d_dt_bias.ptr,
        d_ref_next.ptr,
        nullptr,
        n_heads, d_k, d_v,
        /*use_qk_l2norm=*/true));
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(reference recurrence metadata)");

    const auto restored = d_restored_next.toHost();
    const auto ref = d_ref_next.toHost();
    const auto diff = diffStats(restored, ref, 0, restored.size());
    EXPECT_LT(diff.first, 2e-4f);
    EXPECT_LT(diff.second, 1e-4);
}

TEST(Test__ROCmGDNPaddedRealLength, RecurrenceM4FinalStateMatchesStepwiseReplay)
{
    if (!hasROCm())
        GTEST_SKIP() << "No ROCm device available";
    checkHip(hipSetDevice(0), "hipSetDevice");

    constexpr int n_heads = 40;
    constexpr int d_k = 128;
    constexpr int d_v = 128;
    constexpr int verifier_len = 4;
    constexpr int qk_stride = n_heads * d_k;
    constexpr int v_stride = n_heads * d_v;
    constexpr int state_floats = n_heads * d_k * d_v;
    constexpr size_t output_elems = static_cast<size_t>(verifier_len) * static_cast<size_t>(v_stride);

    const auto Q = makeSequenceRows(verifier_len, qk_stride, verifier_len, verifier_len, 0.0021f, 0.0f, 0.0021f);
    const auto K = makeSequenceRows(verifier_len, qk_stride, verifier_len, verifier_len, -0.0019f, 0.0f, -0.0019f);
    const auto V = makeSequenceRows(verifier_len, v_stride, verifier_len, verifier_len, 0.0025f, 0.0f, 0.0025f);
    const auto alpha = makeSequenceRows(verifier_len, n_heads, verifier_len, verifier_len, 0.025f, 0.0f, 0.025f);
    const auto beta = makeSequenceRows(verifier_len, n_heads, verifier_len, verifier_len, -0.021f, 0.0f, -0.021f);
    const auto initial_state = makeInitialState(static_cast<size_t>(state_floats), 0.00091f);
    const std::vector<float> A_log(static_cast<size_t>(n_heads), -0.5f);
    const std::vector<float> dt_bias(static_cast<size_t>(n_heads), 0.1f);

    HipFloatBuffer d_Q_m4(Q);
    HipFloatBuffer d_K_m4(K);
    HipFloatBuffer d_V_m4(V);
    HipFloatBuffer d_alpha_m4(alpha);
    HipFloatBuffer d_beta_m4(beta);
    HipFloatBuffer d_Q_step(Q);
    HipFloatBuffer d_K_step(K);
    HipFloatBuffer d_V_step(V);
    HipFloatBuffer d_alpha_step(alpha);
    HipFloatBuffer d_beta_step(beta);
    HipFloatBuffer d_A_log(A_log);
    HipFloatBuffer d_dt_bias(dt_bias);
    HipFloatBuffer d_m4_out(output_elems, 0.0f);
    HipFloatBuffer d_step_out(output_elems, 0.0f);
    HipStreamHandle stream;

    ROCmGatedDeltaNet m4_kernel(0);
    m4_kernel.allocateGPUState(state_floats);
    m4_kernel.setGPUStream(stream.stream);
    ASSERT_TRUE(m4_kernel.importState(initial_state.data(), nullptr, stream.stream));
    ASSERT_TRUE(m4_kernel.chunk_forward(
        d_Q_m4.ptr, d_K_m4.ptr, d_V_m4.ptr, d_alpha_m4.ptr, d_beta_m4.ptr, d_A_log.ptr, d_dt_bias.ptr,
        d_m4_out.ptr, nullptr,
        verifier_len, n_heads, d_k, d_v,
        /*chunk_size=*/64, /*use_qk_l2norm=*/true));
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(M4 recurrence)");
    std::vector<float> m4_state(static_cast<size_t>(state_floats));
    ASSERT_TRUE(m4_kernel.exportState(m4_state.data(), nullptr, nullptr));

    ROCmGatedDeltaNet step_kernel(0);
    step_kernel.allocateGPUState(state_floats);
    step_kernel.setGPUStream(stream.stream);
    ASSERT_TRUE(step_kernel.importState(initial_state.data(), nullptr, stream.stream));
    for (int row = 0; row < verifier_len; ++row)
    {
        ASSERT_TRUE(step_kernel.recurrent_step(
            d_Q_step.ptr + static_cast<size_t>(row) * qk_stride,
            d_K_step.ptr + static_cast<size_t>(row) * qk_stride,
            d_V_step.ptr + static_cast<size_t>(row) * v_stride,
            d_alpha_step.ptr + static_cast<size_t>(row) * n_heads,
            d_beta_step.ptr + static_cast<size_t>(row) * n_heads,
            d_A_log.ptr,
            d_dt_bias.ptr,
            d_step_out.ptr + static_cast<size_t>(row) * v_stride,
            nullptr,
            n_heads, d_k, d_v,
            /*use_qk_l2norm=*/true));
    }
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(stepwise recurrence)");
    std::vector<float> step_state(static_cast<size_t>(state_floats));
    ASSERT_TRUE(step_kernel.exportState(step_state.data(), nullptr, nullptr));

    const auto m4_out = d_m4_out.toHost();
    const auto step_out = d_step_out.toHost();
    const auto out_diff = diffStats(m4_out, step_out, 0, output_elems);
    const auto state_diff = diffStats(m4_state, step_state, 0, m4_state.size());
    EXPECT_LT(out_diff.first, 2e-4f);
    EXPECT_LT(out_diff.second, 1e-4);
    EXPECT_LT(state_diff.first, 2e-4f);
    EXPECT_LT(state_diff.second, 1e-4)
        << "ROCm M=4 recurrence must leave the same kernel-owned state as four decode steps";
}

TEST(Test__ROCmGDNPaddedRealLength, RecurrenceM4DV64FinalStateMatchesStepwiseReplay)
{
    if (!hasROCm())
        GTEST_SKIP() << "No ROCm device available";
    checkHip(hipSetDevice(0), "hipSetDevice");

    constexpr int n_heads = 48;
    constexpr int d_k = 128;
    constexpr int d_v = 64;
    constexpr int verifier_len = 4;
    constexpr int qk_stride = n_heads * d_k;
    constexpr int v_stride = n_heads * d_v;
    constexpr int state_floats = n_heads * d_k * d_v;
    constexpr size_t output_elems = static_cast<size_t>(verifier_len) * static_cast<size_t>(v_stride);

    const auto Q = makeSequenceRows(verifier_len, qk_stride, verifier_len, verifier_len, 0.0071f, 0.0f, 0.0071f);
    const auto K = makeSequenceRows(verifier_len, qk_stride, verifier_len, verifier_len, -0.0063f, 0.0f, -0.0063f);
    const auto V = makeSequenceRows(verifier_len, v_stride, verifier_len, verifier_len, 0.0085f, 0.0f, 0.0085f);
    const auto alpha = makeSequenceRows(verifier_len, n_heads, verifier_len, verifier_len, 0.11f, 0.0f, 0.11f);
    const auto beta = makeSequenceRows(verifier_len, n_heads, verifier_len, verifier_len, -0.09f, 0.0f, -0.09f);
    const auto initial_state = makeInitialState(static_cast<size_t>(state_floats), 0.018f);
    const std::vector<float> A_log(static_cast<size_t>(n_heads), -0.5f);
    const std::vector<float> dt_bias(static_cast<size_t>(n_heads), 0.1f);

    HipFloatBuffer d_Q_m4(Q);
    HipFloatBuffer d_K_m4(K);
    HipFloatBuffer d_V_m4(V);
    HipFloatBuffer d_alpha_m4(alpha);
    HipFloatBuffer d_beta_m4(beta);
    HipFloatBuffer d_Q_step(Q);
    HipFloatBuffer d_K_step(K);
    HipFloatBuffer d_V_step(V);
    HipFloatBuffer d_alpha_step(alpha);
    HipFloatBuffer d_beta_step(beta);
    HipFloatBuffer d_A_log(A_log);
    HipFloatBuffer d_dt_bias(dt_bias);
    HipFloatBuffer d_m4_out(output_elems, 0.0f);
    HipFloatBuffer d_step_out(output_elems, 0.0f);
    HipStreamHandle stream;

    ROCmGatedDeltaNet m4_kernel(0);
    m4_kernel.allocateGPUState(state_floats);
    m4_kernel.setGPUStream(stream.stream);
    ASSERT_TRUE(m4_kernel.importState(initial_state.data(), nullptr, stream.stream));
    ASSERT_TRUE(m4_kernel.chunk_forward(
        d_Q_m4.ptr, d_K_m4.ptr, d_V_m4.ptr, d_alpha_m4.ptr, d_beta_m4.ptr, d_A_log.ptr, d_dt_bias.ptr,
        d_m4_out.ptr, nullptr,
        verifier_len, n_heads, d_k, d_v,
        /*chunk_size=*/64, /*use_qk_l2norm=*/true));
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(M4 d_v=64 recurrence)");
    std::vector<float> m4_state(static_cast<size_t>(state_floats));
    ASSERT_TRUE(m4_kernel.exportState(m4_state.data(), nullptr, nullptr));

    ROCmGatedDeltaNet step_kernel(0);
    step_kernel.allocateGPUState(state_floats);
    step_kernel.setGPUStream(stream.stream);
    ASSERT_TRUE(step_kernel.importState(initial_state.data(), nullptr, stream.stream));
    for (int row = 0; row < verifier_len; ++row)
    {
        ASSERT_TRUE(step_kernel.recurrent_step(
            d_Q_step.ptr + static_cast<size_t>(row) * qk_stride,
            d_K_step.ptr + static_cast<size_t>(row) * qk_stride,
            d_V_step.ptr + static_cast<size_t>(row) * v_stride,
            d_alpha_step.ptr + static_cast<size_t>(row) * n_heads,
            d_beta_step.ptr + static_cast<size_t>(row) * n_heads,
            d_A_log.ptr,
            d_dt_bias.ptr,
            d_step_out.ptr + static_cast<size_t>(row) * v_stride,
            nullptr,
            n_heads, d_k, d_v,
            /*use_qk_l2norm=*/true));
    }
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(stepwise d_v=64 recurrence)");
    std::vector<float> step_state(static_cast<size_t>(state_floats));
    ASSERT_TRUE(step_kernel.exportState(step_state.data(), nullptr, nullptr));

    const auto m4_out = d_m4_out.toHost();
    const auto step_out = d_step_out.toHost();
    const auto out_diff = diffStats(m4_out, step_out, 0, output_elems);
    const auto state_diff = diffStats(m4_state, step_state, 0, m4_state.size());
    EXPECT_LT(out_diff.first, 2e-4f);
    EXPECT_LT(out_diff.second, 1e-4);
    EXPECT_LT(state_diff.first, 2e-4f);
    EXPECT_LT(state_diff.second, 1e-4)
        << "ROCm M=4 d_v=64 recurrence must match four decode steps";
}

TEST(Test__ROCmGDNPaddedRealLength, MergedQKVM4FinalStateMatchesStepwiseReplay)
{
    if (!hasROCm())
        GTEST_SKIP() << "No ROCm device available";
    checkHip(hipSetDevice(0), "hipSetDevice");

    constexpr int n_k_heads = 16;
    constexpr int n_v_heads = 40;
    constexpr int d_k = 128;
    constexpr int d_v = 128;
    constexpr int verifier_len = 4;
    constexpr int q_src_dim = n_k_heads * d_k;
    constexpr int k_src_dim = n_k_heads * d_k;
    constexpr int v_dim = n_v_heads * d_v;
    constexpr int qkv_stride = q_src_dim + k_src_dim + v_dim;
    constexpr int qk_stride = n_v_heads * d_k;
    constexpr int state_floats = n_v_heads * d_k * d_v;
    constexpr size_t output_elems = static_cast<size_t>(verifier_len) * static_cast<size_t>(v_dim);
    constexpr size_t deinterleave_elems =
        static_cast<size_t>(verifier_len) *
        static_cast<size_t>(qk_stride + qk_stride + v_dim);

    const auto Q_src = makeSequenceRows(verifier_len, q_src_dim, verifier_len, verifier_len, 0.0021f, 0.0f, 0.0021f);
    const auto K_src = makeSequenceRows(verifier_len, k_src_dim, verifier_len, verifier_len, -0.0019f, 0.0f, -0.0019f);
    const auto V_src = makeSequenceRows(verifier_len, v_dim, verifier_len, verifier_len, 0.0025f, 0.0f, 0.0025f);

    std::vector<float> merged(static_cast<size_t>(verifier_len) * static_cast<size_t>(qkv_stride));
    for (int row = 0; row < verifier_len; ++row)
    {
        float *dst = merged.data() + static_cast<size_t>(row) * qkv_stride;
        std::copy_n(Q_src.data() + static_cast<size_t>(row) * q_src_dim, q_src_dim, dst);
        std::copy_n(K_src.data() + static_cast<size_t>(row) * k_src_dim, k_src_dim, dst + q_src_dim);
        std::copy_n(V_src.data() + static_cast<size_t>(row) * v_dim, v_dim, dst + q_src_dim + k_src_dim);
    }

    const auto alpha = makeSequenceRows(verifier_len, n_v_heads, verifier_len, verifier_len, 0.025f, 0.0f, 0.025f);
    const auto beta = makeSequenceRows(verifier_len, n_v_heads, verifier_len, verifier_len, -0.021f, 0.0f, -0.021f);
    const auto initial_state = makeInitialState(static_cast<size_t>(state_floats), 0.00091f);
    const std::vector<float> A_log(static_cast<size_t>(n_v_heads), -0.5f);
    const std::vector<float> dt_bias(static_cast<size_t>(n_v_heads), 0.1f);

    HipFloatBuffer d_merged_m4(merged);
    HipFloatBuffer d_merged_step(merged);
    HipFloatBuffer d_alpha_m4(alpha);
    HipFloatBuffer d_beta_m4(beta);
    HipFloatBuffer d_alpha_step(alpha);
    HipFloatBuffer d_beta_step(beta);
    HipFloatBuffer d_A_log(A_log);
    HipFloatBuffer d_dt_bias(dt_bias);
    HipFloatBuffer d_m4_out(output_elems, 0.0f);
    HipFloatBuffer d_step_out(output_elems, 0.0f);
    HipFloatBuffer d_m4_scratch(deinterleave_elems, 0.0f);
    HipFloatBuffer d_step_scratch(
        static_cast<size_t>(qk_stride + qk_stride + v_dim),
        0.0f);
    HipStreamHandle stream;

    ROCmGatedDeltaNet m4_kernel(0);
    m4_kernel.allocateGPUState(state_floats);
    m4_kernel.setGPUStream(stream.stream);
    m4_kernel.bindDeinterleaveWorkspace(d_m4_scratch.ptr, d_m4_scratch.count);
    ASSERT_TRUE(m4_kernel.importState(initial_state.data(), nullptr, stream.stream));
    float *m4_q = nullptr;
    float *m4_k = nullptr;
    float *m4_v = nullptr;
    ASSERT_TRUE(m4_kernel.deinterleave_qkv_device(
        d_merged_m4.ptr,
        m4_q,
        m4_k,
        m4_v,
        verifier_len,
        n_k_heads,
        n_v_heads,
        d_k,
        d_v,
        /*global_v_head_offset=*/0));
    ASSERT_TRUE(m4_kernel.chunk_forward(
        m4_q, m4_k, m4_v, d_alpha_m4.ptr, d_beta_m4.ptr, d_A_log.ptr, d_dt_bias.ptr,
        d_m4_out.ptr, nullptr,
        verifier_len, n_v_heads, d_k, d_v,
        /*chunk_size=*/64, /*use_qk_l2norm=*/true));
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(M4 merged-QKV recurrence)");
    std::vector<float> m4_state(static_cast<size_t>(state_floats));
    ASSERT_TRUE(m4_kernel.exportState(m4_state.data(), nullptr, nullptr));

    ROCmGatedDeltaNet step_kernel(0);
    step_kernel.allocateGPUState(state_floats);
    step_kernel.setGPUStream(stream.stream);
    step_kernel.bindDeinterleaveWorkspace(d_step_scratch.ptr, d_step_scratch.count);
    ASSERT_TRUE(step_kernel.importState(initial_state.data(), nullptr, stream.stream));
    for (int row = 0; row < verifier_len; ++row)
    {
        float *step_q = nullptr;
        float *step_k = nullptr;
        float *step_v = nullptr;
        ASSERT_TRUE(step_kernel.deinterleave_qkv_device(
            d_merged_step.ptr + static_cast<size_t>(row) * qkv_stride,
            step_q,
            step_k,
            step_v,
            1,
            n_k_heads,
            n_v_heads,
            d_k,
            d_v,
            /*global_v_head_offset=*/0));
        ASSERT_TRUE(step_kernel.recurrent_step(
            step_q,
            step_k,
            step_v,
            d_alpha_step.ptr + static_cast<size_t>(row) * n_v_heads,
            d_beta_step.ptr + static_cast<size_t>(row) * n_v_heads,
            d_A_log.ptr,
            d_dt_bias.ptr,
            d_step_out.ptr + static_cast<size_t>(row) * v_dim,
            nullptr,
            n_v_heads, d_k, d_v,
            /*use_qk_l2norm=*/true));
    }
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(stepwise merged-QKV recurrence)");
    std::vector<float> step_state(static_cast<size_t>(state_floats));
    ASSERT_TRUE(step_kernel.exportState(step_state.data(), nullptr, nullptr));

    const auto m4_out = d_m4_out.toHost();
    const auto step_out = d_step_out.toHost();
    const auto out_diff = diffStats(m4_out, step_out, 0, output_elems);
    const auto state_diff = diffStats(m4_state, step_state, 0, m4_state.size());
    EXPECT_LT(out_diff.first, 2e-4f);
    EXPECT_LT(out_diff.second, 1e-4);
    EXPECT_LT(state_diff.first, 2e-4f);
    EXPECT_LT(state_diff.second, 1e-4)
        << "ROCm merged-QKV M=4 recurrence must leave the same state as four decode steps";
}

TEST(Test__ROCmGDNPaddedRealLength, MergedQKVM4Qwen36DenseShapeVerifierCaptureIsolatesLiveState)
{
    if (!hasROCm())
        GTEST_SKIP() << "No ROCm device available";
    checkHip(hipSetDevice(0), "hipSetDevice");

    constexpr int n_k_heads = 16;
    constexpr int n_v_heads = 48;
    constexpr int d_k = 128;
    constexpr int d_v = 128;
    constexpr int verifier_len = 4;
    constexpr int q_src_dim = n_k_heads * d_k;
    constexpr int k_src_dim = n_k_heads * d_k;
    constexpr int v_dim = n_v_heads * d_v;
    constexpr int qkv_stride = q_src_dim + k_src_dim + v_dim;
    constexpr int qk_stride = n_v_heads * d_k;
    constexpr int state_floats = n_v_heads * d_k * d_v;
    constexpr size_t output_elems = static_cast<size_t>(verifier_len) * static_cast<size_t>(v_dim);
    constexpr size_t deinterleave_elems =
        static_cast<size_t>(verifier_len) *
        static_cast<size_t>(qk_stride + qk_stride + v_dim);

    const auto Q_src = makeSequenceRows(verifier_len, q_src_dim, verifier_len, verifier_len, 0.0067f, 0.0f, 0.0067f);
    const auto K_src = makeSequenceRows(verifier_len, k_src_dim, verifier_len, verifier_len, -0.0059f, 0.0f, -0.0059f);
    const auto V_src = makeSequenceRows(verifier_len, v_dim, verifier_len, verifier_len, 0.0073f, 0.0f, 0.0073f);

    std::vector<float> merged(static_cast<size_t>(verifier_len) * static_cast<size_t>(qkv_stride));
    for (int row = 0; row < verifier_len; ++row)
    {
        float *dst = merged.data() + static_cast<size_t>(row) * qkv_stride;
        std::copy_n(Q_src.data() + static_cast<size_t>(row) * q_src_dim, q_src_dim, dst);
        std::copy_n(K_src.data() + static_cast<size_t>(row) * k_src_dim, k_src_dim, dst + q_src_dim);
        std::copy_n(V_src.data() + static_cast<size_t>(row) * v_dim, v_dim, dst + q_src_dim + k_src_dim);
    }

    const auto alpha = makeSequenceRows(verifier_len, n_v_heads, verifier_len, verifier_len, 0.087f, 0.0f, 0.087f);
    const auto beta = makeSequenceRows(verifier_len, n_v_heads, verifier_len, verifier_len, -0.071f, 0.0f, -0.071f);
    const auto initial_state = makeInitialState(static_cast<size_t>(state_floats), 0.021f);
    const std::vector<float> A_log(static_cast<size_t>(n_v_heads), -0.5f);
    const std::vector<float> dt_bias(static_cast<size_t>(n_v_heads), 0.1f);

    HipFloatBuffer d_merged_m4(merged);
    HipFloatBuffer d_merged_step(merged);
    HipFloatBuffer d_alpha_m4(alpha);
    HipFloatBuffer d_beta_m4(beta);
    HipFloatBuffer d_alpha_step(alpha);
    HipFloatBuffer d_beta_step(beta);
    HipFloatBuffer d_A_log(A_log);
    HipFloatBuffer d_dt_bias(dt_bias);
    HipFloatBuffer d_m4_out(output_elems, 0.0f);
    HipFloatBuffer d_step_out(output_elems, 0.0f);
    HipFloatBuffer d_m4_scratch(deinterleave_elems, 0.0f);
    HipFloatBuffer d_step_scratch(
        static_cast<size_t>(qk_stride + qk_stride + v_dim),
        0.0f);
    HipFloatBuffer d_snapshots(static_cast<size_t>(verifier_len) * static_cast<size_t>(state_floats), -99.0f);
    HipFloatBuffer d_speculative_state_work(static_cast<size_t>(state_floats), 0.0f);
    HipStreamHandle stream;

    ROCmGatedDeltaNet m4_kernel(0);
    m4_kernel.allocateGPUState(state_floats);
    m4_kernel.setGPUStream(stream.stream);
    m4_kernel.bindDeinterleaveWorkspace(d_m4_scratch.ptr, d_m4_scratch.count);
    m4_kernel.bindVerifierStateCaptureWorkspace(d_snapshots.ptr, verifier_len, state_floats);
    m4_kernel.bindSpeculativeStateWorkspace(d_speculative_state_work.ptr, state_floats);
    ASSERT_TRUE(m4_kernel.importState(initial_state.data(), nullptr, stream.stream));
    float *m4_q = nullptr;
    float *m4_k = nullptr;
    float *m4_v = nullptr;
    ASSERT_TRUE(m4_kernel.deinterleave_qkv_device(
        d_merged_m4.ptr,
        m4_q,
        m4_k,
        m4_v,
        verifier_len,
        n_k_heads,
        n_v_heads,
        d_k,
        d_v,
        /*global_v_head_offset=*/0));
    ASSERT_TRUE(m4_kernel.chunk_forward(
        m4_q, m4_k, m4_v, d_alpha_m4.ptr, d_beta_m4.ptr, d_A_log.ptr, d_dt_bias.ptr,
        d_m4_out.ptr, nullptr,
        verifier_len, n_v_heads, d_k, d_v,
        /*chunk_size=*/64, /*use_qk_l2norm=*/true));
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(M4 Qwen3.6 dense-shape recurrence)");
    std::vector<float> m4_state(static_cast<size_t>(state_floats));
    ASSERT_TRUE(m4_kernel.exportState(m4_state.data(), nullptr, nullptr));

    ROCmGatedDeltaNet step_kernel(0);
    step_kernel.allocateGPUState(state_floats);
    step_kernel.setGPUStream(stream.stream);
    step_kernel.bindDeinterleaveWorkspace(d_step_scratch.ptr, d_step_scratch.count);
    ASSERT_TRUE(step_kernel.importState(initial_state.data(), nullptr, stream.stream));
    for (int row = 0; row < verifier_len; ++row)
    {
        float *step_q = nullptr;
        float *step_k = nullptr;
        float *step_v = nullptr;
        ASSERT_TRUE(step_kernel.deinterleave_qkv_device(
            d_merged_step.ptr + static_cast<size_t>(row) * qkv_stride,
            step_q,
            step_k,
            step_v,
            1,
            n_k_heads,
            n_v_heads,
            d_k,
            d_v,
            /*global_v_head_offset=*/0));
        ASSERT_TRUE(step_kernel.recurrent_step(
            step_q,
            step_k,
            step_v,
            d_alpha_step.ptr + static_cast<size_t>(row) * n_v_heads,
            d_beta_step.ptr + static_cast<size_t>(row) * n_v_heads,
            d_A_log.ptr,
            d_dt_bias.ptr,
            d_step_out.ptr + static_cast<size_t>(row) * v_dim,
            nullptr,
            n_v_heads, d_k, d_v,
            /*use_qk_l2norm=*/true));
    }
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(stepwise Qwen3.6 dense-shape recurrence)");
    std::vector<float> step_state(static_cast<size_t>(state_floats));
    ASSERT_TRUE(step_kernel.exportState(step_state.data(), nullptr, nullptr));

    const auto m4_out = d_m4_out.toHost();
    const auto step_out = d_step_out.toHost();
    const auto snapshots = d_snapshots.toHost();
    std::vector<float> final_snapshot(static_cast<size_t>(state_floats));
    std::copy_n(
        snapshots.data() + static_cast<size_t>(verifier_len - 1) * static_cast<size_t>(state_floats),
        static_cast<size_t>(state_floats),
        final_snapshot.data());

    const auto out_diff = diffStats(m4_out, step_out, 0, output_elems);
    const auto snapshot_diff = diffStats(final_snapshot, step_state, 0, final_snapshot.size());
    const auto live_state_diff = diffStats(m4_state, initial_state, 0, m4_state.size());
    EXPECT_LT(out_diff.first, 2e-4f);
    EXPECT_LT(out_diff.second, 1e-4);
    EXPECT_LT(snapshot_diff.first, 2e-4f);
    EXPECT_LT(snapshot_diff.second, 1e-4)
        << "ROCm Qwen3.6 verifier capture snapshot must match four decode steps";
    EXPECT_LT(live_state_diff.first, 1e-7f);
    EXPECT_LT(live_state_diff.second, 1e-7)
        << "ROCm verifier capture must not mutate live recurrence state before publication";
}

TEST(Test__ROCmGDNPaddedRealLength, ShortConvVerifierStateSnapshotRestoresAcceptedRow)
{
    if (!hasROCm())
        GTEST_SKIP() << "No ROCm device available";
    checkHip(hipSetDevice(0), "hipSetDevice");

    constexpr int channels = 64;
    constexpr int kernel_size = 4;
    constexpr int verifier_len = 5;
    constexpr int accepted_rows = 3;
    constexpr int continuation_row = accepted_rows;
    constexpr int state_floats = channels * (kernel_size - 1);
    constexpr size_t output_elems = static_cast<size_t>(verifier_len) * static_cast<size_t>(channels);

    const auto input = makeSequenceRows(verifier_len, channels, verifier_len, verifier_len, 0.015f, 0.0f, 0.015f);
    const auto weight = makeShortConvWeights(channels, kernel_size);
    const auto bias = makeBias(channels);

    HipFloatBuffer d_input(input);
    HipFloatBuffer d_weight(weight);
    HipFloatBuffer d_bias(bias);
    HipFloatBuffer d_verifier_out(output_elems, 0.0f);
    HipFloatBuffer d_restored_next(static_cast<size_t>(channels), 0.0f);
    HipFloatBuffer d_ref_prefix(static_cast<size_t>(accepted_rows) * static_cast<size_t>(channels), 0.0f);
    HipFloatBuffer d_ref_next(static_cast<size_t>(channels), 0.0f);
    HipFloatBuffer d_snapshots(static_cast<size_t>(verifier_len) * static_cast<size_t>(state_floats), -77.0f);
    HipFloatBuffer d_speculative_state_work(static_cast<size_t>(state_floats), 0.0f);
    HipStreamHandle stream;

    ROCmShortConvolution verifier_kernel(0);
    verifier_kernel.allocateGPUState(state_floats);
    verifier_kernel.setGPUStream(stream.stream);
    verifier_kernel.bindVerifierStateCaptureWorkspace(d_snapshots.ptr, verifier_len, state_floats);
    verifier_kernel.bindSpeculativeStateWorkspace(d_speculative_state_work.ptr, state_floats);
    ASSERT_TRUE(verifier_kernel.forward(
        d_input.ptr,
        d_weight.ptr,
        d_bias.ptr,
        d_verifier_out.ptr,
        nullptr,
        verifier_len, channels, kernel_size,
        /*apply_silu=*/true));
    ASSERT_TRUE(verifier_kernel.restoreVerifierStateCaptureRow(
        nullptr, accepted_rows - 1, stream.stream));
    verifier_kernel.bindVerifierStateCaptureWorkspace(nullptr, 0, state_floats);
    verifier_kernel.bindSpeculativeStateWorkspace(nullptr, state_floats);
    ASSERT_TRUE(verifier_kernel.forward(
        d_input.ptr + static_cast<size_t>(continuation_row) * channels,
        d_weight.ptr,
        d_bias.ptr,
        d_restored_next.ptr,
        nullptr,
        1, channels, kernel_size,
        /*apply_silu=*/true));
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(restored short-conv decode)");

    ROCmShortConvolution ref_kernel(0);
    ref_kernel.allocateGPUState(state_floats);
    ASSERT_TRUE(ref_kernel.forward(
        d_input.ptr,
        d_weight.ptr,
        d_bias.ptr,
        d_ref_prefix.ptr,
        nullptr,
        accepted_rows, channels, kernel_size,
        /*apply_silu=*/true));
    ASSERT_TRUE(ref_kernel.forward(
        d_input.ptr + static_cast<size_t>(continuation_row) * channels,
        d_weight.ptr,
        d_bias.ptr,
        d_ref_next.ptr,
        nullptr,
        1, channels, kernel_size,
        /*apply_silu=*/true));
    checkHip(hipDeviceSynchronize(), "hipDeviceSynchronize(reference short-conv decode)");

    const auto restored = d_restored_next.toHost();
    const auto ref = d_ref_next.toHost();
    const auto diff = diffStats(restored, ref, 0, restored.size());
    EXPECT_LT(diff.first, 1e-5f);
    EXPECT_LT(diff.second, 1e-5);
}

TEST(Test__ROCmGDNPaddedRealLength, ShortConvDeviceMetadataRestoreMatchesAcceptedPrefix)
{
    if (!hasROCm())
        GTEST_SKIP() << "No ROCm device available";
    checkHip(hipSetDevice(0), "hipSetDevice");

    constexpr int channels = 64;
    constexpr int kernel_size = 4;
    constexpr int verifier_len = 5;
    constexpr int accepted_rows = 3;
    constexpr int continuation_row = accepted_rows;
    constexpr int state_floats = channels * (kernel_size - 1);
    constexpr size_t output_elems = static_cast<size_t>(verifier_len) * static_cast<size_t>(channels);

    const auto input = makeSequenceRows(verifier_len, channels, verifier_len, verifier_len, 0.015f, 0.0f, 0.015f);
    const auto weight = makeShortConvWeights(channels, kernel_size);
    const auto bias = makeBias(channels);

    HipFloatBuffer d_input(input);
    HipFloatBuffer d_weight(weight);
    HipFloatBuffer d_bias(bias);
    HipFloatBuffer d_verifier_out(output_elems, 0.0f);
    HipFloatBuffer d_restored_next(static_cast<size_t>(channels), 0.0f);
    HipFloatBuffer d_ref_prefix(static_cast<size_t>(accepted_rows) * static_cast<size_t>(channels), 0.0f);
    HipFloatBuffer d_ref_next(static_cast<size_t>(channels), 0.0f);
    HipFloatBuffer d_snapshots(static_cast<size_t>(verifier_len) * static_cast<size_t>(state_floats), -77.0f);
    HipFloatBuffer d_speculative_state_work(static_cast<size_t>(state_floats), 0.0f);
    HipIntBuffer d_accepted_state_slots({-1, accepted_rows - 1});
    constexpr int metadata_request_index = 1;
    HipStreamHandle stream;

    ROCmShortConvolution verifier_kernel(0);
    verifier_kernel.allocateGPUState(state_floats);
    verifier_kernel.setGPUStream(stream.stream);
    verifier_kernel.bindVerifierStateCaptureWorkspace(d_snapshots.ptr, verifier_len, state_floats);
    verifier_kernel.bindSpeculativeStateWorkspace(d_speculative_state_work.ptr, state_floats);
    ASSERT_TRUE(verifier_kernel.forward(
        d_input.ptr,
        d_weight.ptr,
        d_bias.ptr,
        d_verifier_out.ptr,
        nullptr,
        verifier_len, channels, kernel_size,
        /*apply_silu=*/true));
    ASSERT_FALSE(verifier_kernel.publishAcceptedSpeculativeStateFromDeviceMetadata(
        nullptr, d_accepted_state_slots.ptr, metadata_request_index, nullptr));
    ASSERT_TRUE(verifier_kernel.publishAcceptedSpeculativeStateFromDeviceMetadata(
        nullptr, d_accepted_state_slots.ptr, metadata_request_index, stream.stream));
    ASSERT_TRUE(verifier_kernel.forward(
        d_input.ptr + static_cast<size_t>(continuation_row) * channels,
        d_weight.ptr,
        d_bias.ptr,
        d_restored_next.ptr,
        nullptr,
        1, channels, kernel_size,
        /*apply_silu=*/true));
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(restored short-conv metadata)");

    ROCmShortConvolution ref_kernel(0);
    ref_kernel.allocateGPUState(state_floats);
    ref_kernel.setGPUStream(stream.stream);
    ASSERT_TRUE(ref_kernel.forward(
        d_input.ptr,
        d_weight.ptr,
        d_bias.ptr,
        d_ref_prefix.ptr,
        nullptr,
        accepted_rows, channels, kernel_size,
        /*apply_silu=*/true));
    ASSERT_TRUE(ref_kernel.forward(
        d_input.ptr + static_cast<size_t>(continuation_row) * channels,
        d_weight.ptr,
        d_bias.ptr,
        d_ref_next.ptr,
        nullptr,
        1, channels, kernel_size,
        /*apply_silu=*/true));
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(reference short-conv metadata)");

    const auto restored = d_restored_next.toHost();
    const auto ref = d_ref_next.toHost();
    const auto diff = diffStats(restored, ref, 0, restored.size());
    EXPECT_LT(diff.first, 1e-5f);
    EXPECT_LT(diff.second, 1e-5);
}

TEST(Test__ROCmGDNPaddedRealLength, ShortConvM4FinalStateMatchesStepwiseReplay)
{
    if (!hasROCm())
        GTEST_SKIP() << "No ROCm device available";
    checkHip(hipSetDevice(0), "hipSetDevice");

    constexpr int channels = 4096;
    constexpr int kernel_size = 4;
    constexpr int verifier_len = 4;
    constexpr int state_floats = channels * (kernel_size - 1);
    constexpr size_t output_elems = static_cast<size_t>(verifier_len) * static_cast<size_t>(channels);

    const auto input = makeSequenceRows(verifier_len, channels, verifier_len, verifier_len, 0.015f, 0.0f, 0.015f);
    const auto weight = makeShortConvWeights(channels, kernel_size);
    const auto bias = makeBias(channels);
    const auto initial_state = makeInitialState(static_cast<size_t>(state_floats), 0.0031f);

    HipFloatBuffer d_input_m4(input);
    HipFloatBuffer d_input_step(input);
    HipFloatBuffer d_weight(weight);
    HipFloatBuffer d_bias(bias);
    HipFloatBuffer d_m4_out(output_elems, 0.0f);
    HipFloatBuffer d_step_out(output_elems, 0.0f);
    HipStreamHandle stream;

    ROCmShortConvolution m4_kernel(0);
    m4_kernel.allocateGPUState(state_floats);
    m4_kernel.setGPUStream(stream.stream);
    ASSERT_TRUE(m4_kernel.importState(initial_state.data(), nullptr, stream.stream));
    ASSERT_TRUE(m4_kernel.forward(
        d_input_m4.ptr,
        d_weight.ptr,
        d_bias.ptr,
        d_m4_out.ptr,
        nullptr,
        verifier_len, channels, kernel_size,
        /*apply_silu=*/true));
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(M4 short-conv)");
    std::vector<float> m4_state(static_cast<size_t>(state_floats));
    ASSERT_TRUE(m4_kernel.exportState(m4_state.data(), nullptr, nullptr));

    ROCmShortConvolution step_kernel(0);
    step_kernel.allocateGPUState(state_floats);
    step_kernel.setGPUStream(stream.stream);
    ASSERT_TRUE(step_kernel.importState(initial_state.data(), nullptr, stream.stream));
    for (int row = 0; row < verifier_len; ++row)
    {
        ASSERT_TRUE(step_kernel.forward(
            d_input_step.ptr + static_cast<size_t>(row) * channels,
            d_weight.ptr,
            d_bias.ptr,
            d_step_out.ptr + static_cast<size_t>(row) * channels,
            nullptr,
            1, channels, kernel_size,
            /*apply_silu=*/true));
    }
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(stepwise short-conv)");
    std::vector<float> step_state(static_cast<size_t>(state_floats));
    ASSERT_TRUE(step_kernel.exportState(step_state.data(), nullptr, nullptr));

    const auto m4_out = d_m4_out.toHost();
    const auto step_out = d_step_out.toHost();
    const auto out_diff = diffStats(m4_out, step_out, 0, output_elems);
    const auto state_diff = diffStats(m4_state, step_state, 0, m4_state.size());
    EXPECT_LT(out_diff.first, 1e-5f);
    EXPECT_LT(out_diff.second, 1e-5);
    EXPECT_LT(state_diff.first, 1e-5f);
    EXPECT_LT(state_diff.second, 1e-5)
        << "ROCm M=4 short-conv must leave the same kernel-owned state as four decode steps";
}

TEST(Test__ROCmGDNPaddedRealLength, ShortConvQwen36M4InPlaceStateMatchesStepwiseReplay)
{
    if (!hasROCm())
        GTEST_SKIP() << "No ROCm device available";
    checkHip(hipSetDevice(0), "hipSetDevice");

    constexpr int channels = 10240;
    constexpr int kernel_size = 4;
    constexpr int verifier_len = 4;
    constexpr int state_floats = channels * (kernel_size - 1);
    constexpr size_t output_elems =
        static_cast<size_t>(verifier_len) * static_cast<size_t>(channels);

    const auto input = makeSequenceRows(
        verifier_len, channels, verifier_len, verifier_len,
        0.0095f, 0.0f, 0.0095f);
    const auto weight = makeShortConvWeights(channels, kernel_size);
    const auto bias = makeBias(channels);
    const auto initial_state =
        makeInitialState(static_cast<size_t>(state_floats), 0.0027f);

    HipFloatBuffer d_m4_inout(input);
    HipFloatBuffer d_step_inout(input);
    HipFloatBuffer d_weight(weight);
    HipFloatBuffer d_bias(bias);
    HipStreamHandle stream;

    ROCmShortConvolution m4_kernel(0);
    m4_kernel.allocateGPUState(state_floats);
    ASSERT_TRUE(m4_kernel.allocateGPUScratch(static_cast<int>(output_elems)));
    m4_kernel.setGPUStream(stream.stream);
    ASSERT_TRUE(m4_kernel.importState(initial_state.data(), nullptr, stream.stream));
    ASSERT_TRUE(m4_kernel.forward(
        d_m4_inout.ptr,
        d_weight.ptr,
        d_bias.ptr,
        d_m4_inout.ptr,
        nullptr,
        verifier_len, channels, kernel_size,
        /*apply_silu=*/true));
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(Qwen36 M4 in-place short-conv)");
    std::vector<float> m4_state(static_cast<size_t>(state_floats));
    ASSERT_TRUE(m4_kernel.exportState(m4_state.data(), nullptr, nullptr));

    ROCmShortConvolution step_kernel(0);
    step_kernel.allocateGPUState(state_floats);
    ASSERT_TRUE(step_kernel.allocateGPUScratch(channels));
    step_kernel.setGPUStream(stream.stream);
    ASSERT_TRUE(step_kernel.importState(initial_state.data(), nullptr, stream.stream));
    for (int row = 0; row < verifier_len; ++row)
    {
        ASSERT_TRUE(step_kernel.forward(
            d_step_inout.ptr + static_cast<size_t>(row) * channels,
            d_weight.ptr,
            d_bias.ptr,
            d_step_inout.ptr + static_cast<size_t>(row) * channels,
            nullptr,
            1, channels, kernel_size,
            /*apply_silu=*/true));
    }
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(Qwen36 stepwise in-place short-conv)");
    std::vector<float> step_state(static_cast<size_t>(state_floats));
    ASSERT_TRUE(step_kernel.exportState(step_state.data(), nullptr, nullptr));

    const auto m4_out = d_m4_inout.toHost();
    const auto step_out = d_step_inout.toHost();
    const auto out_diff = diffStats(m4_out, step_out, 0, output_elems);
    const auto state_diff = diffStats(m4_state, step_state, 0, m4_state.size());
    EXPECT_LT(out_diff.first, 1e-5f);
    EXPECT_LT(out_diff.second, 1e-5);
    EXPECT_LT(state_diff.first, 1e-5f);
    EXPECT_LT(state_diff.second, 1e-5)
        << "ROCm Qwen3.6-shaped in-place M=4 short-conv must leave the same "
           "kernel-owned state as four decode steps";
}

TEST(Test__ROCmGDNPaddedRealLength, ShortConvVerifierRowRestoreMatchesMultiStepReplay)
{
    if (!hasROCm())
        GTEST_SKIP() << "No ROCm device available";
    checkHip(hipSetDevice(0), "hipSetDevice");

    constexpr int channels = 64;
    constexpr int kernel_size = 4;
    constexpr int accepted_rows = 4;
    constexpr int continuation_rows = 4;
    constexpr int verifier_len = accepted_rows + continuation_rows;
    constexpr int state_floats = channels * (kernel_size - 1);
    constexpr size_t verifier_output_elems =
        static_cast<size_t>(verifier_len) * static_cast<size_t>(channels);
    constexpr size_t continuation_output_elems =
        static_cast<size_t>(continuation_rows) * static_cast<size_t>(channels);

    const auto input = makeSequenceRows(verifier_len, channels, verifier_len, verifier_len, 0.017f, 0.0f, 0.017f);
    const auto weight = makeShortConvWeights(channels, kernel_size);
    const auto bias = makeBias(channels);
    const auto initial_state = makeInitialState(static_cast<size_t>(state_floats), 0.0029f);

    HipFloatBuffer d_input(input);
    HipFloatBuffer d_input_ref(input);
    HipFloatBuffer d_weight(weight);
    HipFloatBuffer d_bias(bias);
    HipFloatBuffer d_verifier_out(verifier_output_elems, 0.0f);
    HipFloatBuffer d_restored_continuation(continuation_output_elems, 0.0f);
    HipFloatBuffer d_ref_prefix(static_cast<size_t>(accepted_rows) * static_cast<size_t>(channels), 0.0f);
    HipFloatBuffer d_ref_continuation(continuation_output_elems, 0.0f);
    HipFloatBuffer d_snapshots(static_cast<size_t>(verifier_len) * static_cast<size_t>(state_floats), -77.0f);
    HipFloatBuffer d_speculative_state_work(static_cast<size_t>(state_floats), 0.0f);
    HipStreamHandle stream;

    ROCmShortConvolution verifier_kernel(0);
    verifier_kernel.allocateGPUState(state_floats);
    verifier_kernel.setGPUStream(stream.stream);
    verifier_kernel.bindVerifierStateCaptureWorkspace(d_snapshots.ptr, verifier_len, state_floats);
    verifier_kernel.bindSpeculativeStateWorkspace(d_speculative_state_work.ptr, state_floats);
    ASSERT_TRUE(verifier_kernel.importState(initial_state.data(), nullptr, stream.stream));
    ASSERT_TRUE(verifier_kernel.forward(
        d_input.ptr,
        d_weight.ptr,
        d_bias.ptr,
        d_verifier_out.ptr,
        nullptr,
        verifier_len, channels, kernel_size,
        /*apply_silu=*/true));
    ASSERT_TRUE(verifier_kernel.restoreVerifierStateCaptureRow(
        nullptr, accepted_rows - 1, stream.stream));
    verifier_kernel.bindVerifierStateCaptureWorkspace(nullptr, 0, state_floats);
    verifier_kernel.bindSpeculativeStateWorkspace(nullptr, state_floats);
    for (int row = 0; row < continuation_rows; ++row)
    {
        const int source_row = accepted_rows + row;
        ASSERT_TRUE(verifier_kernel.forward(
            d_input.ptr + static_cast<size_t>(source_row) * channels,
            d_weight.ptr,
            d_bias.ptr,
            d_restored_continuation.ptr + static_cast<size_t>(row) * channels,
            nullptr,
            1, channels, kernel_size,
            /*apply_silu=*/true));
    }
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(restored short-conv continuation)");
    std::vector<float> restored_state(static_cast<size_t>(state_floats));
    ASSERT_TRUE(verifier_kernel.exportState(restored_state.data(), nullptr, nullptr));

    ROCmShortConvolution ref_kernel(0);
    ref_kernel.allocateGPUState(state_floats);
    ref_kernel.setGPUStream(stream.stream);
    ASSERT_TRUE(ref_kernel.importState(initial_state.data(), nullptr, stream.stream));
    ASSERT_TRUE(ref_kernel.forward(
        d_input_ref.ptr,
        d_weight.ptr,
        d_bias.ptr,
        d_ref_prefix.ptr,
        nullptr,
        accepted_rows, channels, kernel_size,
        /*apply_silu=*/true));
    for (int row = 0; row < continuation_rows; ++row)
    {
        const int source_row = accepted_rows + row;
        ASSERT_TRUE(ref_kernel.forward(
            d_input_ref.ptr + static_cast<size_t>(source_row) * channels,
            d_weight.ptr,
            d_bias.ptr,
            d_ref_continuation.ptr + static_cast<size_t>(row) * channels,
            nullptr,
            1, channels, kernel_size,
            /*apply_silu=*/true));
    }
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(reference short-conv continuation)");
    std::vector<float> ref_state(static_cast<size_t>(state_floats));
    ASSERT_TRUE(ref_kernel.exportState(ref_state.data(), nullptr, nullptr));

    const auto restored = d_restored_continuation.toHost();
    const auto ref = d_ref_continuation.toHost();
    const auto out_diff = diffStats(restored, ref, 0, restored.size());
    const auto state_diff = diffStats(restored_state, ref_state, 0, restored_state.size());
    EXPECT_LT(out_diff.first, 1e-5f);
    EXPECT_LT(out_diff.second, 1e-5);
    EXPECT_LT(state_diff.first, 1e-5f);
    EXPECT_LT(state_diff.second, 1e-5);
}

#endif
