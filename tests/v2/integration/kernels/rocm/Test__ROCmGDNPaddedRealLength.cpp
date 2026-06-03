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

    /// @brief RAII wrapper for a single int stored on a HIP device.
    struct HipIntBuffer
    {
        int *ptr = nullptr; ///< Device pointer owned by this buffer.

        explicit HipIntBuffer(int value)
        {
            checkHip(hipMalloc(reinterpret_cast<void **>(&ptr), sizeof(int)), "hipMalloc(int)");
            checkHip(hipMemcpy(ptr, &value, sizeof(int), hipMemcpyHostToDevice), "hipMemcpy host-to-device(int)");
        }

        ~HipIntBuffer()
        {
            if (ptr)
                (void)hipFree(ptr);
        }

        HipIntBuffer(const HipIntBuffer &) = delete;
        HipIntBuffer &operator=(const HipIntBuffer &) = delete;
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
    HipFloatBuffer d_A_log(A_log);
    HipFloatBuffer d_dt_bias(dt_bias);
    HipFloatBuffer d_verifier_out(output_elems, 0.0f);
    HipFloatBuffer d_restored_next(static_cast<size_t>(v_stride), 0.0f);
    HipFloatBuffer d_ref_prefix(static_cast<size_t>(accepted_rows) * static_cast<size_t>(v_stride), 0.0f);
    HipFloatBuffer d_ref_next(static_cast<size_t>(v_stride), 0.0f);
    HipFloatBuffer d_snapshots(static_cast<size_t>(verifier_len) * static_cast<size_t>(state_floats), -99.0f);

    ROCmGatedDeltaNet verifier_kernel(0);
    verifier_kernel.allocateGPUState(state_floats);
    verifier_kernel.bindVerifierStateCaptureWorkspace(d_snapshots.ptr, verifier_len, state_floats);
    ASSERT_TRUE(verifier_kernel.chunk_forward(
        d_Q.ptr, d_K.ptr, d_V.ptr, d_alpha.ptr, d_beta.ptr, d_A_log.ptr, d_dt_bias.ptr,
        d_verifier_out.ptr, nullptr,
        verifier_len, n_heads, d_k, d_v,
        /*chunk_size=*/64, /*use_qk_l2norm=*/true));
    ASSERT_TRUE(verifier_kernel.restoreVerifierStateCaptureRow(
        nullptr, accepted_rows - 1, nullptr));
    ASSERT_TRUE(verifier_kernel.recurrent_step(
        d_Q.ptr + static_cast<size_t>(continuation_row) * qk_stride,
        d_K.ptr + static_cast<size_t>(continuation_row) * qk_stride,
        d_V.ptr + static_cast<size_t>(continuation_row) * v_stride,
        d_alpha.ptr + static_cast<size_t>(continuation_row) * n_heads,
        d_beta.ptr + static_cast<size_t>(continuation_row) * n_heads,
        d_A_log.ptr,
        d_dt_bias.ptr,
        d_restored_next.ptr,
        nullptr,
        n_heads, d_k, d_v,
        /*use_qk_l2norm=*/true));
    checkHip(hipDeviceSynchronize(), "hipDeviceSynchronize(restored recurrence decode)");

    ROCmGatedDeltaNet ref_kernel(0);
    ref_kernel.allocateGPUState(state_floats);
    ASSERT_TRUE(ref_kernel.chunk_forward(
        d_Q.ptr, d_K.ptr, d_V.ptr, d_alpha.ptr, d_beta.ptr, d_A_log.ptr, d_dt_bias.ptr,
        d_ref_prefix.ptr, nullptr,
        accepted_rows, n_heads, d_k, d_v,
        /*chunk_size=*/64, /*use_qk_l2norm=*/true));
    ASSERT_TRUE(ref_kernel.recurrent_step(
        d_Q.ptr + static_cast<size_t>(continuation_row) * qk_stride,
        d_K.ptr + static_cast<size_t>(continuation_row) * qk_stride,
        d_V.ptr + static_cast<size_t>(continuation_row) * v_stride,
        d_alpha.ptr + static_cast<size_t>(continuation_row) * n_heads,
        d_beta.ptr + static_cast<size_t>(continuation_row) * n_heads,
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
    EXPECT_LT(diff.first, 2e-4f);
    EXPECT_LT(diff.second, 1e-4);
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

    ROCmShortConvolution verifier_kernel(0);
    verifier_kernel.allocateGPUState(state_floats);
    verifier_kernel.bindVerifierStateCaptureWorkspace(d_snapshots.ptr, verifier_len, state_floats);
    ASSERT_TRUE(verifier_kernel.forward(
        d_input.ptr,
        d_weight.ptr,
        d_bias.ptr,
        d_verifier_out.ptr,
        nullptr,
        verifier_len, channels, kernel_size,
        /*apply_silu=*/true));
    ASSERT_TRUE(verifier_kernel.restoreVerifierStateCaptureRow(
        nullptr, accepted_rows - 1, nullptr));
    ASSERT_TRUE(verifier_kernel.forward(
        d_input.ptr + static_cast<size_t>(continuation_row) * channels,
        d_weight.ptr,
        d_bias.ptr,
        d_restored_next.ptr,
        nullptr,
        1, channels, kernel_size,
        /*apply_silu=*/true));
    checkHip(hipDeviceSynchronize(), "hipDeviceSynchronize(restored short-conv decode)");

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

#endif
