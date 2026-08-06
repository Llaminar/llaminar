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

#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../utils/VerifierRowTestInventory.h"

#ifdef HAVE_ROCM
#include "kernels/rocm/gdn/ROCmGatedDeltaNet.h"
#include "kernels/rocm/gdn/ROCmShortConvolution.h"
#include <hip/hip_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <initializer_list>
#include <memory>
#include <sstream>
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

    /**
     * @brief Test owner for cache-style persistent ROCm GDN device state.
     *
     * The real hybrid cache plans these banks before graph construction.
     * Direct kernel tests use this owner to preserve that exact ownership and
     * one-shot binding contract while keeping allocations outside production
     * kernels and outside every captured region.
     */
    struct HipGDNStateOwner
    {
        template <typename Kernel>
        HipGDNStateOwner(
            Kernel &kernel,
            int primary_state_floats,
            int secondary_state_floats = 0,
            int request_capacity = 2)
            : primary_(
                  secondary_state_floats > 0 &&
                          secondary_state_floats != primary_state_floats
                      ? std::make_unique<HipFloatBuffer>(
                            static_cast<size_t>(primary_state_floats), 0.0f)
                      : nullptr),
              secondary_(
                  secondary_state_floats > 0 &&
                          secondary_state_floats != primary_state_floats
                      ? std::make_unique<HipFloatBuffer>(
                            static_cast<size_t>(secondary_state_floats), 0.0f)
                      : nullptr),
              requests_(std::make_unique<HipFloatBuffer>(
                  static_cast<size_t>(request_capacity) *
                      static_cast<size_t>(std::max(
                          primary_state_floats,
                          secondary_state_floats)),
                  0.0f))
        {
            if (primary_state_floats <= 0 || request_capacity <= 0)
                throw std::invalid_argument(
                    "HipGDNStateOwner requires positive state and request capacity");

            const GDNDeviceStateBinding binding{
                .primary_state = primary_ ? primary_->ptr : requests_->ptr,
                .primary_state_floats = primary_state_floats,
                .secondary_state = secondary_ ? secondary_->ptr : nullptr,
                .secondary_state_floats =
                    secondary_ ? secondary_state_floats : 0,
                .request_state_bank = requests_->ptr,
                .request_state_bank_floats = requests_->count,
                .request_capacity = request_capacity,
            };
            if (!kernel.bindDeviceState(binding))
                throw std::runtime_error(
                    "ROCm GDN kernel rejected explicit test-owned state");
        }

        /**
         * @brief Bind persistent test-owned scratch for in-place short-conv.
         */
        template <typename Kernel>
        void bindScratch(Kernel &kernel, int scratch_floats)
        {
            if (scratch_floats <= 0)
                throw std::invalid_argument(
                    "ROCm short-conv scratch must be positive");
            scratch_ = std::make_unique<HipFloatBuffer>(
                static_cast<size_t>(scratch_floats), 0.0f);
            kernel.bindScratchWorkspace(
                scratch_->ptr, static_cast<int>(scratch_->count));
        }

    private:
        std::unique_ptr<HipFloatBuffer> primary_;
        std::unique_ptr<HipFloatBuffer> secondary_;
        std::unique_ptr<HipFloatBuffer> requests_;
        std::unique_ptr<HipFloatBuffer> scratch_;
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
     * @brief Own a HIP graph executable across device-indexed state publication.
     *
     * Production MTP must restore an accepted GDN/short-conv verifier row into
     * live device state and then reuse the already-captured decode executable.
     * This helper preserves that executable for the complete regression and
     * destroys both HIP graph objects at scope exit.
     */
    struct HipCapturedGraph
    {
        hipGraph_t graph = nullptr;           ///< Captured graph definition.
        hipGraphExec_t executable = nullptr; ///< Instantiated reusable graph.

        template <typename RecordWork>
        HipCapturedGraph(hipStream_t stream, RecordWork &&record_work)
        {
            checkHip(
                hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed),
                "hipStreamBeginCapture(persistent recurrent graph)");
            bool recorded = false;
            {
                GraphCaptureGuard guard;
                recorded = record_work();
            }
            checkHip(
                hipStreamEndCapture(stream, &graph),
                "hipStreamEndCapture(persistent recurrent graph)");
            if (!recorded || !graph)
                throw std::runtime_error("ROCm recurrent wrapper rejected persistent graph capture");
            checkHip(
                hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
                "hipGraphInstantiate(persistent recurrent graph)");
        }

        ~HipCapturedGraph()
        {
            if (executable)
                (void)hipGraphExecDestroy(executable);
            if (graph)
                (void)hipGraphDestroy(graph);
        }

        HipCapturedGraph(const HipCapturedGraph &) = delete;
        HipCapturedGraph &operator=(const HipCapturedGraph &) = delete;

        /// @brief Enqueue one replay of the original captured executable.
        void launch(hipStream_t stream) const
        {
            checkHip(
                hipGraphLaunch(executable, stream),
                "hipGraphLaunch(persistent recurrent graph)");
        }
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

    /// @brief Strict row-distribution metrics for grouped verifier equivalence checks.
    struct StrictVectorMetrics
    {
        float max_abs = 0.0f;         ///< Largest elementwise absolute difference.
        double relative_l2 = 0.0;     ///< L2(actual-expected) normalized by expected.
        double cosine = 1.0;          ///< Cosine similarity between the two vectors.
        double symmetric_kl = 0.0;    ///< Symmetric KL after stable softmax normalization.
    };

    /**
     * @brief Computes strict numerical metrics over one row or state slice.
     *
     * MTP verifier rows must be decode-equivalent, not merely close enough for
     * top-token parity.  This helper mirrors the full-model parity checks by
     * combining L2, cosine, and a softmaxed symmetric-KL view so tiny but
     * systematic row drift is caught before it amplifies through later layers.
     */
    StrictVectorMetrics strictVectorMetrics(
        const std::vector<float> &actual,
        const std::vector<float> &expected,
        size_t offset,
        size_t count)
    {
        StrictVectorMetrics metrics;
        if (count == 0)
            return metrics;

        double dot = 0.0;
        double sum_sq_actual = 0.0;
        double sum_sq_expected = 0.0;
        double sum_sq_diff = 0.0;
        float max_actual = actual[offset];
        float max_expected = expected[offset];

        for (size_t i = 0; i < count; ++i)
        {
            const float a = actual[offset + i];
            const float b = expected[offset + i];
            const float diff = a - b;
            metrics.max_abs = std::max(metrics.max_abs, std::abs(diff));
            dot += static_cast<double>(a) * static_cast<double>(b);
            sum_sq_actual += static_cast<double>(a) * static_cast<double>(a);
            sum_sq_expected += static_cast<double>(b) * static_cast<double>(b);
            sum_sq_diff += static_cast<double>(diff) * static_cast<double>(diff);
            max_actual = std::max(max_actual, a);
            max_expected = std::max(max_expected, b);
        }

        metrics.relative_l2 = std::sqrt(sum_sq_diff / std::max(sum_sq_expected, 1e-30));
        const double denom = std::sqrt(sum_sq_actual * sum_sq_expected);
        metrics.cosine = denom > 0.0 ? dot / denom : (sum_sq_actual == sum_sq_expected ? 1.0 : 0.0);

        std::vector<double> actual_prob(count);
        std::vector<double> expected_prob(count);
        double actual_sum = 0.0;
        double expected_sum = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            actual_prob[i] = std::exp(static_cast<double>(actual[offset + i] - max_actual));
            expected_prob[i] = std::exp(static_cast<double>(expected[offset + i] - max_expected));
            actual_sum += actual_prob[i];
            expected_sum += expected_prob[i];
        }

        constexpr double eps = 1e-30;
        double actual_to_expected = 0.0;
        double expected_to_actual = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            const double p = std::max(actual_prob[i] / actual_sum, eps);
            const double q = std::max(expected_prob[i] / expected_sum, eps);
            actual_to_expected += p * std::log(p / q);
            expected_to_actual += q * std::log(q / p);
        }
        metrics.symmetric_kl = 0.5 * (actual_to_expected + expected_to_actual);
        return metrics;
    }

    /// @brief Asserts strict decode-equivalence and prints all metrics on failure.
    void expectStrictEquivalent(
        const char *label,
        const std::vector<float> &actual,
        const std::vector<float> &expected,
        size_t offset,
        size_t count,
        float max_abs_threshold = 1e-6f,
        double relative_l2_threshold = 1e-7,
        double min_cosine = 0.9999999,
        double max_symmetric_kl = 1e-10)
    {
        const StrictVectorMetrics metrics =
            strictVectorMetrics(actual, expected, offset, count);
        EXPECT_LE(metrics.max_abs, max_abs_threshold)
            << label << " max_abs=" << metrics.max_abs;
        EXPECT_LE(metrics.relative_l2, relative_l2_threshold)
            << label << " relative_l2=" << metrics.relative_l2;
        EXPECT_GE(metrics.cosine, min_cosine)
            << label << " cosine=" << metrics.cosine;
        EXPECT_LE(metrics.symmetric_kl, max_symmetric_kl)
            << label << " symmetric_kl=" << metrics.symmetric_kl;
    }

    /// @brief Returns the raw IEEE-754 bit pattern for an FP32 value without changing it.
    uint32_t fp32BitsForByteExactCheck(float value)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    /**
     * @brief Asserts byte-for-byte FP32 equality over a grouped verifier slice.
     *
     * Grouped MTP verifier rows are allowed to be faster than serial decode,
     * but they are not allowed to be merely numerically close.  Publication
     * copies captured GDN/short-conv state into the live continuation state, so
     * one changed mantissa bit can become a different routed expert or sampled
     * token several layers later.  This helper keeps the older metric-based
     * diagnostics available while making the pass/fail criterion raw equality.
     */
    void expectByteExactEquivalent(
        const char *label,
        const std::vector<float> &actual,
        const std::vector<float> &expected,
        size_t offset,
        size_t count)
    {
        ASSERT_LE(offset + count, actual.size()) << label << " actual slice out of range";
        ASSERT_LE(offset + count, expected.size()) << label << " expected slice out of range";
        if (count == 0)
            return;

        const auto *actual_bytes =
            reinterpret_cast<const unsigned char *>(actual.data() + offset);
        const auto *expected_bytes =
            reinterpret_cast<const unsigned char *>(expected.data() + offset);
        if (std::memcmp(actual_bytes, expected_bytes, count * sizeof(float)) == 0)
            return;

        const StrictVectorMetrics metrics =
            strictVectorMetrics(actual, expected, offset, count);
        for (size_t i = 0; i < count; ++i)
        {
            const float actual_value = actual[offset + i];
            const float expected_value = expected[offset + i];
            if (fp32BitsForByteExactCheck(actual_value) == fp32BitsForByteExactCheck(expected_value))
                continue;

            std::ostringstream message;
            message << label
                    << " first byte mismatch at local_index=" << i
                    << " absolute_index=" << (offset + i)
                    << " actual=" << actual_value
                    << " expected=" << expected_value
                    << " actual_bits=0x" << std::hex << std::setw(8) << std::setfill('0')
                    << fp32BitsForByteExactCheck(actual_value)
                    << " expected_bits=0x" << std::setw(8)
                    << fp32BitsForByteExactCheck(expected_value)
                    << std::dec
                    << " max_abs=" << metrics.max_abs
                    << " relative_l2=" << metrics.relative_l2
                    << " cosine=" << metrics.cosine
                    << " symmetric_kl=" << metrics.symmetric_kl;
            ADD_FAILURE() << message.str();
            return;
        }
    }

    /**
     * @brief Compares an observed device-owned live state with one capture row.
     *
     * Production publication never writes or consults a host mirror. Tests may
     * synchronize the explicit stream and export the resident state afterward
     * as a diagnostic observation. This helper then proves that the device copy
     * selected exactly the requested post-row snapshot.
     */
    void expectPublishedDeviceStateMatchesSnapshotRow(
        const std::vector<float> &published_device_state,
        const HipFloatBuffer &snapshots,
        int row,
        int state_floats)
    {
        ASSERT_GE(row, 0);
        ASSERT_GT(state_floats, 0);
        ASSERT_EQ(published_device_state.size(), static_cast<size_t>(state_floats));
        const std::vector<float> snapshot_host = snapshots.toHost();
        const size_t offset =
            static_cast<size_t>(row) * static_cast<size_t>(state_floats);
        ASSERT_LE(offset + static_cast<size_t>(state_floats),
                  snapshot_host.size());
        for (int i = 0; i < state_floats; ++i)
        {
            EXPECT_FLOAT_EQ(
                published_device_state[static_cast<size_t>(i)],
                snapshot_host[offset + static_cast<size_t>(i)])
                << "state_float=" << i << " row=" << row;
        }
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

/**
 * @brief Proves HIP request batching is byte-identical to independent decode.
 *
 * Unequal real lengths and hostile padding exercise the resident length row.
 * Real prefill outputs must match independent one-request production kernels,
 * and a following grouped decode row must also match, proving that both
 * short-conv and GDN committed request-local state at the real terminal row.
 */
TEST(Test__ROCmGDNPaddedRealLength, RequestBatchedUnequalLengthsPreserveByteExactContinuation)
{
    if (!hasROCm())
        GTEST_SKIP() << "No ROCm device available";
    checkHip(hipSetDevice(0), "hipSetDevice");
    HipStreamHandle stream;

    constexpr int request_count = 2;
    constexpr int request_seq_len = 6;
    constexpr int total_rows = request_count * request_seq_len;
    constexpr int real_lengths[request_count] = {6, 4};

    auto extract_rows = [](
                            const std::vector<float> &matrix,
                            int first_row,
                            int row_count,
                            int width)
    {
        const size_t begin =
            static_cast<size_t>(first_row) * static_cast<size_t>(width);
        const size_t count =
            static_cast<size_t>(row_count) * static_cast<size_t>(width);
        return std::vector<float>(
            matrix.begin() + static_cast<std::ptrdiff_t>(begin),
            matrix.begin() + static_cast<std::ptrdiff_t>(begin + count));
    };

    HipIntBuffer request_lengths({real_lengths[0], real_lengths[1]});
    HipIntBuffer one_row_lengths({1, 1});

    /*
     * Dense Qwen 3.6 uses 16 source Q/K heads and 48 value heads at width
     * 128, so the pre-deinterleave ShortConv row is
     * 16*128 + 16*128 + 48*128 = 10,240 channels.  Keeping the production
     * width here catches request-offset and scratch-capacity defects hidden by
     * toy channel counts.
     */
    constexpr int channels = 10240;
    constexpr int kernel_size = 4;
    std::vector<float> conv_input(
        static_cast<size_t>(total_rows) * channels);
    for (int request = 0; request < request_count; ++request)
    {
        for (int row = 0; row < request_seq_len; ++row)
        {
            for (int channel = 0; channel < channels; ++channel)
            {
                const size_t index =
                    (static_cast<size_t>(request) * request_seq_len + row) * channels +
                    channel;
                conv_input[index] =
                    row < real_lengths[request]
                        ? 0.003f * static_cast<float>((request + 1) * (row + 2)) +
                              0.0002f * static_cast<float>((channel % 13) - 6)
                        : 500.0f + static_cast<float>(17 * row + channel);
            }
        }
    }
    std::vector<float> conv_decode(
        static_cast<size_t>(request_count) * channels);
    for (int request = 0; request < request_count; ++request)
    {
        for (int channel = 0; channel < channels; ++channel)
        {
            conv_decode[static_cast<size_t>(request) * channels + channel] =
                -0.007f * static_cast<float>(request + 1) +
                0.0003f * static_cast<float>((channel % 9) - 4);
        }
    }
    const std::vector<float> conv_weights =
        makeShortConvWeights(channels, kernel_size);
    std::vector<float> conv_bias(channels);
    for (int channel = 0; channel < channels; ++channel)
        conv_bias[channel] = 0.001f * static_cast<float>((channel % 7) - 3);

    HipFloatBuffer d_conv_inout(conv_input);
    HipFloatBuffer d_conv_decode_inout(conv_decode);
    HipFloatBuffer d_conv_scratch(conv_input.size(), -99.0f);
    HipFloatBuffer d_conv_weights(conv_weights);
    HipFloatBuffer d_conv_bias(conv_bias);
    ROCmShortConvolution grouped_conv(0);
    HipGDNStateOwner grouped_conv_state_owner(
        grouped_conv, channels * (kernel_size - 1));
    grouped_conv.setGPUStream(stream.stream);
    grouped_conv.bindScratchWorkspace(
        d_conv_scratch.ptr,
        static_cast<int>(d_conv_scratch.count));

    ASSERT_TRUE(grouped_conv.forwardBatchedRequestsWithDeviceSeqLens(
        d_conv_inout.ptr, d_conv_weights.ptr, d_conv_bias.ptr,
        d_conv_inout.ptr, /*conv_state=*/nullptr,
        total_rows, request_count, request_seq_len,
        channels, kernel_size, request_lengths.ptr,
        /*apply_silu=*/true));
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(short-conv prefill)");
    const std::vector<float> grouped_conv_prefill = d_conv_inout.toHost();

    ASSERT_TRUE(grouped_conv.forwardBatchedRequestsWithDeviceSeqLens(
        d_conv_decode_inout.ptr, d_conv_weights.ptr, d_conv_bias.ptr,
        d_conv_decode_inout.ptr, /*conv_state=*/nullptr,
        /*seq_len=*/request_count,
        request_count, /*request_seq_len=*/1,
        channels, kernel_size, one_row_lengths.ptr,
        /*apply_silu=*/true));
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(short-conv decode)");
    const std::vector<float> grouped_conv_decode = d_conv_decode_inout.toHost();

    for (int request = 0; request < request_count; ++request)
    {
        const std::vector<float> scalar_input = extract_rows(
            conv_input, request * request_seq_len,
            real_lengths[request], channels);
        const std::vector<float> scalar_decode_input = extract_rows(
            conv_decode, request, /*row_count=*/1, channels);
        HipFloatBuffer d_scalar_input(scalar_input);
        HipFloatBuffer d_scalar_output(scalar_input.size(), -77.0f);
        HipFloatBuffer d_scalar_decode(scalar_decode_input);
        HipFloatBuffer d_scalar_decode_output(channels, -77.0f);
        ROCmShortConvolution scalar_conv(0);
        HipGDNStateOwner scalar_conv_state_owner(
            scalar_conv, channels * (kernel_size - 1));
        scalar_conv.setGPUStream(stream.stream);

        ASSERT_TRUE(scalar_conv.forward(
            d_scalar_input.ptr, d_conv_weights.ptr, d_conv_bias.ptr,
            d_scalar_output.ptr, /*conv_state=*/nullptr,
            real_lengths[request], channels, kernel_size,
            /*apply_silu=*/true));
        ASSERT_TRUE(scalar_conv.forward(
            d_scalar_decode.ptr, d_conv_weights.ptr, d_conv_bias.ptr,
            d_scalar_decode_output.ptr, /*conv_state=*/nullptr,
            /*seq_len=*/1, channels, kernel_size,
            /*apply_silu=*/true));
        checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(short-conv oracle)");

        const std::vector<float> grouped_real = extract_rows(
            grouped_conv_prefill,
            request * request_seq_len,
            real_lengths[request], channels);
        expectByteExactEquivalent(
            "ROCm request-batched short-conv prefill",
            grouped_real, d_scalar_output.toHost(),
            /*offset=*/0, grouped_real.size());
        const std::vector<float> grouped_decode_row = extract_rows(
            grouped_conv_decode, request, /*row_count=*/1, channels);
        expectByteExactEquivalent(
            "ROCm request-batched short-conv continuation",
            grouped_decode_row, d_scalar_decode_output.toHost(),
            /*offset=*/0, grouped_decode_row.size());
        const std::vector<float> padded_rows = extract_rows(
            grouped_conv_prefill,
            request * request_seq_len + real_lengths[request],
            request_seq_len - real_lengths[request], channels);
        EXPECT_EQ(maxAbsSpan(padded_rows, 0, padded_rows.size()), 0.0f)
            << "ROCm short-conv must zero padded rows for request " << request;
    }

    constexpr int n_heads = 48;
    constexpr int d_k = 128;
    constexpr int d_v = 128;
    constexpr int qk_width = n_heads * d_k;
    constexpr int v_width = n_heads * d_v;
    constexpr int gate_width = n_heads;
    constexpr int gdn_state_floats = n_heads * d_k * d_v;
    std::vector<float> Q(static_cast<size_t>(total_rows) * qk_width);
    std::vector<float> K(Q.size());
    std::vector<float> V(static_cast<size_t>(total_rows) * v_width);
    std::vector<float> alpha(static_cast<size_t>(total_rows) * gate_width);
    std::vector<float> beta(alpha.size());
    for (int request = 0; request < request_count; ++request)
    {
        for (int row = 0; row < request_seq_len; ++row)
        {
            const bool real = row < real_lengths[request];
            const int flat_row = request * request_seq_len + row;
            for (int col = 0; col < qk_width; ++col)
            {
                const float base =
                    0.0007f * static_cast<float>((flat_row + 1) * ((col % 17) - 8));
                Q[static_cast<size_t>(flat_row) * qk_width + col] =
                    real ? base + 0.013f : 300.0f + static_cast<float>(col);
                K[static_cast<size_t>(flat_row) * qk_width + col] =
                    real ? base * 0.73f - 0.009f : -400.0f - static_cast<float>(col);
            }
            for (int col = 0; col < v_width; ++col)
            {
                V[static_cast<size_t>(flat_row) * v_width + col] =
                    real
                        ? 0.0011f * static_cast<float>((flat_row + 2) * ((col % 11) - 5))
                        : 250.0f + static_cast<float>(col);
            }
            for (int head = 0; head < n_heads; ++head)
            {
                const size_t gate_index =
                    static_cast<size_t>(flat_row) * gate_width + head;
                alpha[gate_index] =
                    real
                        ? -0.3f + 0.02f * row +
                              0.0005f * static_cast<float>((head % 7) - 3)
                        : 20.0f;
                beta[gate_index] =
                    real
                        ? 0.15f - 0.01f * row +
                              0.0004f * static_cast<float>((head % 5) - 2)
                        : -20.0f;
            }
        }
    }

    std::vector<float> decode_Q(static_cast<size_t>(request_count) * qk_width);
    std::vector<float> decode_K(decode_Q.size());
    std::vector<float> decode_V(static_cast<size_t>(request_count) * v_width);
    std::vector<float> decode_alpha(
        static_cast<size_t>(request_count) * gate_width);
    std::vector<float> decode_beta(
        static_cast<size_t>(request_count) * gate_width);
    for (int request = 0; request < request_count; ++request)
    {
        for (int col = 0; col < qk_width; ++col)
        {
            decode_Q[static_cast<size_t>(request) * qk_width + col] =
                0.0009f * static_cast<float>((request + 2) * ((col % 19) - 9));
            decode_K[static_cast<size_t>(request) * qk_width + col] =
                -0.0008f * static_cast<float>((request + 1) * ((col % 13) - 6));
        }
        for (int col = 0; col < v_width; ++col)
        {
            decode_V[static_cast<size_t>(request) * v_width + col] =
                0.0013f * static_cast<float>((request + 1) * ((col % 7) - 3));
        }
        for (int head = 0; head < n_heads; ++head)
        {
            const size_t gate_index =
                static_cast<size_t>(request) * gate_width + head;
            decode_alpha[gate_index] =
                -0.21f + 0.01f * request +
                0.0003f * static_cast<float>((head % 7) - 3);
            decode_beta[gate_index] =
                0.12f - 0.02f * request +
                0.0002f * static_cast<float>((head % 5) - 2);
        }
    }
    const std::vector<float> A_log(n_heads, -0.45f);
    const std::vector<float> dt_bias(n_heads, 0.08f);

    HipFloatBuffer d_Q(Q);
    HipFloatBuffer d_K(K);
    HipFloatBuffer d_V(V);
    HipFloatBuffer d_alpha(alpha);
    HipFloatBuffer d_beta(beta);
    HipFloatBuffer d_A_log(A_log);
    HipFloatBuffer d_dt_bias(dt_bias);
    HipFloatBuffer d_gdn_output(static_cast<size_t>(total_rows) * v_width, -55.0f);
    HipFloatBuffer d_decode_Q(decode_Q);
    HipFloatBuffer d_decode_K(decode_K);
    HipFloatBuffer d_decode_V(decode_V);
    HipFloatBuffer d_decode_alpha(decode_alpha);
    HipFloatBuffer d_decode_beta(decode_beta);
    HipFloatBuffer d_gdn_decode_output(static_cast<size_t>(request_count) * v_width, -55.0f);
    ROCmGatedDeltaNet grouped_gdn(0);
    HipGDNStateOwner grouped_gdn_state_owner(
        grouped_gdn, gdn_state_floats);
    grouped_gdn.setGPUStream(stream.stream);

    ASSERT_TRUE(grouped_gdn.chunkForwardBatchedRequestsWithDeviceSeqLens(
        d_Q.ptr, d_K.ptr, d_V.ptr,
        d_alpha.ptr, d_beta.ptr,
        d_A_log.ptr, d_dt_bias.ptr,
        d_gdn_output.ptr, /*state=*/nullptr,
        total_rows, request_count, request_seq_len,
        n_heads, d_k, d_v,
        /*chunk_size=*/64, /*use_qk_l2norm=*/true,
        request_lengths.ptr));
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(GDN prefill)");
    const std::vector<float> grouped_gdn_prefill = d_gdn_output.toHost();
    std::vector<float> grouped_gdn_prefill_state(
        static_cast<size_t>(request_count) * gdn_state_floats);
    ASSERT_TRUE(grouped_gdn.exportRequestStateBank(
        grouped_gdn_prefill_state.data(),
        request_count,
        gdn_state_floats,
        stream.stream));
    checkHip(
        hipStreamSynchronize(stream.stream),
        "hipStreamSynchronize(GDN request-state observation)");

    ASSERT_TRUE(grouped_gdn.chunkForwardBatchedRequestsWithDeviceSeqLens(
        d_decode_Q.ptr, d_decode_K.ptr, d_decode_V.ptr,
        d_decode_alpha.ptr, d_decode_beta.ptr,
        d_A_log.ptr, d_dt_bias.ptr,
        d_gdn_decode_output.ptr, /*state=*/nullptr,
        /*seq_len=*/request_count,
        request_count, /*request_seq_len=*/1,
        n_heads, d_k, d_v,
        /*chunk_size=*/64, /*use_qk_l2norm=*/true,
        one_row_lengths.ptr));
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(GDN decode)");
    const std::vector<float> grouped_gdn_decode = d_gdn_decode_output.toHost();

    for (int request = 0; request < request_count; ++request)
    {
        HipFloatBuffer d_scalar_Q(extract_rows(
            Q, request * request_seq_len, real_lengths[request], qk_width));
        HipFloatBuffer d_scalar_K(extract_rows(
            K, request * request_seq_len, real_lengths[request], qk_width));
        HipFloatBuffer d_scalar_V(extract_rows(
            V, request * request_seq_len, real_lengths[request], v_width));
        HipFloatBuffer d_scalar_alpha(extract_rows(
            alpha, request * request_seq_len, real_lengths[request], gate_width));
        HipFloatBuffer d_scalar_beta(extract_rows(
            beta, request * request_seq_len, real_lengths[request], gate_width));
        HipFloatBuffer d_scalar_output(
            static_cast<size_t>(real_lengths[request]) * v_width, -33.0f);
        HipFloatBuffer d_scalar_decode_Q(extract_rows(decode_Q, request, 1, qk_width));
        HipFloatBuffer d_scalar_decode_K(extract_rows(decode_K, request, 1, qk_width));
        HipFloatBuffer d_scalar_decode_V(extract_rows(decode_V, request, 1, v_width));
        HipFloatBuffer d_scalar_decode_alpha(extract_rows(decode_alpha, request, 1, gate_width));
        HipFloatBuffer d_scalar_decode_beta(extract_rows(decode_beta, request, 1, gate_width));
        HipFloatBuffer d_scalar_decode_output(v_width, -33.0f);
        ROCmGatedDeltaNet scalar_gdn(0);
        HipGDNStateOwner scalar_gdn_state_owner(
            scalar_gdn, gdn_state_floats);
        scalar_gdn.setGPUStream(stream.stream);

        ASSERT_TRUE(scalar_gdn.chunk_forward(
            d_scalar_Q.ptr, d_scalar_K.ptr, d_scalar_V.ptr,
            d_scalar_alpha.ptr, d_scalar_beta.ptr,
            d_A_log.ptr, d_dt_bias.ptr,
            d_scalar_output.ptr, /*state=*/nullptr,
            real_lengths[request], n_heads, d_k, d_v,
            /*chunk_size=*/64, /*use_qk_l2norm=*/true));
        std::vector<float> scalar_gdn_prefill_state(gdn_state_floats);
        ASSERT_TRUE(scalar_gdn.exportState(
            scalar_gdn_prefill_state.data(),
            /*dst_device=*/nullptr,
            stream.stream));
        ASSERT_TRUE(scalar_gdn.recurrent_step(
            d_scalar_decode_Q.ptr, d_scalar_decode_K.ptr, d_scalar_decode_V.ptr,
            d_scalar_decode_alpha.ptr, d_scalar_decode_beta.ptr,
            d_A_log.ptr, d_dt_bias.ptr,
            d_scalar_decode_output.ptr, /*state=*/nullptr,
            n_heads, d_k, d_v,
            /*use_qk_l2norm=*/true));
        checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(GDN oracle)");

        const size_t grouped_state_begin =
            static_cast<size_t>(request) * gdn_state_floats;
        const std::vector<float> grouped_request_prefill_state(
            grouped_gdn_prefill_state.begin() +
                static_cast<std::ptrdiff_t>(grouped_state_begin),
            grouped_gdn_prefill_state.begin() +
                static_cast<std::ptrdiff_t>(
                    grouped_state_begin + gdn_state_floats));
        expectByteExactEquivalent(
            "ROCm request-batched GDN prefill state",
            grouped_request_prefill_state,
            scalar_gdn_prefill_state,
            /*offset=*/0,
            static_cast<size_t>(gdn_state_floats));

        /*
         * Run the grouped decode entry point with one request from the same
         * imported state.  This separates arithmetic drift in the grouped API
         * itself from request-offset defects that only appear when grid Z is
         * larger than one.
         */
        HipFloatBuffer d_single_grouped_decode_output(v_width, -31.0f);
        ROCmGatedDeltaNet single_grouped_gdn(0);
        single_grouped_gdn.setGPUStream(stream.stream);
        HipGDNStateOwner single_grouped_gdn_state_owner(single_grouped_gdn, gdn_state_floats);
        ASSERT_TRUE(single_grouped_gdn.importState(
            scalar_gdn_prefill_state.data(),
            /*src_device=*/nullptr,
            stream.stream));
        ASSERT_TRUE(single_grouped_gdn.chunkForwardBatchedRequestsWithDeviceSeqLens(
            d_scalar_decode_Q.ptr, d_scalar_decode_K.ptr, d_scalar_decode_V.ptr,
            d_scalar_decode_alpha.ptr, d_scalar_decode_beta.ptr,
            d_A_log.ptr, d_dt_bias.ptr,
            d_single_grouped_decode_output.ptr, /*state=*/nullptr,
            /*seq_len=*/1, /*request_count=*/1, /*request_seq_len=*/1,
            n_heads, d_k, d_v,
            /*chunk_size=*/64, /*use_qk_l2norm=*/true,
            one_row_lengths.ptr));
        checkHip(
            hipStreamSynchronize(stream.stream),
            "hipStreamSynchronize(single-request grouped GDN decode)");
        expectByteExactEquivalent(
            "ROCm single-request grouped GDN continuation",
            d_single_grouped_decode_output.toHost(),
            d_scalar_decode_output.toHost(),
            /*offset=*/0,
            static_cast<size_t>(v_width));

        const std::vector<float> grouped_real = extract_rows(
            grouped_gdn_prefill,
            request * request_seq_len,
            real_lengths[request], v_width);
        expectByteExactEquivalent(
            "ROCm request-batched GDN prefill",
            grouped_real, d_scalar_output.toHost(),
            /*offset=*/0, grouped_real.size());
        const std::vector<float> grouped_decode_row = extract_rows(
            grouped_gdn_decode, request, /*row_count=*/1, v_width);
        expectByteExactEquivalent(
            "ROCm request-batched GDN continuation",
            grouped_decode_row, d_scalar_decode_output.toHost(),
            /*offset=*/0, grouped_decode_row.size());
        const std::vector<float> padded_rows = extract_rows(
            grouped_gdn_prefill,
            request * request_seq_len + real_lengths[request],
            request_seq_len - real_lengths[request], v_width);
        EXPECT_EQ(maxAbsSpan(padded_rows, 0, padded_rows.size()), 0.0f)
            << "ROCm GDN must zero padded rows for request " << request;
    }
}

TEST(
    Test__ROCmGDNPaddedRealLength,
    StateBankGeometryLookupDoesNotMutateStableBindingsUnderCaptureGuard)
{
    if (!hasROCm())
        GTEST_SKIP() << "No ROCm device available";
    checkHip(hipSetDevice(0), "hipSetDevice");

    HipStreamHandle stream;

    constexpr int local_heads = 1;
    constexpr int full_heads = 2;
    constexpr int d_k = 64;
    constexpr int d_v = 64;
    constexpr int seq_len = 2;
    constexpr int local_recurrence_state = local_heads * d_k * d_v;
    constexpr int full_recurrence_state = full_heads * d_k * d_v;
    constexpr int full_qk_stride = full_heads * d_k;
    constexpr int full_v_stride = full_heads * d_v;

    ROCmGatedDeltaNet recurrence(0);
    recurrence.setGPUStream(stream.stream);
    HipGDNStateOwner recurrence_state_owner(
        recurrence, local_recurrence_state, full_recurrence_state);
    ASSERT_TRUE(recurrence.isGPUStateReady(local_recurrence_state));
    ASSERT_TRUE(recurrence.isGPUStateReady(full_recurrence_state));
    ASSERT_TRUE(recurrence.isGPUStateReady(local_recurrence_state))
        << "Full decode-state handoff must not discard the local prefill state slot";
    ASSERT_EQ(recurrence.stateBytes(), static_cast<size_t>(local_recurrence_state) * sizeof(float));

    HipFloatBuffer d_q(static_cast<size_t>(seq_len) * full_qk_stride, 0.01f);
    HipFloatBuffer d_kbuf(static_cast<size_t>(seq_len) * full_qk_stride, 0.02f);
    HipFloatBuffer d_vbuf(static_cast<size_t>(seq_len) * full_v_stride, 0.03f);
    HipFloatBuffer d_alpha(static_cast<size_t>(seq_len) * full_heads, 0.2f);
    HipFloatBuffer d_beta(static_cast<size_t>(seq_len) * full_heads, -0.1f);
    HipFloatBuffer d_A_log(static_cast<size_t>(full_heads), -0.5f);
    HipFloatBuffer d_dt_bias(static_cast<size_t>(full_heads), 0.1f);
    HipFloatBuffer d_recurrence_out(static_cast<size_t>(seq_len) * full_v_stride, 0.0f);

    {
        GraphCaptureGuard guard;
        ASSERT_TRUE(recurrence.chunk_forward(
            d_q.ptr, d_kbuf.ptr, d_vbuf.ptr, d_alpha.ptr, d_beta.ptr, d_A_log.ptr, d_dt_bias.ptr,
            d_recurrence_out.ptr, nullptr,
            seq_len, local_heads, d_k, d_v,
            /*chunk_size=*/64, /*use_qk_l2norm=*/false));
    }
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(local recurrence slot)");
    EXPECT_EQ(recurrence.stateBytes(), static_cast<size_t>(local_recurrence_state) * sizeof(float));

    {
        GraphCaptureGuard guard;
        ASSERT_TRUE(recurrence.recurrent_step(
            d_q.ptr, d_kbuf.ptr, d_vbuf.ptr, d_alpha.ptr, d_beta.ptr, d_A_log.ptr, d_dt_bias.ptr,
            d_recurrence_out.ptr, nullptr,
            full_heads, d_k, d_v,
            /*use_qk_l2norm=*/false));
    }
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(full recurrence slot)");
    EXPECT_EQ(
        recurrence.stateBytes(),
        static_cast<size_t>(local_recurrence_state) * sizeof(float));
    EXPECT_EQ(
        recurrence.largestStateBytes(),
        static_cast<size_t>(full_recurrence_state) * sizeof(float));

    constexpr int local_channels = 64;
    constexpr int full_channels = 128;
    constexpr int kernel_size = 4;
    constexpr int local_conv_state = local_channels * (kernel_size - 1);
    constexpr int full_conv_state = full_channels * (kernel_size - 1);

    ROCmShortConvolution conv(0);
    conv.setGPUStream(stream.stream);
    HipGDNStateOwner conv_state_owner(
        conv, local_conv_state, full_conv_state);
    ASSERT_EQ(conv.stateBytes(), static_cast<size_t>(local_conv_state) * sizeof(float));
    ASSERT_EQ(
        conv.largestStateBytes(),
        static_cast<size_t>(full_conv_state) * sizeof(float));

    const auto weights = makeShortConvWeights(full_channels, kernel_size);
    const auto bias = makeBias(full_channels);
    HipFloatBuffer d_input(static_cast<size_t>(seq_len) * full_channels, 0.04f);
    HipFloatBuffer d_weight(weights);
    HipFloatBuffer d_bias(bias);
    HipFloatBuffer d_conv_out(static_cast<size_t>(seq_len) * full_channels, 0.0f);

    {
        GraphCaptureGuard guard;
        ASSERT_TRUE(conv.forward(
            d_input.ptr, d_weight.ptr, d_bias.ptr,
            d_conv_out.ptr, nullptr,
            seq_len, local_channels, kernel_size,
            /*apply_silu=*/true));
    }
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(local short-conv slot)");
    EXPECT_EQ(conv.stateBytes(), static_cast<size_t>(local_conv_state) * sizeof(float));

    {
        GraphCaptureGuard guard;
        ASSERT_TRUE(conv.forward(
            d_input.ptr, d_weight.ptr, d_bias.ptr,
            d_conv_out.ptr, nullptr,
            seq_len, full_channels, kernel_size,
            /*apply_silu=*/true));
    }
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(full short-conv slot)");
    EXPECT_EQ(
        conv.stateBytes(),
        static_cast<size_t>(local_conv_state) * sizeof(float));
    EXPECT_EQ(
        conv.largestStateBytes(),
        static_cast<size_t>(full_conv_state) * sizeof(float));
}

/**
 * @brief Proves HIP replay republishes LocalTP full state to request zero.
 *
 * The captured executable performs local stateful work followed by a
 * local-to-full handoff. Resetting all banks and replaying that executable must
 * rebuild both the full live bank and the packed request view without a host
 * validity field being updated by the C++ wrappers.
 */
TEST(
    Test__ROCmGDNPaddedRealLength,
    CapturedLocalToFullHandoffPublishesRequestBanksOnEveryReplay)
{
    if (!hasROCm())
        GTEST_SKIP() << "No ROCm device available";
    checkHip(hipSetDevice(0), "hipSetDevice");

    HipStreamHandle stream;

    constexpr int local_heads = 1;
    constexpr int full_heads = 2;
    constexpr int d_k = 64;
    constexpr int d_v = 64;
    constexpr int seq_len = 2;
    constexpr int local_qk_stride = local_heads * d_k;
    constexpr int local_v_stride = local_heads * d_v;
    constexpr int local_recurrence_state = local_heads * d_k * d_v;
    constexpr int full_recurrence_state = full_heads * d_k * d_v;

    ROCmGatedDeltaNet recurrence(0);
    recurrence.setGPUStream(stream.stream);
    HipGDNStateOwner recurrence_state_owner(
        recurrence,
        local_recurrence_state,
        full_recurrence_state);

    HipFloatBuffer d_q(
        static_cast<size_t>(seq_len) * local_qk_stride,
        0.01f);
    HipFloatBuffer d_kbuf(
        static_cast<size_t>(seq_len) * local_qk_stride,
        0.02f);
    HipFloatBuffer d_vbuf(
        static_cast<size_t>(seq_len) * local_v_stride,
        0.03f);
    HipFloatBuffer d_alpha(
        static_cast<size_t>(seq_len) * local_heads,
        0.2f);
    HipFloatBuffer d_beta(
        static_cast<size_t>(seq_len) * local_heads,
        -0.1f);
    HipFloatBuffer d_A_log(static_cast<size_t>(local_heads), -0.5f);
    HipFloatBuffer d_dt_bias(static_cast<size_t>(local_heads), 0.1f);
    HipFloatBuffer d_recurrence_out(
        static_cast<size_t>(seq_len) * local_v_stride,
        0.0f);
    const auto full_recurrence_payload =
        makeInitialState(static_cast<size_t>(full_recurrence_state), 0.00031f);
    HipFloatBuffer d_full_recurrence(full_recurrence_payload);

    HipCapturedGraph recurrence_graph(
        stream.stream,
        [&]()
        {
            return recurrence.chunk_forward(
                       d_q.ptr,
                       d_kbuf.ptr,
                       d_vbuf.ptr,
                       d_alpha.ptr,
                       d_beta.ptr,
                       d_A_log.ptr,
                       d_dt_bias.ptr,
                       d_recurrence_out.ptr,
                       nullptr,
                       seq_len,
                       local_heads,
                       d_k,
                       d_v,
                       /*chunk_size=*/64,
                       /*use_qk_l2norm=*/false) &&
                   recurrence.importStateForSize(
                       full_recurrence_state,
                       /*src_host=*/nullptr,
                       d_full_recurrence.ptr,
                       stream.stream);
        });

    ASSERT_TRUE(recurrence.resetGPUState(stream.stream));
    recurrence_graph.launch(stream.stream);
    std::vector<float> recurrence_live(
        static_cast<size_t>(full_recurrence_state));
    std::vector<float> recurrence_request(
        static_cast<size_t>(full_recurrence_state));
    ASSERT_TRUE(recurrence.exportStateForSize(
        full_recurrence_state,
        recurrence_live.data(),
        /*dst_device=*/nullptr,
        stream.stream));
    ASSERT_TRUE(recurrence.exportRequestStateBank(
        recurrence_request.data(),
        /*request_count=*/1,
        full_recurrence_state,
        stream.stream));
    checkHip(
        hipStreamSynchronize(stream.stream),
        "hipStreamSynchronize(replayed recurrence publication)");
    expectByteExactEquivalent(
        "captured ROCm recurrence request-state publication",
        recurrence_request,
        recurrence_live,
        /*offset=*/0,
        recurrence_live.size());

    constexpr int local_channels = 64;
    constexpr int full_channels = 128;
    constexpr int kernel_size = 4;
    constexpr int local_conv_state = local_channels * (kernel_size - 1);
    constexpr int full_conv_state = full_channels * (kernel_size - 1);

    ROCmShortConvolution conv(0);
    conv.setGPUStream(stream.stream);
    HipGDNStateOwner conv_state_owner(
        conv,
        local_conv_state,
        full_conv_state);
    const auto conv_weights =
        makeShortConvWeights(local_channels, kernel_size);
    const auto conv_bias = makeBias(local_channels);
    HipFloatBuffer d_conv_input(
        static_cast<size_t>(seq_len) * local_channels,
        0.04f);
    HipFloatBuffer d_conv_weight(conv_weights);
    HipFloatBuffer d_conv_bias(conv_bias);
    HipFloatBuffer d_conv_out(
        static_cast<size_t>(seq_len) * local_channels,
        0.0f);
    const auto full_conv_payload =
        makeInitialState(static_cast<size_t>(full_conv_state), 0.0061f);
    HipFloatBuffer d_full_conv(full_conv_payload);

    HipCapturedGraph conv_graph(
        stream.stream,
        [&]()
        {
            return conv.forward(
                       d_conv_input.ptr,
                       d_conv_weight.ptr,
                       d_conv_bias.ptr,
                       d_conv_out.ptr,
                       nullptr,
                       seq_len,
                       local_channels,
                       kernel_size,
                       /*apply_silu=*/true) &&
                   conv.importStateForSize(
                       full_conv_state,
                       /*src_host=*/nullptr,
                       d_full_conv.ptr,
                       stream.stream);
        });

    ASSERT_TRUE(conv.resetGPUState(stream.stream));
    conv_graph.launch(stream.stream);
    std::vector<float> conv_live(static_cast<size_t>(full_conv_state));
    std::vector<float> conv_request(static_cast<size_t>(full_conv_state));
    ASSERT_TRUE(conv.exportStateForSize(
        full_conv_state,
        conv_live.data(),
        /*dst_device=*/nullptr,
        stream.stream));
    ASSERT_TRUE(conv.exportRequestStateBank(
        conv_request.data(),
        /*request_count=*/1,
        full_conv_state,
        stream.stream));
    checkHip(
        hipStreamSynchronize(stream.stream),
        "hipStreamSynchronize(replayed short-conv publication)");
    expectByteExactEquivalent(
        "captured ROCm short-conv request-state publication",
        conv_request,
        conv_live,
        /*offset=*/0,
        conv_live.size());
}

TEST(Test__ROCmGDNPaddedRealLength, RecurrenceEffectivePrefillMatchesUnpaddedDecode)
{
    if (!hasROCm())
        GTEST_SKIP() << "No ROCm device available";
    checkHip(hipSetDevice(0), "hipSetDevice");
    HipStreamHandle stream;

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
        HipGDNStateOwner padded_kernel_state_owner(padded_kernel, n_heads * d_k * d_v);
        padded_kernel.setGPUStream(stream.stream);
        ASSERT_TRUE(padded_kernel.chunkForwardWithEffectiveSeqLen(
            d_Q.ptr, d_K.ptr, d_V.ptr, d_alpha.ptr, d_beta.ptr, d_A_log.ptr, d_dt_bias.ptr,
            d_padded_out.ptr, nullptr,
            bucket_len, n_heads, d_k, d_v,
            /*chunk_size=*/64, /*use_qk_l2norm=*/true,
            d_effective_len.ptr));
        checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(padded recurrence prefill)");
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
        checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(padded recurrence decode)");

        ROCmGatedDeltaNet ref_kernel(0);
        HipGDNStateOwner ref_kernel_state_owner(ref_kernel, n_heads * d_k * d_v);
        ref_kernel.setGPUStream(stream.stream);
        ASSERT_TRUE(ref_kernel.chunk_forward(
            d_Q.ptr, d_K.ptr, d_V.ptr, d_alpha.ptr, d_beta.ptr, d_A_log.ptr, d_dt_bias.ptr,
            d_ref_out.ptr, nullptr,
            real_len, n_heads, d_k, d_v,
            /*chunk_size=*/64, /*use_qk_l2norm=*/true));
        checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(reference recurrence prefill)");
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
        checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(reference recurrence decode)");

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
    HipStreamHandle stream;

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
        HipGDNStateOwner padded_kernel_state_owner(padded_kernel, channels * (kernel_size - 1));
        padded_kernel.setGPUStream(stream.stream);
        ASSERT_TRUE(padded_kernel.forwardWithEffectiveSeqLen(
            d_input.ptr, d_weight.ptr, d_bias.ptr,
            d_padded_out.ptr, nullptr,
            bucket_len, channels, kernel_size,
            d_effective_len.ptr,
            /*apply_silu=*/true));
        checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(padded short-conv prefill)");
        ASSERT_TRUE(padded_kernel.forward(
            d_input.ptr + static_cast<size_t>(decode_row) * channels,
            d_weight.ptr,
            d_bias.ptr,
            d_padded_out.ptr + static_cast<size_t>(decode_row) * channels,
            nullptr,
            1, channels, kernel_size,
            /*apply_silu=*/true));
        checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(padded short-conv decode)");

        ROCmShortConvolution ref_kernel(0);
        HipGDNStateOwner ref_kernel_state_owner(ref_kernel, channels * (kernel_size - 1));
        ref_kernel.setGPUStream(stream.stream);
        ASSERT_TRUE(ref_kernel.forward(
            d_input.ptr, d_weight.ptr, d_bias.ptr,
            d_ref_out.ptr, nullptr,
            real_len, channels, kernel_size,
            /*apply_silu=*/true));
        checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(reference short-conv prefill)");
        ASSERT_TRUE(ref_kernel.forward(
            d_input.ptr + static_cast<size_t>(decode_row) * channels,
            d_weight.ptr,
            d_bias.ptr,
            d_ref_out.ptr + static_cast<size_t>(decode_row) * channels,
            nullptr,
            1, channels, kernel_size,
            /*apply_silu=*/true));
        checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(reference short-conv decode)");

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
    HipGDNStateOwner verifier_kernel_state_owner(verifier_kernel, state_floats);
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
    checkHip(
        hipStreamSynchronize(stream.stream),
        "hipStreamSynchronize(publish recurrence snapshot)");
    std::vector<float> published_device_state(static_cast<size_t>(state_floats));
    ASSERT_TRUE(verifier_kernel.exportState(
        published_device_state.data(), nullptr, nullptr));
    expectPublishedDeviceStateMatchesSnapshotRow(
        published_device_state,
        d_snapshots,
        accepted_rows - 1,
        state_floats);
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
    HipGDNStateOwner ref_kernel_state_owner(ref_kernel, state_floats);
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

TEST(Test__ROCmGDNPaddedRealLength, RecurrenceVerifierRowRestoreMatchesMultiStepReplay)
{
    if (!hasROCm())
        GTEST_SKIP() << "No ROCm device available";
    checkHip(hipSetDevice(0), "hipSetDevice");

    constexpr int n_heads = 2;
    constexpr int d_k = 128;
    constexpr int d_v = 128;
    constexpr int accepted_rows = 4;
    constexpr int continuation_rows = 4;
    constexpr int verifier_len = accepted_rows + continuation_rows;
    constexpr int qk_stride = n_heads * d_k;
    constexpr int v_stride = n_heads * d_v;
    constexpr int state_floats = n_heads * d_k * d_v;
    constexpr size_t verifier_output_elems =
        static_cast<size_t>(verifier_len) * static_cast<size_t>(v_stride);
    constexpr size_t continuation_output_elems =
        static_cast<size_t>(continuation_rows) * static_cast<size_t>(v_stride);

    const auto Q = makeSequenceRows(verifier_len, qk_stride, verifier_len, verifier_len, 0.0023f, 0.0f, 0.0023f);
    const auto K = makeSequenceRows(verifier_len, qk_stride, verifier_len, verifier_len, -0.0017f, 0.0f, -0.0017f);
    const auto V = makeSequenceRows(verifier_len, v_stride, verifier_len, verifier_len, 0.0029f, 0.0f, 0.0029f);
    const auto alpha = makeSequenceRows(verifier_len, n_heads, verifier_len, verifier_len, 0.019f, 0.0f, 0.019f);
    const auto beta = makeSequenceRows(verifier_len, n_heads, verifier_len, verifier_len, -0.017f, 0.0f, -0.017f);
    const auto initial_state = makeInitialState(static_cast<size_t>(state_floats), 0.00073f);
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
    HipFloatBuffer d_verifier_out(verifier_output_elems, 0.0f);
    HipFloatBuffer d_restored_continuation(continuation_output_elems, 0.0f);
    HipFloatBuffer d_ref_prefix(static_cast<size_t>(accepted_rows) * static_cast<size_t>(v_stride), 0.0f);
    HipFloatBuffer d_ref_continuation(continuation_output_elems, 0.0f);
    HipFloatBuffer d_snapshots(static_cast<size_t>(verifier_len) * static_cast<size_t>(state_floats), -99.0f);
    HipFloatBuffer d_speculative_state_work(static_cast<size_t>(state_floats), 0.0f);
    HipStreamHandle stream;

    ROCmGatedDeltaNet verifier_kernel(0);
    HipGDNStateOwner verifier_kernel_state_owner(verifier_kernel, state_floats);
    verifier_kernel.setGPUStream(stream.stream);
    verifier_kernel.bindVerifierStateCaptureWorkspace(d_snapshots.ptr, verifier_len, state_floats);
    verifier_kernel.bindSpeculativeStateWorkspace(d_speculative_state_work.ptr, state_floats);
    ASSERT_TRUE(verifier_kernel.importState(initial_state.data(), nullptr, stream.stream));
    ASSERT_TRUE(verifier_kernel.chunk_forward(
        d_Q.ptr, d_K.ptr, d_V.ptr, d_alpha.ptr, d_beta.ptr, d_A_log.ptr, d_dt_bias.ptr,
        d_verifier_out.ptr, nullptr,
        verifier_len, n_heads, d_k, d_v,
        /*chunk_size=*/64, /*use_qk_l2norm=*/true));
    ASSERT_TRUE(verifier_kernel.restoreVerifierStateCaptureRow(
        nullptr, accepted_rows - 1, stream.stream));
    checkHip(
        hipStreamSynchronize(stream.stream),
        "hipStreamSynchronize(publish multi-row recurrence snapshot)");
    std::vector<float> published_device_state(static_cast<size_t>(state_floats));
    ASSERT_TRUE(verifier_kernel.exportState(
        published_device_state.data(), nullptr, nullptr));
    expectPublishedDeviceStateMatchesSnapshotRow(
        published_device_state,
        d_snapshots,
        accepted_rows - 1,
        state_floats);
    verifier_kernel.bindVerifierStateCaptureWorkspace(nullptr, 0, state_floats);
    verifier_kernel.bindSpeculativeStateWorkspace(nullptr, state_floats);
    for (int row = 0; row < continuation_rows; ++row)
    {
        const int source_row = accepted_rows + row;
        ASSERT_TRUE(verifier_kernel.recurrent_step(
            d_Q_cont.ptr + static_cast<size_t>(source_row) * qk_stride,
            d_K_cont.ptr + static_cast<size_t>(source_row) * qk_stride,
            d_V_cont.ptr + static_cast<size_t>(source_row) * v_stride,
            d_alpha_cont.ptr + static_cast<size_t>(source_row) * n_heads,
            d_beta_cont.ptr + static_cast<size_t>(source_row) * n_heads,
            d_A_log.ptr,
            d_dt_bias.ptr,
            d_restored_continuation.ptr + static_cast<size_t>(row) * v_stride,
            nullptr,
            n_heads, d_k, d_v,
            /*use_qk_l2norm=*/true));
    }
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(restored recurrence continuation)");
    std::vector<float> restored_state(static_cast<size_t>(state_floats));
    ASSERT_TRUE(verifier_kernel.exportState(restored_state.data(), nullptr, stream.stream));

    ROCmGatedDeltaNet ref_kernel(0);
    HipGDNStateOwner ref_kernel_state_owner(ref_kernel, state_floats);
    ref_kernel.setGPUStream(stream.stream);
    ASSERT_TRUE(ref_kernel.importState(initial_state.data(), nullptr, stream.stream));
    ASSERT_TRUE(ref_kernel.chunk_forward(
        d_Q_ref.ptr, d_K_ref.ptr, d_V_ref.ptr, d_alpha_ref.ptr, d_beta_ref.ptr, d_A_log.ptr, d_dt_bias.ptr,
        d_ref_prefix.ptr, nullptr,
        accepted_rows, n_heads, d_k, d_v,
        /*chunk_size=*/64, /*use_qk_l2norm=*/true));
    for (int row = 0; row < continuation_rows; ++row)
    {
        const int source_row = accepted_rows + row;
        ASSERT_TRUE(ref_kernel.recurrent_step(
            d_Q_ref.ptr + static_cast<size_t>(source_row) * qk_stride,
            d_K_ref.ptr + static_cast<size_t>(source_row) * qk_stride,
            d_V_ref.ptr + static_cast<size_t>(source_row) * v_stride,
            d_alpha_ref.ptr + static_cast<size_t>(source_row) * n_heads,
            d_beta_ref.ptr + static_cast<size_t>(source_row) * n_heads,
            d_A_log.ptr,
            d_dt_bias.ptr,
            d_ref_continuation.ptr + static_cast<size_t>(row) * v_stride,
            nullptr,
            n_heads, d_k, d_v,
            /*use_qk_l2norm=*/true));
    }
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(reference recurrence continuation)");
    std::vector<float> ref_state(static_cast<size_t>(state_floats));
    ASSERT_TRUE(ref_kernel.exportState(ref_state.data(), nullptr, stream.stream));

    const auto restored = d_restored_continuation.toHost();
    const auto ref = d_ref_continuation.toHost();
    const auto out_diff = diffStats(restored, ref, 0, restored.size());
    const auto state_diff = diffStats(restored_state, ref_state, 0, restored_state.size());
    EXPECT_LT(out_diff.first, 2e-4f);
    EXPECT_LT(out_diff.second, 1e-4);
    EXPECT_LT(state_diff.first, 2e-4f);
    EXPECT_LT(state_diff.second, 1e-4);
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
    HipGDNStateOwner m4_kernel_state_owner(m4_kernel, state_floats);
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
    HipGDNStateOwner step_kernel_state_owner(step_kernel, state_floats);
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
    HipGDNStateOwner m4_kernel_state_owner(m4_kernel, state_floats);
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
    HipGDNStateOwner step_kernel_state_owner(step_kernel, state_floats);
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
    HipGDNStateOwner m4_kernel_state_owner(m4_kernel, state_floats);
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
    HipGDNStateOwner step_kernel_state_owner(step_kernel, state_floats);
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
    HipGDNStateOwner m4_kernel_state_owner(m4_kernel, state_floats);
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
    HipGDNStateOwner step_kernel_state_owner(step_kernel, state_floats);
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

    expectByteExactEquivalent(
        "ROCm Qwen3.6 GDN M4 verifier output",
        m4_out,
        step_out,
        0,
        output_elems);
    expectByteExactEquivalent(
        "ROCm Qwen3.6 GDN M4 final state snapshot",
        final_snapshot,
        step_state,
        0,
        final_snapshot.size());
    expectByteExactEquivalent(
        "ROCm Qwen3.6 GDN M4 live state remains initial during verifier capture",
        m4_state,
        initial_state,
        0,
        m4_state.size());
}

TEST(Test__ROCmGDNPaddedRealLength, MergedQKVM2Qwen36DenseShapeVerifierCaptureMatchesStepwiseStrict)
{
    if (!hasROCm())
        GTEST_SKIP() << "No ROCm device available";
    checkHip(hipSetDevice(0), "hipSetDevice");

    constexpr int n_k_heads = 16;
    constexpr int n_v_heads = 48;
    constexpr int d_k = 128;
    constexpr int d_v = 128;
    constexpr int verifier_len = 2;
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

    HipFloatBuffer d_merged_grouped(merged);
    HipFloatBuffer d_merged_step(merged);
    HipFloatBuffer d_alpha_grouped(alpha);
    HipFloatBuffer d_beta_grouped(beta);
    HipFloatBuffer d_alpha_step(alpha);
    HipFloatBuffer d_beta_step(beta);
    HipFloatBuffer d_A_log(A_log);
    HipFloatBuffer d_dt_bias(dt_bias);
    HipFloatBuffer d_grouped_out(output_elems, 0.0f);
    HipFloatBuffer d_step_out(output_elems, 0.0f);
    HipFloatBuffer d_grouped_scratch(deinterleave_elems, 0.0f);
    HipFloatBuffer d_step_scratch(static_cast<size_t>(qk_stride + qk_stride + v_dim), 0.0f);
    HipFloatBuffer d_snapshots(static_cast<size_t>(verifier_len) * static_cast<size_t>(state_floats), -99.0f);
    HipFloatBuffer d_speculative_state_work(static_cast<size_t>(state_floats), 0.0f);
    HipStreamHandle stream;

    ROCmGatedDeltaNet grouped_kernel(0);
    HipGDNStateOwner grouped_kernel_state_owner(grouped_kernel, state_floats);
    grouped_kernel.setGPUStream(stream.stream);
    grouped_kernel.bindDeinterleaveWorkspace(d_grouped_scratch.ptr, d_grouped_scratch.count);
    grouped_kernel.bindVerifierStateCaptureWorkspace(d_snapshots.ptr, verifier_len, state_floats);
    grouped_kernel.bindSpeculativeStateWorkspace(d_speculative_state_work.ptr, state_floats);
    ASSERT_TRUE(grouped_kernel.importState(initial_state.data(), nullptr, stream.stream));
    float *grouped_q = nullptr;
    float *grouped_k = nullptr;
    float *grouped_v = nullptr;
    ASSERT_TRUE(grouped_kernel.deinterleave_qkv_device(
        d_merged_grouped.ptr,
        grouped_q,
        grouped_k,
        grouped_v,
        verifier_len,
        n_k_heads,
        n_v_heads,
        d_k,
        d_v,
        /*global_v_head_offset=*/0));
    ASSERT_TRUE(grouped_kernel.chunk_forward(
        grouped_q, grouped_k, grouped_v,
        d_alpha_grouped.ptr, d_beta_grouped.ptr,
        d_A_log.ptr, d_dt_bias.ptr,
        d_grouped_out.ptr, nullptr,
        verifier_len, n_v_heads, d_k, d_v,
        /*chunk_size=*/64, /*use_qk_l2norm=*/true));
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(strict grouped GDN M2)");
    std::vector<float> grouped_live_state(static_cast<size_t>(state_floats));
    ASSERT_TRUE(grouped_kernel.exportState(grouped_live_state.data(), nullptr, nullptr));

    ROCmGatedDeltaNet step_kernel(0);
    HipGDNStateOwner step_kernel_state_owner(step_kernel, state_floats);
    step_kernel.setGPUStream(stream.stream);
    step_kernel.bindDeinterleaveWorkspace(d_step_scratch.ptr, d_step_scratch.count);
    ASSERT_TRUE(step_kernel.importState(initial_state.data(), nullptr, stream.stream));
    std::vector<float> step_state_snapshots(static_cast<size_t>(verifier_len) * static_cast<size_t>(state_floats));
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
        checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(strict stepwise GDN M2 row)");

        std::vector<float> row_state(static_cast<size_t>(state_floats));
        ASSERT_TRUE(step_kernel.exportState(row_state.data(), nullptr, nullptr));
        std::copy_n(
            row_state.data(),
            static_cast<size_t>(state_floats),
            step_state_snapshots.data() + static_cast<size_t>(row) * static_cast<size_t>(state_floats));
    }

    const auto grouped_out = d_grouped_out.toHost();
    const auto step_out = d_step_out.toHost();
    const auto grouped_snapshots = d_snapshots.toHost();
    for (int row = 0; row < verifier_len; ++row)
    {
        const size_t output_offset = static_cast<size_t>(row) * static_cast<size_t>(v_dim);
        expectByteExactEquivalent(
            row == 0 ? "GDN M2 row0 output" : "GDN M2 row1 output",
            grouped_out,
            step_out,
            output_offset,
            static_cast<size_t>(v_dim));

        const size_t state_offset = static_cast<size_t>(row) * static_cast<size_t>(state_floats);
        expectByteExactEquivalent(
            row == 0 ? "GDN M2 row0 state snapshot" : "GDN M2 row1 state snapshot",
            grouped_snapshots,
            step_state_snapshots,
            state_offset,
            static_cast<size_t>(state_floats));
    }

    expectByteExactEquivalent(
        "GDN M2 grouped live state remains initial during verifier capture",
        grouped_live_state,
        initial_state,
        0,
        grouped_live_state.size());
}

TEST(Test__ROCmGDNPaddedRealLength, MergedQKVM3Qwen36DenseShapeVerifierCaptureMatchesStepwiseStrict)
{
    if (!hasROCm())
        GTEST_SKIP() << "No ROCm device available";
    checkHip(hipSetDevice(0), "hipSetDevice");

    /*
     * Depth-3 MTP verifier rows are the production shape that exposed GDN
     * drift in the full continuation proof.  Keep ROCm under the same strict
     * merged-QKV row/snapshot contract as CUDA.
     */
    constexpr int n_k_heads = 16;
    constexpr int n_v_heads = 48;
    constexpr int d_k = 128;
    constexpr int d_v = 128;
    constexpr int verifier_len = 3;
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

    HipFloatBuffer d_merged_grouped(merged);
    HipFloatBuffer d_merged_step(merged);
    HipFloatBuffer d_alpha_grouped(alpha);
    HipFloatBuffer d_beta_grouped(beta);
    HipFloatBuffer d_alpha_step(alpha);
    HipFloatBuffer d_beta_step(beta);
    HipFloatBuffer d_A_log(A_log);
    HipFloatBuffer d_dt_bias(dt_bias);
    HipFloatBuffer d_grouped_out(output_elems, 0.0f);
    HipFloatBuffer d_step_out(output_elems, 0.0f);
    HipFloatBuffer d_grouped_scratch(deinterleave_elems, 0.0f);
    HipFloatBuffer d_step_scratch(static_cast<size_t>(qk_stride + qk_stride + v_dim), 0.0f);
    HipFloatBuffer d_snapshots(static_cast<size_t>(verifier_len) * static_cast<size_t>(state_floats), -99.0f);
    HipFloatBuffer d_speculative_state_work(static_cast<size_t>(state_floats), 0.0f);
    HipStreamHandle stream;

    ROCmGatedDeltaNet grouped_kernel(0);
    HipGDNStateOwner grouped_kernel_state_owner(grouped_kernel, state_floats);
    grouped_kernel.setGPUStream(stream.stream);
    grouped_kernel.bindDeinterleaveWorkspace(d_grouped_scratch.ptr, d_grouped_scratch.count);
    grouped_kernel.bindVerifierStateCaptureWorkspace(d_snapshots.ptr, verifier_len, state_floats);
    grouped_kernel.bindSpeculativeStateWorkspace(d_speculative_state_work.ptr, state_floats);
    ASSERT_TRUE(grouped_kernel.importState(initial_state.data(), nullptr, stream.stream));
    float *grouped_q = nullptr;
    float *grouped_k = nullptr;
    float *grouped_v = nullptr;
    ASSERT_TRUE(grouped_kernel.deinterleave_qkv_device(
        d_merged_grouped.ptr,
        grouped_q,
        grouped_k,
        grouped_v,
        verifier_len,
        n_k_heads,
        n_v_heads,
        d_k,
        d_v,
        /*global_v_head_offset=*/0));
    ASSERT_TRUE(grouped_kernel.chunk_forward(
        grouped_q, grouped_k, grouped_v,
        d_alpha_grouped.ptr, d_beta_grouped.ptr,
        d_A_log.ptr, d_dt_bias.ptr,
        d_grouped_out.ptr, nullptr,
        verifier_len, n_v_heads, d_k, d_v,
        /*chunk_size=*/64, /*use_qk_l2norm=*/true));
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(strict grouped ROCm GDN M3)");
    std::vector<float> grouped_live_state(static_cast<size_t>(state_floats));
    ASSERT_TRUE(grouped_kernel.exportState(grouped_live_state.data(), nullptr, nullptr));

    ROCmGatedDeltaNet step_kernel(0);
    HipGDNStateOwner step_kernel_state_owner(step_kernel, state_floats);
    step_kernel.setGPUStream(stream.stream);
    step_kernel.bindDeinterleaveWorkspace(d_step_scratch.ptr, d_step_scratch.count);
    ASSERT_TRUE(step_kernel.importState(initial_state.data(), nullptr, stream.stream));
    std::vector<float> step_state_snapshots(static_cast<size_t>(verifier_len) * static_cast<size_t>(state_floats));
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
        checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(strict stepwise ROCm GDN M3 row)");

        std::vector<float> row_state(static_cast<size_t>(state_floats));
        ASSERT_TRUE(step_kernel.exportState(row_state.data(), nullptr, nullptr));
        std::copy_n(
            row_state.data(),
            static_cast<size_t>(state_floats),
            step_state_snapshots.data() + static_cast<size_t>(row) * static_cast<size_t>(state_floats));
    }

    const auto grouped_out = d_grouped_out.toHost();
    const auto step_out = d_step_out.toHost();
    const auto grouped_snapshots = d_snapshots.toHost();
    for (int row = 0; row < verifier_len; ++row)
    {
        const size_t output_offset = static_cast<size_t>(row) * static_cast<size_t>(v_dim);
        const std::string output_label =
            "ROCm GDN M3 row" + std::to_string(row) + " output";
        expectByteExactEquivalent(
            output_label.c_str(),
            grouped_out,
            step_out,
            output_offset,
            static_cast<size_t>(v_dim));

        const size_t state_offset = static_cast<size_t>(row) * static_cast<size_t>(state_floats);
        const std::string state_label =
            "ROCm GDN M3 row" + std::to_string(row) + " state snapshot";
        expectByteExactEquivalent(
            state_label.c_str(),
            grouped_snapshots,
            step_state_snapshots,
            state_offset,
            static_cast<size_t>(state_floats));
    }

    expectByteExactEquivalent(
        "ROCm GDN M3 grouped live state remains initial during verifier capture",
        grouped_live_state,
        initial_state,
        0,
        grouped_live_state.size());
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
    HipGDNStateOwner verifier_kernel_state_owner(verifier_kernel, state_floats);
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
    checkHip(
        hipStreamSynchronize(stream.stream),
        "hipStreamSynchronize(publish short-conv snapshot)");
    std::vector<float> published_device_state(static_cast<size_t>(state_floats));
    ASSERT_TRUE(verifier_kernel.exportState(
        published_device_state.data(), nullptr, nullptr));
    expectPublishedDeviceStateMatchesSnapshotRow(
        published_device_state,
        d_snapshots,
        accepted_rows - 1,
        state_floats);
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
    HipGDNStateOwner ref_kernel_state_owner(ref_kernel, state_floats);
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
    checkHip(hipDeviceSynchronize(), "hipDeviceSynchronize(reference short-conv decode)");

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
    HipGDNStateOwner m4_kernel_state_owner(m4_kernel, state_floats);
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
    HipGDNStateOwner step_kernel_state_owner(step_kernel, state_floats);
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

TEST(Test__ROCmGDNPaddedRealLength, ShortConvQwen36M2InPlaceStateMatchesStepwiseReplay)
{
    if (!hasROCm())
        GTEST_SKIP() << "No ROCm device available";
    checkHip(hipSetDevice(0), "hipSetDevice");

    /*
     * Depth-2 MTP is the smallest useful grouped verifier shape.  Exercise it
     * with the full Qwen3.6 short-conv width and in-place QKV buffer contract
     * so the regression covers raw-history preservation, scratch publication,
     * and the device-owned live state update in one production-shaped case.
     */
    constexpr int channels = 10240;
    constexpr int kernel_size = 4;
    constexpr int verifier_len = 2;
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

    HipFloatBuffer d_m2_inout(input);
    HipFloatBuffer d_step_inout(input);
    HipFloatBuffer d_weight(weight);
    HipFloatBuffer d_bias(bias);
    HipStreamHandle stream;

    ROCmShortConvolution m2_kernel(0);
    HipGDNStateOwner m2_kernel_state_owner(m2_kernel, state_floats);
    m2_kernel_state_owner.bindScratch(
        m2_kernel, static_cast<int>(output_elems));
    m2_kernel.setGPUStream(stream.stream);
    ASSERT_TRUE(m2_kernel.importState(initial_state.data(), nullptr, stream.stream));
    ASSERT_TRUE(m2_kernel.forward(
        d_m2_inout.ptr,
        d_weight.ptr,
        d_bias.ptr,
        d_m2_inout.ptr,
        nullptr,
        verifier_len, channels, kernel_size,
        /*apply_silu=*/true));
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(Qwen36 M2 in-place short-conv)");
    std::vector<float> m2_state(static_cast<size_t>(state_floats));
    ASSERT_TRUE(m2_kernel.exportState(m2_state.data(), nullptr, nullptr));

    ROCmShortConvolution step_kernel(0);
    HipGDNStateOwner step_kernel_state_owner(step_kernel, state_floats);
    step_kernel_state_owner.bindScratch(step_kernel, channels);
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
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(Qwen36 stepwise M2 in-place short-conv)");
    std::vector<float> step_state(static_cast<size_t>(state_floats));
    ASSERT_TRUE(step_kernel.exportState(step_state.data(), nullptr, nullptr));

    const auto m2_out = d_m2_inout.toHost();
    const auto step_out = d_step_inout.toHost();
    expectByteExactEquivalent(
        "ROCm Qwen3.6 short-conv M2 output",
        m2_out,
        step_out,
        0,
        output_elems);
    expectByteExactEquivalent(
        "ROCm Qwen3.6 short-conv M2 state",
        m2_state,
        step_state,
        0,
        m2_state.size());
}

TEST(Test__ROCmGDNPaddedRealLength, ShortConvQwen36M3InPlaceStateMatchesStepwiseReplay)
{
    if (!hasROCm())
        GTEST_SKIP() << "No ROCm device available";
    checkHip(hipSetDevice(0), "hipSetDevice");

    /*
     * Keep ROCm aligned with CUDA for the depth-3 verifier shape.  The MTP
     * graph relies on the grouped kernel being numerically equivalent to three
     * explicit decode rows while preserving raw projection history in-place.
     */
    constexpr int channels = 10240;
    constexpr int kernel_size = 4;
    constexpr int verifier_len = 3;
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

    HipFloatBuffer d_m3_inout(input);
    HipFloatBuffer d_step_inout(input);
    HipFloatBuffer d_weight(weight);
    HipFloatBuffer d_bias(bias);
    HipStreamHandle stream;

    ROCmShortConvolution m3_kernel(0);
    HipGDNStateOwner m3_kernel_state_owner(m3_kernel, state_floats);
    m3_kernel_state_owner.bindScratch(
        m3_kernel, static_cast<int>(output_elems));
    m3_kernel.setGPUStream(stream.stream);
    ASSERT_TRUE(m3_kernel.importState(initial_state.data(), nullptr, stream.stream));
    ASSERT_TRUE(m3_kernel.forward(
        d_m3_inout.ptr,
        d_weight.ptr,
        d_bias.ptr,
        d_m3_inout.ptr,
        nullptr,
        verifier_len, channels, kernel_size,
        /*apply_silu=*/true));
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(Qwen36 M3 in-place short-conv)");
    std::vector<float> m3_state(static_cast<size_t>(state_floats));
    ASSERT_TRUE(m3_kernel.exportState(m3_state.data(), nullptr, nullptr));

    ROCmShortConvolution step_kernel(0);
    HipGDNStateOwner step_kernel_state_owner(step_kernel, state_floats);
    step_kernel_state_owner.bindScratch(step_kernel, channels);
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
    checkHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize(Qwen36 stepwise M3 in-place short-conv)");
    std::vector<float> step_state(static_cast<size_t>(state_floats));
    ASSERT_TRUE(step_kernel.exportState(step_state.data(), nullptr, nullptr));

    const auto m3_out = d_m3_inout.toHost();
    const auto step_out = d_step_inout.toHost();
    expectByteExactEquivalent(
        "ROCm Qwen3.6 short-conv M3 output",
        m3_out,
        step_out,
        0,
        output_elems);
    expectByteExactEquivalent(
        "ROCm Qwen3.6 short-conv M3 state",
        m3_state,
        step_state,
        0,
        m3_state.size());
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
    HipGDNStateOwner m4_kernel_state_owner(m4_kernel, state_floats);
    m4_kernel_state_owner.bindScratch(
        m4_kernel, static_cast<int>(output_elems));
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
    HipGDNStateOwner step_kernel_state_owner(step_kernel, state_floats);
    step_kernel_state_owner.bindScratch(step_kernel, channels);
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
    expectByteExactEquivalent(
        "ROCm Qwen3.6 short-conv M4 output",
        m4_out,
        step_out,
        0,
        output_elems);
    expectByteExactEquivalent(
        "ROCm Qwen3.6 short-conv M4 state",
        m4_state,
        step_state,
        0,
        m4_state.size());
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
    HipGDNStateOwner verifier_kernel_state_owner(verifier_kernel, state_floats);
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
    HipGDNStateOwner ref_kernel_state_owner(ref_kernel, state_floats);
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

/**
 * @brief Prove a captured HIP decode graph observes device-published GDN state.
 *
 * The graph is instantiated once before any grouped verifier transaction. For
 * At M=2..16 and M=31, every possible accepted row is selected by a device
 * scalar and copied into the kernel-owned live state. Every grouped output row
 * is also compared directly with scalar M=1 decode. The original graph must
 * continue from that row byte-for-byte; recreating it in the loop would defeat
 * the lifetime regression this test is designed to catch.
 */
TEST(Test__ROCmGDNPaddedRealLength, CapturedVerifierAndDecodeRecurrencePublicationRuntimeM)
{
    if (!hasROCm())
        GTEST_SKIP() << "No ROCm device available";
    checkHip(hipSetDevice(0), "hipSetDevice");

    constexpr int n_heads = 2;
    constexpr int d_k = 128;
    constexpr int d_v = 128;
    constexpr int max_verifier_rows =
        llaminar2::test::kGroupedVerifierRuntimeRows.back();
    constexpr int qk_width = n_heads * d_k;
    constexpr int value_width = n_heads * d_v;
    constexpr int state_floats = n_heads * d_k * d_v;

    const auto q_rows = makeSequenceRows(
        max_verifier_rows, qk_width, max_verifier_rows,
        max_verifier_rows, 0.0023f, 0.0f, 0.0023f);
    const auto k_rows = makeSequenceRows(
        max_verifier_rows, qk_width, max_verifier_rows,
        max_verifier_rows, -0.0017f, 0.0f, -0.0017f);
    const auto v_rows = makeSequenceRows(
        max_verifier_rows, value_width, max_verifier_rows,
        max_verifier_rows, 0.0029f, 0.0f, 0.0029f);
    const auto alpha_rows = makeSequenceRows(
        max_verifier_rows, n_heads, max_verifier_rows,
        max_verifier_rows, 0.019f, 0.0f, 0.019f);
    const auto beta_rows = makeSequenceRows(
        max_verifier_rows, n_heads, max_verifier_rows,
        max_verifier_rows, -0.017f, 0.0f, -0.017f);
    const auto continuation_q = makeSequenceRows(1, qk_width, 1, 1, 0.0037f, 0.0f, 0.0037f);
    const auto continuation_k = makeSequenceRows(1, qk_width, 1, 1, -0.0021f, 0.0f, -0.0021f);
    const auto continuation_v = makeSequenceRows(1, value_width, 1, 1, 0.0041f, 0.0f, 0.0041f);
    const auto continuation_alpha = makeSequenceRows(1, n_heads, 1, 1, 0.023f, 0.0f, 0.023f);
    const auto continuation_beta = makeSequenceRows(1, n_heads, 1, 1, -0.013f, 0.0f, -0.013f);
    const auto initial_state = makeInitialState(static_cast<size_t>(state_floats), 0.00073f);
    const std::vector<float> a_log(static_cast<size_t>(n_heads), -0.5f);
    const std::vector<float> dt_bias(static_cast<size_t>(n_heads), 0.1f);

    HipFloatBuffer d_q_rows(q_rows);
    HipFloatBuffer d_k_rows(k_rows);
    HipFloatBuffer d_v_rows(v_rows);
    HipFloatBuffer d_alpha_rows(alpha_rows);
    HipFloatBuffer d_beta_rows(beta_rows);
    HipFloatBuffer d_continuation_q(continuation_q);
    HipFloatBuffer d_continuation_k(continuation_k);
    HipFloatBuffer d_continuation_v(continuation_v);
    HipFloatBuffer d_continuation_alpha(continuation_alpha);
    HipFloatBuffer d_continuation_beta(continuation_beta);
    HipFloatBuffer d_a_log(a_log);
    HipFloatBuffer d_dt_bias(dt_bias);
    HipFloatBuffer d_grouped_output(
        static_cast<size_t>(max_verifier_rows) * value_width, 0.0f);
    HipFloatBuffer d_replay_output(static_cast<size_t>(value_width), 0.0f);
    HipFloatBuffer d_oracle_output(static_cast<size_t>(value_width), 0.0f);
    HipFloatBuffer d_oracle_row_output(static_cast<size_t>(value_width), 0.0f);
    HipFloatBuffer d_snapshots(
        static_cast<size_t>(max_verifier_rows) * state_floats, -77.0f);
    HipFloatBuffer d_speculative_state(static_cast<size_t>(state_floats), 0.0f);
    HipStreamHandle stream;

    ROCmGatedDeltaNet live_kernel(0);
    HipGDNStateOwner live_kernel_state_owner(live_kernel, state_floats);
    live_kernel.setGPUStream(stream.stream);
    live_kernel.bindVerifierStateCaptureWorkspace(nullptr, 0, state_floats);
    live_kernel.bindSpeculativeStateWorkspace(nullptr, state_floats);

    HipCapturedGraph captured_decode(
        stream.stream,
        [&]()
        {
            return live_kernel.recurrent_step(
                d_continuation_q.ptr,
                d_continuation_k.ptr,
                d_continuation_v.ptr,
                d_continuation_alpha.ptr,
                d_continuation_beta.ptr,
                d_a_log.ptr,
                d_dt_bias.ptr,
                d_replay_output.ptr,
                nullptr,
                n_heads,
                d_k,
                d_v,
                /*use_qk_l2norm=*/true);
        });

    ROCmGatedDeltaNet oracle_kernel(0);
    HipGDNStateOwner oracle_kernel_state_owner(oracle_kernel, state_floats);
    oracle_kernel.setGPUStream(stream.stream);

    for (const int verifier_rows : llaminar2::test::kGroupedVerifierRuntimeRows)
    {
        live_kernel.bindVerifierStateCaptureWorkspace(
            d_snapshots.ptr, verifier_rows, state_floats);
        live_kernel.bindSpeculativeStateWorkspace(
            d_speculative_state.ptr, state_floats);
        HipCapturedGraph captured_verifier(
            stream.stream,
            [&]()
            {
                return live_kernel.chunk_forward(
                    d_q_rows.ptr,
                    d_k_rows.ptr,
                    d_v_rows.ptr,
                    d_alpha_rows.ptr,
                    d_beta_rows.ptr,
                    d_a_log.ptr,
                    d_dt_bias.ptr,
                    d_grouped_output.ptr,
                    nullptr,
                    verifier_rows,
                    n_heads,
                    d_k,
                    d_v,
                    /*chunk_size=*/64,
                    /*use_qk_l2norm=*/true);
            });

        for (int accepted_row = 0; accepted_row < verifier_rows; ++accepted_row)
        {
            SCOPED_TRACE(
                "verifier_rows=" + std::to_string(verifier_rows) +
                " accepted_row=" + std::to_string(accepted_row));

            ASSERT_TRUE(live_kernel.importState(initial_state.data(), nullptr, stream.stream));
            live_kernel.bindVerifierStateCaptureWorkspace(
                d_snapshots.ptr, verifier_rows, state_floats);
            live_kernel.bindSpeculativeStateWorkspace(
                d_speculative_state.ptr, state_floats);
            captured_verifier.launch(stream.stream);

            HipIntBuffer d_accepted_row(accepted_row);
            if (accepted_row == verifier_rows - 1)
            {
                /*
                 * Exercise the production grouped publication contract: the
                 * packed request-zero row and scalar live owner must be
                 * published together before captured decode replay.
                 */
                ASSERT_TRUE(live_kernel.restoreVerifierStateCaptureRowsFromDeviceIndices(
                    nullptr,
                    state_floats,
                    d_accepted_row.ptr,
                    /*request_count=*/1,
                    /*row_index_stride=*/1,
                    stream.stream));
            }
            else
            {
                ASSERT_TRUE(live_kernel.restoreVerifierStateCaptureRowFromDeviceIndex(
                    nullptr, d_accepted_row.ptr, stream.stream));
            }
            live_kernel.bindVerifierStateCaptureWorkspace(nullptr, 0, state_floats);
            live_kernel.bindSpeculativeStateWorkspace(nullptr, state_floats);

            d_replay_output.fill(0.0f);
            captured_decode.launch(stream.stream);
            checkHip(
                hipStreamSynchronize(stream.stream),
                "hipStreamSynchronize(persistent GDN decode replay)");
            std::vector<float> replay_state(static_cast<size_t>(state_floats));
            ASSERT_TRUE(live_kernel.exportState(
                replay_state.data(), nullptr, nullptr));

            ASSERT_TRUE(oracle_kernel.importState(initial_state.data(), nullptr, stream.stream));
            for (int row = 0; row <= accepted_row; ++row)
            {
                ASSERT_TRUE(oracle_kernel.recurrent_step(
                    d_q_rows.ptr + static_cast<size_t>(row) * qk_width,
                    d_k_rows.ptr + static_cast<size_t>(row) * qk_width,
                    d_v_rows.ptr + static_cast<size_t>(row) * value_width,
                    d_alpha_rows.ptr + static_cast<size_t>(row) * n_heads,
                    d_beta_rows.ptr + static_cast<size_t>(row) * n_heads,
                    d_a_log.ptr,
                    d_dt_bias.ptr,
                    d_oracle_row_output.ptr,
                    nullptr,
                    n_heads,
                    d_k,
                    d_v,
                    /*use_qk_l2norm=*/true));
            }
            ASSERT_TRUE(oracle_kernel.recurrent_step(
                d_continuation_q.ptr,
                d_continuation_k.ptr,
                d_continuation_v.ptr,
                d_continuation_alpha.ptr,
                d_continuation_beta.ptr,
                d_a_log.ptr,
                d_dt_bias.ptr,
                d_oracle_output.ptr,
                nullptr,
                n_heads,
                d_k,
                d_v,
                /*use_qk_l2norm=*/true));
            checkHip(
                hipStreamSynchronize(stream.stream),
                "hipStreamSynchronize(serial GDN continuation oracle)");
            std::vector<float> oracle_state(static_cast<size_t>(state_floats));
            ASSERT_TRUE(oracle_kernel.exportState(
                oracle_state.data(), nullptr, nullptr));

            const auto grouped_rows = d_grouped_output.toHost();
            const size_t grouped_row_begin =
                static_cast<size_t>(accepted_row) * value_width;
            const std::vector<float> grouped_row(
                grouped_rows.begin() + grouped_row_begin,
                grouped_rows.begin() + grouped_row_begin + value_width);
            expectByteExactEquivalent(
                "ROCm grouped GDN verifier output row",
                grouped_row,
                d_oracle_row_output.toHost(),
                /*offset=*/0,
                grouped_row.size());

            const auto replay_output = d_replay_output.toHost();
            const auto oracle_output = d_oracle_output.toHost();
            expectByteExactEquivalent(
                "ROCm captured GDN continuation output",
                replay_output,
                oracle_output,
                0,
                replay_output.size());
            expectByteExactEquivalent(
                "ROCm captured GDN continuation state",
                replay_state,
                oracle_state,
                0,
                replay_state.size());
        }
    }
}

/**
 * @brief Prove the captured HIP short-conv graph keeps one stable live-state owner.
 *
 * The 10,240-wide in-place graph mirrors Qwen3.6's production QKV convolution.
 * Device-indexed publication selects each possible accepted row for M=2..16
 * and M=31. Every grouped output row must equal scalar M=1 decode, and replay
 * must match serial continuation in both output bytes and the complete live
 * history buffer without rebuilding the HIP executable.
 */
TEST(Test__ROCmGDNPaddedRealLength, CapturedVerifierAndDecodeShortConvPublicationRuntimeM)
{
    if (!hasROCm())
        GTEST_SKIP() << "No ROCm device available";
    checkHip(hipSetDevice(0), "hipSetDevice");

    constexpr int channels = 10240;
    constexpr int kernel_size = 4;
    constexpr int max_verifier_rows =
        llaminar2::test::kGroupedVerifierRuntimeRows.back();
    constexpr int state_floats = channels * (kernel_size - 1);

    const auto verifier_input = makeSequenceRows(
        max_verifier_rows, channels, max_verifier_rows,
        max_verifier_rows, 0.0095f, 0.0f, 0.0095f);
    const auto continuation_input = makeSequenceRows(
        1, channels, 1, 1, 0.0113f, 0.0f, 0.0113f);
    const auto weight = makeShortConvWeights(channels, kernel_size);
    const auto bias = makeBias(channels);
    const auto initial_state = makeInitialState(
        static_cast<size_t>(state_floats), 0.0027f);

    HipFloatBuffer d_live_verifier(verifier_input);
    HipFloatBuffer d_oracle_verifier(verifier_input);
    HipFloatBuffer d_replay_continuation(continuation_input);
    HipFloatBuffer d_oracle_continuation(continuation_input);
    HipFloatBuffer d_weight(weight);
    HipFloatBuffer d_bias(bias);
    HipFloatBuffer d_snapshots(
        static_cast<size_t>(max_verifier_rows) * state_floats, -77.0f);
    HipFloatBuffer d_speculative_state(static_cast<size_t>(state_floats), 0.0f);
    HipStreamHandle stream;

    ROCmShortConvolution live_kernel(0);
    HipGDNStateOwner live_kernel_state_owner(live_kernel, state_floats);
    live_kernel_state_owner.bindScratch(
        live_kernel, max_verifier_rows * channels);
    live_kernel.setGPUStream(stream.stream);
    live_kernel.bindVerifierStateCaptureWorkspace(nullptr, 0, state_floats);
    live_kernel.bindSpeculativeStateWorkspace(nullptr, state_floats);

    HipCapturedGraph captured_decode(
        stream.stream,
        [&]()
        {
            return live_kernel.forward(
                d_replay_continuation.ptr,
                d_weight.ptr,
                d_bias.ptr,
                d_replay_continuation.ptr,
                nullptr,
                1,
                channels,
                kernel_size,
                /*apply_silu=*/true);
        });

    ROCmShortConvolution oracle_kernel(0);
    HipGDNStateOwner oracle_kernel_state_owner(oracle_kernel, state_floats);
    oracle_kernel_state_owner.bindScratch(
        oracle_kernel, max_verifier_rows * channels);
    oracle_kernel.setGPUStream(stream.stream);

    for (const int verifier_rows : llaminar2::test::kGroupedVerifierRuntimeRows)
    {
        live_kernel.bindVerifierStateCaptureWorkspace(
            d_snapshots.ptr, verifier_rows, state_floats);
        live_kernel.bindSpeculativeStateWorkspace(
            d_speculative_state.ptr, state_floats);
        HipCapturedGraph captured_verifier(
            stream.stream,
            [&]()
            {
                return live_kernel.forward(
                    d_live_verifier.ptr,
                    d_weight.ptr,
                    d_bias.ptr,
                    d_live_verifier.ptr,
                    nullptr,
                    verifier_rows,
                    channels,
                    kernel_size,
                    /*apply_silu=*/true);
            });

        for (int accepted_row = 0; accepted_row < verifier_rows; ++accepted_row)
        {
            SCOPED_TRACE(
                "verifier_rows=" + std::to_string(verifier_rows) +
                " accepted_row=" + std::to_string(accepted_row));

            d_live_verifier.copyFrom(verifier_input);
            d_replay_continuation.copyFrom(continuation_input);
            ASSERT_TRUE(live_kernel.importState(initial_state.data(), nullptr, stream.stream));
            live_kernel.bindVerifierStateCaptureWorkspace(
                d_snapshots.ptr, verifier_rows, state_floats);
            live_kernel.bindSpeculativeStateWorkspace(
                d_speculative_state.ptr, state_floats);
            captured_verifier.launch(stream.stream);

            HipIntBuffer d_accepted_row(accepted_row);
            if (accepted_row == verifier_rows - 1)
            {
                ASSERT_TRUE(live_kernel.restoreVerifierStateCaptureRowsFromDeviceIndices(
                    nullptr,
                    state_floats,
                    d_accepted_row.ptr,
                    /*request_count=*/1,
                    /*row_index_stride=*/1,
                    stream.stream));
            }
            else
            {
                ASSERT_TRUE(live_kernel.restoreVerifierStateCaptureRowFromDeviceIndex(
                    nullptr, d_accepted_row.ptr, stream.stream));
            }
            live_kernel.bindVerifierStateCaptureWorkspace(nullptr, 0, state_floats);
            live_kernel.bindSpeculativeStateWorkspace(nullptr, state_floats);
            captured_decode.launch(stream.stream);
            checkHip(
                hipStreamSynchronize(stream.stream),
                "hipStreamSynchronize(persistent short-conv decode replay)");
            std::vector<float> replay_state(static_cast<size_t>(state_floats));
            ASSERT_TRUE(live_kernel.exportState(
                replay_state.data(), nullptr, nullptr));

            d_oracle_verifier.copyFrom(verifier_input);
            d_oracle_continuation.copyFrom(continuation_input);
            ASSERT_TRUE(oracle_kernel.importState(initial_state.data(), nullptr, stream.stream));
            for (int row = 0; row <= accepted_row; ++row)
            {
                float *row_ptr =
                    d_oracle_verifier.ptr + static_cast<size_t>(row) * channels;
                ASSERT_TRUE(oracle_kernel.forward(
                    row_ptr,
                    d_weight.ptr,
                    d_bias.ptr,
                    row_ptr,
                    nullptr,
                    1,
                    channels,
                    kernel_size,
                    /*apply_silu=*/true));
            }
            ASSERT_TRUE(oracle_kernel.forward(
                d_oracle_continuation.ptr,
                d_weight.ptr,
                d_bias.ptr,
                d_oracle_continuation.ptr,
                nullptr,
                1,
                channels,
                kernel_size,
                /*apply_silu=*/true));
            checkHip(
                hipStreamSynchronize(stream.stream),
                "hipStreamSynchronize(serial short-conv continuation oracle)");
            std::vector<float> oracle_state(static_cast<size_t>(state_floats));
            ASSERT_TRUE(oracle_kernel.exportState(
                oracle_state.data(), nullptr, nullptr));

            const auto grouped_rows = d_live_verifier.toHost();
            const auto scalar_rows = d_oracle_verifier.toHost();
            const size_t grouped_row_begin =
                static_cast<size_t>(accepted_row) * channels;
            const std::vector<float> grouped_row(
                grouped_rows.begin() + grouped_row_begin,
                grouped_rows.begin() + grouped_row_begin + channels);
            const std::vector<float> scalar_row(
                scalar_rows.begin() + grouped_row_begin,
                scalar_rows.begin() + grouped_row_begin + channels);
            expectByteExactEquivalent(
                "ROCm grouped short-conv verifier output row",
                grouped_row,
                scalar_row,
                /*offset=*/0,
                grouped_row.size());

            const auto replay_output = d_replay_continuation.toHost();
            const auto oracle_output = d_oracle_continuation.toHost();
            expectByteExactEquivalent(
                "ROCm captured short-conv continuation output",
                replay_output,
                oracle_output,
                0,
                replay_output.size());
            expectByteExactEquivalent(
                "ROCm captured short-conv continuation state",
                replay_state,
                oracle_state,
                0,
                replay_state.size());
        }
    }
}

#endif
