/**
 * @file Test__GPUSamplingKernels.cpp
 * @brief Integration tests for GPU sampling kernels (argmaxF32, topKF32)
 *
 * **Purpose**: Validates the GPU-side sampling primitives (argmax and top-k)
 * implemented in both CUDA and ROCm backends. These tests mirror the
 * CPU sampler unit tests in Test__Sampler.cpp, adapted for the GPU kernel API.
 *
 * **Tests Cover**:
 * - Argmax: greedy selection on GPU (standard, uniform, peaked, negative, extreme logits)
 * - Top-K: correctness, ordering, boundary conditions, large vocabularies
 * - Numerical stability: very large/small logits, mixed extremes
 * - Edge cases: single element, uniform distribution, all-negative, all-same
 * - Real-world: Qwen2 vocab size (151936), realistic logit distributions
 *
 * **GPU API Model**:
 * - allocate() → hostToDevice() → argmaxF32/topKF32 → free()
 * - argmaxF32/topKF32 perform internal sync + D2H for results (no explicit sync needed)
 *
 * **Backend Selection**:
 * Parameterized over {"CUDA", "ROCm"} — tests skip gracefully if the
 * requested backend is not available.
 *
 * @note Requires CUDA and/or ROCm devices to run. Tests skip gracefully
 *       if the required hardware is not available.
 *
 * @author GitHub Copilot
 * @date June 2026
 */

#include <gtest/gtest.h>
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "backends/IGPUGraphCapture.h"
#include "execution/mtp/MTPRejectionSampler.h"
#include "kernels/common/SamplingMath.h"
#include "utils/Sampler.h"

#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <set>
#include <map>
#include <random>
#include <limits>
#include <array>

using namespace llaminar2;

namespace
{

    // =========================================================================
    // Test Fixture — parameterized over backend name
    // =========================================================================

    class GPUSamplingTest : public ::testing::TestWithParam<std::string>
    {
    protected:
        void SetUp() override
        {
            const auto &backend_name = GetParam();

            if (backend_name == "CUDA")
            {
                backend_ = getCUDABackend();
                if (!backend_)
                    GTEST_SKIP() << "CUDA backend not available";
            }
            else if (backend_name == "ROCm")
            {
                backend_ = getROCmBackend();
                if (!backend_)
                    GTEST_SKIP() << "ROCm backend not available";
            }
            else
            {
                FAIL() << "Unknown backend: " << backend_name;
            }

            device_id_ = 0;

            // Standard logits (5 tokens) — token 2 has highest logit (3.0)
            standard_logits_ = {1.0f, 2.0f, 3.0f, 0.5f, 1.5f};

            // Uniform logits (all same value)
            uniform_logits_ = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f};

            // Single peak logits (one clearly dominant token)
            peaked_logits_ = {0.1f, 0.2f, 10.0f, 0.1f, 0.2f};
        }

        void TearDown() override
        {
            // Release the argmax partial-reduction scratch allocated on first use.
            if (backend_)
            {
                if (argmax_partial_vals_)
                    backend_->free(argmax_partial_vals_, device_id_);
                if (argmax_partial_idxs_)
                    backend_->free(argmax_partial_idxs_, device_id_);
            }
            argmax_partial_vals_ = nullptr;
            argmax_partial_idxs_ = nullptr;
            argmax_partial_capacity_ = 0;
            backend_ = nullptr;
        }

        // ------------------------------------------------------------------
        // Helper: upload host logits to GPU, return device pointer
        // ------------------------------------------------------------------
        void *uploadLogits(const std::vector<float> &logits)
        {
            size_t bytes = logits.size() * sizeof(float);
            void *d_ptr = backend_->allocate(bytes, device_id_);
            EXPECT_NE(d_ptr, nullptr) << "Device allocation failed";
            if (!d_ptr)
                return nullptr;

            bool ok = backend_->hostToDevice(d_ptr, logits.data(), bytes, device_id_);
            EXPECT_TRUE(ok) << "H2D transfer failed";
            if (!ok)
            {
                backend_->free(d_ptr, device_id_);
                return nullptr;
            }
            return d_ptr;
        }

        // ------------------------------------------------------------------
        // Helper: free device memory
        // ------------------------------------------------------------------
        void freeDevice(void *d_ptr)
        {
            if (d_ptr)
                backend_->free(d_ptr, device_id_);
        }

        // ------------------------------------------------------------------
        // Helper: argmax with mandatory device scratch (multi-block reduction).
        //
        // Production callers always supply arena-owned partial-reduction scratch,
        // so the CUDA backend has no single-block fallback. These tests mirror
        // that contract: a persistent scratch pair is allocated lazily on first
        // use and reused across calls, then freed in TearDown.
        // ------------------------------------------------------------------
        bool argmaxF32(void *d_ptr, int n, int device_id,
                       float *out_value, int *out_index)
        {
            if (!argmax_partial_vals_)
            {
                argmax_partial_capacity_ = 1024;
                argmax_partial_vals_ =
                    backend_->allocate(argmax_partial_capacity_ * sizeof(float), device_id_);
                argmax_partial_idxs_ =
                    backend_->allocate(argmax_partial_capacity_ * sizeof(int), device_id_);
            }
            return backend_->argmaxF32(d_ptr, n, device_id, out_value, out_index,
                                       nullptr, argmax_partial_vals_,
                                       argmax_partial_idxs_, argmax_partial_capacity_);
        }

        bool argmaxF32BatchedRows(void *d_ptr, int rows, int cols, int device_id,
                                  float *out_values, int *out_indices)
        {
            if (!argmax_partial_vals_)
            {
                argmax_partial_capacity_ = 1024;
                argmax_partial_vals_ =
                    backend_->allocate(argmax_partial_capacity_ * sizeof(float), device_id_);
                argmax_partial_idxs_ =
                    backend_->allocate(argmax_partial_capacity_ * sizeof(int), device_id_);
            }
            return backend_->argmaxF32BatchedRows(d_ptr,
                                                  rows,
                                                  cols,
                                                  device_id,
                                                  out_values,
                                                  out_indices,
                                                  nullptr,
                                                  argmax_partial_vals_,
                                                  argmax_partial_idxs_,
                                                  argmax_partial_capacity_);
        }

        IBackend *backend_ = nullptr;
        int device_id_ = 0;

        // Persistent argmax partial-reduction scratch (allocated on first use).
        void *argmax_partial_vals_ = nullptr;
        void *argmax_partial_idxs_ = nullptr;
        int argmax_partial_capacity_ = 0;

        std::vector<float> standard_logits_;
        std::vector<float> uniform_logits_;
        std::vector<float> peaked_logits_;
    };

    // =========================================================================
    // Instantiate for both backends
    // =========================================================================
    INSTANTIATE_TEST_SUITE_P(
        GPU,
        GPUSamplingTest,
        ::testing::Values("CUDA", "ROCm"),
        [](const ::testing::TestParamInfo<std::string> &info)
        {
            return info.param; // "CUDA" or "ROCm"
        });

    // =========================================================================
    //  ARGMAX TESTS — mirrors Greedy Sampling from Test__Sampler.cpp
    // =========================================================================

    TEST_P(GPUSamplingTest, Argmax_StandardLogits)
    {
        // Should select token with highest logit (index 2, value 3.0)
        void *d_ptr = uploadLogits(standard_logits_);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, static_cast<int>(standard_logits_.size()),
                                      device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok) << "argmaxF32 not supported on " << GetParam();

        EXPECT_EQ(out_index, 2) << "Argmax should select index 2 (logit 3.0)";
        EXPECT_FLOAT_EQ(out_value, 3.0f) << "Argmax value should be 3.0";

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_UniformLogits)
    {
        // With uniform logits, should select first occurrence (index 0)
        void *d_ptr = uploadLogits(uniform_logits_);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, static_cast<int>(uniform_logits_.size()),
                                      device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, 0) << "Argmax of uniform should select first occurrence";
        EXPECT_FLOAT_EQ(out_value, 2.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_SingleToken)
    {
        std::vector<float> single = {5.0f};
        void *d_ptr = uploadLogits(single);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, 1, device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, 0);
        EXPECT_FLOAT_EQ(out_value, 5.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_Deterministic)
    {
        // Argmax should always return the same result
        void *d_ptr = uploadLogits(standard_logits_);
        ASSERT_NE(d_ptr, nullptr);

        int first_index = -1;
        float first_value = 0.0f;
        bool ok = argmaxF32(d_ptr, static_cast<int>(standard_logits_.size()),
                                      device_id_, &first_value, &first_index);
        ASSERT_TRUE(ok);

        for (int i = 0; i < 10; ++i)
        {
            float out_value = 0.0f;
            int out_index = -1;
            ok = argmaxF32(d_ptr, static_cast<int>(standard_logits_.size()),
                                     device_id_, &out_value, &out_index);
            ASSERT_TRUE(ok);
            EXPECT_EQ(out_index, first_index) << "Iteration " << i;
            EXPECT_FLOAT_EQ(out_value, first_value) << "Iteration " << i;
        }

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_PeakedLogits)
    {
        // Peaked distribution: token 2 has logit 10.0, rest are 0.1-0.2
        void *d_ptr = uploadLogits(peaked_logits_);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, static_cast<int>(peaked_logits_.size()),
                                      device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, 2) << "Argmax should select peaked token";
        EXPECT_FLOAT_EQ(out_value, 10.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_NegativeLogits)
    {
        // All negative logits: max is -1.0 at index 2
        std::vector<float> negative = {-5.0f, -2.0f, -1.0f, -10.0f};
        void *d_ptr = uploadLogits(negative);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, static_cast<int>(negative.size()),
                                      device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, 2) << "Argmax should select least-negative value";
        EXPECT_FLOAT_EQ(out_value, -1.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_ExtremeLogits)
    {
        // Extreme difference: token 2 = 100.0, rest = -1000.0
        std::vector<float> extreme = {-1000.0f, -1000.0f, 100.0f, -1000.0f};
        void *d_ptr = uploadLogits(extreme);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, static_cast<int>(extreme.size()),
                                      device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, 2);
        EXPECT_FLOAT_EQ(out_value, 100.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_AllZeros)
    {
        std::vector<float> zeros = {0.0f, 0.0f, 0.0f, 0.0f};
        void *d_ptr = uploadLogits(zeros);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, static_cast<int>(zeros.size()),
                                      device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, 0) << "All zeros: argmax should select first element";
        EXPECT_FLOAT_EQ(out_value, 0.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_AllSameValue)
    {
        // All same non-zero: should pick index 0
        void *d_ptr = uploadLogits(uniform_logits_);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, static_cast<int>(uniform_logits_.size()),
                                      device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, 0) << "Uniform: argmax should select first occurrence";
        EXPECT_FLOAT_EQ(out_value, 2.0f);

        freeDevice(d_ptr);
    }

    // =========================================================================
    // ARGMAX NUMERICAL STABILITY TESTS
    // =========================================================================

    TEST_P(GPUSamplingTest, Argmax_VeryLargeLogits)
    {
        // Very large values — should not overflow or produce wrong result
        std::vector<float> large = {500.0f, 501.0f, 502.0f, 500.5f};
        void *d_ptr = uploadLogits(large);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, static_cast<int>(large.size()),
                                      device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, 2);
        EXPECT_FLOAT_EQ(out_value, 502.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_VerySmallLogits)
    {
        // Very small (negative) values
        std::vector<float> small = {-500.0f, -501.0f, -499.0f, -500.5f};
        void *d_ptr = uploadLogits(small);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, static_cast<int>(small.size()),
                                      device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, 2) << "Argmax should pick -499.0 (highest)";
        EXPECT_FLOAT_EQ(out_value, -499.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_MixedExtremeLogits)
    {
        // Mix of very large and very small
        std::vector<float> mixed = {-1000.0f, 1000.0f, -1000.0f, -1000.0f};
        void *d_ptr = uploadLogits(mixed);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, static_cast<int>(mixed.size()),
                                      device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, 1);
        EXPECT_FLOAT_EQ(out_value, 1000.0f);

        freeDevice(d_ptr);
    }

    // =========================================================================
    // ARGMAX LARGE VOCABULARY TESTS
    // =========================================================================

    TEST_P(GPUSamplingTest, Argmax_LargeVocab_50K)
    {
        // 50K vocabulary with a peak at a specific index
        const int n = 50000;
        std::vector<float> logits(n, 0.0f);
        logits[12345] = 10.0f;

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, n, device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, 12345);
        EXPECT_FLOAT_EQ(out_value, 10.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_Qwen2VocabSize)
    {
        // Qwen2.5 vocab_size = 151936
        const int n = 151936;
        std::vector<float> logits(n, 0.0f);
        logits[256] = 15.0f;    // Top prediction
        logits[8159] = 14.0f;   // Second
        logits[100160] = 13.5f; // Third

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, n, device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, 256) << "Argmax should pick global max";
        EXPECT_FLOAT_EQ(out_value, 15.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_LargeVocabTieBreaksToLowestTokenId)
    {
        // Regression for CUDA skip-gather greedy decode: equal winning logits
        // must match std::max_element semantics and select the first/lower id,
        // even when the tied candidates land in different reduction lanes.
        const int n = 248320;
        std::vector<float> logits(n, -8.0f);
        logits[248046] = 17.0f;
        logits[248068] = 17.0f;
        logits[1024] = 16.0f;

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, n, device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, 248046);
        EXPECT_FLOAT_EQ(out_value, 17.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_BatchedRowsQwen36VocabMatchesSerialRows)
    {
        // Phase 13.8 MTP verifier rows sample a compact [M, vocab] logits
        // tensor. Keep the Qwen3.6-sized batched-row argmax honest against the
        // serial row path so a sampler bug cannot masquerade as state drift.
        constexpr int rows = 3;
        constexpr int cols = 248320;
        const int expected[rows] = {271, 33075, 248068};

        std::vector<float> logits(
            static_cast<size_t>(rows) * static_cast<size_t>(cols),
            -9.0f);
        for (int row = 0; row < rows; ++row)
        {
            const size_t base = static_cast<size_t>(row) * static_cast<size_t>(cols);
            logits[base + static_cast<size_t>(expected[row])] =
                25.0f + static_cast<float>(row);
            logits[base + static_cast<size_t>(expected[row] + 1)] =
                24.0f + static_cast<float>(row);
        }
        logits[static_cast<size_t>(2) * static_cast<size_t>(cols) + 248100] =
            logits[static_cast<size_t>(2) * static_cast<size_t>(cols) +
                   static_cast<size_t>(expected[2])];

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        float batched_values[rows] = {};
        int batched_indices[rows] = {-1, -1, -1};
        ASSERT_TRUE(argmaxF32BatchedRows(
            d_ptr,
            rows,
            cols,
            device_id_,
            batched_values,
            batched_indices))
            << "argmaxF32BatchedRows not supported on " << GetParam();

        const auto *base = static_cast<const char *>(d_ptr);
        for (int row = 0; row < rows; ++row)
        {
            float serial_value = 0.0f;
            int serial_index = -1;
            void *row_ptr = const_cast<char *>(
                base + static_cast<size_t>(row) *
                           static_cast<size_t>(cols) * sizeof(float));
            ASSERT_TRUE(argmaxF32(row_ptr, cols, device_id_, &serial_value, &serial_index))
                << "serial argmaxF32 failed for row " << row << " on " << GetParam();

            EXPECT_EQ(batched_indices[row], serial_index) << "row=" << row;
            EXPECT_FLOAT_EQ(batched_values[row], serial_value) << "row=" << row;
            EXPECT_EQ(batched_indices[row], expected[row]) << "row=" << row;
        }

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_PeakAtLastElement)
    {
        // Edge case: max at end of large array
        const int n = 100000;
        std::vector<float> logits(n, -1.0f);
        logits[n - 1] = 42.0f;

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, n, device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, n - 1);
        EXPECT_FLOAT_EQ(out_value, 42.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_PeakAtFirstElement)
    {
        // Edge case: max at start of large array
        const int n = 100000;
        std::vector<float> logits(n, -1.0f);
        logits[0] = 42.0f;

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, n, device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, 0);
        EXPECT_FLOAT_EQ(out_value, 42.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_BatchedRows)
    {
        constexpr int rows = 4;
        constexpr int cols = 4096;
        std::vector<float> logits(static_cast<size_t>(rows) * static_cast<size_t>(cols), -7.0f);
        const int expected[rows] = {17, 2048, 4095, 0};
        for (int row = 0; row < rows; ++row)
        {
            logits[static_cast<size_t>(row) * static_cast<size_t>(cols) +
                   static_cast<size_t>(expected[row])] =
                100.0f + static_cast<float>(row);
        }
        logits[static_cast<size_t>(3) * static_cast<size_t>(cols) + 1234] =
            logits[static_cast<size_t>(3) * static_cast<size_t>(cols)];

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        float out_values[rows] = {};
        int out_indices[rows] = {-1, -1, -1, -1};
        ASSERT_TRUE(argmaxF32BatchedRows(d_ptr, rows, cols, device_id_, out_values, out_indices))
            << "argmaxF32BatchedRows not supported on " << GetParam();

        for (int row = 0; row < rows; ++row)
        {
            EXPECT_EQ(out_indices[row], expected[row]) << "row=" << row;
            EXPECT_FLOAT_EQ(
                out_values[row],
                logits[static_cast<size_t>(row) * static_cast<size_t>(cols) +
                       static_cast<size_t>(expected[row])])
                << "row=" << row;
        }

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_BatchedRowsTieBreaksToLowestTokenId)
    {
        constexpr int rows = 2;
        constexpr int cols = 8192;
        std::vector<float> logits(static_cast<size_t>(rows) * static_cast<size_t>(cols), -3.0f);
        logits[2047] = 9.0f;
        logits[4096] = 9.0f;
        logits[static_cast<size_t>(cols) + 7000] = 11.0f;
        logits[static_cast<size_t>(cols) + 123] = 11.0f;

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        float out_values[rows] = {};
        int out_indices[rows] = {-1, -1};
        ASSERT_TRUE(argmaxF32BatchedRows(d_ptr, rows, cols, device_id_, out_values, out_indices))
            << "argmaxF32BatchedRows not supported on " << GetParam();

        EXPECT_EQ(out_indices[0], 2047);
        EXPECT_FLOAT_EQ(out_values[0], 9.0f);
        EXPECT_EQ(out_indices[1], 123);
        EXPECT_FLOAT_EQ(out_values[1], 11.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Argmax_DeviceBatchedRowsSupportsOutputStride)
    {
        /*
         * Request-batched MTP stores sampled draft tokens in request-major
         * slots: slot = draft_index + request * draft_depth.  The GPU argmax
         * primitive therefore needs strided output stores so the runner does
         * not sample on GPU and then re-upload a host shadow before verifier
         * execution.
         */
        constexpr int rows = 3;
        constexpr int cols = 64;
        constexpr int output_stride = 4;
        constexpr int output_span = 1 + (rows - 1) * output_stride;

        std::vector<float> logits(
            static_cast<size_t>(rows) * static_cast<size_t>(cols),
            -20.0f);
        logits[7] = 10.0f;
        logits[static_cast<size_t>(cols) + 31] = 12.0f;
        logits[2 * static_cast<size_t>(cols) + 9] = 11.0f;

        void *d_logits = nullptr;
        void *d_values = nullptr;
        void *d_indices = nullptr;
        void *d_partial_values = nullptr;
        void *d_partial_indices = nullptr;
        auto cleanup = [&]()
        {
            void *ptrs[] = {
                d_logits,
                d_values,
                d_indices,
                d_partial_values,
                d_partial_indices};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        d_logits = backend_->allocate(logits.size() * sizeof(float), device_id_);
        d_values = backend_->allocate(output_span * sizeof(float), device_id_);
        d_indices = backend_->allocate(output_span * sizeof(int), device_id_);
        d_partial_values = backend_->allocate(1024 * sizeof(float), device_id_);
        d_partial_indices = backend_->allocate(1024 * sizeof(int), device_id_);
        ASSERT_NE(d_logits, nullptr);
        ASSERT_NE(d_values, nullptr);
        ASSERT_NE(d_indices, nullptr);
        ASSERT_NE(d_partial_values, nullptr);
        ASSERT_NE(d_partial_indices, nullptr);

        std::array<float, output_span> initial_values{};
        initial_values.fill(-999.0f);
        std::array<int, output_span> initial_indices{};
        initial_indices.fill(-99);

        auto run = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(backend_->hostToDevice(
                    d_logits,
                    logits.data(),
                    logits.size() * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_values,
                    initial_values.data(),
                    initial_values.size() * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_indices,
                    initial_indices.data(),
                    initial_indices.size() * sizeof(int),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->enqueueArgmaxF32BatchedRowsDevice(
                    d_logits,
                    rows,
                    cols,
                    device_id_,
                    stream,
                    d_values,
                    d_indices,
                    d_partial_values,
                    d_partial_indices,
                    1024,
                    output_stride));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run(ctx);
        }

        std::array<float, output_span> values{};
        std::array<int, output_span> indices{};
        ASSERT_TRUE(backend_->deviceToHost(
            values.data(),
            d_values,
            values.size() * sizeof(float),
            device_id_));
        ASSERT_TRUE(backend_->deviceToHost(
            indices.data(),
            d_indices,
            indices.size() * sizeof(int),
            device_id_));

        EXPECT_FLOAT_EQ(values[0], 10.0f);
        EXPECT_FLOAT_EQ(values[output_stride], 12.0f);
        EXPECT_FLOAT_EQ(values[2 * output_stride], 11.0f);
        EXPECT_EQ(indices[0], 7);
        EXPECT_EQ(indices[output_stride], 31);
        EXPECT_EQ(indices[2 * output_stride], 9);
        for (int slot = 0; slot < output_span; ++slot)
        {
            if (slot == 0 || slot == output_stride || slot == 2 * output_stride)
                continue;
            EXPECT_FLOAT_EQ(values[slot], -999.0f) << "slot " << slot;
            EXPECT_EQ(indices[slot], -99) << "slot " << slot;
        }

        cleanup();
    }

    TEST_P(GPUSamplingTest, GreedySpeculativeSummaryIsGraphCapturable)
    {
        using namespace sampling_math;

        constexpr int rows = 3;
        constexpr int cols = 16;
        constexpr int compare_rows = 2;
        const int32_t draft_tokens[rows] = {2, 4, 5};
        const int stop_tokens[1] = {-1};

        std::vector<float> logits(
            static_cast<size_t>(rows) * static_cast<size_t>(cols),
            -10.0f);
        logits[4] = 9.0f;                              // accepts draft_tokens[1]
        logits[static_cast<size_t>(cols) + 7] = 11.0f; // rejects draft_tokens[2]
        logits[2 * static_cast<size_t>(cols) + 6] = 8.0f; // bonus ignored

        int expected_tokens[kSpeculativeBatchMaxOutputTokens] = {};
        int expected_meta[kSpeculativeBatchMetaCount] = {};
        const int expected_verifier_tokens[rows] = {4, 7, 6};
        summarize_greedy_speculative_verify_batch(
            draft_tokens[0],
            expected_verifier_tokens,
            draft_tokens,
            compare_rows,
            stop_tokens,
            /*stop_token_count=*/0,
            expected_tokens,
            expected_meta);
        ASSERT_EQ(expected_meta[kSpecBatchMetaOk], 1);

        void *d_logits = nullptr;
        void *d_argmax_values = nullptr;
        void *d_argmax_indices = nullptr;
        void *d_partial_values = nullptr;
        void *d_partial_indices = nullptr;
        void *d_draft_tokens = nullptr;
        void *d_output_tokens = nullptr;
        void *d_output_meta = nullptr;
        auto cleanup = [&]()
        {
            void *ptrs[] = {
                d_logits,
                d_argmax_values,
                d_argmax_indices,
                d_partial_values,
                d_partial_indices,
                d_draft_tokens,
                d_output_tokens,
                d_output_meta};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        d_logits = backend_->allocate(logits.size() * sizeof(float), device_id_);
        d_argmax_values = backend_->allocate(rows * sizeof(float), device_id_);
        d_argmax_indices = backend_->allocate(rows * sizeof(int), device_id_);
        d_partial_values = backend_->allocate(1024 * sizeof(float), device_id_);
        d_partial_indices = backend_->allocate(1024 * sizeof(int), device_id_);
        d_draft_tokens = backend_->allocate(rows * sizeof(int32_t), device_id_);
        d_output_tokens =
            backend_->allocate(kSpeculativeBatchMaxOutputTokens * sizeof(int), device_id_);
        d_output_meta =
            backend_->allocate(kSpeculativeBatchMetaCount * sizeof(int), device_id_);
        ASSERT_NE(d_logits, nullptr);
        ASSERT_NE(d_argmax_values, nullptr);
        ASSERT_NE(d_argmax_indices, nullptr);
        ASSERT_NE(d_partial_values, nullptr);
        ASSERT_NE(d_partial_indices, nullptr);
        ASSERT_NE(d_draft_tokens, nullptr);
        ASSERT_NE(d_output_tokens, nullptr);
        ASSERT_NE(d_output_meta, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);

                ASSERT_TRUE(backend_->hostToDevice(
                    d_logits,
                    logits.data(),
                    logits.size() * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_draft_tokens,
                    draft_tokens,
                    sizeof(draft_tokens),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                EXPECT_FALSE(backend_->enqueueArgmaxF32BatchedRowsDevice(
                    d_logits,
                    rows,
                    cols,
                    device_id_,
                    nullptr,
                    d_argmax_values,
                    d_argmax_indices,
                    d_partial_values,
                    d_partial_indices,
                    1024))
                    << "device-resident batched argmax must reject the legacy default/null stream";
                EXPECT_FALSE(backend_->enqueueSummarizeGreedySpeculativeVerifyBatch(
                    d_argmax_indices,
                    d_draft_tokens,
                    compare_rows,
                    draft_tokens[0],
                    stop_tokens,
                    0,
                    device_id_,
                    nullptr,
                    d_output_tokens,
                    d_output_meta))
                    << "greedy speculative summary must reject the legacy default/null stream";

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueArgmaxF32BatchedRowsDevice(
                    d_logits,
                    rows,
                    cols,
                    device_id_,
                    stream,
                    d_argmax_values,
                    d_argmax_indices,
                    d_partial_values,
                    d_partial_indices,
                    1024));
                ASSERT_TRUE(backend_->enqueueSummarizeGreedySpeculativeVerifyBatch(
                    d_argmax_indices,
                    d_draft_tokens,
                    compare_rows,
                    draft_tokens[0],
                    stop_tokens,
                    0,
                    device_id_,
                    stream,
                    d_output_tokens,
                    d_output_meta));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        int gpu_tokens[kSpeculativeBatchMaxOutputTokens] = {};
        int gpu_meta[kSpeculativeBatchMetaCount] = {};
        ASSERT_TRUE(backend_->deviceToHost(
            gpu_tokens,
            d_output_tokens,
            sizeof(gpu_tokens),
            device_id_));
        ASSERT_TRUE(backend_->deviceToHost(
            gpu_meta,
            d_output_meta,
            sizeof(gpu_meta),
            device_id_));

        EXPECT_EQ(gpu_meta[kSpecBatchMetaOk], 1);
        for (int i = 0; i < kSpeculativeBatchMetaCount; ++i)
            EXPECT_EQ(gpu_meta[i], expected_meta[i]) << "meta index " << i;
        for (int i = 0; i < kSpeculativeBatchMaxOutputTokens; ++i)
            EXPECT_EQ(gpu_tokens[i], expected_tokens[i]) << "token index " << i;

        cleanup();
    }

    TEST_P(GPUSamplingTest, GreedySpeculativeSummaryQwen36VocabMatchesHostRows)
    {
        using namespace sampling_math;

        /*
         * Regression shape for Qwen3.6 MoE all-position MTP.  The compact
         * vLLM-style path leaves row argmax tokens on device and feeds them
         * directly into the greedy verifier summary.  Keep this equivalent to
         * the host-visible batched argmax path at the real Qwen3.6 vocab size,
         * including the final bonus-ready row.
         */
        constexpr int rows = 2;
        constexpr int cols = 248320;
        constexpr int compare_rows = 1;
        const int32_t draft_tokens[rows] = {198, 248045};
        const int expected_verifier_tokens[rows] = {248045, 248068};

        std::vector<float> logits(
            static_cast<size_t>(rows) * static_cast<size_t>(cols),
            -12.0f);
        logits[248045] = 20.0f;
        logits[74455] = 19.0f;
        logits[static_cast<size_t>(cols) + 248068] = 20.0f;
        logits[static_cast<size_t>(cols) + 74455] = 18.5f;

        int expected_tokens[kSpeculativeBatchMaxOutputTokens] = {};
        int expected_meta[kSpeculativeBatchMetaCount] = {};
        summarize_greedy_speculative_verify_batch(
            draft_tokens[0],
            expected_verifier_tokens,
            draft_tokens,
            compare_rows,
            /*stop_tokens=*/nullptr,
            /*stop_token_count=*/0,
            expected_tokens,
            expected_meta);
        ASSERT_EQ(expected_meta[kSpecBatchMetaOk], 1);
        ASSERT_EQ(expected_meta[kSpecBatchMetaReadyToken], 248068);

        void *d_logits = nullptr;
        void *d_argmax_values = nullptr;
        void *d_argmax_indices = nullptr;
        void *d_partial_values = nullptr;
        void *d_partial_indices = nullptr;
        void *d_draft_tokens = nullptr;
        void *d_output_tokens = nullptr;
        void *d_output_meta = nullptr;
        auto cleanup = [&]()
        {
            void *ptrs[] = {
                d_logits,
                d_argmax_values,
                d_argmax_indices,
                d_partial_values,
                d_partial_indices,
                d_draft_tokens,
                d_output_tokens,
                d_output_meta};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        d_logits = backend_->allocate(logits.size() * sizeof(float), device_id_);
        d_argmax_values = backend_->allocate(rows * sizeof(float), device_id_);
        d_argmax_indices = backend_->allocate(rows * sizeof(int), device_id_);
        d_partial_values = backend_->allocate(1024 * sizeof(float), device_id_);
        d_partial_indices = backend_->allocate(1024 * sizeof(int), device_id_);
        d_draft_tokens = backend_->allocate(rows * sizeof(int32_t), device_id_);
        d_output_tokens =
            backend_->allocate(kSpeculativeBatchMaxOutputTokens * sizeof(int), device_id_);
        d_output_meta =
            backend_->allocate(kSpeculativeBatchMetaCount * sizeof(int), device_id_);
        ASSERT_NE(d_logits, nullptr);
        ASSERT_NE(d_argmax_values, nullptr);
        ASSERT_NE(d_argmax_indices, nullptr);
        ASSERT_NE(d_partial_values, nullptr);
        ASSERT_NE(d_partial_indices, nullptr);
        ASSERT_NE(d_draft_tokens, nullptr);
        ASSERT_NE(d_output_tokens, nullptr);
        ASSERT_NE(d_output_meta, nullptr);

        auto run_on_stream = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);

                ASSERT_TRUE(backend_->hostToDevice(
                    d_logits,
                    logits.data(),
                    logits.size() * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_draft_tokens,
                    draft_tokens,
                    sizeof(draft_tokens),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->enqueueArgmaxF32BatchedRowsDevice(
                    d_logits,
                    rows,
                    cols,
                    device_id_,
                    stream,
                    d_argmax_values,
                    d_argmax_indices,
                    d_partial_values,
                    d_partial_indices,
                    1024));
                ASSERT_TRUE(backend_->enqueueSummarizeGreedySpeculativeVerifyBatch(
                    d_argmax_indices,
                    d_draft_tokens,
                    compare_rows,
                    draft_tokens[0],
                    /*stop_tokens_host=*/nullptr,
                    /*stop_token_count=*/0,
                    device_id_,
                    stream,
                    d_output_tokens,
                    d_output_meta));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_on_stream(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_on_stream(ctx);
        }

        int gpu_indices[rows] = {};
        int gpu_tokens[kSpeculativeBatchMaxOutputTokens] = {};
        int gpu_meta[kSpeculativeBatchMetaCount] = {};
        ASSERT_TRUE(backend_->deviceToHost(
            gpu_indices,
            d_argmax_indices,
            sizeof(gpu_indices),
            device_id_));
        ASSERT_TRUE(backend_->deviceToHost(
            gpu_tokens,
            d_output_tokens,
            sizeof(gpu_tokens),
            device_id_));
        ASSERT_TRUE(backend_->deviceToHost(
            gpu_meta,
            d_output_meta,
            sizeof(gpu_meta),
            device_id_));

        EXPECT_EQ(gpu_indices[0], expected_verifier_tokens[0]);
        EXPECT_EQ(gpu_indices[1], expected_verifier_tokens[1]);
        for (int i = 0; i < kSpeculativeBatchMetaCount; ++i)
            EXPECT_EQ(gpu_meta[i], expected_meta[i]) << "meta index " << i;
        for (int i = 0; i < kSpeculativeBatchMaxOutputTokens; ++i)
            EXPECT_EQ(gpu_tokens[i], expected_tokens[i]) << "token index " << i;

        cleanup();
    }

    TEST_P(GPUSamplingTest, SpeculativePublicationMetadataDerivationCaptures)
    {
        using namespace sampling_math;

        constexpr int request_count = 3;
        constexpr int padded_state_rows_per_request = 4;
        constexpr int max_state_commit_rows = 3;
        constexpr int meta_stride = kSpeculativeBatchMetaCount;

        std::array<int, request_count * meta_stride> meta{};
        std::array<int, request_count * kSpeculativeBatchMaxOutputTokens>
            output_tokens{};
        std::array<int, request_count> base_cached_tokens = {100, 200, 300};

        const int accept_rows[] = {11, 12};
        const int accept_flags[] = {1, 1};
        summarize_speculative_verify_batch(
            /*first_token=*/10,
            accept_rows,
            accept_flags,
            /*row_count=*/2,
            /*stop_tokens=*/nullptr,
            /*stop_token_count=*/0,
            /*bonus_ready_token=*/13,
            /*has_bonus_ready_token=*/1,
            output_tokens.data(),
            meta.data());

        const int reject_rows[] = {21, 22};
        const int reject_flags[] = {0, 1};
        summarize_speculative_verify_batch(
            /*first_token=*/20,
            reject_rows,
            reject_flags,
            /*row_count=*/2,
            /*stop_tokens=*/nullptr,
            /*stop_token_count=*/0,
            /*bonus_ready_token=*/23,
            /*has_bonus_ready_token=*/1,
            output_tokens.data() + kSpeculativeBatchMaxOutputTokens,
            meta.data() + meta_stride);

        meta[2 * meta_stride + kSpecBatchMetaOk] = 1;
        meta[2 * meta_stride + kSpecBatchMetaTargetVerifierStateCommitCount] =
            max_state_commit_rows + 1;

        void *d_meta = backend_->allocate(meta.size() * sizeof(int), device_id_);
        void *d_base = backend_->allocate(base_cached_tokens.size() * sizeof(int), device_id_);
        void *d_output_tokens = backend_->allocate(output_tokens.size() * sizeof(int), device_id_);
        void *d_restore_rows = backend_->allocate(request_count * sizeof(int), device_id_);
        void *d_target_cached_tokens = backend_->allocate(request_count * sizeof(int), device_id_);
        void *d_accepted_state_counts = backend_->allocate(request_count * sizeof(int), device_id_);
        void *d_ok = backend_->allocate(request_count * sizeof(int), device_id_);
        void *d_next_condition_tokens = backend_->allocate(request_count * sizeof(int), device_id_);
        void *d_all_drafts_accepted = backend_->allocate(request_count * sizeof(int), device_id_);
        void *d_stopped = backend_->allocate(request_count * sizeof(int), device_id_);

        auto cleanup = [&]()
        {
            void *ptrs[] = {
                d_meta,
                d_base,
                d_output_tokens,
                d_restore_rows,
                d_target_cached_tokens,
                d_accepted_state_counts,
                d_ok,
                d_next_condition_tokens,
                d_all_drafts_accepted,
                d_stopped};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        ASSERT_NE(d_meta, nullptr);
        ASSERT_NE(d_base, nullptr);
        ASSERT_NE(d_output_tokens, nullptr);
        ASSERT_NE(d_restore_rows, nullptr);
        ASSERT_NE(d_target_cached_tokens, nullptr);
        ASSERT_NE(d_accepted_state_counts, nullptr);
        ASSERT_NE(d_ok, nullptr);
        ASSERT_NE(d_next_condition_tokens, nullptr);
        ASSERT_NE(d_all_drafts_accepted, nullptr);
        ASSERT_NE(d_stopped, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);

                ASSERT_TRUE(backend_->hostToDevice(
                    d_meta,
                    meta.data(),
                    meta.size() * sizeof(int),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_base,
                    base_cached_tokens.data(),
                    base_cached_tokens.size() * sizeof(int),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_output_tokens,
                    output_tokens.data(),
                    output_tokens.size() * sizeof(int),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                EXPECT_FALSE(backend_->enqueueDeriveSpeculativePublicationMetadata(
                    d_meta,
                    meta_stride,
                    d_base,
                    request_count,
                    padded_state_rows_per_request,
                    max_state_commit_rows,
                    device_id_,
                    nullptr,
                    d_restore_rows,
                    d_target_cached_tokens,
                    d_accepted_state_counts,
                    d_ok,
                    d_next_condition_tokens,
                    d_output_tokens,
                    kSpeculativeBatchMaxOutputTokens,
                    d_all_drafts_accepted,
                    d_stopped))
                    << "publication metadata derivation must reject the legacy default/null stream";

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueDeriveSpeculativePublicationMetadata(
                    d_meta,
                    meta_stride,
                    d_base,
                    request_count,
                    padded_state_rows_per_request,
                    max_state_commit_rows,
                    device_id_,
                    stream,
                    d_restore_rows,
                    d_target_cached_tokens,
                    d_accepted_state_counts,
                    d_ok,
                    d_next_condition_tokens,
                    d_output_tokens,
                    kSpeculativeBatchMaxOutputTokens,
                    d_all_drafts_accepted,
                    d_stopped));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::array<int, request_count> restore_rows{};
        std::array<int, request_count> target_cached_tokens{};
        std::array<int, request_count> accepted_state_counts{};
        std::array<int, request_count> ok{};
        std::array<int, request_count> next_condition_tokens{};
        std::array<int, request_count> all_drafts_accepted{};
        std::array<int, request_count> stopped{};
        ASSERT_TRUE(backend_->deviceToHost(
            restore_rows.data(), d_restore_rows,
            restore_rows.size() * sizeof(int), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(
            target_cached_tokens.data(), d_target_cached_tokens,
            target_cached_tokens.size() * sizeof(int), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(
            accepted_state_counts.data(), d_accepted_state_counts,
            accepted_state_counts.size() * sizeof(int), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(
            ok.data(), d_ok,
            ok.size() * sizeof(int), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(
            next_condition_tokens.data(), d_next_condition_tokens,
            next_condition_tokens.size() * sizeof(int), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(
            all_drafts_accepted.data(), d_all_drafts_accepted,
            all_drafts_accepted.size() * sizeof(int), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(
            stopped.data(), d_stopped,
            stopped.size() * sizeof(int), device_id_));

        cleanup();

        EXPECT_EQ(ok[0], 1);
        EXPECT_EQ(accepted_state_counts[0], 3);
        EXPECT_EQ(restore_rows[0], 2);
        EXPECT_EQ(target_cached_tokens[0], 103);
        EXPECT_EQ(next_condition_tokens[0], 13);
        EXPECT_EQ(all_drafts_accepted[0], 1);
        EXPECT_EQ(stopped[0], 0);

        EXPECT_EQ(ok[1], 1);
        EXPECT_EQ(accepted_state_counts[1], 1);
        EXPECT_EQ(restore_rows[1], 4);
        EXPECT_EQ(target_cached_tokens[1], 201);
        EXPECT_EQ(next_condition_tokens[1], 21);
        EXPECT_EQ(all_drafts_accepted[1], 0);
        EXPECT_EQ(stopped[1], 0);

        EXPECT_EQ(ok[2], 0);
        EXPECT_EQ(accepted_state_counts[2], 0);
        EXPECT_EQ(restore_rows[2], -1);
        EXPECT_EQ(target_cached_tokens[2], 300);
        EXPECT_EQ(next_condition_tokens[2], -1);
        EXPECT_EQ(all_drafts_accepted[2], 0);
        EXPECT_EQ(stopped[2], 0);
    }

    // =========================================================================
    // TOP-K TESTS — mirrors Top-K Sampling from Test__Sampler.cpp
    // =========================================================================

    TEST_P(GPUSamplingTest, TopK_K1_IsArgmax)
    {
        // Top-k with k=1 should be equivalent to argmax
        void *d_ptr = uploadLogits(standard_logits_);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = backend_->topKF32(d_ptr, static_cast<int>(standard_logits_.size()),
                                    1, device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok) << "topKF32 not supported on " << GetParam();

        EXPECT_EQ(out_index, 2) << "Top-1 should select index 2 (logit 3.0)";
        EXPECT_FLOAT_EQ(out_value, 3.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_K2_CorrectTokens)
    {
        // Top-2 of standard_logits: idx 2 (3.0), idx 1 (2.0) — descending order
        void *d_ptr = uploadLogits(standard_logits_);
        ASSERT_NE(d_ptr, nullptr);

        std::vector<float> values(2);
        std::vector<int> indices(2);
        bool ok = backend_->topKF32(d_ptr, static_cast<int>(standard_logits_.size()),
                                    2, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        // Results should be in descending order
        EXPECT_EQ(indices[0], 2) << "Rank 0 should be index 2 (logit 3.0)";
        EXPECT_EQ(indices[1], 1) << "Rank 1 should be index 1 (logit 2.0)";
        EXPECT_FLOAT_EQ(values[0], 3.0f);
        EXPECT_FLOAT_EQ(values[1], 2.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_K3_CorrectRanking)
    {
        // Top-3 of standard_logits: idx 2 (3.0), idx 1 (2.0), idx 4 (1.5)
        void *d_ptr = uploadLogits(standard_logits_);
        ASSERT_NE(d_ptr, nullptr);

        std::vector<float> values(3);
        std::vector<int> indices(3);
        bool ok = backend_->topKF32(d_ptr, static_cast<int>(standard_logits_.size()),
                                    3, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        EXPECT_EQ(indices[0], 2) << "Rank 0 should be index 2 (logit 3.0)";
        EXPECT_EQ(indices[1], 1) << "Rank 1 should be index 1 (logit 2.0)";
        EXPECT_EQ(indices[2], 4) << "Rank 2 should be index 4 (logit 1.5)";
        EXPECT_FLOAT_EQ(values[0], 3.0f);
        EXPECT_FLOAT_EQ(values[1], 2.0f);
        EXPECT_FLOAT_EQ(values[2], 1.5f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_KEqualsVocabSize)
    {
        // k equals vocab size — should return all elements sorted descending
        int n = static_cast<int>(standard_logits_.size());
        void *d_ptr = uploadLogits(standard_logits_);
        ASSERT_NE(d_ptr, nullptr);

        std::vector<float> values(n);
        std::vector<int> indices(n);
        bool ok = backend_->topKF32(d_ptr, n, n, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        // Should be sorted descending: 3.0, 2.0, 1.5, 1.0, 0.5
        EXPECT_FLOAT_EQ(values[0], 3.0f);
        EXPECT_EQ(indices[0], 2);
        EXPECT_FLOAT_EQ(values[n - 1], 0.5f);
        EXPECT_EQ(indices[n - 1], 3);

        // Verify descending order
        for (int i = 1; i < n; ++i)
        {
            EXPECT_GE(values[i - 1], values[i])
                << "Values should be in descending order at position " << i;
        }

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_DescendingOrder)
    {
        // Verify results are always in descending value order
        std::vector<float> logits = {5.0f, 1.0f, 3.0f, 7.0f, 2.0f, 6.0f, 4.0f};
        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        const int k = 5;
        std::vector<float> values(k);
        std::vector<int> indices(k);
        bool ok = backend_->topKF32(d_ptr, static_cast<int>(logits.size()),
                                    k, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        // Top-5 sorted descending: 7.0(3), 6.0(5), 5.0(0), 4.0(6), 3.0(2)
        EXPECT_FLOAT_EQ(values[0], 7.0f);
        EXPECT_FLOAT_EQ(values[1], 6.0f);
        EXPECT_FLOAT_EQ(values[2], 5.0f);
        EXPECT_FLOAT_EQ(values[3], 4.0f);
        EXPECT_FLOAT_EQ(values[4], 3.0f);

        EXPECT_EQ(indices[0], 3);
        EXPECT_EQ(indices[1], 5);
        EXPECT_EQ(indices[2], 0);
        EXPECT_EQ(indices[3], 6);
        EXPECT_EQ(indices[4], 2);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_PeakedLogits)
    {
        // Peaked distribution: top-3 should return the peak + 2 closest
        void *d_ptr = uploadLogits(peaked_logits_);
        ASSERT_NE(d_ptr, nullptr);

        std::vector<float> values(3);
        std::vector<int> indices(3);
        bool ok = backend_->topKF32(d_ptr, static_cast<int>(peaked_logits_.size()),
                                    3, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        // Peak at idx 2 (10.0), then idx 1 and 4 (0.2 each), then idx 0 and 3 (0.1 each)
        EXPECT_EQ(indices[0], 2) << "Peak token should be rank 0";
        EXPECT_FLOAT_EQ(values[0], 10.0f);
        // Equal logits are part of the sampler contract: lower token id wins.
        EXPECT_EQ(indices[1], 1);
        EXPECT_EQ(indices[2], 4);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_NegativeLogits)
    {
        std::vector<float> negative = {-5.0f, -2.0f, -1.0f, -10.0f};
        void *d_ptr = uploadLogits(negative);
        ASSERT_NE(d_ptr, nullptr);

        std::vector<float> values(3);
        std::vector<int> indices(3);
        bool ok = backend_->topKF32(d_ptr, static_cast<int>(negative.size()),
                                    3, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        // Descending: -1.0(2), -2.0(1), -5.0(0)
        EXPECT_EQ(indices[0], 2);
        EXPECT_EQ(indices[1], 1);
        EXPECT_EQ(indices[2], 0);
        EXPECT_FLOAT_EQ(values[0], -1.0f);
        EXPECT_FLOAT_EQ(values[1], -2.0f);
        EXPECT_FLOAT_EQ(values[2], -5.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_SingleElement)
    {
        std::vector<float> single = {7.0f};
        void *d_ptr = uploadLogits(single);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = backend_->topKF32(d_ptr, 1, 1, device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, 0);
        EXPECT_FLOAT_EQ(out_value, 7.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_UniformLogits)
    {
        // All same value: top-k should return the lowest token ids in order.
        void *d_ptr = uploadLogits(uniform_logits_);
        ASSERT_NE(d_ptr, nullptr);

        const int k = 3;
        std::vector<float> values(k);
        std::vector<int> indices(k);
        bool ok = backend_->topKF32(d_ptr, static_cast<int>(uniform_logits_.size()),
                                    k, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        // All values should be 2.0 and ties should resolve by token id.
        for (int i = 0; i < k; ++i)
        {
            EXPECT_FLOAT_EQ(values[i], 2.0f) << "Position " << i;
            EXPECT_EQ(indices[i], i) << "Top-k ties must be deterministic";
        }

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_AllZeros)
    {
        std::vector<float> zeros = {0.0f, 0.0f, 0.0f, 0.0f};
        void *d_ptr = uploadLogits(zeros);
        ASSERT_NE(d_ptr, nullptr);

        const int k = 2;
        std::vector<float> values(k);
        std::vector<int> indices(k);
        bool ok = backend_->topKF32(d_ptr, static_cast<int>(zeros.size()),
                                    k, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        for (int i = 0; i < k; ++i)
        {
            EXPECT_FLOAT_EQ(values[i], 0.0f);
            EXPECT_EQ(indices[i], i);
        }

        freeDevice(d_ptr);
    }

    // =========================================================================
    // TOP-K NUMERICAL STABILITY TESTS
    // =========================================================================

    TEST_P(GPUSamplingTest, TopK_VeryLargeLogits)
    {
        std::vector<float> large = {500.0f, 501.0f, 502.0f, 500.5f};
        void *d_ptr = uploadLogits(large);
        ASSERT_NE(d_ptr, nullptr);

        const int k = 3;
        std::vector<float> values(k);
        std::vector<int> indices(k);
        bool ok = backend_->topKF32(d_ptr, static_cast<int>(large.size()),
                                    k, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        EXPECT_EQ(indices[0], 2);
        EXPECT_FLOAT_EQ(values[0], 502.0f);

        // Should be descending
        for (int i = 1; i < k; ++i)
        {
            EXPECT_GE(values[i - 1], values[i]);
        }

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_VerySmallLogits)
    {
        std::vector<float> small = {-500.0f, -501.0f, -499.0f, -500.5f};
        void *d_ptr = uploadLogits(small);
        ASSERT_NE(d_ptr, nullptr);

        const int k = 2;
        std::vector<float> values(k);
        std::vector<int> indices(k);
        bool ok = backend_->topKF32(d_ptr, static_cast<int>(small.size()),
                                    k, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        // Top-2: -499.0(2), -500.0(0)
        EXPECT_EQ(indices[0], 2);
        EXPECT_FLOAT_EQ(values[0], -499.0f);
        EXPECT_EQ(indices[1], 0);
        EXPECT_FLOAT_EQ(values[1], -500.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_MixedExtremeLogits)
    {
        std::vector<float> mixed = {-1000.0f, 1000.0f, -1000.0f, -1000.0f};
        void *d_ptr = uploadLogits(mixed);
        ASSERT_NE(d_ptr, nullptr);

        const int k = 2;
        std::vector<float> values(k);
        std::vector<int> indices(k);
        bool ok = backend_->topKF32(d_ptr, static_cast<int>(mixed.size()),
                                    k, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        EXPECT_EQ(indices[0], 1) << "Token 1 (1000.0) should be rank 0";
        EXPECT_FLOAT_EQ(values[0], 1000.0f);

        freeDevice(d_ptr);
    }

    // =========================================================================
    // TOP-K LARGE VOCABULARY TESTS
    // =========================================================================

    TEST_P(GPUSamplingTest, TopK_LargeVocab_50K)
    {
        // Large vocabulary with known top-3
        const int n = 50000;
        std::vector<float> logits(n, 0.0f);
        logits[100] = 5.0f;
        logits[200] = 4.0f;
        logits[300] = 3.0f;

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        const int k = 3;
        std::vector<float> values(k);
        std::vector<int> indices(k);
        bool ok = backend_->topKF32(d_ptr, n, k, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        EXPECT_EQ(indices[0], 100);
        EXPECT_EQ(indices[1], 200);
        EXPECT_EQ(indices[2], 300);
        EXPECT_FLOAT_EQ(values[0], 5.0f);
        EXPECT_FLOAT_EQ(values[1], 4.0f);
        EXPECT_FLOAT_EQ(values[2], 3.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_Qwen2VocabSize)
    {
        // Qwen2.5 vocab_size = 151936, realistic distribution with top-5
        const int n = 151936;
        std::vector<float> logits(n, 0.0f);
        logits[256] = 15.0f;
        logits[8159] = 14.0f;
        logits[100160] = 13.5f;
        logits[72363] = 13.0f;
        logits[105797] = 12.8f;

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        const int k = 5;
        std::vector<float> values(k);
        std::vector<int> indices(k);
        bool ok = backend_->topKF32(d_ptr, n, k, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        // Verify correct ranking
        EXPECT_EQ(indices[0], 256) << "Rank 0 should be token 256 (logit 15.0)";
        EXPECT_EQ(indices[1], 8159) << "Rank 1 should be token 8159 (logit 14.0)";
        EXPECT_EQ(indices[2], 100160) << "Rank 2 should be token 100160 (logit 13.5)";
        EXPECT_EQ(indices[3], 72363) << "Rank 3 should be token 72363 (logit 13.0)";
        EXPECT_EQ(indices[4], 105797) << "Rank 4 should be token 105797 (logit 12.8)";

        EXPECT_FLOAT_EQ(values[0], 15.0f);
        EXPECT_FLOAT_EQ(values[1], 14.0f);
        EXPECT_FLOAT_EQ(values[2], 13.5f);
        EXPECT_FLOAT_EQ(values[3], 13.0f);
        EXPECT_FLOAT_EQ(values[4], 12.8f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_Qwen2VocabSize_K40)
    {
        // Realistic scenario: k=40 (default) on full Qwen2 vocab
        const int n = 151936;
        const int k = 40;

        // Create a distribution with known top-40
        std::vector<float> logits(n, -10.0f);
        for (int i = 0; i < k; ++i)
        {
            logits[i * 1000] = 20.0f - static_cast<float>(i) * 0.5f;
        }
        // logits[0]=20, logits[1000]=19.5, logits[2000]=19.0, ..., logits[39000]=0.5

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        std::vector<float> values(k);
        std::vector<int> indices(k);
        bool ok = backend_->topKF32(d_ptr, n, k, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        // Verify descending order
        for (int i = 1; i < k; ++i)
        {
            EXPECT_GE(values[i - 1], values[i])
                << "Values should be descending at position " << i;
        }

        // Verify top-1
        EXPECT_EQ(indices[0], 0) << "Rank 0 should be index 0 (logit 20.0)";
        EXPECT_FLOAT_EQ(values[0], 20.0f);

        // Verify all top-k indices are from our planted values
        for (int i = 0; i < k; ++i)
        {
            EXPECT_EQ(indices[i] % 1000, 0)
                << "Index " << indices[i] << " at rank " << i << " should be a multiple of 1000";
        }

        freeDevice(d_ptr);
    }

    // =========================================================================
    // TOP-K EDGE CASE TESTS
    // =========================================================================

    TEST_P(GPUSamplingTest, TopK_PeakAtLastElement)
    {
        // Peak at end of large array
        const int n = 10000;
        std::vector<float> logits(n, -1.0f);
        logits[n - 1] = 42.0f;

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = backend_->topKF32(d_ptr, n, 1, device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, n - 1);
        EXPECT_FLOAT_EQ(out_value, 42.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_PeakAtFirstElement)
    {
        // Peak at start of large array
        const int n = 10000;
        std::vector<float> logits(n, -1.0f);
        logits[0] = 42.0f;

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        float out_value = 0.0f;
        int out_index = -1;
        bool ok = backend_->topKF32(d_ptr, n, 1, device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);

        EXPECT_EQ(out_index, 0);
        EXPECT_FLOAT_EQ(out_value, 42.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_DuplicateValues)
    {
        // Multiple elements with same value: lower token ids define the tie order.
        std::vector<float> dups = {5.0f, 5.0f, 5.0f, 3.0f, 3.0f, 1.0f};
        void *d_ptr = uploadLogits(dups);
        ASSERT_NE(d_ptr, nullptr);

        const int k = 4;
        std::vector<float> values(k);
        std::vector<int> indices(k);
        bool ok = backend_->topKF32(d_ptr, static_cast<int>(dups.size()),
                                    k, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        // Top-4 should be: three 5.0s, then one 3.0
        EXPECT_FLOAT_EQ(values[0], 5.0f);
        EXPECT_FLOAT_EQ(values[1], 5.0f);
        EXPECT_FLOAT_EQ(values[2], 5.0f);
        EXPECT_FLOAT_EQ(values[3], 3.0f);

        EXPECT_EQ(indices[0], 0);
        EXPECT_EQ(indices[1], 1);
        EXPECT_EQ(indices[2], 2);
        EXPECT_EQ(indices[3], 3);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_K128_LargeK)
    {
        // Test a large k value (128) — upper bound that fits GPU shared memory
        // Note: k=256 exceeds 48KB shared memory on some GPUs (32 threads × 256 × 8B = 64KB)
        const int n = 1000;
        const int k = 128;

        // Create descending logits so ranking is trivial
        std::vector<float> logits(n);
        for (int i = 0; i < n; ++i)
        {
            logits[i] = static_cast<float>(n - i);
        }
        // logits[0]=1000, logits[1]=999, ..., logits[999]=1

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        std::vector<float> values(k);
        std::vector<int> indices(k);
        bool ok = backend_->topKF32(d_ptr, n, k, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        // Top element should be index 0 with value 1000
        EXPECT_EQ(indices[0], 0);
        EXPECT_FLOAT_EQ(values[0], 1000.0f);

        // Last top-k element should be index 127 with value 873
        EXPECT_EQ(indices[k - 1], k - 1);
        EXPECT_FLOAT_EQ(values[k - 1], static_cast<float>(n - k + 1));

        // Verify strict descending
        for (int i = 1; i < k; ++i)
        {
            EXPECT_GT(values[i - 1], values[i])
                << "Values should be strictly descending at position " << i;
        }

        freeDevice(d_ptr);
    }

    // =========================================================================
    // ARGMAX vs TOP-K CONSISTENCY TESTS
    // =========================================================================

    TEST_P(GPUSamplingTest, ArgmaxVsTopK1_Consistent)
    {
        // argmax and topK(k=1) should give identical results
        void *d_ptr = uploadLogits(standard_logits_);
        ASSERT_NE(d_ptr, nullptr);

        float argmax_value = 0.0f;
        int argmax_index = -1;
        bool ok1 = argmaxF32(d_ptr, static_cast<int>(standard_logits_.size()),
                                       device_id_, &argmax_value, &argmax_index);

        float topk_value = 0.0f;
        int topk_index = -1;
        bool ok2 = backend_->topKF32(d_ptr, static_cast<int>(standard_logits_.size()),
                                     1, device_id_, &topk_value, &topk_index);

        if (ok1 && ok2)
        {
            EXPECT_EQ(argmax_index, topk_index) << "argmax and topK(1) index should match";
            EXPECT_FLOAT_EQ(argmax_value, topk_value) << "argmax and topK(1) value should match";
        }

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, ArgmaxVsTopK1_LargeVocab)
    {
        // Consistency check on large vocab
        const int n = 151936;
        std::vector<float> logits(n, 0.0f);
        logits[75000] = 99.0f;

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        float argmax_value = 0.0f, topk_value = 0.0f;
        int argmax_index = -1, topk_index = -1;

        bool ok1 = argmaxF32(d_ptr, n, device_id_, &argmax_value, &argmax_index);
        bool ok2 = backend_->topKF32(d_ptr, n, 1, device_id_, &topk_value, &topk_index);

        if (ok1 && ok2)
        {
            EXPECT_EQ(argmax_index, topk_index);
            EXPECT_FLOAT_EQ(argmax_value, topk_value);
        }

        freeDevice(d_ptr);
    }

    // =========================================================================
    // TOP-K INDEX CORRECTNESS — mirrors Token Ranking from Test__Sampler.cpp
    // =========================================================================

    TEST_P(GPUSamplingTest, TopK_IndicesAreGloballyCorrect)
    {
        // Verify that returned indices correctly map back to the original array
        void *d_ptr = uploadLogits(standard_logits_);
        ASSERT_NE(d_ptr, nullptr);

        int n = static_cast<int>(standard_logits_.size());
        std::vector<float> values(n);
        std::vector<int> indices(n);
        bool ok = backend_->topKF32(d_ptr, n, n, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        // Each returned (index, value) pair should match the original logits
        for (int i = 0; i < n; ++i)
        {
            EXPECT_FLOAT_EQ(values[i], standard_logits_[indices[i]])
                << "Rank " << i << ": value " << values[i]
                << " should match logits[" << indices[i] << "]="
                << standard_logits_[indices[i]];
        }

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, TopK_HigherLogitsHaveHigherRank)
    {
        // Verify ranking reflects logit magnitude
        std::vector<float> logits = {1.0f, 5.0f, 3.0f, 7.0f, 2.0f};
        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        const int k = 5;
        std::vector<float> values(k);
        std::vector<int> indices(k);
        bool ok = backend_->topKF32(d_ptr, static_cast<int>(logits.size()),
                                    k, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        // Expected ranking: 7.0(3), 5.0(1), 3.0(2), 2.0(4), 1.0(0)
        EXPECT_EQ(indices[0], 3) << "Highest logit should be rank 0";
        EXPECT_EQ(indices[1], 1) << "Second highest should be rank 1";
        EXPECT_EQ(indices[2], 2) << "Third highest should be rank 2";
        EXPECT_EQ(indices[3], 4) << "Fourth highest should be rank 3";
        EXPECT_EQ(indices[4], 0) << "Lowest logit should be rank 4";

        freeDevice(d_ptr);
    }

    // =========================================================================
    // REAL-WORLD SCENARIO TESTS
    // =========================================================================

    TEST_P(GPUSamplingTest, RealWorld_Qwen2TP2_VocabLocalShard)
    {
        // In TP=2 mode, each device sees vocab_local = 76032 (half of 152064)
        // Validate correct behavior on the local shard size
        const int n = 76032;
        std::vector<float> logits(n, 0.0f);
        logits[50000] = 14.5f;
        logits[25000] = 14.0f;
        logits[75000] = 13.5f;

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        // Argmax on local shard
        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, n, device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);
        EXPECT_EQ(out_index, 50000);
        EXPECT_FLOAT_EQ(out_value, 14.5f);

        // Top-k on local shard
        const int k = 3;
        std::vector<float> values(k);
        std::vector<int> indices(k);
        ok = backend_->topKF32(d_ptr, n, k, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        EXPECT_EQ(indices[0], 50000);
        EXPECT_EQ(indices[1], 25000);
        EXPECT_EQ(indices[2], 75000);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, RealWorld_DecodeLoopSimulation)
    {
        // Simulate repeated greedy sampling during decode loop
        // The same buffer is queried multiple times (mimicking decode iterations)
        const int n = 151936;
        std::vector<float> logits(n, 0.0f);
        logits[256] = 14.54f;

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        // Simulate 10 decode steps all querying the same logits
        for (int step = 0; step < 10; ++step)
        {
            float out_value = 0.0f;
            int out_index = -1;
            bool ok = argmaxF32(d_ptr, n, device_id_, &out_value, &out_index);
            ASSERT_TRUE(ok) << "Decode step " << step;
            EXPECT_EQ(out_index, 256) << "Decode step " << step << " should produce consistent token";
            EXPECT_FLOAT_EQ(out_value, 14.54f);
        }

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, RealWorld_RealisticLogitDistribution)
    {
        // Simulate a realistic logit distribution from an LLM
        // Most logits are small negative, a few tokens have positive values
        const int n = 151936;
        std::mt19937 rng(42);
        std::normal_distribution<float> noise(-5.0f, 2.0f);

        std::vector<float> logits(n);
        for (int i = 0; i < n; ++i)
        {
            logits[i] = noise(rng);
        }

        // Plant known top-5
        logits[256] = 15.0f;
        logits[8159] = 14.0f;
        logits[100160] = 13.5f;
        logits[72363] = 13.0f;
        logits[105797] = 12.8f;

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        // Argmax should find the planted peak
        float out_value = 0.0f;
        int out_index = -1;
        bool ok = argmaxF32(d_ptr, n, device_id_, &out_value, &out_index);
        ASSERT_TRUE(ok);
        EXPECT_EQ(out_index, 256) << "Argmax should find the planted peak in noisy data";
        EXPECT_FLOAT_EQ(out_value, 15.0f);

        // Top-5 should recover all planted peaks
        const int k = 5;
        std::vector<float> values(k);
        std::vector<int> indices(k);
        ok = backend_->topKF32(d_ptr, n, k, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        std::set<int> expected_top5 = {256, 8159, 100160, 72363, 105797};
        std::set<int> actual_top5(indices.begin(), indices.end());
        EXPECT_EQ(actual_top5, expected_top5) << "Top-5 should recover all planted peaks";

        freeDevice(d_ptr);
    }

    // =========================================================================
    // MULTIPLE ALLOCATIONS STRESS TEST
    // =========================================================================

    TEST_P(GPUSamplingTest, Stress_MultipleAllocFreeArgmax)
    {
        // Verify no leaks or corruption across multiple alloc/use/free cycles
        for (int iter = 0; iter < 20; ++iter)
        {
            const int n = 1000 + iter * 500;
            std::vector<float> logits(n, 0.0f);
            logits[iter % n] = 100.0f;

            void *d_ptr = uploadLogits(logits);
            ASSERT_NE(d_ptr, nullptr) << "Iteration " << iter;

            float out_value = 0.0f;
            int out_index = -1;
            bool ok = argmaxF32(d_ptr, n, device_id_, &out_value, &out_index);
            ASSERT_TRUE(ok) << "Iteration " << iter;

            EXPECT_EQ(out_index, iter % n) << "Iteration " << iter;
            EXPECT_FLOAT_EQ(out_value, 100.0f) << "Iteration " << iter;

            freeDevice(d_ptr);
        }
    }

    TEST_P(GPUSamplingTest, Stress_MultipleAllocFreeTopK)
    {
        for (int iter = 0; iter < 20; ++iter)
        {
            const int n = 2000 + iter * 1000;
            const int k = std::min(10, n);
            std::vector<float> logits(n, -1.0f);

            // Plant k peaks at known positions
            for (int j = 0; j < k; ++j)
            {
                logits[j * (n / k)] = 50.0f - static_cast<float>(j);
            }

            void *d_ptr = uploadLogits(logits);
            ASSERT_NE(d_ptr, nullptr) << "Iteration " << iter;

            std::vector<float> values(k);
            std::vector<int> indices(k);
            bool ok = backend_->topKF32(d_ptr, n, k, device_id_, values.data(), indices.data());
            ASSERT_TRUE(ok) << "Iteration " << iter;

            // Top-1 should always be the highest planted peak
            EXPECT_EQ(indices[0], 0) << "Iteration " << iter;
            EXPECT_FLOAT_EQ(values[0], 50.0f) << "Iteration " << iter;

            freeDevice(d_ptr);
        }
    }

    // =========================================================================
    //  LOGIT PENALTY TESTS — GPU-side penalty application + sampling
    // =========================================================================

    // ------------------------------------------------------------------
    // Helper: download GPU logits back to host for verification
    // ------------------------------------------------------------------
    static std::vector<float> downloadLogits(IBackend *backend, void *d_ptr,
                                             int count, int device_id)
    {
        std::vector<float> result(count);
        bool ok = backend->deviceToHost(result.data(), d_ptr,
                                        count * sizeof(float), device_id);
        EXPECT_TRUE(ok) << "D2H transfer failed";
        return result;
    }

    static float samplingUniform01(uint64_t seed, uint64_t offset)
    {
        return sampling_math::uniform01(seed, offset);
    }

    /**
     * @brief Comparator used by CPU-side expected top-k/top-p helpers.
     *
     * The GPU kernels sort candidates by logit descending and token id ascending.
     * Keeping the test oracle on the same rule makes ties explicit and prevents
     * future sampler changes from reintroducing backend-dependent ordering.
     */
    static bool topKCandidateBefore(const std::pair<float, int> &a,
                                    const std::pair<float, int> &b)
    {
        if (a.first > b.first)
            return true;
        if (a.first < b.first)
            return false;
        return a.second < b.second;
    }

    static int expectedTopKTopPSample(const std::vector<float> &logits,
                                      int top_k,
                                      float top_p,
                                      float temperature,
                                      uint64_t seed,
                                      uint64_t offset)
    {
        std::vector<std::pair<float, int>> candidates;
        candidates.reserve(logits.size());
        for (size_t i = 0; i < logits.size(); ++i)
            candidates.emplace_back(logits[i], static_cast<int>(i));

        top_k = std::min<int>(top_k, static_cast<int>(candidates.size()));
        std::partial_sort(candidates.begin(),
                          candidates.begin() + top_k,
                          candidates.end(),
                          topKCandidateBefore);
        candidates.resize(static_cast<size_t>(top_k));

        std::vector<float> sorted_logits(static_cast<size_t>(top_k));
        std::vector<int> sorted_ids(static_cast<size_t>(top_k));
        for (size_t i = 0; i < candidates.size(); ++i)
        {
            sorted_logits[i] = candidates[i].first;
            sorted_ids[i] = candidates[i].second;
        }
        std::vector<float> scratch(static_cast<size_t>(top_k), 0.0f);
        return sampling_math::sample_topk_topp_from_sorted_with_threshold(
            sorted_logits.data(),
            sorted_ids.data(),
            top_k,
            top_p,
            temperature,
            samplingUniform01(seed, offset),
            scratch.data());
    }

    struct ExpectedDistributionEntry
    {
        int token_id = -1;
        float probability = 0.0f;
    };

    static std::vector<ExpectedDistributionEntry> expectedTopKTopPDistribution(
        const std::vector<float> &logits,
        int top_k,
        float top_p,
        float temperature)
    {
        std::vector<std::pair<float, int>> candidates;
        candidates.reserve(logits.size());
        for (size_t i = 0; i < logits.size(); ++i)
            candidates.emplace_back(logits[i], static_cast<int>(i));

        top_k = std::min<int>(top_k, static_cast<int>(candidates.size()));
        std::partial_sort(candidates.begin(),
                          candidates.begin() + top_k,
                          candidates.end(),
                          topKCandidateBefore);
        candidates.resize(static_cast<size_t>(top_k));

        std::vector<float> sorted_logits(static_cast<size_t>(top_k));
        std::vector<int> sorted_ids(static_cast<size_t>(top_k));
        for (size_t i = 0; i < candidates.size(); ++i)
        {
            sorted_logits[i] = candidates[i].first;
            sorted_ids[i] = candidates[i].second;
        }

        std::vector<int> out_ids(static_cast<size_t>(top_k), -1);
        std::vector<float> out_probs(static_cast<size_t>(top_k), 0.0f);
        std::vector<float> scratch(static_cast<size_t>(top_k), 0.0f);
        sampling_math::build_topk_topp_distribution_from_sorted(
            sorted_logits.data(),
            sorted_ids.data(),
            top_k,
            top_p,
            temperature,
            out_ids.data(),
            out_probs.data(),
            scratch.data());

        std::vector<ExpectedDistributionEntry> distribution(static_cast<size_t>(top_k));
        for (int i = 0; i < top_k; ++i)
        {
            distribution[static_cast<size_t>(i)].token_id = out_ids[static_cast<size_t>(i)];
            distribution[static_cast<size_t>(i)].probability = out_probs[static_cast<size_t>(i)];
        }
        return distribution;
    }

    /**
     * @brief Build the full-logit row equivalent to compact top-k/top-p sampling.
     *
     * Active nucleus tokens keep raw logits divided by temperature. Every other
     * token is non-finite, which is the processed-logit contract consumed by the
     * vLLM-style verifier and sampler.
     */
    static std::vector<float> expectedTopKTopPProcessedLogits(
        const std::vector<float> &logits,
        int top_k,
        float top_p,
        float temperature)
    {
        std::vector<float> processed(
            logits.size(),
            -std::numeric_limits<float>::infinity());
        const auto distribution =
            expectedTopKTopPDistribution(logits, top_k, top_p, temperature);
        const float temp = temperature > 0.0f ? temperature : 1.0f;
        for (const auto &entry : distribution)
        {
            if (entry.token_id >= 0 && entry.probability > 0.0f)
                processed[static_cast<size_t>(entry.token_id)] =
                    logits[static_cast<size_t>(entry.token_id)] / temp;
        }
        return processed;
    }

    static std::vector<float> expectedTemperatureOnlyProbabilities(
        const std::vector<float> &logits,
        float temperature)
    {
        const float safe_temperature =
            (std::isfinite(temperature) && temperature > 0.0f) ? temperature : 1.0f;
        float max_logit = -std::numeric_limits<float>::infinity();
        for (float logit : logits)
        {
            if (std::isfinite(logit))
                max_logit = std::max(max_logit, logit / safe_temperature);
        }

        std::vector<float> probabilities(logits.size(), 0.0f);
        if (!std::isfinite(max_logit))
            return probabilities;

        double exp_sum = 0.0;
        for (float logit : logits)
        {
            if (std::isfinite(logit))
                exp_sum += std::exp(static_cast<double>(logit / safe_temperature - max_logit));
        }
        if (!(exp_sum > 0.0))
            return probabilities;

        for (size_t token = 0; token < logits.size(); ++token)
        {
            if (std::isfinite(logits[token]))
            {
                probabilities[token] = static_cast<float>(
                    std::exp(static_cast<double>(logits[token] / safe_temperature - max_logit)) /
                    exp_sum);
            }
        }
        return probabilities;
    }

    static int expectedTemperatureOnlySampleToken(
        const std::vector<float> &probabilities,
        float threshold)
    {
        const float target_mass =
            sampling_math::clamp_unit_threshold(threshold);
        float prefix = 0.0f;
        int selected = 0;
        float best_probability = -1.0f;
        for (size_t token = 0; token < probabilities.size(); ++token)
        {
            const float probability = probabilities[token];
            if (probability > best_probability)
            {
                best_probability = probability;
                selected = static_cast<int>(token);
            }
            if (!(probability > 0.0f))
                continue;
            prefix += probability;
            if (target_mass <= prefix)
                return static_cast<int>(token);
        }
        return selected;
    }

    static float distributionProbability(
        const std::vector<ExpectedDistributionEntry> &distribution,
        int token_id)
    {
        std::vector<int> token_ids(distribution.size(), -1);
        std::vector<float> probs(distribution.size(), 0.0f);
        for (size_t i = 0; i < distribution.size(); ++i)
        {
            token_ids[i] = distribution[i].token_id;
            probs[i] = distribution[i].probability;
        }
        return sampling_math::distribution_probability(
            token_ids.data(),
            probs.data(),
            static_cast<int>(distribution.size()),
            token_id);
    }

    struct ExpectedSpeculativeVerify
    {
        int token_id = -1;
        int accepted = 0;
        float accept_probability = 0.0f;
        float accept_threshold = 0.0f;
    };

    static ExpectedSpeculativeVerify expectedSpeculativeVerifyDistributionWithThresholds(
        const std::vector<ExpectedDistributionEntry> &target,
        const std::vector<ExpectedDistributionEntry> &draft,
        int draft_token,
        float accept_threshold,
        float residual_threshold);

    static ExpectedSpeculativeVerify expectedSpeculativeVerifyOneHotDraftWithThresholds(
        const std::vector<ExpectedDistributionEntry> &target,
        int draft_token,
        float accept_threshold,
        float residual_threshold);

    static ExpectedSpeculativeVerify expectedSpeculativeVerifyOneHotDraftVLLMWithThresholds(
        const std::vector<ExpectedDistributionEntry> &target,
        int draft_token,
        float accept_threshold,
        uint64_t inverse_sample_seed,
        int logical_position,
        int vocab_size);

    static ExpectedSpeculativeVerify expectedSpeculativeVerifyDistribution(
        const std::vector<ExpectedDistributionEntry> &target,
        const std::vector<ExpectedDistributionEntry> &draft,
        int draft_token,
        uint64_t accept_seed,
        uint64_t accept_offset,
        uint64_t residual_seed,
        uint64_t residual_offset)
    {
        return expectedSpeculativeVerifyDistributionWithThresholds(
            target,
            draft,
            draft_token,
            samplingUniform01(accept_seed, accept_offset),
            samplingUniform01(residual_seed, residual_offset));
    }

    static int expectedSampleDistributionWithThreshold(
        const std::vector<ExpectedDistributionEntry> &distribution,
        float threshold)
    {
        std::vector<int> token_ids(distribution.size(), -1);
        std::vector<float> probs(distribution.size(), 0.0f);
        for (size_t i = 0; i < distribution.size(); ++i)
        {
            token_ids[i] = distribution[i].token_id;
            probs[i] = distribution[i].probability;
        }
        return sampling_math::sample_distribution_with_threshold(
            token_ids.data(),
            probs.data(),
            static_cast<int>(distribution.size()),
            threshold);
    }

    static ExpectedSpeculativeVerify expectedSpeculativeVerifyDistributionWithThresholds(
        const std::vector<ExpectedDistributionEntry> &target,
        const std::vector<ExpectedDistributionEntry> &draft,
        int draft_token,
        float accept_threshold,
        float residual_threshold)
    {
        ExpectedSpeculativeVerify result;
        std::vector<int> target_ids(target.size(), -1);
        std::vector<float> target_probs(target.size(), 0.0f);
        std::vector<int> draft_ids(draft.size(), -1);
        std::vector<float> draft_probs(draft.size(), 0.0f);
        for (size_t i = 0; i < target.size(); ++i)
        {
            target_ids[i] = target[i].token_id;
            target_probs[i] = target[i].probability;
        }
        for (size_t i = 0; i < draft.size(); ++i)
        {
            draft_ids[i] = draft[i].token_id;
            draft_probs[i] = draft[i].probability;
        }

        sampling_math::speculative_verify_with_thresholds(
            target_ids.data(),
            target_probs.data(),
            draft_ids.data(),
            draft_probs.data(),
            static_cast<int>(target.size()),
            draft_token,
            accept_threshold,
            residual_threshold,
            &result.token_id,
            &result.accepted,
            &result.accept_probability,
            &result.accept_threshold);
        return result;
    }

    static ExpectedSpeculativeVerify expectedSpeculativeVerifyOneHotDraftWithThresholds(
        const std::vector<ExpectedDistributionEntry> &target,
        int draft_token,
        float accept_threshold,
        float residual_threshold)
    {
        ExpectedSpeculativeVerify result;
        std::vector<int> target_ids(target.size(), -1);
        std::vector<float> target_probs(target.size(), 0.0f);
        for (size_t i = 0; i < target.size(); ++i)
        {
            target_ids[i] = target[i].token_id;
            target_probs[i] = target[i].probability;
        }

        sampling_math::speculative_verify_with_thresholds_one_hot_draft(
            target_ids.data(),
            target_probs.data(),
            static_cast<int>(target.size()),
            draft_token,
            accept_threshold,
            residual_threshold,
            &result.token_id,
            &result.accepted,
            &result.accept_probability,
            &result.accept_threshold);
        return result;
    }

    static ExpectedSpeculativeVerify expectedSpeculativeVerifyOneHotDraftVLLMWithThresholds(
        const std::vector<ExpectedDistributionEntry> &target,
        int draft_token,
        float accept_threshold,
        uint64_t inverse_sample_seed,
        int logical_position,
        int vocab_size)
    {
        ExpectedSpeculativeVerify result;
        std::vector<int> target_ids(target.size(), -1);
        std::vector<float> target_probs(target.size(), 0.0f);
        for (size_t i = 0; i < target.size(); ++i)
        {
            target_ids[i] = target[i].token_id;
            target_probs[i] = target[i].probability;
        }

        sampling_math::speculative_verify_with_thresholds_one_hot_draft_vllm_recovered(
            target_ids.data(),
            target_probs.data(),
            static_cast<int>(target.size()),
            vocab_size,
            draft_token,
            accept_threshold,
            inverse_sample_seed,
            logical_position,
            &result.token_id,
            &result.accepted,
            &result.accept_probability,
            &result.accept_threshold);
        return result;
    }

    TEST_P(GPUSamplingTest, Penalty_SingleToken_Subtracted)
    {
        // Apply a penalty to token 2 and verify logit is reduced
        std::vector<float> logits = {1.0f, 2.0f, 5.0f, 0.5f, 1.5f};
        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        std::vector<int> token_ids = {2};
        std::vector<float> penalties = {3.0f};

        bool ok = backend_->applyLogitPenaltiesF32(
            d_ptr, token_ids.data(), penalties.data(),
            1, static_cast<int>(logits.size()), device_id_);
        ASSERT_TRUE(ok) << "applyLogitPenaltiesF32 not supported on " << GetParam();

        auto result = downloadLogits(backend_, d_ptr,
                                     static_cast<int>(logits.size()), device_id_);

        // Token 2: 5.0 - 3.0 = 2.0
        EXPECT_FLOAT_EQ(result[0], 1.0f) << "Unpenalized tokens should be unchanged";
        EXPECT_FLOAT_EQ(result[1], 2.0f) << "Unpenalized tokens should be unchanged";
        EXPECT_FLOAT_EQ(result[2], 2.0f) << "Token 2 should be penalized: 5.0 - 3.0 = 2.0";
        EXPECT_FLOAT_EQ(result[3], 0.5f) << "Unpenalized tokens should be unchanged";
        EXPECT_FLOAT_EQ(result[4], 1.5f) << "Unpenalized tokens should be unchanged";

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Penalty_MultipleTokens_AllSubtracted)
    {
        std::vector<float> logits = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        std::vector<int> token_ids = {0, 2, 4};
        std::vector<float> penalties = {1.0f, 5.0f, 10.0f};

        bool ok = backend_->applyLogitPenaltiesF32(
            d_ptr, token_ids.data(), penalties.data(),
            3, static_cast<int>(logits.size()), device_id_);
        ASSERT_TRUE(ok);

        auto result = downloadLogits(backend_, d_ptr,
                                     static_cast<int>(logits.size()), device_id_);

        EXPECT_FLOAT_EQ(result[0], 9.0f);  // 10 - 1
        EXPECT_FLOAT_EQ(result[1], 20.0f); // unchanged
        EXPECT_FLOAT_EQ(result[2], 25.0f); // 30 - 5
        EXPECT_FLOAT_EQ(result[3], 40.0f); // unchanged
        EXPECT_FLOAT_EQ(result[4], 40.0f); // 50 - 10

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Penalty_ZeroPenalties_NoOp)
    {
        // Empty penalty list — backends return false (no-op, nothing to do)
        // The caller (OrchestrationRunner) skips the call when the map is empty,
        // so this documents the backend contract rather than a usage pattern.
        std::vector<float> logits = {1.0f, 2.0f, 3.0f};
        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        bool ok = backend_->applyLogitPenaltiesF32(
            d_ptr, nullptr, nullptr, 0,
            static_cast<int>(logits.size()), device_id_);
        // Backends early-return false for num_penalties <= 0
        EXPECT_FALSE(ok) << "Zero penalties → backend returns false (no-op)";

        // Logits should still be unchanged
        auto result = downloadLogits(backend_, d_ptr,
                                     static_cast<int>(logits.size()), device_id_);
        EXPECT_FLOAT_EQ(result[0], 1.0f);
        EXPECT_FLOAT_EQ(result[1], 2.0f);
        EXPECT_FLOAT_EQ(result[2], 3.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Penalty_OutOfBoundsTokenId_Ignored)
    {
        // Token IDs outside [0, vocab_size) should be silently ignored
        std::vector<float> logits = {5.0f, 5.0f, 5.0f};
        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        std::vector<int> token_ids = {-1, 999, 1}; // -1 and 999 are OOB for vocab=3
        std::vector<float> penalties = {100.0f, 100.0f, 2.0f};

        bool ok = backend_->applyLogitPenaltiesF32(
            d_ptr, token_ids.data(), penalties.data(),
            3, static_cast<int>(logits.size()), device_id_);
        ASSERT_TRUE(ok);

        auto result = downloadLogits(backend_, d_ptr,
                                     static_cast<int>(logits.size()), device_id_);

        EXPECT_FLOAT_EQ(result[0], 5.0f) << "OOB tokens should not corrupt logits";
        EXPECT_FLOAT_EQ(result[1], 3.0f) << "Valid token should be penalized: 5.0 - 2.0";
        EXPECT_FLOAT_EQ(result[2], 5.0f) << "OOB tokens should not corrupt logits";

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Penalty_NegativePenalty_BoostsLogit)
    {
        // Negative penalty = boost (used for negative presence penalty)
        std::vector<float> logits = {0.0f, 0.0f, 0.0f};
        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        std::vector<int> token_ids = {1};
        std::vector<float> penalties = {-5.0f}; // Boost by 5

        bool ok = backend_->applyLogitPenaltiesF32(
            d_ptr, token_ids.data(), penalties.data(),
            1, static_cast<int>(logits.size()), device_id_);
        ASSERT_TRUE(ok);

        auto result = downloadLogits(backend_, d_ptr,
                                     static_cast<int>(logits.size()), device_id_);

        EXPECT_FLOAT_EQ(result[0], 0.0f);
        EXPECT_FLOAT_EQ(result[1], 5.0f) << "Negative penalty should boost: 0.0 - (-5.0) = 5.0";
        EXPECT_FLOAT_EQ(result[2], 0.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Penalty_ThenArgmax_ChangesSelection)
    {
        // End-to-end: penalty shifts argmax from token 0 to token 1
        std::vector<float> logits = {10.0f, 9.5f, 1.0f, 1.0f, 1.0f};
        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        // Verify initial argmax is token 0
        float val = 0;
        int idx = -1;
        argmaxF32(d_ptr, static_cast<int>(logits.size()),
                            device_id_, &val, &idx);
        EXPECT_EQ(idx, 0) << "Before penalty, argmax should be token 0";

        // Penalize token 0 enough to make token 1 win
        std::vector<int> token_ids = {0};
        std::vector<float> penalties = {2.0f}; // 10.0 - 2.0 = 8.0 < 9.5

        bool ok = backend_->applyLogitPenaltiesF32(
            d_ptr, token_ids.data(), penalties.data(),
            1, static_cast<int>(logits.size()), device_id_);
        ASSERT_TRUE(ok);

        // After penalty, argmax should shift to token 1
        argmaxF32(d_ptr, static_cast<int>(logits.size()),
                            device_id_, &val, &idx);
        EXPECT_EQ(idx, 1) << "After penalty, argmax should shift to token 1 (9.5 > 8.0)";
        EXPECT_FLOAT_EQ(val, 9.5f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Penalty_LargeVocab_SparseApplication)
    {
        // Real-world: Qwen2 vocab (151936) with sparse penalties
        const int vocab_size = 151936;
        std::vector<float> logits(vocab_size, 0.0f);
        logits[0] = 10.0f;
        logits[42] = 9.0f;
        logits[1000] = 8.0f;

        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        // Penalize only 3 tokens out of 151K
        std::vector<int> token_ids = {0, 42, 1000};
        std::vector<float> penalties = {5.0f, 1.0f, 0.5f};

        bool ok = backend_->applyLogitPenaltiesF32(
            d_ptr, token_ids.data(), penalties.data(),
            3, vocab_size, device_id_);
        ASSERT_TRUE(ok);

        // Verify via argmax — token 42 should now win (9.0 - 1.0 = 8.0 > 10.0 - 5.0 = 5.0)
        // token 1000: 8.0 - 0.5 = 7.5
        float val = 0;
        int idx = -1;
        argmaxF32(d_ptr, vocab_size, device_id_, &val, &idx);
        EXPECT_EQ(idx, 42) << "After penalties, token 42 (8.0) should beat token 0 (5.0)";

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Penalty_ManyPenalties_AllApplied)
    {
        // Stress test: apply 256 penalties
        const int vocab_size = 1000;
        std::vector<float> logits(vocab_size, 1.0f);
        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        const int num_penalties = 256;
        std::vector<int> token_ids(num_penalties);
        std::vector<float> penalties(num_penalties);
        for (int i = 0; i < num_penalties; ++i)
        {
            token_ids[i] = i;
            penalties[i] = 0.5f;
        }

        bool ok = backend_->applyLogitPenaltiesF32(
            d_ptr, token_ids.data(), penalties.data(),
            num_penalties, vocab_size, device_id_);
        ASSERT_TRUE(ok);

        auto result = downloadLogits(backend_, d_ptr, vocab_size, device_id_);

        // First 256 tokens should be 0.5, rest should be 1.0
        for (int i = 0; i < num_penalties; ++i)
        {
            EXPECT_FLOAT_EQ(result[i], 0.5f)
                << "Token " << i << " should be penalized: 1.0 - 0.5 = 0.5";
        }
        for (int i = num_penalties; i < vocab_size; ++i)
        {
            EXPECT_FLOAT_EQ(result[i], 1.0f)
                << "Token " << i << " should be unchanged";
        }

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Penalty_DRYStyle_ExponentialPenalty)
    {
        // Simulate DRY-style exponential penalty: multiple tokens with varying
        // penalty magnitudes that reflect repeat_len differences
        const int vocab_size = 10;
        std::vector<float> logits = {10.0f, 10.0f, 10.0f, 10.0f, 10.0f,
                                     10.0f, 10.0f, 10.0f, 10.0f, 10.0f};
        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        // DRY penalties: multiplier=1.0, base=1.75
        // repeat_len=3, allowed=1 → 1.75^2 = 3.0625
        // repeat_len=5, allowed=1 → 1.75^4 = 9.3789
        float dry_3 = std::pow(1.75f, 2.0f);
        float dry_5 = std::pow(1.75f, 4.0f);

        std::vector<int> token_ids = {2, 7};
        std::vector<float> penalties = {dry_3, dry_5};

        bool ok = backend_->applyLogitPenaltiesF32(
            d_ptr, token_ids.data(), penalties.data(),
            2, vocab_size, device_id_);
        ASSERT_TRUE(ok);

        auto result = downloadLogits(backend_, d_ptr, vocab_size, device_id_);

        EXPECT_NEAR(result[2], 10.0f - dry_3, 0.001f)
            << "Token 2 should have DRY penalty for repeat_len=3";
        EXPECT_NEAR(result[7], 10.0f - dry_5, 0.001f)
            << "Token 7 should have larger DRY penalty for repeat_len=5";

        // Unpenalized tokens should be unchanged
        EXPECT_FLOAT_EQ(result[0], 10.0f);
        EXPECT_FLOAT_EQ(result[5], 10.0f);

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, Penalty_CombinedWithTopK_ChangesRanking)
    {
        // End-to-end: penalty → topK should reflect penalized logits
        std::vector<float> logits = {10.0f, 9.0f, 8.0f, 7.0f, 6.0f};
        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        // Penalize top-2 tokens heavily
        std::vector<int> token_ids = {0, 1};
        std::vector<float> penalties = {8.0f, 7.0f}; // 10→2, 9→2

        bool ok = backend_->applyLogitPenaltiesF32(
            d_ptr, token_ids.data(), penalties.data(),
            2, static_cast<int>(logits.size()), device_id_);
        ASSERT_TRUE(ok);

        // topK=3: should now be [2(8.0), 3(7.0), 4(6.0)] instead of [0,1,2]
        std::vector<float> values(3);
        std::vector<int> indices(3);
        ok = backend_->topKF32(d_ptr, static_cast<int>(logits.size()),
                               3, device_id_, values.data(), indices.data());
        ASSERT_TRUE(ok);

        EXPECT_EQ(indices[0], 2) << "After penalty, token 2 (8.0) should be top-1";
        EXPECT_FLOAT_EQ(values[0], 8.0f);
        EXPECT_EQ(indices[1], 3) << "Token 3 (7.0) should be top-2";
        EXPECT_EQ(indices[2], 4) << "Token 4 (6.0) should be top-3";

        freeDevice(d_ptr);
    }

    // =========================================================================
    //  DRY PENALTY CPU↔GPU PARITY TESTS
    //
    //  These tests compute DRY penalties via the CPU Sampler, then apply them
    //  to identical logit arrays on both CPU and GPU, verifying bit-exact
    //  parity. This validates the full DRY pipeline: CPU computes the sparse
    //  penalty map → GPU applies it in-place → results match CPU application.
    // =========================================================================

    // Helper: apply CPU penalties in-place (same as Sampler::apply_penalties)
    static void applyCpuPenalties(std::vector<float> &logits,
                                  const std::vector<LogitPenalty> &penalties)
    {
        for (const auto &entry : penalties)
            logits[entry.token_id] -= entry.penalty;
    }

    // Helper: apply GPU penalties, download result
    static std::vector<float> applyGpuPenalties(
        IBackend *backend, int device_id,
        const std::vector<float> &logits,
        const std::vector<LogitPenalty> &penalties)
    {
        size_t bytes = logits.size() * sizeof(float);
        void *d_ptr = backend->allocate(bytes, device_id);
        EXPECT_NE(d_ptr, nullptr);
        bool ok = backend->hostToDevice(d_ptr, logits.data(), bytes, device_id);
        EXPECT_TRUE(ok);

        if (!penalties.empty())
        {
            // Convert AoS → SoA
            std::vector<int> token_ids(penalties.size());
            std::vector<float> penalty_vals(penalties.size());
            for (size_t i = 0; i < penalties.size(); ++i)
            {
                token_ids[i] = penalties[i].token_id;
                penalty_vals[i] = penalties[i].penalty;
            }

            ok = backend->applyLogitPenaltiesF32(
                d_ptr, token_ids.data(), penalty_vals.data(),
                static_cast<int>(penalties.size()),
                static_cast<int>(logits.size()), device_id);
            EXPECT_TRUE(ok);
        }

        auto result = downloadLogits(backend, d_ptr,
                                     static_cast<int>(logits.size()), device_id);
        backend->free(d_ptr, device_id);
        return result;
    }

    TEST_P(GPUSamplingTest, LogitPenaltyDeviceInputsAreGraphCapturable)
    {
        const std::vector<float> logits = {1.0f, 8.0f, 4.0f, 3.0f, 6.0f};
        const std::vector<int> token_ids = {1, 4};
        const std::vector<float> penalty_vals = {7.0f, 2.5f};
        std::vector<float> expected = logits;
        expected[1] -= penalty_vals[0];
        expected[4] -= penalty_vals[1];

        void *d_logits = nullptr;
        void *d_token_ids = nullptr;
        void *d_penalties = nullptr;

        auto cleanup = [&]()
        {
            if (d_logits)
                backend_->free(d_logits, device_id_);
            if (d_token_ids)
                backend_->free(d_token_ids, device_id_);
            if (d_penalties)
                backend_->free(d_penalties, device_id_);
        };

        d_logits = backend_->allocate(logits.size() * sizeof(float), device_id_);
        d_token_ids = backend_->allocate(token_ids.size() * sizeof(int), device_id_);
        d_penalties = backend_->allocate(penalty_vals.size() * sizeof(float), device_id_);
        ASSERT_NE(d_logits, nullptr);
        ASSERT_NE(d_token_ids, nullptr);
        ASSERT_NE(d_penalties, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);

                ASSERT_TRUE(backend_->hostToDevice(
                    d_logits, logits.data(), logits.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_token_ids, token_ids.data(), token_ids.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_penalties, penalty_vals.data(), penalty_vals.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueLogitPenaltiesF32Device(
                    d_logits,
                    d_token_ids,
                    d_penalties,
                    static_cast<int>(token_ids.size()),
                    static_cast<int>(logits.size()),
                    device_id_,
                    stream));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        auto result = downloadLogits(
            backend_, d_logits, static_cast<int>(logits.size()), device_id_);
        cleanup();

        ASSERT_EQ(result.size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i)
        {
            EXPECT_FLOAT_EQ(result[i], expected[i])
                << "Graph-captured penalty mismatch at token " << i;
        }
    }

    TEST_P(GPUSamplingTest, TopKTopPSampleDeviceOutputIsGraphCapturable)
    {
        const std::vector<float> logits = {0.1f, 4.5f, 3.8f, 0.0f,
                                           2.2f, 5.0f, -1.0f, 3.2f};
        constexpr int top_k = 4;
        constexpr float top_p = 0.85f;
        constexpr float temperature = 0.6f;
        constexpr uint64_t seed = 1234;
        constexpr uint64_t offset = 7;
        const int expected = expectedTopKTopPSample(
            logits, top_k, top_p, temperature, seed, offset);

        void *d_logits = nullptr;
        void *d_token = nullptr;

        auto cleanup = [&]()
        {
            if (d_logits)
                backend_->free(d_logits, device_id_);
            if (d_token)
                backend_->free(d_token, device_id_);
        };

        d_logits = backend_->allocate(logits.size() * sizeof(float), device_id_);
        d_token = backend_->allocate(sizeof(int), device_id_);
        ASSERT_NE(d_logits, nullptr);
        ASSERT_NE(d_token, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);

                ASSERT_TRUE(backend_->hostToDevice(
                    d_logits, logits.data(), logits.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
                EXPECT_FALSE(backend_->enqueueSampleTopKTopPF32Device(
                    d_logits,
                    static_cast<int>(logits.size()),
                    top_k,
                    top_p,
                    temperature,
                    seed,
                    offset,
                    device_id_,
                    nullptr,
                    d_token))
                    << "graph-capturable sampler must reject the legacy default/null stream";

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueSampleTopKTopPF32Device(
                    d_logits,
                    static_cast<int>(logits.size()),
                    top_k,
                    top_p,
                    temperature,
                    seed,
                    offset,
                    device_id_,
                    stream,
                    d_token));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        int actual = -1;
        ASSERT_TRUE(backend_->deviceToHost(&actual, d_token, sizeof(int), device_id_));
        cleanup();

        EXPECT_EQ(actual, expected)
            << "Graph-captured top-k/top-p sampler selected the wrong token";
    }

    TEST_P(GPUSamplingTest, TopKTopPDistributionMatchesCPUSampler)
    {
        const std::vector<float> logits = {0.1f, 4.5f, 3.8f, 0.0f, 2.2f,
                                           5.0f, -1.0f, 3.2f, 4.1f, 1.3f};
        constexpr int top_k = 6;
        constexpr float top_p = 0.78f;
        constexpr float temperature = 0.7f;

        Sampler cpu_sampler(123);
        SamplingParams params;
        params.temperature = temperature;
        params.top_k = top_k;
        params.top_p = top_p;
        const auto cpu_distribution =
            cpu_sampler.compute_distribution(logits.data(), logits.size(), params);

        void *d_logits = nullptr;
        void *d_token_ids = nullptr;
        void *d_probs = nullptr;

        auto cleanup = [&]()
        {
            if (d_logits)
                backend_->free(d_logits, device_id_);
            if (d_token_ids)
                backend_->free(d_token_ids, device_id_);
            if (d_probs)
                backend_->free(d_probs, device_id_);
        };

        d_logits = backend_->allocate(logits.size() * sizeof(float), device_id_);
        d_token_ids = backend_->allocate(top_k * sizeof(int), device_id_);
        d_probs = backend_->allocate(top_k * sizeof(float), device_id_);
        ASSERT_NE(d_logits, nullptr);
        ASSERT_NE(d_token_ids, nullptr);
        ASSERT_NE(d_probs, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);

                ASSERT_TRUE(backend_->hostToDevice(
                    d_logits, logits.data(), logits.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueBuildTopKTopPDistributionF32Device(
                    d_logits,
                    static_cast<int>(logits.size()),
                    top_k,
                    top_p,
                    temperature,
                    device_id_,
                    stream,
                    d_token_ids,
                    d_probs));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::vector<int> gpu_ids(top_k, -1);
        std::vector<float> gpu_probs(top_k, 0.0f);
        ASSERT_TRUE(backend_->deviceToHost(gpu_ids.data(), d_token_ids, top_k * sizeof(int), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(gpu_probs.data(), d_probs, top_k * sizeof(float), device_id_));
        cleanup();

        for (int i = 0; i < top_k; ++i)
        {
            if (i < static_cast<int>(cpu_distribution.size()))
            {
                EXPECT_EQ(gpu_ids[static_cast<size_t>(i)],
                          cpu_distribution[static_cast<size_t>(i)].token_id)
                    << "CPU/GPU compact distribution token mismatch at slot " << i;
                EXPECT_NEAR(gpu_probs[static_cast<size_t>(i)],
                            cpu_distribution[static_cast<size_t>(i)].probability,
                            1e-5f)
                    << "CPU/GPU compact distribution probability mismatch at slot " << i;
            }
            else
            {
                EXPECT_EQ(gpu_ids[static_cast<size_t>(i)], -1)
                    << "GPU should mark inactive top-p slots with token -1";
                EXPECT_FLOAT_EQ(gpu_probs[static_cast<size_t>(i)], 0.0f)
                    << "GPU should zero inactive top-p probability slots";
            }
        }
    }

    TEST_P(GPUSamplingTest, BatchedTopKTopPDistributionMatchesSerialCPUOracle)
    {
        constexpr int rows = 3;
        constexpr int vocab_size = 4096;
        constexpr int row_stride = vocab_size + 7;
        constexpr int top_k = 40;
        constexpr int out_stride = 64;
        constexpr float top_p = 0.93f;
        constexpr float temperature = 0.72f;

        std::vector<float> logits(static_cast<size_t>(rows * row_stride), -17.0f);
        for (int row = 0; row < rows; ++row)
        {
            for (int i = 0; i < vocab_size; ++i)
            {
                logits[static_cast<size_t>(row * row_stride + i)] =
                    -12.0f - 0.00011f * static_cast<float>((i * 53 + row * 97) % 997);
            }
            for (int rank = 0; rank < top_k; ++rank)
            {
                const int token = (row * 911 + rank * 73 + 17) % vocab_size;
                logits[static_cast<size_t>(row * row_stride + token)] =
                    6.0f - 0.047f * static_cast<float>(rank) +
                    0.013f * static_cast<float>(row);
            }
        }

        std::vector<std::vector<ExpectedDistributionEntry>> expected;
        expected.reserve(rows);
        for (int row = 0; row < rows; ++row)
        {
            std::vector<float> row_logits(
                logits.begin() + static_cast<ptrdiff_t>(row * row_stride),
                logits.begin() + static_cast<ptrdiff_t>(row * row_stride + vocab_size));
            expected.push_back(
                expectedTopKTopPDistribution(row_logits, top_k, top_p, temperature));
        }

        void *d_logits = nullptr;
        void *d_token_ids = nullptr;
        void *d_probs = nullptr;
        void *d_scratch_values = nullptr;
        void *d_scratch_indices = nullptr;
        constexpr int scratch_capacity = rows * 128 * top_k;

        auto cleanup = [&]()
        {
            if (d_logits)
                backend_->free(d_logits, device_id_);
            if (d_token_ids)
                backend_->free(d_token_ids, device_id_);
            if (d_probs)
                backend_->free(d_probs, device_id_);
            if (d_scratch_values)
                backend_->free(d_scratch_values, device_id_);
            if (d_scratch_indices)
                backend_->free(d_scratch_indices, device_id_);
        };

        d_logits = backend_->allocate(logits.size() * sizeof(float), device_id_);
        d_token_ids = backend_->allocate(rows * out_stride * sizeof(int), device_id_);
        d_probs = backend_->allocate(rows * out_stride * sizeof(float), device_id_);
        d_scratch_values = backend_->allocate(scratch_capacity * sizeof(float), device_id_);
        d_scratch_indices = backend_->allocate(scratch_capacity * sizeof(int), device_id_);
        ASSERT_NE(d_logits, nullptr);
        ASSERT_NE(d_token_ids, nullptr);
        ASSERT_NE(d_probs, nullptr);
        ASSERT_NE(d_scratch_values, nullptr);
        ASSERT_NE(d_scratch_indices, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);

                ASSERT_TRUE(backend_->hostToDevice(
                    d_logits, logits.data(), logits.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueBuildTopKTopPDistributionsF32Device(
                    d_logits,
                    rows,
                    vocab_size,
                    row_stride,
                    top_k,
                    top_p,
                    temperature,
                    device_id_,
                    stream,
                    d_token_ids,
                    out_stride,
                    d_probs,
                    d_scratch_values,
                    d_scratch_indices,
                    scratch_capacity));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::vector<int> gpu_ids(static_cast<size_t>(rows * out_stride), -1);
        std::vector<float> gpu_probs(static_cast<size_t>(rows * out_stride), 0.0f);
        ASSERT_TRUE(backend_->deviceToHost(
            gpu_ids.data(), d_token_ids, gpu_ids.size() * sizeof(int), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(
            gpu_probs.data(), d_probs, gpu_probs.size() * sizeof(float), device_id_));
        cleanup();

        for (int row = 0; row < rows; ++row)
        {
            for (int i = 0; i < top_k; ++i)
            {
                const size_t idx = static_cast<size_t>(row * out_stride + i);
                EXPECT_EQ(gpu_ids[idx], expected[static_cast<size_t>(row)][static_cast<size_t>(i)].token_id)
                    << "Batched compact distribution token mismatch at row " << row
                    << ", slot " << i;
                EXPECT_NEAR(gpu_probs[idx],
                            expected[static_cast<size_t>(row)][static_cast<size_t>(i)].probability,
                            1e-5f)
                    << "Batched compact distribution probability mismatch at row " << row
                    << ", slot " << i;
            }
        }
    }

    TEST_P(GPUSamplingTest, TopKTopP_Qwen36VocabTopK40_GraphCapturedDistributionAndSampleMatchCPU)
    {
        constexpr int vocab_size = 248320;
        constexpr int top_k = 40;
        constexpr float top_p = 0.95f;
        constexpr float temperature = 0.6f;
        constexpr uint64_t seed = 424242;
        constexpr uint64_t offset = 17;
        const float threshold = samplingUniform01(seed, offset);

        std::vector<float> logits(static_cast<size_t>(vocab_size));
        for (int i = 0; i < vocab_size; ++i)
            logits[static_cast<size_t>(i)] = -18.0f - 0.00037f * static_cast<float>((i * 37) % 997);

        const int hot_tokens[top_k] = {
            151936, 240001, 17, 248319, 98013,
            2048, 77777, 123456, 190000, 4096,
            222222, 31415, 65536, 101010, 88000,
            54321, 199999, 1, 135791, 246810,
            271, 13962, 96304, 3710, 5839,
            5077, 1414, 248068, 248069, 27775,
            2144, 3766, 16545, 2972, 51121,
            22527, 6157, 5757, 159034, 1503};
        for (int rank = 0; rank < top_k; ++rank)
        {
            logits[static_cast<size_t>(hot_tokens[rank])] =
                7.25f - 0.083f * static_cast<float>(rank);
        }

        const auto expected_distribution =
            expectedTopKTopPDistribution(logits, top_k, top_p, temperature);
        const int expected_sample =
            expectedSampleDistributionWithThreshold(expected_distribution, threshold);
        float expected_sample_probability = 0.0f;
        for (const auto &entry : expected_distribution)
        {
            if (entry.token_id == expected_sample)
            {
                expected_sample_probability = entry.probability;
                break;
            }
        }
        const int expected_direct_sample =
            expectedTopKTopPSample(logits, top_k, top_p, temperature, seed, offset);
        ASSERT_EQ(expected_direct_sample, expected_sample)
            << "direct CPU sample and compact-distribution CPU sample should agree";

        void *d_logits = nullptr;
        void *d_token_ids = nullptr;
        void *d_probs = nullptr;
        void *d_sample_token = nullptr;
        void *d_sample_probability = nullptr;
        void *d_direct_token = nullptr;

        auto cleanup = [&]()
        {
            if (d_logits)
                backend_->free(d_logits, device_id_);
            if (d_token_ids)
                backend_->free(d_token_ids, device_id_);
            if (d_probs)
                backend_->free(d_probs, device_id_);
            if (d_sample_token)
                backend_->free(d_sample_token, device_id_);
            if (d_sample_probability)
                backend_->free(d_sample_probability, device_id_);
            if (d_direct_token)
                backend_->free(d_direct_token, device_id_);
        };

        d_logits = backend_->allocate(logits.size() * sizeof(float), device_id_);
        d_token_ids = backend_->allocate(top_k * sizeof(int), device_id_);
        d_probs = backend_->allocate(top_k * sizeof(float), device_id_);
        d_sample_token = backend_->allocate(sizeof(int), device_id_);
        d_sample_probability = backend_->allocate(sizeof(float), device_id_);
        d_direct_token = backend_->allocate(sizeof(int), device_id_);
        ASSERT_NE(d_logits, nullptr);
        ASSERT_NE(d_token_ids, nullptr);
        ASSERT_NE(d_probs, nullptr);
        ASSERT_NE(d_sample_token, nullptr);
        ASSERT_NE(d_sample_probability, nullptr);
        ASSERT_NE(d_direct_token, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);

                ASSERT_TRUE(backend_->hostToDevice(
                    d_logits, logits.data(), logits.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueBuildTopKTopPDistributionF32Device(
                    d_logits,
                    static_cast<int>(logits.size()),
                    top_k,
                    top_p,
                    temperature,
                    device_id_,
                    stream,
                    d_token_ids,
                    d_probs));
                ASSERT_TRUE(backend_->enqueueSampleDistributionF32Device(
                    d_token_ids,
                    d_probs,
                    top_k,
                    threshold,
                    device_id_,
                    stream,
                    d_sample_token,
                    d_sample_probability));
                ASSERT_TRUE(backend_->enqueueSampleTopKTopPF32Device(
                    d_logits,
                    static_cast<int>(logits.size()),
                    top_k,
                    top_p,
                    temperature,
                    seed,
                    offset,
                    device_id_,
                    stream,
                    d_direct_token));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::vector<int> gpu_ids(top_k, -1);
        std::vector<float> gpu_probs(top_k, 0.0f);
        int sample_token = -1;
        float sample_probability = 0.0f;
        int direct_token = -1;
        ASSERT_TRUE(backend_->deviceToHost(gpu_ids.data(), d_token_ids, top_k * sizeof(int), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(gpu_probs.data(), d_probs, top_k * sizeof(float), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(&sample_token, d_sample_token, sizeof(int), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(&sample_probability, d_sample_probability, sizeof(float), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(&direct_token, d_direct_token, sizeof(int), device_id_));
        cleanup();

        for (int i = 0; i < top_k; ++i)
        {
            EXPECT_EQ(gpu_ids[static_cast<size_t>(i)],
                      expected_distribution[static_cast<size_t>(i)].token_id)
                << "Qwen3.6 CPU/GPU compact distribution token mismatch at slot " << i;
            EXPECT_NEAR(gpu_probs[static_cast<size_t>(i)],
                        expected_distribution[static_cast<size_t>(i)].probability,
                        1e-5f)
                << "Qwen3.6 CPU/GPU compact distribution probability mismatch at slot " << i;
        }
        EXPECT_EQ(sample_token, expected_sample)
            << "Qwen3.6 compact distribution sample mismatch";
        EXPECT_NEAR(sample_probability, expected_sample_probability, 1e-6f)
            << "Qwen3.6 compact distribution sampled probability mismatch";
        EXPECT_EQ(direct_token, expected_direct_sample)
            << "Qwen3.6 direct top-k/top-p sample mismatch";
    }

    TEST_P(GPUSamplingTest, TopKTopP_Qwen36TopK20RepeatedGraphReplayIsStable)
    {
        /*
         * ROCm stochastic MTP repeatability is very sensitive to the first
         * token sampled from prefill logits. This test mirrors that production
         * lane: Qwen3.6 vocab size, Qwen chat-like sampling params, a compact
         * distribution build, then repeated graph replays on one explicit
         * stream. Any drift here means the sampler kernel itself is not a safe
         * building block for graph-captured MTP.
         */
        constexpr int vocab_size = 248320;
        constexpr int top_k = 20;
        constexpr float top_p = 0.95f;
        constexpr float temperature = 0.6f;
        constexpr uint64_t seed = 123;
        constexpr uint64_t offset = 4;
        constexpr int replay_count = 24;
        const float threshold = samplingUniform01(seed, offset);

        struct HotToken
        {
            int token_id;
            float logit;
        };

        const std::vector<HotToken> hot_tokens = {
            {33075, 9.0000f}, {25174, 8.9950f}, {888, 8.25f},
            {279, 8.05f},    {15217, 7.91f},   {5388, 7.80f},
            {13, 7.65f},     {198, 7.62f},     {271, 7.60f},
            {471, 7.30f},    {262, 7.18f},     {256, 7.16f},
            {2972, 7.02f},   {2425, 6.91f},    {2824, 6.80f},
            {64700, 6.69f},  {357, 6.58f},     {15352, 6.47f},
            {11, 6.36f},     {1575, 6.25f},
        };

        std::vector<float> logits(static_cast<size_t>(vocab_size), -18.0f);
        for (int i = 0; i < vocab_size; ++i)
        {
            logits[static_cast<size_t>(i)] -=
                0.00023f * static_cast<float>((i * 47) % 997);
        }
        for (const HotToken &hot : hot_tokens)
            logits[static_cast<size_t>(hot.token_id)] = hot.logit;

        const auto expected_distribution =
            expectedTopKTopPDistribution(logits, top_k, top_p, temperature);
        const int expected_sample =
            expectedSampleDistributionWithThreshold(expected_distribution, threshold);
        ASSERT_GE(expected_sample, 0);

        void *d_logits = nullptr;
        void *d_token_ids = nullptr;
        void *d_probs = nullptr;
        void *d_sample_token = nullptr;

        auto cleanup = [&]()
        {
            if (d_logits)
                backend_->free(d_logits, device_id_);
            if (d_token_ids)
                backend_->free(d_token_ids, device_id_);
            if (d_probs)
                backend_->free(d_probs, device_id_);
            if (d_sample_token)
                backend_->free(d_sample_token, device_id_);
        };

        d_logits = backend_->allocate(logits.size() * sizeof(float), device_id_);
        d_token_ids = backend_->allocate(top_k * sizeof(int), device_id_);
        d_probs = backend_->allocate(top_k * sizeof(float), device_id_);
        d_sample_token = backend_->allocate(sizeof(int), device_id_);
        ASSERT_NE(d_logits, nullptr);
        ASSERT_NE(d_token_ids, nullptr);
        ASSERT_NE(d_probs, nullptr);
        ASSERT_NE(d_sample_token, nullptr);

        auto run_replays = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);

                ASSERT_TRUE(backend_->hostToDevice(
                    d_logits,
                    logits.data(),
                    logits.size() * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueBuildTopKTopPDistributionF32Device(
                    d_logits,
                    vocab_size,
                    top_k,
                    top_p,
                    temperature,
                    device_id_,
                    stream,
                    d_token_ids,
                    d_probs));
                ASSERT_TRUE(backend_->enqueueSampleDistributionF32Device(
                    d_token_ids,
                    d_probs,
                    top_k,
                    threshold,
                    device_id_,
                    stream,
                    d_sample_token));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());

                for (int replay = 0; replay < replay_count; ++replay)
                {
                    ASSERT_TRUE(capture->launch()) << "replay=" << replay;
                    ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_))
                        << "replay=" << replay;

                    int sample_token = -1;
                    ASSERT_TRUE(backend_->deviceToHost(
                        &sample_token, d_sample_token, sizeof(int), device_id_))
                        << "replay=" << replay;
                    EXPECT_EQ(sample_token, expected_sample)
                        << "Qwen3.6 top-k/top-p graph replay changed sampled token at replay "
                        << replay;
                }
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_replays(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_replays(ctx);
        }

        std::vector<int> gpu_ids(top_k, -1);
        std::vector<float> gpu_probs(top_k, 0.0f);
        ASSERT_TRUE(backend_->deviceToHost(
            gpu_ids.data(), d_token_ids, top_k * sizeof(int), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(
            gpu_probs.data(), d_probs, top_k * sizeof(float), device_id_));
        cleanup();

        for (int i = 0; i < top_k; ++i)
        {
            EXPECT_EQ(gpu_ids[static_cast<size_t>(i)],
                      expected_distribution[static_cast<size_t>(i)].token_id)
                << "repeated graph replay compact distribution token mismatch at slot " << i;
            EXPECT_NEAR(gpu_probs[static_cast<size_t>(i)],
                        expected_distribution[static_cast<size_t>(i)].probability,
                        1e-5f)
                << "repeated graph replay compact distribution probability mismatch at slot " << i;
        }
    }

    TEST_P(GPUSamplingTest, TopKTopP_Qwen36RealLogitStyleRowsSeededSamplesMatchCPU)
    {
        constexpr int vocab_size = 248320;
        constexpr int top_k = 40;
        constexpr float top_p = 0.95f;
        constexpr float temperature = 0.6f;
        constexpr uint64_t seed = 123;

        struct HotToken
        {
            int token_id;
            float logit;
        };

        const std::vector<std::vector<HotToken>> rows = {
            {
                {262, 8.9500f}, {256, 8.9475f}, {198, 8.72f}, {471, 8.31f},
                {2972, 8.05f}, {2425, 7.92f}, {2824, 7.81f}, {64700, 7.62f},
                {357, 7.51f}, {15352, 7.42f}, {11, 7.25f}, {1575, 7.10f},
                {12, 6.96f}, {49422, 6.80f}, {6163, 6.68f}, {1358, 6.55f},
                {96220, 6.42f}, {112523, 6.30f}, {96847, 6.22f}, {104980, 6.12f},
                {98936, 6.05f}, {109120, 5.96f}, {271, 5.86f}, {248068, 5.74f},
                {8160, 5.63f}, {579, 5.51f}, {264, 5.40f}, {7047, 5.28f},
                {1817, 5.16f}, {25, 5.04f}, {16, 4.93f}, {13, 4.81f},
                {220, 4.70f}, {2014, 4.59f}, {53983, 4.47f}, {2570, 4.36f},
                {5396, 4.25f}, {1891, 4.13f}, {28758, 4.02f}, {99943, 3.91f},
            },
            {
                {256, 9.0100f}, {262, 9.0070f}, {471, 8.84f}, {2972, 8.55f},
                {1421, 8.36f}, {23398, 8.21f}, {13, 8.04f}, {198, 7.93f},
                {681, 7.80f}, {8193, 7.69f}, {883, 7.57f}, {36515, 7.44f},
                {6163, 7.32f}, {96847, 7.20f}, {1, 7.07f}, {11436, 6.95f},
                {12410, 6.84f}, {13410, 6.73f}, {1414, 6.62f}, {15613, 6.51f},
                {29223, 6.40f}, {28254, 6.29f}, {836, 6.18f}, {1919, 6.07f},
                {11, 5.96f}, {271, 5.85f}, {1835, 5.74f}, {5077, 5.63f},
                {3710, 5.52f}, {5839, 5.41f}, {5757, 5.30f}, {159034, 5.19f},
                {1503, 5.08f}, {2144, 4.97f}, {3766, 4.86f}, {16545, 4.75f},
                {51121, 4.64f}, {22527, 4.53f}, {6157, 4.42f}, {77777, 4.31f},
            },
            {
                {271, 9.15f}, {198, 8.98f}, {220, 8.77f}, {25, 8.51f},
                {16, 8.36f}, {2014, 8.24f}, {2972, 8.12f}, {579, 7.98f},
                {7047, 7.87f}, {64700, 7.74f}, {2824, 7.61f}, {2570, 7.49f},
                {262, 7.31f}, {256, 7.29f}, {11, 7.15f}, {12, 7.01f},
                {6163, 6.88f}, {49422, 6.75f}, {15352, 6.63f}, {357, 6.51f},
                {471, 6.40f}, {2425, 6.29f}, {5396, 6.18f}, {1358, 6.07f},
                {96220, 5.96f}, {112523, 5.85f}, {96847, 5.74f}, {104980, 5.63f},
                {98936, 5.52f}, {109120, 5.41f}, {248068, 5.30f}, {8160, 5.19f},
                {264, 5.08f}, {1817, 4.97f}, {13, 4.86f}, {53983, 4.75f},
                {1891, 4.64f}, {28758, 4.53f}, {99943, 4.42f}, {836, 4.31f},
            },
        };

        void *d_logits = nullptr;
        void *d_token_ids = nullptr;
        void *d_probs = nullptr;
        void *d_sample_token = nullptr;
        void *d_direct_token = nullptr;

        auto cleanup = [&]()
        {
            if (d_logits)
                backend_->free(d_logits, device_id_);
            if (d_token_ids)
                backend_->free(d_token_ids, device_id_);
            if (d_probs)
                backend_->free(d_probs, device_id_);
            if (d_sample_token)
                backend_->free(d_sample_token, device_id_);
            if (d_direct_token)
                backend_->free(d_direct_token, device_id_);
        };

        d_logits = backend_->allocate(static_cast<size_t>(vocab_size) * sizeof(float), device_id_);
        d_token_ids = backend_->allocate(top_k * sizeof(int), device_id_);
        d_probs = backend_->allocate(top_k * sizeof(float), device_id_);
        d_sample_token = backend_->allocate(sizeof(int), device_id_);
        d_direct_token = backend_->allocate(sizeof(int), device_id_);
        ASSERT_NE(d_logits, nullptr);
        ASSERT_NE(d_token_ids, nullptr);
        ASSERT_NE(d_probs, nullptr);
        ASSERT_NE(d_sample_token, nullptr);
        ASSERT_NE(d_direct_token, nullptr);

        auto run_row = [&](IWorkerGPUContext &ctx,
                           const std::vector<float> &logits,
                           float threshold,
                           uint64_t offset)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);

                ASSERT_TRUE(backend_->hostToDevice(
                    d_logits,
                    logits.data(),
                    logits.size() * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueBuildTopKTopPDistributionF32Device(
                    d_logits,
                    vocab_size,
                    top_k,
                    top_p,
                    temperature,
                    device_id_,
                    stream,
                    d_token_ids,
                    d_probs));
                ASSERT_TRUE(backend_->enqueueSampleDistributionF32Device(
                    d_token_ids,
                    d_probs,
                    top_k,
                    threshold,
                    device_id_,
                    stream,
                    d_sample_token));
                ASSERT_TRUE(backend_->enqueueSampleTopKTopPF32Device(
                    d_logits,
                    vocab_size,
                    top_k,
                    top_p,
                    temperature,
                    seed,
                    offset,
                    device_id_,
                    stream,
                    d_direct_token));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        for (size_t row = 0; row < rows.size(); ++row)
        {
            std::vector<float> logits(static_cast<size_t>(vocab_size), -18.0f);
            for (int i = 0; i < vocab_size; ++i)
                logits[static_cast<size_t>(i)] -=
                    0.00029f * static_cast<float>((i * 53 + static_cast<int>(row) * 17) % 997);
            for (const HotToken &hot : rows[row])
                logits[static_cast<size_t>(hot.token_id)] = hot.logit;

            const uint64_t offset = 17 + static_cast<uint64_t>(row) * 11;
            const float threshold = samplingUniform01(seed, offset);
            const auto expected_distribution =
                expectedTopKTopPDistribution(logits, top_k, top_p, temperature);
            const int expected_sample =
                expectedSampleDistributionWithThreshold(expected_distribution, threshold);
            const int expected_direct_sample =
                expectedTopKTopPSample(logits, top_k, top_p, temperature, seed, offset);
            ASSERT_EQ(expected_direct_sample, expected_sample)
                << "CPU direct/distribution sample mismatch at row " << row;

            if (GetParam() == "CUDA")
            {
                auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
                run_row(ctx, logits, threshold, offset);
            }
            else
            {
                auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
                run_row(ctx, logits, threshold, offset);
            }

            std::vector<int> gpu_ids(top_k, -1);
            std::vector<float> gpu_probs(top_k, 0.0f);
            int sample_token = -1;
            int direct_token = -1;
            ASSERT_TRUE(backend_->deviceToHost(
                gpu_ids.data(), d_token_ids, top_k * sizeof(int), device_id_));
            ASSERT_TRUE(backend_->deviceToHost(
                gpu_probs.data(), d_probs, top_k * sizeof(float), device_id_));
            ASSERT_TRUE(backend_->deviceToHost(
                &sample_token, d_sample_token, sizeof(int), device_id_));
            ASSERT_TRUE(backend_->deviceToHost(
                &direct_token, d_direct_token, sizeof(int), device_id_));

            for (int i = 0; i < top_k; ++i)
            {
                EXPECT_EQ(gpu_ids[static_cast<size_t>(i)],
                          expected_distribution[static_cast<size_t>(i)].token_id)
                    << "real-logit-style row " << row
                    << " compact distribution token mismatch at slot " << i;
                EXPECT_NEAR(gpu_probs[static_cast<size_t>(i)],
                            expected_distribution[static_cast<size_t>(i)].probability,
                            1e-5f)
                    << "real-logit-style row " << row
                    << " compact distribution probability mismatch at slot " << i;
            }
            EXPECT_EQ(sample_token, expected_sample)
                << "real-logit-style row " << row
                << " compact distribution sample mismatch";
            EXPECT_EQ(direct_token, expected_direct_sample)
                << "real-logit-style row " << row
                << " direct top-k/top-p sample mismatch";
        }

        cleanup();
    }

    TEST_P(GPUSamplingTest, TopKTopPProcessedLogits_Qwen36VocabTopK40_MatchesCPUAndCaptures)
    {
        constexpr int row_count = 3;
        constexpr int vocab_size = 248320;
        constexpr int row_stride = vocab_size + 7;
        constexpr int out_stride = vocab_size + 11;
        constexpr int top_k = 40;
        constexpr float top_p = 0.95f;
        constexpr float temperature = 0.6f;
        constexpr int scratch_capacity = row_count * 128 * top_k;
        const std::array<float, row_count> thresholds = {0.11f, 0.53f, 0.87f};

        struct HotToken
        {
            int token_id;
            float logit;
        };
        const std::array<std::array<HotToken, top_k>, row_count> hot_rows = {{
            {{
                {262, 8.9500f}, {256, 8.9475f}, {198, 8.72f}, {471, 8.31f},
                {2972, 8.05f}, {2425, 7.92f}, {2824, 7.81f}, {64700, 7.62f},
                {357, 7.51f}, {15352, 7.42f}, {11, 7.25f}, {1575, 7.10f},
                {12, 6.96f}, {49422, 6.80f}, {6163, 6.68f}, {1358, 6.55f},
                {96220, 6.42f}, {112523, 6.30f}, {96847, 6.22f}, {104980, 6.12f},
                {98936, 6.05f}, {109120, 5.96f}, {271, 5.86f}, {248068, 5.74f},
                {8160, 5.63f}, {579, 5.51f}, {264, 5.40f}, {7047, 5.28f},
                {1817, 5.16f}, {25, 5.04f}, {16, 4.93f}, {13, 4.81f},
                {220, 4.70f}, {2014, 4.59f}, {53983, 4.47f}, {2570, 4.36f},
                {5396, 4.25f}, {1891, 4.13f}, {28758, 4.02f}, {99943, 3.91f},
            }},
            {{
                {256, 9.0100f}, {262, 9.0070f}, {471, 8.84f}, {2972, 8.55f},
                {1421, 8.36f}, {23398, 8.21f}, {13, 8.04f}, {198, 7.93f},
                {681, 7.80f}, {8193, 7.69f}, {883, 7.57f}, {36515, 7.44f},
                {6163, 7.32f}, {96847, 7.20f}, {1, 7.07f}, {11436, 6.95f},
                {12410, 6.84f}, {13410, 6.73f}, {1414, 6.62f}, {15613, 6.51f},
                {29223, 6.40f}, {28254, 6.29f}, {836, 6.18f}, {1919, 6.07f},
                {11, 5.96f}, {271, 5.85f}, {1835, 5.74f}, {5077, 5.63f},
                {3710, 5.52f}, {5839, 5.41f}, {5757, 5.30f}, {159034, 5.19f},
                {1503, 5.08f}, {2144, 4.97f}, {3766, 4.86f}, {16545, 4.75f},
                {51121, 4.64f}, {22527, 4.53f}, {6157, 4.42f}, {77777, 4.31f},
            }},
            {{
                {271, 9.15f}, {198, 8.98f}, {220, 8.77f}, {25, 8.51f},
                {16, 8.36f}, {2014, 8.24f}, {2972, 8.12f}, {579, 7.98f},
                {7047, 7.87f}, {64700, 7.74f}, {2824, 7.61f}, {2570, 7.49f},
                {262, 7.31f}, {256, 7.29f}, {11, 7.15f}, {12, 7.01f},
                {6163, 6.88f}, {49422, 6.75f}, {15352, 6.63f}, {357, 6.51f},
                {471, 6.40f}, {2425, 6.29f}, {5396, 6.18f}, {1358, 6.07f},
                {96220, 5.96f}, {112523, 5.85f}, {96847, 5.74f}, {104980, 5.63f},
                {98936, 5.52f}, {109120, 5.41f}, {248068, 5.30f}, {8160, 5.19f},
                {264, 5.08f}, {1817, 4.97f}, {13, 4.86f}, {53983, 4.75f},
                {1891, 4.64f}, {28758, 4.53f}, {99943, 4.42f}, {836, 4.31f},
            }},
        }};

        std::vector<float> logits(static_cast<size_t>(row_count) * row_stride, -18.0f);
        std::vector<std::vector<float>> expected_processed;
        std::vector<std::vector<ExpectedDistributionEntry>> expected_distributions;
        std::array<int, row_count> expected_samples{};
        std::array<float, row_count> expected_sample_probs{};
        expected_processed.reserve(row_count);
        expected_distributions.reserve(row_count);
        for (int row = 0; row < row_count; ++row)
        {
            std::vector<float> row_logits(static_cast<size_t>(vocab_size), -18.0f);
            for (int token = 0; token < vocab_size; ++token)
            {
                row_logits[static_cast<size_t>(token)] -=
                    0.00029f * static_cast<float>((token * 53 + row * 17) % 997);
            }
            for (const HotToken &hot : hot_rows[static_cast<size_t>(row)])
                row_logits[static_cast<size_t>(hot.token_id)] = hot.logit;

            std::copy(row_logits.begin(), row_logits.end(),
                      logits.begin() + static_cast<ptrdiff_t>(row * row_stride));
            expected_distributions.push_back(
                expectedTopKTopPDistribution(row_logits, top_k, top_p, temperature));
            expected_processed.push_back(
                expectedTopKTopPProcessedLogits(row_logits, top_k, top_p, temperature));
            expected_samples[static_cast<size_t>(row)] =
                sampleMTPTokenFromProcessedLogits(
                    expected_processed.back().data(),
                    vocab_size,
                    thresholds[static_cast<size_t>(row)]);
            const MTPFullLogitRowStats stats =
                computeMTPFullLogitRowStats(
                    expected_processed.back().data(),
                    vocab_size);
            ASSERT_TRUE(stats.ok) << stats.error;
            expected_sample_probs[static_cast<size_t>(row)] =
                probabilityFromMTPFullLogits(
                    expected_processed.back().data(),
                    vocab_size,
                    stats,
                    expected_samples[static_cast<size_t>(row)]);
        }

        void *d_logits = backend_->allocate(logits.size() * sizeof(float), device_id_);
        void *d_processed = backend_->allocate(
            static_cast<size_t>(row_count) * out_stride * sizeof(float),
            device_id_);
        void *d_scratch_values = backend_->allocate(
            static_cast<size_t>(scratch_capacity) * sizeof(float),
            device_id_);
        void *d_scratch_indices = backend_->allocate(
            static_cast<size_t>(scratch_capacity) * sizeof(int),
            device_id_);
        void *d_samples = backend_->allocate(row_count * sizeof(int), device_id_);
        void *d_sample_probs =
            backend_->allocate(row_count * sizeof(float), device_id_);

        auto cleanup = [&]()
        {
            void *ptrs[] = {
                d_logits, d_processed, d_scratch_values, d_scratch_indices,
                d_samples, d_sample_probs};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        ASSERT_NE(d_logits, nullptr);
        ASSERT_NE(d_processed, nullptr);
        ASSERT_NE(d_scratch_values, nullptr);
        ASSERT_NE(d_scratch_indices, nullptr);
        ASSERT_NE(d_samples, nullptr);
        ASSERT_NE(d_sample_probs, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(backend_->hostToDevice(
                    d_logits,
                    logits.data(),
                    logits.size() * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                EXPECT_FALSE(backend_->enqueueBuildTopKTopPProcessedLogitsF32Device(
                    d_logits,
                    row_count,
                    vocab_size,
                    row_stride,
                    top_k,
                    top_p,
                    temperature,
                    device_id_,
                    nullptr,
                    d_processed,
                    out_stride,
                    d_scratch_values,
                    d_scratch_indices,
                    scratch_capacity))
                    << "processed-logit top-k/top-p warper must reject the legacy default/null stream";

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueBuildTopKTopPProcessedLogitsF32Device(
                    d_logits,
                    row_count,
                    vocab_size,
                    row_stride,
                    top_k,
                    top_p,
                    temperature,
                    device_id_,
                    stream,
                    d_processed,
                    out_stride,
                    d_scratch_values,
                    d_scratch_indices,
                    scratch_capacity));
                for (int row = 0; row < row_count; ++row)
                {
                    ASSERT_TRUE(backend_->enqueueSampleProcessedLogitsF32Device(
                        static_cast<float *>(d_processed) +
                            static_cast<size_t>(row) * out_stride,
                        vocab_size,
                        out_stride,
                        thresholds[static_cast<size_t>(row)],
                        device_id_,
                        stream,
                        static_cast<int *>(d_samples) + row,
                        static_cast<float *>(d_sample_probs) + row));
                }
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::vector<float> gpu_processed(static_cast<size_t>(row_count) * out_stride, 0.0f);
        std::array<int, row_count> gpu_samples{};
        std::array<float, row_count> gpu_sample_probs{};
        ASSERT_TRUE(backend_->deviceToHost(
            gpu_processed.data(),
            d_processed,
            gpu_processed.size() * sizeof(float),
            device_id_));
        ASSERT_TRUE(backend_->deviceToHost(
            gpu_samples.data(),
            d_samples,
            gpu_samples.size() * sizeof(int),
            device_id_));
        ASSERT_TRUE(backend_->deviceToHost(
            gpu_sample_probs.data(),
            d_sample_probs,
            gpu_sample_probs.size() * sizeof(float),
            device_id_));
        cleanup();

        for (int row = 0; row < row_count; ++row)
        {
            const float *gpu_row =
                gpu_processed.data() + static_cast<size_t>(row) * out_stride;
            const auto &expected_row = expected_processed[static_cast<size_t>(row)];
            int finite_count = 0;
            int expected_finite_count = 0;
            int mismatch_count = 0;
            for (int token = 0; token < vocab_size; ++token)
            {
                const bool gpu_active = std::isfinite(gpu_row[token]);
                const bool expected_active =
                    std::isfinite(expected_row[static_cast<size_t>(token)]);
                finite_count += gpu_active ? 1 : 0;
                expected_finite_count += expected_active ? 1 : 0;
                if (gpu_active != expected_active)
                {
                    ++mismatch_count;
                    continue;
                }
                if (expected_active &&
                    std::fabs(gpu_row[token] -
                              expected_row[static_cast<size_t>(token)]) > 1e-5f)
                {
                    ++mismatch_count;
                }
            }

            EXPECT_EQ(finite_count, expected_finite_count)
                << "processed-logit active-token count mismatch at row " << row;
            EXPECT_EQ(mismatch_count, 0)
                << "processed-logit mask/value mismatch at row " << row;

            const MTPFullLogitRowStats stats =
                computeMTPFullLogitRowStats(gpu_row, vocab_size);
            ASSERT_TRUE(stats.ok) << stats.error;
            for (const auto &entry : expected_distributions[static_cast<size_t>(row)])
            {
                if (entry.token_id < 0 || !(entry.probability > 0.0f))
                    continue;
                const float gpu_probability =
                    probabilityFromMTPFullLogits(
                        gpu_row,
                        vocab_size,
                        stats,
                        entry.token_id);
                EXPECT_NEAR(gpu_probability, entry.probability, 2e-5f)
                    << "processed-logit probability mismatch at row " << row
                    << ", token " << entry.token_id;
            }

            EXPECT_EQ(gpu_samples[static_cast<size_t>(row)],
                      expected_samples[static_cast<size_t>(row)])
                << "processed-logit sample mismatch at row " << row;
            EXPECT_NEAR(gpu_sample_probs[static_cast<size_t>(row)],
                        expected_sample_probs[static_cast<size_t>(row)],
                        2e-5f)
                << "processed-logit selected probability mismatch at row "
                << row;
        }
    }

    TEST_P(GPUSamplingTest, ProcessedLogitSpeculativeVerifyMatchesReferenceAndCaptures)
    {
        constexpr int vocab_size = 16;
        constexpr int row_count = 2;
        const std::array<int, row_count> draft_tokens = {0, 0};
        const std::array<float, row_count> accept_thresholds = {0.39f, 0.41f};
        const std::array<float, row_count> residual_thresholds = {0.0f, 0.0f};

        auto logits_from_probs = [](std::initializer_list<float> probs)
        {
            std::vector<float> logits;
            logits.reserve(probs.size());
            for (float p : probs)
            {
                logits.push_back(
                    p > 0.0f
                        ? std::log(p)
                        : -std::numeric_limits<float>::infinity());
            }
            return logits;
        };
        auto append_row = [](std::vector<float> &rows,
                             const std::vector<float> &row)
        {
            rows.insert(rows.end(), row.begin(), row.end());
        };

        const std::vector<float> target_row =
            logits_from_probs({0.2f, 0.8f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f});
        const std::vector<float> draft_row =
            logits_from_probs({0.5f, 0.5f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f});

        std::vector<float> target_rows;
        std::vector<float> draft_rows;
        append_row(target_rows, target_row);
        append_row(target_rows, target_row);
        append_row(draft_rows, draft_row);
        append_row(draft_rows, draft_row);

        std::array<MTPRejectionSampleRowResult, row_count> expected;
        for (int row = 0; row < row_count; ++row)
        {
            expected[static_cast<size_t>(row)] =
                sampleMTPRejectionRowFromProcessedLogits(
                    target_rows.data() + static_cast<size_t>(row) * vocab_size,
                    draft_rows.data() + static_cast<size_t>(row) * vocab_size,
                    vocab_size,
                    draft_tokens[static_cast<size_t>(row)],
                    accept_thresholds[static_cast<size_t>(row)],
                    residual_thresholds[static_cast<size_t>(row)]);
            ASSERT_TRUE(expected[static_cast<size_t>(row)].ok)
                << expected[static_cast<size_t>(row)].error;
        }
        ASSERT_TRUE(expected[0].accepted);
        ASSERT_FALSE(expected[1].accepted);

        void *d_target = backend_->allocate(target_rows.size() * sizeof(float), device_id_);
        void *d_draft = backend_->allocate(draft_rows.size() * sizeof(float), device_id_);
        void *d_draft_tokens = backend_->allocate(draft_tokens.size() * sizeof(int), device_id_);
        void *d_tokens = backend_->allocate(row_count * sizeof(int), device_id_);
        void *d_accepted = backend_->allocate(row_count * sizeof(int), device_id_);
        void *d_accept_probs = backend_->allocate(row_count * sizeof(float), device_id_);
        void *d_accept_thresholds = backend_->allocate(row_count * sizeof(float), device_id_);

        auto cleanup = [&]()
        {
            void *ptrs[] = {
                d_target, d_draft, d_draft_tokens, d_tokens,
                d_accepted, d_accept_probs, d_accept_thresholds};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        ASSERT_NE(d_target, nullptr);
        ASSERT_NE(d_draft, nullptr);
        ASSERT_NE(d_draft_tokens, nullptr);
        ASSERT_NE(d_tokens, nullptr);
        ASSERT_NE(d_accepted, nullptr);
        ASSERT_NE(d_accept_probs, nullptr);
        ASSERT_NE(d_accept_thresholds, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(backend_->hostToDevice(
                    d_target, target_rows.data(),
                    target_rows.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_draft, draft_rows.data(),
                    draft_rows.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_draft_tokens, draft_tokens.data(),
                    draft_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                EXPECT_FALSE(
                    backend_->enqueueSpeculativeVerifyProcessedLogitsF32DeviceThresholdsBatchDeviceTokens(
                        d_target, d_draft, row_count, vocab_size, vocab_size,
                        vocab_size, d_draft_tokens, accept_thresholds.data(),
                        residual_thresholds.data(), device_id_, nullptr,
                        d_tokens, d_accepted, d_accept_probs,
                        d_accept_thresholds))
                    << "processed-logit verifier must reject the legacy default/null stream";

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(
                    backend_->enqueueSpeculativeVerifyProcessedLogitsF32DeviceThresholdsBatchDeviceTokens(
                        d_target, d_draft, row_count, vocab_size, vocab_size,
                        vocab_size, d_draft_tokens, accept_thresholds.data(),
                        residual_thresholds.data(), device_id_, stream,
                        d_tokens, d_accepted, d_accept_probs,
                        d_accept_thresholds));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::array<int, row_count> tokens{};
        std::array<int, row_count> accepted{};
        std::array<float, row_count> accept_probs{};
        ASSERT_TRUE(backend_->deviceToHost(tokens.data(), d_tokens,
                                           tokens.size() * sizeof(int), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(accepted.data(), d_accepted,
                                           accepted.size() * sizeof(int), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(accept_probs.data(), d_accept_probs,
                                           accept_probs.size() * sizeof(float), device_id_));

        cleanup();

        for (int row = 0; row < row_count; ++row)
        {
            const auto &exp = expected[static_cast<size_t>(row)];
            EXPECT_EQ(tokens[static_cast<size_t>(row)], exp.token)
                << "row " << row;
            EXPECT_EQ(accepted[static_cast<size_t>(row)], exp.accepted ? 1 : 0)
                << "row " << row;
            EXPECT_NEAR(accept_probs[static_cast<size_t>(row)],
                        exp.accept_probability,
                        1e-5f)
                << "row " << row;
        }
    }

    TEST_P(GPUSamplingTest, ProcessedLogitSoftmaxMatchesReferenceAndCaptures)
    {
        constexpr int vocab_size = 8;
        constexpr int row_count = 2;
        const float neg_inf = -std::numeric_limits<float>::infinity();
        const std::vector<float> logits = {
            neg_inf, 2.0f, 1.0f, neg_inf, 0.0f, neg_inf, neg_inf, neg_inf,
            4.0f, 4.0f, neg_inf, 1.0f, neg_inf, neg_inf, 0.0f, neg_inf};
        std::vector<float> expected(logits.size(), 0.0f);
        for (int row = 0; row < row_count; ++row)
        {
            const float *row_logits =
                logits.data() + static_cast<size_t>(row) * vocab_size;
            const MTPFullLogitRowStats stats =
                computeMTPFullLogitRowStats(row_logits, vocab_size);
            ASSERT_TRUE(stats.ok) << stats.error;
            for (int token = 0; token < vocab_size; ++token)
            {
                expected[static_cast<size_t>(row) * vocab_size + token] =
                    probabilityFromMTPFullLogits(
                        row_logits,
                        vocab_size,
                        stats,
                        token);
            }
        }

        void *d_logits = backend_->allocate(logits.size() * sizeof(float), device_id_);
        void *d_probs = backend_->allocate(expected.size() * sizeof(float), device_id_);
        auto cleanup = [&]()
        {
            if (d_logits)
                backend_->free(d_logits, device_id_);
            if (d_probs)
                backend_->free(d_probs, device_id_);
        };

        ASSERT_NE(d_logits, nullptr);
        ASSERT_NE(d_probs, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(backend_->hostToDevice(
                    d_logits, logits.data(), logits.size() * sizeof(float),
                    device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                EXPECT_FALSE(backend_->enqueueSoftmaxProcessedLogitsF32Device(
                    d_logits, row_count, vocab_size, vocab_size,
                    device_id_, nullptr, d_probs, vocab_size))
                    << "processed-logit softmax must reject the legacy default/null stream";

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueSoftmaxProcessedLogitsF32Device(
                    d_logits, row_count, vocab_size, vocab_size,
                    device_id_, stream, d_probs, vocab_size));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::vector<float> actual(expected.size(), -1.0f);
        ASSERT_TRUE(backend_->deviceToHost(
            actual.data(), d_probs, actual.size() * sizeof(float), device_id_));
        cleanup();

        for (size_t i = 0; i < actual.size(); ++i)
        {
            EXPECT_NEAR(actual[i], expected[i], 1e-6f)
                << "probability index " << i;
        }
    }

    TEST_P(GPUSamplingTest, TemperatureDraftProposalSoftmaxSampleMatchesReferenceAndCaptures)
    {
        constexpr int vocab_size = 257;
        constexpr float temperature = 0.73f;
        constexpr float threshold = 0.61f;

        std::vector<float> logits(vocab_size, -7.0f);
        for (int token = 0; token < vocab_size; ++token)
        {
            logits[static_cast<size_t>(token)] =
                std::sin(static_cast<float>(token) * 0.13f) * 2.0f -
                static_cast<float>(token % 11) * 0.07f;
        }
        logits[3] = 8.0f;
        logits[17] = 7.5f;
        logits[199] = 6.75f;

        const std::vector<float> expected_probs =
            expectedTemperatureOnlyProbabilities(logits, temperature);
        const int expected_token =
            expectedTemperatureOnlySampleToken(expected_probs, threshold);
        ASSERT_GE(expected_token, 0);
        ASSERT_LT(expected_token, vocab_size);
        const float expected_probability =
            expected_probs[static_cast<size_t>(expected_token)];

        void *d_logits = backend_->allocate(logits.size() * sizeof(float), device_id_);
        void *d_probs = backend_->allocate(expected_probs.size() * sizeof(float), device_id_);
        void *d_token = backend_->allocate(sizeof(int), device_id_);
        void *d_probability = backend_->allocate(sizeof(float), device_id_);
        auto cleanup = [&]()
        {
            if (d_logits)
                backend_->free(d_logits, device_id_);
            if (d_probs)
                backend_->free(d_probs, device_id_);
            if (d_token)
                backend_->free(d_token, device_id_);
            if (d_probability)
                backend_->free(d_probability, device_id_);
        };
        ASSERT_NE(d_logits, nullptr);
        ASSERT_NE(d_probs, nullptr);
        ASSERT_NE(d_token, nullptr);
        ASSERT_NE(d_probability, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(backend_->hostToDevice(
                    d_logits, logits.data(), logits.size() * sizeof(float),
                    device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                EXPECT_FALSE(backend_->enqueueSoftmaxAndSampleTemperatureLogitsF32Device(
                    d_logits,
                    vocab_size,
                    vocab_size,
                    temperature,
                    threshold,
                    device_id_,
                    nullptr,
                    d_probs,
                    vocab_size,
                    d_token,
                    d_probability))
                    << "temperature draft proposal must reject the legacy default/null stream";

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueSoftmaxAndSampleTemperatureLogitsF32Device(
                    d_logits,
                    vocab_size,
                    vocab_size,
                    temperature,
                    threshold,
                    device_id_,
                    stream,
                    d_probs,
                    vocab_size,
                    d_token,
                    d_probability));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::vector<float> actual_probs(expected_probs.size(), -1.0f);
        int actual_token = -1;
        float actual_probability = -1.0f;
        ASSERT_TRUE(backend_->deviceToHost(
            actual_probs.data(),
            d_probs,
            actual_probs.size() * sizeof(float),
            device_id_));
        ASSERT_TRUE(backend_->deviceToHost(
            &actual_token,
            d_token,
            sizeof(int),
            device_id_));
        ASSERT_TRUE(backend_->deviceToHost(
            &actual_probability,
            d_probability,
            sizeof(float),
            device_id_));
        cleanup();

        float actual_sum = 0.0f;
        for (size_t i = 0; i < actual_probs.size(); ++i)
        {
            actual_sum += actual_probs[i];
            EXPECT_NEAR(actual_probs[i], expected_probs[i], 2e-6f)
                << "probability index " << i;
        }
        EXPECT_NEAR(actual_sum, 1.0f, 2e-5f);
        EXPECT_EQ(actual_token, expected_token);
        EXPECT_NEAR(actual_probability, expected_probability, 2e-6f);
    }

    TEST_P(GPUSamplingTest, TemperatureDraftLogitProposalAndVerifierMatchesReferenceAndCaptures)
    {
        constexpr int vocab_size = 257;
        constexpr int row_count = 3;
        constexpr float temperature = 0.77f;
        constexpr uint64_t inverse_sample_seed = 98765;
        constexpr int first_logical_position = 19;
        constexpr uint64_t kInverseSampleDomain = 0xA0761D6478BD642FULL;
        const float neg_inf = -std::numeric_limits<float>::infinity();

        const std::array<float, row_count> proposal_thresholds = {0.19f, 0.47f, 0.83f};
        const std::array<float, row_count> accept_thresholds = {0.25f, 0.92f, 0.10f};

        std::vector<float> draft_raw_logits(
            static_cast<size_t>(row_count) * vocab_size,
            -5.0f);
        for (int row = 0; row < row_count; ++row)
        {
            const size_t base = static_cast<size_t>(row) * vocab_size;
            for (int token = 0; token < vocab_size; ++token)
            {
                draft_raw_logits[base + static_cast<size_t>(token)] =
                    std::sin(static_cast<float>(token + row * 17) * 0.17f) * 1.7f -
                    0.015f * static_cast<float>((token + row * 31) % 29);
            }
            draft_raw_logits[base + static_cast<size_t>((13 + row * 41) % vocab_size)] = 6.1f;
            draft_raw_logits[base + static_cast<size_t>((71 + row * 29) % vocab_size)] = 5.3f;
        }

        std::vector<float> expected_draft_logits(draft_raw_logits.size(), neg_inf);
        std::vector<float> draft_probs(draft_raw_logits.size(), 0.0f);
        std::array<int, row_count> draft_tokens{};
        std::array<float, row_count> draft_token_probs{};
        for (int row = 0; row < row_count; ++row)
        {
            const size_t base = static_cast<size_t>(row) * vocab_size;
            std::vector<float> row_logits(
                draft_raw_logits.begin() + static_cast<std::ptrdiff_t>(base),
                draft_raw_logits.begin() + static_cast<std::ptrdiff_t>(base + vocab_size));
            std::vector<float> row_probs =
                expectedTemperatureOnlyProbabilities(row_logits, temperature);
            draft_tokens[static_cast<size_t>(row)] =
                expectedTemperatureOnlySampleToken(
                    row_probs, proposal_thresholds[static_cast<size_t>(row)]);
            draft_token_probs[static_cast<size_t>(row)] =
                row_probs[static_cast<size_t>(draft_tokens[static_cast<size_t>(row)])];
            for (int token = 0; token < vocab_size; ++token)
            {
                expected_draft_logits[base + static_cast<size_t>(token)] =
                    draft_raw_logits[base + static_cast<size_t>(token)] / temperature;
                draft_probs[base + static_cast<size_t>(token)] =
                    row_probs[static_cast<size_t>(token)];
            }
        }

        std::vector<float> target_logits(
            static_cast<size_t>(row_count) * vocab_size,
            neg_inf);
        std::vector<float> target_probs(
            static_cast<size_t>(row_count) * vocab_size,
            0.0f);
        auto set_target_row =
            [&](int row, float draft_prob, float alt0_prob, float alt1_prob)
        {
            ASSERT_NEAR(draft_prob + alt0_prob + alt1_prob, 1.0f, 1e-6f);
            const size_t base = static_cast<size_t>(row) * vocab_size;
            const int draft_token = draft_tokens[static_cast<size_t>(row)];
            const int alt0 = (draft_token + 17) % vocab_size;
            const int alt1 = (draft_token + 89) % vocab_size;
            target_probs[base + static_cast<size_t>(draft_token)] = draft_prob;
            target_probs[base + static_cast<size_t>(alt0)] = alt0_prob;
            target_probs[base + static_cast<size_t>(alt1)] = alt1_prob;
            target_logits[base + static_cast<size_t>(draft_token)] = std::log(draft_prob);
            target_logits[base + static_cast<size_t>(alt0)] = std::log(alt0_prob);
            target_logits[base + static_cast<size_t>(alt1)] = std::log(alt1_prob);
        };
        set_target_row(0, 0.80f, 0.12f, 0.08f);
        set_target_row(1, 0.01f, 0.70f, 0.29f);
        set_target_row(2, 0.55f, 0.25f, 0.20f);

        std::vector<float> inverse_rows(draft_raw_logits.size(), 0.0f);
        for (int row = 0; row < row_count; ++row)
        {
            for (int token = 0; token < vocab_size; ++token)
            {
                const uint64_t offset =
                    static_cast<uint64_t>(first_logical_position + row) *
                        static_cast<uint64_t>(vocab_size) +
                    static_cast<uint64_t>(token);
                const float uniform =
                    sampling_math::uniform01(
                        inverse_sample_seed ^ kInverseSampleDomain,
                        offset);
                inverse_rows[static_cast<size_t>(row) * vocab_size +
                             static_cast<size_t>(token)] =
                    sampling_math::inverse_exponential_from_uniform(uniform);
            }
        }

        std::array<MTPRejectionSampleRowResult, row_count> expected{};
        for (int row = 0; row < row_count; ++row)
        {
            const size_t base = static_cast<size_t>(row) * vocab_size;
            expected[static_cast<size_t>(row)] =
                sampleMTPRejectionRowFromProbabilities(
                    target_probs.data() + base,
                    draft_probs.data() + base,
                    inverse_rows.data() + base,
                    vocab_size,
                    draft_tokens[static_cast<size_t>(row)],
                    accept_thresholds[static_cast<size_t>(row)]);
            ASSERT_TRUE(expected[static_cast<size_t>(row)].ok)
                << expected[static_cast<size_t>(row)].error;
        }
        ASSERT_TRUE(expected[0].accepted);
        ASSERT_FALSE(expected[1].accepted);
        ASSERT_TRUE(expected[2].accepted);

        void *d_target = backend_->allocate(target_logits.size() * sizeof(float), device_id_);
        void *d_draft_raw = backend_->allocate(draft_raw_logits.size() * sizeof(float), device_id_);
        void *d_draft_logits = backend_->allocate(expected_draft_logits.size() * sizeof(float), device_id_);
        void *d_draft_tokens = backend_->allocate(draft_tokens.size() * sizeof(int), device_id_);
        void *d_draft_token_probs = backend_->allocate(draft_token_probs.size() * sizeof(float), device_id_);
        void *d_tokens = backend_->allocate(row_count * sizeof(int), device_id_);
        void *d_accepted = backend_->allocate(row_count * sizeof(int), device_id_);
        void *d_accept_probs = backend_->allocate(row_count * sizeof(float), device_id_);
        void *d_accept_thresholds = backend_->allocate(row_count * sizeof(float), device_id_);
        auto cleanup = [&]()
        {
            void *ptrs[] = {
                d_target, d_draft_raw, d_draft_logits, d_draft_tokens,
                d_draft_token_probs, d_tokens, d_accepted,
                d_accept_probs, d_accept_thresholds};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        ASSERT_NE(d_target, nullptr);
        ASSERT_NE(d_draft_raw, nullptr);
        ASSERT_NE(d_draft_logits, nullptr);
        ASSERT_NE(d_draft_tokens, nullptr);
        ASSERT_NE(d_draft_token_probs, nullptr);
        ASSERT_NE(d_tokens, nullptr);
        ASSERT_NE(d_accepted, nullptr);
        ASSERT_NE(d_accept_probs, nullptr);
        ASSERT_NE(d_accept_thresholds, nullptr);

        std::array<int, row_count> first_tokens{};
        std::array<int, row_count> first_accepted{};
        std::array<float, row_count> first_accept_probs{};
        std::array<int, row_count> second_tokens{};
        std::array<int, row_count> second_accepted{};
        std::vector<float> actual_draft_logits(expected_draft_logits.size(), 0.0f);
        std::array<int, row_count> actual_draft_tokens{};
        std::array<float, row_count> actual_draft_token_probs{};

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(backend_->hostToDevice(
                    d_target, target_logits.data(),
                    target_logits.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_draft_raw, draft_raw_logits.data(),
                    draft_raw_logits.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                EXPECT_FALSE(backend_->enqueueScaleAndSampleTemperatureLogitsF32Device(
                    d_draft_raw, vocab_size, vocab_size, temperature,
                    proposal_thresholds[0], device_id_, nullptr,
                    d_draft_logits, vocab_size, d_draft_tokens,
                    d_draft_token_probs))
                    << "temperature draft-logit proposal must reject the legacy default/null stream";
                EXPECT_FALSE(
                    backend_->enqueueSpeculativeVerifyProcessedTargetDraftLogitsF32DeviceThresholdsBatchDeviceTokens(
                        d_target, d_draft_logits, row_count, vocab_size, vocab_size,
                        vocab_size, d_draft_tokens, accept_thresholds.data(),
                        inverse_sample_seed, first_logical_position,
                        device_id_, nullptr, d_tokens, d_accepted,
                        d_accept_probs, d_accept_thresholds, d_draft_token_probs))
                    << "processed-target/draft-logit verifier must reject the legacy default/null stream";

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                for (int row = 0; row < row_count; ++row)
                {
                    ASSERT_TRUE(backend_->enqueueScaleAndSampleTemperatureLogitsF32Device(
                        static_cast<float *>(d_draft_raw) +
                            static_cast<size_t>(row) * vocab_size,
                        vocab_size,
                        vocab_size,
                        temperature,
                        proposal_thresholds[static_cast<size_t>(row)],
                        device_id_,
                        stream,
                        static_cast<float *>(d_draft_logits) +
                            static_cast<size_t>(row) * vocab_size,
                        vocab_size,
                        static_cast<int *>(d_draft_tokens) + row,
                        static_cast<float *>(d_draft_token_probs) + row));
                }
                ASSERT_TRUE(
                    backend_->enqueueSpeculativeVerifyProcessedTargetDraftLogitsF32DeviceThresholdsBatchDeviceTokens(
                        d_target, d_draft_logits, row_count, vocab_size, vocab_size,
                        vocab_size, d_draft_tokens, accept_thresholds.data(),
                        inverse_sample_seed, first_logical_position,
                        device_id_, stream, d_tokens, d_accepted,
                        d_accept_probs, d_accept_thresholds, d_draft_token_probs));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());

                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
                ASSERT_TRUE(backend_->deviceToHost(
                    first_tokens.data(), d_tokens,
                    first_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->deviceToHost(
                    first_accepted.data(), d_accepted,
                    first_accepted.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->deviceToHost(
                    first_accept_probs.data(), d_accept_probs,
                    first_accept_probs.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->deviceToHost(
                    actual_draft_logits.data(), d_draft_logits,
                    actual_draft_logits.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->deviceToHost(
                    actual_draft_tokens.data(), d_draft_tokens,
                    actual_draft_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->deviceToHost(
                    actual_draft_token_probs.data(), d_draft_token_probs,
                    actual_draft_token_probs.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                const std::array<int, row_count> sentinel_tokens = {-1, -1, -1};
                const std::array<int, row_count> sentinel_accepted = {-7, -7, -7};
                ASSERT_TRUE(backend_->hostToDevice(
                    d_tokens, sentinel_tokens.data(),
                    sentinel_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_accepted, sentinel_accepted.data(),
                    sentinel_accepted.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
                ASSERT_TRUE(backend_->deviceToHost(
                    second_tokens.data(), d_tokens,
                    second_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->deviceToHost(
                    second_accepted.data(), d_accepted,
                    second_accepted.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        cleanup();

        EXPECT_EQ(second_tokens, first_tokens);
        EXPECT_EQ(second_accepted, first_accepted);
        for (int row = 0; row < row_count; ++row)
        {
            EXPECT_EQ(actual_draft_tokens[static_cast<size_t>(row)],
                      draft_tokens[static_cast<size_t>(row)])
                << "draft token row=" << row;
            EXPECT_NEAR(actual_draft_token_probs[static_cast<size_t>(row)],
                        draft_token_probs[static_cast<size_t>(row)],
                        2e-6f)
                << "draft probability row=" << row;

            const auto &exp = expected[static_cast<size_t>(row)];
            EXPECT_EQ(first_tokens[static_cast<size_t>(row)], exp.token)
                << "verifier token row=" << row;
            EXPECT_EQ(first_accepted[static_cast<size_t>(row)], exp.accepted ? 1 : 0)
                << "verifier accepted row=" << row;
            EXPECT_NEAR(first_accept_probs[static_cast<size_t>(row)],
                        exp.accept_probability,
                        2e-6f)
                << "accept probability row=" << row;
        }
        for (size_t i = 0; i < actual_draft_logits.size(); ++i)
        {
            EXPECT_NEAR(actual_draft_logits[i], expected_draft_logits[i], 2e-6f)
                << "draft logit index=" << i;
        }
    }

    TEST_P(GPUSamplingTest, InverseExponentialSamplesMatchSharedMathAndCapture)
    {
        constexpr int vocab_size = 17;
        constexpr int row_count = 2;
        constexpr int row_stride = 20;
        constexpr uint64_t seed = 12345;
        constexpr int first_logical_position = 7;
        constexpr uint64_t kInverseSampleDomain = 0xA0761D6478BD642FULL;

        std::vector<float> expected(static_cast<size_t>(row_count) * row_stride,
                                    -1.0f);
        for (int row = 0; row < row_count; ++row)
        {
            for (int token = 0; token < vocab_size; ++token)
            {
                const uint64_t logical_position =
                    static_cast<uint64_t>(first_logical_position + row);
                const uint64_t offset =
                    logical_position * static_cast<uint64_t>(vocab_size) +
                    static_cast<uint64_t>(token);
                const float uniform =
                    sampling_math::uniform01(seed ^ kInverseSampleDomain, offset);
                expected[static_cast<size_t>(row) * row_stride + token] =
                    sampling_math::inverse_exponential_from_uniform(uniform);
            }
        }

        void *d_samples = backend_->allocate(expected.size() * sizeof(float), device_id_);
        auto cleanup = [&]()
        {
            if (d_samples)
                backend_->free(d_samples, device_id_);
        };
        ASSERT_NE(d_samples, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);

                EXPECT_FALSE(backend_->enqueueFillInverseExponentialSamplesF32Device(
                    d_samples, row_count, vocab_size, row_stride,
                    seed, first_logical_position, device_id_, nullptr))
                    << "inverse-sample fill must reject the legacy default/null stream";

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueFillInverseExponentialSamplesF32Device(
                    d_samples,
                    row_count,
                    vocab_size,
                    row_stride,
                    seed,
                    first_logical_position,
                    device_id_,
                    stream));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::vector<float> actual(expected.size(), -1.0f);
        ASSERT_TRUE(backend_->deviceToHost(
            actual.data(), d_samples, actual.size() * sizeof(float), device_id_));
        cleanup();

        for (int row = 0; row < row_count; ++row)
        {
            for (int token = 0; token < vocab_size; ++token)
            {
                const size_t index = static_cast<size_t>(row) * row_stride + token;
                const float tolerance =
                    std::max(1.0e-5f, std::abs(expected[index]) * 2.0e-5f);
                EXPECT_NEAR(actual[index], expected[index], tolerance)
                    << "row=" << row << " token=" << token;
            }
        }
    }

    TEST_P(GPUSamplingTest, ProcessedTargetDraftProbabilitySpeculativeVerifyMatchesReferenceAndReplay)
    {
        constexpr int vocab_size = 248320;
        constexpr int row_count = 3;
        constexpr uint64_t inverse_sample_seed = 123;
        constexpr int first_logical_position = 11;
        constexpr uint64_t kInverseSampleDomain = 0xA0761D6478BD642FULL;
        const float neg_inf = -std::numeric_limits<float>::infinity();

        const std::array<int, row_count> draft_tokens = {13, 13, 271};
        const std::array<float, row_count> accept_thresholds = {0.95f, 0.80f, 0.49f};

        std::vector<float> target_logits(
            static_cast<size_t>(row_count) * static_cast<size_t>(vocab_size),
            neg_inf);
        std::vector<float> target_probs(
            static_cast<size_t>(row_count) * static_cast<size_t>(vocab_size),
            0.0f);
        std::vector<float> draft_probs(target_probs.size(), 0.0f);
        std::vector<float> inverse_rows(target_probs.size(), 0.0f);

        auto set_target_row =
            [&](int row, std::initializer_list<std::pair<int, float>> entries)
        {
            const size_t base =
                static_cast<size_t>(row) * static_cast<size_t>(vocab_size);
            float total = 0.0f;
            for (const auto &entry : entries)
                total += entry.second;
            ASSERT_NEAR(total, 1.0f, 1e-5f);
            for (const auto &entry : entries)
            {
                ASSERT_GE(entry.first, 0);
                ASSERT_LT(entry.first, vocab_size);
                target_probs[base + static_cast<size_t>(entry.first)] =
                    entry.second;
                target_logits[base + static_cast<size_t>(entry.first)] =
                    std::log(entry.second);
            }
        };

        auto set_draft_row =
            [&](int row, std::initializer_list<std::pair<int, float>> entries)
        {
            const size_t base =
                static_cast<size_t>(row) * static_cast<size_t>(vocab_size);
            float special_total = 0.0f;
            std::set<int> special_tokens;
            for (const auto &entry : entries)
            {
                ASSERT_GE(entry.first, 0);
                ASSERT_LT(entry.first, vocab_size);
                special_total += entry.second;
                special_tokens.insert(entry.first);
            }
            ASSERT_LT(special_total, 1.0f);
            const float background =
                (1.0f - special_total) /
                static_cast<float>(vocab_size - static_cast<int>(special_tokens.size()));
            for (int token = 0; token < vocab_size; ++token)
            {
                draft_probs[base + static_cast<size_t>(token)] =
                    special_tokens.count(token) ? 0.0f : background;
            }
            for (const auto &entry : entries)
            {
                draft_probs[base + static_cast<size_t>(entry.first)] =
                    entry.second;
            }
        };

        set_target_row(0, {{13, 0.60f}, {271, 0.20f}, {1061, 0.10f}, {33075, 0.10f}});
        set_draft_row(0, {{13, 0.20f}, {271, 0.30f}, {1061, 0.10f}, {33075, 0.05f}});

        set_target_row(1, {{13, 0.10f}, {271, 0.55f}, {1061, 0.25f}, {88, 0.10f}});
        set_draft_row(1, {{13, 0.50f}, {271, 0.05f}, {1061, 0.05f}, {88, 0.05f}});

        set_target_row(2, {{271, 0.35f}, {1061, 0.25f}, {33075, 0.20f}, {248068, 0.20f}});
        set_draft_row(2, {{271, 0.70f}, {1061, 0.05f}, {33075, 0.05f}, {248068, 0.05f}});

        for (int row = 0; row < row_count; ++row)
        {
            for (int token = 0; token < vocab_size; ++token)
            {
                const uint64_t offset =
                    static_cast<uint64_t>(first_logical_position + row) *
                        static_cast<uint64_t>(vocab_size) +
                    static_cast<uint64_t>(token);
                const float uniform = sampling_math::uniform01(
                    inverse_sample_seed ^ kInverseSampleDomain,
                    offset);
                inverse_rows[static_cast<size_t>(row) * vocab_size +
                             static_cast<size_t>(token)] =
                    sampling_math::inverse_exponential_from_uniform(uniform);
            }
        }

        std::array<MTPRejectionSampleRowResult, row_count> expected;
        for (int row = 0; row < row_count; ++row)
        {
            const size_t base =
                static_cast<size_t>(row) * static_cast<size_t>(vocab_size);
            expected[static_cast<size_t>(row)] =
                sampleMTPRejectionRowFromProbabilities(
                    target_probs.data() + base,
                    draft_probs.data() + base,
                    inverse_rows.data() + base,
                    vocab_size,
                    draft_tokens[static_cast<size_t>(row)],
                    accept_thresholds[static_cast<size_t>(row)]);
            ASSERT_TRUE(expected[static_cast<size_t>(row)].ok)
                << expected[static_cast<size_t>(row)].error;
        }
        ASSERT_TRUE(expected[0].accepted);
        ASSERT_FALSE(expected[1].accepted);
        ASSERT_TRUE(expected[2].accepted);

        void *d_target = backend_->allocate(target_logits.size() * sizeof(float), device_id_);
        void *d_draft = backend_->allocate(draft_probs.size() * sizeof(float), device_id_);
        void *d_draft_tokens = backend_->allocate(draft_tokens.size() * sizeof(int), device_id_);
        void *d_tokens = backend_->allocate(row_count * sizeof(int), device_id_);
        void *d_accepted = backend_->allocate(row_count * sizeof(int), device_id_);
        void *d_accept_probs = backend_->allocate(row_count * sizeof(float), device_id_);
        void *d_accept_thresholds = backend_->allocate(row_count * sizeof(float), device_id_);
        auto cleanup = [&]()
        {
            void *ptrs[] = {
                d_target, d_draft, d_draft_tokens, d_tokens,
                d_accepted, d_accept_probs, d_accept_thresholds};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        ASSERT_NE(d_target, nullptr);
        ASSERT_NE(d_draft, nullptr);
        ASSERT_NE(d_draft_tokens, nullptr);
        ASSERT_NE(d_tokens, nullptr);
        ASSERT_NE(d_accepted, nullptr);
        ASSERT_NE(d_accept_probs, nullptr);
        ASSERT_NE(d_accept_thresholds, nullptr);

        std::array<int, row_count> first_tokens{};
        std::array<int, row_count> first_accepted{};
        std::array<float, row_count> first_accept_probs{};
        std::array<int, row_count> second_tokens{};
        std::array<int, row_count> second_accepted{};
        std::array<float, row_count> second_accept_probs{};

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(backend_->hostToDevice(
                    d_target, target_logits.data(),
                    target_logits.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_draft, draft_probs.data(),
                    draft_probs.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_draft_tokens, draft_tokens.data(),
                    draft_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                EXPECT_FALSE(
                    backend_->enqueueSpeculativeVerifyProcessedTargetDraftProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
                        d_target, d_draft, row_count, vocab_size, vocab_size,
                        vocab_size, d_draft_tokens, accept_thresholds.data(),
                        inverse_sample_seed, first_logical_position,
                        device_id_, nullptr, d_tokens, d_accepted,
                        d_accept_probs, d_accept_thresholds))
                    << "fused processed-target verifier must reject the legacy default/null stream";

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(
                    backend_->enqueueSpeculativeVerifyProcessedTargetDraftProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
                        d_target, d_draft, row_count, vocab_size, vocab_size,
                        vocab_size, d_draft_tokens, accept_thresholds.data(),
                        inverse_sample_seed, first_logical_position,
                        device_id_, stream, d_tokens, d_accepted,
                        d_accept_probs, d_accept_thresholds));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());

                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
                ASSERT_TRUE(backend_->deviceToHost(
                    first_tokens.data(), d_tokens,
                    first_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->deviceToHost(
                    first_accepted.data(), d_accepted,
                    first_accepted.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->deviceToHost(
                    first_accept_probs.data(), d_accept_probs,
                    first_accept_probs.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                const std::array<int, row_count> sentinel_tokens = {-1, -1, -1};
                const std::array<int, row_count> sentinel_accepted = {-7, -7, -7};
                const std::array<float, row_count> sentinel_probs = {-1.0f, -1.0f, -1.0f};
                ASSERT_TRUE(backend_->hostToDevice(
                    d_tokens, sentinel_tokens.data(),
                    sentinel_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_accepted, sentinel_accepted.data(),
                    sentinel_accepted.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_accept_probs, sentinel_probs.data(),
                    sentinel_probs.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
                ASSERT_TRUE(backend_->deviceToHost(
                    second_tokens.data(), d_tokens,
                    second_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->deviceToHost(
                    second_accepted.data(), d_accepted,
                    second_accepted.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->deviceToHost(
                    second_accept_probs.data(), d_accept_probs,
                    second_accept_probs.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        cleanup();

        EXPECT_EQ(second_tokens, first_tokens)
            << "captured fused verifier must replay deterministically";
        EXPECT_EQ(second_accepted, first_accepted)
            << "captured fused verifier acceptance bits must replay deterministically";
        for (int row = 0; row < row_count; ++row)
        {
            const auto &exp = expected[static_cast<size_t>(row)];
            EXPECT_EQ(first_tokens[static_cast<size_t>(row)], exp.token)
                << "first replay row=" << row;
            EXPECT_EQ(second_tokens[static_cast<size_t>(row)], exp.token)
                << "second replay row=" << row;
            EXPECT_EQ(first_accepted[static_cast<size_t>(row)], exp.accepted ? 1 : 0)
                << "first replay row=" << row;
            EXPECT_EQ(second_accepted[static_cast<size_t>(row)], exp.accepted ? 1 : 0)
                << "second replay row=" << row;
            EXPECT_NEAR(first_accept_probs[static_cast<size_t>(row)],
                        exp.accept_probability,
                        1e-5f)
                << "first replay row=" << row;
            EXPECT_NEAR(second_accept_probs[static_cast<size_t>(row)],
                        exp.accept_probability,
                        1e-5f)
                << "second replay row=" << row;
        }
    }

    TEST_P(GPUSamplingTest, ProcessedTargetGreedyDraftSpeculativeVerifyMatchesNoDraftReferenceAndReplay)
    {
        constexpr int vocab_size = 248320;
        constexpr int row_count = 3;
        constexpr uint64_t inverse_sample_seed = 98765;
        constexpr int first_logical_position = 29;
        constexpr uint64_t kInverseSampleDomain = 0xA0761D6478BD642FULL;
        const float neg_inf = -std::numeric_limits<float>::infinity();

        const std::array<int, row_count> draft_tokens = {13, 271, 1061};
        const std::array<float, row_count> accept_thresholds = {0.55f, 0.85f, 0.24f};

        std::vector<float> target_logits(
            static_cast<size_t>(row_count) * static_cast<size_t>(vocab_size),
            neg_inf);
        std::vector<float> target_probs(target_logits.size(), 0.0f);
        std::vector<float> inverse_rows(target_logits.size(), 0.0f);

        auto set_target_row =
            [&](int row, std::initializer_list<std::pair<int, float>> entries)
        {
            const size_t base =
                static_cast<size_t>(row) * static_cast<size_t>(vocab_size);
            float total = 0.0f;
            for (const auto &entry : entries)
                total += entry.second;
            ASSERT_NEAR(total, 1.0f, 1e-5f);
            for (const auto &entry : entries)
            {
                ASSERT_GE(entry.first, 0);
                ASSERT_LT(entry.first, vocab_size);
                target_probs[base + static_cast<size_t>(entry.first)] =
                    entry.second;
                target_logits[base + static_cast<size_t>(entry.first)] =
                    std::log(entry.second);
            }
        };

        set_target_row(0, {{13, 0.60f}, {271, 0.20f}, {1061, 0.10f}, {33075, 0.10f}});
        set_target_row(1, {{271, 0.10f}, {1061, 0.45f}, {33075, 0.35f}, {88, 0.10f}});
        set_target_row(2, {{1061, 0.25f}, {271, 0.35f}, {33075, 0.25f}, {248068, 0.15f}});

        for (int row = 0; row < row_count; ++row)
        {
            for (int token = 0; token < vocab_size; ++token)
            {
                const uint64_t offset =
                    static_cast<uint64_t>(first_logical_position + row) *
                        static_cast<uint64_t>(vocab_size) +
                    static_cast<uint64_t>(token);
                const float uniform = sampling_math::uniform01(
                    inverse_sample_seed ^ kInverseSampleDomain,
                    offset);
                inverse_rows[static_cast<size_t>(row) * vocab_size +
                             static_cast<size_t>(token)] =
                    sampling_math::inverse_exponential_from_uniform(uniform);
            }
        }

        std::array<MTPRejectionSampleRowResult, row_count> expected;
        for (int row = 0; row < row_count; ++row)
        {
            const size_t base =
                static_cast<size_t>(row) * static_cast<size_t>(vocab_size);
            expected[static_cast<size_t>(row)] =
                sampleMTPRejectionRowFromProbabilities(
                    target_probs.data() + base,
                    /*draft_probabilities=*/nullptr,
                    inverse_rows.data() + base,
                    vocab_size,
                    draft_tokens[static_cast<size_t>(row)],
                    accept_thresholds[static_cast<size_t>(row)],
                    /*no_draft_probabilities=*/true);
            ASSERT_TRUE(expected[static_cast<size_t>(row)].ok)
                << expected[static_cast<size_t>(row)].error;
        }
        ASSERT_TRUE(expected[0].accepted);
        ASSERT_FALSE(expected[1].accepted);
        ASSERT_TRUE(expected[2].accepted);

        void *d_target = backend_->allocate(target_logits.size() * sizeof(float), device_id_);
        void *d_draft_tokens = backend_->allocate(draft_tokens.size() * sizeof(int), device_id_);
        void *d_tokens = backend_->allocate(row_count * sizeof(int), device_id_);
        void *d_accepted = backend_->allocate(row_count * sizeof(int), device_id_);
        void *d_accept_probs = backend_->allocate(row_count * sizeof(float), device_id_);
        void *d_accept_thresholds = backend_->allocate(row_count * sizeof(float), device_id_);
        auto cleanup = [&]()
        {
            void *ptrs[] = {
                d_target, d_draft_tokens, d_tokens, d_accepted,
                d_accept_probs, d_accept_thresholds};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        ASSERT_NE(d_target, nullptr);
        ASSERT_NE(d_draft_tokens, nullptr);
        ASSERT_NE(d_tokens, nullptr);
        ASSERT_NE(d_accepted, nullptr);
        ASSERT_NE(d_accept_probs, nullptr);
        ASSERT_NE(d_accept_thresholds, nullptr);

        std::array<int, row_count> first_tokens{};
        std::array<int, row_count> first_accepted{};
        std::array<float, row_count> first_accept_probs{};
        std::array<int, row_count> second_tokens{};
        std::array<int, row_count> second_accepted{};
        std::array<float, row_count> second_accept_probs{};

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(backend_->hostToDevice(
                    d_target, target_logits.data(),
                    target_logits.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_draft_tokens, draft_tokens.data(),
                    draft_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                EXPECT_FALSE(
                    backend_->enqueueSpeculativeVerifyProcessedTargetDraftProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
                        d_target,
                        /*draft_probabilities_device=*/nullptr,
                        row_count,
                        vocab_size,
                        vocab_size,
                        /*draft_row_stride=*/0,
                        d_draft_tokens,
                        accept_thresholds.data(),
                        inverse_sample_seed,
                        first_logical_position,
                        device_id_,
                        nullptr,
                        d_tokens,
                        d_accepted,
                        d_accept_probs,
                        d_accept_thresholds,
                        /*no_draft_probabilities=*/true))
                    << "processed-target no-draft verifier must reject the legacy default/null stream";
                EXPECT_FALSE(
                    backend_->enqueueSpeculativeVerifyProcessedTargetDraftProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
                        d_target,
                        /*draft_probabilities_device=*/nullptr,
                        row_count,
                        vocab_size,
                        vocab_size,
                        /*draft_row_stride=*/0,
                        d_draft_tokens,
                        accept_thresholds.data(),
                        inverse_sample_seed,
                        first_logical_position,
                        device_id_,
                        stream,
                        d_tokens,
                        d_accepted,
                        d_accept_probs,
                        d_accept_thresholds))
                    << "null draft probabilities require explicit no-draft mode";

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(
                    backend_->enqueueSpeculativeVerifyProcessedTargetDraftProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
                        d_target,
                        /*draft_probabilities_device=*/nullptr,
                        row_count,
                        vocab_size,
                        vocab_size,
                        /*draft_row_stride=*/0,
                        d_draft_tokens,
                        accept_thresholds.data(),
                        inverse_sample_seed,
                        first_logical_position,
                        device_id_,
                        stream,
                        d_tokens,
                        d_accepted,
                        d_accept_probs,
                        d_accept_thresholds,
                        /*no_draft_probabilities=*/true));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());

                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
                ASSERT_TRUE(backend_->deviceToHost(
                    first_tokens.data(), d_tokens,
                    first_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->deviceToHost(
                    first_accepted.data(), d_accepted,
                    first_accepted.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->deviceToHost(
                    first_accept_probs.data(), d_accept_probs,
                    first_accept_probs.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                const std::array<int, row_count> sentinel_tokens = {-1, -1, -1};
                const std::array<int, row_count> sentinel_accepted = {-7, -7, -7};
                ASSERT_TRUE(backend_->hostToDevice(
                    d_tokens, sentinel_tokens.data(),
                    sentinel_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_accepted, sentinel_accepted.data(),
                    sentinel_accepted.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
                ASSERT_TRUE(backend_->deviceToHost(
                    second_tokens.data(), d_tokens,
                    second_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->deviceToHost(
                    second_accepted.data(), d_accepted,
                    second_accepted.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->deviceToHost(
                    second_accept_probs.data(), d_accept_probs,
                    second_accept_probs.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        cleanup();

        EXPECT_EQ(second_tokens, first_tokens)
            << "captured processed-target no-draft verifier must replay deterministically";
        EXPECT_EQ(second_accepted, first_accepted)
            << "captured processed-target no-draft acceptance bits must replay deterministically";
        for (int row = 0; row < row_count; ++row)
        {
            const auto &exp = expected[static_cast<size_t>(row)];
            EXPECT_EQ(first_tokens[static_cast<size_t>(row)], exp.token)
                << "first replay row=" << row;
            EXPECT_EQ(second_tokens[static_cast<size_t>(row)], exp.token)
                << "second replay row=" << row;
            EXPECT_EQ(first_accepted[static_cast<size_t>(row)], exp.accepted ? 1 : 0)
                << "first replay row=" << row;
            EXPECT_EQ(second_accepted[static_cast<size_t>(row)], exp.accepted ? 1 : 0)
                << "second replay row=" << row;
            EXPECT_NEAR(first_accept_probs[static_cast<size_t>(row)],
                        exp.accept_probability,
                        1e-5f)
                << "first replay row=" << row;
            EXPECT_NEAR(second_accept_probs[static_cast<size_t>(row)],
                        exp.accept_probability,
                        1e-5f)
                << "second replay row=" << row;
        }
    }

    TEST_P(GPUSamplingTest, FullProbabilitySpeculativeVerifyMatchesVLLMReferenceAndCaptures)
    {
        constexpr int vocab_size = 8;
        constexpr int row_count = 3;
        const std::array<int, row_count> draft_tokens = {0, 0, 0};
        const std::array<float, row_count> accept_thresholds = {0.9f, 0.9f, 0.4f};

        auto append_row = [](std::vector<float> &rows,
                             std::initializer_list<float> row)
        {
            rows.insert(rows.end(), row.begin(), row.end());
        };

        std::vector<float> target_rows;
        std::vector<float> draft_rows;
        std::vector<float> inverse_rows;
        append_row(target_rows, {0.80f, 0.05f, 0.05f, 0.05f,
                                 0.05f, 0.0f, 0.0f, 0.0f});
        append_row(draft_rows, {0.50f, 0.10f, 0.10f, 0.10f,
                                0.20f, 0.0f, 0.0f, 0.0f});
        append_row(inverse_rows, {1.0f, 1.0f, 1.0f, 1.0f,
                                  1.0f, 1.0f, 1.0f, 1.0f});

        append_row(target_rows, {0.10f, 0.20f, 0.60f, 0.10f,
                                 0.0f, 0.0f, 0.0f, 0.0f});
        append_row(draft_rows, {0.40f, 0.10f, 0.20f, 0.30f,
                                0.0f, 0.0f, 0.0f, 0.0f});
        append_row(inverse_rows, {100.0f, 1.0f, 0.5f, 20.0f,
                                  1.0f, 1.0f, 1.0f, 1.0f});

        append_row(target_rows, {0.20f, 0.80f, 0.0f, 0.0f,
                                 0.0f, 0.0f, 0.0f, 0.0f});
        append_row(draft_rows, {0.50f, 0.50f, 0.0f, 0.0f,
                                0.0f, 0.0f, 0.0f, 0.0f});
        append_row(inverse_rows, {1.0f, 1.0f, 1.0f, 1.0f,
                                  1.0f, 1.0f, 1.0f, 1.0f});

        std::array<MTPRejectionSampleRowResult, row_count> expected;
        for (int row = 0; row < row_count; ++row)
        {
            expected[static_cast<size_t>(row)] =
                sampleMTPRejectionRowFromProbabilities(
                    target_rows.data() + static_cast<size_t>(row) * vocab_size,
                    draft_rows.data() + static_cast<size_t>(row) * vocab_size,
                    inverse_rows.data() + static_cast<size_t>(row) * vocab_size,
                    vocab_size,
                    draft_tokens[static_cast<size_t>(row)],
                    accept_thresholds[static_cast<size_t>(row)]);
            ASSERT_TRUE(expected[static_cast<size_t>(row)].ok)
                << expected[static_cast<size_t>(row)].error;
        }
        ASSERT_TRUE(expected[0].accepted);
        ASSERT_FALSE(expected[1].accepted);
        ASSERT_TRUE(expected[2].accepted)
            << "vLLM accepts when p/q equals the uniform threshold";

        void *d_target = backend_->allocate(target_rows.size() * sizeof(float), device_id_);
        void *d_draft = backend_->allocate(draft_rows.size() * sizeof(float), device_id_);
        void *d_inverse = backend_->allocate(inverse_rows.size() * sizeof(float), device_id_);
        void *d_draft_tokens = backend_->allocate(draft_tokens.size() * sizeof(int), device_id_);
        void *d_tokens = backend_->allocate(row_count * sizeof(int), device_id_);
        void *d_accepted = backend_->allocate(row_count * sizeof(int), device_id_);
        void *d_accept_probs = backend_->allocate(row_count * sizeof(float), device_id_);
        void *d_accept_thresholds = backend_->allocate(row_count * sizeof(float), device_id_);

        auto cleanup = [&]()
        {
            void *ptrs[] = {
                d_target, d_draft, d_inverse, d_draft_tokens, d_tokens,
                d_accepted, d_accept_probs, d_accept_thresholds};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        ASSERT_NE(d_target, nullptr);
        ASSERT_NE(d_draft, nullptr);
        ASSERT_NE(d_inverse, nullptr);
        ASSERT_NE(d_draft_tokens, nullptr);
        ASSERT_NE(d_tokens, nullptr);
        ASSERT_NE(d_accepted, nullptr);
        ASSERT_NE(d_accept_probs, nullptr);
        ASSERT_NE(d_accept_thresholds, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(backend_->hostToDevice(
                    d_target, target_rows.data(),
                    target_rows.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_draft, draft_rows.data(),
                    draft_rows.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_inverse, inverse_rows.data(),
                    inverse_rows.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_draft_tokens, draft_tokens.data(),
                    draft_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                EXPECT_FALSE(
                    backend_->enqueueSpeculativeVerifyProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
                        d_target, d_draft, d_inverse, row_count, vocab_size,
                        vocab_size, vocab_size, vocab_size, d_draft_tokens,
                        accept_thresholds.data(), device_id_, nullptr,
                        d_tokens, d_accepted, d_accept_probs,
                        d_accept_thresholds))
                    << "full-probability verifier must reject the legacy default/null stream";

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(
                    backend_->enqueueSpeculativeVerifyProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
                        d_target, d_draft, d_inverse, row_count, vocab_size,
                        vocab_size, vocab_size, vocab_size, d_draft_tokens,
                        accept_thresholds.data(), device_id_, stream,
                        d_tokens, d_accepted, d_accept_probs,
                        d_accept_thresholds));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::array<int, row_count> tokens{};
        std::array<int, row_count> accepted{};
        std::array<float, row_count> accept_probs{};
        ASSERT_TRUE(backend_->deviceToHost(tokens.data(), d_tokens,
                                           tokens.size() * sizeof(int), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(accepted.data(), d_accepted,
                                           accepted.size() * sizeof(int), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(accept_probs.data(), d_accept_probs,
                                           accept_probs.size() * sizeof(float), device_id_));

        cleanup();

        for (int row = 0; row < row_count; ++row)
        {
            const auto &exp = expected[static_cast<size_t>(row)];
            EXPECT_EQ(tokens[static_cast<size_t>(row)], exp.token)
                << "row " << row;
            EXPECT_EQ(accepted[static_cast<size_t>(row)], exp.accepted ? 1 : 0)
                << "row " << row;
            EXPECT_NEAR(accept_probs[static_cast<size_t>(row)],
                        exp.accept_probability,
                        1e-6f)
                << "row " << row;
        }
    }

    TEST_P(GPUSamplingTest, FullProbabilitySpeculativeVerifySupportsNoDraftProbabilitiesMode)
    {
        constexpr int vocab_size = 4;
        constexpr int row_count = 1;
        const std::array<float, vocab_size> target = {0.50f, 0.20f, 0.20f, 0.10f};
        const std::array<float, vocab_size> inverse_samples = {100.0f, 1.0f, 5.0f, 1.0f};
        const std::array<int, row_count> draft_tokens = {0};
        const std::array<float, row_count> accept_thresholds = {0.99f};

        MTPRejectionSampleRowResult expected =
            sampleMTPRejectionRowFromProbabilities(
                target.data(),
                /*draft_probabilities=*/nullptr,
                inverse_samples.data(),
                vocab_size,
                draft_tokens[0],
                accept_thresholds[0],
                /*no_draft_probabilities=*/true);
        ASSERT_TRUE(expected.ok) << expected.error;
        ASSERT_FALSE(expected.accepted);
        ASSERT_EQ(expected.token, 2);

        void *d_target = backend_->allocate(target.size() * sizeof(float), device_id_);
        void *d_inverse = backend_->allocate(inverse_samples.size() * sizeof(float), device_id_);
        void *d_draft_tokens = backend_->allocate(draft_tokens.size() * sizeof(int), device_id_);
        void *d_tokens = backend_->allocate(row_count * sizeof(int), device_id_);
        void *d_accepted = backend_->allocate(row_count * sizeof(int), device_id_);

        auto cleanup = [&]()
        {
            void *ptrs[] = {d_target, d_inverse, d_draft_tokens, d_tokens, d_accepted};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        ASSERT_NE(d_target, nullptr);
        ASSERT_NE(d_inverse, nullptr);
        ASSERT_NE(d_draft_tokens, nullptr);
        ASSERT_NE(d_tokens, nullptr);
        ASSERT_NE(d_accepted, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(backend_->hostToDevice(
                    d_target, target.data(), target.size() * sizeof(float),
                    device_id_, stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_inverse, inverse_samples.data(),
                    inverse_samples.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_draft_tokens, draft_tokens.data(),
                    draft_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(
                    backend_->enqueueSpeculativeVerifyProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
                        d_target,
                        /*draft_probabilities_device=*/nullptr,
                        d_inverse,
                        row_count,
                        vocab_size,
                        vocab_size,
                        /*draft_row_stride=*/0,
                        vocab_size,
                        d_draft_tokens,
                        accept_thresholds.data(),
                        device_id_,
                        stream,
                        d_tokens,
                        d_accepted,
                        /*out_accept_probability_device=*/nullptr,
                        /*out_accept_threshold_device=*/nullptr,
                        /*no_draft_probabilities=*/true));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::array<int, row_count> tokens{};
        std::array<int, row_count> accepted{};
        ASSERT_TRUE(backend_->deviceToHost(tokens.data(), d_tokens,
                                           tokens.size() * sizeof(int), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(accepted.data(), d_accepted,
                                           accepted.size() * sizeof(int), device_id_));
        cleanup();

        EXPECT_EQ(tokens[0], expected.token);
        EXPECT_EQ(accepted[0], 0);
    }

    TEST_P(GPUSamplingTest, ProcessedLogitVerifierSamplesBonusAndSummarizes)
    {
        constexpr int vocab_size = 16;
        constexpr int row_count = 2;
        const std::array<int, row_count + 1> request_tokens = {10, 11, 12};
        const std::array<int, row_count> draft_tokens = {11, 12};
        const std::array<float, row_count> accept_thresholds = {0.0f, 0.0f};
        const std::array<float, row_count> residual_thresholds = {0.0f, 0.0f};
        constexpr float bonus_threshold = 0.5f;

        auto logits_from_hot_token = [](int hot_token)
        {
            std::vector<float> logits(vocab_size, -std::numeric_limits<float>::infinity());
            logits[static_cast<size_t>(hot_token)] = 4.0f;
            logits[static_cast<size_t>((hot_token + 1) % vocab_size)] = 2.0f;
            return logits;
        };
        auto append_row = [](std::vector<float> &rows,
                             const std::vector<float> &row)
        {
            rows.insert(rows.end(), row.begin(), row.end());
        };

        std::vector<float> target_rows;
        std::vector<float> draft_rows;
        append_row(target_rows, logits_from_hot_token(11));
        append_row(target_rows, logits_from_hot_token(12));
        append_row(draft_rows, logits_from_hot_token(11));
        append_row(draft_rows, logits_from_hot_token(12));
        const std::vector<float> bonus_row = logits_from_hot_token(7);
        std::array<float, row_count> draft_token_probabilities{};
        for (int row = 0; row < row_count; ++row)
        {
            const float *draft_row =
                draft_rows.data() + static_cast<size_t>(row) * vocab_size;
            const MTPFullLogitRowStats stats =
                computeMTPFullLogitRowStats(draft_row, vocab_size);
            ASSERT_TRUE(stats.ok) << stats.error;
            draft_token_probabilities[static_cast<size_t>(row)] =
                probabilityFromMTPFullLogits(
                    draft_row,
                    vocab_size,
                    stats,
                    draft_tokens[static_cast<size_t>(row)]);
            ASSERT_GT(draft_token_probabilities[static_cast<size_t>(row)], 0.0f);
        }

        MTPDecodeCatchupGreedyRequest request;
        request.draft_tokens.assign(request_tokens.begin(), request_tokens.end());
        MTPRejectionBatchOutcome expected =
            summarizeAllPositionMTPRejectionBatchFromProcessedLogits(
                request,
                target_rows.data(),
                draft_rows.data(),
                row_count,
                vocab_size,
                vocab_size,
                vocab_size,
                std::vector<float>(accept_thresholds.begin(), accept_thresholds.end()),
                std::vector<float>(residual_thresholds.begin(), residual_thresholds.end()),
                bonus_row.data(),
                bonus_threshold);
        ASSERT_TRUE(expected.ok) << expected.error;
        ASSERT_TRUE(expected.all_speculative_accepted);

        void *d_target = backend_->allocate(target_rows.size() * sizeof(float), device_id_);
        void *d_draft = backend_->allocate(draft_rows.size() * sizeof(float), device_id_);
        void *d_draft_tokens = backend_->allocate(draft_tokens.size() * sizeof(int), device_id_);
        void *d_draft_token_probs = backend_->allocate(
            draft_token_probabilities.size() * sizeof(float),
            device_id_);
        void *d_bonus = backend_->allocate(bonus_row.size() * sizeof(float), device_id_);
        void *d_verify_tokens = backend_->allocate(row_count * sizeof(int), device_id_);
        void *d_verify_accepted = backend_->allocate(row_count * sizeof(int), device_id_);
        void *d_bonus_token = backend_->allocate(sizeof(int), device_id_);
        void *d_output_tokens = backend_->allocate(
            sampling_math::kSpeculativeBatchMaxOutputTokens * sizeof(int),
            device_id_);
        void *d_output_meta = backend_->allocate(
            sampling_math::kSpeculativeBatchMetaCount * sizeof(int),
            device_id_);

        auto cleanup = [&]()
        {
            void *ptrs[] = {
                d_target, d_draft, d_draft_tokens, d_draft_token_probs, d_bonus,
                d_verify_tokens, d_verify_accepted, d_bonus_token,
                d_output_tokens, d_output_meta};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        ASSERT_NE(d_target, nullptr);
        ASSERT_NE(d_draft, nullptr);
        ASSERT_NE(d_draft_tokens, nullptr);
        ASSERT_NE(d_draft_token_probs, nullptr);
        ASSERT_NE(d_bonus, nullptr);
        ASSERT_NE(d_verify_tokens, nullptr);
        ASSERT_NE(d_verify_accepted, nullptr);
        ASSERT_NE(d_bonus_token, nullptr);
        ASSERT_NE(d_output_tokens, nullptr);
        ASSERT_NE(d_output_meta, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                ASSERT_TRUE(backend_->hostToDevice(
                    d_target, target_rows.data(),
                    target_rows.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_draft, draft_rows.data(),
                    draft_rows.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_draft_tokens, draft_tokens.data(),
                    draft_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_draft_token_probs, draft_token_probabilities.data(),
                    draft_token_probabilities.size() * sizeof(float),
                    device_id_, stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_bonus, bonus_row.data(),
                    bonus_row.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                EXPECT_FALSE(
                    backend_->enqueueSampleProcessedLogitsF32DeviceIfSpeculativeBatchNeedsBonus(
                        d_bonus, vocab_size, vocab_size, bonus_threshold,
                        d_verify_tokens, d_verify_accepted, row_count,
                        request_tokens[0],
                        /*first_token_device=*/nullptr,
                        /*stop_tokens_host=*/nullptr,
                        /*stop_token_count=*/0,
                        device_id_, nullptr, d_bonus_token))
                    << "lazy processed-logit bonus sampler must reject the legacy default/null stream";

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(
                    backend_->enqueueSpeculativeVerifyProcessedLogitsF32DeviceThresholdsBatchDeviceTokens(
                        d_target, d_draft, row_count, vocab_size, vocab_size,
                        vocab_size, d_draft_tokens, accept_thresholds.data(),
                        residual_thresholds.data(), device_id_, stream,
                        d_verify_tokens, d_verify_accepted,
                        nullptr, nullptr, d_draft_token_probs));
                ASSERT_TRUE(
                    backend_->enqueueSampleProcessedLogitsF32DeviceIfSpeculativeBatchNeedsBonus(
                        d_bonus, vocab_size, vocab_size, bonus_threshold,
                        d_verify_tokens, d_verify_accepted, row_count,
                        request_tokens[0],
                        /*first_token_device=*/nullptr,
                        /*stop_tokens_host=*/nullptr,
                        /*stop_token_count=*/0,
                        device_id_, stream, d_bonus_token));
                ASSERT_TRUE(backend_->enqueueSummarizeSpeculativeVerifyBatch(
                    d_verify_tokens,
                    d_verify_accepted,
                    row_count,
                    request_tokens[0],
                    /*stop_tokens_host=*/nullptr,
                    /*stop_token_count=*/0,
                    d_bonus_token,
                    /*has_bonus_token=*/true,
                    device_id_,
                    stream,
                    d_output_tokens,
                    d_output_meta));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::array<int, sampling_math::kSpeculativeBatchMaxOutputTokens> output_tokens{};
        std::array<int, sampling_math::kSpeculativeBatchMetaCount> output_meta{};
        int bonus_token = -1;
        ASSERT_TRUE(backend_->deviceToHost(
            &bonus_token, d_bonus_token, sizeof(int), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(
            output_tokens.data(), d_output_tokens,
            output_tokens.size() * sizeof(int), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(
            output_meta.data(), d_output_meta,
            output_meta.size() * sizeof(int), device_id_));

        cleanup();

        ASSERT_EQ(output_meta[sampling_math::kSpecBatchMetaOk], 1);
        EXPECT_EQ(bonus_token, expected.ready_token);
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaOutputCount],
                  static_cast<int>(expected.output_tokens.size()));
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaAcceptedSpeculativePrefix],
                  expected.accepted_speculative_prefix);
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaTargetVerifierStateCommitCount],
                  expected.target_verifier_state_commit_count);
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaReadyToken],
                  expected.ready_token);
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaSampledTerminal],
                  expected.sampled_terminal ? 1 : 0);

        for (size_t i = 0; i < expected.output_tokens.size(); ++i)
        {
            EXPECT_EQ(output_tokens[i], expected.output_tokens[i])
                << "output token " << i;
        }
    }

    TEST_P(GPUSamplingTest, LazyProcessedBonusSamplerSkipsRejectedBatchAndCaptures)
    {
        constexpr int vocab_size = 16;
        constexpr int row_count = 2;
        constexpr float bonus_threshold = 0.5f;
        const std::array<int, row_count> verify_tokens = {11, 42};
        const std::array<int, row_count> verify_accepted = {1, 0};
        const std::array<int, sampling_math::kSpeculativeBatchMaxStopTokens>
            stop_tokens = {-1, -1, -1, -1, -1, -1, -1, -1};
        const int first_token = 10;

        std::vector<float> bonus_row(vocab_size, -std::numeric_limits<float>::infinity());
        bonus_row[7] = 4.0f;
        bonus_row[8] = 2.0f;

        void *d_bonus = backend_->allocate(bonus_row.size() * sizeof(float), device_id_);
        void *d_verify_tokens = backend_->allocate(verify_tokens.size() * sizeof(int), device_id_);
        void *d_verify_accepted = backend_->allocate(verify_accepted.size() * sizeof(int), device_id_);
        void *d_bonus_token = backend_->allocate(sizeof(int), device_id_);
        void *d_output_tokens = backend_->allocate(
            sampling_math::kSpeculativeBatchMaxOutputTokens * sizeof(int),
            device_id_);
        void *d_output_meta = backend_->allocate(
            sampling_math::kSpeculativeBatchMetaCount * sizeof(int),
            device_id_);

        auto cleanup = [&]()
        {
            void *ptrs[] = {
                d_bonus,
                d_verify_tokens,
                d_verify_accepted,
                d_bonus_token,
                d_output_tokens,
                d_output_meta};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        ASSERT_NE(d_bonus, nullptr);
        ASSERT_NE(d_verify_tokens, nullptr);
        ASSERT_NE(d_verify_accepted, nullptr);
        ASSERT_NE(d_bonus_token, nullptr);
        ASSERT_NE(d_output_tokens, nullptr);
        ASSERT_NE(d_output_meta, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);
                int stale_bonus_token = 12345;
                ASSERT_TRUE(backend_->hostToDevice(
                    d_bonus, bonus_row.data(),
                    bonus_row.size() * sizeof(float), device_id_, stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_verify_tokens, verify_tokens.data(),
                    verify_tokens.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_verify_accepted, verify_accepted.data(),
                    verify_accepted.size() * sizeof(int), device_id_, stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_bonus_token, &stale_bonus_token, sizeof(int),
                    device_id_, stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(
                    backend_->enqueueSampleProcessedLogitsF32DeviceIfSpeculativeBatchNeedsBonus(
                        d_bonus, vocab_size, vocab_size, bonus_threshold,
                        d_verify_tokens, d_verify_accepted, row_count,
                        first_token,
                        /*first_token_device=*/nullptr,
                        stop_tokens.data(),
                        /*stop_token_count=*/0,
                        device_id_, stream, d_bonus_token));
                ASSERT_TRUE(backend_->enqueueSummarizeSpeculativeVerifyBatch(
                    d_verify_tokens,
                    d_verify_accepted,
                    row_count,
                    first_token,
                    /*stop_tokens_host=*/nullptr,
                    /*stop_token_count=*/0,
                    d_bonus_token,
                    /*has_bonus_token=*/true,
                    device_id_,
                    stream,
                    d_output_tokens,
                    d_output_meta));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        int bonus_token = 0;
        std::array<int, sampling_math::kSpeculativeBatchMaxOutputTokens> output_tokens{};
        std::array<int, sampling_math::kSpeculativeBatchMetaCount> output_meta{};
        ASSERT_TRUE(backend_->deviceToHost(
            &bonus_token, d_bonus_token, sizeof(int), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(
            output_tokens.data(), d_output_tokens,
            output_tokens.size() * sizeof(int), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(
            output_meta.data(), d_output_meta,
            output_meta.size() * sizeof(int), device_id_));

        cleanup();

        EXPECT_EQ(bonus_token, -1);
        ASSERT_EQ(output_meta[sampling_math::kSpecBatchMetaOk], 1);
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaOutputCount], 3);
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaAcceptedSpeculativePrefix], 1);
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaReadyToken], -1);
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaSampledTerminal], 0);
        EXPECT_EQ(output_tokens[0], first_token);
        EXPECT_EQ(output_tokens[1], verify_tokens[0]);
        EXPECT_EQ(output_tokens[2], verify_tokens[1]);
    }

    TEST_P(GPUSamplingTest, CompactOneHotDraftVLLMSpeculativeVerifyMatchesReferenceAndCaptures)
    {
        constexpr int top_k = 4;
        constexpr int row_count = 2;
        constexpr int distribution_stride = top_k;

        const std::vector<ExpectedDistributionEntry> target_distribution = {
            {5, 0.20f},
            {7, 0.30f},
            {11, 0.10f},
            {13, 0.40f}};
        const std::vector<int> target_ids = {
            5, 7, 11, 13,
            5, 7, 11, 13};
        const std::vector<float> target_probs = {
            0.20f, 0.30f, 0.10f, 0.40f,
            0.20f, 0.30f, 0.10f, 0.40f};
        constexpr uint64_t inverse_sample_seed = 98765;
        constexpr int inverse_sample_first_logical_position = 0;
        constexpr int inverse_sample_vocab_size = 32;
        const int draft_tokens[row_count] = {7, 7};
        const float accept_thresholds[row_count] = {
            sampling_math::mtp_spec_threshold_from_seed(
                inverse_sample_seed,
                inverse_sample_first_logical_position,
                1 /* MTPSpecStochasticDrawPurpose::Accept */),
            sampling_math::mtp_spec_threshold_from_seed(
                inverse_sample_seed,
                inverse_sample_first_logical_position + 1,
                1 /* MTPSpecStochasticDrawPurpose::Accept */)};
        const float residual_thresholds[row_count] = {
            sampling_math::mtp_spec_threshold_from_seed(
                inverse_sample_seed,
                inverse_sample_first_logical_position,
                2 /* MTPSpecStochasticDrawPurpose::Residual */),
            sampling_math::mtp_spec_threshold_from_seed(
                inverse_sample_seed,
                inverse_sample_first_logical_position + 1,
                2 /* MTPSpecStochasticDrawPurpose::Residual */)};

        const auto expected_accept =
            expectedSpeculativeVerifyOneHotDraftVLLMWithThresholds(
                target_distribution,
                draft_tokens[0],
                accept_thresholds[0],
                inverse_sample_seed,
                inverse_sample_first_logical_position,
                inverse_sample_vocab_size);
        const auto expected_reject =
            expectedSpeculativeVerifyOneHotDraftVLLMWithThresholds(
                target_distribution,
                draft_tokens[1],
                accept_thresholds[1],
                inverse_sample_seed,
                inverse_sample_first_logical_position + 1,
                inverse_sample_vocab_size);
        ASSERT_EQ(expected_accept.accepted, 1);
        ASSERT_EQ(expected_accept.token_id, 7);
        ASSERT_EQ(expected_reject.accepted, 0);

        void *d_target_ids = nullptr;
        void *d_target_probs = nullptr;
        void *d_draft_tokens = nullptr;
        void *d_out_tokens = nullptr;
        void *d_out_accepted = nullptr;
        void *d_out_accept_probability = nullptr;
        void *d_out_accept_threshold = nullptr;
        void *d_seeded_out_tokens = nullptr;
        void *d_seeded_out_accepted = nullptr;
        void *d_seeded_out_accept_probability = nullptr;
        void *d_seeded_out_accept_threshold = nullptr;

        auto cleanup = [&]()
        {
            void *ptrs[] = {
                d_target_ids,
                d_target_probs,
                d_draft_tokens,
                d_out_tokens,
                d_out_accepted,
                d_out_accept_probability,
                d_out_accept_threshold,
                d_seeded_out_tokens,
                d_seeded_out_accepted,
                d_seeded_out_accept_probability,
                d_seeded_out_accept_threshold};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        d_target_ids = backend_->allocate(target_ids.size() * sizeof(int), device_id_);
        d_target_probs = backend_->allocate(target_probs.size() * sizeof(float), device_id_);
        d_draft_tokens = backend_->allocate(sizeof(draft_tokens), device_id_);
        d_out_tokens = backend_->allocate(row_count * sizeof(int), device_id_);
        d_out_accepted = backend_->allocate(row_count * sizeof(int), device_id_);
        d_out_accept_probability = backend_->allocate(row_count * sizeof(float), device_id_);
        d_out_accept_threshold = backend_->allocate(row_count * sizeof(float), device_id_);
        d_seeded_out_tokens = backend_->allocate(row_count * sizeof(int), device_id_);
        d_seeded_out_accepted = backend_->allocate(row_count * sizeof(int), device_id_);
        d_seeded_out_accept_probability = backend_->allocate(row_count * sizeof(float), device_id_);
        d_seeded_out_accept_threshold = backend_->allocate(row_count * sizeof(float), device_id_);

        ASSERT_NE(d_target_ids, nullptr);
        ASSERT_NE(d_target_probs, nullptr);
        ASSERT_NE(d_draft_tokens, nullptr);
        ASSERT_NE(d_out_tokens, nullptr);
        ASSERT_NE(d_out_accepted, nullptr);
        ASSERT_NE(d_out_accept_probability, nullptr);
        ASSERT_NE(d_out_accept_threshold, nullptr);
        ASSERT_NE(d_seeded_out_tokens, nullptr);
        ASSERT_NE(d_seeded_out_accepted, nullptr);
        ASSERT_NE(d_seeded_out_accept_probability, nullptr);
        ASSERT_NE(d_seeded_out_accept_threshold, nullptr);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);

                ASSERT_TRUE(backend_->hostToDevice(
                    d_target_ids,
                    target_ids.data(),
                    target_ids.size() * sizeof(int),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_target_probs,
                    target_probs.data(),
                    target_probs.size() * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_draft_tokens,
                    draft_tokens,
                    sizeof(draft_tokens),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                EXPECT_FALSE(backend_->enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatchDeviceTokens(
                    d_target_ids,
                    d_target_probs,
                    nullptr,
                    nullptr,
                    top_k,
                    distribution_stride,
                    d_draft_tokens,
                    accept_thresholds,
                    residual_thresholds,
                    row_count,
                    device_id_,
                    nullptr,
                    d_out_tokens,
                    d_out_accepted,
                    d_out_accept_probability,
                    d_out_accept_threshold,
                    /*draft_token_probabilities_device=*/nullptr,
                    inverse_sample_seed,
                    inverse_sample_first_logical_position,
                    inverse_sample_vocab_size))
                    << "one-hot compact verifier must still reject the legacy default/null stream";
                EXPECT_FALSE(backend_->enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatchDeviceTokens(
                    d_target_ids,
                    d_target_probs,
                    nullptr,
                    d_target_probs,
                    top_k,
                    distribution_stride,
                    d_draft_tokens,
                    accept_thresholds,
                    residual_thresholds,
                    row_count,
                    device_id_,
                    stream,
                    d_out_tokens,
                    d_out_accepted,
                    d_out_accept_probability,
                    d_out_accept_threshold,
                    /*draft_token_probabilities_device=*/nullptr,
                    inverse_sample_seed,
                    inverse_sample_first_logical_position,
                    inverse_sample_vocab_size))
                    << "passing only one null draft-distribution pointer is invalid";

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatchDeviceTokens(
                    d_target_ids,
                    d_target_probs,
                    nullptr,
                    nullptr,
                    top_k,
                    distribution_stride,
                    d_draft_tokens,
                    accept_thresholds,
                    residual_thresholds,
                    row_count,
                    device_id_,
                    stream,
                    d_out_tokens,
                    d_out_accepted,
                    d_out_accept_probability,
                    d_out_accept_threshold,
                    /*draft_token_probabilities_device=*/nullptr,
                    inverse_sample_seed,
                    inverse_sample_first_logical_position,
                    inverse_sample_vocab_size));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());

                auto seeded_capture = ctx.createGraphCapture(stream);
                ASSERT_NE(seeded_capture, nullptr);
                ASSERT_TRUE(seeded_capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatchDeviceTokens(
                    d_target_ids,
                    d_target_probs,
                    nullptr,
                    nullptr,
                    top_k,
                    distribution_stride,
                    d_draft_tokens,
                    /*accept_thresholds_host=*/nullptr,
                    /*residual_thresholds_host=*/nullptr,
                    row_count,
                    device_id_,
                    stream,
                    d_seeded_out_tokens,
                    d_seeded_out_accepted,
                    d_seeded_out_accept_probability,
                    d_seeded_out_accept_threshold,
                    /*draft_token_probabilities_device=*/nullptr,
                    inverse_sample_seed,
                    inverse_sample_first_logical_position,
                    inverse_sample_vocab_size));
                ASSERT_TRUE(seeded_capture->endCapture());
                ASSERT_TRUE(seeded_capture->instantiate());
                ASSERT_TRUE(seeded_capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::vector<int> out_tokens(row_count, -1);
        std::vector<int> out_accepted(row_count, -1);
        std::vector<float> out_accept_probabilities(row_count, -1.0f);
        std::vector<float> out_accept_thresholds(row_count, -1.0f);
        std::vector<int> seeded_out_tokens(row_count, -1);
        std::vector<int> seeded_out_accepted(row_count, -1);
        std::vector<float> seeded_out_accept_probabilities(row_count, -1.0f);
        std::vector<float> seeded_out_accept_thresholds(row_count, -1.0f);

        ASSERT_TRUE(backend_->deviceToHost(
            out_tokens.data(),
            d_out_tokens,
            row_count * sizeof(int),
            device_id_));
        ASSERT_TRUE(backend_->deviceToHost(
            out_accepted.data(),
            d_out_accepted,
            row_count * sizeof(int),
            device_id_));
        ASSERT_TRUE(backend_->deviceToHost(
            out_accept_probabilities.data(),
            d_out_accept_probability,
            row_count * sizeof(float),
            device_id_));
        ASSERT_TRUE(backend_->deviceToHost(
            out_accept_thresholds.data(),
            d_out_accept_threshold,
            row_count * sizeof(float),
            device_id_));
        ASSERT_TRUE(backend_->deviceToHost(
            seeded_out_tokens.data(),
            d_seeded_out_tokens,
            row_count * sizeof(int),
            device_id_));
        ASSERT_TRUE(backend_->deviceToHost(
            seeded_out_accepted.data(),
            d_seeded_out_accepted,
            row_count * sizeof(int),
            device_id_));
        ASSERT_TRUE(backend_->deviceToHost(
            seeded_out_accept_probabilities.data(),
            d_seeded_out_accept_probability,
            row_count * sizeof(float),
            device_id_));
        ASSERT_TRUE(backend_->deviceToHost(
            seeded_out_accept_thresholds.data(),
            d_seeded_out_accept_threshold,
            row_count * sizeof(float),
            device_id_));

        cleanup();

        EXPECT_EQ(out_tokens[0], expected_accept.token_id);
        EXPECT_EQ(out_accepted[0], expected_accept.accepted);
        EXPECT_NEAR(out_accept_probabilities[0], expected_accept.accept_probability, 1e-6f);
        EXPECT_NEAR(out_accept_thresholds[0], expected_accept.accept_threshold, 1e-6f);

        EXPECT_EQ(out_tokens[1], expected_reject.token_id);
        EXPECT_EQ(out_accepted[1], expected_reject.accepted);
        EXPECT_NEAR(out_accept_probabilities[1], expected_reject.accept_probability, 1e-6f);
        EXPECT_NEAR(out_accept_thresholds[1], expected_reject.accept_threshold, 1e-6f);

        EXPECT_EQ(seeded_out_tokens, out_tokens);
        EXPECT_EQ(seeded_out_accepted, out_accepted);
        EXPECT_NEAR(seeded_out_accept_probabilities[0],
                    out_accept_probabilities[0],
                    1e-6f);
        EXPECT_NEAR(seeded_out_accept_probabilities[1],
                    out_accept_probabilities[1],
                    1e-6f);
        EXPECT_NEAR(seeded_out_accept_thresholds[0],
                    out_accept_thresholds[0],
                    1e-6f);
        EXPECT_NEAR(seeded_out_accept_thresholds[1],
                    out_accept_thresholds[1],
                    1e-6f);
    }

    TEST_P(GPUSamplingTest, SpeculativeVerifyDistributionsAreGraphCapturable)
    {
        const std::vector<float> target_logits = {0.1f, 3.2f, 2.0f, 1.2f,
                                                  4.5f, 0.5f, 2.6f, 3.7f};
        const std::vector<float> draft_accept_logits = {0.3f, 3.8f, 1.9f, 0.7f,
                                                        2.6f, 0.1f, 2.1f, 3.0f};
        const std::vector<float> draft_reject_logits = {0.2f, 5.2f, 1.8f, 0.4f,
                                                        2.0f, 0.3f, 2.4f, 3.3f};
        constexpr int top_k = 4;
        constexpr float top_p = 0.95f;
        constexpr float temperature = 0.7f;
        constexpr uint64_t accept_seed_accept_case = 1234;
        constexpr uint64_t accept_offset_accept_case = 7;
        constexpr uint64_t accept_seed_reject_case = 1;
        constexpr uint64_t accept_offset_reject_case = 0;
        constexpr uint64_t residual_seed = 999;
        constexpr uint64_t residual_offset = 11;
        constexpr int accept_draft_token = 7;
        constexpr int reject_draft_token = 1;

        const auto expected_target =
            expectedTopKTopPDistribution(target_logits, top_k, top_p, temperature);
        const auto expected_draft_accept =
            expectedTopKTopPDistribution(draft_accept_logits, top_k, top_p, temperature);
        const auto expected_draft_reject =
            expectedTopKTopPDistribution(draft_reject_logits, top_k, top_p, temperature);
        const auto expected_accept = expectedSpeculativeVerifyDistribution(
            expected_target,
            expected_draft_accept,
            accept_draft_token,
            accept_seed_accept_case,
            accept_offset_accept_case,
            residual_seed,
            residual_offset);
        const auto expected_reject = expectedSpeculativeVerifyDistribution(
            expected_target,
            expected_draft_reject,
            reject_draft_token,
            accept_seed_reject_case,
            accept_offset_reject_case,
            residual_seed,
            residual_offset);
        ASSERT_EQ(expected_accept.accepted, 1);
        ASSERT_EQ(expected_reject.accepted, 0);

        void *d_target_logits = nullptr;
        void *d_draft_accept_logits = nullptr;
        void *d_draft_reject_logits = nullptr;
        void *d_target_ids = nullptr;
        void *d_target_probs = nullptr;
        void *d_draft_accept_ids = nullptr;
        void *d_draft_accept_probs = nullptr;
        void *d_draft_reject_ids = nullptr;
        void *d_draft_reject_probs = nullptr;
        void *d_accept_token = nullptr;
        void *d_accept_flag = nullptr;
        void *d_accept_probability = nullptr;
        void *d_accept_threshold = nullptr;
        void *d_reject_token = nullptr;
        void *d_reject_flag = nullptr;
        void *d_reject_probability = nullptr;
        void *d_reject_threshold = nullptr;
        void *d_threshold_sample_token = nullptr;
        void *d_threshold_verify_token = nullptr;
        void *d_threshold_verify_flag = nullptr;
        void *d_threshold_verify_probability = nullptr;
        void *d_threshold_verify_threshold = nullptr;
        void *d_batch_target_ids = nullptr;
        void *d_batch_target_probs = nullptr;
        void *d_batch_draft_ids = nullptr;
        void *d_batch_draft_probs = nullptr;
        void *d_batch_verify_tokens = nullptr;
        void *d_batch_accept_flags = nullptr;
        void *d_batch_accept_probabilities = nullptr;
        void *d_batch_accept_thresholds = nullptr;
        void *d_batch_sampled_draft_tokens = nullptr;
        void *d_batch_sampled_draft_probabilities = nullptr;
        void *d_batch_device_token_verify_tokens = nullptr;
        void *d_batch_device_token_accept_flags = nullptr;
        void *d_batch_device_token_accept_probabilities = nullptr;
        void *d_batch_device_token_accept_thresholds = nullptr;

        auto cleanup = [&]()
        {
            void *ptrs[] = {
                d_target_logits,
                d_draft_accept_logits,
                d_draft_reject_logits,
                d_target_ids,
                d_target_probs,
                d_draft_accept_ids,
                d_draft_accept_probs,
                d_draft_reject_ids,
                d_draft_reject_probs,
                d_accept_token,
                d_accept_flag,
                d_accept_probability,
                d_accept_threshold,
                d_reject_token,
                d_reject_flag,
                d_reject_probability,
                d_reject_threshold,
                d_threshold_sample_token,
                d_threshold_verify_token,
                d_threshold_verify_flag,
                d_threshold_verify_probability,
                d_threshold_verify_threshold,
                d_batch_target_ids,
                d_batch_target_probs,
                d_batch_draft_ids,
                d_batch_draft_probs,
                d_batch_verify_tokens,
                d_batch_accept_flags,
                d_batch_accept_probabilities,
                d_batch_accept_thresholds,
                d_batch_sampled_draft_tokens,
                d_batch_sampled_draft_probabilities,
                d_batch_device_token_verify_tokens,
                d_batch_device_token_accept_flags,
                d_batch_device_token_accept_probabilities,
                d_batch_device_token_accept_thresholds};
            for (void *ptr : ptrs)
            {
                if (ptr)
                    backend_->free(ptr, device_id_);
            }
        };

        d_target_logits = backend_->allocate(target_logits.size() * sizeof(float), device_id_);
        d_draft_accept_logits = backend_->allocate(draft_accept_logits.size() * sizeof(float), device_id_);
        d_draft_reject_logits = backend_->allocate(draft_reject_logits.size() * sizeof(float), device_id_);
        d_target_ids = backend_->allocate(top_k * sizeof(int), device_id_);
        d_target_probs = backend_->allocate(top_k * sizeof(float), device_id_);
        d_draft_accept_ids = backend_->allocate(top_k * sizeof(int), device_id_);
        d_draft_accept_probs = backend_->allocate(top_k * sizeof(float), device_id_);
        d_draft_reject_ids = backend_->allocate(top_k * sizeof(int), device_id_);
        d_draft_reject_probs = backend_->allocate(top_k * sizeof(float), device_id_);
        d_accept_token = backend_->allocate(sizeof(int), device_id_);
        d_accept_flag = backend_->allocate(sizeof(int), device_id_);
        d_accept_probability = backend_->allocate(sizeof(float), device_id_);
        d_accept_threshold = backend_->allocate(sizeof(float), device_id_);
        d_reject_token = backend_->allocate(sizeof(int), device_id_);
        d_reject_flag = backend_->allocate(sizeof(int), device_id_);
        d_reject_probability = backend_->allocate(sizeof(float), device_id_);
        d_reject_threshold = backend_->allocate(sizeof(float), device_id_);
        d_threshold_sample_token = backend_->allocate(sizeof(int), device_id_);
        d_threshold_verify_token = backend_->allocate(sizeof(int), device_id_);
        d_threshold_verify_flag = backend_->allocate(sizeof(int), device_id_);
        d_threshold_verify_probability = backend_->allocate(sizeof(float), device_id_);
        d_threshold_verify_threshold = backend_->allocate(sizeof(float), device_id_);
        d_batch_target_ids = backend_->allocate(2 * top_k * sizeof(int), device_id_);
        d_batch_target_probs = backend_->allocate(2 * top_k * sizeof(float), device_id_);
        d_batch_draft_ids = backend_->allocate(2 * top_k * sizeof(int), device_id_);
        d_batch_draft_probs = backend_->allocate(2 * top_k * sizeof(float), device_id_);
        d_batch_verify_tokens = backend_->allocate(2 * sizeof(int), device_id_);
        d_batch_accept_flags = backend_->allocate(2 * sizeof(int), device_id_);
        d_batch_accept_probabilities = backend_->allocate(2 * sizeof(float), device_id_);
        d_batch_accept_thresholds = backend_->allocate(2 * sizeof(float), device_id_);
        d_batch_sampled_draft_tokens = backend_->allocate(2 * sizeof(int), device_id_);
        d_batch_sampled_draft_probabilities = backend_->allocate(2 * sizeof(float), device_id_);
        d_batch_device_token_verify_tokens = backend_->allocate(2 * sizeof(int), device_id_);
        d_batch_device_token_accept_flags = backend_->allocate(2 * sizeof(int), device_id_);
        d_batch_device_token_accept_probabilities = backend_->allocate(2 * sizeof(float), device_id_);
        d_batch_device_token_accept_thresholds = backend_->allocate(2 * sizeof(float), device_id_);

        ASSERT_NE(d_target_logits, nullptr);
        ASSERT_NE(d_draft_accept_logits, nullptr);
        ASSERT_NE(d_draft_reject_logits, nullptr);
        ASSERT_NE(d_target_ids, nullptr);
        ASSERT_NE(d_target_probs, nullptr);
        ASSERT_NE(d_draft_accept_ids, nullptr);
        ASSERT_NE(d_draft_accept_probs, nullptr);
        ASSERT_NE(d_draft_reject_ids, nullptr);
        ASSERT_NE(d_draft_reject_probs, nullptr);
        ASSERT_NE(d_accept_token, nullptr);
        ASSERT_NE(d_accept_flag, nullptr);
        ASSERT_NE(d_accept_probability, nullptr);
        ASSERT_NE(d_accept_threshold, nullptr);
        ASSERT_NE(d_reject_token, nullptr);
        ASSERT_NE(d_reject_flag, nullptr);
        ASSERT_NE(d_reject_probability, nullptr);
        ASSERT_NE(d_reject_threshold, nullptr);
        ASSERT_NE(d_threshold_sample_token, nullptr);
        ASSERT_NE(d_threshold_verify_token, nullptr);
        ASSERT_NE(d_threshold_verify_flag, nullptr);
        ASSERT_NE(d_threshold_verify_probability, nullptr);
        ASSERT_NE(d_threshold_verify_threshold, nullptr);
        ASSERT_NE(d_batch_target_ids, nullptr);
        ASSERT_NE(d_batch_target_probs, nullptr);
        ASSERT_NE(d_batch_draft_ids, nullptr);
        ASSERT_NE(d_batch_draft_probs, nullptr);
        ASSERT_NE(d_batch_verify_tokens, nullptr);
        ASSERT_NE(d_batch_accept_flags, nullptr);
        ASSERT_NE(d_batch_accept_probabilities, nullptr);
        ASSERT_NE(d_batch_accept_thresholds, nullptr);
        ASSERT_NE(d_batch_sampled_draft_tokens, nullptr);
        ASSERT_NE(d_batch_sampled_draft_probabilities, nullptr);
        ASSERT_NE(d_batch_device_token_verify_tokens, nullptr);
        ASSERT_NE(d_batch_device_token_accept_flags, nullptr);
        ASSERT_NE(d_batch_device_token_accept_probabilities, nullptr);
        ASSERT_NE(d_batch_device_token_accept_thresholds, nullptr);

        const int batch_draft_tokens[2] = {accept_draft_token, reject_draft_token};
        const float batch_accept_thresholds[2] = {0.0f, 0.99f};
        const float batch_residual_thresholds[2] = {0.0f, 0.0f};
        const float batch_draft_token_probabilities[2] = {
            distributionProbability(expected_draft_accept, accept_draft_token) * 2.0f,
            distributionProbability(expected_draft_reject, reject_draft_token)};
        const auto expected_batch_accept =
            expectedSpeculativeVerifyDistributionWithThresholds(
                expected_target,
                expected_draft_accept,
                accept_draft_token,
                batch_accept_thresholds[0],
                batch_residual_thresholds[0]);
        const auto expected_batch_reject =
            expectedSpeculativeVerifyDistributionWithThresholds(
                expected_target,
                expected_draft_reject,
                reject_draft_token,
                batch_accept_thresholds[1],
                batch_residual_thresholds[1]);
        ASSERT_EQ(expected_batch_accept.accepted, 1);
        ASSERT_EQ(expected_batch_reject.accepted, 0);
        const float expected_device_token_accept_probability =
            std::min(
                1.0f,
                distributionProbability(expected_target, accept_draft_token) /
                    batch_draft_token_probabilities[0]);

        auto run_capture = [&](IWorkerGPUContext &ctx)
        {
            ctx.submitAndWait([&]()
            {
                void *stream = ctx.defaultStream();
                ASSERT_NE(stream, nullptr);

                ASSERT_TRUE(backend_->hostToDevice(
                    d_target_logits,
                    target_logits.data(),
                    target_logits.size() * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_draft_accept_logits,
                    draft_accept_logits.data(),
                    draft_accept_logits.size() * sizeof(float),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_draft_reject_logits,
                    draft_reject_logits.data(),
                    draft_reject_logits.size() * sizeof(float),
                    device_id_,
                    stream));
                // Device-token verifier regression setup: sampled MTP draft
                // tokens must already live in device scratch before capture.
                ASSERT_TRUE(backend_->hostToDevice(
                    d_batch_sampled_draft_tokens,
                    batch_draft_tokens,
                    sizeof(batch_draft_tokens),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->hostToDevice(
                    d_batch_sampled_draft_probabilities,
                    batch_draft_token_probabilities,
                    sizeof(batch_draft_token_probabilities),
                    device_id_,
                    stream));
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));

                EXPECT_FALSE(backend_->enqueueBuildTopKTopPDistributionF32Device(
                    d_target_logits,
                    static_cast<int>(target_logits.size()),
                    top_k,
                    top_p,
                    temperature,
                    device_id_,
                    nullptr,
                    d_target_ids,
                    d_target_probs))
                    << "distribution builder must reject the legacy default/null stream";
                EXPECT_FALSE(backend_->enqueueSpeculativeVerifyDistributionsF32Device(
                    d_target_ids,
                    d_target_probs,
                    d_draft_accept_ids,
                    d_draft_accept_probs,
                    top_k,
                    accept_draft_token,
                    accept_seed_accept_case,
                    accept_offset_accept_case,
                    residual_seed,
                    residual_offset,
                    device_id_,
                    nullptr,
                    d_accept_token,
                    d_accept_flag,
                    d_accept_probability,
                    d_accept_threshold))
                    << "speculative verifier must reject the legacy default/null stream";
                EXPECT_FALSE(backend_->enqueueSampleDistributionF32Device(
                    d_target_ids,
                    d_target_probs,
                    top_k,
                    0.25f,
                    device_id_,
                    nullptr,
                    d_threshold_sample_token))
                    << "compact distribution sampler must reject the legacy default/null stream";
                EXPECT_FALSE(backend_->enqueueSpeculativeVerifyDistributionsF32DeviceThresholds(
                    d_target_ids,
                    d_target_probs,
                    d_draft_reject_ids,
                    d_draft_reject_probs,
                    top_k,
                    reject_draft_token,
                    0.99f,
                    0.0f,
                    device_id_,
                    nullptr,
                    d_threshold_verify_token,
                    d_threshold_verify_flag,
                    d_threshold_verify_probability,
                    d_threshold_verify_threshold))
                    << "threshold verifier must reject the legacy default/null stream";
                EXPECT_FALSE(backend_->enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatch(
                    d_batch_target_ids,
                    d_batch_target_probs,
                    d_batch_draft_ids,
                    d_batch_draft_probs,
                    top_k,
                    top_k,
                    batch_draft_tokens,
                    batch_accept_thresholds,
                    batch_residual_thresholds,
                    2,
                    device_id_,
                    nullptr,
                    d_batch_verify_tokens,
                    d_batch_accept_flags,
                    d_batch_accept_probabilities,
                    d_batch_accept_thresholds))
                    << "batched speculative verifier must reject the legacy default/null stream";
                EXPECT_FALSE(backend_->enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatchDeviceTokens(
                    d_batch_target_ids,
                    d_batch_target_probs,
                    d_batch_draft_ids,
                    d_batch_draft_probs,
                    top_k,
                    top_k,
                    d_batch_sampled_draft_tokens,
                    batch_accept_thresholds,
                    batch_residual_thresholds,
                    2,
                    device_id_,
                    nullptr,
                    d_batch_device_token_verify_tokens,
                    d_batch_device_token_accept_flags,
                    d_batch_device_token_accept_probabilities,
                    d_batch_device_token_accept_thresholds,
                    d_batch_sampled_draft_probabilities))
                    << "device-token batched verifier must reject the legacy default/null stream";

                auto capture = ctx.createGraphCapture(stream);
                ASSERT_NE(capture, nullptr);
                ASSERT_TRUE(capture->beginCapture());
                ASSERT_TRUE(backend_->enqueueBuildTopKTopPDistributionF32Device(
                    d_target_logits,
                    static_cast<int>(target_logits.size()),
                    top_k,
                    top_p,
                    temperature,
                    device_id_,
                    stream,
                    d_target_ids,
                    d_target_probs));
                ASSERT_TRUE(backend_->enqueueBuildTopKTopPDistributionF32Device(
                    d_draft_accept_logits,
                    static_cast<int>(draft_accept_logits.size()),
                    top_k,
                    top_p,
                    temperature,
                    device_id_,
                    stream,
                    d_draft_accept_ids,
                    d_draft_accept_probs));
                ASSERT_TRUE(backend_->enqueueBuildTopKTopPDistributionF32Device(
                    d_draft_reject_logits,
                    static_cast<int>(draft_reject_logits.size()),
                    top_k,
                    top_p,
                    temperature,
                    device_id_,
                    stream,
                    d_draft_reject_ids,
                    d_draft_reject_probs));
                ASSERT_TRUE(backend_->enqueueSpeculativeVerifyDistributionsF32Device(
                    d_target_ids,
                    d_target_probs,
                    d_draft_accept_ids,
                    d_draft_accept_probs,
                    top_k,
                    accept_draft_token,
                    accept_seed_accept_case,
                    accept_offset_accept_case,
                    residual_seed,
                    residual_offset,
                    device_id_,
                    stream,
                    d_accept_token,
                    d_accept_flag,
                    d_accept_probability,
                    d_accept_threshold));
                ASSERT_TRUE(backend_->enqueueSpeculativeVerifyDistributionsF32Device(
                    d_target_ids,
                    d_target_probs,
                    d_draft_reject_ids,
                    d_draft_reject_probs,
                    top_k,
                    reject_draft_token,
                    accept_seed_reject_case,
                    accept_offset_reject_case,
                    residual_seed,
                    residual_offset,
                    device_id_,
                    stream,
                    d_reject_token,
                    d_reject_flag,
                    d_reject_probability,
                    d_reject_threshold));
                ASSERT_TRUE(backend_->enqueueSampleDistributionF32Device(
                    d_target_ids,
                    d_target_probs,
                    top_k,
                    0.25f,
                    device_id_,
                    stream,
                    d_threshold_sample_token));
                ASSERT_TRUE(backend_->enqueueSpeculativeVerifyDistributionsF32DeviceThresholds(
                    d_target_ids,
                    d_target_probs,
                    d_draft_reject_ids,
                    d_draft_reject_probs,
                    top_k,
                    reject_draft_token,
                    0.99f,
                    0.0f,
                    device_id_,
                    stream,
                    d_threshold_verify_token,
                    d_threshold_verify_flag,
                    d_threshold_verify_probability,
                    d_threshold_verify_threshold));
                ASSERT_TRUE(backend_->enqueueBuildTopKTopPDistributionF32Device(
                    d_target_logits,
                    static_cast<int>(target_logits.size()),
                    top_k,
                    top_p,
                    temperature,
                    device_id_,
                    stream,
                    static_cast<int *>(d_batch_target_ids),
                    static_cast<float *>(d_batch_target_probs)));
                ASSERT_TRUE(backend_->enqueueBuildTopKTopPDistributionF32Device(
                    d_target_logits,
                    static_cast<int>(target_logits.size()),
                    top_k,
                    top_p,
                    temperature,
                    device_id_,
                    stream,
                    static_cast<int *>(d_batch_target_ids) + top_k,
                    static_cast<float *>(d_batch_target_probs) + top_k));
                ASSERT_TRUE(backend_->enqueueBuildTopKTopPDistributionF32Device(
                    d_draft_accept_logits,
                    static_cast<int>(draft_accept_logits.size()),
                    top_k,
                    top_p,
                    temperature,
                    device_id_,
                    stream,
                    static_cast<int *>(d_batch_draft_ids),
                    static_cast<float *>(d_batch_draft_probs)));
                ASSERT_TRUE(backend_->enqueueBuildTopKTopPDistributionF32Device(
                    d_draft_reject_logits,
                    static_cast<int>(draft_reject_logits.size()),
                    top_k,
                    top_p,
                    temperature,
                    device_id_,
                    stream,
                    static_cast<int *>(d_batch_draft_ids) + top_k,
                    static_cast<float *>(d_batch_draft_probs) + top_k));
                ASSERT_TRUE(backend_->enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatch(
                    d_batch_target_ids,
                    d_batch_target_probs,
                    d_batch_draft_ids,
                    d_batch_draft_probs,
                    top_k,
                    top_k,
                    batch_draft_tokens,
                    batch_accept_thresholds,
                    batch_residual_thresholds,
                    2,
                    device_id_,
                    stream,
                    d_batch_verify_tokens,
                    d_batch_accept_flags,
                    d_batch_accept_probabilities,
                    d_batch_accept_thresholds));
                ASSERT_TRUE(backend_->enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatchDeviceTokens(
                    d_batch_target_ids,
                    d_batch_target_probs,
                    d_batch_draft_ids,
                    d_batch_draft_probs,
                    top_k,
                    top_k,
                    d_batch_sampled_draft_tokens,
                    batch_accept_thresholds,
                    batch_residual_thresholds,
                    2,
                    device_id_,
                    stream,
                    d_batch_device_token_verify_tokens,
                    d_batch_device_token_accept_flags,
                    d_batch_device_token_accept_probabilities,
                    d_batch_device_token_accept_thresholds,
                    d_batch_sampled_draft_probabilities));
                ASSERT_TRUE(capture->endCapture());
                ASSERT_TRUE(capture->instantiate());
                ASSERT_TRUE(capture->launch());
                ASSERT_TRUE(backend_->synchronizeStream(stream, device_id_));
            });
        };

        if (GetParam() == "CUDA")
        {
            auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(device_id_);
            run_capture(ctx);
        }
        else
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id_);
            run_capture(ctx);
        }

        std::vector<int> target_ids(top_k, -1);
        std::vector<float> target_probs(top_k, 0.0f);
        int accept_token = -1;
        int accept_flag = -1;
        float accept_probability = -1.0f;
        float accept_threshold = -1.0f;
        int reject_token = -1;
        int reject_flag = -1;
        float reject_probability = -1.0f;
        float reject_threshold = -1.0f;
        int threshold_sample_token = -1;
        int threshold_verify_token = -1;
        int threshold_verify_flag = -1;
        float threshold_verify_probability = -1.0f;
        float threshold_verify_threshold = -1.0f;
        std::vector<int> batch_verify_tokens(2, -1);
        std::vector<int> batch_accept_flags(2, -1);
        std::vector<float> batch_accept_probabilities(2, -1.0f);
        std::vector<float> batch_accept_threshold_results(2, -1.0f);
        std::vector<int> batch_device_token_verify_tokens(2, -1);
        std::vector<int> batch_device_token_accept_flags(2, -1);
        std::vector<float> batch_device_token_accept_probabilities(2, -1.0f);
        std::vector<float> batch_device_token_accept_threshold_results(2, -1.0f);

        ASSERT_TRUE(backend_->deviceToHost(target_ids.data(), d_target_ids, top_k * sizeof(int), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(target_probs.data(), d_target_probs, top_k * sizeof(float), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(&accept_token, d_accept_token, sizeof(int), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(&accept_flag, d_accept_flag, sizeof(int), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(&accept_probability, d_accept_probability, sizeof(float), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(&accept_threshold, d_accept_threshold, sizeof(float), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(&reject_token, d_reject_token, sizeof(int), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(&reject_flag, d_reject_flag, sizeof(int), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(&reject_probability, d_reject_probability, sizeof(float), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(&reject_threshold, d_reject_threshold, sizeof(float), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(&threshold_sample_token, d_threshold_sample_token, sizeof(int), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(&threshold_verify_token, d_threshold_verify_token, sizeof(int), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(&threshold_verify_flag, d_threshold_verify_flag, sizeof(int), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(&threshold_verify_probability, d_threshold_verify_probability, sizeof(float), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(&threshold_verify_threshold, d_threshold_verify_threshold, sizeof(float), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(batch_verify_tokens.data(), d_batch_verify_tokens, 2 * sizeof(int), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(batch_accept_flags.data(), d_batch_accept_flags, 2 * sizeof(int), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(batch_accept_probabilities.data(), d_batch_accept_probabilities, 2 * sizeof(float), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(batch_accept_threshold_results.data(), d_batch_accept_thresholds, 2 * sizeof(float), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(batch_device_token_verify_tokens.data(), d_batch_device_token_verify_tokens, 2 * sizeof(int), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(batch_device_token_accept_flags.data(), d_batch_device_token_accept_flags, 2 * sizeof(int), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(batch_device_token_accept_probabilities.data(), d_batch_device_token_accept_probabilities, 2 * sizeof(float), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(batch_device_token_accept_threshold_results.data(), d_batch_device_token_accept_thresholds, 2 * sizeof(float), device_id_));

        cleanup();

        for (int i = 0; i < top_k; ++i)
        {
            EXPECT_EQ(target_ids[static_cast<size_t>(i)],
                      expected_target[static_cast<size_t>(i)].token_id)
                << "target compact distribution token mismatch at slot " << i;
            EXPECT_NEAR(target_probs[static_cast<size_t>(i)],
                        expected_target[static_cast<size_t>(i)].probability,
                        1e-5f)
                << "target compact distribution probability mismatch at slot " << i;
        }

        EXPECT_EQ(accept_flag, expected_accept.accepted);
        EXPECT_EQ(accept_token, expected_accept.token_id);
        EXPECT_NEAR(accept_probability, expected_accept.accept_probability, 1e-5f);
        EXPECT_NEAR(accept_threshold, expected_accept.accept_threshold, 1e-6f);

        EXPECT_EQ(reject_flag, expected_reject.accepted);
        EXPECT_EQ(reject_token, expected_reject.token_id);
        EXPECT_NEAR(reject_probability, expected_reject.accept_probability, 1e-5f);
        EXPECT_NEAR(reject_threshold, expected_reject.accept_threshold, 1e-6f);

        const int expected_threshold_sample =
            expectedSampleDistributionWithThreshold(expected_target, 0.25f);
        const auto expected_threshold_verify =
            expectedSpeculativeVerifyDistributionWithThresholds(
                expected_target,
                expected_draft_reject,
                reject_draft_token,
                0.99f,
                0.0f);
        EXPECT_EQ(threshold_sample_token, expected_threshold_sample);
        EXPECT_EQ(threshold_verify_flag, expected_threshold_verify.accepted);
        EXPECT_EQ(threshold_verify_token, expected_threshold_verify.token_id);
        EXPECT_NEAR(threshold_verify_probability, expected_threshold_verify.accept_probability, 1e-5f);
        EXPECT_NEAR(threshold_verify_threshold, expected_threshold_verify.accept_threshold, 1e-6f);

        EXPECT_EQ(batch_verify_tokens[0], expected_batch_accept.token_id);
        EXPECT_EQ(batch_accept_flags[0], expected_batch_accept.accepted);
        EXPECT_NEAR(batch_accept_probabilities[0], expected_batch_accept.accept_probability, 1e-5f);
        EXPECT_NEAR(batch_accept_threshold_results[0], expected_batch_accept.accept_threshold, 1e-6f);
        EXPECT_EQ(batch_device_token_verify_tokens[0], expected_batch_accept.token_id);
        EXPECT_EQ(batch_device_token_accept_flags[0], expected_batch_accept.accepted);
        EXPECT_NEAR(batch_device_token_accept_probabilities[0], expected_device_token_accept_probability, 1e-5f);
        EXPECT_NEAR(batch_device_token_accept_threshold_results[0], expected_batch_accept.accept_threshold, 1e-6f);

        EXPECT_EQ(batch_verify_tokens[1], expected_batch_reject.token_id);
        EXPECT_EQ(batch_accept_flags[1], expected_batch_reject.accepted);
        EXPECT_NEAR(batch_accept_probabilities[1], expected_batch_reject.accept_probability, 1e-5f);
        EXPECT_NEAR(batch_accept_threshold_results[1], expected_batch_reject.accept_threshold, 1e-6f);
        EXPECT_EQ(batch_device_token_verify_tokens[1], expected_batch_reject.token_id);
        EXPECT_EQ(batch_device_token_accept_flags[1], expected_batch_reject.accepted);
        EXPECT_NEAR(batch_device_token_accept_probabilities[1], expected_batch_reject.accept_probability, 1e-5f);
        EXPECT_NEAR(batch_device_token_accept_threshold_results[1], expected_batch_reject.accept_threshold, 1e-6f);

        EXPECT_EQ(batch_device_token_verify_tokens, batch_verify_tokens)
            << "device-token batch verifier must match host-token verifier tokens";
        EXPECT_EQ(batch_device_token_accept_flags, batch_accept_flags)
            << "device-token batch verifier must match host-token verifier accept flags";
    }

    TEST_P(GPUSamplingTest, DRYParity_SimpleRepeat)
    {
        // History: [A, B, C, A, B, C] — "A B C" repeated
        // DRY should penalize token A (extending the repeat)
        const int vocab_size = 100;
        const int A = 10, B = 20, C = 30;

        Sampler sampler(42);
        for (int token : {A, B, C, A, B, C})
            sampler.record_token(token);

        SamplingParams params;
        params.dry_multiplier = 1.0f;
        params.dry_base = 1.75f;
        params.dry_allowed_length = 1;
        params.dry_penalty_last_n = -1;

        auto penalties = sampler.compute_penalty_map(params, vocab_size);
        ASSERT_FALSE(penalties.empty()) << "DRY should produce penalties for repeated pattern";

        // Identical logits for CPU and GPU
        std::vector<float> logits(vocab_size, 5.0f);

        auto cpu_result = logits;
        applyCpuPenalties(cpu_result, penalties);

        auto gpu_result = applyGpuPenalties(backend_, device_id_, logits, penalties);

        for (int i = 0; i < vocab_size; ++i)
        {
            EXPECT_FLOAT_EQ(cpu_result[i], gpu_result[i])
                << "CPU↔GPU mismatch at token " << i;
        }
    }

    TEST_P(GPUSamplingTest, DRYParity_ExponentialScaling)
    {
        // History: [A, B, C, D, A, B, C, D] — repeat of length 4
        // Produces penalty = 2.0 * 1.75^(4-1) = 10.72 on token A
        const int vocab_size = 100;
        const int A = 10, B = 20, C = 30, D = 40;

        Sampler sampler(42);
        for (int token : {A, B, C, D, A, B, C, D})
            sampler.record_token(token);

        SamplingParams params;
        params.dry_multiplier = 2.0f;
        params.dry_base = 1.75f;
        params.dry_allowed_length = 1;
        params.dry_penalty_last_n = -1;

        auto penalties = sampler.compute_penalty_map(params, vocab_size);
        ASSERT_FALSE(penalties.empty());

        std::vector<float> logits(vocab_size, 10.0f);

        auto cpu_result = logits;
        applyCpuPenalties(cpu_result, penalties);

        auto gpu_result = applyGpuPenalties(backend_, device_id_, logits, penalties);

        for (int i = 0; i < vocab_size; ++i)
        {
            EXPECT_FLOAT_EQ(cpu_result[i], gpu_result[i])
                << "CPU↔GPU mismatch at token " << i;
        }

        // Sanity: token A should have the expected exponential penalty
        float expected_penalty = 2.0f * std::pow(1.75f, 3.0f);
        EXPECT_NEAR(gpu_result[A], 10.0f - expected_penalty, 0.01f);
    }

    TEST_P(GPUSamplingTest, DRYParity_CombinedWithPresenceFrequency)
    {
        // DRY + presence + frequency penalties all combined
        const int vocab_size = 100;
        const int A = 10;

        Sampler sampler(42);
        for (int i = 0; i < 4; ++i)
            sampler.record_token(A);

        SamplingParams params;
        params.presence_penalty = 1.0f;
        params.frequency_penalty = 0.5f;
        params.dry_multiplier = 1.0f;
        params.dry_base = 1.75f;
        params.dry_allowed_length = 0;
        params.dry_penalty_last_n = -1;

        auto penalties = sampler.compute_penalty_map(params, vocab_size);
        ASSERT_FALSE(penalties.empty());

        std::vector<float> logits(vocab_size, 5.0f);

        auto cpu_result = logits;
        applyCpuPenalties(cpu_result, penalties);

        auto gpu_result = applyGpuPenalties(backend_, device_id_, logits, penalties);

        for (int i = 0; i < vocab_size; ++i)
        {
            EXPECT_FLOAT_EQ(cpu_result[i], gpu_result[i])
                << "CPU↔GPU mismatch at token " << i;
        }

        // Verify the combined penalty is additive (pres+freq + DRY)
        float pf_penalty = 1.0f + 0.5f * 4.0f; // 3.0
        EXPECT_GT(5.0f - gpu_result[A], pf_penalty)
            << "Combined penalty should exceed presence+frequency alone";
    }

    TEST_P(GPUSamplingTest, DRYParity_SequenceBreakers)
    {
        // History with a breaker in the middle — should NOT penalize across it
        const int vocab_size = 100;
        const int A = 10, NEWLINE = 50;

        Sampler sampler(42);
        sampler.initDryBreakers({"\n"}, [&](const std::string &) -> std::vector<int> {
            return {NEWLINE};
        });
        for (int token : {A, NEWLINE, A})
            sampler.record_token(token);

        SamplingParams params;
        params.dry_multiplier = 1.0f;
        params.dry_allowed_length = 1;
        params.dry_penalty_last_n = -1;

        auto penalties = sampler.compute_penalty_map(params, vocab_size);

        // With breaker, A should not be penalized — penalty map may be empty
        std::vector<float> logits(vocab_size, 5.0f);

        auto cpu_result = logits;
        applyCpuPenalties(cpu_result, penalties);

        auto gpu_result = applyGpuPenalties(backend_, device_id_, logits, penalties);

        for (int i = 0; i < vocab_size; ++i)
        {
            EXPECT_FLOAT_EQ(cpu_result[i], gpu_result[i])
                << "CPU↔GPU mismatch at token " << i;
        }

        // Token A should be unpenalized (breaker prevents detection)
        EXPECT_FLOAT_EQ(gpu_result[A], 5.0f)
            << "Sequence breaker should prevent DRY penalty on token A";
    }

    TEST_P(GPUSamplingTest, DRYParity_SingleTokenBreakerExemption)
    {
        // Token that is itself a single-token breaker should be exempt
        const int vocab_size = 100;
        const int A = 10, NEWLINE = 50;

        Sampler sampler(42);
        sampler.initDryBreakers({"\n"}, [&](const std::string &) -> std::vector<int> {
            return {NEWLINE};
        });
        // NEWLINE A NEWLINE A NEWLINE — repeat pattern, but NEWLINE is a breaker
        for (int token : {NEWLINE, A, NEWLINE, A, NEWLINE})
            sampler.record_token(token);

        SamplingParams params;
        params.dry_multiplier = 1.0f;
        params.dry_allowed_length = 0;
        params.dry_penalty_last_n = -1;

        auto penalties = sampler.compute_penalty_map(params, vocab_size);

        std::vector<float> logits(vocab_size, 5.0f);

        auto cpu_result = logits;
        applyCpuPenalties(cpu_result, penalties);

        auto gpu_result = applyGpuPenalties(backend_, device_id_, logits, penalties);

        for (int i = 0; i < vocab_size; ++i)
        {
            EXPECT_FLOAT_EQ(cpu_result[i], gpu_result[i])
                << "CPU↔GPU mismatch at token " << i;
        }

        // NEWLINE should be exempt (single-token breaker)
        EXPECT_FLOAT_EQ(gpu_result[NEWLINE], 5.0f)
            << "Single-token breaker should be exempt from DRY penalty";
    }

    TEST_P(GPUSamplingTest, DRYParity_OverflowProtection)
    {
        // Large repeat count with base=2.0 — should not overflow to inf
        const int vocab_size = 100;
        const int A = 10;

        Sampler sampler(42);
        for (int i = 0; i < 50; ++i)
            sampler.record_token(A);

        SamplingParams params;
        params.dry_multiplier = 1.0f;
        params.dry_base = 2.0f;
        params.dry_allowed_length = 0;
        params.dry_penalty_last_n = -1;

        auto penalties = sampler.compute_penalty_map(params, vocab_size);
        ASSERT_FALSE(penalties.empty());

        // Verify no overflow in CPU computation
        for (const auto &p : penalties)
        {
            EXPECT_FALSE(std::isinf(p.penalty)) << "CPU penalty should not overflow";
            EXPECT_FALSE(std::isnan(p.penalty)) << "CPU penalty should not be NaN";
        }

        std::vector<float> logits(vocab_size, 100.0f);

        auto cpu_result = logits;
        applyCpuPenalties(cpu_result, penalties);

        auto gpu_result = applyGpuPenalties(backend_, device_id_, logits, penalties);

        for (int i = 0; i < vocab_size; ++i)
        {
            EXPECT_FLOAT_EQ(cpu_result[i], gpu_result[i])
                << "CPU↔GPU mismatch at token " << i;
        }
    }

    TEST_P(GPUSamplingTest, DRYParity_WindowLimitsDetection)
    {
        // With a small window, long repeats outside the window should not be detected
        const int vocab_size = 100;

        Sampler sampler(42);
        for (int token : {1, 2, 3, 4, 5, 1, 2, 3, 4, 5})
            sampler.record_token(token);

        SamplingParams params;
        params.dry_multiplier = 1.0f;
        params.dry_allowed_length = 0;
        params.dry_penalty_last_n = 3; // Only see last 3 tokens

        auto penalties = sampler.compute_penalty_map(params, vocab_size);

        std::vector<float> logits(vocab_size, 5.0f);

        auto cpu_result = logits;
        applyCpuPenalties(cpu_result, penalties);

        auto gpu_result = applyGpuPenalties(backend_, device_id_, logits, penalties);

        for (int i = 0; i < vocab_size; ++i)
        {
            EXPECT_FLOAT_EQ(cpu_result[i], gpu_result[i])
                << "CPU↔GPU mismatch at token " << i;
        }

        // With only 3-token window, cannot detect the full 5-length repeat
        float full_penalty = std::pow(1.75f, 4.0f);
        for (const auto &p : penalties)
        {
            EXPECT_LT(p.penalty, full_penalty)
                << "Window should prevent detection of full repeat";
        }
    }

    TEST_P(GPUSamplingTest, DRYParity_ArgmaxShift)
    {
        // End-to-end: DRY penalty shifts GPU argmax to match CPU argmax
        const int vocab_size = 10;

        Sampler sampler(42);
        // Token 5 and 7 alternate — so token 5 would extend the repeat
        for (int token : {5, 7, 5, 7})
            sampler.record_token(token);

        SamplingParams params;
        params.dry_multiplier = 5.0f;
        params.dry_base = 1.75f;
        params.dry_allowed_length = 0;
        params.dry_penalty_last_n = -1;

        auto penalties = sampler.compute_penalty_map(params, vocab_size);
        ASSERT_FALSE(penalties.empty());

        // Token 5 has highest logit but will be penalized by DRY
        std::vector<float> logits = {0.0f, 0.0f, 0.0f, 9.5f, 0.0f,
                                     10.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        // CPU: apply penalties and find argmax
        auto cpu_logits = logits;
        applyCpuPenalties(cpu_logits, penalties);
        int cpu_argmax = static_cast<int>(
            std::max_element(cpu_logits.begin(), cpu_logits.end()) - cpu_logits.begin());

        // GPU: apply penalties and find argmax
        void *d_ptr = uploadLogits(logits);
        ASSERT_NE(d_ptr, nullptr);

        std::vector<int> token_ids(penalties.size());
        std::vector<float> penalty_vals(penalties.size());
        for (size_t i = 0; i < penalties.size(); ++i)
        {
            token_ids[i] = penalties[i].token_id;
            penalty_vals[i] = penalties[i].penalty;
        }

        bool ok = backend_->applyLogitPenaltiesF32(
            d_ptr, token_ids.data(), penalty_vals.data(),
            static_cast<int>(penalties.size()), vocab_size, device_id_);
        ASSERT_TRUE(ok);

        float gpu_val = 0;
        int gpu_argmax = -1;
        ok = argmaxF32(d_ptr, vocab_size, device_id_, &gpu_val, &gpu_argmax);
        ASSERT_TRUE(ok);

        EXPECT_EQ(gpu_argmax, cpu_argmax)
            << "GPU argmax should match CPU argmax after DRY penalties";
        EXPECT_EQ(gpu_argmax, 3)
            << "Token 3 (9.5) should win after token 5 is penalized by DRY";

        freeDevice(d_ptr);
    }

    TEST_P(GPUSamplingTest, DRYParity_LargeVocab_RealisticScenario)
    {
        // Realistic scenario: Qwen2 vocab size, natural-looking token history
        const int vocab_size = 151936;
        std::mt19937 rng(12345);

        Sampler sampler(42);
        // Simulate a conversation with some repetitive patterns
        std::vector<int> history = {
            100, 200, 300, 400, 500,   // unique intro
            100, 200, 300, 400, 500,   // exact repeat
            600, 700, 800,             // break
            100, 200, 300, 400, 500,   // another repeat
            900, 1000                  // end
        };
        for (int token : history)
            sampler.record_token(token);

        SamplingParams params;
        params.dry_multiplier = 1.5f;
        params.dry_base = 1.75f;
        params.dry_allowed_length = 2;
        params.dry_penalty_last_n = -1;
        params.presence_penalty = 0.5f;
        params.frequency_penalty = 0.3f;

        auto penalties = sampler.compute_penalty_map(params, vocab_size);
        ASSERT_FALSE(penalties.empty());

        // Generate logits with some structure
        std::vector<float> logits(vocab_size);
        std::uniform_real_distribution<float> dist(-5.0f, 15.0f);
        for (auto &l : logits)
            l = dist(rng);

        auto cpu_result = logits;
        applyCpuPenalties(cpu_result, penalties);

        auto gpu_result = applyGpuPenalties(backend_, device_id_, logits, penalties);

        // Spot-check penalized tokens
        for (const auto &p : penalties)
        {
            EXPECT_FLOAT_EQ(cpu_result[p.token_id], gpu_result[p.token_id])
                << "CPU↔GPU mismatch at penalized token " << p.token_id;
        }

        // Spot-check unpenalized tokens
        std::set<int> penalized_ids;
        for (const auto &p : penalties)
            penalized_ids.insert(p.token_id);

        int checked = 0;
        for (int i = 0; i < vocab_size && checked < 100; ++i)
        {
            if (penalized_ids.find(i) == penalized_ids.end())
            {
                EXPECT_FLOAT_EQ(cpu_result[i], gpu_result[i])
                    << "Unpenalized token " << i << " should be unchanged";
                checked++;
            }
        }
    }

} // anonymous namespace
