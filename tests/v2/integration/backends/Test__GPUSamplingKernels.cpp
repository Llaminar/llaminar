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
#include "kernels/common/SamplingMath.h"
#include "utils/Sampler.h"

#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <set>
#include <map>
#include <random>

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
        const int expected_direct_sample =
            expectedTopKTopPSample(logits, top_k, top_p, temperature, seed, offset);
        ASSERT_EQ(expected_direct_sample, expected_sample)
            << "direct CPU sample and compact-distribution CPU sample should agree";

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

        d_logits = backend_->allocate(logits.size() * sizeof(float), device_id_);
        d_token_ids = backend_->allocate(top_k * sizeof(int), device_id_);
        d_probs = backend_->allocate(top_k * sizeof(float), device_id_);
        d_sample_token = backend_->allocate(sizeof(int), device_id_);
        d_direct_token = backend_->allocate(sizeof(int), device_id_);
        ASSERT_NE(d_logits, nullptr);
        ASSERT_NE(d_token_ids, nullptr);
        ASSERT_NE(d_probs, nullptr);
        ASSERT_NE(d_sample_token, nullptr);
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
                    d_sample_token));
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
        int direct_token = -1;
        ASSERT_TRUE(backend_->deviceToHost(gpu_ids.data(), d_token_ids, top_k * sizeof(int), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(gpu_probs.data(), d_probs, top_k * sizeof(float), device_id_));
        ASSERT_TRUE(backend_->deviceToHost(&sample_token, d_sample_token, sizeof(int), device_id_));
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
        ASSERT_NE(d_batch_device_token_verify_tokens, nullptr);
        ASSERT_NE(d_batch_device_token_accept_flags, nullptr);
        ASSERT_NE(d_batch_device_token_accept_probabilities, nullptr);
        ASSERT_NE(d_batch_device_token_accept_thresholds, nullptr);

        const int batch_draft_tokens[2] = {accept_draft_token, reject_draft_token};
        const float batch_accept_thresholds[2] = {0.0f, 0.99f};
        const float batch_residual_thresholds[2] = {0.0f, 0.0f};
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
                    d_batch_device_token_accept_thresholds))
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
                    d_batch_device_token_accept_thresholds));
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
        EXPECT_NEAR(batch_device_token_accept_probabilities[0], expected_batch_accept.accept_probability, 1e-5f);
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
