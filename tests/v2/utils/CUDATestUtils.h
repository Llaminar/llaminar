/**
 * @file CUDATestUtils.h
 * @brief Test utilities for CUDA kernel testing
 * @author David Sanftenberg
 * @date January 2026
 *
 * Provides:
 * - GPU availability skip guards
 * - Array comparison utilities with tolerance
 * - Random data generation
 * - CPU reference implementations
 *
 * IMPORTANT: Include project headers BEFORE this file:
 *   #include "devices/DeviceManager.h"
 *   #include "execution/local_execution/device/DeviceContext.h"
 *   #ifdef HAVE_CUDA
 *   #include "backends/cuda/CUDABackend.h"
 *   #endif
 *   #include "../utils/CUDATestUtils.h"   // This file
 *
 * Usage:
 *   using namespace llaminar2::test::cuda;
 *
 *   class MyTest : public CUDATestBase { ... };
 *
 *   TEST_F(MyTest, SomeTest) {
 *       auto data = generateRandomFP32(1024);
 *       ...
 *   }
 */

#pragma once

#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cstring>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include "backends/DeviceId.h" // DeviceId for GPU tests

// Project headers are expected to be included BEFORE this file
// (DeviceManager.h, DeviceContext.h, CUDABackend.h)

namespace llaminar2::test::cuda
{

    // =========================================================================
    // Tolerance Constants
    // =========================================================================

    // FP32 → FP32 comparison (strict)
    constexpr float FP32_ABS_TOL = 1e-5f;
    constexpr float FP32_REL_TOL = 1e-4f;

    // FP32 GEMM comparison (accumulation error scales with sqrt(K))
    // For typical K=128-4096, expect errors up to ~1e-3 to 2e-2
    constexpr float GEMM_ABS_TOL = 2e-2f; // Allows for accumulation differences
    constexpr float GEMM_REL_TOL = 1e-2f; // 1% relative tolerance

    // BF16/FP16 → FP32 comparison (precision loss expected)
    constexpr float HALF_ABS_TOL = 1e-3f;
    constexpr float HALF_REL_TOL = 1e-2f;

    // Quantized → FP32 comparison (quantization error expected)
    constexpr float QUANT_ABS_TOL = 1e-2f;
    constexpr float QUANT_REL_TOL = 5e-2f;

    // =========================================================================
    // Skip Guard Macro
    // =========================================================================

/**
 * @brief Skip test if no CUDA GPU is available
 *
 * Usage: Place at the start of any CUDA-dependent test
 *
 * NOTE: This macro is safe to use with CUDATestBase, which already initializes
 * DeviceManager. The initialize() call only runs if devices haven't been
 * enumerated yet. However, for tests using CUDATestBase, this macro is
 * redundant since CUDATestBase::SetUp() already handles the skip logic.
 */
#define SKIP_IF_NO_CUDA()                                           \
    do                                                              \
    {                                                               \
        auto &dm = ::llaminar2::DeviceManager::instance();          \
        if (dm.devices().empty())                                   \
        {                                                           \
            dm.initialize(-1); /* Enumerate all devices */          \
        }                                                           \
        if (!dm.has_gpu())                                          \
        {                                                           \
            GTEST_SKIP() << "No CUDA GPU available, skipping test"; \
        }                                                           \
    } while (0)

/**
 * @brief Skip test if HAVE_CUDA is not defined at compile time
 */
#ifndef HAVE_CUDA
#define SKIP_IF_NO_CUDA_BUILD() \
    GTEST_SKIP() << "Built without CUDA support (-DHAVE_CUDA=OFF)"
#else
#define SKIP_IF_NO_CUDA_BUILD() ((void)0)
#endif

    // =========================================================================
    // Test Base Class
    // =========================================================================

    /**
     * @brief Clear any lingering CUDA errors from previous operations
     *
     * CUDA errors are "sticky" - once an error occurs, all subsequent CUDA calls
     * will return the same error until it is cleared. This function clears the
     * error state and optionally synchronizes the device.
     *
     * @param sync If true, also synchronize all CUDA devices (slower but more thorough)
     * @return The error that was cleared, or cudaSuccess if no error
     */
#ifdef HAVE_CUDA
    inline cudaError_t clearCudaErrors(bool sync = false)
    {
        // Get and clear the last error
        cudaError_t err = cudaGetLastError();

        if (sync)
        {
            // Synchronize all devices to ensure all async work is complete
            int device_count = 0;
            cudaGetDeviceCount(&device_count);
            for (int i = 0; i < device_count; ++i)
            {
                cudaSetDevice(i);
                cudaDeviceSynchronize();
                cudaGetLastError(); // Clear any sync errors
            }
        }

        return err;
    }
#endif

    /**
     * @brief Base test fixture for CUDA tests
     *
     * Automatically skips tests when no GPU is available.
     * Provides gpu_device_ (DeviceId) and gpu_idx_ (legacy int) for derived tests.
     *
     * IMPORTANT: Clears CUDA error state in SetUp and TearDown to prevent
     * errors from one test affecting subsequent tests.
     */
    class CUDATestBase : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
#ifndef HAVE_CUDA
            GTEST_SKIP() << "Built without CUDA support";
#else
            // Clear any lingering CUDA errors from previous tests
            cudaError_t prior_err = clearCudaErrors(true);
            if (prior_err != cudaSuccess)
            {
                std::cerr << "[CUDATestBase::SetUp] Cleared prior CUDA error: "
                          << cudaGetErrorString(prior_err) << std::endl;
            }

            // Initialize DeviceManager to enumerate GPU devices
            // Must be called before has_gpu() or find_device()
            auto &dm = DeviceManager::instance();
            dm.initialize(-1); // -1 = no NUMA filtering, enumerate all devices

            if (!dm.has_gpu())
            {
                GTEST_SKIP() << "No CUDA GPU available";
            }

            gpu_idx_ = dm.find_device(ComputeBackendType::GPU_CUDA);
            if (gpu_idx_ < 0)
            {
                GTEST_SKIP() << "No CUDA device found";
            }

            // Get the backend-specific CUDA ordinal from DeviceManager
            // (gpu_idx_ - 1 is wrong when ROCm devices are also present)
            const auto &device_info = dm.devices()[gpu_idx_];
            int cuda_ordinal = device_info.device_id;
            gpu_device_ = DeviceId::cuda(cuda_ordinal);

            // CRITICAL: Set the active CUDA device for subsequent cudaMalloc/etc calls
            cudaError_t set_err = cudaSetDevice(cuda_ordinal);
            if (set_err != cudaSuccess)
            {
                GTEST_SKIP() << "Failed to set CUDA device " << cuda_ordinal
                             << ": " << cudaGetErrorString(set_err);
            }

            backend_ = std::make_unique<CUDABackend>();
            device_count_ = backend_->deviceCount();

            if (device_count_ == 0)
            {
                GTEST_SKIP() << "CUDABackend reports 0 devices";
            }

            std::cout << "Using CUDA device " << cuda_ordinal << ": "
                      << backend_->deviceName(cuda_ordinal) << std::endl;

            // Store the CUDA ordinal for tests that need direct CUDA API calls
            cuda_ordinal_ = cuda_ordinal;
#endif
        }

        void TearDown() override
        {
#ifdef HAVE_CUDA
            // Clear CUDA errors to prevent cascading to next test
            cudaError_t err = clearCudaErrors(true);
            if (err != cudaSuccess)
            {
                std::cerr << "[CUDATestBase::TearDown] Test left CUDA error: "
                          << cudaGetErrorString(err) << std::endl;
            }
            backend_.reset();
#endif
        }

        int gpu_idx_ = -1;                      ///< Legacy DeviceManager index (for backward compat)
        int cuda_ordinal_ = 0;                  ///< CUDA device ordinal for cudaSetDevice() etc
        DeviceId gpu_device_ = DeviceId::cpu(); ///< Preferred: DeviceId for the CUDA device
        int device_count_ = 0;

#ifdef HAVE_CUDA
        std::unique_ptr<CUDABackend> backend_;
#endif
    };

    // =========================================================================
    // Array Comparison Utilities
    // =========================================================================

    /**
     * @brief Comparison result with detailed statistics
     */
    struct ComparisonResult
    {
        bool passed = true;
        size_t mismatch_count = 0;
        size_t total_count = 0;
        float max_abs_diff = 0.0f;
        float max_rel_diff = 0.0f;
        size_t max_abs_idx = 0;
        size_t max_rel_idx = 0;

        void print() const
        {
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "  Max absolute difference: " << max_abs_diff
                      << " at index " << max_abs_idx << std::endl;
            std::cout << "  Max relative difference: " << max_rel_diff
                      << " at index " << max_rel_idx << std::endl;
            if (mismatch_count > 0)
            {
                std::cout << "  Mismatches: " << mismatch_count << "/"
                          << total_count << " ("
                          << (100.0f * mismatch_count / total_count) << "%)"
                          << std::endl;
            }
        }
    };

    /**
     * @brief Compare two float arrays with tolerance
     *
     * @param actual Test output (e.g., CUDA result)
     * @param expected Reference output (e.g., CPU result)
     * @param count Number of elements
     * @param abs_tol Absolute tolerance
     * @param rel_tol Relative tolerance
     * @param max_print_mismatches How many mismatches to print (0 = none)
     * @return ComparisonResult with statistics
     */
    inline ComparisonResult compareArrays(
        const float *actual,
        const float *expected,
        size_t count,
        float abs_tol = FP32_ABS_TOL,
        float rel_tol = FP32_REL_TOL,
        size_t max_print_mismatches = 5)
    {
        ComparisonResult result;
        result.total_count = count;

        for (size_t i = 0; i < count; ++i)
        {
            float abs_diff = std::abs(actual[i] - expected[i]);
            float rel_diff = abs_diff / (std::abs(expected[i]) + 1e-8f);

            if (abs_diff > result.max_abs_diff)
            {
                result.max_abs_diff = abs_diff;
                result.max_abs_idx = i;
            }
            if (rel_diff > result.max_rel_diff)
            {
                result.max_rel_diff = rel_diff;
                result.max_rel_idx = i;
            }

            // Fail if both tolerances exceeded
            if (abs_diff > abs_tol && rel_diff > rel_tol)
            {
                if (result.mismatch_count < max_print_mismatches)
                {
                    std::cout << "  Mismatch at [" << i << "]: actual="
                              << actual[i] << " expected=" << expected[i]
                              << " (abs_diff=" << abs_diff
                              << ", rel_diff=" << rel_diff << ")" << std::endl;
                }
                result.mismatch_count++;
                result.passed = false;
            }
        }

        return result;
    }

    /**
     * @brief Simple pass/fail comparison
     */
    inline bool arraysMatch(
        const float *actual,
        const float *expected,
        size_t count,
        float abs_tol = FP32_ABS_TOL,
        float rel_tol = FP32_REL_TOL)
    {
        auto result = compareArrays(actual, expected, count, abs_tol, rel_tol, 0);
        return result.passed;
    }

    // =========================================================================
    // Random Data Generation
    // =========================================================================

    /**
     * @brief Generate random FP32 data
     *
     * @param count Number of elements
     * @param min Minimum value
     * @param max Maximum value
     * @param seed Random seed (for reproducibility)
     */
    inline std::vector<float> generateRandomFP32(
        size_t count,
        float min = -1.0f,
        float max = 1.0f,
        unsigned seed = 12345)
    {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(min, max);

        std::vector<float> data(count);
        for (auto &val : data)
        {
            val = dist(gen);
        }
        return data;
    }

    /**
     * @brief Generate sequential FP32 data (for debugging)
     */
    inline std::vector<float> generateSequentialFP32(
        size_t count,
        float start = 0.0f,
        float step = 0.001f)
    {
        std::vector<float> data(count);
        for (size_t i = 0; i < count; ++i)
        {
            data[i] = start + static_cast<float>(i) * step;
        }
        return data;
    }

    // =========================================================================
    // CPU Reference Implementations
    // =========================================================================

    /**
     * @brief CPU reference GEMM: C = A * B^T (row-major)
     *
     * This matches the common pattern where B is stored row-major [N×K]
     * and we compute A[M×K] × B[N×K]^T → C[M×N]
     *
     * @param A Input matrix [M×K] row-major
     * @param B Weight matrix [N×K] row-major (transposed during multiply)
     * @param C Output matrix [M×N] row-major
     * @param M Number of rows in A/C
     * @param N Number of rows in B / columns in C
     * @param K Common dimension
     */
    inline void cpuGemmNT(
        const float *A,
        const float *B,
        float *C,
        int M, int N, int K)
    {
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                float sum = 0.0f;
                for (int p = 0; p < K; ++p)
                {
                    sum += A[i * K + p] * B[j * K + p]; // B[j, p] in row-major [N×K]
                }
                C[i * N + j] = sum;
            }
        }
    }

    /**
     * @brief CPU reference GEMM: C = A * B (both row-major, no transpose)
     *
     * Standard GEMM: A[M×K] × B[K×N] → C[M×N]
     */
    inline void cpuGemmNN(
        const float *A,
        const float *B,
        float *C,
        int M, int N, int K)
    {
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                float sum = 0.0f;
                for (int p = 0; p < K; ++p)
                {
                    sum += A[i * K + p] * B[p * N + j]; // B[p, j] in row-major [K×N]
                }
                C[i * N + j] = sum;
            }
        }
    }

    /**
     * @brief Check if array contains NaN or Inf values
     */
    inline bool hasNaNOrInf(const float *data, size_t count)
    {
        for (size_t i = 0; i < count; ++i)
        {
            if (std::isnan(data[i]) || std::isinf(data[i]))
            {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Print array statistics for debugging
     */
    inline void printArrayStats(const float *data, size_t count, const char *name = "array")
    {
        if (count == 0)
            return;

        float min_val = data[0];
        float max_val = data[0];
        double sum = 0.0;

        for (size_t i = 0; i < count; ++i)
        {
            min_val = std::min(min_val, data[i]);
            max_val = std::max(max_val, data[i]);
            sum += data[i];
        }

        std::cout << name << ": min=" << min_val
                  << " max=" << max_val
                  << " mean=" << (sum / count)
                  << " count=" << count << std::endl;
    }

} // namespace llaminar2::test::cuda
