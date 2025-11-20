/**
 * @file Test__OneDNNGemmKernel_ScratchBuffers.cpp
 * @brief Unit tests for OneDNNGemmKernel scratch buffer functionality
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <memory>
#include <cstring>

#ifdef HAVE_ONEDNN

#include "kernels/cpu/gemm_v4/OneDNNGemmKernel.h"
#include "tensors/FP16Utils.h"
#include "tensors/SIMDHelpers.h"
#include "utils/MPIContext.h"

using llaminar2::ActivationFormat;
using llaminar2::gemm_v4::OneDNNGemmKernel;

namespace
{
    /**
     * Test fixture for scratch buffer tests
     */
    class OneDNNGemmKernelScratchBuffers : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            kernel_ = std::make_unique<OneDNNGemmKernel>();
        }

        void TearDown() override
        {
            kernel_.reset();
        }

        std::unique_ptr<OneDNNGemmKernel> kernel_;
    };

    /**
     * @brief Verify scratch buffers are reused across multiple calls
     *
     * Tests that scratch buffers grow to accommodate the largest request
     * and are reused without reallocation on subsequent smaller calls.
     */
    TEST_F(OneDNNGemmKernelScratchBuffers, BufferReuse)
    {
        const int device_idx = -1; // Default device (CPU)
        llaminar2::MPIContext mpi_ctx(0, 1, MPI_COMM_WORLD);

        // First call with small matrices (should allocate buffers)
        {
            const int m = 8, n = 16, k = 12;
            std::vector<float> A(m * k, 1.0f);
            std::vector<float> B(k * n, 1.0f);
            std::vector<float> C(m * n, 0.0f);

            // Strided layout to force buffer usage
            const int lda = k + 5; // Padding
            const int ldb = n + 3;
            const int ldc = n + 2;

            std::vector<float> A_strided(m * lda, 0.0f);
            std::vector<float> B_strided(k * ldb, 0.0f);
            std::vector<float> C_strided(m * ldc, 0.0f);

            // Copy data with stride
            for (int i = 0; i < m; ++i)
            {
                std::memcpy(&A_strided[i * lda], &A[i * k], k * sizeof(float));
            }
            for (int i = 0; i < k; ++i)
            {
                std::memcpy(&B_strided[i * ldb], &B[i * n], n * sizeof(float));
            }

            bool result = kernel_->multiply_activations_strided(
                A_strided.data(), B_strided.data(), C_strided.data(),
                m, n, k, lda, ldb, ldc, false, 1.0f, 0.0f, &mpi_ctx, device_idx);

            ASSERT_TRUE(result);
        }

        // Second call with larger matrices (buffers should grow)
        {
            const int m = 32, n = 64, k = 48;
            std::vector<float> A(m * k, 2.0f);
            std::vector<float> B(k * n, 2.0f);
            std::vector<float> C(m * n, 0.0f);

            const int lda = k + 10;
            const int ldb = n + 8;
            const int ldc = n + 6;

            std::vector<float> A_strided(m * lda, 0.0f);
            std::vector<float> B_strided(k * ldb, 0.0f);
            std::vector<float> C_strided(m * ldc, 0.0f);

            for (int i = 0; i < m; ++i)
            {
                std::memcpy(&A_strided[i * lda], &A[i * k], k * sizeof(float));
            }
            for (int i = 0; i < k; ++i)
            {
                std::memcpy(&B_strided[i * ldb], &B[i * n], n * sizeof(float));
            }

            bool result = kernel_->multiply_activations_strided(
                A_strided.data(), B_strided.data(), C_strided.data(),
                m, n, k, lda, ldb, ldc, false, 1.0f, 0.0f, &mpi_ctx, device_idx);

            ASSERT_TRUE(result);
        }

        // Third call with smaller matrices (should reuse existing buffers)
        {
            const int m = 16, n = 32, k = 24;
            std::vector<float> A(m * k, 3.0f);
            std::vector<float> B(k * n, 3.0f);
            std::vector<float> C(m * n, 0.0f);

            const int lda = k + 4;
            const int ldb = n + 2;
            const int ldc = n + 1;

            std::vector<float> A_strided(m * lda, 0.0f);
            std::vector<float> B_strided(k * ldb, 0.0f);
            std::vector<float> C_strided(m * ldc, 0.0f);

            for (int i = 0; i < m; ++i)
            {
                std::memcpy(&A_strided[i * lda], &A[i * k], k * sizeof(float));
            }
            for (int i = 0; i < k; ++i)
            {
                std::memcpy(&B_strided[i * ldb], &B[i * n], n * sizeof(float));
            }

            bool result = kernel_->multiply_activations_strided(
                A_strided.data(), B_strided.data(), C_strided.data(),
                m, n, k, lda, ldb, ldc, false, 1.0f, 0.0f, &mpi_ctx, device_idx);

            ASSERT_TRUE(result);

            // Verify result correctness
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    float expected = k * 3.0f * 3.0f; // A[i,*] dot B[*,j]
                    float actual = C_strided[i * ldc + j];
                    EXPECT_NEAR(expected, actual, 1e-3f)
                        << "Mismatch at C[" << i << "," << j << "]";
                }
            }
        }
    }

    /**
     * @brief Test typed GEMM scratch buffer reuse with mixed precision
     */
    TEST_F(OneDNNGemmKernelScratchBuffers, TypedGemmBufferReuse)
    {
        const int device_idx = -1; // Default device (CPU)
        llaminar2::MPIContext mpi_ctx(0, 1, MPI_COMM_WORLD);

        // Test FP16 path (requires buffer conversions)
        {
            const int m = 16, n = 32, k = 24;
            std::vector<uint16_t> A_fp16(m * k);
            std::vector<uint16_t> B_fp16(k * n);
            std::vector<float> C(m * n, 0.0f);

            // Initialize with simple pattern
            for (size_t i = 0; i < A_fp16.size(); ++i)
            {
                A_fp16[i] = llaminar2::fp32_to_fp16(1.5f);
            }
            for (size_t i = 0; i < B_fp16.size(); ++i)
            {
                B_fp16[i] = llaminar2::fp32_to_fp16(2.0f);
            }

            bool result = kernel_->multiply_activations_typed_impl(
                A_fp16.data(), B_fp16.data(), C.data(),
                m, n, k, false, 1.0f, 0.0f, &mpi_ctx, device_idx,
                ActivationFormat::FP16, ActivationFormat::FP16);

            ASSERT_TRUE(result);

            // Verify correctness
            float expected = k * 1.5f * 2.0f;
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    EXPECT_NEAR(expected, C[i * n + j], 0.5f)
                        << "Mismatch at C[" << i << "," << j << "]";
                }
            }
        }

        // Test BF16 path (different buffer requirements)
        {
            const int m = 24, n = 48, k = 32;
            std::vector<uint16_t> A_bf16(m * k);
            std::vector<uint16_t> B_bf16(k * n);
            std::vector<float> C(m * n, 0.0f);

            for (size_t i = 0; i < A_bf16.size(); ++i)
            {
                A_bf16[i] = llaminar2::simd::fp32_to_bf16(2.5f);
            }
            for (size_t i = 0; i < B_bf16.size(); ++i)
            {
                B_bf16[i] = llaminar2::simd::fp32_to_bf16(1.5f);
            }

            bool result = kernel_->multiply_activations_typed_impl(
                A_bf16.data(), B_bf16.data(), C.data(),
                m, n, k, false, 1.0f, 0.0f, &mpi_ctx, device_idx,
                ActivationFormat::BF16, ActivationFormat::BF16);

            ASSERT_TRUE(result);

            // Verify correctness
            float expected = k * 2.5f * 1.5f;
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    EXPECT_NEAR(expected, C[i * n + j], 1.0f)
                        << "Mismatch at C[" << i << "," << j << "]";
                }
            }
        }
    }

    /**
     * @brief Test thread-local scratch buffer isolation
     *
     * Verifies that each thread gets its own scratch buffer instance
     * and there's no contention or corruption between threads.
     */
    TEST_F(OneDNNGemmKernelScratchBuffers, ThreadLocalIsolation)
    {
        const int num_threads = 4;
        const int device_idx = -1; // Default device (CPU)
        std::vector<std::thread> threads;
        std::vector<bool> results(num_threads, false);

        for (int t = 0; t < num_threads; ++t)
        {
            threads.emplace_back([this, t, &results, device_idx]()
                                 {
                llaminar2::MPIContext mpi_ctx(0, 1, MPI_COMM_WORLD);

                // Each thread uses different matrix sizes to ensure isolation
                const int m = 8 + t * 4;
                const int n = 16 + t * 8;
                const int k = 12 + t * 6;

                std::vector<float> A(m * k, static_cast<float>(t + 1));
                std::vector<float> B(k * n, static_cast<float>(t + 2));
                std::vector<float> C(m * n, 0.0f);

                const int lda = k + t + 1;
                const int ldb = n + t + 2;
                const int ldc = n + t + 1;

                std::vector<float> A_strided(m * lda, 0.0f);
                std::vector<float> B_strided(k * ldb, 0.0f);
                std::vector<float> C_strided(m * ldc, 0.0f);

                for (int i = 0; i < m; ++i)
                {
                    std::memcpy(&A_strided[i * lda], &A[i * k], k * sizeof(float));
                }
                for (int i = 0; i < k; ++i)
                {
                    std::memcpy(&B_strided[i * ldb], &B[i * n], n * sizeof(float));
                }

                bool success = kernel_->multiply_activations_strided(
                    A_strided.data(), B_strided.data(), C_strided.data(),
                    m, n, k, lda, ldb, ldc, false, 1.0f, 0.0f, &mpi_ctx, device_idx);

                if (success)
                {
                    // Verify correctness
                    float expected = k * static_cast<float>(t + 1) * static_cast<float>(t + 2);
                    bool correct = true;
                    for (int i = 0; i < m && correct; ++i)
                    {
                        for (int j = 0; j < n && correct; ++j)
                        {
                            float actual = C_strided[i * ldc + j];
                            if (std::abs(actual - expected) > 1e-2f)
                            {
                                correct = false;
                            }
                        }
                    }
                    results[t] = correct;
                } });
        }

        for (auto &thread : threads)
        {
            thread.join();
        }

        // Verify all threads succeeded
        for (int t = 0; t < num_threads; ++t)
        {
            EXPECT_TRUE(results[t]) << "Thread " << t << " failed";
        }
    }

    /**
     * @brief Test strided typed GEMM buffer usage
     *
     * Verifies that strided typed GEMM operations correctly use
     * scratch buffers for byte-level copies.
     */
    TEST_F(OneDNNGemmKernelScratchBuffers, StridedTypedGemmBuffers)
    {
        const int device_idx = -1; // Default device (CPU)
        llaminar2::MPIContext mpi_ctx(0, 1, MPI_COMM_WORLD);

        const int m = 16, n = 32, k = 24;

        // Test with FP16 and stride
        {
            std::vector<uint16_t> A_fp16(m * k);
            std::vector<uint16_t> B_fp16(k * n);
            std::vector<float> C(m * n, 0.0f);

            for (size_t i = 0; i < A_fp16.size(); ++i)
            {
                A_fp16[i] = llaminar2::fp32_to_fp16(1.0f);
            }
            for (size_t i = 0; i < B_fp16.size(); ++i)
            {
                B_fp16[i] = llaminar2::fp32_to_fp16(1.0f);
            }

            const int lda = k + 5;
            const int ldb = n + 3;
            const int ldc = n + 2;

            std::vector<uint16_t> A_strided(m * lda, 0);
            std::vector<uint16_t> B_strided(k * ldb, 0);
            std::vector<float> C_strided(m * ldc, 0.0f);

            for (int i = 0; i < m; ++i)
            {
                std::memcpy(&A_strided[i * lda], &A_fp16[i * k], k * sizeof(uint16_t));
            }
            for (int i = 0; i < k; ++i)
            {
                std::memcpy(&B_strided[i * ldb], &B_fp16[i * n], n * sizeof(uint16_t));
            }

            bool result = kernel_->multiply_activations_strided_typed_impl(
                A_strided.data(), B_strided.data(), C_strided.data(),
                m, n, k, lda, ldb, ldc, false, 1.0f, 0.0f, &mpi_ctx, device_idx,
                ActivationFormat::FP16, ActivationFormat::FP16);

            ASSERT_TRUE(result);

            // Verify result
            float expected = static_cast<float>(k);
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    EXPECT_NEAR(expected, C_strided[i * ldc + j], 0.5f)
                        << "Mismatch at C[" << i << "," << j << "]";
                }
            }
        }
    }

    /**
     * @brief Test softmax with strided GEMM scratch buffers
     */
    TEST_F(OneDNNGemmKernelScratchBuffers, SoftmaxStridedGemmBuffers)
    {
        const int device_idx = -1; // Default device (CPU)
        llaminar2::MPIContext mpi_ctx(0, 1, MPI_COMM_WORLD);

        const int m = 8, n = 16, k = 12;
        std::vector<uint16_t> A_fp16(m * k);
        std::vector<uint16_t> B_fp16(k * n);
        std::vector<float> C(m * n, 0.0f);

        // Initialize with varying values
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < k; ++j)
            {
                float val = static_cast<float>(i + j) / 10.0f;
                A_fp16[i * k + j] = llaminar2::fp32_to_fp16(val);
            }
        }
        for (int i = 0; i < k; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                float val = static_cast<float>(i - j) / 10.0f;
                B_fp16[i * n + j] = llaminar2::fp32_to_fp16(val);
            }
        }

        const int lda = k + 3;
        const int ldb = n + 2;
        const int ldc = n + 1;

        std::vector<uint16_t> A_strided(m * lda, 0);
        std::vector<uint16_t> B_strided(k * ldb, 0);
        std::vector<float> C_strided(m * ldc, 0.0f);

        for (int i = 0; i < m; ++i)
        {
            std::memcpy(&A_strided[i * lda], &A_fp16[i * k], k * sizeof(uint16_t));
        }
        for (int i = 0; i < k; ++i)
        {
            std::memcpy(&B_strided[i * ldb], &B_fp16[i * n], n * sizeof(uint16_t));
        }

        bool result = kernel_->multiply_with_softmax_strided_typed_impl(
            A_strided.data(), B_strided.data(), C_strided.data(),
            m, n, k, lda, ldb, ldc, 1.0f, false, 1, nullptr, false,
            &mpi_ctx, device_idx,
            ActivationFormat::FP16, ActivationFormat::FP16);

        ASSERT_TRUE(result);

        // Verify softmax properties (rows sum to 1.0)
        for (int i = 0; i < m; ++i)
        {
            float row_sum = 0.0f;
            for (int j = 0; j < n; ++j)
            {
                float val = C_strided[i * ldc + j];
                EXPECT_GE(val, 0.0f) << "Softmax output should be non-negative";
                EXPECT_LE(val, 1.0f) << "Softmax output should be <= 1.0";
                row_sum += val;
            }
            EXPECT_NEAR(1.0f, row_sum, 1e-4f)
                << "Row " << i << " should sum to 1.0 after softmax";
        }
    }

    /**
     * @brief Stress test with rapid buffer size changes
     *
     * Tests that buffers handle rapid size changes correctly
     * without memory corruption or performance degradation.
     */
    TEST_F(OneDNNGemmKernelScratchBuffers, RapidSizeChanges)
    {
        const int device_idx = -1; // Default device (CPU)
        llaminar2::MPIContext mpi_ctx(0, 1, MPI_COMM_WORLD);

        // Sequence of matrix sizes that alternate between small and large
        std::vector<std::tuple<int, int, int>> sizes = {
            {8, 16, 12},     // Small
            {64, 128, 96},   // Large
            {4, 8, 6},       // Very small
            {128, 256, 192}, // Very large
            {16, 32, 24},    // Medium
            {32, 64, 48},    // Medium-large
            {2, 4, 3},       // Tiny
            {256, 512, 384}  // Huge
        };

        for (const auto &[m, n, k] : sizes)
        {
            std::vector<float> A(m * k, 1.0f);
            std::vector<float> B(k * n, 1.0f);
            std::vector<float> C(m * n, 0.0f);

            const int lda = k + 2;
            const int ldb = n + 1;
            const int ldc = n + 1;

            std::vector<float> A_strided(m * lda, 0.0f);
            std::vector<float> B_strided(k * ldb, 0.0f);
            std::vector<float> C_strided(m * ldc, 0.0f);

            for (int i = 0; i < m; ++i)
            {
                std::memcpy(&A_strided[i * lda], &A[i * k], k * sizeof(float));
            }
            for (int i = 0; i < k; ++i)
            {
                std::memcpy(&B_strided[i * ldb], &B[i * n], n * sizeof(float));
            }

            bool result = kernel_->multiply_activations_strided(
                A_strided.data(), B_strided.data(), C_strided.data(),
                m, n, k, lda, ldb, ldc, false, 1.0f, 0.0f, &mpi_ctx, device_idx);

            ASSERT_TRUE(result) << "Failed for size (" << m << "x" << n << "x" << k << ")";

            // Verify correctness
            float expected = static_cast<float>(k);
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    EXPECT_NEAR(expected, C_strided[i * ldc + j], 1e-3f)
                        << "Mismatch at C[" << i << "," << j << "] for size ("
                        << m << "x" << n << "x" << k << ")";
                }
            }
        }
    }

} // anonymous namespace

#endif // HAVE_ONEDNN
