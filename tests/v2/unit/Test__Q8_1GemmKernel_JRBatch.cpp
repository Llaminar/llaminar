/**
 * @file Test__Q8_1GemmKernel_JRBatch.cpp
 * @brief Unit tests for Q8_1 GEMM JR_BATCH parameterization
 * @author David Sanftenberg
 *
 * Tests all supported JR_BATCH values: 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20
 * Validates correctness across different batch sizes to catch register pressure issues
 * and ensure proper vectorization at all batch sizes.
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>
#include "kernels/cpu/gemm_v2/Q8_1GemmKernel.h"
#include "tensors/Tensors.h"

using namespace llaminar2;

namespace
{

    /**
     * @brief Helper to create FP32 test tensor with random values
     */
    std::shared_ptr<FP32Tensor> create_random_fp32(const std::vector<size_t> &shape, float min_val = -1.0f, float max_val = 1.0f)
    {
        auto tensor = std::make_shared<FP32Tensor>(shape);
        std::mt19937 gen(42); // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dist(min_val, max_val);

        // FP32Tensor::data() returns const float* in const context, but we need to modify
        // Access raw data via mutable getter
        size_t total = 1;
        for (auto dim : shape)
            total *= dim;

        // Use const_cast since we know we're modifying our own tensor
        float *data = const_cast<float *>(tensor->data());
        for (size_t i = 0; i < total; ++i)
        {
            data[i] = dist(gen);
        }

        return tensor;
    }

    /**
     * @brief Helper to create Q8_0 test tensor from FP32
     */
    std::shared_ptr<Q8_0Tensor> create_q8_0_from_fp32(const FP32Tensor &fp32_tensor)
    {
        const auto &shape = fp32_tensor.shape();

        // Calculate required raw_data size for Q8_0
        // Q8_0 block: 34 bytes (2 byte scale + 32 bytes quants)
        size_t K = shape[0];
        size_t N = shape[1];
        size_t K_blocks = (K + 31) / 32;          // Round up to block boundary
        size_t raw_data_size = K_blocks * N * 34; // 34 bytes per block

        // Allocate properly sized raw_data
        std::vector<uint8_t> raw_data(raw_data_size, 0);
        auto q8_0 = std::make_shared<Q8_0Tensor>(shape, raw_data);

        // copyFrom expects TensorBase* and will populate the quantized data
        q8_0->copyFrom(&fp32_tensor);

        return q8_0;
    }

    /**
     * @brief Compare two FP32 matrices with relative tolerance
     */
    void compare_matrices(const float *expected, const float *actual, int M, int N,
                          float rel_tol = 1e-3f, float abs_tol = 1e-5f,
                          const std::string &context = "")
    {
        float max_rel_error = 0.0f;
        float max_abs_error = 0.0f;
        int error_count = 0;
        const int max_errors_to_print = 5;

        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                int idx = i * N + j;
                float exp = expected[idx];
                float act = actual[idx];
                float abs_error = std::abs(exp - act);
                float rel_error = abs_error / (std::abs(exp) + 1e-10f);

                max_abs_error = std::max(max_abs_error, abs_error);
                max_rel_error = std::max(max_rel_error, rel_error);

                if (rel_error > rel_tol && abs_error > abs_tol)
                {
                    if (error_count < max_errors_to_print)
                    {
                        std::cout << context << " Mismatch at [" << i << "," << j << "]: "
                                  << "expected=" << exp << ", actual=" << act
                                  << ", rel_error=" << rel_error << ", abs_error=" << abs_error << std::endl;
                    }
                    error_count++;
                }
            }
        }

        if (error_count > 0)
        {
            std::cout << context << " Total errors: " << error_count << "/" << (M * N)
                      << ", max_rel_error=" << max_rel_error
                      << ", max_abs_error=" << max_abs_error << std::endl;
        }

        EXPECT_EQ(error_count, 0) << context << " Found " << error_count << " mismatches";
    }

    /**
     * @brief Test fixture for Q8_1 GEMM JR_BATCH tests
     */
    class Q8_1GemmKernel_JRBatch_Test : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Common test dimensions
            M = 32;
            N = 128; // Works with all JR_BATCH values (tail handling tests the remainder)
            K = 896; // Qwen 2.5 0.5B d_model (multiple of 32)

            // Create test data
            A_fp32 = create_random_fp32({static_cast<size_t>(M), static_cast<size_t>(K)}, -0.5f, 0.5f);
            B_fp32 = create_random_fp32({static_cast<size_t>(K), static_cast<size_t>(N)}, -0.5f, 0.5f);
            B_q8_0 = create_q8_0_from_fp32(*B_fp32);

            // Allocate output buffers
            C_reference.resize(M * N);
            C_test.resize(M * N);
        }

        void TearDown() override
        {
            // Cleanup handled by shared_ptr and vector destructors
        }

        // Test dimensions
        int M, N, K;

        // Test data
        std::shared_ptr<FP32Tensor> A_fp32;
        std::shared_ptr<FP32Tensor> B_fp32;
        std::shared_ptr<Q8_0Tensor> B_q8_0;

        // Output buffers
        std::vector<float> C_reference;
        std::vector<float> C_test;
    };

    // ====================================================================
    // CORRECTNESS TESTS: All JR_BATCH values
    // ====================================================================

    TEST_F(Q8_1GemmKernel_JRBatch_Test, JRBatch1_Correctness)
    {
        // JR_BATCH=1: Single column at a time (minimal batching)
        using Kernel_JRB1 = Q8_1GemmKernelTemplate<32, 128, 4, 0, 0, 2, 1>;
        using Kernel_Default = Q8_1GemmKernel; // JR_BATCH=8 (reference)

        // Compute reference with default JR_BATCH=8
        std::fill(C_reference.begin(), C_reference.end(), 0.0f);
        Kernel_Default::gemm(M, N, K, *A_fp32, *B_q8_0, C_reference.data(), N);

        // Compute test with JR_BATCH=1
        std::fill(C_test.begin(), C_test.end(), 0.0f);
        Kernel_JRB1::gemm(M, N, K, *A_fp32, *B_q8_0, C_test.data(), N);

        compare_matrices(C_reference.data(), C_test.data(), M, N, 1e-3f, 1e-5f, "[JR_BATCH=1]");
    }

    TEST_F(Q8_1GemmKernel_JRBatch_Test, JRBatch2_Correctness)
    {
        using Kernel_JRB2 = Q8_1GemmKernelTemplate<32, 128, 4, 0, 0, 2, 2>;
        using Kernel_Default = Q8_1GemmKernel;

        std::fill(C_reference.begin(), C_reference.end(), 0.0f);
        Kernel_Default::gemm(M, N, K, *A_fp32, *B_q8_0, C_reference.data(), N);

        std::fill(C_test.begin(), C_test.end(), 0.0f);
        Kernel_JRB2::gemm(M, N, K, *A_fp32, *B_q8_0, C_test.data(), N);

        compare_matrices(C_reference.data(), C_test.data(), M, N, 1e-3f, 1e-5f, "[JR_BATCH=2]");
    }

    TEST_F(Q8_1GemmKernel_JRBatch_Test, JRBatch4_Correctness)
    {
        using Kernel_JRB4 = Q8_1GemmKernelTemplate<32, 128, 4, 0, 0, 2, 4>;
        using Kernel_Default = Q8_1GemmKernel;

        std::fill(C_reference.begin(), C_reference.end(), 0.0f);
        Kernel_Default::gemm(M, N, K, *A_fp32, *B_q8_0, C_reference.data(), N);

        std::fill(C_test.begin(), C_test.end(), 0.0f);
        Kernel_JRB4::gemm(M, N, K, *A_fp32, *B_q8_0, C_test.data(), N);

        compare_matrices(C_reference.data(), C_test.data(), M, N, 1e-3f, 1e-5f, "[JR_BATCH=4]");
    }

    TEST_F(Q8_1GemmKernel_JRBatch_Test, JRBatch6_Correctness)
    {
        using Kernel_JRB6 = Q8_1GemmKernelTemplate<32, 128, 4, 0, 0, 2, 6>;
        using Kernel_Default = Q8_1GemmKernel;

        std::fill(C_reference.begin(), C_reference.end(), 0.0f);
        Kernel_Default::gemm(M, N, K, *A_fp32, *B_q8_0, C_reference.data(), N);

        std::fill(C_test.begin(), C_test.end(), 0.0f);
        Kernel_JRB6::gemm(M, N, K, *A_fp32, *B_q8_0, C_test.data(), N);

        compare_matrices(C_reference.data(), C_test.data(), M, N, 1e-3f, 1e-5f, "[JR_BATCH=6]");
    }

    TEST_F(Q8_1GemmKernel_JRBatch_Test, JRBatch8_Correctness)
    {
        // JR_BATCH=8: Default (should be identical to reference)
        using Kernel_JRB8 = Q8_1GemmKernelTemplate<32, 128, 4, 0, 0, 2, 8>;
        using Kernel_Default = Q8_1GemmKernel;

        std::fill(C_reference.begin(), C_reference.end(), 0.0f);
        Kernel_Default::gemm(M, N, K, *A_fp32, *B_q8_0, C_reference.data(), N);

        std::fill(C_test.begin(), C_test.end(), 0.0f);
        Kernel_JRB8::gemm(M, N, K, *A_fp32, *B_q8_0, C_test.data(), N);

        compare_matrices(C_reference.data(), C_test.data(), M, N, 1e-6f, 1e-8f, "[JR_BATCH=8]");
    }

    TEST_F(Q8_1GemmKernel_JRBatch_Test, JRBatch10_Correctness)
    {
        using Kernel_JRB10 = Q8_1GemmKernelTemplate<32, 128, 4, 0, 0, 2, 10>;
        using Kernel_Default = Q8_1GemmKernel;

        std::fill(C_reference.begin(), C_reference.end(), 0.0f);
        Kernel_Default::gemm(M, N, K, *A_fp32, *B_q8_0, C_reference.data(), N);

        std::fill(C_test.begin(), C_test.end(), 0.0f);
        Kernel_JRB10::gemm(M, N, K, *A_fp32, *B_q8_0, C_test.data(), N);

        compare_matrices(C_reference.data(), C_test.data(), M, N, 1e-3f, 1e-5f, "[JR_BATCH=10]");
    }

    TEST_F(Q8_1GemmKernel_JRBatch_Test, JRBatch12_Correctness)
    {
        using Kernel_JRB12 = Q8_1GemmKernelTemplate<32, 128, 4, 0, 0, 2, 12>;
        using Kernel_Default = Q8_1GemmKernel;

        std::fill(C_reference.begin(), C_reference.end(), 0.0f);
        Kernel_Default::gemm(M, N, K, *A_fp32, *B_q8_0, C_reference.data(), N);

        std::fill(C_test.begin(), C_test.end(), 0.0f);
        Kernel_JRB12::gemm(M, N, K, *A_fp32, *B_q8_0, C_test.data(), N);

        compare_matrices(C_reference.data(), C_test.data(), M, N, 1e-3f, 1e-5f, "[JR_BATCH=12]");
    }

    TEST_F(Q8_1GemmKernel_JRBatch_Test, JRBatch14_Correctness)
    {
        using Kernel_JRB14 = Q8_1GemmKernelTemplate<32, 128, 4, 0, 0, 2, 14>;
        using Kernel_Default = Q8_1GemmKernel;

        std::fill(C_reference.begin(), C_reference.end(), 0.0f);
        Kernel_Default::gemm(M, N, K, *A_fp32, *B_q8_0, C_reference.data(), N);

        std::fill(C_test.begin(), C_test.end(), 0.0f);
        Kernel_JRB14::gemm(M, N, K, *A_fp32, *B_q8_0, C_test.data(), N);

        compare_matrices(C_reference.data(), C_test.data(), M, N, 1e-3f, 1e-5f, "[JR_BATCH=14]");
    }

    TEST_F(Q8_1GemmKernel_JRBatch_Test, JRBatch16_Correctness)
    {
        using Kernel_JRB16 = Q8_1GemmKernelTemplate<32, 128, 4, 0, 0, 2, 16>;
        using Kernel_Default = Q8_1GemmKernel;

        std::fill(C_reference.begin(), C_reference.end(), 0.0f);
        Kernel_Default::gemm(M, N, K, *A_fp32, *B_q8_0, C_reference.data(), N);

        std::fill(C_test.begin(), C_test.end(), 0.0f);
        Kernel_JRB16::gemm(M, N, K, *A_fp32, *B_q8_0, C_test.data(), N);

        compare_matrices(C_reference.data(), C_test.data(), M, N, 1e-3f, 1e-5f, "[JR_BATCH=16]");
    }

    TEST_F(Q8_1GemmKernel_JRBatch_Test, JRBatch18_Correctness)
    {
        using Kernel_JRB18 = Q8_1GemmKernelTemplate<32, 128, 4, 0, 0, 2, 18>;
        using Kernel_Default = Q8_1GemmKernel;

        std::fill(C_reference.begin(), C_reference.end(), 0.0f);
        Kernel_Default::gemm(M, N, K, *A_fp32, *B_q8_0, C_reference.data(), N);

        std::fill(C_test.begin(), C_test.end(), 0.0f);
        Kernel_JRB18::gemm(M, N, K, *A_fp32, *B_q8_0, C_test.data(), N);

        compare_matrices(C_reference.data(), C_test.data(), M, N, 1e-3f, 1e-5f, "[JR_BATCH=18]");
    }

    TEST_F(Q8_1GemmKernel_JRBatch_Test, JRBatch20_Correctness)
    {
        using Kernel_JRB20 = Q8_1GemmKernelTemplate<32, 128, 4, 0, 0, 2, 20>;
        using Kernel_Default = Q8_1GemmKernel;

        std::fill(C_reference.begin(), C_reference.end(), 0.0f);
        Kernel_Default::gemm(M, N, K, *A_fp32, *B_q8_0, C_reference.data(), N);

        std::fill(C_test.begin(), C_test.end(), 0.0f);
        Kernel_JRB20::gemm(M, N, K, *A_fp32, *B_q8_0, C_test.data(), N);

        compare_matrices(C_reference.data(), C_test.data(), M, N, 1e-3f, 1e-5f, "[JR_BATCH=20]");
    }

    // ====================================================================
    // EDGE CASE TESTS
    // ====================================================================

    TEST_F(Q8_1GemmKernel_JRBatch_Test, SmallN_JRBatch1)
    {
        // N=8: Only one batch iteration per row (8/8=1)
        // For JR_BATCH=1: 8 batch iterations (8/1=8)
        int small_N = 8;
        auto B_small_fp32 = create_random_fp32({static_cast<size_t>(K), static_cast<size_t>(small_N)});
        auto B_small_q8_0 = create_q8_0_from_fp32(*B_small_fp32);

        std::vector<float> C_ref(M * small_N);
        std::vector<float> C_test_val(M * small_N);

        using Kernel_JRB1 = Q8_1GemmKernelTemplate<32, 128, 4, 0, 0, 2, 1>;
        using Kernel_Default = Q8_1GemmKernel;

        Kernel_Default::gemm(M, small_N, K, *A_fp32, *B_small_q8_0, C_ref.data(), small_N);
        Kernel_JRB1::gemm(M, small_N, K, *A_fp32, *B_small_q8_0, C_test_val.data(), small_N);

        compare_matrices(C_ref.data(), C_test_val.data(), M, small_N, 1e-3f, 1e-5f, "[SmallN_JRB1]");
    }

    TEST_F(Q8_1GemmKernel_JRBatch_Test, LargeK_JRBatch20)
    {
        // Large K to test register pressure with JR_BATCH=20
        int large_K = 4096; // Qwen 2.5 7B d_model
        auto A_large = create_random_fp32({static_cast<size_t>(M), static_cast<size_t>(large_K)});
        auto B_large_fp32 = create_random_fp32({static_cast<size_t>(large_K), static_cast<size_t>(N)});
        auto B_large_q8_0 = create_q8_0_from_fp32(*B_large_fp32);

        std::vector<float> C_ref(M * N);
        std::vector<float> C_test_val(M * N);

        using Kernel_JRB20 = Q8_1GemmKernelTemplate<32, 128, 4, 0, 0, 2, 20>;
        using Kernel_Default = Q8_1GemmKernel;

        Kernel_Default::gemm(M, N, large_K, *A_large, *B_large_q8_0, C_ref.data(), N);
        Kernel_JRB20::gemm(M, N, large_K, *A_large, *B_large_q8_0, C_test_val.data(), N);

        compare_matrices(C_ref.data(), C_test_val.data(), M, N, 1e-3f, 1e-5f, "[LargeK_JRB20]");
    }

    // ====================================================================
    // STATIC ASSERT VALIDATION (Compile-time tests)
    // ====================================================================

    /**
     * @brief Test that JR_BATCH <= NR constraint is enforced at compile time
     *
     * These tests verify the static_assert catches invalid configurations.
     * They are compile-time tests - if they compile, the test passes.
     * If JR_BATCH > NR, compilation should fail with static_assert error.
     */

    // Valid: JR_BATCH == NR (boundary case)
    TEST_F(Q8_1GemmKernel_JRBatch_Test, StaticAssert_JRBatch_Equal_NR)
    {
        // NR=8, JR_BATCH=8: Should compile (JR_BATCH == NR is valid)
        using Kernel_Valid = Q8_1GemmKernelTemplate<32, 8, 4, 0, 0, 2, 8>;

        int small_N = 8;
        auto B_small_fp32 = create_random_fp32({static_cast<size_t>(K), static_cast<size_t>(small_N)});
        auto B_small_q8_0 = create_q8_0_from_fp32(*B_small_fp32);

        std::vector<float> C_test_val(M * small_N);
        Kernel_Valid::gemm(M, small_N, K, *A_fp32, *B_small_q8_0, C_test_val.data(), small_N);

        // Test passes if it compiles
        SUCCEED();
    }

    // Valid: JR_BATCH < NR (normal case)
    TEST_F(Q8_1GemmKernel_JRBatch_Test, StaticAssert_JRBatch_Less_Than_NR)
    {
        // NR=32, JR_BATCH=8: Should compile (JR_BATCH < NR)
        using Kernel_Valid = Q8_1GemmKernelTemplate<32, 32, 4, 0, 0, 2, 8>;

        int small_N = 32;
        auto B_small_fp32 = create_random_fp32({static_cast<size_t>(K), static_cast<size_t>(small_N)});
        auto B_small_q8_0 = create_q8_0_from_fp32(*B_small_fp32);

        std::vector<float> C_test_val(M * small_N);
        Kernel_Valid::gemm(M, small_N, K, *A_fp32, *B_small_q8_0, C_test_val.data(), small_N);

        // Test passes if it compiles
        SUCCEED();
    }

    // The following would FAIL to compile (static_assert violation):
    //
    // TEST_F(Q8_1GemmKernel_JRBatch_Test, StaticAssert_JRBatch_Greater_Than_NR) {
    //     // NR=8, JR_BATCH=16: Should NOT compile (JR_BATCH > NR)
    //     using Kernel_Invalid = Q8_1GemmKernelTemplate<32, 8, 4, 0, 0, 2, 16>;
    //     // Compilation error: static assertion failed: JR_BATCH must be less than or equal to NR
    // }

} // anonymous namespace
