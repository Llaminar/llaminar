/**
 * @file Test__MicroKernelCorrectness.cpp
 * @brief Mathematical correctness tests for micro-kernel GEMM
 *
 * Validates that micro-kernel variants produce numerically correct results
 * by comparing against:
 *   1. Reference single-threaded implementation
 *   2. OpenBLAS cblas_sgemm (ground truth)
 *   3. Known analytical test cases
 *
 * Test coverage:
 *   - Identity matrices
 *   - Zero matrices
 *   - Diagonal matrices
 *   - Random matrices (various sizes)
 *   - Edge cases (1×1, non-multiples of tile size)
 *   - Numerical stability (condition numbers)
 *   - All tile size variants
 *
 * @author David Sanftenberg
 * @date October 2025
 */

#include <gtest/gtest.h>
#include <cblas.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "kernels/cpu/GemmAutoTuner.h"
#include "kernels/cpu/GemmMicroKernelAdapter.h"
#include "tensors/TensorKernels.h"

using namespace llaminar2;
using namespace llaminar::v2::kernels;

namespace {

/**
 * @brief Test block decoder for correctness testing
 *
 * Wraps a pre-computed FP32 matrix as a "quantized" decoder.
 * The input matrix is k×n row-major (k rows, n columns).
 * The decoder represents it as n×k (n output features, k input features each).
 * decode_block_at() extracts column `row_idx` from the k×n matrix.
 *
 * @param data Pointer to k×n matrix stored row-major
 * @param k_rows Number of rows in the input matrix (k dimension)
 * @param n_cols Number of columns in the input matrix (n dimension)
 */
class FP32BlockDecoder : public IBlockDecoder {
public:
    FP32BlockDecoder(const float* data, int k_rows, int n_cols)
        : data_(data), k_rows_(k_rows), n_cols_(n_cols) {
        blocks_per_feature_ = (k_rows + 31) / 32;
    }

    __attribute__((always_inline))
    void decode_block_at(size_t row_idx, size_t k_block_offset, float* output) const override {
        // row_idx is the output feature index (column in the k×n matrix)
        // k_block_offset is which 32-element block along k dimension
        size_t k_start = k_block_offset * 32;
        size_t k_end = std::min(k_start + 32, static_cast<size_t>(k_rows_));
        
        // Extract column row_idx from the k×n matrix
        // Elements are at: data[i * n_cols + row_idx] for i in [k_start, k_end)
        for (size_t i = k_start; i < k_end; ++i) {
            output[i - k_start] = data_[i * n_cols_ + row_idx];
        }
        
        // Pad remaining with zeros
        for (size_t i = k_end - k_start; i < 32; ++i) {
            output[i] = 0.0f;
        }
    }

    const void* get_raw_block_at(size_t row_idx, size_t k_block_offset) const override {
        return nullptr;
    }

    size_t decoder_rows() const override { return n_cols_; }  // n output features
    size_t decoder_cols() const override { return k_rows_; }  // k input features each
    size_t block_size() const override { return 32; }

private:
    const float* data_;      // Pointer to k×n matrix (row-major)
    int k_rows_;             // Number of rows in input matrix (k)
    int n_cols_;             // Number of columns in input matrix (n)
    size_t blocks_per_feature_;
};

/**
 * @brief Reference GEMM implementation (naive, single-threaded, correct)
 */
void reference_gemm(const float* A, const float* B, float* C,
                   int m, int n, int k) {
    // C = A * B
    // A: m × k
    // B: k × n (row-major, so each row has n elements)
    // C: m × n
    
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

/**
 * @brief Compute relative L2 error between two matrices
 */
double relative_l2_error(const float* computed, const float* expected,
                        int m, int n) {
    double num = 0.0;
    double denom = 0.0;
    
    for (int i = 0; i < m * n; ++i) {
        double diff = computed[i] - expected[i];
        num += diff * diff;
        denom += expected[i] * expected[i];
    }
    
    if (denom < 1e-20) {
        // Expected is near zero, use absolute error instead
        return std::sqrt(num);
    }
    
    return std::sqrt(num / denom);
}

/**
 * @brief Compute maximum absolute difference between two matrices
 */
float max_abs_diff(const float* computed, const float* expected,
                  int m, int n) {
    float max_diff = 0.0f;
    for (int i = 0; i < m * n; ++i) {
        float diff = std::abs(computed[i] - expected[i]);
        max_diff = std::max(max_diff, diff);
    }
    return max_diff;
}

/**
 * @brief Fill matrix with identity pattern
 */
void fill_identity(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
}

/**
 * @brief Fill matrix with diagonal values
 */
void fill_diagonal(float* matrix, int rows, int cols, float value) {
    std::fill(matrix, matrix + rows * cols, 0.0f);
    int min_dim = std::min(rows, cols);
    for (int i = 0; i < min_dim; ++i) {
        matrix[i * cols + i] = value;
    }
}

/**
 * @brief Fill matrix with random values
 */
void fill_random(float* matrix, int rows, int cols, float min_val, float max_val) {
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = dis(gen);
    }
}

} // anonymous namespace

/**
 * @brief Test fixture for micro-kernel correctness
 */
class MicroKernelCorrectness : public ::testing::Test {
protected:
    void SetUp() override {
        // Clear auto-tuner cache before each test
        auto &tuner = GemmAutoTuner::instance();
        tuner.clearCache();
    }
};

/**
 * @brief Test: Identity matrix multiplication
 *
 * A = I, B = random => C should equal B
 */
TEST_F(MicroKernelCorrectness, IdentityMatrix) {
    const int m = 64;
    const int n = 896;
    const int k = 896;
    
    std::vector<float> A(m * k);
    std::vector<float> B(k * n);
    std::vector<float> C_micro(m * n, 0.0f);
    std::vector<float> C_ref(m * n, 0.0f);
    
    // A = identity, B = random
    fill_identity(A.data(), m, k);
    fill_random(B.data(), k, n, -1.0f, 1.0f);
    
    // Reference: C = I * B = B
    reference_gemm(A.data(), B.data(), C_ref.data(), m, n, k);
    
    // Micro-kernel (B is k×n row-major, decoder sees it as n×k)
    FP32BlockDecoder decoder(B.data(), k, n);
    auto &tuner = GemmAutoTuner::instance();
    auto variants = kernels::gemm::registerMicroKernelVariants(&decoder);
    for (auto &variant : variants) {
        tuner.registerVariant(std::move(variant));
    }
    
    auto *kernel = tuner.getOptimalKernel(m, n, k);
    ASSERT_NE(kernel, nullptr);
    
    bool success = kernel->multiply(A.data(), C_micro.data(), m, n, k, &decoder, 1.0f, 0.0f);
    ASSERT_TRUE(success);
    
    // Verify correctness
    double rel_error = relative_l2_error(C_micro.data(), C_ref.data(), m, n);
    float max_diff = max_abs_diff(C_micro.data(), C_ref.data(), m, n);
    
    std::cout << "IdentityMatrix: rel_error=" << rel_error << ", max_diff=" << max_diff << std::endl;
    
    EXPECT_LT(rel_error, 1e-5) << "Relative L2 error too high";
    EXPECT_LT(max_diff, 1e-4) << "Max absolute difference too high";
}

/**
 * @brief Test: Zero matrix multiplication
 *
 * A = 0 => C should be all zeros
 */
TEST_F(MicroKernelCorrectness, ZeroMatrix) {
    const int m = 32;
    const int n = 896;
    const int k = 896;
    
    std::vector<float> A(m * k, 0.0f);
    std::vector<float> B(k * n);
    std::vector<float> C_micro(m * n, 0.0f);
    
    fill_random(B.data(), k, n, -1.0f, 1.0f);
    
    FP32BlockDecoder decoder(B.data(), k, n);
    auto &tuner = GemmAutoTuner::instance();
    auto variants = kernels::gemm::registerMicroKernelVariants(&decoder);
    for (auto &variant : variants) {
        tuner.registerVariant(std::move(variant));
    }
    
    auto *kernel = tuner.getOptimalKernel(m, n, k);
    ASSERT_NE(kernel, nullptr);
    
    bool success = kernel->multiply(A.data(), C_micro.data(), m, n, k, &decoder, 1.0f, 0.0f);
    ASSERT_TRUE(success);
    
    // All values should be zero (or very close)
    for (int i = 0; i < m * n; ++i) {
        EXPECT_NEAR(C_micro[i], 0.0f, 1e-6) << "Non-zero value at index " << i;
    }
}

/**
 * @brief Test: Diagonal matrix multiplication
 *
 * A = diag(2), B = diag(3) => C should be diag(6)
 */
TEST_F(MicroKernelCorrectness, DiagonalMatrix) {
    const int m = 128;
    const int n = 128;
    const int k = 128;
    
    std::vector<float> A(m * k);
    std::vector<float> B(k * n);
    std::vector<float> C_micro(m * n, 0.0f);
    std::vector<float> C_expected(m * n);
    
    fill_diagonal(A.data(), m, k, 2.0f);
    fill_diagonal(B.data(), k, n, 3.0f);
    fill_diagonal(C_expected.data(), m, n, 6.0f);
    
    FP32BlockDecoder decoder(B.data(), k, n);
    auto &tuner = GemmAutoTuner::instance();
    auto variants = kernels::gemm::registerMicroKernelVariants(&decoder);
    for (auto &variant : variants) {
        tuner.registerVariant(std::move(variant));
    }
    
    auto *kernel = tuner.getOptimalKernel(m, n, k);
    ASSERT_NE(kernel, nullptr);
    
    bool success = kernel->multiply(A.data(), C_micro.data(), m, n, k, &decoder, 1.0f, 0.0f);
    ASSERT_TRUE(success);
    
    double rel_error = relative_l2_error(C_micro.data(), C_expected.data(), m, n);
    EXPECT_LT(rel_error, 1e-5) << "Relative L2 error too high for diagonal matrix";
}

/**
 * @brief Test: Random matrices vs reference implementation
 */
TEST_F(MicroKernelCorrectness, RandomMatricesSmall) {
    const int m = 32;
    const int n = 64;
    const int k = 64;
    
    std::vector<float> A(m * k);
    std::vector<float> B(k * n);
    std::vector<float> C_micro(m * n, 0.0f);
    std::vector<float> C_ref(m * n, 0.0f);
    
    fill_random(A.data(), m, k, -1.0f, 1.0f);
    fill_random(B.data(), k, n, -1.0f, 1.0f);
    
    // Reference
    reference_gemm(A.data(), B.data(), C_ref.data(), m, n, k);
    
    // Micro-kernel
    FP32BlockDecoder decoder(B.data(), k, n);
    auto &tuner = GemmAutoTuner::instance();
    auto variants = kernels::gemm::registerMicroKernelVariants(&decoder);
    for (auto &variant : variants) {
        tuner.registerVariant(std::move(variant));
    }
    
    auto *kernel = tuner.getOptimalKernel(m, n, k);
    ASSERT_NE(kernel, nullptr);
    
    bool success = kernel->multiply(A.data(), C_micro.data(), m, n, k, &decoder, 1.0f, 0.0f);
    ASSERT_TRUE(success);
    
    double rel_error = relative_l2_error(C_micro.data(), C_ref.data(), m, n);
    float max_diff = max_abs_diff(C_micro.data(), C_ref.data(), m, n);
    
    EXPECT_LT(rel_error, 1e-4) << "Relative L2 error too high";
    EXPECT_LT(max_diff, 1e-3) << "Max absolute difference too high";
}

/**
 * @brief Test: Random matrices (medium size) vs reference
 */
TEST_F(MicroKernelCorrectness, RandomMatricesMedium) {
    const int m = 128;
    const int n = 896;
    const int k = 896;
    
    std::vector<float> A(m * k);
    std::vector<float> B(k * n);
    std::vector<float> C_micro(m * n, 0.0f);
    std::vector<float> C_ref(m * n, 0.0f);
    
    fill_random(A.data(), m, k, -1.0f, 1.0f);
    fill_random(B.data(), k, n, -1.0f, 1.0f);
    
    // Reference
    reference_gemm(A.data(), B.data(), C_ref.data(), m, n, k);
    
    // Micro-kernel
    FP32BlockDecoder decoder(B.data(), k, n);
    auto &tuner = GemmAutoTuner::instance();
    auto variants = kernels::gemm::registerMicroKernelVariants(&decoder);
    for (auto &variant : variants) {
        tuner.registerVariant(std::move(variant));
    }
    
    auto *kernel = tuner.getOptimalKernel(m, n, k);
    ASSERT_NE(kernel, nullptr);
    
    bool success = kernel->multiply(A.data(), C_micro.data(), m, n, k, &decoder, 1.0f, 0.0f);
    ASSERT_TRUE(success);
    
    double rel_error = relative_l2_error(C_micro.data(), C_ref.data(), m, n);
    float max_diff = max_abs_diff(C_micro.data(), C_ref.data(), m, n);
    
    EXPECT_LT(rel_error, 1e-4) << "Relative L2 error too high";
    EXPECT_LT(max_diff, 1e-3) << "Max absolute difference too high";
}

/**
 * @brief Test: Random matrices vs OpenBLAS (ground truth)
 */
TEST_F(MicroKernelCorrectness, VsOpenBLAS) {
    const int m = 256;
    const int n = 896;
    const int k = 896;
    
    std::vector<float> A(m * k);
    std::vector<float> B(k * n);
    std::vector<float> C_micro(m * n, 0.0f);
    std::vector<float> C_blas(m * n, 0.0f);
    
    fill_random(A.data(), m, k, -1.0f, 1.0f);
    fill_random(B.data(), k, n, -1.0f, 1.0f);
    
    // OpenBLAS: C = A * B
    // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    //            m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
               m, n, k,
               1.0f, A.data(), k,    // A is m×k
               B.data(), n,          // B is k×n
               0.0f, C_blas.data(), n);
    
    // Micro-kernel
    FP32BlockDecoder decoder(B.data(), k, n);
    auto &tuner = GemmAutoTuner::instance();
    auto variants = kernels::gemm::registerMicroKernelVariants(&decoder);
    for (auto &variant : variants) {
        tuner.registerVariant(std::move(variant));
    }
    
    auto *kernel = tuner.getOptimalKernel(m, n, k);
    ASSERT_NE(kernel, nullptr);
    
    bool success = kernel->multiply(A.data(), C_micro.data(), m, n, k, &decoder, 1.0f, 0.0f);
    ASSERT_TRUE(success);
    
    double rel_error = relative_l2_error(C_micro.data(), C_blas.data(), m, n);
    float max_diff = max_abs_diff(C_micro.data(), C_blas.data(), m, n);
    
    std::cout << "VsOpenBLAS: rel_error=" << rel_error << ", max_diff=" << max_diff << std::endl;
    
    EXPECT_LT(rel_error, 1e-4) << "Relative L2 error vs OpenBLAS too high";
    EXPECT_LT(max_diff, 1e-3) << "Max absolute difference vs OpenBLAS too high";
}

/**
 * @brief Test: Edge case - single row (m=1)
 */
TEST_F(MicroKernelCorrectness, SingleRow) {
    const int m = 1;
    const int n = 896;
    const int k = 896;
    
    std::vector<float> A(m * k);
    std::vector<float> B(k * n);
    std::vector<float> C_micro(m * n, 0.0f);
    std::vector<float> C_ref(m * n, 0.0f);
    
    fill_random(A.data(), m, k, -1.0f, 1.0f);
    fill_random(B.data(), k, n, -1.0f, 1.0f);
    
    reference_gemm(A.data(), B.data(), C_ref.data(), m, n, k);
    
    FP32BlockDecoder decoder(B.data(), k, n);
    auto &tuner = GemmAutoTuner::instance();
    auto variants = kernels::gemm::registerMicroKernelVariants(&decoder);
    for (auto &variant : variants) {
        tuner.registerVariant(std::move(variant));
    }
    
    auto *kernel = tuner.getOptimalKernel(m, n, k);
    ASSERT_NE(kernel, nullptr);
    
    bool success = kernel->multiply(A.data(), C_micro.data(), m, n, k, &decoder, 1.0f, 0.0f);
    ASSERT_TRUE(success);
    
    double rel_error = relative_l2_error(C_micro.data(), C_ref.data(), m, n);
    EXPECT_LT(rel_error, 1e-4) << "Single row correctness failed";
}

/**
 * @brief Test: Edge case - tiny matrix (1×1×1)
 */
TEST_F(MicroKernelCorrectness, TinyMatrix) {
    const int m = 1;
    const int n = 1;
    const int k = 1;
    
    std::vector<float> A = {2.5f};
    std::vector<float> B = {3.0f};
    std::vector<float> C_micro(1, 0.0f);
    float expected = 7.5f; // 2.5 * 3.0
    
    FP32BlockDecoder decoder(B.data(), k, n);
    auto &tuner = GemmAutoTuner::instance();
    auto variants = kernels::gemm::registerMicroKernelVariants(&decoder);
    for (auto &variant : variants) {
        tuner.registerVariant(std::move(variant));
    }
    
    auto *kernel = tuner.getOptimalKernel(m, n, k);
    ASSERT_NE(kernel, nullptr);
    
    bool success = kernel->multiply(A.data(), C_micro.data(), m, n, k, &decoder, 1.0f, 0.0f);
    ASSERT_TRUE(success);
    
    EXPECT_NEAR(C_micro[0], expected, 1e-5) << "1×1×1 multiplication incorrect";
}

/**
 * @brief Test: Non-multiple of tile size
 *
 * Tests that padding/boundary handling works correctly
 */
TEST_F(MicroKernelCorrectness, NonMultipleOfTileSize) {
    const int m = 17;  // Not a multiple of common tile sizes
    const int n = 33;
    const int k = 65;
    
    std::vector<float> A(m * k);
    std::vector<float> B(k * n);
    std::vector<float> C_micro(m * n, 0.0f);
    std::vector<float> C_ref(m * n, 0.0f);
    
    fill_random(A.data(), m, k, -1.0f, 1.0f);
    fill_random(B.data(), k, n, -1.0f, 1.0f);
    
    reference_gemm(A.data(), B.data(), C_ref.data(), m, n, k);
    
    FP32BlockDecoder decoder(B.data(), k, n);
    auto &tuner = GemmAutoTuner::instance();
    auto variants = kernels::gemm::registerMicroKernelVariants(&decoder);
    for (auto &variant : variants) {
        tuner.registerVariant(std::move(variant));
    }
    
    auto *kernel = tuner.getOptimalKernel(m, n, k);
    ASSERT_NE(kernel, nullptr);
    
    bool success = kernel->multiply(A.data(), C_micro.data(), m, n, k, &decoder, 1.0f, 0.0f);
    ASSERT_TRUE(success);
    
    double rel_error = relative_l2_error(C_micro.data(), C_ref.data(), m, n);
    EXPECT_LT(rel_error, 1e-4) << "Non-multiple tile size correctness failed";
}

/**
 * @brief Test: All registered variants produce same result
 *
 * Validates that all tile configurations are mathematically equivalent
 */
TEST_F(MicroKernelCorrectness, AllVariantsAgree) {
    const int m = 64;
    const int n = 896;
    const int k = 896;
    
    std::vector<float> A(m * k);
    std::vector<float> B(k * n);
    std::vector<float> C_reference(m * n, 0.0f);
    
    fill_random(A.data(), m, k, -1.0f, 1.0f);
    fill_random(B.data(), k, n, -1.0f, 1.0f);
    
    // Compute reference
    reference_gemm(A.data(), B.data(), C_reference.data(), m, n, k);
    
    FP32BlockDecoder decoder(B.data(), k, n);
    auto variants = kernels::gemm::registerMicroKernelVariants(&decoder);
    
    // Test each variant individually
    for (size_t i = 0; i < variants.size(); ++i) {
        std::vector<float> C_variant(m * n, 0.0f);
        
        bool success = variants[i]->multiply(A.data(), C_variant.data(), m, n, k, &decoder, 1.0f, 0.0f);
        ASSERT_TRUE(success) << "Variant " << i << " failed to execute";
        
        double rel_error = relative_l2_error(C_variant.data(), C_reference.data(), m, n);
        float max_diff = max_abs_diff(C_variant.data(), C_reference.data(), m, n);
        
        EXPECT_LT(rel_error, 1e-4) << "Variant " << i << " has high relative error";
        EXPECT_LT(max_diff, 1e-3) << "Variant " << i << " has high max diff";
    }
}

/**
 * @brief Test: Large matrix multiplication
 *
 * Stress test with larger dimensions
 */
TEST_F(MicroKernelCorrectness, LargeMatrix) {
    const int m = 1024;
    const int n = 896;
    const int k = 896;
    
    std::vector<float> A(m * k);
    std::vector<float> B(k * n);
    std::vector<float> C_micro(m * n, 0.0f);
    std::vector<float> C_blas(m * n, 0.0f);
    
    fill_random(A.data(), m, k, -0.5f, 0.5f);
    fill_random(B.data(), k, n, -0.5f, 0.5f);
    
    // OpenBLAS reference
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
               m, n, k,
               1.0f, A.data(), k,
               B.data(), n,
               0.0f, C_blas.data(), n);
    
    // Micro-kernel
    FP32BlockDecoder decoder(B.data(), k, n);
    auto &tuner = GemmAutoTuner::instance();
    auto variants = kernels::gemm::registerMicroKernelVariants(&decoder);
    for (auto &variant : variants) {
        tuner.registerVariant(std::move(variant));
    }
    
    auto *kernel = tuner.getOptimalKernel(m, n, k);
    ASSERT_NE(kernel, nullptr);
    
    bool success = kernel->multiply(A.data(), C_micro.data(), m, n, k, &decoder, 1.0f, 0.0f);
    ASSERT_TRUE(success);
    
    double rel_error = relative_l2_error(C_micro.data(), C_blas.data(), m, n);
    
    // Slightly relaxed tolerance for large matrices due to accumulation differences
    EXPECT_LT(rel_error, 5e-4) << "Large matrix relative error too high";
}

/**
 * @brief Test: Numerical stability with varying magnitudes
 */
TEST_F(MicroKernelCorrectness, NumericalStability) {
    const int m = 128;
    const int n = 128;
    const int k = 128;
    
    std::vector<float> A(m * k);
    std::vector<float> B(k * n);
    std::vector<float> C_micro(m * n, 0.0f);
    std::vector<float> C_ref(m * n, 0.0f);
    
    // Mix of large and small values
    fill_random(A.data(), m, k, -100.0f, 100.0f);
    fill_random(B.data(), k, n, -0.01f, 0.01f);
    
    reference_gemm(A.data(), B.data(), C_ref.data(), m, n, k);
    
    FP32BlockDecoder decoder(B.data(), k, n);
    auto &tuner = GemmAutoTuner::instance();
    auto variants = kernels::gemm::registerMicroKernelVariants(&decoder);
    for (auto &variant : variants) {
        tuner.registerVariant(std::move(variant));
    }
    
    auto *kernel = tuner.getOptimalKernel(m, n, k);
    ASSERT_NE(kernel, nullptr);
    
    bool success = kernel->multiply(A.data(), C_micro.data(), m, n, k, &decoder, 1.0f, 0.0f);
    ASSERT_TRUE(success);
    
    double rel_error = relative_l2_error(C_micro.data(), C_ref.data(), m, n);
    EXPECT_LT(rel_error, 1e-3) << "Numerical stability test failed";
}
