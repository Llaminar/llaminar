/**
 * @file Test__GemmAutoTunerCorrectness.cpp
 * @brief Correctness tests for auto-tuned GEMM kernel variants
 *
 * These tests validate that the GEMM autotuner produces numerically
 * correct results compared to a reference implementation.
 *
 * @author David Sanftenberg
 * @date October 2025
 */

#include <gtest/gtest.h>
#include "../../../src/v2/kernels/cpu/GemmAutoTuner.h"
#include "../../../src/v2/tensors/Tensors.h"
#include <random>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <cblas.h>

using namespace llaminar::v2::kernels;
using namespace llaminar2;

/**
 * @brief Mock decoder that wraps FP32 data (no quantization)
 *
 * This decouples GEMM testing from quantization correctness.
 * We test the GEMM kernel logic separately from IQ4_NL decoding.
 */
class MockFP32Decoder : public llaminar2::IBlockDecoder
{
private:
    const float *data_; // FP32 tensor data [rows × cols]
    size_t rows_;
    size_t cols_;
    size_t block_size_;

public:
    MockFP32Decoder(const float *data, size_t rows, size_t cols, size_t block_size = 32)
        : data_(data), rows_(rows), cols_(cols), block_size_(block_size) {}

    __attribute__((always_inline)) void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override
    {
        // Simple copy from FP32 source (no actual decoding)
        const size_t k_start = k_block_offset * block_size_;
        const size_t k_end = std::min(k_start + block_size_, cols_);
        const size_t count = k_end - k_start;

        const size_t offset = row_idx * cols_ + k_start;

        std::memcpy(output, data_ + offset, count * sizeof(float));

        // Zero-pad if partial block
        if (count < block_size_)
        {
            std::memset(output + count, 0, (block_size_ - count) * sizeof(float));
        }
    }

    const void *get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
    {
        return data_ + row_idx * cols_ + k_block_offset * block_size_;
    }

    size_t decoder_rows() const override { return rows_; }
    size_t decoder_cols() const override { return cols_; }
    size_t block_size() const override { return block_size_; }
};

/**
 * @brief Test fixture for GEMM correctness validation
 */
class Test__GemmAutoTunerCorrectness : public ::testing::Test
{
protected:
    std::mt19937 rng_{42}; // Fixed seed for reproducibility

    void SetUp() override
    {
        // Clear any cached auto-tuning results
        auto &tuner = GemmAutoTuner::instance();
        tuner.clearCache();
    }

    /**
     * @brief Reference GEMM using CBLAS (OpenBLAS or MKL)
     *
     * Computes C[m×n] = A[m×k] × B^T[n×k]
     * Note: B is provided in row-major format (same as IQ4_NL tensor layout)
     *
     * Uses cblas_sgemm which is a known-good implementation for validation.
     */
    void referenceGEMM(const float *A, const float *B, float *C, int m, int n, int k)
    {
        // C[m×n] = A[m×k] × B^T[n×k]
        // cblas_sgemm parameters:
        // - Order: CblasRowMajor (matrices stored row-major)
        // - TransA: CblasNoTrans (A is not transposed)
        // - TransB: CblasTrans (B is transposed: we have B[n×k] but need B^T)
        // - M: m (rows of A and C)
        // - N: n (columns of C, rows of B)
        // - K: k (columns of A, columns of B)
        // - alpha: 1.0
        // - A: row-major [m×k], lda = k
        // - B: row-major [n×k], ldb = k
        // - beta: 0.0
        // - C: row-major [m×n], ldc = n

        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans, // A is not transposed
            CblasTrans,   // B is transposed (we have [n×k] but need [k×n])
            m,            // rows of A and C
            n,            // columns of C and rows of B
            k,            // columns of A and rows of B^T
            1.0f,         // alpha
            A, k,         // A[m×k], leading dimension k
            B, k,         // B[n×k], leading dimension k
            0.0f,         // beta (overwrite C)
            C, n          // C[m×n], leading dimension n
        );
    }

    /**
     * @brief Check if two matrices are equal within tolerance
     */
    bool matricesEqual(const float *expected, const float *actual, int size, float tolerance)
    {
        int mismatches = 0;
        const int max_mismatches_to_report = 5;

        for (int i = 0; i < size; ++i)
        {
            float diff = std::abs(expected[i] - actual[i]);
            float rel_diff = (std::abs(expected[i]) > 1e-6f)
                                 ? diff / std::abs(expected[i])
                                 : diff;

            if (diff > tolerance && rel_diff > tolerance)
            {
                if (mismatches < max_mismatches_to_report)
                {
                    std::cout << "Mismatch at index " << i << ": "
                              << "expected " << expected[i] << ", "
                              << "got " << actual[i] << " "
                              << "(diff=" << diff << ", rel=" << rel_diff << ")"
                              << std::endl;
                }
                ++mismatches;
            }
        }

        if (mismatches > 0)
        {
            std::cout << "Total mismatches: " << mismatches << "/" << size << std::endl;
        }

        return mismatches == 0;
    }

    /**
     * @brief Create a test IQ4_NL tensor with random quantized data
     */
    std::shared_ptr<IQ4_NLTensor> createRandomTensor(int rows, int cols)
    {
        // IQ4_NL lookup grid values (must match IQQuantTables.h)
        static constexpr float kvalues_iq4nl[16] = {
            -127.0f / 127.0f, -104.0f / 127.0f, -83.0f / 127.0f, -65.0f / 127.0f,
            -49.0f / 127.0f, -35.0f / 127.0f, -22.0f / 127.0f, -10.0f / 127.0f,
            1.0f / 127.0f, 13.0f / 127.0f, 25.0f / 127.0f, 38.0f / 127.0f,
            53.0f / 127.0f, 69.0f / 127.0f, 89.0f / 127.0f, 113.0f / 127.0f};

        // Create temporary FP32 data
        std::vector<float> temp(rows * cols);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : temp)
            v = dist(rng_);

        // Quantize to IQ4_NL format using actual lookup grid
        const size_t block_size = 32;
        const size_t blocks_per_row = (cols + block_size - 1) / block_size;
        const size_t block_struct_size = 18; // sizeof(IQ4_NLBlock): 16 bytes nibbles + 2 bytes scale (FP16)

        std::vector<uint8_t> raw_data(rows * blocks_per_row * block_struct_size);

        // Quantize using IQ4_NL grid
        for (int r = 0; r < rows; ++r)
        {
            for (size_t b = 0; b < blocks_per_row; ++b)
            {
                size_t block_offset = (r * blocks_per_row + b) * block_struct_size;

                // Find max absolute value for this block
                size_t start_col = b * block_size;
                size_t end_col = std::min(start_col + block_size, (size_t)cols);

                float max_abs = 0.0f;
                for (size_t c = start_col; c < end_col; ++c)
                {
                    max_abs = std::max(max_abs, std::abs(temp[r * cols + c]));
                }

                // Compute scale: max_abs / max_grid_value
                // max grid value is kvalues_iq4nl[15] = 113/127 ≈ 0.8898
                float scale = (max_abs > 0.0f) ? (max_abs / 0.8898f) : 1.0f;

                // Convert scale to FP16 (proper IEEE 754 half precision)
                auto float_to_fp16 = [](float f) -> uint16_t
                {
                    uint32_t f32;
                    std::memcpy(&f32, &f, sizeof(f));

                    uint32_t sign = (f32 >> 16) & 0x8000;
                    int32_t exponent = ((f32 >> 23) & 0xFF) - 127 + 15;
                    uint32_t mantissa = (f32 >> 13) & 0x03FF;

                    if (exponent <= 0)
                    {
                        // Denormal or zero
                        return static_cast<uint16_t>(sign);
                    }
                    else if (exponent >= 31)
                    {
                        // Infinity or overflow
                        return static_cast<uint16_t>(sign | 0x7C00);
                    }

                    return static_cast<uint16_t>(sign | (exponent << 10) | mantissa);
                };

                uint16_t scale_half = float_to_fp16(scale);
                memcpy(&raw_data[block_offset], &scale_half, sizeof(uint16_t));

                // Quantize each element to nearest grid value
                for (size_t c = start_col; c < end_col; ++c)
                {
                    size_t local_idx = c - start_col;
                    float value = temp[r * cols + c];
                    float normalized = (scale > 0.0f) ? (value / scale) : 0.0f;

                    // Find nearest grid value
                    int best_idx = 0;
                    float best_dist = std::abs(normalized - kvalues_iq4nl[0]);
                    for (int i = 1; i < 16; ++i)
                    {
                        float dist = std::abs(normalized - kvalues_iq4nl[i]);
                        if (dist < best_dist)
                        {
                            best_dist = dist;
                            best_idx = i;
                        }
                    }

                    // Pack nibble into byte array
                    size_t nibble_byte = block_offset + 2 + (local_idx / 2);
                    if (local_idx % 2 == 0)
                        raw_data[nibble_byte] = (raw_data[nibble_byte] & 0xF0) | (best_idx & 0x0F);
                    else
                        raw_data[nibble_byte] = (raw_data[nibble_byte] & 0x0F) | ((best_idx & 0x0F) << 4);
                }
            }
        }

        std::vector<size_t> shape = {(size_t)rows, (size_t)cols};
        return std::make_shared<IQ4_NLTensor>(shape, raw_data);
    }

    /**
     * @brief Test a specific kernel variant for correctness (using FP32 mock decoder)
     */
    void testKernelVariant(const GemmKernelConfig &config, int m, int n, int k)
    {
        // Create random FP32 weight tensor (no quantization!)
        std::vector<float> B_fp32(n * k);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : B_fp32)
            v = dist(rng_);

        // Mock decoder wraps FP32 data directly
        MockFP32Decoder decoder(B_fp32.data(), n, k);

        // Create the specific kernel variant using test API
        auto &tuner = GemmAutoTuner::instance();
        auto kernel = tuner.createVariant(config, &decoder);
        ASSERT_NE(kernel, nullptr) << "Failed to create kernel: " << config.id();

        // Input matrix A (FP32)
        std::vector<float> A(m * k);
        for (auto &v : A)
            v = dist(rng_);

        // Output matrix C (result from kernel)
        std::vector<float> C(m * n, 0.0f);

        // Execute the kernel
        bool success = kernel->multiply(A.data(), C.data(), m, n, k, &decoder);
        ASSERT_TRUE(success) << "Kernel execution failed: " << config.id();

        // Compute reference result using CBLAS
        std::vector<float> C_expected(m * n, 0.0f);
        referenceGEMM(A.data(), B_fp32.data(), C_expected.data(), m, n, k);

        // Compare results
        EXPECT_TRUE(matricesEqual(C_expected.data(), C.data(), m * n, 1e-3f))
            << "Correctness test failed for kernel: " << config.id()
            << " with shape [" << m << ", " << n << ", " << k << "]";
    }
};

/**
 * @brief Test all kernel variants with single token inference shape (1×n×k)
 */
TEST_F(Test__GemmAutoTunerCorrectness, SingleToken_AllVariants)
{
    const int m = 1;   // Single token
    const int n = 128; // Output features
    const int k = 256; // Input features

    // Get all registered variants
    auto &tuner = GemmAutoTuner::instance();

    // Register variants by creating a dummy kernel with mock decoder
    std::vector<float> dummy_data(n * k, 1.0f);
    MockFP32Decoder dummy_decoder(dummy_data.data(), n, k);
    auto dummy_kernel = createAutoTunedGemm(&dummy_decoder);

    auto variants = tuner.getAvailableVariants();
    ASSERT_FALSE(variants.empty()) << "No variants registered!";

    std::cout << "Testing " << variants.size() << " kernel variants for single token shape ["
              << m << ", " << n << ", " << k << "]..." << std::endl;

    for (const auto &variant : variants)
    {
        testKernelVariant(variant, m, n, k);
    }
}

/**
 * @brief Test all kernel variants with small batch shape (32×n×k)
 */
TEST_F(Test__GemmAutoTunerCorrectness, SmallBatch_AllVariants)
{
    const int m = 32;  // Small batch
    const int n = 64;  // Output features
    const int k = 128; // Input features

    auto &tuner = GemmAutoTuner::instance();

    // Register variants
    auto dummy_tensor = createRandomTensor(n, k);
    auto dummy_kernel = createAutoTunedGemm(dummy_tensor.get());

    auto variants = tuner.getAvailableVariants();

    std::cout << "Testing " << variants.size() << " kernel variants for small batch shape ["
              << m << ", " << n << ", " << k << "]..." << std::endl;

    for (const auto &variant : variants)
    {
        testKernelVariant(variant, m, n, k);
    }
}

/**
 * @brief Test all kernel variants with medium batch shape (128×n×k)
 */
TEST_F(Test__GemmAutoTunerCorrectness, MediumBatch_AllVariants)
{
    const int m = 128; // Medium batch
    const int n = 32;  // Output features
    const int k = 64;  // Input features

    auto &tuner = GemmAutoTuner::instance();

    // Register variants with mock decoder
    std::vector<float> dummy_data(n * k, 1.0f);
    MockFP32Decoder dummy_decoder(dummy_data.data(), n, k);
    auto dummy_kernel = createAutoTunedGemm(&dummy_decoder);

    auto variants = tuner.getAvailableVariants();

    std::cout << "Testing " << variants.size() << " kernel variants for medium batch shape ["
              << m << ", " << n << ", " << k << "]..." << std::endl;

    for (const auto &variant : variants)
    {
        testKernelVariant(variant, m, n, k);
    }
}

/**
 * @brief Test all kernel variants with large batch shape (512×n×k)
 */
TEST_F(Test__GemmAutoTunerCorrectness, LargeBatch_AllVariants)
{
    const int m = 512; // Large batch
    const int n = 16;  // Output features
    const int k = 32;  // Input features

    auto &tuner = GemmAutoTuner::instance();

    // Register variants with mock decoder
    std::vector<float> dummy_data(n * k, 1.0f);
    MockFP32Decoder dummy_decoder(dummy_data.data(), n, k);
    auto dummy_kernel = createAutoTunedGemm(&dummy_decoder);

    auto variants = tuner.getAvailableVariants();

    std::cout << "Testing " << variants.size() << " kernel variants for large batch shape ["
              << m << ", " << n << ", " << k << "]..." << std::endl;

    for (const auto &variant : variants)
    {
        testKernelVariant(variant, m, n, k);
    }
}

/**
 * @brief Test specific high-performance variants with various shapes
 */
TEST_F(Test__GemmAutoTunerCorrectness, HighPerformanceVariants_MultipleShapes)
{
    // High-performance configurations (TILE_N=4 only, kernel hardcoded for 8x4 accumulator layout)
    std::vector<GemmKernelConfig> high_perf_configs = {
        {16, 5, 8, 4}, // unroll16_prefetch5_tile8x4
        {8, 5, 8, 4},  // unroll8_prefetch5_tile8x4
        {4, 3, 8, 4},  // unroll4_prefetch3_tile8x4
    };

    // Test shapes: {m, n, k}
    std::vector<std::tuple<int, int, int>> shapes = {
        {1, 896, 896},   // Qwen 0.5B single token
        {8, 896, 896},   // Qwen 0.5B small batch
        {32, 448, 896},  // Mixed dimensions
        {128, 256, 512}, // Medium batch
        {512, 128, 256}, // Large batch
    };

    for (const auto &config : high_perf_configs)
    {
        std::cout << "\nTesting variant: " << config.id() << std::endl;

        for (const auto &[m, n, k] : shapes)
        {
            std::cout << "  Shape [" << m << ", " << n << ", " << k << "]..." << std::endl;
            testKernelVariant(config, m, n, k);
        }
    }
}

/**
 * @brief Test non-aligned dimensions (k not multiple of block size)
 */
TEST_F(Test__GemmAutoTunerCorrectness, NonAlignedDimensions_AllVariants)
{
    const int m = 4;
    const int n = 8;
    const int k = 50; // Not a multiple of 32 (IQ4_NL block size)

    auto &tuner = GemmAutoTuner::instance();

    // Register variants with mock decoder
    std::vector<float> dummy_data(n * k, 1.0f);
    MockFP32Decoder dummy_decoder(dummy_data.data(), n, k);
    auto dummy_kernel = createAutoTunedGemm(&dummy_decoder);

    auto variants = tuner.getAvailableVariants();

    std::cout << "Testing " << variants.size() << " kernel variants for non-aligned shape ["
              << m << ", " << n << ", " << k << "]..." << std::endl;

    for (const auto &variant : variants)
    {
        testKernelVariant(variant, m, n, k);
    }
}

/**
 * @brief Test edge case: very small matrices
 */
TEST_F(Test__GemmAutoTunerCorrectness, VerySmallMatrices_AllVariants)
{
    const int m = 2;
    const int n = 2;
    const int k = 32; // Minimum aligned dimension

    auto &tuner = GemmAutoTuner::instance();

    // Register variants
    auto dummy_tensor = createRandomTensor(n, k);
    auto dummy_kernel = createAutoTunedGemm(dummy_tensor.get());

    auto variants = tuner.getAvailableVariants();

    std::cout << "Testing " << variants.size() << " kernel variants for very small shape ["
              << m << ", " << n << ", " << k << "]..." << std::endl;

    for (const auto &variant : variants)
    {
        testKernelVariant(variant, m, n, k);
    }
}

/**
 * @brief Test numerical stability with extreme values
 */
TEST_F(Test__GemmAutoTunerCorrectness, NumericalStability_AllVariants)
{
    const int m = 4;
    const int n = 8;
    const int k = 64;

    // Create FP32 weight tensor (mock decoder)
    std::vector<float> B_fp32(n * k);
    std::uniform_real_distribution<float> weight_dist(-1.0f, 1.0f);
    for (auto &v : B_fp32)
        v = weight_dist(rng_);

    MockFP32Decoder decoder(B_fp32.data(), n, k);

    auto &tuner = GemmAutoTuner::instance();
    auto dummy_kernel = createAutoTunedGemm(&decoder);
    auto variants = tuner.getAvailableVariants();

    // Create input with mix of small and large values
    std::vector<float> A(m * k);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto &v : A)
        v = dist(rng_);

    std::cout << "Testing " << variants.size() << " kernel variants for numerical stability..."
              << std::endl;

    for (const auto &variant : variants)
    {
        auto kernel = tuner.createVariant(variant, &decoder);
        ASSERT_NE(kernel, nullptr);

        std::vector<float> C(m * n, 0.0f);
        bool success = kernel->multiply(A.data(), C.data(), m, n, k, &decoder);
        ASSERT_TRUE(success);

        // Verify no NaN or Inf values
        for (int i = 0; i < m * n; ++i)
        {
            EXPECT_FALSE(std::isnan(C[i]))
                << "NaN detected in output at index " << i
                << " for kernel: " << variant.id();
            EXPECT_FALSE(std::isinf(C[i]))
                << "Inf detected in output at index " << i
                << " for kernel: " << variant.id();
        }

        // Compare with reference
        std::vector<float> C_expected(m * n, 0.0f);
        referenceGEMM(A.data(), B_fp32.data(), C_expected.data(), m, n, k);

        EXPECT_TRUE(matricesEqual(C_expected.data(), C.data(), m * n, 1e-3f))
            << "Numerical stability test failed for kernel: " << variant.id();
    }
}

/**
 * @brief Test that autotuner selects a variant that produces correct results
 */
TEST_F(Test__GemmAutoTunerCorrectness, AutoTunerSelection_ProducesCorrectResults)
{
    const int m = 32;
    const int n = 64;
    const int k = 128;

    // Create random FP32 weight tensor (mock decoder, no quantization)
    std::vector<float> B_fp32(n * k);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &v : B_fp32)
        v = dist(rng_);

    // DEBUG: Print first few values of B_fp32
    std::cout << "B_fp32[0:5] = ";
    for (int i = 0; i < 5; ++i)
    {
        std::cout << B_fp32[i] << " ";
    }
    std::cout << std::endl;

    MockFP32Decoder decoder(B_fp32.data(), n, k);

    // Let autotuner select the best variant
    auto kernel = createAutoTunedGemm(&decoder);
    ASSERT_NE(kernel, nullptr);

    // Input matrix
    std::vector<float> A(m * k);
    for (auto &v : A)
        v = dist(rng_);

    // Execute
    std::vector<float> C(m * n, 0.0f);
    bool success = kernel->multiply(A.data(), C.data(), m, n, k); // Use defaults: transpose_B=true, alpha=1, beta=0
    ASSERT_TRUE(success);

    // Debug: Print first 16 values
    std::cout << "First 16 C values: ";
    for (int i = 0; i < 16 && i < m * n; ++i)
    {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    // Reference using CBLAS
    std::vector<float> C_expected(m * n, 0.0f);
    referenceGEMM(A.data(), B_fp32.data(), C_expected.data(), m, n, k);

    // Verify autotuned kernel is correct
    EXPECT_TRUE(matricesEqual(C_expected.data(), C.data(), m * n, 1e-3f))
        << "Autotuner selected a variant that produces incorrect results!";
}
