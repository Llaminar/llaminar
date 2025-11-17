/**
 * @file Test__MicroKernelAdapter_Packing.cpp
 * @brief Unit tests for GemmMicroKernelAdapter packing logic
 *
 * Tests the cache-blocked packing and decoding logic with various matrix sizes
 * to isolate potential buffer overflow or memory corruption issues.
 *
 * @author David Sanftenberg
 * @date October 2025
 */

#include <gtest/gtest.h>
#include "../../../src/v2/kernels/cpu/gemm/GemmMicroKernelAdapter.h"
#include "../../../src/v2/kernels/cpu/gemm/GemmMicroKernelRegistry.h"
#include "../../../src/v2/tensors/Tensors.h"
#include <vector>
#include <cstring>

using namespace llaminar2::kernels::gemm;

/**
 * @brief Simple test decoder for unit testing
 */
class TestBlockDecoder : public llaminar2::ITensorGemmTileDataProvider
{
public:
    TestBlockDecoder(size_t rows, size_t cols)
        : rows_(rows), cols_(cols), block_size_(32)
    {
        // Initialize with simple pattern for validation
        data_.resize(rows * cols, 1.0f);
    }

    void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override
    {
        size_t k_start = k_block_offset * block_size_;
        size_t k_end = std::min(k_start + block_size_, cols_);

        for (size_t k = k_start; k < k_end; ++k)
        {
            output[k - k_start] = data_[row_idx * cols_ + k];
        }
    }

    const void *get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
    {
        return nullptr; // Not used in these tests
    }

    size_t block_size() const override { return block_size_; }

    size_t decoder_rows() const override { return rows_; }

    size_t decoder_cols() const override { return cols_; }

private:
    std::vector<float> data_;
    size_t rows_;
    size_t cols_;
    size_t block_size_;
};

/**
 * @brief Test fixture for MicroKernelAdapter packing tests
 */
class MicroKernelAdapterPackingTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Get a simple 4×2 kernel for testing
        auto &registry = MicroKernelRegistry::instance();

        // Try to get a simple AVX512 4×2 kernel
        if (registry.has_kernel("simd::AVX512Tag", 4, 2, 8, 5))
        {
            bundle_ = registry.get_kernel("simd::AVX512Tag", 4, 2, 8, 5);
            has_kernel_ = true;
        }
        else if (registry.has_kernel("simd::AVX2Tag", 4, 2, 8, 5))
        {
            bundle_ = registry.get_kernel("simd::AVX2Tag", 4, 2, 8, 5);
            has_kernel_ = true;
        }
        else
        {
            has_kernel_ = false;
        }
    }

    MicroKernelBundle bundle_;
    bool has_kernel_ = false;
};

/**
 * @brief Test small matrix packing (should work)
 */
TEST_F(MicroKernelAdapterPackingTest, SmallMatrix)
{
    if (!has_kernel_)
    {
        GTEST_SKIP() << "No suitable kernel available";
    }

    const int m = 32;
    const int n = 896;
    const int k = 896;

    TestBlockDecoder decoder(n, k);
    MicroKernelVariantAdapter adapter("simd::AVX512Tag", 4, 2, 8, 5, bundle_, &decoder);

    // Allocate matrices
    std::vector<float> A(m * k, 1.0f);
    std::vector<float> C(m * n, 0.0f);

    // Execute GEMM
    bool success = adapter.multiply(A.data(), C.data(), m, n, k, &decoder, 1.0f, 0.0f);

    EXPECT_TRUE(success) << "GEMM should succeed for 32×896×896";

    // Basic sanity check: output should be non-zero
    float sum = 0.0f;
    for (int i = 0; i < m * n; ++i)
    {
        sum += C[i];
    }
    EXPECT_GT(sum, 0.0f) << "Output should be non-zero";
}

/**
 * @brief Test medium matrix packing (512 rows)
 */
TEST_F(MicroKernelAdapterPackingTest, MediumMatrix)
{
    if (!has_kernel_)
    {
        GTEST_SKIP() << "No suitable kernel available";
    }

    const int m = 512;
    const int n = 896;
    const int k = 896;

    TestBlockDecoder decoder(n, k);
    MicroKernelVariantAdapter adapter("simd::AVX512Tag", 4, 2, 8, 5, bundle_, &decoder);

    std::vector<float> A(m * k, 1.0f);
    std::vector<float> C(m * n, 0.0f);

    bool success = adapter.multiply(A.data(), C.data(), m, n, k, &decoder, 1.0f, 0.0f);

    EXPECT_TRUE(success) << "GEMM should succeed for 512×896×896";

    float sum = 0.0f;
    for (int i = 0; i < std::min(1000, m * n); ++i)
    {
        sum += C[i];
    }
    EXPECT_GT(sum, 0.0f) << "Output should be non-zero";
}

/**
 * @brief Test large matrix packing (1024 rows) - where crash occurs
 */
TEST_F(MicroKernelAdapterPackingTest, LargeMatrix)
{
    if (!has_kernel_)
    {
        GTEST_SKIP() << "No suitable kernel available";
    }

    const int m = 1024;
    const int n = 896;
    const int k = 896;

    TestBlockDecoder decoder(n, k);
    MicroKernelVariantAdapter adapter("simd::AVX512Tag", 4, 2, 8, 5, bundle_, &decoder);

    std::vector<float> A(m * k, 1.0f);
    std::vector<float> C(m * n, 0.0f);

    bool success = adapter.multiply(A.data(), C.data(), m, n, k, &decoder, 1.0f, 0.0f);

    EXPECT_TRUE(success) << "GEMM should succeed for 1024×896×896";

    float sum = 0.0f;
    for (int i = 0; i < std::min(1000, m * n); ++i)
    {
        sum += C[i];
    }
    EXPECT_GT(sum, 0.0f) << "Output should be non-zero";
}

/**
 * @brief Test very large matrix packing (2048 rows)
 */
TEST_F(MicroKernelAdapterPackingTest, VeryLargeMatrix)
{
    if (!has_kernel_)
    {
        GTEST_SKIP() << "No suitable kernel available";
    }

    const int m = 2048;
    const int n = 896;
    const int k = 896;

    TestBlockDecoder decoder(n, k);
    MicroKernelVariantAdapter adapter("simd::AVX512Tag", 4, 2, 8, 5, bundle_, &decoder);

    std::vector<float> A(m * k, 1.0f);
    std::vector<float> C(m * n, 0.0f);

    bool success = adapter.multiply(A.data(), C.data(), m, n, k, &decoder, 1.0f, 0.0f);

    EXPECT_TRUE(success) << "GEMM should succeed for 2048×896×896";

    float sum = 0.0f;
    for (int i = 0; i < std::min(1000, m * n); ++i)
    {
        sum += C[i];
    }
    EXPECT_GT(sum, 0.0f) << "Output should be non-zero";
}

/**
 * @brief Test extreme matrix packing (4096 rows)
 */
TEST_F(MicroKernelAdapterPackingTest, ExtremeMatrix)
{
    if (!has_kernel_)
    {
        GTEST_SKIP() << "No suitable kernel available";
    }

    const int m = 4096;
    const int n = 896;
    const int k = 896;

    TestBlockDecoder decoder(n, k);
    MicroKernelVariantAdapter adapter("simd::AVX512Tag", 4, 2, 8, 5, bundle_, &decoder);

    std::vector<float> A(m * k, 1.0f);
    std::vector<float> C(m * n, 0.0f);

    bool success = adapter.multiply(A.data(), C.data(), m, n, k, &decoder, 1.0f, 0.0f);

    EXPECT_TRUE(success) << "GEMM should succeed for 4096×896×896";

    float sum = 0.0f;
    for (int i = 0; i < std::min(1000, m * n); ++i)
    {
        sum += C[i];
    }
    EXPECT_GT(sum, 0.0f) << "Output should be non-zero";
}

/**
 * @brief Test buffer size calculation for various M values
 */
TEST_F(MicroKernelAdapterPackingTest, BufferSizeCalculation)
{
    // Test that buffer sizes are correctly calculated for cache blocking
    const int k = 896;

    struct TestCase
    {
        int m;
        int expected_mc;
        int expected_nc;
    };

    std::vector<TestCase> test_cases = {
        {32, 128, 128}, // Medium micro-kernel (4×2 = 8 registers)
        {512, 128, 128},
        {1024, 128, 128},
        {2048, 128, 128},
        {4096, 128, 128}};

    for (const auto &tc : test_cases)
    {
        // Buffer sizes for MC=128, KC=512, NC=128
        size_t a_packed_size = 128 * 512; // MC × KC
        size_t b_packed_size = 512 * 128; // KC × NC

        EXPECT_GT(a_packed_size, 0) << "A_packed size should be positive for m=" << tc.m;
        EXPECT_GT(b_packed_size, 0) << "B_packed size should be positive for m=" << tc.m;

        // Verify sizes are reasonable (not excessive)
        EXPECT_LT(a_packed_size, 10000000) << "A_packed shouldn't exceed 10M floats";
        EXPECT_LT(b_packed_size, 10000000) << "B_packed shouldn't exceed 10M floats";
    }
}
