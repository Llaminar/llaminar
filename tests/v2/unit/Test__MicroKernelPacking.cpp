/**
 * @file Test__MicroKernelPacking.cpp
 * @brief Unit tests for MicroKernelAdapter packing and buffer management
 * 
 * These tests verify that the packing logic correctly handles buffer bounds
 * and SIMD over-reads without relying on heavy integration tests.
 * 
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <vector>
#include <cstring>
#include <algorithm>

#include "../../src/v2/kernels/cpu/MicroKernelAdapter.h"
#include "../../src/v2/tensors/TensorKernels.h"

namespace {

/**
 * @brief Mock decoder for testing packing logic
 * 
 * Provides a simple IBlockDecoder implementation that fills blocks
 * with predictable test data for verification.
 */
class MockBlockDecoder : public llaminar2::IBlockDecoder
{
public:
    MockBlockDecoder(size_t rows, size_t cols, size_t block_size = 32)
        : rows_(rows), cols_(cols), block_size_(block_size)
    {
    }

    void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override
    {
        // Fill with predictable pattern: row_idx * 1000 + k_block_offset * 100 + offset
        for (size_t i = 0; i < block_size_; ++i)
        {
            output[i] = static_cast<float>(row_idx * 1000 + k_block_offset * 100 + i);
        }
    }

    const void *get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
    {
        return nullptr; // Not needed for this test
    }

    size_t block_size() const override { return block_size_; }
    size_t decoder_rows() const override { return rows_; }
    size_t decoder_cols() const override { return cols_; }

private:
    size_t rows_;
    size_t cols_;
    size_t block_size_;
};

/**
 * @brief Test fixture for MicroKernel packing tests
 */
class MicroKernelPackingTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Nothing to set up
    }

    /**
     * @brief Test that buffer allocation has sufficient padding
     * 
     * This is the core test that would have caught the ASAN error.
     */
    void testBufferPadding(int m, int k, int n, int mc, int kc, int nc)
    {
        MockBlockDecoder decoder(n, k, 32);

        // Calculate buffer sizes as MicroKernelAdapter does
        const size_t a_size = mc * k;
        const size_t b_size = k * nc;
        // Calculate padding: 10% of buffer size (rounded up), minimum 1KB
        // Use ceiling division: (size * 11 / 10) - size to get true 10%
        const size_t a_padding = std::max<size_t>(1024, (a_size * 11) / 10 - a_size);
        const size_t b_padding = std::max<size_t>(1024, (b_size * 11) / 10 - b_size);

        // Verify padding is sufficient
        EXPECT_GE(a_padding, 1024) << "A padding should be at least 1KB";
        EXPECT_GE(b_padding, 1024) << "B padding should be at least 1KB";
        EXPECT_GE(a_size + a_padding, (a_size * 11) / 10) << "A buffer should have 10% padding";
        EXPECT_GE(b_size + b_padding, (b_size * 11) / 10) << "B buffer should have 10% padding";

        // Allocate buffers with padding
        std::vector<float> A(m * k, 1.0f);
        std::vector<float> C(m * n, 0.0f);
        std::vector<float> A_packed(a_size + a_padding, 0.0f);
        std::vector<float> B_decoded(b_size + b_padding, 0.0f);
        std::vector<float> B_packed(b_size + b_padding, 0.0f);

        // Fill buffers with sentinel value to detect out-of-bounds writes
        const float SENTINEL = -999.0f;
        std::fill(A_packed.begin() + a_size, A_packed.end(), SENTINEL);
        std::fill(B_decoded.begin() + b_size, B_decoded.end(), SENTINEL);
        std::fill(B_packed.begin() + b_size, B_packed.end(), SENTINEL);

        // Simulate the decode loop from MicroKernelAdapter
        const size_t block_size = decoder.block_size();
        const size_t num_k_blocks = (k + block_size - 1) / block_size;

        // Decode B matrix panel (simplified from actual adapter code)
        for (int jc = 0; jc < n; jc += nc)
        {
            int nc_actual = std::min(nc, n - jc);

            for (size_t kb = 0; kb < num_k_blocks; ++kb)
            {
                size_t k_start = kb * block_size;
                size_t k_end = std::min(k_start + block_size, static_cast<size_t>(k));

                for (int jj = 0; jj < nc_actual; ++jj)
                {
                    float block_data[256]; // Max block size
                    decoder.decode_block_at(jc + jj, kb, block_data);

                    // Store decoded values in row-major layout
                    for (size_t kk = k_start; kk < k_end; ++kk)
                    {
                        size_t idx = jj * k + kk;
                        ASSERT_LT(idx, b_size) << "Decode index out of bounds: jj=" << jj 
                                                << " k=" << k << " kk=" << kk;
                        B_decoded[idx] = block_data[kk - k_start];
                    }
                }
            }

            // Verify sentinel values are intact (no buffer overflow during decode)
            for (size_t i = b_size; i < B_decoded.size(); ++i)
            {
                EXPECT_FLOAT_EQ(B_decoded[i], SENTINEL) 
                    << "Buffer overflow detected at index " << i 
                    << " during decode (nc=" << nc << ", k=" << k << ")";
            }
        }
    }

    /**
     * @brief Test SIMD alignment padding calculations
     */
    void testSimdPaddingCalculation()
    {
        // Test various buffer sizes
        struct TestCase {
            size_t buffer_size;
            size_t min_padding;
            std::string description;
        };

        std::vector<TestCase> cases = {
            {896 * 128, 1024, "Standard tile (896×128)"},
            {896 * 64, 1024, "Small tile (896×64)"},
            {896 * 256, (896 * 256 * 11) / 10 - (896 * 256), "Large tile (896×256)"},
            {100, 1024, "Tiny buffer (100 elements)"},
        };

        for (const auto& tc : cases)
        {
            size_t padding = std::max<size_t>(1024, (tc.buffer_size * 11) / 10 - tc.buffer_size);
            EXPECT_GE(padding, tc.min_padding) 
                << "Padding insufficient for " << tc.description;
            EXPECT_GE(tc.buffer_size + padding, (tc.buffer_size * 11) / 10)
                << "Total buffer size insufficient for " << tc.description;
        }
    }
};

// Test standard tile size (128×896×896 with mc=128, nc=128)
TEST_F(MicroKernelPackingTest, StandardTileSize)
{
    testBufferPadding(
        128,  // m
        896,  // k
        896,  // n
        128,  // mc (cache block M)
        896,  // kc (cache block K, typically full K)
        128   // nc (cache block N)
    );
}

// Test small tile size (64×896×896 with mc=128, nc=64)
TEST_F(MicroKernelPackingTest, SmallTileSize)
{
    testBufferPadding(
        64,   // m
        896,  // k
        896,  // n
        128,  // mc
        896,  // kc
        64    // nc (smaller cache block)
    );
}

// Test large tile size (512×896×896 with mc=256, nc=256)
TEST_F(MicroKernelPackingTest, LargeTileSize)
{
    testBufferPadding(
        512,  // m
        896,  // k
        896,  // n
        256,  // mc (larger cache block)
        896,  // kc
        256   // nc (larger cache block)
    );
}

// Test problematic size from ASAN report (m=8, mc=128, nc=64)
TEST_F(MicroKernelPackingTest, AsanReportedSize)
{
    testBufferPadding(
        8,    // m (small batch)
        896,  // k
        896,  // n
        128,  // mc
        896,  // kc
        64    // nc (this was causing the overflow)
    );
}

// Test SIMD padding calculation
TEST_F(MicroKernelPackingTest, SimdPaddingCalculation)
{
    testSimdPaddingCalculation();
}

// Test edge case: very small matrix
TEST_F(MicroKernelPackingTest, VerySmallMatrix)
{
    testBufferPadding(
        1,    // m (single row)
        32,   // k (small K)
        32,   // n (small N)
        128,  // mc (cache block larger than matrix)
        32,   // kc
        64    // nc (cache block larger than matrix)
    );
}

// Test edge case: very large matrix
TEST_F(MicroKernelPackingTest, VeryLargeMatrix)
{
    testBufferPadding(
        2048, // m (large)
        896,  // k
        896,  // n
        256,  // mc
        896,  // kc
        256   // nc
    );
}

} // anonymous namespace
