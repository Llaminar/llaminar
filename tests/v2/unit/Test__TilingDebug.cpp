#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <memory>
#include <cblas.h>
#include "tensors/Tensors.h"
#include "kernels/cpu/GemmAutoTuner.h"

using namespace llaminar2;
using namespace llaminar::v2::kernels;

/**
 * Test to debug tiling loop - check if n=8 (2 tiles) works correctly
 */
TEST(TilingDebug, TwoColumnTiles)
{
    // IQ4_NL lookup grid
    static constexpr float kvalues_iq4nl[16] = {
        -127.0f / 127.0f, -104.0f / 127.0f, -83.0f / 127.0f, -65.0f / 127.0f,
        -49.0f / 127.0f, -35.0f / 127.0f, -22.0f / 127.0f, -10.0f / 127.0f,
        1.0f / 127.0f, 13.0f / 127.0f, 25.0f / 127.0f, 38.0f / 127.0f,
        53.0f / 127.0f, 69.0f / 127.0f, 89.0f / 127.0f, 113.0f / 127.0f};

    // Shape matching AutoTunerSelection test
    const int m = 32; // Multiple rows!
    const int n = 64; // Requires 16 tiles (64/4=16)
    const int k = 128;

    const size_t blocks_per_row = (k + 31) / 32;            // 128/32 = 4 blocks per row
    std::vector<uint8_t> raw_data(n * blocks_per_row * 18); // n rows × blocks/row × 18 bytes/block

    // All blocks: scale=1.0, indices alternating 0 and 15
    uint16_t scale = 0x3C00; // FP16(1.0)
    for (int row = 0; row < n; ++row)
    {
        for (size_t block_idx = 0; block_idx < blocks_per_row; ++block_idx)
        {
            size_t offset = (row * blocks_per_row + block_idx) * 18;
            std::memcpy(&raw_data[offset], &scale, 2);
            // Even rows: all 0s (-1.0), odd rows: all 15s (+0.89)
            uint8_t fill = (row % 2 == 0) ? 0x00 : 0xFF;
            std::memset(&raw_data[offset + 2], fill, 16);
        }
    }

    std::vector<size_t> shape = {(size_t)n, (size_t)k};
    auto tensor = std::make_shared<IQ4_NLTensor>(shape, raw_data);

    // Input: all ones
    std::vector<float> A(m * k, 1.0f);
    std::vector<float> C(m * n, 0.0f);

    // Get first variant
    auto &tuner = GemmAutoTuner::instance();
    auto variants = tuner.getAvailableVariants();
    std::cout << "Testing with variant: " << variants[0].id() << std::endl;
    std::cout << "Shape: m=" << m << ", n=" << n << ", k=" << k << std::endl;
    std::cout << "Expected tiles: " << (n / 4) << " column tiles" << std::endl;

    auto kernel = tuner.createVariant(variants[0], tensor.get());
    bool success = kernel->multiply(A.data(), C.data(), m, n, k, tensor.get());

    ASSERT_TRUE(success);

    std::cout << "\nResults:" << std::endl;
    bool all_correct = true;
    for (int j = 0; j < n; ++j)
    {
        float expected = k * kvalues_iq4nl[(j % 2 == 0) ? 0 : 15]; // k elements, not 32!
        std::cout << "  C[" << j << "] = " << C[j] << " (expected " << expected << ")";

        float diff = std::abs(C[j] - expected);
        if (diff < 0.01f)
        {
            std::cout << " ✓" << std::endl;
        }
        else
        {
            std::cout << " ✗ MISMATCH (diff=" << diff << ")" << std::endl;
            all_correct = false;
        }
    }

    EXPECT_TRUE(all_correct) << "Some values don't match expected results!";
}
