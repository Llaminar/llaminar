#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <memory>
#include <cblas.h>
#include "tensors/Tensors.h"
#include "kernels/cpu/GemmAutoTuner.h"

using namespace llaminar2;
using namespace llaminar::v2::kernels;

TEST(GemmDebug, SimpleTest)
{
    // IQ4_NL lookup grid
    static constexpr float kvalues_iq4nl[16] = {
        -127.0f / 127.0f, -104.0f / 127.0f, -83.0f / 127.0f, -65.0f / 127.0f,
        -49.0f / 127.0f, -35.0f / 127.0f, -22.0f / 127.0f, -10.0f / 127.0f,
        1.0f / 127.0f, 13.0f / 127.0f, 25.0f / 127.0f, 38.0f / 127.0f,
        53.0f / 127.0f, 69.0f / 127.0f, 89.0f / 127.0f, 113.0f / 127.0f};

    // Create small test: 2x32 tensor (2 rows, 32 cols = 1 block per row)
    const int n = 2, k = 32;
    std::vector<uint8_t> raw_data(n * 1 * 18); // 2 rows × 1 block/row × 18 bytes/block

    // Block 0 (row 0): scale=1.0, all indices = 0 (value -127/127)
    uint16_t scale0 = 0x3C00; // FP16(1.0) = 0x3C00
    std::memcpy(&raw_data[0], &scale0, 2);
    std::memset(&raw_data[2], 0x00, 16); // All nibbles = 0

    // Block 1 (row 1): scale=1.0, all indices = 15 (value 113/127)
    uint16_t scale1 = 0x3C00;
    std::memcpy(&raw_data[18], &scale1, 2);
    std::memset(&raw_data[20], 0xFF, 16); // All nibbles = 15

    std::vector<size_t> shape = {(size_t)n, (size_t)k};
    auto tensor = std::make_shared<IQ4_NLTensor>(shape, raw_data);

    // Decode to verify
    std::vector<float> decoded(n * k);
    tensor->to_fp32(decoded.data());

    std::cout << "Decoded row 0 [0:4]: ";
    for (int i = 0; i < 4; ++i)
    {
        std::cout << decoded[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Decoded row 1 [32:36]: ";
    for (int i = 32; i < 36; ++i)
    {
        std::cout << decoded[i] << " ";
    }
    std::cout << std::endl;

    // Simple GEMM: A[1×32] × B[2×32]^T = C[1×2]
    const int m = 1;
    std::vector<float> A(m * k, 1.0f);
    std::vector<float> C(m * n, 0.0f);

    // Get kernel
    auto &tuner = GemmAutoTuner::instance();
    auto variants = tuner.getAvailableVariants();
    std::cout << "Found " << variants.size() << " variants" << std::endl;

    ASSERT_FALSE(variants.empty());

    auto kernel = tuner.createVariant(variants[0], tensor.get());
    std::cout << "Testing variant: " << variants[0].id() << std::endl;

    bool success = kernel->multiply(A.data(), C.data(), m, n, k, tensor.get());
    std::cout << "Multiply returned: " << (success ? "true" : "false") << std::endl;
    std::cout << "C = [" << C[0] << ", " << C[1] << "]" << std::endl;

    float expected0 = 32.0f * kvalues_iq4nl[0];
    float expected1 = 32.0f * kvalues_iq4nl[15];
    std::cout << "Expected: [" << expected0 << ", " << expected1 << "]" << std::endl;

    EXPECT_TRUE(success);
    EXPECT_NEAR(C[0], expected0, 0.01f);
    EXPECT_NEAR(C[1], expected1, 0.01f);
}
