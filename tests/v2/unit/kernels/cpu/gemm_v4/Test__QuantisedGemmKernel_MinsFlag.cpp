#include <gtest/gtest.h>
#include "v2/tensors/Tensors.h"
#include "v2/kernels/cpu/gemm_v4/QuantisedGemmKernel.h"
#include <vector>
#include <random>

using namespace llaminar2;
using namespace llaminar2::gemm_v4;

TEST(Test__Q4_0_Mins, VerifyMinsAreZero) {
    // Create a Q4_0 tensor
    // Shape: [128, 128]
    // 128 rows, 128 cols.
    // 128 cols = 4 blocks per row.
    // Total blocks = 128 * 4 = 512 blocks.
    // Total bytes = 512 * 18 = 9216 bytes.

    std::vector<size_t> shape = {128, 128};
    std::vector<uint8_t> raw_data(512 * sizeof(Q4_0Block));
    
    // Fill with random data
    std::mt19937 gen(42);
    std::uniform_int_distribution<uint8_t> dist(0, 255);
    for (auto &b : raw_data) b = dist(gen);

    Q4_0Tensor tensor(shape, raw_data);

    // Pack weights
    QuantisedPackedWeights packed;
    bool success = QuantisedGemmKernel::packWeightsInto(&tensor, packed);
    ASSERT_TRUE(success);

    // Verify mins are all zero
    ASSERT_FALSE(packed.mins.empty());
    EXPECT_FALSE(packed.has_mins) << "has_mins should be false for Q4_0";
    for (size_t i = 0; i < packed.mins.size(); ++i) {
        if (packed.mins[i] != 0.0f) {
            EXPECT_EQ(packed.mins[i], 0.0f) << "Min at index " << i << " is not zero";
            break; // Fail fast
        }
    }
}

TEST(Test__Q4_0_Mins, VerifyQ4_1_SetsFlag) {
    // Create Q4_1 tensor
    std::vector<size_t> shape = {128, 128};
    std::vector<uint8_t> raw_data(512 * sizeof(Q4_1Block));
    
    // Fill with random data
    std::mt19937 gen(42);
    std::uniform_int_distribution<uint8_t> dist(0, 255);
    for (auto &b : raw_data) b = dist(gen);

    // Force at least one min to be non-zero
    Q4_1Block* blocks = reinterpret_cast<Q4_1Block*>(raw_data.data());
    blocks[0].m = 0x3C00; // 1.0 in FP16

    Q4_1Tensor tensor(shape, raw_data);

    // Pack weights
    QuantisedPackedWeights packed;
    bool success = QuantisedGemmKernel::packWeightsInto(&tensor, packed);
    ASSERT_TRUE(success);

    EXPECT_TRUE(packed.has_mins) << "has_mins should be true for Q4_1 with non-zero min";
}
