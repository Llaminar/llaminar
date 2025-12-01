/**
 * @file Test__IQ3_STensor.cpp
 * @brief Unit tests for IQ3_STensor IINT8Unpackable interface
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "../../../../src/v2/tensors/Tensors.h"
#include "../../../../src/v2/tensors/TensorFactory.h"
#include "../../../../src/v2/tensors/FP16Utils.h"
#include "../../../../src/v2/tensors/IQQuantTables.h"
#include <vector>
#include <cstring>

using namespace llaminar2;

class IQ3_STensorTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
    }
};

TEST_F(IQ3_STensorTest, UnpackBlockToInt8)
{
    // Create a single block IQ3_S tensor
    // Shape [1, 256] (1 row, 256 cols = 1 super-block)

    std::vector<size_t> shape = {1, 256};
    std::vector<uint8_t> raw_data(sizeof(IQ3_SBlock));
    IQ3_SBlock *block = reinterpret_cast<IQ3_SBlock *>(raw_data.data());

    // Initialize block
    // d = 2.0
    block->d = fp32_to_fp16(2.0f);

    // qs[0..63] = 0 (grid index 0 -> 1)
    std::memset(block->qs, 0, 64);

    // qh[0..7] = 0
    std::memset(block->qh, 0, 8);

    // signs[0..31] = 0 (all positive)
    std::memset(block->signs, 0, 32);

    // scales[0..3] = 0 (scale index 0)
    std::memset(block->scales, 0, 4);

    IQ3_STensor tensor(shape, raw_data);

    // Check block size
    EXPECT_EQ(tensor.block_size(), 32);

    // Unpack sub-block 0
    int8_t unpacked[32];
    tensor.unpack_block_to_int8(0, 0, unpacked);
    float scale = tensor.get_block_scale(0, 0);

    // We don't know exact values without checking tables, but we can check consistency.
    // If we decode to FP32 first, we can verify against that.

    float fp32_values[32]; // Sub-block decode
    tensor.decode_block_at(0, 0, fp32_values);

    // Check that unpacked * scale matches fp32_values approximately for the first sub-block (0-31)
    for (int i = 0; i < 32; ++i)
    {
        float reconstructed = unpacked[i] * scale;
        EXPECT_NEAR(reconstructed, fp32_values[i], 0.05f); // Allow some quantization error
    }
}
