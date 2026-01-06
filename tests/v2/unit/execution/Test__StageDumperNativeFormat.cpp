/**
 * @file Test__StageDumperNativeFormat.cpp
 * @brief Unit tests for native format tensor dumping in StageDumper
 *
 * Validates that tensors are dumped in their native format (Q8_1, Q16_1, etc.)
 * rather than being dequantized to FP32 first.
 *
 * This regression test ensures:
 * 1. Q8_1 tensors are dumped as raw Q8_1Block data
 * 2. Q16_1 tensors are dumped as raw Q16_1Block data
 * 3. FP32 tensors are dumped as FP32
 * 4. Byte sizes match native storage sizes
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include "execution/compute_stages/IComputeStage.h"
#include "tensors/Tensors.h"

namespace fs = std::filesystem;

using namespace llaminar2;

namespace
{

    // Q8_1Block: 2 bytes (d FP16) + 2 bytes (sum_qs INT16) + 32 bytes (qs INT8) = 36 bytes
    static_assert(sizeof(Q8_1Block) == 36, "Q8_1Block should be 36 bytes");

    // Q16_1Block: 4 bytes (d FP32) + 4 bytes (sum_qs INT32) + 64 bytes (qs INT16) = 72 bytes
    static_assert(sizeof(Q16_1Block) == 72, "Q16_1Block should be 72 bytes");

    /**
     * @brief Create a simple Q8_1 tensor for testing
     */
    std::unique_ptr<Q8_1Tensor> createTestQ8_1Tensor(int rows, int cols)
    {
        // Constructor: Q8_1Tensor(shape, device_idx)
        auto tensor = std::make_unique<Q8_1Tensor>(
            std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
            -1); // CPU device

        // Initialize with test pattern
        Q8_1Block *blocks = tensor->mutable_q8_1_blocks();
        size_t num_blocks = tensor->total_blocks();

        for (size_t i = 0; i < num_blocks; ++i)
        {
            blocks[i].d = static_cast<uint16_t>(0x3C00 + (i & 0xFF)); // FP16 ~1.0 + offset
            blocks[i].sum_qs = static_cast<int16_t>(i * 10);
            for (int j = 0; j < 32; ++j)
            {
                blocks[i].qs[j] = static_cast<int8_t>((i + j) % 256 - 128);
            }
        }

        return tensor;
    }

    /**
     * @brief Create a simple Q16_1 tensor for testing
     */
    std::unique_ptr<Q16_1Tensor> createTestQ16_1Tensor(int rows, int cols)
    {
        // Constructor: Q16_1Tensor(shape, device_idx)
        auto tensor = std::make_unique<Q16_1Tensor>(
            std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
            -1); // CPU device

        // Initialize with test pattern
        Q16_1Block *blocks = tensor->mutable_q16_1_blocks();
        size_t num_blocks = tensor->total_blocks();

        for (size_t i = 0; i < num_blocks; ++i)
        {
            blocks[i].d = 1.0f + static_cast<float>(i) * 0.01f;
            blocks[i].sum_qs = static_cast<int32_t>(i * 100);
            for (int j = 0; j < 32; ++j)
            {
                blocks[i].qs[j] = static_cast<int16_t>((i + j) % 65536 - 32768);
            }
        }

        return tensor;
    }

    /**
     * @brief Create a simple FP32 tensor for testing
     */
    std::unique_ptr<FP32Tensor> createTestFP32Tensor(int rows, int cols)
    {
        // Constructor: FP32Tensor(shape, device_idx)
        auto tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
            -1); // CPU device

        float *data = tensor->mutable_data();
        size_t count = rows * cols;
        for (size_t i = 0; i < count; ++i)
        {
            data[i] = static_cast<float>(i) * 0.1f - 50.0f;
        }

        return tensor;
    }

} // anonymous namespace

class Test__StageDumperNativeFormat : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create temp directory for test dumps
        test_dir_ = fs::temp_directory_path() / "llaminar_stage_dump_test";
        fs::create_directories(test_dir_);
    }

    void TearDown() override
    {
        // Clean up
        fs::remove_all(test_dir_);
    }

    fs::path test_dir_;
};

/**
 * @brief Test that addInput with Q8_1Tensor stores raw data, not dequantized
 */
TEST_F(Test__StageDumperNativeFormat, AddInputQ8_1_StoresRawData)
{
    constexpr int ROWS = 4;
    constexpr int COLS = 64; // 2 Q8_1 blocks per row

    auto tensor = createTestQ8_1Tensor(ROWS, COLS);

    // Get raw data pointer and size for comparison
    const void *raw_ptr = tensor->raw_data();
    size_t raw_size = tensor->size_bytes();

    // Expected: 4 rows * 2 blocks/row = 8 blocks * 36 bytes = 288 bytes
    ASSERT_EQ(raw_size, 8 * sizeof(Q8_1Block)) << "Q8_1 tensor should be 8 blocks * 36 bytes";

    // Create dump info
    StageDumpInfo info;
    info.addInput("test_q8_1", tensor.get(), ROWS, COLS);

    ASSERT_EQ(info.inputs.size(), 1);
    const auto &input = info.inputs[0];

    // Verify dump info stores RAW pointer, not FP32
    EXPECT_EQ(input.data, raw_ptr) << "addInput should store raw_data(), not fp32_data()";
    EXPECT_STREQ(input.dtype, "Q8_1");
    EXPECT_EQ(input.byte_size, raw_size) << "byte_size should match native storage";

    // Verify it's NOT pointing to FP32 data (which would be larger)
    size_t fp32_size = ROWS * COLS * sizeof(float);
    EXPECT_NE(input.byte_size, fp32_size) << "Should NOT be FP32 size";
}

/**
 * @brief Test that addInput with Q16_1Tensor stores raw data
 */
TEST_F(Test__StageDumperNativeFormat, AddInputQ16_1_StoresRawData)
{
    constexpr int ROWS = 4;
    constexpr int COLS = 64; // 2 Q16_1 blocks per row

    auto tensor = createTestQ16_1Tensor(ROWS, COLS);

    const void *raw_ptr = tensor->raw_data();
    size_t raw_size = tensor->size_bytes();

    // Expected: 4 rows * 2 blocks/row = 8 blocks * 72 bytes = 576 bytes
    ASSERT_EQ(raw_size, 8 * sizeof(Q16_1Block)) << "Q16_1 tensor should be 8 blocks * 72 bytes";

    StageDumpInfo info;
    info.addInput("test_q16_1", tensor.get(), ROWS, COLS);

    ASSERT_EQ(info.inputs.size(), 1);
    const auto &input = info.inputs[0];

    EXPECT_EQ(input.data, raw_ptr);
    EXPECT_STREQ(input.dtype, "Q16_1");
    EXPECT_EQ(input.byte_size, raw_size);
}

/**
 * @brief Test that addInput with FP32Tensor works correctly
 */
TEST_F(Test__StageDumperNativeFormat, AddInputFP32_StoresData)
{
    constexpr int ROWS = 4;
    constexpr int COLS = 64;

    auto tensor = createTestFP32Tensor(ROWS, COLS);

    const void *raw_ptr = tensor->raw_data();
    size_t raw_size = tensor->size_bytes();

    ASSERT_EQ(raw_size, ROWS * COLS * sizeof(float));

    StageDumpInfo info;
    info.addInput("test_fp32", tensor.get(), ROWS, COLS);

    ASSERT_EQ(info.inputs.size(), 1);
    const auto &input = info.inputs[0];

    EXPECT_EQ(input.data, raw_ptr);
    EXPECT_STREQ(input.dtype, "FP32");
    EXPECT_EQ(input.byte_size, raw_size);
}

/**
 * @brief Test that addOutput with Q8_1Tensor stores raw data
 */
TEST_F(Test__StageDumperNativeFormat, AddOutputQ8_1_StoresRawData)
{
    constexpr int ROWS = 4;
    constexpr int COLS = 64;

    auto tensor = createTestQ8_1Tensor(ROWS, COLS);

    const void *raw_ptr = tensor->raw_data();
    size_t raw_size = tensor->size_bytes();

    StageDumpInfo info;
    info.addOutput("test_q8_1_out", tensor.get(), ROWS, COLS);

    ASSERT_EQ(info.outputs.size(), 1);
    const auto &output = info.outputs[0];

    EXPECT_EQ(output.data, raw_ptr);
    EXPECT_STREQ(output.dtype, "Q8_1");
    EXPECT_EQ(output.byte_size, raw_size);
}

/**
 * @brief Verify that the raw data can be reconstructed from dumped bytes
 */
TEST_F(Test__StageDumperNativeFormat, Q8_1_RawDataIsReconstructible)
{
    constexpr int ROWS = 4;
    constexpr int COLS = 64; // 2 Q8_1 blocks per row

    auto tensor = createTestQ8_1Tensor(ROWS, COLS);

    // Get original block data
    const Q8_1Block *original_blocks = tensor->q8_1_blocks();
    size_t num_blocks = tensor->total_blocks();
    size_t raw_size = tensor->size_bytes();

    // Create dump info
    StageDumpInfo info;
    info.addInput("test", tensor.get(), ROWS, COLS);

    // Simulate what StageDumper would write - the raw bytes
    const void *dump_data = info.inputs[0].data;
    size_t dump_size = info.inputs[0].byte_size;

    ASSERT_EQ(dump_size, raw_size);

    // Cast the dumped data back to Q8_1Block and verify contents
    const Q8_1Block *loaded_blocks = static_cast<const Q8_1Block *>(dump_data);

    for (size_t i = 0; i < num_blocks; ++i)
    {
        EXPECT_EQ(loaded_blocks[i].d, original_blocks[i].d)
            << "Block " << i << " scale mismatch";
        EXPECT_EQ(loaded_blocks[i].sum_qs, original_blocks[i].sum_qs)
            << "Block " << i << " sum_qs mismatch";

        for (int j = 0; j < 32; ++j)
        {
            EXPECT_EQ(loaded_blocks[i].qs[j], original_blocks[i].qs[j])
                << "Block " << i << " qs[" << j << "] mismatch";
        }
    }
}

/**
 * @brief Verify that the raw data can be reconstructed from dumped Q16_1 bytes
 */
TEST_F(Test__StageDumperNativeFormat, Q16_1_RawDataIsReconstructible)
{
    constexpr int ROWS = 4;
    constexpr int COLS = 64;

    auto tensor = createTestQ16_1Tensor(ROWS, COLS);

    const Q16_1Block *original_blocks = tensor->q16_1_blocks();
    size_t num_blocks = tensor->total_blocks();
    size_t raw_size = tensor->size_bytes();

    StageDumpInfo info;
    info.addInput("test", tensor.get(), ROWS, COLS);

    const void *dump_data = info.inputs[0].data;
    size_t dump_size = info.inputs[0].byte_size;

    ASSERT_EQ(dump_size, raw_size);

    const Q16_1Block *loaded_blocks = static_cast<const Q16_1Block *>(dump_data);

    for (size_t i = 0; i < num_blocks; ++i)
    {
        EXPECT_FLOAT_EQ(loaded_blocks[i].d, original_blocks[i].d)
            << "Block " << i << " scale mismatch";
        EXPECT_EQ(loaded_blocks[i].sum_qs, original_blocks[i].sum_qs)
            << "Block " << i << " sum_qs mismatch";

        for (int j = 0; j < 32; ++j)
        {
            EXPECT_EQ(loaded_blocks[i].qs[j], original_blocks[i].qs[j])
                << "Block " << i << " qs[" << j << "] mismatch";
        }
    }
}

/**
 * @brief Test that legacy addInput with float* still works
 */
TEST_F(Test__StageDumperNativeFormat, AddInputFP32Pointer_LegacyInterface)
{
    constexpr int ROWS = 4;
    constexpr int COLS = 64;

    std::vector<float> data(ROWS * COLS);
    for (size_t i = 0; i < data.size(); ++i)
    {
        data[i] = static_cast<float>(i) * 0.5f;
    }

    StageDumpInfo info;
    info.addInput("legacy_fp32", data.data(), ROWS, COLS);

    ASSERT_EQ(info.inputs.size(), 1);
    const auto &input = info.inputs[0];

    EXPECT_EQ(input.data, data.data());
    EXPECT_STREQ(input.dtype, "FP32");
    EXPECT_EQ(input.element_size, sizeof(float));
    // Legacy interface doesn't set byte_size (stays 0, computed on dump)
    EXPECT_EQ(input.byte_size, 0);
}
