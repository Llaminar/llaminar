/**
 * @file Test__IQ4_NLTensor_Views.cpp
 * @brief Unit tests for IQ4_NL tensor view support
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "tensors/IQQuantTables.h"
#include <memory>
#include <vector>
#include <cstring>

using namespace llaminar2;

/**
 * @brief Test fixture for IQ4_NL tensor view tests
 */
class Test__IQ4_NLTensor_Views : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // IQ4_NL quantization tables are initialized statically
    }

    /**
     * @brief Create a simple IQ4_NL tensor with known pattern for testing
     *
     * Creates a tensor where each block has a unique scale value
     * to make verification easier.
     */
    std::shared_ptr<IQ4_NLTensor> createTestTensor(size_t rows, size_t cols)
    {
        // Calculate number of blocks per row
        size_t blocks_per_row = (cols + IQ4_NLBlock::BLOCK_SIZE - 1) / IQ4_NLBlock::BLOCK_SIZE;
        size_t total_blocks = rows * blocks_per_row;
        size_t raw_size = total_blocks * sizeof(IQ4_NLBlock);

        std::vector<uint8_t> raw_data(raw_size);
        IQ4_NLBlock *blocks = reinterpret_cast<IQ4_NLBlock *>(raw_data.data());

        // Fill with a pattern: each block has scale = (row_idx * blocks_per_row + block_idx)
        for (size_t r = 0; r < rows; ++r)
        {
            for (size_t b = 0; b < blocks_per_row; ++b)
            {
                size_t block_idx = r * blocks_per_row + b;
                float scale = static_cast<float>(block_idx);

                // Convert scale to FP16 and store
                uint32_t bits;
                std::memcpy(&bits, &scale, sizeof(float));
                uint32_t sign = (bits >> 16) & 0x8000;
                int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFF) - 112;
                uint32_t mantissa = (bits >> 13) & 0x3FF;

                if (exponent <= 0)
                {
                    blocks[block_idx].d = static_cast<uint16_t>(sign);
                }
                else if (exponent >= 0x1F)
                {
                    blocks[block_idx].d = static_cast<uint16_t>(sign | 0x7C00);
                }
                else
                {
                    blocks[block_idx].d = static_cast<uint16_t>(sign | (exponent << 10) | mantissa);
                }

                // Fill quantized data with pattern
                for (size_t i = 0; i < 16; ++i)
                {
                    blocks[block_idx].qs[i] = static_cast<uint8_t>((block_idx + i) & 0xFF);
                }
            }
        }

        return std::make_shared<IQ4_NLTensor>(std::vector<size_t>{rows, cols}, raw_data);
    }
};

/**
 * @brief Test basic view creation with zero offset
 */
TEST_F(Test__IQ4_NLTensor_Views, BasicViewCreation)
{
    // Create parent: 10 rows × 64 cols (2 blocks per row)
    auto parent = createTestTensor(10, 64);

    // Create view of first 5 rows
    auto view = parent->create_view({5, 64}, 0);

    ASSERT_NE(view, nullptr) << "View creation failed";
    EXPECT_TRUE(view->is_view());
    EXPECT_EQ(view->shape().size(), 2);
    EXPECT_EQ(view->shape()[0], 5);
    EXPECT_EQ(view->shape()[1], 64);
}

/**
 * @brief Test view creation with non-zero offset
 */
TEST_F(Test__IQ4_NLTensor_Views, ViewWithOffset)
{
    // Create parent: 10 rows × 64 cols
    auto parent = createTestTensor(10, 64);

    // Create view of rows 5-9 (offset = 5 × 64 = 320 elements)
    auto view = parent->create_view({5, 64}, 5 * 64);

    ASSERT_NE(view, nullptr) << "View creation with offset failed";
    EXPECT_TRUE(view->is_view());
    EXPECT_EQ(view->shape()[0], 5);
    EXPECT_EQ(view->shape()[1], 64);
}

/**
 * @brief Test row-slice restriction - K dimension must match
 */
TEST_F(Test__IQ4_NLTensor_Views, KDimensionMustMatch)
{
    auto parent = createTestTensor(10, 64);

    // Try to create view with different K dimension - should fail
    EXPECT_THROW(parent->create_view({5, 32}, 0), std::invalid_argument);
}

/**
 * @brief Test offset must be row-aligned
 */
TEST_F(Test__IQ4_NLTensor_Views, OffsetMustBeRowAligned)
{
    auto parent = createTestTensor(10, 64);

    // Try to create view with non-row-aligned offset - should fail
    EXPECT_THROW(parent->create_view({5, 64}, 32), std::invalid_argument);
}

/**
 * @brief Test view bounds checking
 */
TEST_F(Test__IQ4_NLTensor_Views, ViewBoundsChecking)
{
    auto parent = createTestTensor(10, 64);

    // Try to create view that exceeds parent bounds - should fail
    EXPECT_THROW(parent->create_view({8, 64}, 5 * 64), std::out_of_range);
}

/**
 * @brief Test view data access via decode
 */
TEST_F(Test__IQ4_NLTensor_Views, ViewDataAccess)
{
    // Create parent with distinct pattern per row
    auto parent = createTestTensor(10, 64);

    // Create view of rows 5-9
    auto view = parent->create_view({5, 64}, 5 * 64);
    ASSERT_NE(view, nullptr);

    // Decode both parent and view
    std::vector<float> parent_data(10 * 64);
    parent->to_fp32(parent_data.data());

    std::vector<float> view_data(5 * 64);
    auto *view_iq4nl = dynamic_cast<IQ4_NLTensor *>(view.get());
    ASSERT_NE(view_iq4nl, nullptr);
    view_iq4nl->to_fp32(view_data.data());

    // View should contain rows 5-9 from parent
    for (size_t i = 0; i < 5 * 64; ++i)
    {
        size_t parent_idx = (5 * 64) + i; // Offset by 5 rows
        EXPECT_FLOAT_EQ(view_data[i], parent_data[parent_idx])
            << "View data mismatch at index " << i;
    }
}

/**
 * @brief Test view lifetime - parent kept alive by view
 */
TEST_F(Test__IQ4_NLTensor_Views, ViewLifetime)
{
    std::shared_ptr<TensorBase> view;

    {
        auto parent = createTestTensor(10, 64);
        view = parent->create_view({5, 64}, 0);
        ASSERT_NE(view, nullptr);
        // parent goes out of scope here
    }

    // View should still be valid and accessible
    EXPECT_TRUE(view->is_view());
    EXPECT_EQ(view->shape()[0], 5);

    // Should be able to decode from view
    auto *view_iq4nl = dynamic_cast<IQ4_NLTensor *>(view.get());
    ASSERT_NE(view_iq4nl, nullptr);

    std::vector<float> data(5 * 64);
    EXPECT_NO_THROW({
        view_iq4nl->to_fp32(data.data());
    });
}

/**
 * @brief Test view chaining - view of view chains to root
 */
TEST_F(Test__IQ4_NLTensor_Views, ViewChaining)
{
    auto parent = createTestTensor(20, 64);

    // Create first view: rows 5-14 (10 rows)
    auto view1 = parent->create_view({10, 64}, 5 * 64);
    ASSERT_NE(view1, nullptr);

    // Create view of view: rows 2-6 of view1 (which is rows 7-11 of parent)
    auto view2 = view1->create_view({5, 64}, 2 * 64);
    ASSERT_NE(view2, nullptr);

    EXPECT_TRUE(view2->is_view());
    EXPECT_EQ(view2->shape()[0], 5);

    // Decode all three and verify view2 = parent[7:12]
    std::vector<float> parent_data(20 * 64);
    parent->to_fp32(parent_data.data());

    std::vector<float> view2_data(5 * 64);
    auto *view2_iq4nl = dynamic_cast<IQ4_NLTensor *>(view2.get());
    view2_iq4nl->to_fp32(view2_data.data());

    // view2 should equal parent rows 7-11
    for (size_t i = 0; i < 5 * 64; ++i)
    {
        size_t parent_idx = (7 * 64) + i;
        EXPECT_FLOAT_EQ(view2_data[i], parent_data[parent_idx])
            << "Chained view data mismatch at index " << i;
    }
}

/**
 * @brief Test ITensorGemmTileDataProvider interface works correctly for views
 */
TEST_F(Test__IQ4_NLTensor_Views, IBlockDecoderInterface)
{
    auto parent = createTestTensor(10, 64);
    auto view = parent->create_view({5, 64}, 5 * 64); // Rows 5-9
    ASSERT_NE(view, nullptr);

    auto *view_iq4nl = dynamic_cast<IQ4_NLTensor *>(view.get());
    ASSERT_NE(view_iq4nl, nullptr);

    // Decode first block of view using ITensorGemmTileDataProvider interface
    float decoded_block[IQ4_NLBlock::BLOCK_SIZE];
    view_iq4nl->decode_block_at(0, 0, decoded_block);

    // Should match first block of row 5 from parent
    auto *parent_iq4nl = dynamic_cast<IQ4_NLTensor *>(parent.get());
    float parent_block[IQ4_NLBlock::BLOCK_SIZE];
    parent_iq4nl->decode_block_at(5, 0, parent_block);

    for (size_t i = 0; i < IQ4_NLBlock::BLOCK_SIZE; ++i)
    {
        EXPECT_FLOAT_EQ(decoded_block[i], parent_block[i])
            << "ITensorGemmTileDataProvider mismatch at element " << i;
    }
}

/**
 * @brief Test multiple concurrent views
 */
TEST_F(Test__IQ4_NLTensor_Views, MultipleViews)
{
    auto parent = createTestTensor(20, 64);

    // Create three non-overlapping views
    auto view1 = parent->create_view({5, 64}, 0 * 64);  // Rows 0-4
    auto view2 = parent->create_view({5, 64}, 5 * 64);  // Rows 5-9
    auto view3 = parent->create_view({5, 64}, 10 * 64); // Rows 10-14

    ASSERT_NE(view1, nullptr);
    ASSERT_NE(view2, nullptr);
    ASSERT_NE(view3, nullptr);

    // All should be valid
    EXPECT_TRUE(view1->is_view());
    EXPECT_TRUE(view2->is_view());
    EXPECT_TRUE(view3->is_view());

    // Decode all and verify independence
    auto *v1_iq4nl = dynamic_cast<IQ4_NLTensor *>(view1.get());
    auto *v2_iq4nl = dynamic_cast<IQ4_NLTensor *>(view2.get());
    auto *v3_iq4nl = dynamic_cast<IQ4_NLTensor *>(view3.get());

    std::vector<float> data1(5 * 64), data2(5 * 64), data3(5 * 64);
    v1_iq4nl->to_fp32(data1.data());
    v2_iq4nl->to_fp32(data2.data());
    v3_iq4nl->to_fp32(data3.data());

    // Views should have distinct data (based on our test pattern)
    bool all_same = true;
    for (size_t i = 0; i < 64; ++i)
    {
        if (data1[i] != data2[i] || data2[i] != data3[i])
        {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same) << "Multiple views should have distinct data";
}

/**
 * @brief Test view with non-standard column count (tail blocks)
 */
TEST_F(Test__IQ4_NLTensor_Views, NonStandardColumns)
{
    // Create tensor with cols not multiple of 32
    auto parent = createTestTensor(10, 50); // 50 cols = 1 full block + 18 element tail

    // Create view
    auto view = parent->create_view({5, 50}, 0);

    ASSERT_NE(view, nullptr) << "View with non-standard columns failed";
    EXPECT_EQ(view->shape()[1], 50);

    // Decode should work correctly
    auto *view_iq4nl = dynamic_cast<IQ4_NLTensor *>(view.get());
    std::vector<float> data(5 * 50);
    EXPECT_NO_THROW({
        view_iq4nl->to_fp32(data.data());
    });
}

/**
 * @brief Test decodeRow works correctly for views
 */
TEST_F(Test__IQ4_NLTensor_Views, DecodeRowFromView)
{
    auto parent = createTestTensor(10, 64);
    auto view = parent->create_view({5, 64}, 5 * 64); // Rows 5-9 of parent
    ASSERT_NE(view, nullptr);

    auto *view_iq4nl = dynamic_cast<IQ4_NLTensor *>(view.get());
    auto *parent_iq4nl = dynamic_cast<IQ4_NLTensor *>(parent.get());

    // Decode row 0 from view (which is row 5 from parent)
    std::vector<float> view_row(64);
    std::vector<float> parent_row(64);

    view_iq4nl->to_fp32_row(0, view_row.data());
    parent_iq4nl->to_fp32_row(5, parent_row.data());

    for (size_t i = 0; i < 64; ++i)
    {
        EXPECT_FLOAT_EQ(view_row[i], parent_row[i])
            << "decodeRow mismatch at column " << i;
    }
}

// --- 3D Parent → 2D View regression tests ---
// NOTE: IQ4_NLTensor constructor does not yet support 3D shapes.
// These tests verify the rejection and will be updated when 3D support is added.

TEST_F(Test__IQ4_NLTensor_Views, View3DParentBasicSlice)
{
    // 3D GGUF shape: [cols=128(ne[0]), rows_per_expert=8(ne[1]), num_experts=4(ne[2])]
    const size_t N = 4, R = 8, K = 128;
    const size_t blocks_per_row = K / IQ4_NLBlock::BLOCK_SIZE;
    const size_t total_blocks = N * R * blocks_per_row;
    std::vector<uint8_t> raw(total_blocks * sizeof(IQ4_NLBlock));

    // IQ4_NL doesn't support 3D shapes yet - constructor should throw
    EXPECT_THROW(
        std::make_shared<IQ4_NLTensor>(std::vector<size_t>{K, R, N}, raw),
        std::invalid_argument);
}

TEST_F(Test__IQ4_NLTensor_Views, View3DParentWithOffset)
{
    const size_t N = 4, R = 8, K = 128;
    const size_t blocks_per_row = K / IQ4_NLBlock::BLOCK_SIZE;
    const size_t total_blocks = N * R * blocks_per_row;
    std::vector<uint8_t> raw(total_blocks * sizeof(IQ4_NLBlock));

    // IQ4_NL doesn't support 3D shapes yet - constructor should throw
    EXPECT_THROW(
        std::make_shared<IQ4_NLTensor>(std::vector<size_t>{K, R, N}, raw),
        std::invalid_argument);
}

TEST_F(Test__IQ4_NLTensor_Views, View3DParentBoundsCheck)
{
    const size_t N = 4, R = 8, K = 128;
    const size_t blocks_per_row = K / IQ4_NLBlock::BLOCK_SIZE;
    const size_t total_blocks = N * R * blocks_per_row;
    std::vector<uint8_t> raw(total_blocks * sizeof(IQ4_NLBlock));

    // IQ4_NL doesn't support 3D shapes yet - constructor should throw
    EXPECT_THROW(
        std::make_shared<IQ4_NLTensor>(std::vector<size_t>{K, R, N}, raw),
        std::invalid_argument);
}
