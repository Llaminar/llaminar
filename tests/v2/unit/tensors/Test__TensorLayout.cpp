/**
 * @file Test__TensorLayout.cpp
 * @brief Unit tests for TensorLayout enum and helper functions
 *
 * Phase 3 of Tensor Contracts and Validation Framework.
 * Tests the tensor memory layout contract system.
 */

#include <gtest/gtest.h>
#include "tensors/TensorLayout.h"
#include "tensors/Tensors.h"
#include "tensors/TensorFactory.h"
#include "utils/MPIContext.h"

namespace
{
    using namespace llaminar2;

    // =============================================================================
    // Test Fixture
    // =============================================================================

    class Test__TensorLayout : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            mpi_ctx_ = std::make_unique<MPIContext>(0, 1, MPI_COMM_WORLD);
            factory_ = std::make_unique<TensorFactory>(*mpi_ctx_);
        }

        std::unique_ptr<MPIContext> mpi_ctx_;
        std::unique_ptr<TensorFactory> factory_;
    };

    // =============================================================================
    // TensorLayout Enum Tests
    // =============================================================================

    TEST_F(Test__TensorLayout, EnumValuesAreDistinct)
    {
        // All enum values should be distinct
        std::set<uint8_t> values;
        values.insert(static_cast<uint8_t>(TensorLayout::Q_SEQ_HEAD_DIM));
        values.insert(static_cast<uint8_t>(TensorLayout::Q_HEAD_SEQ_DIM));
        values.insert(static_cast<uint8_t>(TensorLayout::KV_POS_HEAD_DIM));
        values.insert(static_cast<uint8_t>(TensorLayout::KV_HEAD_POS_DIM));
        values.insert(static_cast<uint8_t>(TensorLayout::ROW_MAJOR_2D));
        values.insert(static_cast<uint8_t>(TensorLayout::ROW_MAJOR_1D));
        values.insert(static_cast<uint8_t>(TensorLayout::UNKNOWN));

        EXPECT_EQ(values.size(), 7) << "All TensorLayout enum values should be distinct";
    }

    // =============================================================================
    // layoutName() Tests
    // =============================================================================

    TEST_F(Test__TensorLayout, LayoutNameReturnsCorrectStrings)
    {
        EXPECT_STREQ(layoutName(TensorLayout::Q_SEQ_HEAD_DIM),
                     "Q_SEQ_HEAD_DIM [seq][n_heads][head_dim]");
        EXPECT_STREQ(layoutName(TensorLayout::Q_HEAD_SEQ_DIM),
                     "Q_HEAD_SEQ_DIM [n_heads][seq][head_dim]");
        EXPECT_STREQ(layoutName(TensorLayout::KV_POS_HEAD_DIM),
                     "KV_POS_HEAD_DIM [pos][n_kv_heads][head_dim]");
        EXPECT_STREQ(layoutName(TensorLayout::KV_HEAD_POS_DIM),
                     "KV_HEAD_POS_DIM [n_kv_heads][pos][head_dim]");
        EXPECT_STREQ(layoutName(TensorLayout::ROW_MAJOR_2D),
                     "ROW_MAJOR_2D [rows][cols]");
        EXPECT_STREQ(layoutName(TensorLayout::ROW_MAJOR_1D),
                     "ROW_MAJOR_1D [elements]");
        EXPECT_STREQ(layoutName(TensorLayout::UNKNOWN),
                     "UNKNOWN");
    }

    TEST_F(Test__TensorLayout, LayoutNameShortReturnsCorrectStrings)
    {
        EXPECT_STREQ(layoutNameShort(TensorLayout::Q_SEQ_HEAD_DIM), "Q_SHD");
        EXPECT_STREQ(layoutNameShort(TensorLayout::Q_HEAD_SEQ_DIM), "Q_HSD");
        EXPECT_STREQ(layoutNameShort(TensorLayout::KV_POS_HEAD_DIM), "KV_PHD");
        EXPECT_STREQ(layoutNameShort(TensorLayout::KV_HEAD_POS_DIM), "KV_HPD");
        EXPECT_STREQ(layoutNameShort(TensorLayout::ROW_MAJOR_2D), "2D");
        EXPECT_STREQ(layoutNameShort(TensorLayout::ROW_MAJOR_1D), "1D");
        EXPECT_STREQ(layoutNameShort(TensorLayout::UNKNOWN), "UNK");
    }

    // =============================================================================
    // layoutsCompatible() Tests
    // =============================================================================

    TEST_F(Test__TensorLayout, SameLayoutIsCompatible)
    {
        EXPECT_TRUE(layoutsCompatible(TensorLayout::Q_SEQ_HEAD_DIM, TensorLayout::Q_SEQ_HEAD_DIM));
        EXPECT_TRUE(layoutsCompatible(TensorLayout::KV_POS_HEAD_DIM, TensorLayout::KV_POS_HEAD_DIM));
        EXPECT_TRUE(layoutsCompatible(TensorLayout::KV_HEAD_POS_DIM, TensorLayout::KV_HEAD_POS_DIM));
        EXPECT_TRUE(layoutsCompatible(TensorLayout::ROW_MAJOR_2D, TensorLayout::ROW_MAJOR_2D));
        EXPECT_TRUE(layoutsCompatible(TensorLayout::UNKNOWN, TensorLayout::UNKNOWN));
    }

    TEST_F(Test__TensorLayout, UnknownIsCompatibleWithAnything)
    {
        // UNKNOWN layout is compatible with everything (permissive for migration)
        EXPECT_TRUE(layoutsCompatible(TensorLayout::UNKNOWN, TensorLayout::Q_SEQ_HEAD_DIM));
        EXPECT_TRUE(layoutsCompatible(TensorLayout::Q_SEQ_HEAD_DIM, TensorLayout::UNKNOWN));
        EXPECT_TRUE(layoutsCompatible(TensorLayout::UNKNOWN, TensorLayout::KV_POS_HEAD_DIM));
        EXPECT_TRUE(layoutsCompatible(TensorLayout::KV_HEAD_POS_DIM, TensorLayout::UNKNOWN));
        EXPECT_TRUE(layoutsCompatible(TensorLayout::UNKNOWN, TensorLayout::ROW_MAJOR_2D));
    }

    TEST_F(Test__TensorLayout, RowMajor2DIsCompatibleWithSpecificLayouts)
    {
        // ROW_MAJOR_2D can be reinterpreted as specific layouts
        EXPECT_TRUE(layoutsCompatible(TensorLayout::ROW_MAJOR_2D, TensorLayout::Q_SEQ_HEAD_DIM));
        EXPECT_TRUE(layoutsCompatible(TensorLayout::Q_SEQ_HEAD_DIM, TensorLayout::ROW_MAJOR_2D));
        EXPECT_TRUE(layoutsCompatible(TensorLayout::ROW_MAJOR_2D, TensorLayout::KV_POS_HEAD_DIM));
        EXPECT_TRUE(layoutsCompatible(TensorLayout::KV_HEAD_POS_DIM, TensorLayout::ROW_MAJOR_2D));
    }

    TEST_F(Test__TensorLayout, IncompatibleLayoutsRequireTranspose)
    {
        // Different specific layouts require transpose
        EXPECT_FALSE(layoutsCompatible(TensorLayout::Q_SEQ_HEAD_DIM, TensorLayout::Q_HEAD_SEQ_DIM));
        EXPECT_FALSE(layoutsCompatible(TensorLayout::KV_POS_HEAD_DIM, TensorLayout::KV_HEAD_POS_DIM));
        EXPECT_FALSE(layoutsCompatible(TensorLayout::Q_SEQ_HEAD_DIM, TensorLayout::KV_POS_HEAD_DIM));
    }

    // =============================================================================
    // Layout Classification Tests
    // =============================================================================

    TEST_F(Test__TensorLayout, IsQueryLayout)
    {
        EXPECT_TRUE(isQueryLayout(TensorLayout::Q_SEQ_HEAD_DIM));
        EXPECT_TRUE(isQueryLayout(TensorLayout::Q_HEAD_SEQ_DIM));
        EXPECT_FALSE(isQueryLayout(TensorLayout::KV_POS_HEAD_DIM));
        EXPECT_FALSE(isQueryLayout(TensorLayout::KV_HEAD_POS_DIM));
        EXPECT_FALSE(isQueryLayout(TensorLayout::ROW_MAJOR_2D));
        EXPECT_FALSE(isQueryLayout(TensorLayout::UNKNOWN));
    }

    TEST_F(Test__TensorLayout, IsKVLayout)
    {
        EXPECT_FALSE(isKVLayout(TensorLayout::Q_SEQ_HEAD_DIM));
        EXPECT_FALSE(isKVLayout(TensorLayout::Q_HEAD_SEQ_DIM));
        EXPECT_TRUE(isKVLayout(TensorLayout::KV_POS_HEAD_DIM));
        EXPECT_TRUE(isKVLayout(TensorLayout::KV_HEAD_POS_DIM));
        EXPECT_FALSE(isKVLayout(TensorLayout::ROW_MAJOR_2D));
        EXPECT_FALSE(isKVLayout(TensorLayout::UNKNOWN));
    }

    TEST_F(Test__TensorLayout, IsGenericLayout)
    {
        EXPECT_FALSE(isGenericLayout(TensorLayout::Q_SEQ_HEAD_DIM));
        EXPECT_FALSE(isGenericLayout(TensorLayout::Q_HEAD_SEQ_DIM));
        EXPECT_FALSE(isGenericLayout(TensorLayout::KV_POS_HEAD_DIM));
        EXPECT_FALSE(isGenericLayout(TensorLayout::KV_HEAD_POS_DIM));
        EXPECT_TRUE(isGenericLayout(TensorLayout::ROW_MAJOR_2D));
        EXPECT_TRUE(isGenericLayout(TensorLayout::ROW_MAJOR_1D));
        EXPECT_TRUE(isGenericLayout(TensorLayout::UNKNOWN));
    }

    // =============================================================================
    // Transpose Target Tests
    // =============================================================================

    TEST_F(Test__TensorLayout, KVTransposeTarget)
    {
        EXPECT_EQ(kvTransposeTarget(TensorLayout::KV_POS_HEAD_DIM), TensorLayout::KV_HEAD_POS_DIM);
        EXPECT_EQ(kvTransposeTarget(TensorLayout::KV_HEAD_POS_DIM), TensorLayout::KV_POS_HEAD_DIM);
        EXPECT_EQ(kvTransposeTarget(TensorLayout::Q_SEQ_HEAD_DIM), TensorLayout::UNKNOWN);
        EXPECT_EQ(kvTransposeTarget(TensorLayout::ROW_MAJOR_2D), TensorLayout::UNKNOWN);
    }

    TEST_F(Test__TensorLayout, QueryTransposeTarget)
    {
        EXPECT_EQ(queryTransposeTarget(TensorLayout::Q_SEQ_HEAD_DIM), TensorLayout::Q_HEAD_SEQ_DIM);
        EXPECT_EQ(queryTransposeTarget(TensorLayout::Q_HEAD_SEQ_DIM), TensorLayout::Q_SEQ_HEAD_DIM);
        EXPECT_EQ(queryTransposeTarget(TensorLayout::KV_POS_HEAD_DIM), TensorLayout::UNKNOWN);
        EXPECT_EQ(queryTransposeTarget(TensorLayout::ROW_MAJOR_2D), TensorLayout::UNKNOWN);
    }

    // =============================================================================
    // TensorBase Layout Integration Tests
    // =============================================================================

    TEST_F(Test__TensorLayout, TensorBaseDefaultLayoutIsUnknown)
    {
        auto tensor = factory_->createFP32({16, 64});
        EXPECT_EQ(tensor->layout(), TensorLayout::UNKNOWN);
    }

    TEST_F(Test__TensorLayout, TensorBaseSetLayoutWorks)
    {
        auto tensor = factory_->createFP32({16, 64});

        // Set Q layout
        tensor->setLayout(TensorLayout::Q_SEQ_HEAD_DIM);
        EXPECT_EQ(tensor->layout(), TensorLayout::Q_SEQ_HEAD_DIM);

        // Change to KV layout
        tensor->setLayout(TensorLayout::KV_POS_HEAD_DIM);
        EXPECT_EQ(tensor->layout(), TensorLayout::KV_POS_HEAD_DIM);

        // Reset to UNKNOWN
        tensor->setLayout(TensorLayout::UNKNOWN);
        EXPECT_EQ(tensor->layout(), TensorLayout::UNKNOWN);
    }

    TEST_F(Test__TensorLayout, MultipleTypesSupportsLayout)
    {
        // FP32
        auto fp32 = factory_->createFP32({8, 64});
        fp32->setLayout(TensorLayout::Q_SEQ_HEAD_DIM);
        EXPECT_EQ(fp32->layout(), TensorLayout::Q_SEQ_HEAD_DIM);

        // BF16
        auto bf16 = factory_->createBF16({8, 64});
        bf16->setLayout(TensorLayout::KV_HEAD_POS_DIM);
        EXPECT_EQ(bf16->layout(), TensorLayout::KV_HEAD_POS_DIM);

        // Q16_1
        auto q16 = factory_->createQ16_1({8, 64});
        q16->setLayout(TensorLayout::KV_POS_HEAD_DIM);
        EXPECT_EQ(q16->layout(), TensorLayout::KV_POS_HEAD_DIM);

        // Q8_1
        auto q8 = factory_->createQ8_1({8, 64});
        q8->setLayout(TensorLayout::ROW_MAJOR_2D);
        EXPECT_EQ(q8->layout(), TensorLayout::ROW_MAJOR_2D);
    }

    TEST_F(Test__TensorLayout, LayoutPreservedAfterCopy)
    {
        auto src = factory_->createFP32({8, 64});
        src->setLayout(TensorLayout::Q_SEQ_HEAD_DIM);

        // Fill with some data
        float *data = src->mutable_data();
        for (size_t i = 0; i < src->numel(); ++i)
        {
            data[i] = static_cast<float>(i);
        }

        // Create destination and copy
        auto dst = factory_->createFP32({8, 64});
        dst->setLayout(TensorLayout::Q_SEQ_HEAD_DIM);
        dst->copyFrom(src.get());

        // Layout should be preserved (set on dst before copy)
        EXPECT_EQ(dst->layout(), TensorLayout::Q_SEQ_HEAD_DIM);
    }

    // =============================================================================
    // Layout Contract Enforcement Tests (informational - not hard errors yet)
    // =============================================================================

    TEST_F(Test__TensorLayout, LayoutCompatibilityCheckForAttention)
    {
        // Simulate attention stage inputs
        auto Q = factory_->createFP32({4, 128}); // 4 tokens, 128 = 2 heads * 64 head_dim
        auto K = factory_->createFP32({16, 64}); // 16 positions, 64 = 1 kv_head * 64
        auto V = factory_->createFP32({16, 64});

        // Set expected layouts
        Q->setLayout(TensorLayout::Q_SEQ_HEAD_DIM);
        K->setLayout(TensorLayout::KV_HEAD_POS_DIM);
        V->setLayout(TensorLayout::KV_HEAD_POS_DIM);

        // Verify layouts are set correctly
        EXPECT_EQ(Q->layout(), TensorLayout::Q_SEQ_HEAD_DIM);
        EXPECT_EQ(K->layout(), TensorLayout::KV_HEAD_POS_DIM);
        EXPECT_EQ(V->layout(), TensorLayout::KV_HEAD_POS_DIM);

        // Check that Q is a query layout
        EXPECT_TRUE(isQueryLayout(Q->layout()));
        EXPECT_FALSE(isKVLayout(Q->layout()));

        // Check that K/V are KV layouts
        EXPECT_TRUE(isKVLayout(K->layout()));
        EXPECT_TRUE(isKVLayout(V->layout()));
    }

    TEST_F(Test__TensorLayout, DetectLayoutMismatch)
    {
        auto cache_K = factory_->createFP32({16, 64});
        cache_K->setLayout(TensorLayout::KV_POS_HEAD_DIM); // Position-major (cache storage)

        // Q16 attention kernel expects head-major
        TensorLayout expected_for_kernel = TensorLayout::KV_HEAD_POS_DIM;

        // Check if transpose is needed
        bool needs_transpose = !layoutsCompatible(cache_K->layout(), expected_for_kernel);
        EXPECT_TRUE(needs_transpose) << "Position-major cache should need transpose for head-major kernel";
    }

} // anonymous namespace
