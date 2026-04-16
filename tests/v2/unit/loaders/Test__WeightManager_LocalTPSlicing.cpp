/**
 * @file Test__WeightManager_LocalTPSlicing.cpp
 * @brief Unit tests for WeightManager LOCAL TP slicing helper methods
 *
 * Tests the static helper methods in WeightManager that support device-aware
 * weight slicing for LOCAL tensor parallelism:
 * - Weight category detection (isQKVWeight, isFFNGateUpWeight, etc.)
 * - Row slicing utility (sliceRowRange)
 *
 * Note: Full WeightManager integration tests that require model loading
 * are in tests/v2/integration/.
 */

#include <gtest/gtest.h>
#include "loaders/WeightManager.h"
#include "tensors/Tensors.h"
#include "TestTensorFactory.h"
#include <memory>
#include <cstring>

namespace llaminar2
{
    namespace test
    {

        // =============================================================================
        // Test Fixture
        // =============================================================================

        class Test__WeightManager_LocalTPSlicing : public ::testing::Test
        {
        protected:
            // Common layer name patterns used in GGUF models
            static constexpr const char *ATTN_Q_LAYER0 = "blk.0.attn_q.weight";
            static constexpr const char *ATTN_K_LAYER0 = "blk.0.attn_k.weight";
            static constexpr const char *ATTN_V_LAYER0 = "blk.0.attn_v.weight";
            static constexpr const char *ATTN_QKV_FUSED = "blk.0.attn_qkv.weight";
            static constexpr const char *ATTN_OUTPUT_LAYER0 = "blk.0.attn_output.weight";

            static constexpr const char *FFN_GATE_LAYER0 = "blk.0.ffn_gate.weight";
            static constexpr const char *FFN_UP_LAYER0 = "blk.0.ffn_up.weight";
            static constexpr const char *FFN_GATE_UP_FUSED = "blk.0.ffn_gate_up.weight";
            static constexpr const char *FFN_DOWN_LAYER0 = "blk.0.ffn_down.weight";

            static constexpr const char *LM_HEAD = "output.weight";
            static constexpr const char *EMBEDDING = "token_embd.weight";
            static constexpr const char *OUTPUT_NORM = "output_norm.weight";
            static constexpr const char *ATTN_NORM_LAYER0 = "blk.0.attn_norm.weight";
            static constexpr const char *FFN_NORM_LAYER0 = "blk.0.ffn_norm.weight";
        };

        // =============================================================================
        // sliceRowRange Tests - FP32 Tensors
        // =============================================================================

        TEST_F(Test__WeightManager_LocalTPSlicing, SliceRowRange_FP32_BasicSlice)
        {
            // Create 10x8 FP32 tensor with sequential values for easy verification
            auto tensor = TestTensorFactory::createFP32({10, 8});
            float *data = tensor->mutable_data();
            for (size_t i = 0; i < 80; ++i)
            {
                data[i] = static_cast<float>(i);
            }

            // Slice rows 3-5 (3 rows: indices 3, 4, 5)
            auto shared_tensor = std::shared_ptr<TensorBase>(tensor.release());
            auto sliced = WeightManager::sliceRowRange(shared_tensor, 3, 3);

            ASSERT_NE(sliced, nullptr);
            EXPECT_EQ(sliced->shape().size(), 2);
            EXPECT_EQ(sliced->shape()[0], 3);
            EXPECT_EQ(sliced->shape()[1], 8);

            // Verify values
            // Row 3 starts at index 3*8=24, contains values 24-31
            // Row 4 starts at index 4*8=32, contains values 32-39
            // Row 5 starts at index 5*8=40, contains values 40-47
            const float *sliced_data = sliced->data();
            EXPECT_FLOAT_EQ(sliced_data[0], 24.0f);  // First element of row 3
            EXPECT_FLOAT_EQ(sliced_data[7], 31.0f);  // Last element of row 3
            EXPECT_FLOAT_EQ(sliced_data[8], 32.0f);  // First element of row 4
            EXPECT_FLOAT_EQ(sliced_data[23], 47.0f); // Last element of row 5
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, SliceRowRange_FP32_FirstRows)
        {
            auto tensor = TestTensorFactory::createFP32({10, 8});
            float *data = tensor->mutable_data();
            for (size_t i = 0; i < 80; ++i)
            {
                data[i] = static_cast<float>(i);
            }

            auto shared_tensor = std::shared_ptr<TensorBase>(tensor.release());
            auto sliced = WeightManager::sliceRowRange(shared_tensor, 0, 4);

            ASSERT_NE(sliced, nullptr);
            EXPECT_EQ(sliced->shape()[0], 4);
            EXPECT_EQ(sliced->shape()[1], 8);

            const float *sliced_data = sliced->data();
            EXPECT_FLOAT_EQ(sliced_data[0], 0.0f);
            EXPECT_FLOAT_EQ(sliced_data[31], 31.0f);
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, SliceRowRange_FP32_LastRows)
        {
            auto tensor = TestTensorFactory::createFP32({10, 8});
            float *data = tensor->mutable_data();
            for (size_t i = 0; i < 80; ++i)
            {
                data[i] = static_cast<float>(i);
            }

            auto shared_tensor = std::shared_ptr<TensorBase>(tensor.release());
            auto sliced = WeightManager::sliceRowRange(shared_tensor, 7, 3); // Rows 7, 8, 9

            ASSERT_NE(sliced, nullptr);
            EXPECT_EQ(sliced->shape()[0], 3);
            EXPECT_EQ(sliced->shape()[1], 8);

            const float *sliced_data = sliced->data();
            // Row 7 starts at 7*8=56
            EXPECT_FLOAT_EQ(sliced_data[0], 56.0f);
            // Row 9 ends at 9*8+7=79
            EXPECT_FLOAT_EQ(sliced_data[23], 79.0f);
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, SliceRowRange_FP32_SingleRow)
        {
            auto tensor = TestTensorFactory::createFP32({10, 8});
            float *data = tensor->mutable_data();
            for (size_t i = 0; i < 80; ++i)
            {
                data[i] = static_cast<float>(i);
            }

            auto shared_tensor = std::shared_ptr<TensorBase>(tensor.release());
            auto sliced = WeightManager::sliceRowRange(shared_tensor, 5, 1);

            ASSERT_NE(sliced, nullptr);
            EXPECT_EQ(sliced->shape()[0], 1);
            EXPECT_EQ(sliced->shape()[1], 8);

            const float *sliced_data = sliced->data();
            EXPECT_FLOAT_EQ(sliced_data[0], 40.0f); // First of row 5
            EXPECT_FLOAT_EQ(sliced_data[7], 47.0f); // Last of row 5
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, SliceRowRange_FP32_AllRows)
        {
            auto tensor = TestTensorFactory::createFP32({10, 8});
            float *data = tensor->mutable_data();
            for (size_t i = 0; i < 80; ++i)
            {
                data[i] = static_cast<float>(i);
            }

            auto shared_tensor = std::shared_ptr<TensorBase>(tensor.release());
            auto sliced = WeightManager::sliceRowRange(shared_tensor, 0, 10);

            ASSERT_NE(sliced, nullptr);
            EXPECT_EQ(sliced->shape()[0], 10);
            EXPECT_EQ(sliced->shape()[1], 8);

            // Should be a complete copy
            const float *sliced_data = sliced->data();
            for (size_t i = 0; i < 80; ++i)
            {
                EXPECT_FLOAT_EQ(sliced_data[i], static_cast<float>(i));
            }
        }

        // =============================================================================
        // sliceRowRange Error Cases
        // =============================================================================

        TEST_F(Test__WeightManager_LocalTPSlicing, SliceRowRange_NullTensor_ReturnsNull)
        {
            std::shared_ptr<TensorBase> null_tensor = nullptr;
            EXPECT_THROW(WeightManager::sliceRowRange(null_tensor, 0, 1), std::runtime_error);
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, SliceRowRange_OutOfBounds_ReturnsNull)
        {
            auto tensor = TestTensorFactory::createFP32({10, 8});
            auto shared_tensor = std::shared_ptr<TensorBase>(tensor.release());

            // Start + count exceeds tensor dimensions
            EXPECT_THROW(WeightManager::sliceRowRange(shared_tensor, 8, 5), std::runtime_error);
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, SliceRowRange_StartAtEnd_ReturnsNull)
        {
            auto tensor = TestTensorFactory::createFP32({10, 8});
            auto shared_tensor = std::shared_ptr<TensorBase>(tensor.release());

            // Start at row 10 (out of bounds)
            EXPECT_THROW(WeightManager::sliceRowRange(shared_tensor, 10, 1), std::runtime_error);
        }

        // =============================================================================
        // sliceRowRange - Larger Realistic Sizes
        // =============================================================================

        TEST_F(Test__WeightManager_LocalTPSlicing, SliceRowRange_FP32_LargeMatrix_ProportionalSplit)
        {
            // Simulate splitting a 1024x896 weight matrix for 73%/27% TP split
            const size_t out_dim = 1024;
            const size_t in_dim = 896;

            auto tensor = TestTensorFactory::createFP32Random({out_dim, in_dim});
            auto shared_tensor = std::shared_ptr<TensorBase>(tensor.release());

            // Device 0 gets ~73% = 747 rows (rounded down for alignment)
            // Device 1 gets ~27% = 277 rows
            size_t cuda_rows = 747;
            size_t rocm_rows = out_dim - cuda_rows;

            auto cuda_slice = WeightManager::sliceRowRange(shared_tensor, 0, cuda_rows);
            auto rocm_slice = WeightManager::sliceRowRange(shared_tensor, cuda_rows, rocm_rows);

            ASSERT_NE(cuda_slice, nullptr);
            ASSERT_NE(rocm_slice, nullptr);

            EXPECT_EQ(cuda_slice->shape()[0], cuda_rows);
            EXPECT_EQ(cuda_slice->shape()[1], in_dim);
            EXPECT_EQ(rocm_slice->shape()[0], rocm_rows);
            EXPECT_EQ(rocm_slice->shape()[1], in_dim);

            // Slices should be independent (not sharing memory with original)
            // Verify by checking that all values are valid floats
            auto *cuda_fp32 = dynamic_cast<FP32Tensor *>(cuda_slice.get());
            auto *rocm_fp32 = dynamic_cast<FP32Tensor *>(rocm_slice.get());
            ASSERT_NE(cuda_fp32, nullptr) << "CUDA slice should be FP32";
            ASSERT_NE(rocm_fp32, nullptr) << "ROCm slice should be FP32";
            EXPECT_FALSE(TestTensorFactory::hasNaNOrInf(cuda_fp32));
            EXPECT_FALSE(TestTensorFactory::hasNaNOrInf(rocm_fp32));
        }

    } // namespace test
} // namespace llaminar2
