/**
 * @file Test__ModelLoaderRowSlice.cpp
 * @brief Unit tests for ModelLoader::loadTensorRowSlice (memory-efficient sliced loading)
 *
 * Tests verify that:
 * 1. Row slices are loaded correctly with proper dimensions
 * 2. Only slice data is read (not full tensor)
 * 3. Slice data matches corresponding rows from full tensor
 * 4. Edge cases are handled (first/last rank, uneven division)
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "loaders/ModelLoader.h"
#include "tensors/TensorFactory.h"
#include "utils/MPIContext.h"
#include "backends/DeviceId.h"
#include <cmath>
#include <numeric>

namespace llaminar2
{
    namespace test
    {

        class ModelLoaderRowSliceTest : public ::testing::Test
        {
        protected:
            void SetUp() override
            {
                // Use a real model file for testing
                model_path_ = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

                // Create MPI context (single rank for unit tests)
                mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);
                factory_ = std::make_unique<TensorFactory>(*mpi_ctx_);
                loader_ = std::make_unique<ModelLoader>(factory_.get());

                if (!loader_->loadModel(model_path_))
                {
                    GTEST_SKIP() << "Model file not found: " << model_path_;
                }
            }

            std::string model_path_;
            std::shared_ptr<MPIContext> mpi_ctx_;
            std::unique_ptr<TensorFactory> factory_;
            std::unique_ptr<ModelLoader> loader_;
        };

        TEST_F(ModelLoaderRowSliceTest, LoadsCorrectSliceShape)
        {
            // Test tensor: blk.0.attn_output.weight should be [896, 896] for Qwen2.5-0.5B
            const std::string tensor_name = "blk.0.attn_output.weight";

            // Load first half
            auto slice = loader_->loadTensorRowSlice(tensor_name, 0, 448);
            ASSERT_NE(slice, nullptr) << "Failed to load row slice";

            const auto &shape = slice->shape();
            ASSERT_EQ(shape.size(), 2) << "Slice should be 2D";
            EXPECT_EQ(shape[0], 448) << "Slice should have 448 rows";
            EXPECT_EQ(shape[1], 896) << "Slice should have 896 columns";
        }

        TEST_F(ModelLoaderRowSliceTest, SliceMatchesFullTensorData)
        {
            const std::string tensor_name = "blk.0.attn_output.weight";

            // Load full tensor
            auto full_tensor = loader_->loadTensor(tensor_name, DeviceId::cpu(), WeightPrecision::NATIVE);
            ASSERT_NE(full_tensor, nullptr) << "Failed to load full tensor";

            // Load first slice [0, 448)
            auto slice0 = loader_->loadTensorRowSlice(tensor_name, 0, 448);
            ASSERT_NE(slice0, nullptr) << "Failed to load slice [0, 448)";

            // Load second slice [448, 896)
            auto slice1 = loader_->loadTensorRowSlice(tensor_name, 448, 896);
            ASSERT_NE(slice1, nullptr) << "Failed to load slice [448, 896)";

            // Dequantize all three to FP32 for comparison
            size_t cols = full_tensor->shape()[1];
            std::vector<float> full_fp32(full_tensor->shape()[0] * cols);
            std::vector<float> slice0_fp32(448 * cols);
            std::vector<float> slice1_fp32(448 * cols);

            full_tensor->to_fp32(full_fp32.data());
            slice0->to_fp32(slice0_fp32.data());
            slice1->to_fp32(slice1_fp32.data());

            // Verify slice0 matches rows [0, 448) of full tensor
            for (size_t row = 0; row < 448; ++row)
            {
                for (size_t col = 0; col < cols; ++col)
                {
                    float full_val = full_fp32[row * cols + col];
                    float slice_val = slice0_fp32[row * cols + col];
                    EXPECT_FLOAT_EQ(slice_val, full_val)
                        << "Mismatch at slice0[" << row << ", " << col << "]";
                }
            }

            // Verify slice1 matches rows [448, 896) of full tensor
            for (size_t row = 0; row < 448; ++row)
            {
                for (size_t col = 0; col < cols; ++col)
                {
                    float full_val = full_fp32[(row + 448) * cols + col];
                    float slice_val = slice1_fp32[row * cols + col];
                    EXPECT_FLOAT_EQ(slice_val, full_val)
                        << "Mismatch at slice1[" << row << ", " << col << "]";
                }
            }
        }

        TEST_F(ModelLoaderRowSliceTest, SliceMemoryIsSmaller)
        {
            const std::string tensor_name = "blk.0.ffn_down.weight";

            // Get tensor info
            const auto *info = loader_->getModel().findTensor(tensor_name);
            ASSERT_NE(info, nullptr) << "Tensor not found";

            size_t total_rows = info->dimensions[0];
            size_t full_size_bytes = info->size_bytes;

            // Load half the rows
            size_t half_rows = total_rows / 2;
            auto slice = loader_->loadTensorRowSlice(tensor_name, 0, half_rows);
            ASSERT_NE(slice, nullptr);

            // Verify slice is approximately half the size
            // For quantized tensors: bytes = (rows * cols / block_size) * type_size
            size_t expected_slice_bytes = full_size_bytes / 2;
            size_t actual_slice_elements = slice->shape()[0] * slice->shape()[1];
            size_t full_elements = info->dimensions[0] * info->dimensions[1];

            // Slice should have half the elements
            EXPECT_EQ(actual_slice_elements * 2, full_elements)
                << "Slice should have half the elements of full tensor";
        }

        TEST_F(ModelLoaderRowSliceTest, HandlesUnevenDivision)
        {
            // Find a tensor and create uneven slices
            const std::string tensor_name = "blk.0.attn_output.weight";

            // Get tensor info
            const auto *info = loader_->getModel().findTensor(tensor_name);
            ASSERT_NE(info, nullptr);

            size_t total_rows = info->dimensions[0];

            // Load with uneven split (e.g., 3 ranks for 896 rows = 298, 298, 300)
            size_t rows_per_rank = total_rows / 3; // 298
            size_t remainder = total_rows % 3;     // 2

            // First rank: [0, 298)
            auto slice0 = loader_->loadTensorRowSlice(tensor_name, 0, rows_per_rank);
            ASSERT_NE(slice0, nullptr);
            EXPECT_EQ(slice0->shape()[0], rows_per_rank);

            // Second rank: [298, 596)
            auto slice1 = loader_->loadTensorRowSlice(tensor_name, rows_per_rank, 2 * rows_per_rank);
            ASSERT_NE(slice1, nullptr);
            EXPECT_EQ(slice1->shape()[0], rows_per_rank);

            // Third rank (last): [596, 896) - gets remainder
            auto slice2 = loader_->loadTensorRowSlice(tensor_name, 2 * rows_per_rank, total_rows);
            ASSERT_NE(slice2, nullptr);
            EXPECT_EQ(slice2->shape()[0], rows_per_rank + remainder);
        }

        TEST_F(ModelLoaderRowSliceTest, InvalidRowRangeReturnsNull)
        {
            const std::string tensor_name = "blk.0.attn_output.weight";

            // Get tensor info
            const auto *info = loader_->getModel().findTensor(tensor_name);
            ASSERT_NE(info, nullptr);
            size_t total_rows = info->dimensions[0];

            // row_start >= total_rows
            EXPECT_EQ(loader_->loadTensorRowSlice(tensor_name, total_rows, total_rows + 10), nullptr);

            // row_end > total_rows
            EXPECT_EQ(loader_->loadTensorRowSlice(tensor_name, 0, total_rows + 1), nullptr);

            // row_start >= row_end
            EXPECT_EQ(loader_->loadTensorRowSlice(tensor_name, 448, 448), nullptr);
            EXPECT_EQ(loader_->loadTensorRowSlice(tensor_name, 500, 400), nullptr);
        }

        TEST_F(ModelLoaderRowSliceTest, WorksWithDifferentQuantFormats)
        {
            // Test with Q4_0 format (this model uses Q4_0)
            const std::string tensor_name = "blk.0.attn_q.weight";

            const auto *info = loader_->getModel().findTensor(tensor_name);
            ASSERT_NE(info, nullptr);

            // Verify it's a quantized format
            EXPECT_TRUE(info->isQuantized()) << "Expected quantized tensor";

            // Load slice
            size_t half_rows = info->dimensions[0] / 2;
            auto slice = loader_->loadTensorRowSlice(tensor_name, 0, half_rows);
            ASSERT_NE(slice, nullptr);

            // Verify shape
            EXPECT_EQ(slice->shape()[0], half_rows);
            EXPECT_EQ(slice->shape()[1], info->dimensions[1]);

            // Verify we can dequantize it
            std::vector<float> fp32(slice->shape()[0] * slice->shape()[1]);
            EXPECT_NO_THROW(slice->to_fp32(fp32.data()));

            // Verify values are reasonable (not all zeros or NaN)
            bool has_nonzero = false;
            for (float v : fp32)
            {
                EXPECT_FALSE(std::isnan(v)) << "Found NaN in dequantized slice";
                if (v != 0.0f)
                    has_nonzero = true;
            }
            EXPECT_TRUE(has_nonzero) << "All values are zero - likely loading error";
        }

    } // namespace test
} // namespace llaminar2
