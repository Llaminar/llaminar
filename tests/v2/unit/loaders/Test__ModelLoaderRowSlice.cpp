/**
 * @file Test__ModelLoaderRowSlice.cpp
 * @brief Unit tests for memory-efficient row and explicit expert loading.
 *
 * Tests verify that:
 * 1. Row slices are loaded correctly with proper dimensions
 * 2. Only slice data is read (not full tensor)
 * 3. Slice data matches corresponding rows from full tensor
 * 4. Edge cases are handled (first/last rank, uneven division)
 * 5. Non-contiguous expert selections preserve native bytes for every format
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "loaders/ModelLoader.h"
#include "tensors/TensorFactory.h"
#include "utils/MPIContext.h"
#include "../../utils/TestModelHelper.h"
#include "backends/DeviceId.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
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

                if (!tryLoadModel(*loader_, model_path_))
                {
                    GTEST_SKIP() << "Model file not found: " << model_path_;
                }
            }

            std::string model_path_;
            std::shared_ptr<IMPIContext> mpi_ctx_;
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
            EXPECT_TRUE(slice->is_mmap_data())
                << "Native quantized row slices should retain zero-copy mmap backing";
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

        namespace
        {
            template <typename T>
            void writeScalar(std::ofstream &stream, const T &value)
            {
                stream.write(
                    reinterpret_cast<const char *>(&value),
                    static_cast<std::streamsize>(sizeof(T)));
            }

            void writeString(std::ofstream &stream, const std::string &value)
            {
                const uint64_t size = value.size();
                writeScalar(stream, size);
                stream.write(value.data(), static_cast<std::streamsize>(value.size()));
            }

            TensorType expectedTensorType(GGUFTensorType type)
            {
                switch (type)
                {
                case GGUFTensorType::F32: return TensorType::FP32;
                case GGUFTensorType::F16: return TensorType::FP16;
                case GGUFTensorType::BF16: return TensorType::BF16;
                case GGUFTensorType::Q4_0: return TensorType::Q4_0;
                case GGUFTensorType::Q4_1: return TensorType::Q4_1;
                case GGUFTensorType::Q5_0: return TensorType::Q5_0;
                case GGUFTensorType::Q5_1: return TensorType::Q5_1;
                case GGUFTensorType::Q8_0: return TensorType::Q8_0;
                case GGUFTensorType::Q2_K: return TensorType::Q2_K;
                case GGUFTensorType::Q3_K: return TensorType::Q3_K;
                case GGUFTensorType::Q4_K: return TensorType::Q4_K;
                case GGUFTensorType::Q5_K: return TensorType::Q5_K;
                case GGUFTensorType::Q6_K: return TensorType::Q6_K;
                case GGUFTensorType::Q8_K: return TensorType::Q8_K;
                case GGUFTensorType::IQ2_XXS: return TensorType::IQ2_XXS;
                case GGUFTensorType::IQ2_XS: return TensorType::IQ2_XS;
                case GGUFTensorType::IQ3_XXS: return TensorType::IQ3_XXS;
                case GGUFTensorType::IQ1_S: return TensorType::IQ1_S;
                case GGUFTensorType::IQ4_NL: return TensorType::IQ4_NL;
                case GGUFTensorType::IQ3_S: return TensorType::IQ3_S;
                case GGUFTensorType::IQ2_S: return TensorType::IQ2_S;
                case GGUFTensorType::IQ4_XS: return TensorType::IQ4_XS;
                case GGUFTensorType::IQ1_M: return TensorType::IQ1_M;
                }
                throw std::invalid_argument("Unsupported test tensor type");
            }
        } // namespace

        /**
         * @brief Synthetic GGUF fixture for exact expert-axis selection.
         *
         * The fixture writes one tiny 3D tensor without relying on a downloaded
         * model. Every logical expert owns a distinct native byte slab, making
         * an incorrect source offset, packed order, or tensor-type conversion
         * immediately observable with a byte comparison.
         */
        class ModelLoaderExpertSelectionTest
            : public ::testing::TestWithParam<GGUFTensorType>
        {
        protected:
            static constexpr const char *kTensorName =
                "blk.7.ffn_gate_exps.weight";
            static constexpr size_t kRowsPerExpert = 2u;
            static constexpr size_t kExpertCount = 5u;

            void SetUp() override
            {
                const GGUFTensorType type = GetParam();
                GGUFTensorInfo info;
                info.type = type;
                const size_t block_size = info.getBlockSize();
                columns_ = block_size == 0u ? 8u : block_size;
                bytes_per_expert_ =
                    kRowsPerExpert *
                    (block_size == 0u
                         ? columns_ * info.getTypeSize()
                         : (columns_ / block_size) * info.getTypeSize());

                source_bytes_.resize(kExpertCount * bytes_per_expert_);
                for (size_t expert = 0; expert < kExpertCount; ++expert)
                {
                    for (size_t byte = 0; byte < bytes_per_expert_; ++byte)
                    {
                        source_bytes_[expert * bytes_per_expert_ + byte] =
                            static_cast<uint8_t>((31u * (expert + 1u) + byte) % 251u);
                    }
                }

                const auto unique_suffix =
                    std::to_string(reinterpret_cast<uintptr_t>(this)) + "_" +
                    std::to_string(static_cast<uint32_t>(type));
                model_path_ = std::filesystem::temp_directory_path() /
                              ("llaminar_expert_selection_" + unique_suffix + ".gguf");
                writeSyntheticModel(type);

                mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);
                factory_ = std::make_unique<TensorFactory>(*mpi_ctx_);
                loader_ = std::make_unique<ModelLoader>(factory_.get());
                ASSERT_TRUE(loader_->loadModel(model_path_.string()));
            }

            void TearDown() override
            {
                loader_.reset();
                factory_.reset();
                mpi_ctx_.reset();
                std::error_code ignored;
                std::filesystem::remove(model_path_, ignored);
            }

            void writeSyntheticModel(GGUFTensorType type)
            {
                std::ofstream stream(
                    model_path_, std::ios::binary | std::ios::trunc);
                ASSERT_TRUE(stream.is_open());

                stream.write("GGUF", 4);
                writeScalar(stream, uint32_t{3});
                writeScalar(stream, uint64_t{1});
                writeScalar(stream, uint64_t{0});

                writeString(stream, kTensorName);
                writeScalar(stream, uint32_t{3});
                writeScalar(stream, static_cast<uint64_t>(columns_));
                writeScalar(stream, static_cast<uint64_t>(kRowsPerExpert));
                writeScalar(stream, static_cast<uint64_t>(kExpertCount));
                writeScalar(stream, static_cast<uint32_t>(type));
                writeScalar(stream, uint64_t{0});

                const auto directory_end =
                    static_cast<uint64_t>(stream.tellp());
                const uint64_t data_offset =
                    (directory_end + 31u) / 32u * 32u;
                const std::vector<char> padding(
                    static_cast<size_t>(data_offset - directory_end), 0);
                stream.write(
                    padding.data(),
                    static_cast<std::streamsize>(padding.size()));
                stream.write(
                    reinterpret_cast<const char *>(source_bytes_.data()),
                    static_cast<std::streamsize>(source_bytes_.size()));
                ASSERT_TRUE(stream.good());
            }

            std::filesystem::path model_path_;
            std::shared_ptr<IMPIContext> mpi_ctx_;
            std::unique_ptr<TensorFactory> factory_;
            std::unique_ptr<ModelLoader> loader_;
            std::vector<uint8_t> source_bytes_;
            size_t columns_ = 0u;
            size_t bytes_per_expert_ = 0u;
        };

        TEST_P(ModelLoaderExpertSelectionTest,
               PacksNonContiguousExpertsInRequestedSourceOrder)
        {
            const std::vector<size_t> selected_ids = {0u, 2u, 4u};
            auto selected = loader_->loadTensorExpertSelection(
                kTensorName,
                selected_ids,
                DeviceId::cpu(),
                WeightPrecision::NATIVE);
            ASSERT_NE(selected, nullptr);
            EXPECT_EQ(
                selected->shape(),
                (std::vector<size_t>{columns_, kRowsPerExpert, selected_ids.size()}));
            EXPECT_EQ(selected->native_type(), expectedTensorType(GetParam()));

            const auto *packed =
                static_cast<const uint8_t *>(selected->raw_data());
            ASSERT_NE(packed, nullptr);
            for (size_t packed_index = 0;
                 packed_index < selected_ids.size();
                 ++packed_index)
            {
                EXPECT_EQ(
                    std::memcmp(
                        packed + packed_index * bytes_per_expert_,
                        source_bytes_.data() +
                            selected_ids[packed_index] * bytes_per_expert_,
                        bytes_per_expert_),
                    0)
                    << "Native bytes differ for packed expert " << packed_index;
            }
        }

        TEST_P(ModelLoaderExpertSelectionTest,
               RejectsEmptyDuplicateUnsortedAndOutOfRangeIds)
        {
            EXPECT_THROW(
                (void)loader_->loadTensorExpertSelection(kTensorName, {}),
                std::invalid_argument);
            EXPECT_THROW(
                (void)loader_->loadTensorExpertSelection(kTensorName, {1u, 1u}),
                std::invalid_argument);
            EXPECT_THROW(
                (void)loader_->loadTensorExpertSelection(kTensorName, {2u, 1u}),
                std::invalid_argument);
            EXPECT_THROW(
                (void)loader_->loadTensorExpertSelection(
                    kTensorName, {0u, kExpertCount}),
                std::invalid_argument);
        }

        INSTANTIATE_TEST_SUITE_P(
            AllNativeFormats,
            ModelLoaderExpertSelectionTest,
            ::testing::Values(
                GGUFTensorType::F32,
                GGUFTensorType::F16,
                GGUFTensorType::BF16,
                GGUFTensorType::Q4_0,
                GGUFTensorType::Q4_1,
                GGUFTensorType::Q5_0,
                GGUFTensorType::Q5_1,
                GGUFTensorType::Q8_0,
                GGUFTensorType::Q2_K,
                GGUFTensorType::Q3_K,
                GGUFTensorType::Q4_K,
                GGUFTensorType::Q5_K,
                GGUFTensorType::Q6_K,
                GGUFTensorType::Q8_K,
                GGUFTensorType::IQ2_XXS,
                GGUFTensorType::IQ2_XS,
                GGUFTensorType::IQ3_XXS,
                GGUFTensorType::IQ1_S,
                GGUFTensorType::IQ4_NL,
                GGUFTensorType::IQ3_S,
                GGUFTensorType::IQ2_S,
                GGUFTensorType::IQ4_XS,
                GGUFTensorType::IQ1_M),
            [](const ::testing::TestParamInfo<GGUFTensorType> &info)
            {
                return "Type" +
                       std::to_string(static_cast<uint32_t>(info.param));
            });

    } // namespace test
} // namespace llaminar2
