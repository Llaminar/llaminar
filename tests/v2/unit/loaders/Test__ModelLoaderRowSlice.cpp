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
 * 6. Explicit original K/N windows copy only native blocks into compact output
 * 7. WeightManager retains complete static partition identity through native CPU preparation
 * 8. Complete frozen static lowering preserves ordinary source bindings and rejects identity collisions
 * 9. Real local-shard CPU kernels match the independent full-source CPU kernel
 *    with zero activation outside the slice and exactly declared invocation scratch
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "loaders/ModelLoader.h"
#include "../../utils/NativeWeightSourceTestRead.h"
#include "loaders/NativeWeightLoadStaging.h"
#include "tensors/NativeWeightSourceRange.h"
#include "execution/moe/MoEExpertOverlayPreparationPlan.h"
#include "execution/moe/ExpertPreparedMemoryGeometry.h"
#include "../../utils/CPUJointSourceTestPlan.h"
#include "loaders/WeightIdentity.h"
#include "loaders/WeightManager.h"
#include "loaders/PreparedWeightStore.h"
#include "tensors/TensorKernels.h"
#include "../../utils/QuantizedVerifierFormats.h"
#include "../../utils/CPUProjectionTestWorkspace.h"
#include "../../utils/CPUPreparedStoreTestAdmission.h"
#include "../../utils/CPUWeightPlanTestAdmission.h"
#include "../../utils/PreparedWeightTestHarness.h"
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
#include <limits>
#include <omp.h>
#include <sstream>
#include <cstdlib>

namespace llaminar2
{
    namespace test
    {

        class ModelLoaderRowSliceTest : public ::testing::Test
        {
        protected:
            /** @brief Open the optional legacy row-slice fixture without device work. */
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
        /** @brief Count actual native source calls while forwarding to the real GGUF loader. */
        class ProjectionObservedLoader final : public ModelLoader
        {
        public:
            using ModelLoader::ModelLoader;
            int projection_calls = 0;
            int whole_calls = 0;
            int selection_calls = 0;
            bool missing_projection = false;
            size_t native_calls=0;
            std::function<void(const NativeWeightSourceWindow&,std::span<const std::byte>)> observe_native;
            /** @brief Exercise the actual common reader and independently observe its exact source window. */
            void readNativeWeightSourceWindow(const std::string& name,const NativeWeightSourceWindow& window,
                std::span<std::byte> bytes)override {
                ++native_calls;
                if(missing_projection)throw std::runtime_error("Injected missing explicit native projection source");
                ModelLoader::readNativeWeightSourceWindow(name,window,bytes);
                if(observe_native)observe_native(window,bytes);
            }
            /** @brief Forward the original exact reader, or inject one explicit missing result. */
            std::shared_ptr<TensorBase> loadTensorExpertProjectionSlice(
                const std::string &name, const WeightSliceSpec &slice, DeviceId device,
                WeightPrecision precision) override
            {
                ++projection_calls;
                return missing_projection ? nullptr :
                    ModelLoader::loadTensorExpertProjectionSlice(name, slice, device, precision);
            }
            /** @brief Observe forbidden whole-parent substitution without altering legacy behavior. */
            std::shared_ptr<TensorBase> loadTensor(const std::string &name, DeviceId device,
                WeightPrecision precision) override
            {
                ++whole_calls;
                return ModelLoader::loadTensor(name, device, precision);
            }
            /** @brief Preserve and count the existing complete-expert operation. */
            std::shared_ptr<TensorBase> loadTensorExpertSelection(const std::string &name,
                const std::vector<size_t> &experts, DeviceId device, WeightPrecision precision) override
            {
                ++selection_calls;
                return ModelLoader::loadTensorExpertSelection(name, experts, device, precision);
            }
        };

        class ModelLoaderExpertSelectionTest
            : public ::testing::TestWithParam<GGUFTensorType>
        {
        protected:
            static constexpr const char *kTensorName =
                "blk.7.ffn_gate_exps.weight";
            static constexpr size_t kRowsPerExpert = 2u;
            static constexpr size_t kExpertCount = 5u;

            /** @brief Create an independently owned native-format synthetic model. */
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

            /** @brief Release mappings before removing the private synthetic fixture. */
            void TearDown() override
            {
                loader_.reset();
                factory_.reset();
                mpi_ctx_.reset();
                std::error_code ignored;
                std::filesystem::remove(model_path_, ignored);
            }

            /** @brief Serialize a minimal GGUF with exactly the test's native bytes. */
            void writeSyntheticModel(GGUFTensorType type)
            {
                std::ofstream stream(
                    model_path_, std::ios::binary | std::ios::trunc);
                ASSERT_TRUE(stream.is_open());

                stream.write("GGUF", 4);
                writeScalar(stream, uint32_t{3});
                writeScalar(stream, uint64_t{1});
                writeScalar(stream, uint64_t{0});

                writeString(stream, source_tensor_name_);
                writeScalar(stream, static_cast<uint32_t>(source_rank_));
                writeScalar(stream, static_cast<uint64_t>(columns_));
                writeScalar(stream, static_cast<uint64_t>(kRowsPerExpert));
                if(source_rank_==3)writeScalar(stream, static_cast<uint64_t>(kExpertCount));
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

            /** @brief Rebuild only this tiny source with several native blocks per K row. */
            void rebuildProjectionSource(bool mapped=true, std::string source_name=kTensorName, size_t native_units_per_row=20, size_t source_rank=3)
            {
                loader_.reset();
                source_tensor_name_=std::move(source_name);
                if(source_rank!=2 && source_rank!=3)throw std::invalid_argument("Synthetic original source requires rank2 or3");
                source_rank_=source_rank;
                GGUFTensorInfo info;info.type=GetParam();
                const size_t unit=info.getBlockSize() ? info.getBlockSize() : 8u;
                columns_=native_units_per_row*unit;
                bytes_per_expert_=kRowsPerExpert*(info.getBlockSize()
                    ? columns_/info.getBlockSize()*info.getTypeSize() : columns_*info.getTypeSize());
                source_bytes_.resize((source_rank_==3?kExpertCount:1)*bytes_per_expert_);
                for(size_t i=0;i<source_bytes_.size();++i)
                    source_bytes_[i]=static_cast<uint8_t>((i+31u*(i/bytes_per_expert_+1u))%251u);
                writeSyntheticModel(GetParam());
                loader_=std::make_unique<ModelLoader>(factory_.get());
                loader_->setUseMmap(mapped);
                ASSERT_TRUE(loader_->loadModel(model_path_.string()));
            }
            /** @return Complete original-axis request; expert IDs are physical original E slots. */
            WeightSliceSpec projection(size_t begin,size_t count,size_t firstN=0,size_t n=kRowsPerExpert) const
            {
                WeightSliceSpec slice;
                slice.source_rows=columns_;slice.source_cols=kRowsPerExpert;
                slice.row_start=begin;slice.row_count=count;slice.col_start=firstN;slice.col_count=n;
                slice.expert_count=3;slice.expert_ids={0,2,4};return slice;
            }
            /** @brief Compare every output row to its exact original encoded block bytes. */
            void expectProjectionBytes(const WeightSliceSpec &slice,const TensorBase &actual) const
            {
                GGUFTensorInfo info;info.type=GetParam();
                const auto block=info.getBlockSize(),typeBytes=info.getTypeSize();
                const auto rowBytes=block ? columns_/block*typeBytes : columns_*typeBytes;
                const auto begin=block ? slice.row_start/block*typeBytes : slice.row_start*typeBytes;
                const auto bytes=block ? slice.row_count/block*typeBytes : slice.row_count*typeBytes;
                ASSERT_EQ(actual.native_type(),expectedTensorType(GetParam()));
                ASSERT_EQ(actual.shape(),(std::vector<size_t>{slice.row_count,slice.col_count,slice.expert_count}));
                ASSERT_EQ(actual.size_bytes(),bytes*slice.col_count*slice.expert_count);
                const auto *data=static_cast<const uint8_t *>(actual.raw_data());ASSERT_NE(data,nullptr);
                for(size_t e=0;e<slice.expert_count;++e)
                for(size_t n=0;n<slice.col_count;++n)
                    EXPECT_EQ(std::memcmp(data+(e*slice.col_count+n)*bytes,
                        source_bytes_.data()+(slice.expert_ids.empty() ? slice.expert_start+e : size_t(slice.expert_ids[e]))*bytes_per_expert_+
                        (slice.col_start+n)*rowBytes+begin,bytes),0);
            }

            /** @brief Attach instrumentation around the real native loader, without fake source data. */
            ProjectionObservedLoader *observeProjectionSource(bool mapped)
            {
                loader_.reset();
                auto observed=std::make_unique<ProjectionObservedLoader>(factory_.get());
                observed->setUseMmap(mapped);
                if (!observed->loadModel(model_path_.string()))
                    throw std::runtime_error("Synthetic projection GGUF did not load");
                auto *result=observed.get();
                loader_=std::move(observed);
                return result;
            }

            /** @brief Write finite original encoded weights using the existing canonical format creators. */
            void rebuildFiniteProjectionSource(bool mapped=true,std::string name="blk.7.ffn_down_exps.weight")
            {
                rebuildProjectionSource(mapped,std::move(name));
                loader_.reset();
                const std::vector<size_t> shape{kRowsPerExpert*kExpertCount,columns_};
                std::unique_ptr<TensorBase> weights;
                for (const auto &format:quantizedVerifierFormats())
                    if (format.tensor_type==expectedTensorType(GetParam()))
                        weights=format.create(shape,73);
                if (GetParam()==GGUFTensorType::F32)
                    weights=TestTensorFactory::createFP32Random(shape,-.5f,.5f,73);
                if (GetParam()==GGUFTensorType::F16)
                    weights=TestTensorFactory::createFP16Random(shape,-.5f,.5f,73);
                if (GetParam()==GGUFTensorType::BF16)
                    weights=TestTensorFactory::createBF16Random(shape,-.5f,.5f,73);
                if (!weights || weights->size_bytes()!=source_bytes_.size())
                    throw std::runtime_error("Canonical native projection fixture has inconsistent source bytes");
                const auto *bytes=static_cast<const uint8_t *>(weights->raw_data());
                source_bytes_.assign(bytes,bytes+weights->size_bytes());
                writeSyntheticModel(GetParam());
            }

            /** @return A startup request carrying only one immutable geometry authority. */
            WeightRequirement projectionRequirement(const MoEExpertProjectionOwnership &ownership,
                WeightRole role) const
            {
                WeightRequirement request;
                request.canonical_name=source_tensor_name_;
                request.role=role;
                request.derivation=WeightDerivationKind::ExpertSlice;
                request.layer=7;
                request.lookup_device=DeviceId::cpu();
                request.target_device=DeviceId::cpu();
                request.bypass_tensor_parallel=true;
                request.tp_rank_or_device_index=ownership.participant();
                request.host_policy=WeightHostPolicy::RequiredForCPUExecution;
                request.projection_ownership=ownership;
                return request;
            }

            size_t source_rank_=3; ///< Original GGUF matrix or expert-parent axes; default fixtures unchanged.
            std::string source_tensor_name_=kTensorName;
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
               ContiguousSliceAndSelectionRetainExactBytesAfterLoaderRelease)
        {
            auto slice = loader_->loadTensorExpertSlice(kTensorName, 1u, 4u);
            auto selected = loader_->loadTensorExpertSelection(kTensorName, {1u, 2u, 3u});
            ASSERT_NE(slice, nullptr);
            ASSERT_NE(selected, nullptr);
            // Quantized slices retain their mmap lease; floating slices retain
            // their single owned allocation. Neither may borrow the loader.
            loader_.reset();
            for (const auto &tensor : {slice, selected})
            {
                EXPECT_EQ(tensor->native_type(), expectedTensorType(GetParam()));
                EXPECT_EQ(tensor->shape(),
                          (std::vector<size_t>{columns_, kRowsPerExpert, 3u}));
                ASSERT_EQ(tensor->size_bytes(), 3u * bytes_per_expert_);
                EXPECT_EQ(std::memcmp(tensor->raw_data(),
                                      source_bytes_.data() + bytes_per_expert_,
                                      tensor->size_bytes()), 0);
            }
        }

        TEST_P(ModelLoaderExpertSelectionTest,
               WholeNativePayloadPreservesBytesWithAndWithoutMappedFactory)
        {
            for (bool mapped : {false, true})
                for (bool use_factory : {false, true})
                {
                    SCOPED_TRACE(mapped);
                    SCOPED_TRACE(use_factory);
                    auto loader = std::make_unique<ModelLoader>(use_factory ? factory_.get() : nullptr);
                    loader->setUseMmap(mapped);
                    ASSERT_TRUE(loader->loadModel(model_path_.string()));
                    auto tensor = loader->loadTensor(kTensorName);
                    ASSERT_NE(tensor, nullptr);
                    loader.reset();
                    EXPECT_EQ(tensor->native_type(), expectedTensorType(GetParam()));
                    EXPECT_EQ(tensor->shape(),
                        (std::vector<size_t>{columns_, kRowsPerExpert, kExpertCount}));
                    ASSERT_EQ(tensor->size_bytes(), source_bytes_.size());
                    EXPECT_EQ(std::memcmp(tensor->raw_data(), source_bytes_.data(), source_bytes_.size()), 0);
                }
        }

        TEST_P(ModelLoaderExpertSelectionTest,
               ContiguousSliceRejectsEmptyReversedAndOutOfRangeIntervals)
        {
            EXPECT_EQ(loader_->loadTensorExpertSlice(kTensorName, 1u, 1u), nullptr);
            EXPECT_EQ(loader_->loadTensorExpertSlice(kTensorName, 3u, 1u), nullptr);
            EXPECT_EQ(loader_->loadTensorExpertSlice(kTensorName, 0u, kExpertCount + 1u), nullptr);
        }

        TEST_P(ModelLoaderExpertSelectionTest,
               LargeContiguousFirstTouchPreservesBytesWithIndependentWorkers)
        {
            if (GetParam() != GGUFTensorType::F32)
                return; // The byte-copy scheduler is dtype-independent.
            loader_.reset();
            columns_ = 393217u; // Odd tails cross each 1 MiB copy boundary.
            bytes_per_expert_ = kRowsPerExpert * columns_ * sizeof(float);
            source_bytes_.resize(kExpertCount * bytes_per_expert_);
            for (size_t byte = 0; byte < source_bytes_.size(); ++byte)
                source_bytes_[byte] = static_cast<uint8_t>((byte * 29u + byte / 257u) & 255u);
            writeSyntheticModel(GetParam());
            loader_ = std::make_unique<ModelLoader>(factory_.get());
            ASSERT_TRUE(loader_->loadModel(model_path_.string()));
            const int previous = omp_get_max_threads();
            for (const int workers : {1, 2, 4})
            {
                omp_set_num_threads(workers);
                auto selected = loader_->loadTensorExpertSlice(kTensorName, 1u, 4u);
                ASSERT_NE(selected, nullptr);
                EXPECT_EQ(std::memcmp(selected->raw_data(),
                    source_bytes_.data() + bytes_per_expert_, 3u * bytes_per_expert_), 0);
                auto whole = loader_->loadTensor(kTensorName);
                ASSERT_NE(whole, nullptr);
                EXPECT_EQ(std::memcmp(whole->raw_data(), source_bytes_.data(), source_bytes_.size()), 0);
            }
            // Each existing worker owns a separate load; using collective
            // worksharing here would silently omit chunks or deadlock.
            std::array<std::shared_ptr<TensorBase>, 2> concurrent;
            std::array<std::shared_ptr<TensorBase>, 2> concurrent_whole;
#pragma omp parallel num_threads(2)
            {
                concurrent[omp_get_thread_num()] =
                    loader_->loadTensorExpertSlice(kTensorName, 1u, 4u);
                concurrent_whole[omp_get_thread_num()] = loader_->loadTensor(kTensorName);
            }
            omp_set_num_threads(previous);
            for (const auto &selected : concurrent)
            {
                ASSERT_NE(selected, nullptr);
                EXPECT_EQ(std::memcmp(selected->raw_data(),
                    source_bytes_.data() + bytes_per_expert_, 3u * bytes_per_expert_), 0);
            }
            for (const auto &whole : concurrent_whole)
            {
                ASSERT_NE(whole, nullptr);
                EXPECT_EQ(std::memcmp(whole->raw_data(), source_bytes_.data(), source_bytes_.size()), 0);
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

        /** @test Unequal general intervals preserve all native block metadata, including QFN640 cuts128/128/128/256. */
        TEST_P(ModelLoaderExpertSelectionTest, NativeProjectionUnequalKIntervalsPreserveOriginalBytes)
        {
            for(bool mapped:{false,true})
            {
                rebuildProjectionSource(mapped);
                GGUFTensorInfo info;info.type=GetParam();
                const size_t unit=info.getBlockSize() ? info.getBlockSize() : 8u;
                const std::array<size_t,5> cuts{0,4u*unit,8u*unit,12u*unit,20u*unit};
                for(size_t i=0;i+1<cuts.size();++i)
                {
                    auto slice=projection(cuts[i],cuts[i+1]-cuts[i]);
                    auto actual=loader_->loadTensorExpertProjectionSlice(kTensorName,slice);
                    ASSERT_NE(actual,nullptr);expectProjectionBytes(slice,*actual);
                    slice.expert_ids.clear();slice.expert_start=1;
                    actual=loader_->loadTensorExpertProjectionSlice(kTensorName,slice);
                    ASSERT_NE(actual,nullptr);expectProjectionBytes(slice,*actual);
                }
            }
        }

        /** @test N slicing and combined N/K slicing retain original sorted expert IDs without reading adjacent rows into output. */
        TEST_P(ModelLoaderExpertSelectionTest, NativeProjectionRowsAndRectanglesPreserveOriginalBytes)
        {
            for(bool mapped:{false,true})
            {
                rebuildProjectionSource(mapped);
                for(bool fullK:{false,true})
                {
                    const size_t begin=fullK ? 0 : columns_/5;
                    const size_t count=fullK ? columns_ : columns_/5;
                    auto slice=projection(begin,count,1,1);
                    auto actual=loader_->loadTensorExpertProjectionSlice(kTensorName,slice);
                    ASSERT_NE(actual,nullptr);expectProjectionBytes(slice,*actual);
                }
                auto rectangle=projection(columns_/5,columns_/5,1,1);
                auto compact=loader_->loadTensorExpertProjectionSlice(kTensorName,rectangle);
                ASSERT_NE(compact,nullptr);
                auto full=projection(0,columns_);
                auto actual=loader_->loadTensorExpertProjectionSlice(kTensorName,full);
                auto previous=loader_->loadTensorExpertSelection(kTensorName,{0,2,4});
                ASSERT_NE(actual,nullptr);ASSERT_NE(previous,nullptr);
                ASSERT_EQ(actual->size_bytes(),previous->size_bytes());
                EXPECT_EQ(std::memcmp(actual->raw_data(),previous->raw_data(),actual->size_bytes()),0);
                loader_.reset();
                expectProjectionBytes(rectangle,*compact);
                expectProjectionBytes(full,*actual);
                expectProjectionBytes(full,*previous);
            }
        }

        /** @test Malformed source dimensions, bounds, identities and conversions reject rather than returning a different native slice. */
        TEST_P(ModelLoaderExpertSelectionTest, NativeProjectionRejectsInvalidOriginalGeometry)
        {
            rebuildProjectionSource();
            for(int mutation=0;mutation<18;++mutation)
            {
                auto slice=projection(columns_/5,columns_/5,1,1);
                switch(mutation)
                {
                case 0: ++slice.source_rows;break;
                case 1: ++slice.source_cols;break;
                case 2: slice.row_count=0;break;
                case 3: slice.col_count=0;break;
                case 4: slice.row_start=columns_;break;
                case 5: slice.col_start=kRowsPerExpert;break;
                case 6: slice.expert_count=0;break;
                case 7: slice.expert_ids={0,2,2};break;
                case 8: slice.expert_ids={0,4,2};break;
                case 9: slice.expert_ids={0,2,5};break;
                case 10: slice.expert_ids={-1,2,4};break;
                case 11: slice.inner_is_presliced=true;break;
                case 12: slice.expert_ids.clear();slice.expert_start=kExpertCount;break;
                case 13: slice.expert_count=2;break;
                case 14: slice.row_start=std::numeric_limits<size_t>::max();break;
                case 15: slice.row_count=std::numeric_limits<size_t>::max();break;
                case 16: slice.col_start=std::numeric_limits<size_t>::max();break;
                case 17: slice.col_count=std::numeric_limits<size_t>::max();break;
                }
                EXPECT_THROW((void)loader_->loadTensorExpertProjectionSlice(kTensorName,slice),std::invalid_argument)<<mutation;
            }
            const auto slice=projection(0,columns_);
            EXPECT_THROW((void)loader_->loadTensorExpertProjectionSlice(kTensorName,slice,
                DeviceId::cpu(),WeightPrecision::CONVERT_TO_FP32),std::invalid_argument);
            GGUFTensorInfo info;info.type=GetParam();
            if(info.getBlockSize()>1)
            {
                auto invalid=slice;invalid.row_start=1;invalid.row_count=columns_-1;
                EXPECT_THROW((void)loader_->loadTensorExpertProjectionSlice(kTensorName,invalid),std::invalid_argument);
                invalid=slice;--invalid.row_count;
                EXPECT_THROW((void)loader_->loadTensorExpertProjectionSlice(kTensorName,invalid),std::invalid_argument);
            }
        }

        /** @test The real materializer reads exact native rectangles, never a full parent or old selection. */
        TEST_P(ModelLoaderExpertSelectionTest, NativeProjectionWeightManagerFreezesExactOriginalSource)
        {
            using Ownership=MoEExpertProjectionOwnership;
            for (const bool mapped:{false,true})
            for (const auto role:{WeightRole::MoEExpertGate,WeightRole::MoEExpertDown})
            {
                const bool down=role==WeightRole::MoEExpertDown;
                rebuildFiniteProjectionSource(mapped,down ? "blk.7.ffn_down_exps.weight" : kTensorName);
                auto *observed=observeProjectionSource(mapped);
                const int quantum=static_cast<int>(columns_/20);
                for (const int participant:{1,3})
                {
                    const auto ownership=down
                        ? Ownership::withinExpertIntermediateSlices({5,2,static_cast<int>(columns_)},participant,
                            {{0,4*quantum},{4*quantum,4*quantum},{8*quantum,4*quantum},{12*quantum,8*quantum}})
                        : Ownership::withinExpertIntermediateSlices({5,static_cast<int>(columns_),2},0,{{0,1},{1,1}});
                    WeightPlan plan(InferenceStrategy{.model_id=ModelContextId{88}},PhysicalMemoryOwner::RoutedExpertWeights);
                    plan.add(projectionRequirement(ownership,role));
                    const auto &request=plan.requirements().front();
                    EXPECT_FALSE(request.slice.inner_is_presliced);
                    EXPECT_EQ(request.slice.expert_count,kExpertCount);
                    WeightManager manager(*loader_,mpi_ctx_);admitCPUWeightPlansForTest(manager,*loader_,{&plan});
                    const auto expected=projectNativeWeightSource(request,loader_->nativeWeightSourceGeometry(source_tensor_name_));
                    const auto source_geometry=loader_->nativeWeightSourceGeometry(source_tensor_name_);
                    const auto source_row_bytes=columns_/source_geometry.native_group_elements*source_geometry.native_group_bytes;
                    const auto projection=ownership.projection(role);
                    observed->observe_native=[&](const auto& window,auto bytes) {
                        EXPECT_LT(window.original_expert,kExpertCount);
                        EXPECT_GE(window.first_row,size_t(projection.first_row));
                        EXPECT_LE(window.first_row+window.rows,size_t(projection.first_row+projection.rows));
                        EXPECT_GE(window.first_column,size_t(projection.first_column));
                        EXPECT_LE(window.first_column+window.columns,size_t(projection.first_column+projection.columns));
                        for(size_t row=0;row<window.rows;++row) {
                            const auto offset=window.original_expert*bytes_per_expert_+(window.first_row+row)*source_row_bytes+
                                window.first_column/source_geometry.native_group_elements*source_geometry.native_group_bytes;
                            EXPECT_EQ(std::memcmp(bytes.data()+row*window.native_row_bytes,source_bytes_.data()+offset,window.native_row_bytes),0);
                        }
                    };
                    const auto frozen=manager.materialize(plan);ASSERT_EQ(frozen.bindings().size(),kExpertCount);
                    const auto& binding=frozen.bindings().front();
                    EXPECT_EQ(frozen.physicalMemoryOwner(),PhysicalMemoryOwner::RoutedExpertWeights);
                    const auto& matrices=frozen;
                    // Committed per-expert consumers preserve ordinary bindings;
                    // they cannot be reinterpreted as a new compact3D raw parent.
                    std::shared_ptr<TensorBase> ordinary_tensor=TestTensorFactory::createFP32({2});WeightBinding ordinary;
                    ordinary.binding_id=700;ordinary.identity=makeSourceWeightIdentity("blk.7.ffn_norm.weight",plan.strategy().model_id,700);
                    ordinary.tensor_owner=ordinary_tensor;ordinary.tensor=ordinary_tensor.get();ordinary.immutable=true;
                    for(const bool ordinary_first:{false,true}) {
                        auto complete=frozen.bindings();complete.insert(ordinary_first?complete.begin():complete.end(),ordinary);
                        const FrozenModelWeightSet combined(plan.strategy(),complete,frozen.physicalMemoryOwner());combined.validateForGraph();
                        const auto& retained=combined.layer(7,"ffn_norm.weight");
                        EXPECT_EQ(retained.identity,ordinary.identity);EXPECT_EQ(retained.tensor,ordinary.tensor);
                        EXPECT_EQ(retained.tensor_owner,ordinary.tensor_owner);EXPECT_EQ(retained.prepared.has_value(),ordinary.prepared.has_value());
                        EXPECT_FALSE(retained.projection_ownership);EXPECT_EQ(combined.bindings().size(),kExpertCount+1);
                        EXPECT_THROW((void)combined.projectedExpertMatrices(),std::invalid_argument);
                        auto collision=ordinary;collision.binding_id=binding.binding_id;collision.identity.instance_id=binding.binding_id;
                        complete.insert(complete.begin(),collision);
                        const FrozenModelWeightSet conflicting(plan.strategy(),complete,frozen.physicalMemoryOwner());
                        EXPECT_THROW((void)conflicting.projectedExpertMatrices(),std::invalid_argument);
                    }
                    for (size_t expert=0;expert<kExpertCount;++expert)
                    {
                        const auto &matrix=matrices.bindings()[expert];
                        EXPECT_EQ(matrix.tensor->shape(),(std::vector<size_t>{
                            static_cast<size_t>(projection.rows),static_cast<size_t>(projection.columns)}));
                        EXPECT_EQ(matrix.tensor->native_type(),expectedTensorType(GetParam()));
                        EXPECT_EQ(matrix.identity.expert,static_cast<int>(expert));
                        EXPECT_EQ(matrix.slice.source_rows,static_cast<size_t>(projection.source_rows));
                        EXPECT_EQ(matrix.slice.source_cols,static_cast<size_t>(projection.source_columns));
                        EXPECT_EQ(matrix.slice.row_start,static_cast<size_t>(projection.first_row));
                        EXPECT_EQ(matrix.slice.col_start,static_cast<size_t>(projection.first_column));
                        EXPECT_EQ(matrix.slice.expert_ids,(std::vector<int>{static_cast<int>(expert)}));
                        EXPECT_EQ(matrix.projection_ownership,binding.projection_ownership);
                        EXPECT_FALSE(matrix.projection_source_owner);EXPECT_EQ(matrix.native_source,expected[expert]);
                        ASSERT_TRUE(matrix.prepared);EXPECT_TRUE(matrix.tensor->hasPreparedDeviceState());
                    }
                    // Composing an already normalized requirement keeps the same binding identity.
                    WeightPlan copy(plan.strategy(),plan.physicalMemoryOwner());
                    copy.add(request);
                    const auto calls=observed->native_calls;
                    EXPECT_EQ(manager.materialize(copy).bindings().front().binding_id,binding.binding_id);
                    EXPECT_EQ(observed->native_calls,calls);observed->observe_native={};
                }
                EXPECT_EQ(observed->projection_calls,0);EXPECT_GT(observed->native_calls,0u);
                EXPECT_EQ(observed->selection_calls,0);
                EXPECT_EQ(observed->whole_calls,0);
            }
        }

        /** @test Unsupported/missing explicit sources reject before a generic full-weight substitution. */
        TEST_P(ModelLoaderExpertSelectionTest, NativeProjectionWeightManagerRejectsForeignOrMissingIntent)
        {
            using Ownership=MoEExpertProjectionOwnership;
            rebuildFiniteProjectionSource();
            auto *observed=observeProjectionSource(true);
            const int quantum=static_cast<int>(columns_/20);
            const auto ownership=Ownership::withinExpertIntermediateSlices({5,2,static_cast<int>(columns_)},1,
                {{0,4*quantum},{4*quantum,4*quantum},{8*quantum,4*quantum},{12*quantum,8*quantum}});
            for (int mutation=0;mutation<10;++mutation)
            {
                auto request=projectionRequirement(ownership,WeightRole::MoEExpertDown);
                switch (mutation)
                {
                case 0: request.slice.row_start=1;break;
                case 1: request.role=WeightRole::MoEExpertGate;break;
                case 2: request.source_name=kTensorName;break;
                case 3: request.layer=6;break;
                case 4: request.bypass_tensor_parallel=false;break;
                case 5: request.expected_prepared_kind=PreparedWeightKind::CpuPackedGemm;break;
                case 6: request.overlay_participant_index=0;break;
                case 7: request.projection_ownership=Ownership::completeExperts(ownership.geometry());break;
                case 8: request.tp_rank_or_device_index=0;break;
                case 9: request.canonical_name="blk.6.ffn_down_exps.weight";break;
                }
                WeightPlan plan;
                EXPECT_THROW(plan.add(request),std::invalid_argument)<<mutation;
            }
            EXPECT_EQ(observed->projection_calls,0);
            const FrozenModelWeightSet empty_source(InferenceStrategy{},{});
            EXPECT_THROW((void)empty_source.projectedExpertMatrices(),std::invalid_argument);
            WeightPlan plan(InferenceStrategy{.model_id=ModelContextId{58}});
            plan.add(projectionRequirement(ownership,WeightRole::MoEExpertDown));
            GGUFTensorInfo native_info;native_info.type=GetParam();
            if(native_info.getBlockSize())
                EXPECT_THROW((void)quoteCPUWeightPreparation(expectedTensorType(GetParam()),2,4*quantum,
                    WeightPrecision::CONVERT_TO_INT8,CPUWeightConsumer::PreparedGemm),std::invalid_argument);
            WeightManager manager(*loader_,mpi_ctx_);admitCPUWeightPlansForTest(manager,*loader_,{&plan});
            observed->missing_projection=true;
            EXPECT_THROW((void)manager.materialize(plan),std::runtime_error);
            observed->missing_projection=false;
            EXPECT_EQ(observed->native_calls,1u);EXPECT_EQ(observed->projection_calls,0);
            EXPECT_EQ(observed->whole_calls,0);EXPECT_EQ(observed->selection_calls,0);
            EXPECT_EQ(manager.physicalMemoryAuthority()->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::WeightLoadStaging,
                PhysicalMemoryMaterializationKind::NewAllocation),0u);
            // The old complete-expert request is now another original range
            // consumed by the SAME common reader, never the obsolete selection loader.
            auto old=projectionRequirement(ownership,WeightRole::MoEExpertDown);old.projection_ownership.reset();
            old.slice=projection(0,columns_);old.slice.inner_is_presliced=true;
            WeightPlan old_plan(InferenceStrategy{.model_id=ModelContextId{59}});old_plan.add(old);
            WeightManager complete(*loader_,mpi_ctx_);admitCPUWeightPlansForTest(complete,*loader_,{&old_plan});
            const auto full=complete.materialize(old_plan);ASSERT_EQ(full.bindings().size(),3u);
            for(const auto& binding:full.bindings()) {
                EXPECT_EQ(binding.tensor->shape(),(std::vector<size_t>{kRowsPerExpert,columns_}));
                ASSERT_TRUE(binding.native_source);EXPECT_EQ(binding.native_source->range.columns,columns_);
            }
            EXPECT_GT(observed->native_calls,1u);EXPECT_EQ(observed->selection_calls,0);
            EXPECT_EQ(observed->projection_calls,0);EXPECT_EQ(observed->whole_calls,0);

        }

        /** @test Full partition participates in stable binding identity, even with identical local bytes. */
        TEST_P(ModelLoaderExpertSelectionTest, NativeProjectionWeightManagerRetainsGlobalIdentityAndSourceLifetime)
        {
            using Ownership=MoEExpertProjectionOwnership;
            rebuildFiniteProjectionSource();
            observeProjectionSource(true);
            const int unit=static_cast<int>(columns_/20);
            const auto first=Ownership::withinExpertIntermediateSlices({5,2,static_cast<int>(columns_)},0,
                {{0,4*unit},{4*unit,4*unit},{8*unit,4*unit},{12*unit,8*unit}});
            const auto different_peers=Ownership::withinExpertIntermediateSlices(first.geometry(),0,
                {{0,4*unit},{4*unit,2*unit},{6*unit,6*unit},{12*unit,8*unit}});
            std::vector<WeightBinding> retained;std::vector<std::shared_ptr<ITensorGemm>> engines;
            std::vector<std::array<float,2>> before;
            const auto execute=[&](ITensorGemm& engine,std::array<float,2>& result) {
                auto input=TestTensorFactory::createFP32Random({size_t(engine.get_k()),1},-.5f,.5f,71);
                auto output=TestTensorFactory::createFP32Zeros({size_t(engine.get_n()),1});
                const auto requirements=cpuProjectionTestRequirements(1,{&engine});
                std::unique_ptr<CPUProjectionTestWorkspace> workspace;
                if(requirements.total_bytes_with_alignment())workspace=std::make_unique<CPUProjectionTestWorkspace>(requirements);
                ASSERT_TRUE(engine.multiply_tensor(input.get(),output.get(),1,engine.get_n(),engine.get_k(),true,
                    1.f,0.f,nullptr,mpi_ctx_.get(),-1,workspace?workspace->get():nullptr));
                std::copy_n(static_cast<const float*>(output->raw_data()),2,result.begin());
            };
            WeightPlan a(InferenceStrategy{.model_id=ModelContextId{89}}),b(a.strategy());
            a.add(projectionRequirement(first,WeightRole::MoEExpertDown));
            b.add(projectionRequirement(different_peers,WeightRole::MoEExpertDown));
            uint64_t first_id=0;
            for(const auto* plan:{&a,&b}) {
                // These are independent component owners, not simultaneous
                // replicas sharing one application's physical allocation credit.
                WeightManager manager(*loader_,mpi_ctx_);admitCPUWeightPlansForTest(manager,*loader_,{plan});
                const auto frozen=manager.materialize(*plan);ASSERT_EQ(frozen.bindings().size(),kExpertCount);
                if(plan==&a) {
                    first_id=frozen.bindings().front().binding_id;retained=frozen.bindings();
                    for(const auto& binding:retained) {
                        engines.push_back(manager.preparedWeightStore()->gemmKernelLifetime(*binding.prepared));
                        ASSERT_TRUE(engines.back());before.emplace_back();execute(*engines.back(),before.back());
                    }
                } else EXPECT_NE(frozen.bindings().front().binding_id,first_id);
            }
            loader_.reset(); // Only actual final tensors/engines remain; no source parent is retained.
            ASSERT_EQ(retained.size(),kExpertCount);ASSERT_EQ(engines.size(),kExpertCount);
            for(size_t expert=0;expert<retained.size();++expert) {
                const auto& binding=retained[expert];EXPECT_FALSE(binding.projection_source_owner);
                ASSERT_TRUE(binding.native_source);EXPECT_EQ(binding.native_source->range.original_expert,expert);
                EXPECT_EQ(binding.projection_ownership,std::optional(first));ASSERT_TRUE(engines[expert]);
                EXPECT_TRUE(engines[expert]->weights_converted());EXPECT_EQ(engines[expert]->get_k(),4*unit);
                EXPECT_EQ(engines[expert]->get_n(),2);
                std::array<float,2> after{};execute(*engines[expert],after);
                EXPECT_EQ(std::memcmp(before[expert].data(),after.data(),sizeof(after)),0);
            }

        }

        /** @test Actual CPU preparation consumes the native local K/N and refuses foreign full-plan adoption. */
        TEST_P(ModelLoaderExpertSelectionTest, NativeProjectionWeightManagerPreparesActualLocalCPUKernels)
        {
            using Ownership=MoEExpertProjectionOwnership;
            rebuildFiniteProjectionSource();
            auto *observed=observeProjectionSource(true);
            WeightManager manager(*loader_,mpi_ctx_);
            const int unit=static_cast<int>(columns_/20);
            const auto ownership=Ownership::withinExpertIntermediateSlices({5,2,static_cast<int>(columns_)},0,
                {{0,4*unit},{4*unit,4*unit},{8*unit,4*unit},{12*unit,8*unit}});
            WeightPlan plan(InferenceStrategy{.model_id=ModelContextId{90}},PhysicalMemoryOwner::RoutedExpertWeights);
            plan.add(projectionRequirement(ownership,WeightRole::MoEExpertDown));
            admitCPUWeightPlansForTest(manager,*loader_,{&plan});
            const auto matrices=manager.materialize(plan);ASSERT_EQ(matrices.bindings().size(),kExpertCount);
            ASSERT_TRUE(manager.prepareWeightsForDevice(matrices,DeviceId::cpu()));
            EXPECT_THROW((void)matrices.projectedExpertMatrices(),std::invalid_argument);
            auto& store=*manager.preparedWeightStore();
            // A separate unsliced reader/preparation supplies the arithmetic
            // oracle. The observed production reader below must still read
            // only its exact compact rectangle. Native prepared codebooks can
            // round effective scales differently from TensorBase::data(), so
            // direct dequantization is not the CPU kernel's arithmetic oracle.
            ModelLoader reference_loader(factory_.get());
            ASSERT_TRUE(reference_loader.loadModel(model_path_.string()));
            const auto reference_parent=reference_loader.loadTensor(source_tensor_name_);
            ASSERT_NE(reference_parent,nullptr);
            for (const auto &binding:matrices.bindings())
            {
                ASSERT_TRUE(binding.prepared);const auto prepared=*binding.prepared;
                auto* kernel=store.gemmKernel(prepared);
                ASSERT_NE(kernel,nullptr);
                EXPECT_EQ(kernel->get_n(),2);
                EXPECT_EQ(kernel->get_k(),4*unit);
                for (const auto &format:quantizedVerifierFormats())
                    if (format.tensor_type==binding.tensor->native_type())
                    {
                        NativeVnniSourceIdentity original;
                        ASSERT_TRUE(kernel->exportNativeVNNISourceIdentity(original));
                        EXPECT_TRUE(original.present);
                        EXPECT_EQ(original.codebook_id,format.source_codebook_id);
                        EXPECT_EQ(original.is_superblock,format.source_is_superblock);
                    }
                const auto full_matrix=reference_parent->create_view(
                    {kRowsPerExpert,columns_},size_t(binding.identity.expert)*kRowsPerExpert*columns_);
                auto oracle=makePreparedGemmFixture(full_matrix.get(),DeviceId::cpu(),source_tensor_name_);
                auto *oracle_kernel=oracle.store->gemmKernel(oracle.ref);
                ASSERT_NE(oracle_kernel,nullptr);
                const auto requirements=cpuProjectionTestRequirements(3,{kernel,oracle_kernel});
                // Floating kernels declare no external scratch. An empty
                // declaration needs neither a physical resource nor an arena.
                std::unique_ptr<CPUProjectionTestWorkspace> workspace;
                if (requirements.total_bytes_with_alignment()!=0)
                    workspace=std::make_unique<CPUProjectionTestWorkspace>(requirements);
                auto *invocation=workspace ? workspace->get() : nullptr;
                const auto projection=ownership.projection(WeightRole::MoEExpertDown);
                for (const int rows : {1,3})
                {
                    auto input=TestTensorFactory::createFP32({size_t(rows),size_t(4*unit)});
                    auto output=TestTensorFactory::createFP32({size_t(rows),size_t(2)});
                    auto full_input=TestTensorFactory::createFP32({size_t(rows),columns_});
                    auto reference=TestTensorFactory::createFP32({size_t(rows),size_t(2)});
                    float *stimulus=input->mutable_data();
                    std::fill_n(stimulus,size_t(rows)*4*unit,0.f);
                    float *full_stimulus=full_input->mutable_data();
                    std::fill_n(full_stimulus,size_t(rows)*columns_,0.f);
                    for (int row=0;row<rows;++row)
                    {
                        const int column=row==0 ? 0 : row==1 ? 2*unit : 4*unit-1;
                        ASSERT_GE(column,0);
                        ASSERT_LT(column,projection.columns);
                        // The amplitude represents exactly as Q8 scale1/code127
                        // in both shapes, including the first/middle/last block.
                        stimulus[size_t(row)*4*unit+column]=127.f;
                        full_stimulus[size_t(row)*columns_+projection.first_column+column]=127.f;
                    }
                    ASSERT_TRUE(kernel->multiply_tensor(input.get(),output.get(),rows,2,4*unit,
                        true,1.f,0.f,nullptr,mpi_ctx_.get(),-1,invocation));
                    ASSERT_TRUE(oracle_kernel->multiply_tensor(full_input.get(),reference.get(),rows,2,columns_,
                        true,1.f,0.f,nullptr,mpi_ctx_.get(),-1,invocation));
                    for (int row=0;row<rows;++row)
                        for (int n=0;n<2;++n)
                            EXPECT_FLOAT_EQ(output->data()[size_t(row)*2+n],
                                reference->data()[size_t(row)*2+n]);
                }
                EXPECT_FALSE(binding.projection_source_owner);ASSERT_TRUE(binding.native_source);
                EXPECT_EQ(binding.native_source->range.rows,2u);
                auto foreign=binding;
                foreign.binding_id^=0x91838261;
                foreign.prepared->binding_id=foreign.binding_id;
                foreign.projection_ownership=Ownership::withinExpertIntermediateSlices(ownership.geometry(),0,
                    {{0,4*unit},{4*unit,2*unit},{6*unit,6*unit},{12*unit,8*unit}});
                EXPECT_FALSE(store.adoptPreparedGemmForBinding(foreign,DeviceId::cpu()));
                auto same=binding;
                same.binding_id^=0x4173921;
                same.prepared->binding_id=same.binding_id;
                EXPECT_TRUE(store.adoptPreparedGemmForBinding(same,DeviceId::cpu()));
            }
            EXPECT_EQ(observed->projection_calls,0);EXPECT_GT(observed->native_calls,0u);
            EXPECT_EQ(observed->selection_calls,0);EXPECT_EQ(observed->whole_calls,0);
            const auto reads=observed->native_calls;
            EXPECT_TRUE(manager.prepareWeightsForDevice(matrices,DeviceId::cpu()));EXPECT_EQ(observed->native_calls,reads);
            // An unrelated device with no selected bindings is a legitimate
            // empty participant, so it is not a negative preparation request.
            // Deliberately replacing the ORIGINAL target on these CPU records
            // must instead reject at the cold source/participant validator.
            for(const auto device:{DeviceId::cuda(0),DeviceId::rocm(0)}) {
                auto replaced=matrices.bindings();
                for(auto& binding:replaced) {
                    binding.residency.home_device=device;binding.residency.resident_device=device;
                }
                const FrozenModelWeightSet foreign(matrices.strategy(),replaced,matrices.physicalMemoryOwner());
                EXPECT_THROW((void)manager.prepareWeightsForDevice(foreign,device),std::invalid_argument);
            }
            EXPECT_EQ(observed->native_calls,reads);

        }

        /** @test K128 is illegal for original256-element source blocks, even when a typed F interval is positive. */
        TEST_P(ModelLoaderExpertSelectionTest, StaticProjectionNativeSuperblockRejects128K)
        {
            GGUFTensorInfo info; info.type=GetParam();
            if (info.getBlockSize()!=256u) return; // Original32-element and floating sources have another legal geometry.
            for (const bool mapped : {false,true})
            {
                rebuildProjectionSource(mapped,"blk.7.ffn_down_exps.weight");
                auto *observed=observeProjectionSource(mapped);
                WeightManager manager(*loader_,mpi_ctx_);
                const auto ownership=MoEExpertProjectionOwnership::withinExpertIntermediateSlices(
                    {int(kExpertCount),int(kRowsPerExpert),int(columns_)},0,{{0,128},{128,int(columns_)-128}});
                WeightPlan plan(InferenceStrategy{.model_id=ModelContextId{56}},PhysicalMemoryOwner::RoutedExpertWeights);
                plan.add(projectionRequirement(ownership,WeightRole::MoEExpertDown));
                EXPECT_THROW(admitCPUWeightPlansForTest(manager,*loader_,{&plan}),std::invalid_argument);
                EXPECT_EQ(observed->native_calls,0u);EXPECT_EQ(observed->projection_calls,0);
                EXPECT_EQ(observed->whole_calls,0);
                EXPECT_EQ(observed->selection_calls,0);
            }
        }

        /** @test Original native blocks survive bounded sparse CPU windows for every existing loader format. */
        TEST_P(ModelLoaderExpertSelectionTest, CPUJointBoundedNativeWindowsRetainOriginalEncodedBytes)
        {
            for(const size_t target:{size_t{0},size_t{1},size_t{2}}) {
            const bool target_down=target==2;
            SCOPED_TRACE(target_down?"native-down-K-cut":target==1?"native-up-N-cut":"native-gate-N-cut");
            rebuildFiniteProjectionSource();
            GGUFTensorInfo info;info.type=GetParam();
            const size_t block_columns=info.getBlockSize()?info.getBlockSize():8u;
            const size_t block_bytes=info.getBlockSize()?info.getTypeSize():block_columns*info.getTypeSize();
            const size_t hidden=target_down?2:block_columns,intermediate=target_down?2*block_columns:2;
            const size_t row_bytes=target_down?2*block_bytes:block_bytes;
            const auto role=target_down?WeightRole::MoEExpertDown:target==1?WeightRole::MoEExpertUp:WeightRole::MoEExpertGate;
            const auto projection=target_down?ExpertTierWeightProjection::Down:target==1?ExpertTierWeightProjection::Up:ExpertTierWeightProjection::Gate;
            const std::string target_name=target_down?"blk.7.ffn_down_exps.weight":target==1?"blk.7.ffn_up_exps.weight":"blk.7.ffn_gate_exps.weight";
            const auto prototype=source_bytes_;
            const auto prototype_row_bytes=bytes_per_expert_/kRowsPerExpert;
            std::array<std::vector<uint8_t>,3> payloads;
            payloads[target].resize(512*2*row_bytes);
            for(size_t expert=0;expert<512;++expert)for(size_t row=0;row<2;++row)
                std::memcpy(payloads[target].data()+(expert*2+row)*row_bytes,
                    prototype.data()+((expert%kExpertCount)*2+row)*prototype_row_bytes,row_bytes);
            for(size_t source_role=0;source_role<3;++source_role)if(source_role!=target)
                payloads[source_role].resize(512*intermediate*hidden*sizeof(float),0);
            std::array<size_t,3> offsets{};
            for(size_t role=1;role<3;++role)offsets[role]=(offsets[role-1]+payloads[role-1].size()+31)/32*32;
            loader_.reset();
            {
                std::ofstream stream(model_path_,std::ios::binary|std::ios::trunc);ASSERT_TRUE(stream);
                stream.write("GGUF",4);writeScalar(stream,uint32_t{3});writeScalar(stream,uint64_t{3});writeScalar(stream,uint64_t{0});
                for(size_t role=0;role<3;++role) {
                    writeString(stream,std::string("blk.7.ffn_")+(role==0?"gate":role==1?"up":"down")+"_exps.weight");
                    writeScalar(stream,uint32_t{3});writeScalar(stream,uint64_t(role==2?intermediate:hidden));
                    writeScalar(stream,uint64_t(role==2?hidden:intermediate));writeScalar(stream,uint64_t{512});
                    writeScalar(stream,uint32_t(role==target?GetParam():GGUFTensorType::F32));writeScalar(stream,uint64_t(offsets[role]));
                }
                const auto directory=size_t(stream.tellp()),aligned=(directory+31)/32*32;
                std::vector<char> padding(aligned-directory,0);stream.write(padding.data(),std::streamsize(padding.size()));
                size_t written=0;
                for(size_t role=0;role<3;++role) {
                    std::vector<char> gap(offsets[role]-written,0);stream.write(gap.data(),std::streamsize(gap.size()));
                    stream.write(reinterpret_cast<const char*>(payloads[role].data()),std::streamsize(payloads[role].size()));
                    written=offsets[role]+payloads[role].size();
                }
                ASSERT_TRUE(stream);
            }
            loader_=std::make_unique<ModelLoader>(factory_.get());ASSERT_TRUE(loader_->loadModel(model_path_.string()));
            auto original=localCPUJointSourceTestPlan();
            for(size_t expert=0;expert<512;++expert)original.placements.front().routed_expert_tier[expert]=expert%8==0?0:1;
            const auto runtime=resolveMoEExpertOverlayRuntimePlan(std::make_shared<const MoERoutedExpertPlacementPlan>(original),
                {.current_world_rank=0});ASSERT_TRUE(runtime);
            const auto preparation=MoEExpertOverlayPreparationPlan::build(*runtime,*loader_).filteredForDevice(DeviceId::cpu());
            ASSERT_FALSE(preparation.requests().empty());const auto selected=preparation.requests().front().cpu_joint_source;
            ASSERT_TRUE(selected);ASSERT_EQ(selected->expertIds().size(),448u);
            EXPECT_EQ(selected->ownership().geometry().experts,512);
            const auto placement=CPUStoragePlacement::onNode(selected->participant().address.numa_node);
            const auto cap=placement.physicalBytes<uint8_t>(row_bytes*7);
            const auto source=loader_->nativeWeightSourceGeometry(target_name);
            const auto requirement=nativeJointRequirementForTest(selected,role,target_name);
            const auto projections=projectNativeWeightSource(requirement,source);ASSERT_EQ(projections.size(),448u);
            const auto read_request=nativeReadRequestForTest(source,projections.front().range);
            const auto read_geometry=resolveNativeWeightLoadStaging(read_request,placement,cap);
            const auto local=selected->cohort()->localManifest(selected->participantIndex());
            const auto gate=std::find_if(local.projections.begin(),local.projections.end(),
                [&](const auto& item){return item.projection==projection;});ASSERT_NE(gate,local.projections.end());
            const auto quote=quoteCPUWeightPreparation(source.original_type,gate->N,gate->K,
                WeightPrecision::NATIVE,CPUWeightConsumer::PreparedGemm);
            const auto consumer_geometry=quote?std::optional(resolveNativeWeightLoadStaging(*quote,placement,
                configuredNativeWeightLoadStagingBudgetBytes())):std::nullopt;
            const auto staging_bytes=std::max(read_geometry.physical_bytes,consumer_geometry?consumer_geometry->physical_bytes:0);
            const auto prepared_bytes=resolveExpertPreparedProjectionMemoryGeometry(gate->N,gate->K,gate->format).cpu_bytes;
            const auto total=staging_bytes+prepared_bytes;
            PhysicalMemoryPlanBuilder builder;
            builder.add({0,DeviceId::cpu(),total,total},PhysicalMemoryOwner::WeightLoadStaging,staging_bytes);
            builder.add({0,DeviceId::cpu(),total,total},PhysicalMemoryOwner::RoutedExpertWeights,prepared_bytes);
            const auto memory=std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()),0);
            size_t windows=0;
            for(size_t ordinal=0;ordinal<projections.size();++ordinal) {
                const auto& range=projections[ordinal].range;
                EXPECT_EQ(range.original_expert,size_t(selected->expertIds()[ordinal]));EXPECT_NE(range.original_expert%8,0u);
                windows+=visitNativeSourceForTest(*loader_,target_name,range,read_request,placement,memory,DeviceId::cpu(),cap,
                    [&](const auto& window,auto bytes,auto) {
                        for(size_t row=0;row<window.rows;++row) {
                            const size_t offset=(window.original_expert*2+window.first_row+row)*row_bytes+
                                window.first_column/source.native_group_elements*source.native_group_bytes;
                            EXPECT_EQ(std::memcmp(bytes.data()+row*window.native_row_bytes,
                                payloads[target].data()+offset,window.native_row_bytes),0);
                        }
                    });
                EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::WeightLoadStaging,
                    PhysicalMemoryMaterializationKind::NewAllocation),0u);
                // All448 sparse source matrices above retain the independent
                // encoded oracle. First/last matrices also execute the actual
                // common Store consumer and prove final lifetime after input retirement.
                if(ordinal!=0 && ordinal+1!=projections.size())continue;
                const ModelContextId model{279};WeightBinding original;
                original.binding_id=staticProjectedExpertBindingId(900,target_name,int(range.original_expert));
                original.identity=makeSourceWeightIdentity(target_name+"@expert"+std::to_string(range.original_expert),model,original.binding_id);
                original.identity.source_instance_id=900;original.identity.role=role;original.identity.derivation=WeightDerivationKind::ExpertSlice;
                original.identity.expert=int(range.original_expert);original.identity.layer=7;
                original.identity.overlay_domain=selected->participant().domain_name;
                original.identity.overlay_participant_world_rank=selected->participant().world_rank;
                original.identity.overlay_participant_index=original.identity.tp_rank_or_device_index=int(selected->participantIndex());
                original.cpu_joint_source=selected;original.projection_ownership=selected->ownership();original.native_source=projections[ordinal];
                original.slice=requirement.slice;original.slice.row_start=range.first_row;original.slice.row_count=range.rows;
                original.slice.col_start=range.first_column;original.slice.col_count=range.columns;
                original.slice.expert_start=range.original_expert;original.slice.expert_count=1;
                original.slice.expert_ids={int(range.original_expert)};original.slice.inner_is_presliced=true;
                original.residency.home_device=DeviceId::cpu();original.residency.resident_device=DeviceId::cpu();
                original.residency.host_policy=WeightHostPolicy::RequiredForCPUExecution;
                PreparedWeightStore store(model);store.installPhysicalMemoryAuthority(memory);
                auto preparation=store.beginCPUPreparation(original,CPUPreparedWeightSource{.original=source,.range=range},
                    PhysicalMemoryOwner::RoutedExpertWeights,placement);
                if(quote)visitNativeSourceForTest(*loader_,target_name,range,*quote,placement,memory,DeviceId::cpu(),
                    configuredNativeWeightLoadStagingBudgetBytes(),[&](const auto& window,auto bytes,auto scratch) {
                        preparation->append(window,bytes,scratch);
                    });
                else {
                    auto transaction=memory->acquireCPUWeightSourceTransaction(DeviceId::cpu());
                    forEachNativeWeightSourceWindow(source,range,cap,[&](const auto& window) {
                        preparation->readDirect(window,[&](auto bytes){loader_->readNativeWeightSourceWindow(target_name,window,bytes);});
                    });
                }
                auto committed=store.finishCPUPreparation(std::move(preparation));ASSERT_TRUE(committed.prepared);
                EXPECT_FALSE(committed.projection_source_owner);EXPECT_EQ(committed.native_source,projections[ordinal]);
                auto engine=store.gemmKernelLifetime(*committed.prepared);ASSERT_TRUE(engine);
                auto input=TestTensorFactory::createFP32Random({size_t(gate->K),1},-.5f,.5f,77);
                auto output=TestTensorFactory::createFP32Zeros({size_t(gate->N),1});
                const auto requirements=cpuProjectionTestRequirements(1,{engine.get()});
                std::unique_ptr<CPUProjectionTestWorkspace> workspace;
                if(requirements.total_bytes_with_alignment())workspace=std::make_unique<CPUProjectionTestWorkspace>(requirements);
                const auto run=[&] {ASSERT_TRUE(engine->multiply_tensor(input.get(),output.get(),1,gate->N,gate->K,
                    true,1.f,0.f,nullptr,mpi_ctx_.get(),-1,workspace?workspace->get():nullptr));};
                run();const auto* values=static_cast<const float*>(output->raw_data());const std::vector<float> before(values,values+gate->N);
                store.clear();run();EXPECT_EQ(std::memcmp(before.data(),output->raw_data(),before.size()*sizeof(float)),0);
                EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::WeightLoadStaging,
                    PhysicalMemoryMaterializationKind::NewAllocation),0u);
                engine.reset();committed={};
                EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RoutedExpertWeights,
                    PhysicalMemoryMaterializationKind::NewAllocation),0u);
            }
            EXPECT_GE(windows,448u);
            }
        }

#include "CPUJointBufferedSourceFailureLifecycle.inc"

#include "NativeWeightSourceWindowContract.inc"

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

#include "CPUJointBufferedSourceEconomy.inc"
