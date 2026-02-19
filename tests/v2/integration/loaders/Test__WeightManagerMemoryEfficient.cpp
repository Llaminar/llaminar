/**
 * @file Test__WeightManagerMemoryEfficient.cpp
 * @brief Integration tests for memory-efficient weight sharding
 *
 * These tests verify that:
 * 1. Each MPI rank loads ONLY its slice from GGUF (not full tensor)
 * 2. TensorSlice correctly reports inner_is_presliced=true
 * 3. GEMM kernels work correctly with pre-sliced data
 * 4. Inference produces correct results with memory-efficient loading
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "../../src/v2/loaders/WeightManager.h"
#include "../../src/v2/loaders/ModelLoader.h"
#include "../../src/v2/tensors/TensorSlice.h"
#include "../../src/v2/tensors/TensorFactory.h"
#include "../../src/v2/kernels/KernelFactory.h"
#include "../../src/v2/utils/MPIContext.h"
#include "../../src/v2/models/qwen/Qwen2Schema.h"
#include "../../src/v2/backends/DeviceId.h"
#include <cmath>
#include <numeric>

namespace llaminar2
{
    namespace test
    {

        namespace
        {
            ITensorGemm *getPreparedKernel(const TensorBase *tensor, DeviceId device_id = DeviceId::cpu())
            {
                auto *prepared = llaminar::v2::kernels::KernelFactory::getOrCreatePreparedGemmWeights(tensor, device_id);
                if (!prepared)
                {
                    return nullptr;
                }
                return llaminar::v2::kernels::KernelFactory::getOrCreateGemmEngine(prepared);
            }
        }

        class WeightManagerMemoryEfficientTest : public ::testing::Test
        {
        protected:
            void SetUp() override
            {
                model_path_ = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
            }

            std::string model_path_;
        };

        TEST_F(WeightManagerMemoryEfficientTest, ShardedWeightIsPresliced)
        {
            // Simulate 2-rank MPI setup
            auto mpi_ctx_rank0 = std::make_shared<MPIContext>(0, 2, MPI_COMM_NULL);
            auto mpi_ctx_rank1 = std::make_shared<MPIContext>(1, 2, MPI_COMM_NULL);

            TensorFactory factory0(*mpi_ctx_rank0);
            TensorFactory factory1(*mpi_ctx_rank1);

            ModelLoader loader0(&factory0);
            ModelLoader loader1(&factory1);

            if (!loader0.loadModel(model_path_) || !loader1.loadModel(model_path_))
            {
                GTEST_SKIP() << "Model file not found: " << model_path_;
            }

            // Create WeightManagers with SHARDED strategy
            WeightManager wm0(loader0, mpi_ctx_rank0, nullptr, WeightDistributionStrategy::SHARDED);
            WeightManager wm1(loader1, mpi_ctx_rank1, nullptr, WeightDistributionStrategy::SHARDED);

            // Set sharding config (required before accessing weights)
            Qwen2SchemaFactory schema_factory;
            wm0.setWeightShardingConfig(schema_factory.getWeightShardingConfig());
            wm1.setWeightShardingConfig(schema_factory.getWeightShardingConfig());

            // Load a row-parallel weight (attn_output.weight)
            const std::string tensor_name = "blk.0.attn_output.weight";

            auto weight0 = wm0.getWeightForDevice(tensor_name, DeviceId::cpu());
            auto weight1 = wm1.getWeightForDevice(tensor_name, DeviceId::cpu());

            ASSERT_NE(weight0, nullptr) << "Rank 0 failed to load weight";
            ASSERT_NE(weight1, nullptr) << "Rank 1 failed to load weight";

            // Verify weights are TensorSlice instances
            auto *slice0 = dynamic_cast<TensorSlice *>(weight0.get());
            auto *slice1 = dynamic_cast<TensorSlice *>(weight1.get());

            ASSERT_NE(slice0, nullptr) << "Rank 0 weight should be TensorSlice";
            ASSERT_NE(slice1, nullptr) << "Rank 1 weight should be TensorSlice";

            // Verify metadata indicates pre-sliced
            EXPECT_TRUE(slice0->metadata().inner_is_presliced)
                << "Rank 0 slice should have inner_is_presliced=true (memory efficient)";
            EXPECT_TRUE(slice1->metadata().inner_is_presliced)
                << "Rank 1 slice should have inner_is_presliced=true (memory efficient)";

            // Verify slice ranges are different
            EXPECT_EQ(slice0->metadata().slice_start, 0) << "Rank 0 should start at row 0";
            EXPECT_EQ(slice0->metadata().slice_end, 448) << "Rank 0 should end at row 448";

            EXPECT_EQ(slice1->metadata().slice_start, 448) << "Rank 1 should start at row 448";
            EXPECT_EQ(slice1->metadata().slice_end, 896) << "Rank 1 should end at row 896";
        }

        TEST_F(WeightManagerMemoryEfficientTest, InnerTensorHasSliceShape)
        {
            auto mpi_ctx = std::make_shared<MPIContext>(0, 2, MPI_COMM_NULL);
            TensorFactory factory(*mpi_ctx);
            ModelLoader loader(&factory);

            if (!loader.loadModel(model_path_))
            {
                GTEST_SKIP() << "Model file not found: " << model_path_;
            }

            WeightManager wm(loader, mpi_ctx, nullptr, WeightDistributionStrategy::SHARDED);

            // Set sharding config (required before accessing weights)
            Qwen2SchemaFactory schema_factory;
            wm.setWeightShardingConfig(schema_factory.getWeightShardingConfig());

            // FFN Down is INPUT_PARALLEL (column-sliced) in Phase 4b-2
            // This means it splits the input dimension (columns) to match Gate/Up output
            const std::string tensor_name = "blk.0.ffn_down.weight";
            auto weight = wm.getWeightForDevice(tensor_name, DeviceId::cpu());
            ASSERT_NE(weight, nullptr);

            auto *slice = dynamic_cast<TensorSlice *>(weight.get());
            ASSERT_NE(slice, nullptr);

            // Get the inner tensor
            const TensorBase *inner = slice->inner();
            ASSERT_NE(inner, nullptr);

            const auto &inner_shape = inner->shape();
            ASSERT_EQ(inner_shape.size(), 2);

            // For ffn_down.weight [896, 4864] with 2 ranks using INPUT_PARALLEL:
            // - INPUT_PARALLEL splits columns (the input dimension K)
            // - Rank 0 gets cols [0, 2432), shape [896, 2432]
            // - Rank 1 gets cols [2432, 4864), shape [896, 2432]
            // This allows Down to process the local FFN output from Gate/Up
            EXPECT_EQ(inner_shape[0], 896)
                << "Inner tensor should have full rows (output dimension)";
            EXPECT_EQ(inner_shape[1], 2432)
                << "Inner tensor should have ONLY the slice columns (input dimension)";

            // Verify original dimensions in metadata
            EXPECT_EQ(slice->original_rows(), 896)
                << "Metadata should record original total rows";
        }

        TEST_F(WeightManagerMemoryEfficientTest, GemmKernelWorksWithPreslicedData)
        {
            auto mpi_ctx = std::make_shared<MPIContext>(0, 2, MPI_COMM_NULL);
            TensorFactory factory(*mpi_ctx);
            ModelLoader loader(&factory);

            if (!loader.loadModel(model_path_))
            {
                GTEST_SKIP() << "Model file not found: " << model_path_;
            }

            WeightManager wm(loader, mpi_ctx, nullptr, WeightDistributionStrategy::SHARDED);

            // Set sharding config (required before accessing weights)
            Qwen2SchemaFactory schema_factory;
            wm.setWeightShardingConfig(schema_factory.getWeightShardingConfig());

            // Load COLUMN_PARALLEL weight (row-sliced)
            // ffn_gate.weight is [4864, 896] -> each rank gets [2432, 896] (rows sliced, K preserved)
            const std::string tensor_name = "blk.0.ffn_gate.weight";
            auto weight = wm.getWeightForDevice(tensor_name, DeviceId::cpu());
            ASSERT_NE(weight, nullptr);

            auto *slice = dynamic_cast<TensorSlice *>(weight.get());
            ASSERT_NE(slice, nullptr);
            ASSERT_TRUE(slice->metadata().inner_is_presliced);

            // Get cached GEMM engine (should work with pre-sliced data)
            // Note: prepared-weight path is required because raw data may have been released
            // after the WeightManager packed the weights during getWeightForDevice()
            auto *gemm = getPreparedKernel(slice, DeviceId::cpu());
            ASSERT_NE(gemm, nullptr) << "Failed to get GEMM kernel from pre-sliced data";

            // Prepare test input: [batch=4, K=896]
            // ffn_gate.weight [4864, 896] sliced to [2432, 896]
            // GEMM: output[m, n_local] = input[m, k] * W^T, where n_local=2432, k=896
            int m = 4;
            int k = 896;
            int n = 2432; // Output is slice rows (2432 for rank 0)

            std::vector<float> input(m * k, 0.1f);
            std::vector<float> output(m * n, 0.0f);

            // Execute GEMM: output = input * weight^T
            bool success = gemm->multiply(input.data(), output.data(), m, n, k);
            EXPECT_TRUE(success) << "GEMM with pre-sliced weight should succeed";

            // Verify output is not all zeros (indicates computation happened)
            bool has_nonzero = false;
            for (float v : output)
            {
                EXPECT_FALSE(std::isnan(v)) << "GEMM output contains NaN";
                if (std::abs(v) > 1e-6f)
                    has_nonzero = true;
            }
            EXPECT_TRUE(has_nonzero) << "GEMM output is all zeros - computation may have failed";
        }

        TEST_F(WeightManagerMemoryEfficientTest, CombinedSlicesMatchFullTensor)
        {
            // Load full tensor via rank 0 (simulated single-rank mode)
            auto mpi_ctx_single = std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);
            TensorFactory factory_single(*mpi_ctx_single);
            ModelLoader loader_single(&factory_single);

            if (!loader_single.loadModel(model_path_))
            {
                GTEST_SKIP() << "Model file not found: " << model_path_;
            }

            // Load full tensor with REPLICATED strategy (no sharding)
            // Use ffn_gate.weight which is COLUMN_PARALLEL (row-sliced)
            // Shape: [4864, 896] -> each rank gets [2432, 896]
            WeightManager wm_full(loader_single, mpi_ctx_single, nullptr, WeightDistributionStrategy::REPLICATED);

            // Configure weight sharding for all managers (needed for isGemmWeight check)
            Qwen2SchemaFactory schema_factory;
            wm_full.setWeightShardingConfig(schema_factory.getWeightShardingConfig());

            const std::string tensor_name = "blk.0.ffn_gate.weight";
            auto full_weight = wm_full.getWeightForDevice(tensor_name, DeviceId::cpu());
            ASSERT_NE(full_weight, nullptr);

            // Load sliced tensors (simulated 2-rank)
            // IMPORTANT: Use REPLICATED strategy for sliced loaders too, so we can
            // compare FP32 dequantized values (raw data won't be released)
            auto mpi_ctx_rank0 = std::make_shared<MPIContext>(0, 2, MPI_COMM_NULL);
            auto mpi_ctx_rank1 = std::make_shared<MPIContext>(1, 2, MPI_COMM_NULL);

            TensorFactory factory0(*mpi_ctx_rank0);
            TensorFactory factory1(*mpi_ctx_rank1);

            ModelLoader loader0(&factory0);
            ModelLoader loader1(&factory1);
            loader0.loadModel(model_path_);
            loader1.loadModel(model_path_);

            // Use SHARDED to get row-sliced tensors, but note that raw data
            // will be released after GEMM packing
            WeightManager wm0(loader0, mpi_ctx_rank0, nullptr, WeightDistributionStrategy::SHARDED);
            WeightManager wm1(loader1, mpi_ctx_rank1, nullptr, WeightDistributionStrategy::SHARDED);

            // Configure weight sharding for both managers (reuse schema_factory)
            wm0.setWeightShardingConfig(schema_factory.getWeightShardingConfig());
            wm1.setWeightShardingConfig(schema_factory.getWeightShardingConfig());

            auto weight0 = wm0.getWeightForDevice(tensor_name, DeviceId::cpu());
            auto weight1 = wm1.getWeightForDevice(tensor_name, DeviceId::cpu());

            ASSERT_NE(weight0, nullptr);
            ASSERT_NE(weight1, nullptr);

            auto *slice0 = dynamic_cast<TensorSlice *>(weight0.get());
            auto *slice1 = dynamic_cast<TensorSlice *>(weight1.get());

            ASSERT_NE(slice0, nullptr);
            ASSERT_NE(slice1, nullptr);

            // Note: Raw data release happens through WeightPreloader::packWeight() path,
            // not through direct GEMM engine lookup calls.
            // The memory-efficient behavior is tested via the preloader tests.
            // Here we just verify that combined slices produce correct GEMM output.

            // Since raw data is released, we verify correctness by running GEMM
            // and comparing outputs instead of FP32 dequant values
            const auto &full_shape = full_weight->shape();
            size_t total_rows = full_shape[0];
            size_t cols = full_shape[1]; // K dimension
            size_t half_rows = total_rows / 2;

            // Run GEMM on full tensor
            int m = 4;
            int k = static_cast<int>(cols);
            int n_full = static_cast<int>(total_rows);
            int n_half = static_cast<int>(half_rows);

            std::vector<float> input(m * k, 0.1f);
            std::vector<float> output_full(m * n_full, 0.0f);
            std::vector<float> output0(m * n_half, 0.0f);
            std::vector<float> output1(m * n_half, 0.0f);

            auto *gemm_full = getPreparedKernel(full_weight.get(), DeviceId::cpu());
            auto *gemm0 = getPreparedKernel(slice0, DeviceId::cpu());
            auto *gemm1 = getPreparedKernel(slice1, DeviceId::cpu());

            ASSERT_NE(gemm_full, nullptr);
            ASSERT_NE(gemm0, nullptr);
            ASSERT_NE(gemm1, nullptr);

            // Execute GEMMs
            EXPECT_TRUE(gemm_full->multiply(input.data(), output_full.data(), m, n_full, k));
            EXPECT_TRUE(gemm0->multiply(input.data(), output0.data(), m, n_half, k));
            EXPECT_TRUE(gemm1->multiply(input.data(), output1.data(), m, n_half, k));

            // Compare GEMM outputs: slice0 should match rows [0, half_rows) of full
            float max_diff = 0.0f;
            for (int row = 0; row < m; ++row)
            {
                for (size_t col = 0; col < half_rows; ++col)
                {
                    float full_val = output_full[row * n_full + col];
                    float slice_val = output0[row * n_half + col];
                    max_diff = std::max(max_diff, std::abs(full_val - slice_val));
                }
            }
            EXPECT_LT(max_diff, 0.01f)
                << "Slice0 GEMM output should match rows [0, half_rows) of full, max_diff=" << max_diff;

            // slice1 should match rows [half_rows, total_rows) of full
            max_diff = 0.0f;
            for (int row = 0; row < m; ++row)
            {
                for (size_t col = 0; col < half_rows; ++col)
                {
                    float full_val = output_full[row * n_full + half_rows + col];
                    float slice_val = output1[row * n_half + col];
                    max_diff = std::max(max_diff, std::abs(full_val - slice_val));
                }
            }
            EXPECT_LT(max_diff, 0.01f)
                << "Slice1 GEMM output should match rows [half_rows, total_rows) of full, max_diff=" << max_diff;
        }

        TEST_F(WeightManagerMemoryEfficientTest, ReplicatedWeightsNotSliced)
        {
            // Verify that replicated weights (like norms) are not sliced
            auto mpi_ctx = std::make_shared<MPIContext>(0, 2, MPI_COMM_NULL);
            TensorFactory factory(*mpi_ctx);
            ModelLoader loader(&factory);

            if (!loader.loadModel(model_path_))
            {
                GTEST_SKIP() << "Model file not found: " << model_path_;
            }

            WeightManager wm(loader, mpi_ctx, nullptr, WeightDistributionStrategy::SHARDED);

            // Configure weight sharding
            Qwen2SchemaFactory schema_factory;
            wm.setWeightShardingConfig(schema_factory.getWeightShardingConfig());

            // Load a replicated weight (attention norm)
            const std::string norm_name = "blk.0.attn_norm.weight";
            auto norm_weight = wm.getWeightForDevice(norm_name, DeviceId::cpu());
            ASSERT_NE(norm_weight, nullptr);

            // Should NOT be a TensorSlice (replicated weights bypass slicing)
            auto *slice = dynamic_cast<TensorSlice *>(norm_weight.get());
            EXPECT_EQ(slice, nullptr)
                << "Replicated weights should not be wrapped in TensorSlice";
        }

    } // namespace test
} // namespace llaminar2
