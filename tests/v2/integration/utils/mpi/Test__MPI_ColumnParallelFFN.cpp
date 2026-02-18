/**
 * @file Test__MPI_ColumnParallelFFN.cpp
 * @brief Integration tests for Phase 4: Column-Parallel FFN (Gate/Up)
 *
 * Validates that FFN weights are correctly sharded across MPI ranks:
 * - Gate weight: [d_ff, d_model] → [d_ff_local, d_model] per rank
 * - Up weight: [d_ff, d_model] → [d_ff_local, d_model] per rank
 * - Down weight: [d_model, d_ff] → [d_model, d_ff_local] per rank (INPUT_PARALLEL)
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <memory>
#include <cmath>

#include "loaders/WeightManager.h"
#include "loaders/ModelLoader.h"
#include "models/qwen/Qwen2Schema.h"
#include "tensors/TensorFactory.h"
#include "tensors/Tensors.h"
#include "kernels/KernelFactory.h"
#include "backends/DeviceId.h"
#include "utils/MPIContext.h"
#include "utils/Logger.h"

using namespace llaminar2;

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

// Qwen 2.5 0.5B model constants
static constexpr int D_MODEL = 896;
static constexpr int D_FF = 4864;
static constexpr int N_LAYERS = 24;
static const std::string MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

class Test__MPI_ColumnParallelFFN : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Get MPI context
        int rank, world_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        mpi_ctx_ = std::make_shared<MPIContext>(rank, world_size, MPI_COMM_WORLD);

        // Expect 2 ranks for these tests
        ASSERT_EQ(world_size, 2) << "These tests require exactly 2 MPI ranks";

        // Create tensor factory
        factory_ = std::make_shared<TensorFactory>(*mpi_ctx_);

        // Load model
        loader_ = std::make_unique<ModelLoader>(factory_.get());
        if (!loader_->loadModel(MODEL_PATH))
        {
            GTEST_SKIP() << "Model file not found: " << MODEL_PATH;
        }

        // Create weight manager with SHARDED strategy and Qwen2 sharding config
        weight_manager_ = std::make_unique<WeightManager>(
            *loader_,
            mpi_ctx_,
            nullptr, // placement_map
            WeightDistributionStrategy::SHARDED);

        // Set Qwen2 sharding configuration
        Qwen2SchemaFactory schema_factory;
        weight_manager_->setWeightShardingConfig(schema_factory.getWeightShardingConfig());
        sharding_config_ = schema_factory.getWeightShardingConfig();
    }

    void TearDown() override
    {
        weight_manager_.reset();
        loader_.reset();
        factory_.reset();
        mpi_ctx_.reset();
    }

    std::unique_ptr<ModelLoader> loader_;
    std::unique_ptr<WeightManager> weight_manager_;
    std::shared_ptr<MPIContext> mpi_ctx_;
    std::shared_ptr<TensorFactory> factory_;
    WeightShardingConfig sharding_config_;

    /**
     * @brief Helper to get sharding mode using schema config
     */
    ShardingMode getShardingMode(const std::string &name) const
    {
        WeightShardingMode mode = sharding_config_.getMode(name);
        switch (mode)
        {
        case WeightShardingMode::ColumnParallel:
            return ShardingMode::COLUMN_PARALLEL;
        case WeightShardingMode::RowParallel:
            return ShardingMode::ROW_PARALLEL;
        case WeightShardingMode::InputParallel:
            return ShardingMode::INPUT_PARALLEL;
        case WeightShardingMode::Replicate:
        default:
            return ShardingMode::REPLICATE;
        }
    }
};

// =============================================================================
// Test 1: Verify Gate/Up use COLUMN_PARALLEL, Down uses INPUT_PARALLEL
// =============================================================================

TEST_F(Test__MPI_ColumnParallelFFN, ShardingModeDetection)
{
    // Gate should be COLUMN_PARALLEL
    EXPECT_EQ(getShardingMode("blk.0.ffn_gate.weight"),
              ShardingMode::COLUMN_PARALLEL);

    // Up should be COLUMN_PARALLEL
    EXPECT_EQ(getShardingMode("blk.0.ffn_up.weight"),
              ShardingMode::COLUMN_PARALLEL);

    // Down should be INPUT_PARALLEL (split input dimension to match Gate/Up output)
    EXPECT_EQ(getShardingMode("blk.0.ffn_down.weight"),
              ShardingMode::INPUT_PARALLEL);

    LOG_INFO("[Test] FFN sharding mode detection PASSED - "
             << "Gate/Up=COLUMN_PARALLEL, Down=INPUT_PARALLEL");
}

// =============================================================================
// Test 2: Verify local FFN dimension calculation
// =============================================================================

TEST_F(Test__MPI_ColumnParallelFFN, LocalFFNDimensionCalculation)
{
    int world_size = mpi_ctx_->world_size();
    int d_ff_local = D_FF / world_size;

    EXPECT_EQ(d_ff_local, 2432) << "With 2 ranks, d_ff_local should be 4864/2 = 2432";

    LOG_INFO("[Test] Local FFN dimension calculation PASSED: d_ff_local=" << d_ff_local);
}

// =============================================================================
// Test 3: Verify Gate weight has correct local shape
// =============================================================================

TEST_F(Test__MPI_ColumnParallelFFN, GateWeightLocalShape)
{
    int rank = mpi_ctx_->rank();
    int world_size = mpi_ctx_->world_size();
    int d_ff_local = D_FF / world_size;

    // Load gate weight (will be column-sliced)
    auto gate_weight = weight_manager_->getWeight("blk.0.ffn_gate.weight", DeviceId::cpu());
    ASSERT_NE(gate_weight, nullptr) << "Failed to load gate weight";

    // Check shape: should be [d_ff_local, d_model]
    auto shape = gate_weight->shape();
    ASSERT_EQ(shape.size(), 2) << "Gate weight should be 2D";

    // Row dimension should be local (d_ff_local)
    EXPECT_EQ(shape[0], static_cast<size_t>(d_ff_local))
        << "Gate rows should be d_ff_local=" << d_ff_local;

    // Column dimension unchanged (d_model)
    EXPECT_EQ(shape[1], static_cast<size_t>(D_MODEL))
        << "Gate cols should be d_model=" << D_MODEL;

    LOG_DEBUG("[Rank " << rank << "] Gate weight local shape: ["
                       << shape[0] << ", " << shape[1] << "]");

    LOG_INFO("[Test] Gate weight local shape PASSED");
}

// =============================================================================
// Test 4: Verify Up weight has correct local shape
// =============================================================================

TEST_F(Test__MPI_ColumnParallelFFN, UpWeightLocalShape)
{
    int rank = mpi_ctx_->rank();
    int world_size = mpi_ctx_->world_size();
    int d_ff_local = D_FF / world_size;

    // Load up weight (will be column-sliced)
    auto up_weight = weight_manager_->getWeight("blk.0.ffn_up.weight", DeviceId::cpu());
    ASSERT_NE(up_weight, nullptr) << "Failed to load up weight";

    // Check shape: should be [d_ff_local, d_model]
    auto shape = up_weight->shape();
    ASSERT_EQ(shape.size(), 2) << "Up weight should be 2D";

    // Row dimension should be local (d_ff_local)
    EXPECT_EQ(shape[0], static_cast<size_t>(d_ff_local))
        << "Up rows should be d_ff_local=" << d_ff_local;

    // Column dimension unchanged (d_model)
    EXPECT_EQ(shape[1], static_cast<size_t>(D_MODEL))
        << "Up cols should be d_model=" << D_MODEL;

    LOG_DEBUG("[Rank " << rank << "] Up weight local shape: ["
                       << shape[0] << ", " << shape[1] << "]");

    LOG_INFO("[Test] Up weight local shape PASSED");
}

// =============================================================================
// Test 5: Verify Down weight has correct local shape (INPUT_PARALLEL)
// =============================================================================

TEST_F(Test__MPI_ColumnParallelFFN, DownWeightLocalShape)
{
    int rank = mpi_ctx_->rank();
    int world_size = mpi_ctx_->world_size();
    int d_ff_local = D_FF / world_size;

    // Load down weight (will be input-parallel sliced)
    auto down_weight = weight_manager_->getWeight("blk.0.ffn_down.weight", DeviceId::cpu());
    ASSERT_NE(down_weight, nullptr) << "Failed to load down weight";

    // Check shape: should be [d_model, d_ff_local]
    auto shape = down_weight->shape();
    ASSERT_EQ(shape.size(), 2) << "Down weight should be 2D";

    // Row dimension unchanged (d_model)
    EXPECT_EQ(shape[0], static_cast<size_t>(D_MODEL))
        << "Down rows should be d_model=" << D_MODEL;

    // Column dimension should be local (d_ff_local)
    EXPECT_EQ(shape[1], static_cast<size_t>(d_ff_local))
        << "Down cols should be d_ff_local=" << d_ff_local;

    LOG_DEBUG("[Rank " << rank << "] Down weight local shape: ["
                       << shape[0] << ", " << shape[1] << "]");

    LOG_INFO("[Test] Down weight local shape PASSED");
}

// =============================================================================
// Test 6: Verify FFN projection produces correct local output
// =============================================================================

TEST_F(Test__MPI_ColumnParallelFFN, GateProjectionLocalOutput)
{
    int rank = mpi_ctx_->rank();
    int world_size = mpi_ctx_->world_size();
    int d_ff_local = D_FF / world_size;
    int seq_len = 8;

    // Load gate weight
    auto gate_weight = weight_manager_->getWeight("blk.0.ffn_gate.weight", DeviceId::cpu());
    ASSERT_NE(gate_weight, nullptr);

    // Create input tensor [seq_len, d_model]
    auto input = factory_->createFP32({static_cast<size_t>(seq_len),
                                       static_cast<size_t>(D_MODEL)},
                                      DeviceId::cpu());
    ASSERT_NE(input, nullptr);

    // Initialize with deterministic values
    float *input_data = input->mutable_data();
    for (int i = 0; i < seq_len * D_MODEL; ++i)
    {
        input_data[i] = 0.01f * (i % 100);
    }

    // Create output tensor [seq_len, d_ff_local]
    auto output = factory_->createFP32({static_cast<size_t>(seq_len),
                                        static_cast<size_t>(d_ff_local)},
                                       DeviceId::cpu());
    ASSERT_NE(output, nullptr);

    // Use KernelFactory for sliced tensor GEMM
    auto *gemm = getPreparedKernel(gate_weight.get(), DeviceId::cpu());
    ASSERT_NE(gemm, nullptr);

    bool success = gemm->multiply(
        input->data(),
        output->mutable_data(),
        seq_len,    // m
        d_ff_local, // n (local output dimension)
        D_MODEL,    // k
        1.0f, 0.0f);
    EXPECT_TRUE(success) << "Gate GEMM failed";

    // Verify output has expected shape
    EXPECT_EQ(output->shape()[0], static_cast<size_t>(seq_len));
    EXPECT_EQ(output->shape()[1], static_cast<size_t>(d_ff_local));

    // Check for non-zero output (basic sanity)
    const float *out_data = output->data();
    float max_val = 0.0f;
    for (int i = 0; i < seq_len * d_ff_local; ++i)
    {
        max_val = std::max(max_val, std::abs(out_data[i]));
    }
    EXPECT_GT(max_val, 0.0f) << "Gate projection produced all zeros";

    LOG_INFO("[Test] Gate projection local output PASSED - max_val=" << max_val);
}

// =============================================================================
// Test 7: Verify buffer dimensions match config
// =============================================================================

TEST_F(Test__MPI_ColumnParallelFFN, BufferDimensionsWithLocalFFN)
{
    int world_size = mpi_ctx_->world_size();
    int d_ff_local = D_FF / world_size;
    int seq_len = 8;

    // Simulate buffer allocation with local FFN dimension
    // Gate buffer should be [seq_len, d_ff_local]
    auto gate_buffer = factory_->createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_ff_local)}, DeviceId::cpu());
    ASSERT_NE(gate_buffer, nullptr);

    // Up buffer should be [seq_len, d_ff_local]
    auto up_buffer = factory_->createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_ff_local)}, DeviceId::cpu());
    ASSERT_NE(up_buffer, nullptr);

    // FFN output buffer (after SwiGLU) should be [seq_len, d_ff_local]
    auto ffn_output = factory_->createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_ff_local)}, DeviceId::cpu());
    ASSERT_NE(ffn_output, nullptr);

    EXPECT_EQ(gate_buffer->shape()[1], static_cast<size_t>(d_ff_local));
    EXPECT_EQ(up_buffer->shape()[1], static_cast<size_t>(d_ff_local));
    EXPECT_EQ(ffn_output->shape()[1], static_cast<size_t>(d_ff_local));

    LOG_INFO("[Test] Buffer dimensions with local FFN PASSED");
    LOG_INFO("  Buffers: Gate/Up/FFN_output=[" << seq_len << ", " << d_ff_local << "]");
}

// =============================================================================
// Main: Initialize MPI and run tests
// =============================================================================

int main(int argc, char **argv)
{
    // Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    // Initialize GTest
    ::testing::InitGoogleTest(&argc, argv);

    // Run tests
    int result = RUN_ALL_TESTS();

    // Finalize MPI
    MPI_Finalize();

    return result;
}
