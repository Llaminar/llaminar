/**
 * @file Test__MPI_ColumnParallelLMHead.cpp
 * @brief Integration tests for Phase 5: Column-Parallel LM Head
 *
 * Validates that LM head weights are correctly sharded across MPI ranks:
 * - output.weight: [vocab_size, d_model] → [vocab_local, d_model] per rank
 *
 * Tests:
 * 1. ShardingModeDetection - Verify output.weight uses COLUMN_PARALLEL
 * 2. LocalVocabDimensionCalculation - Verify vocab_local = vocab_size / world_size
 * 3. WeightShapeValidation - Verify sharded weight shape is [vocab_local, d_model]
 * 4. AllGatherStageIntegration - Verify AllGather collects distributed logits
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <memory>
#include <cmath>
#include <vector>

#include "loaders/WeightManager.h"
#include "loaders/ModelLoader.h"
#include "models/qwen/Qwen2Schema.h"
#include "tensors/TensorFactory.h"
#include "tensors/Tensors.h"
#include "kernels/KernelFactory.h"
#include "execution/compute_stages/ComputeStages.h"
#include "backends/DeviceId.h"
#include "utils/MPIContext.h"
#include "utils/Logger.h"

using namespace llaminar2;

// Qwen 2.5 0.5B model constants
static constexpr int D_MODEL = 896;
static constexpr int VOCAB_SIZE = 151936;
static constexpr int N_LAYERS = 24;
static const std::string MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

class Test__MPI_ColumnParallelLMHead : public ::testing::Test
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

        // Create weight manager with SHARDED strategy
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
// Test 1: Verify output.weight uses COLUMN_PARALLEL sharding
// =============================================================================

TEST_F(Test__MPI_ColumnParallelLMHead, ShardingModeDetection)
{
    // output.weight should be COLUMN_PARALLEL (split by vocab dimension)
    EXPECT_EQ(getShardingMode("output.weight"),
              ShardingMode::COLUMN_PARALLEL);

    LOG_INFO("[Test] LM Head sharding mode detection PASSED - "
             << "output.weight=COLUMN_PARALLEL");
}

// =============================================================================
// Test 2: Verify vocab_local calculation
// =============================================================================

TEST_F(Test__MPI_ColumnParallelLMHead, LocalVocabDimensionCalculation)
{
    int rank = mpi_ctx_->rank();
    int world_size = mpi_ctx_->world_size();

    // Calculate expected local vocab size
    int vocab_local = VOCAB_SIZE / world_size;

    // Qwen 2.5 0.5B vocab_size = 151936, with 2 ranks: 75968 per rank
    EXPECT_EQ(vocab_local, 75968) << "With 2 ranks, vocab_local should be 151936/2 = 75968";

    LOG_INFO("[Test] Local vocab dimension calculation PASSED: vocab_local=" << vocab_local);
}

// =============================================================================
// Test 3: Verify sharded weight shape is [vocab_local, d_model]
// =============================================================================

TEST_F(Test__MPI_ColumnParallelLMHead, WeightShapeValidation)
{
    int rank = mpi_ctx_->rank();
    int world_size = mpi_ctx_->world_size();
    int vocab_local = VOCAB_SIZE / world_size;

    // Get sharded LM head weight
    auto lm_head = weight_manager_->getWeight("output.weight");
    ASSERT_NE(lm_head, nullptr) << "Failed to load output.weight";

    // Check shape: should be [vocab_local, d_model]
    const auto &shape = lm_head->shape();
    ASSERT_GE(shape.size(), 2) << "Expected 2D tensor for LM head weight";

    // Row dimension should be vocab_local (COLUMN_PARALLEL splits output dim)
    EXPECT_EQ(shape[0], static_cast<size_t>(vocab_local))
        << "LM head rows should be vocab_local=" << vocab_local
        << " (rank " << rank << ")";

    // Column dimension should remain d_model
    EXPECT_EQ(shape[1], static_cast<size_t>(D_MODEL))
        << "LM head cols should be d_model=" << D_MODEL;

    LOG_INFO("[Test] LM Head weight shape PASSED: ["
             << shape[0] << ", " << shape[1] << "] on rank " << rank);
}

// =============================================================================
// Test 4: Verify AllGatherStage collects distributed logits correctly
// NOTE: Uses seq_len=1 to test basic allgather without 2D layout complexity.
// Multi-row allgather requires per-row gathering which is tested in FullForwardDataFlow.
// =============================================================================

TEST_F(Test__MPI_ColumnParallelLMHead, AllGatherStageIntegration)
{
    int rank = mpi_ctx_->rank();
    int world_size = mpi_ctx_->world_size();
    int vocab_local = VOCAB_SIZE / world_size;

    // Use seq_len=1 to test basic 1D allgather
    const int seq_len = 1;

    // Create local logits buffer [seq_len, vocab_local]
    auto local_logits = factory_->createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(vocab_local)}, DeviceId::cpu());

    // Create full logits buffer [seq_len, vocab_size]
    auto full_logits = factory_->createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(VOCAB_SIZE)}, DeviceId::cpu());

    // Fill local logits with rank-specific values
    // Rank 0: fill with values [0, vocab_local)
    // Rank 1: fill with values [vocab_local, vocab_size)
    float *local_data = static_cast<FP32Tensor *>(local_logits.get())->mutable_data();
    for (int s = 0; s < seq_len; ++s)
    {
        for (int v = 0; v < vocab_local; ++v)
        {
            // Value = position_in_vocab + seq_idx * 0.001 (to distinguish sequences)
            int global_vocab_idx = rank * vocab_local + v;
            local_data[s * vocab_local + v] = static_cast<float>(global_vocab_idx) + s * 0.001f;
        }
    }

    // Create AllGatherStage
    AllGatherStage::Params params;
    params.local_input = local_logits.get();
    params.full_output = full_logits.get();
    params.mpi_ctx = mpi_ctx_.get();

    auto stage = ComputeStageFactory::createAllGather(params);
    ASSERT_NE(stage, nullptr);

    // Execute AllGather (no device context needed for MPI operations)
    bool success = stage->execute(nullptr);
    ASSERT_TRUE(success) << "AllGatherStage execution failed";

    // Verify full logits contain combined data from all ranks
    const float *full_data = static_cast<FP32Tensor *>(full_logits.get())->data();

    // Check a few values from each rank's portion
    // With MPI_Allgather, the output is organized as:
    // [rank0_data][rank1_data] for each row

    // Sample positions to check:
    // - Rank 0's first element (global vocab idx = 0)
    // - Rank 0's last element (global vocab idx = vocab_local - 1)
    // - Rank 1's first element (global vocab idx = vocab_local)
    // - Rank 1's last element (global vocab idx = vocab_size - 1)

    for (int s = 0; s < seq_len; ++s)
    {
        // Check first element from rank 0
        int idx_r0_first = s * VOCAB_SIZE + 0;
        float expected_r0_first = 0.0f + s * 0.001f;
        EXPECT_NEAR(full_data[idx_r0_first], expected_r0_first, 1e-5f)
            << "Mismatch at seq=" << s << " vocab=0 (rank 0 first)";

        // Check last element from rank 0
        int idx_r0_last = s * VOCAB_SIZE + (vocab_local - 1);
        float expected_r0_last = static_cast<float>(vocab_local - 1) + s * 0.001f;
        EXPECT_NEAR(full_data[idx_r0_last], expected_r0_last, 1e-5f)
            << "Mismatch at seq=" << s << " vocab=" << (vocab_local - 1) << " (rank 0 last)";

        // Check first element from rank 1
        int idx_r1_first = s * VOCAB_SIZE + vocab_local;
        float expected_r1_first = static_cast<float>(vocab_local) + s * 0.001f;
        EXPECT_NEAR(full_data[idx_r1_first], expected_r1_first, 1e-5f)
            << "Mismatch at seq=" << s << " vocab=" << vocab_local << " (rank 1 first)";

        // Check last element from rank 1
        int idx_r1_last = s * VOCAB_SIZE + (VOCAB_SIZE - 1);
        float expected_r1_last = static_cast<float>(VOCAB_SIZE - 1) + s * 0.001f;
        EXPECT_NEAR(full_data[idx_r1_last], expected_r1_last, 1e-5f)
            << "Mismatch at seq=" << s << " vocab=" << (VOCAB_SIZE - 1) << " (rank 1 last)";
    }

    LOG_INFO("[Test] AllGatherStage integration PASSED on rank " << rank);
}

// =============================================================================
// Test 5: Verify full forward pass with column-parallel LM head
// (Tests the complete data flow: hidden -> LM head -> AllGather -> full logits)
// =============================================================================

TEST_F(Test__MPI_ColumnParallelLMHead, FullForwardDataFlow)
{
    int rank = mpi_ctx_->rank();
    int world_size = mpi_ctx_->world_size();
    int vocab_local = VOCAB_SIZE / world_size;

    const int seq_len = 1; // Single token decode

    // Create mock hidden states [seq_len, d_model]
    auto hidden = factory_->createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(D_MODEL)}, DeviceId::cpu());

    // Fill hidden with deterministic pattern
    float *hidden_data = static_cast<FP32Tensor *>(hidden.get())->mutable_data();
    for (int d = 0; d < D_MODEL; ++d)
    {
        hidden_data[d] = static_cast<float>(d % 100) / 100.0f;
    }

    // Get sharded LM head weight
    auto lm_head = weight_manager_->getWeight("output.weight");
    ASSERT_NE(lm_head, nullptr) << "Failed to load output.weight";

    // Create local and full logits buffers
    auto local_logits = factory_->createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(vocab_local)}, DeviceId::cpu());
    auto full_logits = factory_->createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(VOCAB_SIZE)}, DeviceId::cpu());

    // Compute local logits: hidden @ lm_head.T
    // (This would be done by GEMMStage in full pipeline)
    auto gemm = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(lm_head.get());
    ASSERT_NE(gemm, nullptr) << "Failed to create GEMM kernel for LM head";

    bool gemm_success = gemm->multiply(
        hidden_data,                                                   // A: [seq_len, d_model]
        static_cast<FP32Tensor *>(local_logits.get())->mutable_data(), // C: [seq_len, vocab_local]
        seq_len, vocab_local, D_MODEL,
        1.0f, 0.0f);
    ASSERT_TRUE(gemm_success) << "GEMM for local LM head failed";

    // AllGather to collect full logits
    AllGatherStage::Params params;
    params.local_input = local_logits.get();
    params.full_output = full_logits.get();
    params.mpi_ctx = mpi_ctx_.get();

    auto stage = ComputeStageFactory::createAllGather(params);

    // Execute AllGather (no device context needed for MPI operations)
    bool allgather_success = stage->execute(nullptr);
    ASSERT_TRUE(allgather_success) << "AllGather failed";

    // Verify all ranks have the same full logits
    const float *full_data = static_cast<FP32Tensor *>(full_logits.get())->data();

    // Find argmax (should be same on both ranks)
    int local_argmax = 0;
    float local_max_val = full_data[0];
    for (int v = 1; v < VOCAB_SIZE; ++v)
    {
        if (full_data[v] > local_max_val)
        {
            local_max_val = full_data[v];
            local_argmax = v;
        }
    }

    // Reduce to verify all ranks computed the same argmax
    int global_argmax;
    MPI_Allreduce(&local_argmax, &global_argmax, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    // Verify consistency across ranks
    EXPECT_EQ(local_argmax, global_argmax)
        << "Argmax mismatch: rank " << rank << " got " << local_argmax
        << " but global max is " << global_argmax;

    LOG_INFO("[Test] Full forward data flow PASSED on rank " << rank
                                                             << " - argmax=" << local_argmax << " max_val=" << local_max_val);
}

// =============================================================================
// Test 6: Multi-row AllGather row-by-row interleaving (regression test)
// Validates that vocab slices are properly interleaved per row, not concatenated
// =============================================================================

TEST_F(Test__MPI_ColumnParallelLMHead, AllGatherMultiRowInterleaving)
{
    int rank = mpi_ctx_->rank();
    int world_size = mpi_ctx_->world_size();
    int vocab_local = VOCAB_SIZE / world_size;

    // Use seq_len=4 to test multi-row interleaving
    const int seq_len = 4;

    // Create local logits buffer [seq_len, vocab_local]
    auto local_logits = factory_->createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(vocab_local)}, DeviceId::cpu());

    // Create full logits buffer [seq_len, vocab_size]
    auto full_logits = factory_->createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(VOCAB_SIZE)}, DeviceId::cpu());

    // Fill local logits with rank-specific values
    // Each element encodes: (seq_idx * 10000) + (global_vocab_idx)
    // This allows us to verify both row and column placement
    float *local_data = static_cast<FP32Tensor *>(local_logits.get())->mutable_data();
    for (int s = 0; s < seq_len; ++s)
    {
        for (int v = 0; v < vocab_local; ++v)
        {
            int global_vocab_idx = rank * vocab_local + v;
            local_data[s * vocab_local + v] = static_cast<float>(s * 10000 + global_vocab_idx);
        }
    }

    // Create and execute AllGatherStage
    AllGatherStage::Params params;
    params.local_input = local_logits.get();
    params.full_output = full_logits.get();
    params.mpi_ctx = mpi_ctx_.get();

    auto stage = ComputeStageFactory::createAllGather(params);
    ASSERT_NE(stage, nullptr);

    bool success = stage->execute(nullptr);
    ASSERT_TRUE(success) << "AllGatherStage execution failed";

    // Verify the result has proper row-by-row interleaving
    // Expected layout: each row should have full vocab [0, vocab_size)
    // NOT: all rows from rank 0 followed by all rows from rank 1
    const float *full_data = static_cast<FP32Tensor *>(full_logits.get())->data();

    int errors = 0;
    for (int s = 0; s < seq_len; ++s)
    {
        for (int v = 0; v < VOCAB_SIZE; ++v)
        {
            float expected = static_cast<float>(s * 10000 + v);
            float actual = full_data[s * VOCAB_SIZE + v];
            if (std::abs(actual - expected) > 1e-5f)
            {
                if (errors < 5) // Limit error output
                {
                    LOG_ERROR("[Test] Row " << s << " vocab " << v
                                            << ": expected " << expected << " got " << actual);
                }
                errors++;
            }
        }
    }

    EXPECT_EQ(errors, 0) << "Found " << errors << " mismatched elements in row-by-row interleaving";
    LOG_INFO("[Test] AllGather multi-row interleaving PASSED on rank " << rank);
}

// =============================================================================
// Test 7: AllGather with large sequence length (stress test)
// =============================================================================

TEST_F(Test__MPI_ColumnParallelLMHead, AllGatherLargeSequence)
{
    int rank = mpi_ctx_->rank();
    int world_size = mpi_ctx_->world_size();
    int vocab_local = VOCAB_SIZE / world_size;

    // Use larger seq_len to stress the row-by-row gather
    const int seq_len = 32;

    auto local_logits = factory_->createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(vocab_local)}, DeviceId::cpu());
    auto full_logits = factory_->createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(VOCAB_SIZE)}, DeviceId::cpu());

    // Fill with deterministic pattern
    float *local_data = static_cast<FP32Tensor *>(local_logits.get())->mutable_data();
    for (int s = 0; s < seq_len; ++s)
    {
        for (int v = 0; v < vocab_local; ++v)
        {
            int global_vocab_idx = rank * vocab_local + v;
            // Use a hash-like value to ensure we can detect any swap/shuffle bugs
            local_data[s * vocab_local + v] = static_cast<float>((s * 1000003 + global_vocab_idx * 1009) % 100000);
        }
    }

    AllGatherStage::Params params;
    params.local_input = local_logits.get();
    params.full_output = full_logits.get();
    params.mpi_ctx = mpi_ctx_.get();

    auto stage = ComputeStageFactory::createAllGather(params);
    bool success = stage->execute(nullptr);
    ASSERT_TRUE(success);

    // Verify pattern
    const float *full_data = static_cast<FP32Tensor *>(full_logits.get())->data();
    int errors = 0;
    for (int s = 0; s < seq_len; ++s)
    {
        for (int v = 0; v < VOCAB_SIZE; ++v)
        {
            float expected = static_cast<float>((s * 1000003 + v * 1009) % 100000);
            float actual = full_data[s * VOCAB_SIZE + v];
            if (std::abs(actual - expected) > 1e-5f)
            {
                errors++;
            }
        }
    }

    EXPECT_EQ(errors, 0) << "Large sequence AllGather had " << errors << " errors";
    LOG_INFO("[Test] AllGather large sequence PASSED on rank " << rank);
}

// =============================================================================
// Test 8: AllGather verifies all ranks have identical output
// =============================================================================

TEST_F(Test__MPI_ColumnParallelLMHead, AllGatherOutputConsistency)
{
    int rank = mpi_ctx_->rank();
    int world_size = mpi_ctx_->world_size();
    int vocab_local = VOCAB_SIZE / world_size;
    const int seq_len = 4;

    auto local_logits = factory_->createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(vocab_local)}, DeviceId::cpu());
    auto full_logits = factory_->createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(VOCAB_SIZE)}, DeviceId::cpu());

    // Fill each rank with different data
    float *local_data = static_cast<FP32Tensor *>(local_logits.get())->mutable_data();
    for (int i = 0; i < seq_len * vocab_local; ++i)
    {
        local_data[i] = static_cast<float>(rank * 1000 + i);
    }

    AllGatherStage::Params params;
    params.local_input = local_logits.get();
    params.full_output = full_logits.get();
    params.mpi_ctx = mpi_ctx_.get();

    auto stage = ComputeStageFactory::createAllGather(params);
    bool success = stage->execute(nullptr);
    ASSERT_TRUE(success);

    // Compute checksum of output on each rank
    const float *full_data = static_cast<FP32Tensor *>(full_logits.get())->data();
    double local_sum = 0.0;
    for (size_t i = 0; i < seq_len * VOCAB_SIZE; ++i)
    {
        local_sum += full_data[i];
    }

    // Gather all checksums
    std::vector<double> all_sums(world_size);
    MPI_Allgather(&local_sum, 1, MPI_DOUBLE,
                  all_sums.data(), 1, MPI_DOUBLE, MPI_COMM_WORLD);

    // All ranks should have identical checksums
    for (int r = 1; r < world_size; ++r)
    {
        EXPECT_NEAR(all_sums[r], all_sums[0], 1e-10)
            << "Rank " << r << " checksum differs from rank 0";
    }

    LOG_INFO("[Test] AllGather output consistency PASSED on rank " << rank);
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
