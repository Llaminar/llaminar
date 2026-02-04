/**
 * @file Test__TPAllreduceStage_MPI.cpp
 * @brief MPI integration tests for TPAllreduceStage with GlobalTPContext
 * @author David Sanftenberg
 * @date February 2026
 *
 * These tests verify TPAllreduceStage works correctly with GLOBAL TP context
 * (cross-MPI-rank tensor parallelism). Requires MPI initialization with 2 ranks.
 *
 * Run with: mpirun -np 2 ./v2_integration_tp_allreduce_stage_mpi
 *
 * Test categories:
 * 1. Stage construction with GlobalTPContext
 * 2. Single-rank execute (no-op)
 * 3. Multi-rank allreduce correctness
 * 4. ITPContext polymorphism with GlobalTPContext
 * 5. Context switching (LOCAL to GLOBAL)
 *
 * Test labels: V2, Integration, GlobalTP, MPI, TensorParallel, AllReduce
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <numeric>

#include "execution/compute_stages/stages/TPAllreduceStage.h"
#include "collective/LocalTPContext.h"
#include "collective/GlobalTPContext.h"
#include "collective/ITPContext.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "tensors/TensorFactory.h"
#include "tensors/Tensors.h"
#include "backends/GlobalDeviceAddress.h"
#include "utils/MPIContext.h"
#include "../../../../mocks/MockComputeStage.h"

using namespace llaminar2;
using namespace llaminar2::testing;

// =============================================================================
// Test Fixture
// =============================================================================

/**
 * @brief Test fixture for TPAllreduceStage MPI integration tests
 *
 * Requires exactly 2 MPI ranks. Sets up per-rank tensors and contexts.
 */
class Test__TPAllreduceStage_MPI : public ::testing::Test
{
protected:
    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

        // These tests require exactly 2 ranks
        if (world_size_ != 2)
        {
            GTEST_SKIP() << "Test requires exactly 2 MPI ranks (got " << world_size_ << ")";
        }

        // Create test devices
        cuda0_ = GlobalDeviceAddress::cuda(0, 0);
        cuda1_ = GlobalDeviceAddress::cuda(1, 0);

        // Create MPI context
        mpi_ctx_ = std::make_shared<MPIContext>(world_rank_, world_size_, MPI_COMM_WORLD);

        // Create device context for CPU operations
        ctx_ = std::make_unique<MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);

        // Create tensor factory and test tensor
        TensorFactory factory(*mpi_ctx_);
        test_tensor_ = factory.createFP32({4, 128}, DeviceId::cpu());
    }

    void TearDown() override
    {
        // Synchronize all ranks before test ends
        MPI_Barrier(MPI_COMM_WORLD);
    }

    int world_rank_;
    int world_size_;
    GlobalDeviceAddress cuda0_;
    GlobalDeviceAddress cuda1_;
    std::shared_ptr<MPIContext> mpi_ctx_;
    std::unique_ptr<MockDeviceContext> ctx_;
    std::unique_ptr<FP32Tensor> test_tensor_;
};

// =============================================================================
// GlobalTPContext Construction Tests
// =============================================================================

/**
 * @test getTPContext returns correct GLOBAL context
 */
TEST_F(Test__TPAllreduceStage_MPI, GetTPContextReturnsGlobalContext)
{
    // Create a single-rank global context using MPI_COMM_SELF
    auto tp_ctx = GlobalTPContext::createForTest(MPI_COMM_SELF, 0, {world_rank_});
    ASSERT_NE(tp_ctx, nullptr);

    TPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();
    params.tensor = test_tensor_.get();

    auto stage = std::make_unique<TPAllreduceStage>(params);

    EXPECT_EQ(stage->getTPContext(), tp_ctx.get());
    EXPECT_EQ(stage->getTPContext()->degree(), 1);
    EXPECT_TRUE(stage->getTPContext()->isGlobal());
}

/**
 * @test Stage works with ITPContext* pointing to GlobalTPContext
 */
TEST_F(Test__TPAllreduceStage_MPI, WorksWithGlobalTPContextViaInterface)
{
    auto global_ctx = GlobalTPContext::createForTest(MPI_COMM_SELF, 0, {world_rank_});
    ASSERT_NE(global_ctx, nullptr);
    ITPContext* tp_ctx = global_ctx.get(); // Upcast to base interface

    TPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx;
    params.tensor = test_tensor_.get();

    auto stage = std::make_unique<TPAllreduceStage>(params);

    EXPECT_TRUE(stage->getTPContext()->isGlobal());
    // UPI backend is the default for GlobalTPContext
    EXPECT_EQ(stage->getTPContext()->backend(), CollectiveBackendType::UPI);
    
    // Single device no-op should succeed (degree = 1)
    EXPECT_TRUE(stage->execute(ctx_.get()));
}

// =============================================================================
// Single-Rank Execute Tests (no-op)
// =============================================================================

/**
 * @test Execute succeeds for single device GLOBAL TP (no-op)
 */
TEST_F(Test__TPAllreduceStage_MPI, ExecuteSucceedsSingleDeviceGlobalTP)
{
    auto tp_ctx = GlobalTPContext::createForTest(MPI_COMM_SELF, 0, {world_rank_});
    ASSERT_NE(tp_ctx, nullptr);

    // Initialize tensor with rank-specific values
    auto *tensor = test_tensor_.get();
    float *data = tensor->mutable_data();
    for (size_t i = 0; i < tensor->numel(); ++i)
    {
        data[i] = static_cast<float>(i * 2 + world_rank_);
    }

    TPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();
    params.tensor = tensor;

    auto stage = std::make_unique<TPAllreduceStage>(params);

    // Single device - should be no-op and succeed
    EXPECT_TRUE(stage->execute(ctx_.get()));

    // Data should be unchanged (single rank = no-op)
    for (size_t i = 0; i < tensor->numel(); ++i)
    {
        EXPECT_FLOAT_EQ(data[i], static_cast<float>(i * 2 + world_rank_));
    }
}

// =============================================================================
// Multi-Rank Allreduce Tests
// =============================================================================

/**
 * @test Two-rank allreduce produces correct sum
 *
 * Rank 0 contributes: [0, 1, 2, 3, ...]
 * Rank 1 contributes: [10, 11, 12, 13, ...]
 * Expected result:    [10, 12, 14, 16, ...]
 */
TEST_F(Test__TPAllreduceStage_MPI, TwoRankAllreduceSumCorrect)
{
    // Create 2-rank global context using MPI_COMM_WORLD
    auto tp_ctx = GlobalTPContext::createForTest(
        MPI_COMM_WORLD, 
        0,  // domain_id
        {0, 1}  // world_ranks
    );
    ASSERT_NE(tp_ctx, nullptr);
    EXPECT_EQ(tp_ctx->degree(), 2);

    // Initialize tensor with rank-specific values
    auto *tensor = test_tensor_.get();
    float *data = tensor->mutable_data();
    const float base = (world_rank_ == 0) ? 0.0f : 10.0f;
    for (size_t i = 0; i < tensor->numel(); ++i)
    {
        data[i] = base + static_cast<float>(i);
    }

    TPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();
    params.tensor = tensor;

    auto stage = std::make_unique<TPAllreduceStage>(params);

    // Execute allreduce
    EXPECT_TRUE(stage->execute(ctx_.get()));

    // Verify result: sum of rank 0 and rank 1 values
    // Rank 0: [0, 1, 2, ...] + Rank 1: [10, 11, 12, ...] = [10, 12, 14, ...]
    for (size_t i = 0; i < tensor->numel(); ++i)
    {
        float expected = 10.0f + 2.0f * static_cast<float>(i);
        EXPECT_FLOAT_EQ(data[i], expected) 
            << "Mismatch at index " << i << " on rank " << world_rank_;
    }
}

/**
 * @test Allreduce with larger tensor (multi-block)
 */
TEST_F(Test__TPAllreduceStage_MPI, TwoRankAllreduceLargerTensor)
{
    // Create larger tensor
    TensorFactory factory(*mpi_ctx_);
    auto large_tensor = factory.createFP32({64, 256}, DeviceId::cpu());

    // Create 2-rank global context
    auto tp_ctx = GlobalTPContext::createForTest(
        MPI_COMM_WORLD, 
        0, 
        {0, 1}
    );
    ASSERT_NE(tp_ctx, nullptr);

    // Initialize with simple rank-based values
    float *data = large_tensor->mutable_data();
    const float rank_val = static_cast<float>(world_rank_ + 1);  // 1.0 or 2.0
    for (size_t i = 0; i < large_tensor->numel(); ++i)
    {
        data[i] = rank_val;
    }

    TPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();
    params.tensor = large_tensor.get();

    auto stage = std::make_unique<TPAllreduceStage>(params);

    EXPECT_TRUE(stage->execute(ctx_.get()));

    // Verify: 1.0 + 2.0 = 3.0 everywhere
    for (size_t i = 0; i < large_tensor->numel(); ++i)
    {
        EXPECT_FLOAT_EQ(data[i], 3.0f)
            << "Mismatch at index " << i << " on rank " << world_rank_;
    }
}

// =============================================================================
// Dump Info Tests
// =============================================================================

/**
 * @test buildDumpInfoImpl includes scalars for GLOBAL TP
 */
TEST_F(Test__TPAllreduceStage_MPI, DumpInfoIncludesScalarsGlobalTP)
{
    auto tp_ctx = GlobalTPContext::createForTest(MPI_COMM_WORLD, 0, {0, 1});
    ASSERT_NE(tp_ctx, nullptr);
    auto *tensor = test_tensor_.get();

    TPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();
    params.tensor = tensor;

    auto stage = std::make_unique<TPAllreduceStage>(params);

    auto dump_info = stage->buildDumpInfoImpl();

    // Should have tp_degree, backend, and is_global scalars
    EXPECT_GE(dump_info.scalars.size(), 3);
    
    // Find is_global scalar and verify it's 1 (true)
    bool found_is_global = false;
    for (const auto& scalar : dump_info.scalars)
    {
        if (scalar.name == std::string_view("is_global"))
        {
            found_is_global = true;
            EXPECT_EQ(static_cast<int>(scalar.value), 1);
        }
    }
    EXPECT_TRUE(found_is_global);
}

// =============================================================================
// Context Switching Tests
// =============================================================================

/**
 * @test setParams can switch context type (LOCAL to GLOBAL)
 */
TEST_F(Test__TPAllreduceStage_MPI, SetParamsCanSwitchContextType)
{
    auto local_ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);
    auto global_ctx = GlobalTPContext::createForTest(MPI_COMM_SELF, 0, {world_rank_});
    ASSERT_NE(global_ctx, nullptr);

    // Start with LOCAL context
    TPAllreduceStage::Params params;
    params.tp_ctx = local_ctx.get();
    params.tensor = test_tensor_.get();

    auto stage = std::make_unique<TPAllreduceStage>(params);

    EXPECT_FALSE(stage->getTPContext()->isGlobal());

    // Switch to GLOBAL context
    params.tp_ctx = global_ctx.get();
    stage->setParams(params);

    EXPECT_TRUE(stage->getTPContext()->isGlobal());
}

// Note: main() is provided by integration/mpi_gtest_main.cpp
