/**
 * @file Test__GraphExecutorCollective.cpp
 * @brief Unit tests for GraphExecutor collective context integration
 *
 * Tests that:
 * 1. GraphExecutor accepts a CollectiveContext via setter
 * 2. When collective_ctx is set, ALLREDUCE stages are intercepted
 * 3. When collective_ctx is set, ALLGATHER stages are intercepted
 * 4. When collective_ctx is nullptr, normal stage execution happens
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "execution/GraphExecutor.h"
#include "execution/CollectiveContext.h"
#include "execution/DeviceContext.h"
#include "execution/compute_stages/stages/AllreduceStage.h"
#include "execution/compute_stages/stages/AllGatherStage.h"
#include "collective/test/CollectiveTestMocks.h"
#include "tensors/TensorClasses.h"
#include "backends/DeviceId.h"
#include "mocks/MockCollectiveContext.h"
#include "config/TPDomain.h"

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__GraphExecutorCollective : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create mock backend for testing
        mock_backend_ = std::make_unique<MockCollectiveBackend>(CollectiveBackendType::HOST);
        mock_backend_raw_ = mock_backend_.get();

        // Create mock router
        mock_router_ = std::make_unique<MockBackendRouter>();
        mock_router_->setDefaultBackend(mock_backend_raw_);

        // Create executor
        GraphExecutorConfig config;
        config.enable_profiling = true;
        executor_ = std::make_unique<GraphExecutor>(config);

        // Create device context (CPU context requires DeviceId)
        cpu_ctx_ = std::make_unique<CPUDeviceContext>(DeviceId::cpu());
    }

    // Helper to create a CollectiveContext with mock router
    std::unique_ptr<CollectiveContext> createMockCollectiveContext()
    {
        return CollectiveContextFactory::createWithRouter(
            std::move(mock_router_),
            nullptr, // No MPI
            {DeviceId::cpu()});
    }

    std::unique_ptr<MockCollectiveBackend> mock_backend_;
    MockCollectiveBackend *mock_backend_raw_ = nullptr;
    std::unique_ptr<MockBackendRouter> mock_router_;
    std::unique_ptr<GraphExecutor> executor_;
    std::unique_ptr<CPUDeviceContext> cpu_ctx_;
};

// =============================================================================
// Basic Setter/Getter Tests
// =============================================================================

TEST_F(Test__GraphExecutorCollective, SetCollectiveContext_InitiallyNull)
{
    // By default, collective context should be null
    EXPECT_EQ(executor_->collectiveContext(), nullptr);
}

TEST_F(Test__GraphExecutorCollective, SetCollectiveContext_CanSet)
{
    auto ctx = createMockCollectiveContext();
    ICollectiveContext *raw_ptr = ctx.get();

    executor_->setCollectiveContext(raw_ptr);

    EXPECT_EQ(executor_->collectiveContext(), raw_ptr);
}

TEST_F(Test__GraphExecutorCollective, SetCollectiveContext_CanClear)
{
    auto ctx = createMockCollectiveContext();
    executor_->setCollectiveContext(ctx.get());
    EXPECT_NE(executor_->collectiveContext(), nullptr);

    executor_->setCollectiveContext(nullptr);
    EXPECT_EQ(executor_->collectiveContext(), nullptr);
}

// =============================================================================
// ALLREDUCE Intercept Tests
// =============================================================================

TEST_F(Test__GraphExecutorCollective, AllreduceStage_InterceptedWhenContextSet)
{
    // Create test buffer
    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});
    float *data = buffer->mutable_data();
    for (int i = 0; i < 16; ++i)
    {
        data[i] = static_cast<float>(i);
    }

    // Create CollectiveContext
    auto collective_ctx = createMockCollectiveContext();
    executor_->setCollectiveContext(collective_ctx.get());

    // Create AllreduceStage
    AllreduceStage::Params params;
    params.buffer = buffer.get();
    params.count = 16;
    params.mpi_ctx = nullptr; // No MPI fallback

    auto stage = std::make_unique<AllreduceStage>(params);

    // Build graph with allreduce stage
    ComputeGraph graph;
    graph.addNode("allreduce", std::move(stage), DeviceId::cpu());

    // Execute graph
    bool success = executor_->execute(graph, cpu_ctx_.get());

    // Verify execution succeeded (mock backend returns true by default)
    EXPECT_TRUE(success);

    // Verify the backend was called (through router)
    EXPECT_GE(mock_backend_raw_->allreduceCallCount(), 1);
}

TEST_F(Test__GraphExecutorCollective, AllreduceStage_NotInterceptedWhenContextNull)
{
    // Create test buffer
    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    // Ensure context is null (default)
    EXPECT_EQ(executor_->collectiveContext(), nullptr);

    // Create AllreduceStage WITHOUT collective_ctx (uses MPI fallback)
    // Note: This test verifies that without collective context set on executor,
    // the stage would use its internal execution path (which may fail without MPI)
    AllreduceStage::Params params;
    params.buffer = buffer.get();
    params.count = 16;
    params.mpi_ctx = nullptr;        // No MPI
    params.collective_ctx = nullptr; // No collective context in stage either

    auto stage = std::make_unique<AllreduceStage>(params);

    // Verify backend was NOT called (since we didn't set context)
    EXPECT_EQ(mock_backend_raw_->allreduceCallCount(), 0);
}

// =============================================================================
// ALLGATHER Stage Tests
// NOTE: GraphExecutor does NOT intercept ALLGATHER stages via CollectiveContext.
// This is intentional because column-parallel operations (e.g., LM head) require
// strided placement using MPI_Type_vector, which AllGatherStage::executeViaMPI()
// handles correctly. The CollectiveContext path only supports simple contiguous
// allgather, which is insufficient for the strided output layout needed by
// column-parallel tensor parallelism.
//
// These tests verify that AllGather stages are NOT intercepted and instead
// execute their internal path (which requires MPI context).
// =============================================================================

TEST_F(Test__GraphExecutorCollective, AllgatherStage_NotInterceptedEvenWithContext)
{
    // Create test buffers (local and full)
    auto local_input = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 8});  // [seq, local_dim]
    auto full_output = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 16}); // [seq, full_dim]

    // Initialize local input
    float *data = local_input->mutable_data();
    for (size_t i = 0; i < local_input->numel(); ++i)
    {
        data[i] = static_cast<float>(i);
    }

    // Create CollectiveContext
    auto collective_ctx = createMockCollectiveContext();
    executor_->setCollectiveContext(collective_ctx.get());

    // Create AllGatherStage WITHOUT MPI context (to verify it's not intercepted)
    AllGatherStage::Params params;
    params.local_input = local_input.get();
    params.full_output = full_output.get();
    params.actual_seq_len = 4;
    params.mpi_ctx = nullptr; // No MPI - will fail if stage executes internally

    auto stage = std::make_unique<AllGatherStage>(params);

    // Build graph with allgather stage
    ComputeGraph graph;
    graph.addNode("allgather", std::move(stage), DeviceId::cpu());

    // Execute graph - should FAIL because:
    // 1. GraphExecutor does NOT intercept ALLGATHER (by design)
    // 2. AllGatherStage::execute() falls back to MPI path
    // 3. MPI context is null, so executeViaMPI() returns false
    bool success = executor_->execute(graph, cpu_ctx_.get());

    // Verify execution FAILED (AllGather not intercepted, MPI context missing)
    EXPECT_FALSE(success);

    // Verify the CollectiveContext backend was NOT called
    // (AllGather stages are not routed through CollectiveContext)
    EXPECT_EQ(mock_backend_raw_->allgatherCallCount(), 0);
}

// =============================================================================
// MockCollectiveContext Tests (using test mock)
// =============================================================================

TEST_F(Test__GraphExecutorCollective, WithMockCollectiveContext_TracksAllreduceCalls)
{
    // Create mock collective context (from test mocks)
    auto mock_ctx = MockCollectiveContext::Builder()
                        .withWorldSize(2)
                        .withRank(0)
                        .withDevice(DeviceId::cpu())
                        .build();

    executor_->setCollectiveContext(mock_ctx.get());

    // Create test buffer
    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    // Create AllreduceStage
    AllreduceStage::Params params;
    params.buffer = buffer.get();
    params.count = 16;

    auto stage = std::make_unique<AllreduceStage>(params);

    // Build and execute graph
    ComputeGraph graph;
    graph.addNode("allreduce", std::move(stage), DeviceId::cpu());
    bool success = executor_->execute(graph, cpu_ctx_.get());

    EXPECT_TRUE(success);
    EXPECT_EQ(mock_ctx->allreduce_call_count(), 1);
}

TEST_F(Test__GraphExecutorCollective, WithMockCollectiveContext_FailureHandled)
{
    // Create mock that fails allreduce
    auto mock_ctx = MockCollectiveContext::Builder()
                        .withWorldSize(2)
                        .withRank(0)
                        .withDevice(DeviceId::cpu())
                        .withAllreduceFails(true)
                        .build();

    executor_->setCollectiveContext(mock_ctx.get());

    // Create test buffer
    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    // Create AllreduceStage
    AllreduceStage::Params params;
    params.buffer = buffer.get();
    params.count = 16;

    auto stage = std::make_unique<AllreduceStage>(params);

    // Build and execute graph
    ComputeGraph graph;
    graph.addNode("allreduce", std::move(stage), DeviceId::cpu());
    bool success = executor_->execute(graph, cpu_ctx_.get());

    // Execution should fail when CollectiveContext fails
    EXPECT_FALSE(success);
}

// =============================================================================
// Mixed Graph Tests (collective + regular stages)
// =============================================================================

TEST_F(Test__GraphExecutorCollective, MixedGraph_CollectivesIntercepted_OthersNormal)
{
    // Create buffers
    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    // Create mock collective context
    auto mock_ctx = MockCollectiveContext::Builder()
                        .withWorldSize(2)
                        .withRank(0)
                        .withDevice(DeviceId::cpu())
                        .build();

    executor_->setCollectiveContext(mock_ctx.get());

    // Create AllreduceStage
    AllreduceStage::Params params;
    params.buffer = buffer.get();
    params.count = 16;

    auto allreduce_stage = std::make_unique<AllreduceStage>(params);

    // Build graph with allreduce stage
    ComputeGraph graph;
    graph.addNode("allreduce", std::move(allreduce_stage), DeviceId::cpu());

    // Execute graph
    bool success = executor_->execute(graph, cpu_ctx_.get());

    EXPECT_TRUE(success);

    // Verify collective was intercepted
    EXPECT_EQ(mock_ctx->allreduce_call_count(), 1);
}

// =============================================================================
// Profiling Tests
// =============================================================================

TEST_F(Test__GraphExecutorCollective, CollectiveStage_RecordsTimingWhenProfiling)
{
    // Create mock collective context
    auto mock_ctx = MockCollectiveContext::Builder()
                        .withWorldSize(2)
                        .withRank(0)
                        .withDevice(DeviceId::cpu())
                        .build();

    executor_->setCollectiveContext(mock_ctx.get());
    executor_->setProfilingEnabled(true);
    executor_->resetStats();

    // Create test buffer
    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    // Create AllreduceStage
    AllreduceStage::Params params;
    params.buffer = buffer.get();
    params.count = 16;

    auto stage = std::make_unique<AllreduceStage>(params);

    // Build and execute graph
    ComputeGraph graph;
    graph.addNode("test_allreduce", std::move(stage), DeviceId::cpu());
    executor_->execute(graph, cpu_ctx_.get());

    // Check that timing was recorded
    const auto &stats = executor_->stats();
    auto it = stats.stage_times_ms.find("test_allreduce");
    EXPECT_NE(it, stats.stage_times_ms.end());
    EXPECT_GE(it->second, 0.0); // Time should be non-negative
}

// =============================================================================
// Domain-Aware Collective Tests (Phase 4.1c)
// =============================================================================

TEST_F(Test__GraphExecutorCollective, AllreduceWithDomainUsesInDomainMethod)
{
    // Create mock collective context with call tracking
    auto mock_ctx = MockCollectiveContext::Builder()
                        .withWorldSize(2)
                        .withRank(0)
                        .withDevice(DeviceId::cpu())
                        .withCallTracking(true)
                        .build();

    executor_->setCollectiveContext(mock_ctx.get());

    // Create test buffer
    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    // Create a test TPDomain
    TPDomain test_domain;
    test_domain.name = "test_gpu_domain";
    test_domain.type = TPDomainType::GPU_INTRA_RANK;
    test_domain.domain_size = 2;
    test_domain.local_rank_in_domain = 0;
    test_domain.devices = {DeviceId::cpu()};
    test_domain.communicator = MPI_COMM_NULL;

    // Create AllreduceStage WITH domain
    AllreduceStage::Params params;
    params.buffer = buffer.get();
    params.count = 16;
    params.domain = &test_domain;

    auto stage = std::make_unique<AllreduceStage>(params);

    // Build and execute graph
    ComputeGraph graph;
    graph.addNode("allreduce_with_domain", std::move(stage), DeviceId::cpu());
    bool success = executor_->execute(graph, cpu_ctx_.get());

    // Verify execution succeeded
    EXPECT_TRUE(success);

    // Verify the domain-aware method was called (not the legacy method)
    EXPECT_EQ(mock_ctx->allreduce_in_domain_call_count(), 1);
    EXPECT_EQ(mock_ctx->last_allreduce_domain(), &test_domain);
}

TEST_F(Test__GraphExecutorCollective, AllreduceWithNullDomainUsesLegacyMethod)
{
    // Create mock collective context with call tracking
    auto mock_ctx = MockCollectiveContext::Builder()
                        .withWorldSize(2)
                        .withRank(0)
                        .withDevice(DeviceId::cpu())
                        .withCallTracking(true)
                        .build();

    executor_->setCollectiveContext(mock_ctx.get());

    // Create test buffer
    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    // Create AllreduceStage WITHOUT domain (nullptr)
    AllreduceStage::Params params;
    params.buffer = buffer.get();
    params.count = 16;
    params.domain = nullptr; // No domain - use legacy path

    auto stage = std::make_unique<AllreduceStage>(params);

    // Build and execute graph
    ComputeGraph graph;
    graph.addNode("allreduce_no_domain", std::move(stage), DeviceId::cpu());
    bool success = executor_->execute(graph, cpu_ctx_.get());

    // Verify execution succeeded
    EXPECT_TRUE(success);

    // Verify the legacy method was called (not the domain-aware method)
    EXPECT_EQ(mock_ctx->allreduce_call_count(), 1);
    EXPECT_EQ(mock_ctx->allreduce_in_domain_call_count(), 0);
}

TEST_F(Test__GraphExecutorCollective, AllgatherWithDomain_NotIntercepted_ExecutesViaMPI)
{
    // NOTE: GraphExecutor does NOT intercept ALLGATHER stages, even when they have
    // a domain configured. This is because column-parallel operations require
    // strided output layout via MPI_Type_vector, which CollectiveContext doesn't
    // support. The AllGatherStage handles this internally via executeViaMPI().
    //
    // This test verifies that:
    // 1. AllGather stages are NOT routed through CollectiveContext
    // 2. The stage's internal execution path is used instead
    // 3. Without MPI context, execution fails (as expected)

    // Create mock collective context with call tracking
    auto mock_ctx = MockCollectiveContext::Builder()
                        .withWorldSize(2)
                        .withRank(0)
                        .withDevice(DeviceId::cpu())
                        .withCallTracking(true)
                        .build();

    executor_->setCollectiveContext(mock_ctx.get());

    // Create test buffers
    auto local_input = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 8});
    auto full_output = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 16});

    // Create a test TPDomain
    TPDomain test_domain;
    test_domain.name = "test_gpu_domain";
    test_domain.type = TPDomainType::GPU_INTRA_RANK;
    test_domain.domain_size = 2;
    test_domain.local_rank_in_domain = 0;
    test_domain.devices = {DeviceId::cpu()};
    test_domain.communicator = MPI_COMM_NULL;

    // Create AllGatherStage WITH domain but WITHOUT MPI context
    AllGatherStage::Params params;
    params.local_input = local_input.get();
    params.full_output = full_output.get();
    params.actual_seq_len = 4;
    params.domain = &test_domain;
    params.mpi_ctx = nullptr; // No MPI context - stage will fail if not intercepted

    auto stage = std::make_unique<AllGatherStage>(params);

    // Build and execute graph
    ComputeGraph graph;
    graph.addNode("allgather_with_domain", std::move(stage), DeviceId::cpu());
    bool success = executor_->execute(graph, cpu_ctx_.get());

    // Verify execution FAILED (AllGather not intercepted, MPI context missing)
    EXPECT_FALSE(success);

    // Verify CollectiveContext methods were NOT called
    // (AllGather stages bypass CollectiveContext routing)
    EXPECT_EQ(mock_ctx->allgather_in_domain_call_count(), 0);
    EXPECT_EQ(mock_ctx->allgather_call_count(), 0);
}

TEST_F(Test__GraphExecutorCollective, AllgatherWithNullDomain_NotIntercepted_ExecutesViaMPI)
{
    // Same as above - AllGather is NOT intercepted regardless of domain config

    // Create mock collective context with call tracking
    auto mock_ctx = MockCollectiveContext::Builder()
                        .withWorldSize(2)
                        .withRank(0)
                        .withDevice(DeviceId::cpu())
                        .withCallTracking(true)
                        .build();

    executor_->setCollectiveContext(mock_ctx.get());

    // Create test buffers
    auto local_input = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 8});
    auto full_output = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 16});

    // Create AllGatherStage WITHOUT domain and WITHOUT MPI context
    AllGatherStage::Params params;
    params.local_input = local_input.get();
    params.full_output = full_output.get();
    params.actual_seq_len = 4;
    params.domain = nullptr;  // No domain
    params.mpi_ctx = nullptr; // No MPI context

    auto stage = std::make_unique<AllGatherStage>(params);

    // Build and execute graph
    ComputeGraph graph;
    graph.addNode("allgather_no_domain", std::move(stage), DeviceId::cpu());
    bool success = executor_->execute(graph, cpu_ctx_.get());

    // Verify execution FAILED (AllGather not intercepted, MPI context missing)
    EXPECT_FALSE(success);

    // Verify CollectiveContext methods were NOT called
    EXPECT_EQ(mock_ctx->allgather_call_count(), 0);
    EXPECT_EQ(mock_ctx->allgather_in_domain_call_count(), 0);
}
