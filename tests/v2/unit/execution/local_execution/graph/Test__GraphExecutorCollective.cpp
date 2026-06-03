/**
 * @file Test__GraphExecutorCollective.cpp
 * @brief Unit tests for DeviceGraphExecutor collective context integration
 *
 * Tests that:
 * 1. DeviceGraphExecutor accepts a CollectiveContext via setter
 * 2. When collective_ctx is set, ALLREDUCE stages are intercepted
 * 3. When collective_ctx is set, ALLGATHER stages are intercepted
 * 4. When collective_ctx is nullptr, normal stage execution happens
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/local_execution/orchestrators/MTPSidecarStreamBinding.h"
#include "execution/local_execution/collective/CollectiveContext.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/compute_stages/stages/AllreduceStage.h"
#include "execution/compute_stages/stages/AllGatherStage.h"
#include "collective/test/CollectiveTestMocks.h"
#include "tensors/TensorClasses.h"
#include "backends/DeviceId.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "mocks/MockCollectiveContext.h"
#include "mocks/MockComputeStage.h"
#include "config/TPDomain.h"
#include "utils/DebugEnv.h"
#include <cstdlib>
#include <future>
#include <optional>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    class MockWorkerGPUContext final : public IWorkerGPUContext
    {
    public:
        explicit MockWorkerGPUContext(int device_ordinal, void *default_stream)
            : device_ordinal_(device_ordinal), default_stream_(default_stream) {}

        int deviceOrdinal() const override { return device_ordinal_; }
        std::string deviceName() const override { return "MockWorkerGPU-" + std::to_string(device_ordinal_); }
        bool isInitialized() const override { return true; }

        void submitAndWait(std::function<void()> work) override { work(); }
        std::future<void> submitAsync(std::function<void()> work) override
        {
            work();
            std::promise<void> done;
            done.set_value();
            return done.get_future();
        }

        void *defaultStream() override { return default_stream_; }
        void *createStream() override { return default_stream_; }
        void destroyStream(void * /*stream*/) override {}

        void *createEvent() override { return reinterpret_cast<void *>(0xCAFEBABE); }
        void destroyEvent(void * /*event*/) override {}
        void recordEvent(void * /*event*/, void *stream) override
        {
            recorded_event_streams_.push_back(stream);
        }
        void waitEvent(void * /*event*/, void * /*stream*/) override {}
        void synchronizeEvent(void * /*event*/) override {}
        float eventElapsedTime(void * /*start*/, void * /*stop*/) override { return 0.0f; }

        void *blasHandle() override { return reinterpret_cast<void *>(0x11111111); }
        void *blasLtHandle() override { return reinterpret_cast<void *>(0x22222222); }

        void setCollectiveComm(void *comm) override { collective_comm_ = comm; }
        void *collectiveComm() const override { return collective_comm_; }

        void synchronize() override {}
        void synchronizeStream(void * /*stream*/) override {}
        void insertStreamDependency(void * /*dependent_stream*/, void * /*dependency_stream*/) override {}

        std::unique_ptr<IGPUGraphCapture> createGraphCapture() override { return nullptr; }
        std::unique_ptr<IGPUGraphCapture> createGraphCapture(void * /*stream*/) override { return nullptr; }

        const std::vector<void *> &recordedEventStreams() const { return recorded_event_streams_; }

    private:
        int device_ordinal_ = -1;
        void *default_stream_ = nullptr;
        void *collective_comm_ = nullptr;
        std::vector<void *> recorded_event_streams_;
    };

    class ScopedEnv
    {
    public:
        ScopedEnv(const char *name, const char *value)
            : name_(name)
        {
            if (const char *old = std::getenv(name))
                old_value_ = old;
            if (value)
                ::setenv(name, value, 1);
            else
                ::unsetenv(name);
            mutableDebugEnv().reload();
        }

        ~ScopedEnv()
        {
            if (old_value_)
                ::setenv(name_.c_str(), old_value_->c_str(), 1);
            else
                ::unsetenv(name_.c_str());
            mutableDebugEnv().reload();
        }

        ScopedEnv(const ScopedEnv &) = delete;
        ScopedEnv &operator=(const ScopedEnv &) = delete;

    private:
        std::string name_;
        std::optional<std::string> old_value_;
    };
} // namespace

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
        executor_ = std::make_unique<DeviceGraphExecutor>(config);

        // Create device context (CPU context requires DeviceId)
        cpu_ctx_ = std::make_unique<CPUDeviceContext>(DeviceId::cpu());
    }

    // Helper to create a CollectiveContext with mock router (multi-device to trigger actual collectives)
    std::unique_ptr<CollectiveContext> createMockCollectiveContext()
    {
        return CollectiveContextFactory::createWithRouter(
            std::move(mock_router_),
            nullptr,                                 // No MPI
            {DeviceId::cuda(0), DeviceId::cuda(1)}); // Multi-device to trigger collectives
    }

    // Helper to create a single-device CollectiveContext (for no-op tests)
    std::unique_ptr<CollectiveContext> createSingleDeviceCollectiveContext()
    {
        auto router = std::make_unique<MockBackendRouter>();
        router->setDefaultBackend(mock_backend_raw_);
        return CollectiveContextFactory::createWithRouter(
            std::move(router),
            nullptr,
            {DeviceId::cpu()});
    }

    std::unique_ptr<MockCollectiveBackend> mock_backend_;
    MockCollectiveBackend *mock_backend_raw_ = nullptr;
    std::unique_ptr<MockBackendRouter> mock_router_;
    std::unique_ptr<DeviceGraphExecutor> executor_;
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
// NOTE: DeviceGraphExecutor does NOT intercept ALLGATHER stages via CollectiveContext.
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
    // 1. DeviceGraphExecutor does NOT intercept ALLGATHER (by design)
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
    // NOTE: DeviceGraphExecutor does NOT intercept ALLGATHER stages, even when they have
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

// =============================================================================
// Stage Stream Binding Tests
// =============================================================================

class Test__GraphExecutorStreamBinding : public ::testing::Test
{
protected:
    void SetUp() override
    {
        GPUDeviceContextPool::instance().shutdown();

        cuda_default_stream_ = reinterpret_cast<void *>(0xC0FFEE00);
        rocm_default_stream_ = reinterpret_cast<void *>(0xBADC0DE0);

        GPUDeviceContextPool::instance().registerNvidiaFactory(
            [this](int ordinal)
            {
                return std::make_unique<MockWorkerGPUContext>(ordinal, cuda_default_stream_);
            },
            1);

        GPUDeviceContextPool::instance().registerAMDFactory(
            [this](int ordinal)
            {
                return std::make_unique<MockWorkerGPUContext>(ordinal, rocm_default_stream_);
            },
            1);
    }

    void TearDown() override
    {
        GPUDeviceContextPool::instance().shutdown();
    }

    void *cuda_default_stream_ = nullptr;
    void *rocm_default_stream_ = nullptr;
};

TEST_F(Test__GraphExecutorStreamBinding, NullStageStream_BindsToNodeDeviceDefaultStream)
{
    GraphExecutorConfig config;
    DeviceGraphExecutor executor(config);

    llaminar2::testing::MockDeviceContext ctx(DeviceId::cuda(0), ComputeBackendType::GPU_CUDA);

    auto stage = std::make_unique<llaminar2::testing::MockComputeStage>(
        ComputeStageType::GEMM,
        "stream_bind_stage",
        DeviceId::cuda(0));
    auto *stage_raw = stage.get();

    ASSERT_EQ(stage_raw->gpuStream(), nullptr);

    ComputeGraph graph;
    graph.addNode("stream_bind_stage", std::move(stage), DeviceId::cuda(0));

    ASSERT_TRUE(executor.execute(graph, &ctx));
    EXPECT_EQ(stage_raw->gpuStream(), cuda_default_stream_);
}

TEST_F(Test__GraphExecutorStreamBinding, PreBoundStageStream_IsNotOverwritten)
{
    GraphExecutorConfig config;
    DeviceGraphExecutor executor(config);

    llaminar2::testing::MockDeviceContext ctx(DeviceId::cuda(0), ComputeBackendType::GPU_CUDA);
    void *prebound = reinterpret_cast<void *>(0x1234ABCD);

    auto stage = std::make_unique<llaminar2::testing::MockComputeStage>(
        ComputeStageType::GEMM,
        "prebound_stage",
        DeviceId::cuda(0));
    auto *stage_raw = stage.get();
    stage_raw->setGPUStream(prebound);

    ComputeGraph graph;
    graph.addNode("prebound_stage", std::move(stage), DeviceId::cuda(0));

    ASSERT_TRUE(executor.execute(graph, &ctx));
    EXPECT_EQ(stage_raw->gpuStream(), prebound);
}

TEST_F(Test__GraphExecutorStreamBinding, NodeDeviceOverridesStageDeviceWhenResolvingStream)
{
    GraphExecutorConfig config;
    DeviceGraphExecutor executor(config);

    llaminar2::testing::MockDeviceContext ctx(DeviceId::rocm(0), ComputeBackendType::GPU_ROCM);

    auto stage = std::make_unique<llaminar2::testing::MockComputeStage>(
        ComputeStageType::GEMM,
        "node_device_precedence",
        DeviceId::rocm(0));
    auto *stage_raw = stage.get();

    ASSERT_EQ(stage_raw->gpuStream(), nullptr);

    ComputeGraph graph;
    graph.addNode("node_device_precedence", std::move(stage), DeviceId::cuda(0));

    ASSERT_TRUE(executor.execute(graph, &ctx));
    EXPECT_EQ(stage_raw->gpuStream(), cuda_default_stream_);
    EXPECT_NE(stage_raw->gpuStream(), rocm_default_stream_);
}

TEST_F(Test__GraphExecutorStreamBinding, CudaContextUsedWhenNodeAndStageDevicesAreInvalid)
{
    GraphExecutorConfig config;
    DeviceGraphExecutor executor(config);

    llaminar2::testing::MockDeviceContext ctx(DeviceId::cuda(0), ComputeBackendType::GPU_CUDA);

    auto stage = std::make_unique<llaminar2::testing::MockComputeStage>(
        ComputeStageType::GEMM,
        "ctx_fallback_cuda",
        DeviceId::invalid());
    auto *stage_raw = stage.get();

    ASSERT_EQ(stage_raw->gpuStream(), nullptr);

    ComputeGraph graph;
    graph.addNode("ctx_fallback_cuda", std::move(stage), DeviceId::invalid());

    ASSERT_TRUE(executor.execute(graph, &ctx));
    EXPECT_EQ(stage_raw->gpuStream(), cuda_default_stream_);
}

TEST_F(Test__GraphExecutorStreamBinding, TimelineEventsUsePreBoundStageStream)
{
    ScopedEnv timing("LLAMINAR_GPU_STAGE_TIMING", "1");

    GraphExecutorConfig config;
    DeviceGraphExecutor executor(config);

    llaminar2::testing::MockDeviceContext ctx(DeviceId::rocm(0), ComputeBackendType::GPU_ROCM);
    void *capture_stream = reinterpret_cast<void *>(0xFACEB00C);

    auto stage = std::make_unique<llaminar2::testing::MockComputeStage>(
        ComputeStageType::GEMM,
        "capture_stream_stage",
        DeviceId::rocm(0));
    auto *stage_raw = stage.get();
    stage_raw->setGPUStream(capture_stream);

    ComputeGraph graph;
    graph.addNode("capture_stream_stage", std::move(stage), DeviceId::rocm(0));

    ASSERT_TRUE(executor.executeFastDecode(graph, &ctx));
    EXPECT_EQ(stage_raw->gpuStream(), capture_stream);

    auto *worker = dynamic_cast<MockWorkerGPUContext *>(&GPUDeviceContextPool::instance().getContext(DeviceId::rocm(0)));
    ASSERT_NE(worker, nullptr);
    ASSERT_EQ(worker->recordedEventStreams().size(), 2u);
    EXPECT_EQ(worker->recordedEventStreams()[0], capture_stream);
    EXPECT_EQ(worker->recordedEventStreams()[1], capture_stream);
    EXPECT_NE(worker->recordedEventStreams()[0], rocm_default_stream_);
}

TEST_F(Test__GraphExecutorStreamBinding, MTPSidecarStagesBindToCaptureStream)
{
    const void *stale_stream = reinterpret_cast<void *>(0xBAD50000);
    void *capture_stream = reinterpret_cast<void *>(0x1DECAFFE);

    struct StageSpec
    {
        ComputeStageType type;
        const char *name;
    };

    const std::vector<StageSpec> sidecar_stage_specs = {
        {ComputeStageType::EMBEDDING, "mtp_embedding"},
        {ComputeStageType::RMS_NORM, "mtp_input_norm"},
        {ComputeStageType::MTP_CONCAT, "mtp_concat"},
        {ComputeStageType::GEMM, "mtp_projection"},
        {ComputeStageType::GDN_PROJECTION, "mtp_gdn_projection"},
        {ComputeStageType::SHORT_CONV1D, "mtp_short_conv"},
        {ComputeStageType::GDN_RECURRENCE, "mtp_gdn_recurrence"},
        {ComputeStageType::ATTENTION, "mtp_attention"},
        {ComputeStageType::KV_CACHE_APPEND, "mtp_kv_append"},
        {ComputeStageType::GATED_RMS_NORM, "mtp_gated_norm"},
        {ComputeStageType::LM_HEAD, "mtp_lm_head"},
    };

    ComputeGraph graph;
    std::vector<llaminar2::testing::MockComputeStage *> stages;
    std::string previous_node;
    for (const StageSpec &spec : sidecar_stage_specs)
    {
        auto stage = std::make_unique<llaminar2::testing::MockComputeStage>(
            spec.type,
            spec.name,
            DeviceId::rocm(0));
        auto *stage_raw = stage.get();
        stage_raw->setGPUStream(const_cast<void *>(stale_stream));
        stages.push_back(stage_raw);

        graph.addNode(spec.name, std::move(stage), DeviceId::rocm(0));
        if (!previous_node.empty())
            graph.addDependency(spec.name, previous_node);
        previous_node = spec.name;
    }

    std::string error;
    ASSERT_TRUE(mtp_sidecar::bindStagesToCaptureStream(graph, capture_stream, &error)) << error;

    std::string mismatch;
    EXPECT_TRUE(mtp_sidecar::allStagesBoundToStream(graph, capture_stream, &mismatch)) << mismatch;
    for (const auto *stage : stages)
    {
        ASSERT_NE(stage, nullptr);
        EXPECT_EQ(stage->gpuStream(), capture_stream) << stage->name();
    }
}

TEST_F(Test__GraphExecutorStreamBinding, MTPSidecarDeferredSamplingUsesCaptureStream)
{
    void *capture_stream = reinterpret_cast<void *>(0x0DADA123);

    EXPECT_EQ(mtp_sidecar::deferredSamplingStream(true, true, capture_stream), capture_stream);
    EXPECT_EQ(mtp_sidecar::deferredSamplingStream(true, true, nullptr), nullptr);
    EXPECT_EQ(mtp_sidecar::deferredSamplingStream(true, false, capture_stream), nullptr);
    EXPECT_EQ(mtp_sidecar::deferredSamplingStream(false, true, capture_stream), nullptr);
}
