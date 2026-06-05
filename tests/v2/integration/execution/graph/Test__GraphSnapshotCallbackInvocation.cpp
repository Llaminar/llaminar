/**
 * @file Test__GraphSnapshotCallbackInvocation.cpp
 * @brief Integration tests verifying snapshot callbacks are invoked for ALL stages during graph execution
 * @author David Sanftenberg
 * @date December 2025
 *
 * These tests ensure that when a snapshot callback is set on DeviceGraphExecutor,
 * it is invoked for EVERY stage after successful execution. This is critical
 * for E2E parity testing and debugging.
 *
 * Test coverage:
 * - Basic stage types (RMSNorm, ResidualAdd)
 * - Multi-stage graphs with dependencies
 * - Edge cases (no callback, callback change)
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <mutex>
#include <algorithm>

// Core execution components
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/compute_stages/ComputeStages.h"
#include "execution/local_execution/graph/IGraphExecutor.h"
#include "execution/local_execution/device/DeviceContext.h"

// Tensors and utilities
#include "tensors/Tensors.h"
#include "utils/MPIContext.h"
#include "utils/Logger.h"

// Test utilities
#include "../../../utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// Test Fixture
// =============================================================================

/**
 * @class GraphSnapshotCallbackTest
 * @brief Test fixture for verifying snapshot callback invocation
 */
class GraphSnapshotCallbackTest : public ::testing::Test
{
protected:
    std::shared_ptr<IMPIContext> mpi_ctx_;
    std::unique_ptr<CPUDeviceContext> ctx_;
    int rank_;
    int world_size_;

    // Callback tracking
    std::mutex callback_mutex_;
    std::unordered_set<std::string> invoked_stages_;
    std::unordered_map<std::string, StageDumpInfo> captured_dumps_;

    // Tensor storage (to keep tensors alive)
    std::vector<std::unique_ptr<TensorBase>> tensor_storage_;

    void SetUp() override
    {
        int rank, world_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        rank_ = rank;
        world_size_ = world_size;
        mpi_ctx_ = std::make_shared<MPIContext>(rank, world_size, MPI_COMM_WORLD);
        ctx_ = std::make_unique<CPUDeviceContext>(DeviceId::cpu(), 4); // device 0, 4 threads

        // Clear tracking state
        invoked_stages_.clear();
        captured_dumps_.clear();
        tensor_storage_.clear();
    }

    void TearDown() override
    {
        tensor_storage_.clear();
        ctx_.reset();
        mpi_ctx_->barrier();
    }

    /**
     * @brief Create a tracking callback that records all invocations
     */
    StageSnapshotCallback createTrackingCallback()
    {
        return [this](const std::string &stage_name, const StageDumpInfo &dump_info)
        {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            invoked_stages_.insert(stage_name);
            captured_dumps_[stage_name] = dump_info;
            LOG_DEBUG("[SnapshotCallback] Invoked for stage: " << stage_name
                                                               << " outputs=" << dump_info.outputs.size());
        };
    }

    /**
     * @brief Verify all expected stages had their callbacks invoked
     */
    void verifyAllStagesInvoked(const std::vector<std::string> &expected_stages)
    {
        std::lock_guard<std::mutex> lock(callback_mutex_);

        for (const auto &stage : expected_stages)
        {
            EXPECT_TRUE(invoked_stages_.count(stage) > 0)
                << "Snapshot callback NOT invoked for stage: " << stage;
        }

        // Also check we didn't miss any stages in the graph
        EXPECT_EQ(invoked_stages_.size(), expected_stages.size())
            << "Invoked " << invoked_stages_.size() << " stages, expected " << expected_stages.size();
    }

    /**
     * @brief Create and store an FP32 tensor for testing
     */
    FP32Tensor *createFP32Tensor(const std::vector<size_t> &shape)
    {
        auto tensor = TestTensorFactory::createFP32(shape);
        auto *ptr = tensor.get();
        tensor_storage_.push_back(std::move(tensor));
        return static_cast<FP32Tensor *>(ptr);
    }

    /**
     * @brief Create and store a Q16_1 tensor for testing
     */
    Q16_1Tensor *createQ16_1Tensor(const std::vector<size_t> &shape)
    {
        auto tensor = std::make_unique<Q16_1Tensor>(shape);
        auto *ptr = tensor.get();
        tensor_storage_.push_back(std::move(tensor));
        return static_cast<Q16_1Tensor *>(ptr);
    }
};

// =============================================================================
// Basic Stage Callback Tests
// =============================================================================

/**
 * @brief Test that RMSNormStage triggers snapshot callback
 */
TEST_F(GraphSnapshotCallbackTest, RMSNormStage_CallbackInvoked)
{
    const size_t d_model = 64;
    const size_t seq_len = 4;

    // Create input/output buffers
    auto *input = createFP32Tensor({seq_len, d_model});
    auto *output = createFP32Tensor({seq_len, d_model});
    auto *gamma = createFP32Tensor({d_model});

    // Initialize input with non-zero values (required for buffer validation)
    for (size_t i = 0; i < seq_len * d_model; ++i)
        input->mutable_data()[i] = 0.5f + (i % 10) * 0.1f;

    // Initialize gamma to 1.0 for identity scaling
    for (size_t i = 0; i < d_model; ++i)
        gamma->mutable_data()[i] = 1.0f;

    // Create stage params
    RMSNormStage::Params params;
    params.input = input;
    params.output = output;
    params.gamma = gamma;
    params.eps = 1e-5f;
    params.seq_len = static_cast<int>(seq_len);

    // Build graph
    ComputeGraph graph;
    graph.addNode("test_rmsnorm", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());

    // Create executor with snapshot callback
    GraphExecutorConfig config;
    config.snapshot_callback = createTrackingCallback();

    DeviceGraphExecutor executor(config);

    // Execute with graph
    ASSERT_TRUE(executor.execute(graph, ctx_.get()));

    // Verify callback was invoked
    verifyAllStagesInvoked({"test_rmsnorm"});
}

/**
 * @brief Test that ResidualAddStage triggers snapshot callback
 */
TEST_F(GraphSnapshotCallbackTest, ResidualAddStage_CallbackInvoked)
{
    const size_t d_model = 64;
    const size_t seq_len = 4;
    const size_t num_elements = seq_len * d_model;

    auto *input = createFP32Tensor({seq_len, d_model});
    auto *residual = createFP32Tensor({seq_len, d_model});
    auto *output = createFP32Tensor({seq_len, d_model});

    // Initialize with non-zero values
    for (size_t i = 0; i < num_elements; ++i)
    {
        input->mutable_data()[i] = 0.5f;
        residual->mutable_data()[i] = 0.5f;
    }

    ResidualAddStage::Params params;
    params.input = input;
    params.residual = residual;
    params.output = output;
    params.num_elements = num_elements;

    ComputeGraph graph;
    graph.addNode("test_residual", ComputeStageFactory::createResidualAdd(params), DeviceId::cpu());

    GraphExecutorConfig config;
    config.snapshot_callback = createTrackingCallback();

    DeviceGraphExecutor executor(config);

    ASSERT_TRUE(executor.execute(graph, ctx_.get()));

    verifyAllStagesInvoked({"test_residual"});
}

// =============================================================================
// Multi-Stage Graph Callback Tests
// =============================================================================

/**
 * @brief Test that ALL stages in a multi-stage graph trigger callbacks
 */
TEST_F(GraphSnapshotCallbackTest, MultiStageGraph_AllCallbacksInvoked)
{
    const size_t d_model = 64;
    const size_t seq_len = 4;
    const size_t num_elements = seq_len * d_model;

    // Create shared buffers
    auto *input = createFP32Tensor({seq_len, d_model});
    auto *norm_out1 = createFP32Tensor({seq_len, d_model});
    auto *norm_out2 = createFP32Tensor({seq_len, d_model});
    auto *gamma = createFP32Tensor({d_model});
    auto *residual_out = createFP32Tensor({seq_len, d_model});

    // Initialize gamma to 1.0
    for (size_t i = 0; i < d_model; ++i)
        gamma->mutable_data()[i] = 1.0f;

    // Initialize other tensors with non-zero values
    for (size_t i = 0; i < num_elements; ++i)
        input->mutable_data()[i] = 0.5f;

    // Build multi-stage graph
    ComputeGraph graph;

    // Stage 1: RMSNorm
    {
        RMSNormStage::Params params;
        params.input = input;
        params.output = norm_out1;
        params.gamma = gamma;
        params.eps = 1e-5f;
        params.seq_len = static_cast<int>(seq_len);
        graph.addNode("stage1_norm", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());
    }

    // Stage 2: Second RMSNorm
    {
        RMSNormStage::Params params;
        params.input = norm_out1;
        params.output = norm_out2;
        params.gamma = gamma;
        params.eps = 1e-5f;
        params.seq_len = static_cast<int>(seq_len);
        graph.addNode("stage2_norm", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());
        graph.addDependency("stage2_norm", "stage1_norm");
    }

    // Stage 3: Residual add
    {
        ResidualAddStage::Params params;
        params.input = norm_out2;
        params.residual = input;
        params.output = residual_out;
        params.num_elements = num_elements;
        graph.addNode("stage3_residual", ComputeStageFactory::createResidualAdd(params), DeviceId::cpu());
        graph.addDependency("stage3_residual", "stage2_norm");
    }

    // Create executor with tracking callback
    GraphExecutorConfig config;
    config.snapshot_callback = createTrackingCallback();

    DeviceGraphExecutor executor(config);

    ASSERT_TRUE(executor.execute(graph, ctx_.get()));

    // Verify ALL three stages had callbacks invoked
    verifyAllStagesInvoked({"stage1_norm", "stage2_norm", "stage3_residual"});
}

// =============================================================================
// Edge Cases
// =============================================================================

/**
 * @brief Test that callback not set = no crash
 */
TEST_F(GraphSnapshotCallbackTest, NoCallback_NoFailure)
{
    const size_t d_model = 64;
    const size_t seq_len = 4;

    auto *input = createFP32Tensor({seq_len, d_model});
    auto *output = createFP32Tensor({seq_len, d_model});
    auto *gamma = createFP32Tensor({d_model});

    // Initialize input with non-zero values (required for buffer validation)
    for (size_t i = 0; i < seq_len * d_model; ++i)
        input->mutable_data()[i] = 0.5f + (i % 10) * 0.1f;

    for (size_t i = 0; i < d_model; ++i)
        gamma->mutable_data()[i] = 1.0f;

    RMSNormStage::Params params;
    params.input = input;
    params.output = output;
    params.gamma = gamma;
    params.eps = 1e-5f;
    params.seq_len = static_cast<int>(seq_len);

    ComputeGraph graph;
    graph.addNode("test_norm", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());

    // NO callback set
    GraphExecutorConfig config;
    // config.snapshot_callback is nullptr by default

    DeviceGraphExecutor executor(config);

    // Should not crash even without callback
    EXPECT_TRUE(executor.execute(graph, ctx_.get()));
}

/**
 * @brief Test that callback can be changed between executions
 */
TEST_F(GraphSnapshotCallbackTest, CallbackCanBeChanged)
{
    const size_t d_model = 64;
    const size_t seq_len = 4;

    auto *input = createFP32Tensor({seq_len, d_model});
    auto *output = createFP32Tensor({seq_len, d_model});
    auto *gamma = createFP32Tensor({d_model});

    // Initialize input with non-zero values (required for buffer validation)
    for (size_t i = 0; i < seq_len * d_model; ++i)
        input->mutable_data()[i] = 0.5f + (i % 10) * 0.1f;

    for (size_t i = 0; i < d_model; ++i)
        gamma->mutable_data()[i] = 1.0f;

    RMSNormStage::Params params;
    params.input = input;
    params.output = output;
    params.gamma = gamma;
    params.eps = 1e-5f;
    params.seq_len = static_cast<int>(seq_len);

    ComputeGraph graph;
    graph.addNode("test_norm", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());

    GraphExecutorConfig config;
    config.snapshot_callback = createTrackingCallback();

    DeviceGraphExecutor executor(config);

    // First execution
    ASSERT_TRUE(executor.execute(graph, ctx_.get()));
    EXPECT_EQ(invoked_stages_.size(), 1u);

    // Reset graph for re-execution
    graph.reset();

    // Change callback to a different one that uses a counter
    int invoke_count = 0;
    executor.setSnapshotCallback([&invoke_count](const std::string &, const StageDumpInfo &)
                                 { invoke_count++; });

    // Second execution
    ASSERT_TRUE(executor.execute(graph, ctx_.get()));
    EXPECT_EQ(invoke_count, 1);
}

/**
 * @brief Test that StageDumpInfo contains valid output data when callback is invoked
 */
TEST_F(GraphSnapshotCallbackTest, CallbackReceivesValidDumpInfo)
{
    const size_t d_model = 64;
    const size_t seq_len = 4;

    auto *input = createFP32Tensor({seq_len, d_model});
    auto *output = createFP32Tensor({seq_len, d_model});
    auto *gamma = createFP32Tensor({d_model});

    // Initialize input with non-zero values (required for buffer validation)
    for (size_t i = 0; i < seq_len * d_model; ++i)
        input->mutable_data()[i] = 0.5f + (i % 10) * 0.1f;

    for (size_t i = 0; i < d_model; ++i)
        gamma->mutable_data()[i] = 1.0f;

    RMSNormStage::Params params;
    params.input = input;
    params.output = output;
    params.gamma = gamma;
    params.eps = 1e-5f;
    params.seq_len = static_cast<int>(seq_len);

    ComputeGraph graph;
    graph.addNode("test_norm", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());

    bool callback_invoked = false;
    StageDumpInfo captured_info;

    GraphExecutorConfig config;
    config.snapshot_callback = [&](const std::string &name, const StageDumpInfo &info)
    {
        callback_invoked = true;
        captured_info = info;
    };

    DeviceGraphExecutor executor(config);
    ASSERT_TRUE(executor.execute(graph, ctx_.get()));

    EXPECT_TRUE(callback_invoked);
    // RMSNorm should have at least one output
    EXPECT_GE(captured_info.outputs.size(), 1u)
        << "RMSNormStage should report at least one output in dump info";
}

namespace
{
    class RuntimeShapeSnapshotStage final : public IComputeStage
    {
    public:
        RuntimeShapeSnapshotStage()
            : IComputeStage(DeviceId::cpu())
        {
        }

        bool execute(IDeviceContext *) override
        {
            reported_cols_ = output_.size();
            return true;
        }

        ComputeStageType type() const override { return ComputeStageType::COPY; }
        bool supportsBackend(ComputeBackendType backend) const override
        {
            return backend == ComputeBackendType::CPU;
        }

    protected:
        StageDumpInfo buildDumpInfoImpl() const override
        {
            ++build_count_;
            StageDumpInfo info;
            info.addOutput("dynamic_output", output_.data(), 1, reported_cols_);
            info.addScalarInt("build_count", build_count_);
            return info;
        }

    private:
        std::vector<float> output_{1.0f, 2.0f, 3.0f, 4.0f};
        size_t reported_cols_ = 1;
        mutable int build_count_ = 0;
    };
} // namespace

/**
 * @brief Test that callback snapshots rebuild dump info after execute.
 */
TEST_F(GraphSnapshotCallbackTest, CallbackRebuildsRuntimeDumpInfoAfterExecute)
{
    ComputeGraph graph;
    graph.addNode("runtime_shape", std::make_unique<RuntimeShapeSnapshotStage>(), DeviceId::cpu());

    size_t captured_cols = 0;
    int captured_build_count = 0;

    GraphExecutorConfig config;
    config.snapshot_callback = [&](const std::string &name, const StageDumpInfo &info)
    {
        ASSERT_EQ(name, "runtime_shape");
        ASSERT_EQ(info.outputs.size(), 1u);
        captured_cols = info.outputs[0].cols;
        for (const auto &scalar : info.scalars)
        {
            if (std::string(scalar.name) == "build_count")
                captured_build_count = static_cast<int>(scalar.value);
        }
    };

    DeviceGraphExecutor executor(config);
    ASSERT_TRUE(executor.execute(graph, ctx_.get()));

    EXPECT_EQ(captured_cols, 4u);
    EXPECT_GE(captured_build_count, 2)
        << "Snapshot callback should see post-execute dump info, not the pre-execute cache";
}

// =============================================================================
// MPI-aware main
// =============================================================================

int main(int argc, char **argv)
{
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Initialize GTest
    ::testing::InitGoogleTest(&argc, argv);

    // Run all tests
    int result = RUN_ALL_TESTS();

    // Finalize MPI
    MPI_Finalize();

    return result;
}
