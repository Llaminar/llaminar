/**
 * @file Test__LocalTPAllreduceStage.cpp
 * @brief Unit tests for LocalTPAllreduceStage
 * @author David Sanftenberg
 * @date January 2026
 *
 * Tests for LOCAL TP all-reduce stage:
 * - Stage construction and properties
 * - Stage type and name
 * - Buffer requirements
 * - Parameter validation
 * - Single-device no-op behavior
 *
 * Note: Full collective testing requires integration tests with real devices.
 */

#include <gtest/gtest.h>
#include <vector>

#include "execution/compute_stages/stages/LocalTPAllreduceStage.h"
#include "collective/LocalTPContext.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "tensors/TensorFactory.h"
#include "tensors/Tensors.h"
#include "backends/GlobalDeviceAddress.h"
#include "utils/MPIContext.h"
#include "../mocks/MockComputeStage.h"

using namespace llaminar2;
using namespace llaminar2::testing;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__LocalTPAllreduceStage : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create test devices
        cuda0_ = GlobalDeviceAddress::cuda(0, 0);
        cuda1_ = GlobalDeviceAddress::cuda(1, 0);

        // Create MPI context
        mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);

        // Create device context
        ctx_ = std::make_unique<MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);

        // Create tensor factory
        TensorFactory factory(*mpi_ctx_);
        test_tensor_ = factory.createFP32({4, 128}, DeviceId::cpu());
    }

    GlobalDeviceAddress cuda0_;
    GlobalDeviceAddress cuda1_;
    std::shared_ptr<MPIContext> mpi_ctx_;
    std::unique_ptr<MockDeviceContext> ctx_;
    std::unique_ptr<FP32Tensor> test_tensor_;
};

// =============================================================================
// Stage Type and Properties Tests
// =============================================================================

/**
 * @test Stage type is ALLREDUCE
 */
TEST_F(Test__LocalTPAllreduceStage, StageTypeIsAllreduce)
{
    auto tp_ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);

    LocalTPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();
    params.tensor = test_tensor_.get();

    auto stage = std::make_unique<LocalTPAllreduceStage>(params);

    EXPECT_EQ(stage->type(), ComputeStageType::ALLREDUCE);
}

/**
 * @test Stage name is LocalTPAllreduce
 */
TEST_F(Test__LocalTPAllreduceStage, StageNameCorrect)
{
    auto tp_ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);

    LocalTPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();
    params.tensor = test_tensor_.get();

    auto stage = std::make_unique<LocalTPAllreduceStage>(params);

    EXPECT_EQ(stage->name(), "LocalTPAllreduce");
}

/**
 * @test Stage requires allreduce
 */
TEST_F(Test__LocalTPAllreduceStage, RequiresAllreduce)
{
    auto tp_ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);

    LocalTPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();

    auto stage = std::make_unique<LocalTPAllreduceStage>(params);

    EXPECT_TRUE(stage->requiresAllreduce());
}

/**
 * @test Coherence policy is NONE
 */
TEST_F(Test__LocalTPAllreduceStage, CoherencePolicyIsNone)
{
    auto tp_ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);

    LocalTPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();

    auto stage = std::make_unique<LocalTPAllreduceStage>(params);

    EXPECT_EQ(stage->coherencePolicy(), CoherencePolicy::NONE);
}

/**
 * @test Stage supports all backends
 */
TEST_F(Test__LocalTPAllreduceStage, SupportsAllBackends)
{
    auto tp_ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);

    LocalTPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();

    auto stage = std::make_unique<LocalTPAllreduceStage>(params);

    EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::CPU));
    EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::GPU_CUDA));
    EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::GPU_ROCM));
}

// =============================================================================
// Accessor Tests
// =============================================================================

/**
 * @test getTPContext returns correct context
 */
TEST_F(Test__LocalTPAllreduceStage, GetTPContextReturnsCorrect)
{
    auto tp_ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::NCCL);

    LocalTPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();

    auto stage = std::make_unique<LocalTPAllreduceStage>(params);

    EXPECT_EQ(stage->getTPContext(), tp_ctx.get());
    EXPECT_EQ(stage->getTPContext()->degree(), 2);
}

/**
 * @test getTensor returns correct tensor
 */
TEST_F(Test__LocalTPAllreduceStage, GetTensorReturnsCorrect)
{
    auto tp_ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);
    auto *tensor = test_tensor_.get();

    LocalTPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();
    params.tensor = tensor;

    auto stage = std::make_unique<LocalTPAllreduceStage>(params);

    EXPECT_EQ(stage->getTensor(), tensor);
}

// =============================================================================
// Buffer Requirements Tests
// =============================================================================

/**
 * @test Buffer requirements with valid tensor
 */
TEST_F(Test__LocalTPAllreduceStage, BufferRequirementsWithTensor)
{
    auto tp_ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);
    auto *tensor = test_tensor_.get();

    LocalTPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();
    params.tensor = tensor;

    auto stage = std::make_unique<LocalTPAllreduceStage>(params);

    auto reqs = stage->getBufferRequirements();

    ASSERT_GE(reqs.buffers.size(), 1);
    EXPECT_EQ(reqs.buffers[0].role, BufferRole::INOUT);
    ASSERT_GE(reqs.buffers[0].shape.size(), 2);
    EXPECT_EQ(reqs.buffers[0].shape[0], 4);   // rows
    EXPECT_EQ(reqs.buffers[0].shape[1], 128); // cols
}

/**
 * @test Buffer requirements with null tensor
 */
TEST_F(Test__LocalTPAllreduceStage, BufferRequirementsWithNullTensor)
{
    auto tp_ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);

    LocalTPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();
    params.tensor = nullptr;

    auto stage = std::make_unique<LocalTPAllreduceStage>(params);

    auto reqs = stage->getBufferRequirements();

    EXPECT_TRUE(reqs.buffers.empty());
}

// =============================================================================
// Execute Tests
// =============================================================================

/**
 * @test Execute fails with null tp_ctx
 */
TEST_F(Test__LocalTPAllreduceStage, ExecuteFailsWithNullContext)
{
    LocalTPAllreduceStage::Params params;
    params.tp_ctx = nullptr;
    params.tensor = test_tensor_.get();

    auto stage = std::make_unique<LocalTPAllreduceStage>(params);

    EXPECT_FALSE(stage->execute(ctx_.get()));
}

/**
 * @test Execute fails with null tensor
 */
TEST_F(Test__LocalTPAllreduceStage, ExecuteFailsWithNullTensor)
{
    auto tp_ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::NCCL);

    LocalTPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();
    params.tensor = nullptr;

    auto stage = std::make_unique<LocalTPAllreduceStage>(params);

    EXPECT_FALSE(stage->execute(ctx_.get()));
}

/**
 * @test Execute succeeds for single device (no-op)
 */
TEST_F(Test__LocalTPAllreduceStage, ExecuteSucceedsSingleDevice)
{
    auto tp_ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);

    // Initialize tensor with known values
    auto *tensor = test_tensor_.get();
    float *data = tensor->mutable_data();
    for (size_t i = 0; i < tensor->numel(); ++i)
    {
        data[i] = static_cast<float>(i);
    }

    LocalTPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();
    params.tensor = tensor;

    auto stage = std::make_unique<LocalTPAllreduceStage>(params);

    // Single device - should be no-op and succeed
    EXPECT_TRUE(stage->execute(ctx_.get()));

    // Data should be unchanged
    for (size_t i = 0; i < tensor->numel(); ++i)
    {
        EXPECT_FLOAT_EQ(data[i], static_cast<float>(i));
    }
}

/**
 * @test Execute with multi-device context fails when tensor not on any TP device
 *
 * The tensor is on CPU but the LocalTPContext has CUDA devices, so the
 * allreduce correctly fails because the tensor's device isn't in the TP context.
 * Full collective testing with proper device placement requires integration tests.
 */
TEST_F(Test__LocalTPAllreduceStage, ExecuteMultiDeviceFailsWhenTensorNotOnTPDevice)
{
    auto tp_ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::NCCL);
    auto *tensor = test_tensor_.get(); // tensor is on CPU, not cuda0_ or cuda1_

    LocalTPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();
    params.tensor = tensor;

    auto stage = std::make_unique<LocalTPAllreduceStage>(params);

    // Should fail: tensor device (CPU) not found in LocalTPContext devices (cuda0_, cuda1_)
    EXPECT_FALSE(stage->execute(ctx_.get()));
}

// =============================================================================
// Dump Info Tests
// =============================================================================

/**
 * @test buildDumpInfoImpl includes tensor info
 */
TEST_F(Test__LocalTPAllreduceStage, DumpInfoIncludesTensor)
{
    auto tp_ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::NCCL);
    auto *tensor = test_tensor_.get();

    LocalTPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();
    params.tensor = tensor;

    auto stage = std::make_unique<LocalTPAllreduceStage>(params);

    auto dump_info = stage->buildDumpInfoImpl();

    // Should have tensor as both input and output (in-place)
    EXPECT_GE(dump_info.inputs.size(), 1);
    EXPECT_GE(dump_info.outputs.size(), 1);
}

/**
 * @test buildDumpInfoImpl includes scalars
 */
TEST_F(Test__LocalTPAllreduceStage, DumpInfoIncludesScalars)
{
    auto tp_ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::NCCL);
    auto *tensor = test_tensor_.get();

    LocalTPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();
    params.tensor = tensor;

    auto stage = std::make_unique<LocalTPAllreduceStage>(params);

    auto dump_info = stage->buildDumpInfoImpl();

    // Should have tp_degree and backend scalars
    EXPECT_GE(dump_info.scalars.size(), 2);
}

// =============================================================================
// setParams Tests
// =============================================================================

/**
 * @test setParams updates tensor
 */
TEST_F(Test__LocalTPAllreduceStage, SetParamsUpdatesTensor)
{
    auto tp_ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);

    LocalTPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();
    params.tensor = nullptr;

    auto stage = std::make_unique<LocalTPAllreduceStage>(params);

    EXPECT_EQ(stage->getTensor(), nullptr);

    // Update params with a tensor
    auto *tensor = test_tensor_.get();
    params.tensor = tensor;
    stage->setParams(params);

    EXPECT_EQ(stage->getTensor(), tensor);
}
