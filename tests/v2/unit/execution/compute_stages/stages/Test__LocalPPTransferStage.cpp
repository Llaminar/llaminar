/**
 * @file Test__LocalPPTransferStage.cpp
 * @brief Unit tests for LocalPPTransferStage
 * @author David Sanftenberg
 * @date February 2026
 *
 * Tests for LOCAL PP transfer stage:
 * - Stage construction and properties
 * - Stage type and name
 * - Buffer requirements
 * - Parameter validation
 * - Same-device/same-stage no-op behavior
 *
 * Note: Full transfer testing requires integration tests with real devices.
 * These tests use MockLocalPPContext for isolated unit testing.
 */

#include <gtest/gtest.h>
#include <vector>

#include "execution/compute_stages/stages/LocalPPTransferStage.h"
#include "tensors/TensorFactory.h"
#include "tensors/Tensors.h"
#include "backends/GlobalDeviceAddress.h"
#include "utils/MPIContext.h"
#include "../../../../mocks/MockLocalPPContext.h"
#include "../../../../mocks/MockComputeStage.h"

using namespace llaminar2;
using namespace llaminar2::test;
using namespace llaminar2::testing;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__LocalPPTransferStage : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create test devices
        cuda0_ = GlobalDeviceAddress::cuda(0, 0);
        cuda1_ = GlobalDeviceAddress::cuda(1, 0);
        rocm0_ = GlobalDeviceAddress::rocm(0, 0);

        // Create MPI context
        mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);

        // Create device context
        ctx_ = std::make_unique<MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);

        // Create tensor factory and test tensor
        TensorFactory factory(*mpi_ctx_);
        test_tensor_ = factory.createFP32({4, 128}, DeviceId::cpu());
    }

    std::unique_ptr<MockLocalPPContext> createMockPPContext(
        const std::vector<GlobalDeviceAddress> &devices,
        const std::vector<int> &layer_boundaries = {})
    {
        MockLocalPPContext::Config config;
        config.stage_devices = devices;
        if (layer_boundaries.empty())
        {
            // Default: layers evenly distributed
            config.layer_boundaries = {0};
            for (size_t i = 0; i < devices.size(); ++i)
            {
                config.layer_boundaries.push_back(static_cast<int>((i + 1) * 12));
            }
        }
        else
        {
            config.layer_boundaries = layer_boundaries;
        }
        return std::make_unique<MockLocalPPContext>(config);
    }

    GlobalDeviceAddress cuda0_;
    GlobalDeviceAddress cuda1_;
    GlobalDeviceAddress rocm0_;
    std::shared_ptr<MPIContext> mpi_ctx_;
    std::unique_ptr<MockDeviceContext> ctx_;
    std::unique_ptr<FP32Tensor> test_tensor_;
};

// =============================================================================
// Stage Type and Properties Tests
// =============================================================================

/**
 * @test Stage type is LOCAL_PP_TRANSFER
 */
TEST_F(Test__LocalPPTransferStage, StageTypeIsLocalPPTransfer)
{
    auto pp_ctx = createMockPPContext({cuda0_, cuda1_});

    LocalPPTransferStage::Params params;
    params.pp_ctx = pp_ctx.get();
    params.tensor = test_tensor_.get();
    params.stage_from = 0;
    params.stage_to = 1;

    auto stage = std::make_unique<LocalPPTransferStage>(params);

    EXPECT_EQ(stage->type(), ComputeStageType::LOCAL_PP_TRANSFER);
}

/**
 * @test Stage name defaults to LocalPPTransfer_X_to_Y
 */
TEST_F(Test__LocalPPTransferStage, StageNameDefault)
{
    auto pp_ctx = createMockPPContext({cuda0_, cuda1_});

    LocalPPTransferStage::Params params;
    params.pp_ctx = pp_ctx.get();
    params.tensor = test_tensor_.get();
    params.stage_from = 0;
    params.stage_to = 1;

    auto stage = std::make_unique<LocalPPTransferStage>(params);

    EXPECT_EQ(stage->name(), "LocalPPTransfer_0_to_1");
}

/**
 * @test Stage name uses custom name when provided
 */
TEST_F(Test__LocalPPTransferStage, StageNameCustom)
{
    auto pp_ctx = createMockPPContext({cuda0_, cuda1_});

    LocalPPTransferStage::Params params;
    params.pp_ctx = pp_ctx.get();
    params.tensor = test_tensor_.get();
    params.stage_from = 0;
    params.stage_to = 1;
    params.stage_name = "layer12_activation_transfer";

    auto stage = std::make_unique<LocalPPTransferStage>(params);

    EXPECT_EQ(stage->name(), "layer12_activation_transfer");
}

/**
 * @test Coherence policy is NONE
 */
TEST_F(Test__LocalPPTransferStage, CoherencePolicyIsNone)
{
    auto pp_ctx = createMockPPContext({cuda0_, cuda1_});

    LocalPPTransferStage::Params params;
    params.pp_ctx = pp_ctx.get();
    params.stage_from = 0;
    params.stage_to = 1;

    auto stage = std::make_unique<LocalPPTransferStage>(params);

    EXPECT_EQ(stage->coherencePolicy(), CoherencePolicy::NONE);
}

/**
 * @test Stage supports all backends
 */
TEST_F(Test__LocalPPTransferStage, SupportsAllBackends)
{
    auto pp_ctx = createMockPPContext({cuda0_, cuda1_});

    LocalPPTransferStage::Params params;
    params.pp_ctx = pp_ctx.get();
    params.stage_from = 0;
    params.stage_to = 1;

    auto stage = std::make_unique<LocalPPTransferStage>(params);

    EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::CPU));
    EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::GPU_CUDA));
    EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::GPU_ROCM));
}

// =============================================================================
// Accessor Tests
// =============================================================================

/**
 * @test getPPContext returns correct context
 */
TEST_F(Test__LocalPPTransferStage, GetPPContextReturnsCorrect)
{
    auto pp_ctx = createMockPPContext({cuda0_, cuda1_});

    LocalPPTransferStage::Params params;
    params.pp_ctx = pp_ctx.get();
    params.stage_from = 0;
    params.stage_to = 1;

    auto stage = std::make_unique<LocalPPTransferStage>(params);

    EXPECT_EQ(stage->getPPContext(), pp_ctx.get());
    EXPECT_EQ(stage->getPPContext()->numStages(), 2);
}

/**
 * @test getTensor returns correct tensor
 */
TEST_F(Test__LocalPPTransferStage, GetTensorReturnsCorrect)
{
    auto pp_ctx = createMockPPContext({cuda0_, cuda1_});
    auto *tensor = test_tensor_.get();

    LocalPPTransferStage::Params params;
    params.pp_ctx = pp_ctx.get();
    params.tensor = tensor;
    params.stage_from = 0;
    params.stage_to = 1;

    auto stage = std::make_unique<LocalPPTransferStage>(params);

    EXPECT_EQ(stage->getTensor(), tensor);
}

/**
 * @test getStageFrom and getStageTo return correct values
 */
TEST_F(Test__LocalPPTransferStage, GetStageIndicesCorrect)
{
    auto pp_ctx = createMockPPContext({cuda0_, cuda1_, rocm0_}, {0, 8, 16, 24});

    LocalPPTransferStage::Params params;
    params.pp_ctx = pp_ctx.get();
    params.stage_from = 1;
    params.stage_to = 2;

    auto stage = std::make_unique<LocalPPTransferStage>(params);

    EXPECT_EQ(stage->getStageFrom(), 1);
    EXPECT_EQ(stage->getStageTo(), 2);
}

// =============================================================================
// Buffer Requirements Tests
// =============================================================================

/**
 * @test Buffer requirements with valid tensor
 */
TEST_F(Test__LocalPPTransferStage, BufferRequirementsWithTensor)
{
    auto pp_ctx = createMockPPContext({cuda0_, cuda1_});
    auto *tensor = test_tensor_.get();

    LocalPPTransferStage::Params params;
    params.pp_ctx = pp_ctx.get();
    params.tensor = tensor;
    params.stage_from = 0;
    params.stage_to = 1;

    auto stage = std::make_unique<LocalPPTransferStage>(params);

    auto reqs = stage->getBufferRequirements();

    // Should have input and output buffers
    ASSERT_GE(reqs.buffers.size(), 2);

    // Find input and output
    bool has_input = false, has_output = false;
    for (const auto &buf : reqs.buffers)
    {
        if (buf.role == BufferRole::INPUT)
        {
            has_input = true;
            ASSERT_GE(buf.shape.size(), 2);
            EXPECT_EQ(buf.shape[0], 4);   // rows
            EXPECT_EQ(buf.shape[1], 128); // cols
        }
        if (buf.role == BufferRole::OUTPUT)
        {
            has_output = true;
            ASSERT_GE(buf.shape.size(), 2);
            EXPECT_EQ(buf.shape[0], 4);   // rows
            EXPECT_EQ(buf.shape[1], 128); // cols
        }
    }
    EXPECT_TRUE(has_input);
    EXPECT_TRUE(has_output);
}

/**
 * @test Buffer requirements with null tensor
 */
TEST_F(Test__LocalPPTransferStage, BufferRequirementsWithNullTensor)
{
    auto pp_ctx = createMockPPContext({cuda0_, cuda1_});

    LocalPPTransferStage::Params params;
    params.pp_ctx = pp_ctx.get();
    params.tensor = nullptr;
    params.stage_from = 0;
    params.stage_to = 1;

    auto stage = std::make_unique<LocalPPTransferStage>(params);

    auto reqs = stage->getBufferRequirements();

    EXPECT_TRUE(reqs.buffers.empty());
}

// =============================================================================
// Execute Tests
// =============================================================================

/**
 * @test Execute fails with null pp_ctx
 */
TEST_F(Test__LocalPPTransferStage, ExecuteFailsWithNullContext)
{
    LocalPPTransferStage::Params params;
    params.pp_ctx = nullptr;
    params.tensor = test_tensor_.get();
    params.stage_from = 0;
    params.stage_to = 1;

    auto stage = std::make_unique<LocalPPTransferStage>(params);

    EXPECT_FALSE(stage->execute(ctx_.get()));
}

/**
 * @test Execute fails with null tensor
 */
TEST_F(Test__LocalPPTransferStage, ExecuteFailsWithNullTensor)
{
    auto pp_ctx = createMockPPContext({cuda0_, cuda1_});

    LocalPPTransferStage::Params params;
    params.pp_ctx = pp_ctx.get();
    params.tensor = nullptr;
    params.stage_from = 0;
    params.stage_to = 1;

    auto stage = std::make_unique<LocalPPTransferStage>(params);

    EXPECT_FALSE(stage->execute(ctx_.get()));
}

/**
 * @test Execute fails with invalid stage_from
 */
TEST_F(Test__LocalPPTransferStage, ExecuteFailsWithInvalidStageFrom)
{
    auto pp_ctx = createMockPPContext({cuda0_, cuda1_});

    LocalPPTransferStage::Params params;
    params.pp_ctx = pp_ctx.get();
    params.tensor = test_tensor_.get();
    params.stage_from = -1; // Invalid
    params.stage_to = 1;

    auto stage = std::make_unique<LocalPPTransferStage>(params);

    EXPECT_FALSE(stage->execute(ctx_.get()));
}

/**
 * @test Execute fails with invalid stage_to
 */
TEST_F(Test__LocalPPTransferStage, ExecuteFailsWithInvalidStageTo)
{
    auto pp_ctx = createMockPPContext({cuda0_, cuda1_});

    LocalPPTransferStage::Params params;
    params.pp_ctx = pp_ctx.get();
    params.tensor = test_tensor_.get();
    params.stage_from = 0;
    params.stage_to = 5; // Invalid (only 2 stages)

    auto stage = std::make_unique<LocalPPTransferStage>(params);

    EXPECT_FALSE(stage->execute(ctx_.get()));
}

/**
 * @test Execute succeeds for same stage (no-op)
 */
TEST_F(Test__LocalPPTransferStage, ExecuteSucceedsSameStage)
{
    auto pp_ctx = createMockPPContext({cuda0_, cuda1_});

    // Initialize tensor with known values
    auto *tensor = test_tensor_.get();
    float *data = tensor->mutable_data();
    for (size_t i = 0; i < tensor->numel(); ++i)
    {
        data[i] = static_cast<float>(i);
    }

    LocalPPTransferStage::Params params;
    params.pp_ctx = pp_ctx.get();
    params.tensor = tensor;
    params.stage_from = 0;
    params.stage_to = 0; // Same stage

    auto stage = std::make_unique<LocalPPTransferStage>(params);

    // Same stage - should be no-op and succeed
    EXPECT_TRUE(stage->execute(ctx_.get()));

    // Data should be unchanged
    for (size_t i = 0; i < tensor->numel(); ++i)
    {
        EXPECT_FLOAT_EQ(data[i], static_cast<float>(i));
    }
}

/**
 * @test Execute succeeds for different stages with mock (mock always succeeds)
 */
TEST_F(Test__LocalPPTransferStage, ExecuteSucceedsWithMock)
{
    auto pp_ctx = createMockPPContext({cuda0_, cuda1_});
    auto *tensor = test_tensor_.get();

    LocalPPTransferStage::Params params;
    params.pp_ctx = pp_ctx.get();
    params.tensor = tensor;
    params.stage_from = 0;
    params.stage_to = 1;

    auto stage = std::make_unique<LocalPPTransferStage>(params);

    // MockLocalPPContext.transfer() returns true by default
    EXPECT_TRUE(stage->execute(ctx_.get()));

    // Verify the mock recorded the transfer
    auto &mock = static_cast<MockLocalPPContext &>(*pp_ctx);
    EXPECT_EQ(mock.transferCallCount(), 1);
    EXPECT_EQ(mock.lastTransferCall().stage_from, 0);
    EXPECT_EQ(mock.lastTransferCall().stage_to, 1);
}

/**
 * @test Execute propagates failure from pp_ctx->transfer()
 */
TEST_F(Test__LocalPPTransferStage, ExecutePropagatesFailure)
{
    MockLocalPPContext::Config config;
    config.stage_devices = {cuda0_, cuda1_};
    config.layer_boundaries = {0, 12, 24};
    config.transfer_should_fail = true; // Force failure

    auto pp_ctx = std::make_unique<MockLocalPPContext>(config);
    auto *tensor = test_tensor_.get();

    LocalPPTransferStage::Params params;
    params.pp_ctx = pp_ctx.get();
    params.tensor = tensor;
    params.stage_from = 0;
    params.stage_to = 1;

    auto stage = std::make_unique<LocalPPTransferStage>(params);

    // Should fail because mock is configured to fail
    EXPECT_FALSE(stage->execute(ctx_.get()));
}

// =============================================================================
// Dump Info Tests
// =============================================================================

/**
 * @test buildDumpInfoImpl includes tensor info
 */
TEST_F(Test__LocalPPTransferStage, DumpInfoIncludesTensor)
{
    auto pp_ctx = createMockPPContext({cuda0_, cuda1_});
    auto *tensor = test_tensor_.get();

    LocalPPTransferStage::Params params;
    params.pp_ctx = pp_ctx.get();
    params.tensor = tensor;
    params.stage_from = 0;
    params.stage_to = 1;

    auto stage = std::make_unique<LocalPPTransferStage>(params);

    auto dump_info = stage->buildDumpInfoImpl();

    // Should have tensor as both input and output
    EXPECT_GE(dump_info.inputs.size(), 1);
    EXPECT_GE(dump_info.outputs.size(), 1);
}

/**
 * @test buildDumpInfoImpl includes scalars (stage indices, backend)
 */
TEST_F(Test__LocalPPTransferStage, DumpInfoIncludesScalars)
{
    auto pp_ctx = createMockPPContext({cuda0_, cuda1_});
    auto *tensor = test_tensor_.get();

    LocalPPTransferStage::Params params;
    params.pp_ctx = pp_ctx.get();
    params.tensor = tensor;
    params.stage_from = 0;
    params.stage_to = 1;

    auto stage = std::make_unique<LocalPPTransferStage>(params);

    auto dump_info = stage->buildDumpInfoImpl();

    // Should have stage_from, stage_to, num_stages, backend scalars
    EXPECT_GE(dump_info.scalars.size(), 4);
}

// =============================================================================
// setParams Tests
// =============================================================================

/**
 * @test setParams updates all fields
 */
TEST_F(Test__LocalPPTransferStage, SetParamsUpdatesAllFields)
{
    auto pp_ctx1 = createMockPPContext({cuda0_, cuda1_});
    auto pp_ctx2 = createMockPPContext({cuda0_, cuda1_, rocm0_}, {0, 8, 16, 24});

    LocalPPTransferStage::Params params;
    params.pp_ctx = pp_ctx1.get();
    params.tensor = nullptr;
    params.stage_from = 0;
    params.stage_to = 1;

    auto stage = std::make_unique<LocalPPTransferStage>(params);

    EXPECT_EQ(stage->getPPContext()->numStages(), 2);
    EXPECT_EQ(stage->getTensor(), nullptr);
    EXPECT_EQ(stage->getStageFrom(), 0);
    EXPECT_EQ(stage->getStageTo(), 1);

    // Update params
    params.pp_ctx = pp_ctx2.get();
    params.tensor = test_tensor_.get();
    params.stage_from = 1;
    params.stage_to = 2;
    stage->setParams(params);

    EXPECT_EQ(stage->getPPContext()->numStages(), 3);
    EXPECT_EQ(stage->getTensor(), test_tensor_.get());
    EXPECT_EQ(stage->getStageFrom(), 1);
    EXPECT_EQ(stage->getStageTo(), 2);
}
// =============================================================================
// REGRESSION TESTS: Multi-Stage PP Context Coverage
// =============================================================================
// These tests document and lock in fixes for bugs found during integration testing.
// Bug: When creating separate MockLocalPPContext instances for each transfer pair,
// the stage indices passed to the stage (e.g., stage_from=1, stage_to=2) may exceed
// the mock's numStages() (which was 2 when only 2 devices were configured).
// Fix: The PP context must cover ALL stages in the pipeline, not just 2 per transfer.

/**
 * @test Regression: Stage indices must not exceed PP context numStages()
 *
 * This test documents the bug where a 3-stage pipeline would create per-transfer
 * contexts with only 2 devices, causing transfer 1→2 to fail because stage_to=2
 * exceeds numStages()=2 (valid indices are 0 and 1).
 */
TEST_F(Test__LocalPPTransferStage, Regression_StageIndiceMustNotExceedNumStages)
{
    // Simulate the BUG: Create a context with only 2 stages for a transfer 1→2
    // This was the pattern that caused failures in multi-stage PP
    auto bad_pp_ctx = createMockPPContext({cuda0_, cuda1_}); // Only 2 stages!

    LocalPPTransferStage::Params params;
    params.pp_ctx = bad_pp_ctx.get();
    params.tensor = test_tensor_.get();
    params.stage_from = 1;
    params.stage_to = 2; // stage_to=2 >= numStages()=2 → INVALID

    auto stage = std::make_unique<LocalPPTransferStage>(params);

    // This should FAIL because stage_to=2 is out of range for a 2-stage context
    EXPECT_FALSE(stage->execute(ctx_.get()));
}

/**
 * @test Regression: Multi-stage PP context must cover all stages
 *
 * This test documents the FIX: For N-stage pipeline, the PP context must have
 * N devices so that stage indices 0..N-1 are all valid.
 */
TEST_F(Test__LocalPPTransferStage, Regression_MultiStagePPContextCoversAllStages)
{
    // Create a CORRECT context with 3 stages for a 3-stage pipeline
    auto good_pp_ctx = createMockPPContext({cuda0_, cuda1_, rocm0_}, {0, 8, 16, 24});

    LocalPPTransferStage::Params params;
    params.pp_ctx = good_pp_ctx.get();
    params.tensor = test_tensor_.get();

    // Transfer 0→1 should work
    params.stage_from = 0;
    params.stage_to = 1;
    auto stage01 = std::make_unique<LocalPPTransferStage>(params);
    EXPECT_TRUE(stage01->execute(ctx_.get()));

    // Transfer 1→2 should also work (stage indices are within numStages()=3)
    params.stage_from = 1;
    params.stage_to = 2;
    auto stage12 = std::make_unique<LocalPPTransferStage>(params);
    EXPECT_TRUE(stage12->execute(ctx_.get()));
}

/**
 * @test Regression: All transfer pairs in N-stage pipeline need consistent context
 *
 * In a 4-stage pipeline, transfers 0→1, 1→2, 2→3 all need access to a context
 * that knows about all 4 stages.
 */
TEST_F(Test__LocalPPTransferStage, Regression_FourStagePipelineAllTransfersValid)
{
    GlobalDeviceAddress cuda2 = GlobalDeviceAddress::cuda(2, 0);
    GlobalDeviceAddress cuda3 = GlobalDeviceAddress::cuda(3, 0);

    // Create a 4-stage context
    auto pp_ctx = createMockPPContext({cuda0_, cuda1_, cuda2, cuda3}, {0, 6, 12, 18, 24});
    ASSERT_EQ(pp_ctx->numStages(), 4);

    LocalPPTransferStage::Params params;
    params.pp_ctx = pp_ctx.get();
    params.tensor = test_tensor_.get();

    // All transfers should succeed
    for (int from = 0; from < 3; ++from)
    {
        params.stage_from = from;
        params.stage_to = from + 1;
        auto stage = std::make_unique<LocalPPTransferStage>(params);
        EXPECT_TRUE(stage->execute(ctx_.get()))
            << "Transfer " << from << "→" << (from + 1) << " failed";
    }
}