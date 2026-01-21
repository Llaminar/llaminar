/**
 * @file Test__PPActivationStages.cpp
 * @brief Unit tests for Pipeline Parallel (PP) activation stages
 *
 * Phase 6.4: Tests SendActivationsStage and ReceiveActivationsStage
 * compute stage interfaces without requiring actual MPI communication.
 *
 * Tests verify:
 * - Stage type and name
 * - Buffer requirements
 * - Dump info serialization
 * - Parameter validation
 * - Coherence policy
 * - Async support interface
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "execution/compute_stages/ComputeStageFactory.h"
#include "execution/compute_stages/stages/SendActivationsStage.h"
#include "execution/compute_stages/stages/ReceiveActivationsStage.h"
#include "tensors/Tensors.h"
#include "tensors/TensorFactory.h"
#include "backends/DeviceId.h"
#include "utils/MPIContext.h"

using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__PPActivationStages : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create a mock MPI context (rank=0, world_size=2, null comm for unit testing)
        mpi_ctx_ = std::make_shared<MPIContext>(0, 2, MPI_COMM_NULL);
    }

    std::shared_ptr<MPIContext> mpi_ctx_;
};

// =============================================================================
// SendActivationsStage Unit Tests
// =============================================================================

/**
 * @test Verify SendActivationsStage type returns SEND_ACTIVATIONS
 */
TEST_F(Test__PPActivationStages, SendStage_TypeIsSendActivations)
{
    TensorFactory factory(*mpi_ctx_);
    auto buffer = factory.createFP32({4, 128}, DeviceId::cpu());

    SendActivationsStage::Params params{
        .device_id = DeviceId::cpu(),
        .mpi_ctx = mpi_ctx_.get(),
        .buffer = buffer.get(),
        .dest_rank = 1,
        .tag = 42,
        .async = false,
    };

    auto stage = ComputeStageFactory::createSendActivations(params);

    EXPECT_EQ(stage->type(), ComputeStageType::SEND_ACTIVATIONS);
}

/**
 * @test Verify SendActivationsStage name generation with dest_rank and tag
 */
TEST_F(Test__PPActivationStages, SendStage_GeneratesCorrectName)
{
    TensorFactory factory(*mpi_ctx_);
    auto buffer = factory.createFP32({4, 128}, DeviceId::cpu());

    SendActivationsStage::Params params{
        .device_id = DeviceId::cpu(),
        .mpi_ctx = mpi_ctx_.get(),
        .buffer = buffer.get(),
        .dest_rank = 3,
        .tag = 100,
        .async = false,
    };

    auto stage = ComputeStageFactory::createSendActivations(params);

    std::string expected_name = "send_activations_to_rank3_tag100";
    EXPECT_EQ(stage->name(), expected_name);
}

/**
 * @test Verify SendActivationsStage uses custom name when provided
 */
TEST_F(Test__PPActivationStages, SendStage_UsesCustomName)
{
    TensorFactory factory(*mpi_ctx_);
    auto buffer = factory.createFP32({4, 128}, DeviceId::cpu());

    SendActivationsStage::Params params{
        .device_id = DeviceId::cpu(),
        .mpi_ctx = mpi_ctx_.get(),
        .buffer = buffer.get(),
        .dest_rank = 1,
        .tag = 42,
        .async = false,
        .stage_name = "layer5_forward_send",
    };

    auto stage = ComputeStageFactory::createSendActivations(params);

    EXPECT_EQ(stage->name(), "layer5_forward_send");
}

/**
 * @test Verify SendActivationsStage supports all backends
 */
TEST_F(Test__PPActivationStages, SendStage_SupportsAllBackends)
{
    TensorFactory factory(*mpi_ctx_);
    auto buffer = factory.createFP32({4, 128}, DeviceId::cpu());

    SendActivationsStage::Params params{
        .device_id = DeviceId::cpu(),
        .mpi_ctx = mpi_ctx_.get(),
        .buffer = buffer.get(),
        .dest_rank = 1,
        .tag = 0,
        .async = false,
    };

    auto stage = ComputeStageFactory::createSendActivations(params);

    // MPI stages are backend-agnostic (work with any device that can provide host data)
    EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::CPU));
    EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::GPU_CUDA));
    EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::GPU_ROCM));
    EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::GPU_VULKAN));
}

/**
 * @test Verify SendActivationsStage coherence policy is NONE
 */
TEST_F(Test__PPActivationStages, SendStage_CoherencePolicyIsNone)
{
    SendActivationsStage::Params params{
        .device_id = DeviceId::cpu(),
        .mpi_ctx = mpi_ctx_.get(),
    };

    auto stage = ComputeStageFactory::createSendActivations(params);

    // MPI stages handle their own synchronization
    EXPECT_EQ(stage->coherencePolicy(), CoherencePolicy::NONE);
}

/**
 * @test Verify SendActivationsStage fails with null buffer
 */
TEST_F(Test__PPActivationStages, SendStage_FailsWithNullBuffer)
{
    SendActivationsStage::Params params{
        .device_id = DeviceId::cpu(),
        .mpi_ctx = mpi_ctx_.get(),
        .buffer = nullptr,
        .dest_rank = 1,
        .tag = 0,
        .async = false,
    };

    auto stage = ComputeStageFactory::createSendActivations(params);

    EXPECT_FALSE(stage->execute(nullptr));
}

/**
 * @test Verify SendActivationsStage fails with null MPI context
 */
TEST_F(Test__PPActivationStages, SendStage_FailsWithNullMpiContext)
{
    TensorFactory factory(*mpi_ctx_);
    auto buffer = factory.createFP32({4, 128}, DeviceId::cpu());

    SendActivationsStage::Params params{
        .device_id = DeviceId::cpu(),
        .mpi_ctx = nullptr,
        .buffer = buffer.get(),
        .dest_rank = 1,
        .tag = 0,
        .async = false,
    };

    auto stage = ComputeStageFactory::createSendActivations(params);

    EXPECT_FALSE(stage->execute(nullptr));
}

/**
 * @test Verify SendActivationsStage fails with invalid destination rank
 */
TEST_F(Test__PPActivationStages, SendStage_FailsWithInvalidDestRank)
{
    TensorFactory factory(*mpi_ctx_);
    auto buffer = factory.createFP32({4, 128}, DeviceId::cpu());

    SendActivationsStage::Params params{
        .device_id = DeviceId::cpu(),
        .mpi_ctx = mpi_ctx_.get(),
        .buffer = buffer.get(),
        .dest_rank = -1, // Invalid
        .tag = 0,
        .async = false,
    };

    auto stage = ComputeStageFactory::createSendActivations(params);

    EXPECT_FALSE(stage->execute(nullptr));
}

/**
 * @test Verify SendActivationsStage buffer requirements
 */
TEST_F(Test__PPActivationStages, SendStage_BufferRequirements)
{
    TensorFactory factory(*mpi_ctx_);
    auto buffer = factory.createFP32({4, 128}, DeviceId::cpu());

    SendActivationsStage::Params params{
        .device_id = DeviceId::cpu(),
        .mpi_ctx = mpi_ctx_.get(),
        .buffer = buffer.get(),
        .dest_rank = 1,
        .tag = 0,
        .async = false,
    };

    auto stage = ComputeStageFactory::createSendActivations(params);
    auto reqs = stage->getBufferRequirements();

    // Send stage reads from one input buffer
    EXPECT_GE(reqs.buffers.size(), 1);
}

/**
 * @test Verify SendActivationsStage dump info contains expected fields
 */
TEST_F(Test__PPActivationStages, SendStage_DumpInfoIsComplete)
{
    TensorFactory factory(*mpi_ctx_);
    auto buffer = factory.createFP32({8, 64}, DeviceId::cpu());

    SendActivationsStage::Params params{
        .device_id = DeviceId::cpu(),
        .mpi_ctx = mpi_ctx_.get(),
        .buffer = buffer.get(),
        .dest_rank = 2,
        .tag = 77,
        .async = true,
    };

    auto stage = ComputeStageFactory::createSendActivations(params);
    auto dump_info = stage->getDumpInfo();

    // Verify inputs
    EXPECT_EQ(dump_info.inputs.size(), 1);
    EXPECT_STREQ(dump_info.inputs[0].name, "buffer");
    EXPECT_EQ(dump_info.inputs[0].rows, 8);
    EXPECT_EQ(dump_info.inputs[0].cols, 64);

    // Verify scalars
    EXPECT_GE(dump_info.scalars.size(), 3); // dest_rank, tag, async

    // Find specific scalars
    bool found_dest_rank = false;
    bool found_tag = false;
    bool found_async = false;

    for (const auto &scalar : dump_info.scalars)
    {
        if (std::string(scalar.name) == "dest_rank")
        {
            EXPECT_EQ(static_cast<int>(scalar.value), 2);
            found_dest_rank = true;
        }
        else if (std::string(scalar.name) == "tag")
        {
            EXPECT_EQ(static_cast<int>(scalar.value), 77);
            found_tag = true;
        }
        else if (std::string(scalar.name) == "async")
        {
            EXPECT_EQ(scalar.value, 1.0); // true = 1.0
            found_async = true;
        }
    }

    EXPECT_TRUE(found_dest_rank) << "Missing dest_rank in dump info";
    EXPECT_TRUE(found_tag) << "Missing tag in dump info";
    EXPECT_TRUE(found_async) << "Missing async in dump info";
}

/**
 * @test Verify SendActivationsStage async methods work without actual MPI
 */
TEST_F(Test__PPActivationStages, SendStage_AsyncMethodsWithoutOperation)
{
    TensorFactory factory(*mpi_ctx_);
    auto buffer = factory.createFP32({4, 128}, DeviceId::cpu());

    SendActivationsStage::Params params{
        .device_id = DeviceId::cpu(),
        .mpi_ctx = mpi_ctx_.get(),
        .buffer = buffer.get(),
        .dest_rank = 1,
        .tag = 0,
        .async = true,
    };

    auto *stage = dynamic_cast<SendActivationsStage *>(
        ComputeStageFactory::createSendActivations(params).release());
    ASSERT_NE(stage, nullptr);

    // Before execute, no pending request
    EXPECT_TRUE(stage->isComplete()); // No operation pending
    EXPECT_EQ(stage->getPendingRequest(), MPI_REQUEST_NULL);

    // wait() should be no-op when nothing pending
    stage->wait(); // Should not crash

    delete stage;
}

// =============================================================================
// ReceiveActivationsStage Unit Tests
// =============================================================================

/**
 * @test Verify ReceiveActivationsStage type returns RECV_ACTIVATIONS
 */
TEST_F(Test__PPActivationStages, RecvStage_TypeIsRecvActivations)
{
    TensorFactory factory(*mpi_ctx_);
    auto buffer = factory.createFP32({4, 128}, DeviceId::cpu());

    ReceiveActivationsStage::Params params{
        .device_id = DeviceId::cpu(),
        .mpi_ctx = mpi_ctx_.get(),
        .buffer = buffer.get(),
        .src_rank = 0,
        .tag = 42,
        .async = false,
    };

    auto stage = ComputeStageFactory::createReceiveActivations(params);

    EXPECT_EQ(stage->type(), ComputeStageType::RECV_ACTIVATIONS);
}

/**
 * @test Verify ReceiveActivationsStage name generation with src_rank and tag
 */
TEST_F(Test__PPActivationStages, RecvStage_GeneratesCorrectName)
{
    TensorFactory factory(*mpi_ctx_);
    auto buffer = factory.createFP32({4, 128}, DeviceId::cpu());

    ReceiveActivationsStage::Params params{
        .device_id = DeviceId::cpu(),
        .mpi_ctx = mpi_ctx_.get(),
        .buffer = buffer.get(),
        .src_rank = 2,
        .tag = 55,
        .async = false,
    };

    auto stage = ComputeStageFactory::createReceiveActivations(params);

    std::string expected_name = "recv_activations_from_rank2_tag55";
    EXPECT_EQ(stage->name(), expected_name);
}

/**
 * @test Verify ReceiveActivationsStage name for ANY_SOURCE
 */
TEST_F(Test__PPActivationStages, RecvStage_NameForAnySource)
{
    TensorFactory factory(*mpi_ctx_);
    auto buffer = factory.createFP32({4, 128}, DeviceId::cpu());

    ReceiveActivationsStage::Params params{
        .device_id = DeviceId::cpu(),
        .mpi_ctx = mpi_ctx_.get(),
        .buffer = buffer.get(),
        .src_rank = -1, // MPI_ANY_SOURCE
        .tag = 33,
        .async = false,
    };

    auto stage = ComputeStageFactory::createReceiveActivations(params);

    std::string expected_name = "recv_activations_any_source_tag33";
    EXPECT_EQ(stage->name(), expected_name);
}

/**
 * @test Verify ReceiveActivationsStage uses custom name when provided
 */
TEST_F(Test__PPActivationStages, RecvStage_UsesCustomName)
{
    TensorFactory factory(*mpi_ctx_);
    auto buffer = factory.createFP32({4, 128}, DeviceId::cpu());

    ReceiveActivationsStage::Params params{
        .device_id = DeviceId::cpu(),
        .mpi_ctx = mpi_ctx_.get(),
        .buffer = buffer.get(),
        .src_rank = 0,
        .tag = 42,
        .async = false,
        .stage_name = "layer10_backward_recv",
    };

    auto stage = ComputeStageFactory::createReceiveActivations(params);

    EXPECT_EQ(stage->name(), "layer10_backward_recv");
}

/**
 * @test Verify ReceiveActivationsStage supports all backends
 */
TEST_F(Test__PPActivationStages, RecvStage_SupportsAllBackends)
{
    TensorFactory factory(*mpi_ctx_);
    auto buffer = factory.createFP32({4, 128}, DeviceId::cpu());

    ReceiveActivationsStage::Params params{
        .device_id = DeviceId::cpu(),
        .mpi_ctx = mpi_ctx_.get(),
        .buffer = buffer.get(),
        .src_rank = 0,
        .tag = 0,
        .async = false,
    };

    auto stage = ComputeStageFactory::createReceiveActivations(params);

    // MPI stages are backend-agnostic
    EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::CPU));
    EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::GPU_CUDA));
    EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::GPU_ROCM));
    EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::GPU_VULKAN));
}

/**
 * @test Verify ReceiveActivationsStage coherence policy is NONE
 */
TEST_F(Test__PPActivationStages, RecvStage_CoherencePolicyIsNone)
{
    ReceiveActivationsStage::Params params{
        .device_id = DeviceId::cpu(),
        .mpi_ctx = mpi_ctx_.get(),
    };

    auto stage = ComputeStageFactory::createReceiveActivations(params);

    EXPECT_EQ(stage->coherencePolicy(), CoherencePolicy::NONE);
}

/**
 * @test Verify ReceiveActivationsStage fails with null buffer
 */
TEST_F(Test__PPActivationStages, RecvStage_FailsWithNullBuffer)
{
    ReceiveActivationsStage::Params params{
        .device_id = DeviceId::cpu(),
        .mpi_ctx = mpi_ctx_.get(),
        .buffer = nullptr,
        .src_rank = 0,
        .tag = 0,
        .async = false,
    };

    auto stage = ComputeStageFactory::createReceiveActivations(params);

    EXPECT_FALSE(stage->execute(nullptr));
}

/**
 * @test Verify ReceiveActivationsStage fails with null MPI context
 */
TEST_F(Test__PPActivationStages, RecvStage_FailsWithNullMpiContext)
{
    TensorFactory factory(*mpi_ctx_);
    auto buffer = factory.createFP32({4, 128}, DeviceId::cpu());

    ReceiveActivationsStage::Params params{
        .device_id = DeviceId::cpu(),
        .mpi_ctx = nullptr,
        .buffer = buffer.get(),
        .src_rank = 0,
        .tag = 0,
        .async = false,
    };

    auto stage = ComputeStageFactory::createReceiveActivations(params);

    EXPECT_FALSE(stage->execute(nullptr));
}

/**
 * @test Verify ReceiveActivationsStage buffer requirements
 */
TEST_F(Test__PPActivationStages, RecvStage_BufferRequirements)
{
    TensorFactory factory(*mpi_ctx_);
    auto buffer = factory.createFP32({4, 128}, DeviceId::cpu());

    ReceiveActivationsStage::Params params{
        .device_id = DeviceId::cpu(),
        .mpi_ctx = mpi_ctx_.get(),
        .buffer = buffer.get(),
        .src_rank = 0,
        .tag = 0,
        .async = false,
    };

    auto stage = ComputeStageFactory::createReceiveActivations(params);
    auto reqs = stage->getBufferRequirements();

    // Receive stage writes to one output buffer
    EXPECT_GE(reqs.buffers.size(), 1);
}

/**
 * @test Verify ReceiveActivationsStage dump info contains expected fields
 */
TEST_F(Test__PPActivationStages, RecvStage_DumpInfoIsComplete)
{
    TensorFactory factory(*mpi_ctx_);
    auto buffer = factory.createFP32({16, 256}, DeviceId::cpu());

    ReceiveActivationsStage::Params params{
        .device_id = DeviceId::cpu(),
        .mpi_ctx = mpi_ctx_.get(),
        .buffer = buffer.get(),
        .src_rank = 5,
        .tag = 88,
        .async = false,
    };

    auto stage = ComputeStageFactory::createReceiveActivations(params);
    auto dump_info = stage->getDumpInfo();

    // Verify outputs (receive writes to buffer)
    EXPECT_EQ(dump_info.outputs.size(), 1);
    EXPECT_STREQ(dump_info.outputs[0].name, "buffer");
    EXPECT_EQ(dump_info.outputs[0].rows, 16);
    EXPECT_EQ(dump_info.outputs[0].cols, 256);

    // Verify scalars
    EXPECT_GE(dump_info.scalars.size(), 3); // src_rank, tag, async

    // Find specific scalars
    bool found_src_rank = false;
    bool found_tag = false;
    bool found_async = false;

    for (const auto &scalar : dump_info.scalars)
    {
        if (std::string(scalar.name) == "src_rank")
        {
            EXPECT_EQ(static_cast<int>(scalar.value), 5);
            found_src_rank = true;
        }
        else if (std::string(scalar.name) == "tag")
        {
            EXPECT_EQ(static_cast<int>(scalar.value), 88);
            found_tag = true;
        }
        else if (std::string(scalar.name) == "async")
        {
            EXPECT_EQ(scalar.value, 0.0); // false = 0.0
            found_async = true;
        }
    }

    EXPECT_TRUE(found_src_rank) << "Missing src_rank in dump info";
    EXPECT_TRUE(found_tag) << "Missing tag in dump info";
    EXPECT_TRUE(found_async) << "Missing async in dump info";
}

/**
 * @test Verify ReceiveActivationsStage async methods work without actual MPI
 */
TEST_F(Test__PPActivationStages, RecvStage_AsyncMethodsWithoutOperation)
{
    TensorFactory factory(*mpi_ctx_);
    auto buffer = factory.createFP32({4, 128}, DeviceId::cpu());

    ReceiveActivationsStage::Params params{
        .device_id = DeviceId::cpu(),
        .mpi_ctx = mpi_ctx_.get(),
        .buffer = buffer.get(),
        .src_rank = 0,
        .tag = 0,
        .async = true,
    };

    auto *stage = dynamic_cast<ReceiveActivationsStage *>(
        ComputeStageFactory::createReceiveActivations(params).release());
    ASSERT_NE(stage, nullptr);

    // Before execute, no pending request
    EXPECT_TRUE(stage->isComplete()); // No operation pending
    EXPECT_EQ(stage->getPendingRequest(), MPI_REQUEST_NULL);

    // wait() should be no-op when nothing pending
    stage->wait(); // Should not crash

    delete stage;
}

// =============================================================================
// Factory Tests
// =============================================================================

/**
 * @test Verify factory creates SendActivationsStage correctly
 */
TEST_F(Test__PPActivationStages, Factory_CreatesSendActivationsStage)
{
    TensorFactory factory(*mpi_ctx_);
    auto buffer = factory.createFP32({4, 128}, DeviceId::cpu());

    SendActivationsStage::Params params{
        .device_id = DeviceId::cpu(),
        .mpi_ctx = mpi_ctx_.get(),
        .buffer = buffer.get(),
        .dest_rank = 1,
        .tag = 42,
        .async = false,
    };

    auto stage = ComputeStageFactory::createSendActivations(params);

    ASSERT_NE(stage, nullptr);
    EXPECT_EQ(stage->type(), ComputeStageType::SEND_ACTIVATIONS);
}

/**
 * @test Verify factory creates ReceiveActivationsStage correctly
 */
TEST_F(Test__PPActivationStages, Factory_CreatesReceiveActivationsStage)
{
    TensorFactory factory(*mpi_ctx_);
    auto buffer = factory.createFP32({4, 128}, DeviceId::cpu());

    ReceiveActivationsStage::Params params{
        .device_id = DeviceId::cpu(),
        .mpi_ctx = mpi_ctx_.get(),
        .buffer = buffer.get(),
        .src_rank = 0,
        .tag = 42,
        .async = false,
    };

    auto stage = ComputeStageFactory::createReceiveActivations(params);

    ASSERT_NE(stage, nullptr);
    EXPECT_EQ(stage->type(), ComputeStageType::RECV_ACTIVATIONS);
}

// =============================================================================
// Params Validation Tests
// =============================================================================

/**
 * @test Verify SendActivationsStage::Params stores all fields
 */
TEST_F(Test__PPActivationStages, SendParams_StoresAllFields)
{
    TensorFactory factory(*mpi_ctx_);
    auto buffer = factory.createFP32({4, 128}, DeviceId::cpu());

    SendActivationsStage::Params params{
        .device_id = DeviceId::cuda(0),
        .mpi_ctx = mpi_ctx_.get(),
        .buffer = buffer.get(),
        .dest_rank = 7,
        .tag = 999,
        .async = true,
        .stage_name = "my_send_stage",
    };

    EXPECT_EQ(params.device_id, DeviceId::cuda(0));
    EXPECT_EQ(params.mpi_ctx, mpi_ctx_.get());
    EXPECT_EQ(params.buffer, buffer.get());
    EXPECT_EQ(params.dest_rank, 7);
    EXPECT_EQ(params.tag, 999);
    EXPECT_TRUE(params.async);
    EXPECT_EQ(params.stage_name, "my_send_stage");
}

/**
 * @test Verify ReceiveActivationsStage::Params stores all fields
 */
TEST_F(Test__PPActivationStages, RecvParams_StoresAllFields)
{
    TensorFactory factory(*mpi_ctx_);
    auto buffer = factory.createFP32({4, 128}, DeviceId::cpu());

    ReceiveActivationsStage::Params params{
        .device_id = DeviceId::rocm(1),
        .mpi_ctx = mpi_ctx_.get(),
        .buffer = buffer.get(),
        .src_rank = 3,
        .tag = 777,
        .async = true,
        .stage_name = "my_recv_stage",
    };

    EXPECT_EQ(params.device_id, DeviceId::rocm(1));
    EXPECT_EQ(params.mpi_ctx, mpi_ctx_.get());
    EXPECT_EQ(params.buffer, buffer.get());
    EXPECT_EQ(params.src_rank, 3);
    EXPECT_EQ(params.tag, 777);
    EXPECT_TRUE(params.async);
    EXPECT_EQ(params.stage_name, "my_recv_stage");
}
