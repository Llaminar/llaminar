/**
 * @file Test__PipelineActivationTransfer.cpp
 * @brief Integration tests for SendActivationsStage and ReceiveActivationsStage
 *
 * Tests MPI point-to-point activation transfer for pipeline parallelism.
 * Run with: mpirun -np 2 ./v2_integration_pipeline_activation_transfer
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <memory>
#include <vector>
#include <cmath>
#include <numeric>
#include <chrono>

#include "execution/compute_stages/ComputeStageFactory.h"
#include "execution/compute_stages/stages/SendActivationsStage.h"
#include "execution/compute_stages/stages/ReceiveActivationsStage.h"
#include "tensors/Tensors.h"
#include "utils/MPIContext.h"
#include "utils/MPITags.h"
#include "utils/Logger.h"

using namespace llaminar2;

class PipelineActivationTransfer : public ::testing::Test
{
protected:
    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

        mpi_ctx_ = std::make_shared<MPIContext>(rank_, world_size_);
    }

    void TearDown() override
    {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    int rank_;
    int world_size_;
    std::shared_ptr<MPIContext> mpi_ctx_;
};

// ============================================================================
// Test 1: Synchronous Send/Recv with Small Tensor (1KB)
// ============================================================================

TEST_F(PipelineActivationTransfer, SyncSendRecv_SmallTensor)
{
    if (world_size_ < 2)
    {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    // 1KB = 256 floats
    constexpr size_t SEQ_LEN = 8;
    constexpr size_t D_MODEL = 32; // 8 * 32 = 256 floats = 1KB
    constexpr size_t COUNT = SEQ_LEN * D_MODEL;
    constexpr int TAG = mpi_tags::forwardTag(0);

    // Create tensor
    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{SEQ_LEN, D_MODEL});
    float *data = tensor->mutable_data();

    if (rank_ == 0)
    {
        // Fill with pattern
        for (size_t i = 0; i < COUNT; ++i)
        {
            data[i] = static_cast<float>(i) * 0.01f;
        }

        // Create and execute send stage
        SendActivationsStage::Params params{
            .device_id = DeviceId::cpu(),
            .mpi_ctx = mpi_ctx_.get(),
            .buffer = tensor.get(),
            .dest_rank = 1,
            .tag = TAG,
            .async = false,
        };

        auto stage = ComputeStageFactory::createSendActivations(params);
        EXPECT_EQ(stage->type(), ComputeStageType::SEND_ACTIVATIONS);

        bool result = stage->execute(nullptr);
        EXPECT_TRUE(result);

        LOG_INFO("[Rank 0] Sent " << COUNT << " floats to rank 1");
    }
    else if (rank_ == 1)
    {
        // Initialize with zeros
        std::fill(data, data + COUNT, 0.0f);

        // Create and execute receive stage
        ReceiveActivationsStage::Params params{
            .device_id = DeviceId::cpu(),
            .mpi_ctx = mpi_ctx_.get(),
            .buffer = tensor.get(),
            .src_rank = 0,
            .tag = TAG,
            .async = false,
        };

        auto stage = ComputeStageFactory::createReceiveActivations(params);
        EXPECT_EQ(stage->type(), ComputeStageType::RECV_ACTIVATIONS);

        bool result = stage->execute(nullptr);
        EXPECT_TRUE(result);

        // Verify received data
        for (size_t i = 0; i < COUNT; ++i)
        {
            EXPECT_FLOAT_EQ(data[i], static_cast<float>(i) * 0.01f)
                << "Data mismatch at index " << i;
        }

        LOG_INFO("[Rank 1] Received and verified " << COUNT << " floats from rank 0");
    }
}

// ============================================================================
// Test 2: Synchronous Send/Recv with Large Tensor (d_model sized)
// ============================================================================

TEST_F(PipelineActivationTransfer, SyncSendRecv_LargeTensor)
{
    if (world_size_ < 2)
    {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    // Typical d_model size for Qwen2.5-0.5B: 896
    // With seq_len=16: 16 * 896 = 14336 floats = ~56KB
    constexpr size_t SEQ_LEN = 16;
    constexpr size_t D_MODEL = 896;
    constexpr size_t COUNT = SEQ_LEN * D_MODEL;
    constexpr int TAG = mpi_tags::forwardTag(5); // Layer 5

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{SEQ_LEN, D_MODEL});
    float *data = tensor->mutable_data();

    if (rank_ == 0)
    {
        // Fill with pattern: row * 1000 + col
        for (size_t r = 0; r < SEQ_LEN; ++r)
        {
            for (size_t c = 0; c < D_MODEL; ++c)
            {
                data[r * D_MODEL + c] = static_cast<float>(r * 1000 + c);
            }
        }

        auto start = std::chrono::high_resolution_clock::now();

        SendActivationsStage::Params params{
            .device_id = DeviceId::cpu(),
            .mpi_ctx = mpi_ctx_.get(),
            .buffer = tensor.get(),
            .dest_rank = 1,
            .tag = TAG,
            .async = false,
            .stage_name = "layer5_activation_send",
        };

        auto stage = ComputeStageFactory::createSendActivations(params);
        EXPECT_TRUE(stage->execute(nullptr));

        auto end = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        double bandwidth = (COUNT * sizeof(float) / 1e6) / (us / 1e6); // MB/s
        LOG_INFO("[Rank 0] Sent " << COUNT << " floats (" << COUNT * sizeof(float) / 1024
                                  << " KB) in " << us << " μs (" << bandwidth << " MB/s)");
    }
    else if (rank_ == 1)
    {
        std::fill(data, data + COUNT, -1.0f);

        auto start = std::chrono::high_resolution_clock::now();

        ReceiveActivationsStage::Params params{
            .device_id = DeviceId::cpu(),
            .mpi_ctx = mpi_ctx_.get(),
            .buffer = tensor.get(),
            .src_rank = 0,
            .tag = TAG,
            .async = false,
            .stage_name = "layer5_activation_recv",
        };

        auto stage = ComputeStageFactory::createReceiveActivations(params);
        EXPECT_TRUE(stage->execute(nullptr));

        auto end = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        LOG_INFO("[Rank 1] Received " << COUNT << " floats in " << us << " μs");

        // Verify corners
        EXPECT_FLOAT_EQ(data[0], 0.0f);                                            // [0,0]
        EXPECT_FLOAT_EQ(data[D_MODEL - 1], D_MODEL - 1);                           // [0, D-1]
        EXPECT_FLOAT_EQ(data[(SEQ_LEN - 1) * D_MODEL], (SEQ_LEN - 1) * 1000.0f);   // [S-1, 0]
        EXPECT_FLOAT_EQ(data[COUNT - 1], (SEQ_LEN - 1) * 1000.0f + (D_MODEL - 1)); // [S-1, D-1]

        LOG_INFO("[Rank 1] Large tensor verification passed");
    }
}

// ============================================================================
// Test 3: Async Send/Recv with Overlap
// ============================================================================

TEST_F(PipelineActivationTransfer, AsyncSendRecv_Overlap)
{
    if (world_size_ < 2)
    {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    constexpr size_t SEQ_LEN = 4;
    constexpr size_t D_MODEL = 256;
    constexpr size_t COUNT = SEQ_LEN * D_MODEL;
    constexpr int TAG = mpi_tags::forwardTag(0);

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{SEQ_LEN, D_MODEL});
    float *data = tensor->mutable_data();

    if (rank_ == 0)
    {
        // Fill with pattern
        for (size_t i = 0; i < COUNT; ++i)
        {
            data[i] = static_cast<float>(i);
        }

        // Create async send stage
        SendActivationsStage::Params params{
            .device_id = DeviceId::cpu(),
            .mpi_ctx = mpi_ctx_.get(),
            .buffer = tensor.get(),
            .dest_rank = 1,
            .tag = TAG,
            .async = true,
        };

        auto stage = std::make_unique<SendActivationsStage>(params);

        // Execute async send
        EXPECT_TRUE(stage->execute(nullptr));

        // Can do overlapped work here...
        double dummy_work = 0;
        for (int i = 0; i < 1000; ++i)
        {
            dummy_work += std::sin(static_cast<double>(i));
        }
        (void)dummy_work;

        // Now wait for completion
        EXPECT_FALSE(stage->isComplete() && false); // May or may not be complete yet
        stage->wait();
        EXPECT_TRUE(stage->isComplete());

        LOG_INFO("[Rank 0] Async send completed with overlapped work");
    }
    else if (rank_ == 1)
    {
        std::fill(data, data + COUNT, -1.0f);

        // Create async receive stage
        ReceiveActivationsStage::Params params{
            .device_id = DeviceId::cpu(),
            .mpi_ctx = mpi_ctx_.get(),
            .buffer = tensor.get(),
            .src_rank = 0,
            .tag = TAG,
            .async = true,
        };

        auto stage = std::make_unique<ReceiveActivationsStage>(params);

        // Execute async recv
        EXPECT_TRUE(stage->execute(nullptr));

        // Wait for completion
        stage->wait();
        EXPECT_TRUE(stage->isComplete());

        // Verify
        EXPECT_FLOAT_EQ(data[0], 0.0f);
        EXPECT_FLOAT_EQ(data[COUNT - 1], static_cast<float>(COUNT - 1));

        LOG_INFO("[Rank 1] Async recv completed and verified");
    }
}

// ============================================================================
// Test 4: Bidirectional Ping-Pong
// ============================================================================

TEST_F(PipelineActivationTransfer, Bidirectional_PingPong)
{
    if (world_size_ < 2)
    {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    constexpr size_t SEQ_LEN = 2;
    constexpr size_t D_MODEL = 64;
    constexpr size_t COUNT = SEQ_LEN * D_MODEL;
    constexpr int TAG_FWD = mpi_tags::forwardTag(0);
    constexpr int TAG_BWD = mpi_tags::backwardTag(0);

    auto send_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{SEQ_LEN, D_MODEL});
    auto recv_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{SEQ_LEN, D_MODEL});

    float *send_data = send_tensor->mutable_data();
    float *recv_data = recv_tensor->mutable_data();

    int partner = (rank_ == 0) ? 1 : 0;

    // Fill send buffer with rank-specific pattern
    for (size_t i = 0; i < COUNT; ++i)
    {
        send_data[i] = static_cast<float>(rank_ * 1000 + i);
    }
    std::fill(recv_data, recv_data + COUNT, -1.0f);

    // Use async operations for bidirectional exchange
    SendActivationsStage::Params send_params{
        .device_id = DeviceId::cpu(),
        .mpi_ctx = mpi_ctx_.get(),
        .buffer = send_tensor.get(),
        .dest_rank = partner,
        .tag = (rank_ == 0) ? TAG_FWD : TAG_BWD,
        .async = true,
    };

    ReceiveActivationsStage::Params recv_params{
        .device_id = DeviceId::cpu(),
        .mpi_ctx = mpi_ctx_.get(),
        .buffer = recv_tensor.get(),
        .src_rank = partner,
        .tag = (rank_ == 0) ? TAG_BWD : TAG_FWD,
        .async = true,
    };

    auto send_stage = std::make_unique<SendActivationsStage>(send_params);
    auto recv_stage = std::make_unique<ReceiveActivationsStage>(recv_params);

    // Post both operations
    EXPECT_TRUE(recv_stage->execute(nullptr)); // Post recv first
    EXPECT_TRUE(send_stage->execute(nullptr));

    // Wait for both
    send_stage->wait();
    recv_stage->wait();

    // Verify received data is from partner
    float expected_base = static_cast<float>(partner * 1000);
    for (size_t i = 0; i < COUNT; ++i)
    {
        EXPECT_FLOAT_EQ(recv_data[i], expected_base + static_cast<float>(i))
            << "Bidirectional data mismatch at index " << i;
    }

    LOG_INFO("[Rank " << rank_ << "] Bidirectional exchange with rank " << partner << " passed");
}

// ============================================================================
// Test 5: Multi-Layer Sequence (simulate pipeline stages)
// ============================================================================

TEST_F(PipelineActivationTransfer, MultiLayer_Sequence)
{
    if (world_size_ < 2)
    {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    constexpr size_t SEQ_LEN = 4;
    constexpr size_t D_MODEL = 128;
    constexpr size_t COUNT = SEQ_LEN * D_MODEL;
    constexpr int NUM_LAYERS = 4;

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{SEQ_LEN, D_MODEL});
    float *data = tensor->mutable_data();

    // Simulate pipeline: rank 0 sends activations for multiple layers in sequence
    // This tests that tags correctly separate different layer boundaries

    for (int layer = 0; layer < NUM_LAYERS; ++layer)
    {
        int tag = mpi_tags::forwardTag(layer);

        if (rank_ == 0)
        {
            // Fill with layer-specific pattern
            for (size_t i = 0; i < COUNT; ++i)
            {
                data[i] = static_cast<float>(layer * 10000 + i);
            }

            SendActivationsStage::Params params{
                .device_id = DeviceId::cpu(),
                .mpi_ctx = mpi_ctx_.get(),
                .buffer = tensor.get(),
                .dest_rank = 1,
                .tag = tag,
                .async = false,
                .stage_name = "layer" + std::to_string(layer) + "_send",
            };

            auto stage = ComputeStageFactory::createSendActivations(params);
            EXPECT_TRUE(stage->execute(nullptr));
        }
        else if (rank_ == 1)
        {
            std::fill(data, data + COUNT, -1.0f);

            ReceiveActivationsStage::Params params{
                .device_id = DeviceId::cpu(),
                .mpi_ctx = mpi_ctx_.get(),
                .buffer = tensor.get(),
                .src_rank = 0,
                .tag = tag,
                .async = false,
                .stage_name = "layer" + std::to_string(layer) + "_recv",
            };

            auto stage = ComputeStageFactory::createReceiveActivations(params);
            EXPECT_TRUE(stage->execute(nullptr));

            // Verify layer-specific pattern
            float expected_base = static_cast<float>(layer * 10000);
            EXPECT_FLOAT_EQ(data[0], expected_base);
            EXPECT_FLOAT_EQ(data[COUNT - 1], expected_base + COUNT - 1);
        }
    }

    LOG_INFO("[Rank " << rank_ << "] Multi-layer sequence (" << NUM_LAYERS << " layers) passed");
}

// ============================================================================
// Test 6: Stage Buffer Requirements
// ============================================================================

TEST_F(PipelineActivationTransfer, StageBufferRequirements)
{
    constexpr size_t SEQ_LEN = 4;
    constexpr size_t D_MODEL = 64;

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{SEQ_LEN, D_MODEL});

    // Test send stage requirements
    {
        SendActivationsStage::Params params{
            .device_id = DeviceId::cpu(),
            .mpi_ctx = mpi_ctx_.get(),
            .buffer = tensor.get(),
            .dest_rank = 1,
            .tag = 0,
            .async = false,
        };

        auto stage = ComputeStageFactory::createSendActivations(params);
        auto reqs = stage->getBufferRequirements();

        // Send should declare buffer as INPUT - count buffers with INPUT role
        size_t input_count = 0;
        size_t output_count = 0;
        for (const auto &buf : reqs.buffers)
        {
            if (buf.role == BufferRole::INPUT)
                input_count++;
            if (buf.role == BufferRole::OUTPUT)
                output_count++;
        }
        EXPECT_GE(input_count, 1);
        EXPECT_EQ(output_count, 0);
    }

    // Test receive stage requirements
    {
        ReceiveActivationsStage::Params params{
            .device_id = DeviceId::cpu(),
            .mpi_ctx = mpi_ctx_.get(),
            .buffer = tensor.get(),
            .src_rank = 0,
            .tag = 0,
            .async = false,
        };

        auto stage = ComputeStageFactory::createReceiveActivations(params);
        auto reqs = stage->getBufferRequirements();

        // Receive should declare buffer as OUTPUT - count buffers with OUTPUT role
        size_t input_count = 0;
        size_t output_count = 0;
        for (const auto &buf : reqs.buffers)
        {
            if (buf.role == BufferRole::INPUT)
                input_count++;
            if (buf.role == BufferRole::OUTPUT)
                output_count++;
        }
        EXPECT_EQ(input_count, 0);
        EXPECT_GE(output_count, 1);
    }

    LOG_INFO("Stage buffer requirements test passed");
}

// ============================================================================
// Test 7: Dump Info
// ============================================================================

TEST_F(PipelineActivationTransfer, StageDumpInfo)
{
    constexpr size_t SEQ_LEN = 4;
    constexpr size_t D_MODEL = 64;

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{SEQ_LEN, D_MODEL});

    // Test send stage dump info
    {
        SendActivationsStage::Params params{
            .device_id = DeviceId::cpu(),
            .mpi_ctx = mpi_ctx_.get(),
            .buffer = tensor.get(),
            .dest_rank = 1,
            .tag = 42,
            .async = true,
        };

        auto stage = ComputeStageFactory::createSendActivations(params);
        auto info = stage->getDumpInfo();

        // Should have input buffer and scalars
        EXPECT_GE(info.inputs.size(), 1);
        EXPECT_GE(info.scalars.size(), 3); // dest_rank, tag, async

        LOG_INFO("Send stage dump info: " << info.inputs.size() << " inputs, "
                                          << info.scalars.size() << " scalars");
    }

    // Test receive stage dump info
    {
        ReceiveActivationsStage::Params params{
            .device_id = DeviceId::cpu(),
            .mpi_ctx = mpi_ctx_.get(),
            .buffer = tensor.get(),
            .src_rank = 0,
            .tag = 42,
            .async = false,
        };

        auto stage = ComputeStageFactory::createReceiveActivations(params);
        auto info = stage->getDumpInfo();

        // Should have output buffer and scalars
        EXPECT_GE(info.outputs.size(), 1);
        EXPECT_GE(info.scalars.size(), 3); // src_rank, tag, async

        LOG_INFO("Recv stage dump info: " << info.outputs.size() << " outputs, "
                                          << info.scalars.size() << " scalars");
    }
}

// ============================================================================
// Test 8: Error Handling - Null Buffer
// ============================================================================

TEST_F(PipelineActivationTransfer, ErrorHandling_NullBuffer)
{
    // Send with null buffer
    {
        SendActivationsStage::Params params{
            .device_id = DeviceId::cpu(),
            .mpi_ctx = mpi_ctx_.get(),
            .buffer = nullptr, // Null!
            .dest_rank = 1,
            .tag = 0,
            .async = false,
        };

        auto stage = ComputeStageFactory::createSendActivations(params);
        EXPECT_FALSE(stage->execute(nullptr));
    }

    // Recv with null buffer
    {
        ReceiveActivationsStage::Params params{
            .device_id = DeviceId::cpu(),
            .mpi_ctx = mpi_ctx_.get(),
            .buffer = nullptr, // Null!
            .src_rank = 0,
            .tag = 0,
            .async = false,
        };

        auto stage = ComputeStageFactory::createReceiveActivations(params);
        EXPECT_FALSE(stage->execute(nullptr));
    }

    LOG_INFO("Null buffer error handling test passed");
}

// ============================================================================
// Test 9: Error Handling - Null MPI Context
// ============================================================================

TEST_F(PipelineActivationTransfer, ErrorHandling_NullMPIContext)
{
    constexpr size_t SEQ_LEN = 4;
    constexpr size_t D_MODEL = 64;

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{SEQ_LEN, D_MODEL});

    // Send with null MPI context
    {
        SendActivationsStage::Params params{
            .device_id = DeviceId::cpu(),
            .mpi_ctx = nullptr, // Null!
            .buffer = tensor.get(),
            .dest_rank = 1,
            .tag = 0,
            .async = false,
        };

        auto stage = ComputeStageFactory::createSendActivations(params);
        EXPECT_FALSE(stage->execute(nullptr));
    }

    // Recv with null MPI context
    {
        ReceiveActivationsStage::Params params{
            .device_id = DeviceId::cpu(),
            .mpi_ctx = nullptr, // Null!
            .buffer = tensor.get(),
            .src_rank = 0,
            .tag = 0,
            .async = false,
        };

        auto stage = ComputeStageFactory::createReceiveActivations(params);
        EXPECT_FALSE(stage->execute(nullptr));
    }

    LOG_INFO("Null MPI context error handling test passed");
}

// Note: main() provided by mpi_gtest_main.cpp
