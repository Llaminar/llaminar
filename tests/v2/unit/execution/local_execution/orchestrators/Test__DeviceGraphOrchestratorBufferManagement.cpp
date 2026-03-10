/**
 * @file Test__DeviceGraphOrchestratorBufferManagement.cpp
 * @brief Unit tests for DeviceGraphOrchestrator buffer management
 * @author David Sanftenberg
 * @date January 2025
 *
 * Tests the DeviceGraphBufferManager integration with DeviceGraphOrchestrator:
 * - setTensorFactory() / initializeBuffers() workflow
 * - Buffer allocation and binding
 * - Statistics tracking
 * - Release cleanup
 *
 * Note: Buffer management was moved from Qwen2Graph to DeviceGraphOrchestrator
 * as part of the declarative refactor (Phase 3).
 */

#include <gtest/gtest.h>
#include "backends/DeviceId.h"
#include "execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "models/qwen/Qwen2Graph.h"
#include "tensors/TensorFactory.h"
#include "utils/MPIContext.h"

using namespace llaminar2;

// ============================================================================
// Test Fixture
// ============================================================================

class Test__DeviceGraphOrchestratorBufferManagement : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create minimal MPI context (single rank)
        mpi_ctx_ = std::make_shared<MPIContext>(
            0, // rank
            1, // world_size
            MPI_COMM_WORLD);

        // Create TensorFactory (takes reference, not pointer)
        tensor_factory_ = std::make_unique<TensorFactory>(*mpi_ctx_);
    }

    GraphConfig createMinimalConfig()
    {
        // Create minimal config for Qwen2.5-0.5B-like model
        GraphConfig config;
        config.d_model = 896;
        config.n_heads = 14;
        config.n_kv_heads = 2;
        config.head_dim = 64;
        config.d_ff = 4864;
        config.vocab_size = 151936;
        config.n_layers = 24;
        config.use_graph_buffer_management = true;
        config.max_seq_len = 128; // Small for tests
        config.default_device = DeviceId::cpu();
        return config;
    }

    std::shared_ptr<MPIContext> mpi_ctx_;
    std::unique_ptr<TensorFactory> tensor_factory_;
};

// ============================================================================
// Basic Tests
// ============================================================================

TEST_F(Test__DeviceGraphOrchestratorBufferManagement, CreateWithBufferManagementFlag)
{
    auto config = createMinimalConfig();
    config.use_graph_buffer_management = true;

    auto graph_builder = std::make_shared<Qwen2Graph>(config, mpi_ctx_);
    DeviceGraphOrchestrator orchestrator(graph_builder, mpi_ctx_);

    EXPECT_TRUE(config.use_graph_buffer_management);
    EXPECT_FALSE(orchestrator.hasGraphManagedBuffers()) << "Should not have buffers until initializeBuffers()";
}

TEST_F(Test__DeviceGraphOrchestratorBufferManagement, InitializeWithoutTensorFactoryFails)
{
    auto config = createMinimalConfig();
    config.use_graph_buffer_management = true;

    auto graph_builder = std::make_shared<Qwen2Graph>(config, mpi_ctx_);
    DeviceGraphOrchestrator orchestrator(graph_builder, mpi_ctx_);
    // Don't call setTensorFactory()

    // Should fail because TensorFactory not set
    bool result = orchestrator.initializeBuffers(64);
    EXPECT_FALSE(result) << "Should fail without TensorFactory";
    EXPECT_FALSE(orchestrator.hasGraphManagedBuffers());
}

TEST_F(Test__DeviceGraphOrchestratorBufferManagement, InitializeBuffersSuccess)
{
    auto config = createMinimalConfig();
    config.use_graph_buffer_management = true;

    auto graph_builder = std::make_shared<Qwen2Graph>(config, mpi_ctx_);
    DeviceGraphOrchestrator orchestrator(graph_builder, mpi_ctx_);
    orchestrator.setTensorFactory(tensor_factory_.get());

    bool result = orchestrator.initializeBuffers(64);
    EXPECT_TRUE(result);
    EXPECT_TRUE(orchestrator.hasGraphManagedBuffers());
}

TEST_F(Test__DeviceGraphOrchestratorBufferManagement, BufferStatsAvailable)
{
    auto config = createMinimalConfig();
    config.use_graph_buffer_management = true;

    auto graph_builder = std::make_shared<Qwen2Graph>(config, mpi_ctx_);
    DeviceGraphOrchestrator orchestrator(graph_builder, mpi_ctx_);
    orchestrator.setTensorFactory(tensor_factory_.get());

    EXPECT_EQ(orchestrator.bufferStats(), nullptr) << "No stats before init";

    ASSERT_TRUE(orchestrator.initializeBuffers(64));

    const auto *stats = orchestrator.bufferStats();
    ASSERT_NE(stats, nullptr);
    EXPECT_GT(stats->total_bytes, 0u) << "Should have allocated some memory";
    EXPECT_GT(stats->total_buffers, 0u) << "Should have allocated buffers";
}

TEST_F(Test__DeviceGraphOrchestratorBufferManagement, ReleaseBuffers)
{
    auto config = createMinimalConfig();
    config.use_graph_buffer_management = true;

    auto graph_builder = std::make_shared<Qwen2Graph>(config, mpi_ctx_);
    DeviceGraphOrchestrator orchestrator(graph_builder, mpi_ctx_);
    orchestrator.setTensorFactory(tensor_factory_.get());

    ASSERT_TRUE(orchestrator.initializeBuffers(64));
    EXPECT_TRUE(orchestrator.hasGraphManagedBuffers());

    orchestrator.releaseBuffers();
    EXPECT_FALSE(orchestrator.hasGraphManagedBuffers());
    EXPECT_EQ(orchestrator.bufferStats(), nullptr);
}

TEST_F(Test__DeviceGraphOrchestratorBufferManagement, ReinitializeAfterRelease)
{
    auto config = createMinimalConfig();
    config.use_graph_buffer_management = true;

    auto graph_builder = std::make_shared<Qwen2Graph>(config, mpi_ctx_);
    DeviceGraphOrchestrator orchestrator(graph_builder, mpi_ctx_);
    orchestrator.setTensorFactory(tensor_factory_.get());

    // First init
    ASSERT_TRUE(orchestrator.initializeBuffers(64));
    EXPECT_TRUE(orchestrator.hasGraphManagedBuffers());

    // Release
    orchestrator.releaseBuffers();
    EXPECT_FALSE(orchestrator.hasGraphManagedBuffers());

    // Re-initialize with different size
    ASSERT_TRUE(orchestrator.initializeBuffers(128));
    EXPECT_TRUE(orchestrator.hasGraphManagedBuffers());

    const auto *stats = orchestrator.bufferStats();
    ASSERT_NE(stats, nullptr);
    EXPECT_GT(stats->total_bytes, 0u);
}

// ============================================================================
// Memory Allocation Tests
// ============================================================================

TEST_F(Test__DeviceGraphOrchestratorBufferManagement, AllocatesReasonableMemory)
{
    auto config = createMinimalConfig();
    config.use_graph_buffer_management = true;

    auto graph_builder = std::make_shared<Qwen2Graph>(config, mpi_ctx_);
    DeviceGraphOrchestrator orchestrator(graph_builder, mpi_ctx_);
    orchestrator.setTensorFactory(tensor_factory_.get());

    ASSERT_TRUE(orchestrator.initializeBuffers(64));

    const auto *stats = orchestrator.bufferStats();
    ASSERT_NE(stats, nullptr);

    // Expected: layer buffers + model buffers
    // Layer: Q, K, V, GATE, UP, etc. all [64, d_model] or [64, d_ff]
    // For d_model=896, d_ff=4864, seq_len=64:
    // - Q/K/V: 64 * 896 * 4 bytes each
    // - GATE/UP: 64 * 4864 * 4 bytes each
    // Should be at least several MB total

    size_t min_expected_bytes = 1 * 1024 * 1024;   // At least 1 MB
    size_t max_expected_bytes = 100 * 1024 * 1024; // Less than 100 MB for seq_len=64

    EXPECT_GE(stats->total_bytes, min_expected_bytes)
        << "Should allocate reasonable memory";
    EXPECT_LE(stats->total_bytes, max_expected_bytes)
        << "Should not allocate excessive memory";
}

TEST_F(Test__DeviceGraphOrchestratorBufferManagement, LargerSeqLenUsesMoreMemory)
{
    auto config = createMinimalConfig();
    config.use_graph_buffer_management = true;

    auto graph_builder1 = std::make_shared<Qwen2Graph>(config, mpi_ctx_);
    DeviceGraphOrchestrator orchestrator1(graph_builder1, mpi_ctx_);
    orchestrator1.setTensorFactory(tensor_factory_.get());
    ASSERT_TRUE(orchestrator1.initializeBuffers(64));
    size_t bytes_64 = orchestrator1.bufferStats()->total_bytes;

    orchestrator1.releaseBuffers();

    auto graph_builder2 = std::make_shared<Qwen2Graph>(config, mpi_ctx_);
    DeviceGraphOrchestrator orchestrator2(graph_builder2, mpi_ctx_);
    orchestrator2.setTensorFactory(tensor_factory_.get());
    ASSERT_TRUE(orchestrator2.initializeBuffers(256));
    size_t bytes_256 = orchestrator2.bufferStats()->total_bytes;

    // Larger seq_len should use more memory
    EXPECT_GT(bytes_256, bytes_64)
        << "256 tokens should use more memory than 64 tokens";

    // Should scale roughly linearly with seq_len for main buffers
    double ratio = static_cast<double>(bytes_256) / bytes_64;
    EXPECT_GT(ratio, 2.0) << "Should scale somewhat with seq_len";
    EXPECT_LT(ratio, 10.0) << "Shouldn't scale excessively";
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(Test__DeviceGraphOrchestratorBufferManagement, VerySmallSeqLen)
{
    auto config = createMinimalConfig();
    config.use_graph_buffer_management = true;

    auto graph_builder = std::make_shared<Qwen2Graph>(config, mpi_ctx_);
    DeviceGraphOrchestrator orchestrator(graph_builder, mpi_ctx_);
    orchestrator.setTensorFactory(tensor_factory_.get());

    ASSERT_TRUE(orchestrator.initializeBuffers(1));
    EXPECT_TRUE(orchestrator.hasGraphManagedBuffers());

    const auto *stats = orchestrator.bufferStats();
    ASSERT_NE(stats, nullptr);
    EXPECT_GT(stats->total_bytes, 0u);
}

TEST_F(Test__DeviceGraphOrchestratorBufferManagement, WithoutUseGraphBufferManagementFlagFails)
{
    auto config = createMinimalConfig();
    config.use_graph_buffer_management = false; // Disabled

    auto graph_builder = std::make_shared<Qwen2Graph>(config, mpi_ctx_);
    DeviceGraphOrchestrator orchestrator(graph_builder, mpi_ctx_);
    orchestrator.setTensorFactory(tensor_factory_.get());

    // initializeBuffers should fail when flag is false
    bool result = orchestrator.initializeBuffers(64);
    EXPECT_FALSE(result) << "Should fail when use_graph_buffer_management=false";
    EXPECT_FALSE(orchestrator.hasGraphManagedBuffers());
}

TEST_F(Test__DeviceGraphOrchestratorBufferManagement, ReleaseWithoutInitIsNoOp)
{
    auto config = createMinimalConfig();
    config.use_graph_buffer_management = true;

    auto graph_builder = std::make_shared<Qwen2Graph>(config, mpi_ctx_);
    DeviceGraphOrchestrator orchestrator(graph_builder, mpi_ctx_);
    // Don't initialize

    // Should not crash
    orchestrator.releaseBuffers();
    EXPECT_FALSE(orchestrator.hasGraphManagedBuffers());
}
