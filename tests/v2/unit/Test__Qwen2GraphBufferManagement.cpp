/**
 * @file Test__Qwen2GraphBufferManagement.cpp
 * @brief Unit tests for Qwen2Graph buffer management integration
 * @author David Sanftenberg
 * @date January 2025
 *
 * Tests the GraphBufferManager integration with Qwen2Graph:
 * - setTensorFactory() / initializeBuffers() workflow
 * - Buffer allocation and binding
 * - Statistics tracking
 * - Release cleanup
 */

#include <gtest/gtest.h>
#include "../../../src/v2/pipelines/qwen/Qwen2Graph.h"
#include "../../../src/v2/tensors/TensorFactory.h"
#include "../../../src/v2/utils/MPIContext.h"

using namespace llaminar2;

// ============================================================================
// Test Fixture
// ============================================================================

class Test__Qwen2GraphBufferManagement : public ::testing::Test
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

    Qwen2GraphConfig createMinimalConfig()
    {
        // Create minimal config for Qwen2.5-0.5B-like model
        Qwen2GraphConfig config;
        config.d_model = 896;
        config.n_heads = 14;
        config.n_kv_heads = 2;
        config.head_dim = 64;
        config.d_ff = 4864;
        config.vocab_size = 151936;
        config.n_layers = 24;
        config.use_graph_buffer_management = true;
        config.max_seq_len = 128; // Small for tests
        config.default_device = 0;
        return config;
    }

    std::shared_ptr<MPIContext> mpi_ctx_;
    std::unique_ptr<TensorFactory> tensor_factory_;
};

// ============================================================================
// Basic Tests
// ============================================================================

TEST_F(Test__Qwen2GraphBufferManagement, CreateWithBufferManagementFlag)
{
    auto config = createMinimalConfig();
    config.use_graph_buffer_management = true;

    Qwen2Graph graph(config, mpi_ctx_);

    EXPECT_TRUE(config.use_graph_buffer_management);
    EXPECT_FALSE(graph.hasGraphManagedBuffers()) << "Should not have buffers until initializeBuffers()";
}

TEST_F(Test__Qwen2GraphBufferManagement, InitializeWithoutTensorFactoryFails)
{
    auto config = createMinimalConfig();
    config.use_graph_buffer_management = true;

    Qwen2Graph graph(config, mpi_ctx_);
    // Don't call setTensorFactory()

    // Should fail because TensorFactory not set
    bool result = graph.initializeBuffers(64);
    EXPECT_FALSE(result) << "Should fail without TensorFactory";
    EXPECT_FALSE(graph.hasGraphManagedBuffers());
}

TEST_F(Test__Qwen2GraphBufferManagement, InitializeBuffersSuccess)
{
    auto config = createMinimalConfig();
    config.use_graph_buffer_management = true;

    Qwen2Graph graph(config, mpi_ctx_);
    graph.setTensorFactory(tensor_factory_.get());

    bool result = graph.initializeBuffers(64);
    EXPECT_TRUE(result);
    EXPECT_TRUE(graph.hasGraphManagedBuffers());
}

TEST_F(Test__Qwen2GraphBufferManagement, BufferStatsAvailable)
{
    auto config = createMinimalConfig();
    config.use_graph_buffer_management = true;

    Qwen2Graph graph(config, mpi_ctx_);
    graph.setTensorFactory(tensor_factory_.get());

    EXPECT_EQ(graph.bufferStats(), nullptr) << "No stats before init";

    ASSERT_TRUE(graph.initializeBuffers(64));

    const auto *stats = graph.bufferStats();
    ASSERT_NE(stats, nullptr);
    EXPECT_GT(stats->total_bytes, 0u) << "Should have allocated some memory";
    EXPECT_GT(stats->total_buffers, 0u) << "Should have allocated buffers";
}

TEST_F(Test__Qwen2GraphBufferManagement, ReleaseBuffers)
{
    auto config = createMinimalConfig();
    config.use_graph_buffer_management = true;

    Qwen2Graph graph(config, mpi_ctx_);
    graph.setTensorFactory(tensor_factory_.get());

    ASSERT_TRUE(graph.initializeBuffers(64));
    EXPECT_TRUE(graph.hasGraphManagedBuffers());

    graph.releaseBuffers();
    EXPECT_FALSE(graph.hasGraphManagedBuffers());
    EXPECT_EQ(graph.bufferStats(), nullptr);
}

TEST_F(Test__Qwen2GraphBufferManagement, ReinitializeAfterRelease)
{
    auto config = createMinimalConfig();
    config.use_graph_buffer_management = true;

    Qwen2Graph graph(config, mpi_ctx_);
    graph.setTensorFactory(tensor_factory_.get());

    // First init
    ASSERT_TRUE(graph.initializeBuffers(64));
    EXPECT_TRUE(graph.hasGraphManagedBuffers());

    // Release
    graph.releaseBuffers();
    EXPECT_FALSE(graph.hasGraphManagedBuffers());

    // Re-initialize with different size
    ASSERT_TRUE(graph.initializeBuffers(128));
    EXPECT_TRUE(graph.hasGraphManagedBuffers());

    const auto *stats = graph.bufferStats();
    ASSERT_NE(stats, nullptr);
    EXPECT_GT(stats->total_bytes, 0u);
}

// ============================================================================
// Memory Allocation Tests
// ============================================================================

TEST_F(Test__Qwen2GraphBufferManagement, AllocatesReasonableMemory)
{
    auto config = createMinimalConfig();
    config.use_graph_buffer_management = true;

    Qwen2Graph graph(config, mpi_ctx_);
    graph.setTensorFactory(tensor_factory_.get());

    ASSERT_TRUE(graph.initializeBuffers(64));

    const auto *stats = graph.bufferStats();
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

TEST_F(Test__Qwen2GraphBufferManagement, LargerSeqLenUsesMoreMemory)
{
    auto config = createMinimalConfig();
    config.use_graph_buffer_management = true;

    Qwen2Graph graph1(config, mpi_ctx_);
    graph1.setTensorFactory(tensor_factory_.get());
    ASSERT_TRUE(graph1.initializeBuffers(64));
    size_t bytes_64 = graph1.bufferStats()->total_bytes;

    graph1.releaseBuffers();

    Qwen2Graph graph2(config, mpi_ctx_);
    graph2.setTensorFactory(tensor_factory_.get());
    ASSERT_TRUE(graph2.initializeBuffers(256));
    size_t bytes_256 = graph2.bufferStats()->total_bytes;

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

TEST_F(Test__Qwen2GraphBufferManagement, VerySmallSeqLen)
{
    auto config = createMinimalConfig();
    config.use_graph_buffer_management = true;

    Qwen2Graph graph(config, mpi_ctx_);
    graph.setTensorFactory(tensor_factory_.get());

    ASSERT_TRUE(graph.initializeBuffers(1));
    EXPECT_TRUE(graph.hasGraphManagedBuffers());

    const auto *stats = graph.bufferStats();
    ASSERT_NE(stats, nullptr);
    EXPECT_GT(stats->total_bytes, 0u);
}

TEST_F(Test__Qwen2GraphBufferManagement, WithoutUseGraphBufferManagementFlagFails)
{
    auto config = createMinimalConfig();
    config.use_graph_buffer_management = false; // Disabled

    Qwen2Graph graph(config, mpi_ctx_);
    graph.setTensorFactory(tensor_factory_.get());

    // initializeBuffers should fail when flag is false
    bool result = graph.initializeBuffers(64);
    EXPECT_FALSE(result) << "Should fail when use_graph_buffer_management=false";
    EXPECT_FALSE(graph.hasGraphManagedBuffers());
}

TEST_F(Test__Qwen2GraphBufferManagement, ReleaseWithoutInitIsNoOp)
{
    auto config = createMinimalConfig();
    config.use_graph_buffer_management = true;

    Qwen2Graph graph(config, mpi_ctx_);
    // Don't initialize

    // Should not crash
    graph.releaseBuffers();
    EXPECT_FALSE(graph.hasGraphManagedBuffers());
}
