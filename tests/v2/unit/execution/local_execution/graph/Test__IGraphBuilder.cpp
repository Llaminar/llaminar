/**
 * @file Test__IGraphBuilder.cpp
 * @brief Unit tests for IGraphBuilder interface and MockGraphBuilder
 * @author David Sanftenberg
 * @date December 19, 2025
 */

#include <gtest/gtest.h>
#include "execution/local_execution/graph/IGraphBuilder.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/compute_stages/ComputeStages.h"

using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__IGraphBuilder : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mock_ = std::make_shared<MockGraphBuilder>();
    }

    std::shared_ptr<MockGraphBuilder> mock_;
};

// =============================================================================
// MockGraphBuilder Basic Tests
// =============================================================================

TEST_F(Test__IGraphBuilder, MockGraphBuilder_DefaultValues)
{
    EXPECT_EQ(mock_->numLayers(), 24);
    EXPECT_EQ(mock_->hiddenDim(), 896);
    EXPECT_TRUE(mock_->isInitialized());
}

TEST_F(Test__IGraphBuilder, MockGraphBuilder_ConfigureProperties)
{
    mock_->setNumLayers(32);
    mock_->setHiddenDim(4096);
    mock_->setInitialized(false);

    EXPECT_EQ(mock_->numLayers(), 32);
    EXPECT_EQ(mock_->hiddenDim(), 4096);
    EXPECT_FALSE(mock_->isInitialized());
}

TEST_F(Test__IGraphBuilder, MockGraphBuilder_InitialCallCounts)
{
    EXPECT_EQ(mock_->buildForwardGraphCallCount(), 0);
    EXPECT_EQ(mock_->buildLayerGraphCallCount(), 0);
    EXPECT_EQ(mock_->lastForwardInput(), nullptr);
    EXPECT_EQ(mock_->lastForwardOutput(), nullptr);
}

// =============================================================================
// buildForwardGraph Tests
// =============================================================================

TEST_F(Test__IGraphBuilder, BuildForwardGraph_ReturnsEmptyGraphByDefault)
{
    ForwardInput input;
    input.seq_len = 10;
    input.batch_size = 1;
    ForwardOutput output;

    ComputeGraph graph = mock_->buildForwardGraph(input, output);

    EXPECT_EQ(graph.size(), 0);
    EXPECT_EQ(mock_->buildForwardGraphCallCount(), 1);
}

TEST_F(Test__IGraphBuilder, BuildForwardGraph_ReturnsConfiguredMockGraph)
{
    // Set factory that creates a graph with nodes
    mock_->setForwardGraphFactory([](const ForwardInput &, ForwardOutput &)
                                  {
        ComputeGraph graph;
        graph.addNode("embedding", nullptr, DeviceId::cpu());
        graph.addNode("layer0", nullptr, DeviceId::cpu());
        graph.addNode("lm_head", nullptr, DeviceId::cpu());
        graph.addDependency("layer0", "embedding");
        graph.addDependency("lm_head", "layer0");
        return graph; });

    ForwardInput input;
    input.seq_len = 10;
    ForwardOutput output;

    ComputeGraph graph = mock_->buildForwardGraph(input, output);

    EXPECT_EQ(graph.size(), 3);
}

TEST_F(Test__IGraphBuilder, BuildForwardGraph_RecordsInput)
{
    ForwardInput input;
    input.seq_len = 128;
    input.batch_size = 4;
    input.device = DeviceId::cuda(0);
    ForwardOutput output;

    mock_->buildForwardGraph(input, output);

    ASSERT_NE(mock_->lastForwardInput(), nullptr);
    EXPECT_EQ(mock_->lastForwardInput()->seq_len, 128);
    EXPECT_EQ(mock_->lastForwardInput()->batch_size, 4);
    EXPECT_EQ(mock_->lastForwardInput()->device, DeviceId::cuda(0));
}

TEST_F(Test__IGraphBuilder, BuildForwardGraph_RecordsOutput)
{
    ForwardInput input;
    ForwardOutput output;
    float dummy_logits = 0.0f;
    output.logits = reinterpret_cast<TensorBase *>(&dummy_logits); // Use dummy pointer

    mock_->buildForwardGraph(input, output);

    ASSERT_NE(mock_->lastForwardOutput(), nullptr);
    EXPECT_EQ(mock_->lastForwardOutput()->logits, output.logits);
}

TEST_F(Test__IGraphBuilder, BuildForwardGraph_MultipleCallsTracked)
{
    ForwardInput input;
    ForwardOutput output;

    mock_->buildForwardGraph(input, output);
    mock_->buildForwardGraph(input, output);
    mock_->buildForwardGraph(input, output);

    EXPECT_EQ(mock_->buildForwardGraphCallCount(), 3);
}

TEST_F(Test__IGraphBuilder, BuildForwardGraph_WithFactory)
{
    int factory_call_count = 0;

    mock_->setForwardGraphFactory([&](const ForwardInput &input, ForwardOutput &output)
                                  {
        ++factory_call_count;
        ComputeGraph graph;
        // Create nodes based on input
        for (int i = 0; i < input.seq_len; ++i)
        {
            graph.addNode("token_" + std::to_string(i), nullptr, DeviceId::cpu());
        }
        return graph; });

    ForwardInput input;
    input.seq_len = 5;
    ForwardOutput output;

    ComputeGraph graph = mock_->buildForwardGraph(input, output);

    EXPECT_EQ(factory_call_count, 1);
    EXPECT_EQ(graph.size(), 5);
}

// =============================================================================
// buildLayerGraph Tests
// =============================================================================

TEST_F(Test__IGraphBuilder, BuildLayerGraph_ReturnsEmptyGraphByDefault)
{
    LayerContext ctx;
    ctx.layer_idx = 0;
    ctx.seq_len = 10;

    ComputeGraph graph = mock_->buildLayerGraph(ctx);

    EXPECT_EQ(graph.size(), 0);
    EXPECT_EQ(mock_->buildLayerGraphCallCount(), 1);
}

TEST_F(Test__IGraphBuilder, BuildLayerGraph_ReturnsConfiguredMockGraph)
{
    mock_->setLayerGraphFactory([](const LayerContext &)
                                {
        ComputeGraph graph;
        graph.addNode("attn", nullptr, DeviceId::cpu());
        graph.addNode("ffn", nullptr, DeviceId::cpu());
        graph.addDependency("ffn", "attn");
        return graph; });

    LayerContext ctx;
    ctx.layer_idx = 5;

    ComputeGraph graph = mock_->buildLayerGraph(ctx);

    EXPECT_EQ(graph.size(), 2);
}

TEST_F(Test__IGraphBuilder, BuildLayerGraph_ReturnsPerLayerMockGraph)
{
    // Set different factories for different layers
    mock_->setLayerGraphFactory(0, [](const LayerContext &)
                                {
        ComputeGraph graph;
        graph.addNode("layer0_attn", nullptr, DeviceId::cpu());
        return graph; });

    mock_->setLayerGraphFactory(1, [](const LayerContext &)
                                {
        ComputeGraph graph;
        graph.addNode("layer1_attn", nullptr, DeviceId::cpu());
        graph.addNode("layer1_ffn", nullptr, DeviceId::cpu());
        graph.addDependency("layer1_ffn", "layer1_attn");
        return graph; });

    LayerContext ctx0;
    ctx0.layer_idx = 0;
    ComputeGraph graph0 = mock_->buildLayerGraph(ctx0);
    EXPECT_EQ(graph0.size(), 1);

    LayerContext ctx1;
    ctx1.layer_idx = 1;
    ComputeGraph graph1 = mock_->buildLayerGraph(ctx1);
    EXPECT_EQ(graph1.size(), 2);
}

TEST_F(Test__IGraphBuilder, BuildLayerGraph_RecordsContext)
{
    LayerContext ctx;
    ctx.layer_idx = 7;
    ctx.seq_len = 64;
    ctx.device = DeviceId::cuda(1);

    mock_->buildLayerGraph(ctx);

    EXPECT_EQ(mock_->lastLayerContext().layer_idx, 7);
    EXPECT_EQ(mock_->lastLayerContext().seq_len, 64);
    EXPECT_EQ(mock_->lastLayerContext().device, DeviceId::cuda(1));
}

TEST_F(Test__IGraphBuilder, BuildLayerGraph_MultipleCallsTracked)
{
    LayerContext ctx;

    for (int i = 0; i < 24; ++i)
    {
        ctx.layer_idx = i;
        mock_->buildLayerGraph(ctx);
    }

    EXPECT_EQ(mock_->buildLayerGraphCallCount(), 24);
}

TEST_F(Test__IGraphBuilder, BuildLayerGraph_WithFactory)
{
    mock_->setLayerGraphFactory([](const LayerContext &ctx)
                                {
        ComputeGraph graph;
        std::string prefix = "layer" + std::to_string(ctx.layer_idx) + "_";
        graph.addNode(prefix + "attn_norm", nullptr, ctx.device);
        graph.addNode(prefix + "attn", nullptr, ctx.device);
        graph.addNode(prefix + "ffn_norm", nullptr, ctx.device);
        graph.addNode(prefix + "ffn", nullptr, ctx.device);
        graph.addDependency(prefix + "attn", prefix + "attn_norm");
        graph.addDependency(prefix + "ffn_norm", prefix + "attn");
        graph.addDependency(prefix + "ffn", prefix + "ffn_norm");
        return graph; });

    LayerContext ctx;
    ctx.layer_idx = 3;
    ctx.device = DeviceId::cuda(0);

    ComputeGraph graph = mock_->buildLayerGraph(ctx);

    EXPECT_EQ(graph.size(), 4);
    auto order = graph.getExecutionOrder();
    EXPECT_EQ(order[0], "layer3_attn_norm");
    EXPECT_EQ(order[3], "layer3_ffn");
}

// =============================================================================
// Reset and State Management Tests
// =============================================================================

TEST_F(Test__IGraphBuilder, ResetCallCounts_ClearsAll)
{
    ForwardInput input;
    ForwardOutput output;
    LayerContext ctx;

    mock_->buildForwardGraph(input, output);
    mock_->buildLayerGraph(ctx);

    EXPECT_EQ(mock_->buildForwardGraphCallCount(), 1);
    EXPECT_EQ(mock_->buildLayerGraphCallCount(), 1);
    EXPECT_NE(mock_->lastForwardInput(), nullptr);

    mock_->resetCallCounts();

    EXPECT_EQ(mock_->buildForwardGraphCallCount(), 0);
    EXPECT_EQ(mock_->buildLayerGraphCallCount(), 0);
    EXPECT_EQ(mock_->lastForwardInput(), nullptr);
    EXPECT_EQ(mock_->lastForwardOutput(), nullptr);
}

// =============================================================================
// BuildPositionIds Static Utility Tests
// =============================================================================

TEST_F(Test__IGraphBuilder, BuildPositionIds_SingleBatch)
{
    auto pos_ids = IGraphBuilder::buildPositionIds(10, 1, 0);

    EXPECT_EQ(pos_ids.size(), 10u);
    for (int i = 0; i < 10; ++i)
    {
        EXPECT_EQ(pos_ids[i], i);
    }
}

TEST_F(Test__IGraphBuilder, BuildPositionIds_WithOffset)
{
    auto pos_ids = IGraphBuilder::buildPositionIds(5, 1, 100);

    EXPECT_EQ(pos_ids.size(), 5u);
    EXPECT_EQ(pos_ids[0], 100);
    EXPECT_EQ(pos_ids[4], 104);
}

TEST_F(Test__IGraphBuilder, BuildPositionIds_MultipleBatches)
{
    auto pos_ids = IGraphBuilder::buildPositionIds(3, 2, 0);

    EXPECT_EQ(pos_ids.size(), 6u);
    // Batch 0
    EXPECT_EQ(pos_ids[0], 0);
    EXPECT_EQ(pos_ids[1], 1);
    EXPECT_EQ(pos_ids[2], 2);
    // Batch 1
    EXPECT_EQ(pos_ids[3], 0);
    EXPECT_EQ(pos_ids[4], 1);
    EXPECT_EQ(pos_ids[5], 2);
}

TEST_F(Test__IGraphBuilder, BuildPositionIds_MultipleBatchesWithOffset)
{
    auto pos_ids = IGraphBuilder::buildPositionIds(2, 3, 50);

    EXPECT_EQ(pos_ids.size(), 6u);
    // All batches start at offset
    EXPECT_EQ(pos_ids[0], 50);
    EXPECT_EQ(pos_ids[1], 51);
    EXPECT_EQ(pos_ids[2], 50);
    EXPECT_EQ(pos_ids[3], 51);
    EXPECT_EQ(pos_ids[4], 50);
    EXPECT_EQ(pos_ids[5], 51);
}

// =============================================================================
// Polymorphism Tests
// =============================================================================

TEST_F(Test__IGraphBuilder, PolymorphicPointer_WorksCorrectly)
{
    // Use IGraphBuilder pointer to MockGraphBuilder
    std::shared_ptr<IGraphBuilder> builder = mock_;

    EXPECT_EQ(builder->numLayers(), 24);
    EXPECT_EQ(builder->hiddenDim(), 896);
    EXPECT_TRUE(builder->isInitialized());

    // Configure via mock
    mock_->setNumLayers(48);

    // Read via interface
    EXPECT_EQ(builder->numLayers(), 48);
}

TEST_F(Test__IGraphBuilder, PolymorphicBuildForwardGraph_WorksCorrectly)
{
    std::shared_ptr<IGraphBuilder> builder = mock_;

    mock_->setForwardGraphFactory([](const ForwardInput &, ForwardOutput &)
                                  {
        ComputeGraph graph;
        graph.addNode("test", nullptr, DeviceId::cpu());
        return graph; });

    ForwardInput input;
    ForwardOutput output;

    ComputeGraph graph = builder->buildForwardGraph(input, output);

    EXPECT_EQ(graph.size(), 1);
}

// =============================================================================
// ForwardInput/ForwardOutput Structure Tests
// =============================================================================

TEST_F(Test__IGraphBuilder, ForwardInput_DefaultValues)
{
    ForwardInput input;

    EXPECT_EQ(input.token_ids, nullptr);
    EXPECT_EQ(input.position_ids, nullptr);
    EXPECT_EQ(input.batch_size, 1);
    EXPECT_EQ(input.seq_len, 0);
    EXPECT_EQ(input.position_offset, 0);
    EXPECT_EQ(input.device, DeviceId::cpu());
    EXPECT_EQ(input.kv_cache, nullptr);
}

TEST_F(Test__IGraphBuilder, ForwardOutput_DefaultValues)
{
    ForwardOutput output;

    EXPECT_EQ(output.logits, nullptr);
    EXPECT_EQ(output.hidden, nullptr);
}

TEST_F(Test__IGraphBuilder, LayerContext_DefaultValues)
{
    LayerContext ctx;

    EXPECT_EQ(ctx.layer_idx, 0);
    EXPECT_EQ(ctx.seq_len, 0);
    EXPECT_EQ(ctx.device, DeviceId::cpu());
    EXPECT_EQ(ctx.position_ids, nullptr);
    EXPECT_EQ(ctx.kv_cache, nullptr);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(Test__IGraphBuilder, BuildLayerGraph_FallbackToDefault)
{
    // Set layer-specific factory only for layer 0
    mock_->setLayerGraphFactory(0, [](const LayerContext &)
                                {
        ComputeGraph graph;
        graph.addNode("specific", nullptr, DeviceId::cpu());
        return graph; });

    // Set default layer factory
    mock_->setLayerGraphFactory([](const LayerContext &)
                                {
        ComputeGraph graph;
        graph.addNode("default", nullptr, DeviceId::cpu());
        return graph; });

    // Layer 0 should use specific factory
    LayerContext ctx0;
    ctx0.layer_idx = 0;
    ComputeGraph graph0 = mock_->buildLayerGraph(ctx0);
    EXPECT_EQ(graph0.size(), 1);

    // Layer 10 should fall back to default
    LayerContext ctx10;
    ctx10.layer_idx = 10;
    ComputeGraph graph10 = mock_->buildLayerGraph(ctx10);
    EXPECT_EQ(graph10.size(), 1);
}

TEST_F(Test__IGraphBuilder, MockGraphBuilder_FactoryCalledEachTime)
{
    int factory_call_count = 0;

    mock_->setForwardGraphFactory([&](const ForwardInput &, ForwardOutput &)
                                  {
        ++factory_call_count;
        ComputeGraph graph;
        graph.addNode("node1", nullptr, DeviceId::cpu());
        graph.addNode("node2", nullptr, DeviceId::cpu());
        graph.addDependency("node2", "node1");
        return graph; });

    ForwardInput input;
    ForwardOutput output;

    // First call
    ComputeGraph graph1 = mock_->buildForwardGraph(input, output);
    EXPECT_EQ(graph1.size(), 2);
    EXPECT_EQ(factory_call_count, 1);

    // Second call - factory called again
    ComputeGraph graph2 = mock_->buildForwardGraph(input, output);
    EXPECT_EQ(graph2.size(), 2);
    EXPECT_EQ(factory_call_count, 2);
}
