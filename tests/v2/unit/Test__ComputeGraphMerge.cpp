/**
 * @file Test__ComputeGraphMerge.cpp
 * @brief Unit tests for ComputeGraph merge, getRootNodes, and getLeafNodes
 * @author David Sanftenberg
 * @date December 2025
 *
 * Tests the graph merging functionality added for Phase 3 of the
 * declarative Qwen2Graph refactoring project.
 */

#include <gtest/gtest.h>
#include "../../src/v2/execution/local_execution/graph/GraphExecutor.h"
#include "execution/compute_stages/ComputeStages.h"
#include <algorithm>
#include <unordered_set>

using namespace llaminar2;

// =============================================================================
// Test Stage Implementation
// =============================================================================

/**
 * @brief Simple no-op stage for testing graph structure
 */
class NoOpStage : public IComputeStage
{
public:
    explicit NoOpStage(std::string name, DeviceId device = DeviceId::cpu())
        : IComputeStage(device), name_(std::move(name)) {}

    bool execute(IDeviceContext *) override { return true; }
    std::string name() const override { return name_; }
    size_t estimatedFlops() const override { return 0; }
    StageBufferRequirements getBufferRequirements() const override { return {}; }
    ComputeStageType type() const override { return ComputeStageType::COPY; }
    bool supportsBackend(ComputeBackendType) const override { return true; }
    StageDumpInfo buildDumpInfoImpl() const override { return {}; }

private:
    std::string name_;
};

// =============================================================================
// Test Fixture
// =============================================================================

class Test__ComputeGraphMerge : public ::testing::Test
{
protected:
    /**
     * @brief Create a simple test stage (no-op)
     */
    std::unique_ptr<IComputeStage> createTestStage(const std::string &name)
    {
        return std::make_unique<NoOpStage>(name);
    }

    /**
     * @brief Check if a vector contains a specific string
     */
    bool contains(const std::vector<std::string> &vec, const std::string &value)
    {
        return std::find(vec.begin(), vec.end(), value) != vec.end();
    }

    /**
     * @brief Check if two vectors have the same elements (order-independent)
     */
    bool sameElements(const std::vector<std::string> &a, const std::vector<std::string> &b)
    {
        if (a.size() != b.size())
            return false;
        std::unordered_set<std::string> set_a(a.begin(), a.end());
        std::unordered_set<std::string> set_b(b.begin(), b.end());
        return set_a == set_b;
    }
};

// =============================================================================
// getRootNodes Tests
// =============================================================================

TEST_F(Test__ComputeGraphMerge, GetRootNodes_EmptyGraph)
{
    ComputeGraph graph;
    auto roots = graph.getRootNodes();
    EXPECT_TRUE(roots.empty());
}

TEST_F(Test__ComputeGraphMerge, GetRootNodes_SingleNode)
{
    ComputeGraph graph;
    graph.addNode("A", createTestStage("A"), DeviceId::cpu());

    auto roots = graph.getRootNodes();
    ASSERT_EQ(roots.size(), 1);
    EXPECT_EQ(roots[0], "A");
}

TEST_F(Test__ComputeGraphMerge, GetRootNodes_LinearChain)
{
    // A -> B -> C
    ComputeGraph graph;
    graph.addNode("A", createTestStage("A"), DeviceId::cpu());
    graph.addNode("B", createTestStage("B"), DeviceId::cpu());
    graph.addNode("C", createTestStage("C"), DeviceId::cpu());
    graph.addDependency("B", "A");
    graph.addDependency("C", "B");

    auto roots = graph.getRootNodes();
    ASSERT_EQ(roots.size(), 1);
    EXPECT_EQ(roots[0], "A");
}

TEST_F(Test__ComputeGraphMerge, GetRootNodes_MultipleRoots)
{
    // A -> C
    // B -> C
    ComputeGraph graph;
    graph.addNode("A", createTestStage("A"), DeviceId::cpu());
    graph.addNode("B", createTestStage("B"), DeviceId::cpu());
    graph.addNode("C", createTestStage("C"), DeviceId::cpu());
    graph.addDependency("C", "A");
    graph.addDependency("C", "B");

    auto roots = graph.getRootNodes();
    ASSERT_EQ(roots.size(), 2);
    EXPECT_TRUE(contains(roots, "A"));
    EXPECT_TRUE(contains(roots, "B"));
}

TEST_F(Test__ComputeGraphMerge, GetRootNodes_AllRoots)
{
    // No dependencies - all nodes are roots
    ComputeGraph graph;
    graph.addNode("A", createTestStage("A"), DeviceId::cpu());
    graph.addNode("B", createTestStage("B"), DeviceId::cpu());
    graph.addNode("C", createTestStage("C"), DeviceId::cpu());

    auto roots = graph.getRootNodes();
    ASSERT_EQ(roots.size(), 3);
    EXPECT_TRUE(contains(roots, "A"));
    EXPECT_TRUE(contains(roots, "B"));
    EXPECT_TRUE(contains(roots, "C"));
}

TEST_F(Test__ComputeGraphMerge, GetRootNodes_DiamondPattern)
{
    //     A
    //    / \
    //   B   C
    //    \ /
    //     D
    ComputeGraph graph;
    graph.addNode("A", createTestStage("A"), DeviceId::cpu());
    graph.addNode("B", createTestStage("B"), DeviceId::cpu());
    graph.addNode("C", createTestStage("C"), DeviceId::cpu());
    graph.addNode("D", createTestStage("D"), DeviceId::cpu());
    graph.addDependency("B", "A");
    graph.addDependency("C", "A");
    graph.addDependency("D", "B");
    graph.addDependency("D", "C");

    auto roots = graph.getRootNodes();
    ASSERT_EQ(roots.size(), 1);
    EXPECT_EQ(roots[0], "A");
}

// =============================================================================
// getLeafNodes Tests
// =============================================================================

TEST_F(Test__ComputeGraphMerge, GetLeafNodes_EmptyGraph)
{
    ComputeGraph graph;
    auto leaves = graph.getLeafNodes();
    EXPECT_TRUE(leaves.empty());
}

TEST_F(Test__ComputeGraphMerge, GetLeafNodes_SingleNode)
{
    ComputeGraph graph;
    graph.addNode("A", createTestStage("A"), DeviceId::cpu());

    auto leaves = graph.getLeafNodes();
    ASSERT_EQ(leaves.size(), 1);
    EXPECT_EQ(leaves[0], "A");
}

TEST_F(Test__ComputeGraphMerge, GetLeafNodes_LinearChain)
{
    // A -> B -> C
    ComputeGraph graph;
    graph.addNode("A", createTestStage("A"), DeviceId::cpu());
    graph.addNode("B", createTestStage("B"), DeviceId::cpu());
    graph.addNode("C", createTestStage("C"), DeviceId::cpu());
    graph.addDependency("B", "A");
    graph.addDependency("C", "B");

    auto leaves = graph.getLeafNodes();
    ASSERT_EQ(leaves.size(), 1);
    EXPECT_EQ(leaves[0], "C");
}

TEST_F(Test__ComputeGraphMerge, GetLeafNodes_MultipleLeaves)
{
    // A -> B
    // A -> C
    ComputeGraph graph;
    graph.addNode("A", createTestStage("A"), DeviceId::cpu());
    graph.addNode("B", createTestStage("B"), DeviceId::cpu());
    graph.addNode("C", createTestStage("C"), DeviceId::cpu());
    graph.addDependency("B", "A");
    graph.addDependency("C", "A");

    auto leaves = graph.getLeafNodes();
    ASSERT_EQ(leaves.size(), 2);
    EXPECT_TRUE(contains(leaves, "B"));
    EXPECT_TRUE(contains(leaves, "C"));
}

TEST_F(Test__ComputeGraphMerge, GetLeafNodes_AllLeaves)
{
    // No dependencies - all nodes are leaves
    ComputeGraph graph;
    graph.addNode("A", createTestStage("A"), DeviceId::cpu());
    graph.addNode("B", createTestStage("B"), DeviceId::cpu());
    graph.addNode("C", createTestStage("C"), DeviceId::cpu());

    auto leaves = graph.getLeafNodes();
    ASSERT_EQ(leaves.size(), 3);
    EXPECT_TRUE(contains(leaves, "A"));
    EXPECT_TRUE(contains(leaves, "B"));
    EXPECT_TRUE(contains(leaves, "C"));
}

TEST_F(Test__ComputeGraphMerge, GetLeafNodes_DiamondPattern)
{
    //     A
    //    / \
    //   B   C
    //    \ /
    //     D
    ComputeGraph graph;
    graph.addNode("A", createTestStage("A"), DeviceId::cpu());
    graph.addNode("B", createTestStage("B"), DeviceId::cpu());
    graph.addNode("C", createTestStage("C"), DeviceId::cpu());
    graph.addNode("D", createTestStage("D"), DeviceId::cpu());
    graph.addDependency("B", "A");
    graph.addDependency("C", "A");
    graph.addDependency("D", "B");
    graph.addDependency("D", "C");

    auto leaves = graph.getLeafNodes();
    ASSERT_EQ(leaves.size(), 1);
    EXPECT_EQ(leaves[0], "D");
}

// =============================================================================
// merge Tests - Basic Operations
// =============================================================================

TEST_F(Test__ComputeGraphMerge, Merge_EmptyIntoNonEmpty)
{
    ComputeGraph target;
    target.addNode("A", createTestStage("A"), DeviceId::cpu());
    target.addNode("B", createTestStage("B"), DeviceId::cpu());
    target.addDependency("B", "A");

    ComputeGraph source; // Empty

    target.merge(std::move(source));

    EXPECT_EQ(target.size(), 2);
    EXPECT_NE(target.getNode("A"), nullptr);
    EXPECT_NE(target.getNode("B"), nullptr);
}

TEST_F(Test__ComputeGraphMerge, Merge_NonEmptyIntoEmpty)
{
    ComputeGraph target; // Empty

    ComputeGraph source;
    source.addNode("X", createTestStage("X"), DeviceId::cpu());
    source.addNode("Y", createTestStage("Y"), DeviceId::cpu());
    source.addDependency("Y", "X");

    target.merge(std::move(source));

    EXPECT_EQ(target.size(), 2);
    EXPECT_NE(target.getNode("X"), nullptr);
    EXPECT_NE(target.getNode("Y"), nullptr);

    // Source should be empty after move
    EXPECT_EQ(source.size(), 0);
}

TEST_F(Test__ComputeGraphMerge, Merge_TwoGraphsWithoutConnect)
{
    // Target: A -> B
    ComputeGraph target;
    target.addNode("A", createTestStage("A"), DeviceId::cpu());
    target.addNode("B", createTestStage("B"), DeviceId::cpu());
    target.addDependency("B", "A");

    // Source: X -> Y
    ComputeGraph source;
    source.addNode("X", createTestStage("X"), DeviceId::cpu());
    source.addNode("Y", createTestStage("Y"), DeviceId::cpu());
    source.addDependency("Y", "X");

    target.merge(std::move(source));

    EXPECT_EQ(target.size(), 4);
    EXPECT_NE(target.getNode("A"), nullptr);
    EXPECT_NE(target.getNode("B"), nullptr);
    EXPECT_NE(target.getNode("X"), nullptr);
    EXPECT_NE(target.getNode("Y"), nullptr);

    // X should still have no dependencies (no connect_from specified)
    auto roots = target.getRootNodes();
    EXPECT_EQ(roots.size(), 2); // A and X are both roots
    EXPECT_TRUE(contains(roots, "A"));
    EXPECT_TRUE(contains(roots, "X"));
}

TEST_F(Test__ComputeGraphMerge, Merge_WithConnectFrom)
{
    // Target: A -> B
    ComputeGraph target;
    target.addNode("A", createTestStage("A"), DeviceId::cpu());
    target.addNode("B", createTestStage("B"), DeviceId::cpu());
    target.addDependency("B", "A");

    // Source: X -> Y
    ComputeGraph source;
    source.addNode("X", createTestStage("X"), DeviceId::cpu());
    source.addNode("Y", createTestStage("Y"), DeviceId::cpu());
    source.addDependency("Y", "X");

    // Connect source to B
    target.merge(std::move(source), "B");

    EXPECT_EQ(target.size(), 4);

    // X should now depend on B
    auto *node_x = target.getNode("X");
    ASSERT_NE(node_x, nullptr);
    EXPECT_TRUE(contains(node_x->dependencies, "B"));

    // Only A should be a root now
    auto roots = target.getRootNodes();
    EXPECT_EQ(roots.size(), 1);
    EXPECT_EQ(roots[0], "A");

    // Y should be the only leaf
    auto leaves = target.getLeafNodes();
    EXPECT_EQ(leaves.size(), 1);
    EXPECT_EQ(leaves[0], "Y");
}

TEST_F(Test__ComputeGraphMerge, Merge_ConnectFromNonExistent)
{
    // If connect_from doesn't exist, roots should remain unconnected
    ComputeGraph target;
    target.addNode("A", createTestStage("A"), DeviceId::cpu());

    ComputeGraph source;
    source.addNode("X", createTestStage("X"), DeviceId::cpu());

    // "nonexistent" doesn't exist in target
    target.merge(std::move(source), "nonexistent");

    EXPECT_EQ(target.size(), 2);

    // X should have no dependencies (connect failed silently)
    auto *node_x = target.getNode("X");
    ASSERT_NE(node_x, nullptr);
    EXPECT_TRUE(node_x->dependencies.empty());

    // Both A and X should be roots
    auto roots = target.getRootNodes();
    EXPECT_EQ(roots.size(), 2);
}

// =============================================================================
// merge Tests - Edge Cases
// =============================================================================

TEST_F(Test__ComputeGraphMerge, Merge_NodeNameCollision)
{
    // Target has node "A"
    ComputeGraph target;
    target.addNode("A", createTestStage("target_A"), DeviceId::cpu());
    target.addNode("B", createTestStage("B"), DeviceId::cpu());
    target.addDependency("B", "A");

    // Source also has node "A" (collision!)
    ComputeGraph source;
    source.addNode("A", createTestStage("source_A"), DeviceId::cpu()); // Duplicate name
    source.addNode("C", createTestStage("C"), DeviceId::cpu());
    source.addDependency("C", "A");

    target.merge(std::move(source));

    // Should have 3 nodes (duplicate "A" skipped)
    EXPECT_EQ(target.size(), 3);
    EXPECT_NE(target.getNode("A"), nullptr);
    EXPECT_NE(target.getNode("B"), nullptr);
    EXPECT_NE(target.getNode("C"), nullptr);

    // The "A" node should be the original target's node
    // (We can't easily verify the stage content, but size check confirms skip)
}

TEST_F(Test__ComputeGraphMerge, Merge_PreservesInternalDependencies)
{
    // Target: A
    ComputeGraph target;
    target.addNode("A", createTestStage("A"), DeviceId::cpu());

    // Source: X -> Y -> Z (chain)
    ComputeGraph source;
    source.addNode("X", createTestStage("X"), DeviceId::cpu());
    source.addNode("Y", createTestStage("Y"), DeviceId::cpu());
    source.addNode("Z", createTestStage("Z"), DeviceId::cpu());
    source.addDependency("Y", "X");
    source.addDependency("Z", "Y");

    target.merge(std::move(source), "A");

    EXPECT_EQ(target.size(), 4);

    // X depends on A (from connect_from)
    auto *node_x = target.getNode("X");
    ASSERT_NE(node_x, nullptr);
    EXPECT_TRUE(contains(node_x->dependencies, "A"));

    // Y still depends on X (internal dependency preserved)
    auto *node_y = target.getNode("Y");
    ASSERT_NE(node_y, nullptr);
    EXPECT_TRUE(contains(node_y->dependencies, "X"));

    // Z still depends on Y (internal dependency preserved)
    auto *node_z = target.getNode("Z");
    ASSERT_NE(node_z, nullptr);
    EXPECT_TRUE(contains(node_z->dependencies, "Y"));
}

TEST_F(Test__ComputeGraphMerge, Merge_MultipleRootsInSource)
{
    // Target: A -> B
    ComputeGraph target;
    target.addNode("A", createTestStage("A"), DeviceId::cpu());
    target.addNode("B", createTestStage("B"), DeviceId::cpu());
    target.addDependency("B", "A");

    // Source: X and Y are both roots, both connect to Z
    ComputeGraph source;
    source.addNode("X", createTestStage("X"), DeviceId::cpu());
    source.addNode("Y", createTestStage("Y"), DeviceId::cpu());
    source.addNode("Z", createTestStage("Z"), DeviceId::cpu());
    source.addDependency("Z", "X");
    source.addDependency("Z", "Y");

    // Connect to B - both X and Y should depend on B
    target.merge(std::move(source), "B");

    EXPECT_EQ(target.size(), 5);

    auto *node_x = target.getNode("X");
    auto *node_y = target.getNode("Y");
    ASSERT_NE(node_x, nullptr);
    ASSERT_NE(node_y, nullptr);

    EXPECT_TRUE(contains(node_x->dependencies, "B"));
    EXPECT_TRUE(contains(node_y->dependencies, "B"));

    // Only A should be a root now
    auto roots = target.getRootNodes();
    EXPECT_EQ(roots.size(), 1);
    EXPECT_EQ(roots[0], "A");
}

TEST_F(Test__ComputeGraphMerge, Merge_SequentialMerges)
{
    // Target: A
    ComputeGraph target;
    target.addNode("A", createTestStage("A"), DeviceId::cpu());

    // First merge: B -> C
    ComputeGraph source1;
    source1.addNode("B", createTestStage("B"), DeviceId::cpu());
    source1.addNode("C", createTestStage("C"), DeviceId::cpu());
    source1.addDependency("C", "B");

    target.merge(std::move(source1), "A");

    // Second merge: D -> E
    ComputeGraph source2;
    source2.addNode("D", createTestStage("D"), DeviceId::cpu());
    source2.addNode("E", createTestStage("E"), DeviceId::cpu());
    source2.addDependency("E", "D");

    target.merge(std::move(source2), "C");

    EXPECT_EQ(target.size(), 5);

    // Check chain: A -> B -> C -> D -> E
    auto *node_b = target.getNode("B");
    auto *node_d = target.getNode("D");
    ASSERT_NE(node_b, nullptr);
    ASSERT_NE(node_d, nullptr);

    EXPECT_TRUE(contains(node_b->dependencies, "A"));
    EXPECT_TRUE(contains(node_d->dependencies, "C"));

    auto roots = target.getRootNodes();
    EXPECT_EQ(roots.size(), 1);
    EXPECT_EQ(roots[0], "A");

    auto leaves = target.getLeafNodes();
    EXPECT_EQ(leaves.size(), 1);
    EXPECT_EQ(leaves[0], "E");
}

TEST_F(Test__ComputeGraphMerge, Merge_ChainReturnsThis)
{
    // Verify merge returns *this for chaining
    ComputeGraph target;
    target.addNode("A", createTestStage("A"), DeviceId::cpu());

    ComputeGraph source1;
    source1.addNode("B", createTestStage("B"), DeviceId::cpu());

    ComputeGraph source2;
    source2.addNode("C", createTestStage("C"), DeviceId::cpu());

    // Chain merges
    target.merge(std::move(source1), "A")
        .merge(std::move(source2), "B");

    EXPECT_EQ(target.size(), 3);
    EXPECT_NE(target.getNode("A"), nullptr);
    EXPECT_NE(target.getNode("B"), nullptr);
    EXPECT_NE(target.getNode("C"), nullptr);
}

// =============================================================================
// merge Tests - Execution Order Verification
// =============================================================================

TEST_F(Test__ComputeGraphMerge, Merge_ExecutionOrderCorrect)
{
    // Verify that merged graphs produce correct execution order
    // Target: A -> B
    ComputeGraph target;
    target.addNode("A", createTestStage("A"), DeviceId::cpu());
    target.addNode("B", createTestStage("B"), DeviceId::cpu());
    target.addDependency("B", "A");

    // Source: X -> Y
    ComputeGraph source;
    source.addNode("X", createTestStage("X"), DeviceId::cpu());
    source.addNode("Y", createTestStage("Y"), DeviceId::cpu());
    source.addDependency("Y", "X");

    // Connect: A -> B -> X -> Y
    target.merge(std::move(source), "B");

    auto order = target.getExecutionOrder();
    ASSERT_EQ(order.size(), 4);

    // Find positions
    auto pos_a = std::find(order.begin(), order.end(), "A") - order.begin();
    auto pos_b = std::find(order.begin(), order.end(), "B") - order.begin();
    auto pos_x = std::find(order.begin(), order.end(), "X") - order.begin();
    auto pos_y = std::find(order.begin(), order.end(), "Y") - order.begin();

    // Verify ordering constraints
    EXPECT_LT(pos_a, pos_b); // A before B
    EXPECT_LT(pos_b, pos_x); // B before X
    EXPECT_LT(pos_x, pos_y); // X before Y
}

TEST_F(Test__ComputeGraphMerge, Merge_ComplexDAGExecutionOrder)
{
    // Build a more complex DAG:
    // Target:
    //     A
    //    / \
    //   B   C
    //    \ /
    //     D

    ComputeGraph target;
    target.addNode("A", createTestStage("A"), DeviceId::cpu());
    target.addNode("B", createTestStage("B"), DeviceId::cpu());
    target.addNode("C", createTestStage("C"), DeviceId::cpu());
    target.addNode("D", createTestStage("D"), DeviceId::cpu());
    target.addDependency("B", "A");
    target.addDependency("C", "A");
    target.addDependency("D", "B");
    target.addDependency("D", "C");

    // Source: E -> F
    ComputeGraph source;
    source.addNode("E", createTestStage("E"), DeviceId::cpu());
    source.addNode("F", createTestStage("F"), DeviceId::cpu());
    source.addDependency("F", "E");

    // Connect after D
    target.merge(std::move(source), "D");

    auto order = target.getExecutionOrder();
    ASSERT_EQ(order.size(), 6);

    auto pos_a = std::find(order.begin(), order.end(), "A") - order.begin();
    auto pos_b = std::find(order.begin(), order.end(), "B") - order.begin();
    auto pos_c = std::find(order.begin(), order.end(), "C") - order.begin();
    auto pos_d = std::find(order.begin(), order.end(), "D") - order.begin();
    auto pos_e = std::find(order.begin(), order.end(), "E") - order.begin();
    auto pos_f = std::find(order.begin(), order.end(), "F") - order.begin();

    // A must come before B and C
    EXPECT_LT(pos_a, pos_b);
    EXPECT_LT(pos_a, pos_c);

    // B and C must come before D
    EXPECT_LT(pos_b, pos_d);
    EXPECT_LT(pos_c, pos_d);

    // D must come before E
    EXPECT_LT(pos_d, pos_e);

    // E must come before F
    EXPECT_LT(pos_e, pos_f);
}

// =============================================================================
// Integration Test - Simulating Layer Graph Merging
// =============================================================================

TEST_F(Test__ComputeGraphMerge, SimulateLayerGraphMerging)
{
    // Simulate how buildFullForwardGraph uses merge for transformer layers

    // Main graph: embedding
    ComputeGraph main_graph;
    main_graph.addNode("embedding", createTestStage("embedding"), DeviceId::cpu());

    std::string prev_node = "embedding";

    // Simulate 3 layers
    for (int layer = 0; layer < 3; ++layer)
    {
        std::string prefix = "layer" + std::to_string(layer) + "_";

        // Build "attention" subgraph
        ComputeGraph attn_graph;
        attn_graph.addNode(prefix + "attn_norm", createTestStage(prefix + "attn_norm"), DeviceId::cpu());
        attn_graph.addNode(prefix + "qkv_proj", createTestStage(prefix + "qkv_proj"), DeviceId::cpu());
        attn_graph.addNode(prefix + "attention", createTestStage(prefix + "attention"), DeviceId::cpu());
        attn_graph.addNode(prefix + "wo_proj", createTestStage(prefix + "wo_proj"), DeviceId::cpu());
        attn_graph.addNode(prefix + "attn_residual", createTestStage(prefix + "attn_residual"), DeviceId::cpu());

        attn_graph.addDependency(prefix + "qkv_proj", prefix + "attn_norm");
        attn_graph.addDependency(prefix + "attention", prefix + "qkv_proj");
        attn_graph.addDependency(prefix + "wo_proj", prefix + "attention");
        attn_graph.addDependency(prefix + "attn_residual", prefix + "wo_proj");

        // Merge attention into main
        main_graph.merge(std::move(attn_graph), prev_node);
        prev_node = prefix + "attn_residual";

        // Build "FFN" subgraph
        ComputeGraph ffn_graph;
        ffn_graph.addNode(prefix + "ffn_norm", createTestStage(prefix + "ffn_norm"), DeviceId::cpu());
        ffn_graph.addNode(prefix + "gate_up", createTestStage(prefix + "gate_up"), DeviceId::cpu());
        ffn_graph.addNode(prefix + "swiglu", createTestStage(prefix + "swiglu"), DeviceId::cpu());
        ffn_graph.addNode(prefix + "down_proj", createTestStage(prefix + "down_proj"), DeviceId::cpu());
        ffn_graph.addNode(prefix + "ffn_residual", createTestStage(prefix + "ffn_residual"), DeviceId::cpu());

        ffn_graph.addDependency(prefix + "gate_up", prefix + "ffn_norm");
        ffn_graph.addDependency(prefix + "swiglu", prefix + "gate_up");
        ffn_graph.addDependency(prefix + "down_proj", prefix + "swiglu");
        ffn_graph.addDependency(prefix + "ffn_residual", prefix + "down_proj");

        // Merge FFN into main
        main_graph.merge(std::move(ffn_graph), prev_node);
        prev_node = prefix + "ffn_residual";
    }

    // Add LM head
    main_graph.addNode("final_norm", createTestStage("final_norm"), DeviceId::cpu());
    main_graph.addNode("lm_head", createTestStage("lm_head"), DeviceId::cpu());
    main_graph.addDependency("final_norm", prev_node);
    main_graph.addDependency("lm_head", "final_norm");

    // Verify structure
    // 1 embedding + 3 layers * (5 attn + 5 ffn) + 2 final = 33 nodes
    EXPECT_EQ(main_graph.size(), 33);

    // Only embedding should be a root
    auto roots = main_graph.getRootNodes();
    EXPECT_EQ(roots.size(), 1);
    EXPECT_EQ(roots[0], "embedding");

    // Only lm_head should be a leaf
    auto leaves = main_graph.getLeafNodes();
    EXPECT_EQ(leaves.size(), 1);
    EXPECT_EQ(leaves[0], "lm_head");

    // Verify execution order is valid
    auto order = main_graph.getExecutionOrder();
    EXPECT_EQ(order.size(), 33);

    // embedding must be first
    EXPECT_EQ(order[0], "embedding");

    // lm_head must be last
    EXPECT_EQ(order[32], "lm_head");
}
