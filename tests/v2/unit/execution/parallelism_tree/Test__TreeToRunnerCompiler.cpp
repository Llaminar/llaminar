/**
 * @file Test__TreeToRunnerCompiler.cpp
 * @brief Unit tests for TreeToRunnerCompiler
 * @author David Sanftenberg
 * @date February 2026
 *
 * Tests for tree-to-runner compilation:
 * - shouldCompile() logic for various node types and ranks
 * - isCrossRankPP() detection
 * - compileDevice with mock factory
 * - compileTP with mock factory
 * - compilePP (local) with mock factory
 * - compileCrossRankPP → PipelineRunner creation
 * - Nested TP+PP compilation
 * - Layer range propagation to stage configs
 *
 * All tests use mock runners — no MPI, no GPU, no model loading.
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>

#include "execution/parallelism_tree/TreeToRunnerCompiler.h"
#include "execution/parallelism_tree/ParallelismTree.h"
#include "execution/parallelism_tree/PipelineRunner.h"
#include "backends/GlobalDeviceAddress.h"
#include "config/OrchestrationConfig.h"

using namespace llaminar2;

// =============================================================================
// Mock IInferenceRunner for Testing
// =============================================================================

/**
 * @brief Mock inference runner for unit testing
 *
 * Tracks calls and allows verification without real model execution.
 */
class MockInferenceRunner : public IInferenceRunner
{
public:
    std::string name;
    int first_layer = 0;
    int last_layer = 0;
    bool has_embedding = false;
    bool has_lm_head = false;
    mutable int forward_count = 0;
    mutable int clear_cache_count = 0;
    int position = 0;
    int vocab_size_ = 128;

    explicit MockInferenceRunner(std::string name) : name(std::move(name)) {}

    bool forward(const int *tokens, int seq_len) override
    {
        (void)tokens;
        forward_count++;
        position += seq_len;
        return true;
    }

    const float *logits() const override
    {
        static std::vector<float> dummy(256, 0.0f);
        return dummy.data();
    }

    void clear_cache() override
    {
        clear_cache_count++;
        position = 0;
    }

    int get_position() const override { return position; }
    int vocab_size() const override { return vocab_size_; }
    ExecutionPath executionPath() const override { return ExecutionPath::GRAPH; }
    const char *architecture() const override { return "mock"; }
};

// =============================================================================
// Helper: Create mock factories
// =============================================================================

/**
 * @brief Create a device runner factory that tracks created runners
 */
static TreeToRunnerCompiler::DeviceRunnerFactory createDeviceFactory(
    std::vector<std::string> &created_names)
{
    return [&created_names](const ParallelismNode &node,
                            const std::shared_ptr<IModelContext> &) -> std::unique_ptr<IInferenceRunner>
    {
        created_names.push_back(node.name);
        auto runner = std::make_unique<MockInferenceRunner>(node.name);
        runner->first_layer = node.first_layer;
        runner->last_layer = node.last_layer;
        runner->has_embedding = node.has_embedding;
        runner->has_lm_head = node.has_lm_head;
        return runner;
    };
}

/**
 * @brief Create a TP runner factory that wraps child runners
 */
static TreeToRunnerCompiler::TPRunnerFactory createTPFactory(
    std::vector<std::string> &created_names)
{
    return [&created_names](const ParallelismNode &node,
                            std::vector<std::unique_ptr<IInferenceRunner>> child_runners,
                            const std::shared_ptr<IModelContext> &) -> std::unique_ptr<IInferenceRunner>
    {
        std::string name = "TP:" + node.name + "(" + std::to_string(child_runners.size()) + ")";
        created_names.push_back(name);
        auto runner = std::make_unique<MockInferenceRunner>(name);
        runner->first_layer = node.first_layer;
        runner->last_layer = node.last_layer;
        runner->has_embedding = node.has_embedding;
        runner->has_lm_head = node.has_lm_head;
        return runner;
    };
}

/**
 * @brief Create a local PP runner factory that wraps child runners
 */
static TreeToRunnerCompiler::LocalPPRunnerFactory createLocalPPFactory(
    std::vector<std::string> &created_names)
{
    return [&created_names](const ParallelismNode &node,
                            std::vector<std::unique_ptr<IInferenceRunner>> child_runners,
                            const std::shared_ptr<IModelContext> &) -> std::unique_ptr<IInferenceRunner>
    {
        std::string name = "PP:" + node.name + "(" + std::to_string(child_runners.size()) + ")";
        created_names.push_back(name);
        auto runner = std::make_unique<MockInferenceRunner>(name);
        runner->first_layer = node.first_layer;
        runner->last_layer = node.last_layer;
        return runner;
    };
}

// =============================================================================
// Helper: Build test trees
// =============================================================================

/**
 * Build a single device tree
 */
static ParallelismTree buildSingleDevice(int rank = 0)
{
    auto root = Device(GlobalDeviceAddress::cuda(0), rank);
    root.name = "device0";

    ParallelismTree tree;
    tree.root = std::move(root);
    tree.world_size = 1;
    tree.total_layers = 24;
    tree.assignLayers(24);
    return tree;
}

/**
 * Build a local TP tree (single rank, 2 devices)
 */
static ParallelismTree buildLocalTP(int rank = 0)
{
    auto root = TP("local_tp",
                   {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)},
                   rank, CollectiveBackendType::NCCL);

    ParallelismTree tree;
    tree.root = std::move(root);
    tree.world_size = 1;
    tree.total_layers = 24;
    tree.assignLayers(24);
    return tree;
}

/**
 * Build a cross-rank PP tree (2 ranks, 1 device each)
 */
static ParallelismTree buildCrossRankPP()
{
    auto d0 = Device(GlobalDeviceAddress::cuda(0), 0);
    d0.name = "device_r0";
    auto d1 = Device(GlobalDeviceAddress::cuda(0), 1);
    d1.name = "device_r1";

    auto root = PP("global", {std::move(d0), std::move(d1)});

    ParallelismTree tree;
    tree.root = std::move(root);
    tree.world_size = 2;
    tree.total_layers = 24;
    tree.assignLayers(24);
    return tree;
}

/**
 * Build a PP with local TP children (2 ranks, each with 2-way TP)
 */
static ParallelismTree buildPPWithTP()
{
    auto root = PP("global", {
                                 TP("tp_r0", {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)}, 0, CollectiveBackendType::NCCL),
                                 TP("tp_r1", {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)}, 1, CollectiveBackendType::NCCL),
                             });

    ParallelismTree tree;
    tree.root = std::move(root);
    tree.world_size = 2;
    tree.total_layers = 24;
    tree.assignLayers(24);
    return tree;
}

/**
 * Build a local PP (single rank, 2 devices in pipeline)
 */
static ParallelismTree buildLocalPP()
{
    auto d0 = Device(GlobalDeviceAddress::cuda(0), 0);
    d0.name = "device0";
    auto d1 = Device(GlobalDeviceAddress::cuda(1), 0);
    d1.name = "device1";

    auto root = PP("local_pp", {std::move(d0), std::move(d1)});

    ParallelismTree tree;
    tree.root = std::move(root);
    tree.world_size = 1;
    tree.total_layers = 24;
    tree.assignLayers(24);
    return tree;
}

// =============================================================================
// Test Suite: shouldCompile
// =============================================================================

TEST(Test__TreeToRunnerCompiler, ShouldCompileDevice_OwnedByThisRank)
{
    auto tree = buildSingleDevice(0);
    EXPECT_TRUE(TreeToRunnerCompiler::shouldCompile(tree.root, 0));
}

TEST(Test__TreeToRunnerCompiler, ShouldCompileDevice_OwnedByOtherRank)
{
    auto tree = buildSingleDevice(1);
    EXPECT_FALSE(TreeToRunnerCompiler::shouldCompile(tree.root, 0));
}

TEST(Test__TreeToRunnerCompiler, ShouldCompileTP_ContainsMyDevice)
{
    auto tree = buildLocalTP(0);
    EXPECT_TRUE(TreeToRunnerCompiler::shouldCompile(tree.root, 0));
}

TEST(Test__TreeToRunnerCompiler, ShouldCompileTP_NoMyDevices)
{
    auto tree = buildLocalTP(1);
    EXPECT_FALSE(TreeToRunnerCompiler::shouldCompile(tree.root, 0));
}

TEST(Test__TreeToRunnerCompiler, ShouldCompilePP_ContainsMySubtree)
{
    auto tree = buildCrossRankPP();

    // Rank 0 should compile (owns first child)
    EXPECT_TRUE(TreeToRunnerCompiler::shouldCompile(tree.root, 0));

    // Rank 1 should compile (owns second child)
    EXPECT_TRUE(TreeToRunnerCompiler::shouldCompile(tree.root, 1));
}

TEST(Test__TreeToRunnerCompiler, ShouldCompilePP_NoMySubtree)
{
    auto tree = buildCrossRankPP();

    // Rank 2 owns nothing
    EXPECT_FALSE(TreeToRunnerCompiler::shouldCompile(tree.root, 2));
}

// =============================================================================
// Test Suite: isCrossRankPP
// =============================================================================

TEST(Test__TreeToRunnerCompiler, IsCrossRankPP_True)
{
    auto tree = buildCrossRankPP();
    EXPECT_TRUE(TreeToRunnerCompiler::isCrossRankPP(tree.root));
}

TEST(Test__TreeToRunnerCompiler, IsCrossRankPP_False_LocalPP)
{
    auto tree = buildLocalPP();
    EXPECT_FALSE(TreeToRunnerCompiler::isCrossRankPP(tree.root));
}

TEST(Test__TreeToRunnerCompiler, IsCrossRankPP_False_NotPP)
{
    auto tree = buildLocalTP(0);
    EXPECT_FALSE(TreeToRunnerCompiler::isCrossRankPP(tree.root));
}

// =============================================================================
// Test Suite: Compile with Factories
// =============================================================================

TEST(Test__TreeToRunnerCompiler, CompileSingleDevice)
{
    auto tree = buildSingleDevice(0);

    std::vector<std::string> created;
    TreeToRunnerCompiler::CompileContext ctx;
    ctx.my_rank = 0;
    ctx.world_size = 1;
    ctx.hidden_dim = 896;
    ctx.vocab_size = 128;
    ctx.device_runner_factory = createDeviceFactory(created);

    auto runner = TreeToRunnerCompiler::compile(tree, ctx);

    ASSERT_NE(runner, nullptr);
    EXPECT_EQ(created.size(), 1);
    EXPECT_EQ(created[0], "device0");

    auto *mock = dynamic_cast<MockInferenceRunner *>(runner.get());
    ASSERT_NE(mock, nullptr);
    EXPECT_EQ(mock->first_layer, 0);
    EXPECT_EQ(mock->last_layer, 23);
    EXPECT_TRUE(mock->has_embedding);
    EXPECT_TRUE(mock->has_lm_head);
}

TEST(Test__TreeToRunnerCompiler, CompileLocalTP)
{
    auto tree = buildLocalTP(0);

    std::vector<std::string> created;
    TreeToRunnerCompiler::CompileContext ctx;
    ctx.my_rank = 0;
    ctx.world_size = 1;
    ctx.hidden_dim = 896;
    ctx.vocab_size = 128;
    ctx.device_runner_factory = createDeviceFactory(created);
    ctx.tp_runner_factory = createTPFactory(created);

    auto runner = TreeToRunnerCompiler::compile(tree, ctx);

    ASSERT_NE(runner, nullptr);
    // TP compiler skips DEVICE children (MDO creates its own TP-aware runners).
    // Only the TP wrapper itself is created via the factory.
    EXPECT_EQ(created.size(), 1);

    // Verify TP wrapper was created with 0 child runners
    auto *mock = dynamic_cast<MockInferenceRunner *>(runner.get());
    ASSERT_NE(mock, nullptr);
    EXPECT_EQ(mock->name, "TP:local_tp(0)");
}

TEST(Test__TreeToRunnerCompiler, CompileLocalPP)
{
    auto tree = buildLocalPP();

    std::vector<std::string> created;
    TreeToRunnerCompiler::CompileContext ctx;
    ctx.my_rank = 0;
    ctx.world_size = 1;
    ctx.hidden_dim = 896;
    ctx.vocab_size = 128;
    ctx.device_runner_factory = createDeviceFactory(created);
    ctx.local_pp_runner_factory = createLocalPPFactory(created);

    auto runner = TreeToRunnerCompiler::compile(tree, ctx);

    ASSERT_NE(runner, nullptr);
    // Should create 1 PP runner only (children are NOT pre-compiled
    // when a factory is provided — the factory creates its own runners)
    EXPECT_EQ(created.size(), 1);

    // Verify PP wrapper was created
    auto *mock = dynamic_cast<MockInferenceRunner *>(runner.get());
    ASSERT_NE(mock, nullptr);
    EXPECT_EQ(mock->name.substr(0, 3), "PP:");
}

TEST(Test__TreeToRunnerCompiler, CompileCrossRankPP)
{
    auto tree = buildCrossRankPP();

    std::vector<std::string> created;
    TreeToRunnerCompiler::CompileContext ctx;
    ctx.my_rank = 0;
    ctx.world_size = 2;
    ctx.hidden_dim = 896;
    ctx.vocab_size = 128;
    ctx.device_runner_factory = createDeviceFactory(created);

    auto runner = TreeToRunnerCompiler::compile(tree, ctx);

    ASSERT_NE(runner, nullptr);

    // Should return PipelineRunner
    auto *pr = dynamic_cast<PipelineRunner *>(runner.get());
    ASSERT_NE(pr, nullptr);

    EXPECT_EQ(pr->numStages(), 2);
    EXPECT_EQ(pr->numTransfers(), 1);
    EXPECT_EQ(pr->myStageIndex(), 0); // Rank 0 owns stage 0

    // Only device_r0 should have been created (rank 0 only)
    EXPECT_EQ(created.size(), 1);
    EXPECT_EQ(created[0], "device_r0");
}

TEST(Test__TreeToRunnerCompiler, CompileNestedTPPP)
{
    auto tree = buildPPWithTP();

    std::vector<std::string> created;
    TreeToRunnerCompiler::CompileContext ctx;
    ctx.my_rank = 0;
    ctx.world_size = 2;
    ctx.hidden_dim = 896;
    ctx.vocab_size = 128;
    ctx.device_runner_factory = createDeviceFactory(created);
    ctx.tp_runner_factory = createTPFactory(created);

    auto runner = TreeToRunnerCompiler::compile(tree, ctx);

    ASSERT_NE(runner, nullptr);

    // Should return PipelineRunner
    auto *pr = dynamic_cast<PipelineRunner *>(runner.get());
    ASSERT_NE(pr, nullptr);

    EXPECT_EQ(pr->numStages(), 2);
    EXPECT_EQ(pr->myStageIndex(), 0);

    // TP skips DEVICE children, so only the TP wrapper is created for rank 0's stage
    EXPECT_EQ(created.size(), 1);
}

TEST(Test__TreeToRunnerCompiler, CompileFiltersOtherRanks)
{
    auto tree = buildCrossRankPP();

    std::vector<std::string> created;
    TreeToRunnerCompiler::CompileContext ctx;
    ctx.my_rank = 1; // We are rank 1
    ctx.world_size = 2;
    ctx.hidden_dim = 896;
    ctx.vocab_size = 128;
    ctx.device_runner_factory = createDeviceFactory(created);

    auto runner = TreeToRunnerCompiler::compile(tree, ctx);

    ASSERT_NE(runner, nullptr);

    auto *pr = dynamic_cast<PipelineRunner *>(runner.get());
    ASSERT_NE(pr, nullptr);

    EXPECT_EQ(pr->myStageIndex(), 1); // Rank 1 owns stage 1

    // Only device_r1 should have been created
    EXPECT_EQ(created.size(), 1);
    EXPECT_EQ(created[0], "device_r1");
}

TEST(Test__TreeToRunnerCompiler, LayerRangePassThrough)
{
    auto tree = buildCrossRankPP();

    std::vector<std::string> created;
    TreeToRunnerCompiler::CompileContext ctx;
    ctx.my_rank = 0;
    ctx.world_size = 2;
    ctx.hidden_dim = 896;
    ctx.vocab_size = 128;

    // Track layer ranges via device factory
    std::vector<std::pair<int, int>> layer_ranges;
    ctx.device_runner_factory = [&layer_ranges](const ParallelismNode &node,
                                                const std::shared_ptr<IModelContext> &) -> std::unique_ptr<IInferenceRunner>
    {
        layer_ranges.push_back({node.first_layer, node.last_layer});
        auto runner = std::make_unique<MockInferenceRunner>(node.name);
        runner->first_layer = node.first_layer;
        runner->last_layer = node.last_layer;
        runner->has_embedding = node.has_embedding;
        runner->has_lm_head = node.has_lm_head;
        return runner;
    };

    auto runner = TreeToRunnerCompiler::compile(tree, ctx);

    ASSERT_NE(runner, nullptr);
    EXPECT_EQ(layer_ranges.size(), 1);

    // Rank 0 gets first half: layers 0-11
    EXPECT_EQ(layer_ranges[0].first, 0);
    EXPECT_EQ(layer_ranges[0].second, 11);
}

// =============================================================================
// Test Suite: Edge Cases
// =============================================================================

TEST(Test__TreeToRunnerCompiler, ThrowsOnUnassignedTree)
{
    // Create tree without calling assignLayers
    auto root = Device(GlobalDeviceAddress::cuda(0), 0);
    ParallelismTree tree;
    tree.root = std::move(root);
    tree.world_size = 1;
    // Don't call tree.assignLayers()

    TreeToRunnerCompiler::CompileContext ctx;
    ctx.my_rank = 0;
    ctx.world_size = 1;

    EXPECT_THROW(TreeToRunnerCompiler::compile(tree, ctx), std::invalid_argument);
}

TEST(Test__TreeToRunnerCompiler, ReturnsNullForNonOwningRank)
{
    auto tree = buildSingleDevice(0);

    TreeToRunnerCompiler::CompileContext ctx;
    ctx.my_rank = 1; // Rank 1 doesn't own anything
    ctx.world_size = 2;
    ctx.hidden_dim = 896;
    ctx.vocab_size = 128;

    auto runner = TreeToRunnerCompiler::compile(tree, ctx);
    EXPECT_EQ(runner, nullptr);
}

TEST(Test__TreeToRunnerCompiler, GetSubtreeRanks_SingleDevice)
{
    auto tree = buildSingleDevice(0);
    auto ranks = TreeToRunnerCompiler::getSubtreeRanks(tree.root);

    EXPECT_EQ(ranks.size(), 1);
    EXPECT_TRUE(ranks.count(0));
}

TEST(Test__TreeToRunnerCompiler, GetSubtreeRanks_CrossRankPP)
{
    auto tree = buildCrossRankPP();
    auto ranks = TreeToRunnerCompiler::getSubtreeRanks(tree.root);

    EXPECT_EQ(ranks.size(), 2);
    EXPECT_TRUE(ranks.count(0));
    EXPECT_TRUE(ranks.count(1));
}
