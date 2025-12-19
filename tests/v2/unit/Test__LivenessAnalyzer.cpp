/**
 * @file Test__LivenessAnalyzer.cpp
 * @brief Unit tests for LivenessAnalyzer buffer lifetime analysis
 * @author David Sanftenberg
 * @date December 2025
 */

#include <gtest/gtest.h>
#include "execution/LivenessAnalyzer.h"
#include "execution/GraphExecutor.h"
#include "execution/ComputeStage.h"

using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__LivenessAnalyzer : public ::testing::Test
{
protected:
    LivenessAnalyzer analyzer_;

    // Helper: Create a simple test stage that reports specific buffers
    class MockStage : public IComputeStage
    {
    public:
        explicit MockStage(StageBufferRequirements reqs)
            : reqs_(std::move(reqs)) {}

        bool execute(IDeviceContext *ctx) override
        {
            (void)ctx;
            return true;
        }

        ComputeStageType type() const override { return ComputeStageType::COPY; }

        bool supportsBackend(ComputeBackendType backend) const override
        {
            (void)backend;
            return true; // Supports all backends for testing
        }

        StageBufferRequirements getBufferRequirements() const override
        {
            return reqs_;
        }

    private:
        StageBufferRequirements reqs_;
    };

    // Helper: Create buffer requirements
    static StageBufferRequirements makeReqs()
    {
        return StageBufferRequirements{};
    }
};

// =============================================================================
// BufferLiveness::overlaps() Tests
// =============================================================================

TEST_F(Test__LivenessAnalyzer, Overlaps_SameRange_ReturnsTrue)
{
    BufferLiveness a{.first_use_idx = 1, .last_use_idx = 3};
    BufferLiveness b{.first_use_idx = 1, .last_use_idx = 3};

    EXPECT_TRUE(a.overlaps(b));
    EXPECT_TRUE(b.overlaps(a));
}

TEST_F(Test__LivenessAnalyzer, Overlaps_PartialOverlap_ReturnsTrue)
{
    BufferLiveness a{.first_use_idx = 1, .last_use_idx = 5};
    BufferLiveness b{.first_use_idx = 3, .last_use_idx = 7};

    EXPECT_TRUE(a.overlaps(b));
    EXPECT_TRUE(b.overlaps(a));
}

TEST_F(Test__LivenessAnalyzer, Overlaps_Contained_ReturnsTrue)
{
    BufferLiveness a{.first_use_idx = 0, .last_use_idx = 10};
    BufferLiveness b{.first_use_idx = 3, .last_use_idx = 7};

    EXPECT_TRUE(a.overlaps(b));
    EXPECT_TRUE(b.overlaps(a));
}

TEST_F(Test__LivenessAnalyzer, Overlaps_Adjacent_ReturnsFalse)
{
    BufferLiveness a{.first_use_idx = 0, .last_use_idx = 3};
    BufferLiveness b{.first_use_idx = 4, .last_use_idx = 7};

    EXPECT_FALSE(a.overlaps(b));
    EXPECT_FALSE(b.overlaps(a));
}

TEST_F(Test__LivenessAnalyzer, Overlaps_NonOverlapping_ReturnsFalse)
{
    BufferLiveness a{.first_use_idx = 0, .last_use_idx = 2};
    BufferLiveness b{.first_use_idx = 5, .last_use_idx = 8};

    EXPECT_FALSE(a.overlaps(b));
    EXPECT_FALSE(b.overlaps(a));
}

TEST_F(Test__LivenessAnalyzer, Overlaps_TouchingEdge_ReturnsFalse)
{
    // When a ends at 3 and b starts at 3, they don't overlap
    // (buffer a is done at stage 3, buffer b starts at stage 3)
    BufferLiveness a{.first_use_idx = 0, .last_use_idx = 2};
    BufferLiveness b{.first_use_idx = 3, .last_use_idx = 5};

    EXPECT_FALSE(a.overlaps(b));
    EXPECT_FALSE(b.overlaps(a));
}

// =============================================================================
// canAlias() Tests
// =============================================================================

TEST_F(Test__LivenessAnalyzer, CanAlias_BothScratchNonOverlapping_ReturnsTrue)
{
    BufferLiveness a{
        .first_use_idx = 0,
        .last_use_idx = 2,
        .role = BufferRole::SCRATCH,
        .tensor_type = BufferTensorType::FP32,
    };
    BufferLiveness b{
        .first_use_idx = 3,
        .last_use_idx = 5,
        .role = BufferRole::SCRATCH,
        .tensor_type = BufferTensorType::FP32,
    };

    EXPECT_TRUE(LivenessAnalyzer::canAlias(a, b));
}

TEST_F(Test__LivenessAnalyzer, CanAlias_NonScratch_ReturnsFalse)
{
    BufferLiveness a{
        .first_use_idx = 0,
        .last_use_idx = 2,
        .role = BufferRole::INPUT, // Not SCRATCH
        .tensor_type = BufferTensorType::FP32,
    };
    BufferLiveness b{
        .first_use_idx = 3,
        .last_use_idx = 5,
        .role = BufferRole::SCRATCH,
        .tensor_type = BufferTensorType::FP32,
    };

    EXPECT_FALSE(LivenessAnalyzer::canAlias(a, b));
}

TEST_F(Test__LivenessAnalyzer, CanAlias_Overlapping_ReturnsFalse)
{
    BufferLiveness a{
        .first_use_idx = 0,
        .last_use_idx = 4,
        .role = BufferRole::SCRATCH,
        .tensor_type = BufferTensorType::FP32,
    };
    BufferLiveness b{
        .first_use_idx = 2,
        .last_use_idx = 6,
        .role = BufferRole::SCRATCH,
        .tensor_type = BufferTensorType::FP32,
    };

    EXPECT_FALSE(LivenessAnalyzer::canAlias(a, b));
}

TEST_F(Test__LivenessAnalyzer, CanAlias_DifferentFloatTypes_ReturnsTrue)
{
    BufferLiveness a{
        .first_use_idx = 0,
        .last_use_idx = 2,
        .role = BufferRole::SCRATCH,
        .tensor_type = BufferTensorType::FP32,
    };
    BufferLiveness b{
        .first_use_idx = 3,
        .last_use_idx = 5,
        .role = BufferRole::SCRATCH,
        .tensor_type = BufferTensorType::BF16, // Different but compatible
    };

    EXPECT_TRUE(LivenessAnalyzer::canAlias(a, b));
}

TEST_F(Test__LivenessAnalyzer, CanAlias_IncompatibleQuantTypes_ReturnsFalse)
{
    BufferLiveness a{
        .first_use_idx = 0,
        .last_use_idx = 2,
        .role = BufferRole::SCRATCH,
        .tensor_type = BufferTensorType::Q8_0,
    };
    BufferLiveness b{
        .first_use_idx = 3,
        .last_use_idx = 5,
        .role = BufferRole::SCRATCH,
        .tensor_type = BufferTensorType::Q4_0, // Different quantization
    };

    EXPECT_FALSE(LivenessAnalyzer::canAlias(a, b));
}

// =============================================================================
// filterScratchBuffers() Tests
// =============================================================================

TEST_F(Test__LivenessAnalyzer, FilterScratchBuffers_MixedRoles)
{
    std::vector<BufferLiveness> lifetimes = {
        {.buffer_name = "input", .role = BufferRole::INPUT},
        {.buffer_name = "scratch1", .role = BufferRole::SCRATCH},
        {.buffer_name = "output", .role = BufferRole::OUTPUT},
        {.buffer_name = "scratch2", .role = BufferRole::SCRATCH},
        {.buffer_name = "weight", .role = BufferRole::WEIGHT},
    };

    auto scratch = LivenessAnalyzer::filterScratchBuffers(lifetimes);

    ASSERT_EQ(scratch.size(), 2);
    EXPECT_EQ(scratch[0].buffer_name, "scratch1");
    EXPECT_EQ(scratch[1].buffer_name, "scratch2");
}

TEST_F(Test__LivenessAnalyzer, FilterScratchBuffers_Empty)
{
    std::vector<BufferLiveness> lifetimes;
    auto scratch = LivenessAnalyzer::filterScratchBuffers(lifetimes);
    EXPECT_TRUE(scratch.empty());
}

TEST_F(Test__LivenessAnalyzer, FilterScratchBuffers_NoScratch)
{
    std::vector<BufferLiveness> lifetimes = {
        {.buffer_name = "input", .role = BufferRole::INPUT},
        {.buffer_name = "output", .role = BufferRole::OUTPUT},
    };

    auto scratch = LivenessAnalyzer::filterScratchBuffers(lifetimes);
    EXPECT_TRUE(scratch.empty());
}

// =============================================================================
// analyze() Tests with Mock Stages
// =============================================================================

TEST_F(Test__LivenessAnalyzer, Analyze_EmptyGraph_ReturnsEmpty)
{
    ComputeGraph graph;

    auto lifetimes = analyzer_.analyze(graph);

    EXPECT_TRUE(lifetimes.empty());
}

TEST_F(Test__LivenessAnalyzer, Analyze_SingleStage_ReturnsBuffers)
{
    ComputeGraph graph;

    auto reqs = makeReqs();
    reqs.addInput("input", {32, 896}, BufferTensorType::FP32);
    reqs.addOutput("output", {32, 896}, BufferTensorType::FP32);

    graph.addNode("stage0", std::make_unique<MockStage>(reqs));

    auto lifetimes = analyzer_.analyze(graph);

    ASSERT_EQ(lifetimes.size(), 2);

    // Both buffers should have same first/last use (single stage)
    for (const auto &l : lifetimes)
    {
        EXPECT_EQ(l.first_use_idx, 0);
        EXPECT_EQ(l.last_use_idx, 0);
    }
}

TEST_F(Test__LivenessAnalyzer, Analyze_LinearChain_TracksLifetimes)
{
    ComputeGraph graph;

    // Stage 0: input -> scratch1
    auto reqs0 = makeReqs();
    reqs0.addInput("in", {32, 896}, BufferTensorType::FP32);
    reqs0.addScratch("temp", {32, 896}, BufferTensorType::FP32);
    graph.addNode("stage0", std::make_unique<MockStage>(reqs0));

    // Stage 1: uses scratch1
    auto reqs1 = makeReqs();
    reqs1.addScratch("temp2", {32, 896}, BufferTensorType::FP32);
    graph.addNode("stage1", std::make_unique<MockStage>(reqs1));
    graph.addDependency("stage1", "stage0");

    // Stage 2: output
    auto reqs2 = makeReqs();
    reqs2.addOutput("out", {32, 896}, BufferTensorType::FP32);
    graph.addNode("stage2", std::make_unique<MockStage>(reqs2));
    graph.addDependency("stage2", "stage1");

    auto lifetimes = analyzer_.analyze(graph);

    // Should have 4 buffers: in, temp, temp2, out
    EXPECT_EQ(lifetimes.size(), 4);
}

// =============================================================================
// computeAliasingGroups() Tests
// =============================================================================

TEST_F(Test__LivenessAnalyzer, ComputeAliasingGroups_NonOverlapping_SingleGroup)
{
    std::vector<BufferLiveness> lifetimes = {
        {
            .buffer_name = "buf_a",
            .first_use_idx = 0,
            .last_use_idx = 2,
            .role = BufferRole::SCRATCH,
            .tensor_type = BufferTensorType::FP32,
            .size_bytes = 1024,
        },
        {
            .buffer_name = "buf_b",
            .first_use_idx = 3,
            .last_use_idx = 5,
            .role = BufferRole::SCRATCH,
            .tensor_type = BufferTensorType::FP32,
            .size_bytes = 2048,
        },
    };

    auto groups = analyzer_.computeAliasingGroups(lifetimes);

    // Both can alias into single group
    ASSERT_EQ(groups.size(), 1);
    EXPECT_EQ(groups[0].buffer_names.size(), 2);
    EXPECT_EQ(groups[0].max_size_bytes, 2048); // Max of both
}

TEST_F(Test__LivenessAnalyzer, ComputeAliasingGroups_Overlapping_SeparateGroups)
{
    std::vector<BufferLiveness> lifetimes = {
        {
            .buffer_name = "buf_a",
            .first_use_idx = 0,
            .last_use_idx = 4,
            .role = BufferRole::SCRATCH,
            .tensor_type = BufferTensorType::FP32,
            .size_bytes = 1024,
        },
        {
            .buffer_name = "buf_b",
            .first_use_idx = 2,
            .last_use_idx = 6,
            .role = BufferRole::SCRATCH,
            .tensor_type = BufferTensorType::FP32,
            .size_bytes = 2048,
        },
    };

    auto groups = analyzer_.computeAliasingGroups(lifetimes);

    // Cannot alias - need separate groups
    ASSERT_EQ(groups.size(), 2);
    EXPECT_EQ(groups[0].buffer_names.size(), 1);
    EXPECT_EQ(groups[1].buffer_names.size(), 1);
}

TEST_F(Test__LivenessAnalyzer, ComputeAliasingGroups_ThreeBuffers_TwoCanAlias)
{
    std::vector<BufferLiveness> lifetimes = {
        {
            .buffer_name = "buf_a",
            .first_use_idx = 0,
            .last_use_idx = 2,
            .role = BufferRole::SCRATCH,
            .tensor_type = BufferTensorType::FP32,
            .size_bytes = 1024,
        },
        {
            .buffer_name = "buf_b",
            .first_use_idx = 1,
            .last_use_idx = 5,
            .role = BufferRole::SCRATCH,
            .tensor_type = BufferTensorType::FP32,
            .size_bytes = 2048,
        },
        {
            .buffer_name = "buf_c",
            .first_use_idx = 6,
            .last_use_idx = 8,
            .role = BufferRole::SCRATCH,
            .tensor_type = BufferTensorType::FP32,
            .size_bytes = 512,
        },
    };

    auto groups = analyzer_.computeAliasingGroups(lifetimes);

    // buf_a overlaps buf_b (can't alias)
    // buf_b doesn't overlap buf_c (can alias)
    // Result: either {a}, {b, c} or {a, c}, {b}
    EXPECT_EQ(groups.size(), 2);

    // Total buffers across all groups should be 3
    size_t total_buffers = 0;
    for (const auto &g : groups)
    {
        total_buffers += g.buffer_names.size();
    }
    EXPECT_EQ(total_buffers, 3);
}

TEST_F(Test__LivenessAnalyzer, ComputeAliasingGroups_NonScratch_Ignored)
{
    std::vector<BufferLiveness> lifetimes = {
        {
            .buffer_name = "input",
            .first_use_idx = 0,
            .last_use_idx = 2,
            .role = BufferRole::INPUT, // Not SCRATCH
            .tensor_type = BufferTensorType::FP32,
            .size_bytes = 1024,
        },
        {
            .buffer_name = "scratch",
            .first_use_idx = 3,
            .last_use_idx = 5,
            .role = BufferRole::SCRATCH,
            .tensor_type = BufferTensorType::FP32,
            .size_bytes = 2048,
        },
    };

    auto groups = analyzer_.computeAliasingGroups(lifetimes);

    // Only SCRATCH buffer should be in groups
    ASSERT_EQ(groups.size(), 1);
    EXPECT_EQ(groups[0].buffer_names.size(), 1);
    EXPECT_EQ(groups[0].buffer_names[0], "scratch");
}

// =============================================================================
// computeMemoryUsage() Tests
// =============================================================================

TEST_F(Test__LivenessAnalyzer, ComputeMemoryUsage_NoAliasing)
{
    std::vector<BufferLiveness> lifetimes = {
        {.buffer_name = "a", .role = BufferRole::SCRATCH, .size_bytes = 1000},
        {.buffer_name = "b", .role = BufferRole::SCRATCH, .size_bytes = 2000},
    };

    // No aliasing - separate groups
    std::vector<AliasingGroup> groups = {
        {.buffer_names = {"a"}, .max_size_bytes = 1000},
        {.buffer_names = {"b"}, .max_size_bytes = 2000},
    };

    auto [original, optimized] = analyzer_.computeMemoryUsage(lifetimes, groups);

    EXPECT_EQ(original, 3000);
    EXPECT_EQ(optimized, 3000); // No savings
}

TEST_F(Test__LivenessAnalyzer, ComputeMemoryUsage_WithAliasing)
{
    std::vector<BufferLiveness> lifetimes = {
        {.buffer_name = "a", .role = BufferRole::SCRATCH, .size_bytes = 1000},
        {.buffer_name = "b", .role = BufferRole::SCRATCH, .size_bytes = 2000},
    };

    // Both alias to single group
    std::vector<AliasingGroup> groups = {
        {.buffer_names = {"a", "b"}, .max_size_bytes = 2000},
    };

    auto [original, optimized] = analyzer_.computeMemoryUsage(lifetimes, groups);

    EXPECT_EQ(original, 3000);
    EXPECT_EQ(optimized, 2000); // Saved 1000 bytes
}

TEST_F(Test__LivenessAnalyzer, ComputeMemoryUsage_MixedRoles)
{
    std::vector<BufferLiveness> lifetimes = {
        {.buffer_name = "input", .role = BufferRole::INPUT, .size_bytes = 500},
        {.buffer_name = "scratch1", .role = BufferRole::SCRATCH, .size_bytes = 1000},
        {.buffer_name = "scratch2", .role = BufferRole::SCRATCH, .size_bytes = 2000},
        {.buffer_name = "output", .role = BufferRole::OUTPUT, .size_bytes = 500},
    };

    // Scratch buffers alias
    std::vector<AliasingGroup> groups = {
        {.buffer_names = {"scratch1", "scratch2"}, .max_size_bytes = 2000},
    };

    auto [original, optimized] = analyzer_.computeMemoryUsage(lifetimes, groups);

    EXPECT_EQ(original, 4000);
    // Non-scratch: 500 + 500 = 1000
    // Aliased scratch: 2000
    EXPECT_EQ(optimized, 3000);
}

// =============================================================================
// computeSavingsPercent() Tests
// =============================================================================

TEST_F(Test__LivenessAnalyzer, ComputeSavingsPercent_NoSavings)
{
    std::vector<BufferLiveness> lifetimes = {
        {.buffer_name = "a", .role = BufferRole::SCRATCH, .size_bytes = 1000},
    };
    std::vector<AliasingGroup> groups = {
        {.buffer_names = {"a"}, .max_size_bytes = 1000},
    };

    double savings = analyzer_.computeSavingsPercent(lifetimes, groups);
    EXPECT_DOUBLE_EQ(savings, 0.0);
}

TEST_F(Test__LivenessAnalyzer, ComputeSavingsPercent_50Percent)
{
    std::vector<BufferLiveness> lifetimes = {
        {.buffer_name = "a", .role = BufferRole::SCRATCH, .size_bytes = 1000},
        {.buffer_name = "b", .role = BufferRole::SCRATCH, .size_bytes = 1000},
    };
    std::vector<AliasingGroup> groups = {
        {.buffer_names = {"a", "b"}, .max_size_bytes = 1000},
    };

    double savings = analyzer_.computeSavingsPercent(lifetimes, groups);
    EXPECT_DOUBLE_EQ(savings, 50.0);
}

TEST_F(Test__LivenessAnalyzer, ComputeSavingsPercent_EmptyLifetimes)
{
    std::vector<BufferLiveness> lifetimes;
    std::vector<AliasingGroup> groups;

    double savings = analyzer_.computeSavingsPercent(lifetimes, groups);
    EXPECT_DOUBLE_EQ(savings, 0.0);
}

// =============================================================================
// Integration: Full Pipeline Analysis
// =============================================================================

TEST_F(Test__LivenessAnalyzer, Integration_AttentionFFNPattern)
{
    // Simulates a typical transformer layer pattern:
    // Attention: uses Q, K, V buffers
    // FFN: uses gate, up buffers (can reuse Q, K, V since attention is done)

    ComputeGraph graph;

    // Attention norm
    auto norm_reqs = makeReqs();
    norm_reqs.addInput("hidden", {32, 896}, BufferTensorType::FP32);
    norm_reqs.addScratch("normalized", {32, 896}, BufferTensorType::FP32);
    graph.addNode("attn_norm", std::make_unique<MockStage>(norm_reqs));

    // QKV projection
    auto qkv_reqs = makeReqs();
    qkv_reqs.addScratch("Q", {32, 896}, BufferTensorType::FP32);
    qkv_reqs.addScratch("K", {32, 128}, BufferTensorType::FP32);
    qkv_reqs.addScratch("V", {32, 128}, BufferTensorType::FP32);
    graph.addNode("qkv_proj", std::make_unique<MockStage>(qkv_reqs));
    graph.addDependency("qkv_proj", "attn_norm");

    // Attention compute
    auto attn_reqs = makeReqs();
    attn_reqs.addScratch("attn_out", {32, 896}, BufferTensorType::FP32);
    graph.addNode("attention", std::make_unique<MockStage>(attn_reqs));
    graph.addDependency("attention", "qkv_proj");

    // FFN norm
    auto ffn_norm_reqs = makeReqs();
    ffn_norm_reqs.addScratch("ffn_normalized", {32, 896}, BufferTensorType::FP32);
    graph.addNode("ffn_norm", std::make_unique<MockStage>(ffn_norm_reqs));
    graph.addDependency("ffn_norm", "attention");

    // FFN gate/up (should be able to reuse Q, K, V)
    auto ffn_reqs = makeReqs();
    ffn_reqs.addScratch("gate", {32, 4864}, BufferTensorType::FP32);
    ffn_reqs.addScratch("up", {32, 4864}, BufferTensorType::FP32);
    graph.addNode("ffn_proj", std::make_unique<MockStage>(ffn_reqs));
    graph.addDependency("ffn_proj", "ffn_norm");

    // Output
    auto out_reqs = makeReqs();
    out_reqs.addOutput("output", {32, 896}, BufferTensorType::FP32);
    graph.addNode("output", std::make_unique<MockStage>(out_reqs));
    graph.addDependency("output", "ffn_proj");

    // Analyze
    auto lifetimes = analyzer_.analyze(graph);
    auto groups = analyzer_.computeAliasingGroups(lifetimes);
    auto [original, optimized] = analyzer_.computeMemoryUsage(lifetimes, groups);

    // We should see some memory savings from aliasing
    EXPECT_LT(optimized, original);

    double savings = analyzer_.computeSavingsPercent(lifetimes, groups);
    LOG_INFO("Attention+FFN pattern savings: " << savings << "%");
    EXPECT_GT(savings, 0.0);
}
