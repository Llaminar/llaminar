/**
 * @file Test__GraphValidator.cpp
 * @brief Unit tests for GraphValidator
 * @author GitHub Copilot
 * @date December 2025
 *
 * Tests buffer flow validation, orphan detection, and compatibility checks.
 */

#include <gtest/gtest.h>
#include "../../../src/v2/execution/GraphValidator.h"
#include "../../../src/v2/execution/GraphExecutor.h"
#include "execution/compute_stages/ComputeStages.h"

using namespace llaminar2;

// =============================================================================
// Mock Stages for Testing
// =============================================================================

/**
 * @brief Mock stage that declares specific buffer requirements
 */
class MockValidatorStage : public IComputeStage
{
public:
    MockValidatorStage(StageBufferRequirements reqs)
        : requirements_(std::move(reqs)) {}

    bool execute(IDeviceContext *) override { return true; }
    ComputeStageType type() const override { return ComputeStageType::COPY; }
    bool supportsBackend(ComputeBackendType) const override { return true; }
    size_t estimatedFlops() const override { return 0; }

    StageBufferRequirements getBufferRequirements() const override
    {
        return requirements_;
    }

private:
    StageBufferRequirements requirements_;
};

// =============================================================================
// Test Fixture
// =============================================================================

class GraphValidatorTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        validator_ = std::make_unique<GraphValidator>();
    }

    std::unique_ptr<GraphValidator> validator_;

    // Helper to create a stage with given requirements
    std::unique_ptr<MockValidatorStage> makeStage(StageBufferRequirements reqs)
    {
        return std::make_unique<MockValidatorStage>(std::move(reqs));
    }
};

// =============================================================================
// Basic Validation Tests
// =============================================================================

TEST_F(GraphValidatorTest, EmptyGraphIsValid)
{
    ComputeGraph graph;
    auto result = validator_->validateAll(graph);

    EXPECT_TRUE(result.valid);
    EXPECT_TRUE(result.errors.empty());
}

TEST_F(GraphValidatorTest, SingleStageNoBuffersIsValid)
{
    ComputeGraph graph;

    StageBufferRequirements reqs;
    graph.addNode("stage1", makeStage(reqs));

    auto result = validator_->validateAll(graph);

    EXPECT_TRUE(result.valid);
    EXPECT_TRUE(result.errors.empty());
}

TEST_F(GraphValidatorTest, ProducerConsumerPairIsValid)
{
    ComputeGraph graph;

    // Stage 1 produces buffer "data"
    StageBufferRequirements reqs1;
    reqs1.addOutput("data", {32, 64}, BufferTensorType::FP32);
    graph.addNode("producer", makeStage(reqs1));

    // Stage 2 consumes buffer "data"
    StageBufferRequirements reqs2;
    reqs2.addInput("data", {32, 64}, BufferTensorType::FP32);
    graph.addNode("consumer", makeStage(reqs2));

    graph.addDependency("consumer", "producer");

    auto result = validator_->validateBufferFlow(graph);

    EXPECT_TRUE(result.valid);
    EXPECT_TRUE(result.errors.empty());
}

// =============================================================================
// Producer/Consumer Ordering Tests
// =============================================================================

TEST_F(GraphValidatorTest, ConsumerBeforeProducerIsWarning)
{
    ComputeGraph graph;

    // Stage 1 consumes buffer "data" (but runs first due to ordering)
    StageBufferRequirements reqs1;
    reqs1.addInput("data", {32, 64}, BufferTensorType::FP32);
    graph.addNode("consumer", makeStage(reqs1));

    // Stage 2 produces buffer "data" (but runs second)
    StageBufferRequirements reqs2;
    reqs2.addOutput("data", {32, 64}, BufferTensorType::FP32);
    graph.addNode("producer", makeStage(reqs2));

    // Wrong dependency order - producer depends on consumer
    // This means consumer runs BEFORE producer
    graph.addDependency("producer", "consumer");

    auto result = validator_->validateBufferFlow(graph);

    // This should detect the ordering issue as a warning
    // (consumer reads before producer writes)
    // Note: Since graph has no explicit ordering, both orderings are checked
    EXPECT_FALSE(result.warnings.empty()) << "Should warn about potential ordering issue";
}

// =============================================================================
// Missing Producer Tests
// =============================================================================

TEST_F(GraphValidatorTest, DeclaredProducerMissingIsError)
{
    ComputeGraph graph;

    // Create a buffer descriptor with a declared producer that doesn't exist
    StageBufferRequirements reqs;
    auto desc = BufferDescriptor::output("data", {32, 64}, BufferTensorType::FP32);
    desc.withProducer("nonexistent_stage");
    reqs.add(desc);

    graph.addNode("stage1", makeStage(reqs));

    auto result = validator_->validateBufferFlow(graph);

    EXPECT_FALSE(result.valid);
    EXPECT_FALSE(result.errors.empty());

    // Check error message mentions the missing producer
    bool found_error = false;
    for (const auto &err : result.errors)
    {
        if (err.find("nonexistent_stage") != std::string::npos &&
            err.find("doesn't exist") != std::string::npos)
        {
            found_error = true;
            break;
        }
    }
    EXPECT_TRUE(found_error) << "Should mention missing producer stage";
}

// =============================================================================
// Orphan Buffer Tests
// =============================================================================

TEST_F(GraphValidatorTest, OrphanOutputGeneratesWarning)
{
    ComputeGraph graph;

    // Stage produces buffer but no one consumes it
    StageBufferRequirements reqs;
    reqs.addOutput("orphan_data", {32, 64}, BufferTensorType::FP32);
    graph.addNode("producer", makeStage(reqs));

    auto result = validator_->validateBufferFlow(graph);

    // Should be valid but with warning
    EXPECT_TRUE(result.valid);
    EXPECT_FALSE(result.warnings.empty());

    bool found_warning = false;
    for (const auto &warn : result.warnings)
    {
        if (warn.find("orphan_data") != std::string::npos &&
            warn.find("never consumed") != std::string::npos)
        {
            found_warning = true;
            break;
        }
    }
    EXPECT_TRUE(found_warning) << "Should warn about orphan buffer";
}

TEST_F(GraphValidatorTest, OrphanAllocationDetected)
{
    ComputeGraph graph;

    // Stage doesn't declare "V_dequant" as output
    StageBufferRequirements reqs;
    reqs.addOutput("other_buffer", {32, 64}, BufferTensorType::FP32);
    graph.addNode("kv_append", makeStage(reqs));

    // But we allocated V_dequant externally
    std::vector<std::string> allocated = {"V_dequant", "other_buffer"};

    auto result = validator_->validateNoOrphanAllocations(allocated, graph);

    EXPECT_FALSE(result.warnings.empty());

    bool found_warning = false;
    for (const auto &warn : result.warnings)
    {
        if (warn.find("V_dequant") != std::string::npos &&
            warn.find("no stage declares") != std::string::npos)
        {
            found_warning = true;
            break;
        }
    }
    EXPECT_TRUE(found_warning) << "Should detect orphan allocation";
}

// =============================================================================
// Type Compatibility Tests
// =============================================================================

TEST_F(GraphValidatorTest, TypeMismatchIsError)
{
    ComputeGraph graph;

    // Producer outputs FP32
    StageBufferRequirements reqs1;
    reqs1.addOutput("data", {32, 64}, BufferTensorType::FP32);
    graph.addNode("producer", makeStage(reqs1));

    // Consumer expects Q8_1
    StageBufferRequirements reqs2;
    reqs2.addInput("data", {32, 64}, BufferTensorType::Q8_1);
    graph.addNode("consumer", makeStage(reqs2));

    graph.addDependency("consumer", "producer");

    auto result = validator_->validateBufferCompatibility(graph);

    EXPECT_FALSE(result.valid);
    EXPECT_FALSE(result.errors.empty());

    bool found_error = false;
    for (const auto &err : result.errors)
    {
        if (err.find("data") != std::string::npos &&
            err.find("type mismatch") != std::string::npos)
        {
            found_error = true;
            break;
        }
    }
    EXPECT_TRUE(found_error) << "Should detect type mismatch";
}

TEST_F(GraphValidatorTest, MatchingTypesAreValid)
{
    ComputeGraph graph;

    // Both stages use FP32
    StageBufferRequirements reqs1;
    reqs1.addOutput("data", {32, 64}, BufferTensorType::FP32);
    graph.addNode("producer", makeStage(reqs1));

    StageBufferRequirements reqs2;
    reqs2.addInput("data", {32, 64}, BufferTensorType::FP32);
    graph.addNode("consumer", makeStage(reqs2));

    graph.addDependency("consumer", "producer");

    auto result = validator_->validateBufferCompatibility(graph);

    EXPECT_TRUE(result.valid);
    EXPECT_TRUE(result.errors.empty());
}

// =============================================================================
// BufferDescriptor Builder Tests
// =============================================================================

TEST_F(GraphValidatorTest, BufferDescriptor_WithProducer)
{
    auto desc = BufferDescriptor::output("V_dequant", {9, 128}, BufferTensorType::FP32);
    desc.withProducer("kv_append");

    EXPECT_EQ(desc.producer_stage, "kv_append");
    EXPECT_TRUE(desc.hasProducer());
}

TEST_F(GraphValidatorTest, BufferDescriptor_ValidatePopulated)
{
    auto desc = BufferDescriptor::output("V_dequant", {9, 128}, BufferTensorType::FP32);
    desc.withProducer("kv_append").validatePopulated();

    EXPECT_TRUE(desc.validate_populated);
    EXPECT_TRUE(desc.hasProducer());
}

TEST_F(GraphValidatorTest, BufferDescriptor_NoProducerByDefault)
{
    auto desc = BufferDescriptor::output("data", {32, 64}, BufferTensorType::FP32);

    EXPECT_TRUE(desc.producer_stage.empty());
    EXPECT_FALSE(desc.hasProducer());
    EXPECT_FALSE(desc.validate_populated);
}

// =============================================================================
// GraphValidationResult Tests
// =============================================================================

TEST_F(GraphValidatorTest, ValidationResult_Merge)
{
    GraphValidationResult result1;
    result1.addError("Error 1");
    result1.addWarning("Warning 1");

    GraphValidationResult result2;
    result2.addError("Error 2");

    result1.merge(result2);

    EXPECT_FALSE(result1.valid);
    EXPECT_EQ(result1.errors.size(), 2u);
    EXPECT_EQ(result1.warnings.size(), 1u);
}

TEST_F(GraphValidatorTest, ValidationResult_AddErrorMarksInvalid)
{
    GraphValidationResult result;
    EXPECT_TRUE(result.valid);

    result.addError("Something went wrong");

    EXPECT_FALSE(result.valid);
    EXPECT_EQ(result.errors.size(), 1u);
}

TEST_F(GraphValidatorTest, ValidationResult_WarningKeepsValid)
{
    GraphValidationResult result;
    result.addWarning("Potential issue");

    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.warnings.size(), 1u);
}

// =============================================================================
// Complex Graph Tests
// =============================================================================

TEST_F(GraphValidatorTest, MultiStageChainIsValid)
{
    ComputeGraph graph;

    // Create a chain: embed -> attn -> ffn -> output
    StageBufferRequirements embed_reqs;
    embed_reqs.addOutput("hidden", {9, 896}, BufferTensorType::FP32);
    graph.addNode("embed", makeStage(embed_reqs));

    StageBufferRequirements attn_reqs;
    attn_reqs.addInput("hidden", {9, 896}, BufferTensorType::FP32);
    attn_reqs.addOutput("attn_out", {9, 896}, BufferTensorType::FP32);
    graph.addNode("attn", makeStage(attn_reqs));

    StageBufferRequirements ffn_reqs;
    ffn_reqs.addInput("attn_out", {9, 896}, BufferTensorType::FP32);
    ffn_reqs.addOutput("ffn_out", {9, 896}, BufferTensorType::FP32);
    graph.addNode("ffn", makeStage(ffn_reqs));

    StageBufferRequirements out_reqs;
    out_reqs.addInput("ffn_out", {9, 896}, BufferTensorType::FP32);
    graph.addNode("output", makeStage(out_reqs));

    graph.addDependency("attn", "embed");
    graph.addDependency("ffn", "attn");
    graph.addDependency("output", "ffn");

    auto result = validator_->validateAll(graph);

    EXPECT_TRUE(result.valid);
    EXPECT_TRUE(result.errors.empty());
}

TEST_F(GraphValidatorTest, HybridModeVDequantPattern)
{
    // Simulate the V_dequant pattern that caused the bug
    ComputeGraph graph;

    // QKV projection produces Q8_1 V
    StageBufferRequirements qkv_reqs;
    qkv_reqs.addOutput("V", {9, 128}, BufferTensorType::Q8_1);
    graph.addNode("qkv_proj", makeStage(qkv_reqs));

    // KV cache append produces V_dequant (FP32)
    StageBufferRequirements kv_reqs;
    kv_reqs.addInput("V", {9, 128}, BufferTensorType::Q8_1);
    // Declare V_dequant as output with producer and validation
    auto v_dequant_desc = BufferDescriptor::output("V_dequant", {9, 128}, BufferTensorType::FP32);
    v_dequant_desc.withProducer("kv_append").validatePopulated();
    kv_reqs.add(v_dequant_desc);
    graph.addNode("kv_append", makeStage(kv_reqs));

    // Attention consumes V_dequant
    StageBufferRequirements attn_reqs;
    attn_reqs.addInput("V_dequant", {9, 128}, BufferTensorType::FP32);
    graph.addNode("attention", makeStage(attn_reqs));

    graph.addDependency("kv_append", "qkv_proj");
    graph.addDependency("attention", "kv_append");

    auto result = validator_->validateAll(graph);

    EXPECT_TRUE(result.valid);
    EXPECT_TRUE(result.errors.empty());

    // Verify the descriptor has the right properties
    const auto *kv_node = graph.getNode("kv_append");
    ASSERT_NE(kv_node, nullptr);
    auto kv_buf_reqs = kv_node->stage->getBufferRequirements();

    bool found_v_dequant = false;
    for (const auto &buf : kv_buf_reqs.buffers)
    {
        if (buf.name == "V_dequant")
        {
            found_v_dequant = true;
            EXPECT_EQ(buf.producer_stage, "kv_append");
            EXPECT_TRUE(buf.validate_populated);
        }
    }
    EXPECT_TRUE(found_v_dequant);
}
