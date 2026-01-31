/**
 * @file Test__StageOutputManifest.cpp
 * @brief Unit tests for Stage Output Manifest (getDeclaredOutputs)
 * @author GitHub Copilot
 * @date December 2025
 *
 * Tests that stages correctly declare their output buffers, enabling
 * GraphValidator to verify buffer contracts at build time.
 */

#include <gtest/gtest.h>
#include "execution/compute_stages/ComputeStages.h"
#include "execution/debug/BufferRole.h"
#include "tensors/Tensors.h"

using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

class StageOutputManifestTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create test tensors
        K_q8 = std::make_unique<Q8_1Tensor>(std::vector<size_t>{9, 128}); // 9 tokens, 128 dim
        V_q8 = std::make_unique<Q8_1Tensor>(std::vector<size_t>{9, 128});
        V_dequant = std::make_unique<FP32Tensor>(std::vector<size_t>{9, 128});

        // Initialize with non-zero data to avoid validation issues
        auto *v_data = V_dequant->mutable_data();
        for (size_t i = 0; i < V_dequant->numel(); ++i)
        {
            v_data[i] = 0.5f;
        }
    }

    std::unique_ptr<Q8_1Tensor> K_q8;
    std::unique_ptr<Q8_1Tensor> V_q8;
    std::unique_ptr<FP32Tensor> V_dequant;
};

// =============================================================================
// KVCacheAppendStage Output Manifest Tests
// =============================================================================

TEST_F(StageOutputManifestTest, KVCacheAppend_NoOutputsWithoutVDequant)
{
    // Standard mode: no V_dequant output
    KVCacheAppendStage::Params params;
    params.K = K_q8.get();
    params.V = V_q8.get();
    params.kv_cache = nullptr; // Not needed for getDeclaredOutputs
    params.layer_idx = 0;
    params.num_tokens = 9;
    // V_dequant_out = nullptr (default)

    KVCacheAppendStage stage(params);

    auto outputs = stage.getDeclaredOutputs();
    EXPECT_TRUE(outputs.empty()) << "Standard mode should have no declared outputs";
    EXPECT_FALSE(stage.producesVDequant());
}

TEST_F(StageOutputManifestTest, KVCacheAppend_DeclaresVDequantInHybridMode)
{
    // Hybrid mode: V_dequant output is configured
    KVCacheAppendStage::Params params;
    params.K = K_q8.get();
    params.V = V_q8.get();
    params.kv_cache = nullptr; // Not needed for getDeclaredOutputs
    params.layer_idx = 0;
    params.num_tokens = 9;
    params.V_dequant_out = V_dequant.get(); // Hybrid mode

    KVCacheAppendStage stage(params);

    EXPECT_TRUE(stage.producesVDequant());

    auto outputs = stage.getDeclaredOutputs();
    ASSERT_EQ(outputs.size(), 1u) << "Hybrid mode should declare V_dequant output";

    const auto &v_desc = outputs[0];
    EXPECT_EQ(v_desc.name, "V_dequant");
    EXPECT_EQ(v_desc.role, BufferRole::OUTPUT);
    EXPECT_EQ(v_desc.tensor_type, BufferTensorType::FP32);
    EXPECT_TRUE(v_desc.hasProducer());
    EXPECT_EQ(v_desc.producer_stage, "kv_append");
    EXPECT_TRUE(v_desc.validate_populated);
}

TEST_F(StageOutputManifestTest, KVCacheAppend_VDequantShapeMatches)
{
    // Verify shape is correctly propagated
    KVCacheAppendStage::Params params;
    params.K = K_q8.get();
    params.V = V_q8.get();
    params.kv_cache = nullptr;
    params.layer_idx = 0;
    params.num_tokens = 9;
    params.V_dequant_out = V_dequant.get();

    KVCacheAppendStage stage(params);

    auto outputs = stage.getDeclaredOutputs();
    ASSERT_EQ(outputs.size(), 1u);

    const auto &v_desc = outputs[0];
    ASSERT_EQ(v_desc.shape.size(), 2u);
    EXPECT_EQ(v_desc.shape[0], 9u);   // seq_len
    EXPECT_EQ(v_desc.shape[1], 128u); // kv_dim
}

// =============================================================================
// Buffer Requirements vs Declared Outputs Tests
// =============================================================================

TEST_F(StageOutputManifestTest, BufferRequirements_IncludesVDequantAsOutput)
{
    // Verify getBufferRequirements also includes V_dequant
    KVCacheAppendStage::Params params;
    params.K = K_q8.get();
    params.V = V_q8.get();
    params.kv_cache = nullptr;
    params.layer_idx = 0;
    params.num_tokens = 9;
    params.V_dequant_out = V_dequant.get();

    KVCacheAppendStage stage(params);

    auto reqs = stage.getBufferRequirements();

    // Should have K (input), V (input), V_dequant (output)
    ASSERT_EQ(reqs.buffers.size(), 3u);

    // Find V_dequant in requirements
    const BufferDescriptor *v_dequant_desc = nullptr;
    for (const auto &buf : reqs.buffers)
    {
        if (buf.name == "V_dequant")
        {
            v_dequant_desc = &buf;
            break;
        }
    }

    ASSERT_NE(v_dequant_desc, nullptr) << "V_dequant should be in buffer requirements";
    EXPECT_EQ(v_dequant_desc->role, BufferRole::OUTPUT);
    EXPECT_EQ(v_dequant_desc->tensor_type, BufferTensorType::FP32);
}

TEST_F(StageOutputManifestTest, DeclaredOutputs_SubsetOfBufferRequirements)
{
    // getDeclaredOutputs should only contain outputs with contracts
    // getBufferRequirements contains all buffers (inputs + outputs)
    KVCacheAppendStage::Params params;
    params.K = K_q8.get();
    params.V = V_q8.get();
    params.kv_cache = nullptr;
    params.layer_idx = 0;
    params.num_tokens = 9;
    params.V_dequant_out = V_dequant.get();

    KVCacheAppendStage stage(params);

    auto outputs = stage.getDeclaredOutputs();
    auto reqs = stage.getBufferRequirements();

    // All declared outputs should be in buffer requirements
    for (const auto &output : outputs)
    {
        bool found = false;
        for (const auto &req : reqs.buffers)
        {
            if (req.name == output.name)
            {
                found = true;
                EXPECT_EQ(req.role, BufferRole::OUTPUT)
                    << "Declared output " << output.name << " should have OUTPUT role in requirements";
                break;
            }
        }
        EXPECT_TRUE(found) << "Declared output " << output.name << " should be in buffer requirements";
    }
}

// =============================================================================
// Default IComputeStage Behavior Tests
// =============================================================================

/**
 * @brief Mock stage that doesn't override getDeclaredOutputs
 */
class MockStageWithDefaults : public IComputeStage
{
public:
    explicit MockStageWithDefaults(DeviceId device = DeviceId::cpu())
        : IComputeStage(device) {}

    bool execute(IDeviceContext *) override { return true; }
    ComputeStageType type() const override { return ComputeStageType::COPY; }
    bool supportsBackend(ComputeBackendType) const override { return true; }
    size_t estimatedFlops() const override { return 0; }
    StageDumpInfo buildDumpInfoImpl() const override { return {}; }
};

TEST_F(StageOutputManifestTest, DefaultStage_EmptyDeclaredOutputs)
{
    MockStageWithDefaults stage;

    auto outputs = stage.getDeclaredOutputs();
    EXPECT_TRUE(outputs.empty()) << "Default getDeclaredOutputs should return empty";
}

TEST_F(StageOutputManifestTest, DefaultStage_EmptyBufferRequirements)
{
    MockStageWithDefaults stage;

    auto reqs = stage.getBufferRequirements();
    EXPECT_TRUE(reqs.empty()) << "Default getBufferRequirements should return empty";
}
