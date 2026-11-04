/**
 * @file Test__WeightManager_PPSafety.cpp
 * @brief Unit tests for WeightManager PP (pipeline parallel) safety fixes
 *
 * Tests verify:
 * 1. isWeightInLayerRange: tied embeddings check for LM head stage
 *    - Model WITH separate output.weight rejects token_embd.weight for LM head stage
 *    - Model WITHOUT output.weight (tied embeddings) allows token_embd.weight
 * 2. tryReleaseReclaimHostRawData: CPU-only host-resident tensor safety
 *    - CPU-only FP32 tensor is NOT released by releaseAllHostWeightData
 *    - Quantized tensor with no device copy is also not released
 * 3. Default WeightPrecision is NATIVE (not CONVERT_TO_FP32)
 *
 * All tests use MockModelLoader (no GGUF file I/O) and test through the public API.
 */

#include <gtest/gtest.h>
#include <memory>
#include <cstring>

#include "loaders/PreparedWeightStore.h"
#include "loaders/WeightManager.h"
#include "tensors/Tensors.h"
#include "mocks/MockModelLoader.h"
#include "utils/CPUWeightPlanTestAdmission.h"
#include "backends/BackendManager.h"

using namespace llaminar2;
using namespace llaminar2::test;

// ============================================================================
// TestableWeightManager — opens lifecycle gates for release decision tests
// ============================================================================

class PPSafetyTestableWeightManager : public WeightManager
{
public:
    PPSafetyTestableWeightManager(IModelLoader &loader,
                                  WeightPrecision precision = WeightPrecision::NATIVE)
        : WeightManager(loader, nullptr, nullptr,
                        WeightDistributionStrategy::LAYER_PARTITIONED,
                        precision)
    {
        // Mark all lifecycle gates open so release decision logic is testable
        markMaterializationComplete();
        markDevicePreparationComplete();
        markGraphMaterializationComplete();
    }
};

// ============================================================================
// Test Fixture
// ============================================================================

class Test__WeightManager_PPSafety : public ::testing::Test
{
protected:
    /** @brief Bind the fixture's complete original CPU source population before preparation. */
    void admit(WeightManager& manager)
    {
        InferenceStrategy original;original.model_id={77};original.devices={DeviceId::cpu()};
        WeightPlan plan(original);
        for(const auto& name:mock_loader_->materializationNames()) {
            WeightRequirement request;request.canonical_name=name;request.role=inferWeightRole(name);
            request.layer=inferWeightLayer(name);request.target_device=DeviceId::cpu();
            if(request.role==WeightRole::Other)request.expected_prepared_kind=PreparedWeightKind::CPUResidentTensor;
            plan.add(std::move(request));
        }
        ASSERT_TRUE(manager.preparedWeightStore()->bindModelIdIfUnset(original.model_id));
        admitCPUWeightPlansForTest(manager,*mock_loader_,{&plan});
    }
    void SetUp() override {if(!hasCPUBackend())initCPUBackend(-1);}
    std::shared_ptr<MockModelLoader> mock_loader_;
};

// ============================================================================
// isWeightInLayerRange: Tied Embeddings Tests (tested via getWeightForDevice)
// ============================================================================

/**
 * @test LM head stage with separate output.weight rejects token_embd.weight
 *
 * When a model has a separate output.weight tensor (non-tied embeddings),
 * the LM head stage should NOT load token_embd.weight — it already has
 * output.weight for the LM head projection.
 */
TEST_F(Test__WeightManager_PPSafety, LMHeadStage_WithSeparateOutputWeight_RejectsEmbedding)
{
    mock_loader_ = MockModelLoader::createMinimal();
    mock_loader_->addQ4_0RandomTensor("token_embd.weight", {256, 64});
    mock_loader_->addQ8_0RandomTensor("output.weight", {256, 64});
    mock_loader_->addFP32RandomTensor("output_norm.weight", {64});

    PPSafetyTestableWeightManager wm(*mock_loader_);
    // LM head stage: layers [12,24), no embedding, has LM head
    wm.setLayerRange(12, 24, false, true);

    // output.weight exists → token_embd.weight should be rejected
    admit(wm);

    auto emb = wm.getWeightForDevice("token_embd.weight", DeviceId::cpu(), 0);
    EXPECT_EQ(emb, nullptr)
        << "LM head stage with separate output.weight must reject token_embd.weight";

    // output.weight and output_norm.weight should still be loaded
    auto out = wm.getWeightForDevice("output.weight", DeviceId::cpu(), 0);
    EXPECT_NE(out, nullptr) << "LM head stage must load output.weight";

    auto norm = wm.getWeightForDevice("output_norm.weight", DeviceId::cpu(), 0);
    EXPECT_NE(norm, nullptr) << "LM head stage must load output_norm.weight";
}

/**
 * @test LM head stage with tied embeddings (no output.weight) allows token_embd.weight
 *
 * When a model does NOT have a separate output.weight tensor, the model uses
 * tied embeddings where token_embd.weight serves as both embedding and LM head.
 */
TEST_F(Test__WeightManager_PPSafety, LMHeadStage_WithTiedEmbeddings_AllowsEmbedding)
{
    // Build a mock WITHOUT output.weight to simulate tied embeddings
    mock_loader_ = MockModelLoaderBuilder()
                       .setArchitecture("qwen2")
                       .setBlockCount(1)
                       .setEmbeddingLength(128)
                       .setHeadCount(4)
                       .setHeadCountKV(2)
                       .setVocabSize(1000)
                       .setContextLength(512)
                       .addEmbeddingLayer()
                       .addTransformerLayer(0)
                       .build();
    // Add output_norm but NOT output.weight — this simulates tied embeddings
    mock_loader_->addFP32RandomTensor("output_norm.weight", {128});

    PPSafetyTestableWeightManager wm(*mock_loader_);
    wm.setLayerRange(12, 24, false, true);

    // No output.weight → token_embd.weight should be allowed (tied)
    admit(wm);

    auto emb = wm.getWeightForDevice("token_embd.weight", DeviceId::cpu(), 0);
    EXPECT_NE(emb, nullptr)
        << "LM head stage with tied embeddings must allow token_embd.weight";
}

/**
 * @test Embedding stage allows token_embd.weight regardless of tied status
 */
TEST_F(Test__WeightManager_PPSafety, EmbeddingStage_AlwaysAllowsEmbedding)
{
    mock_loader_ = MockModelLoader::createMinimal();
    mock_loader_->addQ4_0RandomTensor("token_embd.weight", {256, 64});
    mock_loader_->addQ8_0RandomTensor("output.weight", {256, 64});

    PPSafetyTestableWeightManager wm(*mock_loader_);
    // Embedding stage: layers [0,12), has embedding, no LM head
    wm.setLayerRange(0, 12, true, false);

    admit(wm);

    auto emb = wm.getWeightForDevice("token_embd.weight", DeviceId::cpu(), 0);
    EXPECT_NE(emb, nullptr)
        << "Embedding stage must always allow token_embd.weight";
}

/**
 * @test Middle stage (no embedding, no LM head) rejects token_embd.weight
 */
TEST_F(Test__WeightManager_PPSafety, MiddleStage_RejectsEmbedding)
{
    mock_loader_ = MockModelLoader::createMinimal();
    mock_loader_->addQ4_0RandomTensor("token_embd.weight", {256, 64});

    PPSafetyTestableWeightManager wm(*mock_loader_);
    // Middle stage: layers [6,12), no embedding, no LM head
    wm.setLayerRange(6, 12, false, false);

    admit(wm);

    auto emb = wm.getWeightForDevice("token_embd.weight", DeviceId::cpu(), 0);
    EXPECT_EQ(emb, nullptr)
        << "Middle stage must reject token_embd.weight";
}

/**
 * @test LM head stage with separate output.weight still loads layer weights in range
 *
 * Verifies the tied embeddings fix doesn't break normal layer weight loading.
 */
TEST_F(Test__WeightManager_PPSafety, LMHeadStage_LoadsLayerWeightsInRange)
{
    mock_loader_ = MockModelLoader::createMinimal();
    mock_loader_->addQ4_0RandomTensor("token_embd.weight", {256, 64});
    mock_loader_->addQ8_0RandomTensor("output.weight", {256, 64});
    mock_loader_->addFP32RandomTensor("output_norm.weight", {64});
    mock_loader_->addQ4_0RandomTensor("blk.12.attn_q.weight", {64, 64});
    mock_loader_->addQ4_0RandomTensor("blk.11.attn_q.weight", {64, 64});

    PPSafetyTestableWeightManager wm(*mock_loader_);
    wm.setLayerRange(12, 24, false, true);

    // Layer 12 is in range [12,24)
    admit(wm);

    auto l12 = wm.getWeightForDevice("blk.12.attn_q.weight", DeviceId::cpu(), 0);
    EXPECT_NE(l12, nullptr) << "Layer 12 should be in range";

    // Layer 11 is outside range
    auto l11 = wm.getWeightForDevice("blk.11.attn_q.weight", DeviceId::cpu(), 0);
    EXPECT_EQ(l11, nullptr) << "Layer 11 should NOT be in range";
}

/**
 * @test Qwen3.6 trailing nextn sidecar weights load only on the terminal PP stage.
 *
 * The main graph excludes the trailing nextn/MTP block from ordinary decoder
 * execution. MTP sidecar initialization still needs that block's local
 * attention/FFN tensors, but only on the stage that owns terminal hidden,
 * final norm, and LM head.
 */
TEST_F(Test__WeightManager_PPSafety, LayerRange_AllowsTrailingNextNSidecarWeightsOnlyOnTerminalStage)
{
    mock_loader_ = MockModelLoader::createMinimal();
    mock_loader_->addFP32RandomTensor("blk.64.nextn.eh_proj.weight", {8, 8});
    mock_loader_->addFP32RandomTensor("blk.64.attn_k.weight", {8, 8});
    mock_loader_->addFP32RandomTensor("blk.65.attn_k.weight", {8, 8});

    PPSafetyTestableWeightManager non_terminal(*mock_loader_);
    non_terminal.setLayerRange(0, 32, true, false);
    EXPECT_EQ(non_terminal.getWeightForDevice("blk.64.nextn.eh_proj.weight", DeviceId::cpu(), 0), nullptr)
        << "Non-terminal PP stages must not load the MTP sidecar projection";
    EXPECT_EQ(non_terminal.getWeightForDevice("blk.64.attn_k.weight", DeviceId::cpu(), 0), nullptr)
        << "Non-terminal PP stages must not load MTP sidecar attention weights";

    PPSafetyTestableWeightManager terminal(*mock_loader_);
    terminal.setLayerRange(32, 64, false, true);

    admit(terminal);

    auto sidecar_fc = terminal.getWeightForDevice("blk.64.nextn.eh_proj.weight", DeviceId::cpu(), 0);
    EXPECT_NE(sidecar_fc, nullptr)
        << "Terminal PP stage should load the MTP nextn projection outside the main graph range";

    auto sidecar_attn = terminal.getWeightForDevice("blk.64.attn_k.weight", DeviceId::cpu(), 0);
    EXPECT_NE(sidecar_attn, nullptr)
        << "Terminal PP stage should load MTP sidecar attention weights outside the main graph range";

    auto unrelated_out_of_range = terminal.getWeightForDevice("blk.65.attn_k.weight", DeviceId::cpu(), 0);
    EXPECT_EQ(unrelated_out_of_range, nullptr)
        << "Non-sidecar out-of-range layers must remain filtered";
}

// ============================================================================
// Host Release Safety Tests
// ============================================================================

/**
 * @test CPU-only FP32 tensor is NOT released by releaseAllHostWeightData
 *
 * FP32 tensors on CPU are used live by FloatingPointGemmKernel (oneDNN).
 * Releasing their host data would cause a null pointer dereference during inference.
 */
TEST_F(Test__WeightManager_PPSafety, ReleaseAll_RetainsCPUOnlyFP32Tensor)
{
    mock_loader_ = MockModelLoader::createMinimal();
    mock_loader_->addFP32RandomTensor("token_embd.weight", {64, 64});

    PPSafetyTestableWeightManager wm(*mock_loader_);

    admit(wm);

    auto tensor = wm.getWeightForDevice("token_embd.weight", DeviceId::cpu(), 0);
    ASSERT_NE(tensor, nullptr);

    // CPU-only tensor: no device copy
    ASSERT_FALSE(tensor->deviceValid());

    // releaseAll should NOT release it — no safe device copy exists
    size_t released = wm.releaseAllHostWeightData();
    EXPECT_EQ(released, 0);
    EXPECT_FALSE(tensor->is_raw_data_released())
        << "CPU-only FP32 tensor must not be released";
    EXPECT_NE(tensor->data(), nullptr)
        << "Data pointer must remain valid after releaseAll";
}

/**
 * @brief A typed CPU binding keeps bytes live after accelerator preparation.
 *
 * ExpertOverlay materializes participant plans independently. This deliberately
 * registers the CPU consumer first and an accelerator consumer second, matching
 * the ordering that previously downgraded pointer-keyed metadata and cleared
 * the source vector beneath a retained floating-point expert engine.
 */
TEST_F(Test__WeightManager_PPSafety,
       MaterializedCPUConsumerPreventsAcceleratorPreparedHostReclaim)
{
    mock_loader_ = MockModelLoader::createMinimal();
    mock_loader_->addFP32RandomTensor("blk.0.attn_q.weight", {8, 8});
    PPSafetyTestableWeightManager wm(*mock_loader_);

    const auto materialize_for = [&](DeviceId target, WeightHostPolicy policy)
    {
        InferenceStrategy strategy;
        strategy.mode = WeightInferenceMode::SingleDevice;
        strategy.model_id = ModelContextId{77};
        strategy.devices = {target};
        WeightPlan plan(strategy);

        WeightRequirement requirement;
        requirement.canonical_name = "blk.0.attn_q.weight";
        requirement.target_device = target;
        requirement.lookup_device = DeviceId::cpu();
        requirement.host_policy = policy;
        plan.add(std::move(requirement));
        return wm.materialize(plan);
    };

    admit(wm);

    const auto cpu_weights = materialize_for(
        DeviceId::cpu(), WeightHostPolicy::RequiredForCPUExecution);
    const auto accelerator_weights = materialize_for(
        DeviceId::cuda(0),
        WeightHostPolicy::ReleasableAfterPreparation);
    ASSERT_EQ(cpu_weights.bindings().size(), 1u);
    ASSERT_EQ(accelerator_weights.bindings().size(), 1u);
    ASSERT_NE(cpu_weights.bindings().front().tensor,accelerator_weights.bindings().front().tensor);
    EXPECT_EQ(std::memcmp(cpu_weights.bindings().front().tensor->raw_data(),
        accelerator_weights.bindings().front().tensor->raw_data(),8*8*sizeof(float)),0);

    auto tensor = cpu_weights.bindings().front().tensor_owner;
    ASSERT_NE(tensor, nullptr);
    tensor->setHostResident();
    ASSERT_TRUE(cpu_weights.bindings().front().prepared);
    ASSERT_NE(wm.preparedWeightStore()->gemmKernelLifetime(*cpu_weights.bindings().front().prepared),nullptr);
    ASSERT_TRUE(tensor->hasPreparedDeviceState());

    EXPECT_EQ(wm.releaseAllHostWeightData(), 0u);
    EXPECT_FALSE(tensor->is_raw_data_released());
    EXPECT_NE(tensor->raw_data(), nullptr);
}

/**
 * @test CPU-only quantized tensor is NOT released when no device copy exists
 */
TEST_F(Test__WeightManager_PPSafety, ReleaseAll_RetainsCPUOnlyQuantizedTensor)
{
    mock_loader_ = MockModelLoader::createMinimal();
    mock_loader_->addQ8_0RandomTensor("token_embd.weight", {64, 64});

    PPSafetyTestableWeightManager wm(*mock_loader_);

    admit(wm);

    auto tensor = wm.getWeightForDevice("token_embd.weight", DeviceId::cpu(), 0);
    ASSERT_NE(tensor, nullptr);
    ASSERT_FALSE(tensor->deviceValid());

    size_t released = wm.releaseAllHostWeightData();
    EXPECT_EQ(released, 0);
    EXPECT_FALSE(tensor->is_raw_data_released())
        << "CPU-only quantized tensor must not be released without device copy";
}

/**
 * @test Host-resident tensor IS released when prepared device state exists
 *
 * Verifies the safety fix doesn't break the normal release path for
 * tensors that have been explicitly marked as host-resident (mmap-backed).
 */
TEST_F(Test__WeightManager_PPSafety, ReleaseAll_RetainsFinalCPUEmbeddingWithActualPreparedHandle)
{
    mock_loader_ = MockModelLoader::createMinimal();
    mock_loader_->addQ8_0RandomTensor("token_embd.weight", {64, 64});

    PPSafetyTestableWeightManager wm(*mock_loader_);

    admit(wm);

    auto tensor = wm.getWeightForDevice("token_embd.weight", DeviceId::cpu(), 0);
    ASSERT_NE(tensor, nullptr);
    tensor->setHostResident();
    const auto metadata=wm.weightMetadataRegistry()->metadata(tensor.get());
    ASSERT_TRUE(metadata);
    WeightBinding binding;binding.binding_id=metadata->identity.instance_id;binding.identity=metadata->identity;
    binding.tensor=tensor.get();binding.tensor_owner=tensor;binding.residency=metadata->residency;
    binding.slice=metadata->slice;binding.immutable=true;
    const auto prepared=wm.preparedWeightStore()->prepareEmbedding(binding,64);
    ASSERT_TRUE(wm.preparedWeightStore()->embeddingHandle(prepared)!=nullptr);

    size_t released = wm.releaseAllHostWeightData();
    EXPECT_EQ(released,0u);
    EXPECT_FALSE(tensor->is_raw_data_released())
        << "The actual CPU embedding still reads its one final native root";
}

// ============================================================================
// Default Precision Tests
// ============================================================================

/**
 * @test Default WeightPrecision is NATIVE
 *
 * The WeightManager constructor's default parameter should be NATIVE,
 * not CONVERT_TO_FP32. Using FP32 by default causes FloatingPointGemmKernel
 * to be selected, which doesn't support fused SwiGLU and has host memory
 * lifecycle issues.
 */
TEST_F(Test__WeightManager_PPSafety, DefaultPrecision_IsNative)
{
    mock_loader_ = MockModelLoader::createMinimal();
    mock_loader_->addQ4_0RandomTensor("token_embd.weight", {256, 64});

    // Construct with all defaults (no explicit precision)
    WeightManager wm(*mock_loader_);

    // Load a quantized tensor — should remain quantized (NATIVE), not FP32
    admit(wm);

    auto tensor = wm.getWeightForDevice("token_embd.weight", DeviceId::cpu(), 0);
    ASSERT_NE(tensor, nullptr);

    // With NATIVE precision, Q4_0 tensor stays as Q4_0
    EXPECT_NE(tensor->native_type(), TensorType::FP32)
        << "Default precision should be NATIVE — Q4_0 tensor must not be converted to FP32";
}

/**
 * @test Explicit NATIVE precision keeps quantized format
 */
TEST_F(Test__WeightManager_PPSafety, ExplicitNative_KeepsQuantizedFormat)
{
    mock_loader_ = MockModelLoader::createMinimal();
    mock_loader_->addQ4_0RandomTensor("token_embd.weight", {256, 64});

    WeightManager wm(*mock_loader_, nullptr, nullptr,
                     WeightDistributionStrategy::REPLICATED,
                     WeightPrecision::NATIVE);

    admit(wm);

    auto tensor = wm.getWeightForDevice("token_embd.weight", DeviceId::cpu(), 0);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->native_type(), TensorType::Q4_0)
        << "NATIVE precision must preserve Q4_0 format";
}
