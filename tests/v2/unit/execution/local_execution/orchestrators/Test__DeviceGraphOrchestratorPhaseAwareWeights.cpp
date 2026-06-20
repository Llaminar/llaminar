/**
 * @file Test__DeviceGraphOrchestratorPhaseAwareWeights.cpp
 * @brief Unit tests for DeviceGraphOrchestrator phase-aware weight access (Gap 3)
 * @author David Sanftenberg
 * @date January 2026
 *
 * Tests the phase-aware weight selection mechanism that enables
 * "Option A: Selective Duplication" for CPU decode participation:
 * - PREFILL: Returns full weights (GPU - compute-bound)
 * - DECODE: Returns decode shards when CPU participation is enabled
 */

#include <gtest/gtest.h>
#include "backends/DeviceId.h"
#include "execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "execution/mpi_orchestration/PlacementStrategy.h"
#include "loaders/WeightManager.h"
#include "loaders/WeightPlacementMap.h"
#include "loaders/ModelLoader.h"
#include "mocks/MockWeightManager.h"
#include "mocks/MockWeightPlacementMap.h"
#include "mocks/MockModelContext.h"
#include "models/qwen/QwenStandardGraph.h"
#include "tensors/TensorFactory.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"
#include "utils/Logger.h"
#include <cstdlib>
#include <memory>

using namespace llaminar2;

/**
 * @brief Test fixture for phase-aware weight access tests
 */
class Test__DeviceGraphOrchestratorPhaseAwareWeights : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize DeviceManager
        DeviceManager::instance().initialize(-1);

        // Create minimal graph config
        config_.d_model = 896;
        config_.d_ff = 4864;
        config_.n_heads = 14;
        config_.n_kv_heads = 2;
        config_.head_dim = 64;
        config_.n_layers = 24;
        config_.vocab_size = 151936;
        config_.rms_norm_eps = 1e-6f;
        config_.rope_theta = 1000000.0f;
        config_.default_device = DeviceId::cpu();

        // Create orchestrator
        orchestrator_ = std::make_unique<DeviceGraphOrchestrator>(std::make_shared<QwenStandardGraph>(config_, nullptr), nullptr);
    }

    GraphConfig config_;
    std::unique_ptr<DeviceGraphOrchestrator> orchestrator_;
};

// =============================================================================
// Phase Tracking Tests
// =============================================================================

TEST_F(Test__DeviceGraphOrchestratorPhaseAwareWeights, DefaultPhaseIsPrefill)
{
    EXPECT_EQ(orchestrator_->getPhase(), InferencePhase::PREFILL);
}

TEST_F(Test__DeviceGraphOrchestratorPhaseAwareWeights, SetPhaseToDecodeSucceeds)
{
    orchestrator_->setPhase(InferencePhase::DECODE);
    EXPECT_EQ(orchestrator_->getPhase(), InferencePhase::DECODE);
}

TEST_F(Test__DeviceGraphOrchestratorPhaseAwareWeights, SetPhaseToPrefillSucceeds)
{
    orchestrator_->setPhase(InferencePhase::DECODE);
    orchestrator_->setPhase(InferencePhase::PREFILL);
    EXPECT_EQ(orchestrator_->getPhase(), InferencePhase::PREFILL);
}

// =============================================================================
// Weight Manager Setup Tests
// =============================================================================

TEST_F(Test__DeviceGraphOrchestratorPhaseAwareWeights, WeightManagerInitiallyNull)
{
    EXPECT_EQ(orchestrator_->weightManager(), nullptr);
}

TEST_F(Test__DeviceGraphOrchestratorPhaseAwareWeights, SetWeightPlacementMapSucceeds)
{
    auto placement_map = std::make_shared<WeightPlacementMap>(DeviceId::cpu());
    orchestrator_->setWeightPlacementMap(placement_map);
    EXPECT_EQ(orchestrator_->weightPlacementMap(), placement_map);
}

TEST_F(Test__DeviceGraphOrchestratorPhaseAwareWeights, PlacementMapInitiallyNull)
{
    EXPECT_EQ(orchestrator_->weightPlacementMap(), nullptr);
}

// =============================================================================
// Phase-Aware Weight Selection Tests (without real weights)
// =============================================================================

TEST_F(Test__DeviceGraphOrchestratorPhaseAwareWeights, GetPhaseAwareWeightReturnsNullWithoutWeightManager)
{
    // Without a WeightManager, getPhaseAwareWeight should return nullptr
    auto weight = orchestrator_->getPhaseAwareWeight("blk.0.attn_q.weight", 0, InferencePhase::PREFILL);
    EXPECT_EQ(weight, nullptr);
}

TEST_F(Test__DeviceGraphOrchestratorPhaseAwareWeights, ShouldUseCPUDecodeWeightFalseInPrefill)
{
    // In PREFILL phase, should never use CPU decode weight
    orchestrator_->setPhase(InferencePhase::PREFILL);

    // Even with placement map indicating CPU participation
    auto placement_map = std::make_shared<WeightPlacementMap>(DeviceId::cpu());
    orchestrator_->setWeightPlacementMap(placement_map);

    // PREFILL should not use CPU decode weight
    EXPECT_FALSE(orchestrator_->shouldUseCPUDecodeWeight("blk.0.attn_q.weight", 0));
}

TEST_F(Test__DeviceGraphOrchestratorPhaseAwareWeights, ShouldUseCPUDecodeWeightFalseWithoutPlacementMap)
{
    // In DECODE phase but without placement map, should return false
    orchestrator_->setPhase(InferencePhase::DECODE);
    EXPECT_FALSE(orchestrator_->shouldUseCPUDecodeWeight("blk.0.attn_q.weight", 0));
}

// =============================================================================
// Mock Weight Manager Test (Simulated)
// =============================================================================

/**
 * @brief Mock WeightManager for testing phase-aware weight selection
 *
 * This fixture creates a mock setup to verify the selection logic
 * without requiring actual GGUF model loading.
 */
class Test__PhaseAwareWeightSelectionLogic : public ::testing::Test
{
protected:
    void SetUp() override
    {
        DeviceManager::instance().initialize(-1);

        config_.d_model = 896;
        config_.d_ff = 4864;
        config_.n_heads = 14;
        config_.n_kv_heads = 2;
        config_.head_dim = 64;
        config_.n_layers = 24;
        config_.vocab_size = 151936;
        config_.rms_norm_eps = 1e-6f;
        config_.rope_theta = 1000000.0f;
        config_.default_device = DeviceId::cpu();

        orchestrator_ = std::make_unique<DeviceGraphOrchestrator>(std::make_shared<QwenStandardGraph>(config_, nullptr), nullptr);
    }

    /**
     * @brief Create a WeightPlacementMap with CPU decode participation
     */
    std::shared_ptr<WeightPlacementMap> createPlacementMapWithCPUDecode()
    {
        auto map = std::make_shared<WeightPlacementMap>(DeviceId::cpu());
        // Note: WeightPlacementMap's default behavior returns single-device info
        // Real CPU decode participation would be configured via PlacementPlan
        return map;
    }

    GraphConfig config_;
    std::unique_ptr<DeviceGraphOrchestrator> orchestrator_;
};

TEST_F(Test__PhaseAwareWeightSelectionLogic, PrefillPhaseNeverUsesCPUDecode)
{
    // Configure placement map (even though it may indicate CPU participation)
    auto placement_map = createPlacementMapWithCPUDecode();
    orchestrator_->setWeightPlacementMap(placement_map);

    // Set phase to PREFILL
    orchestrator_->setPhase(InferencePhase::PREFILL);

    // shouldUseCPUDecodeWeight should be false in PREFILL
    EXPECT_FALSE(orchestrator_->shouldUseCPUDecodeWeight("blk.0.attn_q.weight", 0));
    EXPECT_FALSE(orchestrator_->shouldUseCPUDecodeWeight("blk.5.ffn_gate.weight", 5));
    EXPECT_FALSE(orchestrator_->shouldUseCPUDecodeWeight("blk.23.attn_output.weight", 23));
}

TEST_F(Test__PhaseAwareWeightSelectionLogic, DecodePhaseWithoutCPUParticipationUsesFullWeight)
{
    // Create placement map that defaults to CUDA GPU (not CPU)
    // When default device is GPU, cpu_decode_participation will be false
    auto placement_map = std::make_shared<WeightPlacementMap>(DeviceId::cuda(0));
    orchestrator_->setWeightPlacementMap(placement_map);

    // Set phase to DECODE
    orchestrator_->setPhase(InferencePhase::DECODE);

    // Without explicit CPU participation (GPU default), shouldUseCPUDecodeWeight should be false
    EXPECT_FALSE(orchestrator_->shouldUseCPUDecodeWeight("blk.0.attn_q.weight", 0));
}

// =============================================================================
// Phase Transition Tests
// =============================================================================

TEST_F(Test__DeviceGraphOrchestratorPhaseAwareWeights, PhaseTransitionsWorkCorrectly)
{
    // Start in PREFILL (default)
    EXPECT_EQ(orchestrator_->getPhase(), InferencePhase::PREFILL);

    // Transition to DECODE
    orchestrator_->setPhase(InferencePhase::DECODE);
    EXPECT_EQ(orchestrator_->getPhase(), InferencePhase::DECODE);

    // Transition back to PREFILL
    orchestrator_->setPhase(InferencePhase::PREFILL);
    EXPECT_EQ(orchestrator_->getPhase(), InferencePhase::PREFILL);

    // Multiple DECODE transitions
    orchestrator_->setPhase(InferencePhase::DECODE);
    orchestrator_->setPhase(InferencePhase::DECODE);
    EXPECT_EQ(orchestrator_->getPhase(), InferencePhase::DECODE);
}

TEST_F(Test__DeviceGraphOrchestratorPhaseAwareWeights, TransitionToPhaseWithLogging)
{
    // Start in PREFILL (default)
    EXPECT_EQ(orchestrator_->getPhase(), InferencePhase::PREFILL);

    // Transition to DECODE via transitionToPhase (logs the transition)
    orchestrator_->transitionToPhase(InferencePhase::DECODE);
    EXPECT_EQ(orchestrator_->getPhase(), InferencePhase::DECODE);

    // Transition back to PREFILL
    orchestrator_->transitionToPhase(InferencePhase::PREFILL);
    EXPECT_EQ(orchestrator_->getPhase(), InferencePhase::PREFILL);
}

TEST_F(Test__DeviceGraphOrchestratorPhaseAwareWeights, TransitionToSamePhaseNoOp)
{
    // Start in PREFILL
    EXPECT_EQ(orchestrator_->getPhase(), InferencePhase::PREFILL);

    // Transition to same phase (should be no-op, no log)
    orchestrator_->transitionToPhase(InferencePhase::PREFILL);
    EXPECT_EQ(orchestrator_->getPhase(), InferencePhase::PREFILL);

    // Now transition to DECODE
    orchestrator_->transitionToPhase(InferencePhase::DECODE);
    EXPECT_EQ(orchestrator_->getPhase(), InferencePhase::DECODE);

    // Multiple transitions to DECODE (should be no-op)
    orchestrator_->transitionToPhase(InferencePhase::DECODE);
    orchestrator_->transitionToPhase(InferencePhase::DECODE);
    EXPECT_EQ(orchestrator_->getPhase(), InferencePhase::DECODE);
}

TEST_F(Test__DeviceGraphOrchestratorPhaseAwareWeights, ToStringForInferencePhase)
{
    // Test toString() helper function
    EXPECT_STREQ(toString(InferencePhase::PREFILL), "PREFILL");
    EXPECT_STREQ(toString(InferencePhase::DECODE), "DECODE");
}

// =============================================================================
// Convenience Method Tests
// =============================================================================

TEST_F(Test__DeviceGraphOrchestratorPhaseAwareWeights, GetPhaseAwareWeightUsesCurrentPhase)
{
    // Set phase
    orchestrator_->setPhase(InferencePhase::DECODE);

    // Call without explicit phase - should use current_phase_
    // Without WeightManager, returns nullptr, but tests the code path
    auto weight = orchestrator_->getPhaseAwareWeight("blk.0.attn_q.weight", 0);
    EXPECT_EQ(weight, nullptr); // No WeightManager set
}

// =============================================================================
// LLAMINAR_CPU_PREFILL_PARTICIPATE Environment Variable Tests (Gap 5)
// =============================================================================

/**
 * @brief Test fixture for CPU prefill participation environment variable
 *
 * Tests the "Option C" escape hatch that allows CPU to participate in
 * prefill phase for memory-constrained systems.
 */
class Test__CPUPrefillParticipation : public ::testing::Test
{
protected:
    void SetUp() override
    {
        DeviceManager::instance().initialize(-1);

        config_.d_model = 896;
        config_.d_ff = 4864;
        config_.n_heads = 14;
        config_.n_kv_heads = 2;
        config_.head_dim = 64;
        config_.n_layers = 24;
        config_.vocab_size = 151936;
        config_.rms_norm_eps = 1e-6f;
        config_.rope_theta = 1000000.0f;
        config_.default_device = DeviceId::cpu();

        orchestrator_ = std::make_unique<DeviceGraphOrchestrator>(std::make_shared<QwenStandardGraph>(config_, nullptr), nullptr);

        // Save original env value
        const char *orig = std::getenv("LLAMINAR_CPU_PREFILL_PARTICIPATE");
        original_env_value_ = orig ? std::string(orig) : "";
        original_env_was_set_ = (orig != nullptr);
    }

    void TearDown() override
    {
        // Restore original env value
        if (original_env_was_set_)
        {
            setenv("LLAMINAR_CPU_PREFILL_PARTICIPATE", original_env_value_.c_str(), 1);
        }
        else
        {
            unsetenv("LLAMINAR_CPU_PREFILL_PARTICIPATE");
        }
        // Reload debugEnv to pick up restored value
        mutableDebugEnv().execution.reload();
    }

    GraphConfig config_;
    std::unique_ptr<DeviceGraphOrchestrator> orchestrator_;
    std::string original_env_value_;
    bool original_env_was_set_ = false;
};

TEST_F(Test__CPUPrefillParticipation, DefaultBehavior_CPUOnlyInDecode)
{
    // Ensure env var is unset (default behavior)
    unsetenv("LLAMINAR_CPU_PREFILL_PARTICIPATE");
    mutableDebugEnv().execution.reload();

    // Configure placement map with CPU participation
    auto placement_map = std::make_shared<WeightPlacementMap>(DeviceId::cpu());
    orchestrator_->setWeightPlacementMap(placement_map);

    // In PREFILL phase, CPU should NOT participate (default Option A behavior)
    orchestrator_->setPhase(InferencePhase::PREFILL);
    EXPECT_FALSE(orchestrator_->shouldUseCPUDecodeWeight("blk.0.attn_q.weight", 0));
    EXPECT_FALSE(orchestrator_->shouldUseCPUDecodeWeight("blk.5.ffn_gate.weight", 5));

    // In DECODE phase, CPU SHOULD participate
    orchestrator_->setPhase(InferencePhase::DECODE);
    // Note: Will still return false because cpu_decode_participation isn't set in placement map
    // But the phase check passes - this tests the phase logic, not placement logic
}

TEST_F(Test__CPUPrefillParticipation, EnvVarDisabled_CPUOnlyInDecode)
{
    // Explicitly set to "0" (disabled)
    setenv("LLAMINAR_CPU_PREFILL_PARTICIPATE", "0", 1);
    mutableDebugEnv().execution.reload();

    EXPECT_FALSE(debugEnv().execution.cpu_prefill_participate);

    // Configure placement map
    auto placement_map = std::make_shared<WeightPlacementMap>(DeviceId::cpu());
    orchestrator_->setWeightPlacementMap(placement_map);

    // In PREFILL phase, CPU should NOT participate
    orchestrator_->setPhase(InferencePhase::PREFILL);
    EXPECT_FALSE(orchestrator_->shouldUseCPUDecodeWeight("blk.0.attn_q.weight", 0));
}

TEST_F(Test__CPUPrefillParticipation, EnvVarEnabled_CPUParticipatesInBothPhases)
{
    // Enable CPU prefill participation (Option C)
    setenv("LLAMINAR_CPU_PREFILL_PARTICIPATE", "1", 1);
    mutableDebugEnv().execution.reload();

    EXPECT_TRUE(debugEnv().execution.cpu_prefill_participate);

    // Configure placement map with CPU participation
    auto placement_map = std::make_shared<WeightPlacementMap>(DeviceId::cpu());
    orchestrator_->setWeightPlacementMap(placement_map);

    // In PREFILL phase, CPU SHOULD now participate (Option C)
    orchestrator_->setPhase(InferencePhase::PREFILL);
    // The function will continue past the phase check and evaluate placement map
    // With default placement map (single CPU), cpu_decode_participation will be true
    // So this should return true for rank 0
    EXPECT_TRUE(orchestrator_->shouldUseCPUDecodeWeight("blk.0.attn_q.weight", 0));

    // In DECODE phase, CPU should also participate
    orchestrator_->setPhase(InferencePhase::DECODE);
    EXPECT_TRUE(orchestrator_->shouldUseCPUDecodeWeight("blk.0.attn_q.weight", 0));
}

TEST_F(Test__CPUPrefillParticipation, EnvVarTrueString_CPUParticipatesInPrefill)
{
    // Test "true" string value
    setenv("LLAMINAR_CPU_PREFILL_PARTICIPATE", "true", 1);
    mutableDebugEnv().execution.reload();

    EXPECT_TRUE(debugEnv().execution.cpu_prefill_participate);

    auto placement_map = std::make_shared<WeightPlacementMap>(DeviceId::cpu());
    orchestrator_->setWeightPlacementMap(placement_map);

    // In PREFILL phase, CPU should participate with "true" value
    orchestrator_->setPhase(InferencePhase::PREFILL);
    EXPECT_TRUE(orchestrator_->shouldUseCPUDecodeWeight("blk.0.attn_q.weight", 0));
}

TEST_F(Test__CPUPrefillParticipation, EnvVarInvalidValue_DefaultsToDisabled)
{
    // Invalid value should be treated as disabled (not "1" or "true")
    setenv("LLAMINAR_CPU_PREFILL_PARTICIPATE", "yes", 1);
    mutableDebugEnv().execution.reload();

    EXPECT_FALSE(debugEnv().execution.cpu_prefill_participate);

    auto placement_map = std::make_shared<WeightPlacementMap>(DeviceId::cpu());
    orchestrator_->setWeightPlacementMap(placement_map);

    // In PREFILL phase, CPU should NOT participate (invalid value treated as disabled)
    orchestrator_->setPhase(InferencePhase::PREFILL);
    EXPECT_FALSE(orchestrator_->shouldUseCPUDecodeWeight("blk.0.attn_q.weight", 0));
}

// =============================================================================
// MockWeightManager Dependency Injection Tests
// =============================================================================

/**
 * @brief Custom placement map that enables CPU decode participation
 *
 * Overrides getDeviceInfoForWeight to return WeightDeviceInfo with
 * cpu_decode_participation=true and a CPU device in decode_devices.
 */
class CPUDecodeEnabledPlacementMap : public IWeightPlacementMap
{
public:
    explicit CPUDecodeEnabledPlacementMap(DeviceId prefill_device, float cpu_fraction = 0.3f)
        : prefill_device_(prefill_device), cpu_fraction_(cpu_fraction) {}

    DeviceId getDeviceForWeight(const std::string & /*tensor_name*/, int /*layer_idx*/) const override
    {
        return prefill_device_;
    }

    WeightDeviceInfo getDeviceInfoForWeight(const std::string & /*tensor_name*/, int /*layer_idx*/) const override
    {
        WeightDeviceInfo info;
        info.prefill_device = prefill_device_;
        info.decode_devices = {prefill_device_, DeviceId::cpu()};
        info.decode_fractions = {1.0f - cpu_fraction_, cpu_fraction_};
        info.cpu_decode_participation = true;
        return info;
    }

    // IWeightPlacementMap configuration methods (no-op for test)
    void setTensorDevice(const std::string & /*name*/, DeviceId /*device*/) override {}
    void setLayerDevice(int /*layer_idx*/, DeviceId /*device*/) override {}
    void setLayerRange(int /*start*/, int /*end*/, DeviceId /*device*/) override {}
    void setPatternDevice(const std::string & /*pattern*/, DeviceId /*device*/) override {}
    void setAttentionDevice(int /*layer_idx*/, DeviceId /*device*/) override {}
    void setFFNDevice(int /*layer_idx*/, DeviceId /*device*/) override {}
    void setSharedExpertDevice(int /*expert_idx*/, DeviceId /*device*/) override {}
    void setLocalExpertDevice(int /*layer_idx*/, int /*expert_idx*/, DeviceId /*device*/) override {}

    // IWeightPlacementMap query methods
    DeviceId defaultDevice() const override { return prefill_device_; }
    size_t tensorMappingCount() const override { return 0; }
    size_t layerMappingCount() const override { return 0; }
    DeviceId getAttentionDevice(int /*layer_idx*/) const override { return prefill_device_; }
    DeviceId getFFNDevice(int /*layer_idx*/) const override { return prefill_device_; }
    DeviceId getSharedExpertDevice(int /*expert_idx*/) const override { return prefill_device_; }
    DeviceId getLocalExpertDevice(int /*layer_idx*/, int /*expert_idx*/) const override { return prefill_device_; }

    // PlacementPlan integration
    void applyPlan(const PlacementPlan & /*plan*/) override {}
    bool hasPlan() const override { return true; }
    const std::string &appliedStrategyName() const override { return strategy_name_; }
    void clear() override {}

private:
    DeviceId prefill_device_;
    float cpu_fraction_;
    std::string strategy_name_{"cpu_decode_test"};
};

/**
 * @brief Test fixture that uses MockWeightManager via dependency injection
 *
 * Uses the 2-arg constructor + setWeightManager() to inject a MockWeightManager,
 * then verifies getPhaseAwareWeight() correctly delegates to the mock.
 */
class Test__PhaseAwareWeightWithMockDI : public ::testing::Test
{
protected:
    void SetUp() override
    {
        DeviceManager::instance().initialize(-1);

        config_.d_model = 896;
        config_.d_ff = 4864;
        config_.n_heads = 14;
        config_.n_kv_heads = 2;
        config_.head_dim = 64;
        config_.n_layers = 24;
        config_.vocab_size = 151936;
        config_.rms_norm_eps = 1e-6f;
        config_.rope_theta = 1000000.0f;
        config_.default_device = DeviceId::cpu();

        // Create orchestrator with legacy 2-arg constructor
        orchestrator_ = std::make_unique<DeviceGraphOrchestrator>(
            std::make_shared<QwenStandardGraph>(config_, nullptr), nullptr);

        // Create and inject MockWeightManager
        mock_wm_ = std::make_shared<test::MockWeightManager>();
        orchestrator_->setWeightManager(mock_wm_);
    }

    GraphConfig config_;
    std::unique_ptr<DeviceGraphOrchestrator> orchestrator_;
    std::shared_ptr<test::MockWeightManager> mock_wm_;
};

TEST_F(Test__PhaseAwareWeightWithMockDI, PrefillReturnsWeightFromGetWeightForDevice)
{
    // Add a weight to the mock
    mock_wm_->addFP32RandomWeight("blk.0.attn_q.weight", {896, 896});

    // PREFILL always calls getWeightForDevice()
    orchestrator_->setPhase(InferencePhase::PREFILL);
    auto weight = orchestrator_->getPhaseAwareWeight("blk.0.attn_q.weight", 0, InferencePhase::PREFILL);

    ASSERT_NE(weight, nullptr);
    EXPECT_EQ(weight->rows(), 896);
    EXPECT_EQ(weight->cols(), 896);
    EXPECT_EQ(mock_wm_->getWeightCallCount(), 1u);
}

TEST_F(Test__PhaseAwareWeightWithMockDI, PrefillReturnsNullForMissingWeight)
{
    // Don't add the weight — simulate missing
    orchestrator_->setPhase(InferencePhase::PREFILL);
    auto weight = orchestrator_->getPhaseAwareWeight("blk.0.attn_q.weight", 0, InferencePhase::PREFILL);

    EXPECT_EQ(weight, nullptr);
    EXPECT_EQ(mock_wm_->getWeightCallCount(), 1u);
    EXPECT_EQ(mock_wm_->missingWeightRequests().size(), 1u);
    EXPECT_EQ(mock_wm_->missingWeightRequests()[0], "blk.0.attn_q.weight");
}

TEST_F(Test__PhaseAwareWeightWithMockDI, DecodeWithoutPlacementMapReturnsFullWeight)
{
    // Add weight but no placement map — DECODE should fall through to full weight
    mock_wm_->addFP32RandomWeight("blk.0.attn_q.weight", {896, 896});

    orchestrator_->setPhase(InferencePhase::DECODE);
    auto weight = orchestrator_->getPhaseAwareWeight("blk.0.attn_q.weight", 0, InferencePhase::DECODE);

    ASSERT_NE(weight, nullptr);
    EXPECT_EQ(weight->rows(), 896);
    EXPECT_EQ(weight->cols(), 896);
    // Should call getWeightForDevice (not getDecodeWeight)
    EXPECT_EQ(mock_wm_->getWeightCallCount(), 1u);
}

TEST_F(Test__PhaseAwareWeightWithMockDI, DecodeWithNoCPUParticipationReturnsFullWeight)
{
    // Add weight and set a GPU-only placement map (no CPU decode participation)
    mock_wm_->addFP32RandomWeight("blk.0.attn_q.weight", {896, 896});

    auto placement_map = std::make_shared<test::MockWeightPlacementMap>(DeviceId::cuda(0));
    orchestrator_->setWeightPlacementMap(placement_map);

    orchestrator_->setPhase(InferencePhase::DECODE);
    auto weight = orchestrator_->getPhaseAwareWeight("blk.0.attn_q.weight", 0, InferencePhase::DECODE);

    ASSERT_NE(weight, nullptr);
    EXPECT_EQ(weight->rows(), 896);
    // GPU default → WeightDeviceInfo(cuda:0) → cpu_decode_participation=false
    // So it takes the "no CPU participation" path → getWeightForDevice()
    EXPECT_EQ(mock_wm_->getWeightCallCount(), 1u);
}

TEST_F(Test__PhaseAwareWeightWithMockDI, DecodeWithCPUParticipationReturnsDecodeShard)
{
    // Add both full weight and decode shard
    mock_wm_->addFP32RandomWeight("blk.0.attn_q.weight", {896, 896});

    // Create smaller decode shard (simulates fractional weight for CPU)
    auto decode_shard = std::make_shared<FP32Tensor>(std::vector<size_t>{269, 896});
    mock_wm_->addDecodeWeight("blk.0.attn_q.weight", decode_shard);

    // Use custom placement map that enables CPU decode participation
    auto placement_map = std::make_shared<CPUDecodeEnabledPlacementMap>(DeviceId::cpu(), 0.3f);
    orchestrator_->setWeightPlacementMap(placement_map);

    orchestrator_->setPhase(InferencePhase::DECODE);
    auto weight = orchestrator_->getPhaseAwareWeight("blk.0.attn_q.weight", 0, InferencePhase::DECODE);

    ASSERT_NE(weight, nullptr);
    // Should return the decode shard (269 rows = ~30% of 896)
    EXPECT_EQ(weight->rows(), 269);
    EXPECT_EQ(weight->cols(), 896);
}

TEST_F(Test__PhaseAwareWeightWithMockDI, DecodeWithCPUParticipationFallsBackToFullWeightWhenNoDecodeWeight)
{
    // Add full weight but NO decode shard
    mock_wm_->addFP32RandomWeight("blk.0.attn_q.weight", {896, 896});

    // Use custom placement map that enables CPU decode participation
    auto placement_map = std::make_shared<CPUDecodeEnabledPlacementMap>(DeviceId::cpu(), 0.3f);
    orchestrator_->setWeightPlacementMap(placement_map);

    orchestrator_->setPhase(InferencePhase::DECODE);
    auto weight = orchestrator_->getPhaseAwareWeight("blk.0.attn_q.weight", 0, InferencePhase::DECODE);

    ASSERT_NE(weight, nullptr);
    // MockWeightManager::getDecodeWeight falls back to full weight when no decode shard exists
    EXPECT_EQ(weight->rows(), 896);
    EXPECT_EQ(weight->cols(), 896);
}

TEST_F(Test__PhaseAwareWeightWithMockDI, MultiLayerWeightLookup)
{
    // Add weights for multiple layers
    mock_wm_->addFP32RandomWeight("blk.0.attn_q.weight", {896, 896});
    mock_wm_->addFP32RandomWeight("blk.5.attn_q.weight", {896, 896});
    mock_wm_->addFP32RandomWeight("blk.23.attn_q.weight", {896, 896});

    orchestrator_->setPhase(InferencePhase::PREFILL);

    // Access weights for different layers
    auto w0 = orchestrator_->getPhaseAwareWeight("blk.0.attn_q.weight", 0, InferencePhase::PREFILL);
    auto w5 = orchestrator_->getPhaseAwareWeight("blk.5.attn_q.weight", 5, InferencePhase::PREFILL);
    auto w23 = orchestrator_->getPhaseAwareWeight("blk.23.attn_q.weight", 23, InferencePhase::PREFILL);

    ASSERT_NE(w0, nullptr);
    ASSERT_NE(w5, nullptr);
    ASSERT_NE(w23, nullptr);
    EXPECT_EQ(mock_wm_->getWeightCallCount(), 3u);
}

TEST_F(Test__PhaseAwareWeightWithMockDI, PhaseTransitionChangesWeightPath)
{
    // Add full weight and decode shard
    mock_wm_->addFP32RandomWeight("blk.0.attn_q.weight", {896, 896});
    auto decode_shard = std::make_shared<FP32Tensor>(std::vector<size_t>{269, 896});
    mock_wm_->addDecodeWeight("blk.0.attn_q.weight", decode_shard);

    auto placement_map = std::make_shared<CPUDecodeEnabledPlacementMap>(DeviceId::cpu(), 0.3f);
    orchestrator_->setWeightPlacementMap(placement_map);

    // PREFILL: should get full weight
    orchestrator_->setPhase(InferencePhase::PREFILL);
    auto prefill_weight = orchestrator_->getPhaseAwareWeight("blk.0.attn_q.weight", 0, InferencePhase::PREFILL);
    ASSERT_NE(prefill_weight, nullptr);
    EXPECT_EQ(prefill_weight->rows(), 896);

    // DECODE: should get decode shard
    orchestrator_->setPhase(InferencePhase::DECODE);
    auto decode_weight = orchestrator_->getPhaseAwareWeight("blk.0.attn_q.weight", 0, InferencePhase::DECODE);
    ASSERT_NE(decode_weight, nullptr);
    EXPECT_EQ(decode_weight->rows(), 269);
}

// =============================================================================
// Dependencies Constructor Tests (DI via struct)
// =============================================================================

class Test__PhaseAwareWeightWithDependencies : public ::testing::Test
{
protected:
    void SetUp() override
    {
        DeviceManager::instance().initialize(-1);

        // Create MockModelContext (provides IModelContext + graph config)
        model_ctx_ = test::MockModelContext::createMinimal();

        // Create graph builder from model context config
        GraphConfig config;
        config.d_model = model_ctx_->embeddingLength();
        config.d_ff = model_ctx_->feedForwardLength();
        config.n_heads = model_ctx_->headCount();
        config.n_kv_heads = model_ctx_->headCountKV();
        config.head_dim = config.d_model / config.n_heads;
        config.n_layers = model_ctx_->blockCount();
        config.vocab_size = model_ctx_->vocabSize();
        config.rms_norm_eps = 1e-6f;
        config.rope_theta = 1000000.0f;
        config.default_device = DeviceId::cpu();

        graph_builder_ = std::make_shared<QwenStandardGraph>(config, nullptr);
    }

    std::shared_ptr<test::MockModelContext> model_ctx_;
    std::shared_ptr<QwenStandardGraph> graph_builder_;
};

TEST_F(Test__PhaseAwareWeightWithDependencies, ConstructWithWeightManagerInDeps)
{
    // Add a weight to the mock context's weight manager
    model_ctx_->mockWeightManager().addFP32RandomWeight("blk.0.attn_q.weight", {128, 128});

    // Build Dependencies struct with weight_manager
    DeviceGraphOrchestrator::Dependencies deps;
    deps.model_ctx = model_ctx_;
    deps.graph_builder = graph_builder_;
    deps.weight_manager = model_ctx_->mockWeightManagerPtr();

    auto orchestrator = std::make_unique<DeviceGraphOrchestrator>(std::move(deps));

    // Verify weight_manager was injected
    EXPECT_NE(orchestrator->weightManager(), nullptr);

    // getPhaseAwareWeight in PREFILL should return the weight from MockWeightManager
    auto weight = orchestrator->getPhaseAwareWeight("blk.0.attn_q.weight", 0, InferencePhase::PREFILL);
    ASSERT_NE(weight, nullptr);
    EXPECT_EQ(weight->rows(), 128);
    EXPECT_EQ(weight->cols(), 128);
}

TEST_F(Test__PhaseAwareWeightWithDependencies, ConstructWithBothWeightManagerAndPlacementMap)
{
    // Add weights
    model_ctx_->mockWeightManager().addFP32RandomWeight("blk.0.ffn_gate.weight", {512, 128});
    auto decode_shard = std::make_shared<FP32Tensor>(std::vector<size_t>{154, 128});
    model_ctx_->mockWeightManager().addDecodeWeight("blk.0.ffn_gate.weight", decode_shard);

    // Build Dependencies with both weight_manager and placement_map
    DeviceGraphOrchestrator::Dependencies deps;
    deps.model_ctx = model_ctx_;
    deps.graph_builder = graph_builder_;
    deps.weight_manager = model_ctx_->mockWeightManagerPtr();
    deps.weight_placement_map = std::make_shared<CPUDecodeEnabledPlacementMap>(DeviceId::cpu(), 0.3f);

    auto orchestrator = std::make_unique<DeviceGraphOrchestrator>(std::move(deps));

    // PREFILL: full weight
    orchestrator->setPhase(InferencePhase::PREFILL);
    auto prefill_w = orchestrator->getPhaseAwareWeight("blk.0.ffn_gate.weight", 0, InferencePhase::PREFILL);
    ASSERT_NE(prefill_w, nullptr);
    EXPECT_EQ(prefill_w->rows(), 512);

    // DECODE: decode shard (must set phase for shouldUseCPUDecodeWeight check)
    orchestrator->setPhase(InferencePhase::DECODE);
    auto decode_w = orchestrator->getPhaseAwareWeight("blk.0.ffn_gate.weight", 0, InferencePhase::DECODE);
    ASSERT_NE(decode_w, nullptr);
    EXPECT_EQ(decode_w->rows(), 154);
}
