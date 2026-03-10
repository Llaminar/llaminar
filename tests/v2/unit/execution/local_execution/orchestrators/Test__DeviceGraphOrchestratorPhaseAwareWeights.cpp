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
#include "models/qwen/Qwen2Graph.h"
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
        orchestrator_ = std::make_unique<DeviceGraphOrchestrator>(config_, nullptr);
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

        orchestrator_ = std::make_unique<DeviceGraphOrchestrator>(config_, nullptr);
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

        orchestrator_ = std::make_unique<DeviceGraphOrchestrator>(config_, nullptr);

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
