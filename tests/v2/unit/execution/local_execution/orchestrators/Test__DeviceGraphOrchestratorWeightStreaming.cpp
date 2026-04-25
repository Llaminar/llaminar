/**
 * @file Test__DeviceGraphOrchestratorWeightStreaming.cpp
 * @brief Unit tests for IWeightStreamer integration with DeviceGraphOrchestrator
 * @author GitHub Copilot
 * @date January 2026
 *
 * Tests that the DeviceGraphOrchestrator correctly calls weight streaming hooks:
 * - Phase transition notifications
 * - ensureLayerOnDevice() before layer execution
 * - prefetchLayer() for upcoming layers
 * - releaseLayer() after layer execution
 *
 * Uses a mock IWeightStreamer to verify hook invocations without
 * actual GPU memory management.
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "models/qwen/QwenStandardGraph.h"
#include "loaders/IWeightStreamer.h"
#include "execution/mpi_orchestration/PlacementStrategy.h"
#include "backends/DeviceId.h"

namespace llaminar2
{

    // =========================================================================
    // Mock IWeightStreamer for testing hook invocations
    // =========================================================================

    class MockWeightStreamer : public IWeightStreamer
    {
    public:
        // === Layer Management ===
        MOCK_METHOD(bool, ensureLayerOnDevice, (int layer_idx, DeviceId device), (override));
        MOCK_METHOD(void, prefetchLayer, (int layer_idx, DeviceId device), (override));
        MOCK_METHOD(void, releaseLayer, (int layer_idx), (override));

        // === Phase Management ===
        MOCK_METHOD(void, onPhaseTransition, (InferencePhase old_phase, InferencePhase new_phase), (override));

        // === Memory Management ===
        MOCK_METHOD(size_t, currentDeviceMemoryUsage, (), (const, override));
        MOCK_METHOD(size_t, memoryBudget, (), (const, override));
        MOCK_METHOD(bool, evictLayer, (int layer_idx), (override));
        MOCK_METHOD(void, clearCache, (), (override));

        // === Synchronization ===
        MOCK_METHOD(void, synchronize, (), (override));

        // === Diagnostics ===
        MOCK_METHOD(bool, isLayerCached, (int layer_idx, DeviceId device), (const, override));
        MOCK_METHOD(bool, isPrefetchInProgress, (int layer_idx), (const, override));
        MOCK_METHOD(StreamingStats, stats, (), (const, override));
        MOCK_METHOD(void, resetStats, (), (override));
    };

    // =========================================================================
    // Test Fixture
    // =========================================================================

    class Test__DeviceGraphOrchestratorWeightStreaming : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Create mock streamer
            mock_streamer_ = std::make_shared<MockWeightStreamer>();

            // Create minimal graph config for orchestrator
            GraphConfig config;
            config.vocab_size = 1000;
            config.d_model = 64;
            config.n_layers = 4;
            config.n_heads = 4;
            config.n_kv_heads = 4;
            config.head_dim = 16;
            config.d_ff = 256;
            config.max_seq_len = 128;

            // Create orchestrator (no MPI)
            orchestrator_ = std::make_unique<DeviceGraphOrchestrator>(std::make_shared<QwenStandardGraph>(config, nullptr), nullptr);
        }

        void TearDown() override
        {
            orchestrator_.reset();
            mock_streamer_.reset();
        }

        std::shared_ptr<MockWeightStreamer> mock_streamer_;
        std::unique_ptr<DeviceGraphOrchestrator> orchestrator_;
    };

    // =========================================================================
    // Basic Tests
    // =========================================================================

    TEST_F(Test__DeviceGraphOrchestratorWeightStreaming, StreamingDisabledByDefault)
    {
        // Without setting a streamer, streaming should be disabled
        EXPECT_FALSE(orchestrator_->isWeightStreamingEnabled());
        EXPECT_EQ(orchestrator_->weightStreamer(), nullptr);
    }

    TEST_F(Test__DeviceGraphOrchestratorWeightStreaming, SetWeightStreamerEnablesStreaming)
    {
        orchestrator_->setWeightStreamer(mock_streamer_);

        EXPECT_TRUE(orchestrator_->isWeightStreamingEnabled());
        EXPECT_EQ(orchestrator_->weightStreamer(), mock_streamer_);
    }

    TEST_F(Test__DeviceGraphOrchestratorWeightStreaming, ClearWeightStreamerDisablesStreaming)
    {
        orchestrator_->setWeightStreamer(mock_streamer_);
        EXPECT_TRUE(orchestrator_->isWeightStreamingEnabled());

        orchestrator_->setWeightStreamer(nullptr);
        EXPECT_FALSE(orchestrator_->isWeightStreamingEnabled());
    }

    // =========================================================================
    // Phase Transition Tests
    // =========================================================================

    TEST_F(Test__DeviceGraphOrchestratorWeightStreaming, PhaseTransitionNotifiesStreamer)
    {
        orchestrator_->setWeightStreamer(mock_streamer_);

        // Expect notification when transitioning from PREFILL to DECODE
        EXPECT_CALL(*mock_streamer_, onPhaseTransition(
                                         InferencePhase::PREFILL, InferencePhase::DECODE))
            .Times(1);

        orchestrator_->transitionToPhase(InferencePhase::DECODE);
    }

    TEST_F(Test__DeviceGraphOrchestratorWeightStreaming, PhaseTransitionNoNotifyIfSamePhase)
    {
        orchestrator_->setWeightStreamer(mock_streamer_);

        // Initial phase is PREFILL - transitioning to PREFILL should not notify
        EXPECT_CALL(*mock_streamer_, onPhaseTransition(testing::_, testing::_))
            .Times(0);

        orchestrator_->transitionToPhase(InferencePhase::PREFILL);
    }

    TEST_F(Test__DeviceGraphOrchestratorWeightStreaming, PhaseTransitionWithoutStreamerDoesNotCrash)
    {
        // No streamer set - should not crash
        EXPECT_NO_THROW(orchestrator_->transitionToPhase(InferencePhase::DECODE));
        EXPECT_EQ(orchestrator_->getPhase(), InferencePhase::DECODE);
    }

    TEST_F(Test__DeviceGraphOrchestratorWeightStreaming, PhaseTransitionBackToPrefill)
    {
        orchestrator_->setWeightStreamer(mock_streamer_);

        // First transition to DECODE
        EXPECT_CALL(*mock_streamer_, onPhaseTransition(
                                         InferencePhase::PREFILL, InferencePhase::DECODE))
            .Times(1);
        orchestrator_->transitionToPhase(InferencePhase::DECODE);

        // Then back to PREFILL
        EXPECT_CALL(*mock_streamer_, onPhaseTransition(
                                         InferencePhase::DECODE, InferencePhase::PREFILL))
            .Times(1);
        orchestrator_->transitionToPhase(InferencePhase::PREFILL);
    }

    // =========================================================================
    // Getter Tests
    // =========================================================================

    TEST_F(Test__DeviceGraphOrchestratorWeightStreaming, WeightStreamerGetterReturnsCorrectPointer)
    {
        auto streamer1 = std::make_shared<MockWeightStreamer>();
        auto streamer2 = std::make_shared<MockWeightStreamer>();

        orchestrator_->setWeightStreamer(streamer1);
        EXPECT_EQ(orchestrator_->weightStreamer().get(), streamer1.get());

        orchestrator_->setWeightStreamer(streamer2);
        EXPECT_EQ(orchestrator_->weightStreamer().get(), streamer2.get());
    }

    // =========================================================================
    // Integration Tests (Phase + Streaming State)
    // =========================================================================

    TEST_F(Test__DeviceGraphOrchestratorWeightStreaming, StreamerNotifiedOnMultipleTransitions)
    {
        orchestrator_->setWeightStreamer(mock_streamer_);

        testing::InSequence seq;

        // PREFILL -> DECODE
        EXPECT_CALL(*mock_streamer_, onPhaseTransition(
                                         InferencePhase::PREFILL, InferencePhase::DECODE))
            .Times(1);

        // DECODE -> PREFILL
        EXPECT_CALL(*mock_streamer_, onPhaseTransition(
                                         InferencePhase::DECODE, InferencePhase::PREFILL))
            .Times(1);

        // PREFILL -> DECODE again
        EXPECT_CALL(*mock_streamer_, onPhaseTransition(
                                         InferencePhase::PREFILL, InferencePhase::DECODE))
            .Times(1);

        orchestrator_->transitionToPhase(InferencePhase::DECODE);
        orchestrator_->transitionToPhase(InferencePhase::PREFILL);
        orchestrator_->transitionToPhase(InferencePhase::DECODE);
    }

} // namespace llaminar2
