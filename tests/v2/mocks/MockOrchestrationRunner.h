/**
 * @file MockOrchestrationRunner.h
 * @brief Mock implementation of IOrchestrationRunner for unit testing
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "execution/runner/IOrchestrationRunner.h"
#include "execution/runner/IOrchestrationRunnerFactory.h"
#include <gmock/gmock.h>

namespace llaminar2::test
{

    /**
     * @brief Mock implementation of IOrchestrationRunner
     *
     * Use this mock in unit tests to verify code that depends on
     * IOrchestrationRunner without requiring actual model loading
     * or device initialization.
     *
     * Example usage:
     * @code
     * auto mock = std::make_unique<MockOrchestrationRunner>();
     * EXPECT_CALL(*mock, initialize()).WillOnce(testing::Return(true));
     * EXPECT_CALL(*mock, isInitialized()).WillRepeatedly(testing::Return(true));
     *
     * // Inject mock into code under test
     * MyClass obj(std::move(mock));
     * @endcode
     */
    class MockOrchestrationRunner : public IOrchestrationRunner
    {
    public:
        MockOrchestrationRunner()
        {
            // Set up default return values
            ON_CALL(*this, isInitialized()).WillByDefault(testing::Return(false));
            ON_CALL(*this, vocabSize()).WillByDefault(testing::Return(32000));
            ON_CALL(*this, currentPosition()).WillByDefault(testing::Return(0));
            ON_CALL(*this, lastLogits()).WillByDefault(testing::Return(nullptr));
            ON_CALL(*this, lastError()).WillByDefault(testing::ReturnRef(empty_error_));
            ON_CALL(*this, executionPlan()).WillByDefault(testing::ReturnRef(default_plan_));
            ON_CALL(*this, config()).WillByDefault(testing::ReturnRef(default_config_));
        }

        // Lifecycle
        MOCK_METHOD(bool, initialize, (), (override));
        MOCK_METHOD(void, shutdown, (), (override));

        // Inference
        MOCK_METHOD(bool, prefill, (const std::vector<int32_t> &tokens), (override));
        MOCK_METHOD(GenerationResult, decodeStep, (), (override));
        MOCK_METHOD(GenerationResult, generate,
                    (const std::vector<int32_t> &prompt_tokens,
                     int max_new_tokens,
                     const SamplingParams &sampling),
                    (override));
        MOCK_METHOD(bool, maybeApplyMoERebalance, (), (override));

        // Configuration
        MOCK_METHOD(const RankExecutionPlan &, executionPlan, (), (const, override));
        MOCK_METHOD(const OrchestrationConfig &, config, (), (const, override));

        // Status
        MOCK_METHOD(bool, isInitialized, (), (const, override));
        MOCK_METHOD(const std::string &, lastError, (), (const, override));
        MOCK_METHOD(int, vocabSize, (), (const, override));
        MOCK_METHOD(int, currentPosition, (), (const, override));
        MOCK_METHOD(void, clearCache, (), (override));

        // Advanced
        MOCK_METHOD(const float *, lastLogits, (), (const, override));
        MOCK_METHOD(void, setStopTokens, (const std::vector<int32_t> &stop_tokens), (override));
        MOCK_METHOD(std::shared_ptr<ITokenizer>, tokenizer, (), (const, override));
        MOCK_METHOD(const std::string &, architecture, (), (const, override));

        // Snapshot
        MOCK_METHOD(void, enableSnapshotCapture, (const std::string &output_dir), (override));
        MOCK_METHOD(void, disableSnapshotCapture, (), (override));
        MOCK_METHOD(void, clearSnapshots, (), (override));
        MOCK_METHOD(const float *, getSnapshot, (const std::string &key, size_t &out_size), (const, override));
        MOCK_METHOD(std::vector<std::string>, getSnapshotKeys, (), (const, override));

        // Sampling and profiling
        MOCK_METHOD(void, setSamplingParams, (const SamplingParams &params), (override));
        MOCK_METHOD(SamplingParams, getRecommendedSamplingParams, (), (const, override));
        MOCK_METHOD(void, flushStageTimeline, (), (override));
        MOCK_METHOD(void, setSkipLogitsGatherDecode, (bool skip), (override));
        MOCK_METHOD(void, setSkipLogitsGatherPrefill, (bool skip), (override));
        MOCK_METHOD(std::string, getStopThinkingPrompt, (), (const, override));
        MOCK_METHOD(ToolCallFormat, getToolCallFormat, (), (const, override));

        // MPI worker coordination
        MOCK_METHOD(void, runMPIWorkerLoop, (), (override));
        MOCK_METHOD(void, shutdownMPIWorkers, (), (override));
        MOCK_METHOD(void, abortMPIWorkers, (const std::string &reason), (override));
        MOCK_METHOD(void, setMPICoordinatedMode, (bool enabled), (override));

        // =====================================================================
        // Test Helpers
        // =====================================================================

        /**
         * @brief Set up mock to simulate successful initialization
         */
        void simulateInitialized()
        {
            ON_CALL(*this, initialize()).WillByDefault(testing::Return(true));
            ON_CALL(*this, isInitialized()).WillByDefault(testing::Return(true));
        }

        /**
         * @brief Set up mock to simulate failed initialization
         */
        void simulateInitializeFailed(const std::string &error)
        {
            error_ = error;
            ON_CALL(*this, initialize()).WillByDefault(testing::Return(false));
            ON_CALL(*this, isInitialized()).WillByDefault(testing::Return(false));
            ON_CALL(*this, lastError()).WillByDefault(testing::ReturnRef(error_));
        }

        /**
         * @brief Set up mock to return specific generation result
         */
        void setGenerationResult(const GenerationResult &result)
        {
            gen_result_ = result;
            ON_CALL(*this, generate(testing::_, testing::_, testing::_))
                .WillByDefault(testing::Return(gen_result_));
        }

        /**
         * @brief Set execution plan for testing
         */
        void setExecutionPlan(const RankExecutionPlan &plan)
        {
            plan_ = plan;
            ON_CALL(*this, executionPlan()).WillByDefault(testing::ReturnRef(plan_));
        }

        /**
         * @brief Set config for testing
         */
        void setConfig(const OrchestrationConfig &config)
        {
            config_ = config;
            ON_CALL(*this, config()).WillByDefault(testing::ReturnRef(config_));
        }

    private:
        std::string empty_error_;
        std::string error_;
        RankExecutionPlan default_plan_;
        RankExecutionPlan plan_;
        OrchestrationConfig default_config_;
        OrchestrationConfig config_;
        GenerationResult gen_result_;
    };

    /**
     * @brief Mock implementation of IOrchestrationRunnerFactory
     */
    class MockOrchestrationRunnerFactory : public IOrchestrationRunnerFactory
    {
    public:
        MOCK_METHOD(std::unique_ptr<IOrchestrationRunner>, createFromArgs,
                    (int argc, const char *argv[]), (override));
        MOCK_METHOD(std::unique_ptr<IOrchestrationRunner>, createFromConfig,
                    (const std::string &config_path), (override));
        MOCK_METHOD(std::unique_ptr<IOrchestrationRunner>, createFromOrchestrationConfig,
                    (OrchestrationConfig config), (override));
        MOCK_METHOD(std::unique_ptr<IOrchestrationRunner>, createSimple,
                    (const std::string &model_path, const std::string &device_spec), (override));
    };

} // namespace llaminar2::test
