/**
 * @file Test__OrchestrationRunnerModelValidation.cpp
 * @brief Unit tests for OrchestrationRunner model file validation.
 *
 * Verifies that buildExecutionPlan() (called from initialize()) hard-fails
 * when the model file does not exist or is not valid GGUF, instead of silently
 * falling back to defaults.
 */

#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>

#include "execution/runner/OrchestrationRunner.h"
#include "execution/mpi_orchestration/IExecutionPlanBuilder.h"
#include "config/OrchestrationConfig.h"

using namespace llaminar2;

namespace
{

    // =========================================================================
    // Stub plan builder — never reached in failure tests
    // =========================================================================

    class StubPlanBuilder : public IExecutionPlanBuilder
    {
    public:
        std::vector<RankExecutionPlan> buildAllPlans(
            const OrchestrationConfig &,
            const ModelConfig &,
            const ClusterInventory &) override
        {
            return {};
        }

        RankExecutionPlan buildPlanForRank(
            const OrchestrationConfig &,
            const ModelConfig &model_config,
            const ClusterInventory &,
            int) override
        {
            // Record model config so we can verify defaults path
            last_model_config_ = model_config;
            return RankExecutionPlan{};
        }

        std::vector<std::string> validateConfig(
            const OrchestrationConfig &,
            const ModelConfig &,
            const ClusterInventory &) override
        {
            return {}; // No errors
        }

        ModelConfig last_model_config_{};
    };

    // =========================================================================
    // Test Fixture
    // =========================================================================

    class Test__OrchestrationRunnerModelValidation : public ::testing::Test
    {
    protected:
        OrchestrationConfig makeConfig(const std::string &model_path)
        {
            OrchestrationConfig config = OrchestrationConfig::defaults();
            config.model_path = model_path;
            config.tp_degree = 1;
            config.pp_degree = 1;
            return config;
        }
    };

    // =========================================================================
    // Tests
    // =========================================================================

    TEST_F(Test__OrchestrationRunnerModelValidation, FailsWhenModelFileDoesNotExist)
    {
        auto config = makeConfig("/nonexistent/path/to/model.gguf");
        auto builder = std::make_unique<StubPlanBuilder>();

        OrchestrationRunner runner(std::move(config), std::move(builder));
        EXPECT_FALSE(runner.initialize());
        EXPECT_NE(runner.lastError().find("Model file not found"), std::string::npos)
            << "Error was: " << runner.lastError();
    }

    TEST_F(Test__OrchestrationRunnerModelValidation, FailsWhenModelFileIsInvalidGGUF)
    {
        // Create a temp file with garbage content (not valid GGUF)
        auto tmp_path = std::filesystem::temp_directory_path() / "invalid_model_test.gguf";
        {
            std::ofstream out(tmp_path, std::ios::binary);
            out << "this is not a valid GGUF file";
        }

        auto config = makeConfig(tmp_path.string());
        auto builder = std::make_unique<StubPlanBuilder>();

        OrchestrationRunner runner(std::move(config), std::move(builder));
        EXPECT_FALSE(runner.initialize());
        // Should mention the file path in the error
        EXPECT_NE(runner.lastError().find(tmp_path.string()), std::string::npos)
            << "Error was: " << runner.lastError();
        // Should NOT say "Model file not found" — it exists but is invalid
        EXPECT_EQ(runner.lastError().find("Model file not found"), std::string::npos)
            << "Error was: " << runner.lastError();

        std::filesystem::remove(tmp_path);
    }

    TEST_F(Test__OrchestrationRunnerModelValidation, SucceedsWithEmptyPathUsingDefaults)
    {
        // Empty model_path is the testing-only path that uses defaults
        auto config = makeConfig("");
        auto builder_raw = new StubPlanBuilder();
        auto builder = std::unique_ptr<IExecutionPlanBuilder>(builder_raw);

        OrchestrationRunner runner(std::move(config), std::move(builder));

        // initialize() should get past buildExecutionPlan() without error
        // (it may fail later on graph construction, but the plan phase succeeds)
        // We just verify it doesn't fail with a model-related error
        bool result = runner.initialize();
        if (!result)
        {
            // If it failed, it should NOT be due to model file validation
            EXPECT_EQ(runner.lastError().find("Model file not found"), std::string::npos)
                << "Error was: " << runner.lastError();
            EXPECT_EQ(runner.lastError().find("Failed to read model metadata"), std::string::npos)
                << "Error was: " << runner.lastError();
        }
    }

    TEST_F(Test__OrchestrationRunnerModelValidation, ErrorMessageIncludesFilePath)
    {
        const std::string path = "/some/specific/path/mymodel.gguf";
        auto config = makeConfig(path);
        auto builder = std::make_unique<StubPlanBuilder>();

        OrchestrationRunner runner(std::move(config), std::move(builder));
        EXPECT_FALSE(runner.initialize());
        EXPECT_NE(runner.lastError().find(path), std::string::npos)
            << "Error should contain the file path. Error was: " << runner.lastError();
    }

} // namespace
