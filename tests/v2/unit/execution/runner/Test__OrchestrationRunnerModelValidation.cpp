/**
 * @file Test__OrchestrationRunnerModelValidation.cpp
 * @brief Unit tests for OrchestrationRunner model and retirement lifecycles.
 *
 * Verifies that buildExecutionPlan() (called from initialize()) hard-fails
 * when the model file does not exist or is not valid GGUF, instead of silently
 * falling back to defaults.
 */

#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>

#include "execution/runner/OrchestrationRunner.h"
#include "execution/runner/ModelContextRetirement.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "execution/local_execution/device/ReusableExecutionWorkspace.h"
#include "execution/mpi_orchestration/IExecutionPlanBuilder.h"
#include "config/OrchestrationConfig.h"
#include "loaders/ModelContext.h"

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

    TEST(Test__ModelContextReuseAuthority, PublishesFinalRetentionOnlyAtReusableSeal)
    {
        ModelContextReuseAuthority authority;
        std::string error;
        EXPECT_FALSE(authority.sealedDeviceMemoryRetention(&error).has_value());
        EXPECT_NE(error.find("no sealed"), std::string::npos) << error;

        ASSERT_TRUE(authority.beginSealing(&error)) << error;
        const std::vector<ModelDeviceMemoryRetention> retention{
            ModelDeviceMemoryRetention{
                .device = DeviceId::rocm(0),
                .prepared_weight_bytes = 1024u,
                .reusable_workspace_bytes = 256u,
            },
        };
        ASSERT_TRUE(authority.publishReusable(retention, &error)) << error;
        const auto sealed = authority.sealedDeviceMemoryRetention(&error);
        ASSERT_TRUE(sealed.has_value()) << error;
        EXPECT_EQ(*sealed, retention);

        ASSERT_TRUE(authority.acquireRunnerExclusive(&error)) << error;
        EXPECT_FALSE(authority.sealedDeviceMemoryRetention(&error).has_value())
            << "A new runner must not inherit a prior generation's retirement BOM";
    }

    TEST(Test__ModelContextReuseAuthority, AcceptsWeightsOnlyParticipantRetention)
    {
        ModelContextReuseAuthority authority;
        std::string error;
        ASSERT_TRUE(authority.beginSealing(&error)) << error;

        const std::vector<ModelDeviceMemoryRetention> retention{
            ModelDeviceMemoryRetention{
                .device = DeviceId::rocm(0),
                .prepared_weight_bytes = 4096u,
                .reusable_workspace_bytes = 0u,
            },
        };
        ASSERT_TRUE(authority.publishReusable(retention, &error)) << error;
        const auto sealed = authority.sealedDeviceMemoryRetention(&error);
        ASSERT_TRUE(sealed.has_value()) << error;
        EXPECT_EQ(*sealed, retention);
    }

    TEST(Test__ModelContextReuseAuthority, RejectsInvalidOrDuplicateRetentionRows)
    {
        ModelContextReuseAuthority authority;
        std::string error;
        ASSERT_TRUE(authority.beginSealing(&error)) << error;
        EXPECT_FALSE(authority.publishReusable(
            {
                ModelDeviceMemoryRetention{
                    .device = DeviceId::cuda(0),
                    .prepared_weight_bytes = 10u,
                    .reusable_workspace_bytes = 1u,
                },
                ModelDeviceMemoryRetention{
                    .device = DeviceId::cuda(0),
                    .prepared_weight_bytes = 20u,
                    .reusable_workspace_bytes = 2u,
                },
            },
            &error));
        EXPECT_NE(error.find("duplicate"), std::string::npos) << error;
        EXPECT_EQ(authority.state(), ModelContextReuseAuthority::State::Sealing);
    }

    TEST(Test__ModelContextReuseAuthority, CpuOnlySealPublishesAnExplicitEmptyBom)
    {
        ModelContextReuseAuthority authority;
        std::string error;
        ASSERT_TRUE(authority.beginSealing(&error)) << error;
        ASSERT_TRUE(authority.publishReusable({}, &error)) << error;
        const auto sealed = authority.sealedDeviceMemoryRetention(&error);
        ASSERT_TRUE(sealed.has_value()) << error;
        EXPECT_TRUE(sealed->empty());
    }

    TEST(Test__ModelContextRetirement,
         RejectsAnIncompleteContractBeforeChangingOwnership)
    {
        ModelContextReuseContract contract;
        contract.reuse_authority =
            std::make_shared<ModelContextReuseAuthority>();

        EXPECT_THROW(
            (void)retireExclusiveModelContextReuseContract(contract),
            std::invalid_argument);
        EXPECT_NE(contract.reuse_authority, nullptr)
            << "Validation failure must leave the caller's lifecycle authority intact";
        EXPECT_EQ(
            contract.reuse_authority->state(),
            ModelContextReuseAuthority::State::RunnerExclusive);
    }

    TEST(Test__ModelContextRetirement,
         RetiresACompleteCpuOnlyContractWithoutInventingAGpuTransaction)
    {
        auto reuse_authority =
            std::make_shared<ModelContextReuseAuthority>();
        std::string error;
        ASSERT_TRUE(reuse_authority->beginSealing(&error)) << error;
        ASSERT_TRUE(reuse_authority->publishReusable({}, &error)) << error;

        ModelContextReuseContract contract;
        contract.context = ModelContext::createForTesting(
            "cpu-only-retirement.gguf");
        contract.reuse_authority = std::move(reuse_authority);
        contract.reusable_execution_workspaces =
            std::make_shared<ReusableExecutionWorkspaceRegistry>();
        PhysicalMemoryPlanBuilder memory_plan;
        memory_plan.add(
            PhysicalMemoryResource{
                .world_rank = 0,
                .device = DeviceId::cpu(),
                .total_bytes = 1024u,
                .admission_available_bytes = 1024u,
            },
            PhysicalMemoryOwner::ModelSourcePayload,
            1u);
        const auto memory_admission = std::make_shared<
            const PhysicalMemoryPlanAdmissionCertificate>(
            memory_plan.build());
        contract.physical_memory_authority =
            std::make_shared<PhysicalMemoryAuthority>(
                memory_admission, 0);

        const ModelContextRetirementReceipt receipt =
            retireExclusiveModelContextReuseContract(contract);

        EXPECT_EQ(receipt.retiredDeviceCount(), 0u);
        EXPECT_EQ(contract.context, nullptr);
        EXPECT_EQ(contract.reuse_authority, nullptr);
        EXPECT_EQ(contract.reusable_execution_workspaces, nullptr);
        EXPECT_EQ(contract.physical_memory_authority, nullptr);
    }

    TEST(Test__ModelContextRetirement,
         TypedCpuOnlyPlanNeverEntersTheGpuBatchAPI)
    {
        PendingExclusiveModelRetirement pending =
            PendingExclusiveModelRetirement::begin({});
        EXPECT_EQ(
            pending.kind(),
            PendingExclusiveModelRetirement::Kind::HostOnly);
        EXPECT_EQ(pending.pendingDeviceCount(), 0u);

        const ModelContextRetirementReceipt receipt = pending.complete();
        EXPECT_EQ(receipt.retiredDeviceCount(), 0u);
        EXPECT_THROW((void)pending.complete(), std::logic_error)
            << "The typed ownership boundary must be consumed exactly once";
    }

} // namespace
