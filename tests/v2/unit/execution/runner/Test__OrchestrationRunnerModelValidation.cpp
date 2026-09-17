/**
 * @file Test__OrchestrationRunnerModelValidation.cpp
 * @brief Unit tests for OrchestrationRunner model and retirement lifecycles.
 *
 * Exercises the runner's shared metadata reader without starting MPI or
 * hardware discovery. Distributed runner initialization and propagation of
 * these errors belong to Test__RankInitializationLifecycleMPI, not Unit.
 * Terminal-policy regressions distinguish ordinary disposal from an exported
 * prepared-model reuse obligation without loading a model or starting a GPU.
 */

#include <gtest/gtest.h>
#include <filesystem>
#include <cstdio>
#include <unistd.h>

#include "planning/PlanningModelMetadata.h"
#include "execution/runner/IOrchestrationRunnerFactory.h"
#include "execution/runner/ModelContextRetirement.h"
#include "execution/moe/MoEOverlayDeviceControllerGraphService.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "execution/local_execution/device/ReusableExecutionWorkspace.h"
#include "loaders/ModelContext.h"

using namespace llaminar2;

namespace
{

    /** @brief Own a unique invalid GGUF without sharing a fixed /tmp pathname. */
    class InvalidModelFile final
    {
    public:
        /** @brief Create a tiny invalid input, never any tensor payload. */
        InvalidModelFile()
        {
            char pattern[] = "/tmp/llaminar-invalid-model-XXXXXX";
            const int descriptor = mkstemp(pattern);
            if (descriptor < 0) throw std::runtime_error("Cannot create invalid model fixture");
            close(descriptor);
            path_ = pattern;
        }
        /** @brief Remove only this fixture's recoverable, empty test input. */
        ~InvalidModelFile() { std::remove(path_.c_str()); }
        InvalidModelFile(const InvalidModelFile &) = delete;
        InvalidModelFile &operator=(const InvalidModelFile &) = delete;
        /** @return Exact path used for error-provenance assertions. */
        const std::string &path() const noexcept { return path_; }
    private:
        std::string path_;
    };

    /** @brief Require a metadata failure and preserve its diagnostic for assertions. */
    std::string metadataError(const std::string &path)
    {
        try
        {
            (void)readPlanningModelMetadata(path);
            ADD_FAILURE() << "Invalid model unexpectedly supplied planning metadata: " << path;
        }
        catch (const std::runtime_error &error)
        {
            return error.what();
        }
        return {};
    }

    TEST(Test__OrchestrationRunnerModelValidation, FailsWhenModelFileDoesNotExist)
    {
        const std::string path = "/nonexistent/path/to/model.gguf";
        const auto error = metadataError(path);
        EXPECT_NE(error.find(path), std::string::npos) << error;
    }

    TEST(Test__OrchestrationRunnerModelValidation, FailsWhenModelFileIsInvalidGGUF)
    {
        const InvalidModelFile input;
        ASSERT_TRUE(std::filesystem::exists(input.path()));
        const auto error = metadataError(input.path());
        EXPECT_NE(error.find(input.path()), std::string::npos) << error;
    }

    TEST(Test__OrchestrationRunnerModelValidation, EmptyPathCannotInventModelDefaults)
    {
        EXPECT_THROW((void)readPlanningModelMetadata(""), std::invalid_argument);
    }

    TEST(Test__OrchestrationRunnerModelValidation, ErrorMessageIncludesFilePath)
    {
        const std::string path = "/some/specific/path/mymodel.gguf";
        const auto error = metadataError(path);
        EXPECT_NE(error.find(path), std::string::npos) << error;
    }

    /** Disposal cannot manufacture an absent prepared-context consumer. */
    TEST(Test__ModelContextRetirement, UnexportedModelNeverRestoresExpertPlacement)
    {
        for (const auto mode : {MoERebalanceRuntimeMode::Off,
                                MoERebalanceRuntimeMode::Observe,
                                MoERebalanceRuntimeMode::Dynamic})
        {
            EXPECT_EQ(modelContextOverlayDrainIntent(nullptr, mode),
                      MoEOverlayDeviceControllerDrainIntent::ReleaseResources);
        }
    }

    /** Both entry points to retained teardown preserve its physical obligation. */
    TEST(Test__ModelContextRetirement, RetainedDynamicModelRequiresRestoration)
    {
        auto authority = std::make_shared<ModelContextReuseAuthority>();
        for (int phase = 0; phase < 2; ++phase)
        {
            EXPECT_EQ(modelContextOverlayDrainIntent(authority, MoERebalanceRuntimeMode::Dynamic),
                      MoEOverlayDeviceControllerDrainIntent::RestorePreparedContext);
            for (const auto mode : {MoERebalanceRuntimeMode::Off, MoERebalanceRuntimeMode::Observe})
                EXPECT_EQ(modelContextOverlayDrainIntent(authority, mode),
                          MoEOverlayDeviceControllerDrainIntent::ReleaseResources);
            if (phase == 0)
                ASSERT_TRUE(authority->beginSealing());
        }
        // Reading the policy cannot certify or mutate the retained context.
        EXPECT_EQ(authority->state(), ModelContextReuseAuthority::State::Sealing);
        EXPECT_FALSE(authority->sealedDeviceMemoryRetention().has_value());
    }

    /** Invalid policy/state must fail, not masquerade as ordinary disposal. */
    TEST(Test__ModelContextRetirement, OverlayDrainRejectsUnownedContextAndInvalidPolicy)
    {
        auto authority = std::make_shared<ModelContextReuseAuthority>();
        ASSERT_TRUE(authority->beginSealing());
        ASSERT_TRUE(authority->publishReusable({}));
        EXPECT_THROW((void)modelContextOverlayDrainIntent(authority, MoERebalanceRuntimeMode::Dynamic),
                     std::logic_error);
        authority->invalidate("test retirement failure");
        EXPECT_THROW((void)modelContextOverlayDrainIntent(authority, MoERebalanceRuntimeMode::Off),
                     std::logic_error);
        EXPECT_THROW((void)modelContextOverlayDrainIntent(nullptr, static_cast<MoERebalanceRuntimeMode>(99)),
                     std::logic_error);
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

    TEST(OrchestrationRunnerModelValidation,
         AutomaticIntentCannotReachRunnerConstructionOrReadAModel)
    {
        OrchestrationConfig config;
        config.model_path = "/__factory_must_not_search__/missing.gguf";
        config.planning_mode = OrchestrationPlanningMode::Automatic;
        // No MPI/backend initialization exists in this Unit process. An
        // apply-only factory must reject before attempting either discovery
        // or model parsing; only frontend publication may authorize selection.
        auto factory = createOrchestrationRunnerFactory();
        ASSERT_NE(factory, nullptr);
        EXPECT_EQ(factory->createFromOrchestrationConfig(config), nullptr);
    }

} // namespace
