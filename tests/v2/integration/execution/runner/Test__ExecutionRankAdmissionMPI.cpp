/**
 * @file Test__ExecutionRankAdmissionMPI.cpp
 * @brief Real selected-communicator ownership and runner admission regressions.
 *
 * Reversed and nested subsets reuse the observed hardware while inactive peers
 * do not enter runner initialization. An active nonzero discovery rank reaches
 * the production root-metadata error independently: any hidden WORLD operation
 * mismatches the inactive peer's terminal collective and fails this proof.
 * No weights or accelerators are opened; CTest supplies the usual 30-second
 * protocol bound through the existing initialization-lifecycle preflight group.
 */
#include "utils/MPIContext.h"
#include "planning/ExecutionRankMembership.h"
#include "execution/runner/OrchestrationRunner.h"
#include "execution/runner/IOrchestrationRunnerFactory.h"
#include "execution/mpi_orchestration/ExecutionPlanBuilder.h"
#include "collective/GlobalTPContext.h"
#include "../../../utils/TestTensorFactory.h"
#include <gtest/gtest.h>

using namespace llaminar2;
namespace
{
    /** @return The selected context, without constructing an inactive runner. */
    std::shared_ptr<MPIContext> activeContext(
        const std::variant<InactiveMPIRank, std::shared_ptr<MPIContext>> &result)
    {
        const auto *active = std::get_if<std::shared_ptr<MPIContext>>(&result);
        return active ? *active : nullptr;
    }
}

TEST(ExecutionRankAdmissionMPI, ReversedAndNestedMembershipRetainsObservedOwnershipTwentyTimes)
{
    auto discovery = MPIContextFactory::global();
    ASSERT_EQ(discovery->world_size(), 2);
    const auto observed = discovery->clusterInventory();
    for (int attempt = 0; attempt < 20; ++attempt)
    {
        SCOPED_TRACE(attempt);
        auto result = MPIContextFactory::selectRanks(discovery, {1, 0});
        auto active = activeContext(result);
        ASSERT_TRUE(active);
        EXPECT_EQ(active->rank(), 1 - discovery->rank());
        EXPECT_EQ(active->world_size(), 2);
        EXPECT_EQ(active->executionMembership()->discoveryRanks(), (std::vector<int>{1, 0}));
        const auto projection = active->clusterInventory();
        EXPECT_EQ(projection->ranks[active->rank()].cpu.numa_node,
                  observed->ranks[discovery->rank()].cpu.numa_node);
        EXPECT_EQ(projection.get(), &active->topology()->clusterInventory());
        auto nested = MPIContextFactory::selectRanks(active, {0});
        if (discovery->rank() == 1)
        {
            auto only = activeContext(nested);
            ASSERT_TRUE(only);
            EXPECT_EQ(only->rank(), 0);
            EXPECT_EQ(only->world_size(), 1);
            // The inactive peer enters the parent's next collective now.
            // Repeated observation must not rediscover against that parent.
            for (int read = 0; read < 20; ++read)
                EXPECT_EQ(only->clusterInventory()->ranks[0].cpu.numa_node, observed->ranks[1].cpu.numa_node);
        }
        else EXPECT_TRUE(std::holds_alternative<InactiveMPIRank>(nested));
        discovery->barrier();
    }
}

TEST(ExecutionRankAdmissionMPI, NonzeroDiscoveryRankInitializesWithoutInactivePeer)
{
    auto discovery = MPIContextFactory::global();
    ASSERT_EQ(discovery->world_size(), 2);
    auto admission = MPIContextFactory::selectRanks(discovery, {1});
    if (auto active = activeContext(admission))
    {
        OrchestrationConfig config;
        config.model_path = "/__llaminar_missing_execution_admission_model__/model.gguf";
        config.device_for_this_rank = GlobalDeviceAddress::cpu(active->clusterInventory()->ranks[0].cpu.numa_node);
        // The frontend's factory must preserve admission too; testing only the
        // explicit runner constructor cannot catch a hidden WORLD factory query.
        auto factory = createOrchestrationRunnerFactory(active);
        auto runner = factory->createFromOrchestrationConfig(config);
        ASSERT_TRUE(runner);
        EXPECT_FALSE(runner->initializeForDryRun());
        EXPECT_NE(runner->lastError().find("planning metadata"), std::string::npos) << runner->lastError();
        runner->shutdown();
    }
    else
    {
        EXPECT_EQ(std::get<InactiveMPIRank>(admission).discovery_rank, 0);
    }
    // Inactive peers perform no work while active admission fails. This is a
    // test-only rendezvous, not an idle-rank inference or shutdown controller.
    discovery->barrier();
}

TEST(ExecutionRankAdmissionMPI, AsymmetricInvalidOrDifferentSelectionFailsBeforeSplit)
{
    auto discovery = MPIContextFactory::global();
    ASSERT_EQ(discovery->world_size(), 2);
    for (const auto &bad : std::vector<std::vector<int>>{{}, {0, 0}, {2}, {1, 0}, {1}})
    {
        const auto requested = discovery->is_root() ? bad : std::vector<int>{0, 1};
        EXPECT_THROW(MPIContextFactory::selectRanks(discovery, requested), std::runtime_error);
    }
    auto malformed = std::make_shared<MPIContext>(discovery->rank(), discovery->is_root() ? 1 : 2,
                                                discovery->communicator());
    EXPECT_THROW(MPIContextFactory::selectRanks(malformed, {0}), std::runtime_error);
    EXPECT_NO_THROW(MPIContextFactory::selectRanks(discovery, {0, 1}));
}

TEST(ExecutionRankAdmissionMPI, GlobalTPPreservesSelectedParentRanksAndPhysicalEndpoint)
{
    auto discovery = MPIContextFactory::global();
    auto admitted = MPIContextFactory::selectRanks(discovery, {1, 0});
    auto active = activeContext(admitted);
    ASSERT_TRUE(active);
    const auto inventory = active->clusterInventory();
    const auto &observed = inventory->ranks.at(active->rank());
    const auto endpoint = GlobalDeviceAddress::cpu(observed.cpu.numa_node, observed.hostname);
    // Reorder the TP domain again. Parent rank IDs must refer to the admitted
    // context, while the bound physical endpoint remains this process's CPU.
    auto tp = GlobalTPContext::createWithSplit(active->communicator(), 8301, 0,
                                              1 - active->rank(), endpoint);
    ASSERT_TRUE(tp);
    EXPECT_EQ(tp->worldRanks(), (std::vector<int>{1, 0}));
    EXPECT_EQ(tp->myIndex(), discovery->rank());
    EXPECT_EQ(tp->localDevice(), endpoint);
    GlobalTPContext moved(std::move(*tp));
    EXPECT_EQ(moved.localDevice(), endpoint);
    auto value = llaminar2::test::TestTensorFactory::createFP32({1});
    llaminar2::test::TestTensorFactory::fillValue(value.get(), static_cast<float>(active->rank() + 1));
    EXPECT_TRUE(moved.allreduce(value.get()));
    EXPECT_FLOAT_EQ(value->typed_data()[0], 3.0f);
}

TEST(ExecutionRankAdmissionMPI, RunnerRejectsMalformedInjectedMembershipCollectively)
{
    auto discovery = MPIContextFactory::global();
    auto malformed = std::make_shared<MPIContext>(discovery->rank(), discovery->is_root() ? 1 : 2,
                                                 discovery->communicator());
    OrchestrationConfig config;
    config.model_path = "/__llaminar_missing_execution_admission_model__/model.gguf";
    OrchestrationRunner runner(malformed, config, std::make_unique<ExecutionPlanBuilder>());
    EXPECT_FALSE(runner.initializeForDryRun());
    runner.shutdown();
    discovery->barrier();
}

TEST(ExecutionRankAdmissionMPI, GlobalTPRejectsAsymmetricUnboundEndpointBeforeSplit)
{
    auto discovery = MPIContextFactory::global();
    const auto inventory = discovery->clusterInventory();
    const auto &observed = inventory->ranks.at(discovery->rank());
    const auto endpoint = discovery->is_root() ? std::nullopt :
        std::optional<GlobalDeviceAddress>(GlobalDeviceAddress::cpu(observed.cpu.numa_node, observed.hostname));
    EXPECT_THROW(GlobalTPContext::createWithSplit(discovery->communicator(), 8302, 0,
                                                 discovery->rank(), endpoint), std::invalid_argument);
    discovery->barrier();
}

TEST(ExecutionRankAdmissionMPI, ContextRetainsDiscoveryOwnerUntilItsDerivedTopologyRetires)
{
    auto original = MPIContextFactory::global();
    auto parent = std::make_shared<MPIContext>(original->rank(), original->world_size(),
                                              original->local_rank(), original->communicator());
    std::weak_ptr<MPIContext> weak = parent;
    auto result = MPIContextFactory::selectRanks(parent, {1, 0});
    parent.reset();
    EXPECT_FALSE(weak.expired());
    auto active = activeContext(result);
    ASSERT_TRUE(active);
    EXPECT_NE(active->topology(), nullptr);
    result = InactiveMPIRank{original->rank()};
    EXPECT_FALSE(weak.expired());
    active.reset();
    EXPECT_TRUE(weak.expired());
}
