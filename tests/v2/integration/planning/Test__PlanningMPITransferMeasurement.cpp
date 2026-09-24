/**
 * @file Test__PlanningMPITransferMeasurement.cpp
 * @brief Real multi-rank startup transfer measurement and failure-atomic memory proofs.
 *
 * Two- and three-rank launches use the production MPI interface and PMA claims.
 * Receipts must name exact discovery ranks and physical nodes, including
 * reversed initiators and temporarily uninvolved ranks. Timings are functional
 * evidence only; no performance threshold is installed in the preflight gate.
 */
#include "planning/PlanningMPITransferMeasurement.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "planning/PlanningPublication.h"
#include "utils/MPIContext.h"
#include "interfaces/IMPITopology.h"
#include "backends/BackendManager.h"
#include <gtest/gtest.h>
#include <cmath>
#include <iostream>

using namespace llaminar2;

namespace
{
    /**
     * @brief Register the host allocator for every planning integration filter.
     *
     * GPU observation tests still execute real CPU samples on other ranks.
     * Discovery supplies inventory, not an allocator. Install the test's
     * aggregate CPU backend after discovery and before any sample preparation;
     * individual tests must not depend on another test having run first.
     */
    class PlanningHostBackendEnvironment final : public ::testing::Environment
    {
    public:
        /** @brief Preserve an existing rank backend or install this fixture's host allocator. */
        void SetUp() override
        {
            (void)MPIContextFactory::global();
            if (!hasCPUBackend()) initCPUBackend(-1);
        }
    };

    [[maybe_unused]] ::testing::Environment *const planning_host_backend_environment =
        ::testing::AddGlobalTestEnvironment(new PlanningHostBackendEnvironment());

    /** @return Unequal request/reply sizes and both directions, without rank-zero authority assumptions. */
    std::vector<PlanningMPITransferRequest> samplePlan(int count)
    {
        std::vector<PlanningMPITransferRequest> plan{{0, 1, 17, 257}, {1, 0, 1024 * 1024, 65536}};
        if (count > 2)
        {
            plan.push_back({1, 2, 4096, 13});
            plan.push_back({2, 0, 32768, 1048576});
        }
        return plan;
    }

    /** @brief Admit this fixture's exact payload against the canonical observed CPU allocator. */
    std::shared_ptr<PhysicalMemoryAuthority> memoryFor(const std::shared_ptr<IMPIContext> &context,
        std::span<const PlanningMPITransferRequest> plan)
    {
        const auto inventory = context->clusterInventory();
        const auto &rank = inventory->ranks.at(context->rank());
        PhysicalMemoryPlanBuilder builder;
        builder.add({.world_rank = context->rank(), .device = DeviceId::cpu(),
            .total_bytes = rank.cpu.memory_bytes, .admission_available_bytes = rank.cpu.free_memory_bytes},
            PhysicalMemoryOwner::ActivationTransportStaging,
            PlanningMPITransferMeasurement::workspaceBytes(plan, context->rank(), context->world_size()));
        return std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), context->rank());
    }

    /** @brief Detect leaked claims after success or asymmetric pre-traffic failure. */
    void expectRetired(const std::shared_ptr<PhysicalMemoryAuthority> &memory)
    {
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging,
            PhysicalMemoryMaterializationKind::NewAllocation), 0u);
    }
}

TEST(PlanningMPITransferMeasurementIntegration, TwentyBatchesPreserveExactTopologyAndRetireAllPayloads)
{
    const auto context = MPIContextFactory::global();
    ASSERT_GE(context->world_size(), 2);
    const auto inventory = context->clusterInventory();
    const auto plan = samplePlan(context->world_size());
    const auto memory = memoryFor(context, plan);
    for (int iteration = 0; iteration != 20; ++iteration)
    {
        int descriptions = 0;
        const auto observations = PlanningMPITransferMeasurement::measure(context, memory, DeviceId::cpu(), [&] {
            ++descriptions;
            return plan;
        });
        EXPECT_EQ(descriptions, context->is_root() ? 1 : 0);
        expectRetired(memory);
        ASSERT_EQ(observations.size(), context->is_root() ? plan.size() : 0);
        for (size_t index = 0; index != observations.size(); ++index)
        {
            const auto &observation = observations[index];
            EXPECT_EQ(observation.request(), plan[index]);
            EXPECT_EQ(observation.topology(), inventory->connectionBetweenRanks(plan[index].initiator, plan[index].responder));
            EXPECT_TRUE(std::isfinite(observation.secondsPerExchange()));
            EXPECT_GT(observation.secondsPerExchange(), 0);
            if (iteration == 0)
                std::cout << "MPI_SAMPLE rank=" << plan[index].initiator << "->" << plan[index].responder
                    << " node=" << observation.topology().sourceNode() << "->" << observation.topology().destinationNode()
                    << " request_bytes=" << plan[index].request_bytes << " reply_bytes=" << plan[index].reply_bytes
                    << " seconds_per_exchange=" << observation.secondsPerExchange() << '\n';
        }
    }
}

TEST(PlanningMPITransferMeasurementIntegration, AsymmetricPreparationFailuresDoNotStartTrafficOrLeakClaims)
{
    const auto context = MPIContextFactory::global();
    const auto plan = samplePlan(context->world_size());
    const auto memory = memoryFor(context, plan);
    for (int failed_rank : {0, 1})
        for (int defect = 0; defect != 3; ++defect)
        {
            const auto owner = context->rank() == failed_rank && defect == 0 ? nullptr : memory;
            const auto device = context->rank() == failed_rank && defect == 1 ? DeviceId::cuda(0) : DeviceId::cpu();
            // Holding the admission line establishes deterministic exhaustion
            // without exhausting host RAM or creating a second byte ledger.
            PhysicalMemoryAllocationLease occupied;
            if (context->rank() == failed_rank && defect == 2)
                occupied = memory->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging,
                    PlanningMPITransferMeasurement::workspaceBytes(plan, context->rank(), context->world_size()));
            EXPECT_THROW(PlanningMPITransferMeasurement::measure(context, owner, device, [&] { return plan; }), std::runtime_error);
            occupied = {};
            expectRetired(memory);
            EXPECT_NO_THROW(PlanningMPITransferMeasurement::measure(context, memory, DeviceId::cpu(), [&] { return plan; }));
            expectRetired(memory);
        }
}

TEST(PlanningMPITransferMeasurementIntegration, InvalidRootPlanAndEmptyBatchAreCollective)
{
    const auto context = MPIContextFactory::global();
    const auto plan = samplePlan(context->world_size());
    const auto memory = memoryFor(context, plan);
    for (int defect = 0; defect != 5; ++defect)
    {
        EXPECT_THROW(PlanningMPITransferMeasurement::measure(context, memory, DeviceId::cpu(), [&] {
            auto malformed = plan;
            if (defect == 0) throw std::runtime_error("Root plan failed");
            if (defect == 1) malformed[0].responder = context->world_size();
            if (defect == 2) malformed[0].reply_bytes = 0;
            if (defect == 3) malformed[0].initiator = 1;
            if (defect == 4) malformed[0].request_bytes = size_t(std::numeric_limits<int>::max()) + 1;
            return malformed;
        }), std::runtime_error);
        expectRetired(memory);
    }
    const auto empty = PlanningMPITransferMeasurement::measure(context, nullptr, DeviceId::invalid(), [] {
        return std::vector<PlanningMPITransferRequest>{};
    });
    EXPECT_TRUE(empty.empty());
    EXPECT_NO_THROW(PlanningMPITransferMeasurement::measure(context, memory, DeviceId::cpu(), [&] { return plan; }));
    expectRetired(memory);
}

TEST(PlanningMPITransferMeasurementIntegration, TopologyConsumerUsesPublishedMachineMembership)
{
    const auto context = MPIContextFactory::global();
    const auto inventory = context->clusterInventory();
    const auto *topology = context->topology();
    ASSERT_NE(topology, nullptr);
    for (int a = 0; a != context->world_size(); ++a)
        for (int b = 0; b != context->world_size(); ++b)
            EXPECT_EQ(topology->same_node(a, b),
                inventory->connectionBetweenRanks(a, b).locality() != RankConnectionLocality::CrossNode);
    EXPECT_THROW(topology->same_node(-1, -1), std::invalid_argument);
    EXPECT_THROW(topology->same_node(context->world_size(), context->world_size()), std::invalid_argument);
}
