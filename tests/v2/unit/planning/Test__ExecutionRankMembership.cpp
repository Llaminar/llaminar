/**
 * @file Test__ExecutionRankMembership.cpp
 * @brief Device-free subset/reordering proofs for execution admission.
 *
 * Sparse physical CPU IDs, nonzero GPU ordinals and omitted physical nodes
 * expose accidental use of rank indices as device identity. These tests use
 * only immutable observations; no MPI initialization or device allocation.
 */
#include "planning/ExecutionRankMembership.h"
#include "execution/mpi_orchestration/ExecutionPlanBuilder.h"
#include "execution/global_pp/GlobalPPTopology.h"
#include <gtest/gtest.h>

using namespace llaminar2;
namespace
{
    /** @return Two physical-node observations with deliberately sparse IDs. */
    ClusterInventory discovery()
    {
        ClusterInventory inventory;
        inventory.world_size = 4;
        for (int rank = 0; rank < 4; ++rank)
        {
            RankInventory record;
            record.rank = rank;
            record.node_id = rank / 2 + 8;
            record.local_rank = rank % 2;
            record.hostname = "observed-node-" + std::to_string(record.node_id);
            record.cpu.numa_node = rank % 2 ? 7 : 4;
            record.cpu_cores = 8;
            record.cpu_worker_threads = 8;
            record.cpu_memory_bytes = 4096;
            for (auto backend : {DeviceType::CUDA, DeviceType::ROCm})
            {
                DeviceInfo gpu;
                gpu.type = backend;
                gpu.local_device_id = 6;
                gpu.numa_node = 7;
                gpu.uuid = std::string(deviceTypeToString(backend)) + "-uuid";
                gpu.memory_bytes = 8192;
                gpu.free_memory_bytes = 6144;
                record.gpus.push_back(gpu);
            }
            inventory.ranks.push_back(record);
        }
        return inventory;
    }
}

TEST(ExecutionRankMembership, ReordersLogicalRanksWithoutChangingPhysicalEndpoints)
{
    const auto original = discovery();
    ExecutionRankMembership membership(original, {3, 2, 0});
    EXPECT_EQ(membership.discoveryRanks(), (std::vector<int>{3, 2, 0}));
    EXPECT_EQ(membership.selection(), ExecutionRankSelection({3, 2, 0}));
    EXPECT_EQ(membership.executionRank(3), 0);
    EXPECT_EQ(membership.executionRank(0), 2);
    EXPECT_FALSE(membership.executionRank(1));
    const auto &projected = membership.inventory();
    EXPECT_EQ(projected.world_size, 3);
    EXPECT_EQ(projected.node_count, 2);
    EXPECT_EQ(projected.ranks[0].local_rank, 0);
    EXPECT_EQ(projected.ranks[1].local_rank, 1);
    EXPECT_EQ(projected.ranks[2].local_rank, 0);
    EXPECT_EQ(projected.ranks[2].node_id, 1);
    for (int rank = 0; rank < 3; ++rank)
    {
        const auto &before = original.ranks[membership.discoveryRanks()[rank]];
        const auto &after = projected.ranks[rank];
        EXPECT_EQ(after.rank, rank);
        EXPECT_EQ(after.hostname, before.hostname);
        EXPECT_EQ(after.cpu.numa_node, before.cpu.numa_node);
        EXPECT_EQ(after.cpu_memory_bytes, before.cpu_memory_bytes);
        ASSERT_EQ(after.gpus.size(), 2u);
        for (int gpu = 0; gpu < 2; ++gpu)
        {
            EXPECT_EQ(after.gpus[gpu].local_device_id, 6);
            EXPECT_EQ(after.gpus[gpu].numa_node, 7);
            EXPECT_EQ(after.gpus[gpu].uuid, before.gpus[gpu].uuid);
            EXPECT_EQ(after.gpus[gpu].free_memory_bytes, 6144u);
        }
    }
    EXPECT_EQ(projected.total_gpus, 4) << "Repeated visibility must not add physical GPUs";
    for (int a = 0; a != 3; ++a)
        for (int b = 0; b != 3; ++b)
            EXPECT_EQ(projected.connectionBetweenRanks(a, b).locality(),
                original.connectionBetweenRanks(membership.discoveryRanks()[a], membership.discoveryRanks()[b]).locality());
    EXPECT_EQ(original.ranks[3].rank, 3);
    EXPECT_EQ(original.ranks[3].node_id, 9);
}

TEST(ExecutionRankMembership, SelectionOwnsValidatedOrderWithoutAnInventoryCopy)
{
    const ExecutionRankSelection ordered({7, 3, 0});
    EXPECT_EQ(ordered.size(), 3);
    EXPECT_EQ(ordered.executionRank(7), 0);
    EXPECT_EQ(ordered.executionRank(3), 1);
    EXPECT_EQ(ordered.executionRank(0), 2);
    EXPECT_FALSE(ordered.executionRank(5));
    EXPECT_THROW(ordered.executionRank(-1), std::out_of_range);
    EXPECT_EQ(ExecutionRankSelection::all(4).discoveryRanks(), (std::vector<int>{0, 1, 2, 3}));
    for (const auto &bad : std::vector<std::vector<int>>{{}, {-1}, {0, 0}, {1, 3, 1}})
        EXPECT_THROW((void)ExecutionRankSelection(bad), std::invalid_argument);
    EXPECT_THROW((void)ExecutionRankSelection::all(0), std::invalid_argument);
    EXPECT_THROW((void)ExecutionRankSelection::all(-1), std::invalid_argument);
}

TEST(ExecutionRankMembership, NestedSelectionPreservesHardwareAndRemovesEmptyNodes)
{
    ExecutionRankMembership outer(discovery(), {3, 2, 0});
    ExecutionRankMembership inner(outer.inventory(), {1});
    EXPECT_EQ(inner.inventory().node_count, 1);
    EXPECT_EQ(inner.inventory().nodes.size(), 1u);
    EXPECT_EQ(inner.inventory().ranks[0].rank, 0);
    EXPECT_EQ(inner.inventory().ranks[0].cpu.numa_node, 4);
    EXPECT_EQ(inner.inventory().ranks[0].hostname, "observed-node-9");
    EXPECT_EQ(inner.inventory().total_gpus, 2);
}

TEST(ExecutionRankMembership, RejectsInvalidMembershipBeforeProjectionEscapes)
{
    for (const auto &ranks : std::vector<std::vector<int>>{{}, {0, 0}, {-1}, {4}, {0, 1, 2, 3, 0}})
        EXPECT_THROW(ExecutionRankMembership(discovery(), ranks), std::invalid_argument);
    auto malformed = discovery();
    malformed.ranks[3].rank = 0;
    EXPECT_THROW(ExecutionRankMembership(malformed, {0}), std::invalid_argument);
    malformed = discovery();
    malformed.world_size = 5;
    EXPECT_THROW(ExecutionRankMembership(malformed, {0}), std::invalid_argument);
    ExecutionRankMembership good(discovery(), {1});
    EXPECT_THROW(good.executionRank(-1), std::out_of_range);
    EXPECT_THROW(good.executionRank(4), std::out_of_range);
}

TEST(ExecutionRankMembership, DomainCompilerKeepsReversedOwnerDeviceAndWorkShareTogether)
{
    const ExecutionRankMembership membership(discovery(), {3, 2});
    DomainDefinition domain;
    domain.name = "continuation";
    domain.scope = TPScope::NODE_LOCAL;
    domain.backend = CollectiveBackendType::UPI;
    domain.devices = {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()};
    domain.explicit_ranks = {1, 0};
    domain.weights = {0.75f, 0.25f};
    OrchestrationConfig config;
    config.domain_definitions = {domain};
    ExecutionPlanBuilder builder;
    const auto model = ModelConfig::qwen2_7b();
    const auto plans = builder.buildAllPlans(config, model, membership.inventory());
    for (int rank = 0; rank < 2; ++rank)
    {
        const auto &observed = membership.inventory().ranks[rank];
        EXPECT_EQ(plans[rank].primary_device.numa_node, observed.cpu.numa_node);
        EXPECT_EQ(plans[rank].primary_device.hostname, observed.hostname);
        ASSERT_EQ(plans[rank].my_domains.size(), 1u);
        EXPECT_EQ(plans[rank].my_domains[0].devices[rank].numa_node, observed.cpu.numa_node);
        EXPECT_EQ(plans[rank].weight_shard.work_fraction, rank == 0 ? 0.25f : 0.75f);
    }
    const auto topology = builder.buildGlobalPPTopology(config, model, membership.inventory());
    ASSERT_EQ(topology.stages.size(), 1u);
    EXPECT_EQ(topology.stages[0].participating_ranks, (std::vector<int>{0, 1}));
    ASSERT_EQ(topology.stages[0].per_rank_devices.size(), 2u);
    EXPECT_EQ(topology.stages[0].per_rank_devices[0].numa_node, 7);
    EXPECT_EQ(topology.stages[0].per_rank_devices[1].numa_node, 4);
    EXPECT_EQ(config.domain_definitions[0].devices[0].numa_node, NUMA_NODE_UNKNOWN);
}
