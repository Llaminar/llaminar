/**
 * @file Test__CPUShorthandPlacement.cpp
 * @brief Device-free rank-versus-NUMA proofs for the production CPU CLI shorthand.
 *
 * Hostfile slot order and visible NUMA IDs can change independently. These
 * tests exercise the same typed specialization used before runtime allocator
 * creation, without MPI initialization or any hardware discovery.
 */
#include "app/RuntimeInitPhase.h"
#include "utils/NUMATopology.h"
#include <gtest/gtest.h>

using namespace llaminar2;

TEST(CPUShorthandPlacement, AffinityNotMPIOrdinalOwnsTheCPU)
{
    for (int rank : {0, 1, 4, 7})
        for (int numa : {0, 1, 3, 9})
        {
            OrchestrationConfig config;
            config.cpu_global_tp_all_local = true;
            config.device_for_this_rank = GlobalDeviceAddress::cpu(0);
            const NUMAInfo observed{.local_numa_node = numa, .total_numa_nodes = 2,
                                   .detection_succeeded = true, .detection_method = "fixture"};
            for (int repeat = 0; repeat < 3; ++repeat)
            {
                RuntimeInitPhase::resolveCPUShorthand(config, rank, 8, observed);
                ASSERT_EQ(config.device_map.size(), 1u);
                EXPECT_EQ(config.device_map[0].first, rank);
                EXPECT_EQ(config.device_map[0].second.numa_node, numa);
                EXPECT_FALSE(config.device_for_this_rank);
                EXPECT_EQ(config.tp_degree, 8);
                EXPECT_EQ(config.tp_scope, TPScope::GLOBAL);
                EXPECT_EQ(RuntimeInitPhase::resolveCPUBackendNUMANode(config, rank, 8, observed), numa);
            }
        }
}

TEST(CPUShorthandPlacement, UnknownAffinityFailsWithoutPartiallyChangingIntent)
{
    OrchestrationConfig config;
    config.cpu_global_tp_all_local = true;
    config.device_for_this_rank = GlobalDeviceAddress::cpu(0);
    EXPECT_THROW(RuntimeInitPhase::resolveCPUShorthand(config, 1, 2, {}), std::runtime_error);
    EXPECT_TRUE(config.device_for_this_rank);
    EXPECT_TRUE(config.device_map.empty());
    EXPECT_EQ(config.device_mode, DeviceAssignmentMode::AUTO);
}

TEST(CPUShorthandPlacement, OtherSelectionModesAreUntouched)
{
    OrchestrationConfig config;
    config.device_for_this_rank = GlobalDeviceAddress::cuda(3);
    RuntimeInitPhase::resolveCPUShorthand(config, 0, 1, {});
    ASSERT_TRUE(config.device_for_this_rank);
    EXPECT_TRUE(config.device_for_this_rank->isGPU());
    EXPECT_TRUE(config.device_map.empty());
}
