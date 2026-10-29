/**
 * @file CPUActivationArenaFixture.h
 * @brief Explicit model-free admission for tests constructing CPU arena roots.
 *
 * Production never invents this authority: each fixture admits its own bounded
 * allocator budget, then the production arena claims actual roots against it.
 */
#pragma once
#include "memory/BufferArena.h"
#include "planning/PhysicalMemoryAuthority.h"

namespace llaminar2::test
{
/**
 * @brief Build an explicit fixture admission using the production typed BOM.
 * @param factory Optional NUMA-aware tensor construction factory.
 * @param bytes Fixture-owned CPU activation capacity, not materialized storage.
 * @return Arena configuration retaining the fixture's allocation authority.
 */
inline ArenaConfig cpuArenaFixtureConfig(TensorFactory *factory = nullptr,
                                        size_t bytes = 64u * 1024u * 1024u)
{
    PhysicalMemoryPlanBuilder plan;
    plan.add(PhysicalMemoryResource{.world_rank = 0, .device = DeviceId::cpu(),
        .total_bytes = bytes, .admission_available_bytes = bytes},
        PhysicalMemoryOwner::ActivationArena, bytes);
    return {.factory = factory, .physical_memory_authority =
        std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0)};
}
}
