/**
 * @file CPUSerialArenaTestAdmission.h
 * @brief Explicit production-layout admission for positive serial CPU arena fixtures.
 *
 * Each fixture authors the same normalized immutable root declaration used by
 * its arena. PMA alone owns live claims; no estimated reserve or test ledger is
 * substituted for the final ActivationArena and ExecutionWorkspace roots.
 */
#pragma once
#include "execution/compute_stages/stages/MoELocalExpertStage.h"
#include "planning/PhysicalMemoryAuthority.h"
#include <algorithm>
namespace llaminar2::test
{
/** @brief Author exact storage and admission for one positive CPU arena fixture.
 * @param config Explicit coherent-family and canonical/grouped storage policies.
 * @return Config retaining its immutable declaration and sole PMA issuer.
 */
inline MoELocalExpertSerialBufferArena::Config admitCPUSerialArenaForTest(
    MoELocalExpertSerialBufferArena::Config config)
{
    if (!config.device_id.is_cpu()) return config;
    auto rows = config.row_capacity_buckets;
    rows.push_back(config.row_capacity);
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    const bool grouped = config.cpu_grouped_scratch_storage ==
        MoELocalExpertSerialBufferArena::CPUGroupedScratchStoragePolicy::RetainSerialMaximum;
    config.cpu_storage_layout = std::make_shared<const CPUSerialExpertStorageLayout>(
        config.device_id, config.row_capacity, config.d_model, config.routing_top_k, rows,
        config.cpu_canonical_route_storage == MoELocalExpertSerialBufferArena::CPUCanonicalRouteStoragePolicy::RetainSerialMaximum,
        grouped ? config.expert_intermediate : 0, grouped ? config.num_experts : 0,
        config.cpu_canonical_route_gpu_consumer);
    const auto &layout = *config.cpu_storage_layout;
    const size_t capacity = CPUSerialExpertStorageLayout::add(layout.physicalBytes(), layout.workspaceBytes());
    PhysicalMemoryPlanBuilder plan;
    const PhysicalMemoryResource host{0, config.device_id, capacity, capacity};
    plan.add(host, PhysicalMemoryOwner::ActivationArena, layout.physicalBytes());
    plan.add(host, PhysicalMemoryOwner::ExecutionWorkspace, layout.workspaceBytes());
    config.physical_memory_authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    return config;
}
}
