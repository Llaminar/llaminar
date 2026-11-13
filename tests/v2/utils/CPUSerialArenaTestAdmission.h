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
#include <span>
#include <vector>
#include <utility>
namespace llaminar2::test
{
/** @brief Declare the exact original CPU arena layout before any backing materializes.
 * @param config Explicit coherent-family and canonical/grouped storage policies.
 * @return Config retaining its exact immutable layout; no backing or PMA is created.
 */
inline MoELocalExpertSerialBufferArena::Config declareCPUSerialArenaForTest(
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
    return config;
}
/** @brief Admit concurrent original serial roots and a typed auxiliary program on one CPU authority.
 * @param declarations Explicit original arena policies; no backing exists yet.
 * @param host Original shared CPU allocator observation for every contribution.
 * @param additional Actual typed worker/program BOM, declared before either creator.
 * @return Configurations retaining their exact layouts and the same final PMA/control block.
 */
inline std::vector<MoELocalExpertSerialBufferArena::Config> admitCPUSerialArenasForTest(
    std::span<const MoELocalExpertSerialBufferArena::Config> declarations,
    PhysicalMemoryResource host,const PhysicalMemoryPlan& additional)
{
    if(declarations.empty() || !host.device.is_cpu())
        throw std::invalid_argument("Serial arena group requires original CPU declarations");
    PhysicalMemoryPlanBuilder plan(additional);
    std::vector<MoELocalExpertSerialBufferArena::Config> configurations;
    for(auto config:declarations){
        if(config.device_id!=host.device)throw std::invalid_argument("Serial arena group changed original CPU allocator");
        config=declareCPUSerialArenaForTest(std::move(config));
        plan.add(host,PhysicalMemoryOwner::ActivationArena,config.cpu_storage_layout->physicalBytes());
        plan.add(host,PhysicalMemoryOwner::ExecutionWorkspace,config.cpu_storage_layout->workspaceBytes());
        configurations.push_back(std::move(config));
    }
    auto memory=std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),host.world_rank);
    for(auto& config:configurations)config.physical_memory_authority=memory;
    return configurations;
}
/** @brief Retain the existing single-arena exact-capacity fixture contract.
 * @param config Original serial storage policy and geometry.
 * @return Exact layout and sole admitted issuer, before storage materialization.
 */
inline MoELocalExpertSerialBufferArena::Config admitCPUSerialArenaForTest(
    MoELocalExpertSerialBufferArena::Config config)
{
    if(!config.device_id.is_cpu())return config;
    config=declareCPUSerialArenaForTest(std::move(config));
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
