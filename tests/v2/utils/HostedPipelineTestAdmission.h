/**
 * @file HostedPipelineTestAdmission.h
 * @brief Exact source-authored host boundary admission for weight-free pipeline tests.
 *
 * Mock stages retain their own model storage. This helper prices only the real
 * parent activation bank using the same declaration as production; its admitted
 * rows are the fixture's largest authored request, never a communication count.
 */
#pragma once
#include "execution/global/GlobalOrchestrator.h"
#include "execution/global/HostedPipelineActivationStorage.h"
#include "execution/global_pp/GlobalPPRankPlanBuilder.h"
namespace llaminar2::test {
/**
 * @brief Attach the original local CPU root for an explicitly bounded fixture.
 * @param config Actual mock/original MPI parent and complete stage topology.
 * @param maximum_rows Largest request submitted by this source fixture.
 * @return Same move-only configuration with its sole parent bank admission.
 */
inline GlobalOrchestrator::Config admitHostedPipelineFixture(
    GlobalOrchestrator::Config config,int maximum_rows) {
    config.hosted_activation_rows=maximum_rows;
    const auto plan=GlobalPPRankPlanBuilder::build(config.topology,config.rank);
    if(HostedPipelineActivationStorage::requiresBank(config.topology, plan)) {
        const auto root=HostedPipelineActivationStorage::forRows(maximum_rows,config.d_model);
        PhysicalMemoryPlanBuilder admission;
        const auto bytes=root.physicalBytes();
        admission.add({config.rank,DeviceId::cpu(),bytes,bytes},
            PhysicalMemoryOwner::ActivationTransportStaging,bytes);
        config.physical_memory=std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(admission.build()),config.rank);
    }
    return config;
}
}
