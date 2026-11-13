/** @file NativeContextSourceProbePma.h
 * @brief Phase-safe read-only projection of the original source authority for diagnostics.
 *
 * rankAttestation is the canonical observation in Preparing and Admitted phases.
 * Ordinary reservation APIs intentionally require final admission and are never
 * used to observe pre-admission source commitments here.
 */
#pragma once
#include "planning/PhysicalMemoryAuthority.h"
#include <nlohmann/json.hpp>
namespace llaminar2::test {
/** @return One immutable ledger observation, including exact named per-allocator opaque commitments. */
inline nlohmann::ordered_json nativeSourcePmaProjection(const std::shared_ptr<PhysicalMemoryAuthority>& memory,DeviceId device){
    if(!memory || !device.is_gpu())throw std::invalid_argument("Source PMA observation requires original authority/GPU");
    std::size_t exact=0,opaque=0,host=0,gpu=0;
    for(const auto& row:memory->rankAttestation()){
        exact+=row.preparation_materialized_bytes;opaque+=row.opaque_runtime_committed_bytes;
        if(row.owner==PhysicalMemoryOwner::NativeExecutionContext){
            if(row.identity.device==DeviceId::cpu())host+=row.opaque_runtime_committed_bytes;
            if(row.identity.device==device)gpu+=row.opaque_runtime_committed_bytes;
        }
    }
    return {{"phase",memory->phase()==PhysicalMemoryAuthorityPhase::Preparing?"Preparing":"Admitted"},
        {"exact_preparation_bytes",exact},{"opaque_committed_bytes",opaque},{"host_opaque_bytes",host},{"device_opaque_bytes",gpu}};
}
}
