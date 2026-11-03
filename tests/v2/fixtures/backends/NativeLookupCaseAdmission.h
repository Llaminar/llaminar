/**
 * @file NativeLookupCaseAdmission.h
 * @brief Original typed lookup-case composition for serialized native fixtures.
 *
 * Payload/root owners retire between cases and use their exact maximum.
 * Native ordering and CPU runtime backing can survive individual handles;
 * their original per-case obligations are additive until source retirement.
 * This helper copies existing BOM lines only and has no capacity estimator,
 * runtime allocation ledger, source creator or native-readiness authority.
 */
#pragma once
#include "planning/PhysicalMemoryAuthority.h"
#include <span>
#include <vector>
namespace llaminar2::test {
/**
 * @brief Compose original serialized case contributions before one final admission.
 * @param cases Complete payload declarations, without the shared preparation BOM.
 * @return Exact maximum payload/storage plus every original retained native/runtime obligation.
 * @throws std::invalid_argument For no cases or conflicting physical observations.
 * @throws std::overflow_error For unrepresentable original sums.
 */
inline PhysicalMemoryPlan serialNativeLookupCaseEnvelope(std::span<const PhysicalMemoryPlan> cases)
{
    if(cases.empty())throw std::invalid_argument("Native lookup fixture needs declared original cases");
    PhysicalMemoryPlanBuilder retained;
    std::vector<PhysicalMemoryPlan> payload;
    payload.reserve(cases.size());
    for(const auto& original:cases){
        PhysicalMemoryPlanBuilder local;
        for(const auto& row:original.resources())
            for(std::size_t index=0;index<PhysicalMemoryBOM::ownerCount();++index){
                const auto owner=PhysicalMemoryBOM::ownerAt(index);
                const auto& charge=row.charge(owner);
                if(!charge.planned_bytes)continue;
                auto& destination=(owner==PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent ||
                    owner==PhysicalMemoryOwner::RetainedHostServiceRuntime)?retained:local;
                destination.add(row.resource(),owner,charge.planned_bytes,charge.already_resident_bytes);
            }
        payload.push_back(local.build());
    }
    retained.addMutuallyExclusive(payload);
    return retained.build();
}
}
