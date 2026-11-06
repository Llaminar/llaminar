/**
 * @file FinalPhysicalMemoryCostEvidence.h
 * @brief Passive source-fixture CSV projection of the original admitted PMA.
 *
 * Every allocator is written only by its owning rank. Resource totals and owner
 * lines are separate, so neither repeated totals nor included preparation/SDK
 * subsets become additional physical bytes. This code performs no admission,
 * reservation, allocation, capacity subtraction, GPU query or synchronization.
 */
#pragma once
#include "planning/PhysicalMemoryAuthority.h"
#include <iomanip>
#include <map>
#include <ostream>
#include <stdexcept>
#include <string>
#include <tuple>

namespace llaminar2::test {
/** @brief Already-authenticated source fixture labels; they confer no physical authority. */
struct FinalPhysicalMemoryCostIdentity {
    std::string cell,source,profile;
};
/**
 * @brief Write one passive, rank-local snapshot from the exact live authority.
 * @param resources One resource-level CSV, with each capacity/total written once.
 * @param owners One owner-level CSV, without repeated resource totals.
 * @param authority Original live ModelContextReuseContract authority.
 * @param identity Existing source/case/profile labels verified by the source fixture.
 * @throws std::invalid_argument For incomplete identity or nonfinal/stale/duplicate attestation.
 * @throws std::runtime_error For a failed output stream.
 * @note Snapshot materialization is not a new promise of final full allocation.
 * Committed pools and opaque limits are not mislabeled as exact live backing.
 */
inline void writeFinalPhysicalMemoryCostCSV(std::ostream& resources,std::ostream& owners,
    const PhysicalMemoryAuthority& authority,const FinalPhysicalMemoryCostIdentity& identity)
{
    if(identity.cell.empty() || identity.source.empty() || identity.profile.empty() ||
        authority.phase()!=PhysicalMemoryAuthorityPhase::Admitted || !authority.admission())
        throw std::invalid_argument("Final PMA cost evidence requires original final source/profile authority");
    const auto& plan=authority.admission()->plan();
    const auto observed=authority.rankAttestation();
    using Key=std::tuple<int,int,int,int>;
    const auto key=[](PhysicalMemoryAllocatorIdentity id,PhysicalMemoryOwner owner) {
        return Key{id.world_rank,static_cast<int>(id.device.type),id.device.ordinal,static_cast<int>(owner)};
    };
    std::map<Key,const PhysicalMemoryOwnerAttestation*> joined;
    for(const auto& row:observed) {
        const auto* bom=plan.find(row.identity);
        if(row.identity.world_rank!=authority.worldRank() || row.owner==PhysicalMemoryOwner::Count ||
            row.phase!=PhysicalMemoryAuthorityPhase::Admitted || !bom ||
            row.planned_new_bytes!=bom->charge(row.owner).incrementalBytes() ||
            row.planned_resident_bytes!=bom->alreadyResidentBytes(row.owner) ||
            !joined.emplace(key(row.identity,row.owner),&row).second)
            throw std::invalid_argument("Final PMA cost evidence changed or duplicated an original owner line");
    }
    // Use the standard quoted formatter with CSV's quote-as-escape character:
    // embedded quotes double, while commas/newlines remain inside one field.
    const auto labels=[&](std::ostream& out) {
        out<<std::quoted(identity.cell,'"','"')<<','<<std::quoted(identity.source,'"','"')<<','
           <<std::quoted(identity.profile,'"','"')<<','<<authority.worldRank()<<',';
    };
    resources<<"cell,source,profile,world_rank,device,capacity_bytes,admission_available_bytes,admitted_total_bytes,admitted_new_bytes,admitted_resident_bytes\n";
    owners<<"cell,source,profile,world_rank,device,owner,admitted_total_bytes,admitted_new_bytes,admitted_resident_bytes,materialized_new_bytes,adopted_resident_bytes,committed_new_bytes,included_preparation_committed_bytes,included_preparation_materialized_bytes,native_declared_preclaimed_bytes,borrowed_opaque_runtime_committed_bytes,opaque_limit_committed_bytes\n";
    for(const auto& bom:plan.resources()) {
        const auto& resource=bom.resource();
        if(resource.world_rank!=authority.worldRank())continue;
        labels(resources);resources<<std::quoted(resource.device.toString(),'"','"')<<','
            <<resource.total_bytes<<','<<resource.admission_available_bytes<<','<<bom.totalBytes()<<','
            <<bom.incrementalBytes()<<','<<bom.alreadyResidentBytes()<<'\n';
        for(std::size_t index=0;index<PhysicalMemoryBOM::ownerCount();++index) {
            const auto owner=PhysicalMemoryBOM::ownerAt(index);
            const auto found=joined.find(key({resource.world_rank,resource.device},owner));
            if(found==joined.end())throw std::invalid_argument("Final PMA cost evidence omitted an original owner attestation");
            const auto& row=*found->second;
            labels(owners);owners<<std::quoted(resource.device.toString(),'"','"')<<','<<toString(owner)<<','
                <<bom.bytes(owner)<<','<<row.planned_new_bytes<<','<<row.planned_resident_bytes<<','
                <<row.totalMaterializedNewBytes()<<','<<row.adopted_resident_bytes<<','<<row.totalCommittedNewBytes()<<','
                <<row.preparation_committed_bytes<<','<<row.preparation_materialized_bytes<<','
                <<row.native_declared_preclaimed_bytes<<','<<row.borrowed_opaque_runtime_committed_bytes<<','
                <<row.opaque_runtime_committed_bytes<<'\n';
        }
    }
    if(!resources || !owners)throw std::runtime_error("Cannot write original final PMA cost evidence");
}
/**
 * @brief Require exact live backing for one complete final-plan owner category.
 * @param authority Original final PMA, not a reconstructed source-only budget.
 * @param allocator Exact allocator containing both retained and runner-owned roots.
 * @param owner Original typed category whose complete physical backing is required.
 * @throws std::logic_error If a final plan/attestation is missing, stale or incomplete.
 *
 * A retained source observation describes only its own leases. Other disjoint
 * owners, such as the runner's terminal destination, share the same category;
 * their exact aggregate is authenticated here by the sole materialization ledger.
 */
inline void requireCompleteFinalPhysicalMemoryOwner(const PhysicalMemoryAuthority& authority,
    PhysicalMemoryAllocatorIdentity allocator,PhysicalMemoryOwner owner)
{
    if(authority.phase()!=PhysicalMemoryAuthorityPhase::Admitted || !authority.admission() ||
        allocator.world_rank!=authority.worldRank() || owner==PhysicalMemoryOwner::Count)
        throw std::logic_error("Complete final PMA owner requires its original admitted rank/category");
    const auto* resource=authority.admission()->plan().find(allocator);
    if(!resource)throw std::logic_error("Complete final PMA owner omitted its allocator");
    const auto rows=authority.rankAttestation();
    const PhysicalMemoryOwnerAttestation* found=nullptr;
    for(const auto& row:rows)if(row.identity==allocator && row.owner==owner) {
        if(found)throw std::logic_error("Complete final PMA owner duplicated an attestation");
        found=&row;
    }
    if(!found || found->planned_new_bytes!=resource->charge(owner).incrementalBytes() ||
        found->planned_resident_bytes!=resource->alreadyResidentBytes(owner) || !found->complete())
        throw std::logic_error("Complete final PMA owner has missing or unmaterialized original physical bytes");
}

}
