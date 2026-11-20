/**
 * @file MoEOverlayOriginalGPUCohortContexts.h
 * @brief Revocable borrows of the original admitted GPU expert-domain contexts.
 *
 * Domain setup owns each native communicator once. Setup calibration and
 * follower execution borrow that same lifetime; this roster neither constructs
 * a communicator nor retains one through reverse shared ownership.
 * A genuine GPU singleton retains its original identity and performs no wire
 * operation, including when AUTO selected a HOST identity placeholder. Target
 * lookup authenticates the complete original cohort in its collective order,
 * so another domain with the same degree cannot stand in for a priced target.
 */
#pragma once
#include "MoEExpertOwnerMap.h"
#include "MoEExpertShardCohort.h"
#include "collective/ILocalTPContext.h"
#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace llaminar2 {
/** @brief Cold immutable-by-publication map; only original context borrows are stored. */
class MoEOverlayOriginalGPUCohortContexts final {
public:
    /**
     * @brief Register one existing original native domain, without creating resources.
     * @param name Authored owner-map domain name.
     * @param original Same live context retained by the original domain setup owner.
     * @throws std::invalid_argument For empty/non-native/incomplete/duplicate identities.
     * @note Re-registering the same original context and name is idempotent.
     */
    void addOriginalDomain(const std::string& name,const ILocalTPContext& original)
    {
        const auto& devices=original.devices();
        if(name.empty() || original.degree()<1 || devices.size()!=size_t(original.degree()) ||
           (original.backend()!=CollectiveBackendType::NCCL && original.backend()!=CollectiveBackendType::RCCL &&
            !(original.degree()==1 && original.backend()==CollectiveBackendType::HOST)))
            throw std::invalid_argument("Original GPU expert domain requires its complete native context");
        const bool identity=original.degree()==1 && original.backend()==CollectiveBackendType::HOST;
        std::set<DeviceId> unique;
        for(const auto& device:devices)
            if(!device.isGPU() ||
               (!device.toLocalDeviceId().is_cuda() && !device.toLocalDeviceId().is_rocm()) ||
               device.device_type!=devices.front().device_type ||
               !unique.insert(device.toLocalDeviceId()).second ||
               (!identity && (device.toLocalDeviceId().is_cuda()
                    ? original.backend()!=CollectiveBackendType::NCCL : original.backend()!=CollectiveBackendType::RCCL)))
                throw std::invalid_argument("Original GPU expert domain has foreign or repeated native devices");
        const auto borrowed=original.borrowOriginalNativeContext();
        const auto [found,inserted]=domains_.try_emplace(name,borrowed);
        if(!inserted && !found->second.sameOriginal(borrowed))
            throw std::invalid_argument("Original GPU expert domain cannot replace its native owner");
    }

    /**
     * @brief Borrow the one existing context for this exact original source cohort.
     * @param owners Original admitted participant catalog paired with the source.
     * @param cohort Original source-native full member roster, not a selected subset.
     * @param world_rank Actual rank owning this complete GPU cohort.
     * @return Revocable same-owner native borrow, retaining no communicator.
     * @throws std::invalid_argument For missing members/domain/rank/order/backend.
     * @throws std::logic_error If the original domain has already retired.
     */
    [[nodiscard]] NativeLocalTPContextBorrow forOriginalCohort(
        const MoEExpertOwnerMap& owners,const MoEExpertShardCohort& cohort,int world_rank) const
    {
        const auto members=cohort.participants();
        if(members.empty() || world_rank<0)
            throw std::invalid_argument("Prepared GPU target requires its complete original cohort");
        const auto* first=owners.participantForId(members.front().participant_id);
        if(!first || first->domain_name.empty())
            throw std::invalid_argument("Prepared GPU target lost its original domain identity");
        const auto found=domains_.find(first->domain_name);
        if(found==domains_.end())
            throw std::invalid_argument("Prepared GPU target has no admitted original native domain");
        const auto use=found->second.acquire();
        const auto& native=use.context();
        if(native.degree()!=int(members.size()) || native.devices().size()!=members.size() ||
           (native.backend()!=CollectiveBackendType::NCCL && native.backend()!=CollectiveBackendType::RCCL &&
            !(members.size()==1 && native.backend()==CollectiveBackendType::HOST)))
            throw std::invalid_argument("Prepared GPU target differs from its original native cardinality");
        const bool identity=members.size()==1 && native.backend()==CollectiveBackendType::HOST;
        for(size_t member=0;member<members.size();++member) {
            const auto& selected=members[member];
            const auto* original=owners.participantForId(selected.participant_id);
            if(!selected.address.isGPU() || selected.world_rank!=world_rank || !original ||
               !original->world_rank_known || original->world_rank!=selected.world_rank ||
               original->domain_name!=first->domain_name || original->domain_participant_index!=int(member) ||
               original->device!=selected.address.toLocalDeviceId() || original->address!=selected.address ||
               native.devices()[member]!=selected.address ||
               (!identity && (selected.address.toLocalDeviceId().is_cuda()
                    ? native.backend()!=CollectiveBackendType::NCCL : native.backend()!=CollectiveBackendType::RCCL)))
                throw std::invalid_argument("Prepared GPU target has a missing, reordered or foreign original member");
        }
        const auto domain_members=std::count_if(owners.participants().begin(),owners.participants().end(),
            [&](const auto& member){return member.domain_name==first->domain_name;});
        if(size_t(domain_members)!=members.size())
            throw std::invalid_argument("Prepared GPU target omitted an original domain member");
        return found->second;
    }

    /** @return Number of original native domains, not the number of devices. */
    [[nodiscard]] size_t size() const noexcept {return domains_.size();}
private:
    std::map<std::string,NativeLocalTPContextBorrow> domains_;
};
} // namespace llaminar2
