/** @file PreparedNativeLookupFixture.h
 * @brief Original public source/PMA lifecycle for explicit native lookup components.
 *
 * Startup observation precedes selected worker birth. Payload planning merges
 * with that original source's frozen mandatory declaration; no independent
 * authority or post-birth capacity observation can adopt native backing.
 */
#pragma once
#include "backends/BackendManager.h"
#include "backends/NativeContextPreparation.h"
#include "transfer/TransferEngine.h"
#include <algorithm>
#include <exception>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>
#include <unistd.h>
namespace llaminar2::test {
struct PreparedNativeLookupFixtureReplayAccess;
/** @brief Lexical original source session; all payload/native owners must die before retirement. */
class PreparedNativeLookupFixture final {
public:
    /** @brief Prepare only the original rank's selected native population.
     * @param selected Exact GPU participants owned by this original MPI rank.
     * @param original_rank Actual original MPI/discovery rank, not a NUMA or GPU ordinal.
     * Existing single-rank owners retain their explicit rank-zero contract.
     * @return Actual setup-only allocator observations before the first selected worker creator.
     * @throws std::exception For invalid rank/participants, prior worker birth or incomplete native observations.
     */
    static std::unique_ptr<PreparedNativeLookupFixture> prepare(
        std::span<const DeviceId> selected, const int original_rank = 0){
        if(original_rank<0)throw std::invalid_argument("Lookup fixture requires its nonnegative original MPI rank");
        if(selected.empty())throw std::invalid_argument("Lookup fixture requires an explicit selected native population");
        std::vector<DeviceId> devices;
        for(const auto device:selected){
            if(!device.is_gpu() || std::find(devices.begin(),devices.end(),device)!=devices.end())
                throw std::invalid_argument("Lookup fixture requires unique explicit native participants");
            devices.push_back(device);
            bool existing=false;
            try{(void)GPUDeviceContextPool::instance().retainExistingContextProvenance(device);existing=true;}catch(const std::logic_error&){}
            if(existing)throw std::logic_error("Lookup fixture cannot adopt an already-created native worker");
        }
        const auto pages=::sysconf(_SC_PHYS_PAGES),available=::sysconf(_SC_AVPHYS_PAGES),page=::sysconf(_SC_PAGESIZE);
        if(pages<=0 || available<=0 || page<=0)throw std::runtime_error("Lookup fixture original CPU observation is unavailable");
        std::vector<PhysicalMemoryResource> observations{{original_rank,DeviceId::cpu(),std::size_t(pages)*std::size_t(page),std::size_t(available)*std::size_t(page)}};
        for(const auto device:devices){
            auto* backend=getBackendFor(device);
            if(!backend || backend->deviceCount()<=device.ordinal)throw std::runtime_error("Lookup fixture selected native participant is unavailable");
            const auto total=backend->deviceMemoryTotal(device.ordinal),free=backend->deviceMemoryFree(device.ordinal);
            if(!total || !free || free>total)throw std::runtime_error("Lookup fixture original GPU observation is incomplete");
            observations.push_back({original_rank,device,total,free});
        }
        auto result=std::unique_ptr<PreparedNativeLookupFixture>(new PreparedNativeLookupFixture(std::move(devices)));
        try{
            result->bindSource(NativeContextPreparation::create(observations,original_rank,result->devices_));
            return result;
        }catch(...){result->retire();throw;}
    }
    /** @brief Retire the actual original native population after its last payload owner. */
    ~PreparedNativeLookupFixture()noexcept{try{retire();}catch(...){std::terminate();}}
    /** @return The sole original source used by public factory/pool admission. */
    const std::shared_ptr<NativeContextPreparation>& source()const noexcept{return source_;}
    /** @return The same Preparing/admitted authority; never a fixture-created second ledger. */
    const std::shared_ptr<PhysicalMemoryAuthority>& memory()const{return source_->memory();}
    /** @return Original immutable observation, never another driver-free sample. */
    const PhysicalMemoryResource& resource(DeviceId device)const{return source_->originalResource(device);}
    /** @return Exact prepared worker/stream/provenance, strongly retained until payload teardown. */
    const PreparedGPUContextUsePermit& permit(DeviceId device)const{
        for(const auto& value:permits_)if(value.device()==device)return value;
        throw std::logic_error("Lookup fixture omitted or retired this original worker");
    }
    /** @brief Merge typed payload contributions before sealing the original authority exactly once. */
    void admit(const PhysicalMemoryPlan& payload){
        if(!source_ || retired_ || admitted_)throw std::logic_error("Lookup fixture admission is original and one-shot");
        PhysicalMemoryPlanBuilder plan(source_->mandatoryBOM());for(const auto& row:payload.resources())plan.add(row);
        source_->admitFinal(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()));admitted_=true;
    }
    /** @brief Require every case payload gone while exact original worker preparation roots remain.
     * The canonical source attestation distinguishes those roots from additional
     * allocations even when both use CPU0/LocalCollective. No owner is exempted.
     */
    void requirePayloadRetired()const{
        if(!admitted_ || retired_)throw std::logic_error("Lookup payload retirement requires an active admitted original source");
        for(const auto& row:memory()->rankAttestation()){
            const auto actual=row.materialized_new_bytes;
            if(actual!=row.preparation_materialized_bytes)
                throw std::logic_error("Lookup payload remains live at "+row.identity.device.toString()+" owner "+std::to_string(static_cast<unsigned>(row.owner))+
                    ": exact="+std::to_string(actual)+" original_preparation="+std::to_string(row.preparation_materialized_bytes));
        }
    }
    /** @brief Require zero exact roots after original worker join/unmap and validated context retirement.
     * ProcessRuntime opaque commitments deliberately remain and are not exact roots.
     */
    void requireExactRetired()const{
        if(!retired_)throw std::logic_error("Lookup exact retirement precedes original context retirement");
        for(const auto& row:memory()->rankAttestation())
            if(row.preparation_materialized_bytes || row.materialized_new_bytes)
                throw std::logic_error("Lookup exact root survived original source retirement at "+row.identity.device.toString()+" owner "+std::to_string(static_cast<unsigned>(row.owner)));
    }
    /** @brief Release original borrows then use the canonical validated native-only retirement authority.
     * All payload handles/allocations must already be gone; the public zero-allocation gate proves this.
     * ProcessRuntime opaque commitments remain with the original source/process owner after reset. */
    void retire(){
        if(retired_ || !source_)return;
        permits_.clear();
        if(!admitted_)source_->retireUnadmitted();
        else{
            std::vector<ExclusiveModelRetirementTicket> tickets;
            for(const auto device:devices_)tickets.push_back(TransferEngine::instance().beginExclusiveNativeContextRetirement(device));
            retirementReceipts_=TransferEngine::instance().completeExclusiveModelRetirements(std::move(tickets));
        }
        retired_=true;requireExactRetired();
    }
    /** @return Original validated reset receipts after retirement; no additional probe or reset is performed. */
    std::span<const DeviceMemoryReclamationReceipt> retirementReceipts()const{
        if(!retired_)throw std::logic_error("Native lookup reset receipts precede original retirement");
        return retirementReceipts_;
    }
private:
    friend struct PreparedNativeLookupFixtureReplayAccess;
    /** @brief Consume the exact original source through the same acquire/freeze lifecycle. */
    void bindSource(std::shared_ptr<NativeContextPreparation> source){
        if(source_ || !source)throw std::logic_error("Lookup fixture source binding is one-shot");source_=std::move(source);
        for(const auto device:devices_)permits_.push_back(source_->acquire(device));
        source_->freezeForPlanning();
    }
    /** @brief Store only declared immutable participant identity before source creation. */
    explicit PreparedNativeLookupFixture(std::vector<DeviceId> devices):devices_(std::move(devices)){}
    std::vector<DeviceId> devices_;
    std::shared_ptr<NativeContextPreparation> source_;
    std::vector<PreparedGPUContextUsePermit> permits_;
    std::vector<DeviceMemoryReclamationReceipt> retirementReceipts_; ///< Actual original TransferEngine results, never synthesized observations.
    bool admitted_=false,retired_=false;
};
}
