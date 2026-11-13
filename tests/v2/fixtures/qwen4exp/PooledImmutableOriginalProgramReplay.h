/**
 * @file PooledImmutableOriginalProgramReplay.h
 * @brief Predeclare serialized original source recipes before CPU/GPU workers and final same-PMA admission.
 *
 * This device-free bridge uses the existing actual Preparing/source/pool path.
 * Only SDK/worker operations are doubled. Every IO instance has an immutable
 * case ordinal; ProcessRuntime roots survive reset and require process isolation.
 */
#pragma once
#include "PooledImmutableReadGraphFixture.h"
#include "fixtures/backends/NativeContextPreparationReplayAccess.h"
namespace llaminar2::test {
/** @brief One immutable recorded source recipe, never live device/model state. */
struct PooledOriginalRecipe {
    std::vector<std::uint32_t> capacities;
    bool reverse=false;
    std::uint32_t local=q::adaptive_lookup::None;
    bool nativeIds=false;
    DeviceType source=DeviceType::CUDA;
    std::uint32_t originalIds=4; ///< Exact original query words;16 supports one genuine sixteen-head PLE query.
    bool operator==(const PooledOriginalRecipe&)const=default;
};
/** @brief Recorded native worker body lives in the actual preclaimed compiled object/stack root. */
class PooledOriginalWorker final:public testing::MockWorkerGPUContext {
public:
    PooledOriginalWorker(int ordinal,std::shared_ptr<PreparedGPUWorkerStack>):MockWorkerGPUContext(ordinal){}
};
/** @brief One source and final authority for all serialized recipes of an actual test process. */
class PooledImmutableOriginalPrograms final {
public:
    /** @brief Freeze exact payload alternatives and additive original runtime/event obligations before materialization. */
    explicit PooledImmutableOriginalPrograms(std::vector<PooledOriginalRecipe> recipes):recipes_(std::move(recipes)),backends_(std::make_shared<OwnerChainBackendScope>()){
        if(recipes_.empty())throw std::invalid_argument("Original source replay needs its complete declared case set");
        std::vector<PhysicalMemoryPlan> payloads;PhysicalMemoryPlanBuilder retained;std::vector<DeviceId> devices;
        for(std::size_t ordinal=0;ordinal<recipes_.size();++ordinal){const auto& recipe=recipes_[ordinal];
            SourceGraphCase declaration(recipe.capacities,recipe.reverse,recipe.local,recipe.nativeIds,recipe.source,backends_,nullptr,{},SourceGraphRecipePhase::DeclareOnly,ordinal,recipe.originalIds);
            PhysicalMemoryPlanBuilder payload;
            for(const auto& row:declaration.declaredProgram.resources())for(unsigned i=0;i<static_cast<unsigned>(PhysicalMemoryOwner::Count);++i){
                const auto owner=static_cast<PhysicalMemoryOwner>(i);const auto charge=row.charge(owner);
                auto& out=owner==PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent || owner==PhysicalMemoryOwner::RetainedHostServiceRuntime?retained:payload;
                out.add(row.resource(),owner,charge.planned_bytes,charge.already_resident_bytes);
            }
            payloads.push_back(payload.build());
            const auto add=[&](DeviceId device){if(std::find(devices.begin(),devices.end(),device)==devices.end())devices.push_back(device);};
            add(declaration.physical.coordinator().device);for(const auto& owner:declaration.physical.owners())add(owner.allocator.device);
        }
        std::vector<PhysicalMemoryResource> observed{{0,DeviceId::cpu(),1ULL<<30,1ULL<<30}};
        for(const auto device:devices){observed.push_back({0,device,1ULL<<30,1ULL<<30});
            NativeContextPreparationReplayAccess::seed(device,[device](auto stack){return NativeContextPreparationReplayAccess::construct<PooledOriginalWorker>(std::move(stack),device.ordinal);});}
        source_=NativeContextPreparationReplayAccess::source<PooledOriginalWorker>(devices,observed);
        for(const auto device:devices)permits_.push_back(source_->acquire(device));source_->freezeForPlanning();
        PhysicalMemoryPlanBuilder full(source_->mandatoryBOM());full.addMutuallyExclusive(payloads);const auto runtime=retained.build();for(const auto& row:runtime.resources())full.add(row);
        source_->admitFinal(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(full.build()));
    }
    /** @brief Close original GPU use only after all case aliases drain; CPU ProcessRuntime obligations remain owned. */
    ~PooledImmutableOriginalPrograms(){try{retire();}catch(...){std::terminate();}}
    /** @return The unchanged real source factory body on this predeclared same-PMA case. */
    SourceGraphCase make(PooledOriginalRecipe recipe,std::shared_ptr<PhysicalMemoryAuthority>* observation=nullptr){
        if(retired_ || next_>=recipes_.size() || recipe!=recipes_[next_])throw std::logic_error("Original source recipe changed or was not predeclared");
        const auto ordinal=next_++;
        return SourceGraphCase(recipe.capacities,recipe.reverse,recipe.local,recipe.nativeIds,recipe.source,backends_,observation,source_->memory(),SourceGraphRecipePhase::Materialize,ordinal,recipe.originalIds);
    }
    /**
     * @brief Materialize the same recipe and retain its original Preparing-source worker permits.
     * @param recipe Next immutable predeclared case. No native completion/readiness is fabricated.
     * @return Actual original local phase roots plus validated original source-worker provenance.
     * @throws std::logic_error For undeclared/reordered or already-retired recipes.
     */
    SourceGraphCase makeWithPreparedWorkers(PooledOriginalRecipe recipe){
        if(retired_ || next_>=recipes_.size() || recipe!=recipes_[next_])throw std::logic_error("Original source recipe changed or was not predeclared");
        const auto ordinal=next_++;
        return SourceGraphCase(recipe.capacities,recipe.reverse,recipe.local,recipe.nativeIds,recipe.source,backends_,nullptr,
            source_->memory(),SourceGraphRecipePhase::Materialize,ordinal,recipe.originalIds,source_);
    }
    /** @brief Replay the actual original validated reset after aliases/handles have retired. */
    void retire(){if(retired_)return;permits_.clear();PooledImmutableFactoryReplayAccess::retireFixtureGenerations();retired_=true;}
    /** @return Exact recorded backend evidence, not a replacement source or ledger. */
    OwnerChainBackendScope& backends(){return *backends_;}
    /** @return Original Preparing/admitted authority for independent reservation oracles. */
    const std::shared_ptr<PhysicalMemoryAuthority>& memory()const{return source_->memory();}
private:
    std::vector<PooledOriginalRecipe> recipes_;std::size_t next_=0;bool retired_=false;
    std::shared_ptr<OwnerChainBackendScope> backends_;
    std::shared_ptr<NativeContextPreparation> source_;std::vector<PreparedGPUContextUsePermit> permits_;
};
}
