/** @file NativeOrderingEventBudgetProjection.cpp
 * @brief Same immutable selector and checked group-growth arithmetic for production and recorded inputs.
 */
#include "NativeOrderingEventBudgetProjection.h"
#include "PooledRemoteGraphHostAllowance.h"
#include <optional>
#include "GPUDeviceContextPool.h"
#include "NativeContextPreparation.h"
#include "RetainedParticipantGraphExecutable.h"
#include <stdexcept>
namespace llaminar2 {
struct NativeOrderingEventBudgetProjection::State final {std::optional<std::vector<RecordedBound>> recorded;std::vector<RecordedRemoteBound> remote;};
NativeOrderingEventBudgetProjection::NativeOrderingEventBudgetProjection(std::shared_ptr<const State> state):state_(std::move(state)){}
NativeOrderingEventBudgetProjection NativeOrderingEventBudgetProjection::installed(){return NativeOrderingEventBudgetProjection(std::make_shared<State>());}
NativeOrderingEventBudgetProjection NativeOrderingEventBudgetProjection::fromRecorded(std::vector<RecordedBound> rows,std::vector<RecordedRemoteBound> remote){
    for(std::size_t i=0;i<rows.size();++i){const auto& row=rows[i];
        if(!row.device.is_gpu() || !row.hostBytes || row.evidence.size()!=64 || row.sourceIdentity.size()!=64 ||
           (row.lifetime!=NativePhysicalLifetime::ContextGeneration && row.lifetime!=NativePhysicalLifetime::ProcessRuntime))throw std::invalid_argument("Recorded ordering trace has incomplete source/bounds/lifetime");
        for(std::size_t j=0;j<i;++j)if(rows[j].device==row.device && rows[j].program==row.program)throw std::invalid_argument("Recorded ordering trace duplicates a program");}
    for(std::size_t i=0;i<remote.size();++i){const auto& row=remote[i];
        if(!row.device.is_gpu() || !row.hostBytes || !row.nativeObjectBytes || row.evidence.size()!=64 || row.sourceIdentity.size()!=64 || !row.family.valid())
            throw std::invalid_argument("Recorded remote graph planning trace has incomplete source/bounds/compiled object facts");
        for(std::size_t j=0;j<i;++j)if(remote[j].device==row.device)throw std::invalid_argument("Recorded remote graph planning trace duplicates an original device");
    }
    auto state=std::make_shared<State>();state->recorded.emplace(std::move(rows));state->remote=std::move(remote);return NativeOrderingEventBudgetProjection(std::move(state));
}
void NativeOrderingEventBudgetProjection::requireOriginalSource(const std::shared_ptr<PhysicalMemoryAuthority>& memory,const GPUContextProvenance& original)const{
    if(!state_)throw std::logic_error("Ordering projection has no sealed origin");
    if(state_->recorded)return; // Only the private recorded-fixture constructor can mint this origin.
    const auto prepared=GPUDeviceContextPool::instance().retainExistingPreparedContext(original.device(),memory);
    original.requireWorker(prepared.worker());
    if(original.generation()!=prepared.generation())throw std::logic_error("Installed ordering group substituted original source generation");
}
NativeAuxiliaryResourceSelection NativeOrderingEventBudgetProjection::select(NativeOrderingEventProgram program,DeviceId device)const{
    if(!state_ || !device.is_gpu())throw std::invalid_argument("Ordering projection lacks original source/device");
    if(state_->recorded){for(const auto& row:*state_->recorded)if(row.device==device && row.program==program)
        return {row.hostBytes,row.deviceBytes,row.evidence,row.sourceIdentity,"recorded-program-source-v1",row.lifetime,0};
        throw std::invalid_argument("Recorded ordering trace omits this exact device/program");}
    return NativeAuxiliaryResourceCatalog::require(device,NativeAuxiliaryResourceProgram::ordering(program));
}
NativeAuxiliaryResourceAllowance NativeOrderingEventBudgetProjection::bounds(NativeOrderingEventProgram program,DeviceId device)const{
    if(!state_ || !device.is_gpu())throw std::invalid_argument("Ordering projection lacks original source/device");
    if(!state_->recorded)return NativeAuxiliaryResourceCatalog::allowance(device.type,NativeAuxiliaryResourceProgram::ordering(program));
    const auto row=select(program,device);
    return {row.hostBytes,row.deviceBytes,row.evidence,row.lifetime,NativeAuxiliaryResourceProgram::ordering(program),row.remoteFamily};
}
namespace {
void resources(PhysicalMemoryResource host,PhysicalMemoryResource gpu){if(!host.valid() || !gpu.valid() || host.device!=DeviceId::cpu() || !gpu.device.is_gpu() || host.world_rank!=gpu.world_rank)
    throw std::invalid_argument("Ordering group changed its original CPU/GPU observations");}
PhysicalMemoryOwner owner(NativeOrderingEventProgram program){return program.kind()==NativeOrderingEventProgram::Kind::ReadAheadParentTerminal?PhysicalMemoryOwner::NativeReadAheadParentTerminal:PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent;}
}
PhysicalMemoryPlan NativeOrderingEventBudgetProjection::bom(NativeOrderingEventProgram program,PhysicalMemoryResource host,PhysicalMemoryResource gpu)const{
    resources(host,gpu);const auto selected=bounds(program,gpu.device);if(!selected.hostBytes)throw std::logic_error("Installed ordering group lacks positive original host commitment");
    PhysicalMemoryPlanBuilder result;result.add(host,owner(program),selected.hostBytes);result.add(gpu,owner(program),selected.deviceBytes);return result.build();
}
PhysicalMemoryPlan NativeOrderingEventBudgetProjection::remoteHostBOM(const PooledRemoteGraphProgram& program,PhysicalMemoryResource host,PhysicalMemoryResource gpu)const{
    resources(host,gpu);if(!state_)throw std::invalid_argument("Remote graph planning lost its sealed original source");
    if(!state_->recorded)return RetainedParticipantGraphExecutable::hostBOM(program,host,gpu);
    for(const auto& row:state_->remote)if(row.device==gpu.device){
        if(!row.family.accepts(program.limits(),PooledRemoteGraphProgramClass::argumentLayout()))
            throw std::invalid_argument("Recorded remote graph planning program exceeds its distinct graph-family trace");
        const NativeAuxiliaryResourceSelection selected{PooledRemoteGraphHostAllowance::scaledBytes(program.limits().maximumNodes,row.hostBytes),0,row.evidence,row.sourceIdentity,"recorded-remote-planning-v1",
            NativePhysicalLifetime::ProcessRuntime,0,NativeAuxiliaryResourceProgram::pooledRemoteOwner(program),row.family};
        return RetainedParticipantGraphExecutable::hostBOMWithSelection(program,host,gpu,selected,row.nativeObjectBytes);
    }
    throw std::invalid_argument("Recorded ordering source omits this distinct remote graph planning family");
}
PhysicalMemoryPlan NativeOrderingEventBudgetProjection::growthBOM(NativeOrderingEventProgram floor,NativeOrderingEventProgram actual,PhysicalMemoryResource host,PhysicalMemoryResource gpu)const{
    resources(host,gpu);if(floor.kind()!=actual.kind() || (floor.nonemptyRootMask()&actual.nonemptyRootMask())!=floor.nonemptyRootMask())throw std::invalid_argument("Ordering growth changed role or removed original roots");
    if(state_ && state_->recorded){
        const auto base=select(floor,gpu.device),full=floor==actual?base:select(actual,gpu.device);
        if(base.sourceIdentity!=full.sourceIdentity || base.resourceSource!=full.resourceSource || base.contextGeneration!=full.contextGeneration)
            throw std::logic_error("Recorded ordering growth changed its original source");
    }
    const auto base=bounds(floor,gpu.device),full=floor==actual?base:bounds(actual,gpu.device);
    if(base.lifetime!=full.lifetime || full.hostBytes<base.hostBytes || full.deviceBytes<base.deviceBytes)throw std::logic_error("Reviewed ordering growth is incompatible or nonmonotone");
    if(floor==actual)return PhysicalMemoryPlanBuilder().build();
    PhysicalMemoryPlanBuilder result;result.add(host,owner(actual),full.hostBytes-base.hostBytes);result.add(gpu,owner(actual),full.deviceBytes-base.deviceBytes);return result.build();
}
}
