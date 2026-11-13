/** @file NativeAuxiliaryResourceCatalog.cpp
 * @brief Select supported-backend resource-family allowances and retain original live ownership metadata.
 */
#include "NativeAuxiliaryResourceCatalog.h"
#include "NativeAuxiliaryStreamOwnership.h"
#include "NativeContextPreparation.h"
#include "ReadAheadNativeResourceObligation.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include <limits>
#include "NativeAuxiliaryQualifiedLimits.h"
#include "GPUDeviceContextPool.h"
#include "PooledRemoteGraphHostAllowance.h"
#include "utils/Sha256.h"
#include <algorithm>
#include <stdexcept>
namespace llaminar2 {
namespace {
/** @return Whether a diagnostic evidence/provider digest is complete; it is not an allowance class. */
bool digest(std::string_view s){return s.size()==64 && s.find_first_not_of('0')!=s.npos && std::all_of(s.begin(),s.end(),[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');});}
/** @brief Length-delimited immutable metadata encoding. */
void append(std::string& out,std::string_view s){out+=std::to_string(s.size())+":";out+=s;}
/** @return SHA256 of metadata only; no native-library file is reopened. */
std::string hash(const std::string& value){const auto out=sha256BytesHex(value);if(!out)throw std::runtime_error("Native auxiliary source identity hashing failed");return *out;}
}
std::string NativeAuxiliaryRemoteFamilyScope::identity()const{
    return "remote-family:"+std::to_string(version)+":"+std::to_string(native.maximumNodes)+":"+
        std::to_string(native.maximumEdges)+":"+std::to_string(arguments.count)+":"+std::to_string(arguments.bytes)+":"+
        std::to_string(arguments.alignment);
}
std::string NativeAuxiliaryResourceProgram::identity()const{
    if(isPeerTransferEvents())return peerTransferTiming()?"peer-transfer-ordering1-timing2-v1":"peer-transfer-ordering1-timing0-v1";
    if(isAuxiliaryStream())return "auxiliary-stream1-ordering0-v1";
    if(isReadAheadSide())return "side-stream1-ordering2-v1";
    if(isPooledRemoteOwner())return remoteProgramClass().identity();
    return "ordering-v1:"+std::to_string(static_cast<unsigned>(orderingProgram().kind()))+":"+std::to_string(orderingProgram().nonemptyRootMask())+
        (orderingProgram().kind()==NativeOrderingEventProgram::Kind::RetainedCaptureTerminals
            ? ":events="+std::to_string(orderingProgram().orderingEvents()):std::string{});
}
const std::string& NativeAuxiliaryResourceCatalog::resourceSourceIdentity(){static const std::string value="native-auxiliary-operations";return value;}
std::string NativeAuxiliaryResourceCatalog::portableSourceIdentity(const NativeContextBirthKey& k){
    k.requireComplete();std::string out="native-auxiliary-source-class-v1:";
    for(const auto& s:{std::to_string(static_cast<unsigned>(k.device.type)),std::to_string(k.source_program_version),std::to_string(static_cast<unsigned>(k.population)),k.architecture,
        std::to_string(k.compute_units),std::to_string(k.native_wave_width),k.runtime_version,k.driver_version,k.backend_sha256,k.runtime_sha256,k.driver_sha256,k.runtime_policy_sha256,k.hsa_sha256,k.pthread_defaults_identity,
        std::to_string(k.worker_stack_bytes),std::to_string(k.worker_guard_bytes),std::to_string(k.context_object_bytes),std::to_string(k.context_object_alignment),
        std::to_string(k.primary_streams),std::to_string(k.primary_blas_handles),std::to_string(k.secondary_blas_handles),std::to_string(k.dependency_events)})append(out,s);
    return hash(out);
}
std::string NativeAuxiliaryResourceCatalog::liveSourceIdentity(const NativeContextBirthKey& k){
    std::string out=portableSourceIdentity(k);for(const auto& s:{k.device.toString(),k.physical_uuid,k.visibility_identity,std::to_string(k.visible_device_count)})append(out,s);return hash(out);
}
NativeAuxiliaryResourceAllowance NativeAuxiliaryResourceCatalog::selectAllowance(DeviceType backend,
    NativeAuxiliaryResourceProgram program,std::span<const detail::NativeAuxiliaryQualifiedClass> rows){
    if(backend!=DeviceType::CUDA && backend!=DeviceType::ROCm)
        throw std::invalid_argument("Native auxiliary allowance requires a supported GPU backend");
    if(program.isPeerTransferEvents()) {
        // Keep timestamp semantics explicit. The timed trio uses the entire
        // conservative four-event family envelope, not a per-handle startup
        // multiplication. These are opaque ceilings, never measured exact bytes.
        // Original source/flags review and CUDA/ROCm native peer-lane replay
        // qualify this closed timing operation independently of Ordering.
        auto bound=selectAllowance(backend,NativeAuxiliaryResourceProgram::ordering(
            program.peerTransferTiming()?NativeOrderingEventProgram::pooledOwner(true):
                NativeOrderingEventProgram::lookupSubmissionRetirement()),rows);
        bound.program=program;return bound;
    }
    // The same ordinary ordering-event native operation is repeated over the
    // original admitted slot roster. Scale its installed UNIT allowance here,
    // once, rather than treating one fixture's borrowed population as many.
    if(program.isOrdering() && (program.orderingProgram().kind()==NativeOrderingEventProgram::Kind::RetainedCaptureTerminals ||
        program.orderingProgram().kind()==NativeOrderingEventProgram::Kind::PooledLocalHotInitialization)){
        auto unit=selectAllowance(backend,NativeAuxiliaryResourceProgram::ordering(
            NativeOrderingEventProgram::lookupSubmissionRetirement()),rows);
        const auto count=program.orderingProgram().orderingEvents();
        if(!count || count>std::numeric_limits<std::size_t>::max()/unit.hostBytes ||
            (unit.deviceBytes && count>std::numeric_limits<std::size_t>::max()/unit.deviceBytes))
            throw std::overflow_error("Original capture terminal event population allowance overflow");
        unit.hostBytes*=count;unit.deviceBytes*=count;unit.program=program;return unit;
    }
    // Exact capture terminals perform the SAME ordinary one-event operation:
    // timing-disabled create, record, wait/query and destroy. Reuse the
    // certified unit bound, not the read-ahead fixture's borrowed population.
    // Actual instances claim distinct NativeOrderingEventGroups below. The
    // existing NativeImmutableLookupOrderingEvent bucket already covers these
    // ordinary pooled/boundary primitives; program identity remains distinct.
    if(program.isOrdering() && program.orderingProgram().kind()==NativeOrderingEventProgram::Kind::CaptureSubmissionRetirement){
        auto unit=selectAllowance(backend,NativeAuxiliaryResourceProgram::ordering(
            NativeOrderingEventProgram::lookupSubmissionRetirement()),rows);
        unit.program=program;return unit;
    }
    for(const auto& row:rows){if(row.backend!=backend)continue;
        if(row.remoteFamily && (!row.remoteFamily->valid() || !row.program.isPooledRemoteOwner() ||
            !row.remoteFamily->accepts(row.program.remoteProgramClass().limits(),PooledRemoteGraphProgramClass::argumentLayout())))
            throw std::logic_error("Installed remote graph family has invalid reviewed source scope");
        const bool remote=row.remoteFamily && program.isPooledRemoteOwner() &&
            row.remoteFamily->accepts(program.remoteProgramClass().limits(),PooledRemoteGraphProgramClass::argumentLayout());
        if(row.remoteFamily ? !remote : row.program!=program)continue;
        if(program.isPooledRemoteOwner() && !row.remoteFamily)
            throw std::logic_error("Remote graph host qualification requires reviewed family ceilings");
        if(!row.hostBytes || ((program.isReadAheadSide() || program.isAuxiliaryStream()) && !row.deviceBytes) || !digest(row.evidence) || (row.lifetime!=NativePhysicalLifetime::ContextGeneration && row.lifetime!=NativePhysicalLifetime::ProcessRuntime))throw std::logic_error("Installed auxiliary program has incomplete reviewed evidence/lifetime");
        if(program.isPooledRemoteOwner() && (row.deviceBytes || row.lifetime!=NativePhysicalLifetime::ProcessRuntime))
            throw std::logic_error("Remote graph host class must keep original device bytes separate and retain host backing through ProcessRuntime");
        const auto hostBytes=program.isPooledRemoteOwner()?PooledRemoteGraphHostAllowance::scaledBytes(program.remoteProgramClass().limits().maximumNodes,row.hostBytes):row.hostBytes;
        return {hostBytes,row.deviceBytes,std::string(row.evidence),row.lifetime,program,row.remoteFamily};}
    throw std::runtime_error("Native auxiliary program is unqualified: no reviewed source/runtime/program class for "+std::to_string(static_cast<unsigned>(backend))+" "+program.identity());
}
NativeAuxiliaryResourceAllowance NativeAuxiliaryResourceCatalog::allowance(DeviceType backend,NativeAuxiliaryResourceProgram program){
    return selectAllowance(backend,program,detail::kNativeAuxiliaryQualifiedClasses);
}
NativeAuxiliaryResourceSelection NativeAuxiliaryResourceCatalog::select(const NativeContextBirthKey& k,std::uint64_t generation,
    NativeAuxiliaryResourceProgram program,std::string_view resource,std::span<const detail::NativeAuxiliaryQualifiedClass> rows){
    k.requireComplete();if(!generation || resource.empty())throw std::invalid_argument("Native auxiliary class requires original generation/source");
    const auto bound=selectAllowance(k.device.type,program,rows);
    return {bound.hostBytes,bound.deviceBytes,bound.evidence,liveSourceIdentity(k),std::string(resource),bound.lifetime,generation,program,bound.remoteFamily};
}
NativeAuxiliaryResourceSelection NativeAuxiliaryResourceCatalog::require(DeviceId device,NativeAuxiliaryResourceProgram program){
    if(!device.is_gpu())throw std::invalid_argument("Native auxiliary program requires an actual GPU");
    if(detail::kNativeAuxiliaryQualifiedClasses.empty())throw std::runtime_error("Native auxiliary program is unqualified: reviewed data catalog is empty for "+device.toString()+" "+program.identity());
    const auto original=GPUDeviceContextPool::instance().retainExistingContextProvenance(device);
    const auto key=original.originalBirthKey(); // Strong alias of actual original contract; no cache/query/hash.
    return select(*key,original.generation(),program,resourceSourceIdentity(),detail::kNativeAuxiliaryQualifiedClasses);
}
}

namespace llaminar2 {
/** @brief No worker alias: the context record itself retains this original obligation. */
struct NativeAuxiliaryStreamOwnership::State final {
    std::shared_ptr<PhysicalMemoryAuthority> memory;
    DeviceId device;std::uint64_t generation;
    const IWorkerGPUContext* worker;std::string name;GPUAuxiliaryStreamSchedulingClass scheduling;
    NativeContextRetirementSlot slot;
    std::unique_ptr<ReadAheadNativeResourceObligation> pending;
    enum class Phase {Claimed,Attempted,ContextOwned,Retired};Phase phase=Phase::Claimed;
    State(const PreparedGPUContextUsePermit& p,std::string n,GPUAuxiliaryStreamSchedulingClass priority,NativeContextRetirementSlot s,
          std::unique_ptr<ReadAheadNativeResourceObligation> leaf)
        :memory(p.memory()),device(p.device()),generation(p.generation()),worker(p.worker().get()),
         name(std::move(n)),scheduling(priority),slot(std::move(s)),pending(std::move(leaf)){}
};
PhysicalMemoryPlan NativeAuxiliaryStreamOwnership::bom(PhysicalMemoryResource host,PhysicalMemoryResource gpu){
    if(!host.valid() || !gpu.valid() || !host.device.is_cpu() || !gpu.device.is_gpu() || host.world_rank!=gpu.world_rank)
        throw std::invalid_argument("Auxiliary stream requires exact original host/GPU resources");
    const auto selected=NativeAuxiliaryResourceCatalog::allowance(gpu.device.type,NativeAuxiliaryResourceProgram::auxiliaryStream());
    PhysicalMemoryPlanBuilder plan;
    plan.add(host,PhysicalMemoryOwner::NativeAuxiliaryStream,selected.hostBytes);
    if(selected.deviceBytes)plan.add(gpu,PhysicalMemoryOwner::NativeAuxiliaryStream,selected.deviceBytes);
    return plan.build();
}
NativeAuxiliaryStreamOwnership::NativeAuxiliaryStreamOwnership(std::unique_ptr<State> state):state_(std::move(state)){}
std::unique_ptr<NativeAuxiliaryStreamOwnership> NativeAuxiliaryStreamOwnership::claim(
    const PreparedGPUContextUsePermit& original,const std::string& name){
    return claim(original,name,GPUAuxiliaryStreamSchedulingClass::Normal);
}
std::unique_ptr<NativeAuxiliaryStreamOwnership> NativeAuxiliaryStreamOwnership::claim(
    const PreparedGPUContextUsePermit& original,const std::string& name,GPUAuxiliaryStreamSchedulingClass scheduling){
    if(scheduling!=GPUAuxiliaryStreamSchedulingClass::Normal && scheduling!=GPUAuxiliaryStreamSchedulingClass::LatencyCritical &&
       scheduling!=GPUAuxiliaryStreamSchedulingClass::BackgroundMaintenance)
        throw std::invalid_argument("Auxiliary stream requires a declared scheduling class");
    original.requireWorker(original.worker());
    if(name.empty() || isGraphCaptureActive())throw std::invalid_argument("Auxiliary stream commitment requires original cold source/name");
    const auto selected=NativeAuxiliaryResourceCatalog::require(original.device(),NativeAuxiliaryResourceProgram::auxiliaryStream());
    if(selected.contextGeneration!=original.generation() || !selected.hostBytes)
        throw std::logic_error("Auxiliary stream changed original source generation");
    // Exact immutable strings, not another source hash or a guessed physical allowance.
    const auto certificate=selected.sourceIdentity+":"+selected.resourceSource+":"+selected.evidence+":auxiliary-stream1-v1";
    auto slot=original.provenance().reserveNativeRetirementSlot(original.memory(),certificate);
    const auto population="auxiliary-stream1-v1:"+std::to_string(name.size())+":"+name+
        (scheduling==GPUAuxiliaryStreamSchedulingClass::Normal?std::string{}:":scheduling="+std::to_string(static_cast<unsigned>(scheduling)));
    std::unique_ptr<ReadAheadNativeResourceObligation> leaf;
    const bool borrowed=selected.lifetime==NativePhysicalLifetime::ProcessRuntime && slot.tryBorrowProcessRuntime(population);
    if(!borrowed){
        auto host=original.memory()->reserveNewAllocations(DeviceId::cpu(),PhysicalMemoryOwner::NativeAuxiliaryStream,selected.hostBytes);
        std::optional<PhysicalMemoryOwnerReservation> gpu;
        if(selected.deviceBytes)gpu.emplace(original.memory()->reserveNewAllocations(original.device(),PhysicalMemoryOwner::NativeAuxiliaryStream,selected.deviceBytes));
        if(selected.lifetime==NativePhysicalLifetime::ProcessRuntime)slot.prepareProcessRuntime(population);
        leaf=std::unique_ptr<ReadAheadNativeResourceObligation>(new ReadAheadNativeResourceObligation(original.memory(),original.device(),original.generation(),certificate,std::move(host),std::move(gpu),selected.lifetime));
    }
    return std::unique_ptr<NativeAuxiliaryStreamOwnership>(new NativeAuxiliaryStreamOwnership(
        std::make_unique<State>(original,name,scheduling,std::move(slot),std::move(leaf))));
}
void NativeAuxiliaryStreamOwnership::requireOriginal(const PreparedGPUContextUsePermit& p,
    const IWorkerGPUContext* worker,const std::string& name)const{
    requireOriginal(p,worker,name,GPUAuxiliaryStreamSchedulingClass::Normal);
}
void NativeAuxiliaryStreamOwnership::requireOriginal(const PreparedGPUContextUsePermit& p,
    const IWorkerGPUContext* worker,const std::string& name,GPUAuxiliaryStreamSchedulingClass scheduling)const{
    if(!state_ || state_->phase!=State::Phase::ContextOwned || p.device()!=state_->device || p.generation()!=state_->generation ||
       p.worker().get()!=worker || worker!=state_->worker || name!=state_->name || scheduling!=state_->scheduling ||
       p.memory().get()!=state_->memory.get() || p.memory().owner_before(state_->memory) || state_->memory.owner_before(p.memory()))
        throw std::logic_error("Named auxiliary stream changed original authority/device/generation/name/scheduling");
    p.requireWorker(p.worker());
}
void NativeAuxiliaryStreamOwnership::requireRetirement(RetirementScope scope)const{
    if(!state_ || state_->phase!=State::Phase::ContextOwned || scope!=RetirementScope::OriginalContextShutdown)
        throw std::logic_error("Original admitted auxiliary streams retire only with their original context");
}
void NativeAuxiliaryStreamOwnership::beginNativeAttempt(){
    if(!state_ || state_->phase!=State::Phase::Claimed || isGraphCaptureActive())throw std::logic_error("Auxiliary stream native creation is cold and one-shot");
    if(state_->pending)state_->slot.adopt(std::move(state_->pending));
    state_->slot.armNativeAttempt();state_->phase=State::Phase::Attempted;
}
void NativeAuxiliaryStreamOwnership::delegateToContext()noexcept{
    if(!state_ || state_->phase!=State::Phase::Attempted)std::terminate();
    state_->slot.delegateHandlesToContext();state_->phase=State::Phase::ContextOwned;
}
void NativeAuxiliaryStreamOwnership::completeNativeHandles()noexcept{
    if(!state_ || state_->phase==State::Phase::Retired)std::terminate();
    if(state_->phase==State::Phase::Claimed)state_->pending.reset();
    else state_->slot.completeNativeHandles();
    state_->phase=State::Phase::Retired;
}
NativeAuxiliaryStreamOwnership::~NativeAuxiliaryStreamOwnership()noexcept{
    if(state_ && state_->phase!=State::Phase::Claimed && state_->phase!=State::Phase::Retired)std::terminate();
}
}
