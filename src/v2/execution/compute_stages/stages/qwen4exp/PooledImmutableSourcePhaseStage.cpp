/**
 * @file PooledImmutableSourcePhaseStage.cpp
 * @brief Exact arena/capture/lifetime binding for the production immutable source service.
 *
 * CPU worker retirement releases only its own request. Strong boundary and
 * external arena aliases retain original PMA roots until complete graph and
 * final native readers retire. Kernel publication uses the actual executor
 * stream; no hidden host wait, transfer, allocation or cache shadow is present.
 */
#include "PooledImmutableSourcePhaseStage.h"
#include "kernels/qwen4exp/PooledImmutableSourcePreparedStorage.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "transfer/TransferEngine.h"
#include <array>
#include <limits>
#include <stdexcept>
namespace llaminar2 {
namespace {
using Roots=qwen4exp::PooledImmutableNativeRoots;
/** @return Exact original root order under its distinct canonical authority roles. */
std::array<std::shared_ptr<TensorBase>,3> roots(const Roots& r){return {r.cache,r.frame,r.native};}
/** @return Actual participant before the base stage is constructed; foreign/absent source arms fail. */
DeviceId participant(const std::shared_ptr<qwen4exp::ImmutableLookupExecutionBoundary>& boundary){
    if(!boundary || !boundary->isPooled())throw std::invalid_argument("Pooled source phase requires its original real-provider boundary");
    return boundary->device();
}
/** @return Immutable role/root pairs; building this small array never reads a live device value. */
auto bindings(const qwen4exp::PooledImmutableSourcePreparedStorage& storage){
    const auto source=storage.roots(),owner=storage.owner()->roots(),coordinator=storage.coordinator()->roots();
    return std::array<std::pair<BufferId,std::shared_ptr<TensorBase>>,8>{{
        {BufferId::QFN_POOLED_SOURCE_FRAME,source.frame},{BufferId::QFN_POOLED_SOURCE_NATIVE,source.native},
        {BufferId::QFN_POOLED_OWNER_CACHE,owner.cache},{BufferId::QFN_POOLED_OWNER_FRAME,owner.frame},{BufferId::QFN_POOLED_OWNER_NATIVE,owner.native},
        {BufferId::QFN_POOLED_COORDINATOR_POLICY,coordinator.cache},{BufferId::QFN_POOLED_COORDINATOR_FRAME,coordinator.frame},{BufferId::QFN_POOLED_COORDINATOR_NATIVE,coordinator.native}}};
}
}
void bindPooledSourceRoots(BufferArena& arena,const qwen4exp::ImmutableLookupExecutionBoundary& boundary){
    if(isGraphCaptureActive())throw std::logic_error("Pooled source roots must be declared before capture");
    const auto source=boundary.pooledStorage()->roots();
    const std::array pairs{std::pair{BufferId::QFN_POOLED_SOURCE_FRAME,source.frame},
        std::pair{BufferId::QFN_POOLED_SOURCE_NATIVE,source.native}};
    for(const auto& [role,root]:pairs)
        if(arena.isRegistered(role) && arena.getTensor(role)!=root.get())
            throw std::logic_error("Pooled source canonical arena role belongs to another original root");
    for(const auto& [role,root]:pairs)
        if(!arena.isRegistered(role) && !arena.registerExternalBuffer(role,root))
            throw std::logic_error("Pooled source original root registration failed");
}
PooledImmutableSourcePhaseStage::PooledImmutableSourcePhaseStage(BufferArena& arena,
    std::shared_ptr<qwen4exp::ImmutableLookupExecutionBoundary> boundary,qwen4exp::PooledSourceAction action,
    DeviceGenerationControlBinding control,std::optional<qwen4exp::PooledNativeTensorSpan> output)
    :IComputeStage(participant(boundary)),arena_(&arena),boundary_(std::move(boundary)),action_(action),control_(control),output_(std::move(output)){
    if(isGraphCaptureActive())throw std::logic_error("Pooled source phase must be declared before capture");
    boundary_->bindPooledGenerationControl(control_);
    const auto& p=boundary_->pooledStorage()->geometry().parameters();
    const auto count=std::size_t{p.coordinator.policy.maximumOriginal()}*p.row_elements;
    if(count>std::numeric_limits<std::size_t>::max()/sizeof(float))throw std::overflow_error("Pooled source output extent overflows");
    if(bool(output_)!=(action_==qwen4exp::PooledSourceAction::Read))throw std::invalid_argument("Only the original source reader owns an output span");
    if(output_){
        const auto& out=*output_;
        if(!out.root || out.root->is_view() || out.root->transferStorageOwner()!=out.root.get() || out.root->native_type()!=TensorType::FP32 ||
           out.buffer>=BufferId::_COUNT || !out.root->cols() || out.bytes%(out.root->cols()*sizeof(float)) || out.offset%4 || (!out.bytes || out.bytes>count*sizeof(float) || out.bytes%(p.row_elements*sizeof(float))) || out.offset>out.root->size_bytes() ||
           out.bytes>out.root->size_bytes()-out.offset)
            throw std::invalid_argument("Pooled reader requires its original exact FP32 span declaration");
        const std::array<TensorBase*,1> original{out.root.get()};prepared_.emplace(device(),original);
    }
    requireDeclarations();identity_=graphCaptureVariantSignature();
}
void PooledImmutableSourcePhaseStage::requireDeclarations()const{
    control_.requireFor(device());const auto& storage=*boundary_->pooledStorage();
    for(const auto& [role,root]:bindings(storage))
        if(arena_->getTensor(role)!=root.get())throw std::logic_error("Pooled source original arena authority changed");
    if(arena_->getTensor(BufferId::MTP_GENERATION_CONTROL)!=control_.owner())throw std::logic_error("Pooled source controller role changed");
    if(output_){
        if(arena_->getTensor(output_->buffer)!=output_->root.get() || !prepared_)
            throw std::logic_error("Pooled source output role lost its original physical owner");
        prepared_->validateIfSealed();
        for(const auto& [role,root]:bindings(storage))if(output_->buffer==role || (root && output_->root==root))
            throw std::logic_error("Pooled source output aliases a native authority role");
    }
}
void PooledImmutableSourcePhaseStage::requirePreparedBindings()const{
    requireDeclarations();
    if(output_){
        prepared_->validate();
        if(!captured_)throw std::logic_error("Pooled source output has no retained original allocation borrower");
        TransferEngine::instance().requireCapturedDeviceStorage(*captured_,*output_->root,device(),output_->root->size_bytes());
    }
}
bool PooledImmutableSourcePhaseStage::prepareGraphLaunch(IDeviceContext* context,void* stream){
    if(!context || context->deviceId()!=device() || !stream || reinterpret_cast<std::uintptr_t>(stream)<=2)
        throw std::invalid_argument("Pooled source preparation requires its exact participant/stream");
    requireDeclarations();if(graphCaptureVariantSignature()!=identity_)throw std::logic_error("Pooled source capture identity changed");
    if(output_){
        // Ordinary graph storage is bound by the executor's cold arena pass.
        // The captured borrower must retain that exact allocation until graph retirement.
        prepared_->seal();
        if(!captured_)captured_=TransferEngine::instance().retainCapturedDeviceStorage(*output_->root,device(),output_->root->size_bytes());
    }
    requirePreparedBindings();
    setGPUStream(stream);boundary_->pooledStorage()->acquireSetup(stream);return true;
}
bool PooledImmutableSourcePhaseStage::execute(IDeviceContext* context){
    if(!context || context->deviceId()!=device())throw std::invalid_argument("Pooled source execution has a foreign participant");
    requirePreparedBindings();if(graphCaptureVariantSignature()!=identity_)throw std::logic_error("Pooled source executable identity changed");
    if(!isGraphCaptureActive())boundary_->requireArmedForLaunch();
    auto execution=gpuExecution();const auto& storage=*boundary_->pooledStorage();
    for(const auto& [role,root]:bindings(storage))if(root)execution.requirePreparedInput(root.get());
    for(const auto& root:{storage.roots().frame,storage.roots().native})execution.requirePreparedOutput(root.get());
    execution.requirePreparedInput(control_.owner());execution.requirePreparedOutput(control_.owner());
    qwen4exp::PooledSourceQuery query;
    if(output_){execution.requirePreparedOutput(output_->root.get());query={reinterpret_cast<float*>(static_cast<std::byte*>(output_->root->gpu_data_ptr())+output_->offset),output_->bytes/sizeof(float)};}
    boundary_->pooledKernels().enqueue(action_,*boundary_->pooledStorage(),control_,query,execution.nativeStream());
    execution.publish(storage.roots().frame.get());execution.publish(storage.roots().native.get());execution.publish(control_.owner());
    if(output_)execution.publish(output_->root.get());return true;
}
std::uint64_t PooledImmutableSourcePhaseStage::graphCaptureVariantSignature()const{
    requireDeclarations();std::uint64_t h=1469598103934665603ULL;const auto mix=[&](std::uint64_t word){h=(h^word)*1099511628211ULL;};
    mix(1);mix(static_cast<unsigned>(action_));mix(boundary_->captureIdentity());mix(control_.captureIdentity());
    mix(boundary_->pooledKernels().captureIdentity(*boundary_->pooledStorage()));
    for(const auto& [role,root]:bindings(*boundary_->pooledStorage())){mix(static_cast<std::uint64_t>(role));mix(reinterpret_cast<std::uintptr_t>(root.get()));}
    if(output_){mix(static_cast<std::uint64_t>(output_->buffer));mix(reinterpret_cast<std::uintptr_t>(output_->root.get()));mix(output_->offset);mix(output_->bytes);mix(static_cast<std::uint64_t>(output_->root->native_type()));mix(output_->root->size_bytes());}
    return h?h:1;
}
bool PooledImmutableSourcePhaseStage::supportsBackend(ComputeBackendType backend)const{return
    (device().is_cuda() && backend==ComputeBackendType::GPU_CUDA) || (device().is_rocm() && backend==ComputeBackendType::GPU_ROCM);}
StageBufferContract PooledImmutableSourcePhaseStage::bufferContract()const{
    requireDeclarations();auto contract=StageBufferContract::build();
    for(const auto& [role,root]:bindings(*boundary_->pooledStorage()))if(root){
        if(role==BufferId::QFN_POOLED_SOURCE_FRAME || role==BufferId::QFN_POOLED_SOURCE_NATIVE)contract.addPreallocatedInOut(role,"INT8");
        else contract.addInput(role,"INT8");
    }
    contract.addPreallocatedInOut(BufferId::MTP_GENERATION_CONTROL,"INT32");
    if(output_)contract.addOutput(output_->buffer,"FP32");return contract;
}
StageDumpInfo PooledImmutableSourcePhaseStage::buildDumpInfoImpl()const{
    StageDumpInfo dump;dump.addScalarInt("source_phase",static_cast<unsigned>(action_));
    if(output_){
        const auto& output=*output_;const auto* base=static_cast<const std::byte*>(output.root->raw_data());
        const auto columns=output.root->cols(),rows=output.bytes/(columns*sizeof(float));
        dump.outputs.push_back({"gathered_rows",base?base+output.offset:nullptr,rows,columns,"FP32",sizeof(float),
            output.bytes,output.root.get(),{},output.offset});
        dump.addScalarInt("native_row_bytes",boundary_->pooledStorage()->geometry().parameters().owner.row_bytes);
    }
    return dump;
}
}
