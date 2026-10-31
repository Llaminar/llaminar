/**
 * @file PooledImmutableNativePhaseStage.cpp
 * @brief Exact arena/stream/capture binding through the public transfer authority.
 *
 * The stage executes one participant's declared phase. Every cross-participant
 * observation must already be an acquired graph input; this machinery never
 * hides an eager collective, source service or host execution-state mirror.
 */
#include "PooledImmutableNativePhaseStage.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include <array>
#include <stdexcept>
namespace llaminar2 {
namespace {
using Roots=qwen4exp::PooledImmutableNativeRoots;
/** @return Canonical roles are disjoint even when an owner and coordinator share one device. */
std::array<BufferId,3> roles(bool owner){return owner?std::array{BufferId::QFN_POOLED_OWNER_CACHE,BufferId::QFN_POOLED_OWNER_FRAME,BufferId::QFN_POOLED_OWNER_NATIVE}:
    std::array{BufferId::QFN_POOLED_COORDINATOR_POLICY,BufferId::QFN_POOLED_COORDINATOR_FRAME,BufferId::QFN_POOLED_COORDINATOR_NATIVE};}
/** @return Exact original root ordering. */
std::array<std::shared_ptr<TensorBase>,3> tensors(const Roots& roots){return {roots.cache,roots.frame,roots.native};}
/** @brief Validate every role before registering any root, without partial replacement. */
void bind(BufferArena& arena,const Roots& roots,bool owner){
    if(isGraphCaptureActive())throw std::logic_error("Pooled native roots must be declared before capture");
    const auto ids=roles(owner);const auto values=tensors(roots);
    for(std::size_t i=0;i<values.size();++i)if(arena.isRegistered(ids[i]))
        throw std::logic_error("Pooled native canonical arena role was already declared");
    if(!values[1] || !values[2])throw std::invalid_argument("Pooled native binding requires original complete mandatory roots");
    for(std::size_t i=0;i<values.size();++i)if(values[i] && !arena.registerExternalBuffer(ids[i],values[i]))
        throw std::logic_error("Pooled native original root registration failed");
}
/** @return Exact native participant, rejecting absent retained storage before any member access. */
DeviceId participant(const PooledImmutableNativePhaseStage::Phase& phase){
    return std::visit([](const auto& value){if(!value.storage)throw std::invalid_argument("Pooled native phase has no original storage owner");return value.storage->plan().allocator().device;},phase);
}
/** @brief Reject null/default streams before changing stage launch metadata. */
void exactStream(void* stream){if(!stream || reinterpret_cast<std::uintptr_t>(stream)<=2)throw std::invalid_argument("Pooled native stage requires its exact non-default stream");}
}
void bindPooledNativeRoots(BufferArena& arena,const qwen4exp::PooledImmutableOwnerPreparedStorage& storage){storage.captureIdentity();bind(arena,storage.roots(),true);}
void bindPooledNativeRoots(BufferArena& arena,const qwen4exp::PooledImmutableCoordinatorPreparedStorage& storage){storage.captureIdentity();bind(arena,storage.roots(),false);}
PooledImmutableNativePhaseStage::PooledImmutableNativePhaseStage(BufferArena& arena,
    std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels> code,Phase phase)
    :IComputeStage(participant(phase)),arena_(&arena),code_(std::move(code)),phase_(std::move(phase))
{
    if(isGraphCaptureActive() || !device().is_gpu() || !code_ || code_->backend()!=device().type || !code_->codeIdentity())
        throw std::invalid_argument("Pooled native stage requires exact prepared GPU storage and matching native code before capture");
    std::visit([](auto& value){value.operands.seal();value.operands.require(value.action,value.storage->plan());},phase_);
    identity_=graphCaptureVariantSignature();
}
Roots PooledImmutableNativePhaseStage::roots()const{return std::visit([](const auto& value){return value.storage->roots();},phase_);}
const qwen4exp::PooledNativeOperandSpans& PooledImmutableNativePhaseStage::inputs()const{return std::visit([](const auto& value)->const qwen4exp::PooledNativeOperandSpans&{return value.operands.spans();},phase_);}
void PooledImmutableNativePhaseStage::requireBindings()const{
    if(!arena_ || !code_ || code_->backend()!=device().type)throw std::logic_error("Pooled native phase lost its original arena or backend code");
    const auto ids=roles(std::holds_alternative<Owner>(phase_));const auto values=tensors(roots());
    for(std::size_t i=0;i<values.size();++i)
        if(arena_->getTensor(ids[i])!=values[i].get())throw std::logic_error("Pooled native arena authority changed after binding");
    for(const auto& span:inputs())if(span){
        if(arena_->getTensor(span->buffer)!=span->root.get())throw std::logic_error("Pooled native producer role changed its original tensor");
        for(std::size_t i=0;i<values.size();++i)if(values[i] &&
            ((span->root==values[i])!=(span->buffer==ids[i])))
            throw std::logic_error("Pooled native input role aliases a different mutable authority");
    }
    std::visit([](const auto& value){value.storage->captureIdentity();value.operands.require(value.action,value.storage->plan());},phase_);
}
bool PooledImmutableNativePhaseStage::prepareGraphLaunch(IDeviceContext* context,void* stream){
    exactStream(stream);if(!context || context->deviceId()!=device())throw std::invalid_argument("Pooled native capture has a foreign participant");
    requireBindings();if(graphCaptureVariantSignature()!=identity_)throw std::logic_error("Pooled native capture identity changed");
    setGPUStream(stream);std::visit([&](const auto& value){value.storage->acquireSetup(stream);},phase_);return true;
}
bool PooledImmutableNativePhaseStage::execute(IDeviceContext* context){
    if(!context || context->deviceId()!=device())throw std::invalid_argument("Pooled native phase has a foreign execution context");
    requireBindings();if(graphCaptureVariantSignature()!=identity_)throw std::logic_error("Pooled native phase changed after capture");
    auto execution=gpuExecution();
    for(const auto& root:tensors(roots()))if(root){execution.requirePreparedInput(root.get());execution.requirePreparedOutput(root.get());}
    for(const auto& span:inputs())if(span)execution.requirePreparedInput(span->root.get());
    std::visit([&](auto& value){code_->enqueue(value.action,*value.storage,value.operands,execution.nativeStream());},phase_);
    for(const auto& root:tensors(roots()))if(root)execution.publish(root.get());
    return true;
}
std::uint64_t PooledImmutableNativePhaseStage::graphCaptureVariantSignature()const{
    requireBindings();std::uint64_t hash=1469598103934665603ULL;const auto mix=[&](std::uint64_t word){hash=(hash^word)*1099511628211ULL;};
    mix(1);mix(phase_.index());mix(static_cast<std::uint64_t>(device().type));mix(device().ordinal);mix(code_->codeIdentity());
    std::visit([&](const auto& value){mix(static_cast<std::uint64_t>(value.action));mix(value.storage->captureIdentity());mix(value.operands.captureIdentity());},phase_);
    for(const auto id:roles(std::holds_alternative<Owner>(phase_)))mix(static_cast<std::uint64_t>(id));return hash?hash:1;
}
bool PooledImmutableNativePhaseStage::supportsBackend(ComputeBackendType backend)const{return
    (device().is_cuda() && backend==ComputeBackendType::GPU_CUDA) || (device().is_rocm() && backend==ComputeBackendType::GPU_ROCM);}
StageBufferContract PooledImmutableNativePhaseStage::bufferContract()const{
    requireBindings();auto contract=StageBufferContract::build();const auto ids=roles(std::holds_alternative<Owner>(phase_));const auto values=tensors(roots());
    for(std::size_t i=0;i<values.size();++i)if(values[i])contract.addPreallocatedInOut(ids[i],"INT8");
    std::array<bool,static_cast<std::size_t>(BufferId::_COUNT)> declared{};
    for(std::size_t i=0;i<values.size();++i)if(values[i])declared[static_cast<std::size_t>(ids[i])]=true;
    for(const auto& span:inputs())if(span && !declared[static_cast<std::size_t>(span->buffer)]){
        contract.addInput(span->buffer,span->root->native_type()==TensorType::INT8?"INT8":"INT32");declared[static_cast<std::size_t>(span->buffer)]=true;
    }return contract;
}
StageDumpInfo PooledImmutableNativePhaseStage::buildDumpInfoImpl()const{StageDumpInfo result;result.addScalarInt("native_phase_kind",phase_.index());
    std::visit([&](const auto& value){result.addScalarInt("native_phase_action",static_cast<unsigned>(value.action));},phase_);return result;}
}
