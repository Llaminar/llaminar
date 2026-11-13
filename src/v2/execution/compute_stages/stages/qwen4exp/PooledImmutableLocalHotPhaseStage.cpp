/**
 * @file PooledImmutableLocalHotPhaseStage.cpp
 * @brief Bind original local private protocol storage without an owner-result model dependency.
 *
 * The main model sees exactly one ordinary output writer. Native private-frame
 * publication is governed by the explicit ready/pin protocol, so auxiliary
 * policy maintenance cannot accidentally become a model-data dependency.
 */
#include "PooledImmutableLocalHotPhaseStage.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
namespace llaminar2 {
namespace {
using Action=qwen4exp::PooledImmutableLocalHotAction;
/** @return Actual original local allocator, rejecting absence before dereference. */
DeviceId localDevice(const std::shared_ptr<qwen4exp::PooledImmutableLocalHotPreparedStorage>& storage)
{if(!storage)throw std::invalid_argument("Local native stage requires original prepared storage");return storage->geometry().allocator().device;}
/** @brief Reject null/default streams before recording or changing stage bindings. */
void exactLocalStream(void* stream)
{if(!stream || reinterpret_cast<std::uintptr_t>(stream)<=2)throw std::invalid_argument("Local native stage requires its exact nondefault stream");}
/** @return Matching actual native backend, without hardware discovery. */
bool localBackend(DeviceId device,ComputeBackendType backend)
{return (device.is_cuda() && backend==ComputeBackendType::GPU_CUDA) || (device.is_rocm() && backend==ComputeBackendType::GPU_ROCM);}
}
PooledImmutableLocalHotBeginStage::PooledImmutableLocalHotBeginStage(BufferArena& arena,
    std::shared_ptr<qwen4exp::PooledImmutableSourcePreparedStorage> source,
    std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels> code,qwen4exp::PooledImmutableLocalHotPoolQuery query)
    :IComputeStage(source?source->geometry().allocator().device:DeviceId::invalid()),arena_(&arena),source_(std::move(source)),
     code_(std::move(code)),query_(std::move(query))
{
    if(isGraphCaptureActive() || !source_ || !code_ || code_->backend()!=device().type || !code_->codeIdentity())
        throw std::invalid_argument("Compact Begin requires its original source/coordinator/code before capture");
    identity_=graphCaptureVariantSignature();
}
void PooledImmutableLocalHotBeginStage::requireDeclarations()const
{
    if(!arena_ || !source_ || !code_ || source_->geometry().allocator().device!=device() || code_->backend()!=device().type)
        throw std::logic_error("Compact Begin changed original source/backend");
    source_->captureIdentity();const auto roots=source_->coordinator()->roots();
    for(const auto& [id,root]:std::array<std::pair<BufferId,std::shared_ptr<TensorBase>>,4>{
        std::pair{BufferId::QFN_POOLED_COORDINATOR_POLICY,roots.cache},
        std::pair{BufferId::QFN_POOLED_COORDINATOR_FRAME,roots.frame},
        std::pair{BufferId::QFN_POOLED_COORDINATOR_NATIVE,roots.native},
        std::pair{BufferId::QFN_POOLED_SOURCE_NATIVE,source_->roots().native}})
        if(root && arena_->getTensor(id)!=root.get())throw std::logic_error("Compact Begin changed a canonical original source/coordinator role");
}
bool PooledImmutableLocalHotBeginStage::prepareGraphLaunch(IDeviceContext* context,void* stream)
{
    exactLocalStream(stream);if(!context || context->deviceId()!=device())throw std::invalid_argument("Compact Begin capture changed participant");
    requireDeclarations();if(graphCaptureVariantSignature()!=identity_)throw std::logic_error("Compact Begin declaration changed");
    query_.seal();query_.require(source_->coordinator()->plan());source_->acquireSetup(stream);source_->coordinator()->acquireSetup(stream);
    setGPUStream(stream);return true;
}
bool PooledImmutableLocalHotBeginStage::execute(IDeviceContext* context)
{
    if(!context || context->deviceId()!=device())throw std::invalid_argument("Compact Begin execution changed participant");
    requireDeclarations();query_.require(source_->coordinator()->plan());
    if(graphCaptureVariantSignature()!=identity_)throw std::logic_error("Compact Begin capture identity changed");
    auto execution=gpuExecution();const auto roots=source_->coordinator()->roots();
    for(const auto& root:{roots.cache,roots.frame,roots.native})if(root){execution.requirePreparedInput(root.get());execution.requirePreparedOutput(root.get());}
    execution.requirePreparedInput(source_->roots().native.get());
    code_->enqueueCompactBegin(*source_->coordinator(),query_,execution.nativeStream());
    for(const auto& root:{roots.cache,roots.frame,roots.native})if(root)execution.publish(root.get());
    return true;
}
std::uint64_t PooledImmutableLocalHotBeginStage::graphCaptureVariantSignature()const
{
    requireDeclarations();auto h=source_->captureIdentity();
    for(const auto word:{std::uint64_t{1},code_->codeIdentity(),query_.declarationIdentity()})h=(h^word)*1099511628211ULL;
    return h?h:1;
}
bool PooledImmutableLocalHotBeginStage::supportsBackend(ComputeBackendType backend)const{return localBackend(device(),backend);}
StageBufferContract PooledImmutableLocalHotBeginStage::bufferContract()const
{
    requireDeclarations();auto result=StageBufferContract::build();const auto roots=source_->coordinator()->roots();
    if(roots.cache)result.addPreallocatedInOut(BufferId::QFN_POOLED_COORDINATOR_POLICY,"INT8");
    result.addPreallocatedInOut(BufferId::QFN_POOLED_COORDINATOR_FRAME,"INT8");result.addPreallocatedInOut(BufferId::QFN_POOLED_COORDINATOR_NATIVE,"INT8");
    result.addInput(BufferId::QFN_POOLED_SOURCE_NATIVE,"INT8");return result;
}
StageDumpInfo PooledImmutableLocalHotBeginStage::buildDumpInfoImpl()const
{StageDumpInfo result;result.addScalarInt("compact_pool_begin",1);return result;}
PooledImmutableLocalHotPhaseStage::PooledImmutableLocalHotPhaseStage(BufferArena& arena,
    std::shared_ptr<qwen4exp::PooledImmutableLocalHotPreparedStorage> storage,
    std::shared_ptr<const qwen4exp::IPooledImmutableLocalHotKernels> code,Action action,
    qwen4exp::PooledImmutableLocalHotOperands operands)
    :IComputeStage(localDevice(storage)),arena_(&arena),storage_(std::move(storage)),code_(std::move(code)),
     action_(action),operands_(std::move(operands))
{
    if(isGraphCaptureActive() || !code_ || code_->backend()!=device().type || !code_->codeIdentity())
        throw std::invalid_argument("Local phase requires original native roots/code before capture");
    identity_=graphCaptureVariantSignature();
}
void PooledImmutableLocalHotPhaseStage::requireDeclarations()const
{
    if(!arena_ || !storage_ || !code_ || storage_->geometry().allocator().device!=device() || code_->backend()!=device().type ||
        arena_->getTensor(BufferId::MTP_GENERATION_CONTROL)!=operands_.controllerOwner())
        throw std::logic_error("Local phase changed original participant/arena/controller owner");
    storage_->captureIdentity();
    if(sealed_operands_){operands_.require(action_,*storage_);if(*sealed_operands_!=operands_.captureIdentity())throw std::logic_error("Local phase changed sealed original physical producers");}
    for(const auto& span:operands_.spans())if(arena_->getTensor(span.buffer)!=span.root.get())
        throw std::logic_error("Local phase changed its original ordinary producer/output role");
}
bool PooledImmutableLocalHotPhaseStage::prepareGraphLaunch(IDeviceContext* context,void* stream)
{
    exactLocalStream(stream);if(!context || context->deviceId()!=device())throw std::invalid_argument("Local phase capture changed participant");
    requireDeclarations();if(graphCaptureVariantSignature()!=identity_)throw std::logic_error("Local phase capture identity changed");
    operands_.seal();operands_.require(action_,*storage_);sealed_operands_=operands_.captureIdentity();storage_->acquireSetup(stream);setGPUStream(stream);return true;
}
bool PooledImmutableLocalHotPhaseStage::execute(IDeviceContext* context)
{
    if(!context || context->deviceId()!=device())throw std::invalid_argument("Local phase execution changed participant");
    requireDeclarations();operands_.require(action_,*storage_);
    if(graphCaptureVariantSignature()!=identity_)throw std::logic_error("Local phase changed after capture");
    auto execution=gpuExecution();
    execution.requirePreparedInput(operands_.controllerOwner());execution.requirePreparedOutput(operands_.controllerOwner());
    for(const auto& span:operands_.spans()){
        if(action_==Action::LocalRead)execution.requirePreparedOutput(span.root.get());
        else execution.requirePreparedInput(span.root.get());
    }
    code_->enqueue(action_,*storage_,operands_,execution.nativeStream());
    if(action_==Action::LocalRead)for(const auto& span:operands_.spans())execution.publish(span.root.get());
    execution.publish(operands_.controllerOwner());return true;
}
std::uint64_t PooledImmutableLocalHotPhaseStage::graphCaptureVariantSignature()const
{
    requireDeclarations();std::uint64_t h=1469598103934665603ULL;
    const auto mix=[&](std::uint64_t x){h=(h^x)*1099511628211ULL;};
    mix(1);mix(static_cast<std::uint64_t>(action_));mix(code_->codeIdentity());mix(storage_->captureIdentity());mix(operands_.declarationIdentity());
    return h?h:1;
}
bool PooledImmutableLocalHotPhaseStage::supportsBackend(ComputeBackendType backend)const{return localBackend(device(),backend);}
StageBufferContract PooledImmutableLocalHotPhaseStage::bufferContract()const
{
    requireDeclarations();auto contract=StageBufferContract::build();
    contract.addPreallocatedInOut(BufferId::MTP_GENERATION_CONTROL,"INT32");
    for(const auto& span:operands_.spans()){
        if(action_==Action::LocalRead)contract.addOutput(span.buffer,"FP32");
        else contract.addInput(span.buffer,span.root->native_type()==TensorType::INT8?"INT8":"INT32");
    }
    return contract;
}
StageDumpInfo PooledImmutableLocalHotPhaseStage::buildDumpInfoImpl()const
{
    StageDumpInfo result;result.addScalarInt("local_phase",static_cast<unsigned>(action_));
    if(action_==Action::LocalRead){
        const auto& output=operands_.spans().at(0);const auto* base=static_cast<const std::byte*>(output.root->raw_data());const auto columns=output.root->cols();
        result.outputs.push_back({"gathered_rows",base?base+output.offset:nullptr,output.bytes/(columns*sizeof(float)),columns,"FP32",sizeof(float),
            output.bytes,output.root.get(),{},output.offset});
        result.addScalarInt("native_row_bytes",storage_->geometry().extent().rowBytes());
    }
    return result;
}
PooledImmutableLocalHotReadyStage::PooledImmutableLocalHotReadyStage(
    std::shared_ptr<qwen4exp::PooledImmutableLocalHotPreparedStorage> storage,Frontier frontier)
    :IComputeStage(localDevice(storage)),storage_(std::move(storage)),frontier_(frontier),identity_(0)
{
    if(isGraphCaptureActive() || (frontier!=Frontier::DataReady && frontier!=Frontier::LocalReadComplete))
        throw std::invalid_argument("Local ready stage requires its actual nominal frontier before capture");
    identity_=graphCaptureVariantSignature();
}
bool PooledImmutableLocalHotReadyStage::prepareGraphLaunch(IDeviceContext* context,void* stream)
{
    exactLocalStream(stream);if(!context || context->deviceId()!=device())throw std::invalid_argument("Local ready capture changed participant");
    if(graphCaptureVariantSignature()!=identity_)throw std::logic_error("Local ready root identity changed");
    storage_->acquireSetup(stream);setGPUStream(stream);return true;
}
bool PooledImmutableLocalHotReadyStage::execute(IDeviceContext* context)
{
    if(!context || context->deviceId()!=device() || graphCaptureVariantSignature()!=identity_)
        throw std::logic_error("Local ready stage changed original participant/root");
    auto execution=gpuExecution();
    if(frontier_==Frontier::DataReady)storage_->enqueueReadyWait(execution.nativeStream());
    else storage_->enqueueLocalReadCompleteWait(execution.nativeStream());
    return true;
}
std::uint64_t PooledImmutableLocalHotReadyStage::graphCaptureVariantSignature()const
{const auto value=storage_->captureIdentity()^(0x4c48524741544501ULL+static_cast<unsigned>(frontier_));return value?value:1;}
bool PooledImmutableLocalHotReadyStage::supportsBackend(ComputeBackendType backend)const{return localBackend(device(),backend);}
StageDumpInfo PooledImmutableLocalHotReadyStage::buildDumpInfoImpl()const
{StageDumpInfo result;result.addScalarInt("local_ready_frontier",static_cast<unsigned>(frontier_));return result;}
} // namespace llaminar2
