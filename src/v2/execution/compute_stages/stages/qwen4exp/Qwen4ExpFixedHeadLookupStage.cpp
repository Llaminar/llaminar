/**
 * @file Qwen4ExpFixedHeadLookupStage.cpp
 * @brief Checked local fixed-head native kernels with ordinary arena preparation.
 *
 * The original DGO query authority is immutable through the complete graph
 * exchange. Cohort consensus creates only a transport extent; neither request
 * nor accepted/prior history state is replaced. Replay performs no allocation,
 * file work, host count read, event creation, collective or source preparation.
 */
#include "Qwen4ExpFixedHeadLookupStage.h"
#include "tensors/Tensors.h"
#include "transfer/TransferEngine.h"
#include <array>
#include <limits>
#include <stdexcept>
namespace llaminar2 {
namespace {
/** @brief Reject invalid construction before graph mutation or kernel submission. */
void require(bool condition,const char *message){if(!condition)throw std::invalid_argument(message);}
/** @return Canonical roots whose ordinary graph outputs own all operand allocation. */
std::array<TensorBase *,7> roots(const Qwen4ExpFixedHeadLookupStage::Frame &f) {
    const auto &s=f.storage;return {f.ids.tensor,f.output.tensor,s.localHeader.tensor,s.gatheredHeaders.tensor,
        s.transportRows.tensor,s.localNative.tensor,s.gatheredNative.tensor};
}
/** @brief Validate exact typed subview extent without resolving cold device pointers. */
template<TensorType Type> void binding(Qwen4ExpArenaBinding<Type> b,std::size_t bytes,std::size_t alignment) {
    require(b.tensor && b.buffer<BufferId::_COUNT && b.tensor->native_type()==Type &&
        b.tensor->transferStorageOwner()==b.tensor && !b.tensor->is_view() &&
        b.byteOffset%alignment==0 && b.byteOffset<=b.tensor->size_bytes() && bytes<=b.tensor->size_bytes()-b.byteOffset,
        "Fixed-head lookup requires exact canonical typed arena subviews");
}
}
std::shared_ptr<const Qwen4ExpFixedHeadLookupStage::Frame> Qwen4ExpFixedHeadLookupStage::frame(
    Qwen4ExpFixedHeadLookupStorage storage,Qwen4ExpIntBinding ids,Qwen4ExpFloatBinding output)
{
    require(storage.cohort && storage.table,"Fixed-head lookup requires actual cohort and admitted source owner");
    storage.cohort->requireNativeOwner(storage.table);
    const auto device=storage.table->device();
    const auto assignment=storage.cohort->requireParticipant(device,storage.query.participant());
    require(storage.query.cohort()==storage.cohort.get() && storage.query.device()==device &&
        storage.table->sourceRange()==assignment.source,"Fixed-head lookup source/query/cohort identity disagrees");
    storage.control.requireRequest(device,0);
    const auto rows=storage.query.rows().captureIdentity();
    require(rows.first_physical_row==0 && rows.repeats_per_row==1 && rows.count_capacity==rows.physical_rows,
        "Fixed-head lookup requires the complete original query prefix");
    const auto plan=qwen4exp::ImmutableFixedHeadLookupPlan::bind(storage.table->rangeView(),storage.cohort->policy(),
        storage.cohort->degree(),storage.query.participant());
    const auto multiply=qwen4exp::fixed_head_lookup_detail::product;
    const auto m=static_cast<std::size_t>(rows.physical_rows),p=static_cast<std::size_t>(plan.participants());
    const auto packet=multiply(m,multiply(plan.heads(),plan.table().rowBytes()));
    binding(ids,multiply(m,16*sizeof(std::int64_t)),alignof(std::int64_t));
    binding(output,multiply(m,multiply(16*sizeof(float),plan.table().rowElements())),alignof(float));
    binding(storage.localHeader,sizeof(qwen4exp::FixedHeadValidationHeader),alignof(std::int32_t));
    binding(storage.gatheredHeaders,multiply(p,sizeof(qwen4exp::FixedHeadValidationHeader)),alignof(std::int32_t));
    binding(storage.transportRows,sizeof(std::int32_t),alignof(std::int32_t));
    binding(storage.localNative,packet,alignof(std::int32_t));
    binding(storage.gatheredNative,multiply(p,packet),alignof(std::int32_t));
    const std::array tensors{ids.tensor,output.tensor,storage.localHeader.tensor,storage.gatheredHeaders.tensor,
        storage.transportRows.tensor,storage.localNative.tensor,storage.gatheredNative.tensor};
    const std::array buffers{ids.buffer,output.buffer,storage.localHeader.buffer,storage.gatheredHeaders.buffer,
        storage.transportRows.buffer,storage.localNative.buffer,storage.gatheredNative.buffer};
    for(std::size_t i=0;i<tensors.size();++i)for(std::size_t j=0;j<i;++j)
        require(tensors[i]!=tensors[j] && buffers[i]!=buffers[j],"Fixed-head lookup banks require distinct arena owners and roles");
    auto kernels=device.is_cuda()?qwen4exp::createCUDAImmutableFixedHeadLookupKernels(plan):
        qwen4exp::createROCmImmutableFixedHeadLookupKernels(plan);
    require(kernels!=nullptr,"Fixed-head lookup native decoder preparation failed");
    return std::make_shared<const Frame>(Frame{std::move(storage),ids,output,std::move(kernels)});
}
Qwen4ExpFixedHeadLookupStage::Qwen4ExpFixedHeadLookupStage(std::shared_ptr<const Frame> f,Operation operation,std::string name)
    :IComputeStage(f?f->storage.table->device():DeviceId::invalid()),frame_(std::move(f)),operation_(operation),name_(std::move(name)),
      prepared_(device(),frame_?roots(*frame_):std::array<TensorBase *,7>{})
{
    require(frame_ && !name_.empty() && (operation==Operation::ValidateLocal || operation==Operation::ValidateCohort ||
        operation==Operation::Pack || operation==Operation::Decode),"Fixed-head lookup needs its checked frame and exact operation");
}
template<class T,TensorType Type> T *Qwen4ExpFixedHeadLookupStage::pointer(Qwen4ExpArenaBinding<Type> b) const {
    auto *base=static_cast<std::byte *>(b.tensor->gpu_data_ptr());
    require(base,"Fixed-head lookup operand is not already prepared");
    auto *result=base+b.byteOffset;
    require(reinterpret_cast<std::uintptr_t>(result)%alignof(T)==0,"Fixed-head lookup native pointer is misaligned");
    return reinterpret_cast<T *>(result);
}
qwen4exp::ImmutableFixedHeadLookupArgs Qwen4ExpFixedHeadLookupStage::arguments() const {
    const auto &s=frame_->storage;const auto &plan=frame_->kernels->plan();
    const auto m=static_cast<std::size_t>(s.query.rows().physicalRows());
    const auto packet=qwen4exp::fixed_head_lookup_detail::product(m,
        qwen4exp::fixed_head_lookup_detail::product(plan.heads(),plan.table().rowBytes()));
    return {s.query.rows(),pointer<const std::int64_t>(frame_->ids),m*16,
        pointer<qwen4exp::FixedHeadValidationHeader>(s.localHeader),pointer<const qwen4exp::FixedHeadValidationHeader>(s.gatheredHeaders),
        static_cast<std::size_t>(plan.participants()),pointer<std::int32_t>(s.transportRows),pointer<std::byte>(s.localNative),packet,
        pointer<const std::byte>(s.gatheredNative),packet*static_cast<std::size_t>(plan.participants()),
        pointer<float>(frame_->output),m*16*static_cast<std::size_t>(plan.table().rowElements()),s.control.healthy(),s.control.error(),{s.control.row(),sampling_math::kDeviceGenerationControlCount}};
}
bool Qwen4ExpFixedHeadLookupStage::prepareGraphLaunch(IDeviceContext *context,void *stream) {
    frame_->storage.query.requireExecutable();
    require(context && context->deviceId()==device() && stream,"Fixed-head lookup preparation requires its exact context and stream");
    frame_->storage.control.requireRequest(device(),0);prepared_.seal();
    const auto identity=frame_->kernels->captureIdentity(arguments(),stream);
    if(capture_ && *capture_!=identity)throw std::logic_error("Fixed-head lookup prepared frame/stream changed");
    if(!read_)read_.emplace(frame_->storage.table->acquire(stream,TransferEngine::instance()));
    TransferEngine::instance().requireImmutableNativeDeviceInput(*read_,device(),stream);
    capture_=identity;return true;
}
bool Qwen4ExpFixedHeadLookupStage::execute(IDeviceContext *context) {
    frame_->storage.query.requireExecutable();
    require(context && context->deviceId()==device(),"Fixed-head lookup execution requires its exact participant");
    prepared_.validate();frame_->storage.control.requireRequest(device(),0);
    const auto execution=gpuExecution();const auto args=arguments();
    if(!read_ || !capture_ || *capture_!=frame_->kernels->captureIdentity(args,execution.nativeStream()))
        throw std::logic_error("Fixed-head lookup capture identity/storage/stream changed");
    TransferEngine::instance().requireImmutableNativeDeviceInput(*read_,device(),execution.nativeStream());
    const auto input=[&](auto b){execution.requirePreparedInput(b.tensor);};
    const auto output=[&](auto b){execution.requirePreparedOutput(b.tensor);};
    const auto &s=frame_->storage;
    execution.requirePreparedInput(s.control.owner());
    execution.requirePreparedOutput(s.control.owner());
    switch(operation_) {
    case Operation::ValidateLocal: input(frame_->ids);output(s.localHeader);output(s.transportRows);frame_->kernels->validateLocal(args,execution.nativeStream());break;
    case Operation::ValidateCohort: input(s.gatheredHeaders);output(s.transportRows);frame_->kernels->validateCohort(args,execution.nativeStream());break;
    case Operation::Pack: input(frame_->ids);input(s.transportRows);output(s.localNative);frame_->kernels->pack(args,execution.nativeStream());break;
    case Operation::Decode:
        // The native frame validator retains all packet/header/ID identities
        // through decode. Keep those real borrowed records live and published,
        // preventing ordinary scratch aliasing with the final output bank.
        input(frame_->ids);input(s.localHeader);input(s.gatheredHeaders);input(s.localNative);
        input(s.gatheredNative);input(s.transportRows);output(frame_->output);
        frame_->kernels->decode(args,execution.nativeStream());break;
    }
    execution.publish(s.control.owner());
    return true;
}
StageBufferContract Qwen4ExpFixedHeadLookupStage::bufferContract() const {
    const auto &s=frame_->storage;StageBufferContract c;
    c.addPreallocatedInOut(BufferId::MTP_GENERATION_CONTROL,"INT32");
    switch(operation_) {
    case Operation::ValidateLocal:return c.addInput(frame_->ids.buffer,"INT32").addOutput(s.localHeader.buffer,"INT32").addOutput(s.transportRows.buffer,"INT32");
    case Operation::ValidateCohort:return c.addInput(s.gatheredHeaders.buffer,"INT32").addOutput(s.transportRows.buffer,"INT32");
    case Operation::Pack:return c.addInput(frame_->ids.buffer,"INT32").addInput(s.transportRows.buffer,"INT32").addOutput(s.localNative.buffer,"INT32");
    case Operation::Decode:return c.addInput(frame_->ids.buffer,"INT32").addInput(s.localHeader.buffer,"INT32")
        .addInput(s.gatheredHeaders.buffer,"INT32").addInput(s.localNative.buffer,"INT32")
        .addInput(s.gatheredNative.buffer,"INT32").addInput(s.transportRows.buffer,"INT32").addOutput(frame_->output.buffer,"FP32");
    }
    throw std::logic_error("Unknown fixed-head operation");
}
bool Qwen4ExpFixedHeadLookupStage::supportsBackend(ComputeBackendType backend) const {
    return (device().is_cuda() && backend==ComputeBackendType::GPU_CUDA) || (device().is_rocm() && backend==ComputeBackendType::GPU_ROCM);
}
StageDumpInfo Qwen4ExpFixedHeadLookupStage::buildDumpInfoImpl() const {
    StageDumpInfo dump;if(operation_!=Operation::Decode)return dump;
    const auto &b=frame_->output;const auto *base=static_cast<const std::byte *>(b.tensor->raw_data());
    const auto m=static_cast<std::size_t>(frame_->storage.query.rows().physicalRows());
    const auto width=static_cast<std::size_t>(frame_->kernels->plan().table().rowElements())*16;
    dump.outputs.push_back({"gathered_rows",base?base+b.byteOffset:nullptr,m,width,"FP32",sizeof(float),m*width*sizeof(float),b.tensor,{},b.byteOffset});
    return dump;
}
std::uint64_t Qwen4ExpFixedHeadLookupStage::graphCaptureVariantSignature() const {
    prepared_.validateIfSealed();const auto &s=frame_->storage;
    const auto assignment=s.cohort->requireParticipant(device(),s.query.participant());
    if(s.table->sourceRange()!=assignment.source || s.table->rangeView()!=frame_->kernels->plan().table())
        throw std::logic_error("Fixed-head lookup native source changed after construction");
    s.control.requireRequest(device(),0);
    std::uint64_t value=s.query.captureIdentity();
    const auto mix=[&](std::uint64_t word){value^=word;value*=1099511628211ULL;};
    mix(1);mix(static_cast<std::uint64_t>(operation_));mix(reinterpret_cast<std::uintptr_t>(s.cohort.get()));
    mix(reinterpret_cast<std::uintptr_t>(s.table.get()));mix(reinterpret_cast<std::uintptr_t>(frame_->kernels.get()));
    mix(s.table->rangeView().storageGeneration());mix(s.control.captureIdentity());
    const auto bind=[&](const auto &b){mix(reinterpret_cast<std::uintptr_t>(b.tensor));mix(static_cast<std::uint64_t>(b.buffer));mix(b.byteOffset);};
    bind(frame_->ids);bind(frame_->output);bind(s.localHeader);bind(s.gatheredHeaders);bind(s.transportRows);bind(s.localNative);bind(s.gatheredNative);
    return value?value:1;
}
} // namespace llaminar2
