/**
 * @file HCDownOutputPublication.h
 * @brief Sealed original HC reducer-output publication and required up consumer.
 *
 * This record owns no bytes and no live activation shadow. It retains the
 * existing raw/activated tensors and four named up-workspace allocations through
 * captured replay. Cold declaration identity is separate from physical seals.
 */
#pragma once
#include "HCDownOutputPublicationArgs.h"
#include "Q8ResidualCorrection.h"
#include "kernels/rocm/gemm/ROCmNativeQuantisedSerialGeometry.h"
#include "backends/BackendManager.h"
#include "backends/ExplicitGPUStream.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "tensors/Tensors.h"
#include "transfer/TransferEngine.h"
#include <array>
#include <memory>
#include <limits>
#include <optional>
#include <vector>
namespace llaminar2 {
/** @brief Exact original producer/consumer contract; a new source cannot adopt old banks. */
class HCDownOutputPublication final {
 struct Region {const char* name;void* pointer;std::size_t bytes;std::shared_ptr<const WorkspaceBufferLease> lease;};
 struct Seal {
  DeviceWorkspaceManager* workspace;std::uint64_t id;void* stream;IBackend* backend;std::uint64_t runtime;
  std::array<const void*,2> pointers;std::array<std::size_t,2> bytes;
  std::array<std::vector<std::size_t>,2> shapes;
  std::array<std::shared_ptr<const CapturedDeviceStorageLease>,2> tensors;
  std::array<Region,6> banks;
  std::optional<rocm::NativeQ8SerialGeometryAdmission> producer{}; ///< Cold exact symbol/resources; no GPU state mirror.
 };
 DeviceId device_;TensorBase* raw_;TensorBase* activated_;int columns_;DeviceRowRange rows_;
 const HCDownProducerProgram producer_program_; ///< Immutable before graph declaration/cold native admission.
 std::optional<Seal> seal_;
 mutable bool published_=false,capture_recorded_=false;
public:
 /** @brief Freeze original row/tensor roles without allocating or reading data.
  * @param device Original ROCm participant. @param raw Original raw down output.
  * @param activated Original /4-SiLU output, allowing exact raw alias.
  * @param columns Exact positive bottleneck, a multiple of32. @param rows Original publisher count authority.
  * @param producer_program Explicit original/qualified lookahead producer; ordinary construction retains Original.
  */
 HCDownOutputPublication(DeviceId device,TensorBase& raw,TensorBase& activated,int columns,DeviceRowRange rows,
  HCDownProducerProgram producer_program=HCDownProducerProgram::Original)
  :device_(device),raw_(&raw),activated_(&activated),columns_(columns),rows_(rows),producer_program_(producer_program) {
  const auto r=rows.captureIdentity();
  if((producer_program!=HCDownProducerProgram::Original&&producer_program!=HCDownProducerProgram::TwoK32Lookahead)||
     !device.is_rocm()||columns<64||columns>512||columns%32||r.physical_rows!=1||r.count_capacity!=1||
     r.first_physical_row||r.repeats_per_row!=1||raw.native_type()!=TensorType::FP32||activated.native_type()!=TensorType::FP32||
     raw.numel()<std::size_t(columns)||activated.numel()<std::size_t(columns))
   throw std::invalid_argument("HC reducer publication requires original ROCm M1 complete32-column FP32 roles");
 }
 /** @brief Retain already admitted original tensors/banks once, outside capture.
  * @param workspace Existing original up workspace owner. @param stream Exact producer/consumer stream.
  */
 void prepare(DeviceWorkspaceManager& workspace,void* stream) {
  if(isGraphCaptureActive()||workspace.device()!=device_||!workspace.isAllocated()||reinterpret_cast<std::uintptr_t>(stream)<=2)
   throw std::logic_error("HC reducer publication needs cold admitted original workspace and stream");
  if(seal_){validate(&workspace,stream);return;}
  auto* backend=getBackendFor(device_);const auto runtime=backend?backend->deviceRuntimeGeneration(device_.ordinal):0;
  if(!runtime)throw std::logic_error("HC reducer publication has no original native runtime");
  Seal result{&workspace,workspace.id(),stream,backend,runtime,{},{},{},{},{}};
  const std::array<TensorBase*,2> tensors{raw_,activated_};
  for(std::size_t i=0;i<2;++i) {
   auto* t=tensors[i];if(t->current_device()!=device_||!t->gpu_data_ptr())
    throw std::logic_error("HC reducer publication operands must be prebound");
   result.pointers[i]=t->gpu_data_ptr();result.bytes[i]=t->size_bytes();result.shapes[i]=t->shape();
   result.tensors[i]=TransferEngine{}.retainCapturedDeviceStorage(*t,device_,t->size_bytes());
  }
  const std::array<const char*,6> names{kGPUProjectionPrimaryQ8.c_str(),kGPUProjectionResidualQ8.c_str(),
    kGPUProjectionPrimaryQ8Scales.c_str(),kGPUProjectionResidualQ8Scales.c_str(),
    GemmWorkspaceBuffers::SCALES_A,GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL};
  for(std::size_t i=0;i<6;++i) {
   const auto bytes=workspace.getBufferSize(names[i]);auto* pointer=workspace.getBuffer(names[i]);
   const auto needed=i<2?std::size_t(columns_):i<4?std::size_t(columns_/32)*sizeof(float):
     i==4?sizeof(float):std::size_t(columns_)*sizeof(float);
   if(!pointer||bytes<needed)throw std::logic_error("HC reducer publication has incomplete original up banks");
   result.banks[i]={names[i],pointer,bytes,workspace.retainBuffer(names[i],0,bytes)};
  }
  const auto overlap=[](const void* a,std::size_t an,const void* b,std::size_t bn) {
   const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);
   if(an>std::numeric_limits<std::uintptr_t>::max()-x||bn>std::numeric_limits<std::uintptr_t>::max()-y)
    throw std::invalid_argument("HC reducer publication interval overflows");
   return x<y+bn&&y<x+an;
  };
  if(result.pointers[0]!=result.pointers[1]&&overlap(result.pointers[0],columns_*sizeof(float),result.pointers[1],columns_*sizeof(float)))
   throw std::invalid_argument("HC raw/activated roles only allow exact alias or disjoint storage");
  for(std::size_t i=0;i<6;++i) {
   for(std::size_t t=0;t<2;++t)if(overlap(result.banks[i].pointer,result.banks[i].bytes,result.pointers[t],result.bytes[t]))
    throw std::invalid_argument("HC reducer Q8 bank aliases original public output");
   for(std::size_t j=0;j<i;++j)if(overlap(result.banks[i].pointer,result.banks[i].bytes,result.banks[j].pointer,result.banks[j].bytes))
    throw std::invalid_argument("HC reducer Q8 banks overlap");
  }
  if(producer_program_==HCDownProducerProgram::TwoK32Lookahead) {
#ifdef HAVE_ROCM
   // Existing cold admission owns native function/resource inspection. The
   // model's ordinary HC projection is HasWork=false; an explicit original
   // work scope, when present in a diagnostic, selects/seals its true program.
   result.producer.emplace(rocm::admitNativeQ8SerialGeometry(rocm::NativeQ8SerialGeometry::TN128CPT1,
       device_.ordinal,rocm::NativeQ8SerialProducerLoads::TwoK32Lookahead));
#else
   throw std::logic_error("HC lookahead publication requires the compiled ROCm backend");
#endif
  }
  seal_.emplace(std::move(result));validate(&workspace,stream);
 }
 /** @brief Authenticate full storage/shape/runtime/bank identity before capture/replay.
  * @param workspace Exact original owner. @param stream Exact currently bound stream.
  */
 void validate(DeviceWorkspaceManager* workspace,void* stream) const {
  if(!seal_||!workspace||workspace!=seal_->workspace||workspace->id()!=seal_->id||!workspace->isAllocated()||
     stream!=seal_->stream||reinterpret_cast<std::uintptr_t>(stream)<=2||
     !seal_->backend||seal_->backend->deviceRuntimeGeneration(device_.ordinal)!=seal_->runtime)
   throw std::logic_error("HC reducer publication lost original workspace/stream/runtime");
  if(producer_program_==HCDownProducerProgram::TwoK32Lookahead&&
     (!seal_->producer||seal_->producer->deviceOrdinal()!=device_.ordinal||
      seal_->producer->geometry()!=rocm::NativeQ8SerialGeometry::TN128CPT1||
      seal_->producer->producerLoads()!=rocm::NativeQ8SerialProducerLoads::TwoK32Lookahead||
      !seal_->producer->producerResources().function||!seal_->producer->reducerResources().function))
   throw std::logic_error("HC lookahead publication lost its exact cold native program");
  const std::array<TensorBase*,2> tensors{raw_,activated_};
  for(std::size_t i=0;i<2;++i) {
   auto* t=tensors[i];if(t->gpu_data_ptr()!=seal_->pointers[i]||t->size_bytes()!=seal_->bytes[i]||
    t->shape()!=seal_->shapes[i]||t->native_type()!=TensorType::FP32)
    throw std::logic_error("HC reducer publication original tensor binding changed");
   TransferEngine{}.requireCapturedDeviceStorage(*seal_->tensors[i],*t,device_,seal_->bytes[i]);
  }
  for(const auto& b:seal_->banks)if(workspace->getBuffer(b.name)!=b.pointer||workspace->getBufferSize(b.name)!=b.bytes||
    b.lease->data()!=b.pointer)throw std::logic_error("HC reducer publication original named bank changed");
 }
 /** @brief Validate a known prepared binding without sampling live GPU state. */
 void validatePrepared() const {if(seal_)validate(seal_->workspace,seal_->stream);}
 /** @brief Resolve immutable device destinations after exact producer authentication.
  * @param output Original raw down output. @param workspace Existing original owner. @param stream Exact current stream.
  * @return Original public banks and publisher rows.
  */
 GPUHCDownOutputView deviceView(TensorBase* output,DeviceWorkspaceManager* workspace,void* stream) const {
  if(output!=raw_)throw std::logic_error("HC reducer publication raw output was substituted");validate(workspace,stream);
  TransferEngine::requireDeviceOutput(activated_,device_,stream);
  return {rows_,columns_,static_cast<float*>(activated_->gpu_data_ptr()),
   static_cast<signed char*>(seal_->banks[0].pointer),static_cast<signed char*>(seal_->banks[1].pointer),
   static_cast<float*>(seal_->banks[2].pointer),static_cast<float*>(seal_->banks[3].pointer)};
 }
 /** @return Immutable declaration identity; no cold/prepared hash drift. */
 HCDownProducerProgram producerProgram() const noexcept {return producer_program_;}
 /** @brief Borrow the exact cold admitted producer; Original needs no new resource query.
  * @return Sealed lookahead program or null for the unchanged ordinary producer.
  * @throws std::logic_error If an explicitly selected lookahead program was not prepared.
  */
 const rocm::NativeQ8SerialGeometryAdmission* preparedProducer() const {
  if(producer_program_==HCDownProducerProgram::Original)return nullptr;
  if(!seal_||!seal_->producer)throw std::logic_error("HC lookahead producer requires original cold preparation");
  return &*seal_->producer;
 }
 /** @brief Record only ordered eager/capture provenance after accepted producer submission. */
 void recordPublished() const {
  if(published_)throw std::logic_error("HC reducer publication previous consumer did not retire");
  published_=true;capture_recorded_=isGraphCaptureActive();
 }
 /** @brief Required consumer cannot quantize again or consume a merely recorded eager payload.
  * @param raw Original source object. @param activated Original public activation.
  * @param rows Same publisher row authority. @param workspace Original up owner. @param stream Exact current stream.
  */
 void requirePublished(const TensorBase& raw,const TensorBase& activated,DeviceRowRange rows,
  DeviceWorkspaceManager* workspace,void* stream) const {
  validate(workspace,stream);
  if(&raw!=raw_||&activated!=activated_||rows.captureIdentity()!=rows_.captureIdentity()||!published_||
     (capture_recorded_&&!isGraphCaptureActive()))
   throw std::logic_error("Required HC reducer payload/source/rows/provenance is absent");
 }
 /** @brief Retire one successful original up submission; replay topology owns subsequent use. */
 void consume() const {if(!published_)throw std::logic_error("HC reducer publication was already consumed");published_=false;}
 /** @return Stable declaration source object, never a last-input cache. */
 TensorBase* raw() const noexcept{return raw_;}
 /** @return Original public activated object. */
 TensorBase* activated() const noexcept{return activated_;}
 /** @return Same immutable publisher row authority. */
 DeviceRowRange rows() const noexcept{return rows_;}
};
}
