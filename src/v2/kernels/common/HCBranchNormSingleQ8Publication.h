/**
 * @file HCBranchNormSingleQ8Publication.h
 * @brief Original HC-down scratch leases and single-consumer norm publication authority.
 *
 * One original norm stage produces FP32 and the HC-down's existing SingleQ8
 * bank. The original down consumes that bank once before a later projection
 * may reuse the scratch prefix. No payload mirror, registry, new arena bytes
 * or host shadow of GPU execution state exists. Pending/consumed state describes
 * only ordered eager submission or native graph recording; replay uses its
 * retained norm-to-down DAG and never consults this host submission ticket.
 */
#pragma once
#include "backends/BackendManager.h"
#include "backends/ExplicitGPUStream.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/device/WorkspaceBufferLease.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/common/Q8ResidualCorrection.h"
#include "tensors/Tensors.h"
#include "transfer/TransferEngine.h"
#include "utils/DebugEnv.h"
#include <array>
#include <bit>
#include <cmath>
#include <climits>
#include <memory>
#include <stdexcept>
#include <vector>
namespace llaminar2 {
class Qwen4ExpPrimitiveStage;
/** @brief Cold checked original operands, banks and exact submission provenance. */
class HCBranchNormSingleQ8Publication final {
 enum class Submission {None,Eager,Capture};
 DeviceId device_;int hidden_;float epsilon_;
 std::array<const TensorBase*,4> tensors_;
 DeviceWorkspaceManager* workspace_=nullptr;
 std::uint64_t workspaceId_=0,generation_=0;
 IBackend* backend_=nullptr;void* stream_=nullptr;
 std::array<const void*,4> pointers_{};
 std::array<std::size_t,4> bytes_{};
 std::array<std::vector<std::size_t>,4> shapes_;
 std::array<std::shared_ptr<const CapturedDeviceStorageLease>,4> operands_;
 std::array<std::shared_ptr<const WorkspaceBufferLease>,4> banks_;
 mutable Submission pending_=Submission::None;
 /** @brief Validate the source declarations without observing GPU values. */
 void declarations()const {
  for(const auto* t:tensors_)if(!t||t->native_type()!=TensorType::FP32||t->transferStorageOwner()!=t)
   throw std::logic_error("HC norm SingleQ8 lost its canonical FP32 root declaration");
 }
public:
 /** @brief Declare one original ordinary M1 norm before workspace admission. */
 HCBranchNormSingleQ8Publication(DeviceId device,int hidden,float epsilon,const TensorBase& input,
     const TensorBase& gain,const TensorBase& normalized,const TensorBase& down)
  :device_(device),hidden_(hidden),epsilon_(epsilon),tensors_{&input,&gain,&normalized,&down} {
  if(!device.is_gpu()||hidden<=0||hidden%32||hidden>INT_MAX/4||!std::isfinite(epsilon)||epsilon<=0||
     &input==&normalized||&gain==&normalized||&input==&gain||
     &down==&input||&down==&gain||&down==&normalized||down.isMapped()||!down.numel()||
     input.numel()<std::size_t(4)*hidden||gain.numel()<std::size_t(4)*hidden||normalized.numel()<std::size_t(4)*hidden)
   throw std::invalid_argument("HC norm SingleQ8 requires distinct original GPU M1 FP32 branches and complete K32 gains");
  declarations();
 }
 /** @brief Bind the existing consumer owner; no memory is allocated or rebound. */
 void bindConsumerWorkspace(DeviceWorkspaceManager* workspace) {
  if(!workspace||workspace->device()!=device_||(workspace_&&workspace_!=workspace))
   throw std::logic_error("HC norm SingleQ8 consumer workspace changed");
  workspace_=workspace;
  if(prepared())validate(stream_);
 }
 /** @brief Seal original allocations and named banks once, outside capture. */
 void prepare(void* stream) {
  (void)ExplicitGPUStream(stream);declarations();
  if(prepared()){validate(stream);return;}
  if(isGraphCaptureActive()||!workspace_||!workspace_->isAllocated())
   throw std::logic_error("HC norm SingleQ8 requires its cold admitted consumer workspace");
  auto* backend=getBackendFor(device_);const auto generation=backend?backend->deviceRuntimeGeneration(device_.ordinal):0;
  if(!generation)throw std::logic_error("HC norm SingleQ8 requires its original backend runtime generation");
  std::array<std::shared_ptr<const CapturedDeviceStorageLease>,4> operands;
  std::array<std::vector<std::size_t>,4> shapes;
  for(std::size_t i=0;i<tensors_.size();++i) {
   const auto& t=*tensors_[i];
   if(t.current_device()!=device_||!t.gpu_data_ptr())throw std::logic_error("HC norm SingleQ8 original root is not prebound");
   shapes[i]=t.shape();operands[i]=TransferEngine{}.retainCapturedDeviceStorage(t,device_,t.size_bytes());
  }
  const std::array<const char*,4> names{GemmWorkspaceBuffers::QUANT_A,GemmWorkspaceBuffers::SCALES_A_BLOCKWISE,
   device_.is_rocm()?GemmWorkspaceBuffers::SCALES_A:GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS,
   device_.is_rocm()?GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL:GemmWorkspaceBuffers::ACC_INT32};
  const std::array<std::size_t,4> needed{std::size_t(4)*hidden_,std::size_t(4)*hidden_/32*sizeof(float),sizeof(float),sizeof(float)};
  std::array<std::shared_ptr<const WorkspaceBufferLease>,4> banks;
  for(std::size_t i=0;i<names.size();++i) {
   if(!workspace_->getBuffer(names[i])||workspace_->getBufferSize(names[i])<needed[i])
    throw std::logic_error("HC norm SingleQ8 lacks the original named consumer bank");
   banks[i]=workspace_->retainBuffer(names[i],0,workspace_->getBufferSize(names[i]));
  }
  const std::array writes{Q8ResidualStorageSpan{tensors_[2]->gpu_data_ptr(),tensors_[2]->size_bytes(),4},
   Q8ResidualStorageSpan{banks[0]->data(),banks[0]->sizeBytes(),16},Q8ResidualStorageSpan{banks[1]->data(),banks[1]->sizeBytes(),4},
   Q8ResidualStorageSpan{tensors_[3]->gpu_data_ptr(),tensors_[3]->size_bytes(),4},
   Q8ResidualStorageSpan{banks[device_.is_rocm()?3:2]->data(),banks[device_.is_rocm()?3:2]->sizeBytes(),4}};
  const std::array reads{Q8ResidualStorageSpan{tensors_[0]->gpu_data_ptr(),tensors_[0]->size_bytes(),4},
   Q8ResidualStorageSpan{tensors_[1]->gpu_data_ptr(),tensors_[1]->size_bytes(),4}};
  validateQ8ResidualStorage(writes,reads);
  for(std::size_t i=0;i<tensors_.size();++i){pointers_[i]=tensors_[i]->gpu_data_ptr();bytes_[i]=tensors_[i]->size_bytes();}
  operands_=std::move(operands);shapes_=std::move(shapes);banks_=std::move(banks);
  backend_=backend;generation_=generation;workspaceId_=workspace_->id();stream_=stream;validate(stream);
 }
 /** @brief Validate all physical authority without allocation, GPU query or payload observation. */
 void validate(void* stream)const {
  declarations();(void)ExplicitGPUStream(stream);
  if(device_.is_rocm()&&debugEnv().rocm.blockwise_quant_variant!=0&&debugEnv().rocm.blockwise_quant_variant!=4)
   throw std::logic_error("HC norm SingleQ8 requires the original ordinary M1 V4 quantizer policy");
  if(!prepared()||stream!=stream_||!workspace_||!workspace_->isAllocated()||workspace_->id()!=workspaceId_||
     workspace_->device()!=device_||getBackendFor(device_)!=backend_||backend_->deviceRuntimeGeneration(device_.ordinal)!=generation_)
   throw std::logic_error("HC norm SingleQ8 original workspace/stream/runtime changed");
  const std::array<const char*,4> names{GemmWorkspaceBuffers::QUANT_A,GemmWorkspaceBuffers::SCALES_A_BLOCKWISE,
   device_.is_rocm()?GemmWorkspaceBuffers::SCALES_A:GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS,
   device_.is_rocm()?GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL:GemmWorkspaceBuffers::ACC_INT32};
  for(std::size_t i=0;i<banks_.size();++i)if(!banks_[i]||banks_[i]->device()!=device_||
      workspace_->getBuffer(names[i])!=banks_[i]->data()||workspace_->getBufferSize(names[i])!=banks_[i]->sizeBytes())
   throw std::logic_error("HC norm SingleQ8 original named bank changed");
  for(std::size_t i=0;i<tensors_.size();++i) {
   const auto& t=*tensors_[i];
   if(t.gpu_data_ptr()!=pointers_[i]||t.size_bytes()!=bytes_[i]||t.shape()!=shapes_[i])
    throw std::logic_error("HC norm SingleQ8 original pointer/shape changed");
   TransferEngine{}.requireCapturedDeviceStorage(*operands_[i],t,device_,bytes_[i]);
  }
 }
 /** @brief Authenticate producer declarations as well as the exact prepared bindings. */
 void requireProducer(const TensorBase& input,const TensorBase& gain,const TensorBase& normalized,
     int hidden,float epsilon,void* stream)const {
  if(pending_!=Submission::None)throw std::logic_error("HC norm SingleQ8 prior original consumer was not submitted");
  if(&input!=tensors_[0]||&gain!=tensors_[1]||&normalized!=tensors_[2]||hidden!=hidden_||
     std::bit_cast<std::uint32_t>(epsilon)!=std::bit_cast<std::uint32_t>(epsilon_))
   throw std::logic_error("HC norm SingleQ8 producer declaration changed");
  validate(stream);
 }
private:
 friend class Qwen4ExpPrimitiveStage;
 /** @brief Issue one successful producer submission, without claiming device execution. */
 void recordProducer() {
  if(!prepared()||pending_!=Submission::None)throw std::logic_error("HC norm SingleQ8 producer has no empty prepared submission ticket");
  pending_=isGraphCaptureActive()?Submission::Capture:Submission::Eager;
 }
public:
 /** @brief Validate the unchanged consumer declaration independently of submission state. */
 void validateConsumer(const TensorBase& normalized,const TensorBase& down,DeviceWorkspaceManager* workspace,void* stream=nullptr)const {
  if(&normalized!=tensors_[2]||&down!=tensors_[3]||workspace!=workspace_)throw std::logic_error("HC norm SingleQ8 consumer declaration changed");
  validateIfPrepared(stream);
 }
 /** @brief Require the one original consumer on the producer's exact source/workspace/stream. */
 void requireRead(const TensorBase& normalized,const TensorBase& down,DeviceWorkspaceManager* workspace,void* stream)const {
  if(&normalized!=tensors_[2]||&down!=tensors_[3]||workspace!=workspace_||pending_==Submission::None||
      pending_!=(isGraphCaptureActive()?Submission::Capture:Submission::Eager))
   throw std::logic_error("HC norm SingleQ8 lacks its original pending input publication");
  validate(stream);
 }
 /** @brief Retire the recording/eager ticket after successful original down submission. */
 void consumeRead()const {if(pending_==Submission::None)throw std::logic_error("HC norm SingleQ8 original input was already consumed");pending_=Submission::None;}
 /** @brief Validate prepared capture identity while reset may temporarily clear the host stage stream. */
 void validateIfPrepared(void* currentStream=nullptr)const {
  declarations();if(prepared())validate(currentStream?currentStream:stream_);
 }
 /** @return Whether cold preparation installed complete original allocation leases. */
 bool prepared()const noexcept{return stream_!=nullptr;}
 /** @return Original writable producer code bank; never a new allocation. */
 std::int8_t* codes()const {if(!prepared())throw std::logic_error("HC norm SingleQ8 codes were not prepared");return static_cast<std::int8_t*>(banks_[0]->data());}
 /** @return Original FP32 scale bank. */
 float* scales()const {if(!prepared())throw std::logic_error("HC norm SingleQ8 scales were not prepared");return static_cast<float*>(banks_[1]->data());}
 /** @return Original complete normalized width. */
 int width()const{return 4*hidden_;}
 /** @return Original participant. */
 DeviceId device()const{return device_;}
};
}
