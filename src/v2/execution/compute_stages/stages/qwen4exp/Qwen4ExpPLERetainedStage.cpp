/**
 * @file Qwen4ExpPLERetainedStage.cpp
 * @brief Prepared-arena retention and immutable PLE accepted-prefix publication.
 *
 * Constructor descriptors own all extent validation and contracts. Execute and
 * publication neither allocate nor initiate coherence transfers. The complete CPU
 * decision vector is validated before returning any Ready native plan; the existing
 * publisher still validates every other stage and destination before committing.
 */
#include "Qwen4ExpPLERetainedStage.h"
#include "../../../../tensors/Tensors.h"
#include "../../../../transfer/TransferEngine.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
namespace llaminar2 {
namespace {
/** @brief Reject malformed immutable graph geometry at setup. */
void require(bool b){if(!b)throw std::invalid_argument("Invalid PLE retained stage binding/geometry");}
}
template<TensorType Type>void Qwen4ExpPLERetainedStage::add(Qwen4ExpArenaBinding<Type>b,std::size_t bytes,bool write,const char*name){
 require(b.tensor&&b.tensor->native_type()==Type&&b.buffer<BufferId::_COUNT&&bytes&&bytes%4==0&&b.byteOffset%4==0&&b.byteOffset<=b.tensor->size_bytes()&&bytes<=b.tensor->size_bytes()-b.byteOffset);
 operands_.push_back({b.tensor,b.buffer,Type,b.byteOffset,bytes,write,name});
}
template<class T,TensorType Type>T*Qwen4ExpPLERetainedStage::pointer(Qwen4ExpArenaBinding<Type>b)const{
 auto*base=device().is_gpu()?b.tensor->gpu_data_ptr():b.tensor->raw_mutable_data();
 if(!base)throw std::logic_error("PLE retained stage storage is not already prepared");
 auto*p=static_cast<std::byte*>(base)+b.byteOffset;if(reinterpret_cast<std::uintptr_t>(p)%alignof(T))throw std::logic_error("PLE retained address misaligned");return reinterpret_cast<T*>(p);
}
Qwen4ExpPLERetainedStage::Qwen4ExpPLERetainedStage(Params p):IComputeStage(p.device_id),params_(std::move(p)),operations_(qwen4exp::createPLETransactionOperations(device())){
 require((params_.operation==Operation::SnapshotPrior||params_.operation==Operation::RetainCandidates)&&params_.requests>0&&params_.requestIndex>=0&&params_.requestIndex<params_.requests&&params_.rowCapacity>0&&params_.channels>0&&params_.channels<=(std::numeric_limits<int>::max()-2)/9&&params_.verifierOrigins.validForRequests(params_.requests));
 (void)DeviceRequestRowRanges::fullyActiveRequests(params_.requests,params_.rowCapacity);
 const auto c=std::size_t(params_.channels),m=std::size_t(params_.rowCapacity);require(c<=std::numeric_limits<std::size_t>::max()/36&&m<=std::numeric_limits<std::size_t>::max()/4/c);
 const bool snapshot=params_.operation==Operation::SnapshotPrior;
 if (!snapshot)
 {
     acceptedPublicationOutputs_ = {{
         {params_.acceptedTokens.buffer, BufferAccess::WRITE, "INT32", false},
         {params_.acceptedConvolution.buffer, BufferAccess::WRITE, "FP32", false}}};
 }

 add(params_.transactionLiveCounts,std::size_t(params_.requests)*4,snapshot,"transaction_live_counts");
 add(params_.priorTokens,8,snapshot,"transaction_prior_tokens");add(params_.priorConvolution,c*36,snapshot,"transaction_prior_convolution");
 if(snapshot){if(params_.liveCounts)std::visit([&](const auto&binding){using B=std::decay_t<decltype(binding)>;if constexpr(std::is_same_v<B,Qwen4ExpIntBinding>)add(binding,std::size_t(params_.requests)*4,false,"canonical_live_counts");else require(binding.owner==device()&&binding.data&&binding.capacity>=std::size_t(params_.requests)&&reinterpret_cast<std::uintptr_t>(binding.data)%4==0&&binding.capacity<=(std::numeric_limits<std::uintptr_t>::max()-reinterpret_cast<std::uintptr_t>(binding.data))/4);},params_.liveCounts->value);add(params_.acceptedTokens,8,false,"accepted_tokens");add(params_.acceptedConvolution,c*36,false,"accepted_convolution");}
 else {std::visit([&](const auto&binding){using B=std::decay_t<decltype(binding)>;if constexpr(std::is_same_v<B,Qwen4ExpIntBinding>)add(binding,m*4,false,"token_candidates");else require(binding.owner==device()&&binding.data&&binding.capacity>=m&&reinterpret_cast<std::uintptr_t>(binding.data)%4==0&&binding.capacity<=(std::numeric_limits<std::uintptr_t>::max()-reinterpret_cast<std::uintptr_t>(binding.data))/4);},params_.tokens.value);add(params_.normalized,m*c*4,false,"normalized_candidates");add(params_.retainedTokens,m*4,true,"retained_tokens");add(params_.retainedNormalized,m*c*4,true,"retained_normalized");add(params_.acceptedTokens,8,true,"accepted_tokens");add(params_.acceptedConvolution,c*36,true,"accepted_convolution");}
 require(params_.copiedBytes.byteOffset%8==0);add(params_.copiedBytes,8,true,"copied_payload_bytes");
 for(std::size_t i=0;i<operands_.size();++i)for(std::size_t j=i+1;j<operands_.size();++j){const auto&a=operands_[i];const auto&b=operands_[j];if((a.write||b.write)&&a.tensor==b.tensor)require(a.offset+a.bytes<=b.offset||b.offset+b.bytes<=a.offset);}
 std::vector<TensorBase*>roots;roots.reserve(operands_.size());for(const auto&o:operands_)roots.push_back(o.tensor);storage_.emplace(device(),roots);
}
const std::int32_t *Qwen4ExpPLERetainedStage::intInput(const Qwen4ExpIntInput&input)const{return std::visit([&](const auto&binding)->const std::int32_t*{using B=std::decay_t<decltype(binding)>;if constexpr(std::is_same_v<B,Qwen4ExpIntBinding>)return pointer<const std::int32_t>(binding);else return binding.data;},input.value);}
const std::int32_t *Qwen4ExpPLERetainedStage::tokenPointer()const{return intInput(params_.tokens);}
DeviceRequestRowRanges Qwen4ExpPLERetainedStage::transactionRows()const{return DeviceRequestRowRanges::deviceCounted(params_.requests,params_.rowCapacity,pointer<const int>(params_.transactionLiveCounts));}
qwen4exp::PLEPublishAccepted Qwen4ExpPLERetainedStage::publication(const int*rows,int stride)const{
 storage_->validateIfSealed();
 return {AcceptedRequestPrefixes::fromFlatRows(transactionRows(),rows,params_.verifierOrigins,stride),params_.requestIndex,params_.channels,pointer<const int>(params_.priorTokens),pointer<const float>(params_.priorConvolution),pointer<const int>(params_.retainedTokens),pointer<const float>(params_.retainedNormalized),pointer<int>(params_.acceptedTokens),pointer<float>(params_.acceptedConvolution)};
}
bool Qwen4ExpPLERetainedStage::prepareGraphLaunch(IDeviceContext*context,void*stream){
 require(context&&context->deviceId()==device()&&(!device().is_gpu()||stream));storage_->seal();return true;
}
bool Qwen4ExpPLERetainedStage::execute(IDeviceContext*context){
 require(context&&context->deviceId()==device());if(!device().is_gpu())storage_->seal();storage_->validate();void*stream=nullptr;
 if(device().is_gpu()){const auto execution=gpuExecution();stream=gpuStream();for(const auto&o:operands_)if(o.write)execution.requirePreparedOutput(o.tensor);else execution.requirePreparedInput(o.tensor);}
 if(params_.operation==Operation::SnapshotPrior)operations_.snapshot({(params_.liveCounts?DeviceRequestRowRanges::deviceCounted(params_.requests,params_.rowCapacity,intInput(*params_.liveCounts)):DeviceRequestRowRanges::fullyActiveRequests(params_.requests,params_.rowCapacity)),params_.requestIndex,params_.channels,pointer<const int>(params_.acceptedTokens),pointer<const float>(params_.acceptedConvolution),pointer<int>(params_.priorTokens),pointer<float>(params_.priorConvolution),pointer<int>(params_.transactionLiveCounts),pointer<std::uint64_t>(params_.copiedBytes)},stream);
 else operations_.retain({transactionRows(),params_.requestIndex,params_.channels,tokenPointer(),pointer<const float>(params_.normalized),pointer<int>(params_.retainedTokens),pointer<float>(params_.retainedNormalized),pointer<std::uint64_t>(params_.copiedBytes)},stream);
 return true;
}
CPUVerifierStateRestorePlan Qwen4ExpPLERetainedStage::planCPUVerifierStateRestoreRow(int row){
 if(device().is_gpu()||!hasVerifierStateCapture()||params_.requests!=1)return {.status=CPUVerifierStateRestorePlanStatus::Invalid};
 const auto p=publication(&row,1);qwen4exp::validatePLEPublication(p);if(!qwen4exp::validPLEDecisions(p.acceptance))return {.status=CPUVerifierStateRestorePlanStatus::Invalid};
 writes_={CPUVerifierStateRestoreDestination{p.acceptedTokens,8},CPUVerifierStateRestoreDestination{p.acceptedConvolution,std::size_t(params_.channels)*36}};
 return {.status=CPUVerifierStateRestorePlanStatus::Ready,.kind=CPUVerifierStateRestorePlanKind::NativeAcceptedPrefixCommit,.native_destinations=p.acceptance.candidates().lengthOwner()[0]?std::span<const CPUVerifierStateRestoreDestination>{writes_}:std::span<const CPUVerifierStateRestoreDestination>{}};
}
bool Qwen4ExpPLERetainedStage::planCPUVerifierStateRestoreRows(const int*rows,int requests,std::vector<CPUVerifierStateRestorePlan>&plans){
 if(device().is_gpu()||!hasVerifierStateCapture()||!rows||requests!=params_.requests)return false;
 const auto p=publication(rows,1);qwen4exp::validatePLEPublication(p);if(!qwen4exp::validPLEDecisions(p.acceptance))return false;
 const int live=p.acceptance.candidates().lengthOwner()[params_.requestIndex];writes_={CPUVerifierStateRestoreDestination{p.acceptedTokens,8},CPUVerifierStateRestoreDestination{p.acceptedConvolution,std::size_t(params_.channels)*36}};
 plans.push_back({.status=CPUVerifierStateRestorePlanStatus::Ready,.kind=CPUVerifierStateRestorePlanKind::NativeAcceptedPrefixCommit,.native_destinations=live?std::span<const CPUVerifierStateRestoreDestination>{writes_}:std::span<const CPUVerifierStateRestoreDestination>{}});return true;
}
bool Qwen4ExpPLERetainedStage::publish(const int*rows,int requests,int stride,void*stream){
 if(!hasVerifierStateCapture()||!rows||requests!=params_.requests||stride<=0||(device().is_gpu()?!stream:stream!=nullptr))return false;
 storage_->validate();
 std::optional<StageGPUExecution> execution;
 if(device().is_gpu()){
  // The generic accepted-prefix hook supplies this invocation's producer
  // stream; it need not equal the earlier verifier recording stream.
  execution=gpuExecutionForVerifierPublication(stream);stream=execution->nativeStream();
  for(auto*t:{params_.transactionLiveCounts.tensor,params_.priorTokens.tensor,params_.priorConvolution.tensor,
      params_.retainedTokens.tensor,params_.retainedNormalized.tensor})execution->requirePreparedInput(t);
  execution->requirePreparedOutput(params_.acceptedTokens.tensor);
  execution->requirePreparedOutput(params_.acceptedConvolution.tensor);
 }
 operations_.publish(publication(rows,stride),stream);
 if(execution){execution->publish(params_.acceptedTokens.tensor);execution->publish(params_.acceptedConvolution.tensor);}return true;
}
bool Qwen4ExpPLERetainedStage::restoreVerifierStateCaptureRow(int row,void*s){return !device().is_gpu()&&params_.requests==1&&publish(&row,1,1,s);}
bool Qwen4ExpPLERetainedStage::restoreVerifierStateCaptureRows(const int*r,int n,void*s){return !device().is_gpu()&&publish(r,n,1,s);}
std::span<const BufferBinding> Qwen4ExpPLERetainedStage::verifierStatePublicationOutputs() const noexcept
{
    return params_.operation == Operation::RetainCandidates
        ? std::span<const BufferBinding>{acceptedPublicationOutputs_}
        : std::span<const BufferBinding>{};
}

bool Qwen4ExpPLERetainedStage::restoreVerifierStateCaptureRowFromDeviceIndex(std::span<const int> r_view,void*s){
    if (!deviceVerifierStateMetadataCovers(r_view, 1, 1)) return false;
    const int *r = r_view.data();return device().is_gpu()&&params_.requests==1&&publish(r,1,1,s);}
bool Qwen4ExpPLERetainedStage::restoreVerifierStateCaptureRowsFromDeviceIndices(std::span<const int> r_view,int n,int stride,void*s){
    if (!deviceVerifierStateMetadataCovers(r_view, n, stride)) return false;
    const int *r = r_view.data();return device().is_gpu()&&publish(r,n,stride,s);}
ComputeStageType Qwen4ExpPLERetainedStage::type()const{return params_.operation==Operation::SnapshotPrior?ComputeStageType::QWEN4EXP_PLE_SNAPSHOT_PRIOR:ComputeStageType::QWEN4EXP_PLE_RETAIN_CANDIDATES;}
std::size_t Qwen4ExpPLERetainedStage::estimatedMemoryBytes()const{std::size_t n=0;for(const auto&o:operands_)n+=o.bytes;return n;}
bool Qwen4ExpPLERetainedStage::supportsBackend(ComputeBackendType b)const{return (device().is_cpu()&&b==ComputeBackendType::CPU)||(device().is_cuda()&&b==ComputeBackendType::GPU_CUDA)||(device().is_rocm()&&b==ComputeBackendType::GPU_ROCM);}
// Dedicated write-only prior/live/traffic roots are materialized by the ordinary
// cold arena pass. Read/write accepted banks retain their explicit InOut policy.
StageBufferContract Qwen4ExpPLERetainedStage::bufferContract()const{StageBufferContract c;for(const auto&o:operands_){const bool read=std::any_of(operands_.begin(),operands_.end(),[&](const auto&b){return b.buffer==o.buffer&&!b.write;});const bool write=std::any_of(operands_.begin(),operands_.end(),[&](const auto&b){return b.buffer==o.buffer&&b.write;});const auto present=[&](const auto&v){return std::any_of(v.begin(),v.end(),[&](const auto&b){return b.id==o.buffer;});};if(read&&write){if(!present(c.inouts))c.addPreallocatedInOut(o.buffer,tensorTypeName(o.dtype));}else if(write){if(!present(c.outputs))c.addOutput(o.buffer,tensorTypeName(o.dtype));}else if(!present(c.inputs))c.addInput(o.buffer,tensorTypeName(o.dtype));}return c;}
StageDumpInfo Qwen4ExpPLERetainedStage::buildDumpInfoImpl()const{StageDumpInfo d;for(const auto&o:operands_){const auto*b=static_cast<const std::byte*>(o.tensor->raw_data());const void*p=b?b+o.offset:nullptr;if(o.write)d.outputs.push_back({o.name,p,1,o.bytes/4,tensorTypeName(o.dtype),4,o.bytes,o.tensor,{},o.offset});else d.inputs.push_back({o.name,p,1,o.bytes/4,tensorTypeName(o.dtype),4,o.bytes,o.tensor,o.offset});}d.addScalarInt("request_index",params_.requestIndex);d.addScalarInt("requests",params_.requests);d.addScalarInt("row_capacity",params_.rowCapacity);d.addScalarInt("channels",params_.channels);d.addScalarInt("operation",static_cast<int>(params_.operation));if(const auto*n=std::get_if<Qwen4ExpNativeIntInput>(&params_.tokens.value)){d.addScalarInt("native_token_pointer",static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(n->data)));d.addScalarInt("native_token_capacity",n->capacity);}d.addScalarInt("verifier_origins_pointer",static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(params_.verifierOrigins.words())));d.addScalarInt("verifier_origin_word_stride",params_.verifierOrigins.wordStride());if(!params_.verifierOrigins.words()){d.addScalarInt("verifier_first_row",params_.verifierOrigins.resolveNative(0));if(params_.requests>1)d.addScalarInt("verifier_request_stride",params_.verifierOrigins.resolveNative(1)-params_.verifierOrigins.resolveNative(0));}return d;}
std::uint64_t Qwen4ExpPLERetainedStage::graphCaptureVariantSignature()const{
 storage_->validateIfSealed();
 std::uint64_t hash=1469598103934665603ull;
 const auto mix=[&](std::uint64_t value){hash^=value;hash*=1099511628211ull;};
 mix(1);mix(static_cast<std::uint64_t>(type()));mix(static_cast<std::uint64_t>(device().type));mix(device().ordinal);
 mix(params_.requests);mix(params_.requestIndex);mix(params_.rowCapacity);mix(params_.channels);mix(static_cast<std::uint64_t>(params_.operation));
 for(const auto&o:operands_){mix(reinterpret_cast<std::uintptr_t>(o.tensor));mix(static_cast<std::uint64_t>(o.buffer));mix(static_cast<std::uint64_t>(o.dtype));mix(o.offset);mix(o.bytes);mix(o.write);}
 const auto input=[&](const Qwen4ExpIntInput&value){
  mix(value.value.index());std::visit([&](const auto&binding){using B=std::decay_t<decltype(binding)>;
   if constexpr(std::is_same_v<B,Qwen4ExpIntBinding>){mix(reinterpret_cast<std::uintptr_t>(binding.tensor));mix(static_cast<std::uint64_t>(binding.buffer));mix(binding.byteOffset);}
   else {mix(static_cast<std::uint64_t>(binding.owner.type));mix(binding.owner.ordinal);mix(reinterpret_cast<std::uintptr_t>(binding.data));mix(binding.capacity);}
  },value.value);
 };
 mix(params_.liveCounts.has_value());if(params_.liveCounts)input(*params_.liveCounts);input(params_.tokens);
 mix(reinterpret_cast<std::uintptr_t>(params_.verifierOrigins.words()));mix(params_.verifierOrigins.wordStride());
 // Native origin words remain device-owned. Affine coordinates are immutable
 // and are hashed only over their admitted request domain, never beyond it.
 if(!params_.verifierOrigins.words())for(int r=0;r<params_.requests;++r)mix(params_.verifierOrigins.resolveNative(r));
 mix(reinterpret_cast<std::uintptr_t>(operations_.snapshot));mix(reinterpret_cast<std::uintptr_t>(operations_.retain));mix(reinterpret_cast<std::uintptr_t>(operations_.publish));
 return hash;
}

}
