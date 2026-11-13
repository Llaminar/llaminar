/** @file NativeContextSourceProbeCUDA.cpp
 * @brief Integration-only exact SDK provider and original ordering-event observation.
 */
#include "NativeContextSourceProbeSupport.h"
#include "backends/cuda/NvidiaDeviceContext.h"
namespace llaminar2::test {
std::vector<NativeSourceSDKBinding> nativeSourceCUDABindings(){return {
    nativeSourceSDKBinding("cublasCreate_v2",reinterpret_cast<const void*>(&cublasCreate_v2),NATIVE_SOURCE_EXPECTED_CUDA_BLAS),
    nativeSourceSDKBinding("cublasLtCreate",reinterpret_cast<const void*>(&cublasLtCreate),NATIVE_SOURCE_EXPECTED_CUDA_LT)};}
std::vector<void*> NativeContextSourceLifecycleDiagnosticAccess::cudaEvents(const PreparedGPUContextUsePermit& permit){
    requireOriginalWorker(permit);auto* owner=dynamic_cast<NvidiaDeviceContext*>(permit.worker().get());
    if(!owner || !permit.device().is_cuda())throw std::logic_error("Original ordering observer received a foreign backend worker");
    std::vector<void*> events;events.reserve(owner->stream_dependency_events_.size());
    for(auto event:owner->stream_dependency_events_){if(!event)throw std::logic_error("Original ordering family has a missing handle");events.push_back(reinterpret_cast<void*>(event));}
    if(events.size()!=32)throw std::logic_error("Original ordering family geometry changed");return events;
}
}
