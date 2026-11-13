/** @file NativeContextSourceProbeROCm.cpp
 * @brief Integration-only exact SDK provider and original ordering-event observation.
 */
#include "NativeContextSourceProbeSupport.h"
#include "backends/rocm/AMDDeviceContext.h"
namespace llaminar2::test {
std::vector<NativeSourceSDKBinding> nativeSourceROCmBindings(){return {
    nativeSourceSDKBinding("hipblasCreate",reinterpret_cast<const void*>(&hipblasCreate),NATIVE_SOURCE_EXPECTED_ROCM_BLAS),
    nativeSourceSDKBinding("hipblasLtCreate",reinterpret_cast<const void*>(&hipblasLtCreate),NATIVE_SOURCE_EXPECTED_ROCM_LT)};}
std::vector<void*> NativeContextSourceLifecycleDiagnosticAccess::rocmEvents(const PreparedGPUContextUsePermit& permit){
    requireOriginalWorker(permit);auto* owner=dynamic_cast<AMDDeviceContext*>(permit.worker().get());
    if(!owner || !permit.device().is_rocm())throw std::logic_error("Original ordering observer received a foreign backend worker");
    std::vector<void*> events;events.reserve(owner->stream_dependency_events_.size());
    for(auto event:owner->stream_dependency_events_){if(!event)throw std::logic_error("Original ordering family has a missing handle");events.push_back(reinterpret_cast<void*>(event));}
    if(events.size()!=32)throw std::logic_error("Original ordering family geometry changed");return events;
}
}
