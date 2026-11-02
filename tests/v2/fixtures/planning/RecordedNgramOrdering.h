/** @file RecordedNgramOrdering.h
 * @brief Explicit recorded resource inputs for device-free canonical ngram planning fixtures.
 * These sealed toy certificates authorize no native work and replace no installed catalog.
 */
#pragma once
#include "fixtures/backends/NativeOrderingEventGroupReplayAccess.h"
#include "fixtures/loaders/AdaptiveImmutableLookupGeometryReplayAccess.h"
#include <algorithm>
namespace llaminar2::test {
/** @return Sealed recorded source for exactly the fixture's declared physical GPUs. */
inline auto recordedNgramOrdering(std::span<const PhysicalMemoryResource> resources){
    std::vector<DeviceId> devices;for(const auto& resource:resources){
        if(!resource.device.is_gpu())throw std::invalid_argument("Recorded ordering source requires explicit original GPUs");
        if(std::find(devices.begin(),devices.end(),resource.device)==devices.end())devices.push_back(resource.device);}
    return NativeOrderingEventGroupReplayAccess::projection(devices);
}
/** @return The fixture's one explicit native consumer, or an unused empty source for CPU-only planning. */
inline auto recordedNgramOrdering(DeviceId device){
    std::vector<DeviceId> devices;if(device.is_gpu())devices.push_back(device);
    return NativeOrderingEventGroupReplayAccess::projection(devices);
}
/** @return Actual mandatory contribution using the same sealed recorded original device observations. */
inline auto recordedNgramMandatory(const AdaptiveImmutableLookupGeometry& geometry,std::span<const PhysicalMemoryResource> devices,PhysicalMemoryResource host){
    return AdaptiveImmutableLookupGeometryReplayAccess::mandatory(geometry,devices,host,recordedNgramOrdering(devices));
}
/** @return One-device spelling of the same actual common implementation. */
inline auto recordedNgramMandatory(const AdaptiveImmutableLookupGeometry& geometry,PhysicalMemoryResource device,PhysicalMemoryResource host){
    return recordedNgramMandatory(geometry,std::span{&device,1},host);
}
/** @return Actual optional contribution with the same sealed source as its fixed floor. */
inline auto recordedNgramOptional(const AdaptiveImmutableLookupGeometry& geometry,std::span<const PhysicalMemoryResource> devices,PhysicalMemoryResource host){
    return AdaptiveImmutableLookupGeometryReplayAccess::optional(geometry,devices,host,recordedNgramOrdering(devices));
}
/** @return One-device spelling of the same actual common implementation. */
inline auto recordedNgramOptional(const AdaptiveImmutableLookupGeometry& geometry,PhysicalMemoryResource device,PhysicalMemoryResource host){
    return recordedNgramOptional(geometry,std::span{&device,1},host);
}
}
