/** @file AdaptiveImmutableLookupGeometryReplayAccess.h
 * @brief Sealed recorded-budget access to the same production geometry contribution bodies.
 */
#pragma once
#include "loaders/AdaptiveImmutableLookupGeometry.h"
namespace llaminar2::test {
/** @brief No numeric bounds, catalog override or alternate solver exists in this seam. */
struct AdaptiveImmutableLookupGeometryReplayAccess {
    static auto mandatory(const AdaptiveImmutableLookupGeometry& geometry,std::span<const PhysicalMemoryResource> devices,
        PhysicalMemoryResource host,const NativeOrderingEventBudgetProjection& projection){
        return geometry.mandatoryBOMWithOrderingProjection(devices,host,projection);
    }
    static auto optional(const AdaptiveImmutableLookupGeometry& geometry,std::span<const PhysicalMemoryResource> devices,
        PhysicalMemoryResource host,const NativeOrderingEventBudgetProjection& projection){
        return geometry.slotDependentBOMWithOrderingProjection(devices,host,projection);
    }
};
}
