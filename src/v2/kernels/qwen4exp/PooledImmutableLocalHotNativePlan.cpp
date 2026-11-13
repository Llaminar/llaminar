/**
 * @file PooledImmutableLocalHotNativePlan.cpp
 * @brief Authenticate the same original pooled source before issuing local native geometry.
 */
#include "PooledImmutableLocalHotNativePlan.h"
#include "loaders/ImmutableRowTable.h"
#include "PooledImmutableSourcePreparedStorage.h"
#include "PooledImmutableLocalHotNativeTypes.h"
#include <stdexcept>
namespace llaminar2::qwen4exp {
PooledImmutableLocalHotNativePlan PooledImmutableLocalHotNativePlan::fromOriginalSource(
    PooledImmutableLocalHotGeometry geometry,std::shared_ptr<PooledImmutableSourcePreparedStorage> source)
{
    if(!source)throw std::invalid_argument("Local hot cache requires its genuine original pooled source");
    const auto& p=source->geometry().parameters();
    if(geometry.allocator().world_rank!=source->geometry().allocator().world_rank ||
        geometry.extent().rows()!=p.coordinator.policy.sourceRows() ||
        geometry.rowBytes()!=p.owner.row_bytes || geometry.rowElements()!=p.row_elements ||
        ggufToTensorType(geometry.format())!=p.format || !p.coordinator.table ||
        !p.coordinator.source_storage_generation)
        throw std::invalid_argument("Local hot native geometry differs from its original table/source");
    std::uint64_t identity=1469598103934665603ULL;
    const auto mix=[&](std::uint64_t value){identity=(identity^value)*1099511628211ULL;};
    mix(local_hot::NativeVersion);mix(source->captureIdentity());mix(p.coordinator.table);mix(p.coordinator.source_storage_generation);
    mix(geometry.allocator().world_rank);mix(unsigned(geometry.allocator().device.type));mix(geometry.allocator().device.ordinal);
    mix(geometry.extent().rows());mix(unsigned(geometry.format()));mix(geometry.rowElements());mix(geometry.rowBytes());
    mix(geometry.maximumOriginalIds());mix(geometry.slots());mix(geometry.physicalBytes());
    for(const auto bytes:geometry.rootBytes())mix(bytes);
    const auto policy=geometry.policy();mix(policy.aging_accesses);mix(policy.protected_numerator);mix(policy.protected_denominator);
    return PooledImmutableLocalHotNativePlan(std::move(geometry),std::move(source),p.coordinator.table,
        p.coordinator.source_storage_generation,p.format,identity?identity:1);
}
} // namespace llaminar2::qwen4exp
