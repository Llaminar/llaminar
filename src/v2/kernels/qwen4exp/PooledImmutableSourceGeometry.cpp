/**
 * @file PooledImmutableSourceGeometry.cpp
 * @brief Source-native exact bounded staging BOM with complete immutable identity.
 *
 * All byte arithmetic declares allocation requirements only. The sole existing
 * PMA admits/materializes them, and the service-issued source binder retains
 * actual provider/file/mapped-channel lifetime before any native launch.
 */
#include "PooledImmutableSourceGeometry.h"
#include "loaders/ImmutableRowTable.h"
#include "ImmutableLookupDecode.h"
#include "loaders/ModelLoader.h"
#include <limits>
#include <algorithm>
#include <stdexcept>
namespace llaminar2::qwen4exp {
namespace {
/** @return Checked exact byte product before exposing any declaration. */
std::size_t product(std::size_t count,std::size_t bytes){
    if(count && bytes>std::numeric_limits<std::size_t>::max()/count)throw std::overflow_error("Pooled source staging product overflows");
    return count*bytes;
}
/** @brief Append one declared nonempty subrange; alignment is storage, never a live communication extent. */
void append(std::size_t& end,std::size_t& offset,std::size_t bytes){
    if(!bytes){offset=0;return;}
    if(end>std::numeric_limits<std::size_t>::max()-63)throw std::overflow_error("Pooled source alignment overflows");
    offset=(end+63)/64*64;
    if(bytes>std::numeric_limits<std::size_t>::max()-offset)throw std::overflow_error("Pooled source bank overflows");
    end=offset+bytes;
}
/** @return The same canonical bank shape for pre-source BOMs and authenticated runtime declarations. */
PooledSourceOffsets sourceOffsets(std::uint32_t k,std::uint32_t local,std::size_t rowBytes){
    PooledSourceOffsets offsets;std::size_t end=0;
    append(end,offsets.coordinates,product(k,sizeof(std::uint32_t)));
    append(end,offsets.responses,product(k,rowBytes));
    append(end,offsets.invalidations,product(k,sizeof(ImmutableRowInvalidationAcknowledgement)));
    append(end,offsets.fills,product(local,sizeof(pooled_native::Fill)));
    append(end,offsets.payload,product(local,rowBytes));
    if(end>std::numeric_limits<std::size_t>::max()-63)throw std::overflow_error("Pooled source root extent overflows");
    offsets.frame_bytes=(end+63)/64*64;
    return offsets;
}
/** @return Typed source staging on the exact independent original coordinator resource. */
PhysicalMemoryPlan sourceBOM(PhysicalMemoryAllocatorIdentity allocator,PooledSourceOffsets offsets,PhysicalMemoryResource resource){
    if(!resource.valid() || PhysicalMemoryAllocatorIdentity{resource.world_rank,resource.device}!=allocator)
        throw std::invalid_argument("Pooled source staging BOM requires its original coordinator allocator");
    PhysicalMemoryPlanBuilder builder;builder.add(resource,PhysicalMemoryOwner::ImmutableLookupStaging,offsets.frame_bytes);
    builder.add(resource,PhysicalMemoryOwner::ImmutableLookupStaging,offsets.native_bytes);return builder.build();
}

}
PooledImmutableSourceGeometry PooledImmutableSourceGeometry::fromPlans(const PooledImmutableOwnerNativePlan& owner,
    const PooledImmutableCoordinatorNativePlan& coordinator,const ImmutableRowTableSource& source)
{
    coordinator.requireOwnerPlan(owner);
    const auto& o=owner.parameters();const auto& c=coordinator.parameters();
    if(!source.rows || !source.row_elements || source.row_elements>static_cast<std::uint64_t>(std::numeric_limits<int>::max()) || source.rows>static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
        o.owner.sourceRows()!=source.rows || c.policy.sourceRows()!=source.rows || o.row_bytes!=source.rowBytes() ||
        o.table!=c.table || o.source_storage_generation!=c.source_storage_generation || o.owner.maximumUnique()!=c.policy.maximumUnique() ||
        o.maximum_original!=c.policy.maximumOriginal() || o.owner.participants()!=c.policy.participants() ||
        o.owner.coordinate()>=coordinator.owners().size() || coordinator.owners()[o.owner.coordinate()].slots!=o.owner.capacity())
        throw std::invalid_argument("Pooled source geometry differs from its original provider/owner/coordinator declarations");
    const auto format=ggufToTensorType(source.format);
    lookup_detail::validateGeometry(format,static_cast<int>(source.row_elements),source.rowBytes());
    const auto k=c.policy.maximumUnique(),local=o.native.reply_capacity;
    const auto offsets=sourceOffsets(k,local,o.row_bytes);
    std::uint64_t identity=1469598103934665603ULL;const auto mix=[&](std::uint64_t x){identity=(identity^x)*1099511628211ULL;};
    mix(pooled_native::Version);mix(1);mix(o.identity);mix(c.identity);mix(static_cast<std::uint64_t>(format));mix(source.row_elements);
    for(const auto offset:{offsets.coordinates,offsets.responses,offsets.invalidations,offsets.fills,offsets.payload,offsets.frame_bytes,
        offsets.protocol,offsets.frontier,offsets.record_bytes,offsets.payload_bytes,offsets.invocation,offsets.response_stage,offsets.observation,offsets.observation_ordinal,offsets.reported_ordinal,offsets.native_bytes})mix(offset);
    return {{o,c,offsets,format,static_cast<std::uint32_t>(source.row_elements),identity?identity:1},coordinator.allocator()};
}
PhysicalMemoryPlan PooledImmutableSourceGeometry::mandatoryBOM(PhysicalMemoryResource resource)const{
    return sourceBOM(allocator_,parameters_.offsets,resource);
}
PhysicalMemoryPlan PooledImmutableSourceGeometry::mandatoryBOM(const PooledImmutablePhysicalOwnerGeometry& physical,
    const PooledImmutableCoordinatorGeometry& coordinator,std::uint32_t coordinate,const ImmutableRowTableSource& source,
    PhysicalMemoryResource resource){
    return sourceBOM(physical.coordinator(),rootOffsets(physical,coordinator,coordinate,source),resource);
}
PooledSourceOffsets PooledImmutableSourceGeometry::rootOffsets(const PooledImmutablePhysicalOwnerGeometry& physical,
    const PooledImmutableCoordinatorGeometry& coordinator,std::uint32_t coordinate,const ImmutableRowTableSource& source){
    const auto policy=coordinator.parameters();
    if(coordinate>=physical.owners().size() || physical.coordinator()!=coordinator.coordinator() ||
       physical.extent().format()!=source.format || physical.extent().rows()!=source.rows || physical.rowBytes()!=source.rowBytes() ||
       policy.sourceRows()!=source.rows || policy.maximumUnique()!=physical.maximumUniqueRows() ||
       policy.participants()!=physical.owners().size() || !source.row_elements ||
       source.row_elements>static_cast<std::uint64_t>(std::numeric_limits<int>::max()) || source.rows>static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
        throw std::invalid_argument("Pooled source BOM requires original matching physical/coalescing/source declarations");
    lookup_detail::validateGeometry(ggufToTensorType(source.format),static_cast<int>(source.row_elements),source.rowBytes());
    const auto local=std::min(physical.owners()[coordinate].slots,physical.maximumUniqueRows());
    return sourceOffsets(physical.maximumUniqueRows(),local,source.rowBytes());
}
ImmutableLookupLayout PooledImmutableSourceGeometry::channelLayout()const{
    return ImmutableLookupLayout::createSubordinate(parameters_.coordinator.policy.maximumUnique(),parameters_.owner.row_bytes);
}
}
