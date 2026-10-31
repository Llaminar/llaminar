/**
 * @file PooledImmutableNativePlan.cpp
 * @brief Checked exact native adapter layout without allocation or duplicate capacity accounting.
 *
 * All additional bytes are mandatory typed staging on the original allocator.
 * Immutable owner order, every static offset/policy and native launch version
 * enter identity. Hashes inspect declarations only, never rows or mutable state.
 */
#include "PooledImmutableNativePlan.h"
#include <limits>
#include <stdexcept>
namespace llaminar2::qwen4exp {
namespace {
namespace n=pooled_native;
/** @return Checked declaration sum before exposing a root extent. */
std::size_t plus(std::size_t a,std::size_t b)
{
    if(b>std::numeric_limits<std::size_t>::max()-a)throw std::overflow_error("Pooled native byte extent overflows");
    return a+b;
}
/** @return Checked declaration product. */
std::size_t times(std::size_t a,std::size_t b)
{
    if(a && b>std::numeric_limits<std::size_t>::max()/a)throw std::overflow_error("Pooled native array extent overflows");
    return a*b;
}
/** @return Root alignment only; PMA owns allocation admission. */
std::size_t align(std::size_t x){return plus(x,63)/64*64;}
/** @brief Append one exact nonempty subrange. */
void append(std::size_t& end,std::size_t& offset,std::size_t bytes)
{if(!bytes){offset=0;return;}offset=align(end);end=plus(offset,bytes);}
/** @brief Explicit field hashing avoids object padding and mutable values. */
class Identity
{
public:
    /** @brief Mix one immutable field. */
    void mix(std::uint64_t value){value_= (value_^value)*1099511628211ULL;}
    /** @brief Mix all exact cache offsets. */
    void ownerCache(PooledPhysicalOwnerOffsets c){mix(c.control);mix(c.slots);mix(c.payload);mix(c.bytes);}
    /** @brief Mix all exact frozen owner-frame offsets, including explicit lifetime aliases. */
    void ownerFrame(PooledPhysicalFrameOffsets f)
    {
        mix(f.control);mix(f.commands);mix(f.fill_order);mix(f.replies);mix(f.acknowledgements);
        mix(f.allocation_control);mix(f.accepted_plan);mix(f.initial_pins);
        mix(f.validation_a);mix(f.validation_b);mix(f.validation_counts);
        mix(f.pin_headers);mix(f.publish_headers);mix(f.free_headers);mix(f.victim_headers);mix(f.release_receipts);mix(f.bytes);
    }
    /** @brief Mix the original independent allocator identity. */
    void allocator(PhysicalMemoryAllocatorIdentity a){mix(a.world_rank);mix(static_cast<std::uint64_t>(a.device.type));mix(a.device.ordinal);}
    /** @return Nonzero complete declaration identity. */
    std::uint64_t value()const{return value_?value_:1;}
private:
    std::uint64_t value_=1469598103934665603ULL;
};
/** @brief Reject mixed declarations before deriving native offsets. */
void requireGeometry(const PooledImmutablePhysicalOwnerGeometry& physical,const PooledImmutableCoordinatorGeometry& coordinator)
{
    const auto p=coordinator.parameters();
    if(coordinator.coordinator()!=physical.coordinator() ||
        p.sourceRows()!=physical.extent().rows() || p.maximumUnique()!=physical.maximumUniqueRows() ||
        p.participants()!=physical.owners().size() || coordinator.owners().size()!=physical.owners().size())
        throw std::invalid_argument("Pooled native plan requires the exact original owner/coordinator/source declaration");
    for(std::size_t i=0;i<physical.owners().size();++i)
        if(coordinator.owners()[i].slots!=physical.owners()[i].slots)
            throw std::invalid_argument("Pooled native plan cannot replace the original owner geometry");
}
/** @brief Runtime declarations additionally require actual positive source identities. */
void require(const PooledImmutablePhysicalOwnerGeometry& physical,const PooledImmutableCoordinatorGeometry& coordinator,
    std::uint64_t table,std::uint64_t generation){
    requireGeometry(physical,coordinator);
    if(!table || !generation)throw std::invalid_argument("Pooled native runtime plan requires its actual source identity and storage generation");
}
/** @return Typed additional mandatory staging only, on the exact original resource. */
PhysicalMemoryPlan bom(PhysicalMemoryAllocatorIdentity allocator,std::size_t bytes,PhysicalMemoryResource observed)
{
    if(!observed.valid() || PhysicalMemoryAllocatorIdentity{observed.world_rank,observed.device}!=allocator)
        throw std::invalid_argument("Pooled native BOM requires its exact original allocator");
    PhysicalMemoryPlanBuilder builder;builder.add(observed,PhysicalMemoryOwner::ImmutableLookupStaging,bytes);return builder.build();
}
/** @return One canonical owner layout shared by pre-source BOM declaration and actual native materialization. */
n::OwnerOffsets ownerOffsets(std::uint32_t slots,std::uint32_t maximumUnique){
    n::OwnerOffsets offsets{};
    offsets.reply_capacity=std::min(slots,maximumUnique);
    std::size_t end=0;
    append(end,offsets.progress,sizeof(n::OwnerProgress));append(end,offsets.header,64);
    append(end,offsets.replies,times(offsets.reply_capacity,sizeof(adaptive_lookup::pooled::Grant)));
    offsets.bytes=align(end);
    return offsets;
}
/** @return One canonical coordinator layout; source identity affects capture, never physical byte geometry. */
n::CoordinatorOffsets coordinatorOffsets(adaptive_lookup::pooled::CoordinatorParameters policy){
    n::CoordinatorOffsets offsets{};std::size_t end=0;
    const auto k=policy.maximumUnique(),p=policy.participants();
    offsets.reply_capacity=std::min(k,policy.entries());
    append(end,offsets.invocation,sizeof(n::Invocation));
    append(end,offsets.pin_commands,times(offsets.reply_capacity,sizeof(adaptive_lookup::pooled::OwnerCommand)));
    append(end,offsets.pin_slices,times(p,sizeof(n::DirectedSlice)));
    append(end,offsets.pin_cursors,times(p,sizeof(std::uint32_t)));
    offsets.radix_blocks=static_cast<std::uint32_t>((std::uint64_t{policy.maximumOriginal()}+255u)/256u);
    append(end,offsets.radix_offsets,times(times(offsets.radix_blocks,16),sizeof(std::uint32_t)));
    append(end,offsets.availability,times(p,sizeof(adaptive_lookup::pooled::AvailabilityLease)));
    append(end,offsets.free_headers,times(p,sizeof(n::FreePlanHeader)));
    append(end,offsets.free_slices,times(p,sizeof(n::DirectedSlice)));
    append(end,offsets.victim_headers,times(p,sizeof(n::VictimPlanHeader)));
    append(end,offsets.victim_slices,times(p,sizeof(n::DirectedSlice)));
    append(end,offsets.sorted_releases,times(offsets.reply_capacity,sizeof(adaptive_lookup::pooled::ReleaseReceipt)));
    offsets.bytes=align(end);
    return offsets;
}
/** @brief Bind unchanged source extent and complete independent ordered owner/coordinator identity. */
Identity common(const PooledImmutablePhysicalOwnerGeometry& physical,std::uint64_t table,std::uint64_t generation)
{
    Identity identity;identity.mix(n::Version);identity.mix(table);identity.mix(generation);
    identity.mix(static_cast<std::uint64_t>(physical.extent().format()));identity.mix(physical.extent().rows());
    identity.mix(physical.rowBytes());identity.mix(physical.slotStride());identity.mix(physical.maximumUniqueRows());
    identity.allocator(physical.coordinator());identity.mix(physical.owners().size());
    for(const auto& owner:physical.owners()){
        identity.allocator(owner.allocator);identity.mix(owner.coordinate);identity.mix(owner.slots);
    }
    return identity;
}
}
PooledImmutableOwnerNativePlan PooledImmutableOwnerNativePlan::fromGeometry(
    const PooledImmutablePhysicalOwnerGeometry& physical,const PooledImmutableCoordinatorGeometry& coordinator,
    std::uint32_t coordinate,std::uint64_t table,std::uint64_t generation)
{
    require(physical,coordinator,table,generation);
    if(coordinate>=physical.owners().size())throw std::out_of_range("Pooled native owner coordinate is not declared");
    const auto& owner=physical.owners()[coordinate];
    const auto offsets=ownerOffsets(owner.slots,physical.maximumUniqueRows());
    auto identity=common(physical,table,generation);
    identity.mix(0);identity.mix(coordinate);identity.mix(coordinator.parameters().maximumOriginal());
    identity.ownerCache(owner.cache);identity.ownerFrame(owner.frame);
    identity.mix(offsets.progress);identity.mix(offsets.header);identity.mix(offsets.replies);
    identity.mix(offsets.reply_capacity);identity.mix(offsets.bytes);
    return {{physical.ownerParameters(coordinate),owner.cache,owner.frame,offsets,owner.allocator.device,
        coordinator.parameters().maximumOriginal(),physical.rowBytes(),physical.slotStride(),table,generation,identity.value()},owner.allocator};
}
PhysicalMemoryPlan PooledImmutableOwnerNativePlan::mandatoryBOM(PhysicalMemoryResource observed)const
{return bom(allocator_,parameters_.native.bytes,observed);}
PooledImmutableCoordinatorNativePlan PooledImmutableCoordinatorNativePlan::fromGeometry(
    const PooledImmutablePhysicalOwnerGeometry& physical,const PooledImmutableCoordinatorGeometry& coordinator,
    std::uint64_t table,std::uint64_t generation)
{
    require(physical,coordinator,table,generation);
    const auto policy=coordinator.parameters();const auto offsets=coordinatorOffsets(policy);
    auto identity=common(physical,table,generation);
    identity.mix(1);identity.mix(policy.entries());identity.mix(policy.maximumOriginal());identity.mix(policy.maximumUnique());
    identity.mix(policy.participants());identity.mix(policy.laneWidth());identity.mix(policy.protectedLimit());identity.mix(policy.agingAccesses());
    const auto c=coordinator.policyOffsets();identity.mix(c.control);identity.mix(c.entries);identity.mix(c.nodes);
    identity.mix(c.epochs);identity.mix(c.counts);identity.mix(c.bytes);
    const auto f=coordinator.frameOffsets();identity.mix(f.control);identity.mix(f.unique);identity.mix(f.sort_a);identity.mix(f.sort_b);
    identity.mix(f.radix_counts);identity.mix(f.original_to_unique);identity.mix(f.owner_plans);identity.mix(f.decisions);
    identity.mix(f.owners);identity.mix(f.bytes);
    identity.mix(offsets.invocation);identity.mix(offsets.pin_commands);identity.mix(offsets.pin_slices);
    identity.mix(offsets.reply_capacity);identity.mix(offsets.pin_cursors);identity.mix(offsets.radix_offsets);identity.mix(offsets.radix_blocks);identity.mix(offsets.availability);identity.mix(offsets.free_headers);identity.mix(offsets.free_slices);identity.mix(offsets.victim_headers);
    identity.mix(offsets.victim_slices);identity.mix(offsets.sorted_releases);identity.mix(offsets.bytes);
    for(const auto owner:coordinator.owners()){identity.mix(owner.first_entry);identity.mix(owner.slots);}
    return {{policy,c,f,offsets,physical.coordinator().device,table,generation,identity.value()},physical.coordinator(),
        {coordinator.owners().begin(),coordinator.owners().end()}};
}
PhysicalMemoryPlan PooledImmutableCoordinatorNativePlan::mandatoryBOM(PhysicalMemoryResource observed)const
{return bom(allocator_,parameters_.native.bytes,observed);}
PhysicalMemoryPlan PooledImmutableOwnerNativePlan::mandatoryBOM(const PooledImmutablePhysicalOwnerGeometry& physical,
    const PooledImmutableCoordinatorGeometry& coordinator,std::uint32_t coordinate,PhysicalMemoryResource observed){
    requireGeometry(physical,coordinator);
    if(coordinate>=physical.owners().size())throw std::out_of_range("Native owner BOM coordinate is not declared");
    const auto& owner=physical.owners()[coordinate];return bom(owner.allocator,ownerOffsets(owner.slots,physical.maximumUniqueRows()).bytes,observed);
}
PhysicalMemoryPlan PooledImmutableCoordinatorNativePlan::mandatoryBOM(const PooledImmutablePhysicalOwnerGeometry& physical,
    const PooledImmutableCoordinatorGeometry& coordinator,PhysicalMemoryResource observed){
    requireGeometry(physical,coordinator);return bom(physical.coordinator(),coordinatorOffsets(coordinator.parameters()).bytes,observed);
}

}
