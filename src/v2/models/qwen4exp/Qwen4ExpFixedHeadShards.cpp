/**
 * @file Qwen4ExpFixedHeadShards.cpp
 * @brief Partition complete original native storage at exact QFN head boundaries.
 *
 * Source ranges are retained only by LayoutGeometry. Equal head workloads use
 * unequal native storage whenever original vocab sizes/gaps differ. This is a
 * preparation component; request replication, transport and serving are absent.
 */
#include "Qwen4ExpFixedHeadShards.h"
#include <algorithm>
#include <array>
#include <stdexcept>
namespace llaminar2 {
std::shared_ptr<const ResolvedImmutableRowTablePlacement> Qwen4ExpFixedHeadShards::declare(
    ImmutableRowTableExtent source,std::span<const std::uint64_t> offsets,
    std::span<const std::uint64_t> sizes,std::span<const PhysicalMemoryAllocatorIdentity> participants)
{
    constexpr std::size_t heads=16;const auto count=participants.size();
    if(offsets.size()!=heads || sizes.size()!=heads || (count!=2 && count!=4 && count!=8))
        throw std::invalid_argument("QFN fixed head shards require all16 source heads and TP degree2/4/8");
    const auto first=participants.front();
    for(std::size_t i=0;i<count;++i)
        if(!participants[i].valid() || !participants[i].device.is_gpu() || participants[i].world_rank<0 ||
            participants[i].world_rank!=first.world_rank || participants[i].device.type!=first.device.type ||
            std::find(participants.begin(),participants.begin()+i,participants[i])!=participants.begin()+i)
            throw std::invalid_argument("Fixed head shards require unique homogeneous rank-local GPU owners==consumers");
    std::uint64_t end=0;
    for(std::size_t h=0;h<heads;++h)
    {
        if(!sizes[h] || offsets[h]<end || offsets[h]>=source.rows() || sizes[h]>source.rows()-offsets[h])
            throw std::invalid_argument("QFN original head ranges overlap or exceed the full native source");
        end=offsets[h]+sizes[h];
    }
    std::vector<ImmutableRowTableLayoutOwner> owners;
    std::vector<ImmutableRowTableFixedRange> ranges;
    const auto local_heads=heads/count;
    for(std::size_t p=0;p<count;++p)
    {
        // First starts at0 even if head0 has a leading gap; the final source
        // owner retains trailing rows. No original byte is silently discarded.
        const auto begin=p?offsets[p*local_heads]:0;
        const auto finish=p+1==count?source.rows():offsets[(p+1)*local_heads];
        const auto range=ImmutableRowTableSourceRange::checked(source,begin,finish-begin);
        const ImmutableRowTableOwnerKey key{participants[p],ImmutableRowStorage::VRAM};
        owners.push_back({key,range.rows(),0,std::nullopt});
        ranges.push_back({range.firstRow(),range.rows(),key,ImmutableRowTableCopyRole::Primary});
    }
    const std::array tiers{ImmutableRowStorage::VRAM};
    auto layout=ImmutableRowTableLayoutGeometry::checked(source,ImmutableRowTableTiers::of(tiers),
        ImmutableRowTableResidency::Fixed,std::move(owners),{participants.begin(),participants.end()},std::move(ranges),{});
    return std::shared_ptr<const ResolvedImmutableRowTablePlacement>(new ResolvedImmutableRowTablePlacement(std::move(layout)));
}
Qwen4ExpFixedHeadShardAssignment Qwen4ExpFixedHeadShards::assignmentFor(
    const ResolvedImmutableRowTablePlacement &placement,PhysicalMemoryAllocatorIdentity owner)
{
    const auto &ranges=placement.fixedHeadShardLayout().fixedRanges();
    for(const auto &range:ranges)
        if(range.owner.allocator==owner)
        {
            // LayoutGeometry canonicalizes by physical owner, not TP order.
            // Derive the head coordinate from original source order; never
            // duplicate or reinterpret the canonical owner/range inventory.
            const auto p=static_cast<std::size_t>(std::count_if(ranges.begin(),ranges.end(),
                [&](const auto &other){return other.first_row<range.first_row;}));
            return {p,p*(16/ranges.size()),16/ranges.size(),placement.fixedHeadSourceRangeFor(owner)};
        }
    throw std::invalid_argument("QFN fixed head assignment queried by a foreign owner");
}
} // namespace llaminar2
