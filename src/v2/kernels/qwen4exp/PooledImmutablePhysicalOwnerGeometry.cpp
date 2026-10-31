/**
 * @file PooledImmutablePhysicalOwnerGeometry.cpp
 * @brief Pure checked physical-owner offsets and typed PMA contributions.
 *
 * No source payload, backend, free-memory query, allocation or live cache state
 * is touched. Empty participants retain mandatory command/progress storage.
 * Owner ordering comes from the explicit declaration, never sorted devices or
 * the subset with positive capacity. This is not a complete serving-domain BOM.
 */
#include "PooledImmutablePhysicalOwnerGeometry.h"
#include "PooledImmutableOwnerPlanOps.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace llaminar2::qwen4exp
{
namespace
{
namespace a = adaptive_lookup;
namespace p = adaptive_lookup::pooled;

/** @return Checked byte sum before publishing any root descriptor. */
std::size_t add(std::size_t a, std::size_t b)
{
    if (a > std::numeric_limits<std::size_t>::max() - b)
        throw std::overflow_error("Pooled physical owner byte sum overflows");
    return a + b;
}

/** @return Checked native/metadata product without capacity truncation. */
std::size_t mul(std::size_t a, std::size_t b)
{
    if (a && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::overflow_error("Pooled physical owner byte extent overflows");
    return a * b;
}

/** @return Requested root alignment, not an invented PMA allocator rounding. */
std::size_t align(std::size_t n, std::size_t alignment = 64)
{
    return add(n, alignment - 1) / alignment * alignment;
}

/** @brief Append one admitted named subrange; omitted arrays consume no padding. */
void append(std::size_t &end, std::size_t &offset, std::size_t bytes)
{
    if (!bytes)
    {
        offset = 0;
        return;
    }
    offset = align(end);
    end = add(offset, bytes);
}
} // namespace

PooledImmutablePhysicalOwnerGeometry PooledImmutablePhysicalOwnerGeometry::fromLayout(
    const ImmutableRowTableLayoutGeometry &layout,
    std::span<const PhysicalMemoryAllocatorIdentity> orderedOwners,
    PhysicalMemoryAllocatorIdentity requester)
{
    const auto participants = orderedOwners.size();
    if ((!participants || participants > a::MaximumSlots) ||
        layout.residency() != ImmutableRowTableResidency::Adaptive ||
        !layout.tiers().contains(ImmutableRowStorage::VRAM) ||
        !layout.tiers().contains(ImmutableRowStorage::MemoryMapped) ||
        !layout.fixedRanges().empty() || !layout.replicaGroups().empty() ||
        layout.sourceServices().empty() ||
        !layout.maximumInflightRows() || layout.maximumInflightRows() > a::MaximumSlots ||
        requester.world_rank < 0 || !requester.device.is_gpu() || !requester.device.is_valid() ||
        !layout.sourceServices().front().allocator.device.is_cpu())
        throw std::invalid_argument("Pooled physical owners require explicit bounded owner metadata and an independent GPU coordinator");

    if (std::find(layout.consumers().begin(), layout.consumers().end(), requester) == layout.consumers().end() &&
        std::none_of(layout.owners().begin(), layout.owners().end(),
            [&](const auto &owner) { return owner.key.allocator == requester; }))
        throw std::invalid_argument("Pooled coordinator is not an original declared endpoint");
    PooledImmutablePhysicalOwnerGeometry result(layout.extent());
    result.coordinator_ = requester;
    result.maximum_unique_ = static_cast<std::uint32_t>(layout.maximumInflightRows());
    result.stride_ = align(result.rowBytes(), 4);
    result.width_ = result.extent_.rows() <= 0x100000000ULL ? a::RowKeyWidth::UInt32 : a::RowKeyWidth::UInt64;
    bool positive = false;
    for (std::uint32_t coordinate = 0; coordinate < participants; ++coordinate)
    {
        const auto allocator = orderedOwners[coordinate];
        if (allocator.world_rank < 0 || !allocator.device.is_valid() ||
            !allocator.device.is_gpu() ||
            std::find(orderedOwners.begin(), orderedOwners.begin() + coordinate, allocator) !=
                orderedOwners.begin() + coordinate)
            throw std::invalid_argument("Pooled physical owner order contains foreign or duplicate storage owners");
        if (allocator == requester)
        {
            result.requester_ = coordinate;
        }
        const auto entry = std::find_if(layout.owners().begin(), layout.owners().end(), [&](const auto &owner) {
            return owner.key == ImmutableRowTableOwnerKey{allocator, ImmutableRowStorage::VRAM};
        });
        if (entry == layout.owners().end() || entry->fixed_rows || entry->cache_rows > a::MaximumSlots)
            throw std::invalid_argument("Every pooled physical participant requires an explicit bounded VRAM owner");
        PooledPhysicalOwnerDeclaration owner{allocator, coordinate, static_cast<std::uint32_t>(entry->cache_rows),
                                             {}, {}, entry->physical_byte_limit};
        positive |= owner.slots != 0;
        std::size_t end = 0;
        if (owner.slots)
        {
            append(end, owner.cache.control, sizeof(p::PhysicalOwnerControl));
            append(end, owner.cache.slots, mul(owner.slots, result.width_ == a::RowKeyWidth::UInt32
                                                              ? sizeof(a::Slot32) : sizeof(a::Slot64)));
            append(end, owner.cache.payload, mul(owner.slots, result.stride_));
            owner.cache.bytes = align(end);
            if (entry->physical_byte_limit && owner.cache.bytes > *entry->physical_byte_limit)
                throw std::invalid_argument("Pooled physical owner payload/metadata exceeds its requested ceiling");
        }
        end = 0;
        append(end, owner.frame.control, sizeof(p::OwnerFrame));
        append(end, owner.frame.initial_pins, mul(result.maximum_unique_, sizeof(p::Grant)));
        append(end, owner.frame.allocation_control, sizeof(p::AllocationFrame));
        append(end, owner.frame.accepted_plan, mul(result.maximum_unique_, sizeof(p::AllocationDecision)));
        append(end, owner.frame.validation_a, mul(result.maximum_unique_, sizeof(std::uint64_t)));
        append(end, owner.frame.validation_b, mul(result.maximum_unique_, sizeof(std::uint64_t)));
        append(end, owner.frame.validation_counts, 256 * sizeof(std::uint32_t));
        append(end, owner.frame.commands, mul(result.maximum_unique_, sizeof(p::OwnerCommand)));
        // Both plan waves finish before source publication. Their validation
        // scratch then retains one immutable unique-order permutation; accepted
        // decisions and original grants remain untouched until final retirement.
        owner.frame.fill_order = owner.frame.validation_a;
        append(end, owner.frame.replies, mul(result.maximum_unique_, sizeof(p::Grant)));
        append(end, owner.frame.acknowledgements, mul(result.maximum_unique_, sizeof(p::FillAcknowledgement)));
        append(end, owner.frame.pin_headers, mul(participants, sizeof(p::PinHeader)));
        append(end, owner.frame.publish_headers, mul(participants, sizeof(p::PublishHeader)));
        // Free-frontier records retire before source fill/publication begins; one
        // admitted immutable header bank therefore serves both disjoint phases.
        static_assert(sizeof(p::FreeAllocationReceipt) <= sizeof(p::PublishHeader));
        static_assert(sizeof(p::VictimAllocationReceipt) <= sizeof(p::PublishHeader));
        owner.frame.free_headers = owner.frame.victim_headers = owner.frame.publish_headers;
        append(end, owner.frame.release_receipts, mul(result.maximum_unique_, sizeof(p::ReleaseReceipt)));
        owner.frame.bytes = align(end);
        result.owners_.push_back(owner);
    }
    if (positive ? layout.gpuCoordinator() != std::optional(requester)
                 : layout.gpuCoordinator().has_value())
        throw std::invalid_argument("Pooled physical requester differs from the declared coordinator");
    for (const auto &owner : layout.owners())
        if (owner.key.tier == ImmutableRowStorage::VRAM &&
            std::find(orderedOwners.begin(), orderedOwners.end(), owner.key.allocator) == orderedOwners.end())
            throw std::invalid_argument("Pooled declaration contains an unbound physical GPU owner");
    return result;
}

const PooledPhysicalOwnerDeclaration &PooledImmutablePhysicalOwnerGeometry::require(
    std::uint32_t coordinate, PhysicalMemoryResource observed) const
{
    if (coordinate >= owners_.size() || !observed.valid() ||
        PhysicalMemoryAllocatorIdentity{observed.world_rank, observed.device} != owners_[coordinate].allocator)
        throw std::invalid_argument("Pooled physical BOM requires its exact declared allocator observation");
    return owners_[coordinate];
}

adaptive_lookup::pooled::OwnerParameters PooledImmutablePhysicalOwnerGeometry::ownerParameters(
    std::uint32_t coordinate) const
{
    if (coordinate >= owners_.size())
        throw std::out_of_range("Pooled physical native coordinate is not declared");
    return {extent_.rows(), owners_[coordinate].slots, maximum_unique_, coordinate,
            static_cast<std::uint32_t>(owners_.size()), width_};
}

PhysicalMemoryPlan PooledImmutablePhysicalOwnerGeometry::mandatoryBOM(
    std::uint32_t coordinate, PhysicalMemoryResource observed) const
{
    PhysicalMemoryPlanBuilder builder;
    builder.add(observed, PhysicalMemoryOwner::ImmutableLookupStaging, require(coordinate, observed).frame.bytes);
    return builder.build();
}

PhysicalMemoryPlan PooledImmutablePhysicalOwnerGeometry::cacheBOM(
    std::uint32_t coordinate, PhysicalMemoryResource observed) const
{
    const auto &owner = require(coordinate, observed);
    PhysicalMemoryPlanBuilder builder;
    if (owner.cache.bytes)
        builder.add(observed, PhysicalMemoryOwner::ImmutableTableStorage, owner.cache.bytes);
    return builder.build();
}
} // namespace llaminar2::qwen4exp
