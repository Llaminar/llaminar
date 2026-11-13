/**
 * @file PooledImmutableCoordinatorGeometry.cpp
 * @brief Checked aligned pooled coordinator BOM, with no allocation or capacity decision.
 */
#include "PooledImmutableCoordinatorGeometry.h"
#include <limits>
#include <stdexcept>
namespace llaminar2::qwen4exp
{
namespace
{
/** @return Checked size arithmetic before a descriptor is published. */
std::size_t plus(std::size_t a, std::size_t b)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::overflow_error("Pooled coordinator byte sum overflows");
    return a + b;
}
/** @return Exact array bytes without multiplication truncation. */
std::size_t times(std::size_t a, std::size_t b)
{
    if (a && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::overflow_error("Pooled coordinator byte product overflows");
    return a * b;
}
/** @return Root/subarray alignment only, not PMA's allocator accounting. */
std::size_t aligned(std::size_t n) { return plus(n, 63) / 64 * 64; }
/** @brief Append exactly one nonempty named subrange. */
void append(std::size_t &end, std::size_t &offset, std::size_t bytes)
{
    if (!bytes) { offset = 0; return; }
    offset = aligned(end);
    end = plus(offset, bytes);
}
} // namespace
PooledImmutableCoordinatorGeometry PooledImmutableCoordinatorGeometry::fromOwners(
    const PooledImmutablePhysicalOwnerGeometry &owners, adaptive_lookup::Policy policy, std::uint32_t maximumOriginalIds)
{
    namespace a = adaptive_lookup;
    namespace p = a::pooled;
    if (!maximumOriginalIds || maximumOriginalIds > a::MaximumSlots ||
        maximumOriginalIds < owners.maximumUniqueRows() || !policy.aging_accesses || !policy.protected_denominator ||
        policy.protected_numerator > policy.protected_denominator)
        throw std::invalid_argument("Pooled coordinator policy is invalid");
    PooledImmutableCoordinatorGeometry result;
    auto &v = result.parameters_;
    v.rows_ = owners.extent().rows();
    v.aging_ = policy.aging_accesses;
    v.maximum_unique_ = owners.maximumUniqueRows();
    v.maximum_original_ = maximumOriginalIds;
    v.participants_ = static_cast<std::uint32_t>(owners.owners().size());
    result.coordinator_ = owners.coordinator();
    std::uint64_t total = 0;
    for (std::uint32_t i = 0; i < v.participants_; ++i)
    {
        result.owners_.push_back({static_cast<std::uint32_t>(total), owners.owners()[i].slots});
        total += owners.owners()[i].slots;
        if (total > a::MaximumSlots)
            throw std::invalid_argument("Pooled coordinator radix index exceeds uint31");

    }
    v.entries_ = static_cast<std::uint32_t>(total);
    if (total)
    {
        v.lane_width_ = 1;
        while (v.lane_width_ < total)
            v.lane_width_ <<= 1;
        v.protected_ = static_cast<std::uint32_t>(total * policy.protected_numerator / policy.protected_denominator);
        std::size_t end = 0;
        append(end, result.policy_.control, sizeof(p::CoordinatorPolicyControl));
        append(end, result.policy_.entries, times(total, sizeof(p::DirectoryEntry)));
        append(end, result.policy_.nodes, times(total - 1, sizeof(a::RadixNode)));
        append(end, result.policy_.epochs, times(times(4, v.lane_width_), sizeof(std::uint32_t)));
        append(end, result.policy_.counts, times(times(4, v.lane_width_), sizeof(std::uint16_t)));
        result.policy_.bytes = aligned(end);
    }
    if (owners.requesterCoordinate())
    {
        const auto &coordinator = owners.owners()[*owners.requesterCoordinate()];
        if (coordinator.physical_byte_limit &&
            plus(coordinator.cache.bytes, result.policy_.bytes) > *coordinator.physical_byte_limit)
            throw std::invalid_argument("Pooled coordinator physical cache plus policy exceeds original ceiling");
    }
    std::size_t end = 0;
    append(end, result.frame_.control, sizeof(p::LookupFrame));
    append(end, result.frame_.unique, times(v.maximum_unique_, sizeof(p::CoordinatorUnique)));
    append(end, result.frame_.sort_a, times(v.maximum_original_, sizeof(p::CoordinatorOccurrence)));
    append(end, result.frame_.sort_b, times(v.maximum_original_, sizeof(p::CoordinatorOccurrence)));
    append(end, result.frame_.radix_counts, 256 * sizeof(std::uint32_t));
    append(end, result.frame_.original_to_unique, times(v.maximum_original_, sizeof(std::uint32_t)));
    append(end, result.frame_.owner_plans, times(v.participants_, sizeof(p::CoordinatorOwnerPlan)));
    append(end, result.frame_.decisions, times(v.maximum_unique_, sizeof(p::AllocationDecision)));
    append(end, result.frame_.owners, times(v.participants_, sizeof(p::CoordinatorOwnerGeometry)));
    result.frame_.bytes = aligned(end);
    return result;
}
void PooledImmutableCoordinatorGeometry::require(PhysicalMemoryResource observed) const
{
    if (!observed.valid() || PhysicalMemoryAllocatorIdentity{observed.world_rank, observed.device} != coordinator_)
        throw std::invalid_argument("Pooled coordinator BOM requires its exact declared resource");
}
PhysicalMemoryPlan PooledImmutableCoordinatorGeometry::policyBOM(PhysicalMemoryResource observed) const
{
    require(observed);
    PhysicalMemoryPlanBuilder builder;
    if (policy_.bytes)
        builder.add(observed, PhysicalMemoryOwner::ImmutableTableStorage, policy_.bytes);
    return builder.build();
}
PhysicalMemoryPlan PooledImmutableCoordinatorGeometry::mandatoryBOM(PhysicalMemoryResource observed) const
{
    require(observed);
    PhysicalMemoryPlanBuilder builder;
    builder.add(observed, PhysicalMemoryOwner::ImmutableLookupStaging, frame_.bytes);
    return builder.build();
}
} // namespace llaminar2::qwen4exp
