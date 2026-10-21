/**
 * @file AdaptiveImmutableRowCacheGeometry.h
 * @brief Checked byte geometry for one native adaptive GPU cache and its mandatory frame.
 *
 * This pure source-owned geometry contributes exact typed BOMs to PMA. It does
 * not select tiers, observe free memory, materialize buffers, or create a second
 * ledger. Optional slots and mandatory frame/protocol storage remain distinct.
 */
#pragma once
#include "AdaptiveImmutableRowCacheNativePlan.h"
#include "loaders/ImmutableRowTable.h"
#include "loaders/ImmutableRowTableLayoutGeometry.h"
#include "execution/qwen4exp/ImmutableLookupProtocol.h"
#include "planning/PhysicalMemoryAuthority.h"
#include <cstddef>
namespace llaminar2::qwen4exp
{
/** @brief Immutable source/owner policy geometry, independent of live occupancy. */
class AdaptiveImmutableRowCacheGeometry final
{
public:
    /**
     * @brief Validate exact one GPU requester/coordinator plus its explicit local SSD service.
     * @param layout Checked adaptive layout, with optional native tier slots and no replicas/ranges.
     * @param requester Exact same-rank GPU consumer; positive slots require this coordinator.
     * @param source Original source metadata; no native bytes are read or hashed.
     * @param maximumTokenRows Original physical token capacity, covered by the declared inflight-ID bound.
     * @param policy Explicit immutable per-coordinate frequency/protected policy.
     * @return Fully checked original-format offsets and capacity, including legal zero slots.
     * @throws std::invalid_argument or std::overflow_error Before any mutation for unsupported
     * ownership/geometry.
     */
    static AdaptiveImmutableRowCacheGeometry fromLayout(const ImmutableRowTableLayoutGeometry &layout,
                                                        PhysicalMemoryAllocatorIdentity requester,
                                                        const ImmutableRowTableSource &source,
                                                        std::size_t maximumTokenRows,
                                                        adaptive_lookup::Policy policy);
    /** @return Exact allocator whose PMA owns native slots and device frame. */
    PhysicalMemoryAllocatorIdentity requester() const noexcept
    {
        return requester_;
    }
    /** @return Original native source extent; never a current slot-location map. */
    ImmutableRowTableExtent extent() const noexcept
    {
        return extent_;
    }
    /** @return Original source scalar/codebook. */
    TensorType format() const noexcept
    {
        return format_;
    }
    /** @return Exact original logical columns per source row. */
    int rowElements() const noexcept
    {
        return row_elements_;
    }
    /** @return Exact original native row bytes, excluding allocation-only slot padding. */
    std::size_t rowBytes() const noexcept
    {
        return extent_.rowBytes();
    }
    /** @return Original format-aligned physical slot stride; wire payload always uses rowBytes(). */
    std::size_t slotStride() const noexcept
    {
        return slot_stride_;
    }
    /** @return Exact slot count, including legal0 and values above16M. */
    std::uint32_t slots() const noexcept
    {
        return slots_;
    }
    /** @return Exact admitted original IDs, not the runtime compact miss count. */
    std::uint32_t maximumIds() const noexcept
    {
        return maximum_ids_;
    }
    /** @return Immutable original-source row-key record width. */
    adaptive_lookup::RowKeyWidth rowKeyWidth() const noexcept
    {
        return row_width_;
    }
    /** @return Per-lane four-lane frequency capacity, zero when there are no slots. */
    std::uint32_t laneWidth() const noexcept
    {
        return lane_width_;
    }
    /** @return Exact protected-region ceiling, including pinned/pending list entries. */
    std::uint32_t protectedLimit() const noexcept
    {
        return protected_limit_;
    }
    /** @return Explicit frozen policy values, part of complete capture identity. */
    adaptive_lookup::Policy policy() const noexcept
    {
        return policy_;
    }
    /** @return All optional cache subranges in the canonical root. */
    const AdaptiveCacheOffsets &cache() const noexcept
    {
        return cache_;
    }
    /** @return All mandatory frame subranges, excluding the separate original protocol root. */
    const AdaptiveFrameOffsets &frame() const noexcept
    {
        return frame_;
    }
    /** @return Exact mandatory device root sum, frame plus original protocol64B. */
    std::size_t mandatoryDeviceBytes() const noexcept
    {
        return frame_.bytes + sizeof(ImmutableLookupDeviceState);
    }
    /** @return Exact protocol2 host-channel requested geometry; TransferEngine owns physical mapping
     * rounding. */
    ImmutableLookupLayout channelLayout() const
    {
        return ImmutableLookupLayout::createSubordinate(maximum_ids_, rowBytes());
    }
    /** @return Epoch/count scan scratch bytes in the mandatory frame. */
    std::size_t scanBytes() const noexcept
    {
        return scan_bytes_;
    }
    /**
     * @brief Contribute mandatory GPU frame/protocol bytes before optional bulk-cache admission.
     * @param observed Exact canonical requester resource observation, including its rank/device.
     * @return Typed ImmutableLookupStaging input to the existing PMA planner; no allocation decision occurs
     * here.
     * @throws std::invalid_argument For a foreign or invalid observed resource.
     */
    PhysicalMemoryPlan mandatoryBOM(PhysicalMemoryResource observed) const;
    /**
     * @brief Contribute optional native slots and all directory/policy metadata at the lowest bulk priority.
     * @param observed Exact canonical requester resource observation.
     * @return Typed ImmutableTableStorage BOM, empty at C0; never a runtime occupancy ledger.
     * @throws std::invalid_argument For a foreign or invalid observed resource.
     */
    PhysicalMemoryPlan cacheBOM(PhysicalMemoryResource observed) const;
    /** @return Lightweight checked native declaration, with no physical ownership permit. */
    AdaptiveImmutableRowCacheNativePlan nativePlan() const;
    /** @return Complete immutable source/owner/offset/policy equality. */
    bool operator==(const AdaptiveImmutableRowCacheGeometry &) const = default;

private:
    /** @brief Only the checked source factory constructs this pure declaration. */
    AdaptiveImmutableRowCacheGeometry(PhysicalMemoryAllocatorIdentity requester,
                                      ImmutableRowTableExtent extent, TensorType format, int columns,
                                      adaptive_lookup::Policy policy)
        : requester_(requester), extent_(extent), format_(format), row_elements_(columns), policy_(policy)
    {
    }
    PhysicalMemoryAllocatorIdentity requester_;
    ImmutableRowTableExtent extent_;
    TensorType format_;
    int row_elements_;
    adaptive_lookup::Policy policy_;
    adaptive_lookup::RowKeyWidth row_width_ = adaptive_lookup::RowKeyWidth::UInt32;
    std::uint32_t slots_ = 0, maximum_ids_ = 0, lane_width_ = 0, protected_limit_ = 0;
    std::size_t slot_stride_ = 0, scan_bytes_ = 0;
    AdaptiveCacheOffsets cache_;
    AdaptiveFrameOffsets frame_;
};
} // namespace llaminar2::qwen4exp
