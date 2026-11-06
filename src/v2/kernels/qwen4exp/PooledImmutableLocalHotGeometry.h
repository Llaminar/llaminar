/**
 * @file PooledImmutableLocalHotGeometry.h
 * @brief One participant's immutable local native-cache and single-frame PMA declaration.
 *
 * Geometry contributes exact root and ordinary-event BOMs to the sole PMA. It
 * neither selects capacity nor creates a second source. The mandatory response
 * window exists at C0 and is reused for current/future intent; only one frame
 * may remain active until its actual consumed/discarded readers retire.
 */
#pragma once
#include "PooledImmutableLocalHotOffsets.h"
#include "loaders/ImmutableRowTableExtent.h"
#include "AdaptiveImmutableRowCacheState.h"
#include "backends/NativeOrderingEventBudgetProjection.h"
#include <array>
namespace llaminar2 {struct ImmutableRowTableSource;}
namespace llaminar2::qwen4exp {
/** @brief Source-owned bounded geometry shared by planner, root factory and launch validation. */
class PooledImmutableLocalHotGeometry final {
public:
    /**
     * @brief Declare a separate local cache over the SAME source and actual compute consumer.
     * @param source Original native row metadata; no payload is opened or hashed.
     * @param consumer Actual rank/GPU allocator, validated by the enclosing resolved role map.
     * @param maximum_original_ids Original complete sixteen-head query capacity.
     * @param slots Optional admitted native slots; C0 retains mandatory execution storage.
     * @param policy SAME existing deterministic frequency/recency policy.
     * @return Immutable offsets; no available-memory arithmetic or allocation occurs.
     * @throws std::exception For foreign geometry, incomplete native rows or overflow.
     */
    static PooledImmutableLocalHotGeometry declare(const ImmutableRowTableSource& source,
        PhysicalMemoryAllocatorIdentity consumer,std::uint32_t maximum_original_ids,
        std::uint32_t slots,adaptive_lookup::Policy policy);
    /** @return Original rank/device identity, not an allocation permit. */
    PhysicalMemoryAllocatorIdentity allocator()const noexcept{return consumer_;}
    /** @return Exact original source extent. */
    ImmutableRowTableExtent extent()const noexcept{return extent_;}
    /** @return Original source format, retained through native response/local decode. */
    GGUFTensorType format()const noexcept{return extent_.format();}
    /** @return Unnarrowed source logical columns; declaration rejects decoder-int overflow. */
    std::uint64_t rowElements()const noexcept{return row_elements_;}
    /** @return Exact native bytes, independent of FP32 output width or slot stride. */
    std::size_t rowBytes()const noexcept{return extent_.rowBytes();}
    /** @return Four-byte allocation stride; no padding is a transfer extent. */
    std::size_t slotStride()const noexcept{return stride_;}
    /** @return Original full query bound, divisible by sixteen. */
    std::uint32_t maximumOriginalIds()const noexcept{return maximum_;}
    /** @return Compact prefix capacity; its live count is arbitrary, including zero. */
    std::uint32_t maximumCompactMissRows()const noexcept{return maximum_;}
    /** @return Actual optional physical slots. */
    std::uint32_t slots()const noexcept{return slots_;}
    /** @return Existing policy constants; no mutable cache state is projected. */
    adaptive_lookup::Policy policy()const noexcept{return policy_;}
    /** @return Original source-key width, never inferred from the current IDs. */
    adaptive_lookup::RowKeyWidth rowKeyWidth()const noexcept{return width_;}
    /** @return Exact existing frequency sketch lane bound. */
    std::uint32_t laneWidth()const noexcept{return lane_width_;}
    /** @return Existing protected-stream limit. */
    std::uint32_t protectedLimit()const noexcept{return protected_;}
    /** @return Optional native payload+canonical slot/directory/policy bytes only. */
    std::size_t physicalBytes()const noexcept{return cache_.bytes;}
    /** @return Exact named optional-root subranges. */
    const PooledImmutableLocalHotCacheOffsets& cacheOffsets()const noexcept{return cache_;}
    /** @return Exact mandatory response/identity/inverse subranges. */
    const PooledImmutableLocalHotFrameOffsets& frameOffsets()const noexcept{return frame_;}
    /** @return Exact cooperative coalescing/scan scratch. */
    const PooledImmutableLocalHotScratchOffsets& scratchOffsets()const noexcept{return scratch_;}
    /** @return Exactly the three root extents consumed by the original root materializer. */
    std::array<std::size_t,3> rootBytes()const noexcept{return {cache_.bytes,frame_.bytes,scratch_.bytes};}
    /** @return Mandatory frame+scratch+three original setup/publication events at C0. */
    PhysicalMemoryPlan mandatoryBOM(PhysicalMemoryResource consumer,PhysicalMemoryResource host)const;
    /** @return SAME mandatory contribution from the original installed/recorded projection. */
    PhysicalMemoryPlan mandatoryBOM(PhysicalMemoryResource consumer,PhysicalMemoryResource host,
        const NativeOrderingEventBudgetProjection& projection)const;
    /** @return Optional cache root plus its actual additional tensor publication event. */
    PhysicalMemoryPlan slotDependentBOM(PhysicalMemoryResource consumer,PhysicalMemoryResource host)const;
    /** @return SAME optional growth under one authentic ordering-resource projection. */
    PhysicalMemoryPlan slotDependentBOM(PhysicalMemoryResource consumer,PhysicalMemoryResource host,
        const NativeOrderingEventBudgetProjection& projection)const;
    bool operator==(const PooledImmutableLocalHotGeometry&)const=default;
private:
    /** @brief Only checked declaration can create geometry. */
    explicit PooledImmutableLocalHotGeometry(ImmutableRowTableExtent extent):extent_(extent){}
    /** @brief Reject a foreign resource before contributing any canonical BOM. */
    void require(PhysicalMemoryResource consumer,PhysicalMemoryResource host)const;
    ImmutableRowTableExtent extent_;
    PhysicalMemoryAllocatorIdentity consumer_{};
    std::uint64_t row_elements_=0;
    std::size_t stride_=0;
    std::uint32_t maximum_=0,slots_=0,lane_width_=0,protected_=0;
    adaptive_lookup::Policy policy_{};
    adaptive_lookup::RowKeyWidth width_=adaptive_lookup::RowKeyWidth::UInt32;
    PooledImmutableLocalHotCacheOffsets cache_;
    PooledImmutableLocalHotFrameOffsets frame_;
    PooledImmutableLocalHotScratchOffsets scratch_;
};
} // namespace llaminar2::qwen4exp
