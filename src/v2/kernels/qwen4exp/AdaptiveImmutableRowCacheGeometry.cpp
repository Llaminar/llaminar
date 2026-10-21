/**
 * @file AdaptiveImmutableRowCacheGeometry.cpp
 * @brief Exact source-owned offsets and BOM contributions for bounded adaptive GPU lookup.
 *
 * Pure checked arithmetic prices optional native slots/metadata separately from
 * mandatory frame/protocol work. It performs no free-memory query, file read,
 * physical allocation, policy fallback or runtime occupancy accounting.
 */
#include "AdaptiveImmutableRowCacheGeometry.h"
#include "ImmutableLookupDecode.h"
#include <algorithm>
#include <bit>
#include <limits>
#include <stdexcept>
namespace llaminar2::qwen4exp
{
namespace
{
/** @return Checked byte addition before publishing any geometry. */
std::size_t add(std::size_t a, std::size_t b)
{
    if (a > std::numeric_limits<std::size_t>::max() - b)
        throw std::overflow_error("Adaptive native byte sum overflows");
    return a + b;
}
/** @return Checked exact native/metadata extent. */
std::size_t mul(std::size_t a, std::size_t b)
{
    if (a && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::overflow_error("Adaptive native byte extent overflows");
    return a * b;
}
/** @return Aligned requested root coordinate, never an invented physical allocator rounding. */
std::size_t aligned(std::size_t n, std::size_t alignment = 64)
{
    return add(n, alignment - 1) / alignment * alignment;
}
/** @brief Append one nonempty named subarray; omitted arrays consume no padding. */
void append(std::size_t &end, std::size_t &offset, std::size_t bytes)
{
    if (!bytes)
    {
        offset = 0;
        return;
    }
    offset = aligned(end);
    end = add(offset, bytes);
}
/** @brief Reject a BOM observation from another rank/device before returning charges. */
void requireResource(PhysicalMemoryResource observed, PhysicalMemoryAllocatorIdentity exact)
{
    if (!observed.valid() || PhysicalMemoryAllocatorIdentity{observed.world_rank, observed.device} != exact)
        throw std::invalid_argument("Adaptive native BOM requires the exact observed GPU allocator");
}
} // namespace
AdaptiveImmutableRowCacheGeometry AdaptiveImmutableRowCacheGeometry::fromLayout(
    const ImmutableRowTableLayoutGeometry &layout, PhysicalMemoryAllocatorIdentity requester,
    const ImmutableRowTableSource &source, std::size_t maximumTokenRows, adaptive_lookup::Policy policy)
{
    const auto extent = ImmutableRowTableExtent::fromNative(source.format, source.rows, source.rowBytes());
    if (requester.world_rank < 0 || !requester.device.is_gpu() || !requester.device.is_valid() ||
        layout.residency() != ImmutableRowTableResidency::Adaptive ||
        !layout.tiers().contains(ImmutableRowStorage::MemoryMapped) || !layout.fixedRanges().empty() ||
        !layout.replicaGroups().empty() || layout.consumers().size() != 1 ||
        layout.consumers().front() != requester || layout.extent() != extent ||
        layout.sourceServices().size() != 1 ||
        layout.sourceServices().front().allocator.world_rank != requester.world_rank ||
        !layout.sourceServices().front().allocator.device.is_cpu() || !policy.aging_accesses ||
        !policy.protected_denominator || policy.protected_numerator > policy.protected_denominator ||
        !maximumTokenRows || maximumTokenRows > adaptive_lookup::MaximumSlots / 16 ||
        !layout.maximumInflightRows() || maximumTokenRows > layout.maximumInflightRows() / 16 ||
        !source.row_elements ||
        source.row_elements > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument(
            "Adaptive native lookup requires one exact GPU/SSD domain and complete bounded original source");
    const auto format = ggufToTensorType(source.format);
    const auto width = static_cast<int>(source.row_elements);
    lookup_detail::validateGeometry(format, width, source.rowBytes());
    if (NativeEmbeddingLayout::supports(format) &&
        static_cast<std::size_t>(width) % NativeEmbeddingLayout::forSource(format, 1, width).block_elements)
        throw std::invalid_argument("Adaptive native lookup requires complete original quantized row blocks");
    AdaptiveImmutableRowCacheGeometry g(requester, extent, format, width, policy);
    std::uint64_t slots = 0;
    std::optional<std::size_t> ceiling;
    for (const auto &owner : layout.owners())
    {
        if (owner.key.tier == ImmutableRowStorage::VRAM)
        {
            if (owner.key.allocator != requester || owner.fixed_rows ||
                owner.cache_rows > adaptive_lookup::MaximumSlots)
                throw std::invalid_argument(
                    "Adaptive native cache has foreign or unrepresentable GPU slot ownership");
            slots = owner.cache_rows;
            ceiling = owner.physical_byte_limit;
        }
        else if (owner.key.allocator != layout.sourceServices().front().allocator || owner.fixed_rows)
            throw std::invalid_argument("Adaptive native source has a foreign host pool");
    }
    if ((slots && layout.gpuCoordinator() != requester) || (!slots && layout.gpuCoordinator()))
        throw std::invalid_argument(
            "Adaptive native coordinator must own exactly the declared positive GPU pool");
    g.slots_ = static_cast<std::uint32_t>(slots);
    g.maximum_ids_ = static_cast<std::uint32_t>(maximumTokenRows * 16);
    g.row_width_ = source.rows <= 0x100000000ULL ? adaptive_lookup::RowKeyWidth::UInt32
                                                 : adaptive_lookup::RowKeyWidth::UInt64;
    const auto sourceAlignment =
        format == TensorType::FP32 || format == TensorType::Q8_K ? alignof(float) : alignof(std::uint16_t);
    g.slot_stride_ = aligned(source.rowBytes(), sourceAlignment);
    if (slots)
    {
        g.lane_width_ = std::bit_ceil(std::max<std::uint32_t>(16, (g.slots_ + 3U) / 4U));
        const auto quota = slots * policy.protected_numerator / policy.protected_denominator;
        g.protected_limit_ =
            g.slots_ <= 1 ? 0 : static_cast<std::uint32_t>(std::min<std::uint64_t>(quota, slots - 1));
        auto &o = g.cache_;
        std::size_t end = 0;
        append(end, o.owner, sizeof(adaptive_lookup::OwnerControl));
        append(end, o.slots,
               mul(g.slots_, g.row_width_ == adaptive_lookup::RowKeyWidth::UInt32
                                 ? sizeof(adaptive_lookup::Slot32)
                                 : sizeof(adaptive_lookup::Slot64)));
        append(end, o.nodes, mul(g.slots_ - 1, sizeof(adaptive_lookup::RadixNode)));
        append(end, o.epochs, mul(g.lane_width_, 4 * sizeof(std::uint32_t)));
        append(end, o.counts, mul(g.lane_width_, 4 * sizeof(std::uint16_t)));
        append(end, o.payload, mul(g.slots_, g.slot_stride_));
        o.bytes = aligned(end);
    }
    if (ceiling && g.cache_.bytes > *ceiling)
        throw std::invalid_argument(
            "Adaptive native ALL cache payload/metadata exceed the declared owner limit");
    auto &f = g.frame_;
    std::size_t end = 0;
    const auto k = static_cast<std::size_t>(g.maximum_ids_);
    append(end, f.control, sizeof(adaptive_lookup::FrameControl));
    append(end, f.keys_a, mul(k, sizeof(adaptive_lookup::KeyIndex)));
    append(end, f.keys_b, mul(k, sizeof(adaptive_lookup::KeyIndex)));
    append(end, f.first, mul(k, sizeof(std::uint32_t)));
    append(end, f.original_to_unique, mul(k, sizeof(std::uint32_t)));
    append(end, f.first_prefix, mul(k, sizeof(std::uint32_t)));
    append(end, f.unique, mul(k, sizeof(adaptive_lookup::UniqueRecord)));
    append(end, f.miss_prefix, mul(k, sizeof(std::uint32_t)));
    append(end, f.acknowledgements, mul(k, sizeof(ImmutableRowInvalidationAcknowledgement)));
    for (auto count = k;;)
    {
        count = (count + 255) / 256;
        g.scan_bytes_ = add(g.scan_bytes_, mul(count, 2 * sizeof(std::uint32_t)));
        if (count == 1)
            break;
    }
    append(end, f.scan, g.scan_bytes_);
    append(end, f.native_response, mul(k, g.rowBytes()));
    f.bytes = aligned(end);
    (void)add(f.bytes, sizeof(ImmutableLookupDeviceState));
    (void)g.channelLayout();
    return g;
}
AdaptiveImmutableRowCacheNativePlan AdaptiveImmutableRowCacheGeometry::nativePlan() const
{
    return AdaptiveImmutableRowCacheNativePlan({requester_.world_rank, requester_.device, format_,
                                                row_elements_, extent_.rows(), rowBytes(), slot_stride_,
                                                slots_, maximum_ids_, lane_width_, protected_limit_,
                                                row_width_, policy_, cache_, frame_});
}
PhysicalMemoryPlan AdaptiveImmutableRowCacheGeometry::mandatoryBOM(PhysicalMemoryResource observed) const
{
    requireResource(observed, requester_);
    PhysicalMemoryPlanBuilder b;
    b.add(observed, PhysicalMemoryOwner::ImmutableLookupStaging, mandatoryDeviceBytes());
    return b.build();
}
PhysicalMemoryPlan AdaptiveImmutableRowCacheGeometry::cacheBOM(PhysicalMemoryResource observed) const
{
    requireResource(observed, requester_);
    PhysicalMemoryPlanBuilder b;
    if (cache_.bytes)
        b.add(observed, PhysicalMemoryOwner::ImmutableTableStorage, cache_.bytes);
    return b.build();
}
} // namespace llaminar2::qwen4exp
