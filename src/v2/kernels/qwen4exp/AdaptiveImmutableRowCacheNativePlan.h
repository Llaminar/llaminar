/**
 * @file AdaptiveImmutableRowCacheNativePlan.h
 * @brief Lightweight checked source geometry consumed by CUDA/HIP adaptive kernels.
 *
 * The checked host geometry alone constructs this declaration. It grants no
 * physical allocation or source access; actual prepared roots and the existing
 * service binding separately authenticate lifetime/runtime/coherence ownership.
 */
#pragma once
#include "AdaptiveImmutableRowCacheState.h"
#include "backends/DeviceId.h"
#include "tensors/TensorType.h"
#include <cstddef>
namespace llaminar2::qwen4exp
{
class AdaptiveImmutableRowCacheGeometry;
/** @brief Named offsets in the optional canonical native cache byte root. */
struct AdaptiveCacheOffsets
{
    std::size_t owner = 0, slots = 0, nodes = 0, epochs = 0, counts = 0, payload = 0, bytes = 0;
    bool operator==(const AdaptiveCacheOffsets &) const = default;
};
/** @brief Mandatory frame-root offsets; original64B protocol remains a separate canonical root. */
struct AdaptiveFrameOffsets
{
    std::size_t control = 0, keys_a = 0, keys_b = 0, first = 0, original_to_unique = 0, first_prefix = 0;
    std::size_t unique = 0, miss_prefix = 0, acknowledgements = 0, scan = 0, native_response = 0, bytes = 0;
    bool operator==(const AdaptiveFrameOffsets &) const = default;
};
/** @brief Trivially copyable immutable native parameters; no live row/count/location state. */
struct AdaptiveNativeParameters
{
    int world_rank;
    DeviceId device;
    TensorType format;
    int row_elements;
    std::uint64_t source_rows;
    std::size_t row_bytes, slot_stride;
    std::uint32_t slots, maximum_ids, lane_width, protected_limit;
    adaptive_lookup::RowKeyWidth row_width;
    adaptive_lookup::Policy policy;
    AdaptiveCacheOffsets cache;
    AdaptiveFrameOffsets frame;
    bool operator==(const AdaptiveNativeParameters &) const = default;
};
/** @brief Factory-only checked geometry declaration, never a materialization permit. */
class AdaptiveImmutableRowCacheNativePlan final
{
public:
    /** @return Full immutable native geometry for launch/capture identity. */
    const AdaptiveNativeParameters &parameters() const noexcept
    {
        return parameters_;
    }
    /** @return Every immutable scalar/offset, without hashing structure padding or mutable state. */
    std::uint64_t captureIdentity() const noexcept
    {
        const auto &p = parameters_;
        std::uint64_t h = 1469598103934665603ULL;
        const auto mix = [&](std::uint64_t word)
        {
            h ^= word;
            h *= 1099511628211ULL;
        };
        mix(1);
        mix(p.world_rank);
        mix(static_cast<std::uint64_t>(p.device.type));
        mix(p.device.ordinal);
        mix(static_cast<std::uint64_t>(p.format));
        mix(p.row_elements);
        mix(p.source_rows);
        mix(p.row_bytes);
        mix(p.slot_stride);
        mix(p.slots);
        mix(p.maximum_ids);
        mix(p.lane_width);
        mix(p.protected_limit);
        mix(static_cast<std::uint64_t>(p.row_width));
        mix(p.policy.aging_accesses);
        mix(p.policy.protected_numerator);
        mix(p.policy.protected_denominator);
        mix(p.cache.owner);
        mix(p.cache.slots);
        mix(p.cache.nodes);
        mix(p.cache.epochs);
        mix(p.cache.counts);
        mix(p.cache.payload);
        mix(p.cache.bytes);
        mix(p.frame.control);
        mix(p.frame.keys_a);
        mix(p.frame.keys_b);
        mix(p.frame.first);
        mix(p.frame.original_to_unique);
        mix(p.frame.first_prefix);
        mix(p.frame.unique);
        mix(p.frame.miss_prefix);
        mix(p.frame.acknowledgements);
        mix(p.frame.scan);
        mix(p.frame.native_response);
        mix(p.frame.bytes);
        return h ? h : 1;
    }
    /** @return Every source/policy/offset field agrees; no pointer-only compatibility. */
    bool operator==(const AdaptiveImmutableRowCacheNativePlan &) const = default;

private:
    friend class AdaptiveImmutableRowCacheGeometry;
    /** @brief Checked geometry passes already-authenticated fields without payload reads. */
    explicit AdaptiveImmutableRowCacheNativePlan(AdaptiveNativeParameters p) : parameters_(p) {}
    AdaptiveNativeParameters parameters_;
};
} // namespace llaminar2::qwen4exp
