/**
 * @file DeviceDecodePositionBinding.h
 * @brief Immutable bindings for graph-owned scalar decode position snapshots.
 *
 * The KV cache owns its mutable count. A forward needs a stable position for
 * RoPE and routing even after attention appends to that cache. The captured
 * root snapshots the count into an existing arena row before model work. This
 * value owns neither allocation nor live state; all addresses are capture
 * identity, while the count behind them is replay data.
 */
#pragma once

#include <cstdint>

namespace llaminar2
{
class IBackend;

/** @brief Complete borrowed input/output binding for one scalar decode root. */
struct DeviceDecodePositionBinding
{
    IBackend *backend = nullptr; ///< Exact backend of the graph's device.
    const int32_t *cached_tokens = nullptr; ///< Canonical cache-owned live count.
    int32_t *position = nullptr; ///< Separate persistent arena snapshot row.

    /** @return Whether the snapshot has distinct, non-null producers/consumers. */
    [[nodiscard]] bool valid() const noexcept
    {
        return backend && cached_tokens && position && cached_tokens != position;
    }

    /** @return Exact embedded identity; mutable device contents are excluded. */
    bool operator==(const DeviceDecodePositionBinding &) const = default;
};
}
