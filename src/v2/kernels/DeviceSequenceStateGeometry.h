/**
 * @file DeviceSequenceStateGeometry.h
 * @brief Shared device sequence-metadata shape for cache owners and admission.
 *
 * A recurrent-only pipeline slice owns no attention payload, but still has a
 * sequence frontier. Its single head/count row is metadata, not a dummy KV
 * layer. Ordinary caches reuse their existing per-attention-layer rows.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace llaminar2
{
    /** @brief Immutable checkpoint geometry, independent of KV payload codec. */
    class DeviceSequenceStateGeometry
    {
    public:
        /** @brief Resolve an existing cache's metadata from its attention-layer count. */
        explicit constexpr DeviceSequenceStateGeometry(int attention_layers)
            : rows_(attention_layers > 0 ? attention_layers : 1)
        {
            if (attention_layers < 0)
                throw std::invalid_argument("Device sequence metadata needs a nonnegative layer count");
        }

        /** @return Number of head/count pairs in one sequence checkpoint. */
        [[nodiscard]] constexpr int rows() const noexcept { return rows_; }

        /** @return Bytes in one sequence checkpoint, including a recurrent-only frontier. */
        [[nodiscard]] constexpr std::size_t checkpointBytes() const noexcept
        {
            return static_cast<std::size_t>(rows_) * 2u * sizeof(int32_t);
        }

    private:
        int rows_; ///< Physical metadata rows, never a claim of attention payload ownership.
    };
}
