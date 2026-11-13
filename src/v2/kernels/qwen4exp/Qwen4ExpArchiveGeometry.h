/**
 * @file Qwen4ExpArchiveGeometry.h
 * @brief Checked compact QSA-pool and lexical endpoint archive geometry.
 *
 * This immutable descriptor is shared by cache owners and pre-allocation
 * planning. It describes live archive sections, which exclude opaque rollback
 * headers and unused raw-tail rows. Capacity selection never changes I/O size.
 */
#pragma once
#include "Qwen4ExpCheckpointGeometry.h"

namespace llaminar2::qwen4exp
{
/** @brief One request's participant-local sparse layers and optional lexical history. */
struct Qwen4ExpArchiveGeometry final
{
    std::size_t sparse_layers = 0;
    std::size_t ple_channels = 0; ///< Zero when this participant owns no PLE layer.
    static constexpr std::size_t CompressionRatio = Qwen4ExpCheckpointGeometry::TailRows + 1;
    static constexpr std::size_t KeyBytes = Qwen4ExpCheckpointGeometry::IndexDimension * sizeof(float);

    /** @return Whether this cache has no model auxiliary archive sections. */
    [[nodiscard]] bool empty() const noexcept { return sparse_layers == 0 && ple_channels == 0; }
    /** @brief Reject arithmetic overflow before constructing an archive extent. */
    [[nodiscard]] static std::size_t multiply(std::size_t left, std::size_t right)
    {
        if (right && left > std::numeric_limits<std::size_t>::max() / right)
            throw std::overflow_error("QFN compact archive extent exceeds addressable bytes");
        return left * right;
    }
    /** @brief Reject arithmetic overflow before joining physical archive sections. */
    [[nodiscard]] static std::size_t add(std::size_t left, std::size_t right)
    {
        if (right > std::numeric_limits<std::size_t>::max() - left)
            throw std::overflow_error("QFN compact archive section sum exceeds addressable bytes");
        return left + right;
    }
    /** @return Exact completed pool rows for one sparse layer in the immutable range. */
    [[nodiscard]] static std::size_t blockBytes(std::size_t first, std::size_t count)
    {
        const auto end = add(first, count);
        return multiply(end / CompressionRatio - first / CompressionRatio, KeyBytes);
    }
    /** @return Exact raw-tail rows at this logical prefix; no rollback header or padding. */
    [[nodiscard]] std::size_t endpointBytes(std::size_t logical_prefix) const
    {
        const auto tails = multiply(sparse_layers, multiply(logical_prefix % CompressionRatio, KeyBytes));
        return add(tails, ple_channels ? Qwen4ExpCheckpointGeometry::pleHistoryBytes(ple_channels) : 0);
    }
};
} // namespace llaminar2::qwen4exp
