/**
 * @file Qwen4ExpCheckpointGeometry.h
 * @brief One checked physical-capacity contract for QFN opaque rollback state.
 *
 * Rollback retains only raw QSA tails and accepted lexical histories, never
 * complete pooled-key banks. Archive endpoints use a different compact format.
 * Planner and cache owners consume this geometry without a second live ledger.
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace llaminar2::qwen4exp
{
/** @brief Native opaque state geometry shared by admission and materialization. */
struct Qwen4ExpCheckpointGeometry final
{
    static constexpr std::size_t TailHeaderBytes = 16; ///< Native live-row count header, padded to four words.
    static constexpr std::size_t TailRows = 3; ///< Maximum incomplete pool suffix for compression ratio four.
    static constexpr std::size_t IndexDimension = 128; ///< One raw indexer key, FP32.
    static constexpr std::size_t PLETokenHistory = 2; ///< Accepted raw suffix, INT32.
    static constexpr std::size_t PLEConvolutionHistory = 9; ///< Accepted dilated convolution history rows.
    /** @return Exact reserved bytes for one sparse layer's rollback tail. */
    static constexpr std::size_t sparseLayerBytes() noexcept
    { return TailHeaderBytes + TailRows * IndexDimension * sizeof(float); }
    /** @return Exact reserved bytes for one PLE layer, with checked channel geometry. */
    static std::size_t pleHistoryBytes(std::size_t channels)
    {
        if (channels > (std::numeric_limits<std::size_t>::max() - PLETokenHistory * sizeof(std::int32_t)) /
                           (PLEConvolutionHistory * sizeof(float)))
            throw std::overflow_error("QFN lexical rollback capacity exceeds addressable bytes");
        return PLETokenHistory * sizeof(std::int32_t) + PLEConvolutionHistory * channels * sizeof(float);
    }
    /**
     * @brief Price one request's cache-owned opaque auxiliary checkpoint.
     * @param sparse_layers Exact number of sparse layers in this cache participant.
     * @param ple_channels Exact local PLE trunk channels, or zero when this cache owns no PLE layer.
     * @return Tail capacity plus one accepted PLE history when present.
     */
    static std::size_t capacity(std::size_t sparse_layers, std::size_t ple_channels)
    {
        if (sparse_layers > std::numeric_limits<std::size_t>::max() / sparseLayerBytes())
            throw std::overflow_error("QFN sparse rollback capacity exceeds addressable bytes");
        const auto tails = sparse_layers * sparseLayerBytes();
        const auto ple = ple_channels ? pleHistoryBytes(ple_channels) : 0;
        if (ple > std::numeric_limits<std::size_t>::max() - tails)
            throw std::overflow_error("QFN combined rollback capacity exceeds addressable bytes");
        return tails + ple;
    }
};
} // namespace llaminar2::qwen4exp
