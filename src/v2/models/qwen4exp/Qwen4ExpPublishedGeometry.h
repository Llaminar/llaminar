/**
 * @file Qwen4ExpPublishedGeometry.h
 * @brief Published checkpoint dimensions authenticated by source and wire parsers.
 *
 * These constants validate explicitly present metadata; they never synthesize
 * absent fields. Keeping them together prevents transport from admitting a
 * coherent foreign geometry that the exact GGUF parser would reject.
 */
#pragma once
#include <cstdint>

namespace llaminar2::qwen4exp::published
{
inline constexpr std::uint32_t ComputeWidth = 2560, MainLayers = 48, AttentionInterval = 4;
inline constexpr std::uint32_t Branches = 4, GateRank = 320;
inline constexpr std::uint32_t QueryHeads = 24, KVHeads = 2, HeadDimension = 256;
inline constexpr std::uint32_t IndexerQueryHeads = 4, IndexerKeyHeads = 1, IndexerDimension = 128;
inline constexpr std::uint32_t CompressionRatio = 4, SelectedTokenBudget = 2048;
inline constexpr std::uint32_t PLELayer = 1, NgramOrder = 3, HeadsPerOrder = 8, RowElements = 160;
inline constexpr std::uint32_t PLEKernel = 4;
} // namespace llaminar2::qwen4exp::published
