/**
 * @file QwenSparseSelectionCases.h
 * @brief Small immutable score distributions and an independent sort oracle.
 *
 * This diagnostic oracle is never a production implementation. It compares
 * numeric FP32 scores directly, preserving lower original IDs for ties, then
 * sorts the chosen IDs. It does not call bounded key/radix helper code.
 */
#pragma once
#include "kernels/qwen4exp/QwenSparseAttention.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>
namespace llaminar2::qwen4exp::selection_test
{
inline constexpr std::array policies{QwenSparsePoolSelection::Bounded256_512,
    QwenSparsePoolSelection::Bounded512_512, QwenSparsePoolSelection::Bounded1024_1024};
enum class Distribution { Hash, Equal, Ascending, Descending, Alternating, ByteBoundary, FiniteExtremes };
inline constexpr std::array distributions{Distribution::Hash, Distribution::Equal, Distribution::Ascending,
    Distribution::Descending, Distribution::Alternating, Distribution::ByteBoundary, Distribution::FiniteExtremes};
/** @brief Construct finite nonnegative immutable FP32 data without floating reductions. */
inline float score(int id, int row, Distribution distribution)
{
    switch (distribution)
    {
    case Distribution::Equal: return 0.0f; // Actual score producer publishes canonical +0.
    case Distribution::Ascending: return float(id / 3);
    case Distribution::Descending: return 16777216.0f - float(id / 3);
    case Distribution::Alternating: return id % 2 ? float(id % 521) : 1024.0f;
    case Distribution::ByteBoundary:
    {
        const std::uint32_t bits = 0x3f7fff00u + std::uint32_t(id % 1025);
        float result; std::memcpy(&result, &bits, sizeof(result)); return result;
    }
    case Distribution::FiniteExtremes:
        return id % 3 == 0 ? std::numeric_limits<float>::max() :
            id % 3 == 1 ? std::numeric_limits<float>::denorm_min() : 0.0f;
    case Distribution::Hash:
    {
        std::uint32_t value = std::uint32_t(id) + 0x9e3779b9u * (std::uint32_t(row) + 1u);
        value ^= value >> 16; value *= 0x7feb352du; value ^= value >> 15;
        return float(value % 65537u) / 256.0f;
    }
    }
    return 0.0f;
}
/** @brief Fill each admitted score row; causal consumption is tested separately. */
inline std::vector<float> scores(int rows, int blocks, Distribution distribution)
{
    std::vector<float> result(std::size_t(rows) * blocks);
    for (int row = 0; row < rows; ++row)
        for (int block = 0; block < blocks; ++block)
            result[std::size_t(row) * blocks + block] = score(block, row, distribution);
    return result;
}
/** @brief Independent numeric score/ID sort, exact pool expansion and raw tail. */
inline std::vector<std::int32_t> oracle(const float *scores, int visible)
{
    const int blocks = visible / QwenSparsePool;
    std::vector<int> ids(blocks);
    std::iota(ids.begin(), ids.end(), 0);
    if (blocks > QwenSparseBlocks)
    {
        std::sort(ids.begin(), ids.end(), [&](int a, int b)
        { return scores[a] > scores[b] || (scores[a] == scores[b] && a < b); });
        ids.resize(QwenSparseBlocks);
        std::sort(ids.begin(), ids.end());
    }
    std::vector<std::int32_t> result;
    for (int id : ids)
        for (int member = 0; member < QwenSparsePool; ++member) result.push_back(id * QwenSparsePool + member);
    for (int tail = 0; tail < visible % QwenSparsePool; ++tail) result.push_back(blocks * QwenSparsePool + tail);
    return result;
}
} // namespace llaminar2::qwen4exp::selection_test
