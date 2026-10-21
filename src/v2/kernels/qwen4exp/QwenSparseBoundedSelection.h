/**
 * @file QwenSparseBoundedSelection.h
 * @brief Exact bounded-candidate integer phases for the private QSA experiment.
 *
 * Adapts DeepSelect's pinned scan/filter/compact algorithm, without upstream
 * cluster, TMA or PTX machinery. Source traversal is ascending and deliberately
 * has no randomized expected-work claim. Stable compaction preserves ascending
 * original IDs, including canonical lower-ID score ties. This header owns no
 * persistent state, allocation, source bytes or request/capture authority.
 * @see https://github.com/deepseek-ai/DeepSelect/tree/bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18
 * Original bounded integer implementation; upstream MIT source receipt is kept
 * with the private candidate handoff, not loaded by the inference executable.
 */
#pragma once
#include "QwenSparseMath.h"
#include <cstdint>
#include <limits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define QSA_BOUNDED_HD __host__ __device__ __forceinline__
#else
#define QSA_BOUNDED_HD inline
#endif

namespace llaminar2::qwen4exp::bounded_selection
{
inline constexpr int Keep = QwenSparseBlocks;
inline constexpr int Radix = 256;

/** @brief Trivial lexical storage; native kernels place it in CTA shared memory. */
template<int Block, int Slack>
struct Storage final
{
    static_assert(Block > 0 && Slack > 0);
    static constexpr int Capacity = Keep + Block + Slack;
    std::uint64_t keys[2][Capacity];
    std::uint32_t histogram[Radix];
    std::uint64_t warp_min[8], warp_max[8];
    int warp_counts[8];
    std::uint64_t threshold;
    std::uint64_t pivot;
    std::uint64_t prefix;
    int size;
    int active;
    int shift;
    int rank;
    int written;
    int done;
    QwenSparseSelectionObservation observation;
};

/**
 * @brief Accept finite nonnegative FP32 values, including numerically equal signed zeros.
 * @param bits Original immutable FP32 representation; never an arithmetic conversion.
 * @return False for a negative nonzero, infinity or NaN encoding.
 */
QSA_BOUNDED_HD bool validScoreBits(std::uint32_t bits)
{
    return (bits & 0x7fffffffu) < 0x7f800000u &&
        (!(bits & 0x80000000u) || (bits & 0x7fffffffu) == 0);
}

/**
 * @brief Preserve FP32 value ordering and canonical lower original ID for ties.
 * @param bits Original immutable FP32 score encoding, already domain-validated.
 * @param id Original nonnegative complete-pool index, never a compact ordinal.
 * @return Unique total order; larger is better. Signed zeros compare numerically equal.
 */
QSA_BOUNDED_HD std::uint64_t key(std::uint32_t bits, std::uint32_t id)
{
    if ((bits & 0x7fffffffu) == 0) bits = 0;
    return (std::uint64_t(bits) << 32) | std::uint64_t(UINT32_MAX - id);
}

/**
 * @brief Recover the original index without reconstructing or changing a score.
 * @param value Valid total key from this admitted complete-pool geometry.
 * @return Original pool index, representable because token capacity is signed32.
 */
QSA_BOUNDED_HD int originalId(std::uint64_t value)
{
    return static_cast<int>(UINT32_MAX - static_cast<std::uint32_t>(value));
}

/**
 * @brief Skip the bounded bank's proven common high radix bytes.
 * @param minimum Minimum live total key in the current bounded bank.
 * @param maximum Maximum live total key in the same bank.
 * @return First differing byte shift in {0,8,...,56}; keys are unique by original ID.
 */
QSA_BOUNDED_HD int firstShift(std::uint64_t minimum, std::uint64_t maximum)
{
    const auto different = minimum ^ maximum;
    int shift = 56;
    while (shift > 0 && ((different >> shift) & 255u) == 0) shift -= 8;
    return shift;
}

/**
 * @brief Construct the already resolved prefix without a word-width shift.
 * @param shift Checked radix byte shift in {0,8,...,56}, derived by firstShift/descent.
 * @return Mask of bytes strictly above the current digit.
 */
QSA_BOUNDED_HD std::uint64_t prefixMask(int shift)
{
    return shift == 56 ? 0 : UINT64_MAX << (shift + 8);
}

/**
 * @brief Start one lexical row; dirty key/histogram banks are overwritten before reads.
 * @param storage Exclusive automatic/shared row storage, never a persistent authority.
 */
template<int Block, int Slack>
QSA_BOUNDED_HD void begin(Storage<Block, Slack> &storage)
{
    storage.size = 0;
    storage.active = 0;
    storage.threshold = 0;
    storage.observation = {0, 0, 0, 0};
}

/**
 * @brief Exact scalar counterpart of bounded radix threshold and stable compaction.
 * @param storage Actual lexical bank, with unique keys in original-ID order.
 * @return False for malformed size or a non-512 retained frontier.
 * @details Histogram counts are bounded integers. A full selected bucket can
 * terminate early; stable compaction then computes the TRUE retained minimum,
 * preserving an exact monotone threshold even when low radix bytes were skipped.
 */
template<int Block, int Slack>
inline bool compactScalar(Storage<Block, Slack> &storage)
{
    if (storage.size < Keep || storage.size > storage.Capacity) return false;
    auto *source = storage.keys[storage.active];
    auto *target = storage.keys[storage.active ^ 1];
    std::uint64_t minimum = UINT64_MAX, maximum = 0;
    for (int i = 0; i < storage.size; ++i)
    {
        minimum = source[i] < minimum ? source[i] : minimum;
        maximum = source[i] > maximum ? source[i] : maximum;
    }
    int shift = firstShift(minimum, maximum);
    std::uint64_t prefix = maximum & prefixMask(shift);
    int rank = Keep;
    for (;; shift -= 8)
    {
        for (auto &count : storage.histogram) count = 0;
        const auto mask = prefixMask(shift);
        for (int i = 0; i < storage.size; ++i)
            if ((source[i] & mask) == prefix)
                ++storage.histogram[(source[i] >> shift) & 255u];
        ++storage.observation.radix_passes;
        int bucket = 255;
        for (; bucket >= 0; --bucket)
        {
            const auto count = static_cast<int>(storage.histogram[bucket]);
            if (rank <= count) break;
            rank -= count;
        }
        if (bucket < 0) return false;
        prefix |= std::uint64_t(bucket) << shift;
        if (rank == static_cast<int>(storage.histogram[bucket]) || shift == 0) break;
    }
    int written = 0;
    minimum = UINT64_MAX;
    for (int i = 0; i < storage.size; ++i)
    {
        const auto value = source[i];
        if (value < prefix) continue;
        if (written == Keep) return false;
        target[written++] = value;
        minimum = value < minimum ? value : minimum;
    }
    if (written != Keep || minimum < storage.threshold) return false;
    ++storage.observation.compactions;
    storage.observation.compacted_keys += static_cast<std::uint64_t>(storage.size);
    storage.threshold = minimum;
    storage.size = Keep;
    storage.active ^= 1;
    return true;
}
/**
 * @brief Scalar scan/filter/compact implementation shared by actual CPU selection.
 * @param scores Immutable, consumed complete-pool FP32 scores.
 * @param blocks Number of complete pools, strictly greater than Keep.
 * @param selected Exclusive output with Keep original pool IDs in ascending order.
 * @param observation Optional exclusive diagnostic row, never a traffic receipt.
 * @return False for an invalid score domain or malformed bounded frontier.
 */
template<int Block, int Slack>
inline bool selectScalar(const float *scores, int blocks, std::int32_t *selected,
                         QwenSparseSelectionObservation *observation)
{
    Storage<Block, Slack> storage;
    begin(storage);
    for (int base = 0; base < blocks; base += Block)
    {
        const int end = blocks - base < Block ? blocks : base + Block;
        for (int id = base; id < end; ++id)
        {
            const auto bits = detail::floatBits(scores[id]);
            if (!validScoreBits(bits)) return false;
            const auto value = key(bits, static_cast<std::uint32_t>(id));
            if (value > storage.threshold)
            {
                if (storage.size == storage.Capacity) return false;
                storage.keys[storage.active][storage.size++] = value;
            }
            ++storage.observation.score_elements;
        }
        if (storage.size >= Keep + Slack && !compactScalar(storage)) return false;
    }
    if (storage.size > Keep && !compactScalar(storage)) return false;
    if (storage.size != Keep) return false;
    for (int i = 0; i < Keep; ++i)
        selected[i] = originalId(storage.keys[storage.active][i]);
    if (observation) *observation = storage.observation;
    return true;
}
} // namespace llaminar2::qwen4exp::bounded_selection

#undef QSA_BOUNDED_HD
