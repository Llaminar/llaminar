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
#include <climits>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define QSA_BOUNDED_HD __host__ __device__ __forceinline__
#else
#define QSA_BOUNDED_HD inline
#endif

namespace llaminar2::qwen4exp::bounded_selection
{
inline constexpr int Keep = QwenSparseBlocks;
inline constexpr int Radix = 256;


/**
 * @brief Proven row-body work bounds for the score-only radix implementation.
 * @details These are mathematical source-work counts, never observed traffic,
 * latency predictions or a dispatch policy. Output copying and inactive-row
 * retirement are outside this bounded selection-body scope.
 */
struct WorkBound final
{
    std::uint64_t score_elements = 0; ///< Exact live score reads; zero when every pool is selected.
    std::uint64_t source_scan_rounds = 0; ///< Exact 256-thread source filtering rounds.
    std::uint64_t compactions_max = 0; ///< Tight maximum over valid score arrangements.
    std::uint64_t compacted_keys_max = 0; ///< Tight summed compaction bank sizes.
    std::uint64_t score_radix_passes_max = 0; ///< At most four score digits per compaction; may be zero in execution.
    std::uint64_t score_radix_key_visits_max = 0; ///< Bank-key reads inside histogram passes, not bytes or atomic counts.
    std::uint64_t compact_output_rounds_max = 0; ///< Tight maximum for stable tie-compaction rounds over aligned source segments.
    std::uint64_t body_barriers_max = 0; ///< Bounded selection-body CTA barriers, excluding output copying.
    /** @return Exact fieldwise source-bound equality, useful for admission witnesses. */
    constexpr bool operator==(const WorkBound&) const = default;
};

/**
 * @brief Bound exact supported selector geometries without reading any scores.
 * @param policy One of the three explicit bounded implementations.
 * @param completePools Nonnegative complete pools within native signed-int token geometry.
 * @param result Published only on success; invalid admission leaves every field unchanged.
 * @return False for another policy, negative pools, or pools beyond floor(INT_MAX/4).
 * @details All supported Block values are multiples of256, and each Slack
 * is a whole number of Blocks (or one Block for B1024). Initially every valid total key passes, so first regular compaction
 * is after1024 pools for B256/S512 and B512/S512, or2048 for B1024/S1024.
 * Each later regular call requires at least Slack new arrivals; a last partial
 * group can instead force one final call. Ascending scores attain the count and
 * W=N+512*(C-1). Every regular call ends at a whole Block, and Blocks
 * contain whole256-thread rounds. For arrivals A_j since the prior compact,
 * bank sizes are A_1 and512+A_j. Disjoint source segments L_j bound A_j;
 * only the final segment may be partial, so sum ceil(A_j/256)<=ceil(N/256).
 * Hence compact output rounds <=ceil(N/256)+2*(C-1), attained by ascending
 * scores. A skipped redundant final call only omits rejected-only trailing
 * source arrivals. Four score digits then bound histogram/compaction work. Admission bounds every intermediate well below
 * uint64 overflow; this proves selector arithmetic, not all extreme-context
 * native traversals. It does not choose a kernel or advertise an economic win.
 */
QSA_BOUNDED_HD constexpr bool workBound(QwenSparsePoolSelection policy,
    std::int64_t completePools, WorkBound& result) noexcept
{
    static_assert(Keep == 512 && Radix == 256 && QwenSparsePool == 4);
    if (completePools < 0 || completePools > INT_MAX / QwenSparsePool)
        return false;
    struct Geometry final { std::uint64_t block, slack; };
    constexpr Geometry b256{256,512}, b512{512,512}, b1024{1024,1024};
    static_assert(b256.slack % b256.block == 0 && b512.slack % b512.block == 0 &&
        b1024.slack % b1024.block == 0);
    static_assert(b256.block % 256 == 0 && b512.block % 256 == 0 && b1024.block % 256 == 0);
    Geometry geometry{};
    switch (policy)
    {
    case QwenSparsePoolSelection::Bounded256_512: geometry = b256; break;
    case QwenSparsePoolSelection::Bounded512_512: geometry = b512; break;
    case QwenSparsePoolSelection::Bounded1024_1024: geometry = b1024; break;
    default: return false;
    }
    const auto trigger = std::uint64_t(Keep) + geometry.slack;
    const auto first = (trigger / geometry.block + (trigger % geometry.block != 0)) * geometry.block;
    const auto period = geometry.slack; // Proven whole-block interval by the static assertions above.
    WorkBound bound;
    const auto pools = static_cast<std::uint64_t>(completePools);
    if (pools > static_cast<std::uint64_t>(Keep))
    {
        const auto remaining = pools > first ? pools - first : 0;
        bound.score_elements = pools;
        bound.source_scan_rounds = pools / 256 + (pools % 256 != 0);
        bound.compactions_max = 1 + remaining / period + (remaining % period != 0);
        bound.compacted_keys_max = pools + std::uint64_t(Keep) * (bound.compactions_max - 1);
        bound.score_radix_passes_max = 4 * bound.compactions_max;
        bound.score_radix_key_visits_max = 4 * bound.compacted_keys_max;
        bound.compact_output_rounds_max = bound.source_scan_rounds +
            (std::uint64_t(Keep) / 256) * (bound.compactions_max - 1);
        // Per compact: 2 min/max + 3 per digit + 1 pivot + 4 per
        // tie-compaction round + 2 retained-minimum publication barriers.
        bound.body_barriers_max = 1 + 3 * bound.source_scan_rounds + 5 * bound.compactions_max +
            3 * bound.score_radix_passes_max + 4 * bound.compact_output_rounds_max;
    }
    result = bound;
    return true;
}

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
 * @brief Canonicalize only signed-zero encodings without changing any nonzero sign/domain.
 * @param bits Original FP32 score bits; native score producers publish finite nonnegative values.
 * @return Positive zero for either zero, otherwise the original unchanged encoding.
 * @details This is not validation: callers retain validScoreBits and their existing
 * failure contract. In particular a negative nonzero, NaN or infinity is never repaired.
 */
QSA_BOUNDED_HD std::uint32_t canonicalScoreBits(std::uint32_t bits)
{
    return (bits & 0x7fffffffu) == 0 ? 0u : bits;
}

/**
 * @brief Preserve FP32 value ordering and canonical lower original ID for ties.
 * @param bits Original immutable FP32 score encoding, already domain-validated.
 * @param id Original nonnegative complete-pool index, never a compact ordinal.
 * @return Unique total order; larger is better. Signed zeros compare numerically equal.
 */
QSA_BOUNDED_HD std::uint64_t key(std::uint32_t bits, std::uint32_t id)
{
    bits = canonicalScoreBits(bits);
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
 * @details Radix visits score bytes only (shifts 56 through 32). Banks remain
 * in ascending original-ID order, so a partial cutoff-score tie retains its
 * earliest rank members without visiting ID bytes. Full-bucket termination
 * remains valid at a coarse score lower bound. Stable compaction still computes
 * the true minimum complete 64-bit key, preserving the exact scan threshold.
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
    // Equal score words differ only in their ID suffix. No histogram is needed.
    std::uint64_t prefix = shift < 32 ? minimum & 0xffffffff00000000ULL : maximum & prefixMask(shift);
    int rank = Keep;
    for (; shift >= 32; shift -= 8)
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
        if (rank == static_cast<int>(storage.histogram[bucket]) || shift == 32) break;
    }
    int written = 0, tie_seen = 0;
    minimum = UINT64_MAX;
    for (int i = 0; i < storage.size; ++i)
    {
        const auto value = source[i];
        const auto score = value & 0xffffffff00000000ULL;
        const bool equal = score == prefix;
        const bool keep = score > prefix || (equal && tie_seen < rank);
        tie_seen += equal;
        if (!keep) continue;
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
