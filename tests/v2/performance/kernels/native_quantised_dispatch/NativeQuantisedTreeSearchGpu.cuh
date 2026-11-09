/**
 * @file NativeQuantisedTreeSearchGpu.cuh
 * @brief Device-resident bounded beam search shared by CUDA and ROCm.
 *
 * This implementation consumes one complete generic-policy fold. The two
 * canonical fitting matrices already live in the scorer session; feature
 * partitions and later measured objective surfaces are uploaded once for the
 * search transaction. Every breadth-first frontier, legal split expansion,
 * exact structural deduplication, and deterministic top-64 selection remains
 * on the selected accelerator until the best tree for each leaf budget is
 * copied back to the caller.
 *
 * Development refinement can grow a transfer domain beyond one machine word.
 * ABI v12 therefore carries fixed eight-word masks for up to 512 points and
 * shape groups while preserving the same fully device-resident search. The
 * capacity is the next power of two above the 444-point collapsed-aspect CPU
 * decode domain measured in the production corpus. There is no hidden host
 * path for larger folds within that reviewed graph-allocation bound.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

namespace {

constexpr std::uint32_t kNativeQuantisedTreeBeamWidth = 64;
constexpr std::uint32_t kNativeQuantisedTreeThreads = 256;
#if defined(LLAMINAR_NATIVE_QUANTISED_SCORER_ROCM)
constexpr std::uint32_t kNativeQuantisedTreeEnumerationThreads = 64;
#else
constexpr std::uint32_t kNativeQuantisedTreeEnumerationThreads = 256;
#endif
constexpr std::uint32_t kNativeQuantisedTreeEvaluationThreads = 64;
constexpr std::uint32_t kNativeQuantisedLeafCandidateGroups = 8;
constexpr std::uint32_t kNativeQuantisedLeafGroupThreads = 32;
constexpr std::uint32_t kNativeQuantisedExpandedHashGroups = 8;
constexpr std::uint32_t kNativeQuantisedHeldoutExactWarpsPerBlock = 4;
constexpr std::uint32_t kNativeQuantisedTreeMaximumAxes = 72;
constexpr std::uint32_t kNativeQuantisedTreeSplitSignatureWords = 10;
constexpr std::uint32_t kNativeQuantisedTreeMaximumSignatureTokens =
    kLlaminarNativeQuantisedTreeMaximumPoints
    + 13U * kLlaminarNativeQuantisedTreeMaximumLeaves - 10U;
constexpr std::uint32_t kNativeQuantisedTreeSelectionChunk = 256;
constexpr std::uint32_t kNativeQuantisedTreeSelectionRetained = 64;
constexpr std::size_t kNativeQuantisedTreeMaximumCachedGraphs = 64;
constexpr std::uint32_t kNativeQuantisedLeafP95TailIterations =
    (5U * kLlaminarNativeQuantisedTreeMaximumPoints) / 100U + 1U;
constexpr std::uint32_t kNativeQuantisedTreeInvalidIndex =
    std::numeric_limits<std::uint32_t>::max();
constexpr std::uint32_t kNativeQuantisedInvalidThreshold =
    std::numeric_limits<std::uint32_t>::max();

using NativeQuantisedPointMask = LlaminarNativeQuantisedTreePointMask;

/** Return whether a fixed-width point mask contains no selected bit. */
__host__ __device__ __forceinline__ bool nativeQuantisedMaskEmpty(
    const NativeQuantisedPointMask& mask) {
    for (std::uint32_t word = 0;
         word < kLlaminarNativeQuantisedTreePointMaskWords; ++word) {
        if (mask.words[word] != 0) {
            return false;
        }
    }
    return true;
}

/** Return exact equality across every point-mask word. */
__host__ __device__ __forceinline__ bool nativeQuantisedMaskEqual(
    const NativeQuantisedPointMask& left,
    const NativeQuantisedPointMask& right) {
    for (std::uint32_t word = 0;
         word < kLlaminarNativeQuantisedTreePointMaskWords; ++word) {
        if (left.words[word] != right.words[word]) {
            return false;
        }
    }
    return true;
}

/** Return the word-wise intersection of two point masks. */
__host__ __device__ __forceinline__ NativeQuantisedPointMask nativeQuantisedMaskAnd(
    const NativeQuantisedPointMask& left,
    const NativeQuantisedPointMask& right) {
    NativeQuantisedPointMask result = {};
    for (std::uint32_t word = 0;
         word < kLlaminarNativeQuantisedTreePointMaskWords; ++word) {
        result.words[word] = left.words[word] & right.words[word];
    }
    return result;
}

/** Return the word-wise exclusive difference of two point masks. */
__host__ __device__ __forceinline__ NativeQuantisedPointMask nativeQuantisedMaskXor(
    const NativeQuantisedPointMask& left,
    const NativeQuantisedPointMask& right) {
    NativeQuantisedPointMask result = {};
    for (std::uint32_t word = 0;
         word < kLlaminarNativeQuantisedTreePointMaskWords; ++word) {
        result.words[word] = left.words[word] ^ right.words[word];
    }
    return result;
}

/** Return whether one logical point is selected by a fixed-width mask. */
__host__ __device__ __forceinline__ bool nativeQuantisedMaskContains(
    const NativeQuantisedPointMask& mask,
    std::uint32_t point) {
    return (mask.words[point >> 6U]
            & (std::uint64_t{1} << (point & 63U))) != 0;
}

/** Count every selected point across the fixed-width mask. */
__device__ __forceinline__ std::uint32_t nativeQuantisedMaskPopcount(
    const NativeQuantisedPointMask& mask) {
    std::uint32_t result = 0;
    for (std::uint32_t word = 0;
         word < kLlaminarNativeQuantisedTreePointMaskWords; ++word) {
        result += static_cast<std::uint32_t>(__popcll(mask.words[word]));
    }
    return result;
}

/**
 * Exchange one scalar inside an exact 32-lane candidate subgroup.
 *
 * CUDA warps are already 32 lanes. ROCm executes two of these logical groups
 * in one wave64; HIP's explicit width argument keeps both reductions disjoint.
 * The scorer only compares FP64 values and adds integer multiplicities, so the
 * shuffle tree cannot perturb any floating-point sum used by the policy
 * objective.
 */
template <typename T>
__device__ __forceinline__ T nativeQuantisedSubgroupShuffleDown(
    T value, std::uint32_t delta) {
#if defined(LLAMINAR_NATIVE_QUANTISED_SCORER_CUDA)
    return __shfl_down_sync(
        0xffffffffU, value, delta, kNativeQuantisedLeafGroupThreads);
#else
    return __shfl_down(value, delta, kNativeQuantisedLeafGroupThreads);
#endif
}

/** Publish shared-memory writes within one logical 32-lane subgroup. */
__device__ __forceinline__ void nativeQuantisedSubgroupSynchronize() {
#if defined(LLAMINAR_NATIVE_QUANTISED_SCORER_CUDA)
    __syncwarp(0xffffffffU);
#else
    __builtin_amdgcn_wave_barrier();
#endif
}

/** Broadcast lane zero within one exact 32-lane candidate subgroup. */
template <typename T>
__device__ __forceinline__ T nativeQuantisedSubgroupBroadcast(T value) {
#if defined(LLAMINAR_NATIVE_QUANTISED_SCORER_CUDA)
    return __shfl_sync(0xffffffffU, value, 0, kNativeQuantisedLeafGroupThreads);
#else
    return __shfl(value, 0, kNativeQuantisedLeafGroupThreads);
#endif
}

/**
 * Return one 32-bit predicate mask for the caller's logical 32-lane subgroup.
 *
 * CUDA maps one logical subgroup to one warp. gfx906 maps two logical subgroups
 * to one wave64, so the HIP branch extracts the lower or upper 32-bit half of
 * the hardware ballot. Using the runtime-provided wave width also keeps this
 * helper correct for HIP targets configured for wave32 execution.
 */
__device__ __forceinline__ std::uint32_t nativeQuantisedSubgroupBallot(
    bool predicate) {
#if defined(LLAMINAR_NATIVE_QUANTISED_SCORER_CUDA)
    return __ballot_sync(0xffffffffU, predicate);
#else
    const std::uint64_t ballot = __ballot(predicate);
    const std::uint32_t subgroupInWave =
        (threadIdx.x % warpSize) / kNativeQuantisedLeafGroupThreads;
    return static_cast<std::uint32_t>(
        ballot >> (subgroupInWave * kNativeQuantisedLeafGroupThreads));
#endif
}

/** Return the untouched IEEE-754 representation of one double. */
__device__ __forceinline__ std::uint64_t nativeQuantisedDoubleBits(double value) {
    return static_cast<std::uint64_t>(__double_as_longlong(value));
}

/** Reconstruct one double without changing any payload bit. */
__device__ __forceinline__ double nativeQuantisedDoubleFromBits(
    std::uint64_t bits) {
    return __longlong_as_double(static_cast<long long>(bits));
}

/**
 * Map a finite IEEE-754 value to an unsigned key with numeric sort order.
 *
 * Regret p95 is an order statistic: it compares values but never performs
 * floating-point arithmetic. GA102 executes FP64 comparisons at a small
 * fraction of its integer rate, while gfx906 also benefits from moving this
 * branch-heavy work off the FP64 path. Signed zero is normalized for comparison
 * exactly as `double` equality requires; the separately retained source bits
 * still preserve the selected input representation.
 */
__device__ __forceinline__ std::uint64_t nativeQuantisedDoubleOrderKey(
    std::uint64_t bits) {
    constexpr std::uint64_t sign = std::uint64_t{1} << 63U;
    if ((bits & ~sign) == 0U) {
        bits = 0U;
    }
    return (bits & sign) != 0U ? ~bits : (bits ^ sign);
}

constexpr std::uint64_t kNativeQuantisedSignatureHashBase =
    0x9e3779b185ebca87ULL;
constexpr std::uint64_t kNativeQuantisedSignatureHashSalt =
    0xd6e8feb86659fd93ULL;

/** Map one canonical signature token to a nonzero polynomial coefficient. */
__host__ __device__ __forceinline__ std::uint64_t
nativeQuantisedSignatureTokenCoefficient(std::uint32_t token) {
    return static_cast<std::uint64_t>(token) + kNativeQuantisedSignatureHashSalt;
}

/** Append one token to an order-sensitive modulo-2^64 signature hash. */
__host__ __device__ __forceinline__ std::uint64_t
nativeQuantisedSignatureHashAppend(std::uint64_t hash, std::uint32_t token) {
    return hash * kNativeQuantisedSignatureHashBase
        + nativeQuantisedSignatureTokenCoefficient(token);
}

/** Return the rolling-hash base raised to an exact sequence length. */
__host__ __device__ __forceinline__ std::uint64_t
nativeQuantisedSignatureHashPower(std::uint32_t exponent) {
    std::uint64_t result = 1U;
    std::uint64_t factor = kNativeQuantisedSignatureHashBase;
    while (exponent != 0U) {
        if ((exponent & 1U) != 0U) {
            result *= factor;
        }
        factor *= factor;
        exponent >>= 1U;
    }
    return result;
}

/** Concatenate two normalized rolling hashes without reading their tokens. */
__host__ __device__ __forceinline__ std::uint64_t
nativeQuantisedSignatureHashCombine(
    std::uint64_t left,
    std::uint64_t right,
    std::uint32_t rightLength) {
    return left * nativeQuantisedSignatureHashPower(rightLength) + right;
}

/** Mix the sequence length into the hash-table probe value. */
__host__ __device__ __forceinline__ std::uint64_t
nativeQuantisedFinalizeSignatureHash(
    std::uint64_t hash,
    std::uint32_t length) {
    return hash ^ (
        nativeQuantisedSignatureTokenCoefficient(length)
        * 0x94d049bb133111ebULL);
}

/** Choose only as many candidate groups as the matrix can use. */
std::uint32_t nativeQuantisedLeafScoreThreadCount(std::uint32_t candidateCount) {
    std::uint32_t groups = std::max<std::uint32_t>(
        1U, std::min(candidateCount, kNativeQuantisedLeafCandidateGroups));
#if defined(LLAMINAR_NATIVE_QUANTISED_SCORER_ROCM)
    // Two 32-lane candidate groups share one gfx906 wavefront. Keep every HIP
    // block wave-aligned so the final odd group cannot create a partial wave.
    groups = (groups + 1U) & ~1U;
#endif
    return groups * kNativeQuantisedLeafGroupThreads;
}

/** Complete measured and fitting score for one selected tree leaf. */
struct NativeQuantisedDeviceLeaf {
    NativeQuantisedPointMask pointMask;
    std::uint32_t candidate;
    std::uint32_t signatureCount;
    std::uint64_t signatureHash;
    double fittingP95;
    double fittingMean;
    double measuredP95;
    double measuredMean;
    double measuredMaximum;
    std::uint32_t measuredFailureCount;
    std::uint32_t failedLeafCount;
    bool valid;
};

/** Scalar lexicographic keys stored once for each materialized tree. */
struct NativeQuantisedDeviceTreeObjective {
    std::uint32_t failedLeafCount = 0;
    double fittingP95 = kNativeQuantisedInfinity;
    double fittingMean = kNativeQuantisedInfinity;
    double worstLeafP95 = kNativeQuantisedInfinity;
    std::uint32_t measuredFailureCount = 0;
    double measuredMaximum = kNativeQuantisedInfinity;
};

/**
 * Fixed-width tree used by device sort and exact structural deduplication.
 *
 * Leaves are stored in preorder. `signature` is the flattened canonical Python
 * tuple ordering: leaf-kind zero precedes split-kind one; a split is followed
 * by canonical threshold rank and its child signatures; a leaf is followed by
 * candidate rank, point-group ranks in canonical point order, and a zero
 * terminator. Group/candidate/threshold ranks are stored one-based so the zero
 * terminator preserves Python's shorter-tuple-first comparison.
 */
struct alignas(16) NativeQuantisedDeviceTree {
    std::uint32_t leafCount = 0;
    NativeQuantisedDeviceLeaf leaves[kLlaminarNativeQuantisedTreeMaximumLeaves];
    std::uint32_t signatureCount = 0;
    std::uint32_t signature[kNativeQuantisedTreeMaximumSignatureTokens] = {};
    NativeQuantisedDeviceTreeObjective objective;
    // The beam retains only 64 trees, while one depth may score close to one
    // million expansions. Cache each survivor's complete selected-regret order
    // once so every expansion can merge unchanged parent values with only the
    // two replacement leaves. Source ranks preserve the exact subgroup tie
    // order: 16-point lane, preorder leaf, then ascending point.
    std::uint64_t fittingOrderBits[kLlaminarNativeQuantisedTreeMaximumPoints] = {};
    std::uint32_t fittingOrderSources[kLlaminarNativeQuantisedTreeMaximumPoints] = {};
    std::uint64_t leafSignaturePrefixHashes[
        kLlaminarNativeQuantisedTreeMaximumLeaves] = {};
    std::uint64_t leafSignatureSuffixHashes[
        kLlaminarNativeQuantisedTreeMaximumLeaves] = {};
};

__device__ bool nativeQuantisedLeafSignatureSpan(
    const NativeQuantisedDeviceTree& tree,
    std::uint32_t targetLeaf,
    std::uint32_t* spanBegin,
    std::uint32_t* spanEnd);

/**
 * Compact description of one legal parent-leaf split.
 *
 * Enumeration writes only these scalar fields. Leaf scoring and tree assembly
 * consume the descriptor in later kernels, which prevents the combinatorial
 * search phase from carrying a 3 KiB tree through every worker thread.
 */
struct NativeQuantisedSplitDescriptor {
    NativeQuantisedPointMask leftMask;
    NativeQuantisedPointMask rightMask;
    std::uint32_t frontierIndex;
    std::uint32_t targetLeaf;
    LlaminarNativeQuantisedTreeThreshold threshold;
};

/**
 * Device-resident expanded tree represented without copying its parent payload.
 *
 * A breadth-first depth can contain close to one million legal splits while the
 * next frontier retains at most 64 trees. Storing a complete 32-leaf tree and
 * canonical signature for every split consumed several GiB and made candidate
 * generation a memory-bandwidth operation. This record instead identifies the
 * immutable parent and split descriptor, retains the signature splice bounds,
 * and stores only the scalar objective required by top-k selection. Exact
 * signature hashing/comparison reads the virtual spliced signature directly;
 * the full tree is materialized only after a candidate survives selection.
 */
struct alignas(16) NativeQuantisedExpandedCandidate {
    std::uint32_t descriptorIndex = 0;
    std::uint32_t signatureBegin = 0;
    std::uint32_t signatureEnd = 0;
    std::uint32_t signatureCount = 0;
    std::uint64_t signatureHash = 0;
    NativeQuantisedDeviceTreeObjective objective;
};

struct NativeQuantisedTreeOrderKey;
struct NativeQuantisedTreeSelectionContext;

/** Geometry fields embedded into one captured complete-search graph. */
struct NativeQuantisedTreeGraphKey {
    std::uint32_t pointCount = 0;
    std::uint32_t candidateCount = 0;
    std::uint32_t axisCount = 0;
    std::uint32_t minShapeGroupsPerLeaf = 0;
    std::uint32_t maxLeaves = 0;
    std::uint32_t expansionCapacity = 0;
    std::uint32_t boundaryPlacement = 0;
    std::uint32_t parallelismWidth = 0;
    std::uint32_t taskMultiplier = 0;
    std::uint32_t heldoutPointCount = 0;
    bool evaluateHeldout = false;

    bool operator==(const NativeQuantisedTreeGraphKey& other) const {
        return pointCount == other.pointCount
            && candidateCount == other.candidateCount
            && axisCount == other.axisCount
            && minShapeGroupsPerLeaf == other.minShapeGroupsPerLeaf
            && maxLeaves == other.maxLeaves
            && expansionCapacity == other.expansionCapacity
            && boundaryPlacement == other.boundaryPlacement
            && parallelismWidth == other.parallelismWidth
            && taskMultiplier == other.taskMultiplier
            && heldoutPointCount == other.heldoutPointCount
            && evaluateHeldout == other.evaluateHeldout;
    }
};

/** One executable graph whose nodes reference persistent scratch addresses. */
struct NativeQuantisedTreeGraphEntry {
    NativeQuantisedTreeGraphKey key;
    NativeQuantisedScorerGraph graph = nullptr;
    NativeQuantisedScorerGraphExec executable = nullptr;
};

/**
 * Grow-only device workspace retained by one policy worker.
 *
 * The two frontier arrays ping-pong between exact leaf depths. Expanded splits
 * remain compact virtual trees, are deduplicated through an exact-signature hash
 * table, and are ranked through uint32 indices. Only the selected 64 candidates
 * become full tree payloads. Every pointer in this object remains stable until
 * an explicit high-water growth, at which point all graphs are invalidated before
 * the old address is released.
 */
struct NativeQuantisedTreeScratch {
    double* measuredMeans = nullptr;
    double* measuredMaxima = nullptr;
    std::size_t measuredMeanCapacity = 0;
    std::size_t measuredMaximumCapacity = 0;
    std::uint16_t* fittingP95PointOrder = nullptr;
    std::uint16_t* measuredP95PointOrder = nullptr;
    std::size_t fittingP95PointOrderCapacity = 0;
    std::size_t measuredP95PointOrderCapacity = 0;
    std::uint32_t* pointGroupRanks = nullptr;
    std::size_t pointRankCapacity = 0;
    std::uint64_t* trainingAggregateN = nullptr;
    std::uint64_t* trainingK = nullptr;
    std::uint32_t* trainingLaunchKTiles = nullptr;
    std::size_t trainingAggregateNCapacity = 0;
    std::size_t trainingKCapacity = 0;
    std::size_t trainingLaunchKTilesCapacity = 0;
    LlaminarNativeQuantisedTreeFeatureAxis* featureAxes = nullptr;
    std::size_t featureAxisCapacity = 0;
    std::uint32_t* axisValueCounts = nullptr;
    std::size_t axisCapacity = 0;
    NativeQuantisedPointMask* axisValueMasks = nullptr;
    NativeQuantisedPointMask* axisPrefixMasks = nullptr;
    std::size_t axisMaskCapacity = 0;
    std::size_t axisPrefixCapacity = 0;
    std::uint64_t* axisValueNumerators = nullptr;
    std::uint64_t* axisValueDenominators = nullptr;
    std::size_t axisValueNumeratorCapacity = 0;
    std::size_t axisValueDenominatorCapacity = 0;

    std::uint64_t* heldoutAggregateN = nullptr;
    std::uint64_t* heldoutK = nullptr;
    std::uint32_t* heldoutLaunchKTiles = nullptr;
    double* heldoutMeasuredP95 = nullptr;
    double* heldoutMeasuredMeans = nullptr;
    double* heldoutMeasuredMaxima = nullptr;
    std::uint32_t* heldoutExactCandidates = nullptr;
    LlaminarNativeQuantisedTreeFoldEvaluation* foldEvaluations = nullptr;
    std::size_t heldoutAggregateNCapacity = 0;
    std::size_t heldoutKCapacity = 0;
    std::size_t heldoutLaunchKTilesCapacity = 0;
    std::size_t heldoutMeasuredP95Capacity = 0;
    std::size_t heldoutMeasuredMeanCapacity = 0;
    std::size_t heldoutMeasuredMaximumCapacity = 0;
    std::size_t heldoutExactCandidateCapacity = 0;
    std::size_t foldEvaluationCapacity = 0;

    NativeQuantisedDeviceTree* frontierA = nullptr;
    NativeQuantisedDeviceTree* frontierB = nullptr;
    NativeQuantisedDeviceTree* incumbent = nullptr;
    NativeQuantisedExpandedCandidate* expanded = nullptr;
    NativeQuantisedTreeOrderKey* expandedOrderKeys = nullptr;
    NativeQuantisedSplitDescriptor* splitDescriptors = nullptr;
    NativeQuantisedPointMask* childMasks = nullptr;
    NativeQuantisedDeviceLeaf* childLeaves = nullptr;
    std::uint16_t* childFittingTailPoints = nullptr;
    std::uint32_t* childUniqueIndices = nullptr;
    std::uint32_t* maskHashRepresentatives = nullptr;
    std::uint32_t* maskHashUniqueIndices = nullptr;
    std::size_t expansionCapacity = 0;
    std::size_t expandedOrderKeyCapacity = 0;
    std::size_t splitDescriptorCapacity = 0;
    std::size_t childMaskCapacity = 0;
    std::size_t childLeafCapacity = 0;
    std::size_t childFittingTailPointCapacity = 0;
    std::size_t childUniqueIndexCapacity = 0;
    std::size_t maskHashCapacity = 0;
    std::size_t maskHashUniqueIndexCapacity = 0;
    std::size_t frontierCapacityA = 0;
    std::size_t frontierCapacityB = 0;
    std::size_t incumbentCapacity = 0;
    std::uint32_t* frontierCountA = nullptr;
    std::uint32_t* frontierCountB = nullptr;
    std::uint32_t* expandedCount = nullptr;
    std::uint32_t* splitDescriptorCount = nullptr;
    std::uint32_t* childMaskCount = nullptr;
    std::uint32_t* uniqueCount = nullptr;
    std::uint32_t* status = nullptr;
    std::uint64_t* expandedCandidateTotal = nullptr;
    std::uint64_t* structurallyUniqueCandidateTotal = nullptr;
    std::size_t frontierCountCapacityA = 0;
    std::size_t frontierCountCapacityB = 0;
    std::size_t expandedCountCapacity = 0;
    std::size_t splitDescriptorCountCapacity = 0;
    std::size_t childMaskCountCapacity = 0;
    std::size_t uniqueCountCapacity = 0;
    std::size_t statusCapacity = 0;
    std::size_t expandedCandidateTotalCapacity = 0;
    std::size_t structurallyUniqueCandidateTotalCapacity = 0;

    std::uint32_t* hashSlots = nullptr;
    std::size_t hashCapacity = 0;
    std::uint32_t* uniqueIndices = nullptr;
    std::size_t uniqueIndexCapacity = 0;
    std::uint32_t* selectionA = nullptr;
    std::uint32_t* selectionB = nullptr;
    std::uint32_t* selectionCandidates = nullptr;
    std::uint32_t* selectionCandidateCounts = nullptr;
    std::uint16_t* frontierLcpA = nullptr;
    std::uint16_t* frontierLcpB = nullptr;
    NativeQuantisedTreeSelectionContext* selectionContexts = nullptr;
    std::size_t selectionCapacityA = 0;
    std::size_t selectionCapacityB = 0;
    std::size_t selectionCandidateCapacity = 0;
    std::size_t selectionCandidateCountCapacity = 0;
    std::size_t frontierLcpCapacityA = 0;
    std::size_t frontierLcpCapacityB = 0;
    std::size_t selectionContextCapacity = 0;
    LlaminarNativeQuantisedTreeFitResult* results = nullptr;
    std::size_t resultCapacity = 0;

    std::vector<NativeQuantisedTreeGraphEntry> graphs;
};

/** Compare two finite objective scalars without changing input FP64 bits. */
__host__ __device__ bool nativeQuantisedTreeObjectiveLess(
    const NativeQuantisedDeviceTreeObjective& left,
    const NativeQuantisedDeviceTreeObjective& right) {
    if (left.failedLeafCount != right.failedLeafCount) {
        return left.failedLeafCount < right.failedLeafCount;
    }
    if (left.fittingP95 != right.fittingP95) {
        return left.fittingP95 < right.fittingP95;
    }
    if (left.fittingMean != right.fittingMean) {
        return left.fittingMean < right.fittingMean;
    }
    if (left.worstLeafP95 != right.worstLeafP95) {
        return left.worstLeafP95 < right.worstLeafP95;
    }
    if (left.measuredFailureCount != right.measuredFailureCount) {
        return left.measuredFailureCount < right.measuredFailureCount;
    }
    return left.measuredMaximum < right.measuredMaximum;
}

/** Return the three-way canonical signature comparison. */
__host__ __device__ int nativeQuantisedTreeSignatureCompare(
    const NativeQuantisedDeviceTree& left,
    const NativeQuantisedDeviceTree& right) {
    const std::uint32_t common =
        left.signatureCount < right.signatureCount
            ? left.signatureCount
            : right.signatureCount;
    for (std::uint32_t index = 0; index < common; ++index) {
        if (left.signature[index] < right.signature[index]) {
            return -1;
        }
        if (left.signature[index] > right.signature[index]) {
            return 1;
        }
    }
    if (left.signatureCount < right.signatureCount) {
        return -1;
    }
    if (left.signatureCount > right.signatureCount) {
        return 1;
    }
    return 0;
}

/** Complete canonical beam ordering, including stable tree signature. */
struct NativeQuantisedTreeObjectiveLess {
    __host__ __device__ bool operator()(
        const NativeQuantisedDeviceTree& left,
        const NativeQuantisedDeviceTree& right) const {
        if (nativeQuantisedTreeObjectiveLess(left.objective, right.objective)) {
            return true;
        }
        if (nativeQuantisedTreeObjectiveLess(right.objective, left.objective)) {
            return false;
        }
        if (left.leafCount != right.leafCount) {
            return left.leafCount < right.leafCount;
        }
        return nativeQuantisedTreeSignatureCompare(left, right) < 0;
    }
};

/** Return the measured shape-group count represented by one point mask. */
__device__ std::uint32_t nativeQuantisedShapeGroupCount(
    const NativeQuantisedPointMask& pointMask,
    const std::uint32_t* pointGroupRanks,
    std::uint32_t pointCount) {
    NativeQuantisedPointMask groups = {};
    for (std::uint32_t pointWord = 0;
         pointWord < kLlaminarNativeQuantisedTreePointMaskWords; ++pointWord) {
        std::uint64_t selected = pointMask.words[pointWord];
        while (selected != 0) {
            const std::uint32_t point = pointWord * 64U
                + static_cast<std::uint32_t>(
                    __ffsll(static_cast<long long>(selected)) - 1);
            selected &= selected - 1U;
            if (point >= pointCount) {
                continue;
            }
            const std::uint32_t group = pointGroupRanks[point];
            if (group >= kLlaminarNativeQuantisedTreeMaximumPoints) {
                continue;
            }
            groups.words[group >> 6U] |=
                std::uint64_t{1} << (group & 63U);
        }
    }
    return nativeQuantisedMaskPopcount(groups);
}

/** Compare two complete leaf candidates under the canonical objective. */
__device__ bool nativeQuantisedLeafLess(
    const NativeQuantisedDeviceLeaf& left,
    const NativeQuantisedDeviceLeaf& right) {
    if (left.failedLeafCount != right.failedLeafCount) {
        return left.failedLeafCount < right.failedLeafCount;
    }
    if (left.fittingP95 != right.fittingP95) {
        return left.fittingP95 < right.fittingP95;
    }
    if (left.fittingMean != right.fittingMean) {
        return left.fittingMean < right.fittingMean;
    }
    if (left.measuredP95 != right.measuredP95) {
        return left.measuredP95 < right.measuredP95;
    }
    if (left.measuredMean != right.measuredMean) {
        return left.measuredMean < right.measuredMean;
    }
    if (left.measuredFailureCount != right.measuredFailureCount) {
        return left.measuredFailureCount < right.measuredFailureCount;
    }
    if (left.measuredMaximum != right.measuredMaximum) {
        return left.measuredMaximum < right.measuredMaximum;
    }
    return left.candidate < right.candidate;
}

/** Copy one leaf field-by-field without materializing a private-memory struct. */
__device__ __forceinline__ void nativeQuantisedCopyLeaf(
    NativeQuantisedDeviceLeaf* destination,
    const NativeQuantisedDeviceLeaf* source) {
    destination->pointMask = source->pointMask;
    destination->candidate = source->candidate;
    destination->signatureCount = source->signatureCount;
    destination->signatureHash = source->signatureHash;
    destination->fittingP95 = source->fittingP95;
    destination->fittingMean = source->fittingMean;
    destination->measuredP95 = source->measuredP95;
    destination->measuredMean = source->measuredMean;
    destination->measuredMaximum = source->measuredMaximum;
    destination->measuredFailureCount = source->measuredFailureCount;
    destination->failedLeafCount = source->failedLeafCount;
    destination->valid = source->valid;
}

/**
 * Select exact stable nearest-rank p95 from one preordered candidate column.
 *
 * The transaction builds each candidate's numeric point order once. A logical
 * 32-lane subgroup then intersects that order with the leaf mask in 32-point
 * chunks and finds the requested selected rank through integer ballot counts.
 * At the 512-point ABI maximum this performs at most sixteen fixed chunks,
 * replacing the former worst-case twenty-six complete matrix rescans for every
 * child leaf. The order is stable by original point index, so equal numeric
 * values, including mixed signed zero, reproduce Python's stable `sorted`
 * result and retain the selected source's untouched FP64 bit pattern.
 */
__device__ __forceinline__ void nativeQuantisedSelectLeafCandidateP95(
    const std::uint16_t* pointOrder,
    const double* values,
    const NativeQuantisedPointMask& pointMask,
    std::uint32_t pointCount,
    std::uint32_t candidateCount,
    std::uint32_t candidateBase,
    std::uint32_t selectedPointCount,
    std::uint32_t activeCandidates,
    const NativeQuantisedDeviceLeaf* candidateLeaves,
    double* selectedValues) {
    const std::uint32_t group = threadIdx.x / kNativeQuantisedLeafGroupThreads;
    const std::uint32_t lane = threadIdx.x % kNativeQuantisedLeafGroupThreads;
    const bool active =
        group < activeCandidates && candidateLeaves[group].valid;
    const std::uint32_t candidate = candidateBase + group;
    const std::uint32_t requestedRank =
        static_cast<std::uint32_t>(
            (95ULL * selectedPointCount + 99ULL) / 100ULL) - 1U;
    std::uint32_t selectedBefore = 0U;
    std::uint32_t selectedPoint = kNativeQuantisedTreeInvalidIndex;
    bool selectionDone = !active;
    for (std::uint32_t orderBase = 0;
         orderBase < pointCount && !selectionDone;
         orderBase += kNativeQuantisedLeafGroupThreads) {
        const std::uint32_t orderPosition = orderBase + lane;
        const std::uint16_t point = active && orderPosition < pointCount
            ? pointOrder[static_cast<std::uint64_t>(candidate) * pointCount
                         + orderPosition]
            : std::uint16_t{0xffffU};
        const std::uint32_t selected = nativeQuantisedSubgroupBallot(
            active && point < pointCount
            && nativeQuantisedMaskContains(pointMask, point));
        if (lane == 0U) {
            const std::uint32_t chunkCount = __popc(selected);
            if (requestedRank < selectedBefore + chunkCount) {
                std::uint32_t remaining = requestedRank - selectedBefore;
                std::uint32_t remainingMask = selected;
                while (remaining != 0U) {
                    remainingMask &= remainingMask - 1U;
                    --remaining;
                }
                const std::uint32_t selectedLane =
                    static_cast<std::uint32_t>(__ffs(
                        static_cast<int>(remainingMask)) - 1);
                selectedPoint = pointOrder[
                    static_cast<std::uint64_t>(candidate) * pointCount
                    + orderBase + selectedLane];
                selectionDone = true;
            } else {
                selectedBefore += chunkCount;
            }
        }
        selectionDone = nativeQuantisedSubgroupBroadcast(
            static_cast<std::uint32_t>(selectionDone)) != 0U;
    }
    if (lane == 0U) {
        selectedValues[group] = active
            && selectedPoint < pointCount
            ? values[static_cast<std::uint64_t>(selectedPoint) * candidateCount
                     + candidate]
            : kNativeQuantisedInfinity;
    }
}

/**
 * Build stable numeric point orders for fitting and measured-p95 matrices.
 *
 * One 512-thread block owns one candidate column from one matrix. The fixed
 * bitonic network is graph-capturable and runs only once per fit transaction;
 * every child leaf at every subsequent tree depth reuses the resulting compact
 * uint16 point order. Numeric order uses the same reversible IEEE-754 key as
 * device comparisons. Original point index is the secondary key, reproducing
 * Python's stable input order for equal values without changing any FP64 bits.
 * Non-finite values may participate in this metadata sort, but a child whose
 * mask selects one is rejected by the leaf scorer before the order is consumed.
 */
__global__ void buildNativeQuantisedP95PointOrders(
    const double* fittingRegrets,
    const double* measuredP95Regrets,
    std::uint32_t pointCount,
    std::uint32_t candidateCount,
    std::uint16_t* fittingPointOrder,
    std::uint16_t* measuredPointOrder) {
    __shared__ std::uint64_t keys[kLlaminarNativeQuantisedTreeMaximumPoints];
    __shared__ std::uint32_t points[kLlaminarNativeQuantisedTreeMaximumPoints];
    const std::uint32_t lane = threadIdx.x;
    const bool measuredMatrix = blockIdx.x >= candidateCount;
    const std::uint32_t candidate = blockIdx.x % candidateCount;
    const double* matrix = measuredMatrix
        ? measuredP95Regrets : fittingRegrets;
    std::uint16_t* output = measuredMatrix
        ? measuredPointOrder : fittingPointOrder;
    if (lane < pointCount) {
        keys[lane] = nativeQuantisedDoubleOrderKey(nativeQuantisedDoubleBits(
            matrix[static_cast<std::uint64_t>(lane) * candidateCount
                   + candidate]));
        points[lane] = lane;
    } else {
        keys[lane] = ~std::uint64_t{0};
        points[lane] = kNativeQuantisedTreeInvalidIndex;
    }
    __syncthreads();

    for (std::uint32_t width = 2U;
         width <= kLlaminarNativeQuantisedTreeMaximumPoints; width <<= 1U) {
        for (std::uint32_t stride = width >> 1U; stride != 0U; stride >>= 1U) {
            const std::uint32_t partner = lane ^ stride;
            if (partner > lane) {
                const bool ascending = (lane & width) == 0U;
                const bool greater = keys[lane] > keys[partner]
                    || (keys[lane] == keys[partner]
                        && points[lane] > points[partner]);
                const bool less = keys[lane] < keys[partner]
                    || (keys[lane] == keys[partner]
                        && points[lane] < points[partner]);
                if ((ascending && greater) || (!ascending && less)) {
                    const std::uint64_t key = keys[lane];
                    const std::uint32_t point = points[lane];
                    keys[lane] = keys[partner];
                    points[lane] = points[partner];
                    keys[partner] = key;
                    points[partner] = point;
                }
            }
            __syncthreads();
        }
    }
    if (lane < pointCount) {
        output[static_cast<std::uint64_t>(candidate) * pointCount + lane] =
            static_cast<std::uint16_t>(points[lane]);
    }
}

/** Scalar candidate evidence retained while one mask tile performs p95. */
struct NativeQuantisedChildCandidateMetrics {
    double fittingMean;
    double measuredMean;
    double measuredMaximum;
    std::uint32_t measuredFailureCount;
    bool valid;
};

/**
 * Score 32 unique child masks together on wave64 accelerators.
 *
 * During ordered-mean evaluation, each logical subgroup lane owns a different
 * child mask while the subgroup identifies the candidate column. All 32 lanes
 * therefore execute independent left-to-right FP64 sums instead of leaving 31
 * lanes idle behind one candidate owner. The subsequent p95 phase restores the
 * usual mapping: one complete logical subgroup intersects a single mask with a
 * candidate's stable point order. Every floating-point operation consequently
 * retains the Python oracle's serial order; only independent masks overlap.
 *
 * Shape-group legality is also computed once per mask tile in shared memory.
 * Each lane owns one 16-word row, so no atomics, private aggregate, or cross-lane
 * ordering can affect the exact integer result.
 */
template <std::uint32_t CandidateGroups>
__global__ void scoreNativeQuantisedChildMaskTiles(
    const NativeQuantisedPointMask* childMasks,
    const std::uint32_t* childMaskCount,
    std::uint32_t childMaskCapacity,
    const double* fittingRegrets,
    const double* measuredP95Regrets,
    const double* measuredMeanRegrets,
    const double* measuredMaxRegrets,
    const std::uint16_t* fittingP95PointOrder,
    const std::uint16_t* measuredP95PointOrder,
    const std::uint32_t* pointGroupRanks,
    std::uint32_t pointCount,
    std::uint32_t candidateCount,
    std::uint32_t minShapeGroupsPerLeaf,
    NativeQuantisedDeviceLeaf* childLeaves) {
    static_assert(CandidateGroups > 0U);
    static_assert(CandidateGroups <= kNativeQuantisedLeafCandidateGroups);
    constexpr std::uint32_t masksPerTile = kNativeQuantisedLeafGroupThreads;
    constexpr std::uint32_t shapeGroupWords =
        kLlaminarNativeQuantisedTreeMaximumPoints / 32U;
    __shared__ NativeQuantisedChildCandidateMetrics
        metrics[masksPerTile][CandidateGroups];
    __shared__ NativeQuantisedDeviceLeaf maskBest[masksPerTile];
    __shared__ NativeQuantisedDeviceLeaf candidateLeaves[CandidateGroups];
    __shared__ double fittingP95[CandidateGroups];
    __shared__ double measuredP95[CandidateGroups];
    __shared__ std::uint32_t selectedPointCounts[masksPerTile];
    __shared__ bool legalMasks[masksPerTile];
    __shared__ std::uint32_t
        selectedShapeGroups[masksPerTile][shapeGroupWords];

    const std::uint32_t group =
        threadIdx.x / kNativeQuantisedLeafGroupThreads;
    const std::uint32_t lane =
        threadIdx.x % kNativeQuantisedLeafGroupThreads;
    const std::uint32_t count =
        *childMaskCount < childMaskCapacity ? *childMaskCount : childMaskCapacity;
    const std::uint64_t tileStride =
        static_cast<std::uint64_t>(gridDim.x) * masksPerTile;
    for (std::uint64_t maskBase =
             static_cast<std::uint64_t>(blockIdx.x) * masksPerTile;
         maskBase < count; maskBase += tileStride) {
        const std::uint64_t ownedMaskIndex = maskBase + lane;
        const bool ownsMask = ownedMaskIndex < count;
        if (group == 0U) {
            maskBest[lane].valid = false;
            legalMasks[lane] = false;
            selectedPointCounts[lane] = 0U;
            for (std::uint32_t word = 0; word < shapeGroupWords; ++word) {
                selectedShapeGroups[lane][word] = 0U;
            }
            if (ownsMask) {
                const NativeQuantisedPointMask& pointMask = childMasks[ownedMaskIndex];
                selectedPointCounts[lane] = nativeQuantisedMaskPopcount(pointMask);
                for (std::uint32_t word = 0;
                     word < kLlaminarNativeQuantisedTreePointMaskWords; ++word) {
                    std::uint64_t selected = pointMask.words[word];
                    while (selected != 0U) {
                        const std::uint32_t bit = static_cast<std::uint32_t>(
                            __ffsll(static_cast<long long>(selected)) - 1);
                        selected &= selected - 1U;
                        const std::uint32_t point = word * 64U + bit;
                        if (point >= pointCount) {
                            continue;
                        }
                        const std::uint32_t shapeGroup = pointGroupRanks[point];
                        if (shapeGroup
                            < kLlaminarNativeQuantisedTreeMaximumPoints) {
                            selectedShapeGroups[lane][shapeGroup >> 5U] |=
                                std::uint32_t{1} << (shapeGroup & 31U);
                        }
                    }
                }
                std::uint32_t shapeGroupCount = 0U;
                for (std::uint32_t word = 0; word < shapeGroupWords; ++word) {
                    shapeGroupCount += static_cast<std::uint32_t>(
                        __popc(selectedShapeGroups[lane][word]));
                }
                legalMasks[lane] =
                    shapeGroupCount >= minShapeGroupsPerLeaf;
                maskBest[lane].pointMask = pointMask;
            }
        }
        __syncthreads();

        for (std::uint32_t candidateBase = 0;
             candidateBase < candidateCount;
             candidateBase += CandidateGroups) {
            const std::uint32_t activeCandidates =
                candidateCount - candidateBase < CandidateGroups
                    ? candidateCount - candidateBase
                    : CandidateGroups;
            NativeQuantisedChildCandidateMetrics ownedMetrics;
            ownedMetrics.fittingMean = kNativeQuantisedInfinity;
            ownedMetrics.measuredMean = kNativeQuantisedInfinity;
            ownedMetrics.measuredMaximum = kNativeQuantisedInfinity;
            ownedMetrics.measuredFailureCount = 0U;
            ownedMetrics.valid = false;
            if (ownsMask && legalMasks[lane] && group < activeCandidates) {
                const NativeQuantisedPointMask& pointMask = childMasks[ownedMaskIndex];
                const std::uint32_t candidate = candidateBase + group;
                double fittingSum = 0.0;
                double measuredMeanSum = 0.0;
                double measuredMaximum = -kNativeQuantisedInfinity;
                std::uint32_t measuredFailureCount = 0U;
                bool valid = selectedPointCounts[lane] != 0U;
                for (std::uint32_t word = 0;
                     word < kLlaminarNativeQuantisedTreePointMaskWords && valid;
                     ++word) {
                    std::uint64_t selected = pointMask.words[word];
                    while (selected != 0U && valid) {
                        const std::uint32_t bit = static_cast<std::uint32_t>(
                            __ffsll(static_cast<long long>(selected)) - 1);
                        selected &= selected - 1U;
                        const std::uint32_t point = word * 64U + bit;
                        if (point >= pointCount) {
                            continue;
                        }
                        const std::uint64_t source =
                            static_cast<std::uint64_t>(point) * candidateCount
                            + candidate;
                        const double fitting = fittingRegrets[source];
                        const double measuredP95Value =
                            measuredP95Regrets[source];
                        const double measuredMean = measuredMeanRegrets[source];
                        const double measuredMax = measuredMaxRegrets[source];
                        if (!isfinite(fitting) || !isfinite(measuredP95Value)
                            || !isfinite(measuredMean)
                            || !isfinite(measuredMax)) {
                            valid = false;
                            break;
                        }
                        fittingSum += fitting;
                        measuredMeanSum += measuredMean;
                        measuredMaximum = measuredMax > measuredMaximum
                            ? measuredMax : measuredMaximum;
                        measuredFailureCount +=
                            measuredMax >= kNativeQuantisedRegretBudget ? 1U : 0U;
                    }
                }
                if (valid) {
                    ownedMetrics.fittingMean =
                        fittingSum / selectedPointCounts[lane];
                    ownedMetrics.measuredMean =
                        measuredMeanSum / selectedPointCounts[lane];
                    ownedMetrics.measuredMaximum = measuredMaximum;
                    ownedMetrics.measuredFailureCount = measuredFailureCount;
                    ownedMetrics.valid = true;
                }
            }
            metrics[lane][group] = ownedMetrics;
            __syncthreads();

            const std::uint32_t activeMaskCount = static_cast<std::uint32_t>(
                count - maskBase < masksPerTile ? count - maskBase : masksPerTile);
            for (std::uint32_t maskOffset = 0;
                 maskOffset < activeMaskCount; ++maskOffset) {
                if (lane == 0U) {
                    const NativeQuantisedChildCandidateMetrics& candidateMetrics =
                        metrics[maskOffset][group];
                    NativeQuantisedDeviceLeaf& candidateLeaf = candidateLeaves[group];
                    candidateLeaf.valid = group < activeCandidates
                        && legalMasks[maskOffset] && candidateMetrics.valid;
                    if (candidateLeaf.valid) {
                        candidateLeaf.pointMask = childMasks[maskBase + maskOffset];
                        candidateLeaf.candidate = candidateBase + group;
                        candidateLeaf.fittingMean = candidateMetrics.fittingMean;
                        candidateLeaf.measuredMean = candidateMetrics.measuredMean;
                        candidateLeaf.measuredMaximum =
                            candidateMetrics.measuredMaximum;
                        candidateLeaf.measuredFailureCount =
                            candidateMetrics.measuredFailureCount;
                    }
                }
                __syncthreads();
                const NativeQuantisedPointMask& pointMask =
                    childMasks[maskBase + maskOffset];
                nativeQuantisedSelectLeafCandidateP95(
                    fittingP95PointOrder, fittingRegrets, pointMask,
                    pointCount, candidateCount, candidateBase,
                    selectedPointCounts[maskOffset], activeCandidates,
                    candidateLeaves, fittingP95);
                __syncthreads();
                nativeQuantisedSelectLeafCandidateP95(
                    measuredP95PointOrder, measuredP95Regrets, pointMask,
                    pointCount, candidateCount, candidateBase,
                    selectedPointCounts[maskOffset], activeCandidates,
                    candidateLeaves, measuredP95);
                if (lane == 0U && group < activeCandidates
                    && candidateLeaves[group].valid) {
                    candidateLeaves[group].fittingP95 = fittingP95[group];
                    candidateLeaves[group].measuredP95 = measuredP95[group];
                    candidateLeaves[group].failedLeafCount =
                        measuredP95[group] >= kNativeQuantisedRegretBudget ? 1U : 0U;
                }
                __syncthreads();
                if (threadIdx.x == 0U) {
                    for (std::uint32_t candidate = 0;
                         candidate < activeCandidates; ++candidate) {
                        if (candidateLeaves[candidate].valid
                            && (!maskBest[maskOffset].valid
                                || nativeQuantisedLeafLess(
                                    candidateLeaves[candidate],
                                    maskBest[maskOffset]))) {
                            nativeQuantisedCopyLeaf(
                                &maskBest[maskOffset],
                                &candidateLeaves[candidate]);
                        }
                    }
                }
                __syncthreads();
            }
        }

        if (group == 0U && ownsMask) {
            NativeQuantisedDeviceLeaf& best = maskBest[lane];
            if (best.valid) {
                std::uint64_t signatureHash = 0U;
                signatureHash = nativeQuantisedSignatureHashAppend(signatureHash, 0U);
                signatureHash = nativeQuantisedSignatureHashAppend(
                    signatureHash, best.candidate + 1U);
                std::uint32_t signatureCount = 2U;
                for (std::uint32_t word = 0;
                     word < kLlaminarNativeQuantisedTreePointMaskWords; ++word) {
                    std::uint64_t selected = best.pointMask.words[word];
                    while (selected != 0U) {
                        const std::uint32_t bit = static_cast<std::uint32_t>(
                            __ffsll(static_cast<long long>(selected)) - 1);
                        selected &= selected - 1U;
                        const std::uint32_t point = word * 64U + bit;
                        if (point < pointCount) {
                            signatureHash = nativeQuantisedSignatureHashAppend(
                                signatureHash, pointGroupRanks[point] + 1U);
                            ++signatureCount;
                        }
                    }
                }
                signatureHash = nativeQuantisedSignatureHashAppend(signatureHash, 0U);
                best.signatureCount = signatureCount + 1U;
                best.signatureHash = signatureHash;
            }
            nativeQuantisedCopyLeaf(&childLeaves[ownedMaskIndex], &best);
        }
        __syncthreads();
    }
}
/**
 * Materialize the exact upper fitting tail once for every scored child leaf.
 *
 * Expanded-tree p95 needs at most 26 values at the 512-point ABI limit. One
 * logical 32-lane subgroup therefore walks a child mask in 16-point segments
 * and emits those values in descending numeric order with ascending point as
 * the exact within-leaf source tie-break. The compact uint16 point indices are
 * sufficient because the winning candidate and fitting matrix remain resident.
 * Paying this scan once per unique child replaces paying it once per expanded
 * parent/split occurrence, which is the much larger cardinality.
 */
__global__ void buildNativeQuantisedChildFittingTails(
    const NativeQuantisedDeviceLeaf* childLeaves,
    const std::uint32_t* childMaskCount,
    std::uint32_t childMaskCapacity,
    const double* fittingRegrets,
    std::uint32_t pointCount,
    std::uint32_t candidateCount,
    std::uint16_t* childFittingTailPoints) {
    constexpr std::uint16_t invalidPoint = 0xffffU;
    const std::uint32_t group =
        threadIdx.x / kNativeQuantisedLeafGroupThreads;
    const std::uint32_t lane =
        threadIdx.x % kNativeQuantisedLeafGroupThreads;
    const std::uint32_t groupsPerBlock =
        blockDim.x / kNativeQuantisedLeafGroupThreads;
    const std::uint32_t count = *childMaskCount < childMaskCapacity
        ? *childMaskCount : childMaskCapacity;
    const std::uint64_t first =
        static_cast<std::uint64_t>(blockIdx.x) * groupsPerBlock + group;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * groupsPerBlock;
    for (std::uint64_t child = first; child < count; child += stride) {
        const NativeQuantisedDeviceLeaf& leaf = childLeaves[child];
        if (!leaf.valid) {
            if (lane == 0U) {
                for (std::uint32_t tail = 0;
                     tail < kNativeQuantisedLeafP95TailIterations; ++tail) {
                    childFittingTailPoints[
                        child * kNativeQuantisedLeafP95TailIterations + tail] =
                        invalidPoint;
                }
            }
            continue;
        }

        std::uint64_t previousKey = nativeQuantisedDoubleOrderKey(
            nativeQuantisedDoubleBits(kNativeQuantisedInfinity));
        std::uint32_t previousPoint = 0U;
        for (std::uint32_t tail = 0;
             tail < kNativeQuantisedLeafP95TailIterations; ++tail) {
            std::uint64_t bestKey = nativeQuantisedDoubleOrderKey(
                nativeQuantisedDoubleBits(-kNativeQuantisedInfinity));
            std::uint32_t bestPoint = kNativeQuantisedTreeInvalidIndex;
            std::uint32_t segment = static_cast<std::uint32_t>(
                leaf.pointMask.words[lane >> 2U]
                >> ((lane & 3U) * 16U)) & 0xffffU;
            while (segment != 0U) {
                const std::uint32_t bit = static_cast<std::uint32_t>(
                    __ffs(static_cast<int>(segment)) - 1);
                segment &= segment - 1U;
                const std::uint32_t point = lane * 16U + bit;
                if (point >= pointCount) {
                    continue;
                }
                const std::uint64_t key = nativeQuantisedDoubleOrderKey(
                    nativeQuantisedDoubleBits(fittingRegrets[
                        static_cast<std::uint64_t>(point) * candidateCount
                        + leaf.candidate]));
                const bool followsPrevious = tail == 0U
                    || key < previousKey
                    || (key == previousKey && point > previousPoint);
                if (followsPrevious
                    && (key > bestKey
                        || (key == bestKey && point < bestPoint))) {
                    bestKey = key;
                    bestPoint = point;
                }
            }
            for (std::uint32_t delta = kNativeQuantisedLeafGroupThreads >> 1U;
                 delta != 0U; delta >>= 1U) {
                const std::uint64_t otherKey =
                    nativeQuantisedSubgroupShuffleDown(bestKey, delta);
                const std::uint32_t otherPoint =
                    nativeQuantisedSubgroupShuffleDown(bestPoint, delta);
                if (otherKey > bestKey
                    || (otherKey == bestKey && otherPoint < bestPoint)) {
                    bestKey = otherKey;
                    bestPoint = otherPoint;
                }
            }
            bestKey = nativeQuantisedSubgroupBroadcast(bestKey);
            bestPoint = nativeQuantisedSubgroupBroadcast(bestPoint);
            if (lane == 0U) {
                childFittingTailPoints[
                    child * kNativeQuantisedLeafP95TailIterations + tail] =
                    bestPoint == kNativeQuantisedTreeInvalidIndex
                    ? invalidPoint : static_cast<std::uint16_t>(bestPoint);
            }
            previousKey = bestKey;
            previousPoint = bestPoint;
        }
    }
}

/** Return nearest-rank p95 over the candidates selected by all tree leaves. */
__device__ double nativeQuantisedTreeFittingP95(
    const NativeQuantisedDeviceTree& tree,
    const double* fittingRegrets,
    std::uint32_t pointCount,
    std::uint32_t candidateCount) {
    const std::uint32_t rank =
        static_cast<std::uint32_t>((95ULL * pointCount + 99ULL) / 100ULL);
    std::uint32_t remainingFromTop = pointCount - rank;
    double exclusiveUpper = kNativeQuantisedInfinity;
    while (true) {
        double nextLargest = -kNativeQuantisedInfinity;
        std::uint32_t multiplicity = 0;
        for (std::uint32_t leaf = 0; leaf < tree.leafCount; ++leaf) {
            const auto& selected = tree.leaves[leaf];
            for (std::uint32_t point = 0; point < pointCount; ++point) {
                if (!nativeQuantisedMaskContains(selected.pointMask, point)) {
                    continue;
                }
                const double value = fittingRegrets[
                    static_cast<std::uint64_t>(point) * candidateCount
                    + selected.candidate];
                if (value >= exclusiveUpper) {
                    continue;
                }
                if (value > nextLargest) {
                    nextLargest = value;
                    multiplicity = 1;
                } else if (value == nextLargest) {
                    ++multiplicity;
                }
            }
        }
        if (remainingFromTop < multiplicity) {
            return nextLargest;
        }
        remainingFromTop -= multiplicity;
        exclusiveUpper = nextLargest;
    }
}

/** Compute every tree objective key in canonical preorder. */
__device__ void nativeQuantisedComputeTreeObjective(
    NativeQuantisedDeviceTree* tree,
    const double* fittingRegrets,
    std::uint32_t pointCount,
    std::uint32_t candidateCount) {
    NativeQuantisedDeviceTreeObjective objective;
    objective.failedLeafCount = 0;
    objective.fittingP95 = nativeQuantisedTreeFittingP95(
        *tree, fittingRegrets, pointCount, candidateCount);
    objective.fittingMean = 0.0;
    objective.worstLeafP95 = -kNativeQuantisedInfinity;
    objective.measuredFailureCount = 0;
    objective.measuredMaximum = -kNativeQuantisedInfinity;

    std::uint32_t selectedPointCount = 0;
    for (std::uint32_t leaf = 0; leaf < tree->leafCount; ++leaf) {
        const auto& selected = tree->leaves[leaf];
        objective.failedLeafCount += selected.failedLeafCount;
        objective.worstLeafP95 =
            selected.measuredP95 > objective.worstLeafP95
                ? selected.measuredP95
                : objective.worstLeafP95;
        objective.measuredFailureCount += selected.measuredFailureCount;
        objective.measuredMaximum =
            selected.measuredMaximum > objective.measuredMaximum
                ? selected.measuredMaximum
                : objective.measuredMaximum;
        for (std::uint32_t point = 0; point < pointCount; ++point) {
            if (!nativeQuantisedMaskContains(selected.pointMask, point)) {
                continue;
            }
            objective.fittingMean += fittingRegrets[
                static_cast<std::uint64_t>(point) * candidateCount
                + selected.candidate];
            ++selectedPointCount;
        }
    }
    objective.fittingMean /= selectedPointCount;
    tree->objective = objective;
}

/** Evaluate the small materialized frontier used at exact leaf depth one. */
__global__ void computeNativeQuantisedTreeObjectives(
    NativeQuantisedDeviceTree* trees,
    const std::uint32_t* treeCount,
    std::uint32_t treeCapacity,
    const double* fittingRegrets,
    std::uint32_t pointCount,
    std::uint32_t candidateCount) {
    const std::uint32_t count =
        *treeCount < treeCapacity ? *treeCount : treeCapacity;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t index =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        nativeQuantisedComputeTreeObjective(
            &trees[index], fittingRegrets, pointCount, candidateCount);
    }
}

/** Encode the source priority used by the 32-lane exact p95 reduction. */
__host__ __device__ __forceinline__ std::uint32_t
nativeQuantisedFittingOrderSource(
    std::uint32_t leafOrdinal,
    std::uint32_t point) {
    const std::uint32_t lane = point / 16U;
    return (
        lane * kLlaminarNativeQuantisedTreeMaximumLeaves + leafOrdinal)
        * kLlaminarNativeQuantisedTreeMaximumPoints + point;
}

/** Recover the preorder leaf ordinal from one fitting-order source rank. */
__host__ __device__ __forceinline__ std::uint32_t
nativeQuantisedFittingOrderLeaf(std::uint32_t source) {
    return (source / kLlaminarNativeQuantisedTreeMaximumPoints)
        % kLlaminarNativeQuantisedTreeMaximumLeaves;
}

/** Recover the absolute point index from one fitting-order source rank. */
__host__ __device__ __forceinline__ std::uint32_t
nativeQuantisedFittingOrderPoint(std::uint32_t source) {
    return source % kLlaminarNativeQuantisedTreeMaximumPoints;
}

/** Return whether one fitting-order item precedes another. */
__device__ __forceinline__ bool nativeQuantisedFittingOrderLess(
    std::uint64_t leftBits,
    std::uint32_t leftSource,
    std::uint64_t rightBits,
    std::uint32_t rightSource) {
    const std::uint64_t leftKey = nativeQuantisedDoubleOrderKey(leftBits);
    const std::uint64_t rightKey = nativeQuantisedDoubleOrderKey(rightBits);
    if (leftKey != rightKey) {
        return leftKey > rightKey;
    }
    return leftSource < rightSource;
}

/**
 * Build one exact selected-regret order for every materialized frontier tree.
 *
 * One 512-thread block owns one of at most 64 beam survivors. Each point reads
 * its selected candidate once, records the subgroup source priority used by the
 * expanded p95 implementation, and participates in a fixed bitonic network.
 * The work is paid once per survivor after gathering, then reused by every
 * legal split of every leaf in the following breadth-first depth.
 */
__global__ void buildNativeQuantisedTreeFittingOrders(
    NativeQuantisedDeviceTree* trees,
    const std::uint32_t* treeCount,
    std::uint32_t treeCapacity,
    const double* fittingRegrets,
    std::uint32_t pointCount,
    std::uint32_t candidateCount) {
    __shared__ std::uint64_t bits[kLlaminarNativeQuantisedTreeMaximumPoints];
    __shared__ std::uint32_t sources[kLlaminarNativeQuantisedTreeMaximumPoints];
    const std::uint32_t count =
        *treeCount < treeCapacity ? *treeCount : treeCapacity;
    if (blockIdx.x >= count) {
        return;
    }
    NativeQuantisedDeviceTree& tree = trees[blockIdx.x];
    const std::uint32_t point = threadIdx.x;
    std::uint64_t valueBits = nativeQuantisedDoubleBits(-kNativeQuantisedInfinity);
    std::uint32_t source = 0xffffffffU;
    if (point < pointCount) {
        for (std::uint32_t leaf = 0; leaf < tree.leafCount; ++leaf) {
            if (!nativeQuantisedMaskContains(tree.leaves[leaf].pointMask, point)) {
                continue;
            }
            valueBits = nativeQuantisedDoubleBits(fittingRegrets[
                static_cast<std::uint64_t>(point) * candidateCount
                + tree.leaves[leaf].candidate]);
            source = nativeQuantisedFittingOrderSource(leaf, point);
            break;
        }
    }
    bits[point] = valueBits;
    sources[point] = source;
    if (point == 0U) {
        for (std::uint32_t leaf = 0; leaf < tree.leafCount; ++leaf) {
            std::uint32_t spanBegin = 0U;
            std::uint32_t spanEnd = 0U;
            if (!nativeQuantisedLeafSignatureSpan(
                    tree, leaf, &spanBegin, &spanEnd)) {
                continue;
            }
            std::uint64_t prefixHash = 0U;
            for (std::uint32_t position = 0;
                 position < spanBegin; ++position) {
                prefixHash = nativeQuantisedSignatureHashAppend(
                    prefixHash, tree.signature[position]);
            }
            std::uint64_t suffixHash = 0U;
            for (std::uint32_t position = spanEnd;
                 position < tree.signatureCount; ++position) {
                suffixHash = nativeQuantisedSignatureHashAppend(
                    suffixHash, tree.signature[position]);
            }
            tree.leafSignaturePrefixHashes[leaf] = prefixHash;
            tree.leafSignatureSuffixHashes[leaf] = suffixHash;
        }
    }
    __syncthreads();

    // Sort into descending numeric order and ascending exact source priority.
    // The full 512-wide fixed network keeps graph geometry independent of the
    // active point count; invalid tail lanes carry -infinity and sort last.
    for (std::uint32_t width = 2U;
         width <= kLlaminarNativeQuantisedTreeMaximumPoints; width <<= 1U) {
        for (std::uint32_t stride = width >> 1U;
             stride != 0U; stride >>= 1U) {
            const std::uint32_t partner = point ^ stride;
            if (partner > point) {
                const bool ascending = (point & width) == 0U;
                const bool rightBeforeLeft = nativeQuantisedFittingOrderLess(
                    bits[partner], sources[partner], bits[point], sources[point]);
                const bool leftBeforeRight = nativeQuantisedFittingOrderLess(
                    bits[point], sources[point], bits[partner], sources[partner]);
                const bool swap = ascending ? rightBeforeLeft : leftBeforeRight;
                if (swap) {
                    const std::uint64_t swapBits = bits[point];
                    const std::uint32_t swapSource = sources[point];
                    bits[point] = bits[partner];
                    sources[point] = sources[partner];
                    bits[partner] = swapBits;
                    sources[partner] = swapSource;
                }
            }
            __syncthreads();
        }
    }
    if (point < pointCount) {
        tree.fittingOrderBits[point] = bits[point];
        tree.fittingOrderSources[point] = sources[point];
    }
}

/** Append one canonical leaf signature and return the next write offset. */
__device__ std::uint32_t nativeQuantisedWriteLeafSignature(
    std::uint32_t* destination,
    std::uint32_t offset,
    const NativeQuantisedDeviceLeaf& leaf,
    const std::uint32_t* pointGroupRanks,
    std::uint32_t pointCount) {
    destination[offset++] = 0;
    destination[offset++] = leaf.candidate + 1U;
    for (std::uint32_t point = 0; point < pointCount; ++point) {
        if (nativeQuantisedMaskContains(leaf.pointMask, point)) {
            destination[offset++] = pointGroupRanks[point] + 1U;
        }
    }
    destination[offset++] = 0;
    return offset;
}

/** Append one exact threshold in canonical Python tuple field order. */
__host__ __device__ std::uint32_t nativeQuantisedWriteThresholdSignature(
    std::uint32_t* destination,
    std::uint32_t offset,
    const LlaminarNativeQuantisedTreeThreshold& threshold) {
    destination[offset++] = 1U;
    destination[offset++] = threshold.axis_priority;
    destination[offset++] = static_cast<std::uint32_t>(
        threshold.numerator >> 32U);
    destination[offset++] = static_cast<std::uint32_t>(threshold.numerator);
    destination[offset++] = static_cast<std::uint32_t>(
        threshold.denominator >> 32U);
    destination[offset++] = static_cast<std::uint32_t>(threshold.denominator);
    destination[offset++] = threshold.parallelism_width;
    destination[offset++] = threshold.task_multiplier;
    destination[offset++] = threshold.operation;
    destination[offset++] = threshold.tile_width;
    return offset;
}

/** Decode one threshold from a validated split-signature position. */
__host__ __device__ LlaminarNativeQuantisedTreeThreshold
nativeQuantisedReadThresholdSignature(
    const std::uint32_t* source,
    std::uint32_t offset) {
    LlaminarNativeQuantisedTreeThreshold threshold = {};
    threshold.axis_priority = source[offset + 1U];
    threshold.numerator =
        (static_cast<std::uint64_t>(source[offset + 2U]) << 32U)
        | source[offset + 3U];
    threshold.denominator =
        (static_cast<std::uint64_t>(source[offset + 4U]) << 32U)
        | source[offset + 5U];
    threshold.parallelism_width = source[offset + 6U];
    threshold.task_multiplier = source[offset + 7U];
    threshold.operation = source[offset + 8U];
    threshold.tile_width = source[offset + 9U];
    return threshold;
}

/** Return one canonical split-signature word without materializing an array. */
__host__ __device__ std::uint32_t nativeQuantisedThresholdSignatureToken(
    const LlaminarNativeQuantisedTreeThreshold& threshold,
    std::uint32_t position) {
    switch (position) {
        case 0: return 1U;
        case 1: return threshold.axis_priority;
        case 2: return static_cast<std::uint32_t>(threshold.numerator >> 32U);
        case 3: return static_cast<std::uint32_t>(threshold.numerator);
        case 4: return static_cast<std::uint32_t>(threshold.denominator >> 32U);
        case 5: return static_cast<std::uint32_t>(threshold.denominator);
        case 6: return threshold.parallelism_width;
        case 7: return threshold.task_multiplier;
        case 8: return threshold.operation;
        case 9: return threshold.tile_width;
        default: return 0U;
    }
}

/** Compare two thresholds in their exact canonical signature-word order. */
__device__ __forceinline__ int nativeQuantisedThresholdSignatureCompare(
    const LlaminarNativeQuantisedTreeThreshold& left,
    const LlaminarNativeQuantisedTreeThreshold& right) {
#define LLAMINAR_COMPARE_THRESHOLD_FIELD(member) \
    if (left.member < right.member) { \
        return -1; \
    } \
    if (left.member > right.member) { \
        return 1; \
    }
    LLAMINAR_COMPARE_THRESHOLD_FIELD(axis_priority)
    LLAMINAR_COMPARE_THRESHOLD_FIELD(numerator)
    LLAMINAR_COMPARE_THRESHOLD_FIELD(denominator)
    LLAMINAR_COMPARE_THRESHOLD_FIELD(parallelism_width)
    LLAMINAR_COMPARE_THRESHOLD_FIELD(task_multiplier)
    LLAMINAR_COMPARE_THRESHOLD_FIELD(operation)
    LLAMINAR_COMPARE_THRESHOLD_FIELD(tile_width)
#undef LLAMINAR_COMPARE_THRESHOLD_FIELD
    return 0;
}

/** Return the next selected point from one mask in ascending order. */
__device__ __forceinline__ std::uint32_t nativeQuantisedNextSelectedPoint(
    const NativeQuantisedPointMask& pointMask,
    std::uint32_t pointCount,
    std::uint32_t* cursor) {
    std::uint32_t word = *cursor >> 6U;
    if (word >= kLlaminarNativeQuantisedTreePointMaskWords) {
        return kNativeQuantisedTreeInvalidIndex;
    }
    std::uint64_t selected = pointMask.words[word]
        & (~std::uint64_t{0} << (*cursor & 63U));
    while (selected == 0U
           && ++word < kLlaminarNativeQuantisedTreePointMaskWords) {
        selected = pointMask.words[word];
    }
    if (selected == 0U) {
        return kNativeQuantisedTreeInvalidIndex;
    }
    const std::uint32_t point = word * 64U
        + static_cast<std::uint32_t>(
            __ffsll(static_cast<long long>(selected)) - 1);
    if (point >= pointCount) {
        return kNativeQuantisedTreeInvalidIndex;
    }
    *cursor = point + 1U;
    return point;
}

/** Compare two materialized leaf signatures without rebuilding token cursors. */
__device__ __forceinline__ int nativeQuantisedLeafSignatureTokenCompare(
    const NativeQuantisedDeviceLeaf& left,
    const NativeQuantisedDeviceLeaf& right,
    const std::uint32_t* pointGroupRanks,
    std::uint32_t pointCount) {
    if (left.candidate < right.candidate) {
        return -1;
    }
    if (left.candidate > right.candidate) {
        return 1;
    }
    std::uint32_t leftCursor = 0U;
    std::uint32_t rightCursor = 0U;
    while (true) {
        const std::uint32_t leftPoint = nativeQuantisedNextSelectedPoint(
            left.pointMask, pointCount, &leftCursor);
        const std::uint32_t rightPoint = nativeQuantisedNextSelectedPoint(
            right.pointMask, pointCount, &rightCursor);
        if (leftPoint == kNativeQuantisedTreeInvalidIndex
            || rightPoint == kNativeQuantisedTreeInvalidIndex) {
            if (leftPoint == rightPoint) {
                return 0;
            }
            return leftPoint == kNativeQuantisedTreeInvalidIndex ? -1 : 1;
        }
        const std::uint32_t leftToken = pointGroupRanks[leftPoint] + 1U;
        const std::uint32_t rightToken = pointGroupRanks[rightPoint] + 1U;
        if (leftToken < rightToken) {
            return -1;
        }
        if (leftToken > rightToken) {
            return 1;
        }
    }
}

/** Locate one preorder leaf's complete canonical signature span. */
__device__ bool nativeQuantisedLeafSignatureSpan(
    const NativeQuantisedDeviceTree& tree,
    std::uint32_t targetLeaf,
    std::uint32_t* begin,
    std::uint32_t* end) {
    std::uint32_t pendingNodes = 1;
    std::uint32_t position = 0;
    std::uint32_t leafOrdinal = 0;
    while (pendingNodes != 0 && position < tree.signatureCount) {
        const std::uint32_t nodeBegin = position;
        const std::uint32_t kind = tree.signature[position++];
        --pendingNodes;
        if (kind == 1U) {
            position += kNativeQuantisedTreeSplitSignatureWords - 1U;
            pendingNodes += 2;
            continue;
        }
        ++position;
        while (position < tree.signatureCount
               && tree.signature[position] != 0U) {
            ++position;
        }
        if (position >= tree.signatureCount) {
            return false;
        }
        ++position;
        if (leafOrdinal == targetLeaf) {
            *begin = nodeBegin;
            *end = position;
            return true;
        }
        ++leafOrdinal;
    }
    return false;
}

constexpr std::uint32_t kNativeQuantisedTreeStatusExpansionOverflow = 1U << 0U;
constexpr std::uint32_t kNativeQuantisedTreeStatusHashOverflow = 1U << 1U;
constexpr std::uint32_t kNativeQuantisedTreeStatusExportFailure = 1U << 2U;
constexpr std::uint32_t kNativeQuantisedTreeStatusMaskHashOverflow = 1U << 3U;
constexpr std::uint32_t kNativeQuantisedTreeStatusThresholdOverflow = 1U << 4U;
constexpr std::uint32_t kNativeQuantisedTreeStatusEvaluationFailure = 1U << 5U;
/** Multiply two unsigned feature terms without permitting modulo wraparound. */
__device__ __forceinline__ bool nativeQuantisedCheckedMultiply(
    std::uint64_t left,
    std::uint64_t right,
    std::uint64_t* product) {
    if (left != 0 && right > ~std::uint64_t{0} / left) {
        return false;
    }
    *product = left * right;
    return true;
}

/** Add two unsigned feature terms without permitting modulo wraparound. */
__device__ __forceinline__ bool nativeQuantisedCheckedAdd(
    std::uint64_t left,
    std::uint64_t right,
    std::uint64_t* sum) {
    if (right > ~std::uint64_t{0} - left) {
        return false;
    }
    *sum = left + right;
    return true;
}

/** Return positive integer ceil-div without overflowing the dividend. */
__device__ __forceinline__ std::uint64_t nativeQuantisedCeilDiv(
    std::uint64_t value,
    std::uint64_t divisor) {
    return value / divisor + static_cast<std::uint64_t>(value % divisor != 0);
}

/** Return the exact greatest common divisor of two unsigned feature terms. */
__device__ __forceinline__ std::uint64_t nativeQuantisedGreatestCommonDivisor(
    std::uint64_t left,
    std::uint64_t right) {
    while (right != 0) {
        const std::uint64_t remainder = left % right;
        left = right;
        right = remainder;
    }
    return left;
}

/** Compare positive rational values, reporting overflow instead of rounding. */
__device__ __forceinline__ int nativeQuantisedRationalCompare(
    std::uint64_t leftNumerator,
    std::uint64_t leftDenominator,
    std::uint64_t rightNumerator,
    std::uint64_t rightDenominator,
    std::uint32_t* status) {
    std::uint64_t leftProduct = 0;
    std::uint64_t rightProduct = 0;
    if (!nativeQuantisedCheckedMultiply(
            leftNumerator, rightDenominator, &leftProduct)
        || !nativeQuantisedCheckedMultiply(
            rightNumerator, leftDenominator, &rightProduct)) {
        atomicOr(status, kNativeQuantisedTreeStatusThresholdOverflow);
        // Inputs outside the reviewed uint64 feature envelope hard-fail the
        // transaction. This deterministic order merely keeps later kernels
        // memory-safe until the terminal status copy reaches the host.
        if (leftNumerator != rightNumerator) {
            return leftNumerator < rightNumerator ? -1 : 1;
        }
        if (leftDenominator != rightDenominator) {
            return leftDenominator > rightDenominator ? -1 : 1;
        }
        return 0;
    }
    return leftProduct < rightProduct ? -1 : leftProduct > rightProduct ? 1 : 0;
}

/** Exact work spans produced by the frozen serial-K-part arithmetic policy. */
struct NativeQuantisedKPartGeometry {
    std::uint64_t blocksPerTile;
    std::uint64_t finalTileBlocks;
};

/**
 * Reconstruct production's K partition without changing its launch inventory.
 *
 * `launchKTiles == 0` is historical full-K telemetry and therefore denotes one
 * tile. An explicit tile count is never clamped: ceil division can make the
 * last producer empty, but that producer still launches and affects economy.
 * Only the final producer's work span is saturated at zero, matching the
 * production `kb_start >= K_blocks` no-op.
 */
__device__ __forceinline__ bool nativeQuantisedComputeKPartGeometry(
    std::uint64_t k,
    std::uint64_t blockWidth,
    std::uint32_t launchKTiles,
    NativeQuantisedKPartGeometry* geometry,
    std::uint32_t* status) {
    if (k == 0 || blockWidth == 0 || geometry == nullptr) {
        atomicOr(status, kNativeQuantisedTreeStatusEvaluationFailure);
        return false;
    }
    const std::uint64_t kBlocks = nativeQuantisedCeilDiv(k, blockWidth);
    const std::uint64_t kTiles = launchKTiles > 0 ? launchKTiles : 1U;
    const std::uint64_t blocksPerTile = nativeQuantisedCeilDiv(kBlocks, kTiles);
    std::uint64_t prefixBlocks = 0;
    if (!nativeQuantisedCheckedMultiply(
            kTiles - 1U, blocksPerTile, &prefixBlocks)) {
        atomicOr(status, kNativeQuantisedTreeStatusThresholdOverflow);
        return false;
    }
    const std::uint64_t remainingBlocks =
        prefixBlocks < kBlocks ? kBlocks - prefixBlocks : 0U;
    geometry->blocksPerTile = blocksPerTile;
    geometry->finalTileBlocks = remainingBlocks < blocksPerTile
        ? remainingBlocks : blocksPerTile;
    return true;
}

/** Compute one normalized runtime feature value from raw training geometry. */
__device__ bool nativeQuantisedFeatureValue(
    const LlaminarNativeQuantisedTreeFeatureAxis& axis,
    std::uint64_t aggregateN,
    std::uint64_t k,
    std::uint32_t launchKTiles,
    std::uint32_t parallelismWidth,
    std::uint32_t taskMultiplier,
    std::uint64_t* numerator,
    std::uint64_t* denominator,
    std::uint32_t* status) {
    if (aggregateN == 0 || k == 0 || parallelismWidth == 0
        || taskMultiplier == 0) {
        atomicOr(status, kNativeQuantisedTreeStatusEvaluationFailure);
        return false;
    }
    std::uint64_t resultNumerator = 0;
    std::uint64_t resultDenominator = 1;
    switch (axis.operation) {
        case kLlaminarNativeQuantisedTreeThresholdAggregateN:
            resultNumerator = aggregateN;
            break;
        case kLlaminarNativeQuantisedTreeThresholdK:
            resultNumerator = k;
            break;
        case kLlaminarNativeQuantisedTreeThresholdWorkItems:
            if (!nativeQuantisedCheckedMultiply(
                    aggregateN, k, &resultNumerator)) {
                atomicOr(status, kNativeQuantisedTreeStatusThresholdOverflow);
                return false;
            }
            break;
        case kLlaminarNativeQuantisedTreeThresholdAspectRatio:
            resultNumerator = aggregateN;
            resultDenominator = k;
            break;
        default:
            break;
    }
    if (axis.operation > kLlaminarNativeQuantisedTreeThresholdAspectRatio) {
        if (axis.tile_width == 0) {
            atomicOr(status, kNativeQuantisedTreeStatusEvaluationFailure);
            return false;
        }
        const std::uint64_t tileWidth = axis.tile_width;
        const std::uint64_t nTiles = nativeQuantisedCeilDiv(aggregateN, tileWidth);
        switch (axis.operation) {
            case kLlaminarNativeQuantisedTreeThresholdNTiles:
                resultNumerator = nTiles;
                break;
            case kLlaminarNativeQuantisedTreeThresholdKGroupsPerNTile:
                resultNumerator = k;
                if (!nativeQuantisedCheckedMultiply(
                        32U, nTiles, &resultDenominator)) {
                    atomicOr(status, kNativeQuantisedTreeStatusThresholdOverflow);
                    return false;
                }
                break;
            case kLlaminarNativeQuantisedTreeThresholdNFinalTile:
                resultNumerator = (aggregateN - 1U) % tileWidth + 1U;
                break;
            case kLlaminarNativeQuantisedTreeThresholdNTileUtilization:
                resultNumerator = aggregateN;
                if (!nativeQuantisedCheckedMultiply(
                        nTiles, tileWidth, &resultDenominator)) {
                    atomicOr(status, kNativeQuantisedTreeStatusThresholdOverflow);
                    return false;
                }
                break;
            case kLlaminarNativeQuantisedTreeThresholdNTileAligned:
                resultNumerator = aggregateN % tileWidth == 0 ? 1U : 0U;
                break;
            case kLlaminarNativeQuantisedTreeThresholdKFinalTile:
                resultNumerator = (k - 1U) % tileWidth + 1U;
                break;
            case kLlaminarNativeQuantisedTreeThresholdNParallelWaves:
                resultNumerator = nativeQuantisedCeilDiv(
                    nTiles, parallelismWidth);
                break;
            case kLlaminarNativeQuantisedTreeThresholdNFinalParallelWaveUtilization:
                resultNumerator = (nTiles - 1U) % parallelismWidth + 1U;
                resultDenominator = parallelismWidth;
                break;
            case kLlaminarNativeQuantisedTreeThresholdKPartProducerWaves: {
                const std::uint64_t kTiles = launchKTiles > 0 ? launchKTiles : 1U;
                std::uint64_t tasks = 0;
                if (!nativeQuantisedCheckedMultiply(kTiles, nTiles, &tasks)) {
                    atomicOr(status, kNativeQuantisedTreeStatusThresholdOverflow);
                    return false;
                }
                resultNumerator = nativeQuantisedCeilDiv(tasks, parallelismWidth);
                break;
            }
            case kLlaminarNativeQuantisedTreeThresholdKPartFinalProducerWaveUtilization: {
                const std::uint64_t kTiles = launchKTiles > 0 ? launchKTiles : 1U;
                std::uint64_t tasks = 0;
                if (!nativeQuantisedCheckedMultiply(kTiles, nTiles, &tasks)) {
                    atomicOr(status, kNativeQuantisedTreeStatusThresholdOverflow);
                    return false;
                }
                resultNumerator = (tasks - 1U) % parallelismWidth + 1U;
                resultDenominator = parallelismWidth;
                break;
            }
            case kLlaminarNativeQuantisedTreeThresholdKPartKBlocksPerTile:
            case kLlaminarNativeQuantisedTreeThresholdKPartFinalKTileBlocks:
            case kLlaminarNativeQuantisedTreeThresholdKPartFinalKTileUtilization: {
                NativeQuantisedKPartGeometry geometry = {};
                if (!nativeQuantisedComputeKPartGeometry(
                        k, tileWidth, launchKTiles, &geometry, status)) {
                    return false;
                }
                if (axis.operation
                    == kLlaminarNativeQuantisedTreeThresholdKPartKBlocksPerTile) {
                    resultNumerator = geometry.blocksPerTile;
                } else if (axis.operation
                           == kLlaminarNativeQuantisedTreeThresholdKPartFinalKTileBlocks) {
                    resultNumerator = geometry.finalTileBlocks;
                } else {
                    resultNumerator = geometry.finalTileBlocks;
                    resultDenominator = geometry.blocksPerTile;
                }
                break;
            }
            case kLlaminarNativeQuantisedTreeThresholdKPartKTileCount:
                resultNumerator = launchKTiles > 0 ? launchKTiles : 1U;
                break;
            case kLlaminarNativeQuantisedTreeThresholdMNParallelWaves: {
                std::uint64_t tasks = 0;
                if (!nativeQuantisedCheckedMultiply(
                        taskMultiplier, nTiles, &tasks)) {
                    atomicOr(status, kNativeQuantisedTreeStatusThresholdOverflow);
                    return false;
                }
                resultNumerator = nativeQuantisedCeilDiv(tasks, parallelismWidth);
                break;
            }
            case kLlaminarNativeQuantisedTreeThresholdMNFinalParallelWaveUtilization: {
                std::uint64_t tasks = 0;
                if (!nativeQuantisedCheckedMultiply(
                        taskMultiplier, nTiles, &tasks)) {
                    atomicOr(status, kNativeQuantisedTreeStatusThresholdOverflow);
                    return false;
                }
                resultNumerator = (tasks - 1U) % parallelismWidth + 1U;
                resultDenominator = parallelismWidth;
                break;
            }
            default:
                atomicOr(status, kNativeQuantisedTreeStatusEvaluationFailure);
                return false;
        }
    }
    const std::uint64_t divisor = nativeQuantisedGreatestCommonDivisor(
        resultNumerator, resultDenominator);
    *numerator = resultNumerator / divisor;
    *denominator = resultDenominator / divisor;
    return true;
}

/**
 * Build sorted exact feature values and masks entirely inside the graph.
 *
 * One 512-thread block owns one feature axis. Bitonic sorting has fixed launch
 * geometry and therefore remains graph-capturable. The final serial compaction
 * is only 512 entries per axis and writes the exact stable value/prefix surfaces
 * consumed by every subsequent tree depth.
 */
__global__ void buildNativeQuantisedAxisMetadata(
    const std::uint64_t* aggregateN,
    const std::uint64_t* k,
    const std::uint32_t* launchKTiles,
    const LlaminarNativeQuantisedTreeFeatureAxis* axes,
    std::uint32_t pointCount,
    std::uint32_t parallelismWidth,
    std::uint32_t taskMultiplier,
    std::uint32_t* valueCounts,
    NativeQuantisedPointMask* valueMasks,
    NativeQuantisedPointMask* prefixMasks,
    std::uint64_t* valueNumerators,
    std::uint64_t* valueDenominators,
    std::uint32_t* status) {
    __shared__ std::uint64_t numerators[
        kLlaminarNativeQuantisedTreeMaximumPoints];
    __shared__ std::uint64_t denominators[
        kLlaminarNativeQuantisedTreeMaximumPoints];
    __shared__ std::uint32_t points[
        kLlaminarNativeQuantisedTreeMaximumPoints];
    const std::uint32_t axis = blockIdx.x;
    const std::uint32_t lane = threadIdx.x;
    if (lane < pointCount) {
        std::uint64_t numerator = 0;
        std::uint64_t denominator = 1;
        (void)nativeQuantisedFeatureValue(
            axes[axis], aggregateN[lane], k[lane], launchKTiles[lane],
            parallelismWidth, taskMultiplier, &numerator, &denominator,
            status);
        numerators[lane] = numerator;
        denominators[lane] = denominator;
        points[lane] = lane;
    } else {
        numerators[lane] = ~std::uint64_t{0};
        denominators[lane] = 1;
        points[lane] = kNativeQuantisedTreeInvalidIndex;
    }
    __syncthreads();

    for (std::uint32_t width = 2;
         width <= kLlaminarNativeQuantisedTreeMaximumPoints; width <<= 1U) {
        for (std::uint32_t stride = width >> 1U; stride != 0; stride >>= 1U) {
            const std::uint32_t partner = lane ^ stride;
            if (partner > lane) {
                int comparison = 0;
                if (points[lane] == kNativeQuantisedTreeInvalidIndex) {
                    comparison = points[partner]
                            == kNativeQuantisedTreeInvalidIndex
                        ? 0 : 1;
                } else if (points[partner]
                           == kNativeQuantisedTreeInvalidIndex) {
                    comparison = -1;
                } else {
                    comparison = nativeQuantisedRationalCompare(
                        numerators[lane], denominators[lane],
                        numerators[partner], denominators[partner], status);
                }
                const bool ascending = (lane & width) == 0;
                const bool exchange = ascending
                    ? (comparison > 0
                       || (comparison == 0 && points[lane] > points[partner]))
                    : (comparison < 0
                       || (comparison == 0 && points[lane] < points[partner]));
                if (exchange) {
                    const std::uint64_t numerator = numerators[lane];
                    const std::uint64_t denominator = denominators[lane];
                    const std::uint32_t point = points[lane];
                    numerators[lane] = numerators[partner];
                    denominators[lane] = denominators[partner];
                    points[lane] = points[partner];
                    numerators[partner] = numerator;
                    denominators[partner] = denominator;
                    points[partner] = point;
                }
            }
            __syncthreads();
        }
    }

    if (lane == 0) {
        const std::size_t base =
            static_cast<std::size_t>(axis) * pointCount;
        for (std::uint32_t rank = 0; rank < pointCount; ++rank) {
            valueMasks[base + rank] = {};
            prefixMasks[base + rank] = {};
            valueNumerators[base + rank] = 0;
            valueDenominators[base + rank] = 1;
        }
        std::uint32_t rank = 0;
        for (std::uint32_t position = 0; position < pointCount; ++position) {
            if (position != 0
                && nativeQuantisedRationalCompare(
                    numerators[position - 1U], denominators[position - 1U],
                    numerators[position], denominators[position], status) != 0) {
                ++rank;
            }
            if (position == 0
                || nativeQuantisedMaskEmpty(valueMasks[base + rank])) {
                valueNumerators[base + rank] = numerators[position];
                valueDenominators[base + rank] = denominators[position];
            }
            const std::uint32_t point = points[position];
            valueMasks[base + rank].words[point >> 6U] |=
                std::uint64_t{1} << (point & 63U);
        }
        valueCounts[axis] = rank + 1U;
        NativeQuantisedPointMask prefix = {};
        for (std::uint32_t value = 0; value < pointCount; ++value) {
            if (value < valueCounts[axis]) {
                for (std::uint32_t word = 0;
                     word < kLlaminarNativeQuantisedTreePointMaskWords; ++word) {
                    prefix.words[word] |= valueMasks[base + value].words[word];
                }
            }
            prefixMasks[base + value] = prefix;
        }
    }
}

/** Construct one normalized split threshold from two sorted feature values. */
__device__ bool nativeQuantisedBuildThreshold(
    const LlaminarNativeQuantisedTreeFeatureAxis& axis,
    std::uint64_t lowerNumerator,
    std::uint64_t lowerDenominator,
    std::uint64_t upperNumerator,
    std::uint64_t upperDenominator,
    std::uint32_t boundaryPlacement,
    std::uint32_t parallelismWidth,
    std::uint32_t taskMultiplier,
    LlaminarNativeQuantisedTreeThreshold* threshold,
    std::uint32_t* status) {
    std::uint64_t numerator = lowerNumerator;
    std::uint64_t denominator = lowerDenominator;
    if (boundaryPlacement == kLlaminarNativeQuantisedTreeBoundaryMidpoint) {
        std::uint64_t left = 0;
        std::uint64_t right = 0;
        std::uint64_t product = 0;
        if (!nativeQuantisedCheckedMultiply(
                lowerNumerator, upperDenominator, &left)
            || !nativeQuantisedCheckedMultiply(
                upperNumerator, lowerDenominator, &right)
            || !nativeQuantisedCheckedAdd(left, right, &numerator)
            || !nativeQuantisedCheckedMultiply(
                lowerDenominator, upperDenominator, &product)
            || !nativeQuantisedCheckedMultiply(2U, product, &denominator)) {
            atomicOr(status, kNativeQuantisedTreeStatusThresholdOverflow);
            return false;
        }
        const std::uint64_t divisor = nativeQuantisedGreatestCommonDivisor(
            numerator, denominator);
        numerator /= divisor;
        denominator /= divisor;
    } else if (boundaryPlacement
               != kLlaminarNativeQuantisedTreeBoundaryLowerEdge) {
        atomicOr(status, kNativeQuantisedTreeStatusEvaluationFailure);
        return false;
    }
    *threshold = {
        axis.axis_priority,
        axis.operation,
        axis.tile_width,
        numerator,
        denominator,
        (axis.operation == kLlaminarNativeQuantisedTreeThresholdNParallelWaves
         || axis.operation
            == kLlaminarNativeQuantisedTreeThresholdNFinalParallelWaveUtilization
         || axis.operation
            == kLlaminarNativeQuantisedTreeThresholdKPartProducerWaves
         || axis.operation
            == kLlaminarNativeQuantisedTreeThresholdKPartFinalProducerWaveUtilization
         || axis.operation == kLlaminarNativeQuantisedTreeThresholdMNParallelWaves
         || axis.operation
            == kLlaminarNativeQuantisedTreeThresholdMNFinalParallelWaveUtilization)
            ? parallelismWidth : 1U,
        (axis.operation == kLlaminarNativeQuantisedTreeThresholdMNParallelWaves
         || axis.operation
            == kLlaminarNativeQuantisedTreeThresholdMNFinalParallelWaveUtilization)
            ? taskMultiplier : 1U,
    };
    return true;
}

/**
 * Enumerate every legal `(frontier tree, leaf, axis, lower value)` tuple.
 *
 * `frontierCount` is device-owned, so an entire captured depth can consume the
 * prior depth without publishing cardinality to the host. The launch geometry
 * covers the bounded beam; inactive frontier slots return immediately. An
 * undersized high-water buffer sets a device status bit and the host retries the
 * whole transaction after growing it. No partial result is ever accepted.
 */
__global__ __launch_bounds__(kNativeQuantisedTreeEnumerationThreads, 1)
void enumerateNativeQuantisedSplitDescriptors(
    const NativeQuantisedDeviceTree* frontier,
    const std::uint32_t* frontierCount,
    const LlaminarNativeQuantisedTreeFeatureAxis* featureAxes,
    const std::uint32_t* axisValueCounts,
    const NativeQuantisedPointMask* axisValueMasks,
    const NativeQuantisedPointMask* axisPrefixMasks,
    const std::uint64_t* axisValueNumerators,
    const std::uint64_t* axisValueDenominators,
    std::uint32_t pointCount,
    std::uint32_t axisCount,
    std::uint32_t boundaryPlacement,
    std::uint32_t parallelismWidth,
    std::uint32_t taskMultiplier,
    NativeQuantisedSplitDescriptor* descriptors,
    std::uint32_t descriptorCapacity,
    std::uint32_t* descriptorCount,
    std::uint32_t* status) {
    __shared__ std::uint8_t occupiedValues[
        kLlaminarNativeQuantisedTreeMaximumPoints];

    // The graph launch maps these coordinates directly onto hardware grid
    // dimensions. Threads therefore spend their work on thresholds instead of
    // repeatedly dividing a flattened 64-bit combination index.
    const std::uint32_t axis = blockIdx.x;
    const std::uint32_t targetLeaf = blockIdx.y;
    const std::uint32_t frontierIndex = blockIdx.z;
    if (axis >= axisCount || frontierIndex >= *frontierCount) {
        return;
    }
    const NativeQuantisedDeviceTree& parent = frontier[frontierIndex];
    if (targetLeaf >= parent.leafCount
        || parent.leafCount >= kLlaminarNativeQuantisedTreeMaximumLeaves) {
        return;
    }
    const std::uint32_t valueCount = axisValueCounts[axis];
    const NativeQuantisedPointMask subset = parent.leaves[targetLeaf].pointMask;

    // Determine subset occupancy once per axis value. The prior implementation
    // repeated an eight-word mask intersection while every occupied value
    // searched for its next occupied neighbour. Sparse axes therefore reread
    // the same global masks many times. The shared byte map preserves the exact
    // adjacent-selected-value split set while making those later probes local.
    for (std::uint32_t value = threadIdx.x;
         value < valueCount; value += blockDim.x) {
        occupiedValues[value] = static_cast<std::uint8_t>(
            !nativeQuantisedMaskEmpty(nativeQuantisedMaskAnd(
                subset,
                axisValueMasks[
                    static_cast<std::uint64_t>(axis) * pointCount + value])));
    }
    __syncthreads();

    for (std::uint32_t lower = threadIdx.x;
         lower + 1U < valueCount; lower += blockDim.x) {
        if (occupiedValues[lower] == 0U) {
            continue;
        }
        std::uint32_t upper = lower + 1U;
        while (upper < valueCount && occupiedValues[upper] == 0U) {
            ++upper;
        }
        if (upper >= valueCount) {
            continue;
        }
        LlaminarNativeQuantisedTreeThreshold threshold = {};
        const std::size_t lowerIndex =
            static_cast<std::size_t>(axis) * pointCount + lower;
        const std::size_t upperIndex =
            static_cast<std::size_t>(axis) * pointCount + upper;
        if (!nativeQuantisedBuildThreshold(
                featureAxes[axis], axisValueNumerators[lowerIndex],
                axisValueDenominators[lowerIndex],
                axisValueNumerators[upperIndex],
                axisValueDenominators[upperIndex], boundaryPlacement,
                parallelismWidth, taskMultiplier, &threshold, status)) {
            continue;
        }
        const NativeQuantisedPointMask leftMask = nativeQuantisedMaskAnd(
            subset,
            axisPrefixMasks[
                static_cast<std::uint64_t>(axis) * pointCount + lower]);
        const NativeQuantisedPointMask rightMask =
            nativeQuantisedMaskXor(subset, leftMask);

        const std::uint32_t outputIndex = atomicAdd(descriptorCount, 1U);
        if (outputIndex >= descriptorCapacity) {
            atomicOr(status, kNativeQuantisedTreeStatusExpansionOverflow);
            continue;
        }
        descriptors[outputIndex] = {
            leftMask,
            rightMask,
            frontierIndex,
            targetLeaf,
            threshold,
        };
    }
}

/** Return one descriptor child mask by flattened child index. */
__device__ NativeQuantisedPointMask nativeQuantisedDescriptorChildMask(
    const NativeQuantisedSplitDescriptor* descriptors,
    std::uint32_t childIndex) {
    const NativeQuantisedSplitDescriptor& descriptor = descriptors[childIndex >> 1U];
    return (childIndex & 1U) == 0 ? descriptor.leftMask : descriptor.rightMask;
}

/** Mix every exact point-mask word into an open-addressed table probe. */
__device__ std::uint32_t nativeQuantisedChildMaskHash(
    const NativeQuantisedPointMask& mask) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::uint32_t word = 0;
         word < kLlaminarNativeQuantisedTreePointMaskWords; ++word) {
        hash ^= mask.words[word];
        hash *= 1099511628211ULL;
    }
    return static_cast<std::uint32_t>(hash ^ (hash >> 32U));
}

/**
 * Insert descriptor children into an exact mask set.
 *
 * Hash slots retain representative child indices. Every collision compares the
 * complete fixed-width mask, so equal hashes never merge distinct leaves. A later
 * kernel assigns dense unique indices after all representatives are visible;
 * this avoids cross-block spin waits and their forward-progress hazards.
 */
__global__ void insertNativeQuantisedChildMaskRepresentatives(
    const NativeQuantisedSplitDescriptor* descriptors,
    const std::uint32_t* descriptorCount,
    std::uint32_t descriptorCapacity,
    std::uint32_t* maskHashRepresentatives,
    std::uint32_t maskHashCapacity,
    std::uint32_t* status) {
    const std::uint32_t count =
        *descriptorCount < descriptorCapacity ? *descriptorCount : descriptorCapacity;
    const std::uint64_t childCount = static_cast<std::uint64_t>(count) * 2U;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t child =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         child < childCount; child += stride) {
        const NativeQuantisedPointMask mask = nativeQuantisedDescriptorChildMask(
            descriptors, static_cast<std::uint32_t>(child));
        std::uint32_t slot = nativeQuantisedChildMaskHash(mask)
            & (maskHashCapacity - 1U);
        bool found = false;
        for (std::uint32_t probe = 0; probe < maskHashCapacity; ++probe) {
            const std::uint32_t prior = atomicCAS(
                &maskHashRepresentatives[slot], kNativeQuantisedTreeInvalidIndex,
                static_cast<std::uint32_t>(child));
            if (prior == kNativeQuantisedTreeInvalidIndex
                || nativeQuantisedMaskEqual(
                    nativeQuantisedDescriptorChildMask(descriptors, prior),
                    mask)) {
                found = true;
                break;
            }
            slot = (slot + 1U) & (maskHashCapacity - 1U);
        }
        if (!found) {
            atomicOr(status, kNativeQuantisedTreeStatusMaskHashOverflow);
        }
    }
}

/** Assign dense indices to all occupied exact-mask hash slots. */
__global__ void compactNativeQuantisedChildMasks(
    const NativeQuantisedSplitDescriptor* descriptors,
    const std::uint32_t* maskHashRepresentatives,
    std::uint32_t* maskHashUniqueIndices,
    std::uint32_t maskHashCapacity,
    NativeQuantisedPointMask* childMasks,
    std::uint32_t childMaskCapacity,
    std::uint32_t* childMaskCount,
    std::uint32_t* status) {
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t slot =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         slot < maskHashCapacity; slot += stride) {
        const std::uint32_t representative = maskHashRepresentatives[slot];
        if (representative == kNativeQuantisedTreeInvalidIndex) {
            continue;
        }
        const std::uint32_t uniqueIndex = atomicAdd(childMaskCount, 1U);
        if (uniqueIndex >= childMaskCapacity) {
            atomicOr(status, kNativeQuantisedTreeStatusExpansionOverflow);
            continue;
        }
        childMasks[uniqueIndex] =
            nativeQuantisedDescriptorChildMask(descriptors, representative);
        maskHashUniqueIndices[slot] = uniqueIndex;
    }
}

/** Resolve every descriptor child to its dense exact-mask index. */
__global__ void mapNativeQuantisedDescriptorChildren(
    const NativeQuantisedSplitDescriptor* descriptors,
    const std::uint32_t* descriptorCount,
    std::uint32_t descriptorCapacity,
    const std::uint32_t* maskHashRepresentatives,
    const std::uint32_t* maskHashUniqueIndices,
    std::uint32_t maskHashCapacity,
    std::uint32_t* childUniqueIndices,
    std::uint32_t* status) {
    const std::uint32_t count =
        *descriptorCount < descriptorCapacity ? *descriptorCount : descriptorCapacity;
    const std::uint64_t childCount = static_cast<std::uint64_t>(count) * 2U;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t child =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         child < childCount; child += stride) {
        const NativeQuantisedPointMask mask = nativeQuantisedDescriptorChildMask(
            descriptors, static_cast<std::uint32_t>(child));
        std::uint32_t slot = nativeQuantisedChildMaskHash(mask)
            & (maskHashCapacity - 1U);
        bool found = false;
        for (std::uint32_t probe = 0; probe < maskHashCapacity; ++probe) {
            const std::uint32_t representative = maskHashRepresentatives[slot];
            if (representative == kNativeQuantisedTreeInvalidIndex) {
                break;
            }
            if (nativeQuantisedMaskEqual(
                    nativeQuantisedDescriptorChildMask(
                        descriptors, representative),
                    mask)) {
                childUniqueIndices[child] = maskHashUniqueIndices[slot];
                found = true;
                break;
            }
            slot = (slot + 1U) & (maskHashCapacity - 1U);
        }
        if (!found) {
            atomicOr(status, kNativeQuantisedTreeStatusMaskHashOverflow);
        }
    }
}

/** Copy one tree through aligned 128-bit words cooperatively. */
__device__ void nativeQuantisedCopyTreeCooperatively(
    NativeQuantisedDeviceTree* destination,
    const NativeQuantisedDeviceTree* source) {
    static_assert(alignof(NativeQuantisedDeviceTree) >= alignof(uint4));
    static_assert(sizeof(NativeQuantisedDeviceTree) % sizeof(uint4) == 0);
    auto* output = reinterpret_cast<uint4*>(destination);
    const auto* input = reinterpret_cast<const uint4*>(source);
    constexpr std::uint32_t vectorCount = static_cast<std::uint32_t>(
        sizeof(NativeQuantisedDeviceTree) / sizeof(uint4));
    for (std::uint32_t vector = threadIdx.x; vector < vectorCount;
         vector += blockDim.x) {
        output[vector] = input[vector];
    }
}

/**
 * Return one leaf from a virtual expanded tree without copying its parent.
 */
__device__ __forceinline__ const NativeQuantisedDeviceLeaf*
nativeQuantisedExpandedLeaf(
    const NativeQuantisedExpandedCandidate& candidate,
    const NativeQuantisedDeviceTree* frontier,
    const NativeQuantisedSplitDescriptor* descriptors,
    const std::uint32_t* childUniqueIndices,
    const NativeQuantisedDeviceLeaf* childLeaves,
    std::uint32_t leafOrdinal) {
    const NativeQuantisedSplitDescriptor& descriptor =
        descriptors[candidate.descriptorIndex];
    const NativeQuantisedDeviceTree& parent = frontier[descriptor.frontierIndex];
    if (leafOrdinal < descriptor.targetLeaf) {
        return &parent.leaves[leafOrdinal];
    }
    if (leafOrdinal == descriptor.targetLeaf) {
        return &childLeaves[childUniqueIndices[candidate.descriptorIndex * 2U]];
    }
    if (leafOrdinal == descriptor.targetLeaf + 1U) {
        return &childLeaves[
            childUniqueIndices[candidate.descriptorIndex * 2U + 1U]];
    }
    return &parent.leaves[leafOrdinal - 1U];
}

/**
 * Merge the parent and two preordered child tails into exact nearest-rank p95.
 *
 * The target parent's values are omitted, later parent leaf ordinals are
 * shifted by one, and child ordinals occupy the vacated position in preorder.
 * Comparing the reconstructed source rank after the untouched FP64 value key
 * preserves the former subgroup implementation's tie order byte for byte.
 */
__device__ double nativeQuantisedExpandedFittingP95FromTails(
    const NativeQuantisedExpandedCandidate& candidate,
    const NativeQuantisedDeviceTree* frontier,
    const NativeQuantisedSplitDescriptor* descriptors,
    const std::uint32_t* childUniqueIndices,
    const NativeQuantisedDeviceLeaf* childLeaves,
    const std::uint16_t* childFittingTailPoints,
    const double* fittingRegrets,
    std::uint32_t pointCount,
    std::uint32_t candidateCount) {
    constexpr std::uint16_t invalidPoint = 0xffffU;
    const NativeQuantisedSplitDescriptor& descriptor =
        descriptors[candidate.descriptorIndex];
    const NativeQuantisedDeviceTree& parent = frontier[descriptor.frontierIndex];
    const std::uint32_t childIndices[2] = {
        childUniqueIndices[candidate.descriptorIndex * 2U],
        childUniqueIndices[candidate.descriptorIndex * 2U + 1U],
    };
    const std::uint32_t rank =
        static_cast<std::uint32_t>((95ULL * pointCount + 99ULL) / 100ULL);
    const std::uint32_t selectedTailIndex = pointCount - rank;
    std::uint32_t parentCursor = 0U;
    std::uint32_t childCursors[2] = {0U, 0U};
    std::uint64_t selectedBits = nativeQuantisedDoubleBits(kNativeQuantisedInfinity);
    for (std::uint32_t output = 0; output <= selectedTailIndex; ++output) {
        std::uint64_t bestBits = nativeQuantisedDoubleBits(-kNativeQuantisedInfinity);
        std::uint32_t bestSource = kNativeQuantisedTreeInvalidIndex;
        std::uint32_t bestStream = kNativeQuantisedTreeInvalidIndex;

        while (parentCursor < pointCount
               && nativeQuantisedFittingOrderLeaf(
                      parent.fittingOrderSources[parentCursor])
                   == descriptor.targetLeaf) {
            ++parentCursor;
        }
        if (parentCursor < pointCount) {
            const std::uint32_t oldSource =
                parent.fittingOrderSources[parentCursor];
            const std::uint32_t oldLeaf = nativeQuantisedFittingOrderLeaf(oldSource);
            const std::uint32_t point = nativeQuantisedFittingOrderPoint(oldSource);
            const std::uint32_t expandedLeaf = oldLeaf > descriptor.targetLeaf
                ? oldLeaf + 1U : oldLeaf;
            bestBits = parent.fittingOrderBits[parentCursor];
            bestSource = nativeQuantisedFittingOrderSource(expandedLeaf, point);
            bestStream = 0U;
        }
        for (std::uint32_t child = 0; child < 2U; ++child) {
            const std::uint16_t point = childFittingTailPoints[
                childIndices[child] * kNativeQuantisedLeafP95TailIterations
                + childCursors[child]];
            if (point == invalidPoint) {
                continue;
            }
            const std::uint64_t bits = nativeQuantisedDoubleBits(fittingRegrets[
                static_cast<std::uint64_t>(point) * candidateCount
                + childLeaves[childIndices[child]].candidate]);
            const std::uint32_t source = nativeQuantisedFittingOrderSource(
                descriptor.targetLeaf + child, point);
            if (bestStream == kNativeQuantisedTreeInvalidIndex
                || nativeQuantisedFittingOrderLess(
                    bits, source, bestBits, bestSource)) {
                bestBits = bits;
                bestSource = source;
                bestStream = child + 1U;
            }
        }
        selectedBits = bestBits;
        if (bestStream == 0U) {
            ++parentCursor;
        } else if (bestStream != kNativeQuantisedTreeInvalidIndex) {
            ++childCursors[bestStream - 1U];
        }
    }
    return nativeQuantisedDoubleFromBits(selectedBits);
}

/**
 * Score and finalize legal descriptors into compact expanded-tree records.
 *
 * One dense thread owns one descriptor. It reads parent leaves in place and
 * substitutes the two scored children while computing the exact scalar tree
 * objective. The same owner then merges the preordered child tails for exact
 * fitting p95 and composes the structural hash from immutable parent/child
 * fragments. Keeping these operations together preserves coalesced descriptor
 * traversal and avoids a scattered second pass through deduplicated indices.
 */
__global__ void scoreNativeQuantisedExpandedCandidates(
    const NativeQuantisedDeviceTree* frontier,
    const NativeQuantisedSplitDescriptor* descriptors,
    const std::uint32_t* descriptorCount,
    std::uint32_t descriptorCapacity,
    const std::uint32_t* childUniqueIndices,
    const std::uint32_t* childMaskCount,
    std::uint32_t childMaskCapacity,
    const NativeQuantisedDeviceLeaf* childLeaves,
    const std::uint16_t* childFittingTailPoints,
    const double* fittingRegrets,
    std::uint32_t pointCount,
    std::uint32_t candidateCount,
    NativeQuantisedExpandedCandidate* expanded,
    std::uint32_t expandedCapacity,
    std::uint32_t* expandedCount,
    std::uint32_t* status) {
    const std::uint32_t count =
        *descriptorCount < descriptorCapacity ? *descriptorCount : descriptorCapacity;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t descriptorIndex =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         descriptorIndex < count; descriptorIndex += stride) {
        const std::uint32_t uniqueCount = *childMaskCount < childMaskCapacity
            ? *childMaskCount : childMaskCapacity;
        const std::uint32_t leftIndex = childUniqueIndices[descriptorIndex * 2U];
        const std::uint32_t rightIndex =
            childUniqueIndices[descriptorIndex * 2U + 1U];
        if (leftIndex >= uniqueCount || rightIndex >= uniqueCount
            || !childLeaves[leftIndex].valid || !childLeaves[rightIndex].valid) {
            continue;
        }
        const NativeQuantisedSplitDescriptor& descriptor = descriptors[descriptorIndex];
        const NativeQuantisedDeviceTree& parent = frontier[descriptor.frontierIndex];
        const NativeQuantisedDeviceLeaf& left = childLeaves[leftIndex];
        const NativeQuantisedDeviceLeaf& right = childLeaves[rightIndex];
        std::uint32_t signatureBegin = 0;
        std::uint32_t signatureEnd = 0;
        if (!nativeQuantisedLeafSignatureSpan(
                parent, descriptor.targetLeaf, &signatureBegin, &signatureEnd)) {
            continue;
        }
        const std::uint32_t replacementSignatureCount =
            kNativeQuantisedTreeSplitSignatureWords
            + 3U + nativeQuantisedMaskPopcount(left.pointMask)
            + 3U + nativeQuantisedMaskPopcount(right.pointMask);
        const std::uint32_t signatureCount =
            parent.signatureCount - (signatureEnd - signatureBegin)
            + replacementSignatureCount;
        if (signatureCount > kNativeQuantisedTreeMaximumSignatureTokens) {
            continue;
        }

        const std::uint32_t outputIndex = atomicAdd(expandedCount, 1U);
        if (outputIndex >= expandedCapacity) {
            atomicOr(status, kNativeQuantisedTreeStatusExpansionOverflow);
            continue;
        }
        NativeQuantisedExpandedCandidate candidate;
        candidate.descriptorIndex = static_cast<std::uint32_t>(descriptorIndex);
        candidate.signatureBegin = signatureBegin;
        candidate.signatureEnd = signatureEnd;
        candidate.signatureCount = signatureCount;
        candidate.objective.failedLeafCount = 0;
        candidate.objective.fittingP95 = kNativeQuantisedInfinity;
        candidate.objective.fittingMean = 0.0;
        candidate.objective.worstLeafP95 = -kNativeQuantisedInfinity;
        candidate.objective.measuredFailureCount = 0;
        candidate.objective.measuredMaximum = -kNativeQuantisedInfinity;

        std::uint32_t selectedPointCount = 0;
        for (std::uint32_t leaf = 0; leaf < parent.leafCount + 1U; ++leaf) {
            const NativeQuantisedDeviceLeaf& selected = *nativeQuantisedExpandedLeaf(
                candidate, frontier, descriptors, childUniqueIndices,
                childLeaves, leaf);
            candidate.objective.failedLeafCount += selected.failedLeafCount;
            candidate.objective.worstLeafP95 =
                selected.measuredP95 > candidate.objective.worstLeafP95
                    ? selected.measuredP95
                    : candidate.objective.worstLeafP95;
            candidate.objective.measuredFailureCount +=
                selected.measuredFailureCount;
            candidate.objective.measuredMaximum =
                selected.measuredMaximum > candidate.objective.measuredMaximum
                    ? selected.measuredMaximum
                    : candidate.objective.measuredMaximum;
            // Visit selected points in exactly the same ascending order as the
            // former `point = 0..pointCount` loop. Skipping zero bits removes
            // O(leafCount * pointCount) membership probes while retaining the
            // byte-identical left-to-right FP64 addition sequence required by
            // the Python policy oracle.
            for (std::uint32_t word = 0;
                 word < kLlaminarNativeQuantisedTreePointMaskWords; ++word) {
                std::uint64_t selectedBits = selected.pointMask.words[word];
                while (selectedBits != 0U) {
                    const std::uint32_t bit = static_cast<std::uint32_t>(
                        __ffsll(static_cast<long long>(selectedBits)) - 1);
                    selectedBits &= selectedBits - 1U;
                    const std::uint32_t point = word * 64U + bit;
                    if (point >= pointCount) {
                        continue;
                    }
                    candidate.objective.fittingMean += fittingRegrets[
                        static_cast<std::uint64_t>(point) * candidateCount
                        + selected.candidate];
                    ++selectedPointCount;
                }
            }
        }
        candidate.objective.fittingMean /= selectedPointCount;

        // The nearest-rank merge retains the canonical parent-preorder source
        // tie break. Only the bounded high tail is visited; the complete point
        // inventory is never sorted again for an expanded candidate.
        candidate.objective.fittingP95 =
            nativeQuantisedExpandedFittingP95FromTails(
                candidate, frontier, descriptors, childUniqueIndices,
                childLeaves, childFittingTailPoints, fittingRegrets,
                pointCount, candidateCount);

        // Hashes only select exact-deduplication probe slots. Parent prefix and
        // suffix hashes plus complete child hashes let this owner compose the
        // virtual signature without materializing it; every collision is still
        // resolved by the authoritative canonical token stream.
        std::uint64_t thresholdHash = 0U;
        for (std::uint32_t token = 0;
             token < kNativeQuantisedTreeSplitSignatureWords; ++token) {
            thresholdHash = nativeQuantisedSignatureHashAppend(
                thresholdHash,
                nativeQuantisedThresholdSignatureToken(descriptor.threshold, token));
        }
        std::uint64_t replacementHash = nativeQuantisedSignatureHashCombine(
            thresholdHash, left.signatureHash, left.signatureCount);
        replacementHash = nativeQuantisedSignatureHashCombine(
            replacementHash, right.signatureHash, right.signatureCount);
        const std::uint32_t replacementCount =
            kNativeQuantisedTreeSplitSignatureWords
            + left.signatureCount + right.signatureCount;
        const std::uint32_t suffixCount =
            parent.signatureCount - candidate.signatureEnd;
        std::uint64_t hash = nativeQuantisedSignatureHashCombine(
            parent.leafSignaturePrefixHashes[descriptor.targetLeaf],
            replacementHash, replacementCount);
        hash = nativeQuantisedSignatureHashCombine(
            hash, parent.leafSignatureSuffixHashes[descriptor.targetLeaf],
            suffixCount);
        candidate.signatureHash = nativeQuantisedFinalizeSignatureHash(
            hash, candidate.signatureCount);
        expanded[outputIndex] = candidate;
    }
}

/** Forward-only state for one compact candidate's virtual signature. */
struct NativeQuantisedExpandedSignatureCursor {
    std::uint32_t position = 0;
    std::uint32_t childIndex = 0;
    std::uint32_t childPhase = 0;
    std::uint32_t childPoint = 0;
};

/**
 * Consume the next virtual signature token in O(1) amortized work.
 *
 * Point membership is scanned monotonically across each replacement leaf. A
 * random-access token lookup would restart that scan for every emitted point and
 * make hashing a quadratic operation for large leaves.
 */
__device__ std::uint32_t nativeQuantisedNextExpandedSignatureToken(
    const NativeQuantisedExpandedCandidate& candidate,
    const NativeQuantisedDeviceTree* frontier,
    const NativeQuantisedSplitDescriptor* descriptors,
    const std::uint32_t* childUniqueIndices,
    const NativeQuantisedDeviceLeaf* childLeaves,
    const std::uint32_t* pointGroupRanks,
    std::uint32_t pointCount,
    NativeQuantisedExpandedSignatureCursor* cursor) {
    const NativeQuantisedSplitDescriptor& descriptor =
        descriptors[candidate.descriptorIndex];
    const NativeQuantisedDeviceTree& parent = frontier[descriptor.frontierIndex];
    const NativeQuantisedDeviceLeaf& left =
        childLeaves[childUniqueIndices[candidate.descriptorIndex * 2U]];
    const NativeQuantisedDeviceLeaf& right =
        childLeaves[childUniqueIndices[candidate.descriptorIndex * 2U + 1U]];
    if (cursor->position < candidate.signatureBegin) {
        return parent.signature[cursor->position++];
    }
    const std::uint32_t oldSpan = candidate.signatureEnd - candidate.signatureBegin;
    const std::uint32_t replacementCount =
        candidate.signatureCount - (parent.signatureCount - oldSpan);
    if (cursor->position >= candidate.signatureBegin + replacementCount) {
        const std::uint32_t token = parent.signature[
            cursor->position - replacementCount + oldSpan];
        ++cursor->position;
        return token;
    }

    const std::uint32_t local = cursor->position - candidate.signatureBegin;
    if (local < kNativeQuantisedTreeSplitSignatureWords) {
        ++cursor->position;
        return nativeQuantisedThresholdSignatureToken(descriptor.threshold, local);
    }
    const NativeQuantisedDeviceLeaf* leaves[2] = {&left, &right};
    while (cursor->childIndex < 2U) {
        const NativeQuantisedDeviceLeaf* leaf = leaves[cursor->childIndex];
        if (cursor->childPhase == 0U) {
            cursor->childPhase = 1U;
            ++cursor->position;
            return 0U;
        }
        if (cursor->childPhase == 1U) {
            cursor->childPhase = 2U;
            ++cursor->position;
            return leaf->candidate + 1U;
        }
        // The mask becomes sparse as tree depth grows. Walk its 64-bit words
        // and jump directly to the next set bit instead of testing every absent
        // point. Clearing the already-consumed low bits retains the canonical
        // ascending point order and leaves the externally visible token stream
        // byte-for-byte unchanged.
        std::uint32_t word = cursor->childPoint >> 6U;
        if (word < kLlaminarNativeQuantisedTreePointMaskWords) {
            std::uint64_t selected = leaf->pointMask.words[word]
                & (~std::uint64_t{0} << (cursor->childPoint & 63U));
            while (selected == 0U
                   && ++word < kLlaminarNativeQuantisedTreePointMaskWords) {
                selected = leaf->pointMask.words[word];
            }
            if (selected != 0U) {
                const std::uint32_t point = word * 64U
                    + static_cast<std::uint32_t>(
                        __ffsll(static_cast<long long>(selected)) - 1);
                if (point < pointCount) {
                    cursor->childPoint = point + 1U;
                    ++cursor->position;
                    return pointGroupRanks[point] + 1U;
                }
            }
        }
        ++cursor->childIndex;
        cursor->childPhase = 0U;
        cursor->childPoint = 0U;
        ++cursor->position;
        return 0U;
    }
    return 0U;
}

/** Compare two virtual signatures exactly in canonical token order. */
__device__ int nativeQuantisedExpandedSignatureCompare(
    const NativeQuantisedExpandedCandidate& left,
    const NativeQuantisedExpandedCandidate& right,
    const NativeQuantisedDeviceTree* frontier,
    const NativeQuantisedSplitDescriptor* descriptors,
    const std::uint32_t* childUniqueIndices,
    const NativeQuantisedDeviceLeaf* childLeaves,
    const std::uint32_t* pointGroupRanks,
    const std::uint16_t* frontierLcp,
    std::uint32_t pointCount) {
    const std::uint32_t common = left.signatureCount < right.signatureCount
        ? left.signatureCount : right.signatureCount;
    const NativeQuantisedSplitDescriptor& leftDescriptor =
        descriptors[left.descriptorIndex];
    const NativeQuantisedSplitDescriptor& rightDescriptor =
        descriptors[right.descriptorIndex];

    // Each expansion copies its parent unchanged until its replacement begins.
    // Clamp the parents' proven token LCP at both replacement boundaries. The
    // resulting position is outside both virtual replacement bodies, so default
    // child cursors remain complete descriptions of every token that follows.
    std::uint32_t begin = frontierLcp != nullptr
        ? frontierLcp[
            leftDescriptor.frontierIndex * kNativeQuantisedTreeBeamWidth
            + rightDescriptor.frontierIndex]
        : (leftDescriptor.frontierIndex == rightDescriptor.frontierIndex
            ? (left.signatureBegin < right.signatureBegin
                ? left.signatureBegin : right.signatureBegin)
            : 0U);
    begin = begin < left.signatureBegin ? begin : left.signatureBegin;
    begin = begin < right.signatureBegin ? begin : right.signatureBegin;
    // When both replacements begin at the proven common prefix, compare the
    // complete ten-word threshold directly from its typed fields. Unsigned
    // 64-bit numerator/denominator ordering is identical to comparing their
    // high then low uint32 signature words. Equal thresholds let both cursors
    // start at the first child token with no skipped uncertainty.
    if (begin == left.signatureBegin && begin == right.signatureBegin) {
        const int thresholdOrder = nativeQuantisedThresholdSignatureCompare(
            leftDescriptor.threshold, rightDescriptor.threshold);
        if (thresholdOrder != 0) {
            return thresholdOrder;
        }
        const NativeQuantisedDeviceLeaf& leftExpansionLeft = childLeaves[
            childUniqueIndices[left.descriptorIndex * 2U]];
        const NativeQuantisedDeviceLeaf& leftExpansionRight = childLeaves[
            childUniqueIndices[left.descriptorIndex * 2U + 1U]];
        const NativeQuantisedDeviceLeaf& rightExpansionLeft = childLeaves[
            childUniqueIndices[right.descriptorIndex * 2U]];
        const NativeQuantisedDeviceLeaf& rightExpansionRight = childLeaves[
            childUniqueIndices[right.descriptorIndex * 2U + 1U]];
        const int leftLeafOrder = nativeQuantisedLeafSignatureTokenCompare(
            leftExpansionLeft, rightExpansionLeft, pointGroupRanks,
            pointCount);
        if (leftLeafOrder != 0) {
            return leftLeafOrder;
        }
        const int rightLeafOrder = nativeQuantisedLeafSignatureTokenCompare(
            leftExpansionRight, rightExpansionRight, pointGroupRanks,
            pointCount);
        if (rightLeafOrder != 0) {
            return rightLeafOrder;
        }
        begin += kNativeQuantisedTreeSplitSignatureWords
            + leftExpansionLeft.signatureCount
            + leftExpansionRight.signatureCount;
    }
    NativeQuantisedExpandedSignatureCursor leftCursor;
    NativeQuantisedExpandedSignatureCursor rightCursor;
    leftCursor.position = begin;
    rightCursor.position = begin;
    for (std::uint32_t position = begin; position < common; ++position) {
        const std::uint32_t leftToken = nativeQuantisedNextExpandedSignatureToken(
            left, frontier, descriptors, childUniqueIndices, childLeaves,
            pointGroupRanks, pointCount, &leftCursor);
        const std::uint32_t rightToken = nativeQuantisedNextExpandedSignatureToken(
            right, frontier, descriptors, childUniqueIndices, childLeaves,
            pointGroupRanks, pointCount, &rightCursor);
        if (leftToken < rightToken) {
            return -1;
        }
        if (leftToken > rightToken) {
            return 1;
        }
    }
    if (left.signatureCount < right.signatureCount) {
        return -1;
    }
    if (left.signatureCount > right.signatureCount) {
        return 1;
    }
    return 0;
}

/**
 * Insert every expanded signature into an open-addressed exact set.
 *
 * Hashes choose probe locations only. Every occupied slot is compared through
 * the complete canonical signature, so collisions cannot merge distinct trees.
 * The winning index for an identical signature may vary with scheduling, but
 * identical signatures carry identical canonical leaves and objectives.
 */
__global__ void deduplicateNativeQuantisedTrees(
    const NativeQuantisedExpandedCandidate* expanded,
    const std::uint32_t* expandedCount,
    std::uint32_t expandedCapacity,
    const NativeQuantisedDeviceTree* frontier,
    const NativeQuantisedSplitDescriptor* descriptors,
    const std::uint32_t* childUniqueIndices,
    const NativeQuantisedDeviceLeaf* childLeaves,
    const std::uint32_t* pointGroupRanks,
    std::uint32_t pointCount,
    std::uint32_t* hashSlots,
    std::uint32_t hashCapacity,
    std::uint32_t* uniqueIndices,
    std::uint32_t* uniqueCount,
    std::uint32_t* status) {
    const std::uint32_t count =
        *expandedCount < expandedCapacity ? *expandedCount : expandedCapacity;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t index =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        const NativeQuantisedExpandedCandidate& candidate = expanded[index];
        std::uint32_t slot = static_cast<std::uint32_t>(
            candidate.signatureHash) & (hashCapacity - 1U);
        bool inserted = false;
        for (std::uint32_t probe = 0; probe < hashCapacity; ++probe) {
            const std::uint32_t prior = atomicCAS(
                &hashSlots[slot], kNativeQuantisedTreeInvalidIndex,
                static_cast<std::uint32_t>(index));
            if (prior == kNativeQuantisedTreeInvalidIndex) {
                const std::uint32_t uniquePosition = atomicAdd(uniqueCount, 1U);
                uniqueIndices[uniquePosition] = static_cast<std::uint32_t>(index);
                inserted = true;
                break;
            }
            if (nativeQuantisedExpandedSignatureCompare(
                    candidate, expanded[prior], frontier, descriptors,
                    childUniqueIndices, childLeaves, pointGroupRanks,
                    nullptr, pointCount) == 0) {
                inserted = true;
                break;
            }
            slot = (slot + 1U) & (hashCapacity - 1U);
        }
        if (!inserted) {
            atomicOr(status, kNativeQuantisedTreeStatusHashOverflow);
        }
    }
}

/**
 * Dense immutable scalar ordering key extracted from one expanded tree.
 *
 * Floating-point objectives are converted once to monotonically ordered
 * unsigned integers. Selection only compares these values; it never performs
 * arithmetic on them. The conversion preserves finite numeric ordering and
 * treats positive and negative zero as equal, exactly matching the source
 * double comparator without repeatedly spending scarce GPU FP64 throughput.
 */
struct NativeQuantisedTreeOrderKey {
    std::uint64_t words[5];
};

/** Uniform immutable pointer bundle shared by one depth's ordering kernels. */
struct NativeQuantisedTreeSelectionContext {
    const NativeQuantisedExpandedCandidate* candidates;
    const NativeQuantisedTreeOrderKey* orderKeys;
    const NativeQuantisedDeviceTree* frontier;
    const NativeQuantisedSplitDescriptor* descriptors;
    const std::uint32_t* childUniqueIndices;
    const NativeQuantisedDeviceLeaf* childLeaves;
    const std::uint32_t* pointGroupRanks;
    const std::uint16_t* frontierLcp;
    std::uint32_t pointCount;
};

/** Publish both ping-pong frontier contexts inside the captured graph. */
__global__ void initializeNativeQuantisedTreeSelectionContexts(
    NativeQuantisedTreeSelectionContext* contexts,
    const NativeQuantisedExpandedCandidate* candidates,
    const NativeQuantisedTreeOrderKey* orderKeys,
    const NativeQuantisedDeviceTree* frontierA,
    const NativeQuantisedDeviceTree* frontierB,
    const NativeQuantisedSplitDescriptor* descriptors,
    const std::uint32_t* childUniqueIndices,
    const NativeQuantisedDeviceLeaf* childLeaves,
    const std::uint32_t* pointGroupRanks,
    const std::uint16_t* frontierLcpA,
    const std::uint16_t* frontierLcpB,
    std::uint32_t pointCount) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) {
        return;
    }
    const NativeQuantisedDeviceTree* frontiers[2] = {frontierA, frontierB};
    const std::uint16_t* frontierLcps[2] = {frontierLcpA, frontierLcpB};
    for (std::uint32_t index = 0; index < 2U; ++index) {
        contexts[index] = NativeQuantisedTreeSelectionContext{
            candidates,
            orderKeys,
            frontiers[index],
            descriptors,
            childUniqueIndices,
            childLeaves,
            pointGroupRanks,
            frontierLcps[index],
            pointCount,
        };
    }
}

/**
 * Build the exact pairwise longest-common-prefix table for one frontier.
 *
 * Frontier signatures are fully materialized and contain at most 918 tokens.
 * One thread compares one ordered parent pair and stores the number of leading
 * tokens that are byte-for-byte equal. Selection later clamps this count at each
 * expansion's replacement boundary, allowing virtual signature comparators to
 * skip only a prefix whose equality was proven inside the same captured graph.
 */
__global__ void buildNativeQuantisedFrontierLcp(
    const NativeQuantisedDeviceTree* frontier,
    const std::uint32_t* frontierCount,
    std::uint16_t* lcp) {
    const std::uint32_t left = blockIdx.x;
    const std::uint32_t right = threadIdx.x;
    const std::uint32_t count = *frontierCount < kNativeQuantisedTreeBeamWidth
        ? *frontierCount : kNativeQuantisedTreeBeamWidth;
    if (left >= count || right >= count) {
        return;
    }
    const NativeQuantisedDeviceTree& leftTree = frontier[left];
    const NativeQuantisedDeviceTree& rightTree = frontier[right];
    const std::uint32_t common = leftTree.signatureCount
            < rightTree.signatureCount
        ? leftTree.signatureCount : rightTree.signatureCount;
    std::uint32_t position = 0U;
    while (position < common
           && leftTree.signature[position] == rightTree.signature[position]) {
        ++position;
    }
    lcp[left * kNativeQuantisedTreeBeamWidth + right] =
        static_cast<std::uint16_t>(position);
}

/** Compare only the dense scalar portion of two canonical tree order keys. */
__device__ __forceinline__ int nativeQuantisedTreeScalarCompare(
    const NativeQuantisedTreeOrderKey& left,
    const NativeQuantisedTreeOrderKey& right) {
    #pragma unroll
    for (std::uint32_t word = 0U; word < 5U; ++word) {
        if (left.words[word] < right.words[word]) {
            return -1;
        }
        if (left.words[word] > right.words[word]) {
            return 1;
        }
    }
    return 0;
}

/** Extract dense ordering keys with coalesced writes. */
__global__ void extractNativeQuantisedTreeOrderKeys(
    const NativeQuantisedExpandedCandidate* candidates,
    const std::uint32_t* expandedCount,
    const std::uint32_t* uniqueIndices,
    const std::uint32_t* uniqueCount,
    std::uint32_t uniqueCapacity,
    std::uint64_t* expandedCandidateTotal,
    std::uint64_t* structurallyUniqueCandidateTotal,
    NativeQuantisedTreeOrderKey* keys) {
    const std::uint32_t count =
        *uniqueCount < uniqueCapacity ? *uniqueCount : uniqueCapacity;
    if (blockIdx.x == 0U && threadIdx.x == 0U) {
        atomicAdd(
            reinterpret_cast<unsigned long long*>(expandedCandidateTotal),
            static_cast<unsigned long long>(
                *expandedCount < uniqueCapacity ? *expandedCount : uniqueCapacity));
        atomicAdd(
            reinterpret_cast<unsigned long long*>(
                structurallyUniqueCandidateTotal),
            static_cast<unsigned long long>(count));
    }
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t uniquePosition =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         uniquePosition < count; uniquePosition += stride) {
        const std::uint32_t index = uniqueIndices[uniquePosition];
        const NativeQuantisedExpandedCandidate& candidate = candidates[index];
        const NativeQuantisedDeviceTreeObjective& objective = candidate.objective;
        NativeQuantisedTreeOrderKey* key = &keys[index];
        const std::uint64_t fittingP95 = nativeQuantisedDoubleOrderKey(
            nativeQuantisedDoubleBits(objective.fittingP95));
        const std::uint64_t fittingMean = nativeQuantisedDoubleOrderKey(
            nativeQuantisedDoubleBits(objective.fittingMean));
        const std::uint64_t worstLeafP95 = nativeQuantisedDoubleOrderKey(
            nativeQuantisedDoubleBits(objective.worstLeafP95));
        key->words[0] =
            static_cast<std::uint64_t>(objective.failedLeafCount) << 32U
            | fittingP95 >> 32U;
        key->words[1] = fittingP95 << 32U | fittingMean >> 32U;
        key->words[2] = fittingMean << 32U | worstLeafP95 >> 32U;
        key->words[3] = worstLeafP95 << 32U
            | static_cast<std::uint64_t>(objective.measuredFailureCount);
        key->words[4] = nativeQuantisedDoubleOrderKey(
            nativeQuantisedDoubleBits(objective.measuredMaximum));
    }
}

/**
 * Apply one most-significant-first radix byte to a block-local rank boundary.
 *
 * Every thread calls this helper, including inactive threads, because its
 * barriers protect one shared histogram and the shared rank-63 state. Values in
 * lower buckets become irrevocable winners; values in higher buckets leave the
 * tournament; only the bucket containing the remaining boundary stays active.
 */
__device__ __forceinline__ void nativeQuantisedNarrowSelectionByte(
    std::uint32_t keyByte,
    bool* active,
    bool* selected,
    std::uint32_t* histogram,
    std::uint32_t* activeCount,
    std::uint32_t* selectedCount,
    std::uint32_t* selectedBucket,
    std::uint32_t* narrowingDone) {
    histogram[threadIdx.x] = 0U;
    __syncthreads();
    if (*active) {
        atomicAdd(&histogram[keyByte], 1U);
    }
    __syncthreads();
    if (threadIdx.x == 0U) {
        const std::uint32_t target =
            kNativeQuantisedTreeSelectionRetained - 1U - *selectedCount;
        std::uint32_t less = 0U;
        *selectedBucket = 0U;
        for (; *selectedBucket < 256U; ++*selectedBucket) {
            const std::uint32_t bucketCount = histogram[*selectedBucket];
            if (target < less + bucketCount) {
                *selectedCount += less;
                *activeCount = bucketCount;
                break;
            }
            less += bucketCount;
        }
        if (*selectedCount + *activeCount
            <= kNativeQuantisedTreeSelectionRetained) {
            *narrowingDone = 1U;
        }
    }
    __syncthreads();
    if (*active) {
        if (keyByte < *selectedBucket) {
            *active = false;
            *selected = true;
        } else if (keyByte > *selectedBucket) {
            *active = false;
        }
    }
    __syncthreads();
}

/** Compare staged scalar keys while retaining invalid-index ordering. */
__device__ __forceinline__ bool nativeQuantisedStagedScalarIndexLess(
    std::uint32_t left,
    std::uint32_t right,
    const NativeQuantisedTreeOrderKey& leftKey,
    const NativeQuantisedTreeOrderKey& rightKey) {
    if (left == kNativeQuantisedTreeInvalidIndex) {
        return false;
    }
    if (right == kNativeQuantisedTreeInvalidIndex) {
        return true;
    }
    return nativeQuantisedTreeScalarCompare(leftKey, rightKey) < 0;
}

/** Compare staged scalar keys and exact virtual signatures as one total key. */
__device__ __forceinline__ bool nativeQuantisedStagedTreeIndexLess(
    std::uint32_t left,
    std::uint32_t right,
    const NativeQuantisedTreeOrderKey& leftKey,
    const NativeQuantisedTreeOrderKey& rightKey,
    const NativeQuantisedTreeSelectionContext* context) {
    if (left == kNativeQuantisedTreeInvalidIndex) {
        return false;
    }
    if (right == kNativeQuantisedTreeInvalidIndex) {
        return true;
    }
    const int scalarOrder = nativeQuantisedTreeScalarCompare(leftKey, rightKey);
    if (scalarOrder != 0) {
        return scalarOrder < 0;
    }
    return nativeQuantisedExpandedSignatureCompare(
        context->candidates[left], context->candidates[right],
        context->frontier, context->descriptors, context->childUniqueIndices,
        context->childLeaves, context->pointGroupRanks, context->frontierLcp,
        context->pointCount) < 0;
}

/** Swap one shared scalar key field-by-field without a private aggregate. */
__device__ __forceinline__ void nativeQuantisedSwapTreeOrderKeys(
    NativeQuantisedTreeOrderKey* left,
    NativeQuantisedTreeOrderKey* right) {
    #pragma unroll
    for (std::uint32_t word = 0U; word < 5U; ++word) {
        const std::uint64_t temporary = left->words[word];
        left->words[word] = right->words[word];
        right->words[word] = temporary;
    }
}

/**
 * Narrow each 256-index partition to the exact total-order prefix at rank 63.
 *
 * A shared comparator network first finds the rank-63 scalar objective using the
 * precomputed integer keys. Values strictly below that key are winners, values
 * above it are discarded, and only its exact tie bucket enters signature radix.
 * Every still-active thread then advances its virtual canonical signature cursor
 * one token at a time. Deduplication removed equal signatures, so the boundary
 * must eventually contain at most the remaining winner slots. The following
 * comparator kernel orders no more than 64 exact survivors.
 */
__global__ void narrowNativeQuantisedTreeTopIndices(
    const NativeQuantisedTreeSelectionContext* context,
    const std::uint32_t* input,
    std::uint32_t inputCapacity,
    const std::uint32_t* deviceInputCount,
    std::uint32_t* narrowedIndices,
    std::uint32_t* narrowedCounts) {
    __shared__ std::uint32_t histogram[256];
    __shared__ std::uint32_t scalarIndices[kNativeQuantisedTreeSelectionChunk];
    __shared__ NativeQuantisedTreeOrderKey
        scalarKeys[kNativeQuantisedTreeSelectionChunk];
    __shared__ std::uint32_t activeCount;
    __shared__ std::uint32_t selectedCount;
    __shared__ std::uint32_t selectedBucket;
    __shared__ std::uint32_t narrowingDone;
    __shared__ std::uint32_t compactCount;
    __shared__ std::uint32_t signatureReference;
    __shared__ std::uint32_t commonSignaturePrefix;
    const std::uint32_t inputCount = deviceInputCount == nullptr
        ? inputCapacity
        : (*deviceInputCount < inputCapacity ? *deviceInputCount : inputCapacity);
    const std::uint32_t inputIndex =
        blockIdx.x * kNativeQuantisedTreeSelectionChunk + threadIdx.x;
    std::uint32_t value = inputIndex < inputCount
        ? input[inputIndex]
        : kNativeQuantisedTreeInvalidIndex;

    // Scalar comparisons touch only the compact 48-byte order-key array and do
    // not traverse virtual signatures. Sorting that cheap prefix takes 36
    // barriers, substantially fewer than the former 44 byte histograms. The
    // expensive signature path still sees only the rank-boundary tie bucket.
    bool active = value != kNativeQuantisedTreeInvalidIndex;
    bool selected = false;
    NativeQuantisedExpandedSignatureCursor signatureCursor;
    scalarIndices[threadIdx.x] = value;
    if (active) {
        scalarKeys[threadIdx.x] = context->orderKeys[value];
    } else {
        NativeQuantisedTreeOrderKey* key = &scalarKeys[threadIdx.x];
        #pragma unroll
        for (std::uint32_t word = 0U; word < 5U; ++word) {
            key->words[word] = ~std::uint64_t{0};
        }
    }
    if (threadIdx.x == 0U) {
        activeCount = 0U;
        selectedCount = 0U;
        narrowingDone = 0U;
        signatureReference = kNativeQuantisedTreeInvalidIndex;
        commonSignaturePrefix = 0U;
    }
    __syncthreads();
    if (active) {
        atomicAdd(&activeCount, 1U);
    }
    __syncthreads();
    if (threadIdx.x == 0U
        && activeCount <= kNativeQuantisedTreeSelectionRetained) {
        narrowingDone = 1U;
    }
    __syncthreads();

    if (narrowingDone == 0U) {
        for (std::uint32_t width = 2U;
             width <= kNativeQuantisedTreeSelectionChunk; width <<= 1U) {
            if (width > kNativeQuantisedLeafGroupThreads) {
                // The preceding width ends with subgroup-local strides. Make
                // those writes visible before this width first crosses groups.
                __syncthreads();
            }
            for (std::uint32_t stride = width >> 1U;
                 stride != 0U; stride >>= 1U) {
                const std::uint32_t partner = threadIdx.x ^ stride;
                if (partner > threadIdx.x) {
                    const bool ascending = (threadIdx.x & width) == 0U;
                    const std::uint32_t left = scalarIndices[threadIdx.x];
                    const std::uint32_t right = scalarIndices[partner];
                    const bool swap = ascending
                        ? nativeQuantisedStagedScalarIndexLess(
                            right, left, scalarKeys[partner],
                            scalarKeys[threadIdx.x])
                        : nativeQuantisedStagedScalarIndexLess(
                            left, right, scalarKeys[threadIdx.x],
                            scalarKeys[partner]);
                    if (swap) {
                        scalarIndices[threadIdx.x] = right;
                        scalarIndices[partner] = left;
                        nativeQuantisedSwapTreeOrderKeys(
                            &scalarKeys[threadIdx.x], &scalarKeys[partner]);
                    }
                }
                if (stride >= kNativeQuantisedLeafGroupThreads) {
                    __syncthreads();
                } else {
                    nativeQuantisedSubgroupSynchronize();
                }
            }
        }
        // Rank 63 is consumed by every subgroup below, so publish the final
        // subgroup-local sorting stages across the complete block once.
        __syncthreads();
        if (threadIdx.x == 0U) {
            activeCount = 0U;
            selectedCount = 0U;
        }
        __syncthreads();
        if (active) {
            const int relation = nativeQuantisedTreeScalarCompare(
                context->orderKeys[value],
                scalarKeys[kNativeQuantisedTreeSelectionRetained - 1U]);
            selected = relation < 0;
            active = relation == 0;
            if (selected) {
                atomicAdd(&selectedCount, 1U);
            } else if (active) {
                atomicAdd(&activeCount, 1U);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0U
            && selectedCount + activeCount
                <= kNativeQuantisedTreeSelectionRetained) {
            narrowingDone = 1U;
        }
        __syncthreads();
    }

    // Every candidate still active here has the same complete scalar key. Find
    // a prefix that all of their virtual signatures provably share: compare
    // each parent with one active reference using the exact frontier LCP table,
    // then clamp at both replacement boundaries. The minimum proof is common to
    // the whole bucket, so no skipped token can affect lexicographic ordering.
    if (narrowingDone == 0U) {
        if (threadIdx.x == 0U) {
            signatureReference = kNativeQuantisedTreeInvalidIndex;
            commonSignaturePrefix =
                kNativeQuantisedTreeMaximumSignatureTokens;
        }
        __syncthreads();
        if (active) {
            atomicMin(&signatureReference, value);
        }
        __syncthreads();
        if (active) {
            const NativeQuantisedExpandedCandidate& reference =
                context->candidates[signatureReference];
            const NativeQuantisedExpandedCandidate& candidate =
                context->candidates[value];
            const NativeQuantisedSplitDescriptor& referenceDescriptor =
                context->descriptors[reference.descriptorIndex];
            const NativeQuantisedSplitDescriptor& candidateDescriptor =
                context->descriptors[candidate.descriptorIndex];
            std::uint32_t prefix = context->frontierLcp[
                referenceDescriptor.frontierIndex * kNativeQuantisedTreeBeamWidth
                + candidateDescriptor.frontierIndex];
            prefix = prefix < reference.signatureBegin
                ? prefix : reference.signatureBegin;
            prefix = prefix < candidate.signatureBegin
                ? prefix : candidate.signatureBegin;
            atomicMin(&commonSignaturePrefix, prefix);
        }
        __syncthreads();
        signatureCursor.position = commonSignaturePrefix;
    }

    for (std::uint32_t position = commonSignaturePrefix;
         position < kNativeQuantisedTreeMaximumSignatureTokens
         && narrowingDone == 0U; ++position) {
        const std::uint32_t signatureCount = active
            ? context->candidates[value].signatureCount : 0U;
        nativeQuantisedNarrowSelectionByte(
            position < signatureCount ? 1U : 0U,
            &active, &selected, histogram, &activeCount,
            &selectedCount, &selectedBucket, &narrowingDone);
        if (narrowingDone != 0U) {
            break;
        }

        std::uint32_t token = 0U;
        if (active) {
            token = nativeQuantisedNextExpandedSignatureToken(
                context->candidates[value], context->frontier,
                context->descriptors, context->childUniqueIndices,
                context->childLeaves, context->pointGroupRanks,
                context->pointCount, &signatureCursor);
        }
        for (std::uint32_t tokenByte = 0U;
             tokenByte < 4U && narrowingDone == 0U; ++tokenByte) {
            nativeQuantisedNarrowSelectionByte(
                static_cast<std::uint32_t>(
                    (token >> ((3U - tokenByte) * 8U)) & 0xffU),
                &active, &selected, histogram, &activeCount,
                &selectedCount, &selectedBucket, &narrowingDone);
        }
    }

    // Compact the exact objective prefix into a fixed 256-index block segment.
    // Atomic arrival order is irrelevant because the next kernel imposes the
    // complete deterministic order before publishing any winner.
    if (threadIdx.x == 0U) {
        compactCount = 0U;
    }
    __syncthreads();
    if (selected || active) {
        const std::uint32_t destination = atomicAdd(&compactCount, 1U);
        narrowedIndices[
            blockIdx.x * kNativeQuantisedTreeSelectionChunk + destination] = value;
    }
    __syncthreads();
    if (threadIdx.x == 0U) {
        narrowedCounts[blockIdx.x] = compactCount;
    }
}

/** Sort one narrowed partition and publish its exact canonical best 64. */
__global__ void sortNativeQuantisedTreeNarrowedIndices(
    const NativeQuantisedTreeSelectionContext* context,
    const std::uint32_t* narrowedIndices,
    const std::uint32_t* narrowedCounts,
    std::uint32_t* output) {
    __shared__ std::uint32_t indices[kNativeQuantisedTreeSelectionChunk];
    __shared__ NativeQuantisedTreeOrderKey keys[kNativeQuantisedTreeSelectionChunk];
    const std::uint32_t compactCount = narrowedCounts[blockIdx.x];
    indices[threadIdx.x] = threadIdx.x < compactCount
        ? narrowedIndices[
            blockIdx.x * kNativeQuantisedTreeSelectionChunk + threadIdx.x]
        : kNativeQuantisedTreeInvalidIndex;
    if (indices[threadIdx.x] != kNativeQuantisedTreeInvalidIndex) {
        keys[threadIdx.x] = context->orderKeys[indices[threadIdx.x]];
    }
    __syncthreads();
    const std::uint32_t sortWidth = compactCount
            <= kNativeQuantisedTreeSelectionRetained
        ? kNativeQuantisedTreeSelectionRetained
        : kNativeQuantisedTreeSelectionChunk;
    for (std::uint32_t width = 2U; width <= sortWidth; width <<= 1U) {
        if (width > kNativeQuantisedLeafGroupThreads) {
            __syncthreads();
        }
        for (std::uint32_t stride = width >> 1U; stride != 0U; stride >>= 1U) {
            const std::uint32_t partner = threadIdx.x ^ stride;
            if (threadIdx.x < sortWidth && partner > threadIdx.x) {
                const bool ascending = (threadIdx.x & width) == 0U;
                const std::uint32_t left = indices[threadIdx.x];
                const std::uint32_t right = indices[partner];
                const bool swap = ascending
                    ? nativeQuantisedStagedTreeIndexLess(
                        right, left, keys[partner], keys[threadIdx.x], context)
                    : nativeQuantisedStagedTreeIndexLess(
                        left, right, keys[threadIdx.x], keys[partner], context);
                if (swap) {
                    indices[threadIdx.x] = right;
                    indices[partner] = left;
                    nativeQuantisedSwapTreeOrderKeys(
                        &keys[threadIdx.x], &keys[partner]);
                }
            }
            if (stride >= kNativeQuantisedLeafGroupThreads) {
                __syncthreads();
            } else {
                nativeQuantisedSubgroupSynchronize();
            }
        }
    }
    if (threadIdx.x < kNativeQuantisedTreeSelectionRetained) {
        output[blockIdx.x * kNativeQuantisedTreeSelectionRetained + threadIdx.x] =
            indices[threadIdx.x];
    }
}

/**
 * Merge four already-sorted 64-index runs and retain the exact best 64.
 *
 * Runs zero and one, then two and three, form independent bitonic sequences
 * because each odd run is loaded in reverse. Their best halves form one final
 * bitonic sequence. The hierarchy therefore needs fourteen comparator stages
 * instead of re-sorting all 256 inputs from scratch, while preserving the full
 * objective and canonical-signature order used by the Python oracle.
 */
__global__ void mergeNativeQuantisedTreeTopIndices(
    const NativeQuantisedTreeSelectionContext* context,
    const std::uint32_t* input,
    std::uint32_t inputCapacity,
    std::uint32_t* output) {
    __shared__ std::uint32_t indices[kNativeQuantisedTreeSelectionChunk];
    __shared__ NativeQuantisedTreeOrderKey keys[kNativeQuantisedTreeSelectionChunk];
    const std::uint32_t lane = threadIdx.x;
    const std::uint32_t run = lane / kNativeQuantisedTreeSelectionRetained;
    const std::uint32_t position =
        lane % kNativeQuantisedTreeSelectionRetained;
    const std::uint32_t sourcePosition = (run & 1U) != 0U
        ? kNativeQuantisedTreeSelectionRetained - 1U - position
        : position;
    const std::uint32_t inputIndex =
        blockIdx.x * kNativeQuantisedTreeSelectionChunk
        + run * kNativeQuantisedTreeSelectionRetained + sourcePosition;
    indices[lane] = inputIndex < inputCapacity
        ? input[inputIndex]
        : kNativeQuantisedTreeInvalidIndex;
    if (indices[lane] != kNativeQuantisedTreeInvalidIndex) {
        keys[lane] = context->orderKeys[indices[lane]];
    }
    __syncthreads();

    // Merge runs 0+1 and 2+3 independently. Strides below 32 stay inside one
    // logical subgroup, so they only require a subgroup barrier.
    for (std::uint32_t stride = kNativeQuantisedTreeSelectionRetained;
         stride >= kNativeQuantisedLeafGroupThreads; stride >>= 1U) {
        const std::uint32_t partner = lane ^ stride;
        if (partner > lane && (partner / 128U) == (lane / 128U)) {
            const std::uint32_t left = indices[lane];
            const std::uint32_t right = indices[partner];
            if (nativeQuantisedStagedTreeIndexLess(
                    right, left, keys[partner], keys[lane], context)) {
                indices[lane] = right;
                indices[partner] = left;
                nativeQuantisedSwapTreeOrderKeys(&keys[lane], &keys[partner]);
            }
        }
        __syncthreads();
    }
    for (std::uint32_t stride = kNativeQuantisedLeafGroupThreads >> 1U;
         stride != 0U; stride >>= 1U) {
        const std::uint32_t partner = lane ^ stride;
        if (partner > lane) {
            const std::uint32_t left = indices[lane];
            const std::uint32_t right = indices[partner];
            if (nativeQuantisedStagedTreeIndexLess(
                    right, left, keys[partner], keys[lane], context)) {
                indices[lane] = right;
                indices[partner] = left;
                nativeQuantisedSwapTreeOrderKeys(&keys[lane], &keys[partner]);
            }
        }
        nativeQuantisedSubgroupSynchronize();
    }
    __syncthreads();

    // Reverse the second pair's best half to create the final 128-value
    // bitonic sequence, then retain its exact ascending prefix.
    if (lane >= kNativeQuantisedTreeSelectionRetained
        && lane < 2U * kNativeQuantisedTreeSelectionRetained) {
        const std::uint32_t source =
            4U * kNativeQuantisedTreeSelectionRetained - 1U - lane;
        indices[lane] = indices[source];
        keys[lane] = keys[source];
    }
    __syncthreads();
    for (std::uint32_t stride = kNativeQuantisedTreeSelectionRetained;
         stride >= kNativeQuantisedLeafGroupThreads; stride >>= 1U) {
        const std::uint32_t partner = lane ^ stride;
        if (lane < 2U * kNativeQuantisedTreeSelectionRetained
            && partner > lane) {
            const std::uint32_t left = indices[lane];
            const std::uint32_t right = indices[partner];
            if (nativeQuantisedStagedTreeIndexLess(
                    right, left, keys[partner], keys[lane], context)) {
                indices[lane] = right;
                indices[partner] = left;
                nativeQuantisedSwapTreeOrderKeys(&keys[lane], &keys[partner]);
            }
        }
        __syncthreads();
    }
    if (lane < 2U * kNativeQuantisedTreeSelectionRetained) {
        for (std::uint32_t stride = kNativeQuantisedLeafGroupThreads >> 1U;
             stride != 0U; stride >>= 1U) {
            const std::uint32_t partner = lane ^ stride;
            if (partner > lane) {
                const std::uint32_t left = indices[lane];
                const std::uint32_t right = indices[partner];
                if (nativeQuantisedStagedTreeIndexLess(
                        right, left, keys[partner], keys[lane], context)) {
                    indices[lane] = right;
                    indices[partner] = left;
                    nativeQuantisedSwapTreeOrderKeys(
                        &keys[lane], &keys[partner]);
                }
            }
            nativeQuantisedSubgroupSynchronize();
        }
    }
    if (lane < kNativeQuantisedTreeSelectionRetained) {
        output[blockIdx.x * kNativeQuantisedTreeSelectionRetained + lane] =
            indices[lane];
    }
}

/** Materialize only the final ordered winners into the next full frontier. */
__global__ void gatherNativeQuantisedTreeFrontier(
    const NativeQuantisedExpandedCandidate* expanded,
    const NativeQuantisedDeviceTree* inputFrontier,
    const NativeQuantisedSplitDescriptor* descriptors,
    const std::uint32_t* childUniqueIndices,
    const NativeQuantisedDeviceLeaf* childLeaves,
    const std::uint32_t* pointGroupRanks,
    std::uint32_t pointCount,
    const std::uint32_t* selected,
    NativeQuantisedDeviceTree* frontier,
    std::uint32_t* frontierCount) {
    const std::uint32_t frontierIndex = blockIdx.x;
    const std::uint32_t expandedIndex = selected[frontierIndex];
    if (expandedIndex != kNativeQuantisedTreeInvalidIndex) {
        const NativeQuantisedExpandedCandidate& candidate = expanded[expandedIndex];
        const NativeQuantisedSplitDescriptor& descriptor =
            descriptors[candidate.descriptorIndex];
        const NativeQuantisedDeviceTree* parent =
            &inputFrontier[descriptor.frontierIndex];
        NativeQuantisedDeviceTree* output = &frontier[frontierIndex];
        nativeQuantisedCopyTreeCooperatively(output, parent);
        __syncthreads();
        if (threadIdx.x == 0) {
            const NativeQuantisedDeviceLeaf& left = childLeaves[
                childUniqueIndices[candidate.descriptorIndex * 2U]];
            const NativeQuantisedDeviceLeaf& right = childLeaves[
                childUniqueIndices[candidate.descriptorIndex * 2U + 1U]];
            for (std::uint32_t leaf = parent->leafCount;
                 leaf > descriptor.targetLeaf + 1U; --leaf) {
                nativeQuantisedCopyLeaf(
                    &output->leaves[leaf], &output->leaves[leaf - 1U]);
            }
            nativeQuantisedCopyLeaf(&output->leaves[descriptor.targetLeaf], &left);
            nativeQuantisedCopyLeaf(
                &output->leaves[descriptor.targetLeaf + 1U], &right);
            output->leafCount = parent->leafCount + 1U;

            const std::uint32_t oldSpan =
                candidate.signatureEnd - candidate.signatureBegin;
            const std::uint32_t replacementCount =
                candidate.signatureCount - (parent->signatureCount - oldSpan);
            for (std::uint32_t source = parent->signatureCount;
                 source > candidate.signatureEnd; --source) {
                output->signature[
                    source - oldSpan + replacementCount - 1U] =
                    output->signature[source - 1U];
            }
            std::uint32_t write = candidate.signatureBegin;
            write = nativeQuantisedWriteThresholdSignature(
                output->signature, write, descriptor.threshold);
            write = nativeQuantisedWriteLeafSignature(
                output->signature, write, left, pointGroupRanks, pointCount);
            write = nativeQuantisedWriteLeafSignature(
                output->signature, write, right, pointGroupRanks, pointCount);
            output->signatureCount = candidate.signatureCount;
            output->objective = candidate.objective;
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        std::uint32_t count = 0;
        while (count < kNativeQuantisedTreeBeamWidth
               && selected[count] != kNativeQuantisedTreeInvalidIndex) {
            ++count;
        }
        *frontierCount = count;
    }
}

/** Convert one internal tree into the stable fixed-size result ABI. */
__host__ __device__ bool nativeQuantisedExportTree(
    const NativeQuantisedDeviceTree& tree,
    LlaminarNativeQuantisedTreeFitResult* output) {
    LlaminarNativeQuantisedTreeFitResult cleared = {};
    *output = cleared;
    output->leaf_count = tree.leafCount;
    for (std::uint32_t leaf = 0; leaf < tree.leafCount; ++leaf) {
        output->leaf_masks[leaf] = tree.leaves[leaf].pointMask;
        output->candidate_indices[leaf] = tree.leaves[leaf].candidate;
    }
    std::uint32_t pendingNodes = 1;
    std::uint32_t signaturePosition = 0;
    std::uint32_t structurePosition = 0;
    while (pendingNodes != 0 && signaturePosition < tree.signatureCount) {
        const std::uint32_t kind = tree.signature[signaturePosition++];
        --pendingNodes;
        if (structurePosition
            >= kLlaminarNativeQuantisedTreeMaximumStructureTokens) {
            return false;
        }
        if (kind == 1U) {
            const std::uint32_t thresholdPosition = signaturePosition - 1U;
            if (thresholdPosition + kNativeQuantisedTreeSplitSignatureWords
                > tree.signatureCount) {
                return false;
            }
            output->structure_tokens[structurePosition] = 1U;
            output->split_thresholds[structurePosition] =
                nativeQuantisedReadThresholdSignature(
                    tree.signature, thresholdPosition);
            ++structurePosition;
            signaturePosition =
                thresholdPosition + kNativeQuantisedTreeSplitSignatureWords;
            pendingNodes += 2;
            continue;
        }
        output->structure_tokens[structurePosition++] = 0;
        ++signaturePosition;
        while (signaturePosition < tree.signatureCount
               && tree.signature[signaturePosition] != 0U) {
            ++signaturePosition;
        }
        if (signaturePosition >= tree.signatureCount) {
            return false;
        }
        ++signaturePosition;
    }
    if (pendingNodes != 0 || signaturePosition != tree.signatureCount) {
        return false;
    }
    output->structure_token_count = structurePosition;
    return true;
}

/** Publish one depth's best tree and preserve the incumbent for empty depths. */
__global__ void updateNativeQuantisedTreeIncumbent(
    const NativeQuantisedDeviceTree* frontier,
    const std::uint32_t* frontierCount,
    NativeQuantisedDeviceTree* incumbent,
    LlaminarNativeQuantisedTreeFitResult* results,
    std::uint32_t resultIndex,
    std::uint32_t* status) {
    __shared__ std::uint32_t replaceIncumbent;
    if (blockIdx.x != 0) {
        return;
    }
    if (threadIdx.x == 0) {
        replaceIncumbent = *frontierCount != 0
            && (incumbent->leafCount == 0
                || NativeQuantisedTreeObjectiveLess{}(frontier[0], *incumbent));
    }
    __syncthreads();
    if (replaceIncumbent != 0) {
        nativeQuantisedCopyTreeCooperatively(incumbent, &frontier[0]);
    }
    __syncthreads();
    if (threadIdx.x != 0) {
        return;
    }
    if (incumbent->leafCount == 0) {
        LlaminarNativeQuantisedTreeFitResult empty = {};
        results[resultIndex] = empty;
        return;
    }
    if (!nativeQuantisedExportTree(*incumbent, &results[resultIndex])) {
        atomicOr(status, kNativeQuantisedTreeStatusExportFailure);
    }
}

/**
 * Compare two exact products and surface unsupported integer growth as status.
 *
 * Corpus dimensions are far below uint64 limits. Keeping this guard in device
 * code nevertheless makes overflow a hard transaction failure instead of a
 * silently different branch from Python's arbitrary-precision arithmetic.
 */
__device__ __forceinline__ bool nativeQuantisedProductLessEqual(
    std::uint64_t leftA,
    std::uint64_t leftB,
    std::uint64_t rightA,
    std::uint64_t rightB,
    std::uint32_t* status) {
    std::uint64_t left = 0;
    std::uint64_t right = 0;
    if (!nativeQuantisedCheckedMultiply(leftA, leftB, &left)
        || !nativeQuantisedCheckedMultiply(rightA, rightB, &right)) {
        atomicOr(status, kNativeQuantisedTreeStatusThresholdOverflow);
        return false;
    }
    return left <= right;
}

/** Evaluate one Python-equivalent integer feature threshold on raw geometry. */
__device__ bool nativeQuantisedThresholdMatchesLessEqual(
    const LlaminarNativeQuantisedTreeThreshold& threshold,
    std::uint64_t aggregateN,
    std::uint64_t k,
    std::uint32_t launchKTiles,
    std::uint32_t* status) {
    if (aggregateN == 0 || k == 0 || threshold.denominator == 0
        || threshold.parallelism_width == 0
        || threshold.task_multiplier == 0) {
        atomicOr(status, kNativeQuantisedTreeStatusEvaluationFailure);
        return false;
    }
    const std::uint64_t denominator = threshold.denominator;
    const std::uint64_t numerator = threshold.numerator;
    const std::uint64_t tileWidth = threshold.tile_width;
    switch (threshold.operation) {
        case kLlaminarNativeQuantisedTreeThresholdAggregateN:
            return nativeQuantisedProductLessEqual(
                aggregateN, denominator, numerator, 1U, status);
        case kLlaminarNativeQuantisedTreeThresholdK:
            return nativeQuantisedProductLessEqual(
                k, denominator, numerator, 1U, status);
        case kLlaminarNativeQuantisedTreeThresholdWorkItems: {
            std::uint64_t workItems = 0;
            if (!nativeQuantisedCheckedMultiply(aggregateN, k, &workItems)) {
                atomicOr(status, kNativeQuantisedTreeStatusThresholdOverflow);
                return false;
            }
            return nativeQuantisedProductLessEqual(
                workItems, denominator, numerator, 1U, status);
        }
        case kLlaminarNativeQuantisedTreeThresholdAspectRatio:
            return nativeQuantisedProductLessEqual(
                aggregateN, denominator, k, numerator, status);
        default:
            break;
    }
    if (tileWidth == 0) {
        atomicOr(status, kNativeQuantisedTreeStatusEvaluationFailure);
        return false;
    }
    const std::uint64_t nTiles = nativeQuantisedCeilDiv(aggregateN, tileWidth);
    switch (threshold.operation) {
        case kLlaminarNativeQuantisedTreeThresholdNTiles:
            return nativeQuantisedProductLessEqual(
                nTiles, denominator, numerator, 1U, status);
        case kLlaminarNativeQuantisedTreeThresholdKGroupsPerNTile: {
            std::uint64_t scaledTiles = 0;
            if (!nativeQuantisedCheckedMultiply(32U, nTiles, &scaledTiles)) {
                atomicOr(status, kNativeQuantisedTreeStatusThresholdOverflow);
                return false;
            }
            return nativeQuantisedProductLessEqual(
                k, denominator, scaledTiles, numerator, status);
        }
        case kLlaminarNativeQuantisedTreeThresholdNFinalTile: {
            const std::uint64_t finalTile = (aggregateN - 1U) % tileWidth + 1U;
            return nativeQuantisedProductLessEqual(
                finalTile, denominator, numerator, 1U, status);
        }
        case kLlaminarNativeQuantisedTreeThresholdNTileUtilization: {
            std::uint64_t tiledWidth = 0;
            if (!nativeQuantisedCheckedMultiply(nTiles, tileWidth, &tiledWidth)) {
                atomicOr(status, kNativeQuantisedTreeStatusThresholdOverflow);
                return false;
            }
            return nativeQuantisedProductLessEqual(
                aggregateN, denominator, tiledWidth, numerator, status);
        }
        case kLlaminarNativeQuantisedTreeThresholdNTileAligned: {
            const std::uint64_t aligned = aggregateN % tileWidth == 0 ? 1U : 0U;
            return nativeQuantisedProductLessEqual(
                aligned, denominator, numerator, 1U, status);
        }
        case kLlaminarNativeQuantisedTreeThresholdKFinalTile: {
            const std::uint64_t finalTile = (k - 1U) % tileWidth + 1U;
            return nativeQuantisedProductLessEqual(
                finalTile, denominator, numerator, 1U, status);
        }
        case kLlaminarNativeQuantisedTreeThresholdKPartKBlocksPerTile:
        case kLlaminarNativeQuantisedTreeThresholdKPartFinalKTileBlocks:
        case kLlaminarNativeQuantisedTreeThresholdKPartFinalKTileUtilization: {
            NativeQuantisedKPartGeometry geometry = {};
            if (!nativeQuantisedComputeKPartGeometry(
                    k, tileWidth, launchKTiles, &geometry, status)) {
                return false;
            }
            if (threshold.operation
                == kLlaminarNativeQuantisedTreeThresholdKPartKBlocksPerTile) {
                return nativeQuantisedProductLessEqual(
                    geometry.blocksPerTile, denominator,
                    numerator, 1U, status);
            }
            if (threshold.operation
                == kLlaminarNativeQuantisedTreeThresholdKPartFinalKTileBlocks) {
                return nativeQuantisedProductLessEqual(
                    geometry.finalTileBlocks, denominator,
                    numerator, 1U, status);
            }
            return nativeQuantisedProductLessEqual(
                geometry.finalTileBlocks, denominator,
                geometry.blocksPerTile, numerator, status);
        }
        case kLlaminarNativeQuantisedTreeThresholdKPartKTileCount:
            return nativeQuantisedProductLessEqual(
                launchKTiles > 0 ? launchKTiles : 1U,
                denominator, numerator, 1U, status);
        default:
            break;
    }

    const std::uint64_t parallelism = threshold.parallelism_width;
    std::uint64_t tasks = nTiles;
    if (threshold.operation
            == kLlaminarNativeQuantisedTreeThresholdKPartProducerWaves
        || threshold.operation
            == kLlaminarNativeQuantisedTreeThresholdKPartFinalProducerWaveUtilization) {
        if (!nativeQuantisedCheckedMultiply(
                launchKTiles > 0 ? launchKTiles : 1U, nTiles, &tasks)) {
            atomicOr(status, kNativeQuantisedTreeStatusThresholdOverflow);
            return false;
        }
    }
    if (threshold.operation == kLlaminarNativeQuantisedTreeThresholdMNParallelWaves
        || threshold.operation
            == kLlaminarNativeQuantisedTreeThresholdMNFinalParallelWaveUtilization) {
        if (!nativeQuantisedCheckedMultiply(
                threshold.task_multiplier, nTiles, &tasks)) {
            atomicOr(status, kNativeQuantisedTreeStatusThresholdOverflow);
            return false;
        }
    }
    if (threshold.operation == kLlaminarNativeQuantisedTreeThresholdNParallelWaves
        || threshold.operation
            == kLlaminarNativeQuantisedTreeThresholdKPartProducerWaves
        || threshold.operation
            == kLlaminarNativeQuantisedTreeThresholdMNParallelWaves) {
        const std::uint64_t waves = nativeQuantisedCeilDiv(tasks, parallelism);
        return nativeQuantisedProductLessEqual(
            waves, denominator, numerator, 1U, status);
    }
    if (threshold.operation
            == kLlaminarNativeQuantisedTreeThresholdNFinalParallelWaveUtilization
        || threshold.operation
            == kLlaminarNativeQuantisedTreeThresholdKPartFinalProducerWaveUtilization
        || threshold.operation
            == kLlaminarNativeQuantisedTreeThresholdMNFinalParallelWaveUtilization) {
        const std::uint64_t finalTasks = (tasks - 1U) % parallelism + 1U;
        return nativeQuantisedProductLessEqual(
            finalTasks, denominator, parallelism, numerator, status);
    }
    atomicOr(status, kNativeQuantisedTreeStatusEvaluationFailure);
    return false;
}

/** Skip one preorder subtree and report how many candidate leaves it owns. */
__device__ bool nativeQuantisedSkipEvaluationSubtree(
    const LlaminarNativeQuantisedTreeFitResult& tree,
    std::uint32_t* tokenIndex,
    std::uint32_t* skippedLeaves) {
    std::uint32_t pendingNodes = 1;
    while (pendingNodes != 0 && *tokenIndex < tree.structure_token_count) {
        const std::uint32_t token = tree.structure_tokens[(*tokenIndex)++];
        --pendingNodes;
        if (token == 0) {
            ++*skippedLeaves;
        } else {
            pendingNodes += 2;
        }
    }
    return pendingNodes == 0;
}

/** Route one heldout point through a compact preorder tree entirely on-device. */
__device__ std::uint32_t nativeQuantisedSelectHeldoutCandidate(
    const LlaminarNativeQuantisedTreeFitResult& tree,
    std::uint64_t aggregateN,
    std::uint64_t k,
    std::uint32_t launchKTiles,
    std::uint32_t* status) {
    if (tree.leaf_count == 0 || tree.structure_token_count == 0) {
        return kNativeQuantisedTreeInvalidIndex;
    }
    std::uint32_t tokenIndex = 0;
    std::uint32_t leafIndex = 0;
    while (tokenIndex < tree.structure_token_count) {
        const std::uint32_t position = tokenIndex++;
        const std::uint32_t token = tree.structure_tokens[position];
        if (token == 0) {
            if (leafIndex >= tree.leaf_count) {
                atomicOr(status, kNativeQuantisedTreeStatusEvaluationFailure);
                return kNativeQuantisedTreeInvalidIndex;
            }
            return tree.candidate_indices[leafIndex];
        }
        if (token != 1U) {
            atomicOr(status, kNativeQuantisedTreeStatusEvaluationFailure);
            return kNativeQuantisedTreeInvalidIndex;
        }
        if (!nativeQuantisedThresholdMatchesLessEqual(
                tree.split_thresholds[position], aggregateN, k,
                launchKTiles, status)) {
            std::uint32_t skippedLeaves = 0;
            if (!nativeQuantisedSkipEvaluationSubtree(
                    tree, &tokenIndex, &skippedLeaves)) {
                atomicOr(status, kNativeQuantisedTreeStatusEvaluationFailure);
                return kNativeQuantisedTreeInvalidIndex;
            }
            leafIndex += skippedLeaves;
        }
    }
    atomicOr(status, kNativeQuantisedTreeStatusEvaluationFailure);
    return kNativeQuantisedTreeInvalidIndex;
}

/** Select the canonical measured exact winner once for each heldout point. */
__global__ void selectNativeQuantisedHeldoutExactCandidates(
    const double* measuredP95,
    const double* measuredMeans,
    const double* measuredMaxima,
    std::uint32_t heldoutPointCount,
    std::uint32_t candidateCount,
    std::uint32_t* exactCandidates,
    std::uint32_t* status) {
    const std::uint32_t lane = threadIdx.x % kNativeQuantisedLeafGroupThreads;
    const std::uint32_t warp = threadIdx.x / kNativeQuantisedLeafGroupThreads;
    const std::uint32_t point =
        blockIdx.x * kNativeQuantisedHeldoutExactWarpsPerBlock + warp;
    if (point >= heldoutPointCount
        || blockDim.x
            != kNativeQuantisedLeafGroupThreads
                * kNativeQuantisedHeldoutExactWarpsPerBlock) {
        return;
    }
    const std::size_t row =
        static_cast<std::size_t>(point) * candidateCount;
    std::uint32_t best = kNativeQuantisedTreeInvalidIndex;
    double bestMaximum = kNativeQuantisedInfinity;
    double bestP95 = kNativeQuantisedInfinity;
    double bestMean = kNativeQuantisedInfinity;
    for (std::uint32_t candidate = lane; candidate < candidateCount;
         candidate += kNativeQuantisedLeafGroupThreads) {
        const double maximum = measuredMaxima[row + candidate];
        const double p95 = measuredP95[row + candidate];
        const double mean = measuredMeans[row + candidate];
        if (!isfinite(maximum) || !isfinite(p95) || !isfinite(mean)) {
            continue;
        }
        if (best == kNativeQuantisedTreeInvalidIndex
            || maximum < bestMaximum
            || (maximum == bestMaximum
                && (p95 < bestP95
                    || (p95 == bestP95
                        && (mean < bestMean
                            || (mean == bestMean
                                && candidate < best)))))) {
            best = candidate;
            bestMaximum = maximum;
            bestP95 = p95;
            bestMean = mean;
        }
    }
    for (std::uint32_t offset = kNativeQuantisedLeafGroupThreads / 2U;
         offset != 0; offset >>= 1U) {
        const std::uint32_t other =
            nativeQuantisedSubgroupShuffleDown(best, offset);
        const double otherMaximum =
            nativeQuantisedSubgroupShuffleDown(bestMaximum, offset);
        const double otherP95 = nativeQuantisedSubgroupShuffleDown(bestP95, offset);
        const double otherMean = nativeQuantisedSubgroupShuffleDown(bestMean, offset);
        if (lane + offset < kNativeQuantisedLeafGroupThreads
            && other != kNativeQuantisedTreeInvalidIndex
            && (best == kNativeQuantisedTreeInvalidIndex
                || otherMaximum < bestMaximum
                || (otherMaximum == bestMaximum
                    && (otherP95 < bestP95
                        || (otherP95 == bestP95
                            && (otherMean < bestMean
                                || (otherMean == bestMean && other < best))))))) {
            best = other;
            bestMaximum = otherMaximum;
            bestP95 = otherP95;
            bestMean = otherMean;
        }
    }
    if (lane == 0) {
        exactCandidates[point] = best;
        if (best == kNativeQuantisedTreeInvalidIndex) {
            atomicOr(status, kNativeQuantisedTreeStatusEvaluationFailure);
        }
    }
}

/** Initialize fold headers before tiled evaluation blocks accumulate coverage. */
__global__ void initializeNativeQuantisedFoldEvaluationHeaders(
    LlaminarNativeQuantisedTreeFoldEvaluation* evaluations,
    std::uint32_t maxLeaves,
    std::uint32_t heldoutPointCount) {
    const std::uint32_t budget = blockIdx.x * blockDim.x + threadIdx.x;
    if (budget < maxLeaves) {
        evaluations[budget].required_point_count = heldoutPointCount;
        evaluations[budget].covered_point_count = 0;
    }
}

/**
 * Evaluate all maximum-leaf budgets in parallel without exporting tree bytes.
 *
 * One block owns one budget and one lane owns one heldout point. The shared
 * integer reduction publishes coverage without floating-point atomics or any
 * host-visible cardinality read inside the captured graph.
 */
__global__ void evaluateNativeQuantisedHeldoutTrees(
    const LlaminarNativeQuantisedTreeFitResult* trees,
    const std::uint64_t* heldoutAggregateN,
    const std::uint64_t* heldoutK,
    const std::uint32_t* heldoutLaunchKTiles,
    const double* heldoutMeasuredP95,
    std::uint32_t heldoutPointCount,
    std::uint32_t candidateCount,
    LlaminarNativeQuantisedTreeFoldEvaluation* evaluations,
    std::uint32_t* status) {
    __shared__ std::uint32_t covered[kNativeQuantisedTreeEvaluationThreads];
    const std::uint32_t pointTiles = static_cast<std::uint32_t>(
        nativeQuantisedCeilDiv(
            heldoutPointCount, kNativeQuantisedTreeEvaluationThreads));
    const std::uint32_t budget = blockIdx.x / pointTiles;
    const std::uint32_t point =
        (blockIdx.x % pointTiles) * kNativeQuantisedTreeEvaluationThreads
        + threadIdx.x;
    std::uint32_t selected = kNativeQuantisedTreeInvalidIndex;
    if (point < heldoutPointCount) {
        selected = nativeQuantisedSelectHeldoutCandidate(
            trees[budget], heldoutAggregateN[point], heldoutK[point],
            heldoutLaunchKTiles[point], status);
        if (selected != kNativeQuantisedTreeInvalidIndex) {
            if (selected >= candidateCount
                || !isfinite(heldoutMeasuredP95[
                    static_cast<std::size_t>(point) * candidateCount
                    + selected])) {
                selected = kNativeQuantisedTreeInvalidIndex;
            }
        }
        evaluations[budget].selected_candidate_indices[point] = selected;
    }
    // `point` is the global held-out row and can be larger than this block's
    // shared tile once evaluation crosses 256 rows.  The reduction storage is
    // block-local, so every thread must publish through its local lane.  Using
    // `point` here used to write beyond `covered` for the second and later
    // tiles, corrupting the exported cardinality while the independently
    // written selected-candidate rows often remained valid.
    covered[threadIdx.x] = static_cast<std::uint32_t>(
        point < heldoutPointCount && selected != kNativeQuantisedTreeInvalidIndex);
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2U; stride != 0; stride >>= 1U) {
        if (threadIdx.x < stride) {
            covered[threadIdx.x] += covered[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&evaluations[budget].covered_point_count, covered[0]);
    }
}

/** Publish the full-domain mask for candidate-parallel initial-leaf scoring. */
__global__ void initializeNativeQuantisedFullMask(
    NativeQuantisedPointMask* childMasks,
    std::uint32_t* childMaskCount,
    std::uint32_t pointCount) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        NativeQuantisedPointMask full = {};
        for (std::uint32_t word = 0;
             word < kLlaminarNativeQuantisedTreePointMaskWords; ++word) {
            const std::uint32_t firstPoint = word * 64U;
            if (firstPoint >= pointCount) {
                break;
            }
            const std::uint32_t remaining = pointCount - firstPoint;
            full.words[word] = remaining >= 64U
                ? ~std::uint64_t{0}
                : (std::uint64_t{1} << remaining) - 1U;
        }
        childMasks[0] = full;
        *childMaskCount = 1;
    }
}

/** Build the one-leaf frontier from the parallel-scored full-domain leaf. */
__global__ void buildNativeQuantisedInitialTree(
    NativeQuantisedDeviceTree* frontier,
    std::uint32_t* frontierCount,
    const NativeQuantisedDeviceLeaf* initialLeaf,
    const std::uint32_t* pointGroupRanks,
    std::uint32_t pointCount) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const NativeQuantisedDeviceLeaf* leaf = &initialLeaf[0];
    if (!leaf->valid) {
        frontier->leafCount = 0;
        *frontierCount = 0;
        return;
    }
    frontier->leafCount = 1;
    nativeQuantisedCopyLeaf(&frontier->leaves[0], leaf);
    frontier->signatureCount = nativeQuantisedWriteLeafSignature(
        frontier->signature, 0, *leaf, pointGroupRanks, pointCount);
    *frontierCount = 1;
}

/** Return the next power of two, failing through zero on uint32 overflow. */
std::uint32_t nativeQuantisedNextPowerOfTwo(std::uint64_t value) {
    std::uint64_t result = 1;
    while (result < value) {
        result <<= 1U;
        if (result > std::numeric_limits<std::uint32_t>::max()) {
            return 0;
        }
    }
    return static_cast<std::uint32_t>(result);
}

/** Grow one typed persistent allocation while preserving the old buffer on error. */
template <typename T>
bool ensureNativeQuantisedTreeBuffer(
    LlaminarNativeQuantisedLeafPrimaryScorerSession* session,
    T** pointer,
    std::size_t* capacity,
    std::size_t required,
    bool* grew,
    char* error,
    std::size_t errorCapacity,
    const char* operation) {
    if (*pointer != nullptr && required <= *capacity) {
        return true;
    }
    if (!nativeQuantisedScorerByteCountFits(required, sizeof(T))) {
        writeNativeQuantisedScorerError(error, errorCapacity, operation);
        return false;
    }
    T* replacement = nullptr;
    const NativeQuantisedScorerError result = allocateNativeQuantisedScorerBuffer(session,
        reinterpret_cast<void**>(&replacement), required * sizeof(T));
    if (result != kNativeQuantisedScorerSuccess) {
        (void)failNativeQuantisedScorer(error, errorCapacity, operation, result);
        return false;
    }
    invalidateNativeQuantisedTreeGraphs(session);
    (void)freeNativeQuantisedScorerBuffer(session, *pointer);
    *pointer = replacement;
    *capacity = required;
    *grew = true;
    return true;
}

/** Recompute the exact persistent device-byte high-water diagnostic. */
std::uint64_t nativeQuantisedTreeScratchBytes(const NativeQuantisedTreeScratch& scratch) {
    return (scratch.measuredMeanCapacity + scratch.measuredMaximumCapacity)
              * sizeof(double)
        + (scratch.fittingP95PointOrderCapacity
           + scratch.measuredP95PointOrderCapacity
           + scratch.frontierLcpCapacityA
           + scratch.frontierLcpCapacityB) * sizeof(std::uint16_t)
        + scratch.pointRankCapacity * sizeof(std::uint32_t)
        + (scratch.trainingAggregateNCapacity + scratch.trainingKCapacity)
              * sizeof(std::uint64_t)
        + (scratch.expandedCandidateTotalCapacity
           + scratch.structurallyUniqueCandidateTotalCapacity)
              * sizeof(std::uint64_t)
        + scratch.trainingLaunchKTilesCapacity * sizeof(std::uint32_t)
        + scratch.featureAxisCapacity
              * sizeof(LlaminarNativeQuantisedTreeFeatureAxis)
        + scratch.axisCapacity * sizeof(std::uint32_t)
        + (scratch.axisMaskCapacity + scratch.axisPrefixCapacity)
              * sizeof(NativeQuantisedPointMask)
        + (scratch.axisValueNumeratorCapacity
           + scratch.axisValueDenominatorCapacity) * sizeof(std::uint64_t)
        + (scratch.heldoutAggregateNCapacity + scratch.heldoutKCapacity)
              * sizeof(std::uint64_t)
        + scratch.heldoutLaunchKTilesCapacity * sizeof(std::uint32_t)
        + (scratch.heldoutMeasuredP95Capacity
           + scratch.heldoutMeasuredMeanCapacity
           + scratch.heldoutMeasuredMaximumCapacity) * sizeof(double)
        + scratch.heldoutExactCandidateCapacity * sizeof(std::uint32_t)
        + scratch.foldEvaluationCapacity
              * sizeof(LlaminarNativeQuantisedTreeFoldEvaluation)
        + (scratch.frontierCapacityA + scratch.frontierCapacityB
           + scratch.incumbentCapacity)
              * sizeof(NativeQuantisedDeviceTree)
        + scratch.expansionCapacity * sizeof(NativeQuantisedExpandedCandidate)
        + scratch.splitDescriptorCapacity * sizeof(NativeQuantisedSplitDescriptor)
        + scratch.expandedOrderKeyCapacity * sizeof(NativeQuantisedTreeOrderKey)
        + scratch.childMaskCapacity * sizeof(NativeQuantisedPointMask)
        + scratch.childLeafCapacity * sizeof(NativeQuantisedDeviceLeaf)
        + scratch.childFittingTailPointCapacity * sizeof(std::uint16_t)
        + (scratch.frontierCountCapacityA + scratch.frontierCountCapacityB
           + scratch.expandedCountCapacity + scratch.splitDescriptorCountCapacity
           + scratch.childMaskCountCapacity + scratch.uniqueCountCapacity
           + scratch.statusCapacity + scratch.hashCapacity
           + scratch.uniqueIndexCapacity + scratch.selectionCapacityA
           + scratch.selectionCapacityB + scratch.selectionCandidateCapacity
           + scratch.selectionCandidateCountCapacity
           + scratch.childUniqueIndexCapacity
           + scratch.maskHashCapacity + scratch.maskHashUniqueIndexCapacity)
              * sizeof(std::uint32_t)
        + scratch.selectionContextCapacity
              * sizeof(NativeQuantisedTreeSelectionContext)
        + scratch.resultCapacity * sizeof(LlaminarNativeQuantisedTreeFitResult);
}

/** Ensure every metadata and search array can serve the requested geometry. */
bool ensureNativeQuantisedTreeScratch(
    LlaminarNativeQuantisedLeafPrimaryScorerSession* session,
    std::size_t matrixCount,
    std::size_t pointCount,
    std::size_t axisCount,
    std::size_t axisSlotCount,
    std::size_t heldoutPointCount,
    bool evaluateHeldout,
    std::uint32_t expansionCapacity,
    std::uint32_t maxLeaves,
    char* error,
    std::size_t errorCapacity) {
    auto* scratch = static_cast<NativeQuantisedTreeScratch*>(session->treeScratch);
    if (scratch == nullptr) {
        scratch = new (std::nothrow) NativeQuantisedTreeScratch();
        if (scratch == nullptr) {
            writeNativeQuantisedScorerError(
                error, errorCapacity, "tree scratch host allocation failed");
            return false;
        }
        // Graph entries are bounded for the complete session lifetime. Reserve
        // their host metadata once with the scratch owner so publishing a newly
        // captured geometry never triggers a vector reallocation in the fit
        // path. Device buffers below follow the same persistent high-water
        // ownership rule and are fully established before capture begins.
        scratch->graphs.reserve(kNativeQuantisedTreeMaximumCachedGraphs);
        session->treeScratch = scratch;
    }
    const std::uint32_t treeHashCapacity =
        nativeQuantisedNextPowerOfTwo(static_cast<std::uint64_t>(expansionCapacity) * 2U);
    const std::uint64_t childCapacity64 =
        static_cast<std::uint64_t>(expansionCapacity) * 2U;
    const std::uint32_t maskHashCapacity =
        nativeQuantisedNextPowerOfTwo(childCapacity64 * 2U);
    if (treeHashCapacity == 0 || maskHashCapacity == 0
        || childCapacity64 > std::numeric_limits<std::uint32_t>::max()) {
        writeNativeQuantisedScorerError(
            error, errorCapacity, "tree or child-mask capacity overflows uint32");
        return false;
    }
    const std::uint32_t childCapacity =
        static_cast<std::uint32_t>(childCapacity64);
    const std::size_t childFittingTailPointCapacity =
        static_cast<std::size_t>(childCapacity)
        * kNativeQuantisedLeafP95TailIterations;
    const std::size_t selectionCapacity =
        ((static_cast<std::size_t>(expansionCapacity)
          + kNativeQuantisedTreeSelectionChunk - 1U)
         / kNativeQuantisedTreeSelectionChunk)
        * kNativeQuantisedTreeSelectionRetained;
    const std::size_t selectionBlockCapacity =
        (static_cast<std::size_t>(expansionCapacity)
         + kNativeQuantisedTreeSelectionChunk - 1U)
        / kNativeQuantisedTreeSelectionChunk;
    const std::size_t evaluationPointCount =
        evaluateHeldout ? heldoutPointCount : 1U;
    const std::size_t evaluationMatrixCount = evaluateHeldout
        ? heldoutPointCount * session->candidateCount : 1U;
    bool grew = false;
#define LLAMINAR_ENSURE_TREE_BUFFER(member, capacity, required, message) \
    ensureNativeQuantisedTreeBuffer(session, &scratch->member, &scratch->capacity, \
        required, &grew, error, errorCapacity, message)
    const bool ready =
        LLAMINAR_ENSURE_TREE_BUFFER(measuredMeans, measuredMeanCapacity,
            matrixCount, "measured-mean high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(measuredMaxima, measuredMaximumCapacity,
            matrixCount, "measured-maximum high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(fittingP95PointOrder,
            fittingP95PointOrderCapacity, matrixCount,
            "fitting-p95 point-order high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(measuredP95PointOrder,
            measuredP95PointOrderCapacity, matrixCount,
            "measured-p95 point-order high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(pointGroupRanks, pointRankCapacity,
            pointCount, "point-rank high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(trainingAggregateN,
            trainingAggregateNCapacity, pointCount,
            "training-N high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(trainingK, trainingKCapacity,
            pointCount, "training-K high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(trainingLaunchKTiles,
            trainingLaunchKTilesCapacity, pointCount,
            "training-K-tiles high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(featureAxes, featureAxisCapacity,
            axisCount, "feature-axis high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(axisValueCounts, axisCapacity,
            axisCount, "axis-count high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(axisValueMasks, axisMaskCapacity,
            axisSlotCount, "axis-mask high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(axisPrefixMasks, axisPrefixCapacity,
            axisSlotCount, "axis-prefix high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(axisValueNumerators,
            axisValueNumeratorCapacity, axisSlotCount,
            "axis-value-numerator high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(axisValueDenominators,
            axisValueDenominatorCapacity, axisSlotCount,
            "axis-value-denominator high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(heldoutAggregateN,
            heldoutAggregateNCapacity, evaluationPointCount,
            "heldout-N high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(heldoutK, heldoutKCapacity,
            evaluationPointCount, "heldout-K high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(heldoutLaunchKTiles,
            heldoutLaunchKTilesCapacity, evaluationPointCount,
            "heldout-K-tiles high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(heldoutMeasuredP95,
            heldoutMeasuredP95Capacity, evaluationMatrixCount,
            "heldout-p95 high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(heldoutMeasuredMeans,
            heldoutMeasuredMeanCapacity, evaluationMatrixCount,
            "heldout-mean high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(heldoutMeasuredMaxima,
            heldoutMeasuredMaximumCapacity, evaluationMatrixCount,
            "heldout-maximum high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(heldoutExactCandidates,
            heldoutExactCandidateCapacity, evaluationPointCount,
            "heldout-exact-candidate high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(foldEvaluations,
            foldEvaluationCapacity, maxLeaves,
            "fold-evaluation high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(frontierA, frontierCapacityA,
            kNativeQuantisedTreeBeamWidth, "frontier-A high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(frontierB, frontierCapacityB,
            kNativeQuantisedTreeBeamWidth, "frontier-B high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(incumbent, incumbentCapacity,
            1, "incumbent high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(expanded, expansionCapacity,
            expansionCapacity, "expanded-tree high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(expandedOrderKeys,
            expandedOrderKeyCapacity, expansionCapacity,
            "expanded-order-key high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(splitDescriptors, splitDescriptorCapacity,
            expansionCapacity, "split-descriptor high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(childMasks, childMaskCapacity,
            childCapacity, "child-mask high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(childLeaves, childLeafCapacity,
            childCapacity, "child-leaf high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(childFittingTailPoints,
            childFittingTailPointCapacity, childFittingTailPointCapacity,
            "child fitting-tail high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(childUniqueIndices,
            childUniqueIndexCapacity, childCapacity,
            "child-unique-index high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(maskHashRepresentatives,
            maskHashCapacity, maskHashCapacity,
            "mask-hash representative high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(maskHashUniqueIndices,
            maskHashUniqueIndexCapacity, maskHashCapacity,
            "mask-hash unique-index high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(frontierCountA, frontierCountCapacityA,
            1, "frontier-count-A high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(frontierCountB, frontierCountCapacityB,
            1, "frontier-count-B high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(expandedCount, expandedCountCapacity,
            1, "expanded-count high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(splitDescriptorCount,
            splitDescriptorCountCapacity, 1,
            "split-descriptor-count high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(childMaskCount, childMaskCountCapacity,
            1, "child-mask-count high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(uniqueCount, uniqueCountCapacity,
            1, "unique-count high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(status, statusCapacity,
            1, "tree-status high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(expandedCandidateTotal,
            expandedCandidateTotalCapacity, 1,
            "expanded-candidate diagnostic high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(structurallyUniqueCandidateTotal,
            structurallyUniqueCandidateTotalCapacity, 1,
            "unique-candidate diagnostic high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(hashSlots, hashCapacity,
            treeHashCapacity, "signature-hash high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(uniqueIndices, uniqueIndexCapacity,
            expansionCapacity, "unique-index high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(selectionA, selectionCapacityA,
            selectionCapacity, "selection-A high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(selectionB, selectionCapacityB,
            selectionCapacity, "selection-B high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(selectionCandidates,
            selectionCandidateCapacity, expansionCapacity,
            "selection-candidate high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(selectionCandidateCounts,
            selectionCandidateCountCapacity, selectionBlockCapacity,
            "selection-candidate-count high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(frontierLcpA,
            frontierLcpCapacityA,
            kNativeQuantisedTreeBeamWidth * kNativeQuantisedTreeBeamWidth,
            "frontier-LCP-A high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(frontierLcpB,
            frontierLcpCapacityB,
            kNativeQuantisedTreeBeamWidth * kNativeQuantisedTreeBeamWidth,
            "frontier-LCP-B high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(selectionContexts,
            selectionContextCapacity, 2,
            "selection-context high-water growth")
        && LLAMINAR_ENSURE_TREE_BUFFER(results, resultCapacity,
            maxLeaves, "tree-result high-water growth");
#undef LLAMINAR_ENSURE_TREE_BUFFER
    if (!ready) {
        return false;
    }
    if (grew) {
        ++session->runtimeStats.tree_scratch_growth_count;
        session->runtimeStats.tree_scratch_high_water_bytes =
            nativeQuantisedTreeScratchBytes(*scratch);
    }
    session->runtimeStats.last_expansion_capacity = expansionCapacity;
    return true;
}

/**
 * Launch the child-mask scorer with storage specialized to its active groups.
 *
 * The tree ABI admits arbitrary candidate counts by processing chunks. CUDA
 * launches one 32-lane group per candidate up to eight; ROCm rounds the group
 * count to an even number so each wave64 contains two complete subgroups. The
 * template specialization prevents a three-candidate domain from reserving
 * shared memory for eight candidates throughout a graph replay.
 */
void launchNativeQuantisedChildMaskScorer(
    LlaminarNativeQuantisedLeafPrimaryScorerSession* session,
    NativeQuantisedTreeScratch* scratch,
    std::uint32_t blockCount,
    std::uint32_t childMaskCapacity,
    std::uint32_t minShapeGroupsPerLeaf) {
    const std::uint32_t groupCount =
        nativeQuantisedLeafScoreThreadCount(session->candidateCount)
        / kNativeQuantisedLeafGroupThreads;
    const std::uint32_t threadCount =
        groupCount * kNativeQuantisedLeafGroupThreads;
#define LLAMINAR_LAUNCH_CHILD_MASK_SCORER(candidate_groups) \
    scoreNativeQuantisedChildMaskTiles<candidate_groups><<< \
        blockCount, threadCount, 0, session->stream>>>( \
        scratch->childMasks, scratch->childMaskCount, childMaskCapacity, \
        session->deviceFittingRegrets, session->deviceMeasuredP95Regrets, \
        scratch->measuredMeans, scratch->measuredMaxima, \
        scratch->fittingP95PointOrder, scratch->measuredP95PointOrder, \
        scratch->pointGroupRanks, session->pointCount, \
        session->candidateCount, minShapeGroupsPerLeaf, scratch->childLeaves)
    switch (groupCount) {
        case 1:
            LLAMINAR_LAUNCH_CHILD_MASK_SCORER(1);
            break;
        case 2:
            LLAMINAR_LAUNCH_CHILD_MASK_SCORER(2);
            break;
        case 3:
            LLAMINAR_LAUNCH_CHILD_MASK_SCORER(3);
            break;
        case 4:
            LLAMINAR_LAUNCH_CHILD_MASK_SCORER(4);
            break;
        case 5:
            LLAMINAR_LAUNCH_CHILD_MASK_SCORER(5);
            break;
        case 6:
            LLAMINAR_LAUNCH_CHILD_MASK_SCORER(6);
            break;
        case 7:
            LLAMINAR_LAUNCH_CHILD_MASK_SCORER(7);
            break;
        case 8:
            LLAMINAR_LAUNCH_CHILD_MASK_SCORER(8);
            break;
    }
#undef LLAMINAR_LAUNCH_CHILD_MASK_SCORER
}

/** Launch the fixed complete-search pipeline onto the session stream. */
void launchNativeQuantisedTreePipeline(
    LlaminarNativeQuantisedLeafPrimaryScorerSession* session,
    NativeQuantisedTreeScratch* scratch,
    std::uint32_t axisCount,
    std::uint32_t minShapeGroupsPerLeaf,
    std::uint32_t maxLeaves,
    std::uint32_t expansionCapacity,
    std::uint32_t boundaryPlacement,
    std::uint32_t parallelismWidth,
    std::uint32_t taskMultiplier,
    std::uint32_t heldoutPointCount,
    bool evaluateHeldout) {
    (void)LLAMINAR_SCORER_MEMSET_ASYNC(
        scratch->status, 0, sizeof(std::uint32_t), session->stream);
    (void)LLAMINAR_SCORER_MEMSET_ASYNC(
        scratch->expandedCandidateTotal, 0, sizeof(std::uint64_t),
        session->stream);
    (void)LLAMINAR_SCORER_MEMSET_ASYNC(
        scratch->structurallyUniqueCandidateTotal, 0, sizeof(std::uint64_t),
        session->stream);
    (void)LLAMINAR_SCORER_MEMSET_ASYNC(
        scratch->results, 0,
        maxLeaves * sizeof(LlaminarNativeQuantisedTreeFitResult), session->stream);
    (void)LLAMINAR_SCORER_MEMSET_ASYNC(
        scratch->incumbent, 0, sizeof(NativeQuantisedDeviceTree), session->stream);
    buildNativeQuantisedAxisMetadata<<<
        axisCount, kLlaminarNativeQuantisedTreeMaximumPoints, 0,
        session->stream>>>(
        scratch->trainingAggregateN, scratch->trainingK,
        scratch->trainingLaunchKTiles,
        scratch->featureAxes, session->pointCount, parallelismWidth,
            taskMultiplier, scratch->axisValueCounts,
            scratch->axisValueMasks, scratch->axisPrefixMasks,
            scratch->axisValueNumerators, scratch->axisValueDenominators,
            scratch->status);
    buildNativeQuantisedP95PointOrders<<<
        session->candidateCount * 2U,
        kLlaminarNativeQuantisedTreeMaximumPoints, 0, session->stream>>>(
        session->deviceFittingRegrets, session->deviceMeasuredP95Regrets,
        session->pointCount, session->candidateCount,
        scratch->fittingP95PointOrder, scratch->measuredP95PointOrder);
    initializeNativeQuantisedFullMask<<<1, 1, 0, session->stream>>>(
        scratch->childMasks, scratch->childMaskCount, session->pointCount);
    launchNativeQuantisedChildMaskScorer(
        session, scratch, 1, 1, minShapeGroupsPerLeaf);
    buildNativeQuantisedInitialTree<<<1, 1, 0, session->stream>>>(
        scratch->frontierA, scratch->frontierCountA, scratch->childLeaves,
        scratch->pointGroupRanks, session->pointCount);
    computeNativeQuantisedTreeObjectives<<<1, 1, 0, session->stream>>>(
        scratch->frontierA, scratch->frontierCountA, 1,
        session->deviceFittingRegrets, session->pointCount,
        session->candidateCount);
    buildNativeQuantisedTreeFittingOrders<<<
        1, kLlaminarNativeQuantisedTreeMaximumPoints, 0, session->stream>>>(
        scratch->frontierA, scratch->frontierCountA, 1,
        session->deviceFittingRegrets, session->pointCount,
        session->candidateCount);
    buildNativeQuantisedFrontierLcp<<<
        kNativeQuantisedTreeBeamWidth, kNativeQuantisedTreeBeamWidth, 0,
        session->stream>>>(
        scratch->frontierA, scratch->frontierCountA, scratch->frontierLcpA);
    updateNativeQuantisedTreeIncumbent<<<
        1, kNativeQuantisedTreeThreads, 0, session->stream>>>(
        scratch->frontierA, scratch->frontierCountA, scratch->incumbent,
        scratch->results, 0, scratch->status);
    initializeNativeQuantisedTreeSelectionContexts<<<1, 1, 0, session->stream>>>(
        scratch->selectionContexts, scratch->expanded,
        scratch->expandedOrderKeys, scratch->frontierA, scratch->frontierB,
        scratch->splitDescriptors, scratch->childUniqueIndices,
        scratch->childLeaves, scratch->pointGroupRanks,
        scratch->frontierLcpA, scratch->frontierLcpB, session->pointCount);

    NativeQuantisedDeviceTree* inputFrontier = scratch->frontierA;
    NativeQuantisedDeviceTree* outputFrontier = scratch->frontierB;
    std::uint32_t* inputCount = scratch->frontierCountA;
    std::uint32_t* outputCount = scratch->frontierCountB;
    for (std::uint32_t exactLeafCount = 2;
         exactLeafCount <= maxLeaves; ++exactLeafCount) {
        const std::uint32_t launchFrontiers = exactLeafCount == 2
            ? 1U : kNativeQuantisedTreeBeamWidth;
        const std::uint64_t combinationCount =
            static_cast<std::uint64_t>(launchFrontiers)
            * (exactLeafCount - 1U) * axisCount * session->pointCount;
        const std::uint32_t depthExpansionCapacity =
            static_cast<std::uint32_t>(std::min(
                combinationCount,
                static_cast<std::uint64_t>(expansionCapacity)));
        const std::uint32_t depthChildCapacity = depthExpansionCapacity * 2U;
        const std::uint32_t depthTreeHashCapacity = nativeQuantisedNextPowerOfTwo(
            static_cast<std::uint64_t>(depthExpansionCapacity) * 2U);
        const std::uint32_t depthMaskHashCapacity = nativeQuantisedNextPowerOfTwo(
            static_cast<std::uint64_t>(depthChildCapacity) * 2U);
        (void)LLAMINAR_SCORER_MEMSET_ASYNC(
            scratch->expandedCount, 0, sizeof(std::uint32_t), session->stream);
        (void)LLAMINAR_SCORER_MEMSET_ASYNC(
            scratch->splitDescriptorCount, 0, sizeof(std::uint32_t),
            session->stream);
        (void)LLAMINAR_SCORER_MEMSET_ASYNC(
            scratch->childMaskCount, 0, sizeof(std::uint32_t), session->stream);
        (void)LLAMINAR_SCORER_MEMSET_ASYNC(
            scratch->uniqueCount, 0, sizeof(std::uint32_t), session->stream);
        (void)LLAMINAR_SCORER_MEMSET_ASYNC(
            scratch->hashSlots, 0xff,
            depthTreeHashCapacity * sizeof(std::uint32_t),
            session->stream);
        (void)LLAMINAR_SCORER_MEMSET_ASYNC(
            scratch->maskHashRepresentatives, 0xff,
            depthMaskHashCapacity * sizeof(std::uint32_t), session->stream);

        const dim3 descriptorGrid(
            axisCount, exactLeafCount - 1U, launchFrontiers);
        enumerateNativeQuantisedSplitDescriptors<<<
            descriptorGrid, kNativeQuantisedTreeEnumerationThreads, 0,
            session->stream>>>(
            inputFrontier, inputCount, scratch->featureAxes,
            scratch->axisValueCounts,
            scratch->axisValueMasks, scratch->axisPrefixMasks,
            scratch->axisValueNumerators, scratch->axisValueDenominators,
            session->pointCount, axisCount, boundaryPlacement,
            parallelismWidth, taskMultiplier, scratch->splitDescriptors,
            depthExpansionCapacity, scratch->splitDescriptorCount,
            scratch->status);

        (void)LLAMINAR_SCORER_MEMSET_ASYNC(
            scratch->childUniqueIndices, 0xff,
            depthChildCapacity * sizeof(std::uint32_t), session->stream);
        const std::uint32_t childBlocks = std::min(
            (depthChildCapacity + kNativeQuantisedTreeThreads - 1U)
                / kNativeQuantisedTreeThreads,
            kNativeQuantisedScorerMaximumBlocks);
        insertNativeQuantisedChildMaskRepresentatives<<<
            childBlocks, kNativeQuantisedTreeThreads, 0, session->stream>>>(
            scratch->splitDescriptors, scratch->splitDescriptorCount,
            depthExpansionCapacity, scratch->maskHashRepresentatives,
            depthMaskHashCapacity,
            scratch->status);
        const std::uint32_t maskHashBlocks = std::min(
            static_cast<std::uint32_t>(
                (depthMaskHashCapacity + kNativeQuantisedTreeThreads - 1U)
                / kNativeQuantisedTreeThreads),
            kNativeQuantisedScorerMaximumBlocks);
        compactNativeQuantisedChildMasks<<<
            maskHashBlocks, kNativeQuantisedTreeThreads, 0, session->stream>>>(
            scratch->splitDescriptors, scratch->maskHashRepresentatives,
            scratch->maskHashUniqueIndices,
            depthMaskHashCapacity, scratch->childMasks, depthChildCapacity,
            scratch->childMaskCount,
            scratch->status);
        mapNativeQuantisedDescriptorChildren<<<
            childBlocks, kNativeQuantisedTreeThreads, 0, session->stream>>>(
            scratch->splitDescriptors, scratch->splitDescriptorCount,
            depthExpansionCapacity, scratch->maskHashRepresentatives,
            scratch->maskHashUniqueIndices,
            depthMaskHashCapacity,
            scratch->childUniqueIndices, scratch->status);

        const std::uint32_t scoreBlocks = std::min(
            (depthChildCapacity + kNativeQuantisedLeafGroupThreads - 1U)
                / kNativeQuantisedLeafGroupThreads,
            kNativeQuantisedScorerMaximumBlocks);
        launchNativeQuantisedChildMaskScorer(
            session, scratch, scoreBlocks, depthChildCapacity,
            minShapeGroupsPerLeaf);
        const std::uint32_t childTailBlocks = std::min(
            (depthChildCapacity + kNativeQuantisedExpandedHashGroups - 1U)
                / kNativeQuantisedExpandedHashGroups,
            kNativeQuantisedScorerMaximumBlocks);
        buildNativeQuantisedChildFittingTails<<<
            childTailBlocks, kNativeQuantisedTreeThreads, 0, session->stream>>>(
            scratch->childLeaves, scratch->childMaskCount,
            depthChildCapacity, session->deviceFittingRegrets,
            session->pointCount, session->candidateCount,
            scratch->childFittingTailPoints);
        const std::uint32_t objectiveBlocks = std::min(
            (depthExpansionCapacity + kNativeQuantisedTreeThreads - 1U)
                / kNativeQuantisedTreeThreads,
            kNativeQuantisedScorerMaximumBlocks);
        scoreNativeQuantisedExpandedCandidates<<<
            objectiveBlocks, kNativeQuantisedTreeThreads, 0, session->stream>>>(
            inputFrontier, scratch->splitDescriptors,
            scratch->splitDescriptorCount, depthExpansionCapacity,
            scratch->childUniqueIndices, scratch->childMaskCount,
            depthChildCapacity,
            scratch->childLeaves, scratch->childFittingTailPoints,
            session->deviceFittingRegrets, session->pointCount,
            session->candidateCount, scratch->expanded,
            depthExpansionCapacity, scratch->expandedCount, scratch->status);

        const std::uint32_t dedupeBlocks = std::min(
            (depthExpansionCapacity + kNativeQuantisedTreeThreads - 1U)
                / kNativeQuantisedTreeThreads,
            kNativeQuantisedScorerMaximumBlocks);
        deduplicateNativeQuantisedTrees<<<
            dedupeBlocks, kNativeQuantisedTreeThreads, 0, session->stream>>>(
            scratch->expanded, scratch->expandedCount, depthExpansionCapacity,
            inputFrontier, scratch->splitDescriptors,
            scratch->childUniqueIndices, scratch->childLeaves,
            scratch->pointGroupRanks, session->pointCount,
            scratch->hashSlots, depthTreeHashCapacity,
            scratch->uniqueIndices, scratch->uniqueCount, scratch->status);

        extractNativeQuantisedTreeOrderKeys<<<
            objectiveBlocks, kNativeQuantisedTreeThreads, 0, session->stream>>>(
            scratch->expanded, scratch->expandedCount,
            scratch->uniqueIndices, scratch->uniqueCount,
            depthExpansionCapacity, scratch->expandedCandidateTotal,
            scratch->structurallyUniqueCandidateTotal,
            scratch->expandedOrderKeys);

        const std::uint32_t* selectionInput = scratch->uniqueIndices;
        const std::uint32_t* deviceSelectionCount = scratch->uniqueCount;
        std::uint32_t selectionInputCapacity = depthExpansionCapacity;
        std::uint32_t* selectionOutput = scratch->selectionA;
        const NativeQuantisedTreeSelectionContext* selectionContext =
            inputFrontier == scratch->frontierA
            ? &scratch->selectionContexts[0]
            : &scratch->selectionContexts[1];
        bool firstSelectionPass = true;
        while (true) {
            const std::uint32_t selectionBlocks =
                (selectionInputCapacity + kNativeQuantisedTreeSelectionChunk - 1U)
                / kNativeQuantisedTreeSelectionChunk;
            if (firstSelectionPass) {
                narrowNativeQuantisedTreeTopIndices<<<
                    selectionBlocks, kNativeQuantisedTreeSelectionChunk, 0,
                    session->stream>>>(
                    selectionContext, selectionInput, selectionInputCapacity,
                    deviceSelectionCount, scratch->selectionCandidates,
                    scratch->selectionCandidateCounts);
                sortNativeQuantisedTreeNarrowedIndices<<<
                    selectionBlocks, kNativeQuantisedTreeSelectionChunk, 0,
                    session->stream>>>(
                    selectionContext, scratch->selectionCandidates,
                    scratch->selectionCandidateCounts, selectionOutput);
            } else {
                mergeNativeQuantisedTreeTopIndices<<<
                    selectionBlocks, kNativeQuantisedTreeSelectionChunk, 0,
                    session->stream>>>(
                    selectionContext, selectionInput, selectionInputCapacity,
                    selectionOutput);
            }
            const std::uint32_t outputCapacity =
                selectionBlocks * kNativeQuantisedTreeSelectionRetained;
            if (outputCapacity <= kNativeQuantisedTreeBeamWidth) {
                selectionInput = selectionOutput;
                break;
            }
            selectionInput = selectionOutput;
            selectionInputCapacity = outputCapacity;
            deviceSelectionCount = nullptr;
            selectionOutput = selectionOutput == scratch->selectionA
                ? scratch->selectionB : scratch->selectionA;
            firstSelectionPass = false;
        }

        gatherNativeQuantisedTreeFrontier<<<
            kNativeQuantisedTreeBeamWidth, kNativeQuantisedTreeThreads, 0,
            session->stream>>>(
            scratch->expanded, inputFrontier, scratch->splitDescriptors,
            scratch->childUniqueIndices, scratch->childLeaves,
            scratch->pointGroupRanks, session->pointCount, selectionInput,
            outputFrontier, outputCount);
        std::uint16_t* outputFrontierLcp = outputFrontier == scratch->frontierA
            ? scratch->frontierLcpA : scratch->frontierLcpB;
        buildNativeQuantisedFrontierLcp<<<
            kNativeQuantisedTreeBeamWidth, kNativeQuantisedTreeBeamWidth, 0,
            session->stream>>>(outputFrontier, outputCount, outputFrontierLcp);
        buildNativeQuantisedTreeFittingOrders<<<
            kNativeQuantisedTreeBeamWidth,
            kLlaminarNativeQuantisedTreeMaximumPoints, 0, session->stream>>>(
            outputFrontier, outputCount, kNativeQuantisedTreeBeamWidth,
            session->deviceFittingRegrets, session->pointCount,
            session->candidateCount);
        updateNativeQuantisedTreeIncumbent<<<
            1, kNativeQuantisedTreeThreads, 0, session->stream>>>(
            outputFrontier, outputCount, scratch->incumbent, scratch->results,
            exactLeafCount - 1U, scratch->status);
        std::swap(inputFrontier, outputFrontier);
        std::swap(inputCount, outputCount);
    }
    if (evaluateHeldout) {
        (void)LLAMINAR_SCORER_MEMSET_ASYNC(
            scratch->heldoutExactCandidates, 0xff,
            heldoutPointCount * sizeof(std::uint32_t), session->stream);
        (void)LLAMINAR_SCORER_MEMSET_ASYNC(
            scratch->foldEvaluations, 0xff,
            maxLeaves * sizeof(LlaminarNativeQuantisedTreeFoldEvaluation),
            session->stream);
        selectNativeQuantisedHeldoutExactCandidates<<<
            (heldoutPointCount + kNativeQuantisedHeldoutExactWarpsPerBlock - 1U)
                / kNativeQuantisedHeldoutExactWarpsPerBlock,
            kNativeQuantisedLeafGroupThreads
                * kNativeQuantisedHeldoutExactWarpsPerBlock,
            0, session->stream>>>(
            scratch->heldoutMeasuredP95, scratch->heldoutMeasuredMeans,
            scratch->heldoutMeasuredMaxima, heldoutPointCount,
            session->candidateCount, scratch->heldoutExactCandidates,
            scratch->status);
        initializeNativeQuantisedFoldEvaluationHeaders<<<
            1, kNativeQuantisedTreeThreads, 0, session->stream>>>(
            scratch->foldEvaluations, maxLeaves, heldoutPointCount);
        const std::uint32_t evaluationPointTiles =
            (heldoutPointCount + kNativeQuantisedTreeEvaluationThreads - 1U)
            / kNativeQuantisedTreeEvaluationThreads;
        evaluateNativeQuantisedHeldoutTrees<<<
            maxLeaves * evaluationPointTiles,
            kNativeQuantisedTreeEvaluationThreads, 0, session->stream>>>(
            scratch->results, scratch->heldoutAggregateN, scratch->heldoutK,
            scratch->heldoutLaunchKTiles,
            scratch->heldoutMeasuredP95, heldoutPointCount,
            session->candidateCount, scratch->foldEvaluations,
            scratch->status);
    }
}

/**
 * Begin capture for the caller's lane without serializing sibling lane setup.
 *
 * Every scorer session owns its stream and all commands that enter its graph
 * are issued by one lane thread. Thread-local capture therefore expresses the
 * real dependency boundary and permits other sessions on the same device to
 * capture or replay their independent graphs concurrently.
 */
NativeQuantisedScorerError beginNativeQuantisedTreeGraphCapture(
    NativeQuantisedScorerStream stream) {
#if defined(LLAMINAR_NATIVE_QUANTISED_SCORER_CUDA)
    return cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
#else
    return hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal);
#endif
}

/** End stream capture and return the newly owned graph. */
NativeQuantisedScorerError endNativeQuantisedTreeGraphCapture(
    NativeQuantisedScorerStream stream, NativeQuantisedScorerGraph* graph) {
#if defined(LLAMINAR_NATIVE_QUANTISED_SCORER_CUDA)
    return cudaStreamEndCapture(stream, graph);
#else
    return hipStreamEndCapture(stream, graph);
#endif
}

/** Instantiate an executable graph without upload or implicit launch. */
NativeQuantisedScorerError instantiateNativeQuantisedTreeGraph(
    NativeQuantisedScorerGraphExec* executable, NativeQuantisedScorerGraph graph) {
#if defined(LLAMINAR_NATIVE_QUANTISED_SCORER_CUDA)
    return cudaGraphInstantiate(executable, graph, nullptr, nullptr, 0);
#else
    return hipGraphInstantiate(executable, graph, nullptr, nullptr, 0);
#endif
}

/** Launch one previously captured complete search. */
NativeQuantisedScorerError launchNativeQuantisedTreeGraph(
    NativeQuantisedScorerGraphExec executable, NativeQuantisedScorerStream stream) {
#if defined(LLAMINAR_NATIVE_QUANTISED_SCORER_CUDA)
    return cudaGraphLaunch(executable, stream);
#else
    return hipGraphLaunch(executable, stream);
#endif
}

/** Destroy one executable graph. */
NativeQuantisedScorerError destroyNativeQuantisedTreeGraphExec(
    NativeQuantisedScorerGraphExec executable) {
#if defined(LLAMINAR_NATIVE_QUANTISED_SCORER_CUDA)
    return cudaGraphExecDestroy(executable);
#else
    return hipGraphExecDestroy(executable);
#endif
}

/** Destroy one captured graph definition. */
NativeQuantisedScorerError destroyNativeQuantisedTreeGraph(
    NativeQuantisedScorerGraph graph) {
#if defined(LLAMINAR_NATIVE_QUANTISED_SCORER_CUDA)
    return cudaGraphDestroy(graph);
#else
    return hipGraphDestroy(graph);
#endif
}

} // namespace

void invalidateNativeQuantisedTreeGraphs(
    LlaminarNativeQuantisedLeafPrimaryScorerSession* session) {
    if (session == nullptr || session->treeScratch == nullptr) {
        return;
    }
    auto* scratch = static_cast<NativeQuantisedTreeScratch*>(session->treeScratch);
    for (auto& entry : scratch->graphs) {
        if (entry.executable != nullptr) {
            (void)destroyNativeQuantisedTreeGraphExec(entry.executable);
        }
        if (entry.graph != nullptr) {
            (void)destroyNativeQuantisedTreeGraph(entry.graph);
        }
    }
    scratch->graphs.clear();
}

NativeQuantisedScorerError releaseNativeQuantisedTreeScratch(
    LlaminarNativeQuantisedLeafPrimaryScorerSession* session) {
    if (session == nullptr || session->treeScratch == nullptr) {
        return kNativeQuantisedScorerSuccess;
    }
    auto* scratch = static_cast<NativeQuantisedTreeScratch*>(session->treeScratch);
    NativeQuantisedScorerError first = kNativeQuantisedScorerSuccess;
    auto record = [&](NativeQuantisedScorerError result) {
        if (first == kNativeQuantisedScorerSuccess
            && result != kNativeQuantisedScorerSuccess) {
            first = result;
        }
    };
    for (auto& entry : scratch->graphs) {
        if (entry.executable != nullptr) {
            record(destroyNativeQuantisedTreeGraphExec(entry.executable));
        }
        if (entry.graph != nullptr) {
            record(destroyNativeQuantisedTreeGraph(entry.graph));
        }
    }
    auto release = [&](void* pointer) {
        if (pointer != nullptr) {
            record(freeNativeQuantisedScorerBuffer(session, pointer));
        }
    };
    release(scratch->measuredMeans);
    release(scratch->measuredMaxima);
    release(scratch->fittingP95PointOrder);
    release(scratch->measuredP95PointOrder);
    release(scratch->pointGroupRanks);
    release(scratch->trainingAggregateN);
    release(scratch->trainingK);
    release(scratch->trainingLaunchKTiles);
    release(scratch->featureAxes);
    release(scratch->axisValueCounts);
    release(scratch->axisValueMasks);
    release(scratch->axisPrefixMasks);
    release(scratch->axisValueNumerators);
    release(scratch->axisValueDenominators);
    release(scratch->heldoutAggregateN);
    release(scratch->heldoutK);
    release(scratch->heldoutLaunchKTiles);
    release(scratch->heldoutMeasuredP95);
    release(scratch->heldoutMeasuredMeans);
    release(scratch->heldoutMeasuredMaxima);
    release(scratch->heldoutExactCandidates);
    release(scratch->foldEvaluations);
    release(scratch->frontierA);
    release(scratch->frontierB);
    release(scratch->incumbent);
    release(scratch->expanded);
    release(scratch->expandedOrderKeys);
    release(scratch->splitDescriptors);
    release(scratch->childMasks);
    release(scratch->childLeaves);
    release(scratch->childFittingTailPoints);
    release(scratch->childUniqueIndices);
    release(scratch->maskHashRepresentatives);
    release(scratch->maskHashUniqueIndices);
    release(scratch->frontierCountA);
    release(scratch->frontierCountB);
    release(scratch->expandedCount);
    release(scratch->splitDescriptorCount);
    release(scratch->childMaskCount);
    release(scratch->uniqueCount);
    release(scratch->status);
    release(scratch->expandedCandidateTotal);
    release(scratch->structurallyUniqueCandidateTotal);
    release(scratch->hashSlots);
    release(scratch->uniqueIndices);
    release(scratch->selectionA);
    release(scratch->selectionB);
    release(scratch->selectionCandidates);
    release(scratch->selectionCandidateCounts);
    release(scratch->frontierLcpA);
    release(scratch->frontierLcpB);
    release(scratch->selectionContexts);
    release(scratch->results);
    delete scratch;
    session->treeScratch = nullptr;
    return first;
}

/** Shared host orchestration for publication and fused heldout-evaluation ABIs. */
int runNativeQuantisedTreeSearch(
    LlaminarNativeQuantisedLeafPrimaryScorerSession* session,
    const double* measuredMeanRegrets,
    const double* measuredMaxRegrets,
    const std::uint32_t* pointGroupRanks,
    const std::uint64_t* trainingAggregateN,
    const std::uint64_t* trainingK,
    const std::uint32_t* trainingLaunchKTiles,
    const LlaminarNativeQuantisedTreeFeatureAxis* featureAxes,
    const std::uint64_t* heldoutAggregateN,
    const std::uint64_t* heldoutK,
    const std::uint32_t* heldoutLaunchKTiles,
    const double* heldoutMeasuredP95,
    const double* heldoutMeasuredMeans,
    const double* heldoutMeasuredMaxima,
    std::uint32_t heldoutPointCount,
    std::uint32_t axisCount,
    std::uint32_t boundaryPlacement,
    std::uint32_t parallelismWidth,
    std::uint32_t taskMultiplier,
    std::uint32_t minShapeGroupsPerLeaf,
    std::uint32_t maxLeaves,
    LlaminarNativeQuantisedTreeFitResult* results,
    LlaminarNativeQuantisedTreeFoldEvaluation* evaluations,
    std::uint32_t* exactCandidateIndices,
    bool evaluateHeldout,
    char* error,
    std::size_t errorCapacity) {
    if (session == nullptr || measuredMeanRegrets == nullptr
        || measuredMaxRegrets == nullptr || pointGroupRanks == nullptr
        || trainingAggregateN == nullptr || trainingK == nullptr
        || trainingLaunchKTiles == nullptr
        || featureAxes == nullptr
        || (!evaluateHeldout && results == nullptr)
        || (evaluateHeldout
            && (heldoutAggregateN == nullptr || heldoutK == nullptr
                || heldoutLaunchKTiles == nullptr
                || heldoutMeasuredP95 == nullptr
                || heldoutMeasuredMeans == nullptr
                || heldoutMeasuredMaxima == nullptr || evaluations == nullptr
                || exactCandidateIndices == nullptr))) {
        writeNativeQuantisedScorerError(
            error, errorCapacity, "tree search received a null input");
        return 1;
    }
    if (session->pointCount == 0
        || session->pointCount > kLlaminarNativeQuantisedTreeMaximumPoints
        || maxLeaves == 0
        || maxLeaves > kLlaminarNativeQuantisedTreeMaximumLeaves
        || axisCount == 0 || axisCount > kNativeQuantisedTreeMaximumAxes
        || minShapeGroupsPerLeaf == 0
        || boundaryPlacement > kLlaminarNativeQuantisedTreeBoundaryLowerEdge
        || parallelismWidth == 0 || taskMultiplier == 0
        || (evaluateHeldout
            && (heldoutPointCount == 0
                || heldoutPointCount
                    > kLlaminarNativeQuantisedTreeMaximumHeldoutPoints))) {
        writeNativeQuantisedScorerError(
            error, errorCapacity, "tree search dimensions exceed the v14 ABI");
        return 1;
    }
    bool axisPriorities[kNativeQuantisedTreeMaximumAxes] = {};
    for (std::uint32_t axis = 0; axis < axisCount; ++axis) {
        const auto& descriptor = featureAxes[axis];
        if (descriptor.axis_priority >= kNativeQuantisedTreeMaximumAxes
            || axisPriorities[descriptor.axis_priority]
            || descriptor.operation
                > kLlaminarNativeQuantisedTreeThresholdKPartKTileCount
            || (descriptor.operation
                    <= kLlaminarNativeQuantisedTreeThresholdAspectRatio
                ? descriptor.tile_width != 0
                : descriptor.tile_width == 0)) {
            writeNativeQuantisedScorerError(
                error, errorCapacity, "tree search feature axis is invalid");
            return 1;
        }
        axisPriorities[descriptor.axis_priority] = true;
    }
    for (std::uint32_t point = 0; point < session->pointCount; ++point) {
        if (trainingAggregateN[point] == 0 || trainingK[point] == 0
            || pointGroupRanks[point]
                >= kLlaminarNativeQuantisedTreeMaximumPoints) {
            writeNativeQuantisedScorerError(
                error, errorCapacity,
                "training geometry and group ranks must be valid");
            return 1;
        }
    }
    if (evaluateHeldout) {
        for (std::uint32_t point = 0;
             point < heldoutPointCount; ++point) {
            if (heldoutAggregateN[point] == 0 || heldoutK[point] == 0) {
                writeNativeQuantisedScorerError(
                    error, errorCapacity,
                    "heldout geometry must be strictly positive");
                return 1;
            }
        }
    }

    NativeQuantisedScorerError runtimeResult =
        LLAMINAR_SCORER_SET_DEVICE(session->deviceOrdinal);
    if (runtimeResult != kNativeQuantisedScorerSuccess) {
        return failNativeQuantisedScorer(
            error, errorCapacity, "tree-search device selection", runtimeResult);
    }
    const std::size_t matrixCount =
        static_cast<std::size_t>(session->pointCount) * session->candidateCount;
    const std::size_t axisSlotCount =
        static_cast<std::size_t>(axisCount) * session->pointCount;
    const std::size_t heldoutMatrixCount =
        static_cast<std::size_t>(heldoutPointCount) * session->candidateCount;
    const std::uint64_t maximumCombinationCount = maxLeaves <= 1
        ? 1U
        : static_cast<std::uint64_t>(kNativeQuantisedTreeBeamWidth)
            * (maxLeaves - 1U) * axisCount * session->pointCount;
    if (maximumCombinationCount > std::numeric_limits<std::uint32_t>::max()) {
        writeNativeQuantisedScorerError(
            error, errorCapacity, "tree combination space exceeds uint32");
        return 1;
    }
    std::uint32_t requestedExpansionCapacity = std::max<std::uint32_t>(
        1U, kNativeQuantisedTreeBeamWidth * axisCount * session->pointCount);
    requestedExpansionCapacity = static_cast<std::uint32_t>(std::min(
        static_cast<std::uint64_t>(requestedExpansionCapacity),
        maximumCombinationCount));

    ++session->runtimeStats.tree_search_count;
    if (evaluateHeldout) {
        ++session->runtimeStats.fused_evaluation_count;
    }
    bool metadataUploaded = false;
    while (true) {
        if (!ensureNativeQuantisedTreeScratch(
                session, matrixCount, session->pointCount, axisCount,
                axisSlotCount, heldoutPointCount, evaluateHeldout,
                requestedExpansionCapacity, maxLeaves, error, errorCapacity)) {
            return 2;
        }
        auto* scratch = static_cast<NativeQuantisedTreeScratch*>(session->treeScratch);
        if (!metadataUploaded) {
#define LLAMINAR_UPLOAD_TREE_METADATA(destination, source, count, type, label) \
    do { \
        runtimeResult = copyNativeQuantisedScorerBytesAsync(session, destination, source, \
            static_cast<std::size_t>(count) * sizeof(type), \
            LLAMINAR_SCORER_HOST_TO_DEVICE); \
        if (runtimeResult != kNativeQuantisedScorerSuccess) { \
            return failNativeQuantisedScorer(error, errorCapacity, label, runtimeResult); \
        } \
    } while (false)
            LLAMINAR_UPLOAD_TREE_METADATA(
                scratch->measuredMeans, measuredMeanRegrets, matrixCount,
                double, "measured-mean upload");
            LLAMINAR_UPLOAD_TREE_METADATA(
                scratch->measuredMaxima, measuredMaxRegrets, matrixCount,
                double, "measured-maximum upload");
            LLAMINAR_UPLOAD_TREE_METADATA(
                scratch->pointGroupRanks, pointGroupRanks, session->pointCount,
                std::uint32_t, "point-rank upload");
            LLAMINAR_UPLOAD_TREE_METADATA(
                scratch->trainingAggregateN, trainingAggregateN,
                session->pointCount, std::uint64_t, "training-N upload");
            LLAMINAR_UPLOAD_TREE_METADATA(
                scratch->trainingK, trainingK, session->pointCount,
                std::uint64_t, "training-K upload");
            LLAMINAR_UPLOAD_TREE_METADATA(
                scratch->trainingLaunchKTiles, trainingLaunchKTiles,
                session->pointCount, std::uint32_t,
                "training-K-tiles upload");
            LLAMINAR_UPLOAD_TREE_METADATA(
                scratch->featureAxes, featureAxes, axisCount,
                LlaminarNativeQuantisedTreeFeatureAxis, "feature-axis upload");
            if (evaluateHeldout) {
                LLAMINAR_UPLOAD_TREE_METADATA(
                    scratch->heldoutAggregateN, heldoutAggregateN,
                    heldoutPointCount, std::uint64_t, "heldout-N upload");
                LLAMINAR_UPLOAD_TREE_METADATA(
                    scratch->heldoutK, heldoutK, heldoutPointCount,
                    std::uint64_t, "heldout-K upload");
                LLAMINAR_UPLOAD_TREE_METADATA(
                    scratch->heldoutLaunchKTiles, heldoutLaunchKTiles,
                    heldoutPointCount, std::uint32_t,
                    "heldout-K-tiles upload");
                LLAMINAR_UPLOAD_TREE_METADATA(
                    scratch->heldoutMeasuredP95, heldoutMeasuredP95,
                    heldoutMatrixCount, double, "heldout-p95 upload");
                LLAMINAR_UPLOAD_TREE_METADATA(
                    scratch->heldoutMeasuredMeans, heldoutMeasuredMeans,
                    heldoutMatrixCount, double, "heldout-mean upload");
                LLAMINAR_UPLOAD_TREE_METADATA(
                    scratch->heldoutMeasuredMaxima, heldoutMeasuredMaxima,
                    heldoutMatrixCount, double, "heldout-maximum upload");
            }
#undef LLAMINAR_UPLOAD_TREE_METADATA
            metadataUploaded = true;
        }

        const NativeQuantisedTreeGraphKey key = {
            session->pointCount,
            session->candidateCount,
            axisCount,
            minShapeGroupsPerLeaf,
            maxLeaves,
            requestedExpansionCapacity,
            boundaryPlacement,
            parallelismWidth,
            taskMultiplier,
            heldoutPointCount,
            evaluateHeldout,
        };
        NativeQuantisedTreeGraphEntry* entry = nullptr;
        for (auto& candidate : scratch->graphs) {
            if (candidate.key == key) {
                entry = &candidate;
                break;
            }
        }
        if (entry == nullptr) {
            runtimeResult = beginNativeQuantisedTreeGraphCapture(session->stream);
            if (runtimeResult != kNativeQuantisedScorerSuccess) {
                return failNativeQuantisedScorer(
                    error, errorCapacity, "tree graph capture begin", runtimeResult);
            }
            session->graphCaptureActive = true;
            launchNativeQuantisedTreePipeline(
                session, scratch, axisCount, minShapeGroupsPerLeaf, maxLeaves,
                requestedExpansionCapacity, boundaryPlacement,
                parallelismWidth, taskMultiplier,
                heldoutPointCount, evaluateHeldout);
            NativeQuantisedTreeGraphEntry created;
            created.key = key;
            runtimeResult =
                endNativeQuantisedTreeGraphCapture(session->stream, &created.graph);
            session->graphCaptureActive = false;
            if (runtimeResult != kNativeQuantisedScorerSuccess) {
                return failNativeQuantisedScorer(
                    error, errorCapacity, "tree graph capture end", runtimeResult);
            }
            runtimeResult = instantiateNativeQuantisedTreeGraph(
                &created.executable, created.graph);
            if (runtimeResult != kNativeQuantisedScorerSuccess) {
                (void)destroyNativeQuantisedTreeGraph(created.graph);
                return failNativeQuantisedScorer(
                    error, errorCapacity, "tree graph instantiation", runtimeResult);
            }
            // Corpus geometry is diverse, so an unbounded graph map would turn
            // persistent ownership into a slow memory leak. FIFO is sufficient:
            // dynamic task scheduling revisits nearby domain geometries, while
            // the fixed cap keeps backend graph metadata strictly bounded.
            if (scratch->graphs.size() >= kNativeQuantisedTreeMaximumCachedGraphs) {
                auto& oldest = scratch->graphs.front();
                if (oldest.executable != nullptr) {
                    (void)destroyNativeQuantisedTreeGraphExec(oldest.executable);
                }
                if (oldest.graph != nullptr) {
                    (void)destroyNativeQuantisedTreeGraph(oldest.graph);
                }
                scratch->graphs.erase(scratch->graphs.begin());
            }
            scratch->graphs.push_back(created);
            entry = &scratch->graphs.back();
            ++session->runtimeStats.graph_capture_count;
        } else {
            ++session->runtimeStats.graph_replay_count;
        }
        runtimeResult = launchNativeQuantisedTreeGraph(entry->executable, session->stream);
        if (runtimeResult != kNativeQuantisedScorerSuccess) {
            return failNativeQuantisedScorer(
                error, errorCapacity, "tree graph launch", runtimeResult);
        }

        std::uint32_t hostStatus = 0;
        std::uint64_t hostExpandedCandidateCount = 0;
        std::uint64_t hostStructurallyUniqueCandidateCount = 0;
        std::uint64_t finalResultBytes = 0;
        if (evaluateHeldout) {
            const std::size_t evaluationBytes =
                maxLeaves * sizeof(LlaminarNativeQuantisedTreeFoldEvaluation);
            const std::size_t exactBytes =
                heldoutPointCount * sizeof(std::uint32_t);
            runtimeResult = copyNativeQuantisedScorerBytesAsync(session,
                evaluations, scratch->foldEvaluations, evaluationBytes,
                LLAMINAR_SCORER_DEVICE_TO_HOST);
            if (runtimeResult == kNativeQuantisedScorerSuccess) {
                runtimeResult = copyNativeQuantisedScorerBytesAsync(session,
                    exactCandidateIndices, scratch->heldoutExactCandidates,
                    exactBytes, LLAMINAR_SCORER_DEVICE_TO_HOST);
            }
            finalResultBytes = evaluationBytes + exactBytes;
        } else {
            const std::size_t resultBytes =
                maxLeaves * sizeof(LlaminarNativeQuantisedTreeFitResult);
            runtimeResult = copyNativeQuantisedScorerBytesAsync(session,
                results, scratch->results, resultBytes,
                LLAMINAR_SCORER_DEVICE_TO_HOST);
            finalResultBytes = resultBytes;
        }
        if (runtimeResult == kNativeQuantisedScorerSuccess) {
            runtimeResult = copyNativeQuantisedScorerBytesAsync(session,
                &hostStatus, scratch->status, sizeof(hostStatus),
                LLAMINAR_SCORER_DEVICE_TO_HOST);
        }
        if (runtimeResult == kNativeQuantisedScorerSuccess) {
            runtimeResult = copyNativeQuantisedScorerBytesAsync(session,
                &hostExpandedCandidateCount, scratch->expandedCandidateTotal,
                sizeof(hostExpandedCandidateCount),
                LLAMINAR_SCORER_DEVICE_TO_HOST);
        }
        if (runtimeResult == kNativeQuantisedScorerSuccess) {
            runtimeResult = copyNativeQuantisedScorerBytesAsync(session,
                &hostStructurallyUniqueCandidateCount,
                scratch->structurallyUniqueCandidateTotal,
                sizeof(hostStructurallyUniqueCandidateCount),
                LLAMINAR_SCORER_DEVICE_TO_HOST);
        }
        if (runtimeResult == kNativeQuantisedScorerSuccess) {
            runtimeResult = synchronizeNativeQuantisedScorerStream(session);
            ++session->runtimeStats.tree_search_stream_sync_count;
            session->runtimeStats.final_result_d2h_bytes +=
                finalResultBytes + sizeof(hostStatus)
                + sizeof(hostExpandedCandidateCount)
                + sizeof(hostStructurallyUniqueCandidateCount);
            session->runtimeStats.expanded_candidate_count +=
                hostExpandedCandidateCount;
            session->runtimeStats.structurally_unique_candidate_count +=
                hostStructurallyUniqueCandidateCount;
        }
        if (runtimeResult != kNativeQuantisedScorerSuccess) {
            return failNativeQuantisedScorer(
                error, errorCapacity, "tree result transaction", runtimeResult);
        }
        if ((hostStatus & kNativeQuantisedTreeStatusExpansionOverflow) != 0
            && requestedExpansionCapacity < maximumCombinationCount) {
            requestedExpansionCapacity = static_cast<std::uint32_t>(std::min(
                static_cast<std::uint64_t>(requestedExpansionCapacity) * 2U,
                maximumCombinationCount));
            ++session->runtimeStats.tree_search_retry_count;
            metadataUploaded = true;
            continue;
        }
        if (hostStatus != 0) {
            writeNativeQuantisedScorerError(
                error, errorCapacity,
                (hostStatus & kNativeQuantisedTreeStatusExpansionOverflow) != 0
                    ? "tree expansion exhausted its exact maximum capacity"
                    : (hostStatus & kNativeQuantisedTreeStatusHashOverflow) != 0
                        ? "tree signature hash exhausted its exact probe table"
                        : (hostStatus
                           & kNativeQuantisedTreeStatusMaskHashOverflow) != 0
                            ? "child-mask hash exhausted its exact probe table"
                            : (hostStatus
                               & kNativeQuantisedTreeStatusThresholdOverflow) != 0
                                ? "heldout threshold arithmetic overflowed uint64"
                                : (hostStatus
                                   & kNativeQuantisedTreeStatusEvaluationFailure) != 0
                                    ? "heldout tree evaluation failed on device"
                                    : "tree result export failed on device");
            return 2;
        }
        break;
    }

    if (error != nullptr && errorCapacity > 0) {
        error[0] = '\0';
    }
    return 0;
}

extern "C" int llaminarNativeQuantisedTreeSearch(
    LlaminarNativeQuantisedLeafPrimaryScorerSession* session,
    const double* measuredMeanRegrets,
    const double* measuredMaxRegrets,
    const std::uint32_t* pointGroupRanks,
    const std::uint64_t* trainingAggregateN,
    const std::uint64_t* trainingK,
    const std::uint32_t* trainingLaunchKTiles,
    const LlaminarNativeQuantisedTreeFeatureAxis* featureAxes,
    std::uint32_t axisCount,
    std::uint32_t boundaryPlacement,
    std::uint32_t parallelismWidth,
    std::uint32_t taskMultiplier,
    std::uint32_t minShapeGroupsPerLeaf,
    std::uint32_t maxLeaves,
    LlaminarNativeQuantisedTreeFitResult* results,
    char* error,
    std::size_t errorCapacity) {
    return runNativeQuantisedTreeSearch(
        session, measuredMeanRegrets, measuredMaxRegrets, pointGroupRanks,
        trainingAggregateN, trainingK, trainingLaunchKTiles, featureAxes,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, axisCount,
        boundaryPlacement, parallelismWidth,
        taskMultiplier, minShapeGroupsPerLeaf, maxLeaves, results, nullptr,
        nullptr, false, error, errorCapacity);
}

extern "C" int llaminarNativeQuantisedTreeSearchAndEvaluate(
    LlaminarNativeQuantisedLeafPrimaryScorerSession* session,
    const double* measuredMeanRegrets,
    const double* measuredMaxRegrets,
    const std::uint32_t* pointGroupRanks,
    const std::uint64_t* trainingAggregateN,
    const std::uint64_t* trainingK,
    const std::uint32_t* trainingLaunchKTiles,
    const LlaminarNativeQuantisedTreeFeatureAxis* featureAxes,
    const std::uint64_t* heldoutAggregateN,
    const std::uint64_t* heldoutK,
    const std::uint32_t* heldoutLaunchKTiles,
    const double* heldoutMeasuredP95Regrets,
    const double* heldoutMeasuredMeanRegrets,
    const double* heldoutMeasuredMaxRegrets,
    std::uint32_t heldoutPointCount,
    std::uint32_t axisCount,
    std::uint32_t boundaryPlacement,
    std::uint32_t parallelismWidth,
    std::uint32_t taskMultiplier,
    std::uint32_t minShapeGroupsPerLeaf,
    std::uint32_t maxLeaves,
    LlaminarNativeQuantisedTreeFoldEvaluation* evaluations,
    std::uint32_t* exactCandidateIndices,
    char* error,
    std::size_t errorCapacity) {
    return runNativeQuantisedTreeSearch(
        session, measuredMeanRegrets, measuredMaxRegrets, pointGroupRanks,
        trainingAggregateN, trainingK, trainingLaunchKTiles, featureAxes,
        heldoutAggregateN, heldoutK, heldoutLaunchKTiles,
        heldoutMeasuredP95Regrets, heldoutMeasuredMeanRegrets,
        heldoutMeasuredMaxRegrets, heldoutPointCount, axisCount,
        boundaryPlacement, parallelismWidth, taskMultiplier,
        minShapeGroupsPerLeaf, maxLeaves, nullptr, evaluations,
        exactCandidateIndices, true, error, errorCapacity);
}
