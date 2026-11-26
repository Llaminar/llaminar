/**
 * @file CUDANativeQuantisedGemvDispatchHeuristic.h
 * @brief Lightweight host dispatch declarations for CUDA NativeQuantised GEMV shards.
 *
 * Isolates the 76,381-line auto-generated CUDA dispatch table into a single host C++
 * translation unit (`CUDANativeQuantisedGemvDispatchHeuristic.cpp`) so the eight
 * `CUDANativeQuantisedGemvShard*.cu` nvcc compiles never re-parse the generated tables.
 */
#pragma once

#include <cstdint>

#define LLAMINAR_CUDA_GROUPED_DISPATCH_POLICY_V2 1

namespace llaminar2::cuda::native_quantised_dispatch
{

/** @brief Shape-classified CUDA NativeQuantised GEMV kernel family. */
enum class NativeGemvShape
{
    WIDE,
    KPAR,
    DIRECT,
    ROWPAR,
    FUSED_KPAR
};

/** @brief Physical publication may differ while the ordered K tree is identical. */
inline constexpr bool usesOrderedKpartTree(NativeGemvShape shape)
{
    return shape == NativeGemvShape::KPAR || shape == NativeGemvShape::FUSED_KPAR;
}

/** @brief Immutable launch geometry tuning record for M=1 CUDA NativeQuantised GEMV. */
struct GeneratedDispatchTuning
{
    int tile_n = 0;
    int cpt = 0;
    int target_waves = 0;
    int mkg = 0;
    int max_kb = 0;
    int force_two_phase = 0;
    int exact_kb = 0;
};

/** @brief Grouped verifier kernel family selected for M >= 2 CUDA NativeQuantised GEMV. */
enum class GeneratedGroupedKernel : uint8_t
{
    Invalid = 0,
    Dp4aRows = 1,
    TensorCoreMma16 = 2,
};

/** @brief Immutable grouped-row launch tuning record for M >= 2 CUDA NativeQuantised GEMV. */
struct GeneratedGroupedTuning
{
    GeneratedGroupedKernel kernel = GeneratedGroupedKernel::Invalid;
    int grouped_rows = 0;
};

/** @brief Resolve K-block count from target CTA count (generated rule helper). */
int resolveGeneratedTargetBlocksKBlocks(
    int grid_n, int k_groups, int target_blocks,
    int min_kgroups_per_cta, int max_kb);

/** @brief Resolve canonical K-block count preserving ordered partition boundaries. */
int resolveGeneratedCanonicalTargetBlocksKBlocks(
    int grid_n, int k_groups, int target_blocks,
    int min_kgroups_per_cta, int max_kb);

/** @brief Resolve K-block count from blocks-per-partition target. */
int resolveGeneratedBlocksPerPartitionKBlocks(
    int k_groups, int blocks_per_partition, int max_kb);

/**
 * @brief Resolve M=1 CUDA NativeQuantised launch policy for a trained policy codebook.
 * @param policy_cb Trained physical codebook identifier.
 * @param graph_captured Whether dispatch runs inside a captured CUDA graph.
 * @param m Row count (must be 1).
 * @param n Policy-normalized output columns.
 * @param k Reduction width.
 * @param shape Receives selected kernel shape on match.
 * @param tuning Receives selected tuning parameters on match.
 * @return True when an exact or certified generic policy matched.
 */
bool selectGeneratedDispatchByCodebook(
    uint8_t policy_cb, bool graph_captured, int m, int n, int k,
    NativeGemvShape &shape, GeneratedDispatchTuning &tuning);

/**
 * @brief Resolve grouped verifier CUDA NativeQuantised policy for a trained codebook.
 * @param policy_cb Trained physical codebook identifier.
 * @param graph_captured Whether dispatch runs inside a captured CUDA graph.
 * @param m Grouped row count.
 * @param n Policy-normalized output columns.
 * @param k Reduction width.
 * @param tuning Receives selected grouped kernel and row count on match.
 * @return True when an exact or certified generic policy matched.
 */
bool selectGeneratedGroupedTuningByCodebook(
    uint8_t policy_cb, bool graph_captured, int m, int n, int k,
    GeneratedGroupedTuning &tuning);

/** @brief Template-compatible wrapper forwarding to out-of-line M=1 policy lookup. */
template <uint8_t CB>
inline bool selectGeneratedDispatch(
    bool graph_captured, int m, int n, int k,
    NativeGemvShape &shape, GeneratedDispatchTuning &tuning)
{
    return selectGeneratedDispatchByCodebook(CB, graph_captured, m, n, k, shape, tuning);
}

/** @brief Template-compatible wrapper forwarding to out-of-line grouped policy lookup. */
template <uint8_t CB>
inline bool selectGeneratedGroupedTuning(
    bool graph_captured, int m, int n, int k, GeneratedGroupedTuning &tuning)
{
    return selectGeneratedGroupedTuningByCodebook(CB, graph_captured, m, n, k, tuning);
}

} // namespace llaminar2::cuda::native_quantised_dispatch
