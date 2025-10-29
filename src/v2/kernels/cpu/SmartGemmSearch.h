/**
 * @file SmartGemmSearch.h
 * @brief Intelligent search strategies for GEMM kernel selection
 *
 * Implements hierarchical filtering and heuristic-based selection inspired by
 * Intel MKL, OpenBLAS, and ATLAS to efficiently search through 972 kernel variants.
 *
 * Key strategies:
 * 1. Problem-size filtering: Eliminate tiles larger than matrix dimensions
 * 2. ISA filtering: Select best ISA first (AVX512 > AVX2 > Scalar)
 * 3. Analytical models: Predict good candidates based on cache behavior
 * 4. Top-N selection: Only benchmark most promising 5-10 candidates
 * 5. Fallback heuristics: Fast defaults when auto-tuning disabled
 *
 * @author David Sanftenberg
 * @date October 2025
 */

#pragma once

#include "../../tensors/TensorKernels.h"
#include <vector>
#include <memory>

namespace llaminar
{
    namespace v2
    {
        namespace kernels
        {

            // Forward declarations
            struct GemmKernelConfig;
            class IQuantizedGemmVariant;

            /**
             * @brief Smart search strategies for GEMM kernel selection
             */
            class SmartGemmSearch
            {
            public:
                /**
                 * @brief Filter variants to only those suitable for given problem size
                 *
                 * Eliminates variants where:
                 * - TILE_M > m or TILE_N > n (wasteful for small matrices)
                 * - TILE_M or TILE_N is excessively small for large matrices
                 *
                 * @param variants All available kernel variants
                 * @param m Number of rows in output matrix
                 * @param n Number of columns in output matrix
                 * @param k Inner dimension
                 * @return Filtered list of potentially optimal variants
                 */
                static std::vector<IQuantizedGemmVariant *> filterByProblemSize(
                    const std::vector<std::unique_ptr<IQuantizedGemmVariant>> &variants,
                    int m, int n, int k);

                /**
                 * @brief Select best ISA variants based on CPU capabilities
                 *
                 * Priority: AVX512 > AVX2 > Scalar
                 *
                 * @param variants Filtered variants from problem-size filtering
                 * @return Variants using best available ISA
                 */
                static std::vector<IQuantizedGemmVariant *> filterByISA(
                    const std::vector<IQuantizedGemmVariant *> &variants);

                /**
                 * @brief Score variants using analytical performance model
                 *
                 * Model considers:
                 * - Cache efficiency (L1: 32KB, L2: 256KB, L3: 16MB typical)
                 * - Register blocking efficiency
                 * - TLB pressure
                 * - Prefetch effectiveness
                 *
                 * @param variants ISA-filtered variants
                 * @param m Number of rows in output matrix
                 * @param n Number of columns in output matrix
                 * @param k Inner dimension
                 * @return Variants sorted by predicted performance (best first)
                 */
                static std::vector<IQuantizedGemmVariant *> rankByPerformanceModel(
                    const std::vector<IQuantizedGemmVariant *> &variants,
                    int m, int n, int k);

                /**
                 * @brief Select top N candidates for benchmarking
                 *
                 * @param ranked_variants Variants sorted by analytical model
                 * @param max_candidates Maximum number to benchmark (default: 10)
                 * @return Top N most promising variants
                 */
                static std::vector<IQuantizedGemmVariant *> selectTopCandidates(
                    const std::vector<IQuantizedGemmVariant *> &ranked_variants,
                    size_t max_candidates = 10);

                /**
                 * @brief Fast heuristic-based selection (no benchmarking)
                 *
                 * Used when auto-tuning is disabled or for very small matrices.
                 * Returns single "good enough" variant based on problem size.
                 *
                 * @param variants All available variants
                 * @param m Number of rows in output matrix
                 * @param n Number of columns in output matrix
                 * @param k Inner dimension
                 * @return Single heuristically-selected variant
                 */
                static IQuantizedGemmVariant *selectHeuristic(
                    const std::vector<std::unique_ptr<IQuantizedGemmVariant>> &variants,
                    int m, int n, int k);

                /**
                 * @brief Get recommended tile size for given problem dimensions
                 *
                 * Based on cache-oblivious principles and empirical data.
                 *
                 * @param m Number of rows in output matrix
                 * @param n Number of columns in output matrix
                 * @param k Inner dimension
                 * @return Pair of (recommended_tile_m, recommended_tile_n)
                 */
                static std::pair<int, int> getRecommendedTileSize(int m, int n, int k);

            private:
                /**
                 * @brief Extract tile dimensions from variant config
                 */
                static std::pair<int, int> getTileDims(const GemmKernelConfig &config);

                /**
                 * @brief Check if variant uses AVX512
                 */
                static bool isAVX512(const char *variant_name);

                /**
                 * @brief Check if variant uses AVX2
                 */
                static bool isAVX2(const char *variant_name);

                /**
                 * @brief Estimate L1 cache efficiency for given tile size
                 *
                 * L1 cache: 32KB typical, holds ~8K floats
                 * Ideal: TILE_M * TILE_N + TILE_M * k_block + TILE_N * k_block < 8192
                 */
                static double scoreL1Efficiency(int tile_m, int tile_n, int k);

                /**
                 * @brief Estimate L2 cache efficiency
                 *
                 * L2 cache: 256KB typical, holds ~64K floats
                 */
                static double scoreL2Efficiency(int tile_m, int tile_n, int k);

                /**
                 * @brief Score prefetch effectiveness
                 *
                 * Higher prefetch distances better for large matrices, wasteful for small
                 */
                static double scorePrefetch(int prefetch_dist, int m, int n, int k);

                /**
                 * @brief Score unroll factor effectiveness
                 *
                 * Higher unroll better for large k, overhead for small k
                 */
                static double scoreUnroll(int unroll, int k);

                /**
                 * @brief Score ISA preference based on problem size
                 *
                 * Small matrices: AVX2 often faster (less frequency scaling penalty)
                 * Large matrices: AVX512 faster (more SIMD parallelism)
                 *
                 * @param variant_name Name of variant (to check ISA)
                 * @param m Number of rows
                 * @param n Number of columns
                 * @param k Inner dimension
                 * @return Score 0.0 to 1.0 (higher = better ISA choice for this problem)
                 */
                static double scoreISAPreference(const char *variant_name, int m, int n, int k);

                /**
                 * @brief Combined analytical score (0.0 to 1.0, higher is better)
                 */
                static double computeAnalyticalScore(
                    const GemmKernelConfig &config,
                    int m, int n, int k);
            };

        } // namespace kernels
    } // namespace v2
} // namespace llaminar
