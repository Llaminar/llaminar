/**
 * @file SmartGemmSearch.cpp
 * @brief Implementation of intelligent GEMM kernel search strategies
 *
 * @author David Sanftenberg
 * @date October 2025
 */

#include "SmartGemmSearch.h"
#include "GemmKernelTemplate.h"
#include "GemmMicroKernelAdapter.h" // For IQuantizedGemmVariant
#include "../../utils/CPUFeatures.h"
#include "../../utils/Logger.h"
#include <algorithm>
#include <cstring>
#include <cmath>

namespace llaminar
{
    namespace v2
    {
        namespace kernels
        {

            std::vector<IQuantizedGemmVariant *> SmartGemmSearch::filterByProblemSize(
                const std::vector<std::unique_ptr<IQuantizedGemmVariant>> &variants,
                int m, int n, int k)
            {
                std::vector<IQuantizedGemmVariant *> filtered;
                filtered.reserve(variants.size() / 4); // Expect to keep ~25%

                for (const auto &variant : variants)
                {
                    auto config = variant->config();
                    auto [tile_m, tile_n] = getTileDims(config);

                    // Rule 1: Don't use tiles larger than matrix dimensions
                    // (wasteful - does unnecessary work)
                    if (tile_m > m || tile_n > n)
                    {
                        continue;
                    }

                    // Rule 2: For very small matrices (< 64×64), prefer small tiles
                    if (m < 64 && n < 64)
                    {
                        if (tile_m > 16 || tile_n > 16)
                        {
                            continue;
                        }
                    }

                    // Rule 3: For large matrices (> 512×512), avoid tiny tiles
                    // (too much loop overhead)
                    // NOTE: tile_m/tile_n are micro-kernel register tiles (MR×NR),
                    // not cache tiles. Cache blocking happens in MicroKernelAdapter.
                    // Typical register tiles are 1-8 in each dimension.
                    if (m > 512 && n > 512)
                    {
                        // Only reject truly tiny tiles (1×1 with no unrolling)
                        if (tile_m == 1 && tile_n == 1 && config.unroll_factor <= 1)
                        {
                            continue;
                        }
                    }

                    // Rule 4: Tile should divide problem size reasonably well
                    // (avoid excessive partial tiles)
                    int m_remainder = m % tile_m;
                    int n_remainder = n % tile_n;
                    double efficiency = 1.0 - (static_cast<double>(m_remainder * n) +
                                               static_cast<double>(n_remainder * m)) /
                                                  (static_cast<double>(m * n) * 2.0);

                    if (efficiency < 0.7)
                    { // At least 70% utilization
                        continue;
                    }

                    filtered.push_back(variant.get());
                }

                LOG_DEBUG("Problem-size filtering: " << variants.size() << " -> "
                                                     << filtered.size() << " variants");

                return filtered;
            }

            std::vector<IQuantizedGemmVariant *> SmartGemmSearch::filterByISA(
                const std::vector<IQuantizedGemmVariant *> &variants)
            {
                // Detect CPU capabilities
                bool has_avx512 = llaminar2::cpu_supports_avx512();
                bool has_avx2 = llaminar2::cpu_supports_avx2();

                std::vector<IQuantizedGemmVariant *> filtered;
                filtered.reserve(variants.size());

                // Strategy: Include ALL variants that the CPU can execute
                // Don't artificially prefer AVX512 - let benchmarking decide!
                // (AVX2 often beats AVX512 on small matrices due to frequency scaling)

                for (auto *variant : variants)
                {
                    const char *name = variant->name();
                    bool is_avx512 = isAVX512(name);
                    bool is_avx2 = isAVX2(name);
                    bool is_legacy = !is_avx512 && !is_avx2;

                    // Include variant if CPU can execute it
                    if (is_avx512 && has_avx512)
                    {
                        filtered.push_back(variant);
                    }
                    else if (is_avx2 && has_avx2)
                    {
                        filtered.push_back(variant);
                    }
                    else if (is_legacy)
                    {
                        // Always include scalar/legacy variants (universal fallback)
                        filtered.push_back(variant);
                    }
                }

                const char *isa_label = has_avx512 ? "AVX512+AVX2+Legacy" : (has_avx2 ? "AVX2+Legacy" : "Legacy");
                LOG_DEBUG("ISA filtering (" << isa_label << "): "
                                            << variants.size() << " -> " << filtered.size() << " variants");

                return filtered;
            }

            std::vector<IQuantizedGemmVariant *> SmartGemmSearch::rankByPerformanceModel(
                const std::vector<IQuantizedGemmVariant *> &variants,
                int m, int n, int k)
            {
                // Score each variant
                std::vector<std::pair<double, IQuantizedGemmVariant *>> scored;
                scored.reserve(variants.size());

                for (auto *variant : variants)
                {
                    // Base analytical score (cache efficiency, unrolling, prefetch)
                    double base_score = computeAnalyticalScore(variant->config(), m, n, k);

                    // ISA preference score (AVX2 vs AVX512 based on problem size)
                    double isa_score = scoreISAPreference(variant->name(), m, n, k);

                    // Combined score: 70% base (cache/unroll/prefetch) + 30% ISA preference
                    // Prioritize good tile/cache characteristics over ISA to avoid selecting bad tiles
                    double total_score = 0.70 * base_score + 0.30 * isa_score;

                    scored.push_back({total_score, variant});
                }

                // Sort by score (descending - higher is better)
                std::sort(scored.begin(), scored.end(),
                          [](const auto &a, const auto &b)
                          { return a.first > b.first; });

                // Extract sorted variants
                std::vector<IQuantizedGemmVariant *> ranked;
                ranked.reserve(variants.size());

                for (const auto &[score, variant] : scored)
                {
                    ranked.push_back(variant);
                    LOG_TRACE("  " << variant->name() << " score: " << score);
                }

                LOG_DEBUG("Performance model ranking complete (top score: "
                          << (scored.empty() ? 0.0 : scored[0].first) << ")");

                return ranked;
            }

            std::vector<IQuantizedGemmVariant *> SmartGemmSearch::selectTopCandidates(
                const std::vector<IQuantizedGemmVariant *> &ranked_variants,
                size_t max_candidates)
            {
                size_t count = std::min(ranked_variants.size(), max_candidates);

                std::vector<IQuantizedGemmVariant *> top_n(
                    ranked_variants.begin(),
                    ranked_variants.begin() + count);

                LOG_DEBUG("Selected top " << count << " candidates for benchmarking");

                return top_n;
            }

            IQuantizedGemmVariant *SmartGemmSearch::selectHeuristic(
                const std::vector<std::unique_ptr<IQuantizedGemmVariant>> &variants,
                int m, int n, int k)
            {
                if (variants.empty())
                {
                    return nullptr;
                }

                // Get recommended tile size
                auto [rec_tile_m, rec_tile_n] = getRecommendedTileSize(m, n, k);

                // Find variant closest to recommended tile size with best ISA
                IQuantizedGemmVariant *best = nullptr;
                int best_distance = std::numeric_limits<int>::max();
                bool has_avx512 = llaminar2::cpu_supports_avx512();

                for (const auto &variant : variants)
                {
                    const char *name = variant->name();

                    // Prefer AVX512 if available
                    if (has_avx512 && !isAVX512(name))
                    {
                        continue;
                    }

                    auto config = variant->config();
                    auto [tile_m, tile_n] = getTileDims(config);

                    // Manhattan distance from recommended tile size
                    int distance = std::abs(tile_m - rec_tile_m) + std::abs(tile_n - rec_tile_n);

                    if (distance < best_distance)
                    {
                        best_distance = distance;
                        best = variant.get();
                    }
                }

                if (best)
                {
                    LOG_DEBUG("Heuristic selection: " << best->name()
                                                      << " for shape [" << m << ", " << n << ", " << k << "]");
                }

                return best ? best : variants[0].get(); // Fallback to first variant
            }

            std::pair<int, int> SmartGemmSearch::getRecommendedTileSize(int m, int n, int k)
            {
                // Based on cache-oblivious principles and empirical data

                // For very small matrices: use small tiles
                if (m <= 32 && n <= 32)
                {
                    return {8, 8};
                }

                // For small matrices: moderate tiles
                if (m <= 128 && n <= 128)
                {
                    return {16, 16};
                }

                // For medium matrices: classic 64×32 (V1's sweet spot)
                if (m <= 512 && n <= 512)
                {
                    return {64, 32};
                }

                // For large matrices: larger tiles for better cache reuse
                if (m <= 1024 && n <= 1024)
                {
                    return {128, 64};
                }

                // For very large matrices: maximum tiles
                return {256, 128};
            }

            // ============================================================================
            // Private helper methods
            // ============================================================================

            std::pair<int, int> SmartGemmSearch::getTileDims(const GemmKernelConfig &config)
            {
                // Extract tile dimensions from config
                return {config.tile_m, config.tile_n};
            }

            bool SmartGemmSearch::isAVX512(const char *variant_name)
            {
                return std::strstr(variant_name, "AVX512") != nullptr;
            }

            bool SmartGemmSearch::isAVX2(const char *variant_name)
            {
                return std::strstr(variant_name, "AVX2") != nullptr;
            }

            double SmartGemmSearch::scoreL1Efficiency(int tile_m, int tile_n, int k)
            {
                // L1 cache: 32KB typical = 8192 floats
                // Working set: A tile (tile_m × k_block) + B tile (tile_n × k_block) + C tile (tile_m × tile_n)
                // Assume k_block = 32 (IQ4_NL block size)

                constexpr int k_block = 32;
                constexpr int L1_floats = 8192;

                int working_set = tile_m * k_block + tile_n * k_block + tile_m * tile_n;

                if (working_set <= L1_floats)
                {
                    return 1.0; // Perfect L1 fit
                }
                else if (working_set <= L1_floats * 2)
                {
                    return 0.7; // Some L1 thrashing
                }
                else
                {
                    return 0.3; // Poor L1 utilization
                }
            }

            double SmartGemmSearch::scoreL2Efficiency(int tile_m, int tile_n, int k)
            {
                // L2 cache: 256KB typical = 65536 floats
                // Larger tiles benefit from L2

                constexpr int L2_floats = 65536;
                constexpr int k_block = 32;

                int working_set = tile_m * k_block + tile_n * k_block + tile_m * tile_n;

                if (working_set <= L2_floats / 2)
                {
                    return 1.0; // Good L2 utilization
                }
                else if (working_set <= L2_floats)
                {
                    return 0.8; // Acceptable
                }
                else
                {
                    return 0.4; // L2 cache pressure
                }
            }

            double SmartGemmSearch::scorePrefetch(int prefetch_dist, int m, int n, int k)
            {
                // Prefetch effectiveness depends on problem size
                // Large problems benefit from aggressive prefetching

                size_t problem_size = static_cast<size_t>(m) * n * k;

                if (problem_size > 1000000)
                {
                    // Large problem: prefer higher prefetch distance
                    return prefetch_dist >= 5 ? 1.0 : 0.6;
                }
                else if (problem_size > 100000)
                {
                    // Medium problem: moderate prefetch
                    return prefetch_dist >= 3 ? 0.9 : 0.7;
                }
                else
                {
                    // Small problem: low prefetch to avoid overhead
                    return prefetch_dist <= 3 ? 1.0 : 0.5;
                }
            }

            double SmartGemmSearch::scoreUnroll(int unroll, int k)
            {
                // Unrolling benefits depend on k dimension
                // Large k benefits from higher unroll (amortizes loop overhead)

                if (k >= 1024)
                {
                    // Large k: aggressive unroll good
                    return unroll >= 16 ? 1.0 : (unroll >= 8 ? 0.8 : 0.6);
                }
                else if (k >= 256)
                {
                    // Medium k: moderate unroll
                    return unroll >= 8 ? 1.0 : (unroll >= 4 ? 0.9 : 0.7);
                }
                else
                {
                    // Small k: low unroll to avoid overhead
                    return unroll <= 4 ? 1.0 : 0.7;
                }
            }

            double SmartGemmSearch::scoreISAPreference(const char *variant_name, int m, int n, int k)
            {
                // ISA preference depends on problem size
                // Small matrices: AVX2 often wins due to lower frequency scaling penalty
                // Large matrices: AVX512 wins due to more SIMD parallelism

                size_t problem_size = static_cast<size_t>(m) * n * k;
                bool is_avx512 = isAVX512(variant_name);
                bool is_avx2 = isAVX2(variant_name);

                // Single token (1×896×896 = 802K): Very strong AVX2 preference
                if (m <= 8 || problem_size < 2000000)
                {
                    if (is_avx2)
                        return 1.0;
                    if (is_avx512)
                        return 0.40; // 60% penalty - must be strong to override tile optimization
                    return 0.3;
                }
                // Small batch 32 tokens (32×896×896 = 25M): Moderate AVX2 preference
                else if (problem_size < 50000000)
                {
                    if (is_avx2)
                        return 1.0;
                    if (is_avx512)
                        return 0.75; // 25% penalty
                    return 0.3;
                }
                // Medium batch 128 tokens (128×896×896 = 102M): Slight AVX2 preference
                else if (problem_size < 200000000)
                {
                    if (is_avx2)
                        return 1.0;
                    if (is_avx512)
                        return 0.92; // 8% penalty
                    return 0.3;
                }
                // Large batch 512+ tokens (>200M): Neutral
                else
                {
                    if (is_avx512)
                        return 1.0;
                    if (is_avx2)
                        return 1.0; // Completely neutral for large batches
                    return 0.3;
                }
            }

            double SmartGemmSearch::computeAnalyticalScore(
                const GemmKernelConfig &config,
                int m, int n, int k)
            {
                auto [tile_m, tile_n] = getTileDims(config);

                // Component scores (each 0.0 to 1.0)
                double l1_score = scoreL1Efficiency(tile_m, tile_n, k);
                double l2_score = scoreL2Efficiency(tile_m, tile_n, k);
                double prefetch_score = scorePrefetch(config.prefetch_blocks, m, n, k);
                double unroll_score = scoreUnroll(config.unroll_factor, k);
                // NOTE: We can't call scoreISAPreference here because we don't have variant_name
                // This will be handled in rankByPerformanceModel where we have access to variant

                // Weighted combination
                // L1 cache is most critical (40%), then L2 (30%), then unroll (20%), then prefetch (10%)
                double total_score = 0.40 * l1_score +
                                     0.30 * l2_score +
                                     0.20 * unroll_score +
                                     0.10 * prefetch_score;

                return total_score;
            }

        } // namespace kernels
    } // namespace v2
} // namespace llaminar
