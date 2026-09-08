/**
 * @file NativeVNNIEquivalenceInventory.h
 * @brief Canonical finite witnesses for cross-backend NativeVNNI M totality.
 *
 * NativeVNNI launchers accept every positive row count, while graph-prefill
 * execution rounds an active request up to one of the canonical capture
 * buckets. It is impossible to execute an infinite number of M values in a
 * regression suite, so this inventory turns the totality proof into two finite
 * obligations:
 *
 * 1. Exhaust every ordinary-prefill M through two complete 128-row device
 *    tiles. This covers every row residue and all interior/border transitions
 *    used by the current CUDA and ROCm batch-invariant kernels.
 * 2. Witness both open ends and the exact upper threshold of every larger
 *    canonical graph-bucket interval. This catches automatic-dispatch changes
 *    at larger work sizes without testing every interior M after the kernel's
 *    row decomposition has already been exhausted.
 *
 * CPU, CUDA, and ROCm integration tests consume this exact inventory for every
 * loader-supported quantized format. Host-only planner tests separately prove
 * that positive values above the largest graph bucket remain accepted and
 * overflow-safe. Non-power-of-two K surfaces also exercise odd partition
 * boundaries and short final partitions, so an optimized boundary cursor
 * cannot change the serial FP32 fold at an interior or border tile.
 */

#pragma once

#include "utils/PrefillGraphBucketDefaults.h"

#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace llaminar2::test
{
    /**
     * @brief One exact-M versus graph-bucket active-row equivalence witness.
     */
    struct NativeVNNIPrefillBucketEquivalenceCase
    {
        int active_rows = 0; ///< Semantic request rows whose bytes are compared.
        int bucket_rows = 0; ///< Captured launch rows, including hostile padding.
    };

    /**
     * @brief One all-format N/K surface used to prove physical launch families.
     *
     * `active_rows` deliberately remains a compact boundary set here. The
     * separate residue surface exhausts M; these surfaces vary N and K to force
     * every output-tile family without turning the integration gate into a
     * Cartesian benchmark.
     */
    struct NativeVNNIPrefillGeometryEquivalenceCase
    {
        const char *label = nullptr; ///< Stable diagnostic surface name.
        int output_columns = 0;      ///< GEMM N, including tile-edge witnesses.
        int reduction_columns = 0;   ///< GEMM K, legal for every source format.
        std::vector<NativeVNNIPrefillBucketEquivalenceCase> row_cases;
    };

    /**
     * @brief Resolve explicit active M witnesses to canonical graph buckets.
     */
    inline std::vector<NativeVNNIPrefillBucketEquivalenceCase>
    nativeVNNIPrefillBucketCasesForRows(
        std::initializer_list<int> active_rows)
    {
        std::vector<NativeVNNIPrefillBucketEquivalenceCase> cases;
        cases.reserve(active_rows.size());
        for (const int active : active_rows)
        {
            const auto bucket = std::lower_bound(
                kDefaultPrefillGraphBucketSizes.begin(),
                kDefaultPrefillGraphBucketSizes.end(),
                active);
            if (active <= kDefaultNativeVNNIVerifierRowCapacity ||
                bucket == kDefaultPrefillGraphBucketSizes.end())
            {
                throw std::logic_error(
                    "NativeVNNI geometry witness is outside ordinary-prefill "
                    "graph coverage");
            }
            cases.push_back({active, *bucket});
        }
        return cases;
    }

    /**
     * @brief Build the canonical, ordered, duplicate-free M-totality inventory.
     *
     * Grouped verifier rows own `M=2..16`; ordinary prefill therefore starts at
     * 17. Every M through 256 is exhaustive, including exact bucket thresholds
     * where dispatch geometry may change. Larger intervals contribute the
     * first row, last row, and exact upper bucket so both open-interval behavior
     * and threshold dispatch are exercised.
     */
    inline std::vector<NativeVNNIPrefillBucketEquivalenceCase>
    nativeVNNIPrefillBucketEquivalenceCases()
    {
        constexpr int first_prefill_m =
            kDefaultNativeVNNIVerifierRowCapacity + 1;
        constexpr int exhaustive_m = 256;

        std::vector<NativeVNNIPrefillBucketEquivalenceCase> cases;
        const auto add_active_rows = [&cases](int active_rows)
        {
            const auto bucket = std::lower_bound(
                kDefaultPrefillGraphBucketSizes.begin(),
                kDefaultPrefillGraphBucketSizes.end(),
                active_rows);
            if (bucket == kDefaultPrefillGraphBucketSizes.end())
            {
                throw std::logic_error(
                    "NativeVNNI equivalence witness exceeds the largest "
                    "canonical graph-prefill bucket");
            }
            cases.push_back({active_rows, *bucket});
        };

        for (int m = first_prefill_m; m <= exhaustive_m; ++m)
            add_active_rows(m);

        for (size_t index = 1;
             index < kDefaultPrefillGraphBucketSizes.size();
             ++index)
        {
            const int lower =
                kDefaultPrefillGraphBucketSizes[index - 1];
            const int upper =
                kDefaultPrefillGraphBucketSizes[index];
            if (upper <= exhaustive_m)
                continue;
            add_active_rows(lower + 1);
            add_active_rows(upper - 1);
            add_active_rows(upper);
        }

        std::sort(
            cases.begin(),
            cases.end(),
            [](const auto &lhs, const auto &rhs)
            {
                return lhs.active_rows < rhs.active_rows;
            });
        cases.erase(
            std::unique(
                cases.begin(),
                cases.end(),
                [](const auto &lhs, const auto &rhs)
                {
                    return lhs.active_rows == rhs.active_rows &&
                           lhs.bucket_rows == rhs.bucket_rows;
                }),
            cases.end());
        return cases;
    }

    /**
     * @brief Return all-format N/K witnesses for every CUDA output-tile family.
     *
     * The first surface uses an N border and the complete M inventory. At
     * `K=512` it reaches T64x64, T64x128-w2x2/w4x2, and T128x128-w4x2.
     * The second surface uses a larger, non-aligned N and `K=1024` to cross the
     * T64x128-w2x4 to T128x128-w4x4 boundary. The final large-K surface covers
     * expensive reduction geometry and the BK256 narrow/wide boundary for Q4_0;
     * other formats continue through their production BK64 path. The large-N
     * surface crosses the ROCm streaming thresholds for every format family
     * that advertises that production route and also exercises CUDA's widest
     * output geometry.
     */
    inline std::vector<NativeVNNIPrefillGeometryEquivalenceCase>
    nativeVNNIPrefillAllFormatGeometryEquivalenceCases()
    {
        return {
            {
                "tile_residue_n385_k512",
                385,
                512,
                nativeVNNIPrefillBucketEquivalenceCases(),
            },
            {
                "large_tile_n2049_k1024",
                2049,
                1024,
                nativeVNNIPrefillBucketCasesForRows(
                    {257, 385, 511, 513}),
            },
            {
                "large_k_n513_k2048",
                513,
                2048,
                nativeVNNIPrefillBucketCasesForRows(
                    {17, 31, 33, 63, 65, 1023, 1025}),
            },
            {
                "bk256_wide_n1153_k4096",
                1153,
                4096,
                nativeVNNIPrefillBucketCasesForRows(
                    {17, 31}),
            },
            {
                "streaming_n16385_k512",
                16385,
                512,
                nativeVNNIPrefillBucketCasesForRows(
                    {17, 31, 33}),
            },
            {
                // 56 quantization blocks give odd three-block spans and a
                // short final partition for several canonical GPU schedules.
                "ragged_partition_n2049_k1792",
                2049,
                1792,
                nativeVNNIPrefillBucketCasesForRows({17, 33, 65, 129}),
            },
            {
                // A second N selects different serial partitions, including
                // five-block spans, while retaining the same source formats.
                "ragged_partition_n4097_k1792",
                4097,
                1792,
                nativeVNNIPrefillBucketCasesForRows({17, 33, 65, 129}),
            },
        };
    }
} // namespace llaminar2::test
