/**
 * @file CPUNativeVNNITileConfig.h
 * @brief Cache-aware tile configuration for CPU NativeVNNI GEMV/GEMM.
 *
 * Uses the exact prepared-weight footprint plus detected cache capacity and
 * associativity to select tile sizes per shape category. Source GGUF payload
 * bytes are intentionally not part of this API because they do not describe
 * the bytes consumed by the execution kernel.
 *
 * ## Shape Categories
 *
 * | Category   | N range        | K range        | Example (Qwen 3B)     |
 * |------------|----------------|----------------|-----------------------|
 * | Attention  | 128–2048       | 896–3584       | K_proj [256×2048]     |
 * | FFN        | 4864–18944     | 896–18944      | FFN_Gate [11008×2048] |
 * | LM_Head    | 32000–152000   | 896–3584       | LM_Head [151936×3584] |
 *
 * ## Tile Strategy
 *
 * For GEMV (M=1), the weight matrix is streamed once (no reuse). The key
 * parameters are:
 * - **N_chunk**: Fixed at 64 (AVX-512 ZMM width).
 * - **N_block**: Number of sequential N chunks assigned to one task.
 * - **K_tile**: Number of K blocks in each decode partition. For M=1, full K
 *   remains the ordinary path unless parallelism requires a fixed reduction.
 *
 * For GEMM (M>1), B-tile reuse across M rows matters:
 * - **n_block_size**: Number of adjacent 64-column chunks per N task.
 * - **k_tile_blocks**: Largest B/A/output tile that fits an encoding- and
 *   grouped-depth-specific fraction of associativity-safe private L2.
 * - **m_unroll**: 2 for M≥2 (process 2 rows to amortize B loads)
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "CPUNativeVNNIPreparedFootprint.h"
#include "utils/CPUFeatures.h"
#include "utils/DebugEnv.h"

namespace llaminar2::cpu::native_vnni
{

    /**
     * @brief Shape category for tile selection.
     */
    enum class ShapeCategory
    {
        ATTENTION, ///< Small N projections (Q/K/V/Wo)
        FFN,       ///< Medium-large N (Gate/Up/Down)
        LM_HEAD,   ///< Very large N (vocabulary projection)
        GENERIC    ///< Fallback
    };

    /**
     * @brief Tile configuration for NativeVNNI GEMV/GEMM.
     */
    struct NativeVNNITileConfig
    {
        int n_block_chunks; ///< N-chunks per parallel task (each chunk = 64 cols)
        int k_tile_blocks;  ///< K-blocks per tile (0 = full K, no tiling)
        int m_unroll;       ///< M-loop unroll factor (1 for GEMV, 2 for GEMM)
        int omp_min_tasks;  ///< Minimum parallel tasks before falling back to serial
        ShapeCategory category;
        int k_tiles; ///< K-parallel tiles for GEMV (0 = N-parallel only)
    };

    /**
     * @brief Cache geometry consumed by the deterministic tile policy.
     *
     * Production constructs this value from CPUID-backed `CacheInfo`; tests may
     * provide explicit values to prove boundary behavior independently of the
     * machine running CTest.
     */
    struct NativeVNNICacheTopology
    {
        std::uint64_t private_l2_bytes = 0;
        std::uint64_t shared_l3_bytes = 0;
        std::uint32_t private_l2_ways = 0;
        std::uint32_t shared_l3_ways = 0;

        /** @brief Return whether every required cache dimension is valid. */
        [[nodiscard]] constexpr bool isValid() const noexcept
        {
            return private_l2_bytes > 0 && shared_l3_bytes > 0 &&
                   private_l2_ways > 0 && shared_l3_ways > 0;
        }
    };

    /**
     * @brief Exact rational fraction of private L2 assigned to a prefill tile.
     *
     * A rational value keeps policy deterministic across hosts and avoids
     * floating-point rounding in graph/corpus tools. The remaining cache is
     * intentionally available to adjacent activation rows, code, stack data,
     * hardware prefetch, and concurrently scheduled workers.
     */
    struct NativeVNNIL2ResidencyFraction
    {
        std::uint32_t numerator = 0;
        std::uint32_t denominator = 0;

        /** @brief Compare deterministic policy values in tests and diagnostics. */
        [[nodiscard]] friend constexpr bool operator==(
            const NativeVNNIL2ResidencyFraction &,
            const NativeVNNIL2ResidencyFraction &) noexcept = default;

        /** @brief Return whether this is a non-zero fraction no greater than one. */
        [[nodiscard]] constexpr bool isValid() const noexcept
        {
            return numerator > 0 && denominator > 0 &&
                   numerator <= denominator;
        }

        /**
         * @brief Apply the fraction without overflow or floating-point math.
         *
         * @param bytes Associativity-safe private-L2 bytes.
         * @return Bytes assigned to one physical two-row tile.
         */
        [[nodiscard]] constexpr std::uint64_t apply(
            std::uint64_t bytes) const noexcept
        {
            if (!isValid())
                return 0;
            return (bytes / denominator) * numerator +
                   ((bytes % denominator) * numerator) / denominator;
        }
    };

    /**
     * @brief Select the counter-validated private-L2 residency regime.
     *
     * Isolated production-kernel measurements across symmetric/asymmetric
     * nibble and expanded-INT8 formats showed that consuming nearly all of L2
     * increases LLC misses without reducing executed instructions. Reserving
     * three quarters of associativity-safe L2 keeps those decode-heavy streams
     * from evicting their next activation/weight panels.
     *
     * Native Q6_K has a different work/reuse balance. M=2..4 is latency- and
     * conflict-sensitive and wins with a short one-eighth-L2 stream. At M>=8,
     * repeated native-payload reuse amortizes a larger three-quarter-L2 tile.
     * The exact prepared byte footprint still distinguishes symmetric and
     * asymmetric layouts inside each regime.
     *
     * @param footprint Physical prepared encoding and byte footprint.
     * @param M Positive runtime row count.
     * @return Deterministic fraction applied after associativity headroom.
     */
    [[nodiscard]] inline constexpr NativeVNNIL2ResidencyFraction
    prefillL2ResidencyFraction(
        const NativeVNNIPreparedFootprint &footprint,
        int M) noexcept
    {
        if (footprint.encoding ==
            CPUNativeVNNIEncoding::Q6KNativeDualScale)
        {
            return M < 8
                       ? NativeVNNIL2ResidencyFraction{1, 8}
                       : NativeVNNIL2ResidencyFraction{3, 4};
        }
        return NativeVNNIL2ResidencyFraction{1, 4};
    }

    /**
     * @brief Reserve one cache way to avoid conflict-driven capacity thrashing.
     *
     * This is topology-derived headroom: an N-way cache contributes N-1 ways
     * to the planned tile. A one-way or unknown-associativity topology keeps
     * its full reported capacity because subtracting its only way would make a
     * valid platform undispatchable.
     */
    [[nodiscard]] inline constexpr std::uint64_t associativitySafeBytes(
        std::uint64_t capacity,
        std::uint32_t ways) noexcept
    {
        return ways > 1 ? capacity - capacity / ways : capacity;
    }

    /**
     * @brief Compute tile configuration based on shape, cache sizes, and format.
     *
     * @param N           Output dimension (number of weight rows)
     * @param K           Input dimension (weight row length)
     * @param M           Batch size (1 = GEMV, >1 = GEMM)
     * @param footprint Exact prepared B/A/output bytes consumed by the kernel
     * @param num_threads Number of OpenMP threads
     * @param cache Explicit cache topology for deterministic policy evaluation
     * @return Optimal tile configuration
     */
    inline NativeVNNITileConfig computeTileConfig(
        int N,
        int K,
        int M,
        const NativeVNNIPreparedFootprint &footprint,
        int num_threads,
        const NativeVNNICacheTopology &cache)
    {
        if (N <= 0 || K <= 0 || M <= 0 || !footprint.isValid())
        {
            throw std::invalid_argument(
                "CPU NativeVNNI tile geometry requires positive N, K, M, and "
                "prepared-footprint dimensions");
        }
        if (num_threads <= 0)
        {
            throw std::invalid_argument(
                "CPU NativeVNNI tile geometry requires a positive thread count");
        }
        if (!cache.isValid())
        {
            throw std::invalid_argument(
                "CPU NativeVNNI tile geometry requires positive cache capacity "
                "and associativity");
        }

        NativeVNNITileConfig cfg{};

        const int N_chunks = static_cast<int>(
            (static_cast<std::int64_t>(N) + 63) / 64);
        const int blocks_per_row = static_cast<int>(
            (static_cast<std::int64_t>(K) + 31) / 32);

        // Classify shape
        if (N <= 2048)
            cfg.category = ShapeCategory::ATTENTION;
        else if (N <= 20000)
            cfg.category = ShapeCategory::FFN;
        else
            cfg.category = ShapeCategory::LM_HEAD;

        // ---------------------------------------------------------------
        // GEMV (M=1): Weight matrix streamed once, no B reuse
        // ---------------------------------------------------------------
        if (M <= 1)
        {
            cfg.m_unroll = 1;
            cfg.k_tile_blocks = 0; // Full K (no K-tiling for M=1)
            cfg.k_tiles = 0;       // Default: N-parallel only

            // N-blocking is task granularity for streaming GEMV, not a claim
            // that several complete chunks remain resident simultaneously.
            // Keep at least four tasks per worker and use 64-bit arithmetic so
            // every positive thread count and geometry remains total.
            const std::int64_t target_tasks =
                std::max<std::int64_t>(
                    1, static_cast<std::int64_t>(num_threads) * 4);
            const int chunks_per_task = static_cast<int>(
                std::max<std::int64_t>(
                    1,
                    (static_cast<std::int64_t>(N_chunks) + target_tasks - 1) /
                        target_tasks));

            // ---------------------------------------------------------------
            // Compute-bound detection: when the entire weight matrix fits
            // in L3 cache, the kernel is compute-bound (served from L3 at
            // ~500-900 GB/s, not DRAM at ~117 GB/s). Different parallelism
            // tradeoffs apply:
            //  - Large N, small K: per-task compute is tiny → OMP scheduling
            //    overhead dominates → use larger nbc to reduce task count
            //  - Small N, large K: not enough N-parallel work → aggressive
            //    k-parallel creates more useful tasks
            //
            // Sweep-validated across Qwen 0.5B/3B/7B/14B/32B:
            //  - 3B FFN_Gate (11008×2048): nbc=8 → 1.78× vs nbc=1 default
            //  - 3B FFN_Down (2048×11008): kt=7  → 1.85× vs no k-parallel
            //  - 7B/14B/32B FFN (weights > L3): nbc=1 remains optimal
            // ---------------------------------------------------------------
            const std::uint64_t weight_bytes =
                static_cast<std::uint64_t>(N_chunks) *
                static_cast<std::uint64_t>(blocks_per_row) *
                footprint.weight_bytes_per_n_chunk_k_block;
            const std::uint64_t activation_bytes =
                static_cast<std::uint64_t>(blocks_per_row) *
                footprint.activation_bytes_per_row_k_block;
            const std::uint64_t output_bytes =
                static_cast<std::uint64_t>(N_chunks) *
                footprint.output_bytes_per_row_n_chunk;
            const std::uint64_t l3_usable = associativitySafeBytes(
                cache.shared_l3_bytes, cache.shared_l3_ways);
            const bool compute_bound =
                weight_bytes + activation_bytes + output_bytes <= l3_usable;

            // Per-task FLOPs at nbc=1: each 64-row chunk does 2*64*K ops
            long long flops_per_chunk = 2LL * 64 * K;

            // For Attention (small N), use 1 chunk per task for maximum parallelism
            // For FFN/LM_Head (large N), amortize scheduling when needed.
            switch (cfg.category)
            {
            case ShapeCategory::ATTENTION:
                // Small N (128-2048): 2-32 chunks total
                // Use 1 chunk per task to maximize parallelism
                cfg.n_block_chunks = 1;
                break;

            case ShapeCategory::FFN:
                if (compute_bound &&
                    static_cast<std::int64_t>(N_chunks) >
                        static_cast<std::int64_t>(num_threads) * 4 &&
                    flops_per_chunk < 500000)
                {
                    // Compute-bound with many fine-grained tasks: OMP overhead
                    // dominates. Increase nbc so total tasks ≈ 2× threads.
                    // Example: 3B FFN_Gate (172 chunks, K=2048, 262K FLOP/chunk)
                    const std::int64_t target_tasks =
                        static_cast<std::int64_t>(num_threads) * 2;
                    cfg.n_block_chunks = static_cast<int>(
                        std::max<std::int64_t>(
                            1,
                            static_cast<std::int64_t>(N_chunks) /
                                target_tasks));
                }
                else
                {
                    // Memory-bound or sufficient per-task compute: nbc=1
                    // Sweep-validated: nbc=1 beats nbc=2 by 11% for 7B FFN.
                    cfg.n_block_chunks = 1;
                }
                break;

            case ShapeCategory::LM_HEAD:
                // Large N (32000+): 500+ chunks total
                // More chunks per task to amortize OpenMP overhead
                cfg.n_block_chunks = std::max(chunks_per_task, 2);
                break;

            default:
                cfg.n_block_chunks = std::max(chunks_per_task, 1);
                break;
            }

            // ---------------------------------------------------------------
            // K-parallel: split the K dimension across multiple threads per
            // N-chunk. Each thread computes partial sums over a K-block range,
            // then a reduction phase combines them.
            //
            // Two regimes:
            //
            // 1. Memory-bound (weights in DRAM): conservative activation.
            //    Only when N-tasks < threads AND K is very large (bpr >= 256).
            //    Medium-K shapes get no benefit — reduction cost > bandwidth.
            //
            // 2. Compute-bound (weights in L3): aggressive activation.
            //    When N-tasks < 2× threads AND K large enough (bpr >= 128).
            //    K-parallel improves L2 utilization even when N provides
            //    enough tasks. Target 8 tasks/thread for full throughput.
            //    Example: 3B FFN_Down (32 chunks, bpr=344) → kt=7, 1.85×.
            // ---------------------------------------------------------------
            constexpr int MIN_K_BLOCKS_PER_TILE = 4;
            constexpr int DEFAULT_MIN_BPR = 256;
            constexpr int COMPUTE_BOUND_MIN_BPR = 128;
            const int min_bpr = debugEnv().cpu_vnni.min_bpr_k_parallel > 0
                                    ? debugEnv().cpu_vnni.min_bpr_k_parallel
                                    : (compute_bound ? COMPUTE_BOUND_MIN_BPR : DEFAULT_MIN_BPR);
            const std::int64_t total_n_tasks =
                (static_cast<std::int64_t>(N_chunks) +
                 cfg.n_block_chunks - 1) /
                cfg.n_block_chunks;

            // Compute-bound: aggressive k-parallel when N-tasks < 2× threads
            // Memory-bound:  conservative k-parallel when N-tasks < threads
            //
            // GUARD: total FLOPs must be large enough to amortize k-parallel
            // reduction overhead. Small matrices (e.g. 0.5B FFN_Down 896×4864
            // = 8.7M FLOPs, or 14B K_proj 1024×5120 = 10.5M FLOPs) see
            // regressions from k-parallel because reduction cost dominates.
            // 3B FFN_Down (2048×11008 = 45M FLOPs) benefits clearly.
            constexpr long long MIN_FLOPS_FOR_K_PARALLEL = 16'000'000LL; // 16M
            long long total_flops = 2LL * N * K;

            const std::int64_t k_parallel_threshold =
                static_cast<std::int64_t>(num_threads) *
                (compute_bound ? 2 : 1);
            const std::int64_t target_multiplier =
                compute_bound ? 8 : 1; // tasks/thread target

            if (total_n_tasks < k_parallel_threshold && blocks_per_row >= min_bpr
                && total_flops >= MIN_FLOPS_FOR_K_PARALLEL)
            {
                const std::int64_t desired_total =
                    static_cast<std::int64_t>(num_threads) *
                    target_multiplier;
                const std::int64_t desired_k_tiles =
                    std::max<std::int64_t>(
                        1,
                        (desired_total + total_n_tasks - 1) /
                            total_n_tasks);
                const int max_k_tiles =
                    std::max(1, blocks_per_row / MIN_K_BLOCKS_PER_TILE);
                cfg.k_tiles = static_cast<int>(
                    std::min<std::int64_t>(desired_k_tiles, max_k_tiles));

                // Guard: each K-tile must process enough K-blocks
                int k_blocks_per_tile = (blocks_per_row + cfg.k_tiles - 1) / cfg.k_tiles;
                if (k_blocks_per_tile < MIN_K_BLOCKS_PER_TILE)
                    cfg.k_tiles = 0;
                // Don't bother if we'd only get 1 tile
                if (cfg.k_tiles <= 1)
                    cfg.k_tiles = 0;
            }

            // Minimum tasks threshold: below this, overhead > benefit
            cfg.omp_min_tasks = std::max(2, num_threads / 2);

            // Apply LLAMINAR_CPU_VNNI_* overrides (for parameter sweep testing)
            const auto &vnni_gemv = debugEnv().cpu_vnni;
            if (vnni_gemv.n_block_chunks > 0)
                cfg.n_block_chunks = vnni_gemv.n_block_chunks;
            if (vnni_gemv.k_tile_blocks > 0)
                cfg.k_tile_blocks = vnni_gemv.k_tile_blocks;
            if (vnni_gemv.k_tiles > 0)
                cfg.k_tiles = vnni_gemv.k_tiles;

            return cfg;
        }

        // ---------------------------------------------------------------
        // GEMM (M>1): B-tile reuse across M rows matters
        // ---------------------------------------------------------------
        cfg.m_unroll = (M >= 2) ? 2 : 1;
        cfg.k_tiles = 0; // K-parallel only for GEMV

        // The physical prefill microkernel owns two rows. Model the exact live
        // B tile, both Q8_1 activation rows, and both FP32 output chunks. First
        // reserve one associativity way, then apply the counter-validated
        // encoding/depth residency regime. This prevents each OpenMP worker's
        // streaming panel from consuming all private-L2 ways and repeatedly
        // displacing the data needed by its next tile.
        const std::uint64_t associativity_safe_l2 = associativitySafeBytes(
            cache.private_l2_bytes, cache.private_l2_ways);
        const NativeVNNIL2ResidencyFraction residency =
            prefillL2ResidencyFraction(footprint, M);
        const std::uint64_t l2_for_tile =
            residency.apply(associativity_safe_l2);
        constexpr std::uint64_t physical_row_tile = 2;
        const std::uint64_t output_tile_bytes =
            physical_row_tile * footprint.output_bytes_per_row_n_chunk;
        const std::uint64_t bytes_per_k_block =
            footprint.weight_bytes_per_n_chunk_k_block +
            physical_row_tile * footprint.activation_bytes_per_row_k_block;
        const std::uint64_t full_k_tile_bytes =
            output_tile_bytes +
            static_cast<std::uint64_t>(blocks_per_row) * bytes_per_k_block;

        // Check whether one complete-K, one-N-chunk tile fits. If not, choose
        // the largest exact K tile resident in the associativity-safe L2.
        if (full_k_tile_bytes > l2_for_tile)
        {
            const std::uint64_t bytes_available_for_k =
                l2_for_tile > output_tile_bytes
                    ? l2_for_tile - output_tile_bytes
                    : bytes_per_k_block;
            int resident_k_blocks = static_cast<int>(
                std::min<std::uint64_t>(
                    std::max<std::uint64_t>(
                        1, bytes_available_for_k / bytes_per_k_block),
                    static_cast<std::uint64_t>(blocks_per_row)));
            // One 32-element K block is already the complete SIMD/decode unit.
            // Cross-block multiples do not improve alignment and measured odd
            // tile lengths can win, so preserve every block that really fits.
            cfg.k_tile_blocks = std::max(1, resident_k_blocks);
            cfg.n_block_chunks = 1;
        }
        else
        {
            cfg.k_tile_blocks = 0; // Full K
            // Sweep-validated: nbc=1 consistently wins for GEMM prefill
            // across all shapes and batch sizes (M=64 to M=1788).
            // FFN_Gate 1788: +10.7%, Wo_proj 1024: +10.8%.
            // With M>1, there's already massive 2D parallelism
            // (M-tasks × N-tasks), so finer N-granularity gives better
            // load balance without cache penalty.
            cfg.n_block_chunks = 1;
        }

        cfg.omp_min_tasks = std::max(2, num_threads / 2);

        // -------------------------------------------------------------------
        // Apply LLAMINAR_CPU_VNNI_* overrides (for parameter sweep testing)
        // -------------------------------------------------------------------
        const auto &vnni = debugEnv().cpu_vnni;
        if (vnni.n_block_chunks > 0)
            cfg.n_block_chunks = vnni.n_block_chunks;
        if (vnni.k_tile_blocks > 0)
            cfg.k_tile_blocks = vnni.k_tile_blocks;
        if (vnni.m_unroll > 0)
            cfg.m_unroll = vnni.m_unroll;
        if (vnni.k_tiles > 0)
            cfg.k_tiles = vnni.k_tiles;

        return cfg;
    }

    /**
     * @brief Compute production tile geometry from detected cache topology.
     *
     * @param N Output dimension.
     * @param K Input dimension.
     * @param M Runtime row count.
     * @param footprint Exact prepared execution footprint.
     * @param num_threads Positive OpenMP team size.
     * @return Deterministic tile configuration for the detected host.
     */
    inline NativeVNNITileConfig computeTileConfig(
        int N,
        int K,
        int M,
        const NativeVNNIPreparedFootprint &footprint,
        int num_threads)
    {
        const CacheInfo detected;
        return computeTileConfig(
            N,
            K,
            M,
            footprint,
            num_threads,
            NativeVNNICacheTopology{
                .private_l2_bytes = detected.l2_size,
                .shared_l3_bytes = detected.l3_size,
                .private_l2_ways = detected.l2_ways,
                .shared_l3_ways = detected.l3_ways,
            });
    }

} // namespace llaminar2::cpu::native_vnni
