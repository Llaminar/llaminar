/**
 * @file FixedKProjectionContract.h
 * @brief Row-, chunk- and schedule-invariant arithmetic for explicit-operand prefill projections.
 *
 * Explicit GDN operand modes (half-RNE and strict FP32) previously used vendor
 * BLAS for ordinary multi-row prefill. Vendor libraries legally select
 * different kernels, tile shapes and split-K reductions as the physical row
 * count changes, so a prompt row produced different bits when a prefix-cache
 * checkpoint moved it into a different prefill chunk. This contract replaces
 * that with one arithmetic program whose result for output (m, n) depends only
 * on input row m, weight row n and the reduction width K:
 *
 *  1. K is divided into reduction blocks: 64 elements for half tensor-core
 *     MMA (four m16n8k16 units), one element for scalar FMA.
 *  2. Blocks are split into exactly kPartitions contiguous partitions. Partition
 *     p owns blocks [partitionBegin(p), partitionBegin(p + 1)). The boundaries
 *     depend on K alone, never on M, N, the row offset or the launch schedule.
 *  3. Each partition starts from +0 and accumulates its blocks in ascending
 *     order into one FP32 accumulator (four MMA units per half block, in unit
 *     order; one IEEE fused multiply-add per scalar block).
 *  4. Partition results are folded left to right:
 *     R = P0; R = R + P1; ... ; R = R + P7 (round-to-nearest FP32 adds).
 *
 * Half MMA slot assignment within a 64-element block starting at k0: for MMA
 * quad lane t (0..3) and unit u (0..3), fragment slots 2t and 2t+1 hold
 * k0 + 16t + 4u + {0, 1} and slots 2t+8 and 2t+9 hold k0 + 16t + 4u + {2, 3}.
 * Every element of the block is used exactly once. This places each lane's
 * operands for a whole block in one contiguous 32-byte span, so direct global
 * schedules use 128-bit loads and staged schedules use 128-bit shared loads.
 *
 * Kernels may assign rows, columns and partitions to threads in any geometry.
 * A split schedule gives each warp one partition and folds through shared
 * memory; a tiled schedule walks K sequentially per output and flushes its
 * partition accumulator at each boundary. Both evaluate exactly the same FP32
 * operations, so the schedule is an economy choice that never changes bytes.
 * Empty partitions (fewer blocks than partitions) contribute an exact +0 add,
 * identically in every schedule. Elements past K are zero-filled operands.
 *
 * Serial decode and MTP verifier rows keep their separate fixed 256-lane tree.
 * This contract governs ordinary multi-row prefill only, so prompt bytes are
 * independent of chunking, bucket size and row position on every backend.
 */
#pragma once

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_FIXED_K_HD __host__ __device__
#else
#define LLAMINAR_FIXED_K_HD
#endif

namespace llaminar2::fixed_k_projection
{
    /** @brief Number of K partitions. Arithmetic identity; never a tuning knob. */
    inline constexpr int kPartitions = 8;

    /** @brief Elements per half-RNE tensor-core reduction block (four m16n8k16 units). */
    inline constexpr int kHalfMMABlockWidth = 64;

    /** @brief K alignment that lets staging use 16-byte vector copies; other K gathers. */
    inline constexpr int kRequiredKAlignment = 8;

    /**
     * @brief First reduction block owned by partition @p p.
     * @param p Partition index in [0, kPartitions]; kPartitions yields @p blocks.
     * @param blocks Total reduction blocks for this K.
     * @return Floor(p * blocks / kPartitions), computed without overflow for int K.
     */
    LLAMINAR_FIXED_K_HD constexpr int partitionBegin(int p, int blocks) noexcept
    {
        return static_cast<int>(static_cast<long long>(p) * blocks / kPartitions);
    }

    /** @brief Chains per partition that one admitted bank covers (see partialBound()). */
    inline constexpr int kPartitionedOutputCap = 65536;

    /**
     * @brief Whether a group is narrow enough to need K partitions spread over CTAs.
     * @param m Physical rows. @param columns Total group columns.
     * @return True for at most 128 columns and at most sixteen 64x64 output
     *         tiles; such groups cannot occupy a GPU by walking K per tile.
     * @details Shared by every backend so the partition-chain bank is one
     * device-free admission quantity (see partialBound()).
     */
    LLAMINAR_FIXED_K_HD constexpr bool usesPartitionedSchedule(int m, int columns) noexcept
    {
        return columns <= 128 && ((m + 63) / 64) * ((columns + 63) / 64) <= 16;
    }

    /**
     * @brief Exact partial-chain bytes of a partitioned launch.
     * @param m Physical rows. @param columns Total group columns.
     * @return kPartitions * m * columns FP32 values.
     */
    LLAMINAR_FIXED_K_HD constexpr unsigned long long partialBytes(int m, int columns) noexcept
    {
        return static_cast<unsigned long long>(kPartitions) * static_cast<unsigned long long>(m) *
               static_cast<unsigned long long>(columns) * 4ull;
    }

    /**
     * @brief Upper bound of partialBytes() over every partitioned group at @p m rows.
     * @param m Physical rows of the retained launch envelope.
     * @return Bytes covering any group the schedules partition at m rows.
     *
     * usesPartitionedSchedule() implies m * columns <= min(128 m, 65536), so one
     * named bank of this size serves every group of a stage without summing
     * member widths; launches still check their exact extent.
     */
    LLAMINAR_FIXED_K_HD constexpr unsigned long long partialBound(int m) noexcept
    {
        const unsigned long long rows = static_cast<unsigned long long>(m > 0 ? m : 0);
        const unsigned long long cap = static_cast<unsigned long long>(kPartitionedOutputCap);
        const unsigned long long outputs = rows * 128ull < cap ? rows * 128ull : cap;
        return static_cast<unsigned long long>(kPartitions) * outputs * 4ull;
    }

    /**
     * @brief Height of one bounded row slab when a narrow group is partitioned beyond the tile rule.
     * @param columns Positive total group columns.
     * @return Largest multiple of 64 with rows * columns <= kPartitionedOutputCap
     *         (at least 512 for the at most 128 columns any policy partitions),
     *         and never below 64 so explicitly forced wide partitioned launches progress.
     *
     * Rows are independent, so a backend may partition a taller narrow group as
     * consecutive slabs of at most this height, each publishing and folding its
     * own chains, without changing bytes. A slab of min(m, slab) rows needs at
     * most min(128 m, kPartitionedOutputCap) chains per partition, so it fits
     * partialBound(m) and the admitted bank is unchanged.
     */
    LLAMINAR_FIXED_K_HD constexpr int partitionedSlabRows(int columns) noexcept
    {
        const int rows = kPartitionedOutputCap / (columns > 0 ? columns : 1) / 64 * 64;
        return rows > 64 ? rows : 64;
    }

    /**
     * @brief Number of half tensor-core reduction blocks for width @p k.
     * @param k Positive reduction width; the final block zero-fills its tail.
     * @return Ceil(k / 64).
     */
    LLAMINAR_FIXED_K_HD constexpr int halfMMABlocks(int k) noexcept
    {
        return (k + kHalfMMABlockWidth - 1) / kHalfMMABlockWidth;
    }
} // namespace llaminar2::fixed_k_projection

#undef LLAMINAR_FIXED_K_HD
