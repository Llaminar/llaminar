/**
 * @file CUDAFixedKProjectionLaunch.h
 * @brief Host launch ABI for CUDA fixed-K explicit-operand prefill projections.
 *
 * Ordinary multi-row prefill for explicit GDN operand modes evaluates the
 * fixed-K partition contract (`kernels/common/FixedKProjectionContract.h`).
 * Schedules are typed economy choices that never change output bytes; the
 * policy chooses one from the live geometry and every schedule is directly
 * launchable so regressions can prove their byte equivalence. All launches
 * require an exact non-null stream, allocate nothing, perform no transfer or
 * synchronization, and are safe inside graph capture.
 */
#pragma once

#include "kernels/common/FixedHalfProjectionGroup.h"
#include "kernels/common/FixedKProjectionContract.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace llaminar2::cuda
{
    /**
     * @brief Compiled half-RNE tensor-core schedules.
     *
     * Partitioned schedules run one CTA per (row tile, compact column tile,
     * K partition), publish the partition chains to named workspace and fold
     * them in a second kernel; they serve narrow groups whose output tiles are
     * too few to occupy the device. Tiled schedules walk K once per output
     * tile and fold partitions in registers; they serve wide groups and need
     * no workspace. Every schedule evaluates identical arithmetic.
     */
    enum class FixedKHalfSchedule : std::uint8_t
    {
        Partitioned64x64,  ///< 64x64 partition tiles, 2x2 warps, three stages, plus fold.
        Tiled32x64W4,      ///< 32x64 CTA, 1x4 warps, four stages.
        Tiled64x32W4,      ///< 64x32 CTA, 4x1 warps, four stages.
        Tiled64x64,        ///< 64x64 CTA, 2x2 warps, three stages.
        Tiled64x128W8,     ///< 64x128 CTA, 2x4 warps, three stages.
        Tiled128x64W8,     ///< 128x64 CTA, 4x2 warps, three stages.
        Tiled128x64,       ///< 128x64 CTA, 2x2 warps, two stages.
        Tiled128x128W8,    ///< 128x128 CTA, 2x4 warps, two stages.
    };

    /** @return Whether @p schedule publishes partition partials to workspace. */
    constexpr bool fixedKHalfScheduleNeedsPartials(FixedKHalfSchedule schedule) noexcept
    {
        return schedule == FixedKHalfSchedule::Partitioned64x64;
    }

    /** @brief Every compiled half schedule, for equivalence regressions and tuning. */
    std::span<const FixedKHalfSchedule> fixedKHalfSchedules() noexcept;

    /** @param schedule Compiled schedule. @return Stable diagnostic/PerfStats name. */
    std::string_view fixedKHalfScheduleName(FixedKHalfSchedule schedule) noexcept;

    /**
     * @brief Measured GB10 (sm_121) economy policy; arithmetic is identical for every result.
     * @param m Physical rows. @param columns Total columns of the group.
     * @param k Reduction width. @return Schedule to submit.
     *
     * Device-free so admission declares exactly the workspace the launch uses.
     * Narrow groups with at most sixteen 64x64 output tiles cannot occupy the
     * device without spreading K partitions over CTAs. Larger narrow groups
     * and wide groups walk K per tile; the tile shape follows rows so each
     * launch stays within whole waves of the resident CTA slots. Evidence:
     * docs/v2/projects/2026-11/FIXED_K_PREFILL_PROJECTION.md.
     */
    constexpr FixedKHalfSchedule selectFixedKHalfSchedule(int m, int columns, int k) noexcept
    {
        (void)k;
        if (columns <= 128)
        {
            const int tiles = ((m + 63) / 64) * ((columns + 63) / 64);
            if (tiles <= 16)
                return FixedKHalfSchedule::Partitioned64x64;
            if (columns <= 96)
                return FixedKHalfSchedule::Tiled64x32W4;
            return m <= 1024 ? FixedKHalfSchedule::Tiled32x64W4 : FixedKHalfSchedule::Tiled64x64;
        }
        if (m <= 64)
            return FixedKHalfSchedule::Tiled32x64W4;
        if (m <= 128)
            return FixedKHalfSchedule::Tiled64x32W4;
        if (m <= 256)
            return FixedKHalfSchedule::Tiled64x64;
        if (m <= 384 && columns <= 3072)
            return FixedKHalfSchedule::Tiled128x64W8;
        return FixedKHalfSchedule::Tiled128x64;
    }

    /**
     * @brief Exact partial-chain bytes of a partitioned launch.
     * @param m Physical rows. @param columns Total group columns.
     * @return kPartitions * m * columns FP32 values.
     */
    constexpr std::size_t fixedKHalfPartialBytes(int m, int columns) noexcept
    {
        return static_cast<std::size_t>(fixed_k_projection::kPartitions) * static_cast<std::size_t>(m) *
               static_cast<std::size_t>(columns) * sizeof(float);
    }

    /**
     * @brief Workspace the policy schedule needs for one group launch.
     * @param m Physical rows. @param columns Total group columns. @param k Reduction width.
     * @return Partial bytes for partitioned schedules, zero for tiled schedules.
     */
    constexpr std::size_t fixedKHalfWorkspaceBytes(int m, int columns, int k) noexcept
    {
        return fixedKHalfScheduleNeedsPartials(selectFixedKHalfSchedule(m, columns, k))
                   ? fixedKHalfPartialBytes(m, columns) : 0;
    }

    /**
     * @brief Upper bound of fixedKHalfWorkspaceBytes over every group at @p m rows.
     * @param m Physical rows of the retained launch envelope.
     * @return Bytes covering any group the policy partitions at m rows.
     *
     * The policy partitions only groups of at most 128 columns whose 64x64
     * tile count is at most sixteen, so m * columns <= min(128 m, 65536).
     * A single named bank of this size serves every group of a stage without
     * summing member widths; the launch still checks its exact extent.
     */
    constexpr std::size_t fixedKHalfPartialBound(int m) noexcept
    {
        const std::size_t rows = static_cast<std::size_t>(m > 0 ? m : 0);
        const std::size_t outputs = rows * 128u < 65536u ? rows * 128u : 65536u;
        return static_cast<std::size_t>(fixed_k_projection::kPartitions) * outputs * sizeof(float);
    }

    /**
     * @brief Submit one half-RNE fixed-K group with an explicit schedule.
     * @param group Published half input, persistent half weights and FP32 outputs.
     * @param m Positive physical rows. @param k Positive reduction width, multiple of eight.
     * @param device_id Owning CUDA ordinal. @param stream Exact non-null stream.
     * @param schedule Compiled schedule; arithmetic does not depend on it.
     * @param partials Workspace for partitioned schedules; may be null otherwise.
     * @param partial_bytes Admitted bytes at @p partials; checked against the exact extent.
     * @return True when validation and asynchronous submission succeeded.
     * @throws std::invalid_argument For a null stream or invalid geometry.
     */
    bool launchFixedKHalfProjection(const FixedHalfProjectionGroup &group, int m, int k,
        int device_id, void *stream, FixedKHalfSchedule schedule, float *partials, std::size_t partial_bytes);

    /** @brief Submit with the measured policy schedule; see the explicit overload. */
    bool launchFixedKHalfProjection(const FixedHalfProjectionGroup &group, int m, int k,
        int device_id, void *stream, float *partials, std::size_t partial_bytes);
} // namespace llaminar2::cuda
