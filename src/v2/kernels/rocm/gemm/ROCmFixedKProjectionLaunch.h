/**
 * @file ROCmFixedKProjectionLaunch.h
 * @brief Host launch ABI for ROCm fixed-K explicit-operand prefill projections.
 *
 * Ordinary multi-row prefill for explicit GDN operand modes evaluates the
 * fixed-K partition contract (`kernels/common/FixedKProjectionContract.h`)
 * instead of hipBLAS, whose reduction order follows the row count (#74).
 *
 * - Half-RNE on gfx1100: WMMA f32_16x16x16_f16 units over 64-element blocks
 *   (four units in natural K order per block).
 * - Half-RNE on other targets (gfx906): the shared scalar FMA body over
 *   exactly widened half operands.
 * - Strict FP32 on every target: the shared scalar FMA body.
 *
 * Narrow groups (fixed_k_projection::usesPartitionedSchedule) spread the eight
 * partitions over CTAs, publish chains to the bounded named bank and fold in a
 * second kernel; wide groups walk K per tile. Schedules never change bytes.
 * Every launch requires an exact non-null stream, allocates nothing, performs
 * no transfer or synchronization and is capture safe.
 */
#pragma once

#include "kernels/common/FixedHalfProjectionGroup.h"
#include "kernels/common/FixedKProjectionContract.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace llaminar2::rocm
{
    /** @brief Compiled ROCm fixed-K schedules. */
    enum class FixedKSchedule : std::uint8_t
    {
        WmmaPartitioned64x64, ///< gfx1100 half: 64x64 partition tiles plus fold.
        WmmaTiled64x64,       ///< gfx1100 half: 64x64 tiles, register partition fold.
        WmmaTiled128x64,      ///< gfx1100 half: 128x64 tiles, register partition fold.
        ScalarPartitioned64x64, ///< Scalar FMA: 64x64 partition tiles plus fold.
        ScalarTiled64x64,       ///< Scalar FMA: 64x64 tiles, register partition fold.
    };

    /** @brief Operand arithmetic of a fixed-K launch. */
    enum class FixedKOperand : std::uint8_t { HalfWmma, HalfScalar, StrictFP32 };

    /** @return Whether @p schedule publishes partition chains to workspace. */
    constexpr bool fixedKScheduleNeedsPartials(FixedKSchedule schedule) noexcept
    {
        return schedule == FixedKSchedule::WmmaPartitioned64x64 ||
               schedule == FixedKSchedule::ScalarPartitioned64x64;
    }

    /** @param schedule Compiled schedule. @return Stable diagnostic/PerfStats name. */
    std::string_view fixedKScheduleName(FixedKSchedule schedule) noexcept;

    /** @return Schedules valid for @p operand, for equivalence regressions. */
    std::span<const FixedKSchedule> fixedKSchedules(FixedKOperand operand) noexcept;

    /**
     * @brief Economy policy; every candidate evaluates identical arithmetic.
     * @param operand Operand arithmetic. @param m Physical rows. @param columns Group columns.
     * @return Schedule to submit. Device-free, so admission can bound workspace.
     * @details Scalar (gfx906): partitioning beat walking K per tile for every
     * measured narrow group up to M=1024 (e.g. 48+48 at M=1024: 217 vs 309 us),
     * so every group of at most 128 columns partitions, in bounded row slabs
     * (fixedKPartialRows()); 64x64 tiles beat 128x64 for every wide group and M.
     * gfx1100 WMMA tile choices await native measurements; it keeps the shared
     * partitioned/tiled rule.
     */
    constexpr FixedKSchedule selectFixedKSchedule(FixedKOperand operand, int m, int columns) noexcept
    {
        if (operand == FixedKOperand::HalfWmma)
            return fixed_k_projection::usesPartitionedSchedule(m, columns) ? FixedKSchedule::WmmaPartitioned64x64
                 : m <= 256 ? FixedKSchedule::WmmaTiled64x64 : FixedKSchedule::WmmaTiled128x64;
        return columns <= 128 ? FixedKSchedule::ScalarPartitioned64x64 : FixedKSchedule::ScalarTiled64x64;
    }

    /**
     * @brief Rows of one partitioned launch slab; the chain bank must hold partialBytes(rows, columns).
     * @param schedule Partitioned schedule. @param m Physical rows. @param columns Group columns.
     * @return m for WMMA (its rule already bounds the bank); for scalar groups at
     *         most fixed_k_projection::partitionedSlabRows(columns).
     */
    constexpr int fixedKPartialRows(FixedKSchedule schedule, int m, int columns) noexcept
    {
        if (schedule != FixedKSchedule::ScalarPartitioned64x64)
            return m;
        const int slab = fixed_k_projection::partitionedSlabRows(columns);
        return m < slab ? m : slab;
    }

    /**
     * @brief Half operand arithmetic of a visible device.
     * @param device_id Visible HIP ordinal. @return HalfWmma on gfx1100, HalfScalar otherwise.
     */
    FixedKOperand fixedKHalfOperandForDevice(int device_id);

    /**
     * @brief Submit one half-RNE group with an explicit schedule.
     * @param group Published half bank, persistent half weights and FP32 outputs.
     * @param m Positive physical rows. @param k Positive reduction width.
     * @param device_id Owning HIP ordinal. @param stream Exact non-null stream.
     * @param schedule Schedule valid for the device's half operand arithmetic.
     * @param partials Partition-chain bank for partitioned schedules, otherwise unused.
     * @param partial_bytes Admitted bytes at @p partials.
     * @return True when validation and asynchronous submission succeeded.
     * @throws std::invalid_argument For a null stream, invalid geometry, a WMMA
     *         schedule on a device without gfx1100 WMMA, or a missing bank.
     */
    bool launchFixedKHalfProjection(const FixedHalfProjectionGroup &group, int m, int k, int device_id,
        void *stream, FixedKSchedule schedule, float *partials, std::size_t partial_bytes);

    /** @brief Strict-FP32 counterpart; only scalar schedules are valid. */
    bool launchFixedKFP32Projection(const FixedStrictFP32ProjectionGroup &group, int m, int k, int device_id,
        void *stream, FixedKSchedule schedule, float *partials, std::size_t partial_bytes);
} // namespace llaminar2::rocm
