/**
 * @file FloatingExpertNumericalContract.h
 * @brief Host implementation of the GPU-aligned floating-expert reduction tree.
 *
 * ExpertOverlay may promote or demote an FP16, BF16, or FP32 expert between a
 * CPU tier and a GPU tier while a request is alive. Placement must not select
 * a different numerical program. GPU kernels map the ordered partitions in
 * this contract to threads; the CPU implementation below emulates the same
 * logical lanes and the same binary tree while retaining cache-friendly
 * four-row weight reuse.
 */

#pragma once

#include "DeviceFP32NumericalContract.h"
#include "DeviceSwiGLUNumericalContract.h"
#include "MoEProjectionNumericalContract.h"

#include <array>
#include <cstddef>
#include <stdexcept>

namespace llaminar2::floating_expert_numerical_contract
{
    /** Number of logical K lanes in the cross-tier floating reduction tree. */
    inline constexpr int kReductionLanes =
        MoEProjectionNumericalContract::floating_ordered_k_partitions;

    static_assert(kReductionLanes > 0 &&
                  (kReductionLanes & (kReductionLanes - 1)) == 0,
                  "floating expert reduction lanes must be a power of two");

    /**
     * @brief Reduce one set of logical lane accumulators through the GPU tree.
     * @param partials Mutable lane array populated in strided K order.
     * @return The root binary32 word after the fixed pairwise tree.
     */
    inline float reducePartials(
        std::array<float, kReductionLanes> &partials) noexcept
    {
        for (int stride = kReductionLanes / 2; stride > 0; stride >>= 1)
        {
            for (int lane = 0; lane < stride; ++lane)
            {
                partials[static_cast<std::size_t>(lane)] =
                    device_fp32_contract::add(
                        partials[static_cast<std::size_t>(lane)],
                        partials[static_cast<std::size_t>(lane + stride)]);
            }
        }
        return partials[0];
    }

    /**
     * @brief Compute up to four dot products with the GPU-aligned expert tree.
     *
     * `lhs(row,k)` supplies one activation and `rhs(k)` supplies the shared
     * weight element. K is visited in increasing order, but each term updates
     * logical lane `k mod 256`, exactly matching a 256-thread CUDA/HIP block.
     * A fixed four-row tile lets CPU verifier rows share decoded weights
     * without joining their independent arithmetic trees.
     *
     * @tparam MaximumRows Compile-time row-tile capacity.
     * @tparam LhsFn Callable accepting `(row, k)` and returning FP32.
     * @tparam RhsFn Callable accepting `k` and returning FP32.
     * @param row_count Positive number of rows no greater than MaximumRows.
     * @param reduction_width Positive K dimension.
     * @param lhs Activation accessor.
     * @param rhs Weight accessor.
     * @param outputs Destination for one reduced word per row.
     */
    template <std::size_t MaximumRows, typename LhsFn, typename RhsFn>
    inline void dotRows(
        int row_count,
        int reduction_width,
        LhsFn &&lhs,
        RhsFn &&rhs,
        std::array<float, MaximumRows> &outputs)
    {
        if (row_count <= 0 ||
            row_count > static_cast<int>(MaximumRows) ||
            reduction_width <= 0)
        {
            throw std::invalid_argument(
                "floating expert reduction received invalid row/K geometry");
        }

        std::array<std::array<float, kReductionLanes>, MaximumRows>
            partials{};
        for (int k = 0; k < reduction_width; ++k)
        {
            const std::size_t lane = static_cast<std::size_t>(
                k & (kReductionLanes - 1));
            const float weight = rhs(k);
            for (int row = 0; row < row_count; ++row)
            {
                partials[static_cast<std::size_t>(row)][lane] =
                    device_fp32_contract::fusedMultiplyAdd(
                        lhs(row, k),
                        weight,
                        partials[static_cast<std::size_t>(row)][lane]);
            }
        }

        for (int row = 0; row < row_count; ++row)
        {
            outputs[static_cast<std::size_t>(row)] = reducePartials(
                partials[static_cast<std::size_t>(row)]);
        }
    }

    /**
     * @brief Evaluate one CPU SwiGLU word through the shared CPU/GPU program.
     * @param gate Gate projection value.
     * @param up Up projection value.
     * @return Placement-invariant binary32 SwiGLU word.
     */
    inline float swigluValue(float gate, float up) noexcept
    {
        return device_swiglu_contract::swigluValue(gate, up);
    }
} // namespace llaminar2::floating_expert_numerical_contract
