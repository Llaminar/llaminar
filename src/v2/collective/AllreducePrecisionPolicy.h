/**
 * @file AllreducePrecisionPolicy.h
 * @brief Batch-invariant transport precision policy for tensor-parallel reductions.
 *
 * Mixed-precision allreduce is selected from a minimum element threshold.  A
 * threshold applied to the aggregate matrix size is unsafe for grouped decode:
 * one serial row can select FP32 while the same row inside an M-row verifier
 * matrix selects FP16.  That changes the values published to the next layer and
 * violates the requirement that grouped verification be byte-identical to
 * serial decode.
 *
 * This policy therefore evaluates the threshold against one logical row when
 * the reduced span is an integral row matrix.  Prefill and grouped decode may
 * still move many rows in one economical collective, but adding rows cannot
 * change the arithmetic used for any existing row.
 */

#pragma once

#include <cstddef>

namespace llaminar2
{
    /**
     * @brief Return the element count used to choose allreduce transport precision.
     *
     * @param effective_count Number of elements reduced by this collective.
     * @param logical_row_elements Number of elements in one independently
     *        decoded row, normally `TensorBase::cols()`.
     * @return `logical_row_elements` for an integral multi-row matrix;
     *         otherwise `effective_count` for vectors and partial spans.
     *
     * The divisibility guard is important.  A caller reducing a partial tensor
     * must retain its explicit count semantics rather than accidentally treating
     * an unrelated physical stride as the logical operation width.
     */
    [[nodiscard]] constexpr std::size_t batchInvariantAllreduceDecisionElements(
        std::size_t effective_count,
        std::size_t logical_row_elements) noexcept
    {
        if (logical_row_elements == 0 ||
            effective_count <= logical_row_elements ||
            effective_count % logical_row_elements != 0)
        {
            return effective_count;
        }
        return logical_row_elements;
    }
} // namespace llaminar2
