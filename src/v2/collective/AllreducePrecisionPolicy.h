/**
 * @file AllreducePrecisionPolicy.h
 * @brief Batch-invariant transport precision policy for tensor-parallel reductions.
 *
 * Dense and MoE native GPU sums use FP16 transport by default, including one
 * decode row. Explicit precision overrides remain available. A diagnostic
 * minimum element threshold can opt into mixed transport. A
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
#include <string_view>

namespace llaminar2
{
    /// Sole default for native GPU TP sums, independent of model or layer type.
    inline constexpr char kDefaultAllreducePrecision[] = "fp16";

    /// All non-empty rows use the requested FP16 arithmetic by default.
    inline constexpr std::size_t kDefaultAllreduceFP16MinimumElements = 0;

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

    /**
     * @brief Select FP16 transport for an FP32 sum without changing row arithmetic.
     * @param precision Resolved native collective policy, not activation precision.
     * @param effective_count Complete message extent before any rank partition.
     * @param logical_row_elements Width of the original model row.
     * @param minimum_elements Canonical configured FP16 threshold.
     * @return Whether the original FP32 inputs must be rounded before summing.
     *
     * Allreduce and column reduce-scatter consume the same decision. Partitioning
     * a row between receivers must not lower its policy width or select another
     * arithmetic merely because fewer result bytes are returned locally.
     */
    [[nodiscard]] constexpr bool fp32SumUsesFP16Transport(
        std::string_view precision, std::size_t effective_count,
        std::size_t logical_row_elements, std::size_t minimum_elements) noexcept
    {
        return precision == "fp16" &&
            batchInvariantAllreduceDecisionElements(effective_count, logical_row_elements) >= minimum_elements;
    }
} // namespace llaminar2
