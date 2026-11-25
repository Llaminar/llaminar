/**
 * @file FixedHalfProjectionGroup.h
 * @brief Validated by-value bindings for fixed-order ragged projection groups.
 *
 * Captured kernel arguments retain addresses and immutable column boundaries;
 * they never mirror activation values or the device-owned live-row count.
 * One group shares a published input matrix and owns disjoint FP32 outputs.
 * Ragged widths require neither padded work nor device pointer tables.
 *
 * The operand type names the physical storage of the shared input and of every
 * weight: `std::uint16_t` words for half-RNE operands and `float` for strict
 * FP32 operands. Both instantiations have the same validation and capture ABI,
 * so fixed-order short-row and fixed-K prefill kernels bind identical metadata.
 */
#pragma once

#include "kernels/common/FloatingPointGemmWorkspaceABI.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_HALF_GROUP_HD __host__ __device__
#else
#define LLAMINAR_HALF_GROUP_HD
#endif

namespace llaminar2
{
/** Measured decode/verifier domain; larger rows retain throughput schedules. */
inline constexpr int kShortHalfProjectionMaxRows = 31;

/**
 * @brief Nonempty immutable-address group copied into a retained native launch.
 * @tparam Operand Physical input/weight element: half-RNE words or FP32 values.
 *
 * Host construction validates the complete group before any producer is
 * submitted. Matrix storage remains owned by its existing arena/weight handles.
 * Those owners and every embedded address participate in graph binding identity.
 */
template<class Operand>
class FixedRaggedProjectionGroup final
{
    static_assert(std::is_same_v<Operand, std::uint16_t> || std::is_same_v<Operand, float>,
        "Ragged projection groups bind half words or strict FP32 operands");
public:
    /**
     * @brief Bind one projection without reading any device memory.
     * @param input Published operand matrix with the launch's [M,K] shape.
     * @param weight Persistent operand matrix [columns,K].
     * @param output Exclusive persistent FP32 matrix [M,columns].
     * @param columns Positive output width.
     * @throws std::invalid_argument For null owners or nonpositive width.
     */
    [[nodiscard]] static FixedRaggedProjectionGroup single(
        const Operand *input, const Operand *weight,
        float *output, int columns)
    {
        if (!input || !weight || !output || columns <= 0)
            throw std::invalid_argument("Ragged projection group requires nonnull owners and positive columns");
        return FixedRaggedProjectionGroup(input, weight, output, columns);
    }

    /**
     * @brief Extend valid metadata while retaining a single input authority.
     * @param weight Persistent operand matrix [columns,K].
     * @param output Exclusive persistent FP32 matrix [M,columns].
     * @param columns Positive output width; no capacity padding is added.
     * @throws std::invalid_argument For null owners, width or duplicate output.
     * @throws std::length_error If the bounded launch is already full.
     * @throws std::overflow_error If the compact column grid cannot fit an int.
     */
    void append(const Operand *weight, float *output, int columns)
    {
        if (!weight || !output || columns <= 0)
            throw std::invalid_argument("Ragged projection append requires nonnull owners and positive columns");
        if (count_ == int(floating_gemm_abi::kMaxBatchedProjections))
            throw std::length_error("Ragged projection group exceeds its capture ABI");
        if (columns > std::numeric_limits<int>::max() - totalColumns())
            throw std::overflow_error("Ragged projection compact column extent overflows");
        for (int index = 0; index < count_; ++index)
            if (output == outputs_[index])
                throw std::invalid_argument("Ragged projections require distinct output owners");
        weights_[count_] = weight;
        outputs_[count_] = output;
        ends_[count_] = totalColumns() + columns;
        ++count_;
    }

    /**
     * @brief The same projections restricted to rows starting at @p row.
     * @param row First physical row of the slab. @param k Input row width.
     * @return Group whose input and outputs begin at that row; rows are
     *         independent, so a slab evaluates exactly the original bytes.
     */
    [[nodiscard]] FixedRaggedProjectionGroup rowsFrom(int row, int k) const noexcept
    {
        FixedRaggedProjectionGroup slab = *this;
        slab.input_ += static_cast<std::size_t>(row) * static_cast<std::size_t>(k);
        for (int index = 0; index < count_; ++index)
            slab.outputs_[index] += static_cast<std::size_t>(row) * static_cast<std::size_t>(columns(index));
        return slab;
    }

    /** @return Borrowed shared input matrix, never a downloaded host value. */
    [[nodiscard]] LLAMINAR_HALF_GROUP_HD const Operand *input() const noexcept { return input_; }
    /** @return Number of valid projections, always in [1,8]. */
    [[nodiscard]] LLAMINAR_HALF_GROUP_HD int size() const noexcept { return count_; }
    /** @return Exact compact grid width across every projection. */
    [[nodiscard]] LLAMINAR_HALF_GROUP_HD int totalColumns() const noexcept { return ends_[count_ - 1]; }
    /** @param index Valid projection index. @return Exclusive compact column end. */
    [[nodiscard]] LLAMINAR_HALF_GROUP_HD int columnEnd(int index) const noexcept { return ends_[index]; }
    /** @param index Valid projection index. @return That projection's physical column stride. */
    [[nodiscard]] LLAMINAR_HALF_GROUP_HD int columns(int index) const noexcept
    { return ends_[index] - (index ? ends_[index - 1] : 0); }
    /** @param index Valid projection index. @return Borrowed weight owner. */
    [[nodiscard]] LLAMINAR_HALF_GROUP_HD const Operand *weight(int index) const noexcept { return weights_[index]; }
    /** @param index Valid projection index. @return Exclusive FP32 output owner. */
    [[nodiscard]] LLAMINAR_HALF_GROUP_HD float *output(int index) const noexcept { return outputs_[index]; }

private:
    /** @brief Only the validated nonempty factory can create an initial group. */
    FixedRaggedProjectionGroup(const Operand *input, const Operand *weight,
                               float *output, int columns) noexcept
        : input_(input), weights_{weight}, outputs_{output}, ends_{columns}, count_(1) {}
    const Operand *input_; ///< Borrowed shared activation authority.
    const Operand *weights_[floating_gemm_abi::kMaxBatchedProjections]; ///< Persistent weights.
    float *outputs_[floating_gemm_abi::kMaxBatchedProjections]; ///< Exclusive result owners.
    int ends_[floating_gemm_abi::kMaxBatchedProjections]; ///< Compact cumulative columns.
    int count_; ///< Validated nonempty metadata extent.
};

/** @brief Physical half-RNE words for input and every weight. */
using FixedHalfProjectionGroup = FixedRaggedProjectionGroup<std::uint16_t>;
/** @brief Strict FP32 input and weights with FP32 outputs. */
using FixedStrictFP32ProjectionGroup = FixedRaggedProjectionGroup<float>;
static_assert(std::is_trivially_copyable_v<FixedHalfProjectionGroup>);
static_assert(std::is_standard_layout_v<FixedHalfProjectionGroup>);
static_assert(std::is_trivially_copyable_v<FixedStrictFP32ProjectionGroup>);
static_assert(std::is_standard_layout_v<FixedStrictFP32ProjectionGroup>);
} // namespace llaminar2
#undef LLAMINAR_HALF_GROUP_HD
