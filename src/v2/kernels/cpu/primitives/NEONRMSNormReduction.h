/**
 * @file NEONRMSNormReduction.h
 * @brief Native original-row RMS reductions with an M/worker-independent arithmetic tree.
 *
 * One/two/three/four physical rows share a traversal, never a replay of complete
 * serial calls. Each row retains four fixed FP64 accumulator chains, an exact
 * fold and bounded register-only tails. No decoded row, allocation, workspace
 * bank or worker-dependent partial exists. The separate native T5 program uses
 * increasing-column FP32 FMA, vectorizing across rows rather than changing a
 * row's arithmetic order. Caller-owned spans are synchronous borrows, not leases
 * or a second memory authority; kernel/factory admission still owns lifetime.
 */
#pragma once
#if !defined(__aarch64__) || !defined(__ARM_NEON)
#error "Native RMS reductions require AArch64 ASIMD"
#endif
#include <arm_neon.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include "../../../tensors/NEONFP16.h"

namespace llaminar2::primitives::neon_rmsnorm
{
/** @brief Original physical element encoding, independent of reduction/publication policy. */
enum class Storage { FP32, FP16, BF16, Int32 };
/** @brief Storage word selected at compile time; FP16/BF16 remain distinct encodings. */
template<Storage Encoding>
using Unit = std::conditional_t<Encoding == Storage::FP32, float,
    std::conditional_t<Encoding == Storage::Int32, std::int32_t, std::uint16_t>>;

/** @brief Checked synchronous original-row borrows; no storage ownership/capacity is invented. */
template<Storage Encoding, std::size_t Count>
class OriginalRows
{
    static_assert(Count >= 1 && Count <= 4);
    static_assert(Encoding == Storage::FP32 || Encoding == Storage::FP16 ||
                  Encoding == Storage::BF16 || Encoding == Storage::Int32);
public:
    /** @brief Bind complete original spans before entering any arithmetic loop.
     * @param rows Original noncontiguous row spans, with truthful owner-issued capacities.
     * @param columns Logical elements consumed from every original row.
     * @return Checked immutable addressing, with zero columns requiring no data pointer.
     * @throws std::invalid_argument For missing or short original storage.
     * @throws std::overflow_error For an extent outside the native pointer domain.
     */
    static OriginalRows bind(std::array<std::span<const Unit<Encoding>>, Count> rows,
                             std::size_t columns)
    {
        if (columns > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(Unit<Encoding>))
            throw std::overflow_error("Native RMS original row exceeds the pointer domain");
        for (const auto row : rows)
            if (row.size() < columns || (columns && !row.data()))
                throw std::invalid_argument("Native RMS original row is missing or shorter than its geometry");
        return OriginalRows(rows, columns);
    }
    /** @return Logical element count; it never depends on M, worker count or a cache estimate. */
    [[nodiscard]] std::size_t columns() const noexcept { return columns_; }
    /** @brief Address one statically selected original row.
     * @tparam Index Bound physical row within this borrow.
     * @return Original span, without copying/decoding its values.
     */
    template<std::size_t Index>
    [[nodiscard]] std::span<const Unit<Encoding>> row() const noexcept
    {
        static_assert(Index < Count);
        return rows_[Index];
    }
private:
    /** @brief Retain only validated addressing for the synchronous arithmetic call.
     * @param rows Checked owner-issued spans.
     * @param columns Checked logical element count.
     */
    OriginalRows(std::array<std::span<const Unit<Encoding>>, Count> rows, std::size_t columns)
        : rows_(rows), columns_(columns) {}
    std::array<std::span<const Unit<Encoding>>, Count> rows_;
    std::size_t columns_;
};

namespace detail
{
/** @brief Expand physical row indices at compile time so accumulator addresses never escape.
 * @tparam Count Number of simultaneously live rows.
 * @tparam Function Compile-time row visitor.
 * @param function Row-local operation, invoked exactly once for each physical row.
 */
template<std::size_t Count, class Function>
[[gnu::always_inline]] inline void visit(Function &&function)
{
    static_assert(Count >= 1 && Count <= 4);
    function.template operator()<0>();
    if constexpr (Count > 1) function.template operator()<1>();
    if constexpr (Count > 2) function.template operator()<2>();
    if constexpr (Count > 3) function.template operator()<3>();
}

/** @brief Load at most four original physical words without tail overread or decoded scratch.
 * @tparam Encoding Exact storage encoding.
 * @param row Checked complete original row.
 * @param offset Original element offset strictly below the logical count.
 * @param columns Logical count, independent of padded allocation capacity.
 * @return Original words with inactive tail lanes zero; conversions occur only in registers.
 */
template<Storage Encoding>
[[gnu::always_inline]] inline auto load(std::span<const Unit<Encoding>> row,
                                       std::size_t offset, std::size_t columns)
{
    const auto remaining = columns - offset;
    if constexpr (Encoding == Storage::FP32)
    {
        if (remaining >= 4) return vld1q_f32(row.data() + offset);
        auto value = vdupq_n_f32(0.0f);
        if (remaining > 0) value = vsetq_lane_f32(row[offset], value, 0);
        if (remaining > 1) value = vsetq_lane_f32(row[offset + 1], value, 1);
        if (remaining > 2) value = vsetq_lane_f32(row[offset + 2], value, 2);
        return value;
    }
    else if constexpr (Encoding == Storage::Int32)
    {
        if (remaining >= 4) return vld1q_s32(row.data() + offset);
        auto value = vdupq_n_s32(0);
        if (remaining > 0) value = vsetq_lane_s32(row[offset], value, 0);
        if (remaining > 1) value = vsetq_lane_s32(row[offset + 1], value, 1);
        if (remaining > 2) value = vsetq_lane_s32(row[offset + 2], value, 2);
        return value;
    }
    else
    {
        auto words = vdup_n_u16(0);
        if (remaining >= 4) words = vld1_u16(row.data() + offset);
        else
        {
            if (remaining > 0) words = vset_lane_u16(row[offset], words, 0);
            if (remaining > 1) words = vset_lane_u16(row[offset + 1], words, 1);
            if (remaining > 2) words = vset_lane_u16(row[offset + 2], words, 2);
        }
        if constexpr (Encoding == Storage::FP16) return neon_fp16::widen(words);
        else return vreinterpretq_f32_u32(vshll_n_u16(words, 16));
    }
}

/** @brief Consume four original values into two fixed FP64 chains.
 * @tparam Encoding Exact original storage encoding.
 * @param row Original physical row, not a converted bank.
 * @param offset Logical element offset of this bounded fragment.
 * @param columns Complete logical element count.
 * @param low Ordered accumulator for lanes0/1.
 * @param high Ordered accumulator for lanes2/3.
 */
template<Storage Encoding>
[[gnu::always_inline]] inline void squareFragment(std::span<const Unit<Encoding>> row,
    std::size_t offset, std::size_t columns, float64x2_t &low, float64x2_t &high)
{
    const auto value = load<Encoding>(row, offset, columns);
    float64x2_t a, b;
    if constexpr (Encoding == Storage::Int32)
    {
        // Widen integers before conversion: int32 -> FP32 would lose low bits.
        a = vcvtq_f64_s64(vmovl_s32(vget_low_s32(value)));
        b = vcvtq_f64_s64(vmovl_s32(vget_high_s32(value)));
    }
    else
    {
        a = vcvt_f64_f32(vget_low_f32(value));
        b = vcvt_f64_f32(vget_high_f32(value));
    }
    low = vfmaq_f64(low, a, a);
    high = vfmaq_f64(high, b, b);
}
} // namespace detail

/** @brief Compute per-row squares with the same four-chain program for every physical M.
 * @tparam Encoding Exact original storage kind.
 * @tparam Count Number of shared-traversal rows1..4.
 * @param input Checked original row borrows; no ownership is extended.
 * @return FP64 row summaries with fixed K traversal/fold, including zero-sized rows.
 * @details No thread partition enters arithmetic. Nonfinite input propagates to
 * the summary; the complete normalization/publication boundary must reject it
 * before stores rather than hiding it with an inverse-scale special case.
 */
template<Storage Encoding, std::size_t Count>
[[nodiscard]] inline std::array<double, Count> squaredSums(const OriginalRows<Encoding, Count> &input)
{
    std::array<std::array<float64x2_t, 4>, Count> accumulators{};
    const auto columns = input.columns();
    std::size_t column = 0;
    for (; columns - column >= 8; column += 8)
        detail::visit<Count>([&]<std::size_t Row> {
            auto &a = accumulators[Row];
            detail::squareFragment<Encoding>(input.template row<Row>(), column, columns, a[0], a[1]);
            detail::squareFragment<Encoding>(input.template row<Row>(), column + 4, columns, a[2], a[3]);
        });
    if (column < columns)
        detail::visit<Count>([&]<std::size_t Row> {
            auto &a = accumulators[Row];
            detail::squareFragment<Encoding>(input.template row<Row>(), column, columns, a[0], a[1]);
            if (columns - column > 4)
                detail::squareFragment<Encoding>(input.template row<Row>(), column + 4, columns, a[2], a[3]);
        });
    std::array<double, Count> result{};
    detail::visit<Count>([&]<std::size_t Row> {
        const auto &a = accumulators[Row];
        const auto folded = vaddq_f64(vaddq_f64(a[0], a[1]), vaddq_f64(a[2], a[3]));
        result[Row] = vgetq_lane_f64(folded, 0) + vgetq_lane_f64(folded, 1);
    });
    return result;
}

/** @brief Native T5 FP32 program: increasing-column FMA, SIMD across physical rows.
 * @tparam Count Number of simultaneously live original rows1..4.
 * @param input Checked FP32 rows; each lane retains its own increasing-K chain.
 * @return FP32-program summaries widened to the existing double summary ABI.
 * @details This is a distinct arithmetic program, not a lower-precision version
 * of squaredSums. No horizontal reduction or worker count changes a row's chain.
 */
template<std::size_t Count>
[[nodiscard]] inline std::array<double, Count> t5SquaredSums(const OriginalRows<Storage::FP32, Count> &input)
{
    auto accumulator = vdupq_n_f32(0.0f);
    for (std::size_t column = 0; column < input.columns(); ++column)
    {
        auto values = vdupq_n_f32(0.0f);
        detail::visit<Count>([&]<std::size_t Row> {
            values = vsetq_lane_f32(input.template row<Row>()[column], values, Row);
        });
        accumulator = vfmaq_f32(accumulator, values, values);
    }
    std::array<double, Count> result{};
    detail::visit<Count>([&]<std::size_t Row> { result[Row] = vgetq_lane_f32(accumulator, Row); });
    return result;
}
} // namespace llaminar2::primitives::neon_rmsnorm
