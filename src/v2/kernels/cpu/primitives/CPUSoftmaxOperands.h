/**
 * @file CPUSoftmaxOperands.h
 * @brief Checked original CPU softmax row/batch borrows and causal coordinates.
 *
 * These allocation-free views validate declared geometry, address arithmetic and
 * immutable invocation policy before writes or ISA selection. They neither prove
 * an allocation's actual capacity nor extend its arena/cache lease. The caller
 * retains the original storage authority throughout the synchronous invocation.
 */
#pragma once
#include "../../../tensors/BlockStructures.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace llaminar2::primitives::softmax_detail
{
/** @brief Original physical storage, not a selectable substitute precision. */
enum class Storage : std::uint8_t { FP32, BFloat16, Half, Q8_1 };
/** @brief Map the declared tensor storage to its original physical element. */
template<Storage Format>
using Unit = std::conditional_t<Format == Storage::FP32, float,
    std::conditional_t<Format == Storage::Q8_1, Q8_1Block, std::uint16_t>>;
/** @brief Scalar coordinates per original physical storage unit. */
template<Storage Format>
inline constexpr std::size_t elements_per_unit = Format == Storage::Q8_1 ? 32 : 1;

/** @brief Require a representable declared original bank without inventing capacity.
 * @param bank Original synchronous storage borrow.
 * @param units Original physical element count.
 * @param unit_bytes Exact storage-unit width.
 * @throws std::invalid_argument For a missing nonempty bank or wrapping address envelope.
 */
inline void requireBank(const void *bank, std::size_t units, std::size_t unit_bytes)
{
    if (units > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) / unit_bytes)
        throw std::invalid_argument("CPU softmax declared byte extent is not representable");
    const auto bytes = units * unit_bytes;
    if (bytes && (!bank || reinterpret_cast<std::uintptr_t>(bank) > std::numeric_limits<std::uintptr_t>::max() - bytes))
        throw std::invalid_argument("CPU softmax requires a complete nonwrapping original bank");
}

/** @brief Original coordinate policy, independently testable without a storage-capacity fiction. */
class CausalPrefix final
{
public:
    /** @brief Derive a bounded original prefix using widened signed causal arithmetic.
     * @param columns Complete original scalar count; @param causal Whether to apply a prefix mask.
     * @param row_index Original causal coordinate, including negative/all-masked and INT_MAX boundaries.
     * @return An immutable coordinate policy; no storage is borrowed or dereferenced.
     */
    static CausalPrefix bind(std::size_t columns,bool causal,int row_index) noexcept
    {
        const auto prefix=std::size_t(std::max<std::int64_t>(0,std::int64_t(row_index)+1));
        return CausalPrefix(causal?std::min(columns,prefix):columns);
    }
    /** @return Original active scalar count, never narrowed to an int worker coordinate. */
    std::size_t columns()const noexcept{return columns_;}
    /** @brief Test one original coordinate without narrowing at the INT_MAX boundary.
     * @param column Original nonnegative global scalar coordinate.
     * @return Whether that original coordinate belongs to this active prefix.
     */
    bool contains(std::size_t column)const noexcept{return column<columns_;}
private:
    /** @brief Retain only an already-derived original prefix.
     * @param columns Original active scalar count.
     */
    explicit CausalPrefix(std::size_t columns):columns_(columns){}
    std::size_t columns_;
};

/** @brief Complete original in-place row with a checked original causal prefix.
 * @tparam Format Physical source and result storage.
 */
template<Storage Format>
class Row final
{
public:
    /** @brief Bind original geometry and policy before any scalar/ISA body can write.
     * @param bank Original complete writable row; empty rows may use null.
     * @param units Original physical elements, or complete Q8_1 blocks.
     * @param causal Whether only original columns <= row_index are active.
     * @param scale Finite score multiplier, including zero or negative values.
     * @param row_index Original causal coordinate; negative coordinates mask the whole row.
     * @return One synchronous, nonowning validated invocation.
     * @throws std::invalid_argument For malformed extent, missing storage or nonfinite policy.
     */
    static Row bind(Unit<Format> *bank, int units, bool causal, float scale, int row_index)
    {
        if (units < 0 || !std::isfinite(scale))
            throw std::invalid_argument("CPU softmax requires nonnegative geometry and a finite scale");
        requireBank(bank, std::size_t(units), sizeof(Unit<Format>));
        const auto columns = std::size_t(units) * elements_per_unit<Format>;
        return Row(bank, units, causal, scale, row_index, CausalPrefix::bind(columns,causal,row_index));
    }
    /** @return Original mutable storage, whose owner remains external. */
    Unit<Format> *data() const noexcept { return bank_; }
    /** @return Original physical element/block count. */
    int units() const noexcept { return units_; }
    /** @return Complete original scalar coordinate count. */
    std::size_t columns() const noexcept { return std::size_t(units_) * elements_per_unit<Format>; }
    /** @return Active original prefix, independent of grouping or worker assignment. */
    std::size_t liveColumns() const noexcept { return live_.columns(); }
    /** @brief Test an original global coordinate without an int narrowing edge.
     * @param column Original nonnegative scalar coordinate.
     * @return Whether that original coordinate is active under this invocation's prefix.
     */
    bool containsColumn(std::size_t column)const noexcept{return live_.contains(column);}
    /** @return Original finite score multiplier. */
    float scale() const noexcept { return scale_; }
    /** @return Original masking policy for existing diagnostic/ISA adapters. */
    bool causal() const noexcept { return causal_; }
    /** @return Original causal coordinate, never a worker-local ordinal. */
    int rowIndex() const noexcept { return row_index_; }
private:
    /** @brief Retain only already-checked original borrows and invocation parameters.
     * @param bank Original bank; @param units Physical count; @param causal Mask policy.
     * @param scale Finite multiplier; @param row_index Original coordinate; @param live Active prefix.
     */
    Row(Unit<Format> *bank, int units, bool causal, float scale, int row_index, CausalPrefix live)
        : bank_(bank), units_(units), causal_(causal), scale_(scale), row_index_(row_index), live_(live) {}
    Unit<Format> *bank_;
    int units_;
    bool causal_;
    float scale_;
    int row_index_;
    CausalPrefix live_;
};

/** @brief Checked original row-major bank, retaining original row coordinates.
 * @tparam Format Original tensor storage.
 */
template<Storage Format>
class Batch final
{
public:
    /** @brief Validate the entire original bank before entering a worker team.
     * @param bank Complete original row-major bank; @param rows Original row count.
     * @param units Physical units per row; @param causal Original masking policy.
     * @param scale Finite score multiplier.
     * @return A synchronous original bank borrow, with no allocation/capacity ledger.
     * @throws std::invalid_argument For invalid geometry, policy or declared extent.
     */
    static Batch bind(Unit<Format> *bank, int rows, int units, bool causal, float scale)
    {
        if (rows < 0 || units < 0 || !std::isfinite(scale))
            throw std::invalid_argument("CPU softmax batch requires nonnegative geometry and a finite scale");
        const auto r = std::size_t(rows), u = std::size_t(units);
        if (u && r > std::numeric_limits<std::size_t>::max() / u)
            throw std::invalid_argument("CPU softmax batch element count overflow");
        requireBank(bank, r * u, sizeof(Unit<Format>));
        return Batch(bank, rows, units, causal, scale);
    }
    /** @return Whether the original physical bank is explicitly empty. */
    bool empty() const noexcept { return rows_ == 0 || units_ == 0; }
    /** @return Original row count. */
    int rows() const noexcept { return rows_; }
    /** @brief Bind one original row, never a worker-local causal coordinate.
     * @param index Original row ordinal inside the complete bank.
     * @return Checked row borrow.
     * @throws std::out_of_range If the requested row is outside the original bank.
     */
    Row<Format> row(int index) const
    {
        if (index < 0 || index >= rows_) throw std::out_of_range("CPU softmax original row outside bank");
        auto *original = units_ ? bank_ + std::size_t(index) * std::size_t(units_) : bank_;
        return Row<Format>::bind(original, units_, causal_, scale_, index);
    }
private:
    /** @brief Retain checked original bank and immutable row policy.
     * @param bank Original storage; @param rows Original row count; @param units Units per row.
     * @param causal Mask policy; @param scale Finite multiplier.
     */
    Batch(Unit<Format> *bank, int rows, int units, bool causal, float scale)
        : bank_(bank), rows_(rows), units_(units), causal_(causal), scale_(scale) {}
    Unit<Format> *bank_;
    int rows_, units_;
    bool causal_;
    float scale_;
};
} // namespace llaminar2::primitives::softmax_detail
