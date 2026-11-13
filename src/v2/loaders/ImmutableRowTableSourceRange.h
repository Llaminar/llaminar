/**
 * @file ImmutableRowTableSourceRange.h
 * @brief Checked contiguous original-native rows without inventing a sliced source identity.
 *
 * This immutable geometry is not an allocation permit or a file descriptor.
 * Its full extent remains the original loader source. Materialization retains
 * and authenticates that original file/tensor and reads only the selected rows.
 */
#pragma once
#include "ImmutableRowTableExtent.h"
#include <stdexcept>
namespace llaminar2 {
/** @brief Exact half-open native range, with the unchanged full source geometry. */
class ImmutableRowTableSourceRange final {
public:
    /**
     * @brief Check a positive contiguous subset of the original source.
     * @param source Complete original codec/row/stride extent, never a sub-tensor.
     * @param first_row First global original row.
     * @param rows Number of owned native rows.
     * @return Immutable range; checked source extent already bounds every byte product.
     * @throws std::invalid_argument For empty, outside or overflowing half-open rows.
     */
    static ImmutableRowTableSourceRange checked(ImmutableRowTableExtent source,std::uint64_t first_row,std::uint64_t rows)
    {
        if(!rows || first_row>=source.rows() || rows>source.rows()-first_row)
            throw std::invalid_argument("Native row range is empty or outside its original source");
        return ImmutableRowTableSourceRange(source,first_row,rows);
    }
    /** @return Complete source range without changing its identity. */
    static ImmutableRowTableSourceRange whole(ImmutableRowTableExtent source) {return checked(source,0,source.rows());}
    /** @return Original full source, including all gaps/trailing rows. */
    const ImmutableRowTableExtent &sourceExtent() const noexcept {return source_;}
    /** @return First original global row. */
    std::uint64_t firstRow() const noexcept {return first_;}
    /** @return Exact owned rows, not full source capacity. */
    std::uint64_t rows() const noexcept {return rows_;}
    /** @return Exact original native byte offset relative to the full tensor. */
    std::size_t byteOffset() const noexcept {return static_cast<std::size_t>(first_)*source_.rowBytes();}
    /** @return Exact owned native bytes; no row padding or requantization. */
    std::size_t nativeBytes() const noexcept {return static_cast<std::size_t>(rows_)*source_.rowBytes();}
    /** @return Whether existing whole-source consumers may accept this range. */
    bool isWhole() const noexcept {return first_==0 && rows_==source_.rows();}
    /** @return Whether the global source row belongs to this half-open range. */
    bool contains(std::uint64_t row) const noexcept {return row>=first_ && row-first_<rows_;}
    bool operator==(const ImmutableRowTableSourceRange &) const = default;
private:
    /** @brief Only checked source bounds construct immutable range metadata. */
    ImmutableRowTableSourceRange(ImmutableRowTableExtent source,std::uint64_t first,std::uint64_t rows)
        :source_(source),first_(first),rows_(rows) {}
    ImmutableRowTableExtent source_;
    std::uint64_t first_,rows_;
};
} // namespace llaminar2
