/**
 * @file NativeQuantisedRepackColumns.h
 * @brief Exact native source cover and selected 32-value execution column interval.
 *
 * Encoded GGUF superblocks remain indivisible read units. A prepared GPU matrix
 * may own fewer logical columns: original decoder subgroups write only that
 * interval, preserving every payload bit, source scale and arithmetic identity.
 * This metadata owns no bytes, allocation budget, device state or source reader.
 */
#pragma once
#include <limits>
#include <stdexcept>

namespace llaminar2
{
/** @brief Validated source-cover width and selected execution interval, expressed in columns. */
class NativeQuantisedRepackColumns final
{
public:
    /** @brief Select every original logical block of an already bounded source. @param columns Positive K32 width. */
    static NativeQuantisedRepackColumns whole(int columns) { return slice(columns,0,columns); }
    /**
     * @brief Bind one exact execution interval within the encoded source cover.
     * @param source_columns Complete columns represented by each uploaded native source row.
     * @param first_column First selected column relative to that same uploaded row.
     * @param columns Exact prepared execution width; no padding becomes a consumer.
     * @return Sealed positive K32 geometry, independent of actual source byte ownership.
     * @throws std::invalid_argument For misalignment, absent or out-of-cover intervals.
     */
    static NativeQuantisedRepackColumns slice(int source_columns,int first_column,int columns)
    {
        if(source_columns<=0||columns<=0||first_column<0||source_columns%32||first_column%32||columns%32||
            columns>source_columns||first_column>source_columns-columns)
            throw std::invalid_argument("Native repack columns require a nonempty K32 interval within the original source cover");
        return NativeQuantisedRepackColumns(source_columns,first_column,columns);
    }
    /** @return Full encoded source-row width, independent of final execution K. */
    constexpr int sourceColumns() const noexcept { return source_columns_; }
    /** @return First logical column inside that uploaded source row. */
    constexpr int firstColumn() const noexcept { return first_column_; }
    /** @return Exact prepared execution width. */
    constexpr int columns() const noexcept { return columns_; }
    /** @return Original 32-value subgroup selected by the first output block. */
    constexpr int firstBlock() const noexcept { return first_column_/32; }
    /** @return Number of live execution blocks written by the repacker. */
    constexpr int blocks() const noexcept { return columns_/32; }
    /** @return Complete immutable source/destination geometry equality. */
    bool operator==(const NativeQuantisedRepackColumns &) const = default;
private:
    /** @brief Seal checked arithmetic geometry without granting source or allocation authority. */
    NativeQuantisedRepackColumns(int source_columns,int first_column,int columns)
        :source_columns_(source_columns),first_column_(first_column),columns_(columns) {}
    int source_columns_,first_column_,columns_;
};

/** @brief Minimum original encoded-group cover and its exact selected execution interval. */
struct NativeQuantisedColumnSourceCover final
{
    int first_source_column; ///< Original source coordinate of the first retained encoded group.
    NativeQuantisedRepackColumns selected; ///< Logical cut relative to the bounded cover, never the whole source.
    /**
     * @brief Resolve the smallest source-group cover for any legal native K32 cut.
     * @param original_columns Complete original matrix K, divisible by its encoded source group.
     * @param first_column Original selected K32 column.
     * @param columns Positive final K32 width.
     * @param source_group_elements Original indivisible encoded group, such as32 or256.
     * @return Original cover start plus exact source/selection widths; no source data is read.
     * @throws std::invalid_argument For malformed original source or out-of-range selected columns.
     */
    static NativeQuantisedColumnSourceCover resolve(int original_columns,int first_column,int columns,int source_group_elements)
    {
        if(source_group_elements<=0||source_group_elements%32||original_columns<=0||original_columns%source_group_elements)
            throw std::invalid_argument("Native column cover requires complete original encoded source groups");
        (void)NativeQuantisedRepackColumns::slice(original_columns,first_column,columns);
        const int first=first_column-first_column%source_group_elements;
        const int end=first_column+columns;
        // end<=original_columns and that original bound is group aligned, so
        // rounding cannot overflow the already admitted native INT32 K axis.
        const int remainder=end%source_group_elements;
        const int cover_end=remainder ? end+(source_group_elements-remainder) : end;
        return {first,NativeQuantisedRepackColumns::slice(cover_end-first,first_column-first,columns)};
    }
};
}
