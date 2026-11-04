/**
 * @file NativeWeightSourceRange.h
 * @brief Window an original native rectangle without materializing its full extent.
 *
 * Readers and final consumers share original coordinates. This iterator owns no
 * bytes: the caller's admitted source/scratch arena determines useful capacity.
 * Whole rows stay contiguous; a row larger than the arena is split only on
 * source-native encoded group boundaries. No source padding enters a window.
 */
#pragma once
#include "tensors/NativeWeightSource.h"
#include "tensors/NativeWeightSourceWindow.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
namespace llaminar2 {
/**
 * @brief Visit exact native windows covering one original matrix rectangle once.
 * @param source Original sealed tensor geometry, independent of destination precision.
 * @param range Actual original row/column/outer selection and its useful row bytes.
 * @param native_capacity Useful native prefix of the admitted shared arena (scratch excluded).
 * @param consume Synchronous consumer of one original rectangle before the arena is reused.
 * @throws std::invalid_argument For foreign/empty geometry or an arena smaller than one group.
 * @note Consumers must not retain native payload pointers across calls. This
 * function describes coordinates only and cannot extend payload lifetime.
 */
template<class Consumer>
void forEachNativeWeightSourceWindow(const NativeWeightSourceGeometry &source,
    const NativeWeightSourceWindow &range, size_t native_capacity, Consumer &&consume)
{
    const auto multiply = [](size_t a, size_t b) {
        if (b && a > std::numeric_limits<size_t>::max() / b)
            throw std::overflow_error("Native source range geometry overflows");
        return a * b;
    };
    if (source.original_shape.empty() || !source.native_group_elements || !source.native_group_bytes)
        throw std::invalid_argument("Native source range requires original group geometry");
    const size_t columns = source.original_shape.front();
    const size_t rows = source.original_shape.size() > 1 ? source.original_shape[1] : 1;
    size_t outer = 1;
    for (size_t axis = 2; axis < source.original_shape.size(); ++axis)
        outer = multiply(outer, source.original_shape[axis]);
    if (!range.rows || !range.columns || !rows || !columns || !outer ||
        range.original_expert >= outer || range.first_row > rows || range.rows > rows - range.first_row ||
        range.first_column > columns || range.columns > columns - range.first_column ||
        range.first_column % source.native_group_elements || range.columns % source.native_group_elements ||
        columns % source.native_group_elements ||
        range.native_row_bytes != multiply(range.columns / source.native_group_elements, source.native_group_bytes) ||
        native_capacity < source.native_group_bytes)
        throw std::invalid_argument("Native source range disagrees with original source or admitted arena");
    if (range.native_row_bytes <= native_capacity) {
        const size_t rows_per_window = native_capacity / range.native_row_bytes;
        for (size_t offset = 0; offset < range.rows;) {
            auto window = range;
            window.first_row += offset;
            window.rows = std::min(rows_per_window, range.rows - offset);
            consume(window);
            offset += window.rows;
        }
    } else {
        const size_t groups_per_window = native_capacity / source.native_group_bytes;
        const size_t groups_per_row = range.columns / source.native_group_elements;
        for (size_t row = 0; row < range.rows; ++row) {
            for (size_t group = 0; group < groups_per_row;) {
                const size_t groups = std::min(groups_per_window, groups_per_row - group);
                auto window = range;
                window.first_row += row;
                window.rows = 1;
                window.first_column += multiply(group, source.native_group_elements);
                window.columns = multiply(groups, source.native_group_elements);
                window.native_row_bytes = multiply(groups, source.native_group_bytes);
                consume(window);
                group += groups;
            }
        }
    }
}
/**
 * @brief Validate and visit ordered original stripes with exact dense destination offsets.
 * @param source Same immutable original source geometry used by the reader.
 * @param range True original bounding rectangle, never the compact destination shape.
 * @param assembly Optional original stripe selection; absence visits the rectangle once.
 * @param consume Synchronous callback(original stripe, destination first row, first column).
 * @throws std::invalid_argument For invalid axes, source bounds/groups, holes, overlap or reshaped outer populations.
 * @note Every stripe is validated before any callback publishes output. Destination
 * offsets are cumulative, so no per-element coverage storage or second ledger exists.
 */
template<class Consumer>
void forEachNativeWeightSourceStripe(const NativeWeightSourceGeometry &source,
    const NativeWeightSourceWindow &range, const std::optional<NativeWeightSourceAssembly> &assembly,
    Consumer &&consume)
{
    const auto validate = [&](const NativeWeightSourceWindow &stripe) {
        forEachNativeWeightSourceWindow(source,stripe,stripe.nativeBytes(),[](const auto &) {});
    };
    validate(range);
    if (!assembly) { consume(range,size_t{0},size_t{0}); return; }
    const auto &selection=*assembly;
    if (source.original_shape.size()>2 || !selection.final_rows || !selection.final_columns || selection.segments.empty() ||
        selection.final_rows>std::numeric_limits<size_t>::max()/selection.final_columns)
        throw std::invalid_argument("Native source assembly requires one original matrix/vector and positive final stripes");
    const bool rows=selection.axis==NativeWeightSourceAssembly::Axis::Rows;
    if (!rows && selection.axis!=NativeWeightSourceAssembly::Axis::Columns)
        throw std::invalid_argument("Native source assembly has an invalid declared axis");
    if (source.original_shape.size()==1 && (rows || selection.final_rows!=1))
        throw std::invalid_argument("Native vector assembly must preserve its original vector rank");
    size_t covered=0;
    const size_t extent=rows?selection.final_rows:selection.final_columns;
    for (const auto &stripe:selection.segments) {
        validate(stripe);
        if (stripe.original_expert!=range.original_expert || stripe.selected_expert_ordinal!=range.selected_expert_ordinal ||
            stripe.first_row<range.first_row || stripe.first_row-range.first_row>range.rows ||
            stripe.rows>range.rows-(stripe.first_row-range.first_row) ||
            stripe.first_column<range.first_column || stripe.first_column-range.first_column>range.columns ||
            stripe.columns>range.columns-(stripe.first_column-range.first_column) ||
            (rows && (stripe.first_column!=range.first_column || stripe.columns!=selection.final_columns)) ||
            (!rows && (stripe.first_row!=range.first_row || stripe.rows!=selection.final_rows)))
            throw std::invalid_argument("Native source stripe changed its original range or complete destination cross-axis");
        const size_t length=rows?stripe.rows:stripe.columns;
        if (covered>extent || length>extent-covered)
            throw std::invalid_argument("Native source stripes exceed their exact final destination extent");
        covered+=length;
    }
    if (covered!=extent)
        throw std::invalid_argument("Native source stripes leave an incomplete final destination extent");
    covered=0;
    for (const auto &stripe:selection.segments) {
        consume(stripe,rows?covered:size_t{0},rows?size_t{0}:covered);
        covered+=rows?stripe.rows:stripe.columns;
    }
}

}
