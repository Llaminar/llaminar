/**
 * @file NativeWeightSource.h
 * @brief Original native source geometry, independent of payload and prepared storage.
 *
 * The sealed loader owns the source name and file lifetime. This value carries
 * only original dimensions and canonical native group geometry; it grants no
 * allocation, mutable tensor or execution authority. Precision conversion keeps
 * this declaration distinct from the final executed representation.
 */
#pragma once
#include "tensors/TensorType.h"
#include "tensors/NativeWeightSourceWindow.h"
#include <cstddef>
#include <optional>
#include <vector>
namespace llaminar2 {
/** @brief Immutable original GGUF axis order and source-native encoded group. */
struct NativeWeightSourceGeometry final {
    std::vector<std::size_t> original_shape; ///< Complete original K/N/outer extent, never selected capacity.
    TensorType original_type; ///< Original native format before any requested conversion.
    std::size_t native_group_elements; ///< Canonical indivisible source group along K.
    std::size_t native_group_bytes; ///< Exact encoded bytes for that original group.
    bool operator==(const NativeWeightSourceGeometry &) const = default;
};
/**
 * @brief Ordered native source stripes assembled directly into one final matrix.
 *
 * Original coordinates remain in each segment. Destination offsets are the
 * cumulative extent along the declared axis, so validated stripes cannot leave
 * a hole or overlap. This is passive selection metadata, never a source owner.
 */
struct NativeWeightSourceAssembly final {
    /** @brief Axis whose ordered original stripes form the final dense matrix. */
    enum class Axis { Rows, Columns };
    Axis axis; ///< Typed concatenation axis; the other final axis is complete in every stripe.
    std::size_t final_rows; ///< Genuine destination N extent, separate from original source N.
    std::size_t final_columns; ///< Genuine destination K extent, separate from original source K.
    std::vector<NativeWeightSourceWindow> segments; ///< Ordered actual source rectangles; no compact source parent is implied.
    bool operator==(const NativeWeightSourceAssembly &) const = default;
};
/**
 * @brief Passive source facts retained after the actual prepared consumer commits.
 *
 * The original WeightBinding identity and original retained CPU cohort (where
 * present) remain authority. These values replace a transient source-parent
 * pointer without pretending that prepared packed bytes are a raw native tensor.
 */
struct NativeWeightSourceProjection final {
    NativeWeightSourceGeometry original; ///< Complete original tensor extent and native format.
    NativeWeightSourceWindow range; ///< Actual original matrix selection, independent of input window sizes.
    std::size_t matrix_count = 1; ///< One prepared matrix or complete original outer population for a resident tensor.
    std::optional<NativeWeightSourceAssembly> assembly; ///< Explicit ordered stripes; absence is the single original rectangle.
    /** @return Actual final matrix rows without relabeling source coordinates. */
    std::size_t finalRows() const noexcept { return assembly ? assembly->final_rows : range.rows; }
    /** @return Actual final matrix columns without relabeling source coordinates. */
    std::size_t finalColumns() const noexcept { return assembly ? assembly->final_columns : range.columns; }
    bool operator==(const NativeWeightSourceProjection &) const = default;
};
}
