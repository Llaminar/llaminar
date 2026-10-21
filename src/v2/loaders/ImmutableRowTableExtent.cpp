/**
 * @file ImmutableRowTableExtent.cpp
 * @brief Preserve checked original native byte arithmetic without materialization.
 */
#include "ImmutableRowTableExtent.h"
#include <limits>
#include <stdexcept>
namespace llaminar2 {
ImmutableRowTableExtent ImmutableRowTableExtent::fromNative(
    GGUFTensorType format, std::uint64_t rows, std::size_t row_bytes)
{
    if (!rows || !row_bytes)
        throw std::invalid_argument("Immutable row source requires positive exact native geometry");
    if (rows > std::numeric_limits<std::size_t>::max() / row_bytes)
        throw std::overflow_error("Immutable row source native extent overflows size_t");
    return ImmutableRowTableExtent(format, rows, row_bytes, static_cast<std::size_t>(rows) * row_bytes);
}
}
