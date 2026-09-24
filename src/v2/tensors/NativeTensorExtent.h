/**
 * @file NativeTensorExtent.h
 * @brief Checked native block extents for borrowed tensor storage.
 *
 * A view's empty owning vector does not describe its addressable bytes. Native
 * weight views retain complete source blocks; their shape and block ABI define
 * the logical extent independently of who owns the allocation. This is tensor
 * geometry, not an allocation/admission ledger. No storage is materialized.
 */
#pragma once
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>

namespace llaminar2
{
    /**
     * @brief Calculate native packed bytes with checked element/block arithmetic.
     * @param shape Logical dimensions; empty/zero-sized shapes have zero extent.
     * @param block_elements Elements encoded by one native block.
     * @param block_bytes Bytes in that same block ABI, never a prepared format.
     * @return Rounded whole-block extent, including all axes of expert parents.
     * @throws std::invalid_argument for a zero block geometry.
     * @throws std::overflow_error before any element or byte product wraps.
     */
    inline size_t nativeTensorExtent(std::span<const size_t> shape, size_t block_elements, size_t block_bytes)
    {
        if (!block_elements || !block_bytes) throw std::invalid_argument("Native tensor extent requires nonzero block geometry");
        size_t elements = shape.empty() ? 0 : 1;
        for (const size_t dimension : shape)
        {
            if (dimension && elements > std::numeric_limits<size_t>::max() / dimension)
                throw std::overflow_error("Native tensor element extent overflow");
            elements *= dimension;
        }
        const size_t blocks = elements / block_elements + (elements % block_elements != 0);
        if (blocks > std::numeric_limits<size_t>::max() / block_bytes)
            throw std::overflow_error("Native tensor byte extent overflow");
        return blocks * block_bytes;
    }

    /** @return Fixed source-block extent without a duplicated format-size table. */
    template<class Block>
    size_t nativeTensorExtent(std::span<const size_t> shape)
    {
        return nativeTensorExtent(shape, Block::BLOCK_SIZE, sizeof(Block));
    }
}
