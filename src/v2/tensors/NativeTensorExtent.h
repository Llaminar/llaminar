/**
 * @file NativeTensorExtent.h
 * @brief Checked native block extents for borrowed tensor storage.
 *
 * A view's empty owning vector does not describe its addressable bytes. Native
 * weight views retain complete source blocks; their shape and block ABI define
 * the logical extent independently of who owns the allocation. This is tensor
 * geometry, not an allocation/admission ledger. No storage is materialized.
 * Borrowed matrices retain each row's partial final block; flattened tensor
 * extents keep their separate existing non-matrix storage contract.
 */
#pragma once
#include <array>
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

    /**
     * @brief Calculate a row-padded source matrix extent without flattening K tails.
     * @param rows Number of independently block-padded logical rows.
     * @param columns Logical values per row, excluding final-block padding.
     * @param block_elements Values encoded by each original source block.
     * @param block_bytes Original source bytes per block, never prepared bytes.
     * @return Exact row-padded byte extent; either zero dimension returns zero.
     * @throws std::invalid_argument for either zero block geometry.
     * @throws std::overflow_error for overflowing logical or physical products.
     * @details Callers supply an explicit matrix layout. This function does not
     * infer a row axis for GGUF expert parents, packed KV or arbitrary tensors.
     * Existing checked block arithmetic remains the only extent authority.
     */
    inline size_t nativeMatrixExtent(size_t rows, size_t columns,
                                     size_t block_elements, size_t block_bytes)
    {
        if (!block_elements || !block_bytes)
            throw std::invalid_argument("Native matrix extent requires nonzero block geometry");
        if (!rows || !columns) return 0;
        const std::array matrix_shape{rows, columns};
        (void)nativeTensorExtent(matrix_shape, 1, 1); // Validate logical numel too.
        const std::array row_shape{columns};
        const size_t row_bytes = nativeTensorExtent(row_shape, block_elements, block_bytes);
        const std::array row_count{rows};
        return nativeTensorExtent(row_count, 1, row_bytes);
    }

    /**
     * @brief Resolve an explicit source matrix using its original block ABI.
     * @tparam Block Fixed native source block with BLOCK_SIZE, never a prepared record.
     * @param rows Independently padded source rows.
     * @param columns Logical source values per row.
     * @return Checked complete-row byte extent without materializing storage.
     * @throws std::invalid_argument for invalid block geometry.
     * @throws std::overflow_error for overflowing logical or physical products.
     */
    template<class Block>
    size_t nativeMatrixExtent(size_t rows, size_t columns)
    {
        return nativeMatrixExtent(rows, columns, Block::BLOCK_SIZE, sizeof(Block));
    }

    /** @return Fixed source-block extent without a duplicated format-size table. */
    template<class Block>
    size_t nativeTensorExtent(std::span<const size_t> shape)
    {
        return nativeTensorExtent(shape, Block::BLOCK_SIZE, sizeof(Block));
    }
}
