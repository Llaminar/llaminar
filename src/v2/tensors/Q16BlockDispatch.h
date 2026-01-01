/**
 * @file Q16BlockDispatch.h
 * @brief Type-safe dispatch helpers for Q16_1Tensor variable block sizes
 * @author David Sanftenberg
 * @date January 2026
 *
 * Provides template-based dispatch functions for handling Q16_1Tensor with
 * different block sizes (32, 64, 128) in a type-safe manner.
 *
 * Key Functions:
 * - dispatchQ16Block() - Dispatch to typed block access based on runtime block size
 * - dispatchQ16BlockMutable() - Mutable version for writes
 * - forEachQ16Block() - Iterate over all blocks with typed access
 *
 * Example:
 * @code
 * // Get first block's scale (works for any block size)
 * float scale = dispatchQ16Block(tensor, [](auto* blocks, size_t bs) {
 *     return blocks[0].d;
 * });
 *
 * // Iterate all blocks
 * forEachQ16Block(tensor, [](const auto& block, size_t idx) {
 *     LOG_DEBUG("Block " << idx << " scale: " << block.d);
 * });
 * @endcode
 *
 * @see Q16_1Tensor::as_block_32/64/128() for direct typed access
 * @see BlockStructures.h for Q16_1Block, Q16_1Block_64, Q16_1Block_128
 */

#pragma once

#include "Tensors.h" // For Q16_1Tensor and Q16BlockSize
#include "../utils/Assertions.h"
#include <utility>

namespace llaminar2
{

    /**
     * @brief Dispatch to typed block access based on Q16_1Tensor's runtime block size
     *
     * @tparam Func Callable with signature: ReturnType(const BlockType*, size_t block_size)
     * @param tensor Q16_1Tensor to dispatch on
     * @param func Function receiving typed block pointer and block size
     * @return Return value of func
     *
     * Example:
     * @code
     * float first_scale = dispatchQ16Block(tensor, [](auto* blocks, size_t bs) {
     *     return blocks[0].d;
     * });
     * @endcode
     */
    template <typename Func>
    auto dispatchQ16Block(const Q16_1Tensor *tensor, Func &&func)
        -> decltype(func(std::declval<const Q16_1Block *>(), size_t{}))
    {
        switch (tensor->q16_block_size())
        {
        case Q16BlockSize::BLOCK_32:
            return std::forward<Func>(func)(
                tensor->as_block_32(),
                Q16_1Block::BLOCK_SIZE);
        case Q16BlockSize::BLOCK_64:
            return std::forward<Func>(func)(
                tensor->as_block_64(),
                Q16_1Block_64::BLOCK_SIZE);
        case Q16BlockSize::BLOCK_128:
            return std::forward<Func>(func)(
                tensor->as_block_128(),
                Q16_1Block_128::BLOCK_SIZE);
        default:
            LLAMINAR_UNREACHABLE("Invalid Q16BlockSize: " << static_cast<int>(tensor->q16_block_size()));
        }
    }

    /**
     * @brief Mutable version of dispatchQ16Block for write access
     *
     * @tparam Func Callable with signature: ReturnType(BlockType*, size_t block_size)
     * @param tensor Q16_1Tensor to dispatch on (non-const)
     * @param func Function receiving mutable typed block pointer and block size
     * @return Return value of func
     */
    template <typename Func>
    auto dispatchQ16BlockMutable(Q16_1Tensor *tensor, Func &&func)
        -> decltype(func(std::declval<Q16_1Block *>(), size_t{}))
    {
        switch (tensor->q16_block_size())
        {
        case Q16BlockSize::BLOCK_32:
            return std::forward<Func>(func)(
                tensor->mutable_as_block_32(),
                Q16_1Block::BLOCK_SIZE);
        case Q16BlockSize::BLOCK_64:
            return std::forward<Func>(func)(
                tensor->mutable_as_block_64(),
                Q16_1Block_64::BLOCK_SIZE);
        case Q16BlockSize::BLOCK_128:
            return std::forward<Func>(func)(
                tensor->mutable_as_block_128(),
                Q16_1Block_128::BLOCK_SIZE);
        default:
            LLAMINAR_UNREACHABLE("Invalid Q16BlockSize: " << static_cast<int>(tensor->q16_block_size()));
        }
    }

    /**
     * @brief Iterate over all blocks with typed access (const version)
     *
     * @tparam Func Callable with signature: void(const BlockType& block, size_t block_idx)
     * @param tensor Q16_1Tensor to iterate
     * @param func Function called for each block with typed reference and index
     *
     * Example:
     * @code
     * forEachQ16Block(tensor, [](const auto& block, size_t idx) {
     *     LOG_DEBUG("Block " << idx << " scale: " << block.d);
     * });
     * @endcode
     */
    template <typename Func>
    void forEachQ16Block(const Q16_1Tensor *tensor, Func &&func)
    {
        const size_t total = tensor->total_blocks();
        dispatchQ16Block(tensor, [&](auto *blocks, size_t /*bs*/)
                         {
            for (size_t i = 0; i < total; ++i) {
                func(blocks[i], i);
            } });
    }

    /**
     * @brief Iterate over all blocks with typed access (mutable version)
     *
     * @tparam Func Callable with signature: void(BlockType& block, size_t block_idx)
     * @param tensor Q16_1Tensor to iterate
     * @param func Function called for each block with mutable typed reference and index
     */
    template <typename Func>
    void forEachQ16BlockMutable(Q16_1Tensor *tensor, Func &&func)
    {
        const size_t total = tensor->total_blocks();
        dispatchQ16BlockMutable(tensor, [&](auto *blocks, size_t /*bs*/)
                                {
            for (size_t i = 0; i < total; ++i) {
                func(blocks[i], i);
            } });
    }

    /**
     * @brief Process blocks in a row with typed access
     *
     * @tparam Func Callable with signature: void(const BlockType& block, size_t col_offset)
     * @param tensor Q16_1Tensor to process
     * @param row Row index to process
     * @param func Function called for each block in the row with typed reference and column offset
     */
    template <typename Func>
    void forEachQ16BlockInRow(const Q16_1Tensor *tensor, size_t row, Func &&func)
    {
        const size_t bpr = tensor->blocks_per_row();
        dispatchQ16Block(tensor, [&](auto *blocks, size_t block_size)
                         {
            const size_t row_start = row * bpr;
            for (size_t b = 0; b < bpr; ++b) {
                func(blocks[row_start + b], b * block_size);
            } });
    }

} // namespace llaminar2
