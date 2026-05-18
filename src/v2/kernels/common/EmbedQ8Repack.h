/**
 * @file EmbedQ8Repack.h
 * @brief CPU-side repacking of any quantized embedding tensor into EmbedQ8Block format
 *
 * Uses the IINT8Unpackable interface to convert any quantized embedding table
 * (Q4_0, Q8_0, Q6_K, IQ4_NL, Q4_K, etc.) into a uniform EmbedQ8Block array.
 * This repacked representation is then uploaded to GPU for a single kernel to handle.
 *
 * The repack is a one-time cost at model loading / first inference call.
 * For Qwen2.5-0.5B (151936 × 896): ~4.2 million blocks, takes ~50-100ms on CPU.
 *
 * This file is NOT safe for CUDA/HIP compilation — include only from .cpp files.
 *
 * @author David Sanftenberg
 */

#pragma once

#include "EmbedQ8Block.h"
#include "../../tensors/TensorClasses.h"
#include "../../tensors/FP16Utils.h"
#include "../../utils/Logger.h"

#include <vector>
#include <cstring>
#include <chrono>

namespace llaminar2
{

    /**
     * @brief Repack result containing the EmbedQ8Block array and metadata
     */
    struct EmbedQ8RepackResult
    {
        std::vector<uint8_t> data; ///< Raw bytes of EmbedQ8Block array
        size_t blocks_per_row;     ///< Number of 32-element blocks per vocabulary entry
        size_t vocab_size;         ///< Number of vocabulary entries (rows)
        size_t total_blocks;       ///< Total blocks = vocab_size × blocks_per_row
        size_t byte_size;          ///< Total bytes = total_blocks × sizeof(EmbedQ8Block)
    };

    /**
     * @brief Repack a quantized embedding tensor into EmbedQ8Block format
     *
     * Takes any tensor implementing IINT8Unpackable and repacks every block
     * into the common EmbedQ8Block format: {FP16 scale, FP16 min, int8[32]}.
     *
     * For super-block formats (Q6_K, Q4_K, etc. with 256-element super-blocks),
     * uses the efficient unpack_superblock_to_int8() path which reads the
     * super-block header once for 8 sub-blocks.
     *
     * @param embed_table The quantized embedding tensor (must implement IINT8Unpackable)
     * @param d_model     Embedding dimension (number of columns)
     * @return EmbedQ8RepackResult containing the repacked data and metadata
     * @throws std::runtime_error if tensor doesn't implement IINT8Unpackable
     */
    inline EmbedQ8RepackResult repackEmbeddingToQ8(
        const TensorBase *embed_table,
        int d_model)
    {
        // Cast to IINT8Unpackable
        const auto *unpackable = dynamic_cast<const IINT8Unpackable *>(embed_table);
        if (!unpackable)
        {
            throw std::runtime_error(
                "[repackEmbeddingToQ8] Tensor does not implement IINT8Unpackable. "
                "Type: " +
                std::string(tensorTypeName(embed_table->native_type())));
        }

        const size_t vocab_size = embed_table->rows();
        const size_t blocks_per_row = (static_cast<size_t>(d_model) + 31) / 32;
        const size_t total_blocks = vocab_size * blocks_per_row;
        const size_t byte_size = total_blocks * sizeof(EmbedQ8Block);
        const size_t sb_size = unpackable->superblock_size();

        auto t0 = std::chrono::steady_clock::now();

        // Allocate output buffer
        EmbedQ8RepackResult result;
        result.blocks_per_row = blocks_per_row;
        result.vocab_size = vocab_size;
        result.total_blocks = total_blocks;
        result.byte_size = byte_size;
        result.data.resize(byte_size);

        EmbedQ8Block *out_blocks = reinterpret_cast<EmbedQ8Block *>(result.data.data());

        if (sb_size == 256)
        {
            // =====================================================================
            // Super-block path (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ formats)
            // More efficient: reads super-block header once for 8 sub-blocks
            // =====================================================================
            const size_t superblocks_per_row = (static_cast<size_t>(d_model) + 255) / 256;
            const size_t sub_blocks_per_superblock = 8; // 256 / 32

#pragma omp parallel for schedule(dynamic, 256)
            for (size_t row = 0; row < vocab_size; ++row)
            {
                alignas(64) int8_t sb_int8[256];
                float sb_scales[8];
                float sb_mins[8];

                for (size_t sb = 0; sb < superblocks_per_row; ++sb)
                {
                    unpackable->unpack_superblock_to_int8(row, sb, sb_int8, sb_scales, sb_mins);

                    // Pack 8 sub-blocks into 8 EmbedQ8Blocks
                    for (size_t sub = 0; sub < sub_blocks_per_superblock; ++sub)
                    {
                        size_t block_idx_in_row = sb * sub_blocks_per_superblock + sub;
                        if (block_idx_in_row >= blocks_per_row)
                            break; // d_model not a multiple of 256

                        EmbedQ8Block &out = out_blocks[row * blocks_per_row + block_idx_in_row];
                        out.d = fp32_to_fp16(sb_scales[sub]);
                        out.m = fp32_to_fp16(sb_mins[sub]);
                        std::memcpy(out.qs, &sb_int8[sub * 32], 32);
                    }
                }
            }
        }
        else
        {
            // =====================================================================
            // Simple 32-element block path (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL)
            // =====================================================================

#pragma omp parallel for schedule(dynamic, 256)
            for (size_t row = 0; row < vocab_size; ++row)
            {
                alignas(64) int8_t block_int8[32];

                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    unpackable->unpack_block_to_int8(row, b, block_int8);

                    EmbedQ8Block &out = out_blocks[row * blocks_per_row + b];
                    out.d = fp32_to_fp16(unpackable->get_block_scale(row, b));
                    out.m = fp32_to_fp16(unpackable->get_block_min(row, b));
                    std::memcpy(out.qs, block_int8, 32);
                }
            }
        }

        auto t1 = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        LOG_DEBUG("[repackEmbeddingToQ8] Repacked "
                 << tensorTypeName(embed_table->native_type()) << " embedding "
                 << vocab_size << "×" << d_model << " → EmbedQ8 "
                 << (byte_size / (1024 * 1024)) << " MB (" << total_blocks << " blocks) in "
                 << ms << " ms");

        return result;
    }

    /**
     * @brief Repack a vocab-range slice of a quantized embedding tensor into EmbedQ8Block format
     *
     * Repacks rows [vocab_start, vocab_start + vocab_count) from the original tensor.
     * Used for vocabulary-parallel embedding sharding where each device holds a
     * contiguous slice of the vocabulary.
     *
     * @param embed_table  The quantized embedding tensor (must implement IINT8Unpackable)
     * @param d_model      Embedding dimension (number of columns)
     * @param vocab_start  First vocabulary row to include
     * @param vocab_count  Number of vocabulary rows to include
     * @return EmbedQ8RepackResult containing the repacked slice
     */
    inline EmbedQ8RepackResult repackEmbeddingToQ8(
        const TensorBase *embed_table,
        int d_model,
        size_t vocab_start,
        size_t vocab_count)
    {
        const auto *unpackable = dynamic_cast<const IINT8Unpackable *>(embed_table);
        if (!unpackable)
        {
            throw std::runtime_error(
                "[repackEmbeddingToQ8] Tensor does not implement IINT8Unpackable. "
                "Type: " +
                std::string(tensorTypeName(embed_table->native_type())));
        }

        const size_t total_vocab = embed_table->rows();
        if (vocab_start + vocab_count > total_vocab)
        {
            throw std::runtime_error(
                "[repackEmbeddingToQ8] Vocab range [" + std::to_string(vocab_start) +
                ", " + std::to_string(vocab_start + vocab_count) +
                ") exceeds total vocab " + std::to_string(total_vocab));
        }

        const size_t blocks_per_row = (static_cast<size_t>(d_model) + 31) / 32;
        const size_t total_blocks = vocab_count * blocks_per_row;
        const size_t byte_size = total_blocks * sizeof(EmbedQ8Block);
        const size_t sb_size = unpackable->superblock_size();

        auto t0 = std::chrono::steady_clock::now();

        EmbedQ8RepackResult result;
        result.blocks_per_row = blocks_per_row;
        result.vocab_size = vocab_count;
        result.total_blocks = total_blocks;
        result.byte_size = byte_size;
        result.data.resize(byte_size);

        EmbedQ8Block *out_blocks = reinterpret_cast<EmbedQ8Block *>(result.data.data());

        if (sb_size == 256)
        {
            const size_t superblocks_per_row = (static_cast<size_t>(d_model) + 255) / 256;
            const size_t sub_blocks_per_superblock = 8;

#pragma omp parallel for schedule(dynamic, 256)
            for (size_t i = 0; i < vocab_count; ++i)
            {
                const size_t src_row = vocab_start + i;
                alignas(64) int8_t sb_int8[256];
                float sb_scales[8];
                float sb_mins[8];

                for (size_t sb = 0; sb < superblocks_per_row; ++sb)
                {
                    unpackable->unpack_superblock_to_int8(src_row, sb, sb_int8, sb_scales, sb_mins);

                    for (size_t sub = 0; sub < sub_blocks_per_superblock; ++sub)
                    {
                        size_t block_idx_in_row = sb * sub_blocks_per_superblock + sub;
                        if (block_idx_in_row >= blocks_per_row)
                            break;

                        EmbedQ8Block &out = out_blocks[i * blocks_per_row + block_idx_in_row];
                        out.d = fp32_to_fp16(sb_scales[sub]);
                        out.m = fp32_to_fp16(sb_mins[sub]);
                        std::memcpy(out.qs, &sb_int8[sub * 32], 32);
                    }
                }
            }
        }
        else
        {
#pragma omp parallel for schedule(dynamic, 256)
            for (size_t i = 0; i < vocab_count; ++i)
            {
                const size_t src_row = vocab_start + i;
                alignas(64) int8_t block_int8[32];

                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    unpackable->unpack_block_to_int8(src_row, b, block_int8);

                    EmbedQ8Block &out = out_blocks[i * blocks_per_row + b];
                    out.d = fp32_to_fp16(unpackable->get_block_scale(src_row, b));
                    out.m = fp32_to_fp16(unpackable->get_block_min(src_row, b));
                    std::memcpy(out.qs, block_int8, 32);
                }
            }
        }

        auto t1 = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        LOG_DEBUG("[repackEmbeddingToQ8] Repacked "
                 << tensorTypeName(embed_table->native_type()) << " embedding "
                 << "rows [" << vocab_start << ", " << (vocab_start + vocab_count) << ") of "
                 << total_vocab << "×" << d_model << " → EmbedQ8 "
                 << (byte_size / (1024 * 1024)) << " MB (" << total_blocks << " blocks) in "
                 << ms << " ms");

        return result;
    }

} // namespace llaminar2
