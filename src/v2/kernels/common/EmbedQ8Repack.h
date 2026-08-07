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

#include <array>
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

    namespace detail
    {
        /**
         * @brief Selects the diagnostic wording used by the shared repack implementation.
         */
        enum class EmbedQ8RepackLogStyle
        {
            FullTable,
            VocabRange
        };

        /**
         * @brief Repack one contiguous vocabulary range without reading past a row.
         *
         * `IINT8Unpackable` implementations may expose a 256-element grouped unpack
         * operation even when their native storage consists of independent 32-element
         * blocks. The grouped operation therefore requires eight complete source blocks.
         * This helper invokes it only for complete groups and handles the remaining
         * blocks through the bounded single-block interface. Keeping that partition in
         * one implementation makes full-table and vocabulary-slice repacks obey the
         * same memory-safety and byte-layout contract.
         *
         * @param embed_table Quantized source tensor implementing `IINT8Unpackable`.
         * @param d_model Number of logical columns in each embedding row.
         * @param vocab_start First source row to repack.
         * @param vocab_count Number of consecutive source rows to repack.
         * @param log_style Diagnostic form used after the operation completes.
         * @return Repacked rows and their exact layout metadata.
         * @throws std::invalid_argument for a null tensor or non-positive row width.
         * @throws std::runtime_error for an unsupported tensor or unpack group size.
         * @throws std::runtime_error when the requested row range exceeds the tensor.
         */
        inline EmbedQ8RepackResult repackEmbeddingRangeToQ8(
            const TensorBase *embed_table,
            int d_model,
            size_t vocab_start,
            size_t vocab_count,
            EmbedQ8RepackLogStyle log_style)
        {
            if (embed_table == nullptr)
            {
                throw std::invalid_argument(
                    "[repackEmbeddingToQ8] Embedding tensor must not be null");
            }
            if (d_model <= 0)
            {
                throw std::invalid_argument(
                    "[repackEmbeddingToQ8] Embedding dimension must be positive");
            }

            const auto *unpackable = dynamic_cast<const IINT8Unpackable *>(embed_table);
            if (!unpackable)
            {
                throw std::runtime_error(
                    "[repackEmbeddingToQ8] Tensor does not implement IINT8Unpackable. "
                    "Type: " +
                    std::string(tensorTypeName(embed_table->native_type())));
            }

            const size_t total_vocab = embed_table->rows();
            if (vocab_start > total_vocab || vocab_count > total_vocab - vocab_start)
            {
                throw std::runtime_error(
                    "[repackEmbeddingToQ8] Vocab range [" + std::to_string(vocab_start) +
                    ", " + std::to_string(vocab_start + vocab_count) +
                    ") exceeds total vocab " + std::to_string(total_vocab));
            }

            constexpr size_t kElementsPerOutputBlock = 32;
            constexpr size_t kElementsPerGroupedUnpack = 256;
            constexpr size_t kBlocksPerGroupedUnpack =
                kElementsPerGroupedUnpack / kElementsPerOutputBlock;

            const size_t unpack_group_size = unpackable->superblock_size();
            if (unpack_group_size != kElementsPerOutputBlock &&
                unpack_group_size != kElementsPerGroupedUnpack)
            {
                throw std::runtime_error(
                    "[repackEmbeddingToQ8] Unsupported unpack group size " +
                    std::to_string(unpack_group_size));
            }

            const size_t blocks_per_row =
                (static_cast<size_t>(d_model) + kElementsPerOutputBlock - 1) /
                kElementsPerOutputBlock;
            const size_t total_blocks = vocab_count * blocks_per_row;
            const size_t byte_size = total_blocks * sizeof(EmbedQ8Block);
            const size_t complete_group_count =
                unpack_group_size == kElementsPerGroupedUnpack
                    ? blocks_per_row / kBlocksPerGroupedUnpack
                    : 0;
            const size_t first_tail_block = complete_group_count * kBlocksPerGroupedUnpack;

            const auto started_at = std::chrono::steady_clock::now();

            EmbedQ8RepackResult result;
            result.blocks_per_row = blocks_per_row;
            result.vocab_size = vocab_count;
            result.total_blocks = total_blocks;
            result.byte_size = byte_size;
            result.data.resize(byte_size);

            auto *out_blocks = reinterpret_cast<EmbedQ8Block *>(result.data.data());

#pragma omp parallel for schedule(dynamic, 256)
            for (size_t output_row = 0; output_row < vocab_count; ++output_row)
            {
                const size_t source_row = vocab_start + output_row;

                // A grouped unpack is legal only when all eight source blocks belong
                // to this row. In particular, `ceil(blocks_per_row / 8)` is wrong:
                // its final call crosses the row boundary for short or tailed rows.
                for (size_t group = 0; group < complete_group_count; ++group)
                {
                    alignas(64) std::array<int8_t, kElementsPerGroupedUnpack> values{};
                    std::array<float, kBlocksPerGroupedUnpack> scales{};
                    std::array<float, kBlocksPerGroupedUnpack> mins{};
                    unpackable->unpack_superblock_to_int8(
                        source_row, group, values.data(), scales.data(), mins.data());

                    for (size_t sub_block = 0;
                         sub_block < kBlocksPerGroupedUnpack;
                         ++sub_block)
                    {
                        const size_t block_in_row =
                            group * kBlocksPerGroupedUnpack + sub_block;
                        EmbedQ8Block &out =
                            out_blocks[output_row * blocks_per_row + block_in_row];
                        out.d = fp32_to_fp16(scales[sub_block]);
                        out.m = fp32_to_fp16(mins[sub_block]);
                        std::memcpy(
                            out.qs,
                            values.data() + sub_block * kElementsPerOutputBlock,
                            kElementsPerOutputBlock);
                    }
                }

                // Process every incomplete group through the exact per-block API.
                // This is both bounded and required for dimensions below 256.
                alignas(64) std::array<int8_t, kElementsPerOutputBlock> block_values{};
                for (size_t block = first_tail_block; block < blocks_per_row; ++block)
                {
                    unpackable->unpack_block_to_int8(source_row, block, block_values.data());

                    EmbedQ8Block &out =
                        out_blocks[output_row * blocks_per_row + block];
                    out.d = fp32_to_fp16(unpackable->get_block_scale(source_row, block));
                    out.m = fp32_to_fp16(unpackable->get_block_min(source_row, block));
                    std::memcpy(out.qs, block_values.data(), kElementsPerOutputBlock);
                }
            }

            const auto finished_at = std::chrono::steady_clock::now();
            const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                        finished_at - started_at)
                                        .count();

            if (log_style == EmbedQ8RepackLogStyle::FullTable)
            {
                LOG_DEBUG("[repackEmbeddingToQ8] Repacked "
                          << tensorTypeName(embed_table->native_type()) << " embedding "
                          << vocab_count << "×" << d_model << " → EmbedQ8 "
                          << (byte_size / (1024 * 1024)) << " MB (" << total_blocks
                          << " blocks) in " << elapsed_ms << " ms");
            }
            else
            {
                LOG_DEBUG("[repackEmbeddingToQ8] Repacked "
                          << tensorTypeName(embed_table->native_type()) << " embedding rows ["
                          << vocab_start << ", " << (vocab_start + vocab_count) << ") of "
                          << total_vocab << "×" << d_model << " → EmbedQ8 "
                          << (byte_size / (1024 * 1024)) << " MB (" << total_blocks
                          << " blocks) in " << elapsed_ms << " ms");
            }

            return result;
        }
    } // namespace detail

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
        if (embed_table == nullptr)
        {
            throw std::invalid_argument(
                "[repackEmbeddingToQ8] Embedding tensor must not be null");
        }
        return detail::repackEmbeddingRangeToQ8(
            embed_table,
            d_model,
            0,
            embed_table->rows(),
            detail::EmbedQ8RepackLogStyle::FullTable);
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
        return detail::repackEmbeddingRangeToQ8(
            embed_table,
            d_model,
            vocab_start,
            vocab_count,
            detail::EmbedQ8RepackLogStyle::VocabRange);
    }

} // namespace llaminar2
