/**
 * @file PrefixTerminalLogitsSlice.h
 * @brief Checked vocabulary ownership for a terminal-logit archive copy.
 *
 * A producer may publish a gathered vocabulary or a participant-local shard.
 * Archival must select the same logical token interval from either surface.
 * This pure geometry contract adds no allocation, transfer, or synchronization;
 * CPU copies, device copies and diagnostic hashes consume its identical slice.
 */
#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
    /** @brief Half-open vocabulary interval, expressed in token IDs, not bytes. */
    struct PrefixLogitsVocabularyRange
    {
        size_t first_token;
        size_t token_count;
    };

    /** @brief Immutable FP32 row slice whose logical and physical bounds are proven. */
    class PrefixTerminalLogitsSlice
    {
    public:
        /**
         * @brief Locate an archive's vocabulary interval within its actual producer.
         * @param source Vocabulary represented by row zero of the published tensor.
         * @param archive Vocabulary owned by this participant's immutable archive.
         * @param storage_row_bytes Physical extent of one source row, including padding.
         * @return Validated byte offset and length for the existing archive operation.
         * @throws std::invalid_argument For empty, overflowing or incompatible intervals.
         *
         * Subtract only after checking interval order. This avoids overflow at
         * either end and prevents a local shard from masquerading as a full row.
         */
        static PrefixTerminalLogitsSlice resolve(PrefixLogitsVocabularyRange source,
            PrefixLogitsVocabularyRange archive, size_t storage_row_bytes)
        {
            constexpr size_t max = std::numeric_limits<size_t>::max();
            if (source.token_count == 0 || archive.token_count == 0 ||
                source.token_count > max - source.first_token ||
                archive.token_count > max - archive.first_token ||
                storage_row_bytes % sizeof(float) != 0 ||
                source.token_count > storage_row_bytes / sizeof(float) ||
                archive.first_token < source.first_token)
                throw std::invalid_argument("Invalid prefix terminal-logit vocabulary/storage geometry");

            const size_t offset = archive.first_token - source.first_token;
            if (offset > source.token_count || archive.token_count > source.token_count - offset)
                throw std::invalid_argument("Prefix terminal-logit archive is outside its published vocabulary");

            // Containment in the physical FP32 row proves these products fit.
            return PrefixTerminalLogitsSlice(offset * sizeof(float), archive.token_count * sizeof(float));
        }

        /** @return Byte offset from row zero of the published source tensor. */
        [[nodiscard]] size_t byteOffset() const noexcept { return byte_offset_; }
        /** @return Exact byte count to copy or hash, excluding storage padding. */
        [[nodiscard]] size_t byteCount() const noexcept { return byte_count_; }

    private:
        /** @brief Construct only after resolve() has proved vocabulary containment. */
        PrefixTerminalLogitsSlice(size_t byte_offset, size_t byte_count) noexcept
            : byte_offset_(byte_offset), byte_count_(byte_count) {}

        size_t byte_offset_;
        size_t byte_count_;
    };
}
