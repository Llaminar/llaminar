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
#include <cstdint>
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

    /**
     * @brief Immutable selected-row identity authenticated by the owning request.
     *
     * This is a copy-plan value, not a second publication ledger. The current
     * session authority must authenticate it immediately before consuming bytes.
     */
    class PrefixLogitsRowSelection
    {
    public:
        /**
         * @brief Select one live row from an explicitly bounded producer.
         * @param row Accepted physical row, including any request-row offset.
         * @param rows Physical rows produced by this exact verifier invocation.
         * @param session_epoch Request epoch owning the producer and destination.
         * @throws std::invalid_argument For absent or out-of-range rows.
         */
        PrefixLogitsRowSelection(size_t row, size_t rows, uint64_t session_epoch)
            : row_(row), rows_(rows), session_epoch_(session_epoch)
        {
            if (rows == 0 || row >= rows)
                throw std::invalid_argument("Terminal-logit selection is outside its producer rows");
        }
        /** @return Selected physical producer row. */
        [[nodiscard]] size_t row() const noexcept { return row_; }
        /** @return Exact physical row extent authenticated at selection. */
        [[nodiscard]] size_t rows() const noexcept { return rows_; }
        /**
         * @brief Reject a selected row retained across request reset or restore.
         * @param current_epoch Current canonical request/session authority.
         * @throws std::invalid_argument If the producer belongs to another session.
         */
        void requireCurrent(uint64_t current_epoch) const
        {
            if (current_epoch != session_epoch_)
                throw std::invalid_argument("Terminal-logit selection belongs to a stale session");
        }
    private:
        size_t row_;
        size_t rows_;
        uint64_t session_epoch_;
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

        /**
         * @brief Resolve an accepted row and vocabulary interval together.
         * @param source Vocabulary represented by every physical producer row.
         * @param archive Exact vocabulary interval to publish to canonical storage.
         * @param storage_row_bytes Physical row stride, including storage padding.
         * @param selection Typed accepted row and owning request epoch.
         * @param current_epoch Current canonical request/session authority.
         * @return Bounded byte slice from the beginning of the complete tensor.
         * @throws std::invalid_argument For stale sessions or overflowing storage.
         */
        static PrefixTerminalLogitsSlice resolveSelectedRow(
            PrefixLogitsVocabularyRange source, PrefixLogitsVocabularyRange archive,
            size_t storage_row_bytes, const PrefixLogitsRowSelection &selection,
            uint64_t current_epoch)
        {
            selection.requireCurrent(current_epoch);
            const auto row = resolve(source, archive, storage_row_bytes);
            if (storage_row_bytes > std::numeric_limits<size_t>::max() / selection.rows())
                throw std::invalid_argument("Terminal-logit producer row extent overflows storage");
            return PrefixTerminalLogitsSlice(
                selection.row() * storage_row_bytes + row.byteOffset(), row.byteCount());
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
