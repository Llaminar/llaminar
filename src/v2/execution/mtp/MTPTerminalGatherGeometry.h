/**
 * @file MTPTerminalGatherGeometry.h
 * @brief Shared logical geometry for retained full-vocabulary MTP output storage.
 *
 * The schema and the physical-memory BOM use this value, irrespective of the
 * transport used to assemble vocabulary shards. It owns no allocations and
 * keeps no ledger; PhysicalMemoryAuthority remains the physical byte authority.
 */
#pragma once

#include "execution/config/RuntimeConfig.h"

#include <cstddef>
#include <stdexcept>

namespace llaminar2
{
    /** @brief Validated gathered-logits shape shared by planning and graph setup. */
    class MTPTerminalGatherGeometry
    {
    public:
        /**
         * @brief Resolve the output bank for the retained condition/verifier family.
         * @param layout Participant-local terminal projection ownership.
         * @param target_rows Maximum retained target query rows, not current live M.
         * @param vocabulary Complete model vocabulary, not the local shard width.
         * @return Full rows for sharded output, or the schema's 1x1 placeholder.
         * @throws std::invalid_argument for empty geometry or an unknown layout.
         *
         * LocalTP and GlobalTP both assemble full distributions. A mirrored
         * projection already writes its full distribution in MTP_LOGITS and
         * needs no second bank. Request enablement does not resize this union.
         */
        static MTPTerminalGatherGeometry resolve(
            MTPTerminalLogitsLayout layout,
            std::size_t target_rows,
            std::size_t vocabulary)
        {
            if (target_rows == 0 || vocabulary == 0)
                throw std::invalid_argument("MTP terminal gather geometry must be positive");
            switch (layout)
            {
            case MTPTerminalLogitsLayout::FullVocabularyPerParticipant:
                return {1u, 1u};
            case MTPTerminalLogitsLayout::VocabularyShardPerParticipant:
                return {target_rows, vocabulary};
            }
            throw std::invalid_argument("Unknown MTP terminal logits layout");
        }

        /** @return Exact physical row extent of the schema's retained bank. */
        std::size_t rows() const noexcept { return rows_; }
        /** @return Exact physical column extent of the schema's retained bank. */
        std::size_t columns() const noexcept { return columns_; }

    private:
        /** @brief Construct only a shape validated by the public resolver. */
        MTPTerminalGatherGeometry(std::size_t rows, std::size_t columns)
            : rows_(rows), columns_(columns) {}
        std::size_t rows_;
        std::size_t columns_;
    };
}
