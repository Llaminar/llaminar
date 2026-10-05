/**
 * @file VocabularyGatherWorkspaceContract.h
 * @brief Exact arena requirement for lossless native vocabulary assembly.
 *
 * One-row logits already have the native rank-major layout. Multiple retained
 * rows need a disjoint transpose bank. Planning and the stage share this
 * contract; it contributes a typed requirement to the physical-memory owner
 * and never reserves bytes or maintains an allocation ledger itself.
 */
#pragma once

#include "execution/local_execution/device/WorkspaceDescriptor.h"
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
    /** @brief Stable scratch name shared only by event-ordered terminal graphs. */
    struct VocabularyGatherWorkspaceContract final
    {
        static constexpr const char *rankMajorBank = "native_vocabulary_rank_major";

        /**
         * @brief Price the exact full-vocabulary physical row bank.
         * @param rows Retained projection rows, never prompt or KV capacity.
         * @param vocabulary Complete vocabulary width.
         * @return No scratch for a singleton row; one disjoint FP32 bank otherwise.
         * @throws std::invalid_argument For empty geometry.
         * @throws std::overflow_error For an unrepresentable byte bank.
         */
        static WorkspaceRequirements requirements(std::size_t rows, std::size_t vocabulary)
        {
            if (rows == 0 || vocabulary == 0)
                throw std::invalid_argument("Vocabulary gather needs positive row/vocabulary geometry");
            if (vocabulary > std::numeric_limits<std::size_t>::max() / sizeof(float) / rows)
                throw std::overflow_error("Vocabulary gather bank overflows size_t");
            WorkspaceRequirements result;
            if (rows > 1)
                result.buffers.emplace_back(rankMajorBank, rows * vocabulary * sizeof(float), 256, true);
            return result;
        }
    };
}
