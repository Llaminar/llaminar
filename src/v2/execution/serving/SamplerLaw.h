/**
 * @file SamplerLaw.h
 * @brief Admitted per-row sampler law and per-slot penalty-state BOM obligation (design §6.1).
 *
 * Temperature, top-k/top-p, penalties and seeds are runtime request data read
 * through admitted per-row tables on CPU, CUDA and ROCm; different request
 * laws must not trigger recapture or singleton serialization. The law row is
 * immutable per request after admission. Per-request penalty state is a
 * separate physical-memory obligation: a vocabulary-sized INT32 count table
 * costs 4 x vocabulary_size bytes per slot before sequence-history, DRY,
 * rollback or sampler workspaces — this header is the single estimator input
 * for that obligation (the charge goes through the same BOM authority as
 * every other allocation).
 */

#pragma once

#include "execution/config/ServingCapacity.h"
#include "execution/serving/RequestIntrinsicCoordinates.h"
#include "planning/PhysicalMemoryBOM.h"

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
    /** @brief Stochastic or greedy selection law (admitted, immutable). */
    struct SamplerLaw
    {
        /** @brief Greedy requests select the maximum logit (no draw). */
        bool greedy = false;
        float temperature = 1.0f;
        int top_k = 0;      ///< 0 = disabled.
        float top_p = 1.0f; ///< 1.0 = disabled.
        float repetition_penalty = 1.0f;
        /** @brief Resolved nonzero seed (immutable; zero resolved once at admission). */
        DrawCoordinates seed_coordinates{}; // purpose Sampling, logical position 0 at law admission
    };

    /**
     * @brief Immutable per-row law table (design §6.1).
     *
     * Capacity-bound by the resolved serving capacity (one law per accepted
     * response lease); slot/batch identity is an index, never part of the law.
     */
    struct SamplerLawTable
    {
        static constexpr std::size_t kMaxRows = static_cast<std::size_t>(kServingResponsePoolCeiling);

        /** @brief Table bounded by the resolved serving capacity (R). */
        explicit SamplerLawTable(std::size_t capacity = kMaxRows)
        {
            if (capacity == 0 || capacity > kMaxRows)
                throw std::logic_error("Sampler table capacity must be within the ceiling");
            capacity_ = capacity;
        }

        /** @brief Read one law by accepted row; throws for out-of-range. */
        [[nodiscard]] const SamplerLaw &at(std::size_t row) const
        {
            if (row >= count_)
                throw std::logic_error(
                    "Sampler law row is not admitted (bounds are the admitted capacity)");
            return laws_[row];
        }

        /** @brief Admit exactly one immutable law (throws when bounded capacity reached). */
        void admit(const SamplerLaw &law)
        {
            if (count_ >= capacity_)
                throw std::logic_error("Sampler law table is at its admitted capacity");
            laws_[count_++] = law;
        }

        [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

        [[nodiscard]] std::size_t size() const noexcept { return count_; }

    private:
        std::array<SamplerLaw, kMaxRows> laws_{};
        std::size_t count_ = 0;
        std::size_t capacity_ = kMaxRows;
    };

    /**
     * @brief Per-slot penalty-state hosting obligation (design §6.1).
     *
     * @param slots Accepted slots (capacity-one: 1 today; typed for the supported range).
     * @param vocabulary_size Admitted vocabulary token count.
     * @return Exact bytes: slots * 4 * vocabulary_size (INT32 count table), before
     *         sequence-history/DRY/rollback/sampler workspaces.
     * @throws std::invalid_argument For a zero vocabulary.
     * @throws std::overflow_error When the obligation cannot be represented.
     */
    [[nodiscard]] inline std::uint64_t penaltyStateDemand(std::uint64_t slots,
                                                          std::uint64_t vocabulary_size)
    {
        if (slots == 0)
            throw std::invalid_argument("Penalty state slots must be positive");
        if (vocabulary_size == 0)
            throw std::invalid_argument("Penalty state vocabulary must be positive");
        constexpr std::uint64_t kBytesPerToken = 4; // INT32 count table
        if (vocabulary_size > std::numeric_limits<std::uint64_t>::max() / kBytesPerToken)
            throw std::overflow_error("Penalty state demand overflows uint64");
        const auto per_slot = vocabulary_size * kBytesPerToken; // no wrap
        if (slots > std::numeric_limits<std::uint64_t>::max() / per_slot)
            throw std::overflow_error("Penalty state demand overflows uint64");
        return slots * per_slot;
    }

    /**
     * @brief Charge the per-slot penalty state on a physical-memory BOM builder.
     *
     * Uses the standard KVCache-style owner authority (KVCache family is the
     * slot-addressed host state; kept as an exact typed charge, never a
     * second ledger).
     */
    inline void addPenaltyStateDemand(PhysicalMemoryBOMBuilder &builder, std::uint64_t slots,
                                      std::uint64_t vocabulary_size)
    {
        // Owner is Intentional: penalty counts are per-slot request state, not
        // model weights — the closest existing storage owner is the
        // slot-addressed host state family. The typed estimator is the only
        // source of the number.
        builder.add(PhysicalMemoryOwner::SequenceMetadata, penaltyStateDemand(slots, vocabulary_size));
    }
} // namespace llaminar2
