/**
 * @file CPUFixedEpochObservations.h
 * @brief Integer observations returned only after a complete CPU head operation.
 *
 * This POD is transient telemetry, never recurrence state or a lifecycle owner.
 * The existing CPU kernel team reduces actual completed work and its original
 * observation multiplicity before one caller publishes collector increments.
 */
#pragma once
#include <cstdint>

namespace llaminar2
{
    /** @brief Completed head work; an empty participant returns all zero. */
    struct CPUFixedEpochObservations
    {
        std::uint64_t heads = 0; ///< Successful ordinary or retained head operations.
        std::uint64_t shared_heads = 0; ///< Completed immutable retained head operations.
        std::uint64_t projection_groups = 0; ///< Executed checkpoint projection groups.
        std::uint64_t grouped_rows = 0; ///< Actual rows in groups larger than one.
        std::uint64_t checkpoint_flushes = 0; ///< Executed fixed-eight checkpoint flushes.
        std::uint64_t shared_flushes = 0; ///< Flushes writing retained checkpoints.
        std::uint64_t raw_records = 0; ///< Copied initial plus appended live raw records.
        std::uint64_t candidate_rows = 0; ///< Completed retained candidate rows.

        /**
         * @brief Add disjoint completed work with exact integer arithmetic.
         * @param other Another worker's completed local observation.
         * @return This telemetry total; model arithmetic/state is never accessed.
         */
        CPUFixedEpochObservations &operator+=(const CPUFixedEpochObservations &other) noexcept
        {
            heads += other.heads; shared_heads += other.shared_heads;
            projection_groups += other.projection_groups; grouped_rows += other.grouped_rows;
            checkpoint_flushes += other.checkpoint_flushes; shared_flushes += other.shared_flushes;
            raw_records += other.raw_records; candidate_rows += other.candidate_rows;
            return *this;
        }
    };
#ifdef _OPENMP
#pragma omp declare reduction(cpu_gdn_epoch_observations : CPUFixedEpochObservations : omp_out += omp_in) initializer(omp_priv = CPUFixedEpochObservations{})
#endif
}
