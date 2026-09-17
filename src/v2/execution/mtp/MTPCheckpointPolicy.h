/**
 * @file MTPCheckpointPolicy.h
 * @brief Shared lifetime policy for concurrently retained MTP checkpoints.
 *
 * Runtime checkpoint preallocation and preflight memory planning must use the
 * same concurrency and participant-ownership contract. A pipeline follower
 * still owns main-model rollback, but only the terminal stage owns a learned
 * predictor. Keeping both facts here prevents admission and materialization
 * from drifting without switching off speculative verification on followers.
 */

#pragma once

#include <cstddef>

namespace llaminar2
{
    /** @brief State owned by a participant, independently of the current draft depth. */
    enum class MTPStateRole
    {
        Disabled,          ///< No retained speculative cache or rollback state.
        MainModelFollower, ///< Own-layer verification/rollback, no learned predictor.
        PredictorOwner,    ///< Main-model state plus shifted predictor state.
    };

    /**
     * @brief Project retained capacity and canonical terminal ownership into one role.
     * @param retains_capacity Whether the model-lifetime lease retains MTP state.
     * @param owns_terminal_head Whether this participant executes the terminal head.
     * @return The same role consumed by live cache construction and the PMA BOM.
     *
     * Same-layer TP peers may all own a predictor; disjoint PP stages may not.
     * Neither device type nor MPI rank order determines this ownership.
     */
    [[nodiscard]] constexpr MTPStateRole resolveMTPStateRole(
        bool retains_capacity, bool owns_terminal_head) noexcept
    {
        return !retains_capacity ? MTPStateRole::Disabled
             : owns_terminal_head ? MTPStateRole::PredictorOwner
                                  : MTPStateRole::MainModelFollower;
    }

    /**
     * @brief Maximum complete live-state checkpoints retained concurrently.
     *
     * The scheduler may retain rollback, verifier-base, committed, and
     * diagnostic snapshots at the same time. Each set owns a main payload and
     * only those shifted-cache payloads that contain recurrent state.
     */
    inline constexpr size_t kMTPConcurrentLiveCheckpointSets = 4;
}
