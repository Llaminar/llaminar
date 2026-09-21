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
     * @brief Runtime ownership of an MTP transaction on one graph participant.
     *
     * This is deliberately distinct from @ref MTPStateRole.  A service may
     * retain an MTP-capable graph, weight, and KV envelope while an individual
     * request executes ordinary decode.  In that capacity-only state the
     * terminal participant remains the physical owner of the retained
     * sidecar, but it must not append a shifted-KV transaction or publish an
     * MTP terminal-hidden handoff during ordinary prefill.
     */
    enum class MTPRuntimeTransactionRole
    {
        Disabled,          ///< The request executes no speculative transaction.
        MainModelFollower, ///< The request executes MTP but this participant lacks the terminal predictor.
        PredictorOwner,    ///< The request executes MTP and this participant owns its shifted sidecar transaction.
    };

    /**
     * @brief Resolve transaction ownership without conflating it with retained capacity.
     * @param execution_enabled Whether this request's graph is allowed to execute MTP.
     * @param owns_terminal_head Whether this participant executes the terminal head.
     * @return The only runtime transaction role permitted for this participant.
     *
     * Capacity-only setup uses @ref resolveMTPStateRole instead.  Keeping the
     * two projections separate makes it impossible for a retained MTP envelope
     * to accidentally require a shifted-prefill transaction on an MTP-off
     * request.
     */
    [[nodiscard]] constexpr MTPRuntimeTransactionRole
    resolveMTPRuntimeTransactionRole(
        bool execution_enabled, bool owns_terminal_head) noexcept
    {
        return !execution_enabled
                   ? MTPRuntimeTransactionRole::Disabled
                   : owns_terminal_head
                         ? MTPRuntimeTransactionRole::PredictorOwner
                         : MTPRuntimeTransactionRole::MainModelFollower;
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
