/**
 * @file MTPCheckpointPolicy.h
 * @brief Shared lifetime policy for concurrently retained MTP checkpoints.
 *
 * Runtime checkpoint preallocation and preflight memory planning must use the
 * same concurrency contract. Keeping the cardinality here prevents the two
 * systems from drifting when scheduler checkpoint lifetimes evolve.
 */

#pragma once

#include <cstddef>

namespace llaminar2
{
    /**
     * @brief Maximum complete live-state checkpoints retained concurrently.
     *
     * The scheduler may retain rollback, verifier-base, committed, and
     * diagnostic snapshots at the same time. Each set owns a main payload and
     * only those shifted-cache payloads that contain recurrent state.
     */
    inline constexpr size_t kMTPConcurrentLiveCheckpointSets = 4;
}
