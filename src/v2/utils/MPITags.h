/**
 * @file MPITags.h
 * @brief MPI tag constants for pipeline parallelism
 *
 * Defines tag namespaces to prevent message collision between different
 * communication patterns (activation forwarding, KV cache, control messages).
 *
 * @author David Sanftenberg
 */

#pragma once

namespace llaminar2
{
    namespace mpi_tags
    {

        // ============================================================================
        // Base Tag Ranges
        // ============================================================================
        // Each category gets a 1000-tag range to allow per-layer addressing

        constexpr int ACTIVATION_FORWARD = 1000;  // Forward pass activations (1000-1999)
        constexpr int ACTIVATION_BACKWARD = 2000; // Future: backward pass (2000-2999)
        constexpr int KV_CACHE = 3000;            // KV cache transfer (3000-3999)
        constexpr int CONTROL = 4000;             // Control messages (4000-4999)
        constexpr int TENSOR_DATA = 5000;         // Generic tensor data (5000-5999)
        constexpr int SYNC = 6000;                // Synchronization signals (6000-6999)

        // ============================================================================
        // Per-Layer Tag Generation
        // ============================================================================

        /**
         * @brief Generate tag for forward pass activation at a specific layer
         * @param layer Layer index (0-based)
         * @return Tag in range [ACTIVATION_FORWARD, ACTIVATION_FORWARD + 999]
         */
        inline constexpr int forwardTag(int layer)
        {
            return ACTIVATION_FORWARD + layer;
        }

        /**
         * @brief Generate tag for backward pass activation at a specific layer
         * @param layer Layer index (0-based)
         * @return Tag in range [ACTIVATION_BACKWARD, ACTIVATION_BACKWARD + 999]
         */
        inline constexpr int backwardTag(int layer)
        {
            return ACTIVATION_BACKWARD + layer;
        }

        /**
         * @brief Generate tag for KV cache transfer at a specific layer
         * @param layer Layer index (0-based)
         * @return Tag in range [KV_CACHE, KV_CACHE + 999]
         */
        inline constexpr int kvCacheTag(int layer)
        {
            return KV_CACHE + layer;
        }

        /**
         * @brief Generate tag for KV cache K values at a specific layer
         * @param layer Layer index (0-based)
         * @return Tag for K cache
         */
        inline constexpr int kvCacheKeyTag(int layer)
        {
            return KV_CACHE + layer * 2;
        }

        /**
         * @brief Generate tag for KV cache V values at a specific layer
         * @param layer Layer index (0-based)
         * @return Tag for V cache
         */
        inline constexpr int kvCacheValueTag(int layer)
        {
            return KV_CACHE + layer * 2 + 1;
        }

        /**
         * @brief Generate tag for generic tensor data with ID
         * @param tensor_id Unique tensor identifier
         * @return Tag in range [TENSOR_DATA, TENSOR_DATA + 999]
         */
        inline constexpr int tensorTag(int tensor_id)
        {
            return TENSOR_DATA + tensor_id;
        }

        // ============================================================================
        // Control Message Subtypes
        // ============================================================================

        namespace control
        {
            constexpr int READY = CONTROL + 0;       // Rank ready signal
            constexpr int START_BATCH = CONTROL + 1; // Start batch processing
            constexpr int END_BATCH = CONTROL + 2;   // End batch processing
            constexpr int SHUTDOWN = CONTROL + 3;    // Graceful shutdown
            constexpr int ERROR = CONTROL + 4;       // Error notification
            constexpr int HEARTBEAT = CONTROL + 5;   // Health check
        }

        // ============================================================================
        // MoE Expert Weight Transfer Tags
        // ============================================================================
        // Two-phase protocol: size exchange (blocking) then data exchange (non-blocking).
        // Size tags:  10000 + layer * 256 + expert_id          → max ~26k
        // Data tags: 100000 + layer * 256 * 3 + expert * 3 + proj → max ~149k

        constexpr int WEIGHT_TRANSFER_SIZE = 10000;  // [10000-25999] size exchange
        constexpr int WEIGHT_TRANSFER_DATA = 100000; // [100000-149999] data exchange
        constexpr int MAX_EXPERTS_PER_TAG = 256;     // Maximum experts for tag computation

        /**
         * @brief Tag for weight transfer SIZE exchange (Phase A).
         * @param layer      MoE layer index (0-based)
         * @param expert_id  Expert index (0-based, < MAX_EXPERTS_PER_TAG)
         * @return Unique tag in [WEIGHT_TRANSFER_SIZE, WEIGHT_TRANSFER_SIZE + 16383]
         */
        inline constexpr int weightTransferSizeTag(int layer, int expert_id)
        {
            return WEIGHT_TRANSFER_SIZE + layer * MAX_EXPERTS_PER_TAG + expert_id;
        }

        /**
         * @brief Tag for weight transfer DATA exchange (Phase B).
         * @param layer      MoE layer index (0-based)
         * @param expert_id  Expert index (0-based, < MAX_EXPERTS_PER_TAG)
         * @param proj       Projection index: 0=gate, 1=up, 2=down
         * @return Unique tag in [WEIGHT_TRANSFER_DATA, WEIGHT_TRANSFER_DATA + 49151]
         */
        inline constexpr int weightTransferDataTag(int layer, int expert_id, int proj)
        {
            return WEIGHT_TRANSFER_DATA + layer * MAX_EXPERTS_PER_TAG * 3 + expert_id * 3 + proj;
        }

        // ============================================================================
        // Wildcard Constants
        // ============================================================================

        constexpr int ANY_SOURCE = -1; // MPI_ANY_SOURCE equivalent for convenience
        constexpr int ANY_TAG = -1;    // MPI_ANY_TAG equivalent for convenience

    } // namespace mpi_tags
} // namespace llaminar2
