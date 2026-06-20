/**
 * @file ExpertWeightPayloadProvider.h
 * @brief Provides serialized expert weight payloads for runtime GPU expert arrivals.
 *
 * Replaces the implicit fallback to raw GGUF host tensor data in
 * MoEExpertWeightService::registerAndPrepareNewExpertsGPU(). After graph
 * materialization and GEMM preparation, expert weights are serialized into
 * transferable blobs and registered here. When a GPU expert arrives through
 * dynamic rebalancing without MPI-received blobs, the provider supplies the
 * serialized payload instead of requiring retained raw host tensor data.
 *
 * Ownership: model-context or orchestration-owned (not stage-local).
 * Thread safety: all methods are thread-safe (internal mutex).
 */

#pragma once

#include "ExpertWeightTransfer.h" // ExpertWeightBlobs

#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace llaminar2
{

    /// Tracks the preparation state of an individual expert on a specific device.
    struct ExpertPreparationState
    {
        bool prepared = false;    ///< Has a GEMM engine been created for this expert?
        bool transferred = false; ///< Has a serialized transferable blob been created?
        bool raw_released = false; ///< Has the source raw host data been released?
    };

    /**
     * @brief Provides serialized expert weight payloads for GPU expert preparation.
     *
     * After initial graph-build GEMM preparation, expert weights are serialized
     * and stored here as transferable blobs. This enables:
      * - Runtime GPU expert arrival preparation without retaining raw host tensor data
     * - Dynamic MoE rebalancing with pre-materialized payloads
     * - Explicit tracking of which experts are prepared/transferred/releasable
     */
    class ExpertWeightPayloadProvider
    {
    public:
        ExpertWeightPayloadProvider() = default;
        ~ExpertWeightPayloadProvider() = default;

        // Non-copyable, movable
        ExpertWeightPayloadProvider(const ExpertWeightPayloadProvider &) = delete;
        ExpertWeightPayloadProvider &operator=(const ExpertWeightPayloadProvider &) = delete;
        ExpertWeightPayloadProvider(ExpertWeightPayloadProvider &&) = default;
        ExpertWeightPayloadProvider &operator=(ExpertWeightPayloadProvider &&) = default;

        // ── Payload registration ─────────────────────────────────────

        /// Register a serialized expert weight payload for a given layer and expert.
        /// Overwrites any existing payload for the same (layer, expert).
        void registerPayload(int layer, int expert_id, ExpertWeightBlobs blobs);

        /// Register payloads for multiple experts in a layer at once.
        void registerPayloads(int layer, std::unordered_map<int, ExpertWeightBlobs> blobs);

        // ── Payload queries ──────────────────────────────────────────

        /// Check if a serialized payload exists for (layer, expert).
        bool hasPayload(int layer, int expert_id) const;

        /// Get the serialized payload for (layer, expert).
        /// Returns nullopt if no payload is registered.
        std::optional<ExpertWeightBlobs> payloadFor(int layer, int expert_id) const;

        /// Get a read-only reference to the payload for (layer, expert).
        /// Returns nullptr if no payload is registered.
        const ExpertWeightBlobs *payloadPtr(int layer, int expert_id) const;

        /// Build a received-weights map for a single layer (for registerAndPrepareNewExperts).
        std::unordered_map<int, ExpertWeightBlobs> payloadsForLayer(int layer) const;

        // ── Preparation state tracking ───────────────────────────────

        /// Mark an expert as having a prepared GEMM engine on a device.
        void markExpertPrepared(int layer, int expert_id);

        /// Mark an expert as having a serialized transferable blob.
        void markExpertTransferred(int layer, int expert_id);

        /// Mark an expert's raw host data as released.
        void markExpertRawReleased(int layer, int expert_id);

        /// Check if an expert has a prepared GEMM engine.
        bool isExpertPrepared(int layer, int expert_id) const;

        /// Check if an expert has a serialized transferable blob.
        bool isExpertTransferred(int layer, int expert_id) const;

        /// Check if an expert's raw host data is still needed.
        /// Returns true if the expert is NOT yet prepared AND NOT yet transferred.
        bool isRawDataRequired(int layer, int expert_id) const;

        /// Check if ALL experts for a given layer have been prepared or transferred.
        /// Useful for determining when raw host data for the entire layer can be released.
        bool allExpertsPreparedOrTransferred(int layer, int num_experts) const;

        // ── Cleanup ──────────────────────────────────────────────────

        /// Remove payload for a specific expert (e.g., after it departs during rebalancing).
        void removePayload(int layer, int expert_id);

        /// Remove all payloads for a layer.
        void removeLayer(int layer);

        /// Clear all payloads and preparation state.
        void clear();

        /// Returns total number of stored payloads across all layers.
        size_t totalPayloadCount() const;

        /// Returns total bytes stored across all payloads.
        size_t totalPayloadBytes() const;

    private:
        /// Key: layer index → (expert_id → blobs)
        using LayerPayloads = std::unordered_map<int, ExpertWeightBlobs>;

        /// Key: layer index → (expert_id → preparation state)
        using LayerStates = std::unordered_map<int, ExpertPreparationState>;

        ExpertPreparationState &getOrCreateState(int layer, int expert_id);
        const ExpertPreparationState *getState(int layer, int expert_id) const;

        mutable std::mutex mutex_;
        std::unordered_map<int, LayerPayloads> payloads_;
        std::unordered_map<int, LayerStates> states_;
    };

} // namespace llaminar2
