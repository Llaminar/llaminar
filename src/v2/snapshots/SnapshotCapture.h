/**
 * @file SnapshotCapture.h
 * @brief Snapshot capture for intermediate activations during inference
 *
 * Extracted from DeviceGraphOrchestrator.h (Phase 2 of DGO refactor).
 * Handles stage output capture, dequantization, and name-to-key mapping
 * for parity testing against PyTorch reference implementations.
 */

#pragma once

#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../execution/compute_stages/IComputeStage.h" // StageDumpInfo
#include "../tensors/BlockStructures.h"                 // Q8_1Block, Q16_1Block
#include "../tensors/FP16Utils.h"                       // fp16_to_fp32, bf16_to_fp32
#include "../utils/Logger.h"

namespace llaminar2
{

    /**
     * @brief Internal storage for a captured snapshot with shape metadata
     */
    struct StoredSnapshot
    {
        std::vector<float> data;
        size_t rows = 0;
        size_t cols = 0;
    };

    /**
     * @brief Immutable ownership handle for one captured diagnostic tensor.
     *
     * Snapshot callbacks replace a semantic key when a later graph execution
     * publishes a newer value.  A reader must therefore retain this handle,
     * rather than a raw pointer into the capture map, while it copies or
     * compares the data.  The pointed-to snapshot is immutable after
     * publication, so map replacement and clear can safely proceed without
     * invalidating a reader that already acquired a handle.
     */
    using StoredSnapshotHandle = std::shared_ptr<const StoredSnapshot>;

    /**
     * @brief Describe one ordered, logical prefill chunk retained for diagnostics.
     *
     * Captured GPU prefill executes a fixed physical bucket, while parity only
     * compares the real prefix of that bucket.  The graph executor projects
     * those live rows before publishing a snapshot.  This descriptor records
     * the stable callback namespace and its expected logical row count so the
     * snapshot owner can reconstruct one prompt-wide checkpoint after all
     * chunks have completed.
     */
    struct SnapshotChunkSequencePart
    {
        std::string context;   ///< Callback namespace for this chunk.
        size_t logical_rows{}; ///< Number of real rows represented by the namespace.
    };

    /**
     * @brief Outcome of joining context-scoped prefill snapshots into sequences.
     *
     * A checkpoint is sequence-shaped only when every chunk published the same
     * column geometry and each chunk's row count equals its real-token count.
     * Terminal-only values such as last-token logits intentionally remain as
     * the final chunk's ordinary snapshot rather than being falsely expanded.
     */
    struct SnapshotChunkSequenceAggregation
    {
        bool ok = false;                       ///< True when no sequence checkpoint was malformed.
        size_t aggregated_sequence_keys = 0;   ///< Number of prompt-wide semantic snapshots published.
        size_t terminal_or_nonsequence_keys = 0; ///< Context keys intentionally kept as final-value snapshots.
        std::string error;                     ///< Failure reason for a malformed sequence checkpoint.

        /** @brief Allow idiomatic success checks without conflating an empty capture with failure. */
        explicit operator bool() const noexcept { return ok; }
    };

    /**
     * @brief Captures and stores intermediate activation snapshots for parity testing
     *
     * This class owns the snapshot storage and routing logic previously inline
     * in DeviceGraphOrchestrator.h. It handles:
     * - Multi-output stage routing (QKV, GateUp, RoPE, FusedAttentionWo)
     * - FP32 extraction from quantized formats (Q8_1, Q16_1, BF16, FP16)
     * - Stage name → snapshot key conversion
     *
     * Thread safety: capture, clear, and handle-based reads are safe for
     * concurrent executor callbacks.  Handle-based reads are the production
     * diagnostic contract: the returned immutable object remains alive even
     * when a later graph replaces the key or clears the capture bank.  The
     * legacy raw-pointer accessor remains only for older test helpers and is
     * valid only while the caller has independently quiesced capture/reset.
     */
    class SnapshotCapture
    {
    public:
        using SnapshotMap = std::unordered_map<std::string, StoredSnapshotHandle>;

        SnapshotCapture() = default;
        SnapshotCapture(const SnapshotCapture &other)
        {
            std::lock_guard<std::mutex> lock(other.mutex_);
            snapshots_ = other.snapshots_;
        }
        SnapshotCapture &operator=(const SnapshotCapture &other)
        {
            if (this == &other)
                return *this;
            std::scoped_lock lock(mutex_, other.mutex_);
            snapshots_ = other.snapshots_;
            return *this;
        }
        SnapshotCapture(SnapshotCapture &&other) noexcept
        {
            std::lock_guard<std::mutex> lock(other.mutex_);
            snapshots_ = std::move(other.snapshots_);
        }
        SnapshotCapture &operator=(SnapshotCapture &&other) noexcept
        {
            if (this == &other)
                return *this;
            std::scoped_lock lock(mutex_, other.mutex_);
            snapshots_ = std::move(other.snapshots_);
            return *this;
        }

        /**
         * @brief Process a stage callback and store snapshot(s)
         *
         * Routes multi-output stages (QKV, GateUp, etc.) to separate snapshot keys.
         * Called by the executor's snapshot callback.
         */
        void captureStage(const std::string &name, const StageDumpInfo &dump);

        /**
         * @brief Replace bare semantic keys with complete ordered prefill sequences.
         *
         * During a segmented request, DeviceGraphOrchestrator publishes both a
         * context-qualified copy (for per-chunk diagnosis) and the historical
         * bare semantic key (which naturally contains only the most recent
         * chunk).  This diagnostic-only method validates the qualified row
         * geometry, concatenates every sequence-shaped checkpoint in request
         * order, and writes that complete value back under the bare key used by
         * existing parity comparison and CSV code.  It never touches live
         * device state or inserts work into the captured graph.
         *
         * @param chunks Ordered real-row descriptors from the prefill scheduler.
         * @return Aggregation counts, or a precise error when a checkpoint that
         *         began as sequence-shaped is incomplete or malformed.
         */
        SnapshotChunkSequenceAggregation aggregateSequentialChunkSnapshots(
            const std::vector<SnapshotChunkSequencePart> &chunks);

        /**
         * @brief Clear all stored snapshots
         */
        void clear()
        {
            std::lock_guard<std::mutex> lock(mutex_);
            snapshots_.clear();
        }

        /**
         * @brief Retrieve a snapshot by key with explicit lifetime ownership.
         * @param key Snapshot key (e.g., "layer0_Q_PROJECTION")
         * @return Immutable snapshot handle, or an empty handle when absent.
         *
         * The map lock protects only the lookup.  Returning a shared handle
         * then keeps this exact publication alive after the lock is released,
         * which lets parity copy it outside the executor callback timeline.
         */
        StoredSnapshotHandle getShared(const std::string &key) const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = snapshots_.find(key);
            return it != snapshots_.end() ? it->second : StoredSnapshotHandle{};
        }

        /**
         * @brief Retrieve a legacy raw view of one snapshot.
         * @param key Snapshot key (e.g., "layer0_Q_PROJECTION")
         * @return Pointer to the map-owned snapshot, or nullptr if absent.
         *
         * New production diagnostic code must use getShared().  This adapter
         * exists for older unit fixtures that deliberately read after capture
         * callbacks have quiesced; clear() or replacement may invalidate it.
         */
        const StoredSnapshot *get(const std::string &key) const
        {
            const StoredSnapshotHandle snapshot = getShared(key);
            return snapshot ? snapshot.get() : nullptr;
        }

        /**
         * @brief Return an immutable-handle copy of the complete capture bank.
         *
         * Returning the map by value is intentional.  Exposing a reference to
         * the mutable hash table would let a reader race a graph callback that
         * inserts, replaces, or clears snapshots.  Copying shared handles is
         * cheap and preserves each published tensor's lifetime.
         */
        SnapshotMap all() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return snapshots_;
        }

        /**
         * @brief Return the current number of semantic snapshot keys.
         *
         * This is a locked count for diagnostics that need to report an absent
         * key without taking a copy of the entire capture bank.
         */
        size_t size() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return snapshots_.size();
        }

        /**
         * @brief Get list of all snapshot keys
         */
        std::vector<std::string> keys() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            std::vector<std::string> result;
            result.reserve(snapshots_.size());
            for (const auto &p : snapshots_)
                result.push_back(p.first);
            return result;
        }

        /**
         * @brief Extract FP32 data from a StageDumpInfo output buffer
         *
         * Handles dequantization for Q8_1, Q16_1 variants, BF16, and FP16.
         */
        static std::vector<float> extractFp32FromOutput(const StageDumpInfo::OutputBuffer &out);

        /**
         * @brief Convert graph stage name to pipeline-style snapshot key
         *
         * Maps snake_case graph names (e.g., "layer0_q_proj") to
         * SCREAMING_CASE pipeline keys (e.g., "layer0_Q_PROJECTION").
         */
        static std::string convertStageNameToSnapshotKey(const std::string &stage_name);

        /**
         * @brief Return all snapshot keys a stage may publish.
         *
         * Some graph stages publish multiple semantic snapshots from one
         * callback, or select a key based on output names. Snapshot filters use
         * this conservative key set to decide whether a stage is relevant before
         * allocating graph-stable device copies.
         */
        static std::vector<std::string> possibleKeysForStageName(const std::string &stage_name);

        /**
         * @brief Return semantic snapshot keys for one concrete publication.
         *
         * Fused and policy-selected stages can retain a node name while changing
         * which value they own. This descriptor-aware form narrows the
         * conservative name map to outputs that the current graph actually
         * publishes.
         *
         * @param stage_name Graph-local producer name.
         * @param dump_info Stable named output descriptors for that producer.
         */
        static std::vector<std::string> possibleKeysForStage(
            const std::string &stage_name,
            const StageDumpInfo &dump_info);

    private:
        /**
         * @brief Publish one immutable FP32 snapshot while the capture lock is held.
         *
         * Every write allocates a new immutable object instead of modifying an
         * existing vector in place.  Readers holding an older handle therefore
         * observe a complete old publication, never a vector being reallocated
         * by a later callback.
         *
         * @param key Semantic diagnostic key.
         * @param data Fully materialized FP32 values.
         * @param rows Logical row count reported by the producing stage.
         * @param cols Logical column count reported by the producing stage.
         */
        void storeSnapshot(
            const std::string &key,
            std::vector<float> data,
            size_t rows,
            size_t cols);

        /**
         * @brief Extract and publish one stage output while the capture lock is held.
         *
         * This helper keeps dtype conversion in one place and delegates the
         * replacement/lifetime rule to storeSnapshot().
         */
        void storeOutput(const std::string &key, const StageDumpInfo::OutputBuffer &out);

        mutable std::mutex mutex_;
        SnapshotMap snapshots_;
    };

} // namespace llaminar2
