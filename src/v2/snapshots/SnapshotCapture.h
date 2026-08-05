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
     * @brief Captures and stores intermediate activation snapshots for parity testing
     *
     * This class owns the snapshot storage and routing logic previously inline
     * in DeviceGraphOrchestrator.h. It handles:
     * - Multi-output stage routing (QKV, GateUp, RoPE, FusedAttentionWo)
     * - FP32 extraction from quantized formats (Q8_1, Q16_1, BF16, FP16)
     * - Stage name → snapshot key conversion
     *
     * Thread safety: capture/clear are safe for concurrent executor callbacks.
     * Accessors returning raw pointers/references should be used after capture
     * callbacks have quiesced.
     */
    class SnapshotCapture
    {
    public:
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
         * @brief Clear all stored snapshots
         */
        void clear()
        {
            std::lock_guard<std::mutex> lock(mutex_);
            snapshots_.clear();
        }

        /**
         * @brief Retrieve a snapshot by key
         * @param key Snapshot key (e.g., "layer0_Q_PROJECTION")
         * @return Pointer to StoredSnapshot, or nullptr if not found
         */
        const StoredSnapshot *get(const std::string &key) const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = snapshots_.find(key);
            return it != snapshots_.end() ? &it->second : nullptr;
        }

        /**
         * @brief Get all stored snapshots
         */
        const std::unordered_map<std::string, StoredSnapshot> &all() const { return snapshots_; }

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
        void storeOutput(const std::string &key, const StageDumpInfo::OutputBuffer &out);

        mutable std::mutex mutex_;
        std::unordered_map<std::string, StoredSnapshot> snapshots_;
    };

} // namespace llaminar2
