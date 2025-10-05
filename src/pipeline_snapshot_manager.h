/**
 * @file pipeline_snapshot_manager.h
 * @brief Debug-only layer snapshot capture for parity testing
 * @author David Sanftenberg
 *
 * This module provides zero-overhead snapshot capture in release mode via
 * conditional compilation. In debug builds, it integrates with the parity
 * test framework to enable systematic layer-by-layer comparison.
 *
 * Design principles:
 * - Zero runtime overhead in release builds (compiled out)
 * - Environment-controlled activation (LLAMINAR_PARITY_CAPTURE=1)
 * - Automatic registration through AbstractPipeline
 * - NPZ export for Python comparison tools
 */

#pragma once

#include "pipeline_stages.h"
#include <string>
#include <memory>

namespace llaminar
{
    /**
     * @brief Snapshot manager for debug builds only
     *
     * In debug builds, this class provides snapshot capture functionality.
     * In release builds, all methods become no-ops that are optimized away.
     *
     * Usage:
     *   PipelineSnapshotManager::instance().capture(
     *       PipelineStage::ATTENTION_OUTPUT,
     *       layer_idx,
     *       data,
     *       seq_len,
     *       feature_dim
     *   );
     */
    class PipelineSnapshotManager
    {
    public:
        /**
         * @brief Get singleton instance
         *
         * In release builds, returns a stub instance that does nothing.
         */
        static PipelineSnapshotManager &instance();

        /**
         * @brief Capture a layer snapshot
         *
         * @param stage Pipeline stage identifier
         * @param layer_index Layer index (-1 for non-layer stages)
         * @param data Pointer to tensor data (float array)
         * @param seq_len Sequence length dimension
         * @param feature_dim Feature dimension
         * @param source Source identifier (e.g., "llaminar", "pytorch")
         *
         * In release builds, this is a no-op that gets optimized away.
         * In debug builds, captures if LLAMINAR_PARITY_CAPTURE=1 is set.
         */
        void capture(
            PipelineStage stage,
            int layer_index,
            const float *data,
            int seq_len,
            int feature_dim,
            const std::string &source = "llaminar");

        /**
         * @brief Check if capture is enabled
         *
         * @return true if LLAMINAR_PARITY_CAPTURE=1 in debug builds, false otherwise
         */
        bool isEnabled() const;

        /**
         * @brief Enable/disable capture at runtime
         *
         * @param enabled Whether to enable snapshot capture
         *
         * Only has effect in debug builds.
         */
        void setEnabled(bool enabled);

        /**
         * @brief Export all captured snapshots to NPZ file
         *
         * @param filepath Path to output NPZ file
         * @return Number of snapshots exported
         *
         * In release builds, always returns 0.
         */
        size_t exportToNPZ(const std::string &filepath) const;

        /**
         * @brief Clear all captured snapshots
         */
        void clear();

        /**
         * @brief Get number of captured snapshots
         *
         * @return Snapshot count (0 in release builds)
         */
        size_t count() const;

    private:
        PipelineSnapshotManager();
        ~PipelineSnapshotManager() = default;

        // Non-copyable
        PipelineSnapshotManager(const PipelineSnapshotManager &) = delete;
        PipelineSnapshotManager &operator=(const PipelineSnapshotManager &) = delete;

#ifdef NDEBUG
        // Release build: no state needed
#else
        // Debug build: actual implementation
        struct Impl;
        std::unique_ptr<Impl> impl_;
#endif
    };

    /**
     * @brief RAII guard for automatic snapshot capture scope
     *
     * Useful for capturing specific pipeline regions:
     *
     * {
     *     SnapshotScope scope(true);  // Enable for this scope
     *     // ... pipeline operations ...
     * }  // Automatically restores previous state
     */
    class SnapshotScope
    {
    public:
        explicit SnapshotScope(bool enable);
        ~SnapshotScope();

    private:
#ifndef NDEBUG
        bool previous_state_;
#endif
    };

} // namespace llaminar
