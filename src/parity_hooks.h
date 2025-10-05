/**
 * @file parity_hooks.h
 * @brief Parity testing snapshot capture hooks (production-safe interface)
 * @author David Sanftenberg
 *
 * This header provides the interface for capturing pipeline snapshots for parity
 * testing. The actual implementation lives in tests/parity_test_framework.cpp and
 * is conditionally compiled/linked.
 *
 * Design:
 * - Interface is always available (no #ifdef needed)
 * - Default implementation is no-op (zero overhead)
 * - Tests provide concrete implementation by linking parity framework
 * - Environment variable controls enable/disable at runtime
 */

#pragma once

#include "pipeline_stages.h"
#include <string>

namespace llaminar
{
    namespace parity
    {
        /**
         * @brief Hook interface for capturing Llaminar pipeline snapshots
         *
         * This class provides static methods for snapshot capture. When linked with
         * the parity test framework (tests/parity_test_framework.cpp), it captures
         * snapshots to a registry for comparison. When not linked (production builds),
         * it's a no-op.
         */
        class LlaminarSnapshotHook
        {
        public:
            /**
             * @brief Capture a snapshot from Llaminar pipeline
             * @param stage Pipeline stage
             * @param layer_index Layer index (-1 for non-layer stages)
             * @param data Pointer to tensor data
             * @param seq_len Sequence length
             * @param feature_dim Feature dimension
             *
             * This is a no-op unless:
             * 1. Linked with parity test framework
             * 2. Enabled via set_enabled() or LLAMINAR_PARITY_CAPTURE env var
             */
            static void capture(
                PipelineStage stage,
                int layer_index,
                const float *data,
                int seq_len,
                int feature_dim);

            /**
             * @brief Capture with custom stage name
             */
            static void capture(
                const std::string &stage_name,
                int layer_index,
                const float *data,
                int seq_len,
                int feature_dim);

            /**
             * @brief Enable/disable snapshot capture
             * @param enabled true to enable capture, false to disable
             */
            static void set_enabled(bool enabled);

            /**
             * @brief Check if snapshot capture is enabled
             * @return true if enabled, false otherwise
             */
            static bool is_enabled();

        private:
            static bool enabled_; ///< Enable flag (defaults to false)
        };

    } // namespace parity
} // namespace llaminar
