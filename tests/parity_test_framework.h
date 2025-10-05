/**
 * @file parity_test_framework.h
 * @brief Extensible parity test framework for comparing Llaminar pipeline stages with llama.cpp
 * @author David Sanftenberg
 *
 * This framework provides infrastructure for capturing and comparing intermediate tensor states
 * at various stages of the transformer pipeline between Llaminar and llama.cpp implementations.
 *
 * Key features:
 * - Stage-based snapshot capture (uses core PipelineStage enum from src/pipeline_stages.h)
 * - Configurable comparison metrics (max_abs, rel_l2, mean_abs)
 * - Extensible to new model architectures
 * - MPI-aware for distributed execution
 */

#pragma once

#include "pipeline_stages.h" // Core PipelineStage enum from src/
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>
#include <cmath>
#include <algorithm>

namespace llaminar
{
    namespace parity
    {
        // Use core PipelineStage enum from llaminar::PipelineStage
        // (No duplicate definition - we import from src/pipeline_stages.h)

        /**
         * @brief Metadata for a captured tensor snapshot
         */
        struct SnapshotMetadata
        {
            std::string stage_name;
            PipelineStage stage;
            int layer_index; // -1 for non-layer stages (embedding, final norm, etc.)
            int seq_len;
            int feature_dim; // Hidden size, vocab size, etc.
            int64_t total_elements;
            std::string source; // "llaminar" or "llama.cpp"

            SnapshotMetadata()
                : stage_name(""), stage(PipelineStage::CUSTOM), layer_index(-1),
                  seq_len(0), feature_dim(0), total_elements(0), source("") {}
        };

        /**
         * @brief A captured tensor snapshot with metadata
         */
        struct TensorSnapshot
        {
            SnapshotMetadata metadata;
            std::vector<float> data;

            TensorSnapshot() = default;

            TensorSnapshot(const SnapshotMetadata &meta, const float *data_ptr, size_t count)
                : metadata(meta), data(data_ptr, data_ptr + count)
            {
                metadata.total_elements = static_cast<int64_t>(count);
            }
        };

        /**
         * @brief Comparison metrics between two tensors
         */
        struct ComparisonMetrics
        {
            float max_abs_diff;
            float mean_abs_diff;
            double rel_l2;
            size_t worst_index;
            float worst_expected;
            float worst_actual;
            bool passed;

            ComparisonMetrics()
                : max_abs_diff(0.0f), mean_abs_diff(0.0f), rel_l2(0.0),
                  worst_index(0), worst_expected(0.0f), worst_actual(0.0f), passed(false) {}
        };

        /**
         * @brief Tolerance specification for a comparison
         */
        struct ComparisonTolerance
        {
            float max_abs;
            double rel_l2;

            ComparisonTolerance(float abs_tol = 1e-3f, double l2_tol = 1e-4)
                : max_abs(abs_tol), rel_l2(l2_tol) {}
        };

        /**
         * @brief Result of comparing two snapshots
         */
        struct ComparisonResult
        {
            std::string stage_name;
            int layer_index;
            ComparisonMetrics metrics;
            ComparisonTolerance tolerance;
            std::string error_message;

            bool passed() const { return metrics.passed; }
        };

        /**
         * @brief Registry for storing snapshots during test execution
         */
        class SnapshotRegistry
        {
        public:
            static SnapshotRegistry &instance();

            void clear();

            void register_snapshot(const std::string &key, const TensorSnapshot &snapshot);

            bool has_snapshot(const std::string &key) const;

            bool get_snapshot(const std::string &key, TensorSnapshot &out_snapshot) const;

            std::vector<std::string> list_keys() const;

            // Convenience methods for common key patterns
            std::string make_key(const std::string &source, PipelineStage stage, int layer = -1) const;
            std::string make_key(const std::string &source, const std::string &stage_name, int layer = -1) const;

        private:
            SnapshotRegistry() = default;
            std::unordered_map<std::string, TensorSnapshot> snapshots_;
        };

        /**
         * @brief Comparator for tensor snapshots
         */
        class SnapshotComparator
        {
        public:
            /**
             * @brief Compare two tensor snapshots
             * @param expected Reference snapshot (e.g., from llama.cpp)
             * @param actual Test snapshot (e.g., from Llaminar)
             * @param tolerance Comparison tolerance
             * @return Comparison result with metrics
             */
            static ComparisonResult compare(
                const TensorSnapshot &expected,
                const TensorSnapshot &actual,
                const ComparisonTolerance &tolerance = ComparisonTolerance());

            /**
             * @brief Compute comparison metrics between two tensors
             */
            static ComparisonMetrics compute_metrics(
                const std::vector<float> &expected,
                const std::vector<float> &actual);

            /**
             * @brief Log top-k differences for debugging
             */
            static void log_top_differences(
                const std::vector<float> &expected,
                const std::vector<float> &actual,
                int cols,
                int top_k,
                const std::string &label);
        };

        /**
         * @brief Hook for capturing Llaminar pipeline snapshots
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
             */
            static void set_enabled(bool enabled);

            static bool is_enabled();

        private:
            static bool enabled_;
        };

        // Note: stage_to_string() and string_to_stage() utilities are provided
        // by the core pipeline_stages.h header (inline functions in llaminar namespace)

    } // namespace parity
} // namespace llaminar
