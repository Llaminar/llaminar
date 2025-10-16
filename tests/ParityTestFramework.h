/**
 * @file ParityTestFramework.h
 * @brief Extensible parity test framework for comparing Llaminar pipeline stages with llama.cpp
 * @author David Sanftenberg
 *
 * This framework provides infrastructure for capturing and comparing intermediate tensor states
 * at various stages of the transformer pipeline between Llaminar and llama.cpp implementations.
 *
 * Key features:
 * - Stage-based snapshot capture (uses core PipelineStage enum from src/PipelineStages.h)
 * - Configurable comparison metrics (max_abs, rel_l2, mean_abs)
 * - Extensible to new model architectures
 * - MPI-aware for distributed execution
 */

#pragma once

#include "PipelineStages.h" // Core PipelineStage enum from src/
#include <string>
#include <vector>
#include <map>
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
        // (No duplicate definition - we import from src/PipelineStages.h)

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

            /**
             * @brief Save all snapshots to a directory as individual .npy files
             * @param output_dir Directory to save snapshots (will be created if it doesn't exist)
             * @return true if successful, false on error
             *
             * This saves all registered snapshots to {output_dir}/{stage_name}.npy files.
             * Used for generating Llaminar snapshots to compare with PyTorch.
             *
             * Example: If output_dir="llaminar_snapshots/token_0", saves:
             *   - EMBEDDING.npy
             *   - ATTENTION_NORM_layer0.npy
             *   - Q_PROJECTION_layer0.npy
             *   - etc.
             */
            bool saveToDirectory(const std::string &output_dir) const;

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
             * @param source Source identifier (default: "llaminar")
             */
            static void capture(
                PipelineStage stage,
                int layer_index,
                const float *data,
                int seq_len,
                int feature_dim,
                const std::string &source = "llaminar");

            /**
             * @brief Capture with custom stage name
             * @param stage_name Custom stage name
             * @param layer_index Layer index (-1 for non-layer stages)
             * @param data Pointer to tensor data
             * @param seq_len Sequence length
             * @param feature_dim Feature dimension
             * @param source Source identifier (default: "llaminar")
             */
            static void capture(
                const std::string &stage_name,
                int layer_index,
                const float *data,
                int seq_len,
                int feature_dim,
                const std::string &source = "llaminar");

            /**
             * @brief Enable/disable snapshot capture
             */
            static void set_enabled(bool enabled);

            static bool is_enabled();

        private:
            static bool enabled_;
        };

        /**
         * @brief Helper for per-token snapshot saving during incremental decode
         *
         * This utility enables capturing and saving snapshots for each token
         * during incremental decode, matching the PyTorch incremental snapshot format.
         *
         * Usage:
         *   IncrementalSnapshotHelper helper("llaminar_incremental_snapshots");
         *   for (int i = 0; i < num_tokens; ++i) {
         *       helper.beforeToken(i);
         *       pipeline->incrementalDecodeToken(token_ids[i], ...);
         *       helper.afterToken(i);
         *   }
         */
        class IncrementalSnapshotHelper
        {
        public:
            /**
             * @brief Constructor
             * @param output_base_dir Base directory for snapshot output (e.g., "llaminar_incremental_snapshots")
             */
            explicit IncrementalSnapshotHelper(const std::string &output_base_dir);

            /**
             * @brief Prepare for capturing a token
             * @param token_index Token index (0-based)
             *
             * Clears the snapshot registry and enables capture.
             */
            void beforeToken(int token_index);

            /**
             * @brief Finalize token capture and save snapshots
             * @param token_index Token index (0-based)
             *
             * Saves all captured snapshots to {output_base_dir}/token_{token_index}/
             * and clears the registry for the next token.
             *
             * @return true if successful, false on error
             */
            bool afterToken(int token_index);

            /**
             * @brief Get the output directory for a specific token
             */
            std::string getTokenDir(int token_index) const;

        private:
            std::string output_base_dir_;
        };

        /**
         * @brief Load PyTorch reference snapshots from NPZ file for comparison
         *
         * This class extends the parity framework to support loading reference
         * snapshots captured from PyTorch models. The NPZ file should contain
         * layer outputs with keys matching the naming convention.
         */
        class PytorchSnapshotLoader
        {
        public:
            /**
             * @brief Load PyTorch snapshots from NPZ file into registry
             *
             * @param filepath Path to NPZ file containing PyTorch captures
             * @param source_name Source identifier (default: "pytorch")
             * @return Number of snapshots loaded
             */
            static size_t load_from_npz(const std::string &filepath,
                                        const std::string &source_name = "pytorch");

            /**
             * @brief Parse layer information from snapshot key
             *
             * Extracts layer index and stage name from keys like:
             * - "layer_0_attn_out" -> layer=0, stage="attn_out"
             * - "layer_5_ffn_out" -> layer=5, stage="ffn_out"
             * - "embeddings" -> layer=-1, stage="embeddings"
             *
             * @param key Snapshot key from NPZ
             * @param layer_index Output: extracted layer index (-1 for non-layer)
             * @param stage_name Output: extracted stage name
             * @return true if parsing succeeded
             */
            static bool parse_key(const std::string &key,
                                  int &layer_index,
                                  std::string &stage_name);
        };

        /**
         * @brief Automated layer-by-layer comparison between implementations
         *
         * Compares corresponding snapshots from different sources (e.g., Llaminar vs PyTorch)
         * and generates a detailed report of differences.
         */
        class LayerByLayerComparator
        {
        public:
            /**
             * @brief Compare all matching snapshots between two sources
             *
             * @param source1 First source name (e.g., "llaminar")
             * @param source2 Second source name (e.g., "pytorch")
             * @param tolerance Comparison tolerances
             * @return Map of layer keys to comparison results
             */
            static std::map<std::string, ComparisonResult> compare_all(
                const std::string &source1,
                const std::string &source2,
                const ComparisonTolerance &tolerance = ComparisonTolerance());

            /**
             * @brief Find first layer with divergence above tolerance
             *
             * @param source1 First source name
             * @param source2 Second source name
             * @param tolerance Comparison tolerances
             * @return Key of first diverging layer (empty if all match)
             */
            static std::string find_first_divergence(
                const std::string &source1,
                const std::string &source2,
                const ComparisonTolerance &tolerance = ComparisonTolerance());

            /**
             * @brief Print comparison report to stdout
             *
             * @param results Map of comparison results
             * @param verbose Print all layers or only failures
             */
            static void print_report(
                const std::map<std::string, ComparisonResult> &results,
                bool verbose = false);
        };

        // Note: stage_to_string() and string_to_stage() utilities are provided
        // by the core PipelineStages.h header (inline functions in llaminar namespace)

    } // namespace parity
} // namespace llaminar