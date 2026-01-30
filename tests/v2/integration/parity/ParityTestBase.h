/**
 * @file ParityTestBase.h
 * @brief Base class and utilities for PyTorch parity tests
 *
 * Provides standardized infrastructure for comparing Llaminar inference
 * against PyTorch ground truth. All parity tests should inherit from
 * ParityTestBase to ensure consistent:
 *
 * - Metric calculations (cosine similarity, KL divergence, Top-K overlap)
 * - Table visualization of layer-by-layer results
 * - Pass/fail assertions with configurable thresholds
 * - Snapshot loading and regeneration
 *
 * MPI Support:
 *   For tensor-parallel tests, set mpi_ctx_ in SetUp() and override
 *   getDeviceForRank() instead of getDevice(). The base class handles:
 *   - Printing only on rank 0
 *   - Snapshot regeneration only on rank 0 (with barrier)
 *   - MPI barriers at key synchronization points
 *
 * Usage (single-rank):
 *   class Test__MyCUDAParity : public ParityTestBase {
 *   protected:
 *       void SetUp() override {
 *           config_.cosine_threshold = 0.99f;
 *           config_.early_layers_count = 6;
 *           ParityTestBase::SetUp();  // Regenerates snapshots
 *       }
 *
 *       DeviceId getDevice() override { return gpu_device_; }
 *       std::string getBackendName() override { return "CUDA"; }
 *   };
 *
 * Usage (tensor-parallel MPI):
 *   class Test__MyTPParity : public ParityTestBase {
 *   protected:
 *       void SetUp() override {
 *           mpi_ctx_ = std::make_shared<MPIContext>();
 *           config_.cosine_threshold = 0.94f;  // Relaxed for TP
 *           ParityTestBase::SetUp();
 *       }
 *       DeviceId getDeviceForRank() override {
 *           return (mpi_ctx_->rank() == 0) ? DeviceId::cuda(0) : DeviceId::rocm(0);
 *       }
 *       std::string getBackendName() override { return "TensorParallel"; }
 *   };
 *
 * @author David Sanftenberg
 * @date 2026-01-11
 */

#pragma once

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <iomanip>
#include <algorithm>
#include <set>
#include <string>

// libfort for formatted table output
#include "fort.hpp"

#include "loaders/ModelContext.h"
#include "execution/InferenceRunnerFactory.h"
#include "execution/IInferenceRunner.h"
#include "execution/TPSnapshot.h"
#include "execution/MultiDeviceOrchestrator.h"
#include "kernels/KernelFactory.h"
#include "utils/Logger.h"
#include "utils/MPIContext.h"
#include "backends/DeviceId.h"
#include "backends/ComputeBackend.h"

// NumPy .npy file loading
#include <cnpy.h>

// MPI for tensor-parallel tests
#include <mpi.h>

// For snapshot cache mutex
#include <mutex>

namespace llaminar2::test::parity
{

    // =============================================================================
    // Configuration
    // =============================================================================

    /**
     * @brief Configuration for parity test thresholds
     *
     * Different backends (CPU, CUDA, ROCm) may need different thresholds
     * due to varying quantization schemes and numerical precision.
     */
    struct ParityConfig
    {
        // Model and test setup
        std::string model_path = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
        std::string snapshot_dir = "pytorch_qwen2_snapshots";
        std::string prompt = "The quick brown fox jumps over the lazy dog";
        std::vector<int> token_ids = {785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562};
        int decode_steps = 5;

        // Layer-by-layer thresholds
        float cosine_threshold = 0.99f;  ///< Minimum avg cosine similarity for layer pass
        bool use_avg_cosine = true;      ///< Use avg (true) or min (false) cosine for pass criteria
        int early_layers_count = 6;      ///< Number of early layers to enforce threshold on
        int min_early_layers_passed = 6; ///< Minimum early layers that must pass

        // LM_HEAD thresholds
        float kl_threshold = 0.15f;      ///< Maximum KL divergence for logits
        float min_top1_accuracy = 60.0f; ///< Minimum Top-1 accuracy percentage

        // Decode thresholds (for incremental decode tests)
        float decode_cosine_threshold = 0.99f;
        float min_decode_pass_rate = 0.8f; ///< Minimum fraction of decode steps that must pass

        /// Stages to exclude from per-layer parity comparison.
        /// Used for GLOBAL scope TP where column-parallel stages (Q/K/V projections)
        /// produce partial outputs that can't be directly compared to full PyTorch outputs.
        std::vector<std::string> excluded_stages;
    };

    // =============================================================================
    // Result Structures
    // =============================================================================

    /**
     * @brief Result of comparing a single tensor/stage
     */
    struct StageComparisonResult
    {
        std::string stage_name;
        bool passed = false;
        float cosine_similarity = 0.0f;
        float rel_l2_norm = 0.0f;
        float max_abs_diff = 0.0f;
        float kl_divergence = 0.0f;
        size_t total_elements = 0;
    };

    /**
     * @brief Aggregated statistics for a single layer
     */
    struct LayerStats
    {
        int layer_idx = 0;
        float avg_cosine_sim = 0.0f;
        float min_cosine_sim = 1.0f;
        std::string worst_stage;
        int stages_compared = 0;
        bool passed = false;
    };

    /**
     * @brief Summary of parity test results
     */
    struct ParityTestSummary
    {
        // Embedding
        float embedding_cosine = 0.0f;
        bool embedding_passed = false;

        // Per-layer stats
        std::vector<LayerStats> layer_stats;

        // LM_HEAD
        float lm_head_cosine = 0.0f;
        float lm_head_kl = 0.0f;
        float lm_head_top1 = 0.0f;
        float lm_head_top5 = 0.0f;
        bool lm_head_passed = false;

        // Overall
        int early_layers_passed = 0;
        int total_layers_passed = 0;
        bool overall_passed = false;
    };

    /**
     * @brief Statistics for a single decode step
     */
    struct DecodeStepStats
    {
        int step_idx = 0;
        float cosine_similarity = 0.0f;
        float kl_divergence = 0.0f;
        float top1_overlap = 0.0f;
        int llaminar_token = -1;
        int pytorch_token = -1;
        bool token_match = false;
        bool passed = false;
    };

    /**
     * @brief Summary of incremental decode parity results
     */
    struct DecodeParitySummary
    {
        std::vector<DecodeStepStats> step_stats;
        int steps_passed = 0;
        int steps_total = 0;
        int top1_matches = 0;
        float avg_cosine = 0.0f;
        float avg_kl = 0.0f;
        float top1_accuracy = 0.0f;
        bool overall_passed = false;
    };

    // =============================================================================
    // TP-Aware Result Structures
    // =============================================================================

    /**
     * @brief Per-device comparison result for tensor-parallel parity
     */
    struct TPDeviceComparisonResult
    {
        std::string device_id;          ///< Device identifier (e.g., "rank0_cuda0")
        int device_index = 0;           ///< Index in TP group
        float cosine_similarity = 0.0f; ///< Cosine vs corresponding PyTorch slice
        size_t slice_start = 0;         ///< Start column in PyTorch reference
        size_t slice_size = 0;          ///< Number of elements compared
        bool passed = false;
    };

    /**
     * @brief TP-aware comparison result for a single stage
     */
    struct TPStageComparisonResult
    {
        std::string stage_name;
        SnapshotShardingMode sharding_mode = SnapshotShardingMode::UNKNOWN;

        // Per-device comparisons (for column-parallel stages)
        std::vector<TPDeviceComparisonResult> device_results;

        // Combined result (concatenated partial outputs vs full PyTorch)
        float combined_cosine = 0.0f;
        size_t combined_elements = 0;
        bool combined_passed = false;

        // Overall for this stage
        bool passed = false;
    };

    /**
     * @brief TP-aware layer statistics
     */
    struct TPLayerStats
    {
        int layer_idx = 0;
        int tp_degree = 1;

        // Per-stage results
        std::vector<TPStageComparisonResult> stage_results;

        // Aggregated metrics
        float avg_combined_cosine = 0.0f;
        float min_combined_cosine = 1.0f;
        std::string worst_stage;
        int stages_compared = 0;
        bool passed = false;
    };

    /**
     * @brief TP-aware parity test summary
     */
    struct TPParityTestSummary
    {
        int tp_degree = 1;
        std::vector<std::string> device_names; ///< Device IDs in TP group

        // Embedding
        TPStageComparisonResult embedding_result;

        // Per-layer stats
        std::vector<TPLayerStats> layer_stats;

        // LM_HEAD (always gathered, so single combined result)
        float lm_head_cosine = 0.0f;
        float lm_head_kl = 0.0f;
        float lm_head_top1 = 0.0f;
        float lm_head_top5 = 0.0f;
        bool lm_head_passed = false;

        // Overall
        int early_layers_passed = 0;
        int total_layers_passed = 0;
        bool overall_passed = false;
    };

    // =============================================================================
    // Metric Computation Functions
    // =============================================================================

    /**
     * @brief Compute cosine similarity between two vectors
     *
     * Cosine similarity measures directional alignment, ignoring magnitude.
     * Preferred for embedding comparisons because quantization noise affects
     * magnitude but preserves direction.
     *
     * @return Value in [-1, 1], where 1 = identical direction
     */
    inline float computeCosineSimilarity(const float *a, const float *b, size_t size)
    {
        double dot_product = 0.0;
        double norm_a = 0.0;
        double norm_b = 0.0;

        for (size_t i = 0; i < size; ++i)
        {
            dot_product += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
            norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }

        double denominator = std::sqrt(norm_a) * std::sqrt(norm_b);
        if (denominator < 1e-10)
        {
            return 0.0f;
        }

        return static_cast<float>(dot_product / denominator);
    }

    /**
     * @brief Compute KL divergence between probability distributions
     *
     * KL(P || Q) measures how P diverges from Q.
     * First applies softmax to convert logits to probabilities.
     *
     * @param actual_logits Llaminar logits (unnormalized)
     * @param expected_logits PyTorch logits (unnormalized)
     * @param size Total elements (seq_len * vocab_size)
     * @param vocab_size Vocabulary size for per-position softmax
     * @return Average KL divergence per position (in nats)
     */
    inline float computeKLDivergence(
        const float *actual_logits,
        const float *expected_logits,
        size_t size,
        size_t vocab_size)
    {
        size_t seq_len = size / vocab_size;
        double total_kl = 0.0;

        for (size_t pos = 0; pos < seq_len; ++pos)
        {
            const float *actual_row = actual_logits + pos * vocab_size;
            const float *expected_row = expected_logits + pos * vocab_size;

            // Find max for numerical stability (log-sum-exp trick)
            float max_actual = actual_row[0];
            float max_expected = expected_row[0];
            for (size_t i = 1; i < vocab_size; ++i)
            {
                max_actual = std::max(max_actual, actual_row[i]);
                max_expected = std::max(max_expected, expected_row[i]);
            }

            // Compute softmax denominators
            double sum_exp_actual = 0.0;
            double sum_exp_expected = 0.0;
            for (size_t i = 0; i < vocab_size; ++i)
            {
                sum_exp_actual += std::exp(actual_row[i] - max_actual);
                sum_exp_expected += std::exp(expected_row[i] - max_expected);
            }
            double log_sum_actual = max_actual + std::log(sum_exp_actual);
            double log_sum_expected = max_expected + std::log(sum_exp_expected);

            // KL divergence: KL(expected || actual)
            double pos_kl = 0.0;
            for (size_t i = 0; i < vocab_size; ++i)
            {
                double log_p = expected_row[i] - log_sum_expected;
                double log_q = actual_row[i] - log_sum_actual;
                double p = std::exp(log_p);

                if (p > 1e-10)
                {
                    pos_kl += p * (log_p - log_q);
                }
            }
            total_kl += pos_kl;
        }

        return static_cast<float>(total_kl / seq_len);
    }

    /**
     * @brief Compute Top-K overlap between two sets of logits
     *
     * Checks if the top K tokens predicted by both models overlap.
     * This is a "smoke test" for decision quality.
     *
     * @return Overlap percentage in [0, 1]
     */
    inline float computeTopKOverlap(
        const float *actual_logits,
        const float *expected_logits,
        size_t size,
        size_t vocab_size,
        int k)
    {
        size_t seq_len = size / vocab_size;
        double total_overlap = 0.0;

        for (size_t pos = 0; pos < seq_len; ++pos)
        {
            const float *actual_row = actual_logits + pos * vocab_size;
            const float *expected_row = expected_logits + pos * vocab_size;

            auto get_top_k = [&](const float *logits)
            {
                std::vector<std::pair<float, int>> scores(vocab_size);
                for (size_t i = 0; i < vocab_size; ++i)
                {
                    scores[i] = {logits[i], static_cast<int>(i)};
                }
                std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
                                  [](const auto &a, const auto &b)
                                  { return a.first > b.first; });

                std::vector<int> indices(k);
                for (int i = 0; i < k; ++i)
                    indices[i] = scores[i].second;
                std::sort(indices.begin(), indices.end());
                return indices;
            };

            auto actual_topk = get_top_k(actual_row);
            auto expected_topk = get_top_k(expected_row);

            std::vector<int> intersection;
            std::set_intersection(
                actual_topk.begin(), actual_topk.end(),
                expected_topk.begin(), expected_topk.end(),
                std::back_inserter(intersection));

            total_overlap += static_cast<double>(intersection.size()) / k;
        }

        return static_cast<float>(total_overlap / seq_len);
    }

    // =============================================================================
    // Table Rendering
    // =============================================================================

    /**
     * @brief Render a formatted parity results table to stdout using libfort
     *
     * Produces a consistent Unicode box-drawing table showing:
     * - Per-layer cosine similarity (avg and min)
     * - Worst stage per layer
     * - Pass/fail status with checkmarks
     * - LM_HEAD KL divergence and Top-K accuracy
     *
     * Uses libfort for clean, automatic column sizing and Unicode borders.
     */
    inline void renderParityTable(
        const ParityTestSummary &summary,
        const ParityConfig &config,
        const std::string &backend_name)
    {
        // Helper: format float to string with precision
        auto fmt_f6 = [](float v) -> std::string
        {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(6) << v;
            return ss.str();
        };

        auto fmt_f4 = [](float v) -> std::string
        {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(4) << v;
            return ss.str();
        };

        auto fmt_f1 = [](float v) -> std::string
        {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(1) << v;
            return ss.str();
        };

        // Helper: status icon
        auto status_str = [](bool passed) -> std::string
        {
            return passed ? "✓" : "✗";
        };

        std::cout << "\n";

        // =========================================================================
        // Title table
        // =========================================================================
        {
            fort::utf8_table title_table;
            title_table.set_border_style(FT_DOUBLE2_STYLE);

            std::ostringstream title_ss;
            title_ss << backend_name << " vs PyTorch LAYER-BY-LAYER PARITY";

            std::ostringstream subtitle_ss;
            subtitle_ss << "Threshold: " << (config.use_avg_cosine ? "avg" : "min")
                        << " cosine similarity >= " << std::fixed << std::setprecision(3)
                        << config.cosine_threshold;

            title_table << title_ss.str() << fort::endr;
            title_table << subtitle_ss.str() << fort::endr;

            title_table[0][0].set_cell_text_align(fort::text_align::center);
            title_table[1][0].set_cell_text_align(fort::text_align::center);
            title_table.row(0).set_cell_row_type(fort::row_type::header);

            std::cout << title_table.to_string();
        }

        // =========================================================================
        // Main parity table
        // =========================================================================
        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);

        // Header row
        table << fort::header
              << "Layer" << "Avg Cosine" << "Min Cosine" << "Worst Stage" << "OK"
              << fort::endr;

        // Set column alignments
        table.column(0).set_cell_text_align(fort::text_align::center);
        table.column(1).set_cell_text_align(fort::text_align::right);
        table.column(2).set_cell_text_align(fort::text_align::right);
        table.column(3).set_cell_text_align(fort::text_align::left);
        table.column(4).set_cell_text_align(fort::text_align::center);

        // Embedding row
        table << "EMBEDDING"
              << fmt_f6(summary.embedding_cosine)
              << fmt_f6(summary.embedding_cosine)
              << "-"
              << status_str(summary.embedding_passed)
              << fort::endr;

        // Per-layer rows
        for (const auto &stats : summary.layer_stats)
        {
            std::ostringstream layer_ss;
            layer_ss << "Layer " << stats.layer_idx;

            table << layer_ss.str()
                  << fmt_f6(stats.avg_cosine_sim)
                  << fmt_f6(stats.min_cosine_sim)
                  << stats.worst_stage
                  << status_str(stats.passed)
                  << fort::endr;
        }

        // Separator before LM_HEAD
        table << fort::separator;

        // LM_HEAD row with extra info
        std::ostringstream lm_info;
        lm_info << "KL=" << fmt_f4(summary.lm_head_kl)
                << " Top1=" << fmt_f1(summary.lm_head_top1 * 100.0f) << "%";

        table << "LM_HEAD"
              << fmt_f6(summary.lm_head_cosine)
              << fmt_f6(summary.lm_head_cosine)
              << lm_info.str()
              << status_str(summary.lm_head_passed)
              << fort::endr;

        std::cout << table.to_string();

        // =========================================================================
        // Summary footer
        // =========================================================================
        std::cout << "\nLM_HEAD Top-5: " << std::fixed << std::setprecision(1)
                  << (summary.lm_head_top5 * 100.0f) << "%\n";
        std::cout << "Early layers passed: " << summary.early_layers_passed
                  << "/" << config.early_layers_count << "\n";
        std::cout << "LM_HEAD KL divergence: " << std::fixed << std::setprecision(4)
                  << summary.lm_head_kl << " (threshold: " << config.kl_threshold << ")\n";
    }

    /**
     * @brief Render a TP-aware parity results table to stdout using libfort
     *
     * For tensor-parallel tests, shows:
     * - Per-device cosine similarity against PyTorch slices
     * - Combined (concatenated) result vs full PyTorch
     * - Sharding mode for each stage
     *
     * Supports an arbitrary number of TP devices with dynamic column sizing.
     * Uses libfort for clean, Unicode box-drawing table formatting.
     */
    inline void renderTPParityTable(
        const TPParityTestSummary &summary,
        const ParityConfig &config,
        const std::string &backend_name)
    {
        const size_t num_devices = summary.device_names.size();

        // Helper: sharding mode to string
        auto mode_str = [](SnapshotShardingMode mode) -> std::string
        {
            switch (mode)
            {
            case SnapshotShardingMode::REPLICATED:
                return "R";
            case SnapshotShardingMode::COLUMN_PARALLEL:
                return "C";
            case SnapshotShardingMode::ROW_PARALLEL:
                return "W";
            case SnapshotShardingMode::GATHERED:
                return "G";
            default:
                return "?";
            }
        };

        // Helper: status icon
        auto status_str = [](bool passed) -> std::string
        {
            return passed ? "✓" : "✗";
        };

        // Helper: format float to string
        auto fmt_f6 = [](float v) -> std::string
        {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(6) << v;
            return ss.str();
        };

        // =========================================================================
        // Title table (separate table for clean title block)
        // =========================================================================
        std::cout << "\n";
        {
            fort::utf8_table title_table;
            title_table.set_border_style(FT_DOUBLE2_STYLE);

            std::ostringstream title_ss;
            title_ss << backend_name << " vs PyTorch TP-AWARE PARITY";

            std::ostringstream subtitle_ss;
            subtitle_ss << summary.tp_degree << "-way LOCAL TP, Threshold: cosine >= "
                        << std::fixed << std::setprecision(3) << config.cosine_threshold;

            title_table << title_ss.str() << fort::endr;
            title_table << subtitle_ss.str() << fort::endr;

            title_table[0][0].set_cell_text_align(fort::text_align::center);
            title_table[1][0].set_cell_text_align(fort::text_align::center);
            title_table.row(0).set_cell_row_type(fort::row_type::header);

            std::cout << title_table.to_string();
        }

        // =========================================================================
        // Main parity table
        // =========================================================================
        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);

        // Build header row: Stage | Dev0 | Dev1 | ... | Combined | Mode | Status
        table << fort::header << "Stage";
        for (size_t i = 0; i < num_devices; ++i)
        {
            table << summary.device_names[i];
        }
        table << "Combined" << "Mode" << "OK" << fort::endr;

        // Set center alignment for all columns
        for (size_t col = 0; col < num_devices + 4; ++col)
        {
            table.column(col).set_cell_text_align(fort::text_align::center);
        }

        // Embedding row
        {
            const auto &emb = summary.embedding_result;
            table << "EMBEDDING";
            for (size_t i = 0; i < num_devices; ++i)
            {
                if (i < emb.device_results.size())
                {
                    table << fmt_f6(emb.device_results[i].cosine_similarity);
                }
                else
                {
                    table << "-";
                }
            }
            table << fmt_f6(emb.combined_cosine)
                  << mode_str(emb.sharding_mode)
                  << status_str(emb.passed)
                  << fort::endr;
        }

        // Per-layer rows
        for (const auto &layer : summary.layer_stats)
        {
            // Layer header row
            std::ostringstream layer_ss;
            layer_ss << "Layer " << layer.layer_idx;

            table << fort::separator;
            table << layer_ss.str();
            for (size_t i = 0; i < num_devices; ++i)
            {
                table << "";
            }
            table << fmt_f6(layer.avg_combined_cosine)
                  << ""
                  << status_str(layer.passed)
                  << fort::endr;

            // Per-stage details (limit to 6 for readability)
            int stage_count = 0;
            for (const auto &stage : layer.stage_results)
            {
                if (stage_count++ >= 6)
                    break;

                // Truncate stage name if needed
                std::string stage_name = stage.stage_name;
                if (stage_name.size() > 14)
                {
                    stage_name = stage_name.substr(0, 12) + "..";
                }

                table << stage_name;
                for (size_t i = 0; i < num_devices; ++i)
                {
                    if (i < stage.device_results.size())
                    {
                        table << fmt_f6(stage.device_results[i].cosine_similarity);
                    }
                    else
                    {
                        table << "-";
                    }
                }
                table << fmt_f6(stage.combined_cosine)
                      << mode_str(stage.sharding_mode)
                      << status_str(stage.passed)
                      << fort::endr;
            }
        }

        // LM_HEAD row
        table << fort::separator;
        table << "LM_HEAD";
        for (size_t i = 0; i < num_devices; ++i)
        {
            table << "[gathered]";
        }
        table << fmt_f6(summary.lm_head_cosine)
              << "G"
              << status_str(summary.lm_head_passed)
              << fort::endr;

        std::cout << table.to_string();

        // =========================================================================
        // Summary footer
        // =========================================================================
        std::cout << "\nSharding modes: R=Replicated, C=Column-parallel, W=roW-parallel, G=Gathered\n";
        std::cout << std::fixed << std::setprecision(4)
                  << "LM_HEAD: KL=" << summary.lm_head_kl
                  << std::setprecision(1)
                  << " Top-1=" << (summary.lm_head_top1 * 100.0f) << "%"
                  << " Top-5=" << (summary.lm_head_top5 * 100.0f) << "%\n";
        std::cout << "Early layers passed: " << summary.early_layers_passed
                  << "/" << config.early_layers_count << "\n";
    }

    // =============================================================================
    // Base Test Class
    // =============================================================================

    /**
     * @brief Base class for PyTorch parity tests
     *
     * Provides common infrastructure for comparing Llaminar backends against
     * PyTorch ground truth. Subclasses must implement:
     * - getDevice() OR getDeviceForRank() - Return the DeviceId to use for inference
     * - getBackendName() - Return a display name (e.g., "CUDA", "CPU", "ROCm")
     *
     * For MPI/tensor-parallel tests:
     * - Set mpi_ctx_ in SetUp() before calling ParityTestBase::SetUp()
     * - Override getDeviceForRank() instead of getDevice()
     *
     * Optional overrides:
     * - setupDeviceSpecific() - Device-specific initialization (e.g., CUDA checks)
     */
    class ParityTestBase : public ::testing::Test
    {
    private:
        // Static set of snapshot directories that have been generated this run.
        // This prevents regenerating snapshots for every test case in a suite.
        // Key: snapshot_dir path, Value: true if successfully generated
        static inline std::set<std::string> s_generated_snapshots_;
        static inline std::mutex s_snapshot_mutex_;

    protected:
        ParityConfig config_;
        std::shared_ptr<ModelContext> model_ctx_;
        std::unique_ptr<IInferenceRunner> runner_;
        std::unordered_map<std::string, std::vector<float>> pytorch_snapshots_;

        // MPI context for tensor-parallel tests (optional, null for single-rank)
        std::shared_ptr<MPIContext> mpi_ctx_;

        // =============================================================================
        // MPI Helper Methods
        // =============================================================================

        /**
         * @brief Check if this is rank 0 (or single-rank mode)
         * @return true if rank 0 or no MPI context
         */
        bool isRank0() const
        {
            return !mpi_ctx_ || mpi_ctx_->rank() == 0;
        }

        /**
         * @brief Get MPI rank (0 if no MPI context)
         */
        int mpiRank() const
        {
            return mpi_ctx_ ? mpi_ctx_->rank() : 0;
        }

        /**
         * @brief Get MPI world size (1 if no MPI context)
         */
        int mpiWorldSize() const
        {
            return mpi_ctx_ ? mpi_ctx_->world_size() : 1;
        }

        /**
         * @brief Execute MPI barrier if MPI context exists
         */
        void mpiBarrier()
        {
            if (mpi_ctx_)
            {
                mpi_ctx_->barrier();
            }
        }

        // =============================================================================
        // Device Selection (override one of these)
        // =============================================================================

        /**
         * @brief Get the device to use for inference (single-rank tests)
         * @return DeviceId (e.g., DeviceId::cpu(), DeviceId::cuda(0))
         *
         * Override this for single-rank tests. For MPI tests, override
         * getDeviceForRank() instead.
         */
        virtual DeviceId getDevice()
        {
            // Default implementation calls getDeviceForRank() for MPI compatibility
            return getDeviceForRank();
        }

        /**
         * @brief Get the device for this MPI rank (tensor-parallel tests)
         * @return DeviceId for the current rank
         *
         * Override this for MPI tests to return different devices per rank.
         * Default implementation returns CPU.
         */
        virtual DeviceId getDeviceForRank()
        {
            return DeviceId::cpu();
        }

        /**
         * @brief Get the backend name for display
         * @return Name string (e.g., "CUDA", "CPU", "ROCm")
         */
        virtual std::string getBackendName() = 0;

        // =============================================================================
        // Weight Distribution (override for tensor parallelism)
        // =============================================================================

        /**
         * @brief Get the weight distribution strategy
         * @return WeightDistributionStrategy (REPLICATED for single-rank, SHARDED for TP)
         *
         * Override this for tensor-parallel tests to enable weight sharding.
         */
        virtual WeightDistributionStrategy getWeightStrategy()
        {
            return WeightDistributionStrategy::REPLICATED;
        }

        /**
         * @brief Configure model after loading (optional hook)
         * @param model_ctx The loaded model context
         *
         * Override this for tensor-parallel tests to configure weight sharding schema.
         * Called after ModelContext::create() but before createInferenceRunner().
         */
        virtual void configureModel(std::shared_ptr<ModelContext> model_ctx)
        {
            // Default: no additional configuration
        }

        /**
         * @brief Device-specific setup (optional)
         *
         * Override to add device availability checks, GPU initialization, etc.
         * Call GTEST_SKIP() if device is not available.
         */
        virtual void setupDeviceSpecific() {}

        void SetUp() override
        {
            // Device-specific setup first (may skip)
            setupDeviceSpecific();

            // Regenerate snapshots only on rank 0 to avoid race conditions
            // and redundant work. All ranks wait at barrier before proceeding.
            // OPTIMIZATION: Only regenerate if this snapshot_dir hasn't been done yet.
            if (isRank0())
            {
                bool need_regen = false;
                {
                    std::lock_guard<std::mutex> lock(s_snapshot_mutex_);
                    need_regen = (s_generated_snapshots_.find(config_.snapshot_dir) == s_generated_snapshots_.end());
                }

                if (need_regen)
                {
                    if (!regeneratePyTorchSnapshots())
                    {
                        FAIL() << "PyTorch snapshot generation failed";
                    }
                    // Mark as generated
                    std::lock_guard<std::mutex> lock(s_snapshot_mutex_);
                    s_generated_snapshots_.insert(config_.snapshot_dir);
                }
                else
                {
                    LOG_DEBUG("[" << getBackendName() << " Parity] Reusing cached snapshots from: " << config_.snapshot_dir);
                }
            }
            mpiBarrier(); // All ranks wait for snapshots to be ready
        }

        void TearDown() override
        {
            // Barrier before teardown to ensure all ranks are done
            mpiBarrier();

            // CRITICAL: Clear kernel cache BEFORE destroying model context!
            // KernelFactory::clearCache() accesses tensor->rocm_cache_ and tensor->cuda_cache_
            // to free device memory. If we destroy the tensors first (via model_ctx_.reset()),
            // clearCache() would be accessing freed memory (use-after-free).
            llaminar::v2::kernels::KernelFactory::clearCache();

            model_ctx_.reset();
            runner_.reset();
            pytorch_snapshots_.clear();
        }

        /**
         * @brief Regenerate PyTorch snapshots from the GGUF model
         */
        bool regeneratePyTorchSnapshots()
        {
            LOG_INFO("[" << getBackendName() << " Parity] Regenerating PyTorch snapshots from GGUF: " << config_.model_path);

            std::ostringstream cmd;
            cmd << "bash -c 'source /workspaces/llaminar/.venv/bin/activate && python3"
                << " python/reference/generate_qwen2_pipeline_snapshots.py"
                << " --model " << config_.model_path
                << " --prompt \"" << config_.prompt << "\""
                << " --output " << config_.snapshot_dir
                << " --decode-steps " << config_.decode_steps
                << "' 2>&1";

            FILE *pipe = popen(cmd.str().c_str(), "r");
            if (!pipe)
            {
                LOG_ERROR("[Parity] Failed to execute snapshot generator");
                return false;
            }

            char buffer[256];
            std::string output;
            while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
            {
                output += buffer;
            }

            int exit_code = pclose(pipe);
            if (exit_code != 0)
            {
                LOG_ERROR("[Parity] Snapshot generation failed:\n"
                          << output);
                return false;
            }

            LOG_INFO("[Parity] Snapshots regenerated successfully");
            return true;
        }

        /**
         * @brief Load PyTorch snapshot from .npy file
         */
        std::vector<float> loadPyTorchSnapshot(const std::string &name)
        {
            if (pytorch_snapshots_.find(name) != pytorch_snapshots_.end())
            {
                return pytorch_snapshots_[name];
            }

            std::string npy_path = config_.snapshot_dir + "/" + name + ".npy";

            try
            {
                cnpy::NpyArray arr = cnpy::npy_load(npy_path);

                std::vector<float> data;
                if (arr.word_size == sizeof(float))
                {
                    float *data_ptr = arr.data<float>();
                    data.assign(data_ptr, data_ptr + arr.num_vals);
                }
                else if (arr.word_size == sizeof(double))
                {
                    double *data_ptr = arr.data<double>();
                    data.resize(arr.num_vals);
                    for (size_t i = 0; i < arr.num_vals; ++i)
                    {
                        data[i] = static_cast<float>(data_ptr[i]);
                    }
                }
                else
                {
                    LOG_ERROR("[Parity] Unsupported data type in snapshot '" << name << "'");
                    return {};
                }

                pytorch_snapshots_[name] = data;
                return data;
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("[Parity] Failed to load snapshot '" << name << "': " << e.what());
                return {};
            }
        }

        /**
         * @brief Compare tensors and compute metrics
         */
        StageComparisonResult compareTensors(
            const float *actual,
            const std::vector<float> &expected,
            size_t size,
            const std::string &stage_name = "")
        {
            StageComparisonResult result;
            result.stage_name = stage_name;
            result.total_elements = size;

            if (expected.empty() || expected.size() != size)
            {
                return result;
            }

            double sum_sq_diff = 0.0;
            double sum_sq_expected = 0.0;
            double dot_product = 0.0;
            double norm_actual_sq = 0.0;
            double norm_expected_sq = 0.0;

            for (size_t i = 0; i < size; ++i)
            {
                float diff = actual[i] - expected[i];
                sum_sq_diff += diff * diff;
                sum_sq_expected += expected[i] * expected[i];
                dot_product += actual[i] * expected[i];
                norm_actual_sq += actual[i] * actual[i];
                norm_expected_sq += expected[i] * expected[i];

                float abs_diff = std::abs(diff);
                if (abs_diff > result.max_abs_diff)
                {
                    result.max_abs_diff = abs_diff;
                }
            }

            // Relative L2
            if (sum_sq_expected > 1e-10)
            {
                result.rel_l2_norm = static_cast<float>(std::sqrt(sum_sq_diff / sum_sq_expected));
            }

            // Cosine similarity
            double norm_product = std::sqrt(norm_actual_sq) * std::sqrt(norm_expected_sq);
            if (norm_product > 1e-10)
            {
                result.cosine_similarity = static_cast<float>(dot_product / norm_product);
            }

            result.passed = (result.cosine_similarity >= config_.cosine_threshold);
            return result;
        }

        /**
         * @brief Compare TP snapshot against PyTorch reference with sharding awareness
         *
         * For column-parallel stages:
         * 1. Compare each device's partial output against corresponding PyTorch slice
         * 2. Concatenate all device outputs and compare against full PyTorch
         *
         * For row-parallel/replicated stages:
         * 1. Verify all devices have consistent data
         * 2. Compare combined output against full PyTorch
         *
         * @param tp_snapshot The TPSnapshot from MultiDeviceOrchestrator::getTPSnapshot()
         * @param pytorch_data Full PyTorch reference data
         * @param pytorch_rows Number of rows in PyTorch data (seq_len)
         * @param pytorch_cols Number of columns in PyTorch data (feature_dim)
         * @return TPStageComparisonResult with per-device and combined metrics
         */
        TPStageComparisonResult compareTPSnapshot(
            TPSnapshot &tp_snapshot,
            const std::vector<float> &pytorch_data,
            size_t pytorch_rows,
            size_t pytorch_cols)
        {
            TPStageComparisonResult result;

            // Debug: print first few values for layer0 attention to diagnose
            bool debug_print = (tp_snapshot.key.find("layer0_ATTENTION_CONTEXT") != std::string::npos);
            result.stage_name = tp_snapshot.key;
            result.sharding_mode = tp_snapshot.mode;

            if (pytorch_data.empty())
            {
                LOG_WARN("[TP Parity] No PyTorch data for stage " << tp_snapshot.key);
                return result;
            }

            const int tp_degree = static_cast<int>(tp_snapshot.device_data.size());
            if (tp_degree == 0)
            {
                LOG_WARN("[TP Parity] No device data for stage " << tp_snapshot.key);
                return result;
            }

            LOG_DEBUG("[TP Parity] Comparing stage " << tp_snapshot.key
                                                     << " mode=" << shardingModeToString(tp_snapshot.mode)
                                                     << " tp_degree=" << tp_degree
                                                     << " pytorch_size=" << pytorch_data.size()
                                                     << " (" << pytorch_rows << "x" << pytorch_cols << ")");

            // Per-device comparison for column-parallel stages
            if (tp_snapshot.mode == SnapshotShardingMode::COLUMN_PARALLEL)
            {
                for (int dev_idx = 0; dev_idx < tp_degree; ++dev_idx)
                {
                    const auto &dev_data = tp_snapshot.device_data[dev_idx];

                    // Compute which slice of PyTorch this device should match
                    size_t slice_start = computeSliceStartCol(dev_idx, tp_degree, pytorch_cols);
                    size_t slice_cols = computeSliceColCount(dev_idx, tp_degree, pytorch_cols);

                    // Extract PyTorch slice
                    std::vector<float> pytorch_slice = extractColumnSlice(
                        pytorch_data.data(), pytorch_rows, pytorch_cols, slice_start, slice_cols);

                    // Debug: print actual values for diagnosis - VERBOSE for all ATTENTION_CONTEXT stages
                    bool debug_verbose = (tp_snapshot.key.find("ATTENTION_CONTEXT") != std::string::npos);
                    if (debug_verbose)
                    {
                        LOG_INFO("[TP Debug] " << tp_snapshot.key << " device " << dev_idx
                                               << " pytorch_rows=" << pytorch_rows
                                               << " pytorch_cols=" << pytorch_cols
                                               << " slice_start=" << slice_start
                                               << " slice_cols=" << slice_cols
                                               << " pytorch_slice.size=" << pytorch_slice.size()
                                               << " dev_data.size=" << dev_data.data.size()
                                               << " dev_data.cols=" << dev_data.cols);

                        // Print first 8 values from each
                        std::stringstream ss_pt, ss_ll;
                        for (size_t i = 0; i < std::min(size_t(8), pytorch_slice.size()); ++i)
                            ss_pt << std::setprecision(6) << pytorch_slice[i] << ", ";
                        for (size_t i = 0; i < std::min(size_t(8), dev_data.data.size()); ++i)
                            ss_ll << std::setprecision(6) << dev_data.data[i] << ", ";
                        LOG_INFO("[TP Debug]   Row0 PyTorch: " << ss_pt.str());
                        LOG_INFO("[TP Debug]   Row0 Llaminar: " << ss_ll.str());

                        // Row 1 values (at offset slice_cols for Llaminar, slice_cols for pytorch_slice)
                        size_t row_stride_ll = dev_data.cols > 0 ? dev_data.cols : slice_cols;
                        size_t row_stride_pt = slice_cols;
                        if (pytorch_slice.size() > row_stride_pt + 8 && dev_data.data.size() > row_stride_ll + 8)
                        {
                            std::stringstream ss3, ss4;
                            for (size_t i = 0; i < 8; ++i)
                                ss3 << std::setprecision(6) << pytorch_slice[row_stride_pt + i] << ", ";
                            for (size_t i = 0; i < 8; ++i)
                                ss4 << std::setprecision(6) << dev_data.data[row_stride_ll + i] << ", ";
                            LOG_INFO("[TP Debug]   Row1 PyTorch (stride=" << row_stride_pt << "): " << ss3.str());
                            LOG_INFO("[TP Debug]   Row1 Llaminar (stride=" << row_stride_ll << "): " << ss4.str());
                        }
                    }

                    // Compare device data against slice
                    size_t compare_size = std::min(dev_data.data.size(), pytorch_slice.size());
                    float cosine = computeCosineSimilarity(
                        dev_data.data.data(), pytorch_slice.data(), compare_size);

                    TPDeviceComparisonResult dev_result;
                    dev_result.device_id = dev_data.device_id.toString();
                    dev_result.device_index = dev_idx;
                    dev_result.cosine_similarity = cosine;
                    dev_result.slice_start = slice_start;
                    dev_result.slice_size = compare_size;
                    dev_result.passed = (cosine >= config_.cosine_threshold);

                    LOG_DEBUG("[TP Parity] Device " << dev_idx << " (" << dev_result.device_id << ")"
                                                    << " slice=[" << slice_start << "," << slice_start + slice_cols << ")"
                                                    << " cosine=" << cosine
                                                    << " device_size=" << dev_data.data.size()
                                                    << " slice_size=" << pytorch_slice.size());

                    result.device_results.push_back(std::move(dev_result));
                }
            }
            else
            {
                // For replicated/row-parallel, each device should match full PyTorch
                for (int dev_idx = 0; dev_idx < tp_degree; ++dev_idx)
                {
                    const auto &dev_data = tp_snapshot.device_data[dev_idx];

                    size_t compare_size = std::min(dev_data.data.size(), pytorch_data.size());
                    float cosine = computeCosineSimilarity(
                        dev_data.data.data(), pytorch_data.data(), compare_size);

                    TPDeviceComparisonResult dev_result;
                    dev_result.device_id = dev_data.device_id.toString();
                    dev_result.device_index = dev_idx;
                    dev_result.cosine_similarity = cosine;
                    dev_result.slice_start = 0;
                    dev_result.slice_size = compare_size;
                    dev_result.passed = (cosine >= config_.cosine_threshold);

                    result.device_results.push_back(std::move(dev_result));
                }
            }

            // Compute combined result
            size_t combined_size = 0;
            const float *combined_ptr = tp_snapshot.getCombinedData(combined_size);

            if (combined_ptr && combined_size > 0)
            {
                size_t compare_size = std::min(combined_size, pytorch_data.size());
                result.combined_cosine = computeCosineSimilarity(
                    combined_ptr, pytorch_data.data(), compare_size);
                result.combined_elements = compare_size;
                result.combined_passed = (result.combined_cosine >= config_.cosine_threshold);

                LOG_DEBUG("[TP Parity] Combined: size=" << combined_size
                                                        << " pytorch_size=" << pytorch_data.size()
                                                        << " cosine=" << result.combined_cosine);
            }
            else
            {
                LOG_WARN("[TP Parity] Failed to compute combined data for " << tp_snapshot.key);
            }

            // Overall pass: combined must pass
            result.passed = result.combined_passed;

            return result;
        }

        /**
         * @brief Setup the inference pipeline
         *
         * MPI-aware: Uses getDevice() which calls getDeviceForRank() for per-rank device selection.
         * Passes mpi_ctx_ to createInferenceRunner (nullptr for single-rank tests).
         * Uses getWeightStrategy() for weight distribution (REPLICATED or SHARDED).
         * Calls configureModel() hook for tensor-parallel schema configuration.
         */
        bool setupPipeline()
        {
            DeviceManager::instance().initialize(-1);

            // Load model with MPI context and weight strategy
            model_ctx_ = ModelContext::create(
                config_.model_path,
                mpi_ctx_,           // nullptr for single-rank
                nullptr,            // placement_map
                nullptr,            // factory
                getWeightStrategy() // REPLICATED or SHARDED
            );
            if (!model_ctx_)
            {
                LOG_ERROR("[Parity] Failed to load model");
                return false;
            }

            // Allow subclasses to configure model (e.g., weight sharding schema)
            configureModel(model_ctx_);

            InferenceRunnerConfig inf_config;
            inf_config.max_seq_len = 4096;
            inf_config.batch_size = 1;
            inf_config.force_graph = true;

            // Enable mapped memory for GPU devices to avoid slow D2H syncs during snapshot capture
            // This works for both CUDA and ROCm - mapped memory enables zero-copy host access
            DeviceId device = getDevice();
            if (device.is_gpu())
            {
                inf_config.use_mapped_memory = true;
                if (isRank0())
                {
                    LOG_INFO("[" << getBackendName() << " Parity] Enabling mapped memory for GPU snapshot capture");
                }
            }

            // Pass mpi_ctx_ to enable tensor parallelism (nullptr for single-rank tests)
            runner_ = createInferenceRunner(model_ctx_, mpi_ctx_, device, inf_config);
            if (!runner_)
            {
                LOG_ERROR("[Parity] Failed to create inference runner");
                return false;
            }

            runner_->enableSnapshotCapture();
            if (isRank0())
            {
                LOG_INFO("[" << getBackendName() << " Parity] Inference runner created"
                             << (mpi_ctx_ ? " (MPI world_size=" + std::to_string(mpiWorldSize()) + ")" : ""));
            }
            return true;
        }

        /**
         * @brief Read decode tokens from metadata file
         */
        std::vector<int> readDecodeTokensFromMetadata()
        {
            std::string metadata_path = config_.snapshot_dir + "/metadata.txt";
            std::ifstream file(metadata_path);
            if (!file.is_open())
            {
                LOG_WARN("[Parity] Could not open metadata file: " << metadata_path);
                return {};
            }

            std::vector<int> decode_tokens;
            std::string line;
            while (std::getline(file, line))
            {
                // Look for "decode_tokens: X,Y,Z" format
                if (line.find("decode_tokens:") == 0)
                {
                    size_t colon_pos = line.find(':');
                    if (colon_pos != std::string::npos)
                    {
                        std::string tokens_str = line.substr(colon_pos + 1);
                        // Trim leading whitespace
                        size_t start = tokens_str.find_first_not_of(" \t");
                        if (start != std::string::npos)
                        {
                            tokens_str = tokens_str.substr(start);
                        }
                        // Parse comma-separated token IDs
                        std::stringstream ss(tokens_str);
                        std::string token_str;
                        while (std::getline(ss, token_str, ','))
                        {
                            // Trim whitespace from token
                            size_t tok_start = token_str.find_first_not_of(" \t");
                            size_t tok_end = token_str.find_last_not_of(" \t");
                            if (tok_start != std::string::npos && tok_end != std::string::npos)
                            {
                                token_str = token_str.substr(tok_start, tok_end - tok_start + 1);
                            }
                            if (!token_str.empty())
                            {
                                try
                                {
                                    decode_tokens.push_back(std::stoi(token_str));
                                }
                                catch (const std::exception &e)
                                {
                                    LOG_WARN("[Parity] Failed to parse token: " << token_str);
                                }
                            }
                        }
                    }
                    break; // Found decode_tokens line, done
                }
            }
            return decode_tokens;
        }

        /**
         * @brief Run single-device prefill parity test and return summary
         *
         * This is the main test driver for SINGLE-DEVICE tests - compares
         * layer-by-layer against PyTorch. Always calls setupPipeline() to
         * create a single-device runner.
         *
         * For multi-device LOCAL TP tests, use runTPPrefillParity() instead.
         */
        ParityTestSummary runSingleDevicePrefillParity()
        {
            ParityTestSummary summary;

            // Always setup single-device pipeline
            EXPECT_TRUE(setupPipeline()) << "Pipeline setup failed";
            if (!runner_)
                return summary;

            // Run prefill
            bool success = runner_->forward(config_.token_ids.data(), config_.token_ids.size());
            EXPECT_TRUE(success) << "Prefill forward pass failed";
            if (!success)
                return summary;

            int n_layers = static_cast<int>(model_ctx_->model().block_count);

            // Stages to compare per layer
            std::vector<std::string> per_layer_stages = {
                "ATTENTION_NORM", "Q_PROJECTION", "K_PROJECTION", "V_PROJECTION",
                "Q_ROPE", "K_ROPE",
                "ATTENTION_CONTEXT", "ATTENTION_OUTPUT", "ATTENTION_RESIDUAL",
                "FFN_NORM", "FFN_GATE", "FFN_UP", "FFN_SWIGLU", "FFN_DOWN", "FFN_RESIDUAL"};

            // Get snapshot keys
            auto snapshot_keys = runner_->getSnapshotKeys();
            std::set<std::string> available_snapshots(snapshot_keys.begin(), snapshot_keys.end());

            // Compare embedding
            auto pytorch_embedding = loadPyTorchSnapshot("EMBEDDING");
            if (available_snapshots.count("EMBEDDING"))
            {
                size_t llaminar_size;
                const float *llaminar_data = runner_->getSnapshot("EMBEDDING", llaminar_size);

                // Debug sizes and first values (rank 0 only)
                if (isRank0())
                {
                    LOG_INFO("[Parity Debug] EMBEDDING - llaminar_size=" << llaminar_size
                                                                         << " pytorch_size=" << pytorch_embedding.size());
                    if (llaminar_data && llaminar_size >= 8)
                    {
                        LOG_INFO("[Parity Debug] Llaminar first 8: "
                                 << llaminar_data[0] << "," << llaminar_data[1] << ","
                                 << llaminar_data[2] << "," << llaminar_data[3] << ","
                                 << llaminar_data[4] << "," << llaminar_data[5] << ","
                                 << llaminar_data[6] << "," << llaminar_data[7]);
                    }
                    if (!pytorch_embedding.empty() && pytorch_embedding.size() >= 8)
                    {
                        LOG_INFO("[Parity Debug] PyTorch first 8: "
                                 << pytorch_embedding[0] << "," << pytorch_embedding[1] << ","
                                 << pytorch_embedding[2] << "," << pytorch_embedding[3] << ","
                                 << pytorch_embedding[4] << "," << pytorch_embedding[5] << ","
                                 << pytorch_embedding[6] << "," << pytorch_embedding[7]);
                    }
                }

                if (llaminar_data && !pytorch_embedding.empty())
                {
                    summary.embedding_cosine = computeCosineSimilarity(
                        llaminar_data, pytorch_embedding.data(),
                        std::min(llaminar_size, pytorch_embedding.size()));
                }
            }
            summary.embedding_passed = (summary.embedding_cosine >= config_.cosine_threshold);

            // Compare each layer
            for (int layer_idx = 0; layer_idx < n_layers; ++layer_idx)
            {
                LayerStats stats;
                stats.layer_idx = layer_idx;
                float sum_cosine = 0.0f;

                for (const auto &stage : per_layer_stages)
                {
                    // Skip excluded stages (e.g., Q/K/V_PROJECTION for GLOBAL scope TP)
                    if (!config_.excluded_stages.empty())
                    {
                        bool is_excluded = std::find(
                                               config_.excluded_stages.begin(),
                                               config_.excluded_stages.end(),
                                               stage) != config_.excluded_stages.end();
                        if (is_excluded)
                            continue;
                    }

                    std::string llaminar_key = "layer" + std::to_string(layer_idx) + "_" + stage;
                    std::string pytorch_key = llaminar_key;

                    if (!available_snapshots.count(llaminar_key))
                        continue;

                    auto pytorch_data = loadPyTorchSnapshot(pytorch_key);
                    if (pytorch_data.empty())
                        continue;

                    size_t llaminar_size;
                    const float *llaminar_data = runner_->getSnapshot(llaminar_key, llaminar_size);
                    if (!llaminar_data)
                        continue;

                    auto result = compareTensors(llaminar_data, pytorch_data, llaminar_size, stage);
                    stats.stages_compared++;
                    sum_cosine += result.cosine_similarity;

                    if (result.cosine_similarity < stats.min_cosine_sim)
                    {
                        stats.min_cosine_sim = result.cosine_similarity;
                        stats.worst_stage = stage;
                    }
                }

                if (stats.stages_compared > 0)
                {
                    stats.avg_cosine_sim = sum_cosine / stats.stages_compared;
                }

                // Pass criteria based on config
                float check_value = config_.use_avg_cosine ? stats.avg_cosine_sim : stats.min_cosine_sim;
                stats.passed = (check_value >= config_.cosine_threshold);

                summary.layer_stats.push_back(stats);
            }

            // Compare LM_HEAD
            auto pytorch_lm_head = loadPyTorchSnapshot("LM_HEAD");
            if (available_snapshots.count("LM_HEAD") && !pytorch_lm_head.empty())
            {
                size_t llaminar_size;
                const float *llaminar_data = runner_->getSnapshot("LM_HEAD", llaminar_size);
                if (llaminar_data)
                {
                    auto result = compareTensors(llaminar_data, pytorch_lm_head, llaminar_size, "LM_HEAD");
                    summary.lm_head_cosine = result.cosine_similarity;

                    size_t vocab_size = model_ctx_->model().vocab_size;
                    size_t seq_len = llaminar_size / vocab_size;

                    if (seq_len > 0)
                    {
                        size_t last_offset = (seq_len - 1) * vocab_size;
                        summary.lm_head_kl = computeKLDivergence(
                            llaminar_data + last_offset,
                            pytorch_lm_head.data() + last_offset,
                            vocab_size, vocab_size);

                        summary.lm_head_top1 = computeTopKOverlap(
                            llaminar_data + last_offset,
                            pytorch_lm_head.data() + last_offset,
                            vocab_size, vocab_size, 1);

                        summary.lm_head_top5 = computeTopKOverlap(
                            llaminar_data + last_offset,
                            pytorch_lm_head.data() + last_offset,
                            vocab_size, vocab_size, 5);
                    }
                }
            }
            summary.lm_head_passed = (summary.lm_head_kl < config_.kl_threshold);

            // Count early layers passed
            summary.early_layers_passed = summary.embedding_passed ? 1 : 0;
            for (int i = 0; i < std::min(config_.early_layers_count, static_cast<int>(summary.layer_stats.size())); ++i)
            {
                if (summary.layer_stats[i].passed)
                    summary.early_layers_passed++;
            }

            // Count total layers passed
            summary.total_layers_passed = summary.embedding_passed ? 1 : 0;
            for (const auto &stats : summary.layer_stats)
            {
                if (stats.passed)
                    summary.total_layers_passed++;
            }

            // Overall pass
            summary.overall_passed = (summary.early_layers_passed >= config_.min_early_layers_passed) &&
                                     summary.lm_head_passed;

            return summary;
        }

        /**
         * @brief Run TP-aware prefill parity test
         *
         * For tensor-parallel tests, this compares:
         * 1. Per-device partial outputs against corresponding PyTorch slices
         * 2. Combined (concatenated) outputs against full PyTorch reference
         *
         * Requires runner_ to be a MultiDeviceOrchestrator (will cast and check).
         *
         * @return TPParityTestSummary with per-device and combined metrics
         */
        TPParityTestSummary runTPPrefillParity()
        {
            TPParityTestSummary summary;

            // Only setup pipeline if not already configured
            // (Test may have already called setupLocalTPPipeline() or similar)
            if (!runner_)
            {
                EXPECT_TRUE(setupPipeline()) << "Pipeline setup failed";
                if (!runner_)
                    return summary;
            }

            // Try to cast to MultiDeviceOrchestrator for TP snapshot access
            auto *multi_device = dynamic_cast<MultiDeviceOrchestrator *>(runner_.get());
            if (!multi_device)
            {
                LOG_ERROR("[TP Parity] runner_ is not a MultiDeviceOrchestrator - "
                          "ensure test calls setupLocalTPPipeline() or similar before runTPPrefillParity()");
                return summary;
            }

            summary.tp_degree = multi_device->device_count();
            for (int i = 0; i < summary.tp_degree; ++i)
            {
                auto *dev_runner = multi_device->deviceRunner(i);
                if (dev_runner)
                {
                    // Use index-based name for simplicity
                    // MultiDeviceOrchestrator stores device info in config
                    summary.device_names.push_back("TP_rank_" + std::to_string(i));
                }
            }

            LOG_INFO("[TP Parity] Running with " << summary.tp_degree << " devices");

            // Run prefill
            LOG_INFO("[TP Parity] Calling forward() with " << config_.token_ids.size() << " tokens...");
            bool success = runner_->forward(config_.token_ids.data(), config_.token_ids.size());
            LOG_INFO("[TP Parity] forward() returned: " << (success ? "SUCCESS" : "FAILURE"));
            EXPECT_TRUE(success) << "Prefill forward pass failed";
            if (!success)
                return summary;

            int n_layers = static_cast<int>(model_ctx_->model().block_count);
            size_t seq_len = config_.token_ids.size();
            size_t d_model = model_ctx_->model().embedding_length;
            size_t n_heads = model_ctx_->headCount();
            size_t d_ff = model_ctx_->feedForwardLength();
            size_t vocab_size = model_ctx_->model().vocab_size;

            // Stages to compare per layer with their expected dimensions
            struct StageInfo
            {
                std::string name;
                size_t cols; // Expected columns (may be sharded)
            };
            std::vector<StageInfo> per_layer_stages = {
                {"ATTENTION_NORM", d_model},
                {"ATTENTION_CONTEXT", d_model}, // num_heads * head_dim = d_model
                {"ATTENTION_OUTPUT", d_model},
                {"FFN_NORM", d_model},
                {"FFN_SWIGLU", d_ff},
                {"FFN_DOWN", d_model},
                {"FFN_RESIDUAL", d_model}};

            // Get snapshot keys
            auto snapshot_keys = runner_->getSnapshotKeys();
            std::set<std::string> available_snapshots(snapshot_keys.begin(), snapshot_keys.end());
            LOG_INFO("[TP Parity] Got " << snapshot_keys.size() << " snapshot keys after forward()");
            if (snapshot_keys.size() < 50)
            {
                LOG_WARN("[TP Parity] WARNING: Expected 200+ snapshot keys, got only " << snapshot_keys.size());
                for (const auto &key : snapshot_keys)
                {
                    LOG_INFO("[TP Parity]   Available: " << key);
                }
            }

            // Compare embedding
            auto pytorch_embedding = loadPyTorchSnapshot("EMBEDDING");
            if (available_snapshots.count("EMBEDDING") && !pytorch_embedding.empty())
            {
                TPSnapshot tp_snap = multi_device->getTPSnapshot("EMBEDDING");
                summary.embedding_result = compareTPSnapshot(
                    tp_snap, pytorch_embedding, seq_len, d_model);
            }

            // Compare each layer
            for (int layer_idx = 0; layer_idx < n_layers; ++layer_idx)
            {
                TPLayerStats layer_stats;
                layer_stats.layer_idx = layer_idx;
                layer_stats.tp_degree = summary.tp_degree;

                float sum_combined_cosine = 0.0f;

                for (const auto &stage_info : per_layer_stages)
                {
                    std::string llaminar_key = "layer" + std::to_string(layer_idx) + "_" + stage_info.name;
                    std::string pytorch_key = llaminar_key;

                    if (!available_snapshots.count(llaminar_key))
                        continue;

                    auto pytorch_data = loadPyTorchSnapshot(pytorch_key);
                    if (pytorch_data.empty())
                        continue;

                    TPSnapshot tp_snap = multi_device->getTPSnapshot(llaminar_key);
                    auto stage_result = compareTPSnapshot(
                        tp_snap, pytorch_data, seq_len, stage_info.cols);

                    layer_stats.stage_results.push_back(stage_result);
                    layer_stats.stages_compared++;
                    sum_combined_cosine += stage_result.combined_cosine;

                    if (stage_result.combined_cosine < layer_stats.min_combined_cosine)
                    {
                        layer_stats.min_combined_cosine = stage_result.combined_cosine;
                        layer_stats.worst_stage = stage_info.name;
                    }
                }

                if (layer_stats.stages_compared > 0)
                {
                    layer_stats.avg_combined_cosine = sum_combined_cosine / layer_stats.stages_compared;
                }

                // Pass criteria based on combined cosine
                layer_stats.passed = (layer_stats.avg_combined_cosine >= config_.cosine_threshold);

                summary.layer_stats.push_back(layer_stats);
            }

            // Compare LM_HEAD (always gathered)
            auto pytorch_lm_head = loadPyTorchSnapshot("LM_HEAD");
            if (available_snapshots.count("LM_HEAD") && !pytorch_lm_head.empty())
            {
                TPSnapshot tp_snap = multi_device->getTPSnapshot("LM_HEAD");

                size_t combined_size = 0;
                const float *combined_ptr = tp_snap.getCombinedData(combined_size);

                if (combined_ptr && combined_size > 0)
                {
                    size_t compare_size = std::min(combined_size, pytorch_lm_head.size());
                    summary.lm_head_cosine = computeCosineSimilarity(
                        combined_ptr, pytorch_lm_head.data(), compare_size);

                    size_t lm_seq_len = combined_size / vocab_size;
                    if (lm_seq_len > 0)
                    {
                        size_t last_offset = (lm_seq_len - 1) * vocab_size;
                        summary.lm_head_kl = computeKLDivergence(
                            combined_ptr + last_offset,
                            pytorch_lm_head.data() + last_offset,
                            vocab_size, vocab_size);

                        summary.lm_head_top1 = computeTopKOverlap(
                            combined_ptr + last_offset,
                            pytorch_lm_head.data() + last_offset,
                            vocab_size, vocab_size, 1);

                        summary.lm_head_top5 = computeTopKOverlap(
                            combined_ptr + last_offset,
                            pytorch_lm_head.data() + last_offset,
                            vocab_size, vocab_size, 5);
                    }
                }
            }
            summary.lm_head_passed = (summary.lm_head_kl < config_.kl_threshold);

            // Count early layers passed
            summary.early_layers_passed = summary.embedding_result.passed ? 1 : 0;
            for (int i = 0; i < std::min(config_.early_layers_count, static_cast<int>(summary.layer_stats.size())); ++i)
            {
                if (summary.layer_stats[i].passed)
                    summary.early_layers_passed++;
            }

            // Count total layers passed
            summary.total_layers_passed = summary.embedding_result.passed ? 1 : 0;
            for (const auto &stats : summary.layer_stats)
            {
                if (stats.passed)
                    summary.total_layers_passed++;
            }

            // Overall pass
            summary.overall_passed = (summary.early_layers_passed >= config_.min_early_layers_passed) &&
                                     summary.lm_head_passed;

            return summary;
        }

        /**
         * @brief Run TP-aware incremental decode parity test
         *
         * Tests autoregressive generation by comparing LM_HEAD outputs at each
         * decode step against PyTorch reference. For TP tests, the multi-device
         * orchestrator handles logit gathering internally.
         *
         * Requires:
         * - runner_ to be pre-configured (typically by setupLocalTPPipeline())
         * - PyTorch snapshots with decode_step{N}_LM_HEAD.npy files
         * - metadata.txt with decode_tokens line
         *
         * This is for MULTI-DEVICE LOCAL TP tests.
         * For single-device tests, use runSingleDeviceDecodeParity() instead.
         *
         * @return DecodeParitySummary with per-step and aggregate results
         */
        DecodeParitySummary runTPDecodeParity()
        {
            DecodeParitySummary summary;

            // TP tests require pre-configured runner
            if (!runner_)
            {
                LOG_ERROR("[TP Decode Parity] runner_ is null - "
                          "ensure test calls setupLocalTPPipeline() or similar before runTPDecodeParity()");
                return summary;
            }

            // Verify we have a multi-device orchestrator
            auto *multi_device = dynamic_cast<MultiDeviceOrchestrator *>(runner_.get());
            if (!multi_device)
            {
                LOG_ERROR("[TP Decode Parity] runner_ is not a MultiDeviceOrchestrator - "
                          "ensure test calls setupLocalTPPipeline() or similar before runTPDecodeParity()");
                return summary;
            }

            // Check if decode snapshots exist
            auto decode_step0 = loadPyTorchSnapshot("decode_step0_LM_HEAD");
            if (decode_step0.empty())
            {
                LOG_WARN("Decode snapshots not found - skipping decode parity test");
                return summary;
            }

            // Read expected tokens from metadata
            std::vector<int> pytorch_decode_tokens = readDecodeTokensFromMetadata();
            if (pytorch_decode_tokens.empty())
            {
                LOG_WARN("No decode_tokens in metadata - skipping decode parity test");
                return summary;
            }

            // Run prefill first (required to initialize KV cache)
            bool success = runner_->forward(config_.token_ids.data(), config_.token_ids.size());
            EXPECT_TRUE(success) << "Prefill failed";
            if (!success)
                return summary;

            size_t vocab_size = model_ctx_->model().vocab_size;

            // Process each decode step
            size_t num_decode_steps = std::min(pytorch_decode_tokens.size(),
                                               static_cast<size_t>(config_.decode_steps));

            float sum_cosine = 0.0f;
            float sum_kl = 0.0f;

            for (size_t step = 0; step < num_decode_steps; ++step)
            {
                std::string step_prefix = "decode_step" + std::to_string(step);

                // Load PyTorch reference for this step
                auto pytorch_lm_head = loadPyTorchSnapshot(step_prefix + "_LM_HEAD");
                if (pytorch_lm_head.empty())
                {
                    break; // No more decode snapshots
                }

                summary.steps_total++;

                // Get token for this decode step
                int current_token = pytorch_decode_tokens[step];

                // Clear snapshots from previous step
                runner_->clearSnapshots();

                // Run single-token decode
                std::vector<int> decode_token = {current_token};
                success = runner_->forward(decode_token.data(), 1);
                EXPECT_TRUE(success) << "Decode step " << step << " failed";
                if (!success)
                    continue;

                // Get Llaminar's logits (MultiDeviceOrchestrator gathers from all devices)
                const float *llaminar_logits = runner_->logits();
                if (!llaminar_logits)
                {
                    LOG_WARN("No logits for decode step " << step);
                    continue;
                }

                // Compare with PyTorch
                DecodeStepStats step_stats;
                step_stats.step_idx = static_cast<int>(step);

                step_stats.cosine_similarity = computeCosineSimilarity(
                    llaminar_logits, pytorch_lm_head.data(),
                    std::min(vocab_size, pytorch_lm_head.size()));

                step_stats.kl_divergence = computeKLDivergence(
                    llaminar_logits, pytorch_lm_head.data(),
                    vocab_size, vocab_size);

                step_stats.top1_overlap = computeTopKOverlap(
                    llaminar_logits, pytorch_lm_head.data(),
                    vocab_size, vocab_size, 1);

                // Find argmax tokens
                step_stats.llaminar_token = 0;
                step_stats.pytorch_token = 0;
                float max_l = llaminar_logits[0];
                float max_p = pytorch_lm_head[0];

                for (size_t i = 1; i < vocab_size; ++i)
                {
                    if (llaminar_logits[i] > max_l)
                    {
                        max_l = llaminar_logits[i];
                        step_stats.llaminar_token = static_cast<int>(i);
                    }
                    if (pytorch_lm_head[i] > max_p)
                    {
                        max_p = pytorch_lm_head[i];
                        step_stats.pytorch_token = static_cast<int>(i);
                    }
                }

                step_stats.token_match = (step_stats.llaminar_token == step_stats.pytorch_token);
                if (step_stats.token_match)
                    summary.top1_matches++;

                // Pass criteria: either cosine >= threshold OR KL < threshold
                step_stats.passed = (step_stats.cosine_similarity >= config_.decode_cosine_threshold) ||
                                    (step_stats.kl_divergence < config_.kl_threshold);

                if (step_stats.passed)
                {
                    summary.steps_passed++;
                }

                sum_cosine += step_stats.cosine_similarity;
                sum_kl += step_stats.kl_divergence;

                summary.step_stats.push_back(step_stats);
            }

            // Compute averages and top1 accuracy
            if (summary.steps_total > 0)
            {
                summary.avg_cosine = sum_cosine / summary.steps_total;
                summary.avg_kl = sum_kl / summary.steps_total;
                summary.top1_accuracy = 100.0f * summary.top1_matches / summary.steps_total;
            }

            // Overall pass criteria
            int min_steps_required = static_cast<int>(summary.steps_total * config_.min_decode_pass_rate);
            summary.overall_passed = (summary.steps_passed >= min_steps_required) &&
                                     (summary.top1_accuracy >= config_.min_top1_accuracy) &&
                                     (summary.avg_cosine >= config_.decode_cosine_threshold);

            return summary;
        }

        /**
         * @brief Assert TP parity criteria and render table
         *
         * Call this after runTPPrefillParity() to apply assertions.
         */
        void assertTPParity(const TPParityTestSummary &summary)
        {
            // Render the TP-aware table (rank 0 only)
            if (isRank0())
            {
                renderTPParityTable(summary, config_, getBackendName());
            }

            // Assertions
            EXPECT_GE(summary.early_layers_passed, config_.min_early_layers_passed)
                << "At least " << config_.min_early_layers_passed << " of the first "
                << config_.early_layers_count << " layers should pass TP parity";

            EXPECT_LT(summary.lm_head_kl, config_.kl_threshold)
                << "LM_HEAD KL divergence too high: " << summary.lm_head_kl;
        }

        /**
         * @brief Assert standard parity criteria
         *
         * Call this after runSingleDevicePrefillParity() to apply standard assertions.
         * MPI-aware: Only renders table on rank 0.
         */
        void assertParity(const ParityTestSummary &summary)
        {
            // Render the table first (rank 0 only)
            if (isRank0())
            {
                renderParityTable(summary, config_, getBackendName());
            }

            // Assertions (all ranks - GTest will aggregate failures)
            EXPECT_GE(summary.early_layers_passed, config_.min_early_layers_passed)
                << "At least " << config_.min_early_layers_passed << " of the first "
                << config_.early_layers_count << " layers should pass parity (cosine >= "
                << config_.cosine_threshold << ")";

            EXPECT_LT(summary.lm_head_kl, config_.kl_threshold)
                << "LM_HEAD KL divergence too high: " << summary.lm_head_kl
                << " (threshold: " << config_.kl_threshold << ")";
        }

        // =========================================================================
        // Incremental Decode Parity Testing
        // =========================================================================

        /**
         * @brief Run single-device incremental decode parity test
         *
         * Tests autoregressive generation by comparing LM_HEAD outputs at each
         * decode step against PyTorch reference. Requires:
         * - PyTorch snapshots with decode_step{N}_LM_HEAD.npy files
         * - metadata.txt with decode_tokens line
         *
         * This is for SINGLE-DEVICE tests. Always calls setupPipeline().
         * For multi-device LOCAL TP tests, use runTPDecodeParity() instead.
         *
         * @return DecodeParitySummary with per-step and aggregate results
         */
        DecodeParitySummary runSingleDeviceDecodeParity()
        {
            DecodeParitySummary summary;

            // Check if decode snapshots exist
            auto decode_step0 = loadPyTorchSnapshot("decode_step0_LM_HEAD");
            if (decode_step0.empty())
            {
                LOG_WARN("Decode snapshots not found - skipping decode parity test");
                return summary;
            }

            // Read expected tokens from metadata
            std::vector<int> pytorch_decode_tokens = readDecodeTokensFromMetadata();
            if (pytorch_decode_tokens.empty())
            {
                LOG_WARN("No decode_tokens in metadata - skipping decode parity test");
                return summary;
            }

            // Always setup single-device pipeline
            EXPECT_TRUE(setupPipeline()) << "Pipeline setup failed";
            if (!runner_)
                return summary;

            // Run prefill first (required to initialize KV cache)
            bool success = runner_->forward(config_.token_ids.data(), config_.token_ids.size());
            EXPECT_TRUE(success) << "Prefill failed";
            if (!success)
                return summary;

            size_t vocab_size = model_ctx_->model().vocab_size;

            // Process each decode step
            size_t num_decode_steps = std::min(pytorch_decode_tokens.size(),
                                               static_cast<size_t>(config_.decode_steps));

            float sum_cosine = 0.0f;
            float sum_kl = 0.0f;

            for (size_t step = 0; step < num_decode_steps; ++step)
            {
                std::string step_prefix = "decode_step" + std::to_string(step);

                // Load PyTorch reference for this step
                auto pytorch_lm_head = loadPyTorchSnapshot(step_prefix + "_LM_HEAD");
                if (pytorch_lm_head.empty())
                {
                    break; // No more decode snapshots
                }

                summary.steps_total++;

                // Get token for this decode step
                int current_token = pytorch_decode_tokens[step];

                // Clear snapshots from previous step
                runner_->clearSnapshots();

                // Run single-token decode
                std::vector<int> decode_token = {current_token};
                success = runner_->forward(decode_token.data(), 1);
                EXPECT_TRUE(success) << "Decode step " << step << " failed";
                if (!success)
                    continue;

                // Get Llaminar's LM_HEAD output
                size_t decode_logits_size;
                const float *llaminar_logits = runner_->getSnapshot("LM_HEAD", decode_logits_size);
                if (!llaminar_logits)
                {
                    LOG_WARN("No LM_HEAD snapshot for decode step " << step);
                    continue;
                }

                // Compare with PyTorch
                DecodeStepStats step_stats;
                step_stats.step_idx = static_cast<int>(step);

                step_stats.cosine_similarity = computeCosineSimilarity(
                    llaminar_logits, pytorch_lm_head.data(),
                    std::min(decode_logits_size, pytorch_lm_head.size()));

                step_stats.kl_divergence = computeKLDivergence(
                    llaminar_logits, pytorch_lm_head.data(),
                    decode_logits_size, vocab_size);

                step_stats.top1_overlap = computeTopKOverlap(
                    llaminar_logits, pytorch_lm_head.data(),
                    decode_logits_size, vocab_size, 1);

                // Find argmax tokens
                step_stats.llaminar_token = 0;
                step_stats.pytorch_token = 0;
                float max_l = llaminar_logits[0];
                float max_p = pytorch_lm_head[0];

                for (size_t i = 1; i < vocab_size; ++i)
                {
                    if (llaminar_logits[i] > max_l)
                    {
                        max_l = llaminar_logits[i];
                        step_stats.llaminar_token = static_cast<int>(i);
                    }
                    if (pytorch_lm_head[i] > max_p)
                    {
                        max_p = pytorch_lm_head[i];
                        step_stats.pytorch_token = static_cast<int>(i);
                    }
                }

                step_stats.token_match = (step_stats.llaminar_token == step_stats.pytorch_token);
                if (step_stats.token_match)
                {
                    summary.top1_matches++;
                }

                // Pass criteria: either cosine >= threshold OR KL < threshold
                step_stats.passed = (step_stats.cosine_similarity >= config_.decode_cosine_threshold) ||
                                    (step_stats.kl_divergence < config_.kl_threshold);

                if (step_stats.passed)
                {
                    summary.steps_passed++;
                }

                sum_cosine += step_stats.cosine_similarity;
                sum_kl += step_stats.kl_divergence;

                summary.step_stats.push_back(step_stats);
            }

            // Compute aggregate statistics
            if (summary.steps_total > 0)
            {
                summary.avg_cosine = sum_cosine / summary.steps_total;
                summary.avg_kl = sum_kl / summary.steps_total;
                summary.top1_accuracy = 100.0f * summary.top1_matches / summary.steps_total;
            }

            // Overall pass criteria
            int min_steps_required = static_cast<int>(summary.steps_total * config_.min_decode_pass_rate);
            summary.overall_passed = (summary.steps_passed >= min_steps_required) &&
                                     (summary.top1_accuracy >= config_.min_top1_accuracy) &&
                                     (summary.avg_cosine >= config_.decode_cosine_threshold);

            return summary;
        }

        /**
         * @brief Render decode parity results as Unicode table using libfort
         */
        void renderDecodeParityTable(const DecodeParitySummary &summary, const std::string &backend_name)
        {
            // Helper: format float to string with precision
            auto fmt_f6 = [](float v) -> std::string
            {
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(6) << v;
                return ss.str();
            };

            auto fmt_f4 = [](float v) -> std::string
            {
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(4) << v;
                return ss.str();
            };

            auto fmt_f1 = [](float v) -> std::string
            {
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(1) << v;
                return ss.str();
            };

            auto fmt_f3 = [](float v) -> std::string
            {
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(3) << v;
                return ss.str();
            };

            // Helper: status icon
            auto status_str = [](bool passed) -> std::string
            {
                return passed ? "✓" : "✗";
            };

            std::cout << "\n";

            // =========================================================================
            // Title table
            // =========================================================================
            {
                fort::utf8_table title_table;
                title_table.set_border_style(FT_DOUBLE2_STYLE);

                std::ostringstream title_ss;
                title_ss << backend_name << " INCREMENTAL DECODE PARITY";

                std::ostringstream subtitle_ss;
                subtitle_ss << "Threshold: cosine >= " << fmt_f3(config_.decode_cosine_threshold)
                            << " OR KL < " << fmt_f3(config_.kl_threshold);

                title_table << title_ss.str() << fort::endr;
                title_table << subtitle_ss.str() << fort::endr;

                title_table[0][0].set_cell_text_align(fort::text_align::center);
                title_table[1][0].set_cell_text_align(fort::text_align::center);
                title_table.row(0).set_cell_row_type(fort::row_type::header);

                std::cout << title_table.to_string();
            }

            // =========================================================================
            // Main decode parity table
            // =========================================================================
            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);

            // Header row
            table << fort::header
                  << "Step" << "Cosine" << "KL" << "Llaminar" << "PyTorch" << "OK"
                  << fort::endr;

            // Set column alignments
            table.column(0).set_cell_text_align(fort::text_align::right);
            table.column(1).set_cell_text_align(fort::text_align::right);
            table.column(2).set_cell_text_align(fort::text_align::right);
            table.column(3).set_cell_text_align(fort::text_align::right);
            table.column(4).set_cell_text_align(fort::text_align::right);
            table.column(5).set_cell_text_align(fort::text_align::center);

            // Per-step rows
            for (const auto &step : summary.step_stats)
            {
                std::string match_marker = step.token_match ? " ✓" : " ✗";

                std::ostringstream llaminar_ss;
                llaminar_ss << step.llaminar_token << match_marker;

                table << step.step_idx
                      << fmt_f6(step.cosine_similarity)
                      << fmt_f6(step.kl_divergence)
                      << llaminar_ss.str()
                      << step.pytorch_token
                      << status_str(step.passed)
                      << fort::endr;
            }

            std::cout << table.to_string();

            // =========================================================================
            // Summary footer table
            // =========================================================================
            {
                fort::utf8_table summary_table;
                summary_table.set_border_style(FT_DOUBLE2_STYLE);

                std::ostringstream summary_ss;
                summary_ss << "SUMMARY:  Steps=" << summary.steps_passed << "/" << summary.steps_total
                           << "  AvgCosine=" << fmt_f4(summary.avg_cosine)
                           << "  Top1=" << fmt_f1(summary.top1_accuracy) << "%"
                           << "  " << (summary.overall_passed ? "✓ PASSED" : "✗ FAILED");

                summary_table << summary_ss.str() << fort::endr;
                summary_table[0][0].set_cell_text_align(fort::text_align::center);

                std::cout << summary_table.to_string();
            }

            std::cout << std::endl;
        }

        /**
         * @brief Assert decode parity criteria
         *
         * Call this after runSingleDeviceDecodeParity() or runTPDecodeParity()
         * to apply standard assertions. MPI-aware: Only renders table on rank 0.
         */
        void assertDecodeParity(const DecodeParitySummary &summary)
        {
            // Skip if no decode steps were tested
            if (summary.steps_total == 0)
            {
                GTEST_SKIP() << "No decode snapshots found - skipping decode parity assertions";
            }

            // Render table first (rank 0 only)
            if (isRank0())
            {
                renderDecodeParityTable(summary, getBackendName());
            }

            // Assertions (all ranks - GTest will aggregate failures)
            int min_steps_required = static_cast<int>(summary.steps_total * config_.min_decode_pass_rate);
            EXPECT_GE(summary.steps_passed, min_steps_required)
                << "Not enough decode steps passed: " << summary.steps_passed << "/" << summary.steps_total
                << " (required: " << min_steps_required << ")";

            EXPECT_GE(summary.top1_accuracy, config_.min_top1_accuracy)
                << "Top-1 accuracy too low: " << summary.top1_accuracy << "%"
                << " (required: " << config_.min_top1_accuracy << "%)";

            EXPECT_GE(summary.avg_cosine, config_.decode_cosine_threshold)
                << "Average cosine too low: " << summary.avg_cosine
                << " (required: " << config_.decode_cosine_threshold << ")";
        }
    };

} // namespace llaminar2::test::parity
