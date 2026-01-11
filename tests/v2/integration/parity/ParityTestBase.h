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
 * Usage:
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

#include "loaders/ModelContext.h"
#include "execution/InferenceRunnerFactory.h"
#include "execution/IInferenceRunner.h"
#include "kernels/KernelFactory.h"
#include "utils/Logger.h"
#include "backends/DeviceId.h"
#include "backends/ComputeBackend.h"

// NumPy .npy file loading
#include <cnpy.h>

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
    float cosine_threshold = 0.99f;       ///< Minimum avg cosine similarity for layer pass
    bool use_avg_cosine = true;           ///< Use avg (true) or min (false) cosine for pass criteria
    int early_layers_count = 6;           ///< Number of early layers to enforce threshold on
    int min_early_layers_passed = 6;      ///< Minimum early layers that must pass

    // LM_HEAD thresholds
    float kl_threshold = 0.15f;           ///< Maximum KL divergence for logits
    float min_top1_accuracy = 60.0f;      ///< Minimum Top-1 accuracy percentage

    // Decode thresholds (for incremental decode tests)
    float decode_cosine_threshold = 0.99f;
    float min_decode_pass_rate = 0.8f;    ///< Minimum fraction of decode steps that must pass
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
 * @brief Render a formatted parity results table to stdout
 *
 * Produces a consistent Unicode box-drawing table showing:
 * - Per-layer cosine similarity (avg and min)
 * - Worst stage per layer
 * - Pass/fail status with checkmarks
 * - LM_HEAD KL divergence and Top-K accuracy
 */
inline void renderParityTable(
    const ParityTestSummary &summary,
    const ParityConfig &config,
    const std::string &backend_name)
{
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    " << backend_name << " vs PyTorch LAYER-BY-LAYER PARITY"
              << std::string(std::max(0, 37 - static_cast<int>(backend_name.length())), ' ') << "║\n";
    std::cout << "║                    (Threshold: " << (config.use_avg_cosine ? "avg" : "min")
              << " cosine similarity >= " << std::fixed << std::setprecision(3) << config.cosine_threshold
              << ")                      ║\n";
    std::cout << "╠═══════════╦═══════════════╦═══════════════╦════════════════════════════════════════╦══════╣\n";
    std::cout << "║   Layer   ║   Avg Cosine  ║   Min Cosine  ║            Worst Stage                 ║Status║\n";
    std::cout << "╠═══════════╬═══════════════╬═══════════════╬════════════════════════════════════════╬══════╣\n";

    // Embedding
    std::cout << "║ EMBEDDING ║"
              << std::setw(13) << std::fixed << std::setprecision(6) << summary.embedding_cosine << " ║"
              << std::setw(13) << summary.embedding_cosine << " ║"
              << std::setw(39) << "-" << " ║"
              << (summary.embedding_passed ? "  ✓  " : "  ✗  ") << "║\n";

    // Per-layer stats
    for (const auto &stats : summary.layer_stats)
    {
        std::string layer_str = "Layer " + std::to_string(stats.layer_idx);
        std::cout << "║" << std::setw(10) << layer_str << " ║"
                  << std::setw(13) << std::fixed << std::setprecision(6) << stats.avg_cosine_sim << " ║"
                  << std::setw(13) << stats.min_cosine_sim << " ║"
                  << std::setw(39) << stats.worst_stage << " ║"
                  << (stats.passed ? "  ✓  " : "  ✗  ") << "║\n";
    }

    // LM_HEAD
    std::cout << "╠═══════════╬═══════════════╬═══════════════╬════════════════════════════════════════╬══════╣\n";
    std::cout << "║  LM_HEAD  ║"
              << std::setw(13) << std::fixed << std::setprecision(6) << summary.lm_head_cosine << " ║"
              << std::setw(13) << summary.lm_head_cosine << " ║"
              << "    KL=" << std::setw(8) << std::setprecision(4) << summary.lm_head_kl
              << " Top1=" << std::setw(5) << std::setprecision(1) << (summary.lm_head_top1 * 100) << "%" << "      ║"
              << (summary.lm_head_passed ? "  ✓  " : "  ✗  ") << "║\n";

    std::cout << "╚═══════════╩═══════════════╩═══════════════╩════════════════════════════════════════╩══════╝\n";

    // Summary line
    std::cout << "\nLM_HEAD Top-5: " << std::fixed << std::setprecision(1) << (summary.lm_head_top5 * 100.0f) << "%\n";
    std::cout << "Early layers passed: " << summary.early_layers_passed << "/" << config.early_layers_count << "\n";
    std::cout << "LM_HEAD KL divergence: " << summary.lm_head_kl << " (threshold: " << config.kl_threshold << ")\n";
}

// =============================================================================
// Base Test Class
// =============================================================================

/**
 * @brief Base class for PyTorch parity tests
 *
 * Provides common infrastructure for comparing Llaminar backends against
 * PyTorch ground truth. Subclasses must implement:
 * - getDevice() - Return the DeviceId to use for inference
 * - getBackendName() - Return a display name (e.g., "CUDA", "CPU", "ROCm")
 *
 * Optional overrides:
 * - setupDeviceSpecific() - Device-specific initialization (e.g., CUDA checks)
 */
class ParityTestBase : public ::testing::Test
{
protected:
    ParityConfig config_;
    std::shared_ptr<ModelContext> model_ctx_;
    std::unique_ptr<IInferenceRunner> runner_;
    std::unordered_map<std::string, std::vector<float>> pytorch_snapshots_;

    /**
     * @brief Get the device to use for inference
     * @return DeviceId (e.g., DeviceId::cpu(), DeviceId::cuda(0))
     */
    virtual DeviceId getDevice() = 0;

    /**
     * @brief Get the backend name for display
     * @return Name string (e.g., "CUDA", "CPU", "ROCm")
     */
    virtual std::string getBackendName() = 0;

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

        // Regenerate snapshots to ensure consistency
        if (!regeneratePyTorchSnapshots())
        {
            FAIL() << "PyTorch snapshot generation failed";
        }
    }

    void TearDown() override
    {
        model_ctx_.reset();
        runner_.reset();
        pytorch_snapshots_.clear();
        llaminar::v2::kernels::KernelFactory::clearCache();
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
            LOG_ERROR("[Parity] Snapshot generation failed:\n" << output);
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
     * @brief Setup the inference pipeline
     */
    bool setupPipeline()
    {
        DeviceManager::instance().initialize(-1);

        model_ctx_ = ModelContext::create(config_.model_path);
        if (!model_ctx_)
        {
            LOG_ERROR("[Parity] Failed to load model");
            return false;
        }

        InferenceRunnerConfig inf_config;
        inf_config.max_seq_len = 4096;
        inf_config.batch_size = 1;
        inf_config.force_graph = true;

        runner_ = createInferenceRunner(model_ctx_, nullptr, getDevice(), inf_config);
        if (!runner_)
        {
            LOG_ERROR("[Parity] Failed to create inference runner");
            return false;
        }

        runner_->enableSnapshotCapture();
        LOG_INFO("[" << getBackendName() << " Parity] Inference runner created");
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
            if (line.find("decode_step_") == 0)
            {
                size_t eq_pos = line.find('=');
                if (eq_pos != std::string::npos)
                {
                    int token = std::stoi(line.substr(eq_pos + 1));
                    decode_tokens.push_back(token);
                }
            }
        }
        return decode_tokens;
    }

    /**
     * @brief Run prefill parity test and return summary
     *
     * This is the main test driver - compares layer-by-layer against PyTorch.
     */
    ParityTestSummary runPrefillParity()
    {
        ParityTestSummary summary;

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
     * @brief Assert standard parity criteria
     *
     * Call this after runPrefillParity() to apply standard assertions.
     */
    void assertParity(const ParityTestSummary &summary)
    {
        // Render the table first
        renderParityTable(summary, config_, getBackendName());

        // Assertions
        EXPECT_GE(summary.early_layers_passed, config_.min_early_layers_passed)
            << "At least " << config_.min_early_layers_passed << " of the first "
            << config_.early_layers_count << " layers should pass parity (cosine >= "
            << config_.cosine_threshold << ")";

        EXPECT_LT(summary.lm_head_kl, config_.kl_threshold)
            << "LM_HEAD KL divergence too high: " << summary.lm_head_kl
            << " (threshold: " << config_.kl_threshold << ")";
    }
};

} // namespace llaminar2::test::parity
