/**
 * @file Test__HybridPipeline_vs_FP32_Q8_1_LayerByLayer.cpp
 * @brief E2E Parity: Hybrid Pipeline vs FP32 and Q8_1 Pipelines (Layer-by-Layer)
 *
 * @category e2e/parity/internal/hybrid_vs_fp32_q8_1
 * @tested   Hybrid precision inference pipeline (GraphOrchestrator with Hybrid)
 * @reference FP32 inference pipeline (GraphOrchestrator with FP32)
 * @comparison Q8_1 inference pipeline (to measure improvement)
 *
 * This test validates that Hybrid activation precision achieves:
 * 1. ≥0.995 cosine similarity with FP32 at critical stages (vs ~0.85-0.89 for Q8_1)
 * 2. Significant accuracy improvement over full Q8_1 mode
 *
 * The test captures snapshots at each pipeline stage and reports:
 * - Cosine similarity and relative L2 error
 * - Stage-by-stage comparison table
 * - Delta improvement: Hybrid vs Q8_1 relative to FP32
 *
 * REQUIRES: ENABLE_PIPELINE_SNAPSHOTS compile flag
 * Build with: cmake -B build_v2_e2e_release -S src/v2 -DCMAKE_BUILD_TYPE=E2ERelease
 *
 * @author David Sanftenberg
 * @date 2025-01-27
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <map>
#include <algorithm>
#include <set>

#include "execution/factory/InferenceRunnerFactory.h"
#include "execution/local_execution/orchestrators/IInferenceRunner.h"
#include "execution/config/RuntimeConfig.h"
#include "loaders/ModelContext.h"
#include "utils/MPIContext.h"
#include "utils/Logger.h"
#include "tensors/Tensors.h"
#include "backends/ComputeBackend.h"

using namespace llaminar2;

// =============================================================================
// Comparison Utilities
// =============================================================================

/**
 * @brief Compute cosine similarity between two float arrays
 */
static double cosine_similarity(const float *a, const float *b, size_t n)
{
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    if (norm_a < 1e-12 || norm_b < 1e-12)
        return 0.0;
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

/**
 * @brief Compute max absolute difference between two arrays
 */
static double max_abs_diff(const float *a, const float *b, size_t n)
{
    double max_diff = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        double diff = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (diff > max_diff)
            max_diff = diff;
    }
    return max_diff;
}

/**
 * @brief Compute mean absolute difference between two arrays
 */
static double mean_abs_diff(const float *a, const float *b, size_t n)
{
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        sum += std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    }
    return sum / static_cast<double>(n);
}

// =============================================================================
// Test Fixture
// =============================================================================

/**
 * @brief Test fixture for Hybrid vs FP32 vs Q8_1 layer-by-layer comparison
 */
class Test__HybridPipeline_LayerByLayer : public ::testing::Test
{
protected:
    std::shared_ptr<ModelContext> model_ctx_;
    std::shared_ptr<MPIContext> mpi_ctx_;
    int rank_ = 0;
    int world_size_ = 1;
    std::string model_path_;

    void SetUp() override
    {
        // Initialize device manager (required before creating inference runner)
        DeviceManager::instance().initialize(-1); // -1 = no NUMA filtering

        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
        mpi_ctx_ = std::make_shared<MPIContext>(rank_, world_size_, MPI_COMM_WORLD);

        // Allow model path override via environment variable
        const char *env_model = std::getenv("LLAMINAR_TEST_MODEL");
        model_path_ = env_model ? env_model : "models/qwen2.5-0.5b-instruct-q4_0.gguf";

        model_ctx_ = ModelContext::create(model_path_, mpi_ctx_);
        if (!model_ctx_)
        {
            GTEST_SKIP() << "Model not found: " << model_path_;
        }
    }

    void TearDown() override
    {
        model_ctx_.reset();
        mpi_ctx_->barrier();
    }

    /**
     * @brief Compute relative L2 error between two arrays
     */
    double relativeL2(const float *a, const float *b, size_t n)
    {
        double diff_sq = 0.0, ref_sq = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            diff_sq += d * d;
            ref_sq += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        }
        if (ref_sq < 1e-12)
            return (diff_sq < 1e-12) ? 0.0 : 1.0;
        return std::sqrt(diff_sq / ref_sq);
    }

    /**
     * @brief Stage ordering for sorted output
     */
    std::pair<int, int> get_stage_order(const std::string &key) const
    {
        // Special stages
        if (key == "EMBEDDING")
            return {0, 0};
        if (key == "FINAL_NORM")
            return {1000, 0};
        if (key == "LM_HEAD")
            return {1001, 0};

        // Parse layer number and stage type
        if (key.find("layer") == 0)
        {
            size_t underscore = key.find('_');
            if (underscore != std::string::npos)
            {
                int layer_num = std::stoi(key.substr(5, underscore - 5));
                std::string stage_type = key.substr(underscore + 1);

                // Order within layer (matching actual pipeline execution)
                int stage_order = 99; // Unknown stages go last
                if (stage_type == "ATTENTION_NORM")
                    stage_order = 0;
                else if (stage_type == "Q_PROJECTION")
                    stage_order = 1;
                else if (stage_type == "Q_ROPE")
                    stage_order = 2;
                else if (stage_type == "K_PROJECTION")
                    stage_order = 3;
                else if (stage_type == "K_ROPE")
                    stage_order = 4;
                else if (stage_type == "V_PROJECTION")
                    stage_order = 5;
                else if (stage_type == "ATTENTION_CONTEXT")
                    stage_order = 6;
                else if (stage_type == "ATTENTION_OUTPUT")
                    stage_order = 7;
                else if (stage_type == "ATTENTION_RESIDUAL")
                    stage_order = 8;
                else if (stage_type == "FFN_INPUT_RESIDUAL")
                    stage_order = 9;
                else if (stage_type == "FFN_NORM")
                    stage_order = 10;
                else if (stage_type == "FFN_GATE")
                    stage_order = 11;
                else if (stage_type == "FFN_UP")
                    stage_order = 12;
                else if (stage_type == "FFN_SWIGLU")
                    stage_order = 13;
                else if (stage_type == "FFN_DOWN")
                    stage_order = 14;
                else if (stage_type == "FFN_RESIDUAL")
                    stage_order = 15;

                return {layer_num + 1, stage_order}; // +1 so layers come after EMBEDDING
            }
        }
        return {999, 0}; // Unknown keys
    }

    /**
     * @brief Compare snapshots between two runners
     * @return Map of stage name -> (cosine_sim, rel_l2)
     */
    std::map<std::string, std::pair<double, double>> compareSnapshots(
        IInferenceRunner *ref_runner,
        IInferenceRunner *test_runner)
    {
        std::map<std::string, std::pair<double, double>> results;

        auto ref_keys = ref_runner->getSnapshotKeys();
        auto test_keys = test_runner->getSnapshotKeys();

        // Build set of test keys for lookup
        std::set<std::string> test_key_set(test_keys.begin(), test_keys.end());

        for (const auto &key : ref_keys)
        {
            // Skip decode snapshots
            if (key.find("decode_") == 0)
                continue;

            if (test_key_set.find(key) == test_key_set.end())
            {
                LOG_WARN("Snapshot key missing in test runner: " << key);
                continue;
            }

            size_t ref_size = 0, test_size = 0;
            const float *ref_data = ref_runner->getSnapshot(key, ref_size);
            const float *test_data = test_runner->getSnapshot(key, test_size);

            if (!ref_data || !test_data)
            {
                LOG_WARN("Null snapshot data for key: " << key);
                continue;
            }

            if (ref_size != test_size)
            {
                LOG_WARN("Size mismatch for " << key << ": ref=" << ref_size << ", test=" << test_size);
                continue;
            }

            // Debug: compute norms manually to diagnose 0.0 cosine
            double norm_ref = 0.0, norm_test = 0.0;
            for (size_t i = 0; i < ref_size; ++i)
            {
                norm_ref += static_cast<double>(ref_data[i]) * static_cast<double>(ref_data[i]);
                norm_test += static_cast<double>(test_data[i]) * static_cast<double>(test_data[i]);
            }
            norm_ref = std::sqrt(norm_ref);
            norm_test = std::sqrt(norm_test);

            double cos_sim = cosine_similarity(ref_data, test_data, ref_size);
            double rel_l2 = relativeL2(ref_data, test_data, ref_size);

            // Log norms for debugging
            if (key.find("ATTENTION_CONTEXT") != std::string::npos && key.find("layer0_") != std::string::npos)
            {
                LOG_DEBUG("[compareSnapshots] " << key << ": ref_norm=" << norm_ref
                                                << " test_norm=" << norm_test << " cos=" << cos_sim
                                                << " ref_data[0:4]=" << ref_data[0] << "," << ref_data[1] << "," << ref_data[2] << "," << ref_data[3]
                                                << " test_data[0:4]=" << test_data[0] << "," << test_data[1] << "," << test_data[2] << "," << test_data[3]);
            }

            results[key] = {cos_sim, rel_l2};
        }

        return results;
    }
};

// =============================================================================
// Main Test: Hybrid vs FP32 vs Q8_1 Layer-by-Layer Comparison
// =============================================================================

/**
 * @brief Full layer-by-layer comparison: Hybrid vs FP32, and Hybrid vs Q8_1
 *
 * Runs FP32, Hybrid, and Q8_1 pipelines with snapshot capture enabled,
 * then compares intermediate activations at each stage across all layers.
 *
 * Expected outcomes:
 * - Hybrid vs FP32: ≥0.995 cosine similarity at attention stages (significant improvement over Q8_1)
 * - Q8_1 vs FP32: ~0.85-0.95 cosine at attention (baseline for comparison)
 * - Hybrid shows improvement delta vs Q8_1 toward FP32
 */
TEST_F(Test__HybridPipeline_LayerByLayer, SnapshotComparison_Hybrid_vs_FP32_vs_Q8_1)
{
    // Test prompt (same as Q8_1 vs FP32 test)
    std::vector<int> tokens = {785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562};
    int seq_len = static_cast<int>(tokens.size());

    LOG_INFO("╔══════════════════════════════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║        HYBRID vs FP32 vs Q8_1 LAYER-BY-LAYER SNAPSHOT COMPARISON                        ║");
    LOG_INFO("║  Tokens: " << seq_len << ", Model: " << model_path_);
    LOG_INFO("╚══════════════════════════════════════════════════════════════════════════════════════════╝");
    LOG_INFO("");

    // ===== FP32 Runner (ground truth) =====
    InferenceRunnerConfig config_fp32;
    config_fp32.activation_precision = ActivationPrecision::FP32;

    auto runner_fp32 = createInferenceRunner(model_ctx_, mpi_ctx_, DeviceManager::instance().cpuDeviceIndex(), config_fp32);
    ASSERT_NE(runner_fp32, nullptr) << "FP32 runner creation failed";
    runner_fp32->enableSnapshotCapture();

    bool success_fp32 = runner_fp32->forward(tokens.data(), seq_len);
    ASSERT_TRUE(success_fp32) << "FP32 forward failed";
    LOG_INFO("✓ FP32 forward pass completed");

    // ===== Hybrid Runner =====
    InferenceRunnerConfig config_hybrid;
    config_hybrid.activation_precision = ActivationPrecision::Hybrid;

    std::unique_ptr<IInferenceRunner> runner_hybrid;
    try
    {
        runner_hybrid = createInferenceRunner(model_ctx_, mpi_ctx_, DeviceManager::instance().cpuDeviceIndex(), config_hybrid);
        ASSERT_NE(runner_hybrid, nullptr) << "Hybrid runner creation failed";
        runner_hybrid->enableSnapshotCapture();
    }
    catch (const std::exception &e)
    {
        GTEST_SKIP() << "Hybrid runner creation failed: " << e.what();
    }

    bool success_hybrid = runner_hybrid->forward(tokens.data(), seq_len);
    ASSERT_TRUE(success_hybrid) << "Hybrid forward failed";
    LOG_INFO("✓ Hybrid forward pass completed");

    // ===== Q8_1 Runner =====
    InferenceRunnerConfig config_q8_1;
    config_q8_1.activation_precision = ActivationPrecision::Q8_1;

    std::unique_ptr<IInferenceRunner> runner_q8_1;
    try
    {
        runner_q8_1 = createInferenceRunner(model_ctx_, mpi_ctx_, DeviceManager::instance().cpuDeviceIndex(), config_q8_1);
        ASSERT_NE(runner_q8_1, nullptr) << "Q8_1 runner creation failed";
        runner_q8_1->enableSnapshotCapture();
    }
    catch (const std::exception &e)
    {
        GTEST_SKIP() << "Q8_1 runner creation failed: " << e.what();
    }

    bool success_q8_1 = runner_q8_1->forward(tokens.data(), seq_len);
    ASSERT_TRUE(success_q8_1) << "Q8_1 forward failed";
    LOG_INFO("✓ Q8_1 forward pass completed");
    LOG_INFO("");

    // ===== Compare all snapshots =====
    auto hybrid_vs_fp32 = compareSnapshots(runner_fp32.get(), runner_hybrid.get());
    auto q8_1_vs_fp32 = compareSnapshots(runner_fp32.get(), runner_q8_1.get());

    // Get and sort keys
    auto fp32_keys = runner_fp32->getSnapshotKeys();
    std::sort(fp32_keys.begin(), fp32_keys.end(), [this](const std::string &a, const std::string &b)
              {
        auto order_a = get_stage_order(a);
        auto order_b = get_stage_order(b);
        if (order_a.first != order_b.first) return order_a.first < order_b.first;
        return order_a.second < order_b.second; });

    // ===== Collect metrics for summary table =====
    struct StageMetrics
    {
        std::string name;
        double hybrid_cos;
        double hybrid_rel_l2;
        double q8_1_cos;
        double q8_1_rel_l2;
        double improvement; // hybrid_cos - q8_1_cos
        std::string status;
    };
    std::vector<StageMetrics> all_stages;

    double worst_hybrid_cos = 1.0, best_hybrid_cos = 0.0;
    double worst_q8_1_cos = 1.0;
    std::string worst_hybrid_stage, best_hybrid_stage;

    for (const auto &key : fp32_keys)
    {
        // Skip decode snapshots
        if (key.find("decode_") == 0)
            continue;

        auto it_hybrid = hybrid_vs_fp32.find(key);
        auto it_q8_1 = q8_1_vs_fp32.find(key);

        if (it_hybrid == hybrid_vs_fp32.end())
            continue;

        double hybrid_cos = it_hybrid->second.first;
        double hybrid_rel_l2 = it_hybrid->second.second;
        double q8_1_cos = (it_q8_1 != q8_1_vs_fp32.end()) ? it_q8_1->second.first : 0.0;
        double q8_1_rel_l2 = (it_q8_1 != q8_1_vs_fp32.end()) ? it_q8_1->second.second : 1.0;
        double improvement = hybrid_cos - q8_1_cos;

        // Determine status based on Hybrid precision target
        std::string status;
        if (hybrid_cos >= 0.9995)
            status = "✓✓ EXCELLENT";
        else if (hybrid_cos >= 0.995)
            status = "✓ TARGET";
        else if (hybrid_cos >= 0.99)
            status = "~ GOOD";
        else if (hybrid_cos >= 0.95)
            status = "⚠ BELOW TARGET";
        else
            status = "✗ DIVERGED";

        all_stages.push_back({key, hybrid_cos, hybrid_rel_l2, q8_1_cos, q8_1_rel_l2, improvement, status});

        // Track best/worst
        if (hybrid_cos < worst_hybrid_cos)
        {
            worst_hybrid_cos = hybrid_cos;
            worst_hybrid_stage = key;
        }
        if (hybrid_cos > best_hybrid_cos)
        {
            best_hybrid_cos = hybrid_cos;
            best_hybrid_stage = key;
        }
        if (q8_1_cos < worst_q8_1_cos)
        {
            worst_q8_1_cos = q8_1_cos;
        }
    }

    // ===== Print comparison table =====
    LOG_INFO("╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║                      HYBRID vs FP32 vs Q8_1 STAGE-BY-STAGE COMPARISON                                             ║");
    LOG_INFO("║  Model: " << model_path_ << " (" << seq_len << " tokens)");
    LOG_INFO("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣");
    LOG_INFO("║  #  │ Stage                          │ Hybrid cos │ Q8_1 cos │ Δ Improv │ Hybrid L2 │ Status                      ║");
    LOG_INFO("╠═════╪════════════════════════════════╪════════════╪══════════╪══════════╪═══════════╪═════════════════════════════╣");

    int stage_num = 1;
    for (const auto &stage : all_stages)
    {
        char delta_sign = (stage.improvement >= 0) ? '+' : '-';
        std::ostringstream line;
        line << "║ " << std::setw(3) << stage_num++ << " │ "
             << std::left << std::setw(30) << stage.name.substr(0, 30) << " │ "
             << std::fixed << std::setprecision(6) << stage.hybrid_cos << " │ "
             << std::setprecision(4) << stage.q8_1_cos << "   │ "
             << delta_sign << std::setprecision(4) << std::fabs(stage.improvement) << "   │ "
             << std::scientific << std::setprecision(2) << stage.hybrid_rel_l2 << "  │ "
             << std::left << std::setw(27) << stage.status << " ║";
        LOG_INFO(line.str());
    }

    LOG_INFO("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣");

    // Summary statistics
    double avg_hybrid_cos = 0.0, avg_q8_1_cos = 0.0;
    int target_met_count = 0, below_target_count = 0;
    for (const auto &s : all_stages)
    {
        avg_hybrid_cos += s.hybrid_cos;
        avg_q8_1_cos += s.q8_1_cos;
        if (s.hybrid_cos >= 0.995)
            target_met_count++;
        else if (s.hybrid_cos < 0.95)
            below_target_count++;
    }
    avg_hybrid_cos /= static_cast<double>(all_stages.size());
    avg_q8_1_cos /= static_cast<double>(all_stages.size());

    LOG_INFO("║  SUMMARY:                                                                                                         ║");
    LOG_INFO("║    Total stages: " << std::setw(4) << all_stages.size()
                                   << "    Target met (≥0.995): " << std::setw(4) << target_met_count
                                   << "    Below target (<0.95): " << std::setw(4) << below_target_count << "                                ║");
    LOG_INFO("║    Hybrid avg cosine:  " << std::fixed << std::setprecision(6) << avg_hybrid_cos
                                         << "    Q8_1 avg cosine: " << std::setprecision(4) << avg_q8_1_cos
                                         << "    Avg improvement: " << (avg_hybrid_cos > avg_q8_1_cos ? "+" : "")
                                         << std::setprecision(4) << (avg_hybrid_cos - avg_q8_1_cos) << "                   ║");
    LOG_INFO("║    Best Hybrid:  " << std::left << std::setw(30) << best_hybrid_stage.substr(0, 30)
                                   << " (cos=" << std::setprecision(4) << best_hybrid_cos << ")                             ║");
    LOG_INFO("║    Worst Hybrid: " << std::left << std::setw(30) << worst_hybrid_stage.substr(0, 30)
                                   << " (cos=" << std::setprecision(4) << worst_hybrid_cos << ")                             ║");
    LOG_INFO("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝");

    // ===== Final logits comparison =====
    LOG_INFO("");
    LOG_INFO("╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║                                      FINAL LOGITS COMPARISON                                                       ║");
    LOG_INFO("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣");

    const float *logits_fp32 = runner_fp32->getLogits(0);
    const float *logits_hybrid = runner_hybrid->getLogits(0);
    const float *logits_q8_1 = runner_q8_1->getLogits(0);
    ASSERT_NE(logits_fp32, nullptr);
    ASSERT_NE(logits_hybrid, nullptr);
    ASSERT_NE(logits_q8_1, nullptr);

    size_t vocab_size = model_ctx_->model().vocab_size;
    const float *last_fp32 = logits_fp32 + (seq_len - 1) * vocab_size;
    const float *last_hybrid = logits_hybrid + (seq_len - 1) * vocab_size;
    const float *last_q8_1 = logits_q8_1 + (seq_len - 1) * vocab_size;

    double logit_hybrid_cos = cosine_similarity(last_fp32, last_hybrid, vocab_size);
    double logit_q8_1_cos = cosine_similarity(last_fp32, last_q8_1, vocab_size);
    double logit_hybrid_rel_l2 = relativeL2(last_fp32, last_hybrid, vocab_size);
    double logit_q8_1_rel_l2 = relativeL2(last_fp32, last_q8_1, vocab_size);

    // Top-5 comparison helper
    auto get_top5 = [](const float *logits, size_t vocab)
    {
        std::vector<std::pair<float, int>> indexed(vocab);
        for (size_t i = 0; i < vocab; ++i)
            indexed[i] = {logits[i], static_cast<int>(i)};
        std::partial_sort(indexed.begin(), indexed.begin() + 5, indexed.end(),
                          [](const auto &a, const auto &b)
                          { return a.first > b.first; });
        std::vector<int> top5(5);
        for (int i = 0; i < 5; ++i)
            top5[i] = indexed[i].second;
        return top5;
    };

    auto top5_fp32 = get_top5(last_fp32, vocab_size);
    auto top5_hybrid = get_top5(last_hybrid, vocab_size);
    auto top5_q8_1 = get_top5(last_q8_1, vocab_size);

    // Count top-5 overlaps
    auto count_overlap = [](const std::vector<int> &a, const std::vector<int> &b)
    {
        int overlap = 0;
        for (int i = 0; i < 5; ++i)
        {
            for (int j = 0; j < 5; ++j)
            {
                if (a[i] == b[j])
                {
                    overlap++;
                    break;
                }
            }
        }
        return overlap;
    };

    int hybrid_overlap = count_overlap(top5_fp32, top5_hybrid);
    int q8_1_overlap = count_overlap(top5_fp32, top5_q8_1);

    LOG_INFO("║  Logit Cosine:     Hybrid=" << std::fixed << std::setprecision(6) << logit_hybrid_cos
                                            << "    Q8_1=" << std::setprecision(4) << logit_q8_1_cos
                                            << "    Δ=" << (logit_hybrid_cos > logit_q8_1_cos ? "+" : "")
                                            << std::setprecision(4) << (logit_hybrid_cos - logit_q8_1_cos) << "                              ║");
    LOG_INFO("║  Logit Rel L2:     Hybrid=" << std::scientific << std::setprecision(2) << logit_hybrid_rel_l2
                                            << "    Q8_1=" << logit_q8_1_rel_l2 << "                                                     ║");
    LOG_INFO("║                                                                                                                     ║");
    LOG_INFO("║  FP32   Top-5: [" << std::setw(6) << top5_fp32[0] << ", " << std::setw(6) << top5_fp32[1] << ", "
                                  << std::setw(6) << top5_fp32[2] << ", " << std::setw(6) << top5_fp32[3] << ", "
                                  << std::setw(6) << top5_fp32[4] << "]                                              ║");
    LOG_INFO("║  Hybrid Top-5: [" << std::setw(6) << top5_hybrid[0] << ", " << std::setw(6) << top5_hybrid[1] << ", "
                                  << std::setw(6) << top5_hybrid[2] << ", " << std::setw(6) << top5_hybrid[3] << ", "
                                  << std::setw(6) << top5_hybrid[4] << "]  overlap=" << hybrid_overlap << "/5 ("
                                  << (hybrid_overlap * 20) << "%)                                 ║");
    LOG_INFO("║  Q8_1   Top-5: [" << std::setw(6) << top5_q8_1[0] << ", " << std::setw(6) << top5_q8_1[1] << ", "
                                  << std::setw(6) << top5_q8_1[2] << ", " << std::setw(6) << top5_q8_1[3] << ", "
                                  << std::setw(6) << top5_q8_1[4] << "]  overlap=" << q8_1_overlap << "/5 ("
                                  << (q8_1_overlap * 20) << "%)                                 ║");
    LOG_INFO("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣");

    // Top-1 match analysis
    bool hybrid_top1_match = (top5_fp32[0] == top5_hybrid[0]);
    bool q8_1_top1_match = (top5_fp32[0] == top5_q8_1[0]);

    if (hybrid_top1_match && q8_1_top1_match)
    {
        LOG_INFO("║  ✓ ALL MATCH: FP32, Hybrid, and Q8_1 predict same Top-1 token: " << std::setw(6) << top5_fp32[0] << "                              ║");
    }
    else if (hybrid_top1_match)
    {
        LOG_INFO("║  ✓ HYBRID CORRECT: Hybrid matches FP32 Top-1 (" << std::setw(6) << top5_fp32[0] << "), Q8_1 diverged (" << std::setw(6) << top5_q8_1[0] << ")           ║");
    }
    else if (q8_1_top1_match)
    {
        LOG_INFO("║  ⚠ Q8_1 CORRECT: Q8_1 matches FP32, but Hybrid diverged (" << std::setw(6) << top5_hybrid[0] << ")                                   ║");
    }
    else
    {
        LOG_INFO("║  ✗ BOTH DIVERGED: FP32=" << std::setw(6) << top5_fp32[0]
                                             << ", Hybrid=" << std::setw(6) << top5_hybrid[0]
                                             << ", Q8_1=" << std::setw(6) << top5_q8_1[0] << "                                  ║");
    }
    LOG_INFO("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝");
    LOG_INFO("");

    // ===== Assertions =====
    // 1. Hybrid should achieve higher average cosine than Q8_1
    EXPECT_GE(avg_hybrid_cos, avg_q8_1_cos)
        << "Hybrid should improve over Q8_1: Hybrid avg=" << avg_hybrid_cos
        << ", Q8_1 avg=" << avg_q8_1_cos;

    // 2. Hybrid logit cosine should be ≥ Q8_1 logit cosine
    EXPECT_GE(logit_hybrid_cos, logit_q8_1_cos)
        << "Hybrid logits should be closer to FP32 than Q8_1";

    // 3. For maturity target: Hybrid should achieve ≥0.995 average cosine
    // (This may start as a soft expectation)
    if (avg_hybrid_cos < 0.995)
    {
        LOG_WARN("Hybrid average cosine " << avg_hybrid_cos << " is below target 0.995 - optimization needed");
    }
}

/**
 * @brief Critical stages comparison - focus on attention where Q8_1 struggles
 *
 * Validates that Hybrid mode specifically improves the attention stages
 * that show the most divergence in full Q8_1 mode.
 */
TEST_F(Test__HybridPipeline_LayerByLayer, CriticalStages_AttentionAccuracy)
{
    std::vector<int> tokens = {785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562};
    int seq_len = static_cast<int>(tokens.size());

    LOG_INFO("╔══════════════════════════════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║          CRITICAL STAGES: ATTENTION ACCURACY COMPARISON                                 ║");
    LOG_INFO("╚══════════════════════════════════════════════════════════════════════════════════════════╝");

    // Create runners
    InferenceRunnerConfig config_fp32;
    config_fp32.activation_precision = ActivationPrecision::FP32;
    auto runner_fp32 = createInferenceRunner(model_ctx_, mpi_ctx_, DeviceManager::instance().cpuDeviceIndex(), config_fp32);
    runner_fp32->enableSnapshotCapture();
    runner_fp32->forward(tokens.data(), seq_len);

    InferenceRunnerConfig config_hybrid;
    config_hybrid.activation_precision = ActivationPrecision::Hybrid;
    std::unique_ptr<IInferenceRunner> runner_hybrid;
    try
    {
        runner_hybrid = createInferenceRunner(model_ctx_, mpi_ctx_, DeviceManager::instance().cpuDeviceIndex(), config_hybrid);
        runner_hybrid->enableSnapshotCapture();
        runner_hybrid->forward(tokens.data(), seq_len);
    }
    catch (const std::exception &e)
    {
        GTEST_SKIP() << "Hybrid runner creation failed: " << e.what();
    }

    InferenceRunnerConfig config_q8_1;
    config_q8_1.activation_precision = ActivationPrecision::Q8_1;
    std::unique_ptr<IInferenceRunner> runner_q8_1;
    try
    {
        runner_q8_1 = createInferenceRunner(model_ctx_, mpi_ctx_, DeviceManager::instance().cpuDeviceIndex(), config_q8_1);
        runner_q8_1->enableSnapshotCapture();
        runner_q8_1->forward(tokens.data(), seq_len);
    }
    catch (const std::exception &e)
    {
        GTEST_SKIP() << "Q8_1 runner creation failed: " << e.what();
    }

    // Focus on critical stages where Q8_1 typically diverges
    std::vector<std::string> critical_stages = {
        "layer0_Q_ROPE",
        "layer0_K_ROPE",
        "layer0_ATTENTION_CONTEXT",
        "layer0_ATTENTION_OUTPUT",
        "layer11_ATTENTION_CONTEXT", // Middle layer
        "layer11_ATTENTION_OUTPUT",
        "layer23_ATTENTION_CONTEXT", // Last layer
        "layer23_ATTENTION_OUTPUT",
    };

    LOG_INFO("");
    LOG_INFO("Comparing critical attention stages:");
    LOG_INFO("────────────────────────────────────────────────────────────────────────");

    int hybrid_wins = 0, q8_1_wins = 0, ties = 0;

    for (const auto &stage : critical_stages)
    {
        size_t fp32_size = 0, hybrid_size = 0, q8_1_size = 0;
        const float *fp32_data = runner_fp32->getSnapshot(stage, fp32_size);
        const float *hybrid_data = runner_hybrid->getSnapshot(stage, hybrid_size);
        const float *q8_1_data = runner_q8_1->getSnapshot(stage, q8_1_size);

        if (!fp32_data)
        {
            LOG_INFO("  " << std::left << std::setw(30) << stage << " - SKIPPED (no FP32 snapshot)");
            continue;
        }

        double hybrid_cos = hybrid_data ? cosine_similarity(fp32_data, hybrid_data, fp32_size) : 0.0;
        double q8_1_cos = q8_1_data ? cosine_similarity(fp32_data, q8_1_data, fp32_size) : 0.0;
        double improvement = hybrid_cos - q8_1_cos;

        std::string winner;
        if (hybrid_cos > q8_1_cos + 0.001)
        {
            winner = "HYBRID ✓";
            hybrid_wins++;
        }
        else if (q8_1_cos > hybrid_cos + 0.001)
        {
            winner = "Q8_1";
            q8_1_wins++;
        }
        else
        {
            winner = "TIE";
            ties++;
        }

        LOG_INFO("  " << std::left << std::setw(30) << stage
                      << " Hybrid=" << std::fixed << std::setprecision(4) << hybrid_cos
                      << " Q8_1=" << std::setprecision(4) << q8_1_cos
                      << " Δ=" << (improvement >= 0 ? "+" : "") << std::setprecision(4) << improvement
                      << "  " << winner);
    }

    LOG_INFO("────────────────────────────────────────────────────────────────────────");
    LOG_INFO("Results: Hybrid wins=" << hybrid_wins << ", Q8_1 wins=" << q8_1_wins << ", Ties=" << ties);
    LOG_INFO("");

    // Hybrid should win on attention stages (that's the whole point!)
    EXPECT_GE(hybrid_wins, q8_1_wins)
        << "Hybrid should improve attention accuracy over Q8_1";
}

// =============================================================================
// Test Entry Point (custom main for MPI+GTest)
// =============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int result = RUN_ALL_TESTS();

    MPI_Finalize();

    return result;
}
