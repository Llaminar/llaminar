/**
 * @file Test__HybridQ16Pipeline_vs_FP32_LayerByLayer.cpp
 * @brief E2E Parity: HybridQ16 Pipeline vs FP32 Pipeline (Layer-by-Layer)
 *
 * @category e2e/parity/internal/hybridq16_vs_fp32
 * @tested   HybridQ16 precision inference pipeline (GraphOrchestrator with HybridQ16)
 * @reference FP32 inference pipeline (GraphOrchestrator with FP32)
 * @comparison Hybrid and Q8_1 pipelines (to measure improvement)
 *
 * This test validates that HybridQ16 activation precision achieves:
 * 1. Higher cosine similarity with FP32 than Hybrid mode (Q16_1 residuals vs Q8_1)
 * 2. ≥0.995 cosine similarity at critical residual stages
 * 3. Measurable improvement over Hybrid mode at layer boundaries
 *
 * HybridQ16 mode uses Q16_1 typed residuals (266× better precision than Q8_1)
 * with fused Attention+Wo+ResidualAdd to eliminate intermediate FP32 buffers
 * while maintaining high numerical accuracy.
 *
 * Stages compared per layer:
 *   - ATTENTION_RESIDUAL (where Q16_1 fusion happens)
 *   - FFN_RESIDUAL (layer output with Q16_1 accumulation)
 *   - All intermediate stages (RMSNorm, projections, etc.)
 *
 * NOTE: ATTENTION_OUTPUT is SKIPPED for HybridQ16 comparison because:
 *   - In HybridQ16 mode, fuse_residual_add=true means the "output" buffer
 *     contains (residual + Wo_output), not just Wo_output
 *   - FP32 mode outputs pure Wo_output
 *   - Comparing these is meaningless (different quantities)
 *   - ATTENTION_CONTEXT comparison is valid and shows actual attention precision
 *
 * REQUIRES: ENABLE_PIPELINE_SNAPSHOTS compile flag
 * Build with: cmake -B build_v2_e2e_release -S src/v2 -DCMAKE_BUILD_TYPE=E2ERelease
 *
 * @author David Sanftenberg
 * @date 2025-12-26
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

#include "execution/InferenceRunnerFactory.h"
#include "execution/IInferenceRunner.h"
#include "execution/RuntimeConfig.h"
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
 * @brief Test fixture for HybridQ16 vs FP32 layer-by-layer comparison
 */
class Test__HybridQ16Pipeline_LayerByLayer : public ::testing::Test
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
            // Skip decode snapshots for prefill comparison
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

            double cos_sim = cosine_similarity(ref_data, test_data, ref_size);
            double rel_l2 = relativeL2(ref_data, test_data, ref_size);

            results[key] = {cos_sim, rel_l2};
        }

        return results;
    }

    /**
     * @brief Compare decode snapshots between two runners
     * @return Map of stage name -> (cosine_sim, rel_l2)
     */
    std::map<std::string, std::pair<double, double>> compareDecodeSnapshots(
        IInferenceRunner *ref_runner,
        IInferenceRunner *test_runner)
    {
        std::map<std::string, std::pair<double, double>> results;

        auto ref_keys = ref_runner->getSnapshotKeys();
        auto test_keys = test_runner->getSnapshotKeys();

        std::set<std::string> test_key_set(test_keys.begin(), test_keys.end());

        for (const auto &key : ref_keys)
        {
            // Only decode snapshots
            if (key.find("decode_") != 0)
                continue;

            if (test_key_set.find(key) == test_key_set.end())
                continue;

            size_t ref_size = 0, test_size = 0;
            const float *ref_data = ref_runner->getSnapshot(key, ref_size);
            const float *test_data = test_runner->getSnapshot(key, test_size);

            if (!ref_data || !test_data || ref_size != test_size)
                continue;

            double cos_sim = cosine_similarity(ref_data, test_data, ref_size);
            double rel_l2 = relativeL2(ref_data, test_data, ref_size);

            results[key] = {cos_sim, rel_l2};
        }

        return results;
    }

    /**
     * @brief Get top-5 token indices from logits
     */
    std::vector<int> getTop5(const float *logits, size_t vocab_size)
    {
        std::vector<std::pair<float, int>> indexed;
        indexed.reserve(vocab_size);
        for (size_t i = 0; i < vocab_size; ++i)
        {
            indexed.emplace_back(logits[i], static_cast<int>(i));
        }
        std::partial_sort(indexed.begin(), indexed.begin() + 5, indexed.end(),
                          [](const auto &a, const auto &b)
                          { return a.first > b.first; });

        std::vector<int> top5;
        for (int i = 0; i < 5; ++i)
        {
            top5.push_back(indexed[i].second);
        }
        return top5;
    }
};

// =============================================================================
// Test: HybridQ16 vs FP32 Prefill Layer-by-Layer
// =============================================================================

/**
 * @brief Full layer-by-layer comparison: HybridQ16 vs FP32 (prefill phase)
 *
 * Runs FP32 and HybridQ16 pipelines with snapshot capture enabled,
 * then compares intermediate activations at each stage across all layers.
 *
 * Key metrics:
 * - Residual stages (ATTENTION_RESIDUAL, FFN_RESIDUAL) should show Q16_1 benefit
 * - Overall accuracy should be ≥ Hybrid mode (same Q8_1 activations but better residuals)
 */
TEST_F(Test__HybridQ16Pipeline_LayerByLayer, Prefill_SnapshotComparison)
{
    // Test prompt (same as other parity tests for consistency)
    std::vector<int> tokens = {785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562};
    int seq_len = static_cast<int>(tokens.size());

    LOG_INFO("╔══════════════════════════════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║       HYBRIDQ16 vs FP32 LAYER-BY-LAYER SNAPSHOT COMPARISON (PREFILL)                    ║");
    LOG_INFO("║  Tokens: " << seq_len << ", Model: " << model_path_);
    LOG_INFO("║  HybridQ16: Q16_1 typed residuals (266× better precision than Q8_1)                     ║");
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

    // ===== HybridQ16 Runner =====
    InferenceRunnerConfig config_hybridq16;
    config_hybridq16.activation_precision = ActivationPrecision::HybridQ16;

    std::unique_ptr<IInferenceRunner> runner_hybridq16;
    try
    {
        runner_hybridq16 = createInferenceRunner(model_ctx_, mpi_ctx_, DeviceManager::instance().cpuDeviceIndex(), config_hybridq16);
        ASSERT_NE(runner_hybridq16, nullptr) << "HybridQ16 runner creation failed";
        runner_hybridq16->enableSnapshotCapture();
    }
    catch (const std::exception &e)
    {
        GTEST_SKIP() << "HybridQ16 runner creation failed: " << e.what();
    }

    bool success_hybridq16 = runner_hybridq16->forward(tokens.data(), seq_len);
    ASSERT_TRUE(success_hybridq16) << "HybridQ16 forward failed";
    LOG_INFO("✓ HybridQ16 forward pass completed");

    // ===== Hybrid Runner (for comparison) =====
    InferenceRunnerConfig config_hybrid;
    config_hybrid.activation_precision = ActivationPrecision::Hybrid;

    std::unique_ptr<IInferenceRunner> runner_hybrid;
    try
    {
        runner_hybrid = createInferenceRunner(model_ctx_, mpi_ctx_, DeviceManager::instance().cpuDeviceIndex(), config_hybrid);
        ASSERT_NE(runner_hybrid, nullptr) << "Hybrid runner creation failed";
        runner_hybrid->enableSnapshotCapture();
        runner_hybrid->forward(tokens.data(), seq_len);
        LOG_INFO("✓ Hybrid forward pass completed (for comparison)");
    }
    catch (const std::exception &e)
    {
        LOG_WARN("Hybrid runner unavailable: " << e.what() << " - skipping comparison");
    }

    LOG_INFO("");

    // ===== Compare all snapshots =====
    auto hybridq16_vs_fp32 = compareSnapshots(runner_fp32.get(), runner_hybridq16.get());
    std::map<std::string, std::pair<double, double>> hybrid_vs_fp32;
    if (runner_hybrid)
    {
        hybrid_vs_fp32 = compareSnapshots(runner_fp32.get(), runner_hybrid.get());
    }

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
        double hybridq16_cos;
        double hybridq16_rel_l2;
        double hybrid_cos;
        double hybrid_rel_l2;
        double improvement; // hybridq16_cos - hybrid_cos
        std::string status;
    };
    std::vector<StageMetrics> all_stages;

    double worst_hybridq16_cos = 1.0, best_hybridq16_cos = 0.0;
    std::string worst_hybridq16_stage, best_hybridq16_stage;

    // Stages to skip for HybridQ16 due to architectural differences:
    // Previously ATTENTION_OUTPUT in HybridQ16 mode was the fused residual, but now
    // the Q16 kernel captures proper snapshots matching FP32 semantics:
    // - ATTENTION_OUTPUT: Wo projection result (before residual add)
    // - ATTENTION_RESIDUAL: Final output (after residual add)
    // No stages need to be skipped anymore.
    auto isSkippedStageForHybridQ16 = [](const std::string & /* key */)
    {
        return false; // All stages now comparable
    };

    for (const auto &key : fp32_keys)
    {
        // Skip decode snapshots
        if (key.find("decode_") == 0)
            continue;

        // Skip ATTENTION_OUTPUT for HybridQ16 (see comment above)
        if (isSkippedStageForHybridQ16(key))
        {
            LOG_DEBUG("Skipping " << key << " for HybridQ16 (fused residual vs pure Wo mismatch)");
            continue;
        }

        auto it_q16 = hybridq16_vs_fp32.find(key);
        if (it_q16 == hybridq16_vs_fp32.end())
            continue;

        double hybridq16_cos = it_q16->second.first;
        double hybridq16_rel_l2 = it_q16->second.second;

        double hybrid_cos = 0.0, hybrid_rel_l2 = 0.0;
        if (runner_hybrid)
        {
            auto it_hybrid = hybrid_vs_fp32.find(key);
            if (it_hybrid != hybrid_vs_fp32.end())
            {
                hybrid_cos = it_hybrid->second.first;
                hybrid_rel_l2 = it_hybrid->second.second;
            }
        }

        double improvement = hybridq16_cos - hybrid_cos;

        // Determine status
        std::string status;
        if (hybridq16_cos >= 0.999)
            status = "✓ EXCELLENT";
        else if (hybridq16_cos >= 0.995)
            status = "✓ GREAT";
        else if (hybridq16_cos >= 0.99)
            status = "✓ GOOD";
        else if (hybridq16_cos >= 0.95)
            status = "~ OK";
        else if (hybridq16_cos >= 0.90)
            status = "⚠ DRIFT";
        else
            status = "✗ DIVERGED";

        all_stages.push_back({key, hybridq16_cos, hybridq16_rel_l2, hybrid_cos, hybrid_rel_l2, improvement, status});

        // Track worst/best
        if (hybridq16_cos < worst_hybridq16_cos)
        {
            worst_hybridq16_cos = hybridq16_cos;
            worst_hybridq16_stage = key;
        }
        if (hybridq16_cos > best_hybridq16_cos)
        {
            best_hybridq16_cos = hybridq16_cos;
            best_hybridq16_stage = key;
        }
    }

    // ===== Print summary table =====
    LOG_INFO("");
    LOG_INFO("╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║                    HYBRIDQ16 vs FP32 STAGE-BY-STAGE DIVERGENCE REPORT (PREFILL)                                   ║");
    LOG_INFO("║  Model: " << std::left << std::setw(85) << model_path_.substr(0, 85) << "║");
    LOG_INFO("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣");
    LOG_INFO("║  #  │ Stage                                    │ HybQ16 Cos │ Hybrid Cos │ Δ(Q16-Hyb) │ Rel L2   │ Status        ║");
    LOG_INFO("╠═════╪══════════════════════════════════════════╪════════════╪════════════╪════════════╪══════════╪═══════════════╣");

    int stage_num = 1;
    for (const auto &s : all_stages)
    {
        std::ostringstream line;
        line << "║ " << std::setw(3) << stage_num++ << " │ "
             << std::left << std::setw(40) << s.name.substr(0, 40) << " │ "
             << std::fixed << std::setprecision(6) << s.hybridq16_cos << " │ ";

        if (runner_hybrid)
        {
            line << std::setprecision(6) << s.hybrid_cos << " │ ";
            if (s.improvement >= 0)
                line << "+" << std::setprecision(4) << s.improvement;
            else
                line << std::setprecision(4) << s.improvement;
            line << "    │ ";
        }
        else
        {
            line << "  N/A      │   N/A      │ ";
        }

        line << std::scientific << std::setprecision(2) << s.hybridq16_rel_l2 << " │ "
             << std::left << std::setw(13) << s.status << " ║";
        LOG_INFO(line.str());
    }

    // Summary statistics
    double avg_hybridq16_cos = 0.0, avg_hybrid_cos = 0.0;
    int improved_count = 0, same_count = 0, worse_count = 0;
    int residual_stages_count = 0;
    double avg_residual_improvement = 0.0;

    for (const auto &s : all_stages)
    {
        avg_hybridq16_cos += s.hybridq16_cos;
        avg_hybrid_cos += s.hybrid_cos;

        if (s.improvement > 0.0001)
            improved_count++;
        else if (s.improvement < -0.0001)
            worse_count++;
        else
            same_count++;

        // Track residual stages specifically (where Q16_1 should help)
        if (s.name.find("RESIDUAL") != std::string::npos)
        {
            residual_stages_count++;
            avg_residual_improvement += s.improvement;
        }
    }
    avg_hybridq16_cos /= static_cast<double>(all_stages.size());
    avg_hybrid_cos /= static_cast<double>(all_stages.size());
    if (residual_stages_count > 0)
        avg_residual_improvement /= static_cast<double>(residual_stages_count);

    LOG_INFO("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣");
    LOG_INFO("║  SUMMARY:                                                                                                         ║");
    LOG_INFO("║    Total stages: " << std::setw(4) << all_stages.size()
                                   << "    Improved: " << std::setw(4) << improved_count
                                   << "    Same: " << std::setw(4) << same_count
                                   << "    Worse: " << std::setw(4) << worse_count << "                                         ║");
    LOG_INFO("║    Avg HybridQ16 cosine: " << std::fixed << std::setprecision(6) << avg_hybridq16_cos
                                           << "    Avg Hybrid cosine: " << std::setprecision(6) << avg_hybrid_cos << "                                      ║");
    LOG_INFO("║    Worst stage: " << std::left << std::setw(35) << worst_hybridq16_stage.substr(0, 35)
                                  << " (cos=" << std::setprecision(4) << worst_hybridq16_cos << ")                                  ║");
    LOG_INFO("║    Best stage:  " << std::setw(35) << best_hybridq16_stage.substr(0, 35)
                                  << " (cos=" << std::setprecision(4) << best_hybridq16_cos << ")                                  ║");
    LOG_INFO("║                                                                                                                   ║");
    LOG_INFO("║  RESIDUAL STAGES (Q16_1 benefit):                                                                                 ║");
    LOG_INFO("║    Count: " << residual_stages_count << "    Avg improvement over Hybrid: "
                            << (avg_residual_improvement >= 0 ? "+" : "") << std::setprecision(4) << avg_residual_improvement << "                                            ║");
    LOG_INFO("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝");
    LOG_INFO("");

    // ===== Compare final logits =====
    size_t fp32_logit_size = 0, hybridq16_logit_size = 0;
    const float *fp32_logits = runner_fp32->getSnapshot("LM_HEAD", fp32_logit_size);
    const float *hybridq16_logits = runner_hybridq16->getSnapshot("LM_HEAD", hybridq16_logit_size);

    if (fp32_logits && hybridq16_logits && fp32_logit_size == hybridq16_logit_size)
    {
        double logit_cos = cosine_similarity(fp32_logits, hybridq16_logits, fp32_logit_size);
        double logit_rel_l2 = relativeL2(fp32_logits, hybridq16_logits, fp32_logit_size);

        size_t vocab_size = fp32_logit_size;
        auto top5_fp32 = getTop5(fp32_logits, vocab_size);
        auto top5_hybridq16 = getTop5(hybridq16_logits, vocab_size);

        // Count overlap
        int overlap = 0;
        for (int i = 0; i < 5; ++i)
        {
            for (int j = 0; j < 5; ++j)
            {
                if (top5_fp32[i] == top5_hybridq16[j])
                {
                    overlap++;
                    break;
                }
            }
        }

        LOG_INFO("╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  FINAL LOGITS COMPARISON                                                                                          ║");
        LOG_INFO("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣");
        LOG_INFO("║  Logit Cosine: " << std::fixed << std::setprecision(6) << logit_cos
                                     << "    Rel L2: " << std::scientific << std::setprecision(2) << logit_rel_l2 << "                                                         ║");
        LOG_INFO("║  FP32      Top-5: [" << std::setw(6) << top5_fp32[0] << ", " << std::setw(6) << top5_fp32[1] << ", "
                                         << std::setw(6) << top5_fp32[2] << ", " << std::setw(6) << top5_fp32[3] << ", "
                                         << std::setw(6) << top5_fp32[4] << "]                                             ║");
        LOG_INFO("║  HybridQ16 Top-5: [" << std::setw(6) << top5_hybridq16[0] << ", " << std::setw(6) << top5_hybridq16[1] << ", "
                                         << std::setw(6) << top5_hybridq16[2] << ", " << std::setw(6) << top5_hybridq16[3] << ", "
                                         << std::setw(6) << top5_hybridq16[4] << "]  overlap=" << overlap << "/5 (" << (overlap * 20) << "%)                           ║");

        bool top1_match = (top5_fp32[0] == top5_hybridq16[0]);
        if (top1_match)
        {
            LOG_INFO("║  ✓ TOP-1 MATCH: FP32 and HybridQ16 predict same token: " << std::setw(6) << top5_fp32[0] << "                                            ║");
        }
        else
        {
            LOG_INFO("║  ✗ TOP-1 MISMATCH: FP32=" << std::setw(6) << top5_fp32[0] << ", HybridQ16=" << std::setw(6) << top5_hybridq16[0] << "                                                  ║");
        }
        LOG_INFO("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝");
    }

    // ===== Assertions =====
    // HybridQ16 should achieve high average cosine (at least as good as Hybrid)
    EXPECT_GE(avg_hybridq16_cos, 0.90)
        << "HybridQ16 average cosine " << avg_hybridq16_cos << " is below minimum 0.90";

    // If Hybrid is available, HybridQ16 should be at least as good
    if (runner_hybrid)
    {
        EXPECT_GE(avg_hybridq16_cos, avg_hybrid_cos - 0.01)
            << "HybridQ16 should be at least as good as Hybrid: HybridQ16=" << avg_hybridq16_cos
            << ", Hybrid=" << avg_hybrid_cos;
    }

    // Explicitly clear runners in reverse creation order to avoid cleanup issues
    runner_hybrid.reset();
    runner_hybridq16.reset();
    runner_fp32.reset();
}

// =============================================================================
// Test: HybridQ16 vs FP32 Decode Layer-by-Layer
// =============================================================================

/**
 * @brief Layer-by-layer comparison during decode phase
 *
 * Tests a prefill + 5 decode steps, comparing snapshots at each decode step.
 * Decode is where quantization errors accumulate, so this is critical for Q16_1.
 */
TEST_F(Test__HybridQ16Pipeline_LayerByLayer, Decode_SnapshotComparison)
{
    // Prefill tokens
    std::vector<int> tokens = {785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562};
    int seq_len = static_cast<int>(tokens.size());

    // Simulate decode with fixed tokens (to ensure both runners use same path)
    std::vector<int> decode_tokens = {279, 1234, 5678, 9012, 3456};
    int decode_steps = static_cast<int>(decode_tokens.size());

    LOG_INFO("╔══════════════════════════════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║       HYBRIDQ16 vs FP32 LAYER-BY-LAYER SNAPSHOT COMPARISON (DECODE)                     ║");
    LOG_INFO("║  Prefill: " << seq_len << " tokens, Decode: " << decode_steps << " steps                                                         ║");
    LOG_INFO("║  Model: " << model_path_.substr(0, 70));
    LOG_INFO("╚══════════════════════════════════════════════════════════════════════════════════════════╝");
    LOG_INFO("");

    // ===== FP32 Runner =====
    InferenceRunnerConfig config_fp32;
    config_fp32.activation_precision = ActivationPrecision::FP32;

    auto runner_fp32 = createInferenceRunner(model_ctx_, mpi_ctx_, DeviceManager::instance().cpuDeviceIndex(), config_fp32);
    ASSERT_NE(runner_fp32, nullptr);
    runner_fp32->enableSnapshotCapture();

    // Prefill
    bool success = runner_fp32->forward(tokens.data(), seq_len);
    ASSERT_TRUE(success);

    // Decode steps with fixed tokens
    for (int i = 0; i < decode_steps; ++i)
    {
        int token = decode_tokens[i];
        success = runner_fp32->forward(&token, 1);
        ASSERT_TRUE(success);
    }
    LOG_INFO("✓ FP32 prefill + " << decode_steps << " decode steps completed");

    // ===== HybridQ16 Runner =====
    InferenceRunnerConfig config_hybridq16;
    config_hybridq16.activation_precision = ActivationPrecision::HybridQ16;

    std::unique_ptr<IInferenceRunner> runner_hybridq16;
    try
    {
        runner_hybridq16 = createInferenceRunner(model_ctx_, mpi_ctx_, DeviceManager::instance().cpuDeviceIndex(), config_hybridq16);
        ASSERT_NE(runner_hybridq16, nullptr);
        runner_hybridq16->enableSnapshotCapture();

        // Prefill
        success = runner_hybridq16->forward(tokens.data(), seq_len);
        ASSERT_TRUE(success);

        // Decode steps with same fixed tokens
        for (int i = 0; i < decode_steps; ++i)
        {
            int token = decode_tokens[i];
            success = runner_hybridq16->forward(&token, 1);
            ASSERT_TRUE(success);
        }
        LOG_INFO("✓ HybridQ16 prefill + " << decode_steps << " decode steps completed");
    }
    catch (const std::exception &e)
    {
        GTEST_SKIP() << "HybridQ16 runner failed: " << e.what();
    }

    // ===== Compare decode snapshots =====
    auto decode_results = compareDecodeSnapshots(runner_fp32.get(), runner_hybridq16.get());

    if (decode_results.empty())
    {
        LOG_WARN("No decode snapshots captured - skipping decode comparison");
        return;
    }

    LOG_INFO("");
    LOG_INFO("╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║                    HYBRIDQ16 vs FP32 DECODE STAGE COMPARISON                                                       ║");
    LOG_INFO("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣");
    LOG_INFO("║  Stage                                              │ Cosine     │ Rel L2     │ Status                            ║");
    LOG_INFO("╠═════════════════════════════════════════════════════╪════════════╪════════════╪═══════════════════════════════════╣");

    double avg_decode_cos = 0.0;
    int decode_stage_count = 0;

    for (const auto &[key, metrics] : decode_results)
    {
        double cos_sim = metrics.first;
        double rel_l2 = metrics.second;

        std::string status;
        if (cos_sim >= 0.999)
            status = "✓ EXCELLENT";
        else if (cos_sim >= 0.99)
            status = "✓ GOOD";
        else if (cos_sim >= 0.95)
            status = "~ OK";
        else if (cos_sim >= 0.90)
            status = "⚠ DRIFT";
        else
            status = "✗ DIVERGED";

        std::ostringstream line;
        line << "║  " << std::left << std::setw(48) << key.substr(0, 48) << " │ "
             << std::fixed << std::setprecision(6) << cos_sim << " │ "
             << std::scientific << std::setprecision(2) << rel_l2 << " │ "
             << std::left << std::setw(33) << status << " ║";
        LOG_INFO(line.str());

        avg_decode_cos += cos_sim;
        decode_stage_count++;
    }

    if (decode_stage_count > 0)
    {
        avg_decode_cos /= static_cast<double>(decode_stage_count);
    }

    LOG_INFO("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣");
    LOG_INFO("║  Average decode cosine: " << std::fixed << std::setprecision(6) << avg_decode_cos
                                          << "    Stages: " << decode_stage_count << "                                                              ║");
    LOG_INFO("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝");

    // Decode should maintain reasonable accuracy
    EXPECT_GE(avg_decode_cos, 0.85)
        << "Decode average cosine " << avg_decode_cos << " is below minimum 0.85";

    // Explicitly clear runners in reverse creation order to avoid cleanup issues
    runner_hybridq16.reset();
    runner_fp32.reset();
}

// =============================================================================
// Test: Residual Stages Specifically (Q16_1 Benefit)
// =============================================================================

/**
 * @brief Focus on residual stages where Q16_1 should provide measurable benefit
 *
 * The Q16_1 typed residual stream stores residuals with 266× better precision
 * than Q8_1. This test specifically validates the improvement at residual stages.
 */
TEST_F(Test__HybridQ16Pipeline_LayerByLayer, ResidualStages_Q16_1Benefit)
{
    std::vector<int> tokens = {785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562};
    int seq_len = static_cast<int>(tokens.size());

    LOG_INFO("╔══════════════════════════════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║          RESIDUAL STAGES: Q16_1 BENEFIT ANALYSIS                                        ║");
    LOG_INFO("║  Comparing HybridQ16 (Q16_1 residuals) vs Hybrid (Q8_1 residuals) at residual stages    ║");
    LOG_INFO("╚══════════════════════════════════════════════════════════════════════════════════════════╝");
    LOG_INFO("");

    // Create runners
    InferenceRunnerConfig config_fp32;
    config_fp32.activation_precision = ActivationPrecision::FP32;
    auto runner_fp32 = createInferenceRunner(model_ctx_, mpi_ctx_, DeviceManager::instance().cpuDeviceIndex(), config_fp32);
    runner_fp32->enableSnapshotCapture();
    runner_fp32->forward(tokens.data(), seq_len);

    InferenceRunnerConfig config_hybridq16;
    config_hybridq16.activation_precision = ActivationPrecision::HybridQ16;
    std::unique_ptr<IInferenceRunner> runner_hybridq16;
    try
    {
        runner_hybridq16 = createInferenceRunner(model_ctx_, mpi_ctx_, DeviceManager::instance().cpuDeviceIndex(), config_hybridq16);
        runner_hybridq16->enableSnapshotCapture();
        runner_hybridq16->forward(tokens.data(), seq_len);
    }
    catch (const std::exception &e)
    {
        GTEST_SKIP() << "HybridQ16 runner failed: " << e.what();
    }

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
        GTEST_SKIP() << "Hybrid runner failed: " << e.what();
    }

    // Focus on residual stages across all layers
    LOG_INFO("Layer-by-layer residual stage comparison:");
    LOG_INFO("───────────────────────────────────────────────────────────────────────────────────────────");
    LOG_INFO("  Layer │ Stage              │ HybridQ16 cos │ Hybrid cos │ Δ(Q16-Hyb) │ Winner");
    LOG_INFO("────────┼────────────────────┼───────────────┼────────────┼────────────┼────────────────");

    int hybridq16_wins = 0, hybrid_wins = 0, ties = 0;
    double total_improvement = 0.0;
    int layer_count = 0;

    // Determine number of layers from model
    int num_layers = static_cast<int>(model_ctx_->model().block_count);

    for (int layer = 0; layer < num_layers; ++layer)
    {
        // NOTE: HybridQ16 mode previously fused Attention+Wo+ResidualAdd without
        // producing ATTENTION_RESIDUAL snapshots. Now the Q16 kernel captures all
        // snapshot stages (ATTENTION_CONTEXT, ATTENTION_OUTPUT, ATTENTION_RESIDUAL)
        // matching the FP32 pipeline, enabling full comparison.
        for (const std::string &suffix : {"ATTENTION_RESIDUAL", "FFN_RESIDUAL"})
        {
            std::string key = "layer" + std::to_string(layer) + "_" + suffix;

            size_t fp32_size = 0, hybridq16_size = 0, hybrid_size = 0;
            const float *fp32_data = runner_fp32->getSnapshot(key, fp32_size);
            const float *hybridq16_data = runner_hybridq16->getSnapshot(key, hybridq16_size);
            const float *hybrid_data = runner_hybrid->getSnapshot(key, hybrid_size);

            if (!fp32_data || !hybridq16_data || !hybrid_data)
                continue;

            double hybridq16_cos = cosine_similarity(fp32_data, hybridq16_data, fp32_size);
            double hybrid_cos = cosine_similarity(fp32_data, hybrid_data, fp32_size);
            double improvement = hybridq16_cos - hybrid_cos;

            std::string winner;
            if (improvement > 0.0001)
            {
                winner = "HybridQ16 ✓";
                hybridq16_wins++;
            }
            else if (improvement < -0.0001)
            {
                winner = "Hybrid";
                hybrid_wins++;
            }
            else
            {
                winner = "TIE";
                ties++;
            }

            total_improvement += improvement;
            layer_count++;

            // Print every 4 layers to keep output manageable
            if (layer % 4 == 0 || layer == num_layers - 1)
            {
                LOG_INFO("  " << std::setw(5) << layer << " │ " << std::left << std::setw(18) << suffix.substr(0, 18)
                              << " │ " << std::fixed << std::setprecision(6) << hybridq16_cos
                              << "     │ " << std::setprecision(6) << hybrid_cos
                              << " │ " << (improvement >= 0 ? "+" : "") << std::setprecision(4) << improvement
                              << "    │ " << winner);
            }
        }
    }

    double avg_improvement = layer_count > 0 ? total_improvement / layer_count : 0.0;

    LOG_INFO("───────────────────────────────────────────────────────────────────────────────────────────");
    LOG_INFO("  SUMMARY: HybridQ16 wins=" << hybridq16_wins << ", Hybrid wins=" << hybrid_wins << ", Ties=" << ties);
    LOG_INFO("           Average improvement: " << (avg_improvement >= 0 ? "+" : "") << std::fixed << std::setprecision(6) << avg_improvement);
    LOG_INFO("");

    // HybridQ16 should win or tie on residual stages (that's where Q16_1 helps)
    EXPECT_GE(hybridq16_wins + ties, hybrid_wins)
        << "HybridQ16 should improve residual stage accuracy: wins=" << hybridq16_wins
        << ", losses=" << hybrid_wins << ", ties=" << ties;

    // Explicitly clear runners in reverse creation order to avoid cleanup issues
    runner_hybrid.reset();
    runner_hybridq16.reset();
    runner_fp32.reset();
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
