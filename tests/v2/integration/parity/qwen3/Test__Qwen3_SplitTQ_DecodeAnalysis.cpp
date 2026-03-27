/**
 * @file Test__Qwen3_SplitTQ_DecodeAnalysis.cpp
 * @brief Diagnostic test for Qwen3 SplitTQ decode step 4 divergence analysis
 *
 * This test investigates the root cause of the cosine similarity drop at decode
 * step 4 (cosine 0.824, KL 4.67) when using SplitTQ KV cache. It compares per-layer
 * intermediate activations between Llaminar (SplitTQ and FP16 KV cache) and PyTorch
 * reference snapshots to determine whether the divergence is:
 *
 *   (a) Inherent to SplitTQ 4-bit quantization error amplified through the residual stream
 *   (b) A bug in our SplitTQ quantize/dequantize implementation
 *   (c) Specific to certain layers or attention heads
 *
 * Key findings from prior analysis:
 *   - Step 4 has entropy 2.619 nats (flat distribution, 44.9% confidence)
 *   - PyTorch top-1: token 374 (44.9%), top-2: token 34208 (26.1%)
 *   - Llaminar SplitTQ picks token 34208 (PyTorch's #2)
 *   - FFN_RESIDUAL norms grow 59× from layer 0 to layer 27
 *   - Prior steps (0-3) all pass with cosine > 0.90
 *
 * @author David Sanftenberg
 * @date 2026
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <cnpy.h>
#include <cmath>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>
#include <numeric>

#include <filesystem>
#include <fstream>

#include "Qwen3ParityTestBase.h"
#include "collective/BackendRouter.h"
#include "tensors/Tensors.h"
#include "kernels/cpu/turboquant/TurboQuantContext.h"
#include "kernels/cpu/turboquant/TurboQuantDequantizeTQ4.h"
#include "kernels/cpu/turboquant/TurboQuantDequantizeTQ8.h"

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen3;

// =============================================================================
// Diagnostic Test Fixture
// =============================================================================

class Qwen3SplitTQDecodeAnalysis : public Qwen3ConfigDrivenParityTest<Qwen3SplitTQDecodeAnalysis>
{
public:
    const TestConfig &getTestConfig() const { return test_cfg_; }

protected:
    // Per-layer comparison result
    struct LayerDivergence
    {
        int layer;
        std::string stage;
        float cosine;        // vs PyTorch
        float norm_llaminar; // L2 norm of Llaminar snapshot
        float norm_pytorch;  // L2 norm of PyTorch snapshot
        float max_abs_diff;  // Max absolute difference
        float mean_abs_diff; // Mean absolute difference
    };

    // Run a full decode pass and capture per-layer snapshots at a target step
    struct DecodeRunResult
    {
        std::string label;
        std::vector<LayerDivergence> layer_divergences;
        float lm_head_cosine = 0.0f;
        float lm_head_kl = 0.0f;
        int predicted_token = -1;
        int pytorch_token = -1;
        bool valid = false;
    };

    // Compute cosine similarity between two float vectors
    static float cosine(const float *a, const float *b, size_t n)
    {
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += static_cast<double>(a[i]) * b[i];
            na += static_cast<double>(a[i]) * a[i];
            nb += static_cast<double>(b[i]) * b[i];
        }
        if (na < 1e-30 || nb < 1e-30)
            return 0.0f;
        return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
    }

    // Compute L2 norm
    static float l2norm(const float *a, size_t n)
    {
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i)
            sum += static_cast<double>(a[i]) * a[i];
        return static_cast<float>(std::sqrt(sum));
    }

    // Compute max absolute difference
    static float maxAbsDiff(const float *a, const float *b, size_t n)
    {
        float maxd = 0.0f;
        for (size_t i = 0; i < n; ++i)
            maxd = std::max(maxd, std::abs(a[i] - b[i]));
        return maxd;
    }

    // Compute mean absolute difference
    static float meanAbsDiff(const float *a, const float *b, size_t n)
    {
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i)
            sum += std::abs(static_cast<double>(a[i]) - b[i]);
        return static_cast<float>(sum / n);
    }

    // Compute KL divergence between logit vectors
    static float klDivergence(const float *p_logits, const float *q_logits, size_t vocab_size)
    {
        // Convert logits to log-probabilities via log-softmax
        auto logSoftmax = [](const float *logits, size_t n) -> std::vector<float>
        {
            float max_val = *std::max_element(logits, logits + n);
            double sum_exp = 0.0;
            for (size_t i = 0; i < n; ++i)
                sum_exp += std::exp(static_cast<double>(logits[i] - max_val));
            double log_sum = std::log(sum_exp) + max_val;

            std::vector<float> log_probs(n);
            for (size_t i = 0; i < n; ++i)
                log_probs[i] = static_cast<float>(logits[i] - log_sum);
            return log_probs;
        };

        auto log_p = logSoftmax(p_logits, vocab_size);
        auto log_q = logSoftmax(q_logits, vocab_size);

        // KL(P || Q) = sum_i P_i * (log P_i - log Q_i)
        double kl = 0.0;
        for (size_t i = 0; i < vocab_size; ++i)
        {
            float p_i = std::exp(log_p[i]);
            if (p_i > 1e-10f)
                kl += p_i * (log_p[i] - log_q[i]);
        }
        return static_cast<float>(std::max(0.0, kl));
    }

    // Load a PyTorch snapshot .npy file directly
    std::vector<float> loadNpy(const std::string &path)
    {
        try
        {
            cnpy::NpyArray arr = cnpy::npy_load(path);
            if (arr.word_size == sizeof(float))
            {
                float *ptr = arr.data<float>();
                return {ptr, ptr + arr.num_vals};
            }
            else if (arr.word_size == sizeof(double))
            {
                double *ptr = arr.data<double>();
                std::vector<float> data(arr.num_vals);
                for (size_t i = 0; i < arr.num_vals; ++i)
                    data[i] = static_cast<float>(ptr[i]);
                return data;
            }
        }
        catch (const std::exception &e)
        {
            LOG_WARN("Failed to load " << path << ": " << e.what());
        }
        return {};
    }

    // Run decode and collect per-layer divergence at a target step
    DecodeRunResult runDecodeAndAnalyze(
        const std::string &label,
        KVCachePrecision kv_precision,
        int target_step,
        const std::vector<int> &decode_tokens,
        const std::string &snapshot_dir)
    {
        DecodeRunResult result;
        result.label = label;

        // Configure and setup pipeline
        test_cfg_.kv_cache_precision = kv_precision;

        if (!setupPipeline())
        {
            LOG_ERROR("[Analysis] Pipeline setup failed for " << label);
            return result;
        }

        // Run prefill
        bool success = runner_->forward(config_.token_ids.data(), config_.token_ids.size());
        if (!success)
        {
            LOG_ERROR("[Analysis] Prefill failed for " << label);
            return result;
        }

        // Run decode steps up to and including target
        for (int step = 0; step <= target_step; ++step)
        {
            runner_->clearSnapshots();

            int token = decode_tokens[step];
            success = runner_->forward(&token, 1);
            if (!success)
            {
                LOG_ERROR("[Analysis] Decode step " << step << " failed for " << label);
                return result;
            }
        }

        // Now we have snapshots from the target step
        auto keys = runner_->getSnapshotKeys();
        std::set<std::string> available(keys.begin(), keys.end());

        int n_layers = static_cast<int>(model_ctx_->model().block_count);
        size_t vocab_size = model_ctx_->model().vocab_size;

        // Stages to analyze (ones that reveal SplitTQ impact most)
        std::vector<std::string> analysis_stages = {
            "ATTENTION_CONTEXT",
            "ATTENTION_OUTPUT",
            "ATTENTION_RESIDUAL",
            "FFN_NORM",
            "FFN_DOWN",
            "FFN_RESIDUAL",
        };

        std::string step_prefix = "decode_step" + std::to_string(target_step);

        for (int layer = 0; layer < n_layers; ++layer)
        {
            for (const auto &stage : analysis_stages)
            {
                std::string llaminar_key = "layer" + std::to_string(layer) + "_" + stage;
                std::string pytorch_file = snapshot_dir + "/" + step_prefix +
                                           "_layer" + std::to_string(layer) + "_" + stage + ".npy";

                // Check if Llaminar has this snapshot
                if (available.find(llaminar_key) == available.end())
                    continue;

                // Load PyTorch reference
                auto pytorch_data = loadNpy(pytorch_file);
                if (pytorch_data.empty())
                    continue;

                // Get Llaminar snapshot
                size_t llaminar_size = 0;
                const float *llaminar_data = runner_->getSnapshot(llaminar_key, llaminar_size);
                if (!llaminar_data || llaminar_size == 0)
                    continue;

                size_t compare_size = std::min(llaminar_size, pytorch_data.size());

                LayerDivergence div;
                div.layer = layer;
                div.stage = stage;
                div.cosine = cosine(llaminar_data, pytorch_data.data(), compare_size);
                div.norm_llaminar = l2norm(llaminar_data, compare_size);
                div.norm_pytorch = l2norm(pytorch_data.data(), compare_size);
                div.max_abs_diff = maxAbsDiff(llaminar_data, pytorch_data.data(), compare_size);
                div.mean_abs_diff = meanAbsDiff(llaminar_data, pytorch_data.data(), compare_size);

                result.layer_divergences.push_back(div);
            }
        }

        // LM_HEAD comparison
        if (available.find("LM_HEAD") != available.end())
        {
            size_t lm_size = 0;
            const float *lm_data = runner_->getSnapshot("LM_HEAD", lm_size);
            auto pytorch_lm = loadNpy(snapshot_dir + "/" + step_prefix + "_LM_HEAD.npy");

            if (lm_data && lm_size > 0 && !pytorch_lm.empty())
            {
                size_t compare_size = std::min(lm_size, pytorch_lm.size());
                result.lm_head_cosine = cosine(lm_data, pytorch_lm.data(), compare_size);
                result.lm_head_kl = klDivergence(pytorch_lm.data(), lm_data, vocab_size);

                // Find predicted tokens
                float max_l = lm_data[0], max_p = pytorch_lm[0];
                result.predicted_token = 0;
                result.pytorch_token = 0;
                for (size_t i = 1; i < vocab_size; ++i)
                {
                    if (lm_data[i] > max_l)
                    {
                        max_l = lm_data[i];
                        result.predicted_token = static_cast<int>(i);
                    }
                    if (pytorch_lm[i] > max_p)
                    {
                        max_p = pytorch_lm[i];
                        result.pytorch_token = static_cast<int>(i);
                    }
                }
            }
        }

        result.valid = true;

        // Teardown runner for reuse
        runner_.reset();
        model_ctx_.reset();

        return result;
    }

    // Print a detailed divergence table
    void printDivergenceTable(const DecodeRunResult &result)
    {
        std::cout << "\n"
                  << std::string(100, '=') << "\n";
        std::cout << "  " << result.label << " — Per-Layer Divergence vs PyTorch\n";
        std::cout << std::string(100, '=') << "\n";
        std::cout << "  LM_HEAD: cosine=" << std::fixed << std::setprecision(6) << result.lm_head_cosine
                  << "  KL=" << std::setprecision(4) << result.lm_head_kl
                  << "  predicted=" << result.predicted_token
                  << "  pytorch=" << result.pytorch_token
                  << (result.predicted_token == result.pytorch_token ? "  ✓ MATCH" : "  ✗ MISMATCH")
                  << "\n"
                  << std::string(100, '-') << "\n";

        std::cout << std::left
                  << std::setw(7) << "Layer"
                  << std::setw(22) << "Stage"
                  << std::right
                  << std::setw(10) << "Cosine"
                  << std::setw(12) << "Norm(L)"
                  << std::setw(12) << "Norm(PT)"
                  << std::setw(12) << "MaxAbsDiff"
                  << std::setw(12) << "MeanAbsDiff"
                  << std::setw(8) << "Status"
                  << "\n"
                  << std::string(100, '-') << "\n";

        std::string last_stage;
        for (const auto &d : result.layer_divergences)
        {
            // Print separator between stage groups
            if (!last_stage.empty() && d.stage != last_stage && d.layer == 0)
                std::cout << std::string(100, '.') << "\n";

            std::cout << std::left
                      << std::setw(7) << ("L" + std::to_string(d.layer))
                      << std::setw(22) << d.stage
                      << std::right << std::fixed
                      << std::setw(10) << std::setprecision(6) << d.cosine
                      << std::setw(12) << std::setprecision(2) << d.norm_llaminar
                      << std::setw(12) << std::setprecision(2) << d.norm_pytorch
                      << std::setw(12) << std::setprecision(4) << d.max_abs_diff
                      << std::setw(12) << std::setprecision(4) << d.mean_abs_diff
                      << std::setw(8) << (d.cosine >= 0.99f ? " ✓" : d.cosine >= 0.95f ? " ~"
                                                                                       : " ✗")
                      << "\n";

            last_stage = d.stage;
        }
        std::cout << std::string(100, '=') << "\n\n";
    }

    // Print a side-by-side comparison table for two runs
    void printComparisonTable(const DecodeRunResult &fp16_result, const DecodeRunResult &split_result)
    {
        std::cout << "\n"
                  << std::string(120, '=') << "\n";
        std::cout << "  SIDE-BY-SIDE: FP16 vs SplitTQ KV Cache — Per-Layer Cosine vs PyTorch\n";
        std::cout << std::string(120, '=') << "\n";

        // Build lookup from SplitTQ results
        std::map<std::string, float> split_cosines;
        for (const auto &d : split_result.layer_divergences)
            split_cosines["L" + std::to_string(d.layer) + "_" + d.stage] = d.cosine;

        std::cout << std::left
                  << std::setw(7) << "Layer"
                  << std::setw(22) << "Stage"
                  << std::right
                  << std::setw(12) << "FP16 cos"
                  << std::setw(12) << "SplitTQ cos"
                  << std::setw(12) << "Delta"
                  << std::setw(10) << "FP16"
                  << std::setw(10) << "SplitTQ"
                  << "\n"
                  << std::string(120, '-') << "\n";

        for (const auto &fp16_d : fp16_result.layer_divergences)
        {
            std::string key = "L" + std::to_string(fp16_d.layer) + "_" + fp16_d.stage;
            float split_cos = 0.0f;
            auto it = split_cosines.find(key);
            if (it != split_cosines.end())
                split_cos = it->second;

            float delta = split_cos - fp16_d.cosine;

            std::cout << std::left
                      << std::setw(7) << ("L" + std::to_string(fp16_d.layer))
                      << std::setw(22) << fp16_d.stage
                      << std::right << std::fixed
                      << std::setw(12) << std::setprecision(6) << fp16_d.cosine
                      << std::setw(12) << std::setprecision(6) << split_cos
                      << std::setw(12) << std::setprecision(6) << delta
                      << std::setw(10) << (fp16_d.cosine >= 0.99f ? "✓" : fp16_d.cosine >= 0.95f ? "~"
                                                                                                 : "✗")
                      << std::setw(10) << (split_cos >= 0.99f ? "✓" : split_cos >= 0.95f ? "~"
                                                                                         : "✗")
                      << "\n";
        }

        std::cout << std::string(120, '-') << "\n";
        std::cout << "  FP16 LM_HEAD: cosine=" << std::setprecision(6) << fp16_result.lm_head_cosine
                  << "  KL=" << std::setprecision(4) << fp16_result.lm_head_kl
                  << "  token=" << fp16_result.predicted_token << "\n";
        std::cout << "  SplitTQ  LM_HEAD: cosine=" << std::setprecision(6) << split_result.lm_head_cosine
                  << "  KL=" << std::setprecision(4) << split_result.lm_head_kl
                  << "  token=" << split_result.predicted_token << "\n";
        std::cout << "  PyTorch token: " << fp16_result.pytorch_token << "\n";
        std::cout << std::string(120, '=') << "\n\n";
    }

    // Print summary analysis
    void printAnalysis(const DecodeRunResult &fp16_result, const DecodeRunResult &split_result)
    {
        // Compute where SplitTQ diverges more than FP16
        int split_worse_count = 0;
        int fp16_worse_count = 0;
        float max_split_gap = 0.0f;
        std::string worst_gap_location;

        // Build FP16 lookup
        std::map<std::string, float> fp16_cosines;
        for (const auto &d : fp16_result.layer_divergences)
            fp16_cosines["L" + std::to_string(d.layer) + "_" + d.stage] = d.cosine;

        for (const auto &d : split_result.layer_divergences)
        {
            std::string key = "L" + std::to_string(d.layer) + "_" + d.stage;
            auto it = fp16_cosines.find(key);
            if (it == fp16_cosines.end())
                continue;

            float gap = it->second - d.cosine; // positive = SplitTQ is worse
            if (gap > 0.001f)
                split_worse_count++;
            else if (gap < -0.001f)
                fp16_worse_count++;

            if (gap > max_split_gap)
            {
                max_split_gap = gap;
                worst_gap_location = key;
            }
        }

        // Find first layer where SplitTQ cosine drops below 0.99
        int first_drop_layer = -1;
        std::string first_drop_stage;
        for (const auto &d : split_result.layer_divergences)
        {
            if (d.cosine < 0.99f)
            {
                first_drop_layer = d.layer;
                first_drop_stage = d.stage;
                break;
            }
        }

        // Find last layer where SplitTQ still passes (cosine >= 0.95)
        int last_good_layer = -1;
        for (const auto &d : split_result.layer_divergences)
        {
            if (d.stage == "FFN_RESIDUAL" && d.cosine >= 0.95f)
                last_good_layer = d.layer;
        }

        std::cout << "\n"
                  << std::string(80, '=') << "\n";
        std::cout << "  ANALYSIS SUMMARY\n";
        std::cout << std::string(80, '=') << "\n";
        std::cout << "  Stages where SplitTQ is worse than FP16: " << split_worse_count << "\n";
        std::cout << "  Stages where FP16 is worse than SplitTQ: " << fp16_worse_count << "\n";
        std::cout << "  Worst SplitTQ-vs-FP16 gap: " << std::fixed << std::setprecision(6)
                  << max_split_gap << " at " << worst_gap_location << "\n";

        if (first_drop_layer >= 0)
            std::cout << "  First SplitTQ cosine < 0.99: layer " << first_drop_layer
                      << " " << first_drop_stage << "\n";

        if (last_good_layer >= 0)
            std::cout << "  Last SplitTQ FFN_RESIDUAL cosine >= 0.95: layer " << last_good_layer << "\n";

        // Determine diagnosis
        std::cout << "\n  DIAGNOSIS:\n";

        if (split_worse_count <= 2 && max_split_gap < 0.01f)
        {
            std::cout << "  → SplitTQ and FP16 diverge similarly from PyTorch.\n";
            std::cout << "  → The step 4 divergence is NOT specific to SplitTQ KV quantization.\n";
            std::cout << "  → Root cause: base GEMM quantization error (Q8_0 weights) amplified\n";
            std::cout << "    through 28 layers, reaching a high-entropy decision boundary.\n";
        }
        else if (first_drop_layer >= 0 && first_drop_layer <= 5)
        {
            std::cout << "  → SplitTQ diverges early (layer " << first_drop_layer << ")!\n";
            std::cout << "  → This suggests a BUG in SplitTQ quantize/dequantize.\n";
            std::cout << "  → Investigate K/V quantization at layer " << first_drop_layer << ".\n";
        }
        else if (max_split_gap > 0.05f)
        {
            std::cout << "  → SplitTQ has significant additional divergence vs FP16.\n";
            std::cout << "  → The " << std::setprecision(4) << max_split_gap
                      << " gap at " << worst_gap_location << " is substantial.\n";
            std::cout << "  → SplitTQ quantization error is a meaningful contributor.\n";
        }
        else
        {
            std::cout << "  → SplitTQ adds moderate additional divergence over FP16 baseline.\n";
            std::cout << "  → The divergence is gradual and cumulative (inherent to 4-bit KV).\n";
            std::cout << "  → Token prediction divergence at step 4 results from:\n";
            std::cout << "    1. Base Q8_0 GEMM quantization error (shared with FP16 KV)\n";
            std::cout << "    2. Additional SplitTQ K/V quantization error (4-bit → " << std::setprecision(1)
                      << (max_split_gap * 100.0f) << "% extra cosine loss)\n";
            std::cout << "    3. High-entropy output distribution (44.9% confidence) amplifies both\n";
        }
        std::cout << std::string(80, '=') << "\n\n";
    }

    TestConfig test_cfg_ = {
        .name = "Qwen3_SplitTQ_Analysis",
        .devices = {ParityDeviceType::CPU},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.94f,
            .decode_cosine_threshold = 0.80f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.01f,
        },
        .model_path = "models/Qwen3-0.6B-Q8_0.gguf",
        .snapshot_dir = "pytorch_qwen3_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::TQ, // Will be overridden per-run
    };
};

// =============================================================================
// Test: Layer-by-layer analysis of step 4 divergence
//
// Runs the full pipeline twice (FP16 KV and SplitTQ KV), captures per-layer
// snapshots at step 4, and compares both against PyTorch ground truth.
// If SplitTQ diverges significantly more than FP16, the issue is in SplitTQ.
// If both diverge similarly, the issue is in base Q8_0 GEMM error.
// =============================================================================

TEST_F(Qwen3SplitTQDecodeAnalysis, Step4LayerByLayerDivergence)
{
    // Read decode tokens from metadata
    std::vector<int> decode_tokens = readDecodeTokensFromMetadata();
    ASSERT_GE(decode_tokens.size(), 5u) << "Need at least 5 decode tokens";

    const int target_step = 4; // Step 4 is the problematic one
    const std::string snapshot_dir = "pytorch_qwen3_snapshots";

    std::cout << "\n"
              << std::string(80, '=') << "\n";
    std::cout << "  Qwen3 SplitTQ Decode Step 4 — Layer-by-Layer Divergence Analysis\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "  Model: Qwen3-0.6B-Q8_0.gguf (28 layers, d=1024, d_head=128)\n";
    std::cout << "  Target step: " << target_step << " (token " << decode_tokens[target_step] << ")\n";
    std::cout << "  Decode sequence: ";
    for (size_t i = 0; i < decode_tokens.size(); ++i)
        std::cout << decode_tokens[i] << (i + 1 < decode_tokens.size() ? "," : "");
    std::cout << "\n"
              << std::string(80, '=') << "\n\n";

    // Run 1: FP16 KV cache (baseline — how much does Q8_0 GEMM diverge alone?)
    std::cout << ">>> Running FP16 KV cache baseline...\n";
    auto fp16_result = runDecodeAndAnalyze("FP16 KV Cache", KVCachePrecision::FP16,
                                           target_step, decode_tokens, snapshot_dir);
    ASSERT_TRUE(fp16_result.valid) << "FP16 decode run failed";
    printDivergenceTable(fp16_result);

    // Run 2: SplitTQ KV cache (the one that fails step 4)
    std::cout << ">>> Running SplitTQ KV cache...\n";
    auto split_result = runDecodeAndAnalyze("SplitTQ KV Cache", KVCachePrecision::TQ,
                                            target_step, decode_tokens, snapshot_dir);
    ASSERT_TRUE(split_result.valid) << "SplitTQ decode run failed";
    printDivergenceTable(split_result);

    // Side-by-side comparison
    printComparisonTable(fp16_result, split_result);

    // Analysis
    printAnalysis(fp16_result, split_result);

    // Soft assertions (test should always pass — it's diagnostic)
    // But flag if something is unexpectedly broken
    EXPECT_GT(fp16_result.lm_head_cosine, 0.50f)
        << "FP16 LM_HEAD cosine catastrophically low — possible infrastructure issue";
    EXPECT_GT(split_result.lm_head_cosine, 0.50f)
        << "SplitTQ LM_HEAD cosine catastrophically low — possible infrastructure issue";
}

// =============================================================================
// Test: Track divergence progression across all decode steps
//
// For each decode step 0-4, compare SplitTQ vs FP16 LM_HEAD cosine and token match
// to show how divergence accumulates over the sequence.
// =============================================================================

TEST_F(Qwen3SplitTQDecodeAnalysis, DivergenceProgression)
{
    std::vector<int> decode_tokens = readDecodeTokensFromMetadata();
    ASSERT_GE(decode_tokens.size(), 5u) << "Need at least 5 decode tokens";

    const std::string snapshot_dir = "pytorch_qwen3_snapshots";
    const int num_steps = std::min(static_cast<int>(decode_tokens.size()), 5);

    // Setup SplitTQ pipeline once
    test_cfg_.kv_cache_precision = KVCachePrecision::TQ;
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";

    // Run prefill
    ASSERT_TRUE(runner_->forward(config_.token_ids.data(), config_.token_ids.size()))
        << "Prefill failed";

    size_t vocab_size = model_ctx_->model().vocab_size;

    std::cout << "\n"
              << std::string(90, '=') << "\n";
    std::cout << "  SplitTQ Divergence Progression Across Decode Steps\n";
    std::cout << std::string(90, '=') << "\n";

    std::cout << std::left
              << std::setw(7) << "Step"
              << std::setw(8) << "Token"
              << std::right
              << std::setw(12) << "Cosine"
              << std::setw(10) << "KL"
              << std::setw(10) << "L_token"
              << std::setw(10) << "PT_token"
              << std::setw(10) << "Match"
              << std::setw(12) << "Entropy"
              << std::setw(10) << "Conf%"
              << "\n"
              << std::string(90, '-') << "\n";

    for (int step = 0; step < num_steps; ++step)
    {
        runner_->clearSnapshots();

        int token = decode_tokens[step];
        ASSERT_TRUE(runner_->forward(&token, 1)) << "Decode step " << step << " failed";

        // Get Llaminar logits
        size_t lm_size = 0;
        const float *lm_data = runner_->getSnapshot("LM_HEAD", lm_size);
        ASSERT_NE(lm_data, nullptr) << "No LM_HEAD snapshot at step " << step;

        // Load PyTorch reference
        std::string pytorch_file = snapshot_dir + "/decode_step" + std::to_string(step) + "_LM_HEAD.npy";
        auto pytorch_lm = loadNpy(pytorch_file);
        ASSERT_FALSE(pytorch_lm.empty()) << "Failed to load PyTorch LM_HEAD for step " << step;

        size_t compare_size = std::min(lm_size, pytorch_lm.size());
        float cos_val = cosine(lm_data, pytorch_lm.data(), compare_size);
        float kl_val = klDivergence(pytorch_lm.data(), lm_data, vocab_size);

        // Find tokens
        int l_token = 0, pt_token = 0;
        float max_l = lm_data[0], max_p = pytorch_lm[0];
        for (size_t i = 1; i < vocab_size; ++i)
        {
            if (lm_data[i] > max_l)
            {
                max_l = lm_data[i];
                l_token = static_cast<int>(i);
            }
            if (pytorch_lm[i] > max_p)
            {
                max_p = pytorch_lm[i];
                pt_token = static_cast<int>(i);
            }
        }

        // Compute entropy and confidence of PyTorch distribution
        float pt_max_logit = *std::max_element(pytorch_lm.begin(), pytorch_lm.end());
        double sum_exp = 0.0;
        for (size_t i = 0; i < vocab_size; ++i)
            sum_exp += std::exp(static_cast<double>(pytorch_lm[i] - pt_max_logit));

        double entropy = 0.0;
        float top1_prob = 0.0f;
        for (size_t i = 0; i < vocab_size; ++i)
        {
            double p = std::exp(static_cast<double>(pytorch_lm[i] - pt_max_logit)) / sum_exp;
            if (p > 1e-10)
                entropy -= p * std::log(p);
            if (static_cast<int>(i) == pt_token)
                top1_prob = static_cast<float>(p);
        }

        std::cout << std::left
                  << std::setw(7) << step
                  << std::setw(8) << token
                  << std::right << std::fixed
                  << std::setw(12) << std::setprecision(6) << cos_val
                  << std::setw(10) << std::setprecision(4) << kl_val
                  << std::setw(10) << l_token
                  << std::setw(10) << pt_token
                  << std::setw(10) << (l_token == pt_token ? "✓" : "✗")
                  << std::setw(12) << std::setprecision(3) << entropy
                  << std::setw(10) << std::setprecision(1) << (top1_prob * 100.0f)
                  << "\n";
    }

    std::cout << std::string(90, '=') << "\n\n";
}

// =============================================================================
// Test: FP32 vs SplitTQ pipeline decode step 0 — ALL stages compared
//
// Runs the pipeline TWICE at decode step 0 (first decode after prefill):
//   1. FP32 KV cache (ground truth for Llaminar)
//   2. SplitTQ KV cache (the suspect)
//
// At decode step 0, the ONLY difference should be cache precision:
//   - Embedding: identical (same input token)
//   - Q/K/V projections: identical (depend only on embedding, not cache)
//   - Q_ROPE, K_ROPE: identical
//   - ATTENTION_CONTEXT: diverges (reads cached K/V — FP32 vs SplitTQ)
//   - Everything downstream: propagates from attention divergence
//
// If ATTENTION_CONTEXT divergence is ~0.995 (matching SplitTQ round-trip quality),
// the pipeline is correct and divergence is inherent to 4-bit quantization.
// If ATTENTION_CONTEXT divergence is much worse (e.g., 0.738), there's a bug
// in how the pipeline stores/retrieves/dequantizes SplitTQ KV cache data.
// =============================================================================

TEST_F(Qwen3SplitTQDecodeAnalysis, DecodeStep0_FP32vsSplitTQ_AllStages)
{
    std::vector<int> decode_tokens = readDecodeTokensFromMetadata();
    ASSERT_GE(decode_tokens.size(), 1u) << "Need at least 1 decode token";

    int n_layers = -1;

    // ---- Run 1: FP32 KV cache ----
    std::cout << "\n>>> Running FP32 KV cache (decode step 0)...\n";
    test_cfg_.kv_cache_precision = KVCachePrecision::FP32;
    ASSERT_TRUE(setupPipeline()) << "FP32 pipeline setup failed";

    n_layers = static_cast<int>(model_ctx_->model().block_count);

    // Prefill
    ASSERT_TRUE(runner_->forward(config_.token_ids.data(), config_.token_ids.size()))
        << "FP32 prefill failed";

    // Clear snapshots from prefill, run 1 decode step
    runner_->clearSnapshots();
    int decode_token = decode_tokens[0];
    ASSERT_TRUE(runner_->forward(&decode_token, 1)) << "FP32 decode step 0 failed";

    // Capture ALL snapshots
    auto fp32_keys = runner_->getSnapshotKeys();
    std::map<std::string, std::pair<std::vector<float>, size_t>> fp32_snapshots;
    for (const auto &key : fp32_keys)
    {
        size_t sz = 0;
        const float *data = runner_->getSnapshot(key, sz);
        if (data && sz > 0)
        {
            fp32_snapshots[key] = {std::vector<float>(data, data + sz), sz};
        }
    }

    std::cout << "  Captured " << fp32_snapshots.size() << " FP32 snapshots\n";

    // Teardown FP32 runner
    runner_.reset();
    model_ctx_.reset();

    // ---- Run 2: SplitTQ KV cache ----
    std::cout << ">>> Running SplitTQ KV cache (decode step 0)...\n";
    test_cfg_.kv_cache_precision = KVCachePrecision::TQ;
    ASSERT_TRUE(setupPipeline()) << "SplitTQ pipeline setup failed";

    // Prefill
    ASSERT_TRUE(runner_->forward(config_.token_ids.data(), config_.token_ids.size()))
        << "SplitTQ prefill failed";

    // Clear snapshots from prefill, run 1 decode step
    runner_->clearSnapshots();
    ASSERT_TRUE(runner_->forward(&decode_token, 1)) << "SplitTQ decode step 0 failed";

    // Capture ALL snapshots
    auto tq4_keys = runner_->getSnapshotKeys();
    std::map<std::string, std::pair<std::vector<float>, size_t>> tq4_snapshots;
    for (const auto &key : tq4_keys)
    {
        size_t sz = 0;
        const float *data = runner_->getSnapshot(key, sz);
        if (data && sz > 0)
        {
            tq4_snapshots[key] = {std::vector<float>(data, data + sz), sz};
        }
    }

    std::cout << "  Captured " << tq4_snapshots.size() << " SplitTQ snapshots\n";

    // ---- Compare all stages ----
    // Ordered stage suffixes (pipeline order within each layer)
    const std::vector<std::string> stage_order = {
        "ATTENTION_NORM",
        "Q_PROJECTION",
        "K_PROJECTION",
        "V_PROJECTION",
        "Q_NORM", // Qwen3 only
        "K_NORM", // Qwen3 only
        "Q_ROPE",
        "K_ROPE",
        "ATTENTION_CONTEXT",
        "ATTENTION_OUTPUT",
        "ATTENTION_RESIDUAL",
        "FFN_NORM",
        "FFN_GATE",
        "FFN_UP",
        "FFN_SWIGLU",
        "FFN_DOWN",
        "FFN_RESIDUAL",
    };

    // Global stages first
    const std::vector<std::string> global_stages = {"EMBEDDING", "FINAL_NORM", "LM_HEAD"};

    struct CompareResult
    {
        std::string key;
        float cosine_fp32_vs_tq4;
        float max_abs_diff;
        size_t size;
        bool fp32_available;
        bool tq4_available;
    };

    std::vector<CompareResult> results;

    auto compare = [&](const std::string &key)
    {
        CompareResult r;
        r.key = key;
        r.fp32_available = fp32_snapshots.count(key) > 0;
        r.tq4_available = tq4_snapshots.count(key) > 0;

        if (!r.fp32_available || !r.tq4_available)
        {
            r.cosine_fp32_vs_tq4 = -1.0f;
            r.max_abs_diff = -1.0f;
            r.size = 0;
            results.push_back(r);
            return;
        }

        const auto &fp32_data = fp32_snapshots[key].first;
        const auto &tq4_data = tq4_snapshots[key].first;
        size_t sz = std::min(fp32_data.size(), tq4_data.size());
        r.size = sz;
        r.cosine_fp32_vs_tq4 = cosine(fp32_data.data(), tq4_data.data(), sz);
        r.max_abs_diff = maxAbsDiff(fp32_data.data(), tq4_data.data(), sz);
        results.push_back(r);
    };

    // Compare global stages
    for (const auto &key : global_stages)
        compare(key);

    // Compare per-layer stages
    for (int layer = 0; layer < n_layers; ++layer)
    {
        for (const auto &suffix : stage_order)
        {
            std::string key = "layer" + std::to_string(layer) + "_" + suffix;
            compare(key);
        }
    }

    // ---- Print results ----
    std::cout << "\n"
              << std::string(100, '=') << "\n";
    std::cout << "  DECODE STEP 0: FP32 vs SplitTQ KV Cache — All Stages Compared\n";
    std::cout << "  (FP32 = ground truth, SplitTQ = test subject)\n";
    std::cout << "  Decode token: " << decode_token
              << "  Prefill tokens: " << config_.token_ids.size() << "\n";
    std::cout << std::string(100, '=') << "\n";

    std::cout << std::left
              << std::setw(35) << "Stage"
              << std::right
              << std::setw(14) << "Cosine(F/T)"
              << std::setw(14) << "MaxAbsDiff"
              << std::setw(10) << "Size"
              << std::setw(8) << "Status"
              << "\n"
              << std::string(100, '-') << "\n";

    int first_diverge_layer = -1;
    std::string first_diverge_stage;
    float first_diverge_cosine = 1.0f;

    for (const auto &r : results)
    {
        if (!r.fp32_available || !r.tq4_available)
            continue;

        std::string status;
        if (r.cosine_fp32_vs_tq4 >= 0.9999f)
            status = " =="; // Effectively identical
        else if (r.cosine_fp32_vs_tq4 >= 0.99f)
            status = " ✓";
        else if (r.cosine_fp32_vs_tq4 >= 0.95f)
            status = " ~";
        else
            status = " ✗";

        std::cout << std::left
                  << std::setw(35) << r.key
                  << std::right << std::fixed
                  << std::setw(14) << std::setprecision(6) << r.cosine_fp32_vs_tq4
                  << std::setw(14) << std::setprecision(6) << r.max_abs_diff
                  << std::setw(10) << r.size
                  << std::setw(8) << status
                  << "\n";

        // Track first significant divergence
        if (first_diverge_layer < 0 && r.cosine_fp32_vs_tq4 < 0.9999f &&
            r.key.find("layer") == 0)
        {
            // Extract layer number
            int layer = -1;
            if (sscanf(r.key.c_str(), "layer%d_", &layer) == 1)
            {
                first_diverge_layer = layer;
                first_diverge_stage = r.key;
                first_diverge_cosine = r.cosine_fp32_vs_tq4;
            }
        }
    }

    std::cout << std::string(100, '=') << "\n";

    if (first_diverge_layer >= 0)
    {
        std::cout << "  First divergence: " << first_diverge_stage
                  << " (cosine=" << std::setprecision(6) << first_diverge_cosine << ")\n";
    }
    else
    {
        std::cout << "  No significant divergence detected!\n";
    }

    // Print layer 0 summary specifically (most important for diagnosing the issue)
    std::cout << "\n  LAYER 0 SUMMARY (decode step 0):\n";
    for (const auto &r : results)
    {
        if (r.key.find("layer0_") != 0)
            continue;
        if (!r.fp32_available || !r.tq4_available)
            continue;

        std::string verdict;
        if (r.cosine_fp32_vs_tq4 >= 0.9999f)
            verdict = "IDENTICAL";
        else if (r.cosine_fp32_vs_tq4 >= 0.995f)
            verdict = "EXPECTED (within SplitTQ round-trip tolerance)";
        else if (r.cosine_fp32_vs_tq4 >= 0.95f)
            verdict = "SUSPICIOUS (worse than SplitTQ round-trip)";
        else
            verdict = "BUG (much worse than SplitTQ round-trip of ~0.995)";

        std::cout << "    " << std::left << std::setw(30) << r.key
                  << std::right << std::fixed << std::setprecision(6)
                  << r.cosine_fp32_vs_tq4 << "  " << verdict << "\n";
    }

    std::cout << std::string(100, '=') << "\n\n";

    // Key diagnostic assertions
    // Q/K/V projections at L0 should be IDENTICAL (they don't read from cache)
    for (const auto &r : results)
    {
        if (r.key == "layer0_ATTENTION_NORM" ||
            r.key == "layer0_Q_PROJECTION" ||
            r.key == "layer0_K_PROJECTION" ||
            r.key == "layer0_V_PROJECTION" ||
            r.key == "layer0_Q_NORM" ||
            r.key == "layer0_K_NORM" ||
            r.key == "layer0_Q_ROPE" ||
            r.key == "layer0_K_ROPE")
        {
            if (r.fp32_available && r.tq4_available)
            {
                EXPECT_GE(r.cosine_fp32_vs_tq4, 0.9999f)
                    << r.key << " should be IDENTICAL at L0 decode step 0 "
                    << "(not affected by KV cache precision)";
            }
        }
    }

    // ATTENTION_CONTEXT at L0 is the KEY diagnostic
    for (const auto &r : results)
    {
        if (r.key == "layer0_ATTENTION_CONTEXT")
        {
            if (r.fp32_available && r.tq4_available)
            {
                // If this is ~0.995, SplitTQ pipeline is correct (error = round-trip loss)
                // If this is <<0.99, there's a pipeline bug
                EXPECT_GE(r.cosine_fp32_vs_tq4, 0.98f)
                    << "layer0_ATTENTION_CONTEXT has cosine " << r.cosine_fp32_vs_tq4
                    << " which is MUCH worse than SplitTQ round-trip quality (~0.995). "
                    << "This indicates a pipeline bug, not just quantization noise.";
            }
        }
    }

    // Teardown
    runner_.reset();
    model_ctx_.reset();
}

// =============================================================================
// Test: SplitTQ round-trip quality on REAL model K/V projections
//
// Loads the Qwen3-0.6B model, runs prefill with FP32 KV cache to capture
// real K and V projections from each layer, then measures SplitTQ quantize →
// dequantize round-trip fidelity on those actual tensors.
//
// This distinguishes between:
//   (a) SplitTQ works fine on random data but fails on real activations
//       (data distribution problem — e.g., outliers, heavy tails)
//   (b) SplitTQ round-trip is ~0.99 even on real data, so the divergence
//       is cumulative amplification through 28 layers, not per-layer quality
// =============================================================================

TEST_F(Qwen3SplitTQDecodeAnalysis, SplitTQRoundTripOnRealModelTensors)
{
    // --- Run prefill with FP32 KV cache to get real K/V projections ---
    test_cfg_.kv_cache_precision = KVCachePrecision::FP32;
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";

    bool success = runner_->forward(config_.token_ids.data(), config_.token_ids.size());
    ASSERT_TRUE(success) << "Prefill failed";

    auto keys = runner_->getSnapshotKeys();
    std::set<std::string> available(keys.begin(), keys.end());

    int n_layers = static_cast<int>(model_ctx_->model().block_count);
    int head_dim = static_cast<int>(model_ctx_->model().key_length);
    int n_kv_heads = static_cast<int>(model_ctx_->model().head_count_kv);
    if (head_dim == 0)
        head_dim = static_cast<int>(model_ctx_->model().embedding_length / model_ctx_->model().head_count);
    int kv_dim = n_kv_heads * head_dim;
    int seq_len = static_cast<int>(config_.token_ids.size());

    std::cout << "\n"
              << std::string(100, '=') << "\n";
    std::cout << "  SplitTQ Round-Trip Quality on Real Qwen3 K/V Projections\n";
    std::cout << std::string(100, '=') << "\n";
    std::cout << "  Model: " << test_cfg_.model_path
              << " (" << n_layers << " layers, head_dim=" << head_dim
              << ", n_kv_heads=" << n_kv_heads << ", kv_dim=" << kv_dim << ")\n";
    std::cout << "  Prefill tokens: " << seq_len << "\n";
    std::cout << std::string(100, '=') << "\n\n";

    // Create TurboQuantContext with same params as production pipeline
    TurboQuantContext tq_ctx(head_dim); // default seeds (42, 42)

    struct PerLayerResult
    {
        int layer;
        std::string tensor_name; // "K" or "V"
        float overall_cosine;
        float worst_head_cosine;
        int worst_head_idx;
        float best_head_cosine;
        float mean_head_cosine;
        float data_norm;
        float data_max_abs;
    };

    std::vector<PerLayerResult> results;

    for (int layer = 0; layer < n_layers; ++layer)
    {
        // Derive the per-layer context (matching pipeline: root -> for_layer(layer_idx))
        const auto &layer_ctx = tq_ctx.for_layer(layer);

        for (const char *proj : {"K", "V"})
        {
            std::string key = "layer" + std::to_string(layer) + "_" + proj + "_PROJECTION";
            if (available.find(key) == available.end())
                continue;

            size_t snap_size = 0;
            const float *fp32_data = runner_->getSnapshot(key, snap_size);
            if (!fp32_data || snap_size == 0)
                continue;

            // Shape: [seq_len, kv_dim]
            size_t expected_size = static_cast<size_t>(seq_len) * kv_dim;
            if (snap_size < expected_size)
            {
                // Might be smaller if projection shape differs
                expected_size = snap_size;
            }
            size_t actual_rows = expected_size / static_cast<size_t>(kv_dim);
            if (actual_rows == 0)
                continue;

            // Compute data statistics
            float data_norm = l2norm(fp32_data, actual_rows * kv_dim);
            float data_max_abs = 0.0f;
            for (size_t i = 0; i < actual_rows * kv_dim; ++i)
                data_max_abs = std::max(data_max_abs, std::abs(fp32_data[i]));

            // SplitTQ round-trip: K uses TQ8, V uses TQ4
            std::vector<size_t> shape = {actual_rows, static_cast<size_t>(kv_dim)};
            std::vector<float> reconstructed(actual_rows * kv_dim);

            bool is_k = (std::string(proj) == "K");
            if (is_k)
            {
                auto tq8_tensor = TQ8Tensor::quantize_from_fp32(
                    fp32_data, shape, head_dim, layer_ctx);
                tq8_tensor->dequantize_to_fp32(reconstructed.data(), layer_ctx);
            }
            else
            {
                auto tq4_tensor = TQ4Tensor::quantize_from_fp32(
                    fp32_data, shape, head_dim, layer_ctx);
                tq4_tensor->dequantize_to_fp32(reconstructed.data(), layer_ctx);
            }

            // Overall cosine similarity
            float overall_cos = cosine(fp32_data, reconstructed.data(),
                                       actual_rows * kv_dim);

            // Per-head cosine similarity (across ALL positions for each head)
            float worst_head_cos = 1.0f;
            float best_head_cos = 0.0f;
            int worst_head_idx = 0;
            double head_cos_sum = 0.0;

            for (int h = 0; h < n_kv_heads; ++h)
            {
                // Gather all positions for this head
                std::vector<float> head_orig(actual_rows * head_dim);
                std::vector<float> head_recon(actual_rows * head_dim);
                for (size_t r = 0; r < actual_rows; ++r)
                {
                    const float *orig_row = fp32_data + r * kv_dim + h * head_dim;
                    const float *recon_row = reconstructed.data() + r * kv_dim + h * head_dim;
                    std::memcpy(head_orig.data() + r * head_dim, orig_row,
                                head_dim * sizeof(float));
                    std::memcpy(head_recon.data() + r * head_dim, recon_row,
                                head_dim * sizeof(float));
                }

                float head_cos = cosine(head_orig.data(), head_recon.data(),
                                        actual_rows * head_dim);
                head_cos_sum += head_cos;

                if (head_cos < worst_head_cos)
                {
                    worst_head_cos = head_cos;
                    worst_head_idx = h;
                }
                if (head_cos > best_head_cos)
                    best_head_cos = head_cos;
            }

            float mean_head_cos = static_cast<float>(head_cos_sum / n_kv_heads);

            results.push_back({layer, std::string(proj), overall_cos,
                               worst_head_cos, worst_head_idx, best_head_cos,
                               mean_head_cos, data_norm, data_max_abs});
        }
    }

    // --- Print results ---
    std::cout << std::left
              << std::setw(7) << "Layer"
              << std::setw(4) << "T"
              << std::right
              << std::setw(12) << "Overall"
              << std::setw(12) << "WorstHead"
              << std::setw(8) << "WH#"
              << std::setw(12) << "BestHead"
              << std::setw(12) << "MeanHead"
              << std::setw(12) << "L2Norm"
              << std::setw(12) << "MaxAbs"
              << std::setw(8) << "Status"
              << "\n"
              << std::string(100, '-') << "\n";

    int bad_count = 0;
    float worst_overall = 1.0f;
    std::string worst_location;

    for (const auto &r : results)
    {
        bool ok = r.overall_cosine >= 0.99f;
        if (!ok)
            bad_count++;
        if (r.overall_cosine < worst_overall)
        {
            worst_overall = r.overall_cosine;
            worst_location = "L" + std::to_string(r.layer) + "_" + r.tensor_name;
        }

        std::cout << std::left
                  << std::setw(7) << ("L" + std::to_string(r.layer))
                  << std::setw(4) << r.tensor_name
                  << std::right << std::fixed
                  << std::setw(12) << std::setprecision(6) << r.overall_cosine
                  << std::setw(12) << std::setprecision(6) << r.worst_head_cosine
                  << std::setw(8) << r.worst_head_idx
                  << std::setw(12) << std::setprecision(6) << r.best_head_cosine
                  << std::setw(12) << std::setprecision(6) << r.mean_head_cosine
                  << std::setw(12) << std::setprecision(2) << r.data_norm
                  << std::setw(12) << std::setprecision(4) << r.data_max_abs
                  << std::setw(8) << (ok ? " ✓" : " ✗")
                  << "\n";
    }

    std::cout << std::string(100, '=') << "\n";
    std::cout << "  Layers with overall cosine < 0.99: " << bad_count
              << " / " << results.size() << "\n";
    std::cout << "  Worst overall: " << std::setprecision(6) << worst_overall
              << " at " << worst_location << "\n";
    std::cout << std::string(100, '=') << "\n\n";

    // Soft assertion: SplitTQ round-trip should maintain reasonable quality
    // If this fails, SplitTQ has a data-distribution-dependent quality problem
    EXPECT_GE(worst_overall, 0.95f)
        << "SplitTQ round-trip quality catastrophically bad at " << worst_location
        << " — suggests data distribution issue";

    // Cleanup for subsequent tests
    runner_.reset();
    model_ctx_.reset();
}

// =============================================================================
// Test: Per-position SplitTQ round-trip on real L0 K projections
//
// Drills into a single layer's K projection to check per-position (per-token)
// round-trip quality. This reveals if certain token positions have much worse
// SplitTQ fidelity than others (e.g., BOS token, repeated tokens, etc.)
// =============================================================================

TEST_F(Qwen3SplitTQDecodeAnalysis, SplitTQRoundTripPerPosition_Layer0_K)
{
    test_cfg_.kv_cache_precision = KVCachePrecision::FP32;
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";

    bool success = runner_->forward(config_.token_ids.data(), config_.token_ids.size());
    ASSERT_TRUE(success) << "Prefill failed";

    int head_dim = static_cast<int>(model_ctx_->model().key_length);
    int n_kv_heads = static_cast<int>(model_ctx_->model().head_count_kv);
    if (head_dim == 0)
        head_dim = static_cast<int>(model_ctx_->model().embedding_length / model_ctx_->model().head_count);
    int kv_dim = n_kv_heads * head_dim;

    size_t snap_size = 0;
    const float *fp32_data = runner_->getSnapshot("layer0_K_PROJECTION", snap_size);
    ASSERT_NE(fp32_data, nullptr) << "No layer0_K_PROJECTION snapshot";
    ASSERT_GT(snap_size, 0u);

    size_t num_tokens = snap_size / static_cast<size_t>(kv_dim);
    ASSERT_GT(num_tokens, 0u);

    TurboQuantContext tq_ctx(head_dim);
    const auto &layer_ctx = tq_ctx.for_layer(0);

    // Quantize K as TQ8 (production split regime) and dequantize
    std::vector<size_t> shape = {num_tokens, static_cast<size_t>(kv_dim)};
    auto tq8_tensor = TQ8Tensor::quantize_from_fp32(fp32_data, shape, head_dim, layer_ctx);

    std::vector<float> reconstructed(num_tokens * kv_dim);
    tq8_tensor->dequantize_to_fp32(reconstructed.data(), layer_ctx);

    std::cout << "\n"
              << std::string(90, '=') << "\n";
    std::cout << "  SplitTQ Per-Position Round-Trip: Layer 0 K Projection (TQ8)\n";
    std::cout << "  Tokens: " << num_tokens << ", head_dim=" << head_dim
              << ", n_kv_heads=" << n_kv_heads << "\n";
    std::cout << std::string(90, '=') << "\n";

    std::cout << std::left
              << std::setw(8) << "Pos"
              << std::right
              << std::setw(12) << "Cosine"
              << std::setw(12) << "L2 Norm"
              << std::setw(12) << "MaxAbs"
              << std::setw(12) << "MaxErr"
              << std::setw(8) << "Status"
              << "\n"
              << std::string(90, '-') << "\n";

    float worst_pos_cos = 1.0f;
    int worst_pos = 0;
    double cos_sum = 0.0;

    for (size_t pos = 0; pos < num_tokens; ++pos)
    {
        const float *orig = fp32_data + pos * kv_dim;
        const float *recon = reconstructed.data() + pos * kv_dim;

        float pos_cos = cosine(orig, recon, static_cast<size_t>(kv_dim));
        float pos_norm = l2norm(orig, static_cast<size_t>(kv_dim));
        float pos_max_abs = 0.0f;
        float pos_max_err = 0.0f;
        for (int i = 0; i < kv_dim; ++i)
        {
            pos_max_abs = std::max(pos_max_abs, std::abs(orig[i]));
            pos_max_err = std::max(pos_max_err, std::abs(orig[i] - recon[i]));
        }

        cos_sum += pos_cos;
        if (pos_cos < worst_pos_cos)
        {
            worst_pos_cos = pos_cos;
            worst_pos = static_cast<int>(pos);
        }

        // Print every position if small, or sample if large
        bool should_print = (num_tokens <= 20) ||
                            (pos < 5) || (pos >= num_tokens - 3) ||
                            (pos_cos < 0.98f);
        if (should_print)
        {
            std::cout << std::left
                      << std::setw(8) << pos
                      << std::right << std::fixed
                      << std::setw(12) << std::setprecision(6) << pos_cos
                      << std::setw(12) << std::setprecision(4) << pos_norm
                      << std::setw(12) << std::setprecision(4) << pos_max_abs
                      << std::setw(12) << std::setprecision(4) << pos_max_err
                      << std::setw(8) << (pos_cos >= 0.99f ? " ✓" : pos_cos >= 0.95f ? " ~"
                                                                                     : " ✗")
                      << "\n";
        }
    }

    float mean_cos = static_cast<float>(cos_sum / num_tokens);

    std::cout << std::string(90, '-') << "\n";
    std::cout << "  Mean cosine: " << std::setprecision(6) << mean_cos
              << "  Worst: " << worst_pos_cos
              << " at position " << worst_pos << "\n";
    std::cout << std::string(90, '=') << "\n\n";

    EXPECT_GE(worst_pos_cos, 0.90f) << "SplitTQ catastrophically bad at position " << worst_pos;
    EXPECT_GE(mean_cos, 0.98f) << "SplitTQ mean round-trip quality too low";

    runner_.reset();
    model_ctx_.reset();
}

// =============================================================================
// Test: Empirical SplitTQ cache flow — simulates exact pipeline data path
//
// This test captures the REAL K_ROPE data from FP32 prefill, then simulates
// the EXACT cache flow:
//   1. quantize_from_fp32 (KVCacheAppendStage)
//   2. memcpy into a cache-sized buffer (CPURingKVCache::append_kv_impl)
//   3. turboquant_dequantize_kv_rows from cache buffer (AttentionComputeStage)
// and compares against:
//   (A) Direct SplitTQ round-trip (quantize → dequantize, no cache buffer)
//   (B) The real SplitTQ pipeline's ATTENTION_CONTEXT snapshot from decode step 0
//
// This isolates whether the bug is in:
//   - SplitTQ quantize/dequant math (measured by A)
//   - Cache buffer storage/retrieval (measured by A vs simulated cache)
//   - Something else in the pipeline (measured by B vs simulated cache)
// =============================================================================

TEST_F(Qwen3SplitTQDecodeAnalysis, SplitTQCacheFlowSimulation_Layer0)
{
    // ---- Step 1: Run FP32 pipeline, capture K_ROPE and V_PROJECTION from prefill ----
    std::cout << "\n>>> Step 1: Running FP32 prefill to capture reference K/V data...\n";
    test_cfg_.kv_cache_precision = KVCachePrecision::FP32;
    ASSERT_TRUE(setupPipeline()) << "FP32 pipeline setup failed";

    int n_layers = static_cast<int>(model_ctx_->model().block_count);
    int head_dim = static_cast<int>(model_ctx_->model().key_length);
    int n_kv_heads = static_cast<int>(model_ctx_->model().head_count_kv);
    if (head_dim == 0)
        head_dim = static_cast<int>(model_ctx_->model().embedding_length / model_ctx_->model().head_count);
    int kv_dim = n_kv_heads * head_dim;
    int seq_len = static_cast<int>(config_.token_ids.size());

    std::cout << "  Model: n_layers=" << n_layers << " head_dim=" << head_dim
              << " n_kv_heads=" << n_kv_heads << " kv_dim=" << kv_dim
              << " seq_len=" << seq_len << "\n";

    // FP32 prefill captures snapshots
    ASSERT_TRUE(runner_->forward(config_.token_ids.data(), config_.token_ids.size()))
        << "FP32 prefill failed";

    // Capture K_ROPE for layer 0 (this is what goes into the KV cache during prefill)
    size_t k_rope_size = 0;
    const float *k_rope_fp32 = runner_->getSnapshot("layer0_K_ROPE", k_rope_size);
    ASSERT_NE(k_rope_fp32, nullptr) << "No layer0_K_ROPE snapshot from FP32 prefill";
    ASSERT_GT(k_rope_size, 0u);

    // Also get V data (V_PROJECTION - no RoPE on V)
    size_t v_proj_size = 0;
    const float *v_proj_fp32 = runner_->getSnapshot("layer0_V_PROJECTION", v_proj_size);
    ASSERT_NE(v_proj_fp32, nullptr) << "No layer0_V_PROJECTION snapshot from FP32 prefill";

    // Copy the reference data (pipeline teardown will invalidate pointers)
    std::vector<float> ref_k_rope(k_rope_fp32, k_rope_fp32 + k_rope_size);
    std::vector<float> ref_v_proj(v_proj_fp32, v_proj_fp32 + v_proj_size);

    // Also run 1 decode step with FP32 to capture FP32 ATTENTION_CONTEXT as ground truth
    std::vector<int> decode_tokens = readDecodeTokensFromMetadata();
    ASSERT_GE(decode_tokens.size(), 1u) << "Need at least 1 decode token";
    int decode_token = decode_tokens[0];

    runner_->clearSnapshots();
    ASSERT_TRUE(runner_->forward(&decode_token, 1)) << "FP32 decode step 0 failed";

    // Capture FP32 decode step 0 K_ROPE (this is the single new token's K after RoPE)
    size_t decode_k_size = 0;
    const float *decode_k_fp32 = runner_->getSnapshot("layer0_K_ROPE", decode_k_size);
    std::vector<float> ref_decode_k;
    if (decode_k_fp32 && decode_k_size > 0)
        ref_decode_k.assign(decode_k_fp32, decode_k_fp32 + decode_k_size);

    // Capture FP32 ATTENTION_CONTEXT from decode step 0 as ultimate ground truth
    size_t attn_ctx_fp32_size = 0;
    const float *attn_ctx_fp32 = runner_->getSnapshot("layer0_ATTENTION_CONTEXT", attn_ctx_fp32_size);
    std::vector<float> ref_attn_context;
    if (attn_ctx_fp32 && attn_ctx_fp32_size > 0)
        ref_attn_context.assign(attn_ctx_fp32, attn_ctx_fp32 + attn_ctx_fp32_size);

    runner_.reset();
    model_ctx_.reset();

    // ---- Step 2: SplitTQ round-trip WITHOUT cache buffer (direct quantize → dequant) ----
    std::cout << ">>> Step 2: Direct SplitTQ round-trip (no cache buffer)...\n";

    TurboQuantContext tq_ctx(head_dim); // default seeds (42, 42) — same as pipeline
    const auto &layer0_ctx = tq_ctx.for_layer(0);

    // Quantize K_ROPE to TQ8 (production split regime: K uses TQ8)
    size_t k_rows = k_rope_size / static_cast<size_t>(kv_dim);
    std::vector<size_t> k_shape = {k_rows, static_cast<size_t>(kv_dim)};
    auto k_tq8_direct = TQ8Tensor::quantize_from_fp32(ref_k_rope.data(), k_shape, head_dim, layer0_ctx);
    ASSERT_NE(k_tq8_direct, nullptr) << "TQ8 quantization of K failed";

    // Dequantize using TQ8Tensor::dequantize_to_fp32
    std::vector<float> k_direct_dequant(k_rows * kv_dim);
    k_tq8_direct->dequantize_to_fp32(k_direct_dequant.data(), layer0_ctx);

    float direct_cosine = cosine(ref_k_rope.data(), k_direct_dequant.data(), k_rows * kv_dim);
    std::cout << "  Direct K round-trip cosine (TQ8): " << std::fixed << std::setprecision(6) << direct_cosine << "\n";

    // V uses TQ4 (production split regime)
    size_t v_rows = v_proj_size / static_cast<size_t>(kv_dim);
    std::vector<size_t> v_shape = {v_rows, static_cast<size_t>(kv_dim)};
    auto v_tq4_direct = TQ4Tensor::quantize_from_fp32(ref_v_proj.data(), v_shape, head_dim, layer0_ctx);
    ASSERT_NE(v_tq4_direct, nullptr);

    std::vector<float> v_direct_dequant(v_rows * kv_dim);
    v_tq4_direct->dequantize_to_fp32(v_direct_dequant.data(), layer0_ctx);

    float v_direct_cosine = cosine(ref_v_proj.data(), v_direct_dequant.data(), v_rows * kv_dim);
    std::cout << "  Direct V round-trip cosine: " << std::fixed << std::setprecision(6) << v_direct_cosine << "\n";

    // ---- Step 3: Simulate cache buffer flow (what the pipeline actually does) ----
    std::cout << ">>> Step 3: Simulated cache buffer flow...\n";

    // The pipeline does:
    //   1. KVCacheAppendStage: quantize_from_fp32 → TQ8Tensor(K) / TQ4Tensor(V)
    //   2. CPURingKVCache::append_kv: memcpy rows from source tensors into
    //      pre-allocated cache TQ8Tensor(K) / TQ4Tensor(V) of (max_seq_len, kv_dim)
    //   3. AttentionComputeStage: read from cache, separate TQ8 K / TQ4 V dequant

    const int max_seq_len = 2048; // typical cache size

    // Create cache-sized tensors (K=TQ8, V=TQ4)
    auto k_cache = std::make_shared<TQ8Tensor>(
        std::vector<size_t>{static_cast<size_t>(max_seq_len), static_cast<size_t>(kv_dim)},
        head_dim);
    auto v_cache = std::make_shared<TQ4Tensor>(
        std::vector<size_t>{static_cast<size_t>(max_seq_len), static_cast<size_t>(kv_dim)},
        head_dim);

    // Simulate append: memcpy each row from source to cache
    // This matches CPURingKVCache::copy_position_major_token
    size_t k_row_bytes = static_cast<size_t>(kv_dim / head_dim) * k_tq8_direct->block_bytes();
    size_t v_row_bytes = static_cast<size_t>(kv_dim / head_dim) * v_tq4_direct->block_bytes();

    // Verify row_bytes matches the cache tensor's layout
    size_t cache_k_row_bytes = static_cast<size_t>(kv_dim / head_dim) * k_cache->block_bytes();
    size_t cache_v_row_bytes = static_cast<size_t>(kv_dim / head_dim) * v_cache->block_bytes();
    ASSERT_EQ(k_row_bytes, cache_k_row_bytes) << "K row_bytes mismatch between source and cache";
    ASSERT_EQ(v_row_bytes, cache_v_row_bytes) << "V row_bytes mismatch between source and cache";

    std::cout << "  K (TQ8): blocks_per_row=" << k_tq8_direct->blocks_per_row()
              << " block_bytes=" << k_tq8_direct->block_bytes()
              << " row_bytes=" << k_row_bytes << "\n";
    std::cout << "  V (TQ4): blocks_per_row=" << v_tq4_direct->blocks_per_row()
              << " block_bytes=" << v_tq4_direct->block_bytes()
              << " row_bytes=" << v_row_bytes << "\n";
    std::cout << "  Cache K (TQ8): blocks_per_row=" << k_cache->blocks_per_row()
              << " block_bytes=" << k_cache->block_bytes() << "\n";

    const auto *k_src = reinterpret_cast<const uint8_t *>(k_tq8_direct->raw_data());
    const auto *v_src = reinterpret_cast<const uint8_t *>(v_tq4_direct->raw_data());
    auto *k_dst = reinterpret_cast<uint8_t *>(k_cache->raw_mutable_data());
    auto *v_dst = reinterpret_cast<uint8_t *>(v_cache->raw_mutable_data());

    for (size_t row = 0; row < k_rows; ++row)
    {
        std::memcpy(k_dst + row * cache_k_row_bytes, k_src + row * k_row_bytes, k_row_bytes);
        std::memcpy(v_dst + row * cache_v_row_bytes, v_src + row * v_row_bytes, v_row_bytes);
    }

    // Verify byte-for-byte equality between source and cache for row 0
    bool byte_match = (std::memcmp(k_src, k_dst, k_row_bytes) == 0);
    EXPECT_TRUE(byte_match) << "CRITICAL: First row of K cache doesn't match source!";
    std::cout << "  Row 0 byte-for-byte match: " << (byte_match ? "YES" : "NO!!! BUG!!!") << "\n";

    // ---- Step 4: Dequantize from cache using separate TQ8 K / TQ4 V paths ----
    // This is EXACTLY what AttentionComputeStage does in split regime
    std::cout << ">>> Step 4: Dequantize from cache buffer (pipeline-exact, split TQ8/TQ4)...\n";

    std::vector<float> k_cache_dequant(k_rows * kv_dim, 0.0f);
    std::vector<float> v_cache_dequant(v_rows * kv_dim, 0.0f);

    // K dequant via TQ8Tensor
    k_cache->set_turboquant_context(&tq_ctx);
    k_cache->dequantize_to_fp32(k_cache_dequant.data(), layer0_ctx);

    // V dequant via turboquant_dequantize_v_rows (TQ4)
    turboquant_dequantize_v_rows(
        v_cache->typed_data(),
        layer0_ctx,
        v_cache_dequant.data(),
        0, static_cast<int>(v_rows),
        head_dim, n_kv_heads,
        cache_v_row_bytes,
        v_cache->block_bytes());

    float cache_k_cosine = cosine(ref_k_rope.data(), k_cache_dequant.data(), k_rows * kv_dim);
    float cache_v_cosine = cosine(ref_v_proj.data(), v_cache_dequant.data(), v_rows * kv_dim);

    std::cout << "  Cache dequant K cosine vs FP32 ref: " << std::fixed << std::setprecision(6) << cache_k_cosine << "\n";
    std::cout << "  Cache dequant V cosine vs FP32 ref: " << std::fixed << std::setprecision(6) << cache_v_cosine << "\n";

    // Compare direct dequant vs cache dequant (should be IDENTICAL if cache flow is correct)
    float k_direct_vs_cache = cosine(k_direct_dequant.data(), k_cache_dequant.data(), k_rows * kv_dim);
    float v_direct_vs_cache = cosine(v_direct_dequant.data(), v_cache_dequant.data(), v_rows * kv_dim);
    std::cout << "  Direct vs Cache K dequant cosine: " << std::fixed << std::setprecision(6) << k_direct_vs_cache << "\n";
    std::cout << "  Direct vs Cache V dequant cosine: " << std::fixed << std::setprecision(6) << v_direct_vs_cache << "\n";

    // Check for exact float equality between direct and cache dequant
    int k_diff_count = 0, v_diff_count = 0;
    float k_max_diff = 0.0f, v_max_diff = 0.0f;
    for (size_t i = 0; i < k_rows * kv_dim; ++i)
    {
        float diff = std::abs(k_direct_dequant[i] - k_cache_dequant[i]);
        if (diff > 0.0f)
            k_diff_count++;
        k_max_diff = std::max(k_max_diff, diff);
    }
    for (size_t i = 0; i < v_rows * kv_dim; ++i)
    {
        float diff = std::abs(v_direct_dequant[i] - v_cache_dequant[i]);
        if (diff > 0.0f)
            v_diff_count++;
        v_max_diff = std::max(v_max_diff, diff);
    }
    std::cout << "  K float diffs: " << k_diff_count << "/" << (k_rows * kv_dim)
              << " max_diff=" << std::scientific << k_max_diff << "\n";
    std::cout << "  V float diffs: " << v_diff_count << "/" << (v_rows * kv_dim)
              << " max_diff=" << std::scientific << v_max_diff << "\n";

    // ---- Step 5: Now run the REAL SplitTQ pipeline and dump effective K/V ----
    std::cout << "\n>>> Step 5: Running REAL SplitTQ pipeline with K/V dumps...\n";

    // Set env var to enable dumps — use putenv since setenv may not be thread-safe
    setenv("LLAMINAR_DUMP_EFFECTIVE_KV", "1", 1);

    test_cfg_.kv_cache_precision = KVCachePrecision::TQ;
    ASSERT_TRUE(setupPipeline()) << "SplitTQ pipeline setup failed";

    // Prefill
    ASSERT_TRUE(runner_->forward(config_.token_ids.data(), config_.token_ids.size()))
        << "SplitTQ prefill failed";

    // Decode step 0
    runner_->clearSnapshots();
    ASSERT_TRUE(runner_->forward(&decode_token, 1)) << "SplitTQ decode step 0 failed";

    // Capture SplitTQ ATTENTION_CONTEXT
    size_t tq4_attn_size = 0;
    const float *tq4_attn_ctx = runner_->getSnapshot("layer0_ATTENTION_CONTEXT", tq4_attn_size);
    std::vector<float> tq4_attn_context;
    if (tq4_attn_ctx && tq4_attn_size > 0)
        tq4_attn_context.assign(tq4_attn_ctx, tq4_attn_ctx + tq4_attn_size);

    // Disable the dump env var
    unsetenv("LLAMINAR_DUMP_EFFECTIVE_KV");

    // ---- Step 6: Load dumped effective K/V from disk and analyze ----
    std::cout << ">>> Step 6: Analyzing dumped effective K/V...\n";

    const std::string dump_dir = "/tmp/effective_kv_dump/layer0_iter0";
    const std::string k_dump_path = dump_dir + "/K_effective.bin";
    const std::string v_dump_path = dump_dir + "/V_effective.bin";
    const std::string meta_path = dump_dir + "/meta.txt";

    bool dumps_available = std::filesystem::exists(k_dump_path) && std::filesystem::exists(v_dump_path);

    std::vector<float> pipeline_k_effective;
    std::vector<float> pipeline_v_effective;
    int dump_kv_len = 0;

    if (dumps_available)
    {
        // Read metadata to get dimensions
        {
            std::ifstream meta(meta_path);
            std::string line;
            while (std::getline(meta, line))
            {
                if (line.find("kv_len=") == 0)
                    dump_kv_len = std::stoi(line.substr(7));
            }
        }

        if (dump_kv_len > 0)
        {
            size_t k_elems = static_cast<size_t>(dump_kv_len) * n_kv_heads * head_dim;
            pipeline_k_effective.resize(k_elems);
            pipeline_v_effective.resize(k_elems);

            std::ifstream kf(k_dump_path, std::ios::binary);
            kf.read(reinterpret_cast<char *>(pipeline_k_effective.data()), k_elems * sizeof(float));
            size_t k_read = static_cast<size_t>(kf.gcount()) / sizeof(float);

            std::ifstream vf(v_dump_path, std::ios::binary);
            vf.read(reinterpret_cast<char *>(pipeline_v_effective.data()), k_elems * sizeof(float));
            size_t v_read = static_cast<size_t>(vf.gcount()) / sizeof(float);

            std::cout << "  Loaded dumped K: " << k_read << " floats, V: " << v_read << " floats\n";
            std::cout << "  dump_kv_len=" << dump_kv_len
                      << " (prefill=" << seq_len << " + decode=1 = " << (seq_len + 1) << ")\n";

            // Compare dumped effective K vs our simulated cache dequant
            // The dumped data has (seq_len+1) rows, our simulated cache has seq_len rows
            // Compare just the first seq_len rows:
            size_t compare_elems = std::min(k_rows * kv_dim, k_read);

            float pipeline_vs_sim_k = cosine(pipeline_k_effective.data(), k_cache_dequant.data(), compare_elems);
            float pipeline_vs_sim_v = cosine(pipeline_v_effective.data(), v_cache_dequant.data(), compare_elems);
            std::cout << "  Pipeline effective K vs simulated cache K: " << std::fixed << std::setprecision(6)
                      << pipeline_vs_sim_k << "\n";
            std::cout << "  Pipeline effective V vs simulated cache V: " << std::fixed << std::setprecision(6)
                      << pipeline_vs_sim_v << "\n";

            // Compare dumped effective K vs FP32 reference
            float pipeline_vs_ref_k = cosine(pipeline_k_effective.data(), ref_k_rope.data(), compare_elems);
            float pipeline_vs_ref_v = cosine(pipeline_v_effective.data(), ref_v_proj.data(), compare_elems);
            std::cout << "  Pipeline effective K vs FP32 ref: " << std::fixed << std::setprecision(6)
                      << pipeline_vs_ref_k << "\n";
            std::cout << "  Pipeline effective V vs FP32 ref: " << std::fixed << std::setprecision(6)
                      << pipeline_vs_ref_v << "\n";

            // Per-position analysis of the first few rows
            std::cout << "\n  Per-position K analysis (pipeline dump vs FP32 ref):\n";
            std::cout << "  " << std::left << std::setw(6) << "Pos"
                      << std::right
                      << std::setw(14) << "PipeVsRef"
                      << std::setw(14) << "SimVsRef"
                      << std::setw(14) << "PipeVsSim"
                      << std::setw(14) << "Pipe_L2"
                      << std::setw(14) << "Ref_L2"
                      << "\n";
            std::cout << "  " << std::string(80, '-') << "\n";

            size_t show_rows = std::min(k_rows, static_cast<size_t>(10));
            for (size_t row = 0; row < show_rows; ++row)
            {
                const float *ref_row = ref_k_rope.data() + row * kv_dim;
                const float *sim_row = k_cache_dequant.data() + row * kv_dim;
                const float *pipe_row = pipeline_k_effective.data() + row * kv_dim;

                float pipe_vs_ref = cosine(pipe_row, ref_row, kv_dim);
                float sim_vs_ref = cosine(sim_row, ref_row, kv_dim);
                float pipe_vs_sim = cosine(pipe_row, sim_row, kv_dim);
                float pipe_l2 = l2norm(pipe_row, kv_dim);
                float ref_l2 = l2norm(ref_row, kv_dim);

                std::cout << "  " << std::left << std::setw(6) << row
                          << std::right << std::fixed
                          << std::setw(14) << std::setprecision(6) << pipe_vs_ref
                          << std::setw(14) << std::setprecision(6) << sim_vs_ref
                          << std::setw(14) << std::setprecision(6) << pipe_vs_sim
                          << std::setw(14) << std::setprecision(4) << pipe_l2
                          << std::setw(14) << std::setprecision(4) << ref_l2
                          << "\n";
            }

            // Per-head analysis for position 0
            std::cout << "\n  Per-head K analysis at position 0 (pipeline dump vs FP32 ref):\n";
            std::cout << "  " << std::left << std::setw(6) << "Head"
                      << std::right
                      << std::setw(14) << "PipeVsRef"
                      << std::setw(14) << "SimVsRef"
                      << std::setw(14) << "PipeVsSim"
                      << std::setw(14) << "Pipe_first4"
                      << std::setw(14) << "Ref_first4"
                      << "\n";
            std::cout << "  " << std::string(80, '-') << "\n";

            for (int h = 0; h < n_kv_heads; ++h)
            {
                const float *ref_head = ref_k_rope.data() + h * head_dim;
                const float *sim_head = k_cache_dequant.data() + h * head_dim;
                const float *pipe_head = pipeline_k_effective.data() + h * head_dim;

                float pipe_vs_ref = cosine(pipe_head, ref_head, head_dim);
                float sim_vs_ref = cosine(sim_head, ref_head, head_dim);
                float pipe_vs_sim = cosine(pipe_head, sim_head, head_dim);

                // Print first 4 values of each for visual comparison
                char pipe_vals[80], ref_vals[80];
                snprintf(pipe_vals, sizeof(pipe_vals), "[%.3f,%.3f,%.3f,%.3f]",
                         pipe_head[0], pipe_head[1], pipe_head[2], pipe_head[3]);
                snprintf(ref_vals, sizeof(ref_vals), "[%.3f,%.3f,%.3f,%.3f]",
                         ref_head[0], ref_head[1], ref_head[2], ref_head[3]);

                std::cout << "  " << std::left << std::setw(6) << h
                          << std::right << std::fixed
                          << std::setw(14) << std::setprecision(6) << pipe_vs_ref
                          << std::setw(14) << std::setprecision(6) << sim_vs_ref
                          << std::setw(14) << std::setprecision(6) << pipe_vs_sim
                          << "  " << pipe_vals
                          << "  " << ref_vals
                          << "\n";
            }
        }
        else
        {
            std::cout << "  WARNING: Dumps available but kv_len=0 in metadata\n";
        }
    }
    else
    {
        std::cout << "  NOTE: No dumps found at " << dump_dir << "\n";
        std::cout << "  (LLAMINAR_DUMP_EFFECTIVE_KV uses static bool — may need fresh process)\n";
    }

    // ---- Step 7: Summary ----
    std::cout << "\n"
              << std::string(100, '=') << "\n";
    std::cout << "  SUMMARY: SplitTQ Cache Flow Simulation vs Real Pipeline (Layer 0)\n";
    std::cout << std::string(100, '=') << "\n";
    std::cout << "  Direct SplitTQ round-trip K:           " << std::fixed << std::setprecision(6) << direct_cosine << "\n";
    std::cout << "  Direct SplitTQ round-trip V:           " << std::fixed << std::setprecision(6) << v_direct_cosine << "\n";
    std::cout << "  Simulated cache dequant K:         " << std::fixed << std::setprecision(6) << cache_k_cosine << "\n";
    std::cout << "  Simulated cache dequant V:         " << std::fixed << std::setprecision(6) << cache_v_cosine << "\n";
    std::cout << "  Direct vs Cache K:                 " << std::fixed << std::setprecision(6) << k_direct_vs_cache << "\n";
    std::cout << "  Direct vs Cache V:                 " << std::fixed << std::setprecision(6) << v_direct_vs_cache << "\n";
    if (!pipeline_k_effective.empty())
    {
        size_t cmp = std::min(k_rows * kv_dim, pipeline_k_effective.size());
        float pv = cosine(pipeline_k_effective.data(), k_cache_dequant.data(), cmp);
        float pr = cosine(pipeline_k_effective.data(), ref_k_rope.data(), cmp);
        std::cout << "  Pipeline K vs Simulated Cache:     " << std::fixed << std::setprecision(6) << pv << "\n";
        std::cout << "  Pipeline K vs FP32 Ref:            " << std::fixed << std::setprecision(6) << pr << "\n";
    }
    if (!ref_attn_context.empty() && !tq4_attn_context.empty())
    {
        size_t cmp = std::min(ref_attn_context.size(), tq4_attn_context.size());
        float attn_cos = cosine(ref_attn_context.data(), tq4_attn_context.data(), cmp);
        std::cout << "  Real pipeline ATTN_CONTEXT FP32vsSplitTQ: " << std::fixed << std::setprecision(6) << attn_cos << "\n";
    }
    std::cout << std::string(100, '=') << "\n";

    // ---- Diagnostic assertions ----
    // Direct round-trip should be excellent
    EXPECT_GE(direct_cosine, 0.99f) << "Direct SplitTQ K round-trip is bad";
    EXPECT_GE(v_direct_cosine, 0.99f) << "Direct SplitTQ V round-trip is bad";

    // Cache dequant should match direct dequant (byte-for-byte copy + same dequant)
    EXPECT_GE(k_direct_vs_cache, 0.9999f)
        << "Cache dequant differs from direct dequant! Bug in cache buffer flow!";
    EXPECT_GE(v_direct_vs_cache, 0.9999f)
        << "Cache dequant differs from direct dequant! Bug in cache buffer flow!";

    // Cache dequant should be approximately as good as direct round-trip
    EXPECT_GE(cache_k_cosine, 0.99f) << "Cache K dequant quality is bad";
    EXPECT_GE(cache_v_cosine, 0.99f) << "Cache V dequant quality is bad";

    runner_.reset();
    model_ctx_.reset();
}

// =============================================================================
// Custom Main with MPI Initialization
// =============================================================================

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    GlobalBackendRouter::shutdown();
    MPI_Finalize();
    return result;
}
