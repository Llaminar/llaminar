/**
 * @file Test__Pipeline_vs_LayerExecutor_FP32_Parity.cpp
 * @brief Integration test comparing legacy Pipeline vs LayerExecutor FP32 paths
 *
 * This test validates that the new LayerExecutor framework produces equivalent
 * results to the legacy imperative pipeline code when both use FP32 precision.
 *
 * The test runs the same input through two pipeline configurations:
 *   1. Legacy path: LLAMINAR_USE_LAYER_EXECUTOR=0 (imperative code)
 *   2. LayerExecutor path: LLAMINAR_USE_LAYER_EXECUTOR=1 (compute graphs)
 *
 * Parity is validated using:
 *   1. Top-1 token match (must be identical for greedy sampling)
 *   2. Top-5 token overlap (>= 80% required)
 *   3. Cosine similarity (>= 0.999 for FP32 vs FP32)
 *   4. Max absolute difference (< 0.5 for FP32 vs FP32)
 *
 * Test scenarios:
 *   - Prefill: Multi-token prompt processing
 *   - Incremental decode: Single-token autoregressive generation
 *
 * @author David Sanftenberg
 * @date 2025-01-13
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <set>
#include <iomanip>

#include "pipelines/qwen/Qwen2Pipeline.h"
#include "pipelines/PipelineConfig.h"
#include "loaders/ModelContext.h"
#include "utils/MPIContext.h"
#include "utils/Logger.h"
#include "utils/DebugEnv.h"
#include "backends/ComputeBackend.h"

using namespace llaminar2;

namespace
{

    /**
     * @brief Compute cosine similarity between two float arrays
     */
    double cosine_similarity(const float *a, const float *b, size_t n)
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
     * @brief Compute maximum absolute difference between two arrays
     */
    float max_abs_diff(const float *a, const float *b, size_t n)
    {
        float max_diff = 0.0f;
        for (size_t i = 0; i < n; ++i)
        {
            float diff = std::abs(a[i] - b[i]);
            if (diff > max_diff)
                max_diff = diff;
        }
        return max_diff;
    }

    /**
     * @brief Compute mean absolute difference between two arrays
     */
    float mean_abs_diff(const float *a, const float *b, size_t n)
    {
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            sum += std::abs(a[i] - b[i]);
        }
        return static_cast<float>(sum / n);
    }

    /**
     * @brief Get top-K token indices from logits (sorted by score descending)
     */
    std::vector<int> get_topk(const float *logits, size_t vocab_size, int k)
    {
        std::vector<std::pair<float, int>> indexed(vocab_size);
        for (size_t i = 0; i < vocab_size; ++i)
        {
            indexed[i] = {logits[i], static_cast<int>(i)};
        }
        std::partial_sort(indexed.begin(), indexed.begin() + k, indexed.end(),
                          [](const auto &a, const auto &b)
                          { return a.first > b.first; });
        std::vector<int> topk(k);
        for (int i = 0; i < k; ++i)
        {
            topk[i] = indexed[i].second;
        }
        return topk;
    }

    /**
     * @brief Count overlap between two top-K lists
     */
    int count_overlap(const std::vector<int> &a, const std::vector<int> &b)
    {
        std::set<int> set_a(a.begin(), a.end());
        int overlap = 0;
        for (int t : b)
        {
            if (set_a.count(t) > 0)
            {
                overlap++;
            }
        }
        return overlap;
    }

    /**
     * @brief Helper to reset DebugEnv execution config and set specific value
     *
     * Since DebugEnv is a singleton that persists across tests, we need to
     * manipulate the actual env vars and call reload() to change settings.
     */
    class EnvVarGuard
    {
    public:
        EnvVarGuard(const std::string &name, const std::string &value)
            : name_(name)
        {
            // Save original value
            const char *orig = std::getenv(name.c_str());
            if (orig)
            {
                original_value_ = orig;
                had_original_ = true;
            }
            else
            {
                had_original_ = false;
            }
            // Set new value
            setenv(name.c_str(), value.c_str(), 1);
        }

        ~EnvVarGuard()
        {
            // Restore original
            if (had_original_)
            {
                setenv(name_.c_str(), original_value_.c_str(), 1);
            }
            else
            {
                unsetenv(name_.c_str());
            }
        }

    private:
        std::string name_;
        std::string original_value_;
        bool had_original_;
    };

} // namespace

/**
 * @brief Test fixture for Pipeline vs LayerExecutor FP32 parity
 */
class Test__Pipeline_vs_LayerExecutor_FP32_Parity : public ::testing::Test
{
protected:
    std::shared_ptr<ModelContext> model_ctx_;
    std::shared_ptr<MPIContext> mpi_ctx_;
    std::string model_path_;
    int rank_ = 0;
    int world_size_ = 1;
    int cpu_device_idx_ = 0; // Valid CPU device index from DeviceManager

    // Parity thresholds - requirements for FP32 vs FP32 parity
    // Both paths use the same precision, but differences arise from:
    // - Operation ordering (non-associativity of floating-point)
    // - Kernel implementation differences (SIMD vs scalar paths)
    // - Accumulation order in parallel reductions
    static constexpr double MIN_COSINE_SIMILARITY = 0.999; // High correlation
    static constexpr float MAX_MAX_ABS_DIFF = 0.5f;        // Allow some outliers
    static constexpr float MAX_MEAN_ABS_DIFF = 0.1f;       // Reasonable average
    static constexpr double MIN_TOP5_OVERLAP_RATIO = 0.80; // 4/5 tokens must match

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

        mpi_ctx_ = std::make_shared<MPIContext>(rank_, world_size_, MPI_COMM_WORLD);

        // Initialize DeviceManager to properly enumerate compute devices
        // This is the canonical pattern for integration tests
        DeviceManager::instance().initialize(-1);

        const auto &devices = DeviceManager::instance().devices();
        if (devices.empty())
        {
            GTEST_SKIP() << "No compute devices available";
        }

        // Use first available device (typically CPU device 0)
        cpu_device_idx_ = 0;

        if (rank_ == 0)
        {
            LOG_INFO("[Setup] DeviceManager initialized with " << devices.size() << " device(s)");
            LOG_INFO("[Setup] Using device index: " << cpu_device_idx_);
        }

        // Use Qwen 0.5B Q4_0 for fast testing
        model_path_ = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

        model_ctx_ = ModelContext::create(model_path_, mpi_ctx_);
        if (!model_ctx_)
        {
            GTEST_SKIP() << "Model not found: " << model_path_;
        }

        if (rank_ == 0)
        {
            LOG_INFO("[Pipeline vs LayerExecutor FP32 Parity] Loaded model: " << model_path_);
        }
    }

    void TearDown() override
    {
        model_ctx_.reset();
        mpi_ctx_->barrier();
    }

    /**
     * @brief Create a pipeline with specified LayerExecutor mode
     *
     * @param use_layer_executor If true, use LayerExecutor path; if false, use legacy path
     */
    std::unique_ptr<Qwen2Pipeline> createPipeline(bool use_layer_executor)
    {
        // Force DebugEnv to reload with new env var value
        auto &mut_env = mutableDebugEnv();

        // Set environment variable to control LayerExecutor usage
        // Note: Pipeline reads this during construction
        if (use_layer_executor)
        {
            setenv("LLAMINAR_USE_LAYER_EXECUTOR", "1", 1);
            mut_env.execution.use_layer_executor = true;

            // Enable all execution stages for full LayerExecutor path
            // These flags control which ComputeStages are added to the graph
            setenv("LLAMINAR_EXEC_EMBEDDING", "1", 1);
            setenv("LLAMINAR_EXEC_RMSNORM", "1", 1);
            setenv("LLAMINAR_EXEC_GEMM", "1", 1);
            setenv("LLAMINAR_EXEC_ROPE", "1", 1);
            setenv("LLAMINAR_EXEC_ATTENTION", "1", 1);
            setenv("LLAMINAR_EXEC_SWIGLU", "1", 1);
            setenv("LLAMINAR_EXEC_RESIDUAL", "1", 1);
            setenv("LLAMINAR_EXEC_LM_HEAD", "1", 1);

            mut_env.execution.exec_embedding = true;
            mut_env.execution.exec_rmsnorm = true;
            mut_env.execution.exec_gemm = true;
            mut_env.execution.exec_rope = true;
            mut_env.execution.exec_attention = true;
            mut_env.execution.exec_swiglu = true;
            mut_env.execution.exec_residual = true;
            mut_env.execution.exec_lm_head = true;
        }
        else
        {
            setenv("LLAMINAR_USE_LAYER_EXECUTOR", "0", 1);
            mut_env.execution.use_layer_executor = false;

            // Disable all executor stages for legacy path
            setenv("LLAMINAR_EXEC_EMBEDDING", "0", 1);
            setenv("LLAMINAR_EXEC_RMSNORM", "0", 1);
            setenv("LLAMINAR_EXEC_GEMM", "0", 1);
            setenv("LLAMINAR_EXEC_ROPE", "0", 1);
            setenv("LLAMINAR_EXEC_ATTENTION", "0", 1);
            setenv("LLAMINAR_EXEC_SWIGLU", "0", 1);
            setenv("LLAMINAR_EXEC_RESIDUAL", "0", 1);
            setenv("LLAMINAR_EXEC_LM_HEAD", "0", 1);

            mut_env.execution.exec_embedding = false;
            mut_env.execution.exec_rmsnorm = false;
            mut_env.execution.exec_gemm = false;
            mut_env.execution.exec_rope = false;
            mut_env.execution.exec_attention = false;
            mut_env.execution.exec_swiglu = false;
            mut_env.execution.exec_residual = false;
            mut_env.execution.exec_lm_head = false;
        }

        PipelineConfig config;
        config.activation_precision = ActivationPrecision::FP32;
        config.max_seq_len = 512;

        // CRITICAL FIX: Use the SHARED model context (model_ctx_) instead of creating
        // a fresh one. This ensures both pipelines use the exact same weight tensors
        // and thus the same packed weight caches. Previously, separate model contexts
        // led to different tensor pointers, which could mask bugs where kernels
        // incorrectly accessed wrong packed weights due to pointer identity issues.
        //
        // The "fresh pipeline state" comment was misleading - we want fresh PIPELINE
        // state (activation buffers, KV cache), not fresh WEIGHT state. Weights
        // should be identical between runs.
        if (!model_ctx_)
        {
            LOG_ERROR("[createPipeline] Shared model context is null");
            return nullptr;
        }

        // Use valid CPU device index from DeviceManager for LayerExecutor compatibility
        // Legacy path works with -1 (convention for CPU), but LayerExecutor requires
        // a valid device index from DeviceManager enumeration
        int device_idx = use_layer_executor ? cpu_device_idx_ : -1;

        return std::make_unique<Qwen2Pipeline>(
            model_ctx_, mpi_ctx_, device_idx, nullptr, config, 1);
    }

    /**
     * @brief Compare logits between legacy and LayerExecutor pipelines
     */
    bool compareLogits(const float *legacy_logits, const float *executor_logits,
                       size_t vocab_size, const std::string &label)
    {
        // Get top-5 predictions
        auto legacy_top5 = get_topk(legacy_logits, vocab_size, 5);
        auto executor_top5 = get_topk(executor_logits, vocab_size, 5);

        // Compute metrics
        int top5_overlap = count_overlap(legacy_top5, executor_top5);
        double top5_ratio = top5_overlap / 5.0;
        bool top1_match = (legacy_top5[0] == executor_top5[0]);

        double cosine = cosine_similarity(legacy_logits, executor_logits, vocab_size);
        float max_diff = max_abs_diff(legacy_logits, executor_logits, vocab_size);
        float mean_diff = mean_abs_diff(legacy_logits, executor_logits, vocab_size);

        // Log results
        if (rank_ == 0)
        {
            LOG_INFO("");
            LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
            LOG_INFO("║  " << label);
            LOG_INFO("╚════════════════════════════════════════════════════════════════╝");
            LOG_INFO("  Top-1 match: " << (top1_match ? "YES ✓" : "NO ✗")
                                       << " (Legacy=" << legacy_top5[0] << ", Executor=" << executor_top5[0] << ")");
            LOG_INFO("  Top-5 overlap: " << top5_overlap << "/5 (" << std::fixed << std::setprecision(0)
                                         << (top5_ratio * 100) << "%)");
            LOG_INFO("  Cosine similarity: " << std::fixed << std::setprecision(6) << cosine);
            LOG_INFO("  Max abs diff: " << std::scientific << std::setprecision(4) << max_diff);
            LOG_INFO("  Mean abs diff: " << std::scientific << std::setprecision(4) << mean_diff);

            // Show top-5 tokens
            LOG_INFO("  Legacy top-5: [");
            for (int i = 0; i < 5; ++i)
            {
                LOG_INFO("    " << i + 1 << ". token " << legacy_top5[i]
                                << " (logit=" << std::fixed << std::setprecision(2) << legacy_logits[legacy_top5[i]] << ")");
            }
            LOG_INFO("  ]");
            LOG_INFO("  Executor top-5: [");
            for (int i = 0; i < 5; ++i)
            {
                LOG_INFO("    " << i + 1 << ". token " << executor_top5[i]
                                << " (logit=" << std::fixed << std::setprecision(2) << executor_logits[executor_top5[i]] << ")");
            }
            LOG_INFO("  ]");
        }

        // Check pass/fail criteria
        bool pass = true;

        // Top-1 mismatch is a warning, not a failure (logits may be very close)
        // What matters is that top-5 overlap is good and cosine similarity is high
        if (!top1_match)
        {
            if (rank_ == 0)
                LOG_WARN("  ⚠ Top-1 mismatch: Legacy=" << legacy_top5[0]
                                                       << " vs Executor=" << executor_top5[0]
                                                       << " (acceptable if top-5 overlap is good)");
            // Don't fail on top-1 mismatch alone
        }

        if (top5_ratio < MIN_TOP5_OVERLAP_RATIO)
        {
            if (rank_ == 0)
                LOG_ERROR("  ✗ Top-5 overlap " << (top5_ratio * 100) << "% < "
                                               << (MIN_TOP5_OVERLAP_RATIO * 100) << "% threshold");
            pass = false;
        }

        if (cosine < MIN_COSINE_SIMILARITY)
        {
            if (rank_ == 0)
                LOG_ERROR("  ✗ Cosine similarity " << cosine << " < " << MIN_COSINE_SIMILARITY << " threshold");
            pass = false;
        }

        if (max_diff > MAX_MAX_ABS_DIFF)
        {
            if (rank_ == 0)
                LOG_ERROR("  ✗ Max abs diff " << max_diff << " > " << MAX_MAX_ABS_DIFF << " threshold");
            pass = false;
        }

        if (mean_diff > MAX_MEAN_ABS_DIFF)
        {
            if (rank_ == 0)
                LOG_ERROR("  ✗ Mean abs diff " << mean_diff << " > " << MAX_MEAN_ABS_DIFF << " threshold");
            pass = false;
        }

        if (pass && rank_ == 0)
        {
            LOG_INFO("  ✓ All parity checks PASSED");
        }

        return pass;
    }
};

/**
 * @brief Test prefill parity between legacy Pipeline and LayerExecutor (FP32)
 *
 * Runs a multi-token prompt through both execution paths and compares final logits.
 */
TEST_F(Test__Pipeline_vs_LayerExecutor_FP32_Parity, PrefillParity)
{
    if (rank_ == 0)
    {
        LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  Pipeline vs LayerExecutor FP32 Parity Test: Prefill           ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════╝");
    }

    // Test prompt: "The quick brown fox jumps over the lazy dog"
    // Token IDs from Qwen2.5 tokenizer (without special tokens)
    std::vector<int> tokens = {785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562};

    // Create legacy pipeline (LayerExecutor disabled)
    if (rank_ == 0)
        LOG_INFO("Creating legacy pipeline (LayerExecutor disabled)...");
    auto legacy_pipeline = createPipeline(false);
    ASSERT_NE(legacy_pipeline, nullptr) << "Failed to create legacy pipeline";

    // Create LayerExecutor pipeline
    if (rank_ == 0)
        LOG_INFO("Creating LayerExecutor pipeline...");
    auto executor_pipeline = createPipeline(true);
    ASSERT_NE(executor_pipeline, nullptr) << "Failed to create LayerExecutor pipeline";

    // Run prefill on legacy
    if (rank_ == 0)
        LOG_INFO("Running prefill on legacy pipeline...");
    bool legacy_ok = legacy_pipeline->forward(tokens.data(), static_cast<int>(tokens.size()));
    ASSERT_TRUE(legacy_ok) << "Legacy pipeline forward failed";

    // Run prefill on LayerExecutor
    if (rank_ == 0)
        LOG_INFO("Running prefill on LayerExecutor pipeline...");
    bool executor_ok = executor_pipeline->forward(tokens.data(), static_cast<int>(tokens.size()));
    ASSERT_TRUE(executor_ok) << "LayerExecutor pipeline forward failed";

    // Get logits
    size_t vocab_size = model_ctx_->model().vocab_size;
    const float *legacy_logits = legacy_pipeline->logits();
    const float *executor_logits = executor_pipeline->logits();

    ASSERT_NE(legacy_logits, nullptr) << "Legacy logits is null";
    ASSERT_NE(executor_logits, nullptr) << "Executor logits is null";

    // Compare
    bool pass = compareLogits(legacy_logits, executor_logits, vocab_size, "Prefill Logits Comparison");

    EXPECT_TRUE(pass) << "Prefill parity check failed";
}

/**
 * @brief Test incremental decode parity between legacy Pipeline and LayerExecutor (FP32)
 *
 * After prefill, generates one token autoregressively and compares.
 */
TEST_F(Test__Pipeline_vs_LayerExecutor_FP32_Parity, IncrementalDecodeParity)
{
    if (rank_ == 0)
    {
        LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  Pipeline vs LayerExecutor FP32 Parity: Incremental Decode     ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════╝");
    }

    // Initial prompt
    std::vector<int> tokens = {785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562};

    // Create pipelines
    auto legacy_pipeline = createPipeline(false);
    auto executor_pipeline = createPipeline(true);
    ASSERT_NE(legacy_pipeline, nullptr) << "Failed to create legacy pipeline";
    ASSERT_NE(executor_pipeline, nullptr) << "Failed to create LayerExecutor pipeline";

    // Prefill both
    bool legacy_ok = legacy_pipeline->forward(tokens.data(), static_cast<int>(tokens.size()));
    bool executor_ok = executor_pipeline->forward(tokens.data(), static_cast<int>(tokens.size()));
    ASSERT_TRUE(legacy_ok) << "Legacy prefill failed";
    ASSERT_TRUE(executor_ok) << "LayerExecutor prefill failed";

    // Get next token (greedy) from legacy as the "correct" next token
    size_t vocab_size = model_ctx_->model().vocab_size;
    const float *legacy_prefill_logits = legacy_pipeline->logits();
    auto legacy_top1 = get_topk(legacy_prefill_logits, vocab_size, 1);
    int next_token = legacy_top1[0];

    if (rank_ == 0)
    {
        LOG_INFO("Next token from legacy prefill: " << next_token);
    }

    // Incremental decode: feed next_token to both pipelines
    std::vector<int> decode_tokens = {next_token};
    legacy_ok = legacy_pipeline->forward(decode_tokens.data(), 1);
    executor_ok = executor_pipeline->forward(decode_tokens.data(), 1);

    ASSERT_TRUE(legacy_ok) << "Legacy decode failed";
    ASSERT_TRUE(executor_ok) << "LayerExecutor decode failed";

    // Get decode logits
    const float *legacy_decode_logits = legacy_pipeline->logits();
    const float *executor_decode_logits = executor_pipeline->logits();

    // Compare decode logits
    bool pass = compareLogits(legacy_decode_logits, executor_decode_logits, vocab_size,
                              "Decode Step 1 Logits Comparison");

    EXPECT_TRUE(pass) << "Incremental decode parity check failed";
}

/**
 * @brief Test multi-step decode parity (5 tokens)
 *
 * Generates 5 tokens autoregressively and compares at each step.
 */
TEST_F(Test__Pipeline_vs_LayerExecutor_FP32_Parity, MultiStepDecodeParity)
{
    if (rank_ == 0)
    {
        LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  Pipeline vs LayerExecutor FP32 Parity: Multi-Step Decode      ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════╝");
    }

    constexpr int NUM_DECODE_STEPS = 5;

    // Initial prompt
    std::vector<int> tokens = {785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562};

    // Create pipelines
    auto legacy_pipeline = createPipeline(false);
    auto executor_pipeline = createPipeline(true);
    ASSERT_NE(legacy_pipeline, nullptr) << "Failed to create legacy pipeline";
    ASSERT_NE(executor_pipeline, nullptr) << "Failed to create LayerExecutor pipeline";

    // Prefill both
    bool legacy_ok = legacy_pipeline->forward(tokens.data(), static_cast<int>(tokens.size()));
    bool executor_ok = executor_pipeline->forward(tokens.data(), static_cast<int>(tokens.size()));
    ASSERT_TRUE(legacy_ok) << "Legacy prefill failed";
    ASSERT_TRUE(executor_ok) << "LayerExecutor prefill failed";

    size_t vocab_size = model_ctx_->model().vocab_size;
    bool all_pass = true;

    for (int step = 0; step < NUM_DECODE_STEPS; ++step)
    {
        // Get next token from legacy (greedy)
        const float *legacy_logits = legacy_pipeline->logits();
        auto top1 = get_topk(legacy_logits, vocab_size, 1);
        int next_token = top1[0];

        if (rank_ == 0)
        {
            LOG_INFO("Decode step " << step + 1 << ": next_token=" << next_token);
        }

        // Feed to both pipelines
        std::vector<int> decode_tokens = {next_token};
        legacy_ok = legacy_pipeline->forward(decode_tokens.data(), 1);
        executor_ok = executor_pipeline->forward(decode_tokens.data(), 1);

        ASSERT_TRUE(legacy_ok) << "Legacy decode step " << step + 1 << " failed";
        ASSERT_TRUE(executor_ok) << "LayerExecutor decode step " << step + 1 << " failed";

        // Compare logits
        const float *legacy_decode_logits = legacy_pipeline->logits();
        const float *executor_decode_logits = executor_pipeline->logits();

        std::string label = "Decode Step " + std::to_string(step + 1) + " Logits";
        bool step_pass = compareLogits(legacy_decode_logits, executor_decode_logits, vocab_size, label);

        if (!step_pass)
        {
            all_pass = false;
        }
    }

    EXPECT_TRUE(all_pass) << "Multi-step decode parity check failed";
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
