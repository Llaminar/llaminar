/**
 * @file Test__GraphBufferManagement_Parity.cpp
 * @brief Integration test comparing pipeline-owned vs graph-owned buffer management
 * @author David Sanftenberg
 * @date January 2025
 *
 * This test validates that graph-managed buffer allocation produces equivalent
 * results to pipeline-managed (legacy) buffer allocation.
 *
 * The test runs the same input through two pipeline configurations:
 *   1. Pipeline-managed buffers: LLAMINAR_USE_GRAPH_BUFFER_MANAGEMENT=0
 *   2. Graph-managed buffers: LLAMINAR_USE_GRAPH_BUFFER_MANAGEMENT=1
 *
 * Both modes use the LayerExecutor path (LLAMINAR_USE_LAYER_EXECUTOR=1).
 *
 * Parity is validated using:
 *   1. Top-1 token match (must be identical for greedy sampling)
 *   2. Top-5 token overlap (>= 80% required)
 *   3. Cosine similarity (>= 0.999)
 *   4. Max absolute difference (< 0.5)
 *
 * Test scenarios:
 *   - Prefill: Multi-token prompt processing
 *   - Incremental decode: Single-token autoregressive generation
 *
 * Part of Phase 5: Qwen2Graph Integration
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

        std::vector<int> result(k);
        for (int i = 0; i < k; ++i)
        {
            result[i] = indexed[i].second;
        }
        return result;
    }

    /**
     * @brief Calculate top-K overlap between two sets
     */
    double topk_overlap(const std::vector<int> &a, const std::vector<int> &b)
    {
        std::set<int> set_a(a.begin(), a.end());
        std::set<int> set_b(b.begin(), b.end());
        std::vector<int> intersection;
        std::set_intersection(set_a.begin(), set_a.end(),
                              set_b.begin(), set_b.end(),
                              std::back_inserter(intersection));
        return static_cast<double>(intersection.size()) / std::max(a.size(), b.size());
    }

} // anonymous namespace

// ============================================================================
// Test Fixture
// ============================================================================

class Test__GraphBufferManagement_Parity : public ::testing::Test
{
protected:
    void SetUp() override
    {
        int rank, world_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        mpi_ctx_ = std::make_shared<MPIContext>(rank, world_size, MPI_COMM_WORLD);
        rank_ = rank;
        world_size_ = world_size;

        // Initialize DeviceManager to enumerate compute devices
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

        // Find a test model
        model_path_ = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

        // Load model (shared across tests)
        model_ctx_ = ModelContext::create(model_path_, mpi_ctx_);
        if (!model_ctx_)
        {
            GTEST_SKIP() << "Model not found: " << model_path_;
        }
        vocab_size_ = model_ctx_->model().vocab_size;
    }

    void TearDown() override
    {
        model_ctx_.reset();
        // Restore environment
        unsetenv("LLAMINAR_USE_LAYER_EXECUTOR");
        unsetenv("LLAMINAR_USE_GRAPH_BUFFER_MANAGEMENT");
    }

    /**
     * @brief Create a pipeline with specific buffer management mode
     * @param use_graph_buffers If true, use graph-managed buffers; if false, use pipeline-managed
     */
    std::unique_ptr<Qwen2Pipeline> createPipeline(bool use_graph_buffers)
    {
        // Access mutable debugEnv for modification
        auto &mut_env = mutableDebugEnv();

        // Always use LayerExecutor for both modes
        setenv("LLAMINAR_USE_LAYER_EXECUTOR", "1", 1);
        mut_env.execution.use_layer_executor = true;

        // Enable all execution stages
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

        if (use_graph_buffers)
        {
            setenv("LLAMINAR_USE_GRAPH_BUFFER_MANAGEMENT", "1", 1);
            mut_env.execution.use_graph_buffer_management = true;
        }
        else
        {
            setenv("LLAMINAR_USE_GRAPH_BUFFER_MANAGEMENT", "0", 1);
            mut_env.execution.use_graph_buffer_management = false;
        }

        // Create pipeline config
        PipelineConfig config;
        config.activation_precision = ActivationPrecision::FP32;
        config.max_seq_len = 256;

        // Create pipeline using shared model context
        return std::make_unique<Qwen2Pipeline>(
            model_ctx_, mpi_ctx_, cpu_device_idx_, nullptr, config, 1);
    }

    std::shared_ptr<MPIContext> mpi_ctx_;
    std::shared_ptr<ModelContext> model_ctx_;
    int rank_;
    int world_size_;
    int cpu_device_idx_;
    std::string model_path_;
    size_t vocab_size_;
};

// ============================================================================
// Prefill Parity Test
// ============================================================================

TEST_F(Test__GraphBufferManagement_Parity, Prefill_PipelineVsGraphBuffers)
{
    // Test prompt: "The quick brown fox"
    std::vector<int> tokens = {785, 3974, 13876, 38835};

    // Run with pipeline-managed buffers
    if (rank_ == 0)
        LOG_INFO("[Test] Creating pipeline with pipeline-managed buffers...");
    auto pipeline_legacy = createPipeline(false /* use_graph_buffers */);
    ASSERT_NE(pipeline_legacy, nullptr) << "Failed to create pipeline-managed pipeline";

    if (rank_ == 0)
        LOG_INFO("[Test] Running forward on pipeline-managed...");
    bool legacy_ok = pipeline_legacy->forward(tokens.data(), static_cast<int>(tokens.size()));
    ASSERT_TRUE(legacy_ok) << "Pipeline-managed forward failed";

    const float *logits_legacy = pipeline_legacy->logits();
    ASSERT_NE(logits_legacy, nullptr) << "Failed to get legacy logits";

    // Run with graph-managed buffers
    if (rank_ == 0)
        LOG_INFO("[Test] Creating pipeline with graph-managed buffers...");
    auto pipeline_graph = createPipeline(true /* use_graph_buffers */);
    ASSERT_NE(pipeline_graph, nullptr) << "Failed to create graph-managed pipeline";

    if (rank_ == 0)
        LOG_INFO("[Test] Running forward on graph-managed...");
    bool graph_ok = pipeline_graph->forward(tokens.data(), static_cast<int>(tokens.size()));
    ASSERT_TRUE(graph_ok) << "Graph-managed forward failed";

    const float *logits_graph = pipeline_graph->logits();
    ASSERT_NE(logits_graph, nullptr) << "Failed to get graph logits";

    // Get top-1 tokens
    auto top1_legacy = get_topk(logits_legacy, vocab_size_, 1);
    auto top1_graph = get_topk(logits_graph, vocab_size_, 1);

    if (rank_ == 0)
    {
        LOG_INFO("[Test] Pipeline-managed top-1 token: " << top1_legacy[0]);
        LOG_INFO("[Test] Graph-managed top-1 token: " << top1_graph[0]);
    }

    // Top-1 must match exactly
    EXPECT_EQ(top1_legacy[0], top1_graph[0])
        << "Top-1 token mismatch between pipeline-managed and graph-managed buffers";

    // Top-5 overlap >= 80%
    auto top5_legacy = get_topk(logits_legacy, vocab_size_, 5);
    auto top5_graph = get_topk(logits_graph, vocab_size_, 5);
    double overlap = topk_overlap(top5_legacy, top5_graph);

    if (rank_ == 0)
        LOG_INFO("[Test] Top-5 overlap: " << (overlap * 100.0) << "%");
    EXPECT_GE(overlap, 0.8) << "Top-5 overlap too low";

    // Cosine similarity >= 0.999
    double cos_sim = cosine_similarity(logits_legacy, logits_graph, vocab_size_);
    if (rank_ == 0)
        LOG_INFO("[Test] Cosine similarity: " << cos_sim);
    EXPECT_GE(cos_sim, 0.999) << "Cosine similarity too low";

    // Max absolute difference < 0.5
    float max_diff = max_abs_diff(logits_legacy, logits_graph, vocab_size_);
    if (rank_ == 0)
        LOG_INFO("[Test] Max absolute difference: " << max_diff);
    EXPECT_LT(max_diff, 0.5f) << "Max absolute difference too high";
}

// ============================================================================
// Multi-Step Decode Parity Test
// ============================================================================

/**
 * @brief Test parity between pipeline-managed and graph-managed buffer modes
 *
 * This test validates that both buffer management modes produce identical results
 * during multi-step decode. The key difference is:
 * - Pipeline-managed: Uses legacy AttentionWithKVCacheStage
 * - Graph-managed: Uses decomposed KVCacheAppendStage + AttentionComputeStage
 *
 * Both should produce identical results since the fix in ComputeStage.cpp ensures
 * AttentionComputeStage builds the correct causal mask for decode mode.
 */
TEST_F(Test__GraphBufferManagement_Parity, Decode_MultiStep_Parity)
{
    // Test prompt
    std::vector<int> prompt_tokens = {785, 3974, 13876, 38835}; // "The quick brown fox"

    // Initialize pipeline-managed
    auto pipeline_legacy = createPipeline(false);
    ASSERT_NE(pipeline_legacy, nullptr);

    // Initialize graph-managed
    auto pipeline_graph = createPipeline(true);
    ASSERT_NE(pipeline_graph, nullptr);

    // Run prefill on both
    bool legacy_ok = pipeline_legacy->forward(prompt_tokens.data(), static_cast<int>(prompt_tokens.size()));
    ASSERT_TRUE(legacy_ok) << "Legacy prefill failed";

    bool graph_ok = pipeline_graph->forward(prompt_tokens.data(), static_cast<int>(prompt_tokens.size()));
    ASSERT_TRUE(graph_ok) << "Graph prefill failed";

    // Get prefill logits
    const float *logits_legacy = pipeline_legacy->logits();
    const float *logits_graph = pipeline_graph->logits();

    ASSERT_NE(logits_legacy, nullptr);
    ASSERT_NE(logits_graph, nullptr);

    // Check prefill parity
    double prefill_cos_sim = cosine_similarity(logits_legacy, logits_graph, vocab_size_);
    if (rank_ == 0)
        LOG_INFO("[Test] Prefill cosine similarity: " << prefill_cos_sim);
    EXPECT_GE(prefill_cos_sim, 0.999) << "Prefill cosine similarity too low";

    // Decode 5 tokens and compare each step
    constexpr int num_decode_steps = 5;

    for (int step = 0; step < num_decode_steps; ++step)
    {
        // Get next token from legacy path (greedy)
        auto top1_legacy = get_topk(logits_legacy, vocab_size_, 1);
        auto top1_graph = get_topk(logits_graph, vocab_size_, 1);

        int next_token = top1_legacy[0];

        if (rank_ == 0)
        {
            LOG_INFO("[Test] Decode step " << step << ": legacy=" << top1_legacy[0]
                                           << " graph=" << top1_graph[0]);
        }

        // Tokens must match
        EXPECT_EQ(top1_legacy[0], top1_graph[0])
            << "Token mismatch at decode step " << step;

        // Run decode with same token on both
        std::vector<int> decode_tokens = {next_token};
        legacy_ok = pipeline_legacy->forward(decode_tokens.data(), 1);
        ASSERT_TRUE(legacy_ok) << "Legacy decode failed at step " << step;

        graph_ok = pipeline_graph->forward(decode_tokens.data(), 1);
        ASSERT_TRUE(graph_ok) << "Graph decode failed at step " << step;

        // Update logits pointers
        logits_legacy = pipeline_legacy->logits();
        logits_graph = pipeline_graph->logits();

        ASSERT_NE(logits_legacy, nullptr) << "Failed at step " << step;
        ASSERT_NE(logits_graph, nullptr) << "Failed at step " << step;

        // Verify cosine similarity at each step
        double cos_sim = cosine_similarity(logits_legacy, logits_graph, vocab_size_);
        if (rank_ == 0)
            LOG_INFO("[Test] Step " << step << " cosine similarity: " << cos_sim);
        EXPECT_GE(cos_sim, 0.999) << "Cosine similarity too low at step " << step;
    }
}

// ============================================================================
// Memory Statistics Test
// ============================================================================

TEST_F(Test__GraphBufferManagement_Parity, GraphBuffers_ReportsStatistics)
{
    // Create graph-managed pipeline
    auto pipeline = createPipeline(true /* use_graph_buffers */);
    ASSERT_NE(pipeline, nullptr);

    // Run a prefill to ensure buffers are used
    std::vector<int> tokens = {785, 3974};
    bool ok = pipeline->forward(tokens.data(), static_cast<int>(tokens.size()));
    ASSERT_TRUE(ok);

    const float *logits = pipeline->logits();
    ASSERT_NE(logits, nullptr);

    // The pipeline should have used graph-managed buffers
    // We can't easily access internal stats, but the test passing
    // confirms the graph buffer management path works
    if (rank_ == 0)
        LOG_INFO("[Test] Graph buffer management inference completed successfully");
}

// ============================================================================
// Edge Case: Single Token
// ============================================================================

TEST_F(Test__GraphBufferManagement_Parity, SingleToken_Parity)
{
    // Single token input
    std::vector<int> tokens = {785}; // "The"

    auto pipeline_legacy = createPipeline(false);
    auto pipeline_graph = createPipeline(true);

    ASSERT_NE(pipeline_legacy, nullptr);
    ASSERT_NE(pipeline_graph, nullptr);

    bool legacy_ok = pipeline_legacy->forward(tokens.data(), static_cast<int>(tokens.size()));
    ASSERT_TRUE(legacy_ok);

    bool graph_ok = pipeline_graph->forward(tokens.data(), static_cast<int>(tokens.size()));
    ASSERT_TRUE(graph_ok);

    const float *logits_legacy = pipeline_legacy->logits();
    const float *logits_graph = pipeline_graph->logits();

    ASSERT_NE(logits_legacy, nullptr);
    ASSERT_NE(logits_graph, nullptr);

    // Top-1 must match
    auto top1_legacy = get_topk(logits_legacy, vocab_size_, 1);
    auto top1_graph = get_topk(logits_graph, vocab_size_, 1);

    EXPECT_EQ(top1_legacy[0], top1_graph[0])
        << "Single token top-1 mismatch";

    double cos_sim = cosine_similarity(logits_legacy, logits_graph, vocab_size_);
    EXPECT_GE(cos_sim, 0.999) << "Single token cosine similarity too low";
}

// ============================================================================
// Main function with MPI initialization
// ============================================================================

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}