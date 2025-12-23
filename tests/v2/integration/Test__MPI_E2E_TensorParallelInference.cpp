/**
 * @file Test__MPI_E2E_TensorParallelInference.cpp
 * @brief End-to-end integration test for full tensor-parallel inference
 *
 * This is the capstone integration test that validates the entire distributed
 * inference pipeline from topology detection through token generation.
 *
 * Components exercised:
 *   - MPITopology: Device capability exchange, work distribution
 *   - WeightManager: Column/row-parallel weight sharding (QKV, FFN, LM Head)
 *   - Qwen2Graph: Graph building with TP configuration
 *   - GraphOrchestrator: Full forward execution with all stages
 *   - Sharded KV Cache: Per-rank cache with local heads
 *   - AllReduceStage: After Wo projection, FFN down projection
 *   - AllGatherStage: After LM head for logits aggregation
 *
 * Test scenarios:
 *   1. Prefill: Multi-token prompt processing
 *   2. Decode: Single-token autoregressive generation
 *   3. Multi-step decode: Verifies KV cache accumulation across ranks
 *
 * Validation criteria:
 *   - All ranks produce identical logits after AllGather
 *   - Top-1 token matches across ranks
 *   - KV cache dimensions are correct for sharded heads
 *
 * @author David Sanftenberg
 * @date December 2025
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <set>
#include <iomanip>
#include <sstream>

#include "execution/InferenceRunnerFactory.h"
#include "execution/IInferenceRunner.h"
#include "execution/GraphOrchestrator.h"
#include "models/qwen/Qwen2Graph.h"
#include "loaders/ModelContext.h"
#include "loaders/WeightManager.h"
#include "utils/MPIContext.h"
#include "utils/MPITopology.h"
#include "utils/Logger.h"
#include "utils/DebugEnv.h"
#include "tensors/TensorFactory.h"
#include "tensors/UnifiedKVCache.h"

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
     * @brief Compute max absolute difference
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
     * @brief Get top-K token indices
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

} // namespace

/**
 * @brief Test fixture for E2E tensor-parallel inference
 */
class Test__MPI_E2E_TensorParallelInference : public ::testing::Test
{
protected:
    std::shared_ptr<ModelContext> model_ctx_;
    std::shared_ptr<MPIContext> mpi_ctx_;
    std::unique_ptr<MPITopology> topology_;
    std::string model_path_;
    int rank_ = 0;
    int world_size_ = 1;

    // Model parameters (Qwen2.5 0.5B)
    int n_heads_ = 14;
    int n_kv_heads_ = 2;
    int head_dim_ = 64;
    int d_model_ = 896;
    int d_ff_ = 4864;
    int vocab_size_ = 151936;
    int n_layers_ = 24;

    // Test token IDs (Qwen2.5 tokenizer: "The quick brown fox")
    std::vector<int32_t> test_tokens_ = {785, 3974, 13876, 38835};

    void SetUp() override
    {
        // Initialize device manager (required before creating inference runner)
        DeviceManager::instance().initialize(-1); // -1 = no NUMA filtering

        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

        // Require 2 ranks for tensor parallelism
        if (world_size_ < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 MPI ranks";
        }

        mpi_ctx_ = std::make_shared<MPIContext>(rank_, world_size_, MPI_COMM_WORLD);

        // Create MPITopology - triggers capability exchange
        topology_ = std::make_unique<MPITopology>(MPI_COMM_WORLD);

        // Load model with SHARDED strategy for tensor parallelism
        model_path_ = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
        model_ctx_ = ModelContext::create(
            model_path_,
            mpi_ctx_,
            nullptr,                            // placement_map
            nullptr,                            // factory
            WeightDistributionStrategy::SHARDED // Enable weight sharding
        );
        if (!model_ctx_)
        {
            GTEST_SKIP() << "Could not load model: " << model_path_;
        }

        // Validate model dimensions match expected
        const auto &model = model_ctx_->loader().getModel();
        ASSERT_EQ(model.head_count, n_heads_);
        ASSERT_EQ(model.head_count_kv, n_kv_heads_);
        ASSERT_EQ(model.embedding_length / model.head_count, head_dim_);
        ASSERT_EQ(model.embedding_length, d_model_);

        LOG_INFO("[Rank " << rank_ << "] Test setup complete, world_size=" << world_size_);
    }

    void TearDown() override
    {
        model_ctx_.reset();
        topology_.reset();
        mpi_ctx_->barrier();
    }

    /**
     * @brief Create InferenceRunner with tensor parallelism enabled
     */
    std::unique_ptr<IInferenceRunner> createTPRunner(int max_seq_len = 128)
    {
        InferenceRunnerConfig config;
        config.batch_size = 1;
        config.max_seq_len = max_seq_len;

        // InferenceRunner auto-detects TP from mpi_ctx->world_size() > 1
        int cpu_device = DeviceManager::instance().cpuDeviceIndex();
        return createInferenceRunner(model_ctx_, mpi_ctx_, cpu_device, config);
    }

    /**
     * @brief Verify all ranks have identical logits after AllGather
     */
    void verifyLogitsParity(const float *local_logits, size_t vocab_size)
    {
        // Gather all logits to all ranks for comparison
        std::vector<float> all_logits(world_size_ * vocab_size);
        MPI_Allgather(
            local_logits, static_cast<int>(vocab_size), MPI_FLOAT,
            all_logits.data(), static_cast<int>(vocab_size), MPI_FLOAT,
            MPI_COMM_WORLD);

        // Compare rank 0's logits with all other ranks
        const float *ref_logits = all_logits.data(); // rank 0
        for (int r = 1; r < world_size_; ++r)
        {
            const float *cmp_logits = all_logits.data() + r * vocab_size;
            float max_diff = max_abs_diff(ref_logits, cmp_logits, vocab_size);
            double cosine = cosine_similarity(ref_logits, cmp_logits, vocab_size);

            LOG_DEBUG("[Rank " << rank_ << "] Comparing with rank " << r
                               << ": max_diff=" << max_diff << ", cosine=" << cosine);

            // After AllGather, logits should be identical across ranks
            EXPECT_LT(max_diff, 1e-4f)
                << "Rank 0 vs Rank " << r << " max_diff too large";
            EXPECT_GT(cosine, 0.9999)
                << "Rank 0 vs Rank " << r << " cosine similarity too low";
        }
    }

    /**
     * @brief Verify top-1 token is identical across ranks
     */
    void verifyTopTokenParity(const float *local_logits, size_t vocab_size)
    {
        auto top5 = get_topk(local_logits, vocab_size, 5);
        int local_top1 = top5[0];

        // Gather all top-1 tokens
        std::vector<int> all_top1(world_size_);
        MPI_Allgather(
            &local_top1, 1, MPI_INT,
            all_top1.data(), 1, MPI_INT,
            MPI_COMM_WORLD);

        // Verify all ranks agree on top-1
        for (int r = 0; r < world_size_; ++r)
        {
            EXPECT_EQ(all_top1[r], all_top1[0])
                << "Rank " << r << " disagrees on top-1 token";
        }

        LOG_INFO("[Rank " << rank_ << "] All ranks agree on top-1 token: " << all_top1[0]);
    }
};

// =============================================================================
// Component Verification Tests
// =============================================================================

TEST_F(Test__MPI_E2E_TensorParallelInference, TopologyExchangeSucceeds)
{
    // Verify topology is properly initialized across all ranks
    EXPECT_EQ(topology_->world_size(), world_size_);
    EXPECT_EQ(topology_->rank(), rank_);

    // All ranks should be compute participants
    EXPECT_TRUE(topology_->is_compute_participant());

    // Verify we have placement info for all ranks
    const auto &placements = topology_->all_placements();
    EXPECT_EQ(placements.size(), static_cast<size_t>(world_size_));

    mpi_ctx_->barrier();
    LOG_INFO("[Rank " << rank_ << "] TopologyExchange verified");
}

TEST_F(Test__MPI_E2E_TensorParallelInference, WeightShardingIsCorrect)
{
    // Verify weight manager applies correct sharding strategy
    auto weight_mgr = model_ctx_->weightManager();
    ASSERT_NE(weight_mgr, nullptr);

    // Check column-parallel Q weight (first layer)
    auto wq = weight_mgr->getWeight("blk.0.attn_q.weight");
    ASSERT_NE(wq, nullptr);

    // Expected: [local_n_heads * head_dim, d_model]
    int local_n_heads = n_heads_ / world_size_;
    size_t expected_rows = local_n_heads * head_dim_;
    EXPECT_EQ(wq->shape()[0], expected_rows)
        << "Q weight should be column-sharded";

    // Check column-parallel K weight
    auto wk = weight_mgr->getWeight("blk.0.attn_k.weight");
    ASSERT_NE(wk, nullptr);
    int local_n_kv_heads = n_kv_heads_ / world_size_;
    size_t expected_kv_rows = local_n_kv_heads * head_dim_;
    EXPECT_EQ(wk->shape()[0], expected_kv_rows)
        << "K weight should be column-sharded";

    // Check column-parallel Gate weight (FFN)
    auto gate = weight_mgr->getWeight("blk.0.ffn_gate.weight");
    ASSERT_NE(gate, nullptr);
    size_t expected_gate_rows = d_ff_ / world_size_;
    EXPECT_EQ(gate->shape()[0], expected_gate_rows)
        << "Gate weight should be column-sharded";

    // Check column-parallel LM head
    auto lm_head = weight_mgr->getWeight("output.weight");
    ASSERT_NE(lm_head, nullptr);
    size_t expected_vocab_local = vocab_size_ / world_size_;
    EXPECT_EQ(lm_head->shape()[0], expected_vocab_local)
        << "LM head should be column-sharded by vocab";

    // Verify token embeddings are NOT sharded (replicated)
    auto emb = weight_mgr->getWeight("token_embd.weight");
    ASSERT_NE(emb, nullptr);
    EXPECT_EQ(emb->shape()[0], static_cast<size_t>(vocab_size_))
        << "Embeddings should be replicated (full vocab)";

    mpi_ctx_->barrier();
    LOG_INFO("[Rank " << rank_ << "] Weight sharding verified");
}

TEST_F(Test__MPI_E2E_TensorParallelInference, InferenceRunnerCreatesWithTP)
{
    // Create runner - should auto-detect TP from world_size > 1
    auto runner = createTPRunner();
    ASSERT_NE(runner, nullptr);

    // Cast to GraphOrchestrator to verify internal config
    auto *orchestrator = dynamic_cast<GraphOrchestrator *>(runner.get());
    ASSERT_NE(orchestrator, nullptr) << "Runner should be a GraphOrchestrator";

    // Verify TP configuration via graphBuilder
    const auto *graph = orchestrator->graphBuilder();
    ASSERT_NE(graph, nullptr);
    const auto &config = graph->config();

    EXPECT_TRUE(config.qkv_column_parallel) << "QKV should be column-parallel";
    EXPECT_TRUE(config.ffn_column_parallel) << "FFN should be column-parallel";
    EXPECT_TRUE(config.lm_head_column_parallel) << "LM head should be column-parallel";

    // Verify local dimensions
    EXPECT_EQ(config.local_n_heads, n_heads_ / world_size_);
    EXPECT_EQ(config.local_n_kv_heads, n_kv_heads_ / world_size_);
    EXPECT_EQ(config.d_ff_local, d_ff_ / world_size_);
    EXPECT_EQ(config.vocab_local, vocab_size_ / world_size_);

    mpi_ctx_->barrier();
    LOG_INFO("[Rank " << rank_ << "] TP configuration verified");
}

// =============================================================================
// Inference Tests
// =============================================================================

TEST_F(Test__MPI_E2E_TensorParallelInference, PrefillProducesConsistentLogits)
{
    auto runner = createTPRunner();
    ASSERT_NE(runner, nullptr);

    // Run prefill with test tokens
    int seq_len = static_cast<int>(test_tokens_.size());
    bool success = runner->forward(test_tokens_.data(), seq_len);
    ASSERT_TRUE(success) << "Forward pass failed on rank " << rank_;

    // Get logits - returns const float* pointing to [seq_len * vocab_size]
    const float *logits_ptr = runner->logits();
    ASSERT_NE(logits_ptr, nullptr);

    // Get last position logits (next token prediction)
    const float *last_logits = logits_ptr + (seq_len - 1) * vocab_size_;

    // Verify all ranks have identical logits
    verifyLogitsParity(last_logits, vocab_size_);
    verifyTopTokenParity(last_logits, vocab_size_);

    mpi_ctx_->barrier();
    LOG_INFO("[Rank " << rank_ << "] Prefill parity verified");
}

TEST_F(Test__MPI_E2E_TensorParallelInference, DecodeProducesConsistentLogits)
{
    auto runner = createTPRunner();
    ASSERT_NE(runner, nullptr);

    // First, run prefill
    int seq_len = static_cast<int>(test_tokens_.size());
    ASSERT_TRUE(runner->forward(test_tokens_.data(), seq_len));

    // Get predicted next token (all ranks should agree)
    const float *logits_ptr = runner->logits();
    const float *last_logits = logits_ptr + (seq_len - 1) * vocab_size_;
    auto top5_prefill = get_topk(last_logits, vocab_size_, 5);
    int next_token = top5_prefill[0];

    // Run decode with the predicted token
    int32_t decode_token = static_cast<int32_t>(next_token);
    ASSERT_TRUE(runner->forward(&decode_token, 1));

    // Get decode logits (should be just [vocab_size] for single token)
    const float *decode_logits_ptr = runner->logits();
    ASSERT_NE(decode_logits_ptr, nullptr);

    // Verify all ranks have identical decode logits
    verifyLogitsParity(decode_logits_ptr, vocab_size_);
    verifyTopTokenParity(decode_logits_ptr, vocab_size_);

    mpi_ctx_->barrier();
    LOG_INFO("[Rank " << rank_ << "] Decode parity verified");
}

TEST_F(Test__MPI_E2E_TensorParallelInference, MultiStepDecodeWithKVCache)
{
    auto runner = createTPRunner();
    ASSERT_NE(runner, nullptr);

    // Prefill
    int seq_len = static_cast<int>(test_tokens_.size());
    ASSERT_TRUE(runner->forward(test_tokens_.data(), seq_len));

    // Multi-step decode (5 tokens)
    constexpr int NUM_DECODE_STEPS = 5;
    std::vector<int> generated_tokens;

    for (int step = 0; step < NUM_DECODE_STEPS; ++step)
    {
        const float *logits_ptr = runner->logits();
        ASSERT_NE(logits_ptr, nullptr);

        // Get top-1 token
        // On prefill (step 0), logits are [seq_len, vocab_size] - take last position
        // On decode (step > 0), logits are [1, vocab_size] - take position 0
        size_t logits_offset = (step == 0 && seq_len > 1)
                                   ? (seq_len - 1) * vocab_size_
                                   : 0;
        const float *last_logits = logits_ptr + logits_offset;
        auto top5 = get_topk(last_logits, vocab_size_, 5);
        int next_token = top5[0];

        // Verify all ranks agree on next token
        std::vector<int> all_tokens(world_size_);
        MPI_Allgather(
            &next_token, 1, MPI_INT,
            all_tokens.data(), 1, MPI_INT,
            MPI_COMM_WORLD);

        for (int r = 0; r < world_size_; ++r)
        {
            EXPECT_EQ(all_tokens[r], next_token)
                << "Step " << step << ": Rank " << r << " disagrees on token";
        }

        generated_tokens.push_back(next_token);

        // Feed next token for decode
        if (step < NUM_DECODE_STEPS - 1)
        {
            int32_t decode_token = static_cast<int32_t>(next_token);
            ASSERT_TRUE(runner->forward(&decode_token, 1))
                << "Decode step " << step << " failed";
        }
    }

    // Log generated sequence
    if (rank_ == 0)
    {
        std::ostringstream oss;
        oss << "Generated tokens: [";
        for (size_t i = 0; i < generated_tokens.size(); ++i)
        {
            if (i > 0)
                oss << ", ";
            oss << generated_tokens[i];
        }
        oss << "]";
        LOG_INFO(oss.str());
    }

    mpi_ctx_->barrier();
    LOG_INFO("[Rank " << rank_ << "] Multi-step decode verified");
}

TEST_F(Test__MPI_E2E_TensorParallelInference, KVCacheIsShardedCorrectly)
{
    auto runner = createTPRunner(128);
    ASSERT_NE(runner, nullptr);

    auto *orchestrator = dynamic_cast<GraphOrchestrator *>(runner.get());
    ASSERT_NE(orchestrator, nullptr);

    // Run prefill to populate KV cache
    int seq_len = static_cast<int>(test_tokens_.size());
    ASSERT_TRUE(runner->forward(test_tokens_.data(), seq_len));

    // Access KV cache via orchestrator's inference state
    const auto &state = orchestrator->inferenceState();
    auto *kv_cache = state.kv_cache.get();
    ASSERT_NE(kv_cache, nullptr);

    // Verify cache is sharded
    EXPECT_TRUE(kv_cache->is_sharded())
        << "KV cache should be sharded for TP";

    // Verify local dimensions
    int local_n_kv_heads = n_kv_heads_ / world_size_;
    EXPECT_EQ(kv_cache->local_n_kv_heads(), local_n_kv_heads);
    EXPECT_EQ(kv_cache->local_kv_dim(), local_n_kv_heads * head_dim_);

    // Verify kv_head_start is correct per rank
    int expected_kv_head_start = rank_ * local_n_kv_heads;
    EXPECT_EQ(kv_cache->kv_head_start(), expected_kv_head_start);

    // Verify sequence positions match across ranks (layer 0, seq_idx 0)
    int local_pos = kv_cache->get_cached_tokens(0, 0);
    std::vector<int> all_positions(world_size_);
    MPI_Allgather(
        &local_pos, 1, MPI_INT,
        all_positions.data(), 1, MPI_INT,
        MPI_COMM_WORLD);

    for (int r = 0; r < world_size_; ++r)
    {
        EXPECT_EQ(all_positions[r], seq_len)
            << "Rank " << r << " has wrong KV cache position";
    }

    mpi_ctx_->barrier();
    LOG_INFO("[Rank " << rank_ << "] Sharded KV cache verified");
}

// =============================================================================
// Stress Tests
// =============================================================================

TEST_F(Test__MPI_E2E_TensorParallelInference, LongSequencePrefill)
{
    // Test with longer sequence (up to model max)
    auto runner = createTPRunner(512);
    ASSERT_NE(runner, nullptr);

    // Create a longer token sequence (64 tokens)
    std::vector<int32_t> long_tokens(64);
    for (size_t i = 0; i < long_tokens.size(); ++i)
    {
        long_tokens[i] = test_tokens_[i % test_tokens_.size()];
    }

    int seq_len = static_cast<int>(long_tokens.size());
    ASSERT_TRUE(runner->forward(long_tokens.data(), seq_len));

    const float *logits_ptr = runner->logits();
    ASSERT_NE(logits_ptr, nullptr);

    // Verify last position logits
    const float *last_logits = logits_ptr + (seq_len - 1) * vocab_size_;
    verifyLogitsParity(last_logits, vocab_size_);

    mpi_ctx_->barrier();
    LOG_INFO("[Rank " << rank_ << "] Long sequence prefill verified");
}

TEST_F(Test__MPI_E2E_TensorParallelInference, RepeatedForwardCalls)
{
    // Test multiple forward calls with cache clear between
    auto runner = createTPRunner();
    ASSERT_NE(runner, nullptr);

    for (int iter = 0; iter < 3; ++iter)
    {
        // Clear cache
        runner->clear_cache();

        // Run forward
        int seq_len = static_cast<int>(test_tokens_.size());
        ASSERT_TRUE(runner->forward(test_tokens_.data(), seq_len))
            << "Iteration " << iter << " failed";

        // Verify logits
        const float *logits_ptr = runner->logits();
        ASSERT_NE(logits_ptr, nullptr);

        const float *last_logits = logits_ptr + (seq_len - 1) * vocab_size_;
        verifyTopTokenParity(last_logits, vocab_size_);

        mpi_ctx_->barrier();
    }

    LOG_INFO("[Rank " << rank_ << "] Repeated forward calls verified");
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char **argv)
{
    // Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    // Initialize GTest
    ::testing::InitGoogleTest(&argc, argv);

    // Run tests
    int result = RUN_ALL_TESTS();

    // Finalize MPI
    MPI_Finalize();

    return result;
}
