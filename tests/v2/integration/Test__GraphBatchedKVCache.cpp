/**
 * @file Test__GraphBatchedKVCache.cpp
 * @brief Integration tests for Graph-based batched KV cache handling
 * @author David Sanftenberg
 *
 * Tests cover:
 * - KVCacheAppendStage correctly appends K/V to separate seq_idx per batch sequence
 * - Batched prefill populates KV cache slots independently
 * - Sequential execution matches batched execution for KV cache contents
 * - Per-sequence cache isolation (modifications to seq[i] don't affect seq[j])
 *
 * This is Phase 3 of the Graph Batching Implementation Plan.
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <cmath>
#include <numeric>

#include "v2/tensors/UnifiedKVCache.h"
#include "v2/tensors/Tensors.h"
#include "v2/models/qwen/Qwen2Graph.h"
#include "execution/compute_stages/ComputeStages.h"
#include "v2/loaders/ModelContext.h"
#include "v2/utils/MPIContext.h"
#include "v2/utils/Logger.h"
#include "v2/utils/Tokenizer.h"

using namespace llaminar2;

/**
 * @class GraphBatchedKVCacheTest
 * @brief Integration test fixture for Graph-based batched KV cache
 *
 * Tests validate that KV cache stages handle batch_size > 1 correctly,
 * appending K/V tensors to separate sequence indices.
 */
class GraphBatchedKVCacheTest : public ::testing::Test
{
protected:
    std::shared_ptr<ModelContext> model_ctx_;
    std::shared_ptr<MPIContext> mpi_ctx_;
    std::string model_path_;
    int rank_;
    int world_size_;

    // Model dimensions (populated from model context in SetUp)
    int n_layers_;
    int n_kv_heads_;
    int head_dim_;
    int kv_dim_;
    int d_model_;

    void SetUp() override
    {
        // Initialize MPI context
        int rank, world_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        rank_ = rank;
        world_size_ = world_size;

        mpi_ctx_ = std::make_shared<MPIContext>(rank, world_size, MPI_COMM_WORLD);

        // Use small Qwen model for fast testing
        model_path_ = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

        // Load model (collective operation)
        model_ctx_ = ModelContext::create(model_path_, mpi_ctx_);

        if (!model_ctx_)
        {
            GTEST_SKIP() << "Model not found: " << model_path_;
        }

        // Extract model dimensions
        const auto &model = model_ctx_->model();
        n_layers_ = static_cast<int>(model.block_count);
        n_kv_heads_ = static_cast<int>(model.head_count_kv);
        d_model_ = static_cast<int>(model.embedding_length);
        head_dim_ = d_model_ / static_cast<int>(model.head_count);
        kv_dim_ = n_kv_heads_ * head_dim_;

        if (rank_ == 0)
        {
            LOG_INFO("[GraphBatchedKVCache] Loaded model: " << model_path_);
            LOG_INFO("[GraphBatchedKVCache] n_layers=" << n_layers_
                                                       << ", n_kv_heads=" << n_kv_heads_
                                                       << ", head_dim=" << head_dim_
                                                       << ", kv_dim=" << kv_dim_);
        }
    }

    void TearDown() override
    {
        model_ctx_.reset();
        mpi_ctx_->barrier();
    }

    /**
     * @brief Create an FP32 KV cache for testing
     */
    std::unique_ptr<UnifiedKVCache<ActivationPrecision::FP32>> createTestKVCache(
        int max_seq_len, int batch_size)
    {
        return std::make_unique<UnifiedKVCache<ActivationPrecision::FP32>>(
            *mpi_ctx_,
            n_layers_,
            batch_size,
            max_seq_len,
            n_kv_heads_,
            head_dim_,
            -1); // CPU device
    }

    /**
     * @brief Execute a compute stage (passes nullptr for device context, stages handle this)
     */
    bool executeStage(IComputeStage *stage)
    {
        return stage->execute(nullptr);
    }

    /**
     * @brief Create test K/V tensors with known patterns
     *
     * Pattern: K[b,t,h,d] = b * 1000 + t * 100 + h * 10 + d
     *          V[b,t,h,d] = -(b * 1000 + t * 100 + h * 10 + d)
     *
     * This makes it easy to verify which sequence/position the data came from.
     */
    std::pair<std::unique_ptr<FP32Tensor>, std::unique_ptr<FP32Tensor>>
    createTestKVTensors(int batch_size, int seq_len)
    {
        const size_t total_tokens = batch_size * seq_len;

        auto K = std::make_unique<FP32Tensor>(
            std::vector<size_t>{total_tokens, static_cast<size_t>(kv_dim_)});
        auto V = std::make_unique<FP32Tensor>(
            std::vector<size_t>{total_tokens, static_cast<size_t>(kv_dim_)});

        float *k_data = K->mutable_data();
        float *v_data = V->mutable_data();

        // Fill with identifiable patterns
        for (int b = 0; b < batch_size; ++b)
        {
            for (int t = 0; t < seq_len; ++t)
            {
                for (int h = 0; h < n_kv_heads_; ++h)
                {
                    for (int d = 0; d < head_dim_; ++d)
                    {
                        size_t flat_idx =
                            ((b * seq_len + t) * n_kv_heads_ + h) * head_dim_ + d;

                        float pattern = static_cast<float>(
                            b * 1000 + t * 100 + h * 10 + d);
                        k_data[flat_idx] = pattern;
                        v_data[flat_idx] = -pattern;
                    }
                }
            }
        }

        return {std::move(K), std::move(V)};
    }

    /**
     * @brief Verify cache contents match expected pattern for a specific sequence
     */
    void verifyCacheSequence(
        UnifiedKVCache<ActivationPrecision::FP32> *cache,
        int layer_idx,
        int seq_idx,
        int expected_batch_idx,
        int expected_seq_len,
        const std::string &test_name)
    {
        int cached_tokens = cache->get_cached_tokens(layer_idx, seq_idx);
        ASSERT_EQ(cached_tokens, expected_seq_len)
            << test_name << ": Wrong cached_tokens for layer " << layer_idx
            << " seq " << seq_idx;

        const float *k_base = cache->get_k_base(layer_idx, seq_idx)->data();
        const float *v_base = cache->get_v_base(layer_idx, seq_idx)->data();

        // Verify each position's pattern
        for (int t = 0; t < expected_seq_len; ++t)
        {
            for (int h = 0; h < n_kv_heads_; ++h)
            {
                for (int d = 0; d < head_dim_; ++d)
                {
                    size_t cache_idx = (t * n_kv_heads_ + h) * head_dim_ + d;

                    float expected_k = static_cast<float>(
                        expected_batch_idx * 1000 + t * 100 + h * 10 + d);
                    float expected_v = -expected_k;

                    EXPECT_NEAR(k_base[cache_idx], expected_k, 1e-5f)
                        << test_name << ": K mismatch at layer=" << layer_idx
                        << " seq=" << seq_idx << " t=" << t << " h=" << h << " d=" << d;
                    EXPECT_NEAR(v_base[cache_idx], expected_v, 1e-5f)
                        << test_name << ": V mismatch at layer=" << layer_idx
                        << " seq=" << seq_idx << " t=" << t << " h=" << h << " d=" << d;
                }
            }
        }
    }
};

/**
 * @test KVCacheAppendStage_BatchedAppend
 * @brief Verify KVCacheAppendStage correctly appends to separate seq_idx per batch
 *
 * Validates:
 * - batch_size=2 append creates entries in seq_idx=0 and seq_idx=1
 * - Each sequence gets its correct slice of the K/V tensors
 * - Cached token counts are correct per sequence
 */
TEST_F(GraphBatchedKVCacheTest, KVCacheAppendStage_BatchedAppend)
{
    const int batch_size = 2;
    const int seq_len = 3;      // 3 tokens per sequence
    const int max_seq_len = 64; // Max cache capacity
    const int test_layer = 0;

    // Create cache with batch_size slots
    auto cache = createTestKVCache(max_seq_len, batch_size);

    // Create test K/V tensors [batch_size * seq_len, kv_dim]
    auto [K, V] = createTestKVTensors(batch_size, seq_len);

    // Configure KVCacheAppendStage
    KVCacheAppendStage::Params params;
    params.K = K.get();
    params.V = V.get();
    params.kv_cache = cache.get();
    params.layer_idx = test_layer;
    params.seq_idx = 0; // Base sequence index
    params.batch_size = batch_size;
    params.seq_len = seq_len;

    // Execute append
    auto stage = std::make_unique<KVCacheAppendStage>(params);
    bool success = executeStage(stage.get());

    ASSERT_TRUE(success) << "KVCacheAppendStage::execute() failed";

    // Verify sequence 0 has correct data (from batch index 0)
    verifyCacheSequence(cache.get(), test_layer, 0, /*expected_batch_idx=*/0,
                        seq_len, "KVCacheAppend_Seq0");

    // Verify sequence 1 has correct data (from batch index 1)
    verifyCacheSequence(cache.get(), test_layer, 1, /*expected_batch_idx=*/1,
                        seq_len, "KVCacheAppend_Seq1");

    if (rank_ == 0)
    {
        LOG_INFO("[BatchedAppend] ✓ Batch append correctly separated K/V to seq_idx 0 and 1");
    }
}

/**
 * @test KVCacheAppendStage_SingleSequenceFallback
 * @brief Verify batch_size=1 uses original single-sequence path
 *
 * Regression test to ensure we didn't break the single-sequence case.
 */
TEST_F(GraphBatchedKVCacheTest, KVCacheAppendStage_SingleSequenceFallback)
{
    const int batch_size = 1;
    const int seq_len = 4;
    const int max_seq_len = 64;
    const int test_layer = 0;

    auto cache = createTestKVCache(max_seq_len, batch_size);
    auto [K, V] = createTestKVTensors(batch_size, seq_len);

    KVCacheAppendStage::Params params;
    params.K = K.get();
    params.V = V.get();
    params.kv_cache = cache.get();
    params.layer_idx = test_layer;
    params.seq_idx = 0;
    params.batch_size = batch_size;
    params.seq_len = seq_len;

    auto stage = std::make_unique<KVCacheAppendStage>(params);
    bool success = executeStage(stage.get());

    ASSERT_TRUE(success) << "KVCacheAppendStage::execute() failed for batch_size=1";

    verifyCacheSequence(cache.get(), test_layer, 0, /*expected_batch_idx=*/0,
                        seq_len, "SingleSeq");

    if (rank_ == 0)
    {
        LOG_INFO("[SingleSequence] ✓ batch_size=1 fallback works correctly");
    }
}

/**
 * @test KVCacheAppendStage_MultiLayer
 * @brief Verify batched append works correctly across multiple layers
 *
 * Tests that each layer gets its own independent cache entries.
 */
TEST_F(GraphBatchedKVCacheTest, KVCacheAppendStage_MultiLayer)
{
    const int batch_size = 2;
    const int seq_len = 2;
    const int max_seq_len = 32;
    const int test_layers[] = {0, 5, n_layers_ - 1}; // First, middle, last

    auto cache = createTestKVCache(max_seq_len, batch_size);
    auto [K, V] = createTestKVTensors(batch_size, seq_len);

    // Append same K/V to multiple layers (as would happen in forward pass)
    for (int layer_idx : test_layers)
    {
        KVCacheAppendStage::Params params;
        params.K = K.get();
        params.V = V.get();
        params.kv_cache = cache.get();
        params.layer_idx = layer_idx;
        params.seq_idx = 0;
        params.batch_size = batch_size;
        params.seq_len = seq_len;

        auto stage = std::make_unique<KVCacheAppendStage>(params);
        ASSERT_TRUE(executeStage(stage.get()))
            << "KVCacheAppendStage::execute() failed for layer " << layer_idx;
    }

    // Verify each layer has correct data
    for (int layer_idx : test_layers)
    {
        verifyCacheSequence(cache.get(), layer_idx, 0, 0, seq_len,
                            "MultiLayer_L" + std::to_string(layer_idx) + "_Seq0");
        verifyCacheSequence(cache.get(), layer_idx, 1, 1, seq_len,
                            "MultiLayer_L" + std::to_string(layer_idx) + "_Seq1");
    }

    if (rank_ == 0)
    {
        LOG_INFO("[MultiLayer] ✓ Batched append works across all layers");
    }
}

/**
 * @test BatchedPrefill_SequenceIsolation
 * @brief Verify that batched sequences don't interfere with each other
 *
 * Test: Run batched prefill, then verify seq[0] and seq[1] have distinct data.
 */
TEST_F(GraphBatchedKVCacheTest, BatchedPrefill_SequenceIsolation)
{
    const int batch_size = 2;
    const int seq_len = 4;
    const int max_seq_len = 64;

    auto cache = createTestKVCache(max_seq_len, batch_size);

    // Create K/V with deliberately different patterns per sequence
    auto K = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(batch_size * seq_len),
                            static_cast<size_t>(kv_dim_)});
    auto V = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(batch_size * seq_len),
                            static_cast<size_t>(kv_dim_)});

    float *k_data = K->mutable_data();
    float *v_data = V->mutable_data();

    // Sequence 0: all 1.0 for K, all 2.0 for V
    // Sequence 1: all 100.0 for K, all 200.0 for V
    for (int b = 0; b < batch_size; ++b)
    {
        float k_val = (b == 0) ? 1.0f : 100.0f;
        float v_val = (b == 0) ? 2.0f : 200.0f;

        for (int t = 0; t < seq_len; ++t)
        {
            for (int d = 0; d < kv_dim_; ++d)
            {
                size_t idx = (b * seq_len + t) * kv_dim_ + d;
                k_data[idx] = k_val;
                v_data[idx] = v_val;
            }
        }
    }

    // Run batched append for layer 0
    KVCacheAppendStage::Params params;
    params.K = K.get();
    params.V = V.get();
    params.kv_cache = cache.get();
    params.layer_idx = 0;
    params.seq_idx = 0;
    params.batch_size = batch_size;
    params.seq_len = seq_len;

    auto stage = std::make_unique<KVCacheAppendStage>(params);
    ASSERT_TRUE(executeStage(stage.get()));

    // Verify isolation: seq 0 should have 1.0/2.0, seq 1 should have 100.0/200.0
    const float *k0 = cache->get_k_base(0, 0)->data();
    const float *v0 = cache->get_v_base(0, 0)->data();
    const float *k1 = cache->get_k_base(0, 1)->data();
    const float *v1 = cache->get_v_base(0, 1)->data();

    for (int t = 0; t < seq_len; ++t)
    {
        for (int d = 0; d < kv_dim_; ++d)
        {
            size_t idx = t * kv_dim_ + d;

            EXPECT_NEAR(k0[idx], 1.0f, 1e-5f)
                << "Seq 0 K contaminated at t=" << t << " d=" << d;
            EXPECT_NEAR(v0[idx], 2.0f, 1e-5f)
                << "Seq 0 V contaminated at t=" << t << " d=" << d;
            EXPECT_NEAR(k1[idx], 100.0f, 1e-5f)
                << "Seq 1 K contaminated at t=" << t << " d=" << d;
            EXPECT_NEAR(v1[idx], 200.0f, 1e-5f)
                << "Seq 1 V contaminated at t=" << t << " d=" << d;
        }
    }

    if (rank_ == 0)
    {
        LOG_INFO("[SequenceIsolation] ✓ Batch sequences have isolated cache entries");
    }
}

/**
 * @test Sequential_vs_Batched_CacheParity
 * @brief Verify sequential execution produces same cache as batched execution
 *
 * This is the key correctness test:
 * 1. Run two sequences sequentially (seq_idx=0, seq_idx=1)
 * 2. Run the same two sequences batched (batch_size=2)
 * 3. Compare cache contents - they should match exactly
 */
TEST_F(GraphBatchedKVCacheTest, Sequential_vs_Batched_CacheParity)
{
    const int seq_len = 3;
    const int max_seq_len = 64;
    const int test_layer = 0;

    // Create reference cache (sequential execution)
    auto seq_cache = createTestKVCache(max_seq_len, 2);

    // Create batched cache
    auto batch_cache = createTestKVCache(max_seq_len, 2);

    // --- Sequential Execution ---
    // Sequence 0 first
    {
        auto [K0, V0] = createTestKVTensors(1, seq_len);
        // Adjust pattern to match batch_idx=0
        float *k_data = K0->mutable_data();
        float *v_data = V0->mutable_data();
        for (int t = 0; t < seq_len; ++t)
        {
            for (int h = 0; h < n_kv_heads_; ++h)
            {
                for (int d = 0; d < head_dim_; ++d)
                {
                    size_t idx = (t * n_kv_heads_ + h) * head_dim_ + d;
                    float pattern = static_cast<float>(
                        0 * 1000 + t * 100 + h * 10 + d); // batch_idx = 0
                    k_data[idx] = pattern;
                    v_data[idx] = -pattern;
                }
            }
        }

        KVCacheAppendStage::Params params;
        params.K = K0.get();
        params.V = V0.get();
        params.kv_cache = seq_cache.get();
        params.layer_idx = test_layer;
        params.seq_idx = 0;
        params.batch_size = 1;
        params.seq_len = seq_len;

        auto stage = std::make_unique<KVCacheAppendStage>(params);
        ASSERT_TRUE(executeStage(stage.get()));
    }

    // Sequence 1 second
    {
        auto [K1, V1] = createTestKVTensors(1, seq_len);
        // Adjust pattern to match batch_idx=1
        float *k_data = K1->mutable_data();
        float *v_data = V1->mutable_data();
        for (int t = 0; t < seq_len; ++t)
        {
            for (int h = 0; h < n_kv_heads_; ++h)
            {
                for (int d = 0; d < head_dim_; ++d)
                {
                    size_t idx = (t * n_kv_heads_ + h) * head_dim_ + d;
                    float pattern = static_cast<float>(
                        1 * 1000 + t * 100 + h * 10 + d); // batch_idx = 1
                    k_data[idx] = pattern;
                    v_data[idx] = -pattern;
                }
            }
        }

        KVCacheAppendStage::Params params;
        params.K = K1.get();
        params.V = V1.get();
        params.kv_cache = seq_cache.get();
        params.layer_idx = test_layer;
        params.seq_idx = 1;
        params.batch_size = 1;
        params.seq_len = seq_len;

        auto stage = std::make_unique<KVCacheAppendStage>(params);
        ASSERT_TRUE(executeStage(stage.get()));
    }

    // --- Batched Execution ---
    {
        auto [K_batch, V_batch] = createTestKVTensors(2, seq_len);

        KVCacheAppendStage::Params params;
        params.K = K_batch.get();
        params.V = V_batch.get();
        params.kv_cache = batch_cache.get();
        params.layer_idx = test_layer;
        params.seq_idx = 0;
        params.batch_size = 2;
        params.seq_len = seq_len;

        auto stage = std::make_unique<KVCacheAppendStage>(params);
        ASSERT_TRUE(executeStage(stage.get()));
    }

    // --- Compare Results ---
    // Sequence 0 comparison
    const float *seq_k0 = seq_cache->get_k_base(test_layer, 0)->data();
    const float *seq_v0 = seq_cache->get_v_base(test_layer, 0)->data();
    const float *batch_k0 = batch_cache->get_k_base(test_layer, 0)->data();
    const float *batch_v0 = batch_cache->get_v_base(test_layer, 0)->data();

    int seq0_k_mismatches = 0, seq0_v_mismatches = 0;
    for (int t = 0; t < seq_len; ++t)
    {
        for (int d = 0; d < kv_dim_; ++d)
        {
            size_t idx = t * kv_dim_ + d;
            if (std::abs(seq_k0[idx] - batch_k0[idx]) > 1e-5f)
                seq0_k_mismatches++;
            if (std::abs(seq_v0[idx] - batch_v0[idx]) > 1e-5f)
                seq0_v_mismatches++;
        }
    }

    EXPECT_EQ(seq0_k_mismatches, 0)
        << "Sequential vs Batched: Seq 0 K has " << seq0_k_mismatches << " mismatches";
    EXPECT_EQ(seq0_v_mismatches, 0)
        << "Sequential vs Batched: Seq 0 V has " << seq0_v_mismatches << " mismatches";

    // Sequence 1 comparison
    const float *seq_k1 = seq_cache->get_k_base(test_layer, 1)->data();
    const float *seq_v1 = seq_cache->get_v_base(test_layer, 1)->data();
    const float *batch_k1 = batch_cache->get_k_base(test_layer, 1)->data();
    const float *batch_v1 = batch_cache->get_v_base(test_layer, 1)->data();

    int seq1_k_mismatches = 0, seq1_v_mismatches = 0;
    for (int t = 0; t < seq_len; ++t)
    {
        for (int d = 0; d < kv_dim_; ++d)
        {
            size_t idx = t * kv_dim_ + d;
            if (std::abs(seq_k1[idx] - batch_k1[idx]) > 1e-5f)
                seq1_k_mismatches++;
            if (std::abs(seq_v1[idx] - batch_v1[idx]) > 1e-5f)
                seq1_v_mismatches++;
        }
    }

    EXPECT_EQ(seq1_k_mismatches, 0)
        << "Sequential vs Batched: Seq 1 K has " << seq1_k_mismatches << " mismatches";
    EXPECT_EQ(seq1_v_mismatches, 0)
        << "Sequential vs Batched: Seq 1 V has " << seq1_v_mismatches << " mismatches";

    if (rank_ == 0)
    {
        LOG_INFO("[SequentialVsBatched] ✓ Sequential and batched execution produce identical cache contents");
    }
}

/**
 * @test KVCacheAppendStage_IncrementalDecodeAppend
 * @brief Verify decode-style append (1 token at a time) works with batching
 *
 * This tests the decode path where we append 1 token per sequence per step.
 */
TEST_F(GraphBatchedKVCacheTest, KVCacheAppendStage_IncrementalDecodeAppend)
{
    const int batch_size = 2;
    const int max_seq_len = 64;
    const int test_layer = 0;
    const int num_decode_steps = 3;

    auto cache = createTestKVCache(max_seq_len, batch_size);

    // Simulate decode: append 1 token per sequence at each step
    for (int step = 0; step < num_decode_steps; ++step)
    {
        // Create K/V for 1 token per sequence [batch_size * 1, kv_dim]
        auto K = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(batch_size), static_cast<size_t>(kv_dim_)});
        auto V = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(batch_size), static_cast<size_t>(kv_dim_)});

        float *k_data = K->mutable_data();
        float *v_data = V->mutable_data();

        for (int b = 0; b < batch_size; ++b)
        {
            for (int d = 0; d < kv_dim_; ++d)
            {
                size_t idx = b * kv_dim_ + d;
                // Pattern: step * 1000 + batch_idx * 100 + d
                float pattern = static_cast<float>(step * 1000 + b * 100 + d);
                k_data[idx] = pattern;
                v_data[idx] = -pattern;
            }
        }

        KVCacheAppendStage::Params params;
        params.K = K.get();
        params.V = V.get();
        params.kv_cache = cache.get();
        params.layer_idx = test_layer;
        params.seq_idx = 0;
        params.batch_size = batch_size;
        params.seq_len = 1; // 1 token per sequence

        auto stage = std::make_unique<KVCacheAppendStage>(params);
        ASSERT_TRUE(executeStage(stage.get())) << "Decode step " << step << " failed";
    }

    // Verify each sequence has accumulated the correct number of tokens
    EXPECT_EQ(cache->get_cached_tokens(test_layer, 0), num_decode_steps)
        << "Seq 0 should have " << num_decode_steps << " cached tokens";
    EXPECT_EQ(cache->get_cached_tokens(test_layer, 1), num_decode_steps)
        << "Seq 1 should have " << num_decode_steps << " cached tokens";

    // Verify last appended data is correct for each sequence
    const float *k0 = cache->get_k_base(test_layer, 0)->data();
    const float *k1 = cache->get_k_base(test_layer, 1)->data();

    // Check last position (step = num_decode_steps - 1)
    int last_step = num_decode_steps - 1;
    for (int d = 0; d < std::min(kv_dim_, 5); ++d)
    { // Sample first 5 elements
        size_t idx = last_step * kv_dim_ + d;
        float expected_k0 = static_cast<float>(last_step * 1000 + 0 * 100 + d);
        float expected_k1 = static_cast<float>(last_step * 1000 + 1 * 100 + d);

        EXPECT_NEAR(k0[idx], expected_k0, 1e-5f)
            << "Seq 0 last token K mismatch at d=" << d;
        EXPECT_NEAR(k1[idx], expected_k1, 1e-5f)
            << "Seq 1 last token K mismatch at d=" << d;
    }

    if (rank_ == 0)
    {
        LOG_INFO("[IncrementalDecode] ✓ Batched decode append works correctly");
    }
}

// =============================================================================
// KVCacheGatherStage Tests
// =============================================================================

/**
 * @test KVCacheGatherStage_BasicGather
 * @brief Verify KVCacheGatherStage correctly gathers K/V from multiple cache slots
 *
 * This tests the core batched decode gather functionality.
 */
TEST_F(GraphBatchedKVCacheTest, KVCacheGatherStage_BasicGather)
{
    const int batch_size = 2;
    const int max_seq_len = 64;
    const int test_layer = 0;
    const int prefill_len = 4;

    // Create cache with batch_size slots
    auto cache = createTestKVCache(max_seq_len, batch_size);

    // Pre-populate cache with different data per sequence
    for (int seq_idx = 0; seq_idx < batch_size; ++seq_idx)
    {
        auto K = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(prefill_len), static_cast<size_t>(kv_dim_)});
        auto V = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(prefill_len), static_cast<size_t>(kv_dim_)});

        float *k_data = K->mutable_data();
        float *v_data = V->mutable_data();

        for (int t = 0; t < prefill_len; ++t)
        {
            for (int d = 0; d < kv_dim_; ++d)
            {
                size_t idx = t * kv_dim_ + d;
                // Pattern: seq_idx * 1000 + token * 100 + d
                float pattern = static_cast<float>(seq_idx * 1000 + t * 100 + d);
                k_data[idx] = pattern;
                v_data[idx] = -pattern;
            }
        }

        ASSERT_TRUE(cache->append_kv(test_layer, seq_idx, K.get(), V.get(), prefill_len))
            << "Failed to pre-populate seq " << seq_idx;
    }

    // Verify pre-population
    EXPECT_EQ(cache->get_cached_tokens(test_layer, 0), prefill_len);
    EXPECT_EQ(cache->get_cached_tokens(test_layer, 1), prefill_len);

    // Create gather output tensors
    const size_t gather_rows = batch_size * max_seq_len; // Max capacity
    auto gathered_K = std::make_unique<FP32Tensor>(
        std::vector<size_t>{gather_rows, static_cast<size_t>(kv_dim_)});
    auto gathered_V = std::make_unique<FP32Tensor>(
        std::vector<size_t>{gather_rows, static_cast<size_t>(kv_dim_)});

    // Execute KVCacheGatherStage
    KVCacheGatherStage::Params params;
    params.kv_cache = cache.get();
    params.layer_idx = test_layer;
    params.batch_size = batch_size;
    params.out_K = gathered_K.get();
    params.out_V = gathered_V.get();

    auto stage = std::make_unique<KVCacheGatherStage>(params);
    ASSERT_TRUE(executeStage(stage.get())) << "KVCacheGatherStage execution failed";

    // Verify max_kv_len
    int max_kv_len = stage->getMaxKVLen();
    EXPECT_EQ(max_kv_len, prefill_len)
        << "max_kv_len should equal prefill_len for uniform sequences";

    // Verify per-sequence kv_lens
    const auto &per_seq_lens = stage->getPerSeqKVLens();
    ASSERT_EQ(per_seq_lens.size(), static_cast<size_t>(batch_size));
    EXPECT_EQ(per_seq_lens[0], prefill_len);
    EXPECT_EQ(per_seq_lens[1], prefill_len);

    // Verify gathered data correctness
    const float *out_k = gathered_K->data();
    const float *out_v = gathered_V->data();

    for (int seq_idx = 0; seq_idx < batch_size; ++seq_idx)
    {
        size_t seq_offset = seq_idx * max_kv_len * kv_dim_;

        for (int t = 0; t < prefill_len; ++t)
        {
            for (int d = 0; d < std::min(kv_dim_, 5); ++d)
            { // Sample first 5 dims
                size_t idx = seq_offset + t * kv_dim_ + d;
                float expected_k = static_cast<float>(seq_idx * 1000 + t * 100 + d);
                float expected_v = -expected_k;

                EXPECT_NEAR(out_k[idx], expected_k, 1e-5f)
                    << "Gathered K mismatch at seq=" << seq_idx << " t=" << t << " d=" << d;
                EXPECT_NEAR(out_v[idx], expected_v, 1e-5f)
                    << "Gathered V mismatch at seq=" << seq_idx << " t=" << t << " d=" << d;
            }
        }
    }

    if (rank_ == 0)
    {
        LOG_INFO("[KVCacheGather] ✓ Basic gather works correctly");
    }
}

/**
 * @test KVCacheGatherStage_VariableLengthSequences
 * @brief Verify gather handles sequences with different cached token counts
 *
 * This simulates a batch where one sequence has more history than another.
 */
TEST_F(GraphBatchedKVCacheTest, KVCacheGatherStage_VariableLengthSequences)
{
    const int batch_size = 3;
    const int max_seq_len = 64;
    const int test_layer = 0;

    // Different lengths per sequence
    const std::vector<int> seq_lengths = {3, 7, 5};

    auto cache = createTestKVCache(max_seq_len, batch_size);

    // Pre-populate cache with variable-length sequences
    for (int seq_idx = 0; seq_idx < batch_size; ++seq_idx)
    {
        int len = seq_lengths[seq_idx];

        auto K = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(len), static_cast<size_t>(kv_dim_)});
        auto V = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(len), static_cast<size_t>(kv_dim_)});

        float *k_data = K->mutable_data();
        float *v_data = V->mutable_data();

        for (int t = 0; t < len; ++t)
        {
            for (int d = 0; d < kv_dim_; ++d)
            {
                size_t idx = t * kv_dim_ + d;
                float pattern = static_cast<float>(seq_idx * 1000 + t * 100 + d);
                k_data[idx] = pattern;
                v_data[idx] = -pattern;
            }
        }

        ASSERT_TRUE(cache->append_kv(test_layer, seq_idx, K.get(), V.get(), len));
    }

    // Verify pre-population
    for (int i = 0; i < batch_size; ++i)
    {
        EXPECT_EQ(cache->get_cached_tokens(test_layer, i), seq_lengths[i]);
    }

    // Create gather output tensors (sized for max_seq_len)
    const size_t gather_rows = batch_size * max_seq_len;
    auto gathered_K = std::make_unique<FP32Tensor>(
        std::vector<size_t>{gather_rows, static_cast<size_t>(kv_dim_)});
    auto gathered_V = std::make_unique<FP32Tensor>(
        std::vector<size_t>{gather_rows, static_cast<size_t>(kv_dim_)});

    // Execute gather
    KVCacheGatherStage::Params params;
    params.kv_cache = cache.get();
    params.layer_idx = test_layer;
    params.batch_size = batch_size;
    params.out_K = gathered_K.get();
    params.out_V = gathered_V.get();

    auto stage = std::make_unique<KVCacheGatherStage>(params);
    ASSERT_TRUE(executeStage(stage.get())) << "KVCacheGatherStage execution failed";

    // Verify max_kv_len is the maximum across all sequences
    int max_kv_len = stage->getMaxKVLen();
    EXPECT_EQ(max_kv_len, 7) << "max_kv_len should be 7 (max of 3, 7, 5)";

    // Verify per-sequence lengths
    const auto &per_seq_lens = stage->getPerSeqKVLens();
    ASSERT_EQ(per_seq_lens.size(), static_cast<size_t>(batch_size));
    EXPECT_EQ(per_seq_lens[0], 3);
    EXPECT_EQ(per_seq_lens[1], 7);
    EXPECT_EQ(per_seq_lens[2], 5);

    // Verify gathered data for each sequence (within their valid range)
    const float *out_k = gathered_K->data();

    for (int seq_idx = 0; seq_idx < batch_size; ++seq_idx)
    {
        int valid_len = seq_lengths[seq_idx];
        size_t seq_offset = seq_idx * max_kv_len * kv_dim_;

        for (int t = 0; t < valid_len; ++t)
        {
            for (int d = 0; d < std::min(kv_dim_, 3); ++d)
            {
                size_t idx = seq_offset + t * kv_dim_ + d;
                float expected_k = static_cast<float>(seq_idx * 1000 + t * 100 + d);
                EXPECT_NEAR(out_k[idx], expected_k, 1e-5f)
                    << "Gathered K mismatch at seq=" << seq_idx << " t=" << t << " d=" << d;
            }
        }
    }

    if (rank_ == 0)
    {
        LOG_INFO("[KVCacheGather] ✓ Variable-length gather works correctly");
    }
}

/**
 * @test KVCacheGatherStage_AfterBatchedAppend
 * @brief Verify end-to-end batched decode: append + gather workflow
 *
 * This tests the full workflow for batched decode.
 */
TEST_F(GraphBatchedKVCacheTest, KVCacheGatherStage_AfterBatchedAppend)
{
    const int batch_size = 2;
    const int max_seq_len = 64;
    const int test_layer = 0;
    const int prefill_len = 4;
    const int decode_tokens = 1;

    auto cache = createTestKVCache(max_seq_len, batch_size);

    // Step 1: Prefill each sequence separately
    for (int seq_idx = 0; seq_idx < batch_size; ++seq_idx)
    {
        auto K = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(prefill_len), static_cast<size_t>(kv_dim_)});
        auto V = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(prefill_len), static_cast<size_t>(kv_dim_)});

        // Initialize with pattern
        std::fill(K->mutable_data(), K->mutable_data() + K->numel(),
                  static_cast<float>(seq_idx * 100));
        std::fill(V->mutable_data(), V->mutable_data() + V->numel(),
                  static_cast<float>(-seq_idx * 100));

        ASSERT_TRUE(cache->append_kv(test_layer, seq_idx, K.get(), V.get(), prefill_len));
    }

    // Step 2: Batched decode append (1 token per sequence)
    auto decode_K = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(batch_size * decode_tokens), static_cast<size_t>(kv_dim_)});
    auto decode_V = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(batch_size * decode_tokens), static_cast<size_t>(kv_dim_)});

    float *dk = decode_K->mutable_data();
    float *dv = decode_V->mutable_data();
    for (int b = 0; b < batch_size; ++b)
    {
        for (int d = 0; d < kv_dim_; ++d)
        {
            dk[b * kv_dim_ + d] = static_cast<float>(999 + b); // Distinct decode pattern
            dv[b * kv_dim_ + d] = static_cast<float>(-999 - b);
        }
    }

    KVCacheAppendStage::Params append_params;
    append_params.K = decode_K.get();
    append_params.V = decode_V.get();
    append_params.kv_cache = cache.get();
    append_params.layer_idx = test_layer;
    append_params.seq_idx = 0;
    append_params.batch_size = batch_size;
    append_params.seq_len = decode_tokens;

    auto append_stage = std::make_unique<KVCacheAppendStage>(append_params);
    ASSERT_TRUE(executeStage(append_stage.get())) << "Batched append failed";

    // Verify cache state after append
    int expected_len = prefill_len + decode_tokens;
    EXPECT_EQ(cache->get_cached_tokens(test_layer, 0), expected_len);
    EXPECT_EQ(cache->get_cached_tokens(test_layer, 1), expected_len);

    // Step 3: Gather for attention
    const size_t gather_rows = batch_size * max_seq_len;
    auto gathered_K = std::make_unique<FP32Tensor>(
        std::vector<size_t>{gather_rows, static_cast<size_t>(kv_dim_)});
    auto gathered_V = std::make_unique<FP32Tensor>(
        std::vector<size_t>{gather_rows, static_cast<size_t>(kv_dim_)});

    KVCacheGatherStage::Params gather_params;
    gather_params.kv_cache = cache.get();
    gather_params.layer_idx = test_layer;
    gather_params.batch_size = batch_size;
    gather_params.out_K = gathered_K.get();
    gather_params.out_V = gathered_V.get();

    auto gather_stage = std::make_unique<KVCacheGatherStage>(gather_params);
    ASSERT_TRUE(executeStage(gather_stage.get())) << "Gather failed";

    // Verify gather results
    int max_kv_len = gather_stage->getMaxKVLen();
    EXPECT_EQ(max_kv_len, expected_len) << "max_kv_len should include decode token";

    // Verify the decode token is at the correct position in gathered data
    const float *out_k = gathered_K->data();
    for (int seq_idx = 0; seq_idx < batch_size; ++seq_idx)
    {
        size_t decode_idx = seq_idx * max_kv_len * kv_dim_ + (prefill_len)*kv_dim_;
        float expected_decode = static_cast<float>(999 + seq_idx);

        // Check first element of decode token row
        EXPECT_NEAR(out_k[decode_idx], expected_decode, 1e-5f)
            << "Decode token not found at correct position for seq " << seq_idx;
    }

    if (rank_ == 0)
    {
        LOG_INFO("[KVCacheGather] ✓ Append + Gather workflow works correctly");
    }
}

// Main entry point for MPI tests
int main(int argc, char **argv)
{
    // Initialize MPI before Google Test
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    ::testing::InitGoogleTest(&argc, argv);

    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}
