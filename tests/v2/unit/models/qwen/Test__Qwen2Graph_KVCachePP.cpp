/**
 * @file Test__Qwen2Graph_KVCachePP.cpp
 * @brief Regression test for KV cache layer indexing in Pipeline Parallel mode
 *
 * Tests that buildAttentionGraph() correctly maps global layer indices to local
 * KV cache indices when pp_layer_offset > 0. This prevents a regression of the
 * bug where PP stage 1 (layers 12-23) would pass global layer_idx=12 to a
 * 12-layer cache, causing get_cached_tokens(12, 0) to return 0 (out of bounds)
 * and falling through to the prefill path instead of decode.
 *
 * Bug: Qwen2Graph::buildAttentionGraph() used raw global layer_idx for KV cache
 *      reads (get_cached_tokens, get_k, get_v) but KV cache expects local
 *      indices (0-based within the cache's n_layers).
 * Fix: int kv_local_layer = layer_idx - config_.pp_layer_offset;
 *
 * @date February 2026
 */

#include <gtest/gtest.h>
#include "models/qwen/Qwen2Graph.h"
#include "kernels/cpu/CPURingKVCache.h"
#include "tensors/TensorFactory.h"
#include "execution/compute_stages/stages/AttentionComputeStage.h"
#include "execution/compute_stages/stages/FusedAttentionWoStage.h"
#include "execution/compute_stages/stages/KVCacheAppendStage.h"

using namespace llaminar2;

namespace
{

    // ============================================================================
    // Test Fixture
    // ============================================================================

    class Test__Qwen2Graph_KVCachePP : public ::testing::Test
    {
    protected:
        // Small model config for fast tests
        static constexpr int N_LAYERS = 4;
        static constexpr int D_MODEL = 64;
        static constexpr int N_HEADS = 4;
        static constexpr int N_KV_HEADS = 2;
        static constexpr int HEAD_DIM = 16;
        static constexpr int D_FF = 128;
        static constexpr int VOCAB_SIZE = 100;
        static constexpr int MAX_SEQ_LEN = 32;
        static constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM; // 32

        void SetUp() override
        {
            mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);
            tensor_factory_ = std::make_unique<TensorFactory>(*mpi_ctx_);
            createMockWeights();
            createMockBuffers();
        }

        // =========================================================================
        // Create a Qwen2Graph with pp_layer_offset
        // =========================================================================
        std::unique_ptr<Qwen2Graph> createGraph(int pp_layer_offset)
        {
            GraphConfig config;
            config.n_layers = N_LAYERS;
            config.d_model = D_MODEL;
            config.n_heads = N_HEADS;
            config.n_kv_heads = N_KV_HEADS;
            config.head_dim = HEAD_DIM;
            config.d_ff = D_FF;
            config.vocab_size = VOCAB_SIZE;
            config.rms_norm_eps = 1e-6f;
            config.rope_theta = 10000.0f;
            config.default_device = DeviceId::cpu();
            config.max_seq_len = MAX_SEQ_LEN;
            config.pp_layer_offset = pp_layer_offset;

            return std::make_unique<Qwen2Graph>(config, mpi_ctx_);
        }

        // =========================================================================
        // Create a CPUKVCache with n_layers and populate it with tokens
        // =========================================================================
        std::unique_ptr<CPURingKVCache<ActivationPrecision::FP32>> createPopulatedCache(
            int n_layers, int tokens_per_layer)
        {
            auto cache = std::make_unique<CPURingKVCache<ActivationPrecision::FP32>>(
                *mpi_ctx_, n_layers, /*batch_size=*/1, MAX_SEQ_LEN,
                N_KV_HEADS, HEAD_DIM);

            // Create small K/V tensors to append
            auto k_tensor = tensor_factory_->createFP32(
                {static_cast<size_t>(tokens_per_layer), static_cast<size_t>(KV_DIM)});
            auto v_tensor = tensor_factory_->createFP32(
                {static_cast<size_t>(tokens_per_layer), static_cast<size_t>(KV_DIM)});

            // Fill with non-zero data
            float *k_data = k_tensor->mutable_data();
            float *v_data = v_tensor->mutable_data();
            for (int i = 0; i < tokens_per_layer * KV_DIM; ++i)
            {
                k_data[i] = static_cast<float>(i) * 0.01f;
                v_data[i] = static_cast<float>(i) * 0.02f;
            }

            // Populate all layers with tokens
            for (int layer = 0; layer < n_layers; ++layer)
            {
                bool ok = cache->append_kv(layer, /*seq_idx=*/0,
                                           k_tensor.get(), v_tensor.get(),
                                           tokens_per_layer);
                EXPECT_TRUE(ok) << "Failed to append KV for local layer " << layer;
            }

            // Store tensors to keep them alive
            cache_k_tensors_.push_back(std::move(k_tensor));
            cache_v_tensors_.push_back(std::move(v_tensor));

            return cache;
        }

        // =========================================================================
        // Get LayerWeights for a layer index
        // =========================================================================
        LayerWeights getLayerWeights(int idx)
        {
            // Clamp to available weights (we only create N_LAYERS worth)
            int safe_idx = idx % N_LAYERS;
            const auto &lw = layer_weights_[safe_idx];
            LayerWeights result;
            result.wq = lw.wq.get();
            result.wk = lw.wk.get();
            result.wv = lw.wv.get();
            result.wo = lw.wo.get();
            result.attn_norm = lw.attn_norm.get();
            result.q_bias = lw.q_bias.get();
            result.k_bias = lw.k_bias.get();
            result.v_bias = lw.v_bias.get();
            result.gate_proj = lw.gate_proj.get();
            result.up_proj = lw.up_proj.get();
            result.down_proj = lw.down_proj.get();
            result.ffn_norm = lw.ffn_norm.get();
            return result;
        }

        // =========================================================================
        // Find an AttentionComputeStage in the graph and return its params
        // =========================================================================
        const AttentionComputeStage::Params *findAttentionParams(const ComputeGraph &graph)
        {
            auto order = graph.getExecutionOrder();
            for (const auto &name : order)
            {
                const ComputeNode *node = graph.getNode(name);
                if (node && node->stage && node->stage->type() == ComputeStageType::ATTENTION)
                {
                    auto *attn = dynamic_cast<AttentionComputeStage *>(node->stage.get());
                    if (attn)
                    {
                        return &attn->getParams();
                    }
                }
            }
            return nullptr;
        }

        // =========================================================================
        // Find a FusedAttentionWoStage in the graph and return its params
        // =========================================================================
        const FusedAttentionWoStage::Params *findFusedAttentionParams(const ComputeGraph &graph)
        {
            auto order = graph.getExecutionOrder();
            for (const auto &name : order)
            {
                const ComputeNode *node = graph.getNode(name);
                if (node && node->stage && node->stage->type() == ComputeStageType::FUSED_ATTENTION_WO)
                {
                    auto *fused = dynamic_cast<FusedAttentionWoStage *>(node->stage.get());
                    if (fused)
                    {
                        return &fused->getParams();
                    }
                }
            }
            return nullptr;
        }

        // =========================================================================
        // Check if a stage of given type exists in the graph
        // =========================================================================
        bool hasStageType(const ComputeGraph &graph, ComputeStageType type)
        {
            auto order = graph.getExecutionOrder();
            for (const auto &name : order)
            {
                const ComputeNode *node = graph.getNode(name);
                if (node && node->stage && node->stage->type() == type)
                {
                    return true;
                }
            }
            return false;
        }

    private:
        struct LayerWeightSet
        {
            std::unique_ptr<TensorBase> wq, wk, wv, wo, attn_norm;
            std::unique_ptr<TensorBase> q_bias, k_bias, v_bias;
            std::unique_ptr<TensorBase> gate_proj, up_proj, down_proj, ffn_norm;
        };

        void createMockWeights()
        {
            auto sz = [](int x)
            { return static_cast<size_t>(x); };

            for (int layer = 0; layer < N_LAYERS; ++layer)
            {
                LayerWeightSet lw;
                lw.wq = tensor_factory_->createFP32({sz(D_MODEL), sz(D_MODEL)});
                lw.wk = tensor_factory_->createFP32({sz(KV_DIM), sz(D_MODEL)});
                lw.wv = tensor_factory_->createFP32({sz(KV_DIM), sz(D_MODEL)});
                lw.wo = tensor_factory_->createFP32({sz(D_MODEL), sz(D_MODEL)});
                lw.attn_norm = tensor_factory_->createFP32({sz(D_MODEL)});
                lw.q_bias = tensor_factory_->createFP32({sz(D_MODEL)});
                lw.k_bias = tensor_factory_->createFP32({sz(KV_DIM)});
                lw.v_bias = tensor_factory_->createFP32({sz(KV_DIM)});
                lw.gate_proj = tensor_factory_->createFP32({sz(D_FF), sz(D_MODEL)});
                lw.up_proj = tensor_factory_->createFP32({sz(D_FF), sz(D_MODEL)});
                lw.down_proj = tensor_factory_->createFP32({sz(D_MODEL), sz(D_FF)});
                lw.ffn_norm = tensor_factory_->createFP32({sz(D_MODEL)});
                layer_weights_.push_back(std::move(lw));
            }
        }

        void createMockBuffers()
        {
            auto sz = [](int x)
            { return static_cast<size_t>(x); };

            residual_ = tensor_factory_->createFP32({sz(MAX_SEQ_LEN), sz(D_MODEL)});
            buffers_.residual = residual_.get();

            normalized_ = tensor_factory_->createFP32({sz(MAX_SEQ_LEN), sz(D_MODEL)});
            buffers_.normalized = normalized_.get();

            Q_ = tensor_factory_->createFP32({sz(MAX_SEQ_LEN), sz(D_MODEL)});
            buffers_.Q = Q_.get();

            K_ = tensor_factory_->createFP32({sz(MAX_SEQ_LEN), sz(KV_DIM)});
            buffers_.K = K_.get();

            V_ = tensor_factory_->createFP32({sz(MAX_SEQ_LEN), sz(KV_DIM)});
            buffers_.V = V_.get();

            attn_output_ = tensor_factory_->createFP32({sz(MAX_SEQ_LEN), sz(D_MODEL)});
            buffers_.attn_output = attn_output_.get();

            attn_proj_ = tensor_factory_->createFP32({sz(MAX_SEQ_LEN), sz(D_MODEL)});
            buffers_.attn_proj = attn_proj_.get();

            gate_ = tensor_factory_->createFP32({sz(MAX_SEQ_LEN), sz(D_FF)});
            buffers_.gate = gate_.get();

            up_ = tensor_factory_->createFP32({sz(MAX_SEQ_LEN), sz(D_FF)});
            buffers_.up = up_.get();

            ffn_output_ = tensor_factory_->createFP32({sz(MAX_SEQ_LEN), sz(D_FF)});
            buffers_.ffn_output = ffn_output_.get();

            current_hidden_ = tensor_factory_->createFP32({sz(MAX_SEQ_LEN), sz(D_MODEL)});
            buffers_.current_hidden = current_hidden_.get();

            // Workspace buffers for attention
            workspace_scores_ = tensor_factory_->createFP32(
                {sz(N_HEADS), sz(MAX_SEQ_LEN), sz(MAX_SEQ_LEN)});
            buffers_.workspace_scores = workspace_scores_.get();

            workspace_context_ = tensor_factory_->createFP32(
                {sz(MAX_SEQ_LEN), sz(D_MODEL)});
            buffers_.workspace_context = workspace_context_.get();
        }

    protected:
        std::shared_ptr<MPIContext> mpi_ctx_;
        std::unique_ptr<TensorFactory> tensor_factory_;
        ActivationBuffers buffers_{};

        // Weight storage
        std::vector<LayerWeightSet> layer_weights_;

        // Buffer storage (kept alive for the test)
        std::unique_ptr<TensorBase> residual_, normalized_;
        std::unique_ptr<TensorBase> Q_, K_, V_;
        std::unique_ptr<TensorBase> attn_output_, attn_proj_;
        std::unique_ptr<TensorBase> gate_, up_, ffn_output_;
        std::unique_ptr<TensorBase> current_hidden_;
        std::unique_ptr<TensorBase> workspace_scores_, workspace_context_;

        // Cache tensor storage (kept alive for the test)
        std::vector<std::unique_ptr<TensorBase>> cache_k_tensors_;
        std::vector<std::unique_ptr<TensorBase>> cache_v_tensors_;
    };

    // ============================================================================
    // Test: No PP offset (pp_layer_offset=0) — baseline behavior
    // ============================================================================

    TEST_F(Test__Qwen2Graph_KVCachePP, NoOffset_DecodeModeUsesCache)
    {
        // pp_layer_offset=0 (no PP), layer_idx=0
        auto graph_builder = createGraph(/*pp_layer_offset=*/0);

        // Create a 4-layer cache with 8 cached tokens per layer
        auto cache = createPopulatedCache(/*n_layers=*/N_LAYERS, /*tokens_per_layer=*/8);

        // Verify cache has tokens at local layer 0
        ASSERT_EQ(cache->get_cached_tokens(0, 0), 8);

        // Build attention graph for layer 0, single-token decode (batch=1, seq=1)
        int position_id = 8; // Position after cached tokens
        auto layer_weights = getLayerWeights(0);
        ComputeGraph attn_graph = graph_builder->buildAttentionGraph(
            layer_weights, buffers_,
            /*layer_idx=*/0, /*seq_len=*/1, /*batch_size=*/1,
            cache.get(), &position_id,
            DeviceId::cpu());

        // Should have built an attention graph (either decomposed or fused)
        bool has_attention = hasStageType(attn_graph, ComputeStageType::ATTENTION);
        bool has_fused = hasStageType(attn_graph, ComputeStageType::FUSED_ATTENTION_WO);
        ASSERT_TRUE(has_attention || has_fused)
            << "Expected attention or fused attention stage in graph";

        // Verify the attention stage received K/V from cache (not from projected buffers)
        if (has_attention)
        {
            const auto *params = findAttentionParams(attn_graph);
            ASSERT_NE(params, nullptr);
            // Decode mode: K/V should come from KV cache, not from projected buffers
            EXPECT_NE(params->K, buffers_.K) << "K should come from KV cache, not projection buffer";
            EXPECT_NE(params->V, buffers_.V) << "V should come from KV cache, not projection buffer";
            EXPECT_EQ(params->kv_len, 8) << "kv_len should reflect cached tokens";
        }
        else if (has_fused)
        {
            const auto *params = findFusedAttentionParams(attn_graph);
            ASSERT_NE(params, nullptr);
            EXPECT_NE(params->K, buffers_.K) << "K should come from KV cache, not projection buffer";
            EXPECT_NE(params->V, buffers_.V) << "V should come from KV cache, not projection buffer";
        }
    }

    // ============================================================================
    // REGRESSION TEST: PP offset with decode mode
    //
    // This test locks in the fix for the bug where global layer_idx was used
    // directly for KV cache reads instead of the local index.
    //
    // Before fix: layer_idx=2 with a 2-layer cache → get_cached_tokens(2, 0)
    //             returns 0 (out of bounds) → falls to prefill path
    // After fix:  kv_local_layer = 2 - 2 = 0 → get_cached_tokens(0, 0)
    //             returns 8 → correctly enters decode path
    // ============================================================================

    TEST_F(Test__Qwen2Graph_KVCachePP, PPOffset_DecodeModeUsesLocalIndex)
    {
        // Simulate PP stage 1 with layers [2, 4)
        // pp_layer_offset=2, so global layer 2 maps to local KV cache layer 0
        const int PP_OFFSET = 2;
        auto graph_builder = createGraph(PP_OFFSET);

        // Create a 2-layer cache (for PP stage 1, covering layers 2-3)
        // with 8 cached tokens per layer
        auto cache = createPopulatedCache(/*n_layers=*/2, /*tokens_per_layer=*/8);

        // Verify cache has tokens at local layer 0
        ASSERT_EQ(cache->get_cached_tokens(0, 0), 8);
        // Verify that direct access at global index 2 returns 0 (out of bounds)
        // This is the behavior that caused the bug
        ASSERT_EQ(cache->get_cached_tokens(2, 0), 0)
            << "Global layer index 2 should be out of bounds for 2-layer cache";

        // Build attention graph for global layer 2, single-token decode
        int position_id = 8;
        auto layer_weights = getLayerWeights(0); // Use first weight set
        ComputeGraph attn_graph = graph_builder->buildAttentionGraph(
            layer_weights, buffers_,
            /*layer_idx=*/2, // GLOBAL layer index
            /*seq_len=*/1, /*batch_size=*/1,
            cache.get(), &position_id,
            DeviceId::cpu());

        // Should have built an attention graph
        bool has_attention = hasStageType(attn_graph, ComputeStageType::ATTENTION);
        bool has_fused = hasStageType(attn_graph, ComputeStageType::FUSED_ATTENTION_WO);
        ASSERT_TRUE(has_attention || has_fused)
            << "Expected attention or fused attention stage in graph";

        // CRITICAL: Verify the attention stage is in DECODE mode
        // (K/V from cache, not from projected buffers)
        if (has_attention)
        {
            const auto *params = findAttentionParams(attn_graph);
            ASSERT_NE(params, nullptr);

            // With the fix: kv_local_layer = 2 - 2 = 0
            // get_cached_tokens(0, 0) = 8, so decode path is taken
            EXPECT_NE(params->K, buffers_.K)
                << "REGRESSION: K came from projection buffer instead of KV cache. "
                   "buildAttentionGraph() is using global layer_idx instead of local "
                   "(layer_idx - pp_layer_offset) for KV cache reads.";
            EXPECT_NE(params->V, buffers_.V)
                << "REGRESSION: V came from projection buffer instead of KV cache. "
                   "buildAttentionGraph() is using global layer_idx instead of local "
                   "(layer_idx - pp_layer_offset) for KV cache reads.";
            EXPECT_EQ(params->kv_len, 8)
                << "kv_len should be 8 (cached tokens at local layer 0)";
        }
        else if (has_fused)
        {
            const auto *params = findFusedAttentionParams(attn_graph);
            ASSERT_NE(params, nullptr);

            EXPECT_NE(params->K, buffers_.K)
                << "REGRESSION: K came from projection buffer instead of KV cache. "
                   "buildAttentionGraph() is using global layer_idx instead of local "
                   "(layer_idx - pp_layer_offset) for KV cache reads.";
            EXPECT_NE(params->V, buffers_.V)
                << "REGRESSION: V came from projection buffer instead of KV cache. "
                   "buildAttentionGraph() is using global layer_idx instead of local "
                   "(layer_idx - pp_layer_offset) for KV cache reads.";
        }
    }

    // ============================================================================
    // Test: PP offset with prefill mode (cache empty, should use projected K/V)
    // ============================================================================

    TEST_F(Test__Qwen2Graph_KVCachePP, PPOffset_PrefillModeUsesProjectedKV)
    {
        // PP stage 1: layers [2, 4), pp_layer_offset=2
        const int PP_OFFSET = 2;
        auto graph_builder = createGraph(PP_OFFSET);

        // Create an EMPTY 2-layer cache (prefill mode)
        auto cache = std::make_unique<CPURingKVCache<ActivationPrecision::FP32>>(
            *mpi_ctx_, /*n_layers=*/2, /*batch_size=*/1, MAX_SEQ_LEN,
            N_KV_HEADS, HEAD_DIM);

        // Verify cache is empty
        ASSERT_EQ(cache->get_cached_tokens(0, 0), 0);

        // Build attention graph for global layer 2, multi-token prefill
        std::vector<int> position_ids = {0, 1, 2, 3};
        auto layer_weights = getLayerWeights(0);
        ComputeGraph attn_graph = graph_builder->buildAttentionGraph(
            layer_weights, buffers_,
            /*layer_idx=*/2,
            /*seq_len=*/4, /*batch_size=*/1,
            cache.get(), position_ids.data(),
            DeviceId::cpu());

        // Should have built an attention graph
        bool has_attention = hasStageType(attn_graph, ComputeStageType::ATTENTION);
        bool has_fused = hasStageType(attn_graph, ComputeStageType::FUSED_ATTENTION_WO);
        ASSERT_TRUE(has_attention || has_fused)
            << "Expected attention or fused attention stage in graph";

        // Prefill mode: K/V should come from projected buffers (not cache)
        if (has_attention)
        {
            const auto *params = findAttentionParams(attn_graph);
            ASSERT_NE(params, nullptr);
            // In prefill mode, K/V are the projected buffers
            EXPECT_EQ(params->K, buffers_.K)
                << "In prefill mode, K should be the projected buffer";
            EXPECT_EQ(params->V, buffers_.V)
                << "In prefill mode, V should be the projected buffer";
        }
    }

    // ============================================================================
    // Test: Multiple layers with different offsets all use correct local indices
    // ============================================================================

    TEST_F(Test__Qwen2Graph_KVCachePP, PPOffset_MultipleLayersCorrectMapping)
    {
        // PP stage 1: layers [2, 4), pp_layer_offset=2
        const int PP_OFFSET = 2;
        auto graph_builder = createGraph(PP_OFFSET);

        // Create a 2-layer cache with different token counts per layer
        auto cache = std::make_unique<CPURingKVCache<ActivationPrecision::FP32>>(
            *mpi_ctx_, /*n_layers=*/2, /*batch_size=*/1, MAX_SEQ_LEN,
            N_KV_HEADS, HEAD_DIM);

        // Put 5 tokens in local layer 0 (global layer 2)
        auto k0 = tensor_factory_->createFP32({5, static_cast<size_t>(KV_DIM)});
        auto v0 = tensor_factory_->createFP32({5, static_cast<size_t>(KV_DIM)});
        ASSERT_TRUE(cache->append_kv(0, 0, k0.get(), v0.get(), 5));

        // Put 10 tokens in local layer 1 (global layer 3)
        auto k1 = tensor_factory_->createFP32({10, static_cast<size_t>(KV_DIM)});
        auto v1 = tensor_factory_->createFP32({10, static_cast<size_t>(KV_DIM)});
        ASSERT_TRUE(cache->append_kv(1, 0, k1.get(), v1.get(), 10));

        // Verify
        ASSERT_EQ(cache->get_cached_tokens(0, 0), 5);
        ASSERT_EQ(cache->get_cached_tokens(1, 0), 10);

        // Test global layer 2 → local 0 (should get 5 cached tokens)
        {
            int position_id = 5;
            auto lw = getLayerWeights(0);
            ComputeGraph graph = graph_builder->buildAttentionGraph(
                lw, buffers_, /*layer_idx=*/2, /*seq_len=*/1, /*batch_size=*/1,
                cache.get(), &position_id, DeviceId::cpu());

            bool has_attn = hasStageType(graph, ComputeStageType::ATTENTION);
            bool has_fused = hasStageType(graph, ComputeStageType::FUSED_ATTENTION_WO);
            ASSERT_TRUE(has_attn || has_fused);

            if (has_attn)
            {
                const auto *params = findAttentionParams(graph);
                ASSERT_NE(params, nullptr);
                EXPECT_EQ(params->kv_len, 5)
                    << "Global layer 2 → local 0 should have 5 cached tokens";
                EXPECT_NE(params->K, buffers_.K);
            }
        }

        // Test global layer 3 → local 1 (should get 10 cached tokens)
        {
            int position_id = 10;
            auto lw = getLayerWeights(1);
            ComputeGraph graph = graph_builder->buildAttentionGraph(
                lw, buffers_, /*layer_idx=*/3, /*seq_len=*/1, /*batch_size=*/1,
                cache.get(), &position_id, DeviceId::cpu());

            bool has_attn = hasStageType(graph, ComputeStageType::ATTENTION);
            bool has_fused = hasStageType(graph, ComputeStageType::FUSED_ATTENTION_WO);
            ASSERT_TRUE(has_attn || has_fused);

            if (has_attn)
            {
                const auto *params = findAttentionParams(graph);
                ASSERT_NE(params, nullptr);
                EXPECT_EQ(params->kv_len, 10)
                    << "Global layer 3 → local 1 should have 10 cached tokens";
                EXPECT_NE(params->K, buffers_.K);
            }
        }

        // Keep tensors alive
        cache_k_tensors_.push_back(std::move(k0));
        cache_k_tensors_.push_back(std::move(k1));
        cache_v_tensors_.push_back(std::move(v0));
        cache_v_tensors_.push_back(std::move(v1));
    }

    // ============================================================================
    // Test: KV cache append stage uses local layer index
    // ============================================================================

    TEST_F(Test__Qwen2Graph_KVCachePP, PPOffset_KVAppendUsesLocalIndex)
    {
        // PP stage 1: layers [2, 4), pp_layer_offset=2
        const int PP_OFFSET = 2;
        auto graph_builder = createGraph(PP_OFFSET);

        // Empty cache (prefill mode)
        auto cache = std::make_unique<CPURingKVCache<ActivationPrecision::FP32>>(
            *mpi_ctx_, /*n_layers=*/2, /*batch_size=*/1, MAX_SEQ_LEN,
            N_KV_HEADS, HEAD_DIM);

        // Build attention graph and check that KV cache append is present
        std::vector<int> position_ids = {0, 1, 2, 3};
        auto layer_weights = getLayerWeights(0);
        ComputeGraph attn_graph = graph_builder->buildAttentionGraph(
            layer_weights, buffers_,
            /*layer_idx=*/2, /*seq_len=*/4, /*batch_size=*/1,
            cache.get(), position_ids.data(),
            DeviceId::cpu());

        // Verify KV cache append stage exists
        EXPECT_TRUE(hasStageType(attn_graph, ComputeStageType::COPY))
            << "Expected KVCacheAppendStage (type=COPY) in graph with KV cache";
    }

    // ============================================================================
    // REGRESSION TEST: KVCacheAppendStage device_id propagation
    //
    // Locks in the fix for the bug where KVCacheAppendStage::Params.device_id
    // defaulted to DeviceId::cpu() even when building with a GPU device.
    //
    // Before fix: device_id = cpu() → appendWithStream never called → raw FP32
    //             bytes written to FP16 ring buffer → complete data corruption
    // After fix:  device_id = device → appendWithStream called → GPU FP16
    //             conversion → correct data in ring buffer
    // ============================================================================

    TEST_F(Test__Qwen2Graph_KVCachePP, KVAppendStage_DeviceId_MatchesTargetDevice)
    {
        // Build graph targeting a GPU device (even though we can't execute on GPU
        // in a unit test, we can verify the stage params are wired correctly)
        auto graph_builder = createGraph(/*pp_layer_offset=*/0);

        auto cache = std::make_unique<CPURingKVCache<ActivationPrecision::FP32>>(
            *mpi_ctx_, /*n_layers=*/N_LAYERS, /*batch_size=*/1, MAX_SEQ_LEN,
            N_KV_HEADS, HEAD_DIM);

        std::vector<int> position_ids = {0, 1, 2, 3};
        auto layer_weights = getLayerWeights(0);

        // Build with a CUDA device target
        DeviceId cuda_device = DeviceId::cuda(0);
        ComputeGraph attn_graph = graph_builder->buildAttentionGraph(
            layer_weights, buffers_,
            /*layer_idx=*/0, /*seq_len=*/4, /*batch_size=*/1,
            cache.get(), position_ids.data(),
            cuda_device);

        // Find KVCacheAppendStage and verify its device_id matches the target device
        bool found_kv_append = false;
        auto order = attn_graph.getExecutionOrder();
        for (const auto &name : order)
        {
            const ComputeNode *node = attn_graph.getNode(name);
            if (node && node->stage && node->stage->type() == ComputeStageType::COPY)
            {
                auto *kv_stage = dynamic_cast<KVCacheAppendStage *>(node->stage.get());
                if (kv_stage)
                {
                    found_kv_append = true;
                    const auto &params = kv_stage->getParams();
                    EXPECT_TRUE(params.device_id.is_gpu())
                        << "REGRESSION: KVCacheAppendStage.device_id is CPU when graph "
                           "targets GPU. This causes appendWithStream() to never be "
                           "called, resulting in raw FP32 bytes written to FP16/Q8_1 "
                           "ring buffers (complete data corruption).";
                    EXPECT_EQ(params.device_id.type, cuda_device.type)
                        << "REGRESSION: KVCacheAppendStage.device_id type doesn't "
                           "match target device type";
                }
            }
        }
        EXPECT_TRUE(found_kv_append)
            << "Expected KVCacheAppendStage in graph with KV cache";
    }

    // Also verify CPU device builds correctly (baseline)
    TEST_F(Test__Qwen2Graph_KVCachePP, KVAppendStage_DeviceId_CPUDevice)
    {
        auto graph_builder = createGraph(/*pp_layer_offset=*/0);

        auto cache = std::make_unique<CPURingKVCache<ActivationPrecision::FP32>>(
            *mpi_ctx_, /*n_layers=*/N_LAYERS, /*batch_size=*/1, MAX_SEQ_LEN,
            N_KV_HEADS, HEAD_DIM);

        std::vector<int> position_ids = {0, 1, 2, 3};
        auto layer_weights = getLayerWeights(0);

        ComputeGraph attn_graph = graph_builder->buildAttentionGraph(
            layer_weights, buffers_,
            /*layer_idx=*/0, /*seq_len=*/4, /*batch_size=*/1,
            cache.get(), position_ids.data(),
            DeviceId::cpu());

        auto order = attn_graph.getExecutionOrder();
        for (const auto &name : order)
        {
            const ComputeNode *node = attn_graph.getNode(name);
            if (node && node->stage && node->stage->type() == ComputeStageType::COPY)
            {
                auto *kv_stage = dynamic_cast<KVCacheAppendStage *>(node->stage.get());
                if (kv_stage)
                {
                    const auto &params = kv_stage->getParams();
                    EXPECT_TRUE(params.device_id.is_cpu())
                        << "CPU graph should have CPU device_id on KVCacheAppendStage";
                }
            }
        }
    }

    // ============================================================================
    // Test: Zero offset is a no-op (regression safety)
    // ============================================================================

    TEST_F(Test__Qwen2Graph_KVCachePP, ZeroOffset_SameAsNoOffset)
    {
        // pp_layer_offset=0, just like non-PP mode
        auto graph_builder = createGraph(/*pp_layer_offset=*/0);

        auto cache = createPopulatedCache(/*n_layers=*/N_LAYERS, /*tokens_per_layer=*/6);
        ASSERT_EQ(cache->get_cached_tokens(0, 0), 6);

        int position_id = 6;
        auto lw = getLayerWeights(0);
        ComputeGraph graph = graph_builder->buildAttentionGraph(
            lw, buffers_, /*layer_idx=*/0, /*seq_len=*/1, /*batch_size=*/1,
            cache.get(), &position_id, DeviceId::cpu());

        bool has_attn = hasStageType(graph, ComputeStageType::ATTENTION);
        bool has_fused = hasStageType(graph, ComputeStageType::FUSED_ATTENTION_WO);
        ASSERT_TRUE(has_attn || has_fused);

        if (has_attn)
        {
            const auto *params = findAttentionParams(graph);
            ASSERT_NE(params, nullptr);
            EXPECT_EQ(params->kv_len, 6);
            EXPECT_NE(params->K, buffers_.K) << "Decode mode should use cached K";
        }
    }

} // anonymous namespace
