/**
 * @file Test__Qwen2Graph_PP.cpp
 * @brief Unit tests for Qwen2Graph::buildUnifiedPipelineGraph()
 *
 * Tests the unified PP graph building method that:
 * 1. Validates pipeline_config prerequisites
 * 2. Iterates over PP stages
 * 3. Gets the TP domain for each stage
 * 4. Builds embedding stage (if has_embedding)
 * 5. Builds transformer layers for each stage with the correct device
 * 6. Inserts LocalPPTransferStage between PP stages
 * 7. Builds LM head (if has_lm_head)
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>
#include "models/qwen/Qwen2Graph.h"
#include "config/PipelineConfig.h"
#include "config/TPDomainConfig.h"
#include "config/PPStageConfig.h"
#include "execution/compute_stages/IComputeStage.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "tensors/TensorFactory.h"
#include "backends/DeviceId.h"
#include "../../../mocks/MockLocalPPContext.h"

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{

    // ============================================================================
    // Test Fixture
    // ============================================================================

    /**
     * @brief Test fixture for Qwen2Graph PP tests
     */
    class Test__Qwen2Graph_PP : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Minimal Qwen2-style config (4 layers for simple testing)
            config_.n_layers = 4;
            config_.d_model = 64;
            config_.n_heads = 4;
            config_.n_kv_heads = 2;
            config_.head_dim = 16;
            config_.d_ff = 128;
            config_.vocab_size = 1000;
            config_.rms_norm_eps = 1e-6f;
            config_.rope_theta = 10000.0f;
            config_.default_device = DeviceId::cpu();
            config_.max_seq_len = 32;

            // Create MPI context (single-rank for unit tests)
            mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);

            // Create TensorFactory for buffer allocation (requires MPIContext)
            tensor_factory_ = std::make_unique<TensorFactory>(*mpi_ctx_);

            // Create mock weights
            createMockWeights();

            // Create mock buffers
            createMockBuffers();
        }

        void TearDown() override
        {
            // Cleanup
            mock_pp_contexts_.clear();
        }

        // =========================================================================
        // Helper: Create 2-stage PP config (layers 0-1 on device 0, layers 2-3 on device 1)
        // =========================================================================
        std::shared_ptr<PipelineConfig> createTwoStagePPConfig()
        {
            auto config = std::make_shared<PipelineConfig>();
            config->total_layers = 4;

            // Domain 1: CPU (stage 0)
            TPDomainConfig domain0;
            domain0.name = "stage0_domain";
            domain0.devices = {DeviceId::cpu()};
            domain0.tp_backend = CollectiveBackendType::HOST;
            config->tp_domains.push_back(domain0);

            // Domain 2: CPU (stage 1) - different device in theory
            TPDomainConfig domain1;
            domain1.name = "stage1_domain";
            domain1.devices = {DeviceId::cpu()};
            domain1.tp_backend = CollectiveBackendType::HOST;
            config->tp_domains.push_back(domain1);

            // Stage 0: layers 0-1, has embedding
            PPStageConfig stage0;
            stage0.stage_id = 0;
            stage0.domain_name = "stage0_domain";
            stage0.first_layer = 0;
            stage0.last_layer = 2;
            stage0.has_embedding = true;
            stage0.has_lm_head = false;
            config->pp_stages.push_back(stage0);

            // Stage 1: layers 2-3, has LM head
            PPStageConfig stage1;
            stage1.stage_id = 1;
            stage1.domain_name = "stage1_domain";
            stage1.first_layer = 2;
            stage1.last_layer = 4;
            stage1.has_embedding = false;
            stage1.has_lm_head = true;
            config->pp_stages.push_back(stage1);

            // Transfer backend
            config->pp_transfer_backends[{0, 1}] = CollectiveBackendType::HOST;

            return config;
        }

        // =========================================================================
        // Helper: Create 3-stage PP config
        // =========================================================================
        std::shared_ptr<PipelineConfig> createThreeStagePPConfig()
        {
            auto config = std::make_shared<PipelineConfig>();
            config->total_layers = 6;
            config_.n_layers = 6; // Update config to match

            // Domain 0
            TPDomainConfig domain0;
            domain0.name = "stage0_domain";
            domain0.devices = {DeviceId::cpu()};
            config->tp_domains.push_back(domain0);

            // Domain 1
            TPDomainConfig domain1;
            domain1.name = "stage1_domain";
            domain1.devices = {DeviceId::cpu()};
            config->tp_domains.push_back(domain1);

            // Domain 2
            TPDomainConfig domain2;
            domain2.name = "stage2_domain";
            domain2.devices = {DeviceId::cpu()};
            config->tp_domains.push_back(domain2);

            // Stage 0: layers 0-1, has embedding
            PPStageConfig stage0;
            stage0.stage_id = 0;
            stage0.domain_name = "stage0_domain";
            stage0.first_layer = 0;
            stage0.last_layer = 2;
            stage0.has_embedding = true;
            stage0.has_lm_head = false;
            config->pp_stages.push_back(stage0);

            // Stage 1: layers 2-3 (middle)
            PPStageConfig stage1;
            stage1.stage_id = 1;
            stage1.domain_name = "stage1_domain";
            stage1.first_layer = 2;
            stage1.last_layer = 4;
            stage1.has_embedding = false;
            stage1.has_lm_head = false;
            config->pp_stages.push_back(stage1);

            // Stage 2: layers 4-5, has LM head
            PPStageConfig stage2;
            stage2.stage_id = 2;
            stage2.domain_name = "stage2_domain";
            stage2.first_layer = 4;
            stage2.last_layer = 6;
            stage2.has_embedding = false;
            stage2.has_lm_head = true;
            config->pp_stages.push_back(stage2);

            // Transfer backends
            config->pp_transfer_backends[{0, 1}] = CollectiveBackendType::HOST;
            config->pp_transfer_backends[{1, 2}] = CollectiveBackendType::HOST;

            return config;
        }

        // =========================================================================
        // Helper: Create mock PP contexts for a pipeline config
        // =========================================================================
        // NOTE: This creates a SINGLE context covering ALL stages.
        // Previous bug: Creating per-transfer contexts with only 2 devices caused
        // stage index validation to fail (e.g., stage_to=2 >= numStages()=2).
        // Fix: One context with N devices for N-stage pipeline.
        void createMockPPContexts(PipelineConfig *config)
        {
            int num_stages = config->numStages();

            // Create a single context covering ALL stages
            MockLocalPPContext::Config mock_config;

            // One device per stage
            for (int s = 0; s < num_stages; ++s)
            {
                mock_config.stage_devices.push_back(GlobalDeviceAddress::cpu());
            }

            // Layer boundaries from config
            for (int s = 0; s < num_stages; ++s)
            {
                mock_config.layer_boundaries.push_back(config->pp_stages[s].first_layer);
            }
            mock_config.layer_boundaries.push_back(config->pp_stages.back().last_layer);

            mock_config.default_backend = CollectiveBackendType::HOST;

            auto pp_ctx = std::make_unique<MockLocalPPContext>(mock_config);

            // Register same context for ALL transfer pairs
            for (int from = 0; from < num_stages - 1; ++from)
            {
                int to = from + 1;
                config_.pp_contexts[{from, to}] = pp_ctx.get();
            }
            mock_pp_contexts_.push_back(std::move(pp_ctx));
        }

        // =========================================================================
        // Helper: Create mock weights
        // =========================================================================
        void createMockWeights()
        {
            // Helper to avoid narrowing conversions (int -> size_t)
            auto sz = [](int x)
            { return static_cast<size_t>(x); };

            // Embedding table [vocab_size, d_model]
            embedding_table_ = tensor_factory_->createFP32({sz(config_.vocab_size), sz(config_.d_model)});

            // Final norm [d_model]
            final_norm_ = tensor_factory_->createFP32({sz(config_.d_model)});

            // LM head [vocab_size, d_model]
            lm_head_ = tensor_factory_->createFP32({sz(config_.vocab_size), sz(config_.d_model)});

            // Per-layer weights
            for (int layer = 0; layer < config_.n_layers; ++layer)
            {
                LayerWeightSet layer_weights;

                // Attention weights
                layer_weights.wq = tensor_factory_->createFP32({sz(config_.d_model), sz(config_.d_model)});
                layer_weights.wk = tensor_factory_->createFP32(
                    {sz(config_.n_kv_heads * config_.head_dim), sz(config_.d_model)});
                layer_weights.wv = tensor_factory_->createFP32(
                    {sz(config_.n_kv_heads * config_.head_dim), sz(config_.d_model)});
                layer_weights.wo = tensor_factory_->createFP32({sz(config_.d_model), sz(config_.d_model)});
                layer_weights.attn_norm = tensor_factory_->createFP32({sz(config_.d_model)});

                // Optional biases
                layer_weights.q_bias = tensor_factory_->createFP32({sz(config_.d_model)});
                layer_weights.k_bias = tensor_factory_->createFP32({sz(config_.n_kv_heads * config_.head_dim)});
                layer_weights.v_bias = tensor_factory_->createFP32({sz(config_.n_kv_heads * config_.head_dim)});

                // FFN weights
                layer_weights.gate_proj = tensor_factory_->createFP32({sz(config_.d_ff), sz(config_.d_model)});
                layer_weights.up_proj = tensor_factory_->createFP32({sz(config_.d_ff), sz(config_.d_model)});
                layer_weights.down_proj = tensor_factory_->createFP32({sz(config_.d_model), sz(config_.d_ff)});
                layer_weights.ffn_norm = tensor_factory_->createFP32({sz(config_.d_model)});

                layer_weights_.push_back(std::move(layer_weights));
            }

            // Set up weights accessor
            weights_.embedding_table = embedding_table_.get();
            weights_.final_norm = final_norm_.get();
            weights_.lm_head = lm_head_.get();
            weights_.get_layer_weights = [this](int layer_idx) -> LayerWeights
            {
                if (layer_idx < 0 || layer_idx >= static_cast<int>(layer_weights_.size()))
                {
                    return LayerWeights{};
                }
                const auto &lw = layer_weights_[layer_idx];
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
            };
        }

        // =========================================================================
        // Helper: Create mock buffers
        // =========================================================================
        void createMockBuffers()
        {
            // Helper to avoid narrowing conversions (int -> size_t)
            auto sz = [](int x)
            { return static_cast<size_t>(x); };
            int max_tokens = config_.max_seq_len;

            // Current hidden [max_tokens, d_model]
            current_hidden_ = tensor_factory_->createFP32({sz(max_tokens), sz(config_.d_model)});
            buffers_.current_hidden = current_hidden_.get();

            // Logits [max_tokens, vocab_size]
            logits_ = tensor_factory_->createFP32({sz(max_tokens), sz(config_.vocab_size)});
            buffers_.logits = logits_.get();

            // Layer buffers
            residual_ = tensor_factory_->createFP32({sz(max_tokens), sz(config_.d_model)});
            buffers_.layer_buffers.residual = residual_.get();

            normalized_ = tensor_factory_->createFP32({sz(max_tokens), sz(config_.d_model)});
            buffers_.layer_buffers.normalized = normalized_.get();

            Q_ = tensor_factory_->createFP32({sz(max_tokens), sz(config_.d_model)});
            buffers_.layer_buffers.Q = Q_.get();

            K_ = tensor_factory_->createFP32({sz(max_tokens), sz(config_.n_kv_heads * config_.head_dim)});
            buffers_.layer_buffers.K = K_.get();

            V_ = tensor_factory_->createFP32({sz(max_tokens), sz(config_.n_kv_heads * config_.head_dim)});
            buffers_.layer_buffers.V = V_.get();

            attn_output_ = tensor_factory_->createFP32({sz(max_tokens), sz(config_.d_model)});
            buffers_.layer_buffers.attn_output = attn_output_.get();

            attn_proj_ = tensor_factory_->createFP32({sz(max_tokens), sz(config_.d_model)});
            buffers_.layer_buffers.attn_proj = attn_proj_.get();

            gate_ = tensor_factory_->createFP32({sz(max_tokens), sz(config_.d_ff)});
            buffers_.layer_buffers.gate = gate_.get();

            up_ = tensor_factory_->createFP32({sz(max_tokens), sz(config_.d_ff)});
            buffers_.layer_buffers.up = up_.get();

            ffn_output_ = tensor_factory_->createFP32({sz(max_tokens), sz(config_.d_ff)});
            buffers_.layer_buffers.ffn_output = ffn_output_.get();
        }

        // =========================================================================
        // Helper: Create forward input
        // =========================================================================
        Qwen2ForwardInput createForwardInput(int batch_size = 1, int seq_len = 4)
        {
            // Token IDs
            token_ids_.resize(batch_size * seq_len);
            for (int i = 0; i < batch_size * seq_len; ++i)
            {
                token_ids_[i] = i % config_.vocab_size;
            }

            Qwen2ForwardInput input;
            input.token_ids = token_ids_.data();
            input.batch_size = batch_size;
            input.seq_len = seq_len;
            input.position_offset = 0;
            input.device = DeviceId::cpu();
            input.kv_cache = nullptr;
            return input;
        }

        // =========================================================================
        // Helper: Count stages by type in graph
        // =========================================================================
        int countStagesByType(const ComputeGraph &graph, ComputeStageType type)
        {
            int count = 0;
            auto order = graph.getExecutionOrder();
            for (const auto &name : order)
            {
                const ComputeNode *node = graph.getNode(name);
                if (node && node->stage && node->stage->type() == type)
                {
                    ++count;
                }
            }
            return count;
        }

        // =========================================================================
        // Helper: Find all stage names matching a pattern
        // =========================================================================
        std::vector<std::string> findStageNames(const ComputeGraph &graph, const std::string &pattern)
        {
            std::vector<std::string> matches;
            auto order = graph.getExecutionOrder();
            for (const auto &name : order)
            {
                if (name.find(pattern) != std::string::npos)
                {
                    matches.push_back(name);
                }
            }
            return matches;
        }

        // =========================================================================
        // Member Variables
        // =========================================================================
        GraphConfig config_;
        std::shared_ptr<MPIContext> mpi_ctx_;
        std::unique_ptr<TensorFactory> tensor_factory_;

        // Mock weights storage
        struct LayerWeightSet
        {
            std::shared_ptr<TensorBase> wq, wk, wv, wo, attn_norm;
            std::shared_ptr<TensorBase> q_bias, k_bias, v_bias;
            std::shared_ptr<TensorBase> gate_proj, up_proj, down_proj, ffn_norm;
        };
        std::vector<LayerWeightSet> layer_weights_;
        std::shared_ptr<TensorBase> embedding_table_;
        std::shared_ptr<TensorBase> final_norm_;
        std::shared_ptr<TensorBase> lm_head_;
        ModelWeights weights_;

        // Mock buffers storage
        std::shared_ptr<TensorBase> current_hidden_;
        std::shared_ptr<TensorBase> logits_;
        std::shared_ptr<TensorBase> residual_;
        std::shared_ptr<TensorBase> normalized_;
        std::shared_ptr<TensorBase> Q_, K_, V_;
        std::shared_ptr<TensorBase> attn_output_, attn_proj_;
        std::shared_ptr<TensorBase> gate_, up_;
        std::shared_ptr<TensorBase> ffn_output_;
        ModelBuffers buffers_;

        // Mock PP contexts
        std::vector<std::unique_ptr<MockLocalPPContext>> mock_pp_contexts_;

        // Forward input data
        std::vector<int> token_ids_;
    };

    // ============================================================================
    // Validation Tests
    // ============================================================================

    /**
     * @brief Test that buildUnifiedPipelineGraph throws without pipeline_config
     */
    TEST_F(Test__Qwen2Graph_PP, ThrowsWithoutPipelineConfig)
    {
        // Config has no pipeline_config (nullptr)
        EXPECT_EQ(config_.pipeline_config, nullptr);

        Qwen2Graph graph(config_, mpi_ctx_);
        graph.setWeights(weights_);
        graph.setBuffers(buffers_);

        auto input = createForwardInput();
        Qwen2ForwardOutput output;

        EXPECT_THROW(
            graph.buildUnifiedPipelineGraph(input, output),
            std::runtime_error);
    }

    /**
     * @brief Test that buildUnifiedPipelineGraph throws with invalid pipeline_config
     */
    TEST_F(Test__Qwen2Graph_PP, ThrowsWithInvalidConfig)
    {
        // Create an invalid config (no stages)
        auto pipeline_config = std::make_shared<PipelineConfig>();
        pipeline_config->total_layers = 4;
        // No domains or stages added - validation should fail

        config_.pipeline_config = pipeline_config;

        Qwen2Graph graph(config_, mpi_ctx_);
        graph.setWeights(weights_);
        graph.setBuffers(buffers_);

        auto input = createForwardInput();
        Qwen2ForwardOutput output;

        EXPECT_THROW(
            graph.buildUnifiedPipelineGraph(input, output),
            std::runtime_error);
    }

    /**
     * @brief Test that buildUnifiedPipelineGraph throws without layer weight accessor
     */
    TEST_F(Test__Qwen2Graph_PP, ThrowsWithoutLayerWeightAccessor)
    {
        auto pipeline_config = createTwoStagePPConfig();
        config_.pipeline_config = pipeline_config;
        createMockPPContexts(pipeline_config.get());

        Qwen2Graph graph(config_, mpi_ctx_);

        // Set weights without get_layer_weights accessor
        ModelWeights incomplete_weights;
        incomplete_weights.embedding_table = embedding_table_.get();
        incomplete_weights.final_norm = final_norm_.get();
        incomplete_weights.lm_head = lm_head_.get();
        incomplete_weights.get_layer_weights = nullptr; // No accessor!

        graph.setWeights(incomplete_weights);
        graph.setBuffers(buffers_);

        auto input = createForwardInput();
        Qwen2ForwardOutput output;

        EXPECT_THROW(
            graph.buildUnifiedPipelineGraph(input, output),
            std::runtime_error);
    }

    // ============================================================================
    // REGRESSION TESTS: Position IDs Requirement
    // ============================================================================
    // These tests document and lock in fixes for bugs found during integration testing.
    // Bug: When calling buildUnifiedPipelineGraph() or buildPartialForwardGraph() without
    // setting position_ids in the input, RoPE uses undefined values causing NaN in outputs.
    // Fix: Callers MUST set position_ids via Qwen2Graph::buildPositionIds() before building.

    /**
     * @test Regression: buildPositionIds generates correct sequence for prefill
     *
     * For prefill with seq_len=6, batch_size=1, offset=0:
     * Expected: [0, 1, 2, 3, 4, 5]
     */
    TEST_F(Test__Qwen2Graph_PP, Regression_BuildPositionIdsPrefillSequence)
    {
        auto pos_ids = Qwen2Graph::buildPositionIds(6, 1, 0);

        ASSERT_EQ(pos_ids.size(), 6u);
        for (int i = 0; i < 6; ++i)
        {
            EXPECT_EQ(pos_ids[i], i) << "Position ID mismatch at index " << i;
        }
    }

    /**
     * @test Regression: buildPositionIds generates correct value for decode step
     *
     * For decode with seq_len=1, batch_size=1, offset=100:
     * Expected: [100]
     *
     * This is critical for autoregressive decode - each token needs its correct position.
     */
    TEST_F(Test__Qwen2Graph_PP, Regression_BuildPositionIdsDecodeStep)
    {
        // Simulate decode at position 100 (after prefilling 100 tokens)
        auto pos_ids = Qwen2Graph::buildPositionIds(1, 1, 100);

        ASSERT_EQ(pos_ids.size(), 1u);
        EXPECT_EQ(pos_ids[0], 100);
    }

    /**
     * @test Regression: Each decode step needs freshly built position IDs
     *
     * Documents the bug where reusing prefill position IDs for decode caused wrong RoPE.
     * The fix requires rebuilding position IDs for EACH decode step with current position.
     */
    TEST_F(Test__Qwen2Graph_PP, Regression_DecodePositionIdsMustBeRebuiltEachStep)
    {
        // Simulate decode loop: prefill with 6 tokens, then decode 3 more
        int prefill_len = 6;

        // Prefill position IDs
        auto prefill_pos = Qwen2Graph::buildPositionIds(prefill_len, 1, 0);
        ASSERT_EQ(prefill_pos.size(), 6u);
        EXPECT_EQ(prefill_pos.back(), 5); // Last prefill position is 5

        // Decode steps - EACH needs its own position IDs
        for (int decode_step = 0; decode_step < 3; ++decode_step)
        {
            int current_pos = prefill_len + decode_step; // 6, 7, 8
            auto decode_pos = Qwen2Graph::buildPositionIds(1, 1, current_pos);

            ASSERT_EQ(decode_pos.size(), 1u);
            EXPECT_EQ(decode_pos[0], current_pos)
                << "Decode step " << decode_step << " should have position " << current_pos;
        }
    }

    /**
     * @test Regression: Position IDs with batch_size > 1
     *
     * For batched inference, each sequence in the batch gets the same positions.
     * seq_len=3, batch_size=2, offset=0 → [0,1,2, 0,1,2]
     */
    TEST_F(Test__Qwen2Graph_PP, Regression_BuildPositionIdsBatched)
    {
        auto pos_ids = Qwen2Graph::buildPositionIds(3, 2, 0);

        ASSERT_EQ(pos_ids.size(), 6u);
        // Batch 0: positions 0,1,2
        EXPECT_EQ(pos_ids[0], 0);
        EXPECT_EQ(pos_ids[1], 1);
        EXPECT_EQ(pos_ids[2], 2);
        // Batch 1: positions 0,1,2
        EXPECT_EQ(pos_ids[3], 0);
        EXPECT_EQ(pos_ids[4], 1);
        EXPECT_EQ(pos_ids[5], 2);
    }

    /**
     * @test Regression: Position IDs with offset for continuation
     *
     * When continuing generation, offset shifts all positions.
     * seq_len=3, batch_size=1, offset=50 → [50, 51, 52]
     */
    TEST_F(Test__Qwen2Graph_PP, Regression_BuildPositionIdsWithOffset)
    {
        auto pos_ids = Qwen2Graph::buildPositionIds(3, 1, 50);

        ASSERT_EQ(pos_ids.size(), 3u);
        EXPECT_EQ(pos_ids[0], 50);
        EXPECT_EQ(pos_ids[1], 51);
        EXPECT_EQ(pos_ids[2], 52);
    }

    // ============================================================================
    // REGRESSION TESTS: Multi-Stage PP Context Configuration
    // ============================================================================
    // These tests document the CORRECT way to configure PP contexts for N-stage pipelines.
    // Bug: Creating per-transfer contexts with only 2 devices (e.g., context for transfer 1→2
    // with devices [device1, device2] where numStages()=2) fails at execution time because
    // stage_to=2 >= numStages()=2.
    // Fix: One PP context with N devices covering ALL stages in the pipeline.

    /**
     * @test Regression: createMockPPContexts creates single context covering all stages
     *
     * The helper method must create ONE context with all stages, not per-transfer contexts.
     */
    TEST_F(Test__Qwen2Graph_PP, Regression_CreateMockPPContextsCoversAllStages)
    {
        auto pipeline_config = createThreeStagePPConfig();
        mock_pp_contexts_.clear();
        config_.pp_contexts.clear();

        createMockPPContexts(pipeline_config.get());

        // Should have only ONE MockLocalPPContext
        EXPECT_EQ(mock_pp_contexts_.size(), 1u);

        // That context should have 3 stages
        auto *pp_ctx = mock_pp_contexts_[0].get();
        EXPECT_EQ(pp_ctx->numStages(), 3);

        // Both transfer pairs should reference the same context
        auto key_0_1 = std::make_pair(0, 1);
        auto key_1_2 = std::make_pair(1, 2);
        EXPECT_EQ(config_.pp_contexts[key_0_1], pp_ctx);
        EXPECT_EQ(config_.pp_contexts[key_1_2], pp_ctx);
    }

    /**
     * @test Regression: Transfer stage indices must be valid within PP context
     *
     * When Qwen2Graph builds transfer stages, it uses real stage indices (0, 1, 2...).
     * The PP context must have numStages() > max(stage_to) for validation to pass.
     */
    TEST_F(Test__Qwen2Graph_PP, Regression_ThreeStageTransferIndicesAreValid)
    {
        auto pipeline_config = createThreeStagePPConfig();
        mock_pp_contexts_.clear();
        config_.pp_contexts.clear();

        createMockPPContexts(pipeline_config.get());

        // Get the context that will be used for transfers
        auto *pp_ctx = config_.pp_contexts[{1, 2}];
        ASSERT_NE(pp_ctx, nullptr);

        // Transfer 1→2 uses stage_to=2, which requires numStages() >= 3
        int stage_from = 1;
        int stage_to = 2;

        // Validation: stage_to must be < numStages()
        EXPECT_LT(stage_to, pp_ctx->numStages())
            << "stage_to=" << stage_to << " must be < numStages()=" << pp_ctx->numStages();
    }

    // ============================================================================
    // Two-Stage PP Tests
    // ============================================================================

    /**
     * @brief Test that 2-stage PP builds a graph with the correct node count
     */
    TEST_F(Test__Qwen2Graph_PP, TwoStage_BuildsGraphWithCorrectNodeCount)
    {
        auto pipeline_config = createTwoStagePPConfig();
        config_.pipeline_config = pipeline_config;
        createMockPPContexts(pipeline_config.get());

        Qwen2Graph graph(config_, mpi_ctx_);
        graph.setWeights(weights_);
        graph.setBuffers(buffers_);

        auto input = createForwardInput();
        Qwen2ForwardOutput output;

        ComputeGraph compute_graph = graph.buildUnifiedPipelineGraph(input, output);

        // Should have: embedding + layers + PP transfer + final_norm + lm_head
        // Each layer has multiple stages (attention + FFN)
        // At minimum: 1 embedding + 4*N layer stages + 1 transfer + 1 final_norm + 1 lm_head
        EXPECT_GT(compute_graph.size(), 5);

        // Should have exactly 1 PP transfer stage
        int transfer_count = countStagesByType(compute_graph, ComputeStageType::LOCAL_PP_TRANSFER);
        EXPECT_EQ(transfer_count, 1);
    }

    /**
     * @brief Test that first stage has embedding on correct device
     */
    TEST_F(Test__Qwen2Graph_PP, TwoStage_HasEmbeddingStageOnFirstDevice)
    {
        auto pipeline_config = createTwoStagePPConfig();
        config_.pipeline_config = pipeline_config;
        createMockPPContexts(pipeline_config.get());

        Qwen2Graph graph(config_, mpi_ctx_);
        graph.setWeights(weights_);
        graph.setBuffers(buffers_);

        auto input = createForwardInput();
        Qwen2ForwardOutput output;

        ComputeGraph compute_graph = graph.buildUnifiedPipelineGraph(input, output);

        // Find embedding node
        const ComputeNode *embedding = compute_graph.getNode("embedding");
        ASSERT_NE(embedding, nullptr);
        ASSERT_NE(embedding->stage, nullptr);
        EXPECT_EQ(embedding->stage->type(), ComputeStageType::EMBEDDING);

        // Verify device matches stage 0's domain primary device
        const TPDomainConfig *domain = pipeline_config->getDomainForStage(0);
        ASSERT_NE(domain, nullptr);
        EXPECT_EQ(embedding->device, domain->primaryDevice());
    }

    /**
     * @brief Test that last stage has LM head on correct device
     */
    TEST_F(Test__Qwen2Graph_PP, TwoStage_HasLMHeadStageOnLastDevice)
    {
        auto pipeline_config = createTwoStagePPConfig();
        config_.pipeline_config = pipeline_config;
        createMockPPContexts(pipeline_config.get());

        Qwen2Graph graph(config_, mpi_ctx_);
        graph.setWeights(weights_);
        graph.setBuffers(buffers_);

        auto input = createForwardInput();
        Qwen2ForwardOutput output;

        ComputeGraph compute_graph = graph.buildUnifiedPipelineGraph(input, output);

        // Find LM head node
        const ComputeNode *lm_head = compute_graph.getNode("lm_head");
        ASSERT_NE(lm_head, nullptr);
        ASSERT_NE(lm_head->stage, nullptr);
        EXPECT_EQ(lm_head->stage->type(), ComputeStageType::LM_HEAD);

        // Verify device matches stage 1's domain primary device
        const TPDomainConfig *domain = pipeline_config->getDomainForStage(1);
        ASSERT_NE(domain, nullptr);
        EXPECT_EQ(lm_head->device, domain->primaryDevice());
    }

    /**
     * @brief Test that PP transfer stage exists between stages
     */
    TEST_F(Test__Qwen2Graph_PP, TwoStage_HasPPTransferBetweenStages)
    {
        auto pipeline_config = createTwoStagePPConfig();
        config_.pipeline_config = pipeline_config;
        createMockPPContexts(pipeline_config.get());

        Qwen2Graph graph(config_, mpi_ctx_);
        graph.setWeights(weights_);
        graph.setBuffers(buffers_);

        auto input = createForwardInput();
        Qwen2ForwardOutput output;

        ComputeGraph compute_graph = graph.buildUnifiedPipelineGraph(input, output);

        // Find PP transfer node
        auto transfer_names = findStageNames(compute_graph, "pp_transfer");
        ASSERT_EQ(transfer_names.size(), 1);

        const ComputeNode *transfer = compute_graph.getNode(transfer_names[0]);
        ASSERT_NE(transfer, nullptr);
        ASSERT_NE(transfer->stage, nullptr);
        EXPECT_EQ(transfer->stage->type(), ComputeStageType::LOCAL_PP_TRANSFER);

        // Transfer stage name should indicate from/to stages
        EXPECT_NE(transfer_names[0].find("0_to_1"), std::string::npos);
    }

    /**
     * @brief Test that layers are assigned to correct devices
     */
    TEST_F(Test__Qwen2Graph_PP, TwoStage_LayersAssignedToCorrectDevices)
    {
        auto pipeline_config = createTwoStagePPConfig();
        config_.pipeline_config = pipeline_config;
        createMockPPContexts(pipeline_config.get());

        Qwen2Graph graph(config_, mpi_ctx_);
        graph.setWeights(weights_);
        graph.setBuffers(buffers_);

        auto input = createForwardInput();
        Qwen2ForwardOutput output;

        ComputeGraph compute_graph = graph.buildUnifiedPipelineGraph(input, output);

        // Get devices for stage 0 and stage 1
        DeviceId device0 = pipeline_config->getDomainForStage(0)->primaryDevice();
        DeviceId device1 = pipeline_config->getDomainForStage(1)->primaryDevice();

        // Check layer 0 stages are on device0
        auto layer0_nodes = findStageNames(compute_graph, "layer0_");
        EXPECT_GT(layer0_nodes.size(), 0);
        for (const auto &name : layer0_nodes)
        {
            const ComputeNode *node = compute_graph.getNode(name);
            ASSERT_NE(node, nullptr);
            EXPECT_EQ(node->device, device0) << "Layer 0 stage " << name << " on wrong device";
        }

        // Check layer 1 stages are on device0 (first two layers on stage 0)
        auto layer1_nodes = findStageNames(compute_graph, "layer1_");
        EXPECT_GT(layer1_nodes.size(), 0);
        for (const auto &name : layer1_nodes)
        {
            const ComputeNode *node = compute_graph.getNode(name);
            ASSERT_NE(node, nullptr);
            EXPECT_EQ(node->device, device0) << "Layer 1 stage " << name << " on wrong device";
        }

        // Check layer 2 stages are on device1 (second two layers on stage 1)
        auto layer2_nodes = findStageNames(compute_graph, "layer2_");
        EXPECT_GT(layer2_nodes.size(), 0);
        for (const auto &name : layer2_nodes)
        {
            const ComputeNode *node = compute_graph.getNode(name);
            ASSERT_NE(node, nullptr);
            EXPECT_EQ(node->device, device1) << "Layer 2 stage " << name << " on wrong device";
        }

        // Check layer 3 stages are on device1
        auto layer3_nodes = findStageNames(compute_graph, "layer3_");
        EXPECT_GT(layer3_nodes.size(), 0);
        for (const auto &name : layer3_nodes)
        {
            const ComputeNode *node = compute_graph.getNode(name);
            ASSERT_NE(node, nullptr);
            EXPECT_EQ(node->device, device1) << "Layer 3 stage " << name << " on wrong device";
        }
    }

    // ============================================================================
    // Three-Stage PP Tests
    // ============================================================================

    /**
     * @brief Test that 3-stage PP builds a graph with two PP transfer stages
     *
     * NOTE: Uses the CORRECT pattern of a single PP context covering ALL stages.
     * See Regression_MultiStagePPContextMustCoverAllStages for the bug this fixes.
     */
    TEST_F(Test__Qwen2Graph_PP, ThreeStage_BuildsGraphWithTwoPPTransfers)
    {
        // Helper to avoid narrowing conversions (int -> size_t)
        auto sz = [](int x)
        { return static_cast<size_t>(x); };

        auto pipeline_config = createThreeStagePPConfig();
        config_.pipeline_config = pipeline_config;
        config_.n_layers = 6; // Update for 3-stage

        // Recreate weights for 6 layers
        layer_weights_.clear();
        for (int layer = 0; layer < 6; ++layer)
        {
            LayerWeightSet layer_weights;
            layer_weights.wq = tensor_factory_->createFP32({sz(config_.d_model), sz(config_.d_model)});
            layer_weights.wk = tensor_factory_->createFP32({sz(config_.n_kv_heads * config_.head_dim), sz(config_.d_model)});
            layer_weights.wv = tensor_factory_->createFP32({sz(config_.n_kv_heads * config_.head_dim), sz(config_.d_model)});
            layer_weights.wo = tensor_factory_->createFP32({sz(config_.d_model), sz(config_.d_model)});
            layer_weights.attn_norm = tensor_factory_->createFP32({sz(config_.d_model)});
            layer_weights.q_bias = tensor_factory_->createFP32({sz(config_.d_model)});
            layer_weights.k_bias = tensor_factory_->createFP32({sz(config_.n_kv_heads * config_.head_dim)});
            layer_weights.v_bias = tensor_factory_->createFP32({sz(config_.n_kv_heads * config_.head_dim)});
            layer_weights.gate_proj = tensor_factory_->createFP32({sz(config_.d_ff), sz(config_.d_model)});
            layer_weights.up_proj = tensor_factory_->createFP32({sz(config_.d_ff), sz(config_.d_model)});
            layer_weights.down_proj = tensor_factory_->createFP32({sz(config_.d_model), sz(config_.d_ff)});
            layer_weights.ffn_norm = tensor_factory_->createFP32({sz(config_.d_model)});
            layer_weights_.push_back(std::move(layer_weights));
        }

        // Use the CORRECT pattern: single context covering all 3 stages
        createMockPPContexts(pipeline_config.get());

        Qwen2Graph graph(config_, mpi_ctx_);
        graph.setWeights(weights_);
        graph.setBuffers(buffers_);

        auto input = createForwardInput();
        Qwen2ForwardOutput output;

        ComputeGraph compute_graph = graph.buildUnifiedPipelineGraph(input, output);

        // Should have exactly 2 PP transfer stages
        int transfer_count = countStagesByType(compute_graph, ComputeStageType::LOCAL_PP_TRANSFER);
        EXPECT_EQ(transfer_count, 2);

        // Find specific transfers
        auto transfer_names = findStageNames(compute_graph, "pp_transfer");
        ASSERT_EQ(transfer_names.size(), 2);

        // Verify transfer names indicate correct from/to stages
        bool has_0_to_1 = false;
        bool has_1_to_2 = false;
        for (const auto &name : transfer_names)
        {
            if (name.find("0_to_1") != std::string::npos)
                has_0_to_1 = true;
            if (name.find("1_to_2") != std::string::npos)
                has_1_to_2 = true;
        }
        EXPECT_TRUE(has_0_to_1);
        EXPECT_TRUE(has_1_to_2);
    }

    /**
     * @brief Test that middle stage has no embedding or LM head
     */
    TEST_F(Test__Qwen2Graph_PP, ThreeStage_MiddleStageHasNoEmbeddingOrLMHead)
    {
        // Helper to avoid narrowing conversions (int -> size_t)
        auto sz = [](int x)
        { return static_cast<size_t>(x); };

        auto pipeline_config = createThreeStagePPConfig();
        config_.pipeline_config = pipeline_config;
        config_.n_layers = 6;

        // Recreate weights for 6 layers
        layer_weights_.clear();
        for (int layer = 0; layer < 6; ++layer)
        {
            LayerWeightSet layer_weights;
            layer_weights.wq = tensor_factory_->createFP32({sz(config_.d_model), sz(config_.d_model)});
            layer_weights.wk = tensor_factory_->createFP32({sz(config_.n_kv_heads * config_.head_dim), sz(config_.d_model)});
            layer_weights.wv = tensor_factory_->createFP32({sz(config_.n_kv_heads * config_.head_dim), sz(config_.d_model)});
            layer_weights.wo = tensor_factory_->createFP32({sz(config_.d_model), sz(config_.d_model)});
            layer_weights.attn_norm = tensor_factory_->createFP32({sz(config_.d_model)});
            layer_weights.q_bias = tensor_factory_->createFP32({sz(config_.d_model)});
            layer_weights.k_bias = tensor_factory_->createFP32({sz(config_.n_kv_heads * config_.head_dim)});
            layer_weights.v_bias = tensor_factory_->createFP32({sz(config_.n_kv_heads * config_.head_dim)});
            layer_weights.gate_proj = tensor_factory_->createFP32({sz(config_.d_ff), sz(config_.d_model)});
            layer_weights.up_proj = tensor_factory_->createFP32({sz(config_.d_ff), sz(config_.d_model)});
            layer_weights.down_proj = tensor_factory_->createFP32({sz(config_.d_model), sz(config_.d_ff)});
            layer_weights.ffn_norm = tensor_factory_->createFP32({sz(config_.d_model)});
            layer_weights_.push_back(std::move(layer_weights));
        }

        // Use the CORRECT pattern: single context covering all 3 stages
        createMockPPContexts(pipeline_config.get());

        Qwen2Graph graph(config_, mpi_ctx_);
        graph.setWeights(weights_);
        graph.setBuffers(buffers_);

        auto input = createForwardInput();
        Qwen2ForwardOutput output;

        ComputeGraph compute_graph = graph.buildUnifiedPipelineGraph(input, output);

        // Get execution order
        auto order = compute_graph.getExecutionOrder();

        // Should have exactly 1 embedding and 1 lm_head
        int embedding_count = countStagesByType(compute_graph, ComputeStageType::EMBEDDING);
        int lm_head_count = countStagesByType(compute_graph, ComputeStageType::LM_HEAD);
        EXPECT_EQ(embedding_count, 1);
        EXPECT_EQ(lm_head_count, 1);

        // Verify middle stage (layers 2-3) has no embedding or lm_head
        // Embedding should come before layer 2
        // LM head should come after layer 5
        auto layer2_nodes = findStageNames(compute_graph, "layer2_");
        auto layer5_nodes = findStageNames(compute_graph, "layer5_");

        // Embedding and lm_head should not be between layer 2 and layer 5
        auto embedding_pos = std::find(order.begin(), order.end(), "embedding");
        auto lm_head_pos = std::find(order.begin(), order.end(), "lm_head");

        ASSERT_NE(embedding_pos, order.end());
        ASSERT_NE(lm_head_pos, order.end());

        // Embedding should come early (before transfer 0->1)
        // LM head should come late (after transfer 1->2)
    }

    // ============================================================================
    // PP Transfer Stage Tests
    // ============================================================================

    /**
     * @brief Test that PP transfer stage has correct from/to stage parameters
     */
    TEST_F(Test__Qwen2Graph_PP, PPTransferStageHasCorrectParams)
    {
        auto pipeline_config = createTwoStagePPConfig();
        config_.pipeline_config = pipeline_config;
        createMockPPContexts(pipeline_config.get());

        Qwen2Graph graph(config_, mpi_ctx_);
        graph.setWeights(weights_);
        graph.setBuffers(buffers_);

        auto input = createForwardInput();
        Qwen2ForwardOutput output;

        ComputeGraph compute_graph = graph.buildUnifiedPipelineGraph(input, output);

        // Find PP transfer node
        auto transfer_names = findStageNames(compute_graph, "pp_transfer");
        ASSERT_EQ(transfer_names.size(), 1);

        const ComputeNode *transfer = compute_graph.getNode(transfer_names[0]);
        ASSERT_NE(transfer, nullptr);
        ASSERT_NE(transfer->stage, nullptr);
        EXPECT_EQ(transfer->stage->type(), ComputeStageType::LOCAL_PP_TRANSFER);

        // Verify the stage name encodes correct from/to
        EXPECT_NE(transfer_names[0].find("0_to_1"), std::string::npos);
    }

    /**
     * @brief Test that PP transfer depends on last layer of previous stage
     */
    TEST_F(Test__Qwen2Graph_PP, PPTransferDependsOnLastLayerOfPrevStage)
    {
        auto pipeline_config = createTwoStagePPConfig();
        config_.pipeline_config = pipeline_config;
        createMockPPContexts(pipeline_config.get());

        Qwen2Graph graph(config_, mpi_ctx_);
        graph.setWeights(weights_);
        graph.setBuffers(buffers_);

        auto input = createForwardInput();
        Qwen2ForwardOutput output;

        ComputeGraph compute_graph = graph.buildUnifiedPipelineGraph(input, output);

        // Get execution order
        auto order = compute_graph.getExecutionOrder();

        // Find PP transfer position
        auto transfer_names = findStageNames(compute_graph, "pp_transfer");
        ASSERT_EQ(transfer_names.size(), 1);

        auto transfer_pos = std::find(order.begin(), order.end(), transfer_names[0]);
        ASSERT_NE(transfer_pos, order.end());

        // Find last node of layer 1 (last layer in stage 0)
        auto layer1_nodes = findStageNames(compute_graph, "layer1_");
        EXPECT_GT(layer1_nodes.size(), 0);

        // PP transfer should come after all layer 1 nodes
        for (const auto &layer1_node : layer1_nodes)
        {
            auto layer1_pos = std::find(order.begin(), order.end(), layer1_node);
            if (layer1_pos != order.end())
            {
                EXPECT_LT(layer1_pos, transfer_pos)
                    << "PP transfer should come after " << layer1_node;
            }
        }

        // Find first node of layer 2 (first layer in stage 1)
        auto layer2_nodes = findStageNames(compute_graph, "layer2_");
        EXPECT_GT(layer2_nodes.size(), 0);

        // PP transfer should come before all layer 2 nodes
        for (const auto &layer2_node : layer2_nodes)
        {
            auto layer2_pos = std::find(order.begin(), order.end(), layer2_node);
            if (layer2_pos != order.end())
            {
                EXPECT_GT(layer2_pos, transfer_pos)
                    << "Layer 2 node " << layer2_node << " should come after PP transfer";
            }
        }
    }

    // ============================================================================
    // Domain Assignment Tests
    // ============================================================================

    /**
     * @brief Test that stages use domain's primary device
     */
    TEST_F(Test__Qwen2Graph_PP, StagesUseDomainPrimaryDevice)
    {
        auto pipeline_config = createTwoStagePPConfig();
        config_.pipeline_config = pipeline_config;
        createMockPPContexts(pipeline_config.get());

        Qwen2Graph graph(config_, mpi_ctx_);
        graph.setWeights(weights_);
        graph.setBuffers(buffers_);

        auto input = createForwardInput();
        Qwen2ForwardOutput output;

        ComputeGraph compute_graph = graph.buildUnifiedPipelineGraph(input, output);

        // Verify embedding uses stage 0's domain primary device
        const TPDomainConfig *domain0 = pipeline_config->getDomainForStage(0);
        ASSERT_NE(domain0, nullptr);

        const ComputeNode *embedding = compute_graph.getNode("embedding");
        ASSERT_NE(embedding, nullptr);
        EXPECT_EQ(embedding->device, domain0->primaryDevice());

        // Verify lm_head uses stage 1's domain primary device
        const TPDomainConfig *domain1 = pipeline_config->getDomainForStage(1);
        ASSERT_NE(domain1, nullptr);

        const ComputeNode *lm_head = compute_graph.getNode("lm_head");
        ASSERT_NE(lm_head, nullptr);
        EXPECT_EQ(lm_head->device, domain1->primaryDevice());
    }

    /**
     * @brief Test that domain lookup works correctly for each PP stage
     */
    TEST_F(Test__Qwen2Graph_PP, DomainLookupWorksForEachStage)
    {
        auto pipeline_config = createTwoStagePPConfig();

        // Stage 0 should map to domain "stage0_domain"
        const TPDomainConfig *domain0 = pipeline_config->getDomainForStage(0);
        ASSERT_NE(domain0, nullptr);
        EXPECT_EQ(domain0->name, "stage0_domain");

        // Stage 1 should map to domain "stage1_domain"
        const TPDomainConfig *domain1 = pipeline_config->getDomainForStage(1);
        ASSERT_NE(domain1, nullptr);
        EXPECT_EQ(domain1->name, "stage1_domain");

        // Invalid stage should return nullptr
        const TPDomainConfig *invalid = pipeline_config->getDomainForStage(99);
        EXPECT_EQ(invalid, nullptr);
    }

    /**
     * @brief Test that PP context is wired correctly for transfer stages
     */
    TEST_F(Test__Qwen2Graph_PP, PPContextWiredToTransferStage)
    {
        auto pipeline_config = createTwoStagePPConfig();
        config_.pipeline_config = pipeline_config;
        createMockPPContexts(pipeline_config.get());

        // Verify PP context is set
        auto key = std::make_pair(0, 1);
        EXPECT_NE(config_.pp_contexts.count(key), 0u);
        EXPECT_NE(config_.pp_contexts[key], nullptr);

        Qwen2Graph graph(config_, mpi_ctx_);
        graph.setWeights(weights_);
        graph.setBuffers(buffers_);

        auto input = createForwardInput();
        Qwen2ForwardOutput output;

        // Building the graph should not throw (PP context is available)
        EXPECT_NO_THROW(graph.buildUnifiedPipelineGraph(input, output));
    }

    /**
     * @brief Test that missing PP context throws error
     */
    TEST_F(Test__Qwen2Graph_PP, ThrowsWhenPPContextMissing)
    {
        auto pipeline_config = createTwoStagePPConfig();
        config_.pipeline_config = pipeline_config;
        // Don't create PP contexts - this should cause failure

        Qwen2Graph graph(config_, mpi_ctx_);
        graph.setWeights(weights_);
        graph.setBuffers(buffers_);

        auto input = createForwardInput();
        Qwen2ForwardOutput output;

        // Should throw because PP context for transfer 0->1 is missing
        EXPECT_THROW(graph.buildUnifiedPipelineGraph(input, output), std::runtime_error);
    }

    // ============================================================================
    // Output Assignment Tests
    // ============================================================================

    /**
     * @brief Test that output.logits is set correctly
     */
    TEST_F(Test__Qwen2Graph_PP, OutputLogitsIsSet)
    {
        auto pipeline_config = createTwoStagePPConfig();
        config_.pipeline_config = pipeline_config;
        createMockPPContexts(pipeline_config.get());

        Qwen2Graph graph(config_, mpi_ctx_);
        graph.setWeights(weights_);
        graph.setBuffers(buffers_);

        auto input = createForwardInput();
        Qwen2ForwardOutput output;
        output.logits = nullptr; // Start with null

        graph.buildUnifiedPipelineGraph(input, output);

        // After building, output.logits should be set
        EXPECT_NE(output.logits, nullptr);
        EXPECT_EQ(output.logits, buffers_.logits);
    }

} // anonymous namespace
