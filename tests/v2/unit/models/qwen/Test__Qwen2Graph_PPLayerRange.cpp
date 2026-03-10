/**
 * @file Test__Qwen2Graph_PPLayerRange.cpp
 * @brief Regression tests for PP stage absolute layer index validation
 *
 * Validates that buildPartialForwardGraph() correctly handles absolute layer indices
 * when config.n_layers (local stage count) differs from config.total_n_layers (total model).
 *
 * Root cause: For LocalPP, each PP stage's ModelContext overrides blockCount() to return
 * the local layer count (e.g., 4 for stage handling layers [4,8)). Qwen2GraphConfigBuilder
 * sets config.n_layers from this local count. buildPartialForwardGraph() then validates
 * absolute indices against n_layers, causing "Invalid layer range" for non-first stages.
 *
 * Fix: Added total_n_layers to GraphConfig, populated from IModelContext::totalBlockCount().
 * buildPartialForwardGraph() now validates against total_n_layers when available.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>
#include "models/qwen/Qwen2Graph.h"
#include "tensors/TensorFactory.h"
#include "backends/DeviceId.h"

using namespace llaminar2;

namespace
{

    class Test__Qwen2Graph_PPLayerRange : public ::testing::Test
    {
    protected:
        static constexpr int TOTAL_LAYERS = 8;
        static constexpr int LOCAL_LAYERS = 4;
        static constexpr int D_MODEL = 64;
        static constexpr int N_HEADS = 4;
        static constexpr int N_KV_HEADS = 2;
        static constexpr int HEAD_DIM = 16;
        static constexpr int D_FF = 128;
        static constexpr int VOCAB_SIZE = 1000;
        static constexpr int MAX_SEQ_LEN = 32;

        void SetUp() override
        {
            mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);
            tensor_factory_ = std::make_unique<TensorFactory>(*mpi_ctx_);
        }

        GraphConfig makeConfig(int n_layers, int total_n_layers)
        {
            GraphConfig config;
            config.n_layers = n_layers;
            config.total_n_layers = total_n_layers;
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
            return config;
        }

        struct LayerWeightSet
        {
            std::shared_ptr<TensorBase> wq, wk, wv, wo, attn_norm;
            std::shared_ptr<TensorBase> q_bias, k_bias, v_bias;
            std::shared_ptr<TensorBase> gate_proj, up_proj, down_proj, ffn_norm;
        };

        void createWeightsForLayers(int num_layers)
        {
            auto sz = [](int x)
            { return static_cast<size_t>(x); };

            embedding_table_ = tensor_factory_->createFP32({sz(VOCAB_SIZE), sz(D_MODEL)});
            final_norm_ = tensor_factory_->createFP32({sz(D_MODEL)});
            lm_head_ = tensor_factory_->createFP32({sz(VOCAB_SIZE), sz(D_MODEL)});

            layer_weights_.clear();
            for (int i = 0; i < num_layers; ++i)
            {
                LayerWeightSet lw;
                lw.wq = tensor_factory_->createFP32({sz(D_MODEL), sz(D_MODEL)});
                lw.wk = tensor_factory_->createFP32({sz(N_KV_HEADS * HEAD_DIM), sz(D_MODEL)});
                lw.wv = tensor_factory_->createFP32({sz(N_KV_HEADS * HEAD_DIM), sz(D_MODEL)});
                lw.wo = tensor_factory_->createFP32({sz(D_MODEL), sz(D_MODEL)});
                lw.attn_norm = tensor_factory_->createFP32({sz(D_MODEL)});
                lw.q_bias = tensor_factory_->createFP32({sz(D_MODEL)});
                lw.k_bias = tensor_factory_->createFP32({sz(N_KV_HEADS * HEAD_DIM)});
                lw.v_bias = tensor_factory_->createFP32({sz(N_KV_HEADS * HEAD_DIM)});
                lw.gate_proj = tensor_factory_->createFP32({sz(D_FF), sz(D_MODEL)});
                lw.up_proj = tensor_factory_->createFP32({sz(D_FF), sz(D_MODEL)});
                lw.down_proj = tensor_factory_->createFP32({sz(D_MODEL), sz(D_FF)});
                lw.ffn_norm = tensor_factory_->createFP32({sz(D_MODEL)});
                layer_weights_.push_back(std::move(lw));
            }

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

        void createBuffers()
        {
            auto sz = [](int x)
            { return static_cast<size_t>(x); };
            int max_tokens = MAX_SEQ_LEN;

            current_hidden_ = tensor_factory_->createFP32({sz(max_tokens), sz(D_MODEL)});
            buffers_.current_hidden = current_hidden_.get();
            logits_ = tensor_factory_->createFP32({sz(max_tokens), sz(VOCAB_SIZE)});
            buffers_.logits = logits_.get();
            residual_ = tensor_factory_->createFP32({sz(max_tokens), sz(D_MODEL)});
            buffers_.layer_buffers.residual = residual_.get();
            normalized_ = tensor_factory_->createFP32({sz(max_tokens), sz(D_MODEL)});
            buffers_.layer_buffers.normalized = normalized_.get();
            Q_ = tensor_factory_->createFP32({sz(max_tokens), sz(D_MODEL)});
            buffers_.layer_buffers.Q = Q_.get();
            K_ = tensor_factory_->createFP32({sz(max_tokens), sz(N_KV_HEADS * HEAD_DIM)});
            buffers_.layer_buffers.K = K_.get();
            V_ = tensor_factory_->createFP32({sz(max_tokens), sz(N_KV_HEADS * HEAD_DIM)});
            buffers_.layer_buffers.V = V_.get();
            attn_output_ = tensor_factory_->createFP32({sz(max_tokens), sz(D_MODEL)});
            buffers_.layer_buffers.attn_output = attn_output_.get();
            attn_proj_ = tensor_factory_->createFP32({sz(max_tokens), sz(D_MODEL)});
            buffers_.layer_buffers.attn_proj = attn_proj_.get();
            gate_ = tensor_factory_->createFP32({sz(max_tokens), sz(D_FF)});
            buffers_.layer_buffers.gate = gate_.get();
            up_ = tensor_factory_->createFP32({sz(max_tokens), sz(D_FF)});
            buffers_.layer_buffers.up = up_.get();
            ffn_output_ = tensor_factory_->createFP32({sz(max_tokens), sz(D_FF)});
            buffers_.layer_buffers.ffn_output = ffn_output_.get();
        }

        Qwen2ForwardInput createForwardInput(int batch_size = 1, int seq_len = 4)
        {
            token_ids_.resize(batch_size * seq_len);
            for (int i = 0; i < batch_size * seq_len; ++i)
                token_ids_[i] = i % VOCAB_SIZE;

            Qwen2ForwardInput input;
            input.token_ids = token_ids_.data();
            input.batch_size = batch_size;
            input.seq_len = seq_len;
            input.position_offset = 0;
            input.device = DeviceId::cpu();
            input.kv_cache = nullptr;
            return input;
        }

        std::shared_ptr<MPIContext> mpi_ctx_;
        std::unique_ptr<TensorFactory> tensor_factory_;
        std::vector<LayerWeightSet> layer_weights_;
        std::shared_ptr<TensorBase> embedding_table_, final_norm_, lm_head_;
        ModelWeights weights_;
        ModelBuffers buffers_;
        std::shared_ptr<TensorBase> current_hidden_, logits_;
        std::shared_ptr<TensorBase> residual_, normalized_;
        std::shared_ptr<TensorBase> Q_, K_, V_;
        std::shared_ptr<TensorBase> attn_output_, attn_proj_;
        std::shared_ptr<TensorBase> gate_, up_, ffn_output_;
        std::vector<int> token_ids_;
    };

    // ============================================================================
    // Regression: PP stage 1 absolute layer indices [N/2, N) with n_layers=N/2
    // ============================================================================

    /**
     * @brief Regression test: buildPartialForwardGraph with absolute layer indices
     *        beyond local n_layers should succeed when total_n_layers is set.
     *
     * Before fix: buildPartialForwardGraph(4, 8, ...) threw "Invalid layer range"
     * because config.n_layers=4 (local PP stage count) and 8 > 4.
     */
    TEST_F(Test__Qwen2Graph_PPLayerRange, AbsoluteIndicesBeyondLocalCount)
    {
        // 8-layer model, PP stage handles layers [4, 8)
        // n_layers=4 (local), total_n_layers=8 (full model)
        GraphConfig config = makeConfig(LOCAL_LAYERS, TOTAL_LAYERS);

        // Weights for ALL 8 layers (absolute indices 0-7)
        createWeightsForLayers(TOTAL_LAYERS);
        createBuffers();

        Qwen2Graph graph(config, mpi_ctx_);
        graph.setWeights(weights_);
        graph.setBuffers(buffers_);

        auto input = createForwardInput();
        // PP stage without embedding needs external_hidden_state
        input.external_hidden_state = current_hidden_.get();
        Qwen2ForwardOutput output;
        output.hidden = current_hidden_.get();

        // This should NOT throw — absolute indices [4, 8) are valid for 8-layer model
        EXPECT_NO_THROW(
            graph.buildPartialForwardGraph(input, output, 4, 8, false, true));
    }

    /**
     * @brief Stage 0 with absolute indices [0, N/2) should also work.
     */
    TEST_F(Test__Qwen2Graph_PPLayerRange, FirstStageAbsoluteIndicesStillWork)
    {
        GraphConfig config = makeConfig(LOCAL_LAYERS, TOTAL_LAYERS);
        createWeightsForLayers(TOTAL_LAYERS);
        createBuffers();

        Qwen2Graph graph(config, mpi_ctx_);
        graph.setWeights(weights_);
        graph.setBuffers(buffers_);

        auto input = createForwardInput();
        Qwen2ForwardOutput output;
        output.hidden = current_hidden_.get();

        // First PP stage: layers [0, 4) with embedding
        EXPECT_NO_THROW(
            graph.buildPartialForwardGraph(input, output, 0, 4, true, false));
    }

    /**
     * @brief When total_n_layers is not set (0), fall back to n_layers for validation.
     */
    TEST_F(Test__Qwen2Graph_PPLayerRange, FallbackToNLayersWhenTotalNotSet)
    {
        // total_n_layers=0 → fallback to n_layers=4
        GraphConfig config = makeConfig(4, 0);
        createWeightsForLayers(4);
        createBuffers();

        Qwen2Graph graph(config, mpi_ctx_);
        graph.setWeights(weights_);
        graph.setBuffers(buffers_);

        auto input = createForwardInput();
        Qwen2ForwardOutput output;
        output.hidden = current_hidden_.get();

        // [0, 4) should work
        EXPECT_NO_THROW(
            graph.buildPartialForwardGraph(input, output, 0, 4, true, true));

        // [0, 5) should throw — beyond n_layers=4
        EXPECT_THROW(
            graph.buildPartialForwardGraph(input, output, 0, 5, true, true),
            std::invalid_argument);
    }

    /**
     * @brief Truly invalid layer ranges should still throw.
     */
    TEST_F(Test__Qwen2Graph_PPLayerRange, InvalidRangesStillThrow)
    {
        GraphConfig config = makeConfig(LOCAL_LAYERS, TOTAL_LAYERS);
        createWeightsForLayers(TOTAL_LAYERS);
        createBuffers();

        Qwen2Graph graph(config, mpi_ctx_);
        graph.setWeights(weights_);
        graph.setBuffers(buffers_);

        auto input = createForwardInput();
        Qwen2ForwardOutput output;
        output.hidden = current_hidden_.get();

        // Negative first_layer
        EXPECT_THROW(
            graph.buildPartialForwardGraph(input, output, -1, 4, true, false),
            std::invalid_argument);

        // last_layer > total_n_layers
        EXPECT_THROW(
            graph.buildPartialForwardGraph(input, output, 0, 9, true, true),
            std::invalid_argument);

        // first_layer >= last_layer
        EXPECT_THROW(
            graph.buildPartialForwardGraph(input, output, 4, 4, false, false),
            std::invalid_argument);

        // first_layer > last_layer
        EXPECT_THROW(
            graph.buildPartialForwardGraph(input, output, 5, 4, false, false),
            std::invalid_argument);
    }

    /**
     * @brief 3-way PP split: stage 2 uses layers [6, 9) on a 9-layer model where
     *        n_layers=3 (local) and total_n_layers=9.
     */
    TEST_F(Test__Qwen2Graph_PPLayerRange, ThreeWaySplitLastStage)
    {
        GraphConfig config = makeConfig(3, 9);
        createWeightsForLayers(9);
        createBuffers();

        Qwen2Graph graph(config, mpi_ctx_);
        graph.setWeights(weights_);
        graph.setBuffers(buffers_);

        auto input = createForwardInput();
        // PP stage without embedding needs external_hidden_state
        input.external_hidden_state = current_hidden_.get();
        Qwen2ForwardOutput output;
        output.hidden = current_hidden_.get();

        // Last of 3 stages: layers [6, 9)
        EXPECT_NO_THROW(
            graph.buildPartialForwardGraph(input, output, 6, 9, false, true));
    }

    // ============================================================================
    // Regression: IModelContext::totalBlockCount()
    // ============================================================================

    /**
     * @brief Verify that total_n_layers == n_layers when both are set to same value
     *        (non-PP scenario). Validation should work normally.
     */
    TEST_F(Test__Qwen2Graph_PPLayerRange, NonPPScenarioSameValues)
    {
        // Non-PP: n_layers=total_n_layers=8
        GraphConfig config = makeConfig(TOTAL_LAYERS, TOTAL_LAYERS);
        createWeightsForLayers(TOTAL_LAYERS);
        createBuffers();

        Qwen2Graph graph(config, mpi_ctx_);
        graph.setWeights(weights_);
        graph.setBuffers(buffers_);

        auto input = createForwardInput();
        Qwen2ForwardOutput output;
        output.hidden = current_hidden_.get();

        // Full range should work
        EXPECT_NO_THROW(
            graph.buildPartialForwardGraph(input, output, 0, 8, true, true));

        // Beyond total should still throw
        EXPECT_THROW(
            graph.buildPartialForwardGraph(input, output, 0, 9, true, true),
            std::invalid_argument);
    }

} // namespace
