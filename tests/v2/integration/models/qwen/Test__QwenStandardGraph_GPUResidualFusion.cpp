/**
 * @file Test__QwenStandardGraph_GPUResidualFusion.cpp
 * @brief CUDA and ROCm integration regressions for PP residual-fusion lowering.
 *
 * These tests previously lived in a unit suite that selected a fake CUDA
 * device without supplying a real stream. GPU graph construction now requires
 * the exact non-default stream that owns device-state publication, so the
 * regressions run as integration tests against real CUDA and ROCm streams.
 * The tests inspect graph structure only; creating the stream is nevertheless
 * important because it proves the production graph-builder contract rather
 * than bypassing it with a sentinel pointer.
 */

#include <gtest/gtest.h>

#include "backends/DeviceId.h"
#include "execution/compute_stages/IComputeStage.h"
#include "models/qwen/QwenStandardGraph.h"
#include "tensors/TensorFactory.h"
#include "utils/ScopedGPUStream.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        /**
         * @brief Build small production-shaped graphs on each GPU backend.
         *
         * Tensor payloads remain on the host because these regressions only
         * lower the declarative graph. The selected backend and the publication
         * stream are real, which keeps all GPU-specific API invariants active.
         */
        class Test__QwenStandardGraph_GPUResidualFusion
            : public ::testing::TestWithParam<DeviceId>
        {
        protected:
            static constexpr int kTotalLayers = 8;
            static constexpr int kLocalLayers = 4;
            static constexpr int kDModel = 64;
            static constexpr int kNumHeads = 4;
            static constexpr int kNumKVHeads = 2;
            static constexpr int kHeadDim = 16;
            static constexpr int kIntermediate = 128;
            static constexpr int kVocabSize = 1000;
            static constexpr int kMaxSeqLen = 32;

            void SetUp() override
            {
                mpi_context_ =
                    std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);
                tensor_factory_ =
                    std::make_unique<TensorFactory>(*mpi_context_);
                publication_stream_ =
                    std::make_unique<ScopedGPUStream>(GetParam());
            }

            [[nodiscard]] GraphConfig makeConfig(
                int local_layers,
                int total_layers) const
            {
                GraphConfig config;
                config.n_layers = local_layers;
                config.total_n_layers = total_layers;
                config.d_model = kDModel;
                config.n_heads = kNumHeads;
                config.n_kv_heads = kNumKVHeads;
                config.head_dim = kHeadDim;
                config.d_ff = kIntermediate;
                config.vocab_size = kVocabSize;
                config.rms_norm_eps = 1e-6f;
                config.rope_theta = 10000.0f;
                config.default_device = GetParam();
                config.max_seq_len = kMaxSeqLen;
                return config;
            }

            void createWeights(int layer_count)
            {
                const auto size = [](int value)
                { return static_cast<size_t>(value); };

                embedding_table_ = tensor_factory_->createFP32(
                    {size(kVocabSize), size(kDModel)});
                final_norm_ =
                    tensor_factory_->createFP32({size(kDModel)});
                lm_head_ = tensor_factory_->createFP32(
                    {size(kVocabSize), size(kDModel)});

                layer_weights_.clear();
                layer_weights_.reserve(static_cast<size_t>(layer_count));
                for (int layer = 0; layer < layer_count; ++layer)
                {
                    LayerWeightSet weights;
                    weights.wq = tensor_factory_->createFP32(
                        {size(kDModel), size(kDModel)});
                    weights.wk = tensor_factory_->createFP32(
                        {size(kNumKVHeads * kHeadDim), size(kDModel)});
                    weights.wv = tensor_factory_->createFP32(
                        {size(kNumKVHeads * kHeadDim), size(kDModel)});
                    weights.wo = tensor_factory_->createFP32(
                        {size(kDModel), size(kDModel)});
                    weights.attn_norm =
                        tensor_factory_->createFP32({size(kDModel)});
                    weights.q_bias =
                        tensor_factory_->createFP32({size(kDModel)});
                    weights.k_bias = tensor_factory_->createFP32(
                        {size(kNumKVHeads * kHeadDim)});
                    weights.v_bias = tensor_factory_->createFP32(
                        {size(kNumKVHeads * kHeadDim)});
                    weights.gate_proj = tensor_factory_->createFP32(
                        {size(kIntermediate), size(kDModel)});
                    weights.up_proj = tensor_factory_->createFP32(
                        {size(kIntermediate), size(kDModel)});
                    weights.down_proj = tensor_factory_->createFP32(
                        {size(kDModel), size(kIntermediate)});
                    weights.ffn_norm =
                        tensor_factory_->createFP32({size(kDModel)});
                    layer_weights_.push_back(std::move(weights));
                }

                model_weights_.embedding_table = embedding_table_.get();
                model_weights_.final_norm = final_norm_.get();
                model_weights_.lm_head = lm_head_.get();
                model_weights_.get_layer_weights =
                    [this](int layer) -> LayerWeights
                {
                    if (layer < 0 ||
                        layer >= static_cast<int>(layer_weights_.size()))
                    {
                        return {};
                    }

                    const LayerWeightSet &stored =
                        layer_weights_[static_cast<size_t>(layer)];
                    LayerWeights result;
                    result.wq = stored.wq.get();
                    result.wk = stored.wk.get();
                    result.wv = stored.wv.get();
                    result.wo = stored.wo.get();
                    result.attn_norm = stored.attn_norm.get();
                    result.q_bias = stored.q_bias.get();
                    result.k_bias = stored.k_bias.get();
                    result.v_bias = stored.v_bias.get();
                    result.gate_proj = stored.gate_proj.get();
                    result.up_proj = stored.up_proj.get();
                    result.down_proj = stored.down_proj.get();
                    result.ffn_norm = stored.ffn_norm.get();
                    return result;
                };
            }

            void createBuffers()
            {
                const auto size = [](int value)
                { return static_cast<size_t>(value); };
                const size_t tokens = size(kMaxSeqLen);

                current_hidden_ = tensor_factory_->createFP32(
                    {tokens, size(kDModel)});
                logits_ = tensor_factory_->createFP32(
                    {tokens, size(kVocabSize)});
                residual_ = tensor_factory_->createFP32(
                    {tokens, size(kDModel)});
                normalized_ = tensor_factory_->createFP32(
                    {tokens, size(kDModel)});
                query_ = tensor_factory_->createFP32(
                    {tokens, size(kDModel)});
                key_ = tensor_factory_->createFP32(
                    {tokens, size(kNumKVHeads * kHeadDim)});
                value_ = tensor_factory_->createFP32(
                    {tokens, size(kNumKVHeads * kHeadDim)});
                attention_output_ = tensor_factory_->createFP32(
                    {tokens, size(kDModel)});
                attention_projection_ = tensor_factory_->createFP32(
                    {tokens, size(kDModel)});
                gate_ = tensor_factory_->createFP32(
                    {tokens, size(kIntermediate)});
                up_ = tensor_factory_->createFP32(
                    {tokens, size(kIntermediate)});
                ffn_output_ = tensor_factory_->createFP32(
                    {tokens, size(kIntermediate)});

                model_buffers_.current_hidden = current_hidden_.get();
                model_buffers_.logits = logits_.get();
                model_buffers_.layer_buffers.residual = residual_.get();
                model_buffers_.layer_buffers.normalized = normalized_.get();
                model_buffers_.layer_buffers.Q = query_.get();
                model_buffers_.layer_buffers.K = key_.get();
                model_buffers_.layer_buffers.V = value_.get();
                model_buffers_.layer_buffers.attn_output =
                    attention_output_.get();
                model_buffers_.layer_buffers.attn_proj =
                    attention_projection_.get();
                model_buffers_.layer_buffers.gate = gate_.get();
                model_buffers_.layer_buffers.up = up_.get();
                model_buffers_.layer_buffers.ffn_output = ffn_output_.get();
            }

            [[nodiscard]] ForwardInput makeInput(
                bool external_hidden_state)
            {
                token_ids_.resize(4);
                position_ids_.resize(4);
                for (int index = 0; index < 4; ++index)
                {
                    token_ids_[static_cast<size_t>(index)] =
                        index % kVocabSize;
                    position_ids_[static_cast<size_t>(index)] = index;
                }

                ForwardInput input;
                input.token_ids = token_ids_.data();
                input.position_ids = position_ids_.data();
                input.batch_size = 1;
                input.seq_len = 4;
                input.position_offset = 0;
                input.device = GetParam();
                input.device_state_publication_stream =
                    publication_stream_->get();
                input.external_hidden_state =
                    external_hidden_state ? current_hidden_.get() : nullptr;
                return input;
            }

            [[nodiscard]] static std::optional<ComputeStageType>
            stageType(const ComputeGraph &graph, const std::string &pattern)
            {
                for (const std::string &name : graph.getExecutionOrder())
                {
                    if (name.find(pattern) == std::string::npos)
                        continue;

                    const ComputeNode *node = graph.getNode(name);
                    if (node && node->stage)
                        return node->stage->type();
                }
                return std::nullopt;
            }

            struct LayerWeightSet
            {
                std::shared_ptr<TensorBase> wq;
                std::shared_ptr<TensorBase> wk;
                std::shared_ptr<TensorBase> wv;
                std::shared_ptr<TensorBase> wo;
                std::shared_ptr<TensorBase> attn_norm;
                std::shared_ptr<TensorBase> q_bias;
                std::shared_ptr<TensorBase> k_bias;
                std::shared_ptr<TensorBase> v_bias;
                std::shared_ptr<TensorBase> gate_proj;
                std::shared_ptr<TensorBase> up_proj;
                std::shared_ptr<TensorBase> down_proj;
                std::shared_ptr<TensorBase> ffn_norm;
            };

            std::shared_ptr<IMPIContext> mpi_context_;
            std::unique_ptr<TensorFactory> tensor_factory_;
            std::unique_ptr<ScopedGPUStream> publication_stream_;
            std::vector<LayerWeightSet> layer_weights_;
            std::shared_ptr<TensorBase> embedding_table_;
            std::shared_ptr<TensorBase> final_norm_;
            std::shared_ptr<TensorBase> lm_head_;
            ModelWeights model_weights_;
            ModelBuffers model_buffers_;
            std::shared_ptr<TensorBase> current_hidden_;
            std::shared_ptr<TensorBase> logits_;
            std::shared_ptr<TensorBase> residual_;
            std::shared_ptr<TensorBase> normalized_;
            std::shared_ptr<TensorBase> query_;
            std::shared_ptr<TensorBase> key_;
            std::shared_ptr<TensorBase> value_;
            std::shared_ptr<TensorBase> attention_output_;
            std::shared_ptr<TensorBase> attention_projection_;
            std::shared_ptr<TensorBase> gate_;
            std::shared_ptr<TensorBase> up_;
            std::shared_ptr<TensorBase> ffn_output_;
            std::vector<int> token_ids_;
            std::vector<int> position_ids_;
        };

        TEST_P(
            Test__QwenStandardGraph_GPUResidualFusion,
            FirstLayerOfPPStageDoesNotFuseAnUnavailableResidual)
        {
            GraphConfig config =
                makeConfig(kLocalLayers, kTotalLayers);
            config.pp_layer_offset = 4;
            createWeights(kTotalLayers);
            createBuffers();

            QwenStandardGraph graph(config, mpi_context_);
            graph.setWeights(model_weights_);
            graph.setBuffers(model_buffers_);

            ForwardInput input = makeInput(true);
            ForwardOutput output;
            output.hidden = current_hidden_.get();
            const ComputeGraph compute_graph =
                graph.buildPartialForwardGraph(
                    input, output, 4, 8, false, true);

            const auto layer4 =
                stageType(compute_graph, "layer4_attn_norm");
            ASSERT_TRUE(layer4.has_value());
            EXPECT_EQ(*layer4, ComputeStageType::RMS_NORM);

            const auto layer5 =
                stageType(compute_graph, "layer5_attn_norm");
            ASSERT_TRUE(layer5.has_value());
            EXPECT_EQ(
                *layer5,
                ComputeStageType::FUSED_RESIDUAL_NORM);
        }

        TEST_P(
            Test__QwenStandardGraph_GPUResidualFusion,
            PPStageOnlyMaterializesTheTerminalFFNResidual)
        {
            GraphConfig config =
                makeConfig(kLocalLayers, kTotalLayers);
            config.pp_layer_offset = 4;
            createWeights(kTotalLayers);
            createBuffers();

            QwenStandardGraph graph(config, mpi_context_);
            graph.setWeights(model_weights_);
            graph.setBuffers(model_buffers_);

            ForwardInput input = makeInput(true);
            ForwardOutput output;
            output.hidden = current_hidden_.get();
            const ComputeGraph compute_graph =
                graph.buildPartialForwardGraph(
                    input, output, 4, 8, false, true);

            EXPECT_FALSE(
                stageType(compute_graph, "layer4_ffn_residual")
                    .has_value());
            EXPECT_FALSE(
                stageType(compute_graph, "layer5_ffn_residual")
                    .has_value());
            EXPECT_FALSE(
                stageType(compute_graph, "layer6_ffn_residual")
                    .has_value());

            const auto layer7 =
                stageType(compute_graph, "layer7_ffn_residual");
            ASSERT_TRUE(layer7.has_value());
            EXPECT_EQ(*layer7, ComputeStageType::ADD_RESIDUAL);
        }

        TEST_P(
            Test__QwenStandardGraph_GPUResidualFusion,
            SingleDeviceResidualFusionPolicyIsUnchanged)
        {
            GraphConfig config =
                makeConfig(kTotalLayers, kTotalLayers);
            config.pp_layer_offset = 0;
            createWeights(kTotalLayers);
            createBuffers();

            QwenStandardGraph graph(config, mpi_context_);
            graph.setWeights(model_weights_);
            graph.setBuffers(model_buffers_);

            ForwardInput input = makeInput(false);
            ForwardOutput output;
            output.hidden = current_hidden_.get();
            const ComputeGraph compute_graph =
                graph.buildPartialForwardGraph(
                    input, output, 0, 8, true, true);

            const auto layer0 =
                stageType(compute_graph, "layer0_attn_norm");
            ASSERT_TRUE(layer0.has_value());
            EXPECT_EQ(*layer0, ComputeStageType::RMS_NORM);

            const auto layer1 =
                stageType(compute_graph, "layer1_attn_norm");
            ASSERT_TRUE(layer1.has_value());
            EXPECT_EQ(
                *layer1,
                ComputeStageType::FUSED_RESIDUAL_NORM);

            EXPECT_TRUE(
                stageType(compute_graph, "layer7_ffn_residual")
                    .has_value());
            EXPECT_FALSE(
                stageType(compute_graph, "layer6_ffn_residual")
                    .has_value());
        }

        TEST_P(
            Test__QwenStandardGraph_GPUResidualFusion,
            ThreeWayPPStageUsesItsOwnFirstLayerBoundary)
        {
            GraphConfig config = makeConfig(3, 9);
            config.pp_layer_offset = 3;
            createWeights(9);
            createBuffers();

            QwenStandardGraph graph(config, mpi_context_);
            graph.setWeights(model_weights_);
            graph.setBuffers(model_buffers_);

            ForwardInput input = makeInput(true);
            ForwardOutput output;
            output.hidden = current_hidden_.get();
            const ComputeGraph compute_graph =
                graph.buildPartialForwardGraph(
                    input, output, 3, 6, false, false);

            const auto layer3 =
                stageType(compute_graph, "layer3_attn_norm");
            ASSERT_TRUE(layer3.has_value());
            EXPECT_EQ(*layer3, ComputeStageType::RMS_NORM);

            const auto layer4 =
                stageType(compute_graph, "layer4_attn_norm");
            ASSERT_TRUE(layer4.has_value());
            EXPECT_EQ(
                *layer4,
                ComputeStageType::FUSED_RESIDUAL_NORM);

            EXPECT_TRUE(
                stageType(compute_graph, "layer5_ffn_residual")
                    .has_value());
            EXPECT_FALSE(
                stageType(compute_graph, "layer3_ffn_residual")
                    .has_value());
            EXPECT_FALSE(
                stageType(compute_graph, "layer4_ffn_residual")
                    .has_value());
        }

        [[nodiscard]] std::string gpuBackendName(
            const ::testing::TestParamInfo<DeviceId> &info)
        {
            return info.param.is_cuda() ? "CUDA" : "ROCm";
        }

        INSTANTIATE_TEST_SUITE_P(
            GPUBackends,
            Test__QwenStandardGraph_GPUResidualFusion,
            ::testing::Values(DeviceId::cuda(0), DeviceId::rocm(0)),
            gpuBackendName);
    }
}
