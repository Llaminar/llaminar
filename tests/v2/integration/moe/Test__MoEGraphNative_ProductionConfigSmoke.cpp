/**
 * @file Test__MoEGraphNative_ProductionConfigSmoke.cpp
 * @brief Model-light production config smoke test for graph-native MoE overlay lowering.
 */

#include <gtest/gtest.h>

#include "execution/compute_stages/stages/MoESparseDispatchStage.h"
#include "execution/compute_stages/stages/MoESparseReturnReduceStage.h"
#include "execution/factory/InferenceRunnerFactory.h"
#include "mocks/MockLocalTPContext.h"
#include "mocks/MockMPIContext.h"
#include "mocks/MockModelContext.h"
#include "models/qwen35moe/Qwen35MoEGraph.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        constexpr int kDModel = 8;
        constexpr int kIntermediate = 5;
        constexpr int kNumExperts = 6;
        constexpr int kTopK = 2;
        constexpr int kSeqLen = 2;
        constexpr int kBatchSize = 1;

        class TensorArena
        {
        public:
            FP32Tensor *fp32(std::vector<size_t> shape)
            {
                auto tensor = std::make_shared<FP32Tensor>(std::move(shape));
                auto *ptr = tensor.get();
                tensors_.push_back(std::move(tensor));
                return ptr;
            }

        private:
            std::vector<std::shared_ptr<TensorBase>> tensors_;
        };

        void fill(FP32Tensor *tensor, float value)
        {
            std::fill_n(tensor->mutable_data(), tensor->numel(), value);
        }

        ExecutionDomainDefinition denseContinuationDomain()
        {
            ExecutionDomainDefinition domain;
            domain.name = "dense_cont";
            domain.scope = ExecutionDomainScope::LOCAL;
            domain.participants = {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)};
            domain.weights = {0.5f, 0.5f};
            domain.backend = CollectiveBackendType::HOST;
            domain.owner_rank = 0;
            domain.compute_kind = ExecutionDomainComputeKind::REPLICATED_EXPERTS;
            return domain;
        }

        ExpertComputeDomain routedDomain(const std::string &name, GlobalDeviceAddress participant)
        {
            ExpertComputeDomain domain;
            domain.name = name;
            domain.kind = ExpertDomainKind::SingleDevice;
            domain.backend = CollectiveBackendType::HOST;
            domain.participants = {std::move(participant)};
            domain.owner_rank = 0;
            domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
            return domain;
        }

        ExpertRoutedTier routedTier(const std::string &name, const std::string &domain, int priority, bool fallback = false)
        {
            ExpertRoutedTier tier;
            tier.name = name;
            tier.domain = domain;
            tier.priority = priority;
            tier.fallback = fallback;
            return tier;
        }

        std::shared_ptr<MoEExpertParallelPlan> makeProductionPlan()
        {
            auto plan = std::make_shared<MoEExpertParallelPlan>();
            plan->enabled = true;
            plan->execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
            plan->continuation_domain = "dense_cont";
            plan->base_model_domain = "dense_cont";
            plan->shared_expert_domain = "dense_cont";
            plan->continuation_domain_spec.domain = "dense_cont";
            plan->continuation_domain_spec.logical_root_participant = 0;
            plan->continuation_domain_spec.dense_tp_enabled = true;
            plan->continuation_domain_spec.hidden_layout = MoEContinuationActivationLayout::ReplicatedHidden;
            plan->residency_policy = ExpertResidencyPolicy::ExplicitMasks;
            plan->dense_domains = {denseContinuationDomain()};
            plan->domains = {
                routedDomain("hot_domain", GlobalDeviceAddress::cpu(0)),
                routedDomain("cold_domain", GlobalDeviceAddress::cpu(1)),
            };
            plan->routed_tiers = {
                routedTier("hot", "hot_domain", 0),
                routedTier("cold", "cold_domain", 99, true),
            };
            plan->placements.push_back(ExpertLayerPlacement{
                .layer = 0,
                .routed_expert_tier = {0, 1, 0, 1, 0, 1},
            });
            validateMoEExpertParallelPlanOrThrow(
                *plan,
                {.layer_count = 1, .routed_expert_count = kNumExperts});
            return plan;
        }

        std::shared_ptr<MockModelContext> makeModelContext()
        {
            auto model_ctx = MockModelContextBuilder()
                                 .setArchitecture("qwen35moe")
                                 .setBlockCount(1)
                                 .setEmbeddingLength(kDModel)
                                 .setHeadCount(2)
                                 .setHeadCountKV(2)
                                 .setVocabSize(32)
                                 .setContextLength(32)
                                 .setFeedForwardLength(kIntermediate)
                                 .build();
            model_ctx->mockLoader().setIntParam("qwen3moe.expert_count", kNumExperts);
            model_ctx->mockLoader().setIntParam("qwen3moe.expert_feed_forward_length", kIntermediate);
            model_ctx->mockLoader().setIntParam("qwen3moe.expert_shared_count", 0);
            return model_ctx;
        }

        void configureGraphDimensions(GraphConfig &config)
        {
            config.n_layers = 1;
            config.total_n_layers = 1;
            config.d_model = kDModel;
            config.n_heads = 2;
            config.n_kv_heads = 2;
            config.head_dim = 4;
            config.d_ff = 16;
            config.vocab_size = 32;
            config.rms_norm_eps = 1e-6f;
            config.default_device = DeviceId::cpu();
            config.moe.num_experts = kNumExperts;
            config.moe.top_k = kTopK;
            config.moe.intermediate_size = kIntermediate;
            config.moe.norm_topk_prob = true;
        }

        void fillExpert3D(FP32Tensor *tensor, int rows, int cols, float scale)
        {
            float *data = tensor->mutable_data();
            for (int expert = 0; expert < kNumExperts; ++expert)
            {
                for (int row = 0; row < rows; ++row)
                {
                    for (int col = 0; col < cols; ++col)
                    {
                        const size_t offset = static_cast<size_t>(expert) * rows * cols +
                                              static_cast<size_t>(row) * cols +
                                              static_cast<size_t>(col);
                        data[offset] = scale * static_cast<float>((expert + 1) * 3 + row + 1) +
                                       0.001f * static_cast<float>(col + 1);
                    }
                }
            }
        }

        LayerWeights makeLayerWeights(TensorArena &arena)
        {
            LayerWeights layer;
            layer.ffn_norm = arena.fp32({kDModel});
            fill(static_cast<FP32Tensor *>(layer.ffn_norm), 1.0f);
            layer.moe_gate = arena.fp32({kNumExperts, kDModel});
            fill(static_cast<FP32Tensor *>(layer.moe_gate), 0.25f);
            layer.moe_gate_exps = arena.fp32({kDModel, kIntermediate, kNumExperts});
            layer.moe_up_exps = arena.fp32({kDModel, kIntermediate, kNumExperts});
            layer.moe_down_exps = arena.fp32({kIntermediate, kDModel, kNumExperts});
            fillExpert3D(static_cast<FP32Tensor *>(layer.moe_gate_exps), kIntermediate, kDModel, 0.010f);
            fillExpert3D(static_cast<FP32Tensor *>(layer.moe_up_exps), kIntermediate, kDModel, 0.012f);
            fillExpert3D(static_cast<FP32Tensor *>(layer.moe_down_exps), kDModel, kIntermediate, 0.008f);
            return layer;
        }

        ActivationBuffers makeActivationBuffers(TensorArena &arena)
        {
            ActivationBuffers buffers;
            buffers.attn_proj = arena.fp32({kSeqLen, kDModel});
            buffers.current_hidden = arena.fp32({kSeqLen, kDModel});
            buffers.normalized = arena.fp32({kSeqLen, kDModel});
            fill(static_cast<FP32Tensor *>(buffers.attn_proj), 0.0f);
            fill(static_cast<FP32Tensor *>(buffers.current_hidden), 0.0f);
            fill(static_cast<FP32Tensor *>(buffers.normalized), 0.0f);
            buffers.extensions[BufferId::MOE_EXPERT_INDICES] = arena.fp32({kSeqLen, kTopK});
            buffers.extensions[BufferId::MOE_EXPERT_WEIGHTS] = arena.fp32({kSeqLen, kTopK});
            buffers.extensions[BufferId::MOE_COMBINED_OUTPUT] = arena.fp32({kSeqLen, kDModel});
            buffers.extensions[BufferId::MOE_SHARED_EXPERT_OUTPUT] = arena.fp32({kSeqLen, kDModel});
            buffers.extensions[BufferId::MOE_GATE_SCRATCH] = arena.fp32({kSeqLen, kIntermediate});
            buffers.extensions[BufferId::MOE_UP_SCRATCH] = arena.fp32({kSeqLen, kIntermediate});
            return buffers;
        }

        size_t countStagesOfType(const ComputeGraph &graph, ComputeStageType type)
        {
            size_t count = 0;
            for (const auto &node_name : graph.getExecutionOrder())
            {
                const auto *node = graph.getNode(node_name);
                if (node && node->stage->type() == type)
                    ++count;
            }
            return count;
        }

    } // namespace

    TEST(Test__MoEGraphNative_ProductionConfigSmoke, FactoryAcceptsTieredOverlayWithoutLegacyRuntime)
    {
        auto model_ctx = makeModelContext();
        auto plan = makeProductionPlan();
        auto runner_mpi_ctx = std::make_shared<MockMPIContext>(0, 1);

        MockLocalTPContext continuation_tp_context;
        continuation_tp_context.setDevices({GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)});
        continuation_tp_context.setWeights({0.5f, 0.5f});
        continuation_tp_context.setBackend(CollectiveBackendType::HOST);

        InferenceRunnerConfig runner_config;
        runner_config.moe_expert_parallel_plan = plan;
        runner_config.moe_expert_overlay_mpi_ctx = runner_mpi_ctx;
        runner_config.tp_ctx = &continuation_tp_context;

        GraphConfig graph_config;
        configureGraphDimensions(graph_config);
        DomainTPContextMap owned_domain_tp_contexts;
        ASSERT_TRUE(applyMoEExpertOverlayConfigToGraphForTesting(
            *model_ctx,
            runner_config,
            runner_mpi_ctx,
            graph_config,
            owned_domain_tp_contexts,
            "[MoEGraphNativeSmoke]"));

        ASSERT_NE(graph_config.moe.expert_parallel_plan, nullptr);
        EXPECT_EQ(graph_config.moe.expert_parallel_plan.get(), plan.get());
        EXPECT_EQ(graph_config.moe.overlay_mpi_ctx, runner_mpi_ctx);
        EXPECT_EQ(graph_config.moe.expert_overlay_runtime_plan, nullptr);
        EXPECT_EQ(graph_config.moe.expert_overlay_execution_plan, nullptr);
        ASSERT_NE(graph_config.domain_tp_contexts.find("dense_cont"), graph_config.domain_tp_contexts.end());
        EXPECT_EQ(graph_config.domain_tp_contexts.at("dense_cont"), &continuation_tp_context);

        TensorArena weight_arena;
        auto layer = makeLayerWeights(weight_arena);
        TensorArena activation_arena;
        auto buffers = makeActivationBuffers(activation_arena);

        Qwen35MoEGraph graph_builder(graph_config, nullptr);
        ComputeGraph graph = graph_builder.buildFFNGraph(layer, buffers, 0, kSeqLen, kBatchSize, DeviceId::cpu());

        EXPECT_GT(countStagesOfType(graph, ComputeStageType::MOE_SPARSE_DISPATCH), 0u);
        EXPECT_GT(countStagesOfType(graph, ComputeStageType::MOE_SPARSE_RETURN_REDUCE), 0u);
        EXPECT_EQ(countStagesOfType(graph, ComputeStageType::MOE_EXPERT_FFN), 0u);
    }

} // namespace llaminar2::test