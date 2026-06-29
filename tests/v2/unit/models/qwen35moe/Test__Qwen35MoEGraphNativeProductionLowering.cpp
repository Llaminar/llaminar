/**
 * @file Test__Qwen35MoEGraphNativeProductionLowering.cpp
 * @brief Production default graph-native lowering checks for Qwen3.5 MoE overlay tiers.
 */

#include <gtest/gtest.h>

#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/compute_stages/stages/MoEExpertDispatchStage.h"
#include "execution/compute_stages/stages/MoELocalExpertStage.h"
#include "execution/compute_stages/stages/MoESparseDispatchStage.h"
#include "execution/compute_stages/stages/MoESparseReturnReduceStage.h"
#include "execution/moe/MoEExpertOwnerMap.h"
#include "loaders/ExpertGemmRegistry.h"
#include "loaders/ModelContext.h"
#include "models/qwen35moe/Qwen35MoEGraph.h"
#include "mocks/MockLocalTPContext.h"
#include "tensors/TensorKernels.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
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
        constexpr int kSeqLen = 3;
        constexpr int kBatchSize = 1;

        class TestExpertGemm : public ITensorGemm
        {
        public:
            explicit TestExpertGemm(int tag) : tag_(tag) {}

            bool supports_device(int /*device_idx*/) const override { return true; }

            bool multiply_tensor(
                const TensorBase * /*A*/,
                TensorBase * /*C*/,
                int /*m*/,
                int /*n*/,
                int /*k*/,
                bool /*transpose_B*/,
                float /*alpha*/,
                float /*beta*/,
                const TensorBase * /*bias*/,
                const IMPIContext * /*mpi_ctx*/,
                int /*device_idx*/,
                DeviceWorkspaceManager * /*workspace*/,
                int /*activation_row_offset*/) override
            {
                return false;
            }

        private:
            int tag_ = 0;
        };

        using ExpertRole = ExpertGemmRegistry::WeightRole;

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

        ExpertComputeDomain domain(const std::string &name, GlobalDeviceAddress participant)
        {
            ExpertComputeDomain result;
            result.name = name;
            result.kind = ExpertDomainKind::SingleDevice;
            result.backend = CollectiveBackendType::HOST;
            result.participants = {std::move(participant)};
            result.compute_kind = ExpertDomainComputeKind::ApportionedExperts;
            result.owner_rank = 0;
            return result;
        }

        ExpertComputeDomain localTPDomain(
            const std::string &name,
            std::vector<GlobalDeviceAddress> participants)
        {
            ExpertComputeDomain result;
            result.name = name;
            result.kind = ExpertDomainKind::LocalTP;
            result.backend = CollectiveBackendType::RCCL;
            result.participants = std::move(participants);
            result.compute_kind = ExpertDomainComputeKind::ApportionedExperts;
            result.owner_rank = 0;
            return result;
        }

        ExpertRoutedTier tier(const std::string &name, const std::string &domain_name, int priority, bool fallback = false)
        {
            ExpertRoutedTier result;
            result.name = name;
            result.domain = domain_name;
            result.priority = priority;
            result.fallback = fallback;
            return result;
        }

        std::shared_ptr<MoEExpertParallelPlan> makeProductionStylePlan()
        {
            auto plan = std::make_shared<MoEExpertParallelPlan>();
            plan->enabled = true;
            plan->execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
            plan->continuation_domain = "continuation";
            plan->base_model_domain = "continuation";
            plan->shared_expert_domain = "continuation";
            plan->continuation_domain_spec.domain = "continuation";
            plan->continuation_domain_spec.logical_root_participant = 0;
            plan->residency_policy = ExpertResidencyPolicy::ExplicitMasks;
            plan->domains = {
                domain("continuation", GlobalDeviceAddress::cpu(0)),
                domain("hot_domain", GlobalDeviceAddress::cpu(0)),
                domain("warm_domain", GlobalDeviceAddress::cpu(1)),
                domain("cold_domain", GlobalDeviceAddress::cpu(2)),
            };
            plan->routed_tiers = {
                tier("hot", "hot_domain", 0),
                tier("warm", "warm_domain", 1),
                tier("cold", "cold_domain", 99, true),
            };
            plan->placements.push_back(ExpertLayerPlacement{
                .layer = 0,
                .routed_expert_tier = {0, 1, 2, 0, 1, 2},
            });
            validateMoEExpertParallelPlanOrThrow(
                *plan,
                {.layer_count = 1, .routed_expert_count = kNumExperts});
            return plan;
        }

        std::shared_ptr<MoEExpertParallelPlan> makeLocalTPApportionedHotPlan()
        {
            auto plan = std::make_shared<MoEExpertParallelPlan>();
            plan->enabled = true;
            plan->execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
            plan->continuation_domain = "hot_domain";
            plan->base_model_domain = "hot_domain";
            plan->shared_expert_domain = "hot_domain";
            plan->continuation_domain_spec.domain = "hot_domain";
            plan->continuation_domain_spec.logical_root_participant = 0;
            plan->residency_policy = ExpertResidencyPolicy::ExplicitMasks;
            plan->domains = {
                localTPDomain(
                    "hot_domain",
                    {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)}),
            };
            plan->routed_tiers = {
                tier("hot", "hot_domain", 0),
            };
            plan->placements.push_back(ExpertLayerPlacement{
                .layer = 0,
                .routed_expert_tier = {0, 0, 0, 0, 0, 0},
            });
            validateMoEExpertParallelPlanOrThrow(
                *plan,
                {.layer_count = 1, .routed_expert_count = kNumExperts});
            return plan;
        }

        GraphConfig makeConfig(std::shared_ptr<MoEExpertParallelPlan> plan)
        {
            GraphConfig config;
            config.n_layers = 2;
            config.total_n_layers = 2;
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
            config.moe.expert_parallel_plan = std::move(plan);
            return config;
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

        void registerDomainExpertEngine(
            ExpertGemmRegistry &registry,
            const std::string &domain_name,
            DeviceId device,
            int layer_idx,
            int expert,
            ExpertRole role,
            int tag)
        {
            auto engine = std::make_shared<TestExpertGemm>(tag);
            registry.registerEngineForDomain(
                domain_name,
                device,
                layer_idx,
                expert,
                role,
                engine.get(),
                engine);
        }

        void registerCompleteDomainExpertLayer(
            ExpertGemmRegistry &registry,
            const std::string &domain_name,
            DeviceId device,
            int layer_idx)
        {
            for (int expert = 0; expert < kNumExperts; ++expert)
            {
                const int tag_base = 1000 * device.ordinal + 10 * expert;
                registerDomainExpertEngine(
                    registry, domain_name, device, layer_idx, expert, ExpertRole::GATE, tag_base + 1);
                registerDomainExpertEngine(
                    registry, domain_name, device, layer_idx, expert, ExpertRole::UP, tag_base + 2);
                registerDomainExpertEngine(
                    registry, domain_name, device, layer_idx, expert, ExpertRole::DOWN, tag_base + 3);
            }
        }

        std::shared_ptr<ModelContext> makeTestingModelContextWithHotDomainExperts()
        {
            auto model_ctx = ModelContext::createForTesting(
                "test.gguf",
                nullptr,
                1,
                /*with_weight_manager=*/true);
            if (!model_ctx || !model_ctx->concreteWeightManager())
                throw std::runtime_error("ModelContext test WeightManager was not created");

            auto &registry = model_ctx->concreteWeightManager()->expertGemmRegistry();
            registerCompleteDomainExpertLayer(registry, "hot_domain", DeviceId::rocm(0), 0);
            registerCompleteDomainExpertLayer(registry, "hot_domain", DeviceId::rocm(1), 0);
            return model_ctx;
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

        std::vector<std::string> stageNamesOfType(const ComputeGraph &graph, ComputeStageType type)
        {
            std::vector<std::string> names;
            for (const auto &node_name : graph.getExecutionOrder())
            {
                const auto *node = graph.getNode(node_name);
                if (node && node->stage->type() == type)
                    names.push_back(node_name);
            }
            return names;
        }

        const MoESparseDispatchStage *firstSparseDispatchStage(const ComputeGraph &graph)
        {
            for (const auto &node_name : graph.getExecutionOrder())
            {
                const auto *node = graph.getNode(node_name);
                if (!node)
                    continue;
                if (const auto *stage = dynamic_cast<const MoESparseDispatchStage *>(node->stage.get()))
                    return stage;
            }
            return nullptr;
        }

        std::vector<const MoESparseDispatchStage *> sparseDispatchStages(const ComputeGraph &graph)
        {
            std::vector<const MoESparseDispatchStage *> stages;
            for (const auto &node_name : graph.getExecutionOrder())
            {
                const auto *node = graph.getNode(node_name);
                if (!node)
                    continue;
                if (const auto *stage = dynamic_cast<const MoESparseDispatchStage *>(node->stage.get()))
                    stages.push_back(stage);
            }
            return stages;
        }

        std::vector<const MoELocalExpertStage *> localExpertStages(const ComputeGraph &graph)
        {
            std::vector<const MoELocalExpertStage *> stages;
            for (const auto &node_name : graph.getExecutionOrder())
            {
                const auto *node = graph.getNode(node_name);
                if (!node)
                    continue;
                if (const auto *stage = dynamic_cast<const MoELocalExpertStage *>(node->stage.get()))
                    stages.push_back(stage);
            }
            return stages;
        }

        const MoEExpertComputeStage *expertComputeStage(const ComputeGraph &graph, const std::string &node_name)
        {
            const auto *node = graph.getNode(node_name);
            if (!node)
                return nullptr;
            return dynamic_cast<const MoEExpertComputeStage *>(node->stage.get());
        }

    } // namespace

    TEST(Test__Qwen35MoEGraphNativeProductionLowering, TieredOverlayDefaultsToGraphNativeSparseStages)
    {
        auto plan = makeProductionStylePlan();
        const auto owner_map = MoEExpertOwnerMap::build(*plan);
        GraphConfig config = makeConfig(plan);
        ASSERT_EQ(config.moe.expert_overlay_runtime_plan, nullptr);

        TensorArena weight_arena;
        auto layer = makeLayerWeights(weight_arena);
        TensorArena activation_arena;
        auto buffers = makeActivationBuffers(activation_arena);

        Qwen35MoEGraph graph_builder(config, nullptr);
        ComputeGraph graph = graph_builder.buildFFNGraph(layer, buffers, 0, kSeqLen, kBatchSize, DeviceId::cpu());

        const auto *dispatch_node = graph.getNode("layer0_moe_expert_dispatch");
        ASSERT_NE(dispatch_node, nullptr);
        const auto *dispatch_stage =
            dynamic_cast<const MoEExpertDispatchStage *>(dispatch_node->stage.get());
        ASSERT_NE(dispatch_stage, nullptr);
        EXPECT_EQ(dispatch_stage->params().routing_indices_buffer_id, BufferId::MOE_EXPERT_INDICES);
        EXPECT_EQ(dispatch_stage->params().routing_weights_buffer_id, BufferId::MOE_EXPERT_WEIGHTS);
        EXPECT_EQ(dispatch_stage->params().hidden_buffer_id, BufferId::NORMALIZED);
        EXPECT_EQ(countStagesOfType(graph, ComputeStageType::MOE_EXPERT_DISPATCH), 1u);
        EXPECT_GT(countStagesOfType(graph, ComputeStageType::MOE_SPARSE_DISPATCH), 0u);
        EXPECT_GT(countStagesOfType(graph, ComputeStageType::MOE_LOCAL_EXPERT), 0u);
        EXPECT_GT(countStagesOfType(graph, ComputeStageType::MOE_SPARSE_RETURN_REDUCE), 0u);
        EXPECT_EQ(countStagesOfType(graph, ComputeStageType::MOE_EXPERT_FFN), 0u);

        const auto *sparse_dispatch = firstSparseDispatchStage(graph);
        ASSERT_NE(sparse_dispatch, nullptr);
        const auto &params = sparse_dispatch->params();
        EXPECT_TRUE(params.key.isValid());
        EXPECT_NE(params.workspace, nullptr);
        EXPECT_NE(params.collective_context, nullptr);
        EXPECT_GE(params.source_participant, 0);
        EXPECT_GE(params.target_participant, 0);

        const auto participant_ids = owner_map.participantIdsForTier(params.key.tier_idx);
        EXPECT_NE(std::find(participant_ids.begin(), participant_ids.end(), params.target_participant),
                  participant_ids.end());

        bool found_root_sparse_dispatch = false;
        for (const auto *stage : sparseDispatchStages(graph))
        {
            const auto &sparse_params = stage->params();
            if (!sparse_params.hidden)
                continue;
            found_root_sparse_dispatch = true;
            EXPECT_EQ(sparse_params.hidden_buffer_id, BufferId::NORMALIZED);
            EXPECT_EQ(sparse_params.routing_indices_buffer_id, BufferId::MOE_EXPERT_INDICES);
            EXPECT_EQ(sparse_params.routing_weights_buffer_id, BufferId::MOE_EXPERT_WEIGHTS);
        }
        EXPECT_TRUE(found_root_sparse_dispatch);
    }

    TEST(Test__Qwen35MoEGraphNativeProductionLowering,
         LocalTPApportionedExpertsGpuPrefillUsesCapturableFastPathByDefault)
    {
        GraphConfig config = makeConfig(makeLocalTPApportionedHotPlan());
        config.default_device = DeviceId::rocm(0);
        MockLocalTPContext tp_ctx;
        tp_ctx.setDevices({GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});
        tp_ctx.setBackend(CollectiveBackendType::RCCL);
        config.tp_ctx = &tp_ctx;
        config.tp_device_idx = 0;

        TensorArena weight_arena;
        auto layer = makeLayerWeights(weight_arena);
        TensorArena activation_arena;
        auto buffers = makeActivationBuffers(activation_arena);

        auto model_ctx = makeTestingModelContextWithHotDomainExperts();
        Qwen35MoEGraph graph_builder(model_ctx, nullptr, config);
        ComputeGraph graph = graph_builder.buildFFNGraph(layer, buffers, 0, kSeqLen, kBatchSize, DeviceId::rocm(0));

        EXPECT_EQ(countStagesOfType(graph, ComputeStageType::MOE_EXPERT_DISPATCH), 0u)
            << "Homogeneous LocalTP GPU prefill must not lower through the host dispatch descriptor path";
        EXPECT_EQ(countStagesOfType(graph, ComputeStageType::MOE_SPARSE_DISPATCH), 0u)
            << "Homogeneous LocalTP GPU prefill must use the fixed-topology grouped prefill path";
        EXPECT_EQ(countStagesOfType(graph, ComputeStageType::MOE_SPARSE_RETURN_REDUCE), 0u);
        EXPECT_EQ(countStagesOfType(graph, ComputeStageType::MOE_LOCAL_EXPERT), 0u);
        EXPECT_EQ(countStagesOfType(graph, ComputeStageType::MOE_EXPERT_FFN), 1u);

        const auto *expert_node = graph.getNode("layer0_moe_expert_ffn_overlay_fast");
        ASSERT_NE(expert_node, nullptr);
        EXPECT_EQ(expert_node->device, DeviceId::rocm(0));
        const auto *expert_stage = dynamic_cast<const MoEExpertComputeStage *>(expert_node->stage.get());
        ASSERT_NE(expert_stage, nullptr);
        EXPECT_EQ(expert_stage->fixedTopologyPrefillExpertIdsForTesting(),
                  (std::vector<int>{0, 1, 2}));
        EXPECT_TRUE(expert_node->stage->supportsWarmupDependentGraphCapture())
            << "Cold preflight must allow warmup to build MoE grouped-prefill capture resources";
        EXPECT_TRUE(expert_node->stage->supportsLazyPrefillGraphCapturePreflight())
            << "The fixed-topology grouped prefill path is the graph-capturable MoE dispatch contract";

        EXPECT_NE(graph.getNode("layer0_moe_expert_overlay_fast_allreduce"), nullptr)
            << "Graph-local owner subsets must be rejoined through the continuation TP domain";
        EXPECT_EQ(countStagesOfType(graph, ComputeStageType::ALLREDUCE), 1u);
    }

    TEST(Test__Qwen35MoEGraphNativeProductionLowering,
         LocalTPApportionedExpertAllreduceNameIsStableAcrossParticipants)
    {
        auto plan = makeLocalTPApportionedHotPlan();
        MockLocalTPContext tp_ctx;
        tp_ctx.setDevices({GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});
        tp_ctx.setBackend(CollectiveBackendType::RCCL);

        GraphConfig config0 = makeConfig(plan);
        config0.default_device = DeviceId::rocm(0);
        config0.tp_ctx = &tp_ctx;
        config0.tp_device_idx = 0;

        GraphConfig config1 = makeConfig(plan);
        config1.default_device = DeviceId::rocm(1);
        config1.tp_ctx = &tp_ctx;
        config1.tp_device_idx = 1;

        TensorArena weight_arena;
        auto layer = makeLayerWeights(weight_arena);

        TensorArena activation_arena0;
        auto buffers0 = makeActivationBuffers(activation_arena0);
        TensorArena activation_arena1;
        auto buffers1 = makeActivationBuffers(activation_arena1);

        auto model_ctx = makeTestingModelContextWithHotDomainExperts();

        Qwen35MoEGraph graph_builder0(model_ctx, nullptr, config0);
        ComputeGraph graph0 = graph_builder0.buildFFNGraph(
            layer, buffers0, 0, kSeqLen, kBatchSize, DeviceId::rocm(0));

        Qwen35MoEGraph graph_builder1(model_ctx, nullptr, config1);
        ComputeGraph graph1 = graph_builder1.buildFFNGraph(
            layer, buffers1, 0, kSeqLen, kBatchSize, DeviceId::rocm(1));

        const auto allreduces0 = stageNamesOfType(graph0, ComputeStageType::ALLREDUCE);
        const auto allreduces1 = stageNamesOfType(graph1, ComputeStageType::ALLREDUCE);
        ASSERT_EQ(allreduces0.size(), 1u);
        ASSERT_EQ(allreduces1.size(), 1u);
        EXPECT_EQ(allreduces0, allreduces1)
            << "LocalTP grouped collectives require every participant graph to enter the same stage name";
        EXPECT_EQ(allreduces0.front(), "layer0_moe_expert_overlay_fast_allreduce");

        const auto *expert_stage0 = expertComputeStage(graph0, "layer0_moe_expert_ffn_overlay_fast");
        const auto *expert_stage1 = expertComputeStage(graph1, "layer0_moe_expert_ffn_overlay_fast");
        ASSERT_NE(expert_stage0, nullptr);
        ASSERT_NE(expert_stage1, nullptr);
        EXPECT_EQ(expert_stage0->fixedTopologyPrefillExpertIdsForTesting(),
                  (std::vector<int>{0, 1, 2}));
        EXPECT_EQ(expert_stage1->fixedTopologyPrefillExpertIdsForTesting(),
                  (std::vector<int>{3, 4, 5}));
    }

} // namespace llaminar2::test
