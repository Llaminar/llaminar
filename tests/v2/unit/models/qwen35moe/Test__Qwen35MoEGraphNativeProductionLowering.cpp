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
#include <cstdlib>
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
        constexpr int kDModel = 32;
        constexpr int kIntermediate = 32;
        constexpr int kNumExperts = 6;
        constexpr int kTopK = 2;
        constexpr int kSeqLen = 3;
        constexpr int kBatchSize = 1;

        class ScopedDebugEnv
        {
        public:
            explicit ScopedDebugEnv(std::initializer_list<std::pair<const char *, const char *>> values)
            {
                for (const auto &[name, value] : values)
                {
                    Entry entry;
                    entry.name = name;
                    if (const char *old_value = std::getenv(name))
                    {
                        entry.had_value = true;
                        entry.old_value = old_value;
                    }
                    entries_.push_back(entry);
                    ::setenv(name, value, 1);
                }
                mutableDebugEnv().reload();
            }

            ~ScopedDebugEnv()
            {
                for (const auto &entry : entries_)
                {
                    if (entry.had_value)
                        ::setenv(entry.name.c_str(), entry.old_value.c_str(), 1);
                    else
                        ::unsetenv(entry.name.c_str());
                }
                mutableDebugEnv().reload();
            }

            ScopedDebugEnv(const ScopedDebugEnv &) = delete;
            ScopedDebugEnv &operator=(const ScopedDebugEnv &) = delete;

        private:
            struct Entry
            {
                std::string name;
                bool had_value = false;
                std::string old_value;
            };

            std::vector<Entry> entries_;
        };

        class TestExpertGemm : public ITensorGemm
        {
        public:
            TestExpertGemm(int tag, ExpertGemmRegistry::WeightRole role)
                : tag_(tag), role_(role)
            {
                payload_[0] = static_cast<uint8_t>(tag_ & 0xff);
                scale_[0] = 1.0f;
            }

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

            bool exportNativeVNNIMatrixDesc(DeviceNativeVNNIMatrixDesc &out) override
            {
                out = {};
                out.payload = payload_;
                out.scales = scale_;
                out.blocks_per_row = 1;
                out.codebook_id = 4;
                if (role_ == ExpertGemmRegistry::WeightRole::DOWN)
                {
                    out.n = kDModel;
                    out.k = kIntermediate;
                }
                else
                {
                    out.n = kIntermediate;
                    out.k = kDModel;
                }
                return true;
            }

        private:
            int tag_ = 0;
            ExpertGemmRegistry::WeightRole role_ = ExpertGemmRegistry::WeightRole::GATE;
            uint8_t payload_[16] = {};
            float scale_[1] = {};
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

        RoutedExpertDomain domain(const std::string &name, GlobalDeviceAddress participant)
        {
            RoutedExpertDomain result;
            result.name = name;
            result.scope = ExecutionDomainScope::SINGLE;
            result.backend = CollectiveBackendType::HOST;
            result.participants = {std::move(participant)};
            result.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
            result.owner_rank = 0;
            return result;
        }

        RoutedExpertDomain localTPDomain(
            const std::string &name,
            std::vector<GlobalDeviceAddress> participants)
        {
            RoutedExpertDomain result;
            result.name = name;
            result.scope = ExecutionDomainScope::LOCAL;
            result.backend = CollectiveBackendType::RCCL;
            result.participants = std::move(participants);
            result.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
            result.owner_rank = 0;
            return result;
        }

        RoutedExpertTier tier(const std::string &name, const std::string &domain_name, int priority, bool fallback = false)
        {
            RoutedExpertTier result;
            result.name = name;
            result.domain = domain_name;
            result.priority = priority;
            result.fallback = fallback;
            return result;
        }

        std::shared_ptr<MoERoutedExpertPlacementPlan> makeProductionStylePlan()
        {
            auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
            plan->enabled = true;
            plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan->continuation_domain = "continuation";
            plan->base_model_domain = "continuation";
            plan->shared_expert_domain = "continuation";
            plan->continuation_domain_spec.domain = "continuation";
            plan->continuation_domain_spec.logical_root_participant = 0;
            plan->residency_policy = RoutedExpertResidencyPolicy::ExplicitMasks;
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
            plan->placements.push_back(RoutedExpertLayerPlacement{
                .layer = 0,
                .routed_expert_tier = {0, 1, 2, 0, 1, 2},
            });
            validateMoERoutedExpertPlacementPlanOrThrow(
                *plan,
                {.layer_count = 1, .routed_expert_count = kNumExperts});
            return plan;
        }

        std::shared_ptr<MoERoutedExpertPlacementPlan> makeLocalTPApportionedHotPlan(int layer_count = 1)
        {
            auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
            plan->enabled = true;
            plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan->continuation_domain = "hot_domain";
            plan->base_model_domain = "hot_domain";
            plan->shared_expert_domain = "hot_domain";
            plan->continuation_domain_spec.domain = "hot_domain";
            plan->continuation_domain_spec.logical_root_participant = 0;
            plan->residency_policy = RoutedExpertResidencyPolicy::ExplicitMasks;
            plan->domains = {
                localTPDomain(
                    "hot_domain",
                    {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)}),
            };
            plan->routed_tiers = {
                tier("hot", "hot_domain", 0),
            };
            for (int layer = 0; layer < layer_count; ++layer)
            {
                plan->placements.push_back(RoutedExpertLayerPlacement{
                    .layer = layer,
                    .routed_expert_tier = {0, 0, 0, 0, 0, 0},
                });
            }
            validateMoERoutedExpertPlacementPlanOrThrow(
                *plan,
                {.layer_count = layer_count, .routed_expert_count = kNumExperts});
            return plan;
        }

        GraphConfig makeConfig(
            std::shared_ptr<MoERoutedExpertPlacementPlan> plan,
            int layer_count = 2)
        {
            GraphConfig config;
            config.n_layers = layer_count;
            config.total_n_layers = layer_count;
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
            config.moe.routed_expert_plan = std::move(plan);
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
            auto engine = std::make_shared<TestExpertGemm>(tag, role);
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

        void registerCompleteDeviceExpertLayer(
            ExpertGemmRegistry &registry,
            DeviceId device,
            int layer_idx)
        {
            for (int expert = 0; expert < kNumExperts; ++expert)
            {
                for (const ExpertRole role : {
                         ExpertRole::GATE,
                         ExpertRole::UP,
                         ExpertRole::DOWN,
                     })
                {
                    const int tag =
                        100 * expert + static_cast<int>(role) + 1;
                    auto engine = std::make_shared<TestExpertGemm>(tag, role);
                    registry.registerEngine(
                        device,
                        layer_idx,
                        expert,
                        role,
                        engine.get(),
                        engine);
                }
            }
        }

        std::shared_ptr<ModelContext> makeTestingModelContextWithHotDomainExperts(int layer_count = 1)
        {
            auto model_ctx = ModelContext::createForTesting(
                "test.gguf",
                nullptr,
                static_cast<uint32_t>(std::max(1, layer_count)),
                /*with_weight_manager=*/true);
            if (!model_ctx || !model_ctx->concreteWeightManager())
                throw std::runtime_error("ModelContext test WeightManager was not created");

            auto &registry = model_ctx->concreteWeightManager()->expertGemmRegistry();
            for (int layer = 0; layer < std::max(1, layer_count); ++layer)
            {
                registerCompleteDomainExpertLayer(registry, "hot_domain", DeviceId::rocm(0), layer);
                registerCompleteDomainExpertLayer(registry, "hot_domain", DeviceId::rocm(1), layer);
            }
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

        bool hasDependency(
            const ComputeGraph &graph,
            const std::string &node_name,
            const std::string &dependency)
        {
            const auto *node = graph.getNode(node_name);
            if (!node)
                return false;
            return std::find(
                       node->dependencies.begin(),
                       node->dependencies.end(),
                       dependency) != node->dependencies.end();
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
         LocalTPApportionedRoutedComputeGpuPrefillUsesCapturableFastPathByDefault)
    {
        GraphConfig config = makeConfig(makeLocalTPApportionedHotPlan());
        config.default_device = DeviceId::rocm(0);
        MockLocalTPContext tp_ctx;
        tp_ctx.setDevices({GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});
        tp_ctx.setBackend(CollectiveBackendType::RCCL);
        tp_ctx.setRawAllgatherGraphCaptureSupported(true);
        config.tp_ctx = &tp_ctx;
        config.tp_device_idx = 0;

        TensorArena weight_arena;
        auto layer = makeLayerWeights(weight_arena);
        TensorArena activation_arena;
        auto buffers = makeActivationBuffers(activation_arena);

        auto model_ctx =
            makeTestingModelContextWithHotDomainExperts(config.n_layers);
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

    /**
     * @brief GPU shared-verifier nodes wait for their router-owned Q8 input rows.
     *
     * The production grouped router publishes a device-resident Q8 copy of each
     * normalized verifier row.  Both the routed experts and the always-active
     * shared expert reuse that publication.  NORMALIZED alone therefore does not
     * describe the shared stage's complete dependency set: without an edge from
     * `moe_routing`, graph topological ordering may run the shared sibling first
     * and consume a stale publication left by an earlier layer.
     *
     * This fixture supplies the same prepared expert-engine registry required by
     * production GPU lowering, but it deliberately creates no device context,
     * allocates no GPU memory, and executes no GPU work.  Unit coverage therefore
     * proves the CUDA and ROCm graph contract without violating the integration-
     * only rule for GPU execution.
     */
    TEST(Test__Qwen35MoEGraphNativeProductionLowering,
         GPUGroupedSharedVerifierDependsOnRouterQ8Producer)
    {
        auto model_ctx = ModelContext::createForTesting(
            "test.gguf",
            nullptr,
            /*layer_count=*/1,
            /*with_weight_manager=*/true);
        ASSERT_NE(model_ctx, nullptr);
        ASSERT_NE(model_ctx->concreteWeightManager(), nullptr);

        auto &registry = model_ctx->concreteWeightManager()->expertGemmRegistry();
        for (const DeviceId device : {DeviceId::cuda(0), DeviceId::rocm(0)})
            registerCompleteDeviceExpertLayer(registry, device, /*layer_idx=*/0);

        for (const DeviceId device : {DeviceId::cuda(0), DeviceId::rocm(0)})
        {
            SCOPED_TRACE("device=" + device.to_string());

            GraphConfig config = makeConfig(nullptr);
            config.default_device = device;
            config.compute_all_position_logits = true;
            config.moe.has_shared_expert = true;
            config.moe.shared_intermediate_size = kIntermediate;

            TensorArena weight_arena;
            auto layer = makeLayerWeights(weight_arena);
            layer.shared_expert_gate =
                weight_arena.fp32({kIntermediate, kDModel});
            layer.shared_expert_up =
                weight_arena.fp32({kIntermediate, kDModel});
            layer.shared_expert_down =
                weight_arena.fp32({kDModel, kIntermediate});

            TensorArena activation_arena;
            auto buffers = makeActivationBuffers(activation_arena);

            Qwen35MoEGraph graph_builder(model_ctx, nullptr, config);
            ComputeGraph graph = graph_builder.buildFFNGraph(
                layer,
                buffers,
                /*layer_idx=*/0,
                /*seq_len=*/2,
                kBatchSize,
                device);

            ASSERT_NE(graph.getNode("layer0_moe_routing"), nullptr);
            const auto *shared_node =
                graph.getNode("layer0_shared_expert_ffn");
            ASSERT_NE(shared_node, nullptr);
            const auto *shared_stage =
                dynamic_cast<const SharedExpertFFNStage *>(
                    shared_node->stage.get());
            ASSERT_NE(shared_stage, nullptr);
            EXPECT_TRUE(
                shared_stage->usesGroupedVerifierPrefillRouteForTesting());
            EXPECT_TRUE(hasDependency(
                graph,
                "layer0_shared_expert_ffn",
                "layer0_moe_routing"))
                << "The grouped shared verifier consumes the router's device Q8 publication";
        }
    }

    TEST(Test__Qwen35MoEGraphNativeProductionLowering,
         LocalTPApportionedLeastLoadedTinyPrefillUsesStaticOwnerCostGate)
    {
        ScopedDebugEnv env({
            {"LLAMINAR_MOE_LLEP_PREFILL_TRANSFER_MODE", "full"},
            {"LLAMINAR_MOE_LLEP_PREFILL_MIN_ROUTED_ROWS", "8192"},
        });
        auto plan = makeLocalTPApportionedHotPlan();
        ASSERT_FALSE(plan->domains.empty());
        plan->domains[0].routed_assignment_policy = RoutedExpertAssignmentPolicy::LeastLoadedResident;

        GraphConfig config = makeConfig(plan);
        config.default_device = DeviceId::rocm(0);
        config.moe.routed_assignment_policy = RoutedExpertAssignmentPolicy::LeastLoadedResident;

        MockLocalTPContext tp_ctx;
        tp_ctx.setDevices({GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});
        tp_ctx.setBackend(CollectiveBackendType::RCCL);
        tp_ctx.setRawAllgatherGraphCaptureSupported(true);
        config.tp_ctx = &tp_ctx;
        config.tp_device_idx = 0;

        TensorArena weight_arena;
        auto layer = makeLayerWeights(weight_arena);
        TensorArena activation_arena;
        auto buffers = makeActivationBuffers(activation_arena);

        auto model_ctx =
            makeTestingModelContextWithHotDomainExperts(config.n_layers);
        Qwen35MoEGraph graph_builder(model_ctx, nullptr, config);
        ComputeGraph graph = graph_builder.buildFFNGraph(
            layer, buffers, 0, kSeqLen, kBatchSize, DeviceId::rocm(0));

        const auto *expert_node = graph.getNode("layer0_moe_expert_ffn_overlay_fast");
        ASSERT_NE(expert_node, nullptr);
        const auto *expert_stage = dynamic_cast<const MoEExpertComputeStage *>(expert_node->stage.get());
        ASSERT_NE(expert_stage, nullptr);
        EXPECT_EQ(expert_stage->routedExpertAssignmentPolicyForTesting(),
                  RoutedExpertAssignmentPolicy::StaticOwner)
            << "Tiny prefill below the LLEP routed-row gate must lower as standard "
               "apportioned-expert work instead of paying transfer-backed current-batch movement.";
        EXPECT_FALSE(expert_stage->usesRuntimePrefillGroupingForTesting());
        EXPECT_FALSE(expert_stage->hasPrefillLLEPTPContextForTesting());
        EXPECT_FALSE(expert_stage->hasTransferBackedPrefillLLEPForTesting());
        EXPECT_TRUE(expert_stage->supportsRequestedRoutedAssignmentPolicyForTesting());
    }

    TEST(Test__Qwen35MoEGraphNativeProductionLowering,
         LocalTPApportionedLeastLoadedAssignmentIsStampedOntoFastExpertStage)
    {
        ScopedDebugEnv env({
            {"LLAMINAR_MOE_LLEP_PREFILL_TRANSFER_MODE", "full"},
            {"LLAMINAR_MOE_LLEP_PREFILL_MIN_ROUTED_ROWS", "0"},
        });
        auto plan = makeLocalTPApportionedHotPlan();
        ASSERT_FALSE(plan->domains.empty());
        plan->domains[0].routed_assignment_policy = RoutedExpertAssignmentPolicy::LeastLoadedResident;

        GraphConfig config = makeConfig(plan);
        config.default_device = DeviceId::rocm(0);
        config.moe.routed_assignment_policy = RoutedExpertAssignmentPolicy::LeastLoadedResident;

        MockLocalTPContext tp_ctx;
        tp_ctx.setDevices({GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});
        tp_ctx.setBackend(CollectiveBackendType::RCCL);
        tp_ctx.setRawAllgatherGraphCaptureSupported(true);
        config.tp_ctx = &tp_ctx;
        config.tp_device_idx = 0;

        TensorArena weight_arena;
        auto layer = makeLayerWeights(weight_arena);
        TensorArena activation_arena;
        auto buffers = makeActivationBuffers(activation_arena);

        auto model_ctx =
            makeTestingModelContextWithHotDomainExperts(config.n_layers);
        Qwen35MoEGraph graph_builder(model_ctx, nullptr, config);
        ComputeGraph graph = graph_builder.buildFFNGraph(
            layer, buffers, 0, kSeqLen, kBatchSize, DeviceId::rocm(0));

        const auto *expert_node = graph.getNode("layer0_moe_expert_ffn_overlay_fast");
        ASSERT_NE(expert_node, nullptr);
        const auto *expert_stage = dynamic_cast<const MoEExpertComputeStage *>(expert_node->stage.get());
        ASSERT_NE(expert_stage, nullptr);
        EXPECT_EQ(expert_stage->routedExpertAssignmentPolicyForTesting(),
                  RoutedExpertAssignmentPolicy::LeastLoadedResident)
            << "assignment=least-loaded-resident must reach the production expert stage; otherwise "
               "the graph can silently execute StaticOwner under an LLEP label.";
        EXPECT_TRUE(expert_stage->usesRuntimePrefillGroupingForTesting())
            << "LLEP prefill must use runtime grouping so the device-side "
               "route-participant assignment kernel is reachable in production graphs.";
        EXPECT_TRUE(expert_stage->hasPrefillLLEPTPContextForTesting())
            << "LLEP prefill must carry its LocalTP context so full "
               "current-batch row exchange can use grouped NCCL/RCCL collectives.";
        EXPECT_EQ(
            expert_stage->prefillLLEPAssignmentModeForTesting(),
            PrefillLLEPAssignmentMode::TransferBackedCurrentBatch);
        EXPECT_TRUE(expert_stage->hasTransferBackedPrefillLLEPForTesting())
            << "LLEP prefill must carry compact transfer-slot backing; "
               "otherwise foreign current-batch spans silently collapse back to resident-only routing.";
        EXPECT_TRUE(expert_stage->supportsRequestedRoutedAssignmentPolicyForTesting());
    }

    /**
     * @brief Prove grouped verifier rows cannot inherit long-prefill migration.
     *
     * The production LLEP movement probe deliberately sets the routed-row
     * threshold to zero and requests full compact transport. That environment
     * must still leave MTP verifier rows resident-only: moving an expert payload
     * in every MoE layer for a three-row verifier batch is not an economical
     * grouped implementation. Prefix rehydration has a separate graph-build
     * flag and is therefore unaffected by this current-batch policy assertion.
     */
    TEST(Test__Qwen35MoEGraphNativeProductionLowering,
         LocalTPGroupedVerifierUsesResidentOnlyLLEPAssignment)
    {
        ScopedDebugEnv env({
            {"LLAMINAR_MOE_LLEP_PREFILL_TRANSFER_MODE", "full"},
            {"LLAMINAR_MOE_LLEP_PREFILL_MIN_ROUTED_ROWS", "0"},
        });
        auto plan = makeLocalTPApportionedHotPlan();
        ASSERT_FALSE(plan->domains.empty());
        plan->domains[0].routed_assignment_policy =
            RoutedExpertAssignmentPolicy::LeastLoadedResident;

        GraphConfig config = makeConfig(plan);
        config.default_device = DeviceId::rocm(0);
        config.compute_all_position_logits = true;
        config.moe.routed_assignment_policy =
            RoutedExpertAssignmentPolicy::LeastLoadedResident;

        MockLocalTPContext tp_ctx;
        tp_ctx.setDevices(
            {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});
        tp_ctx.setBackend(CollectiveBackendType::RCCL);
        tp_ctx.setRawAllgatherGraphCaptureSupported(true);
        config.tp_ctx = &tp_ctx;
        config.tp_device_idx = 0;

        TensorArena weight_arena;
        auto layer = makeLayerWeights(weight_arena);
        TensorArena activation_arena;
        auto buffers = makeActivationBuffers(activation_arena);

        auto model_ctx =
            makeTestingModelContextWithHotDomainExperts(config.n_layers);
        Qwen35MoEGraph graph_builder(model_ctx, nullptr, config);
        ComputeGraph graph = graph_builder.buildFFNGraph(
            layer,
            buffers,
            /*layer_idx=*/0,
            /*seq_len=*/3,
            kBatchSize,
            DeviceId::rocm(0));

        const auto *expert_node =
            graph.getNode("layer0_moe_expert_ffn_overlay_fast");
        ASSERT_NE(expert_node, nullptr);
        const auto *expert_stage =
            dynamic_cast<const MoEExpertComputeStage *>(
                expert_node->stage.get());
        ASSERT_NE(expert_stage, nullptr);

        EXPECT_EQ(
            expert_stage->routedExpertAssignmentPolicyForTesting(),
            RoutedExpertAssignmentPolicy::LeastLoadedResident);
        EXPECT_TRUE(expert_stage->usesRuntimePrefillGroupingForTesting());
        EXPECT_TRUE(expert_stage->hasPrefillLLEPTPContextForTesting());
        EXPECT_EQ(
            expert_stage->prefillLLEPAssignmentModeForTesting(),
            PrefillLLEPAssignmentMode::ResidentOnly);
        EXPECT_FALSE(expert_stage->hasTransferBackedPrefillLLEPForTesting())
            << "Grouped verifier rows must never execute current-batch expert "
               "payload transport, even when long-prefill full mode is enabled.";
        EXPECT_TRUE(
            expert_stage->supportsRequestedRoutedAssignmentPolicyForTesting());
    }

    TEST(Test__Qwen35MoEGraphNativeProductionLowering,
         LocalTPApportionedLeastLoadedPrefillTransferWorkspacesUseBoundedRollingLanes)
    {
        ScopedDebugEnv env({
            {"LLAMINAR_MOE_LLEP_PREFILL_TRANSFER_MODE", "full"},
            {"LLAMINAR_MOE_LLEP_PREFILL_MIN_ROUTED_ROWS", "0"},
        });
        constexpr int kLayerCount = 3;
        auto plan = makeLocalTPApportionedHotPlan(kLayerCount);
        ASSERT_FALSE(plan->domains.empty());
        plan->domains[0].routed_assignment_policy = RoutedExpertAssignmentPolicy::LeastLoadedResident;

        GraphConfig config = makeConfig(plan, kLayerCount);
        config.default_device = DeviceId::rocm(0);
        config.moe.routed_assignment_policy = RoutedExpertAssignmentPolicy::LeastLoadedResident;

        MockLocalTPContext tp_ctx;
        tp_ctx.setDevices({GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});
        tp_ctx.setBackend(CollectiveBackendType::RCCL);
        tp_ctx.setRawAllgatherGraphCaptureSupported(true);
        config.tp_ctx = &tp_ctx;
        config.tp_device_idx = 0;

        TensorArena weight_arena;
        auto layer_weights = makeLayerWeights(weight_arena);
        TensorArena activation_arena0;
        auto buffers0 = makeActivationBuffers(activation_arena0);
        TensorArena activation_arena1;
        auto buffers1 = makeActivationBuffers(activation_arena1);
        TensorArena activation_arena2;
        auto buffers2 = makeActivationBuffers(activation_arena2);

        auto model_ctx = makeTestingModelContextWithHotDomainExperts(kLayerCount);
        Qwen35MoEGraph graph_builder(model_ctx, nullptr, config);
        ComputeGraph graph0 = graph_builder.buildFFNGraph(
            layer_weights, buffers0, 0, kSeqLen, kBatchSize, DeviceId::rocm(0));
        ComputeGraph graph1 = graph_builder.buildFFNGraph(
            layer_weights, buffers1, 1, kSeqLen, kBatchSize, DeviceId::rocm(0));
        ComputeGraph graph2 = graph_builder.buildFFNGraph(
            layer_weights, buffers2, 2, kSeqLen, kBatchSize, DeviceId::rocm(0));

        const auto *stage0 = expertComputeStage(graph0, "layer0_moe_expert_ffn_overlay_fast");
        const auto *stage1 = expertComputeStage(graph1, "layer1_moe_expert_ffn_overlay_fast");
        const auto *stage2 = expertComputeStage(graph2, "layer2_moe_expert_ffn_overlay_fast");
        ASSERT_NE(stage0, nullptr);
        ASSERT_NE(stage1, nullptr);
        ASSERT_NE(stage2, nullptr);
        ASSERT_TRUE(stage0->hasTransferBackedPrefillLLEPForTesting());
        ASSERT_TRUE(stage1->hasTransferBackedPrefillLLEPForTesting());
        ASSERT_TRUE(stage2->hasTransferBackedPrefillLLEPForTesting());

        const std::string workspace0 = stage0->prefillLLEPWorkspaceNameForTesting();
        const std::string workspace1 = stage1->prefillLLEPWorkspaceNameForTesting();
        const std::string workspace2 = stage2->prefillLLEPWorkspaceNameForTesting();
        EXPECT_NE(workspace0, workspace1)
            << "Transfer-backed prefill LLEP stages run compute and transfer streams "
               "concurrently; adjacent layers must not share plan/payload "
               "workspace while an earlier layer's transfer can still be in flight.";
        EXPECT_EQ(workspace0, workspace2)
            << "The graph must bound persistent transfer storage instead of "
               "allocating one expert payload arena per model layer.";
        EXPECT_NE(workspace0.find("prefill_lane=0"), std::string::npos);
        EXPECT_NE(workspace1.find("prefill_lane=1"), std::string::npos);
        EXPECT_NE(workspace2.find("prefill_lane=0"), std::string::npos);

        auto publication_buffer_name =
            [](const MoEExpertComputeStage *stage,
               const char *buffer_family) -> std::string
        {
            const auto requirements =
                stage->getWorkspaceRequirements(kSeqLen);
            const auto buffer = std::find_if(
                requirements.buffers.begin(),
                requirements.buffers.end(),
                [buffer_family](const WorkspaceDescriptor &descriptor)
                {
                    return descriptor.name.rfind(buffer_family, 0) == 0;
                });
            return buffer == requirements.buffers.end()
                       ? std::string{}
                       : buffer->name;
        };
        const std::string status0 =
            publication_buffer_name(
                stage0,
                MoEDeviceRebalanceStage::WS_STATUS);
        const std::string status1 =
            publication_buffer_name(
                stage1,
                MoEDeviceRebalanceStage::WS_STATUS);
        const std::string status2 =
            publication_buffer_name(
                stage2,
                MoEDeviceRebalanceStage::WS_STATUS);
        const std::string apply_status0 =
            publication_buffer_name(
                stage0,
                MoEDeviceRebalanceStage::WS_APPLY_STATUS);
        const std::string apply_status1 =
            publication_buffer_name(
                stage1,
                MoEDeviceRebalanceStage::WS_APPLY_STATUS);
        const std::string apply_status2 =
            publication_buffer_name(
                stage2,
                MoEDeviceRebalanceStage::WS_APPLY_STATUS);
        ASSERT_FALSE(status0.empty());
        ASSERT_FALSE(status1.empty());
        ASSERT_FALSE(status2.empty());
        ASSERT_FALSE(apply_status0.empty());
        ASSERT_FALSE(apply_status1.empty());
        ASSERT_FALSE(apply_status2.empty());
        EXPECT_NE(status0, status1);
        EXPECT_NE(status0, status2)
            << "Tiny transfer-status publications are layer-owned even when "
               "their bulk payload lanes roll over.";
        EXPECT_NE(status1, status2);
        EXPECT_NE(apply_status0, apply_status1);
        EXPECT_NE(apply_status0, apply_status2)
            << "Arrival publication must not alias across model layers.";
        EXPECT_NE(apply_status1, apply_status2);

        const auto *transfer_state0 =
            stage0->prefillLLEPTransferStateForTesting();
        const auto *transfer_state1 =
            stage1->prefillLLEPTransferStateForTesting();
        const auto *transfer_state2 =
            stage2->prefillLLEPTransferStateForTesting();
        ASSERT_NE(transfer_state0, nullptr);
        ASSERT_NE(transfer_state1, nullptr);
        ASSERT_NE(transfer_state2, nullptr);
        EXPECT_NE(transfer_state0, transfer_state1)
            << "Adjacent layers need independent rolling-lane stream/event state.";
        EXPECT_NE(transfer_state0, transfer_state2)
            << "Layers may reuse a bounded workspace/stream lane only after the "
               "graph gives each layer a distinct persistent event pair. Event "
               "identity cannot alias across multiple records in one capture.";
        EXPECT_TRUE(stage0->needsGraphLaunchPreparation());
        EXPECT_TRUE(stage1->needsGraphLaunchPreparation());
        EXPECT_TRUE(stage2->needsGraphLaunchPreparation())
            << "Every transfer-backed prefill stage must preflight lane resources "
               "before CUDA/HIP capture begins.";
    }

    TEST(Test__Qwen35MoEGraphNativeProductionLowering,
         LocalTPApportionedLeastLoadedSingleTokenDecodeIsSupported)
    {
        auto plan = makeLocalTPApportionedHotPlan();
        ASSERT_FALSE(plan->domains.empty());
        plan->domains[0].routed_assignment_policy = RoutedExpertAssignmentPolicy::LeastLoadedResident;

        GraphConfig config = makeConfig(plan);
        config.default_device = DeviceId::rocm(0);
        config.moe.routed_assignment_policy = RoutedExpertAssignmentPolicy::LeastLoadedResident;

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
        ComputeGraph graph = graph_builder.buildFFNGraph(
            layer, buffers, 0, 1, kBatchSize, DeviceId::rocm(0));

        const auto *expert_node = graph.getNode("layer0_moe_expert_ffn_overlay_fast");
        ASSERT_NE(expert_node, nullptr);
        const auto *expert_stage = dynamic_cast<const MoEExpertComputeStage *>(expert_node->stage.get());
        ASSERT_NE(expert_stage, nullptr);
        EXPECT_EQ(expert_stage->routedExpertAssignmentPolicyForTesting(),
                  RoutedExpertAssignmentPolicy::LeastLoadedResident);
        EXPECT_FALSE(expert_stage->usesRuntimePrefillGroupingForTesting())
            << "Single-token decode uses the runtime decode table, not the multi-token prefill grouper.";
        EXPECT_TRUE(expert_stage->hasMoERuntimeTableForTesting())
            << "LLEP decode must be given the runtime placement table; without it "
               "production decode fails closed before the device-routed path can run.";
        EXPECT_TRUE(expert_stage->supportsRequestedRoutedAssignmentPolicyForTesting())
            << "LLEP decode must be allowed to enter the existing device-routed "
               "runtime table path instead of failing before executeSingleToken().";
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
