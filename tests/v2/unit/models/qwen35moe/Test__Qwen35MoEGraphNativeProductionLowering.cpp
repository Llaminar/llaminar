/**
 * @file Test__Qwen35MoEGraphNativeProductionLowering.cpp
 * @brief Production default graph-native lowering checks for Qwen3.5 MoE overlay tiers.
 */

#include <gtest/gtest.h>

#include "execution/compute_stages/stages/MoEExpertDispatchStage.h"
#include "execution/compute_stages/stages/MoELocalExpertStage.h"
#include "execution/compute_stages/stages/MoESparseDispatchStage.h"
#include "execution/compute_stages/stages/MoESparseReturnReduceStage.h"
#include "execution/moe/MoEExpertOwnerMap.h"
#include "models/qwen35moe/Qwen35MoEGraph.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"

#include <algorithm>
#include <cstring>
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
        constexpr int kSeqLen = 3;
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

        ExpertComputeDomain domain(const std::string &name, GlobalDeviceAddress participant)
        {
            ExpertComputeDomain result;
            result.name = name;
            result.kind = ExpertDomainKind::SingleDevice;
            result.backend = CollectiveBackendType::HOST;
            result.participants = {std::move(participant)};
            result.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
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

        EXPECT_NE(graph.getNode("layer0_moe_expert_dispatch"), nullptr);
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
    }

} // namespace llaminar2::test