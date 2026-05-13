/**
 * @file Test__Qwen35MoEExpertOverlayGraph.cpp
 * @brief Model-light tests for Qwen3.5 MoE expert overlay graph construction.
 */

#include <gtest/gtest.h>

#include "execution/compute_stages/stages/MoEExpertDispatchStage.h"
#include "execution/compute_stages/stages/MoEExpertOverlayCPUFallbackStage.h"
#include "execution/compute_stages/stages/MoEExpertOverlayLocalTPStage.h"
#include "execution/compute_stages/stages/MoEExpertParallelReduceStage.h"
#include "execution/compute_stages/stages/MoEOverlayDomainRuntimeStage.h"
#include "execution/moe/MoEExpertParallelPlan.h"
#include "execution/moe/MoEExpertOverlayRuntimePlan.h"
#include "mocks/MockComputeStage.h"
#include "mocks/MockLocalTPContext.h"
#include "models/qwen35moe/Qwen35MoEGraph.h"
#include "tensors/FP16Utils.h"
#include "tensors/TensorKernels.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <random>
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

    class EnvVarGuard
    {
    public:
        EnvVarGuard(std::string name, std::string value)
            : name_(std::move(name))
        {
            const char *previous = std::getenv(name_.c_str());
            if (previous)
            {
                had_previous_ = true;
                previous_ = previous;
            }
            setenv(name_.c_str(), value.c_str(), 1);
            mutableDebugEnv().reload();
        }

        ~EnvVarGuard()
        {
            if (had_previous_)
                setenv(name_.c_str(), previous_.c_str(), 1);
            else
                unsetenv(name_.c_str());
            mutableDebugEnv().reload();
        }

    private:
        std::string name_;
        std::string previous_;
        bool had_previous_ = false;
    };

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

        TensorBase *keep(std::shared_ptr<TensorBase> tensor)
        {
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

    GraphConfig makeConfig(
        std::shared_ptr<MoEExpertParallelPlan> plan = nullptr,
        std::map<std::string, ILocalTPContext *> domain_tp_contexts = {})
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
        config.moe.has_shared_expert = false;
        config.moe.shared_intermediate_size = 0;
        config.moe.expert_parallel_plan = std::move(plan);
        config.moe.expert_overlay_runtime_plan = resolveMoEExpertOverlayRuntimePlan(
            config.moe.expert_parallel_plan);
        config.domain_tp_contexts = std::move(domain_tp_contexts);
        return config;
    }

    ExpertComputeDomain domain(const std::string &name)
    {
        ExpertComputeDomain result;
        result.name = name;
        result.kind = ExpertDomainKind::SingleDevice;
        result.backend = CollectiveBackendType::HOST;
        result.participants = {GlobalDeviceAddress::cpu()};
        result.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
        return result;
    }

    ExpertComputeDomain domainWith(
        const std::string &name,
        ExpertDomainKind kind,
        std::vector<GlobalDeviceAddress> participants,
        ExpertDomainComputeKind compute_kind,
        CollectiveBackendType backend)
    {
        ExpertComputeDomain result;
        result.name = name;
        result.kind = kind;
        result.backend = backend;
        result.participants = std::move(participants);
        result.compute_kind = compute_kind;
        result.owner_rank = 0;
        return result;
    }

    std::shared_ptr<MoEExpertParallelPlan> makePlan(
        std::vector<ExpertRoutedTier> tiers,
        std::vector<int> placement)
    {
        auto plan = std::make_shared<MoEExpertParallelPlan>();
        plan->enabled = true;
        plan->execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
        plan->continuation_domain = "continuation";
        plan->shared_expert_domain = "continuation";
        plan->residency_policy = ExpertResidencyPolicy::ExplicitMasks;
        plan->domains.push_back(domain("continuation"));
        for (const auto &tier : tiers)
            plan->domains.push_back(domain(tier.domain));
        plan->routed_tiers = std::move(tiers);
        plan->placements.push_back(ExpertLayerPlacement{
            .layer = 0,
            .routed_expert_tier = std::move(placement),
        });
        validateMoEExpertParallelPlanOrThrow(*plan, {.layer_count = 1, .routed_expert_count = kNumExperts});
        return plan;
    }

    std::shared_ptr<MoEExpertParallelPlan> makePlanWithDomains(
        std::vector<ExpertComputeDomain> domains,
        std::vector<ExpertRoutedTier> tiers,
        std::vector<int> placement,
        std::string continuation_domain,
        std::string shared_expert_domain = {})
    {
        auto plan = std::make_shared<MoEExpertParallelPlan>();
        plan->enabled = true;
        plan->execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
        plan->continuation_domain = std::move(continuation_domain);
        plan->shared_expert_domain = shared_expert_domain.empty()
                                          ? plan->continuation_domain
                                          : std::move(shared_expert_domain);
        plan->residency_policy = ExpertResidencyPolicy::ExplicitMasks;
        plan->domains = std::move(domains);
        plan->routed_tiers = std::move(tiers);
        plan->placements.push_back(ExpertLayerPlacement{
            .layer = 0,
            .routed_expert_tier = std::move(placement),
        });
        validateMoEExpertParallelPlanOrThrow(*plan, {.layer_count = 1, .routed_expert_count = kNumExperts});
        return plan;
    }

    std::vector<ExpertRoutedTier> acceleratorTiers()
    {
        return {
            ExpertRoutedTier{.name = "cuda_hot", .domain = "cuda_domain", .priority = 0, .fallback = false},
            ExpertRoutedTier{.name = "rocm_hot", .domain = "rocm_domain", .priority = 1, .fallback = false},
            ExpertRoutedTier{.name = "cpu_cold", .domain = "cpu_domain", .priority = 2, .fallback = true},
        };
    }

    std::vector<ExpertRoutedTier> rocmAndCpuTiers()
    {
        return {
            ExpertRoutedTier{.name = "rocm_hot", .domain = "rocm_domain", .priority = 0, .fallback = false},
            ExpertRoutedTier{.name = "cpu_cold", .domain = "cpu_domain", .priority = 1, .fallback = true},
        };
    }

    std::vector<ExpertComputeDomain> cpuContinuationRocmAndCpuDomains()
    {
        return {
            domain("continuation"),
            domainWith("rocm_domain", ExpertDomainKind::LocalTP,
                       {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)},
                       ExpertDomainComputeKind::TensorParallelExperts,
                       CollectiveBackendType::AUTO),
            domainWith("cpu_domain", ExpertDomainKind::NodeLocalTP,
                       {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)},
                       ExpertDomainComputeKind::TensorParallelExperts,
                       CollectiveBackendType::UPI),
        };
    }

    std::vector<ExpertComputeDomain> acceleratorDomains()
    {
        return {
            domainWith("cuda_domain", ExpertDomainKind::SingleDevice,
                       {GlobalDeviceAddress::cuda(0)},
                       ExpertDomainComputeKind::ReplicatedExperts,
                       CollectiveBackendType::NCCL),
            domainWith("rocm_domain", ExpertDomainKind::LocalTP,
                       {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)},
                       ExpertDomainComputeKind::TensorParallelExperts,
                       CollectiveBackendType::RCCL),
            domainWith("cpu_domain", ExpertDomainKind::NodeLocalTP,
                       {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)},
                       ExpertDomainComputeKind::TensorParallelExperts,
                       CollectiveBackendType::UPI),
        };
    }

    std::vector<ExpertRoutedTier> twoTiers()
    {
        return {
            ExpertRoutedTier{.name = "hot", .domain = "hot_domain", .priority = 0, .fallback = false},
            ExpertRoutedTier{.name = "cold", .domain = "cold_domain", .priority = 1, .fallback = true},
        };
    }

    std::vector<ExpertRoutedTier> threeTiers()
    {
        return {
            ExpertRoutedTier{.name = "hot", .domain = "hot_domain", .priority = 0, .fallback = false},
            ExpertRoutedTier{.name = "warm", .domain = "warm_domain", .priority = 1, .fallback = false},
            ExpertRoutedTier{.name = "cold", .domain = "cold_domain", .priority = 2, .fallback = true},
        };
    }

    void fillRouterWeights(FP32Tensor *router)
    {
        fill(router, -2.0f);
        float *data = router->mutable_data();
        auto set_weight = [&](int expert, int dim, float value) {
            data[static_cast<size_t>(expert) * kDModel + static_cast<size_t>(dim)] = value;
        };

        set_weight(0, 0, 6.0f);
        set_weight(1, 0, 5.0f);
        set_weight(2, 1, 6.0f);
        set_weight(3, 1, 5.0f);
        set_weight(4, 2, 6.0f);
        set_weight(5, 2, 5.0f);
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
        fillRouterWeights(static_cast<FP32Tensor *>(layer.moe_gate));

        layer.moe_gate_exps = arena.fp32({kDModel, kIntermediate, kNumExperts});
        layer.moe_up_exps = arena.fp32({kDModel, kIntermediate, kNumExperts});
        layer.moe_down_exps = arena.fp32({kIntermediate, kDModel, kNumExperts});
        fillExpert3D(static_cast<FP32Tensor *>(layer.moe_gate_exps), kIntermediate, kDModel, 0.010f);
        fillExpert3D(static_cast<FP32Tensor *>(layer.moe_up_exps), kIntermediate, kDModel, 0.012f);
        fillExpert3D(static_cast<FP32Tensor *>(layer.moe_down_exps), kDModel, kIntermediate, 0.008f);

        return layer;
    }

    LayerWeights makeLayerWeightsWithSharedExpert(TensorArena &arena)
    {
        LayerWeights layer = makeLayerWeights(arena);

        layer.shared_expert_gate = arena.fp32({kIntermediate, kDModel});
        layer.shared_expert_up = arena.fp32({kIntermediate, kDModel});
        layer.shared_expert_down = arena.fp32({kDModel, kIntermediate});
        layer.shared_expert_gate_inp = arena.fp32({kDModel});

        fill(static_cast<FP32Tensor *>(layer.shared_expert_gate), 0.010f);
        fill(static_cast<FP32Tensor *>(layer.shared_expert_up), 0.012f);
        fill(static_cast<FP32Tensor *>(layer.shared_expert_down), 0.008f);
        fill(static_cast<FP32Tensor *>(layer.shared_expert_gate_inp), 0.25f);

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

        float *attn = buffers.attn_proj->mutable_data();
        attn[0 * kDModel + 0] = 1.0f;
        attn[1 * kDModel + 1] = 1.0f;
        attn[2 * kDModel + 2] = 1.0f;

        buffers.extensions[BufferId::MOE_EXPERT_INDICES] = arena.fp32({kSeqLen, kTopK});
        buffers.extensions[BufferId::MOE_EXPERT_WEIGHTS] = arena.fp32({kSeqLen, kTopK});
        buffers.extensions[BufferId::MOE_COMBINED_OUTPUT] = buffers.attn_proj;
        buffers.extensions[BufferId::MOE_SHARED_EXPERT_OUTPUT] = arena.fp32({kSeqLen, kDModel});
        buffers.extensions[BufferId::MOE_GATE_SCRATCH] = arena.fp32({kSeqLen, kIntermediate});
        buffers.extensions[BufferId::MOE_UP_SCRATCH] = arena.fp32({kSeqLen, kIntermediate});
        return buffers;
    }

    ActivationBuffers makeActivationBuffersWithSeparateMoEOutput(TensorArena &arena)
    {
        ActivationBuffers buffers = makeActivationBuffers(arena);
        buffers.extensions[BufferId::MOE_COMBINED_OUTPUT] = arena.fp32({kSeqLen, kDModel});
        return buffers;
    }

    void executeGraph(ComputeGraph &graph, llaminar2::testing::MockDeviceContext *ctx)
    {
        for (const auto &node_name : graph.getExecutionOrder())
        {
            auto *node = graph.getNode(node_name);
            ASSERT_NE(node, nullptr) << node_name;
            ASSERT_TRUE(node->stage->execute(ctx)) << node_name;
        }
    }

    const MoEExpertDispatchOutput &dispatchOutput(const ComputeGraph &graph)
    {
        const auto *node = graph.getNode("layer0_moe_expert_dispatch");
        EXPECT_NE(node, nullptr);
        const auto *stage = dynamic_cast<const MoEExpertDispatchStage *>(node->stage.get());
        EXPECT_NE(stage, nullptr);
        EXPECT_NE(stage->params().output, nullptr);
        return *stage->params().output;
    }

    void expectAllExpertsCoveredOnceByDispatch(
        const ComputeGraph &graph,
        const std::vector<int> &placement,
        size_t tier_count)
    {
        const auto &output = dispatchOutput(graph);
        ASSERT_EQ(output.tiers.size(), tier_count);

        std::vector<int> expert_counts(kNumExperts, 0);
        std::vector<int> expert_tiers(kNumExperts, -1);
        for (size_t tier_index = 0; tier_index < output.tiers.size(); ++tier_index)
        {
            for (const auto &entry : output.tiers[tier_index].entries)
            {
                ASSERT_GE(entry.expert_id, 0);
                ASSERT_LT(entry.expert_id, kNumExperts);
                ++expert_counts[static_cast<size_t>(entry.expert_id)];
                expert_tiers[static_cast<size_t>(entry.expert_id)] = static_cast<int>(tier_index);
            }
        }

        for (int expert = 0; expert < kNumExperts; ++expert)
        {
            EXPECT_EQ(expert_counts[static_cast<size_t>(expert)], 1) << "expert=" << expert;
            EXPECT_EQ(expert_tiers[static_cast<size_t>(expert)], placement[static_cast<size_t>(expert)])
                << "expert=" << expert;
        }
    }

    void expectTensorNear(const TensorBase *actual, const TensorBase *expected)
    {
        ASSERT_EQ(actual->numel(), expected->numel());
        const float *actual_data = actual->data();
        const float *expected_data = expected->data();
        for (size_t i = 0; i < actual->numel(); ++i)
        {
            EXPECT_NEAR(actual_data[i], expected_data[i], 1e-4f) << "index=" << i;
        }
    }

    bool hasNonZeroValue(const TensorBase *tensor)
    {
        const float *data = tensor->data();
        for (size_t i = 0; i < tensor->numel(); ++i)
        {
            if (std::fabs(data[i]) > 1e-8f)
                return true;
        }
        return false;
    }

    std::shared_ptr<Q4_KTensor> makeQ4KExpertTensor3D(
        const std::vector<size_t> &shape,
        uint32_t seed)
    {
        if (shape.size() != 3)
            throw std::invalid_argument("makeQ4KExpertTensor3D requires a 3D GGUF expert shape");

        constexpr size_t kBlockSize = Q4_KBlock::BLOCK_SIZE;
        const size_t cols = shape[0];
        const size_t rows_per_expert = shape[1];
        const size_t expert_count = shape[2];
        const size_t blocks_per_row = (cols + kBlockSize - 1) / kBlockSize;
        const size_t total_blocks = rows_per_expert * expert_count * blocks_per_row;

        std::vector<uint8_t> raw_data(total_blocks * sizeof(Q4_KBlock));
        auto *blocks = reinterpret_cast<Q4_KBlock *>(raw_data.data());
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> qdist(0, 15);

        for (size_t block_index = 0; block_index < total_blocks; ++block_index)
        {
            blocks[block_index].d = fp32_to_fp16(0.002f + 0.0002f * static_cast<float>(block_index % 7));
            blocks[block_index].dmin = fp32_to_fp16(0.0f);
            std::memset(blocks[block_index].scales, 0x44, sizeof(blocks[block_index].scales));
            for (size_t byte_index = 0; byte_index < sizeof(blocks[block_index].qs); ++byte_index)
            {
                const uint8_t lo = static_cast<uint8_t>(qdist(rng));
                const uint8_t hi = static_cast<uint8_t>(qdist(rng));
                blocks[block_index].qs[byte_index] = static_cast<uint8_t>((hi << 4) | lo);
            }
        }

        return std::make_shared<Q4_KTensor>(shape, raw_data);
    }

    FP32Tensor *dequantizeExpertTensor3D(
        TensorArena &arena,
        TensorBase *source,
        int rows,
        int cols)
    {
        auto *dequantized = arena.fp32(source->shape());
        float *dst = dequantized->mutable_data();
        const size_t expert_elements = static_cast<size_t>(rows) * static_cast<size_t>(cols);
        for (int expert = 0; expert < kNumExperts; ++expert)
        {
            const size_t offset = static_cast<size_t>(expert) * expert_elements;
            auto view = source->create_view({static_cast<size_t>(rows), static_cast<size_t>(cols)}, offset);
            if (!view)
                throw std::runtime_error("failed to create expert dequantization view");
            const float *view_data = view->data();
            if (!view_data)
                throw std::runtime_error("expert dequantization view returned null data");
            std::copy_n(view_data, expert_elements, dst + offset);
        }
        return dequantized;
    }

    void fillDirectLocalTPInputs(ActivationBuffers &buffers)
    {
        float *input = buffers.normalized->mutable_data();
        for (int token = 0; token < kSeqLen; ++token)
        {
            for (int dim = 0; dim < kDModel; ++dim)
            {
                input[static_cast<size_t>(token) * kDModel + dim] =
                    0.04f * static_cast<float>(token + 1) +
                    0.01f * static_cast<float>(dim + 1);
            }
        }

        static constexpr int routes[kSeqLen][kTopK] = {
            {0, 1},
            {1, 4},
            {4, 0},
        };
        static constexpr float route_weights[kSeqLen][kTopK] = {
            {0.70f, 0.30f},
            {0.25f, 0.75f},
            {0.60f, 0.40f},
        };

        float *routing_indices = buffers.extensions[BufferId::MOE_EXPERT_INDICES]->mutable_data();
        float *routing_weights = buffers.extensions[BufferId::MOE_EXPERT_WEIGHTS]->mutable_data();
        for (int token = 0; token < kSeqLen; ++token)
        {
            for (int topk = 0; topk < kTopK; ++topk)
            {
                const size_t offset = static_cast<size_t>(token) * kTopK + topk;
                routing_indices[offset] = static_cast<float>(routes[token][topk]);
                routing_weights[offset] = route_weights[token][topk];
            }
        }
    }

    bool runDenseLocalTPReference(
        const MoEOverlayRuntimeDomain &domain,
        ILocalTPContext &local_tp_context,
        const LayerWeights &layer,
        ActivationBuffers &buffers,
        const std::vector<bool> &expert_mask,
        TensorBase *output)
    {
        MoEExpertOverlayLocalTPRunParams params;
        params.input = buffers.normalized;
        params.routing_indices = buffers.extensions[BufferId::MOE_EXPERT_INDICES];
        params.routing_weights = buffers.extensions[BufferId::MOE_EXPERT_WEIGHTS];
        params.gate_exps = layer.moe_gate_exps;
        params.up_exps = layer.moe_up_exps;
        params.down_exps = layer.moe_down_exps;
        params.output = output;
        params.seq_len = kSeqLen;
        params.d_model = kDModel;
        params.num_experts = kNumExperts;
        params.top_k = kTopK;
        params.expert_intermediate = kIntermediate;
        params.layer_idx = 0;
        params.expert_mask = expert_mask;
        params.upload_partials_to_participant_devices = false;

        return MoEExpertOverlayLocalTPExecutor::runTensorParallelExperts(
            domain,
            local_tp_context,
            params,
            nullptr);
    }

    class CountingPreparedGemm : public ITensorGemm
    {
    public:
        explicit CountingPreparedGemm(float fill_value)
            : fill_value_(fill_value)
        {
        }

        bool supports_device(int) const override { return true; }

        bool multiply_tensor(
            const TensorBase *, TensorBase *output,
            int m, int n, int,
            bool,
            float alpha,
            float,
            const TensorBase *,
            const IMPIContext *,
            int,
            DeviceWorkspaceManager *,
            int) override
        {
            ++multiply_calls;
            float *out = output->mutable_data();
            for (int row = 0; row < m; ++row)
            {
                for (int col = 0; col < n; ++col)
                    out[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)] = alpha * fill_value_;
            }
            return true;
        }

        bool multiply_tensor_with_fused_swiglu(
            const TensorBase *, const TensorBase *, TensorBase *output,
            int m, int n, int,
            float alpha = 1.0f, float = 0.0f) override
        {
            ++fused_swiglu_calls;
            float *out = output->mutable_data();
            for (int row = 0; row < m; ++row)
            {
                for (int col = 0; col < n; ++col)
                    out[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)] = alpha * fill_value_;
            }
            return true;
        }

        int multiply_calls = 0;
        int fused_swiglu_calls = 0;

    private:
        float fill_value_ = 0.0f;
    };

    MoEOverlayRuntimeDomain cpuLocalTPRuntimeDomain()
    {
        MoEOverlayRuntimeDomain domain;
        domain.name = "cpu_localtp";
        domain.kind = ExpertDomainKind::LocalTP;
        domain.backend = CollectiveBackendType::HOST;
        domain.compute_kind = ExpertDomainComputeKind::TensorParallelExperts;
        domain.primary_device = DeviceId::cpu();
        domain.participants = {
            MoEOverlayDomainParticipant{
                .address = GlobalDeviceAddress::cpu(0),
                .participant_index = 0,
                .owned_by_current_rank = true,
                .locally_addressable = true,
                .local_device = DeviceId::cpu(),
            },
            MoEOverlayDomainParticipant{
                .address = GlobalDeviceAddress::cpu(1),
                .participant_index = 1,
                .owned_by_current_rank = true,
                .locally_addressable = true,
                .local_device = DeviceId::cpu(),
            },
        };
        return domain;
    }

    MoEOverlayRuntimeDomain rocmLocalTPRuntimeDomain()
    {
        MoEOverlayRuntimeDomain domain;
        domain.name = "rocm_localtp";
        domain.kind = ExpertDomainKind::LocalTP;
        domain.backend = CollectiveBackendType::RCCL;
        domain.compute_kind = ExpertDomainComputeKind::TensorParallelExperts;
        domain.primary_device = DeviceId::rocm(0);
        domain.participants = {
            MoEOverlayDomainParticipant{
                .address = GlobalDeviceAddress::rocm(0),
                .participant_index = 0,
                .owned_by_current_rank = true,
                .locally_addressable = true,
                .local_device = DeviceId::rocm(0),
            },
            MoEOverlayDomainParticipant{
                .address = GlobalDeviceAddress::rocm(1),
                .participant_index = 1,
                .owned_by_current_rank = true,
                .locally_addressable = true,
                .local_device = DeviceId::rocm(1),
            },
        };
        return domain;
    }

    MoEExpertOverlayLocalTPPreparedParticipant makePreparedParticipant(
        int participant_index,
        DeviceId device,
        ITensorGemm *gate,
        ITensorGemm *up,
        ITensorGemm *down)
    {
        MoEExpertOverlayLocalTPPreparedParticipant participant;
        participant.participant_index = participant_index;
        participant.device = device;
        participant.gate_gemm.assign(kNumExperts, nullptr);
        participant.up_gemm.assign(kNumExperts, nullptr);
        participant.down_gemm.assign(kNumExperts, nullptr);
        participant.gate_gemm[0] = gate;
        participant.up_gemm[0] = up;
        participant.down_gemm[0] = down;
        participant.input_scratch = std::make_shared<FP32Tensor>(std::vector<size_t>{kSeqLen, kDModel});
        participant.batch_scratch = std::make_shared<FP32Tensor>(std::vector<size_t>{kSeqLen, kDModel});
        participant.gate_scratch = std::make_shared<FP32Tensor>(std::vector<size_t>{kSeqLen, kIntermediate});
        participant.up_scratch = std::make_shared<FP32Tensor>(std::vector<size_t>{kSeqLen, kIntermediate});
        participant.partial_scratch = std::make_shared<FP32Tensor>(std::vector<size_t>{kSeqLen, kDModel});
        participant.token_indices.reserve(static_cast<size_t>(kSeqLen) * kTopK);
        participant.token_weights.reserve(static_cast<size_t>(kSeqLen) * kTopK);
        return participant;
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

    void runOverlayCase(
        std::shared_ptr<MoEExpertParallelPlan> overlay_plan,
        std::vector<int> placement,
        size_t tier_count)
    {
        TensorArena weight_arena;
        auto layer = makeLayerWeights(weight_arena);

        auto ctx = std::make_unique<llaminar2::testing::MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);

        GraphConfig reference_config = makeConfig();
        Qwen35MoEGraph reference_builder(reference_config, nullptr);
        TensorArena reference_activation_arena;
        auto reference_buffers = makeActivationBuffers(reference_activation_arena);
        ComputeGraph reference_graph = reference_builder.buildFFNGraph(
            layer, reference_buffers, 0, kSeqLen, kBatchSize, DeviceId::cpu());
        ASSERT_NE(reference_graph.getNode("layer0_moe_expert_ffn"), nullptr);
        ASSERT_EQ(reference_graph.getNode("layer0_moe_expert_dispatch"), nullptr);
        executeGraph(reference_graph, ctx.get());

        GraphConfig overlay_config = makeConfig(std::move(overlay_plan));
        Qwen35MoEGraph overlay_builder(overlay_config, nullptr);
        TensorArena overlay_activation_arena;
        auto overlay_buffers = makeActivationBuffers(overlay_activation_arena);
        ComputeGraph overlay_graph = overlay_builder.buildFFNGraph(
            layer, overlay_buffers, 0, kSeqLen, kBatchSize, DeviceId::cpu());

        ASSERT_EQ(overlay_graph.getNode("layer0_moe_expert_ffn"), nullptr);
        ASSERT_NE(overlay_graph.getNode("layer0_moe_expert_dispatch"), nullptr);
        ASSERT_NE(overlay_graph.getNode("layer0_moe_expert_parallel_reduce"), nullptr);
        EXPECT_EQ(countStagesOfType(overlay_graph, ComputeStageType::MOE_EXPERT_FFN), tier_count);

        executeGraph(overlay_graph, ctx.get());
        expectAllExpertsCoveredOnceByDispatch(overlay_graph, placement, tier_count);
        expectTensorNear(overlay_buffers.attn_proj, reference_buffers.attn_proj);
    }
} // namespace

TEST(Test__Qwen35MoEExpertOverlayGraph, TwoTierOverlayCoversExpertsOnceAndMatchesFullExpertCompute)
{
    std::vector<int> placement = {0, 1, 0, 1, 0, 1};
    runOverlayCase(makePlan(twoTiers(), placement), placement, 2);
}

TEST(Test__Qwen35MoEExpertOverlayGraph, ThreeTierOverlayCoversExpertsOnceAndMatchesFullExpertCompute)
{
    std::vector<int> placement = {0, 1, 2, 0, 1, 2};
    runOverlayCase(makePlan(threeTiers(), placement), placement, 3);
}

TEST(Test__Qwen35MoEExpertOverlayGraph, NonOverlayConfigUsesLegacyRoutedExpertPath)
{
    TensorArena weight_arena;
    auto layer = makeLayerWeights(weight_arena);
    TensorArena activation_arena;
    auto buffers = makeActivationBuffers(activation_arena);

    Qwen35MoEGraph graph_builder(makeConfig(), nullptr);
    ComputeGraph graph = graph_builder.buildFFNGraph(layer, buffers, 0, kSeqLen, kBatchSize, DeviceId::cpu());

    EXPECT_NE(graph.getNode("layer0_moe_expert_ffn"), nullptr);
    EXPECT_EQ(graph.getNode("layer0_moe_expert_dispatch"), nullptr);
    EXPECT_EQ(graph.getNode("layer0_moe_expert_parallel_reduce"), nullptr);
}

TEST(Test__Qwen35MoEExpertOverlayGraph, CpuNodeLocalTierLowersToDomainFallbackStageWithoutPrimaryOnlyGpuTiers)
{
    TensorArena weight_arena;
    auto layer = makeLayerWeights(weight_arena);
    TensorArena activation_arena;
    auto buffers = makeActivationBuffers(activation_arena);

    auto plan = makePlanWithDomains(
        acceleratorDomains(),
        acceleratorTiers(),
        /*placement=*/{2, 2, 2, 2, 2, 2},
        /*continuation_domain=*/"cuda_domain");

    Qwen35MoEGraph graph_builder(makeConfig(std::move(plan)), nullptr);
    ComputeGraph graph = graph_builder.buildFFNGraph(layer, buffers, 0, kSeqLen, kBatchSize, DeviceId::cpu());

    const auto *cuda_node = graph.getNode("layer0_moe_expert_ffn_tier0_cuda_hot");
    const auto *rocm_node = graph.getNode("layer0_moe_expert_ffn_tier1_rocm_hot");
    const auto *cpu_node = graph.getNode("layer0_moe_expert_ffn_tier2_cpu_cold");
    EXPECT_EQ(cuda_node, nullptr);
    EXPECT_EQ(rocm_node, nullptr);
    ASSERT_NE(cpu_node, nullptr);

    EXPECT_EQ(cpu_node->device, DeviceId::cpu());
    EXPECT_EQ(cpu_node->stage->device(), DeviceId::cpu());
    const auto *runtime_stage = dynamic_cast<const MoEOverlayDomainRuntimeStage *>(cpu_node->stage.get());
    ASSERT_NE(runtime_stage, nullptr);
    EXPECT_NE(runtime_stage->params().runtime, nullptr);
    EXPECT_NE(runtime_stage->params().result_lifetime, nullptr);
    const auto &request = runtime_stage->params().request;
    EXPECT_EQ(request.runtime_domain.name, "cpu_domain");
    EXPECT_EQ(request.runtime_domain.kind, ExpertDomainKind::NodeLocalTP);
    EXPECT_EQ(request.runtime_domain.compute_kind, ExpertDomainComputeKind::TensorParallelExperts);
    EXPECT_TRUE(request.dispatch_group.isValid());
    EXPECT_EQ(request.dispatch_group.layer_id, 0);
    EXPECT_EQ(request.dispatch_group.participant_count, 2);
    EXPECT_EQ(request.dispatch_group.participant_index, 0);
    EXPECT_EQ(request.dispatch_group.owner_participant_index, 0);
    EXPECT_EQ(request.dispatch_group.executor_participant_index, 0);
    ASSERT_TRUE(request.has_cpu_fallback_params);
    EXPECT_EQ(request.cpu_fallback_params.domain.name, "cpu_domain");
    EXPECT_EQ(request.cpu_fallback_params.domain.kind, ExpertDomainKind::NodeLocalTP);
    EXPECT_EQ(request.cpu_fallback_params.domain.compute_kind, ExpertDomainComputeKind::TensorParallelExperts);
    EXPECT_EQ(request.cpu_fallback_params.domain.participants.size(), 2u);
    EXPECT_EQ(request.cpu_fallback_params.root_world_rank, 0);
}

TEST(Test__Qwen35MoEExpertOverlayGraph, ActiveRocmLocalTPTierRequiresDomainTPContext)
{
    TensorArena weight_arena;
    auto layer = makeLayerWeights(weight_arena);
    TensorArena activation_arena;
    auto buffers = makeActivationBuffers(activation_arena);

    auto plan = makePlanWithDomains(
        acceleratorDomains(),
        acceleratorTiers(),
        /*placement=*/{1, 1, 1, 1, 1, 1},
        /*continuation_domain=*/"cuda_domain");

    Qwen35MoEGraph graph_builder(makeConfig(std::move(plan)), nullptr);

    try
    {
        (void)graph_builder.buildFFNGraph(layer, buffers, 0, kSeqLen, kBatchSize, DeviceId::cpu());
        FAIL() << "Expected active ROCm LocalTP routed tier to fail graph construction";
    }
    catch (const std::runtime_error &e)
    {
        const std::string message = e.what();
        EXPECT_NE(message.find("Bridge Phase 5D missing LocalTP domain context"), std::string::npos) << message;
        EXPECT_NE(message.find("rocm_hot"), std::string::npos) << message;
        EXPECT_NE(message.find("LocalTP"), std::string::npos) << message;
        EXPECT_NE(message.find("TensorParallelExperts"), std::string::npos) << message;
        EXPECT_NE(message.find("domain_tp_contexts"), std::string::npos) << message;
    }
}

TEST(Test__Qwen35MoEExpertOverlayGraph, ActiveRocmLocalTPTierBuildsAndRunsWithMockContextAndKeepsCpuFallbackStage)
{
    TensorArena weight_arena;
    auto layer = makeLayerWeights(weight_arena);
    TensorArena activation_arena;
    auto buffers = makeActivationBuffers(activation_arena);

    auto plan = makePlanWithDomains(
        cpuContinuationRocmAndCpuDomains(),
        rocmAndCpuTiers(),
        /*placement=*/{0, 0, 1, 1, 0, 1},
        /*continuation_domain=*/"continuation");

    MockLocalTPContext rocm_tp_context;
    rocm_tp_context.setDevices({GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});
    rocm_tp_context.setBackend(CollectiveBackendType::HOST);

    GraphConfig config = makeConfig(std::move(plan), {{"rocm_domain", &rocm_tp_context}});
    Qwen35MoEGraph graph_builder(config, nullptr);
    ComputeGraph graph = graph_builder.buildFFNGraph(layer, buffers, 0, kSeqLen, kBatchSize, DeviceId::cpu());

    const auto *rocm_node = graph.getNode("layer0_moe_expert_ffn_tier0_rocm_hot");
    const auto *cpu_node = graph.getNode("layer0_moe_expert_ffn_tier1_cpu_cold");
    ASSERT_NE(rocm_node, nullptr);
    ASSERT_NE(cpu_node, nullptr);
    EXPECT_EQ(rocm_node->device, DeviceId::rocm(0));
    EXPECT_EQ(rocm_node->stage->device(), DeviceId::rocm(0));

    const auto *local_runtime_stage = dynamic_cast<const MoEOverlayDomainRuntimeStage *>(rocm_node->stage.get());
    ASSERT_NE(local_runtime_stage, nullptr);
    EXPECT_EQ(local_runtime_stage->coherencePolicy(), CoherencePolicy::NONE);
    EXPECT_TRUE(local_runtime_stage->bufferContract().empty());
    EXPECT_NE(local_runtime_stage->params().runtime, nullptr);
    ASSERT_TRUE(local_runtime_stage->params().request.has_local_tp_params);
    EXPECT_TRUE(local_runtime_stage->params().request.dispatch_group.isValid());
    EXPECT_EQ(local_runtime_stage->params().request.dispatch_group.layer_id, 0);
    EXPECT_EQ(local_runtime_stage->params().request.dispatch_group.participant_count, 2);
    EXPECT_EQ(local_runtime_stage->params().request.dispatch_group.participant_index, 0);
    const auto &local_tp_params = local_runtime_stage->params().request.local_tp_params;
    EXPECT_EQ(local_runtime_stage->params().request.runtime_domain.name, "rocm_domain");
    EXPECT_EQ(local_tp_params.domain.name, "rocm_domain");
    EXPECT_EQ(local_tp_params.domain.kind, ExpertDomainKind::LocalTP);
    EXPECT_EQ(local_tp_params.domain.compute_kind, ExpertDomainComputeKind::TensorParallelExperts);
    EXPECT_EQ(local_tp_params.local_tp_context, &rocm_tp_context);
    EXPECT_NE(local_tp_params.dispatch_output_lifetime, nullptr);
    EXPECT_EQ(local_tp_params.dispatch_tier_index, 0);
    EXPECT_NE(local_tp_params.diagnostics_lifetime, nullptr);

    const auto *cpu_runtime_stage = dynamic_cast<const MoEOverlayDomainRuntimeStage *>(cpu_node->stage.get());
    ASSERT_NE(cpu_runtime_stage, nullptr);
    EXPECT_NE(cpu_runtime_stage->params().runtime, nullptr);
    ASSERT_TRUE(cpu_runtime_stage->params().request.has_cpu_fallback_params);
    EXPECT_TRUE(cpu_runtime_stage->params().request.dispatch_group.isValid());
    EXPECT_EQ(cpu_runtime_stage->params().request.dispatch_group.layer_id, 0);
    EXPECT_EQ(cpu_runtime_stage->params().request.dispatch_group.participant_count, 2);
    EXPECT_EQ(cpu_runtime_stage->params().request.dispatch_group.participant_index, 0);
    const auto &fallback_params = cpu_runtime_stage->params().request.cpu_fallback_params;
    EXPECT_EQ(cpu_runtime_stage->params().request.runtime_domain.name, "cpu_domain");
    EXPECT_EQ(fallback_params.domain.name, "cpu_domain");
    EXPECT_EQ(fallback_params.domain.kind, ExpertDomainKind::NodeLocalTP);
    EXPECT_EQ(fallback_params.domain.compute_kind, ExpertDomainComputeKind::TensorParallelExperts);
    EXPECT_EQ(fallback_params.domain.participants.size(), 2u);
    EXPECT_NE(fallback_params.dispatch_output_lifetime, nullptr);
    EXPECT_EQ(fallback_params.dispatch_tier_index, 1);
    EXPECT_EQ(fallback_params.transfer_mode, MoEExpertTransferMode::Auto);
    EXPECT_EQ(local_tp_params.dispatch_output_lifetime.get(), fallback_params.dispatch_output_lifetime.get());

    auto ctx = std::make_unique<llaminar2::testing::MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);
    for (const std::string &node_name : {
             "layer0_ffn_norm",
             "layer0_moe_routing",
             "layer0_moe_expert_dispatch",
             "layer0_moe_expert_ffn_tier0_rocm_hot"})
    {
        auto *node = graph.getNode(node_name);
        ASSERT_NE(node, nullptr) << node_name;
        ASSERT_TRUE(node->stage->execute(ctx.get())) << node_name;
    }

    const auto *dispatch = local_tp_params.dispatch_output_lifetime.get();
    ASSERT_NE(dispatch, nullptr);
    ASSERT_EQ(dispatch->tiers.size(), 2u);
    EXPECT_EQ(dispatch->tiers[0].token_rows, (std::vector<int>{0, 2}));
    EXPECT_EQ(dispatch->tiers[0].entries.size(), 3u);
    EXPECT_TRUE(dispatch->tiers[0].transfer_required);
    EXPECT_EQ(dispatch->tiers[0].transfer_mode, MoEExpertTransferMode::SparseTokenRows);
    EXPECT_EQ(dispatch->tiers[0].transfer_volume.selected_rows, 2u);
    EXPECT_LT(dispatch->tiers[0].transfer_volume.totalBytes(), dispatch->tiers[0].transfer_volume.denseTotalBytes());
    EXPECT_EQ(dispatch->tiers[1].token_rows, (std::vector<int>{1, 2}));
    EXPECT_EQ(dispatch->tiers[1].entries.size(), 3u);
    EXPECT_TRUE(dispatch->tiers[1].transfer_required);
    EXPECT_EQ(dispatch->tiers[1].transfer_mode, MoEExpertTransferMode::SparseTokenRows);
    EXPECT_EQ(dispatch->tiers[1].transfer_volume.selected_rows, 2u);

    EXPECT_EQ(rocm_tp_context.allreduceCallCount(), 6);
    const auto *diagnostics = local_tp_params.diagnostics_lifetime.get();
    ASSERT_NE(diagnostics, nullptr);
    EXPECT_EQ(diagnostics->domain_name, "rocm_domain");
    EXPECT_EQ(diagnostics->backend, CollectiveBackendType::HOST);
    EXPECT_EQ(diagnostics->degree, 2);
    EXPECT_EQ(diagnostics->total_routed_entries, 3);
    EXPECT_EQ(diagnostics->transfer_mode, MoEExpertTransferMode::SparseTokenRows);
    EXPECT_EQ(diagnostics->selected_token_rows, (std::vector<int>{0, 2}));
    EXPECT_EQ(diagnostics->transfer_volume.selected_rows, 2u);
    ASSERT_EQ(diagnostics->participants.size(), 2u);
    EXPECT_EQ(diagnostics->participants[0].device, DeviceId::rocm(0));
    EXPECT_EQ(diagnostics->participants[0].executed_expert_ids, (std::vector<int>{0, 1, 4}));
    EXPECT_EQ(diagnostics->participants[0].expert_allreduce_count, 3);
    EXPECT_EQ(diagnostics->participants[1].device, DeviceId::rocm(1));
    EXPECT_EQ(diagnostics->participants[1].executed_expert_ids, (std::vector<int>{0, 1, 4}));
    EXPECT_EQ(diagnostics->participants[1].expert_allreduce_count, 3);
    EXPECT_TRUE(hasNonZeroValue(local_tp_params.output));

    auto reference = activation_arena.fp32({kSeqLen, kDModel});
    ASSERT_TRUE(runDenseLocalTPReference(local_tp_params.domain,
                                         rocm_tp_context,
                                         layer,
                                         buffers,
                                         {true, true, false, false, true, false},
                                         reference));
    expectTensorNear(local_tp_params.output, reference);
}

TEST(Test__Qwen35MoEExpertOverlayGraph, LocalTPExecutorConsumesQ4KExpertWeightsViaHostDequantBridge)
{
    TensorArena weight_arena;
    auto layer = makeLayerWeights(weight_arena);
    layer.moe_gate_exps = weight_arena.keep(makeQ4KExpertTensor3D({kDModel, kIntermediate, kNumExperts}, 11));
    layer.moe_up_exps = weight_arena.keep(makeQ4KExpertTensor3D({kDModel, kIntermediate, kNumExperts}, 23));
    layer.moe_down_exps = weight_arena.keep(makeQ4KExpertTensor3D({kIntermediate, kDModel, kNumExperts}, 37));

    TensorArena reference_weight_arena;
    LayerWeights reference_layer = layer;
    reference_layer.moe_gate_exps = dequantizeExpertTensor3D(reference_weight_arena,
                                                             layer.moe_gate_exps,
                                                             kIntermediate,
                                                             kDModel);
    reference_layer.moe_up_exps = dequantizeExpertTensor3D(reference_weight_arena,
                                                           layer.moe_up_exps,
                                                           kIntermediate,
                                                           kDModel);
    reference_layer.moe_down_exps = dequantizeExpertTensor3D(reference_weight_arena,
                                                             layer.moe_down_exps,
                                                             kDModel,
                                                             kIntermediate);

    auto plan = makePlanWithDomains(
        cpuContinuationRocmAndCpuDomains(),
        rocmAndCpuTiers(),
        /*placement=*/{0, 0, 1, 1, 0, 1},
        /*continuation_domain=*/"continuation");
    auto runtime_plan = resolveMoEExpertOverlayRuntimePlan(plan);
    const auto *domain = runtime_plan->domainForName("rocm_domain");
    ASSERT_NE(domain, nullptr);

    TensorArena activation_arena;
    auto buffers = makeActivationBuffers(activation_arena);
    fillDirectLocalTPInputs(buffers);
    auto *actual = activation_arena.fp32({kSeqLen, kDModel});

    MockLocalTPContext actual_tp_context;
    actual_tp_context.setDevices({GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});
    actual_tp_context.setBackend(CollectiveBackendType::HOST);
    ASSERT_TRUE(runDenseLocalTPReference(*domain,
                                         actual_tp_context,
                                         layer,
                                         buffers,
                                         {true, true, false, false, true, false},
                                         actual));

    TensorArena reference_activation_arena;
    auto reference_buffers = makeActivationBuffers(reference_activation_arena);
    fillDirectLocalTPInputs(reference_buffers);
    auto *expected = reference_activation_arena.fp32({kSeqLen, kDModel});

    MockLocalTPContext reference_tp_context;
    reference_tp_context.setDevices({GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});
    reference_tp_context.setBackend(CollectiveBackendType::HOST);
    ASSERT_TRUE(runDenseLocalTPReference(*domain,
                                         reference_tp_context,
                                         reference_layer,
                                         reference_buffers,
                                         {true, true, false, false, true, false},
                                         expected));

    EXPECT_TRUE(hasNonZeroValue(actual));
    expectTensorNear(actual, expected);
}

TEST(Test__Qwen35MoEExpertOverlayGraph, AcceleratorLocalTPExecutorRequiresPreparedEngines)
{
    auto domain = rocmLocalTPRuntimeDomain();
    MockLocalTPContext tp_context;
    tp_context.setDevices({GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});
    tp_context.setBackend(CollectiveBackendType::RCCL);

    TensorArena arena;
    auto *input = arena.fp32({kSeqLen, kDModel});
    auto *routing_indices = arena.fp32({kSeqLen, kTopK});
    auto *routing_weights = arena.fp32({kSeqLen, kTopK});
    auto *output = arena.fp32({kSeqLen, kDModel});
    fill(input, 1.0f);
    fill(routing_indices, 0.0f);
    fill(routing_weights, 1.0f);

    MoEExpertOverlayLocalTPRunParams run;
    run.input = input;
    run.routing_indices = routing_indices;
    run.routing_weights = routing_weights;
    run.output = output;
    run.seq_len = kSeqLen;
    run.d_model = kDModel;
    run.num_experts = kNumExperts;
    run.top_k = kTopK;
    run.expert_intermediate = kIntermediate;
    run.layer_idx = 0;
    run.expert_mask = std::vector<bool>(kNumExperts, false);
    run.expert_mask[0] = true;

    MoEExpertOverlayLocalTPDiagnostics diagnostics;
    EXPECT_FALSE(MoEExpertOverlayLocalTPExecutor::runTensorParallelExperts(
        domain,
        tp_context,
        run,
        &diagnostics));
    EXPECT_EQ(tp_context.allreduceCallCount(), 0);
}

TEST(Test__Qwen35MoEExpertOverlayGraph, LocalTPExecutorUsesPreparedEnginesWhenProvided)
{
    auto domain = cpuLocalTPRuntimeDomain();
    MockLocalTPContext tp_context;
    tp_context.setDevices({GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)});
    tp_context.setBackend(CollectiveBackendType::HOST);

    TensorArena arena;
    auto *input = arena.fp32({kSeqLen, kDModel});
    auto *routing_indices = arena.fp32({kSeqLen, kTopK});
    auto *routing_weights = arena.fp32({kSeqLen, kTopK});
    auto *output = arena.fp32({kSeqLen, kDModel});
    fill(input, 1.0f);
    fill(routing_indices, -1.0f);
    fill(routing_weights, 0.0f);
    fill(output, 0.0f);

    CountingPreparedGemm gate0(0.25f), up0(0.50f), down0(2.0f);
    CountingPreparedGemm gate1(0.25f), up1(0.50f), down1(2.0f);
    std::vector<MoEExpertOverlayLocalTPPreparedParticipant> prepared;
    prepared.push_back(makePreparedParticipant(0, DeviceId::cpu(), &gate0, &up0, &down0));
    prepared.push_back(makePreparedParticipant(1, DeviceId::cpu(), &gate1, &up1, &down1));

    std::vector<MoEExpertDispatchEntry> entries = {
        MoEExpertDispatchEntry{.token_row = 0, .route_slot = 0, .expert_id = 0, .route_weight = 0.75f},
        MoEExpertDispatchEntry{.token_row = 2, .route_slot = 1, .expert_id = 0, .route_weight = 0.25f},
    };

    MoEExpertOverlayLocalTPRunParams run;
    run.input = input;
    run.routing_indices = routing_indices;
    run.routing_weights = routing_weights;
    run.output = output;
    run.seq_len = kSeqLen;
    run.d_model = kDModel;
    run.num_experts = kNumExperts;
    run.top_k = kTopK;
    run.expert_intermediate = kIntermediate;
    run.layer_idx = 0;
    run.expert_mask = std::vector<bool>(kNumExperts, false);
    run.expert_mask[0] = true;
    run.prepared_participants = &prepared;
    std::vector<FP32Tensor *> prepared_partial_views;
    prepared_partial_views.reserve(prepared.size());
    for (auto &participant : prepared)
        prepared_partial_views.push_back(participant.partial_scratch.get());
    run.prepared_partial_views = &prepared_partial_views;
    run.dispatch_entries = &entries;

    MoEExpertOverlayLocalTPDiagnostics diagnostics;
    ASSERT_TRUE(MoEExpertOverlayLocalTPExecutor::runTensorParallelExperts(
        domain,
        tp_context,
        run,
        &diagnostics));

    EXPECT_EQ(gate0.multiply_calls, 1);
    EXPECT_EQ(up0.multiply_calls, 1);
    EXPECT_EQ(down0.fused_swiglu_calls, 1);
    EXPECT_EQ(gate1.multiply_calls, 1);
    EXPECT_EQ(up1.multiply_calls, 1);
    EXPECT_EQ(down1.fused_swiglu_calls, 1);
    EXPECT_EQ(tp_context.allreduceCallCount(), 2);
    EXPECT_EQ(diagnostics.total_routed_entries, 2);
    ASSERT_EQ(diagnostics.participants.size(), 2u);
    EXPECT_EQ(diagnostics.participants[0].executed_expert_ids, (std::vector<int>{0}));
    EXPECT_EQ(diagnostics.participants[1].executed_expert_ids, (std::vector<int>{0}));
    EXPECT_TRUE(hasNonZeroValue(output));
}

TEST(Test__Qwen35MoEExpertOverlayGraph, DenseTransferCompatibilityModeIsExplicitDebugEnvControl)
{
    EnvVarGuard dense_transfer("LLAMINAR_MOE_EP_DENSE_TRANSFER", "1");

    TensorArena weight_arena;
    auto layer = makeLayerWeights(weight_arena);
    TensorArena activation_arena;
    auto buffers = makeActivationBuffers(activation_arena);

    auto plan = makePlanWithDomains(
        cpuContinuationRocmAndCpuDomains(),
        rocmAndCpuTiers(),
        /*placement=*/{0, 0, 1, 1, 0, 1},
        /*continuation_domain=*/"continuation");

    MockLocalTPContext rocm_tp_context;
    rocm_tp_context.setDevices({GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});
    rocm_tp_context.setBackend(CollectiveBackendType::HOST);

    GraphConfig config = makeConfig(std::move(plan), {{"rocm_domain", &rocm_tp_context}});
    Qwen35MoEGraph graph_builder(config, nullptr);
    ComputeGraph graph = graph_builder.buildFFNGraph(layer, buffers, 0, kSeqLen, kBatchSize, DeviceId::cpu());

    const auto *dispatch_node = graph.getNode("layer0_moe_expert_dispatch");
    ASSERT_NE(dispatch_node, nullptr);
    const auto *dispatch_stage = dynamic_cast<const MoEExpertDispatchStage *>(dispatch_node->stage.get());
    ASSERT_NE(dispatch_stage, nullptr);
    EXPECT_EQ(dispatch_stage->params().transfer_mode, MoEExpertTransferMode::DenseFullSequence);

    const auto *cpu_node = graph.getNode("layer0_moe_expert_ffn_tier1_cpu_cold");
    ASSERT_NE(cpu_node, nullptr);
    const auto *runtime_stage = dynamic_cast<const MoEOverlayDomainRuntimeStage *>(cpu_node->stage.get());
    ASSERT_NE(runtime_stage, nullptr);
    ASSERT_TRUE(runtime_stage->params().request.has_cpu_fallback_params);
    const auto &fallback_params = runtime_stage->params().request.cpu_fallback_params;
    EXPECT_EQ(fallback_params.transfer_mode, MoEExpertTransferMode::DenseFullSequence);
    EXPECT_NE(fallback_params.dispatch_output_lifetime, nullptr);

    auto ctx = std::make_unique<llaminar2::testing::MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);
    for (const std::string &node_name : {
             "layer0_ffn_norm",
             "layer0_moe_routing",
             "layer0_moe_expert_dispatch"})
    {
        auto *node = graph.getNode(node_name);
        ASSERT_NE(node, nullptr) << node_name;
        ASSERT_TRUE(node->stage->execute(ctx.get())) << node_name;
    }

    const auto &dispatch = *dispatch_stage->params().output;
    ASSERT_EQ(dispatch.tiers.size(), 2u);
    EXPECT_EQ(dispatch.tiers[0].transfer_mode, MoEExpertTransferMode::DenseFullSequence);
    EXPECT_EQ(dispatch.tiers[1].transfer_mode, MoEExpertTransferMode::DenseFullSequence);
    EXPECT_EQ(dispatch.tiers[1].transfer_volume.totalBytes(),
              dispatch.tiers[1].transfer_volume.denseTotalBytes());
    EXPECT_GT(dispatch.tiers[1].transfer_volume.totalBytes(),
              dispatch.tiers[1].transfer_volume.sparseTotalBytes());
}

TEST(Test__Qwen35MoEExpertOverlayGraph, ActiveGpuOverlayTierRequiresPreparedExpertGemmRegistry)
{
    TensorArena weight_arena;
    auto layer = makeLayerWeights(weight_arena);
    TensorArena activation_arena;
    auto buffers = makeActivationBuffers(activation_arena);

    auto plan = makePlanWithDomains(
        acceleratorDomains(),
        acceleratorTiers(),
        /*placement=*/{0, 1, 2, 0, 1, 2},
        /*continuation_domain=*/"cuda_domain");

    Qwen35MoEGraph graph_builder(makeConfig(std::move(plan)), nullptr);
    try
    {
        (void)graph_builder.buildFFNGraph(layer, buffers, 0, kSeqLen, kBatchSize, DeviceId::cpu());
        FAIL() << "Expected missing GPU expert registry entries to fail graph construction";
    }
    catch (const std::runtime_error &e)
    {
        const std::string message = e.what();
        EXPECT_NE(message.find("tier=0"), std::string::npos) << message;
        EXPECT_NE(message.find("name=cuda_hot"), std::string::npos) << message;
        EXPECT_NE(message.find("domain=cuda_domain"), std::string::npos) << message;
        EXPECT_NE(message.find("participant=0"), std::string::npos) << message;
        EXPECT_NE(message.find("device=CUDA:0"), std::string::npos) << message;
        EXPECT_NE(message.find("expert=0 role=gate"), std::string::npos) << message;
    }
}

TEST(Test__Qwen35MoEExpertOverlayGraph, ContinuationDomainOverridesCallerDeviceForRootFlow)
{
    TensorArena weight_arena;
    auto layer = makeLayerWeights(weight_arena);
    TensorArena activation_arena;
    auto buffers = makeActivationBuffersWithSeparateMoEOutput(activation_arena);

    auto plan = makePlanWithDomains(
        {domainWith("rocm_continuation", ExpertDomainKind::LocalTP,
                    {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)},
                    ExpertDomainComputeKind::TensorParallelExperts,
                    CollectiveBackendType::RCCL),
         domain("cpu_domain")},
        {ExpertRoutedTier{.name = "cold", .domain = "cpu_domain", .priority = 0, .fallback = true}},
        /*placement=*/{0, 0, 0, 0, 0, 0},
        /*continuation_domain=*/"rocm_continuation");

    GraphConfig config = makeConfig(std::move(plan));
    config.n_layers = 1;
    config.total_n_layers = 1;
    config.default_device = DeviceId::cpu();
    Qwen35MoEGraph graph_builder(config, nullptr);

    ComputeGraph graph = graph_builder.buildFFNGraph(layer, buffers, 0, kSeqLen, kBatchSize, DeviceId::cpu());

    const auto *norm_node = graph.getNode("layer0_ffn_norm");
    const auto *routing_node = graph.getNode("layer0_moe_routing");
    const auto *combine_node = graph.getNode("layer0_moe_combine");
    const auto *residual_node = graph.getNode("layer0_ffn_residual");
    ASSERT_NE(norm_node, nullptr);
    ASSERT_NE(routing_node, nullptr);
    ASSERT_NE(combine_node, nullptr);
    ASSERT_NE(residual_node, nullptr);

    EXPECT_EQ(norm_node->device, DeviceId::rocm(0));
    EXPECT_EQ(norm_node->stage->device(), DeviceId::rocm(0));
    EXPECT_EQ(routing_node->device, DeviceId::rocm(0));
    EXPECT_EQ(routing_node->stage->device(), DeviceId::rocm(0));
    EXPECT_EQ(combine_node->device, DeviceId::rocm(0));
    EXPECT_EQ(combine_node->stage->device(), DeviceId::rocm(0));
    EXPECT_EQ(residual_node->device, DeviceId::rocm(0));
    EXPECT_EQ(residual_node->stage->device(), DeviceId::rocm(0));

    const auto *cpu_tier_node = graph.getNode("layer0_moe_expert_ffn_tier0_cold");
    ASSERT_NE(cpu_tier_node, nullptr);
    EXPECT_EQ(cpu_tier_node->device, DeviceId::cpu());
    EXPECT_EQ(cpu_tier_node->stage->device(), DeviceId::cpu());
}

TEST(Test__Qwen35MoEExpertOverlayGraph, LayoutASharedExpertRunsOnRocmContinuationTier)
{
    TensorArena weight_arena;
    auto layer = makeLayerWeightsWithSharedExpert(weight_arena);
    TensorArena activation_arena;
    auto buffers = makeActivationBuffersWithSeparateMoEOutput(activation_arena);

    auto plan = makePlanWithDomains(
        {domainWith("rocm_hot", ExpertDomainKind::LocalTP,
                    {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)},
                    ExpertDomainComputeKind::TensorParallelExperts,
                    CollectiveBackendType::RCCL),
         domain("cpu_cold")},
        {ExpertRoutedTier{.name = "cold", .domain = "cpu_cold", .priority = 0, .fallback = true}},
        /*placement=*/{0, 0, 0, 0, 0, 0},
        /*continuation_domain=*/"rocm_hot");

    Qwen35MoEGraph graph_builder(makeConfig(std::move(plan)), nullptr);
    ComputeGraph graph = graph_builder.buildFFNGraph(layer, buffers, 0, kSeqLen, kBatchSize, DeviceId::cpu());

    const auto *shared_ffn = graph.getNode("layer0_shared_expert_ffn");
    const auto *shared_gate = graph.getNode("layer0_shared_expert_gate");
    const auto *combine = graph.getNode("layer0_moe_combine");
    ASSERT_NE(shared_ffn, nullptr);
    ASSERT_NE(shared_gate, nullptr);
    ASSERT_NE(combine, nullptr);

    EXPECT_EQ(shared_ffn->device, DeviceId::rocm(0));
    EXPECT_EQ(shared_ffn->stage->device(), DeviceId::rocm(0));
    EXPECT_EQ(shared_gate->device, DeviceId::rocm(0));
    EXPECT_EQ(shared_gate->stage->device(), DeviceId::rocm(0));
    EXPECT_EQ(combine->device, DeviceId::rocm(0));
    EXPECT_EQ(combine->stage->device(), DeviceId::rocm(0));
}

TEST(Test__Qwen35MoEExpertOverlayGraph, LayoutBSharedExpertRunsOnCudaContinuationTier)
{
    TensorArena weight_arena;
    auto layer = makeLayerWeightsWithSharedExpert(weight_arena);
    TensorArena activation_arena;
    auto buffers = makeActivationBuffersWithSeparateMoEOutput(activation_arena);

    auto plan = makePlanWithDomains(
        {domainWith("cuda_fast", ExpertDomainKind::SingleDevice,
                    {GlobalDeviceAddress::cuda(0)},
                    ExpertDomainComputeKind::ReplicatedExperts,
                    CollectiveBackendType::NCCL),
         domain("cpu_cold")},
        {ExpertRoutedTier{.name = "cold", .domain = "cpu_cold", .priority = 0, .fallback = true}},
        /*placement=*/{0, 0, 0, 0, 0, 0},
        /*continuation_domain=*/"cuda_fast");

    Qwen35MoEGraph graph_builder(makeConfig(std::move(plan)), nullptr);
    ComputeGraph graph = graph_builder.buildFFNGraph(layer, buffers, 0, kSeqLen, kBatchSize, DeviceId::cpu());

    const auto *shared_ffn = graph.getNode("layer0_shared_expert_ffn");
    const auto *shared_gate = graph.getNode("layer0_shared_expert_gate");
    const auto *combine = graph.getNode("layer0_moe_combine");
    ASSERT_NE(shared_ffn, nullptr);
    ASSERT_NE(shared_gate, nullptr);
    ASSERT_NE(combine, nullptr);

    EXPECT_EQ(shared_ffn->device, DeviceId::cuda(0));
    EXPECT_EQ(shared_ffn->stage->device(), DeviceId::cuda(0));
    EXPECT_EQ(shared_gate->device, DeviceId::cuda(0));
    EXPECT_EQ(shared_gate->stage->device(), DeviceId::cuda(0));
    EXPECT_EQ(combine->device, DeviceId::cuda(0));
    EXPECT_EQ(combine->stage->device(), DeviceId::cuda(0));
}

TEST(Test__Qwen35MoEExpertOverlayGraph, SharedExpertDifferentFromContinuationReturnsThroughCrossDomainReduce)
{
    TensorArena weight_arena;
    auto layer = makeLayerWeightsWithSharedExpert(weight_arena);
    TensorArena activation_arena;
    auto buffers = makeActivationBuffers(activation_arena);

    auto plan = makePlanWithDomains(
        {domainWith("cuda_fast", ExpertDomainKind::SingleDevice,
                    {GlobalDeviceAddress::cuda(0)},
                    ExpertDomainComputeKind::ReplicatedExperts,
                    CollectiveBackendType::NCCL),
         domainWith("rocm_shared", ExpertDomainKind::LocalTP,
                    {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)},
                    ExpertDomainComputeKind::TensorParallelExperts,
                    CollectiveBackendType::RCCL),
         domain("cpu_cold")},
        {ExpertRoutedTier{.name = "cold", .domain = "cpu_cold", .priority = 0, .fallback = true}},
        /*placement=*/{0, 0, 0, 0, 0, 0},
        /*continuation_domain=*/"cuda_fast",
        /*shared_expert_domain=*/"rocm_shared");

    Qwen35MoEGraph graph_builder(makeConfig(std::move(plan)), nullptr);
    ComputeGraph graph = graph_builder.buildFFNGraph(layer, buffers, 0, kSeqLen, kBatchSize, DeviceId::cpu());

    const auto *shared_ffn = graph.getNode("layer0_shared_expert_ffn");
    const auto *shared_gate = graph.getNode("layer0_shared_expert_gate");
    const auto *reduce_node = graph.getNode("layer0_moe_expert_parallel_reduce");
    ASSERT_NE(shared_ffn, nullptr);
    ASSERT_NE(shared_gate, nullptr);
    ASSERT_NE(reduce_node, nullptr);

    EXPECT_EQ(shared_ffn->device, DeviceId::rocm(0));
    EXPECT_EQ(shared_gate->device, DeviceId::rocm(0));
    EXPECT_EQ(reduce_node->device, DeviceId::cuda(0));
    const auto *reduce_stage = dynamic_cast<const MoEExpertParallelReduceStage *>(reduce_node->stage.get());
    ASSERT_NE(reduce_stage, nullptr);
    EXPECT_EQ(reduce_stage->coherencePolicy(), CoherencePolicy::NONE);
    const auto reduce_contract = reduce_stage->bufferContract();
    ASSERT_EQ(reduce_contract.outputs.size(), 1u);
    EXPECT_EQ(reduce_contract.outputs[0].id, BufferId::MOE_COMBINED_OUTPUT);
    EXPECT_EQ(reduce_contract.outputs[0].access, BufferAccess::WRITE);
    EXPECT_EQ(reduce_stage->params().mode, MoEExpertParallelReduceMode::ContinuationDeviceOptimized);
    EXPECT_EQ(reduce_stage->params().continuation_domain, "cuda_fast");
    EXPECT_EQ(reduce_stage->params().continuation_device, DeviceId::cuda(0));
    ASSERT_EQ(reduce_stage->params().partial_infos.size(), 2u);
    EXPECT_EQ(reduce_stage->params().partial_infos[0].source_domain, "cpu_cold");
    EXPECT_EQ(reduce_stage->params().partial_infos[1].name, "shared_expert");
    EXPECT_EQ(reduce_stage->params().partial_infos[1].source_domain, "rocm_shared");
    EXPECT_EQ(reduce_stage->params().partial_infos[1].source_device, DeviceId::rocm(0));
}

TEST(Test__Qwen35MoEExpertOverlayGraph, RemoteSharedExpertDomainFailsDuringGraphBuild)
{
    TensorArena weight_arena;
    auto layer = makeLayerWeightsWithSharedExpert(weight_arena);
    TensorArena activation_arena;
    auto buffers = makeActivationBuffers(activation_arena);

    auto remote_shared = domainWith(
        "remote_shared",
        ExpertDomainKind::SingleDevice,
        {GlobalDeviceAddress::cuda(0, 0, "remote-node")},
        ExpertDomainComputeKind::ReplicatedExperts,
        CollectiveBackendType::NCCL);
    auto plan = makePlanWithDomains(
        {domainWith("cuda_fast", ExpertDomainKind::SingleDevice,
                    {GlobalDeviceAddress::cuda(0)},
                    ExpertDomainComputeKind::ReplicatedExperts,
                    CollectiveBackendType::NCCL),
         remote_shared,
         domain("cpu_cold")},
        {ExpertRoutedTier{.name = "cold", .domain = "cpu_cold", .priority = 0, .fallback = true}},
        /*placement=*/{0, 0, 0, 0, 0, 0},
        /*continuation_domain=*/"cuda_fast",
        /*shared_expert_domain=*/"remote_shared");

    GraphConfig config = makeConfig();
    config.moe.expert_parallel_plan = std::move(plan);
    Qwen35MoEGraph graph_builder(config, nullptr);

    try
    {
        (void)graph_builder.buildFFNGraph(layer, buffers, 0, kSeqLen, kBatchSize, DeviceId::cpu());
        FAIL() << "Expected unreachable shared expert domain to fail graph construction";
    }
    catch (const std::runtime_error &e)
    {
        const std::string message = e.what();
        EXPECT_NE(message.find("shared expert domain"), std::string::npos) << message;
        EXPECT_NE(message.find("not locally reachable"), std::string::npos) << message;
    }
}

TEST(Test__Qwen35MoEExpertOverlayGraph, RemoteContinuationDomainFailsDuringGraphBuild)
{
    TensorArena weight_arena;
    auto layer = makeLayerWeights(weight_arena);
    TensorArena activation_arena;
    auto buffers = makeActivationBuffers(activation_arena);

    auto remote_continuation = domainWith(
        "remote_continuation",
        ExpertDomainKind::SingleDevice,
        {GlobalDeviceAddress::cuda(0, 0, "remote-node")},
        ExpertDomainComputeKind::ReplicatedExperts,
        CollectiveBackendType::NCCL);
    auto plan = makePlanWithDomains(
        {remote_continuation, domain("cpu_domain")},
        {ExpertRoutedTier{.name = "cold", .domain = "cpu_domain", .priority = 0, .fallback = true}},
        /*placement=*/{0, 0, 0, 0, 0, 0},
        /*continuation_domain=*/"remote_continuation");

    GraphConfig config = makeConfig();
    config.moe.expert_parallel_plan = std::move(plan);
    Qwen35MoEGraph graph_builder(config, nullptr);

    EXPECT_THROW(
        (void)graph_builder.buildFFNGraph(layer, buffers, 0, kSeqLen, kBatchSize, DeviceId::cpu()),
        std::runtime_error);
}

} // namespace llaminar2::test