/**
 * @file Test__Qwen35MoEGraph.cpp
 * @brief Regression tests for Qwen3.5 MoE graph construction.
 */

#include <gtest/gtest.h>

#include "execution/moe/MoEExpertOverlayRuntimePlan.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/compute_stages/stages/GDNLiveStateAllGatherStage.h"
#include "execution/prefix_cache/PrefixCacheFingerprint.h"
#include "kernels/cpu/CPUHybridRingKVCache.h"
#include "models/qwen35moe/Qwen35MoEGraph.h"
#include "models/qwen35moe/Qwen35MoESchema.h"
#include "kernels/KernelFactory.h"
#include "mocks/MockLocalTPContext.h"
#include "mocks/MockMPIContext.h"
#include "tensors/TensorSlice.h"
#include "utils/TestTensorFactory.h"

#include <algorithm>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    bool hasDependency(const ComputeGraph &graph, const std::string &node_name, const std::string &dependency)
    {
        const auto *node = graph.getNode(node_name);
        if (!node)
            return false;
        return std::find(node->dependencies.begin(), node->dependencies.end(), dependency) != node->dependencies.end();
    }

    bool contractReads(const StageBufferContract &contract, BufferId id)
    {
        const auto reads = contract.allArenaReads();
        return std::any_of(reads.begin(), reads.end(),
                           [id](const BufferBinding &binding) { return binding.id == id; });
    }

    bool contractWrites(const StageBufferContract &contract, BufferId id)
    {
        const auto writes = contract.allWrites();
        return std::any_of(writes.begin(), writes.end(),
                           [id](const BufferBinding &binding) { return binding.id == id; });
    }

    bool hasFingerprintField(
        const PrefixFingerprintMaterial &material,
        const std::string &name,
        const std::string &value)
    {
        return std::any_of(material.moe.begin(),
                           material.moe.end(),
                           [&](const PrefixFingerprintField &field)
                           {
                               return field.name == name && field.value == value;
                           });
    }

    bool hasFingerprintFieldWithPrefix(
        const PrefixFingerprintMaterial &material,
        const std::string &prefix)
    {
        return std::any_of(material.moe.begin(),
                           material.moe.end(),
                           [&](const PrefixFingerprintField &field)
                           {
                               return field.name.rfind(prefix, 0) == 0;
                           });
    }

    GraphConfig makeMoEConfig(ITPContext *tp_ctx = nullptr)
    {
        GraphConfig config;
        config.n_layers = 2;
        config.total_n_layers = 2;
        config.d_model = 4;
        config.n_heads = 2;
        config.n_kv_heads = 2;
        config.head_dim = 2;
        config.d_ff = 8;
        config.vocab_size = 16;
        config.rms_norm_eps = 1e-6f;
        config.default_device = DeviceId::cpu();
        config.tp_ctx = tp_ctx;
        config.tp_device_idx = 0;
        config.moe.num_experts = 2;
        config.moe.top_k = 1;
        config.moe.intermediate_size = 3;
        config.moe.has_shared_expert = true;
        config.moe.shared_intermediate_size = 3;
        return config;
    }

    ExecutionDomainDefinition denseDomain(
        std::string name,
        ExecutionDomainScope scope,
        CollectiveBackendType backend,
        std::vector<GlobalDeviceAddress> participants)
    {
        ExecutionDomainDefinition domain;
        domain.name = std::move(name);
        domain.scope = scope;
        domain.backend = backend;
        domain.participants = std::move(participants);
        domain.compute_kind = ExecutionDomainComputeKind::APPORTIONED_EXPERTS;
        domain.owner_rank = 0;
        domain.ranks = {0};
        return domain;
    }

    ExpertComputeDomain expertDomain(
        std::string name,
        ExpertDomainKind kind,
        CollectiveBackendType backend,
        ExpertDomainComputeKind compute_kind,
        std::vector<GlobalDeviceAddress> participants,
        std::vector<int> ranks)
    {
        ExpertComputeDomain domain;
        domain.name = std::move(name);
        domain.kind = kind;
        domain.backend = backend;
        domain.compute_kind = compute_kind;
        domain.participants = std::move(participants);
        domain.world_ranks = std::move(ranks);
        domain.owner_rank = 0;
        return domain;
    }

    std::shared_ptr<MoEExpertParallelPlan> makeOverlayPlan(const std::string &routed_domain)
    {
        auto plan = std::make_shared<MoEExpertParallelPlan>();
        plan->enabled = true;
        plan->execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
        plan->continuation_domain = "continuation";
        plan->base_model_domain = "base";
        plan->shared_expert_domain = "shared";
        plan->residency_policy = ExpertResidencyPolicy::RoutedTierRebalanced;
        plan->continuation_domain_spec.domain = "continuation";
        plan->continuation_domain_spec.logical_root_participant = 0;
        plan->continuation_domain_spec.dense_tp_enabled = false;
        plan->continuation_domain_spec.hidden_layout = MoEContinuationActivationLayout::ReplicatedHidden;
        plan->continuation_domain_spec.shared_expert_uses_dense_tp = true;

        plan->dense_domains.push_back(denseDomain(
            "continuation",
            ExecutionDomainScope::SINGLE,
            CollectiveBackendType::HOST,
            {GlobalDeviceAddress::cpu(0, "node0")}));
        plan->dense_domains.push_back(denseDomain(
            "base",
            ExecutionDomainScope::SINGLE,
            CollectiveBackendType::HOST,
            {GlobalDeviceAddress::cpu(0, "node0")}));
        plan->dense_domains.push_back(denseDomain(
            "shared",
            ExecutionDomainScope::SINGLE,
            CollectiveBackendType::HOST,
            {GlobalDeviceAddress::cpu(0, "node0")}));

        plan->domains.push_back(expertDomain(
            routed_domain,
            ExpertDomainKind::NodeLocalTP,
            CollectiveBackendType::HOST,
            ExpertDomainComputeKind::ApportionedExperts,
            {GlobalDeviceAddress::cpu(0, "node0"), GlobalDeviceAddress::cpu(1, "node0")},
            {0, 1}));

        plan->routed_tiers.push_back(ExpertRoutedTier{
            .name = "cold",
            .domain = routed_domain,
            .priority = 10,
            .max_experts_per_layer = 2,
            .memory_budget_bytes = 4096,
            .fallback = true,
        });
        plan->placements.push_back(ExpertLayerPlacement{
            .layer = 0,
            .routed_expert_tier = {0, 0},
        });
        return plan;
    }

    std::shared_ptr<MoEExpertParallelPlan> makeLocalTPApportionedOverlayPlan(const std::string &domain_name)
    {
        auto plan = std::make_shared<MoEExpertParallelPlan>();
        plan->enabled = true;
        plan->execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
        plan->continuation_domain = domain_name;
        plan->base_model_domain = domain_name;
        plan->shared_expert_domain = domain_name;
        plan->residency_policy = ExpertResidencyPolicy::StaticById;
        plan->continuation_domain_spec.domain = domain_name;
        plan->continuation_domain_spec.logical_root_participant = 0;
        plan->continuation_domain_spec.dense_tp_enabled = false;
        plan->continuation_domain_spec.hidden_layout = MoEContinuationActivationLayout::ReplicatedHidden;
        plan->continuation_domain_spec.shared_expert_uses_dense_tp = true;

        plan->domains.push_back(expertDomain(
            domain_name,
            ExpertDomainKind::LocalTP,
            CollectiveBackendType::HOST,
            ExpertDomainComputeKind::ApportionedExperts,
            {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)},
            {}));

        plan->routed_tiers.push_back(ExpertRoutedTier{
            .name = "hot",
            .domain = domain_name,
            .priority = 0,
            .max_experts_per_layer = 2,
            .memory_budget_bytes = 4096,
            .fallback = true,
        });
        plan->placements.push_back(ExpertLayerPlacement{
            .layer = 0,
            .routed_expert_tier = {0, 0},
        });
        return plan;
    }

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

        TensorSlice *rowParallelFP32(std::vector<size_t> shape, int rank = 0, int world_size = 2)
        {
            auto inner = std::make_shared<FP32Tensor>(shape);
            auto metadata = SliceMetadata::forRowParallel(
                shape.at(0), shape.at(1), rank, world_size,
                /*inner_is_presliced=*/false);
            auto tensor = std::make_shared<TensorSlice>(std::move(inner), std::move(metadata));
            auto *ptr = tensor.get();
            tensors_.push_back(std::move(tensor));
            return ptr;
        }

    private:
        std::vector<std::shared_ptr<TensorBase>> tensors_;
    };

    class TestableQwen35MoEGraph : public Qwen35MoEGraph
    {
    public:
        using Qwen35MoEGraph::Qwen35MoEGraph;

        ComputeGraph buildFFNGraphForTokenCount(
            const LayerWeights &layer,
            ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            int batch_size,
            DeviceId device)
        {
            DecodeReplicatedDenseScope decode_dense_scope(*this, seq_len * batch_size);
            ComputeGraph graph = buildFFNGraph(layer, buffers, layer_idx, seq_len, batch_size, device);
            return graph;
        }

        ComputeGraph buildAttentionGraphForTokenCount(
            const LayerWeights &layer,
            ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            int batch_size,
            IKVCache *kv_cache,
            const int *position_ids,
            DeviceId device)
        {
            DecodeReplicatedDenseScope decode_dense_scope(*this, seq_len * batch_size);
            return buildAttentionGraph(
                layer, buffers, layer_idx, seq_len, batch_size,
                kv_cache, position_ids, device);
        }

        LayerWeights selectedLayerWeightsForTokenCount(int layer_idx, int total_tokens)
        {
            DecodeReplicatedDenseScope decode_dense_scope(*this, total_tokens);
            return layerWeightsForGraph(layer_idx);
        }

        void setDecodeHistogramForTesting(DecodeExpertHistogram *histogram)
        {
            config_.moe.decode_histogram = histogram;
        }

        void registerRuntimeTableHistogramSyncForTesting(
            const std::string &key,
            IMoERuntimeTable *table,
            bool enabled = true)
        {
            registerRuntimeTableHistogramSyncIfNeeded(key, table, enabled);
        }
    };

    class FakeRuntimeTable : public IMoERuntimeTable
    {
    public:
        std::vector<uint64_t> counts{0, 0, 0, 0};
        int sync_calls = 0;

        DeviceMoELayerRuntime *deviceLayerState(int) override { return nullptr; }
        int layerCount() const override { return 1; }
        const DeviceMoELayerRuntime &hostLayerState(int) const override
        {
            static DeviceMoELayerRuntime state{};
            return state;
        }
        bool prepareInactiveBank(int, const MoEPlacementUpdate &) override { return false; }
        bool flipActiveBank(int, uint32_t, void *) override { return false; }
        bool hasPrefillRouteScratchCapacity(int, int) const override { return false; }
        void recordDecodeHistogramProducerStream(void *stream) override
        {
            producer_stream = stream;
        }
        void *decodeHistogramProducerStream() const override
        {
            return producer_stream;
        }
        bool syncDecodeHistogramToHost(
            DecodeExpertHistogram &histogram,
            void * = nullptr,
            bool reset_runtime_counts = true) override
        {
            ++sync_calls;
            histogram.mergeLayerCounts(0, counts.data(), static_cast<int>(counts.size()));
            if (reset_runtime_counts)
                std::fill(counts.begin(), counts.end(), 0);
            return true;
        }
        void resetDecodeHistogramCounts(void * = nullptr) override
        {
            std::fill(counts.begin(), counts.end(), 0);
        }
        void resetDecodeRuntimeState(void * = nullptr) override {}

        void *producer_stream = nullptr;
    };

    ModelWeightBindings makeDecodeDenseBindingSource()
    {
        ModelWeightBindings bindings;
        bindings.get_layer_weights = [](int)
        {
            return LayerWeightBindings{};
        };
        return bindings;
    }

    LayerWeights makeMoELayerWeights(TensorArena &arena)
    {
        LayerWeights layer;
        layer.ffn_norm = arena.fp32({4});
        layer.moe_gate = arena.fp32({2, 4});

        // Expert tensor shapes follow GGUF order: [cols, rows, experts].
        layer.moe_gate_exps = arena.fp32({4, 3, 2});
        layer.moe_up_exps = arena.fp32({4, 3, 2});
        layer.moe_down_exps = arena.fp32({3, 4, 2});

        layer.shared_expert_gate = arena.fp32({3, 4});
        layer.shared_expert_up = arena.fp32({3, 4});
        layer.shared_expert_down = arena.fp32({4, 3});
        return layer;
    }

    LayerWeights makeDenseFFNLayerWeights(TensorArena &arena, const GraphConfig &config)
    {
        LayerWeights layer;
        layer.ffn_norm = arena.fp32({static_cast<size_t>(config.d_model)});
        layer.gate_proj = arena.fp32({static_cast<size_t>(config.d_ff), static_cast<size_t>(config.d_model)});
        layer.up_proj = arena.fp32({static_cast<size_t>(config.d_ff), static_cast<size_t>(config.d_model)});
        layer.down_proj = arena.fp32({static_cast<size_t>(config.d_model), static_cast<size_t>(config.d_ff)});
        return layer;
    }

    LayerWeights makeFALayerWeights(TensorArena &arena, const GraphConfig &config)
    {
        LayerWeights layer;
        const int q_width = config.n_heads * config.head_dim * 2;
        const int kv_width = config.n_kv_heads * config.head_dim;

        layer.attn_norm = arena.fp32({static_cast<size_t>(config.d_model)});
        layer.wq = arena.fp32({static_cast<size_t>(q_width), static_cast<size_t>(config.d_model)});
        layer.wk = arena.fp32({static_cast<size_t>(kv_width), static_cast<size_t>(config.d_model)});
        layer.wv = arena.fp32({static_cast<size_t>(kv_width), static_cast<size_t>(config.d_model)});
        layer.wo = arena.fp32({static_cast<size_t>(config.d_model), static_cast<size_t>(config.d_model)});
        layer.q_norm = arena.fp32({static_cast<size_t>(config.head_dim)});
        layer.k_norm = arena.fp32({static_cast<size_t>(config.head_dim)});
        return layer;
    }

    GraphConfig makeGDNTPConfig(ITPContext *tp_ctx)
    {
        GraphConfig config = makeMoEConfig(tp_ctx);
        config.n_layers = 1;
        config.total_n_layers = 1;
        config.d_model = 4;
        config.n_heads = 2;
        config.head_dim = 2;
        config.layer_types = {"gdn"};
        config.gdn.conv_kernel_size = 4;
        config.gdn.state_size = 2;
        config.gdn.inner_size = 4;
        config.gdn.group_count = 2;
        config.gdn.time_step_rank = 2;
        config.qkv_column_parallel = true;
        config.local_n_heads = 1;
        config.local_n_kv_heads = 1;
        return config;
    }

    GraphConfig makeGDNTPModularConfig(ITPContext *tp_ctx)
    {
        GraphConfig config = makeGDNTPConfig(tp_ctx);
        config.n_heads = 4;
        config.local_n_heads = 2;
        config.gdn.inner_size = 8;
        config.gdn.group_count = 2;
        config.gdn.time_step_rank = 4;
        return config;
    }

    HybridKVCacheConfig makeGDNTPHybridConfig(const GraphConfig &config)
    {
        HybridKVCacheConfig hybrid;
        hybrid.layer_types = config.layer_types;
        hybrid.gdn_conv_kernel_size = config.gdn.conv_kernel_size;
        hybrid.gdn_state_size = config.gdn.state_size;
        hybrid.gdn_inner_size = config.gdn.inner_size;
        hybrid.gdn_group_count = config.gdn.group_count;
        hybrid.gdn_time_step_rank = config.gdn.time_step_rank;
        hybrid.n_heads = config.n_heads;
        hybrid.local_n_heads = config.local_n_heads;
        return hybrid;
    }

    LayerWeights makeGDNTPLayerWeights(
        TensorArena &arena,
        const GraphConfig &config,
        int value_heads,
        bool row_parallel_out)
    {
        const size_t d = static_cast<size_t>(config.d_model);
        const size_t d_state = static_cast<size_t>(config.gdn.state_size);
        const size_t v_heads = static_cast<size_t>(value_heads);
        const size_t value_dim = v_heads * d_state;
        const size_t qkv_dim = 2 * v_heads * d_state + value_dim;
        const size_t kernel = static_cast<size_t>(config.gdn.conv_kernel_size);

        LayerWeights layer;
        layer.attn_norm = arena.fp32({d});
        layer.attn_qkv = arena.fp32({qkv_dim, d});
        layer.attn_gate = arena.fp32({value_dim, d});
        layer.ssm_alpha = arena.fp32({v_heads, d});
        layer.ssm_beta = arena.fp32({v_heads, d});
        layer.ssm_conv1d = arena.fp32({kernel, qkv_dim});
        layer.ssm_dt_bias = arena.fp32({v_heads});
        layer.ssm_a = arena.fp32({v_heads});
        layer.ssm_norm = arena.fp32({d_state});
        layer.ssm_out = row_parallel_out
                            ? static_cast<TensorBase *>(arena.rowParallelFP32({d, value_dim}))
                            : static_cast<TensorBase *>(arena.fp32({d, value_dim}));
        return layer;
    }

    LayerWeights makeGDNTPModularLayerWeights(
        TensorArena &arena,
        const GraphConfig &config,
        int key_heads,
        int value_heads,
        bool row_parallel_out)
    {
        const size_t d = static_cast<size_t>(config.d_model);
        const size_t d_state = static_cast<size_t>(config.gdn.state_size);
        const size_t k_heads = static_cast<size_t>(key_heads);
        const size_t v_heads = static_cast<size_t>(value_heads);
        const size_t qk_dim = 2 * k_heads * d_state;
        const size_t value_dim = v_heads * d_state;
        const size_t qkv_dim = qk_dim + value_dim;
        const size_t kernel = static_cast<size_t>(config.gdn.conv_kernel_size);

        LayerWeights layer;
        layer.attn_norm = arena.fp32({d});
        layer.attn_qkv = arena.fp32({qkv_dim, d});
        layer.attn_gate = arena.fp32({value_dim, d});
        layer.ssm_alpha = arena.fp32({v_heads, d});
        layer.ssm_beta = arena.fp32({v_heads, d});
        layer.ssm_conv1d = arena.fp32({kernel, qkv_dim});
        layer.ssm_dt_bias = arena.fp32({v_heads});
        layer.ssm_a = arena.fp32({v_heads});
        layer.ssm_norm = arena.fp32({d_state});
        layer.ssm_out = row_parallel_out
                            ? static_cast<TensorBase *>(arena.rowParallelFP32({d, value_dim}))
                            : static_cast<TensorBase *>(arena.fp32({d, value_dim}));
        return layer;
    }

    ActivationBuffers makeGDNTPActivationBuffers(
        TensorArena &arena,
        int tokens,
        const GraphConfig &config,
        int value_heads)
    {
        const size_t rows = static_cast<size_t>(tokens);
        const size_t d = static_cast<size_t>(config.d_model);
        const size_t d_state = static_cast<size_t>(config.gdn.state_size);
        const size_t v_heads = static_cast<size_t>(value_heads);
        const size_t value_dim = v_heads * d_state;
        const size_t qkv_dim = 2 * v_heads * d_state + value_dim;

        ActivationBuffers buffers;
        buffers.current_hidden = arena.fp32({rows, d});
        buffers.normalized = arena.fp32({rows, d});
        buffers.attn_output = arena.fp32({rows, value_dim});
        buffers.attn_proj = arena.fp32({rows, d});
        buffers.extensions[BufferId::GDN_QKV] = arena.fp32({rows, qkv_dim});
        buffers.extensions[BufferId::GDN_Z] = arena.fp32({rows, value_dim});
        buffers.extensions[BufferId::GDN_ALPHA] = arena.fp32({rows, v_heads});
        buffers.extensions[BufferId::GDN_BETA] = arena.fp32({rows, v_heads});
        return buffers;
    }

    ActivationBuffers makeGDNTPModularActivationBuffers(
        TensorArena &arena,
        int tokens,
        const GraphConfig &config,
        int key_heads,
        int value_heads)
    {
        const size_t rows = static_cast<size_t>(tokens);
        const size_t d = static_cast<size_t>(config.d_model);
        const size_t d_state = static_cast<size_t>(config.gdn.state_size);
        const size_t k_heads = static_cast<size_t>(key_heads);
        const size_t v_heads = static_cast<size_t>(value_heads);
        const size_t value_dim = v_heads * d_state;
        const size_t qkv_dim = 2 * k_heads * d_state + value_dim;

        ActivationBuffers buffers;
        buffers.current_hidden = arena.fp32({rows, d});
        buffers.normalized = arena.fp32({rows, d});
        buffers.attn_output = arena.fp32({rows, value_dim});
        buffers.attn_proj = arena.fp32({rows, d});
        buffers.extensions[BufferId::GDN_QKV] = arena.fp32({rows, qkv_dim});
        buffers.extensions[BufferId::GDN_Z] = arena.fp32({rows, value_dim});
        buffers.extensions[BufferId::GDN_ALPHA] = arena.fp32({rows, v_heads});
        buffers.extensions[BufferId::GDN_BETA] = arena.fp32({rows, v_heads});
        return buffers;
    }

    WeightBinding makeTestBinding(TensorBase *tensor)
    {
        WeightBinding binding;
        binding.tensor = tensor;
        if (tensor && tensor->shape().size() >= 2)
        {
            binding.slice.source_rows = tensor->shape()[0];
            binding.slice.source_cols = tensor->shape()[1];
            binding.slice.row_count = tensor->shape()[0];
            binding.slice.col_count = tensor->shape()[1];
        }
        return binding;
    }

    ActivationBuffers makeActivationBuffers(TensorArena &arena, int tokens, int d_model, int num_experts, int top_k)
    {
        ActivationBuffers buffers;
        buffers.attn_proj = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(d_model)});
        buffers.current_hidden = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(d_model)});
        buffers.normalized = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(d_model)});

        buffers.extensions[BufferId::MOE_EXPERT_INDICES] = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(top_k)});
        buffers.extensions[BufferId::MOE_EXPERT_WEIGHTS] = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(top_k)});
        buffers.extensions[BufferId::MOE_COMBINED_OUTPUT] = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(d_model)});
        buffers.extensions[BufferId::MOE_SHARED_EXPERT_OUTPUT] = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(d_model)});
        buffers.extensions[BufferId::MOE_GATE_SCRATCH] = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(num_experts)});
        buffers.extensions[BufferId::MOE_UP_SCRATCH] = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(num_experts)});
        return buffers;
    }

    ActivationBuffers makeDenseFFNBuffers(TensorArena &arena, int tokens, const GraphConfig &config)
    {
        ActivationBuffers buffers;
        buffers.attn_proj = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(config.d_model)});
        buffers.current_hidden = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(config.d_model)});
        buffers.normalized = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(config.d_model)});
        buffers.gate = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(config.d_ff)});
        buffers.up = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(config.d_ff)});
        return buffers;
    }

    ActivationBuffers makeFAActivationBuffers(TensorArena &arena, int tokens, const GraphConfig &config)
    {
        ActivationBuffers buffers;
        const int q_width = config.n_heads * config.head_dim;
        const int q_raw_width = q_width * 2;
        const int kv_width = config.n_kv_heads * config.head_dim;

        buffers.current_hidden = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(config.d_model)});
        buffers.normalized = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(config.d_model)});
        buffers.Q = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(q_width)});
        buffers.K = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(kv_width)});
        buffers.V = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(kv_width)});
        buffers.attn_output = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(q_width)});
        buffers.attn_proj = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(config.d_model)});
        buffers.extensions[BufferId::FA_Q_RAW] = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(q_raw_width)});
        buffers.extensions[BufferId::FA_GATE] = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(q_width)});
        return buffers;
    }

    ModelWeights makeFullForwardModelWeights(TensorArena &arena, const GraphConfig &config)
    {
        LayerWeights layer = makeFALayerWeights(arena, config);
        LayerWeights dense = makeDenseFFNLayerWeights(arena, config);
        layer.ffn_norm = dense.ffn_norm;
        layer.gate_proj = dense.gate_proj;
        layer.up_proj = dense.up_proj;
        layer.down_proj = dense.down_proj;

        ModelWeights weights;
        weights.embedding_table = arena.fp32({static_cast<size_t>(config.vocab_size),
                                              static_cast<size_t>(config.d_model)});
        weights.final_norm = arena.fp32({static_cast<size_t>(config.d_model)});
        weights.lm_head = arena.fp32({static_cast<size_t>(config.vocab_size),
                                      static_cast<size_t>(config.d_model)});
        weights.get_layer_weights = [layer](int layer_idx)
        {
            return layer_idx == 0 ? layer : LayerWeights{};
        };
        return weights;
    }

    ModelBuffers makeFullForwardModelBuffers(TensorArena &arena, int tokens, const GraphConfig &config)
    {
        ModelBuffers model_buffers;
        model_buffers.current_hidden = arena.fp32({static_cast<size_t>(tokens),
                                                   static_cast<size_t>(config.d_model)});
        model_buffers.logits = arena.fp32({static_cast<size_t>(tokens),
                                           static_cast<size_t>(config.vocab_size)});
        model_buffers.logits_local = arena.fp32({static_cast<size_t>(tokens),
                                                 static_cast<size_t>(std::max(1, config.vocab_local))});

        model_buffers.layer_buffers = makeFAActivationBuffers(arena, tokens, config);
        model_buffers.layer_buffers.current_hidden = model_buffers.current_hidden;
        model_buffers.layer_buffers.gate = arena.fp32({static_cast<size_t>(tokens),
                                                       static_cast<size_t>(config.d_ff)});
        model_buffers.layer_buffers.up = arena.fp32({static_cast<size_t>(tokens),
                                                     static_cast<size_t>(config.d_ff)});
        model_buffers.layer_buffers.ffn_output = arena.fp32({static_cast<size_t>(tokens),
                                                             static_cast<size_t>(config.d_ff)});
        model_buffers.layer_buffers.workspace_scores = arena.fp32({static_cast<size_t>(tokens * config.n_heads),
                                                                   static_cast<size_t>(tokens)});
        model_buffers.layer_buffers.workspace_context = arena.fp32({static_cast<size_t>(tokens * config.n_heads),
                                                                    static_cast<size_t>(config.head_dim)});
        model_buffers.layer_buffers.workspace_mask = arena.fp32({static_cast<size_t>(tokens),
                                                                 static_cast<size_t>(tokens)});
        return model_buffers;
    }
}

TEST(Test__Qwen35MoEGraph, ReplicatedRoutedExpertOutputFeedsCombineDirectlyUnderTP)
{
    auto tp_ctx = std::make_unique<MockLocalTPContext>();
    tp_ctx->setDevices({GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()});
    tp_ctx->setBackend(CollectiveBackendType::HOST);

    GraphConfig config = makeMoEConfig(tp_ctx.get());
    config.moe.expert_mode = MoEExpertMode::ReplicatedExperts;
    config.moe.local_expert_count = -1;
    Qwen35MoEGraph graph_builder(config, nullptr);

    TensorArena arena;
    auto layer = makeMoELayerWeights(arena);
    auto buffers = makeActivationBuffers(arena, /*tokens=*/2, config.d_model, config.moe.num_experts, config.moe.top_k);

    ComputeGraph graph = graph_builder.buildFFNGraph(layer, buffers, /*layer_idx=*/0, /*seq_len=*/2, /*batch_size=*/1, DeviceId::cpu());

    ASSERT_NE(graph.getNode("layer0_moe_expert_ffn"), nullptr);
    ASSERT_EQ(graph.getNode("layer0_moe_expert_allreduce"), nullptr)
        << "Replicated MoE expert weights already produce a full routed expert output per rank";
    ASSERT_NE(graph.getNode("layer0_moe_combine"), nullptr);

    EXPECT_TRUE(hasDependency(graph, "layer0_moe_combine", "layer0_moe_expert_ffn"));
}

TEST(Test__Qwen35MoEGraph, SingleDeviceSharedGateFusesMoECombine)
{
    GraphConfig config = makeMoEConfig();
    Qwen35MoEGraph graph_builder(config, nullptr);

    TensorArena arena;
    auto layer = makeMoELayerWeights(arena);
    layer.shared_expert_gate_inp = arena.fp32({static_cast<size_t>(config.d_model)});
    auto buffers = makeActivationBuffers(arena, /*tokens=*/2, config.d_model,
                                         config.moe.num_experts, config.moe.top_k);

    ComputeGraph graph = graph_builder.buildFFNGraph(
        layer, buffers, /*layer_idx=*/0, /*seq_len=*/2,
        /*batch_size=*/1, DeviceId::cpu());

    ASSERT_NE(graph.getNode("layer0_moe_expert_ffn"), nullptr);
    ASSERT_NE(graph.getNode("layer0_shared_expert_gate"), nullptr);
    EXPECT_EQ(graph.getNode("layer0_moe_combine"), nullptr)
        << "Single-device MoE should fuse shared-expert gating and routed-output combine";

    EXPECT_TRUE(hasDependency(graph, "layer0_shared_expert_gate", "layer0_moe_expert_ffn"));
    EXPECT_TRUE(hasDependency(graph, "layer0_shared_expert_gate", "layer0_shared_expert_ffn"));

    const auto contract = graph.getNode("layer0_shared_expert_gate")->stage->bufferContract();
    EXPECT_TRUE(contractReads(contract, BufferId::MOE_SHARED_EXPERT_OUTPUT));
    EXPECT_TRUE(contractReads(contract, BufferId::MOE_COMBINED_OUTPUT));
    EXPECT_TRUE(contractWrites(contract, BufferId::ATTN_PROJ));
}

TEST(Test__Qwen35MoEGraph, ExpertParallelRoutedExpertOutputAllreducesUnderTP)
{
    auto tp_ctx = std::make_unique<MockLocalTPContext>();
    tp_ctx->setDevices({GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()});
    tp_ctx->setBackend(CollectiveBackendType::HOST);

    GraphConfig config = makeMoEConfig(tp_ctx.get());
    config.moe.expert_mode = MoEExpertMode::ApportionedExperts;
    config.moe.local_expert_start = 0;
    config.moe.local_expert_count = 1;
    Qwen35MoEGraph graph_builder(config, nullptr);

    TensorArena arena;
    auto layer = makeMoELayerWeights(arena);
    auto buffers = makeActivationBuffers(arena, /*tokens=*/2, config.d_model, config.moe.num_experts, config.moe.top_k);

    ComputeGraph graph = graph_builder.buildFFNGraph(layer, buffers, /*layer_idx=*/0, /*seq_len=*/2, /*batch_size=*/1, DeviceId::cpu());

    ASSERT_NE(graph.getNode("layer0_moe_expert_ffn"), nullptr);
    ASSERT_NE(graph.getNode("layer0_moe_expert_allreduce"), nullptr)
        << "Expert-parallel MoE owns only a local expert range, so routed output is partial until allreduce";
    ASSERT_NE(graph.getNode("layer0_moe_combine"), nullptr);

    EXPECT_TRUE(hasDependency(graph, "layer0_moe_expert_allreduce", "layer0_moe_expert_ffn"));
    EXPECT_TRUE(hasDependency(graph, "layer0_moe_combine", "layer0_moe_expert_allreduce"));
}

TEST(Test__Qwen35MoEGraph, LocalTPApportionedOverlayCombinesMoEBranchesBeforeAllreduce)
{
    auto tp_ctx = std::make_unique<MockLocalTPContext>();
    tp_ctx->setDevices({GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)});
    tp_ctx->setBackend(CollectiveBackendType::HOST);

    GraphConfig config = makeMoEConfig(tp_ctx.get());
    config.moe.expert_parallel_plan = makeLocalTPApportionedOverlayPlan("hot_localtp");
    config.moe.expert_overlay_runtime_plan = resolveMoEExpertOverlayRuntimePlan(
        config.moe.expert_parallel_plan,
        MoEExpertOverlayRuntimeResolverOptions{
            .current_world_rank = 0,
            .validate_mvp_root_reachability = false,
        });

    Qwen35MoEGraph graph_builder(config, nullptr);

    TensorArena arena;
    auto layer = makeMoELayerWeights(arena);
    layer.shared_expert_gate_inp = arena.fp32({static_cast<size_t>(config.d_model)});
    auto buffers = makeActivationBuffers(arena, /*tokens=*/2, config.d_model,
                                         config.moe.num_experts, config.moe.top_k);

    ComputeGraph graph = graph_builder.buildFFNGraph(
        layer, buffers, /*layer_idx=*/0, /*seq_len=*/2,
        /*batch_size=*/1, DeviceId::cpu());

    ASSERT_NE(graph.getNode("layer0_moe_expert_ffn_overlay_fast"), nullptr);
    ASSERT_NE(graph.getNode("layer0_shared_expert_gate"), nullptr);
    ASSERT_NE(graph.getNode("layer0_moe_combined_allreduce"), nullptr);

    EXPECT_EQ(graph.getNode("layer0_moe_expert_overlay_fast_allreduce"), nullptr)
        << "LocalTP apportioned experts can reduce the routed+shared combined partial once";
    EXPECT_EQ(graph.getNode("layer0_shared_expert_allreduce"), nullptr)
        << "Shared expert partial should be gated and combined locally before the TP allreduce";
    EXPECT_EQ(graph.getNode("layer0_moe_combine"), nullptr)
        << "The fused shared gate writes the local combined partial directly to ATTN_PROJ";

    EXPECT_TRUE(hasDependency(
        graph, "layer0_shared_expert_gate", "layer0_moe_expert_ffn_overlay_fast"));
    EXPECT_TRUE(hasDependency(
        graph, "layer0_shared_expert_gate", "layer0_shared_expert_ffn"));
    EXPECT_TRUE(hasDependency(
        graph, "layer0_moe_combined_allreduce", "layer0_shared_expert_gate"));

    const auto gate_contract = graph.getNode("layer0_shared_expert_gate")->stage->bufferContract();
    EXPECT_TRUE(contractReads(gate_contract, BufferId::MOE_COMBINED_OUTPUT));
    EXPECT_TRUE(contractWrites(gate_contract, BufferId::ATTN_PROJ));

    const auto allreduce_contract = graph.getNode("layer0_moe_combined_allreduce")->stage->bufferContract();
    EXPECT_TRUE(contractReads(allreduce_contract, BufferId::ATTN_PROJ));
    EXPECT_TRUE(contractWrites(allreduce_contract, BufferId::ATTN_PROJ));
}

TEST(Test__Qwen35MoEGraph, DenseTPDisabledKeepsExpertParticipantAllreduceOnly)
{
    auto tp_ctx = std::make_unique<MockLocalTPContext>();
    tp_ctx->setDevices({GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)});
    tp_ctx->setBackend(CollectiveBackendType::HOST);

    {
        GraphConfig dense_config = makeMoEConfig(tp_ctx.get());
        dense_config.dense_tp_enabled = false;
        dense_config.ffn_column_parallel = true;

        Qwen35MoEGraph graph_builder(dense_config, nullptr);

        TensorArena arena;
        auto layer = makeDenseFFNLayerWeights(arena, dense_config);
        auto buffers = makeDenseFFNBuffers(arena, /*tokens=*/2, dense_config);

        ComputeGraph graph = graph_builder.buildFFNGraph(
            layer, buffers, /*layer_idx=*/0, /*seq_len=*/2,
            /*batch_size=*/1, DeviceId::cpu());

        ASSERT_NE(graph.getNode("layer0_down_proj"), nullptr);
        EXPECT_EQ(graph.getNode("layer0_down_allreduce"), nullptr)
            << "A LocalTP context used only for MoE expert participants must not make dense FFN emit TP allreduce";
    }

    {
        GraphConfig moe_config = makeMoEConfig(tp_ctx.get());
        moe_config.dense_tp_enabled = false;
        moe_config.moe.expert_parallel_plan = makeLocalTPApportionedOverlayPlan("hot_localtp");
        moe_config.moe.expert_overlay_runtime_plan = resolveMoEExpertOverlayRuntimePlan(
            moe_config.moe.expert_parallel_plan,
            MoEExpertOverlayRuntimeResolverOptions{
                .current_world_rank = 0,
                .validate_mvp_root_reachability = false,
            });

        Qwen35MoEGraph graph_builder(moe_config, nullptr);

        TensorArena arena;
        auto layer = makeMoELayerWeights(arena);
        layer.shared_expert_gate_inp = arena.fp32({static_cast<size_t>(moe_config.d_model)});
        auto buffers = makeActivationBuffers(arena, /*tokens=*/2, moe_config.d_model,
                                             moe_config.moe.num_experts, moe_config.moe.top_k);

        ComputeGraph graph = graph_builder.buildFFNGraph(
            layer, buffers, /*layer_idx=*/0, /*seq_len=*/2,
            /*batch_size=*/1, DeviceId::cpu());

        ASSERT_NE(graph.getNode("layer0_moe_expert_ffn_overlay_fast"), nullptr);
        EXPECT_NE(graph.getNode("layer0_moe_expert_overlay_fast_allreduce"), nullptr)
            << "MoE expert participant reduction must remain active even when dense TP is disabled";
        EXPECT_EQ(graph.getNode("layer0_moe_combined_allreduce"), nullptr)
            << "Dense-TP-disabled overlays keep shared experts replicated; only routed expert partials may be allreduced";
        EXPECT_EQ(graph.getNode("layer0_shared_expert_allreduce"), nullptr)
            << "Shared expert output is replicated when dense TP is disabled";
        EXPECT_EQ(graph.getNode("layer0_moe_combine"), nullptr)
            << "The local shared gate can combine with the already-reduced routed expert output";
        EXPECT_TRUE(hasDependency(
            graph, "layer0_shared_expert_gate", "layer0_moe_expert_overlay_fast_allreduce"));
        EXPECT_TRUE(hasDependency(
            graph, "layer0_shared_expert_gate", "layer0_shared_expert_ffn"));

        const auto gate_contract = graph.getNode("layer0_shared_expert_gate")->stage->bufferContract();
        EXPECT_TRUE(contractReads(gate_contract, BufferId::MOE_COMBINED_OUTPUT));
        EXPECT_TRUE(contractWrites(gate_contract, BufferId::ATTN_PROJ));
        EXPECT_EQ(graph.getNode("layer0_down_allreduce"), nullptr);
    }
}

TEST(Test__Qwen35MoEGraph, DenseDecodeReplicatedSuppressesDenseFFNAllreduceOnlyForDecode)
{
    auto tp_ctx = std::make_unique<MockLocalTPContext>();
    tp_ctx->setDevices({GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)});
    tp_ctx->setBackend(CollectiveBackendType::HOST);

    GraphConfig config = makeMoEConfig(tp_ctx.get());
    config.dense_tp_enabled = true;
    config.dense_tp_decode_replicated = true;
    config.ffn_column_parallel = true;

    TestableQwen35MoEGraph graph_builder(config, nullptr);
    graph_builder.setDecodeReplicatedDenseWeightBindings(makeDecodeDenseBindingSource());

    TensorArena arena;
    auto layer = makeDenseFFNLayerWeights(arena, config);

    auto decode_buffers = makeDenseFFNBuffers(arena, /*tokens=*/1, config);
    ComputeGraph decode_graph = graph_builder.buildFFNGraphForTokenCount(
        layer, decode_buffers, /*layer_idx=*/0, /*seq_len=*/1,
        /*batch_size=*/1, DeviceId::cpu());

    ASSERT_NE(decode_graph.getNode("layer0_down_proj"), nullptr);
    EXPECT_EQ(decode_graph.getNode("layer0_down_allreduce"), nullptr)
        << "Replicated dense decode owns full dense weights on every participant and should not allreduce dense FFN output";

    auto prefill_buffers = makeDenseFFNBuffers(arena, /*tokens=*/2, config);
    ComputeGraph prefill_graph = graph_builder.buildFFNGraphForTokenCount(
        layer, prefill_buffers, /*layer_idx=*/0, /*seq_len=*/2,
        /*batch_size=*/1, DeviceId::cpu());

    ASSERT_NE(prefill_graph.getNode("layer0_down_proj"), nullptr);
    EXPECT_NE(prefill_graph.getNode("layer0_down_allreduce"), nullptr)
        << "Prefill remains dense-TP partial compute and must keep the dense FFN allreduce";
}

TEST(Test__Qwen35MoEGraph, DecodeMirroredEmbeddingSuppressesOnlyDecodeEmbeddingAllreduce)
{
    auto tp_ctx = std::make_unique<MockLocalTPContext>();
    tp_ctx->setDevices({GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)});
    tp_ctx->setBackend(CollectiveBackendType::HOST);

    GraphConfig config = makeMoEConfig(tp_ctx.get());
    config.n_layers = 0;
    config.total_n_layers = 0;
    config.dense_tp_enabled = true;
    config.dense_tp_decode_mirrored_embedding = true;

    TestableQwen35MoEGraph graph_builder(config, nullptr);

    TensorArena arena;
    TensorBase *base_embedding = arena.fp32({8, 4});
    TensorBase *decode_embedding = arena.fp32({16, 4});
    TensorBase *final_norm = arena.fp32({4});
    TensorBase *lm_head = arena.fp32({16, 4});

    WeightBinding base_embedding_binding;
    base_embedding_binding.tensor = base_embedding;
    WeightBinding decode_embedding_binding;
    decode_embedding_binding.tensor = decode_embedding;
    WeightBinding final_norm_binding;
    final_norm_binding.tensor = final_norm;
    WeightBinding lm_head_binding;
    lm_head_binding.tensor = lm_head;

    ModelWeightBindings base_bindings;
    base_bindings.embedding_table = &base_embedding_binding;
    base_bindings.final_norm = &final_norm_binding;
    base_bindings.lm_head = &lm_head_binding;
    base_bindings.get_layer_weights = [](int)
    {
        return LayerWeightBindings{};
    };

    ModelWeightBindings decode_bindings;
    decode_bindings.embedding_table = &decode_embedding_binding;

    graph_builder.setWeightBindings(base_bindings);
    graph_builder.setDecodeReplicatedDenseWeightBindings(decode_bindings);

    ModelBuffers buffers;
    buffers.current_hidden = arena.fp32({2, 4});
    buffers.logits = arena.fp32({2, 16});
    buffers.layer_buffers.normalized = arena.fp32({2, 4});
    graph_builder.setBuffers(buffers);

    int decode_token = 3;
    ForwardInput decode_input;
    decode_input.token_ids = &decode_token;
    decode_input.seq_len = 1;
    decode_input.batch_size = 1;
    ForwardOutput decode_output;
    ComputeGraph decode_graph = graph_builder.buildFullForwardGraph(decode_input, decode_output);

    EXPECT_NE(decode_graph.getNode("embedding"), nullptr);
    EXPECT_EQ(decode_graph.getNode("embedding_allreduce"), nullptr)
        << "Decode mirrored embedding should use the full vocab table and skip the tiny embedding allreduce";
    EXPECT_NE(decode_graph.getNode("lm_head"), nullptr)
        << "The policy should not suppress the rest of the dense graph";

    int prefill_tokens[] = {3, 4};
    ForwardInput prefill_input;
    prefill_input.token_ids = prefill_tokens;
    prefill_input.seq_len = 2;
    prefill_input.batch_size = 1;
    ForwardOutput prefill_output;
    ComputeGraph prefill_graph = graph_builder.buildFullForwardGraph(prefill_input, prefill_output);

    EXPECT_NE(prefill_graph.getNode("embedding_allreduce"), nullptr)
        << "Prefill still uses the sharded base embedding table and must keep the embedding allreduce";
}

TEST(Test__Qwen35MoEGraph, DenseDecodeReplicatedPrefillKeepsShardedAttentionState)
{
    GraphConfig config = makeMoEConfig();
    config.dense_tp_enabled = true;
    config.dense_tp_decode_replicated = true;

    TensorArena arena;
    TensorBase *base_wq_tensor = arena.fp32({4, 4});
    TensorBase *base_gdn_gate_tensor = arena.fp32({4, 4});
    TensorBase *base_down_tensor = arena.fp32({4, 8});
    TensorBase *replicated_wq_tensor = arena.fp32({8, 4});
    TensorBase *replicated_gdn_gate_tensor = arena.fp32({8, 4});
    TensorBase *replicated_down_tensor = arena.fp32({4, 8});

    WeightBinding base_wq;
    base_wq.tensor = base_wq_tensor;
    WeightBinding base_gdn_gate;
    base_gdn_gate.tensor = base_gdn_gate_tensor;
    WeightBinding base_down;
    base_down.tensor = base_down_tensor;
    WeightBinding replicated_wq;
    replicated_wq.tensor = replicated_wq_tensor;
    WeightBinding replicated_gdn_gate;
    replicated_gdn_gate.tensor = replicated_gdn_gate_tensor;
    WeightBinding replicated_down;
    replicated_down.tensor = replicated_down_tensor;

    ModelWeightBindings base_bindings;
    base_bindings.get_layer_weights = [&](int)
    {
        LayerWeightBindings layer;
        layer.wq = &base_wq;
        layer.attn_gate = &base_gdn_gate;
        layer.down_proj = &base_down;
        return layer;
    };

    ModelWeightBindings replicated_bindings;
    replicated_bindings.get_layer_weights = [&](int)
    {
        LayerWeightBindings layer;
        layer.wq = &replicated_wq;
        layer.attn_gate = &replicated_gdn_gate;
        layer.down_proj = &replicated_down;
        return layer;
    };

    TestableQwen35MoEGraph graph_builder(config, nullptr);
    graph_builder.setWeightBindings(base_bindings);
    graph_builder.setDecodeReplicatedDenseWeightBindings(replicated_bindings);

    LayerWeights prefill = graph_builder.selectedLayerWeightsForTokenCount(
        /*layer_idx=*/0, /*total_tokens=*/2);
    EXPECT_EQ(prefill.wq, base_wq_tensor)
        << "Phase-split prefill must keep attention/QKV weights tensor-parallel";
    EXPECT_EQ(prefill.attn_gate, base_gdn_gate_tensor)
        << "Phase-split prefill must keep Qwen3.5 GDN attention-state projections tensor-parallel";
    EXPECT_EQ(prefill.down_proj, base_down_tensor)
        << "Phase-split prefill must keep non-attention dense FFN weights tensor-parallel";

    LayerWeights decode = graph_builder.selectedLayerWeightsForTokenCount(
        /*layer_idx=*/0, /*total_tokens=*/1);
    EXPECT_EQ(decode.wq, replicated_wq_tensor);
    EXPECT_EQ(decode.attn_gate, replicated_gdn_gate_tensor);
    EXPECT_EQ(decode.down_proj, replicated_down_tensor)
        << "Decode uses the full replicated dense set";
}

TEST(Test__Qwen35MoEGraph, DenseDecodeReplicatedKeepsStatefulGDNDecodeTPLocal)
{
    auto tp_ctx = std::make_unique<MockLocalTPContext>();
    tp_ctx->setDevices({GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)});
    tp_ctx->setBackend(CollectiveBackendType::HOST);

    GraphConfig config = makeGDNTPConfig(tp_ctx.get());
    config.dense_tp_enabled = true;
    config.dense_tp_decode_replicated = true;

    TensorArena arena;
    LayerWeights base_layer = makeGDNTPLayerWeights(
        arena, config, /*value_heads=*/1, /*row_parallel_out=*/true);
    LayerWeights decode_layer = makeGDNTPLayerWeights(
        arena, config, /*value_heads=*/2, /*row_parallel_out=*/false);

    WeightBinding base_attn_norm = makeTestBinding(base_layer.attn_norm);
    WeightBinding base_qkv = makeTestBinding(base_layer.attn_qkv);
    WeightBinding base_gate = makeTestBinding(base_layer.attn_gate);
    WeightBinding base_alpha = makeTestBinding(base_layer.ssm_alpha);
    WeightBinding base_beta = makeTestBinding(base_layer.ssm_beta);
    WeightBinding base_conv = makeTestBinding(base_layer.ssm_conv1d);
    WeightBinding base_dt = makeTestBinding(base_layer.ssm_dt_bias);
    WeightBinding base_a = makeTestBinding(base_layer.ssm_a);
    WeightBinding base_norm = makeTestBinding(base_layer.ssm_norm);
    WeightBinding base_out = makeTestBinding(base_layer.ssm_out);

    WeightBinding decode_attn_norm = makeTestBinding(decode_layer.attn_norm);
    WeightBinding decode_qkv = makeTestBinding(decode_layer.attn_qkv);
    WeightBinding decode_gate = makeTestBinding(decode_layer.attn_gate);
    WeightBinding decode_alpha = makeTestBinding(decode_layer.ssm_alpha);
    WeightBinding decode_beta = makeTestBinding(decode_layer.ssm_beta);
    WeightBinding decode_conv = makeTestBinding(decode_layer.ssm_conv1d);
    WeightBinding decode_dt = makeTestBinding(decode_layer.ssm_dt_bias);
    WeightBinding decode_a = makeTestBinding(decode_layer.ssm_a);
    WeightBinding decode_norm = makeTestBinding(decode_layer.ssm_norm);
    WeightBinding decode_out = makeTestBinding(decode_layer.ssm_out);

    ModelWeightBindings base_bindings;
    base_bindings.get_layer_weights = [&](int)
    {
        LayerWeightBindings layer;
        layer.attn_norm = &base_attn_norm;
        layer.attn_qkv = &base_qkv;
        layer.attn_gate = &base_gate;
        layer.ssm_alpha = &base_alpha;
        layer.ssm_beta = &base_beta;
        layer.ssm_conv1d = &base_conv;
        layer.ssm_dt_bias = &base_dt;
        layer.ssm_a = &base_a;
        layer.ssm_norm = &base_norm;
        layer.ssm_out = &base_out;
        return layer;
    };

    ModelWeightBindings decode_bindings;
    decode_bindings.get_layer_weights = [&](int)
    {
        LayerWeightBindings layer;
        layer.attn_norm = &decode_attn_norm;
        layer.attn_qkv = &decode_qkv;
        layer.attn_gate = &decode_gate;
        layer.ssm_alpha = &decode_alpha;
        layer.ssm_beta = &decode_beta;
        layer.ssm_conv1d = &decode_conv;
        layer.ssm_dt_bias = &decode_dt;
        layer.ssm_a = &decode_a;
        layer.ssm_norm = &decode_norm;
        layer.ssm_out = &decode_out;
        return layer;
    };

    TestableQwen35MoEGraph graph_builder(config, nullptr);
    graph_builder.setWeightBindings(base_bindings);
    graph_builder.setDecodeReplicatedDenseWeightBindings(decode_bindings);

    auto mpi = std::make_shared<MockMPIContext>(0, 1);
    CPUHybridRingKVCacheFP32 cache(
        makeGDNTPHybridConfig(config),
        *mpi,
        config.n_layers,
        /*batch_size=*/1,
        config.max_seq_len,
        config.n_kv_heads,
        config.head_dim,
        DeviceId::cpu());

    ActivationBuffers buffers = makeGDNTPActivationBuffers(
        arena, /*tokens=*/1, config, /*value_heads=*/1);
    int position_id = 0;

    ComputeGraph graph = graph_builder.buildAttentionGraphForTokenCount(
        decode_layer,
        buffers,
        /*layer_idx=*/0,
        /*seq_len=*/1,
        /*batch_size=*/1,
        &cache,
        &position_id,
        DeviceId::cpu());

    EXPECT_NE(graph.getNode("layer0_gdn_wo_allreduce"), nullptr)
        << "Dense-replicated decode cannot switch GDN to full mirrored state until "
           "TP-prefill GDN conv/recurrent state is allgathered into every participant.";
    EXPECT_TRUE(hasDependency(graph, "layer0_gdn_wo_allreduce", "layer0_gdn_out_proj"));
}

TEST(Test__Qwen35MoEGraph, DenseDecodeReplicatedUsesGDNLiveStateAllGatherWhenAvailable)
{
    auto tp_ctx = std::make_unique<MockLocalTPContext>();
    tp_ctx->setDevices({GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)});
    tp_ctx->setBackend(CollectiveBackendType::NCCL);

    GraphConfig config = makeGDNTPConfig(tp_ctx.get());
    config.default_device = DeviceId::cuda(0);
    config.dense_tp_enabled = true;
    config.dense_tp_decode_replicated = true;
    config.tp_device_idx = 0;

    TensorArena arena;
    LayerWeights base_layer = makeGDNTPLayerWeights(
        arena, config, /*value_heads=*/1, /*row_parallel_out=*/true);
    LayerWeights decode_layer = makeGDNTPLayerWeights(
        arena, config, /*value_heads=*/2, /*row_parallel_out=*/false);

    WeightBinding base_attn_norm = makeTestBinding(base_layer.attn_norm);
    WeightBinding base_qkv = makeTestBinding(base_layer.attn_qkv);
    WeightBinding base_gate = makeTestBinding(base_layer.attn_gate);
    WeightBinding base_alpha = makeTestBinding(base_layer.ssm_alpha);
    WeightBinding base_beta = makeTestBinding(base_layer.ssm_beta);
    WeightBinding base_conv = makeTestBinding(base_layer.ssm_conv1d);
    WeightBinding base_dt = makeTestBinding(base_layer.ssm_dt_bias);
    WeightBinding base_a = makeTestBinding(base_layer.ssm_a);
    WeightBinding base_norm = makeTestBinding(base_layer.ssm_norm);
    WeightBinding base_out = makeTestBinding(base_layer.ssm_out);

    WeightBinding decode_attn_norm = makeTestBinding(decode_layer.attn_norm);
    WeightBinding decode_qkv = makeTestBinding(decode_layer.attn_qkv);
    WeightBinding decode_gate = makeTestBinding(decode_layer.attn_gate);
    WeightBinding decode_alpha = makeTestBinding(decode_layer.ssm_alpha);
    WeightBinding decode_beta = makeTestBinding(decode_layer.ssm_beta);
    WeightBinding decode_conv = makeTestBinding(decode_layer.ssm_conv1d);
    WeightBinding decode_dt = makeTestBinding(decode_layer.ssm_dt_bias);
    WeightBinding decode_a = makeTestBinding(decode_layer.ssm_a);
    WeightBinding decode_norm = makeTestBinding(decode_layer.ssm_norm);
    WeightBinding decode_out = makeTestBinding(decode_layer.ssm_out);

    ModelWeightBindings base_bindings;
    base_bindings.get_layer_weights = [&](int)
    {
        LayerWeightBindings layer;
        layer.attn_norm = &base_attn_norm;
        layer.attn_qkv = &base_qkv;
        layer.attn_gate = &base_gate;
        layer.ssm_alpha = &base_alpha;
        layer.ssm_beta = &base_beta;
        layer.ssm_conv1d = &base_conv;
        layer.ssm_dt_bias = &base_dt;
        layer.ssm_a = &base_a;
        layer.ssm_norm = &base_norm;
        layer.ssm_out = &base_out;
        return layer;
    };

    ModelWeightBindings decode_bindings;
    decode_bindings.get_layer_weights = [&](int)
    {
        LayerWeightBindings layer;
        layer.attn_norm = &decode_attn_norm;
        layer.attn_qkv = &decode_qkv;
        layer.attn_gate = &decode_gate;
        layer.ssm_alpha = &decode_alpha;
        layer.ssm_beta = &decode_beta;
        layer.ssm_conv1d = &decode_conv;
        layer.ssm_dt_bias = &decode_dt;
        layer.ssm_a = &decode_a;
        layer.ssm_norm = &decode_norm;
        layer.ssm_out = &decode_out;
        return layer;
    };

    TestableQwen35MoEGraph graph_builder(config, nullptr);
    graph_builder.setWeightBindings(base_bindings);
    graph_builder.setDecodeReplicatedDenseWeightBindings(decode_bindings);

    auto mpi = std::make_shared<MockMPIContext>(0, 1);
    CPUHybridRingKVCacheFP32 cache(
        makeGDNTPHybridConfig(config),
        *mpi,
        config.n_layers,
        /*batch_size=*/1,
        config.max_seq_len,
        config.n_kv_heads,
        config.head_dim,
        DeviceId::cpu());

    ActivationBuffers prefill_buffers = makeGDNTPActivationBuffers(
        arena, /*tokens=*/2, config, /*value_heads=*/1);
    int position_ids[2] = {0, 1};
    ComputeGraph prefill_graph = graph_builder.buildAttentionGraphForTokenCount(
        base_layer,
        prefill_buffers,
        /*layer_idx=*/0,
        /*seq_len=*/2,
        /*batch_size=*/1,
        &cache,
        position_ids,
        DeviceId::cuda(0));

    EXPECT_NE(prefill_graph.getNode("layer0_gdn_live_state_allgather"), nullptr);
    EXPECT_TRUE(hasDependency(prefill_graph, "layer0_gdn_live_state_allgather", "layer0_gdn_recurrence"));
    EXPECT_TRUE(hasDependency(prefill_graph, "layer0_gated_norm", "layer0_gdn_live_state_allgather"));

    ActivationBuffers decode_buffers = makeGDNTPActivationBuffers(
        arena, /*tokens=*/1, config, /*value_heads=*/2);
    int position_id = 0;
    ComputeGraph decode_graph = graph_builder.buildAttentionGraphForTokenCount(
        decode_layer,
        decode_buffers,
        /*layer_idx=*/0,
        /*seq_len=*/1,
        /*batch_size=*/1,
        &cache,
        &position_id,
        DeviceId::cuda(0));

    EXPECT_EQ(decode_graph.getNode("layer0_gdn_wo_allreduce"), nullptr)
        << "A valid GDN live-state handoff lets replicated dense decode avoid the tiny GDN output allreduce.";
}

TEST(Test__Qwen35MoEGraph, DenseDecodeReplicatedUsesGDNLiveStateAllGatherForModularRepeat)
{
    auto tp_ctx = std::make_unique<MockLocalTPContext>();
    tp_ctx->setDevices({GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)});
    tp_ctx->setBackend(CollectiveBackendType::NCCL);

    GraphConfig config = makeGDNTPModularConfig(tp_ctx.get());
    config.default_device = DeviceId::cuda(0);
    config.dense_tp_enabled = true;
    config.dense_tp_decode_replicated = true;
    config.tp_device_idx = 0;

    TensorArena arena;
    LayerWeights base_layer = makeGDNTPModularLayerWeights(
        arena, config, /*key_heads=*/2, /*value_heads=*/2, /*row_parallel_out=*/true);
    LayerWeights decode_layer = makeGDNTPModularLayerWeights(
        arena, config, /*key_heads=*/2, /*value_heads=*/4, /*row_parallel_out=*/false);

    WeightBinding base_attn_norm = makeTestBinding(base_layer.attn_norm);
    WeightBinding base_qkv = makeTestBinding(base_layer.attn_qkv);
    WeightBinding base_gate = makeTestBinding(base_layer.attn_gate);
    WeightBinding base_alpha = makeTestBinding(base_layer.ssm_alpha);
    WeightBinding base_beta = makeTestBinding(base_layer.ssm_beta);
    WeightBinding base_conv = makeTestBinding(base_layer.ssm_conv1d);
    WeightBinding base_dt = makeTestBinding(base_layer.ssm_dt_bias);
    WeightBinding base_a = makeTestBinding(base_layer.ssm_a);
    WeightBinding base_norm = makeTestBinding(base_layer.ssm_norm);
    WeightBinding base_out = makeTestBinding(base_layer.ssm_out);

    WeightBinding decode_attn_norm = makeTestBinding(decode_layer.attn_norm);
    WeightBinding decode_qkv = makeTestBinding(decode_layer.attn_qkv);
    WeightBinding decode_gate = makeTestBinding(decode_layer.attn_gate);
    WeightBinding decode_alpha = makeTestBinding(decode_layer.ssm_alpha);
    WeightBinding decode_beta = makeTestBinding(decode_layer.ssm_beta);
    WeightBinding decode_conv = makeTestBinding(decode_layer.ssm_conv1d);
    WeightBinding decode_dt = makeTestBinding(decode_layer.ssm_dt_bias);
    WeightBinding decode_a = makeTestBinding(decode_layer.ssm_a);
    WeightBinding decode_norm = makeTestBinding(decode_layer.ssm_norm);
    WeightBinding decode_out = makeTestBinding(decode_layer.ssm_out);

    ModelWeightBindings base_bindings;
    base_bindings.get_layer_weights = [&](int)
    {
        LayerWeightBindings layer;
        layer.attn_norm = &base_attn_norm;
        layer.attn_qkv = &base_qkv;
        layer.attn_gate = &base_gate;
        layer.ssm_alpha = &base_alpha;
        layer.ssm_beta = &base_beta;
        layer.ssm_conv1d = &base_conv;
        layer.ssm_dt_bias = &base_dt;
        layer.ssm_a = &base_a;
        layer.ssm_norm = &base_norm;
        layer.ssm_out = &base_out;
        return layer;
    };

    ModelWeightBindings decode_bindings;
    decode_bindings.get_layer_weights = [&](int)
    {
        LayerWeightBindings layer;
        layer.attn_norm = &decode_attn_norm;
        layer.attn_qkv = &decode_qkv;
        layer.attn_gate = &decode_gate;
        layer.ssm_alpha = &decode_alpha;
        layer.ssm_beta = &decode_beta;
        layer.ssm_conv1d = &decode_conv;
        layer.ssm_dt_bias = &decode_dt;
        layer.ssm_a = &decode_a;
        layer.ssm_norm = &decode_norm;
        layer.ssm_out = &decode_out;
        return layer;
    };

    TestableQwen35MoEGraph graph_builder(config, nullptr);
    graph_builder.setWeightBindings(base_bindings);
    graph_builder.setDecodeReplicatedDenseWeightBindings(decode_bindings);

    auto mpi = std::make_shared<MockMPIContext>(0, 1);
    CPUHybridRingKVCacheFP32 cache(
        makeGDNTPHybridConfig(config),
        *mpi,
        config.n_layers,
        /*batch_size=*/1,
        config.max_seq_len,
        config.n_kv_heads,
        config.head_dim,
        DeviceId::cpu());

    ActivationBuffers prefill_buffers = makeGDNTPModularActivationBuffers(
        arena, /*tokens=*/2, config, /*key_heads=*/2, /*value_heads=*/2);
    int position_ids[2] = {0, 1};
    ComputeGraph prefill_graph = graph_builder.buildAttentionGraphForTokenCount(
        base_layer,
        prefill_buffers,
        /*layer_idx=*/0,
        /*seq_len=*/2,
        /*batch_size=*/1,
        &cache,
        position_ids,
        DeviceId::cuda(0));

    const auto *handoff_node = prefill_graph.getNode("layer0_gdn_live_state_allgather");
    ASSERT_NE(handoff_node, nullptr);
    const auto *handoff =
        dynamic_cast<const GDNLiveStateAllGatherStage *>(handoff_node->stage.get());
    ASSERT_NE(handoff, nullptr);
    const auto &params = handoff->getParams();
    EXPECT_TRUE(params.modular_conv_state);
    EXPECT_EQ(params.local_conv_state_floats, 36);
    EXPECT_EQ(params.full_conv_state_floats, 48);
    EXPECT_EQ(params.conv_history_len, 3);
    EXPECT_EQ(params.conv_qk_channels, 8);
    EXPECT_EQ(params.conv_local_v_channels, 4);
    EXPECT_EQ(params.conv_full_v_channels, 8);
    EXPECT_EQ(params.local_recurrence_state_floats, 8);
    EXPECT_EQ(params.full_recurrence_state_floats, 16);

    ActivationBuffers decode_buffers = makeGDNTPModularActivationBuffers(
        arena, /*tokens=*/1, config, /*key_heads=*/2, /*value_heads=*/4);
    int position_id = 0;
    ComputeGraph decode_graph = graph_builder.buildAttentionGraphForTokenCount(
        decode_layer,
        decode_buffers,
        /*layer_idx=*/0,
        /*seq_len=*/1,
        /*batch_size=*/1,
        &cache,
        &position_id,
        DeviceId::cuda(0));

    EXPECT_EQ(decode_graph.getNode("layer0_gdn_wo_allreduce"), nullptr)
        << "Modular-repeat GDN handoff should compact Q/K plus gathered V state and avoid decode allreduce.";
}

TEST(Test__Qwen35MoEGraph, DenseDecodeReplicatedPrefillKeepsAttentionTPAllreduce)
{
    auto tp_ctx = std::make_unique<MockLocalTPContext>();
    tp_ctx->setDevices({GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)});
    tp_ctx->setBackend(CollectiveBackendType::HOST);

    GraphConfig config = makeMoEConfig(tp_ctx.get());
    config.dense_tp_enabled = true;
    config.qkv_column_parallel = true;
    config.ffn_column_parallel = true;
    config.local_n_heads = 1;
    config.local_n_kv_heads = 1;

    TensorArena arena;
    auto layer = makeFALayerWeights(arena, config);
    layer.wo = arena.rowParallelFP32({static_cast<size_t>(config.d_model),
                                      static_cast<size_t>(config.d_model)});
    auto buffers = makeFAActivationBuffers(arena, /*tokens=*/2, config);
    std::vector<int> position_ids = {0, 1};

    Qwen35MoEGraph normal_builder(config, nullptr);
    ComputeGraph normal_graph = normal_builder.buildAttentionGraph(
        layer, buffers, /*layer_idx=*/0, /*seq_len=*/2,
        /*batch_size=*/1, /*kv_cache=*/nullptr, position_ids.data(),
        DeviceId::cpu());
    EXPECT_NE(normal_graph.getNode("layer0_wo_allreduce"), nullptr)
        << "Normal dense TP prefill must allreduce row-parallel Wo output";

    GraphConfig replicated_config = config;
    replicated_config.dense_tp_decode_replicated = true;
    TestableQwen35MoEGraph replicated_builder(replicated_config, nullptr);
    replicated_builder.setDecodeReplicatedDenseWeightBindings(makeDecodeDenseBindingSource());

    auto replicated_buffers = makeFAActivationBuffers(arena, /*tokens=*/2, replicated_config);
    ComputeGraph replicated_graph = replicated_builder.buildAttentionGraphForTokenCount(
        layer, replicated_buffers, /*layer_idx=*/0, /*seq_len=*/2,
        /*batch_size=*/1, /*kv_cache=*/nullptr, position_ids.data(),
        DeviceId::cpu());
    EXPECT_NE(replicated_graph.getNode("layer0_wo_allreduce"), nullptr)
        << "Phase-split prefill remains dense-TP and must allreduce row-parallel attention output";

    auto ffn_layer = makeDenseFFNLayerWeights(arena, replicated_config);
    auto ffn_buffers = makeDenseFFNBuffers(arena, /*tokens=*/2, replicated_config);
    ComputeGraph ffn_graph = replicated_builder.buildFFNGraphForTokenCount(
        ffn_layer, ffn_buffers, /*layer_idx=*/0, /*seq_len=*/2,
        /*batch_size=*/1, DeviceId::cpu());
    EXPECT_NE(ffn_graph.getNode("layer0_down_allreduce"), nullptr)
        << "Replicated attention state must not disable dense FFN prefill partial reductions";
}

TEST(Test__Qwen35MoEGraph, FullForwardGraphActivatesDenseDecodeReplicatedScope)
{
    auto tp_ctx = std::make_unique<MockLocalTPContext>();
    tp_ctx->setDevices({GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)});
    tp_ctx->setBackend(CollectiveBackendType::HOST);

    GraphConfig config = makeMoEConfig(tp_ctx.get());
    config.n_layers = 1;
    config.total_n_layers = 1;
    config.max_seq_len = 2;
    config.dense_tp_enabled = true;
    config.dense_tp_decode_replicated = true;
    config.ffn_column_parallel = true;
    config.vocab_local = config.vocab_size;

    TestableQwen35MoEGraph graph_builder(config, nullptr);
    graph_builder.setDecodeReplicatedDenseWeightBindings(makeDecodeDenseBindingSource());

    TensorArena arena;
    graph_builder.setWeights(makeFullForwardModelWeights(arena, config));
    graph_builder.setBuffers(makeFullForwardModelBuffers(arena, /*tokens=*/2, config));

    std::vector<int> token_ids = {0, 1};
    ForwardInput decode_input;
    decode_input.token_ids = token_ids.data();
    decode_input.batch_size = 1;
    decode_input.seq_len = 1;
    decode_input.device = DeviceId::cpu();
    ForwardOutput decode_output;

    ComputeGraph decode_graph = graph_builder.buildFullForwardGraph(decode_input, decode_output);
    ASSERT_NE(decode_graph.getNode("layer0_down_proj"), nullptr);
    EXPECT_EQ(decode_graph.getNode("layer0_down_allreduce"), nullptr)
        << "Direct full-forward decode graphs must enter the replicated-dense scope before inserting dense TP collectives";

    ForwardInput prefill_input = decode_input;
    prefill_input.seq_len = 2;
    ForwardOutput prefill_output;

    ComputeGraph prefill_graph = graph_builder.buildFullForwardGraph(prefill_input, prefill_output);
    ASSERT_NE(prefill_graph.getNode("layer0_down_proj"), nullptr);
    EXPECT_NE(prefill_graph.getNode("layer0_down_allreduce"), nullptr)
        << "Prefill full-forward graphs still compute dense TP partials and must keep the dense FFN allreduce";
}

TEST(Test__Qwen35MoEGraph, CPUAllPositionMoEVerifierUsesDecodeEquivalentExpertPath)
{
    GraphConfig config = makeMoEConfig();
    config.compute_all_position_logits = true;
    Qwen35MoEGraph graph_builder(config, nullptr);

    TensorArena arena;
    auto layer = makeMoELayerWeights(arena);
    auto buffers = makeActivationBuffers(arena, /*tokens=*/2, config.d_model,
                                         config.moe.num_experts, config.moe.top_k);

    ComputeGraph graph = graph_builder.buildFFNGraph(
        layer, buffers, /*layer_idx=*/0, /*seq_len=*/2,
        /*batch_size=*/1, DeviceId::cpu());

    auto *node = graph.getNode("layer0_moe_expert_ffn");
    ASSERT_NE(node, nullptr);
    auto *stage = dynamic_cast<MoEExpertComputeStage *>(node->stage.get());
    ASSERT_NE(stage, nullptr);
    EXPECT_TRUE(stage->usesCPUDecodeEquivalentVerifierPrefillForTesting());

    auto *shared_node = graph.getNode("layer0_shared_expert_ffn");
    ASSERT_NE(shared_node, nullptr);
    auto *shared_stage = dynamic_cast<SharedExpertFFNStage *>(shared_node->stage.get());
    ASSERT_NE(shared_stage, nullptr);
    EXPECT_TRUE(shared_stage->usesCPUDecodeEquivalentVerifierPrefillForTesting());
}

TEST(Test__Qwen35MoEGraph, SchemaDefaultsRoutedExpertWeightsToExpertParallel)
{
    Qwen35MoESchemaFactory factory;
    WeightShardingConfig sharding = factory.getWeightShardingConfig();

    EXPECT_EQ(sharding.getMode("blk.0.ffn_gate_exps.weight"), WeightShardingMode::ExpertParallel);
    EXPECT_EQ(sharding.getMode("blk.0.ffn_up_exps.weight"), WeightShardingMode::ExpertParallel);
    EXPECT_EQ(sharding.getMode("blk.0.ffn_down_exps.weight"), WeightShardingMode::ExpertParallel);
}

TEST(Test__Qwen35MoEGraph, FARopeOnReadAppendsNormalizedKToCache)
{
    GraphConfig config = makeMoEConfig();
    config.layer_types = {"full_attention", "gdn"};
    config.partial_rotary_factor = 0.5f;
    config.rope_on_read = true;

    Qwen35MoEGraph graph_builder(config, nullptr);
    TensorArena arena;
    auto layer = makeFALayerWeights(arena, config);
    auto buffers = makeFAActivationBuffers(arena, /*tokens=*/2, config);

    MockMPIContext mpi_ctx;
    llaminar::v2::kernels::KVCacheConfig kv_config;
    kv_config.precision = ActivationPrecision::FP16;
    kv_config.device = DeviceId::cpu();
    kv_config.num_layers = 1;
    kv_config.batch_size = 1;
    kv_config.max_seq_len = 8;
    kv_config.n_kv_heads = config.n_kv_heads;
    kv_config.head_dim = config.head_dim;
    kv_config.mpi_ctx = &mpi_ctx;
    auto kv_cache = llaminar::v2::kernels::KernelFactory::createKVCache(kv_config);
    ASSERT_NE(kv_cache, nullptr);

    ComputeGraph graph = graph_builder.buildAttentionGraph(
        layer, buffers, /*layer_idx=*/0, /*seq_len=*/2, /*batch_size=*/1,
        kv_cache.get(), /*position_ids=*/nullptr, DeviceId::cpu());

    ASSERT_NE(graph.getNode("layer0_kv_append"), nullptr);
    ASSERT_NE(graph.getNode("layer0_rope"), nullptr);
    ASSERT_NE(graph.getNode("layer0_k_norm"), nullptr);

    EXPECT_TRUE(hasDependency(graph, "layer0_rope", "layer0_k_norm"));
    EXPECT_TRUE(hasDependency(graph, "layer0_kv_append", "layer0_rope"))
        << "rope_on_read stores pre-RoPE K, but it must still wait for K norm";
}

TEST(Test__Qwen35MoEGraph, PrefixFingerprintMaterialIncludesExpertOverlayTopology)
{
    GraphConfig config = makeMoEConfig();
    config.moe.expert_parallel_plan = makeOverlayPlan("cold_cpu");
    config.moe.expert_overlay_runtime_plan = resolveMoEExpertOverlayRuntimePlan(
        config.moe.expert_parallel_plan,
        MoEExpertOverlayRuntimeResolverOptions{
            .current_world_rank = 0,
            .validate_mvp_root_reachability = false,
        });

    Qwen35MoEGraph graph_builder(config, nullptr);
    PrefixFingerprintMaterial material;
    graph_builder.appendPrefixCacheFingerprintMaterial(material);

    EXPECT_TRUE(hasFingerprintField(material, "expert_overlay.plan.enabled", "true"));
    EXPECT_TRUE(hasFingerprintField(material, "expert_overlay.plan.continuation_domain", "continuation"));
    EXPECT_TRUE(hasFingerprintField(material, "expert_overlay.plan.shared_expert_domain", "shared"));
    EXPECT_TRUE(hasFingerprintField(material, "expert_overlay.plan.routed_tier.0.domain", "cold_cpu"));
    EXPECT_TRUE(hasFingerprintField(material, "expert_overlay.plan.expert_domain.0.participant.1",
                                    GlobalDeviceAddress::cpu(1, "node0").toString()));
    EXPECT_TRUE(hasFingerprintField(material, "expert_overlay.runtime.enabled", "true"));
    EXPECT_TRUE(hasFingerprintField(material, "expert_overlay.runtime.domain.3.name", "cold_cpu"));
    EXPECT_TRUE(hasFingerprintField(material, "expert_overlay.runtime.domain.3.participant.1.world_rank", "1"));
    EXPECT_TRUE(hasFingerprintField(material, "expert_overlay.runtime.routed_tier.0.domain_name", "cold_cpu"));

    const uint64_t original_hash = hashPrefixFingerprintFields("moe", material.moe);

    GraphConfig changed_config = makeMoEConfig();
    changed_config.moe.expert_parallel_plan = makeOverlayPlan("warm_rocm");
    changed_config.moe.expert_overlay_runtime_plan = resolveMoEExpertOverlayRuntimePlan(
        changed_config.moe.expert_parallel_plan,
        MoEExpertOverlayRuntimeResolverOptions{
            .current_world_rank = 0,
            .validate_mvp_root_reachability = false,
        });

    Qwen35MoEGraph changed_graph_builder(changed_config, nullptr);
    PrefixFingerprintMaterial changed_material;
    changed_graph_builder.appendPrefixCacheFingerprintMaterial(changed_material);
    const uint64_t changed_hash = hashPrefixFingerprintFields("moe", changed_material.moe);

    EXPECT_NE(changed_hash, original_hash)
        << "Changing routed expert overlay domains must invalidate MoE prefix-cache payloads";
}

TEST(Test__Qwen35MoEGraph, PrefixFingerprintMaterialExcludesTransientRuntimeTables)
{
    GraphConfig config = makeMoEConfig();
    Qwen35MoEGraph graph_builder(config, nullptr);

    PrefixFingerprintMaterial material;
    graph_builder.appendPrefixCacheFingerprintMaterial(material);

    EXPECT_FALSE(hasFingerprintFieldWithPrefix(material, "graph.runtime_table"))
        << "MoE runtime tables are lazy graph-execution state and must not drift prefix keys";
    EXPECT_FALSE(hasFingerprintFieldWithPrefix(material, "runtime_table."))
        << "Prefix compatibility is represented by overlay/rebalance placement material instead";
    EXPECT_TRUE(hasFingerprintField(material, "graph.num_experts", "2"));
    EXPECT_TRUE(hasFingerprintField(material, "graph.top_k", "1"));
}

TEST(Test__Qwen35MoEGraph, ReusedRuntimeTableRegistersDecodeHistogramAfterLateControllerAttach)
{
    GraphConfig config = makeMoEConfig();
    config.n_layers = 1;
    config.total_n_layers = 1;
    config.moe.num_experts = 4;
    config.moe.top_k = 2;
    TestableQwen35MoEGraph graph_builder(config, nullptr);

    FakeRuntimeTable table;
    table.counts = {3, 0, 2, 1};

    graph_builder.registerRuntimeTableHistogramSyncForTesting("cuda:0", &table);

    DecodeExpertHistogramConfig hist_config;
    hist_config.num_layers = 1;
    hist_config.num_experts = 4;
    hist_config.top_k = 2;
    hist_config.window_size = 1;
    hist_config.sockets = {DeviceId(DeviceType::CPU, 0), DeviceId(DeviceType::CPU, 1)};
    hist_config.expert_to_socket = {0, 1, 0, 1};
    DecodeExpertHistogram histogram(hist_config);
    graph_builder.setDecodeHistogramForTesting(&histogram);

    graph_builder.registerRuntimeTableHistogramSyncForTesting("cuda:0", &table);
    graph_builder.registerRuntimeTableHistogramSyncForTesting("cuda:0", &table);

    int fake_stream = 0;
    table.recordDecodeHistogramProducerStream(&fake_stream);
    histogram.recordTokenBoundary(0);
    ASSERT_TRUE(histogram.windowFull());
    ASSERT_TRUE(histogram.syncRuntimeHistograms());

    EXPECT_EQ(table.sync_calls, 1);
    EXPECT_EQ(histogram.activationCount(0, 0), 3u);
    EXPECT_EQ(histogram.activationCount(0, 2), 2u);
    EXPECT_EQ(histogram.activationCount(0, 3), 1u);
}

TEST(Test__Qwen35MoEGraph, PrefillRuntimeTableDoesNotRegisterDecodeHistogramSync)
{
    GraphConfig config = makeMoEConfig();
    config.n_layers = 1;
    config.total_n_layers = 1;
    config.moe.num_experts = 4;
    config.moe.top_k = 2;
    TestableQwen35MoEGraph graph_builder(config, nullptr);

    DecodeExpertHistogramConfig hist_config;
    hist_config.num_layers = 1;
    hist_config.num_experts = 4;
    hist_config.top_k = 2;
    hist_config.window_size = 1;
    hist_config.sockets = {DeviceId(DeviceType::CPU, 0), DeviceId(DeviceType::CPU, 1)};
    hist_config.expert_to_socket = {0, 1, 0, 1};
    DecodeExpertHistogram histogram(hist_config);
    graph_builder.setDecodeHistogramForTesting(&histogram);

    FakeRuntimeTable prefill_only_table;
    prefill_only_table.counts = {7, 0, 0, 0};
    graph_builder.registerRuntimeTableHistogramSyncForTesting(
        "cuda:0#prefill", &prefill_only_table, /*enabled=*/false);

    histogram.recordTokenBoundary(0);
    ASSERT_TRUE(histogram.windowFull());
    EXPECT_TRUE(histogram.syncRuntimeHistograms())
        << "Prefill-only runtime tables must not become decode histogram sync sources";
    EXPECT_EQ(prefill_only_table.sync_calls, 0);
    EXPECT_EQ(histogram.activationCount(0, 0), 0u);

    FakeRuntimeTable decode_table_without_stream;
    graph_builder.registerRuntimeTableHistogramSyncForTesting(
        "cuda:0#decode", &decode_table_without_stream, /*enabled=*/true);
    EXPECT_THROW(
        (void)histogram.syncRuntimeHistograms(),
        std::runtime_error)
        << "Decode runtime-table sync should fail fast when no producer stream was recorded";
}

TEST(Test__Qwen35MoEGraph, RuntimeHistogramRegistrationIsDecodeOnly)
{
    std::ifstream in(LLAMINAR_QWEN35_MOE_GRAPH_SOURCE);
    ASSERT_TRUE(in.is_open()) << "Unable to open " << LLAMINAR_QWEN35_MOE_GRAPH_SOURCE;
    const std::string source(
        (std::istreambuf_iterator<char>(in)),
        std::istreambuf_iterator<char>());

    const size_t decode_branch = source.find("if (total_tokens == 1 &&");
    ASSERT_NE(decode_branch, std::string::npos);
    const size_t prefill_branch = source.find("else if (total_tokens > 1", decode_branch);
    ASSERT_NE(prefill_branch, std::string::npos);

    const size_t decode_call = source.find("moeRuntimeTableForDevice(", decode_branch);
    ASSERT_NE(decode_call, std::string::npos);
    ASSERT_LT(decode_call, prefill_branch);
    const size_t decode_call_end = source.find(");", decode_call);
    ASSERT_NE(decode_call_end, std::string::npos);
    const std::string decode_call_text =
        source.substr(decode_call, decode_call_end - decode_call);
    EXPECT_NE(decode_call_text.find("register_runtime_histogram_for_decode"), std::string::npos)
        << "Decode runtime tables should register host histogram sync only when host maintenance owns rebalance";
    EXPECT_EQ(decode_call_text.find("register_decode_histogram=*/false"), std::string::npos);
    EXPECT_NE(source.find("device_side_graph_rebalance_candidate"), std::string::npos)
        << "Device-side graph rebalance must not register host histogram sync callbacks";
    EXPECT_NE(source.find("!device_side_graph_rebalance_candidate"), std::string::npos);
    EXPECT_NE(source.find("env.moe_rebalance.device_rebalance_graph_controller &&"),
              std::string::npos)
        << "Host-window rebalance should keep decode histogram registration unless the graph-native controller is enabled.";
    const size_t prefill_call = source.find("moeRuntimeTableForDevice(", prefill_branch);
    ASSERT_NE(prefill_call, std::string::npos);
    const size_t prefill_call_end = source.find(");", prefill_call);
    ASSERT_NE(prefill_call_end, std::string::npos);
    const std::string prefill_call_text =
        source.substr(prefill_call, prefill_call_end - prefill_call);
    EXPECT_NE(prefill_call_text.find("/*register_decode_histogram=*/false"), std::string::npos)
        << "Prefill runtime tables must not register stale decode histogram sync callbacks";
}

TEST(Test__Qwen35MoEGraph, DeviceSideRebalanceApplyPiggybacksOnRouting)
{
    std::ifstream in(LLAMINAR_QWEN35_MOE_GRAPH_SOURCE);
    ASSERT_TRUE(in.is_open()) << "Unable to open " << LLAMINAR_QWEN35_MOE_GRAPH_SOURCE;
    const std::string source(
        (std::istreambuf_iterator<char>(in)),
        std::istreambuf_iterator<char>());
    std::string routing_stage_path = LLAMINAR_QWEN35_MOE_GRAPH_SOURCE;
    const std::string graph_suffix = "models/qwen35moe/Qwen35MoEGraph.cpp";
    const size_t graph_suffix_pos = routing_stage_path.find(graph_suffix);
    ASSERT_NE(graph_suffix_pos, std::string::npos);
    routing_stage_path.replace(
        graph_suffix_pos,
        graph_suffix.size(),
        "execution/compute_stages/stages/MoERoutingStage.cpp");
    std::ifstream routing_in(routing_stage_path);
    ASSERT_TRUE(routing_in.is_open()) << "Unable to open " << routing_stage_path;
    const std::string routing_source(
        (std::istreambuf_iterator<char>(routing_in)),
        std::istreambuf_iterator<char>());

    EXPECT_NE(source.find("graph.addDependency(plan_node, prefix + \"ffn_norm\")"),
              std::string::npos)
        << "The graph-captured rebalance producer must run after ffn_norm and before layer-local apply.";
    EXPECT_EQ(source.find("graph.addDependency(prefix + \"ffn_norm\", plan_node)"),
              std::string::npos)
        << "Reversing this edge makes ffn_norm depend on the rebalance producer and breaks captured dataflow.";
    EXPECT_NE(source.find("route_params.device_rebalance_route_apply = true"),
              std::string::npos)
        << "Maintenance-graph mode should apply ready waves inside the route kernel, not through a standalone apply stage.";
    EXPECT_NE(routing_source.find("decodeRouteSelectWithReadyRebalanceApply"),
              std::string::npos)
        << "The routing stage must call the fused route/apply kernel entrypoint.";
    EXPECT_NE(source.find("MoE routing ready-wave apply piggyback"),
              std::string::npos)
        << "The graph builder should create the rebalance binding before the first routing node.";
    EXPECT_NE(source.find("graph.addDependency(graph_rebalance_apply_node, prefix + \"ffn_norm\")"),
              std::string::npos)
        << "The opt-in standalone apply fallback should still run after ffn_norm.";
    EXPECT_EQ(source.find("graph.addDependency(graph_rebalance_apply_node, prefix + \"moe_routing\")"),
              std::string::npos)
        << "Apply-after-routing makes hot-cache dispatch invisible until a later token.";

    auto readSource = [](const std::string &path)
    {
        std::ifstream source_in(path);
        EXPECT_TRUE(source_in.is_open()) << "Unable to open " << path;
        return std::string(
            (std::istreambuf_iterator<char>(source_in)),
            std::istreambuf_iterator<char>());
    };
    auto expectNoImmediateRouteApplyBarrier =
        [](const std::string &kernel_source, const std::string &path)
    {
        const std::string marker = "try_apply_ready_rebalance_wave_for_layer_thread0(";
        size_t pos = 0;
        while ((pos = kernel_source.find(marker, pos)) != std::string::npos)
        {
            const bool call_site =
                pos > 0 &&
                (kernel_source[pos - 1] == ' ' || kernel_source[pos - 1] == '\t');
            const size_t close = kernel_source.find(");", pos);
            ASSERT_NE(close, std::string::npos) << "Malformed route-apply call in " << path;
            const size_t next = kernel_source.find_first_not_of(" \t\r\n", close + 2);
            ASSERT_NE(next, std::string::npos) << "Unexpected EOF after route-apply call in " << path;
            if (call_site)
            {
                EXPECT_NE(kernel_source.compare(next, std::string("__syncthreads();").size(), "__syncthreads();"), 0)
                    << "Ready-wave route piggyback must not pay an unconditional no-op block barrier in "
                    << path;
            }
            pos = close + 2;
        }
    };
    std::string cuda_kernel_path = LLAMINAR_QWEN35_MOE_GRAPH_SOURCE;
    cuda_kernel_path.replace(
        graph_suffix_pos,
        graph_suffix.size(),
        "kernels/cuda/moe/CUDAMoEKernels.cu");
    std::string rocm_kernel_path = LLAMINAR_QWEN35_MOE_GRAPH_SOURCE;
    rocm_kernel_path.replace(
        graph_suffix_pos,
        graph_suffix.size(),
        "kernels/rocm/moe/ROCmMoEKernels.hip");
    expectNoImmediateRouteApplyBarrier(readSource(cuda_kernel_path), cuda_kernel_path);
    expectNoImmediateRouteApplyBarrier(readSource(rocm_kernel_path), rocm_kernel_path);
}

TEST(Test__Qwen35MoEGraph, DeviceSideRebalanceRejectsFixedPayloadArenas)
{
    std::ifstream in(LLAMINAR_QWEN35_MOE_GRAPH_SOURCE);
    ASSERT_TRUE(in.is_open()) << "Unable to open " << LLAMINAR_QWEN35_MOE_GRAPH_SOURCE;
    const std::string source(
        (std::istreambuf_iterator<char>(in)),
        std::istreambuf_iterator<char>());

    const size_t selector = source.find("selectGraphRebalanceTransferMode");
    ASSERT_NE(selector, std::string::npos);
    const size_t payload_gate =
        source.find("device_rebalance_payload_sideband", selector);
    ASSERT_NE(payload_gate, std::string::npos)
        << "Bulk payload sideband requests must be rejected explicitly.";
    const size_t legacy_gate =
        source.find("allow_legacy_collective_rebalance_transfer", payload_gate);
    ASSERT_NE(legacy_gate, std::string::npos)
        << "Legacy fixed payload allgather requests must be rejected explicitly.";
    EXPECT_NE(source.find("fixed-size collective payload arenas move empty expert slots",
                          payload_gate),
              std::string::npos)
        << "Measured fixed-arena transfer moved mostly empty slots; selector must fail closed.";
    EXPECT_NE(source.find("Use CompactTransferSlots async maintenance for non-empty transfer-slot arrivals",
                          payload_gate),
              std::string::npos)
        << "The production path should use compact non-empty transfer slots.";
    EXPECT_NE(source.find("deviceMoERebalanceModeMovesFixedPayloadCapacity"),
              std::string::npos)
        << "Graph code should use the shared fixed-capacity transfer classifier instead of duplicating enum semantics.";
    const size_t selector_end =
        source.find("int gpuOrdinalForGraphDevice", selector);
    ASSERT_NE(selector_end, std::string::npos);
    const std::string selector_body =
        source.substr(selector, selector_end - selector);
    EXPECT_EQ(selector_body.find("DeviceMoERebalanceTransferMode::CollectiveSidebandPayload"),
              std::string::npos)
        << "The graph transfer selector must not choose fixed-size payload arenas.";
    EXPECT_EQ(selector_body.find("DeviceMoERebalanceTransferMode::LegacyCollectiveAllGather"),
              std::string::npos)
        << "The graph transfer selector must not choose legacy fixed-size payload allgather.";
    EXPECT_NE(selector_body.find("DeviceMoERebalanceTransferMode::CompactTransferSlots"),
              std::string::npos)
        << "The graph transfer selector should choose compact transfer slots for same-backend maintenance.";
}

TEST(Test__Qwen35MoEGraph, DeviceSideRebalanceMaintenanceSkipsDecodeHistogramSideband)
{
    std::ifstream in(LLAMINAR_QWEN35_MOE_GRAPH_SOURCE);
    ASSERT_TRUE(in.is_open()) << "Unable to open " << LLAMINAR_QWEN35_MOE_GRAPH_SOURCE;
    const std::string source(
        (std::istreambuf_iterator<char>(in)),
        std::istreambuf_iterator<char>());

    const size_t producer_mode =
        source.find("const bool producer_runs_in_maintenance_graph");
    ASSERT_NE(producer_mode, std::string::npos);
    const size_t sideband_gate =
        source.find("const bool producer_can_use_collective_sideband", producer_mode);
    ASSERT_NE(sideband_gate, std::string::npos);
    const size_t sideband_backend =
        source.find("supportsCollectiveSidebandOnStreamGraphCapture", sideband_gate);
    ASSERT_NE(sideband_backend, std::string::npos);
    const std::string gate_body =
        source.substr(sideband_gate, sideband_backend - sideband_gate);

    EXPECT_NE(gate_body.find("!producer_runs_in_maintenance_graph"),
              std::string::npos)
        << "Async maintenance must not also attach histogram sidebands to decode collectives.";
    EXPECT_NE(gate_body.find("DeviceMoERebalanceTransferMode::CollectiveSidebandPayload"),
              std::string::npos)
        << "Decode-side histogram sidebands may exist only behind the now-refused fixed-payload mode.";
}
