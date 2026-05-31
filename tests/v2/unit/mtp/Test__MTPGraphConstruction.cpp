#include <gtest/gtest.h>

#include "backends/ComputeBackend.h"
#include "execution/compute_stages/stages/MTPConcatStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "kernels/cpu/CPURingKVCache.h"
#include "loaders/PreparedWeightStore.h"
#include "mocks/MockMPIContext.h"
#include "models/qwen/QwenStandardGraph.h"
#include "models/qwen35/Qwen35Graph.h"
#include "utils/TestTensorFactory.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    struct DenseMTPGraphFixture
    {
        GraphConfig config;
        std::shared_ptr<MockMPIContext> mpi = std::make_shared<MockMPIContext>(0, 1);

        std::unique_ptr<FP32Tensor> embedding_table;
        std::unique_ptr<FP32Tensor> lm_head;

        std::unique_ptr<FP32Tensor> fc;
        std::unique_ptr<FP32Tensor> pre_hidden_norm;
        std::unique_ptr<FP32Tensor> pre_embedding_norm;
        std::unique_ptr<FP32Tensor> final_norm;
        std::unique_ptr<FP32Tensor> attn_norm;
        std::unique_ptr<FP32Tensor> wq;
        std::unique_ptr<FP32Tensor> wk;
        std::unique_ptr<FP32Tensor> wv;
        std::unique_ptr<FP32Tensor> wo;
        std::unique_ptr<FP32Tensor> q_norm;
        std::unique_ptr<FP32Tensor> k_norm;
        std::unique_ptr<FP32Tensor> ffn_norm;
        std::unique_ptr<FP32Tensor> gate_proj;
        std::unique_ptr<FP32Tensor> up_proj;
        std::unique_ptr<FP32Tensor> down_proj;

        std::unique_ptr<FP32Tensor> terminal_hidden;
        std::unique_ptr<FP32Tensor> embedding;
        std::unique_ptr<FP32Tensor> norm_hidden;
        std::unique_ptr<FP32Tensor> norm_embedding;
        std::unique_ptr<FP32Tensor> concat;
        std::unique_ptr<FP32Tensor> projected;
        std::unique_ptr<FP32Tensor> hidden;
        std::unique_ptr<FP32Tensor> q;
        std::unique_ptr<FP32Tensor> k;
        std::unique_ptr<FP32Tensor> v;
        std::unique_ptr<FP32Tensor> q_raw;
        std::unique_ptr<FP32Tensor> q_gate;
        std::unique_ptr<FP32Tensor> attn_output;
        std::unique_ptr<FP32Tensor> attn_proj;
        std::unique_ptr<FP32Tensor> gate;
        std::unique_ptr<FP32Tensor> up;
        std::unique_ptr<FP32Tensor> ffn_output;
        std::unique_ptr<FP32Tensor> logits;

        std::unique_ptr<ICPUKVCache> kv_cache;
        int draft_token = 17;
        int position_id = 5;

        DenseMTPGraphFixture()
        {
            config.n_layers = 2;
            config.d_model = 64;
            config.n_heads = 4;
            config.n_kv_heads = 2;
            config.head_dim = 16;
            config.d_ff = 128;
            config.vocab_size = 1000;
            config.rms_norm_eps = 1e-6f;
            config.rope_theta = 10000.0f;
            config.partial_rotary_factor = 1.0f;
            config.default_device = DeviceId::cpu();
            config.max_seq_len = 16;
            config.layer_types = {"full_attention", "full_attention"};

            const size_t d = static_cast<size_t>(config.d_model);
            const size_t q_dim = static_cast<size_t>(config.n_heads * config.head_dim);
            const size_t kv_dim = static_cast<size_t>(config.n_kv_heads * config.head_dim);
            const size_t ff = static_cast<size_t>(config.d_ff);
            const size_t vocab = static_cast<size_t>(config.vocab_size);

            embedding_table = TestTensorFactory::createFP32Random({vocab, d});
            lm_head = TestTensorFactory::createFP32Random({vocab, d});

            fc = TestTensorFactory::createFP32Random({d, d * 2});
            pre_hidden_norm = TestTensorFactory::createFP32Ones({d});
            pre_embedding_norm = TestTensorFactory::createFP32Ones({d});
            final_norm = TestTensorFactory::createFP32Ones({d});
            attn_norm = TestTensorFactory::createFP32Ones({d});
            wq = TestTensorFactory::createFP32Random({q_dim * 2, d});
            wk = TestTensorFactory::createFP32Random({kv_dim, d});
            wv = TestTensorFactory::createFP32Random({kv_dim, d});
            wo = TestTensorFactory::createFP32Random({d, q_dim});
            q_norm = TestTensorFactory::createFP32Ones({static_cast<size_t>(config.head_dim)});
            k_norm = TestTensorFactory::createFP32Ones({static_cast<size_t>(config.head_dim)});
            ffn_norm = TestTensorFactory::createFP32Ones({d});
            gate_proj = TestTensorFactory::createFP32Random({ff, d});
            up_proj = TestTensorFactory::createFP32Random({ff, d});
            down_proj = TestTensorFactory::createFP32Random({d, ff});

            terminal_hidden = TestTensorFactory::createFP32Random({1, d});
            embedding = TestTensorFactory::createFP32({1, d});
            norm_hidden = TestTensorFactory::createFP32({1, d});
            norm_embedding = TestTensorFactory::createFP32({1, d});
            concat = TestTensorFactory::createFP32({1, d * 2});
            projected = TestTensorFactory::createFP32({1, d});
            hidden = TestTensorFactory::createFP32({1, d});
            q = TestTensorFactory::createFP32({1, q_dim});
            k = TestTensorFactory::createFP32({1, kv_dim});
            v = TestTensorFactory::createFP32({1, kv_dim});
            q_raw = TestTensorFactory::createFP32({1, q_dim * 2});
            q_gate = TestTensorFactory::createFP32({1, q_dim});
            attn_output = TestTensorFactory::createFP32({1, q_dim});
            attn_proj = TestTensorFactory::createFP32({1, d});
            gate = TestTensorFactory::createFP32({1, ff});
            up = TestTensorFactory::createFP32({1, ff});
            ffn_output = TestTensorFactory::createFP32({1, d});
            logits = TestTensorFactory::createFP32({1, vocab});

            kv_cache = createCPURingKVCache(
                ActivationPrecision::FP32,
                *mpi,
                /*n_layers=*/1,
                /*batch_size=*/1,
                /*max_seq_len=*/8,
                config.n_kv_heads,
                config.head_dim,
                DeviceId::cpu());
        }

        ModelWeights modelWeights() const
        {
            ModelWeights weights;
            weights.embedding_table = embedding_table.get();
            weights.lm_head = lm_head.get();
            return weights;
        }

        MTPDepthWeights mtpWeights() const
        {
            MTPDepthWeights weights;
            weights.depth_index = 0;
            weights.source_layer_index = 64;
            weights.nextn_block_layout = true;
            weights.fc = fc.get();
            weights.pre_fc_norm_hidden = pre_hidden_norm.get();
            weights.pre_fc_norm_embedding = pre_embedding_norm.get();
            weights.final_norm = final_norm.get();
            weights.fa_block.attn_norm = attn_norm.get();
            weights.fa_block.wq = wq.get();
            weights.fa_block.wk = wk.get();
            weights.fa_block.wv = wv.get();
            weights.fa_block.wo = wo.get();
            weights.fa_block.q_norm = q_norm.get();
            weights.fa_block.k_norm = k_norm.get();
            weights.fa_block.ffn_norm = ffn_norm.get();
            weights.fa_block.gate_proj = gate_proj.get();
            weights.fa_block.up_proj = up_proj.get();
            weights.fa_block.down_proj = down_proj.get();
            return weights;
        }

        MTPForwardInput input()
        {
            MTPForwardInput in;
            in.draft_token_ids = &draft_token;
            in.terminal_hidden = terminal_hidden.get();
            in.kv_cache = kv_cache.get();
            in.position_ids = &position_id;
            in.batch_size = 1;
            in.seq_len = 1;
            in.device = DeviceId::cpu();
            return in;
        }

        MTPForwardOutput output()
        {
            MTPForwardOutput out;
            out.logits = logits.get();
            out.hidden = hidden.get();
            out.embedding = embedding.get();
            out.norm_hidden = norm_hidden.get();
            out.norm_embedding = norm_embedding.get();
            out.concat = concat.get();
            out.projected = projected.get();
            out.q = q.get();
            out.k = k.get();
            out.v = v.get();
            out.q_raw = q_raw.get();
            out.q_gate = q_gate.get();
            out.attn_output = attn_output.get();
            out.attn_proj = attn_proj.get();
            out.gate = gate.get();
            out.up = up.get();
            out.ffn_output = ffn_output.get();
            return out;
        }
    };

    struct TinyQwenForwardFixture
    {
        struct LayerTensors
        {
            std::unique_ptr<FP32Tensor> attn_norm;
            std::unique_ptr<FP32Tensor> wq;
            std::unique_ptr<FP32Tensor> wk;
            std::unique_ptr<FP32Tensor> wv;
            std::unique_ptr<FP32Tensor> wo;
            std::unique_ptr<FP32Tensor> ffn_norm;
            std::unique_ptr<FP32Tensor> gate_proj;
            std::unique_ptr<FP32Tensor> up_proj;
            std::unique_ptr<FP32Tensor> down_proj;
        };

        GraphConfig config;
        std::shared_ptr<MockMPIContext> mpi = std::make_shared<MockMPIContext>(0, 1);

        std::unique_ptr<FP32Tensor> embedding_table;
        std::unique_ptr<FP32Tensor> final_norm;
        std::unique_ptr<FP32Tensor> lm_head;
        std::vector<LayerTensors> layers;

        TinyQwenForwardFixture(DeviceId device, KVCachePrecision kv_precision)
        {
            config.n_layers = 1;
            config.total_n_layers = 1;
            config.d_model = 64;
            config.n_heads = 4;
            config.n_kv_heads = 2;
            config.head_dim = 16;
            config.d_ff = 128;
            config.vocab_size = 128;
            config.rms_norm_eps = 1e-6f;
            config.rope_theta = 10000.0f;
            config.default_device = device;
            config.max_seq_len = 8;
            config.activation_precision = ActivationPrecision::FP32;
            config.kv_cache_precision = kv_precision;
            config.use_graph_buffer_management = true;
            config.mtp.enabled = true;
            config.mtp.draft_tokens = 1;

            const size_t d = static_cast<size_t>(config.d_model);
            const size_t q_dim = static_cast<size_t>(config.n_heads * config.head_dim);
            const size_t kv_dim = static_cast<size_t>(config.n_kv_heads * config.head_dim);
            const size_t ff = static_cast<size_t>(config.d_ff);
            const size_t vocab = static_cast<size_t>(config.vocab_size);

            embedding_table = TestTensorFactory::createFP32Random({vocab, d}, -0.02f, 0.02f, 101);
            final_norm = TestTensorFactory::createFP32Ones({d});
            lm_head = TestTensorFactory::createFP32Random({vocab, d}, -0.02f, 0.02f, 102);

            layers.resize(static_cast<size_t>(config.n_layers));
            for (int i = 0; i < config.n_layers; ++i)
            {
                auto &layer = layers[static_cast<size_t>(i)];
                layer.attn_norm = TestTensorFactory::createFP32Ones({d});
                layer.wq = TestTensorFactory::createFP32Random({q_dim, d}, -0.02f, 0.02f, 110 + i);
                layer.wk = TestTensorFactory::createFP32Random({kv_dim, d}, -0.02f, 0.02f, 120 + i);
                layer.wv = TestTensorFactory::createFP32Random({kv_dim, d}, -0.02f, 0.02f, 130 + i);
                layer.wo = TestTensorFactory::createFP32Random({d, q_dim}, -0.02f, 0.02f, 140 + i);
                layer.ffn_norm = TestTensorFactory::createFP32Ones({d});
                layer.gate_proj = TestTensorFactory::createFP32Random({ff, d}, -0.02f, 0.02f, 150 + i);
                layer.up_proj = TestTensorFactory::createFP32Random({ff, d}, -0.02f, 0.02f, 160 + i);
                layer.down_proj = TestTensorFactory::createFP32Random({d, ff}, -0.02f, 0.02f, 170 + i);
            }
        }

        ModelWeights modelWeights()
        {
            ModelWeights weights;
            weights.embedding_table = embedding_table.get();
            weights.final_norm = final_norm.get();
            weights.lm_head = lm_head.get();
            weights.get_layer_weights = [this](int layer_idx)
            {
                const auto &src = layers.at(static_cast<size_t>(layer_idx));
                LayerWeights layer;
                layer.attn_norm = src.attn_norm.get();
                layer.wq = src.wq.get();
                layer.wk = src.wk.get();
                layer.wv = src.wv.get();
                layer.wo = src.wo.get();
                layer.ffn_norm = src.ffn_norm.get();
                return layer;
            };
            return weights;
        }
    };

    struct TinyQwen35MTPForwardFixture
    {
        GraphConfig config;
        std::shared_ptr<MockMPIContext> mpi = std::make_shared<MockMPIContext>(0, 1);

        std::unique_ptr<FP32Tensor> embedding_table;
        std::unique_ptr<FP32Tensor> final_norm;
        std::unique_ptr<FP32Tensor> lm_head;

        std::unique_ptr<FP32Tensor> attn_norm;
        std::unique_ptr<FP32Tensor> wq;
        std::unique_ptr<FP32Tensor> wk;
        std::unique_ptr<FP32Tensor> wv;
        std::unique_ptr<FP32Tensor> wo;
        std::unique_ptr<FP32Tensor> q_norm;
        std::unique_ptr<FP32Tensor> k_norm;
        std::unique_ptr<FP32Tensor> ffn_norm;
        std::unique_ptr<FP32Tensor> gate_proj;
        std::unique_ptr<FP32Tensor> up_proj;
        std::unique_ptr<FP32Tensor> down_proj;

        std::unique_ptr<FP32Tensor> mtp_fc;
        std::unique_ptr<FP32Tensor> mtp_pre_hidden_norm;
        std::unique_ptr<FP32Tensor> mtp_pre_embedding_norm;
        std::unique_ptr<FP32Tensor> mtp_final_norm;
        std::unique_ptr<FP32Tensor> mtp_attn_norm;
        std::unique_ptr<FP32Tensor> mtp_wq;
        std::unique_ptr<FP32Tensor> mtp_wk;
        std::unique_ptr<FP32Tensor> mtp_wv;
        std::unique_ptr<FP32Tensor> mtp_wo;
        std::unique_ptr<FP32Tensor> mtp_q_norm;
        std::unique_ptr<FP32Tensor> mtp_k_norm;
        std::unique_ptr<FP32Tensor> mtp_ffn_norm;
        std::unique_ptr<FP32Tensor> mtp_gate_proj;
        std::unique_ptr<FP32Tensor> mtp_up_proj;
        std::unique_ptr<FP32Tensor> mtp_down_proj;

        TinyQwen35MTPForwardFixture()
        {
            config.n_layers = 1;
            config.total_n_layers = 1;
            config.d_model = 64;
            config.n_heads = 4;
            config.n_kv_heads = 2;
            config.head_dim = 16;
            config.d_ff = 128;
            config.vocab_size = 128;
            config.rms_norm_eps = 1e-6f;
            config.rope_theta = 10000.0f;
            config.partial_rotary_factor = 1.0f;
            config.default_device = DeviceId::cpu();
            config.max_seq_len = 8;
            config.activation_precision = ActivationPrecision::FP32;
            config.kv_cache_precision = KVCachePrecision::FP32;
            config.use_graph_buffer_management = true;
            config.layer_types = {"full_attention"};
            config.mtp.enabled = true;
            config.mtp.draft_tokens = 1;

            const size_t d = static_cast<size_t>(config.d_model);
            const size_t q_dim = static_cast<size_t>(config.n_heads * config.head_dim);
            const size_t kv_dim = static_cast<size_t>(config.n_kv_heads * config.head_dim);
            const size_t ff = static_cast<size_t>(config.d_ff);
            const size_t vocab = static_cast<size_t>(config.vocab_size);

            embedding_table = TestTensorFactory::createFP32Random({vocab, d}, -0.02f, 0.02f, 201);
            final_norm = TestTensorFactory::createFP32Ones({d});
            lm_head = TestTensorFactory::createFP32Random({vocab, d}, -0.02f, 0.02f, 202);

            attn_norm = TestTensorFactory::createFP32Ones({d});
            wq = TestTensorFactory::createFP32Random({q_dim * 2, d}, -0.02f, 0.02f, 210);
            wk = TestTensorFactory::createFP32Random({kv_dim, d}, -0.02f, 0.02f, 211);
            wv = TestTensorFactory::createFP32Random({kv_dim, d}, -0.02f, 0.02f, 212);
            wo = TestTensorFactory::createFP32Random({d, q_dim}, -0.02f, 0.02f, 213);
            q_norm = TestTensorFactory::createFP32Ones({static_cast<size_t>(config.head_dim)});
            k_norm = TestTensorFactory::createFP32Ones({static_cast<size_t>(config.head_dim)});
            ffn_norm = TestTensorFactory::createFP32Ones({d});
            gate_proj = TestTensorFactory::createFP32Random({ff, d}, -0.02f, 0.02f, 214);
            up_proj = TestTensorFactory::createFP32Random({ff, d}, -0.02f, 0.02f, 215);
            down_proj = TestTensorFactory::createFP32Random({d, ff}, -0.02f, 0.02f, 216);

            mtp_fc = TestTensorFactory::createFP32Random({d, d * 2}, -0.02f, 0.02f, 220);
            mtp_pre_hidden_norm = TestTensorFactory::createFP32Ones({d});
            mtp_pre_embedding_norm = TestTensorFactory::createFP32Ones({d});
            mtp_final_norm = TestTensorFactory::createFP32Ones({d});
            mtp_attn_norm = TestTensorFactory::createFP32Ones({d});
            mtp_wq = TestTensorFactory::createFP32Random({q_dim * 2, d}, -0.02f, 0.02f, 221);
            mtp_wk = TestTensorFactory::createFP32Random({kv_dim, d}, -0.02f, 0.02f, 222);
            mtp_wv = TestTensorFactory::createFP32Random({kv_dim, d}, -0.02f, 0.02f, 223);
            mtp_wo = TestTensorFactory::createFP32Random({d, q_dim}, -0.02f, 0.02f, 224);
            mtp_q_norm = TestTensorFactory::createFP32Ones({static_cast<size_t>(config.head_dim)});
            mtp_k_norm = TestTensorFactory::createFP32Ones({static_cast<size_t>(config.head_dim)});
            mtp_ffn_norm = TestTensorFactory::createFP32Ones({d});
            mtp_gate_proj = TestTensorFactory::createFP32Random({ff, d}, -0.02f, 0.02f, 225);
            mtp_up_proj = TestTensorFactory::createFP32Random({ff, d}, -0.02f, 0.02f, 226);
            mtp_down_proj = TestTensorFactory::createFP32Random({d, ff}, -0.02f, 0.02f, 227);
        }
    };

    bool hasDependency(const ComputeGraph &graph, const std::string &node, const std::string &dep)
    {
        const auto *n = graph.getNode(node);
        if (!n)
            return false;
        return std::find(n->dependencies.begin(), n->dependencies.end(), dep) != n->dependencies.end();
    }

    int maxCachedTokens(const std::vector<PrefixKVCacheProbe> &caches)
    {
        int max_tokens = 0;
        for (const auto &cache : caches)
        {
            for (const auto &layer : cache.layers)
                max_tokens = std::max(max_tokens, layer.cached_tokens);
        }
        return max_tokens;
    }

    void prepareDenseForwardWeights(
        const DeviceGraphOrchestrator &orchestrator,
        QwenStandardGraph &graph_builder,
        PreparedWeightStore &store,
        DeviceId device)
    {
        const FrozenModelWeightSet *frozen = orchestrator.frozenWeightSet();
        ASSERT_NE(frozen, nullptr);

        for (const auto &source_binding : frozen->bindings())
        {
            if (!source_binding.tensor ||
                source_binding.tensor->shape().size() != 2 ||
                source_binding.identity.role == WeightRole::Embedding)
            {
                continue;
            }

            WeightBinding binding = source_binding;
            binding.residency.home_device = device;
            binding.residency.resident_device = device;
            ASSERT_TRUE(binding.tensor->ensureOnDevice(device));
            store.prepareGemm(binding);
        }

        graph_builder.setPreparedWeightStore(&store);
    }

    std::unique_ptr<FrozenModelWeightSet> makeDenseMTPFrozenWeightSet(
        const DenseMTPGraphFixture &fixture)
    {
        InferenceStrategy strategy;
        strategy.mode = WeightInferenceMode::SingleDevice;
        strategy.devices.push_back(DeviceId::cpu());

        ModelWeightSetBuilder builder(strategy);
        auto add_global = [&](const std::string &name, TensorBase *tensor, WeightRole role)
        {
            WeightBinding binding;
            binding.identity.canonical_name = name;
            binding.identity.role = role;
            binding.identity.logical_id = stableWeightLogicalId(name);
            binding.residency.home_device = DeviceId::cpu();
            binding.residency.resident_device = DeviceId::cpu();
            binding.tensor = tensor;
            binding.immutable = true;
            builder.addBinding(std::move(binding));
        };

        add_global("token_embd.weight", fixture.embedding_table.get(), WeightRole::Embedding);
        add_global("output.weight", fixture.lm_head.get(), WeightRole::LMHead);

        add_global("mtp.fc.weight", fixture.fc.get(), WeightRole::Other);
        add_global("mtp.pre_fc_norm_hidden.weight", fixture.pre_hidden_norm.get(), WeightRole::Norm);
        add_global("mtp.pre_fc_norm_embedding.weight", fixture.pre_embedding_norm.get(), WeightRole::Norm);
        add_global("mtp.norm.weight", fixture.final_norm.get(), WeightRole::Norm);
        add_global("mtp.layers.0.input_layernorm.weight", fixture.attn_norm.get(), WeightRole::Norm);
        add_global("mtp.layers.0.self_attn.q_proj.weight", fixture.wq.get(), WeightRole::AttentionQ);
        add_global("mtp.layers.0.self_attn.k_proj.weight", fixture.wk.get(), WeightRole::AttentionK);
        add_global("mtp.layers.0.self_attn.v_proj.weight", fixture.wv.get(), WeightRole::AttentionV);
        add_global("mtp.layers.0.self_attn.o_proj.weight", fixture.wo.get(), WeightRole::AttentionWO);
        add_global("mtp.layers.0.self_attn.q_norm.weight", fixture.q_norm.get(), WeightRole::Norm);
        add_global("mtp.layers.0.self_attn.k_norm.weight", fixture.k_norm.get(), WeightRole::Norm);
        add_global("mtp.layers.0.post_attention_layernorm.weight", fixture.ffn_norm.get(), WeightRole::Norm);
        add_global("mtp.layers.0.mlp.gate_proj.weight", fixture.gate_proj.get(), WeightRole::FFNGate);
        add_global("mtp.layers.0.mlp.up_proj.weight", fixture.up_proj.get(), WeightRole::FFNUp);
        add_global("mtp.layers.0.mlp.down_proj.weight", fixture.down_proj.get(), WeightRole::FFNDown);

        auto frozen = std::make_unique<FrozenModelWeightSet>(
            strategy,
            builder.freezeBindings());
        frozen->validateForGraph();
        return frozen;
    }

    std::unique_ptr<FrozenModelWeightSet> makeTinyQwen35MTPFrozenWeightSet(
        const TinyQwen35MTPForwardFixture &fixture)
    {
        InferenceStrategy strategy;
        strategy.mode = WeightInferenceMode::SingleDevice;
        strategy.devices.push_back(DeviceId::cpu());

        ModelWeightSetBuilder builder(strategy);
        auto add = [&](const std::string &name, TensorBase *tensor, WeightRole role)
        {
            WeightBinding binding;
            binding.identity.canonical_name = name;
            binding.identity.role = role;
            binding.identity.layer = inferWeightLayer(name);
            binding.identity.logical_id = stableWeightLogicalId(name);
            binding.residency.home_device = DeviceId::cpu();
            binding.residency.resident_device = DeviceId::cpu();
            binding.tensor = tensor;
            binding.immutable = true;
            builder.addBinding(std::move(binding));
        };
        auto add_global = [&](const std::string &name, TensorBase *tensor, WeightRole role)
        {
            WeightBinding binding;
            binding.identity.canonical_name = name;
            binding.identity.role = role;
            binding.identity.logical_id = stableWeightLogicalId(name);
            binding.residency.home_device = DeviceId::cpu();
            binding.residency.resident_device = DeviceId::cpu();
            binding.tensor = tensor;
            binding.immutable = true;
            builder.addBinding(std::move(binding));
        };

        add_global("token_embd.weight", fixture.embedding_table.get(), WeightRole::Embedding);
        add_global("output_norm.weight", fixture.final_norm.get(), WeightRole::OutputNorm);
        add_global("output.weight", fixture.lm_head.get(), WeightRole::LMHead);

        add("blk.0.attn_norm.weight", fixture.attn_norm.get(), WeightRole::Norm);
        add("blk.0.attn_q.weight", fixture.wq.get(), WeightRole::AttentionQ);
        add("blk.0.attn_k.weight", fixture.wk.get(), WeightRole::AttentionK);
        add("blk.0.attn_v.weight", fixture.wv.get(), WeightRole::AttentionV);
        add("blk.0.attn_output.weight", fixture.wo.get(), WeightRole::AttentionWO);
        add("blk.0.attn_q_norm.weight", fixture.q_norm.get(), WeightRole::Norm);
        add("blk.0.attn_k_norm.weight", fixture.k_norm.get(), WeightRole::Norm);
        add("blk.0.post_attention_norm.weight", fixture.ffn_norm.get(), WeightRole::Norm);
        add("blk.0.ffn_gate.weight", fixture.gate_proj.get(), WeightRole::FFNGate);
        add("blk.0.ffn_up.weight", fixture.up_proj.get(), WeightRole::FFNUp);
        add("blk.0.ffn_down.weight", fixture.down_proj.get(), WeightRole::FFNDown);

        add_global("mtp.fc.weight", fixture.mtp_fc.get(), WeightRole::Other);
        add_global("mtp.pre_fc_norm_hidden.weight", fixture.mtp_pre_hidden_norm.get(), WeightRole::Norm);
        add_global("mtp.pre_fc_norm_embedding.weight", fixture.mtp_pre_embedding_norm.get(), WeightRole::Norm);
        add_global("mtp.norm.weight", fixture.mtp_final_norm.get(), WeightRole::Norm);
        add_global("mtp.layers.0.input_layernorm.weight", fixture.mtp_attn_norm.get(), WeightRole::Norm);
        add_global("mtp.layers.0.self_attn.q_proj.weight", fixture.mtp_wq.get(), WeightRole::AttentionQ);
        add_global("mtp.layers.0.self_attn.k_proj.weight", fixture.mtp_wk.get(), WeightRole::AttentionK);
        add_global("mtp.layers.0.self_attn.v_proj.weight", fixture.mtp_wv.get(), WeightRole::AttentionV);
        add_global("mtp.layers.0.self_attn.o_proj.weight", fixture.mtp_wo.get(), WeightRole::AttentionWO);
        add_global("mtp.layers.0.self_attn.q_norm.weight", fixture.mtp_q_norm.get(), WeightRole::Norm);
        add_global("mtp.layers.0.self_attn.k_norm.weight", fixture.mtp_k_norm.get(), WeightRole::Norm);
        add_global("mtp.layers.0.post_attention_layernorm.weight", fixture.mtp_ffn_norm.get(), WeightRole::Norm);
        add_global("mtp.layers.0.mlp.gate_proj.weight", fixture.mtp_gate_proj.get(), WeightRole::FFNGate);
        add_global("mtp.layers.0.mlp.up_proj.weight", fixture.mtp_up_proj.get(), WeightRole::FFNUp);
        add_global("mtp.layers.0.mlp.down_proj.weight", fixture.mtp_down_proj.get(), WeightRole::FFNDown);

        auto frozen = std::make_unique<FrozenModelWeightSet>(
            strategy,
            builder.freezeBindings());
        frozen->validateForGraph();
        return frozen;
    }

    void prepareFrozenGemmWeightsForCPU(
        const FrozenModelWeightSet &frozen,
        PreparedWeightStore &store)
    {
        for (const auto &source_binding : frozen.bindings())
        {
            if (!source_binding.tensor ||
                source_binding.tensor->shape().size() != 2 ||
                source_binding.identity.role == WeightRole::Embedding)
            {
                continue;
            }

            WeightBinding binding = source_binding;
            binding.residency.home_device = DeviceId::cpu();
            binding.residency.resident_device = DeviceId::cpu();
            store.prepareGemm(binding);
        }
    }
} // namespace

TEST(Test__MTPGraphConstruction, ConcatStageCopiesHiddenThenEmbedding)
{
    auto hidden = TestTensorFactory::createFP32({2, 3});
    auto embedding = TestTensorFactory::createFP32({2, 3});
    auto output = TestTensorFactory::createFP32Zeros({2, 6});

    for (int i = 0; i < 6; ++i)
    {
        hidden->mutable_data()[i] = static_cast<float>(i + 1);
        embedding->mutable_data()[i] = static_cast<float>(101 + i);
    }

    MTPConcatStage stage({
        .device_id = DeviceId::cpu(),
        .hidden = hidden.get(),
        .embedding = embedding.get(),
        .output = output.get(),
        .num_tokens = 2,
        .hidden_dim = 3,
    });
    CPUDeviceContext ctx(DeviceId::cpu());
    ASSERT_TRUE(stage.execute(&ctx));

    const std::vector<float> expected = {
        1, 2, 3, 101, 102, 103,
        4, 5, 6, 104, 105, 106};
    for (size_t i = 0; i < expected.size(); ++i)
        EXPECT_FLOAT_EQ(output->data()[i], expected[i]);
}

TEST(Test__MTPGraphConstruction, BuildsDenseQwen35SidecarGraph)
{
    DenseMTPGraphFixture fixture;
    Qwen35Graph graph_builder(fixture.config, fixture.mpi);
    graph_builder.setWeights(fixture.modelWeights());

    auto weights = fixture.mtpWeights();
    auto input = fixture.input();
    auto output = fixture.output();
    ComputeGraph graph = graph_builder.buildMTPGraph(0, weights, input, output);

    ASSERT_GT(graph.size(), 0u);
    EXPECT_EQ(graph.terminalNode(), "mtp0_lm_head");

    ASSERT_NE(graph.getNode("mtp0_embedding"), nullptr);
    ASSERT_NE(graph.getNode("mtp0_norm_hidden"), nullptr);
    ASSERT_NE(graph.getNode("mtp0_norm_embedding"), nullptr);
    ASSERT_NE(graph.getNode("mtp0_concat"), nullptr);
    ASSERT_NE(graph.getNode("mtp0_fc"), nullptr);
    ASSERT_NE(graph.getNode("layer0_kv_append"), nullptr);
    ASSERT_NE(graph.getNode("layer0_attention"), nullptr);
    ASSERT_NE(graph.getNode("layer1_ffn_residual"), nullptr);
    ASSERT_NE(graph.getNode("mtp0_final_norm"), nullptr);
    ASSERT_NE(graph.getNode("mtp0_lm_head"), nullptr);

    EXPECT_EQ(graph.getNode("mtp0_concat")->stage->type(), ComputeStageType::MTP_CONCAT);
    EXPECT_EQ(graph.getNode("mtp0_fc")->stage->type(), ComputeStageType::GEMM);
    EXPECT_EQ(graph.getNode("layer0_kv_append")->stage->type(), ComputeStageType::KV_CACHE_APPEND);
    EXPECT_EQ(graph.getNode("mtp0_lm_head")->stage->type(), ComputeStageType::LM_HEAD);

    EXPECT_TRUE(hasDependency(graph, "mtp0_concat", "mtp0_norm_hidden"));
    EXPECT_TRUE(hasDependency(graph, "mtp0_concat", "mtp0_norm_embedding"));
    EXPECT_TRUE(hasDependency(graph, "mtp0_fc", "mtp0_concat"));
    EXPECT_TRUE(hasDependency(graph, "mtp0_final_norm", "layer1_ffn_residual"));
    EXPECT_TRUE(hasDependency(graph, "mtp0_lm_head", "mtp0_final_norm"));
}

TEST(Test__MTPGraphConstruction, RejectsMoESidecarForNow)
{
    DenseMTPGraphFixture fixture;
    Qwen35Graph graph_builder(fixture.config, fixture.mpi);
    graph_builder.setWeights(fixture.modelWeights());

    auto weights = fixture.mtpWeights();
    weights.fa_block.moe_gate = fixture.gate_proj.get();
    auto input = fixture.input();
    auto output = fixture.output();
    ComputeGraph graph = graph_builder.buildMTPGraph(0, weights, input, output);

    EXPECT_EQ(graph.size(), 0u);
}

TEST(Test__MTPGraphConstruction, DenseSidecarExecutionAppendsRealKVPayload)
{
    DenseMTPGraphFixture fixture;
    Qwen35Graph graph_builder(fixture.config, fixture.mpi);

    auto frozen = makeDenseMTPFrozenWeightSet(fixture);
    auto bindings = makeModelWeightBindings(*frozen);
    graph_builder.setWeightBindings(bindings);
    graph_builder.setWeights(toLegacyModelWeights(bindings));

    PreparedWeightStore store;
    prepareFrozenGemmWeightsForCPU(*frozen, store);
    graph_builder.setPreparedWeightStore(&store);

    auto input = fixture.input();
    auto output = fixture.output();
    ASSERT_FALSE(bindings.mtp.depths.empty());
    ComputeGraph graph = graph_builder.buildMTPGraph(0, bindings.mtp.depths[0], input, output);
    ASSERT_GT(graph.size(), 0u);

    CPUDeviceContext ctx(DeviceId::cpu());
    DeviceGraphExecutor executor;
    ASSERT_TRUE(executor.execute(graph, &ctx));

    ASSERT_EQ(fixture.kv_cache->get_cached_tokens(0, 0), 1);
    const auto layout = fixture.kv_cache->logicalBlockLayout(0, 1);
    ASSERT_GT(layout.k_bytes, 0u);
    ASSERT_GT(layout.v_bytes, 0u);

    std::vector<uint8_t> k_payload(layout.k_bytes, 0);
    std::vector<uint8_t> v_payload(layout.v_bytes, 0);
    IKVCache::KVCacheLogicalBlockDescriptor desc;
    desc.layer = 0;
    desc.seq_idx = 0;
    desc.logical_token_start = 0;
    desc.token_count = 1;
    ASSERT_TRUE(fixture.kv_cache->exportLogicalBlock(
        desc,
        k_payload.data(),
        v_payload.data()));

    auto fp32_abs_sum = [](const std::vector<uint8_t> &bytes)
    {
        const auto *values = reinterpret_cast<const float *>(bytes.data());
        const size_t count = bytes.size() / sizeof(float);
        float sum = 0.0f;
        for (size_t i = 0; i < count; ++i)
            sum += std::abs(values[i]);
        return sum;
    };
    EXPECT_GT(fp32_abs_sum(k_payload), 0.0f);
    EXPECT_GT(fp32_abs_sum(v_payload), 0.0f);
}

TEST(Test__MTPGraphConstruction, Qwen35PrefillPopulatesRealShiftedMTPKVPayload)
{
    DeviceManager::instance().initialize(-1, false);

    TinyQwen35MTPForwardFixture fixture;
    auto graph_builder = std::make_shared<Qwen35Graph>(fixture.config, fixture.mpi);
    DeviceGraphOrchestrator orchestrator(graph_builder, fixture.mpi);

    ASSERT_TRUE(orchestrator.initializeInferenceStateFromArena(
        /*batch_size=*/1,
        fixture.config.max_seq_len,
        DeviceId::cpu()));

    auto frozen = makeTinyQwen35MTPFrozenWeightSet(fixture);
    orchestrator.setFrozenWeightSet(std::move(frozen));
    ASSERT_NE(orchestrator.frozenWeightSet(), nullptr);

    PreparedWeightStore store;
    prepareFrozenGemmWeightsForCPU(*orchestrator.frozenWeightSet(), store);
    graph_builder->setPreparedWeightStore(&store);

    const std::vector<int> prefix_tokens = {1, 2, 3, 4};
    ASSERT_NE(orchestrator.forward(prefix_tokens.data(), static_cast<int>(prefix_tokens.size()), 1), nullptr);

    PrefixStateSnapshot snapshot = orchestrator.captureLivePrefixState();
    ASSERT_TRUE(snapshot.valid);
    ASSERT_EQ(snapshot.mtp_blocks.size(), 1u);
    EXPECT_EQ(snapshot.mtp_blocks[0].key.token_count, static_cast<int>(prefix_tokens.size()) - 1);

    const PrefixBlockHandle &mtp = snapshot.mtp_blocks[0];
    ASSERT_NE(mtp.kvKData(), nullptr);
    ASSERT_NE(mtp.kvVData(), nullptr);

    auto payload_abs_sum = [](const uint8_t *bytes, size_t count)
    {
        const auto *values = reinterpret_cast<const float *>(bytes);
        const size_t n = count / sizeof(float);
        float sum = 0.0f;
        for (size_t i = 0; i < n; ++i)
            sum += std::abs(values[i]);
        return sum;
    };
    EXPECT_GT(payload_abs_sum(mtp.kvKData(), mtp.layout.bytes_per_fa_layer_k), 0.0f);
    EXPECT_GT(payload_abs_sum(mtp.kvVData(), mtp.layout.bytes_per_fa_layer_v), 0.0f);
}

TEST(Test__MTPGraphConstruction, PrefixHarvestPersistsAndRestoresShiftedMTPKVPayload)
{
    DeviceManager::instance().initialize(-1, false);

    TinyQwen35MTPForwardFixture fixture;
    fixture.config.prefix_cache.enabled = true;
    fixture.config.prefix_cache.storage_mode = PrefixCacheStorageMode::Ram;
    fixture.config.prefix_cache.block_size = 2;
    fixture.config.prefix_cache.terminal_state = PrefixCacheTerminalStateMode::Off;
    fixture.config.prefix_cache.ram_budget_bytes = 1024ull * 1024ull;

    auto graph_builder = std::make_shared<Qwen35Graph>(fixture.config, fixture.mpi);
    DeviceGraphOrchestrator orchestrator(graph_builder, fixture.mpi);

    ASSERT_TRUE(orchestrator.initializeInferenceStateFromArena(
        /*batch_size=*/1,
        fixture.config.max_seq_len,
        DeviceId::cpu()));

    auto frozen = makeTinyQwen35MTPFrozenWeightSet(fixture);
    orchestrator.setFrozenWeightSet(std::move(frozen));
    ASSERT_NE(orchestrator.frozenWeightSet(), nullptr);

    PreparedWeightStore store;
    prepareFrozenGemmWeightsForCPU(*orchestrator.frozenWeightSet(), store);
    graph_builder->setPreparedWeightStore(&store);

    const std::vector<int> prompt_tokens = {1, 2, 3, 4};
    const std::vector<int32_t> prefix_tokens(prompt_tokens.begin(), prompt_tokens.end());
    ASSERT_NE(orchestrator.forward(prompt_tokens.data(), static_cast<int>(prompt_tokens.size()), 1), nullptr);

    PrefixStateSnapshot before = orchestrator.captureLivePrefixState();
    ASSERT_TRUE(before.valid);
    ASSERT_EQ(before.mtp_blocks.size(), 1u);
    ASSERT_NE(before.mtp_blocks[0].kv_storage, nullptr);
    EXPECT_EQ(before.mtp_blocks[0].key.token_count, static_cast<int>(prompt_tokens.size()) - 1);

    ASSERT_TRUE(orchestrator.harvestPrefix(prefix_tokens, static_cast<int>(prefix_tokens.size())));
    orchestrator.clear_cache();

    PrefixLookupResult hit = orchestrator.lookupPrefix(prefix_tokens);
    ASSERT_TRUE(hit.supported);
    EXPECT_EQ(hit.cached_tokens, static_cast<int>(prefix_tokens.size()));
    ASSERT_EQ(hit.blocks.size(), 2u);
    for (const auto &block : hit.blocks)
    {
        ASSERT_TRUE(block.layout.includes_mtp_state);
        ASSERT_NE(block.mtp_storage, nullptr);
        EXPECT_EQ(block.mtp_storage->size(), block.layout.mtpKVBytes());
    }

    ASSERT_TRUE(orchestrator.populatePrefix(hit));
    PrefixStateSnapshot restored = orchestrator.captureLivePrefixState();
    ASSERT_TRUE(restored.valid);
    ASSERT_EQ(restored.mtp_blocks.size(), 1u);
    ASSERT_NE(restored.mtp_blocks[0].kv_storage, nullptr);
    EXPECT_EQ(restored.cached_tokens, static_cast<int>(prefix_tokens.size()));
    EXPECT_EQ(restored.mtp_blocks[0].key.token_count, static_cast<int>(prefix_tokens.size()) - 1);
    EXPECT_EQ(*restored.mtp_blocks[0].kv_storage, *before.mtp_blocks[0].kv_storage);
}

TEST(Test__MTPGraphConstruction, CPUForwardUpdatesShiftedMTPCacheProbe)
{
    DeviceManager::instance().initialize(-1, false);

    TinyQwenForwardFixture fixture(DeviceId::cpu(), KVCachePrecision::FP32);
    auto graph_builder = std::make_shared<QwenStandardGraph>(fixture.config, fixture.mpi);
    DeviceGraphOrchestrator orchestrator(graph_builder, fixture.mpi);

    ASSERT_TRUE(orchestrator.initializeInferenceStateFromArena(
        /*batch_size=*/1,
        fixture.config.max_seq_len,
        DeviceId::cpu()));
    orchestrator.setWeights(fixture.modelWeights());
    PreparedWeightStore prepared_store;
    ASSERT_NO_THROW(prepareDenseForwardWeights(orchestrator, *graph_builder, prepared_store, DeviceId::cpu()));

    const std::vector<int> prefix_tokens = {1, 2, 3, 4};
    ASSERT_NE(orchestrator.forward(prefix_tokens.data(), static_cast<int>(prefix_tokens.size()), 1), nullptr);

    const auto after_prefill = orchestrator.prefixStateProbe();
    ASSERT_FALSE(after_prefill.kv_caches.empty());
    ASSERT_EQ(after_prefill.mtp_kv_caches.size(), 1u);
    ASSERT_FALSE(after_prefill.positions.empty());
    ASSERT_FALSE(after_prefill.sequence_lengths.empty());

    EXPECT_EQ(after_prefill.positions[0], static_cast<int>(prefix_tokens.size()));
    EXPECT_EQ(after_prefill.sequence_lengths[0], static_cast<int>(prefix_tokens.size()));
    EXPECT_EQ(maxCachedTokens(after_prefill.kv_caches), static_cast<int>(prefix_tokens.size()));
    EXPECT_EQ(after_prefill.mtp_kv_caches[0].owner, "mtp:0");
    EXPECT_EQ(after_prefill.mtp_kv_caches[0].n_layers, 1);
    EXPECT_EQ(maxCachedTokens(after_prefill.mtp_kv_caches), static_cast<int>(prefix_tokens.size()) - 1);
    EXPECT_EQ(after_prefill.totalMTPCachedTokens(), static_cast<int>(prefix_tokens.size()) - 1);

    orchestrator.clear_cache();
    const auto after_clear = orchestrator.prefixStateProbe();
    EXPECT_EQ(maxCachedTokens(after_clear.mtp_kv_caches), 0);
    EXPECT_EQ(after_clear.totalMTPCachedTokens(), 0);
    EXPECT_TRUE(std::all_of(after_clear.positions.begin(), after_clear.positions.end(),
                            [](int value) { return value == 0; }));
    EXPECT_TRUE(std::all_of(after_clear.sequence_lengths.begin(), after_clear.sequence_lengths.end(),
                            [](int value) { return value == 0; }));
}

TEST(Test__MTPGraphConstruction, LivePrefixSnapshotRestoresDenseCPUState)
{
    DeviceManager::instance().initialize(-1, false);

    TinyQwenForwardFixture fixture(DeviceId::cpu(), KVCachePrecision::FP32);
    auto graph_builder = std::make_shared<QwenStandardGraph>(fixture.config, fixture.mpi);
    DeviceGraphOrchestrator orchestrator(graph_builder, fixture.mpi);

    ASSERT_TRUE(orchestrator.initializeInferenceStateFromArena(
        /*batch_size=*/1,
        fixture.config.max_seq_len,
        DeviceId::cpu()));
    orchestrator.setWeights(fixture.modelWeights());
    PreparedWeightStore prepared_store;
    ASSERT_NO_THROW(prepareDenseForwardWeights(orchestrator, *graph_builder, prepared_store, DeviceId::cpu()));

    const std::vector<int> prefix_tokens = {1, 2, 3};
    ASSERT_NE(orchestrator.forward(prefix_tokens.data(), static_cast<int>(prefix_tokens.size()), 1), nullptr);

    PrefixStateSnapshot snapshot = orchestrator.captureLivePrefixState();
    ASSERT_TRUE(snapshot.valid);
    EXPECT_EQ(snapshot.cached_tokens, static_cast<int>(prefix_tokens.size()));
    ASSERT_EQ(snapshot.blocks.size(), 1u);
    ASSERT_EQ(snapshot.mtp_blocks.size(), 1u);
    EXPECT_EQ(snapshot.mtp_blocks[0].key.block_index, 0);
    EXPECT_EQ(snapshot.mtp_blocks[0].key.token_count, static_cast<int>(prefix_tokens.size()) - 1);

    orchestrator.clear_cache();
    const auto after_clear = orchestrator.prefixStateProbe();
    EXPECT_EQ(maxCachedTokens(after_clear.kv_caches), 0);
    EXPECT_EQ(maxCachedTokens(after_clear.mtp_kv_caches), 0);

    ASSERT_TRUE(orchestrator.restoreLivePrefixState(snapshot));
    const auto after_restore = orchestrator.prefixStateProbe();
    ASSERT_FALSE(after_restore.positions.empty());
    ASSERT_FALSE(after_restore.sequence_lengths.empty());
    EXPECT_EQ(after_restore.positions[0], static_cast<int>(prefix_tokens.size()));
    EXPECT_EQ(after_restore.sequence_lengths[0], static_cast<int>(prefix_tokens.size()));
    EXPECT_EQ(maxCachedTokens(after_restore.kv_caches), static_cast<int>(prefix_tokens.size()));
    EXPECT_EQ(maxCachedTokens(after_restore.mtp_kv_caches), static_cast<int>(prefix_tokens.size()) - 1);

    ASSERT_TRUE(orchestrator.truncateLivePrefixState(1));
    const auto after_truncate = orchestrator.prefixStateProbe();
    EXPECT_EQ(maxCachedTokens(after_truncate.kv_caches), 1);
    EXPECT_EQ(maxCachedTokens(after_truncate.mtp_kv_caches), 0);

    ASSERT_TRUE(orchestrator.restoreLivePrefixState(snapshot));
    const auto after_second_restore = orchestrator.prefixStateProbe();
    EXPECT_EQ(maxCachedTokens(after_second_restore.kv_caches), static_cast<int>(prefix_tokens.size()));
    EXPECT_EQ(maxCachedTokens(after_second_restore.mtp_kv_caches), static_cast<int>(prefix_tokens.size()) - 1);
}

TEST(Test__MTPGraphConstruction, LiveForwardExposesAllPositionLogitsOnCPU)
{
    DeviceManager::instance().initialize(-1, false);

    TinyQwenForwardFixture fixture(DeviceId::cpu(), KVCachePrecision::FP32);
    auto graph_builder = std::make_shared<QwenStandardGraph>(fixture.config, fixture.mpi);
    DeviceGraphOrchestrator orchestrator(graph_builder, fixture.mpi);

    ASSERT_TRUE(orchestrator.initializeInferenceStateFromArena(
        /*batch_size=*/1,
        fixture.config.max_seq_len,
        DeviceId::cpu()));
    orchestrator.setWeights(fixture.modelWeights());
    PreparedWeightStore prepared_store;
    ASSERT_NO_THROW(prepareDenseForwardWeights(orchestrator, *graph_builder, prepared_store, DeviceId::cpu()));

    ASSERT_TRUE(orchestrator.setComputeAllPositionLogits(true));
    const std::vector<int> tokens = {1, 2, 3};
    const float *returned_logits = orchestrator.forward(tokens.data(), static_cast<int>(tokens.size()), 1);
    ASSERT_NE(returned_logits, nullptr);

    const float *all_logits = orchestrator.getAllPositionLogits();
    ASSERT_NE(all_logits, nullptr);
    EXPECT_EQ(returned_logits, all_logits);

    const size_t count = static_cast<size_t>(tokens.size()) *
                         static_cast<size_t>(fixture.config.vocab_size);
    float abs_sum = 0.0f;
    for (size_t i = 0; i < count; ++i)
    {
        ASSERT_TRUE(std::isfinite(all_logits[i])) << "non-finite all-position logit at " << i;
        abs_sum += std::abs(all_logits[i]);
    }
    EXPECT_GT(abs_sum, 0.0f);

    ASSERT_TRUE(orchestrator.setComputeAllPositionLogits(false));
}
