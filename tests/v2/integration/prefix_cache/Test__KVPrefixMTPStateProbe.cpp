#include <gtest/gtest.h>

#include "backends/ComputeBackend.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/GlobalDeviceAddress.h"
#include "backends/IWorkerGPUContext.h"
#include "config/OrchestrationConfig.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/device/WorkspaceAllocator.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "execution/mtp/MTPWeightManifest.h"
#include "kernels/KernelFactory.h"
#include "loaders/PreparedWeightStore.h"
#include "execution/runner/IOrchestrationRunnerFactory.h"
#include "loaders/ModelLoader.h"
#include "models/qwen/QwenStandardGraph.h"
#include "models/qwen35/Qwen35Graph.h"
#include "utils/MPIContext.h"
#include "utils/Sampler.h"
#include "utils/TestTensorFactory.h"
#include "utils/Tokenizer.h"

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    constexpr const char *kDenseModelPath = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

    std::optional<std::string> firstGpuDeviceSpec()
    {
        auto &dm = DeviceManager::instance();
        dm.initialize(-1, false);
        if (dm.cuda_device_count() > 0)
        {
            return std::string("cuda:0");
        }
        if (dm.rocm_device_count() > 0)
        {
            return std::string("rocm:0");
        }
        return std::nullopt;
    }

    std::optional<DeviceId> firstGpuDeviceId()
    {
        auto &dm = DeviceManager::instance();
        dm.initialize(-1, false);
        if (dm.cuda_device_count() > 0)
        {
            return DeviceId::cuda(0);
        }
        if (dm.rocm_device_count() > 0)
        {
            return DeviceId::rocm(0);
        }
        return std::nullopt;
    }

    OrchestrationConfig makeSingleGpuConfig(const std::string &device_spec)
    {
        OrchestrationConfig config = OrchestrationConfig::defaults();
        config.model_path = kDenseModelPath;
        config.max_seq_len = 32;
        config.batch_size = 1;
        config.tp_degree = 1;
        config.pp_degree = 1;
        config.activation_precision = "fp32";
        config.kv_cache_precision = "fp16";
        auto parsed = GlobalDeviceAddress::tryParse(device_spec);
        if (!parsed)
        {
            throw std::runtime_error("invalid GPU device spec: " + device_spec);
        }
        config.device_for_this_rank = *parsed;
        return config;
    }

    OrchestrationConfig makeSingleGpuPrefixCacheConfig(const std::string &device_spec)
    {
        OrchestrationConfig config = makeSingleGpuConfig(device_spec);
        config.prefix_cache.enabled = true;
        config.prefix_cache.storage_mode = PrefixCacheStorageMode::Ram;
        config.prefix_cache.block_size = 2;
        config.prefix_cache.terminal_state = PrefixCacheTerminalStateMode::Auto;
        config.prefix_cache.ram_budget_bytes = 64ull * 1024ull * 1024ull;
        return config;
    }

    OrchestrationConfig makeSingleGpuTieredPrefixCacheConfig(
        const std::string &device_spec,
        const std::filesystem::path &disk_dir)
    {
        OrchestrationConfig config = makeSingleGpuPrefixCacheConfig(device_spec);
        config.prefix_cache.storage_mode = PrefixCacheStorageMode::Tiered;
        config.prefix_cache.ram_budget_bytes = 1ull * 1024ull * 1024ull;
        config.prefix_cache.device_budget_bytes = 0;
        config.prefix_cache.disk_budget_bytes = 32ull * 1024ull * 1024ull;
        config.prefix_cache.disk_dir = disk_dir.string();
        return config;
    }

    OrchestrationConfig makeSingleGpuDeviceHotPrefixCacheConfig(
        const std::string &device_spec,
        const std::filesystem::path &disk_dir)
    {
        OrchestrationConfig config = makeSingleGpuPrefixCacheConfig(device_spec);
        config.prefix_cache.storage_mode = PrefixCacheStorageMode::Tiered;
        config.prefix_cache.ram_budget_bytes = 1ull * 1024ull * 1024ull;
        config.prefix_cache.device_budget_bytes = 32ull * 1024ull * 1024ull;
        config.prefix_cache.disk_budget_bytes = 32ull * 1024ull * 1024ull;
        config.prefix_cache.disk_dir = disk_dir.string();
        return config;
    }

    OrchestrationConfig makeSingleCpuConfig(bool prefix_cache_enabled)
    {
        OrchestrationConfig config = OrchestrationConfig::defaults();
        config.model_path = kDenseModelPath;
        config.max_seq_len = 16;
        config.batch_size = 1;
        config.tp_degree = 1;
        config.pp_degree = 1;
        config.activation_precision = "fp32";
        config.kv_cache_precision = "q16_1";
        config.device_for_this_rank = GlobalDeviceAddress::cpu();
        config.prefix_cache.enabled = prefix_cache_enabled;
        config.prefix_cache.storage_mode = PrefixCacheStorageMode::Ram;
        config.prefix_cache.block_size = 2;
        config.prefix_cache.terminal_state = PrefixCacheTerminalStateMode::Auto;
        config.prefix_cache.ram_budget_bytes = 64ull * 1024ull * 1024ull;
        return config;
    }

    int maxLayerCachedTokens(const PrefixRuntimeStateSnapshot &snapshot)
    {
        int max_tokens = 0;
        for (const auto &cache : snapshot.kv_caches)
        {
            for (const auto &layer : cache.layers)
            {
                max_tokens = std::max(max_tokens, layer.cached_tokens);
            }
        }
        return max_tokens;
    }

    int maxCachedTokensIn(const std::vector<PrefixKVCacheProbe> &caches)
    {
        int max_tokens = 0;
        for (const auto &cache : caches)
        {
            for (const auto &layer : cache.layers)
            {
                max_tokens = std::max(max_tokens, layer.cached_tokens);
            }
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

    bool allValuesZero(const std::vector<int> &values)
    {
        return std::all_of(values.begin(), values.end(), [](int value) { return value == 0; });
    }

    std::string lowercase(std::string value)
    {
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        return value;
    }

    bool isMTPInventoryKey(const std::string &name)
    {
        const std::string lower = lowercase(name);
        return lower.find("mtp") != std::string::npos ||
               lower.find("nextn") != std::string::npos;
    }

    std::filesystem::path tempPrefixDiskDir()
    {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        return std::filesystem::temp_directory_path() /
               ("llaminar_prefix_cache_integration_" + std::to_string(stamp));
    }

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
        std::shared_ptr<MPIContext> mpi = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);

        std::unique_ptr<FP32Tensor> embedding_table;
        std::unique_ptr<FP32Tensor> final_norm;
        std::unique_ptr<FP32Tensor> lm_head;
        std::vector<LayerTensors> layers;

        explicit TinyQwenForwardFixture(DeviceId device)
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
            config.kv_cache_precision = KVCachePrecision::FP16;
            config.use_graph_buffer_management = true;
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

            layers.resize(static_cast<size_t>(config.n_layers));
            for (int i = 0; i < config.n_layers; ++i)
            {
                auto &layer = layers[static_cast<size_t>(i)];
                layer.attn_norm = TestTensorFactory::createFP32Ones({d});
                layer.wq = TestTensorFactory::createFP32Random({q_dim, d}, -0.02f, 0.02f, 210 + i);
                layer.wk = TestTensorFactory::createFP32Random({kv_dim, d}, -0.02f, 0.02f, 220 + i);
                layer.wv = TestTensorFactory::createFP32Random({kv_dim, d}, -0.02f, 0.02f, 230 + i);
                layer.wo = TestTensorFactory::createFP32Random({d, q_dim}, -0.02f, 0.02f, 240 + i);
                layer.ffn_norm = TestTensorFactory::createFP32Ones({d});
                layer.gate_proj = TestTensorFactory::createFP32Random({ff, d}, -0.02f, 0.02f, 250 + i);
                layer.up_proj = TestTensorFactory::createFP32Random({ff, d}, -0.02f, 0.02f, 260 + i);
                layer.down_proj = TestTensorFactory::createFP32Random({d, ff}, -0.02f, 0.02f, 270 + i);
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

    struct TinyMTPSidecarFixture
    {
        GraphConfig config;
        std::shared_ptr<MPIContext> mpi = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);

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

        std::vector<std::unique_ptr<WeightBinding>> binding_storage;
        ModelWeightBindings bindings;
        MTPDepthWeightBindings mtp_depth_bindings;
        uint64_t next_binding_id = 1;

        std::unique_ptr<IKVCache> kv_cache;
        int draft_token = 17;
        int position_id = 0;
        std::vector<int> sequence_lengths{0};
        DeviceId device;

        explicit TinyMTPSidecarFixture(DeviceId target_device)
            : device(target_device)
        {
            config.n_layers = 2;
            config.total_n_layers = 2;
            config.d_model = 64;
            config.n_heads = 4;
            config.n_kv_heads = 2;
            config.head_dim = 16;
            config.d_ff = 128;
            config.vocab_size = 128;
            config.rms_norm_eps = 1e-6f;
            config.rope_theta = 10000.0f;
            config.partial_rotary_factor = 1.0f;
            config.default_device = device;
            config.max_seq_len = 8;
            config.activation_precision = ActivationPrecision::FP32;
            config.kv_cache_precision = KVCachePrecision::FP16;
            config.layer_types = {"full_attention", "full_attention"};

            const size_t d = static_cast<size_t>(config.d_model);
            const size_t q_dim = static_cast<size_t>(config.n_heads * config.head_dim);
            const size_t kv_dim = static_cast<size_t>(config.n_kv_heads * config.head_dim);
            const size_t ff = static_cast<size_t>(config.d_ff);
            const size_t vocab = static_cast<size_t>(config.vocab_size);

            embedding_table = TestTensorFactory::createFP32Random({vocab, d}, -0.02f, 0.02f, 301);
            lm_head = TestTensorFactory::createFP32Random({vocab, d}, -0.02f, 0.02f, 302);
            fc = TestTensorFactory::createFP32Random({d, d * 2}, -0.02f, 0.02f, 303);
            pre_hidden_norm = TestTensorFactory::createFP32Ones({d});
            pre_embedding_norm = TestTensorFactory::createFP32Ones({d});
            final_norm = TestTensorFactory::createFP32Ones({d});
            attn_norm = TestTensorFactory::createFP32Ones({d});
            wq = TestTensorFactory::createFP32Random({q_dim * 2, d}, -0.02f, 0.02f, 304);
            wk = TestTensorFactory::createFP32Random({kv_dim, d}, -0.02f, 0.02f, 305);
            wv = TestTensorFactory::createFP32Random({kv_dim, d}, -0.02f, 0.02f, 306);
            wo = TestTensorFactory::createFP32Random({d, q_dim}, -0.02f, 0.02f, 307);
            q_norm = TestTensorFactory::createFP32Ones({static_cast<size_t>(config.head_dim)});
            k_norm = TestTensorFactory::createFP32Ones({static_cast<size_t>(config.head_dim)});
            ffn_norm = TestTensorFactory::createFP32Ones({d});
            gate_proj = TestTensorFactory::createFP32Random({ff, d}, -0.02f, 0.02f, 308);
            up_proj = TestTensorFactory::createFP32Random({ff, d}, -0.02f, 0.02f, 309);
            down_proj = TestTensorFactory::createFP32Random({d, ff}, -0.02f, 0.02f, 310);

            terminal_hidden = TestTensorFactory::createFP32Random({1, d}, -0.02f, 0.02f, 311);
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

            llaminar::v2::kernels::KVCacheConfig kv_config;
            kv_config.precision = ActivationPrecision::FP32;
            kv_config.device = device;
            kv_config.num_layers = 1;
            kv_config.batch_size = 1;
            kv_config.max_seq_len = 8;
            kv_config.n_kv_heads = config.n_kv_heads;
            kv_config.head_dim = config.head_dim;
            kv_config.mpi_ctx = mpi.get();
            kv_cache = llaminar::v2::kernels::KernelFactory::createKVCache(kv_config);

            buildBindings();
        }

        const WeightBinding *addBinding(TensorBase *tensor,
                                        std::string canonical_name,
                                        WeightRole role,
                                        int layer = -1)
        {
            auto binding = std::make_unique<WeightBinding>();
            binding->binding_id = next_binding_id++;
            binding->identity.canonical_name = std::move(canonical_name);
            binding->identity.logical_id = binding->binding_id;
            binding->identity.role = role;
            binding->identity.layer = layer;
            binding->residency.home_device = device;
            binding->residency.resident_device = device;
            binding->tensor = tensor;
            binding->immutable = true;
            const WeightBinding *ptr = binding.get();
            binding_storage.push_back(std::move(binding));
            return ptr;
        }

        void buildBindings()
        {
            bindings.embedding_table = addBinding(embedding_table.get(), "token_embd.weight", WeightRole::Embedding);
            bindings.lm_head = addBinding(lm_head.get(), "output.weight", WeightRole::LMHead);
            mtp_depth_bindings.depth_index = 0;
            mtp_depth_bindings.source_layer_index = 64;
            mtp_depth_bindings.nextn_block_layout = true;
            mtp_depth_bindings.fc = addBinding(fc.get(), "blk.64.nextn.eh_proj.weight", WeightRole::Other, 64);
            mtp_depth_bindings.pre_fc_norm_hidden = addBinding(pre_hidden_norm.get(), "blk.64.nextn.hnorm.weight", WeightRole::Norm, 64);
            mtp_depth_bindings.pre_fc_norm_embedding = addBinding(pre_embedding_norm.get(), "blk.64.nextn.enorm.weight", WeightRole::Norm, 64);
            mtp_depth_bindings.final_norm = addBinding(final_norm.get(), "blk.64.nextn.shared_head_norm.weight", WeightRole::Norm, 64);
            mtp_depth_bindings.fa_block.attn_norm = addBinding(attn_norm.get(), "blk.64.attn_norm.weight", WeightRole::Norm, 64);
            mtp_depth_bindings.fa_block.wq = addBinding(wq.get(), "blk.64.attn_q.weight", WeightRole::AttentionQ, 64);
            mtp_depth_bindings.fa_block.wk = addBinding(wk.get(), "blk.64.attn_k.weight", WeightRole::AttentionK, 64);
            mtp_depth_bindings.fa_block.wv = addBinding(wv.get(), "blk.64.attn_v.weight", WeightRole::AttentionV, 64);
            mtp_depth_bindings.fa_block.wo = addBinding(wo.get(), "blk.64.attn_output.weight", WeightRole::AttentionWO, 64);
            mtp_depth_bindings.fa_block.q_norm = addBinding(q_norm.get(), "blk.64.attn_q_norm.weight", WeightRole::Norm, 64);
            mtp_depth_bindings.fa_block.k_norm = addBinding(k_norm.get(), "blk.64.attn_k_norm.weight", WeightRole::Norm, 64);
            mtp_depth_bindings.fa_block.ffn_norm = addBinding(ffn_norm.get(), "blk.64.post_attention_norm.weight", WeightRole::Norm, 64);
            mtp_depth_bindings.fa_block.gate_proj = addBinding(gate_proj.get(), "blk.64.ffn_gate.weight", WeightRole::FFNGate, 64);
            mtp_depth_bindings.fa_block.up_proj = addBinding(up_proj.get(), "blk.64.ffn_up.weight", WeightRole::FFNUp, 64);
            mtp_depth_bindings.fa_block.down_proj = addBinding(down_proj.get(), "blk.64.ffn_down.weight", WeightRole::FFNDown, 64);
            bindings.mtp.depth = 1;
            bindings.mtp.depths.push_back(mtp_depth_bindings);
        }

        void prepareWeights(PreparedWeightStore &store)
        {
            for (const auto &owned : binding_storage)
            {
                ASSERT_NE(owned, nullptr);
                ASSERT_NE(owned->tensor, nullptr);
                ASSERT_TRUE(owned->tensor->ensureOnDevice(device));
                if (owned->tensor->shape().size() == 2 &&
                    owned->identity.role != WeightRole::Embedding)
                {
                    store.prepareGemm(*owned);
                }
            }
            ASSERT_TRUE(terminal_hidden->ensureOnDevice(device));

            const std::vector<TensorBase *> activation_outputs = {
                embedding.get(),
                norm_hidden.get(),
                norm_embedding.get(),
                concat.get(),
                projected.get(),
                hidden.get(),
                q.get(),
                k.get(),
                v.get(),
                q_raw.get(),
                q_gate.get(),
                attn_output.get(),
                attn_proj.get(),
                gate.get(),
                up.get(),
                ffn_output.get(),
                logits.get(),
            };
            for (TensorBase *tensor : activation_outputs)
            {
                ASSERT_NE(tensor, nullptr);
                ASSERT_TRUE(tensor->allocateOnDevice(device));
            }
        }

        MTPForwardInput input()
        {
            MTPForwardInput in;
            in.draft_token_ids = &draft_token;
            in.terminal_hidden = terminal_hidden.get();
            in.kv_cache = kv_cache.get();
            in.position_ids = &position_id;
            in.sequence_lengths = &sequence_lengths;
            in.batch_size = 1;
            in.seq_len = 1;
            in.device = device;
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
} // namespace

TEST(Test__KVPrefixMTPStateProbe, DenseQwen25_ResetStateInventory)
{
    if (!std::filesystem::exists(kDenseModelPath))
    {
        GTEST_SKIP() << "Dense probe model not found: " << kDenseModelPath;
    }

    const auto device_spec = firstGpuDeviceSpec();
    if (!device_spec)
    {
        GTEST_SKIP() << "No CUDA or ROCm GPU available for prefix-cache state probe";
    }

    auto factory = createOrchestrationRunnerFactory();
    auto runner = factory->createFromOrchestrationConfig(makeSingleGpuConfig(*device_spec));
    ASSERT_NE(runner, nullptr);
    ASSERT_TRUE(runner->initialize()) << runner->lastError();

    SamplingParams greedy;
    greedy.temperature = 0.0f;
    runner->setSamplingParams(greedy);

    const auto initial = runner->prefixStateProbe();
    EXPECT_TRUE(initial.initialized);
    EXPECT_FALSE(initial.prefill_logits_ready);
    EXPECT_EQ(initial.current_position, 0);
    EXPECT_EQ(maxLayerCachedTokens(initial), 0);

    const std::vector<int32_t> prefix_tokens = {1, 2, 3, 4};
    ASSERT_TRUE(runner->prefill(prefix_tokens)) << runner->lastError();

    const auto after_prefill = runner->prefixStateProbe();
    EXPECT_TRUE(after_prefill.prefill_logits_ready);
    EXPECT_EQ(after_prefill.current_position, static_cast<int>(prefix_tokens.size()));
    ASSERT_FALSE(after_prefill.kv_caches.empty());
    EXPECT_EQ(maxLayerCachedTokens(after_prefill), static_cast<int>(prefix_tokens.size()));
    ASSERT_FALSE(after_prefill.positions.empty());
    EXPECT_EQ(after_prefill.positions[0], static_cast<int>(prefix_tokens.size()));
    ASSERT_FALSE(after_prefill.sequence_lengths.empty());
    EXPECT_EQ(after_prefill.sequence_lengths[0], static_cast<int>(prefix_tokens.size()));

    auto first_decode = runner->decodeStep();
    ASSERT_TRUE(first_decode.error.empty()) << first_decode.error;

    const auto after_first_decode = runner->prefixStateProbe();
    EXPECT_FALSE(after_first_decode.prefill_logits_ready);
    EXPECT_EQ(after_first_decode.current_position, static_cast<int>(prefix_tokens.size()));
    EXPECT_EQ(maxLayerCachedTokens(after_first_decode), static_cast<int>(prefix_tokens.size()));

    auto second_decode = runner->decodeStep();
    ASSERT_TRUE(second_decode.error.empty()) << second_decode.error;

    const auto after_second_decode = runner->prefixStateProbe();
    EXPECT_EQ(after_second_decode.current_position, static_cast<int>(prefix_tokens.size()) + 1);
    EXPECT_EQ(maxLayerCachedTokens(after_second_decode), static_cast<int>(prefix_tokens.size()) + 1);

    runner->clearCache();
    const auto after_clear = runner->prefixStateProbe();
    EXPECT_FALSE(after_clear.prefill_logits_ready);
    EXPECT_EQ(after_clear.current_position, 0);
    EXPECT_EQ(maxLayerCachedTokens(after_clear), 0);
    EXPECT_TRUE(allValuesZero(after_clear.positions));
    EXPECT_TRUE(allValuesZero(after_clear.sequence_lengths));
    EXPECT_GT(after_clear.session_epoch, after_second_decode.session_epoch);
}

TEST(Test__KVPrefixMTPStateProbe, DenseQwen25_GPUCacheFlagPreservesGreedyInference)
{
    if (!std::filesystem::exists(kDenseModelPath))
    {
        GTEST_SKIP() << "Dense probe model not found: " << kDenseModelPath;
    }

    const auto device_spec = firstGpuDeviceSpec();
    if (!device_spec)
    {
        GTEST_SKIP() << "No CUDA or ROCm GPU available for prefix-cache GPU integration probe";
    }

    auto factory = createOrchestrationRunnerFactory();
    auto runner = factory->createFromOrchestrationConfig(makeSingleGpuPrefixCacheConfig(*device_spec));
    ASSERT_NE(runner, nullptr);
    ASSERT_TRUE(runner->initialize()) << runner->lastError();

    SamplingParams greedy;
    greedy.temperature = 0.0f;

    const std::vector<int32_t> prompt = {1, 2, 3, 4};
    auto first = runner->generate(prompt, 2, greedy);
    ASSERT_TRUE(first.error.empty()) << first.error;
    ASSERT_EQ(first.tokens.size(), 2u);

    const auto after_first = runner->prefixStateProbe();
    EXPECT_TRUE(after_first.prefix_cache_ready);
    EXPECT_GE(after_first.prefix_cache_inserts, 2u)
        << "First GPU prompt should harvest both 2-token dense prefix blocks";

    auto second = runner->generate(prompt, 2, greedy);
    ASSERT_TRUE(second.error.empty()) << second.error;
    ASSERT_EQ(second.tokens.size(), 2u);
    EXPECT_EQ(second.tokens.front(), first.tokens.front())
        << "Full-hit terminal logits should preserve the first greedy token";

    const auto after_repeated_prompt = runner->prefixStateProbe();
    EXPECT_TRUE(after_repeated_prompt.initialized);
    EXPECT_TRUE(after_repeated_prompt.prefix_cache_ready);
    EXPECT_GE(after_repeated_prompt.prefix_cache_hits, 2u)
        << "Second GPU prompt should reuse both cached 2-token dense prefix blocks";
    EXPECT_GE(after_repeated_prompt.current_position, static_cast<int>(prompt.size()) + 1)
        << "Second decode step should advance over the restored/imported prefix KV state";
    EXPECT_GT(maxLayerCachedTokens(after_repeated_prompt), 0);
}

TEST(Test__KVPrefixMTPStateProbe, DenseQwen25_GPUDeviceHotTierHydratesEvictedBlocks)
{
    if (!std::filesystem::exists(kDenseModelPath))
    {
        GTEST_SKIP() << "Dense probe model not found: " << kDenseModelPath;
    }

    const auto device_spec = firstGpuDeviceSpec();
    if (!device_spec)
    {
        GTEST_SKIP() << "No CUDA or ROCm GPU available for device-hot prefix-cache probe";
    }

    const auto disk_dir = tempPrefixDiskDir();
    const auto cleanup = [&]() { std::filesystem::remove_all(disk_dir); };

    auto factory = createOrchestrationRunnerFactory();
    auto runner = factory->createFromOrchestrationConfig(
        makeSingleGpuDeviceHotPrefixCacheConfig(*device_spec, disk_dir));
    ASSERT_NE(runner, nullptr);
    ASSERT_TRUE(runner->initialize()) << runner->lastError();

    SamplingParams greedy;
    greedy.temperature = 0.0f;
    const std::vector<int32_t> prompt = {1, 2, 3, 4};

    auto first = runner->generate(prompt, 1, greedy);
    ASSERT_TRUE(first.error.empty()) << first.error;
    ASSERT_EQ(first.tokens.size(), 1u);

    const auto after_first = runner->prefixStateProbe();
    EXPECT_TRUE(after_first.prefix_cache_ready);
    EXPECT_GE(after_first.prefix_cache_inserts, 2u);
    EXPECT_GE(after_first.prefix_cache_evictions, 1u);
    EXPECT_GT(after_first.prefix_cache_device_bytes, 0u)
        << "Tiered GPU cache should keep a device-hot mirror of evicted blocks";
    EXPECT_GT(after_first.prefix_cache_disk_bytes, 0u)
        << "RAM eviction should still persist a durable disk copy";
    EXPECT_GE(after_first.prefix_cache_promotions, 2u);

    auto second = runner->generate(prompt, 1, greedy);
    ASSERT_TRUE(second.error.empty()) << second.error;
    ASSERT_EQ(second.tokens.size(), 1u);
    EXPECT_EQ(second.tokens.front(), first.tokens.front());

    const auto after_second = runner->prefixStateProbe();
    EXPECT_GE(after_second.prefix_cache_hits, 2u);
    EXPECT_EQ(after_second.prefix_cache_disk_hydrations, 0u)
        << "Device-hot mirrors should hydrate evicted blocks before disk fallback";
    EXPECT_GT(after_second.prefix_cache_device_bytes, 0u);

    cleanup();
}

TEST(Test__KVPrefixMTPStateProbe, DenseQwen25_GPUTieredDiskCacheHydratesEvictedBlocks)
{
    if (!std::filesystem::exists(kDenseModelPath))
    {
        GTEST_SKIP() << "Dense probe model not found: " << kDenseModelPath;
    }

    const auto device_spec = firstGpuDeviceSpec();
    if (!device_spec)
    {
        GTEST_SKIP() << "No CUDA or ROCm GPU available for tiered prefix-cache probe";
    }

    const auto disk_dir = tempPrefixDiskDir();
    const auto cleanup = [&]() { std::filesystem::remove_all(disk_dir); };

    auto factory = createOrchestrationRunnerFactory();
    auto runner = factory->createFromOrchestrationConfig(
        makeSingleGpuTieredPrefixCacheConfig(*device_spec, disk_dir));
    ASSERT_NE(runner, nullptr);
    ASSERT_TRUE(runner->initialize()) << runner->lastError();

    SamplingParams greedy;
    greedy.temperature = 0.0f;
    const std::vector<int32_t> prompt = {1, 2, 3, 4};

    auto first = runner->generate(prompt, 1, greedy);
    ASSERT_TRUE(first.error.empty()) << first.error;
    ASSERT_EQ(first.tokens.size(), 1u);

    const auto after_first = runner->prefixStateProbe();
    EXPECT_TRUE(after_first.prefix_cache_ready);
    EXPECT_GE(after_first.prefix_cache_inserts, 2u);
    EXPECT_GE(after_first.prefix_cache_evictions, 1u)
        << "A one-block RAM budget should force tiered eviction";
    EXPECT_GT(after_first.prefix_cache_disk_bytes, 0u);

    auto second = runner->generate(prompt, 1, greedy);
    ASSERT_TRUE(second.error.empty()) << second.error;
    ASSERT_EQ(second.tokens.size(), 1u);
    EXPECT_EQ(second.tokens.front(), first.tokens.front())
        << "Disk-hydrated full hits should preserve terminal logits";

    const auto after_second = runner->prefixStateProbe();
    EXPECT_TRUE(after_second.prefix_cache_ready);
    EXPECT_GE(after_second.prefix_cache_hits, 2u);
    EXPECT_GE(after_second.prefix_cache_disk_hydrations, 2u)
        << "Both 2-token blocks should be served through disk hydration";
    EXPECT_GT(after_second.prefix_cache_disk_bytes, 0u);

    cleanup();
}

TEST(Test__KVPrefixMTPStateProbe, DenseQwen25_CPUPrefixCacheFullHitRecordsReuse)
{
    if (!std::filesystem::exists(kDenseModelPath))
    {
        GTEST_SKIP() << "Dense probe model not found: " << kDenseModelPath;
    }

    SamplingParams greedy;
    greedy.temperature = 0.0f;
    const std::vector<int32_t> prompt = {1, 2, 3, 4};

    auto factory = createOrchestrationRunnerFactory();
    auto baseline = factory->createFromOrchestrationConfig(makeSingleCpuConfig(false));
    ASSERT_NE(baseline, nullptr);
    ASSERT_TRUE(baseline->initialize()) << baseline->lastError();
    auto baseline_result = baseline->generate(prompt, 1, greedy);
    ASSERT_TRUE(baseline_result.error.empty()) << baseline_result.error;
    ASSERT_EQ(baseline_result.tokens.size(), 1u);

    auto cached = factory->createFromOrchestrationConfig(makeSingleCpuConfig(true));
    ASSERT_NE(cached, nullptr);
    ASSERT_TRUE(cached->initialize()) << cached->lastError();

    auto first = cached->generate(prompt, 1, greedy);
    ASSERT_TRUE(first.error.empty()) << first.error;
    ASSERT_EQ(first.tokens.size(), 1u);
    EXPECT_EQ(first.tokens[0], baseline_result.tokens[0]);

    const auto after_first = cached->prefixStateProbe();
    EXPECT_TRUE(after_first.prefix_cache_ready);
    EXPECT_GE(after_first.prefix_cache_inserts, 2u);

    auto second = cached->generate(prompt, 1, greedy);
    ASSERT_TRUE(second.error.empty()) << second.error;
    ASSERT_EQ(second.tokens.size(), 1u);
    EXPECT_EQ(second.tokens[0], baseline_result.tokens[0]);

    const auto after_second = cached->prefixStateProbe();
    EXPECT_TRUE(after_second.prefix_cache_ready);
    EXPECT_GE(after_second.prefix_cache_hits, 2u)
        << "Second full prompt should reuse both cached 2-token dense prefix blocks";
    EXPECT_EQ(after_second.current_position, static_cast<int>(prompt.size()));
}

TEST(Test__KVPrefixMTPStateProbe, MTP_ModelInventoryWhenAvailable)
{
    const std::vector<std::string> model_paths = {
        "/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf",
        "/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf",
    };

    bool saw_model = false;
    for (const auto &path : model_paths)
    {
        if (!std::filesystem::exists(path))
        {
            continue;
        }
        saw_model = true;

        ModelLoader loader;
        loader.setUseMmap(false);
        ASSERT_TRUE(loader.loadModel(path)) << "failed to read GGUF metadata: " << path;

        size_t mtp_metadata = 0;
        uint64_t nextn_predict_layers = 0;
        for (const auto &[key, value] : loader.getModel().metadata)
        {
            if (isMTPInventoryKey(key))
            {
                ++mtp_metadata;
            }

            const std::string lower_key = lowercase(key);
            if (lower_key.find("nextn_predict_layers") != std::string::npos ||
                lower_key.find("mtp_num_hidden_layers") != std::string::npos ||
                lower_key.find("mtp.num_hidden_layers") != std::string::npos)
            {
                nextn_predict_layers = value.asUInt64();
            }
        }

        size_t mtp_tensors = 0;
        for (const auto &name : loader.tensorNames())
        {
            if (isMTPInventoryKey(name))
            {
                ++mtp_tensors;
            }
        }

        EXPECT_GT(mtp_metadata, 0u)
            << "expected MTP/nextn metadata in " << path;
        EXPECT_EQ(nextn_predict_layers, 1u)
            << "expected one MTP/nextn prediction layer in " << path;
        EXPECT_GT(mtp_tensors, 0u)
            << "expected MTP/nextn tensors in " << path;

        auto manifest = discoverMTPWeightManifest(
            loader,
            loader.architecture(),
            static_cast<int>(loader.blockCount()),
            /*explicit_mtp=*/true);
        ASSERT_TRUE(manifest.available) << manifest.diagnostic << " in " << path;
        ASSERT_EQ(manifest.depth, 1);
        ASSERT_EQ(manifest.depths.size(), 1u);
        EXPECT_TRUE(manifest.depths[0].nextn_block_layout);
        EXPECT_EQ(
            manifest.depths[0].source_layer_index,
            static_cast<int>(loader.blockCount() - nextn_predict_layers));
        if (lowercase(loader.architecture()).find("moe") != std::string::npos)
        {
            EXPECT_TRUE(manifest.depths[0].moe_ffn_layout);
        }
    }

    if (!saw_model)
    {
        GTEST_SKIP() << "Qwen3.6 MTP probe models are not available";
    }
}

TEST(Test__KVPrefixMTPStateProbe, Qwen36ROCmMTPRealModelSmokeOptIn)
{
    const char *enabled = std::getenv("LLAMINAR_ENABLE_QWEN36_REAL_MODEL_SMOKE");
    if (!enabled || std::string(enabled) != "1")
    {
        GTEST_SKIP() << "Set LLAMINAR_ENABLE_QWEN36_REAL_MODEL_SMOKE=1 to run real Qwen3.6 ROCm MTP smoke";
    }

    const char *env_model = std::getenv("LLAMINAR_QWEN36_DENSE_MODEL");
    if (!env_model)
        env_model = std::getenv("LLAMINAR_PARITY_DENSE_MODEL");
    const std::string model_path = env_model ? env_model : "/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf";

    if (!std::filesystem::exists(model_path))
    {
        GTEST_SKIP() << "Qwen3.6 dense smoke model not found: " << model_path;
    }

    auto &dm = DeviceManager::instance();
    dm.initialize(-1, false);
    if (dm.rocm_device_count() <= 0)
    {
        GTEST_SKIP() << "No ROCm device available for Qwen3.6 MTP smoke";
    }

    auto factory = createOrchestrationRunnerFactory();
    SamplingParams greedy;
    greedy.temperature = 0.0f;
    const std::string prompt_text = "Paris is";

    auto make_config = [&](bool enable_mtp)
    {
        OrchestrationConfig config = OrchestrationConfig::defaults();
        config.model_path = model_path;
        config.max_seq_len = 32;
        config.batch_size = 1;
        config.tp_degree = 1;
        config.pp_degree = 1;
        config.device_for_this_rank = GlobalDeviceAddress::rocm(0);
        config.kv_cache_precision = "auto";
        config.mtp.enabled = enable_mtp;
        config.mtp.draft_tokens = 1;
        return config;
    };

    auto run_once = [&](bool enable_mtp,
                        GenerationResult *result,
                        PrefixRuntimeStateSnapshot *snapshot)
    {
        auto runner = factory->createFromOrchestrationConfig(make_config(enable_mtp));
        ASSERT_NE(runner, nullptr);
        ASSERT_TRUE(runner->initialize()) << runner->lastError();
        auto tokenizer = runner->tokenizer();
        ASSERT_NE(tokenizer, nullptr);
        const auto encoded = tokenizer->encode(prompt_text, /*add_bos=*/false, /*add_eos=*/false);
        ASSERT_FALSE(encoded.empty());
        const std::vector<int32_t> prompt(encoded.begin(), encoded.end());
        *result = runner->generate(prompt, 4, greedy);
        *snapshot = runner->prefixStateProbe();
        runner->shutdown();
    };

    GenerationResult baseline_result;
    PrefixRuntimeStateSnapshot baseline_snapshot;
    run_once(false, &baseline_result, &baseline_snapshot);

    GenerationResult mtp_result;
    PrefixRuntimeStateSnapshot mtp_snapshot;
    run_once(true, &mtp_result, &mtp_snapshot);

    ASSERT_TRUE(baseline_result.error.empty()) << baseline_result.error;
    ASSERT_TRUE(mtp_result.error.empty()) << mtp_result.error;
    ASSERT_EQ(baseline_result.tokens.size(), 4u);
    ASSERT_EQ(mtp_result.tokens.size(), 4u);
    EXPECT_EQ(mtp_result.tokens, baseline_result.tokens);
    EXPECT_EQ(baseline_snapshot.mtp_draft_steps, 0u);
    EXPECT_GE(mtp_snapshot.mtp_draft_steps, 2u);
    EXPECT_GE(mtp_snapshot.mtp_verifier_runs, 2u);
    EXPECT_GE(mtp_snapshot.mtp_rollbacks, 2u);
}

TEST(Test__KVPrefixMTPStateProbe, MTP_ShiftedCacheCountProbeOnGPU)
{
    const auto device = firstGpuDeviceId();
    if (!device)
    {
        GTEST_SKIP() << "No CUDA or ROCm GPU available for MTP shifted-cache probe";
    }

    TinyQwenForwardFixture fixture(*device);
    auto graph_builder = std::make_shared<QwenStandardGraph>(fixture.config, fixture.mpi);
    DeviceGraphOrchestrator orchestrator(graph_builder, fixture.mpi);

    ASSERT_TRUE(orchestrator.initializeInferenceStateFromArena(
        /*batch_size=*/1,
        fixture.config.max_seq_len,
        *device));
    orchestrator.setWeights(fixture.modelWeights());
    PreparedWeightStore prepared_store;
    ASSERT_NO_THROW(prepareDenseForwardWeights(orchestrator, *graph_builder, prepared_store, *device));

    const std::vector<int> prefix_tokens = {1, 2, 3, 4};
    ASSERT_NE(orchestrator.forward(prefix_tokens.data(), static_cast<int>(prefix_tokens.size()), 1), nullptr);

    const auto after_prefill = orchestrator.prefixStateProbe();
    EXPECT_TRUE(after_prefill.initialized);
    EXPECT_TRUE(after_prefill.primary_device.is_gpu());
    EXPECT_EQ(after_prefill.current_position, static_cast<int>(prefix_tokens.size()));
    EXPECT_EQ(maxLayerCachedTokens(after_prefill), static_cast<int>(prefix_tokens.size()));
    ASSERT_EQ(after_prefill.mtp_kv_caches.size(), 1u);
    EXPECT_EQ(after_prefill.mtp_kv_caches[0].owner, "mtp:0");
    EXPECT_EQ(after_prefill.mtp_kv_caches[0].n_layers, 1);
    EXPECT_EQ(maxCachedTokensIn(after_prefill.mtp_kv_caches), static_cast<int>(prefix_tokens.size()) - 1);
    EXPECT_EQ(after_prefill.totalMTPCachedTokens(), static_cast<int>(prefix_tokens.size()) - 1);

    orchestrator.clear_cache();
    const auto after_clear = orchestrator.prefixStateProbe();
    EXPECT_EQ(maxCachedTokensIn(after_clear.mtp_kv_caches), 0);
    EXPECT_EQ(after_clear.totalMTPCachedTokens(), 0);
}

TEST(Test__KVPrefixMTPStateProbe, MTP_SidecarOneTokenExecutesOnGPU)
{
    const auto device = firstGpuDeviceId();
    if (!device)
    {
        GTEST_SKIP() << "No CUDA or ROCm GPU available for MTP sidecar smoke test";
    }

    TinyMTPSidecarFixture fixture(*device);
    ASSERT_NE(fixture.kv_cache, nullptr);

    PreparedWeightStore prepared_store;
    ASSERT_NO_THROW(fixture.prepareWeights(prepared_store));

    Qwen35Graph graph_builder(fixture.config, fixture.mpi);
    graph_builder.setWeightBindings(fixture.bindings);
    graph_builder.setWeights(toLegacyModelWeights(fixture.bindings));
    graph_builder.setPreparedWeightStore(&prepared_store);

    auto input = fixture.input();
    auto output = fixture.output();
    ComputeGraph graph = graph_builder.buildMTPGraph(
        /*depth_idx=*/0,
        fixture.mtp_depth_bindings,
        input,
        output);
    ASSERT_GT(graph.size(), 0u);
    EXPECT_EQ(graph.terminalNode(), "mtp0_lm_head");

    auto &pool = GPUDeviceContextPool::instance();
    IWorkerGPUContext *ctx = nullptr;
    if (device->is_cuda())
    {
        ctx = &pool.getNvidiaContext(device->cuda_ordinal());
    }
    else if (device->is_rocm())
    {
        ctx = &pool.getAMDContext(device->rocm_ordinal());
    }
    ASSERT_NE(ctx, nullptr);
    auto device_ctx = IDeviceContext::create(*device, 1);
    ASSERT_NE(device_ctx, nullptr);

    WorkspaceSizingHints workspace_hints;
    workspace_hints.max_seq_len = input.seq_len;
    workspace_hints.n_heads = fixture.config.n_heads;
    workspace_hints.head_dim = fixture.config.head_dim;
    workspace_hints.d_model = fixture.config.d_model;
    workspace_hints.batch_size = input.batch_size;
    workspace_hints.vocab_size = fixture.config.vocab_size;
    WorkspaceAllocator workspace_allocator;
    ASSERT_TRUE(workspace_allocator.allocateForGraph(graph, workspace_hints));

    DeviceGraphExecutor executor;
    bool executed = false;
    ctx->submitAndWait([&]
                       { executed = executor.executeFastDecode(graph, device_ctx.get()); });
    ASSERT_TRUE(executed);

    ASSERT_TRUE(fixture.hidden->ensureOnHost());
    const float *hidden = fixture.hidden->fp32_data();
    ASSERT_NE(hidden, nullptr);
    float hidden_abs_sum = 0.0f;
    for (size_t i = 0; i < fixture.hidden->numel(); ++i)
    {
        ASSERT_TRUE(std::isfinite(hidden[i])) << "non-finite MTP hidden value at index " << i;
        hidden_abs_sum += std::abs(hidden[i]);
    }
    EXPECT_GT(hidden_abs_sum, 0.0f);

    ASSERT_TRUE(fixture.logits->ensureOnHost());
    const float *logits = fixture.logits->fp32_data();
    ASSERT_NE(logits, nullptr);
    float abs_sum = 0.0f;
    for (size_t i = 0; i < fixture.logits->numel(); ++i)
    {
        ASSERT_TRUE(std::isfinite(logits[i])) << "non-finite MTP logit at index " << i;
        abs_sum += std::abs(logits[i]);
    }
    EXPECT_GT(abs_sum, 0.0f);
    EXPECT_EQ(fixture.kv_cache->get_cached_tokens(/*layer=*/0, /*seq_idx=*/0), 1);
}

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}
