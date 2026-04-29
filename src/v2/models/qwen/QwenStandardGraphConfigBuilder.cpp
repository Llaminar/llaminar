/**
 * @file QwenStandardGraphConfigBuilder.cpp
 * @brief Implementation of QwenStandardGraphConfigBuilder
 * @author David Sanftenberg
 * @date January 2026
 */

#include "QwenStandardGraphConfigBuilder.h"
#include "QwenStandardGraph.h" // For GraphConfig definition (via GraphTypes.h)
#include "../../interfaces/IModelContext.h"
#include "../../loaders/WeightManager.h"
#include "../qwen35/Qwen35GraphConfigBuilder.h"
#include "../qwen35moe/Qwen35MoEGraphConfigBuilder.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace llaminar2
{

    // =========================================================================
    // Factory Functions
    // =========================================================================

    std::unique_ptr<IGraphConfigBuilder> createQwenStandardGraphConfigBuilder()
    {
        return std::make_unique<QwenStandardGraphConfigBuilder>();
    }

    std::unique_ptr<IGraphConfigBuilder> createGraphConfigBuilder(const std::string &model_type)
    {
        // Currently only Qwen2 is supported
        if (model_type == "qwen2" || model_type == "Qwen2" ||
            model_type == "qwen3" || model_type == "Qwen3")
        {
            return createQwenStandardGraphConfigBuilder();
        }
        if (model_type == "qwen35" || model_type == "Qwen35")
        {
            return std::make_unique<Qwen35GraphConfigBuilder>();
        }
        if (model_type == "qwen35moe" || model_type == "Qwen35Moe")
        {
            return std::make_unique<Qwen35MoEGraphConfigBuilder>();
        }
        return nullptr;
    }

    // =========================================================================
    // QwenStandardGraphConfigBuilder Implementation
    // =========================================================================

    GraphConfigBuildResult QwenStandardGraphConfigBuilder::buildConfig(
        const RankExecutionPlan &plan,
        const ModelConfig &model_config,
        IWeightManager & /*weight_manager*/)
    {
        GraphConfigBuildResult result;

        // Create placement strategy
        auto placement = createPlacement(plan, model_config);
        if (!placement)
        {
            result.success = false;
            result.error = "Failed to create layer device placement";
            return result;
        }

        // Extract model info (mapping ModelConfig field names to GraphConfigBuildResult names)
        result.model_info.n_layers = model_config.n_layers;
        result.model_info.n_heads = model_config.n_heads;
        result.model_info.n_kv_heads = model_config.n_kv_heads;
        result.model_info.d_model = model_config.hidden_size;    // ModelConfig uses hidden_size
        result.model_info.d_ff = model_config.intermediate_size; // ModelConfig uses intermediate_size
        result.model_info.vocab_size = model_config.vocab_size;
        result.model_info.head_dim = model_config.head_dim;

        // Extract execution info from plan
        result.execution_info.first_layer = plan.first_layer;
        result.execution_info.last_layer = plan.last_layer;
        result.execution_info.has_embedding = plan.has_embedding;
        result.execution_info.has_lm_head = plan.has_lm_head;
        result.execution_info.shard_index = plan.weight_shard.shard_index;
        result.execution_info.total_shards = plan.weight_shard.total_shards;

        // Calculate local heads using placement
        result.execution_info.local_heads =
            placement->headsForDevice(plan.primary_device);
        result.execution_info.local_kv_heads =
            placement->kvHeadsForDevice(plan.primary_device);

        result.placement = std::move(placement);
        result.success = true;
        return result;
    }

    bool QwenStandardGraphConfigBuilder::buildGraphConfig(
        const RankExecutionPlan &plan,
        const ModelConfig &model_config,
        IWeightManager & /*weight_manager*/,
        GraphConfig &config)
    {
        // Create placement strategy
        auto placement = createPlacement(plan, model_config);
        if (!placement)
        {
            return false;
        }

        // Basic model architecture
        // Note: ModelConfig uses hidden_size/intermediate_size,
        // GraphConfig uses d_model/d_ff
        config.n_layers = model_config.n_layers;
        config.total_n_layers = model_config.n_layers; // In MPI path, n_layers is always total
        config.d_model = model_config.hidden_size;
        config.n_heads = model_config.n_heads;
        config.n_kv_heads = model_config.n_kv_heads;
        config.head_dim = model_config.head_dim;
        config.d_ff = model_config.intermediate_size;
        config.vocab_size = model_config.vocab_size;

        // Precision settings: use defaults from GraphConfig
        // (ModelConfig doesn't carry rms_norm_eps/rope_theta)
        // config.rms_norm_eps = 1e-6f;  // default
        // config.rope_theta = 10000.0f; // default

        // Device settings
        config.default_device = plan.primary_device.toLocalDeviceId();

        // Enable rope_on_read: fuses RoPE into KV cache read path
        // (get_kv_converted applies RoPE during incremental dequant)
        // Supported on CPU (all precisions) and GPU (via get_kv_converted override)
        {
            const char *ror_env = std::getenv("LLAMINAR_ROPE_ON_READ");
            config.rope_on_read = ror_env ? (std::atoi(ror_env) != 0) : true;
        }

        // Configure TP for attention
        configureAttentionTP(config, plan, model_config, *placement);

        // Configure TP for FFN
        configureFFNTP(config, plan, model_config);

        // Configure TP for LM head
        configureLMHeadTP(config, plan, model_config);

        // Configure pipeline parallelism
        configurePipelineParallel(config, plan, model_config);

        return true;
    }

    std::unique_ptr<LayerDevicePlacement> QwenStandardGraphConfigBuilder::createPlacement(
        const RankExecutionPlan &plan,
        const ModelConfig &model_config)
    {
        return LayerDevicePlacement::fromExecutionPlan(
            plan,
            model_config.n_layers,
            model_config.n_heads,
            model_config.n_kv_heads);
    }

    void QwenStandardGraphConfigBuilder::configureAttentionTP(
        GraphConfig &config,
        const RankExecutionPlan &plan,
        const ModelConfig &model_config,
        const LayerDevicePlacement &placement)
    {
        // Get shard info
        int shard_index = plan.weight_shard.shard_index;
        int total_shards = plan.weight_shard.total_shards;

        if (total_shards <= 1)
        {
            // No TP - use full heads
            config.local_n_heads = model_config.n_heads;
            config.local_n_kv_heads = model_config.n_kv_heads;
            config.head_start = 0;
            config.qkv_column_parallel = false;
            return;
        }

        // Get weights for proportional distribution
        std::vector<float> weights;
        if (plan.usesLocalTP())
        {
            weights = plan.local_tp_weights;
        }

        // Calculate local heads from placement
        config.local_n_heads = placement.headsForDevice(plan.primary_device);
        config.local_n_kv_heads = placement.kvHeadsForDevice(plan.primary_device);
        config.head_start = computeHeadStart(
            model_config.n_heads, shard_index, total_shards, weights);
        config.qkv_column_parallel = true;
    }

    void QwenStandardGraphConfigBuilder::configureFFNTP(
        GraphConfig &config,
        const RankExecutionPlan &plan,
        const ModelConfig &model_config)
    {
        int total_shards = plan.weight_shard.total_shards;

        if (total_shards <= 1)
        {
            // No TP
            config.d_ff_local = model_config.intermediate_size;
            config.ffn_column_parallel = false;
            return;
        }

        // Proportional FFN distribution
        if (plan.usesLocalTP() && !plan.local_tp_weights.empty())
        {
            float my_weight = plan.local_tp_weights[plan.weight_shard.shard_index];
            config.d_ff_local = static_cast<int>(
                std::round(model_config.intermediate_size * my_weight));
        }
        else
        {
            // Equal distribution
            config.d_ff_local = model_config.intermediate_size / total_shards;
        }

        // Round to alignment (typically 128 for SIMD)
        constexpr int FFN_ALIGNMENT = 128;
        config.d_ff_local = ((config.d_ff_local + FFN_ALIGNMENT - 1) / FFN_ALIGNMENT) * FFN_ALIGNMENT;

        config.ffn_column_parallel = true;
    }

    void QwenStandardGraphConfigBuilder::configureLMHeadTP(
        GraphConfig &config,
        const RankExecutionPlan &plan,
        const ModelConfig &model_config)
    {
        int total_shards = plan.weight_shard.total_shards;

        if (total_shards <= 1)
        {
            // No TP
            config.vocab_local = model_config.vocab_size;
            config.lm_head_column_parallel = false;
            return;
        }

        // Proportional vocab distribution
        if (plan.usesLocalTP() && !plan.local_tp_weights.empty())
        {
            float my_weight = plan.local_tp_weights[plan.weight_shard.shard_index];
            config.vocab_local = static_cast<int>(
                std::round(model_config.vocab_size * my_weight));
        }
        else
        {
            // Equal distribution
            config.vocab_local = model_config.vocab_size / total_shards;
        }

        config.lm_head_column_parallel = true;
    }

    void QwenStandardGraphConfigBuilder::configurePipelineParallel(
        GraphConfig & /*config*/,
        const RankExecutionPlan & /*plan*/,
        const ModelConfig & /*model_config*/)
    {
        // Pipeline parallelism doesn't directly modify GraphConfig fields.
        // The layer building decisions (which layers, embedding, lm_head) are
        // determined by LayerDevicePlacement::shouldBuildLayer(),
        // shouldBuildEmbedding(), and shouldBuildLMHead().
        //
        // The RankExecutionPlan.has_embedding and has_lm_head flags are used
        // when creating the LayerDevicePlacement, not when configuring the
        // graph config.
        //
        // If future PP features need config changes (e.g., async pipeline
        // execution settings), they would be configured here.
    }

    int QwenStandardGraphConfigBuilder::computeHeadStart(
        int total_heads,
        int shard_index,
        int total_shards,
        const std::vector<float> &weights)
    {
        if (total_shards <= 1 || shard_index == 0)
        {
            return 0;
        }

        if (!weights.empty() && static_cast<int>(weights.size()) >= total_shards)
        {
            // Proportional distribution
            int start = 0;
            for (int i = 0; i < shard_index; ++i)
            {
                start += computeLocalHeads(total_heads, i, total_shards, weights);
            }
            return start;
        }

        // Equal distribution
        int heads_per_shard = total_heads / total_shards;
        return shard_index * heads_per_shard;
    }

    int QwenStandardGraphConfigBuilder::computeLocalHeads(
        int total_heads,
        int shard_index,
        int total_shards,
        const std::vector<float> &weights)
    {
        if (total_shards <= 1)
        {
            return total_heads;
        }

        if (!weights.empty() && static_cast<int>(weights.size()) >= total_shards)
        {
            // Proportional distribution based on weights
            float total_weight = std::accumulate(weights.begin(),
                                                 weights.begin() + total_shards, 0.0f);

            if (shard_index == total_shards - 1)
            {
                // Last shard gets remainder
                int used = 0;
                for (int i = 0; i < total_shards - 1; ++i)
                {
                    used += static_cast<int>(
                        std::round(total_heads * weights[i] / total_weight));
                }
                return total_heads - used;
            }

            return static_cast<int>(
                std::round(total_heads * weights[shard_index] / total_weight));
        }

        // Equal distribution
        if (shard_index == total_shards - 1)
        {
            // Last shard gets remainder
            int used = (total_heads / total_shards) * (total_shards - 1);
            return total_heads - used;
        }

        return total_heads / total_shards;
    }

    // =========================================================================
    // IModelContext-Based Configuration (Polymorphic API)
    // =========================================================================

    bool QwenStandardGraphConfigBuilder::populateFromModelContext(
        IModelContext &ctx,
        GraphConfig &config)
    {
        config.n_layers = ctx.blockCount();
        config.total_n_layers = ctx.totalBlockCount();
        config.d_model = ctx.embeddingLength();
        config.n_heads = ctx.headCount();
        config.n_kv_heads = ctx.headCountKV();
        config.vocab_size = ctx.vocabSize();

        // head_dim: prefer explicit key_length, fall back to d_model / n_heads
        config.head_dim = ctx.keyLength() > 0
                              ? ctx.keyLength()
                              : config.d_model / config.n_heads;

        // d_ff: prefer explicit value, fall back to 4x d_model
        int d_ff = ctx.feedForwardLength();
        if (d_ff > 0)
        {
            config.d_ff = d_ff;
        }
        else
        {
            config.d_ff = config.d_model * 4;
            LOG_WARN("[QwenStandardGraphConfigBuilder] feedForwardLength() returned 0, using estimate: "
                     << config.d_ff);
        }

        // Hyperparameters from loader (rope_theta, rms_norm_eps)
        auto loader = ctx.loader();
        if (loader)
        {
            config.rope_theta = loader->ropeTheta();
            config.rms_norm_eps = loader->rmsNormEps();
        }

        // Enable rope_on_read: fuses RoPE into KV cache read path.
        // K is stored pre-RoPE in the cache; get_kv_converted() applies RoPE
        // during read (fused with dequantization for quantized caches).
        // Required for TQ caches (raw TQ blocks can't be read by attention),
        // and saves a kernel launch for all cache types during decode.
        // TODO: Re-enable once the TQ pipeline parity is acceptable
        // config.rope_on_read = true;

        // KV cache scales: separate K and V scales for Q16_1 quantization.
        // K has large post-RoPE outliers in low-frequency dimensions (up to ±300),
        // requiring a wider scale. V values are typically much smaller (±3-5).
        // With block-diagonal rotation enabled, these scales provide
        // headroom while maximizing precision for each tensor.
        //
        // Qwen3+ models (with QK-norm) produce much larger V activations in
        // later layers (absmax ~64 at layer 25/28) and K post-RoPE outliers
        // up to ~417. The Qwen2 defaults (K=512, V=32) cause severe clipping:
        //   V: 13.5% clip rate at layer 25, 8 of 28 layers clip
        //   K: layer 0 clips at 0.3%
        const std::string &arch = ctx.architecture();
        const bool is_qwen3_family = (arch == "qwen3" || arch == "Qwen3");
        if (is_qwen3_family)
        {
            config.kv_cache_scale_k = 1024.0f; // K: ±512 representable (covers absmax ~417)
            config.kv_cache_scale_v = 256.0f;   // V: ±128 representable (covers absmax ~64)
        }
        else
        {
            config.kv_cache_scale_k = 512.0f; // K: ±256 representable after VNNI headroom
            config.kv_cache_scale_v = 32.0f;  // V: ±16 representable, excellent precision
        }

        return true;
    }

    // =========================================================================
    // Weight Building (Polymorphic API)
    // =========================================================================

    ModelWeights QwenStandardGraphConfigBuilder::buildWeights(WeightAccessor get_weight)
    {
        ModelWeights weights;

        auto embedding = get_weight("token_embd.weight");
        auto final_norm = get_weight("output_norm.weight");
        auto lm_head = get_weight("output.weight");

        // Tied embeddings: if output.weight is missing, reuse token_embd.weight
        if (!lm_head && embedding)
        {
            LOG_INFO("[QwenStandardGraphConfigBuilder] output.weight not found, using tied embeddings");
            lm_head = embedding;
        }

        weights.embedding_table = embedding.get();
        weights.final_norm = final_norm.get();
        weights.lm_head = lm_head.get();

        // Layer weight accessor — captures get_weight for deferred per-layer resolution
        weights.get_layer_weights = [get_weight](int layer_idx) -> LayerWeights
        {
            LayerWeights layer;
            std::string prefix = "blk." + std::to_string(layer_idx) + ".";

            // Attention weights
            layer.wq = get_weight(prefix + "attn_q.weight").get();
            layer.wk = get_weight(prefix + "attn_k.weight").get();
            layer.wv = get_weight(prefix + "attn_v.weight").get();
            layer.wo = get_weight(prefix + "attn_output.weight").get();
            layer.attn_norm = get_weight(prefix + "attn_norm.weight").get();

            // Attention biases (Qwen2 has these, Qwen3 does not)
            auto q_bias = get_weight(prefix + "attn_q.bias");
            auto k_bias = get_weight(prefix + "attn_k.bias");
            auto v_bias = get_weight(prefix + "attn_v.bias");
            layer.q_bias = q_bias ? q_bias.get() : nullptr;
            layer.k_bias = k_bias ? k_bias.get() : nullptr;
            layer.v_bias = v_bias ? v_bias.get() : nullptr;

            // QK norm weights (Qwen3: per-head RMSNorm, null for Qwen2)
            auto q_norm = get_weight(prefix + "attn_q_norm.weight");
            auto k_norm = get_weight(prefix + "attn_k_norm.weight");
            layer.q_norm = q_norm ? q_norm.get() : nullptr;
            layer.k_norm = k_norm ? k_norm.get() : nullptr;

            // FFN weights
            layer.gate_proj = get_weight(prefix + "ffn_gate.weight").get();
            layer.up_proj = get_weight(prefix + "ffn_up.weight").get();
            layer.down_proj = get_weight(prefix + "ffn_down.weight").get();
            layer.ffn_norm = get_weight(prefix + "ffn_norm.weight").get();

            return layer;
        };

        return weights;
    }

} // namespace llaminar2
