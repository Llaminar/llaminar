/**
 * @file DeepSeekV4GraphConfigBuilder.cpp
 * @brief Populate GraphConfig and weight views from an authenticated deepseek4 GGUF.
 */
#include "DeepSeekV4GraphConfigBuilder.h"
#include "DeepSeekV4Config.h"
#include "../GraphTypes.h"
#include "../../interfaces/IModelContext.h"
#include "../../loaders/IModelLoader.h"
#include "../../loaders/ModelLoader.h"

#include <stdexcept>

namespace llaminar2
{
bool DeepSeekV4GraphConfigBuilder::populateFromModelContext(IModelContext &ctx, GraphConfig &config)
{
    const auto loader = ctx.loader();
    if (!loader || !loader->ggufDirectory())
        throw std::invalid_argument("DeepSeek-V4 requires an exact GGUF directory");
    config.n_layers = ctx.blockCount();
    populateFromDirectory(*loader->ggufDirectory(), config);
    return true;
}

void DeepSeekV4GraphConfigBuilder::populateFromDirectory(const GGUFModel &model, GraphConfig &config)
{
    auto exact = std::make_shared<const DeepSeekV4Config>(model);
    const int layers = exact->layers();
    if (config.pp_layer_offset != 0)
        throw std::invalid_argument("DeepSeek-V4 pipeline parallelism is not implemented yet");
    config.n_layers = layers;
    config.total_n_layers = layers;
    config.d_model = exact->hidden();
    config.n_heads = exact->heads();
    config.n_kv_heads = 1;
    config.head_dim = exact->headDim();
    config.d_ff = exact->expertIntermediate();
    config.vocab_size = exact->vocab();
    config.rms_norm_eps = exact->normEps();
    config.rope_theta = exact->layer(0).rope.base;
    config.partial_rotary_factor = static_cast<float>(exact->ropeDim()) / static_cast<float>(exact->headDim());
    config.layer_types.assign(static_cast<std::size_t>(layers), "full_attention");
    // Attention state lives in the DeepSeek-V4 arena banks; the canonical cache
    // owns only the sequence frontier (model_state layers).
    config.attention_state_owner = attentionStateOwnerOf(model.architecture);
    config.layer_head_dim.assign(static_cast<std::size_t>(layers), exact->headDim());
    config.moe.num_experts = exact->experts();
    config.moe.top_k = exact->expertsUsed();
    config.moe.intermediate_size = exact->expertIntermediate();
    config.moe.shared_intermediate_size = exact->expertIntermediate();
    config.moe.has_shared_expert = true;
    config.moe.norm_topk_prob = true;
    config.moe.shared_expert_gate = false;
    config.moe.decode_histogram_token_boundary_layer = layers - 1;
    config.deepseek_v4 = std::move(exact);
}

ModelWeights DeepSeekV4GraphConfigBuilder::buildWeights(WeightAccessor get_weight)
{
    ModelWeights weights;
    weights.embedding_table = get_weight("token_embd.weight").get();
    weights.final_norm = get_weight("output_norm.weight").get();
    weights.lm_head = get_weight("output.weight").get();
    if (!weights.embedding_table || !weights.final_norm || !weights.lm_head)
        throw std::invalid_argument("DeepSeek-V4 requires token embedding, output norm and vocabulary head");
    weights.deepseek_v4_head = {get_weight("output_hc_fn.weight").get(), get_weight("output_hc_base.weight").get(),
                                get_weight("output_hc_scale.weight").get()};
    weights.get_layer_weights = [get_weight](int layer_index)
    {
        LayerWeights layer;
        const std::string prefix = "blk." + std::to_string(layer_index) + ".";
        const auto get = [&](const std::string &name) { return get_weight(prefix + name).get(); };
        layer.attn_norm = get("attn_norm.weight");
        layer.ffn_norm = get("ffn_norm.weight");
        layer.moe_gate = get("ffn_gate_inp.weight");
        layer.moe_gate_exps = get("ffn_gate_exps.weight");
        layer.moe_up_exps = get("ffn_up_exps.weight");
        layer.moe_down_exps = get("ffn_down_exps.weight");
        layer.shared_expert_gate = get("ffn_gate_shexp.weight");
        layer.shared_expert_up = get("ffn_up_shexp.weight");
        layer.shared_expert_down = get("ffn_down_shexp.weight");
        layer.deepseek_v4 = loadDeepSeekV4LayerWeights(get);
        return layer;
    };
    return weights;
}
} // namespace llaminar2
