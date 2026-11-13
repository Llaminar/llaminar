/**
 * @file Qwen4ExpGraphConfigBuilder.cpp
 * @brief Maps authenticated Flash Next geometry into reusable graph contracts.
 *
 * Compute and terminal-head dimensions stay H; only residual transport widens
 * to four branches. Sparse attention retains the existing full KV storage
 * category while its explicit Flash Next contract selects indexed arithmetic.
 * Predictor blocks remain outside the main decoder's layer interval.
 */
#include "Qwen4ExpGraphConfigBuilder.h"
#include "Qwen4ExpConfig.h"
#include "../GraphTypes.h"
#include "../qwen35moe/Qwen35MoEGraphConfigBuilder.h"
#include "../../interfaces/IModelContext.h"
#include <stdexcept>

namespace llaminar2
{
int GraphConfig::residualWidth() const
{
    if (!flash_next)
        return d_model;
    if (d_model != static_cast<int>(flash_next->activations().compute))
        throw std::invalid_argument("Flash Next graph compute width disagrees with its immutable metadata");
    return static_cast<int>(flash_next->activations().trunk);
}

int GraphConfig::terminalHeadWidth() const
{
    if (!flash_next)
        return d_model;
    (void)residualWidth(); // One authentication for all role-specific geometry.
    return static_cast<int>(flash_next->activations().terminal_head);
}

bool Qwen4ExpGraphConfigBuilder::populateFromModelContext(IModelContext &ctx, GraphConfig &config)
{
    const auto loader = ctx.loader();
    if (!loader || !loader->ggufDirectory())
        throw std::invalid_argument("Flash Next requires an exact GGUF directory, including UINT64 n-gram constants");
    config.n_layers = ctx.blockCount();
    populateFromDirectory(*loader->ggufDirectory(), config);
    return true;
}

void Qwen4ExpGraphConfigBuilder::populateFromDirectory(const GGUFModel &model, GraphConfig &config)
{
    auto exact = std::make_shared<const Qwen4ExpConfig>(model);
    const int main_layers = static_cast<int>(exact->mainLayers());
    const int all_layers = main_layers + static_cast<int>(exact->predictorLayers());
    if (config.pp_layer_offset < 0 || config.pp_layer_offset >= main_layers || config.n_layers < 0)
        throw std::invalid_argument("Flash Next graph has an invalid main-decoder PP interval");
    // A complete export may include its trailing predictor in block_count.
    // An explicit PP interval is never silently expanded or truncated.
    if (config.n_layers == 0 || (config.pp_layer_offset == 0 && config.n_layers == all_layers))
        config.n_layers = main_layers - config.pp_layer_offset;
    if (config.n_layers > main_layers - config.pp_layer_offset)
        throw std::invalid_argument("Flash Next PP interval includes a predictor or exceeds the main decoder");
    config.total_n_layers = main_layers;
    config.d_model = exact->activations().compute;
    config.n_heads = exact->sparseAttention().query_heads;
    config.n_kv_heads = exact->sparseAttention().kv_heads;
    config.head_dim = exact->sparseAttention().head_dimension;
    config.d_ff = exact->experts().intermediate;
    config.vocab_size = exact->vocabularySize();
    config.rms_norm_eps = exact->normEpsilon();
    config.rope_theta = exact->rotaryBase();
    config.partial_rotary_factor = static_cast<float>(exact->sparseAttention().rotary_dimension) / config.head_dim;
    config.has_attention_output_gate = true;
    config.kv_cache_scale_k = 1024.0f;
    config.kv_cache_scale_v = 256.0f;
    config.gdn.conv_kernel_size = exact->deltaNet().convolution_kernel;
    config.gdn.state_size = exact->deltaNet().head_dimension;
    config.gdn.inner_size = exact->deltaNet().value_width;
    config.gdn.group_count = exact->deltaNet().key_heads;
    config.gdn.time_step_rank = exact->deltaNet().value_heads;
    config.gdn.full_attention_interval = 4;
    config.layer_types.resize(main_layers);
    config.layer_head_dim.resize(main_layers);
    for (int layer = 0; layer < main_layers; ++layer)
    {
        const bool gdn = exact->mixers()[layer] == Qwen4ExpMixer::GatedDeltaNet;
        config.layer_types[layer] = gdn ? "gdn" : "full_attention";
        config.layer_head_dim[layer] = gdn ? exact->deltaNet().head_dimension : config.head_dim;
    }
    config.moe.num_experts = exact->experts().count;
    config.moe.top_k = exact->experts().selected;
    config.moe.intermediate_size = exact->experts().intermediate;
    config.moe.shared_intermediate_size = exact->experts().shared_intermediate;
    config.moe.has_shared_expert = true;
    config.moe.norm_topk_prob = true;
    config.moe.shared_expert_gate = true;
    config.moe.decode_histogram_token_boundary_layer = config.n_layers - 1;
    config.mtp_verifier_outcome_ownership = MTPVerifierOutcomeOwnershipPolicy::ParticipantLocal;
    config.mtp_request_terminal_hidden_publication = MTPRequestTerminalHiddenPublicationPolicy::GraphCapturedDeviceGeometry;
    config.mtp_shifted_prefill_hidden_publication = MTPShiftedPrefillHiddenPublicationPolicy::GraphIntegratedKVTransaction;
    config.flash_next = std::move(exact);
    // Planning and native graph construction consume this same immutable
    // selection, including replay and persistent-prefix compatibility.
    config.qfn_hc_up_activation = debugEnv().gemm.qfn_hc_up_activation;
    config.qfn_gdn_output_activation = debugEnv().gemm.qfn_gdn_output_activation;
    config.qfn_fa_output_activation = debugEnv().gemm.qfn_fa_output_activation;
    config.qfn_gdn_qkv_activation = debugEnv().gemm.qfn_gdn_qkv_activation;
}

ModelWeights Qwen4ExpGraphConfigBuilder::buildWeights(WeightAccessor get_weight)
{
    // GDN and routed/shared expert tensors retain their existing native roles.
    // The extension supplies the different residual wrappers and indexer.
    Qwen35MoEGraphConfigBuilder common;
    ModelWeights weights = common.buildWeights(get_weight);
    if (!get_weight("token_embd.weight") || !get_weight("output.weight"))
        throw std::invalid_argument("Flash Next requires distinct token embedding and vocabulary head weights");
    weights.final_norm = nullptr;
    weights.hyper_connection_head = loadHyperConnectionWeights(
        [get_weight](const std::string &name) { return get_weight(name).get(); }, "output_hc_");
    auto base = weights.get_layer_weights;
    weights.get_layer_weights = [get_weight, base](int layer_index)
    {
        auto layer = base(layer_index);
        const std::string prefix = "blk." + std::to_string(layer_index) + ".";
        layer.flash_next = loadQwen4ExpLayerWeights(
            [get_weight, &prefix](const std::string &name) { return get_weight(prefix + name).get(); });
        return layer;
    };
    return weights;
}
} // namespace llaminar2
