/**
 * @file DeepSeekV4MetadataFixture.h
 * @brief Device-free synthetic deepseek4 GGUF directory (metadata and tensor shapes only).
 *
 * Three layers cover every attention kind of the release model (window-only,
 * ratio-4 CSA with the lightning indexer, ratio-128 HCA), one hash-routed
 * layer and small widths that keep each packed record representable
 * (head_dim 128 = 64 FP8 nope + 64 BF16 rope; indexer keys 64 wide). No tensor
 * payload or file mapping exists; the directory only authenticates geometry for
 * DeepSeekV4Config, GraphConfig population and ModelMemoryProfile admission.
 */
#pragma once

#include "loaders/ModelLoader.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace llaminar2::test
{
/** @brief Encode one scalar of an exact GGUF subtype. */
template <typename T> GGUFValue dsv4Scalar(GGUFValueType type, T value)
{
    GGUFValue result{.type = type, .data = std::vector<uint8_t>(sizeof(T))};
    std::memcpy(result.data.data(), &value, sizeof(T));
    return result;
}

/** @brief Encode an array with its retained element subtype. */
template <typename T> GGUFValue dsv4Array(GGUFValueType subtype, const std::vector<T> &values)
{
    GGUFValue result{.type = GGUFValueType::ARRAY, .data = std::vector<uint8_t>(values.size() * sizeof(T)),
                     .array_length = values.size(), .array_element_type = subtype};
    if (!values.empty()) std::memcpy(result.data.data(), values.data(), result.data.size());
    return result;
}

/** @brief Small deepseek4 directory with compress ratios [0, 4, 128]. */
inline GGUFModel deepSeekV4MetadataFixture(uint32_t context = 512)
{
    constexpr uint64_t d = 64, heads = 4, hd = 128, rope = 64, qRank = 32, groups = 2, oRank = 16;
    constexpr uint64_t window = 4, iHeads = 2, iDim = 64, topK = 4, experts = 4, used = 2, ff = 32, vocab = 32;
    const std::vector<uint32_t> ratios{0, 4, 128};
    GGUFModel model;
    model.architecture = "deepseek4";
    model.block_count = ratios.size();
    model.embedding_length = d;
    model.head_count = heads;
    model.head_count_kv = 1;
    model.key_length = hd;
    model.value_length = hd;
    model.vocab_size = vocab;
    model.context_length = context;
    const auto u32 = [&](const std::string &key, uint64_t value) {
        model.metadata["deepseek4." + key] = dsv4Scalar(GGUFValueType::UINT32, static_cast<uint32_t>(value));
    };
    const auto f32 = [&](const std::string &key, float value) {
        model.metadata["deepseek4." + key] = dsv4Scalar(GGUFValueType::FLOAT32, value);
    };
    u32("block_count", ratios.size());
    u32("embedding_length", d);
    u32("attention.head_count", heads);
    u32("attention.head_count_kv", 1);
    u32("attention.key_length", hd);
    u32("attention.value_length", hd);
    u32("rope.dimension_count", rope);
    u32("attention.q_lora_rank", qRank);
    u32("attention.output_group_count", groups);
    u32("attention.output_lora_rank", oRank);
    u32("attention.sliding_window", window);
    u32("expert_count", experts);
    u32("expert_used_count", used);
    u32("expert_feed_forward_length", ff);
    u32("expert_shared_count", 1);
    u32("hash_layer_count", 1);
    f32("expert_weights_scale", 1.5f);
    model.metadata["deepseek4.expert_weights_norm"] = dsv4Scalar(GGUFValueType::BOOL, uint8_t{1});
    u32("expert_gating_func", 4);
    u32("attention.indexer.head_count", iHeads);
    u32("attention.indexer.key_length", iDim);
    u32("attention.indexer.top_k", topK);
    u32("hyper_connection.count", 4);
    u32("hyper_connection.sinkhorn_iterations", 20);
    f32("hyper_connection.epsilon", 1e-6f);
    f32("attention.layer_norm_rms_epsilon", 1e-6f);
    u32("context_length", context);
    f32("rope.freq_base", 10000.0f);
    f32("attention.compress_rope_freq_base", 160000.0f);
    model.metadata["deepseek4.attention.compress_ratios"] = dsv4Array(GGUFValueType::UINT32, ratios);
    model.metadata["deepseek4.swiglu_clamp_exp"] =
        dsv4Array(GGUFValueType::FLOAT32, std::vector<float>(ratios.size(), 10.0f));

    const auto tensor = [&](const std::string &name, std::vector<uint64_t> shape) {
        uint64_t elements = 1;
        for (const auto dimension : shape) elements *= dimension;
        model.tensors.push_back({.name = name, .dimensions = std::move(shape), .type = GGUFTensorType::F32,
                                 .offset = 0, .size_bytes = elements * 4, .split_idx = 0});
    };
    tensor("token_embd.weight", {vocab, d});
    tensor("output.weight", {vocab, d});
    tensor("output_norm.weight", {d});
    for (std::size_t il = 0; il < ratios.size(); ++il)
    {
        const std::string p = "blk." + std::to_string(il) + ".";
        tensor(p + "hc_attn_fn.weight", {24, d * 4});
        tensor(p + "hc_ffn_fn.weight", {24, d * 4});
        tensor(p + "attn_q_a.weight", {qRank, d});
        tensor(p + "attn_q_b.weight", {hd * heads, qRank});
        tensor(p + "attn_kv.weight", {hd, d});
        tensor(p + "attn_output_a.weight", {groups * oRank, hd * heads / groups});
        tensor(p + "attn_output_b.weight", {d, groups * oRank});
        tensor(p + "attn_sinks.weight", {heads});
        tensor(p + "ffn_gate_inp.weight", {experts, d});
        // Routed experts use the GGUF [K, N, experts] order.
        tensor(p + "ffn_gate_exps.weight", {d, ff, experts});
        tensor(p + "ffn_up_exps.weight", {d, ff, experts});
        tensor(p + "ffn_down_exps.weight", {ff, d, experts});
        tensor(p + "ffn_gate_shexp.weight", {ff, d});
        tensor(p + "ffn_up_shexp.weight", {ff, d});
        tensor(p + "ffn_down_shexp.weight", {d, ff});
        if (ratios[il] != 0)
        {
            const uint64_t coff = ratios[il] == 4 ? 2 : 1;
            tensor(p + "attn_compressor_kv.weight", {coff * hd, d});
            tensor(p + "attn_compressor_gate.weight", {coff * hd, d});
            tensor(p + "attn_compressor_ape.weight", {ratios[il], coff * hd});
        }
        if (ratios[il] == 4)
        {
            tensor(p + "indexer.attn_q_b.weight", {iHeads * iDim, qRank});
            tensor(p + "indexer.proj.weight", {iHeads, d});
            tensor(p + "indexer_compressor_kv.weight", {2 * iDim, d});
        }
        if (il < 1) tensor(p + "ffn_gate_tid2eid.weight", {vocab, used});
        else tensor(p + "exp_probs_b.bias", {experts});
    }
    return model;
}
} // namespace llaminar2::test
