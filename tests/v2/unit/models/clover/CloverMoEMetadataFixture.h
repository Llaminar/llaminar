/**
 * @file CloverMoEMetadataFixture.h
 * @brief Payload-free Clover directories for source-admission and graph-contract tests.
 *
 * The Preview case records the official export's omitted shared count and four
 * diffusion-only tensor families. The AR case records the upstream converter's
 * explicit count and causal-only inventory. These are synthetic directories,
 * not numerical model observations or replay traces.
 */
#pragma once
#include "loaders/ModelLoader.h"
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

namespace llaminar2::test
{
    /** @brief The two explicitly supported source-directory conventions. */
    enum class CloverFixtureExport { Preview, Autoregressive };
    /** @brief Small test geometry versus the published 35B tensor dimensions. */
    enum class CloverFixtureGeometry { Small, Published35B };

    /** @brief Encode an exact scalar subtype without creating tensor storage. */
    template<class T> GGUFValue cloverScalar(GGUFValueType type, T value)
    {
        GGUFValue result{.type = type, .data = std::vector<uint8_t>(sizeof(T))};
        std::memcpy(result.data.data(), &value, sizeof(T));
        return result;
    }

    /** @brief Populate one integer metadata key in the synthetic source. */
    inline void cloverInteger(GGUFModel &model, const std::string &suffix, uint32_t value)
    {
        model.metadata["clover_moe." + suffix] = cloverScalar(GGUFValueType::UINT32, value);
    }

    /**
     * @brief Construct complete loader-order metadata without allocating weight payloads.
     * @param source Original Preview or current upstream AR export convention.
     * @param scale Small test dimensions or recorded public checkpoint dimensions.
     * @return Independent synthetic directory with exact tensor byte extents.
     */
    inline GGUFModel cloverMetadataFixture(
        CloverFixtureExport source = CloverFixtureExport::Preview,
        CloverFixtureGeometry scale = CloverFixtureGeometry::Small)
    {
        const bool published = scale == CloverFixtureGeometry::Published35B;
        const uint32_t layers = published ? 40 : 4, h = published ? 2048 : 256;
        const uint32_t vocab = published ? 248320 : 320;
        const uint32_t qheads = published ? 16 : 4, kvheads = 2, head = published ? 256 : 64;
        const uint32_t kheads = published ? 16 : 1, vheads = published ? 32 : 2, state = 128;
        const uint32_t inner = state * vheads, qkv = state * (2 * kheads + vheads);
        const uint32_t experts = published ? 256 : 4, topk = published ? 8 : 2, f = published ? 512 : 256;
        GGUFModel model;
        model.architecture = "clover_moe";
        model.block_count = layers;
        model.embedding_length = h;
        model.vocab_size = vocab;
        model.head_count = qheads;
        model.head_count_kv = kvheads;
        model.key_length = model.value_length = head;
        model.context_length = 131072;
        for (const auto &[name, value] : std::vector<std::pair<std::string, uint32_t>>{
            {"block_count", layers}, {"context_length", 131072}, {"embedding_length", h},
            {"attention.head_count", qheads}, {"attention.head_count_kv", kvheads},
            {"attention.key_length", head}, {"attention.value_length", head},
            {"rope.dimension_count", head / 4}, {"full_attention_interval", 4},
            {"ssm.group_count", kheads}, {"ssm.time_step_rank", vheads}, {"ssm.state_size", state},
            {"ssm.inner_size", inner}, {"ssm.conv_kernel", 4}, {"expert_count", experts},
            {"expert_used_count", topk}, {"expert_feed_forward_length", f},
            {"expert_shared_feed_forward_length", f}})
            cloverInteger(model, name, value);
        if (source == CloverFixtureExport::Autoregressive) cloverInteger(model, "expert_shared_count", 1);
        model.metadata["clover_moe.rope.freq_base"] = cloverScalar(GGUFValueType::FLOAT32, 10000000.0f);
        model.metadata["clover_moe.attention.layer_norm_rms_epsilon"] = cloverScalar(GGUFValueType::FLOAT32, 1e-6f);
        GGUFValue mask{.type = GGUFValueType::ARRAY, .data = std::vector<uint8_t>(layers),
            .array_length = layers, .array_element_type = GGUFValueType::BOOL};
        for (uint32_t i = 0; i < layers; ++i) mask.data[i] = (i + 1) % 4 != 0;
        model.metadata["clover_moe.attention.recurrent_layers"] = mask;
        const auto tensor = [&](const std::string &name, std::vector<uint64_t> shape)
        {
            const uint64_t elements = std::accumulate(shape.begin(), shape.end(), uint64_t{1}, std::multiplies<>());
            model.tensors.push_back({.name = name, .dimensions = std::move(shape),
                .type = GGUFTensorType::F32, .size_bytes = elements * sizeof(float)});
        };
        tensor("token_embd.weight", {vocab, h});
        tensor("output.weight", {vocab, h});
        tensor("output_norm.weight", {h});
        for (uint32_t i = 0; i < layers; ++i)
        {
            const std::string p = "blk." + std::to_string(i) + ".";
            tensor(p + "attn_norm.weight", {h});
            tensor(p + "post_attention_norm.weight", {h});
            tensor(p + "ffn_gate_inp.weight", {experts, h});
            tensor(p + "ffn_gate_exps.weight", {h, f, experts});
            tensor(p + "ffn_up_exps.weight", {h, f, experts});
            tensor(p + "ffn_down_exps.weight", {f, h, experts});
            tensor(p + "ffn_gate_inp_shexp.weight", {h});
            tensor(p + "ffn_gate_shexp.weight", {f, h});
            tensor(p + "ffn_up_shexp.weight", {f, h});
            tensor(p + "ffn_down_shexp.weight", {h, f});
            if ((i + 1) % 4 == 0)
            {
                tensor(p + "attn_q.weight", {2 * qheads * head, h});
                tensor(p + "attn_k.weight", {kvheads * head, h});
                tensor(p + "attn_v.weight", {kvheads * head, h});
                tensor(p + "attn_output.weight", {h, qheads * head});
                tensor(p + "attn_q_norm.weight", {head});
                tensor(p + "attn_k_norm.weight", {head});
            }
            else
            {
                tensor(p + "attn_qkv.weight", {qkv, h});
                tensor(p + "attn_gate.weight", {inner, h});
                tensor(p + "ssm_alpha.weight", {vheads, h});
                tensor(p + "ssm_beta.weight", {vheads, h});
                tensor(p + "ssm_a", {vheads});
                tensor(p + "ssm_dt.bias", {vheads});
                tensor(p + "ssm_conv1d.weight", {qkv, 4});
                tensor(p + "ssm_norm.weight", {state});
                tensor(p + "ssm_out.weight", {h, inner});
                if (source == CloverFixtureExport::Preview)
                {
                    tensor(p + "ssm_conv1d.weight_right", {qkv, 3});
                    tensor(p + "ssm_bd_history.bias", {vheads});
                    tensor(p + "ssm_bd_history.weight", {vheads});
                    tensor(p + "ssm_bd_logit_scale.weight", {vheads});
                }
            }
        }
        model.tensor_count = model.tensors.size();
        return model;
    }
}
