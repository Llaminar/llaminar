/**
 * @file Qwen4ExpMetadataFixture.h
 * @brief Device-free synthetic GGUF directory with the published Flash Next geometry.
 *
 * The large row table is metadata only: no model payload or mapping is created.
 * Hash constants are deliberately synthetic integers above FP32's exact range,
 * proving preservation without presenting this fixture as checkpoint identity.
 */
#pragma once

#include "models/qwen4exp/Qwen4ExpConfig.h"
#include <cstring>
#include <string>
#include <vector>

namespace llaminar2::test
{
    /** @brief Encode one scalar in the exact GGUF subtype used by a test. */
    template <typename T>
    GGUFValue qwen4Scalar(GGUFValueType type, T value)
    {
        GGUFValue result{.type = type, .data = std::vector<uint8_t>(sizeof(T))};
        std::memcpy(result.data.data(), &value, sizeof(T));
        return result;
    }

    /** @brief Encode integer-array bytes with retained source subtype. */
    template <typename T>
    GGUFValue qwen4Array(GGUFValueType subtype, const std::vector<T> &values)
    {
        GGUFValue result{
            .type = GGUFValueType::ARRAY,
            .data = std::vector<uint8_t>(values.size() * sizeof(T)),
            .array_length = values.size(),
            .array_element_type = subtype};
        if (!values.empty())
            std::memcpy(result.data.data(), values.data(), result.data.size());
        return result;
    }

    /** @brief Build a directory for main-only or main-plus-predictor Flash Next exports. */
    inline GGUFModel qwen4MetadataFixture(bool with_predictor = true)
    {
        GGUFModel model;
        model.architecture = "qwen4exp";
        const std::vector<std::pair<std::string, uint32_t>> scalars{
            {"embedding_length", 2560}, {"hyper_connection.count", 4},
            {"hyper_connection.low_rank", 320}, {"context_length", 262144},
            {"block_count", with_predictor ? 49u : 48u}, {"full_attention_interval", 4},
            {"ssm.group_count", 16}, {"ssm.time_step_rank", 48}, {"ssm.state_size", 128},
            {"ssm.inner_size", 6144}, {"ssm.conv_kernel", 4},
            {"attention.head_count", 24}, {"attention.head_count_kv", 2},
            {"attention.key_length", 256}, {"attention.value_length", 256},
            {"rope.dimension_count", 64}, {"attention.indexer.head_count", 4},
            {"attention.indexer.key_length", 128}, {"attention.indexer.top_k", 2048},
            {"expert_count", 512}, {"expert_used_count", 10},
            {"expert_feed_forward_length", 640}, {"expert_shared_feed_forward_length", 640},
            {"ple.ngram_size", 3}, {"ple.heads_per_ngram", 8},
            {"embedding_length_per_layer_input", 160}, {"ple.conv_kernel", 4},
            {"ple.eos_token_id", 248044}};
        for (const auto &[suffix, value] : scalars)
            model.metadata["qwen4exp." + suffix] = qwen4Scalar(GGUFValueType::UINT32, value);
        if (with_predictor)
            model.metadata["qwen4exp.nextn_predict_layers"] = qwen4Scalar(GGUFValueType::UINT32, uint32_t{1});
        model.metadata["qwen4exp.attention.layer_norm_rms_epsilon"] = qwen4Scalar(GGUFValueType::FLOAT32, 1e-6f);
        model.metadata["qwen4exp.rope.freq_base"] = qwen4Scalar(GGUFValueType::FLOAT32, 10000000.f);
        model.metadata["qwen4exp.rope.dimension_sections"] = qwen4Array<uint32_t>(GGUFValueType::UINT32, {11, 11, 10, 0});
        model.metadata["qwen4exp.ple.layers"] = qwen4Array<uint32_t>(GGUFValueType::UINT32, {1});
        model.metadata["qwen4exp.ple.layer_multipliers"] = qwen4Array<uint64_t>(GGUFValueType::UINT64,
            {(uint64_t{1} << 44) + 1, (uint64_t{1} << 44) + 3, (uint64_t{1} << 44) + 7});
        std::vector<uint64_t> offsets(16), sizes(16, 20000003);
        for (size_t head = 0; head < offsets.size(); ++head)
            offsets[head] = head * sizes[head];
        model.metadata["qwen4exp.ple.head_offsets"] = qwen4Array(GGUFValueType::UINT64, offsets);
        model.metadata["qwen4exp.ple.head_vocab_sizes"] = qwen4Array(GGUFValueType::UINT64, sizes);
        std::vector<uint32_t> ratios(with_predictor ? 49 : 48);
        std::vector<uint8_t> recurrent(ratios.size());
        for (uint32_t layer = 0; layer < ratios.size(); ++layer)
        {
            const bool sparse = layer >= 48 || (layer + 1) % 4 == 0;
            ratios[layer] = sparse ? 4 : 0;
            recurrent[layer] = sparse ? 0 : 1;
            if (sparse)
            {
                const std::string prefix = "blk." + std::to_string(layer) + ".indexer.";
                model.tensors.push_back({.name = prefix + "q_proj.weight", .dimensions = {512, 2560}, .type = GGUFTensorType::F16});
                model.tensors.push_back({.name = prefix + "k_proj.weight", .dimensions = {128, 2560}, .type = GGUFTensorType::F16});
            }
        }
        model.metadata["qwen4exp.attention.compress_ratios"] = qwen4Array(GGUFValueType::UINT32, ratios);
        model.metadata["qwen4exp.attention.recurrent_layers"] = qwen4Array(GGUFValueType::BOOL, recurrent);
        model.tensors.push_back({.name = "per_layer_token_embd.weight", .dimensions = {320001536, 160},
            .type = GGUFTensorType::IQ4_NL, .offset = 0, .size_bytes = 28800138240ULL, .split_idx = 1});
        model.tensors.push_back({.name = "token_embd.weight", .dimensions = {248320, 2560}, .type = GGUFTensorType::F16});
        model.tensors.push_back({.name = "output.weight", .dimensions = {248320, 2560}, .type = GGUFTensorType::Q4_K});
        return model;
    }
}
