/**
 * @file Qwen4ExpConfig.cpp
 * @brief Fail-closed parsing of the published qwen4exp GGUF text-model contract.
 *
 * Metadata names and dimensions follow llama.cpp converter revision
 * 1b70350c92563449576bcbbf552994b1d1ed847d and Qwen checkpoint revision
 * de4b8e4d43b917e7706784d8bb445c9af86a3540. Array subtypes are decoded exactly;
 * no model payload, hash mirror, prepared weight, or independent storage ledger
 * is created here. GGUF norm weights are the converter's final gains.
 */
#include "Qwen4ExpConfig.h"
#include "Qwen4ExpPublishedGeometry.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace llaminar2
{
    namespace
    {
        /** @brief Reject an unauthenticated model contract with a precise source key. */
        [[noreturn]] void reject(const std::string &key, const std::string &reason)
        {
            throw std::invalid_argument("qwen4exp: " + key + ": " + reason);
        }

        /** @brief Resolve a required raw metadata value without synthesizing defaults. */
        const GGUFValue &required(const GGUFModel &model, const std::string &key)
        {
            const auto found = model.metadata.find(key);
            if (found == model.metadata.end())
                reject(key, "required metadata is missing");
            return found->second;
        }

        /** @brief Read an unsigned scalar with exact subtype, size and range validation. */
        uint32_t unsignedValue(const GGUFModel &model, const std::string &key)
        {
            uint64_t value;
            try { value = required(model, key).asUInt64Checked(); }
            catch (const std::invalid_argument &error) { reject(key, error.what()); }
            if (value > std::numeric_limits<uint32_t>::max())
                reject(key, "value exceeds uint32 geometry");
            return static_cast<uint32_t>(value);
        }

        /** @brief Authenticate an explicitly declared published-checkpoint dimension. */
        uint32_t dimension(const GGUFModel &model, const std::string &suffix, uint32_t expected)
        {
            const std::string key = "qwen4exp." + suffix;
            const auto value = unsignedValue(model, key);
            if (value != expected)
                reject(key, "published Flash Next geometry requires " + std::to_string(expected) +
                            ", got " + std::to_string(value));
            return value;
        }

        /** @brief Decode a required exact-width metadata array with contextual diagnostics. */
        std::vector<uint64_t> integers(const GGUFModel &model, const std::string &suffix)
        {
            const std::string key = "qwen4exp." + suffix;
            try { return required(model, key).asUInt64Array(); }
            catch (const std::invalid_argument &error) { reject(key, error.what()); }
        }

        /** @brief Require a finite positive exported floating scalar rather than a loader default. */
        float positiveFloat(const GGUFModel &model, const std::string &suffix)
        {
            const std::string key = "qwen4exp." + suffix;
            const auto &raw = required(model, key);
            if (raw.type != GGUFValueType::FLOAT32 || raw.data.size() != sizeof(float))
                reject(key, "requires an exact FLOAT32 scalar");
            const float value = raw.asFloat32();
            if (!std::isfinite(value) || value <= 0)
                reject(key, "requires a finite positive value");
            return value;
        }

        /** @brief Resolve a source tensor and authenticate its canonical loader shape. */
        const GGUFTensorInfo &tensor(
            const GGUFModel &model, const std::string &name, const std::vector<uint64_t> &shape)
        {
            const auto *source = model.findTensor(name);
            if (!source)
                reject(name, "required tensor directory entry is missing");
            if (source->dimensions != shape)
                reject(name, "tensor shape does not match declared model geometry");
            return *source;
        }
    }

    Qwen4ExpConfig::Qwen4ExpConfig(const GGUFModel &model)
    {
        namespace published = qwen4exp::published;
        if (model.architecture != "qwen4exp")
            reject("general.architecture", "requires qwen4exp, got " + model.architecture);

        activations_.compute = dimension(model, "embedding_length", published::ComputeWidth);
        hc_.branches = dimension(model, "hyper_connection.count", published::Branches);
        hc_.gate_rank = dimension(model, "hyper_connection.low_rank", published::GateRank);
        // Only HC branches widen the residual. Attention, experts and LM head
        // retain compute width even at the predictor and pipeline boundaries.
        activations_.trunk = activations_.compute * hc_.branches;
        activations_.terminal_head = activations_.compute;
        context_length_ = dimension(model, "context_length", 262144);
        const uint32_t total_layers = unsignedValue(model, "qwen4exp.block_count");
        predictor_layers_ = model.hasMetadata("qwen4exp.nextn_predict_layers")
            ? unsignedValue(model, "qwen4exp.nextn_predict_layers") : 0;
        if (predictor_layers_ > 1 || total_layers != published::MainLayers + predictor_layers_)
            reject("qwen4exp.block_count", "requires 48 main blocks and zero or one exported nextn block");
        main_layers_ = total_layers - predictor_layers_;
        const uint32_t attention_interval = dimension(model, "full_attention_interval", published::AttentionInterval);

        gdn_ = {
            .key_heads = dimension(model, "ssm.group_count", 16),
            .value_heads = dimension(model, "ssm.time_step_rank", 48),
            .head_dimension = dimension(model, "ssm.state_size", 128),
            .value_width = dimension(model, "ssm.inner_size", 6144),
            .convolution_kernel = dimension(model, "ssm.conv_kernel", 4)};
        qsa_.query_heads = dimension(model, "attention.head_count", published::QueryHeads);
        qsa_.kv_heads = dimension(model, "attention.head_count_kv", published::KVHeads);
        qsa_.head_dimension = dimension(model, "attention.key_length", published::HeadDimension);
        dimension(model, "attention.value_length", qsa_.head_dimension);
        qsa_.rotary_dimension = dimension(model, "rope.dimension_count", 64);
        qsa_.indexer_query_heads = dimension(model, "attention.indexer.head_count", published::IndexerQueryHeads);
        qsa_.indexer_dimension = dimension(model, "attention.indexer.key_length", published::IndexerDimension);
        // The converter splits indexer Q/K: K has exactly one head, proved by
        // the actual projection shape below (there is no GGUF KV-head key).
        qsa_.indexer_key_heads = published::IndexerKeyHeads;
        qsa_.selected_token_budget = dimension(model, "attention.indexer.top_k", published::SelectedTokenBudget);
        const auto ratios = integers(model, "attention.compress_ratios");
        if (ratios.size() != total_layers)
            reject("qwen4exp.attention.compress_ratios", "requires one entry per main/predictor block");
        qsa_.compression_ratio = published::CompressionRatio;
        qsa_.selected_complete_blocks = qsa_.selected_token_budget / qsa_.compression_ratio;
        qsa_.maximum_selected_tokens = qsa_.selected_token_budget + qsa_.compression_ratio - 1;
        const auto recurrent = model.metadata.find("qwen4exp.attention.recurrent_layers");
        if (recurrent != model.metadata.end())
        {
            const auto &raw = recurrent->second;
            if (raw.type != GGUFValueType::ARRAY || raw.array_element_type != GGUFValueType::BOOL ||
                raw.array_length != total_layers || raw.data.size() != total_layers)
                reject(recurrent->first, "requires one BOOL per main/predictor block");
        }
        for (uint32_t layer = 0; layer < total_layers; ++layer)
        {
            const bool sparse = layer >= main_layers_ || (layer + 1) % attention_interval == 0;
            if (ratios[layer] != (sparse ? qsa_.compression_ratio : 0u))
                reject("qwen4exp.attention.compress_ratios", "invalid compression at block " + std::to_string(layer));
            if (recurrent != model.metadata.end() && recurrent->second.data[layer] != (sparse ? 0u : 1u))
                reject(recurrent->first, "mixer identity disagrees at block " + std::to_string(layer));
            mixers_.push_back(sparse ? Qwen4ExpMixer::SparseAttention : Qwen4ExpMixer::GatedDeltaNet);
            if (sparse)
            {
                const std::string prefix = "blk." + std::to_string(layer) + ".indexer.";
                tensor(model, prefix + "q_proj.weight", {qsa_.indexer_query_heads * qsa_.indexer_dimension, activations_.compute});
                tensor(model, prefix + "k_proj.weight", {qsa_.indexer_dimension, activations_.compute});
            }
        }
        experts_ = {
            .count = dimension(model, "expert_count", 512),
            .selected = dimension(model, "expert_used_count", 10),
            .intermediate = dimension(model, "expert_feed_forward_length", 640),
            .shared_intermediate = dimension(model, "expert_shared_feed_forward_length", 640)};

        const auto layers = integers(model, "ple.layers");
        if (layers != std::vector<uint64_t>{published::PLELayer})
            reject("qwen4exp.ple.layers", "requires zero-based decoder layer 1");
        ple_.decoder_layer = static_cast<uint32_t>(layers.front());
        ple_.maximum_order = dimension(model, "ple.ngram_size", published::NgramOrder);
        ple_.heads_per_order = dimension(model, "ple.heads_per_ngram", published::HeadsPerOrder);
        ple_.heads = (ple_.maximum_order - 1) * ple_.heads_per_order;
        ple_.row_elements = dimension(model, "embedding_length_per_layer_input", published::RowElements);
        ple_.convolution_kernel = dimension(model, "ple.conv_kernel", published::PLEKernel);
        // Qwen4 PLE dilates by maximum ngram order; this is operator semantics,
        // not a configurable GGUF field or GDN's kernel-minus-one history.
        ple_.convolution_dilation = ple_.maximum_order;
        ple_.convolution_history = (ple_.convolution_kernel - 1) * ple_.convolution_dilation;
        ple_.hash_reset_token = dimension(model, "ple.eos_token_id", generationStopTokens[0]);
        ple_.multipliers = integers(model, "ple.layer_multipliers");
        ple_.head_offsets = integers(model, "ple.head_offsets");
        ple_.head_vocab_sizes = integers(model, "ple.head_vocab_sizes");
        if (ple_.multipliers.size() != ple_.maximum_order ||
            ple_.head_offsets.size() != ple_.heads || ple_.head_vocab_sizes.size() != ple_.heads)
            reject("qwen4exp.ple", "hash multiplier/head arrays have incorrect lengths");
        for (auto multiplier : ple_.multipliers)
            if (multiplier == 0)
                reject("qwen4exp.ple.layer_multipliers", "zero multiplier destroys suffix identity");

        const auto *table = model.findTensor("per_layer_token_embd.weight");
        if (!table || table->dimensions.size() != 2 || table->dimensions[1] != ple_.row_elements ||
            table->dimensions[0] == 0)
            reject("per_layer_token_embd.weight", "requires canonical [rows,160] table directory entry");
        uint64_t previous_end = 0;
        for (uint32_t head = 0; head < ple_.heads; ++head)
        {
            const uint64_t offset = ple_.head_offsets[head];
            const uint64_t count = ple_.head_vocab_sizes[head];
            if (count == 0 || offset < previous_end || offset > table->dimensions[0] ||
                count > table->dimensions[0] - offset)
                reject("qwen4exp.ple.head_offsets", "overlapping or out-of-table head range " + std::to_string(head));
            previous_end = offset + count;
        }
        // Use the source tensor's true format; a Q4_K filename says nothing
        // about the 160-element rows (the published exports use IQ4_NL).
        (void)ggufToTensorType(table->type);
        const size_t block = table->isQuantized() ? table->getBlockSize() : 1;
        const size_t bytes = table->getTypeSize();
        if (block == 0 || bytes == 0 || ple_.row_elements % block != 0)
            reject(table->name, "source format cannot represent independently addressable 160-element rows");
        ple_.row_bytes = ple_.row_elements / block * bytes;
        if (table->dimensions[0] > std::numeric_limits<uint64_t>::max() / ple_.row_bytes ||
            table->size_bytes != table->dimensions[0] * ple_.row_bytes)
            reject(table->name, "source extent disagrees with native row block geometry");
        ple_.table = *table;

        vocabulary_size_ = 248320;
        tensor(model, "token_embd.weight", {vocabulary_size_, activations_.compute});
        tensor(model, "output.weight", {vocabulary_size_, activations_.terminal_head});
        if (model.hasMetadata("qwen4exp.vocab_size"))
            dimension(model, "vocab_size", vocabulary_size_);
        norm_epsilon_ = positiveFloat(model, "attention.layer_norm_rms_epsilon");
        rotary_base_ = positiveFloat(model, "rope.freq_base");
        const auto sections = integers(model, "rope.dimension_sections");
        if (sections != std::vector<uint64_t>{11, 11, 10} && sections != std::vector<uint64_t>{11, 11, 10, 0})
            reject("qwen4exp.rope.dimension_sections", "requires interleaved sections [11,11,10] with optional zero terminator");
        rotary_sections_ = {11, 11, 10};
    }
}
