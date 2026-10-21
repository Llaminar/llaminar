/**
 * @file Qwen4ExpSidecarManifest.cpp
 * @brief Authenticate shared native predictor roles without altering either GGUF.
 *
 * Compression zero is the publisher's dense draft policy; compression four is
 * the upstream QSA policy. Exported but unused dense indexer operands remain
 * source metadata and cannot become accidental prepared-weight requests.
 */
#include "Qwen4ExpSidecarManifest.h"
#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>

namespace llaminar2
{
namespace
{
/** @brief Report the exact offending source field before any materialization. */
void require(bool valid, const std::string &field)
{
    if (!valid) throw std::invalid_argument("Qwen4Exp shared MTP source contract: " + field);
}
/** @brief Read one mandatory exact-subtype metadata entry. */
const GGUFValue &metadata(const GGUFModel &model, const std::string &key)
{
    const auto found = model.metadata.find(key);
    require(found != model.metadata.end(), "missing " + key);
    return found->second;
}
/** @brief Compare metadata values without rounding integer arrays or hashing payloads. */
bool sameValue(const GGUFValue &left, const GGUFValue &right)
{
    return left.type == right.type && left.data == right.data && left.array_length == right.array_length &&
        left.array_element_type == right.array_element_type && left.string_array_value == right.string_array_value;
}
/** @brief Predictor token IDs must agree; target-owned prompt rendering is never borrowed from the draft. */
bool tokenizationKey(const std::string &key)
{
    const bool rendering = key == "tokenizer.chat_template" || key == "tokenizer.chat_templates" ||
        key.rfind("tokenizer.chat_template.", 0) == 0;
    return key.rfind("tokenizer.", 0) == 0 && !rendering;
}
/** @brief Reject native extent overflow/truncation while preserving the actual codebook. */
void nativeExtent(const GGUFTensorInfo &tensor)
{
    (void)ggufToTensorType(tensor.type);
    uint64_t elements = 1;
    for (const auto width : tensor.dimensions)
    {
        require(width && elements <= std::numeric_limits<uint64_t>::max() / width, tensor.name + " element extent");
        elements *= width;
    }
    const auto block = tensor.isQuantized() ? tensor.getBlockSize() : size_t(1);
    const auto bytes = tensor.getTypeSize();
    require(block && bytes && elements % block == 0 && elements / block <= std::numeric_limits<uint64_t>::max() / bytes,
        tensor.name + " native block extent");
    require(tensor.size_bytes == elements / block * bytes, tensor.name + " native byte extent");
    require(tensor.offset <= std::numeric_limits<uint64_t>::max() - tensor.size_bytes, tensor.name + " source offset overflow");
}
} // namespace

Qwen4ExpSidecarManifest::Qwen4ExpSidecarManifest(const GGUFModel &main, const GGUFModel &predictor)
{
    const Qwen4ExpConfig target(main);
    require(target.predictorLayers() == 0, "external and inline predictors are mutually exclusive");
    require(predictor.architecture == main.architecture, "architecture mismatch");
    const auto &shared = metadata(predictor, "qwen4exp.nextn_shared_target_tensors");
    require(shared.type == GGUFValueType::BOOL && shared.data.size() == 1 && shared.data[0] == 1,
        "nextn_shared_target_tensors must explicitly select shared target ownership");
    source_layer_ = target.mainLayers();
    activations_ = target.activations();
    require(metadata(predictor, "qwen4exp.block_count").asUInt64Checked() == source_layer_ + 1 &&
        metadata(predictor, "qwen4exp.nextn_predict_layers").asUInt64Checked() == 1,
        "exactly one trailing predictor block is required");

    for (const char *suffix : {"context_length", "embedding_length", "attention.head_count", "attention.head_count_kv",
        "attention.key_length", "attention.value_length", "expert_count", "expert_used_count",
        "expert_feed_forward_length", "expert_shared_feed_forward_length", "ssm.conv_kernel", "ssm.state_size",
        "ssm.group_count", "ssm.time_step_rank", "ssm.inner_size", "full_attention_interval", "rope.dimension_count",
        "hyper_connection.count", "hyper_connection.low_rank", "attention.indexer.head_count",
        "attention.indexer.key_length", "attention.indexer.top_k"})
    {
        const std::string key = std::string("qwen4exp.") + suffix;
        require(metadata(main, key).asUInt64Checked() == metadata(predictor, key).asUInt64Checked(), key + " mismatch");
    }
    for (const char *suffix : {"rope.dimension_sections", "rope.freq_base", "attention.layer_norm_rms_epsilon"})
    {
        const std::string key = std::string("qwen4exp.") + suffix;
        require(sameValue(metadata(main, key), metadata(predictor, key)), key + " mismatch");
    }
    const auto ratios = metadata(predictor, "qwen4exp.attention.compress_ratios").asUInt64Array();
    const auto target_ratios = metadata(main, "qwen4exp.attention.compress_ratios").asUInt64Array();
    require(ratios.size() == source_layer_ + 1 && target_ratios.size() == source_layer_, "compress_ratios length");
    require(std::equal(target_ratios.begin(), target_ratios.end(), ratios.begin()), "main compression entries mismatch");
    require(ratios.back() == 0 || ratios.back() == target.sparseAttention().compression_ratio,
        "predictor compression must explicitly select dense zero or supported sparse geometry");
    attention_ = ratios.back() == 0 ? Qwen4ExpPredictorAttention::Dense : Qwen4ExpPredictorAttention::Sparse;
    if (attention_ == Qwen4ExpPredictorAttention::Sparse) sparse_ = target.sparseAttention();

    const auto &tokens = metadata(main, "tokenizer.ggml.tokens");
    require(tokens.type == GGUFValueType::ARRAY && tokens.array_element_type == GGUFValueType::STRING &&
        tokens.array_length == target.vocabularySize() && tokens.string_array_value.size() == tokens.array_length,
        "target tokenizer vocabulary extent");
    for (const auto &[key, value] : main.metadata)
        if (tokenizationKey(key))
            require(sameValue(value, metadata(predictor, key)), key + " mismatch");
    for (const auto &[key, value] : predictor.metadata)
        if (tokenizationKey(key))
            require(sameValue(value, metadata(main, key)), key + " unexpected predictor policy");

    std::set<std::string> expected;
    const auto bind = [&](const GGUFModel &directory, Qwen4ExpTensorSource owner, const std::string &name,
                          std::vector<uint64_t> shape, bool fp32, bool consumed)
    {
        require(owner != Qwen4ExpTensorSource::Predictor || !main.findTensor(name),
            name + " predictor role collides with target source");
        const auto *info = directory.findTensor(name);
        require(info && info->dimensions == shape, name + " source shape");
        require(!fp32 || info->type == GGUFTensorType::F32, name + " immutable FP32 format");
        nativeExtent(*info);
        require(std::count_if(directory.tensors.begin(), directory.tensors.end(),
            [&](const auto &item) { return item.name == name; }) == 1, name + " duplicate source role");
        if (owner == Qwen4ExpTensorSource::Predictor) expected.insert(name);
        sources_.push_back({owner, *info, consumed});
    };
    for (const char *name : {"token_embd.weight", "output.weight"})
        bind(main, Qwen4ExpTensorSource::Main, name, {target.vocabularySize(), activations_.compute}, false, true);
    const std::string prefix = "blk." + std::to_string(source_layer_) + ".";
    const auto add = [&](const std::string &suffix, std::vector<uint64_t> shape, bool fp32 = false, bool consumed = true)
    { bind(predictor, Qwen4ExpTensorSource::Predictor, prefix + suffix + ".weight", std::move(shape), fp32, consumed); };
    const uint64_t hidden = target.activations().compute, trunk = target.activations().trunk;
    const auto &attention = target.sparseAttention();
    const auto &expert = target.experts();
    const auto &hc = target.hyperConnections();
    const uint64_t query = uint64_t(attention.query_heads) * attention.head_dimension;
    const uint64_t kv = uint64_t(attention.kv_heads) * attention.head_dimension;
    add("attn_q", {2*query,hidden}); add("attn_k", {kv,hidden}); add("attn_v", {kv,hidden});
    add("attn_output", {hidden,query});
    add("attn_q_norm", {attention.head_dimension}, true); add("attn_k_norm", {attention.head_dimension}, true);
    add("ffn_gate_inp", {expert.count,hidden}, true); add("ffn_gate_inp_shexp", {hidden}, true);
    for (const char *name : {"ffn_gate_exps", "ffn_up_exps"}) add(name, {hidden,expert.intermediate,expert.count});
    add("ffn_down_exps", {expert.intermediate,hidden,expert.count});
    for (const char *name : {"ffn_gate_shexp", "ffn_up_shexp"}) add(name, {expert.shared_intermediate,hidden});
    add("ffn_down_shexp", {hidden,expert.shared_intermediate});
    for (const char *role : {"attn", "ffn"})
    {
        const auto name = std::string("hc_") + role;
        add(name + "_down", {hc.gate_rank,trunk}); add(name + "_up", {trunk,hc.gate_rank});
        add(name + "_inject", {hc.branches,trunk}); add(name + "_norm", {trunk}, true);
    }
    const bool indexed = attention_ == Qwen4ExpPredictorAttention::Sparse;
    add("indexer.q_proj", {uint64_t(attention.indexer_query_heads)*attention.indexer_dimension,hidden}, false, indexed);
    add("indexer.k_proj", {uint64_t(attention.indexer_key_heads)*attention.indexer_dimension,hidden}, false, indexed);
    add("indexer.q_norm", {attention.indexer_dimension}, true, indexed);
    add("indexer.k_norm", {attention.indexer_dimension}, true, indexed);
    add("nextn.eh_proj", {hidden,2*hidden}); add("nextn.enorm", {hidden}, true); add("nextn.hnorm", {trunk}, true);
    add("nextn.hc_head_down", {hc.gate_rank,trunk}); add("nextn.hc_head_up", {trunk,hc.gate_rank});
    add("nextn.hc_head_norm", {trunk}, true);
    require(predictor.tensors.size() == expected.size(), "shared sidecar contains undeclared or duplicated source tensors");
    for (const auto &info : predictor.tensors) require(expected.contains(info.name), info.name + " unexpected predictor role");
}

const Qwen4ExpSidecarTensor &Qwen4ExpSidecarManifest::tensor(const std::string &name) const
{
    const auto found = std::find_if(sources_.begin(), sources_.end(), [&](const auto &item) { return item.tensor.name == name; });
    if (found == sources_.end()) throw std::invalid_argument("Qwen4Exp predictor has no source role " + name);
    return *found;
}

std::vector<std::string> Qwen4ExpSidecarManifest::consumedNames() const
{
    std::vector<std::string> names;
    for (const auto &item : sources_) if (item.consumed) names.push_back(item.tensor.name);
    return names;
}
} // namespace llaminar2
