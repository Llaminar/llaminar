/**
 * @file DeepSeekV4Config.cpp
 * @brief Fail-closed parsing of the llama.cpp `deepseek4` GGUF text contract.
 *
 * Metadata names follow llama.cpp conversion/deepseek.py (b12079). Shapes of
 * the per-layer tensors are authenticated against the declared geometry so a
 * truncated or foreign file fails before any weight is prepared.
 */
#include "DeepSeekV4Config.h"
#include "../../loaders/ModelLoader.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
namespace
{
constexpr const char *kPrefix = "deepseek4.";

[[noreturn]] void reject(const std::string &key, const std::string &reason)
{
    throw std::invalid_argument("deepseek4: " + key + ": " + reason);
}

const GGUFValue &required(const GGUFModel &model, const std::string &suffix)
{
    const std::string key = kPrefix + suffix;
    const auto found = model.metadata.find(key);
    if (found == model.metadata.end())
        reject(key, "required metadata is missing");
    return found->second;
}

int positiveInt(const GGUFModel &model, const std::string &suffix)
{
    uint64_t value = 0;
    try { value = required(model, suffix).asUInt64Checked(); }
    catch (const std::invalid_argument &error) { reject(kPrefix + suffix, error.what()); }
    if (value == 0 || value > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        reject(kPrefix + suffix, "requires a positive int-range value");
    return static_cast<int>(value);
}

float finiteFloat(const GGUFModel &model, const std::string &suffix)
{
    const auto &raw = required(model, suffix);
    if (raw.type != GGUFValueType::FLOAT32 || raw.data.size() != sizeof(float))
        reject(kPrefix + suffix, "requires an exact FLOAT32 scalar");
    const float value = raw.asFloat32();
    if (!std::isfinite(value))
        reject(kPrefix + suffix, "requires a finite value");
    return value;
}

/** @brief Decode a FLOAT32 array (llama.cpp writes per-layer clamp limits as arrays). */
std::vector<float> floatArray(const GGUFModel &model, const std::string &suffix)
{
    const auto &raw = required(model, suffix);
    if (raw.type != GGUFValueType::ARRAY || raw.array_element_type != GGUFValueType::FLOAT32 ||
        raw.data.size() != raw.array_length * sizeof(float))
        reject(kPrefix + suffix, "requires a FLOAT32 array");
    std::vector<float> values(raw.array_length);
    std::memcpy(values.data(), raw.data.data(), raw.data.size());
    return values;
}

bool boolValue(const GGUFModel &model, const std::string &suffix)
{
    const auto &raw = required(model, suffix);
    if (raw.type != GGUFValueType::BOOL || raw.data.size() != 1)
        reject(kPrefix + suffix, "requires a BOOL scalar");
    return raw.data[0] != 0;
}

void requireTensor(const GGUFModel &model, const std::string &name, std::vector<uint64_t> shape)
{
    const auto *tensor = model.findTensor(name);
    if (!tensor)
        reject(name, "required tensor is missing");
    if (tensor->dimensions != shape)
        reject(name, "tensor shape does not match declared geometry");
}
} // namespace

DeepSeekV4Config::DeepSeekV4Config(const GGUFModel &model)
{
    if (model.architecture != "deepseek4")
        reject("general.architecture", "requires deepseek4, got " + model.architecture);
    layers_ = positiveInt(model, "block_count");
    if (model.hasMetadata(std::string(kPrefix) + "nextn_predict_layers"))
        reject("deepseek4.nextn_predict_layers", "embedded MTP blocks are not supported yet");
    hidden_ = positiveInt(model, "embedding_length");
    heads_ = positiveInt(model, "attention.head_count");
    if (positiveInt(model, "attention.head_count_kv") != 1)
        reject("deepseek4.attention.head_count_kv", "DeepSeek-V4 attention is single-latent MQA");
    head_dim_ = positiveInt(model, "attention.key_length");
    if (positiveInt(model, "attention.value_length") != head_dim_)
        reject("deepseek4.attention.value_length", "requires K == V width");
    rope_dim_ = positiveInt(model, "rope.dimension_count");
    if (rope_dim_ % 2 != 0 || rope_dim_ > head_dim_)
        reject("deepseek4.rope.dimension_count", "requires an even width within the head");
    q_lora_rank_ = positiveInt(model, "attention.q_lora_rank");
    o_groups_ = positiveInt(model, "attention.output_group_count");
    o_lora_rank_ = positiveInt(model, "attention.output_lora_rank");
    if (heads_ % o_groups_ != 0)
        reject("deepseek4.attention.output_group_count", "must divide the head count");
    window_ = positiveInt(model, "attention.sliding_window");
    experts_ = positiveInt(model, "expert_count");
    experts_used_ = positiveInt(model, "expert_used_count");
    if (experts_used_ > experts_)
        reject("deepseek4.expert_used_count", "exceeds expert_count");
    expert_ff_ = positiveInt(model, "expert_feed_forward_length");
    shared_experts_ = positiveInt(model, "expert_shared_count");
    if (shared_experts_ != 1)
        reject("deepseek4.expert_shared_count", "requires exactly one shared expert");
    hash_layers_ = static_cast<int>(required(model, "hash_layer_count").asUInt64Checked());
    route_scale_ = finiteFloat(model, "expert_weights_scale");
    if (!boolValue(model, "expert_weights_norm"))
        reject("deepseek4.expert_weights_norm", "requires normalized top-k weights");
    if (required(model, "expert_gating_func").asUInt64Checked() != 4)
        reject("deepseek4.expert_gating_func", "requires sqrtsoftplus gating (4)");
    index_heads_ = positiveInt(model, "attention.indexer.head_count");
    index_head_dim_ = positiveInt(model, "attention.indexer.key_length");
    index_topk_ = positiveInt(model, "attention.indexer.top_k");
    if (index_head_dim_ < rope_dim_)
        reject("deepseek4.attention.indexer.key_length", "must hold the rotary dims");
    hc_mult_ = positiveInt(model, "hyper_connection.count");
    if (hc_mult_ != 4)
        reject("deepseek4.hyper_connection.count", "kernels support hc_mult == 4");
    sinkhorn_iters_ = positiveInt(model, "hyper_connection.sinkhorn_iterations");
    hc_eps_ = finiteFloat(model, "hyper_connection.epsilon");
    norm_eps_ = finiteFloat(model, "attention.layer_norm_rms_epsilon");
    context_length_ = positiveInt(model, "context_length");
    vocab_ = static_cast<int>(model.vocab_size);
    if (vocab_ <= 0)
        reject("tokenizer.ggml.tokens", "vocabulary size is unknown");

    const float rope_base = finiteFloat(model, "rope.freq_base");
    const float compress_base = finiteFloat(model, "attention.compress_rope_freq_base");
    const int original_context = model.hasMetadata("deepseek4.rope.scaling.original_context_length")
                                     ? positiveInt(model, "rope.scaling.original_context_length") : 0;
    const float factor = original_context ? finiteFloat(model, "rope.scaling.factor") : 1.0f;
    const float beta_fast = original_context ? finiteFloat(model, "rope.scaling.yarn_beta_fast") : 32.0f;
    const float beta_slow = original_context ? finiteFloat(model, "rope.scaling.yarn_beta_slow") : 1.0f;

    std::vector<uint64_t> ratios;
    try { ratios = required(model, "attention.compress_ratios").asUInt64Array(); }
    catch (const std::invalid_argument &error) { reject("deepseek4.attention.compress_ratios", error.what()); }
    if (ratios.size() < static_cast<std::size_t>(layers_))
        reject("deepseek4.attention.compress_ratios", "shorter than block_count");
    const auto limits = floatArray(model, "swiglu_clamp_exp");
    const auto shared_limits = model.hasMetadata("deepseek4.swiglu_clamp_shexp")
                                   ? floatArray(model, "swiglu_clamp_shexp") : limits;
    if (limits.size() < static_cast<std::size_t>(layers_) || shared_limits.size() < static_cast<std::size_t>(layers_))
        reject("deepseek4.swiglu_clamp_exp", "shorter than block_count");

    const uint64_t d = static_cast<uint64_t>(hidden_);
    const uint64_t hd = static_cast<uint64_t>(head_dim_);
    layer_policy_.resize(static_cast<std::size_t>(layers_));
    for (int il = 0; il < layers_; ++il)
    {
        auto &policy = layer_policy_[static_cast<std::size_t>(il)];
        policy.ratio = static_cast<int>(ratios[static_cast<std::size_t>(il)]);
        const std::string p = "blk." + std::to_string(il) + ".";
        switch (policy.ratio)
        {
        case 0: policy.kind = DeepSeekV4AttentionKind::Window; break;
        case 4: policy.kind = DeepSeekV4AttentionKind::Compressed; break;
        case 128: policy.kind = DeepSeekV4AttentionKind::Heavy; break;
        default: reject("deepseek4.attention.compress_ratios", "unsupported ratio at layer " + std::to_string(il));
        }
        policy.hashRouted = il < hash_layers_;
        policy.swigluLimit = limits[static_cast<std::size_t>(il)];
        policy.sharedSwigluLimit = shared_limits[static_cast<std::size_t>(il)];
        policy.rope = policy.ratio != 0
            ? deepseek4::RopePolicy::make(rope_dim_, compress_base, original_context, factor, beta_fast, beta_slow)
            : deepseek4::RopePolicy::make(rope_dim_, rope_base, 0, 1.0f, beta_fast, beta_slow);

        // Authenticate the geometry the graph consumes. ModelLoader publishes 2D
        // tensors as row-major [rows, cols] = [ggml ne1, ne0].
        const uint64_t mix = static_cast<uint64_t>((2 + hc_mult_) * hc_mult_);
        requireTensor(model, p + "hc_attn_fn.weight", {mix, d * hc_mult_});
        requireTensor(model, p + "hc_ffn_fn.weight", {mix, d * hc_mult_});
        requireTensor(model, p + "attn_q_a.weight", {static_cast<uint64_t>(q_lora_rank_), d});
        requireTensor(model, p + "attn_q_b.weight", {hd * heads_, static_cast<uint64_t>(q_lora_rank_)});
        requireTensor(model, p + "attn_kv.weight", {hd, d});
        requireTensor(model, p + "attn_output_a.weight",
                      {static_cast<uint64_t>(o_groups_) * o_lora_rank_, hd * heads_ / o_groups_});
        requireTensor(model, p + "attn_output_b.weight", {d, static_cast<uint64_t>(o_groups_) * o_lora_rank_});
        requireTensor(model, p + "attn_sinks.weight", {static_cast<uint64_t>(heads_)});
        if (policy.ratio != 0)
        {
            const uint64_t coff = policy.ratio == 4 ? 2 : 1;
            requireTensor(model, p + "attn_compressor_kv.weight", {coff * hd, d});
            requireTensor(model, p + "attn_compressor_gate.weight", {coff * hd, d});
            requireTensor(model, p + "attn_compressor_ape.weight", {static_cast<uint64_t>(policy.ratio), coff * hd});
        }
        if (policy.ratio == 4)
        {
            requireTensor(model, p + "indexer.attn_q_b.weight",
                          {static_cast<uint64_t>(index_heads_) * index_head_dim_, static_cast<uint64_t>(q_lora_rank_)});
            requireTensor(model, p + "indexer.proj.weight", {static_cast<uint64_t>(index_heads_), d});
            requireTensor(model, p + "indexer_compressor_kv.weight", {2ull * index_head_dim_, d});
        }
        if (policy.hashRouted)
            requireTensor(model, p + "ffn_gate_tid2eid.weight",
                          {static_cast<uint64_t>(vocab_), static_cast<uint64_t>(experts_used_)});
        else if (!model.findTensor(p + "exp_probs_b.bias"))
            reject(p + "exp_probs_b.bias", "score-routed layer requires a selection bias");
    }
    // Vision variants carry the image routing bias on every layer; a partial
    // set is a malformed checkpoint rather than a text model.
    int vl_layers = 0;
    for (int il = 0; il < layers_; ++il)
        if (const auto *t = model.findTensor("blk." + std::to_string(il) + ".exp_probs_b_vl.bias"))
        {
            requireTensor(model, t->name, {static_cast<uint64_t>(experts_)});
            ++vl_layers;
        }
    if (vl_layers != 0 && vl_layers != layers_)
        reject("exp_probs_b_vl.bias", "a vision checkpoint needs the image routing bias on every layer");
    vision_capable_ = vl_layers == layers_;
}

int DeepSeekV4Config::countLayers(DeepSeekV4AttentionKind kind) const noexcept
{
    int count = 0;
    for (const auto &policy : layer_policy_) count += policy.kind == kind;
    return count;
}

float deepSeekV4RoutedSwiGLULimit(const GGUFModel &model, int layer)
{
    if (model.architecture != "deepseek4")
        return 0.0f;
    return DeepSeekV4Config(model).layer(layer).swigluLimit;
}

} // namespace llaminar2
