/**
 * @file DeepSeekV4Config.h
 * @brief Authenticated GGUF geometry for the DeepSeek-V4 text decoder.
 *
 * Parsing consumes only the loader's immutable directory and metadata written
 * by llama.cpp's `deepseek4` converter (b12079). Every value the graph, kernels
 * or memory planner depend on is read exactly once here; unsupported variants
 * (other compression ratios, hc != 4, non-sqrtsoftplus gating) fail closed.
 * The contract creates no tensor payload and advertises no backend.
 */
#pragma once

#include "../../kernels/deepseek4/DeepSeekV4Kernels.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llaminar2
{
struct GGUFModel;

/** @brief Attention family of one main layer. */
enum class DeepSeekV4AttentionKind : std::uint8_t
{
    Window,      ///< compress ratio 0: raw sliding window only.
    Compressed,  ///< ratio 4 (CSA): overlapping compressor + lightning indexer top-k.
    Heavy,       ///< ratio 128 (HCA): plain compressor, every visible block attended.
};

/** @brief Immutable per-layer policy derived from compress_ratios. */
struct DeepSeekV4LayerPolicy
{
    DeepSeekV4AttentionKind kind = DeepSeekV4AttentionKind::Window;
    int ratio = 0;                 ///< 0, 4 or 128.
    bool hashRouted = false;       ///< First hash_layer_count layers route text tokens by token id.
    float swigluLimit = 0.0f;      ///< Routed-expert SwiGLU clamp.
    float sharedSwigluLimit = 0.0f;///< Shared-expert SwiGLU clamp.
    deepseek4::RopePolicy rope;    ///< Layer rotation: YaRN + compress base when ratio != 0.
};

/** @brief Authenticated DeepSeek-V4 text-decoder geometry and policy. */
class DeepSeekV4Config
{
public:
    /**
     * @brief Parse and validate the `deepseek4.*` metadata of @p model.
     * @throws std::invalid_argument with the offending key for unsupported or missing metadata.
     */
    explicit DeepSeekV4Config(const GGUFModel &model);

    int layers() const noexcept { return layers_; }
    int hidden() const noexcept { return hidden_; }
    int hcMult() const noexcept { return hc_mult_; }
    int residualWidth() const noexcept { return hidden_ * hc_mult_; }
    int heads() const noexcept { return heads_; }
    int headDim() const noexcept { return head_dim_; }
    int ropeDim() const noexcept { return rope_dim_; }
    int qLoraRank() const noexcept { return q_lora_rank_; }
    int outputGroups() const noexcept { return o_groups_; }
    int outputLoraRank() const noexcept { return o_lora_rank_; }
    int window() const noexcept { return window_; }
    int experts() const noexcept { return experts_; }
    int expertsUsed() const noexcept { return experts_used_; }
    int expertIntermediate() const noexcept { return expert_ff_; }
    int sharedExperts() const noexcept { return shared_experts_; }
    int hashLayers() const noexcept { return hash_layers_; }
    float routeScale() const noexcept { return route_scale_; }
    int indexerHeads() const noexcept { return index_heads_; }
    int indexerHeadDim() const noexcept { return index_head_dim_; }
    int indexerTopK() const noexcept { return index_topk_; }
    int sinkhornIters() const noexcept { return sinkhorn_iters_; }
    float hcEps() const noexcept { return hc_eps_; }
    float normEps() const noexcept { return norm_eps_; }
    int contextLength() const noexcept { return context_length_; }
    int vocab() const noexcept { return vocab_; }
    /** @brief Policy of main layer @p layer. @throws std::out_of_range */
    const DeepSeekV4LayerPolicy &layer(int layer) const { return layer_policy_.at(static_cast<std::size_t>(layer)); }
    /** @brief Number of main layers with the given attention kind. */
    int countLayers(DeepSeekV4AttentionKind kind) const noexcept;

private:
    int layers_ = 0, hidden_ = 0, hc_mult_ = 0, heads_ = 0, head_dim_ = 0, rope_dim_ = 0;
    int q_lora_rank_ = 0, o_groups_ = 0, o_lora_rank_ = 0, window_ = 0;
    int experts_ = 0, experts_used_ = 0, expert_ff_ = 0, shared_experts_ = 0, hash_layers_ = 0;
    float route_scale_ = 1.0f;
    int index_heads_ = 0, index_head_dim_ = 0, index_topk_ = 0, sinkhorn_iters_ = 0;
    float hc_eps_ = 0.0f, norm_eps_ = 0.0f;
    int context_length_ = 0, vocab_ = 0;
    std::vector<DeepSeekV4LayerPolicy> layer_policy_;
};

/**
 * @brief Routed-expert SwiGLU clamp for runners that hold only the GGUF (e.g.
 *        ExpertOverlay participant graphs).
 * @return The authenticated `swiglu_clamp_exp` of @p layer for a `deepseek4`
 *         model, 0 for every other architecture.
 * @throws std::invalid_argument when DeepSeek-V4 metadata is invalid.
 */
float deepSeekV4RoutedSwiGLULimit(const GGUFModel &model, int layer);

} // namespace llaminar2
