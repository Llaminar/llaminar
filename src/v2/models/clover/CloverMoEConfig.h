/**
 * @file CloverMoEConfig.h
 * @brief Validated source geometry for Clover's causal autoregressive graph.
 *
 * Clover uses the Qwen hybrid GDN/full-attention and routed/shared expert
 * arithmetic in autoregressive mode. The directory, not a Qwen architecture
 * alias, authenticates that reuse. Diffusion-only tensors in the original
 * Preview export are recognized but never materialized for causal inference.
 * No weights, execution state or physical-memory ledger are owned here.
 */
#pragma once

#include <string_view>

namespace llaminar2
{
    struct GGUFModel;
    struct GraphConfig;
    struct ModelMemoryProfile;

    /** @brief Scalar dimensions authenticated together with every causal tensor. */
    struct CloverMoEGeometry
    {
        int layers;                 ///< Main transformer blocks; no nextn sidecar.
        int context_length;         ///< Source maximum token context.
        int model_width;            ///< Residual and projection input width.
        int vocabulary;             ///< Rows in the embedding and terminal head.
        int query_heads;            ///< Full-attention query heads.
        int kv_heads;               ///< Full-attention key/value heads.
        int head_dimension;         ///< Equal full-attention K/V head widths.
        int rotary_dimension;       ///< Even partial NeoX rotary width.
        int attention_interval;     ///< Periodic full-attention layer spacing.
        int key_heads;              ///< GDN key groups in GGUF tiled-head order.
        int value_heads;            ///< GDN value heads, divisible by key_heads.
        int state_dimension;        ///< Equal GDN key/value head width.
        int value_width;            ///< state_dimension * value_heads.
        int convolution_kernel;     ///< Causal short-convolution width.
        int experts;                ///< Total routed experts in each block.
        int selected_experts;       ///< Normalized softmax top-k route count.
        int expert_width;           ///< Routed SwiGLU intermediate width.
        int shared_width;           ///< Aggregate shared expert width, or zero.
        float norm_epsilon;         ///< Positive exported RMSNorm epsilon.
        float rotary_base;          ///< Positive exported rotary frequency base.
    };

    /** @brief Immutable admission proof for the implemented periodic causal family. */
    class CloverMoEConfig final
    {
    public:
        /**
         * @brief Validate a complete canonical ModelLoader directory without payload I/O.
         * @param model GGUF metadata and loader-order tensor shapes ([N,K] for matrices).
         * @throws std::invalid_argument For incomplete, contradictory or unsupported geometry.
         *
         * Both the official Preview export (implicit single shared expert) and
         * the upstream AR converter's explicit shared count are supported.
         * Nonperiodic layer maps and learned nextn prediction are rejected.
         */
        explicit CloverMoEConfig(const GGUFModel &model);

        /** @return Authenticated dimensions; callers cannot mutate this proof. */
        const CloverMoEGeometry &geometry() const noexcept { return geometry_; }

        /**
         * @brief Project the authenticated architecture into an existing participant config.
         * @param config Caller-owned execution settings and optional PP interval.
         * @throws std::invalid_argument If the requested interval exceeds the main model.
         */
        void populate(GraphConfig &config) const;

        /**
         * @brief Publish the identical scalar geometry into the canonical planning BOM input.
         * @param profile Existing model profile whose tensor inventory remains source-owned.
         */
        void populate(ModelMemoryProfile &profile) const;

        /**
         * @brief Identify the four exact original Preview diffusion-only tensor suffixes.
         * @param name Canonical block tensor name, including its numeric layer index.
         * @return True only for noncausal operands that AR neither loads nor charges.
         */
        static bool isDiffusionOnlyTensor(std::string_view name) noexcept;

    private:
        CloverMoEGeometry geometry_{};
    };
}
