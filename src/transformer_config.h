/**
 * @file transformer_config.h
 * @brief Configuration structures for transformer models.
 * @author David Sanftenberg
 */

#pragma once

#include <string>

/**
 * @brief Core transformer layer hyperparameters.
 *
 * This structure contains the essential numerical configuration for transformer layers,
 * independent of architectural variations or feature support.
 */
struct TransformerLayerConfig
{
    int n_head;                      ///< Number of attention heads
    int n_head_kv;                   ///< Number of key-value heads (for grouped attention)
    int head_dim;                    ///< Dimension per attention head
    int d_model;                     ///< Model dimension (embedding size)
    int d_ff;                        ///< Feed-forward hidden dimension
    int vocab_size;                  ///< Vocabulary size
    int max_seq_len;                 ///< Maximum sequence length
    int n_layers;                    ///< Number of transformer layers
    float eps;                       ///< RMS norm epsilon
    float rope_freq_base = 10000.0f; ///< Base frequency for rotary position embeddings
};

/**
 * @brief Model-level configuration with architecture and feature capabilities.
 *
 * This structure wraps TransformerLayerConfig and adds architectural metadata and
 * feature capability flags. It enables the factory and pipeline to make informed
 * decisions about which kernels, optimizations, and paths to use.
 *
 * Feature flags indicate support for various transformer components:
 * - Positional encodings (RoPE, ALiBi)
 * - Normalization schemes (RMSNorm, LayerNorm)
 * - Attention variants (GQA, MQA, sliding window)
 * - Specialized components (MoE)
 */
struct ModelConfig
{
    /**
     * @brief Core layer configuration.
     */
    TransformerLayerConfig layer_config;

    /**
     * @brief Architecture identifier (e.g., "qwen", "llama", "mistral").
     *
     * Used by PipelineFactory to select the appropriate pipeline implementation.
     * Default: "qwen" for backward compatibility.
     */
    std::string architecture = "qwen";

    // === Feature Capability Flags ===

    /**
     * @brief Uses Rotary Position Embeddings (RoPE).
     *
     * When true, the model uses RoPE for positional encoding.
     * The base frequency is specified in layer_config.rope_freq_base.
     */
    bool has_rope = true;

    /**
     * @brief Uses RMSNorm for layer normalization.
     *
     * When true, the model uses Root Mean Square Normalization.
     * When false, standard LayerNorm is assumed.
     */
    bool has_rmsnorm = true;

    /**
     * @brief Uses Attention with Linear Biases (ALiBi).
     *
     * When true, the model uses ALiBi instead of explicit positional embeddings.
     * Mutually exclusive with has_rope in most architectures.
     */
    bool has_alibi = false;

    /**
     * @brief Uses Grouped Query Attention (GQA).
     *
     * When true, n_head_kv < n_head (key/value heads are shared across query heads).
     * When false, Multi-Head Attention (MHA) or Multi-Query Attention (MQA) is used.
     */
    bool has_gqa = false;

    /**
     * @brief Uses sliding window attention.
     *
     * When true, attention is restricted to a local window rather than full causal attention.
     * The window size would be stored in an additional field if needed.
     */
    bool has_sliding_window = false;

    /**
     * @brief Uses Mixture of Experts (MoE).
     *
     * When true, the FFN layer is replaced with a gated MoE architecture.
     */
    bool has_moe = false;

    // === Convenience Constructors ===

    /**
     * @brief Default constructor.
     */
    ModelConfig() = default;

    /**
     * @brief Construct from layer config with default architecture.
     *
     * @param config Core layer configuration
     * @param arch Architecture name (default: "qwen")
     *
     * This constructor provides backward compatibility for code that only has
     * TransformerLayerConfig. Feature flags are set to defaults appropriate for
     * standard Qwen-like architectures.
     */
    explicit ModelConfig(const TransformerLayerConfig &config, const std::string &arch = "qwen")
        : layer_config(config), architecture(arch)
    {
        // Auto-detect GQA from layer config
        has_gqa = (config.n_head_kv < config.n_head);
    }

    /**
     * @brief Get reference to underlying layer config (const).
     */
    const TransformerLayerConfig &getLayerConfig() const { return layer_config; }

    /**
     * @brief Get reference to underlying layer config (mutable).
     */
    TransformerLayerConfig &getLayerConfig() { return layer_config; }
};
