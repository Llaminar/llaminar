#pragma once

// Transformer layer configuration structure
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