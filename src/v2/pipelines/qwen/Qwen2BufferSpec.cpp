/**
 * @file Qwen2BufferSpec.cpp
 * @brief Implementation of Qwen2 buffer specification builder
 * @author David Sanftenberg
 * @date January 2025
 */

#include "Qwen2BufferSpec.h"
#include <stdexcept>

namespace llaminar2
{

    // =========================================================================
    // Qwen2BufferSpecBuilder Implementation
    // =========================================================================

    Qwen2BufferSpecBuilder::Qwen2BufferSpecBuilder(
        int d_model,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int d_ff,
        int vocab_size,
        BufferTensorType activation_type,
        int device_idx)
        : d_model_(d_model), n_heads_(n_heads), n_kv_heads_(n_kv_heads), head_dim_(head_dim), d_ff_(d_ff), vocab_size_(vocab_size), activation_type_(activation_type), device_idx_(device_idx)
    {
        if (d_model <= 0 || n_heads <= 0 || head_dim <= 0)
        {
            throw std::invalid_argument("Invalid model dimensions");
        }
    }

    size_t Qwen2BufferSpecBuilder::elementSize() const
    {
        switch (activation_type_)
        {
        case BufferTensorType::FP32:
            return 4;
        case BufferTensorType::FP16:
        case BufferTensorType::BF16:
            return 2;
        default:
            return 4;
        }
    }

    std::vector<Qwen2BufferSpec> Qwen2BufferSpecBuilder::buildLayerSpecs(int seq_len) const
    {
        std::vector<Qwen2BufferSpec> specs;

        // === INOUT Buffers ===
        // These persist across attention and FFN phases within a layer

        specs.push_back({BufferNames::RESIDUAL,
                         BufferRole::INOUT,
                         activation_type_,
                         {static_cast<size_t>(seq_len), static_cast<size_t>(d_model_)},
                         device_idx_,
                         "Residual connection accumulator [seq_len, d_model]"});

        specs.push_back({BufferNames::NORMALIZED,
                         BufferRole::INOUT,
                         activation_type_,
                         {static_cast<size_t>(seq_len), static_cast<size_t>(d_model_)},
                         device_idx_,
                         "Normalized hidden state output [seq_len, d_model]"});

        // Add attention-specific buffers
        auto attn_specs = buildAttentionSpecs(seq_len);
        specs.insert(specs.end(), attn_specs.begin(), attn_specs.end());

        // Add FFN-specific buffers
        auto ffn_specs = buildFFNSpecs(seq_len);
        specs.insert(specs.end(), ffn_specs.begin(), ffn_specs.end());

        return specs;
    }

    std::vector<Qwen2BufferSpec> Qwen2BufferSpecBuilder::buildModelSpecs(int batch_size, int seq_len) const
    {
        std::vector<Qwen2BufferSpec> specs;

        size_t total_tokens = static_cast<size_t>(batch_size) * static_cast<size_t>(seq_len);

        specs.push_back({BufferNames::CURRENT_HIDDEN,
                         BufferRole::INOUT,
                         activation_type_,
                         {total_tokens, static_cast<size_t>(d_model_)},
                         device_idx_,
                         "Current hidden states [batch * seq_len, d_model]"});

        specs.push_back({BufferNames::LOGITS,
                         BufferRole::OUTPUT,
                         activation_type_,
                         {total_tokens, static_cast<size_t>(vocab_size_)},
                         device_idx_,
                         "Output logits [batch * seq_len, vocab_size]"});

        return specs;
    }

    std::vector<Qwen2BufferSpec> Qwen2BufferSpecBuilder::buildAttentionSpecs(int seq_len) const
    {
        std::vector<Qwen2BufferSpec> specs;

        // Q projection: [seq_len, n_heads * head_dim]
        specs.push_back({BufferNames::Q,
                         BufferRole::SCRATCH,
                         activation_type_,
                         {static_cast<size_t>(seq_len), static_cast<size_t>(n_heads_ * head_dim_)},
                         device_idx_,
                         "Query projection output [seq_len, n_heads * head_dim]"});

        // K projection: [seq_len, n_kv_heads * head_dim]
        specs.push_back({BufferNames::K,
                         BufferRole::SCRATCH,
                         activation_type_,
                         {static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads_ * head_dim_)},
                         device_idx_,
                         "Key projection output [seq_len, n_kv_heads * head_dim]"});

        // V projection: [seq_len, n_kv_heads * head_dim]
        specs.push_back({BufferNames::V,
                         BufferRole::SCRATCH,
                         activation_type_,
                         {static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads_ * head_dim_)},
                         device_idx_,
                         "Value projection output [seq_len, n_kv_heads * head_dim]"});

        // Attention output (pre-Wo): [seq_len, n_heads * head_dim]
        specs.push_back({BufferNames::ATTN_OUTPUT,
                         BufferRole::SCRATCH,
                         activation_type_,
                         {static_cast<size_t>(seq_len), static_cast<size_t>(n_heads_ * head_dim_)},
                         device_idx_,
                         "Attention context output (pre-Wo) [seq_len, n_heads * head_dim]"});

        // Attention projection (post-Wo): [seq_len, d_model]
        specs.push_back({BufferNames::ATTN_PROJ,
                         BufferRole::SCRATCH,
                         activation_type_,
                         {static_cast<size_t>(seq_len), static_cast<size_t>(d_model_)},
                         device_idx_,
                         "Attention projection output (post-Wo) [seq_len, d_model]"});

        // Workspace buffers for attention computation
        // scores: [n_heads, seq_len, kv_seq_len] - for incremental decode, kv_seq_len = position + seq_len
        // For simplicity, allocate worst-case (assuming context length up to some max)
        // In practice, these should be sized dynamically based on context window

        // For now, use seq_len as the workspace dimension (single-token decode case)
        specs.push_back({BufferNames::WORKSPACE_SCORES,
                         BufferRole::SCRATCH,
                         activation_type_,
                         {static_cast<size_t>(n_heads_), static_cast<size_t>(seq_len), static_cast<size_t>(seq_len)},
                         device_idx_,
                         "Attention scores workspace [n_heads, seq_len, seq_len]"});

        specs.push_back({BufferNames::WORKSPACE_CONTEXT,
                         BufferRole::SCRATCH,
                         activation_type_,
                         {static_cast<size_t>(n_heads_), static_cast<size_t>(seq_len), static_cast<size_t>(head_dim_)},
                         device_idx_,
                         "Attention context workspace [n_heads, seq_len, head_dim]"});

        specs.push_back({BufferNames::WORKSPACE_MASK,
                         BufferRole::SCRATCH,
                         activation_type_,
                         {static_cast<size_t>(seq_len), static_cast<size_t>(seq_len)},
                         device_idx_,
                         "Causal mask workspace [seq_len, seq_len]"});

        return specs;
    }

    std::vector<Qwen2BufferSpec> Qwen2BufferSpecBuilder::buildFFNSpecs(int seq_len) const
    {
        std::vector<Qwen2BufferSpec> specs;

        // Gate projection: [seq_len, d_ff]
        specs.push_back({BufferNames::GATE,
                         BufferRole::SCRATCH,
                         activation_type_,
                         {static_cast<size_t>(seq_len), static_cast<size_t>(d_ff_)},
                         device_idx_,
                         "FFN gate projection [seq_len, d_ff]"});

        // Up projection: [seq_len, d_ff]
        specs.push_back({BufferNames::UP,
                         BufferRole::SCRATCH,
                         activation_type_,
                         {static_cast<size_t>(seq_len), static_cast<size_t>(d_ff_)},
                         device_idx_,
                         "FFN up projection [seq_len, d_ff]"});

        // FFN output (post-SwiGLU, pre-down): [seq_len, d_ff]
        specs.push_back({BufferNames::FFN_OUTPUT,
                         BufferRole::SCRATCH,
                         activation_type_,
                         {static_cast<size_t>(seq_len), static_cast<size_t>(d_ff_)},
                         device_idx_,
                         "FFN intermediate (post-SwiGLU) [seq_len, d_ff]"});

        return specs;
    }

    std::vector<std::vector<std::string>> Qwen2BufferSpecBuilder::getAliasingGroups() const
    {
        // Theoretical aliasing groups based on Qwen2 execution order:
        //
        // Within a single layer:
        // 1. Attention phase: Q, K, V computed, then consumed by attention
        // 2. FFN phase: gate, up computed, then consumed by SwiGLU
        //
        // Non-overlapping:
        // - Q ↔ gate (Q consumed before FFN starts)
        // - K ↔ up (K consumed before FFN starts)
        // - V ↔ ffn_output (V consumed before FFN down proj)
        // - attn_output ↔ gate (attn_output consumed by Wo before FFN)

        return {
            // Group 1: Q can alias with gate
            {BufferNames::Q, BufferNames::GATE},
            // Group 2: K can alias with up
            {BufferNames::K, BufferNames::UP},
            // Group 3: V can alias with ffn_output
            {BufferNames::V, BufferNames::FFN_OUTPUT},
            // Group 4: attn_output can alias with ffn_output (if different from V group)
            // Note: In practice, LivenessAnalyzer determines actual aliasing
        };
    }

    std::pair<size_t, size_t> Qwen2BufferSpecBuilder::estimateMemorySavings(int seq_len) const
    {
        size_t elem_size = elementSize();
        size_t s = static_cast<size_t>(seq_len);

        // Attention buffers
        size_t q_size = s * n_heads_ * head_dim_ * elem_size;
        size_t k_size = s * n_kv_heads_ * head_dim_ * elem_size;
        size_t v_size = s * n_kv_heads_ * head_dim_ * elem_size;
        size_t attn_out_size = s * n_heads_ * head_dim_ * elem_size;

        // FFN buffers
        size_t gate_size = s * d_ff_ * elem_size;
        size_t up_size = s * d_ff_ * elem_size;
        size_t ffn_out_size = s * d_ff_ * elem_size;

        // Original: all buffers allocated separately
        size_t original = q_size + k_size + v_size + attn_out_size +
                          gate_size + up_size + ffn_out_size;

        // Optimized (theoretical):
        // - Q aliases with gate: max(q_size, gate_size)
        // - K aliases with up: max(k_size, up_size)
        // - V aliases with ffn_output: max(v_size, ffn_out_size)
        // - attn_output: separate (overlaps with FFN in decomposed execution)
        size_t optimized = std::max(q_size, gate_size) +
                           std::max(k_size, up_size) +
                           std::max(v_size, ffn_out_size) +
                           attn_out_size;

        return {original, optimized};
    }

    // =========================================================================
    // Utility Functions
    // =========================================================================

    StageBufferRequirements toBufferRequirements(const std::vector<Qwen2BufferSpec> &specs)
    {
        StageBufferRequirements reqs;

        for (const auto &spec : specs)
        {
            BufferDescriptor desc;
            desc.name = spec.name;
            desc.role = spec.role;
            desc.tensor_type = spec.tensor_type;
            desc.shape = spec.shape;
            desc.device_idx = spec.device_idx;

            reqs.buffers.push_back(std::move(desc));
        }

        return reqs;
    }

} // namespace llaminar2
