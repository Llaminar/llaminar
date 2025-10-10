# Model Abstraction Strategy & Translation Pipeline

**Author**: AI Analysis based on HuggingFace Transformers Research  
**Date**: October 9, 2025  
**Status**: Strategic Planning Phase

## Executive Summary

This document outlines a comprehensive strategy for transforming Llaminar from a Qwen2-specific inference engine into a **model-generic, HuggingFace Transformers-faithful inference platform** supporting multiple architectures including Qwen2, Qwen3, Qwen3-MoE, DeepSeek-V3, and future models.

**Key Insights from Research**:
1. ✅ **90% code reuse potential** - Core transformer components (RMSNorm, RoPE, Attention, MLP) are nearly identical across models
2. ⚠️ **MoE is the major differentiator** - Qwen3-MoE and DeepSeek-V3 add expert routing, requiring abstraction of the MLP layer
3. ✅ **HuggingFace uses consistent patterns** - All models follow: `Attention → MLP/MoE → DecoderLayer → Model → ForCausalLM`
4. 🎯 **Configuration-driven architecture** - Model differences are primarily in hyperparameters, not fundamental math

---

## Part 1: Current Architecture Analysis

### What We Have (Qwen2-Specific)

```
Current Structure (Qwen2-Centric):
├── src/qwen_pipeline.{h,cpp}         # Qwen2-specific pipeline
├── src/qwen_pipeline_adapter.{h,cpp} # AbstractPipeline wrapper
├── src/kernels/
│   ├── MPIAttentionKernel.{h,cpp}    # ✓ Generic (needs Q/K norm for Qwen3)
│   ├── MPILinearKernel.{h,cpp}       # ✓ Generic
│   ├── MPIRMSNormKernel.{h,cpp}      # ✓ Generic
│   ├── MPIRoPEKernel.{h,cpp}         # ✓ Generic (HuggingFace compatible)
│   ├── MPIMLPKernel.{h,cpp}          # ⚠️ Needs MoE abstraction
│   └── MPISwiGLUKernel.{h,cpp}       # ✓ Generic
```

**Strengths**:
- ✅ Already has `AbstractPipeline` interface with factory pattern
- ✅ Parity Framework can validate against PyTorch/llama.cpp
- ✅ MPI-aware kernels are architecture-agnostic
- ✅ Weight contract system ensures correct tensor orientations

**Limitations**:
- ❌ Kernels live in global namespace (no per-model organization)
- ❌ MLP kernel is dense-only (no MoE support)
- ❌ Configuration is Qwen2-specific (no sliding window, no MoE params)
- ❌ No abstraction for model-specific layer composition

---

## Part 2: HuggingFace Transformer Patterns (Research Findings)

### Universal Components (98% Code Reuse)

All researched models (Qwen2, Qwen3, Qwen3-MoE, DeepSeek-V3) share:

```python
# PATTERN 1: RMSNorm - IDENTICAL across all models
class Qwen3RMSNorm(Qwen2RMSNorm):  # Literally inherits with no changes
    pass

class DeepseekV3RMSNorm(nn.Module):
    # Same implementation: variance = x.pow(2).mean(-1, keepdim=True)
    # hidden_states * torch.rsqrt(variance + eps) * weight

# PATTERN 2: RoPE - Identical math, different scaling
class Qwen3RotaryEmbedding(nn.Module):
    # inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2) / dim))
    # cos/sin computed identically to Qwen2

class DeepseekV3RotaryEmbedding(nn.Module):
    # SAME formula, optional attention_scaling multiplier
    # cos = emb.cos() * self.attention_scaling
    # sin = emb.sin() * self.attention_scaling

# PATTERN 3: Standard Attention (Qwen2, Qwen3, DeepSeek-V3 non-MoE layers)
class Qwen3Attention(nn.Module):
    def __init__(self, config, layer_idx):
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=attention_bias)
        
        # KEY DIFFERENCE: Qwen3 adds query/key normalization (optional feature)
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)  # NEW in Qwen3
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)  # NEW in Qwen3
    
    def forward(self, hidden_states, position_embeddings, attention_mask, ...):
        q = self.q_norm(self.q_proj(hidden_states))  # Extra norm step
        k = self.k_norm(self.k_proj(hidden_states))  # Extra norm step
        v = self.v_proj(hidden_states)
        
        q, k = apply_rotary_pos_emb(q, k, cos, sin)  # Identical RoPE
        
        if past_key_values:
            k, v = past_key_values.update(k, v, layer_idx, cache_kwargs)
        
        attn_out = attention_function(self, q, k, v, mask, scaling, ...)
        return self.o_proj(attn_out), attn_weights
```

### MLP Variants (Dense vs MoE)

**Dense MLP (Qwen2, Qwen3, DeepSeek-V3 first N layers)**:
```python
class Qwen3MLP(nn.Module):  # IDENTICAL to Qwen2
    def __init__(self, config):
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]  # Usually SiLU
    
    def forward(self, x):
        # SwiGLU: down_proj(act_fn(gate) * up)
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

**MoE (Qwen3-MoE, DeepSeek-V3 later layers)**:
```python
class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)  # Router
        self.experts = Qwen3MoeExperts(config)  # ModuleList of MLPs
        self.num_experts_per_tok = config.num_experts_per_tok  # Top-K routing
        self.norm_topk_prob = config.norm_topk_prob
    
    def route_tokens_to_experts(self, hidden_states, router_logits):
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        return selected_experts, routing_weights
    
    def forward(self, hidden_states):
        router_logits = self.gate(hidden_states)
        selected_experts, routing_weights = self.route_tokens_to_experts(
            hidden_states, router_logits
        )
        # Each token routed to top-K experts, outputs weighted & summed
        final_output = self.experts(hidden_states, selected_experts, routing_weights)
        return final_output  # Same shape as input (no residual here)

class Qwen2MoeSparseMoeBlock(nn.Module):
    # EXTENDED version with shared experts (Qwen2-MoE specific)
    def forward(self, hidden_states):
        shared_expert_output = self.shared_expert(hidden_states)  # Always active
        router_logits = self.gate(hidden_states)
        selected_experts, routing_weights = self.route_tokens_to_experts(...)
        expert_output = self.experts(...)
        
        # Shared experts gated and added to routed experts
        shared_output = F.sigmoid(self.shared_expert_gate(h)) * shared_expert_output
        return expert_output + shared_output
```

### DecoderLayer Composition

**Universal Pattern** (All models):
```python
class Qwen3DecoderLayer(nn.Module):  # Also Qwen2, DeepSeek-V3
    def __init__(self, config, layer_idx):
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)  # OR Qwen3MoeSparseMoeBlock for MoE layers
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
    
    def forward(self, hidden_states, attention_mask, position_ids, ...):
        # Pre-norm architecture (RMSNorm before each sub-layer)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, ...)
        hidden_states = residual + hidden_states  # Residual connection
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states  # Residual connection
        
        return hidden_states
```

**DeepSeek-V3 Variant** (Hybrid MoE):
```python
class DeepseekV3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        self.self_attn = DeepseekV3Attention(config, layer_idx)
        
        # CONDITIONAL: MoE after first_k_dense_replace layers
        if layer_idx >= config.first_k_dense_replace:
            self.mlp = DeepseekV3MoE(config)  # Expert layer
        else:
            self.mlp = DeepseekV3MLP(config)  # Dense layer
        
        self.input_layernorm = RMSNorm(...)
        self.post_attention_layernorm = RMSNorm(...)
    
    # forward() is IDENTICAL to Qwen3
```

### Model & CausalLM Wrappers

**All models follow this 2-layer pattern**:
```python
# LAYER 1: Base Model (embedding + layers + final norm)
class Qwen3Model(PreTrainedModel):
    def __init__(self, config):
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config, layer_idx) 
            for layer_idx in range(num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
    
    def forward(self, input_ids, attention_mask, ...):
        hidden_states = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, 
                                  position_embeddings=position_embeddings, ...)
        
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, ...)

# LAYER 2: CausalLM (adds lm_head projection)
class Qwen3ForCausalLM(PreTrainedModel, GenerationMixin):
    def __init__(self, config):
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, input_ids, labels, ...):
        outputs = self.model(input_ids, ...)
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)  # [batch, seq_len, vocab_size]
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, vocab_size), labels.view(-1))
        
        return CausalLMOutputWithPast(loss=loss, logits=logits, ...)
```

---

## Part 3: Proposed Abstraction Layers

### Layer 1: Configuration Hierarchy

```cpp
// Base configuration for all transformers
struct TransformerLayerConfig {
    int n_layers;
    int n_head;
    int n_head_kv;  // For GQA
    int head_dim;
    int d_model;
    int d_ff;
    int vocab_size;
    int max_seq_len;
    float eps;
    float rope_theta;
    std::string hidden_act;  // "silu", "gelu", etc.
};

// MoE-specific extension
struct MoEConfig {
    bool enabled = false;
    int num_experts = 0;
    int num_experts_per_tok = 0;
    int moe_intermediate_size = 0;
    bool norm_topk_prob = false;
    int first_k_dense_replace = 0;  // DeepSeek-V3 hybrid MoE
    
    // Qwen2-MoE specific
    bool shared_expert_intermediate_size = 0;
    bool has_shared_expert_gate = false;
};

// Attention-specific features
struct AttentionConfig {
    bool qk_norm = false;  // Qwen3, Qwen3-MoE
    bool sliding_window = false;  // Qwen3
    int window_size = 0;
    float attention_scaling = 1.0f;  // DeepSeek-V3
    
    // DeepSeek-V3 LoRA-style attention (advanced)
    bool q_lora = false;
    int q_lora_rank = 0;
    int kv_lora_rank = 0;
};

// Unified model config
struct ModelConfig {
    std::string architecture;  // "qwen2", "qwen3", "qwen3_moe", "deepseek_v3"
    TransformerLayerConfig layer_config;
    MoEConfig moe_config;
    AttentionConfig attention_config;
};
```

### Layer 2: Kernel Abstraction (Component Library)

```cpp
namespace llaminar {
namespace kernels {

// ============ UNIVERSAL COMPONENTS (No changes needed) ============

class RMSNormKernel : public MPIKernelBase {
    // ✓ Already generic - works for all models
    bool execute(...) override;
};

class RoPEKernel : public MPIKernelBase {
    // ✓ Already HuggingFace-compliant after recent refactor
    // Supports attention_scaling for DeepSeek-V3
    float attention_scaling = 1.0f;
    bool execute(...) override;
};

class LinearKernel : public MPIKernelBase {
    // ✓ Already generic - standard matrix multiply
    bool execute(...) override;
};

// ============ ATTENTION ABSTRACTION ============

// Base attention interface
class IAttentionKernel {
public:
    virtual ~IAttentionKernel() = default;
    virtual bool execute(const AttentionInputs& inputs, 
                        AttentionOutputs& outputs) = 0;
};

// Standard attention (Qwen2, Qwen3 without QK norm, DeepSeek-V3)
class StandardAttentionKernel : public IAttentionKernel, public MPIKernelBase {
public:
    bool execute(const AttentionInputs& inputs, 
                AttentionOutputs& outputs) override {
        // Standard: Q@K^T, softmax, @V flow
        // Uses MPILinearKernel for Q/K/V/O projections
    }
};

// Q/K normalized attention (Qwen3, Qwen3-MoE)
class QKNormAttentionKernel : public IAttentionKernel, public MPIKernelBase {
private:
    std::unique_ptr<RMSNormKernel> q_norm_;
    std::unique_ptr<RMSNormKernel> k_norm_;
    
public:
    bool execute(const AttentionInputs& inputs,
                AttentionOutputs& outputs) override {
        // Apply RMSNorm to Q and K projections before RoPE
        auto q_proj = linear(inputs.hidden, wq);
        auto k_proj = linear(inputs.hidden, wk);
        auto v_proj = linear(inputs.hidden, wv);
        
        q_proj = q_norm_->execute(q_proj);  // NEW step
        k_proj = k_norm_->execute(k_proj);  // NEW step
        
        apply_rope(q_proj, k_proj, ...);
        // ... rest is standard
    }
};

// Factory for attention variants
std::unique_ptr<IAttentionKernel> createAttentionKernel(
    const AttentionConfig& config, 
    const TransformerLayerConfig& layer_config
) {
    if (config.qk_norm) {
        return std::make_unique<QKNormAttentionKernel>(...);
    }
    return std::make_unique<StandardAttentionKernel>(...);
}

// ============ MLP ABSTRACTION ============

// Base MLP interface
class IMLPKernel {
public:
    virtual ~IMLPKernel() = default;
    virtual bool execute(const std::shared_ptr<TensorBase>& input,
                        std::shared_ptr<TensorBase>& output) = 0;
};

// Dense MLP (Qwen2, Qwen3, DeepSeek-V3 early layers)
class DenseMLPKernel : public IMLPKernel, public MPIKernelBase {
private:
    std::shared_ptr<TensorBase> w_gate_;
    std::shared_ptr<TensorBase> w_up_;
    std::shared_ptr<TensorBase> w_down_;
    std::string activation_;  // "silu", "gelu", etc.
    
public:
    bool execute(const std::shared_ptr<TensorBase>& input,
                std::shared_ptr<TensorBase>& output) override {
        // SwiGLU: down(act(gate) * up)
        auto gate_out = linear(input, w_gate_);
        auto up_out = linear(input, w_up_);
        auto gated = apply_activation(gate_out, activation_) * up_out;
        output = linear(gated, w_down_);
        return true;
    }
};

// Sparse MoE (Qwen3-MoE, DeepSeek-V3 later layers)
class SparseMoEKernel : public IMLPKernel, public MPIKernelBase {
private:
    std::shared_ptr<TensorBase> router_weight_;  // [hidden_size, num_experts]
    std::vector<DenseMLPKernel> experts_;
    int num_experts_per_tok_;
    bool norm_topk_prob_;
    
public:
    bool execute(const std::shared_ptr<TensorBase>& input,
                std::shared_ptr<TensorBase>& output) override {
        // 1. Router: softmax(input @ router_weight)
        auto router_logits = linear(input, router_weight_);
        auto routing_weights = softmax(router_logits, dim=-1);
        
        // 2. Top-K selection per token
        auto [top_k_weights, top_k_indices] = topk(routing_weights, num_experts_per_tok_);
        
        if (norm_topk_prob_) {
            top_k_weights = top_k_weights / sum(top_k_weights, dim=-1, keepdim=true);
        }
        
        // 3. Route tokens to selected experts
        output = zeros_like(input);
        for (int token_idx = 0; token_idx < batch_size * seq_len; ++token_idx) {
            for (int k = 0; k < num_experts_per_tok_; ++k) {
                int expert_idx = top_k_indices[token_idx][k];
                float weight = top_k_weights[token_idx][k];
                
                auto expert_out = experts_[expert_idx].execute(input[token_idx]);
                output[token_idx] += weight * expert_out;
            }
        }
        return true;
    }
};

// Shared-expert MoE (Qwen2-MoE)
class SharedExpertMoEKernel : public SparseMoEKernel {
private:
    DenseMLPKernel shared_expert_;
    std::shared_ptr<TensorBase> shared_expert_gate_;
    
public:
    bool execute(const std::shared_ptr<TensorBase>& input,
                std::shared_ptr<TensorBase>& output) override {
        // Sparse expert routing (base class)
        auto sparse_output = SparseMoEKernel::execute(input, output);
        
        // Shared expert (always active, gated)
        auto shared_output = shared_expert_.execute(input);
        auto gate = sigmoid(linear(input, shared_expert_gate_));
        shared_output = gate * shared_output;
        
        // Combine
        output = sparse_output + shared_output;
        return true;
    }
};

// Factory for MLP variants
std::unique_ptr<IMLPKernel> createMLPKernel(
    const MoEConfig& moe_config,
    const TransformerLayerConfig& layer_config,
    int layer_idx,
    const IModelWeights& weights
) {
    if (!moe_config.enabled) {
        return std::make_unique<DenseMLPKernel>(...);
    }
    
    // DeepSeek-V3 hybrid: dense layers first, then MoE
    if (moe_config.first_k_dense_replace > 0 && 
        layer_idx < moe_config.first_k_dense_replace) {
        return std::make_unique<DenseMLPKernel>(...);
    }
    
    // Qwen2-MoE has shared experts
    if (moe_config.has_shared_expert_gate) {
        return std::make_unique<SharedExpertMoEKernel>(...);
    }
    
    // Standard sparse MoE (Qwen3-MoE)
    return std::make_unique<SparseMoEKernel>(...);
}

}} // namespace llaminar::kernels
```

### Layer 3: DecoderLayer Composition

```cpp
namespace llaminar {

class TransformerDecoderLayer {
private:
    int layer_idx_;
    ModelConfig config_;
    
    // Components
    std::unique_ptr<kernels::RMSNormKernel> input_layernorm_;
    std::unique_ptr<kernels::IAttentionKernel> self_attn_;
    std::unique_ptr<kernels::RMSNormKernel> post_attention_layernorm_;
    std::unique_ptr<kernels::IMLPKernel> mlp_;
    
public:
    TransformerDecoderLayer(int layer_idx, const ModelConfig& config, 
                           const IModelWeights& weights)
        : layer_idx_(layer_idx), config_(config) {
        
        // Create normalization layers (always RMSNorm for our targets)
        input_layernorm_ = std::make_unique<kernels::RMSNormKernel>(
            config.layer_config.d_model, config.layer_config.eps
        );
        post_attention_layernorm_ = std::make_unique<kernels::RMSNormKernel>(
            config.layer_config.d_model, config.layer_config.eps
        );
        
        // Create attention (factory handles QK norm variant)
        self_attn_ = kernels::createAttentionKernel(
            config.attention_config, config.layer_config
        );
        
        // Create MLP (factory handles dense vs MoE)
        mlp_ = kernels::createMLPKernel(
            config.moe_config, config.layer_config, layer_idx, weights
        );
    }
    
    bool execute(const std::shared_ptr<TensorBase>& hidden_states,
                const AttentionMask& mask,
                const PositionEmbeddings& pos_emb,
                std::shared_ptr<TensorBase>& output) {
        // Pre-norm transformer architecture (HuggingFace standard)
        
        // Self-attention block
        auto residual = hidden_states;
        auto normed = input_layernorm_->execute(hidden_states);
        auto attn_out = self_attn_->execute(normed, mask, pos_emb);
        hidden_states = residual + attn_out;  // Residual connection
        
        // MLP block
        residual = hidden_states;
        normed = post_attention_layernorm_->execute(hidden_states);
        auto mlp_out = mlp_->execute(normed);
        output = residual + mlp_out;  // Residual connection
        
        return true;
    }
};

} // namespace llaminar
```

### Layer 4: Model-Specific Pipelines

```cpp
namespace llaminar {

// ============ BASE PIPELINE (shared across models) ============

class GenericTransformerPipeline : public AbstractPipeline {
protected:
    ModelConfig config_;
    std::unique_ptr<IModelWeights> weights_;
    
    // Model components
    std::unique_ptr<kernels::EmbeddingKernel> embed_tokens_;
    std::vector<std::unique_ptr<TransformerDecoderLayer>> layers_;
    std::unique_ptr<kernels::RMSNormKernel> final_norm_;
    std::unique_ptr<kernels::RoPEKernel> rotary_emb_;
    std::unique_ptr<kernels::LinearKernel> lm_head_;
    
    // KV cache (common to all models)
    std::vector<std::shared_ptr<TensorBase>> kv_cache_k_;
    std::vector<std::shared_ptr<TensorBase>> kv_cache_v_;
    
public:
    GenericTransformerPipeline(const ModelConfig& config)
        : config_(config) {
        
        // Create embedding layer
        embed_tokens_ = std::make_unique<kernels::EmbeddingKernel>(
            config.layer_config.vocab_size,
            config.layer_config.d_model
        );
        
        // Create decoder layers
        layers_.reserve(config.layer_config.n_layers);
        for (int i = 0; i < config.layer_config.n_layers; ++i) {
            layers_.push_back(
                std::make_unique<TransformerDecoderLayer>(i, config, *weights_)
            );
        }
        
        // Final normalization
        final_norm_ = std::make_unique<kernels::RMSNormKernel>(
            config.layer_config.d_model, config.layer_config.eps
        );
        
        // RoPE
        rotary_emb_ = std::make_unique<kernels::RoPEKernel>(
            config.layer_config.head_dim,
            config.layer_config.max_seq_len,
            config.layer_config.rope_theta,
            config.attention_config.attention_scaling
        );
        
        // LM head (vocabulary projection)
        lm_head_ = std::make_unique<kernels::LinearKernel>(
            config.layer_config.d_model,
            config.layer_config.vocab_size
        );
    }
    
    bool prefill(const std::vector<int>& tokens,
                const IModelWeights& weights,
                StageContext& ctx) override {
        // 1. Embed tokens
        auto hidden_states = embed_tokens_->execute(tokens);
        
        // 2. Compute position embeddings (RoPE)
        auto position_embeddings = rotary_emb_->execute(
            tokens.size(), ctx.kv_used  // n_past for incremental
        );
        
        // 3. Run through decoder layers
        for (auto& layer : layers_) {
            layer->execute(hidden_states, attention_mask, 
                          position_embeddings, hidden_states);
        }
        
        // 4. Final normalization
        hidden_states = final_norm_->execute(hidden_states);
        
        // 5. Project to vocabulary
        last_logits_ = lm_head_->execute(hidden_states);
        
        ctx.kv_used += tokens.size();
        return true;
    }
    
    bool decode(int next_token, const IModelWeights& weights,
               StageContext& ctx) override {
        // Same as prefill, but with single token (seq_len=1)
        return prefill({next_token}, weights, ctx);
    }
    
    bool logits(std::shared_ptr<TensorBase>& out_logits) override {
        out_logits = last_logits_;
        return true;
    }
    
    std::string name() const override { 
        return "GenericTransformerPipeline(" + config_.architecture + ")"; 
    }
};

// ============ MODEL-SPECIFIC ADAPTERS (thin wrappers) ============

class Qwen2Pipeline : public GenericTransformerPipeline {
public:
    Qwen2Pipeline(const ModelConfig& config) 
        : GenericTransformerPipeline(config) {}
    
    std::unique_ptr<IModelWeights> loadWeights(const std::string& path) override {
        // Qwen2-specific weight loading and validation
        auto weights = loadQwen2Weights(path);
        validateQwen2Weights(weights, config_);
        return weights;
    }
    
    std::string name() const override { return "Qwen2Pipeline"; }
};

class Qwen3Pipeline : public GenericTransformerPipeline {
public:
    Qwen3Pipeline(const ModelConfig& config) 
        : GenericTransformerPipeline(config) {
        // Config should have attention_config.qk_norm=true
        assert(config.attention_config.qk_norm && 
               "Qwen3 requires Q/K normalization");
    }
    
    std::unique_ptr<IModelWeights> loadWeights(const std::string& path) override {
        auto weights = loadQwen3Weights(path);
        validateQwen3Weights(weights, config_);
        return weights;
    }
    
    std::string name() const override { return "Qwen3Pipeline"; }
};

class Qwen3MoEPipeline : public GenericTransformerPipeline {
public:
    Qwen3MoEPipeline(const ModelConfig& config) 
        : GenericTransformerPipeline(config) {
        assert(config.moe_config.enabled && "Qwen3-MoE requires MoE config");
        assert(config.attention_config.qk_norm && "Qwen3-MoE uses QK norm");
    }
    
    std::unique_ptr<IModelWeights> loadWeights(const std::string& path) override {
        auto weights = loadQwen3MoEWeights(path);  // Loads expert weights
        validateQwen3MoEWeights(weights, config_);
        return weights;
    }
    
    std::string name() const override { return "Qwen3MoEPipeline"; }
};

class DeepSeekV3Pipeline : public GenericTransformerPipeline {
public:
    DeepSeekV3Pipeline(const ModelConfig& config) 
        : GenericTransformerPipeline(config) {
        assert(config.moe_config.enabled && "DeepSeek-V3 uses MoE");
        assert(config.moe_config.first_k_dense_replace > 0 && 
               "DeepSeek-V3 has hybrid dense+MoE layers");
    }
    
    std::unique_ptr<IModelWeights> loadWeights(const std::string& path) override {
        auto weights = loadDeepSeekV3Weights(path);
        validateDeepSeekV3Weights(weights, config_);
        return weights;
    }
    
    std::string name() const override { return "DeepSeekV3Pipeline"; }
};

} // namespace llaminar
```

### Layer 5: Pipeline Factory Registration

```cpp
// src/pipeline_factory.cpp

void registerAllPipelines() {
    // Qwen2 (current production)
    PipelineFactory::instance().registerCreator("qwen2", [](const ModelConfig& cfg) {
        return std::make_unique<Qwen2Pipeline>(cfg);
    });
    
    // Qwen3 (adds QK normalization)
    PipelineFactory::instance().registerCreator("qwen3", [](const ModelConfig& cfg) {
        auto config = cfg;
        config.attention_config.qk_norm = true;  // Force QK norm
        return std::make_unique<Qwen3Pipeline>(config);
    });
    
    // Qwen3-MoE (QK norm + sparse MoE)
    PipelineFactory::instance().registerCreator("qwen3_moe", [](const ModelConfig& cfg) {
        auto config = cfg;
        config.attention_config.qk_norm = true;
        config.moe_config.enabled = true;
        // MoE params set from GGUF metadata
        return std::make_unique<Qwen3MoEPipeline>(config);
    });
    
    // DeepSeek-V3 (hybrid dense+MoE with attention scaling)
    PipelineFactory::instance().registerCreator("deepseek_v3", [](const ModelConfig& cfg) {
        auto config = cfg;
        config.moe_config.enabled = true;
        config.moe_config.first_k_dense_replace = cfg.moe_config.first_k_dense_replace;
        config.attention_config.attention_scaling = cfg.attention_config.attention_scaling;
        return std::make_unique<DeepSeekV3Pipeline>(config);
    });
}
```

---

## Part 4: Translation Pipeline (Python → C++)

### Step 1: Configuration Extraction

```python
# scripts/extract_model_config.py
import torch
from transformers import AutoConfig

def extract_llaminar_config(huggingface_model_name: str) -> dict:
    """Extract Llaminar ModelConfig from HuggingFace model."""
    config = AutoConfig.from_pretrained(huggingface_model_name)
    
    llaminar_config = {
        "architecture": config.model_type,  # "qwen3_moe", etc.
        "layer_config": {
            "n_layers": config.num_hidden_layers,
            "n_head": config.num_attention_heads,
            "n_head_kv": config.num_key_value_heads,
            "head_dim": getattr(config, "head_dim", 
                               config.hidden_size // config.num_attention_heads),
            "d_model": config.hidden_size,
            "d_ff": config.intermediate_size,
            "vocab_size": config.vocab_size,
            "max_seq_len": config.max_position_embeddings,
            "eps": config.rms_norm_eps,
            "rope_theta": config.rope_theta,
            "hidden_act": config.hidden_act,
        },
        "moe_config": {
            "enabled": hasattr(config, "num_experts"),
            "num_experts": getattr(config, "num_experts", 0),
            "num_experts_per_tok": getattr(config, "num_experts_per_tok", 0),
            "moe_intermediate_size": getattr(config, "moe_intermediate_size", 0),
            "norm_topk_prob": getattr(config, "norm_topk_prob", False),
            "first_k_dense_replace": getattr(config, "first_k_dense_replace", 0),
        },
        "attention_config": {
            "qk_norm": hasattr(config, "q_norm"),  # Qwen3 feature
            "sliding_window": getattr(config, "sliding_window", 0) > 0,
            "window_size": getattr(config, "sliding_window", 0),
            "attention_scaling": getattr(config, "attention_scaling", 1.0),
        }
    }
    
    return llaminar_config

# Usage:
# python scripts/extract_model_config.py Qwen/Qwen3-8B-Instruct > configs/qwen3_8b.json
```

### Step 2: Reference Implementation Capture

```python
# scripts/capture_huggingface_ops.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def capture_forward_pass(model_name: str, prompt: str = "Hello"):
    """Capture intermediate tensors from HuggingFace forward pass for validation."""
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Hook to capture intermediate outputs
    captures = {}
    
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captures[name] = output[0].detach().cpu().numpy()
            else:
                captures[name] = output.detach().cpu().numpy()
        return hook
    
    # Register hooks on all layers
    for name, module in model.named_modules():
        if any(x in name for x in ["self_attn", "mlp", "input_layernorm"]):
            module.register_forward_hook(make_hook(name))
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Save captures as numpy arrays for parity testing
    np.savez(f"parity_data/{model_name.replace('/', '_')}_forward.npz", **captures)
    
    return captures

# Usage:
# python scripts/capture_huggingface_ops.py Qwen/Qwen3-8B-Instruct
# Generates: parity_data/Qwen_Qwen3-8B-Instruct_forward.npz
```

### Step 3: Parity Validation Framework Extension

```cpp
// tests/test_huggingface_parity.cpp

class HuggingFaceParityTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Load captured HuggingFace tensors
        hf_captures_ = loadNumpyArchive("parity_data/Qwen_Qwen3-8B-Instruct_forward.npz");
    }
    
    void validateLayerOutput(int layer_idx, const std::shared_ptr<TensorBase>& llaminar_output) {
        std::string key = "model.layers." + std::to_string(layer_idx) + ".output";
        auto hf_output = hf_captures_[key];
        
        float max_abs_error = computeMaxAbsError(llaminar_output, hf_output);
        float relative_l2 = computeRelativeL2(llaminar_output, hf_output);
        
        EXPECT_LT(max_abs_error, 1e-3) << "Layer " << layer_idx << " max error too high";
        EXPECT_LT(relative_l2, 1e-5) << "Layer " << layer_idx << " relative error too high";
    }
    
    std::map<std::string, std::shared_ptr<TensorBase>> hf_captures_;
};

TEST_F(HuggingFaceParityTest, Qwen3AttentionLayer) {
    // Create Qwen3 attention kernel with QK norm
    ModelConfig config = loadQwen3Config();
    auto attn = kernels::createAttentionKernel(config.attention_config, config.layer_config);
    
    // Run forward pass
    auto hidden_states = loadTensor("layer_0_input");
    auto output = attn->execute(hidden_states, ...);
    
    // Validate against HuggingFace
    validateLayerOutput(0, output);
}

TEST_F(HuggingFaceParityTest, Qwen3MoELayer) {
    ModelConfig config = loadQwen3MoEConfig();
    auto moe = kernels::createMLPKernel(config.moe_config, config.layer_config, 0, *weights);
    
    auto hidden_states = loadTensor("layer_0_post_attn");
    auto output = moe->execute(hidden_states);
    
    validateLayerOutput(0, output);
}
```

### Step 4: Automated Code Generation (Future)

```python
# scripts/generate_kernel_wrapper.py (FUTURE PHASE 3)

def generate_cpp_kernel_from_pytorch(pytorch_module_path: str):
    """Auto-generate C++ kernel from PyTorch module definition."""
    # Parse modeling_qwen3.py to extract Qwen3Attention class
    # Generate equivalent MPIAttentionKernel variant
    # This is aspirational - manual translation is Phase 1
    pass
```

---

## Part 5: Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

**Deliverables**:
1. ✅ **Extend ModelConfig** with MoE and Attention feature flags
2. ✅ **Create kernel interfaces**: `IAttentionKernel`, `IMLPKernel`
3. ✅ **Implement QKNormAttentionKernel** (Qwen3 variant)
4. ✅ **Implement DenseMLPKernel** (refactor existing MPIMLPKernel)
5. ✅ **Create TransformerDecoderLayer** (composition pattern)
6. ✅ **Parity test**: Qwen2 with new architecture (should be identical)

**Success Criteria**:
- Qwen2 parity unchanged (all existing tests pass)
- New architecture runs Qwen2 with 100% parity to old code
- Clear separation: config → kernel factory → execution

**Estimated Effort**: 5-7 days

---

### Phase 2: Qwen3 Support (Weeks 3-4)

**Deliverables**:
1. ✅ **QK Normalization in attention** (already implemented above)
2. ✅ **Qwen3 config extraction** from GGUF metadata
3. ✅ **Qwen3 weight loader** with QK norm weights
4. ✅ **Qwen3Pipeline adapter** using GenericTransformerPipeline
5. ✅ **HuggingFace parity tests** for Qwen3-8B

**Success Criteria**:
- Qwen3-8B-Instruct runs with HuggingFace parity
- QK norm validated: `||Q||=1` and `||K||=1` after normalization
- Pipeline factory selects correct variant based on architecture string

**Estimated Effort**: 7-10 days

---

### Phase 3: MoE Support (Weeks 5-7)

**Deliverables**:
1. ✅ **SparseMoEKernel implementation**:
   - Router (softmax over experts)
   - Top-K selection per token
   - Expert execution and weighted aggregation
2. ✅ **Qwen3-MoE support** (64 experts, top-8 routing)
3. ✅ **Expert weight loading** from GGUF
4. ✅ **MoE-specific parity tests**
5. ⚠️ **Performance optimization**: Expert batching, memory efficiency

**Success Criteria**:
- Qwen3-MoE-8B runs with HuggingFace parity
- Router logits match PyTorch (before top-k)
- Expert outputs match PyTorch individually
- Aggregated output matches PyTorch

**Estimated Effort**: 14-21 days (MoE is complex)

---

### Phase 4: DeepSeek-V3 & Advanced Features (Weeks 8-10)

**Deliverables**:
1. ✅ **Hybrid MoE** (first_k_dense_replace support)
2. ✅ **Attention scaling** in RoPE
3. ✅ **DeepSeekV3 config & weights**
4. ⚠️ **LoRA-style attention** (optional - DeepSeek-V3 advanced feature)
5. ✅ **Shared-expert MoE** (Qwen2-MoE variant)

**Success Criteria**:
- DeepSeek-V3 runs with HuggingFace parity
- Hybrid layers work (early dense, late MoE)
- Attention scaling validated

**Estimated Effort**: 14-21 days

---

### Phase 5: Generalization & Documentation (Week 11-12)

**Deliverables**:
1. **Model porting guide**: "How to add a new model to Llaminar"
2. **Kernel library documentation**: When to use each kernel variant
3. **Configuration schema**: JSON format for model configs
4. **Automated config extraction**: Script to generate Llaminar config from HuggingFace
5. **CI/CD integration**: Parity tests for all supported models

**Success Criteria**:
- New developer can add Mistral-7B in < 1 day using guide
- All configs have JSON schema validation
- Parity tests run automatically on model updates

**Estimated Effort**: 7-10 days

---

## Part 6: File Structure After Refactor

```
llaminar/
├── src/
│   ├── abstract_pipeline.{h,cpp}              # ✓ Already exists
│   ├── pipeline_factory.{h,cpp}               # ✓ Already exists
│   ├── transformer_decoder_layer.{h,cpp}      # NEW: Generic layer composition
│   ├── model_config.{h,cpp}                   # EXTENDED: MoE + Attention configs
│   │
│   ├── kernels/
│   │   ├── interfaces/
│   │   │   ├── IAttentionKernel.h             # NEW: Attention interface
│   │   │   └── IMLPKernel.h                   # NEW: MLP interface
│   │   │
│   │   ├── attention/
│   │   │   ├── StandardAttentionKernel.{h,cpp}
│   │   │   ├── QKNormAttentionKernel.{h,cpp}  # NEW: Qwen3 variant
│   │   │   └── AttentionKernelFactory.{h,cpp}
│   │   │
│   │   ├── mlp/
│   │   │   ├── DenseMLPKernel.{h,cpp}         # REFACTORED from MPIMLPKernel
│   │   │   ├── SparseMoEKernel.{h,cpp}        # NEW: MoE support
│   │   │   ├── SharedExpertMoEKernel.{h,cpp}  # NEW: Qwen2-MoE variant
│   │   │   └── MLPKernelFactory.{h,cpp}
│   │   │
│   │   └── common/                            # ✓ Already generic
│   │       ├── MPILinearKernel.{h,cpp}
│   │       ├── MPIRMSNormKernel.{h,cpp}
│   │       ├── MPIRoPEKernel.{h,cpp}
│   │       └── attention_primitives.{h,cpp}
│   │
│   ├── pipelines/
│   │   ├── GenericTransformerPipeline.{h,cpp} # NEW: Base implementation
│   │   ├── Qwen2Pipeline.{h,cpp}              # REFACTORED: Thin adapter
│   │   ├── Qwen3Pipeline.{h,cpp}              # NEW
│   │   ├── Qwen3MoEPipeline.{h,cpp}           # NEW
│   │   └── DeepSeekV3Pipeline.{h,cpp}         # NEW
│   │
│   └── weights/
│       ├── IModelWeights.h                    # ✓ Already exists
│       ├── Qwen2Weights.{h,cpp}
│       ├── Qwen3Weights.{h,cpp}               # NEW: Includes QK norm weights
│       ├── Qwen3MoEWeights.{h,cpp}            # NEW: Includes expert weights
│       └── DeepSeekV3Weights.{h,cpp}          # NEW
│
├── tests/
│   ├── test_attention_variants.cpp            # NEW: QK norm, standard, etc.
│   ├── test_moe_kernels.cpp                   # NEW: Sparse MoE, routing, experts
│   ├── test_decoder_layer.cpp                 # NEW: Layer composition
│   ├── test_qwen3_parity.cpp                  # NEW: HuggingFace parity
│   ├── test_qwen3_moe_parity.cpp              # NEW
│   ├── test_deepseek_v3_parity.cpp            # NEW
│   └── test_pipeline_factory.cpp              # ✓ Extend for new models
│
├── scripts/
│   ├── extract_model_config.py                # NEW: HF → Llaminar config
│   ├── capture_huggingface_ops.py             # NEW: Parity data generation
│   └── validate_config_schema.py              # NEW: JSON schema validation
│
├── configs/                                    # NEW: JSON model configs
│   ├── qwen2_7b.json
│   ├── qwen3_8b.json
│   ├── qwen3_moe_8b.json
│   └── deepseek_v3_671b.json
│
└── docs/
    ├── MODEL_ABSTRACTION_STRATEGY.md          # THIS FILE
    ├── KERNEL_LIBRARY_GUIDE.md                # NEW: When to use each kernel
    ├── PORTING_NEW_MODELS.md                  # NEW: Step-by-step guide
    └── CONFIGURATION_SCHEMA.md                # NEW: ModelConfig reference
```

---

## Part 7: Risk Mitigation & Validation Strategy

### Risk 1: Parity Regression During Refactor

**Mitigation**:
1. **Dual path during Phase 1**: Keep old Qwen2 code alongside new architecture
2. **Continuous parity testing**: Run old vs new on every commit
3. **Incremental migration**: Move one kernel at a time
4. **Revert strategy**: Keep old code until 100% parity achieved

### Risk 2: MoE Performance Issues

**Mitigation**:
1. **Baseline with dense MLP**: Ensure dense case is fast first
2. **Expert batching**: Process multiple tokens per expert in one pass
3. **Memory pooling**: Reuse expert buffers across tokens
4. **Profiling from day 1**: Instrument MoE kernel before optimization

### Risk 3: Configuration Explosion

**Mitigation**:
1. **JSON schema validation**: Catch invalid configs early
2. **Config inheritance**: Base configs + model-specific overrides
3. **Automated extraction**: Don't hand-write configs, generate from HF
4. **Version configs**: Track config changes with models

### Risk 4: Weight Loading Complexity

**Mitigation**:
1. **Weight contracts**: Extend existing validation system to MoE
2. **Clear error messages**: "Expected expert_0.gate_proj, found X"
3. **Conversion tools**: Scripts to convert PyTorch → GGUF with correct layout
4. **Test fixtures**: Small synthetic models for unit testing

---

## Part 8: Success Metrics

| Metric | Target | Validation Method |
|--------|--------|-------------------|
| **Qwen2 Parity Unchanged** | 100% | Existing parity tests |
| **Qwen3 HF Parity** | > 99.9% accuracy | Capture → compare 100 layers |
| **Qwen3-MoE HF Parity** | > 99.9% accuracy | Expert-level validation |
| **DeepSeek-V3 HF Parity** | > 99.9% accuracy | Hybrid layer validation |
| **Code Reuse** | > 85% | Lines of shared code / total |
| **Performance** | No regression | Qwen2 latency ± 5% |
| **Time to add new model** | < 2 days | Measured for 5th model |

---

## Conclusion

This strategy provides a **clear, incremental path** from Qwen2-specific to multi-architecture support while:

✅ **Preserving existing functionality** (Qwen2 parity maintained)  
✅ **Enabling rapid model addition** (factory pattern + configuration)  
✅ **Maintaining HuggingFace faithfulness** (parity framework + PyTorch comparison)  
✅ **Supporting advanced features** (MoE, QK norm, sliding windows)  
✅ **Minimizing code duplication** (shared kernels + composition)

**Next Steps**:
1. Review this document with team
2. Create Phase 1 implementation plan (kernel interfaces + config extension)
3. Set up parity data capture for Qwen3-8B-Instruct
4. Begin implementation with TransformerDecoderLayer

The architecture is **extensible by design** - adding Mistral, Gemma, LLaMA-3, or future models will follow the same pattern established with Qwen3 and DeepSeek-V3.
