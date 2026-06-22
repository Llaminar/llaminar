# Qwen 3.5 MoE + Gated DeltaNet — Llaminar Build Plan

**Date**: April 2, 2026
**Author**: David Sanftenberg
**Status**: Sketch / RFC
**Prerequisite**: [MoE Expert Placement Design](MOE_EXPERT_PLACEMENT_DESIGN.md)
**Research**: [Qwen 3.5 Architecture Research](../2026-03/QWEN3_5_RESEARCH.md)
**GDN Kernel Plan**: [Qwen3-Next GDN Project Plan](../2026-03/QWEN3NEXT_GDN_PROJECT_PLAN.md)

---

## 1. Target Models

| Model | Params | Active | Layers | Attention | FFN | Priority |
|-------|--------|--------|--------|-----------|-----|----------|
| **Qwen3.5-4B** | 4B | 4B | 32 | Hybrid GDN+FA | Dense SwiGLU | P0 (first) |
| **Qwen3.5-35B-A3B** | 35B | 3B | 40 | Hybrid GDN+FA | MoE 256×512 + shared | P1 |
| **Qwen3.5-122B-A10B** | 122B | 10B | 48 | Hybrid GDN+FA | MoE 256×1024 + shared | P2 |

All three share the same hybrid attention architecture (75% GDN, 25% full attention) — they differ only in FFN (dense vs MoE) and scale.

---

## 2. What's New vs Qwen2/3

### Novel components (not in Llaminar today)

| Component | Description | Affects |
|-----------|-------------|---------|
| **GDN recurrence** | `S_t = diag(exp(g)) · S_{t-1} + β · k · (v − S^T k)^T` — replaces softmax attention | 75% of layers |
| **GDN state tensor** | Fixed-size `(n_heads, d_k, d_v)` per layer — replaces KV cache for GDN layers | Memory management |
| **Short conv1d** | Causal depthwise conv (kernel=4) on concatenated QKV before recurrence | GDN layers |
| **Gated RMSNorm** | `RMSNorm(x) ⊙ SiLU(z)` where z comes from a separate projection | GDN layers |
| **Attention output gate** | `sigmoid(gate) ⊙ attn_output` on BOTH GDN and full attention layers | All layers |
| **Partial RoPE** | RoPE applied to only first 25% of head dimensions (`partial_rotary_factor: 0.25`) | FA layers |
| **Per-layer-type dispatch** | Layers are either GDN (type 0) or full attention (type 1) — 3:1 pattern | Graph builder |
| **Dual cache system** | KV cache for FA layers; GDN recurrent state for GDN layers — managed per-layer | Cache manager |
| **Variable head dimensions** | FA: `head_dim=256`, GDN: `key_head_dim=128, value_head_dim=128` | Buffers, kernels |
| **Variable KV head counts** | FA: `n_kv_heads=2-4` (GQA 4-16:1), GDN: `key_heads=16, value_heads=32-64` | Kernels |
| **MoE SparseMoeBlock** | 256 experts (top-8) + shared expert + sigmoid gating — all layers in MoE variants | FFN path |
| **3D expert weight tensors** | `gate_up_proj: [256, 2×intermediate, hidden]` — batched expert weights | Tensor system, GGUF loader |
| **Tied/separate embeddings** | Dense: `tie_word_embeddings=true`, MoE: `false` | Weight loading |

### Reusable components (already in Llaminar)

| Component | Status | Notes |
|-----------|--------|-------|
| Dense SwiGLU FFN | ✅ Works as-is | Used by dense variant and shared expert |
| Standard GQA attention | ✅ Works as-is | Used by 25% of layers (full attention) |
| RMSNorm | ✅ Works as-is | Input and post-attention norms |
| RoPE (1D) | ⚠️ Needs partial support | Add `partial_rotary_factor` parameter |
| Embedding stage | ✅ Works as-is | |
| Residual add | ✅ Works as-is | |
| MoE stubs | ⚠️ Need completion | Router, Expert, Combine stages exist as stubs |
| Weight streaming | ✅ Works as-is | `LayerWeightStreamer` — extends to expert-level |
| TP sharding | ✅ Works as-is | Column/Row/Input parallel |
| Named domains | ✅ Works as-is | Expert domain placement |

---

## 3. Graph System Design: Heterogeneous Layer Templates

### 3.1 The Core Problem

Qwen2/3 have a single `LayerTemplate` applied uniformly to all layers. Qwen 3.5 has **two layer types** (GDN and full attention) with **different stage sequences, buffers, and dimensions** — plus the MoE variants swap the dense FFN for a MoE block.

This requires the graph system to support:
1. **Multiple layer templates** — selected per-layer based on `layer_types[]`
2. **Optional FFN variant** — dense SwiGLU or MoE, selectable per-model
3. **Per-template buffer specs** — GDN and FA have different scratch buffer sizes

### 3.2 GraphSchema Extensions

```cpp
// Extension to GraphSchema in GraphSchema.h

struct GraphSchema {
    // ... existing fields ...

    // Single layer template (existing — used by Qwen2, Qwen3, Llama)
    LayerTemplate layer_template;

    // NEW: Named layer templates for hybrid architectures
    // Key = template name (e.g., "gdn", "full_attention")
    // If empty, all layers use layer_template (backward compatible)
    std::unordered_map<std::string, LayerTemplate> named_templates;

    // NEW: Per-layer template assignment
    // Index = layer index, value = template name from named_templates
    // If empty, all layers use layer_template
    std::vector<std::string> layer_template_names;

    // NEW: Optional MoE FFN template (replaces ffn_stages when present)
    // Applies to layers where the model config says "use MoE"
    std::optional<MoELayerTemplate> moe_ffn_template;
};
```

### 3.3 Layer Template Composition

For Qwen 3.5, the graph schema defines **three composable blocks**:

```
┌──────────────────────────────────────────────────────────┐
│ GDN Attention Template ("gdn")                           │
│                                                          │
│  attn_norm → in_proj_qkv → in_proj_z → in_proj_a →      │
│  in_proj_b → conv1d → RoPE(partial) → GDN_recurrence →  │
│  gated_rmsnorm → output_gate → Wo → residual_add        │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ Full Attention Template ("full_attention")                │
│                                                          │
│  attn_norm → fused_qkv → qk_norm → RoPE(partial) →      │
│  KV_cache_append → flash_attention → output_gate →       │
│  Wo → residual_add                                       │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ Dense FFN (shared by both, used for dense model)         │
│                                                          │
│  ffn_norm → gate_up_proj → SwiGLU → down_proj →          │
│  residual_add                                            │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ MoE FFN (replaces Dense FFN for MoE variants)            │
│                                                          │
│  ffn_norm → ┬─ router → dispatch → experts → combine ─┐ │
│             └─ shared_expert → shared_gate ────────────┘ │
│  → add(expert_out, shared_out) → residual_add           │
└──────────────────────────────────────────────────────────┘
```

A complete layer is: `attention_template[layer_type] + ffn_template[dense_or_moe]`

### 3.4 Layer Type Assignment

```cpp
// In GraphResolver::resolve(), for Qwen 3.5:

for (int layer = 0; layer < n_layers; ++layer) {
    // Determine attention type from config
    bool is_full_attn = ((layer + 1) % full_attention_interval == 0);
    const std::string& template_name = is_full_attn ? "full_attention" : "gdn";

    // Get the right template
    const LayerTemplate& attn_template = schema.named_templates.at(template_name);

    // Emit attention stages from this template
    for (const auto& stage : attn_template.attention_stages) {
        emitResolvedStage(stage, layer, tensor_context);
    }

    // Emit FFN stages (dense or MoE based on model variant)
    if (schema.moe_ffn_template.has_value()) {
        emitMoEFFN(*schema.moe_ffn_template, layer, tensor_context, placement_map);
    } else {
        for (const auto& stage : attn_template.ffn_stages) {
            emitResolvedStage(stage, layer, tensor_context);
        }
    }
}
```

---

## 4. GraphConfig Extensions

```cpp
struct GraphConfig {
    // ... existing Qwen2/3 fields ...

    // =====================================================================
    // Qwen 3.5: Hybrid attention configuration
    // =====================================================================

    /// Full attention interval (every Nth layer is full attention, rest are GDN)
    /// 0 = all layers use same type (backward compatible)
    int full_attention_interval = 0;

    /// Layer types: "linear_attention" or "full_attention" per layer
    /// Empty = all same type (backward compatible with Qwen2/3)
    std::vector<std::string> layer_types;

    // =====================================================================
    // Full Attention layer dimensions
    // =====================================================================

    /// head_dim for full attention layers (default: same as head_dim)
    int fa_head_dim = 0;    // 0 = use head_dim

    /// KV heads for full attention layers (default: same as n_kv_heads)
    int fa_n_kv_heads = 0;  // 0 = use n_kv_heads

    /// Partial RoPE factor (1.0 = full RoPE, 0.25 = first 25% of dims)
    float partial_rotary_factor = 1.0f;

    /// Enable attention output gate (sigmoid gating on Wo input)
    bool attn_output_gate = false;

    /// Sliding window size for full attention layers (-1 = no window)
    int sliding_window = -1;

    // =====================================================================
    // GDN layer dimensions
    // =====================================================================

    int gdn_key_head_dim = 0;     ///< GDN key dimension per head
    int gdn_value_head_dim = 0;   ///< GDN value dimension per head
    int gdn_n_key_heads = 0;      ///< Number of GDN key heads
    int gdn_n_value_heads = 0;    ///< Number of GDN value heads
    int gdn_conv_kernel = 4;      ///< Short conv1d kernel size

    // =====================================================================
    // MoE configuration (zero = no MoE)
    // =====================================================================

    int moe_num_experts = 0;                ///< Total expert count (256 for Qwen3.5)
    int moe_top_k = 0;                      ///< Experts activated per token (8)
    int moe_intermediate_size = 0;          ///< Per-expert FFN intermediate dim
    int moe_shared_intermediate_size = 0;   ///< Shared expert intermediate dim
    bool moe_has_shared_expert = false;     ///< Has always-active shared expert
    bool moe_norm_topk_prob = false;        ///< Normalize top-k routing weights

    // =====================================================================
    // Embedding
    // =====================================================================

    bool tie_word_embeddings = false;       ///< LM head aliases embedding table

    // =====================================================================
    // Helper queries
    // =====================================================================

    bool isMoE() const { return moe_num_experts > 0; }
    bool isHybridAttention() const { return full_attention_interval > 0; }

    bool isGDNLayer(int layer_idx) const {
        if (full_attention_interval <= 0) return false;
        return ((layer_idx + 1) % full_attention_interval) != 0;
    }

    bool isFullAttentionLayer(int layer_idx) const {
        if (full_attention_interval <= 0) return true;  // All FA if no hybrid
        return ((layer_idx + 1) % full_attention_interval) == 0;
    }

    /// Get head_dim for a specific layer
    int headDimForLayer(int layer_idx) const {
        if (isGDNLayer(layer_idx)) return gdn_key_head_dim > 0 ? gdn_key_head_dim : head_dim;
        return fa_head_dim > 0 ? fa_head_dim : head_dim;
    }

    /// Get n_kv_heads for a specific layer
    int kvHeadsForLayer(int layer_idx) const {
        if (isGDNLayer(layer_idx)) return gdn_n_key_heads > 0 ? gdn_n_key_heads : n_kv_heads;
        return fa_n_kv_heads > 0 ? fa_n_kv_heads : n_kv_heads;
    }
};
```

---

## 5. New Stage Types

### 5.1 GDN-Specific Stages

| Stage | Type | Description |
|-------|------|-------------|
| `GDNProjectionStage` | `StageType::GDNProjection` | 4 separate GEMMs: `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b` |
| `ShortConv1dStage` | `StageType::ShortConv1d` | Causal depthwise conv (kernel=4) on QKV, plus SiLU activation |
| `GDNRecurrenceStage` | `StageType::GDNRecurrence` | Delta rule recurrence (chunk-parallel prefill, single-step decode) |
| `GatedRMSNormStage` | `StageType::GatedRMSNorm` | `RMSNorm(x) ⊙ SiLU(z)` — extends existing RMSNorm |
| `AttentionOutputGateStage` | `StageType::AttentionOutputGate` | `sigmoid(gate) * attn_output` — for both GDN and FA layers |

### 5.2 MoE Stages (from MoE design doc)

| Stage | Type | Description |
|-------|------|-------------|
| `MoERouterStage` | `StageType::MoERouter` | Exists — complete: softmax + top-k selection |
| `MoEDispatchStage` | `StageType::MoEDispatch` | New — scatter tokens to expert devices |
| `MoEExpertStage` | `StageType::MoEFFN` | Exists — complete: per-expert SwiGLU FFN |
| `MoECombineStage` | `StageType::MoECombine` | Exists — complete: weighted sum of expert outputs |
| `MoESharedExpertStage` | `StageType::MoESharedExpert` | New — dense SwiGLU + sigmoid gating (reuses FFN stages) |

### 5.3 Modified Existing Stages

| Stage | Modification |
|-------|-------------|
| `RoPEStage` | Add `partial_rotary_factor` — apply RoPE to first N dims, pass through rest |
| `FusedQKVGEMMStage` | Support separate `gate` output for attention output gating |
| `RMSNormStage` | Support `subtract_one` mode for Qwen3.5's `(1 + weight) * norm(x)` convention |

---

## 6. New Kernel Interfaces

### 6.1 ITensorGatedDeltaNet

```cpp
// src/v2/tensors/TensorKernels.h

class ITensorGatedDeltaNet {
public:
    virtual ~ITensorGatedDeltaNet() = default;

    /// Chunk-parallel prefill: process full sequence in chunks
    /// Input: Q, K, V [seq_len, dim], gates α/β/g [seq_len, n_heads]
    /// Output: context [seq_len, n_heads * d_v], updated state S
    virtual bool chunk_forward(
        const float* Q, const float* K, const float* V,
        const float* alpha, const float* beta, const float* gate,
        float* output, float* state,
        int seq_len, int n_heads, int d_k, int d_v,
        int chunk_size = 64) = 0;

    /// Single-step decode recurrence
    /// Input: q, k, v [1, dim], gates α/β/g [1, n_heads], state S
    /// Output: context [1, n_heads * d_v], updated state S
    virtual bool recurrent_step(
        const float* q, const float* k, const float* v,
        const float* alpha, const float* beta, const float* gate,
        float* output, float* state,
        int n_heads, int d_k, int d_v) = 0;
};
```

### 6.2 ITensorShortConvolution

```cpp
class ITensorShortConvolution {
public:
    virtual ~ITensorShortConvolution() = default;

    /// Apply causal depthwise conv1d + SiLU activation
    /// Input: x [seq_len, channels], conv_weight [channels, 1, kernel_size]
    /// State: conv_state [channels, kernel_size - 1] (for incremental decode)
    /// Output: y [seq_len, channels]
    virtual bool forward(
        const float* input, const float* weight,
        float* output, float* conv_state,
        int seq_len, int channels, int kernel_size,
        bool apply_silu = true) = 0;
};
```

---

## 7. GDN State Management

### 7.1 Dual Cache System

Qwen 3.5 layers need two different kinds of state:

| Layer Type | State | Shape per layer | Growth |
|------------|-------|-----------------|--------|
| GDN (75%) | Recurrent state `S` | `(n_heads, d_k, d_v)` = `(16, 128, 128)` = 256 KB/layer FP32 | **Fixed** |
| GDN (75%) | Conv state | `(ssm_inner_size, kernel-1)` = `(2048, 3)` = 24 KB/layer FP32 | **Fixed** |
| FA (25%) | KV cache | `(max_seq_len, n_kv_heads, head_dim)` per K and V | **Linear in seq_len** |

**Key insight**: GDN uses ~280 KB fixed memory per layer regardless of sequence length. FA uses growing KV cache. A 40-layer MoE model has 30 GDN layers (8.4 MB fixed state) and 10 FA layers (KV cache).

### 7.2 HybridCacheManager

```cpp
// src/v2/execution/cache/HybridCacheManager.h

class HybridCacheManager {
public:
    HybridCacheManager(const GraphConfig& config, DeviceId device);

    /// Get KV cache for a full-attention layer
    /// Returns nullptr for GDN layers
    ICPUKVCache* getKVCache(int layer_idx);

    /// Get GDN recurrent state for a GDN layer
    /// Returns nullptr for FA layers
    GDNState* getGDNState(int layer_idx);

    /// Get conv1d state for a GDN layer
    float* getConvState(int layer_idx);

    /// Reset all state (new conversation)
    void reset();

    /// Memory footprint
    size_t totalBytes() const;
    size_t fixedBytes() const;   // GDN state (constant)
    size_t dynamicBytes() const; // KV cache (grows with seq_len)

private:
    std::vector<std::unique_ptr<ICPUKVCache>> kv_caches_;  // FA layers only
    std::vector<GDNState> gdn_states_;                      // GDN layers only
    std::vector<std::vector<float>> conv_states_;           // GDN layers only
    std::vector<int> layer_type_map_;                       // layer_idx → type
};
```

### 7.3 GDNState Tensor

```cpp
// src/v2/execution/cache/GDNState.h

struct GDNState {
    // Recurrent state: S shape (n_heads, d_k, d_v) — always FP32
    // per mamba_ssm_dtype config
    std::vector<float> state_data;
    int n_heads;
    int d_k;
    int d_v;

    // Accessors
    float* mutable_data() { return state_data.data(); }
    const float* data() const { return state_data.data(); }
    size_t size_bytes() const { return state_data.size() * sizeof(float); }

    // Per-head view
    float* head_state(int h) {
        return state_data.data() + h * d_k * d_v;
    }

    void reset() {
        std::fill(state_data.begin(), state_data.end(), 0.0f);
    }

    static GDNState create(int n_heads, int d_k, int d_v) {
        GDNState s;
        s.n_heads = n_heads;
        s.d_k = d_k;
        s.d_v = d_v;
        s.state_data.resize(n_heads * d_k * d_v, 0.0f);
        return s;
    }
};
```

---

## 8. Qwen 3.5 Schema Factory

### 8.1 Schema Structure

```cpp
// src/v2/models/qwen35/Qwen35Schema.h

class Qwen35SchemaFactory : public ISchemaFactory {
public:
    std::string architectureName() const override { return "qwen3_5"; }

    GraphSchema createSchema() const override {
        GraphSchema schema;
        schema.name = "qwen3_5";
        schema.version = "1.0";

        // ================================================================
        // Named Layer Templates (hybrid attention)
        // ================================================================

        // GDN attention template (75% of layers)
        LayerTemplate gdn_template;
        gdn_template.attention_stages = {
            stageRMSNorm("attn_norm"),
            stageGDNProjection(),   // 4 separate GEMMs: QKV, Z, A, B
            stageShortConv1d(),     // Causal conv on QKV
            stagePartialRoPE(),     // RoPE on first 25% of head dims
            stageGDNRecurrence(),   // Delta rule recurrence
            stageGatedRMSNorm(),    // RMSNorm(output) ⊙ SiLU(z)
            stageOutputGate(),      // sigmoid(gate) * attn_output
            stageWoProjection(),    // Output projection
            stageResidualAdd("attn_residual"),
        };
        gdn_template.ffn_stages = denseSwiGLU();  // same for dense model

        // Full attention template (25% of layers)
        LayerTemplate fa_template;
        fa_template.attention_stages = {
            stageRMSNorm("attn_norm"),
            stageFusedQKV_WithGate(), // QKV + gate projection
            stageQKNorm(),            // Per-head QK norm (like Qwen3)
            stagePartialRoPE(),       // RoPE on first 25% of head dims
            stageKVCacheAppend(),
            stageFlashAttention(),    // Standard FA with optional sliding window
            stageOutputGate(),        // sigmoid(gate) * attn_output
            stageWoProjection(),
            stageResidualAdd("attn_residual"),
        };
        fa_template.ffn_stages = denseSwiGLU();

        schema.named_templates["gdn"] = std::move(gdn_template);
        schema.named_templates["full_attention"] = std::move(fa_template);

        // Layer assignment pattern: [gdn, gdn, gdn, fa, gdn, gdn, gdn, fa, ...]
        // Populated at resolution time from config.layer_types

        // Embedding + LM head (same as Qwen3)
        schema.embedding = embeddingStage();
        schema.lm_head_stages = lmHeadStages();

        // Buffer specs (union of GDN and FA buffers)
        schema.layer_buffers = allLayerBuffers();
        schema.model_buffers = modelBuffers();

        return schema;
    }
};
```

### 8.2 MoE Variant Schema

```cpp
// src/v2/models/qwen35/Qwen35MoeSchema.h

class Qwen35MoeSchemaFactory : public ISchemaFactory {
public:
    std::string architectureName() const override { return "qwen3_5_moe"; }

    GraphSchema createSchema() const override {
        // Start from the dense schema
        Qwen35SchemaFactory dense_factory;
        GraphSchema schema = dense_factory.createSchema();
        schema.name = "qwen3_5_moe";

        // Replace dense FFN with MoE FFN on ALL templates
        schema.moe_ffn_template = MoELayerTemplate{
            .router = {
                .name = "moe_router",
                .type = StageType::MoERouter,
                .inputs = {{"post_attn_hidden", BufferSemantic::Input},
                           {"weights.ffn_gate_inp", BufferSemantic::Input}},
                .outputs = {{"router_logits", BufferSemantic::Output},
                            {"expert_indices", BufferSemantic::Output},
                            {"expert_weights", BufferSemantic::Output}},
            },
            .dispatch = {
                .name = "moe_dispatch",
                .type = StageType::MoEDispatch,
                .inputs = {{"post_attn_hidden"}, {"expert_indices"}},
                .outputs = {{"dispatched_tokens"}},
            },
            .expert_ffn = {
                .name = "expert_ffn",
                .type = StageType::MoEFFN,
                .tp_mode = TPMode::ExpertParallel,
                // Weights: gate_up_exps [num_experts, 2*intermediate, hidden]
                //          down_exps    [num_experts, hidden, intermediate]
            },
            .combine = {
                .name = "moe_combine",
                .type = StageType::MoECombine,
            },
            .shared_expert_ffn = {
                .name = "shared_expert",
                .type = StageType::MoESharedExpert,
                // Reuses dense SwiGLU: gate_proj, up_proj, down_proj
                // + sigmoid gating via shared_expert_gate weight
            },
            .num_experts = 256,
            .top_k = 8,
            .has_shared_expert = true,
        };

        // Weight sharding: add MoE patterns
        auto sharding = getWeightShardingConfig();
        // Expert weights use ExpertParallel sharding
        sharding.patterns.push_back(
            {"ffn_gate_up_exps", WeightShardingMode::ExpertParallel,
             WeightDimensionType::None, "Expert gate+up (3D) — distribute experts"});
        sharding.patterns.push_back(
            {"ffn_down_exps", WeightShardingMode::ExpertParallel,
             WeightDimensionType::None, "Expert down (3D) — distribute experts"});
        // Router replicated, shared expert uses standard FFN sharding

        return schema;
    }
};
```

---

## 9. Graph Builder: Qwen35Graph

```cpp
// src/v2/models/qwen35/Qwen35Graph.h

class Qwen35Graph : public IGraphBuilder {
public:
    Qwen35Graph(const GraphConfig& config, std::shared_ptr<MPIContext> mpi);

    ComputeGraph buildPrefillGraph(int seq_len) override;
    ComputeGraph buildDecodeGraph() override;

private:
    // Per-layer graph building based on layer type
    void buildGDNAttention(ComputeGraph& graph, int layer_idx);
    void buildFullAttention(ComputeGraph& graph, int layer_idx);

    // FFN variants
    void buildDenseFFN(ComputeGraph& graph, int layer_idx);
    void buildMoEFFN(ComputeGraph& graph, int layer_idx);

    // Layer dispatch
    void buildLayer(ComputeGraph& graph, int layer_idx) {
        // Attention
        if (config_.isGDNLayer(layer_idx)) {
            buildGDNAttention(graph, layer_idx);
        } else {
            buildFullAttention(graph, layer_idx);
        }

        // FFN
        if (config_.isMoE()) {
            buildMoEFFN(graph, layer_idx);
        } else {
            buildDenseFFN(graph, layer_idx);
        }
    }

    GraphConfig config_;
    HybridCacheManager cache_;
    std::shared_ptr<ExpertPlacementMap> expert_placement_;  // MoE only
};
```

---

## 10. GGUF Weight Mapping

### 10.1 GDN Layer Weights (per layer)

| GGUF Name | HuggingFace Name | Shape (35B-A3B) | Notes |
|-----------|------------------|-----------------|-------|
| `blk.{l}.attn_norm.weight` | `layers.{l}.input_layernorm.weight` | (2048,) | Subtract 1 on load |
| `blk.{l}.post_attention_norm.weight` | `layers.{l}.post_attention_layernorm.weight` | (2048,) | Subtract 1 on load |
| `blk.{l}.attn_qkv.weight` | `layers.{l}.linear_attn.in_proj_qkv.weight` | (Q+K+V, 2048) | Q=16×128, K=16×128, V=32×128 |
| `blk.{l}.attn_gate.weight` | `layers.{l}.linear_attn.in_proj_z.weight` | (2048, 2048) | Output gate Z |
| `blk.{l}.ssm_alpha.weight` | `layers.{l}.linear_attn.in_proj_a.weight` | (16, 2048) | Alpha gate |
| `blk.{l}.ssm_beta.weight` | `layers.{l}.linear_attn.in_proj_b.weight` | (16, 2048) | Beta gate |
| `blk.{l}.ssm_conv1d.weight` | `layers.{l}.linear_attn.conv1d.weight` | (ssm_inner, 1, 4) | Unsqueeze dim-1 |
| `blk.{l}.ssm_dt.bias` | `layers.{l}.linear_attn.dt_bias` | (16,) | Decay timestep bias |
| `blk.{l}.ssm_a` | `layers.{l}.linear_attn.A_log` | (16,) | `-exp(A_log)` in GGUF → `log(-x)` |
| `blk.{l}.ssm_norm.weight` | `layers.{l}.linear_attn.norm.weight` | (2048,) | NO subtract-1 |
| `blk.{l}.ssm_out.weight` | `layers.{l}.linear_attn.out_proj.weight` | (2048, 2048) | Wo projection |

### 10.2 Full Attention Layer Weights (per layer)

| GGUF Name | HuggingFace Name | Shape (35B-A3B) | Notes |
|-----------|------------------|-----------------|-------|
| `blk.{l}.attn_q.weight` | `layers.{l}.self_attn.q_proj.weight` | (16×256, 2048) | Includes gate portion |
| `blk.{l}.attn_k.weight` | `layers.{l}.self_attn.k_proj.weight` | (2×256, 2048) | |
| `blk.{l}.attn_v.weight` | `layers.{l}.self_attn.v_proj.weight` | (2×256, 2048) | |
| `blk.{l}.attn_output.weight` | `layers.{l}.self_attn.o_proj.weight` | (2048, 16×256) | |
| `blk.{l}.attn_q_norm.weight` | `layers.{l}.self_attn.q_norm.weight` | (256,) | Per-head QK norm |
| `blk.{l}.attn_k_norm.weight` | `layers.{l}.self_attn.k_norm.weight` | (256,) | Per-head QK norm |

### 10.3 MoE Weights (per layer, MoE variants only)

| GGUF Name | Shape (35B-A3B) | Notes |
|-----------|-----------------|-------|
| `blk.{l}.ffn_gate_inp.weight` | (256, 2048) | Router |
| `blk.{l}.ffn_gate_up_exps.weight` | (256, 1024, 2048) | **3D**: [experts, 2×intermediate, hidden] |
| `blk.{l}.ffn_down_exps.weight` | (256, 2048, 512) | **3D**: [experts, hidden, intermediate] |
| `blk.{l}.ffn_gate.weight` | (512, 2048) | Shared expert gate proj |
| `blk.{l}.ffn_up.weight` | (512, 2048) | Shared expert up proj |
| `blk.{l}.ffn_down.weight` | (2048, 512) | Shared expert down proj |
| `blk.{l}.ffn_shared_expert_gate.weight` | (1, 2048) | Sigmoid gating scalar |

### 10.4 Weight Transforms on Load

| Transform | GGUF Tensors | Operation |
|-----------|-------------|-----------|
| Subtract 1 | `attn_norm.weight`, `post_attention_norm.weight`, `output_norm.weight` | `weight -= 1.0` (Qwen3.5 stores `1 + w`) |
| **No** subtract 1 | `ssm_norm.weight` (GDN gated norm) | Pass through unchanged |
| A_log inverse | `ssm_a` | GGUF stores `-exp(A_log)`, reverse to `log(-x)` |
| Conv1d unsqueeze | `ssm_conv1d.weight` | `(channels, kernel)` → `(channels, 1, kernel)` |

---

## 11. BufferArena Extensions

### 11.1 New BufferIds

```cpp
enum class BufferId : uint32_t {
    // ... existing ...

    // GDN-specific buffers
    GDN_QKV,              // Concatenated QKV projection output
    GDN_Z,                // Gate Z projection output (for gated RMSNorm)
    GDN_ALPHA,            // Alpha gate (write strength)
    GDN_BETA,             // Beta gate (output gate)
    GDN_CONV_OUTPUT,      // Short conv1d output
    GDN_RECURRENCE_OUT,   // GDN recurrence output
    GDN_STATE,            // Per-layer recurrent state (not arena-managed?)
    GDN_CONV_STATE,       // Per-layer conv state (not arena-managed?)

    // Output gate buffer (shared by GDN and FA)
    ATTN_OUTPUT_GATE,     // sigmoid gate for attention output

    // MoE-specific buffers
    MOE_ROUTER_LOGITS,
    MOE_EXPERT_INDICES,
    MOE_EXPERT_WEIGHTS,
    MOE_DISPATCH_SCRATCH,
    MOE_EXPERT_OUTPUT,
    MOE_SHARED_EXPERT_OUTPUT,
    MOE_COMBINED_OUTPUT,
};
```

### 11.2 Buffer Aliasing for Hybrid Layers

GDN and FA layers never execute simultaneously (they're sequential per-layer), so their scratch buffers can alias:

```cpp
// Alias group: "attention_scratch"
// GDN buffers alias with FA buffers since they never coexist
AliasGroupSpec attn_scratch = {
    .name = "attention_scratch",
    .buffer_names = {
        "GDN_QKV", "GDN_Z", "GDN_ALPHA", "GDN_BETA",
        "GDN_CONV_OUTPUT", "GDN_RECURRENCE_OUT",
        "Q", "K", "V", "Q_rope", "K_rope",
        "ATTN_OUTPUT", "workspace_scores"
    },
    .description = "GDN and FA attention scratch share memory",
    .estimated_savings_percent = 40.0f,
};
```

---

## 12. Model Registration

```cpp
// src/v2/models/ModelRegistrations.cpp

void registerBuiltinModels() {
    // ... existing Qwen2, Qwen3 ...

    // Qwen3.5 Dense (GDN + FA, dense FFN)
    GraphBuilderRegistry::registerFactory("qwen3_5",
        [](const GraphConfig& cfg, std::shared_ptr<MPIContext> mpi) {
            return std::make_shared<Qwen35Graph>(cfg, std::move(mpi));
        });
    SchemaFactoryRegistry::registerFactory("qwen3_5",
        []() { return std::make_unique<Qwen35SchemaFactory>(); });

    // Qwen3.5 MoE (GDN + FA, MoE FFN)
    GraphBuilderRegistry::registerFactory("qwen3_5_moe",
        [](const GraphConfig& cfg, std::shared_ptr<MPIContext> mpi) {
            return std::make_shared<Qwen35Graph>(cfg, std::move(mpi));
        });
    SchemaFactoryRegistry::registerFactory("qwen3_5_moe",
        []() { return std::make_unique<Qwen35MoeSchemaFactory>(); });
}
```

---

## 13. Build Order: MoE First, Then GDN

The recommended build order frontloads MoE infrastructure so it's available when we build Qwen 3.5:

### Phase A: MoE Graph Infrastructure (model-agnostic)

These changes are general MoE support, not Qwen 3.5-specific. They benefit any future MoE model (DeepSeek-V3, Mixtral, DBRX).

| Step | Task | Files | Test |
|------|------|-------|------|
| A1 | **`MoELayerTemplate`** in GraphSchema | `GraphSchema.h` | Unit: schema construction |
| A2 | **`ExpertPlacementMap`** data structure | `execution/moe/ExpertPlacementMap.h/cpp` | Unit: placement queries |
| A3 | **`MoERouterStage`** — complete implementation | `MoEStages.cpp` | Unit: softmax + top-k |
| A4 | **`MoEDispatchStage`** — token scatter | `MoEStages.h/cpp` | Unit: scatter/gather |
| A5 | **`MoEExpertStage`** — SwiGLU FFN on 3D weights | `MoEStages.cpp` | Unit: single expert |
| A6 | **`MoECombineStage`** — weighted merge | `MoEStages.cpp` | Unit: weighted sum |
| A7 | **`MoESharedExpertStage`** — dense FFN + sigmoid gate | New stage | Unit: shared expert |
| A8 | **3D tensor support** in GGUF loader | `loaders/GGUFParser.cpp` | Unit: load 3D Q4_0 |
| A9 | **`ExpertParallel`** weight sharding mode | `WeightManager.cpp` | Unit: expert distribution |
| A10 | **`ExpertWeightCache`** for CPU offloading | `execution/moe/ExpertWeightCache.h/cpp` | Unit: LRU cache |
| A11 | **GraphResolver** MoE layer emission | `GraphResolver.cpp` | Integration: resolve MoE schema |
| A12 | **MoE BufferIds** in BufferArena | `BufferId.h`, `BufferArena.cpp` | Unit: buffer allocation |

### Phase B: Heterogeneous Layer Templates (model-agnostic)

This enables any model with mixed layer types (not just Qwen 3.5).

| Step | Task | Files | Test |
|------|------|-------|------|
| B1 | **`named_templates`** in GraphSchema | `GraphSchema.h` | Unit: schema with multiple templates |
| B2 | **`layer_template_names`** resolution | `GraphResolver.cpp` | Unit: per-layer dispatch |
| B3 | **Per-layer buffer sizing** in BufferArena | `BufferArena.cpp` | Unit: variable dims |
| B4 | **`HybridCacheManager`** | New file | Unit: dual KV/GDN state |
| B5 | **`partial_rotary_factor`** in RoPE stage | `RoPEStage.cpp` | Unit: partial RoPE |
| B6 | **`AttentionOutputGateStage`** | New stage | Unit: sigmoid gating |
| B7 | **`GatedRMSNormStage`** | New stage or extend RMSNorm | Unit: gated norm |
| B8 | **`subtract_one`** norm transform in loader | `GGUFLoader.cpp` | Unit: weight transform |
| B9 | **Variable GraphConfig fields** | `GraphTypes.h` | Unit: per-layer queries |

### Phase C: GDN Kernels

| Step | Task | Files | Test |
|------|------|-------|------|
| C1 | **`GDNProjectionStage`** — 4 separate GEMMs | New stage | Unit: projection output shapes |
| C2 | **`ShortConv1dStage`** — causal conv + SiLU | New stage + CPU kernel | Unit: conv1d correctness |
| C3 | **`GDNRecurrenceStage`** — CPU reference | New stage + kernel | Parity: vs PyTorch GDN |
| C4 | **Chunk-parallel prefill** kernel | CPU kernel | Parity: chunk vs sequential |
| C5 | **Single-step decode** kernel | CPU kernel | Parity: decode step |
| C6 | **GDN state management** integration | `HybridCacheManager` | Integration: state persistence |

### Phase D: Qwen 3.5 Dense (4B)

| Step | Task | Files | Test |
|------|------|-------|------|
| D1 | **`Qwen35SchemaFactory`** | `models/qwen35/Qwen35Schema.h` | Unit: schema validation |
| D2 | **`Qwen35Graph`** — hybrid layer builder | `models/qwen35/Qwen35Graph.h/cpp` | Unit: graph construction |
| D3 | **GGUF weight loading** for Qwen3.5 | `loaders/` | Unit: load Qwen3.5-4B weights |
| D4 | **Tied embeddings** support | `WeightManager.cpp` | Unit: alias test |
| D5 | **Model registration** | `ModelRegistrations.cpp` | Unit: factory lookup |
| D6 | **Python reference** parity setup | `python/reference/qwen35.py` | Already done |
| D7 | **End-to-end prefill parity** | Integration test | Parity: Qwen3.5-4B prefill |
| D8 | **End-to-end decode parity** | Integration test | Parity: Qwen3.5-4B decode |
| D9 | **Benchmark** vs PyTorch | Performance test | tok/s comparison |

### Phase E: Qwen 3.5 MoE (35B-A3B)

| Step | Task | Files | Test |
|------|------|-------|------|
| E1 | **`Qwen35MoeSchemaFactory`** | `models/qwen35/Qwen35MoeSchema.h` | Unit: MoE schema |
| E2 | **MoE weight loading** — 3D expert tensors | `loaders/` | Unit: load 256 experts |
| E3 | **Separate LM head** (`tie_word_embeddings=false`) | `WeightManager.cpp` | Unit: separate output weight |
| E4 | **Expert placement** — CPU offloading config | CLI parsing | Unit: config validation |
| E5 | **End-to-end prefill** with CPU-offloaded experts | Integration | Parity: 35B-A3B prefill |
| E6 | **Expert parallelism** — TP across domain | `LocalTPContext` | Integration: multi-device |
| E7 | **Dynamic rebalancing** | `ExpertRebalancer` | Integration: histogram-based |
| E8 | **Benchmark** — expert hit rate, throughput | Performance | tok/s, expert cache stats |

### Phase F: Optimized Kernels

| Step | Task |
|------|------|
| F1 | GDN recurrence — AVX-512 JIT (CPU) |
| F2 | GDN recurrence — CUDA kernel |
| F3 | Short conv1d — fused with projection (CPU/GPU) |
| F4 | MoE expert batching — grouped GEMM for same-expert tokens |
| F5 | MoE dispatch — CUDA kernel for token scatter/gather |
| F6 | Expert weight prefetch pipeline — overlap H2D with compute |

---

## 14. Dependency Graph

```
Phase A (MoE infra) ─────┐
                          ├──→ Phase E (Qwen 3.5 MoE)
Phase B (Hybrid layers) ──┤
                          ├──→ Phase D (Qwen 3.5 Dense 4B)
Phase C (GDN kernels) ────┘
                                    │
                                    ▼
                          Phase F (Optimized kernels)
```

- **A and B are independent** — can be developed in parallel
- **C depends on B** (GDN stages need hybrid layer infrastructure)
- **D depends on B + C** (dense Qwen 3.5 needs GDN + hybrid)
- **E depends on A + D** (MoE Qwen 3.5 needs MoE infra + dense parity first)
- **F is optional** — performance optimization after correctness

---

## 15. What Changes in Core Infrastructure

| Core Component | Change | Impact |
|----------------|--------|--------|
| `GraphSchema.h` | Add `named_templates`, `moe_ffn_template`, `layer_template_names` | Medium — additive, backward compatible |
| `GraphResolver.cpp` | Per-layer template dispatch, MoE layer emission | Medium — new code paths |
| `GraphTypes.h` | Add ~20 new fields to `GraphConfig` | Low — additive |
| `BufferId.h` | Add ~12 new buffer IDs | Low — additive |
| `StageType` enum | Add ~8 new stage types | Low — additive |
| `RoPEStage.cpp` | Add `partial_rotary_factor` parameter | Low — backward compatible |
| `RMSNormStage.cpp` | Add `subtract_one` weight mode | Low — backward compatible |
| `WeightManager.cpp` | Add `ExpertParallel` sharding, 3D tensor load | Medium |
| `GGUFParser.cpp` | Parse Qwen3.5 metadata, 3D tensors | Medium |
| `ModelRegistrations.cpp` | Add 2 new registrations | Trivial |
| `DeviceGraphExecutor.cpp` | **No changes** | None — existing multi-device dispatch works |
| `BufferArena.cpp` | **No changes** (new IDs only) | None |
| `TransferEngine.h` | **No changes** | None |

**Key principle**: all changes are additive. Existing Qwen2/3 paths are untouched.

---

## 16. Open Questions

1. **Conv1d state in BufferArena vs separate**: Conv states are small (24 KB/layer) and persistent across decode steps. Should they live in BufferArena (requires persistent semantics) or in HybridCacheManager (simpler, outside arena)?

2. **GDN state precision**: `mamba_ssm_dtype: float32` suggests FP32 state is required. Can we use FP16 state on GPU for memory savings without losing quality? Needs parity testing.

3. **Buffer sizing for variable head dims**: The arena allocates max-size buffers once. With head_dim=256 (FA) vs 128 (GDN), should we allocate for the max (256) and waste memory on GDN layers, or have per-layer-type buffer allocation? Max-size is simpler and wastes only ~100 KB.

4. **Expert weight quantization format**: 3D expert tensors `[256, 2*intermediate, hidden]` — does Q4_0 apply per-expert-slice (treating each expert as a 2D matrix) or across the full 3D tensor? llama.cpp convention is per-expert slicing.

5. **Shared expert on which device?**: In the MoE domain offloading scenario (attention on CUDA, experts on ROCm), the shared expert runs on all tokens — should it execute on the attention domain (avoiding transfer) or the expert domain (co-located with sparse experts)?
