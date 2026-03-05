# Qwen3.5 Architecture Research & Llaminar Support Analysis

**Date**: 2025-07-27 (updated 2025-07-27 with MoE variants)  
**Status**: Research Complete (Dense + MoE)  
**Companion Documents**:  
- `docs/v2/cleanup/QWEN3NEXT_GDN_PROJECT_PLAN.md` — Phases 1–8: GDN + hybrid attention  
- `docs/v2/cleanup/QWEN3NEXT_MTP_PROJECT_PLAN.md` — Phases 9–15: Multi-Token Prediction  

**Sources (Dense)**:  
- [Qwen3.5-4B config.json](https://huggingface.co/Qwen/Qwen3.5-4B/resolve/main/config.json)  
- [Qwen3.5-4B Model Card](https://huggingface.co/Qwen/Qwen3.5-4B)  
- [HuggingFace `modular_qwen3_5.py`](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_5/modular_qwen3_5.py)  
- [HuggingFace `modular_qwen3_next.py`](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_next/modular_qwen3_next.py)  
- [HuggingFace `configuration_qwen3_next.py`](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_next/configuration_qwen3_next.py)

**Sources (MoE)**:  
- [Qwen3.5-35B-A3B config.json](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/resolve/main/config.json)  
- [Qwen3.5-122B-A10B config.json](https://huggingface.co/Qwen/Qwen3.5-122B-A10B/resolve/main/config.json)  
- [Qwen3.5-35B-A3B Model Card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)  
- [Qwen3.5-122B-A10B Model Card](https://huggingface.co/Qwen/Qwen3.5-122B-A10B)  
- [HuggingFace `modular_qwen3_5_moe.py`](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_5_moe/modular_qwen3_5_moe.py)  
- [HuggingFace `modeling_qwen2_moe.py`](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py)

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Qwen3.5 Architecture Overview](#qwen35-architecture-overview)
- [Qwen3.5 vs Qwen3-Next Relationship](#qwen35-vs-qwen3-next-relationship)
- [Complete Config Parameter Reference (4B Dense)](#complete-config-parameter-reference-4b-dense)
- [Hybrid Layer Architecture](#hybrid-layer-architecture)
- [GDN Implementation Details (Qwen3.5 Variant)](#gdn-implementation-details-qwen35-variant)
- [Full Attention Implementation Details](#full-attention-implementation-details)
- [Dual Cache System](#dual-cache-system)
- [MoE Architecture (35B-A3B and 122B-A10B)](#moe-architecture-35b-a3b-and-122b-a10b)
- [MoE Config Parameter Reference](#moe-config-parameter-reference)
- [MoE Weight Layout and TP Sharding](#moe-weight-layout-and-tp-sharding)
- [Gap Analysis: GDN Plan vs Actual Qwen3.5](#gap-analysis-gdn-plan-vs-actual-qwen35)
- [Gap Analysis: Llaminar Codebase vs Qwen3.5](#gap-analysis-llaminar-codebase-vs-qwen35)
- [MTP Confirmation](#mtp-confirmation)
- [Vision Encoder (Deferred)](#vision-encoder-deferred)
- [Recommended Plan Updates](#recommended-plan-updates)
- [Priority Order for Text-Only Inference](#priority-order-for-text-only-inference)
- [Appendix: Full Qwen3.5 Family Comparison](#appendix-full-qwen35-family-comparison)

---

## Executive Summary

Qwen3.5 is the production released model family that implements the Qwen3-Next architecture. The family includes **three variants**: a dense 4B model and two Mixture-of-Experts models (35B-A3B and 122B-A10B). Key findings:

1. **Qwen3.5 inherits from Qwen3-Next** — it is not a separate architecture but a specialization: the 4B variant removes MoE (dense FFN), changes the GDN projection layout from fused to separate, and adds a vision encoder for VLM support.

2. **Two MoE variants exist**: Qwen3.5-35B-A3B (35B total, 3B activated per token) and Qwen3.5-122B-A10B (122B total, 10B activated per token). Both use 256 experts with top-8 routing plus 1 shared expert. **Every layer uses MoE FFN** — both GDN and full-attention layers.

3. **The GDN project plan (Phases 1–8) covers ~80% of what's needed**, but has **6 significant gaps** that must be addressed before Qwen3.5 can run:
   - Partial RoPE (`partial_rotary_factor: 0.25`)
   - mRoPE (3D rotary for multimodal)
   - Different `head_dim` per layer type (256 for full attention, 128 for GDN)
   - Different key-head counts per layer type (4 GQA for full attention, 16 for GDN)
   - Attention output gate on **full attention** layers (not just GDN layers)
   - Tied word embeddings (dense only — MoE uses **separate** output weights)

4. **The MTP plan (Phases 9–15) is confirmed correct** — all Qwen3.5 variants use `mtp_num_hidden_layers: 1` with shared embeddings, exactly matching the plan's D=1 assumption.

5. **For text-only inference**, the vision encoder can be completely ignored — `Qwen3_5ForCausalLM` strips visual weights via `_keys_to_ignore_on_load_unexpected`.

6. **GDN projection layout is different** from what the GDN plan assumed — Qwen3.5 uses 4 separate projections (`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`) instead of Qwen3-Next's 2 fused projections (`in_proj_qkvz`, `in_proj_ba`). This is actually **simpler** to implement.

7. **MoE is a thin add-on to the dense architecture** — the MoE variants are architecturally identical in the attention path (GDN + full attention). The ONLY structural change is replacing the dense SwiGLU FFN with `SparseMoeBlock` (router + 256 experts + shared expert with gating). The GDN plan's Phase 7 (MoE Integration) directly applies.

8. **MoE variants differ from dense in `tie_word_embeddings`** — dense uses `true` (shared), MoE uses `false` (separate). The weight loader must handle both cases.

---

## Qwen3.5 Architecture Overview

Qwen3.5 is a **unified Vision-Language Model (VLM) family** built on the Qwen3-Next architecture:

| Property | Value |
|----------|-------|
| Model type (dense) | `qwen3_5` |
| Model type (MoE) | `qwen3_5_moe` |
| HF class (dense) | `Qwen3_5ForConditionalGeneration` |
| HF class (MoE) | `Qwen3_5MoeForConditionalGeneration` |
| Text backbone | Hybrid GDN + full-attention transformer |
| Vision encoder | ViT-style with spatial merge |
| Context length | 262,144 (native), 1,010,000 (with YaRN) |
| Vocabulary | 248,320 tokens |
| MTP support | Yes — D=1, shared embeddings, self-speculative decoding |

### Model Size Variants

| Variant | Total Params | Active Params | Hidden | Layers | FA Q Heads | FA KV Heads | FFN Type | GDN V Heads |
|---------|-------------|---------------|--------|--------|-----------|-------------|----------|-------------|
| **Qwen3.5-4B** | ~4B | ~4B | 2560 | 32 | 16 | 4 | Dense (9216) | 32 |
| **Qwen3.5-35B-A3B** | ~35B | ~3B | 2048 | 40 | 16 | 2 | MoE (256×512) | 32 |
| **Qwen3.5-122B-A10B** | ~122B | ~10B | 3072 | 48 | 32 | 2 | MoE (256×1024) | 64 |

**Common across all variants**: Hybrid GDN (75%) + full attention (25%) with `full_attention_interval: 4`, `head_dim: 256` (FA), `linear_key_head_dim: 128`, `linear_value_head_dim: 128`, `partial_rotary_factor: 0.25`, `attn_output_gate: true`, MTP D=1 with shared embeddings.

---

## Qwen3.5 vs Qwen3-Next Relationship

### Inheritance Chain (from HuggingFace transformers)

**Dense (Qwen3.5-4B)**:
```
Qwen3NextConfig
  └── Qwen3_5TextConfig (inherits, deletes MoE fields)

Qwen3NextGatedDeltaNet
  └── Qwen3_5GatedDeltaNet (inherits, replaces fused projections with separate)

Qwen3NextAttention
  └── Qwen3_5Attention (inherits, unchanged)

Qwen3NextDynamicCache
  └── Qwen3_5DynamicCache (inherits, unchanged)
```

**MoE (Qwen3.5-35B-A3B, Qwen3.5-122B-A10B)**:
```
Qwen3NextConfig
  └── Qwen3_5MoeTextConfig (inherits, keeps MoE fields, deletes intermediate_size/decoder_sparse_step/norm_topk_prob/mlp_only_layers)

Qwen3_5GatedDeltaNet
  └── Qwen3_5MoeGatedDeltaNet (pass-through, unchanged)

Qwen3NextAttention
  └── Qwen3_5MoeAttention (pass-through, unchanged)

Qwen3NextDynamicCache
  └── Qwen3_5MoeDynamicCache (pass-through, unchanged)

Qwen3NextExperts (← Qwen2MoeExperts)
  └── Qwen3_5MoeExperts (pass-through, unchanged)

Qwen3NextSparseMoeBlock (← Qwen2MoeSparseMoeBlock)
  └── Qwen3_5MoeSparseMoeBlock (pass-through, unchanged)

Qwen3VLMoeTextTopKRouter (← Qwen2MoeTopKRouter)
  └── Qwen3_5MoeTopKRouter (pass-through, unchanged)

Qwen3NextDecoderLayer
  └── Qwen3_5MoeDecoderLayer (overrides __init__: self.mlp = SparseMoeBlock for ALL layers)
```

**Key insight**: The MoE variant is architecturally a thin wrapper. The only meaningful override is in `Qwen3_5MoeDecoderLayer.__init__()` which replaces the dense FFN with `SparseMoeBlock`. All attention components (GDN, full attention, dual cache) are identical to the dense variant.

### What Qwen3.5 Changes vs Qwen3-Next

| Component | Qwen3-Next Base | Qwen3.5 Dense Override | Qwen3.5 MoE Override |
|-----------|-----------------|------------------------|----------------------|
| MoE | `decoder_sparse_step`, `num_experts`, etc. | **Deleted** — dense FFN only | **Kept** — 256 experts, top-8, all layers MoE |
| GDN QKV projection | Fused `in_proj_qkvz` (Q+K+V+Z interleaved) | Separate `in_proj_qkv` + `in_proj_z` | Same as dense (inherits from `Qwen3_5GatedDeltaNet`) |
| GDN gate projection | Fused `in_proj_ba` (beta+alpha interleaved) | Separate `in_proj_b` + `in_proj_a` | Same as dense |
| GDN deinterleaving | `fix_query_key_value_ordering()` needed | **Not needed** — already separate | **Not needed** |
| FFN | Dense or MoE per `decoder_sparse_step` | Dense SwiGLU only | MoE `SparseMoeBlock` on **every layer** |
| `tie_word_embeddings` | N/A | `true` (shared) | `false` (separate) |
| Vision | None | Full ViT encoder + spatial merge | Full ViT encoder + spatial merge |
| Attention | Unchanged | Unchanged (inherits output gate) | Unchanged |
| Cache | Unchanged | Unchanged (dual cache) | Unchanged |

### Key Insight: Qwen3.5 GDN Projections Are Simpler

The GDN plan assumed Qwen3-Next's fused projection pattern, which requires complex deinterleaving after the GEMM. Qwen3.5 simplifies this to standard separate GEMMs:

```python
# Qwen3-Next (COMPLEX): Fused projections + deinterleave
mixed = self.in_proj_qkvz(hidden_states)   # shape: (B, L, (Q+K+V+Z) interleaved)
q, k, v, z = self.fix_query_key_value_ordering(mixed)  # Complex reshape + permute

mixed_ba = self.in_proj_ba(hidden_states)  # shape: (B, L, (beta+alpha) interleaved)
b, a = split_and_reshape(mixed_ba)

# Qwen3.5 (SIMPLER): Separate projections — no deinterleaving needed
qkv = self.in_proj_qkv(hidden_states)      # shape: (B, L, Q_dim + K_dim + V_dim)
q, k, v = qkv.split([Q_dim, K_dim, V_dim], dim=-1)  # Simple split

z = self.in_proj_z(hidden_states)           # shape: (B, L, hidden_size)
b = self.in_proj_b(hidden_states)           # shape: (B, L, num_heads)  
a = self.in_proj_a(hidden_states)           # shape: (B, L, num_heads)
```

**Impact on GDN plan**: Phase 2 (GGUF weight loading) and Phase 4 (stages) become simpler. The `GDNProjectionStage` in the plan can use 4 standard GEMM calls instead of 2 fused GEMMs + deinterleaving.

---

## Complete Config Parameter Reference (4B Dense)

### Text Model Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `hidden_size` | 2560 | Model dimension |
| `num_hidden_layers` | 32 | Total transformer layers |
| `num_attention_heads` | 16 | For full attention layers |
| `num_key_value_heads` | 4 | GQA ratio 4:1 for full attention layers |
| `head_dim` | 256 | For full attention layers |
| `intermediate_size` | 9216 | Dense SwiGLU FFN |
| `vocab_size` | 248320 | Padded vocabulary |
| `max_position_embeddings` | 262144 | Native context length |
| `rms_norm_eps` | 1e-6 | RMSNorm epsilon |
| `hidden_act` | `silu` | SwiGLU activation |
| `tie_word_embeddings` | `true` | Output weights shared with embeddings |

### Hybrid Attention Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `full_attention_interval` | 4 | Every 4th layer is full attention |
| `layer_types` | `["la","la","la","fa"] × 8` | la=linear_attention (GDN), fa=full_attention |
| `attn_output_gate` | `true` | Applies to BOTH GDN and full attention layers |
| `sliding_window` | 32768 | For full attention layers |

### GDN (Linear Attention) Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `linear_num_key_heads` | 16 | Key heads for GDN layers |
| `linear_num_value_heads` | 32 | Value heads for GDN layers |
| `linear_key_head_dim` | 128 | d_k for GDN |
| `linear_value_head_dim` | 128 | d_v for GDN |
| `linear_conv_kernel_dim` | 4 | Short conv1d kernel size |

### RoPE Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `rope_theta` | 10,000,000 | High-frequency base |
| `partial_rotary_factor` | 0.25 | Only 25% of head_dim gets RoPE |
| `rope_scaling` | YaRN config | For extended context (>262K) |

### mRoPE Parameters (Multimodal)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `mrope_interleaved` | `true` | Interleaved 3D rotary encoding |
| `mrope_section` | `[11, 11, 10]` | Temporal, height, width sections |

### MTP Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `mtp_num_hidden_layers` | 1 | D=1 (single MTP depth) |
| `mtp_use_dedicated_embeddings` | `false` | Shares main model's embeddings |

### Vision Encoder Parameters (Deferred)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `vision_config.depth` | 24 | ViT layers |
| `vision_config.hidden_size` | 1024 | ViT hidden dimension |
| `vision_config.num_heads` | 16 | ViT attention heads |
| `vision_config.patch_size` | 16 | Patch size |
| `vision_config.spatial_merge_size` | 2 | 2×2 merge for token reduction |
| `vision_config.temporal_patch_size` | 1 | Temporal dimension stride |

---

## Hybrid Layer Architecture

### Layer Pattern

The 32 transformer layers follow a repeating pattern of 3 GDN + 1 full attention:

```
Layer  0: GDN (linear_attention)     ─┐
Layer  1: GDN (linear_attention)      │ Repeat
Layer  2: GDN (linear_attention)      │ block 1
Layer  3: Full Attention (full_attention) ─┘

Layer  4: GDN                        ─┐
Layer  5: GDN                         │ Repeat
Layer  6: GDN                         │ block 2
Layer  7: Full Attention             ─┘

... (×8 total = 32 layers)
```

**Ratio**: 24 GDN layers + 8 full attention layers = 75% GDN / 25% full attention.

### Per-Layer-Type Dimensions

This is a **critical finding** — the GDN plan assumed uniform dimensions across all layers:

| Dimension | GDN (linear_attention) | Full Attention |
|-----------|----------------------|----------------|
| `num_attention_heads` | 16 | 16 |
| `num_key_heads` | 16 | 4 (GQA) |
| `num_value_heads` | 32 | 16 |
| `key_head_dim` | 128 | 256 |
| `value_head_dim` | 128 | 256 |
| Q projection output dim | 16 × 128 = 2048 | 16 × 256 × 2 = 8192 (×2 for gate) |
| K projection output dim | 16 × 128 = 2048 | 4 × 256 = 1024 |
| V projection output dim | 32 × 128 = 4096 | 4 × 256 = 1024 |
| RoPE dimensions | 128 (full, no partial_rotary) | 64 (25% of 256) |

**Note**: For GDN layers, `linear_key_head_dim=128` undergoes full RoPE (the `partial_rotary_factor` of 0.25 applies to full attention's `head_dim=256`, giving 64 RoPE dims). GDN layers likely apply RoPE to all 128 dims (needs verification from the pretrained weights — the HF code applies `partial_rotary_factor` in `Qwen3NextAttention` but the GDN layer inherits a separate RoPE config).

### Output Gate Mechanism

Both GDN and full attention layers use an **output gate**. The gate is produced differently:

**Full attention layers** (from `Qwen3NextAttention`):
```python
# Q projection outputs 2× width, split into query + gate
query_states, gate = self.q_proj(hidden_states).split(
    [self.num_heads * self.head_dim, self.num_heads * self.head_dim], dim=-1
)
# After attention computation:
attn_output = attn_output * F.sigmoid(gate)  # Gate applied
```

**GDN layers** (from `Qwen3_5GatedDeltaNet`):
```python
# Z projection provides the gate for gated RMSNorm
z = self.in_proj_z(hidden_states)
# After GDN recurrence:
core_attn_out = self.norm(core_attn_out, z)  # Gated RMSNorm: RMSNorm(x * sigmoid(z))
```

---

## GDN Implementation Details (Qwen3.5 Variant)

### Complete Forward Pass

From `Qwen3_5GatedDeltaNet.forward()`:

```
Input: hidden_states (B, L, 2560)

1. QKV Projection:
   qkv = in_proj_qkv(hidden_states)          # (B, L, Q_dim + K_dim + V_dim)
   → split → q (B, L, 2048), k (B, L, 2048), v (B, L, 4096)

2. Gate Projections:
   z = in_proj_z(hidden_states)               # (B, L, 2560) — for gated RMSNorm
   b = in_proj_b(hidden_states)               # (B, L, 16) — write gate
   a = in_proj_a(hidden_states)               # (B, L, 16) — decay input

3. Short Convolution (causal):
   qkv = transpose → causal_conv1d(qkv, conv_weight, conv_bias) → transpose
   Note: Conv operates on concatenated Q+K+V before splitting

4. Reshape to heads:
   q → (B, L, 16, 128)   [num_key_heads × key_head_dim]
   k → (B, L, 16, 128)   [num_key_heads × key_head_dim]
   v → (B, L, 32, 128)   [num_value_heads × value_head_dim]

5. Compute gates:
   beta = sigmoid(b)                          # (B, L, 16) — write gate
   g = -A_log.exp() * softplus(a + dt_bias)   # (B, L, 16) — decay

6. Value head expansion (if num_v_heads > num_k_heads — YES for Qwen3.5):
   q = repeat_interleave(q, num_v_heads // num_k_heads, dim=2)  # 16 → 32 heads
   k = repeat_interleave(k, num_v_heads // num_k_heads, dim=2)  # 16 → 32 heads

7. GDN Recurrence:
   IF prefill: output, state = chunk_gated_delta_rule(q, k, v, g, beta, initial_state)
   IF decode:  output, state = recurrent_gated_delta_rule(q, k, v, g, beta, initial_state)
   → output: (B, L, 32, 128), state: (B, 32, 128, 128)

8. Reshape + Gated RMSNorm:
   output → (B, L, 4096)                     # 32 × 128
   output = gated_rmsnorm(output, z)          # z provides gate: RMSNorm(x ⊙ sigmoid(z))

9. Output Projection:
   output = out_proj(output)                  # (B, L, 2560)
```

### GDN Weight Layout (Qwen3.5 — GGUF Expected Names)

| Weight | Shape | GGUF Name (Expected) | Sharding |
|--------|-------|----------------------|----------|
| `in_proj_qkv.weight` | (8192, 2560) | `blk.{i}.attn_qkv.weight` | COLUMN |
| `in_proj_z.weight` | (2560, 2560) | `blk.{i}.attn_z.weight` | COLUMN |
| `in_proj_b.weight` | (16, 2560) | `blk.{i}.attn_b.weight` | COLUMN |
| `in_proj_a.weight` | (16, 2560) | `blk.{i}.attn_a.weight` | COLUMN |
| `A_log` | (16,) | `blk.{i}.attn_a_log` | COLUMN |
| `dt_bias` | (16,) | `blk.{i}.attn_dt_bias` | COLUMN |
| `conv_weight` | (8192, 1, 4) | `blk.{i}.attn_conv.weight` | COLUMN |
| `conv_bias` | (8192,) | `blk.{i}.attn_conv.bias` | COLUMN |
| `norm.weight` | (4096,) | `blk.{i}.attn_norm.weight` | REPLICATE |
| `out_proj.weight` | (2560, 4096) | `blk.{i}.attn_output.weight` | ROW |

**Note**: Exact GGUF tensor names will need verification once GGUF exports of Qwen3.5 are available. The names above are extrapolated from Qwen2 patterns.

### Conv1d Detail

The short convolution operates on the **concatenated** Q+K+V (dim=8192) as a single depthwise conv before splitting:

```python
# Concatenate Q+K+V along channel dimension
mixed_qkv = cat([q, k, v], dim=-1)  # (B, L, 8192)
# Depthwise causal conv1d (kernel_size=4)
mixed_qkv = causal_conv1d(mixed_qkv.transpose(1,2), conv_weight, conv_bias).transpose(1,2)
# Split back into Q, K, V
q, k, v = mixed_qkv.split([q_dim, k_dim, v_dim], dim=-1)
```

This means there is a **single** conv per GDN layer, not three separate convolutions. The GDN plan's separate `conv_state_q/k/v` buffers should be replaced with a single `conv_state_qkv` buffer of size (8192, 3).

---

## Full Attention Implementation Details

From `Qwen3NextAttention` (inherited unchanged by `Qwen3_5Attention`):

```
Input: hidden_states (B, L, 2560)

1. Q Projection (with gate):
   mixed = q_proj(hidden_states)              # (B, L, 8192) — 2× width for gate
   query_states, gate = split(mixed)          # each (B, L, 4096) = 16 × 256

2. K Projection:
   key_states = k_proj(hidden_states)         # (B, L, 1024) = 4 × 256

3. V Projection:
   value_states = v_proj(hidden_states)       # (B, L, 1024) = 4 × 256

4. Reshape to heads:
   q → (B, 16, L, 256)
   k → (B, 4, L, 256)
   v → (B, 4, L, 256)

5. Partial RoPE:
   rope_dim = int(256 * 0.25) = 64
   q_rope, q_pass = q[..., :64], q[..., 64:]
   k_rope, k_pass = k[..., :64], k[..., 64:]
   q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope)
   q = cat([q_rope, q_pass], dim=-1)
   k = cat([k_rope, k_pass], dim=-1)

6. KV Cache Append + Sliding Window Attention:
   Standard FlashAttention with sliding_window=32768

7. Output Gate:
   attn_output = attn_output * sigmoid(gate)  # Element-wise gating

8. Output Projection:
   output = o_proj(attn_output)               # (B, L, 2560)
```

### Full Attention Weight Layout (GGUF Expected Names)

| Weight | Shape | GGUF Name (Expected) | Sharding | Notes |
|--------|-------|----------------------|----------|-------|
| `q_proj.weight` | (8192, 2560) | `blk.{i}.attn_q.weight` | COLUMN | 2× width for gate |
| `k_proj.weight` | (1024, 2560) | `blk.{i}.attn_k.weight` | COLUMN | |
| `v_proj.weight` | (1024, 2560) | `blk.{i}.attn_v.weight` | COLUMN | |
| `o_proj.weight` | (2560, 4096) | `blk.{i}.attn_output.weight` | ROW | |

---

## Dual Cache System

Qwen3.5 uses `Qwen3_5DynamicCache` (inherits `Qwen3NextDynamicCache`) with per-layer-type caching:

```python
class Qwen3NextDynamicCache:
    conv_states: Dict[int, Tensor]       # Per GDN layer: (B, channels, kernel_size-1)
    recurrent_states: Dict[int, Tensor]  # Per GDN layer: (B, n_heads, d_k, d_v)
    key_cache: List[Tensor]              # Per full-attention layer: (B, n_kv_heads, L, head_dim)
    value_cache: List[Tensor]            # Per full-attention layer: (B, n_kv_heads, L, head_dim)
```

| Layer Type | Cache Components | Memory Growth |
|-----------|-----------------|---------------|
| GDN | `conv_states` (fixed), `recurrent_states` (fixed) | **Constant** — does not grow with sequence length |
| Full Attention | `key_cache`, `value_cache` | **Linear** — grows with sequence length |

**Memory impact**: Only 8 out of 32 layers (25%) need a growing KV cache. This dramatically reduces memory requirements for long-context inference compared to a pure attention model.

---

## MoE Architecture (35B-A3B and 122B-A10B)

### Overview

The MoE Qwen3.5 variants replace the dense SwiGLU FFN with a **Sparse Mixture-of-Experts** block on **every layer** (both GDN and full-attention layers). The attention path is completely unchanged from the dense variant.

**Architecture pattern**:
- **35B-A3B**: `10 × (3 × (GDN → MoE) → 1 × (Full Attention → MoE))` = 40 layers
- **122B-A10B**: `12 × (3 × (GDN → MoE) → 1 × (Full Attention → MoE))` = 48 layers

### MoE SparseMoeBlock Structure

Each layer's FFN is a `Qwen2MoeSparseMoeBlock` containing:

```
SparseMoeBlock
├── gate (TopKRouter)          # Router: Linear(hidden_size → num_experts) + softmax + top-k
├── experts (Experts)          # Expert bank: 3D weight tensors
│   ├── gate_up_proj           # [num_experts, 2 × moe_intermediate, hidden_size]
│   └── down_proj              # [num_experts, hidden_size, moe_intermediate]
├── shared_expert (MLP)        # Dense SwiGLU expert (always active)
│   ├── gate_proj              # [shared_intermediate, hidden_size]
│   ├── up_proj                # [shared_intermediate, hidden_size]
│   └── down_proj              # [hidden_size, shared_intermediate]
└── shared_expert_gate         # Linear(hidden_size → 1, bias=False) — sigmoid gating
```

### MoE Forward Pass

```python
def forward(hidden_states):
    B, L, D = hidden_states.shape
    hidden_reshaped = hidden_states.view(-1, D)     # (B*L, D)

    # 1. Shared expert — always processes all tokens
    shared_output = shared_expert(hidden_reshaped)    # Dense SwiGLU: (B*L, D)

    # 2. Router — select top-k experts per token
    router_logits = linear(hidden_reshaped, gate.weight)  # (B*L, num_experts)
    router_probs = softmax(router_logits, dim=-1)
    top_k_values, top_k_indices = topk(router_probs, k=8)
    if norm_topk_prob:
        top_k_values /= top_k_values.sum(dim=-1, keepdim=True)

    # 3. Expert computation — conditional execution per expert
    expert_output = zeros_like(hidden_reshaped)       # (B*L, D)
    for expert_idx in active_experts:                 # Only process experts with assigned tokens
        token_indices = tokens_for_expert(expert_idx)
        current = hidden_reshaped[token_indices]
        gate, up = linear(current, experts.gate_up_proj[expert_idx]).chunk(2) 
        out = silu(gate) * up                         # SwiGLU activation
        out = linear(out, experts.down_proj[expert_idx])
        expert_output[token_indices] += out * top_k_values[token_indices]

    # 4. Shared expert gating
    shared_output = sigmoid(shared_expert_gate(hidden_reshaped)) * shared_output

    # 5. Combine
    return (expert_output + shared_output).view(B, L, D)
```

### Key MoE Differences from Dense

| Property | Dense (4B) | MoE (35B-A3B) | MoE (122B-A10B) |
|----------|-----------|----------------|------------------|
| FFN type | Dense SwiGLU (`intermediate_size: 9216`) | MoE: 256 experts + 1 shared | MoE: 256 experts + 1 shared |
| Expert intermediate dim | N/A | 512 (`moe_intermediate_size`) | 1024 (`moe_intermediate_size`) |
| Shared expert dim | N/A | 512 (`shared_expert_intermediate_size`) | 1024 (`shared_expert_intermediate_size`) |
| Experts activated per token | 1 (dense) | 8 (`num_experts_per_tok`) | 8 (`num_experts_per_tok`) |
| Total experts | N/A | 256 (`num_experts`) | 256 (`num_experts`) |
| MoE layers | 0 | **All 40** (no `decoder_sparse_step`) | **All 48** |
| `tie_word_embeddings` | `true` | **`false`** | **`false`** |
| `intermediate_size` | 9216 | N/A (removed from config) | N/A (removed from config) |
| Router normalization | N/A | softmax + top-k (no `norm_topk_prob`) | softmax + top-k |
| Hidden size | 2560 | 2048 | 3072 |
| Total layers | 32 | 40 | 48 |
| FA Q heads | 16 | 16 | 32 |
| FA KV heads | 4 | 2 | 2 |
| GDN key heads | 16 | 16 | 16 |
| GDN value heads | 32 | 32 | 64 |

### What Is Shared with Dense

These components are **identical** between dense and MoE variants:

- GDN attention (GatedDeltaNet) — same projections, same recurrence, same conv1d
- Full attention — same output gate, same partial RoPE, same head_dim=256
- Dual cache system — same conv_states + recurrent_states for GDN, same KV cache for FA
- RMSNorm (input and post-attention)
- Embedding layer (but tied vs separate — see below)
- Layer pattern: `["linear_attention" × 3, "full_attention"] × N`
- `partial_rotary_factor: 0.25`, `attn_output_gate: true`
- mRoPE: `mrope_interleaved: true`, `mrope_section: [11, 11, 10]`
- MTP: `mtp_num_hidden_layers: 1`, `mtp_use_dedicated_embeddings: false`

### Tied vs Separate Embeddings

| Model | `tie_word_embeddings` | Implication |
|-------|----------------------|-------------|
| 4B Dense | `true` | `lm_head.weight` is an alias for `embed_tokens.weight` — no separate tensor |
| 35B-A3B MoE | `false` | `lm_head.weight` is a **separate** learned parameter |
| 122B-A10B MoE | `false` | `lm_head.weight` is a **separate** learned parameter |

The weight loader must handle both cases. For dense: alias the LM head to embeddings. For MoE: load a separate `output.weight` tensor.

---

## MoE Config Parameter Reference

### 35B-A3B Specific Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `hidden_size` | 2048 | Smaller than 4B dense (2560) |
| `num_hidden_layers` | 40 | 10 repeating groups × 4 layers/group |
| `num_attention_heads` | 16 | FA Q heads |
| `num_key_value_heads` | 2 | FA KV heads (GQA 8:1) |
| `head_dim` | 256 | FA head dimension |
| `num_experts` | 256 | Total expert count |
| `num_experts_per_tok` | 8 | Top-k selection |
| `moe_intermediate_size` | 512 | Per-expert SwiGLU intermediate dim |
| `shared_expert_intermediate_size` | 512 | Shared expert dim |
| `linear_num_key_heads` | 16 | GDN key heads |
| `linear_num_value_heads` | 32 | GDN value heads |
| `linear_key_head_dim` | 128 | GDN d_k |
| `linear_value_head_dim` | 128 | GDN d_v |
| `rope_theta` | 10,000,000 | Same as dense |
| `tie_word_embeddings` | `false` | Separate LM head |
| `mamba_ssm_dtype` | `float32` | GDN state precision hint |

### 122B-A10B Specific Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `hidden_size` | 3072 | Largest in family |
| `num_hidden_layers` | 48 | 12 repeating groups × 4 layers/group |
| `num_attention_heads` | 32 | FA Q heads |
| `num_key_value_heads` | 2 | FA KV heads (GQA 16:1) |
| `head_dim` | 256 | FA head dimension |
| `num_experts` | 256 | Same as 35B |
| `num_experts_per_tok` | 8 | Same as 35B |
| `moe_intermediate_size` | 1024 | 2× the 35B expert size |
| `shared_expert_intermediate_size` | 1024 | 2× the 35B shared size |
| `linear_num_key_heads` | 16 | Same as 35B |
| `linear_num_value_heads` | 64 | 2× the 35B value heads |
| `linear_key_head_dim` | 128 | Same |
| `linear_value_head_dim` | 128 | Same |
| `rope_theta` | 10,000,000 | Same |
| `tie_word_embeddings` | `false` | Separate LM head |
| `mamba_ssm_dtype` | `float32` | GDN state precision hint |

### `mamba_ssm_dtype` Parameter

Both MoE configs include `"mamba_ssm_dtype": "float32"`. This specifies that the **GDN recurrent state** (`S` in `S_t = diag(exp(g)) · S_{t-1} + ...`) should be maintained in FP32 precision to avoid numerical drift during long-sequence generation. The Llaminar GDN recurrence kernel should use FP32 for the state tensor regardless of the model's activation dtype.

---

## MoE Weight Layout and TP Sharding

### MoE Weight Names (GGUF Expected)

Per-layer MoE weights (in addition to attention weights):

| Weight | Shape (35B) | Shape (122B) | GGUF Name (Expected) | Notes |
|--------|-------------|-------------|----------------------|-------|
| Router gate | (256, 2048) | (256, 3072) | `blk.{l}.ffn_gate_inp.weight` | Softmax router |
| Expert gate+up (packed) | (256, 1024, 2048) | (256, 2048, 3072) | `blk.{l}.ffn_gate_up_exps.weight` | 3D: [experts, 2×intermediate, hidden] |
| Expert down | (256, 2048, 512) | (256, 3072, 1024) | `blk.{l}.ffn_down_exps.weight` | 3D: [experts, hidden, intermediate] |
| Shared gate | (512, 2048) | (1024, 3072) | `blk.{l}.ffn_gate.weight` | Shared expert SwiGLU gate |
| Shared up | (512, 2048) | (1024, 3072) | `blk.{l}.ffn_up.weight` | Shared expert SwiGLU up |
| Shared down | (2048, 512) | (3072, 1024) | `blk.{l}.ffn_down.weight` | Shared expert SwiGLU down |
| Shared expert gate | (1, 2048) | (1, 3072) | `blk.{l}.ffn_shared_expert_gate.weight` | Sigmoid gating scalar |

**GGUF naming note**: The exact GGUF tensor names will depend on the converter used. The names above follow the llama.cpp convention for Qwen2-MoE models. Verify with actual GGUF files when implementing.

### MoE TP Sharding Plan (from HuggingFace config)

| Weight | TP Strategy | Description |
|--------|-------------|-------------|
| `layers.*.mlp.experts.gate_up_proj` | `packed_colwise` | Split output dimension across TP ranks |
| `layers.*.mlp.experts.down_proj` | `rowwise` | Split input dimension, allreduce after |
| `layers.*.mlp.experts` | `moe_tp_experts` | Special: distribute experts across TP ranks |
| `layers.*.mlp.shared_expert.gate_proj` | `colwise` | Standard column-parallel |
| `layers.*.mlp.shared_expert.up_proj` | `colwise` | Standard column-parallel |
| `layers.*.mlp.shared_expert.down_proj` | `rowwise` | Standard row-parallel |
| `layers.*.mlp.gate.weight` | Replicated | Router weights on all ranks |

**Expert parallelism (`moe_tp_experts`)**: With 256 experts and TP=8, each rank handles 32 experts locally. Tokens are routed, processed on the owning rank, and results are all-to-all communicated. This is distinct from tensor parallelism on individual expert weights.

### MoE Memory Breakdown (35B-A3B Example)

Per layer at FP16:
| Component | Size | Notes |
|-----------|------|-------|
| Router | 256 × 2048 × 2B = ~1 MB | Small |
| Expert gate_up | 256 × 1024 × 2048 × 2B = ~1 GB | Dominant cost |
| Expert down | 256 × 2048 × 512 × 2B = ~512 MB | |
| Shared expert | (512+512+2048) × 2048 × 2B = ~6.3 MB | Negligible |
| **Per-layer total** | **~1.5 GB** | |
| **All 40 layers** | **~60 GB** | FP16, before quantization |

At Q4_0 quantization (~4.5× compression): ~13 GB for expert weights alone. This is where Llaminar's weight streaming (`LLAMINAR_WEIGHT_STREAMING`) will be valuable — offloading inactive experts to host memory.

---

## Gap Analysis: GDN Plan vs Actual Qwen3.5

### Critical Gaps the GDN Plan Must Address

| # | Gap | GDN Plan Assumption | Qwen3.5 Reality | Impact |
|---|-----|---------------------|------------------|--------|
| 1 | **Partial RoPE** | Full RoPE on all head dimensions | Only 25% of head_dim gets RoPE (`partial_rotary_factor: 0.25`) | Need to split head dimensions into RoPE and pass-through portions. Affects full attention layers where head_dim=256, RoPE applied to first 64 dims only. |
| 2 | **mRoPE** | Standard 1D RoPE | 3D multimodal RoPE with `mrope_section: [11, 11, 10]` | Needed for vision-language input. Can defer for text-only inference (mRoPE degenerates to 1D when there's no visual input). |
| 3 | **Different head_dim per layer type** | Uniform `head_dim` for all layers | GDN: `key_head_dim=128`, `value_head_dim=128`; Full attention: `head_dim=256` | `Qwen3NextGraphConfig` needs separate dimension fields per layer type. Buffer sizes vary by layer type. |
| 4 | **Different key-head counts** | Uniform `n_kv_heads` for all layers | GDN: `linear_num_key_heads=16`; Full attention: `num_key_value_heads=4` | GQA ratio is 4:1 for full attention, 1:1 for GDN key heads (but 2:1 for GDN value heads). |
| 5 | **Output gate on full attention** | Only GDN has gating (via GatedRMSNorm) | Full attention also gates output via `sigmoid(gate)` from Q projection | Need a new stage or modified Wo projection stage that applies sigmoid gating. |
| 6 | **Tied word embeddings** | Separate output weight matrix | Dense: `tie_word_embeddings: true`; MoE: `false` | Weight loader must handle both aliased and separate weights depending on variant. |
| 7 | **MoE FFN (35B, 122B)** | Dense SwiGLU FFN | SparseMoeBlock with 256 experts, top-8, shared expert, sigmoid gating | Phase 7 in GDN plan now **confirmed required** — not just "expected". Need router, expert dispatch, shared expert, expert gating stages. |
| 8 | **3D expert weight tensors** | 2D weight matrices | Expert `gate_up_proj` is `[256, 2×intermediate, hidden]` — 3D | GGUF loader and tensor system must support 3D quantized weights. |

### Confirmed Correct in the GDN Plan

| Component | Plan Status | Notes |
|-----------|-------------|-------|
| GDN core recurrence | ✅ Correct | `S_t = diag(exp(g)) · S_{t-1} + β · k · (v − S^T k)^T` matches |
| GDN state tensor | ✅ Correct | Fixed-size (n_heads, d_k, d_v) per layer |
| Short convolution | ⚠️ Partially correct | Plan has 3 separate convs; Qwen3.5 uses 1 combined conv on concatenated QKV |
| Gated RMSNorm | ✅ Correct | `RMSNorm(x ⊙ sigmoid(z))` |
| Per-layer dispatch | ✅ Correct | `attn_type` array drives layer selection |
| Value head expansion | ✅ Correct | `num_v_heads > num_k_heads` with repeat_interleave |
| Dense FFN for all layers | ✅ Correct (4B only) | SwiGLU, same structure for GDN and full attention layers. MoE variants use SparseMoeBlock instead. |
| MoE Phase 7 | ✅ Confirmed needed | 35B and 122B models validate Phase 7 MoE integration |
| Chunk-parallel prefill | ✅ Correct | `chunk_gated_delta_rule` for prefill |
| Single-step decode | ✅ Correct | `recurrent_gated_delta_rule` for decode |

### GDN Projection Layout Change

| Component | GDN Plan (Qwen3-Next style) | Qwen3.5 Actual |
|-----------|------------------------------|-----------------|
| QKV+Z projection | Fused `in_proj_qkvz` → `fix_query_key_value_ordering()` | Separate `in_proj_qkv` + `in_proj_z` |
| Beta+Alpha projection | Fused `in_proj_ba` → split | Separate `in_proj_b` + `in_proj_a` |
| GEMM count | 2 fused + complex reshape | 4 simple GEMMs |
| Complexity | Higher (interleaved layout) | **Lower** (standard GEMMs) |

**Recommendation**: Update the GDN plan to use the Qwen3.5 separate projection layout. This simplifies both the weight loading (Phase 2) and the stage implementation (Phase 4). If a future Qwen3-Next MoE model uses the fused layout, add fused support later.

---

## Gap Analysis: Llaminar Codebase vs Qwen3.5

### Feature Support Matrix

| Feature | Llaminar Status | Effort | Notes |
|---------|----------------|--------|-------|
| Dense SwiGLU FFN | ✅ Fully supported | None | Existing `FusedGateUpGEMM` + `SwiGLU` stages |
| Standard GQA | ✅ Fully supported | None | Existing attention kernels handle GQA |
| RMSNorm | ✅ Fully supported | None | Existing `RMSNormStage` |
| Standard 1D RoPE | ✅ Fully supported | None | Existing `RoPEStage` |
| Embedding stage | ✅ Fully supported | None | Existing `EmbeddingStage` |
| Residual add | ✅ Fully supported | None | Existing `ResidualAddStage` |
| MoE routing | ⚠️ Stub exists | Medium | `RouterStage`, `ExpertStage`, `CombineStage` stubs in `compute_stages/stages/` |
| Sliding window attn | ⚠️ Partial | Low | `window_size` param exists in kernel API (`StageSpec.window_size`) but defaults to -1 (disabled) |
| Partial RoPE | ❌ Not supported | Medium | Need to split head dims and only apply RoPE to first N dims |
| mRoPE (3D) | ❌ Not supported | High | New RoPE implementation for multimodal. **Defer for text-only**. |
| Attention output gate | ❌ Not supported | Medium | New stage to apply `sigmoid(gate) ⊙ attn_output` |
| Different head_dim per layer | ❌ Not supported | Medium | Config and buffer allocation changes |
| Per-layer dispatch | ❌ Not supported | High | Core graph schema change (already planned in GDN plan) |
| Tied embeddings | ❌ Not supported | Low | Weight loader change + LM head pointer aliasing |
| GDN recurrence | ❌ Not supported | High | New kernel (planned in GDN plan Phases 3-6) |
| Short conv1d | ❌ Not supported | Medium | New kernel (planned in GDN plan Phase 5) |
| Gated RMSNorm | ❌ Not supported | Low | Small extension to existing RMSNorm |
| GDN state tensor | ❌ Not supported | Medium | New tensor type (planned in GDN plan Phase 3) |
| Dual cache system | ❌ Not supported | High | Per-layer-type cache management |
| MoE expert bank (3D tensors) | ❌ Not supported | High | 3D quantized tensor support, expert dispatch kernel |
| MoE shared expert | ❌ Not supported | Medium | SwiGLU + sigmoid gating (reuses existing dense FFN stages) |
| MoE TopK router | ❌ Not supported | Medium | Softmax + TopK selection + expert mapping |
| MoE expert parallelism | ❌ Not supported | High | All-to-all communication for expert routing across ranks |
| Separate LM head weight | ❌ Not supported | Low | MoE models use separate `output.weight` (not tied to embedding) |
| MTP self-speculative | ❌ Not supported | High | Planned in MTP plan (Phases 9-15) |
| Vision encoder | ❌ Not supported | Very High | Full ViT + spatial merge. **Defer**. |

### Code Needs Modification

| File/Component | Changes Required |
|----------------|------------------|
| `src/v2/models/qwen/Qwen2Schema.h` | New `Qwen3NextSchema` with per-layer dispatch, variable head dims |
| `src/v2/models/qwen/Qwen2Graph.h` | New `Qwen3NextGraph` with hybrid layer building; conditional dense vs MoE FFN |
| `src/v2/loaders/` | Parse `attn_type` array, `linear_*` dimensions, MoE params, tied embedding flag from GGUF metadata; 3D tensor support for expert weights |
| `src/v2/execution/compute_stages/stages/RoPEStage` | Add `partial_rotary_factor` support |
| `src/v2/execution/compute_stages/stages/Router*` | Activate MoE stubs: implement softmax routing, top-k selection, expert dispatch |
| `src/v2/tensors/TensorKernels.h` | Add `ITensorGatedDeltaNet`, `ITensorShortConvolution` interfaces |
| `src/v2/tensors/TensorClasses.h` | Support 3D quantized tensors for expert `gate_up_proj` and `down_proj` |
| `src/v2/kernels/cpu/attention/` | Add partial RoPE support to existing kernels |
| KV cache manager | Support dual-mode: KV cache for FA layers, GDN state for linear layers |

---

## MTP Confirmation

The MTP project plan (Phases 9–15) is **fully aligned** with Qwen3.5:

| MTP Plan Assumption | Qwen3.5 Config | Status |
|---------------------|----------------|--------|
| D=1 (single MTP depth) | `mtp_num_hidden_layers: 1` | ✅ Confirmed |
| Shared embeddings | `mtp_use_dedicated_embeddings: false` | ✅ Confirmed |
| Self-speculative decoding | Supported via SGLang and vLLM | ✅ Confirmed |

### MTP Serving (from model card)

Qwen3.5 supports MTP inference via:

**SGLang**:
```bash
python -m sglang.launch_server --model Qwen/Qwen3.5-4B --speculative-algo NEXTN --speculative-num-draft-tokens 1
```

**vLLM**:
```python
llm = LLM(model="Qwen/Qwen3.5-4B", speculative_config={"method": "qwen3_next_mtp", "num_speculative_tokens": 1})
```

**Note**: The CausalLM class strips MTP weights by default (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`). MTP weights are only loaded when speculative decoding is explicitly enabled. The MTP plan already accounts for this.

---

## Vision Encoder (Deferred)

| Feature | 4B Dense | 35B-A3B MoE | 122B-A10B MoE |
|---------|----------|-------------|---------------|
| Encoder type | Qwen2.5-VL ViT | Qwen2.5-VL ViT | Qwen2.5-VL ViT |
| Depth | 24 | 27 | 27 |
| Hidden size | 1024 | 1152 | 1152 |
| Heads | 16 | 16 | 16 |
| Patch size | 16 x 16 | 16 x 16 | 16 x 16 |
| Temporal patch size | 1 | 2 | 2 |
| Spatial merge | 2 x 2 | 2 x 2 | 2 x 2 |
| RoPE | 2D spatial + 1D temporal | Same | Same |

### Text-Only Inference Path

`Qwen3_5ForCausalLM` (text-only) ignores vision entirely:
- `_keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]`
- The forward pass accepts `input_ids` only (no `pixel_values`)
- All visual components are simply not loaded

**Recommendation**: Support text-only inference first. Add vision support as a separate future project.

---

## Recommended Plan Updates

### Updates to GDN Project Plan (Phases 1-8)

#### Phase 1 (Python Reference): No changes needed
The Python reference should use the actual HuggingFace `Qwen3_5GatedDeltaNet` implementation for parity testing.

#### Phase 2 (GGUF Weight Loading): Update required

1. **Separate projections**: Load 4 separate GDN weight matrices (`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`) instead of 2 fused matrices. This is simpler.
2. **Tied embeddings (dense)**: Add aliasing so `output.weight` -> `token_embd.weight`. No separate LM head weight.
3. **Separate embeddings (MoE)**: Load separate `output.weight` for MoE models where `tie_word_embeddings: false`.
4. **Per-layer metadata**: Parse `layer_types` array from GGUF metadata to determine which layers are GDN vs full attention.
5. **MoE weights**: Parse `num_experts`, `moe_intermediate_size`, `shared_expert_intermediate_size` from metadata. Load 3D expert tensors.

#### Phase 3 (New Tensor Types): Minor update

1. **Single conv state**: Replace 3 separate conv state buffers (`conv_state_q/k/v`) with a single `conv_state_qkv` buffer of shape `(Q_dim + K_dim + V_dim, kernel_size - 1) = (8192, 3)`.

#### Phase 4 (Graph Schema and Stages): Major update

1. **`Qwen3NextGraphConfig`**: Add per-layer-type dimension fields:
   ```cpp
   // Full attention dimensions
   int fa_head_dim = 256;
   int fa_n_kv_heads = 4;

   // GDN dimensions
   int gdn_key_head_dim = 128;
   int gdn_value_head_dim = 128;
   int gdn_n_key_heads = 16;
   int gdn_n_value_heads = 32;

   // Shared
   float partial_rotary_factor = 0.25;
   bool attn_output_gate = true;
   bool tie_word_embeddings = true;
   ```

2. **New stage: `AttentionOutputGateStage`**: Applies `sigmoid(gate) * attn_output` for full attention layers. The gate comes from the Q projection's extra width.

3. **Modified `RoPEStage`**: Support `partial_rotary_factor` -- apply RoPE to only the first `rope_dim = head_dim * partial_rotary_factor` dimensions, pass through the rest.

4. **Update GDN projection stage**: Use 4 separate GEMMs instead of 2 fused + deinterleave.

5. **Update conv stage**: Single conv on concatenated QKV, not 3 separate convs.

#### Phase 5 (CPU Reference Kernels): Minor update

1. Add partial RoPE reference kernel.
2. Ensure GDN reference kernel handles the value head expansion (16 key heads -> 32 value heads with repeat_interleave).

#### Phase 6 (Optimized Kernels): No structural changes
Optimization of GDN and hybrid attention kernels follows the same plan. No changes needed from the Qwen3.5 research.

#### Phase 7 (MoE Integration): NOW CONFIRMED REQUIRED

Two real models validate this phase: Qwen3.5-35B-A3B and Qwen3.5-122B-A10B. Phase 7 implementation details and weight layout are documented in the [MoE Architecture section](#moe-architecture-35b-a3b-and-122b-a10b) above. Key implementation items:

1. **Router stage**: `TopKRouter` -- softmax over 256 experts, select top-8, normalize weights.
2. **Expert dispatch**: Route tokens to selected experts, execute SwiGLU per expert on 3D weight tensors.
3. **Shared expert**: Dense SwiGLU (reuses existing FFN stages) + sigmoid gating (`sigmoid(linear(h)) * shared_output`).
4. **Expert combination**: Weighted sum of routed expert outputs + shared expert output.
5. **3D tensor support**: Expert `gate_up_proj` is `[256, 2*intermediate, hidden]` -- GGUF loader must handle 3D quantized tensors.
6. **Expert parallelism**: For TP, distribute experts across ranks (256/TP_degree per rank) with all-to-all communication.
7. **Weight streaming**: With 256 experts per layer at ~1.5 GB/layer (FP16), weight streaming is critical for fitting MoE models on limited VRAM.

**Recommended approach**: Implement MoE for Qwen3.5-35B-A3B first (smaller expert dim=512), then scale to 122B-A10B.

**Note**: Dense 4B remains the first end-to-end target (Phase 8). Phase 7 MoE can be developed in parallel or after Phase 8 validation.

#### Phase 8 (End-to-End Validation): Dual Target

- **Primary target**: `Qwen/Qwen3.5-4B` (dense) -- validates all attention, GDN, RoPE, and gating work.
- **Secondary target**: `Qwen/Qwen3.5-35B-A3B` (MoE) -- validates MoE integration.

---

## Priority Order for Text-Only Inference

### Phase 0 Prerequisites (Before GDN plan)

| ID | Task | Effort | Notes |
|----|------|--------|-------|
| 0a | Partial RoPE in existing RoPE stage | Medium | Foundation for both layer types |
| 0b | Tied/separate word embeddings in weight loader | Low | Affects model loading for both dense and MoE |
| 0c | Attention output gate stage | Medium | Needed by full attention layers |
| 0d | Per-layer config in graph schema | Medium | Foundation for hybrid dispatch |
| 0e | Sliding window attention activation | Low | `window_size` param already exists |

### Phase 1-6: GDN Implementation (Core)

Follow the GDN plan with the updates listed above. Key changes:
- Separate projections (simpler)
- Single combined conv (simpler)
- Variable head dimensions per layer type

### Phase 7: MoE Integration

Now a validated requirement. Implement for 35B-A3B first:
1. TopK router (softmax + top-8 selection)
2. Expert dispatch with 3D weight tensors
3. Shared expert (reuse dense FFN stages) + sigmoid gating
4. Expert combination (weighted sum + shared)
5. Expert parallelism for TP

### Phase 8: End-to-End Validation

**Primary target**: Qwen3.5-4B (dense) -- validates all attention, GDN, RoPE, gating work.
**Secondary target**: Qwen3.5-35B-A3B (MoE) -- validates MoE integration.

### Phase 9-15: MTP Implementation

Follow the MTP plan as-is. Applicable to all three variants.

### Future: Vision Encoder

Add VLM support in a later project. Note MoE models have a slightly different vision encoder (depth=27, hidden=1152, temporal_patch_size=2) compared to dense (depth=24, hidden=1024, temporal_patch_size=1). Requires:
- ViT implementation (24-27 layers, configurable depth)
- Spatial merge module (2x2 pixel shuffle)
- 3D RoPE for visual tokens
- Cross-attention between visual and text tokens

---

## Appendix: Full Qwen3.5 Family Comparison

| Feature | Qwen2.5 (Current) | Qwen3.5-4B (Dense) | Qwen3.5-35B-A3B (MoE) | Qwen3.5-122B-A10B (MoE) |
|---------|-------------------|---------------------|------------------------|--------------------------|
| model_type | qwen2 | qwen3_5 | qwen3_5_moe | qwen3_5_moe |
| Attention | Standard GQA | Hybrid GDN (75%) + FA (25%) | Hybrid GDN (75%) + FA (25%) | Hybrid GDN (75%) + FA (25%) |
| FFN | Dense SwiGLU | Dense SwiGLU (9216) | MoE: 256x512 + shared 512 | MoE: 256x1024 + shared 1024 |
| Hidden size | 896-3584 | 2560 | 2048 | 3072 |
| Layers | 24-28 | 32 | 40 | 48 |
| GDN key heads | N/A | 16 | 16 | 16 |
| GDN value heads | N/A | 32 | 32 | 64 |
| FA Q heads | Varies | 20 | 16 | 32 |
| FA KV heads | Varies | 4 | 2 | 2 |
| Head dim (FA) | 128 | 256 | 256 | 256 |
| Head dim (GDN) | N/A | 128 (k), 128 (v) | 128 (k), 128 (v) | 128 (k), 128 (v) |
| KV cache | All layers | 25% of layers (FA only) | 25% of layers (FA only) | 25% of layers (FA only) |
| RoPE | Standard, full | Partial (0.25) + mRoPE | Partial (0.25) + mRoPE | Partial (0.25) + mRoPE |
| Output gate | None | sigmoid(gate) * output | sigmoid(gate) * output | sigmoid(gate) * output |
| Embeddings | Separate | **Tied** | **Separate** | **Separate** |
| Expert dim | N/A | N/A | 512 | 1024 |
| Shared expert dim | N/A | N/A | 512 | 1024 |
| MTP | None | D=1, shared embeddings | D=1, shared embeddings | D=1, shared embeddings |
| Vision | None | ViT (depth=24, hidden=1024) | ViT (depth=27, hidden=1152) | ViT (depth=27, hidden=1152) |
| Context | 32K/128K | 262K to 1M | 262K to 1M | 262K to 1M |
| Vocabulary | 151,936 | 248,320 | 248,320 | 248,320 |
| Total params | 0.5B-72B | ~4B | ~35B | ~122B |
| Active params | Same | ~4B | ~3B | ~10B |