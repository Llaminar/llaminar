# Qwen 3.5 Multi-Token Prediction (MTP) Project Plan

**Date**: 2025-07-27 (initial) / 2026-04-10 (updated with confirmed Qwen 3.5 architecture)  
**Status**: Research Complete — Architecture Confirmed — Implementation Planned  
**Companion Document**: `docs/v2/cleanup/QWEN3NEXT_GDN_PROJECT_PLAN.md` (GDN attention)  
**Motivation**: Multi-Token Prediction enables self-speculative decoding, where the model's own auxiliary MTP heads draft future tokens that the main model verifies. DeepSeek-V3 achieves **1.8× TPS** with an 85-90% acceptance rate using this technique. Qwen 3.5 confirms MTP adoption with `mtp_num_hidden_layers: 1` across all dense model sizes. Llaminar V2 currently has zero MTP infrastructure.

**Confirmed Qwen 3.5 MTP Config** (from [HuggingFace config.json](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json)):
| Key | Value | Notes |
|-----|-------|-------|
| `mtp_num_hidden_layers` | 1 | D=1, predicts 1 extra token |
| `mtp_use_dedicated_embeddings` | false | Shares main model embedding |
| `tie_word_embeddings` | true | LM head = embedding.T (no separate lm_head weight) |
| Model sizes with MTP | 0.8B, 4B, 8B, 14B, 32B | All dense Qwen 3.5 models |

---

## Table of Contents

- [Background: What is Multi-Token Prediction?](#background-what-is-multi-token-prediction)
- [MTP Architecture (DeepSeek-V3 Reference)](#mtp-architecture-deepseek-v3-reference)
- [MTP at Inference: Self-Speculative Decoding](#mtp-at-inference-self-speculative-decoding)
- [Current State Analysis: Llaminar V2 Readiness](#current-state-analysis-llaminar-v2-readiness)
- [Component-Level Impact Analysis](#component-level-impact-analysis)
- [Phase 9: KV Cache Rollback](#phase-9-kv-cache-rollback)
- [Phase 10: MTP Head Weights and Loading](#phase-10-mtp-head-weights-and-loading)
- [Phase 11: MTP Graph Stages](#phase-11-mtp-graph-stages)
- [Phase 12: Self-Speculative Decode Loop](#phase-12-self-speculative-decode-loop)
- [Phase 13: LM Head Multi-Position Logits](#phase-13-lm-head-multi-position-logits)
- [Phase 14: Verification and Acceptance](#phase-14-verification-and-acceptance)
- [Phase 15: End-to-End MTP Validation](#phase-15-end-to-end-mtp-validation)
- [Integration with GDN (Hybrid Model)](#integration-with-gdn-hybrid-model)
- [FastMTP Optimization (Vocabulary Trimming)](#fastmtp-optimization-vocabulary-trimming)
- [Risk Assessment](#risk-assessment)
- [External References](#external-references)

---

## Background: What is Multi-Token Prediction?

### Two Approaches to MTP

There are two major approaches to multi-token prediction in LLMs:

| Approach | Paper | Key Idea | Inference Use |
|----------|-------|----------|---------------|
| **Parallel MTP** (Meta) | arXiv:2404.19737 | n **independent** output heads on shared trunk | Self-speculative decoding |
| **Sequential MTP** (DeepSeek-V3) | arXiv:2412.19437 | D **sequential** modules maintaining causal chain | Self-speculative decoding |

**Meta's approach** (Gloeckle et al., 2024): Attaches n independent linear heads to the last transformer layer. Each head predicts the next token at a different future position. Simple but doesn't maintain causal dependencies between predictions.

**DeepSeek-V3's approach** (DeepSeek-AI, 2024): Attaches D sequential MTP modules, each containing a small transformer block. Each depth k receives the representation from depth k−1, allowing predictions to be causally dependent. This is the **canonical production implementation** and the basis for this plan.

### Why MTP Matters for Inference

During **training**, MTP serves as an auxiliary objective that densifies gradient signals and improves data efficiency. The MTP heads can be discarded at inference with no loss to the base model.

During **inference**, the MTP heads are repurposed for **self-speculative decoding** — the model uses its own MTP heads to draft future tokens, then verifies them with the main model in a single forward pass. This eliminates the need for a separate draft model.

**DeepSeek-V3 results** (Section 5.4.3 of the technical report):
> "The acceptance rate of the second token prediction ranges between 85% and 90% across various generation topics, demonstrating consistent reliability. This high acceptance rate enables DeepSeek-V3 to achieve a significantly improved decoding speed, delivering **1.8 times TPS** (Tokens Per Second)."

### MTP vs External Speculative Decoding

| Property | MTP (Self-Speculative) | External Draft Model |
|----------|----------------------|---------------------|
| Draft model | Same model's MTP heads | Separate smaller model |
| Additional memory | Small (1 transformer block per depth) | Full separate model |
| Weight sharing | Shares embedding + output head | No sharing |
| Acceptance rate | 85-90% (DeepSeek-V3, D=1) | Varies by model pair |
| Implementation complexity | Moderate — integrated | High — two separate models |
| Latency per draft step | Very low (1 shallow block) | Higher (full small model) |

---

## MTP Architecture (DeepSeek-V3 Reference)

### Mathematical Formulation

The k-th MTP module (for prediction at depth k) consists of:

1. **Shared embedding layer** `Emb(·)` — same as main model
2. **Shared output head** `OutHead(·)` — same as main model (LM head)
3. **Per-depth transformer block** `TRM_k(·)` — unique per depth
4. **Per-depth projection matrix** `M_k ∈ ℝ^{d × 2d}` — unique per depth

**Step 1 — Combine representations** (Equation 21 from DeepSeek-V3 paper):

$$h'^k_i = M_k \left[ \text{RMSNorm}(h^{k-1}_i) ; \text{RMSNorm}(\text{Emb}(t_{i+k})) \right]$$

Where:
- $h^{k-1}_i$ is the representation at depth k−1 for position i (when k=1, this is the main model's hidden state)
- $t_{i+k}$ is the (i+k)-th token from ground truth (training) or from the previous depth's prediction (inference)
- $[·;·]$ denotes concatenation along the hidden dimension
- The input to $M_k$ is therefore a 2d-dimensional vector

**Step 2 — Transform through the depth's block** (Equation 22):

$$h^k_{1:T-k} = \text{TRM}_k(h'^k_{1:T-k})$$

The transformer block is a standard attention + FFN block, applying attention over the sequence. The sequence is shortened by k positions (since we need (i+k)-th input tokens).

**Step 3 — Predict via shared output head** (Equation 23):

$$P^k_{i+k+1} = \text{OutHead}(h^k_i)$$

The shared output head (same as main model's LM head) maps each representation to a probability distribution over the vocabulary.

### DeepSeek-V3 Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| MTP depth D | 1 | Predicts 1 additional token beyond next token |
| Projection matrix M_k | d × 2d | d=7168 for DeepSeek-V3 |
| TRM_k | Full transformer block (MLA + MoE) | Shares architecture with main layers |
| MTP loss weight λ | 0.3 (first 10T tokens) → 0.1 (remaining) | Training only |
| Embedding sharing | Yes — Emb and OutHead shared with main model | Physical parameter sharing |

### Confirmed Qwen 3.5 Hyperparameters

| Parameter | 0.8B | 4B | Notes |
|-----------|------|-----|-------|
| MTP depth D | 1 | 1 | All Qwen 3.5 dense models |
| d_model | 1024 | 2048 | Hidden dimension |
| n_heads | 16 | 16 | Attention heads |
| n_kv_heads | 4 | 4 | GQA key/value heads |
| head_dim | 64 | 128 | Per-head dimension |
| intermediate_size | 2816 | 8192 | FFN intermediate |
| **MTP block type** | **Full Attention** | **Full Attention** | NOT GDN — always softmax attention with gated Q, QK norm, partial RoPE |
| Gated Q | Yes (2× Q width) | Yes (2× Q width) | Same as main model FA layers |
| QK norm | Per-head RMSNorm | Per-head RMSNorm | q_norm + k_norm weights present |
| Projection M_k | (d, 2d) | (d, 2d) | Named `mtp.fc.weight` |
| Extra norm | `mtp.norm.weight` | `mtp.norm.weight` | Final RMSNorm after transformer block, before LM head |
| Embedding sharing | Yes | Yes | `mtp_use_dedicated_embeddings: false` |
| LM head sharing | Yes | Yes | `tie_word_embeddings: true` |

> **Key finding**: The MTP transformer block is always a **Full Attention** layer, architecturally identical to the main model's FA layers (gated Q, QK norm, partial RoPE). It is NOT a GDN layer. This means the existing FA stage infrastructure can be directly reused for MTP, and the MTP block has its own small KV cache (1 layer).

### Weight Inventory for MTP Module (D=1)

#### Confirmed HuggingFace Safetensor Names (Qwen 3.5)

The following weight names are **confirmed** from the [Qwen3.5-0.8B safetensors index](https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/model.safetensors.index.json):

| HF Safetensor Name | Shape (0.8B, d=1024) | Purpose |
|----|-------|--------|
| `mtp.fc.weight` | (d, 2d) = (1024, 2048) | Projection: concat[norm(h_main); norm(emb)] → d |
| `mtp.pre_fc_norm_hidden.weight` | (d,) = (1024,) | RMSNorm on main model hidden state |
| `mtp.pre_fc_norm_embedding.weight` | (d,) = (1024,) | RMSNorm on token embedding |
| `mtp.norm.weight` | (d,) = (1024,) | Final RMSNorm after transformer block |
| `mtp.layers.0.input_layernorm.weight` | (d,) = (1024,) | Pre-attention layer norm |
| `mtp.layers.0.self_attn.q_proj.weight` | (n_heads×hd×2, d) = (2048, 1024) | Q projection (2× for gated Q) |
| `mtp.layers.0.self_attn.k_proj.weight` | (n_kv×hd, d) = (256, 1024) | K projection |
| `mtp.layers.0.self_attn.v_proj.weight` | (n_kv×hd, d) = (256, 1024) | V projection |
| `mtp.layers.0.self_attn.o_proj.weight` | (d, n_heads×hd) = (1024, 1024) | Output projection |
| `mtp.layers.0.self_attn.q_norm.weight` | (hd,) = (64,) | Per-head Q RMSNorm |
| `mtp.layers.0.self_attn.k_norm.weight` | (hd,) = (64,) | Per-head K RMSNorm |
| `mtp.layers.0.post_attention_layernorm.weight` | (d,) = (1024,) | Pre-FFN layer norm |
| `mtp.layers.0.mlp.gate_proj.weight` | (inter, d) = (2816, 1024) | FFN gate |
| `mtp.layers.0.mlp.up_proj.weight` | (inter, d) = (2816, 1024) | FFN up |
| `mtp.layers.0.mlp.down_proj.weight` | (d, inter) = (1024, 2816) | FFN down |

**Shared weights** (no MTP-specific copies):
| Shared Weight | Main Model Name | Notes |
|---------------|----------------|-------|
| Embedding | `model.language_model.embed_tokens.weight` | `mtp_use_dedicated_embeddings: false` |
| LM Head | (tied to embedding) | `tie_word_embeddings: true` — no separate `lm_head.weight` |

**Total MTP-specific parameters** (0.8B model): ~7.7M params (≈1 FA layer + projection + 3 norms) = ~4% model overhead.

> **Note on GGUF naming**: The exact GGUF tensor names will depend on how ik_llama.cpp/llama.cpp standardizes the conversion. The llama.cpp PR #20700 uses names like `mtp.fc.weight`, `mtp.layers.0.self_attn.q_proj.weight` etc., closely mirroring the HF naming. The mapping layer should handle both conventions.

#### Previous Speculative Names (Superseded)

The following names were previously estimated from the DeepSeek-V3 architecture and are now **superseded** by the confirmed names above:

| Previously Estimated | Actual HF Name |
|---------------------|---------|
| `mtp.0.proj.weight` | `mtp.fc.weight` |
| `mtp.0.norm_main.weight` | `mtp.pre_fc_norm_hidden.weight` |
| `mtp.0.norm_emb.weight` | `mtp.pre_fc_norm_embedding.weight` |
| `mtp.0.block.attn_norm.weight` | `mtp.layers.0.input_layernorm.weight` |
| `mtp.0.block.attn.q_proj.weight` | `mtp.layers.0.self_attn.q_proj.weight` |
| `mtp.0.block.ffn_norm.weight` | `mtp.layers.0.post_attention_layernorm.weight` |
| `mtp.0.block.ffn.gate_proj.weight` | `mtp.layers.0.mlp.gate_proj.weight` |
| (not anticipated) | `mtp.norm.weight` (final norm before LM head) |
| (not anticipated) | `mtp.layers.0.self_attn.q_norm.weight` (per-head QK norm) |
| (not anticipated) | `mtp.layers.0.self_attn.k_norm.weight` (per-head QK norm) |

### Data Flow Diagram

```
Main Model Forward Pass (Qwen 3.5 hybrid GDN + FA)
────────────────────────────────────────────────────
tokens → Embedding → [GDN/FA Block × N] → FinalNorm → h_main
    (GDN layers 0,1,2,4,5,6,8,9,10,...)                  │
    (FA  layers 3,7,11,15,19,23)                          │ (last hidden state)
                                                            ▼
                                                      ┌──────────┐
                                                      │ LM Head  │ → logits_0 → sample → token_1
                                                      │ (shared) │
                                                      └──────────┘
                                                            │
MTP Module (Depth 1) — Confirmed Qwen 3.5 architecture     │
───────────────────────────────────────────────────────     ▼
                                                 pre_fc_norm_hidden(h_main)
                                                            │
token_1 → Embedding(token_1) → pre_fc_norm_embedding ┐     │
                                                      ▼     ▼
                                              ┌──────────────────┐
                                              │  Concatenate     │
                                              │  [norm_h ; norm_e]│
                                              └────────┬─────────┘
                                                       ▼
                                              ┌──────────────────┐
                                              │  mtp.fc          │ (2d → d)
                                              │  Projection      │
                                              └────────┬─────────┘
                                                       ▼
                                              ┌──────────────────┐
                                              │  FA Block        │ Full Attention layer:
                                              │  (mtp.layers.0)  │  • Gated Q (2× width)
                                              │                  │  • Per-head QK RMSNorm
                                              │                  │  • Partial RoPE
                                              │                  │  • SwiGLU FFN
                                              │                  │  • Own KV cache (1 layer)
                                              └────────┬─────────┘
                                                       ▼
                                              ┌──────────────────┐
                                              │  mtp.norm        │ Final RMSNorm
                                              └────────┬─────────┘
                                                       ▼
                                              ┌──────────────────┐
                                              │  LM Head (shared)│ → logits_1 → sample → token_2
                                              └──────────────────┘
```

For D > 1, additional depths chain: depth 2 takes h^1 from depth 1, combines with Emb(token_2), projects through M_2, runs TRM_2, mtp.norm_2, and produces token_3 via the shared LM head. (All current Qwen 3.5 models use D=1.)

> **Note**: HuggingFace transformers ignores MTP weights by default (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`). The MTP module is only used for self-speculative decoding, not standard greedy/sampling generation.

---

## MTP at Inference: Self-Speculative Decoding

### The Decode Loop with MTP

The MTP-enabled decode loop replaces the simple "generate one token per step" with a **draft-then-verify** cycle:

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: DRAFT — Generate D+1 candidate tokens                  │
│                                                                  │
│   1a. Run main model forward(last_token) → h_main, logits_0     │
│   1b. Sample token_1 from logits_0                               │
│   1c. For k = 1..D:                                              │
│       - Combine h^{k-1} with Emb(predicted_token_k)             │
│       - Project through M_k                                      │
│       - Run TRM_k → h^k                                         │
│       - logits_k = LMHead(h^k)                                  │
│       - Sample token_{k+1} from logits_k                        │
│                                                                  │
│   Result: draft = [token_1, token_2, ..., token_{D+1}]          │
│           KV cache has been updated with last_token              │
├─────────────────────────────────────────────────────────────────┤
│ Step 2: VERIFY — Run main model on draft tokens                 │
│                                                                  │
│   2a. Run main model forward(draft[0..D]) with seq_len=D+1      │
│       → logits at ALL D+1 positions                             │
│   2b. KV cache now has entries for all draft tokens              │
├─────────────────────────────────────────────────────────────────┤
│ Step 3: ACCEPT — Compare and decide                             │
│                                                                  │
│   3a. For each position i = 0..D:                               │
│       - If greedy: does argmax(verify_logits[i]) == draft[i+1]? │
│       - If stochastic: speculative sampling acceptance           │
│   3b. Find first rejection at position r                         │
│   3c. Accept tokens 0..r-1                                       │
│   3d. Sample correction token from verify_logits[r]              │
│   3e. last_token = correction_token                              │
├─────────────────────────────────────────────────────────────────┤
│ Step 4: ROLLBACK — Clean up KV cache                            │
│                                                                  │
│   4a. If r < D: truncate KV cache to remove entries for          │
│       rejected draft tokens (positions r+1..D)                   │
│   4b. accepted_count = r + 1 (r accepted + 1 correction)        │
│                                                                  │
│ Total tokens produced this step: accepted_count                  │
│ Expected: 1.8-1.9 tokens/step for D=1 with 85-90% acceptance   │
└─────────────────────────────────────────────────────────────────┘
```

### Greedy Verification Algorithm

For greedy decoding (temperature=0), verification is simple:

```
accepted = 0
for i in 0..D:
    main_token = argmax(verify_logits[i])
    if main_token == draft[i]:
        accepted += 1
    else:
        correction = main_token
        break

if all accepted:
    # All D draft tokens matched; sample one more from the last verify position
    correction = argmax(verify_logits[D])
    accepted = D + 1

output_tokens = draft[0:accepted] + [correction]
rollback KV cache to position (start + accepted + 1)
```

### Stochastic Verification (Speculative Sampling)

For sampling with temperature > 0, we use the speculative sampling algorithm from Leviathan et al. (2023):

```
for i in 0..D:
    p = softmax(verify_logits[i])   # main model distribution
    q = softmax(draft_logits[i])    # MTP head distribution (saved from draft step)

    # Accept with probability min(1, p[draft[i]] / q[draft[i]])
    r = uniform(0, 1)
    if r < min(1, p[draft[i]] / q[draft[i]]):
        accept draft[i]
    else:
        # Sample correction from adjusted distribution: max(0, p - q) / sum(max(0, p - q))
        p_adjusted = max(0, p - q)
        p_adjusted /= sum(p_adjusted)
        correction = sample(p_adjusted)
        break
```

This guarantees the output distribution is **identical** to the main model's distribution — MTP introduces zero quality degradation.

---

## Current State Analysis: Llaminar V2 Readiness

### Search for Existing MTP Infrastructure

A comprehensive search for `speculative|multi.?token|MTP|draft` across the codebase found **no existing MTP infrastructure**. The only relevant finding:

- `tests/v2/unit/kernels/cpu/attention/q8_1/jit/Test__JitWoProjection.cpp` (line 728): A test named `Causal_MultiToken_Decode` with comment "Multi-token decode (speculative decoding scenario)" — this tests the attention kernel with `seq_len=4` during decode with `position_offset=100`, confirming that **attention kernels already support multi-token decode queries**.

### Readiness Matrix

| Component | Current State | MTP Ready? | Gap |
|-----------|---------------|------------|-----|
| **Attention kernels** | Support seq_len > 1 decode (tested) | ✅ Yes | None |
| **Forward pass** (`IInferenceRunner::forward`) | Accepts arbitrary seq_len | ✅ Yes | None |
| **Graph cache** | Keyed by (seq_len, batch_size, device) | ✅ Yes | Different graphs for draft vs verify automatically |
| **FA layer stages** | Qwen 3.5 FA stages fully implemented in Qwen35Graph | ✅ Yes | MTP block reuses FA stage infrastructure directly |
| **Position tracking** | `positions[b] += seq_len` in forward | ⚠️ Partial | Needs rollback on rejection |
| **LM Head** | Only last-token logits when seq_len > 1 | ❌ No | Must compute ALL positions' logits for verification |
| **KV Cache** | No truncate/rollback method | ❌ No | Critical gap — needs `truncate()` for FA layers |
| **GDN Recurrent State** | No checkpoint/restore | ❌ No | Critical gap — state irreversibly mutated by rejected tokens |
| **Conv1d State** | No checkpoint/restore | ❌ No | Short conv state also needs rollback for GDN layers |
| **Sampler** | Returns single token | ⚠️ Partial | Needs multi-call or batch mode |
| **Decode loop** | Strictly 1 token per step | ❌ No | Needs draft-verify-accept cycle |
| **MTP head stages** | Don't exist | ❌ No | But FA stages can be reused (MTP block = FA block) |
| **Weight loading** | No MTP weight names in GGUF mapping | ❌ No | HF names confirmed; GGUF naming pending standardization |

---

## Component-Level Impact Analysis

### 1. OrchestrationRunner (Decode Loop)

**File**: `src/v2/execution/runner/OrchestrationRunner.cpp`

**Current**: `decodeStep()` calls `runner_->forward(&last_token_, 1)` — exactly 1 token per step.

**Required changes**:
- New method `speculativeDecodeStep()` implementing the draft-verify-accept cycle
- `generate()` calls `speculativeDecodeStep()` when MTP is enabled, falls back to `decodeStep()` otherwise
- Track accepted token count for throughput metrics
- Handle stop tokens appearing in any accepted position

### 2. DeviceGraphOrchestrator (Forward Pass)

**File**: `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp`

**Current**: Forward implementation (L2247-2416) auto-detects phase based on seq_len, updates positions by seq_len.

**Required changes**:
- Add `rollbackPositions(int count)` to undo position advances for rejected tokens
- Add `forwardMTPHead(const float* main_hidden_states, int draft_token, int depth)` for MTP module execution
- Expose `getLastHiddenStates()` to provide h_main to MTP heads (currently only logits are returned)

### 3. LMHeadStage (Logit Computation)

**File**: `src/v2/execution/compute_stages/stages/LMHeadStage.cpp`

**Current optimization** (L90-97):
```cpp
const int lm_m = (params_.seq_len > 1) ? 1 : params_.seq_len;
const int lm_activation_offset = (params_.seq_len > 1) ? (params_.seq_len - 1) : 0;
```

**Required changes**:
- Add a `compute_all_positions` flag to `LMHeadStage::Params`
- When the flag is set, use `lm_m = seq_len` and `lm_activation_offset = 0`
- Verification pass sets this flag; normal prefill keeps the optimization
- Output logits buffer must be (seq_len × vocab_size) instead of (1 × vocab_size)

### 4. IKVCache (Cache Rollback)

**File**: `src/v2/kernels/IKVCache.h`

**Current**: Interface has `append()`, `clear()`, `clear_sequence()`, `clear_layer()`. No rollback.

**Required new method**:
```cpp
/// Truncate KV cache to a shorter length, discarding entries beyond new_len.
/// Used by speculative decoding to remove rejected draft tokens.
/// @param layer    Layer index
/// @param seq_idx  Sequence index (for batched generation)
/// @param new_len  Number of tokens to keep (truncates to [0, new_len))
/// @return true on success
virtual bool truncate(int layer, int seq_idx, int new_len) = 0;
```

**Implementation in backends**:
- `CPURingKVCache`: Update the cached token count; the ring buffer data can stay (it will be overwritten)
- `CUDARingKVCache`: Same — update the head pointer; GPU memory contents don't need zeroing
- `CUDAPagedKVCache` (if exists): Free pages beyond new_len

### 5. Sampler

**File**: `src/v2/utils/Sampler.h`

**Required changes**: Minimal. The sampler can be called D+1 times per speculative decode step. Optionally add:
```cpp
/// Verify a draft token against main model logits using speculative sampling.
/// Returns the accepted token (may be the draft or a correction).
bool verify_draft_token(const float* main_logits, const float* draft_logits,
                        int draft_token, size_t vocab_size,
                        const SamplingParams& params,
                        int* accepted_token);
```

### 6. Graph Schema (MTP Stages)

**Current**: `Qwen2Schema.h` defines buffers and weight names for the standard model.

**New stages needed for MTP module** (per depth k):
| Stage | Description | Inputs | Outputs |
|-------|-------------|--------|---------|
| `MTPEmbedLookupStage` | Lookup embedding for draft token | draft_token, embedding_table | emb_vector |
| `MTPNormMainStage` | RMSNorm on main hidden states | h_main | norm_h_main |
| `MTPNormEmbStage` | RMSNorm on embedding | emb_vector | norm_emb |
| `MTPConcatStage` | Concatenate [norm_h; norm_emb] | norm_h_main, norm_emb | concat_2d |
| `MTPProjectionStage` | M_k projection (2d → d) | concat_2d | h_projected |
| `MTPTransformerBlockStage` | Self-attention + FFN | h_projected | h_depth_k |
| `MTPLMHeadStage` | Shared LM head on h_depth_k | h_depth_k, lm_head_weight | mtp_logits |

These can potentially be simplified into fewer stages for performance.

### 7. Weight Loading

**Current**: `WeightManager` maps GGUF tensor names to schema buffer names.

**New weight mappings needed** (for D=1, confirmed HF names):
| HF Safetensor Name | Schema Name | Shape |
|----|-------------|-------|
| `mtp.fc.weight` | `mtp_depth0_proj` | (d, 2d) |
| `mtp.pre_fc_norm_hidden.weight` | `mtp_depth0_norm_main` | (d,) |
| `mtp.pre_fc_norm_embedding.weight` | `mtp_depth0_norm_emb` | (d,) |
| `mtp.norm.weight` | `mtp_depth0_final_norm` | (d,) |
| `mtp.layers.0.input_layernorm.weight` | `mtp_depth0_attn_norm` | (d,) |
| `mtp.layers.0.self_attn.q_proj.weight` | `mtp_depth0_attn_q` | (n_heads×hd×2, d) — gated Q |
| `mtp.layers.0.self_attn.k_proj.weight` | `mtp_depth0_attn_k` | (n_kv_heads×hd, d) |
| `mtp.layers.0.self_attn.v_proj.weight` | `mtp_depth0_attn_v` | (n_kv_heads×hd, d) |
| `mtp.layers.0.self_attn.o_proj.weight` | `mtp_depth0_attn_o` | (d, n_heads×hd) |
| `mtp.layers.0.self_attn.q_norm.weight` | `mtp_depth0_q_norm` | (hd,) |
| `mtp.layers.0.self_attn.k_norm.weight` | `mtp_depth0_k_norm` | (hd,) |
| `mtp.layers.0.post_attention_layernorm.weight` | `mtp_depth0_ffn_norm` | (d,) |
| `mtp.layers.0.mlp.gate_proj.weight` | `mtp_depth0_ffn_gate` | (inter_dim, d) |
| `mtp.layers.0.mlp.up_proj.weight` | `mtp_depth0_ffn_up` | (inter_dim, d) |
| `mtp.layers.0.mlp.down_proj.weight` | `mtp_depth0_ffn_down` | (d, inter_dim) |

**Note**: The MTP block is architecturally identical to the main model's **Full Attention** layers (gated Q with 2× width, per-head QK RMSNorm, partial RoPE, SwiGLU FFN). The Q projection width is doubled to accommodate the gate.

---

## Phase 9: KV Cache Rollback

**Goal**: Add truncation capability to the KV cache, which is an absolute prerequisite for any speculative decoding.

### 9.1 Interface Extension

Add to `IKVCache.h`:
```cpp
/// Truncate the KV cache for a specific layer and sequence to a new length.
/// All entries at positions >= new_len are logically discarded.
/// @return true if truncation was successful
virtual bool truncate(int layer, int seq_idx, int new_len) = 0;

/// Truncate all layers for a sequence to a new length.
/// Convenience method that calls truncate() for each layer.
virtual bool truncateAll(int seq_idx, int new_len) {
    for (int l = first_layer_index(); l < first_layer_index() + n_layers(); ++l) {
        if (!truncate(l, seq_idx, new_len)) return false;
    }
    return true;
}
```

### 9.2 CPU Ring Buffer Implementation

For `CPURingKVCache`, truncation is trivial:
- The ring buffer stores K and V contiguously per layer
- `cached_tokens[layer][seq_idx]` tracks the current length
- Truncation: set `cached_tokens[layer][seq_idx] = new_len`
- No data movement needed — the stale entries will be overwritten on next append

### 9.3 CUDA Ring Buffer Implementation

For `CUDARingKVCache`:
- Same logic — update the cached token count
- If using graph capture with `advanceHead()`, the head pointer must be adjusted
- No `cudaMemset` needed for the stale entries

### 9.4 Tests

| Test | Description |
|------|-------------|
| `V2_Unit_KVCache_Truncate_Basic` | Append 10 tokens, truncate to 5, verify length |
| `V2_Unit_KVCache_Truncate_AppendAfter` | Truncate then append more, verify contents |
| `V2_Unit_KVCache_Truncate_AllLayers` | Verify `truncateAll()` affects all layers |
| `V2_Unit_KVCache_Truncate_ToZero` | Truncate to 0, verify equivalent to clear_sequence |
| `V2_Unit_KVCache_Truncate_Noop` | Truncate to current length, verify no effect |

### Deliverables

| Item | File/Location |
|------|--------------|
| Interface change | `src/v2/kernels/IKVCache.h` |
| CPU implementation | `src/v2/kernels/cpu/CPURingKVCache.cpp` |
| CUDA implementation | `src/v2/kernels/cuda/kv_cache/CUDARingKVCache.cu` |
| Unit tests | `tests/v2/unit/Test__KVCacheTruncate.cpp` |

---

## Phase 10: MTP Head Weights and Loading

**Goal**: Support loading MTP module weights from GGUF files, including the projection matrix, RMSNorm parameters, and the per-depth transformer block weights.

### 10.1 GGUF Weight Name Mapping

The HuggingFace safetensor names are now **confirmed** (see Weight Inventory above). The GGUF conversion naming depends on how ik_llama.cpp/llama.cpp standardizes the conversion from the `mtp.*` namespace. Implement a flexible mapping layer that handles both:

```cpp
// In WeightNameMapper or similar
struct MTPWeightMapping {
    int depth;              // MTP depth index (0 for D=1)
    std::string component;  // "fc", "pre_fc_norm_hidden", "pre_fc_norm_embedding", "norm",
                            // "layers.0.self_attn.*", "layers.0.mlp.*"
    std::string hf_name;    // Confirmed HF name: "mtp.fc.weight", etc.
    std::string gguf_name;  // GGUF name (TBD, likely mirrors HF naming)
};

// Known HF → GGUF mapping patterns from llama.cpp PR #20700:
//   mtp.fc.weight                     → blk.mtp.0.eh_proj.weight (or similar)
//   mtp.pre_fc_norm_hidden.weight     → blk.mtp.0.enorm.weight
//   mtp.pre_fc_norm_embedding.weight  → blk.mtp.0.hnorm.weight
//   mtp.norm.weight                   → blk.mtp.0.norm.weight
//   mtp.layers.0.self_attn.q_proj     → blk.mtp.0.attn_q.weight
//   etc.
```

### 10.2 Model Config Extension

Add MTP-related fields to the model configuration (values from confirmed HF config):

```cpp
struct Qwen35GraphConfig {
    // ... existing fields from Qwen35GraphConfig ...
    
    // MTP configuration (from HF config.json)
    int mtp_num_hidden_layers = 0; // 0 = no MTP, 1 = predict one extra token
    bool mtp_enabled = true;        // Can disable MTP heads even if weights are present
    bool mtp_use_dedicated_embeddings = false; // Qwen 3.5: always false (shared)
    
    // MTP depth shares the same d_model, n_heads, n_kv_heads, head_dim,
    // intermediate_size as the main model's FA layers.
    // The MTP TRM block matches the FA layer architecture exactly:
    //   gated Q, per-head QK RMSNorm, partial RoPE, SwiGLU FFN
};
```

### 10.3 Weight Sharing

The embedding table and LM head are **shared** between the main model and MTP heads (`mtp_use_dedicated_embeddings: false`, `tie_word_embeddings: true`). No duplication needed — the graph stages reference the same weight tensors:

| Shared Weight | Main Model Reference | MTP Reference |
|---------------|---------------------|---------------|
| `embed_tokens.weight` | Embedding stage | MTP embed lookup (for draft token) |
| `embed_tokens.weight` (as LM head) | `LMHeadStage` | MTP LM head (tied weights) |

### Deliverables

| Item | File/Location |
|------|--------------|
| Config extension | `src/v2/models/qwen35/Qwen35GraphConfigBuilder.h` (extend existing) |
| Weight mapping | `src/v2/loaders/MTPWeightMapper.h` |
| GGUF loader extension | `src/v2/loaders/GGUFLoader.cpp` (extend existing) |
| Unit test | `tests/v2/unit/Test__MTPWeightLoading.cpp` |

---

## Phase 11: MTP Graph Stages

**Goal**: Implement the compute stages that form the MTP module's forward pass.

### 11.1 Simplified Stage Design

Since the MTP transformer block is architecturally **identical to the main model's Full Attention layers** (gated Q, QK norm, partial RoPE, SwiGLU FFN), the implementation can heavily reuse existing FA stage infrastructure from `Qwen35Graph`. The MTP-specific stages are only the projection preamble and the final norm:

**MTP-specific stages** (new):
| Stage | Description | Inputs | Outputs |
|-------|-------------|--------|---------|
| `MTPEmbedLookupStage` | Lookup embedding for draft token | draft_token, embedding_table | emb_vector |
| `MTPNormMainStage` | RMSNorm on main hidden states (`pre_fc_norm_hidden`) | h_main | norm_h_main |
| `MTPNormEmbStage` | RMSNorm on embedding (`pre_fc_norm_embedding`) | emb_vector | norm_emb |
| `MTPConcatProjectStage` | Concatenate [norm_h; norm_emb] + project via `mtp.fc` (2d → d) | norm_h_main, norm_emb | h_projected |
| `MTPFinalNormStage` | RMSNorm via `mtp.norm` (after transformer block, before LM head) | h_depth_k | h_normed |

**Reused FA stages** (from existing `Qwen35Graph::buildFAAttentionGraph`):
| Stage | Notes |
|-------|-------|
| Pre-attention RMSNorm (`input_layernorm`) | Standard, already implemented |
| Fused QKV projection (gated Q, 2× width) | Identical to FA layers |
| Q-gate split | Identical to FA layers |
| Per-head QK RMSNorm | Identical to FA layers |
| Partial RoPE | Identical to FA layers |
| KV Cache (own 1-layer cache) | Separate cache instance for MTP |
| Attention compute | Standard scaled dot-product attention |
| Output gate + Wo projection | Identical to FA layers |
| Pre-FFN RMSNorm + SwiGLU FFN | Identical to FA layers |

Rather than 7+ individual MTP-specific stages, the fused approach below combines the preamble:

```cpp
class MTPForwardStage : public ComputeStage {
    struct Params {
        STAGE_PARAMS_COMMON_FIELDS;
        
        // Inputs
        const ITensor* main_hidden_states;  // h^{k-1} from main model
        int draft_token_id;                  // Token to embed and combine
        
        // MTP preamble weights
        const ITensor* proj_weight;          // mtp.fc: (d, 2d)
        const ITensor* norm_main_weight;     // mtp.pre_fc_norm_hidden
        const ITensor* norm_emb_weight;      // mtp.pre_fc_norm_embedding
        const ITensor* embedding_table;      // Shared embedding
        
        // Transformer block weights — same shape as main model FA layer
        // (reuse buildFAAttentionGraph infrastructure)
        const ITensor* attn_norm_weight;     // mtp.layers.0.input_layernorm
        // ... QKV, q_norm, k_norm, Wo, FFN weights ...
        
        // Post-block weights
        const ITensor* final_norm_weight;    // mtp.norm
        
        // Outputs
        ITensor* depth_hidden_states;        // h^k (before final norm)
        ITensor* depth_logits;               // Optional: logits for this depth
        
        // Config
        int d_model;
        int vocab_size;
    };
    
    bool execute() override {
        // 1. Embed draft token (lookup from shared embedding table)
        // 2. pre_fc_norm_hidden(h_main), pre_fc_norm_embedding(emb)
        // 3. Concatenate [norm_h; norm_emb] → (1, 2d)
        // 4. Project through mtp.fc (GEMM: 1×2d × 2d×d → 1×d)
        // ** 5-8 are standard FA layer stages (can delegate to buildFAAttentionGraph) **
        // 5. input_layernorm → FusedQKV → Q-gate split → QK norm → RoPE
        // 6. KV cache append → Attention compute → Output gate → Wo projection
        // 7. Residual add + post_attention_layernorm → SwiGLU FFN → Residual add
        // 8. mtp.norm (final RMSNorm)
        // 9. LM head (shared) → logits
    }
};
```

### 11.2 MTP Graph Building

Since the MTP block is a Qwen 3.5 FA layer, the graph builder can reuse `buildFAAttentionGraph()` directly:

```cpp
class Qwen35Graph {
    // ... existing graph building methods ...
    
    void buildMTPGraph(int depth) {
        // MTP preamble (new stages):
        // 1. Embed draft token → emb
        // 2. pre_fc_norm_hidden(h_main) + pre_fc_norm_embedding(emb)
        // 3. Concat [norm_h; norm_emb] → project through mtp.fc → h'
        
        // Transformer block (reuse existing FA infrastructure):
        // 4. buildFAAttentionGraph() with MTP-specific weights and separate KV cache
        //    (input_layernorm, gated Q, QK norm, partial RoPE, attention, output gate, Wo, FFN)
        
        // MTP postamble (new stages):
        // 5. mtp.norm (final RMSNorm before LM head)
        // 6. LMHeadStage (shared weight) → draft logits
    }
};
```

**Key implementation detail**: The MTP block's attention uses a **separate 1-layer KV cache** independent from the main model's KV cache. During speculation, the MTP KV cache only ever holds the draft sequence context. It must be truncated along with the main model's cache on rejection.

### 11.3 Key Design Decision: Separate Graph vs Integrated

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A: Separate graph** | MTP heads are a separate `ComputeGraph` executed after main forward | Clean separation, easy to disable | Extra overhead from graph dispatch |
| **B: Integrated graph** | MTP stages appended to main model graph | Single execution path, potential fusion | Complex graph, harder to conditionally skip |

**Recommendation**: Option A (separate graph). The MTP forward is lightweight (1 transformer block) and is only executed during the draft phase, not during verification. Keeping it separate makes the code cleaner and easier to conditionally enable/disable.

### Deliverables

| Item | File/Location |
|------|--------------|
| MTP preamble stages | `src/v2/execution/compute_stages/stages/MTPConcatProjectStage.h/.cpp` |
| MTP final norm stage | `src/v2/execution/compute_stages/stages/MTPFinalNormStage.h/.cpp` |
| MTP graph builder | `src/v2/models/qwen35/MTPGraph.h/.cpp` (reuses FA stage infra) |
| MTP KV cache (1-layer) | Separate `IKVCache` instance for MTP block |
| Unit test | `tests/v2/unit/Test__MTPForwardStage.cpp` |

---

## Phase 12: Self-Speculative Decode Loop

**Goal**: Implement the draft-verify-accept decode loop in OrchestrationRunner.

### 12.1 New Decode Step Method

```cpp
/// Perform one step of self-speculative decoding using MTP heads.
/// May produce 1 to (D+1) tokens per call.
/// @return GenerationResult with all accepted tokens
GenerationResult OrchestrationRunner::speculativeDecodeStep() {
    GenerationResult result;
    
    // === DRAFT PHASE ===
    // Step 1: Run main model forward on last_token_ (single token decode)
    runner_->forward(&last_token_, 1);
    
    // Step 2: Get main model's last hidden states (NOT just logits)
    const float* h_main = runner_->getLastHiddenStates();
    
    // Step 3: Sample first draft token from main model logits
    int draft_token_0 = sampleFromLogits(runner_->logits());
    
    // Step 4: Run MTP heads to generate D additional draft tokens
    std::vector<int> draft_tokens = {draft_token_0};
    std::vector<const float*> draft_logits_list;
    
    const float* h_prev = h_main;
    for (int k = 0; k < mtp_depth_; ++k) {
        auto [h_k, logits_k] = runner_->forwardMTPHead(h_prev, draft_tokens.back(), k);
        int draft_token = sampleFromLogits(logits_k);
        draft_tokens.push_back(draft_token);
        draft_logits_list.push_back(logits_k);
        h_prev = h_k;
    }
    
    // === VERIFY PHASE ===
    // Step 5: Run main model forward on ALL draft tokens (multi-token decode)
    //         This requires computing logits at ALL positions (not just last)
    runner_->setComputeAllPositionLogits(true);
    runner_->forward(draft_tokens.data(), draft_tokens.size());
    runner_->setComputeAllPositionLogits(false);
    
    const float* verify_logits = runner_->logits();  // (D+1) × vocab_size
    
    // === ACCEPT PHASE ===
    // Step 6: Compare draft tokens against verify logits
    int accepted = verifyDraftTokens(verify_logits, draft_tokens, draft_logits_list);
    
    // Step 7: Collect accepted tokens and correction
    // ... (see Phase 14 for details)
    
    // === ROLLBACK PHASE ===
    // Step 8: Truncate KV cache for rejected positions
    int final_position = current_position_ + accepted;
    runner_->truncateKVCache(final_position);
    runner_->setPosition(final_position);
    
    return result;
}
```

### 12.2 Forward Pass Extension: Hidden State Access

The current `IInferenceRunner::forward()` returns only logits. For MTP, we need access to the **last hidden states** before the LM head:

```cpp
class IInferenceRunner {
    // ... existing methods ...
    
    /// Get the last hidden states from the most recent forward pass.
    /// Shape: (seq_len, d_model). Used by MTP heads.
    virtual const float* getLastHiddenStates() const = 0;
    
    /// Run MTP head at specified depth on the given hidden states.
    /// Returns (hidden_states, logits) for the MTP depth.
    virtual MTPHeadOutput forwardMTPHead(const float* h_prev, int draft_token, int depth) = 0;
    
    /// Enable computing logits at all positions (not just last) in next forward.
    virtual void setComputeAllPositionLogits(bool enable) = 0;
    
    /// Truncate KV cache across all layers to the given length.
    virtual void truncateKVCache(int new_len) = 0;
};
```

### 12.3 Integration with generate()

```cpp
GenerationResult OrchestrationRunner::generate(
    const std::vector<int32_t>& prompt_tokens,
    int max_new_tokens,
    const SamplingParams& sampling)
{
    prefill(prompt_tokens);
    
    int tokens_generated = 0;
    while (tokens_generated < max_new_tokens) {
        GenerationResult step_result;
        
        if (mtp_enabled_ && mtp_depth_ > 0) {
            step_result = speculativeDecodeStep();
        } else {
            step_result = decodeStep();
        }
        
        for (int token : step_result.tokens) {
            result.tokens.push_back(token);
            tokens_generated++;
            if (isStopToken(token) || tokens_generated >= max_new_tokens) {
                goto done;
            }
        }
    }
    done:
    return result;
}
```

### Deliverables

| Item | File/Location |
|------|--------------|
| Speculative decode step | `src/v2/execution/runner/OrchestrationRunner.cpp` |
| IInferenceRunner extension | `src/v2/execution/local_execution/orchestrators/IInferenceRunner.h` |
| DeviceGraphOrchestrator impl | `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp` |
| Integration test | `tests/v2/integration/Test__SpeculativeDecodeLoop.cpp` |

---

## Phase 13: LM Head Multi-Position Logits

**Goal**: Allow the LM head to compute logits for all token positions, not just the last.

### 13.1 LMHeadStage Flag

Add a parameter to `LMHeadStage::Params`:

```cpp
struct Params {
    // ... existing fields ...
    
    /// When true, compute logits for ALL seq_len positions.
    /// Default false: only compute last token's logits (prefill optimization).
    bool compute_all_positions = false;
};
```

### 13.2 Modified Execute Logic

```cpp
void LMHeadStage::execute() {
    int lm_m, lm_activation_offset;
    
    if (params_.compute_all_positions) {
        // MTP verification: need logits at every position
        lm_m = params_.seq_len;
        lm_activation_offset = 0;
    } else if (params_.seq_len > 1) {
        // Normal prefill: only last token
        lm_m = 1;
        lm_activation_offset = params_.seq_len - 1;
    } else {
        // Normal decode: single token
        lm_m = 1;
        lm_activation_offset = 0;
    }
    
    // GEMM: (lm_m, d_model) × (vocab_size, d_model)^T → (lm_m, vocab_size)
    gemm->multiply_tensor(hidden_states, logits, lm_m, vocab_size, d_model,
                          ..., lm_activation_offset);
}
```

### 13.3 Logits Buffer Sizing

When `compute_all_positions` is enabled, the logits buffer must be (seq_len × vocab_size) instead of (1 × vocab_size). This needs to be handled in buffer allocation:

```cpp
// In graph buffer allocation:
int logits_rows = mtp_verification_mode ? max_draft_length : 1;
auto logits_buffer = allocate_fp32({logits_rows, vocab_size});
```

### Deliverables

| Item | File/Location |
|------|--------------|
| LMHeadStage modification | `src/v2/execution/compute_stages/stages/LMHeadStage.h/.cpp` |
| Logits buffer sizing | Graph builder / buffer allocation |
| Unit test | `tests/v2/unit/Test__LMHeadMultiPosition.cpp` |

---

## Phase 14: Verification and Acceptance

**Goal**: Implement the token verification and acceptance logic.

### 14.1 Acceptance Algorithm

```cpp
class SpeculativeAcceptor {
public:
    struct AcceptResult {
        int accepted_count;              // Number of draft tokens accepted
        int correction_token;            // Correction token (sampled from verify logits)
        std::vector<int> output_tokens;  // All output tokens (accepted + correction)
    };
    
    /// Greedy verification: accept while argmax matches
    AcceptResult verifyGreedy(
        const float* verify_logits,   // (D+1) × vocab_size
        const std::vector<int>& draft_tokens,  // D+1 draft tokens
        int vocab_size,
        int D);
    
    /// Stochastic verification: speculative sampling algorithm
    AcceptResult verifyStochastic(
        const float* verify_logits,   // (D+1) × vocab_size
        const float* draft_logits,    // D × vocab_size (from MTP heads, NOT for position 0)
        const std::vector<int>& draft_tokens,
        int vocab_size,
        int D,
        const SamplingParams& params);
};
```

### 14.2 Greedy Implementation

```cpp
AcceptResult SpeculativeAcceptor::verifyGreedy(
    const float* verify_logits,
    const std::vector<int>& draft_tokens,
    int vocab_size,
    int D)
{
    AcceptResult result;
    
    for (int i = 0; i <= D; ++i) {
        const float* logits_i = verify_logits + i * vocab_size;
        int main_token = argmax(logits_i, vocab_size);
        
        if (i < D && main_token == draft_tokens[i + 1]) {
            result.output_tokens.push_back(draft_tokens[i + 1]);
        } else {
            // Rejection at position i (or all accepted and i == D)
            result.accepted_count = i;
            result.correction_token = main_token;
            result.output_tokens.push_back(main_token);
            break;
        }
    }
    
    // If all D+1 positions matched (rare), still sample from last verify position
    if (result.output_tokens.size() == D + 1) {
        result.accepted_count = D;
    }
    
    return result;
}
```

### Deliverables

| Item | File/Location |
|------|--------------|
| Acceptor class | `src/v2/execution/speculative/SpeculativeAcceptor.h/.cpp` |
| Unit tests | `tests/v2/unit/Test__SpeculativeAcceptor.cpp` |

---

## Phase 15: End-to-End MTP Validation

**Goal**: Validate that MTP produces correct output and achieves the expected speedup.

### 15.1 Correctness Tests

| Test | Description | Criterion |
|------|-------------|-----------|
| MTP tokens match non-MTP | Run same prompt with and without MTP (greedy) | Identical output tokens |
| MTP acceptance rate | Measure acceptance rate over standard prompts | Should be > 80% for D=1 |
| MTP logit parity | Draft logits vs verify logits for same position | Verify logits are authoritative |
| MTP KV cache state | After rollback, KV cache matches non-speculative path | Bit-exact KV cache contents |

### 15.2 Performance Benchmarks

| Benchmark | Metric | Expected (D=1) |
|-----------|--------|-----------------|
| Tokens per second (greedy) | tok/s | 1.5-1.8× baseline |
| Tokens per second (sampling, t=0.7) | tok/s | 1.3-1.5× baseline |
| Acceptance rate (greedy) | % | 80-90% (see note below) |
| Acceptance rate (sampling) | % | 70-85% |
| Memory overhead for MTP heads | MB | < 5% model size |
| Latency per draft step | ms | < 1 main model decode step |

> **Acceptance rate note**: DeepSeek-V3 reports 85-90% acceptance. Real-world Qwen 3.5 data from llama.cpp PR #20700 shows 40-95% depending on quantization, prompt type, and implementation. The key factors: (1) MTP block using full attention (vs FFN-only) achieves 82% vs 60%, (2) F16 MTP weights vs Q4_K_M quantized MTP weights significantly affect accuracy, (3) code/math prompts have lower acceptance than natural language.

### 15.3 CLI Integration

```bash
# Enable MTP (auto-detects from model weights)
./llaminar2 -m qwen35.gguf -p "Hello" -n 50

# Explicitly disable MTP (useful for benchmarking)
./llaminar2 -m qwen35.gguf -p "Hello" -n 50 --no-mtp

# Benchmark with and without MTP
./llaminar2 --benchmark -m qwen35.gguf -n 128
./llaminar2 --benchmark -m qwen35.gguf -n 128 --no-mtp

# Show MTP statistics during generation
LLAMINAR_MTP_STATS=1 ./llaminar2 -m qwen35.gguf -p "Hello" -n 50
```

**MTP statistics output**:
```
[MTP] Decode complete: 50 tokens in 25 speculative steps
[MTP] Average tokens/step: 2.00
[MTP] Acceptance rate: 87.5%
[MTP] Draft overhead: 12.3% of main model forward time
[MTP] Net speedup: 1.76x
```

### Deliverables

| Item | File/Location |
|------|--------------|
| MTP parity test | `tests/v2/integration/Test__MTPParity.cpp` |
| MTP benchmark | `tests/v2/performance/Test__MTPBenchmark.cpp` |
| CLI flags | `src/v2/config/OrchestrationConfigParser.cpp` |
| Stats tracking | `src/v2/execution/runner/MTPStatistics.h` |

---

## Integration with GDN (Hybrid Model)

### MTP Block Type: Confirmed as Full Attention

The MTP transformer block is **always Full Attention** (confirmed from HuggingFace source code and llama.cpp PR #20700). It does NOT use GDN/linear attention, even though the main model is a GDN+FA hybrid. This means:

| Aspect | Details |
|--------|---------|
| **MTP block attention type** | **Full Attention** (gated Q, QK norm, partial RoPE) — same as main model's FA layers |
| **GDN state in MTP** | None needed — MTP block has no GDN state |
| **KV cache in MTP** | MTP block has its own small 1-layer KV cache |
| **Draft token embedding** | Same as main model — no GDN-specific change |

### The Critical Challenge: Main Model GDN State During Verification

While the MTP block itself is simple (just an FA layer), the **verification pass** runs through the entire main model (all GDN + FA layers). This creates the fundamental challenge:

**Problem**: During the verification pass, draft tokens `[token_1, token_2]` flow through ALL main model layers as a seq_len=2 forward pass. The GDN layers irreversibly update their recurrent state `S_t` for both tokens. If `token_2` is rejected, the GDN state has been corrupted — it now incorporates information from a token that shouldn't exist.

**This is the hardest unsolved problem for MTP on hybrid GDN+FA models.**

### GDN State Rollback Strategies

| Strategy | Complexity | Memory | Accuracy | Throughput Impact |
|----------|-----------|--------|----------|-------------------|
| **A: Checkpoint and restore** | Low | +1 state copy per GDN layer | Exact | Near-zero overhead |
| **B: Two-phase decode** | Medium | None extra | Exact | Halves MTP benefit (~1.3× vs 1.8×) |
| **C: Recompute from last accepted** | High | None extra | Exact | High recompute cost |
| **D: Accept all GDN state updates** | None | None | Approximate | No overhead but quality risk |

#### Strategy A: Checkpoint and Restore (Recommended)

Before the draft phase, snapshot each GDN layer's recurrent state. After rejection, restore and replay only accepted tokens:

```cpp
// Before draft phase — save GDN state
for (auto& gdn_layer : gdn_layers) {
    gdn_layer->checkpointState();  // Copy S → S_checkpoint (+ conv1d state)
}

// Run verification pass through main model (ALL layers process draft tokens)
runner_->setComputeAllPositionLogits(true);
runner_->forward(draft_tokens.data(), draft_tokens.size());

// After verification — determine accepted count
int accepted = verifyDraftTokens(verify_logits, draft_tokens, draft_logits_list);

// Rollback GDN state if any tokens rejected
if (accepted < draft_tokens.size()) {
    // Restore GDN state to pre-draft checkpoint
    for (auto& gdn_layer : gdn_layers) {
        gdn_layer->restoreState();  // Copy S_checkpoint → S
    }
    // Re-apply only accepted tokens through GDN layers
    // (FA layers are handled by KV cache truncation)
    for (int i = 0; i < accepted; ++i) {
        runner_->forwardGDNOnly(&accepted_tokens[i], 1);
    }
    // Truncate FA KV cache
    runner_->truncateKVCache(current_position_ + accepted);
}
```

**Memory cost**: For Qwen 3.5 0.8B with 18 GDN layers, d_k=64, d_v=128, n_kv_heads=4:
- State per layer: 4 × 64 × 128 × 4 bytes = 128 KB
- Total checkpoint: 18 × 128 KB = **2.25 MB** (negligible)
- Plus conv1d state: 18 × (QKV_dim × conv_kernel) ≈ 1 MB

For Qwen 3.5 4B: ~16 MB total (still negligible vs model size).

#### Strategy B: Two-Phase Decode (Simpler but Slower)

From llama.cpp PR #20700: Only run accepted tokens through the main model, never speculative tokens:

```
Phase 1 (Draft): Use MTP head only (no main model GDN involvement)
  - h_main is from the PREVIOUS verified decode step
  - MTP head processes draft token → draft logits
  
Phase 2 (Verify): Run ONLY the first draft token through main model
  - If accepted, run next draft token, etc.
  - Stop at first rejection
  - GDN state is never exposed to rejected tokens
```

**Downside**: This is essentially serial — you lose the parallel verification benefit. Net speedup drops to ~1.3× vs 1.8× theoretical. The llama.cpp PR reports ~15 tok/s with full MTP setup.

### Conv1d State Rollback

GDN layers also have a short conv1d (kernel size 4) that maintains state across tokens. This must be checkpointed alongside the recurrent state `S`:

```cpp
struct GDNLayerCheckpoint {
    std::vector<float> recurrence_state;  // S matrix: [n_heads, d_k, d_v]
    std::vector<float> conv1d_state;      // Conv buffer: [d_qkv, kernel_size-1]
};
```

### Recommendation

**Start with Strategy A (checkpoint/restore)**. The memory overhead is trivial (2-16 MB), the implementation is straightforward (memcpy before draft, memcpy back on rejection), and it preserves the full MTP speedup. Strategy B should be a fallback if Strategy A proves too complex to integrate with the current GDN pipeline.

### Dependency Between Phases

```
GDN Project Plan (Complete)         MTP Project Plan
───────────────────────────         ────────────────
GDN stages + kernels          ───→  Phase 11: MTP graph stages (reuses FA infra)
GGUF loading                  ───→  Phase 10: MTP weight loading (new mtp.* tensors)
GDN state management          ───→  Phase 9.5: GDN State Checkpointing (new)

                               Phase 9: KV cache rollback      (can start immediately)
                               Phase 9.5: GDN state checkpoint  (after GDN is stable)
                               Phase 12: Speculative decode     (needs Phase 9, 9.5, 11, 13)
                               Phase 13: LM head multi-pos      (can start immediately)
                               Phase 14: Verification           (can start immediately)
```

**Critical path**: Phase 9 (KV rollback) and Phase 13 (LM head multi-position) can start **immediately** as they don't depend on GDN. Phase 9.5 (GDN state checkpointing) is the new critical dependency. Phase 12 (the decode loop) integrates everything and is the final piece.

---

## FastMTP Optimization (Vocabulary Trimming)

### Overview

The LM head projection (d → vocab_size) is the most expensive part of the MTP draft step because Qwen 3.5's vocabulary is **151,936 tokens**. The LM head GEMM (`1 × d × vocab_size`) takes ~10ms on CPU, dominating the draft generation overhead that should ideally be < 1ms.

### The FastMTP Technique

From llama.cpp PR #20700: Instead of computing full logits over the entire vocabulary, restrict the LM head to only the **top-K most likely tokens** based on the main model's logits:

```
1. Main model forward → full logits (151,936 tokens)
2. Sort to find top 32,768 tokens (by logit value)
3. Create a trimmed LM head view: only compute logits for these 32K tokens
4. MTP head forward → trimmed logits (32K tokens, not 152K)
5. Sample draft token from trimmed distribution
```

### Expected Performance

| Configuration | LM Head Size | Draft Time | Notes |
|---------------|-------------|------------|-------|
| Full vocabulary | 151,936 | ~10ms | Baseline |
| Top 32K (FastMTP) | 32,768 | ~2.1ms | 4.8× faster draft |
| Top 16K | 16,384 | ~1.1ms | 9× faster, may lose accuracy |

**Accuracy impact**: Negligible. llama.cpp testing showed no measurable accuracy loss with top 32K trimming — the draft only needs to predict the *most likely* next token, and rare tokens are never in the draft anyway.

### Implementation

```cpp
// In MTP draft step:
// 1. Get main model's top-K token indices
auto top_k_indices = getTopKIndices(main_logits, vocab_size, 32768);

// 2. Create a view of the embedding/LM-head matrix for only these tokens
auto trimmed_lm_head = createTrimmedView(embedding_weight, top_k_indices);

// 3. MTP forward produces logits over trimmed vocabulary
auto draft_logits = mtpForward(h', trimmed_lm_head);  // (1, 32K) instead of (1, 152K)

// 4. Sample from trimmed distribution
int draft_token_idx = sample(draft_logits, 32768);
int draft_token = top_k_indices[draft_token_idx];
```

### Priority

FastMTP is an **optimization**, not a correctness requirement. Implement after the basic MTP pipeline is working (Phase 15+). With D=1, the draft only requires one LM head call, so the baseline without FastMTP is still viable.

---

## Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| ~~Qwen3-Next doesn't use MTP~~ | ~~Wasted work~~ | ~~N/A~~ | **Resolved**: Qwen 3.5 confirmed with `mtp_num_hidden_layers: 1` across all dense models |
| GGUF format for MTP weights not standardized | Blocks weight loading | Medium | HF names confirmed; llama.cpp PR #20700 converging on naming; implement flexible mapper |
| **GDN recurrent state cannot be partially rolled back** | **Fundamental challenge for MTP on hybrid models** | **Confirmed** | Checkpoint/restore (2-16 MB memory cost) or two-phase decode (1.3× vs 1.8× speedup) |
| Conv1d state rollback adds complexity | Implementation effort | Low | Checkpoint conv buffer alongside recurrence state |
| Stochastic speculative sampling is tricky | Incorrect token distribution | Medium | Start with greedy-only; add stochastic later |
| MTP head accuracy degrades with quantization | Low acceptance rate (40-60% vs 80-90%) | High | Keep MTP weights at higher precision (F16/Q8); FastMTP helps by reducing vocab |
| Multi-token LM head computation in verification | Reduces net MTP benefit | Low | Only D+1 tokens max; FastMTP (top-32K vocab) for draft step |
| MTP adds complexity to the decode loop | Maintenance burden | Medium | Clean abstraction with `SpeculativeAcceptor`; `--no-mtp` flag for fallback |
| D > 1 (multiple MTP depths) adds complexity | Harder to test/debug | Low | All Qwen 3.5 models use D=1; design for extensibility but don't over-engineer |
| Partial RoPE position tracking in MTP block | MTP attention sees different position offsets than main model | Medium | MTP position = main model position + 1 (offset for the draft token) |

---

## Implementation Priority

| Priority | Phase | Rationale | Dependencies |
|----------|-------|-----------|--------------|
| 1 | **Phase 9**: KV Cache Rollback | Prerequisite for FA layer rollback; no external dependencies | None |
| 2 | **Phase 13**: LM Head Multi-Position | Prerequisite for verification; simple change | None |
| 3 | **Phase 14**: Verification/Acceptance | Core algorithm; can be unit tested independently | None |
| 4 | **Phase 9.5**: GDN State Checkpointing | Prerequisite for GDN rollback on rejection | GDN support stable |
| 5 | **Phase 10**: MTP Weight Loading | Needed once GGUF models exist | GGUF naming standardized |
| 6 | **Phase 11**: MTP Graph Stages | MTP forward pass (reuses FA infra heavily) | Phase 10 |
| 7 | **Phase 12**: Speculative Decode Loop | Integrates all pieces | Phase 9, 9.5, 11, 13, 14 |
| 8 | **Phase 15**: E2E Validation | Confirms correctness + measures speedup | Phase 12 |
| 9 | **FastMTP**: Vocabulary Trimming | Optimization: 5× faster draft via top-32K vocab | Phase 12 working |

**Parallelism**: Phases 9, 13, and 14 can all be developed **in parallel** and **immediately**, before any MTP-specific work. Phase 9.5 (GDN state checkpoint) depends on GDN support being stable but is independent of Phases 10-14.

---

## External References

| Resource | URL | Notes |
|----------|-----|-------|
| **Qwen 3.5 Model Card** | https://huggingface.co/Qwen/Qwen3.5-0.8B | Confirmed MTP config, weight names |
| **Qwen 3.5 config.json** | https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json | `mtp_num_hidden_layers: 1`, `mtp_use_dedicated_embeddings: false` |
| **Qwen 3.5 safetensors index** | https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/model.safetensors.index.json | Confirmed all `mtp.*` weight names |
| **HF modular_qwen3_5.py** | https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modular_qwen3_5.py | Reference Python implementation (ignores MTP by default) |
| **HF modular_qwen3_next.py** | https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modular_qwen3_next.py | Base model with MTP class definitions |
| **llama.cpp MTP PR #20700** | https://github.com/ggml-org/llama.cpp/pull/20700 | Dense Qwen 3.5 MTP: full attention, FastMTP vocab trimming, DeltaNet challenges |
| **vLLM Qwen3NextMTP** | https://github.com/vllm-project/vllm/issues/35031 | vLLM MTP implementation, `eh_proj` weight mapping issues |
| Meta MTP Paper | https://arxiv.org/abs/2404.19737 | "Better & faster LLMs via multi-token prediction" |
| DeepSeek-V3 Technical Report | https://arxiv.org/abs/2412.19437 | Section 2.2: MTP architecture; Section 5.4.3: inference results |
| Speculative Decoding | Leviathan et al., ICML 2023 | Original speculative decoding algorithm |
| EAGLE Speculative Decoding | Li et al., ICML 2024 | Similar causal chain approach to DeepSeek-V3 MTP |
| Qwen 3.5 GDN Plan | `docs/v2/cleanup/QWEN3NEXT_GDN_PROJECT_PLAN.md` | Companion document for GDN attention support |
