# MPI Transformer Pipeline vs llama.cpp (Qwen2.5 0.5B) Deep Dive

> Historical Analysis Notice: This comparison references the former `MPITransformerPipeline` (now `DistributedTransformerPipeline`). Execution ordering and tensor shape discussions still apply; adapter indirection has since been removed.

> Purpose: Document precise dataflow, tensor shapes, and operation ordering to localize causes of golden test divergence between our `MPITransformerPipeline` and the llama.cpp reference implementation.

---
## Table of Contents
1. Scope & Model Assumptions
2. Global Hyperparameters & Notation
3. End-to-End Execution Graph (Overview)
4. Inputs (External + Derived)
5. Layer Block Structure (Pre-Norm Architecture)
6. Detailed Stage Comparisons
   - 6.1 Embedding
   - 6.2 Attention RMSNorm
   - 6.3 QKV Projections
   - 6.4 RoPE Application
   - 6.5 Attention Scores + Softmax
   - 6.6 Context Build + Output Projection
   - 6.7 FFN / SwiGLU Block
   - 6.8 Final RMSNorm + LM Head
7. Output Assembly & Distribution
8. Catalog of Potential Divergence Sources (Ranked)
9. Wiring Diagram (Reference Flow)
10. Diagnostic Strategy (Action Plan)
11. High-Value Immediate Checks
12. Hypotheses & Isolation Path
13. Next Engineering Steps

---
## 1. Scope & Model Assumptions
- Target model: **Qwen2.5 0.5B (quantized q4_0)**
- Architecture: Pre-norm transformer with grouped (GQA) attention (head_kv = 2)
- Environment: Golden parity test (prefill path, capped layers via `LLAMINAR_LAYER_CAP` during debugging)
- Objective: Explain logits divergence by identifying semantic or numerical deviations in our MPI pipeline vs llama.cpp.

---
## 2. Global Hyperparameters & Notation
| Symbol | Meaning | Value (Qwen2.5 0.5B) |
|--------|---------|-----------------------|
| S | Sequence length (prefill) | Test scenario (e.g., 32) |
| D | Model hidden size | 896 |
| L | Total layers (capped for test) | 2 (cap) / 24 (full) |
| H | Attention heads | 14 |
| H_kv | Key/Value heads | 2 |
| Groups | H / H_kv | 7 |
| Dh | Head dim | 64 (= D / H) |
| FF | Feed-forward dim | 4864 |
| ε | RMSNorm epsilon | 1e-6 |

---
## 3. End-to-End Execution Graph (Overview)
```
Tokens → Embedding → (Repeat L layers: Attn Norm → Multi-Head Attn → Residual → FFN Norm → SwiGLU FFN → Residual) → Final RMSNorm → LM Head → Logits
```
MPI adds: sharding (vocab or sequence), collective ops (Allgather/Allreduce), per-rank KV cache slices, optional backend selection (OpenBLAS vs future COSMA).

---
## 4. Inputs (External + Derived)
| Category | MPI Pipeline | llama.cpp | Notes |
|----------|--------------|-----------|-------|
| Token IDs | Provided vector<int32_t> | Same | Source identical |
| Model Weights | Eagerly dequantized (float) | Lazy / quant block handled by ggml | Precision path differs |
| Hyperparameters | Parsed then layer-cap patch | Parsed canonical | Layer cap only in test mode |
| Environment Overrides | Norm/LM head orientation, force unit gammas, etc. | None | Must be disabled for parity |
| KV Cache Strategy | Possibly distributed | Single context | Index alignment critical |

---
## 5. Layer Block Structure (Pre-Norm)
For layer i:
```
Input h_i
 a = RMSNorm_attn(h_i)
 (Q,K,V) = Linear(a)
 (Q',K') = RoPE(Q,K)
 scores = (Q' K'^T) * scale  (scale = 1/sqrt(Dh))
 scores = causal_mask(scores)
 P = softmax(scores)
 ctx = P V
 attn_out = ctx W_o
 h_attn = h_i + attn_out         (residual 1)
 b = RMSNorm_ffn(h_attn)
 u = b W_gate
 v = b W_up
 act = SiLU(u) * v               (SwiGLU)
 ffn_out = act W_down
 h_{i+1} = h_attn + ffn_out      (residual 2)
```
Post-stack:
```
final_norm = RMSNorm_out(h_L)
logits = final_norm W_out  (or tied embedding)
```

---
## 6. Detailed Stage Comparisons
### 6.1 Embedding
| Aspect | MPI | llama.cpp | Risk |
|--------|-----|-----------|------|
| Layout | Enforced (D × V) via helper | Raw tensor used with correct stride | Potential transpose mismatch |
| Sharding | Vocab or sequence partition; gather | None | Collective ordering |
| Type | Dequant to float | Quant blocks (dequant per node) | Minor fp drift |

### 6.2 Attention RMSNorm
Identical formula expected: `y = x * γ / sqrt(mean(x^2)+ε)`. Divergence if gamma override env active or dtype reductions differ.

### 6.3 QKV Projections
| Item | MPI | llama.cpp | Risk |
|------|-----|-----------|------|
| Weight Orientation | Possibly reinterpreted & enforced | Implicit correct orientation in graph | Accidental W^T applied |
| Accum Precision | float | float (post dequant) | Low |
| Distribution | Local matmul then (if needed) gather | Local | Ordering, stride views |

### 6.4 RoPE Application
| Aspect | MPI | llama.cpp | Risk |
|--------|-----|-----------|------|
| Freq dims | Precomputed 64 | Derived in graph (64) | If mismatch → phase error |
| Scaling variant | Linear (rope type 2) | Same | Low |

### 6.5 Attention Scores + Softmax
| Item | MPI | llama.cpp | Risk |
|------|-----|-----------|------|
| Scaling factor | Must explicitly apply 1/√Dh | Built into attention node / flash | Missing scale inflates variance |
| Mask | Manual causal mask | Graph mask | Off-by-one in causal index |
| Softmax | Custom kernel | Flash attention (enabled) | Numeric ordering differences |

### 6.6 Context + Output Projection
| Aspect | MPI | llama.cpp | Risk |
|--------|-----|-----------|------|
| V Combine | MatMul (maybe distributed) | Local fused path | Collective sum precision |
| Wo Orientation | Enforced | Graph inherent | Same transpose concern |
| Residual | h_i + attn_out | Same | Low |

### 6.7 FFN / SwiGLU Block
| Component | MPI | llama.cpp | Risk |
|-----------|-----|-----------|------|
| Activations | Should be SiLU(gate) * up | SiLU(gate) * up | If reversed / plain GLU -> big drift |
| Weight Shapes | Enforced matrices | Raw graph nodes | Orientation again |

### 6.8 Final RMSNorm + LM Head
| Aspect | MPI | llama.cpp | Risk |
|--------|-----|-----------|------|
| RMSNorm | Same formula | Same | Overrides? |
| LM Head | Use `output.weight` (enforced) or tie fallback | Uses `output.weight` or tie if absent | Double transpose, tie mismatch |
| Vocab Assembly | MPI allgather | Not needed | Ordering of slices |

---
## 7. Output Assembly & Distribution
Final logits shape: `[S, vocab]`.
- MPI path may compute `[S, vocab_local]` per rank → gather/consolidate.
- Must preserve token-major ordering (row continuity) and consistent column ordering (vocab id mapping).
- Any off-by-offset in partition boundaries yields structurally correct but semantically scrambled logits.

---
## 8. Catalog of Potential Divergence Sources (Ranked)
| Rank | Cause | Description | Impact Pattern |
|------|-------|-------------|----------------|
| 1 | Weight orientation enforcement | Misinterpreting stored tensor layout leads to using W^T | Systematic layer-wide drift from layer 0 |
| 2 | LM head orientation / assembly | Final projection transposed or mis-gathered | Pre-LM hidden matches, logits diverge only at end |
| 3 | Missing attention scaling | Larger score magnitude → sharper distributions → compounding mismatch | Divergence escalates each layer |
| 4 | SwiGLU activation mis-impl | Wrong nonlinearity or operand order | Rapid nonlinear divergence |
| 5 | RMSNorm gamma overrides | Forced unity or clamped gamma | Distribution flattening; consistent bias |
| 6 | Distributed gather ordering | Token or vocab misalignment | Chaotic logits, hidden might still match pre-head |
| 7 | Precision pathway | Early full-float vs blockwise dequant | Usually subtle (<1e-3) |
| 8 | RoPE freq dims mismatch | Using fewer dims | Positional drift accumulating |
| 9 | KV cache indexing skew | Head/time index mismatch across ranks | Attention noise pattern |
| 10 | Residual ordering deviation | Post-norm vs pre-norm mismatch | Large structural variance |

---
## 9. Wiring Diagram (Canonical Flow)
```
tokens
  → Embedding (gather if sharded)
  → hidden_0 [S,D]
FOR each layer i:
  a = RMSNorm_attn(hidden_i)
  Q,K,V = Linear(a)
  Q',K' = RoPE(Q,K)
  scores = (Q'K'^T) * (1/√Dh)
  scores = causal_mask(scores)
  P = softmax(scores)
  ctx = P V
  attn_out = ctx W_o
  h_attn = hidden_i + attn_out
  b = RMSNorm_ffn(h_attn)
  u = b W_gate
  v = b W_up
  act = SiLU(u) * v
  ffn_out = act W_down
  hidden_{i+1} = h_attn + ffn_out
final_norm = RMSNorm_out(hidden_L)
logits = final_norm W_out   (distributed assembly if needed)
```

---
## 10. Diagnostic Strategy (Action Plan)
| Step | Purpose | Expected Insight |
|------|---------|------------------|
| Capture pre-LM hidden parity | Separate LM head issues from upstream | If matches → focus LM head only |
| Layerwise post-residual checksum | Binary search earliest divergence | Narrow to specific sublayer |
| Weight slice dump (raw vs enforced) | Confirm orientation correctness | Show if transpose error systematic |
| Log attention scale & score stats | Ensure scaling present | Detect missing 1/√Dh |
| SwiGLU intermediate tap | Validate activation | Catch misuse of gate/up ordering |
| Vocab assembly checksum | Validate gather ordering | Detect misaligned concatenation |

---
## 11. High-Value Immediate Checks
1. Print for each linear weight: `(original_shape, enforced_shape, first 4×4 block before/after)`.
2. Add assertion that attention scaling factor equals `1.0f / sqrt(Dh)` numerically (log both).
3. Dump small sample of (gate_pre, gate_post, up, product) for token 0 first layer.
4. Pre-LM hidden L2 diff vs baseline (once test runtime unblocked) to classify LM head vs earlier.
5. Hash / checksum of assembled logits per token to detect rank-order shuffle vs baseline mapping.

---
## 12. Hypotheses & Isolation Path
| Hypothesis | Test | Interpretation |
|-----------|------|----------------|
| Weight layout wrong | Compare raw vs enforced slice; if identical but shapes flipped | Fix enforcement logic or add explicit transpose copy |
| LM head alone wrong | Pre-LM hidden diff ~0, logits diff large | Reorient / retie LM head path only |
| Missing attn scale | Score variance higher than baseline; softmax very peaky | Apply scale before softmax |
| SwiGLU mis-impl | Per-layer divergence spikes at FFN, not after attention | Correct activation expression |
| Vocab assembly bug | Hidden parity but token->logit mapping scrambled | Fix gather ordering & mapping table |

---
## 13. Next Engineering Steps
| Priority | Action | Rationale |
|----------|--------|-----------|
| High | Implement weight orientation audit utility | Fast validation of #1 risk |
| High | Add per-layer post-residual checksum capture + baseline compare | Localize earliest divergence |
| High | Instrument attention scaling & mask logs | Eliminate scaling/mask suspicion early |
| Medium | SwiGLU kernel debug tap | Validate nonlinearity semantics |
| Medium | LM head orientation self-test (multiply a one-hot) | Confirms projection axis alignment |
| Low | Precision drift quantification (one-layer CPU both paths) | Establish upper bound on quant noise |

---
## Appendix: Quick Environment Checklist for Golden Runs
Ensure these are UNSET or set to 0:
```
LLAMINAR_ALL_RMSNORM_FORCE_UNIT
LLAMINAR_OUTPUT_NORM_FORCE_UNIT
LLAMINAR_OUTPUT_NORM_CLAMP
LLAMINAR_LM_HEAD_RAW_ORIENTATION
```
Confirm correct set:
```
LLAMINAR_LAYER_CAP=2   (temporary for speed)
```

---
## Change Log
- Initial draft derived from interactive diagnostic session (Sep 29 2025).
- Focused on Qwen2.5; extend later for other architectures (MoE, etc.).

---
## Maintenance Notes
Update after: (a) orientation audit, (b) first successful pre-LM diff capture, (c) localization of first divergent layer.

---
*End of Document*
