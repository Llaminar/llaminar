# TQ4 KV Cache Quantization: Attention Score Error Analysis

## Executive Summary

TQ4 (TurboQuant 4-bit) KV cache quantization produces catastrophic attention weight distortion on Qwen3-0.6B (`head_dim=128`), despite achieving a healthy per-vector cosine similarity of ~0.995. The root cause is **not** a code bug — the pipeline, rotation, and codebook are all correct. The problem is a fundamental information-theoretic mismatch: 4-bit quantization noise in K, when projected through the QK^T dot product and amplified by softmax, produces attention score errors 10–20× larger than the score gaps that determine which tokens are attended to.

This document presents the empirical methods, mathematical analysis, and conclusions from a multi-session investigation.

---

## Table of Contents

- [Background](#background)
- [Methodology](#methodology)
- [Empirical Findings](#empirical-findings)
  - [Pipeline Verification](#pipeline-verification)
  - [Rotation Effectiveness](#rotation-effectiveness)
  - [Codebook Optimality](#codebook-optimality)
  - [Per-Head Attention Analysis](#per-head-attention-analysis)
  - [K Norm Distribution](#k-norm-distribution)
- [Theoretical Analysis](#theoretical-analysis)
  - [Score Error Formula](#score-error-formula)
  - [Softmax Amplification](#softmax-amplification)
  - [Bit-Width Requirements](#bit-width-requirements)
- [Why Qwen2 Passes but Qwen3 Fails](#why-qwen2-passes-but-qwen3-fails)
- [Comparison of Quantization Approaches](#comparison-of-quantization-approaches)
- [Conclusions](#conclusions)
- [Potential Mitigations](#potential-mitigations)
- [Appendix: Split TQ (TQ8 K + TQ4 V) — Implementation & Parity Results](#appendix-split-tq-tq8-k--tq4-v--implementation--parity-results)

---

## Background

TurboQuant (TQ4) implements rotation-based 4-bit KV cache quantization inspired by QuIP#. The algorithm:

1. **Normalize**: Compute `||K||`, form unit vector `K̂ = K / ||K||`
2. **Rotate**: Apply a Haar-random orthogonal matrix Π to decorrelate: `K̂_rot = Π · K̂`
3. **Scale**: Multiply by `√D` to map to approximately N(0,1): `K̂_scaled = K̂_rot · √D`
4. **Quantize**: Map each element to the nearest of 16 Lloyd-Max centroids for N(0,1)
5. **Store**: Pack 4-bit indices + float32 norm per `head_dim`-element block

Dequantization reverses steps 4→1: look up centroids, divide by `√D`, apply `Π^T`, multiply by stored norm.

The scheme achieves **cosine similarity ~0.995** between original and reconstructed K/V vectors. For weight quantization (where errors average over many output dimensions), this is excellent. For attention scores (where each score is a single dot product fed through softmax), it is insufficient.

---

## Methodology

### Data Collection

Analysis was performed on **Qwen3-0.6B** (28 layers, `d_model=1024`, `head_dim=128`, 16 Q-heads, 8 KV-heads, GQA ratio 2) with a 10-token prefill prompt at decode step 0, layer 0.

Effective K/V tensors were dumped at the attention stage boundary using the `LLAMINAR_DUMP_EFFECTIVE_KV` facility in `AttentionComputeStage.cpp`. Two pipeline runs were captured:

| Iteration | Pipeline | K Source | Dump Contents |
|-----------|----------|----------|---------------|
| `iter0` | FP32 (no KV quantization) | Direct FP32 from RoPE | Q, K_effective, V_effective (all FP32) |
| `iter1` | TQ4 (quantized KV cache) | TQ4 → dequant → FP32 buffer | Q, K_effective (dequanted), V_effective (dequanted) |

Both iterations share the same Q tensor (Q is never cached/quantized).

### Reproduction

All attention computations were reproduced in Python using the dumped binary tensors, implementing standard grouped-query attention:

```python
for each Q head h:
    kv_h = h // gqa_ratio
    scores = (Q[h] @ K[:, kv_h].T) * (1/√head_dim)
    weights = softmax(scores)
    context[h] = weights @ V[:, kv_h]
```

The Python simulation reproduces the pipeline's reported `ATTENTION_CONTEXT` cosine of **0.530073** exactly, confirming no pipeline wiring errors.

---

## Empirical Findings

### Pipeline Verification

The TQ4 quantization path is correctly triggered. Dump metadata from `iter1` confirms:

```
K_is_dequanted=1
V_is_dequanted=1
K_type=FP32  (FP32 dequant buffer, sourced from TQ4)
```

The `dynamic_cast<TQ4Tensor*>` succeeds in `AttentionComputeStage::execute()`, `turboquant_dequantize_kv_rows` runs, and the effective K/V point to the FP32 dequant buffers. **No code bug exists.**

### Rotation Effectiveness

The Haar-random orthogonal rotation matrix (generated via QR decomposition of a Gaussian random matrix with sign correction) was verified to **correctly decorrelate** post-RoPE K vectors:

| Metric | Before Rotation | After Rotation | Ideal |
|--------|----------------|---------------|-------|
| Kurtosis | 111.4 | 4.0 | 3.0 (Gaussian) |
| Pair energy ratio (max/min) | 16,994,807 : 1 | 308 : 1 | ~1 : 1 |
| Element range | [-0.97, 0.24] | [-0.29, 0.29] | symmetric |
| Element std | 0.088 | 0.088 | 1/√D = 0.088 |

Post-RoPE K vectors are **extremely non-uniform** — RoPE concentrates nearly all energy into a few dimension pairs (the low-frequency rotation components). The rotation successfully transforms this into a near-Gaussian distribution with uniform energy across all 128 dimensions.

After scaling by `√128`, the rotated elements fall in `[-3.3, 3.3]` with `std ≈ 0.99`, closely matching the N(0,1) distribution that the Lloyd-Max codebook is designed for.

**Conclusion**: The rotation is mathematically correct and achieves its design goal. The implementation matches the paper.

### Codebook Optimality

The 16 TQ4 centroids are **verified Lloyd-Max optimal** for N(0,1):

```
Centroids: [-2.733, -2.069, -1.618, -1.257, -0.943, -0.657, -0.388, -0.128,
             0.128,  0.388,  0.657,  0.943,  1.257,  1.618,  2.069,  2.733]
```

The centroids are perfectly symmetric about zero. The theoretical MSE for 16-level Lloyd-Max quantization of N(0,1) is:

$$\text{MSE}_{\text{LM}} = \int_{-\infty}^{\infty} (x - Q(x))^2 \, \phi(x) \, dx = 0.009501$$

This gives a signal-to-quantization-noise ratio (SQNR) of **20.22 dB**, which is the theoretical optimum for 4-bit scalar quantization of Gaussian data.

**Conclusion**: The codebook achieves the information-theoretic limit. No better 4-bit scalar codebook exists for this data distribution.

### Per-Head Attention Analysis

The overall attention context cosine is 0.530, but individual heads show a **bimodal distribution**:

| Head | KV Head | FP32 Entropy | Score Error Max | Weight Cosine | Argmax Match |
|------|---------|-------------|-----------------|---------------|--------------|
| 3 | 1 | 0.020 (peaked) | 1.59 | 0.999998 | Yes (9→9) |
| 7 | 3 | 0.157 (peaked) | 3.69 | 0.999910 | Yes (9→9) |
| 6 | 3 | 1.982 (flat) | 2.15 | 0.384549 | No (0→7) |
| 4 | 2 | 0.915 (flat) | 19.42 | 0.035454 | No (9→7) |
| 5 | 2 | 1.100 (flat) | 21.84 | 0.327792 | No (9→7) |
| 13 | 6 | 1.209 (flat) | 13.26 | 0.006625 | No (0→4) |
| 12 | 6 | 0.966 (flat) | 14.33 | 0.256117 | No (6→4) |

**Pattern**: Heads with peaked FP32 attention (low entropy, dominant token) survive TQ4 noise. Heads with flat/spread attention (high entropy) are catastrophically destroyed.

**Example — Head 3 (SURVIVES)**:
```
FP32 scores: [29.7, 29.7, 27.6, 27.6, 26.1, 26.7, 29.4, 35.7, 41.4, 47.2]
TQ4  scores: [29.7, 29.2, 27.6, 28.2, 27.3, 27.3, 28.9, 35.7, 43.0, 48.3]
FP32 weights: [0, 0, 0, 0, 0, 0, 0, 0, 0.003, 0.997]  ← peaked on position 9
TQ4  weights: [0, 0, 0, 0, 0, 0, 0, 0, 0.005, 0.995]  ← still peaked on position 9
```
The dominant token (position 9, score 47.2) has a **6-nat margin** over the runner-up. A 1.6-nat TQ4 error cannot flip the winner.

**Example — Head 4 (DESTROYED)**:
```
FP32 scores: [ 4.3, -5.2, -5.9, -5.5, -4.3, -7.2,  3.5, -3.7, -1.1,  5.1]
TQ4  scores: [ 8.6,-20.8, 13.6,-16.2, -7.0,-12.6, 11.8, 14.7,-13.3, 11.1]
FP32 weights: [0.262, 0, 0, 0, 0, 0, 0.121, 0, 0.001, 0.616]  ← spread
TQ4  weights: [0.002, 0, 0.231, 0, 0, 0, 0.042, 0.706, 0, 0.020]  ← completely different
```
The score gap between positions 9 and 0 is only **0.86 nats**. The TQ4 score error of **19.4 nats** completely reorders all positions. Attention is now focused on position 7 (was -3.7, became 14.7).

### K Norm Distribution

The score error magnitude is **directly proportional** to `||K||`. K norms vary 7× across KV heads:

| KV Head | Mean ||K|| | Max ||K|| | Q Heads | Attention Status |
|---------|-----------|----------|---------|-----------------|
| 1 | 49.4 | 60.0 | 2, 3 ✓ | OK (low norm) |
| 2 | 341.7 | 438.5 | 4 ✗, 5 ✗ | Destroyed (high norm) |
| 3 | 221.3 | 305.1 | 6, 7 ✓ | Mixed |
| 6 | 325.9 | 402.8 | 12 ✗, 13 ✗ | Destroyed (high norm) |

RoPE preserves norms (it is a rotation), so these differences originate from the **K-projection weight matrix** (`W_K`). Different KV heads project different amounts of energy from the residual stream.

The catastrophic heads (4, 5, 12, 13) all share the two KV heads with the **largest norms** (KV heads 2 and 6, both with `||K|| > 325`).

---

## Theoretical Analysis

### Score Error Formula

For a single attention score `s = q^T K / √D`, the TQ4 error is:

$$\Delta s = \frac{q^T \cdot (K_{\text{TQ4}} - K_{\text{FP32}})}{\sqrt{D}}$$

Since the rotation decorrelates K's quantization error into approximately independent components, the score error follows:

$$\sigma(\Delta s) = \frac{\|q\| \cdot \|K\| \cdot \sqrt{\text{MSE}}}{D}$$

where `MSE = 0.009501` is the Lloyd-Max quantization MSE for N(0,1).

For the worst-case head (Q head 4, KV head 2):

$$\sigma = \frac{31.3 \times 341.7 \times \sqrt{0.009501}}{128} = 8.14 \text{ nats}$$

The actual maximum observed score error was **19.4 nats** (2.4σ, consistent with the tail of a Gaussian over 10 positions).

### Softmax Amplification

The softmax function $w_i = e^{s_i} / \sum_j e^{s_j}$ exponentially amplifies score perturbations. For a small perturbation $\Delta s$ to one score:

$$\frac{\partial w_i}{\partial s_j} = w_i(\delta_{ij} - w_j)$$

When attention is flat (all $w_i \approx 1/N$), the Jacobian has eigenvalues of order $1/N$, meaning small score changes produce proportional weight changes. When the score difference between two tokens is $\Delta$, the weight ratio changes by $e^{\Delta s_{\text{error}}}$, where $\Delta s_{\text{error}}$ is the differential score error.

For Head 4 with $\Delta s_{\text{error}} = 19.4$:

$$e^{19.4} \approx 2.65 \times 10^8$$

A weight ratio change of $10^8$ means the entire attention distribution is randomized — we might as well be looking at a different sequence.

### Bit-Width Requirements

Using the score error formula with worst-case parameters (`||Q|| = 36`, `||K|| = 342`, `D = 128`) and the score gap of the most vulnerable head (`gap = 0.86 nats`):

| Bits | Lloyd-Max MSE | σ(score error) | σ / gap | Status |
|------|--------------|----------------|---------|--------|
| 2 | 0.1175 | 32.97 | 38.3× | Destroyed |
| 3 | 0.03454 | 17.88 | 20.8× | Destroyed |
| **4 (TQ4)** | **0.009501** | **9.38** | **10.9×** | **Destroyed** |
| 5 | 0.002499 | 4.81 | 5.6× | Destroyed |
| 6 | 0.000635 | 2.42 | 2.8× | Destroyed |
| 7 | 0.000159 | 1.21 | 1.4× | Marginal |
| 8 | 0.0000397 | 0.61 | 0.7× | Marginal |

Even **7-bit quantization** would be marginal for the worst-case head. The required MSE for `σ < 0.1 × gap` is `1.08 × 10⁻⁶`, which corresponds to approximately **10 bits**.

This analysis is conservative (worst-case head on layer 0, step 0). As sequence length grows and more positions compete for attention, score gaps may narrow further, making the problem worse.

### Why the Paper's Guarantees Don't Apply

QuIP#-style rotation quantization guarantees near-optimal **reconstruction MSE**. The expected squared error is bounded by:

$$\mathbb{E}\|K - \hat{K}\|^2 = \|K\|^2 \cdot \text{MSE}$$

This is indeed achieved (cosine ≈ 0.995). However, the paper targets **weight quantization**, where:

1. Errors average over the output dimension of a GEMM (`y = Wx`)
2. Each output element sums `D` error terms → law of large numbers applies
3. The relative output error scales as `√MSE`, independent of weight magnitude

For **attention scores**, errors do NOT average:

1. Each score is a **single** dot product: `s = q^T k / √D`
2. The score error is a single draw from a distribution with `σ ∝ ||q|| · ||K|| · √MSE / D`
3. Softmax exponentiates the score, amplifying errors by `e^{σ}`
4. KV heads with large `||K||` (300+) produce σ > 9 nats, which is catastrophic

The rotation successfully makes the quantization error **independent** across dimensions. But independence doesn't help when the downstream operation (softmax) is exponentially sensitive to the error magnitude.

---

## Why Qwen2 Passes but Qwen3 Fails

Qwen2.5-0.5B (`head_dim=64`) passes TQ4 parity while Qwen3-0.6B (`head_dim=128`) fails. The key differences:

| Property | Qwen2.5-0.5B | Qwen3-0.6B |
|----------|-------------|------------|
| `head_dim` | 64 | 128 |
| `n_heads` | 14 | 16 |
| `n_kv_heads` | 2 | 8 |
| TQ4 block size | 64 elements | 128 elements |
| K range after RoPE | Smaller | Larger |

The primary driver is that Qwen3's K-projection weights produce **K vectors with much larger norms** (up to 438 for KV head 2). Combined with `head_dim=128`, this yields score errors of 9–20 nats.

Qwen2's smaller `head_dim=64` naturally produces K vectors with lower L2 norms (fewer dimensions contribute to the norm), and its K-projection weights are more uniform across heads. The score error formula shows that `σ ∝ ||K||`, so halving the K norm halves the score error, potentially bringing it below the score gap threshold.

Additionally, Qwen3 architecture uses more KV heads (8 vs 2), with a lower GQA ratio (2 vs 7). This means each KV head serves fewer Q heads, giving the model less redundancy to absorb per-head failures.

---

## Comparison of Quantization Approaches

For the worst-case KV head 2 (position 0, `||K|| = 210.8`):

| Method | Score Error (Head 4) | Score Error (Head 5) | Mechanism |
|--------|---------------------|---------------------|-----------|
| **TQ4** (4-bit + rotation) | 4.34 | 0.40 | Rotation + Lloyd-Max centroids |
| **Q8_1** (8-bit scalar, per-block) | 2.84 | 1.21 | Per-128-block scale + zero |

Surprisingly, standard Q8_1 with per-block scaling **also fails** for the worst-case head (`score_err = 2.84` vs `gap = 0.86`). This is because Q8_1 quantizes absolute values — with K range [-417, 143], the 256 quantization levels have a step size of 1.61, which is too coarse for attention score fidelity.

TQ4's rotation-based norm-preserving approach actually achieves **better per-element MSE** than Q8_1 for high-dynamic-range data. The issue isn't the quantization scheme — it's the fundamental incompatibility of low-bit-width and attention score sensitivity.

---

## Conclusions

1. **The TQ4 implementation is correct.** The rotation properly decorrelates post-RoPE K (kurtosis 111→4). The codebook achieves the Lloyd-Max optimum. The pipeline correctly quantizes, caches, dequantizes, and uses TQ4 K/V.

2. **The problem is fundamental.** With `||K|| ≈ 342` and `||Q|| ≈ 36` on the worst KV head, the attention score error standard deviation is ~9 nats for 4-bit quantization. Since score gaps between competing tokens are often < 1 nat, no 4-bit scheme can preserve attention accuracy for these heads.

3. **The failure is head-selective.** Heads with peaked attention distributions (low entropy, dominant token) tolerate TQ4 noise. Heads with flat/spread distributions (high entropy, close scores) are destroyed. This creates a bimodal quality distribution.

4. **Qwen3's K-projection weights amplify the problem.** K norms vary 7× across KV heads (49 to 438), and the highest-norm heads are precisely the ones that fail. This is a model-specific property, not a universal limitation.

5. **Even 8-bit quantization is not universally safe.** Q8_1 with per-block scaling also fails for the worst-case head (`σ/gap = 3.3×`). Attention-safe K quantization requires either very high precision (≥10 bits) or K-specific strategies that respect the dot-product-then-softmax downstream use.

---

## Potential Mitigations

Listed in approximate order of complexity and expected effectiveness:

| Approach | Mechanism | Expected σ/gap | Complexity |
|----------|-----------|----------------|------------|
| **TQ4 for V, FP16 for K** | Unquantized K eliminates score error entirely | 0× | Low (config change) |
| **Asymmetric precision** (TQ4 V, TQ8 K) | 8-bit rotation quantization for K | ~0.7× | Medium (new TQ8 codebook) |
| **Per-head K norm clipping** | Clip K norms to a max value before quantization | Variable | Low |
| **Score-aware quantization** | Use Q to identify which K dimensions matter most | ~0× for those dims | High |
| **2-layer TQ4 residual** for K | Quantize residual after first TQ4 pass (~8 effective bits) | ~1.1× (marginal) | Medium |
| **Pre-RoPE quantization** | Quantize K before RoPE, apply RoPE after dequant | Eliminates RoPE-induced range | High (needs RoPE at attention time) |

The most practical near-term fix is **asymmetric K/V precision**: keep TQ4 for V (where errors are linear through the weighted sum, not exponential through softmax) and use higher precision for K.

---

## Appendix: Split TQ (TQ8 K + TQ4 V) — Implementation & Parity Results

### Overview

The "asymmetric precision" mitigation recommended above was implemented as **Split TQ** (`KVCachePrecision::TQ`): TQ8 for K projections, TQ4 for V projections. This section documents the implementation, the pipeline bugs that were found and fixed, and the resulting parity numbers compared to FP16, Q8_1, and pure TQ4.

### TQ8 Design

TQ8 uses the same rotation-based framework as TQ4 but with a 256-level Lloyd-Max codebook for N(0,1):

| Property | TQ4 | TQ8 |
|----------|-----|-----|
| Centroids | 16 (4-bit) | 256 (8-bit) |
| Index storage | 4 bits packed | 8 bits (uint8_t) |
| Per-block overhead | float32 norm + float32 residual_norm | float32 norm + float32 residual_norm |
| Block size (head_dim=64) | 40 bytes | 72 bytes |
| Block size (head_dim=128) | 72 bytes | 136 bytes |
| Bytes/element (head_dim=64) | 0.625 | 1.125 |
| Bytes/element (head_dim=128) | 0.5625 | 1.0625 |

### Memory Savings vs Full Q8_1

| Mode | K bytes/elem | V bytes/elem | **Total** | **Savings vs Q8_1** |
|------|-------------|-------------|-----------|---------------------|
| Q8_1 (both K+V) | 1.125 | 1.125 | 2.250 | — |
| Split TQ, head_dim=64 | 1.125 (TQ8) | 0.625 (TQ4) | 1.750 | **22.2%** |
| Split TQ, head_dim=128 | 1.0625 (TQ8) | 0.5625 (TQ4) | 1.625 | **27.8%** |

All savings come from V (half the cache). K stays at roughly the same cost as Q8_1 (identical at head_dim=64, slightly cheaper at 128).

### Pipeline Bugs Found and Fixed

Three bugs prevented Split TQ from working in the inference pipeline:

1. **InferenceRunnerFactory** (`InferenceRunnerFactory.cpp`): `TurboQuantContext` (which holds the per-layer rotation matrices) was only created when `KVCachePrecision::TQ4` was selected. Added `KVCachePrecision::TQ` to the condition at all 3 factory code paths (single-device, TP-local, TP-global).

2. **KVCacheAppendStage** (`KVCacheAppendStage.cpp`): The append stage had explicit precision-dispatch paths for FP16, FP32, Q8_1, Q16_1, and TQ4, but nothing for split TQ. When the cache's `k_precision()` returned `ActivationPrecision::TQ8`, execution fell through to the "direct append" path, which tried to cast FP32 projection tensors to TQ8 and failed. Added a new path that quantizes K→TQ8 and V→TQ4 before appending, in both single-sequence and batched code paths.

3. **AttentionComputeStage** (`AttentionComputeStage.cpp`): The existing TQ4 dequant block in attention matched only when *both* K and V were `TQ4Tensor*`. For split TQ, K is `TQ8Tensor*` and V is `TQ4Tensor*` — both casts failed, so the cached tensors' `data()` method was called without a TurboQuantContext, producing null pointers. Added a new dequant block with `TurboQuantDequantizeSplitTQ.h` that handles K=TQ8 + V=TQ4 dequantization, including the fused RoPE-on-read path.

### Decode Parity: Qwen2 (head_dim=64, 24 layers)

Model: `qwen2.5-0.5b-instruct-q4_0.gguf` (Q4_0 weights, 2 KV heads, kv_dim=128)

**Summary (5 decode steps):**

| KV Precision | Avg Cosine | Top-1 | Top-5 | Ref-in-Top3 |
|-------------|-----------|-------|-------|-------------|
| FP16 | 0.9995 | 100% | 100% | 5/5 |
| Q8_1 | 0.9967 | 80% | 100% | 5/5 |
| **Split TQ** | **0.9943** | **80%** | **100%** | **5/5** |
| TQ4 | 0.9430 | 60% | 100% | 5/5 |

**Per-step cosine similarity:**

| Step | FP16 | Q8_1 | **Split TQ** | TQ4 |
|------|------|------|-------------|-----|
| 0 | 0.9993 | 0.9979 | **0.9958** | 0.9770 |
| 1 | 0.9995 | 0.9948 | **0.9960** | 0.9623 |
| 2 | 0.9995 | 0.9985 | **0.9973** | 0.9666 |
| 3 | 0.9995 | 0.9960 | **0.9955** | 0.9651 |
| 4 | 0.9997 | 0.9961 | **0.9867** | 0.8439 |

### Decode Parity: Qwen3 (head_dim=128, 28 layers)

Model: `Qwen3-0.6B-Q8_0.gguf` (Q8_0 weights, 8 KV heads, kv_dim=1024)

**Summary (5 decode steps):**

| KV Precision | Avg Cosine | Top-1 | Top-5 | Ref-in-Top3 |
|-------------|-----------|-------|-------|-------------|
| FP16 | 0.9994 | 100% | 100% | 5/5 |
| Q8_1 | 0.9993 | 100% | 100% | 5/5 |
| **Split TQ** | **0.9961** | **100%** | **100%** | **5/5** |
| TQ4 | 0.9262 | 80% | 100% | 5/5 |

**Per-step cosine similarity:**

| Step | FP16 | Q8_1 | **Split TQ** | TQ4 |
|------|------|------|-------------|-----|
| 0 | 0.9989 | 0.9992 | **0.9979** | 0.9796 |
| 1 | 0.9995 | 0.9993 | **0.9980** | 0.9656 |
| 2 | 0.9997 | 0.9994 | **0.9944** | 0.9110 |
| 3 | 0.9996 | 0.9994 | **0.9939** | 0.9502 |
| 4 | 0.9994 | 0.9993 | **0.9964** | 0.8245 |

### Analysis

1. **Split TQ achieves near-Q8_1 quality.** The average decode cosine is within 0.003 of Q8_1 on both models. The prediction agreement (Top-1, Top-5, Ref-in-Top3) is identical to Q8_1 on Qwen3 and within one step on Qwen2.

2. **The improvement over pure TQ4 is dramatic.** On Qwen3, TQ4's worst step (0.8245) is catastrophically bad — approaching random attention. Split TQ's worst step (0.9939) is well within the quality band. This validates the theoretical analysis: K quantization noise is amplified exponentially through softmax, while V quantization noise is only linear.

3. **The quality gap is larger on Qwen3.** Split TQ vs TQ4 difference is 0.05 on Qwen2 but 0.07 on Qwen3. This matches the prediction from the error analysis — Qwen3's higher K norms (up to 438 for the worst head) amplify the TQ4 score error more.

4. **Memory efficiency is meaningful.** The 22–28% savings vs full Q8_1 is entirely from V, which tolerates 4-bit quantization well. This is the optimal point on the quality–memory Pareto frontier for TurboQuant.
