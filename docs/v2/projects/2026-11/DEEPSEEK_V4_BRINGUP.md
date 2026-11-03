# DeepSeek-V4-Flash(-Vision) bring-up

Dated project plan for adding the `deepseek4` GGUF architecture
(DeepSeek-V4-Flash-Vision-Exp, HF revision
`6821d6ad3681a4b137b066b76094fa82ebd0a380`) on top of the QFN stack
(`feat/qfn-ngram-continuation`). Stable architecture belongs in the
architecture instruction once installed; this file records scope, decisions
and progress.

## Model summary

43 main layers, d=4096, 64 query heads of 512 (MQA: one 512-wide latent with
K == V), RoPE on the last 64 dims, q LoRA 1024, grouped low-rank output
(8 groups x 1024), sliding window 128, per-layer compression ratios
`[0, 0, (4, 128) x 20, 4]`, lightning indexer (64 heads x 128, top-512) on
ratio-4 layers, attention sinks, 4-way hyper-connections with 20 Sinkhorn
iterations, 256 routed experts top-6 with sqrt(softplus) scores, selection
bias, route scale 1.5 and token-id hash routing on the first 3 layers, one
shared expert, SwiGLU clamped at 10, YaRN (factor 16, original 64k) on
compressed layers only. Vision: 32-layer ViT (1024 wide, 2D RoPE) + 3x3 unfold
aligner; image spans attend bidirectionally and use `exp_probs_b_vl`.

Mathematical oracle: `python/reference/deepseek4.py` (full-recompute FP32),
validated against llama.cpp b12079 with `scripts/deepseek4/`. Prefill stage
cosine >= 0.9999 on non-indexer stages; indexer top-k differs only on
near-ties caused by llama.cpp's Q8 activation quantization.

## Reuse map (QFN primitives)

| DSV4 concept | QFN component | Relation |
|---|---|---|
| hc x d residual stream | QFN trunk (`residualWidth()`, embedding trunk expansion, branch read/write/head hooks in `QwenGraphBase`) | Reused plumbing. DSV4 adds a Sinkhorn read/write/head mixer selected by a typed residual-mixer policy next to Qwen4Exp's low-rank GR mixer. |
| Compressed KV pools, raw window, compressor partial state | `ICacheAuxiliaryState` attached to the canonical cache (pattern of `Qwen4ExpAuxiliaryState`) | Reused lifecycle (reset/rollback/archive). DSV4 owns its own banks: CSA/HCA pools, indexer key pool, overlapping/plain compressor state. |
| Lightning indexer scoring + top-k | `QwenSparseAttention` indexer (relu(q.k) head sum, bounded top-k with stable lower-id ties) | Partly reusable: DSV4 adds learned per-token head weights and selects compressed entries, not token blocks. |
| Sparse attention | `QwenSparseAttentionStage` (selected raw tokens, GQA, gated) | Different: DSV4 attends window raw entries plus selected/all compressed entries directly, MQA K==V, sinks, inverse RoPE on output. New stage. |
| MoE pipeline | `Qwen35MoEGraph` routing/dispatch/grouped experts | Reused with extensions: sqrt(softplus) scoring, selection bias, route scale, hash routing, clamped SwiGLU, ungated shared expert. |

## Phases

1. Single participant, existing quant types (Q8_0/BF16/F32 and the unsloth
   K/IQ expert mixes), CPU + CUDA. ROCm fails with
   `DeepSeek V4: ROCm unimplemented`. Prefix cache, MTP/DSpark, TP/PP and
   ExpertOverlay rejected explicitly for `deepseek4` until implemented.
2. MXFP4 tensor type (E8M0 scale + E2M1 LUT; IQ4_NL-shaped kernels).
3. Multi-GPU for the 144 GiB MXFP4 GGUF (2x H100): TP2 or PP2.
4. Vision tower/aligner, image-token routing, OpenAI image content arrays.
5. DSpark drafter (optional).

## Local validation constraints

The release model (>= 81 GiB) cannot run on the development host; tiny random
checkpoints built by `scripts/deepseek4/make_tiny_checkpoint.py` carry every
structural feature and are converted with stock llama.cpp.
