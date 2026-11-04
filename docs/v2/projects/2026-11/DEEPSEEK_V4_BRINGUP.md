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
   `DeepSeek V4: ROCm unimplemented`. Prefix cache, MTP/DSpark and dense
   TP/PP are rejected explicitly for `deepseek4` until implemented;
   ExpertOverlay (`--expert-tier`, GPU + CPU tiers) is supported.
2. MXFP4 tensor type (E8M0 scale + E2M1 LUT; IQ4_NL-shaped kernels).
3. Multi-GPU for the 144 GiB MXFP4 GGUF (2x H100): TP2 or PP2.
4. Vision tower/aligner, image-token routing, OpenAI image content arrays.
5. DSpark drafter (optional).

## Local validation constraints

The release model (>= 81 GiB) cannot run on the development host; tiny random
checkpoints built by `scripts/deepseek4/make_tiny_checkpoint.py` carry every
structural feature and are converted with stock llama.cpp.

## Progress (2026-11-04)

- Phase 1 text graph runs on CPU and CUDA (RTX 3090) and under ExpertOverlay
  (cuda:0 + cpu:0 tiers, token-identical to pure CUDA).
- Tiny-model evidence: CPU stage cosine >= 0.99 against the FP32 reference
  for every stage/layer (457-token prompt and decode); CPU greedy tokens equal
  llama.cpp for 9 tokens; CUDA follows the FP32 reference except at near-ties.
- SwiGLU clamp implemented in every CPU/CUDA MoE path; a binding clamp (1.0)
  gives CPU == CUDA tokens and 7/8 equal to llama.cpp.
- Unsloth UD recipes (IQ2_XS/IQ2_XXS/IQ3_XXS experts, Q5_K/Q6_K shared,
  Q4_K embed/output) reproduced on a 256-wide tiny model and run on CPU/CUDA.
- Planner BOM on header-only sparse copies (unsloth rev `87ce1581`): one
  device needs 115.9 GiB (UD-IQ1_M) / 117.5 GiB (UD-IQ2_XXS) at 32K context
  because CUDA packs IQ2_XS experts at about 1.41x their GGUF size. A single
  H100 (93 GiB) therefore needs a CPU expert tier, or two GPUs.

## MXFP4 (2026-11-04)

- Official routed experts are FP4 E2M1 + E8M0/32, bit-identical to GGUF
  MXFP4 (`scripts/deepseek4/verify_gguf_exactness.py`). unsloth `UD-Q8_K_XL`
  (MXFP4 experts, BF16 elsewhere) reproduces every sampled official tensor
  exactly; ggml-org `MXFP4` stores the FP8 projections as Q8_0 (~0.5% error).
- `TensorType::MXFP4` / codebook 25: the sixteen code bytes keep IQ4_NL's
  nibble order and payload; the E8M0 exponent becomes the exact FP16 scale
  2^(e-128). Only the 16-entry lookup differs from IQ4_NL, which supplies the
  arithmetic policy and certified schedules. Exponents without an exact FP16
  scale are rejected at preparation (real checkpoint exponents span 2^-9..2^-5).
- CPU (AVX2/AVX-512 nibble LUT), CUDA (GEMV, dense prefill, grouped MoE
  prefill/decode incl. IMMA, expert-tier movement) and embeddings support it;
  ROCm rejects codebook 25 and RepackFormat::MXFP4 explicitly.
- GPU bytes: 18 per 32 values (1.06x the GGUF). UD-Q8_K_XL needs 158.8 GiB of
  weights (MXFP4 experts 137.1 -> 145.1 GiB), 163.9 GiB at 32K context.
  A 17-byte layout would need a codebook-specific scale plane in every
  NativeVNNI kernel; not done.

## Official-weight exactness evidence

`scripts/deepseek4/verify_gguf_exactness.py --json` compares one row of each of
twelve tensors (three routed-expert rows across layers 0/21/42, attention
q_a/q_b/kv/wo_b, shared expert, indexer, router, embedding, head) against
`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp@6821d6ad`. The exact tensors, rows
and every compared byte range (URL, offset, length) on both sides are recorded
in:

- `dsv4-gguf-exactness-ud-q8kxl.json`: unsloth `UD-Q8_K_XL@87ce1581`, all 12 samples bit-identical.
- `dsv4-gguf-exactness-ggml-mxfp4.json`: ggml-org `MXFP4@45ddc483`, experts and router
  bit-identical; the FP8 projections, embedding and head are Q8_0 (relative L2 about 5e-3).

This is sampled evidence: it is not exhaustive checkpoint equality. It shows that
the converters keep the official FP4/E8M0 expert bytes and map FP8 (128x128
E8M0 block scales) and BF16 exactly to BF16 for the sampled tensors.

## Cluster run requirements (pending owner decision; no GPU window authorized)

Target: compare Llaminar on UD-Q8_K_XL (official-quality weights) with the
production deployment. The earlier image `ghcr.io/fank/llaminar:dsv4-fe3d60304bbb`
(sha256:013075b4...) predates MXFP4 and is superseded for this plan; it only
loads the lossy UD-IQ* files.

- Image: rebuild the sm_90 Release image from a commit that includes MXFP4
  (398ad5150 or later), CUDA archs `86;90`, ROCm off; push to GHCR (private,
  needs a pull secret) and record the digest.
- GGUF: `unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF@87ce15810931a270f165e5e50439cd446f6b4ba5`,
  `UD-Q8_K_XL/*-0000{1..5}-of-00005.gguf` (150.75 GiB) in one read-only directory;
  point `-m` at shard 1. Text only: content arrays are not supported yet.
- Command: `llaminar2 serve -m <shard1> --expert-tier 'gpu=cuda:0,cuda:1;priority=0'
  -c 32768 --no-prefix-cache --no-mpi-bootstrap --host 0.0.0.0 --port 8080`,
  env `HOME=/tmp OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close`.
  Health: `GET /health`. No `--mtp`, TP/PP flags or CPU expert tier (the pod's
  default seccomp/no SYS_NICE blocks the CPU expert pool's NUMA calls).
- Memory: the planner's single-device BOM is 163.9 GiB at 32K (weights 158.8,
  of which MXFP4 experts 145.1; KV 2.7; workspace 1.4; graphs/activations 0.1).
  The 2-GPU split is NOT certified: dense/shared weights (~13.7 GiB), KV and the
  continuation workspace live on cuda:0, so an even expert split would put about
  90 GiB on cuda:0 against 93.09 GiB usable. A topology-aware per-device BOM
  (planner `--dry-run --explain-placement` on the 2-GPU topology, or a
  synthetic 2xH100 inventory) must show both devices fitting before declaring
  the plan feasible. Llaminar fails closed at startup if they do not fit.
- Oracle: llama.cpp b12079 (`71efc559a`) CUDA build with
  `-DCMAKE_CUDA_ARCHITECTURES=90`, `llama-server -m <shard1> -ngl 999 -c 32768
  -np 1 --jinja` on the same GPUs (llama.cpp keeps the GGUF at file size).
