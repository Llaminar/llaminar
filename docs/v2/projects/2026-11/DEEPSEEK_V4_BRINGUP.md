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

Mathematical oracle: `python/reference/deepseek4.py` (full-recompute FP32,
official QAT rounding on by default, `qat=False` for the pre-QAT formulation),
validated against llama.cpp b12079 with `scripts/deepseek4/`. Without QAT,
prefill stage cosine >= 0.9999 on non-indexer stages; indexer top-k differs
only on near-ties caused by llama.cpp's Q8 activation quantization. llama.cpp
skips the QAT rounding, so it is no longer the token oracle.

## Reuse map (QFN primitives)

| DSV4 concept | QFN component | Relation |
|---|---|---|
| hc x d residual stream | QFN trunk (`residualWidth()`, embedding trunk expansion, branch read/write/head hooks in `QwenGraphBase`) | Reused plumbing. DSV4 adds a Sinkhorn read/write/head mixer selected by a typed residual-mixer policy next to Qwen4Exp's low-rank GR mixer. |
| Compressed KV pools, raw window, compressor partial state | Model-owned arena banks (`AttentionStateOwner::ModelBanks`); the canonical cache is a frontier-only hybrid of `model_state` layers | Installed: see "KV cache architecture". An `ICacheAuxiliaryState` owner is still needed for prefix archive/restore and MTP rollback. |
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

## KV cache architecture (2026-11-04)

DeepSeek-V4 attention state is owned by the model, not the generic KV cache.
`GraphConfig::attention_state_owner = ModelBanks` (single authority
`models/AttentionStateOwner.h`) makes `DeviceGraphOrchestrator` build the
zero-payload hybrid cache: every layer is a typed `model_state`
`HybridLayerKind`, with no FA slab and no GDN bank. The cache still owns the
sequence frontier (device head/count rows, request reset). On GPU the terminal
layer's `AdvanceSequenceFrontier` stage publishes the forward's real rows once
(`advanceRecurrentSequenceState`, the all-GDN slice contract). CPU positions
remain orchestrator-owned. The frontier stage consumes the same row
extent (device live-row count when bound) as every bank kernel, so a padded or
zero-live chunk cannot advance it past the banks. `addKVCacheAppend`, the duplicate `kv_v` GEMM and
`kv_norm_rope_v` are gone. The planner classifies the layers as
`LayerStateKind::ModelOwned`, which gives 0 KV bytes.

Official QAT (`inference/kernel.py`, `scale_fmt=ue8m0`), shared CPU/CUDA code in
`kernels/deepseek4/DeepSeekV4KVFormat.h`:

- Window kv and main compressor output: nope channels pass through FP8 E4M3 in
  blocks of 64, with scale `2^ceil(log2(max(amax,1e-4)*fl(1/448)))`. The rope
  channels are BF16.
- Indexer q (after RoPE) and indexer compressor output: Hadamard
  (Sylvester order, scaled by d^-0.5), then FP4 E2M1 in blocks of 32, with
  scale `2^ceil(log2(max(amax,6*2^-126)*fl(1/6)))`.
- Rounding is RNE throughout, and the scale bit formulas match `fast_round_scale`.

`HeadNormRope` applies the QAT as a post-operation, so `K_PROJ` holds exactly
the stored window values.

Persistent banks are arena roots in INT32 words carrying packed records,
laid out `[layer region][sequence][rows][record]` (one sequence today):

| Bank | Record | Rows per sequence | Per-token cost (release) |
|---|---|---|---|
| raw window ring | 584 B fp8_ds_mla (448 E4M3 + 64 BF16 + 7 UE8M0 + pad) | `window` (128) | fixed 3.2 MB |
| main compressed pool (CSA/HCA) | 584 B | ctx/4, ctx/128 | 21*146 + 20*4.56 B |
| indexer pool (CSA) | 68 B MXFP4 (64 B nibbles + 4 E8M0) | ctx/4 | 21*17 B |
| compressor kv/score state | FP32, as official | 2*ratio | fixed 22 MB |
| indexer score scratch | FP32 | min(rows, 64 MiB / (4*ctx/4)) | <= 64 MiB |

Each chunk reads its own window keys from `K_PROJ` and its own compressor
projections from scratch. Earlier positions come from the ring/state, which
are overwritten only after their readers (`raw_store` after `attention`,
`comp_store` after `comp_pool`). Blocks that straddle chunks are therefore
exact for any chunking, and nothing in the window ring or compressor state
scales with the prefill chunk. The carried banks are zeroed and published once
at setup (`DeepSeekV4Graph::initializeCacheAuxiliaryState`). Readers decode
records on the fly. The lightning-indexer per-32-block FP4 dot products are
exact in FP32, and heads accumulate with fused multiply-add in a fixed order,
so CPU and CUDA scores are bit-identical. The QAT, Hadamard, record and score
arithmetic uses explicit non-FTZ `.rn` operations on CUDA (the TU is compiled
with `--use_fast_math --ftz=true`), so IEEE subnormal results such as FP4 0.5
at the E8M0 floor (2^-127) match the CPU; norm/RoPE/softmax arithmetic is not
part of this byte contract. GPU indexer scoring runs in passes of
`deepSeekV4IndexerScoreRows` rows (`kDeepSeekV4IndexerScoreBudgetBytes` =
64 MiB), with an exact top-k and an unchanged tie rule (score descending, block
id ascending).

Admission charges these banks exactly once. The GPU arena is priced from the
same schema resolution the orchestrator registers, and is cross-checked
fatally against `deepSeekV4ArenaExtensionBytes` (a unit test proves extension
bytes equal the resolved declarations). This fixed two defects:

- The GPU activation estimate returned before the DSV4 extension, so no DSV4
  bank was ever charged on GPU.
- Caches without attention layers no longer reserve full-context FP32
  conversion scratch (`CUDARingKVCache`/`ROCmRingKVCache` with `n_layers_==0`),
  which had cost ~4 GiB at 1M.

Planner dry-run (UD-Q8_K_XL header, `--device cuda:0 --no-prefix-cache`):

| Context | KV old -> new | DSV4 pools | Indexer scratch | Workspace old -> new | Total old -> new |
|---|---|---|---|---|---|
| 32K | 2.7 GB -> 0 | 115 MB (0.107 GiB) | 2 MiB at 64 rows | 1.4 -> 1.3 GB | 163.9 -> 161.3 GB |
| 1M | 86.0 GB -> 0 | 3.68 GB (3.43 GiB) | 64 MiB | 5.3 -> 1.3 GB | 251.1 -> 164.7 GB |

Context-proportional state: 3,514 B/token. vLLM uses 3.4-3.8 KB with an FP8
indexer; the MXFP4 indexer here is lossless because of the FP4 QAT.

Validation (tiny models, 457-token prompt, greedy):

| Model / path | Tokens vs `deepseek4.py` (QAT) |
|---|---|
| dsv4-vision-tiny-ff256-bf16, CPU | 8/8 equal (prefill bucket 512 and 100) |
| dsv4-vision-tiny-ff256-bf16, CUDA | 8/8 equal, == CPU |
| dsv4-tiny-q8kxl-layout-bf16exps (UD-Q8_K_XL analogue), CPU and CUDA | 8/8 equal, CPU == CUDA |
| dsv4-vision-tiny-q8, CUDA | 6/6 equal (bucket 512, 64, 100); ExpertOverlay cuda:0 (8 experts) + cpu:0 identical |
| dsv4-vision-tiny-q8, CPU | first 3 equal; step 3 picks '070', ref rank 4 (top-1 4.271, rank-4 4.035: 0.24 logits; QAT ref top-1/top-2 margin 0.16). Pre-QAT baseline (old binary): CPU diverged from the reference at step 4 and from CUDA at step 4 |

- `serve` with deterministic chat completions A, B, A gives identical A
  outputs on CPU and CUDA (request reset).
- Given Llaminar's own dumped inputs, the indexer pool equals the official
  compressor output exactly, and the selection equals exact top-k.
- Stage cosines against the QAT oracle (bf16 model, CPU prefill) are >= 0.997
  except attention at the CSA layers (0.980 / 0.973). This is near-tie
  amplification, not an arithmetic error: Llaminar's activations differ from
  the FP32 oracle by about 1% RMS, and the FP4 QAT turns that into flipped
  E2M1 codes (about 5% of indexer q values) and different top-k sets on
  near-ties. The official BF16 path has the same sensitivity.

Kernel resources (ptxas, sm_86 and sm_90, `--warn-on-spills` fatal): every
DSV4 kernel has 0 spill bytes on both architectures, and
`CUDARingKVCache.cu` is also spill-free on sm_90. Notable registers at sm_90:

- `sparseAttentionKernel`: 130 registers and 16 KB smem. At 256 threads that
  is one CTA/SM, which is the occupancy limiter to tune.
- `indexerScoreKernel`: 39 registers.
- `compressorPoolKernel` and `headNormRopeKernel`: 37 registers and 3 KB smem.
- `rawStoreKernel`: 30 registers.

`ncu` is not available inside the dev container, so occupancy and throughput
profiling is still open.

Regressions:

- `V2_Unit_DeepSeekV4CPUKernels`
- `V2_Integration_DeepSeekV4CompactKVContracts` (preflight, device-free): QAT
  hand values, codecs, 584/68 B records, chunk-straddling compressor/window
  state, bounded scratch, and the single admission inventory.
- `V2_Integration_CUDA_DeepSeekV4CompactKV` (preflight): byte-exact CUDA vs
  CPU QAT and packing, device chunking invariance, bit-identical indexer
  scores, and bounded passes == one pass.
- `Test__HybridKVCache` model_state map/cache.
- `V2_Integration_DeepSeekV4ArenaAdmission` (preflight, device-free):
  - the GPU arena prices every registered root once (cross-checked against
    the declared inventory), and only the pools and the bounded scratch scale
    with context;
  - the canonical cache charges no KV payload;
  - no K/V conversion scratch exists without attention layers, with a dense
    control.
- The CUDA suite also proves:
  - the frontier-only CUDA cache publishes no conversion scratch and advances
    its device frontier once per forward, including across reset/replay;
  - every indexer score is the same binary32 word on CPU and CUDA;
  - normal inputs with subnormal QAT/decode/score results survive CUDA FTZ
    (negative control: a plain multiply in the Hadamard scale fails it);
  - captured production stages (pool, store, attention, raw store, frontier)
    replayed over padded chunks with full, partial and zero live rows leave
    rows beyond the live count untouched, make zero-live replays a no-op,
    advance the frontier by exactly the live rows, equal one eager
    whole-prompt execution byte for byte, and reproduce after request reset.
- `DeepSeekV4Admission.AdmittedRootsEqualTheActualGraphInventory` compares the
  admitted roots with the actual `DeepSeekV4Graph` schema/resolver inventory,
  root by root, for CPU and GPU allocators and several row envelopes.

Open:

- Prefix archive/restore of the banks: needs an `ICacheAuxiliaryState` owner,
  and the prefix gate keys on `ModelBanks`.
- Per-sequence banks for 8-way batching: the layout already has the sequence
  axis.
- The GPU indexer selection still emits the selected ids from one thread, and
  scores re-decode each key per head. Profile and tune both at 1M before
  production.

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
- Memory: the planner's single-device BOM is 161.3 GiB at 32K (weights 158.8,
  of which MXFP4 experts 145.1; KV 0; DeepSeek-V4 arena incl. 0.1 GiB pools
  0.26; workspace 1.3; graphs 0.1). Before the compact KV it was 163.9 GiB.
  The 2-GPU split is NOT certified: dense/shared weights (~13.7 GiB), KV and the
  continuation workspace live on cuda:0, so an even expert split would put about
  90 GiB on cuda:0 against 93.09 GiB usable. A topology-aware per-device BOM
  (planner `--dry-run --explain-placement` on the 2-GPU topology, or a
  synthetic 2xH100 inventory) must show both devices fitting before declaring
  the plan feasible. Llaminar fails closed at startup if they do not fit.
- Oracle: llama.cpp b12079 (`71efc559a`) CUDA build with
  `-DCMAKE_CUDA_ARCHITECTURES=90`, `llama-server -m <shard1> -ngl 999 -c 32768
  -np 1 --jinja` on the same GPUs (llama.cpp keeps the GGUF at file size).

## Production vLLM parity matrix (target of the 1:1 comparison)

Reference deployment: PCAI P0 evaluation boot of 2026-11-03 (`vision-vram-p0`,
image `vllm-pcai@sha256:a679f68d…`, exact production arguments), on the
official safetensors checkpoint `6821d6ad`. Its measured results are the
baseline; the comparison reruns the same client suite against Llaminar and
needs no new vLLM boot.

vLLM arguments and the corresponding Llaminar state:

| vLLM setting | Value | Llaminar today | Needed for 1:1 |
|---|---|---|---|
| Checkpoint | official FP8 + MXFP4 safetensors | UD-Q8_K_XL GGUF: same values, FP8 tensors widened to BF16 | FP8 E4M3 128x128 block weight type, rebuilt exactly from the BF16 copy (-4.6 GiB) |
| Expert weights on GPU | MXFP4, 17 B/32 | 18 B/32 (FP16 scale) | 17-byte MXFP4 GPU layout (-8 GiB) |
| `--tensor-parallel-size=2` | heads and experts split across both GPUs | DSV4 rejects TP/PP; only ExpertOverlay (dense on cuda:0) | dense TP2 for DSV4 attention/shared (balanced 2-GPU memory) |
| `--max-model-len=1048576` | 1M | done: no generic KV; 3,514 B/token packed pools (3.43 GiB at 1M) | - |
| `--kv-cache-dtype=fp8`, `--block-size=256` | `fp8_ds_mla` 584 B/entry, FP8 indexer 132 B | done: 584 B fp8_ds_mla + 68 B MXFP4 indexer (lossless under QAT), official QAT rounding | paged blocks with the prefix cache |
| `--max-num-seqs=8`, `--max-num-batched-tokens=4096` | 8 concurrent, 4096-token chunks | exactly one request per graph | per-sequence pools, batched DSV4 graph |
| prefix caching (default) + `SimpleCPUOffloadConnector` 192 GiB | GPU + host tier | rejected for deepseek4 | archive/restore DSV4 rings and pools in the prefix cache |
| `--speculative-config` DSpark k=5, per-batch [[1,2,5],[3,4,3],[5,8,2]] | drafter in checkpoint | MTP rejected; DSpark absent | DSpark drafter (GGUF `dspark` tensors) + dynamic depth |
| `--moe-backend=humming` | FP8 MoE activations | int8 (Q8) activations on NativeVNNI | none required for function; numerics differ |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64` | indexer scratch cap | done: row passes bounded by 64 MiB (`kDeepSeekV4IndexerScoreBudgetBytes`) | - |
| `--offload-backend=uva --cpu-offload-params vision aligner embed_tokens` | 0.93 GiB/GPU freed | no vision; embedding on GPU | vision tower; optional host embedding |
| `--compilation-config FULL_AND_PIECEWISE`, async scheduling | full graphs | full captured graphs | equivalent |
| `--reasoning-parser/--tool-call-parser deepseek_v4`, thinking `high` default | official DSV4 encoding | GGUF Jinja template, generic thinking split | DSV4 encoding (`reasoning_effort`), DSML tool calls |
| `--enable-prompt-tokens-details` | cached-token usage | — | report prefix-cache hits in usage |
| `--attention-config indexer_tp_shard_min_rows` (idxshard candidate) | TP split of indexer prefill rows | n/a without TP | row-sharded indexer once dense TP exists |

Measured vLLM baseline (P0): weights + DSpark 78.61 GiB/GPU, CUDA graphs
0.92 GiB, KV 6.76 GiB -> 1,129,474 tokens (1.08x of 1M), nvidia-smi
~94.4 GiB used of 95.8 per GPU. Decode 1 request 206-218 tok/s; 8 concurrent
943 tok/s total (130 per request); cold 68.7K prompt 4.8 s; cold 273K 25.7 s;
warm 273K re-ask 0.81 s; host-tier 252K reload 0.74-0.93 s; quality 51/60.

## Vision (DeepSeek-V4-Flash-Vision-Exp image input)

Authorities: official `inference/{image_processor,vision,model}.py`; prompt
layout of production vLLM (`vllm/models/deepseek_v4/common/mm_preprocess.py`,
in-vocab sentinels `129257 + type`, `3 - start % 4` IMAGE_PAD prefix);
weights from the llama.cpp `deepseek4v` mmproj (`--mmproj`, unsloth
`mmproj-BF16.gguf` / `mmproj-F16.gguf`).

Done:
- Reference `python/reference/deepseek4_vision.py`: ViT + aligner equal the
  official `vision.py` to 1e-6; end to end on exact-size images greedy tokens
  equal llama.cpp mtmd (6/6, 6/6, 4/6).
- Host admission `multimodal/ImageIO` + `DeepSeekV4ImagePreprocess`: Pillow-
  exact decode (PNG), resize and pad, BF16 patches, blocks, expansion
  (`V2_Unit_DeepSeekV4ImagePreprocess`). JPEG decodes with stb (may differ from
  libjpeg-turbo by a few code values).

Runtime (implemented, CPU and CUDA):
- `--mmproj <gguf>` loads the llama.cpp `deepseek4v` projector into a host
  FP32 tower (`DeepSeekV4VisionEncoder`, oneDNN sgemm) after checking from
  GGUF metadata that the text model is a vision checkpoint
  (`exp_probs_b_vl` on every layer, `DeepSeekV4Config::visionCapable`).
  `app/PromptImageEncoderFactory` owns this; production vLLM likewise keeps the
  tower off the GPU (UVA offload). A GPU tower is later performance work.
- Request side (`multimodal/PromptImageEncoder`, DSV4 implementation
  `DeepSeekV4PromptImages`): OpenAI `image_url` parts with `data:` URLs in user
  messages render as `<｜deepseek_image｜>` content parts (vLLM's
  `flatten_content_blocks`), the conversation is encoded, then each
  placeholder expands to its padded sentinel block. The result is an immutable
  `PromptEmbeddingRows`: bank rows (one learned row per sentinel type at its
  type index, then every image's aligner rows in block order), a per-position
  map, and the atomic `[START, END]` spans. Remote URLs, images outside user
  messages, more than 8 images, undecodable data and forged sentinel text are
  400s; a server without `--mmproj` answers 400 instead of dropping images.
  Single-shot chat accepts `--image <file>` (repeatable) for diagnostics.
- Admission: `IOrchestrationRunner::setNextPromptEmbeddingRows` attaches the
  rows to exactly one `prefill()`, which requires position 0 and a single
  rank and stages them on every participant
  (`IInferenceRunner::stagePromptEmbeddingRows`). The participant's graph
  exposes its arena roots (`IGraphBuilder::promptEmbeddingBank`:
  `DSV4_IMAGE_BANK [5 + 8 x max slots, d_model]`, `DSV4_IMAGE_ROW_MAP [ctx]`).
  CPU writes them on the host; GPU uploads them in the canonical
  request-input admission transaction (same stream, before its readiness
  event), clearing the previous request's map prefix. Text requests stage
  null. Both roots are priced by the DSV4 arena BOM; a text checkpoint
  declares one element each.
- Graph: `DeepSeekV4PrimitiveStage::ImageMerge` overwrites sentinel rows of
  the embedding before the hyper-connection expansion; `SparseAttention`
  takes the chunk token ids and widens the raw window of span rows to the
  whole span in both directions (`imageSpanExtent` = `get_image_visible`);
  the MoE router selects image-sentinel rows by `s + exp_probs_b_vl` on every
  layer, hash layers included (CPU router, CUDA legacy and owned-row router).
- Prefill chunking never splits a span: `PrefillChunkSchedulerPolicy::
  atomic_spans` ends a chunk before a span that would straddle it; a span
  longer than the largest admitted bucket fails planning (block <= 381 rows,
  so a 384/512-row bucket is required on GPU).

Vision validation (tiny `dsv4-vision-tiny-ff256-bf16` + its f32 mmproj,
greedy, prompt `[image] Describe the picture.`, thinking high):
- Stage dumps vs `deepseek4.py` with image rows (CPU): merged embedding equal
  to 2.6e-6; per-row attention and layer-output errors of image rows are at
  the text rows' level (Q8_0 attention weights with quantized activations
  give ~1% per-projection differences vs the float reference on both); router
  ids of image rows at the hash layers 0-2 match the vl-bias selection for
  98.5/93/93% of rows (remaining flips are score near-ties), text rows 100%.
- Tokens: 616x266 image: CUDA 4/6 equal to the reference, CPU 2/6; 588x588:
  CPU == CUDA byte-identical, diverging from the reference at a 0.009-logit
  near-tie at step 1, then following the reference's next choice.
- HTTP server (CPU and CUDA): image, text, image requests in sequence give
  the oneshot tokens each time (the map reset between requests holds).
- Note: short BOS-prefixed chat prompts on these tiny models diverge from the
  float reference after the first token for text-only requests too (old
  image `dsv4-2190fcad1895` identical); the 457-token prompt below stays
  equal. This is the Q8 activation precision of the tiny random model, not a
  vision defect.

Not yet: GPU vision tower, multi-rank image requests, prefix caching of image
prompts (prefix cache is rejected for DSV4), image-content-aware prefix keys.
